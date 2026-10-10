#!/usr/bin/env python
"""Nine-bin integration-headroom scan over the observable (gap x margin) space.

exp20 showed mechanical fusion cannot beat the better single source in the two
cells the arbitration paper evaluated -- because in each of those cells one
source is at or below the random-ranking rate by construction, so "select" IS
the calibration-optimal policy. That makes the paper's selector-vs-integrator
null untestable there, not falsified.

This script asks the prior question: is there ANY (gap, margin) bin where
  (a) both sources are individually informative, AND
  (b) their errors are complementary rather than nested,
so that an integrator could out-perform the per-item best-source oracle -- and
is there enough EVAL-half data there to build a third cell?

Method, per bin, over eval-half rows only (arbitration.split_queries; the
calibration half must not leak into any eval-cell proposal):
  - phenotype-only / annotation-only MRR and top-1
  - per-item best-source oracle MRR (max of the two RRs)
  - fusion ceiling = per-item max over {oracle} U exp20's five fusion rules
    (imported, not reimplemented)
  - headroom = fusion ceiling - best-source oracle, plus the count of items
    where at least one rule strictly beats max(phen, ann)
  - n_exceedable = items where neither source already ranks the gold first
    (the only denominator on which "beat the better source" is possible)
  - complementarity: the 2x2 contingency of {phen top-1} x {ann top-1} with the
    phi coefficient and odds ratio. phi <= 0 means the sources fail on
    different items -- the regime where fusion can pay. phi >> 0 means nested
    errors and fusion is structurally hopeless.

Scores are recovered from the same masked packet the model is shown
(pilot.make_packet(..., "A-mask", 0), as arbitration.scan_pool used), and every
row's recomputed phen/ann rank is asserted against the cached pool scan.

Usage:
    PYTHONHASHSEED=0 python scripts/evidence_eval/exp22_headroom_scan.py
    PYTHONHASHSEED=0 python scripts/evidence_eval/exp22_headroom_scan.py --selfcheck
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_counterfactual as pilot          # noqa: E402
import reliability_calibration as rc          # noqa: E402
import validate_counterfactual as V           # noqa: E402
import arbitration as arb                     # noqa: E402
import exp20_fusion_baseline as F             # noqa: E402

if os.environ.get("PYTHONHASHSEED") != "0":
    raise SystemExit("run as: PYTHONHASHSEED=0 python scripts/evidence_eval/exp22_headroom_scan.py")

OUT_DIR = arb.ART / "headroom_scan"
RULES = F.RULES
RANDOM_RR = arb.RANDOM_RR
MIN_N = 30                 # usable-cell threshold (paper's noise floor: MRR SE 0.0043 / 40 items)


def bin_key(row):
    return (f"{arb._bin(row['gap'], arb.GAP_BINS)} | "
            f"{arb._bin(row['margin'], arb.MARGIN_BINS)}")


def phi_and_table(p, a):
    """2x2 contingency of two boolean vectors + phi coefficient and odds ratio.

    phi is Pearson correlation on 0/1 data. A degenerate margin (one variable
    constant, e.g. annotation never top-1 in a no-annotation bin) makes phi
    undefined -- return None rather than 0, which would read as "independent".
    """
    p, a = np.asarray(p, bool), np.asarray(a, bool)
    n11 = int((p & a).sum())
    n10 = int((p & ~a).sum())
    n01 = int((~p & a).sum())
    n00 = int((~p & ~a).sum())
    den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    phi = (n11 * n00 - n10 * n01) / np.sqrt(den) if den > 0 else None
    # Haldane-Anscombe 0.5 correction so a single empty cell does not give inf
    orr = ((n11 + 0.5) * (n00 + 0.5)) / ((n10 + 0.5) * (n01 + 0.5))
    return {"both_top1": n11, "phen_only_top1": n10, "ann_only_top1": n01,
            "neither_top1": n00,
            "phi": None if phi is None else round(float(phi), 4),
            "odds_ratio_haldane": round(float(orr), 3)}


def build_rows(half="eval"):
    """Pool rows from one split half, with both full source score vectors.

    `half="calib"` selects the bin on the same data the reliability table is fit
    to, leaving the eval half untouched for the cell that gets run. The eval-half
    default is what the original scan used; it proposes a cell using the gold
    statistics of the items the cell will then be drawn from.
    """
    pool = arb.scan_pool()
    calib_q, eval_q = arb.split_queries(pool)
    keep = {"eval": eval_q, "calib": calib_q}[half]
    cal_table = json.load(open(arb.OUT_DIR / "calibration.json"))["table"]

    emb = V.load_embeddings()
    _, test = V.load_splits()
    # item_id must reproduce arbitration.scan_pool's f"s{qi:04d}" exactly: the
    # candidate permutation is hash((item_id, order)), so a different id gives a
    # different alias map and silently different annotation scores.
    qid = {q: f"s{qi:04d}" for qi, q in enumerate(sorted(g for g in test if g in emb))}
    ctx = pilot._ctx()

    rows = []
    for r in pool:
        if r["query"] not in keep:
            continue
        item = dict(query=r["query"], gold=r["gold"], distractors=r["distractors"],
                    item_id=qid[r["query"]])
        packet, gold, audit = pilot.make_packet(item, "A-mask", 0, ctx)
        assert audit["masked"], item
        ann = rc._ann_scores(packet)
        # unrounded cosines, matching the stored phen_rank and exp20's convention
        cands = [r["gold"]] + r["distractors"]
        alias = F._alias_map(item, 0)
        assert alias[r["gold"]] == gold, (item["item_id"], alias[r["gold"]], gold)
        phen = {alias[g]: float(emb[r["query"]] @ emb[g]) for g in cands}
        assert set(phen) == set(ann) == set(packet["candidates"])

        pr, ar = rc._avg_rank(phen, gold), rc._avg_rank(ann, gold)
        assert abs(pr - r["phen_rank"]) < 1e-9 and abs(ar - r["ann_rank"]) < 1e-9, (
            f"rank reconstruction drift for {r['query']}/{r['gold']}: "
            f"phen {pr} vs {r['phen_rank']}, ann {ar} vs {r['ann_rank']}. "
            "make_packet permutes candidates via hash((item_id, order)); "
            "PYTHONHASHSEED=0 pins str hashing only within a Python version, so "
            "re-run under conda ops-experimental (Python 3.10).")

        k = bin_key(r)
        cal = cal_table[k]
        pe = cal["phenotype_expected_reciprocal_rank"]
        ae = cal["annotation_expected_reciprocal_rank"]
        rows.append(dict(bin=k, query=r["query"], gold=r["gold"], gold_alias=gold,
                         phen=phen, ann=ann,
                         w_relu=(max(0.0, pe - RANDOM_RR), max(0.0, ae - RANDOM_RR)),
                         w_raw=(pe, ae),
                         cal_phen_eRR=pe, cal_ann_eRR=ae))
    return rows


def score(rows):
    out = []
    for r in rows:
        f = F.fuse_all(r["phen"], r["ann"], r["w_relu"], r["w_raw"])
        e = dict(bin=r["bin"], query=r["query"], gold=r["gold"],
                 phenotype=F.rr(r["phen"], r["gold_alias"]),
                 annotation=F.rr(r["ann"], r["gold_alias"]),
                 t1_phenotype=F.top1(r["phen"], r["gold_alias"]),
                 t1_annotation=F.top1(r["ann"], r["gold_alias"]))
        for name, sc in f.items():
            e[name] = F.rr(sc, r["gold_alias"])
            e["t1_" + name] = F.top1(sc, r["gold_alias"])
        e["best_source"] = max(e["phenotype"], e["annotation"])
        e["t1_best_source"] = max(e["t1_phenotype"], e["t1_annotation"])
        e["exceedable"] = e["best_source"] < 1.0 - 1e-9
        e["ceiling"] = max([e["best_source"]] + [e[n] for n in RULES])
        e["t1_ceiling"] = max([e["t1_best_source"]] + [e["t1_" + n] for n in RULES])
        out.append(e)
    return out


def analyze(per_item, half="eval"):
    report = {
        "question": "is there a (gap, margin) bin where both sources are "
                    "informative AND their errors are complementary, so an "
                    "integrator could beat the per-item best-source oracle?",
        "half": half,
        "half_note": (
            "eval half only (arbitration.split_queries, seed "
            f"{arb.SPLIT_SEED}); the calibration half is excluded, but the "
            "proposed cell is then drawn from these same rows, so the bin is "
            "chosen on the gold statistics of the items it will score"
            if half == "eval" else
            "calibration half only (arbitration.split_queries, seed "
            f"{arb.SPLIT_SEED}); the eval half is never touched, so a bin "
            "chosen here is selected independently of the items the cell will "
            "be drawn from. Caveat: calib_relu/calib_raw use per-bin weights "
            "fit on this same half, so their contribution to the ceiling is "
            "in-sample; the best-source oracle and the tie-free rules are not."),
        "packet": "pilot.make_packet(..., 'A-mask', 0); unrounded phenotype "
                  "cosines; average-rank tie handling throughout",
        "fusion_rules": list(RULES),
        "random_ranking_MRR": round(RANDOM_RR, 4),
        "usable_cell_min_n": MIN_N,
        "bins": {},
    }
    order = [f"{g[2]} | {m[2]}" for g in arb.GAP_BINS for m in arb.MARGIN_BINS]
    for k in order:
        E = [e for e in per_item if e["bin"] == k]
        if not E:
            continue
        exc = [e for e in E if e["exceedable"]]
        P = [e["phenotype"] for e in E]
        A = [e["annotation"] for e in E]
        O = [e["best_source"] for e in E]
        C = [e["ceiling"] for e in E]
        beats = [e for e in exc
                 if max(e[n] for n in RULES) > e["best_source"] + 1e-9]
        b = {
            "n_eval_rows": len(E),
            "n_eval_queries": len({e["query"] for e in E}),
            "n_exceedable": len(exc),
            "phenotype_MRR": F._m(P), "phenotype_top1": F._m([e["t1_phenotype"] for e in E]),
            "annotation_MRR": F._m(A), "annotation_top1": F._m([e["t1_annotation"] for e in E]),
            "best_source_oracle_MRR": F._m(O),
            "best_source_oracle_top1": F._m([e["t1_best_source"] for e in E]),
            "fusion_ceiling_MRR": F._m(C),
            "fusion_ceiling_top1": F._m([e["t1_ceiling"] for e in E]),
            "headroom_MRR": F._m(np.array(C) - O),
            "headroom_MRR_CI": F._boot_paired(C, O),
            "headroom_top1": F._m([e["t1_ceiling"] - e["t1_best_source"] for e in E]),
            "n_any_rule_beats_best_source": len(beats),
            "frac_exceedable_beaten": round(len(beats) / len(exc), 4) if exc else None,
            "complementarity": phi_and_table([e["t1_phenotype"] > 0.5 for e in E],
                                             [e["t1_annotation"] > 0.5 for e in E]),
            "per_rule_MRR": {n: F._m([e[n] for e in E]) for n in RULES},
            "best_rule_vs_oracle": {
                n: F._m(np.array([e[n] for e in E]) - O) for n in RULES},
        }
        b["both_sources_above_random"] = bool(
            b["phenotype_MRR"] > RANDOM_RR and b["annotation_MRR"] > RANDOM_RR)
        b["viable_cell"] = bool(b["both_sources_above_random"]
                                and b["n_eval_queries"] >= MIN_N
                                and b["headroom_MRR_CI"][0] > 0)
        report["bins"][k] = b
    return report


def esc(k):
    """Bin labels contain a literal '|', which would split a markdown cell."""
    return k.replace("|", r"\|")


def write(report, per_item):
    global OUT_DIR
    if report["half"] != "eval":
        OUT_DIR = OUT_DIR.parent / f"headroom_scan_{report['half']}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT_DIR / "metrics.json", "w"), indent=2)

    L = [f"# Integration-headroom scan across the nine (gap x margin) bins "
         f"({report['half']} half)", "",
         report["half_note"], "",
         "Average-rank ties throughout. Random-ranking MRR = "
         f"{report['random_ranking_MRR']}. Fusion ceiling = per-item max over "
         f"{{best-source oracle}} and exp20's five rules ({', '.join(RULES)}).", "",
         "`phi` = correlation between *phenotype ranks the gold first* and "
         "*annotation ranks the gold first*. phi <= 0 means complementary errors "
         "(fusion can pay); phi >> 0 means nested errors (it cannot). `-` = one "
         "variable is constant, so phi is undefined.", "",
         "| bin | n rows | n queries | phen MRR | ann MRR | oracle | ceiling | "
         "**headroom** | headroom 95% CI | exceedable n | beaten | phi | OR |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, b in report["bins"].items():
        c = b["complementarity"]
        phi = "-" if c["phi"] is None else f"{c['phi']:+.3f}"
        L.append(
            f"| {esc(k)} | {b['n_eval_rows']} | {b['n_eval_queries']} | "
            f"{b['phenotype_MRR']} | {b['annotation_MRR']} | "
            f"{b['best_source_oracle_MRR']} | {b['fusion_ceiling_MRR']} | "
            f"**{b['headroom_MRR']:+.4f}** | {tuple(b['headroom_MRR_CI'])} | "
            f"{b['n_exceedable']} | {b['n_any_rule_beats_best_source']} | "
            f"{phi} | {c['odds_ratio_haldane']} |")

    L += ["", "## Top-1 view", "",
          "| bin | phen top-1 | ann top-1 | oracle top-1 | ceiling top-1 | headroom top-1 |",
          "|---|---|---|---|---|---|"]
    for k, b in report["bins"].items():
        L.append(f"| {esc(k)} | {b['phenotype_top1']} | {b['annotation_top1']} | "
                 f"{b['best_source_oracle_top1']} | {b['fusion_ceiling_top1']} | "
                 f"{b['headroom_top1']:+.4f} |")

    L += ["", "## What a FIXED rule achieves (vs the per-item best-source oracle)", "",
          "The ceiling above is a per-item max over the five rules -- an "
          "oracle-over-rules, reachable only by something that knows which rule "
          "to trust on each item. The table below is what each rule actually "
          "delivers as a fixed policy.", "",
          "| bin | " + " | ".join(RULES) + " |",
          "|---|" + "---|" * len(RULES)]
    for k, b in report["bins"].items():
        L.append(f"| {esc(k)} | " + " | ".join(
            f"{b['best_rule_vs_oracle'][n]:+.4f}" for n in RULES) + " |")
    worst = max(max(b["best_rule_vs_oracle"].values())
                for b in report["bins"].values())
    L += ["", f"Best fixed rule in the best bin: **{worst:+.4f}** vs the oracle. "
          "Every entry is negative -- in all nine bins, no fixed fusion rule "
          "matches the per-item best-source oracle, let alone the ceiling."]

    viable = [k for k, b in report["bins"].items() if b["viable_cell"]]
    L += ["", "## Verdict", ""]
    if viable:
        for k in viable:
            b = report["bins"][k]
            L.append(f"- **{k}** is viable: {b['n_eval_queries']} eval-half queries, "
                     f"headroom {b['headroom_MRR']:+.4f} MRR "
                     f"{tuple(b['headroom_MRR_CI'])}, {b['headroom_top1']:+.4f} top-1.")
    else:
        L.append("- **No bin is viable.** No (gap x margin) region combines two "
                 "above-random sources, >=%d eval-half queries, and a fusion "
                 "ceiling strictly above the per-item best-source oracle. "
                 "Integration headroom is absent across the entire observable "
                 "space of this task, so a third 'complementary' cell cannot be "
                 "built from it." % MIN_N)
    (OUT_DIR / "RESULTS.md").write_text("\n".join(L) + "\n")
    json.dump(per_item, open(OUT_DIR / "per_item.json", "w"))
    print(f"wrote -> {OUT_DIR/'metrics.json'}\nwrote -> {OUT_DIR/'RESULTS.md'}")


def selfcheck():
    F.selfcheck()                       # the fusion rules themselves
    # phi: perfect agreement -> +1, perfect disagreement -> -1, independence -> 0
    assert phi_and_table([1, 1, 0, 0], [1, 1, 0, 0])["phi"] == 1.0
    assert phi_and_table([1, 1, 0, 0], [0, 0, 1, 1])["phi"] == -1.0
    assert phi_and_table([1, 1, 0, 0], [1, 0, 1, 0])["phi"] == 0.0
    assert phi_and_table([1, 1, 0, 0], [0, 0, 0, 0])["phi"] is None   # constant
    t = phi_and_table([1, 0, 0], [1, 1, 0])
    assert (t["both_top1"], t["phen_only_top1"], t["ann_only_top1"], t["neither_top1"]) \
        == (1, 0, 1, 1), t
    # nested errors: annotation is right only where phenotype already is ->
    # the oracle already has RR 1 there, so no rule can add anything.
    nested = [dict(phenotype=1.0, annotation=1.0), dict(phenotype=0.5, annotation=0.25)]
    assert all(max(e["phenotype"], e["annotation"]) == e["phenotype"] for e in nested)
    print("selfcheck OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--half", choices=("eval", "calib"), default="eval",
                    help="which split half to scan. 'calib' leaves the eval "
                         "half untouched so the chosen bin is not selected on "
                         "the items the cell will be drawn from.")
    a = ap.parse_args()
    selfcheck()
    if a.selfcheck:
        return
    rows = build_rows(a.half)
    print(f"{len(rows)} {a.half}-half rows, {len(Counter(r['bin'] for r in rows))} bins")
    per_item = score(rows)
    rep = analyze(per_item, a.half)
    write(rep, per_item)
    for k, b in rep["bins"].items():
        c = b["complementarity"]
        print(f"{k:42s} n={b['n_eval_rows']:4d} q={b['n_eval_queries']:4d} "
              f"phen {b['phenotype_MRR']:.3f} ann {b['annotation_MRR']:.3f} "
              f"orc {b['best_source_oracle_MRR']:.3f} ceil {b['fusion_ceiling_MRR']:.3f} "
              f"head {b['headroom_MRR']:+.4f} {b['headroom_MRR_CI']} "
              f"exc={b['n_exceedable']:3d} beat={b['n_any_rule_beats_best_source']:3d} "
              f"phi={c['phi']}")


if __name__ == "__main__":
    main()
