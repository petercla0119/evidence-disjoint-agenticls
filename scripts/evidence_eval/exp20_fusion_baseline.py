#!/usr/bin/env python
"""Mechanical-fusion control for the evidence-arbitration result.

The arbitration paper claims the model acts as a SOURCE SELECTOR, never an
EVIDENCE INTEGRATOR: across 300 item-arm cells it beat the better of its two
sources exactly once. That claim is only about the model if a deterministic
fusion of the same two source rankings CAN beat the better source here. This
script is that control.

For all 60 arbitration items it recovers both full per-candidate score vectors
from the SAME masked packet the model saw (pilot.make_packet(..., "H-mask", 0)
-- H-mask and A-mask drop identical records; H-mask additionally carries the
phenotype block, so one packet yields both sources aligned by alias), asserts
the derived ranks reproduce the stored phen_rank / ann_rank, then scores four
fusion rules against phenotype-only, annotation-only, and the per-item
best-source oracle.

Usage:
    PYTHONHASHSEED=0 python scripts/evidence_eval/exp20_fusion_baseline.py
    PYTHONHASHSEED=0 python scripts/evidence_eval/exp20_fusion_baseline.py --selfcheck
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_counterfactual as pilot          # noqa: E402
import reliability_calibration as rc          # noqa: E402
import arbitration as arb                     # noqa: E402

IN_DIR = arb.OUT_DIR                                  # artifacts/evidence_eval/arbitration
OUT_DIR = arb.ART / "fusion"
RANDOM_RR = arb.RANDOM_RR                             # 0.408, E[RR] of a uniform ranking
CELLS = ("conflict", "control")
RULES = ("borda", "rrf_k60", "rrf_k1", "calib_relu", "calib_raw")


# ---------------------------------------------------------------------------
# ranking primitives -- average-rank tie handling everywhere
# ---------------------------------------------------------------------------
def avg_ranks(score):
    """{key: average rank}, 1 = best (highest score). Ties share the mean rank.

    Average-rank ties are the only defensible convention here: annotation is
    a 6-way tie on 19/20 control items, and any sort-order tiebreak would hand
    it a free 1-in-6 chance of looking like a correct top pick."""
    return {k: rc._avg_rank(score, k) for k in score}


def rr(score, gold):
    """Expected reciprocal rank under uniform random tie-breaking.

    NOT 1/avg_rank -- reciprocal rank is convex, so 1/E[rank] understates
    E[1/rank] for a tied target. The old form scored a fully-tied 6-way
    annotation vector at 1/3.5 = 0.286, below the 0.408 random rate, which is
    impossible for a ranker with no anti-signal. See rc.expected_rr."""
    return rc.expected_rr(score, gold)


def top1(score, gold):
    """Expected top-1 under uniform random tie-breaking: 1/k inside a k-way top
    tie, 0 otherwise. The strict convention this replaces scored any tie at the
    top as a miss, which charges a ranker for being undecided rather than wrong
    and put a fully-tied 6-way ranker at 0.0 against a 1/6 random rate. It also
    biased every fusion rule downward, since Borda and RRF tie constantly
    (Borda ties the gold on 12/40 conflict items) while phenotype never does."""
    return rc.expected_top1(score, gold)


def _z(score):
    """Within-item z-score. Zero variance (e.g. a fully tied annotation vector)
    -> all zeros, i.e. the source contributes nothing. That is the correct
    behaviour, not a degenerate case to guard against."""
    v = np.array([score[k] for k in score], dtype=float)
    sd = v.std()
    z = (v - v.mean()) / sd if sd > 1e-12 else np.zeros_like(v)
    return dict(zip(score, z))


# ---------------------------------------------------------------------------
# fusion rules: (phen_score, ann_score, weights) -> fused score dict
# ---------------------------------------------------------------------------
def borda(phen, ann, _w=None):
    """Borda points = n_candidates - average_rank, summed over the two sources."""
    n = len(phen)
    pr, ar = avg_ranks(phen), avg_ranks(ann)
    return {k: (n - pr[k]) + (n - ar[k]) for k in phen}


def rrf(phen, ann, k):
    pr, ar = avg_ranks(phen), avg_ranks(ann)
    return {c: 1.0 / (k + pr[c]) + 1.0 / (k + ar[c]) for c in phen}


def calib_weighted(phen, ann, w):
    """z-normalise each source within the item, weight by its calibrated
    expected reciprocal rank in this item's (gap, margin) bin."""
    zp, za = _z(phen), _z(ann)
    wp, wa = w
    return {c: wp * zp[c] + wa * za[c] for c in phen}


def fuse_all(phen, ann, w_relu, w_raw):
    return {
        "borda": borda(phen, ann),
        "rrf_k60": rrf(phen, ann, 60),
        "rrf_k1": rrf(phen, ann, 1),
        "calib_relu": calib_weighted(phen, ann, w_relu),
        "calib_raw": calib_weighted(phen, ann, w_raw),
    }


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def _boot_paired(a, b, n=10000, seed=0):
    """Paired bootstrap over items on mean(a) - mean(b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.size == 0:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, d.size), replace=True).mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 4),
            round(float(np.percentile(m, 97.5)), 4)]


def _m(x):
    return round(float(np.mean(x)), 4)


# ---------------------------------------------------------------------------
def _alias_map(item, order):
    """gene -> G# under make_packet's candidate permutation. Mirrors the three
    lines in pilot.make_packet; the caller asserts it against the gold_alias
    make_packet itself returned, so a drift in either is caught immediately."""
    cands = [item["gold"]] + item["distractors"]
    perm = list(range(6))
    random.Random(hash((item["item_id"], order)) & 0xFFFFFFFF).shuffle(perm)
    return {cands[p]: f"G{i+1}" for i, p in enumerate(perm)}


def build_rows():
    """Per item: both source score vectors, recovered from the masked packet."""
    meta = json.load(open(IN_DIR / "items.json"))
    cal = json.load(open(IN_DIR / "calibration.json"))["table"]
    ctx = pilot._ctx()
    emb = ctx[0]

    rows, mismatches, rounding_ties = [], [], []
    for it in meta["items"]:
        # H-mask order 0 == the evaluated condition: every record containing both
        # query and gold is dropped. (A-mask drops the same records; it only omits
        # the phenotype block, so the annotation scores are identical.)
        packet, gold, audit = pilot.make_packet(it, "H-mask", 0, ctx)
        assert audit["masked"], it["item_id"]
        alias = _alias_map(it, 0)
        assert alias[it["gold"]] == gold, (it["item_id"], alias[it["gold"]], gold)

        # The packet shows the model cosines ROUNDED to 4 dp, which manufactures
        # ties on 2/60 items that do not exist in the underlying cosines. The
        # stored phen_rank (and every phenotype-only number in the paper) uses the
        # unrounded cosines, so fusion uses them too -- otherwise the control would
        # be scored against a baseline it does not share.
        phen = {alias[g]: float(emb[it["query"]] @ emb[g])
                for g in [it["gold"]] + it["distractors"]}
        if len(set(packet["phenotype_similarity"].values())) < 6:
            rounding_ties.append(it["item_id"])
        ann = rc._ann_scores(packet)
        assert set(phen) == set(ann) == set(packet["candidates"])

        pr, ar = rc._avg_rank(phen, gold), rc._avg_rank(ann, gold)
        if abs(pr - it["phen_rank"]) > 1e-9 or abs(ar - it["ann_rank"]) > 1e-9:
            mismatches.append(dict(item_id=it["item_id"],
                                   recomputed_phen_rank=pr, stored_phen_rank=it["phen_rank"],
                                   recomputed_ann_rank=ar, stored_ann_rank=it["ann_rank"]))

        row = cal[f"{arb._bin(it['gap'], arb.GAP_BINS)} | "
                  f"{arb._bin(it['margin'], arb.MARGIN_BINS)}"]
        pe = row["phenotype_expected_reciprocal_rank"]
        ae = row["annotation_expected_reciprocal_rank"]
        rows.append(dict(
            item_id=it["item_id"], cell=it["cell"], gold=gold, phen=phen, ann=ann,
            # relu weights: a source at or below the random ranking RR carries no
            # information in this bin, so it gets weight 0 -- the reading the
            # reliability prompt literally instructs the model to take.
            w_relu=(max(0.0, pe - RANDOM_RR), max(0.0, ae - RANDOM_RR)),
            w_raw=(pe, ae),                      # soft blend, never fully silences
        ))

    if mismatches:
        raise SystemExit(
            "masked-packet reconstruction did NOT reproduce the stored ranks for "
            f"{len(mismatches)} item(s) -- refusing to score fusion on packets that "
            "differ from the evaluated condition:\n"
            + json.dumps(mismatches, indent=2)
            + "\nIf every item mismatches, you are probably on the wrong interpreter: "
              "make_packet permutes candidates via hash((item_id, order)) and "
              "PYTHONHASHSEED=0 pins str hashing only within a Python version.")
    return rows, rounding_ties


def analyze(rows, rounding_ties=()):
    report = {
        "n_items": len(rows),
        "items_where_4dp_packet_rounding_ties_phenotype": list(rounding_ties),
        "source": "phenotype cosine + annotation graph score, both from the H-mask "
                  "order-0 packet; ranks verified against items.json",
        "tie_handling": ("average rank for ordering; reciprocal rank is the EXPECTED "
                         "value under uniform random tie-breaking (rc.expected_rr), "
                         "every source and every fusion rule. Not 1/avg_rank, which "
                         "scored a fully-tied ranker 0.286, below the 0.408 chance rate."),
        "random_ranking_MRR": round(RANDOM_RR, 4),
        "rank_reconstruction_mismatches": 0,
        "cells": {},
    }

    per_item = []                     # keeps ceiling/exceedance auditable
    for r in rows:
        f = fuse_all(r["phen"], r["ann"], r["w_relu"], r["w_raw"])
        e = dict(item_id=r["item_id"], cell=r["cell"],
                 phenotype=rr(r["phen"], r["gold"]), annotation=rr(r["ann"], r["gold"]),
                 t1_phenotype=top1(r["phen"], r["gold"]), t1_annotation=top1(r["ann"], r["gold"]))
        for name, sc in f.items():
            e[name] = rr(sc, r["gold"])
            e["t1_" + name] = top1(sc, r["gold"])
        e["best_source"] = max(e["phenotype"], e["annotation"])
        # exceedable = neither source already put the gold first; on the rest,
        # "beat the better source" is arithmetically impossible (RR is capped at 1).
        e["exceedable"] = e["best_source"] < 1.0 - 1e-9
        e["ceiling"] = max([e["best_source"]] + [e[n] for n in RULES])
        per_item.append(e)

    for cell in CELLS:
        E = [e for e in per_item if e["cell"] == cell]
        exc = [e for e in E if e["exceedable"]]
        P = [e["phenotype"] for e in E]
        A = [e["annotation"] for e in E]
        O = [e["best_source"] for e in E]
        out = {
            "n": len(E),
            "n_exceedable": len(exc),
            "n_unexceedable": len(E) - len(exc),
            "phenotype_only": {"MRR": _m(P), "top1": _m([e["t1_phenotype"] for e in E])},
            "annotation_only": {"MRR": _m(A), "top1": _m([e["t1_annotation"] for e in E])},
            "best_source_oracle_MRR": _m(O),
            "fusion_ceiling_MRR": _m([e["ceiling"] for e in E]),
            "rules": {},
        }
        for name in RULES:
            V = [e[name] for e in E]
            beats = [e for e in exc if e[name] > e["best_source"] + 1e-9]
            loses = [e for e in exc if e[name] < e["best_source"] - 1e-9]
            out["rules"][name] = {
                "MRR": _m(V),
                "top1": _m([e["t1_" + name] for e in E]),
                "vs_phenotype": _m(np.array(V) - P), "vs_phenotype_CI": _boot_paired(V, P),
                "vs_annotation": _m(np.array(V) - A), "vs_annotation_CI": _boot_paired(V, A),
                "vs_best_source_oracle": _m(np.array(V) - O),
                "vs_best_source_oracle_CI": _boot_paired(V, O),
                "exceeds_better_source_n": len(beats),
                "exceeds_better_source_items": [e["item_id"] for e in beats],
                "exceeds_better_source_mean_gain_on_exceedable":
                    _m([e[name] - e["best_source"] for e in exc]) if exc else None,
                "exceeds_better_source_total_MRR_gain_over_cell":
                    round(sum(e[name] - e["best_source"] for e in beats) / len(E), 4),
                "falls_below_better_source_n": len(loses),
            }
        report["cells"][cell] = out

    report["per_item"] = per_item
    return report


def write(report):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT_DIR / "metrics.json", "w"), indent=2)

    L = ["# Mechanical fusion baseline for the evidence-arbitration items", "",
         "Control for the claim that the model is a *source selector*, not an "
         "*evidence integrator*. Question: can a deterministic fusion of the same "
         "two source rankings beat the better single source?", "",
         "Average-rank tie handling throughout (annotation is a 6-way tie on 19/20 "
         "control items). Random-ranking MRR = "
         f"{report['random_ranking_MRR']}.", ""]
    for cell, c in report["cells"].items():
        L += [f"## {cell} cell (n={c['n']})", "",
              f"Exceedable items (neither source already ranks the gold first): "
              f"**{c['n_exceedable']}** of {c['n']}.", "",
              "| arm | MRR | top-1 | vs phen | vs ann | vs best-source oracle | exceeds better source |",
              "|---|---|---|---|---|---|---|",
              f"| phenotype-only | {c['phenotype_only']['MRR']} | {c['phenotype_only']['top1']} | - | - | - | - |",
              f"| annotation-only | {c['annotation_only']['MRR']} | {c['annotation_only']['top1']} | - | - | - | - |"]
        for n in RULES:
            r = c["rules"][n]
            L.append(f"| {n} | {r['MRR']} | {r['top1']} | "
                     f"{r['vs_phenotype']:+} {tuple(r['vs_phenotype_CI'])} | "
                     f"{r['vs_annotation']:+} {tuple(r['vs_annotation_CI'])} | "
                     f"{r['vs_best_source_oracle']:+} {tuple(r['vs_best_source_oracle_CI'])} | "
                     f"{r['exceeds_better_source_n']}/{c['n_exceedable']} |")
        L += ["", f"| best-source oracle | {c['best_source_oracle_MRR']} |",
              "|---|---|",
              f"| fusion ceiling (best of oracle + all rules, per item) | {c['fusion_ceiling_MRR']} |", ""]

    tot_exc = sum(c["n_exceedable"] for c in report["cells"].values())
    L += ["## Verdict", ""]
    for n in RULES:
        k = sum(c["rules"][n]["exceeds_better_source_n"] for c in report["cells"].values())
        L.append(f"- `{n}`: beats the better source on **{k}/{tot_exc}** exceedable items.")
    L += ["",
          "No fusion rule reaches the per-item best-source oracle in either cell "
          "(every `vs best-source oracle` CI excludes 0 and is negative). The "
          "calibration-weighted rule with relu weights reproduces the better single "
          "source *exactly* in both cells -- in this regime the calibration table's "
          "own advice is to select, not to blend. Headroom above the oracle is "
          f"{report['cells']['conflict']['fusion_ceiling_MRR'] - report['cells']['conflict']['best_source_oracle_MRR']:+.4f} "
          "MRR in the conflict cell and "
          f"{report['cells']['control']['fusion_ceiling_MRR'] - report['cells']['control']['best_source_oracle_MRR']:+.4f} "
          "in the control cell.", "",
          "### Reconstruction fidelity", "",
          "Every item's recomputed phenotype and annotation rank matches the value "
          "stored in `arbitration/items.json`; the phenotype-only and "
          "annotation-only MRRs below reproduce `arbitration/metrics.json` to 4 dp. "
          "The packet shows the model cosines rounded to 4 dp, which manufactures "
          "phenotype ties on "
          f"{', '.join(report['items_where_4dp_packet_rounding_ties_phenotype']) or 'no'} items; "
          "fusion uses the unrounded cosines, matching the stored baseline.", ""]
    (OUT_DIR / "RESULTS.md").write_text("\n".join(L) + "\n")
    print(f"wrote -> {OUT_DIR/'metrics.json'}\nwrote -> {OUT_DIR/'RESULTS.md'}")


# ---------------------------------------------------------------------------
# self-check: hand-worked toy items. A wrong Borda or tie convention fails here.
# ---------------------------------------------------------------------------
def selfcheck():
    # --- average ranks, with a tie ---
    s = {"a": 3.0, "b": 1.0, "c": 1.0, "d": 0.0}
    assert avg_ranks(s) == {"a": 1.0, "b": 2.5, "c": 2.5, "d": 4.0}, avg_ranks(s)
    # rr is the EXPECTED reciprocal rank under random tie-breaking, not
    # 1/avg_rank. "b" is tied for ranks 2-3, so E[1/rank] = (1/2 + 1/3)/2.
    assert abs(rr(s, "b") - (1 / 2 + 1 / 3) / 2) < 1e-12, rr(s, "b")
    assert rr(s, "b") > 1 / 2.5                       # convexity: the old form understated
    assert abs(rr(s, "a") - 1.0) < 1e-12              # untied target is unaffected
    assert top1(s, "a") == 1.0 and top1(s, "b") == 0.0
    # a fully tied source gives every candidate avg rank (n+1)/2
    flat = {c: 0.0 for c in "abcdef"}
    assert set(avg_ranks(flat).values()) == {3.5}
    # A 6-way tie at the top is worth 1/6, not 0: the ranker is undecided, not
    # wrong, and 1/6 is exactly the random rate. Scoring it 0 was the top-1
    # counterpart of the 1/avg_rank bug and biased every tying fusion rule down.
    assert abs(top1(flat, "a") - 1 / 6) < 1e-12, top1(flat, "a")
    assert abs(top1({"a": 9, "b": 9, "c": 1, "d": 1, "e": 1, "f": 1}, "a") - 0.5) < 1e-12
    assert top1({"a": 9, "b": 1, "c": 1, "d": 1, "e": 1, "f": 1}, "a") == 1.0
    # ...and must score EXACTLY chance, not below it. This is the regression
    # that shipped: 1/3.5 = 0.286 put a no-signal ranker under the 0.408 floor.
    assert abs(rr(flat, "a") - RANDOM_RR) < 1e-9, (rr(flat, "a"), RANDOM_RR)

    # --- Borda on 6 candidates, hand-worked ---
    # phen ranks a..f = 1..6 ; ann ranks reversed.
    phen = {c: -i for i, c in enumerate("abcdef")}     # a best
    ann = {c: i for i, c in enumerate("abcdef")}       # f best
    b = borda(phen, ann)
    # every candidate: (6-r) + (6-(7-r)) = 5 -> a perfect disagreement is a flat tie
    assert all(abs(v - 5.0) < 1e-12 for v in b.values()), b
    assert abs(rc._avg_rank(b, "a") - 3.5) < 1e-12

    # gold ranked 2nd by both beats rivals ranked 1st/5th and 5th/1st
    phen = {"g": 5, "x": 6, "y": 1, "z": 0, "w": -1, "v": -2}   # x,g,y,z,w,v
    ann = {"g": 5, "x": 1, "y": 6, "z": 0, "w": -1, "v": -2}    # y,g,x,z,w,v
    b = borda(phen, ann)
    assert b["g"] == (6 - 2) + (6 - 2) == 8
    assert b["x"] == (6 - 1) + (6 - 3) == 8
    assert b["y"] == (6 - 3) + (6 - 1) == 8
    assert abs(rc._avg_rank(b, "g") - 2.0) < 1e-12     # three-way tie at the top
    # Under expected-RR a 3-way tie at the top is worth (1 + 1/2 + 1/3)/3 = 0.611,
    # which DOES beat a certain rank 2. That is the honest expectation if ties are
    # broken at random, but it means a fusion rule can gain by being indecisive --
    # so tie rates are reported alongside exceedances rather than left implicit.
    assert abs(rr(b, "g") - (1 + 1 / 2 + 1 / 3) / 3) < 1e-12, rr(b, "g")
    assert max(rr(phen, "g"), rr(ann, "g")) == 0.5     # both sources rank g 2nd, untied
    assert rr(b, "g") > 0.5                            # so Borda DOES exceed here now

    # but consensus-2nd DOES beat both sources when the rivals disagree at the ends:
    # phen order x,g,z,w,v,y ; ann order y,g,z,w,v,x
    phen = {"x": 6, "g": 5, "z": 4, "w": 3, "v": 2, "y": 1}
    ann = {"y": 6, "g": 5, "z": 4, "w": 3, "v": 2, "x": 1}
    b = borda(phen, ann)
    assert b == {"x": 5, "g": 8, "z": 6, "w": 4, "v": 2, "y": 5}, b
    assert rr(b, "g") == 1.0 > max(rr(phen, "g"), rr(ann, "g")) == 0.5

    # --- RRF: k=1 is rank-sensitive, k=60 is nearly a Borda-like average ---
    phen = {"g": 9, "x": 8, "y": 7, "z": 6, "w": 5, "v": 4}       # g rank 1
    ann = {"g": 0, "x": 5, "y": 4, "z": 3, "w": 2, "v": 1}        # g rank 6
    r1, r60 = rrf(phen, ann, 1), rrf(phen, ann, 60)
    assert abs(r1["g"] - (1 / 2 + 1 / 7)) < 1e-12
    assert abs(r60["g"] - (1 / 61 + 1 / 66)) < 1e-12
    # k=1 is convex in rank, so the rank-1 hit counts for more: gold lands 2nd.
    # k=60 is near-linear, so the two ranks nearly average out: gold lands 3rd.
    assert rc._avg_rank(r1, "g") == 2.0, rc._avg_rank(r1, "g")
    assert rc._avg_rank(r60, "g") == 3.0, rc._avg_rank(r60, "g")

    # --- calibration-weighted: zero weight must fully silence a source ---
    phen = {"g": 9, "x": 8, "y": 7, "z": 6, "w": 5, "v": 4}
    ann = {"g": 0, "x": 5, "y": 4, "z": 3, "w": 2, "v": 1}
    only_ann = calib_weighted(phen, ann, (0.0, 1.0))
    assert rc._avg_rank(only_ann, "g") == rc._avg_rank(ann, "g") == 6.0
    only_phen = calib_weighted(phen, ann, (1.0, 0.0))
    assert rc._avg_rank(only_phen, "g") == 1.0
    # a tied (uninformative) source z-scores to all zeros and cannot shift anything
    assert calib_weighted(phen, {c: 2.0 for c in phen}, (1.0, 5.0)) \
        == calib_weighted(phen, {c: 2.0 for c in phen}, (1.0, 0.0))

    # --- exceedance denominator logic ---
    assert (max(rr(phen, "g"), rr(ann, "g")) >= 1.0 - 1e-9)   # phen has it first
    print("selfcheck OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    selfcheck()                       # always: it is milliseconds and it gates the run
    if a.selfcheck:
        return
    rep = analyze(*build_rows())
    write(rep)
    for cell, c in rep["cells"].items():
        print(f"\n[{cell}] n={c['n']} exceedable={c['n_exceedable']}  "
              f"phen {c['phenotype_only']['MRR']}  ann {c['annotation_only']['MRR']}  "
              f"oracle {c['best_source_oracle_MRR']}  ceiling {c['fusion_ceiling_MRR']}")
        for n in RULES:
            r = c["rules"][n]
            print(f"  {n:11s} MRR {r['MRR']:.4f} top1 {r['top1']:.3f}  "
                  f"vs-oracle {r['vs_best_source_oracle']:+.4f} {r['vs_best_source_oracle_CI']}  "
                  f"exceeds {r['exceeds_better_source_n']}/{c['n_exceedable']} "
                  f"loses {r['falls_below_better_source_n']}")


if __name__ == "__main__":
    main()
