#!/usr/bin/env python3
"""Confirmatory counterfactual experiment: UNCONDITIONED sample.

The pilot (pilot_counterfactual.py) was enriched -- items were chosen by gold
phenotype rank (strata 1-2 vs 5-6), so it could show the interaction exists but
could not estimate its size on a fair draw. This script fixes that:

  * Sample 48-60 benchmark-eligible query->gold relationships BEFORE looking at
    phenotype rank. No balancing, no conditioning on morphology similarity. Freeze.
  * Run only the four things that carry the science:
      A-mask    masked annotations, no phenotype   (evidence-disjoint baseline)
      P         phenotype-only numeric ranking, NO LLM (gold RR = 1/phen_rank)
      H-mask    masked annotations + real phenotype (rescue)
      H-shuffle masked annotations + shuffled phenotype (negative control)
    A-open/H-open are skipped -- the pilot already showed the open conditions
    saturate at MRR 1.00 (exposure inflation), which is not the interesting part.
  * Two candidate-order permutations, as before.
  * Stratify by gold phenotype rank AFTER evaluation only; the headline is the
    CONTINUOUS relationship: per-item Delta_i = RR_i(H-mask) - RR_i(A-mask)
    regressed on the gold candidate's phenotype rank.

Reuses the pilot's packet builder / prompt / client / parsers unchanged (the
masked conditions are identical); only the sampler, the run loop, and the
analysis differ.

  build    -> artifacts/evidence_eval/validate/items.json      (offline, unconditioned)
  run      -> artifacts/evidence_eval/validate/responses.jsonl (N*3*2 Vertex calls, resumable)
  analyze  -> metrics.json + validate_rescue_vs_rank.{png,pdf}

Usage:
  python scripts/evidence_eval/validate_counterfactual.py build   [--seed 123] [--n 60]
  python scripts/evidence_eval/validate_counterfactual.py run     [--model claude-sonnet-4-6] [--smoke]
  python scripts/evidence_eval/validate_counterfactual.py analyze
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Reuse the pilot harness (same dir on sys.path when run as a script).
import pilot_counterfactual as pilot
from pilot_counterfactual import (
    load_embeddings, load_splits, load_string, load_memberships,
    make_packet, SYSTEM, USER_TMPL, get_client, parse_response,
    _rr, _top1, POS_THRESH, NEG_THRESH,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ART = BASE_DIR / "artifacts" / "evidence_eval"
OUT_DIR = ART / "validate"  # reassigned by main() from --dir

VCONDITIONS = ["A-mask", "H-mask", "H-shuffle"]  # LLM calls; P is computed, not called
N_ORDERS = 2
N_ITEMS = 60


# ---------------------------------------------------------------------------
# build: unconditioned sample -- eligibility only, NO phenotype-rank filter
# ---------------------------------------------------------------------------
def build_items(seed=123, n=N_ITEMS, min_rank=1, max_rank=6,
                exclude_dir=None, prefix="v"):
    emb = load_embeddings()
    train, test = load_splits()
    train = {g for g in train if g in emb}
    test = {g for g in test if g in emb}
    pairs, degree = load_string(set(emb))
    g_path, g_cplx = load_memberships()

    def deg(g):
        return degree.get(g, 0)

    interactors = defaultdict(set)
    for pr in pairs:
        a, b = tuple(pr)
        interactors[a].add(b)
        interactors[b].add(a)

    # queries to skip (e.g. those already frozen in the fair sample)
    exclude = set()
    if exclude_dir:
        for d in exclude_dir.split(","):          # comma-list: exclude several prior sets
            ex = json.load(open(ART / d.strip() / "items.json"))
            exclude |= {it["query"] for it in ex["items"]}

    rng = random.Random(seed)
    queries = sorted(test)
    rng.shuffle(queries)

    items = []
    for q in queries:
        if len(items) >= n:
            break
        if q in exclude:
            continue
        # gold: highest-confidence STRING partner living in the train pool
        golds = sorted(
            (g for g in interactors[q]
             if g in train and pairs[frozenset((q, g))] >= POS_THRESH),
            key=lambda g: -pairs[frozenset((q, g))],
        )
        if not golds:
            continue
        gold = golds[0]
        q_ix = interactors[q]
        pool = [
            g for g in train
            if g != gold and g not in q_ix and g != q
            and (g in g_path or g in g_cplx)
        ]
        pool.sort(key=lambda g: (abs(deg(g) - deg(gold)), rng.random()))
        distractors = pool[:5]
        if len(distractors) < 5:
            continue
        cands = [gold] + distractors
        # phenotype rank + raw cosine recorded, but NOT used for selection
        sims = sorted(cands, key=lambda c: -float(emb[q] @ emb[c]))
        r = sims.index(gold) + 1
        if not (min_rank <= r <= max_rank):   # rank filter (booster only)
            continue
        items.append(dict(
            query=q, gold=gold, distractors=distractors,
            gold_degree=deg(gold), phen_rank=r,
            gold_cosine=round(float(emb[q] @ emb[gold]), 4),
        ))

    for i, it in enumerate(items):
        it["item_id"] = f"{prefix}{i:02d}"

    conditioned = (min_rank, max_rank) != (1, 6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"seed": seed, "pos_thresh": POS_THRESH, "neg_thresh": NEG_THRESH,
               "unconditioned": not conditioned,
               "rank_filter": [min_rank, max_rank], "excluded_dir": exclude_dir,
               "items": items},
              open(OUT_DIR / "items.json", "w"), indent=2)

    ranks = [it["phen_rank"] for it in items]
    kind = "rank-%d..%d ENRICHED" % (min_rank, max_rank) if conditioned else "UNCONDITIONED"
    print(f"Built {len(items)} {kind} items -> {OUT_DIR/'items.json'}")
    if len(items) < n:
        print(f"WARNING: only {len(items)} eligible items found (< {n})", file=sys.stderr)
    hist = {k: ranks.count(k) for k in range(1, 7)}
    print(f"  gold phenotype-rank histogram (should be ~uniform, NOT curated): {hist}")
    for it in items:
        print(f"    {it['item_id']} q={it['query']:9s} gold={it['gold']:9s} "
              f"rank={it['phen_rank']} cos={it['gold_cosine']} deg={it['gold_degree']}")


# ---------------------------------------------------------------------------
# run: N * 3 * 2 Vertex calls, resumable (P needs no calls)
# ---------------------------------------------------------------------------
def run(model, limit=None, smoke=False):
    items = json.load(open(OUT_DIR / "items.json"))["items"]
    ctx = pilot._ctx()
    client = get_client()

    out_path = OUT_DIR / "responses.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["item_id"], r["condition"], r["order"]))

    jobs = [(it, c, o) for it in items for c in VCONDITIONS for o in range(N_ORDERS)]
    if smoke:
        jobs = jobs[:2]
    if limit:
        jobs = jobs[:limit]
    jobs = [j for j in jobs if (j[0]["item_id"], j[1], j[2]) not in done]
    print(f"{len(jobs)} calls to make ({len(done)} already done) on {model}")

    f = open(out_path, "a")
    for n, (it, cond, order) in enumerate(jobs, 1):
        packet, gold_alias, audit = make_packet(it, cond, order, ctx)
        user = USER_TMPL.format(packet=json.dumps(packet))
        msg = client.messages.create(
            model=model, max_tokens=1024,
            system=SYSTEM, messages=[{"role": "user", "content": user}],
            extra_body={"temperature": 0.0},
        )
        text = msg.content[0].text
        try:
            parsed = parse_response(text)
        except Exception as e:
            parsed = {"_parse_error": str(e), "_raw": text[:500]}
        rec = dict(item_id=it["item_id"], phen_rank=it["phen_rank"],
                   gold_cosine=it["gold_cosine"], condition=cond, order=order,
                   gold_alias=gold_alias, model=model, audit=audit, response=parsed)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        rk = parsed.get("ranking", []) if isinstance(parsed, dict) else []
        pos = (rk.index(gold_alias) + 1) if gold_alias in rk else "-"
        print(f"[{n}/{len(jobs)}] {it['item_id']} {cond:9s} o{order} "
              f"gold={gold_alias} pos={pos}")
    f.close()
    print(f"Wrote -> {out_path}")


# ---------------------------------------------------------------------------
# analyze: P vs H-mask, per-item rank delta, Delta_i vs continuous phen rank
# ---------------------------------------------------------------------------
def _spearman(x, y):
    """Spearman rho = Pearson on ranks (ties -> average rank)."""
    def rankdata(a):
        a = np.asarray(a, float)
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a))
        r[order] = np.arange(len(a))
        # average tied ranks
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        csum = np.cumsum(cnt)
        start = csum - cnt
        avg = (start + csum - 1) / 2.0
        return avg[inv]
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _llm_rank(rec):
    """Position of gold in the LLM ranking (1=best); 6 if absent (worst)."""
    rk = rec["response"].get("ranking", []) if isinstance(rec["response"], dict) else []
    ga = rec["gold_alias"]
    return (rk.index(ga) + 1) if ga in rk else 6


def analyze():
    recs = [json.loads(l) for l in open(OUT_DIR / "responses.jsonl")]
    items = json.load(open(OUT_DIR / "items.json"))["items"]
    phen_rank = {it["item_id"]: it["phen_rank"] for it in items}
    item_ids = sorted({r["item_id"] for r in recs})

    # per (item, condition): average the 2 orders (repeated measures)
    def per_item(metric):
        acc = defaultdict(list)
        for r in recs:
            acc[(r["item_id"], r["condition"])].append(metric(r))
        return {k: float(np.mean(v)) for k, v in acc.items()}

    rr = per_item(_rr)                 # reciprocal rank (LLM)
    llm_rank = per_item(_llm_rank)     # integer rank of gold (LLM, avg of orders)

    def mrr(cond, ids):
        v = [rr[(i, cond)] for i in ids if (i, cond) in rr]
        return float(np.mean(v)) if v else float("nan")

    # P: phenotype-only ranking (no LLM). gold RR = 1/phen_rank.
    P_rr = {i: 1.0 / phen_rank[i] for i in item_ids}

    def P_mrr(ids):
        return float(np.mean([P_rr[i] for i in ids]))

    # post-hoc strata by gold phenotype rank (used ONLY for reporting)
    strata_ids = {
        "all": item_ids,
        "rank1-2": [i for i in item_ids if phen_rank[i] <= 2],
        "rank3-4": [i for i in item_ids if phen_rank[i] in (3, 4)],
        "rank5-6": [i for i in item_ids if phen_rank[i] >= 5],
    }

    def boot(vals, n=2000, seed=0):
        vals = np.asarray(vals, float)
        if len(vals) == 0:
            return (float("nan"),) * 3
        rng = np.random.default_rng(seed)
        bs = vals[rng.integers(0, len(vals), size=(n, len(vals)))].mean(axis=1)
        return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    report = {"model": recs[0]["model"], "n_items": len(item_ids),
              "unconditioned": True, "conditions": {}, "P_vs_Hmask": {},
              "rescue": {}, "continuous": {}}

    for name, ids in strata_ids.items():
        report["conditions"][name] = {"n": len(ids)}
        for c in VCONDITIONS:
            report["conditions"][name][c] = round(mrr(c, ids), 4)
        report["conditions"][name]["P"] = round(P_mrr(ids), 4)

        # P vs H-mask
        Hm = mrr("H-mask", ids)
        Pv = P_mrr(ids)
        # per-item (H-mask RR) - (P RR), bootstrapped
        diff = [rr[(i, "H-mask")] - P_rr[i] for i in ids if (i, "H-mask") in rr]
        m, lo, hi = boot(diff)
        report["P_vs_Hmask"][name] = {
            "P_MRR": round(Pv, 4), "Hmask_MRR": round(Hm, 4),
            "Hmask_minus_P_MRR": round(Hm - Pv, 4),
            "per_item_boot_mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}

        # rescue Delta = RR(H-mask) - RR(A-mask), per item, bootstrapped
        resc = [rr[(i, "H-mask")] - rr[(i, "A-mask")]
                for i in ids if (i, "H-mask") in rr and (i, "A-mask") in rr]
        m, lo, hi = boot(resc)
        # shuffle control: RR(H-shuffle) - RR(A-mask)
        shuf = [rr[(i, "H-shuffle")] - rr[(i, "A-mask")]
                for i in ids if (i, "H-shuffle") in rr and (i, "A-mask") in rr]
        ms, los, his = boot(shuf)
        report["rescue"][name] = {
            "delta_rescue_mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "shuffle_control_mean": round(ms, 4), "shuffle_ci95": [round(los, 4), round(his, 4)]}

    # continuous: Delta_i vs gold phenotype rank (the headline)
    ids = strata_ids["all"]
    x = np.array([phen_rank[i] for i in ids], float)
    delta = np.array([rr[(i, "H-mask")] - rr[(i, "A-mask")] for i in ids], float)
    # per-item LLM-vs-phenotype rank gap (H-mask): negative => LLM better than pheno-only
    rank_gap = {i: llm_rank[(i, "H-mask")] - phen_rank[i]
                for i in ids if (i, "H-mask") in llm_rank}
    slope, intercept = np.polyfit(x, delta, 1)
    report["continuous"] = {
        "predictor": "gold phenotype rank (1=most similar .. 6=least)",
        "spearman_delta_vs_rank": round(_spearman(x, delta), 4),
        "ols_slope": round(float(slope), 4),
        "ols_intercept": round(float(intercept), 4),
        "note": "negative slope/rho => rescue shrinks (then reverses) as phenotype rank worsens",
    }
    report["rank_gap_Hmask"] = {
        "mean_r_llm_minus_r_pheno": round(float(np.mean(list(rank_gap.values()))), 4),
        "n_llm_better": int(sum(1 for v in rank_gap.values() if v < 0)),
        "n_tie": int(sum(1 for v in rank_gap.values() if v == 0)),
        "n_llm_worse": int(sum(1 for v in rank_gap.values() if v > 0)),
    }

    n_err = sum(1 for r in recs if "_parse_error" in r["response"])
    n_masknoop = sum(1 for r in recs
                     if r["audit"]["masked"] and r["audit"]["qgold_records_removed"] == 0)
    report["sanity"] = {"n_calls": len(recs), "parse_errors": n_err,
                        "masked_calls_removing_0_records": n_masknoop}

    json.dump(report, open(OUT_DIR / "metrics.json", "w"), indent=2)
    print(json.dumps(report, indent=2))
    _figure(ids, x, delta, slope, intercept, report)


def _figure(ids, x, delta, slope, intercept, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    jit = x + rng.uniform(-0.12, 0.12, size=len(x))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.scatter(jit, delta, s=42, alpha=0.7, color="#2166ac",
               edgecolor="white", linewidth=0.5, zorder=3)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, slope * xs + intercept, color="#b2182b", lw=2,
            label=f"OLS slope={slope:+.3f}  (Spearman ρ={report['continuous']['spearman_delta_vs_rank']:+.2f})")
    # bin means for readability
    for lo, hi, lab in [(1, 2, "1-2"), (3, 4, "3-4"), (5, 6, "5-6")]:
        m = (x >= lo) & (x <= hi)
        if m.any():
            ax.plot((lo + hi) / 2, delta[m].mean(), marker="D", ms=10,
                    color="#000000", zorder=4)
    ax.set_xlabel("gold candidate's phenotype rank (1 = most morphologically similar)")
    ax.set_ylabel(r"$\Delta_i$ = RR(H-mask) − RR(A-mask)")
    ax.set_title("Phenotype rescue vs. how informative the phenotype is\n(unconditioned sample, n=%d)" % len(ids))
    ax.set_xticks(range(1, 7))
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"validate_rescue_vs_rank.{ext}", dpi=200)
    print(f"Figure -> {OUT_DIR/'validate_rescue_vs_rank.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dir", default="validate")
    b.add_argument("--seed", type=int, default=123)
    b.add_argument("--n", type=int, default=N_ITEMS)
    b.add_argument("--min-rank", type=int, default=1)
    b.add_argument("--max-rank", type=int, default=6)
    b.add_argument("--exclude-dir", default=None)
    b.add_argument("--prefix", default="v")
    r = sub.add_parser("run")
    r.add_argument("--dir", default="validate")
    r.add_argument("--model", default="claude-sonnet-4-6")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--smoke", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("--dir", default="validate")
    args = ap.parse_args()

    global OUT_DIR
    OUT_DIR = ART / args.dir

    if args.cmd == "build":
        build_items(args.seed, args.n, args.min_rank, args.max_rank,
                    args.exclude_dir, args.prefix)
    elif args.cmd == "run":
        run(args.model, args.limit, args.smoke)
    elif args.cmd == "analyze":
        analyze()


if __name__ == "__main__":
    main()
