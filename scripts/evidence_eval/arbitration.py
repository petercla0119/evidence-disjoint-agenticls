#!/usr/bin/env python
"""Evidence arbitration under calibrated source reliability.

Question: can the model arbitrate between two conflicting evidence sources when
it is told, in observable terms, which one is reliable here?

Fixes two flaws in scripts/evidence_eval/reliability_calibration.py (artifacts/evidence_eval/reliability):
  1. That run's ranking arm was handed a near-flat per-rank table saying phenotype
     (0.22 top bin) beats random (0.167) while annotation (0.07) is below it. The
     damning regime table only ever reached the comprehension probe. Following
     phenotype was the rational response, so the null is not interpretable.
  2. Its regime was keyed on the gold's phenotype rank -- unobservable at test
     time. A conditional the model cannot discharge cannot change its behaviour.

Here reliability is keyed on two statistics computable from the packet without
knowing the gold: the phenotype top1-top2 cosine GAP, and the annotation-graph
top1-top2 MARGIN. Measured on a query-disjoint calibration half.

Two cells, both selected on those observables only (never on correctness):
  CONFLICT  gap < 0.031, margin >= 1.5  -> phenotype ~random, annotation ~2.3x better
  CONTROL   gap >= 0.059, annotation tie -> phenotype reliable, annotation silent
The control cell is what separates real arbitration from "drifts toward whichever
source the reliability block mentions".

Usage:
    PYTHONHASHSEED=0 python scripts/evidence_eval/arbitration.py build
    PYTHONHASHSEED=0 python scripts/evidence_eval/arbitration.py calibrate
    PYTHONHASHSEED=0 python scripts/evidence_eval/arbitration.py run --model claude-sonnet-4-6
    python scripts/evidence_eval/arbitration.py analyze
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_counterfactual as pilot          # noqa: E402
import reliability_calibration as rc          # noqa: E402
import validate_counterfactual as V           # noqa: E402

ART = V.ART
OUT_DIR = ART / "arbitration"
SCAN_CACHE = OUT_DIR / "pool_scan.json"

SPLIT_SEED = 20260914      # query-level calibration/eval split
CALIB_FRAC = 0.5
MAX_GOLDS = 8              # cap golds per query so hub genes cannot dominate
N_ORDERS = 2

# cell definitions -- functions of (gap, margin) only, never of the gold's rank
# cell bounds coincide with calibration-table bin edges, so the row injected for
# an item is exactly the claim under test (phenotype 0.08 < random 0.167 < annotation 0.34)
CONFLICT = dict(name="conflict", gap_max=0.013, margin_min=1.5, n=40)
CONTROL = dict(name="control", gap_min=0.059, margin_max=1e-9, n=20)

GAP_BINS = [(0.0, 0.013, "gap<0.013"), (0.013, 0.059, "gap 0.013-0.059"),
            (0.059, 9.0, "gap>=0.059")]
MARGIN_BINS = [(-1.0, 1e-9, "no discriminating annotation"),
               (1e-9, 1.5, "annotation margin 0-1.5"),
               (1.5, 9e9, "annotation margin>=1.5")]
RANDOM_CAND = 1 / 6
RANDOM_RR = float(np.mean([1.0 / k for k in range(1, 7)]))

ARMS = ["naive", "rel", "rel_scaffold"]


# ---------------------------------------------------------------------------
# offline pool scan -- every eligible (query, gold) combo with its observables
# ---------------------------------------------------------------------------
def _bin(value, bins):
    for lo, hi, label in bins:
        if lo <= value < hi:
            return label
    return bins[-1][2]


def scan_pool(force=False):
    """Every eligible (query, gold) combo + its observable statistics.

    gap    = top1 - top2 phenotype cosine over the 6 candidates
    margin = top1 - top2 annotation-graph score on the MASKED packet
    Neither references which candidate is the gold, so selecting on them is
    correctness-blind (unlike validate_booster, which selects on phen_rank).
    """
    if SCAN_CACHE.exists() and not force:
        return json.load(open(SCAN_CACHE))

    emb = V.load_embeddings()
    train, test = V.load_splits()
    train = {g for g in train if g in emb}
    test = {g for g in test if g in emb}
    pairs, degree = V.load_string(set(emb))
    g_path, g_cplx = V.load_memberships()
    ctx = (emb, pairs, degree, g_path, g_cplx)

    interactors = defaultdict(set)
    for pr in pairs:
        a, b = tuple(pr)
        interactors[a].add(b)
        interactors[b].add(a)

    def deg(g):
        return degree.get(g, 0)

    rng = random.Random(999)          # distractor tiebreak only
    rows = []
    for qi, q in enumerate(sorted(test)):
        golds = sorted(
            (g for g in interactors[q]
             if g in train and pairs[frozenset((q, g))] >= V.POS_THRESH),
            key=lambda g: -pairs[frozenset((q, g))],
        )[:MAX_GOLDS]
        for gold in golds:
            pool = [g for g in train
                    if g != gold and g not in interactors[q] and g != q
                    and (g in g_path or g in g_cplx)]
            if len(pool) < 5:
                continue
            pool.sort(key=lambda g: (abs(deg(g) - deg(gold)), rng.random()))
            distractors = pool[:5]
            cands = [gold] + distractors

            sims = np.array([float(emb[q] @ emb[c]) for c in cands])
            srt = np.sort(sims)[::-1]
            gap = float(srt[0] - srt[1])
            phen_rank = int(list(np.argsort(-sims)).index(0)) + 1

            item = dict(query=q, gold=gold, distractors=distractors,
                        gold_degree=deg(gold), phen_rank=phen_rank,
                        gold_cosine=round(float(emb[q] @ emb[gold]), 4),
                        item_id=f"s{qi:04d}")
            packet, gold_alias, _ = pilot.make_packet(item, "A-mask", 0, ctx)
            score = rc._ann_scores(packet)
            vals = sorted(score.values(), reverse=True)
            margin = float(vals[0] - vals[1])

            rows.append(dict(
                query=q, gold=gold, distractors=distractors,
                gold_degree=deg(gold), phen_rank=phen_rank,
                gold_cosine=round(float(emb[q] @ emb[gold]), 4),
                gap=gap, margin=margin,
                ann_rank=float(rc._avg_rank(score, gold_alias)),
                phen_hit=bool(phen_rank == 1),
                ann_hit=bool(margin > 1e-9 and score[gold_alias] == vals[0]),
            ))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(SCAN_CACHE, "w"))
    print(f"scanned {len(rows)} (query, gold) combos -> {SCAN_CACHE}")
    return rows


def split_queries(rows):
    """Query-level calibration/eval split. Reliability is never fit to the
    items it scores, and no query appears on both sides."""
    queries = sorted({r["query"] for r in rows})
    random.Random(SPLIT_SEED).shuffle(queries)
    cut = int(len(queries) * CALIB_FRAC)
    return set(queries[:cut]), set(queries[cut:])


# ---------------------------------------------------------------------------
# build: pick eval items for each cell, one per query
# ---------------------------------------------------------------------------
def _in_cell(row, cell):
    if "gap_max" in cell and not row["gap"] < cell["gap_max"]:
        return False
    if "gap_min" in cell and not row["gap"] >= cell["gap_min"]:
        return False
    if "margin_min" in cell and not row["margin"] >= cell["margin_min"]:
        return False
    if "margin_max" in cell and not row["margin"] < cell["margin_max"]:
        return False
    return True


def build():
    rows = scan_pool()
    calib_q, eval_q = split_queries(rows)
    rng = random.Random(SPLIT_SEED + 1)

    items, used_q = [], set()
    for cell in (CONFLICT, CONTROL):
        pool = [r for r in rows if r["query"] in eval_q and _in_cell(r, cell)]
        rng.shuffle(pool)
        picked = []
        for r in pool:                        # one item per query -> independence
            if r["query"] in used_q:
                continue
            used_q.add(r["query"])
            picked.append(r)
            if len(picked) >= cell["n"]:
                break
        print(f"cell {cell['name']:9s}: {len(pool):4d} eligible in eval half, "
              f"{len({r['query'] for r in pool}):4d} distinct queries -> picked {len(picked)}")
        if len(picked) < cell["n"]:
            print(f"  WARNING: wanted {cell['n']}, got {len(picked)}")
        for i, r in enumerate(picked):
            items.append(dict(
                item_id=f"{cell['name'][:4]}{i:02d}", cell=cell["name"],
                query=r["query"], gold=r["gold"], distractors=r["distractors"],
                gold_degree=r["gold_degree"], phen_rank=r["phen_rank"],
                gold_cosine=r["gold_cosine"],
                gap=round(r["gap"], 4), margin=round(r["margin"], 4),
                ann_rank=r["ann_rank"],
            ))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"split_seed": SPLIT_SEED, "calib_frac": CALIB_FRAC,
               "pos_thresh": V.POS_THRESH,
               "selection": "observable only (gap, margin); never conditioned on "
                            "which candidate is the gold",
               "cells": {"conflict": CONFLICT, "control": CONTROL},
               "n_calib_queries": len(calib_q), "n_eval_queries": len(eval_q),
               "items": items},
              open(OUT_DIR / "items.json", "w"), indent=2)
    print(f"\nwrote {len(items)} items -> {OUT_DIR/'items.json'}")
    return items


# ---------------------------------------------------------------------------
# calibrate: 3x3 observable reliability table on the calibration half
# ---------------------------------------------------------------------------
def calibrate():
    rows = scan_pool()
    calib_q, _ = split_queries(rows)
    cal_rows = [r for r in rows if r["query"] in calib_q]

    table = {}
    for glo, ghi, glab in GAP_BINS:
        for mlo, mhi, mlab in MARGIN_BINS:
            sub = [r for r in cal_rows
                   if glo <= r["gap"] < ghi and mlo <= r["margin"] < mhi]
            if not sub:
                continue
            table[f"{glab} | {mlab}"] = {
                "n": len(sub),
                "phenotype_top_pick_is_true_partner": round(float(np.mean([r["phen_hit"] for r in sub])), 3),
                "annotation_top_pick_is_true_partner": round(float(np.mean([r["ann_hit"] for r in sub])), 3),
                "phenotype_expected_reciprocal_rank": round(float(np.mean([1 / r["phen_rank"] for r in sub])), 3),
                # KNOWN UNDERSTATEMENT, DELIBERATELY NOT FIXED. 1/ann_rank is
                # 1/E[rank], which understates E[1/rank] for a tied gold (see
                # rc.expected_rr). In the "no discriminating annotation" bins the
                # gold is tied 6 ways, so the true expected reciprocal rank is
                # exactly the 0.408 random rate, not the ~0.29 recorded here.
                #
                # calibration.json is a FROZEN input -- it is the table the model
                # was actually shown, and every run is paired against it.
                # Recomputing it would invalidate all stored responses. So this
                # line stays as-is and the discrepancy is documented instead.
                # It does not affect any conclusion: the advice the table encodes
                # ("annotation carries no information here") is correct under both
                # conventions, since 0.408 is *at* chance and 0.29 is below it.
                # The sub-chance phenotype figure that drives the cell-design
                # finding (conflict, 0.369) comes from phen_rank, which is an
                # integer and never tied, so it is unaffected.
                "annotation_expected_reciprocal_rank": round(float(np.mean([1 / r["ann_rank"] for r in sub])), 3),
            }

    cal = {
        "split_seed": SPLIT_SEED,
        "measured_on": f"calibration half, {len(calib_q)} queries / {len(cal_rows)} pairs, "
                       f"query-disjoint from the evaluated items",
        "keys": "phenotype gap = top1-top2 cosine; annotation margin = top1-top2 graph score. "
                "Both computable from the packet without knowing the answer.",
        "random_candidate_is_true_partner": round(RANDOM_CAND, 3),
        "random_ranking_expected_reciprocal_rank": round(RANDOM_RR, 3),
        "table": table,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(cal, open(OUT_DIR / "calibration.json", "w"), indent=2)

    print(f"calibration half: {len(calib_q)} queries, {len(cal_rows)} pairs")
    print(f"{'cell':34s}{'n':>6s}{'P(phen)':>9s}{'P(ann)':>8s}{'phenRR':>8s}{'annRR':>7s}")
    for k, v in table.items():
        print(f"{k:34s}{v['n']:6d}{v['phenotype_top_pick_is_true_partner']:9.2f}"
              f"{v['annotation_top_pick_is_true_partner']:8.2f}"
              f"{v['phenotype_expected_reciprocal_rank']:8.2f}"
              f"{v['annotation_expected_reciprocal_rank']:7.2f}")
    print(f"\nrandom: candidate {RANDOM_CAND:.3f}, ranking RR {RANDOM_RR:.3f}")
    print(f"wrote -> {OUT_DIR/'calibration.json'}")
    return cal


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
SYS_REL = pilot.SYSTEM + (
    " The packet also carries a source_reliability block: hit-rates measured on an "
    "independent, query-disjoint calibration set, keyed on two statistics that are "
    "computed for you in this_item -- the phenotype top1-top2 gap and the annotation "
    "top1-top2 margin. Look up the row matching this_item and weigh each source by "
    "how reliable it is in that row, relative to the random rate. A source whose "
    "top pick is right at or below the random rate carries no information here."
)

SYS_SCAFFOLD = SYS_REL + (
    " Before giving your final ranking, you must explicitly rank the candidates by "
    "each source separately and state which source you are weighting more heavily "
    "and why, citing the calibration row."
)

USER_SCAFFOLD = (
    "Evidence packet (JSON):\n{packet}\n\n"
    "Task: rank all six candidates G1-G6 from most to least likely to be the true "
    "functional partner of G0. Return ONLY machine-readable JSON with keys:\n"
    '  "phenotype_ranking": [six candidate ids, best first by phenotype alone],\n'
    '  "annotation_ranking": [six candidate ids, best first by annotation alone],\n'
    '  "calibration_row": "<the source_reliability key matching this_item>",\n'
    '  "weighted_source": "phenotype" | "annotation" | "neither",\n'
    '  "weighting_reason": "<one sentence citing the calibration numbers>",\n'
    '  "ranking": [six candidate ids, best first],\n'
    '  "probabilities": {{candidate id: probability}}  (must sum to 1),\n'
    '  "first_choice_evidence": [record ids R# supporting your top pick]\n'
    "No prose outside the JSON."
)


def _augment(packet, item, cal, arm):
    """Annotation graph scores go into EVERY arm -- they are a deterministic
    summary of evidence already in the packet, so the only thing that varies
    across arms is the reliability information."""
    score = rc._ann_scores(packet)
    packet["annotation_graph_score"] = {k: round(v, 3) for k, v in score.items()}
    if arm == "naive":
        return packet
    packet["source_reliability"] = {
        "measured_on": cal["measured_on"],
        "keys": cal["keys"],
        "random_candidate_is_true_partner": cal["random_candidate_is_true_partner"],
        "random_ranking_expected_reciprocal_rank": cal["random_ranking_expected_reciprocal_rank"],
        "table": cal["table"],
        "this_item": {
            "phenotype_gap": round(item["gap"], 4),
            "annotation_margin": round(item["margin"], 4),
            "matching_row": f"{_bin(item['gap'], GAP_BINS)} | {_bin(item['margin'], MARGIN_BINS)}",
        },
    }
    return packet


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def run(model, limit=None, smoke=False, retest=False, provider="vertex",
        arms=None):
    """retest=True re-runs the naive arm on the conflict cell into a separate
    file, so test-retest agreement gives a run-to-run noise floor. Needed
    because temperature 0 is NOT bit-deterministic here: the same
    (arm, item, order) key has returned different rankings (RR 0.5 vs 0.333).
    Bootstrap CIs over items absorb order variance but not sampling variance,
    so without this a null cannot be distinguished from the noise floor."""
    cal = json.load(open(OUT_DIR / "calibration.json"))
    items = json.load(open(OUT_DIR / "items.json"))["items"]
    ctx = pilot._ctx()
    # ponytail: second model family reuses tool_agent's Azure client. Separate
    # response file so the two models never share an (arm, item, order) key.
    if provider == "azure":
        import tool_agent as _TA
        client, model = _TA._azure_client(), _TA.AZURE_DEPLOYMENT
    else:
        client = pilot.get_client()
    suffix = "_gpt56" if provider == "azure" else ""
    out_path = OUT_DIR / (f"responses_retest{suffix}.jsonl" if retest
                          else f"responses{suffix}.jsonl")
    done = set()
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["arm"], r["item_id"], r["order"]))

    jobs = [(arm, it, o) for it in items for arm in (arms or ARMS)
            for o in range(N_ORDERS)]
    if retest:
        jobs = [j for j in jobs if j[0] == "naive" and j[1]["cell"] == "conflict"]
    if smoke:
        jobs = [j for j in jobs if j[1]["item_id"] in {"conf00", "cont00"}][:6]
    jobs = [j for j in jobs if (j[0], j[1]["item_id"], j[2]) not in done]
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} calls to make ({len(done)} already done) on {model}")

    f = open(out_path, "a")
    for n, (arm, it, order) in enumerate(jobs, 1):
        packet, gold_alias, audit = pilot.make_packet(it, "H-mask", order, ctx)
        packet = _augment(packet, it, cal, arm)
        system = pilot.SYSTEM if arm == "naive" else (
            SYS_SCAFFOLD if arm == "rel_scaffold" else SYS_REL)
        tmpl = USER_SCAFFOLD if arm == "rel_scaffold" else pilot.USER_TMPL
        user_msg = tmpl.format(packet=json.dumps(packet))
        if provider == "azure":
            # deployment rejects temperature=0 and max_tokens; episodes are sampled
            r_ = client.chat.completions.create(
                # gpt-5.x spends reasoning tokens from this same budget; 1500
                # left empty completions on the longer static-arm JSON.
                model=model, max_completion_tokens=6000,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user_msg}])
            text = r_.choices[0].message.content or ""
        else:
            msg = client.messages.create(
                model=model, max_tokens=1500, system=system,
                messages=[{"role": "user", "content": user_msg}],
                extra_body={"temperature": 0.0},
            )
            text = msg.content[0].text
        try:
            parsed = pilot.parse_response(text)
        except Exception as e:
            parsed = {"_parse_error": str(e), "_raw": text[:500]}

        rec = dict(arm=arm, item_id=it["item_id"], cell=it["cell"], order=order,
                   gold_alias=gold_alias, phen_rank=it["phen_rank"],
                   ann_rank=it["ann_rank"], gap=it["gap"], margin=it["margin"],
                   model=model, audit=audit, response=parsed)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        rk = parsed.get("ranking", []) if isinstance(parsed, dict) else []
        pos = (rk.index(gold_alias) + 1) if gold_alias in rk else "-"
        print(f"[{n}/{len(jobs)}] {arm:12s} {it['item_id']:7s} o{order} "
              f"{it['cell']:8s} gold={gold_alias} pos={pos}")
    f.close()
    print(f"wrote -> {out_path}")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
def dedup():
    """Two sessions appended to responses.jsonl concurrently. Keep the first row
    per (arm, item_id, order); preserve the pre-dedup file as responses.raw.jsonl
    (it is the only record of repeated identical keys, which measure run-to-run
    variance)."""
    path = OUT_DIR / "responses.jsonl"
    raw = OUT_DIR / "responses.raw.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not raw.exists():
        raw.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"preserved {len(rows)} pre-dedup rows -> {raw}")
    seen, keep = set(), []
    for r in rows:
        k = (r["arm"], r["item_id"], r["order"])
        if k in seen:
            continue
        seen.add(k)
        keep.append(r)
    path.write_text("".join(json.dumps(r) + "\n" for r in keep))
    print(f"deduped {len(rows)} -> {len(keep)} rows ({len(rows)-len(keep)} dropped) -> {path}")


def _rr(r):
    resp = r["response"]
    if not isinstance(resp, dict) or "ranking" not in resp:
        return None
    rk = resp["ranking"]
    return 1.0 / (rk.index(r["gold_alias"]) + 1) if r["gold_alias"] in rk else 0.0


def _tie_aware_baseline_rr():
    """{item_id: expected reciprocal rank of the annotation-only ranker}.

    Read from the fusion sidecar (exp20_fusion_baseline.py), which reconstructs
    the masked packet and so knows the tie structure. items.json stores only the
    AVERAGE rank, and the tie structure is not recoverable from it -- an average
    of 3.5 is equally consistent with a 6-way, 4-way or 2-way tie, which have
    expected reciprocal ranks 0.408, 0.319 and 0.292.

    Model responses are strict permutations and need none of this; only the
    deterministic baselines tie.
    """
    side = ART / "fusion" / "metrics.json"
    if not side.exists():
        raise SystemExit(
            f"missing {side}\n"
            "Run `PYTHONHASHSEED=0 python scripts/evidence_eval/exp20_fusion_baseline.py` first.\n"
            "Refusing to fall back to 1/ann_rank: it scores a fully-tied ranker below "
            "chance and silently corrupts every annotation-only comparison.")
    per_item = json.loads(side.read_text())["per_item"]
    return {r["item_id"]: r["annotation"] for r in per_item}


def noise_floor():
    """Test-retest agreement on identical keys = the run-to-run noise floor that
    any reported effect has to clear. Sources: accidental duplicates in
    responses.raw.jsonl, plus the deliberate naive/conflict retest."""
    pairs = []
    raw = OUT_DIR / "responses.raw.jsonl"
    if raw.exists():
        byk = defaultdict(list)
        for line in open(raw):
            r = json.loads(line)
            byk[(r["arm"], r["item_id"], r["order"])].append(r)
        for k, v in byk.items():
            if len(v) > 1:
                a, b = _rr(v[0]), _rr(v[1])
                if a is not None and b is not None:
                    pairs.append(("accidental", k, a, b))
    rt = OUT_DIR / "responses_retest.jsonl"
    if rt.exists():
        first = {}
        for line in open(OUT_DIR / "responses.jsonl"):
            r = json.loads(line)
            first[(r["arm"], r["item_id"], r["order"])] = _rr(r)
        for line in open(rt):
            r = json.loads(line)
            k = (r["arm"], r["item_id"], r["order"])
            a, b = first.get(k), _rr(r)
            if a is not None and b is not None:
                pairs.append(("retest", k, a, b))
    if not pairs:
        print("no repeated keys available")
        return None
    d = [abs(a - b) for _, _, a, b in pairs]
    exact = sum(1 for x in d if x < 1e-9)
    out = {
        "n_repeated_keys": len(pairs),
        "exact_agreement_rate": round(exact / len(pairs), 3),
        "mean_abs_RR_difference": round(float(np.mean(d)), 4),
        "p95_abs_RR_difference": round(float(np.percentile(d, 95)), 4),
        "implied_MRR_noise_se_40_items": round(float(np.std(d, ddof=1) / np.sqrt(40)), 4) if len(d) > 1 else None,
        "sources": dict(Counter(s for s, _, _, _ in pairs)),
    }
    json.dump(out, open(OUT_DIR / "noise_floor.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    return out


def _boot(vals, n=10000, seed=0):
    if not vals:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, dtype=float)
    d = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return (round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4))


def _source_rankings(item, order, ctx):
    """Phenotype and annotation rankings under this item's alias/order, plus the
    gold alias. Strict orderings (sort-order tiebreak), used for exact-ranking
    agreement; the MRR references elsewhere use average-rank tie handling, which
    is stricter on annotation. Conclusions below hold under both."""
    packet, gold, _ = pilot.make_packet(item, "H-mask", order, ctx)
    ph = packet["phenotype_similarity"]
    an = rc._ann_scores(packet)
    return (sorted(ph, key=lambda c: -ph[c]),
            sorted(an, key=lambda c: -an[c]), gold)


def _selector_stats(ids, items, rr, P, A, oracle, suffix="", arms=ARMS):
    """Is the model fusing sources or selecting one? Three tests:
    - paired per-item difference vs each single-source oracle (bootstrap CI)
    - RR ties and EXACT full-ranking matches with each source
    - how often it exceeds max(P, ANN), i.e. produces something better than the
      best source it was handed
    """
    ctx = pilot._ctx()
    rows = [json.loads(l) for l in open(OUT_DIR / f"responses{suffix}.jsonl")]
    by = {(r["arm"], r["item_id"], r["order"]): r for r in rows}
    out = {}
    for arm in arms:
        exact_p = exact_a = 0
        for iid in ids:
            for o in range(N_ORDERS):
                r = by.get((arm, iid, o))
                if not r or not isinstance(r["response"], dict):
                    continue
                mk = r["response"].get("ranking")
                pr, ar, gold = _source_rankings(items[iid], o, ctx)
                # make_packet derives the candidate permutation from
                # hash((item_id, order)). PYTHONHASHSEED=0 pins str hashing only
                # WITHIN a Python version -- 3.11 switched SipHash-2-4 -> 1-3, so
                # hash(('conf00',0)) differs between 3.10 and 3.13. Recomputing
                # packets post-hoc on the wrong interpreter yields different
                # aliases and silently produces exact-match 0 everywhere, which
                # looks like a real null. Crash instead.
                if gold != r["gold_alias"]:
                    raise RuntimeError(
                        f"alias mismatch for {arm}/{iid}/o{o}: recomputed {gold} != "
                        f"stored {r['gold_alias']}. The candidate permutation depends on "
                        f"hash((item_id, order)), which is NOT stable across Python "
                        f"versions even with PYTHONHASHSEED=0. Re-run under the same "
                        f"interpreter as the original run (conda ops-experimental, "
                        f"Python 3.10) with PYTHONHASHSEED=0.")
                if mk == pr:
                    exact_p += 1
                if mk == ar:
                    exact_a += 1
        d_p = [rr[arm][i] - P[i] for i in ids]
        d_a = [rr[arm][i] - A[i] for i in ids]
        out[arm] = {
            "minus_phenotype_oracle": round(float(np.mean(d_p)), 4),
            "minus_phenotype_oracle_CI": _boot(d_p),
            "minus_annotation_oracle": round(float(np.mean(d_a)), 4),
            "minus_annotation_oracle_CI": _boot(d_a),
            "rr_ties_phenotype": sum(1 for i in ids if abs(rr[arm][i] - P[i]) < 1e-9),
            "rr_ties_annotation": sum(1 for i in ids if abs(rr[arm][i] - A[i]) < 1e-9),
            "exact_ranking_matches_phenotype": exact_p,
            "exact_ranking_matches_annotation": exact_a,
            "n_item_order_calls": len(ids) * N_ORDERS,
            "exceeds_both_sources": sum(1 for i in ids if rr[arm][i] > oracle[i] + 1e-9),
        }
    return out


def analyze(provider="vertex"):
    # azure = model B (gpt-5.6-luna), which ran only the naive and rel arms.
    suffix = "_gpt56" if provider == "azure" else ""
    items = {it["item_id"]: it for it in json.load(open(OUT_DIR / "items.json"))["items"]}
    rr = defaultdict(dict)          # rr[arm][item_id] = mean RR over orders
    acc = defaultdict(lambda: defaultdict(list))
    n_parse_err = 0
    for line in open(OUT_DIR / f"responses{suffix}.jsonl"):
        r = json.loads(line)
        resp = r["response"]
        if not isinstance(resp, dict) or "ranking" not in resp:
            n_parse_err += 1
            continue
        rk = resp["ranking"]
        acc[r["arm"]][r["item_id"]].append(
            1.0 / (rk.index(r["gold_alias"]) + 1) if r["gold_alias"] in rk else 0.0)
    for arm, per_item in acc.items():
        for iid, vals in per_item.items():
            rr[arm][iid] = float(np.mean(vals))

    arms = [a for a in ARMS if a in rr]
    report = {"parse_errors": n_parse_err, "cells": {}}
    for cell in ("conflict", "control"):
        ids = [i for i, it in items.items() if it["cell"] == cell and
               all(i in rr[a] for a in arms)]
        if not ids:
            continue
        tie_rr = _tie_aware_baseline_rr()
        P = {i: 1.0 / items[i]["phen_rank"] for i in ids}
        # NOT 1/ann_rank. ann_rank is an AVERAGE rank, so 1/ann_rank is 1/E[rank];
        # reciprocal rank is convex, so that understates E[1/rank] whenever gold is
        # tied. It scored the control cell's 6-way-tied annotation ranker 0.281 --
        # below the 0.408 chance rate, which is impossible for a ranker with no
        # anti-signal. The tie structure is not recoverable from the stored average
        # (3.5 is consistent with a 6-way, 4-way or 2-way tie), so the expected
        # values are read from the fusion sidecar, which reconstructs the scores.
        A = {i: tie_rr[i] for i in ids}
        oracle = {i: max(P[i], A[i]) for i in ids}
        out = {
            "n": len(ids),
            "phenotype_only_MRR": round(float(np.mean([P[i] for i in ids])), 4),
            "annotation_only_MRR": round(float(np.mean([A[i] for i in ids])), 4),
            "random_MRR": round(RANDOM_RR, 4),
            "oracle_arbiter_MRR": round(float(np.mean([oracle[i] for i in ids])), 4),
        }
        for arm in arms:
            v = [rr[arm][i] for i in ids]
            out[f"{arm}_MRR"] = round(float(np.mean(v)), 4)
            out[f"{arm}_CI"] = _boot(v)
            out[f"{arm}_minus_P"] = round(float(np.mean([rr[arm][i] - P[i] for i in ids])), 4)
            out[f"{arm}_minus_P_CI"] = _boot([rr[arm][i] - P[i] for i in ids])
        for arm in [a for a in ("rel", "rel_scaffold") if a in arms]:
            d = [rr[arm][i] - rr["naive"][i] for i in ids]
            out[f"{arm}_minus_naive"] = round(float(np.mean(d)), 4)
            out[f"{arm}_minus_naive_CI"] = _boot(d)
            # fraction of the phenotype->oracle headroom actually recovered.
            # NB: this reads as partial integration but is not -- see the selector
            # block below. Report it only alongside annotation_only_MRR.
            head = float(np.mean([oracle[i] - P[i] for i in ids]))
            got = float(np.mean([rr[arm][i] - P[i] for i in ids]))
            out[f"{arm}_headroom_recovered"] = round(got / head, 3) if abs(head) > 1e-9 else None

        # selector-vs-integrator: does the model fuse the two sources, or just pick one?
        # Paired per-item differences against each single-source oracle; exact-ranking
        # agreement; and how often it beats the better source it had available.
        out["selector"] = _selector_stats(ids, items, rr, P, A, oracle, suffix, arms)
        report["cells"][cell] = out

    out_path = OUT_DIR / f"metrics{suffix}.json"
    json.dump(report, open(out_path, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote -> {out_path}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--rescan", action="store_true")
    sub.add_parser("calibrate")
    r = sub.add_parser("run")
    r.add_argument("--model", default="claude-sonnet-4-6")
    r.add_argument("--arms", nargs="+", choices=ARMS,
                   help="restrict to these arms (default: all three).")
    r.add_argument("--provider", choices=["vertex", "azure"], default="vertex",
                   help="azure runs the second model family (gpt-5.6-luna) into "
                        "responses_gpt56.jsonl beside the vertex file.")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--retest", action="store_true",
                   help="re-run naive arm on the conflict cell for a noise floor")
    sub.add_parser("dedup")
    sub.add_parser("noise-floor")
    an = sub.add_parser("analyze")
    an.add_argument("--provider", choices=["vertex", "azure"], default="vertex",
                    help="azure scores model B's responses_gpt56.jsonl into metrics_gpt56.json.")
    a = ap.parse_args()

    if a.cmd == "build":
        if a.rescan and SCAN_CACHE.exists():
            SCAN_CACHE.unlink()
        build()
    elif a.cmd == "calibrate":
        calibrate()
    elif a.cmd == "run":
        run(a.model, limit=a.limit, smoke=a.smoke, retest=a.retest,
            provider=a.provider, arms=a.arms)
    elif a.cmd == "dedup":
        dedup()
    elif a.cmd == "noise-floor":
        noise_floor()
    elif a.cmd == "analyze":
        analyze(a.provider)


if __name__ == "__main__":
    main()
