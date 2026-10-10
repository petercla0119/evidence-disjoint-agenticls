#!/usr/bin/env python3
"""Reliability-calibration / evidence-weighting experiment.

Prior runs (validate_counterfactual.py) showed that in the phenotype-uninformative
tail the single-shot LLM *follows the misleading cosine off a cliff* -- real-but-
wrong phenotype is worse than shuffled. Question here: if we hand the model an
explicit, calibrated reliability prior for each evidence source, does it weigh the
evidence appropriately and stop over-trusting phenotype in the tail?

Design (small, ~30 conflict items -- not hundreds):

  * Conflict items = the existing phenotype-uninformative tail (gold phenotype
    rank 5-6): validate/ (rank>=5) UNION validate_booster/ (all). ~29 items.
  * Independent calibration set (fresh unconditioned draw, DISJOINT from both the
    tail items and the fair-60). From it, estimate per-source reliability
    conditional on rank bin:
        phenotype: P(a candidate is the true partner | its phenotype-cosine rank in {1-2,3-4,5-6})
        annotation: P(... | its annotation-graph rank in {1-2,3-4,5-6})   (graph score on the MASKED packet)
    plus a regime table (conditioned on the gold's phenotype-rank bin) giving each
    source's expected reciprocal rank -- used to define the correct answer for the
    comprehension control.
  * Two prompt variants on the SAME conflict items:
      without reliability = the existing H-mask responses (masked packet + phenotype)  [reused, not re-run]
      with reliability    = same packet + an injected source_reliability block         [new, 2 orders]
  * Comprehension control: ask the model, per rank regime, "which evidence source
    is more reliable in this regime?" and score against the calibration regime table.

  calibrate -> artifacts/evidence_eval/reliability/calibration.json         (offline)
  run       -> artifacts/evidence_eval/reliability/responses.jsonl          (~61 Vertex calls, resumable)
  analyze   -> metrics.json + reliability_with_vs_without.{png,pdf}

Usage:
  python scripts/evidence_eval/reliability_calibration.py calibrate [--seed 777] [--n 100]
  python scripts/evidence_eval/reliability_calibration.py run       [--model claude-sonnet-4-6] [--smoke]
  python scripts/evidence_eval/reliability_calibration.py analyze
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import pilot_counterfactual as pilot
import validate_counterfactual as validate
from pilot_counterfactual import (
    make_packet, SYSTEM, USER_TMPL, get_client, parse_response, _rr, _top1,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ART = BASE_DIR / "artifacts" / "evidence_eval"
OUT_DIR = ART / "reliability"

# tail dirs + how to pick the tail subset from each
CONFLICT_SOURCES = [("validate", lambda it: it["phen_rank"] >= 5),
                    ("validate_booster", lambda it: True)]
CALIB_DIR = "reliability_calib"
N_ORDERS = 2
BINS = [("1-2", (1, 2)), ("3-4", (3, 4)), ("5-6", (5, 6))]
# per-regime cap for the grounded comprehension control. Tail (5-6) = all 29
# conflict items (the regime that must dissociate); 1-2 / 3-4 are the
# discrimination controls proving the model isn't just always saying one source.
COMP_CAP = {"1-2": 20, "3-4": 13, "5-6": 999}


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def load_conflict_items():
    """The ~30 conflict items = phenotype-uninformative tail from prior runs."""
    items = []
    for d, keep in CONFLICT_SOURCES:
        src = json.load(open(ART / d / "items.json"))["items"]
        for it in src:
            if keep(it):
                it = dict(it)
                it["src_dir"] = d
                items.append(it)
    return items


def _bin(rank):
    for lab, (lo, hi) in BINS:
        if lo <= rank <= hi:
            return lab
    return None


def build_comprehension_set():
    """Item-grounded comprehension probes spanning phenotype-rank regimes.
    5-6 = the full conflict tail (dissociation); 1-2 / 3-4 = discrimination
    controls. Deterministic (item_id-sorted, capped). Truth comes from the
    calibration regime table's `more_reliable` field per regime."""
    per_regime = defaultdict(list)
    for d in ("validate", "validate_booster"):
        for it in json.load(open(ART / d / "items.json"))["items"]:
            lab = _bin(it["phen_rank"])
            if lab:
                it = dict(it)
                it["src_dir"] = d
                per_regime[lab].append(it)
    out = []
    for lab, _ in BINS:
        items = sorted(per_regime[lab], key=lambda it: it["item_id"])
        out.extend((it, lab) for it in items[:COMP_CAP.get(lab, len(items))])
    return out


def _ann_scores(packet):
    """Deterministic annotation-graph score per candidate on a masked packet.
    score(C) = direct(G0,C) + 0.5 * sum_j w(G0,Gj)*w(Gj,C)  (1-hop + 2-hop).
    w = functional_edge confidence + 0.5 per shared_pathway/shared_complex."""
    W = defaultdict(float)
    for e in packet["edges"]:
        a, b = e["genes"]
        key = frozenset((a, b))
        W[key] += float(e["confidence"]) if e["type"] == "functional_edge" else 0.5

    def w(a, b):
        return W.get(frozenset((a, b)), 0.0)

    cands, q = packet["candidates"], "G0"
    score = {}
    for c in cands:
        twohop = sum(w(q, o) * w(o, c) for o in cands if o != c)
        score[c] = w(q, c) + 0.5 * twohop
    return score


def _avg_rank(score, target):
    """Rank of target (1=best) with average-rank tie handling."""
    s = score[target]
    greater = sum(1 for v in score.values() if v > s)
    equal = sum(1 for v in score.values() if v == s)
    return greater + (equal + 1) / 2.0


def expected_rr(score, target):
    """Expected reciprocal rank of target under uniform random tie-breaking.

    Use this, never ``1 / _avg_rank(...)``. Reciprocal rank is convex, so
    ``1/E[rank] < E[1/rank]`` whenever the target is tied -- the two differ by
    enough to push a fully-tied ranker *below* chance, which is impossible.
    A gold tied across all six candidates has avg_rank 3.5, so ``1/avg_rank``
    scores it 0.286 while the true expectation is (1/6)*sum(1/j) = 0.4083,
    exactly the random-ranking rate.
    """
    s = score[target]
    greater = sum(1 for v in score.values() if v > s)
    equal = sum(1 for v in score.values() if v == s)
    # target is equally likely to land on any of ranks greater+1 .. greater+equal
    return sum(1.0 / j for j in range(greater + 1, greater + equal + 1)) / equal


def expected_top1(score, target):
    """P(target ranks first) under uniform random tie-breaking.

    The companion to expected_rr, and needed for the same reason. Scoring a
    k-way tie at the top as a miss charges a ranker for being undecided rather
    than wrong: a fully-tied 6-way ranker would score top-1 of 0 when the honest
    expectation is 1/6, the random rate. Returns 1/k for a target inside a k-way
    top tie, 0 otherwise, and exactly 1.0 for an untied winner.
    """
    s = score[target]
    if any(v > s for v in score.values()):
        return 0.0
    return 1.0 / sum(1 for v in score.values() if v == s)


def ann_rank_of_gold(item, ctx):
    packet, gold_alias, _ = make_packet(item, "A-mask", 0, ctx)
    return _avg_rank(_ann_scores(packet), gold_alias)


# ---------------------------------------------------------------------------
# calibrate: independent set -> per-source reliability by rank bin (offline)
# ---------------------------------------------------------------------------
def calibrate(seed=777, n=100):
    # build a fresh unconditioned calibration set, disjoint from tail + fair-60
    validate.OUT_DIR = ART / CALIB_DIR
    exclude = ",".join(d for d, _ in CONFLICT_SOURCES)  # validate,validate_booster
    validate.build_items(seed=seed, n=n, exclude_dir=exclude, prefix="k")

    items = json.load(open(ART / CALIB_DIR / "items.json"))["items"]
    ctx = pilot._ctx()
    N = len(items)

    # per-candidate reliability: P(candidate is gold | its rank == r), r=1..6.
    # exactly one candidate per rank per item -> denom = N per rank.
    phen_hits = defaultdict(int)   # phenotype rank of gold
    ann_hits = defaultdict(int)    # annotation rank of gold (rounded to nearest int)
    ann_rank = {}
    ann_argmax_hits = 0            # gold is the strict top annotation pick
    ann_uninformative = 0         # all 6 candidates tie -> graph gives no signal
    for it in items:
        phen_hits[it["phen_rank"]] += 1
        packet, gold_alias, _ = make_packet(it, "A-mask", 0, ctx)
        sc = _ann_scores(packet)
        ar = _avg_rank(sc, gold_alias)
        ann_rank[it["item_id"]] = ar
        ann_hits[int(round(ar))] += 1
        vals = list(sc.values())
        if max(vals) - min(vals) < 1e-9:
            ann_uninformative += 1
        elif max(vals) == sc[gold_alias] and sum(1 for v in vals if v == sc[gold_alias]) == 1:
            ann_argmax_hits += 1

    def binned(hits):
        out = {}
        for lab, (lo, hi) in BINS:
            k = sum(hits[r] for r in range(lo, hi + 1))
            denom = N * (hi - lo + 1)          # candidates in this bin across items
            out[lab] = round(k / denom, 4)
        return out

    phen_rel = binned(phen_hits)
    ann_rel = binned(ann_hits)

    random_rr = round(sum(1.0 / k for k in range(1, 7)) / 6, 4)  # E[RR] uniform ranking
    prank = {it["item_id"]: it["phen_rank"] for it in items}

    # regime table: condition on the gold's phenotype-rank bin; report each source's
    # expected reciprocal rank (how often/high it recovers the true partner there),
    # against the random-ranking baseline. "more_reliable" = best recovery in regime.
    regime = {}
    for lab, (lo, hi) in BINS:
        ids = [it["item_id"] for it in items if lo <= it["phen_rank"] <= hi]
        if not ids:
            regime[lab] = {"n": 0}
            continue
        pheno_rr = float(np.mean([1.0 / prank[i] for i in ids]))
        ann_rr = float(np.mean([1.0 / ann_rank[i] for i in ids]))
        best = max([("phenotype", pheno_rr), ("annotation", ann_rr),
                    ("neither (random)", random_rr)], key=lambda t: t[1])[0]
        regime[lab] = {
            "n": len(ids),
            "phenotype_recovery_RR": round(pheno_rr, 4),
            "annotation_recovery_RR": round(ann_rr, 4),
            "random_ranking_RR": random_rr,
            "more_reliable": "phenotype" if pheno_rr >= ann_rr else "annotation",
            "best_strategy": best,
        }

    cal = {
        "seed": seed, "n_calibration_items": N,
        "definition": "per-candidate P(candidate is the true partner | its rank for that source)",
        "phenotype_reliability_by_rank": phen_rel,
        "annotation_reliability_by_rank": ann_rel,
        "annotation_argmax_reliability": round(ann_argmax_hits / N, 4),
        "annotation_uninformative_fraction": round(ann_uninformative / N, 4),
        "random_candidate": round(1 / 6, 4),
        "random_ranking_RR": random_rr,
        "regime_table": regime,
        "phen_rank_hist": {r: phen_hits[r] for r in range(1, 7)},
        "ann_rank_hist": {r: ann_hits[r] for r in range(1, 7)},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(cal, open(OUT_DIR / "calibration.json", "w"), indent=2)
    print(json.dumps(cal, indent=2))
    return cal


# ---------------------------------------------------------------------------
# prompts (with-reliability arm + comprehension control)
# ---------------------------------------------------------------------------
SYSTEM_REL = SYSTEM + (
    " A source_reliability block gives calibrated hit-rates measured on an "
    "independent set: for each evidence source, the probability that a candidate "
    "at a given rank FOR THAT SOURCE is the true partner. To use it: rank the "
    "candidates by each source (phenotype_similarity; and the annotation graph of "
    "functional_edge/shared_pathway/shared_complex), then trust each candidate in "
    "proportion to the reliability of its rank. Discount a source where a "
    "candidate's rank has low reliability (near or below the random rate)."
)


def _rel_block(cal):
    """Injected calibration. Phenotype: honest per-rank curve (the requested
    P(gold | R_phenotype=bin)). Annotation: an honest scalar -- the masked graph
    is non-discriminative for most items, so a per-rank curve would be a tie
    artifact; report its argmax hit-rate + how often it gives no signal."""
    return {
        "measured_on": f"independent calibration set, N={cal['n_calibration_items']} (disjoint from these items)",
        "phenotype_similarity_rank_reliability": {
            "definition": "P(a candidate is the true partner | its phenotype-similarity rank bin)",
            **cal["phenotype_reliability_by_rank"],
        },
        "annotation_graph_reliability": {
            "top_pick_is_true_partner": cal["annotation_argmax_reliability"],
            "fraction_of_items_with_no_discriminating_annotation": cal["annotation_uninformative_fraction"],
            "note": "after masking, the surviving annotation graph usually cannot separate candidates; treat it as weak",
        },
        "random_candidate": cal["random_candidate"],
    }


COMP_SYSTEM = (
    "You are given a calibration table measured on an independent set: for each "
    "phenotype-rank regime (where the TRUE partner falls in phenotype similarity), "
    "how often each evidence source recovers the true partner. Answer from the "
    "table only."
)
COMP_USER = (
    "Calibration regime table (independent set):\n{table}\n\n"
    "In the regime where the true functional partner's phenotype-similarity rank "
    "is {regime} (among 6 candidates), which single evidence source is more "
    "reliable for identifying it?\n"
    'Return ONLY JSON: {{"more_reliable_source": "phenotype" | "annotation", '
    '"reason": "<one sentence>"}}'
)

# item-grounded comprehension probe: same question, but per real item (its own
# phenotype scores shown, its regime stated) so the answer is a rate over ~60
# items with a CI + a per-regime discrimination breakdown, not a single 3/3.
COMP_ITEM_USER = (
    "Calibration regime table (independent set):\n{table}\n\n"
    "This item's phenotype_similarity scores for the six candidates:\n{sims}\n\n"
    "For THIS item, the true functional partner's phenotype-similarity rank falls "
    "in the {regime} regime (among 6 candidates). Given the calibration, which "
    "single evidence source is more reliable for identifying the true partner "
    "here?\n"
    'Return ONLY JSON: {{"more_reliable_source": "phenotype" | "annotation", '
    '"reason": "<one sentence>"}}'
)


# ---------------------------------------------------------------------------
# run: with-reliability arm (2 orders) + comprehension probe (per regime)
# ---------------------------------------------------------------------------
def run(model, limit=None, smoke=False):
    cal = json.load(open(OUT_DIR / "calibration.json"))
    rel_block = _rel_block(cal)
    items = load_conflict_items()
    ctx = pilot._ctx()
    client = get_client()

    out_path = OUT_DIR / "responses.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["kind"], r.get("item_id", r.get("regime")), r.get("order", 0)))

    comp_set = build_comprehension_set()

    # jobs: (kind, item-or-regime, order)
    jobs = [("with_rel", it, o) for it in items for o in range(N_ORDERS)]
    jobs += [("comprehension", lab, 0) for lab, _ in BINS]          # legacy 3-probe
    jobs += [("comp_item", it, 0) for it, _ in comp_set]            # grounded rate
    if smoke:
        jobs = jobs[:1] + [("comp_item", comp_set[-1][0], 0)]
    if limit:
        jobs = jobs[:limit]

    def key(kind, obj, order):
        obj_key = obj["item_id"] if kind in ("with_rel", "comp_item") else obj
        return (kind, obj_key, order)

    jobs = [j for j in jobs if key(*j) not in done]
    print(f"{len(jobs)} calls to make ({len(done)} already done) on {model}")

    def call(system, user):
        msg = client.messages.create(
            model=model, max_tokens=1024, system=system,
            messages=[{"role": "user", "content": user}],
            extra_body={"temperature": 0.0},
        )
        text = msg.content[0].text
        try:
            return parse_response(text)
        except Exception as e:
            return {"_parse_error": str(e), "_raw": text[:500]}

    f = open(out_path, "a")
    for n, (kind, obj, order) in enumerate(jobs, 1):
        if kind == "with_rel":
            packet, gold_alias, audit = make_packet(obj, "H-mask", order, ctx)
            packet["source_reliability"] = rel_block
            parsed = call(SYSTEM_REL, USER_TMPL.format(packet=json.dumps(packet)))
            rec = dict(kind=kind, item_id=obj["item_id"], src_dir=obj["src_dir"],
                       phen_rank=obj["phen_rank"], order=order, gold_alias=gold_alias,
                       model=model, audit=audit, response=parsed)
            rk = parsed.get("ranking", []) if isinstance(parsed, dict) else []
            pos = (rk.index(gold_alias) + 1) if gold_alias in rk else "-"
            print(f"[{n}/{len(jobs)}] with_rel {obj['item_id']:6s} o{order} gold={gold_alias} pos={pos}")
        elif kind == "comp_item":
            lab = _bin(obj["phen_rank"])
            packet, _, _ = make_packet(obj, "H-mask", 0, ctx)
            sims = packet.get("phenotype_similarity", {})
            table = json.dumps(cal["regime_table"], indent=2)
            parsed = call(COMP_SYSTEM, COMP_ITEM_USER.format(
                table=table, sims=json.dumps(sims), regime=lab))
            truth = cal["regime_table"].get(lab, {}).get("more_reliable")
            ans = parsed.get("more_reliable_source") if isinstance(parsed, dict) else None
            rec = dict(kind=kind, item_id=obj["item_id"], src_dir=obj.get("src_dir", ""),
                       phen_rank=obj["phen_rank"], regime=lab, truth=truth,
                       order=0, model=model, response=parsed)
            print(f"[{n}/{len(jobs)}] comp_item {obj['item_id']:6s} regime={lab} -> {ans} (truth {truth})")
        else:  # comprehension (legacy per-regime probe)
            table = json.dumps(cal["regime_table"], indent=2)
            parsed = call(COMP_SYSTEM, COMP_USER.format(table=table, regime=obj))
            rec = dict(kind=kind, regime=obj, order=0, model=model, response=parsed)
            ans = parsed.get("more_reliable_source") if isinstance(parsed, dict) else None
            print(f"[{n}/{len(jobs)}] comprehension regime={obj} -> {ans}")
        f.write(json.dumps(rec) + "\n")
        f.flush()
    f.close()
    print(f"Wrote -> {out_path}")


# ---------------------------------------------------------------------------
# analyze: with vs without reliability on the tail; comprehension accuracy
# ---------------------------------------------------------------------------
def _load_without():
    """Existing H-mask responses on the conflict items, keyed item_id -> [recs]."""
    out = defaultdict(list)
    conflict_ids = {(it["src_dir"], it["item_id"]) for it in load_conflict_items()}
    for d, _ in CONFLICT_SOURCES:
        for line in open(ART / d / "responses.jsonl"):
            r = json.loads(line)
            if r["condition"] == "H-mask" and (d, r["item_id"]) in conflict_ids:
                out[r["item_id"]].append(r)
    return out


def analyze():
    cal = json.load(open(OUT_DIR / "calibration.json"))
    recs = [json.loads(l) for l in open(OUT_DIR / "responses.jsonl")]
    with_rel = defaultdict(list)
    comp = {}                 # legacy per-regime probe
    comp_items = []           # grounded per-item probes
    for r in recs:
        if r["kind"] == "with_rel":
            with_rel[r["item_id"]].append(r)
        elif r["kind"] == "comp_item":
            comp_items.append(r)
        elif r["kind"] == "comprehension":
            comp[r["regime"]] = r
    without = _load_without()
    items = load_conflict_items()
    phen_rank = {it["item_id"]: it["phen_rank"] for it in items}

    ids = sorted(set(with_rel) & set(without))

    def item_metric(recs_by_id, metric):
        return {i: float(np.mean([metric(r) for r in recs_by_id[i]])) for i in ids}

    rr_w = item_metric(with_rel, _rr)
    rr_wo = item_metric(without, _rr)
    t1_w = item_metric(with_rel, _top1)
    t1_wo = item_metric(without, _top1)

    def boot(vals, n=2000, seed=0):
        vals = np.asarray(vals, float)
        if len(vals) == 0:
            return (float("nan"),) * 3
        rng = np.random.default_rng(seed)
        bs = vals[rng.integers(0, len(vals), size=(n, len(vals)))].mean(axis=1)
        return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    P_rr = {i: 1.0 / phen_rank[i] for i in ids}
    delta = [rr_w[i] - rr_wo[i] for i in ids]
    m, lo, hi = boot(delta)

    # comprehension scoring against the regime table's ground truth
    comp_report = {}
    for lab, _ in BINS:
        truth = cal["regime_table"].get(lab, {}).get("more_reliable")
        got = None
        if lab in comp and isinstance(comp[lab]["response"], dict):
            got = comp[lab]["response"].get("more_reliable_source")
        comp_report[lab] = {"truth": truth, "model": got, "correct": (got == truth)}

    # grounded comprehension rate (per-item, spans regimes) + CI + per-regime
    def _comp_correct(r):
        got = r["response"].get("more_reliable_source") if isinstance(r["response"], dict) else None
        return 1.0 if got == r["truth"] else 0.0

    def _cp_ci(k, nn):
        """Clopper-Pearson exact 95% interval — correct for a binomial rate
        (bootstrap of an all-correct vector degenerates to [1,1])."""
        from scipy.stats import beta
        lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, nn - k + 1))
        hi = 1.0 if k == nn else float(beta.ppf(0.975, k + 1, nn - k))
        return round(lo, 4), round(hi, 4)

    comp_grounded = None
    if comp_items:
        k_all = int(sum(_comp_correct(r) for r in comp_items))
        n_all = len(comp_items)
        by_regime = {}
        for lab, _ in BINS:
            grp = [r for r in comp_items if r["regime"] == lab]
            if grp:
                k = int(sum(_comp_correct(r) for r in grp))
                by_regime[lab] = {
                    "n": len(grp),
                    "truth": grp[0]["truth"],
                    "n_correct": k,
                    "accuracy": round(k / len(grp), 4),
                    "ci95_clopper_pearson": list(_cp_ci(k, len(grp))),
                }
        comp_grounded = {
            "n": n_all,
            "n_correct": k_all,
            "accuracy": round(k_all / n_all, 4),
            "ci95_clopper_pearson": list(_cp_ci(k_all, n_all)),
            "by_regime": by_regime,
            "n_parse_errors": sum(1 for r in comp_items if "_parse_error" in r["response"]),
        }

    report = {
        "model": recs[0]["model"], "n_conflict_items": len(ids),
        "calibration": {
            "phenotype_reliability_by_rank": cal["phenotype_reliability_by_rank"],
            "annotation_reliability_by_rank": cal["annotation_reliability_by_rank"],
            "regime_table": cal["regime_table"],
        },
        "with_vs_without": {
            "MRR_without_reliability": round(float(np.mean(list(rr_wo.values()))), 4),
            "MRR_with_reliability": round(float(np.mean(list(rr_w.values()))), 4),
            "top1_without": round(float(np.mean(list(t1_wo.values()))), 4),
            "top1_with": round(float(np.mean(list(t1_w.values()))), 4),
            "MRR_phenotype_oracle_P": round(float(np.mean(list(P_rr.values()))), 4),
            "delta_MRR_mean": round(m, 4), "delta_ci95": [round(lo, 4), round(hi, 4)],
            "n_helped": int(sum(1 for i in ids if rr_w[i] > rr_wo[i] + 1e-9)),
            "n_tie": int(sum(1 for i in ids if abs(rr_w[i] - rr_wo[i]) <= 1e-9)),
            "n_hurt": int(sum(1 for i in ids if rr_w[i] < rr_wo[i] - 1e-9)),
        },
        "comprehension_control": comp_report,
        "comprehension_control_grounded": comp_grounded,
        "dissociation": {
            "note": "same 29 tail (rank 5-6) conflict items: comprehension is a rate, action is null",
            "tail_action_delta_MRR": round(m, 4),
            "tail_action_delta_ci95": [round(lo, 4), round(hi, 4)],
            "tail_comprehension_accuracy": (comp_grounded or {}).get("by_regime", {}).get("5-6", {}).get("accuracy"),
            "tail_comprehension_ci95": (comp_grounded or {}).get("by_regime", {}).get("5-6", {}).get("ci95_clopper_pearson"),
        } if comp_grounded else None,
        "sanity": {
            "n_with_rel_calls": sum(len(v) for v in with_rel.values()),
            "with_rel_parse_errors": sum(
                1 for v in with_rel.values() for r in v if "_parse_error" in r["response"]),
        },
    }
    json.dump(report, open(OUT_DIR / "metrics.json", "w"), indent=2)
    print(json.dumps(report, indent=2))
    _figure(ids, rr_wo, rr_w, P_rr, report)


def _figure(ids, rr_wo, rr_w, P_rr, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["A-mask\n(no pheno)*", "H-mask\nwithout reliability",
              "H-mask\nwith reliability", "phenotype\noracle P"]
    # A-mask baseline pulled from validate metrics for context (rank5-6 stratum)
    amask = None
    try:
        vm = json.load(open(ART / "validate" / "metrics.json"))
        amask = vm["conditions"]["rank5-6"]["A-mask"]
    except Exception:
        pass
    ww = report["with_vs_without"]
    ys = [amask, ww["MRR_without_reliability"], ww["MRR_with_reliability"],
          ww["MRR_phenotype_oracle_P"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = range(len(labels))
    colors = ["#999999", "#b2182b", "#2166ac", "#4d4d4d"]
    ax.bar([x for x, y in zip(xs, ys) if y is not None],
           [y for y in ys if y is not None],
           color=[c for c, y in zip(colors, ys) if y is not None], alpha=0.85)
    ax.axhline(1 / 6, ls=":", color="gray", lw=1, label="random (1/6)")
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MRR (gold partner)"); ax.set_ylim(0, max(y for y in ys if y) * 1.25)
    ax.set_title("Does an explicit reliability prior fix the tail?\n"
                 f"(phenotype-uninformative conflict items, n={report['n_conflict_items']})")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    d = report["with_vs_without"]
    ax.annotate(f"Δ = {d['delta_MRR_mean']:+.3f} [{d['delta_ci95'][0]:+.2f}, {d['delta_ci95'][1]:+.2f}]",
                xy=(2, ww["MRR_with_reliability"]), xytext=(1.3, max(ys[1:]) * 1.1), fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"reliability_with_vs_without.{ext}", dpi=200)
    print(f"Figure -> {OUT_DIR/'reliability_with_vs_without.png'}  (*A-mask from validate rank5-6)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--seed", type=int, default=777)
    c.add_argument("--n", type=int, default=100)
    r = sub.add_parser("run")
    r.add_argument("--model", default="claude-sonnet-4-6")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--smoke", action="store_true")
    sub.add_parser("analyze")
    args = ap.parse_args()

    if args.cmd == "calibrate":
        calibrate(args.seed, args.n)
    elif args.cmd == "run":
        run(args.model, args.limit, args.smoke)
    elif args.cmd == "analyze":
        analyze()


if __name__ == "__main__":
    main()
