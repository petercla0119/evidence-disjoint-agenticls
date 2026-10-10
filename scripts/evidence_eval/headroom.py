#!/usr/bin/env python
"""Three measurement corrections for the AgenticLS extended abstract.

1. Exceedance on the CORRECT denominator. "Beat the better of two sources" is
   arithmetically impossible whenever either source already puts the gold first
   (that source's RR is 1.0, the ceiling). Those items must leave the denominator.
2. Top-1 accuracy alongside MRR. Chance MRR on 6 candidates is 0.408, so every
   effect lives in a 0.408-1.0 band; top-1 has chance 0.167 and ~6x the range.
3. The model-B exact-permutation statistic, against defensible nulls rather than
   a uniform-over-720 strawman.

Reuses arbitration._boot (10k paired percentile bootstrap, seed 0) everywhere so
MRR and top-1 come from one procedure. Recomputed MRRs are cross-checked against
the published metrics.json values and the script fails loudly on drift.

Usage:
    PYTHONHASHSEED=0 python scripts/evidence_eval/headroom.py
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arbitration as A                       # noqa: E402
import pilot_counterfactual as pilot          # noqa: E402
import reliability_calibration as rc          # noqa: E402

if os.environ.get("PYTHONHASHSEED") != "0":
    raise SystemExit("run as: PYTHONHASHSEED=0 python scripts/evidence_eval/headroom.py")

ART = A.ART
OUT = ART / "headroom"
N_ORDERS = 2
EPS = 1e-9

# arms in the "300 item-arm cells" claim, as (label, responses path, arm key)
ARMS = [
    ("naive", ART / "arbitration" / "responses.jsonl", "naive"),
    ("rel", ART / "arbitration" / "responses.jsonl", "rel"),
    ("rel_scaffold", ART / "arbitration" / "responses.jsonl", "rel_scaffold"),
    ("agent_A", ART / "tool_agent" / "responses.jsonl", "agent_free"),
    ("agent_B", ART / "tool_agent_gpt56" / "responses.jsonl", "agent_free"),
]


def rnd(x, n=4):
    return None if x is None else round(float(x), n)


def boot(vals):
    lo, hi = A._boot(list(vals))
    return [rnd(lo), rnd(hi)]


def stat(vals):
    vals = list(vals)
    return {"mean": rnd(np.mean(vals)), "ci95": boot(vals), "n": len(vals)}


# ---------------------------------------------------------------------------
# sources: tie-aware expected RR / top-1 for the two deterministic rankers
# ---------------------------------------------------------------------------
def tie_aware(scores, gold):
    """Expected RR and top-1 of `gold` when equal scores are broken at random.

    items.json stores ann_rank as an AVERAGE rank over the tie group, and the
    published MRR uses 1/avg_rank. That is not E[RR]: for a 6-way tie it gives
    1/3.5 = 0.286, while a coin-flipping ranker actually scores H_6/6 = 0.408
    (= chance). The gap matters in the control cell, where 19/20 items are 6-way
    annotation ties, so we report both.
    """
    order = sorted(scores, key=lambda c: -scores[c])
    better = sum(1 for c in order if scores[c] > scores[gold] + EPS)
    k = sum(1 for c in order if abs(scores[c] - scores[gold]) <= EPS)
    ranks = range(better + 1, better + k + 1)
    return (float(np.mean([1.0 / r for r in ranks])),
            (1.0 / k) if better == 0 else 0.0)


def load_arbitration_sources():
    """Per item: phenotype-only and annotation-only RR/top-1, both conventions."""
    items = {it["item_id"]: it
             for it in json.load(open(ART / "arbitration" / "items.json"))["items"]}
    ctx = pilot._ctx()
    src = {}
    n_round_tie = 0
    for iid, it in items.items():
        packet, gold, _ = pilot.make_packet(it, "H-mask", 0, ctx)
        ph = packet["phenotype_similarity"]
        an = rc._ann_scores(packet)
        a_rr_tie, a_t1 = tie_aware(an, gold)
        # The phenotype-only ranker is defined on the RAW cosine (that is what
        # items.json phen_rank is and what every published MRR uses); the packet
        # shown to the model rounds to 4dp, which creates a gold-involving tie on
        # 2/60 items. We score the ranker on the raw cosine, and count those.
        p_rr, p_t1 = 1.0 / it["phen_rank"], (1.0 if it["phen_rank"] == 1 else 0.0)
        if sum(1 for c in ph if abs(ph[c] - ph[gold]) <= EPS) > 1:
            n_round_tie += 1
        a_rr = 1.0 / it["ann_rank"]
        src[iid] = {
            "cell": it["cell"], "phen_rank": it["phen_rank"],
            "ann_rank": it["ann_rank"], "gap": it["gap"], "margin": it["margin"],
            "P_rr": p_rr, "A_rr": a_rr, "A_rr_tieaware": a_rr_tie,
            "P_top1": p_t1, "A_top1": a_t1,
            "best_rr": max(p_rr, a_rr), "best_top1": max(p_t1, a_t1),
            "n_ann_tied_at_top": sum(1 for c in an
                                     if abs(an[c] - max(an.values())) <= EPS),
        }
    return items, src, ctx, n_round_tie


def per_item(path, arm_key, metric):
    """{(item_id): mean over orders} for one arm of one responses file."""
    acc = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if (r.get("arm") or r.get("condition")) != arm_key:
            continue
        acc[r["item_id"]].append(metric(r))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def rr_of(r):
    rk = r["response"].get("ranking", []) if isinstance(r["response"], dict) else []
    return 1.0 / (rk.index(r["gold_alias"]) + 1) if r["gold_alias"] in rk else 0.0


def t1_of(r):
    rk = r["response"].get("ranking", []) if isinstance(r["response"], dict) else []
    return 1.0 if rk and rk[0] == r["gold_alias"] else 0.0


# ===========================================================================
# TASK 1 -- exceedance on the corrected denominator
# ===========================================================================
def task1(src):
    out = {"definition":
           "An item is EXCEEDABLE only if neither deterministic source already "
           "ranks the gold first. If either does, that source scores RR=1.0 and "
           "top-1=1.0, the ceiling, so 'beats the better of its two sources' is "
           "arithmetically impossible there and the item carries no information "
           "about integration.",
           "cells": {}}

    for cell in ("conflict", "control"):
        ids = sorted(i for i, s in src.items() if s["cell"] == cell)
        exc = [i for i in ids if src[i]["best_rr"] < 1.0 - EPS]
        une = [i for i in ids if i not in exc]
        out["cells"][cell] = {
            "n_items": len(ids),
            "n_unexceedable": len(une),
            "n_exceedable": len(exc),
            "unexceedable_because_phen_rank_1": sum(1 for i in ids if src[i]["phen_rank"] == 1),
            "unexceedable_because_ann_rank_1": sum(1 for i in ids if src[i]["ann_rank"] == 1),
            "unexceedable_because_both": sum(1 for i in ids
                                             if src[i]["phen_rank"] == 1 and src[i]["ann_rank"] == 1),
            "exceedable_item_ids": exc,
            "arms": {},
        }

    exceedable = {c: set(out["cells"][c]["exceedable_item_ids"]) for c in out["cells"]}

    for label, path, key in ARMS:
        rr = per_item(path, key, rr_of)
        # per-call too: averaging 2 orders can hide a single call that exceeded
        calls = defaultdict(list)
        for line in open(path):
            r = json.loads(line)
            if (r.get("arm") or r.get("condition")) == key:
                calls[r["item_id"]].append(rr_of(r))
        for cell in ("conflict", "control"):
            ids = sorted(i for i in exceedable[cell] if i in rr)
            beat = [i for i in ids if rr[i] > src[i]["best_rr"] + EPS]
            tie = [i for i in ids if abs(rr[i] - src[i]["best_rr"]) <= EPS]
            lose = [i for i in ids if rr[i] < src[i]["best_rr"] - EPS]
            ncall = sum(len(calls[i]) for i in ids)
            cbeat = sum(1 for i in ids for v in calls[i] if v > src[i]["best_rr"] + EPS)
            out["cells"][cell]["arms"][label] = {
                "n_exceedable_items_scored": len(ids),
                "beat": len(beat), "tie": len(tie), "lose": len(lose),
                "beat_item_ids": beat,
                "n_exceedable_calls": ncall,
                "calls_that_beat": cbeat,
                "mean_rr_minus_best_source": rnd(np.mean(
                    [rr[i] - src[i]["best_rr"] for i in ids])) if ids else None,
            }

    nA = sum(out["cells"][c]["n_exceedable"] for c in out["cells"])
    out["corrected_denominator"] = {
        "exceedable_items_total": nA,
        "n_arms": len(ARMS),
        "exceedable_item_arm_cells": nA * len(ARMS),
        "stated_in_paper": 300,
        "uninformative_cells_in_stated_denominator": (60 - nA) * len(ARMS),
        "conflict_only_item_arm_cells": out["cells"]["conflict"]["n_exceedable"] * len(ARMS),
    }
    tot_beat = sum(out["cells"][c]["arms"][l]["beat"]
                   for c in out["cells"] for l, _, _ in ARMS)
    conf_beat = sum(out["cells"]["conflict"]["arms"][l]["beat"] for l, _, _ in ARMS)
    out["corrected_denominator"]["total_beats"] = tot_beat
    out["corrected_denominator"]["conflict_beats"] = conf_beat

    # --- what the control cell can and cannot test -------------------------
    ctrl = [i for i, s in src.items() if s["cell"] == "control"]
    out["control_cell_limitation"] = {
        "n_items": len(ctrl),
        "ann_rank_is_3.5": sum(1 for i in ctrl if abs(src[i]["ann_rank"] - 3.5) < 1e-9),
        "ann_rank_distribution": {str(k): v for k, v in
                                  sorted(Counter(src[i]["ann_rank"] for i in ctrl).items())},
        "margin_is_zero": sum(1 for i in ctrl if src[i]["margin"] <= EPS),
        "items_with_all_6_annotation_scores_tied":
            sum(1 for i in ctrl if src[i]["n_ann_tied_at_top"] == 6),
        "annotation_only_MRR_published_1_over_avgrank":
            rnd(np.mean([src[i]["A_rr"] for i in ctrl])),
        "annotation_only_MRR_tieaware_expected":
            rnd(np.mean([src[i]["A_rr_tieaware"] for i in ctrl])),
        "chance_MRR": rnd(A.RANDOM_RR),
        "annotation_only_top1_tieaware": rnd(np.mean([src[i]["A_top1"] for i in ctrl])),
        "reading": (
            "The control cell is defined by margin<=0, i.e. annotation is "
            "non-discriminating BY CONSTRUCTION: 19/20 items have all six "
            "annotation scores tied (ann_rank 3.5), the 20th has the gold below "
            "a tie group. There is therefore no second signal to integrate. The "
            "cell tests NON-DISTRACTION (does the model abandon a good source "
            "when a useless one is named reliable?) -- it cannot test "
            "integration, and its 100 item-arm cells should not enter an "
            "integration denominator. A further artifact: scoring the "
            "annotation-only ranker as 1/avg_rank gives MRR 0.281, BELOW chance "
            "0.408, when a random tie-break scores exactly chance. The reported "
            "'+0.42 over annotation-only' in control is ~0.29 once ties are "
            "handled as ties."),
    }
    return out


# ===========================================================================
# TASK 2 -- top-1 accuracy alongside MRR
# ===========================================================================
def _agree(mrr_d, t1_d):
    """Do the two metrics tell the same story? (sign and CI-excludes-zero)"""
    def sig(d):
        return d["ci95"][0] > 0 or d["ci95"][1] < 0
    sm = 0 if abs(mrr_d["mean"]) < 1e-9 else math.copysign(1, mrr_d["mean"])
    st = 0 if abs(t1_d["mean"]) < 1e-9 else math.copysign(1, t1_d["mean"])
    sign_ok = sm == st
    sig_ok = sig(mrr_d) == sig(t1_d)
    # A sign flip between +0.004 and 0.000 is not a disagreement, it is two ways
    # of writing "null". Only flag it when the metrics could change a sentence:
    # different significance, or a sign flip where at least one side is material.
    material = max(abs(mrr_d["mean"]), abs(t1_d["mean"])) >= 0.02
    return {"sign_agree": sign_ok, "significance_agree": sig_ok,
            "mrr_significant": sig(mrr_d), "top1_significant": sig(t1_d),
            "both_null": not sig(mrr_d) and not sig(t1_d),
            "disagrees": (not sig_ok) or (not sign_ok and material)}


def task2_tables(src):
    """tab:main and tab:selector, in MRR and top-1, with paired bootstrap CIs."""
    out = {"chance": {"MRR": rnd(A.RANDOM_RR), "top1": rnd(1 / 6)},
           "dynamic_range": {"MRR": rnd(1 - A.RANDOM_RR), "top1": rnd(1 - 1 / 6),
                             "ratio": rnd((1 - 1 / 6) / (1 - A.RANDOM_RR), 2)},
           "tab_main": {}, "tab_selector": {}, "disagreements": []}

    arm_rr = {l: per_item(p, k, rr_of) for l, p, k in ARMS}
    arm_t1 = {l: per_item(p, k, t1_of) for l, p, k in ARMS}

    for cell in ("conflict", "control"):
        ids = sorted(i for i, s in src.items() if s["cell"] == cell)
        blk = {"n_items": len(ids), "baselines": {}, "arms": {}}
        blk["baselines"] = {
            "phenotype_only": {"MRR": stat([src[i]["P_rr"] for i in ids]),
                               "top1": stat([src[i]["P_top1"] for i in ids])},
            "annotation_only": {"MRR": stat([src[i]["A_rr"] for i in ids]),
                                "MRR_tieaware": stat([src[i]["A_rr_tieaware"] for i in ids]),
                                "top1": stat([src[i]["A_top1"] for i in ids])},
            "best_source_oracle": {"MRR": stat([src[i]["best_rr"] for i in ids]),
                                   "top1": stat([src[i]["best_top1"] for i in ids])},
        }
        for label, _, _ in ARMS:
            have = [i for i in ids if i in arm_rr[label]]
            m = stat([arm_rr[label][i] for i in have])
            t = stat([arm_t1[label][i] for i in have])
            blk["arms"][label] = {"MRR": m, "top1": t}
            # tab:selector -- paired differences against each single source
            sel = {}
            for src_lab, rk, tk in (("phenotype_only", "P_rr", "P_top1"),
                                    ("annotation_only", "A_rr", "A_top1")):
                dm = stat([arm_rr[label][i] - src[i][rk] for i in have])
                dt = stat([arm_t1[label][i] - src[i][tk] for i in have])
                ag = _agree(dm, dt)
                sel[f"minus_{src_lab}"] = {"MRR": dm, "top1": dt, "agreement": ag}
                if ag["disagrees"]:
                    out["disagreements"].append(
                        {"where": f"tab:selector {cell}/{label} minus {src_lab}",
                         "MRR": dm, "top1": dt, **ag})
            dm = stat([arm_rr[label][i] - src[i]["A_rr_tieaware"] for i in have])
            sel["minus_annotation_only_tieaware"] = {"MRR": dm}
            out["tab_selector"].setdefault(cell, {})[label] = sel
        out["tab_main"][cell] = blk

        # arm-vs-arm contrasts the prose cites: calibration gain, tool equivalence
        for a, b in (("rel", "naive"), ("rel_scaffold", "naive"),
                     ("agent_A", "naive"), ("agent_B", "naive"),
                     ("agent_A", "rel"), ("agent_B", "rel"),
                     ("rel_scaffold", "rel")):
            have = [i for i in ids if i in arm_rr[a] and i in arm_rr[b]]
            dm = stat([arm_rr[a][i] - arm_rr[b][i] for i in have])
            dt = stat([arm_t1[a][i] - arm_t1[b][i] for i in have])
            ag = _agree(dm, dt)
            out.setdefault("arm_contrasts", {}).setdefault(cell, {})[f"{a}_minus_{b}"] = \
                {"MRR": dm, "top1": dt, "agreement": ag}
            if ag["disagrees"]:
                out["disagreements"].append(
                    {"where": f"{cell}: {a} - {b}", "MRR": dm, "top1": dt, **ag})
    return out


def _cond_metrics(path, items_meta, ids, conds):
    """MRR + top-1 per item per condition for a validate/scaffold-style file."""
    rr, t1 = {}, {}
    for c in conds:
        rr[c] = per_item(path, c, rr_of)
        t1[c] = per_item(path, c, t1_of)
    P_rr = {i: 1.0 / items_meta[i] for i in ids}
    P_t1 = {i: 1.0 if items_meta[i] == 1 else 0.0 for i in ids}
    return rr, t1, P_rr, P_t1


def task2_uncalibrated(out):
    """The unconditioned 60-item draw, the 29-item tail, the scaffold arm."""
    res = {}
    V = ART / "validate"
    B = ART / "validate_booster"
    v_items = {it["item_id"]: it["phen_rank"]
               for it in json.load(open(V / "items.json"))["items"]}
    b_items = {it["item_id"]: it["phen_rank"]
               for it in json.load(open(B / "items.json"))["items"]}
    conds = ["A-mask", "H-mask", "H-shuffle"]
    v_rr, v_t1, vP_rr, vP_t1 = _cond_metrics(V / "responses.jsonl", v_items,
                                             list(v_items), conds)
    b_rr, b_t1, bP_rr, bP_t1 = _cond_metrics(B / "responses.jsonl", b_items,
                                             list(b_items), conds)

    def contrast(name, rr, t1, ids, a, b):
        dm = stat([rr[a][i] - rr[b][i] for i in ids])
        dt = stat([t1[a][i] - t1[b][i] for i in ids])
        ag = _agree(dm, dt)
        if ag["disagrees"]:
            out["disagreements"].append({"where": name, "MRR": dm, "top1": dt, **ag})
        return {"MRR": dm, "top1": dt, "agreement": ag}

    ids60 = sorted(v_items)
    res["fair60"] = {
        "n": len(ids60),
        "levels": {c: {"MRR": stat([v_rr[c][i] for i in ids60]),
                       "top1": stat([v_t1[c][i] for i in ids60])} for c in conds},
        "phenotype_only": {"MRR": stat([vP_rr[i] for i in ids60]),
                           "top1": stat([vP_t1[i] for i in ids60])},
        "rescue_Hmask_minus_Amask": contrast(
            "fair60 rescue (H-mask - A-mask)", v_rr, v_t1, ids60, "H-mask", "A-mask"),
        "shuffle_control_Hshuffle_minus_Amask": contrast(
            "fair60 shuffle control (H-shuffle - A-mask)", v_rr, v_t1, ids60,
            "H-shuffle", "A-mask"),
    }

    # 29-item tail = fair-60 phen_rank>=5 (n=5) U booster-24
    tail_v = [i for i in ids60 if v_items[i] >= 5]
    tail_b = sorted(b_items)
    trr = {c: {**{i: v_rr[c][i] for i in tail_v}, **{i: b_rr[c][i] for i in tail_b}}
           for c in conds}
    tt1 = {c: {**{i: v_t1[c][i] for i in tail_v}, **{i: b_t1[c][i] for i in tail_b}}
           for c in conds}
    tail = tail_v + tail_b
    res["tail29"] = {
        "n": len(tail), "n_from_fair60": len(tail_v), "n_from_booster": len(tail_b),
        "levels": {c: {"MRR": stat([trr[c][i] for i in tail]),
                       "top1": stat([tt1[c][i] for i in tail])} for c in conds},
        "misinformation_cost_Hmask_minus_Hshuffle": contrast(
            "tail29 misinformation cost (H-mask - H-shuffle)", trr, tt1, tail,
            "H-mask", "H-shuffle"),
        "rescue_Hmask_minus_Amask": contrast(
            "tail29 rescue (H-mask - A-mask)", trr, tt1, tail, "H-mask", "A-mask"),
    }

    # scaffold WITHOUT the table -- exp7, its own within-run A-mask/H-mask
    sc = {}
    for setname, folder, meta in (("fair60", "fair", v_items),
                                  ("booster24", "booster", b_items)):
        p = ART / "scaffold_only" / folder / "responses.jsonl"
        sconds = ["A-mask", "H-mask", "H-scaffold"]
        ids = sorted({json.loads(l)["item_id"] for l in open(p)})
        rr, t1, P_rr, P_t1 = _cond_metrics(p, meta, ids, sconds)
        rr["P"], t1["P"] = P_rr, P_t1
        sc[setname] = {
            "n": len(ids),
            "levels": {c: {"MRR": stat([rr[c][i] for i in ids]),
                           "top1": stat([t1[c][i] for i in ids])}
                       for c in sconds + ["P"]},
            "scaffold_minus_Hmask": contrast(
                f"scaffold {setname}: H-scaffold - H-mask", rr, t1, ids,
                "H-scaffold", "H-mask"),
            "scaffold_minus_P": contrast(
                f"scaffold {setname}: H-scaffold - P", rr, t1, ids,
                "H-scaffold", "P"),
        }
    res["scaffold_without_table"] = sc
    return res


# ===========================================================================
# TASK 3 -- the model-B exact-permutation statistic
# ===========================================================================
def exact_matches(path, arm_key, items, ctx, cell):
    """Per call: does the emitted ordering equal the phenotype-only ranker's?"""
    rows = []
    for line in open(path):
        r = json.loads(line)
        if (r.get("arm") or r.get("condition")) != arm_key:
            continue
        if items[r["item_id"]]["cell"] != cell:
            continue
        pr, ar, gold = A._source_rankings(items[r["item_id"]], r["order"], ctx)
        if gold != r["gold_alias"]:
            raise SystemExit(
                f"alias mismatch {r['item_id']}/o{r['order']}: recomputed {gold} "
                f"!= stored {r['gold_alias']}. Wrong interpreter -- use conda "
                f"ops-experimental (3.10) with PYTHONHASHSEED=0.")
        mk = r["response"].get("ranking") if isinstance(r["response"], dict) else None
        rows.append({"key": (r["item_id"], r["order"]), "ranking": mk,
                     "phen": pr, "ann": ar, "gold": gold,
                     "presented": A_presented(items[r["item_id"]], r["order"], ctx)})
    return rows


def A_presented(item, order, ctx):
    packet, _, _ = pilot.make_packet(item, "H-mask", order, ctx)
    return list(packet["candidates"])


def fisher_greater(a, b, c, d):
    """One-sided Fisher exact p for a 2x2 table [[a,b],[c,d]] (a/(a+b) > c/(c+d))."""
    def logC(n, k):
        return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))
    n = a + b + c + d
    r1, c1 = a + b, a + c
    lo, hi = max(0, c1 - b), min(r1, c1)
    denom = logC(n, c1)
    p = 0.0
    for x in range(a, hi + 1):
        p += math.exp(logC(r1, x) + logC(n - r1, c1 - x) - denom)
    return min(1.0, p)


def task3(items, ctx):
    out = {}
    B = ART / "tool_agent_gpt56"
    main = {c: exact_matches(B / "responses.jsonl", "agent_free", items, ctx, c)
            for c in ("conflict", "control")}
    retest = {c: exact_matches(B / "responses_retest.jsonl", "agent_free", items, ctx, c)
              for c in ("conflict", "control")}
    Amain = {c: exact_matches(ART / "tool_agent" / "responses.jsonl", "agent_free",
                              items, ctx, c) for c in ("conflict", "control")}

    def count(rows):
        return {"n_calls": len(rows),
                "exact_phenotype": sum(1 for r in rows if r["ranking"] == r["phen"]),
                "exact_annotation": sum(1 for r in rows if r["ranking"] == r["ann"]),
                "distinct_orderings": len({tuple(r["ranking"]) for r in rows
                                           if r["ranking"]}),
                "phen_equals_presented_order":
                    sum(1 for r in rows if r["phen"] == r["presented"]),
                "emitted_equals_presented_order":
                    sum(1 for r in rows if r["ranking"] == r["presented"])}

    out["model_B"] = {c: count(main[c]) for c in main}
    out["model_B_retest"] = {c: count(retest[c]) for c in retest}
    out["model_A"] = {c: count(Amain[c]) for c in Amain}

    # --- self-agreement: same key, two independent samples ------------------
    for name, rows in (("model_B", (main, retest)),):
        m, rt = rows
        agree = {}
        for c in ("conflict", "control"):
            bykey = {r["key"]: r for r in m[c]}
            pairs = [(bykey[r["key"]], r) for r in rt[c] if r["key"] in bykey]
            if not pairs:
                agree[c] = {"n_repeated_keys": 0}
                continue
            def grank(r):
                rk, g = r["ranking"], r["gold"]
                return rk.index(g) if rk and g in rk else -1
            agree[c] = {
                "n_repeated_keys": len(pairs),
                "self_agreement_FULL_ORDERING": rnd(np.mean(
                    [1.0 if x["ranking"] == y["ranking"] else 0.0 for x, y in pairs]), 3),
                "self_agreement_GOLD_RANK_ONLY": rnd(np.mean(
                    [1.0 if grank(x) == grank(y) else 0.0 for x, y in pairs]), 3),
                "note": ("noise_floor.json's 0.911 is GOLD-RANK agreement (it "
                         "compares reciprocal ranks), not full-ordering "
                         "agreement. The paper must not cite it as the latter."),
            }
        out["model_B_self_agreement"] = agree
    out["model_B_self_agreement"]["published_overall"] = {
        "n_repeated_keys": 79, "exact_agreement": 0.911,
        "note": "artifacts/evidence_eval/tool_agent_gpt56/noise_floor.json"}

    # --- the three candidate nulls -----------------------------------------
    ctrl, conf = out["model_B"]["control"], out["model_B"]["conflict"]
    k, n = ctrl["exact_phenotype"], ctrl["n_calls"]
    out["nulls"] = {}

    out["nulls"]["uniform_720_STRAWMAN"] = {
        "description": "P(match a specified 6-permutation) = 1/720 per call, i.i.d.",
        "p_value": f"~1e{round(k * math.log10(1/720))}",
        "log10_p": rnd(k * math.log10(1 / 720), 1),
        "verdict": ("STRAWMAN -- do not use. No LLM samples uniformly over "
                    "orderings; it samples from a sharply peaked distribution "
                    "concentrated on evidence-consistent orders. Rejecting "
                    "uniformity is uninformative and a reviewer will say so."),
    }

    # control-cell match spread over items: 2 orders/item, differently permuted
    byitem = defaultdict(list)
    for r in main["control"]:
        byitem[r["key"][0]].append(r["ranking"] == r["phen"])
    out["model_B"]["control"]["items_matching_on_both_orders"] = \
        sum(1 for v in byitem.values() if all(v))
    out["model_B"]["control"]["items_matching_on_one_order"] = \
        sum(1 for v in byitem.values() if any(v) and not all(v))
    out["model_B"]["control"]["items_matching_on_neither"] = \
        sum(1 for v in byitem.values() if not any(v))

    sa = out["model_B_self_agreement"]["conflict"].get("self_agreement_FULL_ORDERING")
    out["nulls"]["entropy_only"] = {
        "description": (
            "Could low sampling entropy alone explain 39/40? Self-agreement "
            "bounds how concentrated the output distribution is, but says "
            "NOTHING about WHICH ordering it concentrates on. A model that "
            "deterministically emitted any other fixed ordering would show the "
            "same self-agreement and 0/40 external matches. Entropy and mode "
            "identity are logically independent."),
        "conflict_self_agreement_full_ordering": sa,
        "conflict_n_repeated_keys":
            out["model_B_self_agreement"]["conflict"]["n_repeated_keys"],
        "control_self_agreement": None,
        "external_match_rate_control": rnd(k / n, 3),
        "headline": (
            f"At forced T=1 model B reproduces ITS OWN full six-item ordering on "
            f"only {sa} of re-sampled conflict calls, yet reproduces an EXTERNAL "
            f"deterministic ranker's full ordering on {k}/{n} = {rnd(k/n,3)} of "
            f"control calls. It agrees with the phenotype ranker more than it "
            f"agrees with itself."),
        "caveat": ("Two caveats a reviewer will raise, and they are fair. (1) "
                   "Model B's retest covers the CONFLICT cell only (80 calls); "
                   "control-cell self-agreement was never measured, so we cannot "
                   "exclude that control calls are simply near-zero-entropy. (2) "
                   "The two cells differ in difficulty, so 0.575 vs 0.975 is not "
                   "a matched comparison. Neither caveat rescues an entropy "
                   "explanation: low entropy still leaves open WHICH ordering is "
                   "the mode, and the finding is that the mode is the external "
                   "ranker's output."),
        "verdict": "Not a competing explanation; it constrains stability, not target.",
    }

    # Give the sceptic the best version of "it is just near-deterministic":
    # read the measured full-ordering self-agreement as a collision probability
    # sum(p_i^2) and put as much mass as possible on one ordering.
    pmax = math.sqrt(max(sa - 1.0 / 719, 0.0)) if sa else None
    out["nulls"]["matched_entropy_bound"] = {
        "description": (
            "Suppose the control calls had the SAME output entropy as the "
            "measured conflict calls. Full-ordering self-agreement 0.575 is the "
            "collision probability sum(p_i^2); concentrating as much mass as "
            "possible on a single ordering gives p_mode <= sqrt(0.575)."),
        "measured_collision_probability": sa,
        "implied_max_probability_of_any_single_ordering": rnd(pmax, 3),
        "observed_match_rate_to_external_ranker": rnd(k / n, 3),
        "verdict": ("Even the most deterministic reading of the measured entropy "
                    "caps the match rate at ~0.76, below the observed 0.975. "
                    "Conditional on the conflict-cell entropy transferring -- "
                    "which is unverified, since no control retest exists."),
    }

    a, b = ctrl["exact_phenotype"], ctrl["n_calls"] - ctrl["exact_phenotype"]
    c_, d = conf["exact_phenotype"], conf["n_calls"] - conf["exact_phenotype"]
    p = fisher_greater(a, b, c_, d)
    out["nulls"]["within_model_cell_contrast"] = {
        "description": (
            "The defensible null. Same model, same deployment, same forced T=1, "
            "same tool stack, same prompt -- only the EVIDENCE differs. If the "
            "39/40 control match were a decoding/entropy artifact it would hold "
            "in the conflict cell too. It does not."),
        "control": f"{a}/{a+b}", "conflict": f"{c_}/{c_+d}",
        "fisher_exact_one_sided_p": p,
        "log10_p": rnd(math.log10(p), 1) if p > 0 else None,
        "verdict": "Use this one. It is a real empirical contrast, not a coin-flip null.",
    }

    out["confound_checks"] = {
        "phen_ordering_equals_presentation_order_control":
            f"{ctrl['phen_equals_presented_order']}/{ctrl['n_calls']}",
        "phen_ordering_equals_presentation_order_conflict":
            f"{conf['phen_equals_presented_order']}/{conf['n_calls']}",
        "emitted_equals_presentation_order_control":
            f"{ctrl['emitted_equals_presented_order']}/{ctrl['n_calls']}",
        "note": ("If the phenotype-only ordering coincided with the order the "
                 "candidates were listed in, matching it would be trivial. It "
                 "does not, so the match is to the numeric evidence."),
    }
    return out


# ===========================================================================
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items, src, ctx, n_round_tie = load_arbitration_sources()

    report = {"generated_by": "scripts/evidence_eval/headroom.py",
              "bootstrap": "arbitration._boot: 10,000 resamples, seed 0, "
                           "percentile, paired over items",
              "n_items_where_4dp_rounding_ties_the_gold_phenotype": n_round_tie,
              "task1_exceedance": task1(src)}
    t2 = task2_tables(src)
    t2["uncalibrated_60_item_draw"] = task2_uncalibrated(t2)
    report["task2_top1"] = t2
    report["task3_permutation"] = task3(items, ctx)

    # regression guard: recomputed MRRs must match the published metrics.json
    pub = json.load(open(ART / "arbitration" / "metrics.json"))
    for cell, arm, key in (("conflict", "naive", "naive_MRR"),
                           ("conflict", "rel", "rel_MRR"),
                           ("control", "rel_scaffold", "rel_scaffold_MRR")):
        got = t2["tab_main"][cell]["arms"][arm]["MRR"]["mean"]
        want = pub["cells"][cell][key]
        assert abs(got - want) < 5e-4, (cell, arm, got, want)
    assert report["task1_exceedance"]["corrected_denominator"][
        "exceedable_item_arm_cells"] == 115

    json.dump(report, open(OUT / "metrics.json", "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote -> {OUT/'metrics.json'}")


if __name__ == "__main__":
    main()
