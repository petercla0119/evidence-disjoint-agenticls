#!/usr/bin/env python
"""Model A (Vertex) tool-arm retest: T=0 repeat vs API-default temperature.

The paper's 40/40 control claim rests on `artifacts/evidence_eval/tool_agent/responses.jsonl`
(arm agent_free, both cells) but that run was never repeated. Its cited noise floor
comes from arbitration's `responses_retest.jsonl`, which is the *conflict* cell of a
*different* (static `naive`) arm -- wrong arm, wrong cell. This measures model A's
own floor on the arm and cells the claim is about, and asks whether verbatim copying
survives when no temperature is sent at all (what a real user's call looks like).

Reuses the permutation-level agreement helpers from analyze_control_retest so the
model A and model B numbers are computed by identical code.

Usage:
    PYTHONHASHSEED=0 python scripts/evidence_eval/analyze_temp_arms.py
    python scripts/evidence_eval/analyze_temp_arms.py --selfcheck
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arbitration as A                         # noqa: E402
import pilot_counterfactual as pilot            # noqa: E402
from analyze_control_retest import (            # noqa: E402
    _agreement, _pairs, _rank, _rows, _verbatim)

TA = A.ART / "tool_agent"
ARMS = {"retest_t0": TA / "retest_t0", "default_temp": TA / "default_temp"}
CELL_N = {"conflict": 40, "control": 20}


def _by_key(path):
    return {(r["arm"], r["item_id"], r["order"]): r for r in _rows(path)}


def _classify(ra, rb, pr, ar):
    """For a key whose ordering changed: did it switch to the other source's
    ordering, or just perturb the one it had? Mirrors the model B question."""
    if rb == pr:
        return "switched_to_phenotype_ordering"
    if rb == ar:
        return "switched_to_annotation_ordering"
    if ra == pr:
        return "left_phenotype_ordering_for_neither"
    if ra == ar:
        return "left_annotation_ordering_for_neither"
    return "perturbation_neither_source"


def _moved(ra, rb):
    """1-based positions that differ between two orderings."""
    return [i + 1 for i, (x, y) in enumerate(zip(ra, rb)) if x != y]


def _rr_of(rk, gold):
    return 1.0 / (rk.index(gold) + 1) if rk and gold in rk else None


def _item_scores(rows_by_key, ids):
    """Per item: RR and top-1, averaged over the two candidate orders."""
    rr, t1 = {}, {}
    for (arm, iid, order), r in rows_by_key.items():
        if iid not in ids:
            continue
        rk = _rank(r["response"])
        if rk is None:
            continue
        v = _rr_of(rk, r["gold_alias"]) or 0.0
        rr.setdefault(iid, []).append(v)
        t1.setdefault(iid, []).append(float(v == 1.0))
    return ({k: float(np.mean(v)) for k, v in rr.items()},
            {k: float(np.mean(v)) for k, v in t1.items()})


def _arm_report(name, path, orig, items, ctx):
    new = _by_key(path)
    rep = {"n_episodes": len(new), "cells": {}}
    vb = _verbatim(list(new.values()), items, ctx)
    vb_orig = _verbatim(list(orig.values()), items, ctx)

    for cell, n_items in CELL_N.items():
        ids = {i for i, it in items.items() if it["cell"] == cell}
        sub_new = {k: v for k, v in new.items() if k[1] in ids}
        sub_orig = {k: v for k, v in orig.items() if k[1] in ids}
        if not sub_new:
            continue
        pairs = _pairs(sub_orig, sub_new)
        block = {"test_retest_vs_original_T0": _agreement(pairs, n_items)}

        c, co = vb.get(cell, {}), vb_orig.get(cell, {})
        block["verbatim"] = {
            "phenotype_only_ranker": f"{c.get('phenotype', 0)}/{c.get('n', 0)}",
            "annotation_only_ranker": f"{c.get('annotation', 0)}/{c.get('n', 0)}",
            "original_run_phenotype_only_ranker":
                f"{co.get('phenotype', 0)}/{co.get('n', 0)}",
            "original_run_annotation_only_ranker":
                f"{co.get('annotation', 0)}/{co.get('n', 0)}",
        }

        # accuracy: MRR / top-1 per cell, paired bootstrap on the item-level diff
        rr_n, t1_n = _item_scores(sub_new, ids)
        rr_o, t1_o = _item_scores(sub_orig, ids)
        have = sorted(set(rr_n) & set(rr_o))
        d_rr = [rr_n[i] - rr_o[i] for i in have]
        d_t1 = [t1_n[i] - t1_o[i] for i in have]
        block["accuracy"] = {
            "n_items": len(have),
            "MRR": round(float(np.mean([rr_n[i] for i in have])), 4),
            "MRR_CI": A._boot([rr_n[i] for i in have]),
            "original_MRR": round(float(np.mean([rr_o[i] for i in have])), 4),
            "MRR_minus_original": round(float(np.mean(d_rr)), 4),
            "MRR_minus_original_paired_CI": A._boot(d_rr),
            "top1": round(float(np.mean([t1_n[i] for i in have])), 4),
            "original_top1": round(float(np.mean([t1_o[i] for i in have])), 4),
            "top1_minus_original": round(float(np.mean(d_t1)), 4),
            "top1_minus_original_paired_CI": A._boot(d_t1),
        }

        # what kind of disagreement is it?
        kinds, detail = {}, []
        for k, ra, rb, _g in pairs:
            if ra == rb:
                continue
            pr, ar, _ = A._source_rankings(items[k[1]], k[2], ctx)
            kind = _classify(ra, rb, pr, ar)
            kinds[kind] = kinds.get(kind, 0) + 1
            detail.append({"key": list(k), "kind": kind,
                           "positions_changed": _moved(ra, rb),
                           "original": ra, "new": rb})
        block["disagreements"] = {"n": len(detail), "kinds": kinds,
                                  "detail": detail}
        rep["cells"][cell] = block
    return rep


def main():
    items = {it["item_id"]: it
             for it in json.load(open(A.OUT_DIR / "items.json"))["items"]}
    ctx = pilot._ctx()
    orig = _by_key(TA / "responses.jsonl")
    orig = {k: v for k, v in orig.items() if k[0] == "agent_free"}

    out = {"model": "claude-sonnet-4-6 (Vertex)", "arm": "agent_free",
           "original_run": str((TA / "responses.jsonl").relative_to(A.ART.parent.parent)),
           "n_original_episodes": len(orig), "arms": {}}
    for name, d in ARMS.items():
        p = d / "responses.jsonl"
        if not p.exists():
            continue
        rep = _arm_report(name, p, orig, items, ctx)
        out["arms"][name] = rep
        d.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(d / "metrics.json", "w"), indent=2)
        print(f"wrote -> {d/'metrics.json'}")

    out["note"] = (
        "full_ordering_exact_agreement is permutation-level; gold_rank_agreement is "
        "equality of 1/rank(gold), which is what arbitration.noise_floor() reports. "
        "implied_MRR_noise_se uses each cell's own item count (control=20), not the "
        "sqrt(40) hardcoded in noise_floor().")
    json.dump(out, open(TA / "temp_arms_metrics.json", "w"), indent=2)
    print(json.dumps(out, indent=2, default=str)[:4000])
    print(f"\nwrote -> {TA/'temp_arms_metrics.json'}")


def _selfcheck():
    # full-ordering and gold-rank agreement must diverge exactly when the
    # permutation changes but gold keeps its slot.
    p = [(("a", "i", 0), ["A", "B", "C"], ["A", "C", "B"], "A"),   # perm differs, gold same
         (("a", "i", 1), ["A", "B", "C"], ["A", "B", "C"], "A"),   # identical
         (("a", "i", 2), ["A", "B", "C"], ["B", "A", "C"], "A")]   # gold moved
    r = _agreement(p, 3)
    assert r["full_ordering_exact_agreement"] == round(1 / 3, 4), r
    assert r["gold_rank_agreement"] == round(2 / 3, 4), r
    assert r["se_n_items"] == 3, r

    # per-cell SE really does use the cell's own n, so control (20) != sqrt(40)
    a20, a40 = _agreement(p, 20), _agreement(p, 40)
    assert a20["implied_MRR_noise_se"] > a40["implied_MRR_noise_se"], (a20, a40)

    # disagreement classifier
    pr, ar = ["A", "B", "C"], ["C", "B", "A"]
    assert _classify(pr, ar, pr, ar) == "switched_to_annotation_ordering"
    assert _classify(ar, pr, pr, ar) == "switched_to_phenotype_ordering"
    assert _classify(pr, ["B", "A", "C"], pr, ar) == \
        "left_phenotype_ordering_for_neither"
    assert _classify(["B", "A", "C"], ["B", "C", "A"], pr, ar) == \
        "perturbation_neither_source"
    assert _moved(["A", "B", "C", "D"], ["A", "C", "B", "D"]) == [2, 3]
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
