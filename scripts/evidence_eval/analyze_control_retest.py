#!/usr/bin/env python
"""Model B control-cell test-retest: full-ordering entropy, not just gold-rank.

`arbitration.noise_floor()` reports agreement on the *reciprocal rank of the gold
candidate*. Two runs that emit different six-item permutations agree there as long
as gold lands in the same slot. The paper's 39/40 claim is about reproducing a
whole ordering, so it needs permutation-level agreement, measured on the same cell
the claim is about (control).

Usage:
    PYTHONHASHSEED=0 python scripts/evidence_eval/analyze_control_retest.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arbitration as A                       # noqa: E402
import pilot_counterfactual as pilot          # noqa: E402

G56 = A.ART / "tool_agent_gpt56"
OUT = G56 / "control_retest"


def _rows(p):
    return [json.loads(l) for l in open(p)]


def _rank(resp):
    return resp["ranking"] if isinstance(resp, dict) and "ranking" in resp else None


def _rr(rk, gold):
    return 1.0 / (rk.index(gold) + 1) if rk and gold in rk else None


def _pairs(first, second):
    """Keys present and parseable in both runs, with their rankings + gold."""
    out = []
    for k, b in second.items():
        a = first.get(k)
        if a is None:
            continue
        ra, rb = _rank(a["response"]), _rank(b["response"])
        if ra is None or rb is None:
            continue
        assert a["gold_alias"] == b["gold_alias"], k
        out.append((k, ra, rb, a["gold_alias"]))
    return out


def _agreement(pairs, n_items_for_se):
    d = [abs(_rr(ra, g) - _rr(rb, g)) for _, ra, rb, g in pairs]
    return {
        "n_repeated_keys": len(pairs),
        "full_ordering_exact_agreement": round(
            sum(1 for _, ra, rb, _ in pairs if ra == rb) / len(pairs), 4),
        "gold_rank_agreement": round(sum(1 for x in d if x < 1e-9) / len(pairs), 4),
        "mean_abs_RR_difference": round(float(np.mean(d)), 4),
        "p95_abs_RR_difference": round(float(np.percentile(d, 95)), 4),
        "implied_MRR_noise_se": round(
            float(np.std(d, ddof=1) / np.sqrt(n_items_for_se)), 4),
        "se_n_items": n_items_for_se,
    }


def _verbatim(rows, items, ctx):
    """Per cell: how often the emitted ordering IS a source's ordering, verbatim."""
    out = {}
    for r in rows:
        rk = _rank(r["response"])
        if rk is None:
            continue
        pr, ar, gold = A._source_rankings(items[r["item_id"]], r["order"], ctx)
        assert gold == r["gold_alias"], (r["item_id"], r["order"])
        c = out.setdefault(r["cell"], {"n": 0, "phenotype": 0, "annotation": 0})
        c["n"] += 1
        c["phenotype"] += rk == pr
        c["annotation"] += rk == ar
    return out


def main():
    items = {it["item_id"]: it
             for it in json.load(open(A.OUT_DIR / "items.json"))["items"]}
    ctx = pilot._ctx()

    orig = {(r["arm"], r["item_id"], r["order"]): r
            for r in _rows(G56 / "responses.jsonl")}
    conf_rt = {(r["arm"], r["item_id"], r["order"]): r
               for r in _rows(G56 / "responses_retest.jsonl")}
    cont_rt = {(r["arm"], r["item_id"], r["order"]): r
               for r in _rows(OUT / "responses_control_retest.jsonl")}
    assert all(orig[k]["cell"] == "control" for k in cont_rt), "non-control rows"

    # control cell has 20 items, conflict 40; noise_floor.json divided by sqrt(40)
    # for the conflict cell, so use each cell's own item count and say so.
    cont_pairs = _pairs(orig, cont_rt)
    conf_pairs = _pairs(orig, conf_rt)
    control = _agreement(cont_pairs, 20)
    conflict = _agreement(conf_pairs, 40)

    vb_orig = _verbatim([r for r in orig.values()], items, ctx)
    vb_rt = _verbatim(list(cont_rt.values()), items, ctx)

    # cell contrast the paper should cite: does the model reproduce the ordering of
    # the source that dominates the cell? control -> phenotype, conflict -> annotation.
    c_hit, c_n = vb_rt["control"]["phenotype"], vb_rt["control"]["n"]
    f_hit, f_n = vb_orig["conflict"]["annotation"], vb_orig["conflict"]["n"]
    odds, p = fisher_exact([[c_hit, c_n - c_hit], [f_hit, f_n - f_hit]],
                           alternative="greater")

    # the contrast the body's "not decoding stability" sentence needs: SELF-agreement on
    # the full ordering, control vs conflict. Same quantity on both sides, unlike the
    # copy-rate contrast above, which scores a different target ranker in each cell.
    s_hit = sum(1 for _, ra, rb, _ in cont_pairs if ra == rb)
    t_hit = sum(1 for _, ra, rb, _ in conf_pairs if ra == rb)
    s_odds, s_p = fisher_exact(
        [[s_hit, len(cont_pairs) - s_hit], [t_hit, len(conf_pairs) - t_hit]],
        alternative="greater")

    rep = {
        "model": "gpt-5.6-luna (Azure, forced T=1)",
        "arm": "agent_free",
        "control_retest": {
            "n_episodes": len(cont_rt),
            "test_retest": control,
            "verbatim_match_phenotype_only_ranker":
                f"{c_hit}/{c_n}",
            "verbatim_match_annotation_only_ranker":
                f"{vb_rt['control']['annotation']}/{c_n}",
        },
        "control_original_run": {
            "verbatim_match_phenotype_only_ranker":
                f"{vb_orig['control']['phenotype']}/{vb_orig['control']['n']}",
        },
        "conflict_retest_for_contrast": {
            "test_retest": conflict,
            "verbatim_match_annotation_only_ranker_original_run":
                f"{f_hit}/{f_n}",
        },
        "cell_contrast_fisher_exact_one_sided": {
            "control_phenotype_verbatim": f"{c_hit}/{c_n}",
            "conflict_annotation_verbatim": f"{f_hit}/{f_n}",
            "odds_ratio": None if np.isinf(odds) else round(float(odds), 3),
            "p": float(f"{p:.3g}"),
        },
        "self_agreement_cell_contrast_fisher_exact_one_sided": {
            "control_full_ordering_self_agreement": f"{s_hit}/{len(cont_pairs)}",
            "conflict_full_ordering_self_agreement": f"{t_hit}/{len(conf_pairs)}",
            "odds_ratio": None if np.isinf(s_odds) else round(float(s_odds), 3),
            "p": float(f"{s_p:.3g}"),
            "note": (
                "The statistic the body cites for 'this is not decoding stability'. "
                "Both sides measure the SAME thing -- the model reproducing its own "
                "six-item ordering across repeated calls -- so unlike "
                "cell_contrast_fisher_exact_one_sided it compares like with like. "
                "Denominator is repeated keys parseable in BOTH runs, so conflict is "
                "79, not 80: one key failed to parse in one of the two runs."),
        },
        "note": (
            "noise_floor.json's exact_agreement_rate is gold-rank agreement "
            "(equality of 1/rank(gold)), NOT equality of the six-item permutation. "
            "full_ordering_exact_agreement here is the permutation-level quantity."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(OUT / "metrics.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print(f"\nwrote -> {OUT/'metrics.json'}")


def _selfcheck():
    """The two agreement notions must differ exactly when a permutation changes
    but gold's slot does not."""
    p = [(("a", "i", 0), ["A", "B", "C"], ["A", "C", "B"], "A"),
         (("a", "i", 1), ["A", "B", "C"], ["A", "B", "C"], "A"),
         (("a", "i", 2), ["A", "B", "C"], ["B", "A", "C"], "A")]
    r = _agreement(p, 3)
    assert r["full_ordering_exact_agreement"] == round(1 / 3, 4), r
    assert r["gold_rank_agreement"] == round(2 / 3, 4), r
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
