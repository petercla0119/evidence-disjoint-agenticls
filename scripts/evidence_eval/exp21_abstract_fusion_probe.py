#!/usr/bin/env python
"""Abstract fusion probe: the same arithmetic, with all biology removed.

The arbitration result (artifacts/evidence_eval/arbitration, artifacts/evidence_eval/tool_agent) is
that the model SELECTS one evidence source rather than fusing two. Three
mechanisms fit that data:

  (1) output-format copying -- the task asks for a ranking, the tools hand back
      rankings, so emitting one verbatim is the path of least resistance;
  (2) an inability to fuse two numeric score vectors at all;
  (3) salience-following -- it tracks whichever source looks sharper.

This probe separates (1) from (2). Same 60 items, same MASKED packets, same two
score vectors, same reliability table keyed on the same two observables -- but
every biological referent is gone. Six options A-F, "System X" and "System Y",
no gene, cell, annotation, phenotype, pathway, complex, screen or partner
anywhere in the prompt (asserted at run time by _assert_biology_free).

If the model integrates here, the selection behaviour is a property of the
agentic/biological framing, not a numeric-reasoning limit -> mechanism (1).
If it still copies one vector's ordering verbatim, -> mechanism (2).

System X = the annotation graph score, rc._ann_scores(masked packet).
System Y = the phenotype cosine block of the same masked packet.
Both are exactly the vectors the paper's annotation-only / phenotype-only
rankers are computed from, so the comparison is apples to apples.

Usage:
    PYTHONHASHSEED=0 python scripts/evidence_eval/exp21_abstract_fusion_probe.py selfcheck
    PYTHONHASHSEED=0 python scripts/evidence_eval/exp21_abstract_fusion_probe.py smoke
    PYTHONHASHSEED=0 python scripts/evidence_eval/exp21_abstract_fusion_probe.py run
    PYTHONHASHSEED=0 python scripts/evidence_eval/exp21_abstract_fusion_probe.py analyze
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arbitration as A                       # noqa: E402
import pilot_counterfactual as pilot          # noqa: E402
import reliability_calibration as rc          # noqa: E402

OUT_DIR = A.ART / "fusion_probe"
RESP = OUT_DIR / "responses.jsonl"
N_ORDERS = A.N_ORDERS
MAX_CALLS = 130                     # hard budget, includes the smoke call
LETTERS = ["A", "B", "C", "D", "E", "F"]

# Any of these appearing in a rendered prompt means the framing leaked.
BIO_WORDS = [
    "gene", "cell", "annotat", "phenotyp", "biolog", "protein", "pathway",
    "complex", "crispr", "screen", "partner", "morpholog", "perturb",
    "interact", "string", "cosine", "organism", "express", "knockout",
]


# ---------------------------------------------------------------------------
# prompts -- module-level so a reader can audit that the framing is biology-free
# ---------------------------------------------------------------------------
SYSTEM = (
    "You are ranking six options, labelled A through F. Exactly one of them is "
    "the target; the other five are not. Two independent scoring systems, "
    "System X and System Y, each assign a number to every option; a higher "
    "number means that system considers the option more likely to be the "
    "target. Neither system is dependable on its own. You are also given a "
    "reliability table: hit-rates measured on an independent calibration set, "
    "keyed on two statistics that are computed for you in this_item -- System "
    "Y's top1-top2 gap and System X's top1-top2 margin. Look up the row "
    "matching this_item and weigh each system by how reliable it is in that "
    "row, relative to the random rate. A system whose top pick is right at or "
    "below the random rate carries no information here. Reason only from the "
    "numbers you are given."
)

USER_TMPL = (
    "Input (JSON):\n{payload}\n\n"
    "Task: rank all six options A-F from most to least likely to be the "
    "target. Return ONLY machine-readable JSON with keys:\n"
    '  "ranking": [six option labels, best first],\n'
    '  "probabilities": {{option label: probability}}  (must sum to 1)\n'
    "No prose outside the JSON."
)


def _assert_biology_free(text):
    low = text.lower()
    hits = sorted({w for w in BIO_WORDS if w in low})
    if hits:
        raise AssertionError(f"biology leaked into the prompt: {hits}")


# ---------------------------------------------------------------------------
# payload construction (offline, deterministic)
# ---------------------------------------------------------------------------
def _abstract_table(cal):
    """The same calibration numbers with every biological label renamed."""
    def key(k):
        g, m = k.split(" | ")
        return (g.replace("gap", "Y_gap") + " | " +
                m.replace("no discriminating annotation", "no discriminating X")
                 .replace("annotation margin", "X_margin"))
    return {key(k): {
        "n": v["n"],
        "system_Y_top_pick_is_target": v["phenotype_top_pick_is_true_partner"],
        "system_X_top_pick_is_target": v["annotation_top_pick_is_true_partner"],
        "system_Y_expected_reciprocal_rank": v["phenotype_expected_reciprocal_rank"],
        "system_X_expected_reciprocal_rank": v["annotation_expected_reciprocal_rank"],
    } for k, v in cal["table"].items()}


def build_payload(item, order, cal, ctx):
    """(payload, target_letter, x_ranking, y_ranking) for one (item, order).

    x_ranking / y_ranking are the single-system orderings in letters, strict
    sort tiebreak -- the same convention arbitration._source_rankings uses for
    exact-match counting.
    """
    packet, gold_alias, _ = pilot.make_packet(item, "H-mask", order, ctx)
    L = {f"G{i+1}": LETTERS[i] for i in range(6)}      # G1..G6 -> A..F

    x = {L[g]: round(v, 3) for g, v in rc._ann_scores(packet).items()}
    y = {L[g]: v for g, v in packet["phenotype_similarity"].items()}
    x = {k: x[k] for k in LETTERS}
    y = {k: y[k] for k in LETTERS}

    row = (f"{A._bin(item['gap'], A.GAP_BINS).replace('gap', 'Y_gap')} | "
           f"{A._bin(item['margin'], A.MARGIN_BINS).replace('no discriminating annotation', 'no discriminating X').replace('annotation margin', 'X_margin')}")

    payload = {
        "options": LETTERS,
        "system_X_score": x,
        "system_Y_score": y,
        "system_reliability": {
            "measured_on": "an independent calibration set of 2551 cases, "
                           "disjoint from the items scored here",
            "keys": "System Y gap = top1 minus top2 of the System Y scores; "
                    "System X margin = top1 minus top2 of the System X scores. "
                    "Both are computable from the input without knowing the target.",
            "random_option_is_target": cal["random_candidate_is_true_partner"],
            "random_ranking_expected_reciprocal_rank":
                cal["random_ranking_expected_reciprocal_rank"],
            "table": _abstract_table(cal),
            "this_item": {
                "system_Y_gap": round(item["gap"], 4),
                "system_X_margin": round(item["margin"], 4),
                "matching_row": row,
            },
        },
    }
    return (payload, L[gold_alias],
            sorted(LETTERS, key=lambda c: -x[c]),
            sorted(LETTERS, key=lambda c: -y[c]))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def _load():
    cal = json.load(open(A.OUT_DIR / "calibration.json"))
    items = json.load(open(A.OUT_DIR / "items.json"))["items"]
    return cal, items, pilot._ctx()


def run(model, smoke=False):
    cal, items, ctx = _load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done = set()
    if RESP.exists():
        for line in open(RESP):
            r = json.loads(line)
            done.add((r["item_id"], r["order"]))

    jobs = [(it, o) for it in items for o in range(N_ORDERS)]
    if smoke:
        jobs = jobs[:1]
    jobs = [j for j in jobs if (j[0]["item_id"], j[1]) not in done]

    budget = MAX_CALLS - len(done)
    if budget <= 0:
        print(f"ABORT: {len(done)} calls already spent, cap is {MAX_CALLS}")
        return
    if len(jobs) > budget:
        print(f"trimming {len(jobs)} jobs to the remaining budget of {budget}")
        jobs = jobs[:budget]

    client = pilot.get_client()
    calls = len(done)
    f = open(RESP, "a")
    for n, (it, order) in enumerate(jobs, 1):
        if calls >= MAX_CALLS:
            print(f"ABORT: hit the {MAX_CALLS}-call cap")
            break
        payload, target, xr, yr = build_payload(it, order, cal, ctx)
        user = USER_TMPL.format(payload=json.dumps(payload))
        _assert_biology_free(SYSTEM + "\n" + user)

        msg = client.messages.create(
            model=model, max_tokens=1500, system=SYSTEM,
            messages=[{"role": "user", "content": user}],
            extra_body={"temperature": 0.0},
        )
        calls += 1
        text = msg.content[0].text
        try:
            parsed = pilot.parse_response(text)
        except Exception as e:
            parsed = {"_parse_error": str(e), "_raw": text[:500]}

        rec = dict(item_id=it["item_id"], cell=it["cell"], order=order,
                   target=target, x_ranking=xr, y_ranking=yr,
                   phen_rank=it["phen_rank"], ann_rank=it["ann_rank"],
                   gap=it["gap"], margin=it["margin"], model=model,
                   response=parsed)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        rk = parsed.get("ranking", []) if isinstance(parsed, dict) else []
        pos = (rk.index(target) + 1) if target in rk else "-"
        eq = "=X" if rk == xr else ("=Y" if rk == yr else "  ")
        print(f"[{n}/{len(jobs)}] calls={calls:3d}/{MAX_CALLS} {it['item_id']:7s} "
              f"o{order} {it['cell']:8s} target={target} pos={pos} {eq}")
    f.close()
    print(f"{calls} total calls; wrote -> {RESP}")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
def _matched_biology(ids, items):
    """Biology-framed `rel` arm scored on the probe's item subset only.

    arbitration's stored metrics aggregate the whole cell. The probe is a subset
    of it, so the cross-arm contrast is only interpretable item-matched.
    """
    ctx = pilot._ctx()
    by = {(r["arm"], r["item_id"], r["order"]): r
          for r in (json.loads(l) for l in open(A.OUT_DIR / "responses.jsonl")
                    if l.strip())}
    rr, exact_a, exact_p = {}, 0, 0
    for iid in ids:
        per_order = []
        for o in range(A.N_ORDERS):
            r = by.get(("rel", iid, o))
            if not r or not isinstance(r["response"], dict):
                continue
            mk = r["response"].get("ranking")
            pr, ar, gold = A._source_rankings(items[iid], o, ctx)
            # Same guard as arbitration._selector_stats: the candidate permutation
            # comes from hash((item_id, order)), which is not stable across Python
            # versions even at PYTHONHASHSEED=0. Crash rather than report a false null.
            if gold != r["gold_alias"]:
                raise RuntimeError(
                    f"alias mismatch for rel/{iid}/o{o}: recomputed {gold} != stored "
                    f"{r['gold_alias']}. Re-run under conda ops-experimental "
                    f"(Python 3.10) with PYTHONHASHSEED=0.")
            per_order.append(1.0 / (mk.index(gold) + 1) if gold in mk else 0.0)
            exact_a += (mk == ar)
            exact_p += (mk == pr)
        if per_order:
            rr[iid] = float(np.mean(per_order))
    X = {i: 1.0 / items[i]["ann_rank"] for i in rr}
    Y = {i: 1.0 / items[i]["phen_rank"] for i in rr}
    return {
        "n_items": len(rr),
        "rel_MRR": round(float(np.mean(list(rr.values()))), 4),
        "rel_MRR_CI": A._boot(list(rr.values())),
        "exact_matches_annotation": exact_a,
        "exact_matches_phenotype": exact_p,
        "n_calls": len(rr) * A.N_ORDERS,
        "exceeds_both": sum(1 for i in rr if rr[i] > max(X[i], Y[i]) + 1e-9),
    }


def analyze():
    items = {it["item_id"]: it
             for it in json.load(open(A.OUT_DIR / "items.json"))["items"]}
    rows = [json.loads(l) for l in open(RESP) if l.strip()]

    per_item = defaultdict(list)
    exact = defaultdict(lambda: [0, 0])         # cell -> [=X, =Y]
    n_calls = defaultdict(int)
    n_parse_err = 0
    for r in rows:
        resp = r["response"]
        if not isinstance(resp, dict) or "ranking" not in resp:
            n_parse_err += 1
            continue
        rk = resp["ranking"]
        per_item[r["item_id"]].append(
            1.0 / (rk.index(r["target"]) + 1) if r["target"] in rk else 0.0)
        n_calls[r["cell"]] += 1
        if rk == r["x_ranking"]:
            exact[r["cell"]][0] += 1
        if rk == r["y_ranking"]:
            exact[r["cell"]][1] += 1

    bio = json.load(open(A.OUT_DIR / "metrics.json"))
    fusion_path = A.ART / "fusion" / "metrics.json"
    fusion = json.load(open(fusion_path)) if fusion_path.exists() else None

    report = {"parse_errors": n_parse_err,
              "noise_floor_MRR_se_40_items": 0.0043,
              "mechanical_fusion_baseline": "present" if fusion else "absent",
              "cells": {}}
    for cell in ("conflict", "control"):
        ids = [i for i, it in items.items()
               if it["cell"] == cell and i in per_item]
        if not ids:
            continue
        rr = {i: float(np.mean(per_item[i])) for i in ids}
        # X = annotation-only, Y = phenotype-only, same avg-rank tie handling
        # the paper uses, so these MRRs are directly comparable to Table 1.
        X = {i: 1.0 / items[i]["ann_rank"] for i in ids}
        Y = {i: 1.0 / items[i]["phen_rank"] for i in ids}
        oracle = {i: max(X[i], Y[i]) for i in ids}
        d_x = [rr[i] - X[i] for i in ids]
        d_y = [rr[i] - Y[i] for i in ids]
        bcell = bio["cells"][cell]
        out = {
            "n_items": len(ids),
            "n_calls": n_calls[cell],
            "probe_MRR": round(float(np.mean([rr[i] for i in ids])), 4),
            "probe_MRR_CI": A._boot([rr[i] for i in ids]),
            "system_X_only_MRR": round(float(np.mean([X[i] for i in ids])), 4),
            "system_Y_only_MRR": round(float(np.mean([Y[i] for i in ids])), 4),
            "best_source_oracle_MRR": round(float(np.mean([oracle[i] for i in ids])), 4),
            "probe_minus_X": round(float(np.mean(d_x)), 4),
            "probe_minus_X_CI": A._boot(d_x),
            "probe_minus_Y": round(float(np.mean(d_y)), 4),
            "probe_minus_Y_CI": A._boot(d_y),
            "exceeds_both_systems": sum(1 for i in ids if rr[i] > oracle[i] + 1e-9),
            "exact_ranking_matches_X": exact[cell][0],
            "exact_ranking_matches_Y": exact[cell][1],
        }
        # The biology-framed arm ran on the full cell (40 conflict items); the probe
        # ran on a subset. Comparing the two cells' aggregates compares different
        # item sets, so recompute the biology side restricted to the probe's ids.
        out.update({f"matched_biology_{k}": v
                    for k, v in _matched_biology(ids, items).items()})
        out["probe_minus_matched_biology_MRR"] = round(
            float(np.mean([rr[i] for i in ids])) - out["matched_biology_rel_MRR"], 4)
        if fusion and cell in fusion.get("cells", {}):
            out["mechanical_fusion_MRR"] = fusion["cells"][cell].get("fusion_MRR")
        report["cells"][cell] = out

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT_DIR / "metrics.json", "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote -> {OUT_DIR/'metrics.json'}")
    return report


# ---------------------------------------------------------------------------
# self-check: everything that does not touch the API
# ---------------------------------------------------------------------------
def selfcheck():
    cal, items, ctx = _load()
    it = items[0]
    payload, target, xr, yr = build_payload(it, 0, cal, ctx)

    # the prompt really is biology-free
    user = USER_TMPL.format(payload=json.dumps(payload))
    _assert_biology_free(SYSTEM + "\n" + user)
    try:
        _assert_biology_free(user + " gene")
        raise SystemExit("FAIL: the biology guard does not fire")
    except AssertionError:
        pass
    assert not re.search(r"\bG[0-6]\b|\bP\d|\bC\d", user), "alias labels leaked"

    # score vectors are the packet's, just relabelled
    packet, gold_alias, _ = pilot.make_packet(it, "H-mask", 0, ctx)
    Lmap = {f"G{i+1}": LETTERS[i] for i in range(6)}
    assert target == Lmap[gold_alias]
    assert sorted(payload["system_X_score"]) == LETTERS
    assert sorted(payload["system_Y_score"]) == LETTERS
    for g, v in packet["phenotype_similarity"].items():
        assert payload["system_Y_score"][Lmap[g]] == v, "Y vector mangled"
    for g, v in rc._ann_scores(packet).items():
        assert abs(payload["system_X_score"][Lmap[g]] - v) <= 5e-4, "X vector mangled"

    # single-system orderings agree with the arbitration helper, relabelled
    pr, ar, gold2 = A._source_rankings(it, 0, ctx)
    assert gold2 == gold_alias
    assert yr == [Lmap[g] for g in pr], "Y ordering disagrees with _source_rankings"
    assert xr == [Lmap[g] for g in ar], "X ordering disagrees with _source_rankings"

    # reliability row matches the item's observables and exists in the table
    row = payload["system_reliability"]["this_item"]["matching_row"]
    assert row in payload["system_reliability"]["table"], row
    if it["cell"] == "conflict":
        assert row == "Y_gap<0.013 | X_margin>=1.5", row

    # scoring: reciprocal rank of the target
    rr = lambda rk: 1.0 / (rk.index(target) + 1) if target in rk else 0.0
    assert rr([target] + [c for c in LETTERS if c != target]) == 1.0
    assert rr([c for c in LETTERS if c != target] + [target]) == 1.0 / 6
    assert rr([c for c in LETTERS if c != target]) == 0.0

    # the two orders really do permute the options
    p1, t1, _, _ = build_payload(it, 1, cal, ctx)
    assert (p1["system_Y_score"] != payload["system_Y_score"]) or (t1 != target), \
        "order 1 is identical to order 0"

    print("selfcheck OK")
    print(f"  item {it['item_id']} ({it['cell']}) target={target}")
    print(f"  X={payload['system_X_score']}")
    print(f"  Y={payload['system_Y_score']}")
    print(f"  row={row}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("selfcheck", "analyze"):
        sub.add_parser(c)
    for c in ("run", "smoke"):
        p = sub.add_parser(c)
        p.add_argument("--model", default="claude-sonnet-4-6")
    a = ap.parse_args()
    if a.cmd == "selfcheck":
        selfcheck()
    elif a.cmd == "smoke":
        run(a.model, smoke=True)
    elif a.cmd == "run":
        run(a.model)
    elif a.cmd == "analyze":
        analyze()


if __name__ == "__main__":
    main()
