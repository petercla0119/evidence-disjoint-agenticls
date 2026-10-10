#!/usr/bin/env python
"""Exp 7: scaffold-only (uninformed) arbitration on the frozen fair-60 and tail-24.

Exp 4's `rel_scaffold` bundled forced per-source reasoning WITH the calibration
table, so it could only answer "does a scaffold add anything on top of
calibration?" (it does not). This arm removes the table: the model must rank the
candidates by each source separately and commit to a weighting, but is never told
which source is reliable here -- it has to judge that from the packet.

Packets are byte-identical to the H-mask arm (same `make_packet` call, same
condition string), so the only thing that varies is the prompt.

MUST run as `PYTHONHASHSEED=0 python exp7_scaffold_only.py ...` -- make_packet
permutes candidate aliases via hash((item_id, order)), and a different hash seed
silently produces different aliases than the frozen runs used.

  run     -> artifacts/evidence_eval/scaffold_only/<set>/responses.jsonl   (resumable)
  analyze -> artifacts/evidence_eval/scaffold_only/<set>/metrics.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import pilot_counterfactual as pilot
from pilot_counterfactual import make_packet, get_client, parse_response, _rr

BASE = Path(__file__).resolve().parent.parent.parent
ART = BASE / "artifacts" / "evidence_eval"
SETS = {"fair": ART / "validate", "booster": ART / "validate_booster"}
OUT = ART / "scaffold_only"

N_ORDERS = 2
COND = "H-scaffold"
PACKET_COND = "H-mask"  # identical packet to the H-mask arm

# The frozen validate/ and validate_booster/ responses were produced with a
# candidate permutation we cannot regenerate: make_packet derives it from
# hash((item_id, order)), and no interpreter on this machine (3.9-3.13, with
# PYTHONHASHSEED=0) reproduces the stored gold_alias. Their MRRs are unaffected
# -- analysis reads stored aliases and phen_rank, never re-derives packets --
# but a NEW arm cannot be paired against them. So we re-run the comparison arms
# here under one interpreter; every contrast below is within this run.
CONDITIONS = ["A-mask", "H-mask", COND]

SYS_SCAFFOLD_ONLY = pilot.SYSTEM + (
    " Before giving your final ranking you must rank the candidates by each source "
    "separately -- annotation and phenotype -- and state which source you are "
    "weighting more heavily and why. You are NOT told which source is reliable for "
    "this item; judge that yourself from the evidence in the packet, and say so if "
    "you conclude a source carries no information here."
)

USER_SCAFFOLD_ONLY = (
    "Evidence packet (JSON):\n{packet}\n\n"
    "Task: rank all six candidates G1-G6 from most to least likely to be the true "
    "functional partner of G0. Return ONLY machine-readable JSON with keys:\n"
    '  "phenotype_ranking": [six candidate ids, best first by phenotype alone],\n'
    '  "annotation_ranking": [six candidate ids, best first by annotation alone],\n'
    '  "weighted_source": "phenotype" | "annotation" | "neither",\n'
    '  "weighting_reason": "<one sentence>",\n'
    '  "ranking": [six candidate ids, best first],\n'
    '  "probabilities": {{candidate id: probability}}  (must sum to 1),\n'
    '  "first_choice_evidence": [record ids R# supporting your top pick]\n'
    "No prose outside the JSON."
)


def _check_hashseed():
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("refusing to run: set PYTHONHASHSEED=0 (frozen alias permutation)")


def _phen_ranking(packet):
    """Phenotype-only ranking in alias space, best first."""
    sims = packet["phenotype_similarity"]
    return sorted(sims, key=lambda a: -sims[a])


# ---------------------------------------------------------------------------
def run(setname, model, limit=None):
    _check_hashseed()
    src = SETS[setname]
    items = json.load(open(src / "items.json"))["items"]
    ctx = pilot._ctx()
    client = get_client()

    outdir = OUT / setname
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "responses.jsonl"

    done = set()
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            done.add((r["item_id"], r["condition"], r["order"]))

    jobs = [(it, c, o) for it in items for c in CONDITIONS for o in range(N_ORDERS)
            if (it["item_id"], c, o) not in done]
    if limit:
        jobs = jobs[:limit]
    print(f"{setname}: {len(jobs)} calls to make ({len(done)} already done) on {model}")

    f = open(path, "a")
    for n, (it, cond, order) in enumerate(jobs, 1):
        scaf = cond == COND
        packet, gold_alias, audit = make_packet(
            it, PACKET_COND if scaf else cond, order, ctx)
        tmpl = USER_SCAFFOLD_ONLY if scaf else pilot.USER_TMPL
        user = tmpl.format(packet=json.dumps(packet))
        msg = client.messages.create(
            model=model, max_tokens=1024,
            system=SYS_SCAFFOLD_ONLY if scaf else pilot.SYSTEM,
            messages=[{"role": "user", "content": user}],
            extra_body={"temperature": 0.0},
        )
        text = msg.content[0].text
        try:
            parsed = parse_response(text)
        except Exception as e:
            parsed = {"_parse_error": str(e), "_raw": text[:500]}
        rec = dict(item_id=it["item_id"], phen_rank=it["phen_rank"],
                   condition=cond, order=order, gold_alias=gold_alias,
                   model=model, audit=audit,
                   phen_ranking=_phen_ranking(packet) if scaf else None,
                   response=parsed)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        rk = parsed.get("ranking", []) if isinstance(parsed, dict) else []
        pos = (rk.index(gold_alias) + 1) if gold_alias in rk else "-"
        print(f"[{n}/{len(jobs)}] {it['item_id']} {cond:11s} o{order} "
              f"gold={gold_alias} pos={pos}")
    f.close()
    print(f"Wrote -> {path}")


# ---------------------------------------------------------------------------
def _boot(vals, n=2000, seed=0):
    v = np.asarray(vals, float)
    if not len(v):
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return (round(float(v.mean()), 4),
            round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4))


def _per_item(recs, metric):
    acc = defaultdict(list)
    for r in recs:
        acc[r["item_id"]].append(metric(r))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def analyze(setname):
    src = SETS[setname]
    outdir = OUT / setname
    items = json.load(open(src / "items.json"))["items"]
    phen_rank = {it["item_id"]: it["phen_rank"] for it in items}

    recs = [json.loads(l) for l in open(outdir / "responses.jsonl")]
    scaf = [r for r in recs if r["condition"] == COND]

    s_rr = _per_item(scaf, _rr)
    h_rr = _per_item([r for r in recs if r["condition"] == "H-mask"], _rr)
    a_rr = _per_item([r for r in recs if r["condition"] == "A-mask"], _rr)
    p_rr = {i: 1.0 / phen_rank[i] for i in phen_rank}

    # the frozen run's MRR, for reference only -- NOT paired (different packets)
    frozen = [json.loads(l) for l in open(src / "responses.jsonl")]
    fz = {c: _per_item([r for r in frozen if r["condition"] == c], _rr)
          for c in ("A-mask", "H-mask")}

    ids = sorted(set(s_rr) & set(h_rr) & set(a_rr))
    n_parse_err = sum(1 for r in scaf if "_parse_error" in r["response"])

    # does the scaffold reproduce the phenotype-only ordering verbatim?
    exact_phen = sum(
        1 for r in scaf
        if isinstance(r["response"], dict)
        and r["response"].get("ranking") == r["phen_ranking"]
    )
    # which source does it SAY it weighted
    weighted = defaultdict(int)
    for r in scaf:
        if isinstance(r["response"], dict):
            weighted[r["response"].get("weighted_source", "?")] += 1

    rep = {
        "set": setname, "n_items": len(ids), "n_calls": len(scaf),
        "parse_errors": n_parse_err,
        "MRR": {
            "A-mask": round(float(np.mean([a_rr[i] for i in ids])), 4),
            "H-mask": round(float(np.mean([h_rr[i] for i in ids])), 4),
            "H-scaffold": round(float(np.mean([s_rr[i] for i in ids])), 4),
            "P": round(float(np.mean([p_rr[i] for i in ids])), 4),
        },
        "frozen_reference_MRR": {
            c: round(float(np.mean([v[i] for i in ids if i in v])), 4)
            for c, v in fz.items()
        },
        "paired": {
            "scaffold_minus_Hmask": _boot([s_rr[i] - h_rr[i] for i in ids]),
            "scaffold_minus_P": _boot([s_rr[i] - p_rr[i] for i in ids]),
            "scaffold_minus_Amask": _boot([s_rr[i] - a_rr[i] for i in ids]),
        },
        "selector": {
            "exact_ranking_matches_phenotype": exact_phen,
            "n_item_order_calls": len(scaf),
            "rr_ties_phenotype": sum(1 for i in ids if abs(s_rr[i] - p_rr[i]) < 1e-9),
            "exceeds_both_sources": sum(
                1 for i in ids if s_rr[i] > max(a_rr[i], p_rr[i]) + 1e-9),
            "self_reported_weighted_source": dict(weighted),
        },
    }
    json.dump(rep, open(outdir / "metrics.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))
    return rep


def _selftest():
    """Smallest check that the scorer math is right, no network."""
    recs = [{"item_id": "a", "gold_alias": "G1", "response": {"ranking": ["G1", "G2"]},
             "phen_ranking": ["G1", "G2"]},
            {"item_id": "a", "gold_alias": "G2", "response": {"ranking": ["G1", "G2"]},
             "phen_ranking": ["G2", "G1"]}]
    rr = _per_item(recs, _rr)
    assert abs(rr["a"] - 0.75) < 1e-9, rr          # mean(1/1, 1/2)
    m, lo, hi = _boot([0.1, 0.1, 0.1])
    assert (m, lo, hi) == (0.1, 0.1, 0.1), (m, lo, hi)
    exact = sum(1 for r in recs if r["response"]["ranking"] == r["phen_ranking"])
    assert exact == 1, exact
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze", "selftest"])
    ap.add_argument("--set", choices=list(SETS), default="fair")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.set, a.model, a.limit)
    elif a.cmd == "analyze":
        analyze(a.set)
    else:
        _selftest()
