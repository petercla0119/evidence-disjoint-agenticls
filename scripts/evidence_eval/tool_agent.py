#!/usr/bin/env python
"""Tool-using arm: the agent must FETCH its evidence instead of being handed it.

The static arms in scripts/evidence_eval/arbitration.py put annotation, phenotype and the
calibrated reliability table into one packet. That measures reading. This asks
the agentic question: given the same evidence behind tools the agent chooses to
call, does it seek the reliability of a source before trusting it?

Same items, same masking, same calibration table as artifacts/evidence_eval/arbitration --
only the delivery changes, so the arms are directly comparable.

Arms:
    agent_free   3 tools available, reliability never mentioned  -> does it look?
  (static `rel` in arbitration.py = reliability handed over unasked, the top rung)

An `agent_rel` arm (same tools, system prompt names the reliability tool) was cut
after 28 episodes: it called all three tools in the same order as agent_free every
single time, so naming the tool changed nothing. 100% unprompted retrieval in
agent_free already rules out "it didn't know the tool was there", which is the only
thing agent_rel was there to rule out. Its responses stay in responses.jsonl.

Usage:
    PYTHONHASHSEED=0 python scripts/evidence_eval/tool_agent.py run --model claude-sonnet-4-6
    python scripts/evidence_eval/tool_agent.py analyze
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arbitration as A                       # noqa: E402
import pilot_counterfactual as pilot          # noqa: E402
import reliability_calibration as rc          # noqa: E402

OUT_DIR = A.ART / "tool_agent"
ARMS = ["agent_free"]  # agent_rel cut -- see module docstring
N_ORDERS = A.N_ORDERS
MAX_TURNS = 8

# --- second model family (Azure), for the single-model limitation -------------
# NOTE: this deployment REJECTS temperature=0 ("Only the default (1) value is
# supported"), so its episodes are sampled, not deterministic. Vertex's noise
# floor does NOT transfer -- measure this model's own floor with `run --retest`
# and compare its effects against that, not against 0.0043.
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")  # your Azure OpenAI resource URL
AZURE_DEPLOYMENT = "gpt-5.6-luna"
AZURE_API_VERSION = "2024-12-01-preview"
PROVIDER_DIRS = {"vertex": "tool_agent", "azure": "tool_agent_gpt56"}


def _openai_tools():
    """Same three tools, in OpenAI function-calling shape."""
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["input_schema"]}}
            for t in TOOLS]


def _azure_client():
    from openai import AzureOpenAI
    key = os.environ["AZURE_OPENAI_API_KEY"]
    return AzureOpenAI(azure_endpoint=AZURE_ENDPOINT, api_key=key,
                       api_version=AZURE_API_VERSION)


# running count of chat-completion requests actually issued to Azure, so a
# re-run can be held to a stated API budget. REQ_CAP=None means unlimited.
REQUESTS = [0]
REQ_CAP = [None]


def _episode_azure(client, system, user, payloads):
    """One agent episode on Azure. Returns (final_text, tool_call_names)."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    calls = []
    for _ in range(MAX_TURNS):
        if REQ_CAP[0] is not None and REQUESTS[0] >= REQ_CAP[0]:
            raise SystemExit(f"ABORT: API request cap {REQ_CAP[0]} reached "
                             f"({REQUESTS[0]} issued)")
        REQUESTS[0] += 1
        r = client.chat.completions.create(
            model=AZURE_DEPLOYMENT, messages=msgs, tools=_openai_tools(),
            max_completion_tokens=2000,      # gpt-5.x rejects max_tokens
        )                                    # temperature omitted: only 1 allowed
        m = r.choices[0].message
        if not m.tool_calls:
            return (m.content or ""), calls
        msgs.append(m)
        for tc in m.tool_calls:
            calls.append(tc.function.name)
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": json.dumps(payloads.get(
                             tc.function.name, {"error": "unknown tool"}))})
    return "", calls

TOOLS = [
    {
        "name": "get_annotation_evidence",
        "description": "Pairwise annotation records among G0-G6 (functional_edge, "
                       "shared_pathway, shared_complex) plus a graph score per "
                       "candidate summarising 1-hop and 2-hop support for G0.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_phenotype_similarity",
        "description": "Morphological-perturbation similarity of each candidate to "
                       "G0 (higher = more similar).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_source_reliability",
        "description": "How often each evidence source's top pick is actually the "
                       "true partner, measured on an independent query-disjoint "
                       "calibration set and keyed on statistics computed for this "
                       "item. Includes the random-chance rate for comparison.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

SYS_BASE = (
    "You are ranking which candidate gene is the true functional partner of a "
    "query gene. All identities are anonymized (genes G0-G6, pathways P#, "
    "complexes C#). G0 is the query. You start with only the candidate list and "
    "each gene's degree. Every other piece of evidence must be fetched with the "
    "tools provided. Call whichever tools you judge useful, then give your final "
    "answer. Reason only from what the tools return; do not use outside knowledge "
    "(identities are hidden anyway).\n\n"
    "When you are done gathering evidence, reply with ONLY machine-readable JSON:\n"
    '  "ranking": [six candidate ids, best first],\n'
    '  "probabilities": {candidate id: probability}  (must sum to 1),\n'
    '  "first_choice_evidence": [record ids R# or source names supporting your top pick]\n'
    "No prose outside the JSON."
)

SYS_REL = SYS_BASE + (
    "\n\nNote: get_source_reliability tells you how much each source can be "
    "trusted for an item like this one. A source whose top pick is right at or "
    "below the random rate carries no information here."
)

USER = ("Query gene: G0. Candidates: {cands}.\n"
        "Node degrees (JSON): {nodes}\n\n"
        "Rank all six candidates from most to least likely to be G0's true "
        "functional partner.")


def _payloads(item, order, ctx, cal):
    """The three tool return values, carved out of the same masked packet the
    static arms receive. Nothing here is generated fresh -- identical evidence,
    different delivery."""
    packet, gold_alias, audit = pilot.make_packet(item, "H-mask", order, ctx)
    score = rc._ann_scores(packet)
    return {
        "get_annotation_evidence": {
            "edges": packet["edges"],
            "annotation_graph_score": {k: round(v, 3) for k, v in score.items()},
        },
        "get_phenotype_similarity": packet["phenotype_similarity"],
        "get_source_reliability": {
            "measured_on": cal["measured_on"],
            "keys": cal["keys"],
            "random_candidate_is_true_partner": cal["random_candidate_is_true_partner"],
            "random_ranking_expected_reciprocal_rank": cal["random_ranking_expected_reciprocal_rank"],
            "table": cal["table"],
            "this_item": {
                "phenotype_gap": round(item["gap"], 4),
                "annotation_margin": round(item["margin"], 4),
                "matching_row": f"{A._bin(item['gap'], A.GAP_BINS)} | "
                                f"{A._bin(item['margin'], A.MARGIN_BINS)}",
            },
        },
    }, packet, gold_alias, audit


def run(model, limit=None, smoke=False, provider="vertex", retest=False,
        cell=None, out=None, no_temperature=False):
    cal = json.load(open(A.OUT_DIR / "calibration.json"))
    items = json.load(open(A.OUT_DIR / "items.json"))["items"]
    ctx = pilot._ctx()
    if provider == "azure":
        client, model = _azure_client(), AZURE_DEPLOYMENT
    else:
        client = pilot.get_client()

    out_dir = A.ART / PROVIDER_DIRS[provider]
    out_dir.mkdir(parents=True, exist_ok=True)
    # retest writes to its own file so it can never contaminate the primary run
    out_path = Path(out) if out else \
        out_dir / ("responses_retest.jsonl" if retest else "responses.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["arm"], r["item_id"], r["order"]))

    jobs = [(arm, it, o) for it in items for arm in ARMS for o in range(N_ORDERS)
            if cell is None or it["cell"] == cell]
    if smoke:
        jobs = [j for j in jobs if j[1]["item_id"] in {"conf00", "cont00"}][:4]
    jobs = [j for j in jobs if (j[0], j[1]["item_id"], j[2]) not in done]
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} agent episodes to run ({len(done)} already done) on {model}")
    if provider == "vertex":
        print("temperature: " + ("OMITTED (API default applies)" if no_temperature
                                 else "extra_body={'temperature': 0.0}"))

    f = open(out_path, "a")
    for n, (arm, it, order) in enumerate(jobs, 1):
        payloads, packet, gold_alias, audit = _payloads(it, order, ctx, cal)
        system = SYS_BASE if arm == "agent_free" else SYS_REL
        msgs = [{"role": "user", "content": USER.format(
            cands=", ".join(packet["candidates"]),
            nodes=json.dumps(packet["nodes"]))}]

        calls, text = [], ""
        if provider == "azure":
            text, calls = _episode_azure(
                client, system, msgs[0]["content"], payloads)
        else:
            for _ in range(MAX_TURNS):
                if REQ_CAP[0] is not None and REQUESTS[0] >= REQ_CAP[0]:
                    raise SystemExit(f"ABORT: API request cap {REQ_CAP[0]} reached "
                                     f"({REQUESTS[0]} issued)")
                REQUESTS[0] += 1
                # no_temperature: send no temperature at all, so the API default
                # applies (ecological validity). Default stays T=0 so prior runs
                # stay reproducible.
                kw = {} if no_temperature else {"extra_body": {"temperature": 0.0}}
                msg = client.messages.create(
                    model=model, max_tokens=2000, system=system, tools=TOOLS,
                    messages=msgs, **kw,
                )
                msgs.append({"role": "assistant", "content": msg.content})
                uses = [b for b in msg.content if b.type == "tool_use"]
                if not uses:
                    text = "".join(b.text for b in msg.content if b.type == "text")
                    break
                results = []
                for b in uses:
                    calls.append(b.name)
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": json.dumps(payloads.get(b.name, {"error": "unknown tool"}))})
                msgs.append({"role": "user", "content": results})
            else:
                text = ""  # hit MAX_TURNS without settling

        try:
            parsed = pilot.parse_response(text)
        except Exception as e:
            parsed = {"_parse_error": str(e), "_raw": text[:500]}

        rec = dict(arm=arm, item_id=it["item_id"], cell=it["cell"], order=order,
                   gold_alias=gold_alias, phen_rank=it["phen_rank"],
                   ann_rank=it["ann_rank"], gap=it["gap"], margin=it["margin"],
                   model=model, audit=audit, tool_calls=calls, response=parsed)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        rk = parsed.get("ranking", []) if isinstance(parsed, dict) else []
        pos = (rk.index(gold_alias) + 1) if gold_alias in rk else "-"
        print(f"[{n}/{len(jobs)}] {arm:10s} {it['item_id']:7s} o{order} {it['cell']:8s} "
              f"gold={gold_alias} pos={pos} tools={'+'.join(c.replace('get_', '')[:4] for c in calls) or 'none'}"
              f" | api_requests={REQUESTS[0]}")
    f.close()
    print(f"wrote -> {out_path}")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
def _rr(rank):
    return 1.0 / rank if rank else 0.0


def analyze(provider="vertex"):
    out_dir = A.ART / PROVIDER_DIRS[provider]
    items = {it["item_id"]: it for it in json.load(open(A.OUT_DIR / "items.json"))["items"]}
    rows = [json.loads(l) for l in open(out_dir / "responses.jsonl")]
    ctx = pilot._ctx()

    # GUARD. make_packet derives the candidate permutation from
    # hash((item_id, order)), and hash() of a str under PYTHONHASHSEED=0 is stable
    # within a Python version but NOT across versions (system 3.13 vs env 3.10).
    # Recomputing packets on the wrong interpreter silently returns different
    # G#-aliases than the run used, which makes exact-ranking agreement collapse to
    # 0 and look like a real null. RR-based stats are alias-independent and safe;
    # anything recomputed from a packet is not. Crash instead of lying.
    bad = sum(1 for r in rows if r["item_id"] in items
              and A._source_rankings(items[r["item_id"]], r["order"], ctx)[2] != r["gold_alias"])
    if bad:
        raise SystemExit(
            f"alias mismatch on {bad}/{len(rows)} rows -- recomputed packets do not "
            f"match the run. Re-run as:\n"
            f"  conda activate ops-experimental && PYTHONHASHSEED=0 python "
            f"scripts/evidence_eval/tool_agent.py analyze")

    # per (arm, item): RR averaged over orders; tool usage unioned over orders
    rr = defaultdict(list)
    used_rel = defaultdict(list)
    n_calls = defaultdict(list)
    parse_err = 0
    for r in rows:
        resp = r["response"]
        if not isinstance(resp, dict) or "ranking" not in resp:
            parse_err += 1
            continue
        rk = resp["ranking"]
        rank = rk.index(r["gold_alias"]) + 1 if r["gold_alias"] in rk else None
        rr[(r["arm"], r["item_id"])].append(_rr(rank))
        used_rel[(r["arm"], r["item_id"])].append(
            "get_source_reliability" in r["tool_calls"])
        n_calls[(r["arm"], r["item_id"])].append(len(r["tool_calls"]))

    item_rr = {k: float(np.mean(v)) for k, v in rr.items()}
    item_rel = {k: float(np.mean(v)) for k, v in used_rel.items()}

    # phenotype-only ranker: a deterministic morphology baseline, NOT an oracle --
    # it has no privileged access to the answer, it just ranks by cosine.
    P = {i: 1.0 / it["phen_rank"] for i, it in items.items()}
    # NOT 1/ann_rank -- ann_rank is an average rank, and 1/E[rank] understates
    # E[1/rank] for a tied gold. It scored the 6-way-tied control annotation
    # ranker 0.281, below the 0.408 chance rate. See arbitration._tie_aware_baseline_rr.
    _tie = A._tie_aware_baseline_rr()
    AN = {i: _tie[i] for i in items}

    report = {"n_responses": len(rows), "parse_errors": parse_err, "cells": {}}
    for cell in ("conflict", "control", "all"):
        ids = [i for i, it in items.items() if cell == "all" or it["cell"] == cell]
        block = {"n_items": len(ids), "phenotype_only_ranker_MRR": round(float(np.mean([P[i] for i in ids])), 4),
                 "annotation_only_ranker_MRR": round(float(np.mean([AN[i] for i in ids])), 4),
                 "best_source_oracle_MRR": round(float(np.mean([max(P[i], AN[i]) for i in ids])), 4),
                 "random_ranking_MRR": round(A.RANDOM_RR, 4), "arms": {}}
        for arm in ARMS:
            have = [i for i in ids if (arm, i) in item_rr]
            if not have:
                continue
            vals = [item_rr[(arm, i)] for i in have]
            diffP = [item_rr[(arm, i)] - P[i] for i in have]
            diffA = [item_rr[(arm, i)] - AN[i] for i in have]
            # exact full-ranking agreement: does it copy a source wholesale?
            ep = ea = tot = 0
            for r in rows:
                if r["arm"] != arm or r["item_id"] not in set(have):
                    continue
                resp = r["response"]
                if not isinstance(resp, dict) or "ranking" not in resp:
                    continue
                pr, ar, _ = A._source_rankings(items[r["item_id"]], r["order"], ctx)
                tot += 1
                ep += resp["ranking"] == pr
                ea += resp["ranking"] == ar
            block["arms"][arm] = {
                "n_items": len(have),
                "MRR": round(float(np.mean(vals)), 4),
                "MRR_CI": A._boot(vals),
                # paired bootstrap over items -- the correct equivalence test. Do NOT
                # compare an arm difference against the run-to-run replication SE.
                "minus_phenotype_only_ranker": round(float(np.mean(diffP)), 4),
                "minus_phenotype_only_ranker_CI": A._boot(diffP),
                "minus_annotation_only_ranker": round(float(np.mean(diffA)), 4),
                "minus_annotation_only_ranker_CI": A._boot(diffA),
                "rr_ties_phenotype_only_ranker": sum(1 for d in diffP if abs(d) < 1e-9),
                "rr_ties_annotation_only_ranker": sum(1 for d in diffA if abs(d) < 1e-9),
                "exact_ranking_matches_phenotype": ep,
                "exact_ranking_matches_annotation": ea,
                "n_calls_scored": tot,
                "exceeds_best_source_oracle": sum(
                    1 for i in have if item_rr[(arm, i)] > max(P[i], AN[i]) + 1e-9),
                "queried_reliability_rate": round(float(np.mean(
                    [item_rel[(arm, i)] for i in have if (arm, i) in item_rel])), 3),
                "mean_tool_calls": round(float(np.mean(
                    [np.mean(n_calls[(arm, i)]) for i in have if (arm, i) in n_calls])), 2),
            }
        report["cells"][cell] = block

    # does querying reliability change the answer? (within agent_free, conflict cell)
    ids = [i for i, it in items.items() if it["cell"] == "conflict"]
    q = [item_rr[("agent_free", i)] - P[i] for i in ids
         if ("agent_free", i) in item_rr and item_rel[("agent_free", i)] > 0.5]
    nq = [item_rr[("agent_free", i)] - P[i] for i in ids
          if ("agent_free", i) in item_rr and item_rel[("agent_free", i)] <= 0.5]
    report["conflict_queried_vs_not"] = {
        "n_queried": len(q), "n_not_queried": len(nq),
        "queried_minus_phenotype_only_ranker": round(float(np.mean(q)), 4) if q else None,
        "not_queried_minus_phenotype_only_ranker": round(float(np.mean(nq)), 4) if nq else None,
    }

    json.dump(report, open(out_dir / "metrics.json", "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote -> {out_dir/'metrics.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--provider", choices=["vertex", "azure"], default="vertex")
    ap.add_argument("--retest", action="store_true",
                    help="write to responses_retest.jsonl -- for measuring a "
                         "sampled model's own noise floor")
    ap.add_argument("--cell", choices=["conflict", "control"],
                    help="restrict the re-run to one cell")
    ap.add_argument("--out", help="explicit responses path (overrides --retest)")
    ap.add_argument("--max-requests", type=int,
                    help="hard cap on chat-completion requests; abort at it")
    ap.add_argument("--no-temperature", action="store_true",
                    help="omit temperature entirely so the API default applies "
                         "(vertex only; default remains T=0)")
    a = ap.parse_args()
    if a.cmd == "run":
        REQ_CAP[0] = a.max_requests
        run(a.model, limit=a.limit, smoke=a.smoke, provider=a.provider,
            retest=a.retest, cell=a.cell, out=a.out,
            no_temperature=a.no_temperature)
    else:
        analyze(provider=a.provider)


if __name__ == "__main__":
    main()
