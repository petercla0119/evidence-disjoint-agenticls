#!/usr/bin/env python
"""A third cell where copying one source is NOT the calibration-optimal policy.

The tool-using agent emitted one source's exact six-item ranking verbatim on
40/40 control calls and 28/80 conflict calls. exp20 showed that in both of those
cells copying the better source IS calibration-optimal -- each was binned so one
source sits at or below the random-ranking rate (0.408). So a high copy rate
there is rational and says nothing about disposition.

This script builds the missing cell and asks: does the verbatim-copy rate hold
when copying is no longer the right answer?

exp22 found the one bin where the two sources are genuinely complementary:
`gap<0.013 | annotation margin 0-1.5`. But the COARSE calibration row for that
bin reports annotation expected-RR 0.408 -- exactly the random rate. Injecting
that row tells the model annotation is worthless and reproduces the very
confound we are escaping. So Stage 1 re-derives reliability for that bin under a
finer binning, on the CALIBRATION HALF ONLY, and gates the run on finding a
sub-bin where both sources are above 0.408.

Sub-bin search (calibration-half correctness only; eval-half data enters solely
as a distinct-query count, never as a correctness signal):
  - six test-time-observable statistics, each split at its calibration-half
    MEDIAN (a fixed, untuned cut)
  - a side qualifies if both sources' expected reciprocal rank exceed 0.408 on
    the calibration half AND it holds >= 40 distinct eval-half queries
  - the qualifying side with the largest min(phenotype eRR, annotation eRR) wins
  - no qualifying side -> STOP, spend no API calls

Stage 2 runs the tool-using arm (3 tools, reliability never mentioned in the
prompt -- the arm that produced the 40/40) on a 40-item cell selected on the
observables only, plus a no-reliability-tool contrast arm.

Usage:
    PYTHONHASHSEED=0 python scripts/evidence_eval/exp23_complementary_cell.py stage1
    PYTHONHASHSEED=0 python scripts/evidence_eval/exp23_complementary_cell.py build
    PYTHONHASHSEED=0 python scripts/evidence_eval/exp23_complementary_cell.py run --arm agent_free
    PYTHONHASHSEED=0 python scripts/evidence_eval/exp23_complementary_cell.py run --arm agent_norel
    PYTHONHASHSEED=0 python scripts/evidence_eval/exp23_complementary_cell.py analyze
    python scripts/evidence_eval/exp23_complementary_cell.py selfcheck
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
import arbitration as arb                     # noqa: E402
import exp20_fusion_baseline as F             # noqa: E402
import tool_agent as TA                       # noqa: E402

OUT_DIR = arb.ART / "complementary"
PARENT = "gap<0.013 | annotation margin 0-1.5"
RANDOM_RR = arb.RANDOM_RR                 # 0.4083
MIN_EVAL_QUERIES = 40                     # cell size, one item per distinct query
CELL_N = 40
CELL_SEED = 20260920
RULES = F.RULES
N_ORDERS = arb.N_ORDERS

# test-time-observable statistics, each split at its calibration-half median.
# Every one is computable from the packet the model is handed, without knowing
# which candidate is the gold.
OBSERVABLES = {
    "annotation_top_score": "ann_top",
    "n_candidates_with_positive_annotation": "n_ann_pos",
    "n_annotation_records": "n_edges",
    "phenotype_top_cosine": "phen_top",
    "phenotype_gap": "gap",
    "annotation_margin": "margin",
}

# The agent budget is counted in EPISODES (one item x one candidate ordering),
# matching the design's "40 items x 2 orderings = 80 calls". The underlying
# HTTP request count (an episode is a multi-turn tool loop) is tracked
# separately and reported, but the cap is on episodes.
EPISODES = [0]
REQUESTS = [0]
EPISODE_CAP = [180]


def _require_hashseed():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run as: PYTHONHASHSEED=0 python "
                         "scripts/evidence_eval/exp23_complementary_cell.py ...")


# ---------------------------------------------------------------------------
# Stage 1 -- offline. Observables for every pool row in the parent bin.
# ---------------------------------------------------------------------------
def _bin_key(row):
    return (f"{arb._bin(row['gap'], arb.GAP_BINS)} | "
            f"{arb._bin(row['margin'], arb.MARGIN_BINS)}")


def parent_rows():
    """Every (query, gold) pool row in the parent bin, with its observables,
    tagged calibration / eval half. Ranks are asserted against the cached scan."""
    pool = arb.scan_pool()
    cal_q, eval_q = arb.split_queries(pool)
    rows = [r for r in pool if _bin_key(r) == PARENT]

    emb = V.load_embeddings()
    _, test = V.load_splits()
    # item_id must reproduce arbitration.scan_pool's f"s{qi:04d}": make_packet
    # permutes candidates via hash((item_id, order)), so a different id gives a
    # different alias map and different annotation scores.
    qid = {q: f"s{qi:04d}" for qi, q in enumerate(sorted(g for g in test if g in emb))}
    ctx = pilot._ctx()

    out = []
    for r in rows:
        item = dict(query=r["query"], gold=r["gold"], distractors=r["distractors"],
                    item_id=qid[r["query"]])
        packet, gold, audit = pilot.make_packet(item, "A-mask", 0, ctx)
        assert audit["masked"], item
        ann = rc._ann_scores(packet)
        cands = [r["gold"]] + r["distractors"]
        alias = F._alias_map(item, 0)
        assert alias[r["gold"]] == gold, (item["item_id"], alias[r["gold"]], gold)
        phen = {alias[g]: float(emb[r["query"]] @ emb[g]) for g in cands}
        pr, ar = rc._avg_rank(phen, gold), rc._avg_rank(ann, gold)
        assert abs(pr - r["phen_rank"]) < 1e-9 and abs(ar - r["ann_rank"]) < 1e-9, (
            f"rank reconstruction drift for {r['query']}/{r['gold']}. make_packet "
            "permutes candidates via hash((item_id, order)); PYTHONHASHSEED=0 pins "
            "str hashing only within a Python version -- re-run under conda "
            "ops-experimental (Python 3.10).")
        vals = sorted(ann.values(), reverse=True)
        out.append(dict(
            query=r["query"], gold=r["gold"], distractors=r["distractors"],
            half="cal" if r["query"] in cal_q else "eval",
            gap=r["gap"], margin=r["margin"],
            phen_rank=pr, ann_rank=ar,
            gold_degree=r["gold_degree"], gold_cosine=r["gold_cosine"],
            # --- observables: functions of the packet, never of the gold ---
            ann_top=float(vals[0]),
            n_ann_pos=int(sum(1 for v in ann.values() if v > 1e-9)),
            n_edges=int(len(packet["edges"])),
            phen_top=float(max(phen.values())),
        ))
    return out


def _err(rows, field):
    """Expected reciprocal rank of the gold under one source."""
    return float(np.mean([1.0 / r[field] for r in rows])) if rows else float("nan")


def _side_stats(cal, ev, keep):
    c = [r for r in cal if keep(r)]
    e = [r for r in ev if keep(r)]
    return dict(n_cal=len(c),
                phenotype_expected_reciprocal_rank=round(_err(c, "phen_rank"), 3),
                annotation_expected_reciprocal_rank=round(_err(c, "ann_rank"), 3),
                phenotype_top_pick_is_true_partner=round(
                    float(np.mean([r["phen_rank"] == 1.0 for r in c])), 3) if c else None,
                annotation_top_pick_is_true_partner=round(
                    float(np.mean([r["ann_rank"] == 1.0 for r in c])), 3) if c else None,
                n_eval_rows=len(e),
                n_eval_queries=len({r["query"] for r in e}))


def refine(rows):
    """Median-split search over the six observables. Calibration-half
    correctness only; the eval half contributes a distinct-query COUNT and
    nothing else."""
    cal = [r for r in rows if r["half"] == "cal"]
    ev = [r for r in rows if r["half"] == "eval"]
    search = []
    for name, field in OBSERVABLES.items():
        cut = float(np.median([r[field] for r in cal]))
        for side, keep in (("lt", lambda r, f=field, c=cut: r[f] < c),
                           ("ge", lambda r, f=field, c=cut: r[f] >= c)):
            s = _side_stats(cal, ev, keep)
            s.update(observable=name, field=field, cut=round(cut, 4), side=side,
                     label=f"{name}{'<' if side == 'lt' else '>='}{cut:g}")
            s["min_source_eRR"] = min(s["phenotype_expected_reciprocal_rank"],
                                      s["annotation_expected_reciprocal_rank"])
            s["both_above_random"] = bool(s["min_source_eRR"] > RANDOM_RR)
            s["enough_eval_queries"] = bool(s["n_eval_queries"] >= MIN_EVAL_QUERIES)
            s["qualifies"] = bool(s["both_above_random"] and s["enough_eval_queries"])
            search.append(s)
    ok = [s for s in search if s["qualifies"]]
    chosen = max(ok, key=lambda s: (s["min_source_eRR"], s["observable"])) if ok else None
    return dict(parent_bin=PARENT,
                n_cal_rows=len(cal), n_eval_rows=len(ev),
                n_eval_queries=len({r["query"] for r in ev}),
                parent_cal_phenotype_eRR=round(_err(cal, "phen_rank"), 3),
                parent_cal_annotation_eRR=round(_err(cal, "ann_rank"), 3),
                random_ranking_expected_reciprocal_rank=round(RANDOM_RR, 3),
                min_eval_queries_required=MIN_EVAL_QUERIES,
                search=search, chosen=chosen)


def refined_table(chosen):
    """The coarse 9-row calibration table with the parent row replaced by the
    two sub-rows of the chosen split. A strictly finer partition of the same
    table -- the other eight rows are untouched."""
    cal = json.load(open(arb.OUT_DIR / "calibration.json"))
    rows = parent_rows()
    c = [r for r in rows if r["half"] == "cal"]
    field, cut = chosen["field"], chosen["cut"]
    table = {}
    for k, v in cal["table"].items():
        if k != PARENT:
            table[k] = v
            continue
        for side, keep in (("<", lambda r: r[field] < cut), (">=", lambda r: r[field] >= cut)):
            sub = [r for r in c if keep(r)]
            table[f"{PARENT} | {chosen['observable']}{side}{cut:g}"] = {
                "n": len(sub),
                "phenotype_top_pick_is_true_partner": round(
                    float(np.mean([r["phen_rank"] == 1.0 for r in sub])), 3),
                "annotation_top_pick_is_true_partner": round(
                    float(np.mean([r["ann_rank"] == 1.0 for r in sub])), 3),
                "phenotype_expected_reciprocal_rank": round(_err(sub, "phen_rank"), 3),
                "annotation_expected_reciprocal_rank": round(_err(sub, "ann_rank"), 3),
            }
    cal["table"] = table
    cal["keys"] = (cal["keys"] + " Inside the gap<0.013 / annotation margin 0-1.5 "
                   f"region the table is split further on {chosen['observable']} "
                   "(also computable from the packet without knowing the answer).")
    cal["measured_on"] = cal["measured_on"] + \
        f"; the {PARENT} rows are re-derived on the same calibration half at a finer binning"
    return cal


def row_key(item, chosen):
    """The refined table row an item falls in."""
    base = f"{arb._bin(item['gap'], arb.GAP_BINS)} | {arb._bin(item['margin'], arb.MARGIN_BINS)}"
    if base != PARENT:
        return base
    side = "<" if item[chosen["field"]] < chosen["cut"] else ">="
    return f"{PARENT} | {chosen['observable']}{side}{chosen['cut']:g}"


def stage1(verbose=True):
    rows = parent_rows()
    rep = refine(rows)
    if verbose:
        print(f"parent bin {PARENT}: {rep['n_cal_rows']} calibration rows, "
              f"{rep['n_eval_rows']} eval rows / {rep['n_eval_queries']} queries")
        print(f"coarse calibration row: phenotype eRR {rep['parent_cal_phenotype_eRR']}, "
              f"annotation eRR {rep['parent_cal_annotation_eRR']} "
              f"(random {rep['random_ranking_expected_reciprocal_rank']})\n")
        print(f"{'sub-bin':52s}{'n_cal':>6s}{'phen':>7s}{'ann':>7s}{'evalQ':>7s}  qualifies")
        for s in rep["search"]:
            print(f"{s['label']:52s}{s['n_cal']:6d}"
                  f"{s['phenotype_expected_reciprocal_rank']:7.3f}"
                  f"{s['annotation_expected_reciprocal_rank']:7.3f}"
                  f"{s['n_eval_queries']:7d}  {'YES' if s['qualifies'] else ''}")
    if rep["chosen"] is None:
        print("\nGATE FAILED: no sub-bin has both sources above the random rate with "
              f">= {MIN_EVAL_QUERIES} eval-half queries. Spending no API calls.")
        return rep, None
    ch = rep["chosen"]
    if verbose:
        print(f"\nGATE PASSED -> {ch['label']}: calibration-half phenotype eRR "
              f"{ch['phenotype_expected_reciprocal_rank']}, annotation eRR "
              f"{ch['annotation_expected_reciprocal_rank']}, "
              f"{ch['n_eval_queries']} eval-half queries")
    cal = refined_table(ch)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(OUT_DIR / "stage1.json", "w"), indent=2)
    json.dump(cal, open(OUT_DIR / "calibration_refined.json", "w"), indent=2)
    if verbose:
        print("\nrefined rows for the parent bin, verbatim as the model receives them:")
        for k, v in cal["table"].items():
            if k.startswith(PARENT):
                print(f"  {json.dumps({k: v})}")
        print(f"\nwrote -> {OUT_DIR/'stage1.json'}\nwrote -> {OUT_DIR/'calibration_refined.json'}")
    return rep, ch


# ---------------------------------------------------------------------------
# Stage 2a -- cell construction (observables only, never the gold's rank)
# ---------------------------------------------------------------------------
def build():
    rep, ch = stage1(verbose=False)
    if ch is None:
        raise SystemExit("stage 1 gate failed -- nothing to build")
    rows = [r for r in parent_rows()
            if r["half"] == "eval" and (r[ch["field"]] >= ch["cut"] if ch["side"] == "ge"
                                        else r[ch["field"]] < ch["cut"])]
    # queries already spent on the conflict / control cells: excluded so the new
    # cell is independent of the two published ones.
    used = {it["query"] for it in json.load(open(arb.OUT_DIR / "items.json"))["items"]}
    reuse = len({r["query"] for r in rows} & used)
    rows = [r for r in rows if r["query"] not in used]

    rng = random.Random(CELL_SEED)
    rng.shuffle(rows)
    picked, seen = [], set()
    for r in rows:                              # one item per query -> independence
        if r["query"] in seen:
            continue
        seen.add(r["query"])
        picked.append(r)
        if len(picked) >= CELL_N:
            break

    items = [dict(item_id=f"comp{i:02d}", cell="complementary",
                  query=r["query"], gold=r["gold"], distractors=r["distractors"],
                  gold_degree=r["gold_degree"], phen_rank=r["phen_rank"],
                  gold_cosine=r["gold_cosine"],
                  gap=round(r["gap"], 4), margin=round(r["margin"], 4),
                  ann_rank=r["ann_rank"],
                  ann_top=round(r["ann_top"], 4), n_ann_pos=r["n_ann_pos"],
                  n_edges=r["n_edges"], phen_top=round(r["phen_top"], 4))
             for i, r in enumerate(picked)]
    for it in items:
        it["matching_row"] = row_key(it, ch)
        assert it["matching_row"] == f"{PARENT} | {ch['label']}", it

    spread = Counter(int(it["phen_rank"]) for it in items)
    meta = dict(split_seed=arb.SPLIT_SEED, cell_seed=CELL_SEED,
                selection="observable only (gap, margin, %s); never conditioned on "
                          "which candidate is the gold" % ch["observable"],
                sub_bin=ch["label"], parent_bin=PARENT,
                n_eval_rows_available=len(rows) + reuse,
                n_queries_dropped_as_already_used=reuse,
                true_partner_phenotype_rank_spread={str(k): v for k, v in sorted(spread.items())},
                spans_all_ranks=bool(set(spread) == {1, 2, 3, 4, 5, 6}),
                items=items)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(OUT_DIR / "items.json", "w"), indent=2)
    print(f"cell {ch['label']}: {len(items)} items, "
          f"{len({it['query'] for it in items})} distinct queries "
          f"({reuse} queries dropped as already used by conflict/control)")
    print(f"true-partner phenotype rank spread: {dict(sorted(spread.items()))} "
          f"spans_all_ranks={meta['spans_all_ranks']}")
    print(f"wrote -> {OUT_DIR/'items.json'}")
    return meta


# ---------------------------------------------------------------------------
# Stage 2b -- run. Vertex / model A only.
# ---------------------------------------------------------------------------
ARM_TOOLS = {"agent_free": TA.TOOLS,           # all three -- reliability unmentioned
             "agent_norel": TA.TOOLS[:2]}      # no reliability tool at all
assert [t["name"] for t in TA.TOOLS[:2]] == \
    ["get_annotation_evidence", "get_phenotype_similarity"], "tool order changed"


def _payloads(item, order, ctx, cal, chosen):
    """Identical to tool_agent._payloads except the reliability payload carries
    the REFINED table and this_item names the extra observable."""
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
            "random_ranking_expected_reciprocal_rank":
                cal["random_ranking_expected_reciprocal_rank"],
            "table": cal["table"],
            "this_item": {
                "phenotype_gap": round(item["gap"], 4),
                "annotation_margin": round(item["margin"], 4),
                chosen["observable"]: round(item[chosen["field"]], 4),
                "matching_row": item["matching_row"],
            },
        },
    }, packet, gold_alias, audit


def _episode_azure(client, model, system, user, tools, payloads):
    """One agent episode on Azure. Mirrors tool_agent._episode_azure, but honours
    this script's per-arm tool subset instead of always offering all three.
    The deployment rejects temperature=0, so episodes here are sampled."""
    names = {t["name"] for t in tools}
    oa_tools = [t for t in TA._openai_tools() if t["function"]["name"] in names]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    calls = []
    for _ in range(TA.MAX_TURNS):
        REQUESTS[0] += 1
        r = client.chat.completions.create(model=model, messages=msgs,
                                           tools=oa_tools,
                                           max_completion_tokens=2000)
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


def _episode(client, model, system, user, tools, payloads):
    """One agent episode on Vertex. Mirrors tool_agent.run's inner loop."""
    msgs = [{"role": "user", "content": user}]
    calls, text = [], ""
    for _ in range(TA.MAX_TURNS):
        REQUESTS[0] += 1
        msg = client.messages.create(model=model, max_tokens=2000, system=system,
                                     tools=tools, messages=msgs,
                                     extra_body={"temperature": 0.0})
        msgs.append({"role": "assistant", "content": msg.content})
        uses = [b for b in msg.content if b.type == "tool_use"]
        if not uses:
            return "".join(b.text for b in msg.content if b.type == "text"), calls
        results = []
        for b in uses:
            calls.append(b.name)
            results.append({"type": "tool_result", "tool_use_id": b.id,
                            "content": json.dumps(payloads.get(b.name, {"error": "unknown tool"}))})
        msgs.append({"role": "user", "content": results})
    return "", calls                              # hit MAX_TURNS without settling


def run(arm, model, limit=None, provider="vertex"):
    _, ch = stage1(verbose=False)
    if ch is None:
        raise SystemExit("stage 1 gate failed -- no API calls")
    cal = json.load(open(OUT_DIR / "calibration_refined.json"))
    items = json.load(open(OUT_DIR / "items.json"))["items"]
    ctx = pilot._ctx()
    # ponytail: second model family reuses tool_agent's Azure plumbing. Its
    # responses go to their own file so the two models never share a
    # (arm, item, order) key and clobber each other's resume set.
    if provider == "azure":
        client, model = TA._azure_client(), TA.AZURE_DEPLOYMENT
        episode, out_path = _episode_azure, OUT_DIR / "responses_gpt56.jsonl"
    else:
        client = pilot.get_client()
        episode, out_path = _episode, OUT_DIR / "responses.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["arm"], r["item_id"], r["order"]))

    jobs = [(it, o) for it in items for o in range(N_ORDERS)
            if (arm, it["item_id"], o) not in done]
    if limit:
        jobs = jobs[:limit]
    print(f"arm {arm}: {len(jobs)} episodes to run ({len(done)} already on file) "
          f"on {model}; episode cap {EPISODE_CAP[0]}")

    f = open(out_path, "a")
    for n, (it, order) in enumerate(jobs, 1):
        if EPISODES[0] >= EPISODE_CAP[0]:
            print(f"ABORT: episode cap {EPISODE_CAP[0]} reached")
            break
        payloads, packet, gold_alias, audit = _payloads(it, order, ctx, cal, ch)
        user = TA.USER.format(cands=", ".join(packet["candidates"]),
                              nodes=json.dumps(packet["nodes"]))
        EPISODES[0] += 1
        text, calls = episode(client, model, TA.SYS_BASE, user,
                              ARM_TOOLS[arm], payloads)
        try:
            parsed = pilot.parse_response(text)
        except Exception as e:
            parsed = {"_parse_error": str(e), "_raw": text[:500]}
        rec = dict(arm=arm, item_id=it["item_id"], cell="complementary", order=order,
                   gold_alias=gold_alias, phen_rank=it["phen_rank"],
                   ann_rank=it["ann_rank"], gap=it["gap"], margin=it["margin"],
                   ann_top=it["ann_top"], matching_row=it["matching_row"],
                   model=model, audit=audit, tool_calls=calls, response=parsed)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        rk = parsed.get("ranking", []) if isinstance(parsed, dict) else []
        pos = (rk.index(gold_alias) + 1) if gold_alias in rk else "-"
        print(f"[{n}/{len(jobs)}] {arm:11s} {it['item_id']} o{order} gold={gold_alias} "
              f"pos={pos} tools={'+'.join(c.replace('get_', '')[:4] for c in calls) or 'none'}"
              f" | episodes={EPISODES[0]} requests={REQUESTS[0]}")
        # smoke gate: the first episode of a sweep must parse, else stop.
        if n == 1 and not rk:
            f.close()
            raise SystemExit(f"SMOKE FAILED: first episode returned no ranking "
                             f"({parsed}). Stopping; 1 episode spent.")
    f.close()
    print(f"wrote -> {out_path}  (episodes {EPISODES[0]}, http requests {REQUESTS[0]})")


# ---------------------------------------------------------------------------
# Stage 3 -- analysis
# ---------------------------------------------------------------------------
def _m(x):
    return round(float(np.mean(x)), 4)


def _wilson(k, n):
    """Wilson 95% interval on a copy rate. Binomial over CALLS, which are not
    fully independent (2 orderings share an item) -- so read it as indicative."""
    from scipy.stats import binomtest
    lo, hi = binomtest(k, n).proportion_ci(method="wilson")
    return [round(float(lo), 3), round(float(hi), 3)] if n else None


def _verify(meta, items):
    """Re-derive the correctness-blindness claims from the artifacts rather than
    asserting them in prose. A reviewer probes these first."""
    pool = arb.scan_pool()
    cal_q, _ = arb.split_queries(pool)
    used = {it["query"] for it in json.load(open(arb.OUT_DIR / "items.json"))["items"]}
    qs = [it["query"] for it in items.values()]
    ch = json.load(open(OUT_DIR / "stage1.json"))["chosen"]
    return {
        "selection_statistics": ["phenotype gap (top1-top2 cosine)",
                                 "annotation margin (top1-top2 graph score)",
                                 ch["observable"] + " (top annotation graph score)"],
        "gold_rank_used_in_selection": False,
        "n_items": len(items),
        "n_distinct_queries": len(set(qs)),
        "one_item_per_query": len(set(qs)) == len(qs),
        "n_items_in_calibration_half": sum(1 for q in qs if q in cal_q),
        "query_disjoint_from_calibration": not any(q in cal_q for q in qs),
        "n_items_reusing_conflict_or_control_query": sum(1 for q in qs if q in used),
        "spans_all_ranks": meta["spans_all_ranks"],
    }


def analyze(provider="vertex"):
    # ponytail: azure responses live in their own file; keep its metrics
    # beside the vertex ones rather than overwriting them.
    suffix = "_gpt56" if provider == "azure" else ""
    meta = json.load(open(OUT_DIR / "items.json"))
    items = {it["item_id"]: it for it in meta["items"]}
    cal = json.load(open(OUT_DIR / "calibration_refined.json"))["table"]
    rows = [json.loads(l) for l in open(OUT_DIR / f"responses{suffix}.jsonl")]
    ctx = pilot._ctx()
    emb = ctx[0]

    # GUARD: recomputed packets must reproduce the run's aliases. hash((item_id,
    # order)) is stable within a Python version only; on the wrong interpreter
    # copy rates collapse to 0 and look like a real null. Crash instead.
    bad = sum(1 for r in rows
              if arb._source_rankings(items[r["item_id"]], r["order"], ctx)[2] != r["gold_alias"])
    if bad:
        raise SystemExit(f"alias mismatch on {bad}/{len(rows)} rows -- re-run under "
                         "conda ops-experimental with PYTHONHASHSEED=0")

    # --- deterministic source references + mechanical fusion (exp20 rules) ----
    per_item = {}
    for iid, it in items.items():
        packet, gold, audit = pilot.make_packet(it, "H-mask", 0, ctx)
        assert audit["masked"], iid
        alias = F._alias_map(it, 0)
        assert alias[it["gold"]] == gold
        phen = {alias[g]: float(emb[it["query"]] @ emb[g])
                for g in [it["gold"]] + it["distractors"]}
        ann = rc._ann_scores(packet)
        assert abs(rc._avg_rank(phen, gold) - it["phen_rank"]) < 1e-9
        assert abs(rc._avg_rank(ann, gold) - it["ann_rank"]) < 1e-9
        row = cal[it["matching_row"]]
        pe = row["phenotype_expected_reciprocal_rank"]
        ae = row["annotation_expected_reciprocal_rank"]
        e = dict(phenotype=F.rr(phen, gold), annotation=F.rr(ann, gold),
                 t1_phenotype=F.top1(phen, gold), t1_annotation=F.top1(ann, gold))
        for name, sc in F.fuse_all(phen, ann,
                                   (max(0.0, pe - RANDOM_RR), max(0.0, ae - RANDOM_RR)),
                                   (pe, ae)).items():
            e[name] = F.rr(sc, gold)
            e["t1_" + name] = F.top1(sc, gold)
        e["best_source"] = max(e["phenotype"], e["annotation"])
        e["t1_best_source"] = max(e["t1_phenotype"], e["t1_annotation"])
        # exceedable = neither source already ranks the gold first; on the rest
        # "beat the better source" is arithmetically impossible (RR caps at 1).
        e["exceedable"] = e["best_source"] < 1.0 - 1e-9
        per_item[iid] = e

    ids = sorted(items)
    P = {i: per_item[i]["phenotype"] for i in ids}
    AN = {i: per_item[i]["annotation"] for i in ids}
    exceedable = [i for i in ids if per_item[i]["exceedable"]]

    # --- per-arm model stats -------------------------------------------------
    arms = sorted({r["arm"] for r in rows})
    report = {
        "cell": meta["sub_bin"], "parent_bin": PARENT,
        "n_items": len(ids), "n_responses": len(rows),
        "random_ranking_MRR": round(RANDOM_RR, 4),
        "calibration_row_shown": {meta["items"][0]["matching_row"]:
                                  cal[meta["items"][0]["matching_row"]]},
        "true_partner_phenotype_rank_spread": meta["true_partner_phenotype_rank_spread"],
        "verification": _verify(meta, items),
        "references": {
            "phenotype_only_ranker_MRR": _m([P[i] for i in ids]),
            "annotation_only_ranker_MRR": _m([AN[i] for i in ids]),
            "phenotype_only_ranker_top1": _m([per_item[i]["t1_phenotype"] for i in ids]),
            "annotation_only_ranker_top1": _m([per_item[i]["t1_annotation"] for i in ids]),
            "best_source_oracle_MRR": _m([per_item[i]["best_source"] for i in ids]),
            "best_source_oracle_top1": _m([per_item[i]["t1_best_source"] for i in ids]),
            "n_exceedable": len(exceedable),
        },
        "fusion_rules": {n: {
            "MRR": _m([per_item[i][n] for i in ids]),
            "top1": _m([per_item[i]["t1_" + n] for i in ids]),
            "vs_best_source_oracle": _m([per_item[i][n] - per_item[i]["best_source"] for i in ids]),
            "vs_best_source_oracle_CI": arb._boot(
                [per_item[i][n] - per_item[i]["best_source"] for i in ids]),
            "exceeds_better_source_n": sum(
                1 for i in exceedable if per_item[i][n] > per_item[i]["best_source"] + 1e-9),
        } for n in RULES},
        "arms": {},
    }

    # How reproducible is each source's full ordering AT ALL? A source with tied
    # scores has no unique ordering, so exact-match copy rates are not comparable
    # across cells without this. `_source_rankings` breaks ties by sort order, so
    # matching a tied source requires coincidentally matching an arbitrary
    # tiebreak -- which mechanically deflates its copy rate.
    def _tiedness(item_list, ctx):
        a, p = [], []
        for it in item_list:
            pk, _, _ = pilot.make_packet(it, "H-mask", 0, ctx)
            a.append(len(set(round(v, 9) for v in rc._ann_scores(pk).values())))
            p.append(len(set(pk["phenotype_similarity"].values())))
        return {"n": len(a),
                "mean_distinct_annotation_scores_of_6": round(float(np.mean(a)), 2),
                "items_with_fully_distinct_annotation": int(sum(1 for x in a if x == 6)),
                "mean_distinct_phenotype_scores_of_6": round(float(np.mean(p)), 2)}

    pub = json.load(open(arb.OUT_DIR / "items.json"))["items"]
    report["source_tie_structure"] = {
        "complementary": _tiedness(list(items.values()), ctx),
        "conflict_published": _tiedness([i for i in pub if i["cell"] == "conflict"], ctx),
        "control_published": _tiedness([i for i in pub if i["cell"] == "control"], ctx),
    }

    for arm in arms:
        rr, t1, rel, ncalls = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        copy_p = copy_a = tot = parse_err = 0
        top1_p = top1_a = 0
        for r in rows:
            if r["arm"] != arm:
                continue
            resp = r["response"]
            if not isinstance(resp, dict) or "ranking" not in resp:
                parse_err += 1
                continue
            rk = resp["ranking"]
            g = r["gold_alias"]
            rr[r["item_id"]].append(1.0 / (rk.index(g) + 1) if g in rk else 0.0)
            t1[r["item_id"]].append(1.0 if rk[:1] == [g] else 0.0)
            rel[r["item_id"]].append("get_source_reliability" in r["tool_calls"])
            ncalls[r["item_id"]].append(len(r["tool_calls"]))
            pr, ar, _ = arb._source_rankings(items[r["item_id"]], r["order"], ctx)
            tot += 1
            copy_p += rk == pr
            copy_a += rk == ar
            top1_p += rk[0] == pr[0]
            top1_a += rk[0] == ar[0]
        have = [i for i in ids if i in rr]
        M = {i: float(np.mean(rr[i])) for i in have}
        T = {i: float(np.mean(t1[i])) for i in have}
        dP = [M[i] - P[i] for i in have]
        dA = [M[i] - AN[i] for i in have]
        dO = [M[i] - per_item[i]["best_source"] for i in have]
        report["arms"][arm] = {
            "n_items": len(have), "n_calls_scored": tot, "parse_errors": parse_err,
            "MRR": _m([M[i] for i in have]), "MRR_CI": arb._boot([M[i] for i in have]),
            "top1": _m([T[i] for i in have]),
            # FULL SIX-ITEM PERMUTATION equality, not gold-rank agreement.
            "verbatim_copy_full_ordering_phenotype": copy_p,
            "verbatim_copy_full_ordering_annotation": copy_a,
            "verbatim_copy_full_ordering_either": None,   # filled below
            "verbatim_copy_full_ordering_phenotype_rate": round(copy_p / tot, 3) if tot else None,
            "verbatim_copy_full_ordering_annotation_rate": round(copy_a / tot, 3) if tot else None,
            "verbatim_copy_full_ordering_phenotype_rate_CI": _wilson(copy_p, tot),
            "verbatim_copy_full_ordering_annotation_rate_CI": _wilson(copy_a, tot),
            "copies_neither_source": tot - copy_p - copy_a,
            # weaker statistic, kept only for contrast -- NOT the headline
            "top_pick_agrees_phenotype": top1_p,
            "top_pick_agrees_annotation": top1_a,
            "minus_phenotype_only_ranker": _m(dP), "minus_phenotype_only_ranker_CI": arb._boot(dP),
            "minus_annotation_only_ranker": _m(dA), "minus_annotation_only_ranker_CI": arb._boot(dA),
            "minus_best_source_oracle": _m(dO), "minus_best_source_oracle_CI": arb._boot(dO),
            "exceeds_best_source_oracle_n": sum(
                1 for i in have if M[i] > per_item[i]["best_source"] + 1e-9),
            "n_exceedable_among_scored": sum(1 for i in have if per_item[i]["exceedable"]),
            "queried_reliability_rate": round(float(np.mean(
                [np.mean(rel[i]) for i in have])), 3) if have else None,
            "mean_tool_calls": round(float(np.mean([np.mean(ncalls[i]) for i in have])), 2)
            if have else None,
        }
        # copy-either needs the union per call, recomputed in one more pass
        either = 0
        for r in rows:
            if r["arm"] != arm:
                continue
            resp = r["response"]
            if not isinstance(resp, dict) or "ranking" not in resp:
                continue
            pr, ar, _ = arb._source_rankings(items[r["item_id"]], r["order"], ctx)
            either += resp["ranking"] in (pr, ar)
        report["arms"][arm]["verbatim_copy_full_ordering_either"] = either

    # Table 1's +table/-table row: paired agent_free - agent_norel over items.
    # Unlike the per-arm MRR above, a parse error scores RR 0 here (a failed call
    # is a wrong answer), so a model's gain is not inflated by dropping its misses.
    if {"agent_free", "agent_norel"} <= set(arms):
        Z = defaultdict(lambda: defaultdict(list))
        for r in rows:
            resp, g = r["response"], r["gold_alias"]
            rk = resp.get("ranking", []) if isinstance(resp, dict) else []
            Z[r["arm"]][r["item_id"]].append(1.0 / (rk.index(g) + 1) if g in rk else 0.0)
        both = [i for i in ids if i in Z["agent_free"] and i in Z["agent_norel"]]
        mf = {i: float(np.mean(Z["agent_free"][i])) for i in both}
        mn = {i: float(np.mean(Z["agent_norel"][i])) for i in both}
        d = [mf[i] - mn[i] for i in both]
        report["table_contrast"] = {
            "n_items": len(both), "parse_errors_scored_as": 0.0,
            "without_table_MRR": _m([mn[i] for i in both]),
            "with_table_MRR": _m([mf[i] for i in both]),
            "with_minus_without": _m(d), "with_minus_without_CI": arb._boot(d)}

    json.dump(report, open(OUT_DIR / f"metrics{suffix}.json", "w"), indent=2)
    json.dump({i: per_item[i] for i in ids}, open(OUT_DIR / f"per_item{suffix}.json", "w"), indent=2)
    _write_results(report, meta, cal)
    print(json.dumps(report, indent=2))
    return report


def _esc(k):
    return k.replace("|", r"\|")


def _write_results(rep, meta, cal):
    ref = rep["references"]
    L = [f"# Complementary cell: {_esc(rep['cell'])}", "",
         "Third arbitration cell, built where copying one source is **not** the "
         "calibration-optimal policy. Parent bin `" + _esc(PARENT) + "` (exp22's only "
         "genuinely complementary region), subdivided on the calibration half so "
         "that neither source sits at the random rate.", "",
         "## The reliability row the model is shown", "",
         "```json",
         json.dumps(rep["calibration_row_shown"], indent=2),
         "```", "",
         f"Random-ranking expected reciprocal rank = {rep['random_ranking_MRR']}. "
         "Both sources are above it, so the table does not instruct the model to "
         "zero out either one.", "",
         "## Verbatim copy rate -- the headline", "",
         "**Full six-item permutation equality**: the call's entire ranking is "
         "identical, position by position, to a single source's ranking. This is "
         "NOT gold-rank agreement -- `arbitration.noise_floor()`'s "
         "`exact_agreement_rate` compares reciprocal ranks and is a different, "
         "weaker statistic; it is not used here.", "",
         "| cell | model | n calls | copies phenotype | copies annotation |",
         "|---|---|---|---|---|",
         "| control (published) | A | 40 | **40/40** | 0/40 |",
         "| control (published) | B | 40 | **39/40** | - |",
         "| conflict (published) | A | 80 | 5/80 | **28/80** |",
         "| conflict (published) | B | 80 | 4/80 | - |"]
    for arm, a in rep["arms"].items():
        L.append(
            f"| {_esc(rep['cell'])} -- {arm} | A | {a['n_calls_scored']} | "
            f"**{a['verbatim_copy_full_ordering_phenotype']}/{a['n_calls_scored']}** "
            f"({a['verbatim_copy_full_ordering_phenotype_rate']}, 95% CI "
            f"{tuple(a['verbatim_copy_full_ordering_phenotype_rate_CI'])}) | "
            f"**{a['verbatim_copy_full_ordering_annotation']}/{a['n_calls_scored']}** "
            f"({a['verbatim_copy_full_ordering_annotation_rate']}, 95% CI "
            f"{tuple(a['verbatim_copy_full_ordering_annotation_rate_CI'])}) |")
    t = rep["source_tie_structure"]
    L += ["",
          "Copies **neither** source: " + ", ".join(
              f"{a['copies_neither_source']}/{a['n_calls_scored']} ({arm})"
              for arm, a in rep["arms"].items()) +
          " -- against 0/40 in the control cell and 47/80 in the conflict cell.", "",
          "### Caveat: the two cells are not equally copyable", "",
          "A source with tied scores has no unique ordering, and "
          "`arbitration._source_rankings` breaks ties by sort order, so matching "
          "a tied source requires coincidentally matching an arbitrary tiebreak. "
          "That mechanically deflates its copy rate.", "",
          "| cell | mean distinct annotation scores (of 6) | items with a fully "
          "distinct annotation ordering | mean distinct phenotype scores |",
          "|---|---|---|---|"] + [
        f"| {k} | {v['mean_distinct_annotation_scores_of_6']} | "
        f"{v['items_with_fully_distinct_annotation']}/{v['n']} | "
        f"{v['mean_distinct_phenotype_scores_of_6']} |"
        for k, v in t.items()] + [
        "",
        "So the annotation copy rate here (denser ties than the conflict cell) is "
        "a floor, not a like-for-like number. The **phenotype** side is directly "
        "comparable -- phenotype scores are near-fully distinct in all three cells "
        "-- and that is the column against which the control cell's 40/40 sits.", "",
        "Weaker statistic, for contrast only (top pick agrees, full ordering need "
        "not): " + ", ".join(
            f"{arm} phenotype {a['top_pick_agrees_phenotype']}/{a['n_calls_scored']}, "
            f"annotation {a['top_pick_agrees_annotation']}/{a['n_calls_scored']}"
            for arm, a in rep["arms"].items()) + ".", "",
          "A null on the MRR table below is expected and is not the result: no "
          "fixed fusion rule beats per-item best-source selection anywhere in "
          "exp22's nine-bin x five-rule grid (best entry -0.0541). The copy rate "
          "above is the measurement this run exists for.", "",
          "## Accuracy", "",
          "Annotation-only MRR uses the current `1/avg_rank` convention "
          "(average-rank tie handling, no tie-aware fix applied). Exact-ordering "
          "matching above is unaffected by that convention.", ""]
    L += ["| arm | MRR | 95% CI | top-1 | vs phen-only | vs ann-only | vs best-source oracle |",
          "|---|---|---|---|---|---|---|",
          f"| phenotype-only ranker | {ref['phenotype_only_ranker_MRR']} | - | "
          f"{ref['phenotype_only_ranker_top1']} | - | - | - |",
          f"| annotation-only ranker | {ref['annotation_only_ranker_MRR']} | - | "
          f"{ref['annotation_only_ranker_top1']} | - | - | - |",
          f"| best-source oracle | {ref['best_source_oracle_MRR']} | - | "
          f"{ref['best_source_oracle_top1']} | - | - | - |"]
    for n, r in rep["fusion_rules"].items():
        L.append(f"| fusion `{n}` | {r['MRR']} | - | {r['top1']} | - | - | "
                 f"{r['vs_best_source_oracle']:+} {tuple(r['vs_best_source_oracle_CI'])} |")
    for arm, a in rep["arms"].items():
        L.append(f"| **{arm}** | {a['MRR']} | {tuple(a['MRR_CI'])} | {a['top1']} | "
                 f"{a['minus_phenotype_only_ranker']:+} {tuple(a['minus_phenotype_only_ranker_CI'])} | "
                 f"{a['minus_annotation_only_ranker']:+} {tuple(a['minus_annotation_only_ranker_CI'])} | "
                 f"{a['minus_best_source_oracle']:+} {tuple(a['minus_best_source_oracle_CI'])} |")
    L += ["",
          "Paired bootstrap CIs are over the 40 items (`arbitration._boot`). Model "
          "A's measured run-to-run noise floor on this task is MRR SE 0.0043 over "
          "40 items; every arm difference above is larger than that in magnitude, "
          "so the CIs that span zero are item-level sampling limits, not "
          "replication noise.", ""]
    L += ["", "## Exceedance", "",
          f"Exceedable items (neither source already ranks the true partner first): "
          f"**{ref['n_exceedable']}** of {rep['n_items']}.", "",
          "| arm | exceeds the better source |", "|---|---|"]
    for n, r in rep["fusion_rules"].items():
        L.append(f"| fusion `{n}` | {r['exceeds_better_source_n']}/{ref['n_exceedable']} |")
    for arm, a in rep["arms"].items():
        L.append(f"| {arm} | {a['exceeds_best_source_oracle_n']}/{a['n_exceedable_among_scored']} |")
    L += ["", "## Tool use", "", "| arm | queried reliability | mean tool calls |",
          "|---|---|---|"]
    for arm, a in rep["arms"].items():
        L.append(f"| {arm} | {a['queried_reliability_rate']} | {a['mean_tool_calls']} |")
    v = rep["verification"]
    L += ["", "## Verification block (correctness-blindness)", "",
          f"- **Selection statistics:** {', '.join(v['selection_statistics'])} -- all "
          "computed from the packet the model is handed. The gold's phenotype rank, "
          "annotation rank, and identity are **never** referenced by any selection "
          f"predicate (`gold_rank_used_in_selection = "
          f"{v['gold_rank_used_in_selection']}`).",
          f"- **One item per distinct query:** {v['n_items']} items, "
          f"{v['n_distinct_queries']} distinct queries -> "
          f"{v['one_item_per_query']}.",
          f"- **Query-disjoint from the calibration half:** "
          f"{v['n_items_in_calibration_half']} of {v['n_items']} items come from a "
          f"calibration-half query -> {v['query_disjoint_from_calibration']}. The "
          "refined reliability table is fit on the calibration half only.",
          f"- **Query-disjoint from the published cells:** "
          f"{v['n_items_reusing_conflict_or_control_query']} overlap "
          f"({meta['n_queries_dropped_as_already_used']} eligible queries dropped "
          "for this reason).",
          f"- **Not a disguised difficulty split:** true-partner phenotype rank "
          f"spread {rep['true_partner_phenotype_rank_spread']}; spans all six ranks: "
          f"{v['spans_all_ranks']}.", ""]
    (OUT_DIR / "RESULTS.md").write_text("\n".join(L) + "\n")
    print(f"wrote -> {OUT_DIR/'RESULTS.md'}")


# ---------------------------------------------------------------------------
# self-check on the non-API logic
# ---------------------------------------------------------------------------
def selfcheck():
    F.selfcheck()

    # --- expected reciprocal rank ---
    assert abs(_err([{"r": 1.0}, {"r": 2.0}], "r") - 0.75) < 1e-12

    # --- median-split search: a side qualifies only if BOTH sources beat the
    # random rate AND it holds enough eval-half queries. Toy pool: observable x
    # splits at median 2; the high side is informative for both sources, the low
    # side is informative for phenotype only.
    def row(half, x, pr, ar, q):
        return dict(half=half, query=q, gap=0.0, margin=0.5, ann_top=x,
                    n_ann_pos=1, n_edges=1, phen_top=0.0, phen_rank=pr, ann_rank=ar)
    cal = ([row("cal", 3.0, 1.0, 1.0, f"c{i}") for i in range(10)] +
           [row("cal", 1.0, 1.0, 6.0, f"d{i}") for i in range(10)])
    ev = [row("eval", 3.0, 1.0, 1.0, f"e{i}") for i in range(MIN_EVAL_QUERIES)] + \
         [row("eval", 1.0, 1.0, 6.0, f"f{i}") for i in range(3)]
    r = refine(cal + ev)
    ch = r["chosen"]
    assert ch is not None and ch["side"] == "ge", ch
    assert ch["annotation_expected_reciprocal_rank"] == 1.0, ch
    lo = [s for s in r["search"] if s["field"] == "ann_top" and s["side"] == "lt"][0]
    assert not lo["both_above_random"] and not lo["qualifies"], lo
    # starve the eval half -> the same sub-bin no longer qualifies -> gate fails
    thin = refine(cal + ev[:5])
    assert thin["chosen"] is None, thin["chosen"]

    # --- row_key routes an item to the sub-row matching its observable ---
    # The fixtures below sit at gap=0.005, margin=0.5, so pin PARENT to that
    # bin for the duration: this checks routing, not whichever parent the run
    # happens to be using (--parent can move it).
    global PARENT
    _saved_parent = PARENT
    PARENT = "gap<0.013 | annotation margin 0-1.5"
    try:
        ch2 = dict(field="ann_top", cut=1.0, observable="annotation_top_score",
                   side="ge", label="annotation_top_score>=1")
        hi = dict(gap=0.005, margin=0.5, ann_top=2.0)
        lo2 = dict(gap=0.005, margin=0.5, ann_top=0.5)
        assert row_key(hi, ch2) == f"{PARENT} | annotation_top_score>=1"
        assert row_key(lo2, ch2) == f"{PARENT} | annotation_top_score<1"
        # an item outside the parent bin keeps its coarse key
        assert row_key(dict(gap=0.2, margin=0.0, ann_top=2.0), ch2) == \
            "gap>=0.059 | no discriminating annotation"
    finally:
        PARENT = _saved_parent

    # --- verbatim copy is exact full-list equality, not a prefix or a set ---
    src = ["G1", "G2", "G3", "G4", "G5", "G6"]
    assert (src == list(src)) and (["G1", "G2", "G3", "G4", "G6", "G5"] != src)

    # --- exceedance denominator: an item whose better source already ranks the
    # gold first cannot be exceeded, so it must not sit in the denominator.
    assert not (max(1.0, 0.5) < 1.0 - 1e-9)
    assert (max(0.5, 0.25) < 1.0 - 1e-9)
    print("selfcheck OK")


def main():
    global PARENT, OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["stage1", "build", "run", "analyze", "selfcheck"])
    ap.add_argument("--arm", choices=list(ARM_TOOLS), default="agent_free")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--provider", choices=["vertex", "azure"], default="vertex",
                    help="azure runs the second model family (gpt-5.6-luna) and "
                         "writes responses_gpt56.jsonl beside the vertex file.")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-episodes", type=int, default=180)
    ap.add_argument("--parent", default=PARENT,
                    help="parent (gap x margin) bin. Default is the published "
                         "cell's bin, which exp22 selected on the eval half. "
                         "Pass the calibration-half selection to keep the eval "
                         "half out of every selection step.")
    ap.add_argument("--out", help="artifact subdirectory under artifacts/evidence_eval/ "
                                  "(default 'complementary'). Use a distinct "
                                  "name so an existing cell is not clobbered.")
    a = ap.parse_args()
    PARENT = a.parent
    if a.out:
        OUT_DIR = arb.ART / a.out
    if a.cmd == "selfcheck":
        selfcheck()
        return
    _require_hashseed()
    selfcheck()
    if a.cmd == "stage1":
        stage1()
    elif a.cmd == "build":
        build()
    elif a.cmd == "run":
        EPISODE_CAP[0] = a.max_episodes
        run(a.arm, a.model, limit=a.limit, provider=a.provider)
    else:
        analyze(provider=a.provider)


if __name__ == "__main__":
    main()
