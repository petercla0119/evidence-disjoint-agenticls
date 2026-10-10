#!/usr/bin/env python3
"""Counterfactual evidence pilot: does real phenotype evidence rescue an LLM's
ranking of the correct partner when answer-bearing annotations are masked?

One-day feasibility test for a workshop paper on agentic AI in life science.
NO tool-calling agent -- static evidence packets only. See the planning memo for
the full rationale; the headline endpoint is the mask x phenotype INTERACTION:
phenotype should help little when direct annotations are present, but rescue the
ranking once the query->gold records are removed -- and primarily in the
phenotype-informative stratum.

Pipeline (three stages, model only touched in `run`):

  build    -> artifacts/evidence_eval/pilot/items.json   (24 MCQ items, 2 strata, offline)
  run      -> artifacts/evidence_eval/pilot/responses.jsonl  (240 Vertex calls, resumable)
  analyze  -> metrics.json + pilot_conditions.{png,pdf}

Conditions (same model/prompt/temperature/item):
  A-open   unmasked packet, no phenotype       (conventional exposed eval)
  H-open   unmasked packet, real phenotype
  A-mask   query->gold records removed, no pheno (evidence-disjoint baseline)
  H-mask   masked, real phenotype              (main rescue condition)
  H-shuffle masked, phenotype permuted within item (negative control)

24 items x 5 conditions x 2 candidate orders = 240 calls.

Usage:
  python scripts/evidence_eval/pilot_counterfactual.py build   [--seed 42]
  python scripts/evidence_eval/pilot_counterfactual.py run     [--model claude-sonnet-4-6] [--limit N] [--smoke]
  python scripts/evidence_eval/pilot_counterfactual.py analyze
"""

# Thread caps before numpy import (cluster rule; harmless on laptop).
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "artifacts" / "evidence_eval" / "pilot"
EXT = BASE_DIR / "data" / "external"

FEATURE_NPZ = BASE_DIR / "data" / "feature98_interphase_gene.npz"
SPLITS = BASE_DIR / "data" / "splits_v1.json"
STRING_INFO = EXT / "9606.protein.info.v12.0.txt.gz"
STRING_LINKS = EXT / "9606.protein.links.v12.0.txt.gz"
REACTOME_JSON = EXT / "reactome_pathways.json"
COMPLEX_JSON = EXT / "complex_portal_complexes.json"

# STRING confidence bands (combined_score, 0-999).
POS_THRESH = 700    # gold partner must be a high-confidence STRING interactor
NEG_THRESH = 400    # distractors must NOT be a medium+ interactor of the query
DEG_THRESH = 400    # degree counts medium+ interactors (a "how studied" proxy)

CONDITIONS = ["A-open", "H-open", "A-mask", "H-mask", "H-shuffle"]
N_ORDERS = 2
N_PER_STRATUM = 12


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_embeddings():
    z = np.load(FEATURE_NPZ, allow_pickle=True)
    genes = [str(g) for g in z["genes"]]
    emb = z["embeddings"].astype(np.float64)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)  # unit-norm -> cosine = dot
    return {g: emb[i] for i, g in enumerate(genes)}


def load_splits():
    d = json.load(open(SPLITS))
    return set(d["train"]), set(d["test"])


def load_string(gene_set):
    """Return (pairs, degree): pairs[frozenset]=max_score among gene_set genes;
    degree[gene]=# distinct medium+ interactors across all human genes."""
    p2g = {}
    with gzip.open(STRING_INFO, "rt") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                p2g[parts[0]] = parts[1]
    pairs = {}
    degree = defaultdict(set)
    with gzip.open(STRING_LINKS, "rt") as f:
        next(f)
        for line in f:
            a, b, s = line.split()
            s = int(s)
            if s < DEG_THRESH:
                continue
            g1, g2 = p2g.get(a), p2g.get(b)
            if not g1 or not g2 or g1 == g2:
                continue
            degree[g1].add(g2)
            if g1 in gene_set and g2 in gene_set:
                pr = frozenset((g1, g2))
                pairs[pr] = max(pairs.get(pr, 0), s)
    return pairs, {g: len(v) for g, v in degree.items()}


def load_memberships():
    """gene -> set(pathway_id), gene -> set(complex_id)."""
    g_path = defaultdict(set)
    react = json.load(open(REACTOME_JSON))
    for pid, rec in react.items():
        for g in rec.get("genes", []):
            g_path[g].add(pid)
    g_cplx = defaultdict(set)
    for rec in json.load(open(COMPLEX_JSON)):
        for g in rec.get("genes", []):
            g_cplx[g].add(rec["id"])
    return g_path, g_cplx


# ---------------------------------------------------------------------------
# build: construct 24 MCQ items, 12 phenotype-informative + 12 uninformative
# ---------------------------------------------------------------------------
def build_items(seed=42):
    emb = load_embeddings()
    train, test = load_splits()
    train = {g for g in train if g in emb}
    test = {g for g in test if g in emb}
    pairs, degree = load_string(set(emb))
    g_path, g_cplx = load_memberships()

    def deg(g):
        return degree.get(g, 0)

    # query -> interactors (any medium+), and high-confidence train partners
    interactors = defaultdict(set)
    for pr, s in pairs.items():
        a, b = tuple(pr)
        interactors[a].add(b)
        interactors[b].add(a)

    def phen_rank(query, cands, gold):
        # rank of gold among cands by cosine to query (1 = most similar)
        sims = sorted(cands, key=lambda c: -float(emb[query] @ emb[c]))
        return sims.index(gold) + 1

    rng = random.Random(seed)
    queries = sorted(test)
    rng.shuffle(queries)

    informative, uninformative, used = [], [], set()
    for q in queries:
        if len(informative) >= N_PER_STRATUM and len(uninformative) >= N_PER_STRATUM:
            break
        if q in used:
            continue
        # gold: highest-confidence STRING partner that lives in the train pool
        golds = sorted(
            (g for g in interactors[q] if g in train and pairs[frozenset((q, g))] >= POS_THRESH),
            key=lambda g: -pairs[frozenset((q, g))],
        )
        if not golds:
            continue
        gold = golds[0]
        # distractor pool: train genes, NOT a medium+ interactor of q, not gold,
        # annotated (broad-function proxy), degree-matched to gold.
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
        r = phen_rank(q, cands, gold)
        item = dict(query=q, gold=gold, distractors=distractors,
                    gold_degree=deg(gold), phen_rank=r)
        if r in (1, 2) and len(informative) < N_PER_STRATUM:
            item["stratum"] = "informative"
            informative.append(item); used.add(q)
        elif r in (5, 6) and len(uninformative) < N_PER_STRATUM:
            item["stratum"] = "uninformative"
            uninformative.append(item); used.add(q)

    items = informative + uninformative
    if len(informative) < N_PER_STRATUM or len(uninformative) < N_PER_STRATUM:
        print(f"WARNING: only {len(informative)} informative / "
              f"{len(uninformative)} uninformative items found", file=sys.stderr)
    for i, it in enumerate(items):
        it["item_id"] = f"it{i:02d}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"seed": seed, "pos_thresh": POS_THRESH, "neg_thresh": NEG_THRESH,
               "items": items}, open(OUT_DIR / "items.json", "w"), indent=2)
    print(f"Built {len(items)} items -> {OUT_DIR/'items.json'}")
    print(f"  informative (gold phen-rank 1-2): {len(informative)}")
    print(f"  uninformative (gold phen-rank 5-6): {len(uninformative)}")
    for it in items:
        print(f"    {it['item_id']} {it['stratum'][:5]:5s} q={it['query']:8s} "
              f"gold={it['gold']:8s} rank={it['phen_rank']} deg(gold)={it['gold_degree']}")


# ---------------------------------------------------------------------------
# Packet construction (offline, deterministic per item)
# ---------------------------------------------------------------------------
def _ctx():
    """Load the reference maps needed to render packets."""
    emb = load_embeddings()
    pairs, degree = load_string(set(emb))
    g_path, g_cplx = load_memberships()
    return emb, pairs, degree, g_path, g_cplx


def make_packet(item, condition, order, ctx):
    """Return (packet_dict, gold_alias). Aliases: G0=query, G1..G6=candidates
    under `order`; pathways->P#, complexes->C#. Masked conditions drop every
    record containing BOTH the query and the gold (audited on canonical IDs)."""
    emb, pairs, degree, g_path, g_cplx = ctx
    q = item["query"]
    cands = [item["gold"]] + item["distractors"]

    # deterministic candidate ordering
    perm = list(range(6))
    random.Random(hash((item["item_id"], order)) & 0xFFFFFFFF).shuffle(perm)
    ordered = [cands[i] for i in perm]
    alias = {q: "G0"}
    for i, g in enumerate(ordered):
        alias[g] = f"G{i+1}"
    gold_alias = alias[item["gold"]]
    genes7 = [q] + ordered

    masked = condition.endswith("mask") or condition == "H-shuffle"

    # local aliases for pathways / complexes, assigned on first use
    p_alias, c_alias = {}, {}

    def pa(pid):
        return p_alias.setdefault(pid, f"P{len(p_alias)+1}")

    def ca(cid):
        return c_alias.setdefault(cid, f"C{len(c_alias)+1}")

    # node records carry only individual-node degree (retained under masking).
    # Per-node pathway/complex *lists* are deliberately omitted: they would let
    # the model reconstruct the query<->gold shared-pathway by set intersection
    # even after the pairwise records are masked, defeating the manipulation.
    # Pairwise co-membership lives only in the (maskable) edge records below.
    nodes = [dict(id=alias[g], degree=int(degree.get(g, 0))) for g in genes7]

    # pairwise edge records
    edges = []
    n_removed = 0
    for a, b in combinations(genes7, 2):
        pair_is_qgold = {a, b} == {q, item["gold"]}
        recs = []
        pr = frozenset((a, b))
        if pr in pairs:
            recs.append(dict(type="functional_edge",
                             confidence=round(pairs[pr] / 1000.0, 3)))
        shared_p = g_path.get(a, set()) & g_path.get(b, set())
        for pid in sorted(shared_p):
            recs.append(dict(type="shared_pathway", pathway=pa(pid)))
        shared_c = g_cplx.get(a, set()) & g_cplx.get(b, set())
        for cid in sorted(shared_c):
            recs.append(dict(type="shared_complex", complex=ca(cid)))
        if masked and pair_is_qgold:
            n_removed += len(recs)
            continue  # AUDIT: drop every query<->gold record
        for r in recs:
            r["genes"] = sorted((alias[a], alias[b]))
            edges.append(r)

    for i, r in enumerate(edges):
        r["id"] = f"R{i+1}"

    packet = dict(candidates=[f"G{i+1}" for i in range(6)],
                  nodes=nodes, edges=edges)

    # phenotype block
    if condition.startswith("H"):
        sims = {alias[g]: round(float(emb[q] @ emb[g]), 4) for g in ordered}
        if condition == "H-shuffle":
            labs = list(sims.keys())
            vals = list(sims.values())
            random.Random(hash((item["item_id"], order, "shuf")) & 0xFFFFFFFF).shuffle(vals)
            sims = dict(zip(labs, vals))
        packet["phenotype_similarity"] = sims

    audit = dict(masked=masked, qgold_records_removed=n_removed)
    return packet, gold_alias, audit


SYSTEM = (
    "You are ranking which candidate gene is the true functional partner of a "
    "query gene, using ONLY a structured evidence packet. All identities are "
    "anonymized (genes G0-G6, pathways P#, complexes C#). G0 is the query. "
    "Relation types: functional_edge (direct interaction, with a confidence in "
    "[0,1]), shared_pathway, shared_complex. Node records give each gene's "
    "degree and its pathway/complex memberships. A phenotype_similarity block, "
    "when present, gives each candidate's morphological-perturbation similarity "
    "to the query (higher = more similar). Reason only from the packet; do not "
    "use outside knowledge (identities are hidden anyway)."
)

USER_TMPL = (
    "Evidence packet (JSON):\n{packet}\n\n"
    "Task: rank all six candidates G1-G6 from most to least likely to be the "
    "true functional partner of G0. Return ONLY machine-readable JSON with keys:\n"
    '  "ranking": [six candidate ids, best first],\n'
    '  "probabilities": {{candidate id: probability}}  (must sum to 1),\n'
    '  "first_choice_evidence": [record ids R# supporting your top pick]\n'
    "No prose outside the JSON."
)


# ---------------------------------------------------------------------------
# run: 240 Vertex calls, resumable
# ---------------------------------------------------------------------------
def get_client():
    from anthropic import AnthropicVertex
    region = os.environ.get("CLOUD_ML_REGION", "global")
    project = os.environ["ANTHROPIC_VERTEX_PROJECT_ID"]
    return AnthropicVertex(project_id=project, region=region)


def parse_response(text):
    t = text.strip()
    if "```" in t:  # strip a fenced block if present
        seg = t.split("```")
        t = seg[1] if len(seg) > 1 else t
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    # extract the first balanced {...} object, ignoring leading prose (the model
    # sometimes reasons in prose first) and any trailing text after it.
    i = t.find("{")
    if i < 0:
        raise ValueError("no JSON object in response")
    obj, _ = json.JSONDecoder().raw_decode(t[i:])
    return obj


def run(model, limit=None, smoke=False):
    data = json.load(open(OUT_DIR / "items.json"))
    items = data["items"]
    ctx = _ctx()
    client = get_client()

    out_path = OUT_DIR / "responses.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["item_id"], r["condition"], r["order"]))

    jobs = [(it, c, o) for it in items for c in CONDITIONS for o in range(N_ORDERS)]
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
            extra_body={"temperature": 0.0},  # pinned; not an explicit kwarg in this SDK
        )
        text = msg.content[0].text
        try:
            parsed = parse_response(text)
        except Exception as e:
            parsed = {"_parse_error": str(e), "_raw": text[:500]}
        rec = dict(item_id=it["item_id"], stratum=it["stratum"], condition=cond,
                   order=order, gold_alias=gold_alias, model=model,
                   audit=audit, response=parsed)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        rk = parsed.get("ranking", []) if isinstance(parsed, dict) else []
        pos = (rk.index(gold_alias) + 1) if gold_alias in rk else "-"
        print(f"[{n}/{len(jobs)}] {it['item_id']} {cond:9s} o{order} "
              f"gold={gold_alias} pos={pos}")
    f.close()
    print(f"Wrote -> {out_path}")


# ---------------------------------------------------------------------------
# analyze: MRR/top-1, three deltas, bootstrap CIs, baselines, figure
# ---------------------------------------------------------------------------
def _rr(rec):
    rk = rec["response"].get("ranking", []) if isinstance(rec["response"], dict) else []
    ga = rec["gold_alias"]
    return (1.0 / (rk.index(ga) + 1)) if ga in rk else 0.0


def _top1(rec):
    rk = rec["response"].get("ranking", []) if isinstance(rec["response"], dict) else []
    return 1.0 if rk and rk[0] == rec["gold_alias"] else 0.0


def analyze():
    recs = [json.loads(l) for l in open(OUT_DIR / "responses.jsonl")]
    items = json.load(open(OUT_DIR / "items.json"))["items"]
    strata = {it["item_id"]: it["stratum"] for it in items}

    # per (item, condition): average the 2 orders (repeated measures)
    def per_item(metric):
        acc = defaultdict(list)
        for r in recs:
            acc[(r["item_id"], r["condition"])].append(metric(r))
        return {k: float(np.mean(v)) for k, v in acc.items()}

    rr = per_item(_rr)
    t1 = per_item(_top1)
    item_ids = sorted({r["item_id"] for r in recs})

    def mrr(cond, ids):
        vals = [rr[(i, cond)] for i in ids if (i, cond) in rr]
        return float(np.mean(vals)) if vals else float("nan")

    def acc(cond, ids):
        vals = [t1[(i, cond)] for i in ids if (i, cond) in t1]
        return float(np.mean(vals)) if vals else float("nan")

    strata_ids = {
        "all": item_ids,
        "informative": [i for i in item_ids if strata.get(i) == "informative"],
        "uninformative": [i for i in item_ids if strata.get(i) == "uninformative"],
    }

    def deltas(ids):
        Ao, Ho = mrr("A-open", ids), mrr("H-open", ids)
        Am, Hm = mrr("A-mask", ids), mrr("H-mask", ids)
        return dict(
            exposure=Ao - Am,
            rescue=Hm - Am,
            interaction=(Hm - Am) - (Ho - Ao),
        )

    def boot_delta(ids, key, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        ids = list(ids)
        for _ in range(n):
            s = list(rng.choice(ids, size=len(ids), replace=True))
            out.append(deltas(s)[key])
        lo, hi = np.percentile(out, [2.5, 97.5])
        return float(np.mean(out)), float(lo), float(hi)

    # zero-cost baselines
    H6 = sum(1.0 / k for k in range(1, 7))
    random_mrr = H6 / 6.0  # E[RR] uniform over 6
    # phenotype argmax: pick max real phenotype score = rank-1 candidate;
    # RR = 1/phen_rank(gold). (H-shuffle/real scores identical distribution)
    phen_rr = {it["item_id"]: 1.0 / it["phen_rank"] for it in items}
    def phen_argmax_mrr(ids):
        return float(np.mean([phen_rr[i] for i in ids]))

    report = {"model": recs[0]["model"], "n_items": len(item_ids),
              "conditions": {}, "deltas": {}, "baselines": {}}
    for name, ids in strata_ids.items():
        report["conditions"][name] = {
            c: {"MRR": round(mrr(c, ids), 4), "top1": round(acc(c, ids), 4)}
            for c in CONDITIONS}
        d = deltas(ids)
        report["deltas"][name] = {}
        for k in ("exposure", "rescue", "interaction"):
            m, lo, hi = boot_delta(ids, k)
            report["deltas"][name][k] = {"point": round(d[k], 4),
                                         "boot_mean": round(m, 4),
                                         "ci95": [round(lo, 4), round(hi, 4)]}
        report["baselines"][name] = {
            "random_MRR": round(random_mrr, 4),
            "phenotype_argmax_MRR": round(phen_argmax_mrr(ids), 4)}

    # parse-error / audit sanity
    n_err = sum(1 for r in recs if "_parse_error" in r["response"])
    n_masknoop = sum(1 for r in recs
                     if r["audit"]["masked"] and r["audit"]["qgold_records_removed"] == 0)
    report["sanity"] = {"n_calls": len(recs), "parse_errors": n_err,
                        "masked_calls_removing_0_records": n_masknoop}

    json.dump(report, open(OUT_DIR / "metrics.json", "w"), indent=2)
    print(json.dumps(report, indent=2))
    _figure(report)


def _figure(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(CONDITIONS)))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    styles = {"informative": dict(color="#2166ac", marker="o", label="phenotype-informative (rank 1-2)"),
              "uninformative": dict(color="#b2182b", marker="s", label="phenotype-uninformative (rank 5-6)")}
    for name, st in styles.items():
        y = [report["conditions"][name][c]["MRR"] for c in CONDITIONS]
        ax.plot(x, y, linewidth=2, markersize=7, **st)
    b = report["baselines"]["all"]
    ax.axhline(b["random_MRR"], ls=":", color="gray", lw=1, label="random ranking")
    ax.set_xticks(x); ax.set_xticklabels(CONDITIONS, rotation=15)
    ax.set_ylabel("MRR (gold partner)"); ax.set_ylim(0, 1.02)
    ax.set_title("Counterfactual evidence pilot: phenotype rescue under masking")
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"pilot_conditions.{ext}", dpi=200)
    print(f"Figure -> {OUT_DIR/'pilot_conditions.png'}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--seed", type=int, default=42)
    r = sub.add_parser("run")
    r.add_argument("--model", default="claude-sonnet-4-6")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--smoke", action="store_true")
    sub.add_parser("analyze")
    args = ap.parse_args()

    if args.cmd == "build":
        build_items(args.seed)
    elif args.cmd == "run":
        run(args.model, args.limit, args.smoke)
    elif args.cmd == "analyze":
        analyze()


if __name__ == "__main__":
    main()
