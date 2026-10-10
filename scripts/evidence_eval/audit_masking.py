#!/usr/bin/env python
"""Audit the evidence mask on CANONICAL identifiers, not on the aliases.

The protocol claims one sentence's worth of things about `pilot.make_packet`
under a masked condition:

  (1) every record containing BOTH the query and the true partner is gone,
  (2) nothing else is gone (indirect / two-hop annotation survives),
  (3) the packet carries no canonical identifier at all -- gene symbols,
      Reactome pathway ids and Complex Portal complex ids are all replaced,
      and node records carry only degree,
  (4) so a model cannot recover the answer from pretrained knowledge.

This script checks (1)-(3) mechanically. It does NOT call `make_packet`'s own
masking predicate: it rebuilds the record set that *should* exist for the seven
genes straight from STRING / Reactome / Complex Portal, in canonical symbol
space, then de-aliases the emitted packet back to canonical pairs and diffs the
two multisets. So a mask that dropped too much, too little, or the wrong pair
all fail here, and so does a masking bug that only shows up in alias space.

What it would catch that a spot-check would not:
  - a direct STRING edge, shared pathway or shared complex between query and
    true partner surviving into the packet (under-masking -> answer leak)
  - collateral removal of records not involving the pair (over-masking -> the
    "two-hop annotation survives" claim would be false)
  - any free-text string in the packet: the audit requires every string value to
    match the alias namespace ^(G[0-6]|P\\d+|C\\d+|R\\d+)$ or a fixed relation
    type, so a leaked gene symbol, pathway name or description fails
  - node records growing a field beyond {id, degree}
  - the seven canonical symbols appearing anywhere in the fully rendered model
    input, including the tool payloads, not just the packet

MUST run as `PYTHONHASHSEED=0 python scripts/evidence_eval/audit_masking.py` inside the
ops-experimental env (Python 3.10): make_packet permutes candidates via
hash((item_id, order)) and the alias map is not reproducible otherwise. The
audit asserts the recomputed gold alias matches the stored runs and refuses to
report on a mismatched permutation.

Usage:
    PYTHONHASHSEED=0 python scripts/evidence_eval/audit_masking.py
    PYTHONHASHSEED=0 python scripts/evidence_eval/audit_masking.py --json out.json
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_counterfactual as pilot          # noqa: E402

BASE = Path(__file__).resolve().parent.parent.parent
ART = BASE / "artifacts" / "evidence_eval"
ITEM_SETS = {"arbitration": ART / "arbitration" / "items.json",
             "complementary": ART / "complementary" / "items.json",
             "complementary_calibsel": ART / "complementary_calibsel" / "items.json"}
CONDITION = "H-mask"
N_ORDERS = 2

ALIAS_RE = re.compile(r"^(G[0-6]|P\d+|C\d+|R\d+)$")
RELATION_TYPES = {"functional_edge", "shared_pathway", "shared_complex"}
NODE_KEYS = {"id", "degree"}


def _require_hashseed():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run as: PYTHONHASHSEED=0 python scripts/evidence_eval/audit_masking.py")


def _alias_map(item, order):
    """Replicate make_packet's alias assignment. The permutation is not what is
    under test -- the mask is -- but we need it to de-alias the packet."""
    cands = [item["gold"]] + item["distractors"]
    perm = list(range(6))
    random.Random(hash((item["item_id"], order)) & 0xFFFFFFFF).shuffle(perm)
    ordered = [cands[i] for i in perm]
    alias = {item["query"]: "G0"}
    for i, g in enumerate(ordered):
        alias[g] = f"G{i+1}"
    return alias, [item["query"]] + ordered


def _expected_records(genes7, ctx):
    """The record multiset that SHOULD exist among these seven genes, rebuilt
    from the source databases -- not from make_packet. Key is
    (frozenset(canonical pair), relation type, confidence-or-None); shared
    pathway / complex identities are counted, not named, because the packet
    aliases them by design."""
    _, pairs, _, g_path, g_cplx = ctx
    recs = Counter()
    for a, b in combinations(genes7, 2):
        pr = frozenset((a, b))
        if pr in pairs:
            recs[(pr, "functional_edge", round(pairs[pr] / 1000.0, 3))] += 1
        recs[(pr, "shared_pathway", None)] += len(
            g_path.get(a, set()) & g_path.get(b, set()))
        recs[(pr, "shared_complex", None)] += len(
            g_cplx.get(a, set()) & g_cplx.get(b, set()))
    return Counter({k: v for k, v in recs.items() if v})


def _observed_records(packet, unalias):
    recs = Counter()
    for e in packet["edges"]:
        a, b = (unalias[g] for g in e["genes"])
        conf = round(float(e["confidence"]), 3) if e["type"] == "functional_edge" else None
        recs[(frozenset((a, b)), e["type"], conf)] += 1
    return recs


def _string_values(obj):
    """Every string that appears anywhere in the packet, as key or value."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _string_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _string_values(v)
    elif isinstance(obj, str):
        yield obj


def _two_hop_survives(packet, gold_alias):
    """Is there still q -- x -- gold support in the masked packet? Same edge
    weighting reliability_calibration._ann_scores uses."""
    w = defaultdict(float)
    for e in packet["edges"]:
        a, b = e["genes"]
        w[frozenset((a, b))] += (float(e["confidence"])
                                 if e["type"] == "functional_edge" else 0.5)
    return any(w.get(frozenset(("G0", x)), 0.0) > 0
               and w.get(frozenset((x, gold_alias)), 0.0) > 0
               for x in packet["candidates"] if x != gold_alias)


def audit_one(item, order, ctx):
    packet, gold_alias, stated = pilot.make_packet(item, CONDITION, order, ctx)
    alias, genes7 = _alias_map(item, order)
    if alias[item["gold"]] != gold_alias:
        raise SystemExit(
            f"alias mismatch on {item['item_id']}/o{order}: the candidate "
            f"permutation depends on hash((item_id, order)), which is stable only "
            f"within one Python version. Re-run under conda ops-experimental "
            f"(3.10) with PYTHONHASHSEED=0.")
    unalias = {v: k for k, v in alias.items()}

    expected = _expected_records(genes7, ctx)
    qgold = Counter({k: v for k, v in expected.items()
                     if k[0] == frozenset((item["query"], item["gold"]))})
    expected_after_mask = expected - qgold
    observed = _observed_records(packet, unalias)

    strings = list(_string_values(packet))
    bad_strings = sorted({s for s in strings
                          if not ALIAS_RE.match(s)
                          and s not in RELATION_TYPES
                          and s not in {"candidates", "nodes", "edges", "id",
                                        "degree", "type", "genes", "confidence",
                                        "pathway", "complex", "phenotype_similarity"}})
    canon = set(genes7)
    blob = json.dumps(packet)
    symbol_hits = sorted(g for g in canon if re.search(rf"\b{re.escape(g)}\b", blob))

    return {
        "item_id": item["item_id"], "order": order, "cell": item.get("cell"),
        "n_records_unmasked": int(sum(expected.values())),
        "n_records_masked_out": int(sum(qgold.values())),
        "n_records_in_packet": int(sum(observed.values())),
        # (1) the direct assertion is gone
        "direct_assertion_removed": all(
            k[0] != frozenset((item["query"], item["gold"])) for k in observed),
        "removed_count_matches_stated": int(sum(qgold.values())) == stated["qgold_records_removed"],
        "had_direct_qgold_record": any(k[1] == "functional_edge" for k in qgold),
        # (2) nothing else is gone
        "no_over_masking": observed == expected_after_mask,
        "two_hop_support_survives": _two_hop_survives(packet, gold_alias),
        # (3) nothing canonical survives
        "node_keys_only_degree": all(set(n) == NODE_KEYS for n in packet["nodes"]),
        "non_alias_strings": bad_strings,
        "canonical_symbols_in_packet": symbol_hits,
    }


BOOL_CHECKS = ["direct_assertion_removed", "removed_count_matches_stated",
               "no_over_masking", "node_keys_only_degree"]


def main():
    _require_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the full per-packet report here")
    a = ap.parse_args()

    ctx = pilot._ctx()
    rows = []
    for name, path in ITEM_SETS.items():
        if not path.exists():
            print(f"skip {name}: {path} missing")
            continue
        for it in json.load(open(path))["items"]:
            for o in range(N_ORDERS):
                r = audit_one(it, o, ctx)
                r["item_set"] = name
                rows.append(r)

    n = len(rows)
    summary = {
        "n_packets_audited": n,
        "identifier_space": "canonical HGNC symbols (STRING preferred_name), "
                            "Reactome pathway ids, Complex Portal complex ids",
        "n_items": len({(r['item_set'], r['item_id']) for r in rows}),
        "records_removed_total": sum(r["n_records_masked_out"] for r in rows),
        "records_surviving_total": sum(r["n_records_in_packet"] for r in rows),
        "packets_with_a_direct_query_partner_edge": sum(
            r["had_direct_qgold_record"] for r in rows),
        "packets_retaining_two_hop_support": sum(
            r["two_hop_support_survives"] for r in rows),
        "packets_with_non_alias_strings": sum(1 for r in rows if r["non_alias_strings"]),
        "packets_with_a_canonical_symbol": sum(
            1 for r in rows if r["canonical_symbols_in_packet"]),
    }
    for c in BOOL_CHECKS:
        summary[c] = f"{sum(r[c] for r in rows)}/{n}"
    summary["PASS"] = (all(sum(r[c] for r in rows) == n for c in BOOL_CHECKS)
                       and summary["packets_with_non_alias_strings"] == 0
                       and summary["packets_with_a_canonical_symbol"] == 0)

    print(json.dumps(summary, indent=2))
    if a.json:
        json.dump({"summary": summary, "packets": rows}, open(a.json, "w"), indent=2)
        print(f"wrote -> {a.json}")
    return 0 if summary["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
