"""Self-check for the grounded comprehension control (reliability_calibration)."""
import reliability_calibration as rc


def test_comprehension_set_spans_regimes_capped_deterministic():
    s1 = rc.build_comprehension_set()
    s2 = rc.build_comprehension_set()
    assert [it["item_id"] for it, _ in s1] == [it["item_id"] for it, _ in s2], "not deterministic"

    by = {}
    for it, lab in s1:
        assert rc._bin(it["phen_rank"]) == lab, "regime label mismatch"
        by.setdefault(lab, []).append(it["item_id"])

    assert set(by) == {"1-2", "3-4", "5-6"}, f"missing regimes: {set(by)}"
    for lab, ids in by.items():
        assert len(ids) == len(set(ids)), f"dup items in {lab}"
        assert len(ids) <= rc.COMP_CAP.get(lab, len(ids)), f"{lab} over cap"
    # tail (the dissociation regime) must be the full 29-item conflict set
    assert len(by["5-6"]) == len(rc.load_conflict_items()), "tail != conflict set"


if __name__ == "__main__":
    test_comprehension_set_spans_regimes_capped_deterministic()
    print("OK")
