"""Floor-then-proportional subsampling, shared by Stages 4, 6 and 8."""
from collections import Counter

from xai_vlm.sampling import floor_then_proportional_sample


def _rows(counts: dict, group: str) -> list[dict]:
    """counts: {rule_id: n}, all in the same group."""
    out = []
    for rule, n in counts.items():
        for i in range(n):
            out.append({"group": group, "rule_id": rule, "id": f"{group}-{rule}-{i}"})
    return out


def test_every_group_gets_its_target_when_pools_allow():
    rows = _rows({"rule_1": 500, "rule_2": 500}, "YES")
    picked = floor_then_proportional_sample(rows, group_key="group",
                                            stratify_key="rule_id",
                                            targets={"YES": 200}, floor=40, seed=1)
    assert len(picked) == 200


def test_a_starved_stratum_gets_its_floor_even_when_pool_is_small():
    """The scenario that motivated this function: rule_4 with only 70 rows in
    the whole pool must not be crowded out by proportional allocation alone."""
    rows = _rows({"rule_1": 1239, "rule_2": 343, "rule_3": 969, "rule_4": 70}, "YES")
    picked = floor_then_proportional_sample(rows, group_key="group",
                                            stratify_key="rule_id",
                                            targets={"YES": 800}, floor=40, seed=42)
    counts = Counter(r["rule_id"] for r in picked)
    assert counts["rule_4"] >= 40


def test_a_pool_smaller_than_the_floor_gives_everything_it_has():
    rows = _rows({"rule_1": 100, "rule_3": 13}, "NO")
    picked = floor_then_proportional_sample(rows, group_key="group",
                                            stratify_key="rule_id",
                                            targets={"NO": 400}, floor=40, seed=42)
    counts = Counter(r["rule_id"] for r in picked)
    assert counts["rule_3"] == 13


def test_groups_below_target_take_everything_in_the_pool():
    rows = _rows({"rule_1": 5}, "NO")
    picked = floor_then_proportional_sample(rows, group_key="group",
                                            stratify_key="rule_id",
                                            targets={"NO": 400}, floor=40, seed=42)
    assert len(picked) == 5


def test_deterministic_for_a_fixed_seed():
    rows = _rows({"rule_1": 500, "rule_2": 500}, "YES")
    a = floor_then_proportional_sample(rows, "group", "rule_id", {"YES": 200}, 40, seed=7)
    b = floor_then_proportional_sample(rows, "group", "rule_id", {"YES": 200}, 40, seed=7)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_different_seeds_draw_different_samples():
    rows = _rows({"rule_1": 500, "rule_2": 500}, "YES")
    a = floor_then_proportional_sample(rows, "group", "rule_id", {"YES": 200}, 40, seed=1)
    b = floor_then_proportional_sample(rows, "group", "rule_id", {"YES": 200}, 40, seed=2)
    assert [r["id"] for r in a] != [r["id"] for r in b]


def test_multiple_target_groups_are_independent():
    rows = _rows({"rule_1": 500}, "YES") + _rows({"rule_1": 500}, "NO")
    picked = floor_then_proportional_sample(rows, "group", "rule_id",
                                            {"YES": 100, "NO": 50}, floor=40, seed=1)
    counts = Counter(r["group"] for r in picked)
    assert counts["YES"] == 100
    assert counts["NO"] == 50
