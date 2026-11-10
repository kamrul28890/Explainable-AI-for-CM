"""The section 5.5 human-audit draw. Metadata only -- no images, no model."""
import importlib.util
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "audit", STUDY_ROOT / "scripts" / "12_audit_sample.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def _records(n_rich=280, n_poor=210):
    recs = []
    for tier, n in (("rich info", n_rich), ("poor info", n_poor)):
        for i in range(n):
            recs.append({"image_id": f"{tier[0]}{i:04d}", "quality_of_info": tier,
                         "rule_1_violation": {"bounding_box": [[0.1, 0.1, 0.2, 0.2]],
                                              "reason": "no hat"} if i % 7 == 0 else None,
                         "rule_2_violation": None, "rule_3_violation": None,
                         "rule_4_violation": None, "excavator": []})
    return recs


def test_fifty_per_quality_tier():
    got = audit.draw(_records(), per_tier=50, seed=42)
    assert len(got) == 100
    assert sum(r["quality_of_info"] == "rich info" for r in got) == 50
    assert len({r["image_id"] for r in got}) == 100


def test_the_draw_is_seeded_and_independent_of_record_order():
    """A dataset re-read in another order must give the same 100 images."""
    recs = _records()
    a = audit.draw(recs, per_tier=50, seed=42)
    b = audit.draw(list(reversed(recs)), per_tier=50, seed=42)
    assert [r["image_id"] for r in a] == [r["image_id"] for r in b]
    assert a != audit.draw(recs, per_tier=50, seed=7)


def test_each_tier_is_half_hazard_half_compliant():
    """Decided 2026-11-10: the pure random draw (seed 42) held 3 hazard images in
    100, too few to estimate how often a MARKED violation is wrong. Stratified
    by hazard within tier instead, reweighted to prevalence when reported."""
    from xai_vlm.rules import violated_rules
    got = audit.draw(_records(), per_tier=50, seed=42)
    for tier in audit.TIERS:
        sub = [r for r in got if r["quality_of_info"] == tier]
        assert sum(bool(violated_rules(r)) for r in sub) == 25
        assert sum(not violated_rules(r) for r in sub) == 25


def test_item_carries_annotations_in_normalised_boxes_and_rule_applicability():
    rec = _records()[0]
    item = audit.to_item(rec, size=(1200, 900), model_rows={})
    assert item["annotations"]["rule_1"] == {"reason": "no hat",
                                            "boxes": [[0.1, 0.1, 0.2, 0.2]]}
    assert "rule_2" not in item["annotations"]
    assert item["applicable"]["rule_4"] is False       # no excavator in the scene
    assert item["model"] == {}


def test_model_regions_are_normalised_by_the_original_image_size():
    rec = _records()[0]
    rows = {"rule_1": {"verdict": "YES", "region": "(120.0, 90.0, 240.0, 180.0)",
                       "rationale": "a worker has no hat"}}
    item = audit.to_item(rec, size=(1200, 900), model_rows=rows)
    assert item["model"]["rule_1"]["region"] == [0.1, 0.1, 0.2, 0.2]
    assert item["model"]["rule_1"]["verdict"] == "YES"
