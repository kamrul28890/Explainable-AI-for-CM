"""Stage 3: region source and image dimensions. No model, no image decode."""
from xai_vlm.metrics.regions import (
    REGION_FIELDS,
    build_region_rows,
    classify_source,
    no_region_rate,
)


def test_classify_source_model_when_a_box_is_present():
    assert classify_source("(10.0, 20.0, 30.0, 40.0)") == "model"


def test_classify_source_none_when_empty():
    assert classify_source("") == "none"
    assert classify_source("   ") == "none"


def test_no_region_rate():
    rows = [{"source": "model"}, {"source": "none"}, {"source": "model"}, {"source": "none"}]
    assert no_region_rate(rows) == 0.5


def test_build_region_rows_carries_dimensions_and_source():
    baseline = [
        {"image_id": "0001", "rule_id": "rule_1", "verdict": "YES",
         "region": "(10.0, 20.0, 30.0, 40.0)"},
        {"image_id": "0002", "rule_id": "rule_3", "verdict": "NO", "region": ""},
    ]
    sizes = {"0001": (1200, 900), "0002": (675, 1200)}
    rows = build_region_rows(baseline, sizes)

    assert [r["source"] for r in rows] == ["model", "none"]
    assert rows[0]["image_w"] == 1200 and rows[0]["image_h"] == 900
    assert rows[1]["image_w"] == 675 and rows[1]["image_h"] == 1200
    assert list(rows[0]) == REGION_FIELDS


def test_missing_dimensions_are_recorded_as_zero_not_guessed():
    """An image the size pass could not read must be visible, not defaulted."""
    baseline = [{"image_id": "9999", "rule_id": "rule_1", "verdict": "YES",
                 "region": "(1.0, 2.0, 3.0, 4.0)"}]
    rows = build_region_rows(baseline, {})
    assert rows[0]["image_w"] == 0 and rows[0]["image_h"] == 0
