"""Stage 9: bounded completeness. Pure classification over two inputs."""
import pytest

from xai_vlm.metrics.completeness import classify_completeness, completeness_rates


def test_no_region_is_no_usable_explanation():
    assert classify_completeness("none", "") == "no_usable_explanation"


def test_region_that_mattered_is_supported():
    assert classify_completeness("model", "True") == "explanation_supported"


def test_region_that_did_not_matter_is_weak():
    assert classify_completeness("model", "False") == "explanation_weak"


def test_a_region_with_no_stage4_result_is_not_silently_supported():
    """A row that Stage 4 never covered must not be counted either way."""
    assert classify_completeness("model", "") == "not_measured"


def test_rates_sum_to_one():
    rows = [
        {"completeness": "no_usable_explanation"},
        {"completeness": "explanation_supported"},
        {"completeness": "explanation_weak"},
        {"completeness": "explanation_weak"},
    ]
    r = completeness_rates(rows)
    assert r["explanation_supported"] == pytest.approx(0.25)
    assert r["explanation_weak"] == pytest.approx(0.5)
    assert sum(v for k, v in r.items() if k != "n") == pytest.approx(1.0)


def test_empty_input_is_nan_not_zero():
    r = completeness_rates([])
    assert r["n"] == 0
    assert r["explanation_supported"] != r["explanation_supported"]  # nan
