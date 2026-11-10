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


# --- attentional basis (F7 redesign) -----------------------------------------
from xai_vlm.metrics.completeness import classify_completeness_attention  # noqa: E402


def _sp(available="True", mass="0.30", area="0.10"):
    return {"attention_available": available, "mass_inside_region": mass,
            "region_area_fraction": area}


def test_attention_inside_the_region_beyond_its_area_is_supported():
    assert classify_completeness_attention("model", _sp(mass="0.30", area="0.10")) \
        == "explanation_supported"


def test_attention_at_or_below_chance_is_weak():
    """Uniform attention puts exactly the area fraction inside; that is chance."""
    assert classify_completeness_attention("model", _sp(mass="0.10", area="0.10")) \
        == "explanation_weak"
    assert classify_completeness_attention("model", _sp(mass="0.05", area="0.10")) \
        == "explanation_weak"


def test_no_region_stays_no_usable_explanation_whatever_attention_did():
    assert classify_completeness_attention("none", _sp()) == "no_usable_explanation"


def test_unavailable_or_missing_attention_is_not_measured():
    """Same rule as the ablation basis: missing input is never a verdict."""
    assert classify_completeness_attention("model", _sp(available="False")) == "not_measured"
    assert classify_completeness_attention("model", {}) == "not_measured"
    assert classify_completeness_attention("model", _sp(mass="")) == "not_measured"


# --- the Stage 9 script: flag, file names, frozen output untouched ------------
import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "stage9", Path(__file__).resolve().parents[1] / "scripts" / "09_completeness.py")
stage9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage9)


def test_ablation_basis_keeps_the_frozen_file_name():
    assert stage9.output_name("m", "ablation") == "completeness_m.csv"


def test_attention_basis_writes_beside_it_never_over_it():
    assert stage9.output_name("m", "attention") == "completeness_attention_m.csv"


def test_attention_rows_join_stage5_by_image_and_rule():
    regions = [{"image_id": "a", "rule_id": "rule_1", "verdict": "YES", "source": "model"},
               {"image_id": "b", "rule_id": "rule_1", "verdict": "NO", "source": "none"},
               {"image_id": "c", "rule_id": "rule_2", "verdict": "YES", "source": "model"}]
    sp = [{"image_id": "a", "rule_id": "rule_1", **_sp(mass="0.4", area="0.1")}]
    rows = stage9.build_attention_rows(regions, sp)
    got = {r["image_id"]: r["completeness"] for r in rows}
    assert got == {"a": "explanation_supported", "b": "no_usable_explanation",
                   "c": "not_measured"}
    assert rows[0]["attention_lift"] == "4.0000"
