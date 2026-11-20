"""Flip logic for Stage 4. Pure functions."""
import pytest

from xai_vlm.metrics.descriptive_accuracy import (
    MASKING_METHODS,
    artefact_floor,
    flip_rate,
    is_flip,
)


def test_methods_are_black_and_tokens():
    assert MASKING_METHODS == ("black", "tokens")


def test_a_changed_verdict_is_a_flip():
    assert is_flip("YES", "NO") is True
    assert is_flip("NO", "YES") is True


def test_an_unchanged_verdict_is_not_a_flip():
    assert is_flip("YES", "YES") is False
    assert is_flip("NO", "NO") is False


def test_an_unparseable_masked_reply_is_not_a_flip():
    """An empty verdict means the reply could not be read, which is a parse
    failure -- counting it as a flip would inflate faithfulness with noise."""
    assert is_flip("YES", "") is False


def test_flip_rate_over_yes_rows_only():
    rows = [
        {"baseline_verdict": "YES", "flip_tokens": "True"},
        {"baseline_verdict": "YES", "flip_tokens": "False"},
        {"baseline_verdict": "NO", "flip_tokens": "True"},
    ]
    assert flip_rate(rows, "tokens") == pytest.approx(0.5)


def test_artefact_floor_is_the_no_to_yes_rate():
    """Masking only removes information, so a NO cannot rationally become a YES.
    That rate measures how much the intervention perturbs the model by itself."""
    rows = [
        {"baseline_verdict": "NO", "flip_tokens": "True"},
        {"baseline_verdict": "NO", "flip_tokens": "False"},
        {"baseline_verdict": "NO", "flip_tokens": "False"},
        {"baseline_verdict": "NO", "flip_tokens": "False"},
        {"baseline_verdict": "YES", "flip_tokens": "True"},
    ]
    assert artefact_floor(rows, "tokens") == pytest.approx(0.25)


def test_rates_are_nan_when_the_class_is_empty():
    assert flip_rate([], "tokens") != flip_rate([], "tokens")  # nan
