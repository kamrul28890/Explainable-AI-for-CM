"""Stages 4 and 6 must be watched by the guards too. No GPU.

Same defect as Stage 5, found by the same smoke signal: `median 0.0s`. Stage 4
writes its time to `latency_ms_black` and `latency_ms_tokens` and never a plain
`latency_ms`; Stage 6 wrote no latency column at all. Both inherit a `verdict`
column from the CSV they read their input rows from, so the parse-rate guard
graded an earlier stage's output and passed unconditionally.
"""
import importlib.util
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, STUDY_ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stage4 = _load("stage4_g", "04_descriptive_accuracy.py")
stage6 = _load("stage6_g", "06_stability.py")


# --- Stage 4 -----------------------------------------------------------------

def test_stage4_latency_sums_both_masking_calls():
    """A row costs two generations. Reporting one would halve the median and let
    a genuine slowdown sit under the ceiling."""
    row = {"latency_ms_black": 6000.0, "latency_ms_tokens": 4000.0}
    assert stage4.row_latency_s(row) == 10.0


def test_stage4_latency_survives_a_half_finished_row():
    """A row that errored after the first masking has one blank column, and
    float('') raises -- which would kill the run inside the guard."""
    assert stage4.row_latency_s({"latency_ms_black": 6000.0,
                                 "latency_ms_tokens": ""}) == 6.0
    assert stage4.row_latency_s({"latency_ms_black": "",
                                 "latency_ms_tokens": ""}) == 0.0


def test_stage4_row_is_healthy_only_when_both_maskings_answered():
    assert stage4.row_ok({"verdict_black": "YES", "verdict_tokens": "NO"}) is True
    assert stage4.row_ok({"verdict_black": "YES", "verdict_tokens": ""}) is False
    assert stage4.row_ok({"verdict_black": "", "verdict_tokens": ""}) is False


def test_stage4_row_ok_ignores_the_inherited_baseline_verdict():
    """The regions CSV supplies `verdict`, and the default guard reads it. A row
    where both maskings failed must be unhealthy despite that column."""
    row = {"verdict": "YES", "baseline_verdict": "YES",
           "verdict_black": "", "verdict_tokens": ""}
    assert stage4.row_ok(row) is False


# --- Stage 6 -----------------------------------------------------------------

def test_stage6_latency_is_a_declared_result_field():
    assert "latency_ms" in stage6.RESULT_FIELDS


def test_stage6_row_is_healthy_only_when_the_reruns_produced_verdicts():
    assert stage6.row_ok({"verdicts": "YES|YES|NO|YES|NO"}) is True
    assert stage6.row_ok({"verdicts": ""}) is False


def test_stage6_row_ok_requires_every_rerun_to_have_answered():
    """`||` between reruns means a sampled decode produced nothing parseable.
    Agreement computed over blanks is not a stability measurement."""
    assert stage6.row_ok({"verdicts": "YES||NO|YES|NO"}) is False


def test_stage6_row_ok_ignores_the_inherited_baseline_verdict():
    row = {"verdict": "YES", "baseline_verdict": "YES", "verdicts": ""}
    assert stage6.row_ok(row) is False
