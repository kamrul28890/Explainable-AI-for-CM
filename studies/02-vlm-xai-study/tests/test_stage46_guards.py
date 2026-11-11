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


def test_stage6_grades_each_reply_not_each_row():
    """Decided 2026-11-10, after bf16 tripped the guard at T=0.7: 13 of 500
    sampled replies were unreadable (mostly a bare "NONE"), 2.6% per reply, but
    one bad reply in five failed the whole row, so 10% of rows. Stage 2's parse
    rate is per reply; Stage 6's now is too. An unreadable reply still counts as
    a disagreement in the agreement rate -- nothing about the measurement moved."""
    assert stage6.reply_parse_share({"verdicts": "YES|YES|NO|YES|NO"}) == 1.0
    assert stage6.reply_parse_share({"verdicts": "YES||NO|YES|NO"}) == 0.8
    assert stage6.reply_parse_share({"verdicts": "||||"}) == 0.0
    assert stage6.reply_parse_share({"verdicts": ""}) == 0.0


def test_stage6_parse_share_ignores_the_inherited_baseline_verdict():
    row = {"verdict": "YES", "baseline_verdict": "YES", "verdicts": ""}
    assert stage6.reply_parse_share(row) == 0.0


def test_the_bf16_failure_rate_passes_and_a_real_collapse_does_not():
    from xai_vlm.runner import GUARD_MIN_ROWS, GuardTripped, Guards
    import pytest
    g = Guards()
    for i in range(GUARD_MIN_ROWS):                 # 13 bad replies in 500
        g.record(parsed_ok=stage6.reply_parse_share(
            {"verdicts": "YES||NO|YES|NO" if i < 13 else "YES|NO|YES|NO|YES"}), latency_s=9.0)
    g.check()
    g = Guards()
    for i in range(GUARD_MIN_ROWS):                 # half the replies blank
        g.record(parsed_ok=stage6.reply_parse_share({"verdicts": "YES|||NO|"}), latency_s=9.0)
    with pytest.raises(GuardTripped):
        g.check()
