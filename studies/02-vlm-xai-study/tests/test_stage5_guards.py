"""The guards must actually watch Stage 5. No GPU.

Stage 5's smoke run reported `median 0.0s` over every row. That is not a fast
stage -- it is a dead guard: the runner derives latency from `latency_ms`, and
Stage 5 never wrote the field. The latency guard is the only detector this study
has for the silent shared-memory spill (peak VRAM 7.29 GB against a 7.0 GB
budget), so a 6.5 h run with it inert repeats M1 exactly: a run that reports
healthy because the check can only see what it was handed.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm.backends.fake import FakeBackend  # noqa: E402
from xai_vlm.runner import Guards, run_stage  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "stage5", STUDY_ROOT / "scripts" / "05_sparsity.py")
stage5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage5)

REGION = "(100.0, 100.0, 300.0, 400.0)"


def _row(image_id="0001", source="model"):
    return {"image_id": image_id, "rule_id": "rule_1", "verdict": "YES",
            "source": source, "region": REGION,
            "image_w": "1200", "image_h": "900"}


def _image_source(_image_id):
    return Image.new("RGB", (1200, 900), "white")


def test_work_records_the_latency_it_took():
    """Without this field the runner's latency guard sees 0.0 s forever."""
    work = stage5.make_work_fn(FakeBackend(), _image_source)
    out = work(_row())
    assert "latency_ms" in out
    assert float(out["latency_ms"]) > 0.0


def test_latency_is_recorded_even_when_attention_is_unavailable():
    """A backend that returns no attention still costs a forward pass. Recording
    zero there would drag the median down and mask a genuine slowdown."""
    backend = FakeBackend()
    backend.attention = lambda *a, **k: None
    work = stage5.make_work_fn(backend, _image_source)
    out = work(_row())
    assert out["attention_available"] == "False"
    # Timing the call after the `grid is None` early return would leave this
    # empty, and float("") raises -- which is what this asserts against. The
    # stub returns instantly, so the value itself can legitimately round to 0.0.
    assert out["latency_ms"] != ""
    assert float(out["latency_ms"]) >= 0.0


def test_latency_ms_is_a_declared_result_field():
    """A field absent from RESULT_FIELDS is dropped by the CheckpointWriter, so
    it would never survive to the guard or to the CSV."""
    assert "latency_ms" in stage5.RESULT_FIELDS


def test_the_latency_guard_can_actually_trip_on_stage5_rows(tmp_path):
    work = stage5.make_work_fn(FakeBackend(), _image_source)

    def slow_work(row):
        out = work(row)
        out["latency_ms"] = 99_000.0
        return out

    guards = Guards(min_rows=5, latency_ceiling_s=25.0)
    with pytest.raises(Exception) as excinfo:
        run_stage(
            [_row(f"{i:04d}") for i in range(10)], slow_work,
            tmp_path / "out.csv", tmp_path / "state.json",
            stage5.RESULT_FIELDS, {"model": "fake"},
            checkpoint_every=5, guards=guards,
        )
    assert "median latency" in str(excinfo.value)


def test_a_healthy_stage5_run_does_not_trip_the_guard(tmp_path):
    """The guard must distinguish slow from normal, or it is just an abort."""
    work = stage5.make_work_fn(FakeBackend(), _image_source)
    guards = Guards(min_rows=5, latency_ceiling_s=25.0)
    summary = run_stage(
        [_row(f"{i:04d}") for i in range(10)], work,
        tmp_path / "out.csv", tmp_path / "state.json",
        stage5.RESULT_FIELDS, {"model": "fake"},
        checkpoint_every=5, guards=guards,
    )
    assert summary["errors"] == 0
    assert summary["done"] == 10


def test_stage5_sets_its_own_latency_ceiling():
    """LATENCY_CEILING_S = 25.0 is 2.3x Stage 2's measured 9-11 s median. Stage 5
    reads attention, which forces transformers off sdpa onto the manual attention
    implementation, and its measured healthy median is 16 s -- only 1.56x under
    that ceiling. Inheriting it would abort a 17 h run on ordinary variation,
    which is M2 exactly: a threshold calibrated on a stage that is not this one.
    """
    from xai_vlm.runner import LATENCY_CEILING_S

    assert stage5.LATENCY_CEILING_S > LATENCY_CEILING_S
    # Same 2.3x headroom over the measured median that Stage 2 was given.
    assert stage5.LATENCY_CEILING_S >= stage5.MEASURED_MEDIAN_LATENCY_S * 2.0


def test_stage5_guards_are_built_with_that_ceiling():
    """A constant nothing reads is documentation, not a guard."""
    guards = stage5.make_guards()
    assert guards.latency_ceiling_s == stage5.LATENCY_CEILING_S
