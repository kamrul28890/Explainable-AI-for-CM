"""How run_stage decides whether a row was healthy. No GPU.

The guards were written for Stage 2 and hardcode its column names: latency from
`latency_ms`, success from a non-empty `verdict`. Every later stage writes
different columns, and because `out_row` is `{**input_row, **result}`, a stage
reading its input from an earlier stage's CSV *inherits that CSV's verdict
column*. The guard then grades the previous stage's output and passes, while
latency reads a missing field as 0.0 s and can never trip.

Measured before this fix: Stage 4 (a 70 h run), Stage 5 and Stage 6 all had both
guards inert. Stage 2, whose columns they were written against, was the only one
being watched.

A stage must therefore be able to say how its own health is read, and the
defaults must not change for the stage that already works.
"""
import pytest

from xai_vlm.runner import Guards, GuardTripped, run_stage

FIELDS = ["image_id", "rule_id", "verdict", "latency_ms", "attention_available",
          "error"]


def _rows(n=10):
    return [{"image_id": f"{i:04d}", "rule_id": "rule_1"} for i in range(n)]


def _run(tmp_path, work_fn, *, guards, **kw):
    return run_stage(_rows(), work_fn, tmp_path / "out.csv",
                     tmp_path / "state.json", FIELDS, {"model": "fake"},
                     checkpoint_every=5, guards=guards, **kw)


def test_default_latency_still_reads_latency_ms(tmp_path):
    """Stage 2's wiring must be untouched by this change."""
    guards = Guards(min_rows=5, latency_ceiling_s=25.0)
    with pytest.raises(GuardTripped, match="median latency"):
        _run(tmp_path, lambda r: {"verdict": "YES", "latency_ms": 99_000.0},
             guards=guards)


def test_default_success_still_reads_verdict(tmp_path):
    guards = Guards(min_rows=5, parse_rate_floor=0.95)
    with pytest.raises(GuardTripped, match="parse rate"):
        _run(tmp_path, lambda r: {"verdict": "", "latency_ms": 10.0},
             guards=guards)


def test_a_stage_can_supply_its_own_latency(tmp_path):
    """Stage 4 spends its time in latency_ms_black + latency_ms_tokens and never
    writes a plain latency_ms, so the default reads 0.0 s on every row."""
    guards = Guards(min_rows=5, latency_ceiling_s=25.0)

    def work(_row):
        return {"latency_ms_black": 40_000.0, "latency_ms_tokens": 40_000.0,
                "verdict": "YES"}

    with pytest.raises(GuardTripped, match="median latency"):
        _run(tmp_path, work, guards=guards,
             latency_fn=lambda r: (float(r.get("latency_ms_black") or 0.0)
                                   + float(r.get("latency_ms_tokens") or 0.0)) / 1000.0)


def test_a_stage_can_supply_its_own_success_test(tmp_path):
    """Stage 5's analog of 'parsed' is whether attention came back at all."""
    guards = Guards(min_rows=5, parse_rate_floor=0.95)

    def work(_row):
        # Inherits a verdict from the regions CSV, so the default guard passes.
        return {"verdict": "YES", "attention_available": "False",
                "latency_ms": 10.0}

    with pytest.raises(GuardTripped, match="parse rate"):
        _run(tmp_path, work, guards=guards,
             ok_fn=lambda r: r.get("attention_available") == "True")


def test_a_healthy_run_with_custom_wiring_does_not_trip(tmp_path):
    """The custom hooks must distinguish sick from healthy, not just abort."""
    guards = Guards(min_rows=5, parse_rate_floor=0.95, latency_ceiling_s=25.0)
    summary = _run(
        tmp_path,
        lambda r: {"verdict": "YES", "attention_available": "True",
                   "latency_ms": 9_500.0},
        guards=guards,
        ok_fn=lambda r: r.get("attention_available") == "True",
        latency_fn=lambda r: float(r.get("latency_ms") or 0.0) / 1000.0,
    )
    assert summary["done"] == 10
    assert summary["errors"] == 0


def test_an_errored_row_is_never_counted_as_healthy(tmp_path):
    """A row that raised has no latency and no result. Counting it as parsed
    would let a run of pure failures satisfy the parse-rate floor."""
    guards = Guards(min_rows=5, parse_rate_floor=0.95)

    def work(_row):
        raise RuntimeError("boom")

    with pytest.raises(GuardTripped, match="parse rate"):
        _run(tmp_path, work, guards=guards,
             ok_fn=lambda r: r.get("attention_available") == "True")
