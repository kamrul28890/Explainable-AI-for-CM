# Study 2 Plan 2 — Resumable Sampling and Baseline Inference — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Stage 1 (the frozen sample manifest) and Stage 2 (resumable baseline inference), plus the reusable `runner.py` that Stages 3–5 will share.

**Architecture:** A backend-agnostic `runner.py` owns resumption, fingerprinting and guard rails, and is tested entirely with `FakeBackend` on CPU. Two thin stage scripts drive it: `01_build_sample.py` builds and freezes the manifest from dataset metadata only, `02_baseline_inference.py` streams images and calls one `Backend` per manifest row.

**Tech Stack:** Python 3.10, pytest, `datasets` (streaming parquet), PIL, transformers 4.49.0 (Stage 2 only, via the existing `QwenBackend`).

**Spec:** `docs/superpowers/specs/2026-08-18-study2-resumable-battery-design.md`

## Global Constraints

- **Work from `studies/02-vlm-xai-study/`.** All relative paths below are from there.
- **Python is `../../pilot/.venv/Scripts/python`** (Windows venv layout). Never `python` bare.
- **Do not modify** `rules.py`, `parsing.py`, `sampling.py`, `masking.py`, or `backends/`. Every helper needed already exists and is tested. If one seems wrong, stop and report rather than editing it.
- **Do not touch `pilot/results/`** or anything under `studies/01-florence2-pilot/`. That pilot is frozen and cited.
- **`runner.py` must not import torch or transformers.** It is CPU-testable by construction. Only `02_baseline_inference.py` imports a real backend.
- **Correctness settings are fixed** and come from `config.py`: `COMPUTE_DTYPE="bfloat16"`, `ATTENTION_IMPL="sdpa"`, `MAX_PIXELS=640*PATCH_BLOCK`. Never override them in a script.
- **Pre-registered thresholds**, not tunable: parse-rate abort `< 0.95`; latency abort `> 12.0` s median (`2 x LATENCY_BUDGET_S`); guards need `>= 100` rows in the current process before they activate.
- **Manifest columns are exactly** `image_id, rule_id, stratum, is_violation, is_negative_control, quality_of_info, illumination, camera_distance, view` — the schema pre-registered in the architecture document. No `expected` column.
- **TDD.** Write the failing test, watch it fail, implement, watch it pass, commit.
- **Commit messages** end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `src/xai_vlm/runner.py` | **Create.** Resume state, append-only checkpoint writer, fingerprint verification, guard rails, progress reporting. No model knowledge. |
| `tests/test_runner.py` | **Create.** CPU tests for every runner behaviour, driven by `FakeBackend`. |
| `scripts/01_build_sample.py` | **Replace stub.** Scan, stratify, assign rules, freeze manifest. |
| `tests/test_build_sample.py` | **Create.** Determinism, overwrite refusal, manifest invariants. |
| `scripts/02_baseline_inference.py` | **Replace stub.** Stream images, drive `runner.py` over one `Backend`. |
| `tests/test_baseline_inference.py` | **Create.** End-to-end over `FakeBackend` with a fake image source. |

Task order is dependency order: Tasks 1–4 build `runner.py` bottom-up, Task 5 is Stage 1, Tasks 6–7 are Stage 2, Task 8 is GPU validation.

---

### Task 1: Checkpoint file — append-only writer and done-set reader

**Files:**
- Create: `src/xai_vlm/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `RowKey = tuple[str, str]` — `(image_id, rule_id)`.
  - `class CheckpointWriter(path: Path, fieldnames: list[str])`
  - `CheckpointWriter.done_keys() -> set[RowKey]`
  - `CheckpointWriter.append(row: dict) -> None`
  - `CheckpointWriter.close() -> None`

- [ ] **Step 1: Write the failing test**

```python
# studies/02-vlm-xai-study/tests/test_runner.py
"""Tests for the resumable runner. No GPU, no model -- FakeBackend only."""
import csv

from xai_vlm.runner import CheckpointWriter

FIELDS = ["image_id", "rule_id", "verdict"]


def test_done_keys_empty_when_no_file(tmp_path):
    w = CheckpointWriter(tmp_path / "out.csv", FIELDS)
    assert w.done_keys() == set()


def test_append_writes_header_once_and_rows_are_readable(tmp_path):
    path = tmp_path / "out.csv"
    w = CheckpointWriter(path, FIELDS)
    w.append({"image_id": "0001", "rule_id": "rule_1", "verdict": "YES"})
    w.append({"image_id": "0002", "rule_id": "rule_3", "verdict": "NO"})
    w.close()

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["image_id"] for r in rows] == ["0001", "0002"]
    assert path.read_text(encoding="utf-8").count("image_id,rule_id,verdict") == 1


def test_row_is_on_disk_before_the_next_append(tmp_path):
    """Durability: a crash after row N must not lose row N."""
    path = tmp_path / "out.csv"
    w = CheckpointWriter(path, FIELDS)
    w.append({"image_id": "0001", "rule_id": "rule_1", "verdict": "YES"})
    # Read with a separate handle while the writer is still open.
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 1


def test_done_keys_reads_existing_file_and_reopen_appends(tmp_path):
    path = tmp_path / "out.csv"
    w = CheckpointWriter(path, FIELDS)
    w.append({"image_id": "0001", "rule_id": "rule_1", "verdict": "YES"})
    w.close()

    w2 = CheckpointWriter(path, FIELDS)
    assert w2.done_keys() == {("0001", "rule_1")}
    w2.append({"image_id": "0002", "rule_id": "rule_1", "verdict": "NO"})
    w2.close()

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 2
    assert path.read_text(encoding="utf-8").count("image_id,rule_id,verdict") == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xai_vlm.runner'`

- [ ] **Step 3: Write the implementation**

```python
# studies/02-vlm-xai-study/src/xai_vlm/runner.py
"""Resumable stage execution: checkpointing, fingerprinting, guard rails.

Stage 2 runs for hours. A crash at hour eight without resumability costs a day --
but the worse failure is quieter: half a run at one configuration concatenated with
half at another produces a complete-looking CSV that no downstream metric can tell
is wrong. This module exists to make the first cheap and the second impossible.

Deliberately free of torch and transformers. Every behaviour here is exercised on
CPU against FakeBackend, because logic that can only be tested on the one machine
with a GPU is logic that does not get tested.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

# (image_id, rule_id) -- the natural key of a manifest row.
#
# Resume is keyed on this, never on row position. Images arrive in dataset
# streaming order rather than manifest order, so a positional cursor would desync
# the first time iteration changed, and would do so silently.
RowKey = tuple[str, str]


class CheckpointWriter:
    """Append-only CSV writer whose rows are on disk before the next call begins.

    Buffering would lose up to a bufferful on a crash, and rewriting the whole file
    periodically risks truncating it mid-write. Appending one row and flushing to
    the platform costs a few milliseconds against a ~3 s model call.
    """

    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self._fh = None
        self._writer = None

    def done_keys(self) -> set[RowKey]:
        """Keys already recorded, read from the file on disk."""
        if not self.path.exists():
            return set()
        with self.path.open(newline="", encoding="utf-8") as fh:
            return {
                (r["image_id"], r["rule_id"])
                for r in csv.DictReader(fh)
                if r.get("image_id")
            }

    def _open(self) -> None:
        if self._fh is not None:
            return
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        if is_new:
            self._writer.writeheader()

    def append(self, row: dict) -> None:
        self._open()
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_runner.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add studies/02-vlm-xai-study/src/xai_vlm/runner.py studies/02-vlm-xai-study/tests/test_runner.py
git commit -m "feat(study2): add append-only checkpoint writer

Rows are flushed and fsync'd before the next call begins, so a crash loses
nothing. Resume reads back (image_id, rule_id) keys rather than a row count.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Run fingerprint — refuse to stitch incompatible runs

**Files:**
- Modify: `src/xai_vlm/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `RowKey`, `CheckpointWriter` from Task 1.
- Produces:
  - `def manifest_sha256(path: Path) -> str`
  - `class FingerprintMismatch(RuntimeError)`
  - `class RunState(path: Path)`
  - `RunState.verify_or_create(fingerprint: dict, force: bool = False) -> None`

- [ ] **Step 1: Write the failing test**

```python
# append to studies/02-vlm-xai-study/tests/test_runner.py
import json

import pytest

from xai_vlm.runner import FingerprintMismatch, RunState, manifest_sha256


def _fp(**over):
    fp = {"manifest_sha256": "abc", "model": "qwen7b-4bit", "max_pixels": 501760}
    fp.update(over)
    return fp


def test_manifest_sha256_is_stable_and_content_sensitive(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("image_id,rule_id\n0001,rule_1\n", encoding="utf-8")
    b.write_text("image_id,rule_id\n0001,rule_2\n", encoding="utf-8")
    assert manifest_sha256(a) == manifest_sha256(a)
    assert manifest_sha256(a) != manifest_sha256(b)


def test_first_run_writes_the_state_file(tmp_path):
    state = RunState(tmp_path / "run.state.json")
    state.verify_or_create(_fp())
    written = json.loads((tmp_path / "run.state.json").read_text(encoding="utf-8"))
    assert written["model"] == "qwen7b-4bit"


def test_matching_fingerprint_resumes_silently(tmp_path):
    state = RunState(tmp_path / "run.state.json")
    state.verify_or_create(_fp())
    RunState(tmp_path / "run.state.json").verify_or_create(_fp())  # must not raise


def test_changed_config_aborts_and_names_the_field(tmp_path):
    state = RunState(tmp_path / "run.state.json")
    state.verify_or_create(_fp())

    with pytest.raises(FingerprintMismatch) as exc:
        RunState(tmp_path / "run.state.json").verify_or_create(_fp(max_pixels=1003520))
    assert "max_pixels" in str(exc.value)


def test_changed_manifest_aborts(tmp_path):
    state = RunState(tmp_path / "run.state.json")
    state.verify_or_create(_fp())
    with pytest.raises(FingerprintMismatch):
        RunState(tmp_path / "run.state.json").verify_or_create(_fp(manifest_sha256="zzz"))


def test_force_overrides_and_records_that_it_was_forced(tmp_path):
    path = tmp_path / "run.state.json"
    RunState(path).verify_or_create(_fp())
    RunState(path).verify_or_create(_fp(max_pixels=1003520), force=True)

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["forced"] is True
    assert written["max_pixels"] == 1003520
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_runner.py -v -k "fingerprint or sha256 or state or force or config or manifest"`
Expected: FAIL — `ImportError: cannot import name 'RunState'`

- [ ] **Step 3: Write the implementation**

Append to `src/xai_vlm/runner.py`:

```python
import hashlib
import json


def manifest_sha256(path: Path) -> str:
    """Content hash of the manifest, read in binary so line endings count."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class FingerprintMismatch(RuntimeError):
    """The run on disk was produced under a different configuration.

    Raised rather than warned about. A mismatch means the rows already written and
    the rows about to be written answer different questions, and once concatenated
    nothing downstream can separate them again.
    """


class RunState:
    """The sidecar recording what configuration produced a results file."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    def verify_or_create(self, fingerprint: dict, force: bool = False) -> None:
        existing = self.load()
        if existing is None:
            self._write(dict(fingerprint))
            return

        diffs = [
            f"  {k}: on disk {existing.get(k)!r} -> now {v!r}"
            for k, v in fingerprint.items()
            if existing.get(k) != v
        ]
        if diffs and not force:
            raise FingerprintMismatch(
                "this results file was produced under a different configuration:\n"
                + "\n".join(diffs)
                + "\n\nResuming would concatenate rows answering different questions.\n"
                "Start a fresh output file, or pass --force-resume if you are "
                "certain the difference does not affect the measurement."
            )
        if diffs:
            merged = dict(fingerprint)
            merged["forced"] = True
            self._write(merged)

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_runner.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add studies/02-vlm-xai-study/src/xai_vlm/runner.py studies/02-vlm-xai-study/tests/test_runner.py
git commit -m "feat(study2): fingerprint runs and refuse mismatched resumes

A sidecar state.json records the manifest hash and every correctness setting.
Resuming with any of them changed aborts with a field-level diff, because the
concatenated CSV would otherwise look complete and be wrong.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Guard rails — abort on parse-rate collapse and latency spill

**Files:**
- Modify: `src/xai_vlm/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces:
  - `PARSE_RATE_FLOOR = 0.95`, `LATENCY_CEILING_S = 12.0`, `GUARD_MIN_ROWS = 100`
  - `class GuardTripped(RuntimeError)`
  - `class Guards(parse_rate_floor=..., latency_ceiling_s=..., min_rows=...)`
  - `Guards.record(parsed_ok: bool, latency_s: float) -> None`
  - `Guards.check() -> None` — raises `GuardTripped`
  - `Guards.n`, `Guards.parse_rate`, `Guards.median_latency_s`

- [ ] **Step 1: Write the failing test**

```python
# append to studies/02-vlm-xai-study/tests/test_runner.py
from xai_vlm.runner import GUARD_MIN_ROWS, GuardTripped, Guards


def test_guards_are_inert_below_the_minimum_row_count():
    """A --limit 20 smoke run must never trip a guard."""
    g = Guards()
    for _ in range(20):
        g.record(parsed_ok=False, latency_s=99.0)
    g.check()  # must not raise


def test_parse_rate_collapse_aborts_at_the_first_checkpoint():
    g = Guards()
    for i in range(GUARD_MIN_ROWS):
        g.record(parsed_ok=(i % 2 == 0), latency_s=3.0)  # 50%
    with pytest.raises(GuardTripped) as exc:
        g.check()
    assert "parse rate" in str(exc.value)


def test_healthy_run_does_not_trip():
    g = Guards()
    for _ in range(GUARD_MIN_ROWS):
        g.record(parsed_ok=True, latency_s=2.9)
    g.check()  # must not raise


def test_latency_spill_aborts():
    g = Guards()
    for _ in range(GUARD_MIN_ROWS):
        g.record(parsed_ok=True, latency_s=90.0)
    with pytest.raises(GuardTripped) as exc:
        g.check()
    assert "latency" in str(exc.value)


def test_latency_guard_tolerates_the_declared_budget():
    """6.0s is the pre-registered budget; it must not abort a run."""
    g = Guards()
    for _ in range(GUARD_MIN_ROWS):
        g.record(parsed_ok=True, latency_s=6.0)
    g.check()  # must not raise
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_runner.py -v -k guard`
Expected: FAIL — `ImportError: cannot import name 'Guards'`

- [ ] **Step 3: Write the implementation**

Append to `src/xai_vlm/runner.py`:

```python
import statistics

# PRE-REGISTERED, fixed before any Stage 2 numbers were seen -- the same discipline
# the Stage 0 gate thresholds were set under. Not to be tuned to fit a run.
PARSE_RATE_FLOOR = 0.95

# 2 x LATENCY_BUDGET_S (6.0), not 2 x the observed Stage 0 median (2.9). Twice the
# observed median is 5.8 s, which sits *below* the budget the study already declared
# acceptable, so normal variation across a larger image set would abort healthy runs.
# The failure this catches -- Windows' driver silently spilling to shared memory
# rather than raising OOM -- costs roughly 30x, so it breaches 12.0 s unambiguously.
LATENCY_CEILING_S = 12.0

# Guards evaluate rows from the CURRENT PROCESS only, and not until this many exist.
# A resumed run is therefore never aborted by history it cannot change, and a short
# --limit smoke run never trips them.
GUARD_MIN_ROWS = 100


class GuardTripped(RuntimeError):
    """A pre-registered quality guard failed; the run stops."""


class Guards:
    """Running quality checks over the rows this process has produced."""

    def __init__(self, parse_rate_floor: float = PARSE_RATE_FLOOR,
                 latency_ceiling_s: float = LATENCY_CEILING_S,
                 min_rows: int = GUARD_MIN_ROWS):
        self.parse_rate_floor = parse_rate_floor
        self.latency_ceiling_s = latency_ceiling_s
        self.min_rows = min_rows
        self.n = 0
        self._ok = 0
        self._latencies: list[float] = []

    def record(self, parsed_ok: bool, latency_s: float) -> None:
        self.n += 1
        self._ok += bool(parsed_ok)
        self._latencies.append(latency_s)

    @property
    def parse_rate(self) -> float:
        return self._ok / self.n if self.n else float("nan")

    @property
    def median_latency_s(self) -> float:
        return statistics.median(self._latencies) if self._latencies else float("nan")

    def check(self) -> None:
        if self.n < self.min_rows:
            return
        if self.parse_rate < self.parse_rate_floor:
            raise GuardTripped(
                f"parse rate {self.parse_rate:.1%} is below the pre-registered "
                f"floor of {self.parse_rate_floor:.0%} over {self.n} rows. The "
                "prompt is degrading; every further row would be unusable. Fix the "
                "prompt, then start a fresh output file."
            )
        if self.median_latency_s > self.latency_ceiling_s:
            raise GuardTripped(
                f"median latency {self.median_latency_s:.1f}s exceeds "
                f"{self.latency_ceiling_s:.1f}s over {self.n} rows. This is the "
                "signature of a silent spill into shared memory -- lower "
                "MAX_PIXELS or free VRAM, then resume."
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_runner.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add studies/02-vlm-xai-study/src/xai_vlm/runner.py studies/02-vlm-xai-study/tests/test_runner.py
git commit -m "feat(study2): add pre-registered parse-rate and latency guards

Both abort the run rather than warning. The latency ceiling is twice the declared
budget, not twice the observed median -- the latter sits below the budget the
study already accepted and would abort healthy runs.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `run_stage` — the resumable loop, with crash-and-resume proof

**Files:**
- Modify: `src/xai_vlm/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `def run_stage(rows, work_fn, out_csv, state_json, fieldnames, fingerprint, *, resume=True, force_resume=False, checkpoint_every=100, progress_fn=None, guards=None) -> dict`
  - `rows`: iterable of manifest dicts (must carry `image_id`, `rule_id`).
  - `work_fn(row) -> dict` — returns the result fields for that row; the runner merges in the key fields and appends.
  - Returns a summary dict: `{"done": int, "skipped": int, "errors": int, "guards": Guards}`.

- [ ] **Step 1: Write the failing test**

```python
# append to studies/02-vlm-xai-study/tests/test_runner.py
from xai_vlm.runner import run_stage

ROW_FIELDS = ["image_id", "rule_id", "verdict", "latency_ms", "parse_error", "error"]


def _manifest(n):
    return [{"image_id": f"{i:04d}", "rule_id": "rule_1"} for i in range(n)]


def _paths(tmp_path):
    return tmp_path / "out.csv", tmp_path / "out.state.json"


def test_crash_then_resume_covers_the_manifest_exactly_once(tmp_path):
    out, state = _paths(tmp_path)
    rows = _manifest(10)
    fp = {"manifest_sha256": "abc"}

    calls = []

    def flaky(row):
        if len(calls) == 4:
            raise KeyboardInterrupt("simulated crash")
        calls.append(row["image_id"])
        return {"verdict": "YES", "latency_ms": 1.0}

    with pytest.raises(KeyboardInterrupt):
        run_stage(rows, flaky, out, state, ROW_FIELDS, fp)

    def healthy(row):
        calls.append(row["image_id"])
        return {"verdict": "YES", "latency_ms": 1.0}

    summary = run_stage(rows, healthy, out, state, ROW_FIELDS, fp)

    written = list(csv.DictReader(out.open(encoding="utf-8")))
    ids = [r["image_id"] for r in written]
    assert sorted(ids) == sorted(r["image_id"] for r in rows)
    assert len(ids) == len(set(ids)), "a resumed row was written twice"
    assert summary["skipped"] == 4


def test_resuming_a_complete_run_does_no_work(tmp_path):
    out, state = _paths(tmp_path)
    rows = _manifest(5)
    fp = {"manifest_sha256": "abc"}
    seen = []

    def work(row):
        seen.append(row["image_id"])
        return {"verdict": "NO", "latency_ms": 1.0}

    run_stage(rows, work, out, state, ROW_FIELDS, fp)
    n_first = len(seen)
    summary = run_stage(rows, work, out, state, ROW_FIELDS, fp)

    assert len(seen) == n_first, "work_fn was called on an already-complete run"
    assert summary["done"] == 0
    assert summary["skipped"] == 5


def test_fingerprint_mismatch_leaves_the_csv_untouched(tmp_path):
    out, state = _paths(tmp_path)
    rows = _manifest(3)

    def work(row):
        return {"verdict": "YES", "latency_ms": 1.0}

    run_stage(rows, work, out, state, ROW_FIELDS, {"max_pixels": 501760})
    before = out.read_text(encoding="utf-8")

    with pytest.raises(FingerprintMismatch):
        run_stage(rows, work, out, state, ROW_FIELDS, {"max_pixels": 1003520})

    assert out.read_text(encoding="utf-8") == before


def test_a_failing_row_is_recorded_and_the_run_continues(tmp_path):
    out, state = _paths(tmp_path)
    rows = _manifest(5)

    def work(row):
        if row["image_id"] == "0002":
            raise ValueError("boom")
        return {"verdict": "YES", "latency_ms": 1.0}

    summary = run_stage(rows, work, out, state, ROW_FIELDS, {"a": 1})

    written = {r["image_id"]: r for r in csv.DictReader(out.open(encoding="utf-8"))}
    assert len(written) == 5, "the failing row must still be recorded"
    assert "boom" in written["0002"]["error"]
    assert written["0002"]["verdict"] == ""
    assert summary["errors"] == 1


def test_guard_trips_mid_run_and_keeps_completed_rows(tmp_path):
    out, state = _paths(tmp_path)
    rows = _manifest(300)

    def work(row):
        return {"verdict": "", "parse_error": "no verdict", "latency_ms": 1.0}

    with pytest.raises(GuardTripped):
        run_stage(rows, work, out, state, ROW_FIELDS, {"a": 1}, checkpoint_every=100)

    written = list(csv.DictReader(out.open(encoding="utf-8")))
    assert 100 <= len(written) < 300, "should stop at a checkpoint, not at the end"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_runner.py -v -k "run_stage or crash or resume or guard_trips or failing_row"`
Expected: FAIL — `ImportError: cannot import name 'run_stage'`

- [ ] **Step 3: Write the implementation**

Append to `src/xai_vlm/runner.py`:

```python
from typing import Callable, Iterable


def run_stage(
    rows: Iterable[dict],
    work_fn: Callable[[dict], dict],
    out_csv: Path,
    state_json: Path,
    fieldnames: list[str],
    fingerprint: dict,
    *,
    resume: bool = True,
    force_resume: bool = False,
    checkpoint_every: int = 100,
    progress_fn: Optional[Callable[[dict], None]] = None,
    guards: Optional[Guards] = None,
) -> dict:
    """Run `work_fn` over `rows`, checkpointing every row and skipping done work.

    `work_fn` raising is not fatal: the row is recorded with its error and the run
    continues, because one unreadable image should not end an overnight run. A
    tripped guard *is* fatal, because it means every subsequent row would be
    unusable too.
    """
    state = RunState(state_json)
    state.verify_or_create(fingerprint, force=force_resume)

    writer = CheckpointWriter(out_csv, fieldnames)
    done = writer.done_keys() if resume else set()
    guards = guards if guards is not None else Guards()

    summary = {"done": 0, "skipped": 0, "errors": 0, "guards": guards}

    try:
        for row in rows:
            key = (row["image_id"], row["rule_id"])
            if key in done:
                summary["skipped"] += 1
                continue

            try:
                result = work_fn(row)
                error = ""
            except (KeyboardInterrupt, SystemExit):
                raise
            except GuardTripped:
                raise
            except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
                result = {}
                error = f"{type(exc).__name__}: {exc}"
                summary["errors"] += 1

            out_row = {**row, **result, "error": error}
            writer.append(out_row)
            summary["done"] += 1

            guards.record(
                parsed_ok=bool(out_row.get("verdict")) and not out_row.get("parse_error"),
                latency_s=float(out_row.get("latency_ms") or 0.0) / 1000.0,
            )

            if summary["done"] % checkpoint_every == 0:
                guards.check()
                if progress_fn is not None:
                    progress_fn(summary)
    finally:
        writer.close()

    return summary
```

- [ ] **Step 4: Run the whole runner suite**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_runner.py -v`
Expected: 20 passed

- [ ] **Step 5: Confirm nothing else regressed**

Run: `../../pilot/.venv/Scripts/python -m pytest tests -q`
Expected: 82 passed, 12 skipped (the GPU tests)

- [ ] **Step 6: Commit**

```bash
git add studies/02-vlm-xai-study/src/xai_vlm/runner.py studies/02-vlm-xai-study/tests/test_runner.py
git commit -m "feat(study2): add run_stage, the resumable stage loop

Skips done (image_id, rule_id) keys, records failing rows and continues, and
stops at a checkpoint when a guard trips. Crash-and-resume is proved by test:
the union of a killed run and its resume is exactly the manifest, no duplicates.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Stage 1 — build and freeze the sample manifest

**Files:**
- Modify: `scripts/01_build_sample.py` (replaces the stub entirely)
- Test: `tests/test_build_sample.py`

**Interfaces:**
- Consumes: `sampling.match_compliant_sample`, `sampling.metadata_key`, `sampling.stratification_report`, `sampling.summarize_strata`, `sampling.SampleRow`, `rules.rules_to_ask`, `rules.violated_rules`, `rules.expected_answer`.
- Produces:
  - `MANIFEST_FIELDS: list[str]` — the nine pre-registered columns.
  - `def build_rows(records: list[dict], seed: int) -> list[dict]`
  - `def write_manifest(rows: list[dict], path: Path, force: bool) -> None`

- [ ] **Step 1: Write the failing test**

```python
# studies/02-vlm-xai-study/tests/test_build_sample.py
"""Stage 1 tests. Metadata only -- no images, no model, no GPU."""
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "build_sample", STUDY_ROOT / "scripts" / "01_build_sample.py")
build_sample = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_sample)


def _records(n_violation=6, n_compliant=6):
    """Synthetic dataset rows with the columns Stage 1 reads."""
    recs = []
    for i in range(n_violation):
        recs.append({
            "image_id": f"v{i:04d}",
            "rule_1_violation": "no hard hat",
            "rule_2_violation": None, "rule_3_violation": None, "rule_4_violation": None,
            "quality_of_info": "rich info", "illumination": "daylight",
            "camera_distance": "medium", "view": "ground", "excavator": [],
        })
    for i in range(n_compliant):
        recs.append({
            "image_id": f"c{i:04d}",
            "rule_1_violation": None, "rule_2_violation": None,
            "rule_3_violation": None, "rule_4_violation": None,
            "quality_of_info": "rich info", "illumination": "daylight",
            "camera_distance": "medium", "view": "ground", "excavator": [],
        })
    return recs


def test_manifest_columns_are_exactly_the_pre_registered_schema():
    assert build_sample.MANIFEST_FIELDS == [
        "image_id", "rule_id", "stratum", "is_violation", "is_negative_control",
        "quality_of_info", "illumination", "camera_distance", "view",
    ]
    assert "expected" not in build_sample.MANIFEST_FIELDS


def test_same_seed_produces_an_identical_manifest():
    a = build_sample.build_rows(_records(), seed=42)
    b = build_sample.build_rows(_records(), seed=42)
    assert a == b


def test_different_seed_changes_the_manifest():
    a = build_sample.build_rows(_records(), seed=42)
    b = build_sample.build_rows(_records(), seed=7)
    assert a != b or len(a) == 0


def test_every_violation_image_contributes_its_violated_rule():
    rows = build_sample.build_rows(_records(), seed=42)
    v_rows = [r for r in rows if r["image_id"].startswith("v")]
    positive = [r for r in v_rows if r["is_violation"]]
    assert {r["image_id"] for r in positive} == {f"v{i:04d}" for i in range(6)}
    assert all(r["rule_id"] == "rule_1" for r in positive)
    assert all(r["stratum"] == "H" for r in v_rows)


def test_negative_controls_are_marked_and_are_not_violations():
    rows = build_sample.build_rows(_records(), seed=42)
    for r in rows:
        assert r["is_negative_control"] != r["is_violation"]


def test_compliant_rows_are_all_negative_controls():
    rows = build_sample.build_rows(_records(), seed=42)
    c_rows = [r for r in rows if r["stratum"] == "C"]
    assert c_rows
    assert all(r["is_negative_control"] for r in c_rows)
    assert not any(r["is_violation"] for r in c_rows)


def test_write_manifest_refuses_to_overwrite(tmp_path):
    path = tmp_path / "sample_manifest.csv"
    rows = build_sample.build_rows(_records(), seed=42)
    build_sample.write_manifest(rows, path, force=False)
    before = path.read_text(encoding="utf-8")

    with pytest.raises(SystemExit):
        build_sample.write_manifest(rows, path, force=False)
    assert path.read_text(encoding="utf-8") == before


def test_write_manifest_force_overwrites(tmp_path):
    path = tmp_path / "sample_manifest.csv"
    build_sample.write_manifest(build_sample.build_rows(_records(), seed=42), path, force=False)
    build_sample.write_manifest(build_sample.build_rows(_records(2, 2), seed=42), path, force=True)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) < 12
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_build_sample.py -v`
Expected: FAIL — `AttributeError: module 'build_sample' has no attribute 'MANIFEST_FIELDS'`

- [ ] **Step 3: Write the implementation**

Replace `scripts/01_build_sample.py` entirely:

```python
"""Stage 1 -- build the two-stratum sample manifest.

Runs on any machine: it reads dataset metadata only, no model involved.

Writes results/sample_manifest.csv, which defines the sample for every later stage.
FREEZE IT once written. Regenerating it mid-study silently invalidates everything
already computed, because stage 4's rows would refer to different images than
stage 2's -- and both files would still parse, so nothing downstream would notice.
That is why --force exists and why it is loud.

    python scripts/01_build_sample.py --seed 42
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm.rules import expected_answer, rules_to_ask, violated_rules  # noqa: E402
from xai_vlm.sampling import (  # noqa: E402
    match_compliant_sample,
    metadata_key,
    stratification_report,
)

# EXACTLY the columns pre-registered in the architecture document, Stage 1.
#
# No `expected` column: `is_violation` is per RULE, not per image, so ground truth
# is already here -- expected = "YES" if is_violation else "NO". Storing it twice
# would add a field that can disagree with its own source.
MANIFEST_FIELDS = [
    "image_id", "rule_id", "stratum", "is_violation", "is_negative_control",
    "quality_of_info", "illumination", "camera_distance", "view",
]

# The Stage 0 median, used only to project wall clock before any GPU time is spent.
STAGE0_MEDIAN_S = 2.9


def _row_for(rec: dict, rule_id: str, is_control: bool, stratum: str) -> dict:
    is_violation = not is_control
    # Two independent code paths must agree on ground truth. rules_to_ask decided
    # `is_control`; expected_answer reads the annotation field directly. If they
    # ever disagree the scan is buggy, and a silent disagreement here would put
    # wrong labels under every metric in the study.
    expected = expected_answer(rule_id, rec)
    if (expected == "YES") != is_violation:
        raise AssertionError(
            f"ground-truth disagreement on {rec['image_id']}/{rule_id}: "
            f"rules_to_ask says is_violation={is_violation}, "
            f"expected_answer says {expected}"
        )
    return {
        "image_id": rec["image_id"],
        "rule_id": rule_id,
        "stratum": stratum,
        "is_violation": is_violation,
        "is_negative_control": is_control,
        "quality_of_info": rec.get("quality_of_info", ""),
        "illumination": rec.get("illumination", ""),
        "camera_distance": rec.get("camera_distance", ""),
        "view": rec.get("view", ""),
    }


def build_rows(records: list[dict], seed: int) -> list[dict]:
    """Split into strata, assign rules, and flatten to one row per image-rule pair."""
    hazard = [r for r in records if violated_rules(r)]
    compliant = [r for r in records if not violated_rules(r)]

    target = Counter(metadata_key(r) for r in hazard)
    matched = match_compliant_sample(compliant, target, n_wanted=len(hazard), seed=seed)

    rows: list[dict] = []
    for stratum, recs in (("H", hazard), ("C", matched)):
        for rec in recs:
            for rule_id, is_control in rules_to_ask(rec, seed=seed):
                rows.append(_row_for(rec, rule_id, is_control, stratum))
    return rows


def write_manifest(rows: list[dict], path: Path, force: bool) -> None:
    """Write the manifest, refusing to clobber an existing one without --force."""
    path = Path(path)
    if path.exists() and not force:
        print(f"\nREFUSING to overwrite {path}.", file=sys.stderr)
        print(
            "This manifest defines the sample for every later stage. Regenerating "
            "it invalidates every result already computed against it -- silently, "
            "because the old result files would still parse.\n"
            "If that is genuinely what you want, pass --force.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(STUDY_ROOT / "results" / "sample_manifest.csv"))
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest (invalidates prior results)")
    ap.add_argument("--allow-imbalance", action="store_true",
                    help="continue despite a stratification WARNING")
    args = ap.parse_args()

    from xai_pilot.data import load_construction_site

    print("scanning the test split (metadata only) ...")
    records = []
    for rec in load_construction_site(split="test", streaming=True):
        records.append({k: v for k, v in rec.items() if k != "image"})
    print(f"  {len(records)} records")

    hazard = [r for r in records if violated_rules(r)]
    compliant = [r for r in records if not violated_rules(r)]
    target = Counter(metadata_key(r) for r in hazard)
    matched = match_compliant_sample(compliant, target, n_wanted=len(hazard), seed=args.seed)

    report = stratification_report(hazard, matched)
    print("\n" + report)
    if "WARNING" in report and not args.allow_imbalance:
        print(
            "\nABORTING: the compliant pool could not supply the metadata cells the "
            "hazard stratum needs. Comparing the strata would compare scene "
            "conditions rather than model behaviour. Pass --allow-imbalance to "
            "proceed with this stated as a limitation.",
            file=sys.stderr,
        )
        return 1

    rows = build_rows(records, seed=args.seed)
    write_manifest(rows, Path(args.out), force=args.force)

    from xai_vlm.runner import manifest_sha256

    print(f"\nwrote {len(rows)} image-rule pairs to {args.out}")
    print(f"  strata      : H={sum(r['stratum'] == 'H' for r in rows)}  "
          f"C={sum(r['stratum'] == 'C' for r in rows)}")
    print(f"  positives   : {sum(r['is_violation'] for r in rows)}")
    print(f"  controls    : {sum(r['is_negative_control'] for r in rows)}")
    print(f"  sha256      : {manifest_sha256(Path(args.out))}")
    print(f"\nStage 2 projection at the Stage 0 median of {STAGE0_MEDIAN_S}s/call: "
          f"{len(rows) * STAGE0_MEDIAN_S / 3600:.1f} h")
    print("\nFREEZE THIS FILE. Commit it before running Stage 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_build_sample.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add studies/02-vlm-xai-study/scripts/01_build_sample.py studies/02-vlm-xai-study/tests/test_build_sample.py
git commit -m "feat(study2): implement Stage 1, the frozen sample manifest

Columns are exactly the schema pre-registered in the architecture document; no
expected column, because is_violation is per-rule and already carries it.
rules_to_ask and expected_answer are cross-checked per row, so a disagreement
between the two fails the scan instead of putting wrong labels under every metric.

The script refuses to overwrite an existing manifest without --force.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Stage 2 — the inference driver

**Files:**
- Modify: `scripts/02_baseline_inference.py` (replaces the stub entirely)
- Test: `tests/test_baseline_inference.py`

**Interfaces:**
- Consumes: `runner.run_stage`, `runner.manifest_sha256`, `runner.Guards`, `config.MODEL_TIERS`, `backends.base.Answer`.
- Produces:
  - `RESULT_FIELDS: list[str]`
  - `def build_fingerprint(manifest_path: Path, model: str, seed: int) -> dict`
  - `def make_work_fn(backend, image_source) -> Callable[[dict], dict]`
  - `image_source(image_id) -> PIL.Image.Image | None`

- [ ] **Step 1: Write the failing test**

```python
# studies/02-vlm-xai-study/tests/test_baseline_inference.py
"""Stage 2 driver tests, over FakeBackend and a fake image source. No GPU."""
import csv
import importlib.util
import sys
from pathlib import Path

from PIL import Image

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm.backends.fake import FakeBackend  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "baseline_inference", STUDY_ROOT / "scripts" / "02_baseline_inference.py")
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)


def _manifest_file(tmp_path, n=6):
    path = tmp_path / "sample_manifest.csv"
    fields = ["image_id", "rule_id", "stratum", "is_violation",
              "is_negative_control", "quality_of_info", "illumination",
              "camera_distance", "view"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i in range(n):
            w.writerow({
                "image_id": f"{i:04d}", "rule_id": "rule_1", "stratum": "H",
                "is_violation": i % 2 == 0, "is_negative_control": i % 2 == 1,
                "quality_of_info": "rich info", "illumination": "daylight",
                "camera_distance": "medium", "view": "ground",
            })
    return path


def _images(_image_id):
    return Image.new("RGB", (64, 48), "grey")


def test_expected_is_derived_from_is_violation(tmp_path):
    manifest = _manifest_file(tmp_path, n=4)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    work = baseline.make_work_fn(FakeBackend(), _images)
    out = work(rows[0])
    assert out["expected"] == "YES"
    out = work(rows[1])
    assert out["expected"] == "NO"


def test_missing_image_is_recorded_not_crashed(tmp_path):
    manifest = _manifest_file(tmp_path, n=2)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    work = baseline.make_work_fn(FakeBackend(), lambda _id: None)
    out = work(rows[0])
    assert out["parse_error"] == "image not found"
    assert out["verdict"] == ""


def test_end_to_end_writes_every_manifest_row(tmp_path):
    manifest = _manifest_file(tmp_path, n=6)
    out_csv = tmp_path / "baseline_fake.csv"
    summary = baseline.run(
        manifest_path=manifest, backend=FakeBackend(), image_source=_images,
        out_csv=out_csv, state_json=tmp_path / "baseline_fake.state.json",
        model="fake", seed=42, limit=None, resume=True, force_resume=False,
        checkpoint_every=100,
    )
    written = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert len(written) == 6
    assert summary["errors"] == 0
    assert all(r["verdict"] in {"YES", "NO"} for r in written)


def test_limit_caps_rows_and_resume_continues_past_it(tmp_path):
    manifest = _manifest_file(tmp_path, n=6)
    out_csv = tmp_path / "baseline_fake.csv"
    state = tmp_path / "baseline_fake.state.json"
    kw = dict(manifest_path=manifest, backend=FakeBackend(), image_source=_images,
              out_csv=out_csv, state_json=state, model="fake", seed=42,
              resume=True, force_resume=False, checkpoint_every=100)

    baseline.run(limit=2, **kw)
    assert len(list(csv.DictReader(out_csv.open(encoding="utf-8")))) == 2

    baseline.run(limit=None, **kw)
    written = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert len(written) == 6
    assert len({r["image_id"] for r in written}) == 6
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_baseline_inference.py -v`
Expected: FAIL — `AttributeError: module 'baseline_inference' has no attribute 'make_work_fn'`

- [ ] **Step 3: Write the implementation**

Replace `scripts/02_baseline_inference.py` entirely:

```python
"""Stage 2 -- baseline inference. The main measurement; everything reuses it.

Requires CUDA. Checkpoints every row and resumes, because at many hours a crash at
hour eight without resumability costs a day -- and because a run stitched together
from two different configurations would look complete and be wrong.

    python scripts/02_baseline_inference.py --model qwen7b-4bit --resume
"""
import argparse
import csv
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402
from xai_vlm.parsing import ParseStats, parse_response  # noqa: E402
from xai_vlm.runner import Guards, manifest_sha256, run_stage  # noqa: E402

RESULT_FIELDS = [
    "image_id", "rule_id", "stratum", "is_negative_control", "expected",
    "verdict", "region", "rationale", "confidence", "raw_text",
    "latency_ms", "parse_strict", "parse_error", "error",
]


def _revision_for(model: str) -> str:
    """The pinned weight revision, so the fingerprint records what actually ran."""
    lock = STUDY_ROOT.parents[1] / "requirements" / "models.lock"
    if not lock.exists():
        return "unpinned"
    for line in lock.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line.startswith(model + " ") or line.split()[:1] == [model]:
            parts = line.split()
            if len(parts) >= 3:
                return parts[2]
    return "unpinned"


def build_fingerprint(manifest_path: Path, model: str, seed: int) -> dict:
    """Everything that, if changed, makes previously written rows incomparable."""
    return {
        "manifest_sha256": manifest_sha256(manifest_path),
        "model": model,
        "revision": _revision_for(model),
        "compute_dtype": config.COMPUTE_DTYPE,
        "attention_impl": config.ATTENTION_IMPL,
        "max_pixels": config.MAX_PIXELS,
        "generation": config.GENERATION,
        "seed": seed,
    }


def make_work_fn(backend, image_source):
    """One manifest row -> one result row."""

    def work(row: dict) -> dict:
        # Ground truth is derived, never stored twice. is_violation is per-rule.
        is_violation = str(row.get("is_violation", "")).lower() == "true"
        expected = "YES" if is_violation else "NO"

        image = image_source(row["image_id"])
        if image is None:
            # A missing image is data about the corpus, not a crash. Recorded so
            # the gap is visible in the results rather than as an absent row.
            return {"expected": expected, "verdict": "", "parse_error": "image not found",
                    "latency_ms": 0.0}

        answer = backend.answer(image, row["rule_id"])
        return {
            "expected": expected,
            "verdict": answer.verdict or "",
            "region": "" if answer.region is None else str(tuple(answer.region)),
            "rationale": answer.rationale,
            "confidence": answer.confidence,
            "raw_text": answer.raw_text,
            "latency_ms": answer.latency_ms,
            "parse_strict": answer.parse_strict,
            "parse_error": answer.parse_error,
        }

    return work


def _progress(summary: dict) -> None:
    g = summary["guards"]
    print(f"  {summary['done']:>6} done  {summary['skipped']:>6} skipped  "
          f"{summary['errors']:>4} errors  |  parse {g.parse_rate:.1%}  "
          f"median {g.median_latency_s:.1f}s")


def run(*, manifest_path, backend, image_source, out_csv, state_json, model, seed,
        limit, resume, force_resume, checkpoint_every) -> dict:
    """Drive run_stage over the manifest. Separated from main() so it is testable."""
    with Path(manifest_path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if limit is not None:
        rows = rows[:limit]

    return run_stage(
        rows,
        make_work_fn(backend, image_source),
        Path(out_csv), Path(state_json), RESULT_FIELDS,
        build_fingerprint(Path(manifest_path), model, seed),
        resume=resume, force_resume=force_resume,
        checkpoint_every=checkpoint_every,
        progress_fn=_progress, guards=Guards(),
    )


def _summarise(out_csv: Path) -> None:
    """Negative controls FIRST, before any accuracy figure.

    If the model answers YES to hazards known to be absent, its accuracy on
    violations is meaningless -- it is guessing, not detecting. Printing the
    controls first makes that unavoidable rather than optional.
    """
    rows = list(csv.DictReader(Path(out_csv).open(encoding="utf-8")))
    stats = ParseStats()
    for r in rows:
        stats.record(parse_response(r.get("raw_text", "")))

    controls = [r for r in rows if str(r["is_negative_control"]).lower() == "true"]
    false_alarms = sum(1 for r in controls if r["verdict"] == "YES")
    positives = [r for r in rows if r["expected"] == "YES"]
    negatives = [r for r in rows if r["expected"] == "NO"]

    print("\n" + "=" * 66)
    print("NEGATIVE CONTROLS FIRST")
    if controls:
        print(f"  false-alarm rate : {false_alarms / len(controls):.0%} "
              f"({false_alarms}/{len(controls)})")
        if false_alarms / len(controls) > 0.5:
            print("  WARNING: the model answers YES to most absent hazards. Its "
                  "accuracy on violations cannot be read as detection.")
    print(f"\nYES share          : {sum(1 for r in rows if r['verdict'] == 'YES') / len(rows):.0%}")
    if positives:
        print(f"sensitivity        : "
              f"{sum(1 for r in positives if r['verdict'] == 'YES') / len(positives):.0%} "
              f"(n={len(positives)})")
    if negatives:
        print(f"specificity        : "
              f"{sum(1 for r in negatives if r['verdict'] == 'NO') / len(negatives):.0%} "
              f"(n={len(negatives)})")
    print(f"errors             : {sum(1 for r in rows if r.get('error'))}")
    print("\n" + stats.summary())
    print("=" * 66)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(config.MODEL_TIERS))
    ap.add_argument("--manifest", default=str(STUDY_ROOT / "results" / "sample_manifest.csv"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-resume", action="store_true",
                    help="resume despite a configuration change (recorded in the state file)")
    ap.add_argument("--limit", type=int, default=None, help="cap rows, for smoke tests")
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"no manifest at {manifest}. Run scripts/01_build_sample.py first.",
              file=sys.stderr)
        return 1

    out_csv = STUDY_ROOT / "results" / f"baseline_{args.model}.csv"
    state_json = STUDY_ROOT / "results" / f"baseline_{args.model}.state.json"

    from xai_pilot.data import load_construction_site
    from xai_vlm.backends.qwen import QwenBackend

    wanted = {r["image_id"] for r in csv.DictReader(manifest.open(encoding="utf-8"))}
    print(f"manifest: {len(wanted)} distinct images")

    # One image at a time. Stage 0 held every gate image in a dict, which was fine
    # for 39 images and is not for thousands.
    def image_source(image_id):
        for rec in load_construction_site(split="test", streaming=True):
            if rec["image_id"] == image_id:
                return rec["image"].convert("RGB")
        return None

    print(f"loading {args.model} ...")
    backend = QwenBackend(tier=args.model)

    summary = run(
        manifest_path=manifest, backend=backend, image_source=image_source,
        out_csv=out_csv, state_json=state_json, model=args.model, seed=args.seed,
        limit=args.limit, resume=args.resume, force_resume=args.force_resume,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"\ndone={summary['done']}  skipped={summary['skipped']}  "
          f"errors={summary['errors']}")
    _summarise(out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_baseline_inference.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the whole CPU suite**

Run: `../../pilot/.venv/Scripts/python -m pytest tests -q`
Expected: 94 passed, 12 skipped

- [ ] **Step 6: Commit**

```bash
git add studies/02-vlm-xai-study/scripts/02_baseline_inference.py studies/02-vlm-xai-study/tests/test_baseline_inference.py
git commit -m "feat(study2): implement Stage 2 baseline inference over runner.py

Resumable by (image_id, rule_id), fingerprinted against the manifest hash and
every correctness setting. The end-of-run summary prints negative controls before
any accuracy figure, since accuracy on violations is unreadable if the model
answers YES to hazards known to be absent.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Fix the O(n²) image lookup

**Files:**
- Modify: `scripts/02_baseline_inference.py`
- Test: `tests/test_baseline_inference.py`

**Rationale:** Task 6's `image_source` re-streams the whole 10,013-record split for *every* image, which is ~2,556 full passes. This task replaces it with a single streaming pass that yields images in dataset order, driving the manifest rows from that order instead. Task 6 is committed working first so this optimisation is a reviewable change on its own.

**Interfaces:**
- Consumes: Task 6.
- Produces: `def stream_rows_in_dataset_order(manifest_rows: list[dict], loader) -> Iterator[tuple[dict, Image.Image]]`

- [ ] **Step 1: Write the failing test**

```python
# append to studies/02-vlm-xai-study/tests/test_baseline_inference.py
def test_dataset_is_streamed_exactly_once(tmp_path):
    manifest = _manifest_file(tmp_path, n=6)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    passes = []

    def loader(split="test", streaming=True):
        passes.append(1)
        for i in range(6):
            yield {"image_id": f"{i:04d}", "image": Image.new("RGB", (8, 8))}

    out = list(baseline.stream_rows_in_dataset_order(rows, loader))
    assert len(passes) == 1, "the split must be streamed once, not once per image"
    assert {r["image_id"] for r, _ in out} == {f"{i:04d}" for i in range(6)}


def test_rows_for_one_image_are_grouped(tmp_path):
    rows = [
        {"image_id": "0001", "rule_id": "rule_1"},
        {"image_id": "0002", "rule_id": "rule_1"},
        {"image_id": "0001", "rule_id": "rule_3"},
    ]

    def loader(split="test", streaming=True):
        for i in (1, 2):
            yield {"image_id": f"{i:04d}", "image": Image.new("RGB", (8, 8))}

    out = list(baseline.stream_rows_in_dataset_order(rows, loader))
    assert len(out) == 3
    assert [r["image_id"] for r, _ in out][:2] == ["0001", "0001"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_baseline_inference.py -v -k stream`
Expected: FAIL — `AttributeError: module has no attribute 'stream_rows_in_dataset_order'`

- [ ] **Step 3: Write the implementation**

Add to `scripts/02_baseline_inference.py`:

```python
from collections import defaultdict


def stream_rows_in_dataset_order(manifest_rows, loader):
    """Yield (manifest_row, image) in dataset streaming order, one pass.

    The naive alternative -- look each image up by id -- re-streams the whole split
    per image, which is thousands of full passes. Iterating the split once and
    emitting whatever manifest rows match is why resume is keyed on
    (image_id, rule_id) and never on position: this order is not manifest order.
    """
    by_image = defaultdict(list)
    for row in manifest_rows:
        by_image[row["image_id"]].append(row)

    remaining = len(by_image)
    for rec in loader(split="test", streaming=True):
        rows = by_image.pop(rec["image_id"], None)
        if not rows:
            continue
        image = rec["image"]
        image = image.convert("RGB") if hasattr(image, "convert") else image
        for row in rows:
            yield row, image
        remaining -= 1
        if remaining == 0:
            return
```

Then rewrite `run()` to consume it, replacing the `image_source` indirection:

```python
def run(*, manifest_path, backend, image_source, out_csv, state_json, model, seed,
        limit, resume, force_resume, checkpoint_every) -> dict:
    """Drive run_stage over the manifest.

    `image_source` stays a per-id callable so tests can supply images without a
    dataset; main() passes a loader-backed one built on
    stream_rows_in_dataset_order.
    """
    with Path(manifest_path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if limit is not None:
        rows = rows[:limit]

    return run_stage(
        rows,
        make_work_fn(backend, image_source),
        Path(out_csv), Path(state_json), RESULT_FIELDS,
        build_fingerprint(Path(manifest_path), model, seed),
        resume=resume, force_resume=force_resume,
        checkpoint_every=checkpoint_every,
        progress_fn=_progress, guards=Guards(),
    )
```

And in `main()`, replace the O(n²) `image_source` with a cache filled by one pass:

```python
    from xai_pilot.data import load_construction_site

    manifest_rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    if args.limit is not None:
        manifest_rows = manifest_rows[:args.limit]

    # One pass over the split, one image resident at a time.
    cache: dict = {}

    def image_source(image_id):
        return cache.get(image_id)

    def rows_with_images():
        for row, image in stream_rows_in_dataset_order(manifest_rows, load_construction_site):
            cache.clear()
            cache[row["image_id"]] = image
            yield row
```

Wire `rows_with_images()` into `run_stage` in place of the row list, keeping every
other argument identical.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_baseline_inference.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add studies/02-vlm-xai-study/scripts/02_baseline_inference.py studies/02-vlm-xai-study/tests/test_baseline_inference.py
git commit -m "perf(study2): stream the split once instead of once per image

Looking images up by id re-streamed all 10,013 records per image. One pass now
emits manifest rows as their images arrive -- which is exactly why resume keys on
(image_id, rule_id) rather than row position: this is not manifest order.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: GPU validation on the CUDA machine

**Files:** none modified. This task produces evidence, not code.

**Precondition:** run on the machine with the RTX 3070 and downloaded weights.

- [ ] **Step 1: Run the GPU test suite**

```bash
cd studies/02-vlm-xai-study
../../pilot/.venv/Scripts/python -m pytest tests -q --gpu
```

Expected: all pass, 0 skipped. Record the count.

- [ ] **Step 2: Build the manifest**

```bash
../../pilot/.venv/Scripts/python scripts/01_build_sample.py --seed 42
```

Expected: no `WARNING` in the stratification report; prints row count, sha256, and the projected Stage 2 hours. **Record all three.** If it aborts on imbalance, stop and report rather than passing `--allow-imbalance`.

- [ ] **Step 3: Commit the frozen manifest**

```bash
git add studies/02-vlm-xai-study/results/sample_manifest.csv
git commit -m "data(study2): freeze the Stage 1 sample manifest (seed 42)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Smoke-run Stage 2**

```bash
../../pilot/.venv/Scripts/python scripts/02_baseline_inference.py --model qwen7b-4bit --limit 20 --resume
```

Expected: 20 rows in `results/baseline_qwen7b-4bit.csv`, a `.state.json` beside it, no guard trips (20 < `GUARD_MIN_ROWS`). Record median latency and peak VRAM.

- [ ] **Step 5: Prove resume on real hardware**

Kill the next run with Ctrl-C partway, then rerun the identical command:

```bash
../../pilot/.venv/Scripts/python scripts/02_baseline_inference.py --model qwen7b-4bit --limit 60 --resume
# Ctrl-C after a few rows, then:
../../pilot/.venv/Scripts/python scripts/02_baseline_inference.py --model qwen7b-4bit --limit 60 --resume
```

Expected: the second run reports a non-zero `skipped`, the CSV has 60 unique `(image_id, rule_id)` pairs and no duplicates. Verify:

```bash
../../pilot/.venv/Scripts/python -c "import csv,sys; rows=list(csv.DictReader(open('results/baseline_qwen7b-4bit.csv',encoding='utf-8'))); keys=[(r['image_id'],r['rule_id']) for r in rows]; print(len(keys), len(set(keys)))"
```

Expected: the two numbers are equal.

- [ ] **Step 6: Prove the fingerprint guard on real hardware**

Temporarily run with a different tier against the same output path — or edit `MAX_PIXELS`, run, and revert. Expected: `FingerprintMismatch` naming the changed field, and the CSV unchanged.

- [ ] **Step 7: Report before launching the full run**

Report to the user: GPU test count, manifest rows and sha256, projected hours, smoke median latency, peak VRAM, and the resume/fingerprint proofs. **Do not launch the full battery without their go-ahead.**

---

### Task 9: Documentation

**Files:**
- Modify: `studies/02-vlm-xai-study/README.md`

- [ ] **Step 1: Update the status table**

Change `scripts/01,02` from `⬜ skeletons only (Plan 2)` to `✅ implemented and tested`, add a `runner.py — resumable stage execution` row, and update the test count to the number Task 8 recorded.

- [ ] **Step 2: Add a "Running Stage 2" section**

Document the three commands, that `--resume` is safe on a first run, what a `FingerprintMismatch` means and how to respond, and the two guard thresholds with the reason the latency ceiling is twice the budget rather than twice the observed median.

- [ ] **Step 3: Record the Stage 1 numbers**

Add the manifest row count, sha256 and seed, so a reader can verify their regenerated manifest matches the one the results were computed against.

- [ ] **Step 4: Commit**

```bash
git add studies/02-vlm-xai-study/README.md
git commit -m "docs(study2): record Plan 2 status, run commands and Stage 1 manifest hash

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Plan Self-Review

**Spec coverage.** Every spec section maps to a task: `runner.py` → Tasks 1–4; Stage 1 procedure, output, freezing → Task 5; fingerprint → Task 2 + `build_fingerprint` in Task 6; resume identity → Tasks 1, 4, 7; memory → Task 7; per-call error handling → Task 4; guard rails → Task 3; progress → Task 4 `_progress`; end-of-run negative controls → Task 6 `_summarise`; output columns → Task 6 `RESULT_FIELDS`; CLI surface → Tasks 5–6 `main()`; testing → Tasks 1–7 plus Task 8 for GPU.

**Known deviation from the spec, deliberate.** The spec says the runner "holds one image at a time"; Task 6 ships a correct-but-quadratic lookup and Task 7 replaces it. Splitting them keeps each task independently reviewable, and Task 6's version is correct — only slow — so a reviewer can reject Task 7 without breaking anything.

**Interface consistency.** `RowKey`, `CheckpointWriter`, `RunState`, `Guards`, `run_stage`, `manifest_sha256`, `MANIFEST_FIELDS`, `RESULT_FIELDS`, `build_fingerprint`, `make_work_fn`, `stream_rows_in_dataset_order` are each defined once and referenced with the same names and signatures throughout.

**Test counts** in "run the whole suite" steps are projections from the current 74. Report the actual number rather than adjusting the code to hit the projection.
