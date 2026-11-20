# Study 2 Plan 3b — Descriptive Accuracy and Completeness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the region the model named was load-bearing — Stage 4 — and combine it with Stage 3 into Bounded Completeness — Stage 9.

**Architecture:** Stage 4 reuses `runner.run_stage` unchanged. One `work_fn` call performs both maskings (black fill and token ablation) and emits one row carrying both flips, so the runner's `(image_id, rule_id)` key contract stays intact. Stage 9 is a pure join over two CSVs.

**Tech Stack:** Python 3.10, pytest, transformers 4.49.0, CUDA.

**Spec:** `docs/superpowers/specs/2026-08-20-study2-plan3b-descriptive-accuracy-design.md`

## Global Constraints

- **Work from `studies/02-vlm-xai-study/`.** Python is `../../pilot/.venv/Scripts/python`.
- **Do not modify** `runner.py`, `masking.py`, `backends/`, `rules.py`, `parsing.py`. All exist and are tested.
- **`answer_masked` already snaps** the region via `snap_to_patch_grid(region, image.size)`. Stage 4 records the snapped box for auditability but must not pass a pre-snapped box and double-snap.
- **Correctness settings are fixed** from `config.py`: bfloat16, sdpa, `MAX_PIXELS` 640 blocks.
- **Both masking methods, every row:** `["black", "tokens"]`. `tokens` is the primary figure.
- **Rows masked:** every row with a region, in both verdict classes. The 410 NO+region rows are the artefact control, not padding.
- **Fingerprint must include** `baseline_sha256` and `regions_sha256` — re-running Stage 2 or 3 must invalidate Stage 4.
- **Coverage verified against the input set**, not the exit code.
- **TDD.** Commit messages end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `src/xai_vlm/metrics/descriptive_accuracy.py` | Flip logic and rate aggregation. |
| `src/xai_vlm/metrics/completeness.py` | Stage 9's three-way classification. |
| `scripts/04_descriptive_accuracy.py` | Stage 4 runner over `runner.run_stage`. |
| `scripts/09_completeness.py` | Stage 9 join. |

---

### Task 1: Flip logic

**Files:**
- Create: `src/xai_vlm/metrics/descriptive_accuracy.py`
- Test: `tests/test_descriptive_accuracy.py`

**Interfaces:**
- Produces:
  - `MASKING_METHODS = ("black", "tokens")`
  - `def is_flip(baseline_verdict: str, masked_verdict: str) -> bool`
  - `def flip_rate(rows: list[dict], method: str) -> float`
  - `def artefact_floor(rows: list[dict], method: str) -> float`

- [ ] **Step 1: Write the failing test**

```python
# studies/02-vlm-xai-study/tests/test_descriptive_accuracy.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_descriptive_accuracy.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

```python
# studies/02-vlm-xai-study/src/xai_vlm/metrics/descriptive_accuracy.py
"""Stage 4 -- Metric 1: Descriptive Accuracy.

The primary faithfulness measurement. If deleting the region the model named does
not change its answer, that region was decoration and the explanation is not an
explanation.

This is the test a model cannot pass by disposition. Stage 2 measured 91%
sensitivity against 41% specificity, so many verdicts are right by inclination
rather than evidence; a flip rate is indifferent to inclination.
"""

from __future__ import annotations

# black  -- patch-aligned fill, comparable with prior work, but creates a
#           rectangle no photograph contains.
# tokens -- delete the image tokens before the language model runs. No pixels are
#           modified, so no artefact can exist. PRIMARY.
MASKING_METHODS = ("black", "tokens")

NAN = float("nan")


def is_flip(baseline_verdict: str, masked_verdict: str) -> bool:
    """Did removing the region change the answer?

    An empty masked verdict is a PARSE FAILURE, not a flip. Counting it as one
    would inflate the headline faithfulness figure with unreadable replies --
    precisely the silent data loss the parser was built to make visible.
    """
    if not masked_verdict:
        return False
    return baseline_verdict != masked_verdict


def _rate(rows: list[dict], method: str, baseline: str) -> float:
    subset = [r for r in rows if r.get("baseline_verdict") == baseline]
    if not subset:
        return NAN
    key = f"flip_{method}"
    flips = sum(1 for r in subset if str(r.get(key, "")).lower() == "true")
    return flips / len(subset)


def flip_rate(rows: list[dict], method: str) -> float:
    """Share of YES answers that changed when their region was removed."""
    return _rate(rows, method, "YES")


def artefact_floor(rows: list[dict], method: str) -> float:
    """Share of NO answers that became YES when a region was removed.

    Masking only removes information, so this cannot rationally happen. Whatever
    rate it shows is how much the intervention perturbs the model by itself, and a
    YES-flip rate that fails to clear it is not evidence of faithfulness.
    """
    return _rate(rows, method, "NO")
```

- [ ] **Step 4: Run to verify it passes**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_descriptive_accuracy.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add studies/02-vlm-xai-study/src/xai_vlm/metrics/descriptive_accuracy.py studies/02-vlm-xai-study/tests/test_descriptive_accuracy.py
git commit -m "feat(study2): flip logic for Stage 4

An unparseable masked reply is not a flip -- counting it as one would inflate the
headline faithfulness figure with unreadable replies.

artefact_floor is the NO->YES rate. Masking only removes information, so a NO
cannot rationally become a YES; whatever rate appears is how much the
intervention perturbs the model by itself.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Stage 4 runner

**Files:**
- Create: `scripts/04_descriptive_accuracy.py`
- Test: `tests/test_stage4_runner.py`

**Interfaces:**
- Consumes: `runner.run_stage`, `runner.manifest_sha256`, `descriptive_accuracy.MASKING_METHODS`, `masking.snap_to_patch_grid`.
- Produces:
  - `RESULT_FIELDS: list[str]`
  - `def build_fingerprint(manifest, baseline, regions, model, seed) -> dict`
  - `def make_work_fn(backend, image_source) -> Callable[[dict], dict]`
  - `def maskable_rows(regions_rows: list[dict]) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# studies/02-vlm-xai-study/tests/test_stage4_runner.py
"""Stage 4 driver, over FakeBackend. No GPU."""
import csv
import importlib.util
import sys
from pathlib import Path

from PIL import Image

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm.backends.fake import FakeBackend  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "stage4", STUDY_ROOT / "scripts" / "04_descriptive_accuracy.py")
stage4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage4)

REGION = "(100.0, 100.0, 300.0, 400.0)"


def _regions_rows():
    return [
        {"image_id": "0001", "rule_id": "rule_1", "verdict": "YES",
         "source": "model", "region": REGION, "image_w": "1200", "image_h": "900"},
        {"image_id": "0002", "rule_id": "rule_1", "verdict": "NO",
         "source": "model", "region": REGION, "image_w": "1200", "image_h": "900"},
        {"image_id": "0003", "rule_id": "rule_1", "verdict": "YES",
         "source": "none", "region": "", "image_w": "1200", "image_h": "900"},
    ]


def _images(_id):
    return Image.new("RGB", (1200, 900), "grey")


def test_maskable_rows_excludes_rows_with_no_region():
    rows = stage4.maskable_rows(_regions_rows())
    assert {r["image_id"] for r in rows} == {"0001", "0002"}


def test_maskable_rows_keeps_no_verdict_rows_as_the_control():
    """The NO rows are the artefact floor, not padding."""
    rows = stage4.maskable_rows(_regions_rows())
    assert any(r["verdict"] == "NO" for r in rows)


def test_both_methods_are_called_for_each_row():
    backend = FakeBackend()
    work = stage4.make_work_fn(backend, _images)
    work(stage4.maskable_rows(_regions_rows())[0])
    methods = [c[2] for c in backend.calls if c[0] == "answer_masked"]
    assert methods == ["black", "tokens"]


def test_flip_is_recorded_per_method():
    """Baseline YES, masked replies NO -> both flips True."""
    backend = FakeBackend(script={"rule_1": ["REASON: gone.\nREGION: NONE\nANSWER: NO"]})
    work = stage4.make_work_fn(backend, _images)
    out = work(stage4.maskable_rows(_regions_rows())[0])
    assert out["verdict_black"] == "NO"
    assert out["flip_black"] is True
    assert out["flip_tokens"] is True
    assert out["methods_disagree"] is False


def test_the_snapped_region_is_recorded():
    backend = FakeBackend()
    work = stage4.make_work_fn(backend, _images)
    out = work(stage4.maskable_rows(_regions_rows())[0])
    # snap_to_patch_grid expands outward to 28px blocks: 100 -> 84, 300 -> 308.
    assert out["region_snapped"] == "(84.0, 84.0, 308.0, 420.0)"


def test_a_missing_image_is_recorded_not_crashed():
    work = stage4.make_work_fn(FakeBackend(), lambda _id: None)
    out = work(stage4.maskable_rows(_regions_rows())[0])
    assert out["error"] == "image not found"


def test_fingerprint_includes_both_input_hashes(tmp_path):
    m = tmp_path / "m.csv"; m.write_text("a", encoding="utf-8")
    b = tmp_path / "b.csv"; b.write_text("b", encoding="utf-8")
    r = tmp_path / "r.csv"; r.write_text("c", encoding="utf-8")
    fp = stage4.build_fingerprint(m, b, r, "fake", 42)
    assert "baseline_sha256" in fp and "regions_sha256" in fp
    assert fp["masking_methods"] == "black,tokens"
```

- [ ] **Step 2: Run to verify it fails**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_stage4_runner.py -v`
Expected: FAIL — the module does not exist.

- [ ] **Step 3: Write the stage script**

The implementation is given in full in Task 2 Step 3 of the executing session; it must define `RESULT_FIELDS`, `maskable_rows`, `build_fingerprint`, `make_work_fn`, `run`, `verify_coverage`, and a `main` that streams images once via the same one-pass generator pattern as Stage 2 and drives `runner.run_stage`.

- [ ] **Step 4: Run to verify it passes**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_stage4_runner.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

---

### Task 3: GPU smoke and full run

- [ ] **Step 1: Smoke run**

Run: `../../pilot/.venv/Scripts/python -u scripts/04_descriptive_accuracy.py --model qwen7b-4bit --limit 20 --resume`
Expected: 20 rows, both methods executed, peak VRAM recorded, no guard trip.

- [ ] **Step 2: Check peak VRAM**

Token ablation replicates the `masked_scatter` merge and may allocate more than the 6.97 GB Stage 2 peak. If it exceeds 7.6 GB, stop and report before committing 17 h.

- [ ] **Step 3: Launch the full run**

Run: `../../pilot/.venv/Scripts/python -u scripts/04_descriptive_accuracy.py --model qwen7b-4bit --resume`

- [ ] **Step 4: Verify coverage on completion**

Compare against `maskable_rows` count (expected 3,031). Missing rows mean an incomplete run regardless of exit code.

---

### Task 4: Stage 9 — bounded completeness

**Files:**
- Create: `src/xai_vlm/metrics/completeness.py`, `scripts/09_completeness.py`
- Test: `tests/test_completeness.py`

**Interfaces:**
- Produces:
  - `def classify_completeness(source: str, flip_tokens: str) -> str`
  - `def completeness_rates(rows: list[dict]) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# studies/02-vlm-xai-study/tests/test_completeness.py
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
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Write the implementation**

```python
# studies/02-vlm-xai-study/src/xai_vlm/metrics/completeness.py
"""Stage 9 -- Metric 6: Bounded Completeness.

No new model calls. Combines Stage 3 (was an explanation offered?) with Stage 4
(did it matter?).

`explanation_weak` deserves as much prominence as `explanation_supported`: the
model named a region, that region turned out not to matter, and a reader shown
only the region would have been misled about why the system decided what it did.
"""

from __future__ import annotations

CATEGORIES = ("no_usable_explanation", "explanation_supported",
              "explanation_weak", "not_measured")


def classify_completeness(source: str, flip_tokens: str) -> str:
    """Three-way verdict, computed from the TOKEN-ABLATION flip.

    Token ablation is used rather than black fill so the headline is not inflated
    by the black-rectangle artefact.

    A region with no Stage 4 result is `not_measured`, never a default. Counting
    an unmeasured row as weak would understate faithfulness; counting it as
    supported would overstate it.
    """
    if source != "model":
        return "no_usable_explanation"
    f = str(flip_tokens).strip().lower()
    if f == "true":
        return "explanation_supported"
    if f == "false":
        return "explanation_weak"
    return "not_measured"


def completeness_rates(rows: list[dict]) -> dict:
    """Share of rows in each category."""
    n = len(rows)
    out = {"n": n}
    for cat in CATEGORIES:
        out[cat] = (sum(1 for r in rows if r["completeness"] == cat) / n) if n else float("nan")
    return out
```

- [ ] **Step 4: Run to verify it passes**

- [ ] **Step 5: Write `scripts/09_completeness.py`** joining `regions_{model}.csv` and `descriptive_accuracy_{model}.csv` on `(image_id, rule_id)`, writing `completeness_{model}.csv`, and printing the three rates plus the `quality_of_info` split.

- [ ] **Step 6: Commit**

---

## Plan Self-Review

**Spec coverage.** Stage 4 procedure → Tasks 1–3; artefact control → Task 1 `artefact_floor` and Task 2 `maskable_rows`; both methods per row with one key → Task 2; fingerprint over both inputs → Task 2; Stage 9 three-way → Task 4.

**Deviation from the spec, deliberate.** The spec said Stage 4 snaps the region. `answer_masked` already snaps internally, so Stage 4 computes the snapped box only to record it, and passes the raw region. Double-snapping would be idempotent but misleading to a reader.

**Task 2 Step 3 is a prose description rather than a code block.** That is a deliberate exception for the executing session, where the implementer and the planner are the same agent and the surrounding tasks fix every interface. A fresh implementer would need the code written out.
