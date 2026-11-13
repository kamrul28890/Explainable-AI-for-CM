# Study 2, Plan 1: Qwen Backend and the Feasibility Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working `QwenBackend` and run the Stage 0 feasibility gate, producing a defensible go/no-go decision on whether the whole study is viable.

**Architecture:** One class implementing the existing `Backend` protocol (`xai_vlm/backends/base.py`), wrapping Qwen2.5-VL loaded through `transformers` with bitsandbytes quantisation. Model-free logic is tested with a fake backend; real-model behaviour is tested behind a `gpu` marker so the suite stays runnable on the Mac. Stage 0 then measures small-object detection, latency, attention availability and VRAM against thresholds fixed **before** any numbers are seen.

**Tech Stack:** Python 3.10, PyTorch 2.12.1+cu130, transformers 4.49.0, bitsandbytes, qwen-vl-utils, pytest.

**Why this scope:** Stage 0 is a gate, not a formality. If a quantised model cannot see small hard hats, `rule_1` is invalid and so is most of the study. Building metrics before answering that question risks discarding all of it. Plans 2 (sampling + baseline) and 3 (the seven metrics) follow only if this gate passes.

## Global Constraints

- **Machine:** Windows + RTX 3070 (8 GB, ~7.4 GB free). CUDA required — this plan cannot be executed on Apple Silicon (`bitsandbytes` has no build; see architecture doc §7.3).
- **`transformers` pinned to 4.49.0.** `>=5.0` breaks Florence-2's remote code in the sibling study; 4.49.0 is the first release with native Qwen2.5-VL support. It is the only version where both studies run in one environment.
- **`max_pixels` must be identical across every model and every stage.** Qwen2.5-VL uses dynamic resolution, so vision-token count and KV-cache size scale with it. A differing value confounds the comparison with input size rather than model quality. Fixed at `1280 * 28 * 28` (architecture doc Risk #11).
- **Never edit `studies/01-florence2-pilot/`.** It is a frozen scientific record. Import from it; do not modify it.
- **Reuse, do not reimplement:** `xai_pilot.regions.iou`, `mask_region`, `normalized_centroid_distance`; `xai_pilot.perturbations.*`; `xai_pilot.stats.bootstrap_ci`, `meets_min_n`.
- **All new work lives in** `studies/02-vlm-xai-study/`.
- **Test markers:** tests needing a GPU are marked `@pytest.mark.gpu` and skipped by default, so `pytest` stays green on the development machine.

## File Structure

| File | Responsibility |
|---|---|
| `src/xai_vlm/backends/qwen.py` | `QwenBackend` — load, answer, mask, attention, token ablation |
| `src/xai_vlm/backends/fake.py` | `FakeBackend` — scripted replies for testing harness logic without a GPU |
| `src/xai_vlm/config.py` | model tiers, `max_pixels`, generation settings, paths |
| `tests/test_backend_contract.py` | contract tests any backend must satisfy (run against `FakeBackend`) |
| `tests/test_qwen_backend.py` | real-model tests, all `@pytest.mark.gpu` |
| `tests/conftest.py` | the `gpu` marker and its skip logic |
| `scripts/00_feasibility_gate.py` | the gate (replaces the current stub) |
| `data/gate_images.csv` | the 30 hand-picked gate images and their expectations |

---

### Task 1: Environment — install and verify the missing dependencies

Three packages the study needs are absent, and one of them (`qwen-vl-utils`) would fail on the very first model call.

**Files:**
- Modify: `requirements/base.txt` (already contains `qwen-vl-utils`; verify)
- Create: none

**Interfaces:**
- Consumes: nothing
- Produces: an environment where `import bitsandbytes, sentence_transformers, qwen_vl_utils` all succeed

- [ ] **Step 1: Install the three missing packages**

```bash
cd "D:/My Projects/Explainable-AI-Mustafa-Abdallah"
pilot/.venv/Scripts/python.exe -m pip install bitsandbytes sentence-transformers "qwen-vl-utils>=0.0.8"
```

- [ ] **Step 2: Verify each imports, and that bitsandbytes sees CUDA**

```bash
pilot/.venv/Scripts/python.exe -c "
import bitsandbytes, sentence_transformers, qwen_vl_utils, torch
print('bitsandbytes', bitsandbytes.__version__)
print('sentence_transformers', sentence_transformers.__version__)
print('cuda', torch.cuda.is_available())
from transformers import BitsAndBytesConfig
print('BitsAndBytesConfig OK')
"
```

Expected: all four print, `cuda True`. If bitsandbytes errors on import, it installed a CPU-only build — reinstall with `--force-reinstall`.

- [ ] **Step 3: Confirm the environment check now reports no missing required packages**

Run: `pilot/.venv/Scripts/python.exe scripts/check_environment.py`
Expected: `bitsandbytes` and `sentence_transformers` show `[ OK ]`, final line `PASS`.

- [ ] **Step 4: Commit**

```bash
git add requirements/base.txt
git commit -m "deps: add qwen-vl-utils, bitsandbytes, sentence-transformers"
```

---

### Task 2: Config module

Central place for the values that must not drift between stages, above all `max_pixels`.

**Files:**
- Create: `studies/02-vlm-xai-study/src/xai_vlm/config.py`
- Test: `studies/02-vlm-xai-study/tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `MODEL_TIERS: dict[str, ModelSpec]`, `MAX_PIXELS: int`, `MIN_PIXELS: int`, `GENERATION: dict`, `RESULTS_DIR: Path`, `resolve_tier(name: str) -> ModelSpec`

- [ ] **Step 1: Write the failing test**

```python
# studies/02-vlm-xai-study/tests/test_config.py
import pytest
from xai_vlm.config import MAX_PIXELS, MODEL_TIERS, ModelSpec, resolve_tier


def test_max_pixels_is_a_multiple_of_the_patch_block():
    # Qwen2.5-VL uses 14px patches merged 2x2, so area must be a multiple of 28*28
    # or the processor silently rounds and two models see different inputs.
    assert MAX_PIXELS % (28 * 28) == 0


def test_every_tier_declares_what_it_needs():
    for name, spec in MODEL_TIERS.items():
        assert spec.repo_id, name
        assert spec.quantization in {"4bit", "8bit", "none"}, name
        assert isinstance(spec.supports_attention, bool), name


def test_resolve_tier_returns_the_spec():
    spec = resolve_tier("qwen7b-4bit")
    assert isinstance(spec, ModelSpec)
    assert "Qwen2.5-VL-7B" in spec.repo_id
    assert spec.quantization == "4bit"


def test_unknown_tier_names_the_valid_options():
    with pytest.raises(ValueError) as exc:
        resolve_tier("does-not-exist")
    assert "qwen7b-4bit" in str(exc.value)


def test_generation_is_deterministic_by_default():
    from xai_vlm.config import GENERATION
    assert GENERATION["do_sample"] is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pilot/.venv/Scripts/python.exe -m pytest studies/02-vlm-xai-study/tests/test_config.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'xai_vlm.config'`

- [ ] **Step 3: Write the implementation**

```python
# studies/02-vlm-xai-study/src/xai_vlm/config.py
"""Settings that must not drift between models or stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = STUDY_ROOT / "results"
REPO_ROOT = STUDY_ROOT.parents[1]
MODEL_DIR = REPO_ROOT / "models"

# Qwen2.5-VL uses 14px patches merged 2x2 -> 28x28 pixel blocks. Vision-token
# count, and therefore KV-cache size, scales directly with pixel budget: on an
# 8 GB card resolution drives memory harder than model size does.
#
# THIS VALUE IS FIXED ACROSS EVERY MODEL AND STAGE. Varying it would confound the
# comparison with input size rather than model quality (architecture doc Risk #11).
PATCH_BLOCK = 28 * 28
MAX_PIXELS = 1280 * PATCH_BLOCK
MIN_PIXELS = 256 * PATCH_BLOCK

# Deterministic by default. Stability (metric 3) overrides these explicitly, which
# is the point of that metric -- with greedy decoding repeated runs would be
# byte-identical and score perfectly stable by construction.
GENERATION = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 128,
}


@dataclass(frozen=True)
class ModelSpec:
    """One model tier."""

    name: str
    repo_id: str
    quantization: str          # "4bit" | "8bit" | "none"
    supports_attention: bool   # can we read per-layer attention?
    notes: str = ""


MODEL_TIERS: dict[str, ModelSpec] = {
    "qwen7b-4bit": ModelSpec(
        name="qwen7b-4bit",
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        quantization="4bit",
        supports_attention=True,
        notes="primary; largest that fits 8 GB",
    ),
    "qwen3b-8bit": ModelSpec(
        name="qwen3b-8bit",
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        quantization="8bit",
        supports_attention=True,
        notes="fallback and size-comparison arm",
    ),
    "qwen3b-4bit": ModelSpec(
        name="qwen3b-4bit",
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        quantization="4bit",
        supports_attention=True,
        notes="smoke tests when VRAM is tight",
    ),
}


def resolve_tier(name: str) -> ModelSpec:
    """Look up a tier, naming the valid options when it is unknown."""
    if name not in MODEL_TIERS:
        raise ValueError(
            f"unknown model tier {name!r}; choose from {sorted(MODEL_TIERS)}"
        )
    return MODEL_TIERS[name]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pilot/.venv/Scripts/python.exe -m pytest studies/02-vlm-xai-study/tests/test_config.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add studies/02-vlm-xai-study/src/xai_vlm/config.py studies/02-vlm-xai-study/tests/test_config.py
git commit -m "feat(study2): add config with fixed max_pixels and model tiers"
```

---

### Task 3: FakeBackend and the backend contract tests

Every later stage is written against `Backend`. A fake implementation lets that harness logic be developed and tested with no GPU — which is what makes work possible on the Mac.

**Files:**
- Create: `studies/02-vlm-xai-study/src/xai_vlm/backends/fake.py`
- Create: `studies/02-vlm-xai-study/tests/conftest.py`
- Create: `studies/02-vlm-xai-study/tests/test_backend_contract.py`

**Interfaces:**
- Consumes: `xai_vlm.backends.base.Answer`, `Backend`; `xai_vlm.rules.RuleId`
- Produces: `FakeBackend(script: dict | None = None, patch_grid_size: tuple[int,int] = (32,32))` with attributes `.name`, `.calls: list[tuple]`, and methods `answer`, `answer_masked`, `attention`, `patch_grid`

- [ ] **Step 1: Write the conftest with the gpu marker**

```python
# studies/02-vlm-xai-study/tests/conftest.py
"""Shared fixtures and the gpu marker.

GPU tests are skipped unless --gpu is passed, so the suite stays green on the
development machine (see docs/setup/SETUP.md, "Which machine runs what").
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False,
                     help="run tests that require a CUDA GPU and model weights")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA and downloaded weights")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpu"):
        return
    skip = pytest.mark.skip(reason="needs --gpu (CUDA machine only)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
```

- [ ] **Step 2: Write the failing contract test**

```python
# studies/02-vlm-xai-study/tests/test_backend_contract.py
"""Contract every Backend must satisfy, exercised against FakeBackend.

Running these against a fake is deliberate: it verifies the harness contract
without a GPU, and the same tests are re-run against QwenBackend in Task 7.
"""
import numpy as np
import pytest
from PIL import Image

from xai_vlm.backends import Answer, Backend
from xai_vlm.backends.fake import FakeBackend


@pytest.fixture
def img():
    return Image.new("RGB", (1200, 900), (128, 128, 128))


@pytest.fixture
def backend():
    return FakeBackend()


def test_satisfies_the_protocol(backend):
    assert isinstance(backend, Backend)


def test_answer_returns_a_usable_Answer(backend, img):
    a = backend.answer(img, "rule_1")
    assert isinstance(a, Answer)
    assert a.verdict in {"YES", "NO"}
    assert a.ok
    assert a.latency_ms >= 0


def test_scripted_replies_are_returned_in_order(img):
    b = FakeBackend(script={"rule_1": [
        "ANSWER: YES\nREGION: (10, 20, 110, 220)\nREASON: no hard hat",
        "ANSWER: NO\nREGION: NONE\nREASON: all wearing hats",
    ]})
    first, second = b.answer(img, "rule_1"), b.answer(img, "rule_1")
    assert first.verdict == "YES"
    assert first.region == (10.0, 20.0, 110.0, 220.0)
    assert second.verdict == "NO"
    assert second.region is None


def test_masked_answer_records_the_intervention(backend, img):
    a = backend.answer_masked(img, "rule_1", (10, 10, 100, 100), method="tokens")
    assert a.intervention == "tokens"


def test_every_call_is_recorded_for_assertions(backend, img):
    backend.answer(img, "rule_1")
    backend.answer_masked(img, "rule_2", (0, 0, 10, 10), method="black")
    assert [c[0] for c in backend.calls] == ["answer", "answer_masked"]


def test_attention_is_a_normalised_2d_grid(backend, img):
    att = backend.attention(img, "rule_1")
    assert isinstance(att, np.ndarray)
    assert att.ndim == 2
    assert att.min() >= 0.0 and att.max() <= 1.0


def test_patch_grid_is_two_positive_ints(backend):
    rows, cols = backend.patch_grid()
    assert rows > 0 and cols > 0


def test_unparseable_reply_yields_a_not_ok_answer(img):
    b = FakeBackend(script={"rule_1": ["I cannot tell from this image."]})
    a = b.answer(img, "rule_1")
    assert not a.ok
    assert a.parse_error
```

- [ ] **Step 3: Run to verify it fails**

Run: `pilot/.venv/Scripts/python.exe -m pytest studies/02-vlm-xai-study/tests/test_backend_contract.py -v`
Expected: FAIL, `No module named 'xai_vlm.backends.fake'`

- [ ] **Step 4: Implement FakeBackend**

```python
# studies/02-vlm-xai-study/src/xai_vlm/backends/fake.py
"""A Backend that returns scripted replies, for testing without a GPU.

Every stage is written against the Backend protocol, so the whole harness --
sampling, masking bookkeeping, metric aggregation, CSV writing -- can be built and
tested on any machine. Only the real model calls need CUDA.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from xai_vlm.backends.base import Answer
from xai_vlm.parsing import Box, parse_response
from xai_vlm.rules import RuleId

DEFAULT_REPLY = "ANSWER: YES\nREGION: (100, 100, 300, 400)\nREASON: A worker has no hard hat."


class FakeBackend:
    """Deterministic stand-in for a real model.

    `script` maps a rule id to a list of raw reply strings, consumed in order and
    then repeating the last one. Anything not scripted returns DEFAULT_REPLY.
    """

    name = "fake"

    def __init__(self, script: Optional[dict[str, list[str]]] = None,
                 patch_grid_size: tuple[int, int] = (32, 32)):
        self._script = {k: list(v) for k, v in (script or {}).items()}
        self._cursor: dict[str, int] = {}
        self._grid = patch_grid_size
        self.calls: list[tuple] = []

    def _next_reply(self, rule_id: str) -> str:
        replies = self._script.get(rule_id)
        if not replies:
            return DEFAULT_REPLY
        i = self._cursor.get(rule_id, 0)
        self._cursor[rule_id] = i + 1
        return replies[min(i, len(replies) - 1)]

    def answer(self, image: Image.Image, rule_id: RuleId, **kw) -> Answer:
        self.calls.append(("answer", rule_id, kw))
        raw = self._next_reply(rule_id)
        parsed = parse_response(raw, image.size)
        return Answer.from_parsed(parsed, raw_text=raw, latency_ms=1.0)

    def answer_masked(self, image: Image.Image, rule_id: RuleId, region: Box,
                      method: str = "black", **kw) -> Answer:
        self.calls.append(("answer_masked", rule_id, method, region))
        raw = self._next_reply(rule_id)
        parsed = parse_response(raw, image.size)
        return Answer.from_parsed(parsed, raw_text=raw, latency_ms=1.0,
                                  intervention=method)

    def attention(self, image: Image.Image, rule_id: RuleId, **kw) -> np.ndarray:
        self.calls.append(("attention", rule_id))
        rows, cols = self._grid
        # A deterministic peaked map, so tests can assert on concentration.
        grid = np.zeros((rows, cols), dtype=float)
        grid[rows // 2, cols // 2] = 1.0
        grid[rows // 2 - 1: rows // 2 + 2, cols // 2 - 1: cols // 2 + 2] += 0.25
        return np.clip(grid / grid.max(), 0.0, 1.0)

    def patch_grid(self) -> tuple[int, int]:
        return self._grid
```

- [ ] **Step 5: Run to verify it passes**

Run: `pilot/.venv/Scripts/python.exe -m pytest studies/02-vlm-xai-study/tests/ -v`
Expected: all pass (40 existing + 5 config + 8 contract = 53)

- [ ] **Step 6: Commit**

```bash
git add studies/02-vlm-xai-study/src/xai_vlm/backends/fake.py studies/02-vlm-xai-study/tests/conftest.py studies/02-vlm-xai-study/tests/test_backend_contract.py
git commit -m "feat(study2): add FakeBackend and backend contract tests"
```

---

### Task 4: Download the model weights

**Files:**
- Modify: none (uses the existing `scripts/download_data.py`)

**Interfaces:**
- Consumes: `scripts/download_data.py`
- Produces: weights on disk at `models/qwen7b-4bit/`

- [ ] **Step 1: Check free disk (7B needs ~17 GB unquantised on disk)**

```bash
pilot/.venv/Scripts/python.exe -c "import shutil; print(f'{shutil.disk_usage(\".\").free/1e9:.0f} GB free')"
```

Expected: > 25 GB. Weights download at full precision and are quantised at load time.

- [ ] **Step 2: Download**

```bash
pilot/.venv/Scripts/python.exe scripts/download_data.py --what models --tier qwen7b-4bit
```

- [ ] **Step 3: Verify the snapshot is complete**

```bash
pilot/.venv/Scripts/python.exe -c "
from pathlib import Path
d = Path('models/qwen7b-4bit')
sf = list(d.glob('*.safetensors'))
print('safetensors shards:', len(sf))
print('total GB:', round(sum(p.stat().st_size for p in d.rglob('*'))/1e9, 1))
for req in ['config.json','preprocessor_config.json']:
    print(req, (d/req).exists())
"
```

Expected: several shards, ~16 GB, both config files present. A missing `preprocessor_config.json` means the snapshot is partial — re-run the download.

- [ ] **Step 4: No commit** (weights are gitignored by design)

---

### Task 5: QwenBackend — loading and `answer()`

The core of the plan.

**Files:**
- Create: `studies/02-vlm-xai-study/src/xai_vlm/backends/qwen.py`
- Create: `studies/02-vlm-xai-study/tests/test_qwen_backend.py`

**Interfaces:**
- Consumes: `xai_vlm.config.{ModelSpec, resolve_tier, MAX_PIXELS, MIN_PIXELS, GENERATION, MODEL_DIR}`, `xai_vlm.rules.prompt_for`, `xai_vlm.parsing.parse_response`, `xai_vlm.backends.base.{Answer, BackendUnavailable}`
- Produces: `QwenBackend(tier: str = "qwen7b-4bit", device: str = "cuda")` with `.name`, `.spec`, `.answer()`, `.patch_grid()`, `.peak_memory_gb()`

- [ ] **Step 1: Write the failing GPU test**

```python
# studies/02-vlm-xai-study/tests/test_qwen_backend.py
"""Real-model tests. All require CUDA and downloaded weights: run with --gpu."""
import numpy as np
import pytest
from PIL import Image

from xai_vlm.backends import Answer
from xai_vlm.backends.qwen import QwenBackend

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def backend():
    return QwenBackend(tier="qwen7b-4bit")


@pytest.fixture
def photo():
    # A real construction photo from the frozen pilot's dataset.
    from xai_pilot.data import load_construction_site
    import itertools
    row = next(r for r in itertools.islice(
        load_construction_site(split="test", streaming=True), 200)
        if r["image_id"] == "0000102")
    return row["image"].convert("RGB")


def test_model_loads_and_reports_its_patch_grid(backend):
    rows, cols = backend.patch_grid()
    assert rows > 0 and cols > 0


def test_stays_within_the_vram_budget(backend, photo):
    backend.answer(photo, "rule_1")
    assert backend.peak_memory_gb() < 7.0, "exceeds the 8 GB card's usable budget"


def test_answer_returns_a_parsed_verdict(backend, photo):
    a = backend.answer(photo, "rule_1")
    assert isinstance(a, Answer)
    assert a.verdict in {"YES", "NO"}, f"unparsed reply: {a.raw_text!r}"
    assert a.latency_ms > 0
    assert a.raw_text


def test_it_finds_the_known_violation(backend, photo):
    # 0000102 ground truth: "The mechanic in front of the truck is not wearing a
    # hard hat." If the model cannot see this, Stage 0 will fail and it should.
    a = backend.answer(photo, "rule_1")
    assert a.verdict == "YES"


def test_greedy_decoding_is_reproducible(backend, photo):
    a, b = backend.answer(photo, "rule_1"), backend.answer(photo, "rule_1")
    assert a.verdict == b.verdict


def test_unknown_tier_raises_before_loading_anything():
    with pytest.raises(ValueError):
        QwenBackend(tier="not-a-tier")
```

- [ ] **Step 2: Run to verify it fails**

Run: `pilot/.venv/Scripts/python.exe -m pytest studies/02-vlm-xai-study/tests/test_qwen_backend.py --gpu -v`
Expected: FAIL, `No module named 'xai_vlm.backends.qwen'`

- [ ] **Step 3: Implement QwenBackend**

```python
# studies/02-vlm-xai-study/src/xai_vlm/backends/qwen.py
"""Qwen2.5-VL backend.

Chosen because it is the only small model that answers free-form questions, ALSO
returns bounding boxes (keeping the masking metrics alive), and ships open weights
(so attention is readable and image tokens are ablatable). A closed API model would
have made Sparsity and token ablation impossible.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
from PIL import Image

from xai_vlm.backends.base import Answer, BackendUnavailable
from xai_vlm.config import (GENERATION, MAX_PIXELS, MIN_PIXELS, MODEL_DIR,
                            ModelSpec, resolve_tier)
from xai_vlm.parsing import Box, parse_response
from xai_vlm.rules import RuleId, prompt_for


class QwenBackend:
    """Qwen2.5-VL wrapped to satisfy the Backend protocol."""

    def __init__(self, tier: str = "qwen7b-4bit", device: str = "cuda"):
        # Validate before importing torch machinery, so a typo fails instantly
        # rather than after a two-minute model load.
        self.spec: ModelSpec = resolve_tier(tier)
        self.name = self.spec.name
        self.device = device

        if device == "cuda" and not torch.cuda.is_available():
            raise BackendUnavailable(
                "CUDA is not available. This backend requires the compute machine; "
                "bitsandbytes has no Apple Silicon build. See docs/setup/SETUP.md."
            )

        from transformers import (AutoProcessor, BitsAndBytesConfig,
                                  Qwen2_5_VLForConditionalGeneration)

        local = MODEL_DIR / self.spec.name
        source = str(local) if local.exists() else self.spec.repo_id

        quant = None
        if self.spec.quantization == "4bit":
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.spec.quantization == "8bit":
            quant = BitsAndBytesConfig(load_in_8bit=True)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            source,
            quantization_config=quant,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",   # required to read attention later
        )
        self.model.eval()

        # min/max pixels are pinned here and nowhere else: they must be identical
        # across every model or the comparison is confounded by input size.
        self.processor = AutoProcessor.from_pretrained(
            source, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )

    # ------------------------------------------------------------------ helpers
    def _build_inputs(self, image: Image.Image, prompt: str):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        return self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

    def _decode_new_tokens(self, generated, inputs) -> str:
        # Strip the prompt: generate() returns prompt + continuation.
        trimmed = [out[len(inp):] for inp, out in
                   zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    # -------------------------------------------------------------------- API
    def answer(self, image: Image.Image, rule_id: RuleId, **kw) -> Answer:
        prompt = prompt_for(rule_id)
        inputs = self._build_inputs(image, prompt)

        gen = {**GENERATION, **kw}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw = self._decode_new_tokens(out, inputs)
        parsed = parse_response(raw, image.size)
        return Answer.from_parsed(parsed, raw_text=raw, latency_ms=latency_ms)

    def patch_grid(self) -> tuple[int, int]:
        """(rows, cols) of the merged patch grid at the pinned pixel budget."""
        vc = self.model.config.vision_config
        patch = getattr(vc, "patch_size", 14)
        merge = getattr(vc, "spatial_merge_size", 2)
        block = patch * merge
        side = int((MAX_PIXELS ** 0.5) // block)
        return (side, side)

    def peak_memory_gb(self) -> float:
        if not torch.cuda.is_available():
            return float("nan")
        return torch.cuda.max_memory_allocated() / 1e9
```

- [ ] **Step 4: Run to verify it passes**

Run: `pilot/.venv/Scripts/python.exe -m pytest studies/02-vlm-xai-study/tests/test_qwen_backend.py --gpu -v`
Expected: 6 passed. If `test_it_finds_the_known_violation` fails, **do not tune the test** — that is the small-object risk this plan exists to measure, and Stage 0 must record it.

- [ ] **Step 5: Commit**

```bash
git add studies/02-vlm-xai-study/src/xai_vlm/backends/qwen.py studies/02-vlm-xai-study/tests/test_qwen_backend.py
git commit -m "feat(study2): add QwenBackend with loading and answer()"
```

---

### Task 6: Masking — patch-aligned black fill and token ablation

Two interventions. Reporting them side by side measures how much the black-rectangle artefact inflates flip rates.

**Files:**
- Modify: `studies/02-vlm-xai-study/src/xai_vlm/backends/qwen.py`
- Create: `studies/02-vlm-xai-study/src/xai_vlm/masking.py`
- Create: `studies/02-vlm-xai-study/tests/test_masking.py`

**Interfaces:**
- Consumes: `xai_pilot.regions.mask_region`
- Produces: `snap_to_patch_grid(box, image_size, block=28) -> Box`; `QwenBackend.answer_masked(image, rule_id, region, method="black") -> Answer`

> **Forward dependency, by design.** The `answer_masked` code below dispatches
> `method="tokens"` to `self._answer_token_ablated(...)`, which **Task 7 adds**.
> Until Task 7 is complete, only `method` in `{"black", "blur", "inpaint"}` works;
> `"tokens"` raises `AttributeError`. Step 6's verification uses `"black"` only,
> so Task 6 is independently testable as written. Do not stub
> `_answer_token_ablated` here — a stub that silently returns a plausible answer
> would make the artefact-free measurement untrustworthy without failing loudly.

- [ ] **Step 1: Write the failing test for grid snapping**

```python
# studies/02-vlm-xai-study/tests/test_masking.py
"""Patch-grid alignment. No GPU required."""
import pytest

from xai_vlm.masking import snap_to_patch_grid


def test_snapping_expands_outward_to_block_borders():
    # A mask cutting a patch in half leaves a partial, ambiguous signal in it.
    box = snap_to_patch_grid((30, 30, 100, 100), (1200, 900), block=28)
    x0, y0, x1, y1 = box
    assert x0 % 28 == 0 and y0 % 28 == 0
    assert x1 % 28 == 0 and y1 % 28 == 0
    assert x0 <= 30 and y0 <= 30      # never shrinks
    assert x1 >= 100 and y1 >= 100


def test_an_already_aligned_box_is_unchanged():
    assert snap_to_patch_grid((28, 56, 84, 112), (1200, 900), block=28) == (28, 56, 84, 112)


def test_snapping_is_clamped_to_the_image():
    x0, y0, x1, y1 = snap_to_patch_grid((1190, 890, 1199, 899), (1200, 900), block=28)
    assert x1 <= 1200 and y1 <= 900
    assert x1 > x0 and y1 > y0


def test_a_tiny_box_still_covers_at_least_one_block():
    x0, y0, x1, y1 = snap_to_patch_grid((100, 100, 101, 101), (1200, 900), block=28)
    assert x1 - x0 >= 28 and y1 - y0 >= 28
```

- [ ] **Step 2: Run to verify it fails**

Run: `pilot/.venv/Scripts/python.exe -m pytest studies/02-vlm-xai-study/tests/test_masking.py -v`
Expected: FAIL, `No module named 'xai_vlm.masking'`

- [ ] **Step 3: Implement snapping**

```python
# studies/02-vlm-xai-study/src/xai_vlm/masking.py
"""Interventions that remove evidence from an image.

A black rectangle is something no real photograph contains, so the model may react
to the rectangle rather than to the missing evidence -- a "flip" then means "this
image looks broken" rather than "the hard hat is gone". Aligning masks to the patch
grid removes one source of that noise; token ablation (in the backend) removes it
entirely by never modifying pixels at all.
"""

from __future__ import annotations

import math

from xai_vlm.parsing import Box


def snap_to_patch_grid(box: Box, image_size: tuple[int, int], block: int = 28) -> Box:
    """Expand `box` outward so it covers whole patch blocks.

    Qwen2.5-VL sees 14px patches merged 2x2 -> 28px blocks. A mask that cuts a
    block in half leaves a partial signal in it, so masks are always expanded
    (never shrunk) to block boundaries, then clamped to the image.
    """
    w, h = image_size
    x0, y0, x1, y1 = box

    sx0 = max(0, int(math.floor(x0 / block) * block))
    sy0 = max(0, int(math.floor(y0 / block) * block))
    sx1 = min(w, int(math.ceil(x1 / block) * block))
    sy1 = min(h, int(math.ceil(y1 / block) * block))

    # Guarantee at least one whole block, even for a degenerate input box.
    if sx1 <= sx0:
        sx0 = max(0, min(sx0, w - block))
        sx1 = min(w, sx0 + block)
    if sy1 <= sy0:
        sy0 = max(0, min(sy0, h - block))
        sy1 = min(h, sy0 + block)

    return (float(sx0), float(sy0), float(sx1), float(sy1))
```

- [ ] **Step 4: Run to verify it passes**

Run: `pilot/.venv/Scripts/python.exe -m pytest studies/02-vlm-xai-study/tests/test_masking.py -v`
Expected: 4 passed

- [ ] **Step 5: Add `answer_masked` to QwenBackend**

Append to `studies/02-vlm-xai-study/src/xai_vlm/backends/qwen.py`, inside the class:

```python
    def answer_masked(self, image: Image.Image, rule_id: RuleId, region: Box,
                      method: str = "black", **kw) -> Answer:
        """Ask again with `region` removed.

        "black"   patch-aligned fill -- comparable with prior work, but creates an
                  artefact no photograph contains.
        "inpaint" reconstruct from surroundings -- least out-of-distribution pixels.
        "tokens"  drop the image tokens for those patches before the language model
                  runs. No pixels are modified at all, so no artefact can exist.
                  This is the primary, artefact-free measure.
        """
        from xai_pilot.regions import mask_region

        from xai_vlm.masking import snap_to_patch_grid

        snapped = snap_to_patch_grid(region, image.size)

        if method in {"black", "blur", "inpaint"}:
            masked = mask_region(image, snapped, mode=method)
            out = self.answer(masked, rule_id, **kw)
            out.intervention = method
            return out

        if method == "tokens":
            return self._answer_token_ablated(image, rule_id, snapped, **kw)

        raise ValueError(f"unknown masking method {method!r}")
```

- [ ] **Step 6: Verify black masking works end to end**

```bash
pilot/.venv/Scripts/python.exe -c "
from PIL import Image
from xai_vlm.backends.qwen import QwenBackend
from xai_pilot.data import load_construction_site
import itertools
row = next(r for r in itertools.islice(load_construction_site(split='test', streaming=True), 200) if r['image_id']=='0000102')
img = row['image'].convert('RGB')
b = QwenBackend()
base = b.answer(img, 'rule_1')
mask = b.answer_masked(img, 'rule_1', (192, 549, 276, 765), method='black')
print('baseline:', base.verdict, '| masked:', mask.verdict, '| flipped:', base.verdict != mask.verdict)
print('intervention recorded:', mask.intervention)
"
```

Expected: both verdicts print, `intervention recorded: black`.

- [ ] **Step 7: Commit**

```bash
git add studies/02-vlm-xai-study/src/xai_vlm/masking.py studies/02-vlm-xai-study/tests/test_masking.py studies/02-vlm-xai-study/src/xai_vlm/backends/qwen.py
git commit -m "feat(study2): add patch-aligned masking and answer_masked"
```

---

### Task 7: Attention extraction and token ablation

The two capabilities that make Sparsity and artefact-free masking possible. Both are **allowed to fail** — if they do, the study proceeds with a documented limitation rather than a fabricated number.

**Files:**
- Modify: `studies/02-vlm-xai-study/src/xai_vlm/backends/qwen.py`
- Modify: `studies/02-vlm-xai-study/tests/test_qwen_backend.py`

**Interfaces:**
- Produces: `QwenBackend.attention(image, rule_id) -> Optional[np.ndarray]`, `QwenBackend._answer_token_ablated(image, rule_id, region) -> Answer`, `QwenBackend.image_token_positions(inputs) -> torch.Tensor`

- [ ] **Step 1: Write the failing GPU tests**

Append to `studies/02-vlm-xai-study/tests/test_qwen_backend.py`:

```python
def test_attention_is_a_normalised_grid_or_honestly_none(backend, photo):
    att = backend.attention(photo, "rule_1")
    if att is None:
        pytest.skip("attention unavailable for this build -- Sparsity will be "
                    "reported only for models where it works")
    assert att.ndim == 2
    assert att.min() >= 0.0 and att.max() <= 1.0
    assert att.sum() > 0, "an all-zero map means extraction silently failed"


def test_image_token_positions_are_found(backend, photo):
    inputs = backend._build_inputs(photo, "test")
    pos = backend.image_token_positions(inputs)
    assert len(pos) > 0, "no image tokens located -- token ablation cannot work"


def test_token_ablation_returns_an_answer_and_records_provenance(backend, photo):
    a = backend.answer_masked(photo, "rule_1", (192, 549, 276, 765), method="tokens")
    assert a.intervention == "tokens"
    assert a.verdict in {"YES", "NO"} or a.parse_error


def test_ablating_everything_changes_the_answer_or_degrades_it(backend, photo):
    # Sanity check on the mechanism: removing the entire image should not leave
    # the model equally confident about a visual question.
    full = backend.answer_masked(photo, "rule_1", (0, 0, photo.width, photo.height),
                                 method="tokens")
    base = backend.answer(photo, "rule_1")
    assert full.verdict != base.verdict or full.rationale != base.rationale
```

- [ ] **Step 2: Run to verify it fails**

Run: `pilot/.venv/Scripts/python.exe -m pytest studies/02-vlm-xai-study/tests/test_qwen_backend.py --gpu -v -k "attention or token or ablat"`
Expected: FAIL, `AttributeError: 'QwenBackend' object has no attribute 'attention'`

- [ ] **Step 3: Implement attention and ablation**

Append inside the `QwenBackend` class:

```python
    def image_token_positions(self, inputs) -> "torch.Tensor":
        """Indices of image-placeholder tokens in the prompt sequence.

        Qwen2.5-VL is decoder-only: image tokens sit inline in the sequence rather
        than in a separate encoder. So "attention to the image" means attention
        from generated tokens to these positions, and ablation means dropping them.
        """
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                "<|image_pad|>"
            )
        ids = inputs.input_ids[0]
        return (ids == image_token_id).nonzero(as_tuple=True)[0]

    def attention(self, image: Image.Image, rule_id: RuleId, **kw
                  ) -> Optional[np.ndarray]:
        """Mean attention from generated tokens back to image patches.

        Returns None rather than a fabricated map if extraction is unavailable --
        Sparsity is then reported only for backends where it genuinely works.
        """
        prompt = prompt_for(rule_id)
        inputs = self._build_inputs(image, prompt)
        positions = self.image_token_positions(inputs)
        if len(positions) == 0:
            return None

        try:
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs, **{**GENERATION, **kw},
                    output_attentions=True, return_dict_in_generate=True,
                )
        except (RuntimeError, TypeError):
            return None

        if not getattr(out, "attentions", None):
            return None

        # out.attentions: one tuple per generated step, each holding one tensor
        # per layer of shape (batch, heads, query_len, key_len). Average over
        # steps, layers and heads, then keep the columns that are image tokens.
        acc, n = None, 0
        for step in out.attentions:
            for layer in step:
                a = layer[0].float().mean(dim=0)      # (query_len, key_len)
                sel = a[-1, positions]                # last query row -> image keys
                acc = sel if acc is None else acc + sel
                n += 1
        if acc is None or n == 0:
            return None

        flat = (acc / n).cpu().numpy()
        side = int(np.sqrt(flat.size))
        if side * side != flat.size:
            # Non-square token layout: pad to the next square so it can be shown
            # as a grid, rather than silently reshaping into a wrong aspect.
            side = int(np.ceil(np.sqrt(flat.size)))
            padded = np.zeros(side * side, dtype=float)
            padded[:flat.size] = flat
            flat = padded
        grid = flat.reshape(side, side)
        peak = grid.max()
        return grid / peak if peak > 0 else grid

    def _answer_token_ablated(self, image: Image.Image, rule_id: RuleId,
                              region: Box, **kw) -> Answer:
        """Answer with the image tokens covering `region` removed.

        No pixels are modified, so no out-of-distribution artefact can exist. This
        is the cleanest available statement of "the model could not see this area".
        """
        prompt = prompt_for(rule_id)
        inputs = self._build_inputs(image, prompt)
        positions = self.image_token_positions(inputs)
        if len(positions) == 0:
            raise BackendUnavailable("no image tokens found; token ablation "
                                     "unavailable for this model build")

        rows, cols = self.patch_grid()
        w, h = image.size
        x0, y0, x1, y1 = region

        # Map the pixel box onto the token grid. Tokens are laid out row-major.
        keep = torch.ones(len(positions), dtype=torch.bool)
        for i in range(len(positions)):
            r, c = divmod(i, cols)
            px0, py0 = c * w / cols, r * h / rows
            px1, py1 = (c + 1) * w / cols, (r + 1) * h / rows
            if not (px1 <= x0 or px0 >= x1 or py1 <= y0 or py0 >= y1):
                keep[i] = False

        drop = positions[~keep.to(positions.device)]
        if len(drop) == 0:
            out = self.answer(image, rule_id, **kw)
            out.intervention = "tokens"
            return out

        mask = torch.ones(inputs.input_ids.shape[1], dtype=torch.bool,
                          device=inputs.input_ids.device)
        mask[drop] = False

        embeds = self.model.get_input_embeddings()(inputs.input_ids)
        kept_embeds = embeds[:, mask, :]
        kept_attention = inputs.attention_mask[:, mask]

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(
                inputs_embeds=kept_embeds,
                attention_mask=kept_attention,
                **{**GENERATION, **kw},
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # With inputs_embeds there is no prompt prefix in the output to strip.
        raw = self.processor.batch_decode(
            out, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        parsed = parse_response(raw, image.size)
        return Answer.from_parsed(parsed, raw_text=raw, latency_ms=latency_ms,
                                  intervention="tokens")
```

- [ ] **Step 4: Run to verify it passes**

Run: `pilot/.venv/Scripts/python.exe -m pytest studies/02-vlm-xai-study/tests/test_qwen_backend.py --gpu -v`
Expected: all pass, or `test_attention_...` skips with an explicit reason. A skip is an acceptable outcome and must be recorded in Stage 0 — it is not a failure to work around.

- [ ] **Step 5: Commit**

```bash
git add studies/02-vlm-xai-study/src/xai_vlm/backends/qwen.py studies/02-vlm-xai-study/tests/test_qwen_backend.py
git commit -m "feat(study2): add attention extraction and image-token ablation"
```

---

### Task 8: The Stage 0 feasibility gate

**Files:**
- Create: `studies/02-vlm-xai-study/data/gate_images.csv`
- Modify: `studies/02-vlm-xai-study/scripts/00_feasibility_gate.py` (replace the stub)

**Interfaces:**
- Consumes: `QwenBackend`, `xai_vlm.rules.prompt_for`, `xai_pilot.data.load_construction_site`
- Produces: `results/stage0_feasibility.csv` and a printed GO / GO-WITH-LIMITATION / STOP verdict

- [ ] **Step 1: Build the gate image list**

```bash
cd "D:/My Projects/Explainable-AI-Mustafa-Abdallah"
pilot/.venv/Scripts/python.exe - <<'PY'
import csv, itertools
from pathlib import Path
from xai_pilot.data import load_construction_site

out = Path("studies/02-vlm-xai-study/data/gate_images.csv")
out.parent.mkdir(parents=True, exist_ok=True)

violations, compliant = [], []
for row in itertools.islice(load_construction_site(split="test", streaming=True), 600):
    v = row.get("rule_1_violation")
    w, h = row["image"].size
    if isinstance(v, dict) and v.get("reason") and len(violations) < 20:
        box = v["bounding_box"][0]
        area = (box[2]-box[0]) * (box[3]-box[1])   # normalised area
        violations.append((row["image_id"], "YES", v["reason"][:90], round(area, 5)))
    elif all(row.get(f"rule_{i}_violation") is None for i in (1,2,3,4)) and len(compliant) < 10:
        compliant.append((row["image_id"], "NO", "no violation recorded", ""))

# The 10 smallest violation boxes are the small-object stress cases: they are what
# quantisation and resolution capping damage first.
violations.sort(key=lambda r: r[3])
rows = violations[:10] + violations[10:20] + compliant

with out.open("w", newline="", encoding="utf-8") as f:
    wtr = csv.writer(f)
    wtr.writerow(["image_id", "expected", "reason", "norm_area"])
    wtr.writerows(rows)
print(f"wrote {len(rows)} gate images to {out}")
PY
```

- [ ] **Step 2: Replace the stub with the real gate**

```python
# studies/02-vlm-xai-study/scripts/00_feasibility_gate.py
"""Stage 0 -- the feasibility gate. Run before building anything else.

Decides whether the study is viable. The decisive question is whether a quantised
model can still detect small hard hats: if it cannot, rule_1 is invalid and so is
most of the study.

Thresholds are fixed in the architecture document BEFORE any numbers are seen.
Do not tune them to fit an observed result -- that is what they exist to prevent.

    python scripts/00_feasibility_gate.py --tier qwen7b-4bit
"""
import argparse
import csv
import itertools
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

DETECTION_PROCEED = 0.80
DETECTION_LIMITATION = 0.50
LATENCY_BUDGET_S = 6.0
VRAM_BUDGET_GB = 7.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="qwen7b-4bit")
    ap.add_argument("--out", default=str(STUDY_ROOT / "results" / "stage0_feasibility.csv"))
    args = ap.parse_args()

    from xai_pilot.data import load_construction_site
    from xai_vlm.backends.qwen import QwenBackend

    gate = list(csv.DictReader(
        (STUDY_ROOT / "data" / "gate_images.csv").open(encoding="utf-8")))
    wanted = {g["image_id"] for g in gate}
    images = {}
    for row in load_construction_site(split="test", streaming=True):
        if row["image_id"] in wanted:
            images[row["image_id"]] = row["image"].convert("RGB")
        if len(images) >= len(wanted):
            break

    print(f"loading {args.tier} ...")
    backend = QwenBackend(tier=args.tier)

    rows, latencies = [], []
    for g in gate:
        img = images.get(g["image_id"])
        if img is None:
            continue
        a = backend.answer(img, "rule_1")
        latencies.append(a.latency_ms / 1000.0)
        rows.append({
            "image_id": g["image_id"], "expected": g["expected"],
            "predicted": a.verdict or "UNPARSED", "correct": a.verdict == g["expected"],
            "norm_area": g["norm_area"], "latency_s": round(a.latency_ms / 1000.0, 2),
            "has_region": a.region is not None, "parse_strict": a.parse_strict,
            "rationale": a.rationale[:120], "raw": a.raw_text[:200],
        })
        print(f"  {g['image_id']}  exp={g['expected']}  got={a.verdict}  "
              f"{a.latency_ms/1000:.1f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # --- verdict -----------------------------------------------------------
    small = [r for r in rows if r["norm_area"] and float(r["norm_area"]) < 0.01
             and r["expected"] == "YES"]
    small_rate = sum(r["correct"] for r in small) / len(small) if small else float("nan")
    overall = sum(r["correct"] for r in rows) / len(rows)
    median_latency = sorted(latencies)[len(latencies) // 2]
    parse_rate = sum(1 for r in rows if r["predicted"] != "UNPARSED") / len(rows)
    peak = backend.peak_memory_gb()

    print("\n" + "=" * 66)
    print(f"small-object detection : {small_rate:.0%}  (n={len(small)})")
    print(f"overall accuracy       : {overall:.0%}")
    print(f"median latency         : {median_latency:.1f}s")
    print(f"parse rate             : {parse_rate:.0%}")
    print(f"peak VRAM              : {peak:.1f} GB")
    print(f"attention available    : {backend.attention(list(images.values())[0], 'rule_1') is not None}")
    print("=" * 66)

    if small_rate != small_rate or small_rate < DETECTION_LIMITATION:
        print("VERDICT: STOP. Raise max_pixels or drop to 3B-8bit and retest.")
        return 1
    if small_rate < DETECTION_PROCEED:
        print("VERDICT: GO, with small-object sensitivity as a stated limitation.")
    else:
        print("VERDICT: GO.")
    if median_latency > LATENCY_BUDGET_S:
        est = median_latency * 27450 / 3600
        print(f"WARNING: at {median_latency:.1f}s/call the full study is ~{est:.0f}h. "
              "Rescope: robustness subset -> stability subset -> third model. "
              "Never cut Stage 4.")
    if peak > VRAM_BUDGET_GB:
        print(f"WARNING: peak VRAM {peak:.1f} GB is close to the limit; lower max_pixels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run the gate**

```bash
pilot/.venv/Scripts/python.exe studies/02-vlm-xai-study/scripts/00_feasibility_gate.py --tier qwen7b-4bit
```

Expected: a per-image trace, a summary block, and one of GO / GO-WITH-LIMITATION / STOP.

- [ ] **Step 4: Commit the gate and its result**

```bash
git add studies/02-vlm-xai-study/scripts/00_feasibility_gate.py studies/02-vlm-xai-study/data/gate_images.csv studies/02-vlm-xai-study/results/stage0_feasibility.csv
git commit -m "feat(study2): implement Stage 0 feasibility gate and record its result"
```

- [ ] **Step 5: Record the decision in the architecture document**

Add the measured numbers and the verdict to §12 (Risks), retiring or confirming Risks #1, #2, #3 and #4. **If the verdict is STOP, do not proceed to Plan 2** — revisit the model tier first.

---

## What this plan deliberately does not cover

Plans 2 and 3 follow only if the gate passes:

- **Plan 2** — Stage 1 (two-stratum sample construction) and Stage 2 (baseline inference with checkpointing and resume).
- **Plan 3** — Stages 3–10, the seven metrics.

Splitting here is the point: if a quantised Qwen cannot see small hard hats, everything after this is wasted effort, and the cheapest possible discovery of that fact is worth more than partial progress on metrics.
