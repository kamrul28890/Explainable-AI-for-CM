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
