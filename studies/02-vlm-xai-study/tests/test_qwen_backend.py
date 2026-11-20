"""Real-model tests. All require CUDA and downloaded weights: run with --gpu."""
import numpy as np
import torch
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


def test_device_name_is_the_gpu_actually_computing(backend):
    import torch
    assert backend.device_name == torch.cuda.get_device_name(backend.model.device)
    assert backend.device_name not in {"", "unknown", "cuda"}


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


def test_attention_is_a_normalised_grid_or_honestly_none(backend, photo):
    att = backend.attention(photo, "rule_1")
    if att is None:
        pytest.skip("attention unavailable for this build -- Sparsity will be "
                    "reported only for models where it works")
    assert att.ndim == 2
    assert att.min() >= 0.0 and att.max() <= 1.0
    assert att.sum() > 0, "an all-zero map means extraction silently failed"


def test_attention_grid_matches_the_images_real_token_grid(backend, photo):
    # The map must be the image's actual grid (21x29 for 1200x900 at the pinned
    # budget), not a square derived from the pixel budget. A square map would
    # misplace every attention value below the first row.
    att = backend.attention(photo, "rule_1")
    if att is None:
        pytest.skip("attention unavailable")
    assert att.shape == backend.patch_grid(photo)


def test_image_token_positions_are_found(backend, photo):
    inputs = backend._build_inputs(photo, "test")
    pos = backend.image_token_positions(inputs)
    assert len(pos) > 0, "no image tokens located -- token ablation cannot work"
    rows, cols = backend.patch_grid(photo)
    assert len(pos) == rows * cols, "token count must equal the merged grid"


def test_multimodal_embeds_actually_contain_the_image(backend, photo):
    # Guards the failure that would make ablation meaningless: if embeddings were
    # built from input_ids alone, image slots would hold a single repeated
    # placeholder vector and the model would see nothing.
    inputs = backend._build_inputs(photo, "test")
    pos = backend.image_token_positions(inputs)
    merged = backend._multimodal_embeds(inputs)
    placeholder = backend.model.model.embed_tokens(inputs["input_ids"])[0, pos]
    actual = merged[0, pos]
    assert not torch.allclose(actual.float(), placeholder.float()), \
        "image slots still hold placeholder embeddings; the model sees no image"
    # Real vision features vary across patches; placeholders are all identical.
    assert actual.float().std(dim=0).mean() > 0


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
