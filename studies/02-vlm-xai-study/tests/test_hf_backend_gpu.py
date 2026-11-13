"""The shared backend with real weights. Requires CUDA: run with --gpu.

Exact agreement with the frozen backend is checked by scripts/20_backend_parity.py;
these tests check that the shared backend is usable on its own.
"""
import pytest

from xai_vlm.backends import Answer, Backend

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def backend():
    from xai_vlm.backends.hf import HFVLMBackend
    b = HFVLMBackend(tier="qwen7b-4bit")
    yield b
    # Free the weights AND reset the peak counter. This module sorts before
    # test_qwen_backend.py, whose VRAM-budget test reads max_memory_allocated();
    # without both, it measures this model's peak, not its own (job 488516).
    import gc

    import torch
    b.adapter.model = b.adapter.processor = None
    del b
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@pytest.fixture
def photo():
    import itertools

    from xai_pilot.data import load_construction_site
    row = next(r for r in itertools.islice(
        load_construction_site(split="test", streaming=True), 200)
        if r["image_id"] == "0000102")
    return row["image"].convert("RGB")


def test_satisfies_the_protocol(backend):
    assert isinstance(backend, Backend)


def test_answer_parses(backend, photo):
    a = backend.answer(photo, "rule_1")
    assert isinstance(a, Answer) and a.ok


def test_attention_grid_is_the_images_token_grid(backend, photo):
    g = backend.attention(photo, "rule_1")
    assert g is not None and g.shape == backend.patch_grid(photo)


def test_token_ablation_records_provenance(backend, photo):
    a = backend.answer_masked(photo, "rule_1", (100, 100, 400, 400), method="tokens")
    assert a.intervention == "tokens"
