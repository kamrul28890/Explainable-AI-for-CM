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


def test_unquantised_7b_tier_exists_for_the_a40():
    spec = resolve_tier("qwen7b-bf16")
    assert spec.quantization == "none"
    assert spec.repo_id == resolve_tier("qwen7b-4bit").repo_id


def test_bf16_tier_loads_the_same_weights_as_the_4bit_tier():
    """Quantisation happens at load time; the on-disk snapshot is the full-precision
    checkpoint either way. Two copies would be two things to keep in sync -- and a
    second download could land on a different revision."""
    assert resolve_tier("qwen7b-bf16").weights_dir == resolve_tier("qwen7b-4bit").weights_dir
    assert resolve_tier("qwen3b-8bit").weights_dir == "qwen3b-8bit"   # default: own name


def test_every_tier_sharing_weights_shares_a_pinned_revision():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("s2", root / "scripts" / "02_baseline_inference.py")
    s2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s2)
    assert s2._revision_for("qwen7b-bf16") == s2._revision_for("qwen7b-4bit") != "unpinned"
