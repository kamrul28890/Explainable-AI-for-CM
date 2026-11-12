"""Region coordinate frames: model-input frame -> original image. No GPU."""
import importlib
import sys
from pathlib import Path

import pytest

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config, frames  # noqa: E402


def test_to_original_scales_each_axis_by_its_own_ratio():
    # 1200x900 photo fed to the model as 812x588
    box = frames.to_original((406, 294, 812, 588), (1200, 900), (812, 588))
    assert box == pytest.approx((600.0, 450.0, 1200.0, 900.0))


def test_to_original_clamps_to_the_image_and_drops_degenerate_boxes():
    assert frames.to_original((700, 500, 900, 700), (1200, 900), (812, 588)) == pytest.approx(
        (700 * 1200 / 812, 500 * 900 / 588, 1200.0, 900.0))
    assert frames.to_original((812, 10, 900, 20), (1200, 900), (812, 588)) is None


def test_input_size_from_grid_uses_pre_merge_patches():
    # image_grid_thw (t, h, w) in 14 px patches: 42 x 58 -> 588 x 812
    assert frames.input_size_from_grid((1, 42, 58), patch_size=14) == (812, 588)


def test_region_from_reply_parses_in_the_input_frame_then_rescales():
    raw = "REASON: no hard hat\nREGION: (406, 294, 609, 441)\nANSWER: YES"
    assert frames.region_from_reply(raw, (1200, 900), (812, 588)) == pytest.approx(
        (600.0, 450.0, 900.0, 675.0))
    assert frames.region_from_reply("REGION: NONE\nANSWER: NO", (1200, 900), (812, 588)) is None


def test_rescale_box_text_round_trips_the_csv_format():
    assert frames.rescale_box_text("(406.0, 294.0, 609.0, 441.0)", (1200, 900), (812, 588)) \
        == "(600.0, 450.0, 900.0, 675.0)"
    assert frames.rescale_box_text("", (1200, 900), (812, 588)) == ""


def test_region_frame_flag_defaults_to_legacy_and_adds_no_fingerprint_key(monkeypatch):
    monkeypatch.delenv("XAI_REGION_FRAME", raising=False)
    cfg = importlib.reload(config)
    assert cfg.REGION_FRAME == "legacy"
    assert cfg.region_frame_fingerprint() == {}
    monkeypatch.setenv("XAI_REGION_FRAME", "rescaled")
    cfg = importlib.reload(config)
    assert cfg.region_frame_fingerprint() == {"region_frame": "rescaled"}
    monkeypatch.setenv("XAI_REGION_FRAME", "")  # `env XAI_REGION_FRAME= cmd` clears it
    assert importlib.reload(config).REGION_FRAME == "legacy"
    monkeypatch.setenv("XAI_REGION_FRAME", "bogus")
    with pytest.raises(ValueError):
        importlib.reload(config)
    monkeypatch.delenv("XAI_REGION_FRAME")
    importlib.reload(config)


@pytest.mark.skipif(not (config.MODEL_DIR / "qwen7b-4bit").exists(), reason="needs local processor")
def test_processor_frame_matches_the_measured_one():
    """The frame the model actually sees, from the real preprocessing path."""
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(str(config.MODEL_DIR / "qwen7b-4bit"),
                                         min_pixels=config.MIN_PIXELS, max_pixels=config.MAX_PIXELS)
    f = frames.ModelInputFrame(proc)
    assert f.size_for((1200, 900)) == (812, 588)
    w, h = f.size_for((900, 1200))
    assert (w, h) == (588, 812)


def test_qwen_backend_parse_rescales_only_under_the_flag(monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from PIL import Image
    from xai_vlm.backends.qwen import QwenBackend
    b = QwenBackend.__new__(QwenBackend)  # no weights: _parse needs only the config
    b.model = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(patch_size=14)))
    inputs = {"image_grid_thw": torch.tensor([[1, 42, 58]])}
    img = Image.new("RGB", (1200, 900))
    raw = "REASON: no hat\nREGION: (406, 294, 609, 441)\nANSWER: YES"
    monkeypatch.setattr(config, "REGION_FRAME", "legacy")
    assert b._parse(raw, img, inputs).region == (406.0, 294.0, 609.0, 441.0)
    monkeypatch.setattr(config, "REGION_FRAME", "rescaled")
    assert b._parse(raw, img, inputs).region == pytest.approx((600.0, 450.0, 900.0, 675.0))


@pytest.mark.parametrize("script", ["02_baseline_inference", "04_descriptive_accuracy",
                                    "05_sparsity", "06_stability", "08_robustness"])
def test_every_region_stage_fingerprints_the_flag(script):
    text = (STUDY_ROOT / "scripts" / f"{script}.py").read_text(encoding="utf-8")
    assert "**config.region_frame_fingerprint()" in text
