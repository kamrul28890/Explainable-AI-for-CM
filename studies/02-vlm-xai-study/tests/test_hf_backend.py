"""The shared Hugging Face VLM backend (roster step 2), without weights.

What must hold before any new model is run:

* the frozen Qwen2.5-VL arm is untouched by default (XAI_BACKEND=legacy), and the
  generic backend reproduces it exactly where it can be checked on CPU;
* every coordinate convention maps a known box to the same original-image box;
* only the REGION line of the prompt varies between conventions;
* the token and attention arithmetic is the legacy arithmetic, factored out.
"""
import random
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from xai_vlm import config, frames, rules
from xai_vlm.config import MODEL_TIERS

torch = pytest.importorskip("torch")

IMG = Image.new("RGB", (1200, 900))
TRUE = (600.0, 450.0, 900.0, 675.0)      # the box every convention below describes


def _reply(region: str) -> str:
    return f"REASON: no hat\nREGION: {region}\nANSWER: YES"


# ------------------------------------------------------------------ conventions
@pytest.mark.parametrize("convention,region,input_size", [
    ("input_px", "(406, 294, 609, 441)", (812, 588)),     # Qwen2.5-VL, resized frame
    ("norm1000", "(500, 500, 750, 750)", None),           # Qwen2-VL, InternVL, GLM
    ("norm1", "(0.5, 0.5, 0.75, 0.75)", None),            # LLaVA family
])
def test_every_convention_maps_a_known_box_to_the_original_frame(convention, region, input_size):
    p = frames.parse_in_convention(_reply(region), convention, IMG.size, input_size)
    assert p.verdict == "YES"
    assert p.region == pytest.approx(TRUE, abs=0.75)


def test_input_px_reproduces_the_rescaled_qwen_path_exactly():
    raw = _reply("(406, 294, 609, 441)")
    want = frames.region_from_reply(raw, IMG.size, (812, 588))
    assert frames.parse_in_convention(raw, "input_px", IMG.size, (812, 588)).region == want


def test_norm1_is_strict_and_not_mistaken_for_the_pixel_heuristic():
    p = frames.parse_in_convention(_reply("(0.5, 0.5, 0.75, 0.75)"), "norm1", IMG.size, None)
    assert p.strict


def test_out_of_range_boxes_are_clamped_and_flagged():
    p = frames.parse_in_convention(_reply("(500, 500, 1200, 750)"), "norm1000", IMG.size, None)
    assert p.region[2] == pytest.approx(1200.0)
    assert not p.strict


def test_verdict_only_models_keep_the_verdict_and_drop_the_region():
    p = frames.parse_in_convention(_reply("(406, 294, 609, 441)"), "none", IMG.size, None)
    assert p.verdict == "YES" and p.region is None and p.reason == "no hat"


def test_an_unknown_convention_is_refused():
    with pytest.raises(ValueError):
        frames.parse_in_convention(_reply("NONE"), "points", IMG.size, None)


def test_input_px_without_the_input_frame_is_refused():
    with pytest.raises(ValueError):
        frames.parse_in_convention(_reply("(1, 2, 3, 4)"), "input_px", IMG.size, None)


# ----------------------------------------------------------------------- prompt
def test_the_default_prompt_is_byte_identical_to_the_frozen_one():
    for r in rules.RULE_IDS:
        frozen = f"{rules.RULES[r].question}\n\n{rules.RESPONSE_FORMAT}"
        assert rules.prompt_for(r) == frozen
        assert rules.prompt_for(r, "input_px") == frozen


@pytest.mark.parametrize("convention,needle", [("norm1000", "0 to 1000"), ("norm1", "0 to 1")])
def test_only_the_region_line_changes_between_conventions(convention, needle):
    base = rules.prompt_for("rule_1").splitlines()
    other = rules.prompt_for("rule_1", convention).splitlines()
    assert len(base) == len(other)
    diff = [(a, b) for a, b in zip(base, other) if a != b]
    assert len(diff) == 1 and diff[0][0].startswith("REGION:") and needle in diff[0][1]


def test_an_unknown_prompt_convention_is_refused():
    with pytest.raises(ValueError):
        rules.prompt_for("rule_1", "points")


# --------------------------------------------------------------- registry/flag
def test_every_tier_names_a_known_family_and_convention():
    from xai_vlm.backends.families import ADAPTERS
    for spec in MODEL_TIERS.values():
        assert spec.family in ADAPTERS, spec.name
        assert spec.box_convention in frames.CONVENTIONS, spec.name
        assert spec.env in ("A", "B"), spec.name


def test_frozen_tiers_keep_their_old_defaults():
    for t in ("qwen7b-4bit", "qwen7b-bf16"):
        s = MODEL_TIERS[t]
        assert (s.family, s.box_convention, s.env, s.legacy_region_frame) == \
            ("qwen2_5_vl", "input_px", "A", True)


@pytest.mark.parametrize("tier,repo", [("qwen3b-bf16", "Qwen/Qwen2.5-VL-3B-Instruct"),
                                       ("qwen32b-4bit", "Qwen/Qwen2.5-VL-32B-Instruct"),
                                       ("qwen32b-bf16", "Qwen/Qwen2.5-VL-32B-Instruct")])
def test_new_size_arms_are_registered_and_pinned(tier, repo):
    s = MODEL_TIERS[tier]
    assert s.repo_id == repo and not s.legacy_region_frame
    lock = (config.REPO_ROOT / "requirements" / "models.lock").read_text(encoding="utf-8")
    assert any(line.split()[:2] == [tier, repo] for line in lock.splitlines()
               if line and not line.startswith("#"))


def test_legacy_flag_keeps_the_frozen_class_for_frozen_tiers(monkeypatch):
    from xai_vlm.backends.hf import HFVLMBackend
    from xai_vlm.backends.load import backend_class
    from xai_vlm.backends.qwen import QwenBackend
    monkeypatch.setattr(config, "BACKEND", "legacy")
    assert backend_class("qwen7b-4bit") is QwenBackend
    assert backend_class("qwen7b-bf16") is QwenBackend
    # New tiers have no frozen output, so they always take the shared backend.
    assert backend_class("qwen3b-bf16") is HFVLMBackend
    monkeypatch.setattr(config, "BACKEND", "hf")
    assert backend_class("qwen7b-4bit") is HFVLMBackend


def test_backend_fingerprint_is_empty_only_for_the_frozen_path(monkeypatch):
    monkeypatch.setattr(config, "BACKEND", "legacy")
    assert config.backend_fingerprint("qwen7b-4bit") == {}
    assert config.backend_fingerprint("qwen3b-bf16") == {"backend": "hf",
                                                         "box_convention": "input_px"}
    monkeypatch.setattr(config, "BACKEND", "hf")
    assert config.backend_fingerprint("qwen7b-4bit")["backend"] == "hf"


def test_a_library_other_than_environment_As_is_fingerprinted(monkeypatch):
    monkeypatch.setattr(config, "BACKEND", "legacy")
    monkeypatch.setattr(config, "transformers_version", lambda: config.FROZEN_TRANSFORMERS)
    assert config.backend_fingerprint("qwen7b-bf16") == {}
    monkeypatch.setattr(config, "transformers_version", lambda: "5.20.2")
    assert config.backend_fingerprint("qwen7b-bf16") == {"transformers": "5.20.2"}
    assert config.backend_fingerprint("qwen3b-bf16")["transformers"] == "5.20.2"


def test_qwen_adapter_finds_its_modules_in_either_library_layout():
    from xai_vlm.backends.families import Qwen25VLAdapter
    from xai_vlm.config import MODEL_TIERS
    emb, vis = object(), object()
    a = Qwen25VLAdapter(MODEL_TIERS["qwen7b-bf16"])
    # transformers 4.49: model.visual, model.model.embed_tokens
    a.model = SimpleNamespace(visual=vis, get_input_embeddings=lambda: emb)
    assert a._visual() is vis and a._embed() is emb
    # transformers 5: the vision tower under model.model
    a.model = SimpleNamespace(model=SimpleNamespace(visual=vis), get_input_embeddings=lambda: emb)
    assert a._visual() is vis


def test_a_bad_backend_flag_is_refused(monkeypatch):
    monkeypatch.setenv("XAI_BACKEND", "nonsense")
    import importlib
    with pytest.raises(ValueError):
        importlib.reload(config)
    monkeypatch.delenv("XAI_BACKEND")
    importlib.reload(config)


@pytest.mark.parametrize("script", ["00_feasibility_gate", "02_baseline_inference",
                                    "04_descriptive_accuracy", "05_sparsity",
                                    "06_stability", "08_robustness"])
def test_every_model_stage_goes_through_the_factory(script):
    from xai_vlm.config import STUDY_ROOT
    text = (STUDY_ROOT / "scripts" / f"{script}.py").read_text(encoding="utf-8")
    assert "QwenBackend(" not in text and "load_backend(" in text
    if script != "00_feasibility_gate":
        assert "**config.backend_fingerprint(model)" in text


# ------------------------------------------------- legacy arithmetic, factored
def _legacy_keep(region, size, rows, cols, n):
    """The loop in QwenBackend._answer_token_ablated, verbatim."""
    w, h = size
    x0, y0, x1, y1 = region
    keep = torch.ones(n, dtype=torch.bool)
    for i in range(n):
        r, c = divmod(i, cols)
        px0, py0 = c * w / cols, r * h / rows
        px1, py1 = (c + 1) * w / cols, (r + 1) * h / rows
        if not (px1 <= x0 or px0 >= x1 or py1 <= y0 or py0 >= y1):
            keep[i] = False
    return keep


def test_token_cells_match_the_legacy_loop():
    from xai_vlm.backends.hf import cells_to_keep
    rng = random.Random(0)
    for _ in range(200):
        x0, y0 = rng.uniform(0, 1100), rng.uniform(0, 800)
        region = (x0, y0, x0 + rng.uniform(1, 400), y0 + rng.uniform(1, 400))
        rows, cols = rng.choice([(21, 29), (29, 21), (16, 16)])
        assert torch.equal(cells_to_keep(region, (1200, 900), rows, cols, rows * cols),
                           _legacy_keep(region, (1200, 900), rows, cols, rows * cols))


def test_attention_aggregation_matches_the_legacy_loop():
    from xai_vlm.backends.hf import aggregate_attention
    g = torch.Generator().manual_seed(0)
    prompt_len, total, rows, cols = 40, 50, 3, 4
    positions = torch.arange(10, 10 + rows * cols)
    layers = [torch.rand(1, 4, total, total, generator=g) for _ in range(3)]
    acc = None
    for layer in layers:                         # QwenBackend.attention, verbatim
        row = layer[0].float().mean(dim=0)[prompt_len:][:, positions].mean(dim=0)
        acc = row if acc is None else acc + row
    flat = (acc / 3).numpy().reshape(rows, cols)
    want = flat / flat.max()
    got = aggregate_attention(layers, prompt_len, positions, rows, cols)
    assert np.array_equal(got, want)


def test_attention_refuses_a_wrong_layout():
    from xai_vlm.backends.hf import aggregate_attention
    layers = [torch.rand(1, 2, 30, 30)]
    assert aggregate_attention(layers, 20, torch.arange(12), 5, 5) is None
    assert aggregate_attention([], 20, torch.arange(12), 3, 4) is None


# ------------------------------------------- Qwen2.5 adapter against legacy
def _qwen_pair():
    from xai_vlm.backends.hf import HFVLMBackend
    from xai_vlm.backends.qwen import QwenBackend
    cfg = SimpleNamespace(vision_config=SimpleNamespace(patch_size=14, spatial_merge_size=2),
                          image_token_id=151655)
    legacy = QwenBackend.__new__(QwenBackend)
    legacy.model = SimpleNamespace(config=cfg)
    new = HFVLMBackend.for_testing("qwen7b-4bit", model=SimpleNamespace(config=cfg))
    return legacy, new


@pytest.mark.parametrize("flag", ["legacy", "rescaled"])
def test_qwen_adapter_parses_exactly_as_the_frozen_backend(monkeypatch, flag):
    legacy, new = _qwen_pair()
    monkeypatch.setattr(config, "REGION_FRAME", flag)
    inputs = {"image_grid_thw": torch.tensor([[1, 42, 58]])}
    for region in ["(406, 294, 609, 441)", "NONE", "(0.1, 0.2, 0.3, 0.4)", "(900, 10, 5, 700)"]:
        raw = _reply(region)
        assert new._parse(raw, IMG, inputs) == legacy._parse(raw, IMG, inputs)


def test_new_qwen_tiers_are_always_read_in_the_input_frame(monkeypatch):
    from xai_vlm.backends.hf import HFVLMBackend
    cfg = SimpleNamespace(vision_config=SimpleNamespace(patch_size=14, spatial_merge_size=2))
    b = HFVLMBackend.for_testing("qwen3b-bf16", model=SimpleNamespace(config=cfg))
    monkeypatch.setattr(config, "REGION_FRAME", "legacy")
    p = b._parse(_reply("(406, 294, 609, 441)"), IMG, {"image_grid_thw": torch.tensor([[1, 42, 58]])})
    assert p.region == pytest.approx(TRUE, abs=0.75)


def test_qwen_adapter_grid_and_tokens_match_the_frozen_backend():
    legacy, new = _qwen_pair()
    ids = torch.tensor([[1, 2] + [151655] * 6 + [3]])
    inputs = {"image_grid_thw": torch.tensor([[1, 4, 6]]), "input_ids": ids}
    legacy.processor = SimpleNamespace(tokenizer=None)
    assert new.adapter.grid(inputs) == legacy._grid_from_inputs(inputs) == (2, 3)
    assert torch.equal(new.adapter.image_token_positions(inputs),
                       legacy.image_token_positions(inputs))
