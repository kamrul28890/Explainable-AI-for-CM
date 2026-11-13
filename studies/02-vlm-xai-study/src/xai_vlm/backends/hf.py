"""The shared Hugging Face VLM backend (roster step 2).

One class runs every roster model. What differs between families is in
backends.families; everything here is the frozen QwenBackend's logic, written once:
generation and timing, reply parsing in the model's coordinate convention, black
masking, token ablation, and attention read from one teacher-forced forward pass.

Parity with the frozen backend is a tested property, not a hope: the CPU tests in
tests/test_hf_backend.py check the parsing, token-cell and attention arithmetic
against QwenBackend's, and scripts/20_backend_parity.py checks replies, regions,
ablated replies and attention grids on the GPU.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
from PIL import Image

from xai_vlm import config as _config
from xai_vlm.backends.base import Answer, BackendUnavailable
from xai_vlm.backends.families import ADAPTERS, Adapter
from xai_vlm.config import (ATTENTION_IMPL, COMPUTE_DTYPE, GENERATION, MODEL_DIR,
                            ModelSpec, resolve_tier)
from xai_vlm.frames import parse_in_convention
from xai_vlm.parsing import Box, parse_response
from xai_vlm.rules import RuleId, prompt_for


def cells_to_keep(region: Box, image_size: tuple[int, int], rows: int, cols: int,
                  n: int) -> "torch.Tensor":
    """Which of `n` row-major image tokens survive removing `region`.

    A token is dropped when its cell, laid over the ORIGINAL image, overlaps the
    region at all. Identical to the loop in QwenBackend._answer_token_ablated."""
    w, h = image_size
    x0, y0, x1, y1 = region
    keep = torch.ones(n, dtype=torch.bool)
    for i in range(n):
        r, c = divmod(i, cols)
        px0, py0 = c * w / cols, r * h / rows
        px1, py1 = (c + 1) * w / cols, (r + 1) * h / rows
        if not (px1 <= x0 or px0 >= x1 or py1 <= y0 or py0 >= y1):
            keep[i] = False
    return keep


def aggregate_attention(layers, prompt_len: int, positions, rows: int,
                        cols: int) -> Optional[np.ndarray]:
    """Mean attention from generated tokens to image tokens, as a peak-normalised
    (rows, cols) grid. Heads, generated tokens and layers are averaged, in that
    order, exactly as QwenBackend.attention does. None if the layout does not fit."""
    acc, n = None, 0
    for layer in layers or ():
        a = layer[0].float().mean(dim=0)
        sel = a[prompt_len:][:, positions]
        if sel.numel() == 0:
            continue
        row = sel.mean(dim=0)
        acc = row if acc is None else acc + row
        n += 1
    if acc is None or n == 0:
        return None
    flat = (acc / n).cpu().numpy()
    if flat.size != rows * cols:
        return None    # refuse to guess a layout rather than reshape wrongly
    grid = flat.reshape(rows, cols)
    peak = grid.max()
    return grid / peak if peak > 0 else grid


class HFVLMBackend:
    """Any roster model, behind the Backend protocol."""

    def __init__(self, tier: str, device: str = "cuda",
                 attn_implementation: str = ATTENTION_IMPL):
        self.spec: ModelSpec = resolve_tier(tier)
        self.name = self.spec.name
        self.device = device
        if self.spec.family not in ADAPTERS:
            raise BackendUnavailable(f"no adapter for family {self.spec.family!r}")
        if device == "cuda" and not torch.cuda.is_available():
            raise BackendUnavailable(
                "CUDA is not available. This backend requires the compute machine; "
                "bitsandbytes has no Apple Silicon build. See docs/setup/SETUP.md.")

        from transformers import BitsAndBytesConfig

        local = MODEL_DIR / self.spec.weights_dir
        source = str(local) if local.exists() else self.spec.repo_id
        dtype = getattr(torch, COMPUTE_DTYPE)
        quant = None
        if self.spec.quantization == "4bit":
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                                       bnb_4bit_quant_type="nf4",
                                       bnb_4bit_use_double_quant=True)
        elif self.spec.quantization == "8bit":
            quant = BitsAndBytesConfig(load_in_8bit=True)

        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.adapter: Adapter = ADAPTERS[self.spec.family](self.spec)
        self.adapter.load(source, dtype=dtype, quantization_config=quant,
                          attn_implementation=attn_implementation)

    @classmethod
    def for_testing(cls, tier: str, model=None, processor=None) -> "HFVLMBackend":
        """A backend with no weights, for CPU tests of the logic around the model."""
        b = cls.__new__(cls)
        b.spec = resolve_tier(tier)
        b.name, b.device = b.spec.name, "cpu"
        b.adapter = ADAPTERS[b.spec.family](b.spec)
        b.adapter.model, b.adapter.processor = model, processor
        return b

    @property
    def model(self):
        return self.adapter.model

    @property
    def processor(self):
        return self.adapter.processor

    # ---------------------------------------------------------------- helpers
    def _prompt(self, rule_id: RuleId) -> str:
        return prompt_for(rule_id, self.spec.box_convention)

    def _decode_new_tokens(self, generated, inputs) -> str:
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def _parse(self, raw: str, image: Image.Image, inputs):
        """REGION in original-image pixels. The frozen tiers follow
        XAI_REGION_FRAME (legacy reads boxes as original pixels, the frozen
        behaviour); every other tier is always mapped from its own convention."""
        conv = self.spec.box_convention
        if self.spec.legacy_region_frame and _config.REGION_FRAME != "rescaled":
            return parse_response(raw, image.size)
        insize = self.adapter.input_size(inputs, image) if conv == "input_px" else None
        return parse_in_convention(raw, conv, image.size, insize)

    def _token_layout(self, inputs):
        positions = self.adapter.image_token_positions(inputs)
        grid = self.adapter.grid(inputs)
        return positions, grid

    # -------------------------------------------------------------------- API
    def answer(self, image: Image.Image, rule_id: RuleId, **kw) -> Answer:
        inputs = self.adapter.build_inputs(image, self._prompt(rule_id))
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
        parsed = self._parse(raw, image, inputs)
        return Answer.from_parsed(parsed, raw_text=raw, latency_ms=latency_ms)

    def answer_masked(self, image: Image.Image, rule_id: RuleId, region: Box,
                      method: str = "black", **kw) -> Answer:
        from xai_pilot.regions import mask_region

        from xai_vlm.masking import snap_to_patch_grid

        snapped = snap_to_patch_grid(region, image.size, block=self.adapter.mask_block())
        if method in {"black", "blur", "inpaint"}:
            masked = mask_region(image, snapped, mode=method)
            out = self.answer(masked, rule_id, **kw)
            out.intervention = method
            return out
        if method == "tokens":
            return self._answer_token_ablated(image, rule_id, snapped, **kw)
        raise ValueError(f"unknown masking method {method!r}")

    def patch_grid(self, image: Optional[Image.Image] = None) -> tuple[int, int]:
        return self.adapter.patch_grid(image)

    def attention(self, image: Image.Image, rule_id: RuleId, **kw) -> Optional[np.ndarray]:
        """Attention from the generated answer to the image tokens, from ONE
        teacher-forced forward pass over the produced sequence (generation itself
        never runs under the eager kernel). None where the family has no
        row-major image-token grid."""
        inputs = self.adapter.build_inputs(image, self._prompt(rule_id))
        positions, grid = self._token_layout(inputs)
        if grid is None or len(positions) == 0:
            return None
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **{**GENERATION, **kw})
        if generated.shape[1] <= prompt_len:
            return None
        rest = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        try:
            with torch.inference_mode():
                out = self.model(input_ids=generated,
                                 attention_mask=torch.ones_like(generated),
                                 output_attentions=True, **rest)
        except (RuntimeError, TypeError):
            return None
        return aggregate_attention(getattr(out, "attentions", None), prompt_len,
                                   positions, *grid)

    def _answer_token_ablated(self, image: Image.Image, rule_id: RuleId,
                              region: Box, **kw) -> Answer:
        inputs = self.adapter.build_inputs(image, self._prompt(rule_id))
        positions, grid = self._token_layout(inputs)
        if grid is None or len(positions) == 0:
            raise BackendUnavailable(f"token ablation not applicable to {self.name}: "
                                     "no row-major image-token grid")
        rows, cols = grid
        keep = cells_to_keep(region, image.size, rows, cols, len(positions))
        drop = positions[~keep.to(positions.device)]
        if len(drop) == 0:
            out = self.answer(image, rule_id, **kw)
            out.intervention = "tokens"
            return out

        embeds = self.adapter.multimodal_embeds(inputs)
        if embeds is None:
            raise BackendUnavailable(f"token ablation not applicable to {self.name}: "
                                     "no vision-feature merge")
        mask = torch.ones(inputs["input_ids"].shape[1], dtype=torch.bool,
                          device=embeds.device)
        mask[drop] = False
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(inputs_embeds=embeds[:, mask, :],
                                      attention_mask=inputs["attention_mask"][:, mask],
                                      **{**GENERATION, **kw})
        latency_ms = (time.perf_counter() - t0) * 1000.0
        # With inputs_embeds there is no prompt prefix in the output to strip.
        raw = self.processor.batch_decode(
            out, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        parsed = self._parse(raw, image, inputs)
        return Answer.from_parsed(parsed, raw_text=raw, latency_ms=latency_ms,
                                  intervention="tokens")

    @property
    def device_name(self) -> str:
        dev = self.model.device
        if dev.type == "cuda":
            return torch.cuda.get_device_name(dev)
        return dev.type

    def peak_memory_gb(self) -> float:
        if not torch.cuda.is_available():
            return float("nan")
        return torch.cuda.max_memory_allocated() / 1e9
