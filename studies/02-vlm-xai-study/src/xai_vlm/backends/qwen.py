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
from xai_vlm.config import (ATTENTION_IMPL, COMPUTE_DTYPE, GENERATION,
                            MAX_PIXELS, MIN_PIXELS, MODEL_DIR, ModelSpec,
                            resolve_tier)
from xai_vlm.parsing import Box, parse_response
from xai_vlm.rules import RuleId, prompt_for


class QwenBackend:
    """Qwen2.5-VL wrapped to satisfy the Backend protocol."""

    def __init__(self, tier: str = "qwen7b-4bit", device: str = "cuda",
                 attn_implementation: str = ATTENTION_IMPL):
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

        # bfloat16, not float16: see config.COMPUTE_DTYPE. fp16 overflows on this
        # model and generation collapses to empty output.
        dtype = getattr(torch, COMPUTE_DTYPE)

        quant = None
        if self.spec.quantization == "4bit":
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.spec.quantization == "8bit":
            quant = BitsAndBytesConfig(load_in_8bit=True)

        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            source,
            quantization_config=quant,
            torch_dtype=dtype,
            device_map="auto",
            # sdpa, NOT eager: the eager kernel injects spurious tokens into
            # generation. Attention is read from a separate forward pass instead.
            attn_implementation=attn_implementation,
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
