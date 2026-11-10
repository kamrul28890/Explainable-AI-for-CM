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

        local = MODEL_DIR / self.spec.weights_dir
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

    def _merge_size(self) -> int:
        return getattr(self.model.config.vision_config, "spatial_merge_size", 2)

    def _grid_from_inputs(self, inputs) -> tuple[int, int]:
        """(rows, cols) of the merged grid for inputs already built.

        image_grid_thw is (t, h, w) in PRE-merge patches; the language model sees
        them merged 2x2. This is the authoritative grid -- everything mapping
        tokens to pixels must come through here.
        """
        t, h, w = inputs["image_grid_thw"][0].tolist()
        m = self._merge_size()
        return (h // m, w // m)

    def patch_grid(self, image: Optional[Image.Image] = None) -> tuple[int, int]:
        """(rows, cols) of the merged patch grid.

        Exact only when `image` is given. Qwen2.5-VL uses dynamic resolution, so
        the grid follows the image's aspect ratio: a 1200x900 photo at the pinned
        budget is 21x29, not square. Deriving a square grid from the pixel budget
        gives a token COUNT within a fraction of a percent -- which is exactly why
        it is dangerous, since nothing crashes -- but the wrong number of columns,
        and divmod(i, cols) then walks diagonally across the image.
        """
        if image is not None:
            return self._grid_from_inputs(
                self.processor.image_processor(images=[image], return_tensors="pt")
            )
        # Nominal fallback. Approximate by construction: never map tokens to
        # pixels with this.
        block = getattr(self.model.config.vision_config, "patch_size", 14) * self._merge_size()
        side = int((MAX_PIXELS ** 0.5) // block)
        return (side, side)

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
        ids = inputs["input_ids"][0]
        return (ids == image_token_id).nonzero(as_tuple=True)[0]

    def _multimodal_embeds(self, inputs) -> "torch.Tensor":
        """Token embeddings with vision features scattered into the image slots.

        This replicates the merge inside Qwen2_5_VLForConditionalGeneration.forward:
        embed_tokens() returns a PLACEHOLDER vector for every image token, and the
        vision-tower output is scattered over those positions afterwards.

        Building ablation inputs from embed_tokens() alone -- the obvious shortcut --
        produces a sequence in which the model sees no image whatsoever. Every
        ablation would then flip the answer regardless of which region was removed,
        and Descriptive Accuracy would report near-perfect faithfulness while
        measuring nothing at all.
        """
        # INFERENCE MODE IS LOAD-BEARING HERE, not a tidy-up.
        #
        # self.model.visual() is a full vision-tower forward. Outside inference
        # mode it builds an autograd graph and retains every intermediate
        # activation for a backward pass that never comes. Measured on one
        # 1200x900 image: peak allocation 22.65 GB against 6.97 GB for the same
        # model answering normally -- roughly 16 GB of retained activations on an
        # 8 GB card, which Windows silently serves from shared system memory at
        # about a 17x speed penalty (54.5 s per call against 3.1 s).
        #
        # It does not change the numbers. inference_mode disables gradient
        # tracking and version counting; the arithmetic is identical, so results
        # computed before and after this line are directly comparable.
        with torch.inference_mode():
            embeds = self.model.model.embed_tokens(inputs["input_ids"])
            image_embeds = self.model.visual(
                inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]
            ).to(embeds.device, embeds.dtype)
            image_token_id = getattr(self.model.config, "image_token_id")
            mask = (inputs["input_ids"] == image_token_id).unsqueeze(-1).expand_as(embeds)
            return embeds.masked_scatter(mask, image_embeds)

    def attention(self, image: Image.Image, rule_id: RuleId, **kw
                  ) -> Optional[np.ndarray]:
        """Mean attention from the generated answer back to image patches.

        Generation runs on the normal (sdpa) path, and attention is then read from
        a SINGLE teacher-forced forward pass over the produced sequence. The eager
        kernel that attention reading requires corrupts generation (see
        config.COMPUTE_DTYPE), so nothing is ever generated under it. This also
        measures attention for the answer actually given rather than for a separate
        decode that might have gone somewhere else.

        Returns None rather than a fabricated map if extraction is unavailable --
        Sparsity is then reported only for backends where it genuinely works.
        """
        prompt = prompt_for(rule_id)
        inputs = self._build_inputs(image, prompt)
        positions = self.image_token_positions(inputs)
        if len(positions) == 0:
            return None

        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **{**GENERATION, **kw})
        if generated.shape[1] <= prompt_len:
            return None

        rest = {k: v for k, v in inputs.items()
                if k not in ("input_ids", "attention_mask")}
        try:
            with torch.inference_mode():
                out = self.model(
                    input_ids=generated,
                    attention_mask=torch.ones_like(generated),
                    output_attentions=True,
                    **rest,
                )
        except (RuntimeError, TypeError):
            return None

        layers = getattr(out, "attentions", None)
        if not layers:
            return None

        # Each layer is (batch, heads, query, key). Average over heads, keep the
        # generated rows as queries and the image tokens as keys, then average over
        # generated tokens and over layers.
        acc, n = None, 0
        for layer in layers:
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
        rows, cols = self._grid_from_inputs(inputs)
        if flat.size != rows * cols:
            # Refuse to guess a layout rather than reshape into a wrong aspect.
            return None
        grid = flat.reshape(rows, cols)
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

        # The grid comes from image_grid_thw, never from the pixel budget: under
        # dynamic resolution it is not square, and a wrong `cols` would make divmod
        # walk diagonally across the image.
        rows, cols = self._grid_from_inputs(inputs)
        w, h = image.size
        x0, y0, x1, y1 = region

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

        embeds = self._multimodal_embeds(inputs)
        mask = torch.ones(inputs["input_ids"].shape[1], dtype=torch.bool,
                          device=embeds.device)
        mask[drop] = False
        kept_embeds = embeds[:, mask, :]
        kept_attention = inputs["attention_mask"][:, mask]

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

    @property
    def device_name(self) -> str:
        """The GPU actually computing, e.g. 'NVIDIA A40'. Recorded in every
        stage fingerprint: replies are not reproducible across devices."""
        dev = self.model.device
        if dev.type == "cuda":
            return torch.cuda.get_device_name(dev)
        return dev.type

    def peak_memory_gb(self) -> float:
        if not torch.cuda.is_available():
            return float("nan")
        return torch.cuda.max_memory_allocated() / 1e9
