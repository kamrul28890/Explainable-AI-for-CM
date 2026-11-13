"""Per-family adapters for the shared Hugging Face VLM backend (roster step 2).

An adapter holds everything that differs between model families and nothing else:

* how the model and processor load;
* how one image and one prompt become model inputs;
* the frame a box in pixels is stated in (input_px models only);
* where the image tokens sit in the sequence, and the grid they form;
* how vision features are merged into the token embeddings (token ablation).

Everything else -- generation, timing, parsing, masking, ablation, attention
aggregation -- lives once in backends.hf.HFVLMBackend, so a difference between two
models is never a difference in the measuring instrument.

A family whose image tokens do not form one row-major grid over the whole image
(tiled or cross-attention models) returns None from `grid()`. Attention and token
ablation are then reported as not applicable for it, never guessed (roster D4).
"""
from __future__ import annotations

from typing import Optional

from PIL import Image

from xai_vlm.config import MAX_PIXELS, MIN_PIXELS, ModelSpec


def dtype_kwarg(dtype) -> dict:
    """from_pretrained's dtype argument: torch_dtype in transformers 4, dtype in 5."""
    from xai_vlm.config import transformers_version
    major = int((transformers_version() or "4").split(".")[0])
    return {"torch_dtype": dtype} if major < 5 else {"dtype": dtype}


def fit_pixel_budget(image: Image.Image, max_pixels: int = MAX_PIXELS) -> Image.Image:
    """Shrink, keeping the aspect ratio, so width x height <= max_pixels (roster D3:
    every model receives the same pixels). Never enlarges."""
    w, h = image.size
    if w * h <= max_pixels:
        return image
    s = (max_pixels / (w * h)) ** 0.5
    size = (max(1, int(w * s)), max(1, int(h * s)))
    while size[0] * size[1] > max_pixels:          # int() can still round over
        size = (size[0] - 1, size[1])
    return image.resize(size, Image.BICUBIC)


class Adapter:
    """The interface. One instance per loaded backend; holds model and processor."""

    family = ""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.model = None
        self.processor = None

    # -- loading
    def load(self, source: str, *, dtype, quantization_config, attn_implementation):
        raise NotImplementedError

    # -- inputs
    def prepare_image(self, image: Image.Image) -> Image.Image:
        """The image as handed to the processor: the shared pixel budget (D3)."""
        return fit_pixel_budget(image)

    def build_inputs(self, image: Image.Image, prompt: str):
        raise NotImplementedError

    def input_size(self, inputs, image: Image.Image) -> Optional[tuple[int, int]]:
        """(width, height) of the frame an input_px model states boxes in."""
        return None

    # -- the image-token grid
    def image_token_positions(self, inputs):
        raise NotImplementedError

    def grid(self, inputs) -> Optional[tuple[int, int]]:
        """(rows, cols) of the image tokens, row-major over the whole image, or None."""
        return None

    def patch_grid(self, image: Optional[Image.Image]) -> tuple[int, int]:
        raise NotImplementedError

    # -- token ablation
    def multimodal_embeds(self, inputs):
        """Token embeddings with the vision features already merged in, or None."""
        return None

    def mask_block(self) -> int:
        """Pixel block a black mask is snapped outward to."""
        return 28


class Qwen25VLAdapter(Adapter):
    """Qwen2.5-VL: dynamic resolution, 14 px patches merged 2x2, boxes in the
    resized input frame. Every method is the frozen QwenBackend's, moved here."""

    family = "qwen2_5_vl"

    def prepare_image(self, image):
        # Identity: the processor applies the same budget itself (min/max_pixels),
        # and the frozen backend hands it the original image.
        return image

    def load(self, source, *, dtype, quantization_config, attn_implementation):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            source,
            quantization_config=quantization_config,
            **dtype_kwarg(dtype),
            device_map="auto",
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        # min/max pixels pinned here and nowhere else (config.MAX_PIXELS).
        self.processor = AutoProcessor.from_pretrained(
            source, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

    def build_inputs(self, image, prompt):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        return self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

    def _patch(self) -> int:
        return getattr(self.model.config.vision_config, "patch_size", 14)

    def _merge(self) -> int:
        return getattr(self.model.config.vision_config, "spatial_merge_size", 2)

    def input_size(self, inputs, image):
        from xai_vlm.frames import input_size_from_grid
        return input_size_from_grid(inputs["image_grid_thw"][0].tolist(), self._patch())

    def image_token_positions(self, inputs):
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        ids = inputs["input_ids"][0]
        return (ids == image_token_id).nonzero(as_tuple=True)[0]

    def grid(self, inputs):
        # image_grid_thw is (t, h, w) in PRE-merge patches; the language model
        # sees them merged 2x2.
        t, h, w = inputs["image_grid_thw"][0].tolist()
        m = self._merge()
        return (h // m, w // m)

    def patch_grid(self, image):
        if image is not None:
            return self.grid(self.processor.image_processor(images=[image],
                                                            return_tensors="pt"))
        block = self._patch() * self._merge()
        side = int((MAX_PIXELS ** 0.5) // block)
        return (side, side)

    def multimodal_embeds(self, inputs):
        # See QwenBackend._multimodal_embeds: the placeholder embeddings must be
        # replaced by the vision features, under inference mode (memory).
        import torch
        with torch.inference_mode():
            embeds = self._embed()(inputs["input_ids"])
            image_embeds = self._visual()(
                inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]
            ).to(embeds.device, embeds.dtype)
            image_token_id = getattr(self.model.config, "image_token_id")
            mask = (inputs["input_ids"] == image_token_id).unsqueeze(-1).expand_as(embeds)
            return embeds.masked_scatter(mask, image_embeds)

    def mask_block(self):
        return self._patch() * self._merge()

    # transformers 4.49 keeps the vision tower at model.visual; 5.x moves it to
    # model.model.visual. get_input_embeddings() is model.model.embed_tokens in
    # 4.49 -- the same module the frozen backend calls -- and the right one in 5.x.
    def _visual(self):
        v = getattr(self.model, "visual", None)
        return v if v is not None else self.model.model.visual

    def _embed(self):
        return self.model.get_input_embeddings()


class MllamaAdapter(Adapter):
    """Llama 3.2 Vision. The image enters through cross-attention layers, not as
    tokens in the sequence, so there is no image-token grid: attention and token
    ablation are not applicable, and the tier is verdict-only (roster D4)."""

    family = "mllama"

    def load(self, source, *, dtype, quantization_config, attn_implementation):
        from transformers import AutoProcessor, MllamaForConditionalGeneration
        self.model = MllamaForConditionalGeneration.from_pretrained(
            source,
            quantization_config=quantization_config,
            **dtype_kwarg(dtype),
            device_map="auto",
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(source)

    def build_inputs(self, image, prompt):
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        # The chat template already carries <|begin_of_text|>.
        return self.processor(images=image, text=text, add_special_tokens=False,
                              return_tensors="pt").to(self.model.device)

    def image_token_positions(self, inputs):
        image_token_id = getattr(self.model.config, "image_token_index", None)
        ids = inputs["input_ids"][0]
        return (ids == image_token_id).nonzero(as_tuple=True)[0]

    def patch_grid(self, image):
        return (0, 0)      # no image tokens in the sequence


ADAPTERS: dict[str, type[Adapter]] = {
    Qwen25VLAdapter.family: Qwen25VLAdapter,
    MllamaAdapter.family: MllamaAdapter,
}
