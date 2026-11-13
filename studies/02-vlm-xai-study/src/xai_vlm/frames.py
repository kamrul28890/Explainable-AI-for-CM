"""Region coordinate frames.

Qwen2.5-VL states box coordinates in the frame of the image it was FED, after its
processor resized it to the pinned pixel budget -- a 1200x900 photo becomes
812x588 -- not in the frame of the original photograph. Found 2026-11-12: the
harness had read them as original-image pixels, so every model region sat about
two-thirds of the way towards the top-left corner. On the frozen correct-YES rows,
rescaling lifts mean IoU against the annotation from 0.058 to 0.389.

Every region a backend returns must be in ORIGINAL-image pixels, because every
metric (masking, attention mass, IoU, drift) is computed in that frame.
"""
from __future__ import annotations

from typing import Optional

from PIL import Image

from xai_vlm.parsing import Box, parse_response


def to_original(box: Box, image_size: tuple[int, int],
                input_size: tuple[int, int]) -> Optional[Box]:
    """Map a box from the model-input frame to the original image, clamped.
    Returns None if nothing of it is left inside the image."""
    W, H = image_size
    w, h = input_size
    sx, sy = W / w, H / h
    x0, y0, x1, y1 = box
    out = (max(0.0, x0 * sx), max(0.0, y0 * sy), min(float(W), x1 * sx), min(float(H), y1 * sy))
    return out if out[2] > out[0] and out[3] > out[1] else None


def input_size_from_grid(grid_thw, patch_size: int = 14) -> tuple[int, int]:
    """(width, height) the model saw, from image_grid_thw (t, h, w) in pre-merge patches."""
    _, gh, gw = (int(v) for v in grid_thw)
    return (gw * patch_size, gh * patch_size)


def region_from_reply(raw: str, image_size: tuple[int, int],
                      input_size: tuple[int, int]) -> Optional[Box]:
    """Parse REGION in the frame the model answered in, then map it to the original."""
    parsed = parse_response(raw, input_size)
    if parsed.region is None:
        return None
    return to_original(parsed.region, image_size, input_size)


# The units a backend's REGION can come in (roster design D1). Every one is mapped
# to ORIGINAL-image pixels before any metric sees it.
#   input_px  pixels of the resized image the model was fed (Qwen2.5-VL)
#   norm1000  0-1000 over the whole image (Qwen2-VL, Qwen3-VL, InternVL, GLM)
#   norm1     0-1 over the whole image (LLaVA family)
#   none      verdict-only model: the region is not scored, so it is dropped
CONVENTIONS = ("input_px", "norm1000", "norm1", "none")


def parse_in_convention(raw: str, convention: str, image_size: tuple[int, int],
                        input_size: Optional[tuple[int, int]] = None):
    """Parse one reply and return its REGION in original-image pixels.

    `input_size` is the model-input frame, required for input_px only. The input_px
    branch is exactly the rescaled Qwen2.5-VL path (region_from_reply)."""
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown box convention {convention!r}; choose from {CONVENTIONS}")
    if convention == "input_px":
        if input_size is None:
            raise ValueError("input_px needs the model-input frame (input_size)")
        parsed = parse_response(raw, input_size)
        if parsed.region is not None:
            parsed.region = to_original(parsed.region, image_size, input_size)
        return parsed
    if convention == "none":
        parsed = parse_response(raw, image_size)
        parsed.region = None
        return parsed
    if convention == "norm1000":
        frame = (1000, 1000)
        parsed = parse_response(raw, frame)
    else:
        # Not parse_response(raw, (1, 1)): its 0-1 heuristic would mark every
        # reply lenient. Parse unclamped, then clamp to the unit square here.
        frame = (1, 1)
        parsed = parse_response(raw, None)
        if parsed.region is not None:
            x0, y0, x1, y1 = parsed.region
            c = (max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1))
            if c != parsed.region:
                parsed.strict = False
            parsed.region = c if c[2] > c[0] and c[3] > c[1] else None
    if parsed.region is not None:
        parsed.region = to_original(parsed.region, image_size, frame)
    return parsed


def _fmt(box: Optional[Box]) -> str:
    return "" if box is None else str(tuple(round(float(v), 1) for v in box))


def rescale_box_text(text: str, image_size: tuple[int, int],
                     input_size: tuple[int, int]) -> str:
    """'(x0, y0, x1, y1)' in the model-input frame -> the same, in the original frame."""
    import ast
    text = (text or "").strip()
    if not text:
        return ""
    v = tuple(float(x) for x in ast.literal_eval(text))
    return _fmt(to_original(v, image_size, input_size))


class ModelInputFrame:
    """The input frame for an image size, from the backend's real preprocessing path
    (qwen_vl_utils.process_vision_info, then the image processor). Depends on the
    size alone, so it is computed once per size from a blank image."""

    def __init__(self, processor, patch_size: int = 14):
        self.processor = processor
        self.patch_size = patch_size
        self._cache: dict = {}

    def size_for(self, image_size: tuple[int, int]) -> tuple[int, int]:
        image_size = (int(image_size[0]), int(image_size[1]))
        if image_size not in self._cache:
            from qwen_vl_utils import process_vision_info
            img = Image.new("RGB", image_size)
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                  {"type": "text", "text": "x"}]}]
            image_inputs, _ = process_vision_info(msgs)
            out = self.processor.image_processor(images=image_inputs, return_tensors="pt")
            self._cache[image_size] = input_size_from_grid(out["image_grid_thw"][0].tolist(),
                                                           self.patch_size)
        return self._cache[image_size]
