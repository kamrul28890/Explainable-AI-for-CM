"""Interventions that remove evidence from an image.

A black rectangle is something no real photograph contains, so the model may react
to the rectangle rather than to the missing evidence -- a "flip" then means "this
image looks broken" rather than "the hard hat is gone". Aligning masks to the patch
grid removes one source of that noise; token ablation (in the backend) removes it
entirely by never modifying pixels at all.
"""

from __future__ import annotations

import math

from xai_vlm.parsing import Box


def snap_to_patch_grid(box: Box, image_size: tuple[int, int], block: int = 28) -> Box:
    """Expand `box` outward so it covers whole patch blocks.

    Qwen2.5-VL sees 14px patches merged 2x2 -> 28px blocks. A mask that cuts a
    block in half leaves a partial signal in it, so masks are always expanded
    (never shrunk) to block boundaries, then clamped to the image.
    """
    w, h = image_size
    x0, y0, x1, y1 = box

    sx0 = max(0, int(math.floor(x0 / block) * block))
    sy0 = max(0, int(math.floor(y0 / block) * block))
    sx1 = min(w, int(math.ceil(x1 / block) * block))
    sy1 = min(h, int(math.ceil(y1 / block) * block))

    # Guarantee at least one whole block, even for a degenerate input box.
    if sx1 <= sx0:
        sx0 = max(0, min(sx0, w - block))
        sx1 = min(w, sx0 + block)
    if sy1 <= sy0:
        sy0 = max(0, min(sy0, h - block))
        sy1 = min(h, sy0 + block)

    return (float(sx0), float(sy0), float(sx1), float(sy1))
