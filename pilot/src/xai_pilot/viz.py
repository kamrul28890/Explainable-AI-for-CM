"""Saving visual review artifacts: boxes/heatmaps drawn on top of pilot images."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def overlay_boxes(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    labels: list[str] | None = None,
    color: str = "red",
) -> Image.Image:
    """Draw absolute-pixel boxes (x0, y0, x1, y1) on a copy of image."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    labels = labels or [""] * len(boxes)
    # `zip` intentionally stops at the shorter sequence. Callers normally
    # provide one label per box; omitted labels are expanded above.
    for box, label in zip(boxes, labels):
        draw.rectangle(box, outline=color, width=3)
        if label:
            draw.text((box[0] + 2, max(box[1] - 12, 0)), label, fill=color)
    return out


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5, gamma: float = 1.0) -> Image.Image:
    """Upsample a low-res (h, w) heatmap to image size and blend it on as red.

    `gamma` < 1 stretches contrast among low-to-mid values before blending --
    useful since raw cross-attention is often fairly diffuse (most mass
    spread thinly across many cells), which a linear blend renders as a
    near-uniform tint that hides where the relative peaks actually are.
    """
    stretched = np.clip(heatmap, 0, 1) ** gamma
    heat_img = Image.fromarray((stretched * 255).astype(np.uint8)).resize(image.size, resample=Image.BILINEAR)
    heat_arr = np.asarray(heat_img).astype(float) / 255.0

    base = np.asarray(image.convert("RGB")).astype(float)
    red = np.zeros_like(base)
    red[..., 0] = 255.0

    # Convert the scalar heat intensity to a broadcastable RGB blend weight.
    # Cold pixels retain the source image; hot pixels move toward pure red.
    weight = (alpha * heat_arr)[..., None]
    blended = base * (1 - weight) + red * weight
    return Image.fromarray(blended.astype(np.uint8))


def save_figure(image: Image.Image, path: Path) -> None:
    """Create parent directories and persist a PIL image review artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
