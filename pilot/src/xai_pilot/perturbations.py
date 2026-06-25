"""Image perturbations for the Day 9 robustness test (Level 1) and the stretch patch test.

Level 2 (reworded prompts) needs no image perturbation at all -- it reruns
answer_rule with phrasing_index=1 on the unperturbed image, using the
second phrasing already enumerated per rule in prompts.RULE_QUERIES.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

Box = tuple[float, float, float, float]


def blur(image: Image.Image, ksize: float = 8.0) -> Image.Image:
    """Gaussian blur with the given radius (PIL has no integer kernel size, just radius)."""
    return image.filter(ImageFilter.GaussianBlur(radius=ksize))


def low_light(image: Image.Image, gamma: float = 2.5) -> Image.Image:
    """Darken via gamma correction: out = 255 * (in / 255) ** gamma. gamma > 1 darkens."""
    # Normalize before exponentiation and convert back to uint8 only after the
    # transform to avoid integer truncation during gamma correction.
    arr = np.asarray(image.convert("RGB")).astype(float) / 255.0
    arr = arr**gamma
    return Image.fromarray((arr * 255).astype(np.uint8))


def occlude(image: Image.Image, frac: float = 0.2) -> Image.Image:
    """Black out a centered square covering `frac` of the image's area."""
    out = image.convert("RGB").copy()
    width, height = out.size
    # Area = side^2, so the square-root converts the requested image-area
    # fraction into a pixel side length for arbitrary aspect ratios.
    side = int(round((frac * width * height) ** 0.5))
    x0, y0 = (width - side) // 2, (height - side) // 2
    patch = Image.new("RGB", (side, side), (0, 0, 0))
    out.paste(patch, (x0, y0))
    return out


def contrast_shift(image: Image.Image, factor: float = 0.3) -> Image.Image:
    """Scale contrast by `factor` (1.0 = unchanged, < 1 flattens, > 1 exaggerates)."""
    return ImageEnhance.Contrast(image.convert("RGB")).enhance(factor)


def paste_patch(image: Image.Image, box: Box, patch: Image.Image) -> Image.Image:
    """Paste `patch` (resized to fit) into `box` (x0, y0, x1, y1) on a copy of image."""
    out = image.convert("RGB").copy()
    # Round and clip external floating-point coordinates before resizing. An
    # invalid or fully out-of-frame box is treated as a no-op.
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, out.width), min(y1, out.height)
    if x1 <= x0 or y1 <= y0:
        return out
    resized = patch.convert("RGB").resize((x1 - x0, y1 - y0))
    out.paste(resized, (x0, y0))
    return out
