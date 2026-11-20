"""Stage 8 image degradations, at the strengths approved on 2026-11-10.

Strength is a free parameter: too mild and everything survives, too harsh and
nothing does. The Plan 3c spec therefore fixes each one BEFORE the run, tied to a
physical meaning rather than to the spread it produces, and these values were
signed off by the user before Stage 8 was implemented. Changing one invalidates
every Stage 8 row (they are in the fingerprint), and must be argued, not tuned.

| name        | strength                                   | stands for                   |
|-------------|--------------------------------------------|------------------------------|
| blur        | Gaussian, sigma = 1% of the long side      | shake, distance, poor optics |
| low_light   | -2 EV: linear intensity x 0.25             | dusk, shadowed site          |
| occlusion   | mean-grey block over the central 50% x 50% | obstruction                  |
| contrast    | pixel = mean + 0.4 (pixel - mean)          | haze, glare, cheap sensor    |
| desaturate  | ITU-R 601-2 luma, replicated to RGB        | reliance on hue (hi-vis)     |

Each is applied alone, at the image's original resolution, before the model's
processor resizes it.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageStat

PERTURBATIONS = ("blur", "low_light", "occlusion", "contrast", "desaturate")

STRENGTHS = {
    "blur_sigma_frac_long_side": 0.01,
    "low_light_ev": -2.0,
    "occlusion_frac_w": 0.5, "occlusion_frac_h": 0.5,
    "contrast_factor": 0.4,
    "desaturate": "ITU-R 601-2 luma",
}

_fw, _fh = STRENGTHS["occlusion_frac_w"], STRENGTHS["occlusion_frac_h"]
# The block in normalised coordinates -- the frame the dataset's boxes use.
OCCLUDER_NORM = ((1 - _fw) / 2, (1 - _fh) / 2, (1 + _fw) / 2, (1 + _fh) / 2)


def blur_sigma(size: tuple[int, int]) -> float:
    return STRENGTHS["blur_sigma_frac_long_side"] * max(size)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _blur(img: Image.Image) -> Image.Image:
    # PIL's GaussianBlur radius IS the standard deviation.
    return img.filter(ImageFilter.GaussianBlur(radius=blur_sigma(img.size)))


def _low_light(img: Image.Image) -> Image.Image:
    """Exposure is multiplicative in LINEAR light; sRGB values are not linear."""
    lin = _srgb_to_linear(np.asarray(img, dtype=np.float64) / 255.0)
    lin *= 2.0 ** STRENGTHS["low_light_ev"]
    out = _linear_to_srgb(np.clip(lin, 0.0, 1.0)) * 255.0
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")


def _occlusion(img: Image.Image) -> Image.Image:
    """Neutral grey at the image's own mean luminance, so the block is not a
    brightness cue (a black or white block would be a second perturbation)."""
    grey = int(round(ImageStat.Stat(img.convert("L")).mean[0]))
    w, h = img.size
    x0, y0, x1, y1 = OCCLUDER_NORM
    out = img.copy()
    out.paste((grey, grey, grey), (round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
    return out


def _contrast(img: Image.Image) -> Image.Image:
    # ImageEnhance.Contrast blends toward a flat grey at the mean luminance:
    # exactly mean + factor * (pixel - mean).
    return ImageEnhance.Contrast(img).enhance(STRENGTHS["contrast_factor"])


def _desaturate(img: Image.Image) -> Image.Image:
    return img.convert("L").convert("RGB")


_FUNCS = {"blur": _blur, "low_light": _low_light, "occlusion": _occlusion,
          "contrast": _contrast, "desaturate": _desaturate}


def apply(name: str, img: Image.Image) -> Image.Image:
    """A NEW perturbed RGB image; the input is never modified."""
    if name not in _FUNCS:
        raise ValueError(f"unknown perturbation {name!r}; choose from {PERTURBATIONS}")
    return _FUNCS[name](img.convert("RGB"))


def occluder_coverage(boxes_norm) -> float | None:
    """Share of the annotated hazard hidden by the occluder, for the most-hidden box.

    When the block covers the hazard itself, a changed verdict is the CORRECT
    response, not fragility -- so occlusion survival is reported separately for
    rows where this is below and above 10%. None when there is no annotation box
    (compliant images, negative controls)."""
    if not boxes_norm:
        return None
    ox0, oy0, ox1, oy1 = OCCLUDER_NORM
    best = 0.0
    for b in boxes_norm:
        x0, y0, x1, y1 = (float(v) for v in b)
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area <= 0:
            continue
        ix = max(0.0, min(x1, ox1) - max(x0, ox0))
        iy = max(0.0, min(y1, oy1) - max(y0, oy0))
        best = max(best, ix * iy / area)
    return best
