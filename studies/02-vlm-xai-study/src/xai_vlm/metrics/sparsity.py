"""Stage 5 -- Metric 2: Sparsity.

Was the model's attention concentrated on the thing it named, or smeared over the
whole photograph? A focused explanation is readable; a diffuse one tells a human
nothing even when the verdict is right.

Two quantities, and the second is the more interesting one here.
"""

from __future__ import annotations

import numpy as np

NAN = float("nan")
Box = tuple[float, float, float, float]


def gini(grid: np.ndarray) -> float:
    """Gini coefficient of the attention distribution. 0 = even, 1 = one patch.

    GINI IS PRIMARY BECAUSE IT IS SIZE-INDEPENDENT. A "share of attention in the
    top-k patches" measure punishes large objects: a guardrail spanning the frame
    *should* attract spread-out attention, and would score as unfocused for being
    large rather than for being badly explained. Gini describes the shape of the
    distribution without reference to how big the object is.

    An all-zero grid returns nan, not zero. No attention at all is not the same as
    perfectly even attention, and averaging a zero into a Gini distribution would
    quietly pull it toward "unfocused".
    """
    x = np.asarray(grid, dtype=float).ravel()
    if x.size == 0 or not np.isfinite(x).all():
        return NAN
    if x.min() < 0:
        x = x - x.min()
    total = x.sum()
    if total <= 0:
        return NAN

    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    # Standard formula: (2*sum(i*xi) / (n*sum(xi))) - (n+1)/n
    return float((2.0 * np.sum(index * x)) / (n * total) - (n + 1.0) / n)


def _region_slice(grid: np.ndarray, region: Box,
                  image_size: tuple[int, int]) -> tuple[slice, slice]:
    rows, cols = grid.shape
    w, h = image_size
    x0, y0, x1, y1 = region
    c0 = int(np.floor(x0 / w * cols))
    c1 = int(np.ceil(x1 / w * cols))
    r0 = int(np.floor(y0 / h * rows))
    r1 = int(np.ceil(y1 / h * rows))
    return (slice(max(0, r0), min(rows, max(r0 + 1, r1))),
            slice(max(0, c0), min(cols, max(c0 + 1, c1))))


def mass_inside_region(grid: np.ndarray, region: Box,
                       image_size: tuple[int, int]) -> float:
    """Share of total attention falling inside the box the model itself named.

    This is an INDEPENDENT HONESTY CHECK, and it needs no masking at all, so it
    carries none of Stage 4's artefact concerns. If a model names a region while
    its attention lies elsewhere, that is direct evidence of unfaithfulness.

    It matters particularly here: Stage 10 found the model's regions agree with
    human annotations only 6.8% of the time, and this separates two very different
    explanations for that -- the model looked at the right place and described it
    badly, or it never looked there at all.
    """
    g = np.asarray(grid, dtype=float)
    total = g.sum()
    if total <= 0:
        return NAN
    rs, cs = _region_slice(g, region, image_size)
    return float(g[rs, cs].sum() / total)


def region_area_fraction(region: Box, image_size: tuple[int, int]) -> float:
    """Fraction of the image the region covers -- the null for mass_inside_region.

    If attention were uniform noise, mass-inside-region would equal this. Reading
    the two together is what makes the metric interpretable: mass 0.4 is
    impressive for a region covering 5% of the frame and unremarkable for one
    covering 40%.
    """
    w, h = image_size
    x0, y0, x1, y1 = region
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    total = float(w) * float(h)
    return float(area / total) if total > 0 else NAN
