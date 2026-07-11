"""Visual sparsity: how concentrated is the cross-attention heatmap?

Computed only on the 24x24 visual-stream heatmap from attribution.py --
text-prompt-token attention lives in a different space (different sequence
length, mixed with location/label token semantics) and is not merged into
these numbers, per the pilot plan's design decision to report visual and
text attribution streams separately rather than as one combined score.
"""

import numpy as np


def topk_mass_ratio(attr_map: np.ndarray, k: int) -> float:
    """Fraction of total attribution mass held by the top-k highest cells.

    1.0 = maximally sparse (all mass concentrated in k cells); close to
    k/n = scattered (near-uniform attribution).
    """
    flat = attr_map.flatten().astype(float)
    total = flat.sum()
    if total <= 0:
        return float("nan")
    # Sorting is acceptable for the fixed 576-cell map and keeps the metric
    # definition transparent. The largest k cells represent concentrated mass.
    top_k_sum = np.sort(flat)[-k:].sum()
    return float(top_k_sum / total)


def _box_cell_stats(
    attr_map: np.ndarray,
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
):
    """Return (mass_inside_fraction, footprint_fraction) for a pixel-space box.

    Maps the box onto the attention grid and sums the mass of every cell that
    overlaps it. NaN mass fraction when the map is empty; zero footprint when
    the box covers no cell.
    """
    total = float(attr_map.sum())
    rows, cols = attr_map.shape
    width, height = image_size
    cell_w, cell_h = width / cols, height / rows
    bx0, by0, bx1, by1 = box

    inside_mass = 0.0
    footprint_cells = 0
    for r in range(rows):
        cy0, cy1 = r * cell_h, (r + 1) * cell_h
        if cy0 >= by1 or cy1 <= by0:
            continue
        for c in range(cols):
            cx0, cx1 = c * cell_w, (c + 1) * cell_w
            if cx0 >= bx1 or cx1 <= bx0:
                continue
            footprint_cells += 1
            inside_mass += float(attr_map[r, c])

    mass_fraction = (inside_mass / total) if total > 0 else float("nan")
    footprint_fraction = footprint_cells / (rows * cols)
    return mass_fraction, footprint_fraction


def box_mass_inside(
    attr_map: np.ndarray,
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> float:
    """Fraction of total attention mass falling inside the grounded box
    (Scale-up Phase 2.2), the size-invariant concentration measure.

    Unlike the pilot's top-k-of-576 measure, this counts *every* cell the box
    covers, so a large object whose attention correctly spreads over its whole
    (large) box is not scored as ``unfocused'' merely for spanning many cells.
    Returns NaN if the map has no mass.
    """
    mass_fraction, _ = _box_cell_stats(attr_map, box, image_size)
    return mass_fraction


def box_mass_ratio(
    attr_map: np.ndarray,
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> float:
    """Attention mass inside the box divided by the box's cell-footprint
    fraction (the plan's proposed size-invariant formula, Scale-up Phase 2.2).

    Under *uniform* attention this is 1.0 regardless of box size. Note, however,
    that on real (peaked) attention the footprint division *over*-corrects for
    small boxes (a tiny footprint inflates the ratio), so this measure is
    retained for reference but ``box_mass_inside`` is the metric that actually
    removes the size confound (see the Phase 2.2 chapter). Returns NaN if the
    map has no mass or the box covers no cell.
    """
    mass_fraction, footprint_fraction = _box_cell_stats(attr_map, box, image_size)
    if mass_fraction != mass_fraction or footprint_fraction == 0:  # NaN or no cell
        return float("nan")
    return mass_fraction / footprint_fraction


def gini_concentration(attr_map: np.ndarray) -> float:
    """Gini coefficient of the attention map -- a size-agnostic focus measure.

    Unlike top-k or box-relative measures, the Gini coefficient describes the
    *shape* of the attention distribution over all cells without referencing the
    object box, so it does not carry the object-size confound that biases the
    mass-based measures (Scale-up Phase 2.2). 0.0 means perfectly uniform
    (diffuse) attention; values approaching 1 mean the mass is concentrated in a
    few cells (focused). Returns NaN for an all-zero map.
    """
    x = np.sort(attr_map.flatten().astype(float))
    total = x.sum()
    if total <= 0:
        return float("nan")
    n = x.size
    index = np.arange(1, n + 1)
    # Standard sorted-array Gini formula.
    return float((2.0 * np.sum(index * x)) / (n * total) - (n + 1) / n)


def regions_above_threshold(attr_map: np.ndarray, thresh: float) -> int:
    """Count of cells whose normalized attribution is >= thresh.

    Assumes attr_map is already normalized to [0, 1] (as
    attribution.cross_attention_heatmap returns it). Fewer cells above
    threshold means a more focused explanation.
    """
    return int((attr_map >= thresh).sum())
