"""Unit tests for visual-attribution concentration metrics."""

import numpy as np

from xai_pilot.metrics.sparsity import (
    box_mass_inside,
    box_mass_ratio,
    gini_concentration,
    regions_above_threshold,
    topk_mass_ratio,
)


def test_topk_mass_ratio_all_mass_in_one_cell_is_one():
    attr_map = np.zeros((4, 4))
    attr_map[0, 0] = 1.0
    assert topk_mass_ratio(attr_map, k=1) == 1.0


def test_topk_mass_ratio_uniform_map_equals_k_over_n():
    attr_map = np.ones((4, 4))  # 16 cells, uniform
    ratio = topk_mass_ratio(attr_map, k=4)
    assert abs(ratio - 4 / 16) < 1e-9


def test_topk_mass_ratio_zero_map_is_nan():
    attr_map = np.zeros((4, 4))
    assert np.isnan(topk_mass_ratio(attr_map, k=1))


def test_regions_above_threshold_counts_matching_cells():
    attr_map = np.array([[0.9, 0.1], [0.6, 0.4]])
    assert regions_above_threshold(attr_map, thresh=0.5) == 2


def test_regions_above_threshold_zero_when_nothing_qualifies():
    attr_map = np.full((3, 3), 0.2)
    assert regions_above_threshold(attr_map, thresh=0.5) == 0


# --- Phase 2.2: size-invariant box concentration ------------------------------
# box_mass_ratio = (attention mass inside the box) / (box's cell-footprint
# fraction). Uniform attention -> 1.0 regardless of box size (size-invariant);
# >1 means concentrated inside the box, <1 means it leaks outside.

# 4x4 grid over a 40x40 image -> 10x10 px cells.
IMG = (40, 40)


def test_box_mass_ratio_uniform_attention_is_one_regardless_of_size():
    attr = np.ones((4, 4))
    # small box (one quadrant) and a full-frame box both score ~1 under uniform.
    assert abs(box_mass_ratio(attr, (0, 0, 20, 20), IMG) - 1.0) < 1e-9
    assert abs(box_mass_ratio(attr, (0, 0, 40, 40), IMG) - 1.0) < 1e-9


def test_box_mass_ratio_concentrated_inside_small_box_exceeds_one():
    attr = np.zeros((4, 4))
    attr[0, 0] = 1.0  # all mass in the top-left cell
    # Box covering only that one cell: footprint 1/16, mass inside 1.0 -> 16x.
    assert abs(box_mass_ratio(attr, (0, 0, 10, 10), IMG) - 16.0) < 1e-9


def test_box_mass_ratio_full_frame_box_is_one_even_when_concentrated():
    # The size confound: a large object's box spans the frame, so even highly
    # concentrated attention scores ~1 -- exactly what removes the size bias.
    attr = np.zeros((4, 4))
    attr[0, 0] = 1.0
    assert abs(box_mass_ratio(attr, (0, 0, 40, 40), IMG) - 1.0) < 1e-9


def test_box_mass_ratio_zero_when_mass_is_all_outside_box():
    attr = np.zeros((4, 4))
    attr[3, 3] = 1.0  # bottom-right
    assert box_mass_ratio(attr, (0, 0, 10, 10), IMG) == 0.0


def test_box_mass_ratio_nan_on_empty_attention():
    assert np.isnan(box_mass_ratio(np.zeros((4, 4)), (0, 0, 20, 20), IMG))


# box_mass_inside = fraction of total attention mass inside the box, WITHOUT
# dividing by footprint. Counts all the object's cells (unlike top-5), so a
# large object is not penalized for spanning many cells -- the actually
# size-invariant concentration measure.


def test_box_mass_inside_uniform_equals_footprint_fraction():
    attr = np.ones((4, 4))
    assert abs(box_mass_inside(attr, (0, 0, 20, 20), IMG) - 0.25) < 1e-9


def test_box_mass_inside_full_frame_captures_all_mass():
    attr = np.zeros((4, 4))
    attr[0, 0] = 1.0
    assert abs(box_mass_inside(attr, (0, 0, 40, 40), IMG) - 1.0) < 1e-9


def test_box_mass_inside_captures_all_of_a_large_objects_spread_attention():
    # A large object whose attention spreads over its whole (large) box still
    # scores ~1.0 -- exactly what top-5-of-many fails to capture.
    attr = np.zeros((4, 4))
    attr[0:4, 0:2] = 1.0  # left half lit
    # Box covering the whole lit left half (cols 0-1 = x 0..20).
    assert abs(box_mass_inside(attr, (0, 0, 20, 40), IMG) - 1.0) < 1e-9


def test_box_mass_inside_zero_when_mass_outside():
    attr = np.zeros((4, 4))
    attr[3, 3] = 1.0
    assert box_mass_inside(attr, (0, 0, 10, 10), IMG) == 0.0


def test_box_mass_inside_nan_on_empty_attention():
    assert np.isnan(box_mass_inside(np.zeros((4, 4)), (0, 0, 20, 20), IMG))


# gini_concentration: distribution-shape focus measure of the whole attention
# map, independent of any object box -- the genuinely size-agnostic sparsity
# statistic. 0 = uniform (diffuse), ~1 = all mass in one cell (maximally focused).


def test_gini_uniform_map_is_zero():
    assert abs(gini_concentration(np.ones((4, 4)))) < 1e-9


def test_gini_single_cell_is_near_one():
    attr = np.zeros((4, 4))
    attr[0, 0] = 1.0
    # For n=16 cells with all mass in one, Gini = (n-1)/n = 15/16.
    assert abs(gini_concentration(attr) - 15 / 16) < 1e-9


def test_gini_more_concentrated_scores_higher():
    diffuse = np.array([[1.0, 1.0], [1.0, 1.0]])
    peaked = np.array([[10.0, 0.1], [0.1, 0.1]])
    assert gini_concentration(peaked) > gini_concentration(diffuse)


def test_gini_nan_on_empty_attention():
    assert np.isnan(gini_concentration(np.zeros((4, 4))))
