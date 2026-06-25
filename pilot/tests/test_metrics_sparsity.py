"""Unit tests for visual-attribution concentration metrics."""

import numpy as np

from xai_pilot.metrics.sparsity import regions_above_threshold, topk_mass_ratio


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
