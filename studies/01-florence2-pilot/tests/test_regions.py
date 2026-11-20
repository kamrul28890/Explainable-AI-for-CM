"""Unit tests for region geometry, masking, proximity, and fallback ranking."""

import numpy as np
from PIL import Image

import math

from xai_pilot.regions import (
    all_boxes_covered,
    boxes_overlap_or_close,
    fraction_covered,
    grid_fallback_regions,
    iou,
    mask_region,
    normalized_centroid_distance,
    standardize_regions,
)


def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou((0.0, 0.0, 5.0, 5.0), (10.0, 10.0, 15.0, 15.0)) == 0.0


def test_iou_partial_overlap():
    box_a = (0.0, 0.0, 10.0, 10.0)
    box_b = (5.0, 0.0, 15.0, 10.0)
    # overlap area = 5*10=50, union = 100+100-50=150
    assert abs(iou(box_a, box_b) - 50 / 150) < 1e-6


def test_mask_region_black_changes_only_that_region():
    arr = np.full((20, 20, 3), 200, dtype=np.uint8)
    image = Image.fromarray(arr)
    masked = mask_region(image, (5, 5, 10, 10), mode="black")
    masked_arr = np.array(masked)

    inside = masked_arr[5:10, 5:10]
    outside_corner = masked_arr[0:5, 0:5]
    assert inside.mean() < 10
    assert outside_corner.mean() > 190


def test_mask_region_inpaint_fills_from_surroundings_not_black():
    # A bright field with a black square; inpainting the square should fill it
    # from the bright surroundings (content-aware), not leave it black like the
    # out-of-distribution black mask.
    arr = np.full((40, 40, 3), 200, dtype=np.uint8)
    arr[15:25, 15:25] = 0
    image = Image.fromarray(arr)
    inpainted = mask_region(image, (15, 15, 25, 25), mode="inpaint")
    box = np.array(inpainted)[15:25, 15:25]
    assert box.mean() > 150  # filled bright, not black
    assert inpainted.size == image.size


def test_mask_region_rejects_unknown_mode():
    import pytest

    image = Image.fromarray(np.full((10, 10, 3), 200, dtype=np.uint8))
    with pytest.raises(ValueError):
        mask_region(image, (1, 1, 5, 5), mode="nonsense")


def test_grid_fallback_regions_covers_image():
    boxes = grid_fallback_regions((100, 80), grid=(4, 4))
    assert len(boxes) == 16
    xs = [b[0] for b in boxes] + [b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[3] for b in boxes]
    assert min(xs) == 0
    assert max(xs) == 100
    assert min(ys) == 0
    assert max(ys) == 80


def test_boxes_overlap_or_close_true_when_overlapping():
    assert boxes_overlap_or_close((0, 0, 10, 10), (5, 5, 15, 15))


def test_boxes_overlap_or_close_true_when_within_threshold():
    assert boxes_overlap_or_close((0, 0, 10, 10), (12, 0, 20, 10), distance_threshold=5.0)


def test_boxes_overlap_or_close_false_when_far_apart():
    assert not boxes_overlap_or_close((0, 0, 10, 10), (100, 100, 110, 110), distance_threshold=5.0)


def test_all_boxes_covered_true_when_every_subject_has_a_match():
    workers = [(0, 0, 10, 10), (50, 50, 60, 60)]
    objects = [(2, 2, 8, 8), (52, 52, 58, 58)]
    assert all_boxes_covered(workers, objects)


def test_all_boxes_covered_false_when_one_subject_has_no_match():
    workers = [(0, 0, 10, 10), (50, 50, 60, 60)]
    objects = [(2, 2, 8, 8)]  # only covers the first worker
    assert not all_boxes_covered(workers, objects)


def test_all_boxes_covered_false_when_no_reference_boxes():
    assert not all_boxes_covered([(0, 0, 10, 10)], [])


def test_all_boxes_covered_vacuously_true_when_no_subject_boxes():
    assert all_boxes_covered([], [(0, 0, 10, 10)])


# --- Phase 2.1: graded (continuous) coverage signal ---------------------------


def test_fraction_covered_all_subjects_matched_is_one():
    workers = [(0, 0, 10, 10), (50, 50, 60, 60)]
    objects = [(2, 2, 8, 8), (52, 52, 58, 58)]
    assert fraction_covered(workers, objects) == 1.0


def test_fraction_covered_half_matched_is_half():
    workers = [(0, 0, 10, 10), (50, 50, 60, 60)]
    objects = [(2, 2, 8, 8)]  # covers only the first worker
    assert fraction_covered(workers, objects) == 0.5


def test_fraction_covered_no_reference_boxes_is_zero():
    assert fraction_covered([(0, 0, 10, 10)], []) == 0.0


def test_fraction_covered_no_subject_boxes_is_nan():
    # No workers -> the fraction is undefined, not vacuously 1.0, so masking
    # magnitude on a worker-less rerun is not silently counted as full coverage.
    assert math.isnan(fraction_covered([], [(0, 0, 10, 10)]))


def test_fraction_covered_respects_distance_threshold():
    workers = [(0, 0, 10, 10)]
    objects = [(12, 0, 20, 10)]  # 2px gap
    assert fraction_covered(workers, objects, distance_threshold=5.0) == 1.0
    assert fraction_covered(workers, objects, distance_threshold=1.0) == 0.0


def test_normalized_centroid_distance_identical_is_zero():
    assert normalized_centroid_distance((0, 0, 10, 10), (0, 0, 10, 10), (100, 100)) == 0.0


def test_normalized_centroid_distance_known_shift():
    # Centroids at (5,5) and (35,5): distance 30 / diagonal ~141.42 = 0.2121.
    d = normalized_centroid_distance((0, 0, 10, 10), (30, 0, 40, 10), (100, 100))
    assert abs(d - 30 / (100 * 2 ** 0.5)) < 1e-9


def test_standardize_regions_ranks_model_boxes_by_area_descending():
    boxes = [(0, 0, 10, 10), (0, 0, 100, 100), (0, 0, 5, 5)]
    labels = ["small", "big", "tiny"]
    regions = standardize_regions(boxes, labels, image_size=(200, 200))
    assert [r.label for r in regions] == ["big", "small", "tiny"]
    assert all(r.source == "model" for r in regions)


def test_standardize_regions_falls_back_to_grid_when_no_boxes():
    regions = standardize_regions([], [], image_size=(100, 80), grid=(4, 4))
    assert len(regions) == 16
    assert all(r.source == "grid" for r in regions)


def test_standardize_regions_grid_fallback_matches_grid_fallback_regions():
    regions = standardize_regions([], [], image_size=(100, 80), grid=(2, 2))
    expected = grid_fallback_regions((100, 80), grid=(2, 2))
    assert [r.box for r in regions] == expected


def test_standardize_regions_default_is_area_ranking():
    # The frozen pilot behavior: absent an explicit mode, rank by area.
    boxes = [(0, 0, 10, 10), (0, 0, 100, 100)]
    labels = ["worker", "hard hat"]
    regions = standardize_regions(boxes, labels, image_size=(200, 200))
    assert [r.label for r in regions] == ["hard hat", "worker"]


def test_standardize_regions_rule_aware_promotes_object_over_larger_worker():
    # The pilot's #1 bug: the large worker body box outranks the small PPE
    # object under area ranking. rule_aware must invert that so the queried
    # object (hard hat) becomes the top explanation region.
    boxes = [(0, 0, 100, 100), (10, 10, 20, 20)]
    labels = ["worker", "hard hat"]
    regions = standardize_regions(
        boxes, labels, image_size=(200, 200),
        region_ranking="rule_aware", object_labels={"hard hat"},
    )
    assert regions[0].label == "hard hat"
    assert regions[1].label == "worker"
    assert all(r.source == "model" for r in regions)


def test_standardize_regions_rule_aware_sorts_by_area_within_each_tier():
    # Object-labeled regions rank first, area-descending among themselves;
    # non-object regions follow, also area-descending.
    boxes = [
        (0, 0, 30, 30),    # worker, big
        (0, 0, 10, 10),    # worker, small
        (0, 0, 20, 20),    # hard hat, big object
        (0, 0, 5, 5),      # hard hat, small object
    ]
    labels = ["worker", "worker", "hard hat", "hard hat"]
    regions = standardize_regions(
        boxes, labels, image_size=(200, 200),
        region_ranking="rule_aware", object_labels={"hard hat"},
    )
    areas = [ (r.box[2]-r.box[0]) for r in regions ]
    assert [r.label for r in regions] == ["hard hat", "hard hat", "worker", "worker"]
    # 20x20 object before 5x5 object; 30x30 worker before 10x10 worker
    assert areas == [20, 5, 30, 10]


def test_standardize_regions_rule_aware_falls_back_to_grid_when_no_boxes():
    regions = standardize_regions(
        [], [], image_size=(100, 80), grid=(4, 4),
        region_ranking="rule_aware", object_labels={"hard hat"},
    )
    assert len(regions) == 16
    assert all(r.source == "grid" for r in regions)


def test_standardize_regions_rule_aware_requires_object_labels():
    # Fail loud rather than silently degrading to area ranking (guiding
    # principle: no silent hardcoded/implicit assumptions).
    import pytest

    with pytest.raises(ValueError):
        standardize_regions(
            [(0, 0, 10, 10)], ["worker"], image_size=(200, 200),
            region_ranking="rule_aware", object_labels=None,
        )


def test_standardize_regions_unknown_ranking_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        standardize_regions(
            [(0, 0, 10, 10)], ["worker"], image_size=(200, 200),
            region_ranking="nonsense",
        )


def test_standardize_regions_attention_ranking_not_yet_implemented():
    # Phase 4 wires cross-attention / IG mass into ranking; until then the
    # mode must exist as a named option but refuse to run silently.
    import pytest

    with pytest.raises(NotImplementedError):
        standardize_regions(
            [(0, 0, 10, 10)], ["worker"], image_size=(200, 200),
            region_ranking="attention", object_labels={"hard hat"},
        )
