import numpy as np
from PIL import Image

from xai_pilot.regions import boxes_overlap_or_close, grid_fallback_regions, iou, mask_region


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
