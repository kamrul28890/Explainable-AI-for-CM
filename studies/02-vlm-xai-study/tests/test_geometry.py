"""Box arithmetic for metric 7. Pure functions, no model."""
import pytest

from xai_vlm.metrics.geometry import iou, max_iou, to_normalised


def test_to_normalised_divides_by_image_size():
    assert to_normalised((600, 450, 1200, 900), 1200, 900) == (0.5, 0.5, 1.0, 1.0)


def test_normalising_is_not_the_identity():
    """The top-left-corner bug: treating pixels as fractions puts every box at 0,0.

    A pixel box (600, 450, 720, 540) on a 1200x900 image is the middle of the
    frame. Read as normalised it would be far outside 0-1. If these ever compare
    equal, the conversion was skipped.
    """
    pixel_box = (600, 450, 720, 540)
    assert to_normalised(pixel_box, 1200, 900) != pixel_box


def test_iou_identical_boxes_is_one():
    assert iou((0.1, 0.1, 0.5, 0.5), (0.1, 0.1, 0.5, 0.5)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou((0.0, 0.0, 0.1, 0.1), (0.8, 0.8, 0.9, 0.9)) == 0.0


def test_iou_half_overlap():
    # a covers x 0..0.2, b covers x 0.1..0.3, same y. Intersection 0.1 wide,
    # union 0.3 wide -> 1/3.
    assert iou((0.0, 0.0, 0.2, 0.1), (0.1, 0.0, 0.3, 0.1)) == pytest.approx(1 / 3)


def test_max_iou_takes_the_best_annotated_box():
    """A photo with three violations: matching any one of them is success."""
    model = (0.5, 0.5, 0.6, 0.6)
    humans = [(0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6), (0.9, 0.9, 1.0, 1.0)]
    assert max_iou(model, humans) == pytest.approx(1.0)


def test_max_iou_with_no_annotated_boxes_is_zero():
    assert max_iou((0.1, 0.1, 0.2, 0.2), []) == 0.0
