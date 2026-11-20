"""Patch-grid alignment. No GPU required."""
import pytest

from xai_vlm.masking import snap_to_patch_grid


def test_snapping_expands_outward_to_block_borders():
    # A mask cutting a patch in half leaves a partial, ambiguous signal in it.
    box = snap_to_patch_grid((30, 30, 100, 100), (1200, 900), block=28)
    x0, y0, x1, y1 = box
    assert x0 % 28 == 0 and y0 % 28 == 0
    assert x1 % 28 == 0 and y1 % 28 == 0
    assert x0 <= 30 and y0 <= 30      # never shrinks
    assert x1 >= 100 and y1 >= 100


def test_an_already_aligned_box_is_unchanged():
    assert snap_to_patch_grid((28, 56, 84, 112), (1200, 900), block=28) == (28, 56, 84, 112)


def test_snapping_is_clamped_to_the_image():
    x0, y0, x1, y1 = snap_to_patch_grid((1190, 890, 1199, 899), (1200, 900), block=28)
    assert x1 <= 1200 and y1 <= 900
    assert x1 > x0 and y1 > y0


def test_a_tiny_box_still_covers_at_least_one_block():
    x0, y0, x1, y1 = snap_to_patch_grid((100, 100, 101, 101), (1200, 900), block=28)
    assert x1 - x0 >= 28 and y1 - y0 >= 28
