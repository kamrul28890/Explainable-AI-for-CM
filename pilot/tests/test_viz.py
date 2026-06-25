"""Unit tests for heatmap overlay dimensions and color behavior."""

import numpy as np
from PIL import Image

from xai_pilot.viz import overlay_heatmap


def test_overlay_heatmap_preserves_image_size():
    image = Image.new("RGB", (40, 30), color=(100, 100, 100))
    heatmap = np.zeros((24, 24))
    out = overlay_heatmap(image, heatmap)
    assert out.size == (40, 30)


def test_overlay_heatmap_hot_cell_shifts_pixel_toward_red():
    image = Image.new("RGB", (24, 24), color=(100, 100, 100))
    heatmap = np.zeros((24, 24))
    heatmap[0, 0] = 1.0  # top-left cell is maximally "hot"
    out = overlay_heatmap(image, heatmap, alpha=1.0)
    arr = np.array(out)
    hot_pixel = arr[0, 0]
    cold_pixel = arr[-1, -1]
    assert hot_pixel[0] > cold_pixel[0]  # more red where the heatmap is hot
