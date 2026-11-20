"""Unit tests for controlled image perturbations used by Day 9."""

import numpy as np
from PIL import Image

from xai_pilot.perturbations import blur, contrast_shift, low_light, occlude, paste_patch


def _checkerboard(size=40):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[: size // 2, : size // 2] = 255
    arr[size // 2 :, size // 2 :] = 255
    return Image.fromarray(arr)


def test_blur_smooths_a_sharp_edge():
    image = _checkerboard()
    blurred = blur(image, ksize=5)
    sharp_std = np.asarray(image).astype(float).std()
    blurred_std = np.asarray(blurred).astype(float).std()
    assert blurred_std < sharp_std


def test_low_light_darkens_image():
    image = Image.new("RGB", (20, 20), color=(200, 200, 200))
    darker = low_light(image, gamma=2.5)
    assert np.asarray(darker).mean() < np.asarray(image).mean()


def test_low_light_gamma_one_is_unchanged():
    image = Image.new("RGB", (20, 20), color=(128, 128, 128))
    same = low_light(image, gamma=1.0)
    assert np.allclose(np.asarray(same).astype(int), np.asarray(image).astype(int), atol=1)


def test_occlude_blacks_out_center_region():
    image = Image.new("RGB", (40, 40), color=(255, 255, 255))
    out = occlude(image, frac=0.25)
    arr = np.asarray(out)
    assert tuple(arr[20, 20]) == (0, 0, 0)  # center is occluded
    assert tuple(arr[0, 0]) == (255, 255, 255)  # corner is untouched


def test_occlude_area_matches_requested_fraction():
    image = Image.new("RGB", (100, 100), color=(255, 255, 255))
    out = occlude(image, frac=0.16)
    arr = np.asarray(out)
    black_pixels = int((arr.sum(axis=-1) == 0).sum())
    expected = 0.16 * 100 * 100
    assert abs(black_pixels - expected) < 100  # rounding to an integer side length


def test_occlude_random_location_is_seeded_and_off_center(monkeypatch=None):
    # Phase 2.4: random-location occlusion must be reproducible for a seed and
    # (usually) not centered, so "covered the object" and "disrupted the whole
    # scene" are no longer conflated as in the pilot's centered square.
    image = Image.new("RGB", (100, 100), color=(255, 255, 255))
    a = np.asarray(occlude(image, frac=0.1, location="random", seed=1))
    b = np.asarray(occlude(image, frac=0.1, location="random", seed=1))
    c = np.asarray(occlude(image, frac=0.1, location="random", seed=2))
    assert np.array_equal(a, b)  # same seed -> identical
    assert not np.array_equal(a, c)  # different seed -> different placement
    # occluded area still matches the requested fraction
    assert abs(int((a.sum(axis=-1) == 0).sum()) - 0.1 * 100 * 100) < 120


def test_occlude_center_matches_default():
    image = Image.new("RGB", (60, 60), color=(255, 255, 255))
    default = np.asarray(occlude(image, frac=0.2))
    centered = np.asarray(occlude(image, frac=0.2, location="center"))
    assert np.array_equal(default, centered)


def test_contrast_shift_factor_one_is_unchanged():
    image = Image.new("RGB", (10, 10), color=(50, 150, 50))
    same = contrast_shift(image, factor=1.0)
    assert np.allclose(np.asarray(same).astype(int), np.asarray(image).astype(int), atol=1)


def test_contrast_shift_low_factor_flattens_variance():
    image = _checkerboard()
    flattened = contrast_shift(image, factor=0.1)
    assert np.asarray(flattened).astype(float).std() < np.asarray(image).astype(float).std()


def test_paste_patch_replaces_box_region():
    image = Image.new("RGB", (40, 40), color=(0, 0, 0))
    patch = Image.new("RGB", (5, 5), color=(255, 255, 0))
    out = paste_patch(image, box=(10, 10, 20, 20), patch=patch)
    arr = np.asarray(out)
    assert tuple(arr[15, 15]) == (255, 255, 0)
    assert tuple(arr[0, 0]) == (0, 0, 0)
