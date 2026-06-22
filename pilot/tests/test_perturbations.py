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
