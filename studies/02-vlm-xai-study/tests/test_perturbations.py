"""Stage 8 perturbations. Pure image arithmetic, no model, no GPU.

Strengths were approved by the user on 2026-11-10 BEFORE any run, as the Plan 3c
spec requires (strength is a free parameter; tuning it after seeing results would
manufacture whatever spread was wanted). These tests pin them.
"""
import numpy as np
import pytest
from PIL import Image

from xai_vlm import perturbations as P


def _img(w=200, h=100, rgb=(200, 40, 40)):
    return Image.new("RGB", (w, h), rgb)


def _arr(img):
    return np.asarray(img, dtype=float)


def test_the_five_approved_perturbations_in_a_fixed_order():
    assert P.PERTURBATIONS == ("blur", "low_light", "occlusion", "contrast", "desaturate")


def test_approved_strengths_are_pinned():
    assert P.STRENGTHS == {
        "blur_sigma_frac_long_side": 0.01,
        "low_light_ev": -2.0,
        "occlusion_frac_w": 0.5, "occlusion_frac_h": 0.5,
        "contrast_factor": 0.4,
        "desaturate": "ITU-R 601-2 luma",
    }


def test_every_perturbation_keeps_size_and_mode_and_leaves_the_input_alone():
    src = _img()
    before = _arr(src).copy()
    for name in P.PERTURBATIONS:
        out = P.apply(name, src)
        assert out.size == src.size and out.mode == "RGB", name
    assert np.array_equal(_arr(src), before)


def test_unknown_perturbation_is_an_error_not_a_passthrough():
    with pytest.raises(ValueError):
        P.apply("rain", _img())


def test_blur_sigma_scales_with_the_long_side():
    assert P.blur_sigma((800, 600)) == pytest.approx(8.0)
    assert P.blur_sigma((600, 1200)) == pytest.approx(12.0)


def test_blur_smears_a_sharp_edge_and_keeps_flat_regions_flat():
    img = Image.new("RGB", (400, 100), (0, 0, 0))
    img.paste((255, 255, 255), (200, 0, 400, 100))
    out = _arr(P.apply("blur", img))
    assert 20 < out[50, 200, 0] < 235          # the edge is no longer a step
    assert out[50, 20, 0] < 2 and out[50, 380, 0] > 253   # far from it, unchanged


def test_low_light_is_two_stops_in_linear_light_not_in_srgb():
    """-2 EV quarters LINEAR intensity. Quartering sRGB values would darken far
    more than two stops in the shadows and less in the highlights."""
    out = _arr(P.apply("low_light", _img(rgb=(255, 128, 0))))[0, 0]
    assert out[0] == pytest.approx(137, abs=1)   # linear 1.0 -> 0.25 -> sRGB 137
    assert out[1] == pytest.approx(65, abs=1)    # linear 0.216 -> 0.054 -> sRGB 65
    assert out[2] == 0


def test_occlusion_covers_exactly_the_central_quarter_with_mean_grey():
    img = Image.new("RGB", (200, 100), (0, 0, 0))
    img.paste((255, 255, 255), (0, 0, 100, 100))          # left half white
    out = _arr(P.apply("occlusion", img))
    grey = out[50, 100]
    assert grey[0] == grey[1] == grey[2] == pytest.approx(128, abs=1)
    assert np.all(out[25:75, 50:150] == grey)              # the block
    assert np.all(out[:25, :50] == 255) and np.all(out[:25, 150:] == 0)


def test_occluder_box_is_normalised_and_central():
    assert P.OCCLUDER_NORM == (0.25, 0.25, 0.75, 0.75)


def test_contrast_pulls_every_pixel_toward_the_mean_by_the_factor():
    img = Image.new("L", (100, 100), 0)
    img.paste(200, (0, 0, 50, 100))                       # mean 100
    out = _arr(P.apply("contrast", img.convert("RGB")))
    assert out[0, 0, 0] == pytest.approx(100 + 0.4 * 100, abs=1)
    assert out[0, 99, 0] == pytest.approx(100 - 0.4 * 100, abs=1)


def test_desaturate_removes_all_colour():
    out = _arr(P.apply("desaturate", _img(rgb=(255, 120, 0))))
    assert np.all(out[..., 0] == out[..., 1]) and np.all(out[..., 1] == out[..., 2])


def test_occluder_coverage_of_an_annotated_box():
    # Box wholly inside the block, wholly outside, and half in.
    assert P.occluder_coverage([[0.3, 0.3, 0.4, 0.4]]) == pytest.approx(1.0)
    assert P.occluder_coverage([[0.0, 0.0, 0.1, 0.1]]) == pytest.approx(0.0)
    assert P.occluder_coverage([[0.5, 0.3, 1.0, 0.4]]) == pytest.approx(0.5)


def test_coverage_takes_the_most_covered_box_and_is_none_without_boxes():
    """A hazard with two boxes is hidden if either instance is."""
    assert P.occluder_coverage([[0.0, 0.0, 0.1, 0.1], [0.3, 0.3, 0.4, 0.4]]) == 1.0
    assert P.occluder_coverage([]) is None
