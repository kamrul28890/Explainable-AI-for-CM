"""Stage 6: stability. Pure functions over a list of sampled-decode answers."""
import pytest

from xai_vlm.metrics.stability import (
    centroid,
    is_flickering,
    normalized_centroid_distance,
    pairwise_agreement,
    region_presence_rate,
)


def test_pairwise_agreement_all_same_is_one():
    assert pairwise_agreement(["YES", "YES", "YES", "YES", "YES"]) == 1.0


def test_pairwise_agreement_all_different_pairs_disagree():
    """3 of 3, 2 of 2 etc: exactly one matches out of C(3,2)=3 pairs."""
    assert pairwise_agreement(["YES", "YES", "NO"]) == pytest.approx(1 / 3)


def test_pairwise_agreement_needs_at_least_two():
    assert pairwise_agreement(["YES"]) != pairwise_agreement(["YES"])  # nan
    assert pairwise_agreement([]) != pairwise_agreement([])  # nan


def test_region_presence_rate():
    regions = ["(1,2,3,4)", "", "(5,6,7,8)", ""]
    assert region_presence_rate(regions) == 0.5


def test_region_presence_rate_empty_is_nan():
    assert region_presence_rate([]) != region_presence_rate([])


def test_centroid_of_a_box():
    assert centroid((0.0, 0.0, 10.0, 20.0)) == (5.0, 10.0)


def test_normalized_centroid_distance_is_zero_for_identical_boxes():
    b = (10.0, 10.0, 50.0, 50.0)
    assert normalized_centroid_distance(b, b, (1200, 900)) == pytest.approx(0.0)


def test_normalized_centroid_distance_scales_with_image_diagonal():
    """Same absolute drift, smaller image -> larger normalized distance."""
    a, b = (0.0, 0.0, 10.0, 10.0), (100.0, 0.0, 110.0, 10.0)
    small = normalized_centroid_distance(a, b, (200, 150))
    large = normalized_centroid_distance(a, b, (2000, 1500))
    assert small > large


def test_is_flickering_when_present_in_some_but_not_all():
    assert is_flickering(["(1,2,3,4)", "", "(5,6,7,8)"]) is True


def test_is_flickering_false_when_always_present():
    assert is_flickering(["(1,2,3,4)", "(5,6,7,8)"]) is False


def test_is_flickering_false_when_never_present():
    """Never offering a region is consistent, not flickering -- distinguish the
    two failure modes rather than flagging both as unstable the same way."""
    assert is_flickering(["", ""]) is False
