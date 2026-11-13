"""Stage 5: sparsity of attention. Pure arithmetic on grids."""
import numpy as np
import pytest

from xai_vlm.metrics.sparsity import gini, mass_inside_region, region_area_fraction


def test_uniform_attention_has_gini_zero():
    """Attention spread evenly over every patch is maximally unfocused."""
    assert gini(np.ones((8, 8))) == pytest.approx(0.0, abs=1e-9)


def test_all_mass_on_one_patch_approaches_one():
    g = np.zeros((8, 8))
    g[3, 4] = 1.0
    assert gini(g) > 0.98


def test_gini_of_a_hand_computed_case():
    """Four patches with mass 1,2,3,4. Gini for this is 0.25 exactly.

    Worked by hand: sum|xi-xj| over all ordered pairs = 40; 2*n^2*mean = 2*16*2.5
    = 80; 40/80 = 0.25.
    """
    assert gini(np.array([[1.0, 2.0], [3.0, 4.0]])) == pytest.approx(0.25)


def test_gini_is_scale_invariant():
    """Doubling every value changes nothing about the distribution's shape."""
    g = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert gini(g) == pytest.approx(gini(g * 7.0))


def test_gini_of_all_zeros_is_nan_not_zero():
    """No attention at all is not 'perfectly even attention'."""
    assert gini(np.zeros((4, 4))) != gini(np.zeros((4, 4)))  # nan


def test_mass_inside_region_counts_only_the_box():
    grid = np.zeros((10, 10))
    grid[0:5, 0:5] = 1.0          # all mass in the top-left quadrant
    # region covering exactly that quadrant of a 100x100 image
    assert mass_inside_region(grid, (0, 0, 50, 50), (100, 100)) == pytest.approx(1.0)


def test_mass_outside_region_is_zero():
    grid = np.zeros((10, 10))
    grid[8:10, 8:10] = 1.0        # mass in the bottom-right
    assert mass_inside_region(grid, (0, 0, 20, 20), (100, 100)) == pytest.approx(0.0)


def test_region_area_fraction_is_the_null_for_mass_inside():
    """If attention were uniform noise, mass-inside would equal the region's area
    fraction. That is the checkable null the metric is read against."""
    assert region_area_fraction((0, 0, 50, 50), (100, 100)) == pytest.approx(0.25)


def test_mass_inside_equals_area_fraction_for_uniform_attention():
    """The null, demonstrated: uniform attention puts exactly the area fraction
    of its mass inside any region."""
    grid = np.ones((20, 20))
    got = mass_inside_region(grid, (0, 0, 50, 50), (100, 100))
    assert got == pytest.approx(region_area_fraction((0, 0, 50, 50), (100, 100)), abs=0.01)
