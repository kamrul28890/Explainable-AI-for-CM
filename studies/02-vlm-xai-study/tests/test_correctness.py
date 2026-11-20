"""Metric 7 verdict logic, on exact arithmetic via a fake embedder."""
import numpy as np

from xai_vlm.metrics.correctness import (
    HAZARD_DESCRIPTIONS,
    IOU_FLOOR,
    SIMILARITY_FLOOR,
    classify,
    hazard_match,
    is_undefined,
)


def test_thresholds_are_the_pre_registered_values():
    """These were fixed before any value was computed. Changing them is a
    deliberate, visible edit -- never a quiet slide to fit a result."""
    assert SIMILARITY_FLOOR == 0.60
    assert IOU_FLOOR == 0.30


def test_the_architecture_docs_correct_example():
    """sim 0.88, PPE match, IoU 0.63 -> CORRECT (arch doc Stage 10)."""
    assert classify(0.88, True, 0.63) == "CORRECT"


def test_the_architecture_docs_incorrect_example():
    """sim 0.11, no hazard match, IoU 0.03 -> INCORRECT (arch doc Stage 10)."""
    assert classify(0.11, False, 0.03) == "INCORRECT"


def test_all_three_signals_are_required():
    assert classify(0.95, True, 0.10) == "INCORRECT"   # points at the wrong place
    assert classify(0.10, True, 0.95) == "INCORRECT"   # says the wrong thing
    assert classify(0.95, False, 0.95) == "INCORRECT"  # wrong hazard category


def test_a_plausible_sentence_pointing_elsewhere_fails():
    """The spatial signal is the anchor. High text similarity must not rescue a
    region on the wrong side of the photograph."""
    assert classify(0.99, True, 0.0) == "INCORRECT"


def test_boundary_values_pass():
    assert classify(SIMILARITY_FLOOR, True, IOU_FLOOR) == "CORRECT"


def test_undefined_when_the_model_said_no():
    assert is_undefined("NO", "(1,2,3,4)", [(0.1, 0.1, 0.2, 0.2)], 1200) == "verdict_no"


def test_undefined_when_no_region_was_given():
    assert is_undefined("YES", "", [(0.1, 0.1, 0.2, 0.2)], 1200) == "no_region"


def test_undefined_when_the_annotation_has_no_box():
    assert is_undefined("YES", "(1,2,3,4)", [], 1200) == "no_annotation_box"


def test_undefined_when_image_dimensions_are_unknown():
    """Without dimensions the model box cannot be normalised, so IoU is undefined
    rather than zero."""
    assert is_undefined("YES", "(1,2,3,4)", [(0.1, 0.1, 0.2, 0.2)], 0) == "no_dimensions"


def test_defined_rows_return_empty_string():
    assert is_undefined("YES", "(1,2,3,4)", [(0.1, 0.1, 0.2, 0.2)], 1200) == ""


def test_hazard_match_picks_the_nearest_hazard():
    """A rationale about a trench must not pass as a PPE answer."""
    vecs = {
        "rule_1": np.array([1.0, 0.0, 0.0, 0.0]),
        "rule_2": np.array([0.0, 1.0, 0.0, 0.0]),
        "rule_3": np.array([0.0, 0.0, 1.0, 0.0]),
        "rule_4": np.array([0.0, 0.0, 0.0, 1.0]),
    }
    ppe_like = np.array([0.9, 0.1, 0.0, 0.0])
    trench_like = np.array([0.0, 0.0, 0.9, 0.1])

    assert hazard_match(ppe_like, "rule_1", vecs) is True
    assert hazard_match(trench_like, "rule_1", vecs) is False
    assert hazard_match(trench_like, "rule_3", vecs) is True


def test_every_rule_has_a_hazard_description():
    from xai_vlm.rules import RULE_IDS
    assert set(HAZARD_DESCRIPTIONS) == set(RULE_IDS)
