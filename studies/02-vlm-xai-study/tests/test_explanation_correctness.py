"""Stage 10 end to end, over a fake embedder. No model weights."""
import importlib.util
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm.metrics.text_similarity import FakeEmbedder  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "explanation_correctness", STUDY_ROOT / "scripts" / "10_explanation_correctness.py")
stage10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage10)

HUMAN = "The mechanic in front of the truck is not wearing a hard hat."
MODEL_GOOD = "The man working beside the truck cab has no helmet on."
MODEL_BAD = "A worker is not wearing a high-visibility vest."


def _embedder():
    # Good model text sits near the human text and near rule_1's description;
    # bad text is orthogonal to the human text.
    return FakeEmbedder({
        HUMAN:       [1.0, 0.0, 0.0, 0.0],
        MODEL_GOOD:  [0.98, 0.2, 0.0, 0.0],
        MODEL_BAD:   [0.0, 0.0, 0.0, 1.0],
        stage10.HAZARD_DESCRIPTIONS["rule_1"]: [1.0, 0.0, 0.0, 0.0],
        stage10.HAZARD_DESCRIPTIONS["rule_2"]: [0.0, 1.0, 0.0, 0.0],
        stage10.HAZARD_DESCRIPTIONS["rule_3"]: [0.0, 0.0, 1.0, 0.0],
        stage10.HAZARD_DESCRIPTIONS["rule_4"]: [0.0, 0.0, 0.0, 1.0],
    })


def _row(image_id, rationale, verdict="YES", region="(600.0, 450.0, 720.0, 540.0)"):
    return {"image_id": image_id, "rule_id": "rule_1", "verdict": verdict,
            "expected": "YES", "rationale": rationale, "region": region}


def _regions(image_id, region="(600.0, 450.0, 720.0, 540.0)", w=1200, h=900):
    return {"image_id": image_id, "rule_id": "rule_1", "verdict": "YES",
            "source": "model" if region else "none", "region": region,
            "image_w": w, "image_h": h}


def _ann(image_id, boxes, reason=HUMAN, quality="rich info"):
    return {image_id: {"rule_1": {"bounding_box": boxes, "reason": reason},
                       "quality_of_info": quality}}


def test_a_good_explanation_scores_correct():
    boxes = [[0.5, 0.5, 0.6, 0.6]]
    rows = stage10.score_rows([_row("0001", MODEL_GOOD)], [_regions("0001")],
                              _ann("0001", boxes), _embedder())
    assert rows[0]["verdict_correctness"] == "CORRECT"
    assert rows[0]["undefined_reason"] == ""


def test_right_answer_wrong_reason_is_caught():
    """The model says YES, ground truth is YES, but it describes another hazard
    in a different part of the frame."""
    boxes = [[0.0, 0.0, 0.1, 0.1]]
    rows = stage10.score_rows([_row("0002", MODEL_BAD)], [_regions("0002")],
                              _ann("0002", boxes), _embedder())
    assert rows[0]["verdict_correctness"] == "INCORRECT"


def test_no_verdict_rows_are_undefined_not_incorrect():
    rows = stage10.score_rows([_row("0003", MODEL_GOOD, verdict="NO")],
                              [_regions("0003")], _ann("0003", [[0.5, 0.5, 0.6, 0.6]]),
                              _embedder())
    assert rows[0]["undefined_reason"] == "verdict_no"
    assert rows[0]["verdict_correctness"] == ""


def test_summary_reports_undefined_separately_from_incorrect():
    rows = [
        {"verdict_correctness": "CORRECT", "undefined_reason": "", "quality_of_info": "rich info"},
        {"verdict_correctness": "INCORRECT", "undefined_reason": "", "quality_of_info": "rich info"},
        {"verdict_correctness": "", "undefined_reason": "no_region", "quality_of_info": "poor info"},
    ]
    s = stage10.summarise(rows)
    assert s["n_scored"] == 2
    assert s["n_undefined"] == 1
    assert s["correct_rate"] == 0.5


def test_coordinates_are_converted_before_comparison():
    """The model box in pixels covers the image centre; the annotation in
    normalised units covers the same place. Without conversion IoU would be 0."""
    boxes = [[0.5, 0.5, 0.6, 0.6]]
    rows = stage10.score_rows([_row("0004", MODEL_GOOD)], [_regions("0004")],
                              _ann("0004", boxes), _embedder())
    assert rows[0]["iou"] > 0.9
