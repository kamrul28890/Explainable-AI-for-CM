"""Stage 10 -- Metric 7: Explanation Correctness.

Does the model's stated reason match the human's? This is the only metric in the
battery that can catch a RIGHT ANSWER FOR THE WRONG REASON -- and after Stage 2
measured 91% sensitivity against 41% specificity, that is the live question. A
model answering YES to 70% of everything gets many verdicts right by disposition,
and no accuracy measure can tell disposition from understanding.
"""

from __future__ import annotations

import numpy as np

from xai_vlm.metrics.text_similarity import cosine
from xai_vlm.rules import RULES

# PRE-REGISTERED, fixed before any value was computed, taken from convention:
# 0.60 is a standard paraphrase floor for MiniLM-class bi-encoders; 0.30 is a
# deliberately permissive IoU, because the model is asked for ONE region while the
# annotation may mark several.
#
# NOT TO BE TUNED AGAINST OUR RESULTS. All three raw signals are written to the
# output CSV, so anyone wanting a different threshold can re-derive without
# re-running anything -- which is the honest way to explore sensitivity.
SIMILARITY_FLOOR = 0.60
IOU_FLOOR = 0.30

# One canonical sentence per hazard, used to classify what the model's rationale
# is actually about. Phrased as the dataset would describe the hazard, not as the
# question was asked, so a rationale is matched on content rather than on echoing
# the prompt back.
HAZARD_DESCRIPTIONS = {
    "rule_1": "A worker is not wearing required protective equipment such as a "
              "hard hat or high-visibility clothing.",
    "rule_2": "A worker is at height without fall protection such as a harness "
              "or lanyard.",
    "rule_3": "There is an unguarded edge, excavation, trench or floor opening "
              "someone could fall into.",
    "rule_4": "A worker is too close to operating heavy machinery such as an "
              "excavator.",
}


def hazard_match(rationale_vec: np.ndarray, asked_rule: str,
                 hazard_vecs: dict) -> bool:
    """Does the rationale describe the hazard that was asked about?

    Catches answering YES about PPE while describing a trench -- a gross category
    error that text similarity alone can miss when both sentences are about
    construction sites.
    """
    best, best_sim = None, -2.0
    for rule_id, vec in hazard_vecs.items():
        sim = cosine(rationale_vec, vec)
        if sim > best_sim:
            best, best_sim = rule_id, sim
    return best == asked_rule


def classify(similarity: float, hazard_ok: bool, iou_value: float) -> str:
    """Conjunctive verdict: all three signals must pass.

    Conjunctive rather than weighted because the SPATIAL SIGNAL IS THE ANCHOR.
    Text similarity is fuzzy; box overlap is arithmetic. A weighted score would let
    a high text score outvote a region on the wrong side of the photograph, which
    is the exact failure geometry exists to catch.
    """
    ok = (similarity >= SIMILARITY_FLOOR) and hazard_ok and (iou_value >= IOU_FLOOR)
    return "CORRECT" if ok else "INCORRECT"


def is_undefined(verdict: str, region: str, human_boxes: list, image_w: int) -> str:
    """Why this row cannot be scored, or "" if it can.

    Undefined is NOT incorrect. Scoring these as INCORRECT would inflate the
    failure rate with cases the metric does not cover; scoring them CORRECT would
    do the reverse. Either is a silent lie about coverage.
    """
    if verdict != "YES":
        return "verdict_no"
    if not (region or "").strip():
        return "no_region"
    if not human_boxes:
        return "no_annotation_box"
    if not image_w:
        return "no_dimensions"
    return ""


def hazard_of(rule_id: str) -> str:
    """The hazard category a rule belongs to."""
    return RULES[rule_id].hazard
