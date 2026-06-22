"""Bounded completeness: does masking the top-ranked region(s) measurably change the answer?

Per technical-limitations-take-and-mitigation.md's bounded definition: "A
sample passes the bounded completeness check if the model provides a usable
explanation and if perturbing the top-ranked region or top two regions
causes a measurable change in the model's answer, confidence proxy, or
grounding output." Day 5 (descriptive_accuracy.py) already ran exactly this
experiment; classify_sample buckets those already-computed per-sample
results -- no new model calls for the headline classification itself.

The definition also allows "confidence proxy" as a measurable-change signal.
Checked first (results/day10_findings.md): among Day 5's own
answer_changed_top1==False rows, abs(confidence_drop_top1) has essentially
the same distribution (median 0.032) as the full population (median 0.032) --
confidence_drop carries no discriminative signal at this scale, so adding a
confidence-drop threshold here would inject noise into the classification
rather than real signal. Only the categorical answer-change signal is used.
"""

from typing import Literal

Verdict = Literal["explanation_supported", "explanation_weak", "no_usable_explanation"]


def classify_sample(top_region_source: str, answer_changed_top1: bool, answer_changed_top2: bool) -> Verdict:
    """Bucket one sample's Day-5 masking result.

    no_usable_explanation: no native grounding box existed at all (the 4x4
    grid fallback) -- there's no model-pointed region to even test.
    explanation_supported: a usable region existed and masking it (top-1, or
    top-1+top-2) changed the answer.
    explanation_weak: a usable region existed but masking it never changed
    the answer.
    """
    if top_region_source == "grid":
        return "no_usable_explanation"
    if answer_changed_top1 or answer_changed_top2:
        return "explanation_supported"
    return "explanation_weak"
