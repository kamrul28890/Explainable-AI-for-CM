"""Stage 4 -- Metric 1: Descriptive Accuracy.

The primary faithfulness measurement. If deleting the region the model named does
not change its answer, that region was decoration and the explanation is not an
explanation.

This is the test a model cannot pass by disposition. Stage 2 measured 91%
sensitivity against 41% specificity, so many verdicts are right by inclination
rather than evidence; a flip rate is indifferent to inclination.
"""

from __future__ import annotations

# black  -- patch-aligned fill, comparable with prior work, but creates a
#           rectangle no photograph contains.
# tokens -- delete the image tokens before the language model runs. No pixels are
#           modified, so no artefact can exist. PRIMARY.
MASKING_METHODS = ("black", "tokens")

NAN = float("nan")


def is_flip(baseline_verdict: str, masked_verdict: str) -> bool:
    """Did removing the region change the answer?

    An empty masked verdict is a PARSE FAILURE, not a flip. Counting it as one
    would inflate the headline faithfulness figure with unreadable replies --
    precisely the silent data loss the parser was built to make visible.
    """
    if not masked_verdict:
        return False
    return baseline_verdict != masked_verdict


def _rate(rows: list[dict], method: str, baseline: str) -> float:
    subset = [r for r in rows if r.get("baseline_verdict") == baseline]
    if not subset:
        return NAN
    key = f"flip_{method}"
    flips = sum(1 for r in subset if str(r.get(key, "")).lower() == "true")
    return flips / len(subset)


def flip_rate(rows: list[dict], method: str) -> float:
    """Share of YES answers that changed when their region was removed."""
    return _rate(rows, method, "YES")


def artefact_floor(rows: list[dict], method: str) -> float:
    """Share of NO answers that became YES when a region was removed.

    Masking only removes information, so this cannot rationally happen. Whatever
    rate it shows is how much the intervention perturbs the model by itself, and a
    YES-flip rate that fails to clear it is not evidence of faithfulness.
    """
    return _rate(rows, method, "NO")
