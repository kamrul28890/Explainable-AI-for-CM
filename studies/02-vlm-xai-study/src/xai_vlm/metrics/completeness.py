"""Stage 9 -- Metric 6: Bounded Completeness.

No new model calls. Combines Stage 3 (was an explanation offered?) with Stage 4
(did it matter?).

`explanation_weak` deserves as much prominence as `explanation_supported`: the
model named a region, that region turned out not to matter, and a reader shown
only the region would have been misled about why the system decided what it did.
That is a worse failure than offering no explanation, because it looks like one.
"""

from __future__ import annotations

CATEGORIES = ("no_usable_explanation", "explanation_supported",
              "explanation_weak", "not_measured")

NAN = float("nan")


def classify_completeness(source: str, flip_tokens: str) -> str:
    """Three-way verdict, computed from the TOKEN-ABLATION flip.

    Token ablation is used rather than black fill so the headline is not inflated
    by the black-rectangle artefact.

    A region with no Stage 4 result is `not_measured`, never a default. Counting
    an unmeasured row as weak would understate faithfulness; counting it as
    supported would overstate it. Either would quietly convert missing data into
    a finding.
    """
    if source != "model":
        return "no_usable_explanation"
    f = str(flip_tokens).strip().lower()
    if f == "true":
        return "explanation_supported"
    if f == "false":
        return "explanation_weak"
    return "not_measured"


def classify_completeness_attention(source: str, sparsity_row: dict) -> str:
    """Three-way verdict from Stage 5 attention -- the F7 redesign.

    `explanation_supported` here means the model's attention put MORE mass inside
    the region it named than uniform attention would (mass > area fraction, the
    null F6 validated). That is an ATTENTIONAL claim -- "the model looked where it
    said" -- not the causal one the ablation basis makes ("removing the region
    changes the answer"). It replaces that basis because token ablation's
    artefact floor (51.7%) is nine times its signal (5.9%), so its flips cannot
    tell a region that mattered from one that did not (F4, F7). The change of
    meaning is deliberate and must be stated wherever the number is quoted.

    Missing or unavailable attention is `not_measured`, never a verdict.
    """
    if source != "model":
        return "no_usable_explanation"
    sp = sparsity_row or {}
    if str(sp.get("attention_available", "")).strip() != "True":
        return "not_measured"
    try:
        mass = float(sp["mass_inside_region"])
        area = float(sp["region_area_fraction"])
    except (KeyError, TypeError, ValueError):
        return "not_measured"
    return "explanation_supported" if mass > area else "explanation_weak"


def completeness_rates(rows: list[dict]) -> dict:
    """Share of rows in each category."""
    n = len(rows)
    out = {"n": n}
    for cat in CATEGORIES:
        out[cat] = (sum(1 for r in rows if r["completeness"] == cat) / n) if n else NAN
    return out
