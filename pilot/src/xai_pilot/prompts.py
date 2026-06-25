"""Safety-rule -> Florence-2 grounding query mapping.

Florence-2-base-ft has no free-form VQA task token, so each safety rule is
answered by grounding a phrase instead of asking a yes/no question. Rules 1-3
are presence checks (does the relevant safety object exist in the image);
rule 4 is a proximity check between two independently grounded objects.
"""

from typing import Literal

RuleId = Literal["rule_1", "rule_2", "rule_3", "rule_4"]

# Two phrasings per rule (kept short -- Florence-2's open-vocabulary
# detection works best with short noun phrases, not full sentences).
RULE_QUERIES: dict[RuleId, list[str]] = {
    "rule_1": ["hard hat", "high-visibility vest"],
    "rule_2": ["safety harness", "fall-protection lanyard"],
    "rule_3": ["guardrail", "edge protection barrier"],
}

# Shared "person" phrase: rules 1-3 check each detected worker individually
# against the relevant safety object (see inference.py), and rule 4 grounds
# it independently against "excavator" for a proximity check.
PERSON_PHRASE = "worker"

# Rule 4 is proximity-based: it grounds these two phrases independently and
# checks spatial overlap/distance rather than presence of either alone.
RULE_4_PROXIMITY_PAIR = (PERSON_PHRASE, "excavator")

RULE_DESCRIPTIONS: dict[RuleId, str] = {
    "rule_1": "basic PPE (hard hat / high-visibility vest)",
    "rule_2": "fall protection (harness/lanyard) when working at height",
    "rule_3": "edge protection (guardrails) at height or excavation edges",
    "rule_4": "worker proximity to excavator operating radius / blind spot",
}


def build_grounding_query(rule_id: RuleId, phrasing_index: int = 0) -> str:
    """Return one open-vocabulary query for a presence-based rule.

    `phrasing_index=0` is the baseline wording. Index 1 is reserved for the
    Day 9 prompt-robustness test. Rule 4 uses two simultaneous object concepts
    and therefore cannot be represented by this single-phrase helper.
    """
    if rule_id == "rule_4":
        raise ValueError("rule_4 is proximity-based; use RULE_4_PROXIMITY_PAIR")
    return RULE_QUERIES[rule_id][phrasing_index]
