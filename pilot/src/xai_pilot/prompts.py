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
#
# FROZEN: preserved exactly so the pilot's answer query (index 0) and its Day-9
# rewording query (index 1) reproduce byte-for-byte. See RULE_CONCEPTS below for
# why index 1 is a *different concept* (not a synonym) for rules 1 and 2 -- the
# bug Phase 1.5 disentangles.
RULE_QUERIES: dict[RuleId, list[str]] = {
    "rule_1": ["hard hat", "high-visibility vest"],
    "rule_2": ["safety harness", "fall-protection lanyard"],
    "rule_3": ["guardrail", "edge protection barrier"],
}

# Phase 1.5: the reviewed concept/synonym structure. Each rule maps to one or
# more *concepts* (distinct physical objects); each concept maps to a list of
# *validated synonyms* (different phrasings of the SAME object). The rewording-
# robustness test may only swap synonyms within a concept -- never across
# concepts -- so it measures phrasing sensitivity, not object sensitivity.
#
# Note the disentanglement of the frozen RULE_QUERIES:
#   rule_1/rule_2: the two frozen entries are two DISTINCT concepts (hard hat vs
#     vest; harness vs lanyard). The pilot's index-1 "rewording" swapped objects.
#   rule_3: the two frozen entries ("guardrail", "edge protection barrier") are
#     genuine synonyms of ONE concept -- the only rule whose Day-9 rewording was
#     actually a rewording.
#
# Synonym lists beyond the frozen phrasings are proposed and marked for review;
# a synonym is "validated" when it grounds substantially the same boxes as the
# concept's primary phrasing (see the Phase 1.5 grounding-consistency check).
RULE_CONCEPTS: dict[RuleId, dict[str, list[str]]] = {
    "rule_1": {
        "hard hat": ["hard hat", "safety helmet"],
        "high-visibility vest": ["high-visibility vest", "hi-vis vest"],
    },
    "rule_2": {
        "safety harness": ["safety harness", "fall-arrest harness"],
        "fall-protection lanyard": ["fall-protection lanyard", "safety lanyard"],
    },
    "rule_3": {
        # Genuine synonyms of a single concept.
        "guardrail": ["guardrail", "edge protection barrier"],
    },
}

# Whether a rule is satisfied by the presence of ANY one concept or ALL of them.
# "all"  -- every listed concept must be present for compliance.
# "any"  -- at least one concept present satisfies the rule.
# These are proposed defaults for review; the pilot proxy approximated rule_1 by
# grounding only the "hard hat" concept (index 0), so its operative semantics
# were narrower than the "all" intent recorded here.
RULE_SATISFACTION: dict[RuleId, str] = {
    "rule_1": "all",   # basic PPE: both hard hat AND hi-vis vest expected
    "rule_2": "any",   # fall protection: a harness OR a lanyard suffices
    "rule_3": "any",   # edge protection: a guardrail (either phrasing) suffices
}


def rule_concepts(rule_id: RuleId) -> dict[str, list[str]]:
    """Return the {concept: [synonyms]} map for a rule."""
    return RULE_CONCEPTS[rule_id]


def concepts_for(rule_id: RuleId) -> list[str]:
    """Return the list of distinct concept names for a rule."""
    return list(RULE_CONCEPTS[rule_id].keys())


def synonyms_for(rule_id: RuleId, concept: str) -> list[str]:
    """Return a concept's validated synonyms (primary phrasing first)."""
    if concept not in RULE_CONCEPTS[rule_id]:
        raise ValueError(f"{concept!r} is not a concept of {rule_id}")
    return RULE_CONCEPTS[rule_id][concept]


def is_synonym(rule_id: RuleId, phrase_a: str, phrase_b: str) -> bool:
    """True if both phrases are synonyms of the same concept for this rule."""
    for syns in RULE_CONCEPTS[rule_id].values():
        if phrase_a in syns and phrase_b in syns:
            return True
    return False


def reworded_query(rule_id: RuleId, concept: str, variant_index: int = 1) -> str:
    """Return a validated synonym of `concept` for the rewording-robustness test.

    Unlike the frozen RULE_QUERIES index-1 phrasing (which for rules 1--2 is a
    different object entirely), this only ever returns another phrasing of the
    same concept, so a rewording test built on it measures phrasing sensitivity
    rather than object sensitivity. Raises if the concept is unknown or has no
    alternate synonym.
    """
    syns = synonyms_for(rule_id, concept)
    if variant_index >= len(syns):
        raise ValueError(f"concept {concept!r} of {rule_id} has no synonym at index {variant_index}")
    return syns[variant_index]

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

# Short region label used for each rule's queried safety object when boxes are
# fed to standardize_regions. Single source of truth: the metric scripts
# (04/05/07) previously duplicated this dict inline. It is also the object-class
# map for rule-aware region ranking (Scale-up Phase 1.1): a region carrying one
# of these labels is the rule's queried object and outranks the worker body.
RULE_OBJECT_LABEL: dict[RuleId, str] = {
    "rule_1": "hard hat",
    "rule_2": "harness",
    "rule_3": "guardrail",
    "rule_4": "excavator",
}


def rule_object_labels(rule_id: RuleId) -> set[str]:
    """Region labels that count as the rule's queried object for rule-aware ranking."""
    return {RULE_OBJECT_LABEL[rule_id]}


def build_grounding_query(rule_id: RuleId, phrasing_index: int = 0) -> str:
    """Return one open-vocabulary query for a presence-based rule.

    `phrasing_index=0` is the baseline wording. Index 1 is reserved for the
    Day 9 prompt-robustness test. Rule 4 uses two simultaneous object concepts
    and therefore cannot be represented by this single-phrase helper.
    """
    if rule_id == "rule_4":
        raise ValueError("rule_4 is proximity-based; use RULE_4_PROXIMITY_PAIR")
    return RULE_QUERIES[rule_id][phrasing_index]
