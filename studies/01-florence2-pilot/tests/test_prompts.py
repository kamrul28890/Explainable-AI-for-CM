"""Unit tests for the concept/synonym query structure (Phase 1.5)."""

import pytest

from xai_pilot.prompts import (
    RULE_CONCEPTS,
    RULE_QUERIES,
    RULE_SATISFACTION,
    concepts_for,
    is_synonym,
    reworded_query,
    synonyms_for,
)


def test_frozen_rule_queries_unchanged():
    # The pilot's answer/rewording queries must be preserved byte-for-byte.
    assert RULE_QUERIES["rule_1"] == ["hard hat", "high-visibility vest"]
    assert RULE_QUERIES["rule_2"] == ["safety harness", "fall-protection lanyard"]
    assert RULE_QUERIES["rule_3"] == ["guardrail", "edge protection barrier"]


def test_rule_3_guardrail_and_edge_barrier_are_the_same_concept():
    # These are genuine synonyms -> one concept, two phrasings.
    assert is_synonym("rule_3", "guardrail", "edge protection barrier") is True
    assert concepts_for("rule_3") == ["guardrail"]


def test_rule_1_hard_hat_and_vest_are_distinct_concepts():
    # Distinct physical objects, NOT synonyms -- the pilot bug.
    assert is_synonym("rule_1", "hard hat", "high-visibility vest") is False
    assert set(concepts_for("rule_1")) == {"hard hat", "high-visibility vest"}


def test_rule_2_harness_and_lanyard_are_distinct_concepts():
    assert is_synonym("rule_2", "safety harness", "fall-protection lanyard") is False
    assert set(concepts_for("rule_2")) == {"safety harness", "fall-protection lanyard"}


def test_each_concept_lists_itself_as_its_first_synonym():
    for rule_id, concepts in RULE_CONCEPTS.items():
        for concept, syns in concepts.items():
            assert syns[0] == concept


def test_primary_concept_matches_frozen_query_index_0():
    # RULE_QUERIES[rule][0] must remain a real concept key so the frozen answer
    # query still resolves to a defined concept.
    for rule_id in ("rule_1", "rule_2", "rule_3"):
        assert RULE_QUERIES[rule_id][0] in RULE_CONCEPTS[rule_id]


def test_reworded_query_returns_same_concept_synonym_not_a_different_object():
    # The corrected rewording test must swap a validated synonym of the SAME
    # concept, never a different concept (the frozen index-1 behavior for
    # rule_1/rule_2 crossed concepts).
    reworded = reworded_query("rule_1", "hard hat")
    assert reworded != "hard hat"
    assert is_synonym("rule_1", "hard hat", reworded) is True


def test_reworded_query_raises_for_unknown_concept():
    with pytest.raises(ValueError):
        reworded_query("rule_1", "not a real concept")


def test_synonyms_for_includes_primary_and_at_least_one_alternate():
    syns = synonyms_for("rule_3", "guardrail")
    assert syns[0] == "guardrail"
    assert "edge protection barrier" in syns


def test_satisfaction_semantics_defined_for_every_rule():
    for rule_id in ("rule_1", "rule_2", "rule_3"):
        assert RULE_SATISFACTION[rule_id] in {"any", "all"}
