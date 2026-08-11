"""Tests for rule selection and two-stratum sampling. No GPU required."""

from collections import Counter

import pytest

from xai_vlm.rules import (RULES, expected_answer, prompt_for, rule_is_applicable,
                           rules_to_ask, violated_rules)
from xai_vlm.sampling import (match_compliant_sample, metadata_key,
                              reweight_precision, stratification_report)


def row(image_id="0000001", violations=(), excavator=(), **meta):
    r = {"image_id": image_id, "excavator": list(excavator), "rebar": [],
         "quality_of_info": "rich info", "illumination": "daylight",
         "camera_distance": "medium", "view": "ground"}
    r.update(meta)
    for rid in ("rule_1", "rule_2", "rule_3", "rule_4"):
        r[f"{rid}_violation"] = ({"bounding_box": [[0, 0, 1, 1]], "reason": "x"}
                                 if rid in violations else None)
    return r


# ---------------------------------------------------------------------- rules
def test_prompt_includes_question_and_format():
    p = prompt_for("rule_1")
    assert "personal protective equipment" in p
    assert "ANSWER:" in p and "REGION:" in p and "REASON:" in p


def test_rule_1_covers_ppe_broadly_not_just_hard_hats():
    # The pilot grounded only "hard hat", making 16.9% of real PPE violations
    # undetectable by construction. The question must be broader than that.
    q = RULES["rule_1"].question.lower()
    assert "hard hat" in q
    assert "high-visibility" in q
    assert "body covering" in q


def test_violated_rules_detects_all_of_them():
    assert violated_rules(row(violations=("rule_1", "rule_3"))) == ["rule_1", "rule_3"]
    assert violated_rules(row()) == []


def test_every_violated_rule_is_asked_no_priority_collapse():
    # The pilot kept only the highest-priority violation and discarded the rest.
    asked = rules_to_ask(row(violations=("rule_1", "rule_3")))
    asked_ids = {r for r, _ in asked}
    assert {"rule_1", "rule_3"} <= asked_ids


def test_a_negative_control_is_included():
    asked = rules_to_ask(row(violations=("rule_1",)))
    assert any(is_neg for _, is_neg in asked), "no negative control added"
    assert not any(r == "rule_1" and is_neg for r, is_neg in asked)


def test_rule_4_not_asked_as_control_without_machinery():
    # Asking about excavator proximity with no excavator present cannot be failed,
    # so it inflates accuracy without measuring anything.
    assert not rule_is_applicable("rule_4", row())
    assert rule_is_applicable("rule_4", row(excavator=[[0, 0, 1, 1]]))

    for i in range(25):
        asked = rules_to_ask(row(image_id=f"img{i}"))
        assert not any(r == "rule_4" and neg for r, neg in asked)


def test_rule_4_is_available_when_machinery_present():
    seen = set()
    for i in range(60):
        asked = rules_to_ask(row(image_id=f"m{i}", excavator=[[0, 0, 1, 1]]))
        seen.update(r for r, _ in asked)
    assert "rule_4" in seen


def test_compliant_image_still_gets_a_question():
    asked = rules_to_ask(row())
    assert len(asked) >= 1
    assert all(neg for _, neg in asked)


def test_rule_selection_is_deterministic_for_the_same_image():
    a = rules_to_ask(row(image_id="fixed"))
    b = rules_to_ask(row(image_id="fixed"))
    assert a == b


def test_expected_answer_matches_annotations():
    assert expected_answer("rule_1", row(violations=("rule_1",))) == "YES"
    assert expected_answer("rule_2", row(violations=("rule_1",))) == "NO"


# ------------------------------------------------------------------- sampling
def test_metadata_key_uses_the_matching_fields():
    assert metadata_key(row(illumination="night")) == ("night", "medium", "ground")


def test_matching_follows_the_target_distribution():
    target = Counter({("night", "medium", "ground"): 75,
                      ("daylight", "medium", "ground"): 25})
    pool = ([row(f"n{i}", illumination="night") for i in range(200)] +
            [row(f"d{i}", illumination="daylight") for i in range(200)])

    picked = match_compliant_sample(pool, target, n_wanted=100, seed=1)
    assert len(picked) == 100
    night = sum(1 for r in picked if r["illumination"] == "night")
    assert 70 <= night <= 80, f"expected ~75 night images, got {night}"


def test_shortfall_is_backfilled_rather_than_returning_too_few():
    # A cell common among violations may be rare among compliant images. Silently
    # returning fewer rows would unbalance the strata and bias specificity.
    target = Counter({("night", "medium", "ground"): 90,
                      ("daylight", "medium", "ground"): 10})
    pool = ([row(f"n{i}", illumination="night") for i in range(5)] +
            [row(f"d{i}", illumination="daylight") for i in range(200)])

    picked = match_compliant_sample(pool, target, n_wanted=100, seed=1)
    assert len(picked) == 100


def test_matching_is_reproducible():
    target = Counter({("daylight", "medium", "ground"): 10})
    pool = [row(f"d{i}") for i in range(100)]
    a = match_compliant_sample(pool, target, 20, seed=7)
    b = match_compliant_sample(pool, target, 20, seed=7)
    assert [r["image_id"] for r in a] == [r["image_id"] for r in b]


def test_report_flags_a_badly_matched_pair():
    hazard = [row(f"h{i}", illumination="night") for i in range(50)]
    compliant = [row(f"c{i}", illumination="daylight") for i in range(50)]
    assert "WARNING" in stratification_report(hazard, compliant)


def test_report_is_quiet_when_matching_is_good():
    hazard = [row(f"h{i}", illumination="night") for i in range(50)]
    compliant = [row(f"c{i}", illumination="night") for i in range(50)]
    assert "WARNING" not in stratification_report(hazard, compliant)


# --------------------------------------------------------------- reweighting
def test_reweighting_reproduces_the_documented_example():
    # Architecture doc section 5.3: the balanced figure overstates deployed
    # precision by 35 points.
    balanced = reweight_precision(0.80, 0.90, prevalence=0.5)
    deployed = reweight_precision(0.80, 0.90, prevalence=0.128)
    assert balanced == pytest.approx(0.889, abs=0.001)
    assert deployed == pytest.approx(0.540, abs=0.001)


def test_precision_falls_as_hazards_get_rarer():
    vals = [reweight_precision(0.8, 0.9, p) for p in (0.5, 0.25, 0.128, 0.05)]
    assert vals == sorted(vals, reverse=True)


def test_perfect_specificity_gives_perfect_precision():
    assert reweight_precision(0.8, 1.0, 0.128) == pytest.approx(1.0)


def test_no_positive_predictions_gives_nan():
    v = reweight_precision(0.0, 1.0, 0.128)
    assert v != v      # NaN
