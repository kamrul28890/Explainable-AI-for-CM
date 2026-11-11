"""Scoring the section 5.5 annotation audit. No GPU, no network."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "score", STUDY_ROOT / "scripts" / "13_audit_score.py")
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)

ALL = {"rule_1": True, "rule_2": True, "rule_3": True, "rule_4": False}


def _item(image_id, tier, annotated=()):
    return {"image_id": image_id, "quality_of_info": tier, "applicable": dict(ALL),
            "annotations": {r: {"reason": "", "boxes": [[0, 0, 1, 1]]} for r in annotated}}


def _answer(r1="no", r2="no", r3="no", r4="no", boxes=None):
    return {"blind": {"rule_1": r1, "rule_2": r2, "rule_3": r3, "rule_4": r4},
            "boxes": boxes or {}}


def test_questions_are_applicable_rules_only():
    qs = score.questions(_item("a", "rich info", ["rule_1"]), _answer(r4="yes"))
    assert [q["rule_id"] for q in qs] == ["rule_1", "rule_2", "rule_3"]
    assert qs[0] == {"image_id": "a", "tier": "rich info", "rule_id": "rule_1",
                     "annotated": True, "auditor": "no"}


def test_marked_violation_not_confirmed_and_missed_violation():
    """Two different errors, never pooled: a box marking a violation the
    auditor does not see, and a violation the annotation left unmarked.
    'Can't tell' counts in neither, and is reported on its own."""
    qs = (score.questions(_item("a", "rich info", ["rule_1"]), _answer(r1="no"))
          + score.questions(_item("b", "rich info", ["rule_1"]), _answer(r1="yes"))
          + score.questions(_item("c", "rich info", ["rule_1"]), _answer(r1="unsure"))
          + score.questions(_item("d", "rich info"), _answer(r2="yes")))
    rates = score.question_rates(qs)
    assert (rates["marked, not confirmed"]["k"], rates["marked, not confirmed"]["n"]) == (1, 2)
    assert (rates["unmarked, auditor sees one"]["k"], rates["unmarked, auditor sees one"]["n"]) == (1, 9)
    assert (rates["can't tell"]["k"], rates["can't tell"]["n"]) == (1, 12)


def test_an_image_disagrees_if_any_applicable_answer_contradicts_the_annotation():
    assert score.image_disagrees(_item("a", "x", ["rule_1"]), _answer(r1="yes")) is False
    assert score.image_disagrees(_item("a", "x", ["rule_1"]), _answer(r1="no")) is True
    assert score.image_disagrees(_item("b", "x"), _answer(r3="yes")) is True
    assert score.image_disagrees(_item("b", "x"), _answer(r4="yes")) is False   # not applicable
    assert score.image_disagrees(_item("b", "x"), _answer(r2="unsure")) is False


def test_corpus_rate_is_reweighted_from_the_stratified_draw_to_prevalence():
    """The draw is 50% hazard per tier; the corpus is not. A hazard-stratum
    disagreement rate of 40% and a compliant one of 10%, at 12.8% prevalence,
    is 0.128*0.4 + 0.872*0.1 for that tier -- not the 25% the raw draw gives."""
    got = score.reweight({"t": {"H": 0.4, "C": 0.1}}, {"t": {"share": 1.0, "prevalence": 0.128}})
    assert got == pytest.approx(0.128 * 0.4 + 0.872 * 0.1)
    two = score.reweight({"a": {"H": 0.4, "C": 0.1}, "b": {"H": 0.2, "C": 0.0}},
                         {"a": {"share": 0.5, "prevalence": 0.1},
                          "b": {"share": 0.5, "prevalence": 0.2}})
    assert two == pytest.approx(0.5 * (0.1 * 0.4 + 0.9 * 0.1) + 0.5 * (0.2 * 0.2))


def test_box_placement_counts_only_judged_boxes():
    ans = [_answer(boxes={"rule_1": "yes"}), _answer(boxes={"rule_1": "partly"}),
           _answer(boxes={"rule_1": "no"}), _answer(boxes={})]
    got = score.box_counts(ans)
    assert got == {"yes": 1, "partly": 1, "no": 1, "n": 3}


def test_answers_load_from_an_artifactdata_export(tmp_path):
    """`ArtifactData list --out_dir` writes audit/<uid>/images/<id>.json; the
    document may or may not be wrapped in a {data: ...} envelope."""
    d = tmp_path / "audit" / "u_x" / "images"
    d.mkdir(parents=True)
    (d / "0000072.json").write_text(json.dumps({"image_id": "0000072", **_answer()}))
    (d / "0000099.json").write_text(json.dumps({"data": {"image_id": "0000099", **_answer()},
                                                "version": 3}))
    got = score.load_answers(tmp_path)
    assert set(got) == {"u_x"}
    assert set(got["u_x"]) == {"0000072", "0000099"}
