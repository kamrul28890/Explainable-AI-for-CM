"""Scoring the models against the adjudicated audit labels. No GPU."""
import importlib.util
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

_spec = importlib.util.spec_from_file_location("s19", STUDY_ROOT / "scripts" / "19_audit_rescore.py")
s19 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s19)

ITEMS = [
    {"image_id": "a", "applicable": {"rule_1": True, "rule_4": False}, "annotations": ["rule_1"]},
    {"image_id": "b", "applicable": {"rule_1": True, "rule_3": True}, "annotations": []},
]
ANSWERS = {"a": {"blind": {"rule_1": "yes", "rule_4": "yes"}},
           "b": {"blind": {"rule_1": "yes", "rule_3": "unsure"}}}
ROWS = [
    {"image_id": "a", "rule_id": "rule_1", "verdict": "YES"},
    {"image_id": "a", "rule_id": "rule_4", "verdict": "YES"},   # not applicable
    {"image_id": "b", "rule_id": "rule_1", "verdict": "YES"},   # audit flips to positive
    {"image_id": "b", "rule_id": "rule_3", "verdict": "NO"},    # audit unsure
    {"image_id": "z", "rule_id": "rule_1", "verdict": "NO"},    # not in the audit
]


def test_pairs_keep_applicable_rules_with_a_model_answer():
    pairs = s19.pairs(ITEMS, ANSWERS, ROWS)
    got = {(p["image_id"], p["rule_id"]): p for p in pairs}
    assert set(got) == {("a", "rule_1"), ("b", "rule_1"), ("b", "rule_3")}
    assert got[("b", "rule_1")]["annotation"] is False and got[("b", "rule_1")]["audit"] is True
    assert got[("b", "rule_3")]["audit"] is None


def test_scores_against_each_reference_on_the_same_questions():
    pairs = s19.pairs(ITEMS, ANSWERS, ROWS)
    s = s19.score(pairs, "annotation")
    t = s19.score(pairs, "audit")
    # unsure (b, rule_3) is dropped from both, so both use the same 2 questions
    assert s["n"] == t["n"] == 2
    assert s["fp"] == 1 and s["tp"] == 1
    assert t["tp"] == 2 and t["fp"] == 0


def test_flips_record_which_reference_the_model_sides_with():
    f = s19.flips(s19.pairs(ITEMS, ANSWERS, ROWS))
    assert f == [{"image_id": "b", "rule_id": "rule_1", "annotation": False, "audit": True,
                  "model": True, "sides_with": "audit"}]
