"""Merging the owner's review into the first reader's audit answers. No GPU, no network."""
import importlib.util
import json
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "adj", STUDY_ROOT / "scripts" / "14_audit_adjudicate.py")
adj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adj)


def _first(r1="no", r3="no", boxes=None):
    return {"image_id": "a", "blind": {"rule_1": r1, "rule_3": r3}, "boxes": boxes or {}}


def test_reviewed_image_takes_the_owner_answer_and_keeps_boxes():
    out = adj.adjudicate({"a": _first(r1="no", boxes={"rule_1": "yes"})},
                         {"a": {"final": {"rule_1": "yes", "rule_3": "no"}, "note": "cap"}})
    assert out["a"]["blind"] == {"rule_1": "yes", "rule_3": "no"}
    assert out["a"]["boxes"] == {"rule_1": "yes"}
    assert out["a"]["source"] == "owner"
    assert out["a"]["note"] == "cap"


def test_unreviewed_image_keeps_the_first_reader_answer():
    out = adj.adjudicate({"a": _first(r1="yes")}, {})
    assert out["a"]["blind"] == {"rule_1": "yes", "rule_3": "no"}
    assert out["a"]["source"] == "first reader, not reviewed"


def test_owner_rule_missing_from_final_falls_back_to_first_reader():
    out = adj.adjudicate({"a": _first(r1="yes", r3="unsure")},
                         {"a": {"final": {"rule_1": "no"}}})
    assert out["a"]["blind"] == {"rule_1": "no", "rule_3": "unsure"}


def test_changes_lists_only_rules_the_owner_changed():
    ch = adj.changes({"a": _first(r1="no", r3="unsure")},
                     {"a": {"final": {"rule_1": "yes", "rule_3": "unsure"}, "note": "n"}})
    assert ch == [{"image_id": "a", "rule_id": "rule_1", "first": "no", "owner": "yes",
                   "note": "n"}]


def test_agreement_counts_reviewed_questions_only():
    first = {"a": _first(r1="no", r3="no"), "b": _first(r1="yes", r3="yes")}
    reviews = {"a": {"final": {"rule_1": "yes", "rule_3": "no"}}}
    assert adj.agreement(first, reviews) == {"agree": 1, "n": 2}


def test_write_export_matches_the_scorer_layout(tmp_path):
    adj.write_export({"a": {"image_id": "a", "blind": {"rule_1": "no"}}}, tmp_path, "owner")
    f = tmp_path / "audit" / "owner" / "images" / "a.json"
    assert json.loads(f.read_text(encoding="utf-8"))["blind"] == {"rule_1": "no"}
