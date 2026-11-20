"""Merging the owner's rule_4 review into the first reader's answers. No GPU."""
import importlib.util
import json
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "s18", STUDY_ROOT / "scripts" / "18_rule4_adjudicate.py")
s18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s18)

FIRST = {"a": {"image_id": "a", "rule_4": "yes", "note": "first"},
         "b": {"image_id": "b", "rule_4": "unsure", "note": "first"},
         "c": {"image_id": "c", "rule_4": "no", "note": "first"}}
REVIEWS = {"a": {"final": {"rule_4": "no"}, "note": "far away"},
           "b": {"final": {"rule_4": "unsure"}, "note": ""}}


def test_owner_answer_wins_where_reviewed_and_first_reader_kept_elsewhere():
    out = s18.adjudicate(FIRST, REVIEWS)
    assert out["a"]["rule_4"] == "no" and out["a"]["source"] == "owner"
    assert out["a"]["first_reader"] == "yes" and out["a"]["note"] == "far away"
    assert out["b"]["rule_4"] == "unsure" and out["b"]["source"] == "owner"
    assert out["c"]["rule_4"] == "no" and out["c"]["source"] == "first reader, not reviewed"
    assert out["c"]["note"] == "first"


def test_changes_lists_only_differing_answers():
    ch = s18.changes(FIRST, REVIEWS)
    assert ch == [{"image_id": "a", "first": "yes", "owner": "no", "note": "far away"}]


def test_review_for_an_undrawn_image_is_refused():
    try:
        s18.adjudicate(FIRST, {"zz": {"final": {"rule_4": "yes"}}})
    except ValueError as e:
        assert "zz" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_main_writes_one_file_per_drawn_image(tmp_path):
    first, rev, out = tmp_path / "first", tmp_path / "rev", tmp_path / "out"
    first.mkdir(); rev.mkdir()
    for k, v in FIRST.items():
        (first / f"{k}.json").write_text(json.dumps(v), encoding="utf-8")
    for k, v in REVIEWS.items():
        (rev / f"{k}.json").write_text(json.dumps({"image_id": k, **v}), encoding="utf-8")
    assert s18.main(["--first", str(first), "--reviews", str(rev), "--out", str(out)]) == 0
    got = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in out.glob("*.json")}
    assert set(got) == {"a", "b", "c"} and got["a"]["rule_4"] == "no"
