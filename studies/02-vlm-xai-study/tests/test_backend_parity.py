"""The parity check's own logic (scripts/20_backend_parity.py), without a GPU."""
import importlib.util

from xai_vlm.config import STUDY_ROOT

spec = importlib.util.spec_from_file_location(
    "parity", STUDY_ROOT / "scripts" / "20_backend_parity.py")
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


def _rows():
    return [{"image_id": f"{i:07d}", "rule_id": "rule_1",
             "region": "(1, 2, 3, 4)" if i % 3 else ""} for i in range(30)]


def test_rows_are_drawn_only_where_a_region_was_stated_and_reproducibly():
    a, b = parity.pick_rows(_rows(), 5), parity.pick_rows(list(reversed(_rows())), 5)
    assert a == b and len(a) == 5 and all(r["region"] for r in a)


def _rec(**kw):
    r = {"raw": "x", "verdict": "YES", "region": [1, 2, 3, 4], "black": "b",
         "tokens": "t", "attention": [[1.0, 0.5]], "patch_grid": [1, 2]}
    r.update(kw)
    return r


def test_identical_runs_pass():
    rep = parity.compare({"k": _rec()}, {"k": _rec()})
    assert rep["pass"] and rep["same"]["raw"] == 1 and rep["attention_max_abs_diff"] == 0


def test_any_differing_field_fails_and_is_named():
    rep = parity.compare({"k": _rec()}, {"k": _rec(tokens="other")})
    assert not rep["pass"] and rep["mismatches"] == [("k", "tokens")]


def test_attention_beyond_tolerance_or_missing_on_one_side_fails():
    assert not parity.compare({"k": _rec()}, {"k": _rec(attention=[[1.0, 0.4]])})["pass"]
    assert not parity.compare({"k": _rec()}, {"k": _rec(attention=None)})["pass"]
    assert parity.compare({"k": _rec(attention=None)}, {"k": _rec(attention=None)})["pass"]
