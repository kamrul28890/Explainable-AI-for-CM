"""The rule_4 false-alarm audit: draw and score. No GPU, no network."""
import importlib.util
import math
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, STUDY_ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s15 = _load("s15", "15_rule4_audit_sample.py")
s16 = _load("s16", "16_rule4_audit_score.py")
T4, TB = s15.TIERS


def test_cells():
    assert s15.cell_of("C", "NO", "NO", "NO") == "C-NN"
    assert s15.cell_of("C", "NO", "NO", "YES") == "C-NY"
    assert s15.cell_of("C", "NO", "YES", "NO") == "C-Y"
    assert s15.cell_of("H", "NO", "YES", "YES") == "H-YY"
    assert s15.cell_of("H", "NO", "NO", "YES") == "H-NY"
    assert s15.cell_of("H", "YES", "NO", "YES") == "P-NY"
    assert s15.cell_of("H", "YES", "YES", "NO") == "P-Y"


def _base(n=40):
    rows4, rowsb = [], []
    for i in range(n):
        iid = f"{i:04d}"
        st, exp = ("C", "NO") if i < 20 else ("H", "YES" if i >= 35 else "NO")
        common = {"image_id": iid, "rule_id": "rule_4", "stratum": st, "expected": exp}
        rows4.append({**common, "verdict": "YES" if i % 3 == 0 else "NO"})
        rowsb.append({**common, "verdict": "YES" if i % 2 == 0 else "NO"})
        rows4.append({**common, "rule_id": "rule_1", "verdict": "NO"})
    return {T4: rows4, TB: rowsb}


def test_rule4_rows_keeps_rule4_only_and_both_verdicts():
    rows = s15.rule4_rows(_base())
    assert len(rows) == 40
    assert all(T4 in r and TB in r and r["cell"] for r in rows)


def test_draw_is_seeded_order_free_and_takes_small_cells_whole():
    rows = s15.rule4_rows(_base())
    a = s15.draw(rows, per_cell={"C-NN": 3, "C-Y": 100}, seed=42)
    b = s15.draw(list(reversed(rows)), per_cell={"C-NN": 3, "C-Y": 100}, seed=42)
    assert [r["image_id"] for r in a] == [r["image_id"] for r in b]
    cy = [r for r in rows if r["cell"] == "C-Y"]
    assert sum(r["cell"] == "C-Y" for r in a) == len(cy)
    assert all(r["cell_size"] == len(cy) for r in a if r["cell"] == "C-Y")
    assert sum(r["cell"] == "C-NN" for r in a) == 3


def test_blind_order_is_a_seeded_permutation():
    ids = [f"{i}" for i in range(30)]
    o = s15.blind_order(ids)
    assert sorted(o) == sorted(ids) and o != sorted(ids)
    assert o == s15.blind_order(list(reversed(ids)))


def _rows():
    # two cells: C (4-bit YES on both rows) and P (4-bit NO on both)
    return [{"image_id": "a", "stratum": "C", "expected": "NO", "cell": "C-Y", T4: "YES", TB: "NO"},
            {"image_id": "b", "stratum": "C", "expected": "NO", "cell": "C-Y", T4: "YES", TB: "NO"},
            {"image_id": "c", "stratum": "H", "expected": "YES", "cell": "P-NN", T4: "NO", TB: "NO"},
            {"image_id": "d", "stratum": "H", "expected": "YES", "cell": "P-NN", T4: "NO", TB: "NO"}]


def test_annotation_confusion():
    r = s16.confusion(_rows(), s16.annotation_p, T4)
    assert r["sensitivity"] == 0.0 and r["specificity"] == 0.0
    assert r["false_alarm_C"] == 1.0 and math.isnan(r["false_alarm_H"])


def test_audit_correction_turns_confirmed_false_alarms_into_hits():
    cells = s16.cell_answers(
        [{"image_id": i, "cell": c} for i, c in (("a", "C-Y"), ("b", "C-Y"), ("c", "P-NN"))],
        {"a": {"rule_4": "yes"}, "b": {"rule_4": "unsure"}, "c": {"rule_4": "no"}})
    assert cells["C-Y"] == {"yn": [True], "unsure": 1, "missing": 0}
    r = s16.confusion(_rows(), s16.audit_p(cells), T4)
    # C-Y rows are real (p=1): two hits; P-NN rows are not (p=0): two true negatives
    assert r["sensitivity"] == 1.0 and r["violations"] == 2.0
    assert r["specificity"] == 1.0
