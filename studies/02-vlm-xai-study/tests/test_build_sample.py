"""Stage 1 tests. Metadata only -- no images, no model, no GPU."""
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "build_sample", STUDY_ROOT / "scripts" / "01_build_sample.py")
build_sample = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_sample)


def _records(n_violation=6, n_compliant=6):
    """Synthetic dataset rows with the columns Stage 1 reads."""
    recs = []
    for i in range(n_violation):
        recs.append({
            "image_id": f"v{i:04d}",
            "rule_1_violation": "no hard hat",
            "rule_2_violation": None, "rule_3_violation": None, "rule_4_violation": None,
            "quality_of_info": "rich info", "illumination": "daylight",
            "camera_distance": "medium", "view": "ground", "excavator": [],
        })
    for i in range(n_compliant):
        recs.append({
            "image_id": f"c{i:04d}",
            "rule_1_violation": None, "rule_2_violation": None,
            "rule_3_violation": None, "rule_4_violation": None,
            "quality_of_info": "rich info", "illumination": "daylight",
            "camera_distance": "medium", "view": "ground", "excavator": [],
        })
    return recs


def test_manifest_columns_are_exactly_the_pre_registered_schema():
    assert build_sample.MANIFEST_FIELDS == [
        "image_id", "rule_id", "stratum", "is_violation", "is_negative_control",
        "quality_of_info", "illumination", "camera_distance", "view",
    ]
    assert "expected" not in build_sample.MANIFEST_FIELDS


def test_same_seed_produces_an_identical_manifest():
    a = build_sample.build_rows(_records(), seed=42)
    b = build_sample.build_rows(_records(), seed=42)
    assert a == b


def test_different_seed_changes_the_manifest():
    a = build_sample.build_rows(_records(), seed=42)
    b = build_sample.build_rows(_records(), seed=7)
    assert a != b or len(a) == 0


def test_every_violation_image_contributes_its_violated_rule():
    rows = build_sample.build_rows(_records(), seed=42)
    v_rows = [r for r in rows if r["image_id"].startswith("v")]
    positive = [r for r in v_rows if r["is_violation"]]
    assert {r["image_id"] for r in positive} == {f"v{i:04d}" for i in range(6)}
    assert all(r["rule_id"] == "rule_1" for r in positive)
    assert all(r["stratum"] == "H" for r in v_rows)


def test_negative_controls_are_marked_and_are_not_violations():
    rows = build_sample.build_rows(_records(), seed=42)
    for r in rows:
        assert r["is_negative_control"] != r["is_violation"]


def test_compliant_rows_are_all_negative_controls():
    rows = build_sample.build_rows(_records(), seed=42)
    c_rows = [r for r in rows if r["stratum"] == "C"]
    assert c_rows
    assert all(r["is_negative_control"] for r in c_rows)
    assert not any(r["is_violation"] for r in c_rows)


def test_write_manifest_refuses_to_overwrite(tmp_path):
    path = tmp_path / "sample_manifest.csv"
    rows = build_sample.build_rows(_records(), seed=42)
    build_sample.write_manifest(rows, path, force=False)
    before = path.read_text(encoding="utf-8")

    with pytest.raises(SystemExit):
        build_sample.write_manifest(rows, path, force=False)
    assert path.read_text(encoding="utf-8") == before


def test_write_manifest_force_overwrites(tmp_path):
    path = tmp_path / "sample_manifest.csv"
    build_sample.write_manifest(build_sample.build_rows(_records(), seed=42), path, force=False)
    build_sample.write_manifest(build_sample.build_rows(_records(2, 2), seed=42), path, force=True)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) < 12


def test_scan_records_reads_every_split():
    """The pre-registered sample is the full corpus, not one split.

    Architecture doc 5.3 counts violations per split: test 436, train 891.
    Scanning only `test` would put rule_2 (n=25) and rule_4 (n=24) below the
    n=30 reporting floor and make two of the four rules unreportable.
    """
    seen = []

    def loader(split="test", streaming=True):
        seen.append(split)
        yield {"image_id": f"{split}-0001", "image": "PIXELS", "illumination": "daylight"}

    records = build_sample.scan_records(loader)

    assert seen == ["test", "train"]
    assert {r["image_id"] for r in records} == {"test-0001", "train-0001"}


def test_scan_records_drops_the_image_column():
    """Stage 1 is metadata only; carrying pixels would decode the whole corpus."""

    def loader(split="test", streaming=True):
        yield {"image_id": "0001", "image": "PIXELS", "illumination": "daylight"}

    records = build_sample.scan_records(loader, splits=("test",))
    assert "image" not in records[0]
    assert records[0]["illumination"] == "daylight"


# --- the complement: every compliant image the frozen sample left out ---------

def test_complement_is_every_compliant_image_the_sample_left_out():
    recs = _records(n_violation=6, n_compliant=20)
    sample = build_sample.build_rows(recs, seed=42)
    rest = build_sample.build_complement_rows(recs, sample, seed=42)

    in_sample = {r["image_id"] for r in sample}
    rest_ids = {r["image_id"] for r in rest}
    compliant = {r["image_id"] for r in recs if r["image_id"].startswith("c")}
    assert not rest_ids & in_sample
    assert rest_ids | (in_sample & compliant) == compliant
    assert len(rest_ids) == 14


def test_complement_asks_one_negative_control_per_image_under_the_same_policy():
    """Pooling with stratum C is only valid if the question policy is identical."""
    recs = _records(n_violation=6, n_compliant=20)
    sample = build_sample.build_rows(recs, seed=42)
    rest = build_sample.build_complement_rows(recs, sample, seed=42)
    assert len(rest) == len({r["image_id"] for r in rest})
    assert all(r["stratum"] == "R" for r in rest)
    assert all(r["is_negative_control"] and not r["is_violation"] for r in rest)
    by_id = {r["image_id"]: r for r in recs}
    for r in rest:
        [(rule, _)] = build_sample.rules_to_ask(by_id[r["image_id"]], seed=42)
        assert r["rule_id"] == rule


def test_complement_refuses_a_sample_that_does_not_match_the_records():
    """A manifest from a different dataset revision would make the complement
    silently overlap or miss images."""
    recs = _records(n_violation=6, n_compliant=20)
    sample = build_sample.build_rows(recs, seed=42)
    sample.append({**sample[-1], "image_id": "not-in-dataset"})
    with pytest.raises(ValueError, match="not-in-dataset"):
        build_sample.build_complement_rows(recs, sample, seed=42)


def test_complement_never_writes_the_frozen_manifest_path():
    assert build_sample.COMPLEMENT_NAME != "sample_manifest.csv"
