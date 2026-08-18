"""Tests for the resumable runner. No GPU, no model -- FakeBackend only."""
import csv

from xai_vlm.runner import CheckpointWriter

FIELDS = ["image_id", "rule_id", "verdict"]


def test_done_keys_empty_when_no_file(tmp_path):
    w = CheckpointWriter(tmp_path / "out.csv", FIELDS)
    assert w.done_keys() == set()


def test_append_writes_header_once_and_rows_are_readable(tmp_path):
    path = tmp_path / "out.csv"
    w = CheckpointWriter(path, FIELDS)
    w.append({"image_id": "0001", "rule_id": "rule_1", "verdict": "YES"})
    w.append({"image_id": "0002", "rule_id": "rule_3", "verdict": "NO"})
    w.close()

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["image_id"] for r in rows] == ["0001", "0002"]
    assert path.read_text(encoding="utf-8").count("image_id,rule_id,verdict") == 1


def test_row_is_on_disk_before_the_next_append(tmp_path):
    """Durability: a crash after row N must not lose row N."""
    path = tmp_path / "out.csv"
    w = CheckpointWriter(path, FIELDS)
    w.append({"image_id": "0001", "rule_id": "rule_1", "verdict": "YES"})
    # Read with a separate handle while the writer is still open.
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 1


def test_done_keys_reads_existing_file_and_reopen_appends(tmp_path):
    path = tmp_path / "out.csv"
    w = CheckpointWriter(path, FIELDS)
    w.append({"image_id": "0001", "rule_id": "rule_1", "verdict": "YES"})
    w.close()

    w2 = CheckpointWriter(path, FIELDS)
    assert w2.done_keys() == {("0001", "rule_1")}
    w2.append({"image_id": "0002", "rule_id": "rule_1", "verdict": "NO"})
    w2.close()

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 2
    assert path.read_text(encoding="utf-8").count("image_id,rule_id,verdict") == 1


# --- fingerprinting --------------------------------------------------------
import json

import pytest

from xai_vlm.runner import FingerprintMismatch, RunState, manifest_sha256


def _fp(**over):
    fp = {"manifest_sha256": "abc", "model": "qwen7b-4bit", "max_pixels": 501760}
    fp.update(over)
    return fp


def test_manifest_sha256_is_stable_and_content_sensitive(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("image_id,rule_id\n0001,rule_1\n", encoding="utf-8")
    b.write_text("image_id,rule_id\n0001,rule_2\n", encoding="utf-8")
    assert manifest_sha256(a) == manifest_sha256(a)
    assert manifest_sha256(a) != manifest_sha256(b)


def test_first_run_writes_the_state_file(tmp_path):
    state = RunState(tmp_path / "run.state.json")
    state.verify_or_create(_fp())
    written = json.loads((tmp_path / "run.state.json").read_text(encoding="utf-8"))
    assert written["model"] == "qwen7b-4bit"


def test_matching_fingerprint_resumes_silently(tmp_path):
    state = RunState(tmp_path / "run.state.json")
    state.verify_or_create(_fp())
    RunState(tmp_path / "run.state.json").verify_or_create(_fp())  # must not raise


def test_changed_config_aborts_and_names_the_field(tmp_path):
    state = RunState(tmp_path / "run.state.json")
    state.verify_or_create(_fp())

    with pytest.raises(FingerprintMismatch) as exc:
        RunState(tmp_path / "run.state.json").verify_or_create(_fp(max_pixels=1003520))
    assert "max_pixels" in str(exc.value)


def test_changed_manifest_aborts(tmp_path):
    state = RunState(tmp_path / "run.state.json")
    state.verify_or_create(_fp())
    with pytest.raises(FingerprintMismatch):
        RunState(tmp_path / "run.state.json").verify_or_create(_fp(manifest_sha256="zzz"))


def test_force_overrides_and_records_that_it_was_forced(tmp_path):
    path = tmp_path / "run.state.json"
    RunState(path).verify_or_create(_fp())
    RunState(path).verify_or_create(_fp(max_pixels=1003520), force=True)

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["forced"] is True
    assert written["max_pixels"] == 1003520


# --- guard rails -----------------------------------------------------------
from xai_vlm.runner import GUARD_MIN_ROWS, GuardTripped, Guards


def test_guards_are_inert_below_the_minimum_row_count():
    """A --limit 20 smoke run must never trip a guard."""
    g = Guards()
    for _ in range(20):
        g.record(parsed_ok=False, latency_s=99.0)
    g.check()  # must not raise


def test_parse_rate_collapse_aborts_at_the_first_checkpoint():
    g = Guards()
    for i in range(GUARD_MIN_ROWS):
        g.record(parsed_ok=(i % 2 == 0), latency_s=3.0)  # 50%
    with pytest.raises(GuardTripped) as exc:
        g.check()
    assert "parse rate" in str(exc.value)


def test_healthy_run_does_not_trip():
    g = Guards()
    for _ in range(GUARD_MIN_ROWS):
        g.record(parsed_ok=True, latency_s=2.9)
    g.check()  # must not raise


def test_latency_spill_aborts():
    g = Guards()
    for _ in range(GUARD_MIN_ROWS):
        g.record(parsed_ok=True, latency_s=90.0)
    with pytest.raises(GuardTripped) as exc:
        g.check()
    assert "latency" in str(exc.value)


def test_latency_guard_tolerates_the_declared_budget():
    """6.0s is the pre-registered budget; it must not abort a run."""
    g = Guards()
    for _ in range(GUARD_MIN_ROWS):
        g.record(parsed_ok=True, latency_s=6.0)
    g.check()  # must not raise
