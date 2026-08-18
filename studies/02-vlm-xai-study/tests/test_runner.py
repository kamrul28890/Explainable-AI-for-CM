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
