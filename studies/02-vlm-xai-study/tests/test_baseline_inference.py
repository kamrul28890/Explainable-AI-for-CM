"""Stage 2 driver tests, over FakeBackend and a fake image source. No GPU."""
import csv
import importlib.util
import sys
from pathlib import Path

from PIL import Image

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm.backends.fake import FakeBackend  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "baseline_inference", STUDY_ROOT / "scripts" / "02_baseline_inference.py")
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)


def _manifest_file(tmp_path, n=6):
    path = tmp_path / "sample_manifest.csv"
    fields = ["image_id", "rule_id", "stratum", "is_violation",
              "is_negative_control", "quality_of_info", "illumination",
              "camera_distance", "view"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i in range(n):
            w.writerow({
                "image_id": f"{i:04d}", "rule_id": "rule_1", "stratum": "H",
                "is_violation": i % 2 == 0, "is_negative_control": i % 2 == 1,
                "quality_of_info": "rich info", "illumination": "daylight",
                "camera_distance": "medium", "view": "ground",
            })
    return path


def _images(_image_id):
    return Image.new("RGB", (64, 48), "grey")


def test_expected_is_derived_from_is_violation(tmp_path):
    manifest = _manifest_file(tmp_path, n=4)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    work = baseline.make_work_fn(FakeBackend(), _images)
    out = work(rows[0])
    assert out["expected"] == "YES"
    out = work(rows[1])
    assert out["expected"] == "NO"


def test_missing_image_is_recorded_not_crashed(tmp_path):
    manifest = _manifest_file(tmp_path, n=2)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    work = baseline.make_work_fn(FakeBackend(), lambda _id: None)
    out = work(rows[0])
    assert out["parse_error"] == "image not found"
    assert out["verdict"] == ""


def test_end_to_end_writes_every_manifest_row(tmp_path):
    manifest = _manifest_file(tmp_path, n=6)
    out_csv = tmp_path / "baseline_fake.csv"
    summary = baseline.run(
        manifest_path=manifest, backend=FakeBackend(), image_source=_images,
        out_csv=out_csv, state_json=tmp_path / "baseline_fake.state.json",
        model="fake", seed=42, limit=None, resume=True, force_resume=False,
        checkpoint_every=100,
    )
    written = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert len(written) == 6
    assert summary["errors"] == 0
    assert all(r["verdict"] in {"YES", "NO"} for r in written)


def test_limit_caps_rows_and_resume_continues_past_it(tmp_path):
    manifest = _manifest_file(tmp_path, n=6)
    out_csv = tmp_path / "baseline_fake.csv"
    state = tmp_path / "baseline_fake.state.json"
    kw = dict(manifest_path=manifest, backend=FakeBackend(), image_source=_images,
              out_csv=out_csv, state_json=state, model="fake", seed=42,
              resume=True, force_resume=False, checkpoint_every=100)

    baseline.run(limit=2, **kw)
    assert len(list(csv.DictReader(out_csv.open(encoding="utf-8")))) == 2

    baseline.run(limit=None, **kw)
    written = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert len(written) == 6
    assert len({r["image_id"] for r in written}) == 6


def test_dataset_is_streamed_exactly_once(tmp_path):
    manifest = _manifest_file(tmp_path, n=6)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    passes = []

    def loader(split="test", streaming=True):
        passes.append(1)
        for i in range(6):
            yield {"image_id": f"{i:04d}", "image": Image.new("RGB", (8, 8))}

    out = list(baseline.stream_rows_in_dataset_order(rows, loader))
    assert len(passes) == 1, "the split must be streamed once, not once per image"
    assert {r["image_id"] for r, _ in out} == {f"{i:04d}" for i in range(6)}


def test_rows_for_one_image_are_grouped(tmp_path):
    rows = [
        {"image_id": "0001", "rule_id": "rule_1"},
        {"image_id": "0002", "rule_id": "rule_1"},
        {"image_id": "0001", "rule_id": "rule_3"},
    ]

    def loader(split="test", streaming=True):
        for i in (1, 2):
            yield {"image_id": f"{i:04d}", "image": Image.new("RGB", (8, 8))}

    out = list(baseline.stream_rows_in_dataset_order(rows, loader))
    assert len(out) == 3
    assert [r["image_id"] for r, _ in out][:2] == ["0001", "0001"]


def test_every_split_is_streamed_not_just_test():
    """Stage 2 must cover the same splits Stage 1 sampled from.

    Regression: stream_rows_in_dataset_order hardcoded split="test" while Stage 1
    sampled both splits. Every train-split image was silently never found, its rows
    never ran, and the run exited 0 having covered a third of the manifest.
    """
    from xai_vlm import config

    rows = [{"image_id": "test-1", "rule_id": "rule_1"},
            {"image_id": "train-1", "rule_id": "rule_1"}]
    asked = []

    def loader(split="test", streaming=True):
        asked.append(split)
        yield {"image_id": f"{split}-1", "image": Image.new("RGB", (8, 8))}

    out = list(baseline.stream_rows_in_dataset_order(rows, loader))

    assert asked == list(config.SPLITS)
    assert {r["image_id"] for r, _ in out} == {"test-1", "train-1"}


def test_verify_coverage_counts_missing_rows(tmp_path):
    """A short run must be detectable from the artifacts alone."""
    manifest = _manifest_file(tmp_path, n=6)
    out_csv = tmp_path / "partial.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "rule_id"])
        w.writeheader()
        for i in range(4):
            w.writerow({"image_id": f"{i:04d}", "rule_id": "rule_1"})

    missing = baseline.verify_coverage(manifest, out_csv)
    assert missing == 2


def test_verify_coverage_is_zero_on_a_complete_run(tmp_path):
    manifest = _manifest_file(tmp_path, n=6)
    out_csv = tmp_path / "complete.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "rule_id"])
        w.writeheader()
        for i in range(6):
            w.writerow({"image_id": f"{i:04d}", "rule_id": "rule_1"})

    assert baseline.verify_coverage(manifest, out_csv) == 0
