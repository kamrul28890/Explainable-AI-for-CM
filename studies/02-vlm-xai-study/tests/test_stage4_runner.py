"""Stage 4 driver, over FakeBackend. No GPU."""
import importlib.util
import sys
from pathlib import Path

from PIL import Image

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm.backends.fake import FakeBackend  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "stage4", STUDY_ROOT / "scripts" / "04_descriptive_accuracy.py")
stage4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage4)

REGION = "(100.0, 100.0, 300.0, 400.0)"


def _regions_rows():
    return [
        {"image_id": "0001", "rule_id": "rule_1", "verdict": "YES",
         "source": "model", "region": REGION, "image_w": "1200", "image_h": "900"},
        {"image_id": "0002", "rule_id": "rule_1", "verdict": "NO",
         "source": "model", "region": REGION, "image_w": "1200", "image_h": "900"},
        {"image_id": "0003", "rule_id": "rule_1", "verdict": "YES",
         "source": "none", "region": "", "image_w": "1200", "image_h": "900"},
    ]


def _images(_id):
    return Image.new("RGB", (1200, 900), "grey")


def test_maskable_rows_excludes_rows_with_no_region():
    rows = stage4.maskable_rows(_regions_rows())
    assert {r["image_id"] for r in rows} == {"0001", "0002"}


def test_maskable_rows_keeps_no_verdict_rows_as_the_control():
    """The NO rows are the artefact floor, not padding."""
    rows = stage4.maskable_rows(_regions_rows())
    assert any(r["verdict"] == "NO" for r in rows)


def test_both_methods_are_called_for_each_row():
    backend = FakeBackend()
    work = stage4.make_work_fn(backend, _images)
    work(stage4.maskable_rows(_regions_rows())[0])
    methods = [c[2] for c in backend.calls if c[0] == "answer_masked"]
    assert methods == ["black", "tokens"]


def test_flip_is_recorded_per_method():
    """Baseline YES, masked replies NO -> both flips True."""
    backend = FakeBackend(script={"rule_1": ["REASON: gone.\nREGION: NONE\nANSWER: NO"]})
    work = stage4.make_work_fn(backend, _images)
    out = work(stage4.maskable_rows(_regions_rows())[0])
    assert out["verdict_black"] == "NO"
    assert out["flip_black"] is True
    assert out["flip_tokens"] is True
    assert out["methods_disagree"] is False


def test_the_snapped_region_is_recorded():
    backend = FakeBackend()
    work = stage4.make_work_fn(backend, _images)
    out = work(stage4.maskable_rows(_regions_rows())[0])
    # snap_to_patch_grid expands outward to 28px blocks: 100 -> 84, 300 -> 308.
    assert out["region_snapped"] == "(84.0, 84.0, 308.0, 420.0)"


def test_a_missing_image_is_recorded_not_crashed():
    work = stage4.make_work_fn(FakeBackend(), lambda _id: None)
    out = work(stage4.maskable_rows(_regions_rows())[0])
    assert out["error"] == "image not found"


def test_fingerprint_includes_both_input_hashes(tmp_path):
    m = tmp_path / "m.csv"; m.write_text("a", encoding="utf-8")
    b = tmp_path / "b.csv"; b.write_text("b", encoding="utf-8")
    r = tmp_path / "r.csv"; r.write_text("c", encoding="utf-8")
    fp = stage4.build_fingerprint(m, b, r, "fake", 42)
    assert "baseline_sha256" in fp and "regions_sha256" in fp
    assert fp["masking_methods"] == "black,tokens"
