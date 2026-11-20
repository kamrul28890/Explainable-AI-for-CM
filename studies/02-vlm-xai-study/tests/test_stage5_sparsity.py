"""Stage 5 driver, over FakeBackend. No GPU."""
import importlib.util
import sys
from pathlib import Path

from PIL import Image

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm.backends.fake import FakeBackend  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "stage5", STUDY_ROOT / "scripts" / "05_sparsity.py")
stage5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage5)


def _rows():
    return [
        {"image_id": "0001", "rule_id": "rule_1", "verdict": "YES",
         "source": "model", "region": "(100.0, 100.0, 300.0, 400.0)",
         "image_w": "1200", "image_h": "900"},
        {"image_id": "0002", "rule_id": "rule_1", "verdict": "NO",
         "source": "none", "region": "", "image_w": "1200", "image_h": "900"},
    ]


def _images(_id):
    return Image.new("RGB", (1200, 900), "grey")


def test_work_fn_computes_gini_and_mass_inside():
    backend = FakeBackend()
    work = stage5.make_work_fn(backend, _images)
    out = work(_rows()[0])
    assert 0.0 <= out["gini"] <= 1.0
    assert out["attention_available"] == "True"


def test_a_none_attention_is_recorded_not_defaulted():
    """attention() returning None is legitimate; must not silently score as 0."""
    class NoAttentionBackend(FakeBackend):
        def attention(self, image, rule_id, **kw):
            return None

    work = stage5.make_work_fn(NoAttentionBackend(), _images)
    out = work(_rows()[0])
    assert out["attention_available"] == "False"
    assert out["gini"] == ""


def test_mass_inside_is_only_computed_when_a_region_exists():
    backend = FakeBackend()
    work = stage5.make_work_fn(backend, _images)
    out = work(_rows()[1])  # source == "none"
    assert out["mass_inside_region"] == ""
    assert out["region_area_fraction"] == ""


def test_a_missing_image_is_recorded_not_crashed():
    work = stage5.make_work_fn(FakeBackend(), lambda _id: None)
    out = work(_rows()[0])
    assert out["error"] == "image not found"


def test_grids_are_saved_only_when_asked(tmp_path):
    import numpy as np
    stage5.make_work_fn(FakeBackend(), _images)(_rows()[0])
    assert not list(tmp_path.iterdir())
    out = stage5.make_work_fn(FakeBackend(), _images, grid_dir=tmp_path)(_rows()[0])
    g = np.load(tmp_path / "0001_rule_1.npy")
    assert g.ndim == 2 and out["attention_available"] == "True"
