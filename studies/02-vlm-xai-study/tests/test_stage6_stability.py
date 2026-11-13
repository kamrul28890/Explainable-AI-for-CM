"""Stage 6 driver, over FakeBackend. No GPU."""
import importlib.util
import sys
from pathlib import Path

from PIL import Image

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm.backends.fake import FakeBackend  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "stage6", STUDY_ROOT / "scripts" / "06_stability.py")
stage6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage6)


def _row():
    return {"image_id": "0001", "rule_id": "rule_1", "verdict": "YES",
           "image_w": "1200", "image_h": "900"}


def _images(_id):
    return Image.new("RGB", (1200, 900), "grey")


def test_work_fn_makes_n_reruns_all_sampled():
    backend = FakeBackend()
    work = stage6.make_work_fn(backend, _images, n_reruns=5)
    work(_row())
    calls = [c for c in backend.calls if c[0] == "answer"]
    assert len(calls) == 5
    assert all(c[2].get("do_sample") is True for c in calls)
    assert all(c[2].get("temperature") == stage6.STABILITY_TEMPERATURE for c in calls)


def test_sampling_truncation_is_pinned_not_left_to_each_model():
    """Llama 3.2 Vision ships top_p=0.9; the Qwen checkpoints ship none (so
    transformers' 1.0 / 50). Left to each model's generation_config, Llama's
    reruns would be drawn from a truncated distribution and look more stable.
    The values pinned are the ones the frozen Qwen runs already used."""
    backend = FakeBackend()
    stage6.make_work_fn(backend, _images, n_reruns=2)(_row())
    calls = [c for c in backend.calls if c[0] == "answer"]
    assert all(c[2].get("top_p") == 1.0 and c[2].get("top_k") == 50 for c in calls)


def test_work_fn_computes_agreement_and_presence():
    backend = FakeBackend(script={
        "rule_1": [
            "REASON: a.\nREGION: (1,1,2,2)\nANSWER: YES",
            "REASON: b.\nREGION: (1,1,2,2)\nANSWER: YES",
            "REASON: c.\nREGION: NONE\nANSWER: NO",
        ],
    })
    work = stage6.make_work_fn(backend, _images, n_reruns=3)
    out = work(_row())
    assert out["agreement_rate"] == "0.3333"
    assert out["region_presence_rate"] == "0.6667"
    assert out["flickering"] == "True"


def test_a_missing_image_is_recorded_not_crashed():
    work = stage6.make_work_fn(FakeBackend(), lambda _id: None, n_reruns=5)
    out = work(_row())
    assert out["error"] == "image not found"
