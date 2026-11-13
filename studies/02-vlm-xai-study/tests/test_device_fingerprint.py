"""Every model stage records WHICH GPU produced its rows. No GPU needed.

Greedy decoding is reproducible on one machine and not across machines: the
2026-11-09 parity run (same weights, code, data and pinned libraries, RTX 3070
vs A40) agreed on 96.9% of verdicts and only 42.7% of replies byte-for-byte.
So rows from two devices answer subtly different questions, and a resume that
silently stitches them together produces a file no later stage can untangle.
The device goes in the fingerprint so that resume refuses instead.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm.backends.fake import FakeBackend  # noqa: E402
from xai_vlm.runner import FingerprintMismatch, backend_device  # noqa: E402


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, STUDY_ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


baseline = _load("baseline_inference_dev", "02_baseline_inference.py")
stage4 = _load("stage4_dev", "04_descriptive_accuracy.py")
stage5 = _load("stage5_dev", "05_sparsity.py")
stage6 = _load("stage6_dev", "06_stability.py")


class _OtherDevice(FakeBackend):
    device_name = "NVIDIA A40"


def _files(tmp_path):
    out = []
    for name in ("m.csv", "b.csv", "r.csv"):
        p = tmp_path / name
        p.write_text(name, encoding="utf-8")
        out.append(p)
    return out


def test_fake_backend_names_its_device():
    assert backend_device(FakeBackend()) == "fake"


def test_a_backend_without_a_device_name_is_unknown_not_a_crash():
    assert backend_device(object()) == "unknown"


def test_stage2_records_the_device_in_its_state_file(tmp_path):
    from test_baseline_inference import _images, _manifest_file
    state = tmp_path / "s.json"
    baseline.run(manifest_path=_manifest_file(tmp_path, n=2), backend=FakeBackend(),
                 image_source=_images, out_csv=tmp_path / "o.csv", state_json=state,
                 model="fake", seed=42, limit=None, resume=True,
                 force_resume=False, checkpoint_every=100)
    assert json.loads(state.read_text(encoding="utf-8"))["device"] == "fake"


def test_stage2_refuses_to_resume_on_a_different_device(tmp_path):
    from test_baseline_inference import _images, _manifest_file
    kw = dict(manifest_path=_manifest_file(tmp_path, n=4), image_source=_images,
              out_csv=tmp_path / "o.csv", state_json=tmp_path / "s.json",
              model="fake", seed=42, resume=True, force_resume=False,
              checkpoint_every=100)
    baseline.run(backend=FakeBackend(), limit=2, **kw)
    with pytest.raises(FingerprintMismatch, match="device"):
        baseline.run(backend=_OtherDevice(), limit=None, **kw)


def test_stage4_fingerprint_includes_the_device(tmp_path):
    m, b, r = _files(tmp_path)
    assert stage4.build_fingerprint(m, b, r, "fake", 42, device="NVIDIA A40")["device"] \
        == "NVIDIA A40"


def test_stage5_fingerprint_includes_the_device(tmp_path):
    m, b, r = _files(tmp_path)
    fp = stage5.build_fingerprint(m, b, r, "fake", 42, device="NVIDIA A40")
    assert fp["device"] == "NVIDIA A40"
    assert fp["attention_impl_for_reading"] == "eager"


def test_stage6_fingerprint_includes_the_device(tmp_path):
    m, b, _ = _files(tmp_path)
    fp = stage6.build_fingerprint(m, b, "fake", 42, device="NVIDIA A40")
    assert fp["device"] == "NVIDIA A40"
    assert fp["n_reruns"] == stage6.N_RERUNS


def test_the_device_is_required_not_defaulted(tmp_path):
    # A default would let a stage forget to pass it and record a wrong device.
    m, b, r = _files(tmp_path)
    for call in (lambda: stage4.build_fingerprint(m, b, r, "fake", 42),
                 lambda: stage5.build_fingerprint(m, b, r, "fake", 42),
                 lambda: stage6.build_fingerprint(m, b, "fake", 42)):
        with pytest.raises(TypeError):
            call()
