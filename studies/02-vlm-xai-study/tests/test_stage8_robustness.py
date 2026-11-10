"""Stage 8 driver, over FakeBackend. No GPU."""
import importlib.util
import sys
from pathlib import Path

from PIL import Image

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import perturbations as P  # noqa: E402
from xai_vlm.backends.fake import FakeBackend  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "stage8", STUDY_ROOT / "scripts" / "08_robustness.py")
stage8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage8)


def _row(verdict="YES", region="(100.0, 100.0, 200.0, 200.0)"):
    return {"image_id": "0001", "rule_id": "rule_1", "verdict": verdict,
            "region": region, "expected": "YES", "quality_of_info": "rich info"}


def _src(_id):
    return Image.new("RGB", (1000, 1000), "grey")


def _boxes(_id, _rule):
    return [[0.3, 0.3, 0.4, 0.4]]


YES_SAME = "REASON: a.\nREGION: (100,100,200,200)\nANSWER: YES"
YES_MOVED = "REASON: a.\nREGION: (600,600,700,700)\nANSWER: YES"
NO_NONE = "REASON: b.\nREGION: NONE\nANSWER: NO"


def test_each_perturbation_is_asked_exactly_once_per_row_greedy():
    backend = FakeBackend()
    stage8.make_work_fn(backend, _src, _boxes)(_row())
    calls = [c for c in backend.calls if c[0] == "answer"]
    assert len(calls) == len(P.PERTURBATIONS)
    assert all(not c[2].get("do_sample") for c in calls)   # greedy, like Stage 2


def test_the_model_sees_the_perturbed_image_not_the_original():
    seen = []

    class Spy(FakeBackend):
        def answer(self, image, rule_id, **kw):
            seen.append(image.copy())
            return super().answer(image, rule_id, **kw)

    img = Image.new("RGB", (200, 100), (200, 40, 40))
    stage8.make_work_fn(Spy(), lambda _i: img, _boxes)(_row())
    desat = seen[P.PERTURBATIONS.index("desaturate")]
    r, g, b = desat.getpixel((10, 10))
    assert r == g == b
    assert img.getpixel((10, 10)) == (200, 40, 40)          # source untouched


def test_survival_flip_and_drift_are_recorded_per_perturbation():
    script = {"rule_1": [YES_SAME, NO_NONE, YES_MOVED, YES_SAME, YES_SAME]}
    out = stage8.make_work_fn(FakeBackend(script=script), _src, _boxes)(_row())
    assert out["survived_blur"] == "True" and out["drift_blur"] == "0.0000"
    assert out["survived_low_light"] == "False" and out["drift_low_light"] == ""
    assert out["survived_occlusion"] == "True"
    # (150,150) -> (650,650) on a 1000x1000 image: 707/1414 = 0.5 of the diagonal.
    assert out["drift_occlusion"] == "0.5000"
    assert out["drifted_occlusion"] == "True"
    assert out["drifted_blur"] == "False"


def test_occluder_coverage_of_the_annotated_hazard_is_recorded():
    out = stage8.make_work_fn(FakeBackend(), _src, _boxes)(_row())
    assert out["occluder_coverage"] == "1.0000"
    out = stage8.make_work_fn(FakeBackend(), _src, lambda *_: [])(_row())
    assert out["occluder_coverage"] == ""


def test_a_missing_image_is_recorded_not_crashed():
    out = stage8.make_work_fn(FakeBackend(), lambda _i: None, _boxes)(_row())
    assert out["error"] == "image not found"


def test_row_ok_needs_every_perturbed_verdict():
    good = {f"verdict_{p}": "YES" for p in P.PERTURBATIONS}
    assert stage8.row_ok(good)
    assert not stage8.row_ok({**good, "verdict_blur": ""})


def test_fingerprint_pins_the_approved_strengths(tmp_path):
    for name in ("m.csv", "b.csv"):
        (tmp_path / name).write_text("x\n")
    fp = stage8.build_fingerprint(tmp_path / "m.csv", tmp_path / "b.csv", "qwen7b-4bit",
                                  42, device="fake")
    assert fp["strengths"] == P.STRENGTHS
    assert fp["perturbations"] == list(P.PERTURBATIONS)
    assert fp["subset_size"] == 1600 and fp["device"] == "fake"


def test_subset_is_1600_split_evenly_by_baseline_verdict():
    rows = [{"image_id": f"{v}{i}", "rule_id": f"rule_{1 + i % 4}", "verdict": v}
            for v in ("YES", "NO") for i in range(2000)]
    sub = stage8.build_subset(rows, seed=42)
    assert len(sub) == 1600
    assert sum(r["verdict"] == "YES" for r in sub) == 800
