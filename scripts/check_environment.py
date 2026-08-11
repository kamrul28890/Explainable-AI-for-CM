"""Verify a machine is correctly set up, and report what it can and cannot run.

Run this first on any new computer. It answers, in one place, the question that
otherwise costs hours to discover the hard way: *which stages of this study can
this machine actually execute?*

    python scripts/check_environment.py

Exit code 0 means the environment is usable for at least analysis work. Exit code
1 means something required is missing or misconfigured.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

OK, WARN, BAD = "  OK  ", " WARN ", " FAIL "


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label:<34}{detail}")


def check_python() -> bool:
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro}  ({sys.executable})"
    if (v.major, v.minor) == (3, 10):
        line(OK, "python 3.10", detail)
        return True
    # Not fatal, but the pinned transformers + Florence-2 remote code are only
    # verified on 3.10; newer versions have broken the remote code before.
    line(WARN, "python version", detail + "  (3.10.x is the verified version)")
    return True


def check_in_venv() -> bool:
    inside = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if inside:
        line(OK, "virtual environment", sys.prefix)
    else:
        line(WARN, "virtual environment", "not in a venv -- see docs/setup/SETUP.md")
    return True


def check_packages() -> bool:
    required = ["numpy", "pandas", "pyarrow", "PIL", "matplotlib", "transformers",
                "huggingface_hub", "datasets"]
    optional = {
        "torch": "model inference",
        "bitsandbytes": "4/8-bit quantisation (CUDA only)",
        "sentence_transformers": "explanation-correctness grading (metric 7)",
        "cv2": "inpainting masks",
        "mlx": "Apple Silicon local smoke tests",
    }
    ok = True
    for mod in required:
        try:
            m = importlib.import_module(mod)
            line(OK, mod, getattr(m, "__version__", ""))
        except ImportError:
            line(BAD, mod, "MISSING -- pip install -r requirements/base.txt")
            ok = False
    for mod, why in optional.items():
        try:
            m = importlib.import_module(mod)
            line(OK, mod, f"{getattr(m, '__version__', '')}  ({why})")
        except ImportError:
            line(WARN, mod, f"absent -- {why} unavailable")
    return ok


def check_transformers_pin() -> bool:
    try:
        import transformers
    except ImportError:
        return False
    ver = transformers.__version__
    major = int(ver.split(".")[0])
    if major >= 5:
        line(BAD, "transformers pin", f"{ver} -- >=5.0 breaks Florence-2 remote code")
        return False
    has_qwen = hasattr(transformers, "Qwen2_5_VLForConditionalGeneration")
    line(OK if has_qwen else WARN, "transformers pin",
         f"{ver}  Qwen2.5-VL support: {'yes' if has_qwen else 'NO -- need >=4.49'}")
    return True


def check_compute() -> str:
    """Return the role this machine can play: 'compute', 'dev', or 'unusable'."""
    try:
        import torch
    except ImportError:
        line(WARN, "compute backend", "torch not installed -- analysis only")
        return "dev"

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        name = torch.cuda.get_device_name(0)
        line(OK, "CUDA", f"{name}  {free/1e9:.1f} GB free of {total/1e9:.1f} GB")
        if free / 1e9 < 5.0:
            line(WARN, "free VRAM", "under 5 GB -- close other GPU applications")
        return "compute"

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        line(OK, "Apple Metal (MPS)", "available")
        line(WARN, "role", "DEVELOPMENT machine -- see SETUP.md 'Which machine runs what'")
        return "dev"

    line(WARN, "compute backend", "CPU only -- far too slow for model inference")
    return "dev"


def check_dataset() -> bool:
    for rel in ("studies/01-florence2-pilot/data/constructionsite",
                "pilot/data/constructionsite"):
        d = REPO_ROOT / rel
        if d.exists():
            shards = sorted(p.name for p in d.glob("*.parquet"))
            if len(shards) == 3:
                line(OK, "dataset", f"{len(shards)} shards at {rel}")
            else:
                line(WARN, "dataset", f"{len(shards)}/3 shards -- run download_data.py")
            return True
    line(WARN, "dataset", "absent -- python scripts/download_data.py --what dataset")
    return True


def check_pilot_package() -> bool:
    try:
        import xai_pilot
        from xai_pilot.config import RESULTS_DIR
        line(OK, "xai_pilot importable", str(Path(xai_pilot.__file__).parent))
        if RESULTS_DIR.exists():
            n = len(list(RESULTS_DIR.glob("*.csv")))
            line(OK, "pilot results", f"{n} CSV files (analysis works without a GPU)")
        else:
            line(WARN, "pilot results", "RESULTS_DIR missing")
        return True
    except ImportError:
        line(WARN, "xai_pilot importable",
             "no -- pip install -e studies/01-florence2-pilot")
        return True


def main() -> int:
    print("=" * 78)
    print("Environment check")
    print("=" * 78)
    print(f"  {platform.system()} {platform.release()} / {platform.machine()}")
    print(f"  repo: {REPO_ROOT}")
    print(f"  free disk: {shutil.disk_usage(REPO_ROOT).free/1e9:.1f} GB")
    print("-" * 78)

    ok = True
    ok &= check_python()
    ok &= check_in_venv()
    print("-" * 78)
    ok &= check_packages()
    print("-" * 78)
    ok &= check_transformers_pin()
    role = check_compute()
    print("-" * 78)
    ok &= check_dataset()
    ok &= check_pilot_package()
    print("=" * 78)

    if role == "compute":
        print("ROLE: COMPUTE MACHINE -- can run every stage, including model inference.")
    else:
        print("ROLE: DEVELOPMENT MACHINE -- analysis, figures and writing.")
        print("      Run model-inference stages on the CUDA machine.")
        print("      See docs/setup/SETUP.md, 'Which machine runs what'.")

    print("PASS" if ok else "FAIL -- fix the [ FAIL ] lines above")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
