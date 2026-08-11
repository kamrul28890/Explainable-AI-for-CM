"""Fetch the dataset and model weights needed to reproduce this work.

Neither is stored in git: the ConstructionSite snapshot is ~4.5 GB and the model
weights are several GB more. Both come from the Hugging Face Hub, so a fresh
machine needs only this script and an internet connection.

Usage
-----
    python scripts/download_data.py --what dataset
    python scripts/download_data.py --what models --tier auto
    python scripts/download_data.py --what all

The script is idempotent: already-downloaded files are skipped, and an
interrupted download resumes rather than restarting.

Model tier
----------
Which model weights to fetch depends on the machine, so `--tier auto` inspects
the hardware and picks:

    CUDA, >= 7 GB free VRAM   -> qwen7b-4bit   (primary; bitsandbytes quantises at load)
    CUDA, >= 5 GB free VRAM   -> qwen3b-8bit   (fallback)
    Apple Silicon             -> mlx-3b-4bit   (bitsandbytes is CUDA-only; MLX is the
                                                Apple-Silicon equivalent)
    CPU only                  -> qwen3b-8bit   (will be very slow; development only)

See docs/architecture/vlm-xai-study-architecture.md section 7 for why these tiers
were chosen, and docs/setup/SETUP.md for the per-platform install steps.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pilot_data() -> Path:
    """Dataset location, tolerating the pre/post-migration layout.

    The pilot folder is being renamed pilot/ -> studies/01-florence2-pilot/; this
    resolves whichever exists so the script works before and after that move.
    """
    for rel in ("studies/01-florence2-pilot/data/constructionsite",
                "pilot/data/constructionsite"):
        cand = REPO_ROOT / rel
        if cand.exists():
            return cand
    return REPO_ROOT / "studies/01-florence2-pilot/data/constructionsite"


DATASET_ID = "LouisChen15/ConstructionSite"
DATASET_DIR = _pilot_data()

# Pinned by repo id. Revisions are pinned in requirements/models.lock so a rerun
# months later gets identical weights -- an unpinned "latest" silently changes
# results and is the most common reproducibility failure in ML work.
MODELS = {
    "qwen7b-4bit": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3b-8bit": "Qwen/Qwen2.5-VL-3B-Instruct",
    "mlx-3b-4bit": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    "mlx-7b-4bit": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    "florence2": "microsoft/Florence-2-base-ft",
}
MODEL_DIR = REPO_ROOT / "models"

# Expected dataset shards, so a partial download is detected rather than silently
# producing a study run on 2 of 3 shards.
EXPECTED_SHARDS = ["test.parquet", "train-00001-of-00002.parquet", "train-00002-of-00002.parquet"]
EXPECTED_ROWS = {"test.parquet": 3004, "train-00001-of-00002.parquet": 3500,
                 "train-00002-of-00002.parquet": 3509}


def detect_tier() -> str:
    """Pick a model tier from the hardware actually present."""
    try:
        import torch
    except ImportError:
        print("  torch not installed yet; defaulting to qwen3b-8bit")
        return "qwen3b-8bit"

    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        free_gb = free_bytes / 1e9
        name = torch.cuda.get_device_name(0)
        print(f"  CUDA GPU: {name}, {free_gb:.1f} GB free")
        if free_gb >= 7.0:
            return "qwen7b-4bit"
        if free_gb >= 5.0:
            return "qwen3b-8bit"
        print("  WARNING: under 5 GB free. Close other GPU applications.")
        return "qwen3b-8bit"

    if platform.machine() == "arm64" and platform.system() == "Darwin":
        print("  Apple Silicon detected (bitsandbytes is CUDA-only; using MLX weights)")
        return "mlx-3b-4bit"

    print("  No GPU detected. CPU inference is far too slow for a full run.")
    return "qwen3b-8bit"


def download_dataset(force: bool = False) -> int:
    from huggingface_hub import snapshot_download

    if DATASET_DIR.exists() and not force:
        present = [p.name for p in DATASET_DIR.glob("*.parquet")]
        missing = [s for s in EXPECTED_SHARDS if s not in present]
        if not missing:
            print(f"  dataset already present at {DATASET_DIR}")
            return 0
        print(f"  incomplete dataset, missing: {missing} -- resuming")

    DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {DATASET_ID} (~4.5 GB) -> {DATASET_DIR}")
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=str(DATASET_DIR),
        allow_patterns=["*.parquet"],
    )
    return verify_dataset()


def verify_dataset() -> int:
    """Confirm every shard is present and has the expected row count."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("  pyarrow not installed; skipping row-count verification")
        return 0

    ok = True
    total = 0
    for shard in EXPECTED_SHARDS:
        path = DATASET_DIR / shard
        if not path.exists():
            print(f"  MISSING  {shard}")
            ok = False
            continue
        rows = pq.ParquetFile(str(path)).metadata.num_rows
        total += rows
        expected = EXPECTED_ROWS[shard]
        flag = "OK " if rows == expected else "BAD"
        if rows != expected:
            ok = False
        print(f"  {flag}  {shard:<32} {rows:>6,} rows (expected {expected:,})")

    print(f"  total: {total:,} rows (expected 10,013)")
    if not ok:
        print("  VERIFICATION FAILED -- re-run with --force")
        return 1
    print("  dataset verified")
    return 0


def download_models(tier: str) -> int:
    from huggingface_hub import snapshot_download

    if tier == "auto":
        tier = detect_tier()
    if tier not in MODELS:
        print(f"  unknown tier {tier!r}; choose from {sorted(MODELS)}")
        return 1

    repo = MODELS[tier]
    target = MODEL_DIR / tier
    if target.exists() and any(target.iterdir()):
        print(f"  {tier} already present at {target}")
        return 0

    target.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {repo} -> {target}")
    snapshot_download(repo_id=repo, local_dir=str(target))
    print(f"  {tier} ready")
    return 0


def report_disk() -> None:
    free_gb = shutil.disk_usage(REPO_ROOT).free / 1e9
    print(f"  free disk at {REPO_ROOT.drive or REPO_ROOT}: {free_gb:.1f} GB")
    if free_gb < 20:
        print("  WARNING: dataset + weights need roughly 15-20 GB.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", choices=["dataset", "models", "all", "verify"], default="all")
    ap.add_argument("--tier", default="auto",
                    help="model tier, or 'auto' to detect from hardware")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    print("=" * 72)
    print("ConstructionSite XAI -- data and model setup")
    print("=" * 72)
    report_disk()

    rc = 0
    if args.what == "verify":
        print("\n[verify dataset]")
        return verify_dataset()
    if args.what in ("dataset", "all"):
        print("\n[dataset]")
        rc |= download_dataset(force=args.force)
    if args.what in ("models", "all"):
        print("\n[models]")
        rc |= download_models(args.tier)

    print("\n" + "=" * 72)
    print("done" if rc == 0 else "FINISHED WITH ERRORS -- see above")
    print("=" * 72)
    return rc


if __name__ == "__main__":
    sys.exit(main())
