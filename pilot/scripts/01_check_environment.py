"""Day 1 deliverable: confirm the environment is real, not assumed.

Loads Florence-2-base-ft on GPU, pulls one real image from the
ConstructionSite test split, and runs a captioning task on it.
Exits non-zero if any of those three things fail.
"""

import sys
import time

import torch
from datasets import load_dataset

from xai_pilot.config import HF_DATASET_ID
from xai_pilot.model import load_florence2, run_task


def main() -> int:
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available.")
        return 1

    print(f"Loading {HF_DATASET_ID} (test split)...")
    ds = load_dataset(HF_DATASET_ID, split="test", streaming=True)
    sample = next(iter(ds))
    image = sample["image"]
    print(f"Sample image_id={sample.get('image_id')}, size={image.size}")

    print("Loading Florence-2-base-ft...")
    model, processor = load_florence2()
    print(f"Model device: {model.device}, dtype: {model.dtype}")
    if model.device.type != "cuda":
        print("FAIL: model did not load onto cuda.")
        return 1

    print("Running <CAPTION> on the sample image...")
    t0 = time.perf_counter()
    raw_text, parsed, confidence = run_task(model, processor, image, "<CAPTION>")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"Caption: {parsed['<CAPTION>']}")
    print(f"Mean token probability (confidence proxy): {confidence:.4f}")
    print(f"Inference time: {elapsed_ms:.1f} ms")

    print("OK: environment check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
