"""Stage 2 -- baseline inference. The main measurement; everything reuses it.

Requires CUDA. Checkpoints every 100 samples and resumes, because at ~60 hours a
crash at hour eight without resumability costs a full day.

    python scripts/02_baseline_inference.py --model qwen7b-4bit --resume
"""
import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap rows, for smoke tests")
    ap.parse_args()

    print(__doc__)
    print("NOT YET IMPLEMENTED.\n")
    print("To implement, this stage must:")
    print("  1. load the manifest and the requested Backend")
    print("  2. for each image-rule pair: prompt_for(rule) -> backend.answer()")
    print("  3. parse with xai_vlm.parsing.parse_response, tally with ParseStats")
    print("  4. checkpoint every 100 rows; honour --resume")
    print("  5. write results/baseline_{model}.csv")
    print("  6. print the ParseStats summary -- ABORT if parse rate < 95%")
    print("\nCheck the negative controls FIRST. If the model answers YES to hazards")
    print("known to be absent, its accuracy on violations is meaningless -- it is")
    print("guessing, not detecting. That check precedes all other analysis.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
