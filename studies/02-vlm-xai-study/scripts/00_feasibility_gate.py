"""Stage 0 -- the feasibility gate. Run this before building anything else.

Purpose: kill or rescope the study cheaply. The decisive question is whether a
quantised model can still detect small hard hats -- if it cannot, rule_1 is invalid
and so is most of the study.

The stop rules are fixed in the architecture document (section 8, Stage 0) *before*
any numbers are seen. That is deliberate: writing them in advance prevents
rationalising a bad result afterwards.

    python scripts/00_feasibility_gate.py --tier auto

Requires CUDA. See docs/setup/SETUP.md, "Which machine runs what".
"""
import argparse
import sys

# Decision thresholds, quoted from the architecture document. Do not tune these to
# fit an observed result -- that is what they exist to prevent.
DETECTION_PROCEED = 0.80      # >= this: proceed
DETECTION_LIMITATION = 0.50   # >= this: proceed, record as a stated limitation
LATENCY_BUDGET_S = 6.0        # above this: rescope before committing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="auto", help="model tier, or 'auto' to detect")
    ap.add_argument("--n", type=int, default=30, help="images to test")
    ap.parse_args()

    print(__doc__)
    print("NOT YET IMPLEMENTED.\n")
    print("To implement, this stage must:")
    print("  1. load each candidate model at its target quantisation")
    print("  2. ask rule_1 on 30 hand-picked images")
    print("     (10 clear hard hats, 10 clear violations, 10 small/distant workers)")
    print("  3. report the small-hard-hat detection rate")
    print("  4. time 20 calls for a real seconds-per-call figure")
    print("  5. confirm attention can be extracted and image tokens ablated")
    print("  6. record peak VRAM at the chosen max_pixels")
    print(f"\nStop rules: detect >= {DETECTION_PROCEED:.0%} proceed; "
          f">= {DETECTION_LIMITATION:.0%} proceed with a stated limitation; "
          f"below that, STOP.")
    print(f"Latency above {LATENCY_BUDGET_S:.0f}s/call means rescope first.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
