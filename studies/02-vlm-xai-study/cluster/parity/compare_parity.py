"""Compare a tagged Stage 2 run against the frozen baseline, row by row.

    python cluster/parity/compare_parity.py scholar-a40-parity

Greedy decoding on identical weights, code, data and pinned libraries still
differs across GPUs: kernel selection changes bf16/nf4 rounding, and a near-tied
token flips. So exact reply, verdict and region agreement are reported
separately -- they fail differently. Raw text is compared after CRLF
normalisation, since a git checkout used to rewrite the frozen file's line
endings (fixed by .gitattributes in fbaa473).
"""
import ast
import statistics as st
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[2] / "results"
KEY = ["image_id", "rule_id"]


def _iou(a: str, b: str) -> float:
    a, b = ast.literal_eval(a), ast.literal_eval(b)
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main(tag: str) -> None:
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)  # noqa: E731
    frozen = read(RESULTS / "baseline_qwen7b-4bit.csv")
    new = read(RESULTS / f"baseline_qwen7b-4bit_{tag}.csv")
    m = new.merge(frozen, on=KEY, suffixes=("_new", "_frozen"), how="left", indicator=True)
    assert (m._merge == "both").all(), "rows missing from the frozen baseline"

    norm = lambda s: s.replace("\r\n", "\n").strip()  # noqa: E731
    print(f"rows compared        : {len(m)}")
    print(f"errors (new run)     : {(m.error_new != '').sum()}")
    text_eq = m.raw_text_new.map(norm) == m.raw_text_frozen.map(norm)
    print(f"raw_text identical   : {text_eq.sum()}/{len(m)} ({text_eq.mean():.1%})")
    for col in ("verdict", "region"):
        eq = m[f"{col}_new"] == m[f"{col}_frozen"]
        print(f"{col + ' identical':<21}: {eq.sum()}/{len(m)} ({eq.mean():.1%})")

    d = m[(m.region_new != m.region_frozen) & (m.region_new != "") & (m.region_frozen != "")]
    if len(d):
        ious = [_iou(a, b) for a, b in zip(d.region_new, d.region_frozen)]
        print(f"differing regions    : median IoU {st.median(ious):.2f}, "
              f">=0.5 in {sum(i >= .5 for i in ious)}/{len(ious)}")

    lat_n, lat_f = m.latency_ms_new.astype(float), m.latency_ms_frozen.astype(float)
    print(f"median latency       : new {lat_n.median() / 1000:.2f} s, "
          f"frozen {lat_f.median() / 1000:.2f} s")
    pos, neg = m[m.expected_new == "YES"], m[m.expected_new == "NO"]
    for v in ("new", "frozen"):
        print(f"{v:<7} sensitivity {(pos[f'verdict_{v}'] == 'YES').mean():.1%} (n={len(pos)})"
              f"   specificity {(neg[f'verdict_{v}'] == 'NO').mean():.1%} (n={len(neg)})")

    flips = m[m.verdict_new != m.verdict_frozen]
    if len(flips):
        print("\nverdict disagreements:")
        print(flips[KEY + ["expected_new", "verdict_frozen", "verdict_new"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
