"""Draw the stratified parity subset of the FROZEN manifest.

Up to 8 rows from every (rule, cell), cell = H-violation | H-control | C-control,
seed 42 -- 96 rows over 94 images. Rows keep the manifest's columns unchanged so
Stage 2 runs on them as-is. Written beside, never over, sample_manifest.csv.

    python cluster/parity/make_parity_manifest.py
"""
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[2] / "results"

m = pd.read_csv(RESULTS / "sample_manifest.csv", dtype=str, keep_default_na=False)
m["cell"] = m.stratum + "-" + m.is_violation
parts = [g.sample(n=min(8, len(g)), random_state=42)
         for _, g in m.groupby(["rule_id", "cell"])]
p = pd.concat(parts).sort_index().drop(columns="cell")
p.to_csv(RESULTS / "parity_manifest.csv", index=False)
print(len(p), "rows,", p.image_id.nunique(), "images")
print(p.groupby(["rule_id", "stratum", "is_violation"]).size().to_string())
