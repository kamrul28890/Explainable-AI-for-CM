"""Roster step 2 -- does the shared backend reproduce the frozen one on the GPU?

Runs the same rows through the frozen QwenBackend and through HFVLMBackend, one
after the other in one process on one GPU, and compares everything a stage reads:

* the reply to the clean image, its verdict and its parsed region;
* the reply with the region blacked out (Stage 4, black);
* the reply with the region's image tokens removed (Stage 4, tokens);
* the attention grid (Stage 5) and the patch grid.

Rows are drawn at random (seeded) from the rows with a stated region, without
regard to their score -- the lesson of the region-frame check (chapter s2-8).
Replies are compared as text, so "identical" means byte for byte; attention is
compared by its largest absolute difference.

    python scripts/20_backend_parity.py --tier qwen7b-4bit --n 24
"""
import argparse
import ast
import csv
import gc
import json
import random
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402

csv.field_size_limit(10 ** 8)
ATTN_TOL = 1e-6


def pick_rows(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """n rows with a stated region, drawn at random, in a fixed order."""
    with_region = sorted((r for r in rows if r.get("region", "").strip()),
                         key=lambda r: (r["image_id"], r["rule_id"]))
    return random.Random(seed).sample(with_region, min(n, len(with_region)))


def run_one(backend, rows, images) -> dict:
    out = {}
    for r in rows:
        img = images[r["image_id"]]
        region = tuple(float(v) for v in ast.literal_eval(r["region"]))
        a = backend.answer(img, r["rule_id"])
        b = backend.answer_masked(img, r["rule_id"], region, method="black")
        t = backend.answer_masked(img, r["rule_id"], region, method="tokens")
        g = backend.attention(img, r["rule_id"])
        out[f"{r['image_id']}|{r['rule_id']}"] = {
            "raw": a.raw_text, "verdict": a.verdict,
            "region": list(a.region) if a.region else None,
            "black": b.raw_text, "tokens": t.raw_text,
            "attention": None if g is None else g.tolist(),
            "patch_grid": list(backend.patch_grid(img)),
        }
        print(f"  {type(backend).__name__:<13} {r['image_id']} {r['rule_id']} {a.verdict}",
              flush=True)
    return out


def compare(old: dict, new: dict) -> dict:
    """Per-field counts of identical rows, and the worst attention difference."""
    import numpy as np
    fields = ("raw", "verdict", "region", "black", "tokens", "patch_grid")
    same = {f: 0 for f in fields}
    same["attention"], worst, bad = 0, 0.0, []
    for k in old:
        o, n = old[k], new[k]
        for f in fields:
            if o[f] == n[f]:
                same[f] += 1
            else:
                bad.append((k, f))
        if o["attention"] is None or n["attention"] is None:
            ok = o["attention"] is None and n["attention"] is None
            d = 0.0
        else:
            d = float(np.max(np.abs(np.array(o["attention"]) - np.array(n["attention"]))))
            ok = d <= ATTN_TOL
        worst = max(worst, d)
        same["attention"] += ok
        if not ok:
            bad.append((k, "attention"))
    return {"rows": len(old), "same": same, "attention_max_abs_diff": worst,
            "mismatches": bad, "pass": not bad}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="qwen7b-4bit")
    ap.add_argument("--baseline", default=None,
                    help="rows to draw from (default: the region-frame tree's Stage 2)")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import torch
    from xai_pilot.data import load_construction_site

    from xai_vlm.backends.hf import HFVLMBackend
    from xai_vlm.backends.qwen import QwenBackend

    base = Path(a.baseline or STUDY_ROOT / "results" / "scholar-a40-rf"
                / f"baseline_{a.tier}.csv")
    rows = pick_rows(list(csv.DictReader(base.open(newline="", encoding="utf-8"))),
                     a.n, a.seed)
    wanted = {r["image_id"] for r in rows}
    images = {}
    for split in config.SPLITS:
        for rec in load_construction_site(split=split, streaming=True):
            if rec["image_id"] in wanted:
                images[rec["image_id"]] = rec["image"].convert("RGB")
            if len(images) == len(wanted):
                break
        if len(images) == len(wanted):
            break
    print(f"{a.tier}: {len(rows)} rows over {len(images)} images, "
          f"XAI_REGION_FRAME={config.REGION_FRAME}")

    results = {}
    for cls in (QwenBackend, HFVLMBackend):
        b = cls(tier=a.tier)
        results[cls.__name__] = run_one(b, rows, images)
        device = b.device_name
        del b
        gc.collect()
        torch.cuda.empty_cache()

    rep = compare(results["QwenBackend"], results["HFVLMBackend"])
    rep.update(tier=a.tier, device=device, region_frame=config.REGION_FRAME,
               baseline=str(base), seed=a.seed)
    out = Path(a.out or config.RESULTS_DIR / f"backend_parity_{a.tier}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": rep, "rows": results}, indent=1), encoding="utf-8")

    print(f"\nrows {rep['rows']}  identical: " +
          "  ".join(f"{k} {v}" for k, v in rep["same"].items()))
    print(f"attention max |diff| {rep['attention_max_abs_diff']:.2e}")
    for k, f in rep["mismatches"]:
        print("  MISMATCH", k, f)
    print("BACKEND_PARITY", "PASS" if rep["pass"] else "FAIL")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
