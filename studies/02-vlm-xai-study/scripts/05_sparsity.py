"""Stage 5 -- Metric 2: Sparsity. Was attention concentrated, or smeared everywhere?

For each baseline row, reads attention over the image-patch grid and computes the
Gini coefficient (primary, size-independent) and the share of attention mass
falling inside the model's own stated region (an honesty cross-check that needs
no masking at all).

Requires CUDA. Resumable: rerun the same command after any interruption.

    python scripts/05_sparsity.py --model qwen7b-4bit --resume
"""
import argparse
import ast
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402
from xai_vlm.metrics.sparsity import gini, mass_inside_region, region_area_fraction  # noqa: E402
from xai_vlm.runner import Guards, manifest_sha256, run_stage  # noqa: E402

RESULT_FIELDS = [
    "image_id", "rule_id", "verdict", "attention_available", "gini",
    "mass_inside_region", "region_area_fraction", "latency_ms", "error",
]


def _parse_box(text: str):
    try:
        v = ast.literal_eval((text or "").strip())
        return tuple(float(x) for x in v) if len(v) == 4 else None
    except (ValueError, SyntaxError, TypeError):
        return None


def make_work_fn(backend, image_source):
    """One region row -> one sparsity result row."""

    def work(row: dict) -> dict:
        out = {
            "image_id": row["image_id"], "rule_id": row["rule_id"],
            "verdict": row.get("verdict", ""), "attention_available": "",
            "gini": "", "mass_inside_region": "", "region_area_fraction": "",
            "latency_ms": "", "error": "",
        }

        image = image_source(row["image_id"])
        if image is None:
            out["error"] = "image not found"
            return out

        # The runner derives its latency guard from this field, and that guard is
        # the only detector for the silent shared-memory spill: peak VRAM here is
        # 7.29 GB against a 7.0 GB budget, and Windows serves the overflow from
        # system RAM at ~30x cost rather than raising OOM. Omitting it does not
        # disable the guard visibly -- it reports a healthy 0.0 s forever.
        started = time.perf_counter()

        # Legitimate: not every backend exposes attention. Recorded as
        # unavailable, never defaulted to zero -- a false "unfocused" reading.
        grid = backend.attention(image, row["rule_id"])

        # Timed before the early return: a backend that returns no attention has
        # still paid for the forward pass, and recording 0 ms there would drag
        # the median down and hide a real slowdown.
        out["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 1)

        if grid is None:
            out["attention_available"] = "False"
            return out
        out["attention_available"] = "True"
        out["gini"] = round(gini(grid), 4)

        box = _parse_box(row.get("region", "")) if row.get("source") == "model" else None
        if box is not None:
            w = int(row.get("image_w") or 0)
            h = int(row.get("image_h") or 0)
            if w and h:
                out["mass_inside_region"] = round(mass_inside_region(grid, box, (w, h)), 4)
                out["region_area_fraction"] = round(region_area_fraction(box, (w, h)), 4)
        return out

    return work


def _progress(summary: dict) -> None:
    g = summary["guards"]
    print(f"  {summary['done']:>6} done  {summary['skipped']:>6} skipped  "
          f"{summary['errors']:>4} errors  |  median {g.median_latency_s:.1f}s")


def stream_rows_in_dataset_order(work_rows, loader):
    by_image = defaultdict(list)
    for row in work_rows:
        by_image[row["image_id"]].append(row)
    for split in config.SPLITS:
        if not by_image:
            return
        for rec in loader(split=split, streaming=True):
            rows = by_image.pop(rec["image_id"], None)
            if not rows:
                continue
            image = rec["image"]
            image = image.convert("RGB") if hasattr(image, "convert") else image
            for row in rows:
                yield row, image
            if not by_image:
                return


def verify_coverage(work_rows, out_csv) -> int:
    wanted = {(r["image_id"], r["rule_id"]) for r in work_rows}
    out_csv = Path(out_csv)
    if not out_csv.exists():
        return len(wanted)
    with out_csv.open(newline="", encoding="utf-8") as fh:
        got = {(r["image_id"], r["rule_id"]) for r in csv.DictReader(fh)}
    return len(wanted - got)


def _summarise(out_csv: Path) -> None:
    rows = list(csv.DictReader(Path(out_csv).open(encoding="utf-8")))
    if not rows:
        print("no rows written")
        return
    available = [r for r in rows if r["attention_available"] == "True"]
    print("\n" + "=" * 66)
    print(f"attention available : {len(available)}/{len(rows)} "
          f"({len(available) / len(rows):.1%})")
    if available:
        ginis = sorted(float(r["gini"]) for r in available)
        n = len(ginis)
        print(f"Gini  median {ginis[n // 2]:.3f}   "
              f"p25 {ginis[int(n * .25)]:.3f}   p75 {ginis[int(n * .75)]:.3f}")
        withmass = [r for r in available if r["mass_inside_region"]]
        if withmass:
            masses = [float(r["mass_inside_region"]) for r in withmass]
            areas = [float(r["region_area_fraction"]) for r in withmass]
            print(f"mass-inside-region  median {sorted(masses)[len(masses)//2]:.3f}  "
                  f"(n={len(withmass)})")
            print(f"region area fraction  median {sorted(areas)[len(areas)//2]:.3f}  "
                  "-- the null for mass-inside-region under uniform attention")
    print(f"errors : {sum(1 for r in rows if r.get('error'))}")
    print("=" * 66)


# MEASURED ON THIS STAGE, not inherited. Reading attention forces
# `output_attentions=True`, which makes transformers fall back from sdpa to the
# manual attention implementation, so a row here costs far more than a Stage 2
# answer. Smoke-measured median over 30 rows: 15.4 / 13.8 / 16.0 s.
MEASURED_MEDIAN_LATENCY_S = 16.0

# The runner's 25.0 s ceiling is 2.3x Stage 2's 9-11 s median. Against 16 s it is
# only 1.56x, which would abort a 17 h run on ordinary variation -- M2 again, a
# threshold calibrated on an unrepresentative workload. This keeps Stage 2's
# multiple against the median this stage actually measured.
LATENCY_CEILING_S = 37.0


def make_guards() -> Guards:
    return Guards(latency_ceiling_s=LATENCY_CEILING_S)


def row_ok(out_row: dict) -> bool:
    """This stage's analog of "the reply parsed": attention actually came back.

    The default guard reads `verdict`, which this stage copies in from the
    regions CSV -- so it would grade Stage 3's output and pass on every row even
    if attention extraction failed for all of them.
    """
    return out_row.get("attention_available") == "True"


def run(*, work_rows, backend, image_source, out_csv, state_json, fingerprint,
        resume, force_resume, checkpoint_every, rows=None) -> dict:
    return run_stage(
        rows if rows is not None else work_rows,
        make_work_fn(backend, image_source),
        Path(out_csv), Path(state_json), RESULT_FIELDS, fingerprint,
        resume=resume, force_resume=force_resume,
        checkpoint_every=checkpoint_every, progress_fn=_progress,
        guards=make_guards(), ok_fn=row_ok,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(config.MODEL_TIERS))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    res = STUDY_ROOT / "results"
    manifest = res / "sample_manifest.csv"
    baseline = res / f"baseline_{args.model}.csv"
    regions = res / f"regions_{args.model}.csv"
    for p in (manifest, baseline, regions):
        if not p.exists():
            print(f"missing {p}. Run the earlier stages first.", file=sys.stderr)
            return 1

    with regions.open(newline="", encoding="utf-8") as fh:
        work_rows = list(csv.DictReader(fh))
    if args.limit is not None:
        work_rows = work_rows[:args.limit]
    print(f"rows: {len(work_rows)}")

    out_csv = res / f"sparsity_{args.model}.csv"
    state_json = res / f"sparsity_{args.model}.state.json"
    fingerprint = {
        "manifest_sha256": manifest_sha256(manifest),
        "baseline_sha256": manifest_sha256(baseline),
        "regions_sha256": manifest_sha256(regions),
        "model": args.model,
        "compute_dtype": config.COMPUTE_DTYPE,
        "attention_impl_for_reading": config.ATTENTION_IMPL_FOR_READING,
        "max_pixels": config.MAX_PIXELS,
        "seed": args.seed,
    }

    from xai_pilot.data import load_construction_site
    from xai_vlm.backends.qwen import QwenBackend

    print(f"loading {args.model} ...")
    backend = QwenBackend(tier=args.model)

    cache: dict = {}

    def image_source(image_id):
        return cache.get(image_id)

    def rows_with_images():
        for row, image in stream_rows_in_dataset_order(work_rows, load_construction_site):
            cache.clear()
            cache[row["image_id"]] = image
            yield row

    summary = run(
        work_rows=work_rows, backend=backend, image_source=image_source,
        out_csv=out_csv, state_json=state_json, fingerprint=fingerprint,
        resume=args.resume, force_resume=args.force_resume,
        checkpoint_every=args.checkpoint_every, rows=rows_with_images(),
    )
    print(f"\ndone={summary['done']}  skipped={summary['skipped']}  "
          f"errors={summary['errors']}")
    print(f"peak VRAM: {backend.peak_memory_gb():.2f} GB")

    missing = verify_coverage(work_rows, out_csv)
    if missing and args.limit is None:
        print(f"\nINCOMPLETE: {missing} rows never attempted. Rerun with --resume.",
              file=sys.stderr)

    _summarise(out_csv)
    if missing and args.limit is None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
