"""Stage 8 -- Metric 5: Robustness. Degrade the image, ask again -- same answer?

Five perturbations (xai_vlm.perturbations), each applied ALONE at the strengths
approved on 2026-11-10, to a stratified 1,600-row subset of the Stage 2 baseline.
Greedy decoding, as in Stage 2, so a changed verdict is the perturbation's doing
and not sampling noise (Stage 6 measures that separately).

Per perturbation, never pooled: "80% robust" hides a model that is perfect under
blur and useless in low light, and low light is when site accidents happen.

  survived   verdict equals the baseline verdict
  drift      baseline-to-perturbed region centroid distance, / image diagonal
  drifted    survived, but the region moved more than DRIFT_FLAG of the diagonal:
             "survival without stability is coincidence"
  occluder_coverage  share of the annotated hazard the occlusion block hides.
             Above 10%, a flip is the correct response, so occlusion survival is
             reported split on it.

Requires CUDA. Resumable: rerun the same command after any interruption.

    python scripts/08_robustness.py --model qwen7b-4bit --resume
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402
from xai_vlm import perturbations as P  # noqa: E402
from xai_vlm.metrics.stability import (  # noqa: E402
    DRIFT_FLAG, normalized_centroid_distance, parse_box)
from xai_vlm.runner import Guards, backend_device, manifest_sha256, run_stage  # noqa: E402
from xai_vlm.sampling import floor_then_proportional_sample  # noqa: E402

# Plan 3c: 1,600 rows, same stratification scheme as Stages 4 and 6, seed 42.
SUBSET_SIZE = 1600
PER_RULE_FLOOR = 40
# DRIFT_FLAG (metrics.stability, 0.10) is a reporting flag, not a strength: a
# surviving verdict whose region moved more than a tenth of the image diagonal
# is pointing somewhere else. Shared with Stage 6.

RESULT_FIELDS = (
    ["image_id", "rule_id", "quality_of_info", "baseline_verdict", "baseline_region",
     "occluder_coverage"]
    + [f"{k}_{p}" for p in P.PERTURBATIONS
       for k in ("verdict", "region", "survived", "drift", "drifted")]
    + ["latency_ms", "error"]
)


def build_subset(baseline_rows: list[dict], seed: int = 42) -> list[dict]:
    return floor_then_proportional_sample(
        baseline_rows, group_key="verdict", stratify_key="rule_id",
        targets={"YES": SUBSET_SIZE // 2, "NO": SUBSET_SIZE // 2},
        floor=PER_RULE_FLOOR, seed=seed,
    )


def make_work_fn(backend, image_source, boxes_source):
    """One baseline row -> five perturbed greedy answers -> one result row.

    boxes_source(image_id, rule_id) -> the annotation's normalised boxes, or []."""

    def work(row: dict) -> dict:
        out = {k: "" for k in RESULT_FIELDS}
        out.update(image_id=row["image_id"], rule_id=row["rule_id"],
                   quality_of_info=row.get("quality_of_info", ""),
                   baseline_verdict=row.get("verdict", ""),
                   baseline_region=row.get("region", ""))

        image = image_source(row["image_id"])
        if image is None:
            out["error"] = "image not found"
            return out

        cov = P.occluder_coverage(boxes_source(row["image_id"], row["rule_id"]))
        out["occluder_coverage"] = "" if cov is None else f"{cov:.4f}"
        base_box = parse_box(row.get("region", ""))

        started = time.perf_counter()
        for p in P.PERTURBATIONS:
            ans = backend.answer(P.apply(p, image), row["rule_id"])
            verdict = ans.verdict or ""
            out[f"verdict_{p}"] = verdict
            out[f"region_{p}"] = "" if ans.region is None else str(tuple(ans.region))
            if verdict and out["baseline_verdict"]:
                out[f"survived_{p}"] = str(verdict == out["baseline_verdict"])
            if base_box is not None and ans.region is not None:
                d = normalized_centroid_distance(base_box, tuple(ans.region), image.size)
                out[f"drift_{p}"] = f"{d:.4f}"
            if out[f"survived_{p}"] == "True" and out[f"drift_{p}"]:
                out[f"drifted_{p}"] = str(float(out[f"drift_{p}"]) > DRIFT_FLAG)
        out["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        return out

    return work


def row_ok(out_row: dict) -> bool:
    """Every perturbed call answered. Not `verdict`, which is Stage 2's column."""
    return all((out_row.get(f"verdict_{p}") or "").strip() for p in P.PERTURBATIONS)


def _progress(summary: dict) -> None:
    g = summary["guards"]
    print(f"  {summary['done']:>6} done  {summary['skipped']:>6} skipped  "
          f"{summary['errors']:>4} errors  |  median {g.median_latency_s:.1f}s")


def stream_rows_in_dataset_order(work_rows, loader):
    """Yield (row, image, record) in streaming order. The record carries the
    annotation boxes, so they need no second pass over the dataset."""
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
                yield row, image, rec
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
    ok = [r for r in rows if not r.get("error")]
    print("\n" + "=" * 70)
    print(f"ROBUSTNESS  (n={len(ok)}, greedy, one perturbation at a time)")
    print(f"{'':<12}{'survived':>10}{'YES->NO':>9}{'NO->YES':>9}{'drifted':>9}"
          f"{'med drift':>11}")
    for p in P.PERTURBATIONS:
        s = [r for r in ok if r[f"survived_{p}"]]
        surv = sum(r[f"survived_{p}"] == "True" for r in s) / len(s) if s else float("nan")
        y2n = sum(r["baseline_verdict"] == "YES" and r[f"verdict_{p}"] == "NO" for r in ok)
        n2y = sum(r["baseline_verdict"] == "NO" and r[f"verdict_{p}"] == "YES" for r in ok)
        dr = [r for r in ok if r[f"drifted_{p}"]]
        drifted = sum(r[f"drifted_{p}"] == "True" for r in dr) / len(dr) if dr else float("nan")
        d = sorted(float(r[f"drift_{p}"]) for r in ok if r[f"drift_{p}"])
        med = f"{d[len(d) // 2]:.3f}" if d else "-"
        print(f"{p:<12}{surv:>10.1%}{y2n:>9}{n2y:>9}{drifted:>9.1%}{med:>11}")
    hidden = [r for r in ok if r["occluder_coverage"]
              and float(r["occluder_coverage"]) >= 0.10 and r["survived_occlusion"]]
    visible = [r for r in ok if r["occluder_coverage"]
               and float(r["occluder_coverage"]) < 0.10 and r["survived_occlusion"]]
    for label, sub in (("hazard >=10% hidden", hidden), ("hazard <10% hidden", visible)):
        if sub:
            s = sum(r["survived_occlusion"] == "True" for r in sub) / len(sub)
            print(f"  occlusion, {label:<20}: survived {s:.1%} (n={len(sub)})")
    print(f"errors : {sum(1 for r in rows if r.get('error'))}")
    print("=" * 70)


def run(*, work_rows, backend, image_source, boxes_source, out_csv, state_json,
        fingerprint, resume, force_resume, checkpoint_every, rows=None) -> dict:
    return run_stage(
        rows if rows is not None else work_rows,
        make_work_fn(backend, image_source, boxes_source),
        Path(out_csv), Path(state_json), RESULT_FIELDS, fingerprint,
        resume=resume, force_resume=force_resume,
        checkpoint_every=checkpoint_every, progress_fn=_progress, guards=Guards(),
        ok_fn=row_ok,
    )


def build_fingerprint(manifest_path, baseline_path, model, seed, *, device: str) -> dict:
    """Everything that, if changed, makes previously written rows incomparable --
    the approved strengths above all."""
    return {
        "manifest_sha256": manifest_sha256(Path(manifest_path)),
        "baseline_sha256": manifest_sha256(Path(baseline_path)),
        "model": model,
        "device": device,
        "compute_dtype": config.COMPUTE_DTYPE,
        "attention_impl": config.ATTENTION_IMPL,
        "max_pixels": config.MAX_PIXELS,
        "generation": config.GENERATION,
        "perturbations": list(P.PERTURBATIONS),
        "strengths": dict(P.STRENGTHS),
        "drift_flag": DRIFT_FLAG,
        "subset_size": SUBSET_SIZE,
        "per_rule_floor": PER_RULE_FLOOR,
        "seed": seed,
        **config.region_frame_fingerprint(),
        **config.backend_fingerprint(model),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(config.MODEL_TIERS))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    res = config.RESULTS_DIR
    manifest = res / "sample_manifest.csv"
    baseline = res / f"baseline_{args.model}.csv"
    if not baseline.exists():
        print(f"missing {baseline}. Run Stage 2 first.", file=sys.stderr)
        return 1

    with baseline.open(newline="", encoding="utf-8") as fh:
        baseline_rows = list(csv.DictReader(fh))
    with manifest.open(newline="", encoding="utf-8") as fh:
        quality = {(r["image_id"], r["rule_id"]): r.get("quality_of_info", "")
                   for r in csv.DictReader(fh)}
    for r in baseline_rows:
        r["quality_of_info"] = quality.get((r["image_id"], r["rule_id"]), "")

    work_rows = build_subset(baseline_rows, seed=args.seed)
    if args.limit is not None:
        work_rows = work_rows[:args.limit]
    n_yes = sum(1 for r in work_rows if r["verdict"] == "YES")
    print(f"robustness subset: {len(work_rows)} rows "
          f"({n_yes} YES, {len(work_rows) - n_yes} NO), seed {args.seed}")
    print(f"calls: {len(work_rows) * len(P.PERTURBATIONS)}  strengths: {P.STRENGTHS}")

    out_csv = res / f"robustness_{args.model}.csv"
    state_json = res / f"robustness_{args.model}.state.json"
    from xai_pilot.data import load_construction_site
    from xai_vlm.backends.load import load_backend

    print(f"loading {args.model} ...")
    backend = load_backend(args.model)
    fingerprint = build_fingerprint(manifest, baseline, args.model, args.seed,
                                    device=backend_device(backend))

    cache: dict = {}

    def image_source(image_id):
        return cache.get("image") if cache.get("id") == image_id else None

    def boxes_source(image_id, rule_id):
        rec = cache.get("rec") if cache.get("id") == image_id else None
        v = (rec or {}).get(f"{rule_id}_violation") or {}
        return v.get("bounding_box") or []

    def rows_with_images():
        for row, image, rec in stream_rows_in_dataset_order(work_rows, load_construction_site):
            cache.update(id=row["image_id"], image=image, rec=rec)
            yield row

    summary = run(
        work_rows=work_rows, backend=backend, image_source=image_source,
        boxes_source=boxes_source, out_csv=out_csv, state_json=state_json,
        fingerprint=fingerprint, resume=args.resume, force_resume=args.force_resume,
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
    return 1 if missing and args.limit is None else 0


if __name__ == "__main__":
    sys.exit(main())
