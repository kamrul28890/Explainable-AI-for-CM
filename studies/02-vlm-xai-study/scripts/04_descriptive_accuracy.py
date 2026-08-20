"""Stage 4 -- Metric 1: Descriptive Accuracy. The primary faithfulness measurement.

For every row where the model named a region, remove that region and ask again --
twice, once by black fill and once by token ablation. If the answer does not
change, the region was decoration.

Requires CUDA. Resumable: rerun the same command after any interruption.

    python scripts/04_descriptive_accuracy.py --model qwen7b-4bit --resume
"""
import argparse
import ast
import csv
import sys
from collections import defaultdict
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402
from xai_vlm.masking import snap_to_patch_grid  # noqa: E402
from xai_vlm.metrics.descriptive_accuracy import (  # noqa: E402
    MASKING_METHODS,
    artefact_floor,
    flip_rate,
    is_flip,
)
from xai_vlm.runner import Guards, manifest_sha256, run_stage  # noqa: E402

RESULT_FIELDS = [
    "image_id", "rule_id", "baseline_verdict", "region", "region_snapped",
    "verdict_black", "flip_black", "latency_ms_black",
    "verdict_tokens", "flip_tokens", "latency_ms_tokens",
    "methods_disagree", "error",
]


def maskable_rows(regions_rows: list[dict]) -> list[dict]:
    """Every row that named a region, in BOTH verdict classes.

    The NO rows are the artefact control, not padding. If masking only removes
    information a NO cannot rationally become a YES, so their flip rate is the
    floor against which the YES rate must be read.
    """
    return [r for r in regions_rows if r.get("source") == "model" and (r.get("region") or "").strip()]


def build_fingerprint(manifest_path, baseline_path, regions_path, model, seed) -> dict:
    """Everything that, if changed, makes previously written rows incomparable."""
    return {
        "manifest_sha256": manifest_sha256(Path(manifest_path)),
        "baseline_sha256": manifest_sha256(Path(baseline_path)),
        "regions_sha256": manifest_sha256(Path(regions_path)),
        "model": model,
        "compute_dtype": config.COMPUTE_DTYPE,
        "attention_impl": config.ATTENTION_IMPL,
        "max_pixels": config.MAX_PIXELS,
        "generation": config.GENERATION,
        "masking_methods": ",".join(MASKING_METHODS),
        "seed": seed,
    }


def _parse_box(text: str):
    try:
        v = ast.literal_eval((text or "").strip())
        return tuple(float(x) for x in v) if len(v) == 4 else None
    except (ValueError, SyntaxError, TypeError):
        return None


def make_work_fn(backend, image_source):
    """One region row -> one result row carrying BOTH maskings.

    Both methods run inside a single work_fn call so the runner's
    (image_id, rule_id) key contract stays intact. Widening the key to include the
    method would change an interface Stages 5, 6 and 8 will also consume, to save
    about twenty seconds of rework after a crash.
    """

    def work(row: dict) -> dict:
        out = {
            "image_id": row["image_id"], "rule_id": row["rule_id"],
            "baseline_verdict": row.get("verdict", ""),
            "region": row.get("region", ""),
            "region_snapped": "", "verdict_black": "", "flip_black": "",
            "latency_ms_black": "", "verdict_tokens": "", "flip_tokens": "",
            "latency_ms_tokens": "", "methods_disagree": "", "error": "",
        }

        box = _parse_box(row.get("region", ""))
        if box is None:
            out["error"] = "unparseable region"
            return out

        image = image_source(row["image_id"])
        if image is None:
            out["error"] = "image not found"
            return out

        # Recorded for auditability only. answer_masked snaps internally with the
        # same function, so passing a pre-snapped box would be redundant.
        out["region_snapped"] = str(snap_to_patch_grid(box, image.size))

        for method in MASKING_METHODS:
            ans = backend.answer_masked(image, row["rule_id"], box, method=method)
            v = ans.verdict or ""
            out[f"verdict_{method}"] = v
            out[f"flip_{method}"] = is_flip(out["baseline_verdict"], v)
            out[f"latency_ms_{method}"] = ans.latency_ms

        out["methods_disagree"] = out["flip_black"] != out["flip_tokens"]
        return out

    return work


def _progress(summary: dict) -> None:
    g = summary["guards"]
    print(f"  {summary['done']:>6} done  {summary['skipped']:>6} skipped  "
          f"{summary['errors']:>4} errors  |  median {g.median_latency_s:.1f}s")


def stream_rows_in_dataset_order(work_rows, loader):
    """Yield (row, image) in dataset order, one pass over every split."""
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
    """How many maskable rows are absent from the results file."""
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

    print("\n" + "=" * 70)
    print("ARTEFACT FLOOR FIRST")
    for method in MASKING_METHODS:
        floor = artefact_floor(rows, method)
        print(f"  {method:<7} NO->YES : {floor:.1%}"
              if floor == floor else f"  {method:<7} NO->YES : n/a")
    print("  Masking only removes information, so a NO cannot rationally become a")
    print("  YES. A YES-flip rate that fails to clear this is not faithfulness.")

    print("\nFLIP RATE (YES answers whose verdict changed when the region was removed)")
    for method in MASKING_METHODS:
        r = flip_rate(rows, method)
        tag = "  <- PRIMARY, artefact-free" if method == "tokens" else ""
        print(f"  {method:<7} : {r:.1%}{tag}" if r == r else f"  {method:<7} : n/a")

    fb, ft = flip_rate(rows, "black"), flip_rate(rows, "tokens")
    if fb == fb and ft == ft:
        print(f"\n  black - tokens gap : {fb - ft:+.1%}")
        print("  how much the black-rectangle artefact inflates the apparent rate")

    dis = sum(1 for r in rows if str(r.get("methods_disagree", "")).lower() == "true")
    print(f"\nmethods disagree : {dis}/{len(rows)} ({dis / len(rows):.1%})")
    print("  flagged, never averaged -- disagreement between an artefact-free and")
    print("  an artefact-prone intervention diagnoses residual artefact")
    print(f"errors           : {sum(1 for r in rows if r.get('error'))}")
    print("=" * 70)


def run(*, work_rows, backend, image_source, out_csv, state_json, fingerprint,
        resume, force_resume, checkpoint_every, rows=None) -> dict:
    return run_stage(
        rows if rows is not None else work_rows,
        make_work_fn(backend, image_source),
        Path(out_csv), Path(state_json), RESULT_FIELDS, fingerprint,
        resume=resume, force_resume=force_resume,
        checkpoint_every=checkpoint_every, progress_fn=_progress, guards=Guards(),
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
        work_rows = maskable_rows(list(csv.DictReader(fh)))
    if args.limit is not None:
        work_rows = work_rows[:args.limit]

    n_yes = sum(1 for r in work_rows if r["verdict"] == "YES")
    print(f"maskable rows: {len(work_rows)}  "
          f"({n_yes} YES, {len(work_rows) - n_yes} NO controls)")
    print(f"calls: {len(work_rows) * len(MASKING_METHODS)}")

    out_csv = res / f"descriptive_accuracy_{args.model}.csv"
    state_json = res / f"descriptive_accuracy_{args.model}.state.json"

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
        out_csv=out_csv, state_json=state_json,
        fingerprint=build_fingerprint(manifest, baseline, regions, args.model, args.seed),
        resume=args.resume, force_resume=args.force_resume,
        checkpoint_every=args.checkpoint_every, rows=rows_with_images(),
    )
    print(f"\ndone={summary['done']}  skipped={summary['skipped']}  "
          f"errors={summary['errors']}")
    print(f"peak VRAM: {backend.peak_memory_gb():.2f} GB")

    missing = verify_coverage(work_rows, out_csv)
    if missing and args.limit is None:
        print("\n" + "!" * 70, file=sys.stderr)
        print(f"INCOMPLETE: {missing} maskable rows were never attempted.",
              file=sys.stderr)
        print("Rerun with --resume to finish; completed rows are on disk.",
              file=sys.stderr)
        print("!" * 70 + "\n", file=sys.stderr)

    _summarise(out_csv)
    if missing and args.limit is None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
