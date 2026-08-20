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


# Pre-registered subsample sizes, fixed before any flip rate was computed.
#
# WHY SUBSAMPLE AT ALL. The architecture document budgeted Stage 4 at ~11 h on an
# assumed 4 s/call and says explicitly never to cut it. Measured cost is ~96 s per
# row -- two masking calls, each paying an image-preprocessing pass that the
# recorded latency does not even count -- which puts the full 3,031 rows at ~81 h.
# That is not a schedule, it is an abandonment of Stages 5, 6 and 8.
#
# Subsampling preserves the metric rather than cutting it. Precision for a
# proportion near 0.5 at 95%:
#
#     n=400  -> +/- 4.9 points
#     n=800  -> +/- 3.5 points
#     n=2621 -> +/- 1.9 points   (the full set)
#
# The extra 1.6 points of precision from running everything costs about fifty
# additional GPU-hours. 800 YES rows and 400 controls is ample to separate a
# faithful model from an unfaithful one, and the interval is reported so a reader
# can judge the precision directly.
SAMPLE_YES = 800
SAMPLE_NO = 400


# Every rule must clear the reporting floor where its pool allows, before any
# budget is spent on making the common rules more precise.
#
# Purely proportional stratification gave rule_4 twenty-one YES rows -- under the
# n=30 floor, and therefore unreportable. That is the subgroup the baseline found
# the model nearly blind to (30% sensitivity against 93-99% elsewhere), so it is
# precisely the one worth measuring. Its entire pool is 70 rows; taking all of them
# costs 49 extra rows and removes sampling error from the subgroup that matters
# most.
PER_RULE_FLOOR = 40


def stratified_sample(rows: list[dict], n_yes: int = SAMPLE_YES,
                      n_no: int = SAMPLE_NO, seed: int = 42) -> list[dict]:
    """Seeded draw: guarantee each rule a floor, then fill proportionally.

    Stratifying by rule matters here because the rules differ by an order of
    magnitude in frequency, and the rarest one is the most interesting.
    """
    import random

    rng = random.Random(seed)
    out: list[dict] = []

    for verdict, target in (("YES", n_yes), ("NO", n_no)):
        pool = [r for r in rows if r.get("verdict") == verdict]
        if len(pool) <= target:
            out.extend(pool)
            continue

        by_rule: dict[str, list[dict]] = {}
        for r in pool:
            by_rule.setdefault(r["rule_id"], []).append(r)
        for group in by_rule.values():
            rng.shuffle(group)

        # Pass 1: the floor, capped by what each pool actually holds.
        picked: dict[str, list[dict]] = {}
        for rule, group in sorted(by_rule.items()):
            picked[rule] = group[:min(len(group), PER_RULE_FLOOR)]

        # Pass 2: distribute what remains in proportion to the leftover pools.
        used = sum(len(v) for v in picked.values())
        remaining = max(0, target - used)
        leftovers = {rule: by_rule[rule][len(picked[rule]):] for rule in by_rule}
        total_left = sum(len(v) for v in leftovers.values())
        if remaining and total_left:
            for rule, group in sorted(leftovers.items()):
                k = round(remaining * len(group) / total_left)
                picked[rule].extend(group[:k])

        drawn = [r for rule in sorted(picked) for r in picked[rule]]
        rng.shuffle(drawn)
        out.extend(drawn[:max(target, used)])

    return out


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
    ap.add_argument("--sample", action="store_true",
                    help="run the pre-registered stratified subsample "
                         f"({SAMPLE_YES} YES + {SAMPLE_NO} controls) instead of all rows")
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
    if args.sample:
        work_rows = stratified_sample(work_rows, seed=args.seed)
        print(f"stratified subsample: seed {args.seed}, "
              f"{SAMPLE_YES} YES target, {SAMPLE_NO} NO target")
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
