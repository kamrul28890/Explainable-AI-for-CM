"""Stage 2 -- baseline inference. The main measurement; everything reuses it.

Requires CUDA. Checkpoints every row and resumes, because at many hours a crash at
hour eight without resumability costs a day -- and because a run stitched together
from two different configurations would look complete and be wrong.

    python scripts/02_baseline_inference.py --model qwen7b-4bit --resume
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402
from xai_vlm.parsing import ParseStats, parse_response  # noqa: E402
from xai_vlm.runner import Guards, manifest_sha256, run_stage  # noqa: E402

RESULT_FIELDS = [
    "image_id", "rule_id", "stratum", "is_negative_control", "expected",
    "verdict", "region", "rationale", "confidence", "raw_text",
    "latency_ms", "parse_strict", "parse_error", "error",
]


def _revision_for(model: str) -> str:
    """The pinned weight revision, so the fingerprint records what actually ran."""
    lock = STUDY_ROOT.parents[1] / "requirements" / "models.lock"
    if not lock.exists():
        return "unpinned"
    for line in lock.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        parts = line.split()
        if parts[:1] == [model] and len(parts) >= 3:
            return parts[2]
    return "unpinned"


def build_fingerprint(manifest_path: Path, model: str, seed: int) -> dict:
    """Everything that, if changed, makes previously written rows incomparable."""
    return {
        "manifest_sha256": manifest_sha256(manifest_path),
        "model": model,
        "revision": _revision_for(model),
        "compute_dtype": config.COMPUTE_DTYPE,
        "attention_impl": config.ATTENTION_IMPL,
        "max_pixels": config.MAX_PIXELS,
        "generation": config.GENERATION,
        "seed": seed,
    }


def make_work_fn(backend, image_source):
    """One manifest row -> one result row."""

    def work(row: dict) -> dict:
        # Ground truth is derived, never stored twice. is_violation is per-rule.
        is_violation = str(row.get("is_violation", "")).lower() == "true"
        expected = "YES" if is_violation else "NO"

        image = image_source(row["image_id"])
        if image is None:
            # A missing image is data about the corpus, not a crash. Recorded so
            # the gap is visible in the results rather than as an absent row.
            return {"expected": expected, "verdict": "",
                    "parse_error": "image not found", "latency_ms": 0.0}

        answer = backend.answer(image, row["rule_id"])
        return {
            "expected": expected,
            "verdict": answer.verdict or "",
            "region": "" if answer.region is None else str(tuple(answer.region)),
            "rationale": answer.rationale,
            "confidence": answer.confidence,
            "raw_text": answer.raw_text,
            "latency_ms": answer.latency_ms,
            "parse_strict": answer.parse_strict,
            "parse_error": answer.parse_error,
        }

    return work


def stream_rows_in_dataset_order(manifest_rows, loader):
    """Yield (manifest_row, image) in dataset streaming order, one pass.

    The naive alternative -- look each image up by id -- re-streams the whole split
    per image, which is thousands of full passes. Iterating the split once and
    emitting whatever manifest rows match is why resume is keyed on
    (image_id, rule_id) and never on position: this order is not manifest order.
    """
    by_image = defaultdict(list)
    for row in manifest_rows:
        by_image[row["image_id"]].append(row)

    # EVERY split the study samples from, via the shared constant. Hardcoding
    # "test" here while Stage 1 sampled both is what silently truncated a run to a
    # third of the manifest -- see config.SPLITS.
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


def verify_coverage(manifest_path, out_csv) -> int:
    """How many manifest rows are absent from the results file.

    An independent check on the artifact rather than on the process. A run can
    exit 0, report no errors and print a wholly plausible summary while having
    covered part of the manifest -- rows that were never attempted are invisible
    to the parse-rate and latency guards, because those only see attempts.
    """
    with Path(manifest_path).open(newline="", encoding="utf-8") as fh:
        wanted = {(r["image_id"], r["rule_id"]) for r in csv.DictReader(fh)}
    out_csv = Path(out_csv)
    if not out_csv.exists():
        return len(wanted)
    with out_csv.open(newline="", encoding="utf-8") as fh:
        got = {(r["image_id"], r["rule_id"]) for r in csv.DictReader(fh)}
    return len(wanted - got)


def _progress(summary: dict) -> None:
    g = summary["guards"]
    print(f"  {summary['done']:>6} done  {summary['skipped']:>6} skipped  "
          f"{summary['errors']:>4} errors  |  parse {g.parse_rate:.1%}  "
          f"median {g.median_latency_s:.1f}s")


def run(*, manifest_path, backend, image_source, out_csv, state_json, model, seed,
        limit, resume, force_resume, checkpoint_every, rows=None) -> dict:
    """Drive run_stage over the manifest. Separated from main() so it is testable.

    `rows` lets main() supply a generator that streams the split once and keeps a
    single image resident. When omitted the manifest is read directly, which is
    what the tests use with an in-memory image source.
    """
    if rows is None:
        with Path(manifest_path).open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if limit is not None:
            rows = rows[:limit]

    return run_stage(
        rows,
        make_work_fn(backend, image_source),
        Path(out_csv), Path(state_json), RESULT_FIELDS,
        build_fingerprint(Path(manifest_path), model, seed),
        resume=resume, force_resume=force_resume,
        checkpoint_every=checkpoint_every,
        progress_fn=_progress, guards=Guards(),
    )


def _summarise(out_csv: Path) -> None:
    """Negative controls FIRST, before any accuracy figure.

    If the model answers YES to hazards known to be absent, its accuracy on
    violations is meaningless -- it is guessing, not detecting. Printing the
    controls first makes that unavoidable rather than optional.
    """
    rows = list(csv.DictReader(Path(out_csv).open(encoding="utf-8")))
    if not rows:
        print("no rows written")
        return

    stats = ParseStats()
    for r in rows:
        stats.record(parse_response(r.get("raw_text", "")))

    controls = [r for r in rows if str(r["is_negative_control"]).lower() == "true"]
    false_alarms = sum(1 for r in controls if r["verdict"] == "YES")
    positives = [r for r in rows if r["expected"] == "YES"]
    negatives = [r for r in rows if r["expected"] == "NO"]

    print("\n" + "=" * 66)
    print("NEGATIVE CONTROLS FIRST")
    if controls:
        print(f"  false-alarm rate : {false_alarms / len(controls):.0%} "
              f"({false_alarms}/{len(controls)})")
        if false_alarms / len(controls) > 0.5:
            print("  WARNING: the model answers YES to most absent hazards. Its "
                  "accuracy on violations cannot be read as detection.")
    print(f"\nYES share          : "
          f"{sum(1 for r in rows if r['verdict'] == 'YES') / len(rows):.0%}")
    if positives:
        print(f"sensitivity        : "
              f"{sum(1 for r in positives if r['verdict'] == 'YES') / len(positives):.0%} "
              f"(n={len(positives)})")
    if negatives:
        print(f"specificity        : "
              f"{sum(1 for r in negatives if r['verdict'] == 'NO') / len(negatives):.0%} "
              f"(n={len(negatives)})")
    print(f"errors             : {sum(1 for r in rows if r.get('error'))}")
    print("\n" + stats.summary())
    print("=" * 66)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(config.MODEL_TIERS))
    ap.add_argument("--manifest", default=str(STUDY_ROOT / "results" / "sample_manifest.csv"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-resume", action="store_true",
                    help="resume despite a configuration change (recorded in the state file)")
    ap.add_argument("--limit", type=int, default=None, help="cap rows, for smoke tests")
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"no manifest at {manifest}. Run scripts/01_build_sample.py first.",
              file=sys.stderr)
        return 1

    out_csv = STUDY_ROOT / "results" / f"baseline_{args.model}.csv"
    state_json = STUDY_ROOT / "results" / f"baseline_{args.model}.state.json"

    from xai_pilot.data import load_construction_site
    from xai_vlm.backends.qwen import QwenBackend

    with manifest.open(newline="", encoding="utf-8") as fh:
        manifest_rows = list(csv.DictReader(fh))
    if args.limit is not None:
        manifest_rows = manifest_rows[:args.limit]
    print(f"manifest: {len(manifest_rows)} rows, "
          f"{len({r['image_id'] for r in manifest_rows})} distinct images")

    print(f"loading {args.model} ...")
    backend = QwenBackend(tier=args.model)

    # One image resident at a time. Stage 0 held every gate image in a dict, which
    # was fine for 39 images and is not for thousands.
    cache: dict = {}

    def rows_with_images():
        for row, image in stream_rows_in_dataset_order(manifest_rows,
                                                       load_construction_site):
            cache.clear()
            cache[row["image_id"]] = image
            yield row

    summary = run(
        manifest_path=manifest, backend=backend, image_source=cache.get,
        out_csv=out_csv, state_json=state_json, model=args.model, seed=args.seed,
        limit=args.limit, resume=args.resume, force_resume=args.force_resume,
        checkpoint_every=args.checkpoint_every, rows=rows_with_images(),
    )
    print(f"\ndone={summary['done']}  skipped={summary['skipped']}  "
          f"errors={summary['errors']}")

    # Coverage BEFORE the summary. Statistics computed on a partial manifest look
    # entirely reasonable, so the incompleteness has to be stated first or it will
    # be read past.
    missing = verify_coverage(manifest, out_csv)
    if missing and args.limit is None:
        print("\n" + "!" * 66, file=sys.stderr)
        print(f"INCOMPLETE: {missing} of "
              f"{missing + summary['done'] + summary['skipped']} manifest rows were "
              "never attempted.", file=sys.stderr)
        print("Every figure below is computed on a PARTIAL, and therefore biased, "
              "subset. Do not report it.", file=sys.stderr)
        print("Rerun the same command with --resume to finish; completed rows are "
              "already on disk.", file=sys.stderr)
        print("!" * 66 + "\n", file=sys.stderr)

    _summarise(out_csv)

    if missing and args.limit is None:
        print(f"\nINCOMPLETE -- {missing} rows outstanding. Rerun with --resume.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
