"""Stage 6 -- Metric 3: Stability. Ask the same question five times -- same answer?

SAMPLING DECODING IS MANDATORY, not a preference. Under the frozen greedy
GENERATION config the model is deterministic; five reruns would be byte-identical
and score 100% stable by construction, proving nothing. This stage overrides
do_sample/temperature for its calls only -- every other stage keeps greedy -- and
records the override in the fingerprint so a resumed run cannot silently mix
sampling settings with a prior one.

Requires CUDA. Resumable: rerun the same command after any interruption.

    python scripts/06_stability.py --model qwen7b-4bit --resume
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
from xai_vlm.metrics.stability import (  # noqa: E402
    is_flickering,
    mean_pairwise_centroid_drift,
    pairwise_agreement,
    parse_box,
    region_presence_rate,
)
from xai_vlm.runner import Guards, backend_device, manifest_sha256, run_stage  # noqa: E402
from xai_vlm.sampling import floor_then_proportional_sample  # noqa: E402

RESULT_FIELDS = [
    "image_id", "rule_id", "baseline_verdict", "verdicts", "regions",
    "agreement_rate", "region_presence_rate", "flickering",
    "mean_centroid_drift", "latency_ms", "error",
]

N_RERUNS = 5
STABILITY_TEMPERATURE = 0.7

# Same scheme as Stage 4: guarantee every rule its floor before filling
# proportionally, so rule_4 (an order of magnitude scarcer than rule_1) is not
# crowded out. Stability is a property of the MODEL, not of one image, so a
# subset is adequate; the full manifest would cost 4x more for a narrower
# interval on a quantity nowhere near a decision boundary.
SUBSET_SIZE = 800
PER_RULE_FLOOR = 40


def build_subset(baseline_rows: list[dict], seed: int = 42) -> list[dict]:
    """Stratified subset of baseline rows, one entry per (image_id, rule_id)."""
    return floor_then_proportional_sample(
        baseline_rows, group_key="verdict", stratify_key="rule_id",
        targets={"YES": SUBSET_SIZE // 2, "NO": SUBSET_SIZE // 2},
        floor=PER_RULE_FLOOR, seed=seed,
    )


def make_work_fn(backend, image_source, n_reruns: int = N_RERUNS,
                 temperature: float = STABILITY_TEMPERATURE):
    """One baseline row -> n sampled reruns -> one stability result row."""

    def work(row: dict) -> dict:
        out = {
            "image_id": row["image_id"], "rule_id": row["rule_id"],
            "baseline_verdict": row.get("verdict", ""),
            "verdicts": "", "regions": "", "agreement_rate": "",
            "region_presence_rate": "", "flickering": "",
            "mean_centroid_drift": "", "latency_ms": "", "error": "",
        }

        image = image_source(row["image_id"])
        if image is None:
            out["error"] = "image not found"
            return out

        # All five reruns, because that is what one row of this stage costs. The
        # runner's latency guard reads this field; without it the guard sees
        # 0.0 s forever and cannot detect the shared-memory spill, which is the
        # failure mode a 5x-generation stage is most exposed to.
        started = time.perf_counter()

        verdicts, regions = [], []
        for _ in range(n_reruns):
            ans = backend.answer(image, row["rule_id"],
                                 do_sample=True, temperature=temperature)
            verdicts.append(ans.verdict or "")
            regions.append("" if ans.region is None else str(tuple(ans.region)))

        out["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        out["verdicts"] = "|".join(verdicts)
        out["regions"] = "|".join(regions)
        out["agreement_rate"] = f"{pairwise_agreement(verdicts):.4f}"
        out["region_presence_rate"] = f"{region_presence_rate(regions):.4f}"
        out["flickering"] = str(is_flickering(regions))

        w = int(row.get("image_w") or 0)
        h = int(row.get("image_h") or 0)
        boxes = [b for b in (parse_box(r) for r in regions) if b is not None]
        if w and h and len(boxes) >= 2:
            out["mean_centroid_drift"] = f"{mean_pairwise_centroid_drift(boxes, (w, h)):.4f}"
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
    ok = [r for r in rows if r["agreement_rate"]]
    print("\n" + "=" * 66)
    if ok:
        agree = sorted(float(r["agreement_rate"]) for r in ok)
        n = len(agree)
        print(f"agreement rate  median {agree[n // 2]:.1%}  "
              f"p25 {agree[int(n * .25)]:.1%}  p75 {agree[int(n * .75)]:.1%}  (n={n})")
        flick = sum(1 for r in ok if r["flickering"] == "True")
        print(f"flickering regions : {flick}/{n} ({flick / n:.1%})")
        print("  present in some reruns and absent in others -- worse than a")
        print("  consistently missing region, because it is unpredictable")
        drift = [float(r["mean_centroid_drift"]) for r in ok if r["mean_centroid_drift"]]
        if drift:
            print(f"mean centroid drift (normalised)  median {sorted(drift)[len(drift)//2]:.3f}  "
                  f"(n={len(drift)})")
    print(f"errors : {sum(1 for r in rows if r.get('error'))}")
    print("=" * 66)


def reply_parse_share(out_row: dict) -> float:
    """Share of this row's sampled replies that produced a verdict.

    Not `verdict`: that column is inherited from the baseline CSV this stage
    draws its subset from, so the default guard would grade Stage 2's output.

    Per reply, not per row (decided 2026-11-10). The row-level reading -- one
    unreadable reply fails all five -- tripped bf16 at 10% of rows from 2.6% of
    replies, mostly a bare "NONE" at T=0.7. Stage 2's parse rate is per reply,
    and the guard exists to catch a degrading prompt, which a per-reply rate
    still does. An empty slot still counts as a disagreement in the agreement
    rate, and the report gives the unreadable-reply rate on its own line.
    """
    parts = (out_row.get("verdicts") or "").split("|")
    return sum(1 for p in parts if p.strip()) / len(parts)


def run(*, work_rows, backend, image_source, out_csv, state_json, fingerprint,
        resume, force_resume, checkpoint_every, rows=None) -> dict:
    return run_stage(
        rows if rows is not None else work_rows,
        make_work_fn(backend, image_source),
        Path(out_csv), Path(state_json), RESULT_FIELDS, fingerprint,
        resume=resume, force_resume=force_resume,
        checkpoint_every=checkpoint_every, progress_fn=_progress, guards=Guards(),
        ok_fn=reply_parse_share,
    )


def build_fingerprint(manifest_path, baseline_path, model, seed, *, device: str) -> dict:
    """Everything that, if changed, makes previously written rows incomparable.

    Includes the sampling override: under the study-wide greedy config five
    reruns are byte-identical, so a resume must not mix decoding settings.
    """
    return {
        "manifest_sha256": manifest_sha256(Path(manifest_path)),
        "baseline_sha256": manifest_sha256(Path(baseline_path)),
        "model": model,
        "device": device,
        "compute_dtype": config.COMPUTE_DTYPE,
        "attention_impl": config.ATTENTION_IMPL,
        "max_pixels": config.MAX_PIXELS,
        "n_reruns": N_RERUNS,
        "stability_temperature": STABILITY_TEMPERATURE,
        "subset_size": SUBSET_SIZE,
        "per_rule_floor": PER_RULE_FLOOR,
        "seed": seed,
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
    work_rows = build_subset(baseline_rows, seed=args.seed)
    if args.limit is not None:
        work_rows = work_rows[:args.limit]
    n_yes = sum(1 for r in work_rows if r["verdict"] == "YES")
    print(f"stability subset: {len(work_rows)} rows "
          f"({n_yes} YES, {len(work_rows) - n_yes} NO), seed {args.seed}")
    print(f"calls: {len(work_rows) * N_RERUNS} (5 reruns/row, temperature="
          f"{STABILITY_TEMPERATURE})")

    out_csv = res / f"stability_{args.model}.csv"
    state_json = res / f"stability_{args.model}.state.json"
    from xai_pilot.data import load_construction_site
    from xai_vlm.backends.qwen import QwenBackend

    print(f"loading {args.model} ...")
    backend = QwenBackend(tier=args.model)
    fingerprint = build_fingerprint(manifest, baseline, args.model, args.seed,
                                    device=backend_device(backend))

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
