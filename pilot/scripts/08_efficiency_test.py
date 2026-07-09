"""Day 8 deliverable: per-call cost by pilot day, and a 1000-sample extrapolation.

Only Day 3's baseline_predictions.csv logged inference_ms. Day 5 reruns the
exact same call type (answer_rule, default num_beams=3) Day 3 already timed,
so its cost is recovered exactly: rerun_count (1, or 2 if a second region
existed -- from region_extraction.csv's n_regions) times that sample's own
already-logged inference_ms, no new model calls. Days 6 and 7 use decoding
configs nothing has timed before -- output_attentions=True (forces eager
attention, slower than SDPA) and do_sample=True/num_beams=1 (no beam
multiplicity, plausibly faster) -- so this script runs one small (15-sample)
fresh calibration for each of those two call types specifically, rather than
silently assuming they cost the same as Day 3's beam-search calls.
"""

import sys
import time

import matplotlib.pyplot as plt
import pandas as pd

from xai_pilot.attribution import cross_attention_heatmap
from xai_pilot.config import FIGURES_DIR, RESULTS_DIR
from xai_pilot.data import load_construction_site
from xai_pilot.metrics.efficiency import aggregate_timings, extrapolate
from xai_pilot.metrics.stability import run_n_times
from xai_pilot.model import load_florence2
from xai_pilot.prompts import PERSON_PHRASE, RULE_4_PROXIMITY_PAIR, RULE_QUERIES

N_CALIBRATION = 15
N_FULL_STUDY = 1000
DETECTION_TASK = "<OPEN_VOCABULARY_DETECTION>"


def _phrase_for_top_region(rule_id: str, top_label: str) -> str:
    """Map a ranked region label back to its attribution query phrase."""
    if top_label == "worker":
        return PERSON_PHRASE
    if rule_id == "rule_4":
        return RULE_4_PROXIMITY_PAIR[1]
    return RULE_QUERIES[rule_id][0]


def main() -> int:
    """Combine recorded timings with bounded calibration measurements."""
    # Convert numeric columns explicitly because reading with dtype=str keeps
    # identifiers stable but would otherwise make arithmetic concatenate text.
    preds_df = pd.read_csv(RESULTS_DIR / "baseline_predictions.csv", dtype=str)
    preds_df["inference_ms"] = pd.to_numeric(preds_df["inference_ms"])
    regions_df = pd.read_csv(RESULTS_DIR / "region_extraction.csv", dtype=str)
    regions_df["n_regions"] = pd.to_numeric(regions_df["n_regions"])
    merged = preds_df.merge(regions_df, on=["image_id", "assigned_rule_id"])

    n_samples_observed = len(preds_df)

    # --- Day 3: baseline, real per-sample timing already logged ---
    day3_stats = aggregate_timings(preds_df["inference_ms"].tolist())
    day3_total_ms = preds_df["inference_ms"].sum()

    # --- Day 5: same call type as Day 3, exact rerun count from n_regions ---
    merged["day5_reruns"] = merged["n_regions"].apply(lambda n: 2 if n >= 2 else 1)
    merged["day5_cost_ms"] = merged["day5_reruns"] * merged["inference_ms"]
    day5_stats = aggregate_timings(merged["day5_cost_ms"].tolist())
    day5_total_ms = merged["day5_cost_ms"].sum()

    print("Loading Florence-2-base-ft for Day 6/7 calibration...")
    model, processor = load_florence2()

    # Use a fixed prefix of the reproducible manifest. This is a runtime
    # calibration, not a statistical estimate of model accuracy.
    calib_ids = set(merged.head(N_CALIBRATION)["image_id"])
    print(f"Streaming test split to fetch {len(calib_ids)} calibration images...")
    images_by_id = {}
    ds = load_construction_site(split="test", streaming=True)
    for row in ds:
        if row["image_id"] in calib_ids:
            images_by_id[row["image_id"]] = row["image"]
        if len(images_by_id) >= len(calib_ids):
            break

    calib_rows = merged[merged["image_id"].isin(calib_ids)].drop_duplicates("image_id")

    # --- Day 6 calibration: one cross_attention_heatmap call per sample ---
    print("Calibrating Day 6 (cross-attention attribution) timing...")
    day6_timings = []
    for _, row in calib_rows.iterrows():
        if row["top_region_source"] == "grid":
            continue  # excluded from Day 6's own aggregate too -- no phrase to attribute
        image = images_by_id[row["image_id"]]
        phrase = _phrase_for_top_region(row["assigned_rule_id"], row["top_region_label"])
        # Wall-clock timing includes preprocessing, generation, attention
        # extraction, and post-processing, matching the user's actual cost.
        t0 = time.perf_counter()
        cross_attention_heatmap(model, processor, image, DETECTION_TASK, text_input=phrase)
        day6_timings.append((time.perf_counter() - t0) * 1000)
    day6_stats = aggregate_timings(day6_timings)

    # --- Day 7 calibration: one sampled (do_sample=True, num_beams=1) answer_rule call ---
    print("Calibrating Day 7 (sampled rerun) timing...")
    day7_timings = []
    for _, row in calib_rows.iterrows():
        image = images_by_id[row["image_id"]]
        results = run_n_times(model, processor, image, row["assigned_rule_id"], n=1, temperature=0.7)
        day7_timings.append(results[0].inference_ms)
    day7_call_stats = aggregate_timings(day7_timings)
    day7_per_sample_ms = 3 * day7_call_stats["mean_ms"]  # 3 reruns per sample, per metrics/stability.run_n_times
    day7_total_ms = day7_per_sample_ms * n_samples_observed

    # Normalize each pipeline stage to mean milliseconds per pilot sample so
    # stages with different call counts can be summed and extrapolated.
    rows = [
        {
            "day": 3,
            "name": "baseline_inference",
            "calls_per_sample": 1,
            "mean_ms_per_call": day3_stats["mean_ms"],
            "mean_ms_per_sample": day3_stats["mean_ms"],
            "observed_total_seconds": day3_total_ms / 1000,
            "n_observed": n_samples_observed,
        },
        {
            "day": 5,
            "name": "descriptive_accuracy_reruns",
            "calls_per_sample": float(merged["day5_reruns"].mean()),
            "mean_ms_per_call": day3_stats["mean_ms"],
            "mean_ms_per_sample": day5_stats["mean_ms"],
            "observed_total_seconds": day5_total_ms / 1000,
            "n_observed": n_samples_observed,
        },
        {
            "day": 6,
            "name": "cross_attention_attribution",
            "calls_per_sample": 1,
            "mean_ms_per_call": day6_stats["mean_ms"],
            "mean_ms_per_sample": day6_stats["mean_ms"],
            "observed_total_seconds": day6_stats["mean_ms"] * n_samples_observed / 1000,
            "n_observed": f"{N_CALIBRATION} calibration samples, scaled to {n_samples_observed}",
        },
        {
            "day": 7,
            "name": "stability_sampled_reruns",
            "calls_per_sample": 3,
            "mean_ms_per_call": day7_call_stats["mean_ms"],
            "mean_ms_per_sample": day7_per_sample_ms,
            "observed_total_seconds": day7_total_ms / 1000,
            "n_observed": f"{N_CALIBRATION} calibration samples, scaled to {n_samples_observed}",
        },
    ]

    out_df = pd.DataFrame(rows)
    # Linear extrapolation assumes no batching and similar hardware/load. The
    # output is an engineering budget, not a hardware-independent benchmark.
    extrap = out_df["mean_ms_per_sample"].apply(lambda ms: extrapolate(ms, N_FULL_STUDY))
    out_df["extrapolated_1000_minutes"] = [e["total_minutes"] for e in extrap]
    out_df["extrapolated_1000_hours"] = [e["total_hours"] for e in extrap]

    out_csv = RESULTS_DIR / "efficiency_summary.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(out_df)} rows to {out_csv}")
    print(out_df.to_string(index=False))

    total_pipeline_1000_hours = out_df["extrapolated_1000_hours"].sum()
    print(f"\nFull 6-metric pipeline at n=1000: ~{total_pipeline_1000_hours:.2f} GPU-hours (linear extrapolation, single RTX 3070)")

    observed_163_total_minutes = out_df["observed_total_seconds"].sum() / 60
    print(f"Observed/calibrated total for the actual {n_samples_observed}-sample pilot: ~{observed_163_total_minutes:.1f} minutes across days 3/5/6/7")

    fig_dir = FIGURES_DIR / "efficiency"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Plot per-sample cost rather than total observed cost so stages measured
    # on different calibration counts remain visually comparable.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(out_df["name"], out_df["mean_ms_per_sample"], color="steelblue")
    ax.set_ylabel("mean ms per sample")
    ax.set_title(f"Per-sample cost by pilot day ({n_samples_observed}-sample pilot)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(fig_dir / "per_sample_cost.png", dpi=150)
    print(f"Saved runtime plot to {fig_dir / 'per_sample_cost.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
