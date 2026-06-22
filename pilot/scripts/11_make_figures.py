"""Day 11 deliverable: roll up all six metrics into one summary table + chart.

Pure aggregation over Days 5-10's already-computed CSVs -- no new model
calls, no recomputation. Per-class breakdown uses primary_class (compliant,
ppe_violation, fall_hazard, struck_by_risk) to match the dataset's labeled
hazard categories. Note struck_by_risk only has 13 samples (the balanced
sampler couldn't find 50 in the test split -- see data/pilot_samples.csv),
so its per-class numbers are far noisier than the other three classes' n=50
and shouldn't be read with the same confidence.

Rule-level breakdowns that already revealed the real mechanism behind a
metric (rule_2's prompt-phrasing heterogeneity in Day 9, rule_1's
worker-vs-object region-ranking issue in Day 10) are documented in their own
day's findings doc and are not recomputed here -- this script's only job is
the cross-metric rollup the plan asks for.
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd

from xai_pilot.config import FIGURES_DIR, RESULTS_DIR

CLASSES = ["compliant", "ppe_violation", "fall_hazard", "struck_by_risk"]


def _rate_by_class(df: pd.DataFrame, value_col: str) -> dict:
    row = {"overall": df[value_col].mean()}
    for cls in CLASSES:
        subset = df[df["primary_class"] == cls]
        row[cls] = subset[value_col].mean() if len(subset) else float("nan")
    row["n"] = len(df)
    return row


def main() -> int:
    rows = []

    da = pd.read_csv(RESULTS_DIR / "descriptive_accuracy.csv")
    da["answer_changed_top1"] = da["answer_changed_top1"].astype(bool)
    rows.append(
        {
            "metric": "descriptive_accuracy",
            "direction": "higher_better",
            **_rate_by_class(da, "answer_changed_top1"),
            "source_csv": "descriptive_accuracy.csv",
        }
    )

    sp = pd.read_csv(RESULTS_DIR / "visual_sparsity.csv")
    sp_scored = sp[sp["excluded_reason"].isna()]
    rows.append(
        {
            "metric": "visual_sparsity_top5",
            "direction": "higher_better",
            **_rate_by_class(sp_scored, "topk5_mass_ratio"),
            "source_csv": "visual_sparsity.csv",
        }
    )

    st = pd.read_csv(RESULTS_DIR / "stability.csv")
    rows.append(
        {
            "metric": "stability_answer_agreement",
            "direction": "higher_better",
            **_rate_by_class(st, "answer_agreement_rate"),
            "source_csv": "stability.csv",
        }
    )

    rb = pd.read_csv(RESULTS_DIR / "robustness.csv")
    rb_level1 = rb[rb["level"] == "level1_image"].copy()
    rb_level1["answer_survived"] = ~rb_level1["answer_changed"].astype(bool)
    rows.append(
        {
            "metric": "robustness_level1_answer_survival",
            "direction": "higher_better",
            **_rate_by_class(rb_level1, "answer_survived"),
            "source_csv": "robustness.csv",
        }
    )

    bc = pd.read_csv(RESULTS_DIR / "bounded_completeness.csv")
    bc["explanation_supported"] = bc["verdict"] == "explanation_supported"
    rows.append(
        {
            "metric": "bounded_completeness",
            "direction": "higher_better",
            **_rate_by_class(bc, "explanation_supported"),
            "source_csv": "bounded_completeness.csv",
        }
    )

    eff = pd.read_csv(RESULTS_DIR / "efficiency_summary.csv")
    rows.append(
        {
            "metric": "efficiency_total_pipeline_1000hr",
            "direction": "lower_better",
            "overall": eff["extrapolated_1000_hours"].sum(),
            "compliant": None,
            "ppe_violation": None,
            "fall_hazard": None,
            "struck_by_risk": None,
            "n": "n/a -- aggregate runtime budget, not a per-sample classification",
            "source_csv": "efficiency_summary.csv",
        }
    )

    out_df = pd.DataFrame(
        rows,
        columns=["metric", "direction", "overall", "compliant", "ppe_violation", "fall_hazard", "struck_by_risk", "n", "source_csv"],
    )
    out_csv = RESULTS_DIR / "pilot_metric_summary.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {out_csv}")
    print(out_df.to_string(index=False))

    chart_df = out_df[out_df["metric"] != "efficiency_total_pipeline_1000hr"].set_index("metric")
    fig, ax = plt.subplots(figsize=(11, 5))
    x = list(range(len(chart_df)))
    width = 0.2
    for i, cls in enumerate(CLASSES):
        ax.bar([xi + i * width for xi in x], chart_df[cls], width, label=f"{cls} (n={'13' if cls == 'struck_by_risk' else '50'})")
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(chart_df.index, rotation=15, ha="right")
    ax.set_ylabel("metric value (0-1)")
    ax.set_title("Five per-sample metrics by hazard class (163-sample pilot)\nnote: each metric has its own scale -- see pilot_metric_summary.csv for definitions")
    ax.legend()
    plt.tight_layout()

    fig_dir = FIGURES_DIR / "summary"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / "metric_summary_by_class.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved summary chart to {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
