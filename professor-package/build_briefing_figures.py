"""Build evidence-based figures for the finalized meeting briefing.

All numeric charts read the repository's CSV artifacts. Process diagrams encode
the implemented pipeline documented in MEETING_BRIEFING.md and the scale-up
plan. Construction photographs and model-output overlays are reused directly
in the LaTeX document rather than altered here.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
PILOT_DIR = REPO_DIR / "pilot"
OUTPUT_DIR = PACKAGE_DIR / "figures" / "briefing"

NAVY = "#174A7E"
BLUE = "#3A78B4"
LIGHT_BLUE = "#DCEAF6"
GOLD = "#D49A2A"
LIGHT_GOLD = "#F5E8C8"
GREEN = "#2F7D5B"
LIGHT_GREEN = "#DDEFE6"
RED = "#A84A44"
LIGHT_RED = "#F2DEDC"
GRAY = "#58636E"
LIGHT_GRAY = "#EEF1F3"
DARK = "#20252A"


def _save(fig: plt.Figure, name: str) -> None:
    """Save one sharp PNG with consistent whitespace."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = LIGHT_BLUE,
    edgecolor: str = NAVY,
    fontsize: float = 10,
    weight: str = "normal",
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=DARK,
        weight=weight,
        wrap=True,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GRAY,
    style: str = "-|>",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=1.4,
            color=color,
        )
    )


def build_study_overview() -> None:
    """Input -> transformation -> output -> validation overview."""
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    labels = [
        ("INPUT", "Construction image\n+ safety rule", LIGHT_BLUE, NAVY),
        ("MODEL OPERATION", "Florence-2 grounds\nworkers and safety objects", LIGHT_GOLD, GOLD),
        ("TRANSFORMATION", "Regions are ranked,\nmasked, perturbed,\nand re-evaluated", LIGHT_GREEN, GREEN),
        ("OUTPUT", "Six XAI metrics\n+ limitations\n+ scale-up evidence", LIGHT_GRAY, GRAY),
    ]
    x_values = [0.2, 3.2, 6.2, 9.2]
    for x, (header, body, fill, edge) in zip(x_values, labels):
        ax.text(x + 1.2, 2.78, header, ha="center", fontsize=9, weight="bold", color=edge)
        _box(ax, (x, 0.72), 2.4, 1.7, body, facecolor=fill, edgecolor=edge, fontsize=10.5)
    for x in (2.65, 5.65, 8.65):
        _arrow(ax, (x, 1.57), (x + 0.42, 1.57))

    ax.text(
        6,
        0.2,
        "Central validity question: does the metric measure the model's reasoning or the proxy's mechanics?",
        ha="center",
        fontsize=10.5,
        weight="bold",
        color=NAVY,
    )
    _save(fig, "study_overview.png")


def build_pipeline() -> None:
    """Render the 11 implemented pipeline stages and their data flow."""
    fig, ax = plt.subplots(figsize=(12, 7.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    preparation = [
        ("01", "Environment\ncheck"),
        ("02", "Sample\nselection"),
        ("03", "Baseline\ninference"),
        ("04", "Region\nstandardization"),
    ]
    prep_x = [0.35, 3.15, 5.95, 8.75]
    for x, (number, label) in zip(prep_x, preparation):
        _box(ax, (x, 5.45), 2.0, 0.95, f"{number}\n{label}", facecolor=LIGHT_BLUE, edgecolor=NAVY, fontsize=9.2)
    for start_x in (2.4, 5.2, 8.0):
        _arrow(ax, (start_x, 5.93), (start_x + 0.68, 5.93))

    # One clear funnel into the six metric modules avoids a spaghetti dependency diagram.
    _arrow(ax, (6.85, 5.4), (6.85, 4.82))
    metric_container = FancyBboxPatch(
        (0.25, 2.05),
        11.5,
        2.55,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        linewidth=1.5,
        edgecolor=GREEN,
        facecolor="#F7FBF9",
    )
    ax.add_patch(metric_container)
    ax.text(0.55, 4.28, "Six metric evaluations", fontsize=10, weight="bold", color=GREEN)
    metrics = [
        ("05", "Descriptive\nAccuracy"),
        ("06", "Visual\nSparsity"),
        ("07", "Stability"),
        ("08", "Efficiency"),
        ("09", "Robustness"),
        ("10", "Bounded\nCompleteness"),
    ]
    metric_x = [0.55, 2.42, 4.29, 6.16, 8.03, 9.9]
    for x, (number, label) in zip(metric_x, metrics):
        _box(ax, (x, 2.62), 1.48, 1.12, f"{number}\n{label}", facecolor=LIGHT_GREEN, edgecolor=GREEN, fontsize=8.5)
    ax.text(
        6,
        2.28,
        "Baseline predictions and standardized regions are read from saved CSVs; completeness also reuses masking results.",
        ha="center",
        fontsize=8.8,
        color=GRAY,
    )

    _arrow(ax, (6.0, 2.02), (6.0, 1.45))
    _box(
        ax,
        (4.5, 0.35),
        3.0,
        0.9,
        "11  Roll-up, figures,\nand reproducibility check",
        facecolor=LIGHT_GOLD,
        edgecolor=GOLD,
        fontsize=9.2,
    )
    ax.text(1.05, 6.65, "Preparation and standardization", fontsize=10, weight="bold", color=NAVY)
    ax.text(7.85, 0.68, "Traceable outputs", fontsize=10, weight="bold", color=GOLD)
    _save(fig, "pipeline_11_stages.png")


def build_architecture() -> None:
    """Render configuration, library, scripts, and artifact layers."""
    fig, ax = plt.subplots(figsize=(12, 4.7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.7)
    ax.axis("off")

    _box(
        ax,
        (0.25, 1.25),
        2.2,
        2.1,
        "CONFIGURATION\n\nseed\nsample policy\nmodel ID\nranking policy\nfeature flags",
        facecolor=LIGHT_BLUE,
        edgecolor=NAVY,
        fontsize=9.5,
    )
    _box(
        ax,
        (3.15, 0.55),
        3.15,
        3.5,
        "REUSABLE LIBRARY\n\nmodel and data loaders\nprompts and inference\nregion ranking and masking\nattribution and perturbations\nmetric modules\nstatistics and visualization",
        facecolor=LIGHT_GOLD,
        edgecolor=GOLD,
        fontsize=9.5,
    )
    _box(
        ax,
        (7.05, 1.25),
        2.05,
        2.1,
        "PIPELINE SCRIPTS\n\nread prior CSV\ncall library\nwrite next CSV\nstages 01-11",
        facecolor=LIGHT_GREEN,
        edgecolor=GREEN,
        fontsize=9.5,
    )
    _box(
        ax,
        (9.85, 1.25),
        1.9,
        2.1,
        "OUTPUTS\n\nresults CSVs\nfigures\nreports\nreproducibility record",
        facecolor=LIGHT_GRAY,
        edgecolor=GRAY,
        fontsize=9.5,
    )
    for start, end in (((2.5, 2.3), (3.1, 2.3)), ((6.35, 2.3), (7.0, 2.3)), ((9.15, 2.3), (9.8, 2.3))):
        _arrow(ax, start, end)
    ax.text(4.72, 4.36, "Unit tests validate reusable logic independently of GPU runs", ha="center", fontsize=10, color=NAVY)
    _save(fig, "layered_architecture.png")


def build_sample_distribution() -> None:
    """Plot the pilot composition and the verified multi-label recovery."""
    with (PILOT_DIR / "data" / "pilot_samples.csv").open(encoding="utf-8", newline="") as f:
        pilot_counts = Counter(row["primary_class"] for row in csv.DictReader(f))

    metadata_path = PILOT_DIR / "data" / "constructionsite" / "extracted_images" / "test_metadata.csv"
    with metadata_path.open(encoding="utf-8", newline="") as f:
        metadata = list(csv.DictReader(f))
    priority_counts = Counter(row["primary_class"] for row in metadata)
    multi_counts = Counter()
    for row in metadata:
        for label in row["hazard_classes"].split("|"):
            if label:
                multi_counts[label] += 1

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"wspace": 0.35})

    labels = ["PPE", "Fall", "Struck-by", "Compliant"]
    keys = ["ppe_violation", "fall_hazard", "struck_by_risk", "compliant"]
    values = [pilot_counts[key] for key in keys]
    bars = axes[0].bar(labels, values, color=[BLUE, BLUE, GOLD, BLUE], width=0.68)
    axes[0].set_title("A. Pilot sample composition", loc="left", weight="bold", color=NAVY)
    axes[0].set_ylabel("Selected images")
    axes[0].set_ylim(0, 58)
    axes[0].grid(axis="y", color="#D7DCE0", linewidth=0.8)
    axes[0].set_axisbelow(True)
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 1.2, str(value), ha="center", weight="bold")

    minority = ["Fall hazard", "Struck-by risk"]
    priority = [priority_counts["fall_hazard"], priority_counts["struck_by_risk"]]
    multi = [multi_counts["fall_hazard"], multi_counts["struck_by_risk"]]
    x = [0, 1]
    width = 0.34
    b1 = axes[1].bar([v - width / 2 for v in x], priority, width, label="Priority label", color=GRAY)
    b2 = axes[1].bar([v + width / 2 for v in x], multi, width, label="Multi-label coverage", color=GREEN)
    axes[1].set_xticks(x, minority)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Annotated test-split images")
    axes[1].set_title("B. Labels recovered by multi-label counting", loc="left", weight="bold", color=NAVY)
    axes[1].grid(axis="y", color="#D7DCE0", linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, loc="upper right")
    for bars_ in (b1, b2):
        for bar in bars_:
            value = int(bar.get_height())
            axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1.5, str(value), ha="center", weight="bold")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Sampling shortfall and its verified labeling mechanism", weight="bold", fontsize=14, color=DARK, y=1.02)
    _save(fig, "sample_distribution.png")


def build_proxy_comparison() -> None:
    """Show scene-level existence versus person-relative coverage."""
    fig, ax = plt.subplots(figsize=(11.6, 4.3))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 4.3)
    ax.axis("off")

    ax.text(2.75, 3.93, "Version 1: scene-level existence", ha="center", fontsize=12, weight="bold", color=RED)
    _box(ax, (0.25, 2.25), 2.0, 0.9, "Detect\nsafety object", facecolor=LIGHT_RED, edgecolor=RED)
    _arrow(ax, (2.3, 2.7), (3.0, 2.7), color=RED)
    _box(ax, (3.05, 2.25), 2.0, 0.9, "Any box\nanywhere?", facecolor=LIGHT_RED, edgecolor=RED)
    ax.text(2.65, 1.48, "Sensitivity 3.0%\nSpecificity 97.4%", ha="center", fontsize=11, weight="bold", color=RED)
    ax.text(2.65, 0.75, "Near-constant compliant verdict", ha="center", fontsize=9.5, color=GRAY)

    ax.plot([5.8, 5.8], [0.4, 4.0], color="#C7CDD2", linewidth=1.3)

    ax.text(8.75, 3.93, "Version 2: person-relative coverage", ha="center", fontsize=12, weight="bold", color=GREEN)
    _box(ax, (6.15, 2.25), 1.7, 0.9, "Detect\nworkers", facecolor=LIGHT_GREEN, edgecolor=GREEN)
    _box(ax, (6.15, 0.85), 1.7, 0.9, "Detect\nsafety objects", facecolor=LIGHT_GREEN, edgecolor=GREEN)
    _arrow(ax, (7.9, 2.7), (8.65, 2.45), color=GREEN)
    _arrow(ax, (7.9, 1.3), (8.65, 2.15), color=GREEN)
    _box(ax, (8.7, 1.8), 2.25, 1.1, "Does every worker\nhave nearby protection?", facecolor=LIGHT_GREEN, edgecolor=GREEN)
    ax.text(9.82, 1.18, "Sensitivity 27.0%\nSpecificity 73.7%", ha="center", fontsize=11, weight="bold", color=GREEN)
    ax.text(9.82, 0.48, "9x sensitivity improvement", ha="center", fontsize=9.5, color=GRAY)
    _save(fig, "proxy_logic_comparison.png")


def build_ranking_correction() -> None:
    """Chart top-region identity before and after rule-aware ranking."""
    results_dir = PILOT_DIR / "results"
    files = [
        ("Area ranking", results_dir / "region_extraction.csv"),
        ("Rule-aware ranking", results_dir / "region_extraction_rule_aware.csv"),
    ]
    data: list[tuple[str, float, float]] = []
    for label, path in files:
        with path.open(encoding="utf-8", newline="") as f:
            rows = [row for row in csv.DictReader(f) if row["top_region_source"] == "model"]
        worker = sum(row["top_region_label"] == "worker" for row in rows) / len(rows) * 100
        data.append((label, worker, 100 - worker))

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    labels = [row[0] for row in data]
    worker = [row[1] for row in data]
    safety = [row[2] for row in data]
    y = [0, 1]
    ax.barh(y, worker, color=RED, label="Worker body")
    ax.barh(y, safety, left=worker, color=GREEN, label="Safety object")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of 159 real-box samples (%)")
    ax.set_title("Rule-aware ranking moves the safety object to rank 1", loc="left", weight="bold", color=NAVY)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(axis="x", color="#D7DCE0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for i, (worker_value, safety_value) in enumerate(zip(worker, safety)):
        if worker_value < 4:
            ax.text(
                worker_value + 1.0,
                i,
                f"{worker_value:.1f}%",
                va="center",
                ha="left",
                color=RED,
                weight="bold",
            )
        else:
            ax.text(worker_value / 2, i, f"{worker_value:.1f}%", va="center", ha="center", color="white", weight="bold")
        ax.text(worker_value + safety_value / 2, i, f"{safety_value:.1f}%", va="center", ha="center", color="white", weight="bold")
    _save(fig, "ranking_correction.png")


def build_phase_status() -> None:
    """Summarize completed, in-progress, and gated scale-up phases."""
    fig, ax = plt.subplots(figsize=(11.8, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    phases = [
        ("0", "Decisions", "SIGN-OFF", LIGHT_GOLD, GOLD),
        ("1", "Correctness", "DONE", LIGHT_GREEN, GREEN),
        ("2", "Metric validity", "DONE", LIGHT_GREEN, GREEN),
        ("3", "Statistics and\nreproducibility", "IN PROGRESS", LIGHT_BLUE, NAVY),
        ("4", "Attribution", "GATED", LIGHT_GRAY, GRAY),
        ("5", "Models and\ndatasets", "GATED", LIGHT_GRAY, GRAY),
    ]
    for i, (number, label, status, fill, edge) in enumerate(phases):
        x = 0.15 + i * 1.96
        _box(ax, (x, 0.8), 1.62, 1.55, f"PHASE {number}\n{label}\n\n{status}", facecolor=fill, edgecolor=edge, fontsize=8.8)
        if i < len(phases) - 1:
            _arrow(ax, (x + 1.65, 1.57), (x + 1.9, 1.57), color="#929BA2")
    ax.text(6, 2.85, "Scale-up status at the time of the meeting", ha="center", fontsize=13, weight="bold", color=NAVY)
    ax.text(6, 0.28, "Phases 4-5 begin only after the attribution and model-scope decisions are approved.", ha="center", fontsize=9.5, color=GRAY)
    _save(fig, "phase_status.png")


def main() -> None:
    """Build every generated figure used by the final briefing."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": DARK,
            "xtick.color": DARK,
            "ytick.color": DARK,
        }
    )
    build_study_overview()
    build_pipeline()
    build_architecture()
    build_sample_distribution()
    build_proxy_comparison()
    build_ranking_correction()
    build_phase_status()
    print(f"Built briefing figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
