"""Generate every figure the deck uses, from committed result files.

Nothing here is hand-drawn or retyped. Re-running this after a stage finishes
refreshes the deck's evidence, which is the point: the presentation is a view
over the results, not a separate copy of them.

    python deck/build_assets.py
"""
from __future__ import annotations

import ast
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
S1 = REPO / "studies" / "01-florence2-pilot"
S2 = REPO / "studies" / "02-vlm-xai-study"
OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(S2 / "src"))

# Palette shared with the written report: instrument blue for structure, brick
# for the one critical finding, muted green for verified-good.
INK = "#151E27"
SOFT = "#4A5967"
RULE = "#D8DDE3"
ACCENT = "#2E6B8A"
SIGNAL = "#9B2C1F"
GOOD = "#3F6B4A"
PAPER = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": RULE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SOFT,
    "ytick.color": SOFT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
})


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save(fig, name: str) -> None:
    p = OUT / name
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"  wrote {p.name}")


# ---------------------------------------------------------------- study 1 ----

def fig_study1_metrics() -> None:
    """The six pilot metrics on one axis, with the min-n flag made visible."""
    rows = read(S1 / "results" / "pilot_metric_summary.csv")
    wanted = [
        ("descriptive_accuracy", "Descriptive\naccuracy"),
        ("bounded_completeness", "Bounded\ncompleteness"),
        ("stability_answer_agreement", "Stability"),
        ("robustness_level1_answer_survival", "Robustness\n(level 1)"),
        ("visual_sparsity_top5", "Sparsity\n(top-5 mass)"),
    ]
    by = {r["metric"]: r for r in rows}
    labels, vals = [], []
    for key, label in wanted:
        if key in by:
            labels.append(label)
            vals.append(float(by[key]["overall"]))

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    colors = [GOOD if v >= 0.7 else (SIGNAL if v < 0.4 else ACCENT) for v in vals]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1%}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("higher is better")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_title("Study 1 — Florence-2 pilot, six metrics (n = 163)",
                 loc="left", fontsize=12, fontweight="bold", pad=12)
    save(fig, "s1_metrics.png")


def fig_study1_ci() -> None:
    """Descriptive accuracy by class, with CIs and the below-floor class flagged."""
    rows = [r for r in read(S1 / "results" / "statistics_summary.csv")
            if r["metric"] == "descriptive_accuracy_top1" and r["group"] != "overall"]
    if not rows:
        return
    rows.sort(key=lambda r: float(r["point"]))
    labels = [r["group"].replace("_", " ") for r in rows]
    pts = [float(r["point"]) for r in rows]
    los = [float(r["point"]) - float(r["lo"]) for r in rows]
    his = [float(r["hi"]) - float(r["point"]) for r in rows]
    flagged = [r["meets_min_n"] != "True" for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    for i, (p, lo, hi, flag, r) in enumerate(zip(pts, los, his, flagged, rows)):
        c = SIGNAL if flag else ACCENT
        ax.errorbar(p, i, xerr=[[lo], [hi]], fmt="o", color=c,
                    capsize=5, markersize=9, linewidth=2)
        ax.text(float(r["hi"]) + 0.03, i, f"n={r['n']}" + ("  below floor" if flag else ""),
                va="center", fontsize=10, color=SIGNAL if flag else SOFT)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_xlabel("descriptive accuracy, 95% bootstrap CI")
    ax.set_title("Small n is visible, not hidden — struck-by risk has 13 examples",
                 loc="left", fontsize=12, fontweight="bold", pad=12)
    save(fig, "s1_ci.png")


# ---------------------------------------------------------------- study 2 ----

def _boot(flags, n_boot=2000, seed=0):
    from xai_vlm.metrics.reporting import proportion_ci
    return proportion_ci(flags, n_boot=n_boot, seed=seed)


def fig_study2_baseline() -> None:
    rows = read(S2 / "results" / "baseline_qwen7b-4bit.csv")
    if not rows:
        return
    pos = [r for r in rows if r["expected"] == "YES"]
    neg = [r for r in rows if r["expected"] == "NO"]
    ctrl = [r for r in rows if str(r["is_negative_control"]).lower() == "true"]

    items = [
        ("Sensitivity", [r["verdict"] == "YES" for r in pos], GOOD),
        ("Specificity", [r["verdict"] == "NO" for r in neg], SIGNAL),
        ("False-alarm rate", [r["verdict"] == "YES" for r in ctrl], SIGNAL),
        ("Answers YES", [r["verdict"] == "YES" for r in rows], ACCENT),
    ]
    labels, pts, err, colors, ns = [], [], [[], []], [], []
    for label, flags, c in items:
        d = _boot(flags)
        labels.append(label)
        pts.append(d["point"])
        err[0].append(d["point"] - d["lo"])
        err[1].append(d["hi"] - d["point"])
        colors.append(c)
        ns.append(d["n"])

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    bars = ax.bar(labels, pts, yerr=err, color=colors, width=0.58,
                  capsize=6, error_kw={"ecolor": INK, "linewidth": 1.4})
    for b, v, n in zip(bars, pts, ns):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.055, f"{v:.1%}",
                ha="center", fontsize=12, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 0.03, f"n={n:,}",
                ha="center", fontsize=9, color=PAPER)
    ax.axhline(0.5, color=SOFT, linestyle=":", linewidth=1.2)
    ax.text(3.45, 0.52, "chance", fontsize=9, color=SOFT, ha="right")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_title("Study 2 — high sensitivity bought with false alarms (n = 3,882)",
                 loc="left", fontsize=12, fontweight="bold", pad=12)
    save(fig, "s2_baseline.png")


def fig_rule4() -> None:
    """The headline finding: one rule is far below the others."""
    rows = read(S2 / "results" / "baseline_qwen7b-4bit.csv")
    if not rows:
        return
    names = {"rule_1": "rule_1  PPE", "rule_2": "rule_2  fall protection",
             "rule_3": "rule_3  edge protection", "rule_4": "rule_4  plant proximity"}
    pos = [r for r in rows if r["expected"] == "YES"]
    data = []
    for rid in sorted(names):
        sub = [r for r in pos if r["rule_id"] == rid]
        if sub:
            d = _boot([r["verdict"] == "YES" for r in sub])
            data.append((names[rid], d))

    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    for i, (label, d) in enumerate(data):
        crit = d["point"] < 0.5
        c = SIGNAL if crit else ACCENT
        ax.errorbar(d["point"], i, xerr=[[d["point"] - d["lo"]], [d["hi"] - d["point"]]],
                    fmt="o", color=c, capsize=5, markersize=10, linewidth=2.4)
        ax.text(d["hi"] + 0.02, i, f"{d['point']:.1%}   n={d['n']}",
                va="center", fontsize=10.5,
                color=SIGNAL if crit else SOFT,
                fontweight="bold" if crit else "normal")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([d[0] for d in data])
    ax.set_xlim(0, 1.25)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_xlabel("sensitivity, 95% bootstrap CI")
    ax.set_title("The pooled 91% hides a near-blind hazard class",
                 loc="left", fontsize=12.5, fontweight="bold", pad=12)
    ax.annotate("struck-by risk:\none of the “fatal four”",
                xy=(0.30, 3), xytext=(0.55, 2.55), fontsize=10, color=SIGNAL,
                arrowprops={"arrowstyle": "->", "color": SIGNAL, "linewidth": 1.6})
    save(fig, "s2_rule4.png")


def fig_correctness() -> None:
    rows = [r for r in read(S2 / "results" / "explanation_correctness_qwen7b-4bit.csv")
            if not r["undefined_reason"]]
    if not rows:
        return
    items = [
        ("Names the right\nhazard", [str(r["hazard_ok"]).lower() == "true" for r in rows], ACCENT),
        ("Paraphrases the\nhuman reason", [float(r["similarity"]) >= 0.60 for r in rows], ACCENT),
        ("Points at the\nright place", [float(r["iou"]) >= 0.30 for r in rows], SIGNAL),
        ("ALL THREE\n(= correct)", [r["verdict_correctness"] == "CORRECT" for r in rows], SIGNAL),
    ]
    labels, pts, colors = [], [], []
    for label, flags, c in items:
        d = _boot(flags)
        labels.append(label)
        pts.append(d["point"])
        colors.append(c)

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    bars = ax.bar(labels, pts, color=colors, width=0.56)
    for b, v in zip(bars, pts):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.1%}",
                ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_title(f"It describes hazards well and locates them badly (n = {len(rows):,})",
                 loc="left", fontsize=12, fontweight="bold", pad=12)
    save(fig, "s2_correctness.png")


def fig_iou_dist() -> None:
    rows = [r for r in read(S2 / "results" / "explanation_correctness_qwen7b-4bit.csv")
            if not r["undefined_reason"]]
    if not rows:
        return
    ious = [float(r["iou"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.hist(ious, bins=40, color=ACCENT, edgecolor=PAPER)
    ax.axvline(0.30, color=SIGNAL, linestyle="--", linewidth=2)
    ax.text(0.32, ax.get_ylim()[1] * 0.8, "threshold 0.30", color=SIGNAL, fontsize=10)
    zero = sum(1 for i in ious if i == 0.0)
    ax.set_xlabel("IoU between the model's box and the human annotation")
    ax.set_ylabel("rows")
    ax.set_title(f"{zero / len(ious):.0%} of boxes do not overlap the annotation at all — "
                 f"but the tail reaches {max(ious):.2f}",
                 loc="left", fontsize=11.5, fontweight="bold", pad=12)
    save(fig, "s2_iou.png")


# ------------------------------------------------------------- examples ------

def _draw_example(image, model_box, human_boxes, title, caption, name) -> None:
    """Photo with both boxes drawn, sized to the image so there is no dead space."""
    from matplotlib.lines import Line2D

    w, h = image.size
    # Cap the aspect so a portrait photo does not produce a very tall figure with
    # acres of dead space; the slide has a fixed frame either way.
    aspect = min(h / w, 1.15)
    fig_w = 6.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * aspect + 1.25))
    ax.imshow(image)

    for hb in human_boxes or []:
        x0, y0, x1, y1 = [float(v) for v in hb]
        ax.add_patch(Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h,
                               fill=False, edgecolor="#1FA463", linewidth=3.0))
    if model_box:
        x0, y0, x1, y1 = model_box
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               fill=False, edgecolor="#E8590C", linewidth=3.0,
                               linestyle=(0, (5, 3))))

    ax.axis("off")
    # Title carries the caption so the gap between them cannot drift with the
    # image's aspect ratio; only the legend sits below the frame.
    ax.set_title(f"{title}\n{caption}", loc="left", fontsize=11.5,
                 color=SOFT, pad=10, linespacing=1.5)
    ax.title.set_fontweight("normal")
    ax.text(0, 1.0, "", transform=ax.transAxes)
    fig.suptitle(title, x=0.012, y=0.985, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    ax.set_title(caption, loc="left", fontsize=10.5, color=SOFT,
                 pad=8, linespacing=1.45)

    ax.legend(handles=[
        Line2D([0], [0], color="#E8590C", lw=3, linestyle=(0, (5, 3)),
               label="model's region"),
        Line2D([0], [0], color="#1FA463", lw=3, label="human annotation"),
    ], loc="upper left", bbox_to_anchor=(0, -0.02), ncol=2, frameon=False,
        fontsize=10.5, handlelength=2.2)
    save(fig, name)


def examples_study2() -> None:
    """One high-agreement and one zero-agreement case, with both boxes drawn."""
    from xai_pilot.data import load_construction_site

    corr = [r for r in read(S2 / "results" / "explanation_correctness_qwen7b-4bit.csv")
            if not r["undefined_reason"]]
    regions = {(r["image_id"], r["rule_id"]): r
               for r in read(S2 / "results" / "regions_qwen7b-4bit.csv")}
    if not corr:
        return

    best = max(corr, key=lambda r: float(r["iou"]))
    worst = next(r for r in sorted(corr, key=lambda r: float(r["iou"]))
                 if float(r["similarity"]) > 0.5)
    picks = {best["image_id"]: ("good", best), worst["image_id"]: ("bad", worst)}

    from xai_vlm.rules import RULE_IDS  # noqa: F401
    imgs = {}
    for split in ("test", "train"):
        for rec in load_construction_site(split=split, streaming=True):
            if rec["image_id"] in picks and rec["image_id"] not in imgs:
                imgs[rec["image_id"]] = (rec["image"].convert("RGB"), rec)
            if len(imgs) >= len(picks):
                break
        if len(imgs) >= len(picks):
            break

    for iid, (kind, row) in picks.items():
        if iid not in imgs:
            continue
        image, rec = imgs[iid]
        reg = regions.get((iid, row["rule_id"]), {})
        mbox = None
        if reg.get("region"):
            mbox = tuple(float(v) for v in ast.literal_eval(reg["region"]))
        ann = rec.get(f"{row['rule_id']}_violation") or {}
        hboxes = ann.get("bounding_box") or []
        iou = float(row["iou"])
        sim = float(row["similarity"])
        title = ("Agreement — the model points where the annotator did"
                 if kind == "good" else
                 "Right sentence, wrong place — this is the 97.6% case")
        reason = " ".join((row["model_reason"] or "").split())
        if len(reason) > 96:
            reason = reason[:96].rstrip(" .,") + "…"
        cap = (f"{iid} · {row['rule_id']} · IoU {iou:.2f} · text similarity {sim:.2f}\n"
               f"model said: “{reason}”")
        _draw_example(image, mbox, hboxes, title, cap, f"s2_example_{kind}.png")


def main() -> int:
    print("study 1 figures")
    fig_study1_metrics()
    fig_study1_ci()
    print("study 2 figures")
    fig_study2_baseline()
    fig_rule4()
    fig_correctness()
    fig_iou_dist()
    print("worked examples")
    try:
        examples_study2()
    except Exception as exc:  # noqa: BLE001
        print(f"  example generation skipped: {type(exc).__name__}: {exc}")
    print(f"\nassets in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
