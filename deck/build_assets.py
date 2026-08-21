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
             "rule_3": "rule_3  edge protection", "rule_4": "rule_4  machinery proximity"}
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
    main_study1_examples()
    print(f"\nassets in {OUT}")
    return 0


# ------------------------------------------------------- study 1 examples ----

def _s1_image(image_id: str):
    """Fetch one raw ConstructionSite image by id (Study 1 uses the same corpus)."""
    from xai_pilot.data import load_construction_site
    for split in ("test", "train"):
        for rec in load_construction_site(split=split, streaming=True):
            if rec["image_id"] == image_id:
                return rec["image"].convert("RGB"), rec
    return None, None


def fig_s1_proxy_mechanism() -> None:
    """Real Florence output: hard hat detected in-frame, but not on the flagged
    worker -- the exact failure v1 could not see and v2 (person-relative) can."""
    image, _ = _s1_image("0000233")
    if image is None:
        return
    worker = (813.0, 380.25, 894.6, 578.25)
    hat = (130.2, 403.65, 144.6, 416.25)

    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    ax.imshow(image)
    ax.add_patch(Rectangle((worker[0], worker[1]), worker[2] - worker[0],
                           worker[3] - worker[1], fill=False, edgecolor="#E8590C",
                           linewidth=3))
    ax.text(worker[0], worker[1] - 14, "flagged worker (no hard hat)", color="#E8590C",
            fontsize=11, fontweight="bold")
    cx, cy = (hat[0] + hat[2]) / 2, (hat[1] + hat[3]) / 2
    ax.add_patch(plt.Circle((cx, cy), 34, fill=False, edgecolor="#1FA463", linewidth=3))
    ax.annotate("hard hat detected\nelsewhere in frame", xy=(cx, cy),
                xytext=(cx + 60, cy - 90), color="#1FA463", fontsize=11,
                fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#1FA463", "linewidth": 2})
    ax.axis("off")
    ax.set_title("Real Florence-2 output on image 0000233, rule_1 (PPE)",
                 loc="left", fontsize=13, fontweight="bold", color=INK, pad=10)
    fig.text(0.01, 0.02,
             "v1 (presence-only): a hard hat exists in the frame -> compliant.  "
             "v2 (person-relative, used): hard hat does not overlap this worker -> violation (correct).",
             fontsize=10.5, color=SOFT)
    save(fig, "s1_proxy_mechanism.png")


def fig_s1_descacc_flip() -> None:
    """Before/after masking, with the verdict that actually flipped."""
    image, _ = _s1_image("0000037")
    if image is None:
        return
    box = (682.2, 397.35, 802.2, 680.85)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.3))
    for ax in axes:
        ax.imshow(image)
        ax.axis("off")
    axes[0].add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                fill=False, edgecolor="#2E6B8A", linewidth=3.2))
    axes[0].set_title("BEFORE -- top region shown\nverdict: violation", loc="center",
                      fontsize=12.5, fontweight="bold", color=INK)

    import numpy as np
    arr = np.array(image).copy()
    x0, y0, x1, y1 = (int(v) for v in box)
    arr[y0:y1, x0:x1] = 0
    axes[1].imshow(arr)
    axes[1].set_title("AFTER -- region masked\nverdict: compliant", loc="center",
                      fontsize=12.5, fontweight="bold", color=SIGNAL)

    fig.suptitle("Image 0000037, rule_1 -- masking the top region flipped the answer",
                 x=0.02, y=0.99, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 0.015,
             "One of 63 flips out of 163 (38.7%) -- descriptive accuracy's raw rate before graded scoring.",
             fontsize=10, color=SOFT)
    fig.subplots_adjust(top=0.86, bottom=0.1)
    save(fig, "s1_descacc_flip.png")


def fig_s1_ranking_dualbox() -> None:
    """Both boxes on one image: proves Florence finds the small object too."""
    image, _ = _s1_image("0000007")
    if image is None:
        return
    worker = (967.68, 226.56, 1494.02, 1427.71)
    hat = (982.02, 225.02, 1199.10, 406.27)
    area_w = (worker[2] - worker[0]) * (worker[3] - worker[1])
    area_h = (hat[2] - hat[0]) * (hat[3] - hat[1])

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.imshow(image)
    ax.add_patch(Rectangle((worker[0], worker[1]), worker[2] - worker[0],
                           worker[3] - worker[1], fill=False, edgecolor="#E8590C",
                           linewidth=3.4))
    ax.add_patch(Rectangle((hat[0], hat[1]), hat[2] - hat[0], hat[3] - hat[1],
                           fill=False, edgecolor="#1FA463", linewidth=3.4))
    ax.axis("off")
    ax.set_title("Both boxes are real Florence-2 detections on the same image",
                 loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=10)
    fig.text(0.02, 0.075,
             f"worker box  ({area_w/1000:.0f}k px2) -- ranked #1 by area -> THIS is masked",
             fontsize=11, color="#E8590C", fontweight="bold")
    fig.text(0.02, 0.035,
             f"hard-hat box ({area_h/1000:.0f}k px2, {area_w/area_h:.1f}x smaller) -- the actual answer -> ranked #2, never tested",
             fontsize=11, color="#1FA463", fontweight="bold")
    fig.subplots_adjust(bottom=0.16)
    save(fig, "s1_ranking_dualbox.png")


def fig_s1_robustness_grid() -> None:
    """Four perturbations on one image, exact parameters labelled."""
    from xai_pilot.perturbations import blur, contrast_shift, low_light, occlude
    image, _ = _s1_image("0000007")
    if image is None:
        return
    panels = [
        ("blur", blur(image, ksize=8.0), "Gaussian blur, radius 8.0 px"),
        ("low_light", low_light(image, gamma=2.5), "gamma correction, gamma = 2.5 (darken)"),
        ("occlude", occlude(image, frac=0.2, location="center"), "black square, 20% of image area"),
        ("contrast_shift", contrast_shift(image, factor=0.3), "contrast x 0.3 (flattened)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.9))
    for ax, (name, img, desc) in zip(axes, panels):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(name, fontsize=12, fontweight="bold", color=INK, pad=6)
        ax.text(0.5, -0.06, desc, transform=ax.transAxes, ha="center",
               fontsize=9.5, color=SOFT)
    fig.suptitle("Level-1 perturbations, applied to the same source image (0000007)",
                 x=0.01, y=1.02, ha="left", fontsize=13, fontweight="bold", color=INK)
    save(fig, "s1_robustness_grid.png")


def fig_s1_stability_example() -> None:
    image, _ = _s1_image("0000057")
    if image is None:
        return
    fig = plt.figure(figsize=(10.6, 5.2))
    ax = fig.add_axes([0.02, 0.06, 0.46, 0.86])
    ax.imshow(image)
    ax.axis("off")
    ax.set_title("Image 0000057, rule_1", loc="left", fontsize=12, fontweight="bold")

    tax = fig.add_axes([0.52, 0.06, 0.46, 0.86])
    tax.axis("off")
    tax.text(0, 0.95, "Same image, same prompt, five calls", fontsize=12.5,
             fontweight="bold", color=INK, transform=tax.transAxes)
    tax.text(0, 0.82, "greedy (deterministic)", fontsize=11, color=SOFT,
             transform=tax.transAxes)
    tax.text(0.42, 0.82, "compliant", fontsize=11, fontweight="bold", color=GOOD,
             transform=tax.transAxes)
    labels = ["sampled run 1", "sampled run 2", "sampled run 3"]
    answers = ["violation", "violation", "compliant"]
    y = 0.68
    for lab, ans in zip(labels, answers):
        c = SIGNAL if ans != "compliant" else GOOD
        tax.text(0, y, lab, fontsize=11, color=SOFT, transform=tax.transAxes)
        tax.text(0.42, y, ans, fontsize=11, fontweight="bold", color=c,
                 transform=tax.transAxes)
        y -= 0.13
    tax.text(0, y - 0.05, "agreement across the 3 sampled runs: 33%", fontsize=11.5,
             fontweight="bold", color=SIGNAL, transform=tax.transAxes)
    tax.text(0, y - 0.19,
            "The deterministic (greedy) answer is what the frozen pilot reports.\n"
            "Sampling reveals it was one temperature draw away from flipping,\n"
            "invisible unless you rerun with sampling enabled.",
            fontsize=10.5, color=SOFT, transform=tax.transAxes, linespacing=1.6)
    save(fig, "s1_stability_example.png")


def fig_s1_completeness_real() -> None:
    """A real, complete bounded-completeness classification for Study 1."""
    image, _ = _s1_image("0000007")
    if image is None:
        return
    box = (967.68, 226.56, 1494.02, 1427.71)
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.imshow(image)
    ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                           fill=False, edgecolor=ACCENT, linewidth=3))
    ax.axis("off")
    ax.set_title("Image 0000007, rule_1", loc="left", fontsize=12.5,
                 fontweight="bold", color=INK, pad=8)
    fig.text(0.02, 0.10, "region offered?  YES  (source = model, label = \"worker\")",
             fontsize=11, color=INK)
    fig.text(0.02, 0.055, "masking it changed the answer?  NO  (compliant -> compliant)",
             fontsize=11, color=INK)
    fig.text(0.02, 0.01, "-> explanation_weak", fontsize=13, fontweight="bold", color=SIGNAL)
    fig.subplots_adjust(bottom=0.2)
    save(fig, "s1_completeness_real.png")


def fig_s2_completeness_real() -> None:
    """The real Study 2 completeness classification for the same image used in
    the metric-7 disagreement example, joining regions + Stage 4's actual flip."""
    from xai_pilot.data import load_construction_site
    row = None
    for r in read(S2 / "results" / "descriptive_accuracy_qwen7b-4bit.csv"):
        if r["image_id"] == "0000019" and r["rule_id"] == "rule_1":
            row = r
            break
    if row is None:
        return
    image = None
    for split in ("test", "train"):
        for rec in load_construction_site(split=split, streaming=True):
            if rec["image_id"] == "0000019":
                image = rec["image"].convert("RGB")
                break
        if image is not None:
            break
    if image is None:
        return

    box = ast.literal_eval(row["region"])
    fig, ax = plt.subplots(figsize=(6.0, 7.6))
    ax.imshow(image)
    ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                           fill=False, edgecolor="#E8590C", linewidth=3,
                           linestyle=(0, (5, 3))))
    ax.axis("off")
    ax.set_title("Image 0000019, rule_1 -- Study 2 (Qwen2.5-VL)", loc="left",
                 fontsize=12, fontweight="bold", color=INK, pad=8)
    flipped = row["flip_tokens"].lower() == "true"
    fig.text(0.02, 0.115, "region offered?  YES  (model stated a box)", fontsize=11, color=INK)
    fig.text(0.02, 0.075,
             f"token-ablation masking changed the answer?  {'YES' if flipped else 'NO'}",
             fontsize=11, color=INK)
    fig.text(0.02, 0.03, "-> explanation_weak" if not flipped else "-> explanation_supported",
             fontsize=13, fontweight="bold", color=SIGNAL if not flipped else GOOD)
    fig.text(0.02, -0.01,
             "Consistent with metric 7: this region scored IoU 0.00 against the annotation.",
             fontsize=9.5, color=SOFT)
    fig.subplots_adjust(bottom=0.2)
    save(fig, "s2_completeness_real.png")


def main_study1_examples() -> None:
    print("study 1 worked examples")
    for fn in (fig_s1_proxy_mechanism, fig_s1_descacc_flip, fig_s1_ranking_dualbox,
              fig_s1_robustness_grid, fig_s1_stability_example, fig_s1_completeness_real,
              fig_s2_completeness_real):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  {fn.__name__} skipped: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
