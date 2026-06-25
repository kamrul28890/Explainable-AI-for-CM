"""Day 9 deliverable: do answers survive image perturbation and prompt rewording?

Level 1 perturbs the image (blur, low_light, occlude, contrast_shift) and
reruns answer_rule with the same deterministic decoding as the logged
baseline (num_beams=3, do_sample=False) so perturbation sensitivity isn't
confounded with Day 7's sampling-noise sensitivity. Level 2 reruns the
unperturbed image with the rule's second phrasing instead (RULE_QUERIES
index 1) -- rule_4 has no second phrasing (it's a fixed proximity pair, not
a phrase list) and is excluded from Level 2, not silently scored as 0%
change.

Stretch (bounded, droppable): pastes a synthetic hard-hat patch over the
approximate head region of 20 ppe_violation/rule_1 workers and checks
whether the answer flips from "violation" to "compliant".
"""

import json
import sys

import pandas as pd
from PIL import Image, ImageDraw

from xai_pilot.config import FIGURES_DIR, RESULTS_DIR
from xai_pilot.data import load_construction_site
from xai_pilot.inference import AnswerResult, answer_rule
from xai_pilot.metrics.robustness import evaluate
from xai_pilot.model import load_florence2
from xai_pilot.perturbations import blur, contrast_shift, low_light, occlude, paste_patch
from xai_pilot.viz import overlay_boxes, save_figure

N_VISUALIZE_PER_PERTURBATION = 2
N_PATCH_STRETCH = 20

LEVEL1_PERTURBATIONS = {
    "blur": blur,
    "low_light": low_light,
    "occlude": occlude,
    "contrast_shift": contrast_shift,
}


def _baseline_from_row(row) -> AnswerResult:
    """Reconstruct a typed baseline result from one CSV record."""
    return AnswerResult(
        answer=row["answer"],
        worker_boxes=[tuple(b) for b in json.loads(row["worker_boxes"])],
        object_boxes=[tuple(b) for b in json.loads(row["object_boxes"])],
        confidence=float(row["confidence"]),
    )


def _make_hardhat_patch(size=(60, 60)) -> Image.Image:
    """Create the deliberately simple synthetic patch used by the stretch test.

    This is not intended to be photorealistic. It tests whether adding a
    high-contrast hard-hat-like cue near the estimated head can alter the
    grounding proxy.
    """
    patch = Image.new("RGB", size, (235, 220, 200))
    draw = ImageDraw.Draw(patch)
    draw.ellipse((2, 2, size[0] - 2, size[1] - 6), fill=(255, 210, 0))
    return patch


def _head_box(worker_box, scale: float = 0.22) -> tuple[float, float, float, float]:
    """Estimate a centered head region from the upper portion of a worker box."""
    wx0, wy0, wx1, wy1 = worker_box
    w_width, w_height = wx1 - wx0, wy1 - wy0
    head_h = w_height * scale
    head_w = min(w_width, head_h * 1.3)
    head_x0 = wx0 + (w_width - head_w) / 2
    return (head_x0, wy0, head_x0 + head_w, wy0 + head_h)


def main() -> int:
    """Run image, prompt, and bounded synthetic-patch robustness checks."""
    preds_df = pd.read_csv(RESULTS_DIR / "baseline_predictions.csv", dtype=str)
    target_ids = set(preds_df["image_id"])

    print("Loading Florence-2-base-ft...")
    model, processor = load_florence2()

    print(f"Streaming test split to fetch {len(target_ids)} images...")
    ds = load_construction_site(split="test", streaming=True)
    images_by_id = {}
    for row in ds:
        if row["image_id"] in target_ids:
            images_by_id[row["image_id"]] = row["image"]
        if len(images_by_id) >= len(target_ids):
            break

    fig_dir = FIGURES_DIR / "robustness"
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    visualized_count = {name: 0 for name in LEVEL1_PERTURBATIONS}
    # Every Level 1 perturbation starts from the original image. Perturbations
    # are not composed, which keeps each measured factor interpretable.
    for i, row in preds_df.iterrows():
        image_id = row["image_id"]
        rule_id = row["assigned_rule_id"]
        image = images_by_id[image_id]
        baseline = _baseline_from_row(row)

        # --- Level 1: image perturbations ---
        for name, fn in LEVEL1_PERTURBATIONS.items():
            perturbed_image = fn(image)
            result = evaluate(model, processor, perturbed_image, rule_id, baseline)
            out_rows.append(
                {
                    "image_id": image_id,
                    "assigned_rule_id": rule_id,
                    "primary_class": row["primary_class"],
                    "level": "level1_image",
                    "perturbation": name,
                    "answer_changed": result.answer_changed,
                    "confidence_drop": result.confidence_drop,
                    "object_box_iou": result.object_box_iou,
                    "worker_lost": result.worker_lost,
                    "excluded_reason": None,
                }
            )
            if visualized_count[name] < N_VISUALIZE_PER_PERTURBATION:
                save_figure(perturbed_image, fig_dir / f"{image_id}_{rule_id}_{name}.png")
                visualized_count[name] += 1

        # --- Level 2: reworded prompt (rule_4 has no second phrasing) ---
        # Rule 4 is defined by a fixed worker/excavator pair and has no honest
        # synonym variant. Record an explicit exclusion instead of treating it
        # as an unchanged answer.
        if rule_id == "rule_4":
            out_rows.append(
                {
                    "image_id": image_id,
                    "assigned_rule_id": rule_id,
                    "primary_class": row["primary_class"],
                    "level": "level2_prompt",
                    "perturbation": "reworded_prompt",
                    "answer_changed": None,
                    "confidence_drop": None,
                    "object_box_iou": None,
                    "worker_lost": None,
                    "excluded_reason": "rule_4_has_no_second_phrasing",
                }
            )
        else:
            reworded = answer_rule(model, processor, image, rule_id, phrasing_index=1)
            # Reuse the same comparison logic as image perturbations so answer,
            # confidence, box drift, and worker-loss semantics stay identical.
            from xai_pilot.metrics.robustness import _robustness_from_results

            result = _robustness_from_results(baseline, reworded)
            out_rows.append(
                {
                    "image_id": image_id,
                    "assigned_rule_id": rule_id,
                    "primary_class": row["primary_class"],
                    "level": "level2_prompt",
                    "perturbation": "reworded_prompt",
                    "answer_changed": result.answer_changed,
                    "confidence_drop": result.confidence_drop,
                    "object_box_iou": result.object_box_iou,
                    "worker_lost": result.worker_lost,
                    "excluded_reason": None,
                }
            )

        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(preds_df)} done...")

    out_csv = RESULTS_DIR / "robustness.csv"
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(out_df)} rows to {out_csv}")

    # Preserve excluded records in the artifact, but report rates only for
    # conditions that were actually evaluated.
    scored = out_df[out_df["excluded_reason"].isna()].copy()
    scored["answer_changed"] = scored["answer_changed"].astype(bool)
    scored["worker_lost"] = scored["worker_lost"].astype(bool)
    print("\nanswer_changed rate by perturbation:")
    print(scored.groupby("perturbation")["answer_changed"].mean())
    print("\nworker_lost rate by perturbation (Day-5-flagged fallback-rerouting risk):")
    print(scored.groupby("perturbation")["worker_lost"].mean())
    print("\nmean confidence_drop by perturbation:")
    print(scored.groupby("perturbation")["confidence_drop"].mean())

    # --- Stretch: synthetic hard-hat patch on ppe_violation/rule_1 samples ---
    print(f"\nRunning stretch patch test on up to {N_PATCH_STRETCH} ppe_violation/rule_1 samples...")
    # A hard-hat patch is conceptually relevant only to rule 1. Requiring a
    # detected worker also provides a location for the synthetic intervention.
    candidates = preds_df[(preds_df["primary_class"] == "ppe_violation") & (preds_df["assigned_rule_id"] == "rule_1")]
    candidates = candidates[candidates["worker_boxes"].apply(lambda s: len(json.loads(s)) > 0)].head(N_PATCH_STRETCH)

    patch = _make_hardhat_patch()
    patch_rows = []
    patch_visualized = 0
    for _, row in candidates.iterrows():
        image_id = row["image_id"]
        image = images_by_id[image_id]
        baseline = _baseline_from_row(row)
        # Use the first worker because the current Florence-2 query frequently
        # returns only one worker even in multi-worker scenes, a known pilot
        # limitation documented in the report.
        head_box = _head_box(baseline.worker_boxes[0])
        patched_image = paste_patch(image, head_box, patch)
        result = answer_rule(model, processor, patched_image, "rule_1")
        patch_rows.append(
            {
                "image_id": image_id,
                "baseline_answer": baseline.answer,
                "patched_answer": result.answer,
                "flipped_to_compliant": baseline.answer == "violation" and result.answer == "compliant",
            }
        )
        if patch_visualized < 3:
            overlay = overlay_boxes(patched_image, [head_box], labels=["synthetic patch"], color="lime")
            save_figure(overlay, fig_dir / f"{image_id}_patch_stretch.png")
            patch_visualized += 1

    patch_df = pd.DataFrame(patch_rows)
    patch_csv = RESULTS_DIR / "robustness_patch_stretch.csv"
    patch_df.to_csv(patch_csv, index=False)
    print(f"Wrote {len(patch_df)} rows to {patch_csv}")
    print(
        f"Naive flip-to-compliant rate over all {len(patch_df)} candidates: "
        f"{patch_df['flipped_to_compliant'].mean():.1%} ({patch_df['flipped_to_compliant'].sum()}/{len(patch_df)})"
    )
    # "Fooling" is only a meaningful concept where the baseline actually said
    # violation -- candidates the proxy already called compliant (a known
    # baseline sensitivity gap, not this test's subject) can't demonstrate
    # anything by staying compliant, so they'd silently dilute the rate above.
    fooled_eligible = patch_df[patch_df["baseline_answer"] == "violation"]
    if len(fooled_eligible):
        print(
            f"Flip-to-compliant rate among the {len(fooled_eligible)} candidates baseline already called "
            f"'violation' (the only ones fooling is meaningful for): "
            f"{fooled_eligible['flipped_to_compliant'].mean():.1%} "
            f"({fooled_eligible['flipped_to_compliant'].sum()}/{len(fooled_eligible)})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
