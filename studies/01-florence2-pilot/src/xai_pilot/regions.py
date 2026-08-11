"""Region utilities: IoU, masking, and grid fallback for explanation regions."""

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter

Box = tuple[float, float, float, float]

# Candidate region ranking policies (Scale-up Phase 1.1).
#   "area"       -- frozen pilot behavior: largest model box first.
#   "rule_aware" -- promote regions whose label is the rule's queried object
#                   above all others, area-descending within each tier.
#   "attention"  -- rank by a real importance signal (cross-attention / IG
#                   mass); wired in Phase 4, not yet available.
RegionRanking = Literal["area", "rule_aware", "attention"]
REGION_RANKING_MODES: tuple[RegionRanking, ...] = ("area", "rule_aware", "attention")


@dataclass
class Region:
    """One candidate explanation region, ranked by standardize_regions."""

    box: Box
    source: Literal["model", "grid"]
    label: str = ""


def iou(box_a: Box, box_b: Box) -> float:
    """Return intersection-over-union for two pixel-space boxes.

    Invalid or inverted dimensions contribute zero area instead of producing a
    negative intersection. A zero-area union returns 0.0.
    """
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def mask_region(
    image: Image.Image, box: Box, mode: Literal["black", "blur", "inpaint"] = "black"
) -> Image.Image:
    """Mask one absolute-pixel box region.

    Modes (Phase 2.1 adds "inpaint" for mask-mode sensitivity):
    - "black":  fill with black -- removes all evidence but is an out-of-
      distribution artifact the model never sees in training.
    - "blur":   heavy Gaussian blur -- a less destructive alternative.
    - "inpaint": content-aware Telea inpainting (OpenCV) -- fills the region
      from its surroundings, the least out-of-distribution mask, used to confirm
      the descriptive-accuracy flip rate is not an artifact of black patches.
    """
    out = image.convert("RGB").copy()
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, out.width), min(y1, out.height)
    if x1 <= x0 or y1 <= y0:
        return out

    if mode == "black":
        patch = Image.new("RGB", (x1 - x0, y1 - y0), (0, 0, 0))
    elif mode == "blur":
        region = out.crop((x0, y0, x1, y1))
        patch = region.filter(ImageFilter.GaussianBlur(radius=25))
    elif mode == "inpaint":
        return _inpaint_region(out, (x0, y0, x1, y1))
    else:
        raise ValueError(f"unknown mode: {mode}")

    out.paste(patch, (x0, y0))
    return out


def _inpaint_region(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Telea-inpaint the box region, filling it from surrounding pixels."""
    import cv2  # local import: OpenCV is only needed for this mask mode

    x0, y0, x1, y1 = box
    arr = np.array(image)  # RGB; channel order is irrelevant to inpainting
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    filled = cv2.inpaint(arr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(filled)


def grid_fallback_regions(image_size: tuple[int, int], grid: tuple[int, int] = (4, 4)) -> list[Box]:
    """Evenly spaced grid cells covering the image, used when no grounding box exists."""
    width, height = image_size
    cols, rows = grid
    cell_w, cell_h = width / cols, height / rows
    boxes = []
    # Floating-point boundaries preserve complete coverage even when image
    # dimensions are not evenly divisible by the grid dimensions.
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell_w, r * cell_h
            boxes.append((x0, y0, x0 + cell_w, y0 + cell_h))
    return boxes


def boxes_overlap_or_close(box_a: Box, box_b: Box, distance_threshold: float = 0.0) -> bool:
    """True if two boxes overlap, or their nearest edges are within distance_threshold pixels."""
    if iou(box_a, box_b) > 0:
        return True
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    distance = float(np.hypot(dx, dy))
    return distance <= distance_threshold


def all_boxes_covered(
    subject_boxes: list[Box], reference_boxes: list[Box], distance_threshold: float = 0.0
) -> bool:
    """True if every subject box overlaps or is close to at least one reference box.

    Vacuously true if subject_boxes is empty (standard `all()` convention);
    false if subject_boxes is non-empty but reference_boxes is empty.
    """
    return all(
        any(boxes_overlap_or_close(s, r, distance_threshold=distance_threshold) for r in reference_boxes)
        for s in subject_boxes
    )


def normalized_centroid_distance(box_a: Box, box_b: Box, image_size: tuple[int, int]) -> float:
    """Distance between two box centroids, normalized by the image diagonal.

    A size-invariant drift measure (Scale-up Phase 2.2/2.4): unlike IoU, it does
    not collapse when two small boxes at the same location differ slightly in
    extent. 0.0 means coincident centroids; larger means more positional drift.
    """
    ax = (box_a[0] + box_a[2]) / 2.0
    ay = (box_a[1] + box_a[3]) / 2.0
    bx = (box_b[0] + box_b[2]) / 2.0
    by = (box_b[1] + box_b[3]) / 2.0
    diagonal = float(np.hypot(image_size[0], image_size[1]))
    return float(np.hypot(ax - bx, ay - by)) / diagonal


def fraction_covered(
    subject_boxes: list[Box], reference_boxes: list[Box], distance_threshold: float = 0.0
) -> float:
    """Fraction of subject boxes that overlap or are close to a reference box.

    The continuous counterpart of all_boxes_covered (Scale-up Phase 2.1): where
    that returns a single boolean, this returns a graded coverage signal in
    [0, 1], so masking effects can be measured as a magnitude rather than only
    as an answer flip. Returns NaN when there are no subject boxes (the fraction
    is undefined, and a worker-less rerun must not be scored as full coverage).
    """
    if not subject_boxes:
        return float("nan")
    n_covered = sum(
        any(
            boxes_overlap_or_close(s, r, distance_threshold=distance_threshold)
            for r in reference_boxes
        )
        for s in subject_boxes
    )
    return n_covered / len(subject_boxes)


def _box_area(box: Box) -> float:
    """Return non-negative pixel area for ranking candidate regions."""
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def standardize_regions(
    boxes: list[Box],
    labels: list[str],
    image_size: tuple[int, int],
    grid: tuple[int, int] = (4, 4),
    region_ranking: RegionRanking = "area",
    object_labels: Collection[str] | None = None,
) -> list[Region]:
    """Rank candidate explanation regions for one sample.

    `region_ranking` selects the policy (default ``"area"``, the frozen pilot
    behavior):

    - ``"area"``: model-returned boxes ranked by area (descending) as a
      pragmatic importance proxy -- a larger detection is assumed more salient
      than a sliver. The pilot report documents this policy's bias against
      small PPE objects (the worker's body box outranks the hard hat in ~96%
      of PPE samples), which is exactly what ``"rule_aware"`` fixes.
    - ``"rule_aware"``: regions whose label is in ``object_labels`` (the rule's
      queried safety object) are promoted above all others, so masking metrics
      test the object that actually answers the rule rather than the worker
      body. Within each tier, boxes are still area-descending. Requires a
      non-empty ``object_labels``.
    - ``"attention"``: rank by cross-attention / IG importance mass; wired in
      Phase 4, raises ``NotImplementedError`` until then.

    Model-returned boxes are always preferred over the grid. Only when the
    model returned zero boxes (rule_id wasn't grounded at all) do we fall back
    to an unranked grid, so every sample still has *some* candidate region to
    mask for the descriptive-accuracy test. Grid fallback is identical across
    ranking modes (there is nothing to rank).
    """
    if region_ranking not in REGION_RANKING_MODES:
        raise ValueError(
            f"unknown region_ranking={region_ranking!r}; expected one of {REGION_RANKING_MODES}"
        )
    if region_ranking == "attention":
        raise NotImplementedError(
            "region_ranking='attention' is a Phase 4 deliverable (cross-attention / IG mass); "
            "no importance signal is available yet"
        )

    if not boxes:
        return [
            Region(box=b, source="grid", label="grid")
            for b in grid_fallback_regions(image_size, grid)
        ]

    # `zip` preserves the model's box-label association while sorting.
    paired = list(zip(boxes, labels))
    if region_ranking == "area":
        paired.sort(key=lambda bl: _box_area(bl[0]), reverse=True)
    else:  # rule_aware
        if not object_labels:
            raise ValueError(
                "region_ranking='rule_aware' requires a non-empty object_labels set "
                "(the rule's queried safety object)"
            )
        object_set = set(object_labels)
        # Two-key sort: object-labeled regions first (tier 0), then area
        # within each tier. Python's sort is stable, so sorting by area first
        # and then by tier preserves area order inside each tier.
        paired.sort(key=lambda bl: _box_area(bl[0]), reverse=True)
        paired.sort(key=lambda bl: 0 if bl[1] in object_set else 1)
    return [Region(box=b, source="model", label=l) for b, l in paired]
