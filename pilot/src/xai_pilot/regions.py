"""Region utilities: IoU, masking, and grid fallback for explanation regions."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter

Box = tuple[float, float, float, float]


@dataclass
class Region:
    """One candidate explanation region, ranked by standardize_regions."""

    box: Box
    source: Literal["model", "grid"]
    label: str = ""


def iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-union of two (x0, y0, x1, y1) boxes."""
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
    image: Image.Image, box: Box, mode: Literal["black", "blur"] = "black"
) -> Image.Image:
    """Mask one absolute-pixel box region: black-fill or Gaussian blur."""
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
    else:
        raise ValueError(f"unknown mode: {mode}")

    out.paste(patch, (x0, y0))
    return out


def grid_fallback_regions(image_size: tuple[int, int], grid: tuple[int, int] = (4, 4)) -> list[Box]:
    """Evenly spaced grid cells covering the image, used when no grounding box exists."""
    width, height = image_size
    cols, rows = grid
    cell_w, cell_h = width / cols, height / rows
    boxes = []
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


def _box_area(box: Box) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def standardize_regions(
    boxes: list[Box],
    labels: list[str],
    image_size: tuple[int, int],
    grid: tuple[int, int] = (4, 4),
) -> list[Region]:
    """Rank candidate explanation regions for one sample.

    Florence-2 gives no attention/saliency map for these grounding-based
    answers, so there is no real per-region importance score to rank by.
    Model-returned boxes are ranked by area (descending) as the best
    available stand-in -- a larger detection is assumed more salient than a
    sliver -- and are always preferred over the grid. Only when the model
    returned zero boxes (rule_id wasn't grounded at all) do we fall back to
    an unranked 4x4 grid, so every sample still has *some* candidate region
    to mask for Day 5's descriptive-accuracy test.
    """
    if boxes:
        paired = sorted(zip(boxes, labels), key=lambda bl: _box_area(bl[0]), reverse=True)
        return [Region(box=b, source="model", label=l) for b, l in paired]
    return [Region(box=b, source="grid", label="grid") for b in grid_fallback_regions(image_size, grid)]
