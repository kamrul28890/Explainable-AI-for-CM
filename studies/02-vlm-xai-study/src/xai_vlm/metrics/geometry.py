"""Box arithmetic for metric 7.

THE COORDINATE TRAP. Dataset annotation boxes are normalised to 0-1 as fractions
of width and height; the model emits absolute pixels. Comparing them without
converting puts every annotation in the top-left corner and silently produces an
IoU near zero for every row -- a study-wide null result that looks like a finding.
The architecture document calls this out specifically (section 5.2).

Everything here works in NORMALISED space. Convert once, at the edge.
"""

from __future__ import annotations

Box = tuple[float, float, float, float]


def to_normalised(box: Box, width: int, height: int) -> Box:
    """Absolute-pixel box -> fraction-of-image box."""
    x0, y0, x1, y1 = box
    return (x0 / width, y0 / height, x1 / width, y1 / height)


def iou(a: Box, b: Box) -> float:
    """Intersection over union. Both boxes must be in the same space."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def max_iou(model_box: Box, human_boxes: list) -> float:
    """Best agreement with any annotated box.

    The model is asked for the single most relevant area, so agreeing with any one
    annotated instance is success. Averaging over boxes would penalise a correct
    answer on a photograph containing three separate violations.
    """
    if not human_boxes:
        return 0.0
    return max(iou(model_box, tuple(h)) for h in human_boxes)
