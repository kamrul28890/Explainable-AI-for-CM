"""Stage 6 -- Metric 3: Stability.

Ask the same question five times with SAMPLING decoding, not the frozen
GENERATION config. With greedy decoding the model is deterministic and five
reruns would be byte-identical -- 100% stable by construction, proving nothing.
The override is part of the measurement, not a drift from it, and is recorded
in the stage's fingerprint so a resumed run cannot silently mix sampling
settings.
"""

from __future__ import annotations

import ast
import itertools
import math

NAN = float("nan")

Box = tuple[float, float, float, float]


def pairwise_agreement(verdicts: list[str]) -> float:
    """Share of all C(n,2) pairs among the reruns that gave the same verdict.

    Not "matches the majority" -- pairwise agreement is defined even when there
    is no majority (e.g. 2-2-1 over 5 runs), and it is the same statistic Study
    1's stability metric used, so the two studies report on one definition.
    """
    if len(verdicts) < 2:
        return NAN
    pairs = list(itertools.combinations(verdicts, 2))
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def region_presence_rate(regions: list[str]) -> float:
    """Fraction of reruns that offered a region at all."""
    if not regions:
        return NAN
    return sum(1 for r in regions if (r or "").strip()) / len(regions)


# A region "moved" when its centroid shifts by more than this share of the image
# diagonal. One cut-off for Stage 6 (between reruns) and Stage 8 (under a
# perturbation), so the two drift rates read on one scale.
DRIFT_FLAG = 0.10


def centroid(box: Box) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def normalized_centroid_distance(a: Box, b: Box, image_size: tuple[int, int]) -> float:
    """Centroid drift between two boxes, normalised by the image diagonal.

    Size-invariant, unlike IoU: IoU collapses toward 0 for two small boxes that
    jitter by a few pixels, penalising small PPE detections purely for being
    small. A centroid distance normalised by the diagonal stays comparable
    across image sizes and object scales.
    """
    ax, ay = centroid(a)
    bx, by = centroid(b)
    w, h = image_size
    diag = math.hypot(w, h)
    if diag <= 0:
        return NAN
    return math.hypot(ax - bx, ay - by) / diag


def is_flickering(regions: list[str]) -> bool:
    """True when a region appears in SOME but not ALL reruns.

    Distinguished from "never offers a region", which is consistent behaviour,
    not instability. A region that flickers in and out is worse than one that
    is consistently absent: it is unpredictable, and reporting it as merely
    "sometimes missing" would understate the problem.
    """
    present = [bool((r or "").strip()) for r in regions]
    return any(present) and not all(present)


def mean_pairwise_centroid_drift(boxes: list[Box], image_size: tuple[int, int]) -> float:
    """Mean normalised centroid distance over all pairs of boxes across reruns."""
    if len(boxes) < 2:
        return NAN
    pairs = list(itertools.combinations(boxes, 2))
    return sum(normalized_centroid_distance(a, b, image_size) for a, b in pairs) / len(pairs)


def parse_box(text: str) -> Box | None:
    try:
        v = ast.literal_eval((text or "").strip())
        return tuple(float(x) for x in v) if len(v) == 4 else None
    except (ValueError, SyntaxError, TypeError):
        return None
