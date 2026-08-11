"""Two-stratum sample construction with metadata matching.

87.2% of the dataset has no violation at all, and three of the seven metrics are
*undefined* on a NO answer -- if the model says "no violation" there is no region to
mask and no rationale to grade. So compliant images cannot contribute to metrics 1,
6 or 7 however many are run.

The design (architecture doc section 5.3) is therefore two strata:

    Stratum H   all 1,278 violation images
                -> all seven XAI metrics, and SENSITIVITY

    Stratum C   1,278 compliant images matched to H on scene metadata
                -> SPECIFICITY, and the FALSE-ALARM rate

recombined afterwards by prevalence reweighting rather than reported as a single
balanced number, which would overstate deployed precision badly.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

# Scene attributes matched between strata. If violation photos were mostly
# night-time and compliant photos mostly daylight, a measured difference might be
# about lighting rather than safety -- a confound. Matching removes it.
MATCH_FIELDS = ("illumination", "camera_distance", "view")


@dataclass(frozen=True)
class SampleRow:
    """One image-rule pair: the unit every later stage operates on."""

    image_id: str
    rule_id: str
    stratum: str            # "H" (hazard) or "C" (compliant)
    is_violation: bool
    is_negative_control: bool
    quality_of_info: str
    illumination: str = ""
    camera_distance: str = ""
    view: str = ""


def metadata_key(row: dict) -> tuple:
    """The stratification cell an image belongs to."""
    return tuple(str(row.get(f, "") or "") for f in MATCH_FIELDS)


def match_compliant_sample(
    compliant: Sequence[dict],
    target_distribution: Counter,
    n_wanted: int,
    seed: int = 42,
) -> list[dict]:
    """Draw `n_wanted` compliant images matching `target_distribution`.

    `target_distribution` counts the hazard stratum's metadata cells. We draw the
    same *proportions* from the compliant pool.

    Cells are frequently short -- a metadata combination common among violations may
    be rare among compliant images. Rather than silently returning fewer rows (which
    would break the 1:1 stratum balance and quietly bias specificity), the shortfall
    is redistributed across the remaining cells and the deficit is reported by
    `stratification_report`.
    """
    rng = random.Random(seed)

    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for row in compliant:
        by_cell[metadata_key(row)].append(row)
    for rows in by_cell.values():
        rng.shuffle(rows)

    total_target = sum(target_distribution.values())
    if total_target == 0:
        return []

    quota = {
        cell: round(n_wanted * count / total_target)
        for cell, count in target_distribution.items()
    }

    picked: list[dict] = []
    shortfall = 0
    for cell, want in quota.items():
        available = by_cell.get(cell, [])
        take = min(want, len(available))
        picked.extend(available[:take])
        by_cell[cell] = available[take:]
        shortfall += want - take

    # Redistribute the shortfall over whatever remains, largest pools first, so the
    # stratum still reaches its target size.
    if shortfall > 0:
        leftovers = [r for rows in by_cell.values() for r in rows]
        rng.shuffle(leftovers)
        picked.extend(leftovers[:shortfall])

    return picked[:n_wanted]


def stratification_report(hazard: Sequence[dict], compliant: Sequence[dict]) -> str:
    """Human-readable check that matching worked.

    Run this after building the sample. Large divergences mean the compliant pool
    could not supply the cells the hazard stratum needs, and any comparison between
    strata is then partly a comparison of scene types.
    """
    h_dist = Counter(metadata_key(r) for r in hazard)
    c_dist = Counter(metadata_key(r) for r in compliant)
    h_total, c_total = sum(h_dist.values()), sum(c_dist.values())
    if not h_total or not c_total:
        return "one stratum is empty -- cannot report matching"

    lines = [
        f"{'metadata cell':<44}{'hazard':>10}{'compliant':>12}{'delta':>9}",
        "-" * 75,
    ]
    worst = 0.0
    for cell in sorted(set(h_dist) | set(c_dist)):
        hp = h_dist[cell] / h_total
        cp = c_dist[cell] / c_total
        worst = max(worst, abs(hp - cp))
        label = " | ".join(x or "-" for x in cell)
        lines.append(f"{label[:43]:<44}{hp:>9.1%}{cp:>12.1%}{hp - cp:>+9.1%}")
    lines.append("-" * 75)
    lines.append(f"largest divergence: {worst:.1%}")
    if worst > 0.05:
        lines.append("WARNING: divergence above 5 percentage points -- the compliant "
                     "pool could not match the hazard stratum. Report this as a "
                     "limitation; stratum comparisons are partly confounded.")
    return "\n".join(lines)


def reweight_precision(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Precision (PPV) at real-world prevalence, from balanced-set measurements.

    Sensitivity and specificity are properties of the *model* and do not depend on
    how common hazards are. Prevalence is a property of the *population*. Bayes
    combines them::

                      sensitivity x prevalence
        PPV = ----------------------------------------------------
              sens x prev + (1 - spec) x (1 - prev)

    This matters more than it sounds. With sensitivity 0.80 and specificity 0.90,
    a balanced 50/50 evaluation reports precision 0.89 -- but at the true 12.8%
    prevalence the deployed precision is 0.54. Reporting only the balanced figure
    overstates real performance by 35 points.

    Returns NaN when the denominator vanishes (no positive predictions at all).
    """
    tp = sensitivity * prevalence
    fp = (1.0 - specificity) * (1.0 - prevalence)
    denom = tp + fp
    return tp / denom if denom > 0 else float("nan")


def summarize_strata(rows: Iterable[SampleRow]) -> str:
    """Counts per stratum and per rule, to eyeball before freezing the manifest."""
    rows = list(rows)
    by_stratum = Counter(r.stratum for r in rows)
    by_rule = Counter(r.rule_id for r in rows)
    pos_by_rule = Counter(r.rule_id for r in rows if r.is_violation)
    neg_controls = sum(1 for r in rows if r.is_negative_control)

    lines = [
        f"image-rule pairs : {len(rows):,}",
        f"unique images    : {len({r.image_id for r in rows}):,}",
        f"stratum H        : {by_stratum.get('H', 0):,}",
        f"stratum C        : {by_stratum.get('C', 0):,}",
        f"negative controls: {neg_controls:,}",
        "",
        f"{'rule':<10}{'asked':>8}{'positive':>10}",
    ]
    for rule in sorted(by_rule):
        lines.append(f"{rule:<10}{by_rule[rule]:>8,}{pos_by_rule[rule]:>10,}")

    # The reporting floor from the pilot's stats module: below 30 per group, a
    # number is shown but never quoted as a headline.
    thin = [r for r in by_rule if pos_by_rule[r] < 30]
    if thin:
        lines.append("")
        lines.append(f"BELOW REPORTING FLOOR (n<30 positives): {', '.join(sorted(thin))}")
    return "\n".join(lines)
