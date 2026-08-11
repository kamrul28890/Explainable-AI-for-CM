"""Scale-up Phase 3.1: attach bootstrap CIs, Wilcoxon tests, and the minimum-n
floor to the headline metric numbers, from the existing result CSVs (no GPU).

The plan requires every headline metric to carry a bootstrap confidence interval
and paired comparisons to carry a nonparametric significance test before the
number is quoted. This script demonstrates that on the recorded results and
writes a stats summary; the full report-generation pass (Phase 3.4) will apply
the same helpers to every table.
"""

import sys

import pandas as pd

from xai_pilot.config import RESULTS_DIR
from xai_pilot.stats import MIN_N_FLOOR, bootstrap_ci, meets_min_n, wilcoxon_signed_rank


def _fmt_ci(d) -> str:
    return f"{d['point']:.1%} [{d['lo']:.1%}, {d['hi']:.1%}] (n={d['n']})"


def main() -> int:
    rows = []

    # --- Descriptive accuracy (frozen area): overall + per-class CI + min-n ---
    da = pd.read_csv(RESULTS_DIR / "descriptive_accuracy.csv")
    overall = bootstrap_ci(da["answer_changed_top1"].astype(float).tolist())
    print("Descriptive accuracy (top-1, area ranking):")
    print(f"  overall: {_fmt_ci(overall)}")
    rows.append({"metric": "descriptive_accuracy_top1", "group": "overall", **overall})
    print("  by class (min-n floor = %d):" % MIN_N_FLOOR)
    for cls, g in da.groupby("primary_class"):
        ci = bootstrap_ci(g["answer_changed_top1"].astype(float).tolist())
        flag = "" if meets_min_n(ci["n"]) else "  << below min-n, NOT a stable estimate"
        print(f"    {cls:15s}: {_fmt_ci(ci)}{flag}")
        rows.append({"metric": "descriptive_accuracy_top1", "group": cls, **ci,
                     "meets_min_n": meets_min_n(ci["n"])})

    # --- Multi-hazard completeness: CI on the complete rate ---
    mh = pd.read_csv(RESULTS_DIR / "multihazard_completeness.csv")
    mh_ci = bootstrap_ci((mh["verdict"] == "complete").astype(float).tolist())
    print(f"\nMulti-hazard completeness (complete): {_fmt_ci(mh_ci)}"
          f"{'' if meets_min_n(mh_ci['n']) else '  << below min-n'}")
    rows.append({"metric": "multihazard_complete", "group": "overall", **mh_ci})

    # --- Wilcoxon: targeted vs random-location occlusion drift (paired per image) ---
    sweep = pd.read_csv(RESULTS_DIR / "robustness_sweep.csv")
    tgt = sweep[sweep["perturbation"] == "occlude_targeted"].set_index("image_id")["object_centroid_drift"]
    rnd = (sweep[(sweep["perturbation"] == "occlude") & (sweep["location"] == "random")
                 & (sweep["severity"] == 0.35)].set_index("image_id")["object_centroid_drift"])
    paired = pd.concat([tgt.rename("targeted"), rnd.rename("random")], axis=1).dropna()
    w = wilcoxon_signed_rank(paired["targeted"].tolist(), paired["random"].tolist())
    print(f"\nWilcoxon -- targeted vs random-location occlusion drift (paired, n={w['n']}):")
    print(f"  targeted mean drift {paired['targeted'].mean():.3f} vs random {paired['random'].mean():.3f}"
          f"  |  p = {w['p_value']:.2e}")
    rows.append({"metric": "wilcoxon_targeted_vs_random_drift", "group": "paired",
                 "point": paired["targeted"].mean() - paired["random"].mean(),
                 "p_value": w["p_value"], "n": w["n"]})

    # --- Wilcoxon: stability answer-agreement at temp 0.3 vs 1.0 (paired per image) ---
    st = pd.read_csv(RESULTS_DIR / "stability_sweep.csv")
    lo = st[st["temperature"] == 0.3].set_index("image_id")["answer_agreement_rate"]
    hi = st[st["temperature"] == 1.0].set_index("image_id")["answer_agreement_rate"]
    pair2 = pd.concat([lo.rename("t03"), hi.rename("t10")], axis=1).dropna()
    w2 = wilcoxon_signed_rank(pair2["t03"].tolist(), pair2["t10"].tolist())
    print(f"\nWilcoxon -- stability agreement temp 0.3 vs 1.0 (paired, n={w2['n']}):")
    print(f"  agreement {pair2['t03'].mean():.3f} vs {pair2['t10'].mean():.3f}  |  p = {w2['p_value']:.2e}")
    rows.append({"metric": "wilcoxon_stability_temp", "group": "paired",
                 "point": pair2["t03"].mean() - pair2["t10"].mean(),
                 "p_value": w2["p_value"], "n": w2["n"]})

    out_csv = RESULTS_DIR / "statistics_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote {len(rows)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
