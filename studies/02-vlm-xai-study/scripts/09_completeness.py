"""Stage 9 -- Metric 6: Bounded Completeness. No new model calls.

Joins Stage 3 (was an explanation offered?) with evidence that it mattered:

    --basis ablation   Stage 4 token-ablation flip. The original design and the
                       default, so the frozen completeness_{model}.csv reproduces.
                       NOT REPORTABLE (F7): its input has a 51.7% artefact floor.
    --basis attention  Stage 5 attention inside the named region beyond its area
                       (the F7 redesign). Writes completeness_attention_{model}.csv.
                       An attentional claim, not a causal one.

    python scripts/09_completeness.py --model qwen7b-4bit --basis attention
"""
import argparse
import csv
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402

from xai_vlm.metrics.completeness import (  # noqa: E402
    CATEGORIES,
    classify_completeness,
    classify_completeness_attention,
    completeness_rates,
)
from xai_vlm.metrics.reporting import rate_row  # noqa: E402

RESULT_FIELDS = ["image_id", "rule_id", "verdict", "source", "flip_tokens",
                 "flip_black", "completeness"]
ATTENTION_FIELDS = ["image_id", "rule_id", "verdict", "source", "attention_available",
                    "mass_inside_region", "region_area_fraction", "attention_lift",
                    "completeness"]


def output_name(model: str, basis: str) -> str:
    """The ablation basis keeps the frozen name; the attention basis never touches it."""
    return f"completeness_{model}.csv" if basis == "ablation"         else f"completeness_attention_{model}.csv"


def build_attention_rows(regions: list[dict], sparsity: list[dict]) -> list[dict]:
    """One row per region row, joined to Stage 5 by (image_id, rule_id).

    attention_lift = mass / area: 1.0 is chance. Recorded so a reader can see how
    far above the null a `supported` row sits, not just that it crossed it."""
    sp_by_key = {(r["image_id"], r["rule_id"]): r for r in sparsity}
    out = []
    for reg in regions:
        sp = sp_by_key.get((reg["image_id"], reg["rule_id"]), {})
        verdict = classify_completeness_attention(reg.get("source", ""), sp)
        lift = ""
        if verdict in ("explanation_supported", "explanation_weak"):
            area = float(sp["region_area_fraction"])
            lift = f"{float(sp['mass_inside_region']) / area:.4f}" if area > 0 else ""
        out.append({
            "image_id": reg["image_id"], "rule_id": reg["rule_id"],
            "verdict": reg.get("verdict", ""), "source": reg.get("source", ""),
            "attention_available": sp.get("attention_available", ""),
            "mass_inside_region": sp.get("mass_inside_region", ""),
            "region_area_fraction": sp.get("region_area_fraction", ""),
            "attention_lift": lift, "completeness": verdict,
        })
    return out


def build_rows(regions: list[dict], da: list[dict]) -> list[dict]:
    """One completeness row per region row."""
    da_by_key = {(r["image_id"], r["rule_id"]): r for r in da}
    out = []
    for reg in regions:
        key = (reg["image_id"], reg["rule_id"])
        d = da_by_key.get(key, {})
        out.append({
            "image_id": reg["image_id"],
            "rule_id": reg["rule_id"],
            "verdict": reg.get("verdict", ""),
            "source": reg.get("source", ""),
            "flip_tokens": d.get("flip_tokens", ""),
            "flip_black": d.get("flip_black", ""),
            "completeness": classify_completeness(reg.get("source", ""),
                                                  d.get("flip_tokens", "")),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen7b-4bit")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--basis", choices=("ablation", "attention"), default="ablation")
    args = ap.parse_args()

    res = config.RESULTS_DIR
    reg_path = res / f"regions_{args.model}.csv"
    ev_path = res / (f"descriptive_accuracy_{args.model}.csv" if args.basis == "ablation"
                     else f"sparsity_{args.model}.csv")
    for path, stage in ((reg_path, "03_regions"),
                        (ev_path, "04_descriptive_accuracy" if args.basis == "ablation"
                         else "05_sparsity")):
        if not path.exists():
            print(f"missing {path}; run scripts/{stage}.py.", file=sys.stderr)
            return 1

    with reg_path.open(newline="", encoding="utf-8") as fh:
        regions = list(csv.DictReader(fh))
    with ev_path.open(newline="", encoding="utf-8") as fh:
        evidence = list(csv.DictReader(fh))

    if args.basis == "ablation":
        rows, fields = build_rows(regions, evidence), RESULT_FIELDS
    else:
        rows, fields = build_attention_rows(regions, evidence), ATTENTION_FIELDS
    out = res / output_name(args.model, args.basis)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    rates = completeness_rates(rows)
    print("=" * 70)
    print(f"BOUNDED COMPLETENESS, {args.basis} basis  (n={rates['n']})")
    for cat in CATEGORIES:
        print(f"  {cat:<24} {rates[cat]:.1%}")

    not_measured = sum(1 for r in rows if r["completeness"] == "not_measured")
    if not_measured:
        print(f"\n  {not_measured} rows are not_measured -- no {args.basis} evidence for")
        print("  them. They are excluded from the headline, never defaulted.")

    measured = [r for r in rows if r["completeness"] != "not_measured"]
    if measured:
        supported = rate_row("explanation_supported",
                             [r["completeness"] == "explanation_supported" for r in measured],
                             args.n_boot)
        weak = rate_row("explanation_weak",
                        [r["completeness"] == "explanation_weak" for r in measured],
                        args.n_boot)
        print(f"\n  among measured rows:")
        print(f"    supported : {supported['formatted']}")
        print(f"    weak      : {weak['formatted']}")
        if args.basis == "ablation":
            print("\n  `explanation_weak` is the result to read hardest: the model named")
            print("  a region, the region did not matter, and a reader shown only that")
            print("  region would have been misled about why the system decided.")
            print("  NOT REPORTABLE (F7): the token-ablation flip it rests on has a")
            print("  51.7% artefact floor against a 5.9% signal.")
        else:
            lifts = sorted(float(r["attention_lift"]) for r in rows if r["attention_lift"])
            if lifts:
                at2 = sum(1 for x in lifts if x >= 2.0) / len(lifts)
                print(f"\n  attention lift (mass / area, 1.0 = chance): median "
                      f"{lifts[len(lifts) // 2]:.2f}, >= 2x in {at2:.1%} of regions")
            print("  ATTENTIONAL, not causal: `supported` means the model looked inside")
            print("  the region it named more than chance, not that the region drove the")
            print("  answer. `weak`: it named a region and looked no more there than")
            print("  anywhere else.")

    print(f"\nwrote {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
