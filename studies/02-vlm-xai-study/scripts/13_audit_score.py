"""Section 5.5 -- score the human annotation audit.

The dataset's annotations are the reference every Study 2 figure is measured
against. This scores how often a person, answering blind from the photograph,
disagrees with them -- per quality tier, and reweighted to the corpus.

    # 1. export the answers (Claude: ArtifactData list, collection
    #    audit/<uid>/images, out_dir=results/audit_answers)
    # 2. python scripts/13_audit_score.py results/audit_answers

Two errors are kept apart, never pooled:
  marked, not confirmed       the annotation marks a violation; the auditor says No
  unmarked, auditor sees one  the annotation marks none; the auditor says Yes
"Can't tell" counts in neither and is reported on its own line. Only rules that
apply to the image are scored (rule_4 needs machinery in the scene).

The draw is stratified 25 hazard + 25 compliant per tier (decided 2026-11-10),
so the raw image-level rate over-represents hazard images. The corpus rate
reweights each tier's two strata to that tier's real hazard prevalence and the
tiers to their real shares (`results/audit_strata.csv`, written by Stage 12),
with a bootstrap that resamples images within each (tier, stratum) cell.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402
from xai_vlm.metrics.reporting import rate_row  # noqa: E402

RULES = ("rule_1", "rule_2", "rule_3", "rule_4")


def questions(item: dict, answer: dict) -> list[dict]:
    """One row per applicable rule: was a violation annotated, what did the
    auditor say blind (yes / no / unsure)."""
    blind = answer.get("blind") or {}
    return [{"image_id": item["image_id"], "tier": item["quality_of_info"], "rule_id": r,
             "annotated": r in (item.get("annotations") or {}),
             "auditor": blind.get(r, "")}
            for r in RULES if (item.get("applicable") or {}).get(r, True)]


def question_rates(qs: list[dict]) -> dict:
    """k and n for each error type."""
    marked = [q for q in qs if q["annotated"] and q["auditor"] in ("yes", "no")]
    unmarked = [q for q in qs if not q["annotated"] and q["auditor"] in ("yes", "no")]
    answered = [q for q in qs if q["auditor"]]
    return {
        "marked, not confirmed": {"k": sum(q["auditor"] == "no" for q in marked),
                                  "n": len(marked)},
        "unmarked, auditor sees one": {"k": sum(q["auditor"] == "yes" for q in unmarked),
                                       "n": len(unmarked)},
        "can't tell": {"k": sum(q["auditor"] == "unsure" for q in answered),
                       "n": len(answered)},
    }


def image_disagrees(item: dict, answer: dict) -> bool:
    """Any applicable rule where a definite blind answer contradicts the annotation."""
    for q in questions(item, answer):
        if q["auditor"] == "no" and q["annotated"]:
            return True
        if q["auditor"] == "yes" and not q["annotated"]:
            return True
    return False


def reweight(cell_rates: dict, strata: dict) -> float:
    """cell_rates[tier] = {"H": rate, "C": rate}; strata[tier] = {share, prevalence}."""
    return sum(s["share"] * (s["prevalence"] * cell_rates[t]["H"]
                             + (1 - s["prevalence"]) * cell_rates[t]["C"])
               for t, s in strata.items())


def reweighted_ci(cells: dict, strata: dict, n_boot: int = 10_000, seed: int = 0):
    """cells[tier][H|C] = list of bools. Point estimate and a 95% percentile CI
    from resampling images within each cell."""
    def rates(c):
        return {t: {h: (sum(v) / len(v) if v else 0.0) for h, v in c[t].items()} for t in c}
    point = reweight(rates(cells), strata)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        c = {t: {h: [rng.choice(v) for _ in v] if v else [] for h, v in cells[t].items()}
             for t in cells}
        boots.append(reweight(rates(c), strata))
    boots.sort()
    return point, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def box_counts(answers: list[dict]) -> dict:
    out = {"yes": 0, "partly": 0, "no": 0, "n": 0}
    for a in answers:
        for v in (a.get("boxes") or {}).values():
            if v in ("yes", "partly", "no"):
                out[v] += 1
                out["n"] += 1
    return out


def load_answers(export_dir: Path) -> dict:
    """{auditor uid: {image_id: answer}} from an ArtifactData out_dir export."""
    out: dict = {}
    for f in sorted(Path(export_dir).glob("audit/*/images/*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        doc = doc.get("data", doc) if isinstance(doc.get("data"), dict) else doc
        out.setdefault(f.parent.parent.name, {})[doc.get("image_id") or f.stem] = doc
    return out


def load_strata(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    total = sum(int(r["images"]) for r in rows)
    return {r["quality_of_info"]: {"share": int(r["images"]) / total,
                                   "prevalence": int(r["hazard_images"]) / int(r["images"])}
            for r in rows}


def score_auditor(items: list[dict], answers: dict, strata: dict, n_boot: int) -> list[str]:
    done = [it for it in items if it["image_id"] in answers]
    lines = [f"images answered: {len(done)} of {len(items)}"]
    for tier in [None] + sorted({it["quality_of_info"] for it in done}):
        sub = [it for it in done if tier is None or it["quality_of_info"] == tier]
        qs = [q for it in sub for q in questions(it, answers[it["image_id"]])]
        lines.append(f"\n{tier or 'all tiers'} (question level, as drawn -- not reweighted)")
        for label, kn in question_rates(qs).items():
            r = rate_row(label, [True] * kn["k"] + [False] * (kn["n"] - kn["k"]), n_boot)
            note = "  <-- " + r["note"] if r["below_floor"] else ""
            lines.append(f"  {label:<30} {r['formatted']}{note}")
    cells = {t: {"H": [], "C": []} for t in strata}
    for it in done:
        if it["quality_of_info"] in cells:
            cells[it["quality_of_info"]]["H" if it.get("annotations") else "C"].append(
                image_disagrees(it, answers[it["image_id"]]))
    if all(v for c in cells.values() for v in c.values()):
        p, lo, hi = reweighted_ci(cells, strata, n_boot=n_boot)
        lines.append(f"\nimage disagrees with its annotation, reweighted to the corpus: "
                     f"{p:.1%} [{lo:.1%}, {hi:.1%}]")
    else:
        lines.append("\nreweighted corpus rate: needs answers in every (tier, stratum) cell")
    b = box_counts([answers[it["image_id"]] for it in done])
    if b["n"]:
        lines.append(f"annotated boxes on their hazard: yes {b['yes']}, partly {b['partly']}, "
                     f"no {b['no']} (n={b['n']})")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("export_dir", help="ArtifactData out_dir holding audit/<uid>/images/*.json")
    ap.add_argument("--items", default=str(STUDY_ROOT / "audit" / "items.json"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()

    items = json.loads(Path(args.items).read_text(encoding="utf-8"))
    strata = load_strata(config.RESULTS_DIR / "audit_strata.csv")
    by_auditor = load_answers(Path(args.export_dir))
    if not by_auditor:
        print("no answers found", file=sys.stderr)
        return 1
    for uid, answers in by_auditor.items():
        print(f"== auditor {uid}")
        print("\n".join(score_auditor(items, answers, strata, args.n_boot)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
