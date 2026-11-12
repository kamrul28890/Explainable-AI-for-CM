"""Section 5.5 -- merge the owner's review into the first reader's audit answers.

The first reader (Claude Opus 5.5) answered all 100 images blind. The owner
reviewed them on the review page; each saved review is `review/<image_id>`
with `final` (yes / no / unsure per rule) and a `note`. This writes the
adjudicated answer set in the layout `13_audit_score.py` reads, lists every
rule the owner changed, and reports question-level agreement on the reviewed
images.

Images the owner did not open keep the first reader's answers and are marked
so. Box judgements are not part of the review and are carried over unchanged.

    # ArtifactData list, collection review, out_dir=results/audit_review
    python scripts/14_audit_adjudicate.py
    python scripts/13_audit_score.py results/audit_answers_adjudicated
"""
import argparse
import json
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402

RESULTS = config.RESULTS_DIR


def adjudicate(first: dict, reviews: dict) -> dict:
    """{image_id: answer}: the owner's final answer where reviewed, else the first reader's."""
    out = {}
    for iid, a in first.items():
        doc = dict(a)
        rv = reviews.get(iid)
        if rv:
            doc["blind"] = {**(a.get("blind") or {}), **(rv.get("final") or {})}
            doc["note"] = rv.get("note", "")
            doc["source"] = "owner"
        else:
            doc["source"] = "first reader, not reviewed"
        out[iid] = doc
    return out


def changes(first: dict, reviews: dict) -> list[dict]:
    """Every rule where the owner's final answer differs from the first reader's."""
    out = []
    for iid in sorted(reviews):
        blind = (first.get(iid) or {}).get("blind") or {}
        for r, v in sorted((reviews[iid].get("final") or {}).items()):
            if r in blind and blind[r] != v:
                out.append({"image_id": iid, "rule_id": r, "first": blind[r], "owner": v,
                            "note": reviews[iid].get("note", "")})
    return out


def agreement(first: dict, reviews: dict) -> dict:
    """Question-level agreement on the reviewed images."""
    agree = n = 0
    for iid, rv in reviews.items():
        blind = (first.get(iid) or {}).get("blind") or {}
        for r, v in (rv.get("final") or {}).items():
            if r in blind:
                n += 1
                agree += blind[r] == v
    return {"agree": agree, "n": n}


def write_export(answers: dict, out_dir: Path, uid: str) -> None:
    d = Path(out_dir) / "audit" / uid / "images"
    d.mkdir(parents=True, exist_ok=True)
    for iid, doc in answers.items():
        (d / f"{iid}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _load(dir_: Path) -> dict:
    out = {}
    for f in sorted(Path(dir_).glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        doc = doc.get("data", doc) if isinstance(doc.get("data"), dict) else doc
        out[doc.get("image_id") or f.stem] = doc
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first", default=str(RESULTS / "audit_answers/audit/claude-opus-5-5/images"))
    ap.add_argument("--reviews", default=str(RESULTS / "audit_review/review"))
    ap.add_argument("--out", default=str(RESULTS / "audit_answers_adjudicated"))
    ap.add_argument("--uid", default="owner-adjudicated")
    args = ap.parse_args()

    first, reviews = _load(Path(args.first)), _load(Path(args.reviews))
    if not reviews:
        print("no reviews found", file=sys.stderr)
        return 1
    write_export(adjudicate(first, reviews), Path(args.out), args.uid)
    ag = agreement(first, reviews)
    print(f"reviewed images: {len(reviews)} of {len(first)}")
    print(f"question-level agreement on reviewed images: {ag['agree']} of {ag['n']} "
          f"({ag['agree'] / ag['n']:.1%})")
    ch = changes(first, reviews)
    print(f"rules changed by the owner: {len(ch)} on {len({c['image_id'] for c in ch})} images")
    for c in ch:
        print(f"  {c['image_id']} {c['rule_id']}: {c['first']} -> {c['owner']}"
              + (f"  | {c['note']}" if c["note"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
