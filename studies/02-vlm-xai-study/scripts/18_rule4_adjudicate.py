"""Rule_4 audit -- merge the owner's review into the first reader's answers.

The first reader (Claude Opus 5.5) answered every drawn image blind. The owner
reviewed the ones where that answer disagreed with the annotation or was
"unsure"; each saved review is `review/<image_id>` with `final` and a `note`.
Images the owner did not open agree with the annotation and keep the first
reader's answer. The output is read by 16_rule4_audit_score.py.

    # ArtifactData list, collection review, out_dir=results/rule4_audit_review
    python scripts/18_rule4_adjudicate.py
    python scripts/16_rule4_audit_score.py results/rule4_audit_answers/owner-adjudicated
"""
import argparse
import json
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402


def adjudicate(first: dict, reviews: dict) -> dict:
    """{image_id: answer}: the owner's final answer where reviewed, else the first reader's."""
    stray = sorted(set(reviews) - set(first))
    if stray:
        raise ValueError(f"reviews for images not in the draw: {stray}")
    out = {}
    for iid, a in first.items():
        doc = dict(a)
        rv = reviews.get(iid)
        if rv:
            doc["first_reader"] = a.get("rule_4")
            doc["rule_4"] = (rv.get("final") or {}).get("rule_4", a.get("rule_4"))
            doc["note"] = rv.get("note", "")
            doc["source"] = "owner"
        else:
            doc["source"] = "first reader, not reviewed"
        doc["auditor"] = "owner-adjudicated"
        out[iid] = doc
    return out


def changes(first: dict, reviews: dict) -> list[dict]:
    """Every image where the owner's final answer differs from the first reader's."""
    out = []
    for iid in sorted(reviews):
        v = (reviews[iid].get("final") or {}).get("rule_4")
        f = first[iid].get("rule_4")
        if v and v != f:
            out.append({"image_id": iid, "first": f, "owner": v, "note": reviews[iid].get("note", "")})
    return out


def _load(dir_: Path) -> dict:
    out = {}
    for f in sorted(Path(dir_).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        d = d["data"] if isinstance(d.get("data"), dict) else d
        out[d.get("image_id") or f.stem] = d
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ans = config.RESULTS_DIR / "rule4_audit_answers"
    ap.add_argument("--first", default=str(ans / "claude-opus-5-5"))
    ap.add_argument("--reviews", default=str(config.RESULTS_DIR / "rule4_audit_review" / "review"))
    ap.add_argument("--out", default=str(ans / "owner-adjudicated"))
    a = ap.parse_args(argv)
    first, reviews = _load(Path(a.first)), _load(Path(a.reviews))
    merged = adjudicate(first, reviews)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for iid, doc in merged.items():
        (out / f"{iid}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    ch = changes(first, reviews)
    print(f"drawn {len(first)}; owner reviewed {len(reviews)}; changed {len(ch)}")
    for c in ch:
        print(f"  {c['image_id']}  {c['first']:>6} -> {c['owner']:<6}  {c['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
