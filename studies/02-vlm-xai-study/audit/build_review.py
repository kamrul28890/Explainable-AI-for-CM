"""Fill review.tpl.html with the audit items and the frontier-model auditor's answers.

    python audit/build_review.py OUT_DIR
writes OUT_DIR/review.html and OUT_DIR/img/<id>.jpg (900 px long side) from the
Stage 12 export at --img (default: the export used for the labelling page).
"""
import argparse, json
from pathlib import Path
from PIL import Image

d = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--img", required=True, help="Stage 12 export image folder")
ap.add_argument("--auditor", default="claude-opus-5-5")
a = ap.parse_args()
out = Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True)
items = json.loads((d / "items.json").read_text(encoding="utf-8"))
ans_dir = d.parent / "results" / "audit_answers" / "audit" / a.auditor / "images"
rows = []
for x in items:
    s = json.loads((ans_dir / f"{x['image_id']}.json").read_text(encoding="utf-8"))
    app = [r for r, v in x["applicable"].items() if v]
    dis = [r for r in app if (r in x["annotations"] and s["blind"][r] == "no")
           or (r not in x["annotations"] and s["blind"][r] == "yes")]
    rows.append({"id": x["image_id"], "tier": x["quality_of_info"], "light": x["illumination"], "app": app,
                 "ann": x["annotations"], "c": s["blind"], "cn": s.get("blind_note", ""),
                 "cb": s.get("boxes", {}), "cbn": s.get("note", ""), "dis": dis,
                 "uns": [r for r in app if s["blind"][r] == "unsure"]})
    im = Image.open(Path(a.img) / f"{x['image_id']}.jpg").convert("RGB")
    im.thumbnail((900, 900)); im.save(out / "img" / f"{x['image_id']}.jpg", quality=80)
tpl = (d / "review.tpl.html").read_text(encoding="utf-8")
(out / "review.html").write_text(tpl.replace("__ITEMS__", json.dumps(rows, separators=(",", ":"))), encoding="utf-8")
print(len(rows), "items,", sum(bool(r["dis"]) for r in rows), "with a disagreement")
