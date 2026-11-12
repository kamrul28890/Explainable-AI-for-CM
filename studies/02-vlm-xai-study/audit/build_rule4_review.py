"""Fill review.tpl.html for the rule_4 audit: rule_4 only, Claude's blind answer
beside the annotation. Neither model's answer is shown.

    python audit/build_rule4_review.py OUT_DIR --img <15_rule4_audit_sample export>/img
writes OUT_DIR/review.html and OUT_DIR/img/<id>.jpg (900 px long side).
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image

d = Path(__file__).resolve().parent
STUDY = d.parent
sys.path.insert(0, str(STUDY / "src"))
from xai_vlm import config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--img", required=True)
ap.add_argument("--auditor", default="claude-opus-5-5")
a = ap.parse_args()
out = Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True)

with (config.RESULTS_DIR / "rule4_audit_manifest.csv").open(newline="", encoding="utf-8") as fh:
    manifest = list(csv.DictReader(fh))
want = {m["image_id"] for m in manifest}

spec = importlib.util.spec_from_file_location("s1", STUDY / "scripts" / "01_build_sample.py")
s1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s1)
from xai_pilot.data import load_construction_site  # noqa: E402
recs = {r["image_id"]: r for r in s1.scan_records(load_construction_site) if r["image_id"] in want}

ans_dir = config.RESULTS_DIR / "rule4_audit_answers" / a.auditor
rows = []
for m in manifest:
    iid = m["image_id"]
    s = json.loads((ans_dir / f"{iid}.json").read_text(encoding="utf-8"))
    v = (recs.get(iid) or {}).get("rule_4_violation")
    ann = {"rule_4": {"reason": v.get("reason", "") or "",
                      "boxes": [list(map(float, b)) for b in v.get("bounding_box") or []]}} if v else {}
    c = s["rule_4"]
    dis = ["rule_4"] if (ann and c == "no") or (not ann and c == "yes") else []
    rec = recs.get(iid) or {}
    rows.append({"id": iid, "tier": rec.get("quality_of_info", ""), "light": rec.get("illumination", ""),
                 "app": ["rule_4"], "ann": ann, "c": {"rule_4": c}, "cn": s.get("note", ""),
                 "cb": {}, "cbn": "", "dis": dis, "uns": ["rule_4"] if c == "unsure" else []})
    im = Image.open(Path(a.img) / f"{iid}.jpg").convert("RGB")
    im.thumbnail((900, 900)); im.save(out / "img" / f"{iid}.jpg", quality=80)

tpl = (d / "review.tpl.html").read_text(encoding="utf-8")
for old, new in (("<title>Annotation Audit Review</title>", "<title>Excavator Audit Review</title>"),
                 ("<small>§5.5 audit · second reader</small>Annotation Audit Review",
                  "<small>rule_4 audit · second reader</small>Excavator Audit Review"),
                 ("0 / 100 reviewed", f"0 / {len(rows)} reviewed"),
                 ("All 100", f"All {len(rows)}"),
                 ("answered each rule from the photograph alone",
                  "answered the excavator rule from the photograph alone, without either model's answer,")):
    assert old in tpl, old
    tpl = tpl.replace(old, new)
(out / "review.html").write_text(tpl.replace("__ITEMS__", json.dumps(rows, separators=(",", ":"))),
                                 encoding="utf-8")
print(len(rows), "items,", sum(bool(r["dis"]) for r in rows), "disagreements,",
      sum(bool(r["uns"]) for r in rows), "can't tell")
