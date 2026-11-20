"""Diagnostic (2026-11-12): are Qwen2.5-VL regions in the original-image frame?

Qwen2.5-VL gives box coordinates in the frame of the image the processor fed
it (smart_resize to <= 640 blocks: 1200x900 -> 812x588). The harness reads
them as original-image pixels. This scores the frozen correct-YES rows'
IoU against the annotation both ways.

    python scripts/diag_region_frame.py qwen7b-4bit
"""
import sys, csv, math, ast, importlib.util
from pathlib import Path
S=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(S/"src"))
from xai_vlm import config
from xai_vlm.metrics.geometry import max_iou, to_normalised
spec=importlib.util.spec_from_file_location("s10",S/"scripts/10_explanation_correctness.py"); s10=importlib.util.module_from_spec(spec); spec.loader.exec_module(s10)
from xai_pilot.data import load_construction_site
def sr(w,h,maxp=640*784,minp=256*784,f=28):
    hb=max(f,round(h/f)*f); wb=max(f,round(w/f)*f)
    if hb*wb>maxp:
        beta=math.sqrt(h*w/maxp); hb=math.floor(h/beta/f)*f; wb=math.floor(w/beta/f)*f
    elif hb*wb<minp:
        beta=math.sqrt(minp/(h*w)); hb=math.ceil(h*beta/f)*f; wb=math.ceil(w*beta/f)*f
    return wb,hb
tier=sys.argv[1]
base=list(csv.DictReader(open(config.RESULTS_DIR/f"scholar-a40/baseline_{tier}.csv",encoding="utf-8")))
reg={(r["image_id"],r["rule_id"]):r for r in csv.DictReader(open(config.RESULTS_DIR/f"scholar-a40/regions_{tier}.csv",encoding="utf-8"))}
rows=[r for r in base if r["expected"]=="YES" and r["verdict"]=="YES" and r["region"].strip()]
ann=s10.load_annotations(load_construction_site,{r["image_id"] for r in rows})
a=[];b=[];hit_a=hit_b=0
for r in rows:
    g=reg[(r["image_id"],r["rule_id"])]; W=int(g["image_w"]); H=int(g["image_h"])
    hb=(ann.get(r["image_id"],{}).get(r["rule_id"]) or {}).get("bounding_box") or []
    if not hb: continue
    box=tuple(float(x) for x in ast.literal_eval(r["region"]))
    rw,rh=sr(W,H)
    i0=max_iou(to_normalised(box,W,H),[tuple(x) for x in hb])
    scaled=(box[0]*W/rw,box[1]*H/rh,min(W,box[2]*W/rw),min(H,box[3]*H/rh))
    i1=max_iou(to_normalised(scaled,W,H),[tuple(x) for x in hb])
    a.append(i0); b.append(i1)
import statistics as st
n=len(a); print(tier,"n",n)
print(" as harness (original frame): mean IoU %.3f, IoU>=0.5 %.1f%%, IoU>0 %.1f%%"%(st.mean(a),100*sum(x>=.5 for x in a)/n,100*sum(x>0 for x in a)/n))
print(" rescaled from resized frame: mean IoU %.3f, IoU>=0.5 %.1f%%, IoU>0 %.1f%%"%(st.mean(b),100*sum(x>=.5 for x in b)/n,100*sum(x>0 for x in b)/n))
