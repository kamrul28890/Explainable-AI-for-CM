"""Rebuild a results tree with every model region in the original-image frame.

Qwen2.5-VL states boxes in its resized model-input frame (xai_vlm.frames); the
frozen trees read them as original pixels. This copies a frozen tree to a new
one with the regions mapped, without touching the source:

  baseline_<model>.csv     REGION re-parsed from the stored reply in the input
                           frame, then mapped (exact: the reply is kept)
  stability_<model>.csv    each rerun region mapped
  robustness_<model>.csv   baseline and perturbed regions mapped, drift and
                           drifted recomputed
  sample_manifest.csv, complement files   copied unchanged

Stages that only READ regions are then re-run against the new tree with
XAI_RESULTS_DIR pointing at it: 03 and 10 on CPU; 04 and 05 on the GPU (their
masks and attention mass used the misplaced regions), then 09 and 11.

    python scripts/17_region_frame_fix.py --src results/scholar-a40 --dst results/scholar-a40-rf
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402
from xai_vlm.frames import _fmt, region_from_reply, rescale_box_text  # noqa: E402
from xai_vlm.metrics.stability import DRIFT_FLAG, normalized_centroid_distance, parse_box  # noqa: E402

from xai_vlm.perturbations import PERTURBATIONS  # noqa: E402
csv.field_size_limit(10 ** 8)


def fix_baseline_row(row: dict, size, frame_of) -> tuple[dict, str]:
    """(row with REGION in the original frame, change kind)."""
    out = dict(row)
    if not size:
        return out, "no size"
    insize = frame_of(size)
    raw = row.get("raw_text") or ""
    new = _fmt(region_from_reply(raw, size, insize)) if raw.strip() else \
        rescale_box_text(row.get("region", ""), size, insize)
    old = (row.get("region") or "").strip()
    out["region"] = new
    kind = ("rescaled" if old and new else "kept empty" if not old and not new
            else "lost" if old else "gained")
    return out, kind


def fix_stability_row(row: dict, size, frame_of) -> dict:
    out = dict(row)
    if size and row.get("regions") is not None:
        insize = frame_of(size)
        out["regions"] = "|".join(rescale_box_text(r, size, insize) if r.strip() else ""
                                  for r in row["regions"].split("|"))
    return out


def fix_robustness_row(row: dict, size, frame_of) -> dict:
    out = dict(row)
    if not size:
        return out
    insize = frame_of(size)
    out["baseline_region"] = rescale_box_text(row.get("baseline_region", ""), size, insize)
    base = parse_box(out["baseline_region"])
    for p in PERTURBATIONS:
        if f"region_{p}" not in row:
            continue
        out[f"region_{p}"] = rescale_box_text(row.get(f"region_{p}", ""), size, insize)
        box = parse_box(out[f"region_{p}"])
        out[f"drift_{p}"] = ""
        if base is not None and box is not None:
            out[f"drift_{p}"] = f"{normalized_centroid_distance(base, box, size):.4f}"
        out[f"drifted_{p}"] = ""
        if out.get(f"survived_{p}") == "True" and out[f"drift_{p}"]:
            out[f"drifted_{p}"] = str(float(out[f"drift_{p}"]) > DRIFT_FLAG)
    return out


def _read(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        return r.fieldnames, list(r)


def _write(path: Path, fields, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(config.RESULTS_DIR / "scholar-a40"))
    ap.add_argument("--dst", default=str(config.RESULTS_DIR / "scholar-a40-rf"))
    ap.add_argument("--models", nargs="+", default=["qwen7b-4bit", "qwen7b-bf16"])
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if dst.resolve() == src.resolve():
        print("refusing to write into the source tree", file=sys.stderr)
        return 1
    dst.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor
    from xai_vlm.frames import ModelInputFrame
    proc = AutoProcessor.from_pretrained(str(config.MODEL_DIR / "qwen7b-4bit"),
                                         min_pixels=config.MIN_PIXELS, max_pixels=config.MAX_PIXELS)
    frame = ModelInputFrame(proc)

    for name in ("sample_manifest.csv",):
        shutil.copy2(config.RESULTS_DIR / name, dst / name)
    for f in src.glob("*complement*"):
        shutil.copy2(f, dst / f.name)

    for model in args.models:
        _, reg = _read(src / f"regions_{model}.csv")
        size = {(r["image_id"], r["rule_id"]): (int(r["image_w"]), int(r["image_h"]))
                for r in reg if r.get("image_w")}
        by_image = {k[0]: v for k, v in size.items()}

        fields, rows = _read(src / f"baseline_{model}.csv")
        kinds: dict = {}
        fixed = []
        for r in rows:
            out, kind = fix_baseline_row(r, size.get((r["image_id"], r["rule_id"])), frame.size_for)
            kinds[kind] = kinds.get(kind, 0) + 1
            fixed.append(out)
        _write(dst / f"baseline_{model}.csv", fields, fixed)
        print(f"{model} baseline: {len(rows)} rows, {kinds}")

        for stem, fn in (("stability", fix_stability_row), ("robustness", fix_robustness_row)):
            p = src / f"{stem}_{model}.csv"
            if not p.exists():
                continue
            fields, rows = _read(p)
            _write(dst / p.name, fields, [fn(r, by_image.get(r["image_id"]), frame.size_for)
                                          for r in rows])
            print(f"{model} {stem}: {len(rows)} rows")
    print(f"sizes seen: {sorted(frame._cache.items())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
