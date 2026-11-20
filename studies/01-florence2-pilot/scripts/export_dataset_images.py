"""Export the locally cached ConstructionSite images from Parquet.

The Hugging Face snapshot stores each JPEG as bytes inside the ``image``
column. This utility writes those bytes back to ordinary image files while
preserving the dataset's train/test split and original filenames. It also
writes one metadata CSV per split containing every non-image Parquet field,
plus compact multi-label summaries for convenient browsing.

The export is restart-safe: an existing image with the expected byte length
is left in place. The source Parquet files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


PILOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PILOT_DIR / "data" / "constructionsite"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR / "extracted_images"

RULE_TO_CLASS = {
    "rule_1_violation": "ppe_violation",
    "rule_2_violation": "fall_hazard",
    "rule_3_violation": "fall_hazard",
    "rule_4_violation": "struck_by_risk",
}
CLASS_PRIORITY = ("ppe_violation", "fall_hazard", "struck_by_risk")


def _split_for_parquet(path: Path) -> str:
    """Return the dataset split represented by a local Parquet shard."""
    if path.name.startswith("test"):
        return "test"
    if path.name.startswith("train"):
        return "train"
    raise ValueError(f"Cannot infer split from {path.name!r}")


def _json_cell(value: Any) -> Any:
    """Serialize nested Arrow values compactly for a CSV cell."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _label_summary(row: dict[str, Any]) -> tuple[list[str], list[str], str]:
    """Derive multi-label rule/class summaries without replacing raw fields."""
    rules = [rule for rule in RULE_TO_CLASS if row.get(rule) is not None]
    classes = []
    for rule in rules:
        hazard_class = RULE_TO_CLASS[rule]
        if hazard_class not in classes:
            classes.append(hazard_class)
    if not classes:
        classes = ["compliant"]
    primary = next((name for name in CLASS_PRIORITY if name in classes), "compliant")
    return rules, classes, primary


def _write_readme(output_dir: Path, counts: dict[str, int]) -> None:
    """Document the generated layout and its relationship to the source."""
    total = sum(counts.values())
    readme = f"""# Extracted ConstructionSite images

This folder is a browsable export of the JPEG bytes stored inside the local
ConstructionSite Parquet snapshot.

- `test/`: {counts.get("test", 0):,} original test-split JPEGs
- `train/`: {counts.get("train", 0):,} original train-split JPEGs
- `test_metadata.csv` and `train_metadata.csv`: original non-image fields,
  bounding boxes, reasons, and added label summaries
- Total exported images: {total:,}

The JPEG filenames come from the Parquet `image.path` field. The original
Parquet files in the parent folder were not changed. Images are organized by
the dataset's actual train/test split; multi-label categories remain in the
metadata rather than duplicating an image into several category folders.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def export_dataset(dataset_dir: Path, output_dir: Path, batch_size: int = 64) -> int:
    """Export all local shards and return the number of image rows processed."""
    parquet_paths = sorted(dataset_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No Parquet files found in {dataset_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths_by_split: dict[str, list[Path]] = {"test": [], "train": []}
    for path in parquet_paths:
        paths_by_split[_split_for_parquet(path)].append(path)

    counts: dict[str, int] = {}
    total_written = 0
    for split in ("test", "train"):
        shard_paths = paths_by_split[split]
        if not shard_paths:
            continue

        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / f"{split}_metadata.csv"

        schema_names = pq.ParquetFile(shard_paths[0]).schema_arrow.names
        metadata_fields = [name for name in schema_names if name != "image"]
        fieldnames = [
            "split",
            "relative_image_path",
            *metadata_fields,
            "violated_rules",
            "hazard_classes",
            "primary_class",
        ]

        split_count = 0
        with metadata_path.open("w", encoding="utf-8", newline="") as metadata_file:
            writer = csv.DictWriter(metadata_file, fieldnames=fieldnames)
            writer.writeheader()

            for shard_path in shard_paths:
                parquet = pq.ParquetFile(shard_path)
                for batch in parquet.iter_batches(batch_size=batch_size):
                    for row in batch.to_pylist():
                        image = row.pop("image")
                        image_bytes = image.get("bytes") if image else None
                        if not image_bytes:
                            raise ValueError(
                                f"Missing embedded image bytes in {shard_path.name}, "
                                f"image_id={row.get('image_id')!r}"
                            )

                        embedded_path = Path(image.get("path") or "")
                        filename = embedded_path.name or f"{row['image_id']}.jpg"
                        if not Path(filename).suffix:
                            filename += ".jpg"
                        image_path = split_dir / filename

                        if not image_path.exists() or image_path.stat().st_size != len(image_bytes):
                            image_path.write_bytes(image_bytes)

                        rules, classes, primary = _label_summary(row)
                        csv_row = {
                            "split": split,
                            "relative_image_path": image_path.relative_to(output_dir).as_posix(),
                            **{name: _json_cell(row.get(name)) for name in metadata_fields},
                            "violated_rules": "|".join(rules),
                            "hazard_classes": "|".join(classes),
                            "primary_class": primary,
                        }
                        writer.writerow(csv_row)
                        split_count += 1
                        total_written += 1

                        if total_written % 500 == 0:
                            print(f"Processed {total_written:,} images...", flush=True)

        counts[split] = split_count
        print(
            f"Completed {split}: {split_count:,} images -> {split_dir}",
            flush=True,
        )

    _write_readme(output_dir, counts)
    print(f"Export complete: {total_written:,} images -> {output_dir}", flush=True)
    return total_written


def main() -> int:
    """Parse command-line paths and export the local dataset snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    export_dataset(
        dataset_dir=args.dataset_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
