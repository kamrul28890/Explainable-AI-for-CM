"""Saving visual review artifacts: boxes drawn on top of pilot images."""

from pathlib import Path

from PIL import Image, ImageDraw


def overlay_boxes(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    labels: list[str] | None = None,
    color: str = "red",
) -> Image.Image:
    """Draw absolute-pixel boxes (x0, y0, x1, y1) on a copy of image."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    labels = labels or [""] * len(boxes)
    for box, label in zip(boxes, labels):
        draw.rectangle(box, outline=color, width=3)
        if label:
            draw.text((box[0] + 2, max(box[1] - 12, 0)), label, fill=color)
    return out


def save_figure(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
