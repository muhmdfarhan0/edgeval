"""Loads a validation dataset from a YOLO-format zip.

Expected zip layout (flexible about nesting depth):
    images/xxx.jpg
    labels/xxx.txt
    data.yaml        # optional, only used for class names

Each label line: `class_id cx cy w h` (normalized 0-1), the standard
YOLO format already produced by client RKNN/export workflows.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class GroundTruthBox:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class DatasetImage:
    image_path: Path
    boxes: List[GroundTruthBox]


@dataclass
class Dataset:
    images: List[DatasetImage]
    class_names: dict  # class_id -> name, best-effort from data.yaml


def _parse_yaml_names(yaml_path: Path) -> dict:
    """Minimal YOLO data.yaml `names:` parser, avoids a yaml dependency."""
    names: dict = {}
    try:
        text = yaml_path.read_text(encoding="utf-8")
    except OSError:
        return names

    lines = text.splitlines()
    in_names_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("names:"):
            rest = stripped[len("names:"):].strip()
            if rest.startswith("[") and rest.endswith("]"):
                items = [i.strip().strip("'\"") for i in rest[1:-1].split(",")]
                names = {i: n for i, n in enumerate(items) if n}
                return names
            in_names_block = True
            continue
        if in_names_block:
            if not stripped or stripped.startswith("#"):
                continue
            if ":" in stripped and (stripped[0].isdigit()):
                idx_str, _, name = stripped.partition(":")
                try:
                    names[int(idx_str.strip())] = name.strip().strip("'\"")
                except ValueError:
                    in_names_block = False
            elif stripped.startswith("-"):
                names[len(names)] = stripped[1:].strip().strip("'\"")
            else:
                in_names_block = False
    return names


def extract_dataset_zip(zip_path: Path, dest_dir: Path) -> Dataset:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    image_files = [p for p in dest_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    if not image_files:
        raise ValueError("No images found in validation dataset zip.")

    yaml_candidates = list(dest_dir.rglob("data.yaml")) + list(dest_dir.rglob("*.yaml"))
    class_names = _parse_yaml_names(yaml_candidates[0]) if yaml_candidates else {}

    images: List[DatasetImage] = []
    for img_path in sorted(image_files):
        label_path = _find_label_for_image(img_path)
        boxes = _parse_label_file(label_path) if label_path else []
        images.append(DatasetImage(image_path=img_path, boxes=boxes))

    return Dataset(images=images, class_names=class_names)


def _find_label_for_image(img_path: Path) -> Optional[Path]:
    # Standard YOLO layout: .../images/foo.jpg -> .../labels/foo.txt
    candidate = Path(str(img_path.parent).replace("images", "labels", 1)) / f"{img_path.stem}.txt"
    if candidate.exists():
        return candidate
    # Fallback: label sitting right next to the image.
    sibling = img_path.with_suffix(".txt")
    if sibling.exists():
        return sibling
    return None


def _parse_label_file(label_path: Path) -> List[GroundTruthBox]:
    """Reads normalized YOLO boxes; caller denormalizes using image size at match time."""
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, cx, cy, w, h = int(float(parts[0])), *map(float, parts[1:5])
        boxes.append(GroundTruthBox(cls=cls, x1=cx - w / 2, y1=cy - h / 2, x2=cx + w / 2, y2=cy + h / 2))
    return boxes


def denormalize_boxes(boxes: List[GroundTruthBox], img_w: int, img_h: int) -> List[GroundTruthBox]:
    return [
        GroundTruthBox(cls=b.cls, x1=b.x1 * img_w, y1=b.y1 * img_h, x2=b.x2 * img_w, y2=b.y2 * img_h)
        for b in boxes
    ]
