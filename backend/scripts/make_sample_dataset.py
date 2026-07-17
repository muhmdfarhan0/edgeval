"""Builds a small YOLO-format validation dataset zip for testing EdgeEval,
using a subset of the standard COCO128 sample set (already labeled in
YOLO format, so it's a drop-in stand-in for a client's real validation set).

Run from the project root with the venv's python:
    venv\\Scripts\\python.exe scripts\\make_sample_dataset.py
"""
import shutil
import zipfile
from pathlib import Path

from ultralytics.utils.downloads import download

ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = ROOT / "data" / "_coco128_src"
OUT_ZIP = ROOT / "data" / "validation_sample.zip"
NUM_IMAGES = 100

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    zip_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
    print("Downloading coco128 sample dataset...")
    download(zip_url, dir=WORK_DIR, unzip=True, delete=True)

    src_images = sorted((WORK_DIR / "coco128" / "images" / "train2017").glob("*.jpg"))[:NUM_IMAGES]
    src_labels_dir = WORK_DIR / "coco128" / "labels" / "train2017"

    stage_dir = ROOT / "data" / "_validation_stage"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    (stage_dir / "images").mkdir(parents=True)
    (stage_dir / "labels").mkdir(parents=True)

    for img_path in src_images:
        shutil.copy(img_path, stage_dir / "images" / img_path.name)
        label_path = src_labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy(label_path, stage_dir / "labels" / label_path.name)

    yaml_lines = ["names:"] + [f"  {i}: {n}" for i, n in enumerate(COCO_NAMES)]
    (stage_dir / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in stage_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(stage_dir))

    print(f"Validation dataset written to {OUT_ZIP} ({len(src_images)} images)")


if __name__ == "__main__":
    main()
