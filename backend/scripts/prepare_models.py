"""Prepares the baseline + candidate model pair used for Day-1 testing.

Baseline: pretrained FP32 YOLOv8n PyTorch checkpoint.
Candidate: the same model exported to ONNX with int8 quantization,
simulating an optimized edge-deployment target.

Run from the project root with the venv's python:
    venv\\Scripts\\python.exe scripts\\prepare_models.py
"""
from pathlib import Path

from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading baseline FP32 model (yolov8n.pt)...")
    baseline = YOLO("yolov8n.pt")

    baseline_dest = MODELS_DIR / "baseline_yolov8n.pt"
    src = Path(baseline.ckpt_path) if getattr(baseline, "ckpt_path", None) else Path("yolov8n.pt")
    if src.exists() and not baseline_dest.exists():
        baseline_dest.write_bytes(src.read_bytes())
    print(f"Baseline saved to {baseline_dest}")

    print("Exporting candidate ONNX int8 model...")
    exported_path = baseline.export(format="onnx", int8=True)
    exported_path = Path(exported_path)

    candidate_dest = MODELS_DIR / "candidate_yolov8n_int8.onnx"
    if exported_path.exists():
        candidate_dest.write_bytes(exported_path.read_bytes())
    print(f"Candidate saved to {candidate_dest}")


if __name__ == "__main__":
    main()
