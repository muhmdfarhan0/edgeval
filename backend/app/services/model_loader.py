"""Unified loader for baseline/candidate detection models.

Supports Ultralytics `.pt` checkpoints and `.onnx` exports (including
int8-quantized ONNX). Both are wrapped behind the same `.predict(image)`
interface so the inference runner doesn't need to know which backend
produced a given model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int


class LoadedModel:
    """Base interface: predict(image_bgr) -> (detections, inference_ms)."""

    names: dict  # class_id -> class_name

    def predict(self, image_bgr: np.ndarray) -> tuple[List[Detection], float]:
        raise NotImplementedError


class UltralyticsModel(LoadedModel):
    def __init__(self, weights_path: Path, conf_threshold: float):
        from ultralytics import YOLO

        self._model = YOLO(str(weights_path))
        self.names = self._model.names
        self.conf_threshold = conf_threshold

    def predict(self, image_bgr: np.ndarray) -> tuple[List[Detection], float]:
        start = time.perf_counter()
        results = self._model.predict(
            image_bgr, conf=self.conf_threshold, verbose=False
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        dets: List[Detection] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    conf=float(box.conf[0]),
                    cls=int(box.cls[0]),
                )
            )
        return dets, elapsed_ms


class OnnxModel(LoadedModel):
    """Runs a YOLO-exported ONNX graph (fp32 or int8) via onnxruntime.

    Assumes an Ultralytics-style export: single input NCHW, output shape
    (1, 4 + num_classes, num_anchors) in xywh + per-class score format.
    """

    def __init__(self, weights_path: Path, conf_threshold: float, class_names: dict, imgsz: int = 640):
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(weights_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.names = class_names

    def _preprocess(self, image_bgr: np.ndarray):
        h0, w0 = image_bgr.shape[:2]
        scale = self.imgsz / max(h0, w0)
        nh, nw = int(round(h0 * scale)), int(round(w0 * scale))

        import cv2

        resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized

        img = canvas[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = img[None]  # add batch dim
        return img, scale

    def predict(self, image_bgr: np.ndarray) -> tuple[List[Detection], float]:
        img, scale = self._preprocess(image_bgr)

        start = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: img})
        elapsed_ms = (time.perf_counter() - start) * 1000

        pred = outputs[0]
        if pred.shape[1] < pred.shape[2]:
            pred = pred[0].T  # (num_anchors, 4+num_classes)
        else:
            pred = pred[0]

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]
        cls_ids = np.argmax(class_scores, axis=1)
        confs = class_scores[np.arange(len(class_scores)), cls_ids]

        keep = confs >= self.conf_threshold
        boxes_xywh, cls_ids, confs = boxes_xywh[keep], cls_ids[keep], confs[keep]

        dets: List[Detection] = []
        if len(boxes_xywh):
            cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
            x1 = (cx - w / 2) / scale
            y1 = (cy - h / 2) / scale
            x2 = (cx + w / 2) / scale
            y2 = (cy + h / 2) / scale

            keep_idx = _nms(x1, y1, x2, y2, confs, cls_ids, iou_threshold=0.45)
            for i in keep_idx:
                dets.append(
                    Detection(
                        x1=float(x1[i]),
                        y1=float(y1[i]),
                        x2=float(x2[i]),
                        y2=float(y2[i]),
                        conf=float(confs[i]),
                        cls=int(cls_ids[i]),
                    )
                )
        return dets, elapsed_ms


def _nms(x1, y1, x2, y2, scores, cls_ids, iou_threshold: float) -> List[int]:
    """Class-aware non-max suppression."""
    order = scores.argsort()[::-1]
    areas = (x2 - x1) * (y2 - y1)
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)

        same_class = cls_ids[rest] == cls_ids[i]
        suppress = (iou > iou_threshold) & same_class
        order = rest[~suppress]

    return keep


def load_model(weights_path: Path, conf_threshold: float, fallback_names: dict | None = None) -> LoadedModel:
    suffix = weights_path.suffix.lower()
    if suffix == ".pt":
        return UltralyticsModel(weights_path, conf_threshold)
    elif suffix == ".onnx":
        if not fallback_names:
            raise ValueError("ONNX models need class names supplied (e.g. from the baseline model or dataset yaml).")
        return OnnxModel(weights_path, conf_threshold, fallback_names)
    else:
        raise ValueError(f"Unsupported model format: {suffix}. Use .pt or .onnx.")
