"""Runs a loaded model against every image in a dataset, collecting
predictions and per-image inference latency."""
from __future__ import annotations

from typing import List, Tuple

import cv2

from app.services.dataset_loader import Dataset, GroundTruthBox, denormalize_boxes
from app.services.metrics import ModelMetrics, compute_metrics
from app.services.model_loader import Detection, LoadedModel


def run_model_over_dataset(
    model: LoadedModel, dataset: Dataset, class_names: dict
) -> ModelMetrics:
    all_predictions: List[List[Detection]] = []
    all_ground_truths: List[List[GroundTruthBox]] = []
    inference_times_ms: List[float] = []

    for image in dataset.images:
        img_bgr = cv2.imread(str(image.image_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        preds, elapsed_ms = model.predict(img_bgr)
        all_predictions.append(preds)
        inference_times_ms.append(elapsed_ms)

        all_ground_truths.append(denormalize_boxes(image.boxes, w, h))

    return compute_metrics(
        all_predictions=all_predictions,
        all_ground_truths=all_ground_truths,
        class_names=class_names,
        inference_times_ms=inference_times_ms,
    )
