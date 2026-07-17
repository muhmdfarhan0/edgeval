"""Core metrics engine: IoU matching, per-class precision/recall, AP, mAP@0.5.

This is the differentiator of EdgeEval: metrics are computed and stored
PER CLASS, not just as a single aggregate number, so a regression on one
class is visible even when overall mAP barely moves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from app.config import IOU_MATCH_THRESHOLD
from app.services.dataset_loader import GroundTruthBox
from app.services.model_loader import Detection


@dataclass
class PerClassMetrics:
    precision: float
    recall: float
    ap: float
    tp: int
    fp: int
    fn: int
    num_ground_truth: int


@dataclass
class ModelMetrics:
    per_class: Dict[str, PerClassMetrics] = field(default_factory=dict)
    overall_map: float = 0.0
    inference_times_ms: List[float] = field(default_factory=list)


def _iou(a: GroundTruthBox | Detection, b: GroundTruthBox | Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Pascal VOC 2010+ style all-point interpolated AP (area under the
    monotonically-decreasing envelope of the precision/recall curve)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def compute_metrics(
    all_predictions: List[List[Detection]],
    all_ground_truths: List[List[GroundTruthBox]],
    class_names: Dict[int, str],
    inference_times_ms: List[float],
    iou_threshold: float = IOU_MATCH_THRESHOLD,
) -> ModelMetrics:
    """
    all_predictions[i] / all_ground_truths[i] are the detections / GT boxes
    for image i, already in absolute pixel coordinates.
    """
    class_ids = sorted(class_names.keys())
    per_class: Dict[str, PerClassMetrics] = {}
    aps = []

    for cls_id in class_ids:
        cls_name = class_names[cls_id]

        # Flatten all predictions of this class across images, sorted by confidence desc.
        scored_preds = []  # (image_idx, Detection)
        num_gt = 0
        gt_matched = {}  # image_idx -> [bool] parallel to that image's GT boxes of this class
        gt_by_image = {}  # image_idx -> [GroundTruthBox] of this class

        for img_idx, gts in enumerate(all_ground_truths):
            cls_gts = [g for g in gts if g.cls == cls_id]
            gt_by_image[img_idx] = cls_gts
            gt_matched[img_idx] = [False] * len(cls_gts)
            num_gt += len(cls_gts)

        for img_idx, preds in enumerate(all_predictions):
            for det in preds:
                if det.cls == cls_id:
                    scored_preds.append((img_idx, det))

        scored_preds.sort(key=lambda p: p[1].conf, reverse=True)

        tp = np.zeros(len(scored_preds))
        fp = np.zeros(len(scored_preds))

        for i, (img_idx, det) in enumerate(scored_preds):
            cls_gts = gt_by_image[img_idx]
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(cls_gts):
                if gt_matched[img_idx][j]:
                    continue
                iou = _iou(det, gt)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= iou_threshold and best_j >= 0:
                tp[i] = 1
                gt_matched[img_idx][best_j] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall_curve = tp_cum / num_gt if num_gt > 0 else np.zeros_like(tp_cum)
        precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        ap = _average_precision(recall_curve, precision_curve) if num_gt > 0 or len(scored_preds) else 0.0

        total_tp = int(tp.sum())
        total_fp = int(fp.sum())
        total_fn = num_gt - total_tp

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / num_gt if num_gt > 0 else 0.0

        per_class[cls_name] = PerClassMetrics(
            precision=precision,
            recall=recall,
            ap=ap,
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            num_ground_truth=num_gt,
        )

        # Only classes actually present in the validation set count toward mAP.
        if num_gt > 0:
            aps.append(ap)

    overall_map = float(np.mean(aps)) if aps else 0.0

    return ModelMetrics(
        per_class=per_class,
        overall_map=overall_map,
        inference_times_ms=inference_times_ms,
    )
