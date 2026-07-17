"""Turns two raw metric sets (+ latency benchmarks) into a diff report:
per-class AP deltas, threshold-based flags, and a rule-based deployment
recommendation. Deliberately not an LLM call — pass/fail against a
configured numeric tolerance is a deterministic decision, not a
judgment call.
"""
from __future__ import annotations

from typing import Dict

from app.config import AP_DROP_CRITICAL_THRESHOLD, AP_DROP_WARNING_THRESHOLD


def _flag_for_delta(delta: float) -> str:
    if delta <= -AP_DROP_CRITICAL_THRESHOLD:
        return "CRITICAL"
    if delta <= -AP_DROP_WARNING_THRESHOLD:
        return "WARNING"
    return "OK"


def build_report(
    baseline: dict,
    candidate: dict,
    baseline_latency: dict,
    candidate_latency: dict,
) -> dict:
    per_class_deltas = []

    for cls_name, b in baseline["per_class"].items():
        c = candidate["per_class"][cls_name]
        if b["num_ground_truth"] == 0:
            continue  # class not present in this validation set, no signal either way

        delta = c["ap"] - b["ap"]
        per_class_deltas.append(
            {
                "class": cls_name,
                "baseline_ap": b["ap"],
                "candidate_ap": c["ap"],
                "delta": delta,
                "flag": _flag_for_delta(delta),
                "baseline_precision": b["precision"],
                "candidate_precision": c["precision"],
                "baseline_recall": b["recall"],
                "candidate_recall": c["recall"],
                "num_ground_truth": b["num_ground_truth"],
            }
        )

    per_class_deltas.sort(key=lambda d: d["delta"])

    critical_count = sum(1 for d in per_class_deltas if d["flag"] == "CRITICAL")
    warning_count = sum(1 for d in per_class_deltas if d["flag"] == "WARNING")

    overall_map_delta = candidate["overall_map"] - baseline["overall_map"]
    latency_delta_ms = candidate_latency["mean_ms"] - baseline_latency["mean_ms"]
    speedup_factor = (
        baseline_latency["mean_ms"] / candidate_latency["mean_ms"]
        if candidate_latency["mean_ms"] > 0
        else None
    )

    deploy_recommended, summary = _deployment_recommendation(
        critical_count, warning_count, overall_map_delta, speedup_factor
    )

    return {
        "overall_map_delta": overall_map_delta,
        "per_class_deltas": per_class_deltas,
        "latency": {
            "baseline_mean_ms": baseline_latency["mean_ms"],
            "baseline_p95_ms": baseline_latency["p95_ms"],
            "candidate_mean_ms": candidate_latency["mean_ms"],
            "candidate_p95_ms": candidate_latency["p95_ms"],
            "latency_delta_ms": latency_delta_ms,
            "speedup_factor": speedup_factor,
        },
        "flag_counts": {
            "critical": critical_count,
            "warning": warning_count,
            "ok": len(per_class_deltas) - critical_count - warning_count,
        },
        "deploy_recommended": deploy_recommended,
        "recommendation": summary,
    }


def _deployment_recommendation(
    critical_count: int, warning_count: int, overall_map_delta: float, speedup_factor: float | None
) -> tuple[bool, str]:
    speedup_note = f", {speedup_factor:.1f}x speedup" if speedup_factor else ""

    if critical_count > 0:
        return (
            False,
            f"{critical_count} class(es) regressed beyond critical tolerance "
            f"({warning_count} additional warning{'s' if warning_count != 1 else ''}) — "
            f"NOT recommended for deployment.",
        )
    if warning_count > 0:
        return (
            True,
            f"{warning_count} class(es) show a moderate regression (within tolerance) "
            f"— review before deploying{speedup_note}.",
        )
    return (
        True,
        f"All classes within tolerance{speedup_note} — SAFE TO DEPLOY.",
    )
