"""Dedicated latency benchmark, separate from the per-image inference
times collected during metric computation.

Both models are timed on the identical image, in the same process, on
the same machine — isolating the latency delta to what quantization
itself changed rather than differences in image content or hardware.
"""
from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import cv2

from app.config import LATENCY_TIMED_RUNS, LATENCY_WARMUP_RUNS
from app.services.model_loader import LoadedModel


class LatencyBenchmark(TypedDict):
    mean_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    num_runs: int
    times_ms: list[float]


def benchmark_latency(
    model: LoadedModel,
    image_path: Path,
    num_runs: int = LATENCY_TIMED_RUNS,
    warmup_runs: int = LATENCY_WARMUP_RUNS,
) -> LatencyBenchmark:
    image = cv2.imread(str(image_path))

    for _ in range(warmup_runs):
        model.predict(image)

    times_ms = []
    for _ in range(num_runs):
        _, elapsed_ms = model.predict(image)
        times_ms.append(elapsed_ms)

    sorted_times = sorted(times_ms)
    p95_idx = min(int(round(0.95 * len(sorted_times))) - 1, len(sorted_times) - 1)

    return {
        "mean_ms": sum(times_ms) / len(times_ms),
        "p95_ms": sorted_times[p95_idx],
        "min_ms": sorted_times[0],
        "max_ms": sorted_times[-1],
        "num_runs": num_runs,
        "times_ms": times_ms,
    }
