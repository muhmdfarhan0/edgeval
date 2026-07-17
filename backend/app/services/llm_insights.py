"""LLM-generated narrative layered on top of the rule-based report.

This never decides pass/fail — that stays deterministic in diff_engine.py.
The LLM only explains the numbers in plain English and suggests concrete
next steps (bigger calibration set, per-channel quantization, etc.), which
is the kind of qualitative judgment a threshold check can't produce but a
human reading the report benefits from.

Calls Groq's OpenAI-compatible chat completions endpoint directly via
`requests` rather than pulling in the `groq` SDK, since the whole
integration is one POST request.
"""
from __future__ import annotations

import json
import logging

import requests

from app.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 30

SYSTEM_PROMPT = (
    "You are a computer vision deployment engineer reviewing an automated "
    "model regression report (baseline FP32 vs a quantized/optimized candidate). "
    "The pass/fail decision has already been made by a deterministic threshold "
    "check — you are NOT deciding whether to deploy. Your job is to explain the "
    "results in plain English for an engineer skimming the report, and suggest "
    "concrete, technically specific next steps. Be concise and specific to the "
    "actual numbers given — never generic filler. Respond ONLY with a JSON "
    "object matching this exact shape: "
    '{"summary": "2-3 sentence plain-English summary of what happened and why '
    'it matters", "likely_cause": "1-2 sentence technical hypothesis for why '
    'the regressed classes regressed, grounded in the specific classes/deltas '
    'given", "recommendations": ["3-5 short, concrete, actionable suggestions"]}'
)


def _build_user_prompt(report: dict, num_images: int, num_classes: int) -> str:
    worst = report["per_class_deltas"][:10]
    worst_lines = "\n".join(
        f"- {d['class']}: baseline AP {d['baseline_ap']:.3f} -> candidate AP "
        f"{d['candidate_ap']:.3f} (delta {d['delta']:+.3f}, {d['flag']})"
        for d in worst
    )
    lat = report["latency"]

    return f"""Validation set: {num_images} images, {num_classes} classes.

Overall mAP@0.5 delta (candidate - baseline): {report['overall_map_delta']:+.4f}
Flag counts: {report['flag_counts']['critical']} CRITICAL, {report['flag_counts']['warning']} WARNING, {report['flag_counts']['ok']} OK

Latency: baseline mean {lat['baseline_mean_ms']:.1f}ms, candidate mean {lat['candidate_mean_ms']:.1f}ms, speedup {lat['speedup_factor']:.2f}x

Deterministic verdict already decided: {"SAFE TO DEPLOY" if report['deploy_recommended'] else "NOT recommended for deployment"} ({report['recommendation']})

Worst per-class regressions (up to 10, worst first):
{worst_lines}
"""


def generate_insights(report: dict, num_images: int, num_classes: int) -> dict | None:
    """Returns {"summary", "likely_cause", "recommendations"} or None if the
    LLM call isn't configured or fails — this is a best-effort enhancement,
    never a blocker for the evaluation pipeline itself."""
    if not GROQ_API_KEY:
        return None

    try:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(report, num_images, num_classes)},
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        return {
            "summary": parsed.get("summary", ""),
            "likely_cause": parsed.get("likely_cause", ""),
            "recommendations": parsed.get("recommendations", []),
            "model": GROQ_MODEL,
        }
    except Exception:
        logger.exception("LLM insight generation failed; continuing without it")
        return None
