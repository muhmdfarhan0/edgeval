"""API key auth + a basic in-memory rate limiter for the free tier.

Auth is opt-in: if EDGEVAL_API_KEY is unset, requests are allowed through
unauthenticated (keeps local dev/testing frictionless). Set the env var
in production to require the header on every request.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

API_KEY = os.environ.get("EDGEVAL_API_KEY")

# Requests allowed per identity (API key, falling back to client IP) per window.
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("EDGEVAL_RATE_LIMIT", "20"))
RATE_LIMIT_WINDOW_SECONDS = 60

_request_log: dict[str, deque] = defaultdict(deque)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY is None:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


def enforce_rate_limit(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    identity = x_api_key or (request.client.host if request.client else "unknown")
    now = time.time()
    log = _request_log[identity]

    while log and now - log[0] > RATE_LIMIT_WINDOW_SECONDS:
        log.popleft()

    if len(log) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS}s.",
        )

    log.append(now)
