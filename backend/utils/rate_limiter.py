"""
Lightweight in-memory rate limiter middleware for FastAPI.

Uses a per-IP sliding window stored in a dict. Automatically prunes stale
entries to avoid unbounded memory growth.

Configure via environment variables:
    RATE_LIMIT_REQUESTS   – max requests per window  (default: 60)
    RATE_LIMIT_WINDOW_S   – window size in seconds   (default: 60)
"""

import os
import time
import logging
import threading
from collections import defaultdict
from typing import Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_S", "60"))

# Stricter limit for expensive endpoints (upload / auth)
STRICT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_STRICT_REQUESTS", "10"))
STRICT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_STRICT_WINDOW_S", "60"))

# Paths that get the stricter limit
_STRICT_PATHS = {"/api/analyze", "/api/auth/login", "/api/auth/register"}

# ── Storage ───────────────────────────────────────────────────────────────────
_lock = threading.Lock()
_hits: dict[str, list[float]] = defaultdict(list)   # IP -> list of timestamps

_PRUNE_INTERVAL = 300  # seconds between full prune sweeps
_last_prune: float = 0.0


def _get_client_ip(request: Request) -> str:
    """Best-effort client IP (respects X-Forwarded-For behind a reverse proxy)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate(key: str, max_reqs: int, window: int) -> Tuple[bool, int, float]:
    """
    Returns (allowed, remaining, retry_after).
    Prunes timestamps older than the window.
    """
    global _last_prune
    now = time.monotonic()

    with _lock:
        # Periodic full prune of stale IPs
        if now - _last_prune > _PRUNE_INTERVAL:
            stale = [k for k, v in _hits.items() if not v or v[-1] < now - window * 2]
            for k in stale:
                del _hits[k]
            _last_prune = now

        timestamps = _hits[key]
        cutoff = now - window
        # Drop timestamps outside the window
        while timestamps and timestamps[0] < cutoff:
            timestamps.pop(0)

        if len(timestamps) >= max_reqs:
            retry_after = timestamps[0] + window - now
            return False, 0, max(retry_after, 1.0)

        timestamps.append(now)
        remaining = max_reqs - len(timestamps)
        return True, remaining, 0.0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Drop-in FastAPI middleware that enforces per-IP rate limits."""

    async def dispatch(self, request: Request, call_next) -> Response:
        ip = _get_client_ip(request)
        path = request.url.path.rstrip("/")

        # Choose limit tier
        if path in _STRICT_PATHS:
            max_reqs, window = STRICT_MAX_REQUESTS, STRICT_WINDOW_SECONDS
            key = f"strict:{ip}"
        else:
            max_reqs, window = MAX_REQUESTS, WINDOW_SECONDS
            key = f"global:{ip}"

        allowed, remaining, retry_after = _check_rate(key, max_reqs, window)

        if not allowed:
            logger.warning(f"Rate limit exceeded for {ip} on {path}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down."},
                headers={
                    "Retry-After": str(int(retry_after)),
                    "X-RateLimit-Limit": str(max_reqs),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response: Response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(max_reqs)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
