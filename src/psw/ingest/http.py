"""HTTP helper: timeout 60 s, 3 retries with backoff on 429/5xx. Never logs URLs containing keys."""
from __future__ import annotations

import time

import requests

TIMEOUT = 60
RETRIES = 3
BACKOFF = 2.0  # seconds, doubled each retry


def get(url: str, params: dict | None = None, *, secret: str = "") -> requests.Response:
    """GET with retries. `secret` is scrubbed from any error message."""
    last = "no attempt"
    for attempt in range(RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            if r.status_code not in (429, 500, 502, 503, 504):
                return r
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:  # network error: scrub before re-raising
            last = type(e).__name__
        if attempt < RETRIES:
            time.sleep(BACKOFF * 2**attempt)
    msg = f"GET failed after {RETRIES + 1} attempts: {last}"
    raise RuntimeError(msg.replace(secret, "***") if secret else msg)
