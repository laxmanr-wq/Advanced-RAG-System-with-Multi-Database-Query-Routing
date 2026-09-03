"""
RAG Agent with Database Routing - Groq free-tier quota handling.

Groq enforces a rolling tokens-per-day (TPD) limit per model. When it is
exhausted the API returns 429 with a "Please try again in Xm Ys" hint that
says when enough tokens will have rolled out of the window. Instead of
surfacing a raw error, wait out the hinted window and retry (bounded).
"""

from __future__ import annotations

import re
import time

from openai import OpenAI, RateLimitError

_HINT_MIN_SEC = re.compile(r"try again in (\d+)m\s*(\d+(?:\.\d+)?)s", re.IGNORECASE)
_HINT_SEC = re.compile(r"try again in (\d+(?:\.\d+)?)s", re.IGNORECASE)

# Cap on total time a single request may block waiting for quota.
DEFAULT_MAX_WAIT_S = 240.0


def quota_wait_seconds(error_message: str) -> float | None:
    """Parse the 'try again in ...' hint from a Groq 429 message."""
    m = _HINT_MIN_SEC.search(error_message)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = _HINT_SEC.search(error_message)
    if m:
        return float(m.group(1))
    return None


def chat_with_quota_retry(
    client: OpenAI,
    *,
    max_wait_s: float = DEFAULT_MAX_WAIT_S,
    on_wait=None,
    **kwargs,
):
    """chat.completions.create that waits out Groq TPD quota windows.

    on_wait(wait_seconds) is called before each sleep so callers (e.g. the
    UI) can surface a status message. Raises the original RateLimitError if
    the hint is missing or the total wait would exceed max_wait_s.
    """
    waited = 0.0
    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            wait = quota_wait_seconds(str(e))
            if wait is None or waited + wait > max_wait_s:
                raise
            if on_wait:
                on_wait(wait)
            time.sleep(wait + 2)
            waited += wait + 2
