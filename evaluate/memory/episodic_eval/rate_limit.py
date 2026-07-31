"""Process-wide MiniMax concurrency + 429-aware retry for both APILLM and Mem0(OpenAI)."""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

_SEM: Optional[threading.Semaphore] = None
_SEM_N = 0
_LOCK = threading.Lock()
_PATCHED = False


def get_sem(limit: int = 3) -> threading.Semaphore:
    global _SEM, _SEM_N
    limit = max(1, int(limit))
    with _LOCK:
        if _SEM is None or _SEM_N != limit:
            _SEM = threading.Semaphore(limit)
            _SEM_N = limit
        return _SEM


def call_with_limit(
    fn: Callable[..., Any],
    *args: Any,
    limit: int = 3,
    max_retries: int = 10,
    backoff_429: float = 10.0,
    backoff_cap: float = 90.0,
    **kwargs: Any,
) -> Any:
    sem = get_sem(limit)
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            with sem:
                return fn(*args, **kwargs)
        except Exception as e:
            last = e
            msg = str(e).lower()
            is_429 = "429" in msg or "rate_limit" in msg or "速率限制" in str(e)
            if attempt >= max_retries - 1:
                break
            base = backoff_429 if is_429 else 2.0
            time.sleep(min(backoff_cap, base * (2**attempt)))
    assert last is not None
    raise last


def patch_openai_for_mem0(cfg: dict) -> None:
    """Optional: wrap OpenAI chat for Mem0. Safe no-op if openai absent."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        from openai.resources.chat.completions import Completions
    except Exception:
        return

    orig = Completions.create
    limit = int(cfg.get("max_concurrent_api", 3))
    retries = int(cfg.get("max_retries", 10))
    b429 = float(cfg.get("retry_backoff_429_base", 10.0))
    cap = float(cfg.get("retry_backoff_cap", 90.0))

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        return call_with_limit(
            orig,
            self,
            *args,
            limit=limit,
            max_retries=retries,
            backoff_429=b429,
            backoff_cap=cap,
            **kwargs,
        )

    Completions.create = wrapped  # type: ignore[method-assign]
    _PATCHED = True
