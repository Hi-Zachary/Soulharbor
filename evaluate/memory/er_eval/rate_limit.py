"""Process-wide MiniMax concurrency + 429-aware retry for both APILLM and Mem0(OpenAI)."""
from __future__ import annotations

import random
import threading
import time
from typing import Any, Callable, Optional

_SEM: Optional[threading.Semaphore] = None
_SEM_N = 0
_LOCK = threading.Lock()
_PATCHED = False
_LOG_LOCK = threading.Lock()


def get_sem(limit: int = 3) -> threading.Semaphore:
    global _SEM, _SEM_N
    limit = max(1, int(limit))
    with _LOCK:
        if _SEM is None or _SEM_N != limit:
            _SEM = threading.Semaphore(limit)
            _SEM_N = limit
        return _SEM


def _is_rate_limited(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "429" in msg
        or "rate_limit" in msg
        or "rate limit" in msg
        or "速率限制" in str(exc)
        or "too many requests" in msg
    )


def _is_retryable(exc: BaseException) -> bool:
    if _is_rate_limited(exc):
        return True
    msg = str(exc).lower()
    transient = (
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "connection",
        "temporarily",
        "overloaded",
    )
    return any(tok in msg for tok in transient)


def _sleep_for_retry(
    *,
    attempt: int,
    is_429: bool,
    backoff_429: float,
    backoff_other: float,
    backoff_cap: float,
    retry_after: float | None = None,
) -> float:
    """Return seconds slept. Prefer Retry-After; else 1–2s floor on 429 + light backoff."""
    if retry_after is not None and retry_after > 0:
        delay = min(float(backoff_cap), float(retry_after) + random.uniform(0.1, 0.5))
    elif is_429:
        # User-requested short pause on 429, then mild growth so we don't hammer the API.
        base = max(1.0, float(backoff_429))
        delay = min(
            float(backoff_cap),
            base + random.uniform(0.0, 1.0) + 0.75 * attempt,
        )
    else:
        delay = min(
            float(backoff_cap),
            float(backoff_other) * (2**attempt) + random.uniform(0.0, 0.5),
        )
    time.sleep(delay)
    return delay


def call_with_limit(
    fn: Callable[..., Any],
    *args: Any,
    limit: int = 3,
    max_retries: int = 20,
    backoff_429: float = 1.5,
    backoff_other: float = 2.0,
    backoff_cap: float = 60.0,
    retry_after: float | None = None,
    **kwargs: Any,
) -> Any:
    sem = get_sem(limit)
    last: Exception | None = None
    retries = max(1, int(max_retries))
    for attempt in range(retries):
        try:
            with sem:
                return fn(*args, **kwargs)
        except Exception as e:
            last = e
            if attempt >= retries - 1 or not _is_retryable(e):
                break
            is_429 = _is_rate_limited(e)
            # Optional per-call Retry-After (api_client may stash on exception).
            ra = retry_after
            if ra is None:
                ra = getattr(e, "retry_after", None)
            delay = _sleep_for_retry(
                attempt=attempt,
                is_429=is_429,
                backoff_429=backoff_429,
                backoff_other=backoff_other,
                backoff_cap=backoff_cap,
                retry_after=float(ra) if ra is not None else None,
            )
            with _LOG_LOCK:
                kind = "429" if is_429 else "transient"
                print(
                    f"[rate_limit] {kind} retry {attempt + 1}/{retries} "
                    f"sleep={delay:.1f}s err={type(e).__name__}: {str(e)[:120]}",
                    flush=True,
                )
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
    limit = int(cfg.get("max_concurrent_api", 12))
    retries = int(cfg.get("max_retries", 20))
    b429 = float(cfg.get("retry_backoff_429_base", 1.5))
    cap = float(cfg.get("retry_backoff_cap", 60.0))
    b_other = float(cfg.get("retry_backoff_base", 2.0))

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        return call_with_limit(
            orig,
            self,
            *args,
            limit=limit,
            max_retries=retries,
            backoff_429=b429,
            backoff_other=b_other,
            backoff_cap=cap,
            **kwargs,
        )

    Completions.create = wrapped  # type: ignore[method-assign]
    _PATCHED = True
