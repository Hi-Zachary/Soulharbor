"""MiniMax (OpenAI-compatible) API client with thread-safe connection pooling.

base_url: https://api.minimaxi.com/v1

Key point: M2.x models CANNOT disable thinking. We set `reasoning_split=true` so
thinking goes to `reasoning_content`/`reasoning_details` and `message.content` stays
clean. A `<think>...</think>` strip is applied as a safety net.

Also enforces a process-wide concurrency cap and long backoff on HTTP 429 so
nested eval parallelism does not burn the whole run on Token Plan rate limits.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SESSION_LOCK = threading.Lock()
_SHARED_SESSION: requests.Session | None = None


def _strip_think(s: str) -> str:
    s = _THINK_RE.sub("", s or "")
    if "<think>" in s:  # unclosed tail
        s = s.split("<think>")[0]
    return s.strip()


def _get_session(pool_size: int = 32) -> requests.Session:
    """One shared Session with a large connection pool for parallel workers."""
    global _SHARED_SESSION
    if _SHARED_SESSION is not None:
        return _SHARED_SESSION
    with _SESSION_LOCK:
        if _SHARED_SESSION is not None:
            return _SHARED_SESSION
        sess = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=max(8, pool_size),
            pool_maxsize=max(8, pool_size),
            max_retries=Retry(total=0, redirect=0),  # we retry ourselves
        )
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _SHARED_SESSION = sess
        return sess


def _post(cfg: Dict[str, Any], path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    # Share the same process-wide limiter Mem0 uses (rate_limit.py).
    from rate_limit import call_with_limit

    base = str(cfg.get("api_base_url", "")).rstrip("/")
    url = base + path
    headers = {
        "Authorization": f"Bearer {cfg.get('api_key', '')}",
        "Content-Type": "application/json",
    }
    timeout = int(cfg.get("request_timeout", 180))
    pool = int(cfg.get("http_pool_size", 32))
    session = _get_session(pool)
    limit = int(cfg.get("max_concurrent_api", 12))
    retries = int(cfg.get("max_retries", 20))
    b429 = float(cfg.get("retry_backoff_429_base", 1.5))
    cap = float(cfg.get("retry_backoff_cap", 60.0))
    b_other = float(cfg.get("retry_backoff_base", 2.0))

    def _once() -> Dict[str, Any]:
        r = session.post(url, headers=headers, json=body, timeout=timeout)
        if r.status_code in (429, 500, 502, 503, 504):
            err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            ra = r.headers.get("Retry-After") or r.headers.get("retry-after")
            if ra:
                try:
                    err.retry_after = float(ra)  # type: ignore[attr-defined]
                except ValueError:
                    pass
            raise err
        r.raise_for_status()
        return r.json()

    return call_with_limit(
        _once,
        limit=limit,
        max_retries=retries,
        backoff_429=b429,
        backoff_other=b_other,
        backoff_cap=cap,
    )


def chat_completion(
    cfg: Dict[str, Any], messages: List[Dict[str, str]], *, max_tokens: int, temperature: float
) -> str:
    body: Dict[str, Any] = {
        "model": cfg["chat_model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": int(max_tokens),
    }
    th = cfg.get("thinking")
    if th:  # e.g. "disabled" -> {"type":"disabled"} (M3 only; M2.x ignores/cannot disable)
        body["thinking"] = {"type": th} if isinstance(th, str) else th
    elif cfg.get("reasoning_split", True):
        # M2.x: thinking can't be disabled -> separate it so content stays clean
        body["reasoning_split"] = True
    data = _post(cfg, "/chat/completions", body)
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    return _strip_think(content)


def embed_texts(cfg: Dict[str, Any], texts: List[str]) -> List[List[float]]:
    model = cfg.get("embed_model")
    if not model:
        return [[] for _ in texts]  # null -> caller uses local bge embedder
    cleaned = [(t or "").strip() for t in texts]
    if not any(cleaned):
        return [[] for _ in texts]
    body = {"model": model, "input": cleaned}
    data = _post(cfg, "/embeddings", body)
    items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
    return [list(map(float, it["embedding"])) for it in items]
