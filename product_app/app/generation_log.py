from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class GenerationLog:
    """Thread-safe JSONL logger for every LLM generation call."""

    _lock = threading.Lock()
    _path: Optional[Path] = None
    _seq = 0
    _local = threading.local()

    @classmethod
    def configure(cls, path: Path | str | None) -> None:
        with cls._lock:
            cls._path = Path(path) if path else None
            cls._seq = 0
            if cls._path is not None:
                cls._path.parent.mkdir(parents=True, exist_ok=True)
                cls._path.write_text("", encoding="utf-8")

    @classmethod
    def enabled(cls) -> bool:
        return cls._path is not None

    @classmethod
    def set_context(cls, **kwargs: Any) -> None:
        ctx = getattr(cls._local, "context", None)
        if ctx is None:
            ctx = {}
            cls._local.context = ctx
        ctx.update({k: v for k, v in kwargs.items() if v is not None})

    @classmethod
    def clear_context(cls) -> None:
        cls._local.context = {}

    @classmethod
    def record(
        cls,
        *,
        route: str,
        messages: List[Dict[str, Any]],
        output: str,
        system_text: str = "",
        max_new_tokens: Optional[int] = None,
        **extra: Any,
    ) -> None:
        if cls._path is None:
            return
        with cls._lock:
            cls._seq += 1
            seq = cls._seq
            path = cls._path
        row = {
            "seq": seq,
            "ts": time.time(),
            "route": route,
            "max_new_tokens": max_new_tokens,
            "system_text": system_text or "",
            "messages": messages,
            "output": output,
            "context": dict(getattr(cls._local, "context", None) or {}),
        }
        if extra:
            row["extra"] = extra
        line = json.dumps(row, ensure_ascii=False)
        with cls._lock:
            with path.open("a", encoding="utf-8") as fout:
                fout.write(line + "\n")
