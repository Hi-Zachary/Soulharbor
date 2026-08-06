"""Shared token counting for memory / profile budgets."""
from __future__ import annotations

from collections.abc import Callable
from typing import Optional

TokenCounter = Optional[Callable[[str], int]]


def count_tokens(text: str, counter: TokenCounter = None) -> int:
    value = text or ""
    if not value:
        return 0
    if counter is not None:
        try:
            return max(0, int(counter(value)))
        except Exception:
            pass
    # Conservative estimate when tokenizer is unavailable.
    return max(1, int(len(value) / 1.5))
