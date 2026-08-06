"""Shared token counting for memory / profile budgets."""
from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Optional

TokenCounter = Optional[Callable[[str], int]]

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def fallback_token_count(text: str) -> int:
    """Conservative fallback: ~1 token per CJK char, ~3 chars otherwise."""
    value = text or ""
    if not value:
        return 0
    cjk = len(_CJK_RE.findall(value))
    other = max(0, len(value) - cjk)
    return max(1, cjk + math.ceil(other / 3))


def count_tokens(text: str, counter: TokenCounter = None) -> int:
    value = text or ""
    if not value:
        return 0
    if counter is not None:
        try:
            return max(0, int(counter(value)))
        except Exception:
            pass
    return fallback_token_count(value)


def truncate_by_tokens(
    text: str,
    *,
    max_tokens: int,
    counter: TokenCounter = None,
    keep_head_ratio: float = 0.6,
) -> str:
    """Keep head+tail when over budget; prefer head for current user messages."""
    value = text or ""
    if not value or max_tokens <= 0:
        return value
    if count_tokens(value, counter) <= int(max_tokens):
        return value

    ratio = min(0.9, max(0.1, float(keep_head_ratio)))
    head_budget = max(1, int(int(max_tokens) * ratio))
    tail_budget = max(1, int(max_tokens) - head_budget)

    # Character-step approximation then refine.
    approx_chars = max(8, int(max_tokens) * 2)
    head = value[:approx_chars]
    while head and count_tokens(head, counter) > head_budget:
        head = head[: max(0, len(head) - 8)]
    tail = value[-approx_chars:]
    while tail and count_tokens(tail, counter) > tail_budget:
        tail = tail[8:]
    if not head and not tail:
        return value[: max(1, approx_chars // 4)]
    if head.endswith(tail) or tail.startswith(head):
        return head or tail
    return f"{head}…{tail}"
