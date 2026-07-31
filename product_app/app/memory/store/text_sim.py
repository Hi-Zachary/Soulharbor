"""Small text helpers used by expand / link / select."""
from __future__ import annotations

import math
import re
from typing import Iterable, List, Sequence

_WORD = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}")
_ENTITY = re.compile(
    r"[\u4e00-\u9fff]{2,6}|[A-Za-z][A-Za-z0-9_\-]{1,}|\d{1,4}(?:\.\d+)?(?:度|分|周|天|年|月|日)?"
)


def tokens(text: str) -> set[str]:
    return set(_WORD.findall(text or ""))


def entities(text: str) -> set[str]:
    return set(_ENTITY.findall(text or ""))


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = norm_l = norm_r = 0.0
    for x, y in zip(left, right):
        fx, fy = float(x), float(y)
        dot += fx * fy
        norm_l += fx * fx
        norm_r += fy * fy
    if norm_l <= 0.0 or norm_r <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_l) * math.sqrt(norm_r))


def blob_of(messages: List[object], *, limit: int = 12) -> str:
    """Concatenate message contents for a cheap window-level embedding."""
    parts: List[str] = []
    for item in messages[:limit]:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        if content:
            parts.append(str(content))
    return "\n".join(parts)
