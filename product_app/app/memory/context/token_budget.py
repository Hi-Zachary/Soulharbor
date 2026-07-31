"""Rough token accounting for memory injection."""
from __future__ import annotations

from typing import Callable, List, Optional


def estimate_tokens(text: str, counter: Optional[Callable[[str], int]] = None) -> int:
    if counter is not None:
        try:
            return int(counter(text))
        except Exception:
            pass
    # Chinese-heavy text is denser than English BPE; ~1.5 chars/token works well enough.
    return max(1, int(len(text or "") / 1.5))


def trim_lines_to_budget(
    lines: List[str],
    *,
    max_tokens: int,
    counter: Optional[Callable[[str], int]] = None,
) -> List[str]:
    kept: List[str] = []
    used = 0
    for line in lines:
        cost = estimate_tokens(line, counter)
        if kept and used + cost > max_tokens:
            break
        kept.append(line)
        used += cost
    return kept
