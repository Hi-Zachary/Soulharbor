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
    """Keep lines under budget; evidence body + source form an atomic unit."""
    kept: List[str] = []
    used = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            line.startswith("- ")
            and i + 1 < len(lines)
            and str(lines[i + 1]).startswith("  来源：")
        ):
            unit = [line, lines[i + 1]]
            step = 2
        else:
            unit = [line]
            step = 1
        cost = estimate_tokens("\n".join(unit), counter)
        if kept and used + cost > max_tokens:
            break
        kept.extend(unit)
        used += cost
        i += step
    return kept
