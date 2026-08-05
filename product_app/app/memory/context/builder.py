"""Assemble the final <memory> … </memory> injection block."""
from __future__ import annotations

from typing import Callable, List, Optional

from product_app.app.memory.context.formatter import format_sections
from product_app.app.memory.context.token_budget import estimate_tokens, trim_lines_to_budget
from product_app.app.memory.models import Span, ProfileItem
from product_app.app.memory.store.select import sort_by_time


def _ce_key(window: Span) -> tuple[float, float]:
    return (float(window.rerank_score or 0.0), float(window.fused_score or 0.0))


def pack_windows_to_budget(
    bundles: List[Span],
    *,
    profiles: List[ProfileItem],
    token_budget: int,
    token_counter: Optional[Callable[[str], int]] = None,
    query: str = "",
) -> List[Span]:
    """Pick windows by CE relevance until the formatted block would exceed budget.

    Chronological order is applied only after packing, so late high-CE evidence
    is not dropped by front-to-back line trimming of a time-sorted list.
    """
    if not bundles:
        return []
    ranked = sorted(bundles, key=_ce_key, reverse=True)
    packed: List[Span] = []
    for window in ranked:
        trial = packed + [window]
        lines = format_sections(
            bundles=sort_by_time(trial),
            profiles=profiles,
            query=query,
        )
        body = "\n".join(lines).strip()
        block = f"<memory>\n{body}\n</memory>" if body else ""
        cost = estimate_tokens(block, token_counter) if block else 0
        if packed and cost > int(token_budget):
            break
        packed.append(window)
    return packed


def build_memory_block(
    *,
    bundles: List[Span],
    profiles: List[ProfileItem],
    token_budget: int = 1600,
    token_counter: Optional[Callable[[str], int]] = None,
    query: str = "",
) -> str:
    packed = pack_windows_to_budget(
        bundles,
        profiles=profiles,
        token_budget=token_budget,
        token_counter=token_counter,
        query=query,
    )
    chrono = sort_by_time(packed)
    lines = format_sections(bundles=chrono, profiles=profiles, query=query)
    if not lines:
        return ""

    # Safety net only: packing already prefers CE order; avoid chopping late CE hits.
    kept = trim_lines_to_budget(lines, max_tokens=token_budget, counter=token_counter)
    body = "\n".join(kept).strip()
    if not body:
        return ""

    while estimate_tokens(body, token_counter) > token_budget and len(kept) > 1:
        kept = kept[:-1]
        body = "\n".join(kept).strip()

    return f"<memory>\n{body}\n</memory>"
