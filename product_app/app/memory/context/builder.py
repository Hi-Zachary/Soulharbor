"""Assemble the final <memory> … </memory> injection block."""
from __future__ import annotations

from typing import Callable, List, Optional

from product_app.app.memory.context.formatter import format_sections
from product_app.app.memory.context.token_budget import estimate_tokens, trim_lines_to_budget
from product_app.app.memory.models import EpisodeWindow, ProfileItem


def build_memory_block(
    *,
    bundles: List[EpisodeWindow],
    profiles: List[ProfileItem],
    token_budget: int = 1600,
    token_counter: Optional[Callable[[str], int]] = None,
    query: str = "",
) -> str:
    lines = format_sections(bundles=bundles, profiles=profiles, query=query)
    if not lines:
        return ""

    kept = trim_lines_to_budget(lines, max_tokens=token_budget, counter=token_counter)
    body = "\n".join(kept).strip()
    if not body:
        return ""

    # Extra safety: drop trailing lines until we fit the budget.
    while estimate_tokens(body, token_counter) > token_budget and len(kept) > 1:
        kept = kept[:-1]
        body = "\n".join(kept).strip()

    return f"<memory>\n{body}\n</memory>"
