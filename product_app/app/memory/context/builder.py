"""Assemble <user_profile> + <episodic_memory> injection with separate budgets."""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.context.formatter import format_sections
from product_app.app.memory.context.fragment_formatter import format_fragment_sections
from product_app.app.memory.context.profile_formatter import render_user_profile
from product_app.app.memory.context.token_budget import trim_lines_to_budget
from product_app.app.memory.models import Span, ProfileItem
from product_app.app.memory.store.select import sort_by_time
from product_app.app.memory.token_utils import count_tokens

def _ce_key(window: Span) -> tuple[float, float]:
    return (float(window.rerank_score or 0.0), float(window.fused_score or 0.0))


def _format_bundle_lines(bundles: List[Span], *, query: str = "") -> List[str]:
    if bundles and all(b.fragment is not None for b in bundles):
        return format_fragment_sections(bundles=bundles, profiles=[])
    return format_sections(bundles=bundles, profiles=[], query=query)


def pack_windows_to_budget(
    bundles: List[Span],
    *,
    profiles: List[ProfileItem] | None = None,
    token_budget: int,
    token_counter: Optional[Callable[[str], int]] = None,
    query: str = "",
) -> List[Span]:
    """Pack episodic windows by CE under token budget (profiles not in this budget)."""
    del profiles  # profile block is budgeted separately
    if not bundles:
        return []
    ranked = sorted(bundles, key=_ce_key, reverse=True)
    packed: List[Span] = []
    for window in ranked:
        trial = packed + [window]
        lines = _format_bundle_lines(sort_by_time(trial), query=query)
        body = "\n".join(lines).strip()
        block = f"<episodic_memory>\n{body}\n</episodic_memory>" if body else ""
        cost = count_tokens(block, token_counter) if block else 0
        if cost > int(token_budget):
            continue
        packed.append(window)
    return packed


def _drop_trailing_pairs(kept: List[str], token_budget: int, token_counter) -> List[str]:
    body = "\n".join(kept).strip()
    while count_tokens(body, token_counter) > token_budget and kept:
        kept = kept[:-1]
        while kept and kept[-1] == "":
            kept = kept[:-1]
        body = "\n".join(kept).strip()
    return kept


def build_episodic_memory_block(
    *,
    windows: List[Span],
    query: str = "",
    token_budget: int,
    token_counter: Optional[Callable[[str], int]] = None,
) -> Tuple[str, int]:
    packed = pack_windows_to_budget(
        windows,
        token_budget=token_budget,
        token_counter=token_counter,
        query=query,
    )
    chrono = sort_by_time(packed)
    budget = min(int(token_budget), int(mem_cfg.max_episodic_tokens))
    lines = _format_bundle_lines(chrono, query=query)
    if not lines:
        return "", 0
    kept = trim_lines_to_budget(lines, max_tokens=budget, counter=token_counter)
    kept = _drop_trailing_pairs(kept, budget, token_counter)
    body = "\n".join(kept).strip()
    if not body:
        return "", len(packed)
    return f"<episodic_memory>\n{body}\n</episodic_memory>", len(packed)


def build_memory_block(
    *,
    bundles: List[Span],
    profiles: List[ProfileItem],
    token_budget: int = 1600,
    token_counter: Optional[Callable[[str], int]] = None,
    query: str = "",
) -> Tuple[str, int]:
    """Return ``(combined_block, packed_window_count)``.

    Profile is rendered fully first (capped by persistence limits). Remaining
    budget goes to episodic ``<episodic_memory>``.
    """
    profile_block = render_user_profile(profiles) if mem_cfg.profile_enabled else ""
    profile_tokens = count_tokens(profile_block, token_counter)
    if profile_block and profile_tokens > int(mem_cfg.profile_block_max_tokens):
        # Capacity must already be repaired in ProfileStore before injection.
        raise RuntimeError(
            "profile invariant violated: "
            f"tokens={profile_tokens} max={mem_cfg.profile_block_max_tokens}"
        )

    episodic_budget = max(0, int(token_budget) - profile_tokens)
    episodic_block, packed_count = build_episodic_memory_block(
        windows=bundles,
        query=query,
        token_budget=episodic_budget,
        token_counter=token_counter,
    )

    parts = [b for b in (profile_block, episodic_block) if b]
    if not parts:
        return "", 0
    return "\n\n".join(parts), packed_count
