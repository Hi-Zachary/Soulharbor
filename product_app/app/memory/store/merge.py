"""Merge overlapping experience windows in the same conversation."""
from __future__ import annotations

from typing import Dict, List, Sequence

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import Span, SpanTurn


def _window_positions(window: Span) -> set[int]:
    return {t.position for t in window.messages}


def _trim_neighbors(messages: List[SpanTurn], max_msgs: int) -> List[SpanTurn]:
    """Cap neighbor turns only; never drop CE anchors for the soft message limit."""
    if len(messages) <= max_msgs:
        return messages

    anchor_turns = [t for t in messages if t.is_anchor]
    neighbor_turns = [t for t in messages if not t.is_anchor]
    if not neighbor_turns:
        return sorted(anchor_turns, key=lambda t: t.position)

    anchor_pos = {t.position for t in anchor_turns}
    center = (
        sum(anchor_pos) / len(anchor_pos)
        if anchor_pos
        else messages[len(messages) // 2].position
    )
    neighbor_limit = max(0, int(max_msgs) - len(anchor_turns))
    neighbor_turns.sort(key=lambda t: abs(t.position - center))
    return sorted(
        anchor_turns + neighbor_turns[:neighbor_limit],
        key=lambda t: t.position,
    )


def _merge_component(conv_id: int, component: Sequence[Span], max_msgs: int) -> Span:
    turns: Dict[int, SpanTurn] = {}
    for window in component:
        for turn in window.messages:
            existing = turns.get(turn.message_id)
            if existing is None:
                turns[turn.message_id] = turn
                continue
            if turn.is_anchor:
                existing.is_anchor = True
                if turn.matched_chunk:
                    existing.matched_chunk = turn.matched_chunk
            elif turn.matched_chunk and not existing.matched_chunk:
                existing.matched_chunk = turn.matched_chunk

    messages = _trim_neighbors(
        sorted(turns.values(), key=lambda t: t.position),
        max_msgs,
    )
    anchor_ids = sorted(
        {t.message_id for t in messages if t.is_anchor}
        or [mid for w in component for mid in w.anchor_ids]
    )
    queries: List[str] = []
    for window in component:
        for q in window.retrieval_queries or []:
            if q not in queries:
                queries.append(q)

    return Span(
        bundle_id=f"c{conv_id}-" + "-".join(str(i) for i in anchor_ids[:4]),
        conversation_id=conv_id,
        anchor_ids=anchor_ids,
        messages=messages,
        fused_score=max(float(w.fused_score or 0.0) for w in component),
        rerank_score=max(float(w.rerank_score or 0.0) for w in component),
        retrieval_queries=queries,
    )


def _overlap_components(group: Sequence[Span]) -> List[List[Span]]:
    """Union windows that share any message position (handles sparse stitch gaps)."""
    remaining = list(group)
    components: List[List[Span]] = []

    while remaining:
        component = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            component_positions = {
                pos for window in component for pos in _window_positions(window)
            }
            for window in remaining[:]:
                if component_positions & _window_positions(window):
                    component.append(window)
                    remaining.remove(window)
                    changed = True
        components.append(component)

    return components


def merge_windows(windows: List[Span]) -> List[Span]:
    """Merge windows that share a message position; leave pure-adjacent windows apart."""
    by_conversation: Dict[int, List[Span]] = {}
    for window in windows:
        by_conversation.setdefault(window.conversation_id, []).append(window)

    merged_all: List[Span] = []
    max_msgs = int(mem_cfg.bundle_max_messages)

    for conv_id, group in by_conversation.items():
        group = sorted(
            group,
            key=lambda w: (
                min((t.position for t in w.messages), default=0),
                -(float(w.rerank_score or 0.0)),
                -float(w.fused_score or 0.0),
            ),
        )
        for component in _overlap_components(group):
            merged_all.append(_merge_component(conv_id, component, max_msgs))

    merged_all.sort(
        key=lambda w: (float(w.rerank_score or 0.0), float(w.fused_score or 0.0)),
        reverse=True,
    )
    return merged_all
