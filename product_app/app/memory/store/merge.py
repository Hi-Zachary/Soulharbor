"""Merge overlapping experience windows in the same conversation."""
from __future__ import annotations

from typing import Dict, List

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import Span, SpanTurn


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
        merged: List[Span] = []
        for window in group:
            if not merged:
                merged.append(window)
                continue

            prev = merged[-1]
            prev_pos = {t.position for t in prev.messages}
            cur_pos = {t.position for t in window.messages}
            overlaps = bool(prev_pos & cur_pos)

            if not overlaps:
                merged.append(window)
                continue

            turns: Dict[int, SpanTurn] = {t.message_id: t for t in prev.messages}
            for turn in window.messages:
                existing = turns.get(turn.message_id)
                if existing is None:
                    turns[turn.message_id] = turn
                elif turn.is_anchor:
                    existing.is_anchor = True
                    if turn.matched_chunk:
                        existing.matched_chunk = turn.matched_chunk
                elif turn.matched_chunk and not existing.matched_chunk:
                    existing.matched_chunk = turn.matched_chunk

            messages = sorted(turns.values(), key=lambda t: t.position)
            if len(messages) > max_msgs:
                anchor_pos = {t.position for t in messages if t.is_anchor}
                center = (
                    sum(anchor_pos) / len(anchor_pos)
                    if anchor_pos
                    else messages[len(messages) // 2].position
                )
                messages = sorted(
                    messages,
                    key=lambda t: (0 if t.is_anchor else 1, abs(t.position - center)),
                )[:max_msgs]
                messages = sorted(messages, key=lambda t: t.position)

            anchor_ids = sorted(
                {t.message_id for t in messages if t.is_anchor} or prev.anchor_ids
            )
            queries = list(
                dict.fromkeys((prev.retrieval_queries or []) + (window.retrieval_queries or []))
            )
            merged[-1] = Span(
                bundle_id=f"c{conv_id}-" + "-".join(str(i) for i in anchor_ids[:4]),
                conversation_id=conv_id,
                anchor_ids=anchor_ids,
                messages=messages,
                fused_score=max(float(prev.fused_score or 0.0), float(window.fused_score or 0.0)),
                rerank_score=max(
                    float(prev.rerank_score or 0.0),
                    float(window.rerank_score or 0.0),
                ),
                retrieval_queries=queries,
            )

        merged_all.extend(merged)

    merged_all.sort(
        key=lambda w: (float(w.rerank_score or 0.0), float(w.fused_score or 0.0)),
        reverse=True,
    )
    return merged_all
