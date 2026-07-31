"""Merge overlapping windows that sit next to each other in the same conversation."""
from __future__ import annotations

from typing import Dict, List

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import EpisodeWindow, WindowTurn


def merge_windows(windows: List[EpisodeWindow]) -> List[EpisodeWindow]:
    by_conversation: Dict[int, List[EpisodeWindow]] = {}
    for window in windows:
        by_conversation.setdefault(window.conversation_id, []).append(window)

    merged_all: List[EpisodeWindow] = []
    max_msgs = int(mem_cfg.bundle_max_messages)

    for conv_id, group in by_conversation.items():
        group = sorted(
            group,
            key=lambda w: (min((t.position for t in w.messages), default=0), -w.fused_score),
        )
        merged: List[EpisodeWindow] = []
        for window in group:
            if not merged:
                merged.append(window)
                continue

            prev = merged[-1]
            prev_pos = {t.position for t in prev.messages}
            cur_pos = {t.position for t in window.messages}
            overlaps = bool(prev_pos & cur_pos)
            adjacent = bool(prev_pos and cur_pos and min(cur_pos) <= max(prev_pos) + 1)

            if not (overlaps or adjacent):
                merged.append(window)
                continue

            turns: Dict[int, WindowTurn] = {t.message_id: t for t in prev.messages}
            for turn in window.messages:
                existing = turns.get(turn.message_id)
                if existing is None:
                    turns[turn.message_id] = turn
                elif turn.is_seed:
                    existing.is_seed = True

            messages = sorted(turns.values(), key=lambda t: t.position)
            if len(messages) > max_msgs:
                seed_pos = {t.position for t in messages if t.is_seed}
                center = (
                    sum(seed_pos) / len(seed_pos)
                    if seed_pos
                    else messages[len(messages) // 2].position
                )
                messages = sorted(
                    messages,
                    key=lambda t: (0 if t.is_seed else 1, abs(t.position - center)),
                )[:max_msgs]
                messages = sorted(messages, key=lambda t: t.position)

            seed_ids = sorted(
                {t.message_id for t in messages if t.is_seed} or prev.seed_ids
            )
            queries = list(
                dict.fromkeys((prev.retrieval_queries or []) + (window.retrieval_queries or []))
            )
            merged[-1] = EpisodeWindow(
                bundle_id=f"c{conv_id}-" + "-".join(str(i) for i in seed_ids[:4]),
                conversation_id=conv_id,
                seed_ids=seed_ids,
                messages=messages,
                fused_score=max(prev.fused_score, window.fused_score),
                retrieval_queries=queries,
            )

        merged_all.extend(merged)

    merged_all.sort(key=lambda w: w.fused_score, reverse=True)
    return merged_all
