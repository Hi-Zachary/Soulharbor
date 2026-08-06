"""Build non-contiguous windows from scored anchor turns."""
from __future__ import annotations

from typing import Dict, List, Optional

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import RankedTurn, Span, SpanTurn
from product_app.app.memory.store.repository import TraceStore


def _as_turn(
    row: dict,
    conversation_id: int,
    *,
    is_anchor: bool = False,
    segment: str = "anchor",
    matched_chunk: str | None = None,
) -> SpanTurn:
    return SpanTurn(
        message_id=int(row["message_id"]),
        conversation_id=int(conversation_id),
        turn_id=int(row.get("turn_id") or 0),
        role=str(row["role"]),
        position=int(row["position"]),
        content=str(row["content"]),
        created_at=int(row["created_at"]),
        segment=str(segment),
        is_anchor=is_anchor,
        matched_chunk=matched_chunk,
    )


def _span_sort_key(window: Span) -> tuple[float, float]:
    return (float(window.rerank_score or 0.0), float(window.fused_score or 0.0))


class SpanStitcher:
    """Build windows from ranked turns using same-turn completion and user-only expansion."""

    def __init__(self, store: TraceStore, embedder: Optional[object] = None) -> None:
        del embedder
        self._store = store

    def stitch(
        self,
        *,
        user_id: int,
        anchors: List[RankedTurn],
        queries: Optional[List[str]] = None,
    ) -> List[Span]:
        before = max(0, int(mem_cfg.neighbor_before))
        after = max(0, int(mem_cfg.neighbor_after))
        out: List[Span] = []
        used_expansion_ids: set[int] = set()

        for anchor in sorted(anchors, key=lambda a: float(a.score), reverse=True):
            conv_id = int(anchor.conversation_id)
            turn_id = int(anchor.turn_id)
            earlier = [
                _as_turn(row, conv_id, segment="earlier")
                for row in self._store.list_user_messages_before_turn(
                    user_id=user_id,
                    conversation_id=conv_id,
                    turn_id=turn_id,
                    limit=before,
                )
                if int(row["message_id"]) not in used_expansion_ids
            ]
            anchor_rows = self._store.list_turn_messages(
                user_id=user_id,
                conversation_id=conv_id,
                turn_id=turn_id,
            )
            if not anchor_rows:
                continue
            anchor_turn = [
                _as_turn(
                    row,
                    conv_id,
                    is_anchor=int(row["message_id"]) in set(anchor.anchor_message_ids),
                    segment="anchor",
                    matched_chunk=self._matched_chunk(anchor.hits, int(row["message_id"])),
                )
                for row in anchor_rows
            ]
            later = [
                _as_turn(row, conv_id, segment="later")
                for row in self._store.list_user_messages_after_turn(
                    user_id=user_id,
                    conversation_id=conv_id,
                    turn_id=turn_id,
                    limit=after,
                )
                if int(row["message_id"]) not in used_expansion_ids
            ]
            used_expansion_ids.update(t.message_id for t in earlier + later)
            messages = earlier + anchor_turn + later
            out.append(
                Span(
                    bundle_id=f"c{conv_id}-t{turn_id}",
                    conversation_id=conv_id,
                    anchor_turn_id=turn_id,
                    anchor_ids=list(anchor.anchor_message_ids),
                    messages=messages,
                    fused_score=float(anchor.score),
                    rerank_score=float(anchor.score),
                    retrieval_queries=list(queries or []),
                )
            )
        out.sort(key=_span_sort_key, reverse=True)
        return out

    @staticmethod
    def _matched_chunk(hits: List[object], message_id: int) -> str | None:
        for hit in hits:
            if int(getattr(hit, "message_id", 0)) == int(message_id):
                return str(getattr(hit, "content", "") or "")
        return None
