"""Stitch each retrieval anchor into a short conversation span."""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.embeddings import MemoryEmbedder
from product_app.app.memory.models import Span, RankedHit, SpanTurn
from product_app.app.memory.store.repository import TraceStore
from product_app.app.memory.store.text_sim import blob_of, cosine, entities


def _as_turn(row: dict, conversation_id: int, *, is_anchor: bool = False) -> SpanTurn:
    return SpanTurn(
        message_id=int(row["message_id"]),
        conversation_id=int(conversation_id),
        role=str(row["role"]),
        position=int(row["position"]),
        content=str(row["content"]),
        created_at=int(row["created_at"]),
        is_anchor=is_anchor,
    )


def _window_span(turns: Sequence[SpanTurn]) -> int:
    if not turns:
        return 0
    positions = [t.position for t in turns]
    return max(positions) - min(positions) + 1


class SpanStitcher:
    """Turn ranked hits into multi-turn windows (adaptive walk or fixed neighbors)."""

    def __init__(self, store: TraceStore, embedder: Optional[MemoryEmbedder] = None) -> None:
        self._store = store
        self._embedder = embedder
        self._vec_cache: Dict[str, List[float]] = {}

    def stitch(
        self,
        *,
        user_id: int,
        anchors: List[RankedHit],
        queries: Optional[List[str]] = None,
    ) -> List[Span]:
        if (mem_cfg.stitch_mode or "adaptive").lower() == "fixed":
            return self._fixed_windows(user_id, anchors, queries)
        return self._adaptive_windows(user_id, anchors, queries)

    # --- embeddings ---------------------------------------------------------

    @staticmethod
    def _cache_key(text: str) -> str:
        raw = text or ""
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        return f"{len(raw)}:{digest}"

    def _vector(self, text: str) -> List[float]:
        key = self._cache_key(text)
        cached = self._vec_cache.get(key)
        if cached is not None:
            return cached
        embedder = self._embedder or MemoryEmbedder.shared()
        vec = list(embedder.embed(text or "") or [])
        self._vec_cache[key] = vec
        return vec

    # --- adaptive: cos primary + adjacent entity escape ---------------------

    def _keep_neighbor(
        self,
        candidate: SpanTurn,
        anchor: SpanTurn,
        selected: Sequence[SpanTurn],
        threshold: float,
    ) -> bool:
        """Include if cos(c, anchor) >= tau, or adjacent with shared entities."""
        if cosine(self._vector(candidate.content), self._vector(anchor.content)) >= threshold:
            return True

        positions = [t.position for t in selected]
        dist = min(
            abs(candidate.position - min(positions)),
            abs(candidate.position - max(positions)),
        )
        if dist > int(mem_cfg.stitch_entity_dist):
            return False
        window_ents = entities(blob_of(list(selected)))
        return bool(entities(candidate.content) & window_ents)

    def _grow_from_hit(self, user_id: int, hit: RankedHit, query: str) -> Span:
        del query  # expansion no longer uses query-overlap terms
        conv_id = int(hit.conversation_id)
        rows = self._store.list_conversation_messages(user_id=user_id, conversation_id=conv_id)
        by_pos = {int(r["position"]): r for r in rows}

        if hit.position not in by_pos:
            anchor = SpanTurn(
                message_id=hit.message_id,
                conversation_id=conv_id,
                role=hit.role,
                position=hit.position,
                content=hit.content,
                created_at=hit.created_at,
                is_anchor=True,
            )
            return Span(
                bundle_id=f"c{conv_id}-{hit.message_id}",
                conversation_id=conv_id,
                anchor_ids=[hit.message_id],
                messages=[anchor],
                fused_score=float(hit.fused_score or 0.0),
            )

        anchor = _as_turn(by_pos[hit.position], conv_id, is_anchor=True)
        selected: List[SpanTurn] = [anchor]
        threshold = float(mem_cfg.stitch_cos_threshold)
        max_span = int(mem_cfg.stitch_max_span)
        max_msgs = int(mem_cfg.bundle_max_messages)
        max_misses = int(mem_cfg.stitch_max_misses)

        # walk left
        misses = 0
        pos = hit.position - 1
        while pos >= 1 and len(selected) < max_msgs and _window_span(selected) < max_span:
            row = by_pos.get(pos)
            if row is None:
                pos -= 1
                continue
            cand = _as_turn(row, conv_id)
            if self._keep_neighbor(cand, anchor, selected, threshold):
                selected.insert(0, cand)
                misses = 0
            else:
                misses += 1
                if misses >= max_misses:
                    break
            pos -= 1

        # walk right
        misses = 0
        pos = hit.position + 1
        last_pos = max(by_pos) if by_pos else hit.position
        while pos <= last_pos and len(selected) < max_msgs and _window_span(selected) < max_span:
            row = by_pos.get(pos)
            if row is None:
                pos += 1
                continue
            cand = _as_turn(row, conv_id)
            if self._keep_neighbor(cand, anchor, selected, threshold):
                selected.append(cand)
                misses = 0
            else:
                misses += 1
                if misses >= max_misses:
                    break
            pos += 1

        return Span(
            bundle_id=f"c{conv_id}-{hit.message_id}",
            conversation_id=conv_id,
            anchor_ids=[hit.message_id],
            messages=selected,
            fused_score=float(hit.fused_score or 0.0),
        )

    def _adaptive_windows(
        self,
        user_id: int,
        anchors: List[RankedHit],
        queries: Optional[List[str]],
    ) -> List[Span]:
        joined = " ".join(queries or [])
        windows = [
            self._grow_from_hit(user_id, hit, joined or hit.content) for hit in anchors
        ]
        for w in windows:
            w.retrieval_queries = list(queries or [])
        windows.sort(key=lambda w: w.fused_score, reverse=True)
        keep = max(int(mem_cfg.bundle_top_k) * 2, int(mem_cfg.anchor_top_k))
        return windows[:keep]

    # --- fixed neighbor windows ---------------------------------------------

    def _fixed_windows(
        self,
        user_id: int,
        anchors: List[RankedHit],
        queries: Optional[List[str]],
    ) -> List[Span]:
        before = int(mem_cfg.neighbor_before)
        after = int(mem_cfg.neighbor_after)
        max_msgs = int(mem_cfg.bundle_max_messages)

        ranges: Dict[int, List[Tuple[int, int, RankedHit]]] = {}
        for hit in anchors:
            ranges.setdefault(hit.conversation_id, []).append(
                (hit.position - before, hit.position + after, hit)
            )

        out: List[Span] = []
        for conv_id, items in ranges.items():
            items.sort(key=lambda x: x[0])
            merged: List[Tuple[int, int, List[RankedHit]]] = []
            for lo, hi, hit in items:
                if merged and lo <= merged[-1][1] + 1:
                    prev_lo, prev_hi, hits = merged[-1]
                    merged[-1] = (prev_lo, max(prev_hi, hi), hits + [hit])
                else:
                    merged.append((lo, hi, [hit]))

            for _lo, _hi, hits in merged:
                by_id: Dict[int, SpanTurn] = {}
                anchor_ids = sorted({h.message_id for h in hits})
                best_score = max(h.fused_score for h in hits)
                for hit in hits:
                    for row in self._store.neighbor_messages(
                        user_id=user_id,
                        conversation_id=conv_id,
                        position=hit.position,
                        before=before,
                        after=after,
                    ):
                        mid = int(row["message_id"])
                        marked = mid in anchor_ids
                        existing = by_id.get(mid)
                        if existing is None:
                            by_id[mid] = _as_turn(row, conv_id, is_anchor=marked)
                        elif marked:
                            existing.is_anchor = True

                turns = sorted(by_id.values(), key=lambda t: t.position)
                if len(turns) > max_msgs:
                    anchor_pos = {t.position for t in turns if t.is_anchor}
                    if not anchor_pos:
                        turns = turns[:max_msgs]
                    else:
                        center = sum(anchor_pos) / len(anchor_pos)
                        turns = sorted(
                            turns,
                            key=lambda t: (0 if t.is_anchor else 1, abs(t.position - center)),
                        )[:max_msgs]
                        turns = sorted(turns, key=lambda t: t.position)

                out.append(
                    Span(
                        bundle_id=f"c{conv_id}-" + "-".join(str(i) for i in anchor_ids[:4]),
                        conversation_id=conv_id,
                        anchor_ids=anchor_ids,
                        messages=turns,
                        fused_score=best_score,
                        retrieval_queries=list(queries or []),
                    )
                )

        out.sort(key=lambda w: w.fused_score, reverse=True)
        return out[: int(mem_cfg.bundle_top_k)]
