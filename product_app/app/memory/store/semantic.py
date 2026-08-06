"""Dense retrieval over stored trace chunks."""
from __future__ import annotations

import logging
from typing import List, Optional, Set, Tuple

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.embeddings import MemoryEmbedder, cosine_similarity
from product_app.app.memory.models import Block, RankedHit
from product_app.app.memory.store.ann_index import ann_cache
from product_app.app.memory.store.repository import TraceStore

logger = logging.getLogger(__name__)


class SemanticSearcher:
    def __init__(self, store: TraceStore, embedder: Optional[MemoryEmbedder] = None) -> None:
        self._store = store
        self._embedder = embedder or MemoryEmbedder.shared()

    def search(
        self,
        *,
        user_id: int,
        query: str,
        limit: int | None = None,
        exclude_message_ids: Optional[Set[int]] = None,
        role_scope: str = "both",
    ) -> List[RankedHit]:
        text = (query or "").strip()
        if not text:
            return []

        query_vec = self._embedder.embed(text)
        if not query_vec:
            return []

        skip = {int(x) for x in (exclude_message_ids or set())}
        top_n = int(limit or mem_cfg.semantic_top_k)

        if mem_cfg.ann_enabled:
            try:
                hits = self._search_ann(
                    user_id=user_id,
                    query_vec=query_vec,
                    top_n=top_n,
                    skip=skip,
                    role_scope=role_scope,
                )
                if hits is not None:
                    return hits
            except Exception:
                logger.warning("ANN semantic search failed; falling back to scan", exc_info=True)

        return self._search_scan(
            user_id=user_id,
            query_vec=query_vec,
            top_n=top_n,
            skip=skip,
            role_scope=role_scope,
        )

    def _search_ann(
        self,
        *,
        user_id: int,
        query_vec: List[float],
        top_n: int,
        skip: Set[int],
        role_scope: str,
    ) -> Optional[List[RankedHit]]:
        fingerprint = self._store.active_embedding_fingerprint(user_id)
        kind = f"message_level:{role_scope}"
        cached = ann_cache().peek(user_id, kind=kind)
        if cached is None or cached.fingerprint != fingerprint:
            rows = self._store.list_active_with_embeddings(
                user_id, limit=5000, role_scope=role_scope
            )
            user_index = ann_cache().get_or_build(
                user_id, fingerprint, rows, kind=kind
            )
        else:
            user_index = cached

        if user_index is None:
            return None
        if not user_index.rows:
            return []

        # Over-fetch so exclude_message_ids still leave top_n hits.
        overfetch = min(len(user_index.rows), max(top_n * 5, top_n + len(skip) + 8))
        scored = ann_cache().search(user_index, query_vec, top_k=overfetch)

        ranked: List[Tuple[float, RankedHit]] = []
        for score, row in scored:
            if row.parent_message_id in skip or row.message_id in skip:
                continue
            if score <= 0:
                continue
            ranked.append((score, self._to_hit(row)))

        ranked.sort(key=lambda pair: pair[0], reverse=True)
        hits: List[RankedHit] = []
        for rank, (_score, hit) in enumerate(ranked[:top_n], start=1):
            hit.semantic_rank = rank
            hits.append(hit)
        return hits

    def _search_scan(
        self,
        *,
        user_id: int,
        query_vec: List[float],
        top_n: int,
        skip: Set[int],
        role_scope: str,
    ) -> List[RankedHit]:
        scored: List[Tuple[float, RankedHit]] = []
        for row in self._store.list_active_with_embeddings(
            user_id, limit=5000, role_scope=role_scope
        ):
            if row.parent_message_id in skip or row.message_id in skip or not row.embedding:
                continue
            score = cosine_similarity(query_vec, row.embedding)
            if score <= 0:
                continue
            scored.append((score, self._to_hit(row)))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        hits: List[RankedHit] = []
        for rank, (_score, hit) in enumerate(scored[:top_n], start=1):
            hit.semantic_rank = rank
            hits.append(hit)
        return hits

    @staticmethod
    def _to_hit(row: Block) -> RankedHit:
        unit_type = str(row.unit_type or "message")
        unit_id = int(row.segment_id or row.message_id) if unit_type == "segment" else int(row.message_id)
        return RankedHit(
            chunk_id=row.id,
            user_id=row.user_id,
            conversation_id=row.conversation_id,
            message_id=row.message_id,
            turn_id=row.turn_id,
            role=row.role,
            position=row.position,
            content=row.content,
            created_at=row.created_at,
            unit_type=unit_type,
            unit_id=unit_id,
            parent_message_id=int(row.parent_message_id or row.message_id),
            segment_id=int(row.segment_id or 0),
            segment_index=int(row.segment_index),
        )
