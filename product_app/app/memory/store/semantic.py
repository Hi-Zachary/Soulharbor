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
    ) -> List[RankedHit]:
        text = (query or "").strip()
        if not text:
            return []

        query_vec = self._embedder.embed(text)
        if not query_vec:
            return []

        skip = {int(x) for x in (exclude_message_ids or set())}
        top_n = int(limit or mem_cfg.semantic_top_k)
        assistant_scale = float(mem_cfg.assistant_weight)

        if mem_cfg.ann_enabled:
            try:
                hits = self._search_ann(
                    user_id=user_id,
                    query_vec=query_vec,
                    top_n=top_n,
                    skip=skip,
                    assistant_scale=assistant_scale,
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
            assistant_scale=assistant_scale,
        )

    def _search_ann(
        self,
        *,
        user_id: int,
        query_vec: List[float],
        top_n: int,
        skip: Set[int],
        assistant_scale: float,
    ) -> Optional[List[RankedHit]]:
        fingerprint = self._store.active_embedding_fingerprint(user_id)
        # Need rows only when cache miss / fingerprint change.
        cached = ann_cache().peek(user_id)
        if cached is None or cached.fingerprint != fingerprint:
            rows = self._store.list_active_with_embeddings(user_id, limit=5000)
            user_index = ann_cache().get_or_build(user_id, fingerprint, rows)
        else:
            user_index = cached

        if user_index is None:
            return None
        if not user_index.rows:
            return []

        # Over-fetch so assistant down-weight + exclude_message_ids still leave top_n.
        overfetch = min(len(user_index.rows), max(top_n * 5, top_n + len(skip) + 8))
        scored = ann_cache().search(user_index, query_vec, top_k=overfetch)

        ranked: List[Tuple[float, RankedHit]] = []
        for score, row in scored:
            if row.message_id in skip:
                continue
            if row.role == "assistant":
                score *= assistant_scale
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
        assistant_scale: float,
    ) -> List[RankedHit]:
        scored: List[Tuple[float, RankedHit]] = []
        for row in self._store.list_active_with_embeddings(user_id, limit=5000):
            if row.message_id in skip or not row.embedding:
                continue
            score = cosine_similarity(query_vec, row.embedding)
            if row.role == "assistant":
                score *= assistant_scale
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
        return RankedHit(
            chunk_id=row.id,
            user_id=row.user_id,
            conversation_id=row.conversation_id,
            message_id=row.message_id,
            role=row.role,
            position=row.position,
            content=row.content,
            created_at=row.created_at,
        )
