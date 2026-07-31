"""Dense retrieval over stored episode chunks."""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.embeddings import MemoryEmbedder, cosine_similarity
from product_app.app.memory.models import RankedHit
from product_app.app.memory.store.repository import EpisodeStore


class SemanticSearcher:
    def __init__(self, store: EpisodeStore, embedder: Optional[MemoryEmbedder] = None) -> None:
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

        scored: List[Tuple[float, RankedHit]] = []
        for row in self._store.list_active_with_embeddings(user_id, limit=5000):
            if row.message_id in skip or not row.embedding:
                continue
            score = cosine_similarity(query_vec, row.embedding)
            if row.role == "assistant":
                score *= assistant_scale
            if score <= 0:
                continue
            scored.append(
                (
                    score,
                    RankedHit(
                        chunk_id=row.id,
                        user_id=row.user_id,
                        conversation_id=row.conversation_id,
                        message_id=row.message_id,
                        role=row.role,
                        position=row.position,
                        content=row.content,
                        created_at=row.created_at,
                    ),
                )
            )

        scored.sort(key=lambda pair: pair[0], reverse=True)
        hits: List[RankedHit] = []
        for rank, (_score, hit) in enumerate(scored[:top_n], start=1):
            hit.semantic_rank = rank
            hits.append(hit)
        return hits
