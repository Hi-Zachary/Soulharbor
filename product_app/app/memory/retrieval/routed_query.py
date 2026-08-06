"""Run hybrid search for routed subqueries."""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import RankedHit, RoutedQuery
from product_app.app.memory.retrieval.direct import DirectRetriever
from product_app.app.memory.store.fusion import reciprocal_rank


class RoutedQueryRetriever:
    def __init__(self, direct: DirectRetriever) -> None:
        self._direct = direct

    def retrieve(
        self,
        *,
        user_id: int,
        routed_queries: List[RoutedQuery],
        exclude_message_ids: Optional[Set[int]] = None,
    ) -> Tuple[List[RankedHit], int, int]:
        merged: Dict[int, RankedHit] = {}
        dense_total = 0
        sparse_total = 0

        for rq in routed_queries:
            anchors, n_dense, n_sparse = self._direct.retrieve(
                user_id=user_id,
                query=rq.query,
                exclude_message_ids=exclude_message_ids,
                role_scope=rq.role_scope,
            )
            dense_total += n_dense
            sparse_total += n_sparse
            for rank, hit in enumerate(anchors, start=1):
                hit.source_query = rq.query
                existing = merged.get(hit.chunk_id)
                if existing is None:
                    hit.fused_score = reciprocal_rank(rank)
                    merged[hit.chunk_id] = hit
                    continue
                existing.fused_score += reciprocal_rank(rank)
                if hit.semantic_rank is not None:
                    if existing.semantic_rank is None or hit.semantic_rank < existing.semantic_rank:
                        existing.semantic_rank = hit.semantic_rank
                if hit.lexical_rank is not None:
                    if existing.lexical_rank is None or hit.lexical_rank < existing.lexical_rank:
                        existing.lexical_rank = hit.lexical_rank
                if float(hit.rerank_score or 0.0) > float(existing.rerank_score or 0.0):
                    existing.rerank_score = hit.rerank_score

        ranked = sorted(merged.values(), key=lambda h: h.fused_score, reverse=True)
        return ranked[: int(mem_cfg.fuse_top_k())], dense_total, sparse_total
