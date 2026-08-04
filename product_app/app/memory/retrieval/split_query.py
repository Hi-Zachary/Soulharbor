"""Run hybrid search for each subquery, then merge anchors."""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import RankedHit
from product_app.app.memory.retrieval.direct import DirectRetriever
from product_app.app.memory.store.fusion import reciprocal_rank


class SplitQueryRetriever:
    def __init__(self, direct: DirectRetriever) -> None:
        self._direct = direct

    def retrieve(
        self,
        *,
        user_id: int,
        queries: List[str],
        exclude_message_ids: Optional[Set[int]] = None,
    ) -> Tuple[List[RankedHit], int, int]:
        merged: Dict[int, RankedHit] = {}
        dense_total = 0
        sparse_total = 0

        for subquery in queries:
            anchors, n_dense, n_sparse = self._direct.retrieve(
                user_id=user_id,
                query=subquery,
                exclude_message_ids=exclude_message_ids,
            )
            dense_total += n_dense
            sparse_total += n_sparse

            for rank, hit in enumerate(anchors, start=1):
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

        ranked = sorted(merged.values(), key=lambda h: h.fused_score, reverse=True)
        return ranked[: int(mem_cfg.fuse_top_k())], dense_total, sparse_total
