"""Reciprocal-rank fusion for semantic + lexical hit lists."""
from __future__ import annotations

from typing import Dict, List, Optional

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import RankedHit

_RRF_K = 60


def reciprocal_rank(rank: Optional[int], k: int = _RRF_K) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (k + int(rank))


def rrf_fuse(
    semantic_hits: List[RankedHit],
    lexical_hits: List[RankedHit],
    *,
    limit: int | None = None,
) -> List[RankedHit]:
    """Merge two ranked lists; higher fused_score means better anchor."""
    top_n = int(limit if limit is not None else mem_cfg.fuse_top_k())
    by_chunk: Dict[int, RankedHit] = {hit.chunk_id: hit for hit in semantic_hits}

    for hit in lexical_hits:
        existing = by_chunk.get(hit.chunk_id)
        if existing is None:
            by_chunk[hit.chunk_id] = hit
        else:
            existing.lexical_rank = hit.lexical_rank

    ranked: List[RankedHit] = []
    for hit in by_chunk.values():
        hit.fused_score = reciprocal_rank(hit.semantic_rank) + reciprocal_rank(hit.lexical_rank)
        ranked.append(hit)

    ranked.sort(key=lambda h: h.fused_score, reverse=True)
    return ranked[:top_n]
