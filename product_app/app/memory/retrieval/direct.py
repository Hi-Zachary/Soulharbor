"""Single-query hybrid retrieval with role scope."""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import RankedHit
from product_app.app.memory.store.fusion import rrf_fuse
from product_app.app.memory.store.lexical_search import LexicalSearcher
from product_app.app.memory.store.semantic import SemanticSearcher


def retrieval_is_insufficient(hits: List[RankedHit]) -> bool:
    return len(hits) < int(mem_cfg.role_scope_fallback_min_hits)


class DirectRetriever:
    def __init__(self, semantic: SemanticSearcher, lexical: LexicalSearcher) -> None:
        self._semantic = semantic
        self._lexical = lexical

    def retrieve(
        self,
        *,
        user_id: int,
        query: str,
        exclude_message_ids: Optional[Set[int]] = None,
        role_scope: str = "both",
    ) -> Tuple[List[RankedHit], int, int]:
        scope = str(role_scope or "both").lower()
        dense = self._semantic.search(
            user_id=user_id,
            query=query,
            exclude_message_ids=exclude_message_ids,
            role_scope=scope,
        )
        sparse = self._lexical.search(
            user_id=user_id,
            query=query,
            exclude_message_ids=exclude_message_ids,
            role_scope=scope,
        )
        fused = rrf_fuse(dense, sparse)
        if scope != "both" and retrieval_is_insufficient(fused):
            dense = self._semantic.search(
                user_id=user_id,
                query=query,
                exclude_message_ids=exclude_message_ids,
                role_scope="both",
            )
            sparse = self._lexical.search(
                user_id=user_id,
                query=query,
                exclude_message_ids=exclude_message_ids,
                role_scope="both",
            )
            fused = rrf_fuse(dense, sparse)
        return fused, len(dense), len(sparse)
