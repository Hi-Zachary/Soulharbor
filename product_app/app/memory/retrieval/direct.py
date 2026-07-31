"""Single-query hybrid retrieval."""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from product_app.app.memory.models import RankedHit
from product_app.app.memory.store.fusion import rrf_fuse
from product_app.app.memory.store.lexical_search import LexicalSearcher
from product_app.app.memory.store.semantic import SemanticSearcher


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
    ) -> Tuple[List[RankedHit], int, int]:
        dense = self._semantic.search(
            user_id=user_id, query=query, exclude_message_ids=exclude_message_ids
        )
        sparse = self._lexical.search(
            user_id=user_id, query=query, exclude_message_ids=exclude_message_ids
        )
        return rrf_fuse(dense, sparse), len(dense), len(sparse)
