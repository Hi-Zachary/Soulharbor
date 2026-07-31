"""Retrieval pipeline: route → search → expand → merge → rerank → link → select."""
from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Set, Tuple

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.embeddings import MemoryEmbedder
from product_app.app.memory.models import EpisodeWindow, ProfileItem, RetrievalTrace
from product_app.app.memory.profile.service import ProfileService
from product_app.app.memory.retrieval.direct import DirectRetriever
from product_app.app.memory.retrieval.router import QueryRouter
from product_app.app.memory.retrieval.split_query import SplitQueryRetriever
from product_app.app.memory.store.expand import WindowExpander
from product_app.app.memory.store.lexical_search import LexicalSearcher
from product_app.app.memory.store.link import link_windows
from product_app.app.memory.store.merge import merge_windows
from product_app.app.memory.store.rerank import WindowReranker
from product_app.app.memory.store.repository import EpisodeStore
from product_app.app.memory.store.select import select_windows
from product_app.app.memory.store.semantic import SemanticSearcher

logger = logging.getLogger(__name__)


def _drop_excluded(windows: List[EpisodeWindow], exclude: Set[int]) -> List[EpisodeWindow]:
    """Remove turns that are already in the live prompt; keep window if anything remains."""
    if not exclude:
        return windows

    kept: List[EpisodeWindow] = []
    for window in windows:
        turns = [t for t in window.messages if t.message_id not in exclude]
        if not turns:
            continue
        seeds = [mid for mid in window.seed_ids if mid not in exclude]
        if not seeds:
            seeds = [t.message_id for t in turns if t.is_seed] or [turns[0].message_id]
        kept.append(
            EpisodeWindow(
                bundle_id=window.bundle_id,
                conversation_id=window.conversation_id,
                seed_ids=seeds,
                messages=turns,
                fused_score=window.fused_score,
                rerank_score=window.rerank_score,
                retrieval_queries=window.retrieval_queries,
                chain_id=window.chain_id,
                chain_index=window.chain_index,
            )
        )
    return kept


class RetrievalPipeline:
    def __init__(self, store: EpisodeStore, profile: ProfileService, llm: Any = None) -> None:
        self._store = store
        self._profile = profile
        self._embedder = MemoryEmbedder.shared()
        semantic = SemanticSearcher(store)
        lexical = LexicalSearcher(store)
        self._direct = DirectRetriever(semantic, lexical)
        self._split = SplitQueryRetriever(self._direct)
        self._expander = WindowExpander(store, self._embedder)
        self._reranker = WindowReranker(llm)
        self._router = QueryRouter(llm)

    def set_llm(self, llm: Any) -> None:
        self._reranker.set_llm(llm)
        self._router.set_llm(llm)

    def run(
        self,
        *,
        user_id: int,
        query: str,
        exclude_message_ids: Optional[Set[int]] = None,
    ) -> Tuple[List[EpisodeWindow], List[ProfileItem], RetrievalTrace]:
        started = time.time()
        trace = RetrievalTrace(
            expansion_mode=str(mem_cfg.expand_mode),
            selection_mode=str(mem_cfg.evidence_selection_mode),
        )
        exclude = {int(x) for x in (exclude_message_ids or set())}

        try:
            plan = self._router.plan(query)
            trace.mode = plan.mode
            trace.queries = list(plan.queries)
            trace.subquery_count = len(plan.queries)

            if plan.mode == "split" and len(plan.queries) > 1:
                seeds, n_sem, n_lex = self._split.retrieve(
                    user_id=user_id,
                    queries=plan.queries,
                    exclude_message_ids=exclude,
                )
            else:
                q = plan.queries[0] if plan.queries else query
                seeds, n_sem, n_lex = self._direct.retrieve(
                    user_id=user_id,
                    query=q,
                    exclude_message_ids=exclude,
                )

            trace.semantic_hits = n_sem
            trace.lexical_hits = n_lex
            trace.seeds = len(seeds)

            windows = self._expander.expand(
                user_id=user_id, seeds=seeds, queries=plan.queries
            )
            windows = _drop_excluded(windows, exclude)
            windows = merge_windows(windows)

            pool = max(int(mem_cfg.bundle_top_k) * 2, 8)
            windows = self._reranker.rerank(query=query, bundles=windows, limit=pool)

            windows, n_chains = link_windows(
                query=query, bundles=windows, embedder=self._embedder
            )
            trace.linked_chains = n_chains

            windows = select_windows(query=query, bundles=windows)
            trace.bundles = len(windows)

            profiles: List[ProfileItem] = []
            if mem_cfg.profile_enabled:
                profiles = self._profile.search(user_id, query, limit=5)
            trace.profile_hits = len(profiles)
            return windows, profiles, trace

        except Exception:
            logger.warning("memory retrieval failed for user=%s", user_id, exc_info=True)
            trace.fallback = True
            return [], [], trace
        finally:
            trace.latency_ms = int((time.time() - started) * 1000)
