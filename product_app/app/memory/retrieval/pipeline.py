"""Retrieval pipeline: plan → hybrid → CE → collapse → stitch → merge → Top-k → chrono."""
from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Set, Tuple

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.embeddings import MemoryEmbedder
from product_app.app.memory.models import Span, ProfileItem, RetrievalTrace
from product_app.app.memory.profile.service import ProfileService
from product_app.app.memory.retrieval.direct import DirectRetriever
from product_app.app.memory.retrieval.router import QueryRouter
from product_app.app.memory.retrieval.split_query import SplitQueryRetriever
from product_app.app.memory.store.stitch import SpanStitcher
from product_app.app.memory.store.lexical_search import LexicalSearcher
from product_app.app.memory.store.merge import merge_windows
from product_app.app.memory.store.repository import TraceStore
from product_app.app.memory.store.rerank import AnchorCrossEncoder, collapse_anchor_chunks
from product_app.app.memory.store.select import select_windows, sort_by_time
from product_app.app.memory.store.semantic import SemanticSearcher

logger = logging.getLogger(__name__)


def _drop_excluded(windows: List[Span], exclude: Set[int]) -> List[Span]:
    """Remove turns that are already in the live prompt; keep window if anything remains."""
    if not exclude:
        return windows

    kept: List[Span] = []
    for window in windows:
        turns = [t for t in window.messages if t.message_id not in exclude]
        if not turns:
            continue
        anchors = [mid for mid in window.anchor_ids if mid not in exclude]
        if not anchors:
            anchors = [t.message_id for t in turns if t.is_anchor] or [turns[0].message_id]
        kept.append(
            Span(
                bundle_id=window.bundle_id,
                conversation_id=window.conversation_id,
                anchor_ids=anchors,
                messages=turns,
                fused_score=window.fused_score,
                rerank_score=window.rerank_score,
                retrieval_queries=window.retrieval_queries,
                chain_id=None,
                chain_index=0,
            )
        )
    return kept


class RetrievalPipeline:
    def __init__(self, store: TraceStore, profile: ProfileService, llm: Any = None) -> None:
        self._store = store
        self._profile = profile
        self._embedder = MemoryEmbedder.shared()
        semantic = SemanticSearcher(store)
        lexical = LexicalSearcher(store)
        self._direct = DirectRetriever(semantic, lexical)
        self._split = SplitQueryRetriever(self._direct)
        self._stitcher = SpanStitcher(store, self._embedder)
        self._anchor_ce = AnchorCrossEncoder()
        self._router = QueryRouter(llm)

    def set_llm(self, llm: Any) -> None:
        self._router.set_llm(llm)

    def run(
        self,
        *,
        user_id: int,
        query: str,
        exclude_message_ids: Optional[Set[int]] = None,
    ) -> Tuple[List[Span], List[ProfileItem], RetrievalTrace]:
        started = time.time()
        trace = RetrievalTrace(
            stitch_mode=str(mem_cfg.stitch_mode),
            selection_mode=str(mem_cfg.evidence_selection_mode),
        )
        exclude = {int(x) for x in (exclude_message_ids or set())}

        try:
            plan = self._router.plan(query)
            trace.mode = plan.mode
            trace.queries = list(plan.queries)
            trace.subquery_count = len(plan.queries)

            if plan.mode == "split" and len(plan.queries) > 1:
                anchors, n_sem, n_lex = self._split.retrieve(
                    user_id=user_id,
                    queries=plan.queries,
                    exclude_message_ids=exclude,
                )
            else:
                q = plan.queries[0] if plan.queries else query
                anchors, n_sem, n_lex = self._direct.retrieve(
                    user_id=user_id,
                    query=q,
                    exclude_message_ids=exclude,
                )

            before = len(anchors)
            anchors = self._anchor_ce.select(
                original_query=query,
                planner_queries=list(plan.queries),
                anchors=anchors,
            )
            anchors = collapse_anchor_chunks(anchors)
            trace.extra["anchor_ce"] = 1
            trace.extra["anchor_ce_mode"] = "coverage"
            trace.extra["rrf_anchors"] = before
            trace.extra["ce_anchors"] = len(anchors)

            trace.semantic_hits = n_sem
            trace.lexical_hits = n_lex
            trace.anchors = len(anchors)

            windows = self._stitcher.stitch(
                user_id=user_id, anchors=anchors, queries=plan.queries
            )
            windows = _drop_excluded(windows, exclude)
            windows = merge_windows(windows)

            # CE relevance decides which windows to inject.
            windows = select_windows(bundles=windows, limit=mem_cfg.bundle_top_k)
            # Record time only affects display order (budget packing re-ranks by CE).
            windows = sort_by_time(windows)
            trace.linked_chains = 0
            trace.bundles = len(windows)

            profiles: List[ProfileItem] = []
            if mem_cfg.profile_enabled:
                profiles = self._profile.list_for_inject(user_id)
            trace.profile_hits = len(profiles)
            return windows, profiles, trace

        except Exception:
            logger.warning("memory retrieval failed for user=%s", user_id, exc_info=True)
            trace.fallback = True
            return [], [], trace
        finally:
            trace.latency_ms = int((time.time() - started) * 1000)
