"""Retrieval pipeline: planner → role-scoped hybrid → CE → fragments → Top-k."""
from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Set, Tuple

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.embeddings import MemoryEmbedder
from product_app.app.memory.models import RankedHit, RoutedQuery, RetrievalTrace, Span
from product_app.app.memory.profile.service import ProfileService
from product_app.app.memory.retrieval.direct import DirectRetriever
from product_app.app.memory.retrieval.router import QueryRouter
from product_app.app.memory.retrieval.routed_query import RoutedQueryRetriever
from product_app.app.memory.store.fragments import FragmentBuilder
from product_app.app.memory.store.lexical_search import LexicalSearcher
from product_app.app.memory.store.repository import TraceStore
from product_app.app.memory.store.rerank import AnchorCrossEncoder, collapse_anchor_chunks
from product_app.app.memory.store.select import sort_by_time
from product_app.app.memory.store.semantic import SemanticSearcher

logger = logging.getLogger(__name__)


def _dedupe_routed(queries: List[RoutedQuery]) -> List[RoutedQuery]:
    out: List[RoutedQuery] = []
    seen: Set[tuple[str, str]] = set()
    for rq in queries:
        key = (rq.query.casefold(), rq.role_scope)
        if not rq.query.strip() or key in seen:
            continue
        seen.add(key)
        out.append(rq)
    return out


def _fragments_to_spans(fragments: List[Any]) -> List[Span]:
    windows: List[Span] = []
    for idx, frag in enumerate(fragments, start=1):
        windows.append(
            Span(
                bundle_id=f"f{idx}-m{frag.parent_message_id}",
                conversation_id=int(frag.conversation_id),
                anchor_ids=list(frag.core_unit_ids),
                messages=[],
                fused_score=float(frag.score),
                rerank_score=float(frag.score),
                fragment=frag,
            )
        )
    return windows


class RetrievalPipeline:
    def __init__(self, store: TraceStore, profile: ProfileService, llm: Any = None) -> None:
        self._store = store
        self._profile = profile
        self._embedder = MemoryEmbedder.shared()
        semantic = SemanticSearcher(store)
        lexical = LexicalSearcher(store)
        self._direct = DirectRetriever(semantic, lexical)
        self._routed = RoutedQueryRetriever(self._direct)
        self._fragments = FragmentBuilder(store, self._embedder)
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
    ) -> Tuple[List[Span], RetrievalTrace]:
        started = time.time()
        trace = RetrievalTrace(
            stitch_mode="fragment",
            selection_mode=str(mem_cfg.evidence_selection_mode),
            original_query=(query or "").strip(),
        )
        exclude = {int(x) for x in (exclude_message_ids or set())}

        try:
            plan = self._router.plan(query)
            trace.mode = plan.mode
            trace.queries = list(plan.queries)
            trace.subquery_count = max(1, len(plan.subqueries))
            trace.planner_subqueries = [
                {"query": sq.query, "role_scope": sq.role_scope} for sq in plan.subqueries
            ]

            routed_queries = _dedupe_routed(
                [
                    RoutedQuery(query=(query or "").strip(), role_scope=plan.original_role_scope),
                    *plan.subqueries,
                ]
            )
            anchors, n_sem, n_lex = self._routed.retrieve(
                user_id=user_id,
                routed_queries=routed_queries,
                exclude_message_ids=exclude,
            )
            trace.candidate_unit_ids = [int(h.unit_id or h.chunk_id) for h in anchors]

            planner_queries = list(plan.queries[:3]) if plan.mode == "split" else []
            anchors = self._anchor_ce.select(
                original_query=query,
                planner_queries=planner_queries,
                anchors=anchors,
            )
            anchors = collapse_anchor_chunks(anchors)
            trace.reranked_unit_ids = [int(h.unit_id or h.chunk_id) for h in anchors]
            trace.selected_core_unit_ids = trace.reranked_unit_ids[:]

            trace.semantic_hits = n_sem
            trace.lexical_hits = n_lex
            trace.anchors = len(anchors)

            fragments = self._fragments.build(user_id=user_id, hits=anchors, query=query)
            for frag in fragments:
                trace.expanded_segment_ids.extend(int(x) for x in frag.expanded_unit_ids)
                if frag.reply_context_message_id:
                    trace.reply_context_message_ids.append(int(frag.reply_context_message_id))
                trace.expanded_user_message_ids.extend(
                    int(x) for x in frag.earlier_user_message_ids + frag.later_user_message_ids
                )
                trace.included_parent_message_ids.append(int(frag.parent_message_id))
                trace.included_unit_ids.extend(
                    int(x) for x in frag.core_unit_ids + frag.expanded_unit_ids
                )

            windows = _fragments_to_spans(fragments)
            windows = windows[: int(mem_cfg.max_retrieved_fragments)]
            trace.fragment_count = len(windows)
            trace.bundles = len(windows)
            trace.extra["topk_window_count"] = len(windows)
            windows = sort_by_time(windows)
            trace.profile_hits = 0
            return windows, trace

        except Exception:
            logger.warning("memory retrieval failed for user=%s", user_id, exc_info=True)
            trace.fallback = True
            return [], trace
        finally:
            trace.latency_ms = int((time.time() - started) * 1000)
