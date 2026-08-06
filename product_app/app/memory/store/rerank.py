"""Raw-anchor cross-encoder selection (direct / split) before Adaptive Stitch."""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Sequence, Set

import torch

from product_app.app.config import settings
from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import RankedHit

logger = logging.getLogger(__name__)


def _ce_query_set(original: str, planner_queries: Optional[Sequence[str]]) -> List[str]:
    """Always keep the raw user query; add unique Planner subqueries (≤3)."""
    out: List[str] = []
    seen: Set[str] = set()

    def _add(text: str) -> None:
        q = (text or "").strip()
        if not q:
            return
        key = q.casefold()
        if key in seen:
            return
        seen.add(key)
        out.append(q)

    _add(original)
    for pq in list(planner_queries or [])[:3]:
        _add(str(pq))
    return out


def collapse_anchor_chunks(anchors: List[RankedHit]) -> List[RankedHit]:
    """One stitch per message: keep the highest-CE chunk as the probe."""
    by_message: Dict[int, RankedHit] = {}
    for hit in anchors:
        previous = by_message.get(hit.message_id)
        if previous is None or float(hit.rerank_score) > float(previous.rerank_score):
            by_message[hit.message_id] = hit
        elif (
            float(hit.rerank_score) == float(previous.rerank_score)
            and float(hit.fused_score) > float(previous.fused_score)
        ):
            by_message[hit.message_id] = hit
    return sorted(
        by_message.values(),
        key=lambda hit: (float(hit.rerank_score), float(hit.fused_score)),
        reverse=True,
    )


def _take_unique_messages(
    anchors: List[RankedHit],
    limit: int,
) -> List[RankedHit]:
    """Keep first occurrences up to ``limit`` distinct message_id values."""
    if limit <= 0:
        return []
    selected: List[RankedHit] = []
    seen: Set[int] = set()
    for hit in anchors:
        if hit.message_id in seen:
            continue
        selected.append(hit)
        seen.add(hit.message_id)
        if len(selected) >= limit:
            break
    return selected


def _select_direct(
    anchors: List[RankedHit],
    scores: List[float],
    *,
    limit: int = 10,
) -> List[RankedHit]:
    """Original-query CE Top-K, unique by message_id."""
    if not anchors or limit <= 0:
        return []
    ranked_indices = sorted(
        range(len(anchors)),
        key=lambda i: float(scores[i]),
        reverse=True,
    )
    selected: List[RankedHit] = []
    seen_message_ids: Set[int] = set()
    for index in ranked_indices:
        hit = anchors[index]
        if hit.message_id in seen_message_ids:
            continue
        hit.rerank_score = float(scores[index])
        selected.append(hit)
        seen_message_ids.add(hit.message_id)
        if len(selected) >= limit:
            break
    return selected


def _select_split(
    anchors: List[RankedHit],
    score_matrix: List[List[float]],
    *,
    per_subquery: int = 3,
    limit: int = 10,
) -> List[RankedHit]:
    """
    Per-subquery CE Top-K ∪ fill from original-query CE to ``limit``.

    score_matrix[0] = original query; score_matrix[1:] = up to 3 subqueries.
    Final rerank_score = max CE across all queries for that chunk.
    """
    if not anchors or not score_matrix or limit <= 0:
        return []

    original_scores = score_matrix[0]
    subquery_scores = score_matrix[1:]

    def best_indices_for_scores(scores: List[float]) -> List[int]:
        best_by_message: Dict[int, int] = {}
        for index, hit in enumerate(anchors):
            previous = best_by_message.get(hit.message_id)
            if previous is None or float(scores[index]) > float(scores[previous]):
                best_by_message[hit.message_id] = index
        return sorted(
            best_by_message.values(),
            key=lambda i: float(scores[i]),
            reverse=True,
        )

    selected_by_message: Dict[int, int] = {}

    for scores in subquery_scores:
        ranked = best_indices_for_scores(scores)
        for index in ranked[: max(0, int(per_subquery))]:
            message_id = anchors[index].message_id
            previous = selected_by_message.get(message_id)
            if previous is None:
                selected_by_message[message_id] = index
                continue
            previous_max = max(float(row[previous]) for row in score_matrix)
            current_max = max(float(row[index]) for row in score_matrix)
            if current_max > previous_max:
                selected_by_message[message_id] = index

    for index in best_indices_for_scores(original_scores):
        if len(selected_by_message) >= limit:
            break
        message_id = anchors[index].message_id
        if message_id in selected_by_message:
            continue
        selected_by_message[message_id] = index

    selected_indices = list(selected_by_message.values())
    for index in selected_indices:
        anchors[index].rerank_score = max(float(row[index]) for row in score_matrix)

    selected_indices.sort(
        key=lambda i: float(anchors[i].rerank_score),
        reverse=True,
    )
    return [anchors[index] for index in selected_indices[:limit]]


class AnchorCrossEncoder:
    """Shared BGE-reranker for raw-anchor direct / split CE selection."""

    _lock = threading.Lock()
    _shared_model = None
    _shared_tokenizer = None
    _shared_path: Optional[str] = None
    _shared_device: Optional[str] = None

    def __init__(self) -> None:
        self._load_lock = threading.Lock()

    def select(
        self,
        *,
        original_query: str,
        planner_queries: List[str],
        anchors: List[RankedHit],
        keep: int | None = None,
    ) -> List[RankedHit]:
        """
        Score RRF candidates with CE.

        - Direct (no planner_queries): original-query Top-K by message_id.
        - Split: each subquery Top-`per_subquery`, union, fill from original CE.
        """
        if not anchors:
            return []
        subs = [str(q).strip() for q in (planner_queries or []) if str(q).strip()][:3]
        queries = _ce_query_set(original_query, subs)
        if not queries:
            return []

        texts = [(a.content or "").strip() for a in anchors]
        try:
            score_matrix = [self._score_text_pairs(q, texts) for q in queries]
        except Exception:
            logger.warning("raw-anchor cross-encoder scoring failed", exc_info=True)
            fallback_k = int(
                keep
                if keep is not None
                else (
                    mem_cfg.anchor_ce_top_k
                    if subs
                    else mem_cfg.anchor_ce_direct_k
                )
            )
            return _take_unique_messages(anchors, fallback_k)

        if not subs:
            limit = int(keep if keep is not None else mem_cfg.anchor_ce_direct_k)
            return _select_direct(anchors, score_matrix[0], limit=limit)

        limit = int(keep if keep is not None else mem_cfg.anchor_ce_top_k)
        return _select_split(
            anchors,
            score_matrix,
            per_subquery=int(mem_cfg.anchor_ce_per_subquery_k),
            limit=limit,
        )

    def _score_text_pairs(self, query: str, passages: List[str]) -> List[float]:
        model, tokenizer, device = self._ensure_model()
        max_len = int(mem_cfg.rerank_max_length)
        batch_size = max(1, int(mem_cfg.rerank_batch_size))
        scores: List[float] = []

        model.eval()
        with torch.no_grad():
            for start in range(0, len(passages), batch_size):
                batch_passages = passages[start : start + batch_size]
                batch_queries = [query] * len(batch_passages)
                feats = tokenizer(
                    batch_queries,
                    batch_passages,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                feats = {k: v.to(device) for k, v in feats.items()}
                logits = model(**feats).logits.view(-1).float()
                probs = torch.sigmoid(logits)
                scores.extend(float(x) for x in probs.detach().cpu())
        return scores

    def _resolve_device(self) -> str:
        pref = str(
            getattr(settings, "memory_rerank_device", None)
            or getattr(settings, "memory_embed_device", "cpu")
            or "cpu"
        ).lower()
        if pref in {"cuda", "gpu"} and torch.cuda.is_available():
            return "cuda"
        if pref.startswith("cuda:") and torch.cuda.is_available():
            return pref
        return "cpu"

    def _ensure_model(self):
        path = str(getattr(settings, "memory_reranker_base", "") or "")
        device = self._resolve_device()

        with AnchorCrossEncoder._lock:
            if (
                AnchorCrossEncoder._shared_model is not None
                and AnchorCrossEncoder._shared_path == path
                and AnchorCrossEncoder._shared_device == device
            ):
                return (
                    AnchorCrossEncoder._shared_model,
                    AnchorCrossEncoder._shared_tokenizer,
                    device,
                )

        with self._load_lock:
            with AnchorCrossEncoder._lock:
                if (
                    AnchorCrossEncoder._shared_model is not None
                    and AnchorCrossEncoder._shared_path == path
                    and AnchorCrossEncoder._shared_device == device
                ):
                    return (
                        AnchorCrossEncoder._shared_model,
                        AnchorCrossEncoder._shared_tokenizer,
                        device,
                    )

            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(path)
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
            model = AutoModelForSequenceClassification.from_pretrained(path, dtype=dtype)
            model.to(device)
            model.eval()

            with AnchorCrossEncoder._lock:
                AnchorCrossEncoder._shared_model = model
                AnchorCrossEncoder._shared_tokenizer = tokenizer
                AnchorCrossEncoder._shared_path = path
                AnchorCrossEncoder._shared_device = device
            return model, tokenizer, device

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._shared_model = None
            cls._shared_tokenizer = None
            cls._shared_path = None
            cls._shared_device = None
