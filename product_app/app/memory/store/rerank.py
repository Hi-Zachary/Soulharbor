"""Raw-anchor cross-encoder selection (multi-query coverage) before Adaptive Stitch."""
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
    """Always keep the raw user query; add unique Planner rewrites / subqueries."""
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
    for pq in planner_queries or []:
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


def _coverage_select_anchors(
    anchors: List[RankedHit],
    score_matrix: List[List[float]],
    *,
    keep: int,
    orig_top: int,
    sub_top: int,
    min_per_query: int,
) -> List[RankedHit]:
    """
    Coverage-aware CE seed pick:
    original top-`orig_top` ∪ each subquery top-`sub_top`, then fill/truncate by s_max.
    """
    n = len(anchors)
    if n == 0 or keep <= 0 or not score_matrix:
        return []
    n_q = len(score_matrix)
    s_max = [max(float(score_matrix[j][i]) for j in range(n_q)) for i in range(n)]
    for i, anchor in enumerate(anchors):
        anchor.rerank_score = float(s_max[i])

    selected: List[int] = []
    seen_indices: Set[int] = set()
    seen_message_ids: Set[int] = set()

    def _take(scores: List[float], k: int) -> None:
        if k <= 0:
            return
        order = sorted(range(n), key=lambda i: float(scores[i]), reverse=True)
        added = 0
        for index in order:
            message_id = anchors[index].message_id
            if index in seen_indices:
                continue
            if message_id in seen_message_ids:
                continue
            selected.append(index)
            seen_indices.add(index)
            seen_message_ids.add(message_id)
            added += 1
            if added >= k:
                break

    _take(score_matrix[0], int(orig_top))
    for j in range(1, n_q):
        _take(score_matrix[j], int(sub_top))

    if len(selected) < keep:
        order = sorted(range(n), key=lambda i: s_max[i], reverse=True)
        for index in order:
            if len(selected) >= keep:
                break
            message_id = anchors[index].message_id
            if index in seen_indices:
                continue
            if message_id in seen_message_ids:
                continue
            selected.append(index)
            seen_indices.add(index)
            seen_message_ids.add(message_id)

    if len(selected) > keep:
        guaranteed: Set[int] = set()
        floor = max(1, int(min_per_query))
        for j in range(n_q):
            order = sorted(
                selected, key=lambda i: float(score_matrix[j][i]), reverse=True
            )
            for i in order[:floor]:
                guaranteed.add(i)
        if len(guaranteed) >= keep:
            selected = sorted(guaranteed, key=lambda i: s_max[i], reverse=True)[:keep]
        else:
            rest = [i for i in selected if i not in guaranteed]
            rest.sort(key=lambda i: s_max[i], reverse=True)
            selected = list(guaranteed)
            for i in rest:
                if len(selected) >= keep:
                    break
                selected.append(i)

    selected.sort(key=lambda i: s_max[i], reverse=True)
    return [anchors[i] for i in selected]


class AnchorCrossEncoder:
    """Shared BGE-reranker for raw-anchor multi-query coverage selection."""

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
        Score RRF candidates with CE(q_j, text(a)) for q_j in
        {original} ∪ planner queries; keep coverage-selected top-K.
        Writes s_max onto each returned hit as rerank_score.
        """
        if not anchors:
            return []
        top_n = int(keep if keep is not None else mem_cfg.anchor_ce_top_k)
        queries = _ce_query_set(original_query, planner_queries)
        if not queries or top_n <= 0:
            return anchors[:top_n]

        texts = [(a.content or "").strip() for a in anchors]
        try:
            score_matrix = [self._score_text_pairs(q, texts) for q in queries]
        except Exception:
            logger.warning("raw-anchor cross-encoder scoring failed", exc_info=True)
            return anchors[:top_n]

        if len(queries) == 1:
            scores = score_matrix[0]
            for i, anchor in enumerate(anchors):
                anchor.rerank_score = float(scores[i])
            order = sorted(
                range(len(anchors)),
                key=lambda i: float(scores[i]),
                reverse=True,
            )
            picked: List[RankedHit] = []
            seen_message_ids: Set[int] = set()
            for i in order:
                mid = anchors[i].message_id
                if mid in seen_message_ids:
                    continue
                picked.append(anchors[i])
                seen_message_ids.add(mid)
                if len(picked) >= top_n:
                    break
            return picked

        return _coverage_select_anchors(
            anchors,
            score_matrix,
            keep=top_n,
            orig_top=int(mem_cfg.anchor_ce_orig_top),
            sub_top=int(mem_cfg.anchor_ce_sub_top),
            min_per_query=int(mem_cfg.anchor_ce_min_per_query),
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
