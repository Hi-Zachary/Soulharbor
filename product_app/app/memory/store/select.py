"""Pick a small set of windows that cover the query without too much overlap."""
from __future__ import annotations

from typing import List, Set

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import Span
from product_app.app.memory.store.text_sim import entities, jaccard, tokens


def _relevance(query: str, window: Span) -> float:
    q = tokens(query)
    text = "\n".join(t.content for t in window.messages)
    overlap = jaccard(q, tokens(text))
    fused = float(window.fused_score or 0.0)
    rerank = float(window.rerank_score if window.rerank_score is not None else 0.0)
    return fused * 2.0 + rerank * 0.5 + overlap * 1.5


def _week_bucket(window: Span, *, seconds: int = 7 * 86400) -> int:
    if not window.messages:
        return 0
    earliest = min(t.created_at for t in window.messages)
    return int(earliest) // int(seconds)


def _content_tokens(window: Span) -> Set[str]:
    text = "\n".join(t.content for t in window.messages)
    return tokens(text) | entities(text)


def select_windows(
    *,
    query: str,
    bundles: List[Span],
    limit: int | None = None,
) -> List[Span]:
    """
    Default mode (`coverage`): greedy pick for relevance + new info − redundancy.
    `topk` mode: just take the first N windows (already ranked upstream).
    """
    mode = (mem_cfg.evidence_selection_mode or "coverage").lower()
    limit = int(limit or mem_cfg.bundle_top_k)
    if not bundles:
        return []
    if mode != "coverage":
        return bundles[:limit]

    remaining = list(bundles)
    chosen: List[Span] = []
    covered: Set[str] = set()
    seen_weeks: Set[int] = set()
    query_tokens = tokens(query)

    while remaining and len(chosen) < limit:
        best_idx = -1
        best_score = -1e9
        for idx, window in enumerate(remaining):
            rel = min(_relevance(query, window), 2.2)
            cov = _content_tokens(window)
            novelty = 1.0 if not covered else len(cov - covered) / max(1, len(cov))
            query_gain = len((cov - covered) & query_tokens) / max(1, len(query_tokens))
            week = _week_bucket(window)
            time_bonus = 0.9 if week not in seen_weeks else -0.15
            redundancy = jaccard(cov, covered) if covered else 0.0
            score = rel + 1.6 * novelty + 1.0 * query_gain + time_bonus - 1.8 * redundancy
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break
        if chosen and best_score < 0.05:
            break

        pick = remaining.pop(best_idx)
        chosen.append(pick)
        covered |= _content_tokens(pick)
        seen_weeks.add(_week_bucket(pick))

    return chosen
