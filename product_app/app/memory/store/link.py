"""Group related windows across sessions into a short timeline chain."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.embeddings import MemoryEmbedder
from product_app.app.memory.models import Span
from product_app.app.memory.store.text_sim import blob_of, cosine, entities, jaccard, tokens


def _earliest(window: Span) -> int:
    if not window.messages:
        return 0
    return min(int(t.created_at) for t in window.messages)


def _latest(window: Span) -> int:
    if not window.messages:
        return 0
    return max(int(t.created_at) for t in window.messages)


def _link_strength(
    left: Span,
    right: Span,
    *,
    query: str,
    left_vec: List[float],
    right_vec: List[float],
) -> float:
    left_text = blob_of(left.messages)
    right_text = blob_of(right.messages)
    entity_score = jaccard(entities(left_text), entities(right_text))
    semantic_score = cosine(left_vec, right_vec)

    gap = _earliest(right) - _latest(left)
    if gap >= 0:
        # Same chronological order still helps, but far-apart events decay
        # (≈0.35 at 0d → ≈0.12 at 90d → ≈0.05 floor by ~180d).
        days = gap / 86400.0
        time_score = max(0.05, 0.35 * math.exp(-days / 90.0))
    elif abs(gap) < 3 * 86400:
        time_score = 0.15
    else:
        time_score = 0.0

    q_terms = tokens(query)
    already = tokens(left_text)
    gain = len((tokens(right_text) - already) & q_terms) / max(1, len(q_terms))
    same_session = 0.1 if left.conversation_id == right.conversation_id else 0.0

    return (
        0.30 * entity_score
        + 0.30 * semantic_score
        + 0.20 * time_score
        + 0.15 * gain
        + same_session
    )


def link_windows(
    *,
    query: str,
    bundles: List[Span],
    embedder: Optional[MemoryEmbedder] = None,
) -> Tuple[List[Span], int]:
    """Annotate windows with chain_id / chain_index. Returns (windows, chain_count)."""
    windows = bundles
    if not mem_cfg.cross_session_linking or len(windows) <= 1:
        for i, window in enumerate(windows):
            window.chain_id = None
            window.chain_index = i
        return windows, 0

    model = embedder or MemoryEmbedder.shared()
    vectors = [list(model.embed(blob_of(w.messages)) or []) for w in windows]
    threshold = float(mem_cfg.link_score_threshold)

    order = sorted(
        range(len(windows)),
        key=lambda i: float(windows[i].fused_score or 0.0),
        reverse=True,
    )
    used: set[int] = set()
    chains: List[List[int]] = []

    for start_idx in order:
        if start_idx in used:
            continue
        chain = [start_idx]
        used.add(start_idx)

        growing = True
        while growing:
            growing = False
            tip = chain[-1]
            best_idx = None
            best_score = threshold
            for cand in order:
                if cand in used:
                    continue
                # skip events that are way earlier than the tip
                if _earliest(windows[cand]) + 86400 < _earliest(windows[tip]) - 14 * 86400:
                    continue
                score = _link_strength(
                    windows[tip],
                    windows[cand],
                    query=query,
                    left_vec=vectors[tip],
                    right_vec=vectors[cand],
                )
                if score > best_score:
                    best_score = score
                    best_idx = cand
            if best_idx is not None:
                chain.append(best_idx)
                used.add(best_idx)
                growing = True
        chains.append(chain)

    chains.sort(key=lambda ch: min(_earliest(windows[i]) for i in ch))
    result: List[Span] = []
    for chain_no, chain in enumerate(chains, start=1):
        ordered = sorted(chain, key=lambda i: _earliest(windows[i]))
        label = f"E{chain_no}" if len(chains) > 1 or len(ordered) > 1 else None
        for idx, i in enumerate(ordered):
            windows[i].chain_id = label
            windows[i].chain_index = idx
            result.append(windows[i])
    return result, len(chains)
