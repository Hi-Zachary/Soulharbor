"""Select experience windows by CrossEncoder relevance (Top-k)."""
from __future__ import annotations

from typing import List

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import Span
from product_app.app.memory.store.window_key import earliest_ts


def sort_by_time(bundles: List[Span]) -> List[Span]:
    return sorted(bundles, key=lambda w: (earliest_ts(w), w.conversation_id, w.bundle_id))


def select_windows(
    *,
    bundles: List[Span],
    limit: int | None = None,
) -> List[Span]:
    """Keep the top-k windows by CE score (RRF fused_score as fallback)."""
    top_k = int(limit or mem_cfg.bundle_top_k)
    if not bundles or top_k <= 0:
        return []
    return sorted(
        bundles,
        key=lambda window: (
            float(window.rerank_score or 0.0),
            float(window.fused_score or 0.0),
        ),
        reverse=True,
    )[:top_k]
