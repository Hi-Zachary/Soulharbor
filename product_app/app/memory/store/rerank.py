"""Score windows before coverage selection (feature scores only)."""
from __future__ import annotations

import math
import re
from typing import List, Tuple

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import EpisodeWindow

_CHARS = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")


def _char_set(text: str) -> set[str]:
    return set(_CHARS.findall(text or ""))


class WindowReranker:
    def rerank(
        self,
        *,
        query: str,
        bundles: List[EpisodeWindow],
        limit: int | None = None,
    ) -> List[EpisodeWindow]:
        windows = bundles
        top_n = int(limit or mem_cfg.bundle_top_k)
        if not windows:
            return []
        if not mem_cfg.rerank_enabled:
            return windows[:top_n]

        scored: List[Tuple[float, EpisodeWindow]] = [
            (self._score(query, window), window) for window in windows
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        result: List[EpisodeWindow] = []
        for score, window in scored[:top_n]:
            window.rerank_score = score
            result.append(window)

        # Keep a slightly larger pool when the cut would be too aggressive.
        if len(result) < min(4, len(windows)) and len(windows) >= 4:
            result = [w for _, w in scored[: min(6, len(scored))]]
            for i, window in enumerate(result):
                window.rerank_score = scored[i][0]
        return result

    def _score(self, query: str, window: EpisodeWindow) -> float:
        q = _char_set(query)
        text = "\n".join(t.content for t in window.messages)
        d = _char_set(text)
        overlap = len(q & d) / max(1, len(q))
        has_user = 1.0 if any(t.role == "user" for t in window.messages) else 0.0
        length_tax = math.log1p(len(window.messages)) * 0.05
        fused = float(window.fused_score or 0.0)
        return fused * 2.0 + overlap * 1.5 + has_user * 0.3 - length_tax
