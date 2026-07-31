"""Score windows before coverage selection (features first, optional LLM)."""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, List, Tuple

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import EpisodeWindow

logger = logging.getLogger(__name__)

_CHARS = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")


def _char_set(text: str) -> set[str]:
    return set(_CHARS.findall(text or ""))


class WindowReranker:
    def __init__(self, llm: Any = None) -> None:
        self._llm = llm

    def set_llm(self, llm: Any) -> None:
        self._llm = llm

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

        if mem_cfg.llm_rerank_enabled and self._llm is not None and len(scored) > 1:
            try:
                llm_order = self._llm_order(query, [w for _, w in scored[: max(top_n, 8)]])
                if llm_order:
                    for i, window in enumerate(llm_order):
                        window.rerank_score = float(len(llm_order) - i)
                    return llm_order[:top_n]
            except Exception:
                logger.warning("LLM window rerank failed; using features", exc_info=True)

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

    def _llm_order(self, query: str, windows: List[EpisodeWindow]) -> List[EpisodeWindow]:
        by_id = {w.bundle_id: w for w in windows}
        lines = []
        for window in windows:
            snippet = " | ".join(
                t.content[:80] for t in window.messages if t.is_seed or t.role == "user"
            )[:240]
            lines.append(f"{window.bundle_id}: {snippet}")

        prompt = [
            {
                "role": "user",
                "content": (
                    "按与查询的相关性对下列证据包排序，只输出 bundle_id 列表 JSON。\n"
                    f"查询: {query}\n"
                    + "\n".join(lines)
                    + '\n输出: {"ids":["..."]}'
                ),
            }
        ]
        raw = self._llm.generate_structured(prompt, max_new_tokens=200)
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return []
        payload = json.loads(raw[start : end + 1])
        ordered = [by_id[i] for i in (payload.get("ids") or []) if i in by_id]
        for window in windows:
            if window not in ordered:
                ordered.append(window)
        return ordered
