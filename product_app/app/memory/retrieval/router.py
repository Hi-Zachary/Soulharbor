"""Decide whether one query is enough, or we should split a comparison question."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import RetrievalPlan

logger = logging.getLogger(__name__)

_COMPARE = re.compile(
    r"(有什么不同|有何不同|相比|比较|对比|以前.+现在|现在.+以前|两次|两个阶段|过去.+现在|现在.+过去)"
)
_PURE_FEELING = re.compile(r"^(我好难过|好累|好烦|难受|想哭|心情不好)[。.!！]?$")


class QueryRouter:
    def __init__(self, llm: Any = None) -> None:
        self._llm = llm

    def set_llm(self, llm: Any) -> None:
        self._llm = llm

    def plan(self, query: str) -> RetrievalPlan:
        q = (query or "").strip()
        if not q:
            return RetrievalPlan(mode="direct", queries=[""])
        if not mem_cfg.split_query_enabled:
            return RetrievalPlan(mode="direct", queries=[q])
        if _PURE_FEELING.match(q) or not _COMPARE.search(q):
            return RetrievalPlan(mode="direct", queries=[q])

        if self._llm is not None:
            try:
                planned = self._ask_llm(q)
                if planned and planned.queries:
                    return planned
            except Exception:
                logger.warning("split-query LLM failed; using heuristic", exc_info=True)
        return self._fallback_split(q)

    def _fallback_split(self, q: str) -> RetrievalPlan:
        return RetrievalPlan(
            mode="split",
            queries=[
                f"当前情况：{q}",
                f"过去相关经历：{q}",
                "两次经历的差异与变化",
            ],
        )

    def _ask_llm(self, q: str) -> RetrievalPlan | None:
        prompt = [
            {
                "role": "user",
                "content": (
                    "判断检索模式。默认 direct。仅当问题明确包含多个对象、阶段或比较时用 split。"
                    "最多 3 个子 query。纯情绪表达必须 direct。\n"
                    '输出 JSON: {"mode":"direct|split","queries":["..."]}\n'
                    f"问题: {q}"
                ),
            }
        ]
        raw = self._llm.generate_structured(prompt, max_new_tokens=200)
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return None
        obj = json.loads(raw[start : end + 1])
        mode = str(obj.get("mode") or "direct")
        queries = [str(x).strip() for x in (obj.get("queries") or []) if str(x).strip()]
        if mode != "split" or not queries:
            return RetrievalPlan(mode="direct", queries=[q])
        return RetrievalPlan(mode="split", queries=queries[:3])
