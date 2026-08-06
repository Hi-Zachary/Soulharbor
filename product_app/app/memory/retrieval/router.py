"""Decide retrieval mode and role-scoped subqueries."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, List

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import RetrievalPlan, RoutedQuery

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM = """\
你是 SoulHarbor 的记忆检索规划器。

请根据用户当前输入，生成用于检索其过往对话原文的查询。每条子查询还需要判断主要信息来源角色：

* user：主要检索用户说过的事实、经历、偏好
* assistant：主要检索助手曾经给出的建议、解释、方案
* both：用户与助手双方内容都可能相关，或无法明确判断

模式：
* direct：一个检索目标即可
* split：多个彼此独立的检索目标

只输出一行合法 JSON，不要解释：

{"mode":"direct","original_role_scope":"both","subqueries":[{"query":"...","role_scope":"user"}]}
或
{"mode":"split","original_role_scope":"both","subqueries":[{"query":"...","role_scope":"assistant"},{"query":"...","role_scope":"user"}]}

role_scope 只能是 user / assistant / both。
每条 query 不超过 40 个汉字，最多 3 条。
"""


class QueryRouter:
    def __init__(self, llm: Any = None) -> None:
        self._llm = llm

    def set_llm(self, llm: Any) -> None:
        self._llm = llm

    def plan(self, query: str) -> RetrievalPlan:
        q = (query or "").strip()
        if not q:
            return RetrievalPlan(mode="direct", queries=[""], subqueries=[])

        if not mem_cfg.split_query_enabled:
            return RetrievalPlan(
                mode="direct",
                queries=[q],
                subqueries=[RoutedQuery(query=q, role_scope="both")],
                original_role_scope="both",
            )

        if self._llm is not None:
            try:
                planned = self._ask_llm(q)
                if planned is not None:
                    return planned
            except Exception:
                logger.warning("planner LLM failed; using direct", exc_info=True)

        return RetrievalPlan(
            mode="direct",
            queries=[q],
            subqueries=[RoutedQuery(query=q, role_scope=_guess_role_scope(q))],
            original_role_scope=_guess_role_scope(q),
        )

    def _ask_llm(self, q: str) -> RetrievalPlan | None:
        messages = [{"role": "user", "content": f"用户刚说：\n{q}\n\n输出检索规划 JSON："}]
        raw = self._llm.generate_structured(
            messages,
            max_new_tokens=320,
            system_text=_PLANNER_SYSTEM,
        )
        return self._parse_plan(q, raw)

    @staticmethod
    def _parse_plan(original: str, raw: str) -> RetrievalPlan | None:
        text = (raw or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

        mode = str(obj.get("mode") or "direct").strip().lower()
        original_scope = _normalize_scope(obj.get("original_role_scope"), fallback="both")
        subqueries_raw = obj.get("subqueries") or []
        subqueries: List[RoutedQuery] = []
        for item in subqueries_raw[:3]:
            if isinstance(item, str):
                q = item.strip()
                if q:
                    subqueries.append(RoutedQuery(query=q[:80], role_scope="both"))
                continue
            if not isinstance(item, dict):
                continue
            q = str(item.get("query") or "").strip()[:80]
            if not q:
                continue
            subqueries.append(
                RoutedQuery(
                    query=q,
                    role_scope=_normalize_scope(item.get("role_scope"), fallback="both"),
                )
            )

        if not subqueries:
            subqueries = [RoutedQuery(query=original, role_scope=original_scope)]

        queries = [sq.query for sq in subqueries]
        if mode != "split" or len(subqueries) < 2:
            single = subqueries[0]
            return RetrievalPlan(
                mode="direct",
                queries=[single.query or original],
                subqueries=[single],
                original_role_scope=original_scope,
            )

        uniq: List[RoutedQuery] = []
        seen = set()
        for sq in subqueries:
            key = (sq.query.casefold(), sq.role_scope)
            if sq.query and key not in seen:
                seen.add(key)
                uniq.append(sq)
        if len(uniq) < 2:
            return RetrievalPlan(
                mode="direct",
                queries=[original],
                subqueries=[RoutedQuery(query=original, role_scope=original_scope)],
                original_role_scope=original_scope,
            )
        return RetrievalPlan(
            mode="split",
            queries=[sq.query for sq in uniq],
            subqueries=uniq,
            original_role_scope=original_scope,
        )


def _normalize_scope(value: object, *, fallback: str) -> str:
    scope = str(value or fallback).strip().lower()
    return scope if scope in {"user", "assistant", "both"} else fallback


def _guess_role_scope(query: str) -> str:
    text = (query or "").strip()
    if re.search(r"助手|你(之前|曾经|当时)?(建议|说|提到|回答)", text):
        return "assistant"
    if re.search(r"我(之前|曾经|当时)?(说|提到|告诉)", text):
        return "user"
    return "both"
