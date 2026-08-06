"""Decide whether one query is enough, or we should split a comparison question.

No keyword / regex gates. When an LLM is available, ask it with a strict planner
prompt; otherwise stay on a single direct query (hybrid retrieval covers most cases).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import RetrievalPlan

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM = """\
你是 SoulHarbor 的记忆检索规划器。

请根据用户当前输入，生成用于检索其过往对话原文的查询。下游会结合语义检索与词法检索，因此查询应简短、自然，并保留人物、事件、时间、阶段或状态等关键锚点。

模式：

* direct：输入主要围绕一个检索目标。默认使用。
* split：输入包含多个彼此独立的检索目标，单个查询难以完整覆盖时使用。

判断原则：

* 一个核心话题、模糊表达或简单类比：direct
* 明确涉及多段经历、多个对象、多个阶段或比较关系：split
* 拿不准时：direct

查询要求：

* 每条只表达一个独立检索目标
* 查询数量以完整覆盖必要目标为准，最多 3 条，避免重复和过度拆分
* 保留关键锚点，删除寒暄和无关语气
* 不添加用户未提到的事实
* 不写成任务指令
* 不使用“检索、记忆、历史记录、上下文”等系统词
* 每条不超过 40 个汉字

只输出一行合法 JSON，不要解释或输出 Markdown：

{"mode":"direct","queries":["查询"]}
或
{"mode":"split","queries":["查询一","查询二","..."]}

示例：

用户：今天又被导师批评了，心里特别堵
输出：{"mode":"direct","queries":["导师批评后心里很堵"]}

用户：你还记得我之前说过保研的事吗
输出：{"mode":"direct","queries":["之前说过的保研情况"]}

用户：最近写论文时特别紧绷，让我想起之前找工作的时候
输出：{"mode":"split","queries":["最近写论文时很紧绷","之前找工作时的状态"]}

用户：毕业、家里的期待、和室友的关系最近都压得我喘不过气
输出：{"mode":"split","queries":["毕业带来的压力","家里的期待","最近和室友的关系"]}
"""


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

        if self._llm is not None:
            try:
                planned = self._ask_llm(q)
                if planned is not None:
                    return planned
            except Exception:
                logger.warning("split-query LLM failed; using direct", exc_info=True)

        return RetrievalPlan(mode="direct", queries=[q])

    def _ask_llm(self, q: str) -> RetrievalPlan | None:
        messages = [
            {
                "role": "user",
                "content": f"用户刚说：\n{q}\n\n输出检索规划 JSON：",
            }
        ]
        raw = self._llm.generate_structured(
            messages,
            max_new_tokens=256,
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
        queries = [str(x).strip() for x in (obj.get("queries") or []) if str(x).strip()]
        queries = [q[:80] for q in queries]

        if mode != "split" or len(queries) < 2:
            single = queries[0] if len(queries) == 1 else original
            return RetrievalPlan(mode="direct", queries=[single or original])

        uniq: list[str] = []
        for q in queries[:3]:
            if q and q not in uniq:
                uniq.append(q)
        if len(uniq) < 2:
            return RetrievalPlan(mode="direct", queries=[original])
        return RetrievalPlan(mode="split", queries=uniq)
