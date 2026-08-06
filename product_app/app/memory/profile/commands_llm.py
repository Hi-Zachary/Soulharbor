"""LLM parser for explicit profile user intents (no keyword/regex gates)."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from product_app.app.memory.profile.schema import example_prompt_block, normalize_fact

logger = logging.getLogger(__name__)

COMMAND_SYSTEM = f"""\
你是 SoulHarbor 的画像指令解析助手。

根据用户本轮话，判断是否在主动操作长期画像。
{example_prompt_block()}

可能的 action：
- remember：用户明确要求记住某条可确认的长期信息（给出 tag/feature/value）
- forget：用户明确要求忘掉某条信息（给出 query 匹配子串，或 tag+feature）
- correct：用户要求把旧信息改成新信息
- inspect：用户要查看当前记住了什么
- none：普通聊天，不是画像指令

规则：
1. 禁止写入诊断、人格标签、瞬时情绪、单次事件。
2. remember/correct 必须有可核对的长期 value；拿不准 → none。
3. 不要把一时情绪或单次事件写成长期画像。

只输出一行 JSON，例如：
{{"action":"remember","tag":"identity","feature":"name","value":"小王","query":""}}
{{"action":"forget","tag":"","feature":"","value":"","query":"姓名"}}
{{"action":"correct","tag":"preference","feature":"hobby","value":"跑步","query":"篮球"}}
{{"action":"inspect"}}
{{"action":"none"}}
"""


def parse_user_command(llm: Any, *, user_text: str) -> Dict[str, str]:
    """Return a normalized command dict; default action=none."""
    text = (user_text or "").strip()
    if not text or llm is None:
        return {"action": "none"}
    try:
        raw = llm.generate_structured(
            [{"role": "user", "content": f"用户本轮：{text[:500]}"}],
            max_new_tokens=192,
            system_text=COMMAND_SYSTEM,
        )
    except Exception:
        logger.warning("profile command LLM failed", exc_info=True)
        return {"action": "none"}
    return _parse(raw)


def _parse(raw: str) -> Dict[str, str]:
    text = (raw or "").strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {"action": "none"}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return {"action": "none"}
    if not isinstance(obj, dict):
        return {"action": "none"}
    action = str(obj.get("action") or "none").strip().lower()
    if action not in ("remember", "forget", "correct", "inspect", "none"):
        action = "none"
    tag = str(obj.get("tag") or "").strip().lower()
    feature = str(obj.get("feature") or "").strip().lower()
    value = str(obj.get("value") or "").strip()
    query = str(obj.get("query") or "").strip()
    if action in ("remember", "correct"):
        fact = normalize_fact(tag=tag, feature=feature, value=value)
        if not fact:
            return {"action": "none"}
        return {
            "action": action,
            "tag": fact.tag,
            "feature": fact.feature,
            "value": fact.value,
            "query": query[:80],
        }
    return {
        "action": action,
        "tag": tag,
        "feature": feature,
        "value": value[:80],
        "query": query[:80],
    }


def command_fact(cmd: Dict[str, str]) -> Optional[object]:
    return normalize_fact(
        tag=str(cmd.get("tag") or ""),
        feature=str(cmd.get("feature") or ""),
        value=str(cmd.get("value") or ""),
    )
