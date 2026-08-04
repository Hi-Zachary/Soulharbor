"""LLM extractor for strict long-term profile facts (direct write, empty OK)."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from product_app.app.memory.profile.schema import (
    ProfileFact,
    allowed_prompt_block,
    is_allowed,
    normalize_fact,
)

logger = logging.getLogger(__name__)

EXTRACTOR_SYSTEM = f"""\
你是 SoulHarbor 的长期画像抽取助手。

任务：从最近对话中，抽出用户明确说过的、跨会话仍成立的长期信息。
只允许下列 tag/feature（白名单之外一律不抽）：
{allowed_prompt_block()}

规则：
1. 必须有用户自己明确说过的依据；禁止从助理话术或氛围推断。
2. 禁止诊断、人格标签、瞬时情绪、单次事件、猜测。
3. 不确定、证据不足、或只是闲聊 → 返回空命令列表。
4. value 用简短中文（≤40 字），原子、可核对；evidence 须是摘自用户原话的短引语。
5. 同一 tag/feature 最多一条；已有相同事实勿重复 add。
6. 仅当用户明确否定或要求忘掉某条长期信息时才用 delete。

只输出一行 JSON：
{{"commands":[{{"command":"add|delete","tag":"...","feature":"...","value":"...","evidence":"..."}}]}}
无内容时：{{"commands":[]}}
"""


def user_texts(recent_turns: Sequence[dict]) -> List[str]:
    out: List[str] = []
    for t in recent_turns[-8:]:
        if str(t.get("role") or "") != "user":
            continue
        c = str(t.get("content") or "").strip()
        if c:
            out.append(c)
    return out


def extract_from_recent(
    llm: Any,
    *,
    recent_turns: Sequence[dict],
    existing: Sequence[str],
    max_items: int = 3,
) -> List[Dict[str, str]]:
    """Return schema-filtered command dicts."""
    if llm is None or not recent_turns:
        return []
    lines = []
    for t in recent_turns[-8:]:
        role = str(t.get("role") or "")
        content = str(t.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            lines.append(f"{role}: {content[:400]}")
    if not lines:
        return []
    existing_txt = (
        "；".join(str(x).strip() for x in existing if str(x).strip())[:600] or "（无）"
    )
    user_prompt = (
        f"已有画像（勿重复 add）：{existing_txt}\n\n"
        f"最近对话：\n" + "\n".join(lines) + "\n\n"
        "仅输出白名单内的长期事实；没有则 commands 为空。"
    )
    try:
        raw = llm.generate_structured(
            [{"role": "user", "content": user_prompt}],
            max_new_tokens=384,
            system_text=EXTRACTOR_SYSTEM,
        )
    except Exception:
        logger.warning("profile extractor LLM failed", exc_info=True)
        return []
    return _parse_commands(raw, max_items=max(1, int(max_items)))


def _parse_commands(raw: str, *, max_items: int) -> List[Dict[str, str]]:
    text = (raw or "").strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return []
    cmds = obj.get("commands") if isinstance(obj, dict) else None
    if not isinstance(cmds, list):
        return []
    out: List[Dict[str, str]] = []
    for c in cmds:
        if not isinstance(c, dict):
            continue
        command = str(c.get("command") or "").strip().lower()
        if command not in ("add", "delete"):
            continue
        tag = str(c.get("tag") or "").strip().lower()
        feature = str(c.get("feature") or "").strip().lower()
        value = str(c.get("value") or "").strip()
        evidence = str(c.get("evidence") or "").strip()
        if not is_allowed(tag, feature):
            continue
        if command == "add":
            fact = normalize_fact(tag=tag, feature=feature, value=value)
            if not fact:
                continue
            out.append(
                {
                    "command": "add",
                    "tag": fact.tag,
                    "feature": fact.feature,
                    "value": fact.value,
                    "evidence": evidence[:120],
                }
            )
        else:
            out.append(
                {
                    "command": "delete",
                    "tag": tag,
                    "feature": feature,
                    "value": value[:80],
                    "evidence": evidence[:120],
                }
            )
        if len(out) >= max_items:
            break
    return out


def command_to_fact(cmd: Dict[str, str]) -> Optional[ProfileFact]:
    return normalize_fact(
        tag=str(cmd.get("tag") or ""),
        feature=str(cmd.get("feature") or ""),
        value=str(cmd.get("value") or ""),
    )


def evidence_supported(cmd: Dict[str, str], user_msgs: Sequence[str]) -> bool:
    if str(cmd.get("command") or "") == "delete":
        return True
    if not user_msgs:
        return False
    joined = "\n".join(user_msgs)
    ev = (cmd.get("evidence") or "").strip()
    if len(ev) >= 2 and ev in joined:
        return True
    value = (cmd.get("value") or "").strip()
    if not value:
        return False
    overlap = sum(1 for ch in value if ch in joined)
    return overlap >= max(1, len(value) // 6)


def roughly_same_content(a: str, b: str) -> bool:
    x = re.sub(r"\s+", "", (a or "").lower())
    y = re.sub(r"\s+", "", (b or "").lower())
    if not x or not y:
        return False
    return x == y or x in y or y in x
