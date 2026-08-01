"""Chinese LLM profile proposer: candidates go to pending only (consent required)."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)

PROPOSER_SYSTEM = """\
你是 SoulHarbor 的「支持偏好」提议助手（校园心理陪伴）。

任务：从最近对话里，最多抽出 1 条稳定的支持偏好候选（沟通方式、边界、希望被如何回应、自我照顾方式）。

规则：
1. 必须有用户自己明确说过的依据；禁止从助理话术或氛围推断。
2. 禁止诊断词、人格标签、一次性情绪、人格化评判。
3. 不确定、证据不足、或只是闲聊 → 返回空列表。
4. content ≤ 40 字，原子、可确认；evidence 须是摘自用户原话的短引语。

只输出一行 JSON：
{"proposals":[{"content":"简短中文偏好","evidence":"摘自用户的短引语"}]}
无候选时：{"proposals":[]}
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


def propose_from_recent(
    llm: Any,
    *,
    recent_turns: Sequence[dict],
    existing: Sequence[str],
    max_items: int = 1,
) -> List[Dict[str, str]]:
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
    existing_txt = "；".join(str(x).strip() for x in existing if str(x).strip())[:500] or "（无）"
    user_prompt = (
        f"已有偏好/待确认（勿重复）：{existing_txt}\n\n"
        f"最近对话：\n" + "\n".join(lines) + "\n\n"
        "仅当用户明确表达了可长期遵守的支持偏好时输出；否则 proposals 为空。"
    )
    try:
        raw = llm.generate_structured(
            [{"role": "user", "content": user_prompt}],
            max_new_tokens=256,
            system_text=PROPOSER_SYSTEM,
        )
    except Exception:
        logger.warning("profile proposer LLM failed", exc_info=True)
        return []
    return _parse(raw, max_items=max(1, int(max_items)))


def _parse(raw: str, *, max_items: int = 1) -> List[Dict[str, str]]:
    text = (raw or "").strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return []
    props = obj.get("proposals") if isinstance(obj, dict) else None
    if not isinstance(props, list):
        return []
    out: List[Dict[str, str]] = []
    for p in props[:max_items]:
        if not isinstance(p, dict):
            continue
        content = str(p.get("content") or "").strip()
        evidence = str(p.get("evidence") or "").strip()
        if len(content) < 4:
            continue
        out.append({"content": content[:80], "evidence": evidence[:120]})
    return out


def roughly_same(a: str, b: str) -> bool:
    x = re.sub(r"\s+", "", (a or "").lower())
    y = re.sub(r"\s+", "", (b or "").lower())
    if not x or not y:
        return False
    if x == y or x in y or y in x:
        return True
    return False


def evidence_supported(proposal: Dict[str, str], user_msgs: Sequence[str]) -> bool:
    """Require overlap between proposal (or its evidence) and real user text."""
    if not user_msgs:
        return False
    joined = "\n".join(user_msgs)
    ev = (proposal.get("evidence") or "").strip()
    if len(ev) >= 2 and ev in joined:
        return True
    content = (proposal.get("content") or "").strip()
    if not content:
        return False
    overlap = sum(1 for ch in content if ch in joined)
    return overlap >= max(2, len(content) // 6)
