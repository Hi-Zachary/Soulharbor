"""LLM long-term profile maintainer (content ops only; code owns safety)."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Sequence

from product_app.app.memory.models import ProfileItem
from product_app.app.memory.profile.operations import (
    ProfileDecision,
    ProfileOperation,
)
from product_app.app.memory.token_utils import TokenCounter, truncate_by_tokens

logger = logging.getLogger(__name__)


def build_maintainer_system(
    *,
    target_chars: int,
    max_operations: int,
    max_active: int,
) -> str:
    return f"""\
你是 SoulHarbor 的长期用户画像维护器。

输入包括当前全部有效画像、画像容量、本轮用户消息和最近对话。
最近对话仅用于理解本轮消息，不要从中重新抽取旧信息。

你的任务是维护少量、稳定、对未来跨会话交流持续有用的用户背景。
默认不修改画像；信息不够明确或拿不准时，返回空 operations。

每个用户最多保留 {int(max_active)} 条有效画像。
接近或达到上限时：
- 优先 update 已有画像，而不是新增相近画像；
- 新信息明显更重要时，可以 delete 已过时或长期价值较低的画像；
- 不要仅因为容量未满就新增画像；
- 没有适合删除或更新的画像时，可以不保存新信息。

适合写入的通常是：
- 基本身份：姓名、昵称、年龄、常住地；
- 教育与职业背景：学校、专业、学历、职业、长期工作状态；
- 重要关系：伴侣、家人、长期支持者及其关系；
- 稳定偏好：沟通方式、回答风格、语言、兴趣和生活偏好；
- 持续目标或约束：长期目标及持续存在的现实条件；
- 支持方式：用户明确表示长期偏好的求助、决策或压力应对方式。

只有用户本人明确表达，并且未来不同会话中仍有价值的信息才可写入。
以上是典型范围，不是固定字段；属于这些范围也不代表必须保存。

不要写入：
- 某一天、某一周或某一次发生的事件；
- 临时计划、短暂情绪、普通生活细节；
- 他人的经历、偏好或决定；
- 助手的建议，除非用户明确表示将长期采用；
- 根据单次行为推断出的性格、能力、心理状态或偏好；
- 诊断、评价、猜测或只因“可能有用”而保存的信息。

不要把单次行为概括为长期特征，也不要把临时尝试写成稳定习惯。

画像内容应当：
- 简短、原子化、能够独立理解；
- 每条只描述一个客观事实；
- content 尽量不超过 {int(target_chars)} 个中文字符；
- 不与已有画像重复或高度重合；
- 不使用“今天、最近、这周”等脱离上下文的表达。

操作规则：
- 新的独立长期信息使用 add。
- 新信息纠正、替代或明显细化已有画像时使用 update。
- 已有画像明确不再成立，或容量不足且长期价值较低时使用 delete。
- 优先 update，不要重复新增。
- 没有充分理由修改时返回空 operations。
- 每轮最多输出 {int(max_operations)} 个操作。

add 的 target_id 必须为空。
update 和 delete 必须使用输入中的准确画像 id。
update 的 content 是更新后的完整画像。
delete 的 content 必须为空。

只输出合法 JSON：

{{
  "operations": [
    {{
      "op": "add|update|delete",
      "target_id": "",
      "content": ""
    }}
  ]
}}
"""


# Default prompt kept for back-compat imports / tests.
PROFILE_MAINTAINER_SYSTEM = build_maintainer_system(
    target_chars=40,
    max_operations=3,
    max_active=20,
)


def build_profile_maintainer_payload(
    *,
    profiles: Sequence[ProfileItem],
    recent_turns: Sequence[dict],
    current_user_message_id: int,
    max_active: int,
    token_counter: TokenCounter = None,
) -> str:
    current_message = None
    previous_turns: list[dict] = []
    current_mid = int(current_user_message_id)

    for turn in recent_turns:
        try:
            message_id = int(turn["message_id"])
        except (KeyError, TypeError, ValueError):
            continue
        content = str(turn.get("content") or "")
        if not content:
            continue
        if message_id == current_mid:
            current_message = {
                "content": truncate_by_tokens(
                    content,
                    max_tokens=1200,
                    counter=token_counter,
                    keep_head_ratio=0.6,
                )
            }
            continue
        previous_turns.append(
            {
                "role": str(turn.get("role") or ""),
                "content": truncate_by_tokens(
                    content,
                    max_tokens=400,
                    counter=token_counter,
                    keep_head_ratio=0.6,
                ),
            }
        )

    if current_message is None:
        raise ValueError("current user message not found")

    active_count = len(profiles)
    payload = {
        "profile_capacity": {
            "active_count": active_count,
            "max_active": int(max_active),
            "remaining_slots": max(0, int(max_active) - active_count),
        },
        "current_profiles": [
            {"id": item.id, "content": item.content} for item in profiles
        ],
        "recent_context": previous_turns[-8:],
        "current_user_message": current_message,
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_profile_decision(raw: object) -> ProfileDecision:
    try:
        if isinstance(raw, dict):
            data = raw
        else:
            text = str(raw or "").strip()
            m = re.search(r"\{.*\}", text, flags=re.S)
            if not m:
                return ProfileDecision(operations=[])
            data = json.loads(m.group(0))
        rows = data.get("operations", []) if isinstance(data, dict) else []
        operations: list[ProfileOperation] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            op = str(row.get("op") or "").strip().lower()
            if op not in ("add", "update", "delete"):
                continue
            operations.append(
                ProfileOperation(
                    op=op,  # type: ignore[arg-type]
                    target_id=str(row.get("target_id") or ""),
                    content=str(row.get("content") or ""),
                )
            )
        return ProfileDecision(operations=operations)
    except Exception:
        logger.warning("profile decision parse failed", exc_info=True)
        return ProfileDecision(operations=[])


def propose_operations(
    llm: Any,
    *,
    profiles: Sequence[ProfileItem],
    recent_turns: Sequence[dict],
    current_user_message_id: int,
    max_active: int,
    target_chars: int = 40,
    max_operations: int = 3,
    token_counter: TokenCounter = None,
) -> ProfileDecision:
    if llm is None:
        return ProfileDecision(operations=[])
    try:
        payload = build_profile_maintainer_payload(
            profiles=profiles,
            recent_turns=recent_turns,
            current_user_message_id=current_user_message_id,
            max_active=max_active,
            token_counter=token_counter,
        )
    except ValueError:
        logger.warning("profile maintainer payload missing current user message")
        return ProfileDecision(operations=[])
    system_text = build_maintainer_system(
        target_chars=int(target_chars),
        max_operations=int(max_operations),
        max_active=int(max_active),
    )
    try:
        raw = llm.generate_structured(
            [{"role": "user", "content": payload}],
            max_new_tokens=512,
            system_text=system_text,
        )
    except Exception:
        logger.warning("profile maintainer LLM failed", exc_info=True)
        return ProfileDecision(operations=[])
    return parse_profile_decision(raw)
