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
) -> str:
    return f"""\
你是 SoulHarbor 的长期用户画像维护器。

输入包括当前全部有效画像、本轮用户消息和最近对话。最近对话仅用于理解本轮用户消息。

请判断本轮用户消息是否需要新增、更新或删除长期画像。

画像会作为长期用户背景持续提供给后续对话模型，因此只保留用户明确表达、对未来跨会话交流持续有价值的信息。

画像应当：
- 简短、原子化并能够独立理解；
- 每条只描述一个事实；
- content 尽量不超过 {int(target_chars)} 个中文字符；
- 不与已有画像重复。

稳定背景、长期偏好、持续目标、重要关系、沟通方式和持续状态可以成为画像。

单次事件、临时安排、短暂情绪、普通聊天细节，以及助手的判断、建议或推测，不应成为画像。

操作规则：
- 新的独立长期信息使用 add。
- 新信息替代、纠正或明显细化已有画像时使用 update。
- 已有画像不再成立或应被移除时使用 delete。
- 可以同时成立的信息分别保存。
- 优先更新已有画像，不要新增含义相近的画像。
- 接近容量上限时，只保留对未来交流更有价值的画像。
- 没有必要修改时返回空 operations。
- 每轮最多输出 {int(max_operations)} 个操作。

add 的 target_id 必须为空。
update 和 delete 必须使用输入中的准确画像 id。
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
PROFILE_MAINTAINER_SYSTEM = build_maintainer_system(target_chars=40, max_operations=3)


def build_profile_maintainer_payload(
    *,
    profiles: Sequence[ProfileItem],
    recent_turns: Sequence[dict],
    current_user_message_id: int,
    max_active: int,
    block_tokens: int,
    max_block_tokens: int,
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

    payload = {
        "profile_capacity": {
            "active_count": len(profiles),
            "max_active": int(max_active),
            "current_tokens": int(block_tokens),
            "max_tokens": int(max_block_tokens),
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
    block_tokens: int,
    max_block_tokens: int,
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
            block_tokens=block_tokens,
            max_block_tokens=max_block_tokens,
            token_counter=token_counter,
        )
    except ValueError:
        logger.warning("profile maintainer payload missing current user message")
        return ProfileDecision(operations=[])
    system_text = build_maintainer_system(
        target_chars=int(target_chars),
        max_operations=int(max_operations),
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
