"""Helpers that turn stored messages into chat / classifier inputs."""
from __future__ import annotations

from typing import Dict, List

from product_app.app.db import StoredMessage


def messages_for_classifier(
    messages: List[Dict[str, str]],
    *,
    max_user_turns: int = 3,
) -> str:
    """
    Build the USER-only string used by the intent classifier.

    Keep at most `max_user_turns` recent user utterances (1 / 2 / 3 as the
    conversation grows), matching how the classifier was trained.
    """
    user_lines = [
        (m.get("content") or "").strip()
        for m in messages
        if (m.get("role") or "").strip() == "user" and (m.get("content") or "").strip()
    ]
    if not user_lines:
        return ""
    take = min(len(user_lines), max(1, int(max_user_turns)))
    return "\n".join(f"[USER] {line}" for line in user_lines[-take:]).strip()


def stored_to_chat_dicts(
    msgs: List[StoredMessage],
    *,
    recent_turns: int,
) -> List[Dict[str, str]]:
    window = msgs[-recent_turns:] if recent_turns > 0 else msgs
    out: List[Dict[str, str]] = []
    for msg in window:
        if msg.role not in ("user", "assistant"):
            continue
        content = msg.content or ""
        # Streaming inserts an empty assistant placeholder before generation;
        # feeding it back would make the model think the reply already started.
        if msg.role == "assistant" and not str(content).strip():
            continue
        out.append({"role": msg.role, "content": content})
    return out


def format_dialog_pair(user_text: str, assistant_text: str) -> str:
    return f"用户：{user_text}\n助手：{assistant_text}"


def format_messages_slice(msgs: List[StoredMessage], start_idx: int = 0) -> str:
    lines: List[str] = []
    for msg in msgs[start_idx:]:
        if msg.role == "user":
            lines.append(f"用户：{msg.content}")
        elif msg.role == "assistant":
            lines.append(f"助手：{msg.content}")
    return "\n".join(lines)
