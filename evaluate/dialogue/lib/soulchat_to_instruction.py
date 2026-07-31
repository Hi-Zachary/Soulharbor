from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def build_instruction_and_reference(
    *,
    topic: str,
    messages: List[Dict[str, Any]],
    max_turns: int = 4,
    system: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """
    Convert a multi-turn SoulChatCorpus example into a single-turn generation sample
    following SoulChat paper's prompt style:
      instruction: "用户：...\n心理咨询师：...\n用户：...\n心理咨询师："
      reference: the target assistant reply

    We take the last user->assistant pair within the last `max_turns` pairs (best-effort).
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    # Collect (user_idx, assistant_idx) pairs.
    pairs: List[Tuple[int, int]] = []
    for i in range(len(messages) - 1):
        mu = messages[i]
        ma = messages[i + 1]
        if not isinstance(mu, dict) or not isinstance(ma, dict):
            continue
        if str(mu.get("role") or "") != "user":
            continue
        if str(ma.get("role") or "") != "assistant":
            continue
        pairs.append((i, i + 1))

    if not pairs:
        return None

    # Take a late pair but keep context short.
    # We'll build prompt from a prefix ending at user_idx, limiting to last `max_turns` pairs.
    user_i, asst_i = pairs[-1]
    ref = str(messages[asst_i].get("content") or "").strip()
    if not ref:
        return None

    # Determine start pair to include up to max_turns.
    start_pair = max(0, len(pairs) - int(max_turns))
    start_user_i = pairs[start_pair][0]

    prompt_msgs = messages[start_user_i : user_i + 1]  # inclusive of target user

    lines: List[str] = []
    if system:
        lines.append(system.strip())
    for m in prompt_msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"用户：{content}")
        elif role == "assistant":
            lines.append(f"心理咨询师：{content}")

    lines.append("心理咨询师：")
    instruction = "\n".join(lines).strip()
    return instruction, ref
