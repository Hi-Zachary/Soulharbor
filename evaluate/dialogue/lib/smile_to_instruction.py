from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def build_instruction_and_reference(
    *,
    turns: List[Dict[str, Any]],
    max_turns: int = 5,
) -> Optional[Tuple[str, str]]:
    """
    Convert a SMILECHAT (smile-main) multi-turn dialogue into a single-turn sample:

      instruction: "用户：...\n心理咨询师：...\n...\n心理咨询师："
      reference: last counselor reply

    Roles in smile-main:
      - client -> user
      - counselor -> assistant
    """
    if not isinstance(turns, list) or len(turns) < 2:
        return None

    pairs: List[Tuple[int, int]] = []
    for i in range(len(turns) - 1):
        a = turns[i]
        b = turns[i + 1]
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        if str(a.get("role") or "") != "client":
            continue
        if str(b.get("role") or "") != "counselor":
            continue
        pairs.append((i, i + 1))
    if not pairs:
        return None

    user_i, asst_i = pairs[-1]
    ref = str(turns[asst_i].get("content") or "").strip()
    if not ref:
        return None

    start_pair = max(0, len(pairs) - int(max_turns))
    start_user_i = pairs[start_pair][0]
    prompt_turns = turns[start_user_i : user_i + 1]

    lines: List[str] = []
    for t in prompt_turns:
        if not isinstance(t, dict):
            continue
        role = str(t.get("role") or "")
        content = str(t.get("content") or "").strip()
        if not content:
            continue
        if role == "client":
            lines.append(f"用户：{content}")
        elif role == "counselor":
            lines.append(f"心理咨询师：{content}")
    lines.append("心理咨询师：")
    instruction = "\n".join(lines).strip()
    return instruction, ref

