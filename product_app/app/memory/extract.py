from __future__ import annotations

from typing import Dict, List

_SUMMARY_SYSTEM = (
    "你是会话摘要助手。根据已有摘要与新增对话，输出更新后的中文摘要（≤400字）。\n"
    "涵盖：主要议题、情绪、未说完的事。不要给建议。仅输出摘要正文。"
)


def build_summary_prompt(old_summary: str, new_dialog: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {
            "role": "user",
            "content": (
                f"旧摘要：\n{old_summary or '（无）'}\n\n"
                f"新增对话：\n{new_dialog}"
            ),
        },
    ]
