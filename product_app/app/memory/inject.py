from __future__ import annotations


def build_session_summary_block(summary: str) -> str:
    text = (summary or "").strip()
    if not text:
        return ""
    return (
        "<session-so-far>\n"
        "以下是本会话截至目前的关键摘要（可能比更早的原文更精炼）：\n"
        f"{text}\n"
        "</session-so-far>"
    )
