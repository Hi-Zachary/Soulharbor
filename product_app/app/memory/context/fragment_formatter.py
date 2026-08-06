"""Render retrieved fragments for episodic memory injection."""
from __future__ import annotations

from html import escape
from typing import List

from product_app.app.memory.models import ProfileItem, RetrievedFragment, Span
from product_app.app.memory.context.formatter import current_date_label


_EPISODIC_HEADER = (
    "以下内容是从历史对话中检索出的独立片段。\n"
    "“核心命中”表示检索系统直接选中的内容；"
    "“相邻补充”用于补足核心内容的局部语义；"
    "“回复语境”表示助手消息当时直接回应的用户表达。\n"
    "Message 级结果可能附带较早或后续的相关用户消息。"
    "Segment 级结果只展示长消息中的部分连续内容，省略号表示原消息仍有未展示部分。\n"
    "不同记忆片段之间不一定连续，不要推测未展示的中间内容。"
    "“核心命中”只表示检索来源，不表示它比其他内容更真实或更新。"
)


def format_segment_region(
    *,
    core_contents: List[str],
    expanded_before: List[str],
    expanded_after: List[str],
    omitted_before: bool,
    omitted_after: bool,
) -> str:
    parts: List[str] = []
    if omitted_before:
        parts.append("……（前文省略）")
    if expanded_before:
        parts.append("【相邻补充】")
        parts.extend(expanded_before)
    parts.append("【核心命中】")
    parts.extend(core_contents)
    if expanded_after:
        parts.append("【相邻补充】")
        parts.extend(expanded_after)
    if omitted_after:
        parts.append("……（后文省略）")
    return "\n".join(parts)


def _format_message_fragment(index: int, frag: RetrievedFragment) -> List[str]:
    role = frag.anchor_role
    lines = [f'<memory_fragment index="{index}" type="message" role="{role}">']

    if frag.reply_context_content and role == "assistant":
        lines.append("[该助手消息的回复语境]")
        lines.append(f"用户：{escape(frag.reply_context_content, quote=False)}")

    for content in frag.earlier_user_contents:
        lines.append("[较早的相关用户消息｜扩展]")
        lines.append(f"用户：{escape(content, quote=False)}")

    label = "用户" if role == "user" else "助手"
    lines.append(f"[核心命中的{label}消息｜完整]")
    lines.append(f"{label}：{escape(frag.core_message_content or frag.core_contents[0], quote=False)}")

    for content in frag.later_user_contents:
        lines.append("[后续的相关用户消息｜扩展]")
        lines.append(f"用户：{escape(content, quote=False)}")

    lines.append("</memory_fragment>")
    return lines


def _format_segment_fragment(index: int, frag: RetrievedFragment) -> List[str]:
    role = frag.anchor_role
    lines = [f'<memory_fragment index="{index}" type="segment" role="{role}">']

    if frag.reply_context_content and role == "assistant":
        lines.append("[该助手回复所回应的用户消息]")
        lines.append(f"用户：{escape(frag.reply_context_content, quote=False)}")

    label = "用户" if role == "user" else "助手"
    lines.append(f"[核心命中的{label}消息节选]")
    lines.append("原消息较长，以下仅展示部分连续内容：")
    region = frag.segment_region
    expanded_before = frag.expanded_contents[: len(region.before_segment_ids)] if region else []
    expanded_after = frag.expanded_contents[len(region.before_segment_ids) :] if region else []
    body = format_segment_region(
        core_contents=frag.core_contents,
        expanded_before=expanded_before,
        expanded_after=expanded_after,
        omitted_before=bool(frag.omitted_before),
        omitted_after=bool(frag.omitted_after),
    )
    lines.append(escape(body, quote=False))
    lines.append("</memory_fragment>")
    return lines


def format_fragment_sections(
    *,
    bundles: List[Span],
    profiles: List[ProfileItem],
) -> List[str]:
    del profiles
    fragments = [b.fragment for b in bundles if b.fragment is not None]
    if not fragments:
        return []

    ordered = sorted(
        fragments,
        key=lambda f: (int(f.created_at), int(f.conversation_id), int(f.parent_message_id)),
    )
    parts: List[str] = [f"当前日期：{current_date_label()}。", _EPISODIC_HEADER]
    for idx, frag in enumerate(ordered, start=1):
        parts.append("")
        if frag.fragment_type == "segment":
            parts.extend(_format_segment_fragment(idx, frag))
        else:
            parts.extend(_format_message_fragment(idx, frag))
    while parts and parts[-1] == "":
        parts.pop()
    return parts
