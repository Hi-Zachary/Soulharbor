"""Safety checks before writing a profile item."""
from __future__ import annotations

import re
from typing import Sequence

_BLOCKED_CLINICAL = re.compile(
    r"(抑郁症|双相|人格障碍|自杀倾向|精神病|回避型人格|焦虑症患者|诊断为|创伤导致)"
)
_BLOCKED_LABEL = re.compile(r"(高焦虑个体|回避型|依恋类型|人格类型|病态)")
_FLEETING = re.compile(r"^(今天有点|刚才有点|突然觉得|这会儿)(难过|烦|累|开心)")


class ProfilePolicy:
    def validate(
        self,
        *,
        candidate: str,
        origin: str,
        source_messages: Sequence[str],
    ) -> tuple[bool, str]:
        text = (candidate or "").strip()
        if not text:
            return False, "empty"
        if origin not in ("explicit", "confirmed"):
            return False, "bad_origin"
        if not source_messages:
            return False, "no_source"
        if _BLOCKED_CLINICAL.search(text) or _BLOCKED_LABEL.search(text):
            return False, "diagnosis_or_sensitive"
        if _FLEETING.search(text):
            return False, "transient_emotion"
        if len(text) < 4:
            return False, "too_short"

        joined = "\n".join(source_messages)
        overlap = sum(1 for ch in text if ch in joined)
        if overlap < max(2, len(text) // 8):
            return False, "source_mismatch"
        return True, "ok"
