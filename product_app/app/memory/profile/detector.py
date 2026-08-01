"""Heuristic detectors for remember / forget / confirm / inspect / correct phrases."""
from __future__ import annotations

import re
from typing import Optional, Tuple

_REMEMBER = re.compile(r"(记住|以后请|下次可以|长期记得|别忘了|请记得|可以长期记着)")
_FORGET = re.compile(r"(忘掉|不要记|删掉|别再提|忘记|不要再记住)")
_YES = re.compile(r"^(可以|对|好的|以后就这样|这样挺好|记得吧|嗯可以)([。.!！]?)$")
_LOOK_UP = re.compile(r"(你记得我什么|你都记住了什么|我的记忆|查看记忆)")
_FIX = re.compile(r"(我现在不|以后不要|改成|更正一下)")
_OFFER = re.compile(r"(要不要我记住|我可以记住|需要我长期记住)")


class ProfileDetector:
    def detect_explicit(self, user_text: str) -> Optional[str]:
        text = (user_text or "").strip()
        if not text or not _REMEMBER.search(text):
            return None
        match = re.search(
            r"(?:记住|以后请|下次可以|长期记得|别忘了|请记得|可以长期记着)[，,:：]?\s*(.+)$",
            text,
        )
        content = (match.group(1) if match else text).strip().strip("。.!！")
        return content or None

    def detect_forget(self, user_text: str) -> Optional[str]:
        text = (user_text or "").strip()
        if not text or not _FORGET.search(text):
            return None
        match = re.search(
            r"(?:忘掉|不要记|删掉|别再提|忘记|不要再记住)[，,:：]?\s*(?:我)?(.+)$",
            text,
        )
        if match:
            return match.group(1).strip().strip("。.!！") or text
        return text

    def detect_confirm(self, user_text: str) -> bool:
        return bool(_YES.match((user_text or "").strip()))

    def detect_inspect(self, user_text: str) -> bool:
        return bool(_LOOK_UP.search(user_text or ""))

    def detect_correct(self, user_text: str) -> Optional[Tuple[str, str]]:
        """
        Parse a correction utterance into (old_keyword, new_content).
        new_content may be "" when the user only retracts an old preference.
        """
        text = (user_text or "").strip()
        if not text or not _FIX.search(text):
            return None

        # 「旧说法改成新说法」
        match = re.search(
            r"(.+?)(?:改成|更正(?:一下)?(?:为|成)?)[，,:：]?\s*(.+)$",
            text,
        )
        if match:
            old = match.group(1).strip().strip("，,。 ")
            old = re.sub(
                r"^(?:我现在不|以后不要|不要再记)[要再]*",
                "",
                old,
            ).strip()
            new = match.group(2).strip().strip("。.!！")
            if old and new:
                return old, new

        # 「我现在不 / 以后不要 X」（只删旧的）
        match = re.search(
            r"(?:我现在不|以后不要|不要再记)[要再]*[，,:：]?\s*(.+)$",
            text,
        )
        if match:
            old = match.group(1).strip().strip("。.!！")
            if old:
                return old, ""
        return None

    def detect_propose_in_assistant(self, assistant_text: str) -> Optional[str]:
        text = (assistant_text or "").strip()
        if not text or not _OFFER.search(text):
            return None
        match = re.search(
            r"(?:要不要我记住|我可以记住|需要我长期记住)[，,:：]?\s*(.+?)(?:[？?]。?|$)",
            text,
        )
        if match:
            return match.group(1).strip().strip("。.!！")
        return None
