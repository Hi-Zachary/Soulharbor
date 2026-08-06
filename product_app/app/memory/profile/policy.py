"""Structural checks before writing a profile fact (no keyword blocklists)."""
from __future__ import annotations

from typing import Sequence

from product_app.app.memory.profile.schema import ProfileFact, is_allowed, normalize_fact


class ProfilePolicy:
    def validate_fact(
        self,
        *,
        fact: ProfileFact,
        origin: str,
        source_messages: Sequence[str],
    ) -> tuple[bool, str]:
        if origin not in ("extracted", "explicit"):
            return False, "bad_origin"
        if not is_allowed(fact.tag, fact.feature):
            return False, "bad_slug"
        value = (fact.value or "").strip()
        if not value:
            return False, "empty"
        if len(value) < 1:
            return False, "too_short"
        if not source_messages:
            return False, "no_source"
        joined = "\n".join(source_messages)
        # Soft evidence: value chars should overlap user text a bit.
        overlap = sum(1 for ch in value if ch in joined)
        if overlap < max(1, len(value) // 8):
            return False, "source_mismatch"
        return True, "ok"

    def validate_content(
        self,
        *,
        content: str,
        origin: str,
        source_messages: Sequence[str],
    ) -> tuple[bool, str]:
        """Legacy free-text path: accept structured `[tag/feature] …` facts."""
        text = (content or "").strip()
        if not text:
            return False, "empty"
        from product_app.app.memory.profile.schema import parse_content_key

        key = parse_content_key(text)
        if not key:
            return False, "not_structured"
        tag, feature = key
        value = text
        for sep in ("：", ":"):
            if sep in text:
                value = text.split(sep, 1)[-1].strip()
                break
        fact = normalize_fact(tag=tag, feature=feature, value=value)
        if not fact:
            return False, "bad_fact"
        return self.validate_fact(
            fact=fact, origin=origin, source_messages=source_messages or [value]
        )
