"""Long-term profile side channel: strict LLM extract + LLM command parsing."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

from product_app.app.memory.models import ProfileItem
from product_app.app.memory.profile.policy import ProfilePolicy
from product_app.app.memory.profile.repository import ProfileStore
from product_app.app.memory.profile.schema import ProfileFact, normalize_fact

_INJECT_PROFILE_CAP = 30


class ProfileService:
    def __init__(self, db_path: str | Path) -> None:
        self._db = ProfileStore(db_path)
        self._policy = ProfilePolicy()

    def create_explicit(
        self, *, user_id: int, content: str, source_message_ids: Sequence[int]
    ) -> Optional[ProfileItem]:
        return self._create_content(
            user_id,
            content,
            "explicit",
            source_message_ids,
            source_messages=[content],
        )

    def create_confirmed(
        self, *, user_id: int, content: str, source_message_ids: Sequence[int]
    ) -> Optional[ProfileItem]:
        """Back-compat: treat as explicit write of structured content."""
        return self.create_explicit(
            user_id=user_id,
            content=content,
            source_message_ids=source_message_ids,
        )

    def create_fact(
        self,
        *,
        user_id: int,
        fact: ProfileFact,
        origin: str,
        source_message_ids: Sequence[int],
        source_messages: Sequence[str] | None = None,
    ) -> Optional[ProfileItem]:
        sources = list(source_messages) if source_messages else [fact.value]
        ok, _reason = self._policy.validate_fact(
            fact=fact, origin=origin, source_messages=sources
        )
        if not ok:
            return None
        self._db.soft_delete_by_key(user_id, fact.tag, fact.feature)
        return self._db.create(
            user_id=user_id,
            content=fact.to_content(),
            origin=origin,
            source_message_ids=source_message_ids,
        )

    def list_active(self, user_id: int) -> List[ProfileItem]:
        return self._db.list_active(user_id)

    def list_for_inject(
        self, user_id: int, limit: int = _INJECT_PROFILE_CAP
    ) -> List[ProfileItem]:
        return self.list_active(user_id)[: max(1, int(limit))]

    def delete(self, user_id: int, profile_id: str) -> bool:
        return self._db.soft_delete(user_id, profile_id)

    def forget_matching(self, user_id: int, keyword: str) -> int:
        return self._db.soft_delete_matching(user_id, keyword)

    def forget_key(self, user_id: int, tag: str, feature: str) -> int:
        return self._db.soft_delete_by_key(user_id, tag, feature)

    def update(self, user_id: int, profile_id: str, content: str) -> bool:
        return self._db.update_content(user_id, profile_id, content)

    def correct(
        self,
        *,
        user_id: int,
        old_keyword: str,
        new_content: str | None,
        source_message_ids: Sequence[int],
    ) -> Optional[ProfileItem]:
        self.forget_matching(user_id, old_keyword)
        replacement = (new_content or "").strip()
        if not replacement:
            return None
        return self.create_explicit(
            user_id=user_id,
            content=replacement,
            source_message_ids=source_message_ids,
        )

    def maybe_handle_user_command(
        self,
        *,
        user_id: int,
        user_text: str,
        source_message_id: int,
        llm: Any = None,
        parsed: dict | None = None,
    ) -> Optional[str]:
        """Apply remember/forget/correct/inspect from LLM parse (no keyword gates)."""
        if parsed is not None:
            cmd = parsed
        else:
            if llm is None:
                return None
            from product_app.app.memory.profile.commands_llm import parse_user_command

            cmd = parse_user_command(llm, user_text=user_text)
        action = cmd.get("action") or "none"
        if action == "none":
            return None

        if action == "inspect":
            return "profile_inspect"

        if action == "remember":
            fact = normalize_fact(
                tag=str(cmd.get("tag") or ""),
                feature=str(cmd.get("feature") or ""),
                value=str(cmd.get("value") or ""),
            )
            if not fact:
                return "profile_rejected"
            item = self.create_fact(
                user_id=user_id,
                fact=fact,
                origin="explicit",
                source_message_ids=[source_message_id],
                source_messages=[user_text],
            )
            return f"profile_saved:{item.id}" if item else "profile_rejected"

        if action == "forget":
            removed = 0
            tag, feature = str(cmd.get("tag") or ""), str(cmd.get("feature") or "")
            if tag and feature:
                removed += self.forget_key(user_id, tag, feature)
            query = str(cmd.get("query") or cmd.get("value") or "").strip()
            if query:
                removed += self.forget_matching(user_id, query)
            return f"profile_forgotten:{removed}"

        if action == "correct":
            fact = normalize_fact(
                tag=str(cmd.get("tag") or ""),
                feature=str(cmd.get("feature") or ""),
                value=str(cmd.get("value") or ""),
            )
            query = str(cmd.get("query") or "").strip()
            if query:
                self.forget_matching(user_id, query)
            if fact:
                self.forget_key(user_id, fact.tag, fact.feature)
            if not fact:
                return "profile_retracted" if query else "profile_rejected"
            item = self.create_fact(
                user_id=user_id,
                fact=fact,
                origin="explicit",
                source_message_ids=[source_message_id],
                source_messages=[user_text],
            )
            return f"profile_corrected:{item.id}" if item else "profile_rejected"

        return None

    def maybe_llm_extract(
        self,
        *,
        user_id: int,
        llm: Any,
        recent_turns: Sequence[dict],
        source_message_id: int,
        force: bool = False,
        max_items: int | None = None,
        trigger_messages: int | None = None,
        trigger_age_sec: int | None = None,
    ) -> Optional[str]:
        """LLM extracts allowlisted long-term facts → write active directly."""
        from product_app.app.memory.profile.extractor import (
            command_to_fact,
            evidence_supported,
            extract_from_recent,
            roughly_same_content,
            user_texts,
        )

        if (
            max_items is None
            or trigger_messages is None
            or trigger_age_sec is None
        ):
            from product_app.app.memory.config import mem_cfg
        else:
            mem_cfg = None  # unused when all overrides provided

        trig_msgs = (
            int(trigger_messages)
            if trigger_messages is not None
            else int(mem_cfg.profile_llm_trigger_messages)
        )
        trig_age = (
            int(trigger_age_sec)
            if trigger_age_sec is not None
            else int(mem_cfg.profile_llm_trigger_age_sec)
        )
        if not force and not self._db.should_attempt_llm_propose(
            user_id,
            trigger_messages=trig_msgs,
            trigger_age_sec=trig_age,
        ):
            return None

        existing_items = self.list_active(user_id)
        existing = [p.content for p in existing_items]
        limit = (
            int(max_items)
            if max_items is not None
            else max(1, int(mem_cfg.profile_llm_propose_max))
        )
        commands = extract_from_recent(
            llm,
            recent_turns=recent_turns,
            existing=existing,
            max_items=max(1, limit),
        )
        self._db.mark_llm_propose_attempted(user_id, source_message_id)
        if not commands:
            return None

        u_msgs = user_texts(recent_turns)
        added = 0
        deleted = 0
        for cmd in commands:
            if not evidence_supported(cmd, u_msgs):
                continue
            op = str(cmd.get("command") or "")
            if op == "delete":
                deleted += self.forget_key(
                    user_id, str(cmd.get("tag") or ""), str(cmd.get("feature") or "")
                )
                continue
            fact = command_to_fact(cmd)
            if not fact:
                continue
            content = fact.to_content()
            if any(roughly_same_content(content, e) for e in existing):
                continue
            item = self.create_fact(
                user_id=user_id,
                fact=fact,
                origin="extracted",
                source_message_ids=[source_message_id],
                source_messages=u_msgs or [fact.value],
            )
            if item:
                added += 1
                existing.append(content)
        if added or deleted:
            return f"profile_extracted:add={added},del={deleted}"
        return None

    def maybe_llm_propose(self, **kwargs: Any) -> Optional[str]:
        return self.maybe_llm_extract(**kwargs)

    def note_message_for_llm_propose(self, user_id: int) -> None:
        self._db.bump_llm_pending(user_id)

    def _create_content(
        self,
        user_id: int,
        content: str,
        origin: str,
        source_message_ids: Sequence[int],
        source_messages: Sequence[str] | None = None,
    ) -> Optional[ProfileItem]:
        sources = list(source_messages) if source_messages else [content]
        ok, _reason = self._policy.validate_content(
            content=content, origin=origin, source_messages=sources
        )
        if not ok:
            return None
        return self._db.create(
            user_id=user_id,
            content=content.strip(),
            origin=origin,
            source_message_ids=source_message_ids,
        )
