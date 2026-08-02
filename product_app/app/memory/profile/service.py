"""Consent-based profile side channel on top of the trace store."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

from product_app.app.memory.models import ProfileItem
from product_app.app.memory.profile.detector import ProfileDetector
from product_app.app.memory.profile.policy import ProfilePolicy
from product_app.app.memory.profile.repository import ProfileStore

# Soft cap when injecting all active preferences into <memory>.
_INJECT_PROFILE_CAP = 30


class ProfileService:
    def __init__(self, db_path: str | Path) -> None:
        self._db = ProfileStore(db_path)
        self._policy = ProfilePolicy()
        self._detector = ProfileDetector()

    @property
    def detector(self) -> ProfileDetector:
        return self._detector

    def create_explicit(
        self, *, user_id: int, content: str, source_message_ids: Sequence[int]
    ) -> Optional[ProfileItem]:
        return self._create(user_id, content, "explicit", source_message_ids)

    def create_confirmed(
        self, *, user_id: int, content: str, source_message_ids: Sequence[int]
    ) -> Optional[ProfileItem]:
        return self._create(user_id, content, "confirmed", source_message_ids)

    def propose(
        self,
        *,
        user_id: int,
        content: str,
        source_message_ids: Sequence[int],
        source_messages: Sequence[str] | None = None,
    ) -> None:
        # Pending uses the same safety checks as confirmed; overlap vs real user text
        # when provided (LLM path). Regex offer still passes content-as-source.
        sources = list(source_messages) if source_messages else [content]
        ok, _reason = self._policy.validate(
            candidate=content, origin="confirmed", source_messages=sources
        )
        if ok:
            self._db.add_pending(
                user_id=user_id, content=content, source_message_ids=source_message_ids
            )

    def list_active(self, user_id: int) -> List[ProfileItem]:
        return self._db.list_active(user_id)

    def list_for_inject(
        self, user_id: int, limit: int = _INJECT_PROFILE_CAP
    ) -> List[ProfileItem]:
        """All active preferences for prompt injection (small set by design)."""
        return self.list_active(user_id)[: max(1, int(limit))]

    def delete(self, user_id: int, profile_id: str) -> bool:
        return self._db.soft_delete(user_id, profile_id)

    def forget_matching(self, user_id: int, keyword: str) -> int:
        return self._db.soft_delete_matching(user_id, keyword)

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
        """Drop prefs matching old_keyword; optionally write a replacement."""
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
        self, *, user_id: int, user_text: str, source_message_id: int
    ) -> Optional[str]:
        remembered = self._detector.detect_explicit(user_text)
        if remembered:
            item = self.create_explicit(
                user_id=user_id,
                content=remembered,
                source_message_ids=[source_message_id],
            )
            return f"profile_saved:{item.id}" if item else "profile_rejected"

        correction = self._detector.detect_correct(user_text)
        if correction:
            old_keyword, new_content = correction
            item = self.correct(
                user_id=user_id,
                old_keyword=old_keyword,
                new_content=new_content or None,
                source_message_ids=[source_message_id],
            )
            if new_content:
                return f"profile_corrected:{item.id}" if item else "profile_rejected"
            return "profile_retracted"

        if self._detector.detect_confirm(user_text):
            # One "可以" confirms every unanswered assistant proposal.
            pending_items = self._db.pop_all_pending(user_id)
            if not pending_items:
                return None
            confirmed_ids: List[str] = []
            for pending in pending_items:
                item = self.create_confirmed(
                    user_id=user_id,
                    content=pending.content,
                    source_message_ids=pending.source_message_ids
                    or [source_message_id],
                )
                if item:
                    confirmed_ids.append(item.id)
            if not confirmed_ids:
                return "profile_rejected"
            return f"profile_confirmed:{len(confirmed_ids)}:{','.join(confirmed_ids)}"

        forget_key = self._detector.detect_forget(user_text)
        if forget_key:
            removed = self.forget_matching(user_id, forget_key)
            return f"profile_forgotten:{removed}"

        return None

    def maybe_capture_assistant_proposal(
        self, *, user_id: int, assistant_text: str, source_message_id: int
    ) -> Optional[str]:
        """Legacy regex offer path (weak). Prefer maybe_llm_propose after assistant turns."""
        offered = self._detector.detect_propose_in_assistant(assistant_text)
        if not offered:
            return None
        self.propose(
            user_id=user_id, content=offered, source_message_ids=[source_message_id]
        )
        return f"profile_proposed:{offered[:40]}"

    def maybe_llm_propose(
        self,
        *,
        user_id: int,
        llm: Any,
        recent_turns: Sequence[dict],
        source_message_id: int,
        force: bool = False,
    ) -> Optional[str]:
        """LLM extracts support-preference candidates → pending only (never active).

        By default runs only when batch triggers fire (message count
        or age). Pass force=True to bypass the batch gate (tests).
        """
        from product_app.app.memory.config import mem_cfg
        from product_app.app.memory.profile.proposer import (
            evidence_supported,
            propose_from_recent,
            roughly_same,
            user_texts,
        )

        if not force and not self._db.should_attempt_llm_propose(
            user_id,
            trigger_messages=int(mem_cfg.profile_llm_trigger_messages),
            trigger_age_sec=int(mem_cfg.profile_llm_trigger_age_sec),
        ):
            return None

        if mem_cfg.profile_llm_skip_if_pending and self._db.count_pending(user_id) > 0:
            # Still consume the batch so we do not spin every turn while pending waits.
            self._db.mark_llm_propose_attempted(user_id, source_message_id)
            return None

        existing = [p.content for p in self.list_active(user_id)]
        existing += [p.content for p in self._db.list_pending(user_id)]
        max_items = max(1, int(mem_cfg.profile_llm_propose_max))
        proposals = propose_from_recent(
            llm,
            recent_turns=recent_turns,
            existing=existing,
            max_items=max_items,
        )
        # Mark attempted whether or not anything was added.
        self._db.mark_llm_propose_attempted(user_id, source_message_id)
        if not proposals:
            return None
        u_msgs = user_texts(recent_turns)
        added = 0
        for prop in proposals[:max_items]:
            content = prop["content"]
            if any(roughly_same(content, e) for e in existing):
                continue
            if not evidence_supported(prop, u_msgs):
                continue
            before = self._db.count_pending(user_id)
            self.propose(
                user_id=user_id,
                content=content,
                source_message_ids=[source_message_id],
                source_messages=u_msgs or [content],
            )
            if self._db.count_pending(user_id) > before:
                added += 1
                existing.append(content)
        return f"profile_llm_proposed:{added}" if added else None

    def note_message_for_llm_propose(self, user_id: int) -> None:
        """Accumulate toward the next batch trigger (call on each ingested turn)."""
        self._db.bump_llm_pending(user_id)

    def _create(
        self,
        user_id: int,
        content: str,
        origin: str,
        source_message_ids: Sequence[int],
    ) -> Optional[ProfileItem]:
        ok, _reason = self._policy.validate(
            candidate=content, origin=origin, source_messages=[content]
        )
        if not ok:
            return None
        return self._db.create(
            user_id=user_id,
            content=content,
            origin=origin,
            source_message_ids=source_message_ids,
        )
