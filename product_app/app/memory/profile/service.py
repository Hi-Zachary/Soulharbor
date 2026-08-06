"""Long-term profile side channel: LLM proposes ops; code owns safety."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence

from product_app.app.memory.models import ProfileItem
from product_app.app.memory.profile.operations import AppliedProfileChanges, ProfileOperation
from product_app.app.memory.profile.repository import ProfileStore
from product_app.app.memory.token_utils import TokenCounter

logger = logging.getLogger(__name__)


def _message_content_hash(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def _current_message_content(
    recent_turns: Sequence[dict], current_user_message_id: int
) -> str:
    mid = int(current_user_message_id)
    for turn in recent_turns:
        try:
            if int(turn["message_id"]) == mid:
                return str(turn.get("content") or "")
        except (KeyError, TypeError, ValueError):
            continue
    return ""


class ProfileService:
    def __init__(self, db_path: str | Path) -> None:
        self._db = ProfileStore(db_path)

    def _cfg(self):
        from product_app.app.memory.config import mem_cfg

        return mem_cfg

    def list_active(
        self, user_id: int, token_counter: TokenCounter = None
    ) -> List[ProfileItem]:
        cfg = self._cfg()
        return self._db.list_active_with_repair(
            user_id=int(user_id),
            max_active=int(cfg.profile_max_active),
            max_block_tokens=int(cfg.profile_block_max_tokens),
            token_counter=token_counter,
        )

    def list_all_for_context(
        self, *, user_id: int, token_counter: TokenCounter = None
    ) -> List[ProfileItem]:
        cfg = self._cfg()
        return self._db.list_active_with_repair(
            user_id=int(user_id),
            max_active=int(cfg.profile_max_active),
            max_block_tokens=int(cfg.profile_block_max_tokens),
            token_counter=token_counter,
        )

    def list_for_inject(
        self, user_id: int, limit: int | None = None, token_counter: TokenCounter = None
    ) -> List[ProfileItem]:
        cfg = self._cfg()
        cap = int(limit if limit is not None else cfg.profile_max_active)
        items = self.list_all_for_context(user_id=user_id, token_counter=token_counter)
        return items[: max(1, cap)]

    def delete(self, user_id: int, profile_id: str) -> bool:
        return self._db.soft_delete(user_id, profile_id)

    def forget_matching(self, user_id: int, keyword: str) -> int:
        return self._db.soft_delete_matching(user_id, keyword)

    def forget_all(self, user_id: int) -> int:
        return self._db.soft_delete_user(int(user_id))

    def remove_source_message(self, *, user_id: int, message_id: int) -> list[str]:
        return self._db.remove_source_message(
            user_id=int(user_id), message_id=int(message_id)
        )

    def update(self, user_id: int, profile_id: str, content: str) -> bool:
        return self._db.update_content(user_id, profile_id, content)

    def create_explicit(
        self,
        *,
        user_id: int,
        content: str,
        source_message_ids: Sequence[int],
        token_counter: TokenCounter = None,
    ) -> Optional[ProfileItem]:
        """Admin/tool path: one add via maintainer apply (requires owned user message)."""
        cleaned = (content or "").strip()
        mids = [int(x) for x in source_message_ids if int(x) > 0]
        if not cleaned or not mids:
            return None
        cfg = self._cfg()
        changes = self._db.apply_profile_operations(
            user_id=int(user_id),
            current_user_message_id=mids[0],
            operations=[
                ProfileOperation(op="add", target_id="", content=cleaned),
            ],
            token_counter=token_counter,
            max_active=int(cfg.profile_max_active),
            max_block_tokens=int(cfg.profile_block_max_tokens),
            max_operations=1,
            max_chars=int(cfg.profile_item_max_chars),
            max_tokens=int(cfg.profile_item_max_tokens),
        )
        if not changes.added_ids:
            return None
        for item in self.list_all_for_context(
            user_id=user_id, token_counter=token_counter
        ):
            if item.id == changes.added_ids[0]:
                return item
        return None

    def create_confirmed(
        self, *, user_id: int, content: str, source_message_ids: Sequence[int]
    ) -> Optional[ProfileItem]:
        return self.create_explicit(
            user_id=user_id,
            content=content,
            source_message_ids=source_message_ids,
        )

    def correct(
        self,
        *,
        user_id: int,
        old_keyword: str,
        new_content: str | None,
        source_message_ids: Sequence[int],
    ) -> Optional[ProfileItem]:
        """Compat: keyword forget + optional replacement add."""
        self.forget_matching(user_id, old_keyword)
        replacement = (new_content or "").strip()
        if not replacement:
            return None
        return self.create_explicit(
            user_id=user_id,
            content=replacement,
            source_message_ids=source_message_ids,
        )

    def maintain_from_user_turn(
        self,
        *,
        user_id: int,
        current_user_message_id: int,
        recent_turns: Sequence[dict],
        llm: Any,
        token_counter: TokenCounter = None,
    ) -> AppliedProfileChanges:
        from product_app.app.memory.profile.maintainer import propose_operations

        cfg = self._cfg()
        if not cfg.profile_enabled or not cfg.profile_llm_propose or llm is None:
            return AppliedProfileChanges()

        uid = int(user_id)
        mid = int(current_user_message_id)
        content = _current_message_content(recent_turns, mid)
        content_hash = _message_content_hash(content)

        already = self._db.has_maintenance_run(
            user_id=uid, message_id=mid, content_hash=content_hash
        )

        current_profiles = self._db.list_active(
            user_id=uid,
            limit=int(cfg.profile_max_active),
            order_by="created_at ASC, id ASC",
        )

        if already:
            operations: list[ProfileOperation] = []
        else:
            decision = propose_operations(
                llm,
                profiles=current_profiles,
                recent_turns=recent_turns,
                current_user_message_id=mid,
                max_active=int(cfg.profile_max_active),
                target_chars=int(cfg.profile_item_target_chars),
                max_operations=int(cfg.profile_max_operations),
                token_counter=token_counter,
            )
            operations = list(decision.operations)

        try:
            changes = self._db.apply_profile_operations(
                user_id=uid,
                current_user_message_id=mid,
                operations=operations,
                token_counter=token_counter,
                max_active=int(cfg.profile_max_active),
                max_block_tokens=int(cfg.profile_block_max_tokens),
                max_operations=int(cfg.profile_max_operations),
                max_chars=int(cfg.profile_item_max_chars),
                max_tokens=int(cfg.profile_item_max_tokens),
            )
        except ValueError:
            logger.warning(
                "profile maintain skipped: invalid current user message id=%s",
                mid,
            )
            return AppliedProfileChanges()

        if not already:
            self._db.record_maintenance_run(
                user_id=uid, message_id=mid, content_hash=content_hash
            )
        return changes

    def maybe_llm_extract(self, **kwargs: Any) -> Optional[str]:
        changes = self.maintain_from_user_turn(
            user_id=int(kwargs["user_id"]),
            current_user_message_id=int(
                kwargs.get("current_user_message_id")
                or kwargs.get("source_message_id")
                or 0
            ),
            recent_turns=kwargs.get("recent_turns") or [],
            llm=kwargs.get("llm"),
            token_counter=kwargs.get("token_counter"),
        )
        return changes.summary()

    def maybe_llm_propose(self, **kwargs: Any) -> Optional[str]:
        return self.maybe_llm_extract(**kwargs)

    def note_message_for_llm_propose(self, user_id: int) -> None:
        return None
