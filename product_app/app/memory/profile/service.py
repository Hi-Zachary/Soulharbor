"""Long-term profile side channel: LLM proposes ops; code owns safety."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence

from product_app.app.memory.context.profile_formatter import render_user_profile
from product_app.app.memory.models import ProfileItem
from product_app.app.memory.profile.operations import AppliedProfileChanges, ProfileOperation
from product_app.app.memory.profile.repository import ProfileStore
from product_app.app.memory.token_utils import TokenCounter, count_tokens

logger = logging.getLogger(__name__)


class ProfileService:
    def __init__(self, db_path: str | Path) -> None:
        self._db = ProfileStore(db_path)

    def _cfg(self):
        from product_app.app.memory.config import mem_cfg

        return mem_cfg

    def list_active(self, user_id: int) -> List[ProfileItem]:
        cfg = self._cfg()
        return self._db.list_active_with_repair(
            user_id=int(user_id),
            max_active=int(cfg.profile_max_active),
            max_block_tokens=int(cfg.profile_block_max_tokens),
        )

    def list_all_for_context(self, *, user_id: int) -> List[ProfileItem]:
        cfg = self._cfg()
        return self._db.list_active(
            user_id=int(user_id),
            limit=int(cfg.profile_max_active),
            order_by="created_at ASC, id ASC",
        )

    def list_for_inject(
        self, user_id: int, limit: int | None = None
    ) -> List[ProfileItem]:
        cfg = self._cfg()
        cap = int(limit if limit is not None else cfg.profile_max_active)
        items = self.list_all_for_context(user_id=user_id)
        return items[: max(1, cap)]

    def delete(self, user_id: int, profile_id: str) -> bool:
        return self._db.soft_delete(user_id, profile_id)

    def forget_matching(self, user_id: int, keyword: str) -> int:
        return self._db.soft_delete_matching(user_id, keyword)

    def forget_key(self, user_id: int, tag: str, feature: str) -> int:
        return self._db.soft_delete_by_key(user_id, tag, feature)

    def update(self, user_id: int, profile_id: str, content: str) -> bool:
        return self._db.update_content(user_id, profile_id, content)

    def create_explicit(
        self, *, user_id: int, content: str, source_message_ids: Sequence[int]
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
            max_active=int(cfg.profile_max_active),
            max_block_tokens=int(cfg.profile_block_max_tokens),
            max_operations=1,
            max_chars=int(cfg.profile_item_max_chars),
            max_tokens=int(cfg.profile_item_max_tokens),
        )
        if not changes.added_ids:
            return None
        for item in self.list_all_for_context(user_id=user_id):
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

        current_profiles = self._db.list_active(
            user_id=int(user_id),
            limit=int(cfg.profile_max_active),
            order_by="created_at ASC, id ASC",
        )
        current_block = render_user_profile(current_profiles)
        decision = propose_operations(
            llm,
            profiles=current_profiles,
            recent_turns=recent_turns,
            current_user_message_id=int(current_user_message_id),
            max_active=int(cfg.profile_max_active),
            block_tokens=count_tokens(current_block, token_counter),
            max_block_tokens=int(cfg.profile_block_max_tokens),
        )
        if not decision.operations:
            return AppliedProfileChanges()

        try:
            return self._db.apply_profile_operations(
                user_id=int(user_id),
                current_user_message_id=int(current_user_message_id),
                operations=decision.operations,
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
                current_user_message_id,
            )
            return AppliedProfileChanges()

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

    def maybe_handle_user_command(self, **kwargs: Any) -> Optional[str]:
        """Inspect-only remnant; remember/forget handled by maintainer."""
        parsed = kwargs.get("parsed")
        if parsed is None:
            llm = kwargs.get("llm")
            user_text = str(kwargs.get("user_text") or "")
            if llm is None or not user_text:
                return None
            from product_app.app.memory.profile.commands_llm import parse_user_command

            parsed = parse_user_command(llm, user_text=user_text)
        if (parsed or {}).get("action") == "inspect":
            return "profile_inspect"
        return None
