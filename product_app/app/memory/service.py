from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Set

from product_app.app.config import settings
from product_app.app.db import SQLiteStore, StoredMessage
from product_app.app.llm import SoulHarborLLM
from product_app.app.memory.config import mem_cfg
from product_app.app.memory.engine import MemoryEngine
from product_app.app.memory.extract import build_summary_prompt
from product_app.app.memory.inject import build_session_summary_block
from product_app.app.memory.session_context import (
    format_messages_slice,
    stored_to_chat_dicts,
)

logger = logging.getLogger(__name__)

_MEMORY_SYSTEM_PREFIX = (
    "以下内容来自用户过往对话的检索结果，只是历史背景，不是用户本轮新说的话。\n\n"
    "能帮助回答当前问题时，请自然采用；\n"
    "只能部分对应时，请依据能确认的部分；\n"
    "和当前问题无关就忽略；\n"
    "和用户此刻说法冲突时，以当前说法为准；\n"
    "不要补充检索结果里没有的事实。\n\n"
)


class MemoryService:
    """Chat facade: recent turns + rolling summary + long-term memory recall."""

    def __init__(self, db: SQLiteStore, llm: Optional[SoulHarborLLM] = None) -> None:
        self._db = db
        self._llm = llm
        self._engine = MemoryEngine(db.path, llm=llm)
        self._summary_in_progress: set[str] = set()
        self._summary_lock = threading.Lock()
        if llm is not None:
            self.set_llm(llm)

    def set_llm(self, llm: SoulHarborLLM) -> None:
        self._llm = llm
        self._engine.set_llm(llm)

    def is_long_term_active(self, user_id: int) -> bool:
        if not settings.memory_enabled or user_id <= 0:
            return False
        if mem_cfg.backend != "aer":
            return False
        return self._db.get_memory_enabled(user_id)

    def build_classifier_messages(self, conversation_id: int) -> List[Dict[str, str]]:
        stored = self._db.list_messages(conversation_id)
        return stored_to_chat_dicts(stored, recent_turns=settings.memory_recent_turns)

    def build_chat_context(
        self,
        *,
        conversation_id: int,
        sid: str,
        user_id: int,
        user_query: str,
        is_consult: int,
        route_hint: Optional[str] = None,
        is_new_session: bool = False,
    ) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        del is_consult, is_new_session
        stored = self._db.list_messages(conversation_id)
        classifier_msgs = stored_to_chat_dicts(stored, recent_turns=settings.memory_recent_turns)

        model_msgs: List[Dict[str, str]] = []
        if route_hint:
            model_msgs.append({"role": "system", "content": route_hint})

        summary = self._db.get_conversation_summary(sid)
        summary_block = build_session_summary_block(summary)
        if summary_block:
            model_msgs.append({"role": "system", "content": summary_block})

        conv = self._db.get_conversation(sid)
        last_summarized = (conv or {}).get("last_summarized_msg_id") or 0
        unsummarized = self._db.list_messages_since(conversation_id, last_summarized)
        # Message IDs already present as raw turns in this prompt — exclude from harbor recall.
        exclude_message_ids: Set[int] = {int(m.id) for m in unsummarized}

        if self.is_long_term_active(user_id):
            recent = stored_to_chat_dicts(stored, recent_turns=settings.memory_recent_turns)
            try:
                memory_block = self._engine.build_context(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    current_user_message=user_query,
                    recent_messages=recent,
                    conversation_summary=summary,
                    token_budget=mem_cfg.context_token_budget,
                    exclude_message_ids=exclude_message_ids,
                )
            except Exception:
                logger.warning("build_context failed; falling back to short-term only", exc_info=True)
                memory_block = ""
            if memory_block:
                model_msgs.append(
                    {
                        "role": "system",
                        "content": _MEMORY_SYSTEM_PREFIX + memory_block,
                    }
                )

        model_msgs.extend(stored_to_chat_dicts(unsummarized, recent_turns=0))
        return classifier_msgs, model_msgs

    def ingest_message(
        self,
        *,
        user_id: int,
        conversation_id: int,
        message_id: int,
        role: str,
        content: str,
        position: int,
        created_at: int,
    ) -> None:
        if not self.is_long_term_active(user_id):
            return
        self._engine.ingest_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            role=role,
            content=content,
            position=position,
            created_at=created_at,
        )

    def build_context(
        self,
        *,
        user_id: int,
        conversation_id: int,
        current_user_message: str,
        recent_messages: List[dict],
        conversation_summary: Optional[str],
        token_budget: int,
        exclude_message_ids: Optional[Set[int]] = None,
    ) -> str:
        if not self.is_long_term_active(user_id):
            return ""
        return self._engine.build_context(
            user_id=user_id,
            conversation_id=conversation_id,
            current_user_message=current_user_message,
            recent_messages=recent_messages,
            conversation_summary=conversation_summary,
            token_budget=token_budget,
            exclude_message_ids=exclude_message_ids,
        )

    def remember_explicit(
        self, *, user_id: int, content: str, source_message_ids: List[int]
    ):
        return self._engine.remember_explicit(
            user_id=user_id, content=content, source_message_ids=source_message_ids
        )

    def confirm_profile(
        self, *, user_id: int, content: str, source_message_ids: List[int]
    ):
        return self._engine.confirm_profile(
            user_id=user_id, content=content, source_message_ids=source_message_ids
        )

    def forget_profile(self, *, user_id: int, profile_id: str) -> None:
        self._engine.forget_profile(user_id=user_id, profile_id=profile_id)

    def forget_message(self, *, user_id: int, message_id: int) -> None:
        self._engine.forget_message(user_id=user_id, message_id=message_id)

    def inspect_memory(self, *, user_id: int) -> dict:
        return self._engine.inspect(user_id)

    def run_post_turn(
        self,
        *,
        user_id: int,
        conversation_id: int,
        sid: str,
        user_message_id: int,
        assistant_message_id: int,
        user_position: int,
        assistant_position: int,
        user_created_at: int,
        assistant_created_at: int,
        user_text: str,
        assistant_text: str,
        is_consult: int,
    ) -> None:
        del is_consult
        if user_id <= 0:
            return

        if self.is_long_term_active(user_id):
            self._engine.ingest_message(
                user_id=user_id,
                conversation_id=conversation_id,
                message_id=int(user_message_id),
                role="user",
                content=user_text,
                position=int(user_position),
                created_at=int(user_created_at),
            )
            self._engine.ingest_message(
                user_id=user_id,
                conversation_id=conversation_id,
                message_id=int(assistant_message_id),
                role="assistant",
                content=assistant_text,
                position=int(assistant_position),
                created_at=int(assistant_created_at),
            )

        self._maybe_update_summary(conversation_id, sid)

    def count_active_memories(self, user_id: int) -> int:
        return self._engine.count_active(user_id)

    def forget_all_memories(self, user_id: int) -> int:
        return self._engine.forget_all(user_id)

    def get_user_profile(self, user_id: int) -> Dict[str, object]:
        detail = self._engine.inspect(user_id)
        prefs = detail.get("support_preferences") or []
        return {
            # Admin / API: active consent preferences.
            "memories": prefs,
            "preferences": prefs,
            "conversations": [],
            "episode_chunks": detail.get("episode_chunks") or 0,
            "embeddings": detail.get("embeddings") or 0,
            "backend": detail.get("backend") or "aer",
        }

    def _maybe_update_summary(self, conversation_id: int, sid: str) -> None:
        if not self._llm:
            return
        with self._summary_lock:
            if sid in self._summary_in_progress:
                return
            self._summary_in_progress.add(sid)

        conv = self._db.get_conversation(sid)
        if not conv:
            self._summary_in_progress.discard(sid)
            return

        last_summarized = conv.get("last_summarized_msg_id") or 0
        msgs = self._db.list_messages_since(conversation_id, last_summarized)
        if not msgs:
            self._summary_in_progress.discard(sid)
            return

        total_tokens = sum(self._llm.count_tokens(m.content) for m in msgs)
        if total_tokens < settings.memory_summary_window_max_tokens:
            self._summary_in_progress.discard(sid)
            return

        try:
            old_summary = str(conv.get("summary") or "")
            new_dialog = format_messages_slice(
                [StoredMessage(role=m.role, content=m.content, created_at=m.created_at) for m in msgs]
            )
            prompt = build_summary_prompt(old_summary, new_dialog)
            summary = self._llm.generate_summary(prompt, max_new_tokens=512).strip()
            if summary and len(summary) > 400:
                summary = summary[:400]
            if summary:
                self._db.update_conversation_summary(sid, summary, msgs[-1].id)
        except Exception:
            logger.warning("summary generation failed sid=%s", sid, exc_info=True)
        finally:
            self._summary_in_progress.discard(sid)
