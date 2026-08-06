"""Memory engine: ingest turns, build recall context, profile helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from product_app.app.memory.commands.forget import handle_forget
from product_app.app.memory.commands.inspect import handle_inspect
from product_app.app.memory.commands.remember import handle_remember
from product_app.app.memory.config import mem_cfg
from product_app.app.memory.context.builder import build_memory_block
from product_app.app.memory.embeddings import MemoryEmbedder
from product_app.app.memory.models import Turn, RetrievalTrace
from product_app.app.memory.profile.service import ProfileService
from product_app.app.memory.retrieval.pipeline import RetrievalPipeline
from product_app.app.memory.retrieval.sufficiency import is_enough
from product_app.app.memory.store.ingest import TraceIngestor
from product_app.app.memory.store.repository import TraceStore
from product_app.app.memory.token_utils import fallback_token_count

logger = logging.getLogger(__name__)


class MemoryEngine:
    def __init__(self, db_path: str | Path, llm: Any = None) -> None:
        self._store = TraceStore(db_path)
        self._store.init()
        self._embedder = MemoryEmbedder.shared()
        self._ingestor = TraceIngestor(self._store, self._embedder)
        self._profile = ProfileService(db_path)
        self._pipeline = RetrievalPipeline(self._store, self._profile, llm=llm)
        self._llm = llm
        self._last_trace: Optional[RetrievalTrace] = None

    def set_llm(self, llm: Any) -> None:
        self._llm = llm
        self._pipeline.set_llm(llm)

    @property
    def last_trace(self) -> Optional[RetrievalTrace]:
        return self._last_trace

    def ingest_message(
        self,
        *,
        user_id: int,
        conversation_id: int,
        message_id: int,
        turn_id: int | None = None,
        role: str,
        content: str,
        position: int,
        created_at: int,
        reply_to_message_id: int | None = None,
        retrievable: bool = True,
        visible_to_user: bool = True,
        is_final: bool = True,
    ) -> int:
        if not mem_cfg.store_enabled or role not in ("user", "assistant"):
            return 0
        try:
            written = self._ingestor.ingest_message(
                Turn(
                    user_id=int(user_id),
                    conversation_id=int(conversation_id),
                    message_id=int(message_id),
                    turn_id=int(turn_id if turn_id is not None else message_id),
                    reply_to_message_id=reply_to_message_id,
                    role=role,  # type: ignore[arg-type]
                    content=content,
                    position=int(position),
                    created_at=int(created_at),
                    retrievable=bool(retrievable),
                    visible_to_user=bool(visible_to_user),
                    is_final=bool(is_final),
                )
            )
            if (
                mem_cfg.profile_enabled
                and role == "user"
                and mem_cfg.profile_llm_propose
                and self._llm is not None
            ):
                try:
                    recent = self._store.list_recent_messages(int(user_id), limit=9)
                    token_counter = getattr(self._llm, "count_tokens", None)
                    changes = self._profile.maintain_from_user_turn(
                        user_id=int(user_id),
                        current_user_message_id=int(message_id),
                        recent_turns=recent,
                        llm=self._llm,
                        token_counter=token_counter,
                    )
                    summary = changes.summary()
                    if summary and mem_cfg.observability:
                        logger.info("memory_profile %s", summary)
                except Exception:
                    logger.warning("profile maintain failed", exc_info=True)
            try:
                self._ingestor.process_embed_retries(limit=10)
            except Exception:
                pass
            return written
        except Exception:
            logger.warning("memory ingest failed message_id=%s", message_id, exc_info=True)
            return 0

    def build_context(
        self,
        *,
        user_id: int,
        conversation_id: int,
        current_user_message: str,
        recent_messages: List[dict],
        conversation_summary: Optional[str],
        token_budget: int | None = None,
        exclude_message_ids: Optional[Set[int]] = None,
    ) -> str:
        if not mem_cfg.store_enabled:
            return ""

        del recent_messages, conversation_summary, conversation_id

        try:
            query = (current_user_message or "").strip()
            if not query:
                return ""

            windows, trace = self._pipeline.run(
                user_id=user_id,
                query=query,
                exclude_message_ids=exclude_message_ids,
            )
            counter = getattr(self._llm, "count_tokens", None) if self._llm else None
            profiles = (
                self._profile.list_all_for_context(
                    user_id=user_id, token_counter=counter
                )
                if mem_cfg.profile_enabled
                else []
            )
            trace.profile_hits = len(profiles)
            trace.enough = is_enough(windows, profiles)

            budget = int(token_budget or mem_cfg.context_token_budget)
            block, packed_count = build_memory_block(
                bundles=windows,
                profiles=profiles,
                token_budget=budget,
                token_counter=counter,
                query=query,
            )
            trace.selected_bundles = packed_count
            trace.extra["packed_window_count"] = packed_count
            trace.extra["topk_before_budget"] = len(windows)
            if counter and block:
                trace.memory_tokens = int(counter(block))
            else:
                trace.memory_tokens = fallback_token_count(block) if block else 0

            self._last_trace = trace
            if mem_cfg.observability:
                logger.info("memory_trace %s", trace.to_log_dict())
            return block
        except Exception:
            logger.warning("memory build_context failed user=%s", user_id, exc_info=True)
            self._last_trace = RetrievalTrace(fallback=True)
            return ""

    def remember_explicit(self, *, user_id: int, content: str, source_message_ids: List[int]):
        return handle_remember(
            self._profile,
            user_id=user_id,
            content=content,
            source_message_ids=source_message_ids,
        )

    def confirm_profile(self, *, user_id: int, content: str, source_message_ids: List[int]):
        return self._profile.create_confirmed(
            user_id=user_id, content=content, source_message_ids=source_message_ids
        )

    def forget_profile(self, *, user_id: int, profile_id: str) -> None:
        handle_forget(
            repo=self._store, profile=self._profile, user_id=user_id, profile_id=profile_id
        )

    def forget_message(self, *, user_id: int, message_id: int) -> None:
        handle_forget(
            repo=self._store, profile=self._profile, user_id=user_id, message_id=message_id
        )

    def forget_keyword(self, *, user_id: int, keyword: str) -> dict:
        return handle_forget(
            repo=self._store, profile=self._profile, user_id=user_id, keyword=keyword
        )

    def count_active(self, user_id: int) -> int:
        return self._store.count_active(user_id) + len(self._profile.list_active(user_id))

    def forget_all(self, user_id: int) -> int:
        trace_count = self._store.forget_user(user_id)
        profile_count = self._profile.forget_all(user_id)
        return int(trace_count) + int(profile_count)

    def inspect(self, user_id: int) -> Dict[str, object]:
        info = handle_inspect(repo=self._store, profile=self._profile, user_id=user_id)
        info["backend"] = "er"
        return info

    def process_retries(self) -> int:
        return self._ingestor.process_embed_retries(limit=100)
