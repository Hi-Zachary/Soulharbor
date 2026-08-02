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

    # --- write path ---------------------------------------------------------

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
    ) -> int:
        if not mem_cfg.store_enabled or role not in ("user", "assistant"):
            return 0
        try:
            written = self._ingestor.ingest_message(
                Turn(
                    user_id=int(user_id),
                    conversation_id=int(conversation_id),
                    message_id=int(message_id),
                    role=role,  # type: ignore[arg-type]
                    content=content,
                    position=int(position),
                    created_at=int(created_at),
                )
            )
            if mem_cfg.profile_enabled and role == "user":
                self._profile.maybe_handle_user_command(
                    user_id=int(user_id),
                    user_text=content,
                    source_message_id=int(message_id),
                )
            if mem_cfg.profile_enabled and role in ("user", "assistant"):
                # Count toward batch trigger on every turn.
                if mem_cfg.profile_llm_propose:
                    self._profile.note_message_for_llm_propose(int(user_id))
            if mem_cfg.profile_enabled and role == "assistant":
                # Weak regex offer (legacy).
                self._profile.maybe_capture_assistant_proposal(
                    user_id=int(user_id),
                    assistant_text=content,
                    source_message_id=int(message_id),
                )
                # Batch-gated Chinese LLM propose → pending (consent still required).
                if mem_cfg.profile_llm_propose and self._llm is not None:
                    try:
                        recent = self._store.list_recent_messages(int(user_id), limit=8)
                        self._profile.maybe_llm_propose(
                            user_id=int(user_id),
                            llm=self._llm,
                            recent_turns=recent,
                            source_message_id=int(message_id),
                        )
                    except Exception:
                        logger.warning("profile LLM propose failed", exc_info=True)
            try:
                self._ingestor.process_embed_retries(limit=10)
            except Exception:
                pass
            return written
        except Exception:
            logger.warning("memory ingest failed message_id=%s", message_id, exc_info=True)
            return 0

    # --- read path ----------------------------------------------------------

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

        # reserved for callers / future session-scoped filters
        del conversation_id, recent_messages, conversation_summary

        try:
            query = (current_user_message or "").strip()
            if not query:
                return ""

            if self._profile.detector.detect_inspect(query):
                return self._inspect_block(user_id)

            windows, profiles, trace = self._pipeline.run(
                user_id=user_id,
                query=query,
                exclude_message_ids=exclude_message_ids,
            )
            trace.selected_bundles = len(windows)
            trace.enough = is_enough(windows, profiles)

            budget = int(token_budget or mem_cfg.context_token_budget)
            counter = getattr(self._llm, "count_tokens", None) if self._llm else None
            block = build_memory_block(
                bundles=windows,
                profiles=profiles,
                token_budget=budget,
                token_counter=counter,
                query=query,
            )
            if counter and block:
                trace.memory_tokens = int(counter(block))
            else:
                trace.memory_tokens = max(1, int(len(block) / 1.5)) if block else 0

            self._last_trace = trace
            if mem_cfg.observability:
                logger.info("memory_trace %s", trace.to_log_dict())
            return block
        except Exception:
            logger.warning("memory build_context failed user=%s", user_id, exc_info=True)
            self._last_trace = RetrievalTrace(fallback=True)
            return ""

    def _inspect_block(self, user_id: int) -> str:
        info = handle_inspect(repo=self._store, profile=self._profile, user_id=user_id)
        prefs = info.get("support_preferences") or []
        lines = ["[已确认的支持偏好]"]
        if prefs:
            lines.extend(f"- {p.get('content')}" for p in prefs)
        else:
            lines.append("- （暂无经确认的支持偏好）")
        return "<memory>\n" + "\n".join(lines) + "\n</memory>"

    # --- profile helpers ----------------------------------------------------

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
        return self._store.forget_user(user_id)

    def inspect(self, user_id: int) -> Dict[str, object]:
        info = handle_inspect(repo=self._store, profile=self._profile, user_id=user_id)
        info["backend"] = "aer"
        return info

    def process_retries(self) -> int:
        return self._ingestor.process_embed_retries(limit=100)
