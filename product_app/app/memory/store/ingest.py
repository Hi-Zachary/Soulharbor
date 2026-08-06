"""Write chat turns into the trace store and keep embeddings warm."""
from __future__ import annotations

import logging
from typing import Optional

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.embeddings import MemoryEmbedder
from product_app.app.memory.models import Turn, searchable_text
from product_app.app.memory.store.index_units import build_index_units
from product_app.app.memory.store.repository import TraceStore

logger = logging.getLogger(__name__)


class TraceIngestor:
    def __init__(self, store: TraceStore, embedder: Optional[MemoryEmbedder] = None) -> None:
        self._store = store
        self._embedder = embedder or MemoryEmbedder.shared()

    def ingest_message(self, message: Turn) -> int:
        body = (message.content or "").strip()
        if not body or message.role not in ("user", "assistant"):
            return 0

        has_segments, units, split_segments = build_index_units(
            message_id=int(message.message_id),
            role=str(message.role),
            content=body,
        )
        chunk_ids = self._store.upsert_message_index(
            user_id=message.user_id,
            conversation_id=message.conversation_id,
            message_id=message.message_id,
            turn_id=message.turn_id,
            role=message.role,
            position=message.position,
            created_at=message.created_at,
            content=body,
            reply_to_message_id=message.reply_to_message_id,
            retrievable=bool(message.retrievable),
            visible_to_user=bool(message.visible_to_user),
            is_final=bool(message.is_final),
            has_segments=has_segments,
            index_units=[
                {
                    "unit_type": unit.unit_type,
                    "parent_message_id": unit.parent_message_id,
                    "content": unit.content,
                    "segment_index": unit.segment_index,
                }
                for unit in units
            ],
            segments=[
                {
                    "segment_index": seg.segment_index,
                    "content": seg.content,
                    "start_offset": seg.start_offset,
                    "end_offset": seg.end_offset,
                    "token_count": seg.token_count,
                }
                for seg in split_segments
            ],
        )

        for chunk_id, unit in zip(chunk_ids, units):
            self._embed_or_retry(chunk_id, message.user_id, unit.role, unit.content)
        return len(chunk_ids)

    def process_embed_retries(self, *, limit: int = 50) -> int:
        jobs = self._store.list_embed_retries(
            max_attempts=int(mem_cfg.embed_retry_max),
            limit=limit,
        )
        ok = 0
        for job in jobs:
            try:
                vector = self._embedder.embed(
                    searchable_text(str(job["role"]), str(job["content"]))
                )
                if vector:
                    self._store.save_embedding(job["chunk_id"], job["user_id"], vector)
                    ok += 1
                else:
                    self._store.enqueue_embed_retry(
                        job["chunk_id"], job["user_id"], "empty embedding"
                    )
            except Exception as exc:
                self._store.enqueue_embed_retry(job["chunk_id"], job["user_id"], str(exc))
        return ok

    def _embed_or_retry(self, chunk_id: int, user_id: int, role: str, text: str) -> None:
        try:
            vector = self._embedder.embed(searchable_text(role, text))
            if vector:
                self._store.save_embedding(chunk_id, user_id, vector)
            else:
                self._store.enqueue_embed_retry(chunk_id, user_id, "empty embedding")
        except Exception as exc:
            logger.warning("embed failed for chunk_id=%s", chunk_id, exc_info=True)
            self._store.enqueue_embed_retry(chunk_id, user_id, str(exc))
