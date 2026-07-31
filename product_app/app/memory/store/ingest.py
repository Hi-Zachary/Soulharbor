"""Write chat turns into the episode store and keep embeddings warm."""
from __future__ import annotations

import logging
from typing import Optional

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.embeddings import MemoryEmbedder
from product_app.app.memory.models import EpisodeMessage
from product_app.app.memory.store.chunker import chunk_text
from product_app.app.memory.store.repository import EpisodeStore

logger = logging.getLogger(__name__)


class EpisodeIngestor:
    def __init__(self, store: EpisodeStore, embedder: Optional[MemoryEmbedder] = None) -> None:
        self._store = store
        self._embedder = embedder or MemoryEmbedder.shared()

    def ingest_message(self, message: EpisodeMessage) -> int:
        body = (message.content or "").strip()
        if not body or message.role not in ("user", "assistant"):
            return 0

        pieces = chunk_text(body)
        chunk_ids = self._store.upsert_chunks(
            user_id=message.user_id,
            conversation_id=message.conversation_id,
            message_id=message.message_id,
            role=message.role,
            position=message.position,
            created_at=message.created_at,
            chunks=pieces,
        )

        for chunk_id, piece in zip(chunk_ids, pieces):
            self._embed_or_retry(chunk_id, message.user_id, piece)
        return len(chunk_ids)

    def process_embed_retries(self, *, limit: int = 50) -> int:
        jobs = self._store.list_embed_retries(
            max_attempts=int(mem_cfg.embed_retry_max),
            limit=limit,
        )
        ok = 0
        for job in jobs:
            try:
                vector = self._embedder.embed(job["content"])
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

    def _embed_or_retry(self, chunk_id: int, user_id: int, text: str) -> None:
        try:
            vector = self._embedder.embed(text)
            if vector:
                self._store.save_embedding(chunk_id, user_id, vector)
            else:
                self._store.enqueue_embed_retry(chunk_id, user_id, "empty embedding")
        except Exception as exc:
            logger.warning("embed failed for chunk_id=%s", chunk_id, exc_info=True)
            self._store.enqueue_embed_retry(chunk_id, user_id, str(exc))
