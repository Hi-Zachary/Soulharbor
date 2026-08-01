"""SQLite-backed episode chunk / embedding store."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from product_app.app.memory.models import EpisodeChunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_episode_chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  conversation_id INTEGER NOT NULL,
  message_id INTEGER NOT NULL,
  role TEXT NOT NULL,
  position INTEGER NOT NULL,
  chunk_index INTEGER NOT NULL DEFAULT 0,
  content TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  is_deleted INTEGER NOT NULL DEFAULT 0,
  UNIQUE(message_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_ep_user_created
  ON memory_episode_chunks(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ep_conv_pos
  ON memory_episode_chunks(conversation_id, position);
CREATE INDEX IF NOT EXISTS idx_ep_user_msg
  ON memory_episode_chunks(user_id, message_id);

CREATE TABLE IF NOT EXISTS memory_episode_embeddings (
  chunk_id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  embedding_json TEXT NOT NULL,
  updated_at INTEGER NOT NULL,
  FOREIGN KEY(chunk_id) REFERENCES memory_episode_chunks(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_ep_emb_user ON memory_episode_embeddings(user_id);

CREATE TABLE IF NOT EXISTS memory_embed_retry (
  chunk_id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  updated_at INTEGER NOT NULL,
  FOREIGN KEY(chunk_id) REFERENCES memory_episode_chunks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS support_profile_items (
  id TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL,
  content TEXT NOT NULL,
  origin TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_profile_user_status
  ON support_profile_items(user_id, status);

CREATE TABLE IF NOT EXISTS support_profile_sources (
  profile_id TEXT NOT NULL,
  message_id INTEGER NOT NULL,
  PRIMARY KEY(profile_id, message_id)
);

CREATE TABLE IF NOT EXISTS support_profile_pending (
  id TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL,
  content TEXT NOT NULL,
  source_json TEXT NOT NULL,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_profile_pending_user
  ON support_profile_pending(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS memory_backfill_checkpoint (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);
"""


def _now() -> int:
    return int(time.time())


def content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _unique_messages(rows: Sequence[sqlite3.Row], *, mark_seed_at: int | None = None) -> List[Dict[str, Any]]:
    """Collapse multi-chunk rows into one dict per message_id."""
    seen: set[int] = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        mid = int(row["message_id"])
        if mid in seen:
            continue
        seen.add(mid)
        item = {
            "message_id": mid,
            "role": str(row["role"]),
            "position": int(row["position"]),
            "content": str(row["content"]),
            "created_at": int(row["created_at"]),
        }
        if mark_seed_at is not None:
            item["is_seed"] = int(row["position"]) == int(mark_seed_at)
        out.append(item)
    return out


class EpisodeStore:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self._db() as conn:
            conn.executescript(_SCHEMA)

    # --- writes -------------------------------------------------------------

    def upsert_chunks(
        self,
        *,
        user_id: int,
        conversation_id: int,
        message_id: int,
        role: str,
        position: int,
        created_at: int,
        chunks: Sequence[str],
    ) -> List[int]:
        ids: List[int] = []
        with self._db() as conn:
            for idx, piece in enumerate(chunks):
                piece = (piece or "").strip()
                if not piece:
                    continue
                conn.execute(
                    "INSERT INTO memory_episode_chunks("
                    "user_id, conversation_id, message_id, role, position, chunk_index, "
                    "content, content_hash, created_at, is_deleted"
                    ") VALUES(?,?,?,?,?,?,?,?,?,0) "
                    "ON CONFLICT(message_id, chunk_index) DO UPDATE SET "
                    "content=excluded.content, content_hash=excluded.content_hash, "
                    "is_deleted=0, created_at=excluded.created_at, "
                    "role=excluded.role, position=excluded.position, "
                    "conversation_id=excluded.conversation_id, user_id=excluded.user_id",
                    (
                        int(user_id),
                        int(conversation_id),
                        int(message_id),
                        str(role),
                        int(position),
                        int(idx),
                        piece,
                        content_hash(piece),
                        int(created_at),
                    ),
                )
                row = conn.execute(
                    "SELECT id FROM memory_episode_chunks WHERE message_id=? AND chunk_index=?",
                    (int(message_id), int(idx)),
                ).fetchone()
                if row:
                    ids.append(int(row["id"]))
        return ids

    def save_embedding(self, chunk_id: int, user_id: int, embedding: List[float]) -> None:
        with self._db() as conn:
            conn.execute(
                "INSERT INTO memory_episode_embeddings(chunk_id, user_id, embedding_json, updated_at) "
                "VALUES(?,?,?,?) "
                "ON CONFLICT(chunk_id) DO UPDATE SET embedding_json=excluded.embedding_json, "
                "updated_at=excluded.updated_at",
                (int(chunk_id), int(user_id), json.dumps(embedding), _now()),
            )
            conn.execute("DELETE FROM memory_embed_retry WHERE chunk_id=?", (int(chunk_id),))

    def enqueue_embed_retry(self, chunk_id: int, user_id: int, error: str = "") -> None:
        with self._db() as conn:
            conn.execute(
                "INSERT INTO memory_embed_retry(chunk_id, user_id, attempts, last_error, updated_at) "
                "VALUES(?,?,1,?,?) "
                "ON CONFLICT(chunk_id) DO UPDATE SET "
                "attempts=attempts+1, last_error=excluded.last_error, updated_at=excluded.updated_at",
                (int(chunk_id), int(user_id), (error or "")[:500], _now()),
            )

    # --- reads --------------------------------------------------------------

    def list_embed_retries(self, *, max_attempts: int = 5, limit: int = 100) -> List[Dict[str, Any]]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT r.chunk_id, r.user_id, r.attempts, c.content "
                "FROM memory_embed_retry r "
                "JOIN memory_episode_chunks c ON c.id=r.chunk_id "
                "WHERE r.attempts < ? AND c.is_deleted=0 "
                "ORDER BY r.updated_at ASC LIMIT ?",
                (int(max_attempts), int(limit)),
            ).fetchall()
            return [
                {
                    "chunk_id": int(r["chunk_id"]),
                    "user_id": int(r["user_id"]),
                    "attempts": int(r["attempts"]),
                    "content": str(r["content"]),
                }
                for r in rows
            ]

    def list_active_with_embeddings(self, user_id: int, limit: int = 5000) -> List[EpisodeChunk]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT c.*, e.embedding_json FROM memory_episode_chunks c "
                "LEFT JOIN memory_episode_embeddings e ON e.chunk_id=c.id "
                "WHERE c.user_id=? AND c.is_deleted=0 "
                "ORDER BY c.created_at DESC LIMIT ?",
                (int(user_id), int(limit)),
            ).fetchall()
            return [self._to_chunk(r) for r in rows]

    def active_embedding_fingerprint(self, user_id: int) -> tuple:
        """Cheap signature so the FAISS cache can detect ingest/forget without hooks."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, "
                "COALESCE(MAX(c.id), 0) AS max_id, "
                "COALESCE(MAX(e.updated_at), 0) AS max_upd "
                "FROM memory_episode_chunks c "
                "JOIN memory_episode_embeddings e ON e.chunk_id=c.id "
                "WHERE c.user_id=? AND c.is_deleted=0",
                (int(user_id),),
            ).fetchone()
            return (int(row["n"] or 0), int(row["max_id"] or 0), int(row["max_upd"] or 0))

    def message_already_indexed(self, message_id: int) -> bool:
        with self._db() as conn:
            row = conn.execute(
                "SELECT 1 FROM memory_episode_chunks WHERE message_id=? AND is_deleted=0 LIMIT 1",
                (int(message_id),),
            ).fetchone()
            return row is not None

    def get_chunks_by_ids(self, chunk_ids: Sequence[int]) -> List[EpisodeChunk]:
        ids = [int(x) for x in chunk_ids]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._db() as conn:
            rows = conn.execute(
                f"SELECT * FROM memory_episode_chunks WHERE id IN ({placeholders}) AND is_deleted=0",
                ids,
            ).fetchall()
            return [self._to_chunk(r) for r in rows]

    def list_conversation_messages(
        self,
        *,
        user_id: int,
        conversation_id: int,
    ) -> List[Dict[str, Any]]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT message_id, role, position, content, created_at, chunk_index "
                "FROM memory_episode_chunks "
                "WHERE user_id=? AND conversation_id=? AND is_deleted=0 "
                "ORDER BY position ASC, chunk_index ASC",
                (int(user_id), int(conversation_id)),
            ).fetchall()
            return _unique_messages(rows)

    def list_recent_messages(self, user_id: int, limit: int = 8) -> List[Dict[str, Any]]:
        """Latest distinct messages for a user (any conversation), oldest→newest."""
        with self._db() as conn:
            rows = conn.execute(
                "SELECT message_id, role, position, content, created_at, chunk_index "
                "FROM memory_episode_chunks "
                "WHERE user_id=? AND is_deleted=0 "
                "ORDER BY created_at DESC, message_id DESC, chunk_index ASC "
                "LIMIT ?",
                (int(user_id), max(1, int(limit)) * 4),
            ).fetchall()
            msgs = _unique_messages(rows)
            msgs = msgs[: max(1, int(limit))]
            msgs.reverse()
            return msgs

    def neighbor_messages(
        self,
        *,
        user_id: int,
        conversation_id: int,
        position: int,
        before: int,
        after: int,
    ) -> List[Dict[str, Any]]:
        with self._db() as conn:
            owned = conn.execute(
                "SELECT 1 FROM memory_episode_chunks "
                "WHERE user_id=? AND conversation_id=? AND is_deleted=0 LIMIT 1",
                (int(user_id), int(conversation_id)),
            ).fetchone()
            if not owned:
                return []
            lo = int(position) - int(before)
            hi = int(position) + int(after)
            rows = conn.execute(
                "SELECT message_id, role, position, content, created_at, chunk_index "
                "FROM memory_episode_chunks "
                "WHERE user_id=? AND conversation_id=? AND is_deleted=0 "
                "AND position BETWEEN ? AND ? "
                "ORDER BY position ASC, chunk_index ASC",
                (int(user_id), int(conversation_id), lo, hi),
            ).fetchall()
            return _unique_messages(rows, mark_seed_at=int(position))

    # --- deletes ------------------------------------------------------------

    def soft_delete_message(self, user_id: int, message_id: int) -> int:
        with self._db() as conn:
            ids = [
                int(r["id"])
                for r in conn.execute(
                    "SELECT id FROM memory_episode_chunks WHERE user_id=? AND message_id=?",
                    (int(user_id), int(message_id)),
                ).fetchall()
            ]
            cur = conn.execute(
                "UPDATE memory_episode_chunks SET is_deleted=1 "
                "WHERE user_id=? AND message_id=?",
                (int(user_id), int(message_id)),
            )
            for cid in ids:
                conn.execute("DELETE FROM memory_episode_embeddings WHERE chunk_id=?", (cid,))
                conn.execute("DELETE FROM memory_embed_retry WHERE chunk_id=?", (cid,))
            self._drop_orphan_profiles(conn, user_id)
            return int(cur.rowcount or 0)

    def soft_delete_by_keyword(self, user_id: int, keyword: str) -> int:
        key = (keyword or "").strip()
        if not key:
            return 0
        with self._db() as conn:
            rows = conn.execute(
                "SELECT id FROM memory_episode_chunks "
                "WHERE user_id=? AND is_deleted=0 AND content LIKE ?",
                (int(user_id), f"%{key}%"),
            ).fetchall()
            removed = 0
            for row in rows:
                cid = int(row["id"])
                conn.execute(
                    "UPDATE memory_episode_chunks SET is_deleted=1 WHERE id=?",
                    (cid,),
                )
                conn.execute("DELETE FROM memory_episode_embeddings WHERE chunk_id=?", (cid,))
                conn.execute("DELETE FROM memory_embed_retry WHERE chunk_id=?", (cid,))
                removed += 1
            self._drop_orphan_profiles(conn, user_id)
            return removed

    def forget_user(self, user_id: int) -> int:
        with self._db() as conn:
            cur = conn.execute(
                "UPDATE memory_episode_chunks SET is_deleted=1 WHERE user_id=?",
                (int(user_id),),
            )
            conn.execute("DELETE FROM support_profile_items WHERE user_id=?", (int(user_id),))
            conn.execute("DELETE FROM support_profile_pending WHERE user_id=?", (int(user_id),))
            conn.execute("DELETE FROM memory_episode_embeddings WHERE user_id=?", (int(user_id),))
            conn.execute("DELETE FROM memory_embed_retry WHERE user_id=?", (int(user_id),))
            return int(cur.rowcount or 0)

    @staticmethod
    def _drop_orphan_profiles(conn: sqlite3.Connection, user_id: int) -> None:
        profiles = conn.execute(
            "SELECT id FROM support_profile_items WHERE user_id=? AND status='active'",
            (int(user_id),),
        ).fetchall()
        for profile in profiles:
            pid = str(profile["id"])
            sources = conn.execute(
                "SELECT message_id FROM support_profile_sources WHERE profile_id=?",
                (pid,),
            ).fetchall()
            if not sources:
                continue
            still_alive = 0
            for src in sources:
                hit = conn.execute(
                    "SELECT 1 FROM memory_episode_chunks "
                    "WHERE user_id=? AND message_id=? AND is_deleted=0 LIMIT 1",
                    (int(user_id), int(src["message_id"])),
                ).fetchone()
                if hit:
                    still_alive += 1
            if still_alive == 0:
                conn.execute(
                    "UPDATE support_profile_items SET status='deleted', updated_at=? WHERE id=?",
                    (_now(), pid),
                )

    # --- stats / checkpoints ------------------------------------------------

    def count_active(self, user_id: int) -> int:
        with self._db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM memory_episode_chunks "
                "WHERE user_id=? AND is_deleted=0",
                (int(user_id),),
            ).fetchone()
            return int(row["c"] if row else 0)

    def index_stats(self, user_id: int | None = None) -> Dict[str, int]:
        with self._db() as conn:
            if user_id is None:
                chunks = conn.execute(
                    "SELECT COUNT(*) AS c FROM memory_episode_chunks WHERE is_deleted=0"
                ).fetchone()["c"]
                emb = conn.execute(
                    "SELECT COUNT(*) AS c FROM memory_episode_embeddings e "
                    "JOIN memory_episode_chunks c ON c.id=e.chunk_id WHERE c.is_deleted=0"
                ).fetchone()["c"]
                retry = conn.execute("SELECT COUNT(*) AS c FROM memory_embed_retry").fetchone()["c"]
            else:
                uid = int(user_id)
                chunks = conn.execute(
                    "SELECT COUNT(*) AS c FROM memory_episode_chunks "
                    "WHERE user_id=? AND is_deleted=0",
                    (uid,),
                ).fetchone()["c"]
                emb = conn.execute(
                    "SELECT COUNT(*) AS c FROM memory_episode_embeddings e "
                    "JOIN memory_episode_chunks c ON c.id=e.chunk_id "
                    "WHERE c.user_id=? AND c.is_deleted=0",
                    (uid,),
                ).fetchone()["c"]
                retry = conn.execute(
                    "SELECT COUNT(*) AS c FROM memory_embed_retry WHERE user_id=?",
                    (uid,),
                ).fetchone()["c"]
            return {"chunks": int(chunks), "embeddings": int(emb), "embed_retries": int(retry)}

    def get_checkpoint(self, key: str) -> Optional[str]:
        with self._db() as conn:
            row = conn.execute(
                "SELECT value FROM memory_backfill_checkpoint WHERE key=?",
                (key,),
            ).fetchone()
            return str(row["value"]) if row else None

    def set_checkpoint(self, key: str, value: str) -> None:
        with self._db() as conn:
            conn.execute(
                "INSERT INTO memory_backfill_checkpoint(key, value, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, _now()),
            )

    @staticmethod
    def _to_chunk(row: sqlite3.Row) -> EpisodeChunk:
        emb = None
        raw = row["embedding_json"] if "embedding_json" in row.keys() else None
        if raw:
            try:
                emb = [float(x) for x in json.loads(raw)]
            except Exception:
                emb = None
        return EpisodeChunk(
            id=int(row["id"]),
            user_id=int(row["user_id"]),
            conversation_id=int(row["conversation_id"]),
            message_id=int(row["message_id"]),
            role=str(row["role"]),
            position=int(row["position"]),
            chunk_index=int(row["chunk_index"]),
            content=str(row["content"]),
            created_at=int(row["created_at"]),
            is_deleted=bool(int(row["is_deleted"])),
            embedding=emb,
        )
