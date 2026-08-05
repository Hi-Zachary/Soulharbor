"""SQLite-backed trace chunk / embedding store."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from product_app.app.memory.models import Block

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_blocks (
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
CREATE INDEX IF NOT EXISTS idx_blocks_user_created
  ON memory_blocks(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_blocks_conv_pos
  ON memory_blocks(conversation_id, position);
CREATE INDEX IF NOT EXISTS idx_blocks_user_msg
  ON memory_blocks(user_id, message_id);

CREATE TABLE IF NOT EXISTS memory_block_embeddings (
  chunk_id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  embedding_json TEXT NOT NULL,
  updated_at INTEGER NOT NULL,
  FOREIGN KEY(chunk_id) REFERENCES memory_blocks(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_block_emb_user ON memory_block_embeddings(user_id);

CREATE TABLE IF NOT EXISTS memory_embed_retry (
  chunk_id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  updated_at INTEGER NOT NULL,
  FOREIGN KEY(chunk_id) REFERENCES memory_blocks(id) ON DELETE CASCADE
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

-- Batch trigger cursor for profile LLM propose.
CREATE TABLE IF NOT EXISTS support_profile_llm_state (
  user_id INTEGER PRIMARY KEY,
  pending_turns INTEGER NOT NULL DEFAULT 0,
  batch_started_at INTEGER NOT NULL DEFAULT 0,
  last_attempt_at INTEGER NOT NULL DEFAULT 0,
  last_attempt_message_id INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS memory_backfill_checkpoint (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);

-- Window-level strength for decay-aware MMR / usage reinforcement.
CREATE TABLE IF NOT EXISTS memory_window_strength (
  user_id INTEGER NOT NULL,
  window_key TEXT NOT NULL,
  strength REAL NOT NULL DEFAULT 1.0,
  reinforced_at INTEGER NOT NULL,
  last_reinforce_conversation_id INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(user_id, window_key)
);
CREATE INDEX IF NOT EXISTS idx_window_strength_user
  ON memory_window_strength(user_id);
"""


def _now() -> int:
    return int(time.time())


def content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _unique_messages(rows: Sequence[sqlite3.Row], *, mark_anchor_at: int | None = None) -> List[Dict[str, Any]]:
    """Aggregate multi-chunk rows into one full message per message_id."""
    grouped: Dict[int, List[sqlite3.Row]] = {}
    order: List[int] = []

    for row in rows:
        message_id = int(row["message_id"])
        if message_id not in grouped:
            grouped[message_id] = []
            order.append(message_id)
        grouped[message_id].append(row)

    messages: List[Dict[str, Any]] = []
    for message_id in order:
        chunks = sorted(grouped[message_id], key=lambda row: int(row["chunk_index"]))
        first = chunks[0]
        item = {
            "message_id": message_id,
            "role": str(first["role"]),
            "position": int(first["position"]),
            "content": "".join(str(row["content"]) for row in chunks),
            "created_at": int(first["created_at"]),
        }
        if mark_anchor_at is not None:
            item["is_anchor"] = int(first["position"]) == int(mark_anchor_at)
        messages.append(item)
    return messages


class TraceStore:
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
            self._migrate_legacy_names(conn)
            conn.executescript(_SCHEMA)

    @staticmethod
    def _migrate_legacy_names(conn: sqlite3.Connection) -> None:
        """Rename pre-terminology-refresh tables/columns in place."""
        tables = {
            str(r[0])
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "memory_episode_chunks" in tables and "memory_blocks" not in tables:
            conn.execute("ALTER TABLE memory_episode_chunks RENAME TO memory_blocks")
        if "memory_episode_embeddings" in tables and "memory_block_embeddings" not in tables:
            conn.execute(
                "ALTER TABLE memory_episode_embeddings RENAME TO memory_block_embeddings"
            )
        if "support_profile_llm_state" in tables:
            cols = {
                str(r[1])
                for r in conn.execute("PRAGMA table_info(support_profile_llm_state)").fetchall()
            }
            if "uningested_messages" in cols and "pending_turns" not in cols:
                conn.execute(
                    "ALTER TABLE support_profile_llm_state "
                    "RENAME COLUMN uningested_messages TO pending_turns"
                )

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
        pieces = [piece.strip() for piece in chunks if (piece or "").strip()]
        ids: List[int] = []
        with self._db() as conn:
            for idx, piece in enumerate(pieces):
                conn.execute(
                    "INSERT INTO memory_blocks("
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
                    "SELECT id FROM memory_blocks WHERE message_id=? AND chunk_index=?",
                    (int(message_id), int(idx)),
                ).fetchone()
                if row:
                    ids.append(int(row["id"]))

            # Drop leftover higher-index chunks from a previous longer rewrite.
            valid_count = len(pieces)
            stale_rows = conn.execute(
                "SELECT id FROM memory_blocks "
                "WHERE user_id=? AND message_id=? AND chunk_index>=?",
                (int(user_id), int(message_id), int(valid_count)),
            ).fetchall()
            for row in stale_rows:
                chunk_id = int(row["id"])
                conn.execute(
                    "DELETE FROM memory_block_embeddings WHERE chunk_id=?",
                    (chunk_id,),
                )
                conn.execute(
                    "DELETE FROM memory_embed_retry WHERE chunk_id=?",
                    (chunk_id,),
                )
            if stale_rows:
                conn.execute(
                    "DELETE FROM memory_blocks "
                    "WHERE user_id=? AND message_id=? AND chunk_index>=?",
                    (int(user_id), int(message_id), int(valid_count)),
                )
        return ids

    def save_embedding(self, chunk_id: int, user_id: int, embedding: List[float]) -> None:
        with self._db() as conn:
            conn.execute(
                "INSERT INTO memory_block_embeddings(chunk_id, user_id, embedding_json, updated_at) "
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
                "JOIN memory_blocks c ON c.id=r.chunk_id "
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

    def list_active_with_embeddings(self, user_id: int, limit: int = 5000) -> List[Block]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT c.*, e.embedding_json FROM memory_blocks c "
                "LEFT JOIN memory_block_embeddings e ON e.chunk_id=c.id "
                "WHERE c.user_id=? AND c.is_deleted=0 "
                "ORDER BY c.created_at DESC LIMIT ?",
                (int(user_id), int(limit)),
            ).fetchall()
            return [self._to_chunk(r) for r in rows]

    def active_embedding_fingerprint(self, user_id: int) -> tuple:
        """Cheap signature so the FAISS cache can detect ingest/forget without hooks.

        Counts user chunks only — matches the user_only ANN index.
        """
        with self._db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, "
                "COALESCE(MAX(c.id), 0) AS max_id, "
                "COALESCE(MAX(e.updated_at), 0) AS max_upd "
                "FROM memory_blocks c "
                "JOIN memory_block_embeddings e ON e.chunk_id=c.id "
                "WHERE c.user_id=? AND c.is_deleted=0 AND c.role='user'",
                (int(user_id),),
            ).fetchone()
            return (int(row["n"] or 0), int(row["max_id"] or 0), int(row["max_upd"] or 0))

    def message_already_indexed(self, message_id: int) -> bool:
        with self._db() as conn:
            row = conn.execute(
                "SELECT 1 FROM memory_blocks WHERE message_id=? AND is_deleted=0 LIMIT 1",
                (int(message_id),),
            ).fetchone()
            return row is not None

    def get_chunks_by_ids(self, chunk_ids: Sequence[int]) -> List[Block]:
        ids = [int(x) for x in chunk_ids]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._db() as conn:
            rows = conn.execute(
                f"SELECT * FROM memory_blocks WHERE id IN ({placeholders}) AND is_deleted=0",
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
                "FROM memory_blocks "
                "WHERE user_id=? AND conversation_id=? AND is_deleted=0 "
                "ORDER BY position ASC, chunk_index ASC",
                (int(user_id), int(conversation_id)),
            ).fetchall()
            return _unique_messages(rows)

    def list_recent_messages(self, user_id: int, limit: int = 8) -> List[Dict[str, Any]]:
        """Latest distinct messages for a user (any conversation), oldest→newest.

        Fetches N distinct message_ids first, then all chunks for those ids so
        long multi-chunk messages are fully reconstructed.
        """
        n = max(1, int(limit))
        with self._db() as conn:
            id_rows = conn.execute(
                "SELECT message_id FROM ("
                "  SELECT message_id, MAX(created_at) AS ts "
                "  FROM memory_blocks "
                "  WHERE user_id=? AND is_deleted=0 "
                "  GROUP BY message_id "
                "  ORDER BY ts DESC, message_id DESC "
                "  LIMIT ?"
                ")",
                (int(user_id), n),
            ).fetchall()
            if not id_rows:
                return []
            ids_newest_first = [int(r["message_id"]) for r in id_rows]
            ids_chrono = list(reversed(ids_newest_first))
            placeholders = ",".join("?" * len(ids_chrono))
            rows = conn.execute(
                "SELECT message_id, role, position, content, created_at, chunk_index "
                "FROM memory_blocks "
                f"WHERE user_id=? AND is_deleted=0 AND message_id IN ({placeholders}) "
                "ORDER BY created_at ASC, message_id ASC, chunk_index ASC",
                (int(user_id), *ids_chrono),
            ).fetchall()
            msgs = _unique_messages(rows)
            order = {mid: i for i, mid in enumerate(ids_chrono)}
            msgs.sort(key=lambda m: order.get(int(m["message_id"]), 0))
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
                "SELECT 1 FROM memory_blocks "
                "WHERE user_id=? AND conversation_id=? AND is_deleted=0 LIMIT 1",
                (int(user_id), int(conversation_id)),
            ).fetchone()
            if not owned:
                return []
            lo = int(position) - int(before)
            hi = int(position) + int(after)
            rows = conn.execute(
                "SELECT message_id, role, position, content, created_at, chunk_index "
                "FROM memory_blocks "
                "WHERE user_id=? AND conversation_id=? AND is_deleted=0 "
                "AND position BETWEEN ? AND ? "
                "ORDER BY position ASC, chunk_index ASC",
                (int(user_id), int(conversation_id), lo, hi),
            ).fetchall()
            return _unique_messages(rows, mark_anchor_at=int(position))

    # --- deletes ------------------------------------------------------------

    def soft_delete_message(self, user_id: int, message_id: int) -> int:
        with self._db() as conn:
            ids = [
                int(r["id"])
                for r in conn.execute(
                    "SELECT id FROM memory_blocks WHERE user_id=? AND message_id=?",
                    (int(user_id), int(message_id)),
                ).fetchall()
            ]
            cur = conn.execute(
                "UPDATE memory_blocks SET is_deleted=1 "
                "WHERE user_id=? AND message_id=?",
                (int(user_id), int(message_id)),
            )
            for cid in ids:
                conn.execute("DELETE FROM memory_block_embeddings WHERE chunk_id=?", (cid,))
                conn.execute("DELETE FROM memory_embed_retry WHERE chunk_id=?", (cid,))
            self._drop_orphan_profiles(conn, user_id)
            return int(cur.rowcount or 0)

    def soft_delete_by_keyword(self, user_id: int, keyword: str) -> int:
        key = (keyword or "").strip()
        if not key:
            return 0
        with self._db() as conn:
            rows = conn.execute(
                "SELECT id FROM memory_blocks "
                "WHERE user_id=? AND is_deleted=0 AND content LIKE ?",
                (int(user_id), f"%{key}%"),
            ).fetchall()
            removed = 0
            for row in rows:
                cid = int(row["id"])
                conn.execute(
                    "UPDATE memory_blocks SET is_deleted=1 WHERE id=?",
                    (cid,),
                )
                conn.execute("DELETE FROM memory_block_embeddings WHERE chunk_id=?", (cid,))
                conn.execute("DELETE FROM memory_embed_retry WHERE chunk_id=?", (cid,))
                removed += 1
            self._drop_orphan_profiles(conn, user_id)
            return removed

    def forget_user(self, user_id: int) -> int:
        with self._db() as conn:
            cur = conn.execute(
                "UPDATE memory_blocks SET is_deleted=1 WHERE user_id=?",
                (int(user_id),),
            )
            conn.execute("DELETE FROM support_profile_items WHERE user_id=?", (int(user_id),))
            conn.execute("DELETE FROM support_profile_pending WHERE user_id=?", (int(user_id),))
            conn.execute(
                "DELETE FROM support_profile_llm_state WHERE user_id=?", (int(user_id),)
            )
            conn.execute("DELETE FROM memory_block_embeddings WHERE user_id=?", (int(user_id),))
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
                    "SELECT 1 FROM memory_blocks "
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
                "SELECT COUNT(*) AS c FROM memory_blocks "
                "WHERE user_id=? AND is_deleted=0",
                (int(user_id),),
            ).fetchone()
            return int(row["c"] if row else 0)

    def index_stats(self, user_id: int | None = None) -> Dict[str, int]:
        with self._db() as conn:
            if user_id is None:
                chunks = conn.execute(
                    "SELECT COUNT(*) AS c FROM memory_blocks WHERE is_deleted=0"
                ).fetchone()["c"]
                emb = conn.execute(
                    "SELECT COUNT(*) AS c FROM memory_block_embeddings e "
                    "JOIN memory_blocks c ON c.id=e.chunk_id WHERE c.is_deleted=0"
                ).fetchone()["c"]
                retry = conn.execute("SELECT COUNT(*) AS c FROM memory_embed_retry").fetchone()["c"]
            else:
                uid = int(user_id)
                chunks = conn.execute(
                    "SELECT COUNT(*) AS c FROM memory_blocks "
                    "WHERE user_id=? AND is_deleted=0",
                    (uid,),
                ).fetchone()["c"]
                emb = conn.execute(
                    "SELECT COUNT(*) AS c FROM memory_block_embeddings e "
                    "JOIN memory_blocks c ON c.id=e.chunk_id "
                    "WHERE c.user_id=? AND c.is_deleted=0",
                    (uid,),
                ).fetchone()["c"]
                retry = conn.execute(
                    "SELECT COUNT(*) AS c FROM memory_embed_retry WHERE user_id=?",
                    (uid,),
                ).fetchone()["c"]
            return {"chunks": int(chunks), "embeddings": int(emb), "embed_retries": int(retry)}

    # --- window strength (decay / reinforce) --------------------------------

    def get_window_strength(
        self, *, user_id: int, window_key: str
    ) -> Optional[tuple[float, int, int]]:
        with self._db() as conn:
            row = conn.execute(
                "SELECT strength, reinforced_at, last_reinforce_conversation_id "
                "FROM memory_window_strength WHERE user_id=? AND window_key=?",
                (int(user_id), str(window_key)),
            ).fetchone()
            if not row:
                return None
            return (
                float(row["strength"]),
                int(row["reinforced_at"]),
                int(row["last_reinforce_conversation_id"]),
            )

    def get_window_strengths(
        self, *, user_id: int, window_keys: Sequence[str]
    ) -> Dict[str, tuple[float, int]]:
        keys = [str(k) for k in window_keys if k]
        if not keys:
            return {}
        out: Dict[str, tuple[float, int]] = {}
        with self._db() as conn:
            # SQLite has a variable bind limit; batch if needed.
            chunk = 400
            for i in range(0, len(keys), chunk):
                part = keys[i : i + chunk]
                placeholders = ",".join("?" for _ in part)
                rows = conn.execute(
                    "SELECT window_key, strength, reinforced_at "
                    f"FROM memory_window_strength WHERE user_id=? AND window_key IN ({placeholders})",
                    (int(user_id), *part),
                ).fetchall()
                for row in rows:
                    out[str(row["window_key"])] = (
                        float(row["strength"]),
                        int(row["reinforced_at"]),
                    )
        return out

    def upsert_window_strength(
        self,
        *,
        user_id: int,
        window_key: str,
        strength: float,
        reinforced_at: int,
        conversation_id: int,
    ) -> None:
        with self._db() as conn:
            conn.execute(
                "INSERT INTO memory_window_strength("
                "user_id, window_key, strength, reinforced_at, last_reinforce_conversation_id"
                ") VALUES(?,?,?,?,?) "
                "ON CONFLICT(user_id, window_key) DO UPDATE SET "
                "strength=excluded.strength, "
                "reinforced_at=excluded.reinforced_at, "
                "last_reinforce_conversation_id=excluded.last_reinforce_conversation_id",
                (
                    int(user_id),
                    str(window_key),
                    float(strength),
                    int(reinforced_at),
                    int(conversation_id),
                ),
            )

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
    def _to_chunk(row: sqlite3.Row) -> Block:
        emb = None
        raw = row["embedding_json"] if "embedding_json" in row.keys() else None
        if raw:
            try:
                emb = [float(x) for x in json.loads(raw)]
            except Exception:
                emb = None
        return Block(
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
