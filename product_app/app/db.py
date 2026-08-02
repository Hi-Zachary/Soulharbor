from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_sessions (
  token TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  last_seen INTEGER NOT NULL,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sid TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL DEFAULT '',
  user_id INTEGER,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id INTEGER NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conv_time ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id INTEGER NOT NULL,
  type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_conv_time ON events(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS turn_metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  is_consult INTEGER NOT NULL,
  route TEXT NOT NULL,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_turn_metrics_time ON turn_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_turn_metrics_conv_time ON turn_metrics(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS memory_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  action TEXT NOT NULL,
  memory_id INTEGER,
  detail_json TEXT NOT NULL DEFAULT '{}',
  created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_events_user_time ON memory_events(user_id, created_at);

CREATE TABLE IF NOT EXISTS triage_records (
  conversation_id INTEGER PRIMARY KEY,
  status TEXT NOT NULL DEFAULT 'pending',
  assignee TEXT NOT NULL DEFAULT '',
  note TEXT NOT NULL DEFAULT '',
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_triage_records_status_updated ON triage_records(status, updated_at);
"""


# Pre-ER / intermediate long-term memory schemas. Trace + Profile live in the
# same sqlite file but are created by TraceStore.init().
_LEGACY_DROP_TABLES = (
    "user_memory_digests",
    "chat_chunks",
    "memory_consolidation_runs",
    "user_memories_fts",  # FTS5 virtual table (SQLite drops shadow tables with it)
    "user_memories",
    "memory_embeddings",
    "memory_index_meta",
    "memory_nodes",
    "memory_edges",
    "memory_node_embeddings",
)

_TURN_METRICS_DROP_COLUMNS = (
    "risk",
    "emotion",
    "risk_prob",
    "emotion_prob",
)

def _now_ts() -> int:
    return int(time.time())


def _hash_user_id(user_id: int | None) -> str:
    if user_id is None:
        return "anon"
    h = hashlib.sha256(str(int(user_id)).encode()).hexdigest()[:8]
    return f"u_{h}"


_TRIAGE_STATUSES = ("pending", "tracking", "resolved", "ignored")


@dataclass(frozen=True)
class StoredMessage:
    role: str
    content: str
    created_at: int


@dataclass(frozen=True)
class StoredMessageWithId:
    id: int
    role: str
    content: str
    created_at: int


@dataclass(frozen=True)
class AppendedMessage:
    message_id: int
    conversation_id: int
    role: str
    content: str
    position: int
    created_at: int


class SQLiteStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _migrate_columns(self, table: str, columns: list[tuple[str, str]]) -> None:
        conn = self._connect()
        try:
            existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for col_name, col_def in columns:
                if col_name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")
            conn.commit()
        finally:
            conn.close()

    def _drop_column_if_exists(self, table: str, column: str) -> None:
        conn = self._connect()
        try:
            cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            if column in cols:
                conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
                conn.commit()
        finally:
            conn.close()

    def _drop_table_if_exists(self, table: str) -> None:
        conn = self._connect()
        try:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(SCHEMA_SQL)
            # Lightweight schema migration for older DBs.
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()}
            if "title" not in cols:
                conn.execute("ALTER TABLE conversations ADD COLUMN title TEXT NOT NULL DEFAULT ''")
            if "user_id" not in cols:
                conn.execute("ALTER TABLE conversations ADD COLUMN user_id INTEGER")
            if "summary" not in cols:
                conn.execute("ALTER TABLE conversations ADD COLUMN summary TEXT NOT NULL DEFAULT ''")
            if "summary_updated_at" not in cols:
                conn.execute("ALTER TABLE conversations ADD COLUMN summary_updated_at INTEGER")
            if "last_summarized_msg_id" not in cols:
                conn.execute("ALTER TABLE conversations ADD COLUMN last_summarized_msg_id INTEGER")
            ucols = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
            if "memory_enabled" not in ucols:
                conn.execute("ALTER TABLE users ADD COLUMN memory_enabled INTEGER NOT NULL DEFAULT 1")
            conn.commit()
        finally:
            conn.close()
        for col in _TURN_METRICS_DROP_COLUMNS:
            self._drop_column_if_exists("turn_metrics", col)
        self._drop_column_if_exists("conversations", "last_extracted_msg_id")
        for table in _LEGACY_DROP_TABLES:
            self._drop_table_if_exists(table)

    def _table_exists(self, table: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def get_or_create_conversation(self, sid: str, *, user_id: Optional[int] = None) -> int:
        conn = self._connect()
        try:
            now = _now_ts()
            row = conn.execute("SELECT id, user_id FROM conversations WHERE sid = ?", (sid,)).fetchone()
            if row:
                if user_id is not None and row["user_id"] is None:
                    conn.execute("UPDATE conversations SET user_id=?, updated_at=? WHERE id=?", (user_id, now, int(row["id"])))
                else:
                    conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, int(row["id"])))
                conn.commit()
                return int(row["id"])
            cur = conn.execute(
                "INSERT INTO conversations(sid, title, user_id, created_at, updated_at) VALUES(?,?,?,?,?)",
                (sid, "", user_id, now, now),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def append_message(self, conversation_id: int, role: str, content: str) -> AppendedMessage:
        conn = self._connect()
        try:
            now = _now_ts()
            cur = conn.execute(
                "INSERT INTO messages(conversation_id, role, content, created_at) VALUES(?,?,?,?)",
                (conversation_id, role, content, now),
            )
            message_id = int(cur.lastrowid)
            pos_row = conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE conversation_id=? AND id<=?",
                (int(conversation_id), message_id),
            ).fetchone()
            position = int(pos_row["c"] if pos_row else 1)
            conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
            conn.commit()
            return AppendedMessage(
                message_id=message_id,
                conversation_id=int(conversation_id),
                role=str(role),
                content=str(content),
                position=position,
                created_at=now,
            )
        finally:
            conn.close()

    def get_message_record(self, message_id: int) -> Optional[AppendedMessage]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, conversation_id, role, content, created_at FROM messages WHERE id=?",
                (int(message_id),),
            ).fetchone()
            if not row:
                return None
            pos_row = conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE conversation_id=? AND id<=?",
                (int(row["conversation_id"]), int(row["id"])),
            ).fetchone()
            return AppendedMessage(
                message_id=int(row["id"]),
                conversation_id=int(row["conversation_id"]),
                role=str(row["role"]),
                content=str(row["content"]),
                position=int(pos_row["c"] if pos_row else 1),
                created_at=int(row["created_at"]),
            )
        finally:
            conn.close()

    def update_message_content(self, message_id: int, content: str) -> None:
        conn = self._connect()
        try:
            now = _now_ts()
            row = conn.execute(
                "SELECT conversation_id FROM messages WHERE id=?",
                (message_id,),
            ).fetchone()
            if not row:
                return
            conn.execute(
                "UPDATE messages SET content=? WHERE id=?",
                (content, message_id),
            )
            conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (now, int(row["conversation_id"])),
            )
            conn.commit()
        finally:
            conn.close()

    def append_event(self, conversation_id: int, type_: str, payload: Dict[str, Any]) -> None:
        conn = self._connect()
        try:
            now = _now_ts()
            conn.execute(
                "INSERT INTO events(conversation_id, type, payload_json, created_at) VALUES(?,?,?,?)",
                (conversation_id, type_, json.dumps(payload, ensure_ascii=False), now),
            )
            conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
            conn.commit()
        finally:
            conn.close()

    def append_turn_metrics(
        self,
        conversation_id: int,
        *,
        is_consult: int,
        route: str,
    ) -> None:
        conn = self._connect()
        try:
            now = _now_ts()
            conn.execute(
                "INSERT INTO turn_metrics(conversation_id, created_at, is_consult, route) "
                "VALUES(?,?,?,?)",
                (
                    int(conversation_id),
                    now,
                    int(is_consult),
                    str(route or ""),
                ),
            )
            conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
            conn.commit()
        finally:
            conn.close()

    def list_messages(self, conversation_id: int, limit: int = 200) -> List[StoredMessage]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE conversation_id=? ORDER BY id ASC LIMIT ?",
                (conversation_id, int(limit)),
            ).fetchall()
            return [StoredMessage(role=str(r["role"]), content=str(r["content"]), created_at=int(r["created_at"])) for r in rows]
        finally:
            conn.close()

    def list_messages_with_ids(self, conversation_id: int, limit: int = 500) -> List[StoredMessageWithId]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, role, content, created_at FROM messages WHERE conversation_id=? ORDER BY id ASC LIMIT ?",
                (conversation_id, int(limit)),
            ).fetchall()
            return [
                StoredMessageWithId(
                    id=int(r["id"]),
                    role=str(r["role"]),
                    content=str(r["content"]),
                    created_at=int(r["created_at"]),
                )
                for r in rows
            ]
        finally:
            conn.close()

    def list_messages_since(self, conversation_id: int, since_msg_id: int, limit: int = 200) -> List[StoredMessageWithId]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, role, content, created_at FROM messages WHERE conversation_id=? AND id > ? ORDER BY id ASC LIMIT ?",
                (conversation_id, int(since_msg_id), int(limit)),
            ).fetchall()
            return [
                StoredMessageWithId(
                    id=int(r["id"]),
                    role=str(r["role"]),
                    content=str(r["content"]),
                    created_at=int(r["created_at"]),
                )
                for r in rows
            ]
        finally:
            conn.close()

    def count_messages(self, conversation_id: int) -> int:
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) as c FROM messages WHERE conversation_id=?", (conversation_id,)).fetchone()
            return int(row["c"])
        finally:
            conn.close()

    def clear_conversation(self, sid: str) -> None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT id FROM conversations WHERE sid=?", (sid,)).fetchone()
            if not row:
                return
            cid = int(row["id"])
            conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
            conn.execute("DELETE FROM events WHERE conversation_id=?", (cid,))
            conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now_ts(), cid))
            conn.commit()
        finally:
            conn.close()

    def set_title(self, sid: str, title: str) -> None:
        title = (title or "").strip()
        conn = self._connect()
        try:
            now = _now_ts()
            conn.execute("UPDATE conversations SET title=?, updated_at=? WHERE sid=?", (title, now, sid))
            conn.commit()
        finally:
            conn.close()

    def get_title(self, sid: str) -> str:
        conn = self._connect()
        try:
            row = conn.execute("SELECT title FROM conversations WHERE sid=?", (sid,)).fetchone()
            if not row:
                return ""
            return str(row["title"] or "")
        finally:
            conn.close()

    def delete_conversation(self, sid: str) -> None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT id FROM conversations WHERE sid=?", (sid,)).fetchone()
            if not row:
                return
            cid = int(row["id"])
            conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
            conn.execute("DELETE FROM events WHERE conversation_id=?", (cid,))
            conn.execute("DELETE FROM conversations WHERE id=?", (cid,))
            conn.commit()
        finally:
            conn.close()

    def delete_empty_conversations_for_user(
        self, user_id: int, *, keep_sid: Optional[str] = None
    ) -> int:
        """Remove draft conversations (no messages). Optionally keep one sid."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, sid FROM conversations WHERE user_id=? "
                "AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id=conversations.id)",
                (int(user_id),),
            ).fetchall()
            deleted = 0
            keep = (keep_sid or "").strip()
            for r in rows:
                if keep and str(r["sid"]) == keep:
                    continue
                cid = int(r["id"])
                conn.execute("DELETE FROM events WHERE conversation_id=?", (cid,))
                conn.execute("DELETE FROM conversations WHERE id=?", (cid,))
                deleted += 1
            if deleted:
                conn.commit()
            return deleted
        finally:
            conn.close()

    def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List conversations that have at least one message (drafts stay hidden)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT c.sid, c.title, c.created_at, c.updated_at, c.user_id, "
                "  (SELECT MIN(m.created_at) FROM messages m WHERE m.conversation_id=c.id) as started_at, "
                "  (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) as message_count "
                "FROM conversations c "
                "WHERE EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id=c.id) "
                "ORDER BY c.updated_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                started = r["started_at"]
                out.append(
                    {
                        "sid": str(r["sid"]),
                        "title": str(r["title"] or ""),
                        "created_at": int(r["created_at"]),
                        "updated_at": int(r["updated_at"]),
                        "started_at": int(started) if started is not None else int(r["updated_at"]),
                        "message_count": int(r["message_count"] or 0),
                        "user_id": (None if r["user_id"] is None else int(r["user_id"])),
                    }
                )
            return out
        finally:
            conn.close()

    def list_conversations_for_user(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """List this user's conversations that have at least one message."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT c.sid, c.title, c.created_at, c.updated_at, "
                "  (SELECT MIN(m.created_at) FROM messages m WHERE m.conversation_id=c.id) as started_at, "
                "  (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) as message_count "
                "FROM conversations c "
                "WHERE c.user_id=? "
                "AND EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id=c.id) "
                "ORDER BY c.updated_at DESC LIMIT ?",
                (int(user_id), int(limit)),
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                started = r["started_at"]
                out.append(
                    {
                        "sid": str(r["sid"]),
                        "title": str(r["title"] or ""),
                        "created_at": int(r["created_at"]),
                        "updated_at": int(r["updated_at"]),
                        "started_at": int(started) if started is not None else int(r["updated_at"]),
                        "message_count": int(r["message_count"] or 0),
                    }
                )
            return out
        finally:
            conn.close()

    def get_user_activity_stats(self, user_id: int) -> Dict[str, Any]:
        """Conversation / message / recent-route stats for admin profile panels."""
        conn = self._connect()
        try:
            uid = int(user_id)
            stats = conn.execute(
                "SELECT "
                "  (SELECT COUNT(*) FROM conversations WHERE user_id=?) as conversation_count, "
                "  (SELECT COUNT(*) FROM messages WHERE conversation_id IN "
                "     (SELECT id FROM conversations WHERE user_id=?)) as message_count, "
                "  (SELECT MAX(updated_at) FROM conversations WHERE user_id=?) as last_seen",
                (uid, uid, uid),
            ).fetchone()
            recent_metrics = conn.execute(
                "SELECT tm.created_at, tm.route, tm.is_consult "
                "FROM turn_metrics tm JOIN conversations c ON c.id=tm.conversation_id "
                "WHERE c.user_id=? ORDER BY tm.id DESC LIMIT 8",
                (uid,),
            ).fetchall()
            return {
                "conversation_count": int(stats["conversation_count"] or 0),
                "message_count": int(stats["message_count"] or 0),
                "last_seen": int(stats["last_seen"] or 0),
                "recent_metrics": [
                    {
                        "created_at": int(r["created_at"]),
                        "route": str(r["route"] or ""),
                        "is_consult": int(r["is_consult"] or 0),
                    }
                    for r in recent_metrics
                ],
            }
        finally:
            conn.close()

    def get_conversation(self, sid: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, sid, title, user_id, created_at, updated_at, summary, "
                "summary_updated_at, last_summarized_msg_id "
                "FROM conversations WHERE sid=?",
                ((sid or "").strip(),),
            ).fetchone()
            if not row:
                return None
            return {
                "id": int(row["id"]),
                "sid": str(row["sid"]),
                "title": str(row["title"] or ""),
                "user_id": (None if row["user_id"] is None else int(row["user_id"])),
                "created_at": int(row["created_at"]),
                "updated_at": int(row["updated_at"]),
                "summary": str(row["summary"] or ""),
                "summary_updated_at": (None if row["summary_updated_at"] is None else int(row["summary_updated_at"])),
                "last_summarized_msg_id": (
                    None if row["last_summarized_msg_id"] is None else int(row["last_summarized_msg_id"])
                ),
            }
        finally:
            conn.close()

    def get_conversation_summary(self, sid: str) -> str:
        conv = self.get_conversation(sid)
        if not conv:
            return ""
        return str(conv.get("summary") or "")

    def update_conversation_summary(self, sid: str, summary: str, last_msg_id: int) -> None:
        conn = self._connect()
        try:
            now = _now_ts()
            conn.execute(
                "UPDATE conversations SET summary=?, summary_updated_at=?, last_summarized_msg_id=?, updated_at=? WHERE sid=?",
                ((summary or "").strip(), now, int(last_msg_id), now, sid),
            )
            conn.commit()
        finally:
            conn.close()

    # ---- auth ----
    def create_user(self, username: str, password_hash: str) -> int:
        conn = self._connect()
        try:
            now = _now_ts()
            cur = conn.execute(
                "INSERT INTO users(username, password_hash, created_at) VALUES(?,?,?)",
                ((username or "").strip(), password_hash, now),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT id, username, password_hash FROM users WHERE username=?", ((username or "").strip(),)).fetchone()
            if not row:
                return None
            return {"id": int(row["id"]), "username": str(row["username"]), "password_hash": str(row["password_hash"])}
        finally:
            conn.close()

    def create_session(self, token: str, user_id: int) -> None:
        conn = self._connect()
        try:
            now = _now_ts()
            conn.execute(
                "INSERT INTO auth_sessions(token, user_id, created_at, last_seen) VALUES(?,?,?,?)",
                (token, int(user_id), now, now),
            )
            conn.commit()
        finally:
            conn.close()

    def get_user_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT u.id as id, u.username as username FROM auth_sessions s JOIN users u ON u.id=s.user_id WHERE s.token=?",
                ((token or "").strip(),),
            ).fetchone()
            if not row:
                return None
            conn.execute("UPDATE auth_sessions SET last_seen=? WHERE token=?", (_now_ts(), (token or "").strip()))
            conn.commit()
            return {"id": int(row["id"]), "username": str(row["username"])}
        finally:
            conn.close()

    def delete_session(self, token: str) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM auth_sessions WHERE token=?", ((token or "").strip(),))
            conn.commit()
        finally:
            conn.close()

    def get_memory_enabled(self, user_id: int) -> bool:
        if user_id <= 0:
            return False
        conn = self._connect()
        try:
            row = conn.execute("SELECT memory_enabled FROM users WHERE id=?", (int(user_id),)).fetchone()
            if not row:
                return False
            return int(row["memory_enabled"]) != 0
        finally:
            conn.close()

    def set_memory_enabled(self, user_id: int, enabled: bool) -> None:
        if user_id <= 0:
            return
        conn = self._connect()
        try:
            conn.execute("UPDATE users SET memory_enabled=? WHERE id=?", (1 if enabled else 0, int(user_id)))
            conn.commit()
        finally:
            conn.close()

    def append_memory_event(self, user_id: int, action: str, memory_id: Optional[int] = None, detail: Optional[Dict[str, Any]] = None) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO memory_events(user_id, action, memory_id, detail_json, created_at) VALUES(?,?,?,?,?)",
                (int(user_id), action, memory_id, json.dumps(detail or {}, ensure_ascii=False), _now_ts()),
            )
            conn.commit()
        finally:
            conn.close()

    # ---- admin: conversation browsing ----
    def list_conversations_with_metrics(
        self, *, limit: int = 100, offset: int = 0, date_from: int = 0, date_to: int = 0
    ) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            sql = (
                "SELECT c.sid, c.title, c.user_id, c.updated_at, "
                "  (SELECT COUNT(*) FROM messages m2 WHERE m2.conversation_id=c.id) as msg_count, "
                "  tm.is_consult, tm.route "
                "FROM conversations c "
                "LEFT JOIN turn_metrics tm ON tm.id = ("
                "  SELECT t2.id FROM turn_metrics t2 "
                "  WHERE t2.conversation_id=c.id ORDER BY t2.id DESC LIMIT 1"
                ") "
                "WHERE 1=1 "
            )
            params: List[Any] = []
            if date_from:
                sql += " AND c.updated_at >= ?"
                params.append(int(date_from))
            if date_to:
                sql += " AND c.updated_at <= ?"
                params.append(int(date_to))
            sql += " ORDER BY c.updated_at DESC LIMIT ? OFFSET ?"
            params.extend([int(limit), int(offset)])

            rows = conn.execute(sql, params).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                uid = r["user_id"]
                out.append({
                    "sid": str(r["sid"]),
                    "title": str(r["title"] or "新会话"),
                    "user_id_hash": _hash_user_id(uid) if uid else "anon",
                    "user_id": uid,
                    "msg_count": int(r["msg_count"]),
                    "is_consult": int(r["is_consult"] or 0),
                    "route": str(r["route"] or ("consult" if int(r["is_consult"] or 0) else "chat")),
                    "last_active": int(r["updated_at"]),
                })
            return out
        finally:
            conn.close()

    def get_conversation_detail(self, sid: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            conv = conn.execute(
                "SELECT id, sid, title, user_id, created_at, updated_at, summary FROM conversations WHERE sid=?",
                ((sid or "").strip(),),
            ).fetchone()
            if not conv:
                return None

            cid = int(conv["id"])
            msgs = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE conversation_id=? ORDER BY id ASC",
                (cid,),
            ).fetchall()
            metrics = conn.execute(
                "SELECT is_consult, route, created_at "
                "FROM turn_metrics WHERE conversation_id=? ORDER BY id ASC",
                (cid,),
            ).fetchall()

            return {
                "sid": str(conv["sid"]),
                "title": str(conv["title"] or ""),
                "user_id": (None if conv["user_id"] is None else int(conv["user_id"])),
                "user_id_hash": _hash_user_id(conv["user_id"]) if conv["user_id"] else "anon",
                "created_at": int(conv["created_at"]),
                "summary": str(conv["summary"] or ""),
                "messages": [
                    {"role": str(m["role"]), "content": str(m["content"]), "created_at": int(m["created_at"])}
                    for m in msgs
                ],
                "turn_metrics": [
                    {
                        "is_consult": int(t["is_consult"]),
                        "route": str(t["route"]),
                        "created_at": int(t["created_at"]),
                    }
                    for t in metrics
                ],
            }
        finally:
            conn.close()

    # ---- admin: user management ----
    def list_users_with_stats(self, *, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Admin user list. memory_count = ER trace chunks + active profile prefs."""
        has_chunks = self._table_exists("memory_blocks")
        has_profile = self._table_exists("support_profile_items")
        chunk_cnt = (
            "(SELECT COUNT(*) FROM memory_blocks ec "
            "WHERE ec.user_id=u.id AND ec.is_deleted=0)"
            if has_chunks
            else "0"
        )
        profile_cnt = (
            "(SELECT COUNT(*) FROM support_profile_items sp "
            "WHERE sp.user_id=u.id AND sp.status='active')"
            if has_profile
            else "0"
        )
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT u.id as uid, "
                f"  ({chunk_cnt} + {profile_cnt}) as mem_cnt, "
                "  (SELECT COUNT(*) FROM conversations c2 WHERE c2.user_id=u.id) as conv_cnt, "
                "  (SELECT MAX(c2.updated_at) FROM conversations c2 WHERE c2.user_id=u.id) as last_active, "
                "  (SELECT tm.route FROM turn_metrics tm "
                "     JOIN conversations c3 ON c3.id=tm.conversation_id "
                "     WHERE c3.user_id=u.id ORDER BY tm.id DESC LIMIT 1) as recent_route "
                "FROM users u "
                "ORDER BY last_active DESC NULLS LAST LIMIT ? OFFSET ?",
                (int(limit), int(offset)),
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                uid = int(r["uid"])
                out.append({
                    "user_id_hash": _hash_user_id(uid),
                    "user_id": uid,
                    "memory_count": int(r["mem_cnt"]),
                    "conversation_count": int(r["conv_cnt"]),
                    "last_active": (int(r["last_active"]) if r["last_active"] else 0),
                    "recent_route": str(r["recent_route"]) if r["recent_route"] else "",
                })
            return out
        finally:
            conn.close()

    def upsert_triage_record(self, conversation_id: int, *, status: str, assignee: str, note: str) -> None:
        status = str(status or "pending").strip().lower()
        if status not in _TRIAGE_STATUSES:
            status = "pending"
        assignee = str(assignee or "").strip()[:80]
        note = str(note or "").strip()[:4000]
        conn = self._connect()
        try:
            now = _now_ts()
            conn.execute(
                "INSERT INTO triage_records(conversation_id, status, assignee, note, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(conversation_id) DO UPDATE SET "
                "status=excluded.status, assignee=excluded.assignee, note=excluded.note, updated_at=excluded.updated_at",
                (int(conversation_id), status, assignee, note, now, now),
            )
            conn.commit()
        finally:
            conn.close()

    def list_triage_sessions(
        self, *, limit: int = 80, status: str = "all", date_from: int = 0, date_to: int = 0
    ) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            sql = (
                "SELECT c.id, c.sid, c.title, c.user_id, c.created_at, c.updated_at, "
                "tm.route, "
                "COALESCE(tr.status, 'pending') as triage_status, "
                "COALESCE(tr.assignee, '') as triage_assignee, "
                "COALESCE(tr.updated_at, 0) as triage_updated_at "
                "FROM conversations c "
                "LEFT JOIN turn_metrics tm ON tm.id = ("
                "  SELECT t2.id FROM turn_metrics t2 WHERE t2.conversation_id=c.id ORDER BY t2.id DESC LIMIT 1"
                ") "
                "LEFT JOIN triage_records tr ON tr.conversation_id=c.id "
                "WHERE tm.is_consult = 1 "
            )
            params: List[Any] = []
            if status and status != "all":
                sql += " AND COALESCE(tr.status, 'pending') = ?"
                params.append(status)
            if date_from:
                sql += " AND c.updated_at >= ?"
                params.append(int(date_from))
            if date_to:
                sql += " AND c.updated_at <= ?"
                params.append(int(date_to))
            sql += (
                " ORDER BY "
                "CASE COALESCE(tr.status, 'pending') "
                "  WHEN 'pending' THEN 0 "
                "  WHEN 'tracking' THEN 1 "
                "  WHEN 'resolved' THEN 2 "
                "  ELSE 3 END, "
                "COALESCE(tr.updated_at, c.updated_at) DESC "
                "LIMIT ?"
            )
            params.append(int(limit))
            rows = conn.execute(sql, params).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                uid = r["user_id"]
                out.append({
                    "sid": str(r["sid"]),
                    "title": str(r["title"] or "新会话"),
                    "user_id": (None if uid is None else int(uid)),
                    "user_id_hash": _hash_user_id(uid) if uid else "anon",
                    "created_at": int(r["created_at"]),
                    "updated_at": int(r["updated_at"]),
                    "route": str(r["route"] or "consult"),
                    "triage_status": str(r["triage_status"] or "pending"),
                    "triage_assignee": str(r["triage_assignee"] or ""),
                    "triage_updated_at": int(r["triage_updated_at"] or 0),
                })
            return out
        finally:
            conn.close()

    def list_triage_records(
        self, *, limit: int = 100, status: str = "all", date_from: int = 0, date_to: int = 0
    ) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            sql = (
                "SELECT tr.conversation_id, tr.status, tr.assignee, tr.note, tr.created_at, tr.updated_at, "
                "c.sid, c.title, c.user_id, c.updated_at as conv_updated_at "
                "FROM triage_records tr JOIN conversations c ON c.id=tr.conversation_id WHERE 1=1"
            )
            params: List[Any] = []
            if status and status != "all":
                sql += " AND tr.status = ?"
                params.append(status)
            if date_from:
                sql += " AND tr.updated_at >= ?"
                params.append(int(date_from))
            if date_to:
                sql += " AND tr.updated_at <= ?"
                params.append(int(date_to))
            sql += " ORDER BY tr.updated_at DESC LIMIT ?"
            params.append(int(limit))
            rows = conn.execute(sql, params).fetchall()
            return [
                {
                    "sid": str(r["sid"]),
                    "title": str(r["title"] or "新会话"),
                    "user_id_hash": _hash_user_id(r["user_id"]) if r["user_id"] else "anon",
                    "status": str(r["status"]),
                    "assignee": str(r["assignee"] or ""),
                    "note": str(r["note"] or ""),
                    "created_at": int(r["created_at"]),
                    "updated_at": int(r["updated_at"]),
                    "last_active": int(r["conv_updated_at"]),
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_admin_conversation_workspace(self, sid: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            conv = conn.execute(
                "SELECT c.id, c.sid, c.title, c.user_id, c.created_at, c.updated_at, c.summary, "
                "u.username FROM conversations c LEFT JOIN users u ON u.id=c.user_id WHERE c.sid=?",
                ((sid or "").strip(),),
            ).fetchone()
            if not conv:
                return None
            cid = int(conv["id"])
            msgs = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE conversation_id=? ORDER BY id ASC",
                (cid,),
            ).fetchall()
            metrics = conn.execute(
                "SELECT is_consult, route, created_at "
                "FROM turn_metrics WHERE conversation_id=? ORDER BY id DESC",
                (cid,),
            ).fetchall()
            triage = conn.execute(
                "SELECT status, assignee, note, created_at, updated_at FROM triage_records WHERE conversation_id=?",
                (cid,),
            ).fetchone()
            return {
                "conversation": {
                    "id": cid,
                    "sid": str(conv["sid"]),
                    "title": str(conv["title"] or ""),
                    "user_id": (None if conv["user_id"] is None else int(conv["user_id"])),
                    "user_id_hash": _hash_user_id(conv["user_id"]) if conv["user_id"] else "anon",
                    "username": str(conv["username"] or ""),
                    "created_at": int(conv["created_at"]),
                    "updated_at": int(conv["updated_at"]),
                    "summary": str(conv["summary"] or ""),
                },
                "messages": [
                    {"role": str(m["role"]), "content": str(m["content"]), "created_at": int(m["created_at"])}
                    for m in msgs
                ],
                "metrics": [
                    {
                        "is_consult": int(t["is_consult"]),
                        "route": str(t["route"]),
                        "created_at": int(t["created_at"]),
                    }
                    for t in metrics
                ],
                "triage": {
                    "status": str(triage["status"]) if triage else "pending",
                    "assignee": str(triage["assignee"] or "") if triage else "",
                    "note": str(triage["note"] or "") if triage else "",
                    "created_at": int(triage["created_at"]) if triage and triage["created_at"] else 0,
                    "updated_at": int(triage["updated_at"]) if triage and triage["updated_at"] else 0,
                },
            }
        finally:
            conn.close()

    # ---- admin stats (anonymous aggregates) ----
    def stats_today(self, *, date_from: int = 0, date_to: int = 0) -> Dict[str, Any]:
        conn = self._connect()
        try:
            now = int(time.time())
            if date_from and date_to:
                day_start = int(date_from)
                day_end = int(date_to)
            else:
                day_start = int(time.mktime(time.localtime(now)[:3] + (0, 0, 0) + time.localtime(now)[6:]))
                day_end = now

            convs = conn.execute(
                "SELECT COUNT(*) as c FROM conversations WHERE updated_at>=? AND updated_at<=?",
                (day_start, day_end),
            ).fetchone()["c"]
            msgs = conn.execute(
                "SELECT COUNT(*) as c FROM messages WHERE created_at>=? AND created_at<=?",
                (day_start, day_end),
            ).fetchone()["c"]
            users = conn.execute(
                "SELECT COUNT(DISTINCT user_id) as c FROM conversations WHERE updated_at>=? AND updated_at<=? AND user_id IS NOT NULL",
                (day_start, day_end),
            ).fetchone()["c"]
            route_rows = conn.execute(
                "SELECT route, COUNT(*) as c FROM turn_metrics WHERE created_at>=? AND created_at<=? GROUP BY route ORDER BY c DESC",
                (day_start, day_end),
            ).fetchall()
            consult_rows = conn.execute(
                "SELECT COUNT(*) as c FROM turn_metrics WHERE created_at>=? AND created_at<=? AND is_consult=1",
                (day_start, day_end),
            ).fetchone()
            triage_pending = conn.execute(
                "SELECT COUNT(*) as c FROM conversations c "
                "JOIN turn_metrics tm ON tm.id = ("
                "  SELECT t2.id FROM turn_metrics t2 WHERE t2.conversation_id=c.id ORDER BY t2.id DESC LIMIT 1"
                ") "
                "LEFT JOIN triage_records tr ON tr.conversation_id=c.id "
                "WHERE tm.is_consult=1 AND c.updated_at>=? AND c.updated_at<=? "
                "AND COALESCE(tr.status, 'pending') IN ('pending','tracking')",
                (day_start, day_end),
            ).fetchone()["c"]
            triage_processed = conn.execute(
                "SELECT COUNT(*) as c FROM conversations c "
                "JOIN turn_metrics tm ON tm.id = ("
                "  SELECT t2.id FROM turn_metrics t2 WHERE t2.conversation_id=c.id ORDER BY t2.id DESC LIMIT 1"
                ") "
                "JOIN triage_records tr ON tr.conversation_id=c.id "
                "WHERE tm.is_consult=1 AND c.updated_at>=? AND c.updated_at<=? "
                "AND tr.status IN ('resolved','ignored')",
                (day_start, day_end),
            ).fetchone()["c"]
            return {
                "day_start": day_start,
                "day_end": day_end,
                "conversations": int(convs),
                "messages": int(msgs),
                "active_users": int(users),
                "triage_pending": int(triage_pending),
                "triage_processed": int(triage_processed),
                "consult_turns": int(consult_rows["c"] or 0),
                "route_dist": [{"route": str(r["route"]), "count": int(r["c"])} for r in route_rows],
            }
        finally:
            conn.close()
