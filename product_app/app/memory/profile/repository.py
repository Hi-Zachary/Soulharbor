"""SQLite helpers for consent profile rows."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

from product_app.app.memory.models import PendingProfile, ProfileItem

# How many unanswered assistant proposals to keep per user.
_MAX_PENDING = 8


class ProfileStore:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create(
        self,
        *,
        user_id: int,
        content: str,
        origin: str,
        source_message_ids: Sequence[int],
    ) -> ProfileItem:
        profile_id = uuid.uuid4().hex
        now = int(time.time())
        cleaned = content.strip()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO support_profile_items"
                "(id, user_id, content, origin, status, created_at, updated_at) "
                "VALUES(?,?,?,?, 'active', ?, ?)",
                (profile_id, int(user_id), cleaned, origin, now, now),
            )
            for mid in source_message_ids:
                conn.execute(
                    "INSERT OR IGNORE INTO support_profile_sources(profile_id, message_id) "
                    "VALUES(?,?)",
                    (profile_id, int(mid)),
                )
        return ProfileItem(
            id=profile_id,
            user_id=int(user_id),
            content=cleaned,
            origin=origin,
            source_message_ids=[int(x) for x in source_message_ids],
            status="active",
            created_at=now,
            updated_at=now,
        )

    def list_active(self, user_id: int) -> List[ProfileItem]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM support_profile_items "
                "WHERE user_id=? AND status='active' ORDER BY updated_at DESC",
                (int(user_id),),
            ).fetchall()
            items: List[ProfileItem] = []
            for row in rows:
                sources = conn.execute(
                    "SELECT message_id FROM support_profile_sources WHERE profile_id=?",
                    (str(row["id"]),),
                ).fetchall()
                items.append(
                    ProfileItem(
                        id=str(row["id"]),
                        user_id=int(row["user_id"]),
                        content=str(row["content"]),
                        origin=str(row["origin"]),
                        source_message_ids=[int(s["message_id"]) for s in sources],
                        status=str(row["status"]),
                        created_at=int(row["created_at"]),
                        updated_at=int(row["updated_at"]),
                    )
                )
            return items

    def soft_delete(self, user_id: int, profile_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE support_profile_items SET status='deleted', updated_at=? "
                "WHERE id=? AND user_id=?",
                (int(time.time()), str(profile_id), int(user_id)),
            )
            return int(cur.rowcount or 0) > 0

    def soft_delete_matching(self, user_id: int, keyword: str) -> int:
        key = (keyword or "").strip()
        if not key:
            return 0
        removed = 0
        now = int(time.time())
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, content FROM support_profile_items "
                "WHERE user_id=? AND status='active'",
                (int(user_id),),
            ).fetchall()
            for row in rows:
                body = str(row["content"])
                if key in body or body in key:
                    conn.execute(
                        "UPDATE support_profile_items SET status='deleted', updated_at=? "
                        "WHERE id=?",
                        (now, str(row["id"])),
                    )
                    removed += 1
        return removed

    def update_content(self, user_id: int, profile_id: str, content: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE support_profile_items SET content=?, updated_at=? "
                "WHERE id=? AND user_id=? AND status='active'",
                (content.strip(), int(time.time()), str(profile_id), int(user_id)),
            )
            return int(cur.rowcount or 0) > 0

    def add_pending(
        self, *, user_id: int, content: str, source_message_ids: Sequence[int]
    ) -> PendingProfile:
        """Enqueue a proposal. Keeps a per-user queue (does not wipe older pendings)."""
        pending_id = uuid.uuid4().hex
        now = int(time.time())
        cleaned = content.strip()
        sources = [int(x) for x in source_message_ids]
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO support_profile_pending"
                "(id, user_id, content, source_json, created_at) VALUES(?,?,?,?,?)",
                (pending_id, int(user_id), cleaned, json.dumps(sources), now),
            )
            # Cap queue length: drop oldest beyond max.
            rows = conn.execute(
                "SELECT id FROM support_profile_pending WHERE user_id=? "
                "ORDER BY created_at DESC",
                (int(user_id),),
            ).fetchall()
            for stale in rows[_MAX_PENDING:]:
                conn.execute(
                    "DELETE FROM support_profile_pending WHERE id=?",
                    (str(stale["id"]),),
                )
        return PendingProfile(
            id=pending_id,
            user_id=int(user_id),
            content=cleaned,
            source_message_ids=sources,
            created_at=now,
        )

    def pop_pending(self, user_id: int) -> Optional[PendingProfile]:
        """Pop the newest pending item only."""
        items = self.pop_all_pending(user_id, limit=1)
        return items[0] if items else None

    def pop_all_pending(
        self, user_id: int, limit: int | None = None
    ) -> List[PendingProfile]:
        """Pop pending proposals oldest-first (stable confirm order). limit=None → all."""
        with self._conn() as conn:
            sql = (
                "SELECT * FROM support_profile_pending WHERE user_id=? "
                "ORDER BY created_at ASC"
            )
            params: list = [int(user_id)]
            if limit is not None:
                sql += " LIMIT ?"
                params.append(int(limit))
            rows = conn.execute(sql, params).fetchall()
            out: List[PendingProfile] = []
            for row in rows:
                conn.execute(
                    "DELETE FROM support_profile_pending WHERE id=?", (str(row["id"]),)
                )
                sources = json.loads(str(row["source_json"]) or "[]")
                out.append(
                    PendingProfile(
                        id=str(row["id"]),
                        user_id=int(row["user_id"]),
                        content=str(row["content"]),
                        source_message_ids=[int(x) for x in sources],
                        created_at=int(row["created_at"]),
                    )
                )
            return out

    def peek_pending(self, user_id: int) -> Optional[PendingProfile]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM support_profile_pending WHERE user_id=? "
                "ORDER BY created_at DESC LIMIT 1",
                (int(user_id),),
            ).fetchone()
            if not row:
                return None
            sources = json.loads(str(row["source_json"]) or "[]")
            return PendingProfile(
                id=str(row["id"]),
                user_id=int(row["user_id"]),
                content=str(row["content"]),
                source_message_ids=[int(x) for x in sources],
                created_at=int(row["created_at"]),
            )

    def list_pending(self, user_id: int, limit: int = 16) -> List[PendingProfile]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM support_profile_pending WHERE user_id=? "
                "ORDER BY created_at ASC LIMIT ?",
                (int(user_id), int(limit)),
            ).fetchall()
            out: List[PendingProfile] = []
            for row in rows:
                sources = json.loads(str(row["source_json"]) or "[]")
                out.append(
                    PendingProfile(
                        id=str(row["id"]),
                        user_id=int(row["user_id"]),
                        content=str(row["content"]),
                        source_message_ids=[int(x) for x in sources],
                        created_at=int(row["created_at"]),
                    )
                )
            return out

    def count_pending(self, user_id: int) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM support_profile_pending WHERE user_id=?",
                (int(user_id),),
            ).fetchone()
            return int(row["c"] if row else 0)

    def bump_llm_uningested(self, user_id: int) -> int:
        """Count a newly ingested message toward the next LLM-propose batch."""
        now = int(time.time())
        with self._conn() as conn:
            row = conn.execute(
                "SELECT uningested_messages FROM support_profile_llm_state WHERE user_id=?",
                (int(user_id),),
            ).fetchone()
            if not row:
                conn.execute(
                    "INSERT INTO support_profile_llm_state"
                    "(user_id, uningested_messages, batch_started_at, "
                    "last_attempt_at, last_attempt_message_id) "
                    "VALUES(?, 1, ?, 0, 0)",
                    (int(user_id), now),
                )
                return 1
            prev = int(row["uningested_messages"] or 0)
            if prev <= 0:
                conn.execute(
                    "UPDATE support_profile_llm_state SET "
                    "uningested_messages = 1, batch_started_at = ? WHERE user_id=?",
                    (now, int(user_id)),
                )
                return 1
            conn.execute(
                "UPDATE support_profile_llm_state SET "
                "uningested_messages = uningested_messages + 1 WHERE user_id=?",
                (int(user_id),),
            )
            return prev + 1

    def should_attempt_llm_propose(
        self,
        user_id: int,
        *,
        trigger_messages: int,
        trigger_age_sec: int,
        now: int | None = None,
    ) -> bool:
        """True when uningested >= N or batch age exceeded (MemMachine-like)."""
        ts = int(now if now is not None else time.time())
        min_msgs = max(1, int(trigger_messages))
        max_age = max(0, int(trigger_age_sec))
        with self._conn() as conn:
            row = conn.execute(
                "SELECT uningested_messages, batch_started_at FROM support_profile_llm_state "
                "WHERE user_id=?",
                (int(user_id),),
            ).fetchone()
            if not row:
                return False
            uningested = int(row["uningested_messages"] or 0)
            started = int(row["batch_started_at"] or 0)
            if uningested <= 0:
                return False
            if uningested >= min_msgs:
                return True
            if max_age > 0 and started > 0 and (ts - started) >= max_age:
                return True
            return False

    def mark_llm_propose_attempted(self, user_id: int, message_id: int) -> None:
        """Reset uningested counter after a propose attempt (even if empty)."""
        now = int(time.time())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO support_profile_llm_state"
                "(user_id, uningested_messages, batch_started_at, "
                "last_attempt_at, last_attempt_message_id) "
                "VALUES(?, 0, 0, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET "
                "uningested_messages = 0, "
                "batch_started_at = 0, "
                "last_attempt_at = excluded.last_attempt_at, "
                "last_attempt_message_id = excluded.last_attempt_message_id",
                (int(user_id), now, int(message_id)),
            )
