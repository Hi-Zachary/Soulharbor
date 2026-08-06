"""SQLite helpers for long-term profile rows (ownership + capacity in code)."""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

from product_app.app.memory.context.profile_formatter import render_user_profile
from product_app.app.memory.models import PendingProfile, ProfileItem
from product_app.app.memory.profile.operations import (
    AppliedProfileChanges,
    ProfileOperation,
    filter_profile_operations,
)
from product_app.app.memory.token_utils import TokenCounter, count_tokens

logger = logging.getLogger(__name__)

_MAX_PENDING = 8


def _row_to_profile(
    row: sqlite3.Row, source_ids: Sequence[int] | None = None
) -> ProfileItem:
    return ProfileItem(
        id=str(row["id"]),
        user_id=int(row["user_id"]),
        content=str(row["content"]),
        origin=str(row["origin"]),
        source_message_ids=[int(x) for x in (source_ids or [])],
        status=str(row["status"]),
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
    )


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

    @contextmanager
    def _write_tx(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.isolation_level = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()

    def _sources_for(self, conn: sqlite3.Connection, profile_id: str) -> List[int]:
        rows = conn.execute(
            "SELECT message_id FROM support_profile_sources WHERE profile_id=?",
            (str(profile_id),),
        ).fetchall()
        return [int(r["message_id"]) for r in rows]

    def verify_current_user_message(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: int,
        message_id: int,
    ) -> bool:
        row = conn.execute(
            """
            SELECT 1
            FROM memory_blocks
            WHERE message_id = ?
              AND user_id = ?
              AND role = 'user'
              AND is_deleted = 0
            LIMIT 1
            """,
            (int(message_id), int(user_id)),
        ).fetchone()
        return row is not None

    def _list_active_in_conn(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: int,
        limit: int | None = None,
        order_by: str = "created_at ASC, id ASC",
    ) -> List[ProfileItem]:
        allowed = {
            "created_at ASC, id ASC",
            "updated_at DESC, created_at DESC",
            "created_at DESC, id DESC",
        }
        order = order_by if order_by in allowed else "created_at ASC, id ASC"
        sql = (
            "SELECT * FROM support_profile_items "
            f"WHERE user_id=? AND status='active' ORDER BY {order}"
        )
        params: list = [int(user_id)]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, int(limit)))
        rows = conn.execute(sql, params).fetchall()
        return [
            _row_to_profile(row, self._sources_for(conn, str(row["id"])))
            for row in rows
        ]

    def list_active(
        self,
        user_id: int,
        limit: int | None = None,
        order_by: str = "created_at ASC, id ASC",
    ) -> List[ProfileItem]:
        with self._conn() as conn:
            return self._list_active_in_conn(
                conn, user_id=user_id, limit=limit, order_by=order_by
            )

    def list_active_with_repair(
        self,
        *,
        user_id: int,
        max_active: int = 20,
        max_block_tokens: int = 640,
        token_counter: TokenCounter = None,
    ) -> List[ProfileItem]:
        with self._write_tx() as conn:
            self._enforce_profile_limits(
                conn,
                user_id=user_id,
                max_active=max_active,
                max_block_tokens=max_block_tokens,
                token_counter=token_counter,
            )
            return self._list_active_in_conn(
                conn,
                user_id=user_id,
                limit=max_active,
                order_by="created_at ASC, id ASC",
            )

    def _insert_profile(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: int,
        content: str,
        source_message_id: int,
    ) -> str:
        profile_id = uuid.uuid4().hex
        now = int(time.time())
        conn.execute(
            """
            INSERT INTO support_profile_items(
                id, user_id, content, origin, status, created_at, updated_at
            )
            VALUES (?, ?, ?, 'maintained', 'active', ?, ?)
            """,
            (profile_id, int(user_id), content, now, now),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO support_profile_sources(profile_id, message_id)
            VALUES (?, ?)
            """,
            (profile_id, int(source_message_id)),
        )
        return profile_id

    def _update_profile(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: int,
        profile_id: str,
        content: str,
        source_message_id: int,
    ) -> bool:
        now = int(time.time())
        cur = conn.execute(
            """
            UPDATE support_profile_items
            SET content = ?, origin = 'maintained', updated_at = ?
            WHERE id = ?
              AND user_id = ?
              AND status = 'active'
            """,
            (content, now, profile_id, int(user_id)),
        )
        if int(cur.rowcount or 0) != 1:
            return False
        conn.execute(
            """
            INSERT OR IGNORE INTO support_profile_sources(profile_id, message_id)
            VALUES (?, ?)
            """,
            (profile_id, int(source_message_id)),
        )
        return True

    def _delete_profile(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: int,
        profile_id: str,
        source_message_id: int,
    ) -> bool:
        now = int(time.time())
        cur = conn.execute(
            """
            UPDATE support_profile_items
            SET status = 'deleted', updated_at = ?
            WHERE id = ?
              AND user_id = ?
              AND status = 'active'
            """,
            (now, profile_id, int(user_id)),
        )
        if int(cur.rowcount or 0) != 1:
            return False
        conn.execute(
            """
            INSERT OR IGNORE INTO support_profile_sources(profile_id, message_id)
            VALUES (?, ?)
            """,
            (profile_id, int(source_message_id)),
        )
        return True

    def soft_delete(self, user_id: int, profile_id: str) -> bool:
        with self._write_tx() as conn:
            cur = conn.execute(
                """
                UPDATE support_profile_items
                SET status = 'deleted', updated_at = ?
                WHERE id = ? AND user_id = ? AND status = 'active'
                """,
                (int(time.time()), str(profile_id), int(user_id)),
            )
            return int(cur.rowcount or 0) == 1

    def soft_delete_user(self, user_id: int) -> int:
        with self._write_tx() as conn:
            now = int(time.time())
            cur = conn.execute(
                """
                UPDATE support_profile_items
                SET status='deleted', updated_at=?
                WHERE user_id=? AND status='active'
                """,
                (now, int(user_id)),
            )
            conn.execute(
                "DELETE FROM support_profile_pending WHERE user_id=?",
                (int(user_id),),
            )
            conn.execute(
                "DELETE FROM support_profile_llm_state WHERE user_id=?",
                (int(user_id),),
            )
            conn.execute(
                "DELETE FROM support_profile_maintenance_runs WHERE user_id=?",
                (int(user_id),),
            )
            return int(cur.rowcount or 0)

    def soft_delete_matching(self, user_id: int, keyword: str) -> int:
        key = (keyword or "").strip()
        if not key:
            return 0
        removed = 0
        with self._write_tx() as conn:
            rows = conn.execute(
                "SELECT id, content FROM support_profile_items "
                "WHERE user_id=? AND status='active'",
                (int(user_id),),
            ).fetchall()
            for row in rows:
                body = str(row["content"])
                if key in body or body in key:
                    cur = conn.execute(
                        """
                        UPDATE support_profile_items
                        SET status='deleted', updated_at=?
                        WHERE id=? AND user_id=? AND status='active'
                        """,
                        (int(time.time()), str(row["id"]), int(user_id)),
                    )
                    removed += int(cur.rowcount or 0)
        return removed

    def remove_source_message(self, *, user_id: int, message_id: int) -> List[str]:
        """Drop source links for a message; soft-delete profiles with no live sources."""
        uid = int(user_id)
        mid = int(message_id)
        deleted_ids: List[str] = []
        with self._write_tx() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT s.profile_id AS profile_id
                FROM support_profile_sources s
                JOIN support_profile_items p ON p.id = s.profile_id
                WHERE p.user_id = ? AND s.message_id = ?
                """,
                (uid, mid),
            ).fetchall()
            affected = [str(r["profile_id"]) for r in rows]
            conn.execute(
                "DELETE FROM support_profile_sources WHERE message_id=?",
                (mid,),
            )
            now = int(time.time())
            for pid in affected:
                live = conn.execute(
                    """
                    SELECT 1
                    FROM support_profile_sources s
                    JOIN memory_blocks b
                      ON b.message_id = s.message_id
                     AND b.user_id = ?
                     AND b.is_deleted = 0
                    WHERE s.profile_id = ?
                    LIMIT 1
                    """,
                    (uid, pid),
                ).fetchone()
                if live:
                    continue
                cur = conn.execute(
                    """
                    UPDATE support_profile_items
                    SET status='deleted', updated_at=?
                    WHERE id=? AND user_id=? AND status='active'
                    """,
                    (now, pid, uid),
                )
                if int(cur.rowcount or 0) == 1:
                    deleted_ids.append(pid)
        return deleted_ids

    def has_maintenance_run(
        self, *, user_id: int, message_id: int, content_hash: str
    ) -> bool:
        with self._conn() as conn:
            self._ensure_maintenance_runs(conn)
            row = conn.execute(
                """
                SELECT 1 FROM support_profile_maintenance_runs
                WHERE user_id=? AND message_id=? AND content_hash=?
                LIMIT 1
                """,
                (int(user_id), int(message_id), str(content_hash)),
            ).fetchone()
            return row is not None

    def record_maintenance_run(
        self, *, user_id: int, message_id: int, content_hash: str
    ) -> None:
        with self._write_tx() as conn:
            self._ensure_maintenance_runs(conn)
            conn.execute(
                """
                INSERT OR IGNORE INTO support_profile_maintenance_runs(
                    user_id, message_id, content_hash, completed_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    int(user_id),
                    int(message_id),
                    str(content_hash),
                    int(time.time()),
                ),
            )

    @staticmethod
    def _ensure_maintenance_runs(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS support_profile_maintenance_runs (
              user_id INTEGER NOT NULL,
              message_id INTEGER NOT NULL,
              content_hash TEXT NOT NULL,
              completed_at INTEGER NOT NULL,
              PRIMARY KEY(user_id, message_id, content_hash)
            )
            """
        )

    def update_content(self, user_id: int, profile_id: str, content: str) -> bool:
        with self._write_tx() as conn:
            cur = conn.execute(
                """
                UPDATE support_profile_items
                SET content=?, updated_at=?
                WHERE id=? AND user_id=? AND status='active'
                """,
                (content.strip(), int(time.time()), str(profile_id), int(user_id)),
            )
            return int(cur.rowcount or 0) == 1

    def _enforce_profile_limits(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: int,
        max_active: int,
        max_block_tokens: int,
        token_counter: TokenCounter,
    ) -> List[str]:
        deleted_ids: List[str] = []
        while True:
            profiles = self._list_active_in_conn(
                conn,
                user_id=user_id,
                order_by="created_at ASC, id ASC",
            )
            profile_block = render_user_profile(profiles)
            count_overflow = len(profiles) > int(max_active)
            token_overflow = (
                count_tokens(profile_block, token_counter) > int(max_block_tokens)
            )
            if not count_overflow and not token_overflow:
                break
            if not profiles:
                break
            oldest = profiles[0]
            cur = conn.execute(
                """
                UPDATE support_profile_items
                SET status = 'deleted', updated_at = ?
                WHERE id = ? AND user_id = ? AND status = 'active'
                """,
                (int(time.time()), oldest.id, int(user_id)),
            )
            if int(cur.rowcount or 0) != 1:
                break
            deleted_ids.append(oldest.id)
        return deleted_ids

    def apply_profile_operations(
        self,
        *,
        user_id: int,
        current_user_message_id: int,
        operations: Sequence[ProfileOperation],
        token_counter: TokenCounter = None,
        max_active: int = 20,
        max_block_tokens: int = 640,
        max_operations: int = 3,
        max_chars: int = 64,
        max_tokens: int = 48,
    ) -> AppliedProfileChanges:
        changes = AppliedProfileChanges()
        uid = int(user_id)
        mid = int(current_user_message_id)
        with self._write_tx() as conn:
            if not self.verify_current_user_message(
                conn, user_id=uid, message_id=mid
            ):
                raise ValueError("invalid current user message")

            current_profiles = self._list_active_in_conn(conn, user_id=uid)
            active_by_id = {item.id: item for item in current_profiles}
            valid = filter_profile_operations(
                list(operations),
                active_profiles=active_by_id,
                token_counter=token_counter,
                max_operations=max_operations,
                max_chars=max_chars,
                max_tokens=max_tokens,
            )
            deletes = [op for op in valid if op.op == "delete"]
            updates = [op for op in valid if op.op == "update"]
            adds = [op for op in valid if op.op == "add"]

            for operation in deletes:
                if self._delete_profile(
                    conn,
                    user_id=uid,
                    profile_id=operation.target_id,
                    source_message_id=mid,
                ):
                    changes.deleted_ids.append(operation.target_id)
                else:
                    changes.skipped += 1

            for operation in updates:
                if self._update_profile(
                    conn,
                    user_id=uid,
                    profile_id=operation.target_id,
                    content=operation.content,
                    source_message_id=mid,
                ):
                    changes.updated_ids.append(operation.target_id)
                else:
                    changes.skipped += 1

            for operation in adds:
                pid = self._insert_profile(
                    conn,
                    user_id=uid,
                    content=operation.content,
                    source_message_id=mid,
                )
                changes.added_ids.append(pid)

            pruned = self._enforce_profile_limits(
                conn,
                user_id=uid,
                max_active=max_active,
                max_block_tokens=max_block_tokens,
                token_counter=token_counter,
            )
            changes.pruned_ids.extend(pruned)
        return changes

    # --- legacy pending / batch state (compat) -------------------------------

    def add_pending(
        self, *, user_id: int, content: str, source_message_ids: Sequence[int]
    ) -> PendingProfile:
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
        items = self.pop_all_pending(user_id, limit=1)
        return items[0] if items else None

    def pop_all_pending(
        self, user_id: int, limit: int | None = None
    ) -> List[PendingProfile]:
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

    def bump_llm_pending(self, user_id: int) -> int:
        return 0

    def should_attempt_llm_propose(self, *args, **kwargs) -> bool:
        return True

    def mark_llm_propose_attempted(self, user_id: int, message_id: int) -> None:
        return None
