"""Admin memory_count uses ER trace chunks + active profiles."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from product_app.app.db import SQLiteStore
from product_app.app.memory.store.repository import TraceStore


class AerAdminStatsTests(unittest.TestCase):
    def test_list_users_with_stats_counts_trace_and_profile(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "s.db"
            store = SQLiteStore(str(db))
            store.init()
            TraceStore(db).init()
            uid = store.create_user("alice", "hash")

            with store._connect() as conn:
                conn.execute(
                    "INSERT INTO memory_blocks("
                    "user_id, conversation_id, message_id, role, position, chunk_index, "
                    "content, content_hash, created_at, is_deleted) "
                    "VALUES(?,?,?,?,?,?,?,?,?,0)",
                    (uid, 1, 1, "user", 0, 0, "hello", "h1", 1),
                )
                conn.execute(
                    "INSERT INTO memory_blocks("
                    "user_id, conversation_id, message_id, role, position, chunk_index, "
                    "content, content_hash, created_at, is_deleted) "
                    "VALUES(?,?,?,?,?,?,?,?,?,1)",
                    (uid, 1, 2, "user", 1, 0, "gone", "h2", 2),
                )
                conn.execute(
                    "INSERT INTO support_profile_items("
                    "id, user_id, content, origin, status, created_at, updated_at) "
                    "VALUES(?,?,?,?,?,?,?)",
                    ("p1", uid, "希望先倾听", "confirmed", "active", 1, 1),
                )
                conn.execute(
                    "INSERT INTO support_profile_items("
                    "id, user_id, content, origin, status, created_at, updated_at) "
                    "VALUES(?,?,?,?,?,?,?)",
                    ("p2", uid, "旧偏好", "confirmed", "deleted", 1, 1),
                )
                conn.commit()

            rows = store.list_users_with_stats()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["memory_count"], 2)  # 1 chunk + 1 active profile


if __name__ == "__main__":
    unittest.main()
