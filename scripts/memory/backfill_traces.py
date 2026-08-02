#!/usr/bin/env python3
"""Backfill trace index from messages table. No LLM. Idempotent + checkpointed."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from product_app.app.memory.engine import MemoryEngine  # noqa: E402


def iter_messages(db_path: Path, *, user_id: int | None, after_id: int, limit: int):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        sql = (
            "SELECT m.id, m.role, m.content, m.created_at, m.conversation_id, c.user_id "
            "FROM messages m JOIN conversations c ON c.id=m.conversation_id "
            "WHERE m.id > ? AND m.role IN ('user','assistant') "
        )
        args: list = [after_id]
        if user_id is not None:
            sql += "AND c.user_id=? "
            args.append(int(user_id))
        sql += "ORDER BY m.id ASC LIMIT ?"
        args.append(int(limit))
        rows = conn.execute(sql, args).fetchall()
        # positions per conversation
        pos_cache: dict[int, dict[int, int]] = {}
        out = []
        for r in rows:
            cid = int(r["conversation_id"])
            if cid not in pos_cache:
                ids = [
                    int(x["id"])
                    for x in conn.execute(
                        "SELECT id FROM messages WHERE conversation_id=? ORDER BY id ASC",
                        (cid,),
                    ).fetchall()
                ]
                pos_cache[cid] = {mid: i + 1 for i, mid in enumerate(ids)}
            out.append(
                {
                    "message_id": int(r["id"]),
                    "user_id": int(r["user_id"]),
                    "conversation_id": cid,
                    "role": str(r["role"]),
                    "content": str(r["content"]),
                    "created_at": int(r["created_at"]),
                    "position": pos_cache[cid].get(int(r["id"]), 1),
                }
            )
        return out
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite chat DB path")
    ap.add_argument("--user-id", type=int, default=None)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-messages", type=int, default=0)
    args = ap.parse_args()

    db_path = Path(args.db)
    engine = MemoryEngine(db_path)
    ck_key = f"backfill:{args.user_id or 'all'}"
    after = int(engine._store.get_checkpoint(ck_key) or "0")
    stats = {"scanned": 0, "indexed": 0, "chunks": 0, "dry_run": args.dry_run}
    t0 = time.time()
    while True:
        batch = iter_messages(db_path, user_id=args.user_id, after_id=after, limit=args.batch)
        if not batch:
            break
        for m in batch:
            stats["scanned"] += 1
            after = m["message_id"]
            if args.dry_run:
                continue
            n = engine.ingest_message(**m)
            if n:
                stats["indexed"] += 1
                stats["chunks"] += n
            if args.max_messages and stats["scanned"] >= args.max_messages:
                break
        if not args.dry_run:
            engine._store.set_checkpoint(ck_key, str(after))
        if args.max_messages and stats["scanned"] >= args.max_messages:
            break
        if len(batch) < args.batch:
            break
    stats["elapsed_sec"] = round(time.time() - t0, 2)
    stats["checkpoint"] = after
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
