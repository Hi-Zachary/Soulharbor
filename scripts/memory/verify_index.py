#!/usr/bin/env python3
"""Verify trace index integrity."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from product_app.app.memory.store.repository import TraceStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--user-id", type=int, default=None)
    args = ap.parse_args()
    repo = TraceStore(args.db)
    repo.init()
    stats = repo.index_stats(args.user_id)
    missing = stats["chunks"] - stats["embeddings"]
    report = {
        **stats,
        "missing_embeddings": max(0, missing),
        "ok": missing <= 0 and stats["embed_retries"] == 0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
