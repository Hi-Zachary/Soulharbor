#!/usr/bin/env python3
"""Smoke-compare memory blocks for a fixed query."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from product_app.app.memory.engine import MemoryEngine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--user-id", type=int, required=True)
    ap.add_argument("--query", required=True)
    args = ap.parse_args()
    eng = MemoryEngine(args.db)
    block = eng.build_context(
        user_id=args.user_id,
        conversation_id=0,
        current_user_message=args.query,
        recent_messages=[],
        conversation_summary=None,
    )
    trace = eng.last_trace.to_log_dict() if eng.last_trace else {}
    print(json.dumps({"block": block, "trace": trace}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
