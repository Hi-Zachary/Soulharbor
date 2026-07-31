#!/usr/bin/env python3
"""Manual migration helper for audited support preferences (never auto-migrate L3)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from product_app.app.memory.engine import MemoryEngine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Import explicit/confirmed profiles from JSONL")
    ap.add_argument("--db", required=True)
    ap.add_argument("--input", required=True, help="JSONL: user_id,content,origin,source_message_ids")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    engine = MemoryEngine(args.db)
    ok = fail = 0
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        origin = obj.get("origin") or "explicit"
        if origin not in ("explicit", "confirmed"):
            fail += 1
            continue
        if args.dry_run:
            ok += 1
            continue
        if origin == "explicit":
            item = engine.remember_explicit(
                user_id=int(obj["user_id"]),
                content=str(obj["content"]),
                source_message_ids=[int(x) for x in obj.get("source_message_ids") or []],
            )
        else:
            item = engine.confirm_profile(
                user_id=int(obj["user_id"]),
                content=str(obj["content"]),
                source_message_ids=[int(x) for x in obj.get("source_message_ids") or []],
            )
        ok += 1 if item else 0
        fail += 0 if item else 1
    print(json.dumps({"ok": ok, "fail": fail, "dry_run": args.dry_run}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
