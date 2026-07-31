# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/compact_relabel_output.py
# 原先用途: 压缩/整理人工或模型重标注输出，便于入库。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input relabel jsonl (may contain duplicates).")
    ap.add_argument("--output", default="", help="Output jsonl (defaults to overwrite input).")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path
    if not in_path.exists():
        raise SystemExit(f"Not found: {in_path}")

    last: Dict[str, Dict[str, Any]] = {}
    order: list[str] = []

    with in_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Bad json at {in_path}:{line_no}: {e}") from e
            rid = str(obj.get("id"))
            if not rid:
                continue
            if rid not in last:
                order.append(rid)
            last[rid] = obj

    rows = [last[rid] for rid in order]
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
    tmp.replace(out_path)
    print(f"[OK] compacted {in_path} -> {out_path} unique_ids={len(rows)}")


if __name__ == "__main__":
    main()

