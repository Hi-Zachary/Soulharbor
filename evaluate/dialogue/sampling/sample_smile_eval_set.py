# [ARCHIVED — 非运行时依赖]
# 原路径: evaluate/sample_smile_eval_set.py
# 原先用途: 从 SMILE 语料抽样构建评测集。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from smile_to_instruction import build_instruction_and_reference


@dataclass(frozen=True)
class SampleRow:
    instruction: str
    output: str
    source: str


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample 1k zeroshot eval set from smile-main data/*.json.")
    ap.add_argument(
        "--smile-data-dir",
        default="/public/home/shijian/zachary/ZacPro/smile-main/data",
        help="Directory containing smile-main JSON dialogue files.",
    )
    ap.add_argument("--out", default="evaluate/dialogue/data/smile_1k.jsonl")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260426)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--scan-limit", type=int, default=0, help="Optional cap on number of files scanned (0=all).")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    rng = random.Random(int(args.seed))

    data_dir = Path(args.smile_data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Missing: {data_dir}")

    files = sorted(data_dir.glob("*.json"))
    if int(args.scan_limit) > 0:
        files = files[: int(args.scan_limit)]
    if not files:
        raise SystemExit(f"No *.json under: {data_dir}")

    rng.shuffle(files)

    target_n = int(args.n)
    rows: List[SampleRow] = []
    skipped = 0
    for p in files:
        obj = _read_json(p)
        if not isinstance(obj, list):
            skipped += 1
            continue
        ins_ref = build_instruction_and_reference(turns=obj, max_turns=int(args.max_turns))
        if ins_ref is None:
            skipped += 1
            continue
        ins, ref = ins_ref
        rows.append(SampleRow(instruction=ins, output=ref, source=str(p)))
        if len(rows) >= target_n:
            break

    if len(rows) < target_n:
        raise SystemExit(f"Not enough valid samples: got={len(rows)} need={target_n} skipped={skipped}")

    out_path = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    stats = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "smile_data_dir": str(data_dir),
        "n": target_n,
        "seed": int(args.seed),
        "max_turns": int(args.max_turns),
        "scanned_files": len(files),
        "skipped": skipped,
        "out": str(out_path),
    }
    stats_path = out_path.parent / "smile_eval_sampling_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] wrote:", str(out_path))
    print("[OK] wrote:", str(stats_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

