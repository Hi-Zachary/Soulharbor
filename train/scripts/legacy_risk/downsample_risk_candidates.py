# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/downsample_risk_candidates.py
# 原先用途: 对风险候选样本降采样，平衡分类器训练分布（历史 risk 管线）。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _get_heur_risk(row: Dict[str, Any]) -> str:
    out = row.get("output")
    if isinstance(out, str):
        try:
            out = json.loads(out)
        except Exception:
            return ""
    if isinstance(out, dict):
        return str(out.get("risk") or "")
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-r0", type=int, default=0)
    ap.add_argument("--train-r1", type=int, default=1200)
    ap.add_argument("--train-r2", type=int, default=800)
    ap.add_argument("--valid-r0", type=int, default=400)
    ap.add_argument("--valid-r1", type=int, default=250)
    ap.add_argument("--valid-r2", type=int, default=250)
    ap.add_argument("--test-r0", type=int, default=400)
    ap.add_argument("--test-r1", type=int, default=250)
    ap.add_argument("--test-r2", type=int, default=250)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    want = {
        "train": {"R0": args.train_r0, "R1": args.train_r1, "R2": args.train_r2},
        "valid": {"R0": args.valid_r0, "R1": args.valid_r1, "R2": args.valid_r2},
        "test": {"R0": args.test_r0, "R1": args.test_r1, "R2": args.test_r2},
    }

    metrics: Dict[str, Any] = {"in_dir": str(in_dir), "out_dir": str(out_dir), "picked": {}}
    for split in ("train", "valid", "test"):
        rows = _read_jsonl(in_dir / f"risk_emotion_{split}.jsonl")
        pools = {"R0": [], "R1": [], "R2": []}
        for r in rows:
            risk = _get_heur_risk(r)
            if risk in pools:
                pools[risk].append(r)
        picked: List[Dict[str, Any]] = []
        for risk in ("R0", "R1", "R2"):
            k = int(want[split][risk])
            if k <= 0:
                continue
            if len(pools[risk]) < k:
                raise SystemExit(f"Not enough {split} {risk}: got={len(pools[risk])} need={k}")
            rng.shuffle(pools[risk])
            picked.extend(pools[risk][:k])
        rng.shuffle(picked)
        _write_jsonl(out_dir / f"risk_emotion_{split}.jsonl", picked)
        metrics["picked"][split] = {
            "want": want[split],
            "in_pool_sizes": {k: len(v) for k, v in pools.items()},
            "out_total": len(picked),
        }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

