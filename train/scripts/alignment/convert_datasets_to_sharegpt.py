# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/alignment/convert_datasets_to_sharegpt.py
# 原先用途: 将内部数据格式转为 ShareGPT 供 LLaMA-Factory 使用。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _detect_format(obj: Any) -> str:
    # single_turn_dataset_{1,2}.json: list[{prompt, completion}]
    if isinstance(obj, list) and obj:
        ex = obj[0]
        if isinstance(ex, dict) and "prompt" in ex and "completion" in ex:
            return "prompt_completion_list"
        if isinstance(ex, dict) and "question" in ex and "answers" in ex:
            return "psyqa_list"
        if isinstance(ex, dict) and "messages" in ex:
            return "sharegpt_list"
    return "unknown"


def _to_sharegpt_rows(path: Path, data: Any, fmt: str, *, max_samples: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def _cap() -> bool:
        return max_samples > 0 and len(rows) >= max_samples

    if fmt == "prompt_completion_list":
        assert isinstance(data, list)
        for i, ex in enumerate(data):
            if _cap():
                break
            if not isinstance(ex, dict):
                continue
            prompt = str(ex.get("prompt") or "").strip()
            completion = str(ex.get("completion") or "").strip()
            if not prompt or not completion:
                continue
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ],
                    "meta": {"source": str(path), "idx": i, "format": fmt},
                }
            )
        return rows

    if fmt == "psyqa_list":
        assert isinstance(data, list)
        for i, ex in enumerate(data):
            if _cap():
                break
            if not isinstance(ex, dict):
                continue
            question = str(ex.get("question") or "").strip()
            desc = str(ex.get("description") or "").strip()
            prompt = question
            if desc and desc != question:
                prompt = f"{question}\n{desc}"

            answers = ex.get("answers")
            answer_text: Optional[str] = None
            if isinstance(answers, list) and answers:
                a0 = answers[0]
                if isinstance(a0, dict):
                    answer_text = str(a0.get("answer_text") or "").strip()
                elif isinstance(a0, str):
                    answer_text = a0.strip()

            if not prompt or not answer_text:
                continue
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer_text},
                    ],
                    "meta": {
                        "source": str(path),
                        "idx": i,
                        "format": fmt,
                        "questionID": ex.get("questionID"),
                        "keywords": ex.get("keywords"),
                    },
                }
            )
        return rows

    if fmt == "sharegpt_list":
        # Already in ShareGPT-ish; just wrap meta.
        assert isinstance(data, list)
        for i, ex in enumerate(data):
            if _cap():
                break
            if not isinstance(ex, dict):
                continue
            if not isinstance(ex.get("messages"), list):
                continue
            rows.append({"messages": ex["messages"], "meta": {"source": str(path), "idx": i, "format": fmt}})
        return rows

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert local datasets to ShareGPT JSONL (messages schema).")
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more dataset paths (JSON or JSONL).")
    ap.add_argument("--output", required=True, help="Output JSONL path (ShareGPT-ish: {messages:[...], meta:{...}}).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-input", type=int, default=0, help="Optional cap per input file (0 = no cap).")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle output rows.")
    args = ap.parse_args()

    random.seed(args.seed)

    out_rows: List[Dict[str, Any]] = []
    for p in args.inputs:
        path = Path(p)
        if not path.exists():
            continue

        # Try JSON first, then JSONL.
        try:
            data = _read_json(path)
            fmt = _detect_format(data)
            rows = _to_sharegpt_rows(path, data, fmt, max_samples=args.max_per_input)
        except Exception:
            data = list(_iter_jsonl(path))
            fmt = _detect_format(data)
            rows = _to_sharegpt_rows(path, data, fmt, max_samples=args.max_per_input)

        out_rows.extend(rows)

    if args.shuffle:
        random.shuffle(out_rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

