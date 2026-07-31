# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/inspect_sharegpt_sft_quality.py
# 原先用途: 检查 ShareGPT 格式 SFT 数据质量。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROLE_LEAK_PATTERNS = [
    re.compile(r"(^|\n)\s*(user|用户|User)\s*[:：>]", re.IGNORECASE),
    re.compile(r"(^|\n)\s*(assistant|助手|Assistant)\s*[:：>]", re.IGNORECASE),
    re.compile(r"<\|im_start\|>\s*(user|assistant)", re.IGNORECASE),
]

# Things that are suspicious if they appear in assistant messages too often.
ASSISTANT_QUESTION_ENDINGS = ("我该怎么办", "怎么办呢", "怎么做呢", "我该怎么做", "你能帮帮我吗", "求助", "我好难受", "我想死")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def norm_text(t: str) -> str:
    return "".join((t or "").split())


def has_role_leak(t: str) -> bool:
    if not t:
        return False
    for pat in ROLE_LEAK_PATTERNS:
        if pat.search(t):
            return True
    return False


def is_assistant_userlike(t: str) -> bool:
    tt = norm_text(t)
    if not tt:
        return False
    # Many counseling users end with "我该怎么办(呢)".
    return any(k in tt for k in ASSISTANT_QUESTION_ENDINGS)


def load_messages(obj: Dict[str, Any]) -> List[Dict[str, str]]:
    msgs = obj.get("messages")
    if isinstance(msgs, list):
        out = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "").strip()
            content = str(m.get("content") or "")
            if role and content:
                out.append({"role": role, "content": content})
        return out
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ShareGPT jsonl (each line has {messages:[...]}).")
    ap.add_argument("--max-lines", type=int, default=0, help="0 means all.")
    ap.add_argument("--show", type=int, default=5, help="Show top suspicious samples.")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Missing: {path}")

    n = 0
    bad_role = 0
    bad_leak = 0
    bad_assistant_userlike = 0
    role_counts = Counter()
    assistant_prefix = Counter()
    examples: List[Tuple[str, str]] = []

    for obj in iter_jsonl(path):
        n += 1
        if args.max_lines and n > args.max_lines:
            break
        msgs = load_messages(obj)
        if not msgs:
            bad_role += 1
            continue
        for m in msgs:
            role = m["role"]
            role_counts[role] += 1
            content = m["content"]
            if has_role_leak(content):
                bad_leak += 1
                if len(examples) < args.show:
                    examples.append((role, content[:240].replace("\n", "\\n")))
            if role == "assistant":
                if is_assistant_userlike(content):
                    bad_assistant_userlike += 1
                    if len(examples) < args.show:
                        examples.append((role, content[:240].replace("\n", "\\n")))
                pref = norm_text(content)[:12]
                if pref:
                    assistant_prefix[pref] += 1

    print("== ShareGPT SFT quality report ==")
    print("file:", str(path))
    print("lines:", n)
    print("empty/invalid messages:", bad_role)
    print("messages with role-leak markers:", bad_leak)
    print("assistant messages that look user-like:", bad_assistant_userlike)
    print("role_counts:", dict(role_counts))

    if assistant_prefix:
        top = assistant_prefix.most_common(10)
        print("assistant_prefix_top10:", top)

    if examples:
        print("\n-- examples --")
        for i, (role, snippet) in enumerate(examples, 1):
            print(f"[{i}] role={role} snippet={snippet}")


if __name__ == "__main__":
    main()

