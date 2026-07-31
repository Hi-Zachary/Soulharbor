# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/alignment/build_prompts_pool.py
# 原先用途: 构建 DPO/对齐用的 prompt 池。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def _iter_jsonl_many(paths: List[Path], *, limit: int = 0) -> Iterable[Tuple[Path, int, Dict[str, Any]]]:
    """
    Iterate multiple JSONL files, yielding (path, line_no, obj).
    `limit` caps the total number of lines across all files (0 = no cap).
    """
    n = 0
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if limit and n >= limit:
                    return
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                n += 1
                yield (p, line_no, obj)


def _norm(t: str) -> str:
    return "".join((t or "").split()).lower()


_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_RE_PHONE = re.compile(r"(?<!\d)(?:1\d{10}|(?:\+?\d{1,3}[- ]?)?\d{3,4}[- ]?\d{7,8})(?!\d)")
_RE_IDLIKE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")


def _redact_basic(text: str) -> str:
    t = text or ""
    t = _RE_EMAIL.sub("[REDACTED_EMAIL]", t)
    t = _RE_PHONE.sub("[REDACTED_PHONE]", t)
    t = _RE_IDLIKE.sub("[REDACTED_ID]", t)
    return t


def _load_sharegpt_messages(obj: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Accept common ShareGPT-ish schemas:
    - {"messages":[{"role":"user","content":"..."}, ...]}
    - {"conversations":[{"from":"human","value":"..."}, {"from":"gpt","value":"..."}]}
    """
    msgs = obj.get("messages")
    if isinstance(msgs, list):
        out: List[Dict[str, str]] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "").strip()
            content = str(m.get("content") or "")
            if role in ("user", "assistant") and content.strip():
                out.append({"role": role, "content": content})
        return out

    conv = obj.get("conversations")
    if isinstance(conv, list):
        out2: List[Dict[str, str]] = []
        for m in conv:
            if not isinstance(m, dict):
                continue
            fr = str(m.get("from") or "").strip().lower()
            val = str(m.get("value") or "")
            if not val.strip():
                continue
            if fr in ("human", "user"):
                out2.append({"role": "user", "content": val})
            elif fr in ("gpt", "assistant"):
                out2.append({"role": "assistant", "content": val})
        return out2

    return []


def _take_last_turns(msgs: List[Dict[str, str]], max_turns: int) -> List[Dict[str, str]]:
    if max_turns <= 0:
        return list(msgs)
    # A "turn" is (user->assistant) pair. Keep a tail that ends with user.
    tail: List[Dict[str, str]] = []
    turns = 0
    for m in reversed(msgs):
        if (m.get("content") or "").strip():
            tail.append(m)
        if m.get("role") == "user":
            turns += 1
            if turns >= max_turns:
                break
    tail.reverse()
    # Ensure prompt ends with a user message.
    while tail and tail[-1].get("role") != "user":
        tail.pop()
    return tail


_ACK_TEXTS = {
    "嗯",
    "嗯嗯",
    "好",
    "好的",
    "行",
    "可以",
    "ok",
    "okay",
    "谢谢",
    "谢谢你",
    "多谢",
    "明白了",
    "了解了",
    "知道了",
    "懂了",
    "好吧",
    "行吧",
    "是的",
    "不是",
    "不",
    "别",
    "先这样",
}

_CLOSING_SUBSTRS = (
    "谢谢",
    "感谢",
    "多谢",
    "我会",
    "我明白",
    "我知道",
    "明白了",
    "了解了",
    "知道了",
    "懂了",
    "好的",
    "行吧",
    "好吧",
    "先这样",
    "再见",
    "拜拜",
)

_REQUEST_PAT = re.compile(r"[？?]|怎么办|如何|怎么|为什么|能不能|可以吗|行吗|该不该|要不要|是否|有没有|能否|求|帮我|建议|方法")


def _is_requestish(text: str) -> bool:
    return bool(_REQUEST_PAT.search(text or ""))


def _is_closing_text(text: str) -> bool:
    t = text or ""
    if _is_requestish(t):
        return False
    tn = _norm(t)
    return any(k in tn for k in _CLOSING_SUBSTRS)


def _is_low_info_user_text(text: str, *, min_chars: int) -> bool:
    tn = _norm(text)
    if not tn:
        return True
    if len(tn) < int(min_chars):
        return True
    if tn in _ACK_TEXTS:
        return True
    # Long but closing acknowledgements are also low value as prompt ends.
    if _is_closing_text(text):
        return True
    return False


def _pick_user_index(msgs: List[Dict[str, str]], *, min_chars: int, prefer_request: bool) -> Optional[int]:
    """
    Pick a user message index that is more likely to be a meaningful prompt end.
    Preference: last non-ack user message with enough characters.
    """
    last_user: Optional[int] = None
    if prefer_request:
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if m.get("role") != "user":
                continue
            if last_user is None:
                last_user = i
            text = str(m.get("content") or "")
            if _is_requestish(text) and not _is_low_info_user_text(text, min_chars=min_chars):
                return i
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.get("role") != "user":
            continue
        if last_user is None:
            last_user = i
        if not _is_low_info_user_text(str(m.get("content") or ""), min_chars=min_chars):
            return i
    return last_user


def _take_turns_upto(msgs: List[Dict[str, str]], end_index: int, max_turns: int) -> List[Dict[str, str]]:
    """
    Take a tail window ending at msgs[end_index] (inclusive), keeping up to max_turns user turns.
    Ensures the returned list ends with a user message.
    """
    if end_index < 0:
        return []
    if max_turns <= 0:
        window = list(msgs[: end_index + 1])
    else:
        window_rev: List[Dict[str, str]] = []
        turns = 0
        for i in range(end_index, -1, -1):
            m = msgs[i]
            if (m.get("content") or "").strip():
                window_rev.append(m)
            if m.get("role") == "user":
                turns += 1
                if turns >= max_turns:
                    break
        window = list(reversed(window_rev))

    while window and window[-1].get("role") != "user":
        window.pop()
    return window


def _hash_prompt(messages: List[Dict[str, str]]) -> str:
    blob = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


@dataclass(frozen=True)
class Bucket:
    name: str
    keywords: Tuple[str, ...]


BUCKETS: List[Bucket] = [
    Bucket("crisis", ("想死", "自杀", "不想活", "轻生", "结束生命", "自残", "割腕", "跳楼", "吞药", "上吊")),
    Bucket("sleep", ("失眠", "睡不着", "熬夜", "噩梦", "睡眠")),
    Bucket("anxiety", ("焦虑", "恐慌", "紧张", "压力", "崩溃", "害怕")),
    Bucket("mood", ("难过", "不开心", "抑郁", "绝望", "哭", "心累", "无助")),
    Bucket(
        "school",
        (
            "成绩",
            "考试",
            "期末",
            "作业",
            "课程",
            "选课",
            "绩点",
            "挂科",
            "复习",
            "论文",
            "答辩",
            "导师",
            "老师",
            "班主任",
            "辅导员",
            "学院",
            "实验室",
            "考研",
            "保研",
            "实习",
            "就业",
            "面试",
            "毕业",
            "奖学金",
            "处分",
        ),
    ),
    Bucket("social", ("人际", "同学", "室友", "宿舍", "寝室", "朋友", "恋爱", "分手", "孤独", "被排挤")),
    Bucket("family", ("父母", "家庭", "家里", "吵架", "离婚", "家暴", "控制", "冷暴力")),
]


def _bucket_of(text: str) -> str:
    t = _norm(text)
    for b in BUCKETS:
        if any(k in t for k in b.keywords):
            return b.name
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a prompts pool (messages ending with user) from ShareGPT jsonl.")
    ap.add_argument("--input", default="data/llm/sft_instruction.jsonl")
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Optional multiple input JSONL paths (overrides --input).",
    )
    ap.add_argument("--output", default="archive/llm_build/dpo_prompts_pool.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-turns", type=int, default=4, help="How many recent user turns to keep as context.")
    ap.add_argument(
        "--min-user-chars",
        type=int,
        default=10,
        help="Minimum normalized characters for the ending user message to be considered informative.",
    )
    ap.add_argument(
        "--prefer-request",
        action="store_true",
        default=True,
        help="Prefer ending user messages that look like a request/question.",
    )
    ap.add_argument(
        "--no-prefer-request",
        action="store_false",
        dest="prefer_request",
        help="Disable preferring request-like ending user messages.",
    )
    ap.add_argument(
        "--require-request",
        action="store_true",
        help="If set, drop samples whose ending user message is not request-like.",
    )
    ap.add_argument(
        "--user-only",
        action="store_true",
        help="If set, drop all assistant messages from the prompt context (useful when SFT assistant texts are noisy).",
    )
    ap.add_argument("--max-per-bucket", type=int, default=400, help="Cap prompts per bucket (used for non-other buckets).")
    ap.add_argument(
        "--max-other",
        type=int,
        default=0,
        help="Optional cap for bucket=other (0 means use --max-per-bucket).",
    )
    ap.add_argument("--max-total", type=int, default=0, help="Optional cap for final total prompts (0 = no cap).")
    ap.add_argument(
        "--target-total",
        type=int,
        default=0,
        help="If set, build a balanced pool of exactly N prompts via round-robin bucket fill.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Read at most N lines from input (0 = all).")
    ap.add_argument("--redact", action="store_true", help="Basic PII redaction (email/phone/id-like).")
    args = ap.parse_args()

    random.seed(args.seed)

    inps: List[Path]
    if args.inputs:
        inps = [Path(p) for p in args.inputs]
    else:
        inps = [Path(args.input)]

    missing = [p for p in inps if not p.exists()]
    if missing:
        raise SystemExit("Missing inputs:\n" + "\n".join(str(p) for p in missing))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    seen: set[str] = set()

    for inp, line_no, obj in _iter_jsonl_many(inps, limit=int(args.limit)):
        msgs = _load_sharegpt_messages(obj)
        if not msgs:
            continue

        end_idx = _pick_user_index(msgs, min_chars=int(args.min_user_chars), prefer_request=bool(args.prefer_request))
        if end_idx is None:
            continue

        end_user = str(msgs[end_idx].get("content") or "")
        if not end_user.strip():
            continue
        if args.require_request and (not _is_requestish(end_user)):
            continue

        prompt_msgs = _take_turns_upto(msgs, end_idx, args.max_turns)
        if not prompt_msgs:
            continue
        if args.user_only:
            prompt_msgs = [m for m in prompt_msgs if m.get("role") == "user"]
            if not prompt_msgs:
                continue
            # Ensure prompt ends with user.
            while prompt_msgs and prompt_msgs[-1].get("role") != "user":
                prompt_msgs.pop()

        if args.redact:
            prompt_msgs = [{"role": m["role"], "content": _redact_basic(m["content"])} for m in prompt_msgs]

        hid = _hash_prompt(prompt_msgs)
        if hid in seen:
            continue
        seen.add(hid)

        bname = _bucket_of(end_user)
        buckets.setdefault(bname, []).append(
            {
                "id": hid,
                "bucket": bname,
                "messages": prompt_msgs,
                "meta": {"source": str(inp), "line_hint": int(line_no), "picked_end_idx": int(end_idx)},
            }
        )

    # Shuffle items per bucket for stable sampling.
    for items in buckets.values():
        random.shuffle(items)

    # If target_total is set: fill via round-robin by bucket order for better topic balance.
    if int(args.target_total) > 0:
        target = int(args.target_total)
        order = ["school", "social", "family", "mood", "anxiety", "sleep", "crisis", "other"]
        # Apply caps before filling.
        capped: Dict[str, List[Dict[str, Any]]] = {}
        for bname, items in buckets.items():
            cap = int(args.max_per_bucket)
            if bname == "other" and int(args.max_other) > 0:
                cap = int(args.max_other)
            if cap > 0:
                capped[bname] = items[:cap]
            else:
                capped[bname] = items

        final: List[Dict[str, Any]] = []
        idx = {k: 0 for k in capped.keys()}
        while len(final) < target:
            progressed = False
            for bname in order:
                items = capped.get(bname) or []
                i = idx.get(bname, 0)
                if i < len(items):
                    final.append(items[i])
                    idx[bname] = i + 1
                    progressed = True
                    if len(final) >= target:
                        break
            if not progressed:
                break  # no more items anywhere
    else:
        # Downsample per bucket for balance.
        final = []
        for bname, items in buckets.items():
            cap = int(args.max_per_bucket)
            if bname == "other" and int(args.max_other) > 0:
                cap = int(args.max_other)
            if cap > 0:
                final.extend(items[:cap])
            else:
                final.extend(items)

        random.shuffle(final)
        if int(args.max_total) > 0 and len(final) > int(args.max_total):
            final = final[: int(args.max_total)]
    with out.open("w", encoding="utf-8") as f:
        for row in final:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("input:", str(inp))
    print("output:", str(out))
    print("prompts:", len(final))
    print("buckets:", {k: len(v) for k, v in buckets.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
