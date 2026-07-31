# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/alignment/build_dpo_pairs_from_sft_corrupt.py
# 原先用途: 通过对 SFT 回复做腐蚀/负样本构造 DPO 对。
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


def _norm(t: str) -> str:
    return "".join((t or "").split()).lower()


# Basic emoji range; good enough for cleaning stylistic noise.
_RE_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]", re.UNICODE)
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def _clean_assistant_text(text: str) -> str:
    t = (text or "").strip()
    # Remove common forum/platform openings.
    t = re.sub(r"^\s*(?:题主|楼主|谢邀|谢邀。|谢邀，)\s*[:：,，]?\s*", "", t)
    # Drop emojis.
    t = _RE_EMOJI.sub("", t)
    # Collapse whitespace.
    t = _RE_MULTI_SPACE.sub(" ", t)
    t = t.strip()
    return t


def _take_turns_upto(messages: List[Dict[str, str]], end_user_idx: int, max_turns: int) -> List[Dict[str, str]]:
    """
    Take a tail slice that ends at messages[end_user_idx] (which must be user),
    keeping at most `max_turns` user turns (plus the interleaved assistant turns).
    """
    if end_user_idx < 0 or end_user_idx >= len(messages):
        return []
    if messages[end_user_idx].get("role") != "user":
        return []
    start = 0
    if max_turns > 0:
        turns = 0
        for i in range(end_user_idx, -1, -1):
            if messages[i].get("role") == "user":
                turns += 1
                if turns >= max_turns:
                    start = i
                    break
    window = messages[start : end_user_idx + 1]
    # Ensure ends with user.
    while window and window[-1].get("role") != "user":
        window.pop()
    return window


def _hash_row(conversations: List[Dict[str, str]], chosen: str, rejected: str) -> str:
    blob = json.dumps(
        {"conversations": conversations, "chosen": chosen, "rejected": rejected},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def _to_conversations(messages: List[Dict[str, str]], *, user_only_prompt: bool) -> List[Dict[str, str]]:
    conv: List[Dict[str, str]] = []
    for m in messages:
        role = str(m.get("role") or "")
        content = str(m.get("content") or "")
        if not content.strip():
            continue
        if role == "user":
            conv.append({"from": "human", "value": content})
        elif role == "assistant":
            if user_only_prompt:
                continue
            conv.append({"from": "gpt", "value": content})
    # Must end with human for DPO prompt
    while conv and conv[-1]["from"] != "human":
        conv.pop()
    return conv


def _first_sentence(text: str, *, min_chars: int = 30, max_chars: int = 120) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # Split by common sentence delimiters.
    parts = re.split(r"(?<=[。！？!?\n])", t)
    out = ""
    for p in parts:
        if not p.strip():
            continue
        if len(out) + len(p) > max_chars and len(out) >= min_chars:
            break
        out += p
        if len(out) >= min_chars:
            break
    out = out.strip()
    if not out:
        out = t[: max_chars].strip()
    return out


GENERIC_REJECTS = [
    "我理解你的感受。你先别想太多，慢慢来。",
    "听起来你最近挺不容易的。先休息一下，之后再说。",
    "我明白你的困扰。你可以先放松一下，看看会不会好些。",
    "这种情况很常见，不用太担心。",
]


def _pick_mismatch(pool: List[str], avoid: str, *, rng: random.Random, max_tries: int = 20) -> Optional[str]:
    if len(pool) < 2:
        return None
    for _ in range(max_tries):
        cand = pool[rng.randrange(0, len(pool))]
        if cand == avoid:
            continue
        # Avoid near-identical
        if _norm(cand) == _norm(avoid):
            continue
        return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build DPO pairs from an SFT ShareGPT dataset: chosen=dataset assistant reply; "
            "rejected=synthetic (mismatch/truncate/generic), no external judge required."
        )
    )
    ap.add_argument("--input", required=True, help="ShareGPT JSONL (each line contains a messages list).")
    ap.add_argument("--output", required=True, help="Output DPO JSONL (LLaMA-Factory ShareGPT DPO style).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=0, help="Cap number of DPO rows (0 = all).")
    ap.add_argument("--max-turns", type=int, default=3, help="How many recent user turns to keep as prompt context.")
    ap.add_argument(
        "--prompt-user-only",
        action="store_true",
        help="If set, drop assistant messages from the prompt context (chosen still uses the paired assistant reply).",
    )
    ap.add_argument("--min-chosen-chars", type=int, default=60, help="Skip chosen answers shorter than this.")
    ap.add_argument(
        "--reject-mode",
        choices=["mix", "mismatch", "truncate", "generic"],
        default="mix",
        help="How to generate rejected replies.",
    )
    ap.add_argument(
        "--mismatch-prob",
        type=float,
        default=0.7,
        help="When reject-mode=mix, probability of using mismatch negative.",
    )
    ap.add_argument(
        "--truncate-max-chars",
        type=int,
        default=120,
        help="When truncate is used, keep at most this many chars.",
    )
    args = ap.parse_args()

    rng = random.Random(int(args.seed))

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Missing input: {inp}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # First pass: collect all (prompt_conversations, chosen) candidates and a pool of chosen texts.
    candidates: List[Tuple[List[Dict[str, str]], str, Dict[str, Any]]] = []
    chosen_pool: List[str] = []
    for obj in _iter_jsonl(inp):
        msgs = obj.get("messages")
        if not isinstance(msgs, list) or not msgs:
            continue
        # Ensure role/content dicts
        messages: List[Dict[str, str]] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "").strip()
            content = str(m.get("content") or "")
            if role in ("user", "assistant") and content.strip():
                messages.append({"role": role, "content": content})
        if len(messages) < 2:
            continue

        for i in range(len(messages) - 1):
            if messages[i].get("role") != "user":
                continue
            if messages[i + 1].get("role") != "assistant":
                continue
            chosen = _clean_assistant_text(messages[i + 1].get("content") or "")
            if len(_norm(chosen)) < int(args.min_chosen_chars):
                continue
            prompt_msgs = _take_turns_upto(messages, i, int(args.max_turns))
            conv = _to_conversations(prompt_msgs, user_only_prompt=bool(args.prompt_user_only))
            if not conv or conv[-1]["from"] != "human":
                continue
            candidates.append((conv, chosen, {"source": str(inp)}))
            chosen_pool.append(chosen)

    if not candidates:
        raise SystemExit("No usable (prompt, chosen) pairs found. Check input schema and thresholds.")

    # Second pass: produce rejected and write DPO rows.
    rng.shuffle(candidates)
    limit = int(args.max_samples) if int(args.max_samples) > 0 else len(candidates)
    wrote = 0
    with out.open("w", encoding="utf-8") as f:
        for conv, chosen, meta in candidates:
            if wrote >= limit:
                break

            mode = str(args.reject_mode)
            use_mode = mode
            if mode == "mix":
                use_mode = "mismatch" if (rng.random() < float(args.mismatch_prob)) else "truncate"

            rejected: Optional[str] = None
            if use_mode == "mismatch":
                rejected = _pick_mismatch(chosen_pool, chosen, rng=rng)
                if rejected:
                    rejected = _clean_assistant_text(rejected)
            elif use_mode == "truncate":
                rejected = _first_sentence(chosen, max_chars=int(args.truncate_max_chars))
            elif use_mode == "generic":
                rejected = GENERIC_REJECTS[rng.randrange(0, len(GENERIC_REJECTS))]

            rejected = (rejected or "").strip()
            if not rejected or _norm(rejected) == _norm(chosen):
                rejected = _first_sentence(chosen, max_chars=int(args.truncate_max_chars))
                if not rejected or _norm(rejected) == _norm(chosen):
                    continue

            row_id = _hash_row(conv, chosen, rejected)
            f.write(
                json.dumps(
                    {
                        "id": row_id,
                        "conversations": conv,
                        "chosen": {"from": "gpt", "value": chosen},
                        "rejected": {"from": "gpt", "value": rejected},
                        "meta": {"reject_mode": use_mode, **meta},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            wrote += 1

    print(f"[OK] wrote {wrote} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

