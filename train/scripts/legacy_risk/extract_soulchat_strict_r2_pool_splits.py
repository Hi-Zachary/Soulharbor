# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/extract_soulchat_strict_r2_pool_splits.py
# 原先用途: 将严格 R2 风险池划分 train/valid/test。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


RISK_EMOTION_INSTRUCTION = (
    "你是校园心理健康风险与情绪评估模型。输入为心理咨询/倾诉类多轮对话文本（is_consult=1）。"
    "请只输出严格 JSON："
    "{\"risk\":\"R0|R1|R2\",\"emotion\":\"高兴/积极|悲伤/消极|焦虑/紧张|愤怒/生气|迷茫/困惑|中性/未知\"}。"
    "不要输出多余文本。"
)

_RE_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")

STRICT_R2 = [
    "割腕",
    "跳楼",
    "吞药",
    "上吊",
    "遗书",
    "煤气",
    "自杀",
    "结束生命",
    "买好了药",
    "准备跳",
    "准备割",
    "准备吞",
    "已经割",
    "已经吞",
    "已经吃了药",
    "现在就在楼顶",
]


def _norm_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace(" ", "")
    s = s.replace("\u200b", "").replace("\ufeff", "")
    return s.strip()


def _stable_bucket(text: str, seed: int, modulo: int = 10_000) -> int:
    h = hashlib.md5((str(seed) + "::" + (text or "")).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % modulo


def _assign_split_from_id(item_id: Any, seed: int) -> str:
    b = _stable_bucket(str(item_id), seed=seed, modulo=10_000)
    if b < 500:
        return "test"
    if b < 1000:
        return "valid"
    return "train"


def _iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[Any]:
    try:
        import ijson  # type: ignore

        with path.open("rb") as f:
            for item in ijson.items(f, "item"):
                yield item
        return
    except Exception:
        pass

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buf = ""
        eof = False

        while True:
            if not buf and not eof:
                chunk = f.read(chunk_size)
                if chunk:
                    buf += chunk
                else:
                    eof = True
            buf = buf.lstrip()
            if buf.startswith("["):
                buf = buf[1:]
                break
            if eof:
                raise ValueError(f"Cannot find '[' in {path}")

        while True:
            while True:
                buf = buf.lstrip()
                if buf.startswith(","):
                    buf = buf[1:]
                    continue
                break
            if buf.lstrip().startswith("]"):
                return
            while True:
                try:
                    if buf.lstrip().startswith("]"):
                        return
                    obj, idx = decoder.raw_decode(buf)
                    yield obj
                    buf = buf[idx:]
                    break
                except json.JSONDecodeError:
                    if eof:
                        return
                    chunk = f.read(chunk_size)
                    if chunk:
                        buf += chunk
                    else:
                        eof = True


def _soulchat_messages_to_pairs(messages: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    pending_user: Optional[str] = None
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower()
        content = _norm_text(m.get("content", ""))
        if not content:
            continue
        if role == "user":
            pending_user = content
        elif role == "assistant":
            if pending_user is None:
                continue
            pairs.append((pending_user, content))
            pending_user = None
    return pairs


def _format_pairs_context(
    pairs: List[Tuple[str, str]],
    pair_idx: int,
    max_pairs: int = 6,
) -> Tuple[str, str]:
    if not pairs:
        return "", ""
    pair_idx = max(0, min(pair_idx, len(pairs) - 1))
    start = max(0, pair_idx - max_pairs + 1)

    lines: List[str] = []
    user_texts: List[str] = []
    for i in range(start, pair_idx):
        u, a = pairs[i]
        if u:
            lines.append(f"[USER] {u}")
            user_texts.append(u)
        if a:
            lines.append(f"[ASSISTANT] {a}")

    u_cur, _a_cur = pairs[pair_idx]
    if u_cur:
        lines.append(f"[USER] {u_cur}")
        user_texts.append(u_cur)

    return "\n".join(lines).strip(), "\n".join(user_texts).strip()


def _matches_strict_r2(user_focus: str) -> bool:
    s = _norm_text(user_focus)
    return any(k in s for k in STRICT_R2)


def _load_exclude_keys(dirs: List[Path]) -> Set[str]:
    keys: Set[str] = set()
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("risk_emotion_*.jsonl"):
            try:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        meta = obj.get("meta") or {}
                        if meta.get("id") is None or meta.get("pair_idx") is None:
                            continue
                        keys.add(f"{meta.get('id')}:{meta.get('pair_idx')}")
            except Exception:
                continue
    return keys


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--soulchat-path", default="archive/sources/soulchat/SoulChatCorpus-sft-multi-Turn.json")
    ap.add_argument(
        "--exclude-dirs",
        nargs="*",
        default=[
            "archive/classifiers/train/risk_emotion",
            "archive/classifiers/build/0_candidates_relabel",
            "archive/classifiers/build/0_candidates_r2_strict",
        ],
    )
    ap.add_argument("--out-dir", default="archive/classifiers/build/0_candidates_r2_strict_more_splits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=6)
    ap.add_argument("--turns-per-dialog", type=int, default=2)
    ap.add_argument("--valid-k", type=int, default=1200)
    ap.add_argument("--test-k", type=int, default=1200)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    soulchat_path = Path(args.soulchat_path)
    out_dir = Path(args.out_dir)
    exclude = _load_exclude_keys([Path(x) for x in args.exclude_dirs])

    want = {"valid": int(args.valid_k), "test": int(args.test_k)}
    rows = {"valid": [], "test": []}

    scanned = 0
    matched = 0
    for idx, item in enumerate(_iter_json_array(soulchat_path)):
        scanned = idx + 1
        if not isinstance(item, dict):
            continue
        item_id = item.get("id", idx)
        split = _assign_split_from_id(item_id, seed=args.seed)
        if split not in want:
            continue
        if len(rows[split]) >= want[split]:
            if all(len(rows[s]) >= want[s] for s in want):
                break
            continue

        messages = item.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        pairs = _soulchat_messages_to_pairs(messages)
        if len(pairs) < 2:
            continue

        chosen: List[int] = [len(pairs) - 1, max(0, len(pairs) // 2)]
        chosen = list(dict.fromkeys(chosen))
        while len(chosen) < min(int(args.turns_per_dialog), len(pairs)):
            r = rng.randrange(len(pairs))
            if r not in chosen:
                chosen.append(r)

        for pair_idx in chosen[: int(args.turns_per_dialog)]:
            key = f"{item_id}:{pair_idx}"
            if key in exclude:
                continue
            context_text, user_focus = _format_pairs_context(pairs, pair_idx, max_pairs=args.max_pairs)
            if not context_text or not user_focus or not _RE_HAS_CJK.search(user_focus):
                continue
            if not _matches_strict_r2(user_focus):
                continue

            matched += 1
            rows[split].append(
                {
                    "instruction": RISK_EMOTION_INSTRUCTION,
                    "input": context_text,
                    "output": json.dumps({"risk": "R2", "emotion": "中性/未知"}, ensure_ascii=False),
                    "meta": {
                        "source": "SoulChatCorpus",
                        "split": split,
                        "id": item_id,
                        "topic": item.get("topic"),
                        "pair_idx": pair_idx,
                        "derived_from": "r2_strict_pool_more_splits",
                    },
                }
            )
            exclude.add(key)
            if len(rows[split]) >= want[split]:
                break
        if all(len(rows[s]) >= want[s] for s in want):
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "risk_emotion_valid.jsonl", rows["valid"])
    _write_jsonl(out_dir / "risk_emotion_test.jsonl", rows["test"])
    # keep train file present for compatibility (empty)
    _write_jsonl(out_dir / "risk_emotion_train.jsonl", [])

    metrics = {
        "scanned": scanned,
        "matched_turns": matched,
        "written": {k: len(v) for k, v in rows.items()},
        "out_dir": str(out_dir),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

