# [ARCHIVED — 非运行时依赖]
# 原路径: evaluate/sample_soulchat_eval_sets.py
# 原先用途: 从 SoulChat 语料抽样构建 seen/unseen 评测集。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

from json_stream import iter_json_array
from soulchat_signatures import (
    load_used_sigs_from_dpo_jsonl,
    load_used_sigs_from_sharegpt_jsonl,
    signature_from_messages,
)
from soulchat_to_instruction import build_instruction_and_reference


@dataclass(frozen=True)
class SampleRow:
    instruction: str
    output: str
    topic: str
    example_id: str
    signature: str
    split: str


def _reservoir_add(rng: random.Random, reservoir: List[SampleRow], item: SampleRow, seen_so_far: int, k: int) -> None:
    """
    Standard reservoir sampling:
      - keep first k items
      - for item i (1-based), replace random existing with prob k/i
    Here seen_so_far is 1-based index within this stream/bucket.
    """
    if k <= 0:
        return
    if len(reservoir) < k:
        reservoir.append(item)
        return
    j = rng.randrange(seen_so_far)
    if j < k:
        reservoir[j] = item


def _finalize_balanced(
    rng: random.Random,
    by_topic: Dict[str, List[SampleRow]],
    n: int,
) -> List[SampleRow]:
    topics = [t for t, rows in by_topic.items() if rows]
    if not topics:
        return []

    for t in topics:
        rng.shuffle(by_topic[t])
    rng.shuffle(topics)

    per_topic = max(1, n // len(topics))
    chosen: List[SampleRow] = []

    # First pass: take up to per_topic from each topic.
    for t in topics:
        take = min(per_topic, len(by_topic[t]), n - len(chosen))
        if take <= 0:
            continue
        chosen.extend(by_topic[t][:take])
        by_topic[t] = by_topic[t][take:]
        if len(chosen) >= n:
            return chosen[:n]

    # Fill leftovers round-robin across remaining topic pools.
    while len(chosen) < n:
        progressed = False
        for t in topics:
            if not by_topic[t]:
                continue
            chosen.append(by_topic[t].pop())
            progressed = True
            if len(chosen) >= n:
                break
        if not progressed:
            break

    return chosen[:n]


def _write_jsonl(path: Path, rows: List[SampleRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample balanced SoulChatCorpus eval sets (seen/unseen) for BLEU/ROUGE.")
    ap.add_argument(
        "--soulchat-json",
        default="data/SoulChatCorpus/SoulChatCorpus-sft-multi-Turn.json",
        help="SoulChatCorpus multi-turn JSON array path.",
    )
    ap.add_argument(
        "--sft-soulchat-jsonl",
        default="llama_factory/data/sft_soulchat_sharegpt.jsonl",
        help="ShareGPT JSONL used in SFT (SoulChat subset).",
    )
    ap.add_argument(
        "--dpo-soulchat-jsonl",
        default="llama_factory/data/dpo_soulchat_exclusive_minimax_3k_sharegpt.jsonl",
        help="ShareGPT/DPO JSONL used in DPO (SoulChat exclusive subset).",
    )
    ap.add_argument("--seen-out", default="evaluate/dialogue/data/soulchat_seen_1k.jsonl")
    ap.add_argument("--unseen-out", default="evaluate/dialogue/data/soulchat_unseen_1k.jsonl")
    ap.add_argument(
        "--cache-dir",
        default="evaluate/dialogue/cache",
        help="Cache directory for computed signature sets (speeds up repeated sampling).",
    )
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260426)
    ap.add_argument("--max-turns", type=int, default=4, help="Context turns when converting to instruction.")
    ap.add_argument("--max-msgs-sig", type=int, default=6, help="Number of messages used for signature matching.")
    ap.add_argument(
        "--per-topic-reservoir",
        type=int,
        default=300,
        help="Max examples kept per topic per split before balancing.",
    )
    ap.add_argument(
        "--scan-limit",
        type=int,
        default=0,
        help="Only scan first N SoulChatCorpus items for quick debugging (0=all).",
    )
    ap.add_argument(
        "--fast-stop",
        action="store_true",
        help="Stop scanning once we have enough candidates (may be slightly less balanced).",
    )
    ap.add_argument(
        "--fast-stop-buffer",
        type=int,
        default=300,
        help="Extra candidates required beyond --n for fast stop.",
    )
    ap.add_argument(
        "--fast-stop-min-topics",
        type=int,
        default=8,
        help="Minimum distinct topics seen in each split before fast stop.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    rng = random.Random(int(args.seed))

    soulchat_json = (root / args.soulchat_json).resolve() if not Path(args.soulchat_json).is_absolute() else Path(args.soulchat_json)
    sft_jsonl = (root / args.sft_soulchat_jsonl).resolve() if not Path(args.sft_soulchat_jsonl).is_absolute() else Path(args.sft_soulchat_jsonl)
    dpo_jsonl = (root / args.dpo_soulchat_jsonl).resolve() if not Path(args.dpo_soulchat_jsonl).is_absolute() else Path(args.dpo_soulchat_jsonl)

    if not soulchat_json.exists():
        raise SystemExit(f"Missing: {soulchat_json}")
    if not sft_jsonl.exists():
        raise SystemExit(f"Missing: {sft_jsonl}")
    if not dpo_jsonl.exists():
        raise SystemExit(f"Missing: {dpo_jsonl}")

    cache_dir = (root / args.cache_dir).resolve() if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_build(cache_path: Path, builder) -> set[str]:
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        sigs = builder()
        try:
            with cache_path.open("wb") as f:
                pickle.dump(sigs, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
        return sigs

    sft_cache = cache_dir / f"soulchat_sft_sigs_maxmsgs{int(args.max_msgs_sig)}.pkl"
    dpo_cache = cache_dir / f"soulchat_dpo_sigs_maxmsgs{int(args.max_msgs_sig)}.pkl"

    print(f"[LOAD] sft used signatures (cache={sft_cache.name}) ...", flush=True)
    sft_sigs = _load_or_build(
        sft_cache,
        lambda: load_used_sigs_from_sharegpt_jsonl(str(sft_jsonl), max_msgs=int(args.max_msgs_sig)),
    )
    print(f"[LOAD] dpo used signatures (cache={dpo_cache.name}) ...", flush=True)
    dpo_sigs = _load_or_build(
        dpo_cache,
        lambda: load_used_sigs_from_dpo_jsonl(str(dpo_jsonl), max_msgs=int(args.max_msgs_sig)),
    )

    seen_by_topic: DefaultDict[str, List[SampleRow]] = defaultdict(list)
    unseen_by_topic: DefaultDict[str, List[SampleRow]] = defaultdict(list)
    seen_seen_cnt: DefaultDict[str, int] = defaultdict(int)
    unseen_seen_cnt: DefaultDict[str, int] = defaultdict(int)

    target_n = int(args.n)
    scanned = 0
    for obj in iter_json_array(soulchat_json):
        scanned += 1
        if int(args.scan_limit) > 0 and scanned > int(args.scan_limit):
            break

        if not isinstance(obj, dict):
            continue
        topic = str(obj.get("topic") or "未明确")
        example_id = str(obj.get("id") or "")
        messages = obj.get("messages")
        if not isinstance(messages, list) or not messages:
            continue

        sig = signature_from_messages(messages, max_msgs=int(args.max_msgs_sig))
        in_sft = sig in sft_sigs
        in_dpo = sig in dpo_sigs

        if not in_sft and not in_dpo:
            split = "unseen"
        elif in_sft:
            split = "seen"
        else:
            # DPO-only examples are excluded from both sets by design.
            continue

        ins_ref = build_instruction_and_reference(topic=topic, messages=messages, max_turns=int(args.max_turns))
        if ins_ref is None:
            continue
        instruction, reference = ins_ref

        row = SampleRow(
            instruction=instruction,
            output=reference,
            topic=topic,
            example_id=example_id,
            signature=sig,
            split=split,
        )

        if split == "seen":
            seen_seen_cnt[topic] += 1
            _reservoir_add(rng, seen_by_topic[topic], row, seen_seen_cnt[topic], int(args.per_topic_reservoir))
        else:
            unseen_seen_cnt[topic] += 1
            _reservoir_add(rng, unseen_by_topic[topic], row, unseen_seen_cnt[topic], int(args.per_topic_reservoir))

        if scanned % 50000 == 0:
            print(f"[SCAN] {scanned} items ...", flush=True)

        if bool(args.fast_stop):
            seen_total = sum(len(v) for v in seen_by_topic.values())
            unseen_total = sum(len(v) for v in unseen_by_topic.values())
            seen_topics = sum(1 for v in seen_by_topic.values() if v)
            unseen_topics = sum(1 for v in unseen_by_topic.values() if v)
            if (
                seen_total >= target_n + int(args.fast_stop_buffer)
                and unseen_total >= target_n + int(args.fast_stop_buffer)
                and seen_topics >= int(args.fast_stop_min_topics)
                and unseen_topics >= int(args.fast_stop_min_topics)
            ):
                print(
                    f"[FAST-STOP] scanned={scanned} seen_total={seen_total} unseen_total={unseen_total} "
                    f"seen_topics={seen_topics} unseen_topics={unseen_topics}",
                    flush=True,
                )
                break

    seen_rows = _finalize_balanced(rng, dict(seen_by_topic), target_n)
    unseen_rows = _finalize_balanced(rng, dict(unseen_by_topic), target_n)

    if len(seen_rows) < target_n:
        raise SystemExit(f"Seen split too small: {len(seen_rows)} < {target_n}")
    if len(unseen_rows) < target_n:
        raise SystemExit(f"Unseen split too small: {len(unseen_rows)} < {target_n}")

    seen_out = (root / args.seen_out).resolve() if not Path(args.seen_out).is_absolute() else Path(args.seen_out)
    unseen_out = (root / args.unseen_out).resolve() if not Path(args.unseen_out).is_absolute() else Path(args.unseen_out)
    _write_jsonl(seen_out, seen_rows)
    _write_jsonl(unseen_out, unseen_rows)

    stats = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "soulchat_json": str(soulchat_json),
        "sft_soulchat_jsonl": str(sft_jsonl),
        "dpo_soulchat_jsonl": str(dpo_jsonl),
        "seed": int(args.seed),
        "n": target_n,
        "max_turns": int(args.max_turns),
        "max_msgs_sig": int(args.max_msgs_sig),
        "per_topic_reservoir": int(args.per_topic_reservoir),
        "scan_limit": int(args.scan_limit),
        "seen_topic_counts": dict(Counter([r.topic for r in seen_rows])),
        "unseen_topic_counts": dict(Counter([r.topic for r in unseen_rows])),
        "seen_out": str(seen_out),
        "unseen_out": str(unseen_out),
    }
    stats_path = seen_out.parent / "soulchat_eval_sampling_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] wrote:", str(seen_out), flush=True)
    print("[OK] wrote:", str(unseen_out), flush=True)
    print("[OK] wrote:", str(stats_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
