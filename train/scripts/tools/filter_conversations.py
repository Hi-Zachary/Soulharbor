#!/usr/bin/env python3
# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/filter_conversations.py
# 原先用途: 对话语料过滤（长度、质量、敏感等规则）。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

"""
Filter high-quality conversations from sft_instruction.jsonl for extraction data building.
Output: batches of conversations for subagent processing.
"""
import json
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SFT_DATA = PROJECT_ROOT / "data/llm/sft_instruction.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data/llm/conversation_batches"
BATCH_SIZE = 20
TARGET = 880  # 600 train + 100 valid + 100 test + margin
SEED = 42

PSYCH_SIGNALS = [
    "压力", "焦虑", "抑郁", "难过", "崩溃", "失眠", "孤独",
    "父母", "妈妈", "爸爸", "家庭", "室友", "导师", "同学",
    "考研", "保研", "高考", "考试", "挂科", "毕业", "工作",
    "分手", "吵架", "家暴", "创伤", "童年", "小时候",
    "害怕", "担心", "不敢", "自卑", "自伤", "自杀", "痛苦",
    "不知道怎么办", "觉得", "感觉", "经常", "总是", "每次",
    "婚姻", "离婚", "出轨", "婆婆", "孩子", "教育",
]

def score_conversation(d):
    conv = d.get("conversations", [])
    msgs = conv if isinstance(conv, list) else []
    if len(msgs) < 6:
        return -1
    user_turns = [m for m in msgs if m.get("from") == "human"]
    if len(user_turns) < 3:
        return -1
    user_text = " ".join(str(m.get("value", "")) for m in user_turns)
    if len(user_text) < 150:
        return -1
    signal_count = sum(1 for s in PSYCH_SIGNALS if s in user_text)
    if signal_count < 2:
        return -1
    return signal_count * 100 + len(user_text)


def format_dialog(d):
    msgs = d.get("conversations", []) if isinstance(d, dict) else d
    lines = []
    for m in msgs[:10]:
        role = "用户" if m.get("from") == "human" else "助手"
        text = str(m.get("value", "")).strip()
        if not text:
            continue
        if len(text) > 400:
            text = text[:400]
        lines.append(f"{role}：{text}")
    return "\n".join(lines)


def main():
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(SFT_DATA, encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]

    print(f"Loaded {len(all_data)} conversations")

    scored = []
    for idx, d in enumerate(all_data):
        s = score_conversation(d)
        if s > 0:
            scored.append((s, idx, d))

    scored.sort(key=lambda x: -x[0])
    print(f"Filtered to {len(scored)} passing quality threshold")

    # Take top candidates, ensure diversity by sampling
    top_pool = scored[:4000]
    random.shuffle(top_pool)
    selected = top_pool[:TARGET]

    # Check for diversity: track which types of topics are covered
    print(f"Selected {len(selected)} for processing")

    # Write as batches
    batches = []
    for i in range(0, len(selected), BATCH_SIZE):
        batch = selected[i:i+BATCH_SIZE]
        batch_data = []
        for score, idx, d in batch:
            conv_list = d.get("conversations", []) if isinstance(d, dict) else d
            dialog = format_dialog(d)
            user_turns = [m for m in conv_list if m.get("from") == "human"]
            user_text = " ".join(str(m.get("value", "")) for m in user_turns)
            batch_data.append({
                "sft_idx": idx,
                "dialog": dialog,
                "user_text": user_text[:500],
                "system": d.get("system", ""),
                "score": score,
            })
        batches.append(batch_data)

    # Save each batch
    for bi, batch in enumerate(batches):
        path = OUTPUT_DIR / f"batch_{bi:03d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"batch_id": bi, "conversations": batch}, f, ensure_ascii=False, indent=2)

    print(f"Created {len(batches)} batches ({BATCH_SIZE} each) in {OUTPUT_DIR}")

    # Summary stats
    user_lens = [b["score"] for batch in batches for b in batch]
    print(f"Score range: {min(user_lens)} - {max(user_lens)}")
    print(f"Total conversations: {sum(len(b) for b in batches)}")


if __name__ == "__main__":
    main()
