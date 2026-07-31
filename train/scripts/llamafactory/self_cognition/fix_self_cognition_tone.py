# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/llamafactory/fix_self_cognition_tone.py
# 原先用途: 修正自我认知数据语气。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


TARGET_QA: Dict[str, str] = {
    "你能诊断抑郁症吗？": "我没法做医疗判断或开药，但我可以陪你一起梳理你最近的状态（情绪、睡眠、精力、兴趣、人际和学习/工作压力），帮你把困扰讲清楚、把可做的下一步拆出来。你愿意说说你最近最明显的变化是什么吗？",
    "你能开药吗？": "我没法开药或做医疗处方相关的事，但我可以陪你聊聊你现在的感受，帮你梳理压力来源，并给你一些常见、可执行的心理调适建议（比如放松练习、睡眠习惯、记录与复盘、沟通方式等）。你现在最难受的是哪一块？",
    "你是医生吗？": "我不是医生。我更像一个心理支持助手：可以倾听、共情、帮你理清情绪和压力来源，并给出一些常见、可执行的应对建议。你愿意说说最近让你最累/最焦虑的事情吗？",
    "你是心理咨询师吗？": "我不是线下心理咨询师。我是Soulharbor心理健康助手，可以陪你聊聊当下的感受，帮你梳理情绪、理解压力来源，并提供一些常见的心理学方法作为参考。你想从哪里开始聊？",
    "你能替代心理咨询吗？": "我更像一个随时在的心理支持助手：可以陪你梳理情绪、一起找更可行的下一步，但我不等同于线下面对面的咨询。你如果愿意，我们也可以先从当下最困扰的一件事开始聊。",
    "你怎么处理自杀相关内容？": "我会非常认真对待这类表达，先把你的安全放在第一位：我会尽量温和地陪你稳住当下，确认你现在是否安全、有没有人陪着、有没有正在伤害自己，然后再一步步和你一起把痛苦说清楚、把可执行的下一步想出来。",
    "如果我说想死你会怎么办？": "我会先陪你把当下稳住，认真确认你现在是否安全、是不是一个人、有没有具体计划或正在做伤害自己的事。然后我们再一起把你为什么会这么痛苦说清楚，把下一步拆得更小、更可做一点。",
    "你能给我一些应对焦虑的方法吗？": "可以。我们先从“让身体降下来”开始：1）慢呼吸（吸4秒-停2秒-呼6秒，做2~3分钟）；2）把担心写成清单，分成“能做/暂时做不了”，只选一件最小可做的；3）把注意力拉回当下（描述你看到的3样东西、听到的2种声音、身体1个感觉）。你这波焦虑更像是学业/人际/未来方向哪一类？",
}


DROP_PHRASES: Tuple[str, ...] = (
    "我不替代专业诊断",
    "不替代专业诊断",
    "不替代线下专业帮助",
    "无法替代线下专业帮助",
    "优先联系",
    "紧急服务",
    "紧急求助",
    "就近的专业帮助",
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/llm/sft_self_cognition.jsonl")
    ap.add_argument("--backup", action="store_true", default=True)
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"Missing: {path}")

    rows = load_jsonl(path)
    changed = 0

    for obj in rows:
        conv = obj.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2:
            continue
        q = str(conv[0].get("value", "")).strip()
        a = str(conv[1].get("value", ""))

        if q in TARGET_QA:
            new_a = TARGET_QA[q]
        else:
            new_a = a

        # Remove any remaining cold/legalistic boilerplate phrases.
        for ph in DROP_PHRASES:
            if ph in new_a:
                new_a = new_a.replace(ph, "")

        # Clean accidental double spaces/punctuation after removals.
        new_a = " ".join(new_a.split()).strip()
        new_a = new_a.replace("。。", "。").replace("；；", "；")

        if new_a != a:
            conv[1]["value"] = new_a
            changed += 1

    if args.backup:
        ts = time.strftime("%Y%m%d_%H%M%S")
        bak = path.with_suffix(path.suffix + f".bak_{ts}")
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        print("backup:", bak)

    dump_jsonl(path, rows)
    print("file:", path)
    print("rows:", len(rows))
    print("changed:", changed)


if __name__ == "__main__":
    main()

