# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/inspect_classifier_dataset_stats.py
# 原先用途: 统计分类器数据集标签分布与基本信息。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

import argparse
import json
from collections import Counter
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Bad jsonl at {path}:{line_no}: {e}") from e


def get_label(obj: dict, key: str):
    if key in obj:
        return obj[key]
    label = obj.get("label")
    if isinstance(label, dict) and key in label:
        return label[key]
    # Many training jsonl samples store labels in the "output" field as a JSON string
    out = obj.get("output")
    if isinstance(out, dict) and key in out:
        return out[key]
    if isinstance(out, str):
        s = out.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                out_obj = json.loads(s)
                if isinstance(out_obj, dict) and key in out_obj:
                    return out_obj[key]
            except Exception:
                return None
    return None


def print_stats(path: Path):
    risk_counter: Counter = Counter()
    emotion_counter: Counter = Counter()
    joint_counter: Counter = Counter()
    total = 0

    for obj in iter_jsonl(path):
        risk = get_label(obj, "risk")
        emotion = get_label(obj, "emotion")
        risk_counter[risk] += 1
        emotion_counter[emotion] += 1
        joint_counter[(risk, emotion)] += 1
        total += 1

    def fmt(counter: Counter):
        return ", ".join([f"{k}:{v}" for k, v in counter.most_common()])

    print(f"\n== {path.name} ==")
    print(f"N={total}")
    if risk_counter:
        print(f"risk: {fmt(risk_counter)}")
    if emotion_counter:
        print(f"emotion: {fmt(emotion_counter)}")
    if joint_counter:
        top = joint_counter.most_common(12)
        print("top_joint:")
        for (risk, emotion), n in top:
            print(f"  {risk}/{emotion}: {n}")

def print_intent_stats(path: Path):
    counter: Counter = Counter()
    total = 0
    for obj in iter_jsonl(path):
        label = get_label(obj, "is_consult")
        if label is None:
            out = obj.get("output")
            if isinstance(out, str):
                s = out.strip()
                if s.startswith("{") and s.endswith("}"):
                    try:
                        out_obj = json.loads(s)
                        if isinstance(out_obj, dict):
                            label = out_obj.get("is_consult")
                    except Exception:
                        label = None
                else:
                    label = s
            elif isinstance(out, dict):
                label = out.get("is_consult")
        counter[str(label)] += 1
        total += 1

    def fmt(counter: Counter):
        return ", ".join([f"{k}:{v}" for k, v in counter.most_common()])

    print(f"\n== {path.name} ==")
    print(f"N={total}")
    print(f"is_consult: {fmt(counter)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--risk-emotion-dir",
        type=str,
        default="data/classifiers/train/risk_emotion",
        help="Directory containing risk_emotion_{train,valid,test}.jsonl",
    )
    parser.add_argument(
        "--intent-dir",
        type=str,
        default="data/classifiers/train/intent",
        help="Directory containing intent_{train,valid,test}.jsonl",
    )
    args = parser.parse_args()

    re_base = Path(args.risk_emotion_dir)
    if re_base.exists():
        for name in ["risk_emotion_train.jsonl", "risk_emotion_valid.jsonl", "risk_emotion_test.jsonl"]:
            path = re_base / name
            if not path.exists():
                print(f"[WARN] Missing {path}")
                continue
            print_stats(path)
    else:
        print(f"[WARN] Not found: {re_base}")

    intent_base = Path(args.intent_dir)
    if intent_base.exists():
        for name in ["intent_train.jsonl", "intent_valid.jsonl", "intent_test.jsonl"]:
            path = intent_base / name
            if not path.exists():
                print(f"[WARN] Missing {path}")
                continue
            print_intent_stats(path)
    else:
        print(f"[WARN] Not found: {intent_base}")


if __name__ == "__main__":
    main()
