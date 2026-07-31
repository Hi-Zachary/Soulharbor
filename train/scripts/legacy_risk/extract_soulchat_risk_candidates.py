# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/extract_soulchat_risk_candidates.py
# 原先用途: 从 SoulChat 对话中抽取风险相关候选样本（历史 risk 数据管线）。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


RISK_EMOTION_INSTRUCTION = (
    "你是校园心理健康风险与情绪评估模型。输入为心理咨询/倾诉类多轮对话文本（is_consult=1）。"
    "请只输出严格 JSON："
    "{\"risk\":\"R0|R1|R2\",\"emotion\":\"高兴/积极|悲伤/消极|焦虑/紧张|愤怒/生气|迷茫/困惑|中性/未知\"}。"
    "不要输出多余文本。"
)


_RE_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")
_RE_CAMPUS = re.compile(
    r"(学校|校园|大学|学院|老师|同学|室友|宿舍|寝室|宿管|舍友|班级|课表|课程|选课|补考|重修|期中|期末|考试|挂科|绩点|GPA|作业|论文|开题|答辩|实验课|实验室|课题组|实习|面试|简历|秋招|春招|校招|考研|保研|推免|导师|助教|辅导员|班主任|奖学金|竞赛|社团|学生会|图书馆|自习室|教室|食堂|校医院|心理中心|校心理|宿舍楼|军训|学分|选专业|转专业|双学位|四六级|英语|学习|课堂|出勤|点名)"
)


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


def _is_campus_related(s: str) -> bool:
    s = _norm_text(s)
    return bool(_RE_CAMPUS.search(s))


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


def _risk_from_text(s: str) -> str:
    """
    仅用于“候选抽取的粗分桶”，最终标签以 MiniMax 重标注为准。
    新三档：R0/R1/R2（R2 为最高风险）。
    """
    s = _norm_text(s)

    r2 = [
        "自杀",
        "想死",
        "不想活了",
        "结束生命",
        "割腕",
        "跳楼",
        "吞药",
        "上吊",
        "煤气",
        "去死",
        "已经割",
        "已经吞",
        "已经吃了药",
        "现在就在楼顶",
        "准备跳",
        "准备割",
        "准备吞",
        "买好了药",
    ]
    if any(k in s for k in r2):
        return "R2"

    r1 = [
        "活着没意思",
        "没意义",
        "撑不住了",
        "扛不住了",
        "我崩溃了",
        "我快崩溃了",
        "想消失",
        "不如死了算了",
        "不想活",
        "想伤害自己",
        "自残",
        "反复想",
        "每天都想",
        "完全睡不着",
        "整夜睡不着",
    ]
    if any(k in s for k in r1):
        return "R1"

    return "R0"


def _emotion_from_text(s: str) -> str:
    s = _norm_text(s)
    positive = ["开心", "高兴", "快乐", "满足", "充实", "成就感", "兴奋", "欣喜", "感恩"]
    anger = ["生气", "愤怒", "火大", "恼火", "讨厌", "气死", "烦死", "恨", "憎恨", "暴躁"]
    anxiety = ["焦虑", "紧张", "压力", "担心", "害怕", "心慌", "恐慌", "慌", "失眠", "睡不着", "心跳", "惊恐"]
    sad = ["难过", "伤心", "低落", "抑郁", "绝望", "无助", "空虚", "没意义", "不想活", "活着没意思", "孤独", "丧"]
    confused = ["迷茫", "困惑", "纠结", "犹豫", "不知所措", "不知道怎么办", "不知道该怎么办", "不知道怎么做", "没有方向", "找不到方向", "想不明白"]

    if any(k in s for k in positive):
        return "高兴/积极"
    if any(k in s for k in anger):
        return "愤怒/生气"
    if any(k in s for k in anxiety):
        return "焦虑/紧张"
    if any(k in s for k in sad):
        return "悲伤/消极"
    if any(k in s for k in confused):
        return "迷茫/困惑"
    return "中性/未知"


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


def _load_existing_keys(existing_dir: Path) -> Set[str]:
    keys: Set[str] = set()
    if not existing_dir.exists():
        return keys
    for split in ("train", "valid", "test"):
        p = existing_dir / f"risk_emotion_{split}.jsonl"
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                meta = obj.get("meta") or {}
                if meta.get("id") is None or meta.get("pair_idx") is None:
                    continue
                keys.add(f"{meta.get('id')}:{meta.get('pair_idx')}")
    return keys


@dataclass
class Cap:
    r0: int
    r1: int
    r2: int

    def remaining(self) -> int:
        return max(0, self.r0) + max(0, self.r1) + max(0, self.r2)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--soulchat-path",
        default="archive/sources/soulchat/SoulChatCorpus-sft-multi-Turn.json",
        help="SoulChatCorpus JSON array file path",
    )
    ap.add_argument(
        "--existing-dir",
        default="archive/classifiers/baseline_v7",
        help="Existing dataset dir to exclude duplicate (id:pair_idx)",
    )
    ap.add_argument("--out-dir", default="archive/classifiers/build/0_candidates", help="Output dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=6)
    ap.add_argument("--turns-per-dialog", type=int, default=2)

    ap.add_argument("--train-r0", type=int, default=0)
    ap.add_argument("--train-r1", type=int, default=2000)
    ap.add_argument("--train-r2", type=int, default=1200)
    ap.add_argument("--valid-r0", type=int, default=800)
    ap.add_argument("--valid-r1", type=int, default=400)
    ap.add_argument("--valid-r2", type=int, default=400)
    ap.add_argument("--test-r0", type=int, default=800)
    ap.add_argument("--test-r1", type=int, default=400)
    ap.add_argument("--test-r2", type=int, default=400)
    ap.add_argument("--stop-early", action="store_true", help="Stop scanning when all caps are filled.")
    args = ap.parse_args()

    soulchat_path = Path(args.soulchat_path)
    existing_dir = Path(args.existing_dir)
    out_dir = Path(args.out_dir)

    rng = random.Random(args.seed)
    exclude_keys = _load_existing_keys(existing_dir)

    caps = {
        "train": Cap(args.train_r0, args.train_r1, args.train_r2),
        "valid": Cap(args.valid_r0, args.valid_r1, args.valid_r2),
        "test": Cap(args.test_r0, args.test_r1, args.test_r2),
    }
    picked = {"train": {"R0": 0, "R1": 0, "R2": 0}, "valid": {"R0": 0, "R1": 0, "R2": 0}, "test": {"R0": 0, "R1": 0, "R2": 0}}
    out_rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "valid": [], "test": []}

    scanned = 0
    skipped = 0
    for idx, item in enumerate(_iter_json_array(soulchat_path)):
        scanned = idx + 1
        if not isinstance(item, dict):
            skipped += 1
            continue
        item_id = item.get("id", idx)
        split = _assign_split_from_id(item_id, seed=args.seed)
        if split not in ("train", "valid", "test"):
            continue
        cap = caps[split]
        if cap.remaining() <= 0:
            if args.stop_early and all(c.remaining() <= 0 for c in caps.values()):
                break
            continue

        messages = item.get("messages")
        if not isinstance(messages, list) or not messages:
            skipped += 1
            continue
        pairs = _soulchat_messages_to_pairs(messages)
        if len(pairs) < 2:
            skipped += 1
            continue

        tpd = max(1, int(args.turns_per_dialog))
        chosen: List[int] = []
        # deterministic-ish: mid + last, then fill remaining with random unique indices
        chosen.append(len(pairs) - 1)
        chosen.append(max(0, len(pairs) // 2))
        chosen = list(dict.fromkeys(chosen))  # unique, preserve order
        while len(chosen) < min(tpd, len(pairs)):
            r = rng.randrange(len(pairs))
            if r not in chosen:
                chosen.append(r)

        for pair_idx in chosen[:tpd]:
            key = f"{item_id}:{pair_idx}"
            if key in exclude_keys:
                continue

            context_text, user_focus = _format_pairs_context(pairs, pair_idx, max_pairs=args.max_pairs)
            if not context_text or not user_focus or not _RE_HAS_CJK.search(user_focus):
                skipped += 1
                continue

            heur_risk = _risk_from_text(user_focus)
            if heur_risk == "R0" and cap.r0 <= 0:
                continue
            if heur_risk == "R1" and cap.r1 <= 0:
                continue
            if heur_risk == "R2" and cap.r2 <= 0:
                continue

            heur_emotion = _emotion_from_text(user_focus)
            row = {
                "instruction": RISK_EMOTION_INSTRUCTION,
                "input": context_text,
                "output": json.dumps({"risk": heur_risk, "emotion": heur_emotion}, ensure_ascii=False),
                "meta": {
                    "source": "SoulChatCorpus",
                    "split": split,
                    "id": item_id,
                    "topic": item.get("topic"),
                    "pair_idx": pair_idx,
                    "campus_related": _is_campus_related(user_focus),
                    "derived_from": "augment_candidates",
                },
            }

            out_rows[split].append(row)
            exclude_keys.add(key)
            picked[split][heur_risk] += 1
            if heur_risk == "R0":
                cap.r0 -= 1
            elif heur_risk == "R1":
                cap.r1 -= 1
            elif heur_risk == "R2":
                cap.r2 -= 1

            if args.stop_early and all(c.remaining() <= 0 for c in caps.values()):
                break

        if args.stop_early and all(c.remaining() <= 0 for c in caps.values()):
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        _write_jsonl(out_dir / f"risk_emotion_{split}.jsonl", out_rows[split])

    metrics = {"scanned": scanned, "skipped": skipped, "picked_by_split_risk": picked, "out_dir": str(out_dir)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

