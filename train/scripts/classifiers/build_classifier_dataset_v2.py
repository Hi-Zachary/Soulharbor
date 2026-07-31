# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/build_classifier_dataset_v2.py
# 原先用途: 构建意图（及历史 risk/emotion）分类器训练数据 v2。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

import argparse
import json
import os
import random
import re
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


MOTHER_INSTRUCTION = (
    "你是校园心理健康分流与风险评估模型。根据用户文本判断并只输出严格 JSON："
    "{\"is_consult\":0/1,\"risk\":\"R0|R1|R2\",\"emotion\":\"高兴/积极|悲伤/消极|焦虑/紧张|愤怒/生气|迷茫/困惑|中性/未知\"}。"
    "规则：若 is_consult=0，则 risk 必须为 R0 且 emotion 必须为 中性/未知。不要输出多余文本。"
)

INTENT_INSTRUCTION = (
    "你是校园心理健康分流模型。根据多轮对话文本判断是否属于心理咨询/倾诉，只输出严格 JSON："
    "{\"is_consult\":0/1}。不要输出多余文本。"
)

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
_RE_MENTAL_HINT = re.compile(
    r"(心理|咨询|倾诉|抑郁|焦虑|自杀|想死|割腕|跳楼|吞药|精神|崩溃|失眠|恐慌|创伤|PTSD|强迫|双相)"
)


def _norm_text(s: str) -> str:
    s = (s or "").strip()
    # LCCC-base 常见每个词/字之间带空格
    s = s.replace(" ", "")
    # 轻度清理不可见字符
    s = s.replace("\u200b", "").replace("\ufeff", "")
    return s.strip()


def _is_meaningful_chat_utterance(s: str) -> bool:
    s = _norm_text(s)
    if len(s) < 10 or len(s) > 160:
        return False
    if not _RE_HAS_CJK.search(s):
        return False
    if _RE_MENTAL_HINT.search(s):
        return False
    # 太多重复字符/噪声
    if len(set(s)) / max(1, len(s)) < 0.15:
        return False
    # 避免纯表情/符号
    non_cjk = sum(1 for ch in s if not _RE_HAS_CJK.match(ch))
    if non_cjk / max(1, len(s)) > 0.6:
        return False
    return True


def _pick_daily_sample(dialog: List[str]) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not isinstance(dialog, list) or len(dialog) < 2:
        return None
    texts = [_norm_text(x) for x in dialog if isinstance(x, str)]
    candidates = []
    for i, t in enumerate(texts):
        if _is_meaningful_chat_utterance(t):
            # 轻微偏好：带标点/疑问、更长、更“像一句话”
            score = len(t) + (3 if "？" in t or "?" in t else 0) + (2 if "。" in t else 0)
            candidates.append((score, i, t))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, idx, cur = candidates[0]
    if idx > 0 and _is_meaningful_chat_utterance(texts[idx - 1]):
        prev = texts[idx - 1]
        sample_text = f"上文：{prev}\n用户：{cur}"
    else:
        sample_text = cur
    meta = {"turn_idx": idx, "dialog_len": len(texts)}
    return sample_text, meta


def _to_dialogue_text(raw: str) -> str:
    """
    将“单句/带补充/上文-用户”格式统一转换成带角色标签的多轮文本，
    便于训练时与线上“最近N轮上下文”形式一致。
    """
    raw = (raw or "").strip()
    if not raw:
        return raw

    # 处理 LCCC 负样本中我们拼的“上文/用户”
    if raw.startswith("上文：") and "\n用户：" in raw:
        prev, cur = raw.split("\n用户：", 1)
        prev = prev.removeprefix("上文：").strip()
        cur = cur.strip()
        if prev and cur:
            return f"[USER] {prev}\n[ASSISTANT] 我在，慢慢说。\n[USER] {cur}"
        return f"[USER] {cur or prev}".strip()

    # 处理 PsyQA 的 “问题 + 补充描述”
    if "\n补充：" in raw:
        q, desc = raw.split("\n补充：", 1)
        q = q.strip()
        desc = desc.strip()
        if q and desc:
            return f"[USER] {q}\n[ASSISTANT] 我在，慢慢说。\n[USER] {desc}"
        return f"[USER] {q or desc}".strip()

    return f"[USER] {raw}"


def _risk_from_text(s: str) -> str:
    s = _norm_text(s)
    # R2: 明确意图/计划/实施（最高风险）
    r3 = [
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
    if any(k in s for k in r3):
        return "R2"

    # R1: 强烈绝望/自伤念头但不明确计划，或明显功能崩溃（中度风险）
    r2 = [
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
    if any(k in s for k in r2):
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


def _is_campus_related(s: str) -> bool:
    s = _norm_text(s)
    return bool(_RE_CAMPUS.search(s))


def _campus_score(s: str) -> int:
    """
    校园相关打分（比二值更稳）：用于抽样时“尽量让校园场景占多数”，避免过拟合到少数关键词。
    """
    s = _norm_text(s)
    if not s:
        return 0
    hits = _RE_CAMPUS.findall(s)
    score = len(hits)
    # 对“高信息密度”的校园词做一点加权
    for kw in ("辅导员", "奖学金", "推免", "保研", "考研", "挂科", "绩点", "GPA", "校医院", "心理中心", "宿舍"):
        if kw in s:
            score += 1
    return score


def _load_psyqa(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected PsyQA json type: {type(data)}")
    return data


def _load_multiturn_conversations(paths: List[Path]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Unexpected multi-turn dataset type in {p}: {type(data)}")
        for idx, it in enumerate(data):
            if not isinstance(it, dict) or "conversation" not in it:
                continue
            conv = it.get("conversation")
            if not isinstance(conv, list) or not conv:
                continue
            items.append({"_src_file": p.name, "_src_idx": idx, "conversation": conv})
    return items


def _stable_bucket(text: str, seed: int, modulo: int = 10_000) -> int:
    h = hashlib.md5((str(seed) + "::" + (text or "")).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % modulo


def _assign_split_from_id(item_id: Any, seed: int) -> str:
    """
    SoulChatCorpus 只有一个 train 文件，这里用稳定哈希把样本切成 train/valid/test。
    比例：train 9000 / valid 500 / test 500（约 90/5/5）。
    """
    b = _stable_bucket(str(item_id), seed=seed, modulo=10_000)
    if b < 500:
        return "test"
    if b < 1000:
        return "valid"
    return "train"


def _soulchat_messages_to_pairs(messages: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    将 SoulChatCorpus 的 messages(role/content) 转成 (user, assistant) 序列（尽量对齐）。
    若出现连续 user 或连续 assistant，会尽量跳过不完整对。
    """
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
        else:
            continue
    return pairs


def _format_pairs_context(
    pairs: List[Tuple[str, str]],
    pair_idx: int,
    max_pairs: int = 6,
    end_with_user: bool = True,
) -> Tuple[str, str]:
    """
    从 (user,assistant) pairs 构建分类器输入。
    - 若 end_with_user=True：末尾以 [USER] 当前输入结尾（不包含该对的 assistant）
    - user_focus: 用于规则预标注 risk/emotion（拼接至当前的 user 内容）
    """
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

    u_cur, a_cur = pairs[pair_idx]
    if u_cur:
        lines.append(f"[USER] {u_cur}")
        user_texts.append(u_cur)
    if not end_with_user and a_cur:
        lines.append(f"[ASSISTANT] {a_cur}")

    return "\n".join(lines).strip(), "\n".join(user_texts).strip()


def _format_conversation_context(
    conversation: List[Dict[str, Any]],
    turn_idx: int,
    max_pairs: int = 6,
) -> Tuple[str, str]:
    """
    将多轮咨询数据转换为“线上等价”的分类输入：
    - 取到某一轮用户输入 turn_idx 的“用户发言时刻”
    - 上文包含前若干轮的 [USER]/[ASSISTANT]
    - 末尾以 [USER] 当前输入结尾（不包含该轮的 assistant 输出）

    返回：
    - context_text: 用于分类器输入
    - user_focus_text: 用于规则预标注 risk/emotion（拼接到当前为止的用户输入）
    """
    turns = []
    user_texts = []
    for t in conversation:
        if not isinstance(t, dict):
            continue
        inp = _norm_text(t.get("input", ""))
        out = _norm_text(t.get("output", ""))
        if inp:
            user_texts.append(inp)
        turns.append((inp, out))

    if not turns:
        return "", ""

    turn_idx = max(0, min(turn_idx, len(turns) - 1))
    start = max(0, turn_idx - max_pairs + 1)

    lines: List[str] = []
    for i in range(start, turn_idx):
        ui, ao = turns[i]
        if ui:
            lines.append(f"[USER] {ui}")
        if ao:
            lines.append(f"[ASSISTANT] {ao}")

    cur_user, _cur_assistant = turns[turn_idx]
    if cur_user:
        lines.append(f"[USER] {cur_user}")

    context_text = "\n".join(lines).strip()
    user_focus_text = "\n".join(user_texts[: turn_idx + 1]).strip()
    return context_text, user_focus_text


def _iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[Any]:
    # 优先使用 ijson：更稳的流式 JSON 解析（适合超大 JSON 数组文件）
    try:
        import ijson  # type: ignore

        with path.open("rb") as f:
            for item in ijson.items(f, "item"):
                yield item
        return
    except Exception:
        # ijson 不可用或解析失败则回退到轻量实现
        pass

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buf = ""
        eof = False

        # 找到数组起始 [
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
            # 跳过空白和逗号
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
                        # 文件到达 EOF 仍无法继续解析时，认为 JSON 数组结束（兼容极少数尾部边界/尾随字符情况）
                        return
                    chunk = f.read(chunk_size)
                    if chunk:
                        buf += chunk
                    else:
                        eof = True


@dataclass
class Sample:
    instruction: str
    input: str
    output: str
    meta: Dict[str, Any]

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "instruction": self.instruction,
                "input": self.input,
                "output": self.output,
                "meta": self.meta,
            },
            ensure_ascii=False,
        )


def _make_output(is_consult: int, risk: str, emotion: str) -> str:
    obj = {"is_consult": int(is_consult), "risk": risk, "emotion": emotion}
    return json.dumps(obj, ensure_ascii=False)


def _build_psyqa_split(
    items: List[Dict[str, Any]],
    split: str,
    n_total: int,
    campus_ratio: float,
    seed: int,
) -> List[Sample]:
    rng = random.Random(seed)
    pool_campus = []
    pool_other = []
    for it in items:
        q = _norm_text(it.get("question", ""))
        desc = _norm_text(it.get("description", ""))
        text = q if not desc else f"{q}\n补充：{desc}"
        text = text.strip()
        if not text:
            continue
        rec = (text, it)
        (pool_campus if _is_campus_related(text) else pool_other).append(rec)

    want_campus = int(round(n_total * campus_ratio))
    want_other = n_total - want_campus

    chosen = []
    if len(pool_campus) >= want_campus:
        chosen.extend(rng.sample(pool_campus, want_campus))
    else:
        chosen.extend(pool_campus)
        want_other = n_total - len(chosen)

    if len(pool_other) >= want_other:
        chosen.extend(rng.sample(pool_other, want_other))
    else:
        chosen.extend(pool_other)

    if len(chosen) < n_total:
        raise ValueError(
            f"PsyQA {split}: not enough samples after split pools. got={len(chosen)} need={n_total} "
            f"(campus={len(pool_campus)} other={len(pool_other)})"
        )

    out: List[Sample] = []
    for text, it in chosen:
        risk = _risk_from_text(text)
        emotion = _emotion_from_text(text)
        sample = Sample(
            instruction=MOTHER_INSTRUCTION,
            input=_to_dialogue_text(text),
            output=_make_output(1, risk, emotion),
            meta={
                "source": "PsyQA-hf",
                "split": split,
                "questionID": it.get("questionID"),
                "campus_related": _is_campus_related(text),
            },
        )
        out.append(sample)
    rng.shuffle(out)
    return out


def _build_multiturn_consult_split(
    items: List[Dict[str, Any]],
    split: str,
    n_total: int,
    campus_ratio: float,
    seed: int,
    max_pairs: int = 6,
) -> List[Sample]:
    rng = random.Random(seed)
    pool_campus: List[Dict[str, Any]] = []
    pool_other: List[Dict[str, Any]] = []

    for it in items:
        conv = it.get("conversation", [])
        text_blob = " ".join(
            _norm_text(t.get("input", "")) + " " + _norm_text(t.get("output", ""))
            for t in conv
            if isinstance(t, dict)
        )
        (pool_campus if _is_campus_related(text_blob) else pool_other).append(it)

    want_campus = int(round(n_total * campus_ratio))
    want_other = n_total - want_campus

    chosen: List[Dict[str, Any]] = []
    if len(pool_campus) >= want_campus:
        chosen.extend(rng.sample(pool_campus, want_campus))
    else:
        chosen.extend(pool_campus)
        want_other = n_total - len(chosen)

    if len(pool_other) >= want_other:
        chosen.extend(rng.sample(pool_other, want_other))
    else:
        chosen.extend(pool_other)

    if len(chosen) < n_total:
        raise ValueError(
            f"Multi-turn consult {split}: not enough conversations. got={len(chosen)} need={n_total} "
            f"(campus={len(pool_campus)} other={len(pool_other)})"
        )

    out: List[Sample] = []
    for it in chosen:
        conv = it.get("conversation", [])
        if not isinstance(conv, list) or not conv:
            continue
        turn_idx = rng.randrange(len(conv))
        context_text, user_focus = _format_conversation_context(conv, turn_idx, max_pairs=max_pairs)
        if not context_text:
            continue
        risk = _risk_from_text(user_focus)
        emotion = _emotion_from_text(user_focus)
        out.append(
            Sample(
                instruction=MOTHER_INSTRUCTION,
                input=context_text,
                output=_make_output(1, risk, emotion),
                meta={
                    "source": "datasets-multi_turn",
                    "split": split,
                    "src_file": it.get("_src_file"),
                    "src_idx": it.get("_src_idx"),
                    "turn_idx": turn_idx,
                    "campus_related": _is_campus_related(user_focus),
                },
            )
        )

    if len(out) < n_total:
        # 少量样本可能因为格式问题被跳过，做一次补抽
        remaining = n_total - len(out)
        rest_pool = [x for x in items if x not in chosen]
        if len(rest_pool) < remaining:
            raise ValueError(f"Multi-turn consult {split}: not enough usable conversations after filtering.")
        more = rng.sample(rest_pool, remaining)
        for it in more:
            conv = it.get("conversation", [])
            if not isinstance(conv, list) or not conv:
                continue
            turn_idx = rng.randrange(len(conv))
            context_text, user_focus = _format_conversation_context(conv, turn_idx, max_pairs=max_pairs)
            if not context_text:
                continue
            risk = _risk_from_text(user_focus)
            emotion = _emotion_from_text(user_focus)
            out.append(
                Sample(
                    instruction=MOTHER_INSTRUCTION,
                    input=context_text,
                    output=_make_output(1, risk, emotion),
                    meta={
                        "source": "datasets-multi_turn",
                        "split": split,
                        "src_file": it.get("_src_file"),
                        "src_idx": it.get("_src_idx"),
                        "turn_idx": turn_idx,
                        "campus_related": _is_campus_related(user_focus),
                    },
                )
            )

    out = out[:n_total]
    rng.shuffle(out)
    return out


def _build_soulchat_consult_splits_fullscan(
    path: Path,
    train_k: int,
    valid_k: int,
    test_k: int,
    campus_ratio: float,
    seed: int,
    max_pairs: int = 6,
    turns_per_dialog: int = 3,
) -> Tuple[List[Sample], List[Sample], List[Sample], Dict[str, Any]]:
    """
    尽量全量扫描 SoulChatCorpus（大 JSON 数组），并按稳定哈希切分成 train/valid/test，
    再在每个 split 内对 campus/other 分桶做蓄水池抽样，保证校园相关占比接近 campus_ratio。
    """
    rng = random.Random(seed)
    want = {
        "train": {"campus": int(round(train_k * campus_ratio)), "other": train_k - int(round(train_k * campus_ratio))},
        "valid": {"campus": int(round(valid_k * campus_ratio)), "other": valid_k - int(round(valid_k * campus_ratio))},
        "test": {"campus": int(round(test_k * campus_ratio)), "other": test_k - int(round(test_k * campus_ratio))},
    }

    reservoirs: Dict[str, Dict[str, List[Sample]]] = {
        "train": {"campus": [], "other": []},
        "valid": {"campus": [], "other": []},
        "test": {"campus": [], "other": []},
    }
    eligible_seen: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    scanned = 0
    skipped = 0

    for idx, item in enumerate(_iter_json_array(path)):
        scanned = idx + 1
        if not isinstance(item, dict):
            skipped += 1
            continue
        item_id = item.get("id", idx)
        split = _assign_split_from_id(item_id, seed=seed)
        if split not in ("train", "valid", "test"):
            continue

        messages = item.get("messages")
        if not isinstance(messages, list) or not messages:
            skipped += 1
            continue
        pairs = _soulchat_messages_to_pairs(messages)
        if len(pairs) < 2:
            skipped += 1
            continue

        # 每段对话抽取多个“用户发言时刻”作为候选样本（更贴近线上：每轮用户输入都要判断）
        tpd = max(1, int(turns_per_dialog))
        if len(pairs) <= tpd:
            chosen_pair_idx = list(range(len(pairs)))
        else:
            chosen_pair_idx = rng.sample(range(len(pairs)), tpd)

        for pair_idx in chosen_pair_idx:
            context_text, user_focus = _format_pairs_context(
                pairs, pair_idx, max_pairs=max_pairs, end_with_user=True
            )
            if not context_text or not user_focus:
                skipped += 1
                continue

            is_campus = _is_campus_related(user_focus)
            bucket = "campus" if is_campus else "other"
            cap = want[split][bucket]
            if cap <= 0:
                continue

            eligible_seen[split][bucket] += 1
            risk = _risk_from_text(user_focus)
            emotion = _emotion_from_text(user_focus)
            sample = Sample(
                instruction=MOTHER_INSTRUCTION,
                input=context_text,
                output=_make_output(1, risk, emotion),
                meta={
                    "source": "SoulChatCorpus",
                    "split": split,
                    "id": item_id,
                    "topic": item.get("topic"),
                    "pair_idx": pair_idx,
                    "campus_related": is_campus,
                },
            )

            res = reservoirs[split][bucket]
            if len(res) < cap:
                res.append(sample)
            else:
                j = rng.randrange(eligible_seen[split][bucket])
                if j < cap:
                    res[j] = sample

    def _combine(split: str) -> List[Sample]:
        out = reservoirs[split]["campus"] + reservoirs[split]["other"]
        rng.shuffle(out)
        return out

    train = _combine("train")
    valid = _combine("valid")
    test = _combine("test")

    metrics = {
        "scanned": scanned,
        "skipped": skipped,
        "eligible_seen": {s: dict(b) for s, b in eligible_seen.items()},
        "got": {
            "train": {"campus": len(reservoirs["train"]["campus"]), "other": len(reservoirs["train"]["other"])},
            "valid": {"campus": len(reservoirs["valid"]["campus"]), "other": len(reservoirs["valid"]["other"])},
            "test": {"campus": len(reservoirs["test"]["campus"]), "other": len(reservoirs["test"]["other"])},
        },
        "want": want,
    }

    # 若某桶样本不足（极少见），允许用另一桶补齐（牺牲 campus_ratio 但保证数量）
    def _topup(split: str, k_total: int) -> List[Sample]:
        out = reservoirs[split]["campus"] + reservoirs[split]["other"]
        if len(out) >= k_total:
            rng.shuffle(out)
            return out[:k_total]
        # 不足则报错（说明数据文件或过滤逻辑异常）
        raise ValueError(f"SoulChatCorpus {split}: not enough samples. got={len(out)} need={k_total}")

    train = _topup("train", train_k)
    valid = _topup("valid", valid_k)
    test = _topup("test", test_k)
    return train, valid, test, metrics


def _sample_lccc_stream_until(
    path: Path,
    split: str,
    k: int,
    seed: int,
    pool_multiplier: int = 8,
    max_scan: int = 250_000,
    campus_ratio: float = 0.5,
) -> List[Sample]:
    """
    从超大 LCCC-base_train.json 中快速抽取日常负样本（避免扫完整个文件导致耗时过长）。

    注意：
    - 这是“近似随机抽样”：先扫描文件前 max_scan 条对话，筛出可读候选池，再从中随机采样 k 条。
    - 对我们的目的（高质量负样本）更重要的是“过滤质量”，而不是严格均匀覆盖全量。
    """
    rng = random.Random(seed)
    pool_target = max(k + 200, k * max(2, pool_multiplier))
    candidates_campus: List[Sample] = []
    candidates_other: List[Sample] = []
    scanned = 0

    for idx, dialog in enumerate(_iter_json_array(path)):
        scanned = idx + 1
        if idx >= max_scan and (len(candidates_campus) + len(candidates_other)) >= k:
            break
        picked = _pick_daily_sample(dialog)
        if not picked:
            continue
        text, extra_meta = picked
        text_norm = _norm_text(text)
        sample = Sample(
            instruction=MOTHER_INSTRUCTION,
            input=f"[USER] {text_norm}",
            output=_make_output(0, "R0", "中性/未知"),
            meta={
                "source": "LCCC-base",
                "split": split,
                "idx": idx,
                **extra_meta,
                "campus_related": _is_campus_related(text_norm),
            },
        )
        (candidates_campus if sample.meta.get("campus_related") else candidates_other).append(sample)
        if len(candidates_campus) + len(candidates_other) >= pool_target:
            break

    candidates_total = len(candidates_campus) + len(candidates_other)
    if candidates_total < k:
        raise ValueError(
            f"LCCC {split}: not enough eligible dialogs within scan window. "
            f"got={candidates_total} need={k} scanned={scanned}"
        )
    want_campus = int(round(k * campus_ratio))
    want_other = k - want_campus
    if len(candidates_campus) < want_campus:
        want_other = k - len(candidates_campus)
        want_campus = len(candidates_campus)
    if len(candidates_other) < want_other:
        want_campus = min(len(candidates_campus), k - len(candidates_other))
        want_other = k - want_campus
    out = []
    if want_campus:
        out.extend(rng.sample(candidates_campus, want_campus))
    if want_other:
        out.extend(rng.sample(candidates_other, want_other))
    rng.shuffle(out)
    return out


def _reservoir_sample_lccc_full_scan(
    path: Path,
    split: str,
    k: int,
    seed: int,
    campus_ratio: float = 0.5,
) -> Tuple[List[Sample], Dict[str, int]]:
    """
    全量扫描（近似均匀覆盖全量）：对“通过过滤”的日常样本做蓄水池抽样。
    这会扫描整个大文件，耗时更久，但不会忽略后面的数据。
    """
    rng = random.Random(seed)
    want_campus = int(round(k * campus_ratio))
    want_other = k - want_campus
    res_campus: List[Sample] = []
    res_other: List[Sample] = []
    eligible_seen_campus = 0
    eligible_seen_other = 0
    scanned = 0

    for idx, dialog in enumerate(_iter_json_array(path)):
        scanned = idx + 1
        picked = _pick_daily_sample(dialog)
        if not picked:
            continue
        text, extra_meta = picked
        text_norm = _norm_text(text)
        is_campus = _is_campus_related(text_norm)
        if is_campus:
            eligible_seen_campus += 1
        else:
            eligible_seen_other += 1
        sample = Sample(
            instruction=MOTHER_INSTRUCTION,
            input=f"[USER] {text_norm}",
            output=_make_output(0, "R0", "中性/未知"),
            meta={
                "source": "LCCC-base",
                "split": split,
                "idx": idx,
                **extra_meta,
                "campus_related": is_campus,
            },
        )

        if is_campus:
            if want_campus <= 0:
                continue
            if len(res_campus) < want_campus:
                res_campus.append(sample)
                continue
            j = rng.randrange(eligible_seen_campus)
            if j < want_campus:
                res_campus[j] = sample
        else:
            if want_other <= 0:
                continue
            if len(res_other) < want_other:
                res_other.append(sample)
                continue
            j = rng.randrange(eligible_seen_other)
            if j < want_other:
                res_other[j] = sample

    reservoir = res_campus + res_other
    if len(reservoir) < k:
        raise ValueError(
            f"LCCC {split}: not enough eligible dialogs after full scan. got={len(reservoir)} need={k}"
        )
    rng.shuffle(reservoir)
    return reservoir, {
        "scanned": scanned,
        "eligible_seen_campus": eligible_seen_campus,
        "eligible_seen_other": eligible_seen_other,
        "want_campus": want_campus,
        "want_other": want_other,
    }


def _sample_lccc_small(
    path: Path, split: str, k: int, seed: int
) -> List[Sample]:
    rng = random.Random(seed)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    candidates: List[Tuple[int, Sample]] = []
    for idx, dialog in enumerate(data):
        picked = _pick_daily_sample(dialog)
        if not picked:
            continue
        text, extra_meta = picked
        sample = Sample(
            instruction=MOTHER_INSTRUCTION,
            input=f"[USER] {_norm_text(text)}",
            output=_make_output(0, "R0", "中性/未知"),
            meta={
                "source": "LCCC-base",
                "split": split,
                "idx": idx,
                **extra_meta,
            },
        )
        candidates.append((idx, sample))
    if len(candidates) < k:
        raise ValueError(f"LCCC {split}: not enough eligible dialogs. got={len(candidates)} need={k}")
    chosen = rng.sample(candidates, k)
    out = [s for _, s in chosen]
    rng.shuffle(out)
    return out


def _write_jsonl(path: Path, samples: List[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.to_jsonl() + "\n")


def _derive_intent_samples(mother: List[Sample]) -> List[Sample]:
    out: List[Sample] = []
    for s in mother:
        o = json.loads(s.output)
        out.append(
            Sample(
                instruction=INTENT_INSTRUCTION,
                input=s.input,
                output=json.dumps({"is_consult": int(o["is_consult"])}, ensure_ascii=False),
                meta={**(s.meta or {}), "derived_from": "mother"},
            )
        )
    return out


def _derive_risk_emotion_samples(mother: List[Sample]) -> List[Sample]:
    out: List[Sample] = []
    for s in mother:
        o = json.loads(s.output)
        if int(o["is_consult"]) != 1:
            continue
        out.append(
            Sample(
                instruction=RISK_EMOTION_INSTRUCTION,
                input=s.input,
                output=json.dumps({"risk": o["risk"], "emotion": o["emotion"]}, ensure_ascii=False),
                meta={**(s.meta or {}), "derived_from": "mother"},
            )
        )
    return out


def _stats(samples: List[Sample]) -> Dict[str, Any]:
    by_source = Counter()
    by_consult = Counter()
    by_risk = Counter()
    by_emotion = Counter()
    campus_yes = 0
    campus_total = 0
    for s in samples:
        meta = s.meta or {}
        by_source[meta.get("source", "unknown")] += 1
        out = json.loads(s.output)
        by_consult[str(out["is_consult"])] += 1
        by_risk[out["risk"]] += 1
        by_emotion[out["emotion"]] += 1
        if meta.get("source") in ("PsyQA-hf", "datasets-multi_turn", "SoulChatCorpus"):
            campus_total += 1
            if meta.get("campus_related"):
                campus_yes += 1
    return {
        "total": len(samples),
        "by_source": dict(by_source),
        "by_is_consult": dict(by_consult),
        "by_risk": dict(by_risk),
        "by_emotion": dict(by_emotion),
        "consult_campus_ratio": (campus_yes / campus_total) if campus_total else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--psyqa-dir", default="data/PsyQA-hf")
    ap.add_argument("--lccc-dir", default="archive/sources/lccc")
    ap.add_argument("--out-dir", default="data/classifiers/generated")
    ap.add_argument("--seed", type=int, default=20260412)
    ap.add_argument(
        "--consult-source",
        choices=["datasets", "psyqa", "soulchat"],
        default="soulchat",
        help="咨询/倾诉样本来源：soulchat=SoulChatCorpus；psyqa=使用 PsyQA-hf（已归档）。",
    )
    ap.add_argument(
        "--soulchat-path",
        default="archive/sources/soulchat/SoulChatCorpus-sft-multi-Turn.json",
        help="SoulChatCorpus 主文件路径（JSON 数组）。",
    )
    ap.add_argument(
        "--soulchat-turns-per-dialog",
        type=int,
        default=3,
        help="从每段 SoulChat 多轮对话中抽取的用户发言时刻数量（越大越贴近线上，但全量扫描会更慢）。",
    )
    ap.add_argument("--psyqa-train", type=int, default=2000)
    ap.add_argument("--psyqa-valid", type=int, default=200)
    ap.add_argument("--psyqa-test", type=int, default=200)
    ap.add_argument("--lccc-train", type=int, default=1000)
    ap.add_argument("--lccc-valid", type=int, default=100)
    ap.add_argument("--lccc-test", type=int, default=100)
    ap.add_argument(
        "--chat-campus-ratio",
        type=float,
        default=0.5,
        help="日常负样本（LCCC）中校园相关对话的目标占比，用于减少“校园词=咨询”的偏置。",
    )
    ap.add_argument(
        "--lccc-train-sampling",
        choices=["full", "fast"],
        default="full",
        help="LCCC-base train 的抽样方式：full=全量扫描(更慢更全面)；fast=只扫描前一部分(更快)。",
    )
    ap.add_argument("--campus-ratio", type=float, default=0.8)
    args = ap.parse_args()

    psyqa_dir = Path(args.psyqa_dir)
    lccc_dir = Path(args.lccc_dir)
    out_dir = Path(args.out_dir)

    train_samples: List[Sample] = []
    valid_samples: List[Sample] = []
    test_samples: List[Sample] = []

    if args.consult_source == "psyqa":
        psyqa_train = _load_psyqa(psyqa_dir / "train.json")
        psyqa_valid = _load_psyqa(psyqa_dir / "valid.json")
        psyqa_test = _load_psyqa(psyqa_dir / "test.json")
        train_samples += _build_psyqa_split(
            psyqa_train, "train", args.psyqa_train, args.campus_ratio, args.seed + 1
        )
        valid_samples += _build_psyqa_split(
            psyqa_valid, "valid", args.psyqa_valid, args.campus_ratio, args.seed + 2
        )
        test_samples += _build_psyqa_split(
            psyqa_test, "test", args.psyqa_test, args.campus_ratio, args.seed + 3
        )
    else:
        if args.consult_source == "datasets":
            dataset_dir = Path("data/datasets")
            consult_items = _load_multiturn_conversations(
                [
                    dataset_dir / "multi_turn_dataset_1.json",
                    dataset_dir / "multi_turn_dataset_2.json",
                    dataset_dir / "data_pro.json",
                    dataset_dir / "data.json",
                ]
            )
            rng = random.Random(args.seed)
            rng.shuffle(consult_items)
            need_total = args.psyqa_train + args.psyqa_valid + args.psyqa_test
            if len(consult_items) < need_total:
                raise ValueError(
                    f"Not enough consult conversations in data/datasets. got={len(consult_items)} need={need_total}"
                )
            train_pool = consult_items[: args.psyqa_train]
            valid_pool = consult_items[args.psyqa_train : args.psyqa_train + args.psyqa_valid]
            test_pool = consult_items[args.psyqa_train + args.psyqa_valid : need_total]

            train_samples += _build_multiturn_consult_split(
                train_pool, "train", args.psyqa_train, args.campus_ratio, args.seed + 1
            )
            valid_samples += _build_multiturn_consult_split(
                valid_pool, "valid", args.psyqa_valid, args.campus_ratio, args.seed + 2
            )
            test_samples += _build_multiturn_consult_split(
                test_pool, "test", args.psyqa_test, args.campus_ratio, args.seed + 3
            )
        else:
            soulchat_path = Path(args.soulchat_path)
            train_sc, valid_sc, test_sc, sc_metrics = _build_soulchat_consult_splits_fullscan(
                soulchat_path,
                train_k=args.psyqa_train,
                valid_k=args.psyqa_valid,
                test_k=args.psyqa_test,
                campus_ratio=args.campus_ratio,
                seed=args.seed + 101,
                turns_per_dialog=args.soulchat_turns_per_dialog,
            )
            train_samples += train_sc
            valid_samples += valid_sc
            test_samples += test_sc

    lccc_train_metrics: Optional[Dict[str, Any]] = None
    if args.lccc_train_sampling == "full":
        try:
            lccc_train_samples, lccc_train_metrics = _reservoir_sample_lccc_full_scan(
                lccc_dir / "LCCC-base_train.json",
                "train",
                args.lccc_train,
                args.seed + 11,
                campus_ratio=args.chat_campus_ratio,
            )
            train_samples += lccc_train_samples
        except Exception as e:
            # 尽量全量扫：失败则降级 fast，避免卡死整个流程
            lccc_train_metrics = {"mode": "full_failed_fallback_fast", "error": repr(e)}
            train_samples += _sample_lccc_stream_until(
                lccc_dir / "LCCC-base_train.json",
                "train",
                args.lccc_train,
                args.seed + 11,
                campus_ratio=args.chat_campus_ratio,
            )
    else:
        train_samples += _sample_lccc_stream_until(
            lccc_dir / "LCCC-base_train.json",
            "train",
            args.lccc_train,
            args.seed + 11,
            campus_ratio=args.chat_campus_ratio,
        )
    valid_samples += _sample_lccc_small(
        lccc_dir / "LCCC-base_valid.json", "valid", args.lccc_valid, args.seed + 12
    )
    test_samples += _sample_lccc_small(
        lccc_dir / "LCCC-base_test.json", "test", args.lccc_test, args.seed + 13
    )

    rng = random.Random(args.seed)
    rng.shuffle(train_samples)
    rng.shuffle(valid_samples)
    rng.shuffle(test_samples)

    _write_jsonl(out_dir / "train.jsonl", train_samples)
    _write_jsonl(out_dir / "valid.jsonl", valid_samples)
    _write_jsonl(out_dir / "test.jsonl", test_samples)

    # 派生意图二分类训练数据
    intent_train = _derive_intent_samples(train_samples)
    intent_valid = _derive_intent_samples(valid_samples)
    intent_test = _derive_intent_samples(test_samples)
    _write_jsonl(out_dir / "intent_train.jsonl", intent_train)
    _write_jsonl(out_dir / "intent_valid.jsonl", intent_valid)
    _write_jsonl(out_dir / "intent_test.jsonl", intent_test)

    stats = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "sizes": {
            "train": len(train_samples),
            "valid": len(valid_samples),
            "test": len(test_samples),
        },
        "derived_sizes": {
            "intent_train": len(intent_train),
            "intent_valid": len(intent_valid),
            "intent_test": len(intent_test),
        },
        "train": _stats(train_samples),
        "valid": _stats(valid_samples),
        "test": _stats(test_samples),
        "config": {
            "psyqa": {
                "train": args.psyqa_train,
                "valid": args.psyqa_valid,
                "test": args.psyqa_test,
                "campus_ratio": args.campus_ratio,
            },
            "consult_source": args.consult_source,
            "lccc": {
                "train": args.lccc_train,
                "valid": args.lccc_valid,
                "test": args.lccc_test,
                "train_sampling": args.lccc_train_sampling,
            },
        },
        "notes": "标签为规则引擎自动预标注，建议后续人工抽检修订。",
    }
    if lccc_train_metrics:
        stats["lccc_train_metrics"] = lccc_train_metrics
    if args.consult_source == "soulchat":
        stats["soulchat_metrics"] = sc_metrics
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Wrote dataset to: {out_dir}")
    print(json.dumps(stats["sizes"], ensure_ascii=False))


if __name__ == "__main__":
    main()
