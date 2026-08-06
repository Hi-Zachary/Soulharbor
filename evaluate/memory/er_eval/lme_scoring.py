"""QA and retrieval scoring for SoulHarbor-MH-LongMemEval-30."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set

_ABSTAIN_MARKERS = (
    "历史中没有",
    "没有提供",
    "无法从",
    "没有足够",
    "不足以",
    "无法判断",
    "未提及",
    "未提供",
    "对话中没有",
    "现有历史",
    "无法确定",
)


def _normalize_text(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", "", s)
    for ch in "，。！？；：""''\"'（）()[]【】":
        s = s.replace(ch, "")
    return s


def _deterministic_match(hypothesis: str, answer: str, aliases: Optional[List[str]] = None) -> bool:
    hyp = _normalize_text(hypothesis)
    if not hyp:
        return False
    candidates = [str(answer or "")]
    candidates.extend(str(x or "") for x in (aliases or []))
    for cand in candidates:
        norm = _normalize_text(cand)
        if norm and (norm in hyp or hyp in norm):
            return True
    return False


def _parse_judge_json(raw: object) -> Dict[str, Any]:
    text = str(raw or "").strip()
    try:
        if isinstance(raw, dict):
            return raw
        s, e = text.find("{"), text.rfind("}")
        if s >= 0 and e > s:
            obj = json.loads(text[s : e + 1])
            return obj if isinstance(obj, dict) else {}
    except Exception:
        # Some judge outputs contain unescaped quotes inside "reason".
        # Fall back to extracting the main boolean so we do not flip
        # obviously-correct rows into false negatives.
        m = re.search(r'"correct"\s*:\s*(true|false)', text, flags=re.I)
        if m:
            correct = m.group(1).lower() == "true"
            reason = ""
            rm = re.search(r'"reason"\s*:\s*"(.+)"\s*}', text, flags=re.S)
            if rm:
                reason = rm.group(1).replace('\\"', '"').strip()
            return {"correct": correct, "reason": reason}
    return {}


def _llm_judge(
    llm: Any,
    *,
    question: str,
    gold: str,
    hypothesis: str,
    extra_rules: str = "",
) -> Dict[str, Any]:
    prompt = [
        {
            "role": "user",
            "content": (
                "判定模型答案是否与参考答案语义一致。允许表述不同，但核心事实必须一致。\n"
                f"{extra_rules}\n"
                '输出严格 JSON：{"correct": true/false, "reason": "一句"}\n\n'
                f"问题: {question}\n"
                f"参考答案: {gold}\n"
                f"模型答案: {hypothesis}"
            ),
        }
    ]
    raw = llm.generate_structured(prompt, max_new_tokens=160)
    obj = _parse_judge_json(raw)
    return {
        "correct": bool(obj.get("correct")),
        "reason": str(obj.get("reason") or ""),
        "judge_raw": raw,
    }


def reader_answer_open(llm: Any, *, memory_block: str, question: str) -> str:
    prompt = [
        {
            "role": "user",
            "content": (
                "你是固定答题器。只能根据提供的记忆回答开放题。不要编造。\n"
                "用简洁中文作答。\n\n"
                f"记忆:\n{memory_block or '(空)'}\n\n"
                f"问题: {question}"
            ),
        }
    ]
    return (llm.generate_structured(prompt, max_new_tokens=256) or "").strip()


def score_abstention(
    llm: Any,
    *,
    question: str,
    hypothesis: str,
    aliases: Optional[List[str]] = None,
) -> Dict[str, Any]:
    hyp = str(hypothesis or "").strip()
    norm = _normalize_text(hyp)
    abstain_aliases = list(aliases or []) + list(_ABSTAIN_MARKERS)
    matched = any(_normalize_text(x) and _normalize_text(x) in norm for x in abstain_aliases)
    if matched:
        return {
            "correct": True,
            "reason": "deterministic_abstention_match",
            "pred": hyp,
            "gold": "历史中没有提供相关信息。",
        }
    judged = _llm_judge(
        llm,
        question=question,
        gold="历史中没有提供相关信息。",
        hypothesis=hypothesis,
        extra_rules=(
            "这是一个拒答题。只要模型明确表达“历史/记忆中没有足够信息，因此不能给出具体事实答案”，"
            "就应判为正确。若模型捏造了具体事实，则判错。"
        ),
    )
    judged.update({"pred": hyp, "gold": "历史中没有提供相关信息。"})
    return judged


def score_qa(
    llm: Any,
    *,
    item: Dict[str, Any],
    hypothesis: str,
) -> Dict[str, Any]:
    question = str(item.get("question") or "")
    answer = str(item.get("answer") or "")
    aliases = list(item.get("aliases") or [])
    evaluation = dict(item.get("evaluation") or {})
    etype = str(evaluation.get("type") or "semantic_short_answer")

    if etype == "abstention":
        out = score_abstention(
            llm,
            question=question,
            hypothesis=hypothesis,
            aliases=aliases,
        )
        out["evaluation_type"] = etype
        return out

    if _deterministic_match(hypothesis, answer, aliases):
        return {
            "correct": True,
            "reason": "deterministic_match",
            "pred": hypothesis,
            "gold": answer,
            "evaluation_type": etype,
        }

    if etype == "temporal_order":
        first = str(evaluation.get("first") or "")
        second = str(evaluation.get("second") or "")
        gold = f"先发生“{first}”，之后发生“{second}”。"
        judged = _llm_judge(
            llm,
            question=question,
            gold=gold,
            hypothesis=hypothesis,
            extra_rules=(
                "必须体现正确的时间先后顺序。"
                f"第一个事件是“{first}”，第二个事件是“{second}”。"
            ),
        )
        judged.update({"pred": hypothesis, "gold": gold, "evaluation_type": etype})
        return judged

    if etype == "knowledge_update":
        superseded = list(item.get("superseded_message_ids") or [])
        judged = _llm_judge(
            llm,
            question=question,
            gold=answer,
            hypothesis=hypothesis,
            extra_rules=(
                "答案必须体现当前有效值，而不是已被取代的旧值。"
                "若同时提到旧值，必须明确当前值是什么。"
                f"已被取代的信息来源: {json.dumps(superseded, ensure_ascii=False)}"
            ),
        )
        judged.update({"pred": hypothesis, "gold": answer, "evaluation_type": etype})
        return judged

    if etype == "structured_fields":
        answer_fields = dict(item.get("answer_fields") or {})
        field_scores: Dict[str, bool] = {}
        field_reasons: Dict[str, str] = {}
        for key, gold_val in answer_fields.items():
            if _deterministic_match(hypothesis, str(gold_val)):
                field_scores[key] = True
                field_reasons[key] = "deterministic_match"
                continue
            judged = _llm_judge(
                llm,
                question=f"{question}（字段: {key}）",
                gold=str(gold_val),
                hypothesis=hypothesis,
                extra_rules=f"只判断字段“{key}”是否被正确覆盖。",
            )
            field_scores[key] = bool(judged.get("correct"))
            field_reasons[key] = str(judged.get("reason") or "")
        all_ok = bool(field_scores) and all(field_scores.values())
        return {
            "correct": all_ok,
            "reason": json.dumps(field_reasons, ensure_ascii=False),
            "pred": hypothesis,
            "gold": answer,
            "evaluation_type": etype,
            "field_scores": field_scores,
            "structured_field_accuracy": (
                sum(1 for v in field_scores.values() if v) / len(field_scores)
                if field_scores
                else None
            ),
        }

    judged = _llm_judge(llm, question=question, gold=answer, hypothesis=hypothesis)
    judged.update({"pred": hypothesis, "gold": answer, "evaluation_type": etype})
    return judged


def score_retrieval(
    item: Dict[str, Any],
    *,
    retrieved_source_message_ids: List[str],
    retrieved_anchor_source_message_ids: Optional[List[str]] = None,
    retrieved_anchor_turn_ids: Optional[List[int]] = None,
    retrieved_session_ids: List[str],
) -> Dict[str, Any]:
    if not item.get("answerable", True):
        return {
            "message_recall": None,
            "anchor_message_recall": None,
            "included_message_recall": None,
            "session_recall": None,
            "turn_recall": None,
            "all_evidence_recall": None,
            "all_evidence_included_recall": None,
            "any_evidence_recall": None,
            "included_current": None,
            "stale_only": None,
        }

    gold_messages: Set[str] = set(item.get("evidence_message_ids") or [])
    included_messages = set(retrieved_source_message_ids or [])
    anchor_messages = set(retrieved_anchor_source_message_ids or [])
    gold_sessions = set(item.get("answer_session_ids") or [])
    retrieved_sessions = set(retrieved_session_ids or [])
    superseded = set(item.get("superseded_message_ids") or [])
    gold_turns = {
        str(mid).rsplit("-", 1)[0]
        for mid in gold_messages
        if "-" in str(mid)
    }
    retrieved_turns = {
        str(mid).rsplit("-", 1)[0]
        for mid in included_messages
        if "-" in str(mid)
    }

    anchor_message_recall = (
        len(gold_messages & anchor_messages) / len(gold_messages) if gold_messages else None
    )
    included_message_recall = (
        len(gold_messages & included_messages) / len(gold_messages) if gold_messages else None
    )
    message_recall = included_message_recall
    session_recall = (
        len(gold_sessions & retrieved_sessions) / len(gold_sessions) if gold_sessions else None
    )
    turn_recall = (
        len(gold_turns & retrieved_turns) / len(gold_turns) if gold_turns else None
    )
    all_evidence = bool(gold_messages) and gold_messages <= anchor_messages
    all_evidence_included = bool(gold_messages) and gold_messages <= included_messages
    any_evidence = bool(gold_messages & included_messages)
    stale_only = bool(
        superseded
        and (superseded & included_messages)
        and not (gold_messages & included_messages)
    )
    included_current = bool(gold_messages & included_messages) if gold_messages else None
    return {
        "message_recall": message_recall,
        "anchor_message_recall": anchor_message_recall,
        "included_message_recall": included_message_recall,
        "session_recall": session_recall,
        "turn_recall": turn_recall,
        "all_evidence_recall": all_evidence,
        "all_evidence_included_recall": all_evidence_included,
        "any_evidence_recall": any_evidence,
        "included_current": included_current,
        "stale_only": stale_only,
        "gold_message_count": len(gold_messages),
        "retrieved_message_count": len(included_messages),
        "retrieved_anchor_message_count": len(anchor_messages),
        "retrieved_anchor_turn_count": len(set(retrieved_anchor_turn_ids or [])),
    }


def summarize_lme_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    qa_rows = [r for r in results if r.get("qa") and r["qa"].get("correct") is not None]
    qa_correct = sum(1 for r in qa_rows if r["qa"].get("correct"))
    by_capability: Dict[str, Dict[str, float]] = {}
    by_memory_target: Dict[str, Dict[str, float]] = {}
    by_evaluation_type: Dict[str, Dict[str, float]] = {}
    retrieval_rows = [
        r for r in results if r.get("retrieval") and r["retrieval"].get("message_recall") is not None
    ]
    structured_rows = [
        r for r in results if r.get("qa") and r["qa"].get("structured_field_accuracy") is not None
    ]
    abstain_rows = [
        r for r in results if (r.get("evaluation") or {}).get("type") == "abstention"
    ]

    for row in results:
        cap = str(row.get("capability") or "unknown")
        target = str(row.get("memory_target") or "unknown")
        etype = str((row.get("evaluation") or {}).get("type") or "semantic_short_answer")
        for bucket, key in (
            (by_capability, cap),
            (by_memory_target, target),
            (by_evaluation_type, etype),
        ):
            bucket.setdefault(key, {"correct": 0, "total": 0})
            if row.get("qa") and row["qa"].get("correct") is not None:
                bucket[key]["total"] += 1
                if row["qa"].get("correct"):
                    bucket[key]["correct"] += 1

    def _acc(bucket: Dict[str, Dict[str, float]]) -> Dict[str, Optional[float]]:
        return {
            k: (v["correct"] / v["total"] if v["total"] else None)
            for k, v in sorted(bucket.items())
        }

    def _mean(key: str) -> Optional[float]:
        vals = [float(r["retrieval"][key]) for r in retrieval_rows if r["retrieval"].get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    stale_rows = [r for r in retrieval_rows if r["retrieval"].get("stale_only") is not None]
    stale_rate = (
        sum(1 for r in stale_rows if r["retrieval"].get("stale_only")) / len(stale_rows)
        if stale_rows
        else None
    )

    return {
        "qa": {
            "overall_accuracy": (qa_correct / len(qa_rows)) if qa_rows else None,
            "correct": qa_correct,
            "total": len(qa_rows),
            "by_capability": _acc(by_capability),
            "by_memory_target": _acc(by_memory_target),
            "by_evaluation_type": _acc(by_evaluation_type),
            "structured_field_accuracy": (
                sum(float(r["qa"]["structured_field_accuracy"]) for r in structured_rows)
                / len(structured_rows)
                if structured_rows
                else None
            ),
            "abstention_accuracy": (
                sum(1 for r in abstain_rows if r.get("qa", {}).get("correct")) / len(abstain_rows)
                if abstain_rows
                else None
            ),
        },
        "retrieval": {
            "anchor_message_recall_at_k": _mean("anchor_message_recall"),
            "included_message_recall_at_k": _mean("included_message_recall"),
            "message_recall_at_k": _mean("message_recall"),
            "session_recall_at_k": _mean("session_recall"),
            "turn_recall_at_k": _mean("turn_recall"),
            "all_evidence_recall": _mean("all_evidence_recall"),
            "all_evidence_included_recall": _mean("all_evidence_included_recall"),
            "any_evidence_recall": _mean("any_evidence_recall"),
            "included_current_rate": _mean("included_current"),
            "stale_only_rate": stale_rate,
            "n_answerable": len(retrieval_rows),
        },
    }
