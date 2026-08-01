#!/usr/bin/env python3
"""Long-horizon QA eval for SoulHarbor episodic memory (API LLM).

Only evaluates the new episodic backend (no Mem0 / no legacy Full).
Mem0 baseline from 2026-07-30 remains the frozen reference.

  python run_qa_episodic.py --selftest
  python run_qa_episodic.py --limit 1 --workers 1
  python run_qa_episodic.py --workers 4 --qa-workers 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
PROJECT = Path("/root/autodl-tmp/SoulHarbor")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT))

from api_llm import APILLM  # noqa: E402

DATA_DEFAULT = PROJECT / "evaluate/memory/data/all_30.jsonl"
EMBED_DEFAULT = PROJECT / "models/encoders/bge-m3"
BASE_TS = 1_700_000_000

_PRINT_LOCK = threading.Lock()
_WRITE_LOCK = threading.Lock()


def load_config() -> dict:
    p = HERE / "config.json"
    if not p.exists():
        sys.exit(f"缺少 {p}。请从 config.example.json 复制并填入 api_key。")
    cfg = json.loads(p.read_text(encoding="utf-8"))
    if not cfg.get("api_key") or "换成" in str(cfg.get("api_key", "")):
        sys.exit("config.json 里 api_key 没填。")
    cfg.setdefault("http_pool_size", 64)
    return cfg


def week_ts(week: int) -> int:
    return int(BASE_TS + max(1, int(week or 1)) * 7 * 86400)


def _prepare_cpu_embedder() -> None:
    from product_app.app.memory.embeddings import MemoryEmbedder

    MemoryEmbedder.reset()
    emb = MemoryEmbedder.shared()
    emb._device = "cpu"
    emb._use_fp16 = False
    lock = threading.RLock()
    orig = emb.embed_batch

    def locked(texts: List[str]) -> List[List[float]]:
        with lock:
            return orig(texts)

    emb.embed_batch = locked  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Episodic method
# ---------------------------------------------------------------------------


class EpisodicMethod:
    """Ingest conversations into MemoryEngine; retrieve via build_context."""

    def __init__(self, *, work_dir: Path, llm: Any) -> None:
        from product_app.app.memory.engine import MemoryEngine

        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = work_dir / "episodic.db"
        self._llm = llm
        self._engine = MemoryEngine(self._db_path, llm=llm)
        self._user_id = 1
        self._next_message_id = 1
        self._next_conversation_id = 1
        self._sid_to_cid: Dict[str, int] = {}

    def ingest_conversation(self, conv: Dict[str, Any]) -> None:
        sid = str(conv.get("sid") or f"s{self._next_conversation_id}")
        if sid not in self._sid_to_cid:
            self._sid_to_cid[sid] = self._next_conversation_id
            self._next_conversation_id += 1
        cid = self._sid_to_cid[sid]
        created = week_ts(int(conv.get("week") or 1))
        for pos, msg in enumerate(conv.get("messages") or [], start=1):
            role = str(msg.get("role") or "")
            content = str(msg.get("content") or "")
            if role not in ("user", "assistant") or not content.strip():
                continue
            mid = self._next_message_id
            self._next_message_id += 1
            self._engine.ingest_message(
                user_id=self._user_id,
                conversation_id=cid,
                message_id=mid,
                role=role,
                content=content,
                position=pos,
                created_at=created + pos,
            )

    def retrieve(self, query: str) -> str:
        # Delayed QA: no current-session raw messages in the reader prompt.
        return self._engine.build_context(
            user_id=self._user_id,
            conversation_id=0,
            current_user_message=query,
            recent_messages=[],
            conversation_summary=None,
            exclude_message_ids=set(),
        )

    def list_facts(self) -> List[str]:
        info = self._engine.inspect(self._user_id)
        prefs = [str(p.get("content") or "") for p in (info.get("support_preferences") or [])]
        chunks = self._engine._store.list_active_with_embeddings(self._user_id, limit=2000)
        texts = [c.content.strip() for c in chunks if (c.content or "").strip() and c.role == "user"]
        # Prefer unique user lines; cap for judge prompt size
        seen = set()
        out: List[str] = []
        for t in prefs + texts:
            if t in seen:
                continue
            seen.add(t)
            out.append(t[:200])
            if len(out) >= 80:
                break
        return out

    def store_snapshot(self) -> Dict[str, Any]:
        info = self._engine.inspect(self._user_id)
        stats = self._engine._store.index_stats(self._user_id)
        return {
            "backend": "aer",
            "episode_chunks": info.get("episode_chunks"),
            "support_preferences": info.get("support_preferences"),
            "index": stats,
            "facts": self.list_facts()[:40],
            "last_trace": (self._engine.last_trace.to_log_dict() if self._engine.last_trace else None),
        }


# ---------------------------------------------------------------------------
# Reader + scoring
# ---------------------------------------------------------------------------


def _extract_choice(text: str) -> str:
    raw = (text or "").strip().upper()
    m = re.search(r"\b([ABCDE])\b", raw)
    if m:
        return m.group(1)
    m = re.search(r"^([ABCDE])", raw)
    return m.group(1) if m else ""


def reader_answer_mcq(llm: Any, *, memory_block: str, question: Dict[str, Any]) -> Dict[str, Any]:
    opts = question.get("options") or []
    prompt = [
        {
            "role": "user",
            "content": (
                "你是固定答题器。只能根据提供的记忆回答选择题。不要编造记忆里没有的信息。\n"
                "只输出一个选项字母（A/B/C/...），不要解释。\n\n"
                f"记忆:\n{memory_block or '(空)'}\n\n"
                f"问题: {question.get('question')}\n"
                f"选项:\n" + "\n".join(str(o) for o in opts)
            ),
        }
    ]
    raw = llm.generate_structured(prompt, max_new_tokens=32)
    pred = _extract_choice(raw)
    gold = str(question.get("gold") or "").strip().upper()
    return {"raw": raw, "pred": pred, "gold": gold, "correct": bool(pred) and pred == gold}


def reader_answer_judge(llm: Any, *, memory_block: str, question: Dict[str, Any]) -> Dict[str, Any]:
    prompt = [
        {
            "role": "user",
            "content": (
                "你是固定答题器。只能根据提供的记忆回答开放题。不要编造。\n"
                "用一两句中文作答。\n\n"
                f"记忆:\n{memory_block or '(空)'}\n\n"
                f"问题: {question.get('question')}"
            ),
        }
    ]
    answer = (llm.generate_structured(prompt, max_new_tokens=256) or "").strip()
    judge_prompt = [
        {
            "role": "user",
            "content": (
                "判定模型答案是否与参考答案语义一致（允许表述不同）。输出严格 JSON：\n"
                '{"correct": true/false, "reason": "一句"}\n\n'
                f"问题: {question.get('question')}\n"
                f"参考答案: {question.get('gold')}\n"
                f"模型答案: {answer}"
            ),
        }
    ]
    raw = llm.generate_structured(judge_prompt, max_new_tokens=128)
    correct = False
    reason = ""
    try:
        s, e = raw.find("{"), raw.rfind("}")
        if s >= 0 and e > s:
            obj = json.loads(raw[s : e + 1])
            correct = bool(obj.get("correct"))
            reason = str(obj.get("reason") or "")
    except Exception:
        correct = False
    return {
        "raw": answer,
        "pred": answer,
        "gold": question.get("gold"),
        "correct": correct,
        "judge_raw": raw,
        "reason": reason,
    }


def content_f1(llm: Any, *, facts: List[str], gold_facts: Dict[str, Any]) -> Dict[str, Any]:
    should_remember = list(gold_facts.get("should_remember") or [])
    should_forget = list(gold_facts.get("should_forget") or [])
    if not should_remember and not should_forget:
        return {"precision": None, "recall": None, "f1": None}
    prompt = [
        {
            "role": "user",
            "content": (
                "比较记忆库事实与 gold。输出严格 JSON：\n"
                '{"remembered_ok": ["命中的应记"], "forgotten_ok": ["正确未存的应忘"], '
                '"false_remember": ["误存的应忘"], "missed": ["漏记的应记"]}\n\n'
                f"store_facts:\n{json.dumps(facts, ensure_ascii=False)}\n\n"
                f"should_remember:\n{json.dumps(should_remember, ensure_ascii=False)}\n\n"
                f"should_forget:\n{json.dumps(should_forget, ensure_ascii=False)}"
            ),
        }
    ]
    raw = llm.generate_structured(prompt, max_new_tokens=512)
    obj: Dict[str, Any] = {}
    try:
        s, e = raw.find("{"), raw.rfind("}")
        if s >= 0 and e > s:
            parsed = json.loads(raw[s : e + 1])
            if isinstance(parsed, dict):
                obj = parsed
    except Exception:
        obj = {}
    tp = len(obj.get("remembered_ok") or [])
    missed = len(obj.get("missed") or [])
    false_pos = len(obj.get("false_remember") or [])
    if not obj and should_remember:
        missed = len(should_remember)
    precision = tp / (tp + false_pos) if (tp + false_pos) else (1.0 if not should_forget else 0.0)
    recall = tp / (tp + missed) if (tp + missed) else (1.0 if not should_remember else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "detail": obj, "raw": raw}


def score_questions(answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(answers)
    correct = sum(1 for a in answers if a.get("correct"))
    overgen_rows = [a for a in answers if str(a.get("probes") or "") == "overgeneralization"]
    stale_rows = [a for a in answers if str(a.get("probes") or "") in {"staleness", "current"}]
    overgen = sum(1 for a in overgen_rows if not a.get("correct"))
    stale = sum(1 for a in stale_rows if not a.get("correct"))
    return {
        "qa_accuracy": (correct / n) if n else None,
        "correct": correct,
        "total": n,
        "overgeneralization_rate": (overgen / len(overgen_rows)) if overgen_rows else None,
        "overgen_wrong": overgen,
        "overgen_total": len(overgen_rows),
        "staleness_rate": (stale / len(stale_rows)) if stale_rows else None,
        "stale_wrong": stale,
        "stale_total": len(stale_rows),
    }


def _answer_one(llm: Any, engine: EpisodicMethod, q: Dict[str, Any]) -> Dict[str, Any]:
    mem_block = engine.retrieve(str(q.get("question") or ""))
    qtype = str(q.get("type") or "mcq")
    if qtype == "mcq":
        ans = reader_answer_mcq(llm, memory_block=mem_block, question=q)
    else:
        ans = reader_answer_judge(llm, memory_block=mem_block, question=q)
    ans["qid"] = q.get("qid")
    ans["type"] = qtype
    ans["probes"] = q.get("probes")
    ans["memory_block"] = mem_block
    return ans


def run_case(case: Dict[str, Any], *, llm: Any, work_dir: Path, qa_workers: int) -> Dict[str, Any]:
    case_id = str(case["case_id"])
    case_dir = work_dir / case_id
    if case_dir.exists():
        shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Any] = {"case_id": case_id, "category": case.get("category"), "methods": {}}
    try:
        engine = EpisodicMethod(work_dir=case_dir / "episodic", llm=llm)
        for conv in case.get("conversations") or []:
            engine.ingest_conversation(conv)

        questions = list(case.get("test_questions") or [])
        if qa_workers <= 1 or len(questions) <= 1:
            answers = [_answer_one(llm, engine, q) for q in questions]
        else:
            with ThreadPoolExecutor(max_workers=min(qa_workers, len(questions))) as qex:
                futs = [qex.submit(_answer_one, llm, engine, q) for q in questions]
                answers = [f.result() for f in futs]

        f1 = content_f1(llm, facts=engine.list_facts(), gold_facts=dict(case.get("gold_facts") or {}))
        score = score_questions(answers)
        score["content_f1"] = f1.get("f1")
        out["methods"]["episodic"] = {
            "method": "episodic",
            "score": score,
            "answers": answers,
            "content_f1_detail": f1,
            "store": engine.store_snapshot(),
        }
    except Exception as exc:
        out["methods"]["episodic"] = {"method": "episodic", "error": f"{type(exc).__name__}: {exc}"}
    return out


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    method = "episodic"
    rows = [
        r
        for r in results
        if method in (r.get("methods") or {}) and "score" in (r["methods"][method] or {})
    ]
    if not rows:
        return {"methods": {method: {"n": 0}}}
    qa_sum = 0.0
    qa_n = 0
    over_w = over_t = 0
    stale_w = stale_t = 0
    f1_sum = 0.0
    f1_n = 0
    by_cat: Dict[str, Dict[str, float]] = {}
    by_probe: Dict[str, Dict[str, int]] = {}
    for r in rows:
        sc = r["methods"][method]["score"]
        if sc.get("qa_accuracy") is not None:
            qa_sum += float(sc["qa_accuracy"])
            qa_n += 1
        over_w += int(sc.get("overgen_wrong") or 0)
        over_t += int(sc.get("overgen_total") or 0)
        stale_w += int(sc.get("stale_wrong") or 0)
        stale_t += int(sc.get("stale_total") or 0)
        if sc.get("content_f1") is not None:
            f1_sum += float(sc["content_f1"])
            f1_n += 1
        cat = str(r.get("category") or "")
        by_cat.setdefault(cat, {"qa_sum": 0.0, "n": 0})
        if sc.get("qa_accuracy") is not None:
            by_cat[cat]["qa_sum"] += float(sc["qa_accuracy"])
            by_cat[cat]["n"] += 1
        for a in r["methods"][method].get("answers") or []:
            probe = str(a.get("probes") or "unknown")
            by_probe.setdefault(probe, {"correct": 0, "total": 0})
            by_probe[probe]["total"] += 1
            if a.get("correct"):
                by_probe[probe]["correct"] += 1
    return {
        "methods": {
            method: {
                "n": len(rows),
                "qa_accuracy": (qa_sum / qa_n) if qa_n else None,
                "overgeneralization_rate": (over_w / over_t) if over_t else None,
                "staleness_rate": (stale_w / stale_t) if stale_t else None,
                "content_f1": (f1_sum / f1_n) if f1_n else None,
                "subclass_qa": {
                    c: (v["qa_sum"] / v["n"] if v["n"] else None) for c, v in sorted(by_cat.items())
                },
                "probe_qa": {
                    p: (v["correct"] / v["total"] if v["total"] else None)
                    for p, v in sorted(by_probe.items())
                },
            }
        },
    }


def _load_cases(path: Path, *, limit: int, categories: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if categories:
        want = set(categories)
        rows = [r for r in rows if r.get("category") in want]
    if limit > 0:
        return rows[:limit]
    return rows


def selftest(cfg: dict) -> None:
    llm = APILLM(cfg)
    out = llm.generate_structured([{"role": "user", "content": "只回复两个字：OK"}], max_new_tokens=64)
    print("[chat 自测]", repr(out))
    assert "<think>" not in out
    _prepare_cpu_embedder()
    with tempfile.TemporaryDirectory(prefix="episodic_selftest_") as td:
        eng = EpisodicMethod(work_dir=Path(td), llm=llm)
        eng.ingest_conversation(
            {
                "sid": "s1",
                "week": 1,
                "messages": [
                    {"role": "user", "content": "这周参加了学校秋季招聘会，只是逛展没有投简历。"},
                    {"role": "assistant", "content": "了解。"},
                    {"role": "user", "content": "请记住我希望情绪强烈时先被倾听。"},
                ],
            }
        )
        block = eng.retrieve("我最近在忙什么？")
        print("[episodic 自测] retrieve:", repr(block)[:400])
        assert "memory" in block or "秋招" in block or "招聘" in block
    print("自测通过 ✓")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", type=Path, default=DATA_DEFAULT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--categories", type=str, default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--qa-workers", type=int, default=3)
    ap.add_argument("--out", type=Path, default=PROJECT / "evaluate/memory/runs")
    args = ap.parse_args()

    cfg = load_config()
    if args.selftest:
        from rate_limit import patch_openai_for_mem0

        patch_openai_for_mem0(cfg)
        selftest(cfg)
        return

    from rate_limit import patch_openai_for_mem0

    patch_openai_for_mem0(cfg)
    _prepare_cpu_embedder()
    llm = APILLM(cfg)

    cats = [c.strip() for c in args.categories.split(",") if c.strip()] or None
    cases = _load_cases(args.data, limit=args.limit, categories=cats)
    if not cases:
        sys.exit("no cases loaded")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / f"qa_episodic_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir = run_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"

    with _PRINT_LOCK:
        print(
            f"[run] cases={len(cases)} workers={args.workers} qa_workers={args.qa_workers} "
            f"data={args.data} out={run_dir}"
        )

    results: List[Dict[str, Any]] = []

    def _one(case: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        row = run_case(case, llm=llm, work_dir=work_dir, qa_workers=args.qa_workers)
        row["elapsed_sec"] = round(time.time() - t0, 2)
        with _WRITE_LOCK:
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        sc = ((row.get("methods") or {}).get("episodic") or {}).get("score") or {}
        with _PRINT_LOCK:
            print(
                f"  [{row.get('case_id')}] qa={sc.get('qa_accuracy')} "
                f"f1={sc.get('content_f1')} err={((row.get('methods') or {}).get('episodic') or {}).get('error')}"
            )
        return row

    if args.workers <= 1:
        for case in cases:
            results.append(_one(case))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_one, c) for c in cases]
            for fut in as_completed(futs):
                results.append(fut.result())

    # stable order
    order = {str(c["case_id"]): i for i, c in enumerate(cases)}
    results.sort(key=lambda r: order.get(str(r.get("case_id")), 10**9))

    summary = {
        "created_at": stamp,
        "method": "episodic",
        "n_cases": len(results),
        "data": str(args.data),
        "run_dir": str(run_dir),
        "workers": args.workers,
        "qa_workers": args.qa_workers,
        "chat_model": cfg.get("chat_model"),
        **summarize(results),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
