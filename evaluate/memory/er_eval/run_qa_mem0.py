#!/usr/bin/env python3
"""Long-horizon QA eval for official Mem0 on the same protocol as er_eval.

Uses the same reader / judge / content_f1 scoring as run_qa_er.py so
results are comparable on evaluate/memory/data (the final eval set).

  python run_qa_mem0.py --selftest
  python run_qa_mem0.py --data ../data/all_50.jsonl --limit 1 --workers 1
  python run_qa_mem0.py --data ../data/all_50.jsonl --workers 4 --qa-workers 3
"""
from __future__ import annotations

import argparse
import json
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
from mem0_adapter import Mem0Adapter, _ensure_shared_sentence_transformer  # noqa: E402
from run_qa_er import (  # noqa: E402
    _load_cases,
    content_f1,
    load_config,
    reader_answer_judge,
    reader_answer_mcq,
    score_questions,
)

DATA_DEFAULT = PROJECT / "evaluate/memory/data/all_50.jsonl"
EMBED_DEFAULT = PROJECT / "models/encoders/bge-m3"

_PRINT_LOCK = threading.Lock()
_WRITE_LOCK = threading.Lock()


def _answer_one(llm: Any, engine: Mem0Adapter, q: Dict[str, Any]) -> Dict[str, Any]:
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


def run_case(
    case: Dict[str, Any],
    *,
    llm: Any,
    work_dir: Path,
    api_cfg: Dict[str, Any],
    embed_model_path: str,
    qa_workers: int,
) -> Dict[str, Any]:
    case_id = str(case["case_id"])
    case_dir = work_dir / case_id
    if case_dir.exists():
        shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    username = str((case.get("user") or {}).get("username") or case_id)
    out: Dict[str, Any] = {"case_id": case_id, "category": case.get("category"), "methods": {}}
    try:
        engine = Mem0Adapter(
            work_dir=case_dir / "mem0",
            api_cfg=api_cfg,
            embed_model_path=embed_model_path,
            user_id=username,
        )
        for conv in case.get("conversations") or []:
            engine.ingest_messages(list(conv.get("messages") or []))

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
        out["methods"]["mem0"] = {
            "method": "mem0",
            "score": score,
            "answers": answers,
            "content_f1_detail": f1,
            "store": engine.store_snapshot(),
        }
    except Exception as exc:
        out["methods"]["mem0"] = {"method": "mem0", "error": f"{type(exc).__name__}: {exc}"}
    return out


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    method = "mem0"
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
        }
    }


def selftest(cfg: dict) -> None:
    llm = APILLM(cfg)
    out = llm.generate_structured([{"role": "user", "content": "只回复两个字：OK"}], max_new_tokens=64)
    print("[chat 自测]", repr(out))
    assert "<think>" not in out
    embed = str(EMBED_DEFAULT)
    with tempfile.TemporaryDirectory(prefix="mem0_selftest_") as td:
        adapter = Mem0Adapter(
            work_dir=Path(td),
            api_cfg=cfg,
            embed_model_path=embed,
            user_id="selftest_user",
            top_k=5,
        )
        adapter.ingest_messages(
            [
                {"role": "user", "content": "我叫小明，最近在准备英语口语考试。"},
                {"role": "assistant", "content": "备考加油。"},
            ]
        )
        block = adapter.retrieve("用户在准备什么？")
        facts = adapter.list_facts()
        print("[mem0 自测] retrieve:", repr(block)[:300])
        print("[mem0 自测] facts:", facts)
        assert facts, "Mem0 未写入任何事实"
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
    ap.add_argument("--embed-model", type=str, default=str(EMBED_DEFAULT))
    args = ap.parse_args()

    cfg = load_config()
    from rate_limit import patch_openai_for_mem0

    patch_openai_for_mem0(cfg)
    print(
        f"[patch] OpenAI/Mem0 rate limit "
        f"max_concurrent_api={cfg.get('max_concurrent_api', 3)} "
        f"max_retries={cfg.get('max_retries', 10)}"
    )

    if args.selftest:
        selftest(cfg)
        return

    _ensure_shared_sentence_transformer()
    llm = APILLM(cfg)

    cats = [c.strip() for c in args.categories.split(",") if c.strip()] or None
    cases = _load_cases(args.data, limit=args.limit, categories=cats)
    if not cases:
        sys.exit("no cases loaded")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / f"qa_mem0_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir = run_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"

    with _PRINT_LOCK:
        print(
            f"[run] method=mem0 cases={len(cases)} workers={args.workers} "
            f"qa_workers={args.qa_workers} data={args.data} out={run_dir}"
        )

    results: List[Dict[str, Any]] = []

    def _one(case: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        row = run_case(
            case,
            llm=llm,
            work_dir=work_dir,
            api_cfg=cfg,
            embed_model_path=str(args.embed_model),
            qa_workers=args.qa_workers,
        )
        row["elapsed_sec"] = round(time.time() - t0, 2)
        with _WRITE_LOCK:
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        sc = ((row.get("methods") or {}).get("mem0") or {}).get("score") or {}
        with _PRINT_LOCK:
            print(
                f"  [{row.get('case_id')}] qa={sc.get('qa_accuracy')} "
                f"f1={sc.get('content_f1')} err={((row.get('methods') or {}).get('mem0') or {}).get('error')} "
                f"t={row.get('elapsed_sec')}s"
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

    order = {str(c["case_id"]): i for i, c in enumerate(cases)}
    results.sort(key=lambda r: order.get(str(r.get("case_id")), 10**9))

    epi_ref = None
    runs_root = PROJECT / "evaluate/memory/runs"
    epi_runs = sorted(
        list(runs_root.glob("qa_er_*/summary.json"))
        + list(runs_root.glob("qa_aer_*/summary.json"))
        + list(runs_root.glob("qa_episodic_*/summary.json")),
        reverse=True,
    )
    for p in epi_runs:
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
            data_s = str(s.get("data") or "").replace("\\", "/")
            if "/memory/data" in data_s or "data_episodic" in data_s:
                epi_ref = {
                    "qa_accuracy": (s.get("methods") or {}).get("trace", {}).get("qa_accuracy"),
                    "run_dir": s.get("run_dir"),
                    "note": "Same-protocol trace run on evaluate/memory/data.",
                }
                break
        except Exception:
            continue

    summary = {
        "created_at": stamp,
        "method": "mem0",
        "n_cases": len(results),
        "data": str(args.data),
        "run_dir": str(run_dir),
        "workers": args.workers,
        "qa_workers": args.qa_workers,
        "chat_model": cfg.get("chat_model"),
        **summarize(results),
        "reference_episodic_data": epi_ref,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
