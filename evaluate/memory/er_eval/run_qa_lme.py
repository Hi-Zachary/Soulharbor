#!/usr/bin/env python3
"""LongMemEval-style QA eval for SoulHarbor ER trace memory.

Each JSONL record is one independent question with its full haystack history.
Use a fresh database per question_id.

  python run_qa_lme.py --selftest
  python run_qa_lme.py --limit 1 --workers 1
  python run_qa_lme.py --workers 2
  python run_qa_lme.py --oracle   # evidence-only histories
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
BUNDLE = PROJECT / "evaluate/memory/data/soulharbor_mh_longmemeval_30_bundle"
DATA_DEFAULT = BUNDLE / "soulharbor_mh_longmemeval_30.jsonl"
ORACLE_DEFAULT = BUNDLE / "soulharbor_mh_longmemeval_30_oracle.json"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT))

from api_llm import APILLM  # noqa: E402
from lme_dataset import LMETraceEngine, load_lme_instances  # noqa: E402
from lme_scoring import (  # noqa: E402
    reader_answer_open,
    score_qa,
    score_retrieval,
    summarize_lme_results,
)
from run_qa_er import _prepare_embedder, load_config  # noqa: E402

_PRINT_LOCK = threading.Lock()
_WRITE_LOCK = threading.Lock()


def _load_oracle_instances(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else [raw]


def run_instance(
    item: Dict[str, Any],
    *,
    llm: Any,
    work_dir: Path,
) -> Dict[str, Any]:
    qid = str(item["question_id"])
    case_dir = work_dir / qid
    if case_dir.exists():
        shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Any] = {
        "question_id": qid,
        "capability": item.get("capability"),
        "memory_target": item.get("memory_target"),
        "category": item.get("category"),
        "answerable": item.get("answerable", True),
        "evaluation": item.get("evaluation"),
    }
    t0 = time.time()
    try:
        engine = LMETraceEngine(work_dir=case_dir / "trace", llm=llm)
        engine.ingest_instance(item)

        question = str(item.get("question") or "")
        memory_block, retrieval_details = engine.retrieve_with_details(question)
        hypothesis = reader_answer_open(llm, memory_block=memory_block, question=question)
        qa = score_qa(llm, item=item, hypothesis=hypothesis)
        retrieval = score_retrieval(
            item,
            retrieved_source_message_ids=list(
                retrieval_details.get("retrieved_source_message_ids") or []
            ),
            retrieved_anchor_source_message_ids=list(
                retrieval_details.get("retrieved_anchor_source_message_ids") or []
            ),
            retrieved_anchor_turn_ids=list(
                retrieval_details.get("retrieved_anchor_turn_ids") or []
            ),
            retrieved_session_ids=list(retrieval_details.get("retrieved_session_ids") or []),
        )

        out.update(
            {
                "hypothesis": hypothesis,
                "qa": qa,
                "retrieval": retrieval,
                "retrieved_source_message_ids": retrieval_details.get(
                    "retrieved_source_message_ids"
                ),
                "retrieved_session_ids": retrieval_details.get("retrieved_session_ids"),
                "retrieval_trace": retrieval_details.get("retrieval_trace"),
                "active_profiles": engine.list_active_profiles(),
                "store": engine.store_snapshot(),
                "latency_ms": int((time.time() - t0) * 1000),
            }
        )
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["elapsed_sec"] = round(time.time() - t0, 2)
    return out


def selftest(cfg: dict) -> None:
    llm = APILLM(cfg)
    out = llm.generate_structured([{"role": "user", "content": "只回复两个字：OK"}], max_new_tokens=64)
    print("[chat 自测]", repr(out))
    _prepare_embedder()
    items = load_lme_instances(DATA_DEFAULT, limit=1)
    with tempfile.TemporaryDirectory(prefix="lme_selftest_") as td:
        row = run_instance(items[0], llm=llm, work_dir=Path(td))
        assert row.get("hypothesis"), "missing hypothesis"
        assert "qa" in row, "missing qa"
        assert "retrieval" in row, "missing retrieval"
        print("[lme 自测]", row["question_id"], "qa=", row["qa"].get("correct"))
    print("自测通过 ✓")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", type=Path, default=DATA_DEFAULT)
    ap.add_argument("--oracle", action="store_true", help="use evidence-only oracle histories")
    ap.add_argument("--oracle-path", type=Path, default=ORACLE_DEFAULT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--categories", type=str, default="")
    ap.add_argument("--capabilities", type=str, default="")
    ap.add_argument("--workers", type=int, default=2)
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
    _prepare_embedder()
    llm = APILLM(cfg)

    cats = [c.strip() for c in args.categories.split(",") if c.strip()] or None
    caps = [c.strip() for c in args.capabilities.split(",") if c.strip()] or None
    if args.oracle:
        instances = _load_oracle_instances(args.oracle_path)
        if cats:
            instances = [r for r in instances if r.get("category") in set(cats)]
        if caps:
            instances = [r for r in instances if r.get("capability") in set(caps)]
        if args.limit > 0:
            instances = instances[: args.limit]
    else:
        instances = load_lme_instances(
            args.data, limit=args.limit, categories=cats, capabilities=caps
        )
    if not instances:
        sys.exit("no instances loaded")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "qa_lme_oracle" if args.oracle else "qa_lme"
    run_dir = Path(args.out) / f"{tag}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir = run_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    predictions_path = run_dir / "predictions.jsonl"

    with _PRINT_LOCK:
        print(
            f"[run] instances={len(instances)} workers={args.workers} "
            f"oracle={args.oracle} data={args.data} out={run_dir}"
        )

    results: List[Dict[str, Any]] = []

    def _one(item: Dict[str, Any]) -> Dict[str, Any]:
        row = run_instance(item, llm=llm, work_dir=work_dir)
        pred = {"question_id": row.get("question_id"), "hypothesis": row.get("hypothesis")}
        with _WRITE_LOCK:
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            with predictions_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        with _PRINT_LOCK:
            qa = row.get("qa") or {}
            ret = row.get("retrieval") or {}
            print(
                f"  [{row.get('question_id')}] qa={qa.get('correct')} "
                f"msg_recall={ret.get('message_recall')} err={row.get('error')}"
            )
        return row

    if args.workers <= 1:
        for item in instances:
            results.append(_one(item))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_one, item): item for item in instances}
            for fut in as_completed(futs):
                item = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    qid = str(item.get("question_id"))
                    row = {"question_id": qid, "error": f"{type(exc).__name__}: {exc}"}
                    with _WRITE_LOCK:
                        with results_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    with _PRINT_LOCK:
                        print(f"  [{qid}] FATAL {type(exc).__name__}: {exc}")
                    results.append(row)

    order = {str(i["question_id"]): idx for idx, i in enumerate(instances)}
    results.sort(key=lambda r: order.get(str(r.get("question_id")), 10**9))

    summary = {
        "created_at": stamp,
        "method": "trace",
        "benchmark": "soulharbor_mh_longmemeval_30",
        "oracle": bool(args.oracle),
        "n_instances": len(results),
        "data": str(args.oracle_path if args.oracle else args.data),
        "run_dir": str(run_dir),
        "workers": args.workers,
        "chat_model": cfg.get("chat_model"),
        **summarize_lme_results(results),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
