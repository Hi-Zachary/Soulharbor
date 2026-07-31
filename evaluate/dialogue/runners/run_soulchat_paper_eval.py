# [ARCHIVED — 非运行时依赖]
# 原路径: evaluate/run_soulchat_paper_eval.py
# 原先用途: 按论文设定跑 SoulChat 相关评测流水线的 Python 入口。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from soulchat_model import load_model_and_tokenizer
from soulchat_paper_metrics import compute_4b3r

def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _batched(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


@dataclass(frozen=True)
class EvalConfig:
    data: str
    base: str
    adapter: str
    out_dir: str
    limit: int
    batch_size: int
    max_input_len: int
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    repetition_penalty: float
    eos_token_id: int
    device: str
    attn: str
    load_in_4bit: bool
    enable_thinking: bool


def _is_oom_error(e: BaseException) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _load_preds_refs(pred_path: Path) -> tuple[list[str], list[str]]:
    preds: list[str] = []
    refs: list[str] = []
    if not pred_path.exists():
        return preds, refs
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            preds.append(str(obj.get("prediction") or "").strip())
            refs.append(str(obj.get("reference") or "").strip())
    return preds, refs


def _instruction_to_messages(instruction: str) -> list[dict[str, str]]:
    """
    Convert our SoulChat-style plain-text prompt into structured messages for Qwen3 chat template.

    Expected lines begin with:
      - 用户：
      - 心理咨询师：

    The prompt usually ends with a dangling "心理咨询师：" as generation cue; we drop empty final assistant.
    """
    msgs: list[dict[str, str]] = []
    cur_role: str | None = None
    cur_lines: list[str] = []

    def _flush() -> None:
        nonlocal cur_role, cur_lines
        if not cur_role:
            cur_lines = []
            return
        content = "\n".join([x for x in cur_lines if x.strip()]).strip()
        if content:
            msgs.append({"role": cur_role, "content": content})
        cur_role = None
        cur_lines = []

    for raw in (instruction or "").splitlines():
        line = raw.rstrip("\n")
        if line.startswith("用户："):
            _flush()
            cur_role = "user"
            cur_lines = [line[len("用户：") :].lstrip()]
            continue
        if line.startswith("心理咨询师："):
            _flush()
            cur_role = "assistant"
            cur_lines = [line[len("心理咨询师：") :].lstrip()]
            continue
        # Continuation line
        if cur_role:
            cur_lines.append(line)

    _flush()

    # Drop an empty final assistant message (generation cue).
    if msgs and msgs[-1]["role"] == "assistant" and not (msgs[-1].get("content") or "").strip():
        msgs.pop()
    return msgs


def _eval_one_dataset(
    *,
    rows: List[Dict[str, Any]],
    model,
    tokenizer,
    out_dir: Path,
    cfg: EvalConfig,
    resume: bool,
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    start_t = time.time()
    pred_path = out_dir / "predictions.jsonl"
    total = len(rows)

    already = 0
    if resume and pred_path.exists():
        with pred_path.open("r", encoding="utf-8") as f:
            for _ in f:
                already += 1
        already = min(already, total)
    else:
        # Fresh run: ensure we start from empty.
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text("", encoding="utf-8")

    # Actually generate missing part, starting from `already`.
    start_index = already
    remaining_rows = rows[start_index:]
    if remaining_rows:
        with pred_path.open("a", encoding="utf-8") as fo:
            done = already
            for bi, batch in enumerate(_batched(remaining_rows, int(cfg.batch_size)), start=1):
                instructions_raw = [str(x.get("instruction") or "") for x in batch]
                references = [str(x.get("output") or "") for x in batch]

                prompts: list[str] = []
                for ins in instructions_raw:
                    messages = _instruction_to_messages(ins)
                    # Qwen3 official knob: enable_thinking controls whether <think> block is used.
                    # When enable_thinking=False, template inserts an empty <think></think> and the model answers after it.
                    prompt = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                        enable_thinking=bool(cfg.enable_thinking),
                    )
                    prompts.append(prompt)

                tok = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(cfg.max_input_len),
                )
                tok = {k: v.to(model.device) for k, v in tok.items()}

                with torch.no_grad():
                    out_ids = model.generate(
                        **tok,
                        max_new_tokens=int(cfg.max_new_tokens),
                        do_sample=bool(cfg.do_sample),
                        temperature=float(cfg.temperature),
                        top_p=float(cfg.top_p),
                        repetition_penalty=float(cfg.repetition_penalty),
                        eos_token_id=int(cfg.eos_token_id),
                        pad_token_id=int(tokenizer.pad_token_id),
                    )

                input_len = tok["input_ids"].shape[1]
                gen_ids = out_ids[:, input_len:]
                texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

                for ins, ref, pred in zip(instructions_raw, references, texts):
                    pred = (pred or "").strip()
                    # Strip accidental role prefix.
                    for prefix in ("心理咨询师：", "医生："):
                        if pred.startswith(prefix):
                            pred = pred[len(prefix) :].strip()
                            break
                    fo.write(
                        json.dumps({"instruction": ins, "reference": ref, "prediction": pred}, ensure_ascii=False) + "\n"
                    )
                    done += 1

                fo.flush()
                if bi <= 3 or bi % 10 == 0:
                    elapsed = max(1e-6, time.time() - start_t)
                    rate = done / elapsed
                    eta_s = int((total - done) / max(1e-6, rate))
                    print(
                        f"[PROGRESS] {out_dir.name}: {done}/{total} " f"rate={rate:.2f} ex/s eta={eta_s}s",
                        flush=True,
                    )

    preds, refs = _load_preds_refs(pred_path)
    if len(preds) != total or len(refs) != total:
        raise RuntimeError(f"Incomplete predictions: got={len(preds)} expected={total} file={pred_path}")

    metrics = compute_4b3r(preds, refs)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="SoulChat paper automatic evaluation (BLEU1-4 + ROUGE1/2/L) on seen/unseen.")
    ap.add_argument("--seen-data", default="evaluate/dialogue/data/soulchat_seen_1k.jsonl")
    ap.add_argument("--unseen-data", default="evaluate/dialogue/data/soulchat_unseen_1k.jsonl")
    ap.add_argument("--mode", default="both", choices=["both", "seen", "unseen"], help="Which split(s) to evaluate.")
    ap.add_argument("--resume", action="store_true", help="Resume from existing predictions/metrics if present.")
    ap.add_argument(
        "--which",
        default="all",
        choices=["all", "dpo", "base", "self_cognition"],
        help="Evaluate which adapter(s). When used with --resume, can quickly fill missing runs.",
    )
    ap.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable <think> reasoning output (default: disabled to match typical SFT/chat deployments).",
    )
    ap.add_argument(
        "--skip-self-cognition",
        action="store_true",
        help="Skip evaluating base+self_cognition adapter (useful if it OOMs; dpo/base still run).",
    )
    ap.add_argument("--base", default="models/Qwen3-14B")
    ap.add_argument("--adapter-self", default="saves/qwen14b/lora/sft_self_cognition_20260422_112858")
    ap.add_argument("--adapter-dpo", default="saves/qwen14b/lora/dpo_synth_20260425_060332")
    ap.add_argument("--out-dir", default="", help="Output dir (default: evaluate_runs/soulchat_paper_eval_<ts>)")
    ap.add_argument("--limit", type=int, default=0, help="Per-dataset limit (0=all).")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-input-len", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--eos-token-id", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda:0", "cuda:1"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    seen_path = (root / args.seen_data).resolve() if not Path(args.seen_data).is_absolute() else Path(args.seen_data)
    unseen_path = (root / args.unseen_data).resolve() if not Path(args.unseen_data).is_absolute() else Path(args.unseen_data)
    if not seen_path.exists():
        raise SystemExit(f"Missing: {seen_path}")
    if not unseen_path.exists():
        raise SystemExit(f"Missing: {unseen_path}")

    seen_rows = list(_iter_jsonl(seen_path))
    unseen_rows = list(_iter_jsonl(unseen_path))
    if int(args.limit) > 0:
        seen_rows = seen_rows[: int(args.limit)]
        unseen_rows = unseen_rows[: int(args.limit)]
    if not seen_rows or not unseen_rows:
        raise SystemExit("Empty evaluation set(s).")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (root / "evaluate_runs" / f"soulchat_paper_eval_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = str((root / args.base).resolve()) if not Path(args.base).is_absolute() else str(args.base)

    def _adapter_path(p: str) -> str:
        if not p:
            return ""
        return str((root / p).resolve()) if not Path(p).is_absolute() else str(p)

    # Evaluation order requested:
    #   base+dpo -> base -> base+self-cognition
    adapters_all = [("dpo", _adapter_path(str(args.adapter_dpo))), ("base", "")]
    if not bool(args.skip_self_cognition):
        adapters_all.append(("self_cognition", _adapter_path(str(args.adapter_self))))

    if str(args.which) == "all":
        adapters = adapters_all
    else:
        adapters = [x for x in adapters_all if x[0] == str(args.which)]
        if not adapters:
            raise SystemExit(f"Nothing to run for --which={args.which}")

    results: Dict[str, Any] = {"out_dir": str(out_dir), "runs": {}}

    for name, adapter in adapters:
        # If resuming and all required metrics exist for this adapter+mode, skip loading the model entirely.
        if bool(args.resume):
            need: list[Path] = []
            if str(args.mode) in ("both", "seen"):
                need.append(out_dir / f"{name}_seen" / "metrics.json")
            if str(args.mode) in ("both", "unseen"):
                need.append(out_dir / f"{name}_unseen" / "metrics.json")
            if need and all(p.exists() for p in need):
                print(f"[SKIP] {name} all metrics exist for mode={args.mode}", flush=True)
                results["runs"][name] = {}
                if str(args.mode) in ("both", "seen"):
                    results["runs"][name]["seen"] = json.loads((out_dir / f"{name}_seen" / "metrics.json").read_text(encoding="utf-8"))
                if str(args.mode) in ("both", "unseen"):
                    results["runs"][name]["unseen"] = json.loads((out_dir / f"{name}_unseen" / "metrics.json").read_text(encoding="utf-8"))
                continue

        print(f"[CONFIG] mode={args.mode} device={args.device} batch_size={args.batch_size} "
              f"max_input_len={args.max_input_len} max_new_tokens={args.max_new_tokens} "
              f"do_sample={bool(args.do_sample)} top_p={args.top_p} temperature={args.temperature} "
              f"load_in_4bit={bool(args.load_in_4bit)} enable_thinking={bool(args.enable_thinking)}",
              flush=True)
        print(f"[LOAD] model={name} adapter={adapter or '<none>'}", flush=True)
        try:
            model, tokenizer = load_model_and_tokenizer(
                base=base,
                adapter=adapter,
                device=str(args.device),
                attn=str(args.attn),
                load_in_4bit=bool(args.load_in_4bit),
            )
        except RuntimeError as e:
            if _is_oom_error(e):
                print("[OOM] during model load", flush=True)
                return 99
            raise
        eos_token_id = int(args.eos_token_id) if int(args.eos_token_id) > 0 else int(tokenizer.eos_token_id or 0)

        common = dict(
            base=str(args.base),
            adapter=adapter,
            limit=int(args.limit),
            batch_size=int(args.batch_size),
            max_input_len=int(args.max_input_len),
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            eos_token_id=int(eos_token_id),
            device=str(args.device),
            attn=str(args.attn),
            load_in_4bit=bool(args.load_in_4bit),
            enable_thinking=bool(args.enable_thinking),
        )

        results["runs"][name] = {}
        if str(args.mode) in ("both", "seen"):
            print(f"[RUN] {name} seen={len(seen_rows)}", flush=True)
            out_sub = out_dir / f"{name}_seen"
            if bool(args.resume) and (out_sub / "metrics.json").exists():
                results["runs"][name]["seen"] = json.loads((out_sub / "metrics.json").read_text(encoding="utf-8"))
                print(f"[SKIP] {out_sub.name} metrics.json exists", flush=True)
            else:
                try:
                    m_seen = _eval_one_dataset(
                        rows=seen_rows,
                        model=model,
                        tokenizer=tokenizer,
                        out_dir=out_sub,
                        cfg=EvalConfig(data=str(seen_path), out_dir=str(out_sub), **common),
                        resume=bool(args.resume),
                    )
                except RuntimeError as e:
                    if _is_oom_error(e):
                        print("[OOM] during generation", flush=True)
                        return 99
                    raise
                results["runs"][name]["seen"] = m_seen
        if str(args.mode) in ("both", "unseen"):
            print(f"[RUN] {name} unseen={len(unseen_rows)}", flush=True)
            out_sub = out_dir / f"{name}_unseen"
            if bool(args.resume) and (out_sub / "metrics.json").exists():
                results["runs"][name]["unseen"] = json.loads((out_sub / "metrics.json").read_text(encoding="utf-8"))
                print(f"[SKIP] {out_sub.name} metrics.json exists", flush=True)
            else:
                try:
                    m_unseen = _eval_one_dataset(
                        rows=unseen_rows,
                        model=model,
                        tokenizer=tokenizer,
                        out_dir=out_sub,
                        cfg=EvalConfig(data=str(unseen_path), out_dir=str(out_sub), **common),
                        resume=bool(args.resume),
                    )
                except RuntimeError as e:
                    if _is_oom_error(e):
                        print("[OOM] during generation", flush=True)
                        return 99
                    raise
                results["runs"][name]["unseen"] = m_unseen

        del model
        torch.cuda.empty_cache()

    suffix = str(args.mode)
    metrics_path = out_dir / (f"all_metrics_{suffix}.json" if suffix != "both" else "all_metrics.json")
    metrics_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] wrote:", str(metrics_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
