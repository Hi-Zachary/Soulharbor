from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from run_soulchat_paper_eval import _instruction_to_messages  # reuse parser
from soulchat_model import load_model_and_tokenizer
from soulchat_paper_metrics import compute_4b3r


def _is_oom_error(e: BaseException) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


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
    warmup_batch_size: int
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


def _eval_dataset(
    *,
    root: Path,
    rows: List[Dict[str, Any]],
    base: str,
    adapter: str,
    out_dir: Path,
    cfg: EvalConfig,
    resume: bool,
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    pred_path = out_dir / "predictions.jsonl"
    preds: List[str] = []
    refs: List[str] = []
    start_index = 0
    if resume and pred_path.exists():
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
        start_index = min(len(preds), len(rows))
        preds = preds[:start_index]
        refs = refs[:start_index]

    total = len(rows)
    if start_index >= total:
        metrics = compute_4b3r(preds, refs)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics

    model, tokenizer = load_model_and_tokenizer(
        base=base,
        adapter=adapter,
        device=str(cfg.device),
        attn=str(cfg.attn),
        load_in_4bit=bool(cfg.load_in_4bit),
    )
    eos_token_id = int(cfg.eos_token_id)
    if eos_token_id <= 0:
        eos_token_id = int(getattr(tokenizer, "eos_token_id", 0) or 0)

    mode = "a" if resume and pred_path.exists() else "w"
    with pred_path.open(mode, encoding="utf-8") as fo:
        start_t = time.time()
        done = start_index
        warm_bs = max(0, int(cfg.warmup_batch_size))
        target_bs = max(1, int(cfg.batch_size))
        cur_bs = target_bs

        if start_index == 0 and warm_bs > 0:
            print(
                f"[WARMUP] {out_dir.name}: first_batch_size={warm_bs} then batch_size={target_bs}",
                flush=True,
            )

        i = start_index
        bi = 0
        while i < total:
            bi += 1
            desired = warm_bs if (i == 0 and warm_bs > 0) else cur_bs
            bs = min(max(1, desired), total - i)
            batch = rows[i : i + bs]
            ins_raw = [str(x.get("instruction") or "") for x in batch]
            references = [str(x.get("output") or "") for x in batch]

            prompts: List[str] = []
            for ins in ins_raw:
                messages = _instruction_to_messages(ins)
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

            if bi <= 2:
                mx = int(tok["input_ids"].shape[1])
                print(f"[BATCH] {out_dir.name}: bi={bi} bs={len(batch)} input_len={mx}", flush=True)

            try:
                with torch.no_grad():
                    out_ids = model.generate(
                        **tok,
                        max_new_tokens=int(cfg.max_new_tokens),
                        do_sample=bool(cfg.do_sample),
                        temperature=float(cfg.temperature),
                        top_p=float(cfg.top_p),
                        repetition_penalty=float(cfg.repetition_penalty),
                        eos_token_id=int(eos_token_id),
                        pad_token_id=int(tokenizer.pad_token_id),
                    )
            except RuntimeError as e:
                if _is_oom_error(e):
                    print("[OOM] during generation", flush=True)
                    torch.cuda.empty_cache()
                    if bs > 1:
                        new_bs = max(1, (bs + 1) // 2)
                        if i == 0 and warm_bs > 0:
                            warm_bs = new_bs
                            print(f"[OOM] reduce warmup_batch_size: {bs} -> {new_bs}", flush=True)
                        else:
                            cur_bs = new_bs
                            print(f"[OOM] reduce batch_size: {bs} -> {new_bs}", flush=True)
                        continue
                    raise
                raise

            input_len = tok["input_ids"].shape[1]
            gen_ids = out_ids[:, input_len:]
            texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            for ins, ref, pred in zip(ins_raw, references, texts):
                pred = (pred or "").strip()
                for prefix in ("心理咨询师：", "医生："):
                    if pred.startswith(prefix):
                        pred = pred[len(prefix) :].strip()
                        break
                preds.append(pred)
                refs.append(ref)
                fo.write(json.dumps({"instruction": ins, "reference": ref, "prediction": pred}, ensure_ascii=False) + "\n")
                done += 1

            fo.flush()
            if bi <= 2 or bi % 5 == 0:
                elapsed = max(1e-6, time.time() - start_t)
                rate = done / elapsed
                eta_s = int((total - done) / max(1e-6, rate))
                print(f"[PROGRESS] {out_dir.name}: {done}/{total} rate={rate:.2f} ex/s eta={eta_s}s", flush=True)
            i += bs

    metrics = compute_4b3r(preds, refs)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    del model
    torch.cuda.empty_cache()
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Run SoulChat paper metrics on a single JSONL dataset for 3 models.")
    ap.add_argument("--data", required=True, help="JSONL with {instruction, output}.")
    ap.add_argument("--base", default="models/Qwen3-14B")
    ap.add_argument("--adapter-self", default="saves/qwen14b/lora/sft_self_cognition_20260422_112858")
    ap.add_argument("--adapter-dpo", default="saves/qwen14b/lora/dpo_synth_20260425_060332")
    ap.add_argument("--out-dir", default="", help="Output dir (default: evaluate_runs/smile_eval_<ts>)")
    ap.add_argument(
        "--which",
        default="all",
        choices=["all", "dpo", "base", "self_cognition"],
        help="Evaluate which model(s).",
    )
    ap.add_argument("--resume", action="store_true", help="Resume from existing predictions.jsonl if present.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--warmup-batch-size", type=int, default=4, help="Use a small first batch to get quick logs/output.")
    ap.add_argument("--max-input-len", type=int, default=1536)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-p", type=float, default=0.75)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--eos-token-id", type=int, default=0, help="<=0 means use tokenizer.eos_token_id")
    ap.add_argument("--device", default="cuda:0", choices=["cuda:0", "cuda:1", "auto"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--enable-thinking", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_path = (root / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Missing: {data_path}")

    rows = list(_iter_jsonl(data_path))
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]
    if not rows:
        raise SystemExit("Empty dataset.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (root / "evaluate_runs" / f"smile_eval_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = str((root / args.base).resolve()) if not Path(args.base).is_absolute() else str(args.base)

    def _adapter_path(p: str) -> str:
        if not p:
            return ""
        return str((root / p).resolve()) if not Path(p).is_absolute() else str(p)

    order = [
        ("dpo", _adapter_path(str(args.adapter_dpo))),
        ("base", ""),
        ("self_cognition", _adapter_path(str(args.adapter_self))),
    ]

    results: Dict[str, Any] = {"out_dir": str(out_dir), "data": str(data_path), "runs": {}}

    for name, adapter in order:
        if str(args.which) != "all" and str(args.which) != name:
            continue
        eos_token_id = int(args.eos_token_id)
        cfg = EvalConfig(
            data=str(data_path),
            base=str(args.base),
            adapter=adapter,
            out_dir=str(out_dir / name),
            limit=int(args.limit),
            batch_size=int(args.batch_size),
            warmup_batch_size=int(args.warmup_batch_size),
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
        print(f"[RUN] {name} device={args.device} n={len(rows)}", flush=True)
        try:
            m = _eval_dataset(
                root=root,
                rows=rows,
                base=base,
                adapter=adapter,
                out_dir=out_dir / name,
                cfg=cfg,
                resume=bool(args.resume),
            )
        except RuntimeError as e:
            if _is_oom_error(e):
                return 99
            raise
        results["runs"][name] = m

    (out_dir / "all_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] wrote:", str(out_dir / "all_metrics.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
