# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/train_classifier_scheduler.py
# 原先用途: 分类器训练调度/多任务编排辅助。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


OOM_PATTERNS = [
    "CUDA out of memory",
    "CUDNN_STATUS_ALLOC_FAILED",
    "RuntimeError: CUDA error: out of memory",
    "std::bad_alloc",
]


def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def is_oom(log_tail: str) -> bool:
    lower = log_tail.lower()
    return any(p.lower() in lower for p in OOM_PATTERNS)


def tail_file(path: Path, max_bytes: int = 32_000) -> str:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return ""
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def build_train_cmd(
    *,
    task: str,
    nproc: int,
    train_script: str,
    model_name_or_path: str,
    output_dir: str,
    max_length: int,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    epochs: float,
    lr: float,
    seed: int,
    fp16: bool,
    num_workers: int,
) -> List[str]:
    base_args = [
        train_script,
        "--task",
        task,
        "--model-name-or-path",
        model_name_or_path,
        "--output-dir",
        output_dir,
        "--max-length",
        str(max_length),
        "--train-batch-size",
        str(train_batch_size),
        "--eval-batch-size",
        str(eval_batch_size),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--num-workers",
        str(num_workers),
    ]
    if fp16:
        base_args.append("--fp16")

    if nproc <= 1:
        return [sys.executable, *base_args]

    # torchrun alternative that works as long as torch is installed in the env
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        *base_args,
    ]


def run_one(
    *,
    cmd: List[str],
    env: Dict[str, str],
    log_path: Path,
    cwd: Path,
) -> Tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        returncode = proc.wait()
    return returncode, tail_file(log_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequentially train SoulHarbor classifiers; auto-retry with smaller batch size on OOM."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="intent,risk_emotion_joint",
        help="Comma-separated tasks to run in order (intent,risk,emotion,risk_emotion,risk_emotion_joint).",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=2,
        help="Number of processes (GPUs) for DDP. Use 1 for single-GPU.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="0,1",
        help='CUDA_VISIBLE_DEVICES value. Example: "0,1" or "1".',
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="models/encoders/chinese-macbert-large",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--base-output-dir",
        type=str,
        default="outputs/classifiers_runs",
        help="All runs will be stored under base_output_dir/<timestamp>/...",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per task when OOM is detected.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    train_script = str(repo_root / "scripts" / "train_classifiers_hf.py")

    run_root = Path(args.base_output_dir) / now_tag()
    logs_dir = run_root / "logs"
    outputs_dir = run_root / "outputs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise SystemExit("No tasks specified.")

    print(f"[scheduler] run_root={run_root}")
    print(f"[scheduler] tasks={tasks} nproc={args.nproc} CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")

    for task in tasks:
        train_bs = args.train_batch_size
        eval_bs = args.eval_batch_size

        for attempt in range(args.max_retries + 1):
            suffix = "" if attempt == 0 else f"_retry{attempt}"
            task_out = str(outputs_dir)  # train script will create subfolders per task
            log_path = logs_dir / f"{task}{suffix}.log"

            cmd = build_train_cmd(
                task=task,
                nproc=args.nproc,
                train_script=train_script,
                model_name_or_path=args.model_name_or_path,
                output_dir=task_out,
                max_length=args.max_length,
                train_batch_size=train_bs,
                eval_batch_size=eval_bs,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                epochs=args.epochs,
                lr=args.lr,
                seed=args.seed,
                fp16=bool(args.fp16),
                num_workers=args.num_workers,
            )

            print(f"[scheduler] start task={task} attempt={attempt} train_bs={train_bs} eval_bs={eval_bs}")
            code, tail = run_one(cmd=cmd, env=env, log_path=log_path, cwd=repo_root)
            if code == 0:
                print(f"[scheduler] done task={task} log={log_path}")
                break

            oom = is_oom(tail)
            print(f"[scheduler] failed task={task} code={code} oom={oom} log={log_path}")
            if not oom or attempt >= args.max_retries:
                raise SystemExit(f"Task {task} failed. See log: {log_path}")

            # Backoff strategy on OOM: reduce micro-batch sizes.
            if train_bs > 1:
                train_bs = max(1, train_bs // 2)
            else:
                eval_bs = max(1, eval_bs // 2)

    print(f"[scheduler] all done. outputs={outputs_dir} logs={logs_dir}")


if __name__ == "__main__":
    main()
