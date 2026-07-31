#!/usr/bin/env bash
# [ARCHIVED — 非运行时依赖]
# 原路径: evaluate/run_smile_after_soulchat.sh
# 原先用途: 在 SoulChat 评测之后串联 SMILE 评测的 shell 脚本。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIALOGUE_DIR="${ROOT}/evaluate/dialogue"
export PYTHONPATH="${DIALOGUE_DIR}/lib:${DIALOGUE_DIR}/sampling:${DIALOGUE_DIR}/runners${PYTHONPATH:+:$PYTHONPATH}"

SOULCHAT_RUN_DIR="${SOULCHAT_RUN_DIR:-}"
POLL_SECS="${POLL_SECS:-60}"
REQUIRE_SELF_COG="${REQUIRE_SELF_COG:-0}"
RUN_SELF_COG="${RUN_SELF_COG:-0}"

SMILE_DATA_OUT="${SMILE_DATA_OUT:-evaluate/dialogue/data/smile_1k.jsonl}"
SMILE_N="${SMILE_N:-1000}"
SMILE_SEED="${SMILE_SEED:-20260426}"
SMILE_MAX_TURNS="${SMILE_MAX_TURNS:-5}"

BASE_MODEL="${BASE_MODEL:-models/Qwen3-14B}"
ADAPTER_SELF_COG="${ADAPTER_SELF_COG:-saves/qwen14b/lora/sft_self_cognition_20260422_112858}"
ADAPTER_DPO="${ADAPTER_DPO:-saves/qwen14b/lora/dpo_synth_20260425_060332}"

MAX_INPUT_LEN="${MAX_INPUT_LEN:-1536}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
DO_SAMPLE="${DO_SAMPLE:-1}"
TOP_P="${TOP_P:-0.75}"
TEMPERATURE="${TEMPERATURE:-0.95}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

BATCH_SIZE="${BATCH_SIZE:-16}"
WARMUP_BATCH_SIZE="${WARMUP_BATCH_SIZE:-4}"
RESUME="${RESUME:-0}"

PY="/public/home/shijian/miniconda3/envs/soulhar/bin/python"
if [ ! -x "$PY" ]; then
  PY="python"
fi

find_latest_soulchat () {
  ls -1t "$ROOT/evaluate_runs" 2>/dev/null | grep -E '^soulchat_corpus_eval_' | head -n 1 || true
}

if [ -z "$SOULCHAT_RUN_DIR" ]; then
  latest="$(find_latest_soulchat)"
  if [ -z "$latest" ]; then
    echo "[ERR] No soulchat_corpus_eval_* under $ROOT/evaluate_runs"
    exit 2
  fi
  SOULCHAT_RUN_DIR="$ROOT/evaluate/dialogue/runs/$latest"
fi

echo "[WAIT] soulchat run: $SOULCHAT_RUN_DIR"

required=(
  "$SOULCHAT_RUN_DIR/dpo_seen/metrics.json"
  "$SOULCHAT_RUN_DIR/dpo_unseen/metrics.json"
  "$SOULCHAT_RUN_DIR/base_seen/metrics.json"
  "$SOULCHAT_RUN_DIR/base_unseen/metrics.json"
)
if [ "$REQUIRE_SELF_COG" = "1" ]; then
  required+=(
    "$SOULCHAT_RUN_DIR/self_cognition_seen/metrics.json"
    "$SOULCHAT_RUN_DIR/self_cognition_unseen/metrics.json"
  )
fi

while true; do
  missing=0
  for f in "${required[@]}"; do
    if [ ! -f "$f" ]; then
      missing=$((missing+1))
    fi
  done
  if [ "$missing" = "0" ]; then
    break
  fi
  echo "[WAIT] missing=$missing; sleep ${POLL_SECS}s"
  sleep "$POLL_SECS"
done

echo "[OK] soulchat eval completed; start SMILE zeroshot eval."

SMILE_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/evaluate/dialogue/runs/smile_zeroshot_eval_${SMILE_TS}"
mkdir -p "$RUN_DIR"

echo "[1/2] Prepare SMILE eval set ..." |& tee -a "$RUN_DIR/pipeline.log"
if [ -f "$ROOT/$SMILE_DATA_OUT" ]; then
  lines="$(wc -l <"$ROOT/$SMILE_DATA_OUT" | tr -d ' ')"
  if [ "$lines" -ge "$SMILE_N" ]; then
    echo "[SKIP] SMILE data exists ($lines lines): $ROOT/$SMILE_DATA_OUT" |& tee -a "$RUN_DIR/pipeline.log"
  else
    echo "[WARN] SMILE data exists but too small ($lines<$SMILE_N); re-sampling: $ROOT/$SMILE_DATA_OUT" |& tee -a "$RUN_DIR/pipeline.log"
    "$PY" -u "$DIALOGUE_DIR/sampling/sample_smile_eval_set.py" \
      --n "$SMILE_N" \
      --seed "$SMILE_SEED" \
      --max-turns "$SMILE_MAX_TURNS" \
      --out "$SMILE_DATA_OUT" |& tee -a "$RUN_DIR/pipeline.log"
  fi
else
  "$PY" -u "$DIALOGUE_DIR/sampling/sample_smile_eval_set.py" \
    --n "$SMILE_N" \
    --seed "$SMILE_SEED" \
    --max-turns "$SMILE_MAX_TURNS" \
    --out "$SMILE_DATA_OUT" |& tee -a "$RUN_DIR/pipeline.log"
fi

echo "[2/2] Eval 3 models (zeroshot) ..."

COMMON_ARGS=(
  --data "$SMILE_DATA_OUT"
  --base "$BASE_MODEL"
  --adapter-self "$ADAPTER_SELF_COG"
  --adapter-dpo "$ADAPTER_DPO"
  --out-dir "$RUN_DIR"
  --batch-size "$BATCH_SIZE"
  --warmup-batch-size "$WARMUP_BATCH_SIZE"
  --max-input-len "$MAX_INPUT_LEN"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --top-p "$TOP_P"
  --temperature "$TEMPERATURE"
  --repetition-penalty "$REPETITION_PENALTY"
)
if [ "$DO_SAMPLE" = "1" ]; then COMMON_ARGS+=(--do-sample); fi
if [ "$LOAD_IN_4BIT" = "1" ]; then COMMON_ARGS+=(--load-in-4bit); fi
if [ "$ENABLE_THINKING" = "1" ]; then COMMON_ARGS+=(--enable-thinking); fi
if [ "$RESUME" = "1" ]; then COMMON_ARGS+=(--resume); fi

# Use both GPUs: dpo on cuda:0, base on cuda:1; optionally self_cognition on the first freed GPU.
"$PY" -u "$DIALOGUE_DIR/runners/run_single_dataset_paper_eval.py" "${COMMON_ARGS[@]}" --which dpo --device cuda:0 --out-dir "$RUN_DIR" >"$RUN_DIR/dpo_cuda0.log" 2>&1 &
pid_dpo=$!
"$PY" -u "$DIALOGUE_DIR/runners/run_single_dataset_paper_eval.py" "${COMMON_ARGS[@]}" --which base --device cuda:1 --out-dir "$RUN_DIR" >"$RUN_DIR/base_cuda1.log" 2>&1 &
pid_base=$!

pid_self=""
if [ "$RUN_SELF_COG" = "1" ]; then
  set +e
  wait -n "$pid_dpo" "$pid_base"
  set -e

  gpu_for_self="cuda:0"
  if ! kill -0 "$pid_dpo" 2>/dev/null; then
    gpu_for_self="cuda:0"
  else
    gpu_for_self="cuda:1"
  fi

  echo "[INFO] starting self_cognition on $gpu_for_self" |& tee -a "$RUN_DIR/pipeline.log"
  "$PY" -u "$DIALOGUE_DIR/runners/run_single_dataset_paper_eval.py" "${COMMON_ARGS[@]}" --which self_cognition --device "$gpu_for_self" --out-dir "$RUN_DIR" >"$RUN_DIR/self_cognition_${gpu_for_self//:/}.log" 2>&1 &
  pid_self=$!
fi

wait "$pid_dpo" || true
wait "$pid_base" || true
if [ -n "${pid_self:-}" ]; then
  wait "$pid_self" || true
fi

echo "[OK] SMILE eval done: $RUN_DIR"
