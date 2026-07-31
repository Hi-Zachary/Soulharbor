#!/usr/bin/env bash
# SoulChat 主对话评测流水线（seen/unseen × base/sft/dpo）
# 位置: evaluate/dialogue/
# 结构: sampling/ + runners/ + lib/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIALOGUE_DIR="${ROOT}/evaluate/dialogue"
export PYTHONPATH="${DIALOGUE_DIR}/lib:${DIALOGUE_DIR}/sampling:${DIALOGUE_DIR}/runners${PYTHONPATH:+:$PYTHONPATH}"

CONDA_ENV="${CONDA_ENV:-soulhar}"
ENV_PY="/public/home/shijian/miniconda3/envs/${CONDA_ENV}/bin/python"
if [ -x "$ENV_PY" ]; then
  PY=("$ENV_PY" -u)
else
  PY=(python -u)
fi

N="${N:-1000}"
SEED="${SEED:-20260426}"
MAX_TURNS="${MAX_TURNS:-5}"
PER_TOPIC_RESERVOIR="${PER_TOPIC_RESERVOIR:-300}"
SCAN_LIMIT="${SCAN_LIMIT:-0}"
FAST_STOP="${FAST_STOP:-1}"

BASE_MODEL="${BASE_MODEL:-models/Qwen3-14B}"
ADAPTER_SELF_COG="${ADAPTER_SELF_COG:-saves/qwen14b/lora/sft_self_cognition_20260422_112858}"
ADAPTER_DPO="${ADAPTER_DPO:-saves/qwen14b/lora/dpo_synth_20260425_060332}"

LIMIT="${LIMIT:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-1536}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
DEVICE="${DEVICE:-cuda:0}"
ATTN="${ATTN:-sdpa}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
DO_SAMPLE="${DO_SAMPLE:-1}"
TOP_P="${TOP_P:-0.75}"
TEMPERATURE="${TEMPERATURE:-0.95}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
PARALLEL="${PARALLEL:-1}"
DEVICE_SEEN="${DEVICE_SEEN:-cuda:0}"
DEVICE_UNSEEN="${DEVICE_UNSEEN:-cuda:1}"
BATCH_SIZE_START="${BATCH_SIZE_START:-16}"
SKIP_SELF_COG="${SKIP_SELF_COG:-0}"

SEEN_DATA="${SEEN_DATA:-evaluate/dialogue/data/soulchat_seen_1k.jsonl}"
UNSEEN_DATA="${UNSEEN_DATA:-evaluate/dialogue/data/soulchat_unseen_1k.jsonl}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/evaluate/dialogue/runs/soulchat_corpus_eval_${TS}"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/pipeline.log"

echo "[RUN] $RUN_DIR" | tee -a "$LOG"
echo "[1/3] Sampling eval sets ..." | tee -a "$LOG"
"${PY[@]}" "$DIALOGUE_DIR/sampling/sample_soulchat_eval_sets.py" \
  --n "$N" \
  --seed "$SEED" \
  --max-turns "$MAX_TURNS" \
  --per-topic-reservoir "$PER_TOPIC_RESERVOIR" \
  --scan-limit "$SCAN_LIMIT" \
  --seen-out "$SEEN_DATA" \
  --unseen-out "$UNSEEN_DATA" \
  $( [ "$FAST_STOP" = "1" ] && echo "--fast-stop" ) |& tee -a "$LOG"

echo "[2/3] Running metrics ..." | tee -a "$LOG"

COMMON_ARGS=(
  --seen-data "$SEEN_DATA"
  --unseen-data "$UNSEEN_DATA"
  --base "$BASE_MODEL"
  --adapter-self "$ADAPTER_SELF_COG"
  --adapter-dpo "$ADAPTER_DPO"
  --out-dir "$RUN_DIR"
  --limit "$LIMIT"
  --max-input-len "$MAX_INPUT_LEN"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --repetition-penalty "$REPETITION_PENALTY"
  --attn "$ATTN"
)
if [ "$DO_SAMPLE" = "1" ]; then
  COMMON_ARGS+=(--do-sample)
fi
if [ "$LOAD_IN_4BIT" = "1" ]; then
  COMMON_ARGS+=(--load-in-4bit)
fi
if [ "$ENABLE_THINKING" = "1" ]; then
  COMMON_ARGS+=(--enable-thinking)
fi
if [ "$SKIP_SELF_COG" = "1" ]; then
  COMMON_ARGS+=(--skip-self-cognition)
fi

if [ "$PARALLEL" = "1" ]; then
  echo "[INFO] parallel=1 seen_device=$DEVICE_SEEN unseen_device=$DEVICE_UNSEEN" | tee -a "$LOG"
  SEEN_LOG="$RUN_DIR/seen.log"
  UNSEEN_LOG="$RUN_DIR/unseen.log"
  echo "[INFO] logging: $SEEN_LOG , $UNSEEN_LOG" | tee -a "$LOG"

  run_split () {
    local mode="$1"
    local device="$2"
    local log_path="$3"
    local bs="$BATCH_SIZE_START"
    echo "[START] mode=$mode device=$device batch_size_start=$bs" >>"$log_path"
    while true; do
      echo "[TRY] mode=$mode device=$device batch_size=$bs" >>"$log_path"
      rc=0
      "${PY[@]}" "$DIALOGUE_DIR/runners/run_soulchat_paper_eval.py" \
        "${COMMON_ARGS[@]}" \
        --resume \
        --mode "$mode" \
        --device "$device" \
        --batch-size "$bs" >>"$log_path" 2>&1 || rc=$?
      if [ "$rc" = "0" ]; then
        echo "[DONE] mode=$mode device=$device batch_size=$bs" >>"$log_path"
        return 0
      fi
      if [ "$rc" = "99" ] && [ "$bs" -gt 1 ]; then
        new_bs=$(( (bs + 1) / 2 ))
        if [ "$new_bs" -lt 1 ]; then new_bs=1; fi
        echo "[OOM] mode=$mode device=$device batch_size=$bs -> $new_bs" >>"$log_path"
        bs="$new_bs"
        continue
      fi
      echo "[FAIL] mode=$mode device=$device rc=$rc batch_size=$bs" >>"$log_path"
      return "$rc"
    done
  }

  run_split seen "$DEVICE_SEEN" "$SEEN_LOG" &
  pid_seen=$!

  run_split unseen "$DEVICE_UNSEEN" "$UNSEEN_LOG" &
  pid_unseen=$!

  echo "[INFO] pid_seen=$pid_seen pid_unseen=$pid_unseen" | tee -a "$LOG"
  wait "$pid_seen"
  wait "$pid_unseen"
  echo "[OK] parallel runs finished" | tee -a "$LOG"
else
  echo "[INFO] parallel=0 device=$DEVICE" | tee -a "$LOG"
  "${PY[@]}" "$DIALOGUE_DIR/runners/run_soulchat_paper_eval.py" \
    "${COMMON_ARGS[@]}" \
    --mode both \
    --device "$DEVICE" |& tee -a "$LOG"
fi

echo "[3/3] Summarizing ..." | tee -a "$LOG"
"${PY[@]}" "$DIALOGUE_DIR/runners/summarize_soulchat_eval_runs.py" --run-dir "$RUN_DIR" |& tee -a "$LOG"

echo "[OK] Done. Summary: $RUN_DIR/summary.md" | tee -a "$LOG"
