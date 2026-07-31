#!/usr/bin/env bash
# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/llamafactory/run_orpo_synth_minimax.sh
# 原先用途: ORPO 偏好对齐实验脚本（非当前生产主路径）。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

set -euo pipefail

# Preference alignment (ORPO) on synthetic DPO pairs labeled by MiniMax judge.
#
# Requires:
#   MODEL_NAME_OR_PATH=/path/to/Qwen3-14B
#   BASE_ADAPTER=/path/to/sft_adapter   # e.g. archive/weights/lora/sft_self_cognition_20260422_112858
#
# Produces:
#   saves/qwen14b/lora/orpo_synth_${RUN_TAG}
#
# Notes:
# - We use `--pref_loss orpo` to avoid a separate reference model (more stable for LoRA setups).
# - Ensure `data/llm/dpo_synth_minimax.jsonl` exists before running.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH is required}"
BASE_ADAPTER="${BASE_ADAPTER:?BASE_ADAPTER is required}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT="${ROOT_DIR}/saves/qwen14b/lora/orpo_synth_${RUN_TAG}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/train/data/llm}"
DATASET_NAME="${DATASET_NAME:-soulharbor_dpo_synth_minimax}"
TEMPLATE="${TEMPLATE:-qwen3_nothink}"

CUTOFF_LEN="${CUTOFF_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACC="${GRAD_ACC:-16}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-1.0}"

SAVE_STEPS="${SAVE_STEPS:-200}"
LOG_STEPS="${LOG_STEPS:-10}"

SEED="${SEED:-42}"
VAL_SIZE="${VAL_SIZE:-0.02}"
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
EVAL_STEPS="${EVAL_STEPS:-200}"
EVAL_BS="${EVAL_BS:-1}"
LOAD_BEST="${LOAD_BEST:-1}"
FLASH_ATTN="${FLASH_ATTN:-sdpa}"
GRAD_CKPT="${GRAD_CKPT:-1}"

PREF_BETA="${PREF_BETA:-0.1}"

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  export FORCE_TORCHRUN="${NPROC_PER_NODE}"
fi
LAUNCHER=(llamafactory-cli train)

ARGS=(
  --stage dpo \
  --pref_loss orpo \
  --pref_beta "${PREF_BETA}" \
  --do_train \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --adapter_name_or_path "${BASE_ADAPTER}" \
  --template "${TEMPLATE}" \
  --enable_thinking false \
  --dataset "${DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --seed "${SEED}" \
  --finetuning_type lora \
  --quantization_bit 4 \
  --lora_target all \
  --lora_rank 32 \
  --lora_alpha 64 \
  --output_dir "${OUT}" \
  --overwrite_cache \
  --overwrite_output_dir \
  --cutoff_len "${CUTOFF_LEN}" \
  --preprocessing_num_workers 4 \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --lr_scheduler_type cosine \
  --logging_steps "${LOG_STEPS}" \
  --warmup_steps 20 \
  --save_steps "${SAVE_STEPS}" \
  --learning_rate "${LR}" \
  --num_train_epochs "${EPOCHS}" \
  --flash_attn "${FLASH_ATTN}" \
  $([[ "${GRAD_CKPT}" == "1" ]] && echo --gradient_checkpointing true) \
  --plot_loss \
  --fp16
)

# Eval/save strategy compatibility:
if [[ "${EVAL_STRATEGY}" == "no" || "${VAL_SIZE}" == "0" ]]; then
  LOAD_BEST="0"
else
  if [[ "${LOAD_BEST}" == "1" ]]; then
    SAVE_STRATEGY="${EVAL_STRATEGY}"
  else
    SAVE_STRATEGY="steps"
  fi
  ARGS+=(
    --val_size "${VAL_SIZE}"
    --do_eval
    --eval_strategy "${EVAL_STRATEGY}"
    --eval_steps "${EVAL_STEPS}"
    --per_device_eval_batch_size "${EVAL_BS}"
    --save_strategy "${SAVE_STRATEGY}"
  )
fi

if [[ "${LOAD_BEST}" == "1" ]]; then
  ARGS+=(--load_best_model_at_end true)
fi

echo "[orpo] dataset=${DATASET_NAME}"
echo "[orpo] base_adapter=${BASE_ADAPTER}"
echo "[orpo] out=${OUT}"

"${LAUNCHER[@]}" "${ARGS[@]}"

echo "[done] ORPO_OUT=${OUT}"

