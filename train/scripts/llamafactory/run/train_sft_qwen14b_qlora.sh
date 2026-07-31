#!/usr/bin/env bash
# [ARCHIVED — 非运行时依赖]
# 原路径: train/scripts/llamafactory/run/train_sft_qwen14b_qlora.sh
# 原先用途: Qwen3-14B QLoRA 监督微调（LLaMA-Factory stage=sft）。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

set -euo pipefail

# LLaMA-Factory(v0.9.4) SFT with QLoRA on Qwen3-14B.
#
# Required:
#   MODEL_NAME_OR_PATH=/path/to/qwen14b
#
# Optional overrides:
#   DATASET_DIR=data/llm
#   DATASET_NAME=soulharbor_data_pro_sft
#   OUTPUT_DIR=saves/qwen14b/lora/sft
#   TEMPLATE=qwen3_nothink
#   CUTOFF_LEN=2048
#   BATCH_SIZE=1
#   GRAD_ACC=16
#   LR=2e-5
#   EPOCHS=3.0
#   SAVE_STEPS=200
#   LOG_STEPS=10
#   NPROC_PER_NODE=2          # enable DDP via torchrun
#   SEED=42
#   VAL_SIZE=0.02
#   EVAL_STRATEGY=steps       # no/steps/epoch
#   EVAL_STEPS=200
#   EVAL_BS=1
#   LOAD_BEST=1
#   SAVE_STRATEGY=steps       # steps/epoch (auto-syncs with EVAL_STRATEGY if LOAD_BEST=1)
#   RESUME_FROM=              # e.g. /path/to/output_dir/checkpoint-xxx
#   FLASH_ATTN=sdpa           # auto/disabled/sdpa/fa2/fa3
#   GRAD_CKPT=1               # enable gradient checkpointing to save memory
#
# Usage:
#   MODEL_NAME_OR_PATH=/path/to/qwen14b bash train/scripts/llamafactory/run/train_sft_qwen14b_qlora.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH is required}"
ADAPTER_NAME_OR_PATH="${ADAPTER_NAME_OR_PATH:-}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/train/data/llm}"
DATASET_NAME="${DATASET_NAME:-soulharbor_data_pro_sft}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/saves/qwen14b/lora/sft}"
TEMPLATE="${TEMPLATE:-qwen3_nothink}"

CUTOFF_LEN="${CUTOFF_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACC="${GRAD_ACC:-16}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-3.0}"

SAVE_STEPS="${SAVE_STEPS:-200}"
LOG_STEPS="${LOG_STEPS:-10}"

SEED="${SEED:-42}"
VAL_SIZE="${VAL_SIZE:-0.02}"
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
EVAL_STEPS="${EVAL_STEPS:-200}"
EVAL_BS="${EVAL_BS:-1}"
LOAD_BEST="${LOAD_BEST:-1}"
RESUME_FROM="${RESUME_FROM:-}"
FLASH_ATTN="${FLASH_ATTN:-sdpa}"
GRAD_CKPT="${GRAD_CKPT:-1}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  export FORCE_TORCHRUN="${NPROC_PER_NODE}"
fi
LAUNCHER=(llamafactory-cli train)

ARGS=(
  --stage sft \
  --do_train \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  ${ADAPTER_NAME_OR_PATH:+--adapter_name_or_path "${ADAPTER_NAME_OR_PATH}"} \
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
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_cache \
  $([[ -z "${RESUME_FROM}" ]] && echo --overwrite_output_dir) \
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
  ${RESUME_FROM:+--resume_from_checkpoint "${RESUME_FROM}"} \
  --flash_attn "${FLASH_ATTN}" \
  $([[ "${GRAD_CKPT}" == "1" ]] && echo --gradient_checkpointing true) \
  --plot_loss \
  --fp16
)

# Eval/save strategy compatibility:
# transformers requires save_strategy == eval_strategy if load_best_model_at_end is enabled.
if [[ "${EVAL_STRATEGY}" == "no" || "${VAL_SIZE}" == "0" ]]; then
  LOAD_BEST="0"
else
  # keep save strategy aligned when selecting best
  if [[ "${LOAD_BEST}" == "1" ]]; then
    SAVE_STRATEGY="${EVAL_STRATEGY}"
  fi
  ARGS+=(
    --val_size "${VAL_SIZE}"
    --do_eval
    --eval_strategy "${EVAL_STRATEGY}"
    --eval_steps "${EVAL_STEPS}"
    --per_device_eval_batch_size "${EVAL_BS}"
  )
fi

if [[ "${SAVE_STRATEGY}" == "epoch" ]]; then
  ARGS+=(--save_strategy epoch)
elif [[ "${SAVE_STRATEGY}" == "steps" ]]; then
  ARGS+=(--save_strategy steps)
fi

if [[ "${LOAD_BEST}" == "1" ]]; then
  ARGS+=(--load_best_model_at_end true)
fi

"${LAUNCHER[@]}" "${ARGS[@]}"
