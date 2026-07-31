#!/usr/bin/env bash
# [ARCHIVED — 非运行时依赖]
# 原路径: train/scripts/llamafactory/run/train_pt_qwen14b_qlora.sh
# 原先用途: Qwen3-14B QLoRA 继续预训练（LLaMA-Factory stage=pt）。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

set -euo pipefail

# LLaMA-Factory(v0.9.4) PT (continued pretraining) with QLoRA on Qwen3-14B.
#
# Required:
#   MODEL_NAME_OR_PATH=/path/to/qwen14b
#
# Optional overrides:
#   DATASET_DIR=data/llm
#   DATASET_NAME=soulharbor_pt
#   OUTPUT_DIR=saves/qwen14b/lora/pt
#   TEMPLATE=qwen3_nothink
#   CUTOFF_LEN=1024
#   BATCH_SIZE=1
#   GRAD_ACC=16
#   LR=5e-5
#   EPOCHS=1.0
#   SAVE_STEPS=200
#   LOG_STEPS=10
#   NPROC_PER_NODE=2          # enable DDP via torchrun
#   SEED=42
#   VAL_SIZE=0.0              # set >0 to enable hold-out eval split
#   EVAL_STRATEGY=no          # one of: no, steps, epoch
#   EVAL_STEPS=50             # used when EVAL_STRATEGY=steps
#   EVAL_BS=1
#   LOAD_BEST=0               # set 1 to enable load_best_model_at_end
#   RESUME_FROM=              # e.g. /path/to/output_dir/checkpoint-42
#
# Usage:
#   MODEL_NAME_OR_PATH=/path/to/qwen14b bash train/scripts/llamafactory/run/train_pt_qwen14b_qlora.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH is required}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/train/data/llm}"
DATASET_NAME="${DATASET_NAME:-soulharbor_pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/saves/qwen14b/lora/pt}"
TEMPLATE="${TEMPLATE:-qwen3_nothink}"

CUTOFF_LEN="${CUTOFF_LEN:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACC="${GRAD_ACC:-16}"
LR="${LR:-5e-5}"
EPOCHS="${EPOCHS:-1.0}"

SAVE_STEPS="${SAVE_STEPS:-200}"
LOG_STEPS="${LOG_STEPS:-10}"

SEED="${SEED:-42}"
VAL_SIZE="${VAL_SIZE:-0.0}"
EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
EVAL_STEPS="${EVAL_STEPS:-50}"
EVAL_BS="${EVAL_BS:-1}"
LOAD_BEST="${LOAD_BEST:-0}"
RESUME_FROM="${RESUME_FROM:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  export FORCE_TORCHRUN="${NPROC_PER_NODE}"
fi
LAUNCHER=(llamafactory-cli train)

"${LAUNCHER[@]}" \
  --seed "${SEED}" \
  --stage pt \
  --do_train \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --template "${TEMPLATE}" \
  --enable_thinking false \
  --dataset "${DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --finetuning_type lora \
  --quantization_bit 4 \
  --lora_target all \
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
  --plot_loss \
  --fp16 \
  ${RESUME_FROM:+--resume_from_checkpoint "${RESUME_FROM}"} \
  $([[ "${VAL_SIZE}" != "0" && "${VAL_SIZE}" != "0.0" ]] && echo --val_size "${VAL_SIZE}") \
  $([[ "${EVAL_STRATEGY}" != "no" ]] && echo --do_eval --eval_strategy "${EVAL_STRATEGY}" --per_device_eval_batch_size "${EVAL_BS}") \
  $([[ "${EVAL_STRATEGY}" == "steps" ]] && echo --eval_steps "${EVAL_STEPS}") \
  $([[ "${LOAD_BEST}" == "1" ]] && echo --load_best_model_at_end true)
