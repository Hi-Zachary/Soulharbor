#!/usr/bin/env bash
# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/llamafactory/run_main_sft_then_self_cognition.sh
# 原先用途: 主 SFT 完成后串联自我认知 SFT。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

set -euo pipefail

# Serial pipeline: main SFT (one more epoch) -> self-cognition SFT.
#
# This produces TWO separate LoRA artifacts:
# - saves/qwen14b/lora/sft_main2_<RUN_TAG>
# - saves/qwen14b/lora/sft_self_cognition_<RUN_TAG>
#
# Required:
#   MODEL_NAME_OR_PATH=/path/to/Qwen3-14B
#   BASE_ADAPTER=/path/to/previous_main_sft_adapter   # e.g. saves/qwen14b/lora/sft_20260420_083101
#
# Optional:
#   CUDA_VISIBLE_DEVICES=0,1
#   NPROC_PER_NODE=2
#   RUN_TAG=custom_tag
#
# Main SFT knobs:
#   MAIN_DATASET_NAME=soulharbor_mixed_campus_sft
#   MAIN_EPOCHS=1.0
#   MAIN_LR=2e-5
#   MAIN_CUTOFF_LEN=2048
#   MAIN_GRAD_ACC=16
#   MAIN_SAVE_STEPS=1000
#   MAIN_EVAL_STEPS=1000
#   MAIN_VAL_SIZE=0.02
#   MAIN_LOAD_BEST=1
#
# Self-cognition knobs:
#   SELF_DATASET_NAME=soulharbor_self_cognition_sft
#   SELF_EPOCHS=1.0
#   SELF_LR=1e-5
#   SELF_CUTOFF_LEN=1024
#   SELF_GRAD_ACC=1            # important: dataset is tiny; keep updates frequent
#   SELF_SAVE_STEPS=50
#   SELF_EVAL_STEPS=50
#   SELF_VAL_SIZE=0.02
#   SELF_LOAD_BEST=1
#
# Usage (after `conda activate soulhar`):
#   MODEL_NAME_OR_PATH=... BASE_ADAPTER=... CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
#   bash train/scripts/llamafactory/run/run_main_sft_then_self_cognition.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH is required}"
BASE_ADAPTER="${BASE_ADAPTER:?BASE_ADAPTER is required}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

MAIN_OUT="${ROOT_DIR}/saves/qwen14b/lora/sft_main2_${RUN_TAG}"
SELF_OUT="${ROOT_DIR}/saves/qwen14b/lora/sft_self_cognition_${RUN_TAG}"

echo "[1/2] main SFT: base_adapter=${BASE_ADAPTER}"

export OUTPUT_DIR="${MAIN_OUT}"
export ADAPTER_NAME_OR_PATH="${BASE_ADAPTER}"
export DATASET_NAME="${MAIN_DATASET_NAME:-soulharbor_mixed_campus_sft}"
export EPOCHS="${MAIN_EPOCHS:-1.0}"
export LR="${MAIN_LR:-2e-5}"
export CUTOFF_LEN="${MAIN_CUTOFF_LEN:-2048}"
export GRAD_ACC="${MAIN_GRAD_ACC:-16}"
export SAVE_STEPS="${MAIN_SAVE_STEPS:-1000}"
export EVAL_STRATEGY="steps"
export EVAL_STEPS="${MAIN_EVAL_STEPS:-1000}"
export VAL_SIZE="${MAIN_VAL_SIZE:-0.02}"
export LOAD_BEST="${MAIN_LOAD_BEST:-1}"
export TEMPLATE="${TEMPLATE:-qwen3_nothink}"

bash "${ROOT_DIR}/train/scripts/llamafactory/run/train_sft_qwen14b_qlora.sh"
echo "[done] main SFT out: ${MAIN_OUT}"

echo "[2/2] self-cognition SFT: adapter=${MAIN_OUT}"

export OUTPUT_DIR="${SELF_OUT}"
export ADAPTER_NAME_OR_PATH="${MAIN_OUT}"
export DATASET_NAME="${SELF_DATASET_NAME:-soulharbor_self_cognition_sft}"
export EPOCHS="${SELF_EPOCHS:-1.0}"
export LR="${SELF_LR:-1e-5}"
export CUTOFF_LEN="${SELF_CUTOFF_LEN:-1024}"
export GRAD_ACC="${SELF_GRAD_ACC:-1}"
export SAVE_STEPS="${SELF_SAVE_STEPS:-50}"
export EVAL_STRATEGY="steps"
export EVAL_STEPS="${SELF_EVAL_STEPS:-50}"
export VAL_SIZE="${SELF_VAL_SIZE:-0.02}"
export LOAD_BEST="${SELF_LOAD_BEST:-1}"

bash "${ROOT_DIR}/train/scripts/llamafactory/run/train_sft_qwen14b_qlora.sh"
echo "[done] self-cognition out: ${SELF_OUT}"

echo ""
echo "Artifacts:"
echo "  MAIN: ${MAIN_OUT}"
echo "  SELF: ${SELF_OUT}"
