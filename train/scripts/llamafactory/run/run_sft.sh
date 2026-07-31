#!/usr/bin/env bash
# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/llamafactory/run_sft.sh
# 原先用途: 启动主 SFT 的薄封装。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

set -euo pipefail

# Run SFT only (QLoRA).
#
# Required:
#   MODEL_NAME_OR_PATH=/path/to/qwen14b
#
# Optional:
#   CUDA_VISIBLE_DEVICES=0,1
#   RUN_TAG=custom_tag
#   SFT_EPOCHS=3.0
#   SFT_CUTOFF_LEN=2048
#   TEMPLATE=qwen3_nothink
#   SFT_LR=2e-5
#   SFT_LOG_STEPS=10
#   SFT_SAVE_STEPS=50
#   SFT_VAL_SIZE=0.02
#   SFT_EVAL_STRATEGY=steps   # no/steps/epoch
#   SFT_EVAL_STEPS=50
#   SFT_EVAL_BS=1
#   SFT_LOAD_BEST=1
#   SFT_SEED=42
#   SFT_FLASH_ATTN=sdpa
#   SFT_GRAD_CKPT=1
#   ADAPTER_NAME_OR_PATH=/path/to/pt_adapter   # resume from a PT adapter
#
# Usage (after `conda activate soulhar`):
#   MODEL_NAME_OR_PATH=/path/to/qwen14b CUDA_VISIBLE_DEVICES=0,1 bash train/scripts/llamafactory/run/run_sft.sh
#   MODEL_NAME_OR_PATH=/path/to/qwen14b ADAPTER_NAME_OR_PATH=/path/to/pt_adapter bash train/scripts/llamafactory/run/run_sft.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SFT_OUT="${ROOT_DIR}/saves/qwen14b/lora/sft_${RUN_TAG}"

OUTPUT_DIR="${SFT_OUT}" \
EPOCHS="${SFT_EPOCHS:-3.0}" \
CUTOFF_LEN="${SFT_CUTOFF_LEN:-2048}" \
TEMPLATE="${TEMPLATE:-qwen3_nothink}" \
LR="${SFT_LR:-2e-5}" \
LOG_STEPS="${SFT_LOG_STEPS:-10}" \
SAVE_STEPS="${SFT_SAVE_STEPS:-50}" \
VAL_SIZE="${SFT_VAL_SIZE:-0.02}" \
EVAL_STRATEGY="${SFT_EVAL_STRATEGY:-steps}" \
EVAL_STEPS="${SFT_EVAL_STEPS:-50}" \
EVAL_BS="${SFT_EVAL_BS:-1}" \
LOAD_BEST="${SFT_LOAD_BEST:-1}" \
SEED="${SFT_SEED:-42}" \
FLASH_ATTN="${SFT_FLASH_ATTN:-sdpa}" \
GRAD_CKPT="${SFT_GRAD_CKPT:-1}" \
ADAPTER_NAME_OR_PATH="${ADAPTER_NAME_OR_PATH:-}" \
bash "${ROOT_DIR}/train/scripts/llamafactory/run/train_sft_qwen14b_qlora.sh"

echo "[done] SFT=${SFT_OUT}"
