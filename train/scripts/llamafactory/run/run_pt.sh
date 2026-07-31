#!/usr/bin/env bash
# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/llamafactory/run_pt.sh
# 原先用途: 启动 PT 训练的薄封装。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

set -euo pipefail

# Run PT (continued pretraining) only (QLoRA).
#
# Required:
#   MODEL_NAME_OR_PATH=/path/to/qwen14b
#
# Optional:
#   CUDA_VISIBLE_DEVICES=0,1
#   RUN_TAG=custom_tag
#   PT_EPOCHS=3.0
#   PT_CUTOFF_LEN=1024
#   TEMPLATE=qwen3_nothink
#   PT_LR=5e-5
#   PT_LOG_STEPS=10
#   PT_SAVE_STEPS=200
#   PT_VAL_SIZE=0.0
#   PT_EVAL_STRATEGY=no       # no/steps/epoch
#   PT_EVAL_STEPS=50
#   PT_EVAL_BS=1
#   PT_LOAD_BEST=0
#   PT_SEED=42
#   PT_RESUME_FROM=           # e.g. /path/to/output_dir/checkpoint-42
#
# Usage (after `conda activate soulhar`):
#   MODEL_NAME_OR_PATH=/path/to/qwen14b CUDA_VISIBLE_DEVICES=0,1 bash train/scripts/llamafactory/run/run_pt.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PT_OUT="${ROOT_DIR}/saves/qwen14b/lora/pt_${RUN_TAG}"

OUTPUT_DIR="${PT_OUT}" \
EPOCHS="${PT_EPOCHS:-3.0}" \
CUTOFF_LEN="${PT_CUTOFF_LEN:-1024}" \
TEMPLATE="${TEMPLATE:-qwen3_nothink}" \
LR="${PT_LR:-5e-5}" \
LOG_STEPS="${PT_LOG_STEPS:-10}" \
SAVE_STEPS="${PT_SAVE_STEPS:-200}" \
VAL_SIZE="${PT_VAL_SIZE:-0.0}" \
EVAL_STRATEGY="${PT_EVAL_STRATEGY:-no}" \
EVAL_STEPS="${PT_EVAL_STEPS:-50}" \
EVAL_BS="${PT_EVAL_BS:-1}" \
LOAD_BEST="${PT_LOAD_BEST:-0}" \
SEED="${PT_SEED:-42}" \
RESUME_FROM="${PT_RESUME_FROM:-}" \
bash "${ROOT_DIR}/train/scripts/llamafactory/run/train_pt_qwen14b_qlora.sh"

echo "[done] PT=${PT_OUT}"
