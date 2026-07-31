#!/usr/bin/env bash
# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/llamafactory/run_sft_self_cognition.sh
# 原先用途: 启动自我认知 SFT。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

set -euo pipefail

# Run a small SFT pass on self-cognition / identity alignment data.
#
# Required:
#   MODEL_NAME_OR_PATH=/path/to/qwen14b
#
# Optional:
#   CUDA_VISIBLE_DEVICES=0,1
#   NPROC_PER_NODE=2
#   RUN_TAG=custom_tag
#   ADAPTER_NAME_OR_PATH=/path/to/prev_adapter   # e.g. output of main SFT
#
# Knobs:
#   SFT_EPOCHS=1.0
#   SFT_CUTOFF_LEN=1024
#   SFT_LR=1e-5
#   SFT_DATASET_NAME=soulharbor_self_cognition_sft
#   SFT_GRAD_ACC=1            # important: dataset is tiny (78 lines); default 16 makes updates too few
#   SFT_SAVE_STEPS=50
#   SFT_EVAL_STEPS=50
#   SFT_VAL_SIZE=0.02
#   SFT_LOAD_BEST=1
#
# Output:
#   saves/qwen14b/lora/sft_self_cognition_${RUN_TAG}
#
# Usage (after `conda activate soulhar`):
#   MODEL_NAME_OR_PATH=/public/home/shijian/zachary/ZacPro/SoulHarbor/models/Qwen3-14B \
#   ADAPTER_NAME_OR_PATH=/path/to/sft_adapter \
#   CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
#   bash train/scripts/llamafactory/run/run_sft_self_cognition.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT="${ROOT_DIR}/saves/qwen14b/lora/sft_self_cognition_${RUN_TAG}"

OUTPUT_DIR="${OUT}" \
DATASET_NAME="${SFT_DATASET_NAME:-soulharbor_self_cognition_sft}" \
EPOCHS="${SFT_EPOCHS:-1.0}" \
CUTOFF_LEN="${SFT_CUTOFF_LEN:-1024}" \
LR="${SFT_LR:-1e-5}" \
GRAD_ACC="${SFT_GRAD_ACC:-1}" \
SAVE_STEPS="${SFT_SAVE_STEPS:-50}" \
EVAL_STRATEGY="${SFT_EVAL_STRATEGY:-steps}" \
EVAL_STEPS="${SFT_EVAL_STEPS:-50}" \
VAL_SIZE="${SFT_VAL_SIZE:-0.02}" \
LOAD_BEST="${SFT_LOAD_BEST:-1}" \
ADAPTER_NAME_OR_PATH="${ADAPTER_NAME_OR_PATH:-}" \
bash "${ROOT_DIR}/train/scripts/llamafactory/run/train_sft_qwen14b_qlora.sh"

echo "[done] SFT_SELF_COG=${OUT}"
