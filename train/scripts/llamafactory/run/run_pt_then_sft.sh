#!/usr/bin/env bash
# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/llamafactory/run_pt_then_sft.sh
# 原先用途: PT 完成后跑 SFT。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

set -euo pipefail

# Run PT then SFT sequentially (two separate runs, two separate outputs).
#
# Required:
#   MODEL_NAME_OR_PATH=/path/to/qwen14b
#
# Optional:
#   CUDA_VISIBLE_DEVICES=0,1
#   NPROC_PER_NODE=2
#   RUN_TAG=custom_tag
#
# PT knobs:
#   PT_EPOCHS=3.0
#   PT_CUTOFF_LEN=1024
#   PT_LR=5e-5
#
# SFT knobs:
#   SFT_DATASET_NAME=soulharbor_mixed_campus_sft
#   SFT_EPOCHS=3.0
#   SFT_CUTOFF_LEN=2048
#   SFT_LR=2e-5
#
# Behavior:
# - PT output:  saves/qwen14b/lora/pt_${RUN_TAG}
# - SFT output: saves/qwen14b/lora/sft_${RUN_TAG}
# - SFT automatically uses PT adapter as its starting adapter.
#
# Usage (after `conda activate soulhar`):
#   MODEL_NAME_OR_PATH=/public/home/shijian/zachary/ZacPro/SoulHarbor/models/Qwen3-14B \
#   CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
#   bash train/scripts/llamafactory/run/run_pt_then_sft.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH is required}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

PT_OUT="${ROOT_DIR}/saves/qwen14b/lora/pt_${RUN_TAG}"
SFT_OUT="${ROOT_DIR}/saves/qwen14b/lora/sft_${RUN_TAG}"

echo "[pt_then_sft] RUN_TAG=${RUN_TAG}"
echo "[pt_then_sft] PT_OUT=${PT_OUT}"
echo "[pt_then_sft] SFT_OUT=${SFT_OUT}"

# -------- PT --------
RUN_TAG="${RUN_TAG}" \
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" \
PT_EPOCHS="${PT_EPOCHS:-3.0}" \
PT_CUTOFF_LEN="${PT_CUTOFF_LEN:-1024}" \
PT_LR="${PT_LR:-5e-5}" \
bash "${ROOT_DIR}/train/scripts/llamafactory/run/run_pt.sh"

if [[ ! -d "${PT_OUT}" ]]; then
  echo "[pt_then_sft][error] PT output dir not found: ${PT_OUT}" >&2
  exit 2
fi

# -------- SFT --------
RUN_TAG="${RUN_TAG}" \
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" \
ADAPTER_NAME_OR_PATH="${PT_OUT}" \
DATASET_NAME="${SFT_DATASET_NAME:-soulharbor_mixed_campus_sft}" \
SFT_EPOCHS="${SFT_EPOCHS:-3.0}" \
SFT_CUTOFF_LEN="${SFT_CUTOFF_LEN:-2048}" \
SFT_LR="${SFT_LR:-2e-5}" \
bash "${ROOT_DIR}/train/scripts/llamafactory/run/run_sft.sh"

if [[ ! -d "${SFT_OUT}" ]]; then
  echo "[pt_then_sft][error] SFT output dir not found: ${SFT_OUT}" >&2
  exit 3
fi

echo "[done] PT=${PT_OUT}"
echo "[done] SFT=${SFT_OUT}"

