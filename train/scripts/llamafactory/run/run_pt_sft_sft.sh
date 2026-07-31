#!/usr/bin/env bash
# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/llamafactory/run_pt_sft_sft.sh
# 原先用途: PT → 主 SFT → 自我认知 SFT 串联。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

set -euo pipefail

# Run PT -> SFT(main) -> SFT(self-cognition) sequentially.
# Each stage writes to a separate output dir (no overwriting).
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
# Main SFT knobs:
#   SFT_DATASET_NAME=soulharbor_mixed_campus_sft
#   SFT_EPOCHS=3.0
#   SFT_CUTOFF_LEN=2048
#   SFT_LR=2e-5
#
# Self-cognition SFT knobs:
#   SELF_EPOCHS=1.0
#   SELF_CUTOFF_LEN=1024
#   SELF_LR=1e-5
#   SELF_DATASET_NAME=soulharbor_self_cognition_sft
#
# Outputs:
#   saves/qwen14b/lora/pt_${RUN_TAG}
#   saves/qwen14b/lora/sft_${RUN_TAG}
#   saves/qwen14b/lora/sft_self_cognition_${RUN_TAG}
#
# Usage (after `conda activate soulhar`):
#   MODEL_NAME_OR_PATH=/public/home/shijian/zachary/ZacPro/SoulHarbor/models/Qwen3-14B \
#   CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
#   bash train/scripts/llamafactory/run/run_pt_sft_sft.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH is required}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

PT_OUT="${ROOT_DIR}/saves/qwen14b/lora/pt_${RUN_TAG}"
SFT_OUT="${ROOT_DIR}/saves/qwen14b/lora/sft_${RUN_TAG}"
SELF_OUT="${ROOT_DIR}/saves/qwen14b/lora/sft_self_cognition_${RUN_TAG}"

echo "[pt_sft_sft] RUN_TAG=${RUN_TAG}"
echo "[pt_sft_sft] PT_OUT=${PT_OUT}"
echo "[pt_sft_sft] SFT_OUT=${SFT_OUT}"
echo "[pt_sft_sft] SELF_OUT=${SELF_OUT}"

# -------- PT --------
RUN_TAG="${RUN_TAG}" \
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" \
PT_EPOCHS="${PT_EPOCHS:-3.0}" \
PT_CUTOFF_LEN="${PT_CUTOFF_LEN:-1024}" \
PT_LR="${PT_LR:-5e-5}" \
bash "${ROOT_DIR}/train/scripts/llamafactory/run/run_pt.sh"

if [[ ! -d "${PT_OUT}" ]]; then
  echo "[pt_sft_sft][error] PT output dir not found: ${PT_OUT}" >&2
  exit 2
fi

# -------- SFT (main) --------
RUN_TAG="${RUN_TAG}" \
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" \
ADAPTER_NAME_OR_PATH="${PT_OUT}" \
DATASET_NAME="${SFT_DATASET_NAME:-soulharbor_mixed_campus_sft}" \
SFT_EPOCHS="${SFT_EPOCHS:-3.0}" \
SFT_CUTOFF_LEN="${SFT_CUTOFF_LEN:-2048}" \
SFT_LR="${SFT_LR:-2e-5}" \
bash "${ROOT_DIR}/train/scripts/llamafactory/run/run_sft.sh"

if [[ ! -d "${SFT_OUT}" ]]; then
  echo "[pt_sft_sft][error] SFT output dir not found: ${SFT_OUT}" >&2
  exit 3
fi

# -------- SFT (self-cognition) --------
RUN_TAG="${RUN_TAG}" \
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" \
ADAPTER_NAME_OR_PATH="${SFT_OUT}" \
SFT_DATASET_NAME="${SELF_DATASET_NAME:-soulharbor_self_cognition_sft}" \
SFT_EPOCHS="${SELF_EPOCHS:-1.0}" \
SFT_CUTOFF_LEN="${SELF_CUTOFF_LEN:-1024}" \
SFT_LR="${SELF_LR:-1e-5}" \
bash "${ROOT_DIR}/train/scripts/llamafactory/run/run_sft_self_cognition.sh"

if [[ ! -d "${SELF_OUT}" ]]; then
  echo "[pt_sft_sft][error] SELF SFT output dir not found: ${SELF_OUT}" >&2
  exit 4
fi

echo "[done] PT=${PT_OUT}"
echo "[done] SFT=${SFT_OUT}"
echo "[done] SFT_SELF_COG=${SELF_OUT}"

