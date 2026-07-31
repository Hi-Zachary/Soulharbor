#!/usr/bin/env bash
# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/setup_env.sh
# 原先用途: 创建/配置 conda 环境 soulhar 与依赖安装。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

set -euo pipefail

# Create SoulHarbor conda env with LLaMA-Factory installed from pip/editable package.
#
# Usage:
#   bash train/scripts/setup_env.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_DIR="/root/autodl-tmp/CondaEnv/soulhar"
LMF_DIR="/root/autodl-tmp/qixin/LLaMA-Factory"

source /root/miniconda3/etc/profile.d/conda.sh

if [[ ! -d "${ENV_DIR}" ]]; then
  echo "[1/4] Creating conda env at ${ENV_DIR} ..."
  conda create -p "${ENV_DIR}" python=3.11 -c conda-forge --override-channels -y
else
  echo "[1/4] Reusing existing env at ${ENV_DIR}"
fi

conda activate "${ENV_DIR}"

echo "[2/4] Installing PyTorch ..."
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# Avoid pulling a mismatched torch 2.12.x via torchvision/torchaudio.
pip uninstall -y torchvision torchaudio torchdata 2>/dev/null || true

echo "[3/4] Installing LLaMA-Factory ..."
if [[ -d "${LMF_DIR}" ]]; then
  pip install -e "${LMF_DIR}"
else
  pip install "llamafactory>=0.9.0"
fi

echo "[4/4] Installing SoulHarbor requirements ..."
pip install -r "${ROOT_DIR}/requirements.txt"

echo
echo "Done. Activate with:"
echo "  source /root/miniconda3/etc/profile.d/conda.sh && conda activate ${ENV_DIR}"
echo
llamafactory-cli help | head -5
