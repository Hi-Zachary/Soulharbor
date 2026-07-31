#!/usr/bin/env bash
# Train MacBERT-base intent classifier on intent_dataset_v4 (1 epoch).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/root/autodl-tmp/CondaEnv/soulhar}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${ROOT_DIR}/models/encoders/chinese-macbert-base}"
INTENT_DIR="${INTENT_DIR:-${ROOT_DIR}/train/data/classifiers/train/intent}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/classifiers}"

MAX_LENGTH="${MAX_LENGTH:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-2}"
FP16="${FP16:-1}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-1}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_PATH}"

cd "${ROOT_DIR}"

CMD=(
  python train/scripts/classifiers/train_classifiers_hf.py
  --task intent
  --intent-dir "${INTENT_DIR}"
  --model-name-or-path "${MODEL_NAME_OR_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --max-length "${MAX_LENGTH}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --warmup-ratio "${WARMUP_RATIO}"
  --seed "${SEED}"
  --num-workers "${NUM_WORKERS}"
)

if [[ "${FP16}" == "1" ]]; then
  CMD+=(--fp16)
fi

if [[ "${OVERWRITE_OUTPUT_DIR}" == "1" ]]; then
  CMD+=(--overwrite-output-dir)
fi

echo "[intent-v4] ROOT_DIR=${ROOT_DIR}"
echo "[intent-v4] INTENT_DIR=${INTENT_DIR}"
echo "[intent-v4] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[intent-v4] EPOCHS=${EPOCHS}"
echo "[intent-v4] Running: ${CMD[*]}"

"${CMD[@]}"
