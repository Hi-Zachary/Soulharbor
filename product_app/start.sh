#!/usr/bin/env bash
set -euo pipefail

# Start (or safely restart) the SoulHarbor Product App (FastAPI).
#
# Usage:
#   bash product_app/start.sh
#   PORT=8001 bash product_app/start.sh

PORT="${PORT:-8000}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/CondaEnv/soulhar

cd "${ROOT_DIR}"

find_listener_pid() {
  local port="$1"
  local pid=""
  if command -v ss >/dev/null 2>&1; then
    pid="$(ss -ltnp 2>/dev/null | awk -v p=":${port}" '$4 ~ p {print $NF}' | awk -F'pid=' 'NF>1 {print $2}' | awk -F',' 'NF>0 {print $1}' | awk 'NF>0 {print $1; exit}')"
  fi
  if [[ -z "${pid}" ]] && command -v netstat >/dev/null 2>&1; then
    pid="$(netstat -ltnp 2>/dev/null | awk -v p=":${port}" '$4 ~ p {print $7}' | awk -F'/' 'NF>0 {print $1}' | awk 'NF>0 && $1 != "-" {print $1; exit}')"
  fi
  echo "${pid}"
}

PID="$(find_listener_pid "${PORT}")"

if [[ -n "${PID}" ]]; then
  CMDLINE="$(ps -p "${PID}" -o args= 2>/dev/null || true)"
  if [[ "${CMDLINE}" == *"uvicorn product_app.app.main:app"* ]]; then
    echo "Killing existing SoulHarbor server on port ${PORT}: PID=${PID}"
    kill "${PID}" || true
    for _ in {1..20}; do
      sleep 1
      if [[ -z "$(find_listener_pid "${PORT}")" ]]; then
        break
      fi
    done
    if [[ -n "$(find_listener_pid "${PORT}")" ]]; then
      echo "ERROR: SoulHarbor server did not release port ${PORT} in time" >&2
      exit 1
    fi
  else
    echo "ERROR: port ${PORT} is in use by another process:" >&2
    echo "  pid=${PID}" >&2
    echo "  cmd: ${CMDLINE}" >&2
    exit 1
  fi
fi

if [[ -z "${SOULHARBOR_LLM_ADAPTER:-}" ]]; then
  LATEST_DPO="$(ls -dt saves/qwen14b/lora/dpo_synth_* 2>/dev/null | head -n 1)"
  if [[ -n "${LATEST_DPO}" ]]; then
    export SOULHARBOR_LLM_ADAPTER="${LATEST_DPO}"
    echo "Auto-picked SOULHARBOR_LLM_ADAPTER=${SOULHARBOR_LLM_ADAPTER}"
  else
    LATEST_ADAPTER="$(ls -dt archive/weights/lora/sft_self_cognition_* 2>/dev/null | head -n 1)"
    if [[ -n "${LATEST_ADAPTER}" ]]; then
      export SOULHARBOR_LLM_ADAPTER="${LATEST_ADAPTER}"
      echo "Auto-picked SOULHARBOR_LLM_ADAPTER=${LATEST_ADAPTER} (archived SFT fallback)"
    else
      echo "WARN: no dpo_synth_* in saves/qwen14b/lora/ and no archived sft_self_cognition_*" >&2
    fi
  fi
fi

if [[ -z "${SOULHARBOR_MEMORY_ENCODER_BASE:-}" ]]; then
  if [[ -f "models/encoders/bge-m3/config.json" ]]; then
    export SOULHARBOR_MEMORY_ENCODER_BASE="models/encoders/bge-m3"
    echo "Auto-picked SOULHARBOR_MEMORY_ENCODER_BASE=${SOULHARBOR_MEMORY_ENCODER_BASE}"
  else
    echo "WARN: bge-m3 not found; memory encoder must be set via SOULHARBOR_MEMORY_ENCODER_BASE" >&2
  fi
fi

# Prefer GPU for LLM + memory embedder + intent classifier when VRAM allows (e.g. 24GB).
# Override with SOULHARBOR_*_DEVICE=cpu if you need to free GPU memory.
export SOULHARBOR_LLM_4BIT="${SOULHARBOR_LLM_4BIT:-1}"
export SOULHARBOR_MEMORY_EMBED_DEVICE="${SOULHARBOR_MEMORY_EMBED_DEVICE:-cuda}"
export SOULHARBOR_CLASSIFIER_DEVICE="${SOULHARBOR_CLASSIFIER_DEVICE:-cuda}"
export MEMORY_BACKEND="${MEMORY_BACKEND:-er}"
echo "VRAM policy: LLM_4BIT=${SOULHARBOR_LLM_4BIT} embed=${SOULHARBOR_MEMORY_EMBED_DEVICE} cls=${SOULHARBOR_CLASSIFIER_DEVICE} memory=${MEMORY_BACKEND}"

echo "Starting SoulHarbor Product App: port=${PORT}"
exec uvicorn product_app.app.main:app --host 0.0.0.0 --port "${PORT}" --log-level info
