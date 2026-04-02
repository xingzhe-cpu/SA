#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.conda/env/bin/python"
ENV_NAME="${ENV_NAME:-LunarLander-v1}"

export DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data-dir}"
export OUT_DIR="${OUT_DIR:-${ROOT_DIR}/output-dir}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.mplconfig}"
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing project-local python: ${PYTHON_BIN}" >&2
  echo "Create or restore the conda env before running this script." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}" "${MPLCONFIGDIR}"

run_python() {
  "${PYTHON_BIN}" "$@"
}

sweep_line_for_env() {
  case "${ENV_NAME}" in
    LunarLander-v1) echo 0 ;;
    LunarLander-v5) echo 1 ;;
    *)
      echo "Unsupported ENV_NAME for diffusion smoke runs: ${ENV_NAME}" >&2
      exit 1
      ;;
  esac
}

usage() {
  cat <<EOF
Usage: scripts/local_pipeline.sh <command>

Commands:
  env-info              Print the local runtime configuration.
  eval-pretrained       Run eval_assistance with the pretrained checkpoint for ENV_NAME.
  train-sac-smoke       Run a minimal SAC smoke test for ENV_NAME.
  generate-data-smoke   Generate a tiny replay buffer for ENV_NAME.
  train-diffusion-smoke Train a tiny diffusion model for ENV_NAME.
  eval-smoke-model      Evaluate the latest smoke-trained diffusion checkpoint.
  full-smoke            Run the LunarLander smoke pipeline end-to-end.

Environment overrides:
  ENV_NAME              Default: LunarLander-v1
  DATA_DIR              Default: <repo>/data-dir
  OUT_DIR               Default: <repo>/output-dir
  DDPM_MODEL_PATH       Default for smoke diffusion runs: <repo>/output-dir/ddpm-smoke
EOF
}

command="${1:-}"
case "${command}" in
  env-info)
    cat <<EOF
ROOT_DIR=${ROOT_DIR}
PYTHON_BIN=${PYTHON_BIN}
ENV_NAME=${ENV_NAME}
DATA_DIR=${DATA_DIR}
OUT_DIR=${OUT_DIR}
WANDB_MODE=${WANDB_MODE}
MPLCONFIGDIR=${MPLCONFIGDIR}
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION}
EOF
    ;;

  eval-pretrained)
    run_python -m diffusha.diffusion.evaluation.eval_assistance \
      --env-name "${ENV_NAME}" \
      --out-dir "${OUT_DIR}" \
      --num-episodes 1 \
      --num-envs 1
    ;;

  train-sac-smoke)
    run_python -m diffusha.data_collection.train_sac \
      --env-name "${ENV_NAME}" \
      --steps 10 \
      --replay-start-size 2 \
      --batch-size 2 \
      --eval-n-runs 1 \
      --eval-interval 10 \
      --checkpoint-freq 10 \
      --gpu -1
    ;;

  generate-data-smoke)
    run_python -m diffusha.data_collection.generate_data \
      --env-name "${ENV_NAME}" \
      --num-transitions 10 \
      --valid-return-threshold -100000 \
      --randp 0.0
    ;;

  train-diffusion-smoke)
    export DDPM_MODEL_PATH="${DDPM_MODEL_PATH:-${OUT_DIR}/ddpm-smoke}"
    run_python -m diffusha.diffusion.train \
      diffusha/config/sweep/sweep-lunarlander.jsonl \
      -l "$(sweep_line_for_env)" \
      --num-training-steps 2 \
      --save-every 1 \
      --eval-every 1000 \
      --batch-size 4
    ;;

  eval-smoke-model)
    export DDPM_MODEL_PATH="${DDPM_MODEL_PATH:-${OUT_DIR}/ddpm-smoke}"
    run_python -m diffusha.diffusion.evaluation.eval_assistance \
      --env-name "${ENV_NAME}" \
      --out-dir "${OUT_DIR}" \
      --model-step "${MODEL_STEP:-1}" \
      --num-episodes 1 \
      --num-envs 1
    ;;

  full-smoke)
    if [[ "${ENV_NAME}" != "LunarLander-v1" && "${ENV_NAME}" != "LunarLander-v5" ]]; then
      echo "full-smoke currently supports LunarLander-v1 or LunarLander-v5." >&2
      exit 1
    fi
    "${BASH_SOURCE[0]}" train-sac-smoke
    "${BASH_SOURCE[0]}" generate-data-smoke
    "${BASH_SOURCE[0]}" train-diffusion-smoke
    "${BASH_SOURCE[0]}" eval-smoke-model
    ;;

  ""|-h|--help|help)
    usage
    ;;

  *)
    echo "Unknown command: ${command}" >&2
    usage
    exit 1
    ;;
esac
