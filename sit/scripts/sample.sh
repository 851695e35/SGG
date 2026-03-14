#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIT_DIR="${ROOT_DIR}/sit"

# Core settings
NGPU="${NGPU:-4}"
SEED="${SEED:-0}"
MODEL="${MODEL:-SiT-B/2}"
ENCODER_DEPTH="${ENCODER_DEPTH:-4}"
PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
WEAK_TYPE="${WEAK_TYPE:-Segmented}"
WEAK_LAYER="${WEAK_LAYER:-4}"
CONDITION="${CONDITION:-cond}"
EVAL_GUIDANCE_HIGH="${EVAL_GUIDANCE_HIGH:-1.0}"

# Sampling settings
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-64}"
SAMPLE_MODE="${SAMPLE_MODE:-sde}"
SAMPLE_STEPS="${SAMPLE_STEPS:-250}"
CFG_SCALE="${CFG_SCALE:-1.0}"

# Paths
OUTPUT_DIR="${OUTPUT_DIR:-./exps}"
EXP_NAME="${EXP_NAME:-${CONDITION}-B-in256-weak${WEAK_TYPE}-layer${WEAK_LAYER}}"
RUN_DIR="${OUTPUT_DIR}/${EXP_NAME}"
CKPT_DIR="${CKPT_DIR:-${RUN_DIR}/checkpoints}"
SAMPLE_DIR="${SAMPLE_DIR:-${RUN_DIR}/samples}"
CKPT_STEP="${CKPT_STEP:-latest}" # latest or explicit 7-digit step (e.g. 0400000)
SIT_VAE_MODEL="${SIT_VAE_MODEL:-stabilityai/sd-vae-ft-mse}"
CKPT_PATH="${CKPT_PATH:-}"

[[ -d "${SIT_DIR}" ]] || { echo "Missing sit dir: ${SIT_DIR}"; exit 1; }
mkdir -p "${SAMPLE_DIR}"

if [[ -z "${CKPT_PATH}" ]]; then
  if [[ "${CKPT_STEP}" == "latest" ]]; then
    mapfile -t ckpts < <(compgen -G "${CKPT_DIR}/*.pt" | sort)
    [[ ${#ckpts[@]} -gt 0 ]] || { echo "No checkpoints found in ${CKPT_DIR}"; exit 1; }
    CKPT_PATH="${ckpts[${#ckpts[@]}-1]}"
  else
    CKPT_PATH="${CKPT_DIR}/${CKPT_STEP}.pt"
  fi
fi

[[ -f "${CKPT_PATH}" ]] || { echo "Checkpoint not found: ${CKPT_PATH}"; exit 1; }

echo "[sample] generating samples from ${CKPT_PATH}"
(
  cd "${SIT_DIR}"
  torchrun --nnodes=1 --nproc_per_node="${NGPU}" generate.py \
    --model "${MODEL}" \
    --num-fid-samples "${NUM_FID_SAMPLES}" \
    --ckpt "${CKPT_PATH}" \
    --path-type "linear" \
    --encoder-depth "${ENCODER_DEPTH}" \
    --projector-embed-dims "${PROJECTOR_EMBED_DIMS}" \
    --per-proc-batch-size "${PER_PROC_BATCH_SIZE}" \
    --mode "${SAMPLE_MODE}" \
    --num-steps "${SAMPLE_STEPS}" \
    --cfg-scale "${CFG_SCALE}" \
    --guidance-high "${EVAL_GUIDANCE_HIGH}" \
    --weak-type "${WEAK_TYPE}" \
    --weak-layer "${WEAK_LAYER}" \
    --sample-dir "${SAMPLE_DIR}" \
    --condition "${CONDITION}" \
    --global-seed "${SEED}" \
    --vae-model "${SIT_VAE_MODEL}"
)

