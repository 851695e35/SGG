#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-SiT-B/2}"
ENC_TYPE="${ENC_TYPE:-dinov2-vit-b}"
DATA_DIR="${DATA_DIR:-./data/imagenet256}"
OUTPUT_DIR="${OUTPUT_DIR:-./exps}"
EXP_NAME="${EXP_NAME:-condcond-B-in256-proj0.0-weakSegmented-layer4-w0.6-ratio0.2-update4-low0.2-high1.0-alpha0}"
REPORT_TO="${REPORT_TO:-none}"
PROJECT_NAME="${PROJECT_NAME:-segmented-guidance}"

RESOLUTION="${RESOLUTION:-256}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-400000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-50000}"
SAMPLING_STEPS="${SAMPLING_STEPS:-1000}"

PROJ_COEFF="${PROJ_COEFF:-0.0}"
WEIGHTING="${WEIGHTING:-uniform}"
WEAK_TYPE="Segmented"
WEAK_LAYER="${WEAK_LAYER:-4}"
W_SCALE="${W_SCALE:-0.6}"
WEAK_ALPHA="${WEAK_ALPHA:-0.0}"
WEAK_LOSS_RATIO="${WEAK_LOSS_RATIO:-0.2}"
WEAK_UPDATE="${WEAK_UPDATE:-4}"
CONDITION="${CONDITION:-cond}"
GUIDANCE_LOW="${GUIDANCE_LOW:-0.2}"
GUIDANCE_HIGH="${GUIDANCE_HIGH:-0.8}"
VAE_MODEL="${VAE_MODEL:-stabilityai/sd-vae-ft-mse}"

accelerate launch "${ROOT_DIR}/train.py" \
  --report-to "${REPORT_TO}" \
  --model "${MODEL}" \
  --enc-type "${ENC_TYPE}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --exp-name "${EXP_NAME}" \
  --project-name "${PROJECT_NAME}" \
  --resolution "${RESOLUTION}" \
  --batch-size "${BATCH_SIZE}" \
  --max-train-steps "${MAX_TRAIN_STEPS}" \
  --checkpointing-steps "${CHECKPOINTING_STEPS}" \
  --sampling-steps "${SAMPLING_STEPS}" \
  --path-type "linear" \
  --prediction "v" \
  --proj-coeff "${PROJ_COEFF}" \
  --weighting "${WEIGHTING}" \
  --weak-type "${WEAK_TYPE}" \
  --weak-layer "${WEAK_LAYER}" \
  --w-scale "${W_SCALE}" \
  --weak-alpha "${WEAK_ALPHA}" \
  --weak-loss-ratio "${WEAK_LOSS_RATIO}" \
  --weak-update "${WEAK_UPDATE}" \
  --condition "${CONDITION}" \
  --guidance-low "${GUIDANCE_LOW}" \
  --guidance-high "${GUIDANCE_HIGH}" \
  --vae-model "${VAE_MODEL}" \
  "$@"
