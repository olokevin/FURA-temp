#!/bin/bash
# Train one FURA run at a given LR, then run eval_and_log.sh on the ckpt.
# Designed to be queued in series per GPU for an overnight LR sweep.
#
# Required env: LR, DEVICE
# Optional env:
#   RUN_NAME                  (default: fura-default-lr${LR})
#   PROJECT_DIR               (default: /home/yequan/Project/lora/lora-without-regret)
#   BASE_MODEL                (default: meta-llama/Llama-2-7b-hf)
#   DATA_PATH                 (default: /data/yequan/pissa-dataset)
#   CKPT_DIR                  (default: /data/yequan/pissa-runs/${RUN_NAME})
#   WANDB_PROJECT             (default: pissa-code-llama2-7b)
#   ARTIFACT_DIR              (default: /data/yequan/fura/pissa/code/${RUN_NAME})
#   DELETE_CKPT               (default: 1)  delete weights after eval
set -euo pipefail

: "${LR:?LR must be set}"
: "${DEVICE:?DEVICE must be set}"

PROJECT_DIR="${PROJECT_DIR:-/home/yequan/Project/lora/lora-without-regret}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-2-7b-hf}"
DATA_PATH="${DATA_PATH:-/data/yequan/pissa-dataset}"
WANDB_PROJECT="${WANDB_PROJECT:-pissa-code-llama2-7b}"
RUN_NAME="${RUN_NAME:-fura-default-lr${LR}}"
CKPT_DIR="${CKPT_DIR:-/data/yequan/pissa-runs/${RUN_NAME}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/data/yequan/fura/pissa/code/${RUN_NAME}}"
DELETE_CKPT="${DELETE_CKPT:-1}"

export HF_HOME="${HF_HOME:-/data/yequan/huggingface}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_NAME="${RUN_NAME}"

mkdir -p "${CKPT_DIR}" "${ARTIFACT_DIR}"
TRAIN_LOG="${CKPT_DIR}/training.log"
SWEEP_LOG="${ARTIFACT_DIR}/sweep.log"

echo "=== sweep run: ${RUN_NAME} (lr=${LR}, device=${DEVICE}) ===" | tee "${SWEEP_LOG}"
date | tee -a "${SWEEP_LOG}"

cd "${PROJECT_DIR}/ref/PiSSA"

echo "[train] starting ${RUN_NAME}..." | tee -a "${SWEEP_LOG}"
CUDA_VISIBLE_DEVICES=${DEVICE} uv run --project "${PROJECT_DIR}" python train.py \
    --model_name_or_path "${BASE_MODEL}" \
    --full_finetune False \
    --fura True \
    --fura_trainable_type all \
    --fura_decomp_mode output_one_block \
    --fura_blocktt_rank full \
    --fura_train_position small \
    --fura_s_merged_to keep_trainable \
    --bf16 \
    --data_path "${DATA_PATH}" \
    --sub_task python \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "${CKPT_DIR}" \
    --num_train_epochs 1 \
    --model_max_length 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate "${LR}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --lr_scheduler_type "cosine" \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    > "${TRAIN_LOG}" 2>&1
date | tee -a "${SWEEP_LOG}"
echo "[train] finished ${RUN_NAME}" | tee -a "${SWEEP_LOG}"

echo "[eval] starting ${RUN_NAME}..." | tee -a "${SWEEP_LOG}"
CKPT="${CKPT_DIR}" \
RUN_NAME="${RUN_NAME}" \
DEVICE="${DEVICE}" \
WANDB_PROJECT="${WANDB_PROJECT}" \
ARTIFACT_DIR="${ARTIFACT_DIR}" \
DATA_PATH="${DATA_PATH}" \
DELETE_CKPT="${DELETE_CKPT}" \
bash "${PROJECT_DIR}/ref/PiSSA/scripts/python_llama2_7b/eval_and_log.sh" \
    >> "${SWEEP_LOG}" 2>&1

date | tee -a "${SWEEP_LOG}"
echo "=== sweep run done: ${RUN_NAME} ===" | tee -a "${SWEEP_LOG}"
