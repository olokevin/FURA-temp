#!/bin/bash
# Re-run eval (boolq + obqa) for one full-FT 1ep checkpoint that already finished training.
# Args: GPU=<n> LR=<rate>

set -u

PROJECT_DIR="/home/yequan/Project/lora/lora-without-regret"
SRC_DIR="${PROJECT_DIR}/ref/LIFT"
OUTPUT_SRC_DIR="/data/yequan/fura/lift"
MODEL="meta-llama/Meta-Llama-3-8B"
SEED=43

GPU="${GPU:?must set GPU=<n>}"
LR="${LR:?must set LR=<rate>}"

run_name="full-lr_${LR}-1ep-seed_${SEED}"
CKPT="${OUTPUT_SRC_DIR}/commonsense/${MODEL}/${run_name}"
LAUNCH_LOG="${OUTPUT_SRC_DIR}/launch_logs/gpu${GPU}_eval_full_1ep_lr${LR}.log"

mkdir -p "$(dirname "$LAUNCH_LOG")"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export EVAL_DATASETS="boolq openbookqa"
export PROJECT_DIR
export SRC_DIR

echo "=== [GPU${GPU}] eval full lr=${LR} 1ep seed=${SEED} @ $(date) ===" >> "$LAUNCH_LOG"

bash "$SRC_DIR/bash_scripts/eval_commonsense.sh" \
    CKPT="$CKPT" \
    base_model="$MODEL" \
    wandb_project="commonsense-Meta-Llama-3-8B" \
    wandb_run_name="${run_name}-evalretry" \
    wandb_run_id="" >> "$LAUNCH_LOG" 2>&1

echo "=== [GPU${GPU}] eval DONE @ $(date) ===" >> "$LAUNCH_LOG"
