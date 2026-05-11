#!/bin/bash
# Full FT 1-epoch commonsense_170k, lr=3e-5, seed=43, eval = boolq + openbookqa only.

set -u

PROJECT_DIR="/home/yequan/Project/lora/lora-without-regret"
LAUNCH_LOG="/data/yequan/fura/lift/launch_logs/gpu5_full_1ep_lr3e-5.log"
OUTPUT_SRC_DIR="/data/yequan/fura/lift"
MODEL="meta-llama/Meta-Llama-3-8B"
SEED=43
LR=3e-5

mkdir -p "$(dirname "$LAUNCH_LOG")"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=5
export EVAL_DATASETS="boolq openbookqa"
export MODEL
export num_train_epochs=1
export seed="$SEED"

run_name="full-lr_${LR}-1ep-seed_${SEED}"
OUTPUT="${OUTPUT_SRC_DIR}/commonsense/${MODEL}/${run_name}"

echo "=== [GPU5] full lr=${LR} 1ep seed=${SEED} @ $(date) ===" >> "$LAUNCH_LOG"

lr="$LR" OUTPUT="$OUTPUT" run_name="$run_name" \
    bash "$PROJECT_DIR/ref/LIFT/bash_scripts/finetune_commonsense_full.sh" >> "$LAUNCH_LOG" 2>&1

echo "=== [GPU5] DONE @ $(date) ===" >> "$LAUNCH_LOG"
