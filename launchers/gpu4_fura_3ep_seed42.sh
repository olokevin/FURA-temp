#!/bin/bash
# fura/blocktt 3-epoch commonsense_170k, lr=2e-4, seed=42, full 8-task eval.
# Defaults: decomp=output_one_block, pos=small, smerge=keep_trainable, rank=full.

set -u

PROJECT_DIR="/home/yequan/Project/lora/lora-without-regret"
LAUNCH_LOG="/data/yequan/fura/lift/launch_logs/gpu4_fura_3ep_seed42.log"
OUTPUT_SRC_DIR="/data/yequan/fura/lift"
MODEL="meta-llama/Meta-Llama-3-8B"
SEED=42
LR=2e-4

mkdir -p "$(dirname "$LAUNCH_LOG")"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=4
export MODEL
export decomp_mode=output_one_block
export train_position=small
export s_merged_to=keep_trainable
export blocktt_rank=full
export trainable_type=all
export num_train_epochs=3
export seed="$SEED"
unset EVAL_DATASETS

run_name="blocktt-calib_none-lr_${LR}-decomp_output_one_block_pos_small_smerge_keep_trainable-seed_${SEED}"
OUTPUT="${OUTPUT_SRC_DIR}/commonsense/${MODEL}/${run_name}"

echo "=== [GPU4] fura 3ep lr=${LR} seed=${SEED} @ $(date) ===" >> "$LAUNCH_LOG"

lr="$LR" OUTPUT="$OUTPUT" run_name="$run_name" \
    bash "$PROJECT_DIR/ref/LIFT/bash_scripts/finetune_commonsense_blocktt.sh" >> "$LAUNCH_LOG" 2>&1

echo "=== [GPU4] DONE @ $(date) ===" >> "$LAUNCH_LOG"
