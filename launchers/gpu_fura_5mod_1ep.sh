#!/bin/bash
# fura/blocktt 1-epoch commonsense_170k with 5-module set (q/k/v/up/down) — matches LoRA's default 5-mod target.
# Args (env): GPU=<n>, LR=<rate>; SEED defaults to 43.

set -u

PROJECT_DIR="/home/yequan/Project/lora/lora-without-regret"
OUTPUT_SRC_DIR="/data/yequan/fura/lift"
MODEL="meta-llama/Meta-Llama-3-8B"
SEED="${SEED:-43}"

GPU="${GPU:?must set GPU=<n>}"
LR="${LR:?must set LR=<rate>}"

LAUNCH_LOG="${OUTPUT_SRC_DIR}/launch_logs/gpu${GPU}_fura_5mod_1ep_lr${LR}.log"
mkdir -p "$(dirname "$LAUNCH_LOG")"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export MODEL
export decomp_mode=output_one_block
export train_position=small
export s_merged_to=keep_trainable
export blocktt_rank=full
export trainable_type=5mod_lora
export num_train_epochs=1
export seed="$SEED"
unset EVAL_DATASETS  # full 8-task eval

run_name="blocktt-calib_none-lr_${LR}-decomp_output_one_block_pos_small_smerge_keep_trainable-tgt_5mod_lora-1ep-seed_${SEED}"
OUTPUT="${OUTPUT_SRC_DIR}/commonsense/${MODEL}/${run_name}"

echo "=== [GPU${GPU}] fura 5mod_lora 1ep lr=${LR} seed=${SEED} @ $(date) ===" >> "$LAUNCH_LOG"

lr="$LR" OUTPUT="$OUTPUT" run_name="$run_name" \
    bash "$PROJECT_DIR/ref/LIFT/bash_scripts/finetune_commonsense_blocktt.sh" >> "$LAUNCH_LOG" 2>&1

echo "=== [GPU${GPU}] fura 5mod_lora DONE @ $(date) ===" >> "$LAUNCH_LOG"
