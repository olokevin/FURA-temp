#!/bin/bash
# fura/blocktt 1-epoch lr sweep on commonsense_170k, eval boolq+openbookqa only.
# Mirrors the existing 1ep run at lr=1e-4 (see /data/yequan/fura/lift/launch_logs/gpu7_sweep_blocktt_1ep.log).

set -u

PROJECT_DIR="/home/yequan/Project/lora/lora-without-regret"
LAUNCH_LOG="/data/yequan/fura/lift/launch_logs/gpu2_sweep_blocktt_1ep_lr_high.log"
OUTPUT_SRC_DIR="/data/yequan/fura/lift"
MODEL="meta-llama/Meta-Llama-3-8B"
SEED=43

mkdir -p "$(dirname "$LAUNCH_LOG")"

cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=2
export EVAL_DATASETS="boolq openbookqa"
export MODEL
export decomp_mode=output_one_block
export train_position=small
export s_merged_to=keep_trainable
export blocktt_rank=full
export trainable_type=all
export num_train_epochs=1
export seed="$SEED"

for lr in 4e-4 6e-4 8e-4; do
    run_name="blocktt-calib_none-lr_${lr}-decomp_output_one_block_pos_small_smerge_keep_trainable-1ep-seed_${SEED}"
    OUTPUT="${OUTPUT_SRC_DIR}/commonsense/${MODEL}/${run_name}"
    echo "=== [GPU2-sweep] blocktt lr=${lr} 1ep @ $(date) ===" >> "$LAUNCH_LOG"

    lr="$lr" OUTPUT="$OUTPUT" run_name="$run_name" \
        bash "$PROJECT_DIR/ref/LIFT/bash_scripts/finetune_commonsense_blocktt.sh" >> "$LAUNCH_LOG" 2>&1
done

echo "=== [GPU2-sweep] DONE @ $(date) ===" >> "$LAUNCH_LOG"
