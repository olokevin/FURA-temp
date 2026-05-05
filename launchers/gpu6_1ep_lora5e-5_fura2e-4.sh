#!/bin/bash
# Two 1-epoch commonsense_170k runs on GPU 6, eval = boolq + openbookqa only.
# 1) lora lr=5e-5 (rank 64, alpha 128, default 5-module targets)
# 2) fura/blocktt lr=2e-4 (project defaults: output_one_block / pos_small / smerge_keep_trainable / rank_full)

set -u

PROJECT_DIR="/home/yequan/Project/lora/lora-without-regret"
LAUNCH_LOG="/data/yequan/fura/lift/launch_logs/gpu6_1ep_lora5e-5_fura2e-4.log"
OUTPUT_SRC_DIR="/data/yequan/fura/lift"
MODEL="meta-llama/Meta-Llama-3-8B"
SEED=43

mkdir -p "$(dirname "$LAUNCH_LOG")"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=6
export EVAL_DATASETS="boolq openbookqa"
export MODEL
export num_train_epochs=1
export seed="$SEED"

# ---- run 1: lora lr=5e-5 ----
{
    echo "=== [GPU6] lora lr=5e-5 1ep @ $(date) ==="
    lr_lora=5e-5
    lora_r=64
    lora_alpha=128
    run_name_lora="lora-lr_${lr_lora}-rank_${lora_r}-1ep-seed_${SEED}"
    OUTPUT_lora="${OUTPUT_SRC_DIR}/commonsense/${MODEL}/${run_name_lora}"

    lr="$lr_lora" \
        adapter_name=lora \
        lora_r="$lora_r" \
        lora_alpha="$lora_alpha" \
        OUTPUT="$OUTPUT_lora" \
        run_name="$run_name_lora" \
        bash "$PROJECT_DIR/ref/LIFT/bash_scripts/finetune_commonsense_lora.sh"
} >> "$LAUNCH_LOG" 2>&1

# ---- run 2: fura/blocktt lr=2e-4 ----
{
    echo "=== [GPU6] blocktt lr=2e-4 1ep @ $(date) ==="
    lr_fura=2e-4
    run_name_fura="blocktt-calib_none-lr_${lr_fura}-decomp_output_one_block_pos_small_smerge_keep_trainable-1ep-seed_${SEED}"
    OUTPUT_fura="${OUTPUT_SRC_DIR}/commonsense/${MODEL}/${run_name_fura}"

    lr="$lr_fura" \
        decomp_mode=output_one_block \
        train_position=small \
        s_merged_to=keep_trainable \
        blocktt_rank=full \
        trainable_type=all \
        OUTPUT="$OUTPUT_fura" \
        run_name="$run_name_fura" \
        bash "$PROJECT_DIR/ref/LIFT/bash_scripts/finetune_commonsense_blocktt.sh"
} >> "$LAUNCH_LOG" 2>&1

echo "=== [GPU6] DONE @ $(date) ===" >> "$LAUNCH_LOG"
