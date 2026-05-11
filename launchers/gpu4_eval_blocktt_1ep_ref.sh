#!/bin/bash
# Eval the 6 missing commonsense tasks for the 1-epoch fura/blocktt reference
# run (lr=2e-4, default head/closest factorization). The existing partial run
# already has boolq + openbookqa; this fills in the rest.

set -u
PROJECT_DIR="/home/yequan/Project/lora/lora-without-regret"
LAUNCH_LOG="/data/yequan/fura/lift/launch_logs/shape_ablation/gpu4_eval_1ep_ref.log"
mkdir -p "$(dirname "$LAUNCH_LOG")"

MODEL="meta-llama/Meta-Llama-3-8B"
RUN_NAME="blocktt-calib_none-lr_2e-4-decomp_output_one_block_pos_small_smerge_keep_trainable-1ep-seed_43"
CKPT="/data/yequan/fura/lift/commonsense/${MODEL}/${RUN_NAME}/last"

export CUDA_VISIBLE_DEVICES=4
export EVAL_DATASETS="piqa social_i_qa ARC-Challenge ARC-Easy hellaswag winogrande"

echo "==== eval 1-ep reference (6 missing tasks) on GPU 4 @ $(date) ====" >> "$LAUNCH_LOG"
echo "    CKPT=${CKPT}" >> "$LAUNCH_LOG"
echo "    EVAL_DATASETS=${EVAL_DATASETS}" >> "$LAUNCH_LOG"

cd "${PROJECT_DIR}/ref/LIFT"

bash ./bash_scripts/eval_commonsense.sh \
    CKPT="$CKPT" \
    base_model="$MODEL" \
    wandb_project="commonsense-Meta-Llama-3-8B" \
    wandb_run_name="${RUN_NAME}-last-fillin" \
    wandb_run_id="" \
    >> "$LAUNCH_LOG" 2>&1

echo "==== done @ $(date) (rc=$?) ====" >> "$LAUNCH_LOG"
