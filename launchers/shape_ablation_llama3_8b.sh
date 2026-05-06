#!/bin/bash
# Shape ablation for fura/blocktt on Llama-3-8B (output_one_block, m=1, a=dout).
# Three cells differ only in input-side (n, b) factorization for each module.
#
# Llama-3-8B Linear input dims:
#   q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj : din = 4096
#   down_proj                                          : din = 14336
#
# Cell 1 (head-aligned attn, closest mlp):
#   q/k/v/o     -> (n=32,  b=128)  (head-aligned)
#   gate/up     -> closest pair    -> (n=64,  b=64)
#   down        -> closest pair    -> (n=112, b=128)
#
# Cell 2 (unbalanced for both attn and mlp):
#   q/k/v/o, gate/up  -> (n=512,  b=8)   (~8x more imbalanced than closest)
#   down              -> (n=1792, b=8)   (uses divisor 1792 of 14336)
#
# Cell 3 (extreme n=din, b=1):
#   q/k/v/o, gate/up  -> (n=4096,  b=1)
#   down              -> (n=14336, b=1)

set -u

PROJECT_DIR="/home/yequan/Project/lora/lora-without-regret"
LAUNCH_LOG_DIR="/data/yequan/fura/lift/launch_logs/shape_ablation"
OUTPUT_SRC_DIR="/data/yequan/fura/lift"
MODEL="meta-llama/Meta-Llama-3-8B"
SEED="${SEED:-43}"
LR="${LR:-2e-4}"
GPU="${GPU:-0}"
CELL="${CELL:-1}"
EPOCHS="${EPOCHS:-3}"

mkdir -p "$LAUNCH_LOG_DIR"
cd "$PROJECT_DIR"

case "$CELL" in
    1)
        SPEC='{"qkv":"head","o":"head","mlp_upgate":"closest","mlp_down":"closest"}'
        TAG="cell1_head_attn-closest_mlp"
        ;;
    2)
        SPEC='{"qkv":"512,8","o":"512,8","mlp_upgate":"512,8","mlp_down":"1792,8"}'
        TAG="cell2_unbalanced_8col"
        ;;
    3)
        SPEC='{"qkv":"4096,1","o":"4096,1","mlp_upgate":"4096,1","mlp_down":"14336,1"}'
        TAG="cell3_extreme_b1"
        ;;
    *)
        echo "CELL must be 1, 2, or 3 (got '$CELL')" >&2
        exit 1
        ;;
esac

LAUNCH_LOG="${LAUNCH_LOG_DIR}/gpu${GPU}_cell${CELL}_${TAG}_seed${SEED}.log"

export CUDA_VISIBLE_DEVICES="$GPU"
export MODEL
export decomp_mode=output_one_block
export train_position=small
export s_merged_to=keep_trainable
export blocktt_rank=full
export trainable_type=all
export num_train_epochs="$EPOCHS"
export seed="$SEED"
export blocktt_input_factorization="$SPEC"
unset EVAL_DATASETS

run_name="blocktt-shape_${TAG}-ep${EPOCHS}-lr_${LR}-decomp_output_one_block_pos_small_smerge_keep_trainable-seed_${SEED}"
OUTPUT="${OUTPUT_SRC_DIR}/commonsense/${MODEL}/${run_name}"

echo "=== [GPU${GPU}] shape ablation cell=${CELL} (${TAG}) lr=${LR} seed=${SEED} @ $(date) ===" >> "$LAUNCH_LOG"
echo "    blocktt_input_factorization=${SPEC}" >> "$LAUNCH_LOG"

lr="$LR" OUTPUT="$OUTPUT" run_name="$run_name" \
    bash "$PROJECT_DIR/ref/LIFT/bash_scripts/finetune_commonsense_blocktt.sh" >> "$LAUNCH_LOG" 2>&1

echo "=== [GPU${GPU}] DONE cell=${CELL} @ $(date) ===" >> "$LAUNCH_LOG"
