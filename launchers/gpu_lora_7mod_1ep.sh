#!/bin/bash
# lora 1-epoch commonsense_170k with 7-module set (q/k/v/o/gate/up/down) + full 8-task eval.
# Args (env): GPU=<n>, LR=<rate>; SEED defaults to 43.

set -u

PROJECT_DIR="/home/yequan/Project/lora/lora-without-regret"
OUTPUT_SRC_DIR="/data/yequan/fura/lift"
MODEL="meta-llama/Meta-Llama-3-8B"
SEED="${SEED:-43}"

GPU="${GPU:?must set GPU=<n>}"
LR="${LR:?must set LR=<rate>}"

LAUNCH_LOG="${OUTPUT_SRC_DIR}/launch_logs/gpu${GPU}_lora_7mod_1ep_lr${LR}.log"
mkdir -p "$(dirname "$LAUNCH_LOG")"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export MODEL
export adapter_name=lora
export lora_r=64
export lora_alpha=128
export target_modules="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
export num_train_epochs=1
export seed="$SEED"
unset EVAL_DATASETS  # full 8-task eval

run_name="lora-lr_${LR}-rank_${lora_r}-tgt_7mod-1ep-seed_${SEED}"
OUTPUT="${OUTPUT_SRC_DIR}/commonsense/${MODEL}/${run_name}"

echo "=== [GPU${GPU}] lora 7mod 1ep lr=${LR} seed=${SEED} @ $(date) ===" >> "$LAUNCH_LOG"

lr="$LR" OUTPUT="$OUTPUT" run_name="$run_name" \
    bash "$PROJECT_DIR/ref/LIFT/bash_scripts/finetune_commonsense_lora.sh" >> "$LAUNCH_LOG" 2>&1

echo "=== [GPU${GPU}] lora 7mod DONE @ $(date) ===" >> "$LAUNCH_LOG"
