#!/bin/bash
# Single-H100 FURA (BlockTT, full-rank) on Llama-2-7B / CodeFeedback (python subset).
# Effective batch=128 via per_device_train_batch_size=4 * gradient_accumulation_steps=32.
# Bypasses deepspeed (not installed in the active uv env on Python 3.13).
set -euo pipefail
cd "$(dirname "$0")/../.."   # ref/PiSSA

PROJECT_DIR="${PROJECT_DIR:-/home/yequan/Project/lora/lora-without-regret}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-2-7b-hf}"
DATA_PATH="${DATA_PATH:-/data/yequan/pissa-dataset}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/yequan/pissa-runs/python-FURA-Llama-2-7b-singlegpu}"
DEVICE="${DEVICE:-4}"
LR="${LR:-2e-4}"

export HF_HOME="${HF_HOME:-/data/yequan/huggingface}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
mkdir -p "$OUTPUT_PATH"

CUDA_VISIBLE_DEVICES=${DEVICE} uv run --project ${PROJECT_DIR} python train.py \
    --model_name_or_path "$BASE_MODEL" \
    --full_finetune False \
    --fura True \
    --fura_trainable_type all \
    --fura_decomp_mode output_one_block \
    --fura_blocktt_rank full \
    --fura_train_position small \
    --fura_s_merged_to keep_trainable \
    --bf16 \
    --data_path "$DATA_PATH" \
    --sub_task python \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$OUTPUT_PATH" \
    --num_train_epochs 1 \
    --model_max_length 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate "$LR" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard"

# Eval pipeline (requires evalplus + human_eval, not installed in this env).
# Once installed, uncomment:
# uv run --project ${PROJECT_DIR} python utils/gen_vllm.py --model "$OUTPUT_PATH" --sub_task python --output_file "$OUTPUT_PATH/python_response.jsonl" --data_path "$DATA_PATH"
# uv run --project ${PROJECT_DIR} python utils/code_process.py --path "$OUTPUT_PATH/python_response.jsonl"
# uv run --project ${PROJECT_DIR} evalplus.evaluate --dataset humaneval --samples "$OUTPUT_PATH/humaneval.jsonl"
# uv run --project ${PROJECT_DIR} evalplus.evaluate --dataset mbpp     --samples "$OUTPUT_PATH/mbpp.jsonl"
