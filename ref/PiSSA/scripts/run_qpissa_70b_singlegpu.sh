#!/bin/bash
# QPiSSA on Llama-3-70B / MetaMathQA-100K, adapted for a single H100.
#
# Differences from the paper recipe (8 x A800-80GB + DeepSpeed ZeRO-2):
#   - Single H100 (94 GB), no DeepSpeed (use plain accelerate single-GPU).
#   - Pre-initialized residual model fetched from
#       fxmeng/PiSSA-Llama-3-70B-4bit-r64-5iter
#     (bypasses init_qpissa.py which OOMs at the bf16 70B load step).
#   - per_device_train_batch_size=1, gradient_accumulation_steps=128 -> effective batch=128 (matches paper).
#   - gradient_checkpointing required to fit 70B + r=64 LoRA + activations.
#   - MAX_STEPS=20 by default (calibration smoke test).
#
# Override env vars: MAX_STEPS, lr, model_max_length, lora_rank, sub_task_size.

set -euo pipefail

pwd; hostname; date

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export HF_HOME="${HF_HOME:-/data/yequan/huggingface}"

PROJECT_DIR="${PROJECT_DIR:-/home/yequan/Project/lora/lora-without-regret}"
PISSA_DIR="${PISSA_DIR:-${PROJECT_DIR}/ref/PiSSA}"
DATA_PATH="${DATA_PATH:-/data/yequan/pissa-dataset}"
OUTPUT_SRC_DIR="${OUTPUT_SRC_DIR:-/data/yequan/fura/pissa}"

# QPiSSA r=64 5-iter pre-initialized residual model.
RES_MODEL="${RES_MODEL:-fxmeng/PiSSA-Llama-3-70B-4bit-r64-5iter}"

lora_rank="${lora_rank:-64}"
lora_alpha="${lora_alpha:-${lora_rank}}"
lr="${lr:-2e-5}"
seed="${seed:-42}"
MAX_STEPS="${MAX_STEPS:-20}"
model_max_length="${model_max_length:-512}"
sub_task_size="${sub_task_size:-100000}"
per_device_train_batch_size="${per_device_train_batch_size:-1}"
gradient_accumulation_steps="${gradient_accumulation_steps:-128}"
report_to="${report_to:-tensorboard}"
wandb_project="${wandb_project:-}"
wandb_run_name="${wandb_run_name:-}"
if [ -n "$wandb_project" ]; then
    export WANDB_PROJECT="$wandb_project"
fi

OUTPUT="${OUTPUT:-${OUTPUT_SRC_DIR}/metamath100k-Llama-3-70B-qpissa-r${lora_rank}-5iter-singlegpu-maxsteps${MAX_STEPS}}"

mkdir -p "$OUTPUT"
cd "$PISSA_DIR"

uv run --project "$PROJECT_DIR" torchrun \
    --standalone \
    --nproc_per_node=1 \
    train.py \
    --model_name_or_path "$RES_MODEL" \
    --full_finetune False \
    --bf16 \
    --bits 4 \
    --double_quant True \
    --quant_type "nf4" \
    --adapter_name_or_path "pissa_init" \
    --gradient_checkpointing True \
    --data_path "$DATA_PATH" \
    --sub_task "metamath:${sub_task_size}" \
    --dataset_split "train" \
    --dataset_field instruction output \
    --output_dir "$OUTPUT" \
    --num_train_epochs 1 \
    --max_steps ${MAX_STEPS} \
    --model_max_length ${model_max_length} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --seed ${seed} \
    --report_to "${report_to}" \
    ${wandb_run_name:+--run_name "${wandb_run_name}"} \
    2> >(tee "$OUTPUT/err.log" >&2) | tee "$OUTPUT/training.log"

echo "[$(date)] training run exited."
