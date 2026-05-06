#!/bin/bash
# qlora on Llama-3-70B / MetaMathQA-100K, single H100 (94 GB).
#
# Mirrors the QPiSSA paper recipe (lr=2e-5, batch=128 effective, seq=512,
# 1 epoch on 100K examples) but uses NF4 base + LoRA adapters (rank 64) on top.
# Defaults align with the project-wide fura/qfura comparison: PiSSA prompt,
# fp32 adapter, fp32 AdamW so head-to-head qlora vs qfura is apples-to-apples.
#
# 70B in NF4 ≈ 40 GB on GPU; the bf16 model never has to fully materialise.
#
# Override env vars: MAX_STEPS, lr, lora_r, lora_alpha, seed,
# per_device_train_batch_size, gradient_accumulation_steps, model_max_length.

pwd
hostname
date
echo starting job...
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME="${HF_HOME:-/data/yequan/huggingface}"

PROJECT_DIR="${PROJECT_DIR:-/home/yequan/Project/lora/lora-without-regret}"
SRC_DIR="${SRC_DIR:-${PROJECT_DIR}/ref/LIFT}"
DATA_PATH="${DATA_PATH:-/data/yequan/pissa-dataset/metamath_100k.json}"
OUTPUT_SRC_DIR="${OUTPUT_SRC_DIR:-/data/yequan/fura/lift}"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-70B}"
lora_r="${lora_r:-64}"
lora_alpha="${lora_alpha:-64}"
lora_dropout="${lora_dropout:-0.0}"
prompt_style="${prompt_style:-pissa}"
trainable_param_dtype="${trainable_param_dtype:-fp32}"
optimizer_name="${optimizer_name:-adamw}"

lr="${lr:-2e-5}"
seed="${seed:-42}"
MAX_STEPS="${MAX_STEPS:-0}"
num_train_epochs="${num_train_epochs:-1}"
model_max_length="${model_max_length:-512}"
per_device_train_batch_size="${per_device_train_batch_size:-1}"
gradient_accumulation_steps="${gradient_accumulation_steps:-128}"
model_tag="${MODEL##*/}"

wandb_project="${wandb_project:-qlora-metamath-${model_tag}}"
wandb_run_id="${wandb_run_id:-$(uv run --project ${PROJECT_DIR} python -c 'import wandb; print(wandb.util.generate_id())' 2>/dev/null)}"
no_wandb_flag=""
if [ "${no_wandb:-0}" = "1" ]; then
    no_wandb_flag="--no_wandb"
fi
export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

echo $MODEL

OUTPUT="${OUTPUT:-${OUTPUT_SRC_DIR}/metamath100k/${MODEL}/qlora-r_${lora_r}-alpha_${lora_alpha}-bs${per_device_train_batch_size}x${gradient_accumulation_steps}-lr_${lr}-seed_${seed}}"
run_name="${run_name:-$(basename "$OUTPUT")}"

mkdir -p $OUTPUT

cd ${SRC_DIR}

uv run --project ${PROJECT_DIR} accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_qlora.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size 1 \
    --logging_steps 1 \
    --max_seq_len ${model_max_length} \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs ${num_train_epochs} \
    --mixed_precision bf16 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --gradient_checkpointing \
    --instruction_type single \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --prompt_style ${prompt_style} \
    --trainable_param_dtype ${trainable_param_dtype} \
    --optimizer ${optimizer_name} \
    --load_last_model \
    --max_steps ${MAX_STEPS} \
    --val_set_size 0 \
    --data_path ${DATA_PATH} \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    ${no_wandb_flag} \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log
