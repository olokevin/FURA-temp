#!/bin/bash

pwd
hostname
date
echo starting job...
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME="${HF_HOME:-/data/yequan/huggingface}"

SRC_DIR="${SRC_DIR:-/home/yequan/Project/lora/lora-without-regret/ref/LIFT}"      # MODIFY THIS LINE
DATA_DIR="${DATA_DIR:-LLM-Adapters}"      # MODIFY THIS LINE
OUTPUT_SRC_DIR="${OUTPUT_SRC_DIR:-/data/yequan/fura/lift}"    # MODIFY THIS LINE

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
# Spectral Adapter (Zhang & Pilanci 2024): additive or rotation of top-r singular
# vectors. Default rank/LR align with PiSSA (rank 64, best commonsense LR 2e-5).
spectral_mode="${spectral_mode:-additive}"
spectral_rank="${spectral_rank:-64}"
trainable_type="${trainable_type:-pissa5}"
lr="${lr:-2e-5}"
seed="${seed:-43}"
num_train_epochs="${num_train_epochs:-1}"
MAX_STEPS="${MAX_STEPS:-0}"
model_tag="${MODEL##*/}"

wandb_project="${wandb_project:-commonsense-${model_tag}}"
wandb_run_id="${wandb_run_id:-$(python -c 'import wandb; print(wandb.util.generate_id())')}"
export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

echo $MODEL

OUTPUT="${OUTPUT:-${OUTPUT_SRC_DIR}/commonsense/${MODEL}/spectral-${spectral_mode}-lr_${lr}-rank_${spectral_rank}-type_${trainable_type}-ep${num_train_epochs}-seed_${seed}}"
run_name="${run_name:-$(basename "$OUTPUT")}"

mkdir -p $OUTPUT

cd ${SRC_DIR}

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_spectral.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --max_seq_len 2048 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs ${num_train_epochs} \
    --mixed_precision bf16 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type linear \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --gradient_checkpointing \
    --instruction_type single \
    --load_last_model \
    --spectral_mode ${spectral_mode} \
    --spectral_rank ${spectral_rank} \
    --trainable_type ${trainable_type} \
    --save_interval 100000 \
    --val_set_size 120 \
    --eval_step 400 \
    --data_path ${DATA_DIR}/ft-training_set/commonsense_170k.json \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    --max_steps ${MAX_STEPS} \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

if [ "${MAX_STEPS}" = "0" ]; then
    bash ./bash_scripts/eval_commonsense.sh \
        CKPT="$OUTPUT" \
        base_model="${MODEL}" \
        wandb_project="${wandb_project}" \
        wandb_run_name="${run_name}" \
        wandb_run_id="${wandb_run_id}"
fi
