#!/bin/bash
# qfura on Mixtral-8x7B / CodeFeedback-100K, single H100 (94 GB).
#
# Follows the PiSSA QPiSSA recipe (lr=2e-5, batch=1x128, seq=512, 1 epoch on
# ~100K examples) but uses qfura's BTT decomposition. Defaults match the
# project-wide qfura defaults (CLAUDE.md): rank=full, decomp=output_one_block,
# train_position=small, s_merged_to=keep_trainable. trainable_type=mixtral_all
# converts Mixtral attention (q/k/v/o_proj) + MoE experts (w1/w2/w3); the MoE
# router (block_sparse_moe.gate) is left frozen (not a BTT target).
#
# --load_strategy=layer_stream converts+quantizes one Linear at a time so the
# bf16 model never fully materialises on GPU.
#
# Override env vars: MAX_STEPS, lr, seed, per_device_train_batch_size,
# gradient_accumulation_steps, model_max_length, no_wandb.

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
DATA_PATH="${DATA_PATH:-/data/yequan/pissa-dataset/python/train.json}"
OUTPUT_SRC_DIR="${OUTPUT_SRC_DIR:-/data/yequan/fura/lift}"

MODEL="${MODEL:-mistralai/Mixtral-8x7B-v0.1}"
decomp_mode="${decomp_mode:-output_one_block}"
train_position="${train_position:-small}"
blocktt_rank="${blocktt_rank:-full}"
s_merged_to="${s_merged_to:-keep_trainable}"
trainable_type="${trainable_type:-mixtral_all}"
quant_block_layout="${quant_block_layout:-flat}"
load_strategy="${load_strategy:-layer_stream}"
prompt_style="${prompt_style:-pissa}"
trainable_param_dtype="${trainable_param_dtype:-bf16}"
optimizer_name="${optimizer_name:-paged_adamw_8bit}"

lr="${lr:-2e-5}"
seed="${seed:-42}"
MAX_STEPS="${MAX_STEPS:-0}"
num_train_epochs="${num_train_epochs:-1}"
model_max_length="${model_max_length:-512}"
per_device_train_batch_size="${per_device_train_batch_size:-1}"
gradient_accumulation_steps="${gradient_accumulation_steps:-128}"
model_tag="${MODEL##*/}"

wandb_project="${wandb_project:-qfura-code-${model_tag}}"
wandb_run_id="${wandb_run_id:-$(uv run --project ${PROJECT_DIR} python -c 'import wandb; print(wandb.util.generate_id())' 2>/dev/null)}"
no_wandb_flag=""
if [ "${no_wandb:-0}" = "1" ]; then
    no_wandb_flag="--no_wandb"
fi
export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

echo $MODEL

OUTPUT="${OUTPUT:-${OUTPUT_SRC_DIR}/codefeedback100k/${MODEL}/qfura-layout_${quant_block_layout}-decomp_${decomp_mode}_smerge_${s_merged_to}-bs${per_device_train_batch_size}x${gradient_accumulation_steps}-lr_${lr}-seed_${seed}}"
run_name="${run_name:-$(basename "$OUTPUT")}"

mkdir -p $OUTPUT

cd ${SRC_DIR}

uv run --project ${PROJECT_DIR} accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_qfura.py \
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
    --decomp_mode ${decomp_mode} \
    --train_position ${train_position} \
    --blocktt_rank ${blocktt_rank} \
    --s_merged_to ${s_merged_to} \
    --trainable_type ${trainable_type} \
    --quant_block_layout ${quant_block_layout} \
    --load_strategy ${load_strategy} \
    --prompt_style ${prompt_style} \
    --trainable_param_dtype ${trainable_param_dtype} \
    --optimizer ${optimizer_name} \
    --load_last_model \
    --max_steps ${MAX_STEPS} \
    --data_path ${DATA_PATH} \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    ${no_wandb_flag} \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log
