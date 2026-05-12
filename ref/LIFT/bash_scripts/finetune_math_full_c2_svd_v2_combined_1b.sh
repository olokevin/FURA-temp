#!/bin/bash
# Group C2: Llama-3.2-3B -> ~1B via svd_llm_v2_combined, then full FT on MATH-10K, eval gsm8k.
# Launch with: DEVICE=N bash bash_scripts/finetune_math_full_c2_svd_v2_combined_1b.sh

pwd
hostname
date
echo starting job...
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME="${HF_HOME:-/data/ryan/huggingface_cache}"

if [ -n "${DEVICE:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$DEVICE"
fi

SRC_DIR="${SRC_DIR:-/home/yequan/Project/lora/lora-without-regret/ref/LIFT}"
DATA_DIR="${DATA_DIR:-/data/ruijiezhang/llm-adapter_bp/LLM-Adapters}"
OUTPUT_SRC_DIR="${OUTPUT_SRC_DIR:-/data/yequan/compress_train/sft}"

MODEL="${MODEL:-meta-llama/Llama-3.2-3B}"
lr="${lr:-1e-4}"
seed="${seed:-43}"
calib_mode="${calib_mode:-svd_v2_combined}"
compression_ratio="${compression_ratio:-0.250}"
target_tag="${target_tag:-1B}"

calib_source="${calib_source:-training_data}"
calib_num_seqs="${calib_num_seqs:-128}"
calib_max_length="${calib_max_length:-2048}"
calib_batch_size="${calib_batch_size:-4}"
calib_seed="${calib_seed:-3}"

model_tag="${MODEL##*/}"
wandb_project="${wandb_project:-compress_sft_math_llama3.2}"
wandb_run_id="${wandb_run_id:-$(python -c 'import wandb; print(wandb.util.generate_id())')}"

export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

echo $MODEL

OUTPUT="${OUTPUT:-${OUTPUT_SRC_DIR}/math/${MODEL}/full-calib_${calib_mode}-target_${target_tag}-lr_${lr}-seed_${seed}}"
run_name="${run_name:-$(basename "$OUTPUT")}"
mkdir -p $OUTPUT

cd ${SRC_DIR}

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_sft.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BS:-2} \
    --per_device_eval_batch_size 16 \
    --logging_steps 10 \
    --max_seq_len 2048 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --mixed_precision bf16 \
    --gradient_accumulation_steps ${GRAD_ACC_STEPS:-8} \
    --lr_scheduler_type linear \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --instruction_type single \
    ${GRAD_CKPT:+--gradient_checkpointing} \
    --load_last_model \
    --data_path ${DATA_DIR}/ft-training_set/math_10k.json \
    --calib_mode ${calib_mode} \
    --calib_source ${calib_source} \
    --calib_num_seqs ${calib_num_seqs} \
    --calib_max_length ${calib_max_length} \
    --calib_batch_size ${calib_batch_size} \
    --calib_seed ${calib_seed} \
    --compression_ratio ${compression_ratio} \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

EVAL_DATASETS="gsm8k" bash ./bash_scripts/eval_math.sh \
    CKPT="$OUTPUT/last" \
    base_model="${MODEL}" \
    wandb_project="${wandb_project}" \
    wandb_run_name="${run_name}" \
    wandb_run_id="${wandb_run_id}"
