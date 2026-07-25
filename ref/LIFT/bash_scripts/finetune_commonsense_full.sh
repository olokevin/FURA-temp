#!/bin/bash

pwd
hostname
date
echo starting job...
# conda activate lift
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME="${HF_HOME:-/data/yequan/huggingface}"

SRC_DIR=/home/yequan/Project/lora/lora-without-regret/ref/LIFT      # MODIFY THIS LINE
DATA_DIR=LLM-Adapters      # MODIFY THIS LINE
OUTPUT_SRC_DIR=/data/yequan/fura/lift    # MODIFY THIS LINE

# SLURM_ARRAY_TASK_ID=$1
# cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${SRC_DIR}/bash_scripts/slurm_config_full_commonsense.txt)
# MODEL=$(echo $cfg | cut -f 1 -d ' ')
# lr=$(echo $cfg | cut -f 2 -d ' ')
# seed=$(echo $cfg | cut -f 3 -d ' ')

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
lr="${lr:-2e-4}"
wd="${wd:-0.}"
seed="${seed:-43}"
model_tag="${MODEL##*/}"
MAX_STEPS="${MAX_STEPS:-0}"

# Tag output dir with wd only when non-zero, so the default (wd=0) runs keep their
# original path and the weight-decay sweep runs land in distinct directories.
wd_tag=""
case "${wd}" in
    0|0.|0.0|"") ;;
    *) wd_tag="-wd_${wd}" ;;
esac

# --- optimizer selection (adamw | muon). Muon updates decoder weight matrices
# --- with orthogonalized momentum and leaves embeddings/lm_head/norms on AdamW.
OPTIMIZER="${OPTIMIZER:-adamw}"
opt_args=( --optimizer "${OPTIMIZER}" )
opt_tag=""
_add_opt_arg() { if [ -n "$2" ]; then opt_args+=( "$1" "$2" ); fi; return 0; }
if [ "${OPTIMIZER}" != "adamw" ]; then
    opt_tag="-opt_${OPTIMIZER}"
    _add_opt_arg --muon_lr_adam "${MUON_LR_ADAM}"
    _add_opt_arg --muon_lr_embedding "${MUON_LR_EMBEDDING}"
    _add_opt_arg --muon_momentum "${MUON_MOMENTUM}"
    _add_opt_arg --muon_ns_steps "${MUON_NS_STEPS}"
    _add_opt_arg --muon_polar_method "${MUON_POLAR_METHOD}"
    _add_opt_arg --muon_structured_ortho_method "${MUON_STRUCTURED_ORTHO_METHOD}"
    _add_opt_arg --muon_norm_method "${MUON_NORM_METHOD}"
    _add_opt_arg --muon_adamw_betas "${MUON_ADAMW_BETAS}"
fi

wandb_project="${wandb_project:-commonsense-${model_tag}}"
wandb_run_id="${wandb_run_id:-$(python -c 'import wandb; print(wandb.util.generate_id())')}"
export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

echo $MODEL

peft_tuner=sparse

OUTPUT="${OUTPUT:-${OUTPUT_SRC_DIR}/commonsense/${MODEL}/full${opt_tag}-lr_${lr}${wd_tag}-seed_${seed}}"
run_name="${run_name:-$(basename "$OUTPUT")}"
mkdir -p $OUTPUT

cd ${SRC_DIR}

# --per_device_train_batch_size 1 \
# --gradient_accumulation_steps 16 \
# --per_device_eval_batch_size 1 \

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_sft.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 16 \
    --logging_steps 10 \
    --max_seq_len 2048 \
    --learning_rate ${lr} \
    --weight_decay ${wd} \
    --num_train_epochs ${num_train_epochs:-3} \
    --mixed_precision bf16 \
    --lr_scheduler_type ${LR_SCHEDULER:-linear} \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --gradient_checkpointing \
    --save_interval 100000 \
    --instruction_type single \
    --val_set_size 120 \
    --eval_step 400 \
    --load_last_model \
    --data_path ${DATA_DIR}/ft-training_set/commonsense_170k.json \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    --max_steps ${MAX_STEPS} \
    "${opt_args[@]}" \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

if [ "${MAX_STEPS}" = "0" ]; then
    bash ./bash_scripts/eval_commonsense.sh \
        CKPT="$OUTPUT" \
        base_model="${MODEL}" \
        wandb_project="${wandb_project}" \
        wandb_run_name="${run_name}" \
        wandb_run_id="${wandb_run_id}"
fi
