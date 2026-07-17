#!/bin/bash
# Unified MATH-10K full-FT launcher with optional SVD/BTT calibrated
# compression. Subsumes finetune_math_full_{a1,a2,b1,b2,c1,c2}_*.sh.
#
# Picks the right Python entrypoint by calib_mode family:
#   calib_mode=none                                  -> src/finetune_sft.py     (vanilla full FT)
#   calib_mode in {svd_v2, svd_v2_combined}          -> src/finetune_sft.py     (Task 6 SVD plumbing)
#   calib_mode in {v2, v2_bp, v2_combined, twosteps} -> src/finetune_blocktt.py (BTT plumbing)
#
# Typical invocations:
#   # 1) Baseline 1B full FT, no compression:
#   MODEL=meta-llama/Llama-3.2-1B DEVICE=0 bash bash_scripts/finetune_math_calib.sh
#
#   # 2) Llama-3.2-3B -> ~2B via svd_v2_combined, training_data calibration, lr=3e-5:
#   MODEL=meta-llama/Llama-3.2-3B DEVICE=1 \
#       calib_mode=svd_v2_combined compression_ratio=0.589 target_tag=2B \
#       calib_source=training_data lr=3e-5 \
#       bash bash_scripts/finetune_math_calib.sh
#
#   # 3) Llama-3.2-3B BTT calibrated v2_combined, fura defaults:
#   MODEL=meta-llama/Llama-3.2-3B DEVICE=2 \
#       calib_mode=v2_combined blocktt_rank=full \
#       calib_source=training_data lr=2e-4 \
#       bash bash_scripts/finetune_math_calib.sh

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

# --- core training knobs ---
MODEL="${MODEL:-meta-llama/Llama-3.2-3B}"
lr="${lr:-1e-4}"
seed="${seed:-43}"
num_train_epochs="${num_train_epochs:-3}"
PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-2}"
PER_DEVICE_EVAL_BS="${PER_DEVICE_EVAL_BS:-16}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-single}"

# --- calibration knobs (passed to both SVD and BTT entrypoints) ---
calib_mode="${calib_mode:-none}"
calib_source="${calib_source:-training_data}"
calib_num_seqs="${calib_num_seqs:-128}"
calib_max_length="${calib_max_length:-2048}"
calib_batch_size="${calib_batch_size:-4}"
calib_seed="${calib_seed:-3}"
calib_traces_path="${calib_traces_path:-}"

# --- SVD-only knob ---
compression_ratio="${compression_ratio:-1.0}"  # ignored unless calib_mode starts with svd_

# --- BTT-only knobs (defaults match the project-wide fura defaults; see CLAUDE.md) ---
blocktt_rank="${blocktt_rank:-full}"
decomp_mode="${decomp_mode:-output_one_block}"
train_position="${train_position:-small}"
s_merged_to="${s_merged_to:-keep_trainable}"
trainable_type="${trainable_type:-all}"
blocktt_input_factorization="${blocktt_input_factorization:-}"

# --- naming/output ---
target_tag="${target_tag:-}"  # optional; only used in SVD default OUTPUT path
model_tag="${MODEL##*/}"
wandb_project="${wandb_project:-compress_sft_math_llama3.2}"
wandb_run_id="${wandb_run_id:-$(python -c 'import wandb; print(wandb.util.generate_id())')}"
export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

# --- pick entrypoint and default OUTPUT path by calib_mode family ---
case "$calib_mode" in
    none)
        ENTRY="src/finetune_sft.py"
        DEFAULT_OUTPUT="${OUTPUT_SRC_DIR}/math/${MODEL}/full-lr_${lr}-seed_${seed}"
        FAMILY="none"
        ;;
    svd_v2|svd_v2_combined)
        ENTRY="src/finetune_sft.py"
        tag_part=""
        [ -n "$target_tag" ] && tag_part="-target_${target_tag}"
        DEFAULT_OUTPUT="${OUTPUT_SRC_DIR}/math/${MODEL}/full-calib_${calib_mode}-calsrc_${calib_source}${tag_part}-lr_${lr}-seed_${seed}"
        FAMILY="svd"
        ;;
    v2|v2_bp|v2_combined|twosteps)
        ENTRY="src/finetune_blocktt.py"
        DEFAULT_OUTPUT="${OUTPUT_SRC_DIR}/math/${MODEL}/blocktt-calib_${calib_mode}-calsrc_${calib_source}-lr_${lr}-decomp_${decomp_mode}_pos_${train_position}_smerge_${s_merged_to}-rank_${blocktt_rank}-seed_${seed}"
        FAMILY="btt"
        ;;
    *)
        echo "[finetune_math_calib] ERROR: unknown calib_mode='${calib_mode}'." >&2
        echo "  Must be one of: none, svd_v2, svd_v2_combined, v2, v2_bp, v2_combined, twosteps" >&2
        exit 2
        ;;
esac

OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}"
run_name="${run_name:-$(basename "$OUTPUT")}"
mkdir -p "$OUTPUT"

echo "[finetune_math_calib] MODEL=${MODEL} calib_mode=${calib_mode} family=${FAMILY} lr=${lr}"
echo "[finetune_math_calib] OUTPUT=${OUTPUT}"
echo "[finetune_math_calib] ENTRY=${ENTRY}"

cd "${SRC_DIR}"

# --- assemble flags ---
common_flags=(
    --model_name_or_path "${MODEL}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BS}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BS}"
    --logging_steps 10
    --max_seq_len "${MAX_SEQ_LEN}"
    --learning_rate "${lr}"
    --weight_decay "${WEIGHT_DECAY}"
    --num_train_epochs "${num_train_epochs}"
    --mixed_precision bf16
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
    --num_warmup_steps "${NUM_WARMUP_STEPS}"
    --seed "${seed}"
    --instruction_type "${INSTRUCTION_TYPE}"
    --load_last_model
    --data_path "${DATA_DIR}/ft-training_set/math_10k.json"
    --wandb_project "${wandb_project}"
    --wandb_run_name "${run_name}"
    --output_dir "${OUTPUT}"
)
[ -n "${GRAD_CKPT:-}" ] && common_flags+=( --gradient_checkpointing )

# Calibration flags: only when calib_mode != none. Both entrypoints register
# them via add_calibrated_btt_args(hyphen_style=False).
calib_flags=()
if [ "$calib_mode" != "none" ]; then
    calib_flags+=(
        --calib_mode "${calib_mode}"
        --calib_source "${calib_source}"
        --calib_num_seqs "${calib_num_seqs}"
        --calib_max_length "${calib_max_length}"
        --calib_batch_size "${calib_batch_size}"
        --calib_seed "${calib_seed}"
    )
    [ -n "${calib_traces_path}" ] && calib_flags+=( --calib_traces_path "${calib_traces_path}" )
fi

family_flags=()
case "$FAMILY" in
    svd)
        family_flags+=( --compression_ratio "${compression_ratio}" )
        ;;
    btt)
        family_flags+=(
            --decomp_mode "${decomp_mode}"
            --train_position "${train_position}"
            --blocktt_rank "${blocktt_rank}"
            --s_merged_to "${s_merged_to}"
            --trainable_type "${trainable_type}"
        )
        [ -n "${blocktt_input_factorization}" ] && \
            family_flags+=( --blocktt_input_factorization "${blocktt_input_factorization}" )
        ;;
esac

# --- launch training ---
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    "${ENTRY}" \
    "${common_flags[@]}" \
    "${calib_flags[@]}" \
    "${family_flags[@]}" \
    2> >(tee "${OUTPUT}/err.log" >&2) | tee "${OUTPUT}/training.log"

# --- eval on gsm8k only (matches existing per-group scripts) ---
EVAL_DATASETS="${EVAL_DATASETS:-gsm8k}" bash ./bash_scripts/eval_math.sh \
    CKPT="${OUTPUT}/last" \
    base_model="${MODEL}" \
    wandb_project="${wandb_project}" \
    wandb_run_name="${run_name}" \
    wandb_run_id="${wandb_run_id}"
