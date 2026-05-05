#!/bin/bash
# Stage 1: blocktt math LR sweep at bsz=16 for lr ∈ {4e-4, 6e-4, 8e-4}.
# Stage 2: pick best LR among ALL bsz=16 blocktt math runs (existing + stage-1)
#          using gsm8k acc, then launch bsz=64 (lr × 2) and bsz=256 (lr × 4).
set -uo pipefail
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export HF_HOME="${HF_HOME:-/data/yequan/huggingface/cache}"
export EVAL_DATASETS="gsm8k"
export MODEL="meta-llama/Meta-Llama-3-8B"
export decomp_mode=output_one_block
export s_merged_to=keep_trainable

M_BASE="/data/yequan/fura/lift/math/${MODEL}"
mkdir -p /data/yequan/fura/lift/launch_logs

cd /home/yequan/Project/lora/lora-without-regret

# --- helper to fetch a run's gsm8k Result given the run dir ---
get_gsm8k() {
    local run_dir="$1"
    for path in "${run_dir}/last/math/gsm8k/eval.log" "${run_dir}/math/gsm8k/eval.log"; do
        if [ -f "$path" ]; then
            local r
            r=$(grep -oE 'Result [0-9]+(\.[0-9]+)?' "$path" 2>/dev/null | tail -1 | awk '{print $2}')
            if [ -n "$r" ]; then echo "$r"; return 0; fi
        fi
    done
    echo ""
}

# Each candidate: "<lr>:<run_dir>". Use distinct dir names with -bsz16 suffix
# for the new runs so they never collide with prior bsz!=16 runs.
declare -A LR_TO_DIR=(
    ["8e-5"]="${M_BASE}/blocktt-calib_none-lr_8e-5-decomp_output_one_block_pos_small-rank_full-smerge_keep_trainable-type_all-seed_43"
    ["1e-4"]="${M_BASE}/blocktt-calib_none-lr_1e-4-decomp_output_one_block_pos_small-rank_full-smerge_keep_trainable-type_all-seed_43"
    ["2e-4"]="${M_BASE}/blocktt-calib_none-lr_2e-4-decomp_output_one_block_pos_small-rank_full-smerge_keep_trainable-type_all-seed_43"
    ["3e-4"]="${M_BASE}/blocktt-calib_none-lr_3e-4-decomp_output_one_block_pos_small-rank_full-smerge_keep_trainable-type_all-seed_43"
    ["4e-4"]="${M_BASE}/blocktt-calib_none-lr_4e-4-decomp_output_one_block_pos_small-rank_full-smerge_keep_trainable-type_all-bsz16-seed_43"
    ["6e-4"]="${M_BASE}/blocktt-calib_none-lr_6e-4-decomp_output_one_block_pos_small-rank_full-smerge_keep_trainable-type_all-bsz16-seed_43"
    ["8e-4"]="${M_BASE}/blocktt-calib_none-lr_8e-4-decomp_output_one_block_pos_small-rank_full-smerge_keep_trainable-type_all-bsz16-seed_43"
)

# ============ STAGE 1: launch bsz=16 lr in {4e-4, 6e-4, 8e-4} ============
for lr_val in 4e-4 6e-4 8e-4; do
    OUTPUT_DIR="${LR_TO_DIR[$lr_val]}"
    RUN_NAME="$(basename "$OUTPUT_DIR")"
    echo "=== [GPU3-stage1] blocktt bsz=16 lr=${lr_val} @ $(date) ==="
    echo "    OUTPUT=${OUTPUT_DIR}"
    OUTPUT="${OUTPUT_DIR}" run_name="${RUN_NAME}" lr="${lr_val}" \
        bash ref/LIFT/bash_scripts/finetune_math_blocktt.sh
done

echo "=== [GPU3-stage1] DONE @ $(date) ==="

# ============ STAGE 2: find best LR across ALL bsz=16 blocktt math runs ============
echo "=== [GPU3-stage2] picking best LR among bsz=16 blocktt math runs ==="
best_lr=""
best_acc=""
for lr_val in "${!LR_TO_DIR[@]}"; do
    run_dir="${LR_TO_DIR[$lr_val]}"
    acc=$(get_gsm8k "$run_dir")
    if [ -z "$acc" ]; then
        echo "    lr=${lr_val} -> NO_RESULT (run_dir=${run_dir})"
        continue
    fi
    echo "    lr=${lr_val} -> ${acc}"
    if [ -z "$best_acc" ] || awk -v a="$acc" -v b="$best_acc" 'BEGIN{exit !(a > b)}'; then
        best_acc="$acc"
        best_lr="$lr_val"
    fi
done

if [ -z "$best_lr" ]; then
    echo "ERROR: no valid bsz=16 results found; aborting stage 2" >&2
    exit 1
fi
echo "=== [GPU3-stage2] BEST: lr=${best_lr} -> gsm8k=${best_acc} ==="

# Convert best_lr (e.g. "3e-4") to a numeric value, scale, format back to scientific.
# √(64/16)=2, √(256/16)=4. 2*best_lr and 4*best_lr.
scale_lr() {
    local lr_str="$1"
    local factor="$2"
    python -c "
import sys
lr = float('${lr_str}')
factor = ${factor}
scaled = lr * factor
# Format: keep at most 1 decimal in mantissa, exponent integer.
# e.g. 6e-4 → '6e-4', 7.5e-4 → '7.5e-4'.
m, e = '{:.6e}'.format(scaled).split('e')
m = float(m)
e = int(e)
# Round mantissa to 2 sig figs to keep the dirname tidy.
m_str = ('{:.1f}'.format(m).rstrip('0').rstrip('.')) or '0'
print('{}e{}'.format(m_str, e))
"
}

LR_BSZ64=$(scale_lr "$best_lr" 2)
LR_BSZ256=$(scale_lr "$best_lr" 4)
echo "    derived: bsz=64 lr=${LR_BSZ64}, bsz=256 lr=${LR_BSZ256}"

# ============ STAGE 2a: bsz=64 ============
OUT_BSZ64="${M_BASE}/blocktt-calib_none-lr_${LR_BSZ64}-decomp_output_one_block_pos_small-rank_full-smerge_keep_trainable-type_all-bsz64-seed_43"
RUN_BSZ64="$(basename "$OUT_BSZ64")"
echo "=== [GPU3-stage2a] blocktt bsz=64 lr=${LR_BSZ64} (best_lr × 2) @ $(date) ==="
echo "    OUTPUT=${OUT_BSZ64}"
PER_DEVICE_TRAIN_BS=2 GRAD_ACC_STEPS=32 \
    OUTPUT="${OUT_BSZ64}" run_name="${RUN_BSZ64}" lr="${LR_BSZ64}" \
    bash ref/LIFT/bash_scripts/finetune_math_blocktt.sh

# ============ STAGE 2b: bsz=256 ============
OUT_BSZ256="${M_BASE}/blocktt-calib_none-lr_${LR_BSZ256}-decomp_output_one_block_pos_small-rank_full-smerge_keep_trainable-type_all-bsz256-seed_43"
RUN_BSZ256="$(basename "$OUT_BSZ256")"
echo "=== [GPU3-stage2b] blocktt bsz=256 lr=${LR_BSZ256} (best_lr × 4) @ $(date) ==="
echo "    OUTPUT=${OUT_BSZ256}"
PER_DEVICE_TRAIN_BS=2 GRAD_ACC_STEPS=128 \
    OUTPUT="${OUT_BSZ256}" run_name="${RUN_BSZ256}" lr="${LR_BSZ256}" \
    bash ref/LIFT/bash_scripts/finetune_math_blocktt.sh

echo "=== [GPU3] full chain DONE @ $(date) ==="
