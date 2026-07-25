#!/bin/bash
# 1-epoch LIFT commonsense run with Muon, followed by the full 8-task eval.
#
# Naming follows the existing 1ep sweep convention (docs/exp_results/
# lift_commonsense.md -> "Naming conventions actually used") with an `-opt_muon`
# infix so Muon runs never collide with the AdamW ones:
#   full-opt_muon-lr_${lr}-1ep-seed_${seed}
#   blocktt-opt_muon-calib_none-lr_${lr}-decomp_output_one_block_pos_small_smerge_keep_trainable-1ep-seed_${seed}
#
# Usage:
#   GPU=6 METHOD=full lr=1e-3 MUON_LR_ADAM=1e-5 bash launchers/muon_1ep.sh
#   GPU=7 METHOD=fura lr=3e-3 MUON_LR_ADAM=2e-4 bash launchers/muon_1ep.sh

set -u

GPU="${GPU:?set GPU}"
METHOD="${METHOD:?set METHOD to full|fura}"
lr="${lr:?set lr}"
SEED="${SEED:-43}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
MUON_LR_ADAM="${MUON_LR_ADAM:-}"

SRC_DIR=/home/yequan/Project/lora/lora-without-regret/ref/LIFT
RUN_ROOT="/data/yequan/fura/lift/commonsense/${MODEL}"

case "${METHOD}" in
    full)
        SCRIPT=finetune_commonsense_full.sh
        NAME="full-opt_muon-lr_${lr}-1ep-seed_${SEED}"
        ;;
    fura)
        SCRIPT=finetune_commonsense_blocktt.sh
        # Muon on the BTT cores needs per-step core renormalization to stay
        # stable; tag it so those runs never get confused with un-normalized ones.
        norm_tag=""
        if [ "${BLOCKTT_NORMALIZE:-0}" = "1" ]; then norm_tag="-norm"; fi
        NAME="blocktt-opt_muon${norm_tag}-calib_none-lr_${lr}-decomp_output_one_block_pos_small_smerge_keep_trainable-1ep-seed_${SEED}"
        ;;
    *) echo "unknown METHOD=${METHOD}" >&2; exit 2 ;;
esac

OUT="${RUN_ROOT}/${NAME}"
mkdir -p "${OUT}"

echo "=== [gpu${GPU}] 1ep ${METHOD}/muon lr=${lr} lr_adam=${MUON_LR_ADAM:-<inherit>} -> ${OUT}  @ $(date) ==="

# eval_commonsense.sh defaults to WANDB_RESUME=must whenever a run id is passed,
# which hard-fails when training ran in an env without wandb (so the run was
# never created). Results still land in eval.log, but the noise is avoidable.
export WANDB_RESUME="${WANDB_RESUME:-allow}"

cd "${SRC_DIR}/bash_scripts"
CUDA_VISIBLE_DEVICES="${GPU}" \
OPTIMIZER=muon \
MUON_LR_ADAM="${MUON_LR_ADAM}" \
MAX_STEPS=0 \
num_train_epochs=1 \
lr="${lr}" \
seed="${SEED}" \
MODEL="${MODEL}" \
OUTPUT="${OUT}" \
run_name="${NAME}" \
wandb_project="commonsense-${MODEL##*/}" \
wandb_run_id="${NAME}" \
bash "${SCRIPT}"

echo "=== [gpu${GPU}] 1ep ${METHOD}/muon lr=${lr} DONE @ $(date) ==="
