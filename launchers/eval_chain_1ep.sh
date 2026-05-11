#!/bin/bash
# Run a chain of evals on one GPU, with a custom EVAL_DATASETS list.
# Args (env):
#   GPU=<n>                  - target GPU index
#   EVAL_DATASETS="..."      - space-separated tasks to run (passed through)
#   PHASE_TAG=<tag>          - logfile suffix (e.g. 5tasks, hellaswag)
#   CKPT_DIRS_FILE=<path>    - file with one run-dirname per line (relative to OUTPUT_SRC_DIR/commonsense/${MODEL}/)
#
# For lora dirs (basename starts with "lora-") uses eval_commonsense_lora.sh; otherwise eval_commonsense.sh.

set -u

PROJECT_DIR="/home/yequan/Project/lora/lora-without-regret"
SRC_DIR="${PROJECT_DIR}/ref/LIFT"
OUTPUT_SRC_DIR="/data/yequan/fura/lift"
MODEL="meta-llama/Meta-Llama-3-8B"
RUN_ROOT="${OUTPUT_SRC_DIR}/commonsense/${MODEL}"

GPU="${GPU:?must set GPU=<n>}"
PHASE_TAG="${PHASE_TAG:?must set PHASE_TAG}"
CKPT_DIRS_FILE="${CKPT_DIRS_FILE:?must set CKPT_DIRS_FILE}"
EVAL_DATASETS="${EVAL_DATASETS:?must set EVAL_DATASETS}"

LAUNCH_LOG="${OUTPUT_SRC_DIR}/launch_logs/gpu${GPU}_eval_${PHASE_TAG}.log"
mkdir -p "$(dirname "$LAUNCH_LOG")"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export EVAL_DATASETS
export PROJECT_DIR
export SRC_DIR

echo "=== [GPU${GPU}] phase=${PHASE_TAG} EVAL_DATASETS='${EVAL_DATASETS}' @ $(date) ===" >> "$LAUNCH_LOG"
echo "=== ckpt_list ===" >> "$LAUNCH_LOG"
cat "$CKPT_DIRS_FILE" >> "$LAUNCH_LOG"
echo "===" >> "$LAUNCH_LOG"

while IFS= read -r dirname; do
    [ -z "$dirname" ] && continue
    CKPT="${RUN_ROOT}/${dirname}"
    if [ ! -d "$CKPT" ]; then
        echo "[GPU${GPU}] SKIP missing dir: $CKPT" >> "$LAUNCH_LOG"
        continue
    fi
    echo "" >> "$LAUNCH_LOG"
    echo "--- [GPU${GPU}] eval phase=${PHASE_TAG} ckpt=${dirname} @ $(date) ---" >> "$LAUNCH_LOG"

    case "$dirname" in
        lora-*)
            # eval_commonsense_lora.sh has no last/ fallback; point it explicitly at last/
            # so eval outputs land in $CKPT/last/commonsense/ (matching boolq/obqa from training).
            LORA_CKPT="$CKPT/last"
            if [ ! -d "$LORA_CKPT" ]; then
                echo "[GPU${GPU}] SKIP lora ckpt missing last/: $LORA_CKPT" >> "$LAUNCH_LOG"
                continue
            fi
            bash "$SRC_DIR/bash_scripts/eval_commonsense_lora.sh" \
                CKPT="$LORA_CKPT" \
                adapter_name=lora \
                base_model="$MODEL" \
                wandb_project="commonsense-Meta-Llama-3-8B" \
                wandb_run_name="${dirname}-eval${PHASE_TAG}" \
                wandb_run_id="" >> "$LAUNCH_LOG" 2>&1
            ;;
        *)
            bash "$SRC_DIR/bash_scripts/eval_commonsense.sh" \
                CKPT="$CKPT" \
                base_model="$MODEL" \
                wandb_project="commonsense-Meta-Llama-3-8B" \
                wandb_run_name="${dirname}-eval${PHASE_TAG}" \
                wandb_run_id="" >> "$LAUNCH_LOG" 2>&1
            ;;
    esac
    echo "--- [GPU${GPU}] eval phase=${PHASE_TAG} ckpt=${dirname} done @ $(date) ---" >> "$LAUNCH_LOG"
done < "$CKPT_DIRS_FILE"

echo "" >> "$LAUNCH_LOG"
echo "=== [GPU${GPU}] phase=${PHASE_TAG} DONE @ $(date) ===" >> "$LAUNCH_LOG"
