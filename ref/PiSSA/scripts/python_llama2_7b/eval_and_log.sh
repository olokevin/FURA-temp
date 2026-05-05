#!/bin/bash
# Eval a finetuned Llama-2-7B ckpt on HumanEval+MBPP, log to wandb, then delete model weights.
# Usage:
#   CKPT=/path/to/ckpt RUN_NAME=fura-lr2e-4 DEVICE=4 WANDB_PROJECT=pissa-code-llama2-7b \
#     bash eval_and_log.sh
#
# Required env: CKPT, RUN_NAME
# Optional env:
#   DEVICE             (default: 0)         CUDA device for vLLM
#   WANDB_PROJECT      (default: pissa-code-llama2-7b)
#   ARTIFACT_DIR       (default: /data/yequan/fura/pissa/code/<RUN_NAME>)
#   DATA_PATH          (default: /data/yequan/pissa-dataset)
#   DELETE_CKPT        (default: 1)         set to 0 to keep the model weights
#   PROJECT_DIR        (default: /home/yequan/Project/lora/lora-without-regret)
#   MAX_TOKENS         (default: 1024)
#   TEMPERATURE        (default: 0.0)
set -euo pipefail

: "${CKPT:?CKPT must be set}"
: "${RUN_NAME:?RUN_NAME must be set}"

PROJECT_DIR="${PROJECT_DIR:-/home/yequan/Project/lora/lora-without-regret}"
DATA_PATH="${DATA_PATH:-/data/yequan/pissa-dataset}"
WANDB_PROJECT="${WANDB_PROJECT:-pissa-code-llama2-7b}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/data/yequan/fura/pissa/code/${RUN_NAME}}"
DEVICE="${DEVICE:-0}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DELETE_CKPT="${DELETE_CKPT:-1}"

export HF_HOME="${HF_HOME:-/data/yequan/huggingface}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export WANDB_PROJECT="${WANDB_PROJECT}"

mkdir -p "${ARTIFACT_DIR}"
RESPONSE_FILE="${ARTIFACT_DIR}/python_response.jsonl"
HE_FILE="${ARTIFACT_DIR}/humaneval.jsonl"
MB_FILE="${ARTIFACT_DIR}/mbpp.jsonl"
EVAL_LOG="${ARTIFACT_DIR}/eval.log"
METRICS_FILE="${ARTIFACT_DIR}/metrics.json"

# Re-runs: stale output file confuses gen_vllm.py (it appends).
[ -f "${RESPONSE_FILE}" ] && rm -f "${RESPONSE_FILE}"
[ -f "${HE_FILE}" ] && rm -f "${HE_FILE}"
[ -f "${MB_FILE}" ] && rm -f "${MB_FILE}"

echo "=== Eval ${RUN_NAME} (ckpt=${CKPT}) ===" | tee "${EVAL_LOG}"
echo "device=${DEVICE} project=${WANDB_PROJECT} artifact_dir=${ARTIFACT_DIR}" | tee -a "${EVAL_LOG}"

cd "${PROJECT_DIR}/ref/PiSSA"

echo "[1/4] vLLM generation..." | tee -a "${EVAL_LOG}"
CUDA_VISIBLE_DEVICES=${DEVICE} uv run --project "${PROJECT_DIR}" python utils/gen_vllm.py \
    --model "${CKPT}" \
    --data_path "${DATA_PATH}" \
    --sub_task python \
    --dataset_split test \
    --output_file "${RESPONSE_FILE}" \
    --temperature "${TEMPERATURE}" \
    --max_tokens "${MAX_TOKENS}" 2>&1 | tee -a "${EVAL_LOG}"

echo "[2/4] Splitting humaneval / mbpp..." | tee -a "${EVAL_LOG}"
uv run --project "${PROJECT_DIR}" python utils/code_process.py --path "${RESPONSE_FILE}" 2>&1 | tee -a "${EVAL_LOG}"

echo "[3/4] evalplus pass@1 on HumanEval..." | tee -a "${EVAL_LOG}"
HE_OUT=$(uv run --project "${PROJECT_DIR}" evalplus.evaluate --dataset humaneval --samples "${HE_FILE}" 2>&1 | tee -a "${EVAL_LOG}")

echo "[3/4] evalplus pass@1 on MBPP..." | tee -a "${EVAL_LOG}"
MB_OUT=$(uv run --project "${PROJECT_DIR}" evalplus.evaluate --dataset mbpp --samples "${MB_FILE}" 2>&1 | tee -a "${EVAL_LOG}")

echo "[4/4] Parse + log to wandb..." | tee -a "${EVAL_LOG}"
uv run --project "${PROJECT_DIR}" python "${PROJECT_DIR}/ref/PiSSA/scripts/python_llama2_7b/log_eval_to_wandb.py" \
    --run_name "${RUN_NAME}" \
    --project "${WANDB_PROJECT}" \
    --eval_log "${EVAL_LOG}" \
    --metrics_out "${METRICS_FILE}" \
    --ckpt "${CKPT}" 2>&1 | tee -a "${EVAL_LOG}"

if [ "${DELETE_CKPT}" = "1" ]; then
    echo "[cleanup] Deleting model weights under ${CKPT} (keeping training.log/runs/eval/* in artifact dir)..." | tee -a "${EVAL_LOG}"
    # Be defensive: only delete if CKPT looks like a *-singlegpu run dir
    find "${CKPT}" -maxdepth 2 \( -name 'model-*.safetensors' -o -name 'model.safetensors' \
        -o -name 'optimizer.pt' -o -name 'scheduler.pt' -o -name 'rng_state.pth' \
        -o -name 'pytorch_model*.bin' -o -name 'adapter_model*.safetensors' \
        -o -name 'adapter_model*.bin' \) -print -delete | tee -a "${EVAL_LOG}"
    # Also drop any inner checkpoint-N/ subdir model weights
    find "${CKPT}" -maxdepth 3 -path '*/checkpoint-*/*.safetensors' -print -delete | tee -a "${EVAL_LOG}" || true
    find "${CKPT}" -maxdepth 3 -path '*/checkpoint-*/optimizer.pt' -print -delete | tee -a "${EVAL_LOG}" || true
fi

echo "=== Done. metrics -> ${METRICS_FILE} ===" | tee -a "${EVAL_LOG}"
cat "${METRICS_FILE}" 2>/dev/null || true
