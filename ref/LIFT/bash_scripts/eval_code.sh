#!/bin/bash
# HumanEval eval for CodeFeedback-fine-tuned checkpoints (qlora or qfura),
# wrapping the vendored PiSSA pipeline: gen_vllm -> code_process -> evalplus.
#
# Usage:
#   bash bash_scripts/eval_code.sh CKPT=<checkpoint_dir> [base_model=<hf_model>]
#
# Checkpoint layouts (auto-detected, override with EVAL_PREFER=best):
#   qfura:  <CKPT>/last/            -> dense HF model, has config.json (load directly)
#   qlora:  <CKPT>/last_adapter/    -> PEFT adapter, has adapter_config.json (merge first)
# If <CKPT> itself has config.json it is used as-is; if it has
# adapter_config.json it is treated as an adapter.
#
# base_model is REQUIRED for qlora (adapter) checkpoints (the NF4 base to merge
# onto); for qfura dense checkpoints it is ignored.

set -euo pipefail

CKPT=""
base_model="${base_model:-mistralai/Mixtral-8x7B-v0.1}"

usage() {
    echo "Usage: bash bash_scripts/eval_code.sh CKPT=<checkpoint_dir> [base_model=<hf_model>]"
}

for arg in "$@"; do
    case "$arg" in
        CKPT=*|ckpt=*) CKPT="${arg#*=}" ;;
        base_model=*)  base_model="${arg#*=}" ;;
        *) echo "Unknown arg: $arg" >&2; usage; exit 1 ;;
    esac
done

if [ -z "$CKPT" ]; then usage; exit 1; fi

export HF_HOME="${HF_HOME:-/data/yequan/huggingface}"
PROJECT_DIR="${PROJECT_DIR:-/home/yequan/Project/lora/lora-without-regret}"
PISSA_UTILS="${PROJECT_DIR}/ref/PiSSA/utils"
DATA_ROOT="${DATA_ROOT:-/data/yequan/pissa-dataset}"

# --- Resolve checkpoint layout: dense (config.json) vs adapter (adapter_config.json) ---
resolve_dir() {
    local base="$1"
    if [ -f "${base}/config.json" ]; then echo "dense:${base}"; return; fi
    if [ -f "${base}/adapter_config.json" ]; then echo "adapter:${base}"; return; fi
    if [ "${EVAL_PREFER:-last}" = "best" ] && [ -f "${base}/best/config.json" ]; then echo "dense:${base}/best"; return; fi
    if [ -f "${base}/last/config.json" ]; then echo "dense:${base}/last"; return; fi
    if [ -f "${base}/last_adapter/adapter_config.json" ]; then echo "adapter:${base}/last_adapter"; return; fi
    if [ -f "${base}/best/config.json" ]; then echo "dense:${base}/best"; return; fi
    if [ -f "${base}/best_adapter/adapter_config.json" ]; then echo "adapter:${base}/best_adapter"; return; fi
    echo "none:"; return
}

RESOLVED="$(resolve_dir "$CKPT")"
KIND="${RESOLVED%%:*}"
MODEL_DIR="${RESOLVED#*:}"

if [ "$KIND" = "none" ]; then
    echo "Error: no dense checkpoint (config.json) or adapter (adapter_config.json) found under ${CKPT}" >&2
    exit 1
fi
echo "[eval_code] CKPT=${CKPT} -> kind=${KIND} dir=${MODEL_DIR}"

# --- If adapter, merge onto base_model to produce a dense model vLLM can load ---
if [ "$KIND" = "adapter" ]; then
    MERGED_DIR="${MODEL_DIR}/merged"
    if [ ! -f "${MERGED_DIR}/config.json" ]; then
        echo "[eval_code] merging adapter ${MODEL_DIR} onto ${base_model} -> ${MERGED_DIR}"
        uv run --project "$PROJECT_DIR" python "${PISSA_UTILS}/merge_adapter.py" \
            --base_model "$base_model" \
            --adapter "$MODEL_DIR" \
            --output_path "$MERGED_DIR"
    fi
    EVAL_MODEL="$MERGED_DIR"
else
    # QFURA dense checkpoints (materialized via save_hf_format) contain only
    # config.json + pytorch_model.bin — no tokenizer. gen_vllm.py passes only
    # --model to vLLM (no separate tokenizer arg), so vLLM would fail to load a
    # tokenizer from the checkpoint dir. Stage the base model's tokenizer files
    # into the checkpoint dir once if absent. (The adapter path above gets its
    # tokenizer from merge_adapter.py.)
    if [ ! -f "${MODEL_DIR}/tokenizer_config.json" ] && [ ! -f "${MODEL_DIR}/tokenizer.model" ] && [ ! -f "${MODEL_DIR}/tokenizer.json" ]; then
        echo "[eval_code] dense checkpoint has no tokenizer; staging tokenizer from ${base_model} into ${MODEL_DIR}"
        uv run --project "$PROJECT_DIR" python -c "
from transformers import AutoTokenizer
import sys
tok = AutoTokenizer.from_pretrained(sys.argv[1])
tok.save_pretrained(sys.argv[2])
" "$base_model" "$MODEL_DIR"
    fi
    EVAL_MODEL="$MODEL_DIR"
fi

OUT_DIR="${MODEL_DIR}/humaneval"
mkdir -p "$OUT_DIR"
RESP="${OUT_DIR}/python_response.jsonl"

cd "$PROJECT_DIR"

# Re-runs: gen_vllm.py appends to its output file, and a stale humaneval.jsonl
# would be scored instead of this run's. Remove both before regenerating.
[ -f "$RESP" ] && rm -f "$RESP"
[ -f "${OUT_DIR}/humaneval.jsonl" ] && rm -f "${OUT_DIR}/humaneval.jsonl"

# --- 1. Generate completions on the HumanEval (python) test split via vLLM ---
# The eval model is a full-precision (bf16) dense Mixtral-8x7B (~87 GiB) — a
# merged qlora checkpoint or a materialized qfura checkpoint. In bf16 it does NOT
# leave room for a KV cache on a single 95 GiB H100 (even at util 0.95 the KV
# budget is negative), and it is what the sweep runs one-GPU-per-chain. Load it
# 4-bit via bitsandbytes: weights drop to ~25 GiB, leaving ~50+ GiB for the KV
# cache. This is also faithful to how both methods were trained — qlora on an NF4
# base and qfura with an NF4-quantized frozen core — so 4-bit eval matches the
# trained numerics better than bf16 would. Cap max_model_len (HumanEval prompts +
# 1024-token gen are short). All three are overridable via env; set GEN_QUANT=""
# to eval in bf16 (needs >1 GPU via CUDA_VISIBLE_DEVICES for tensor parallelism).
GEN_QUANT="${GEN_QUANT:-bitsandbytes}"
GEN_GPU_MEM_UTIL="${GEN_GPU_MEM_UTIL:-0.9}"
GEN_MAX_MODEL_LEN="${GEN_MAX_MODEL_LEN:-2048}"
# Force a uniform compute dtype. A qfura checkpoint trained with
# --trainable_param_dtype fp32 materializes MIXED-dtype dense weights (fp32
# trainable-side layers, bf16 elsewhere); loading that with bitsandbytes crashes
# in a triton matmul ("Both operands must be same dtype. Got bf16 and fp32").
# Casting everything to bfloat16 on load makes it uniform. (bf16-trained
# checkpoints are unaffected.) Overridable via GEN_DTYPE.
GEN_DTYPE="${GEN_DTYPE:-bfloat16}"
gen_quant_flag=()
if [ -n "$GEN_QUANT" ]; then
    gen_quant_flag=(--quantization "$GEN_QUANT")
fi
uv run --project "$PROJECT_DIR" python "${PISSA_UTILS}/gen_vllm.py" \
    --model "$EVAL_MODEL" \
    --data_path "$DATA_ROOT" \
    --sub_task python \
    --dataset_split test \
    --output_file "$RESP" \
    --temperature 0.0 \
    --max_tokens 1024 \
    --gpu_memory_utilization "$GEN_GPU_MEM_UTIL" \
    --max_model_len "$GEN_MAX_MODEL_LEN" \
    --dtype "$GEN_DTYPE" \
    "${gen_quant_flag[@]}" \
    2> >(tee "${OUT_DIR}/eval_err.log" >&2) | tee "${OUT_DIR}/eval.log"

# --- 2. Post-process into evalplus humaneval.jsonl (written next to $RESP) ---
uv run --project "$PROJECT_DIR" python "${PISSA_UTILS}/code_process.py" \
    --path "$RESP" | tee -a "${OUT_DIR}/eval.log"

# --- 3. Score with evalplus (standard HumanEval; HumanEval+ also printed) ---
uv run --project "$PROJECT_DIR" python -m evalplus.evaluate \
    --dataset humaneval \
    --samples "${OUT_DIR}/humaneval.jsonl" | tee -a "${OUT_DIR}/eval.log"

echo "[eval_code] done. Results in ${OUT_DIR}/eval.log"
