#!/bin/bash
# Run vLLM gen + accuracy test on a qfura-trained Llama-3-70B checkpoint
# against MetaMathQA test set (5000 MATH + 1319 GSM8K).
#
# Required env vars: CKPT, OUTPUT_DIR
# Optional: BATCH_SIZE (default 200), MAX_TOKENS (default 1024)

set -e
pwd; hostname; date

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/data/yequan/huggingface}"

PROJECT_DIR="${PROJECT_DIR:-/home/yequan/Project/lora/lora-without-regret}"
PISSA_DIR="${PISSA_DIR:-${PROJECT_DIR}/ref/PiSSA}"
DATA_PATH="${DATA_PATH:-/data/yequan/pissa-dataset}"

CKPT="${CKPT:?must set CKPT}"
OUTPUT_DIR="${OUTPUT_DIR:-${CKPT}/eval}"
BATCH_SIZE="${BATCH_SIZE:-200}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
QUANT="${QUANT:-bitsandbytes}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

mkdir -p "$OUTPUT_DIR"
RESPONSE_FILE="$OUTPUT_DIR/metamath_response.jsonl"
ACC_FILE="$OUTPUT_DIR/metamath_acc.txt"

cd "$PISSA_DIR"

echo "[$(date)] generating responses with vLLM (quant=$QUANT)..."
uv run --project "$PROJECT_DIR" python utils/gen_vllm.py \
    --model "$CKPT" \
    --data_path "$DATA_PATH" \
    --sub_task metamath \
    --dataset_split test \
    --output_file "$RESPONSE_FILE" \
    --batch_size $BATCH_SIZE \
    --max_tokens $MAX_TOKENS \
    --temperature 0.0 \
    --top_p 1.0 \
    --quantization "$QUANT" \
    --gpu_memory_utilization $GPU_MEM_UTIL \
    --dtype bfloat16 \
    2> >(tee "$OUTPUT_DIR/gen_err.log" >&2) | tee "$OUTPUT_DIR/gen.log"

echo "[$(date)] computing accuracy..."
uv run --project "$PROJECT_DIR" python utils/test_acc.py \
    --input_file "$RESPONSE_FILE" \
    2> >(tee "$OUTPUT_DIR/acc_err.log" >&2) | tee "$ACC_FILE"

echo "[$(date)] done. results in $OUTPUT_DIR"
