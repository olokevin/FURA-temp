#!/bin/bash
# Sequential shape-ablation runs on GPU 6, 1 epoch each.
# Cell 2: unbalanced (b=8) for both attn and mlp.
# Cell 3: extreme n=din, b=1 for both attn and mlp.

set -u
PROJECT_DIR="/home/yequan/Project/lora/lora-without-regret"
LAUNCH_LOG="/data/yequan/fura/lift/launch_logs/shape_ablation/gpu6_seq_ep1.log"
mkdir -p "$(dirname "$LAUNCH_LOG")"

echo "==== sequential shape ablation (cells 2,3 / 1ep) start @ $(date) ====" >> "$LAUNCH_LOG"
for CELL in 2 3; do
    echo "---- launching CELL=${CELL} EPOCHS=1 @ $(date) ----" >> "$LAUNCH_LOG"
    GPU=6 CELL="${CELL}" SEED=43 LR=2e-4 EPOCHS=1 \
        bash "$PROJECT_DIR/launchers/shape_ablation_llama3_8b.sh" \
        >> "$LAUNCH_LOG" 2>&1
    echo "---- CELL=${CELL} done @ $(date) (rc=$?) ----" >> "$LAUNCH_LOG"
done
echo "==== sequential shape ablation finished @ $(date) ====" >> "$LAUNCH_LOG"
