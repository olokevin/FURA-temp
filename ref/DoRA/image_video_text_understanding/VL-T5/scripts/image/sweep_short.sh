#!/usr/bin/env bash
# Short-run LR sweep on image multitask (BUTD features, 1 epoch each).
# 6 runs total: 1 DoRA @ paper LR + 5 fura @ different LRs.
# Packed across GPUs 4 and 5 (3 runs per GPU, concurrent). Each run gets a
# unique master_port and a unique run_name; all log to wandb project
# "fura-vl-bart-image" so they appear together for direct comparison.
#
# Usage: bash scripts/image/sweep_short.sh
# Output: snap/VLBart_multitask/<run_name>/ + log files in snap/.../sweep_logs/
#
# After completion, inspect wandb for which fura LR achieves the lowest train
# loss (and decent VQA/GQA val), then run scripts/image/sweep_full.sh to
# launch the full 10-epoch follow-up on the two winners.

set -uo pipefail
cd "$(dirname "$0")/../.."   # cd to VL-T5/

PROJECT=fura-vl-bart-image
EPOCHS=1
BATCH_SIZE=300
SAVE_STEPS=0   # disable mid-training ckpts during the short sweep

LOGDIR=snap/sweep_logs
mkdir -p ${LOGDIR}

# (gpu, port, script, lr, tag)
declare -a RUNS=(
  "4 26500 dora_butd 1e-3 dora_lr1e-3"
  "4 26501 fura      1e-3 fura_lr1e-3"
  "4 26502 fura      3e-3 fura_lr3e-3"
  "5 26503 fura      5e-3 fura_lr5e-3"
  "5 26504 fura      7e-4 fura_lr7e-4"
  "5 26505 fura      1e-2 fura_lr1e-2"
)

PIDS=()
for spec in "${RUNS[@]}"; do
    read -r gpu port script lr tag <<< "${spec}"
    run_name="sweep_${tag}_e${EPOCHS}"
    log="${LOGDIR}/${run_name}.log"
    echo "[launch] GPU=${gpu} port=${port} script=${script}.sh lr=${lr} -> ${run_name}"
    GPU=${gpu} \
    LR=${lr} \
    EPOCHS=${EPOCHS} \
    BATCH_SIZE=${BATCH_SIZE} \
    SAVE_STEPS=${SAVE_STEPS} \
    PROJECT=${PROJECT} \
    RUN_NAME=${run_name} \
    MASTER_PORT=${port} \
    bash scripts/image/${script}.sh 1 > "${log}" 2>&1 &
    PIDS+=($!)
    sleep 5  # stagger launches so wandb / DDP rendezvous don't collide
done

echo "[sweep] launched ${#PIDS[@]} runs, PIDs: ${PIDS[*]}"
echo "[sweep] waiting for all to finish... (logs in ${LOGDIR}/)"

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAIL=$((FAIL+1))
        echo "[sweep] PID ${pid} exited nonzero"
    fi
done

if [ ${FAIL} -gt 0 ]; then
    echo "[sweep] DONE with ${FAIL} failures. Check ${LOGDIR}/*.log"
    exit 1
fi
echo "[sweep] DONE. All 6 runs completed."
