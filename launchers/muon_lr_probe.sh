#!/bin/bash
# Short-horizon LR probe for LIFT commonsense (used to pick the Muon LR).
#
# Runs a self-contained mini-schedule (linear decay over ${PROBE_STEPS} optimizer
# steps, 3% warmup) at each LR in ${LRS} and reports the tail train loss, so the
# 1-epoch run only has to be launched once at the winning LR. All LRs see the
# same data in the same order (same seed), so tail train loss is a fair ranking.
#
# Usage:
#   GPU=6 METHOD=full LRS="3e-3 1e-3 3e-4 1e-4" MUON_LR_ADAM=1e-5 \
#     bash launchers/muon_lr_probe.sh
#   GPU=6 METHOD=full PROBE_OPT=adamw LRS="1e-5" bash launchers/muon_lr_probe.sh
#
# METHOD is `full` (finetune_commonsense_full.sh) or `fura`
# (finetune_commonsense_blocktt.sh).

set -u

GPU="${GPU:?set GPU}"
METHOD="${METHOD:?set METHOD to full|fura}"
LRS="${LRS:?set LRS to a space-separated list}"
PROBE_OPT="${PROBE_OPT:-muon}"
PROBE_STEPS="${PROBE_STEPS:-500}"
# `linear` gives a self-contained mini-schedule for ranking LRs.
# `constant_with_warmup` is the *stability* screen: the LR is never decayed, so a
# late blow-up cannot be masked by the probe's compressed decay. Use it with a
# longer PROBE_STEPS before committing a 1-epoch run.
PROBE_SCHED="${PROBE_SCHED:-linear}"
SEED="${SEED:-43}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
MUON_LR_ADAM="${MUON_LR_ADAM:-}"
KEEP_CKPT="${KEEP_CKPT:-0}"

SRC_DIR=/home/yequan/Project/lora/lora-without-regret/ref/LIFT
PROBE_ROOT="${PROBE_ROOT:-/data/yequan/fura/lift/commonsense/${MODEL}/_muon_lr_probe}"
mkdir -p "${PROBE_ROOT}"

case "${METHOD}" in
    full) SCRIPT=finetune_commonsense_full.sh ;;
    fura) SCRIPT=finetune_commonsense_blocktt.sh ;;
    *) echo "unknown METHOD=${METHOD}" >&2; exit 2 ;;
esac

# The training loop returns at step == MAX_STEPS *before* the logging block, so
# the last line written is at PROBE_STEPS - logging_steps (10).
LAST_LOGGED=$((PROBE_STEPS - 10))

echo "=== LR probe: method=${METHOD} opt=${PROBE_OPT} gpu=${GPU} steps=${PROBE_STEPS} lrs=${LRS} lr_adam=${MUON_LR_ADAM:-<inherit>} ==="

for lr in ${LRS}; do
    sched_tag=""
    if [ "${PROBE_SCHED}" != "linear" ]; then sched_tag="-${PROBE_SCHED}"; fi
    OUT="${PROBE_ROOT}/${METHOD}-${PROBE_OPT}-lr_${lr}-probe${PROBE_STEPS}${sched_tag}-seed_${SEED}"
    if [ -s "${OUT}/training.log" ] && grep -q "Step: ${LAST_LOGGED}," "${OUT}/training.log"; then
        echo "--- [gpu${GPU}] skip ${METHOD}/${PROBE_OPT} lr=${lr} (already probed)"
        continue
    fi
    mkdir -p "${OUT}"
    echo "--- [gpu${GPU}] probe ${METHOD}/${PROBE_OPT} lr=${lr} -> ${OUT}  @ $(date +%H:%M:%S)"
    ( cd "${SRC_DIR}/bash_scripts" && \
      CUDA_VISIBLE_DEVICES="${GPU}" \
      OPTIMIZER="${PROBE_OPT}" \
      MUON_LR_ADAM="${MUON_LR_ADAM}" \
      MAX_STEPS="${PROBE_STEPS}" \
      LR_SCHEDULER="${PROBE_SCHED}" \
      num_train_epochs=1 \
      lr="${lr}" \
      seed="${SEED}" \
      MODEL="${MODEL}" \
      OUTPUT="${OUT}" \
      run_name="$(basename "${OUT}")" \
      wandb_project="commonsense-muon-lrprobe" \
      wandb_run_id="probe_${METHOD}_${PROBE_OPT}_${lr}" \
      bash "${SCRIPT}" ) >/dev/null 2>&1
    rc=$?
    echo "--- [gpu${GPU}] done ${METHOD}/${PROBE_OPT} lr=${lr} rc=${rc}  @ $(date +%H:%M:%S)"
    grep -E "^  Step: " "${OUT}/training.log" | tail -3
    if [ "${KEEP_CKPT}" != "1" ]; then
        rm -rf "${OUT}/last" "${OUT}/best"
    fi
done

echo "=== LR probe done: method=${METHOD} opt=${PROBE_OPT} gpu=${GPU} @ $(date) ==="
