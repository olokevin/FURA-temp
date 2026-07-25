#!/bin/bash
# fura + Muon: stability-screen a candidate LR, then launch the 1-epoch run at
# the highest LR that survives.
#
# Why this exists: the 500-step linear-decay LR probe ranked fura+Muon 5e-4 best
# (tail loss 0.047), but the 1-epoch run at 5e-4 blew up at step 860
# (loss 0.033 -> 12.50) and settled into a damaged ~0.124 basin for the
# remaining 9k steps (BoolQ 62.2, PIQA 47.3 — near-random). Two things hid it:
#   1. the probe stopped at 500 steps, ~360 steps before the blow-up, and
#   2. the probe's compressed linear decay was already shrinking the LR by then.
# The screen below fixes both: constant LR (no decay to hide behind) held for
# SCREEN_STEPS, which is >2x the full-LR exposure that killed 5e-4.
#
# Usage: GPU=7 bash launchers/muon_fura_retry.sh

set -u

GPU="${GPU:-7}"
SEED="${SEED:-43}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
MUON_LR_ADAM="${MUON_LR_ADAM:-2e-4}"
SCREEN_STEPS="${SCREEN_STEPS:-1200}"
CANDIDATE="${CANDIDATE:-3e-4}"
FALLBACK="${FALLBACK:-2e-4}"
WAIT_FOR="${WAIT_FOR:-}"

REPO=/home/yequan/Project/lora/lora-without-regret
PROBE_ROOT="/data/yequan/fura/lift/commonsense/${MODEL}/_muon_lr_probe"

cd "${REPO}"

if [ -n "${WAIT_FOR}" ]; then
    echo "=== waiting for GPU${GPU} to free (marker in ${WAIT_FOR}) @ $(date) ==="
    until grep -q "DONE" "${WAIT_FOR}" 2>/dev/null; do sleep 60; done
fi
# Belt and braces: wait until the GPU has enough *free* memory for this run.
# This box is shared, so requiring a near-idle GPU deadlocks as soon as a
# co-tenant lands; a fura 8B run peaks around 55 GB, so require NEED_FREE_MIB.
NEED_FREE_MIB="${NEED_FREE_MIB:-65000}"
while :; do
    read -r total used < <(nvidia-smi --query-gpu=memory.total,memory.used \
        --format=csv,noheader,nounits -i "${GPU}" | tr -d ',')
    free=$(( total - used ))
    [ "${free}" -ge "${NEED_FREE_MIB}" ] && break
    echo "    GPU${GPU} free=${free}MiB < ${NEED_FREE_MIB}MiB, waiting @ $(date +%H:%M:%S)"
    sleep 60
done
echo "=== GPU${GPU} has ${free}MiB free @ $(date) ==="

# --- stability screen at CANDIDATE, constant LR ---
echo "=== stability screen: fura/muon lr=${CANDIDATE}, ${SCREEN_STEPS} steps @ constant LR ==="
GPU="${GPU}" METHOD=fura PROBE_OPT=muon PROBE_STEPS="${SCREEN_STEPS}" \
  PROBE_SCHED=constant_with_warmup MUON_LR_ADAM="${MUON_LR_ADAM}" \
  LRS="${CANDIDATE}" SEED="${SEED}" MODEL="${MODEL}" \
  bash launchers/muon_lr_probe.sh

SCREEN_LOG="${PROBE_ROOT}/fura-muon-lr_${CANDIDATE}-probe${SCREEN_STEPS}-constant_with_warmup-seed_${SEED}/training.log"

CHOSEN="${FALLBACK}"
if [ -s "${SCREEN_LOG}" ]; then
    # A blow-up shows up as a logged 10-step mean far above the healthy ~0.03-0.07
    # band. Ignore the first 150 steps (warmup transient).
    WORST=$(grep -oP '^  Step: \K[0-9]+, LR: [0-9.]+, Loss: [0-9.]+' "${SCREEN_LOG}" \
        | awk -F'[ ,]+' '$1 > 150 {print $NF}' | sort -g | tail -1)
    echo "=== screen worst post-150 logged loss at lr=${CANDIDATE}: ${WORST:-<none>} ==="
    if [ -n "${WORST}" ] && awk "BEGIN{exit !(${WORST} < 1.0)}"; then
        CHOSEN="${CANDIDATE}"
        echo "=== lr=${CANDIDATE} survived the screen -> using it for the 1-epoch run ==="
    else
        echo "=== lr=${CANDIDATE} destabilized -> falling back to lr=${FALLBACK} ==="
    fi
else
    echo "=== screen produced no log -> falling back to lr=${FALLBACK} ==="
fi

echo "=== launching 1-epoch fura/muon at lr=${CHOSEN} @ $(date) ==="
GPU="${GPU}" METHOD=fura lr="${CHOSEN}" MUON_LR_ADAM="${MUON_LR_ADAM}" SEED="${SEED}" MODEL="${MODEL}" \
  bash launchers/muon_1ep.sh
