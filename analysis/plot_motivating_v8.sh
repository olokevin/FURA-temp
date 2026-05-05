#!/usr/bin/env bash
# Plot the motivating figures (motivating_svdft.png and motivating_fura.png)
# from the LIFT math projE / blocktt / lora checkpoints.
#
# Outputs:
#   docs/exp_results/figs/motivating_svdft.png  (+ .pdf)
#   docs/exp_results/figs/motivating_fura.png   (+ .pdf)
#
# Caching:
#   Panel (d) effective-rank curves and panel (e) BlockTT |R'-R0| arrays are
#   precomputed once and saved under analysis_results/motivating_v8_cache/.
#   Subsequent runs load from cache (~seconds) unless `recompute` is passed.
#   The cache key is derived from checkpoint dir names, the panel-d module list,
#   and panel-e (layer, module, decomp_mode), so editing CONFIG paths or those
#   panel knobs in the .py invalidates the relevant cache automatically.
#
# Usage:
#   bash analysis/plot_motivating_v8.sh                 # both figures, use cache if present
#   bash analysis/plot_motivating_v8.sh full            # same as above
#   bash analysis/plot_motivating_v8.sh recompute       # force recompute panels d/e
#   bash analysis/plot_motivating_v8.sh dry             # skip d/e entirely (fastest, bar/curve iteration)
#   bash analysis/plot_motivating_v8.sh svdft           # only SVD-FT figure
#   bash analysis/plot_motivating_v8.sh fura            # only FuRA figure (loads d/e from cache)
#   CUDA_VISIBLE_DEVICES=1 bash analysis/plot_motivating_v8.sh

set -euo pipefail
cd "$(dirname "$0")/.."

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

MODE="${1:-full}"
SVDFT="docs/exp_results/figs/motivating_svdft.png"
FURA="docs/exp_results/figs/motivating_fura.png"

case "$MODE" in
  full)
    uv run python analysis/plot_motivating_v8.py \
        --output-svdft "$SVDFT" --output-fura "$FURA"
    ;;
  notitles)
    # Suppress all subplot titles + panel-(a) annotations.
    # Outputs go to motivating_{svdft,fura}_notitles.png.
    uv run python analysis/plot_motivating_v8.py --no-titles \
        --output-svdft "$SVDFT" --output-fura "$FURA"
    ;;
  recompute)
    uv run python analysis/plot_motivating_v8.py \
        --recompute \
        --output-svdft "$SVDFT" --output-fura "$FURA"
    ;;
  dry)
    uv run python analysis/plot_motivating_v8.py \
        --skip-d --skip-e \
        --output-svdft "$SVDFT" --output-fura "$FURA"
    ;;
  svdft)
    uv run python analysis/plot_motivating_v8.py \
        --skip-fura \
        --output-svdft "$SVDFT"
    ;;
  fura)
    uv run python analysis/plot_motivating_v8.py \
        --skip-svdft \
        --output-fura "$FURA"
    ;;
  *)
    echo "Unknown mode: $MODE  (use full | recompute | dry | svdft | fura)" >&2
    exit 2
    ;;
esac
