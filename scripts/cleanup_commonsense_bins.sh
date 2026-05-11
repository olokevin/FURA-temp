#!/usr/bin/env bash
# Cleanup *.bin checkpoints under
#   /data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/
# keeping the runs referenced in docs/exp_results/lift_commonsense.md:
#
# Headline / per-method-best set (7 runs):
#   - blocktt 3ep default recipe, seeds 42 / 43 / 44
#       (output_one_block + pos_small + smerge_keep_trainable + lr_2e-4)
#   - milora   lr_1e-4 rank_64   seed 43
#   - randlora lr_1e-4 rank_64   seed 43
#   - pissa    lr_2e-5 rank_64   seed 43
#   - full     lr_1e-5           seed 43
#
# BlockTT Ablation (LLaMA-3-8B, no calibration) — 7 rows (Full FT and the
# headline L**SR** seed-43 run overlap with the headline set above, so this
# adds 5 additional dirs):
#   - FuRA (full) L S R    : blocktt-calib_none-lr_1e-5 + output_one_block + pos_both + smerge_keep_trainable + seed_43
#   - FuRA (PEFT) (LS) R   : blocktt-lr_2e-4 + output_one_block + pos_small + rank_full + smerge_frozen + seed_43
#   - FuRA (PEFT) L (SR)   : blocktt-lr_2e-4 + output_one_block + pos_small + smerge_trainable + seed_43
#   - FuRA (PEFT) LS R     : blocktt-calib_none-lr_2e-4 + input_one_block + pos_small + smerge_keep_trainable + seed_43
#   - FuRA (PEFT) L (SR)   : blocktt-lr_2e-4 + input_one_block + pos_small + rank_full + smerge_frozen + seed_43
#   - FuRA (PEFT) (LS) R   : blocktt-lr_2e-4 + input_one_block + pos_small + smerge_trainable + seed_43
#
# Additional keeps (per user request):
#   - qfura lr_2e-4 (Avg 87.30; output_one_block + smerge_keep_trainable + seed_43)
#
# Note: qdora-r_64-alpha_128-lr_1e-4-seed_43 and qlora-r_64-alpha_128-lr_1e-4-seed_43
#   were also requested to be kept, but their checkpoint shards are .safetensors,
#   not .bin — this script only touches .bin files, so they are unaffected.
#
# Usage:
#   bash scripts/cleanup_commonsense_bins.sh           # dry-run: print KEEP / DELETE lists
#   bash scripts/cleanup_commonsense_bins.sh --apply   # actually rm the DELETE list

set -euo pipefail

ROOT=/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B

KEEP_DIRS=(
  # Headline / per-method-best set
  "blocktt-calib_none-lr_2e-4-decomp_output_one_block_pos_small_smerge_keep_trainable-seed_42"
  "blocktt-lr_2e-4-decomp_output_one_block_pos_small_smerge_keep_trainable-seed_43"
  "blocktt-calib_none-lr_2e-4-decomp_output_one_block_pos_small_smerge_keep_trainable-seed_44"
  "milora-lr_1e-4-rank_64-seed_43"
  "randlora-lr_1e-4-rank_64-seed_43"
  "pissa-lr_2e-5-rank_64-seed_43"
  "full-lr_1e-5-seed_43"

  # BlockTT Ablation extras (5 dirs; Full FT + L**SR** seed-43 overlap headline)
  "blocktt-calib_none-lr_1e-5-decomp_output_one_block_pos_both_smerge_keep_trainable-seed_43"
  "blocktt-lr_2e-4-decomp_output_one_block_pos_small_rank_full_smerge_frozen-seed_43"
  "blocktt-lr_2e-4-decomp_output_one_block_pos_small_smerge_trainable-seed_43"
  "blocktt-calib_none-lr_2e-4-decomp_input_one_block_pos_small_smerge_keep_trainable-seed_43"
  "blocktt-lr_2e-4-decomp_input_one_block_pos_small_rank_full_smerge_frozen-seed_43"
  "blocktt-lr_2e-4-decomp_input_one_block_pos_small_smerge_trainable-seed_43"

  # qfura best-of-method
  "qfura-layout_flat-decomp_output_one_block_smerge_keep_trainable-lr_2e-4-seed_43"
)

# Validate every keep dir exists before doing anything destructive.
for d in "${KEEP_DIRS[@]}"; do
  if [ ! -d "$ROOT/$d" ]; then
    echo "ERROR: keep dir missing: $ROOT/$d" >&2
    exit 1
  fi
done

# Find all *.bin under ROOT, then filter out anything inside a keep dir.
mapfile -t ALL_BINS < <(find "$ROOT" -type f -name '*.bin' | sort)

KEEP_BINS=()
DELETE_BINS=()
for bin in "${ALL_BINS[@]}"; do
  rel="${bin#$ROOT/}"
  top="${rel%%/*}"
  keep=false
  for kd in "${KEEP_DIRS[@]}"; do
    if [ "$top" = "$kd" ]; then
      keep=true
      break
    fi
  done
  if $keep; then
    KEEP_BINS+=("$bin")
  else
    DELETE_BINS+=("$bin")
  fi
done

echo "==== KEEP (${#KEEP_BINS[@]} files) ===="
for b in "${KEEP_BINS[@]}"; do echo "  KEEP   $b"; done

echo
echo "==== DELETE (${#DELETE_BINS[@]} files) ===="
for b in "${DELETE_BINS[@]}"; do
  size=$(du -h --apparent-size "$b" 2>/dev/null | awk '{print $1}')
  printf "  DELETE %-8s %s\n" "$size" "$b"
done

# Total reclaimable size.
if [ ${#DELETE_BINS[@]} -gt 0 ]; then
  echo
  printf "Total to reclaim: "
  du -ch --apparent-size "${DELETE_BINS[@]}" | tail -1 | awk '{print $1}'
fi

if [ "${1:-}" = "--apply" ]; then
  echo
  echo "Applying deletion..."
  for b in "${DELETE_BINS[@]}"; do
    rm -v -- "$b"
  done
  echo "Done."
else
  echo
  echo "(dry-run) Re-run with --apply to delete the files above."
fi
