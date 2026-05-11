#!/usr/bin/env bash
# Cleanup checkpoints under
#   /data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/
#
# Policy:
#   1. *.bin sweep: keep only the 5 runs in the "Source-Domain Forgetting"
#      table of docs/exp_results/lift_math.md; delete every other *.bin.
#   2. *.safetensors deletion: explicitly delete the .safetensors checkpoint
#      shards for the qlora and qfura math runs (per user request). Adapter
#      .safetensors files inside the unmerged qlora dirs are included.
#
# Keep dirs (rows in the Source-Domain Forgetting table):
#   - full-lr_1e-5-seed_43-projE
#   - blocktt-lr_2e-4-decomp_output_one_block_pos_small-rank_full-smerge_trainable-type_all-seed_43
#   - lora-lr_1e-4-rank_64-alpha_128-seed_43
#   - lora-lr_2e-4-rank_128-alpha_256-seed_43
#   - svd-lr_1e-5-pos_output-smerge_frozen-type_all-seed_43
#
# Usage:
#   bash scripts/cleanup_math_ckpts.sh           # dry-run: print KEEP / DELETE lists
#   bash scripts/cleanup_math_ckpts.sh --apply   # actually rm the DELETE list

set -euo pipefail

ROOT=/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B

KEEP_DIRS=(
  "full-lr_1e-5-seed_43-projE"
  "blocktt-lr_2e-4-decomp_output_one_block_pos_small-rank_full-smerge_trainable-type_all-seed_43"
  "lora-lr_1e-4-rank_64-alpha_128-seed_43"
  "lora-lr_2e-4-rank_128-alpha_256-seed_43"
  "svd-lr_1e-5-pos_output-smerge_frozen-type_all-seed_43"
)

# Dirs whose .safetensors shards should be force-deleted regardless of the
# .bin keep policy. (These dirs typically hold no .bin, but if they did the
# .bin would still go through the keep filter below.)
FORCE_DELETE_SAFETENSORS_DIRS=(
  "qfura-layout_flat-decomp_output_one_block_smerge_keep_trainable-lr_1e-4-seed_43"
  "qlora-r_48-alpha_96-lr_1e-4-seed_43"
  "qlora-r_48-alpha_96-lr_1e-4-seed_43-merged"
  "qlora-r_64-alpha_128-lr_1e-4-seed_43"
  "qlora-r_64-alpha_128-lr_1e-4-seed_43-merged"
)

# Validate every keep dir exists before doing anything destructive.
for d in "${KEEP_DIRS[@]}"; do
  if [ ! -d "$ROOT/$d" ]; then
    echo "ERROR: keep dir missing: $ROOT/$d" >&2
    exit 1
  fi
done

# ---------- .bin sweep ----------
mapfile -t ALL_BINS < <(find "$ROOT" -type f -name '*.bin' | sort)

KEEP_BINS=()
DELETE_BINS=()
for bin in "${ALL_BINS[@]}"; do
  rel="${bin#$ROOT/}"
  top="${rel%%/*}"
  keep=false
  for kd in "${KEEP_DIRS[@]}"; do
    if [ "$top" = "$kd" ]; then keep=true; break; fi
  done
  if $keep; then
    KEEP_BINS+=("$bin")
  else
    DELETE_BINS+=("$bin")
  fi
done

# ---------- .safetensors force-delete ----------
DELETE_ST=()
for sd in "${FORCE_DELETE_SAFETENSORS_DIRS[@]}"; do
  [ -d "$ROOT/$sd" ] || { echo "(skip, missing dir) $ROOT/$sd"; continue; }
  while IFS= read -r f; do
    DELETE_ST+=("$f")
  done < <(find "$ROOT/$sd" -type f -name '*.safetensors' | sort)
done

# ---------- Reporting ----------
echo "==== KEEP .bin (${#KEEP_BINS[@]} files) ===="
for b in "${KEEP_BINS[@]}"; do echo "  KEEP   $b"; done

echo
echo "==== DELETE .bin (${#DELETE_BINS[@]} files) ===="
for b in "${DELETE_BINS[@]}"; do
  size=$(du -h --apparent-size "$b" 2>/dev/null | awk '{print $1}')
  printf "  DELETE %-8s %s\n" "$size" "$b"
done

echo
echo "==== DELETE .safetensors — qlora/qfura (${#DELETE_ST[@]} files) ===="
for f in "${DELETE_ST[@]}"; do
  size=$(du -h --apparent-size "$f" 2>/dev/null | awk '{print $1}')
  printf "  DELETE %-8s %s\n" "$size" "$f"
done

ALL_DELETE=("${DELETE_BINS[@]}" "${DELETE_ST[@]}")
if [ ${#ALL_DELETE[@]} -gt 0 ]; then
  echo
  printf "Total to reclaim: "
  du -ch --apparent-size "${ALL_DELETE[@]}" | tail -1 | awk '{print $1}'
fi

if [ "${1:-}" = "--apply" ]; then
  echo
  echo "Applying deletion..."
  for f in "${ALL_DELETE[@]}"; do
    rm -v -- "$f"
  done
  echo "Done."
else
  echo
  echo "(dry-run) Re-run with --apply to delete the files above."
fi
