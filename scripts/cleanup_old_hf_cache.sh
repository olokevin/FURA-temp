#!/usr/bin/env bash
# Safely clean up ~/.cache/huggingface after HF_HOME was moved to /data/yequan/huggingface.
#
# Strategy:
#   1. Refuse to run if anything has open file handles under ~/.cache/huggingface.
#   2. Move ~/.cache/huggingface -> ~/.cache/huggingface.bak (instant, reversible).
#   3. Let the caller smoke-test their workflows.
#   4. A second invocation with --purge deletes the .bak dir for good.
#
# Also explicitly deletes the only large model that lives ONLY in ~/.cache and not
# in /data: models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B (~15 GB).
#
# Usage:
#   bash cleanup_old_hf_cache.sh           # stage: rename to .bak, drop the Llama-8B dir
#   bash cleanup_old_hf_cache.sh --purge   # commit: rm -rf the .bak dir
#   bash cleanup_old_hf_cache.sh --restore # undo: rename .bak back to original

set -euo pipefail

OLD_CACHE="${HOME}/.cache/huggingface"
BAK_CACHE="${HOME}/.cache/huggingface.bak"
NEW_HF_HOME="/data/yequan/huggingface"

DOOMED_UNIQUE_MODEL="models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
err()  { printf '\033[31m%s\033[0m\n' "$*" >&2; }

mode="${1:-stage}"

case "$mode" in
    --purge)
        if [ ! -d "$BAK_CACHE" ]; then
            err "No $BAK_CACHE to purge. Run without --purge first to stage the rename."
            exit 1
        fi
        bold "About to permanently delete: $BAK_CACHE"
        du -sh "$BAK_CACHE" 2>/dev/null || true
        read -r -p "Type 'yes' to confirm permanent delete: " ans
        [ "$ans" = "yes" ] || { warn "Aborted."; exit 1; }
        rm -rf "$BAK_CACHE"
        bold "Deleted $BAK_CACHE."
        df -h "$HOME" | tail -1
        exit 0
        ;;
    --restore)
        if [ ! -d "$BAK_CACHE" ]; then
            err "No $BAK_CACHE to restore."
            exit 1
        fi
        if [ -e "$OLD_CACHE" ]; then
            err "$OLD_CACHE already exists. Refusing to overwrite."
            exit 1
        fi
        mv "$BAK_CACHE" "$OLD_CACHE"
        bold "Restored $OLD_CACHE."
        exit 0
        ;;
    stage|"")
        ;;
    *)
        err "Unknown mode: $mode (use no arg, --purge, or --restore)"
        exit 1
        ;;
esac

# ---- stage mode ----

bold "[1/5] Sanity checks"
if [ ! -d "$OLD_CACHE" ]; then
    err "$OLD_CACHE does not exist; nothing to do."
    exit 0
fi
if [ -L "$OLD_CACHE" ]; then
    err "$OLD_CACHE is a symlink, not a real dir. Bailing — handle manually."
    exit 1
fi
if [ ! -d "$NEW_HF_HOME/hub" ]; then
    err "$NEW_HF_HOME/hub does not exist — your new HF_HOME looks empty. Bailing."
    exit 1
fi
if [ -e "$BAK_CACHE" ]; then
    err "$BAK_CACHE already exists. Either inspect it and run --purge, or remove it manually."
    exit 1
fi

bold "[2/5] Checking for open files under $OLD_CACHE"
if command -v lsof >/dev/null 2>&1; then
    if lsof +D "$OLD_CACHE" 2>/dev/null | grep -q .; then
        err "Open file handles found under $OLD_CACHE. Refusing to proceed."
        lsof +D "$OLD_CACHE" 2>/dev/null | head -20 >&2
        exit 1
    fi
    echo "  no open handles"
else
    warn "  lsof not installed; skipping open-handle check"
fi

bold "[3/5] Verifying every large item in $OLD_CACHE/hub also exists in $NEW_HF_HOME/hub"
MISSING_FROM_NEW=()
while IFS= read -r entry; do
    name=$(basename "$entry")
    [ "$name" = "version.txt" ] && continue
    if [ ! -e "$NEW_HF_HOME/hub/$name" ]; then
        MISSING_FROM_NEW+=("$name")
    fi
done < <(find "$OLD_CACHE/hub" -mindepth 1 -maxdepth 1)

# Filter out the one we're explicitly OK losing.
FILTERED_MISSING=()
for n in "${MISSING_FROM_NEW[@]:-}"; do
    [ "$n" = "$DOOMED_UNIQUE_MODEL" ] && continue
    FILTERED_MISSING+=("$n")
done

if [ "${#FILTERED_MISSING[@]}" -gt 0 ]; then
    warn "These items exist ONLY in $OLD_CACHE/hub (not in $NEW_HF_HOME/hub):"
    for n in "${FILTERED_MISSING[@]}"; do
        sz=$(du -sh "$OLD_CACHE/hub/$n" 2>/dev/null | awk '{print $1}')
        printf '    %-8s  %s\n' "$sz" "$n" >&2
    done
    echo "" >&2
    warn "If you want to keep any of these, ctrl-C now and move them into $NEW_HF_HOME/hub/ first."
    warn "Otherwise they will go into $BAK_CACHE and be deletable with --purge."
    read -r -p "Continue anyway? Type 'yes': " ans
    [ "$ans" = "yes" ] || { warn "Aborted."; exit 1; }
fi

bold "[4/5] Explicitly removing $DOOMED_UNIQUE_MODEL from $OLD_CACHE before renaming"
if [ -d "$OLD_CACHE/hub/$DOOMED_UNIQUE_MODEL" ]; then
    sz=$(du -sh "$OLD_CACHE/hub/$DOOMED_UNIQUE_MODEL" 2>/dev/null | awk '{print $1}')
    echo "  removing $OLD_CACHE/hub/$DOOMED_UNIQUE_MODEL ($sz)"
    rm -rf "$OLD_CACHE/hub/$DOOMED_UNIQUE_MODEL"
else
    echo "  (not present, skipping)"
fi

bold "[5/5] Renaming $OLD_CACHE -> $BAK_CACHE"
mv "$OLD_CACHE" "$BAK_CACHE"

echo ""
bold "Done."
echo "  - Old cache moved aside to: $BAK_CACHE"
echo "  - HF_HOME is already pointing at: $NEW_HF_HOME"
echo ""
echo "Next steps:"
echo "  1. Run your normal workflows (training, eval, vLLM). If something breaks because"
echo "     it expected a file under $OLD_CACHE, run this script with --restore to undo."
echo "  2. Once you're confident nothing is broken (give it a few days):"
echo "       bash $0 --purge"
echo "     to permanently free the disk space."
echo ""
df -h "$HOME" | tail -1
