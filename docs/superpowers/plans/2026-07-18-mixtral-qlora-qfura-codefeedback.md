# Mixtral-8x7B QLoRA/QFURA on CodeFeedback → HumanEval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add launch scripts to fine-tune Mixtral-8x7B with QLoRA and QFURA on CodeFeedback-100K, an eval script for HumanEval, and verify the training plumbing with a tiny smoke run.

**Architecture:** Copy the proven `finetune_metamath_q{lora,fura}_70b.sh` templates, swapping model → `mistralai/Mixtral-8x7B-v0.1`, dataset → `pissa-dataset/python/train.json`, and target modules → Mixtral's MoE leaf names. One source change extends the QFURA target-name resolver to Mixtral (`w1/w2/w3`). A new `eval_code.sh` wraps the vendored PiSSA HumanEval pipeline (gen_vllm → code_process → evalplus), branching on checkpoint layout (qfura saves a dense model; qlora saves an adapter that must be merged first).

**Tech Stack:** bash, PyTorch, HuggingFace Transformers/PEFT, bitsandbytes (NF4), accelerate, vLLM, evalplus 0.3.1, custom `btt_layer.py` BTT decomposition. Env managed by `uv`.

## Global Constraints

- Recipe (all finetune scripts, from PiSSA paper): AdamW, lr `2e-5`, `--lr_scheduler_type cosine`, `--num_warmup_steps 0.03`, `--weight_decay 0.`, effective batch 128 (`per_device_train_batch_size=1` × `gradient_accumulation_steps=128`), `--max_seq_len 512`, `--num_train_epochs 1`, NF4 base.
- QLoRA/QFURA rank: qlora `--lora_r 64 --lora_alpha 64`; qfura keeps CLAUDE.md defaults `--blocktt_rank full --decomp_mode output_one_block --train_position small --s_merged_to keep_trainable`.
- Every finetune script MUST pass `--load_last_model` (CLAUDE.md checkpoint policy).
- MoE router (`block_sparse_moe.gate`) MUST NOT be adapted. Target modules = `q_proj k_proj v_proj o_proj w1 w2 w3` only.
- Model: `mistralai/Mixtral-8x7B-v0.1`. Training data: `/data/yequan/pissa-dataset/python/train.json` (CodeFeedback-100K, 104,848 Python samples). Eval: HumanEval only (report standard, not HumanEval+).
- `PROJECT_DIR=/home/yequan/Project/lora/lora-without-regret`, `SRC_DIR=${PROJECT_DIR}/ref/LIFT`, `HF_HOME=/data/yequan/huggingface`, `OUTPUT_SRC_DIR=/data/yequan/fura/lift`.
- All scripts must remain env-overridable in the same style as the metamath-70b templates (every tunable defined as `VAR="${VAR:-default}"`).

---

### Task 1: Extend QFURA target-name resolver to Mixtral

**Files:**
- Modify: `btt_layer.py:523-540` (`get_blocktt_target_module_names`)
- Test: `tests/test_btt_target_modules_mixtral.py` (create)

**Interfaces:**
- Consumes: nothing (leaf function).
- Produces: `get_blocktt_target_module_names("mixtral_all") -> ("q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3")`. Consumed by `finetune_qfura.py:515` via `--trainable_type mixtral_all` (Task 3).

- [ ] **Step 1: Write the failing test**

Create `tests/test_btt_target_modules_mixtral.py`:

```python
import os, sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from btt_layer import get_blocktt_target_module_names


class TestMixtralTargetModules(unittest.TestCase):
    def test_mixtral_all_covers_attention_and_experts(self):
        names = get_blocktt_target_module_names("mixtral_all")
        # Mixtral attention leaf names
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            self.assertIn(n, names)
        # Mixtral MoE expert leaf names (w1=gate, w2=down, w3=up)
        for n in ("w1", "w2", "w3"):
            self.assertIn(n, names)

    def test_mixtral_all_excludes_router_and_llama_mlp_names(self):
        names = get_blocktt_target_module_names("mixtral_all")
        # router must never be adapted
        self.assertNotIn("gate", names)
        # Llama-style MLP names must not leak in (Mixtral has none)
        for n in ("gate_proj", "up_proj", "down_proj"):
            self.assertNotIn(n, names)

    def test_existing_types_unchanged(self):
        self.assertEqual(
            get_blocktt_target_module_names("all"),
            ("gate_proj", "up_proj", "down_proj",
             "q_proj", "k_proj", "v_proj", "o_proj"),
        )

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            get_blocktt_target_module_names("nonsense")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/yequan/Project/lora/lora-without-regret && python -m unittest tests/test_btt_target_modules_mixtral.py -v`
Expected: FAIL on `test_mixtral_all_*` with `ValueError: blocktt_type must be one of: all, mlp, attn, 5mod_lora` (the `mixtral_all` branch does not exist yet). `test_existing_types_unchanged` and `test_invalid_type_raises` PASS.

- [ ] **Step 3: Add the `mixtral_all` branch**

In `btt_layer.py`, inside `get_blocktt_target_module_names`, add the branch immediately after the `5mod_lora` branch (before the final `raise ValueError`), and update the error message to list the new type:

```python
    if blocktt_type == "5mod_lora":
        return ("q_proj", "k_proj", "v_proj", "up_proj", "down_proj")
    if blocktt_type == "mixtral_all":
        # Mixtral-8x7B MoE: attention uses *_proj, experts use w1 (gate),
        # w2 (down), w3 (up). The router (block_sparse_moe.gate) is excluded.
        return ("q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3")
    raise ValueError(
        "blocktt_type must be one of: all, mlp, attn, 5mod_lora, mixtral_all"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/yequan/Project/lora/lora-without-regret && python -m unittest tests/test_btt_target_modules_mixtral.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Syntax-check and commit**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -m py_compile btt_layer.py
git add btt_layer.py tests/test_btt_target_modules_mixtral.py
git commit -m "feat(btt): add mixtral_all target-module type (attn + w1/w2/w3 experts)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019dbK8KWDRCvGG3o95YYt1Z"
```

---

### Task 2: QLoRA launch script for Mixtral / CodeFeedback

**Files:**
- Create: `ref/LIFT/bash_scripts/finetune_code_qlora_mixtral.sh`
- Reference (do not modify): `ref/LIFT/bash_scripts/finetune_metamath_qlora_70b.sh`, `ref/LIFT/src/finetune_qlora.py`

**Interfaces:**
- Consumes: `finetune_qlora.py` CLI (already accepts every flag used here — `--target_modules` at `finetune_qlora.py:146-152` is an explicit `nargs="+"`).
- Produces: on run, `<OUTPUT>/last_adapter/` (PEFT adapter) + `<OUTPUT>/training.log`. `<OUTPUT>` default path token is `codefeedback100k/.../qlora-...`. Consumed by `eval_code.sh` (Task 4).

- [ ] **Step 1: Create the script**

Create `ref/LIFT/bash_scripts/finetune_code_qlora_mixtral.sh` (this is the metamath-70b qlora script with model/data/target-modules/tags changed; router excluded from `--target_modules`):

```bash
#!/bin/bash
# qlora on Mixtral-8x7B / CodeFeedback-100K, single H100 (94 GB).
#
# Follows the PiSSA (Q)LoRA recipe (lr=2e-5, batch=1x128, seq=512, 1 epoch on
# ~100K examples), NF4 base + LoRA adapters (rank 64). Target modules cover
# Mixtral attention (q/k/v/o_proj) and MoE experts (w1/w2/w3). The MoE router
# (block_sparse_moe.gate) is deliberately excluded.
#
# Mixtral-8x7B in NF4 ~24 GB on GPU; fits one H100.
#
# Override env vars: MAX_STEPS, lr, lora_r, lora_alpha, seed,
# per_device_train_batch_size, gradient_accumulation_steps, model_max_length,
# DEVICE, no_wandb.

pwd
hostname
date
echo starting job...
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME="${HF_HOME:-/data/yequan/huggingface}"

PROJECT_DIR="${PROJECT_DIR:-/home/yequan/Project/lora/lora-without-regret}"
SRC_DIR="${SRC_DIR:-${PROJECT_DIR}/ref/LIFT}"
DATA_PATH="${DATA_PATH:-/data/yequan/pissa-dataset/python/train.json}"
OUTPUT_SRC_DIR="${OUTPUT_SRC_DIR:-/data/yequan/fura/lift}"

MODEL="${MODEL:-mistralai/Mixtral-8x7B-v0.1}"
lora_r="${lora_r:-64}"
lora_alpha="${lora_alpha:-64}"
lora_dropout="${lora_dropout:-0.0}"
prompt_style="${prompt_style:-pissa}"
trainable_param_dtype="${trainable_param_dtype:-fp32}"
optimizer_name="${optimizer_name:-adamw}"
target_modules="${target_modules:-q_proj k_proj v_proj o_proj w1 w2 w3}"

lr="${lr:-2e-5}"
seed="${seed:-42}"
MAX_STEPS="${MAX_STEPS:-0}"
num_train_epochs="${num_train_epochs:-1}"
model_max_length="${model_max_length:-512}"
per_device_train_batch_size="${per_device_train_batch_size:-1}"
gradient_accumulation_steps="${gradient_accumulation_steps:-128}"
model_tag="${MODEL##*/}"

wandb_project="${wandb_project:-qlora-code-${model_tag}}"
wandb_run_id="${wandb_run_id:-$(uv run --project ${PROJECT_DIR} python -c 'import wandb; print(wandb.util.generate_id())' 2>/dev/null)}"
no_wandb_flag=""
if [ "${no_wandb:-0}" = "1" ]; then
    no_wandb_flag="--no_wandb"
fi
export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

echo $MODEL

OUTPUT="${OUTPUT:-${OUTPUT_SRC_DIR}/codefeedback100k/${MODEL}/qlora-r_${lora_r}-alpha_${lora_alpha}-bs${per_device_train_batch_size}x${gradient_accumulation_steps}-lr_${lr}-seed_${seed}}"
run_name="${run_name:-$(basename "$OUTPUT")}"

mkdir -p $OUTPUT

cd ${SRC_DIR}

uv run --project ${PROJECT_DIR} accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_qlora.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size 1 \
    --logging_steps 1 \
    --max_seq_len ${model_max_length} \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs ${num_train_epochs} \
    --mixed_precision bf16 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --gradient_checkpointing \
    --instruction_type single \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --target_modules ${target_modules} \
    --prompt_style ${prompt_style} \
    --trainable_param_dtype ${trainable_param_dtype} \
    --optimizer ${optimizer_name} \
    --load_last_model \
    --max_steps ${MAX_STEPS} \
    --val_set_size 0 \
    --data_path ${DATA_PATH} \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    ${no_wandb_flag} \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log
```

- [ ] **Step 2: Verify it parses (bash syntax, no run)**

Run: `bash -n /home/yequan/Project/lora/lora-without-regret/ref/LIFT/bash_scripts/finetune_code_qlora_mixtral.sh && echo SYNTAX_OK`
Expected: `SYNTAX_OK`

- [ ] **Step 3: Verify target modules and recipe values are present**

Run:
```bash
cd /home/yequan/Project/lora/lora-without-regret
grep -E 'mistralai/Mixtral-8x7B-v0.1|python/train.json|q_proj k_proj v_proj o_proj w1 w2 w3|lr:-2e-5|load_last_model|val_set_size 0' ref/LIFT/bash_scripts/finetune_code_qlora_mixtral.sh
```
Expected: all six patterns print (model, data, target modules, lr, load_last_model, val_set_size 0). Confirm no `gate_proj`/`up_proj`/`down_proj`/`block_sparse_moe` appears (router excluded):
```bash
! grep -qE 'gate_proj|up_proj|down_proj|block_sparse_moe|--target_modules .*gate([^_]|$)' ref/LIFT/bash_scripts/finetune_code_qlora_mixtral.sh && echo NO_ROUTER_OK
```
Expected: `NO_ROUTER_OK`

- [ ] **Step 4: Commit**

```bash
cd /home/yequan/Project/lora/lora-without-regret
git add ref/LIFT/bash_scripts/finetune_code_qlora_mixtral.sh
git commit -m "feat(scripts): qlora Mixtral-8x7B on CodeFeedback-100K launcher

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019dbK8KWDRCvGG3o95YYt1Z"
```

---

### Task 3: QFURA launch script for Mixtral / CodeFeedback

**Files:**
- Create: `ref/LIFT/bash_scripts/finetune_code_qfura_mixtral.sh`
- Reference (do not modify): `ref/LIFT/bash_scripts/finetune_metamath_qfura_70b.sh`, `ref/LIFT/src/finetune_qfura.py`

**Interfaces:**
- Consumes: `finetune_qfura.py` CLI + `get_blocktt_target_module_names("mixtral_all")` from Task 1 (via `--trainable_type mixtral_all`).
- Produces: on run, `<OUTPUT>/last/` (dense HF checkpoint, materialized from BTT cores) + `<OUTPUT>/training.log`. Consumed by `eval_code.sh` (Task 4).

- [ ] **Step 1: Create the script**

Create `ref/LIFT/bash_scripts/finetune_code_qfura_mixtral.sh` (the metamath-70b qfura script with model/data/tags changed and `trainable_type=mixtral_all`; all CLAUDE.md qfura defaults kept):

```bash
#!/bin/bash
# qfura on Mixtral-8x7B / CodeFeedback-100K, single H100 (94 GB).
#
# Follows the PiSSA QPiSSA recipe (lr=2e-5, batch=1x128, seq=512, 1 epoch on
# ~100K examples) but uses qfura's BTT decomposition. Defaults match the
# project-wide qfura defaults (CLAUDE.md): rank=full, decomp=output_one_block,
# train_position=small, s_merged_to=keep_trainable. trainable_type=mixtral_all
# converts Mixtral attention (q/k/v/o_proj) + MoE experts (w1/w2/w3); the MoE
# router (block_sparse_moe.gate) is left frozen (not a BTT target).
#
# --load_strategy=layer_stream converts+quantizes one Linear at a time so the
# bf16 model never fully materialises on GPU.
#
# Override env vars: MAX_STEPS, lr, seed, per_device_train_batch_size,
# gradient_accumulation_steps, model_max_length, DEVICE, no_wandb.

pwd
hostname
date
echo starting job...
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME="${HF_HOME:-/data/yequan/huggingface}"

PROJECT_DIR="${PROJECT_DIR:-/home/yequan/Project/lora/lora-without-regret}"
SRC_DIR="${SRC_DIR:-${PROJECT_DIR}/ref/LIFT}"
DATA_PATH="${DATA_PATH:-/data/yequan/pissa-dataset/python/train.json}"
OUTPUT_SRC_DIR="${OUTPUT_SRC_DIR:-/data/yequan/fura/lift}"

MODEL="${MODEL:-mistralai/Mixtral-8x7B-v0.1}"
decomp_mode="${decomp_mode:-output_one_block}"
train_position="${train_position:-small}"
blocktt_rank="${blocktt_rank:-full}"
s_merged_to="${s_merged_to:-keep_trainable}"
trainable_type="${trainable_type:-mixtral_all}"
quant_block_layout="${quant_block_layout:-flat}"
load_strategy="${load_strategy:-layer_stream}"
prompt_style="${prompt_style:-pissa}"
trainable_param_dtype="${trainable_param_dtype:-bf16}"
optimizer_name="${optimizer_name:-paged_adamw_8bit}"

lr="${lr:-2e-5}"
seed="${seed:-42}"
MAX_STEPS="${MAX_STEPS:-0}"
num_train_epochs="${num_train_epochs:-1}"
model_max_length="${model_max_length:-512}"
per_device_train_batch_size="${per_device_train_batch_size:-1}"
gradient_accumulation_steps="${gradient_accumulation_steps:-128}"
model_tag="${MODEL##*/}"

wandb_project="${wandb_project:-qfura-code-${model_tag}}"
wandb_run_id="${wandb_run_id:-$(uv run --project ${PROJECT_DIR} python -c 'import wandb; print(wandb.util.generate_id())' 2>/dev/null)}"
no_wandb_flag=""
if [ "${no_wandb:-0}" = "1" ]; then
    no_wandb_flag="--no_wandb"
fi
export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

echo $MODEL

OUTPUT="${OUTPUT:-${OUTPUT_SRC_DIR}/codefeedback100k/${MODEL}/qfura-layout_${quant_block_layout}-decomp_${decomp_mode}_smerge_${s_merged_to}-bs${per_device_train_batch_size}x${gradient_accumulation_steps}-lr_${lr}-seed_${seed}}"
run_name="${run_name:-$(basename "$OUTPUT")}"

mkdir -p $OUTPUT

cd ${SRC_DIR}

uv run --project ${PROJECT_DIR} accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_qfura.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size 1 \
    --logging_steps 1 \
    --max_seq_len ${model_max_length} \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs ${num_train_epochs} \
    --mixed_precision bf16 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --gradient_checkpointing \
    --instruction_type single \
    --decomp_mode ${decomp_mode} \
    --train_position ${train_position} \
    --blocktt_rank ${blocktt_rank} \
    --s_merged_to ${s_merged_to} \
    --trainable_type ${trainable_type} \
    --quant_block_layout ${quant_block_layout} \
    --load_strategy ${load_strategy} \
    --prompt_style ${prompt_style} \
    --trainable_param_dtype ${trainable_param_dtype} \
    --optimizer ${optimizer_name} \
    --load_last_model \
    --max_steps ${MAX_STEPS} \
    --data_path ${DATA_PATH} \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    ${no_wandb_flag} \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log
```

- [ ] **Step 2: Verify it parses (bash syntax, no run)**

Run: `bash -n /home/yequan/Project/lora/lora-without-regret/ref/LIFT/bash_scripts/finetune_code_qfura_mixtral.sh && echo SYNTAX_OK`
Expected: `SYNTAX_OK`

- [ ] **Step 3: Verify model/data/qfura-defaults present**

Run:
```bash
cd /home/yequan/Project/lora/lora-without-regret
grep -E 'mistralai/Mixtral-8x7B-v0.1|python/train.json|trainable_type:-mixtral_all|decomp_mode:-output_one_block|train_position:-small|s_merged_to:-keep_trainable|blocktt_rank:-full|load_strategy:-layer_stream|load_last_model' ref/LIFT/bash_scripts/finetune_code_qfura_mixtral.sh
```
Expected: all nine patterns print.

- [ ] **Step 4: Commit**

```bash
cd /home/yequan/Project/lora/lora-without-regret
git add ref/LIFT/bash_scripts/finetune_code_qfura_mixtral.sh
git commit -m "feat(scripts): qfura Mixtral-8x7B on CodeFeedback-100K launcher

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019dbK8KWDRCvGG3o95YYt1Z"
```

---

### Task 4: HumanEval eval script

**Files:**
- Create: `ref/LIFT/bash_scripts/eval_code.sh`
- Reference (do not modify): `ref/LIFT/bash_scripts/eval_math.sh` (CKPT-resolution + arg-parse pattern), `ref/PiSSA/utils/gen_vllm.py`, `ref/PiSSA/utils/code_process.py`, `ref/PiSSA/utils/merge_adapter.py`

**Interfaces:**
- Consumes:
  - qfura checkpoint dir `<OUTPUT>/last/` (dense HF model, has `config.json`) from Task 3.
  - qlora checkpoint dir `<OUTPUT>/last_adapter/` (PEFT adapter, has `adapter_config.json`) from Task 2 → merged via `merge_adapter.py --base_model <MODEL> --adapter <adapter_dir> --output_path <merged_dir>`.
  - `gen_vllm.py` args: `--model <dir> --data_path <pissa-dataset root> --sub_task python --dataset_split test --output_file <f> --max_tokens 1024 --temperature 0.0`.
  - `code_process.py --path <response.jsonl>` writes `humaneval.jsonl` next to it.
  - `evalplus.evaluate --dataset humaneval --samples <humaneval.jsonl>`.
- Produces: prints HumanEval pass@1 to stdout + `eval.log` in the checkpoint dir.

- [ ] **Step 1: Create the script**

Create `ref/LIFT/bash_scripts/eval_code.sh`:

```bash
#!/bin/bash
# HumanEval eval for CodeFeedback-fine-tuned checkpoints (qlora or qfura),
# wrapping the vendored PiSSA pipeline: gen_vllm -> code_process -> evalplus.
#
# Usage:
#   bash bash_scripts/eval_code.sh CKPT=<checkpoint_dir> [base_model=<hf_model>] \
#        [ADAPTER=1] [DEVICE=<gpu>]
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
    echo "Usage: bash bash_scripts/eval_code.sh CKPT=<checkpoint_dir> [base_model=<hf_model>] [DEVICE=<gpu>]"
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
    EVAL_MODEL="$MODEL_DIR"
fi

OUT_DIR="${MODEL_DIR}/humaneval"
mkdir -p "$OUT_DIR"
RESP="${OUT_DIR}/python_response.jsonl"

cd "$PROJECT_DIR"

# --- 1. Generate completions on the HumanEval (python) test split via vLLM ---
uv run --project "$PROJECT_DIR" python "${PISSA_UTILS}/gen_vllm.py" \
    --model "$EVAL_MODEL" \
    --data_path "$DATA_ROOT" \
    --sub_task python \
    --dataset_split test \
    --output_file "$RESP" \
    --temperature 0.0 \
    --max_tokens 1024 \
    2> >(tee "${OUT_DIR}/eval_err.log" >&2) | tee "${OUT_DIR}/eval.log"

# --- 2. Post-process into evalplus humaneval.jsonl (written next to $RESP) ---
uv run --project "$PROJECT_DIR" python "${PISSA_UTILS}/code_process.py" \
    --path "$RESP" | tee -a "${OUT_DIR}/eval.log"

# --- 3. Score with evalplus (standard HumanEval; HumanEval+ also printed) ---
uv run --project "$PROJECT_DIR" python -m evalplus.evaluate \
    --dataset humaneval \
    --samples "${OUT_DIR}/humaneval.jsonl" | tee -a "${OUT_DIR}/eval.log"

echo "[eval_code] done. Results in ${OUT_DIR}/eval.log"
```

- [ ] **Step 2: Verify it parses (bash syntax, no run)**

Run: `bash -n /home/yequan/Project/lora/lora-without-regret/ref/LIFT/bash_scripts/eval_code.sh && echo SYNTAX_OK`
Expected: `SYNTAX_OK`

- [ ] **Step 3: Verify usage/exit behavior with no CKPT**

Run: `bash /home/yequan/Project/lora/lora-without-regret/ref/LIFT/bash_scripts/eval_code.sh; echo "exit=$?"`
Expected: prints the `Usage:` line and `exit=1`.

- [ ] **Step 4: Verify the three pipeline stages and merge branch are wired**

Run:
```bash
cd /home/yequan/Project/lora/lora-without-regret
grep -E 'merge_adapter.py|gen_vllm.py|code_process.py|evalplus.evaluate|--sub_task python|--dataset humaneval' ref/LIFT/bash_scripts/eval_code.sh
```
Expected: all six patterns print.

- [ ] **Step 5: Commit**

```bash
cd /home/yequan/Project/lora/lora-without-regret
git add ref/LIFT/bash_scripts/eval_code.sh
git commit -m "feat(scripts): eval_code.sh HumanEval pipeline (gen_vllm/code_process/evalplus)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019dbK8KWDRCvGG3o95YYt1Z"
```

---

### Task 5: Smoke test both finetune scripts on Mixtral

**Files:**
- No source changes. Runs Task 2 and Task 3 scripts with a tiny budget.
- Reference: the two finetune scripts from Tasks 2–3.

**Interfaces:**
- Consumes: `finetune_code_qlora_mixtral.sh`, `finetune_code_qfura_mixtral.sh`.
- Produces: verification output only (loss lines, conversion counts, written checkpoints). No commit.

**Note on cost:** `mistralai/Mixtral-8x7B-v0.1` (~90 GB) is NOT cached under `HF_HOME`; the first script run downloads it (long pole). GPUs 4–7 are free (95 GB each). Run on `DEVICE=4`. Use a single-sample micro-slice via `MAX_STEPS` + batch 1 × accum 1 so only ~2 optimizer steps run.

- [ ] **Step 1: Pre-stage the Mixtral download (surfaced, not hidden)**

Run (this is the ~90 GB download; run in background and watch):
```bash
cd /home/yequan/Project/lora/lora-without-regret
HF_HOME=/data/yequan/huggingface uv run --project /home/yequan/Project/lora/lora-without-regret python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('mistralai/Mixtral-8x7B-v0.1',
                      allow_patterns=['*.safetensors','*.json','*.model','tokenizer*'])
print('DOWNLOADED_TO', p)
"
```
Expected: ends with `DOWNLOADED_TO /data/yequan/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/...`. (If it errors with a gated-repo/401, stop and tell the user Mixtral requires `huggingface-cli login` + license acceptance — this is an external blocker, not a code bug.)

- [ ] **Step 2: Smoke-run QLoRA (2 steps, no wandb)**

Run:
```bash
cd /home/yequan/Project/lora/lora-without-regret
CUDA_VISIBLE_DEVICES=4 MAX_STEPS=2 no_wandb=1 \
  per_device_train_batch_size=1 gradient_accumulation_steps=1 \
  OUTPUT=/data/yequan/fura/lift/_smoke/qlora_mixtral \
  bash ref/LIFT/bash_scripts/finetune_code_qlora_mixtral.sh
```
Expected: model loads in NF4; PEFT prints a `trainable params: ... || all params: ... || trainable%:` line with trainable% > 0; two `Step ... Loss: <finite>` lines; a `Saved last-step adapter to .../last_adapter` line; process exits 0.

- [ ] **Step 3: Verify QLoRA smoke artifacts**

Run:
```bash
ls /data/yequan/fura/lift/_smoke/qlora_mixtral/last_adapter/adapter_config.json && \
grep -Eiq 'trainable%|trainable params' /data/yequan/fura/lift/_smoke/qlora_mixtral/training.log && \
! grep -Eiq 'Traceback|CUDA out of memory|nan' /data/yequan/fura/lift/_smoke/qlora_mixtral/err.log && echo QLORA_SMOKE_OK
```
Expected: `QLORA_SMOKE_OK` (adapter written, trainable-params logged, no traceback/OOM/nan).

- [ ] **Step 4: Smoke-run QFURA (2 steps, no wandb)**

Run:
```bash
cd /home/yequan/Project/lora/lora-without-regret
CUDA_VISIBLE_DEVICES=4 MAX_STEPS=2 no_wandb=1 \
  per_device_train_batch_size=1 gradient_accumulation_steps=1 \
  OUTPUT=/data/yequan/fura/lift/_smoke/qfura_mixtral \
  bash ref/LIFT/bash_scripts/finetune_code_qfura_mixtral.sh
```
Expected: model streams+converts; a `Converted modules: <N>` / `Trainable params: ...` line with N > 0 (must cover experts, so N should be ~ layers × 7 modules × experts for MLP, well over the 32×4 attention-only count); two `Step ... Loss: <finite>` lines; a `Saved last-step checkpoint to .../last` line; exits 0.

- [ ] **Step 5: Verify QFURA smoke artifacts (incl. experts were converted)**

Run:
```bash
ls /data/yequan/fura/lift/_smoke/qfura_mixtral/last/config.json && \
grep -Eq 'Converted modules: [0-9]+' /data/yequan/fura/lift/_smoke/qfura_mixtral/training.log && \
! grep -Eiq 'No layers were converted|Traceback|CUDA out of memory|nan' /data/yequan/fura/lift/_smoke/qfura_mixtral/err.log && echo QFURA_SMOKE_OK
```
Expected: `QFURA_SMOKE_OK` (dense checkpoint written, modules converted, no "No layers were converted"/traceback/OOM/nan).

Additionally confirm experts (not just attention) were converted — the converted count must exceed an attention-only run. Run:
```bash
N=$(grep -oE 'Converted modules: [0-9]+' /data/yequan/fura/lift/_smoke/qfura_mixtral/training.log | grep -oE '[0-9]+' | tail -1)
echo "converted=$N"; [ "${N:-0}" -gt 128 ] && echo EXPERTS_CONVERTED_OK
```
Expected: `converted=<N>` with N > 128 (32 layers × 4 attn = 128; experts push it far higher) and `EXPERTS_CONVERTED_OK`. If N ≤ 128, the `mixtral_all` wiring did not reach the experts — investigate Task 1/Task 3.

- [ ] **Step 6: Clean up smoke artifacts**

Run:
```bash
rm -rf /data/yequan/fura/lift/_smoke
echo CLEANED
```
Expected: `CLEANED`. (Do NOT delete the downloaded model — it is reused by real runs.)

---

## Notes for the executor

- **Do not** run a full 1-epoch (100K-sample) training or a full HumanEval eval as part of this plan — those are hours-long GPU jobs, out of scope for "test that training goes correctly."
- If Step 1 of Task 5 reveals Mixtral is a **gated repo** requiring license acceptance, that is an external blocker: report it to the user with the exact remediation (`huggingface-cli login` + accept the license at the model page) and stop; the scripts themselves are still correct.
- The qfura smoke run uses `--load_strategy layer_stream`; expect it to be slower to start than qlora because it converts+quantizes each Linear one at a time.
