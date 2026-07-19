# QLoRA & QFURA training for Mixtral-8x7B on CodeFeedback → HumanEval

Date: 2026-07-18
Status: approved

## Goal

Prepare launch scripts to fine-tune **Mixtral-8x7B** with **QLoRA** and **QFURA**
on the **CodeFeedback-100K** dataset, evaluate on **HumanEval**, following the
settings in the PiSSA paper (`docs/papers/24_PiSSA-*.pdf`). Verify the training
plumbing end-to-end with a tiny smoke run this session.

## Context / what already exists

This work slots into the existing `ref/LIFT/bash_scripts/` pattern. The
`finetune_metamath_q{lora,fura}_70b.sh` scripts are near-exact templates:

- Python entrypoints `ref/LIFT/src/finetune_qlora.py` and `finetune_qfura.py`
  already accept every flag needed.
- `SupervisedDataset` loads a single `.json` with `instruction/input/output`
  fields (the CodeFeedback data is in exactly this shape).
- PiSSA-style prompt (`--prompt_style pissa`), `--load_last_model`, and the
  fura/qfura defaults (rank=full, decomp=output_one_block, train_position=small,
  s_merged_to=keep_trainable) all carry over unchanged.

On-disk assets (verified):

- **Training data:** `/data/yequan/pissa-dataset/python/train.json` — the
  CodeFeedback set, 104,848 Python samples (`README.md`: filtered from
  m-a-p/CodeFeedback-Filtered-Instruction to Python-only).
- **Eval:** `ref/PiSSA/utils/{gen_vllm.py, code_process.py, merge_adapter.py}`
  plus `evalplus` 0.3.1 (installed in the uv env).
- **Model:** `mistralai/Mixtral-8x7B-v0.1` is NOT cached; ~90 GB download is the
  long pole for the smoke test. NF4-quantized it is ~24 GB and fits one H100.
  GPUs 4–7 are free (95 GB each).

## PiSSA paper recipe (encoded in the scripts)

From Section 5 / Appendix: AdamW, **lr 2e-5**, cosine schedule, warmup ratio
**0.03**, no weight decay, **batch 128 effective** (per-device 1 × grad-accum
128), **seq len 512**, **1 epoch** on the 100K subset, 4-bit NF4 base weights.
QPiSSA/QLoRA use **rank 64, alpha 64**. These match the metamath-70b script
defaults verbatim, so the recipe transfers unchanged.

## The one technical wrinkle: Mixtral MoE module names

Mixtral's expert MLP (`MixtralBlockSparseTop2MLP`) uses leaf names
**`w1` (gate), `w2` (down), `w3` (up)** — not Llama's `gate_proj/up_proj/down_proj`.
Attention still uses `q_proj/k_proj/v_proj/o_proj`. The MoE **router**
(`block_sparse_moe.gate`, a tiny top-2 routing Linear) must NOT be adapted.

- **QLoRA** takes `--target_modules` explicitly → pass the Mixtral names directly.
- **QFURA** resolves leaf names via `get_blocktt_target_module_names(trainable_type)`
  in `btt_layer.py`, which only knows Llama names. BTT conversion matches by
  **exact leaf name** (`name.split(".")[-1]` against the target set), so without a
  change qfura would convert zero MLP layers on Mixtral.

## Deliverables

### 1. `ref/LIFT/bash_scripts/finetune_code_qlora_mixtral.sh`
Copy of `finetune_metamath_qlora_70b.sh` with:
- `MODEL=mistralai/Mixtral-8x7B-v0.1`
- `DATA_PATH=/data/yequan/pissa-dataset/python/train.json`
- `--target_modules q_proj k_proj v_proj o_proj w1 w2 w3` (router excluded)
- output path tag `codefeedback100k`, wandb project `qlora-code-<model_tag>`
- keeps: lr 2e-5, cosine, warmup 0.03, bs 1×128, seq 512, 1 epoch, rank/alpha 64,
  `--prompt_style pissa`, fp32 adapter, `--load_last_model`, `--val_set_size 0`.

### 2. `ref/LIFT/bash_scripts/finetune_code_qfura_mixtral.sh`
Copy of `finetune_metamath_qfura_70b.sh` with:
- same MODEL / DATA_PATH / output+wandb tags (`qfura-code-<model_tag>`)
- `--trainable_type mixtral_all` (new; see change 3)
- keeps all CLAUDE.md qfura defaults + `--load_strategy layer_stream`,
  `--load_last_model`.

### 3. Code change: `btt_layer.py` `get_blocktt_target_module_names()`
Add a new branch:
```python
if blocktt_type == "mixtral_all":
    return ("q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3")
```
This is the only source-level change (matching is exact-leaf-name, so this
suffices). Add a unit test asserting the mapping.

### 4. `ref/LIFT/bash_scripts/eval_code.sh`
Wraps the PiSSA HumanEval pipeline on a merged/materialized model dir:
1. `gen_vllm.py --model <dir> --sub_task python --output_file python_response.jsonl`
2. `code_process.py --path python_response.jsonl` → `humaneval.jsonl`
3. `evalplus.evaluate --dataset humaneval --samples humaneval.jsonl`

QLoRA path: merge adapter into base via existing `merge_adapter.py` first.
QFURA path: materialize dense weights via the repo's existing BTT-materialization
(same mechanism the qfura eval uses elsewhere). Report standard HumanEval
(not HumanEval+) to match the paper.

## Smoke test (this session)

Per finetune script, run:
`MAX_STEPS=2 no_wandb=1 per_device_train_batch_size=1 gradient_accumulation_steps=1`
on a free GPU. Success criteria:
- Mixtral loads in NF4.
- Target layers attach/convert (qlora: nonzero trainable %; qfura: prints
  `installed N BTT layers` with N > 0 covering attn + experts).
- Loss finite across 2 steps.
- `step=2/model.safetensors` (or adapter) written.

Long pole = ~90 GB Mixtral download; surfaced, not hidden. No full epoch, no full
HumanEval eval this session.

## Out of scope (YAGNI)

- Multi-GPU / FSDP (NF4 Mixtral fits one H100).
- MBPP eval (HumanEval only, per request).
- Full 1-epoch training runs.
- Dataset preprocessing (data already on disk in the required format).
