# SVD-LLM-v2 compressed full-FT on MATH-10K → gsm8k eval

## Goal

Run six MATH-10K fine-tunings (3 epochs, lr=1e-4, eval gsm8k only) to compare:

- **Group A — uncompressed baselines**
  - **A1**: `meta-llama/Llama-3.2-1B` full FT
  - **A2**: `meta-llama/Llama-3.2-3B` full FT
- **Group B — Llama-3.2-3B compressed to ~2B params, then full FT**
  - **B1**: compression via `svd_llm_v2`
  - **B2**: compression via `svd_llm_v2_combined`
- **Group C — Llama-3.2-3B compressed to ~1B params, then full FT**
  - **C1**: compression via `svd_llm_v2`
  - **C2**: compression via `svd_llm_v2_combined`

All runs report to wandb project `compress_sft_math_llama3.2`.

Output root: `/data/yequan/compress_train/sft/math/<MODEL>/<run_name>` (mirrors
existing `OUTPUT_SRC_DIR/math/<MODEL>/<run_name>` structure).

## Background and current state

- `src/compress/decomposition.py` already supports the two SVD modes
  (`svd_llm_v2`, `svd_llm_v2_combined`) via `decompose_with_loader`.
- `src/compress/svd/svd_llm_v2.py` performs the actual decomposition: replaces
  each `nn.Linear` with `SVDCompressedLinear` (two stacked `nn.Linear`s,
  `svd_a` and `svd_b`). `svd_llm_v2_combined` requires both input-activation
  AND output-gradient covariances.
- `compress_integration.py` (repo root) wires calibration into LIFT training
  scripts. Today its `CALIB_MODE_TO_TRAIN_MODE` mapping covers BTT variants
  only (`v2`, `v2_bp`, `v2_combined`, `twosteps`). `validate_calibrated_btt_args`
  also rejects any `train_mode != "blocktt"`.
- `ref/LIFT/src/finetune_sft.py` has no compression hooks. It is the
  pure-full-FT training script used by `finetune_math_full.sh`.
- `ref/LIFT/bash_scripts/eval_math.sh` supports `EVAL_DATASETS="gsm8k"` to
  limit eval to gsm8k only, and auto-resolves `$OUTPUT/last` for checkpoints
  saved with `--load_last_model`.
- Models are cached locally at
  `/data/ryan/huggingface_cache/hub/models--meta-llama--Llama-3.2-{1B,3B}`.
- MATH-10K training set is at
  `/data/ruijiezhang/llm-adapter_bp/LLM-Adapters/ft-training_set/math_10k.json`.
- 8× H100 NVL (~57 GB free each) available.

## Hyperparameters (shared across all six runs)

| Param | Value |
|---|---|
| `learning_rate` | 1e-4 |
| `num_train_epochs` | 3 |
| `per_device_train_batch_size` | 2 |
| `gradient_accumulation_steps` | 8 (effective batch 16) |
| `lr_scheduler_type` | linear |
| `num_warmup_steps` | 0.03 |
| `seed` | 43 |
| `max_seq_len` | 2048 |
| mixed precision | bf16 |
| `--load_last_model` | yes (project convention) |
| eval dataset | `gsm8k` only (`EVAL_DATASETS="gsm8k"`) |

## Calibration settings (Groups B and C only)

| Param | Value |
|---|---|
| `calib_source` | `training_data` (MATH-10K subset, training collate fn) |
| `calib_num_seqs` | 128 |
| `calib_max_length` | 2048 |
| `calib_batch_size` | 4 (small to leave headroom for `svd_v2_combined`'s backward) |
| `calib_seed` | 3 |

## Compression ratio derivation

Llama-3.2-3B parameter accounting (from HF config):

- Total: ~3.21 B
- Embedding + lm_head (tied, skipped by `skip_layers=lm_head`): ~262 M
  (vocab 128256 × hidden 3072 = 0.394 G entries, tied = 1× of that = 0.394 G;
  effective uncompressed = ~262 M after accounting for tying & layernorms)
- Compressible (nn.Linear inside transformer blocks): ~2.95 B

| Target | Computation | `compression_ratio` |
|---|---|---|
| ~2 B total | (2.0 − 0.262) / 2.95 | **0.589** |
| ~1 B total | (1.0 − 0.262) / 2.95 | **0.250** |

The exact compressible-param count will be measured at runtime by enumerating
`nn.Linear` modules (excluding `lm_head`); the launcher logs the achieved
total model param count. Dir name labels: `target_2B`, `target_1B`.

## Code changes

### 1. `compress_integration.py`

Add the two SVD modes to the public CLI-to-train-mode map:

```python
CALIB_MODE_TO_TRAIN_MODE = {
    "v2": "btt_llm_v2",
    "v2_bp": "btt_llm_v2_bp",
    "v2_combined": "btt_llm_v2_combined",
    "twosteps": "btt_twosteps",
    "svd_v2": "svd_llm_v2",                  # NEW
    "svd_v2_combined": "svd_llm_v2_combined", # NEW
}
```

Add a new `--compression_ratio` CLI arg in `add_calibrated_btt_args` (default
`1.0`), used by SVD modes. (`--blocktt_rank` remains the rank knob for BTT
modes.)

Update `validate_calibrated_btt_args`: when `calib_mode.startswith("svd_")`,
- skip the `train_mode == "blocktt"` check (SVD path runs inside the
  full-FT script, not the blocktt script),
- require `0 < compression_ratio <= 1`.

Update `build_decomposition_config`: when `calib_mode` is an SVD mode, take
`compression_ratio` from `args.compression_ratio` instead of inverting
`--blocktt_rank`. BTT-only fields (`decomp_mode`, `train_position`,
`s_merged_to`, `factorize_by_head`) are ignored by the SVD compress path
inside `compress_model_with_loader`, so they can be left at their defaults.

Add helpers:

```python
def apply_calibrated_svd(model, args, *, calib_loader, device=None):
    """Mirror of apply_calibrated_btt for SVD-LLM-v2 modes.
    Decomposes nn.Linear -> SVDCompressedLinear in-place.
    Does NOT freeze any parameters (full-FT after compression)."""
    cfg = build_decomposition_config(args, model=model)
    model = decompose_with_loader(model, cfg, calib_loader=calib_loader,
                                  device=device, return_trainability_stats=False)
    model.requires_grad_(True)  # safety; decompose_with_loader already does this
    return model

@torch.no_grad()
def materialize_svd_to_linear(model):
    """In-place: replace every SVDCompressedLinear with an nn.Linear holding
    the dense product (svd_a @ svd_b). Yields a vanilla HF checkpoint that
    eval_math.sh can load without custom code. Trade-off: the saved
    checkpoint is dense (no deployment-time compression), but training was
    constrained to the low-rank manifold from svd_llm_v2 init."""
    ...
```

### 2. `ref/LIFT/src/finetune_sft.py`

Port the path-fixup block from `finetune_blocktt.py` (lines 12-16) so
`compress_integration` is importable.

After model load and AFTER train_dataset / data_collator are built, but
BEFORE the optimizer is constructed:

```python
from compress_integration import (
    add_calibrated_btt_args, validate_calibrated_btt_args,
    build_calib_loader, apply_calibrated_svd,
    materialize_svd_to_linear,
)

if getattr(args, "calib_mode", "none") != "none":
    calib_loader = build_calib_loader(
        args, tokenizer=tokenizer,
        training_dataset=train_dataset, training_collate_fn=data_collator,
    )
    model = apply_calibrated_svd(model, args, calib_loader=calib_loader)
    print_rank_0(f"[svd-compress] applied {args.calib_mode}, "
                 f"compression_ratio={args.compression_ratio}", args.global_rank)
```

Register the calib args on the parser. Add the import-and-call near top.

At checkpoint-save time (inside the existing best/last save block):

```python
if getattr(args, "calib_mode", "none").startswith("svd_"):
    materialize_svd_to_linear(model)
# then existing save_hf_format() / save_pretrained() runs unchanged
```

### 3. Six shell scripts under `ref/LIFT/bash_scripts/`

Naming:

- `finetune_math_full_a1_llama32_1b.sh`
- `finetune_math_full_a2_llama32_3b.sh`
- `finetune_math_full_b1_svd_v2_2b.sh`
- `finetune_math_full_b2_svd_v2_combined_2b.sh`
- `finetune_math_full_c1_svd_v2_1b.sh`
- `finetune_math_full_c2_svd_v2_combined_1b.sh`

Each:
- Inherits the structure of `finetune_math_full.sh`.
- Sets `OUTPUT_SRC_DIR=/data/yequan/compress_train/sft`.
- Sets `DATA_DIR=/data/ruijiezhang/llm-adapter_bp/LLM-Adapters`.
- Sets `wandb_project=compress_sft_math_llama3.2`.
- Sets `MODEL`, `lr=1e-4`, `seed=43`, `num_train_epochs=3`.
- For B/C: adds `--calib_mode svd_v2` (or `svd_v2_combined`),
  `--calib_source training_data`, `--calib_num_seqs 128`,
  `--calib_max_length 2048`, `--calib_batch_size 4`, `--calib_seed 3`,
  `--compression_ratio 0.589` (B) or `0.250` (C).
- After training, calls
  `EVAL_DATASETS="gsm8k" bash ./bash_scripts/eval_math.sh CKPT=$OUTPUT/last ...`
- Honors `DEVICE=N` env var for `CUDA_VISIBLE_DEVICES`.

### 4. Unit test

Extend `tests/test_compress_integration_build_config.py`:

- `test_build_config_svd_v2`: pass an args namespace with `calib_mode=svd_v2`,
  `compression_ratio=0.5`, dummy model with two `nn.Linear`s; assert
  returned `DecompositionConfig.train_mode == "svd_llm_v2"` and
  `compression_ratio == 0.5`.
- `test_build_config_svd_v2_combined`: same but combined.
- `test_validate_svd_rejects_bad_ratio`: ratio 0 and 1.5 raise.

## Test plan (gated)

1. **Compile check** (cheap): `python -m py_compile compress_integration.py
   ref/LIFT/src/finetune_sft.py`. Must pass.
2. **Unit test**: `python -m unittest
   tests.test_compress_integration_build_config`. Must pass.
3. **Smoke A1**: run A1 script with `MAX_STEPS=10` (uncompressed
   regression guard — confirms `finetune_sft.py` still works when
   `calib_mode=none`). Skip eval. Must complete + save a checkpoint.
4. **Smoke B1**: run B1 script with `MAX_STEPS=10`. End-to-end check of
   decomposition → train → materialize → save → eval. Confirm
   `materialize_svd_to_linear` yields a checkpoint that
   `eval_math.sh` can load on 5 gsm8k examples (`EVAL_DATASETS=gsm8k`
   + `--per_device_eval_batch_size 1` + early break is overkill; just
   let it run a few). Must complete with non-NaN eval accuracy.
5. **Real runs**: only after smoke 1–4 pass. User launches manually
   per `DEVICE=N bash <script>` on chosen GPU.

## Trade-offs and assumptions

- **Dense checkpoint after materialize**: Saved model is full-size, not
  compressed-at-rest. Eval loads it as a vanilla HF model. This is fine for
  the experimental goal (post-FT accuracy of a model whose weights were
  initialized on the SVD-LLM-v2 low-rank manifold). If a future experiment
  needs a deployable compressed checkpoint, add a `--save_factored` flag.
- **All transformer linears compressed uniformly**: `compression_ratio` is
  applied per-layer at rank `int(ratio * d_out * d_in / (d_out + d_in))`.
  This is uniform allocation (SVD-LLM-v2's heterogeneous allocation flag is
  currently disabled in code at `svd_llm_v2.py:223`). Acceptable for this
  experiment; revisit if results are notably worse than expected.
- **`svd_llm_v2_combined` memory**: needs backward pass on calib batches.
  Llama-3.2-3B in bf16 ≈ 6 GB params + grads + activations; with
  `calib_batch_size=4`, max_len 2048 we are comfortably under H100 NVL
  57 GB. Smoke test (step 4) will surface any OOM.
- **Wandb run_id collision**: each script generates its own `wandb_run_id`
  via `python -c 'import wandb; print(wandb.util.generate_id())'`, then
  exports `WANDB_RUN_ID`. eval is launched with the same run_id, mirroring
  `finetune_math_full.sh`.
- **No multi-LR sweep**: per user, fix lr=1e-4 for now. Re-runs at other
  LRs are a follow-up.

## Out of scope

- Multi-seed averaging.
- LR sweeps.
- Saving a factored-form checkpoint (only materialized dense saved).
- Heterogeneous per-layer rank allocation (uniform ratio for now).
- Reusing any existing Llama-3.2 baseline runs under
  `/data/yequan/fura/lift/math/...` — Group A is re-run under the new
  `OUTPUT_SRC_DIR` for path consistency with the compressed runs.
- Extending `run_sft.py` (top-level, not LIFT) with SVD compression —
  this design touches only `ref/LIFT/src/finetune_sft.py`.

## File touchpoints

New files:
- `ref/LIFT/bash_scripts/finetune_math_full_a1_llama32_1b.sh`
- `ref/LIFT/bash_scripts/finetune_math_full_a2_llama32_3b.sh`
- `ref/LIFT/bash_scripts/finetune_math_full_b1_svd_v2_2b.sh`
- `ref/LIFT/bash_scripts/finetune_math_full_b2_svd_v2_combined_2b.sh`
- `ref/LIFT/bash_scripts/finetune_math_full_c1_svd_v2_1b.sh`
- `ref/LIFT/bash_scripts/finetune_math_full_c2_svd_v2_combined_1b.sh`

Modified files:
- `compress_integration.py` (extend map; new `--compression_ratio` arg;
  validate-rule branch for SVD; `apply_calibrated_svd`;
  `materialize_svd_to_linear`).
- `ref/LIFT/src/finetune_sft.py` (import path fix; register calib args;
  call `apply_calibrated_svd` before optimizer; call
  `materialize_svd_to_linear` before save).
- `tests/test_compress_integration_build_config.py` (three new cases).
