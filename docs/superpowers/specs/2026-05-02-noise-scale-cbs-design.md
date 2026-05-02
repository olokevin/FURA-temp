# Noise-Scale Proxy for Critical Batch Size — Design

**Date:** 2026-05-02
**Status:** Draft (awaiting user approval)
**Owner:** yequan

## 1. Goal and Scope

### Goal

Measure the McCandlish "simple noise scale" `B_simple = tr(Σ) / ‖G‖²` on the trainable parameters of LoRA and BlockTT during a 1-epoch `commonsense_170k` finetune of `meta-llama/Meta-Llama-3-8B`, so we can compare the predicted critical batch size (CBS) for each method's natural parameterization.

Reference: McCandlish, Kaplan, Amodei et al., *An Empirical Model of Large-Batch Training* (arXiv:1812.06162).

### In scope

- A standalone `NoiseScaleProbe` utility module under `ref/LIFT/src/utils/noise_scale.py`. Every `every_n_steps` optimizer steps, runs a non-destructive per-example backward pass on a single batch of `B = 16` examples to estimate `tr(Σ)`, `‖G‖²`, and `B_simple` in the trainable-parameter space (option B1 from brainstorming).
- Probe call-sites in `ref/LIFT/src/finetune_lora.py` and `ref/LIFT/src/finetune_blocktt.py`, gated on a CLI flag (off by default; existing runs unchanged).
- Two new launcher shell scripts (`finetune_commonsense_lora_cbs.sh`, `finetune_commonsense_blocktt_cbs.sh`) that flip the flag on, set 1 epoch, and stamp run dirs distinctively.
- A standalone plot script (`ref/LIFT/scripts/plot_noise_scale.py`) that loads JSONL from one or more run dirs and produces a multi-subplot comparison figure.

### Out of scope

- Other LIFT methods (qlora, qfura, qdora, s2ft, etc.). The probe module is generic and can be wired into those scripts later, but spec-level support is LoRA + BlockTT only.
- The "equivalent full-weight gradient" projection variant (option B2 from brainstorming).
- Multi-batch-size sweep estimator (option C from brainstorming).
- Auto-plot at end of training. Plotting is strictly standalone, invoked manually after runs finish.
- Math-task variants. Commonsense_170k only.
- Multi-GPU. The probe asserts `accelerator.num_processes == 1`.

### Compatibility constraint

When `--track_noise_scale` is **disabled**, training trajectory is bit-identical to the existing scripts (no change to RNG, dataloader iteration, or optimizer state).

When `--track_noise_scale` is **enabled**, training trajectory is still bit-identical: the probe consumes its own dataloader cursor (separate shuffle generator over the same dataset) and saves/restores `param.grad` around each probe so the optimizer never sees probe-induced state.

## 2. Estimator Math and Numerical Stability

### Definitions

At a probe step we draw a batch of `B = 16` examples and compute per-example trainable-parameter gradients `g_1, …, g_B`, each conceptually a flat vector formed by concatenating the gradient of every `requires_grad=True` parameter in the model. Let `ḡ = (1/B) Σᵢ gᵢ`.

### Estimators (McCandlish et al., Appendix A.1)

For a single batch of size `B`:

```
tr_sigma_hat    = (1 / (B − 1)) · Σᵢ ‖gᵢ − ḡ‖²            # unbiased estimate of tr(Σ)
g_norm_sq_hat   = ‖ḡ‖² − tr_sigma_hat / B                 # bias-corrected estimate of ‖G‖²
g_norm_sq_naive = ‖ḡ‖²                                    # uncorrected, for sanity check
b_simple_hat    = tr_sigma_hat / max(g_norm_sq_hat, eps)
```

Identity used: `E[‖ḡ‖²] = ‖G‖² + tr(Σ)/B`, so subtracting `tr_sigma_hat / B` gives an unbiased estimator of `‖G‖²`.

### Online accumulation reformulation

We do **not** materialize per-example flat vectors. Instead, parameter-by-parameter:

```
sum_g_sq_per_example  ← scalar (fp32):  accumulates Σᵢ ‖gᵢ‖²  over i = 1..B
sum_g[p]              ← per-param fp32 buffer: accumulates Σᵢ gᵢ[p]  over i = 1..B
```

After all `B` per-example backwards:

```
‖ḡ‖²       = (1 / B²) · Σ_params ‖sum_g[p]‖²
Σᵢ ‖gᵢ − ḡ‖² = sum_g_sq_per_example − B · ‖ḡ‖²        # variance identity
```

Memory cost: one extra grad-shaped fp32 buffer per trainable parameter. For LoRA r=64 on Llama-3-8B with the default 5-module target set, this is on the order of tens of MB; for BlockTT it depends on `decomp_mode` but is comparable. Buffers are allocated lazily on first probe and reused.

### Numerical-stability rules

1. **Accumulate in fp32.** Both `sum_g[p]` and `sum_g_sq_per_example` are fp32, even when `param.grad` is bf16 (cast on the fly).
2. **`g_norm_sq_hat` clipping for the ratio only.** Late in training, `‖G‖²` can be smaller than `tr(Σ)/B`, making the bias-corrected estimate negative. We:
   - Log the raw (possibly negative) `g_norm_sq_hat` so the user sees it.
   - For the `b_simple` ratio, clip the denominator to `max(g_norm_sq_hat, 1e-12)`.
   - Emit a `b_simple_valid = (g_norm_sq_hat > 0)` boolean flag so the plot script can mask invalid points.
3. **`B = 1` guard.** Construction asserts `probe_batch_size ≥ 2`.
4. **No probe-internal smoothing.** Each row is the raw single-batch estimate. Smoothing is the plot script's job.

### JSONL row schema

One JSON object per line, one line per probe:

```json
{
  "step": 4200,
  "epoch_frac": 0.395,
  "probe_batch_size": 16,
  "tr_sigma": 12.34,
  "g_norm_sq": 0.087,
  "g_norm_sq_naive": 0.858,
  "b_simple": 141.8,
  "b_simple_valid": true,
  "loss_at_probe": 0.612,
  "num_trainable_params": 33554432,
  "wall_time_s": 7.21
}
```

Error rows replace the metric fields with `null` and add an `"error"` field; see Section 3.

## 3. Probe Execution Semantics

### Entry point

```python
NoiseScaleProbe.maybe_run(step: int, total_steps: int) -> dict | None
```

Called once per optimizer step from the training loop, inside the `if accelerator.sync_gradients:` block, immediately after `optimizer.zero_grad()`. Returns the JSONL row dict on probe steps, `None` otherwise. Fires when `step > 0 and step % every_n_steps == 0`.

### Probe-internal flow

1. **Save state.** Snapshot `model.training`, and capture each trainable param's current `.grad` into a temporary fp32 buffer `saved_grads[p]`. (At the call site `param.grad` has just been zeroed by `optimizer.zero_grad()`, so this is cheap; we save anyway to be robust to call-site refactors.)
2. **Reset internal accumulators.** Zero `sum_g_sq_per_example` and zero each `sum_g[p]` buffer.
3. **Draw probe batch.** Pull next batch from the **probe dataloader iterator** (separate from training's iterator). Split into `B` single-example sub-batches.
4. **Per-example backward × B.**
   - For each `i`: `model.zero_grad(set_to_none=False)`, forward + plain `loss.backward()` (not `accelerator.backward()`) on `batch[i:i+1]`.
   - Walk trainable params: cast `p.grad` to fp32, accumulate `‖p.grad‖²` into `sum_g_sq_per_example`, add `p.grad` into `sum_g[p]`.
   - Track `loss_sum_at_probe += loss.item()` for `loss_at_probe`.
5. **Compute statistics.** As in Section 2.
6. **Restore state.** `model.zero_grad(set_to_none=False)`; copy each `saved_grads[p]` back into `param.grad`; restore training-mode flag.
7. **Return row** (and write to JSONL + wandb sinks).

### Invariants

- The probe runs **outside** the `accelerator.accumulate(model)` context. Its per-example backwards are pure PyTorch.
- Probe dataloader has independent shuffle seed (`seed = main_seed + 7919`) and is disjoint in iteration order from training, so probing on/off does not change which examples training sees.
- Model stays in `train()` mode during the probe. Dropout is a legitimate part of the gradient noise we are measuring.
- Gradient checkpointing remains on. Per-example checkpointed backwards are slower but still cheap relative to a training step.

### Failure modes

- **OOM during per-example backward.** With per-example batch = 1 and `max_seq_len = 2048` this is much cheaper than a training step. If it occurs, catch the exception, restore state from `saved_grads`, write a row with all metrics `null` and `"error": "oom"`, return.
- **NaN/Inf in any per-example grad.** Detect post-backward via `torch.isfinite(p.grad).all()` per param. On any non-finite, restore state, write a row with metrics `null` and `"error": "non_finite_grad"`, continue.
- **Probe dataloader exhausted.** Won't happen at our scale (~53 probes × 16 = 848 examples, dataset size 170k), but guard by re-creating the iterator on `StopIteration`.

## 4. Module API and Integration

### File: `ref/LIFT/src/utils/noise_scale.py`

```python
class NoiseScaleProbe:
    def __init__(
        self,
        model: torch.nn.Module,
        accelerator: Accelerator,
        probe_dataset: torch.utils.data.Dataset,
        collate_fn: Callable,
        probe_batch_size: int = 16,
        every_n_steps: int = 200,
        seed: int = 1234,
        output_path: str | os.PathLike | None = None,
        log_to_wandb: bool = True,
        device: torch.device | None = None,
    ): ...

    def maybe_run(self, step: int, total_steps: int) -> dict | None: ...

    def close(self) -> None: ...
```

### Construction details

- Builds an internal `DataLoader(probe_dataset, batch_size=probe_batch_size, shuffle=True, generator=Generator().manual_seed(seed), num_workers=0, collate_fn=collate_fn)` and holds an iterator over it.
- `output_path` is opened append-mode, line-buffered. Each row written + flushed immediately so a crash mid-run preserves prior probes.
- `log_to_wandb=True` is best-effort: silently no-op if wandb isn't initialized. wandb keys: `noise_scale/b_simple`, `noise_scale/tr_sigma`, `noise_scale/g_norm_sq`, `noise_scale/g_norm_sq_naive`, `noise_scale/loss_at_probe`, `noise_scale/probe_wall_time_s`. Step = optimizer step.
- `sum_g[p]` buffers allocated lazily on first probe.
- Asserts `accelerator.num_processes == 1` at construction; raises `RuntimeError` otherwise.

### Training-script integration (LoRA)

Four edits to `ref/LIFT/src/finetune_lora.py`:

**a. CLI flags.** In the argparse section:

```python
parser.add_argument("--track_noise_scale", action="store_true",
                    help="Enable McCandlish noise-scale probe.")
parser.add_argument("--noise_scale_every_n_steps", type=int, default=200,
                    help="Probe cadence in optimizer steps.")
parser.add_argument("--noise_scale_probe_batch_size", type=int, default=16,
                    help="Per-probe batch size B.")
```

**b. Probe construction.** After model/optimizer/dataloader are built and before the training loop begins:

```python
noise_probe = None
if args.track_noise_scale and accelerator.is_main_process:
    from utils.noise_scale import NoiseScaleProbe
    noise_probe = NoiseScaleProbe(
        model=model,
        accelerator=accelerator,
        probe_dataset=train_dataset,
        collate_fn=train_dataloader.collate_fn,
        probe_batch_size=args.noise_scale_probe_batch_size,
        every_n_steps=args.noise_scale_every_n_steps,
        seed=args.seed + 7919,
        output_path=os.path.join(args.output_dir, "noise_scale.jsonl"),
    )
```

**c. Probe call.** Inside the training loop, immediately after `optimizer.zero_grad()` (which itself runs only when `accelerator.sync_gradients` is True):

```python
if noise_probe is not None:
    noise_probe.maybe_run(args.completed_steps, args.max_train_steps)
```

**d. Cleanup.** After the training loop ends:

```python
if noise_probe is not None:
    noise_probe.close()
```

### Training-script integration (BlockTT)

Four structurally analogous edits to `ref/LIFT/src/finetune_blocktt.py`. Probe is parameterization-agnostic: it operates only on `requires_grad=True` params, which `configure_blocktt_trainability` has already set correctly.

## 5. Launcher Shell Scripts and Plot Script

### `ref/LIFT/bash_scripts/finetune_commonsense_lora_cbs.sh`

Copy of `finetune_commonsense_lora.sh` with these deltas:

- Override hardcoded `--num_train_epochs 3` to `--num_train_epochs 1`.
- Append flags to the `accelerate launch` invocation:
  `--track_noise_scale --noise_scale_every_n_steps 200 --noise_scale_probe_batch_size 16`.
- `OUTPUT` and `run_name` get a `-cbs` suffix to avoid collision with regular runs.
- Default `wandb_project` becomes `commonsense-${model_tag}-cbs` (separate project keeps these grouped).
- Skip the auto-eval block at the end. These runs are for measurement, not benchmark scores. Implementation: gate the `if [ "${MAX_STEPS}" = "0" ]` block behind a `RUN_EVAL=${RUN_EVAL:-0}` check.
- Retain `--load_last_model` (per CLAUDE.md).
- Retain `--per_device_train_batch_size 8 --gradient_accumulation_steps 2` so the training effective batch is 16, matching `probe_batch_size = 16`.

### `ref/LIFT/bash_scripts/finetune_commonsense_blocktt_cbs.sh`

Copy of `finetune_commonsense_blocktt.sh` with the same kind of deltas as the LoRA CBS script, **plus** these BlockTT-specific defaults (which align with project-wide fura defaults from CLAUDE.md):

```bash
decomp_mode="${decomp_mode:-output_one_block}"
train_position="${train_position:-small}"
s_merged_to="${s_merged_to:-keep_trainable}"
blocktt_rank="${blocktt_rank:-full}"
trainable_type="${trainable_type:-all}"
```

**Note:** This spec does **not** modify the existing `finetune_commonsense_blocktt.sh`. That script's stale defaults (`input_one_block` / `frozen`) are a separate concern.

### `ref/LIFT/scripts/plot_noise_scale.py`

Standalone plotting script.

**CLI:**

```
python plot_noise_scale.py \
    --runs <path1> <path2> ... \
    [--labels "LoRA r=64" "BlockTT" ...] \
    --out figures/noise_scale_comparison.png \
    [--smooth_window 5] \
    [--mask_invalid / --no-mask_invalid]
```

**Behavior:**

- For each run path, load `<path>/noise_scale.jsonl` into a `pandas.DataFrame`.
- Produce a single figure with **5 subplots in a 2×3 grid** (last cell blank): one subplot per metric — `b_simple`, `tr_sigma`, `g_norm_sq`, `g_norm_sq_naive`, `loss_at_probe`. Each subplot overlays one curve per run, x-axis = `step`, legend = `--labels` (defaults to basename of run path).
- `b_simple` subplot: log-scale y-axis. `tr_sigma` and `g_norm_sq_naive` subplots: log-scale y-axis (these typically span decades too). `g_norm_sq` and `loss_at_probe`: linear.
- `--mask_invalid` (default on): drop rows where `b_simple_valid == False` from the `b_simple` and `g_norm_sq` subplots; annotate each subplot with a per-run masked-row count.
- `--smooth_window N` (default 5): plot a moving-average curve in heavier line on top of the raw curve (raw at alpha=0.3, smoothed at alpha=1.0). `N=1` disables smoothing.
- Saves both `<out>.png` (150 DPI) and `<out>.pdf`.
- Stdout summary, one line per run: `LoRA r=64: 53 probes, B_simple range [12, 1840], median 421, masked 2/53`.

**Failure modes:**

- Missing `noise_scale.jsonl`: error with explicit path. Exit code 2.
- Empty JSONL: warn and skip that run; continue with the remaining runs.
- Mismatched run lengths: plot all curves on the same `step` axis. Don't try to align.

## 6. Testing

Three test files under `ref/LIFT/src/tests/`.

### `test_noise_scale_estimator.py` — pure-math unit tests

- **Synthetic Gaussian gradients.** Sample `g_i = G + ε_i` with `ε_i ∼ N(0, σ² I)` for chosen `G, σ, d`. Verify `tr_sigma_hat ≈ d · σ²` and `g_norm_sq_hat ≈ ‖G‖²` to within 3 standard errors over 1000 trials.
- **Identical-gradient edge case.** All `g_i` equal: `tr_sigma_hat == 0`, `g_norm_sq_hat == ‖ḡ‖²`, `b_simple_hat == 0`.
- **Bias-correction-goes-negative case.** `‖G‖² < tr(Σ)/B`: `b_simple_valid == False`, `tr_sigma` and `g_norm_sq_naive` still finite, `b_simple` denominator clipped (no NaN).

### `test_noise_scale_probe_compat.py` — integration test on a tiny model

- 2-layer MLP with ~1000 trainable params, synthetic dataset.
- Run `N` real training steps with probing **disabled**, snapshot final weights.
- Run identical `N` steps with probing **enabled** (same seed). Assert weight tensors are bit-identical.
- Assert JSONL has `floor(N / every_n_steps)` rows; each row has all expected keys with valid types.

### `test_noise_scale_cli.py` — argparse smoke test

- Import the argparse parser from `finetune_lora.py` and `finetune_blocktt.py`. Parse a fake command line including the three new flags. Assert the namespace has the expected attributes with expected types.

The plot script is not unit-tested. Manual verification on a real or synthetic JSONL.

## 7. File Manifest

**New files:**

- `ref/LIFT/src/utils/noise_scale.py`
- `ref/LIFT/scripts/plot_noise_scale.py`
- `ref/LIFT/bash_scripts/finetune_commonsense_lora_cbs.sh`
- `ref/LIFT/bash_scripts/finetune_commonsense_blocktt_cbs.sh`
- `ref/LIFT/src/tests/test_noise_scale_estimator.py`
- `ref/LIFT/src/tests/test_noise_scale_probe_compat.py`
- `ref/LIFT/src/tests/test_noise_scale_cli.py`

**Modified files:**

- `ref/LIFT/src/finetune_lora.py` — add three CLI flags, probe construction, probe call inside training loop, probe close after loop.
- `ref/LIFT/src/finetune_blocktt.py` — same four edits, structurally analogous locations.

**Unchanged files (explicitly):**

- `ref/LIFT/bash_scripts/finetune_commonsense_lora.sh`
- `ref/LIFT/bash_scripts/finetune_commonsense_blocktt.sh`

## 8. Open Questions

None at design time. Implementation may surface details (e.g. exact `args.max_train_steps` attribute name across the two finetune scripts), to be resolved during plan execution.
