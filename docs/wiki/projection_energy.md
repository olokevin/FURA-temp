# Projection-energy analysis: pretrained W₀ vs gradient G, current weight W′, update ΔW

This page documents the projection-energy tooling added in May 2026 for
characterising how trained weights / gradients / updates align with the
pretrained model's singular subspaces. There are two complementary entry
points:

1. **Offline analysis** of a finished checkpoint vs the base model
   (`analysis/spectral_projection.py` + `analysis/compute_first_step_grad.py`).
2. **Online recording** during a training run
   (`--record_proj_energy` flag in `ref/LIFT/src/finetune_sft.py`).

Both share the same metric definitions and a common interpretation note about
which modules are informative on Llama-3-8B.

---

## 1. Metrics

For each linear-layer weight matrix `X` (a stand-in for one of `G`, `W'`, or
`ΔW = W' − W₀`) with thin SVD `X = U S Vᵀ`, and the pretrained weight's thin
SVD `W₀ = U₀ S₀ V₀ᵀ`:

| # | Name | Definition | Cheap form (used in code) |
|---|---|---|---|
| 1 | **U-projection energy** | `‖U₀ U₀ᵀ X‖_F / ‖X‖_F` | `‖U₀ᵀ X‖_F / ‖X‖_F` (U₀ has orthonormal columns) |
| 2 | **V-projection energy** | `‖X V₀ V₀ᵀ‖_F / ‖X‖_F` | `‖X V₀‖_F / ‖X‖_F` |
| 3 | Singular-value diff | `S − S₀` (vector) and `‖S − S₀‖₂` | direct |
| 4 | Per-column overlap | `\|⟨U₀[:,i], U[:,i]⟩\|`, `i = 0..r−1` | `(U₀ * U).sum(0).abs()` |
| 5 | Per-row overlap | `\|⟨V₀[i,:], V[i,:]⟩\|`, `i = 0..r−1` | `(V₀ᵀ * Vᵀ).sum(1).abs()` |

`r = min(out, in)`. We always use `torch.linalg.svd(..., full_matrices=False)`
in fp32 on GPU.

### Sign ambiguity

Signs of singular vectors are arbitrary; we take `.abs()` of inner products in
metrics (4)/(5). For the projection energies (1)/(2) sign is irrelevant —
they involve orthogonal projectors `U₀ U₀ᵀ` / `V₀ V₀ᵀ`.

### Near-degenerate σ caveat

When two consecutive `S₀[i] ≈ S₀[i+1]`, the columns `U₀[:,i]` and `U₀[:,i+1]`
are only defined up to rotation in their 2-D eigenspace, so per-column dot
products in (4)/(5) can swing wildly even when the **subspace** is preserved.
For a cleaner subspace-level metric use
`compute_principal_angles` in `analysis/analyze_weights.py`.

---

## 2. Trivial-case interpretation note (Llama-3-8B)

With the **full** `U₀` (not a top-k truncation), metric (1) collapses to 1.0
whenever `U₀` spans the full output space — i.e. whenever
`out_features ≤ in_features`. For Llama-3-8B (hidden 4096, intermediate 14336):

| Module | shape (out × in) | metric (1) U-proj | metric (2) V-proj |
|---|---|---|---|
| `q_proj`, `k_proj`, `v_proj`, `o_proj` | 4096 × 4096 (square) | trivial 1.0 | trivial 1.0 |
| `gate_proj`, `up_proj` | 14336 × 4096 | **informative** | trivial 1.0 |
| `down_proj` | 4096 × 14336 | trivial 1.0 | **informative** |

**Convention used everywhere in this codebase:**
- Compute & plot metric (1) only for `gate_proj`, `up_proj`.
- Compute & plot metric (2) only for `down_proj`.
- Compute & plot metrics (3)–(5) for all 7 module types.

(The `analysis/spectral_projection.py` offline tool actually does compute the
trivial cases too, so the interpretation note in `docs/exp_results/motivating.md`
explicitly highlights the flat-1.0 curves; the in-training recorder in
`finetune_sft.py` skips them entirely to save GPU time.)

### Random-Gaussian baseline

For an iid-Gaussian matrix of the same shape as a rectangular module, the
expected u-proj / v-proj is **`√(out/in)` for the informative dimension**. On
Llama-3-8B that's `√(4096/14336) ≈ 0.5345`. Empirically observed value: 0.5343
(matches to 1e-4).

This is the reference value G and ΔW typically sit *near* — meaning gradients
and updates live in approximately a random subspace from W₀'s perspective.
W' itself sits at ≈ 0.9994 because it is dominated by W₀.

---

## 3. Offline analysis (one-shot, end-of-training)

### Files

- `analysis/compute_first_step_grad.py` — replicates the first training step
  on a single GPU (same seed/batch/grad-accum/bf16/grad-ckpt as
  `ref/LIFT/src/finetune_sft.py`) and saves G as a `.pt` dict keyed by HF
  state-dict names (`model.layers.{L}.{self_attn|mlp}.{module}.weight`). Idempotent (won't recompute if output exists; `--overwrite` to force).
- `analysis/spectral_projection.py` — main tool. Loads W₀ + W' + G, computes
  all 5 metrics for all (layer, module) pairs, saves an `.npz` cache and
  PNG figures, writes a markdown report.

Both reuse loaders from `analysis/analyze_weights.py`
(`load_safetensors_index`, `load_tensor`, `detect_num_layers`,
`get_base_weight_key`, `TARGET_MODULES`) and from
`analysis/plot_motivation_figure.py` (`load_checkpoint`, `get_weight`,
`resolve_base_model`, the rcParams styling block, color constants).

### Reproduce

```bash
# 1. Compute G (~3 min on H100, ~50 GB peak VRAM with grad-ckpt)
CUDA_VISIBLE_DEVICES=0 uv run python analysis/compute_first_step_grad.py \
    --base-model meta-llama/Meta-Llama-3-8B \
    --data-path ref/LIFT/LLM-Adapters/ft-training_set/commonsense_170k.json \
    --seed 43 \
    --output analysis_results/grad_step1__llama3-8b__seed43.pt

# 2. Spectral analysis + plots + report (~22 min on H100; 224 modules × 4 SVDs each)
CUDA_VISIBLE_DEVICES=0 uv run python analysis/spectral_projection.py \
    --base-model meta-llama/Meta-Llama-3-8B \
    --checkpoint /data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/full-lr_1e-5-seed_43/last \
    --gradient analysis_results/grad_step1__llama3-8b__seed43.pt
```

### Outputs

- `analysis_results/spectral_projection__<run_tag>.npz` — full metric cache
  (re-plot without recomputing SVD via `--reuse-cache`).
- `docs/exp_results/figs/motivating_uproj_vs_layer.png` — 1×3 (G/W'/ΔW),
  curves for `up_proj` + `gate_proj`.
- `docs/exp_results/figs/motivating_vproj_vs_layer.png` — 1×3, curves for `down_proj`.
- `docs/exp_results/figs/motivating_detail_layer{0,15,31}_{G,Wp,dW}.png` — 9
  detail figures, each a 7×3 grid (rows = 7 modules; cols = `S − S₀`,
  `|⟨U₀,U⟩|`, `|⟨V₀,V⟩|`).
- `docs/exp_results/motivating.md` — auto-written report with setup,
  metrics, interpretation note, embedded figures, discussion, verification
  appendix, reproduction commands.

### Verification (printed at end of run, embedded in report)

1. **Identity check** (W = W₀): max deviation across `u_dot−1`, `v_dot−1`,
   `S − S₀` should be ≈ 1e-3 (float32 noise on 4096-dim inner products plus
   sign / basis-rotation ambiguity within near-degenerate σ₀ eigenspaces).
2. **Square-attn trivial check** (u/v_proj of ΔW on q/k/v/o_proj): every value
   should be in `[0.999, 1.001]`.
3. **Random-Gaussian baseline** for `gate_proj` u_proj and `down_proj`
   v_proj: should match `√(4096/14336) ≈ 0.5345` to 1e-3.

### Cost reference (Llama-3-8B, single H100 NVL)

- `compute_first_step_grad.py`: ~3 min, ~50 GB VRAM.
- `spectral_projection.py`: ~22 min total, ~50 GB VRAM peak. Per-module SVD
  cost: ~1 s for 4096×4096, ~2 s for 4096×14336. Total = 32 layers × 7
  modules × 4 SVDs ≈ 900 SVDs.

---

## 4. Online recording (during training)

The CLI flag `--record_proj_energy` on `ref/LIFT/src/finetune_sft.py` records
the U/V projection energies of W'/G/ΔW at every `--proj_record_interval`
optimizer steps (default = `--logging_steps`).

### CLI flags

```
--record_proj_energy                              # enable recorder
--proj_record_layers 0,15,31                      # comma-separated layer indices
--proj_record_modules gate_proj,up_proj,down_proj # comma-separated module names
--proj_record_interval 0                          # 0 → use --logging_steps
```

Default layer/module choice (`0,15,31` × `gate/up/down_proj`) gives 9 tracked
weights — only the *informative* metric is computed for each (u_proj for
gate/up_proj, v_proj for down_proj), so 9 scalars × 3 quantities (W', G, ΔW)
= **27 scalars per record**. JSONL row size ≈ 1 KB.

### Implementation (class `ProjEnergyRecorder` in `finetune_sft.py`)

- **At init** (after `accelerator.prepare`): for each tracked (layer, module):
  - SVD pretrained weight on GPU in fp32.
  - Cache `U₀` (for u-proj modules) and/or `V₀ᵀ` (for v-proj modules) on **CPU**.
  - Cache full `W₀` on CPU in fp32 (so ΔW = W' − W₀ stays exact across steps).
  - Steady-state extra GPU memory after init: 0 (everything spilled to CPU).
- **Per record** (called inside `accelerator.accumulate(model)` block, between
  `optimizer.step()` and `optimizer.zero_grad()`):
  - For each tracked module: copy the relevant `U₀` / `V₀ᵀ` and `W₀` to GPU,
    read the live `param.weight` (= W') and `param.grad` (= G), compute the 6
    cheap norms, append one JSON row to `proj_energy.jsonl`.
  - Step 0 (called before any backward) intentionally has G/ΔW as `NaN` — not
    a bug. Subsequent records have all 27 fields populated.
- **At end of training** (after `sysmon.dump`): main process re-reads
  `proj_energy.jsonl`, writes `proj_energy.csv`, and renders
  `proj_energy.png` — a 2 × 3 grid (rows = U/V proj; cols = W' / G / ΔW).

### Outputs (under `--output_dir`)

- `proj_energy.jsonl` — append-only, crash-safe (one record per row).
- `proj_energy.csv` — sorted-column dump after training finishes.
- `proj_energy.png` — 2×3 plot at 200 DPI.

### Cost & memory

- CPU RAM per (layer, module) tracked: ~67 MB (W₀) + up to ~234 MB (U₀) +
  up to ~67 MB (V₀ᵀ) — for 9 default modules ≈ 2.5 GB CPU.
- GPU per record: ~1 s additional time, ~500 MB transient allocation that
  gets freed before next training step. Negligible vs the ~5 s/step training
  cost on Llama-3-8B.

### Where the record call is inserted (critical for getting G)

```python
with accelerator.accumulate(model):
    outputs = model(**batch)
    accelerator.backward(outputs.loss)
    optimizer.step()
    if proj_recorder is not None and accelerator.sync_gradients \
            and (args.completed_steps + 1) % proj_interval == 0:
        proj_recorder.record(step=args.completed_steps + 1)  # ← BEFORE zero_grad
    optimizer.zero_grad()
```

If you call `record()` *after* `optimizer.zero_grad()` (which the very first
implementation did), G is `None` everywhere — the `param.grad` field gets
wiped. Keep the record call between `step()` and `zero_grad()`.

### Example launch

```bash
CUDA_VISIBLE_DEVICES=1 uv run accelerate launch \
    --num_machines 1 --num_processes 1 --mixed_precision=bf16 \
    ref/LIFT/src/finetune_sft.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
    --max_seq_len 2048 --learning_rate 1e-5 --num_train_epochs 3 \
    --seed 43 --instruction_type single --load_last_model \
    --data_path LLM-Adapters/ft-training_set/math_10k.json \
    --record_proj_energy \
    --proj_record_layers 0,15,31 \
    --proj_record_modules gate_proj,up_proj,down_proj \
    --output_dir /data/yequan/fura/lift/math/.../full-lr_1e-5-seed_43-projE
```

---

## 5. Empirical reference values (Llama-3-8B)

Baselines collected on the published full-FT commonsense and math checkpoints
at lr=1e-5, seed 43:

| Quantity | u_proj / v_proj | What it means |
|---|---|---|
| `W'` (current weight after training) | ≈ **0.9994** | W' is dominated by W₀; subspaces inherited almost perfectly. |
| `G` (per-step gradient) | ≈ **0.43–0.60** | Per-step direction is ~random vs W₀ (random baseline is 0.534); some modules / layers show modest alignment. |
| `ΔW = W' − W₀` (total update) | ≈ **0.515–0.535** | Cumulative update lives ~near the random baseline — it is *not* low-rank in W₀'s top subspace. This is the LoRA-relevant observation. |

A pure-Gaussian random matrix at the same shape gives 0.5345 — useful sanity
reference.

---

## 6. Pointers to related code

- **Reusable weight loaders & spectral primitives**: `analysis/analyze_weights.py`
- **Existing motivation-figure script** (used as the styling template):
  `analysis/plot_motivation_figure.py` (rcParams block, color constants,
  `load_checkpoint`, `resolve_base_model`).
- **In-training recorder class**: `ref/LIFT/src/finetune_sft.py`
  (search for `class ProjEnergyRecorder`).
- **Standalone gradient collector**: `analysis/compute_first_step_grad.py`.
- **Cleaner subspace-level metric (alternative to per-column dots)**:
  `compute_principal_angles` in `analysis/analyze_weights.py`.
- **Auto-generated offline report**: `docs/exp_results/motivating.md`.

---

## 7. When to use which

| Question | Use |
|---|---|
| "Is the trained W' close to W₀'s subspace?" | offline, metric (1)/(2) on W' |
| "Is the gradient at the start of training high-rank vs W₀?" | offline, metric (1)/(2) on G; or online recorder, step 1 |
| "How does the projection energy of ΔW evolve over training?" | online recorder (gives time series) |
| "Are top singular vectors preserved more than trailing ones?" | offline, metric (4)/(5) detail panels |
| "What's the principal-subspace overlap between W₀ and W'?" | `analyze_weights.compute_principal_angles` (cleaner than per-column dots) |
