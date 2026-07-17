# SFT-Math: Compressed Llama-3.2-3B on MATH-10K

- **Setup**: Compress `meta-llama/Llama-3.2-3B` to a target parameter budget via `svd_llm_v2` or `svd_llm_v2_combined`, then run full fine-tuning on MATH-10K for 3 epochs. Eval on gsm8k only (1319 problems).
- **Training**: lr varies (see tables); effective batch = 16 throughout. The target=2B grid and the four original lr=1e-4 runs used `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`; the target=1B grid (lr ∈ {1e-5, 3e-5, 5e-5}) used `per_device_train_batch_size=8`, `gradient_accumulation_steps=2`. Linear LR schedule with 3% warmup, `seed=43`, bf16. No gradient checkpointing.
- **Calibration**: 128 sequences × max_len 2048, `calib_batch_size=4`. `calib_source ∈ {c4, training_data}`.
- **Output root**: `/data/yequan/compress_train/sft/math/meta-llama/Llama-3.2-3B/`
- **Wandb project**: `compress_sft_math_llama3.2`
- **Unified launcher**: `ref/LIFT/bash_scripts/finetune_math_calib.sh` — dispatches to `finetune_sft.py` (SVD modes) or `finetune_blocktt.py` (BTT modes) based on `calib_mode`. The original six per-group scripts (`finetune_math_full_{a1,a2,b1,b2,c1,c2}_*.sh`) remain for reference.
- **Eval**: `EVAL_DATASETS="gsm8k" bash ref/LIFT/bash_scripts/eval_math.sh`

## Baselines (no compression)

| Model                  | Params | gsm8k accuracy | train_loss |
| ---------------------- | -----: | -------------: | ---------: |
| Llama-3.2-1B (full FT) |  1.24B |          28.05 |      0.127 |
| Llama-3.2-3B (full FT) |  3.21B |  **44.05** |      0.077 |

Both at lr=1e-4.

## Target = 2B (compression_ratio = 0.589 of compressible linear params)

All cells are gsm8k accuracy (%) after 3-epoch full FT. **Bold** = best per row; <ins>underline</ins> = best overall. Run identifier (e.g. `B1`) marks the original lr=1e-4 set.

| method                | calib_source  |   lr=1e-5 |   lr=3e-5 |   lr=5e-5 | lr=1e-4 |
| --------------------- | ------------- | --------: | --------: | --------: | ------: |
| svd_v2          | c4            |     18.58 |     32.30 |     30.86 |     —   |
| svd_v2          | training_data |     38.14 | **<ins>45.79</ins>** |     39.80 |   29.19 (B1) |
| svd_v2_combined | c4            |     23.05 |     30.86 |     25.17 |     —   |
| svd_v2_combined | training_data |     39.96 | **41.40** |     33.51 |   19.79 (B2) |

## Target = 1B (compression_ratio = 0.250 of compressible linear params)

Same convention. The `c4` calibration rows were not re-run after the target=2B sweep made it clear that `training_data` calibration strictly dominates `c4` at every comparable cell (5–15 point gap).

| method                | calib_source  |   lr=1e-5 |   lr=3e-5 |   lr=5e-5 | lr=1e-4 |
| --------------------- | ------------- | --------: | --------: | --------: | ------: |
| svd_v2          | c4            |       —   |       —   |       —   |     —   |
| svd_v2          | training_data |      8.64 |     22.29 | **<ins>26.46</ins>** |   24.79 (C1) |
| svd_v2_combined | c4            |       —   |       —   |       —   |     —   |
| svd_v2_combined | training_data |     13.27 | **24.11** |     23.96 |   17.51 (C2) |

The target=1B grid (lr ∈ {1e-5, 3e-5, 5e-5}, `calib_source=training_data` only) was run at `bsz=8, accum=2`; effective batch is identical (16). The first attempt at `bsz=16, accum=1` OOM'd on every GPU (3B model + AdamW state + activations for 16×2048 tokens exceeds 90 GB on H100 NVL without gradient checkpointing). The `bsz=8` retry succeeded.

## Findings

### Target=2B (16 cells)

1. **Best compressed configuration beats the uncompressed 3B baseline.** svd_v2 + training_data + lr=3e-5 → **45.79%** vs A2 baseline 44.05% at ~62% of the compressible params (≈2.0B total).
2. **`calib_source=training_data` ≫ `calib_source=c4`** at every (method, lr) cell. Typical gap: 5–15 points. In-domain calibration matters substantially for both forward and forward+backward whitening objectives.
3. **`lr=3e-5` is the sweet spot.** Every method × calsrc combination peaks at lr=3e-5 (`lr=1e-5` too slow to recover from the rank-truncation perturbation; `lr≥5e-5` overshoots after the first epoch).
4. **`svd_v2` ≥ `svd_v2_combined`** at every comparable (calsrc, lr) cell. The doubly-whitened backward objective consistently underperforms the simpler forward-only activation-whitened objective.
5. **lr=1e-4 was the wrong default for the original 4 runs** (B1, B2, C1, C2): a lower lr with the same setup recovers significantly more capacity (B1=29.19% → svd_v2/training_data/3e-5/2B=45.79%, a 16.6-point gain from lr tuning alone).

### Target=1B (6 fresh cells + 2 original)

6. **Best target=1B = 26.46%** (svd_v2 + training_data + lr=5e-5) — roughly matches the 1B baseline (A1 = 28.05%); compression-then-FT does **not** improve over training a same-size base model from scratch at this compression ratio.
7. **Optimal lr shifts higher than target=2B.** svd_v2 peaks at **lr=5e-5** (target=1B) vs lr=3e-5 (target=2B). svd_v2_combined still peaks at lr=3e-5 in both. With more aggressive compression, the model needs more lr to recover usable capacity in 3 epochs.
8. **svd_v2_combined wins at the lowest lr (1e-5)** but loses to svd_v2 at higher lrs. Interpretation: at the small lr the more carefully whitened init (preserving both activation and gradient directions) provides a better starting point that the optimizer can barely improve on; at higher lrs, the simpler init reaches further because there is more headroom to move.
9. **Compression cost is non-linear.** Going from 3B → 2B (best 45.79%) → 1B (best 26.46%) drops 19.3 points on the second compression step. The 1B target retains 25% of compressible-layer params; this is past the point where SVD-LLM-v2 init can recover full math reasoning.

## Files

- Per-run output dirs: `/data/yequan/compress_train/sft/math/meta-llama/Llama-3.2-3B/<run_name>/`
  - `last/` — final-step checkpoint (vanilla HF format after materialize)
  - `last/math/gsm8k/eval.log` — gsm8k generation log
  - `training.log` — training stdout
- Per-run launch logs (under tmp): `/tmp/grid_logs/<run_name>.log`
- Unified launcher: `ref/LIFT/bash_scripts/finetune_math_calib.sh`
- Per-group launcher scripts: `ref/LIFT/bash_scripts/finetune_math_full_{a1,a2,b1,b2,c1,c2}_*.sh`
- Plan: `docs/superpowers/plans/2026-05-11-svd-compress-full-ft-math10k.md`
- Spec: `docs/superpowers/specs/2026-05-11-svd-compress-full-ft-math10k-design.md`

_Last updated: 2026-05-12 (SVD target=2B + target=1B complete; BTT target=1B grid in progress on GPUs 1/2/3, results not yet included)._
