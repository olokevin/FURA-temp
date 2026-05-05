# LIFT Math Eval Results

- Source root:`/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B`
- Metric: task accuracy (`Result` from each `eval.log`), averaged across 7 tasks.

| Run                                          |      MultiArith |           GSM8K |          AddSub |            AQuA |        SingleEQ |           SVAMP |           MAWPS |             Avg |
| -------------------------------------------- | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: |
| `output_one_block + smerge_trainable`      | **99.50** | **71.87** | **93.16** |           27.56 | **97.64** |           79.00 | **94.54** | **80.47** |
| `output_one_block + smerge_keep_trainable` |           98.67 |           70.28 |           91.65 | **29.53** |           96.06 | **80.20** |           92.02 |           79.77 |

## Paper Table 2 (Copied)

| Model      | Method                                                                | Best Rank |      MultiArith | GSM8K | AddSub |  AQuA | SingleEQ | SVAMP |           MAWPS |   Avg |
| ---------- | --------------------------------------------------------------------- | --------: | --------------: | ----: | -----: | ----: | -------: | ----: | --------------: | ----: |
|            |                                                                       |           |                 |       |        |       |          |       |                 |       |
| LLaMA-3-8B | Full FT                                                               |         - |           99.00 | 69.83 |  93.42 | 28.74 |    97.83 | 79.60 |           92.86 | 80.18 |
| LLaMA-3-8B | LoRA                                                                  |        64 |           99.17 | 71.57 |  92.15 | 24.41 |    96.26 | 80.50 |           92.02 | 79.44 |
| LLaMA-3-8B | DoRA                                                                  |        64 |           98.83 | 70.96 |  90.89 | 29.53 |    96.65 | 81.80 |           90.76 | 79.92 |
| LLaMA-3-8B | PiSSA                                                                 |       128 |           99.00 | 71.27 |  93.67 | 28.74 |    97.64 | 80.60 |           92.02 | 80.42 |
| LLaMA-3-8B | S2FT                                                                  |        64 |           99.67 | 70.89 |  92.91 | 32.68 |    97.64 | 78.20 |           94.12 | 80.87 |
| LLaMA-3-8B | LIFT                                                                  |       128 |           99.33 | 72.40 |  93.42 | 34.65 |    98.03 | 80.90 |           93.70 | 81.78 |
| LLaMA-3-8B | **Our best eval run** (`output_one_block + smerge_trainable`) |         - | **99.50** | 71.87 |  93.16 | 27.56 |    97.64 | 79.00 | **94.54** | 80.47 |

Arithmetic reasoning, fine-tuned on the MATH-10K dataset (from `docs/25_ICML_Principal Weights Emerge after Rank Reduction for Reasoning-Focused Supervised Fine-Tuning.pdf`).

Runs with missing or incomplete math eval outputs (not included above):

- `blocktt-lr_2e-4-decomp_input_one_block_pos_small-rank_full-smerge_keep_trainable-type_all-seed_43`
- `blocktt-lr_2e-4-decomp_output_one_block_pos_small-rank_full-smerge_frozen-type_all-seed_43`
- `blocktt-lr_1e-4-decomp_output_one_block_pos_small-rank_full-smerge_trainable-type_all-seed_43` (no eval outputs)
- `blocktt-lr_3e-4-decomp_output_one_block_pos_small-rank_full-smerge_trainable-type_all-seed_43` (no eval outputs)
- `lora-lr_2e-4-rank_64-alpha_128-seed_43` (no math eval outputs)

## Full Eval Sweep (LLaMA-3-8B, `/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B`)

All runs with complete 7-task math eval logs. Eval logs were collected from `math/`, `last/math/`, and `best/math/` subdirectories. Bold marks the column-best across this table.

| Method  | Run config                                                                                            |      MultiArith |           GSM8K |          AddSub |            AQuA |        SingleEQ |           SVAMP |           MAWPS |             Avg |
| ------- | ----------------------------------------------------------------------------------------------------- | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: |
| lora    | `lr_2e-4 + rank_64 + alpha_128 (seed 43)`                                                           |           99.33 |           68.84 |           93.67 |           29.53 | **98.43** |           78.90 | **95.38** | **80.58** |
| blocktt | `lr_2e-4 + output_one_block + pos_small + rank_full + smerge_trainable (seed 43)`                   |           99.50 | **71.87** |           93.16 |           27.56 |           97.64 |           79.00 |           94.54 |           80.47 |
| blocktt | `lr_3e-4 + output_one_block + pos_small + rank_full + smerge_trainable (seed 43)`                   |           99.50 |           68.84 |           92.66 | **30.32** |           98.23 |           79.10 |           92.86 |           80.21 |
| qlora   | `r_64 + alpha_128 + lr_1e-4 (seed 43, merged)`                                                      |           99.00 |           70.43 |           92.66 |           27.17 |           96.65 | **82.20** |           92.02 |           80.02 |
| full    | `lr_1e-5 + projE (seed 43, last)`                                                                   |           99.17 |           71.19 |           91.14 |           28.35 |           95.67 |           80.10 |           92.86 |           79.78 |
| blocktt | `lr_2e-4 + output_one_block + pos_small + rank_full + smerge_keep_trainable (seed 43)`              |           98.67 |           70.28 |           91.65 |           29.53 |           96.06 |           80.20 |           92.02 |           79.77 |
| lift    | `sparse + no_head + mask_lift_sparse + rank_128 + filter_128 + interval_400 + lr_1e-4 (seed 43)`    |           99.33 |           68.84 |           93.42 |           29.13 |           97.24 |           78.80 |           91.60 |           79.77 |
| blocktt | `calib_none + lr_2e-4 + output_one_block + pos_small + rank_full + smerge_keep_trainable (seed 43)` |           97.50 |           69.52 |           92.66 |           27.17 |           95.87 |           79.50 |           91.60 |           79.12 |
| blocktt | `lr_1e-4 + output_one_block + pos_small + rank_full + smerge_trainable (seed 43)`                   |           98.33 |           68.99 |           90.38 |           28.74 |           95.47 |           79.40 |           92.02 |           79.05 |
| qlora   | `r_48 + alpha_96 + lr_1e-4 (seed 43, merged)`                                                       |           98.83 |           70.43 |           92.15 |           27.56 |           96.06 |           76.90 |           91.18 |           79.02 |
| lora    | `lr_2e-4 + rank_128 + alpha_256 (seed 43)`                                                          | **99.67** |           63.99 | **94.94** |           27.17 |           97.83 |           76.50 |           91.18 |           78.75 |
| qfura   | `lr_1e-4 + layout_flat + output_one_block + smerge_keep_trainable (seed 43)`                        |           95.67 |           66.72 |           91.90 |           26.38 |           95.47 |           76.50 |           92.02 |           77.81 |
| svd     | `lr_5e-5 + pos_input + smerge_keep_trainable (seed 43)`                                             |           99.17 |           59.97 |           93.92 |           25.59 |           98.03 |           72.00 |           90.76 |           77.06 |
| full    | `lr_5e-5 (seed 43)`                                                                                 |           99.50 |           53.15 |           93.42 |           27.56 |           96.26 |           68.80 |           90.34 |           75.57 |

Runs without complete eval logs (excluded from the table above): `blocktt-calib_none-lr_1e-4-...`, `blocktt-calib_v2_bp-lr_{2e-5,5e-5}-...`, `blocktt-lr_2e-4-decomp_input_one_block-...`, `blocktt-lr_2e-4-decomp_output_one_block-...-smerge_frozen-...`, `full-bs2gacc8-noctk-lr_2e-4-seed_43`, `full-lr_1e-4-seed_43`, `qlora-r_48-alpha_96-lr_1e-4-seed_43` (raw, pre-merge), `qlora-r_64-alpha_128-lr_1e-4-seed_43` (raw, pre-merge — partial logs only), `svd-bs2gacc8-noctk-lr_2e-4-...`, `svd-lr_{1e-4,2e-4}-pos_input-...`, `svd-lr_5e-5-pos_output-smerge_frozen-...`.

## Best Result Per Method (LLaMA-3-8B)

Best by Avg within each method family. Bold marks the column-best across this table.

| Method  | Best run                                                                                           |      MultiArith |           GSM8K |          AddSub |            AQuA |        SingleEQ |           SVAMP |           MAWPS |             Avg |
| ------- | -------------------------------------------------------------------------------------------------- | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: |
| lora    | `lr_2e-4 + rank_64 + alpha_128 (seed 43)`                                                        |           99.33 |           68.84 |           93.67 |           29.53 | **98.43** |           78.90 | **95.38** | **80.58** |
| blocktt | `lr_2e-4 + output_one_block + pos_small + rank_full + smerge_trainable (seed 43)`                | **99.50** | **71.87** |           93.16 |           27.56 |           97.64 |           79.00 |           94.54 |           80.47 |
| qlora   | `r_64 + alpha_128 + lr_1e-4 (seed 43, merged)`                                                   |           99.00 |           70.43 |           92.66 |           27.17 |           96.65 | **82.20** |           92.02 |           80.02 |
| full    | `lr_1e-5 + projE (seed 43, last)`                                                                |           99.17 |           71.19 |           91.14 |           28.35 |           95.67 |           80.10 |           92.86 |           79.78 |
| lift    | `sparse + no_head + mask_lift_sparse + rank_128 + filter_128 + interval_400 + lr_1e-4 (seed 43)` |           99.33 |           68.84 | **93.42** | **29.13** |           97.24 |           78.80 |           91.60 |           79.77 |
| qfura   | `lr_1e-4 + layout_flat + output_one_block + smerge_keep_trainable (seed 43)`                     |           95.67 |           66.72 |           91.90 |           26.38 |           95.47 |           76.50 |           92.02 |           77.81 |
| svd     | `lr_5e-5 + pos_input + smerge_keep_trainable (seed 43)`                                          |           99.17 |           59.97 |           93.92 |           25.59 |           98.03 |           72.00 |           90.76 |           77.06 |

## Source-Domain Forgetting — Math-10K Checkpoints Evaluated on Commonsense (LLaMA-3-8B)

Single model, fine-tuned on Math-10K, evaluated on **both** the 7 arithmetic tasks (in-domain) and the 8 commonsense tasks (out-of-domain, source-domain forgetting probe). Both eval logs live inside the same math run dir under `/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/<run>/{math,commonsense}/`. Higher commonsense Avg = less forgetting.

### Per-task results (4 methods × 7 math + 8 commonsense)

Math (in-domain):

| Method  | Run config                                                                       |      MultiArith |           GSM8K |          AddSub |            AQuA |        SingleEQ |           SVAMP |           MAWPS | Math Avg |
| ------- | -------------------------------------------------------------------------------- | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | -------: |
| full    | `lr_1e-5 + projE (seed 43, last)`                                              |           99.17 |           71.19 |           91.14 |           28.35 |           95.67 |           80.10 |           92.86 |    79.78 |
| blocktt | `lr_2e-4 + output_one_block + pos_small + smerge_trainable (seed 43)`          |           99.50 |           71.87 |           93.16 |           27.56 |           97.64 |           79.00 | **94.54** | **80.47** |
| lora    | `lr_1e-4 + rank_64 + alpha_128 (seed 43, last)`                                | **100.00** |           69.83 |           93.16 |           25.20 | **98.03** | **80.00** |           93.28 |    79.93 |
| lora    | `lr_2e-4 + rank_128 + alpha_256 (seed 43)`                                     |           99.67 |           63.99 | **94.94** |           27.17 |           97.83 |           76.50 |           91.18 |    78.75 |
| svd     | `lr_1e-5 + pos_output + smerge_frozen + type_all (seed 43, last)`              |           98.67 | **71.95** |           91.90 |           27.17 |           95.28 |           78.00 |           91.18 |    79.16 |

Commonsense (out-of-domain, source-domain forgetting probe):

| Method  | Run config                                                                       |           BoolQ |            PIQA |            SIQA |   ARC-Challenge |        ARC-Easy |            OBQA |       HellaSwag |      Winogrande | CS Avg |
| ------- | -------------------------------------------------------------------------------- | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | -----: |
| full    | `lr_1e-5 + projE (seed 43, last)`                                              |           50.3 | **71.5** |            48.5 |            39.8 |            42.8 |            28.8 |            25.4 |            17.8 |  40.61 |
| blocktt | `lr_2e-4 + output_one_block + pos_small + smerge_trainable (seed 43)`          | **57.1** |            67.0 |            42.9 |            48.9 |            55.3 |            36.4 |            18.3 | **38.0** |  45.49 |
| lora    | `lr_1e-4 + rank_64 + alpha_128 (seed 43, last)`                                |           55.2 |            65.0 |            36.0 |            47.6 |            55.9 | **47.8** |            22.4 |            21.7 |  43.95 |
| lora    | `lr_2e-4 + rank_128 + alpha_256 (seed 43)`                                     |           52.5 |            51.3 |            28.2 |            29.7 |            43.0 |            35.6 |            21.8 |            23.0 |  35.64 |
| svd     | `lr_1e-5 + pos_output + smerge_frozen + type_all (seed 43, last)`              |           50.3 |            70.2 | **54.8** | **54.8** | **61.4** |            42.4 | **26.5** |            21.2 | **47.70** |

### Target vs source summary (GSM8K + commonsense Avg)

GSM8K is the standard "did the math SFT actually take" probe; commonsense Avg is the source-domain forgetting probe.

| Method  |          GSM8K (target) | Commonsense Avg (source) |  Δ vs full |
| ------- | ----------------------: | -----------------------: | ---------: |
| full    |                   71.19 |                    40.61 |       0.00 |
| blocktt |                   71.87 |                    45.49 |      +4.88 |
| lora (`lr_1e-4 + r_64`)   |          69.83 |                    43.95 |      +3.34 |
| lora (`lr_2e-4 + r_128`)  |          63.99 |                    35.64 |      −4.97 |
| svd     | **71.95** |        **47.70** | **+7.09** |

Notes:

- All five rows are now evaluated on the full 8-task commonsense suite (HellaSwag and Winogrande added in the May 2026 sweep for `full-lr_1e-5-seed_43-projE`, `svd-lr_1e-5-pos_output-smerge_frozen-type_all-seed_43`, and the new `lora-lr_1e-4-rank_64-alpha_128-seed_43`).
- The base-model commonsense Avg in this evaluation harness is **37.55** — low because the LIFT eval format requires `\boxed{}`-style answer extraction that an unfit base model handles poorly. Absolute Avg numbers here are not directly comparable to the commonsense-trained numbers in `lift_commonsense.md`; only the *relative* ranking across math-trained checkpoints is meaningful.
- **SVD-FT (lr=1e-5, pos_output, smerge_frozen)** simultaneously gets the best GSM8K (71.95) and the highest commonsense retention (+7.09 over full-FT, +12.06 over the worst LoRA), confirming that SVD-style updates leak less into source-domain capabilities than dense full-FT or aggressive LoRA at this lr regime.
- **LoRA is highly lr-sensitive** here: the milder `lr_1e-4 + r_64 + α_128` recipe nearly matches blocktt on commonsense (+3.34 Δ vs full) and stays within 1.4 points on GSM8K, while the aggressive `lr_2e-4 + r_128 + α_256` recipe collapses on both axes (−4.97 commonsense, −7.20 GSM8K). Headline LoRA story depends entirely on which run you pick.
- BlockTT-`trainable` is a clear second on the source-domain probe (+4.88) while topping the math Avg overall — strong target+source trade-off without the SVD-FT extra commonsense bump.

## LR + Batch-Size Sweeps — GSM8K Quick Eval (LLaMA-3-8B)

Two parallel sweeps over math_10k (3 epochs, seed=43), evaluated on **GSM8K only** for a fast turnaround. All runs use `meta-llama/Meta-Llama-3-8B`, `max_seq_len=2048`, the script-default 5 (lora) or 7 (blocktt) target modules, and the dual-save layout (model in `<run>/last/`).

- **lora**: `r=64, α=128`, target_modules = `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` (math-script default — note this is **7-module**, unlike the commonsense lora script which defaults to 5).
- **blocktt** (fura): `output_one_block / pos_small / smerge_keep_trainable / rank=full`, no calibration.

The total optimization-step count differs across runs because both batch size *and* gradient-accumulation steps change while `n_examples × n_epochs = 30K` is fixed.

### LR Sweep (effective batch size = 16)

Per-device batch × accum is the script default: `bsz=1, accum=16` for lora; `bsz=2, accum=8` for blocktt. 1842 optimizer steps per run.

#### LoRA (rank=64, α=128, 7-mod target list)

| LR    | Final loss |   GSM8K |
| ----- | ---------: | ------: |
| 6e-5  |      0.155 | **70.81** |
| 8e-5  |      0.135 |   69.83 |
| 1e-4  |      0.122 |   69.83 |
| 2e-4  |        n/a | (overwritten — see notes) |

#### BlockTT (output_one_block / pos_small / smerge_keep_trainable / rank=full, no calib)

| LR    | Final loss |   GSM8K |
| ----- | ---------: | ------: |
| 8e-5  |      0.308 |   66.41 |
| 1e-4  |      0.290 |   67.70 |
| 2e-4  |      0.220 |   69.52 |
| 3e-4  |      0.165 | **71.57** |
| 4e-4  |      0.130 |   70.43 |
| 6e-4  |      0.095 |   66.34 |
| 8e-4  |      0.084 |   62.85 |

Bold = best within method. lora `lr=1e-4` and blocktt `lr ∈ {1e-4, 2e-4}` are from prior bsz=16 runs (Apr 14–15, 2026, plus the 2026-05-04 re-eval); blocktt `lr ∈ {4e-4, 6e-4, 8e-4}` were added 2026-05-05 to map the curve past the 3e-4 peak.

### Batch-Size Sweep (LR scaled with √(bsz/16))

Per-device bsz × accum: `1×64`, `1×256` for lora; `2×32`, `2×128` for blocktt. Optimizer steps drop from 1842 → 462 (bsz=64) → 117 (bsz=256), but total fwd/bwd compute is unchanged.

LR-scaling baseline differs per method: lora used the bsz=16 lr=2e-4 reference (so bsz=64 ≈ 4e-4, bsz=256 ≈ 8e-4 — but actually bsz=256 used 4e-4 in the original sweep, an under-scale); blocktt used both an early scaling around the bsz=16 lr=4e-4/8e-4 attempts (yielding 70.89 / 71.04) and a re-derived scaling from the best bsz=16 LR (3e-4 → ×2 = 6e-4 for bsz=64, ×4 = 1.2e-3 for bsz=256).

#### LoRA (rank=64, α=128, 7-mod target list)

| Effective bsz | LR     | Final loss |   GSM8K |
| ------------: | ------ | ---------: | ------: |
|            64 | 2e-4   |      0.132 | **70.36** |
|           256 | 4e-4   |      0.177 |   68.54 |

#### BlockTT (output_one_block / pos_small / smerge_keep_trainable / rank=full, no calib)

| Effective bsz | LR     | Final loss |   GSM8K |
| ------------: | ------ | ---------: | ------: |
|            64 | 4e-4   |      0.173 |   70.89 |
|            64 | 6e-4   |      0.127 | **72.18** |
|           256 | 8e-4   |      0.212 |   71.04 |
|           256 | 1.2e-3 |      0.198 |   68.46 |

Bold = best within method. The two extra blocktt rows (`bsz=64 lr=6e-4`, `bsz=256 lr=1.2e-3`) come from the 2026-05-05 follow-up that picked best bsz=16 LR (=3e-4) and applied √(bsz/16) scaling.

### Combined view — all sweep configs

| Method  | Sweep | Effective bsz | LR     |   GSM8K |
| ------- | ----- | ------------: | ------ | ------: |
| blocktt | bsz   |            64 | 6e-4   |   **72.18** 🥇 |
| blocktt | LR    |            16 | 3e-4   |   71.57 |
| blocktt | bsz   |           256 | 8e-4   |   71.04 |
| blocktt | bsz   |            64 | 4e-4   |   70.89 |
| lora    | LR    |            16 | 6e-5   |   70.81 |
| blocktt | LR    |            16 | 4e-4   |   70.43 |
| lora    | bsz   |            64 | 2e-4   |   70.36 |
| lora    | LR    |            16 | 8e-5   |   69.83 |
| lora    | LR    |            16 | 1e-4   |   69.83 |
| blocktt | LR    |            16 | 2e-4   |   69.52 |
| lora    | bsz   |           256 | 4e-4   |   68.54 |
| blocktt | bsz   |           256 | 1.2e-3 |   68.46 |
| blocktt | LR    |            16 | 1e-4   |   67.70 |
| blocktt | LR    |            16 | 8e-5   |   66.41 |
| blocktt | LR    |            16 | 6e-4   |   66.34 |
| blocktt | LR    |            16 | 8e-4   |   62.85 |

### Notes & caveats

- **bsz=64 lora was re-evaluated on 2026-05-04 23:16 PDT** because the original auto-eval picked up a stale Apr-14 model from the parent dir (`pytorch_model.bin` left over from a prior LR-sweep run sharing the same dir name). The 70.36 number is the corrected re-eval; the originally-logged 68.84 is invalid and was overwritten. Other 7 sweep runs are unaffected — their dirs were clean before launch.
- **Wandb logging for math evals was previously broken** — `run_math_parallel.py` had no `wandb.init` and the bash eval scripts didn't pass `--wandb_*` args to it. Fixed in this same window: `run_math_parallel.py:180-202, 230-253` now mirrors `run_commonsense_parallel.py`'s wandb integration, and `eval_math.sh` / `eval_math_lora.sh` now append the wandb args. The bsz=64 lora re-eval verified the wandb push end-to-end (resumed run `d0lb91v0`, logged `eval/gsm8k/accuracy` and `eval/gsm8k/total`).
- The other 7 sweep runs' GSM8K numbers were not pushed to wandb during their original eval. They can be backfilled cheaply (~10 min total) by re-running their evals against the same checkpoints with the patched scripts; not done here.
- **lora prefers low LR at small bsz**: 6e-5 (70.81) > 8e-5 ≈ 1e-4 (both 69.83) at bsz=16; the lr=2e-4 r=64 cell is missing because the prior bsz=16 run was overwritten by the bsz=64 sweep (see notes below). With the LR-scaling rule (lr ∝ √bsz), bsz=64/lr=2e-4 lands at 70.36 (close to the 6e-5 baseline) but bsz=256/lr=4e-4 drops to 68.54 — large-batch LR scaling underperforms on this task.
- **blocktt's LR curve at bsz=16 has a clean peak at 3e-4 then collapses**: 8e-5 → 66.41, 1e-4 → 67.70, 2e-4 → 69.52, 3e-4 → **71.57** (peak), 4e-4 → 70.43, 6e-4 → 66.34, 8e-4 → 62.85. The ~9-pt headroom across the full curve is the largest of either method on this task. Final training loss continues dropping past the GSM8K peak (lr=8e-4 has loss 0.084 vs lr=3e-4's 0.165), so blocktt at high LR overfits the math_10k training distribution at the expense of GSM8K test accuracy.
- **√(bsz/16) LR scaling worked at bsz=64 but broke at bsz=256**: starting from best bsz=16 LR=3e-4, scaled to lr=6e-4 for bsz=64 (= 3e-4 × √(64/16)) → **72.18** (new overall best, +0.61 over bsz=16 lr=3e-4). Scaled to lr=1.2e-3 for bsz=256 (= 3e-4 × √(256/16)) → 68.46 (−3.11 vs the bsz=16 baseline). The bsz=256 collapse at scaled lr is consistent with the bsz=16 LR curve showing similar collapse beyond ~4e-4 — the √-rule pushes LR past blocktt's stable range when the bsz multiplier is too aggressive.
- **lora is more sensitive to LR than to bsz, blocktt the opposite**: lora's max-min gap on this task is 70.81 − 68.54 = 2.27 pts (full sweep). blocktt's max-min is 72.18 − 62.85 = 9.33 pts, almost entirely driven by the LR axis. blocktt's bsz=64 result (72.18) outranks every lora config including the well-tuned ones, so when comparing methods at fixed-compute, blocktt wins decisively but only when the LR is in the narrow [2e-4, 4e-4] window at bsz=16 (or the corresponding √-scaled range at larger bsz).
- All training is single-seed (seed=43); single-config differences in the ~0.5–1.0 GSM8K range are within noise. Run rankings should be treated as approximate.
- **Provenance of bsz=16 lr=1e-4 / 2e-4 rows**: lora lr=1e-4 (`lora-lr_1e-4-rank_64-alpha_128-seed_43`) finished training & eval cleanly with the dual-save layout — `last/math/gsm8k/eval.log` is canonical. blocktt lr=2e-4 (`blocktt-calib_none-lr_2e-4-decomp_output_one_block_pos_small-rank_full-smerge_keep_trainable-type_all-seed_43`) is from an Apr-15 flat-save run (no `last/` subdir; eval is at the parent's `math/gsm8k/eval.log`). blocktt lr=1e-4 had a finished training but no eval; eval was launched on 2026-05-04 23:53 PDT and produced `Result 67.7028`. **lora lr=2e-4 r=64 α=128 bsz=16 was overwritten** when the bsz=64 sweep launch reused the same dir name (the math scripts hardcoded `OUTPUT=…` without an env override at the time; this was patched 2026-05-05 to make `OUTPUT` env-overridable). To recover that cell, the run would need to be re-trained from scratch (~90 min).
- **Provenance of the 2026-05-05 chain** (3 new bsz=16 LRs + 2 derived bsz runs): chain script `tools/run_gpu3_blocktt_lr_then_bsz_sweep.sh`. All 5 runs use distinct `-bsz16` / `-bsz64` / `-bsz256` suffixed output dirs (made possible by the `OUTPUT="${OUTPUT:-…}"` patch on `finetune_math_blocktt.sh`), so they don't collide with the prior bsz=64 lr=4e-4 / bsz=256 lr=8e-4 runs that share the same lr value.
