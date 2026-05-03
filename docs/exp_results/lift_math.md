# LIFT Math Eval Results

- Source root:`/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B`
- Metric: task accuracy (`Result` from each`eval.log`), averaged across 7 tasks.

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

| Method   | Run config                                                                           |  MultiArith |       GSM8K |      AddSub |        AQuA |    SingleEQ |       SVAMP |       MAWPS |         Avg |
| -------- | ------------------------------------------------------------------------------------ | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |
| lora     | `lr_2e-4 + rank_64 + alpha_128 (seed 43)`                                            |       99.33 |       68.84 |       93.67 |       29.53 | **98.43**   |       78.90 | **95.38**   | **80.58**   |
| blocktt  | `lr_2e-4 + output_one_block + pos_small + rank_full + smerge_trainable (seed 43)`    |       99.50 | **71.87**   |       93.16 |       27.56 |       97.64 |       79.00 |       94.54 |       80.47 |
| blocktt  | `lr_3e-4 + output_one_block + pos_small + rank_full + smerge_trainable (seed 43)`    |       99.50 |       68.84 |       92.66 | **30.32**   |       98.23 |       79.10 |       92.86 |       80.21 |
| qlora    | `r_64 + alpha_128 + lr_1e-4 (seed 43, merged)`                                       |       99.00 |       70.43 |       92.66 |       27.17 |       96.65 | **82.20**   |       92.02 |       80.02 |
| full     | `lr_1e-5 + projE (seed 43, last)`                                                    |       99.17 |       71.19 |       91.14 |       28.35 |       95.67 |       80.10 |       92.86 |       79.78 |
| blocktt  | `lr_2e-4 + output_one_block + pos_small + rank_full + smerge_keep_trainable (seed 43)` |     98.67 |       70.28 |       91.65 |       29.53 |       96.06 |       80.20 |       92.02 |       79.77 |
| lift     | `sparse + no_head + mask_lift_sparse + rank_128 + filter_128 + interval_400 + lr_1e-4 (seed 43)` | 99.33 | 68.84 | 93.42 |       29.13 |       97.24 |       78.80 |       91.60 |       79.77 |
| blocktt  | `calib_none + lr_2e-4 + output_one_block + pos_small + rank_full + smerge_keep_trainable (seed 43)` | 97.50 | 69.52 | 92.66 |       27.17 |       95.87 |       79.50 |       91.60 |       79.12 |
| blocktt  | `lr_1e-4 + output_one_block + pos_small + rank_full + smerge_trainable (seed 43)`    |       98.33 |       68.99 |       90.38 |       28.74 |       95.47 |       79.40 |       92.02 |       79.05 |
| qlora    | `r_48 + alpha_96 + lr_1e-4 (seed 43, merged)`                                        |       98.83 |       70.43 |       92.15 |       27.56 |       96.06 |       76.90 |       91.18 |       79.02 |
| lora     | `lr_2e-4 + rank_128 + alpha_256 (seed 43)`                                           | **99.67**   |       63.99 | **94.94**   |       27.17 |       97.83 |       76.50 |       91.18 |       78.75 |
| qfura    | `lr_1e-4 + layout_flat + output_one_block + smerge_keep_trainable (seed 43)`         |       95.67 |       66.72 |       91.90 |       26.38 |       95.47 |       76.50 |       92.02 |       77.81 |
| svd      | `lr_5e-5 + pos_input + smerge_keep_trainable (seed 43)`                              |       99.17 |       59.97 |       93.92 |       25.59 |       98.03 |       72.00 |       90.76 |       77.06 |
| full     | `lr_5e-5 (seed 43)`                                                                  |       99.50 |       53.15 |       93.42 |       27.56 |       96.26 |       68.80 |       90.34 |       75.57 |

Runs without complete eval logs (excluded from the table above): `blocktt-calib_none-lr_1e-4-...`, `blocktt-calib_v2_bp-lr_{2e-5,5e-5}-...`, `blocktt-lr_2e-4-decomp_input_one_block-...`, `blocktt-lr_2e-4-decomp_output_one_block-...-smerge_frozen-...`, `full-bs2gacc8-noctk-lr_2e-4-seed_43`, `full-lr_1e-4-seed_43`, `qlora-r_48-alpha_96-lr_1e-4-seed_43` (raw, pre-merge), `qlora-r_64-alpha_128-lr_1e-4-seed_43` (raw, pre-merge — partial logs only), `svd-bs2gacc8-noctk-lr_2e-4-...`, `svd-lr_{1e-4,2e-4}-pos_input-...`, `svd-lr_5e-5-pos_output-smerge_frozen-...`.

## Best Result Per Method (LLaMA-3-8B)

Best by Avg within each method family. Bold marks the column-best across this table.

| Method  | Best run                                                                              | MultiArith |     GSM8K |    AddSub |      AQuA |  SingleEQ |     SVAMP |     MAWPS |       Avg |
| ------- | ------------------------------------------------------------------------------------- | ---------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: |
| lora    | `lr_2e-4 + rank_64 + alpha_128 (seed 43)`                                             |      99.33 |     68.84 |     93.67 |     29.53 | **98.43** |     78.90 | **95.38** | **80.58** |
| blocktt | `lr_2e-4 + output_one_block + pos_small + rank_full + smerge_trainable (seed 43)`     |  **99.50** | **71.87** |     93.16 |     27.56 |     97.64 |     79.00 |     94.54 |     80.47 |
| qlora   | `r_64 + alpha_128 + lr_1e-4 (seed 43, merged)`                                        |      99.00 |     70.43 |     92.66 |     27.17 |     96.65 | **82.20** |     92.02 |     80.02 |
| full    | `lr_1e-5 + projE (seed 43, last)`                                                     |      99.17 |     71.19 |     91.14 |     28.35 |     95.67 |     80.10 |     92.86 |     79.78 |
| lift    | `sparse + no_head + mask_lift_sparse + rank_128 + filter_128 + interval_400 + lr_1e-4 (seed 43)` |     99.33 |     68.84 | **93.42** | **29.13** |     97.24 |     78.80 |     91.60 |     79.77 |
| qfura   | `lr_1e-4 + layout_flat + output_one_block + smerge_keep_trainable (seed 43)`          |      95.67 |     66.72 |     91.90 |     26.38 |     95.47 |     76.50 |     92.02 |     77.81 |
| svd     | `lr_5e-5 + pos_input + smerge_keep_trainable (seed 43)`                               |      99.17 |     59.97 |     93.92 |     25.59 |     98.03 |     72.00 |     90.76 |     77.06 |

## Source-Domain Forgetting — Math-10K Checkpoints Evaluated on Commonsense (LLaMA-3-8B)

Single model, fine-tuned on Math-10K, evaluated on **both** the 7 arithmetic tasks (in-domain) and the 8 commonsense tasks (out-of-domain, source-domain forgetting probe). Both eval logs live inside the same math run dir under `/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/<run>/{math,commonsense}/`. The commonsense Avg here measures how much general-knowledge capability the model retained after math SFT — higher is less forgetting.

| Method     | Run config                                                                                       | Math Avg (in-domain) | Commonsense Avg (source domain) |
| ---------- | ------------------------------------------------------------------------------------------------ | -------------------: | ------------------------------: |
| base model | (no fine-tuning, reference)                                                                      |                    – |                           37.55 |
| blocktt    | `lr_2e-4 + output_one_block + pos_small + smerge_trainable (seed 43)`                            |            **80.47** |                       **45.49** |
| lift       | `sparse + no_head + mask_lift_sparse + rank_128 + filter_128 + interval_400 + lr_1e-4 (seed 43)` |                79.77 |                           38.19 |
| lora       | `lr_2e-4 + rank_128 + alpha_256 (seed 43)`                                                       |                78.75 |                           35.64 |

Notes:
- `full-lr_1e-5-seed_43-projE/last` also has a partial commonsense eval (BoolQ/PIQA/SIQA/ARC-e/ARC-c/OBQA only — HellaSwag failed and Winogrande was never run), so its 8-task Avg is not yet computable. Excluded until those two tasks are re-run.
- The base-model commonsense Avg of 37.55 is the relevant zero-shot baseline against which to read the post-math-SFT numbers — it's low because the LIFT eval format requires `\boxed{}`-style answer extraction that the unfit base model handles poorly. So the absolute Avg numbers here are not directly comparable to the commonsense-trained numbers in `lift_commonsense.md`; only the *relative* ranking across math-trained checkpoints is meaningful.
- All three math-trained methods retain roughly base-model-level commonsense (35–45 Avg) — i.e., they don't *hurt* commonsense much, but they don't *help* either. blocktt is the only one that visibly lifts commonsense above the base model (+7.94), while lora-rank_128 actually drifts ~2 points below base.
- BlockTT-`trainable` on math also gives the strongest commonsense retention here, even though the project-wide default `keep_trainable` recipe was not evaluated on commonsense from the math run dir (only from the commonsense run dir, where it scored 87.91 — a different setup).
