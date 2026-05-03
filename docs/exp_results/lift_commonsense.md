# LIFT Commonsense Results

## Our Runs (`/data/yequan/fura/lift`)

- Source root:`/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B`
- Metric: task accuracy (`Result` from each `eval.log`), averaged across 8 tasks.
- Bold indicates the best value among our runs for that column.

| Run                                              |           BoolQ |            PIQA |            SIQA |       HellaSwag |            Wino |           ARC-e |           ARC-c |            OBQA |             Avg |
| ------------------------------------------------ | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: |
| `output_one_block + smerge_keep_trainable`     | **76.60** |           89.90 | **83.20** | **96.80** | **89.70** | **93.60** | **84.10** | **89.40** | **87.91** |
| `output_one_block + rank_full + smerge_frozen` |           74.00 | **90.80** |           81.90 |           95.90 |           89.40 |           93.40 |           84.00 |           87.60 |           87.12 |
| `input_one_block + rank_full + smerge_frozen`  |           66.30 |           85.60 |           79.50 |           94.70 |           86.10 |           90.50 |           80.50 |           86.80 |           83.75 |
| `input_one_block + smerge_trainable`           |           67.00 |           85.60 |           80.60 |           93.50 |           86.30 |           90.20 |           78.40 |           84.80 |           83.30 |

Runs with missing or incomplete commonsense eval outputs (not included above):

- `blocktt-lr_2e-4-decomp_output_one_block_pos_small_smerge_trainable-seed_43` (no task eval logs)
- `full-lr_5e-5-seed_43` (no task eval logs found)
- `lora-lr_2e-4-rank_128-seed_43` (task eval logs missing/incomplete)
- `meta-llama/Meta-Llama-3-8B/lora/commonsense/lr_2e-4/rank_128_alpha_256/seed_43` (partial task logs only)

## Paper Table 1 (Copied)

Commonsense reasoning, fine-tuning on Commonsense-170K (from `docs/25_ICML_Principal Weights Emerge after Rank Reduction for Reasoning-Focused Supervised Fine-Tuning.pdf`).

### LLaMA-2-7B

| Method  | Best Rank | BoolQ | PIQA | SIQA | HellaSwag | Wino | ARC-e | ARC-c | OBQA |   Avg |
| ------- | --------: | ----: | ---: | ---: | --------: | ---: | ----: | ----: | ---: | ----: |
| Full FT |         - |  73.8 | 84.2 | 81.0 |      94.7 | 85.2 |  88.9 |  75.6 | 84.8 | 83.53 |
| LoRA    |       128 |  70.8 | 82.8 | 79.4 |      92.9 | 83.4 |  86.3 |  71.6 | 82.8 | 81.25 |
| DoRA    |       128 |  71.3 | 83.4 | 80.1 |      92.3 | 84.0 |  86.1 |  71.4 | 85.8 | 81.80 |
| PiSSA   |       128 |  72.5 | 85.3 | 80.8 |      87.2 | 86.1 |  87.1 |  74.3 | 85.6 | 82.36 |
| S2FT    |       128 |  73.3 | 83.7 | 81.0 |      94.3 | 84.6 |  88.3 |  75.8 | 84.8 | 83.22 |
| LIFT    |       128 |  74.8 | 84.7 | 82.2 |      94.4 | 86.0 |  89.2 |  76.4 | 89.6 | 84.66 |
| `output_one_block + smerge_keep_trainable` (calib_none, lr 1e-4, seed 43, last) | - | 71.8 | 83.9 | 80.5 | 93.5 | 83.3 | 87.4 | 73.2 | 80.0 | 81.70 |

### LLaMA-3-8B

| Method                                       | Best Rank |           BoolQ |  PIQA |  SIQA | HellaSwag |  Wino |           ARC-e | ARC-c |  OBQA |   Avg |
| -------------------------------------------- | --------: | --------------: | ----: | ----: | --------: | ----: | --------------: | ----: | ----: | ----: |
| Full FT                                      |         - |            75.4 |  88.0 |  81.8 |      96.5 |  89.3 |            93.1 |  83.0 |  86.0 | 86.64 |
| LoRA                                         |        64 |            71.8 |  85.3 |  80.9 |      93.4 |  84.5 |            90.0 |  77.0 |  84.8 | 83.46 |
| DoRA                                         |        64 |            74.6 |  87.4 |  81.2 |      94.7 |  87.1 |            89.4 |  79.5 |  86.4 | 85.04 |
| S2FT                                         |        64 |            67.7 |  89.8 |  82.5 |      95.2 |  87.8 |            93.1 |  84.6 |  88.6 | 86.16 |
| LIFT                                         |        32 |            75.7 |  90.5 |  83.2 |      96.5 |  89.4 |            93.6 |  83.9 |  90.2 | 87.88 |
| `output_one_block + smerge_keep_trainable` |         - | **76.60** | 89.90 | 83.20 |     96.80 | 89.70 | **93.60** | 84.10 | 89.40 | 87.91 |

### Best Of Our Runs (LLaMA-3-8B)

| Run                                          |           BoolQ |  PIQA |            SIQA |       HellaSwag |            Wino |           ARC-e |           ARC-c |            OBQA |             Avg |
| :------------------------------------------- | --------------: | ----: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: |
| `output_one_block + smerge_keep_trainable` | **76.60** | 89.90 | **83.20** | **96.80** | **89.70** | **93.60** | **84.10** | **89.40** | **87.91** |

### BlockTT Ablation (LLaMA-3-8B, no calibration)

Decompose each linear as `W = L · S · R` with L on the output side and R on the input side. **Bold** = trainable; `(SR)` / `(LS)` parentheses mean S is merged into the adjacent core and the merged tensor is trained as one parameter; no parentheses on `S` = `btt_s` is held as a separate parameter (its trainability follows the bolding). Restricted to runs without calibration (`calib_none` plus runs with no `calib_*` prefix; `calib_v2*` runs excluded). All runs use `pos_small` and `lr 2e-4`.

| Decomp             | Notation        | Rank | Seed |     BoolQ |      PIQA |      SIQA | HellaSwag |      Wino |     ARC-e |     ARC-c |      OBQA |       Avg |
| :----------------- | :-------------- | :--: | :--: | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: |
| `output_one_block` | L **S** **R**   |  -   |  43  | **76.60** |     89.90 | **83.20** | **96.80** | **89.70** |     93.60 |     84.10 |     89.40 | **87.91** |
| `output_one_block` | L **S** **R**   |  -   |   0  |     69.60 |     89.70 |     82.30 |     96.30 |     89.00 | **94.00** | **84.90** | **90.80** |     87.08 |
| `output_one_block` | L **S** **R**   |  -   | 100  |     64.00 |     89.00 |     82.70 |     95.90 |     89.60 |     93.80 |     82.80 |     90.00 |     85.97 |
| `output_one_block` | (LS) **R**      | full |  43  |     74.00 | **90.80** |     81.90 |     95.90 |     89.40 |     93.40 |     84.00 |     87.60 |     87.12 |
| `output_one_block` | L **(SR)**      |  -   |  43  |     74.80 |     76.60 |     82.20 |     95.40 |     86.40 |     90.90 |     81.60 |     84.80 |     84.09 |
| `input_one_block`  | **L** (SR)      | full |  43  |     66.30 |     85.60 |     79.50 |     94.70 |     86.10 |     90.50 |     80.50 |     86.80 |     83.75 |
| `input_one_block`  | **(LS)** R      |  -   |  43  |     67.00 |     85.60 |     80.60 |     93.50 |     86.30 |     90.20 |     78.40 |     84.80 |     83.30 |

Side note on which side is small: for `output_one_block` (m=1, a=out_features, n=in_blocks, b=in_block_size), `btt_l.numel() = rank · n · out_features` and `btt_r.numel() = rank · in_features`, so **R is the small side** (factor of `n` smaller). With `pos_small`, R trains. Symmetrically for `input_one_block` (n=1), L is the small side and L trains.

Takeaways: (i) `output_one_block` (small core on the input/right side) consistently beats `input_one_block` (top three vs bottom two by Avg). (ii) Holding `S` as a separate trainable scaling (L **S** **R**) is the strongest configuration; merging `S` into the trainable core (L **(SR)**) substantially hurts PIQA. (iii) Seed variance is large (87.91 / 87.08 / 85.97 with the same recipe), so single-seed comparisons should be read with caution.

## Full Eval Sweep (LLaMA-3-8B, `/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B`)

All runs with complete 8-task eval logs. Eval logs were collected from `commonsense/`, `last/commonsense/`, and `best/commonsense/` subdirectories. Bold marks the column-best across this table.

| Method   | Run config                                                                          | BoolQ           | PIQA            | SIQA            | HellaSwag       | Wino            | ARC-e           | ARC-c           | OBQA            | Avg             |
| -------- | ----------------------------------------------------------------------------------- | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: |
| full     | `lr_1e-5 (seed 43, last)`                                                           | **76.60**       | **90.80**       | 82.40           | **97.00**       | **89.90**       | 93.80           | **85.00**       | **90.20**       | **88.21**       |
| blocktt  | `lr_2e-4 + output_one_block + pos_small + smerge_keep_trainable (seed 43)`          | **76.60**       | 89.90           | **83.20**       | 96.80           | 89.70           | 93.60           | 84.10           | 89.40           | 87.91           |
| qfura    | `lr_2e-4 + layout_flat + output_one_block + smerge_keep_trainable (seed 43)`        | 73.00           | 89.90           | 82.70           | 96.60           | 89.10           | 93.10           | 83.40           | 90.60           | 87.30           |
| blocktt  | `lr_2e-4 + output_one_block + pos_small + rank_full + smerge_frozen (seed 43)`      | 74.00           | **90.80**       | 81.90           | 95.90           | 89.40           | 93.40           | 84.00           | 87.60           | 87.12           |
| blocktt  | `calib_none + lr_2e-4 + output_one_block + pos_small + smerge_keep_trainable (seed 0)`   | 69.60      | 89.70           | 82.30           | 96.30           | 89.00           | **94.00**       | 84.90           | **90.80**       | 87.08           |
| blocktt  | `calib_none + lr_2e-4 + output_one_block + pos_small + smerge_keep_trainable (seed 100)` | 64.00      | 89.00           | 82.70           | 95.90           | 89.60           | 93.80           | 82.80           | 90.00           | 85.97           |
| blocktt  | `lr_2e-4 + output_one_block + pos_small + smerge_trainable (seed 43)`               | 74.80           | 76.60           | 82.20           | 95.40           | 86.40           | 90.90           | 81.60           | 84.80           | 84.09           |
| blocktt  | `lr_2e-4 + input_one_block + pos_small + rank_full + smerge_frozen (seed 43)`       | 66.30           | 85.60           | 79.50           | 94.70           | 86.10           | 90.50           | 80.50           | 86.80           | 83.75           |
| blocktt  | `lr_2e-4 + input_one_block + pos_small + smerge_trainable (seed 43)`                | 67.00           | 85.60           | 80.60           | 93.50           | 86.30           | 90.20           | 78.40           | 84.80           | 83.30           |
| lora     | `lr_2e-4 + rank_64 (seed 43, last)`                                                 | 67.80           | 77.30           | 75.90           | 81.80           | 78.00           | 78.10           | 63.30           | 74.80           | 74.62           |
| milora   | `lr_2e-4 + rank_64 (seed 43)`                                                       | 67.80           | 77.10           | 76.00           | 79.70           | 77.10           | 76.90           | 62.50           | 72.20           | 73.66           |
| qdora    | `r_64 + alpha_128 + lr_2e-4 (seed 43, last)`                                        | 66.40           | 77.30           | 73.90           | 80.20           | 77.00           | 77.30           | 60.80           | 72.80           | 73.21           |
| qlora    | `r_48 + alpha_96 + lr_2e-4 (seed 43, merged)`                                       | 65.10           | 71.30           | 70.90           | 70.00           | 72.20           | 67.80           | 55.50           | 68.20           | 67.62           |
| full     | `lr_5e-5 (seed 43)`                                                                 | 38.00           | 68.40           | 63.30           | 72.40           | 74.30           | 79.80           | 63.70           | 60.80           | 65.09           |
| randlora | `lr_2e-4 + rank_64 (seed 43)`                                                       | 64.20           | 62.10           | 64.70           | 41.20           | 65.80           | 48.60           | 38.50           | 54.60           | 54.96           |
| lora     | `lr_2e-4 + rank_128 (seed 43)`                                                      | 62.10           | 48.20           | 33.00           | 24.70           | 48.90           | 23.90           | 22.70           | 26.20           | 36.21           |

Runs without complete eval logs (excluded from the table above): `full-lr_{2e-5,3e-5}-seed_43`, `lora-lr_{1e-4,3e-4,8e-5}-rank_*-seed_43` and `lora-lr_1e-4-rank_128-seed_43`, `qdora-r_64-alpha_128-lr_1e-4-seed_43` (`last/` subdir empty) and `qdora-r_64-alpha_128-lr_2e-4-seed_43/best` (only `boolq` run), `qlora-r_48-alpha_96-lr_2e-4-seed_43` (raw, pre-merge), `qlora-r_64-alpha_128-lr_1e-4-seed_43`, `randlora-lr_1e-4-rank_64-seed_43`, `milora-lr_1e-4-rank_64-seed_43`, `svd-lr_2e-4-pos_input_smerge_keep_trainable-type_all-seed_43`, `blocktt-calib_none-lr_{1e-5,5e-5}-decomp_output_one_block_pos_both_smerge_keep_trainable-seed_43`, `blocktt-calib_v2*-lr_2e-5-*-seed_43`.

## Best Result Per Method (LLaMA-3-8B)

Best by Avg within each method family. Bold marks the column-best across this table.

| Method   | Best run                                                                       | BoolQ     | PIQA      | SIQA      | HellaSwag | Wino      | ARC-e     | ARC-c     | OBQA      | Avg       |
| -------- | ------------------------------------------------------------------------------ | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: |
| full     | `lr_1e-5 (seed 43, last)`                                                      | **76.60** | **90.80** | 82.40     | **97.00** | **89.90** | **93.80** | **85.00** | 90.20     | **88.21** |
| blocktt  | `lr_2e-4 + output_one_block + pos_small + smerge_keep_trainable (seed 43)`     | **76.60** | 89.90     | **83.20** | 96.80     | 89.70     | 93.60     | 84.10     | 89.40     | 87.91     |
| qfura    | `lr_2e-4 + layout_flat + output_one_block + smerge_keep_trainable (seed 43)`   | 73.00     | 89.90     | 82.70     | 96.60     | 89.10     | 93.10     | 83.40     | **90.60** | 87.30     |
| lora     | `lr_2e-4 + rank_64 (seed 43, last)`                                            | 67.80     | 77.30     | 75.90     | 81.80     | 78.00     | 78.10     | 63.30     | 74.80     | 74.62     |
| milora   | `lr_2e-4 + rank_64 (seed 43)`                                                  | 67.80     | 77.10     | 76.00     | 79.70     | 77.10     | 76.90     | 62.50     | 72.20     | 73.66     |
| qdora    | `r_64 + alpha_128 + lr_2e-4 (seed 43, last)`                                   | 66.40     | 77.30     | 73.90     | 80.20     | 77.00     | 77.30     | 60.80     | 72.80     | 73.21     |
| qlora    | `r_48 + alpha_96 + lr_2e-4 (seed 43, merged)`                                  | 65.10     | 71.30     | 70.90     | 70.00     | 72.20     | 67.80     | 55.50     | 68.20     | 67.62     |
| randlora | `lr_2e-4 + rank_64 (seed 43)`                                                  | 64.20     | 62.10     | 64.70     | 41.20     | 65.80     | 48.60     | 38.50     | 54.60     | 54.96     |

## Llama-3.1-8B (`/data/yequan/fura/lift/commonsense/meta-llama/Llama-3.1-8B`)

All runs with complete 8-task eval logs. All numbers come from the base `commonsense/` directory of each run (no `last/` or `best/` subdirs were present here). Bold marks the column-best across this table.

| Method   | Run config                                                                          | BoolQ           | PIQA            | SIQA            | HellaSwag       | Wino            | ARC-e           | ARC-c           | OBQA            | Avg             |
| -------- | ----------------------------------------------------------------------------------- | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: |
| blocktt  | `calib_none + lr_2e-4 + output_one_block + pos_small + smerge_keep_trainable (seed 43)` | **74.10**   | **90.80**       | **83.50**       | **96.80**       | **89.30**       | **93.80**       | **84.50**       | **90.00**       | **87.85**       |
| qfura    | `lr_2e-4 + layout_flat + output_one_block + smerge_keep_trainable (seed 43)`        | 74.00           | **90.80**       | 82.30           | 96.60           | 87.80           | 93.50           | 83.30           | 89.40           | 87.21           |
| lift     | `lr_2e-4 + rank_64 (seed 43)`                                                       | 66.60           | 79.80           | 76.90           | 86.30           | 80.70           | 83.00           | 69.40           | 78.80           | 77.69           |
| dora     | `lr_2e-4 + rank_64 (seed 43)`                                                       | 69.00           | 79.90           | 76.50           | 85.80           | 79.40           | 80.30           | 65.40           | 78.00           | 76.79           |
| lora     | `lr_2e-4 + rank_64 (seed 43)`                                                       | 68.10           | 80.20           | 77.40           | 85.30           | 79.20           | 80.20           | 65.80           | 74.60           | 76.35           |
| qlora    | `r_64 + alpha_128 + lr_2e-4 (seed 43, merged)`                                      | 62.20           | 79.50           | 69.10           | 82.40           | 73.70           | 81.40           | 63.80           | 70.60           | 72.84           |
| full     | `lr_5e-5 + bsz1_accum16_nockpt (seed 43)`                                           | 39.50           | 82.20           | 65.70           | 80.80           | 76.60           | 83.30           | 70.80           | 68.40           | 70.91           |
| randlora | `lr_2e-4 + rank_64 (seed 43)`                                                       | 64.40           | 74.10           | 74.30           | 70.90           | 72.70           | 68.60           | 53.90           | 67.00           | 68.24           |
| lora     | `lr_2e-4 + rank_64 + tgt_7mod (seed 43)`                                            | 64.60           | 72.00           | 70.90           | 64.30           | 70.70           | 69.70           | 53.80           | 64.80           | 66.35           |
| full     | `lr_2e-4 (seed 43)`                                                                 | 56.60           | 49.00           | 39.20           | 25.10           | 50.70           | 23.30           | 25.40           | 28.40           | 37.21           |

Excluded: `pissa-lr_2e-4-rank_64-tgt_7mod-seed_43` returns 0.0 on every task (broken model output); `pissa-lr_2e-4-rank_64-seed_43`, `dora-lr_2e-4-rank_64-tgt_7mod-seed_43`, `randlora-lr_2e-4-rank_64-tgt_7mod-seed_43`, `svd-lr_5e-5-pos_input_smerge_trainable-bsz1_accum16_nockpt-seed_43`, `full-lr_5e-5-seed_43`, `milora-lr_2e-4-rank_64-seed_43`, `blocktt-calib_none-lr_2e-4-decomp_output_one_block_pos_small_smerge_s_keep_trainable-seed_43` have no eval logs (training only); `qdora-r_64-alpha_128-lr_2e-4-seed_43-merged` is missing the boolq log; `qdora-r_64-alpha_128-lr_2e-4-seed_43` and `qlora-r_64-alpha_128-lr_2e-4-seed_43` are unmerged adapter dirs (the corresponding `*-merged` dirs were used where complete).

### Best Result Per Method (Llama-3.1-8B)

Best by Avg within each method family. Bold marks the column-best across this table.

| Method   | Best run                                                                                | BoolQ     | PIQA      | SIQA      | HellaSwag | Wino      | ARC-e     | ARC-c     | OBQA      | Avg       |
| -------- | --------------------------------------------------------------------------------------- | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: |
| blocktt  | `calib_none + lr_2e-4 + output_one_block + pos_small + smerge_keep_trainable (seed 43)` | **74.10** | **90.80** | **83.50** | **96.80** | **89.30** | **93.80** | **84.50** | **90.00** | **87.85** |
| qfura    | `lr_2e-4 + layout_flat + output_one_block + smerge_keep_trainable (seed 43)`            | 74.00     | **90.80** | 82.30     | 96.60     | 87.80     | 93.50     | 83.30     | 89.40     | 87.21     |
| lift     | `lr_2e-4 + rank_64 (seed 43)`                                                           | 66.60     | 79.80     | 76.90     | 86.30     | 80.70     | 83.00     | 69.40     | 78.80     | 77.69     |
| dora     | `lr_2e-4 + rank_64 (seed 43)`                                                           | 69.00     | 79.90     | 76.50     | 85.80     | 79.40     | 80.30     | 65.40     | 78.00     | 76.79     |
| lora     | `lr_2e-4 + rank_64 (seed 43)`                                                           | 68.10     | 80.20     | 77.40     | 85.30     | 79.20     | 80.20     | 65.80     | 74.60     | 76.35     |
| qlora    | `r_64 + alpha_128 + lr_2e-4 (seed 43, merged)`                                          | 62.20     | 79.50     | 69.10     | 82.40     | 73.70     | 81.40     | 63.80     | 70.60     | 72.84     |
| full     | `lr_5e-5 + bsz1_accum16_nockpt (seed 43)`                                               | 39.50     | 82.20     | 65.70     | 80.80     | 76.60     | 83.30     | 70.80     | 68.40     | 70.91     |
| randlora | `lr_2e-4 + rank_64 (seed 43)`                                                           | 64.40     | 74.10     | 74.30     | 70.90     | 72.70     | 68.60     | 53.90     | 67.00     | 68.24     |
