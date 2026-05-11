# FuRA Hyperparameter Configurations

Hyperparameter settings used for FuRA across the three task suites that mirror DoRA's
evaluation protocol (Liu et al., 2024 — Tables 8 / 9 / 10): commonsense reasoning,
RL on math, and visual instruction tuning. FuRA = BlockTT-full with the project-default
recipe (`output_one_block`, `train_position=small`, `s_merged_to=keep_trainable`,
`blocktt_rank=full`).

## Table 1 — FuRA on the commonsense reasoning tasks (LLaMA-2-7B & LLaMA-3-8B)

Eight-task suite (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge,
OBQA) following the LIFT / LLM-Adapters protocol. Source script:
`ref/LIFT/bash_scripts/finetune_commonsense_qfura.sh`. Per-model winning LRs read
off `docs/exp_results/lift_commonsense.md` (LLaMA-2-7B: lr=3e-4, Avg 84.41;
LLaMA-3-8B: lr=2e-4, Avg 87.91).

| Hyperparameters (FuRA)                | LLaMA-2-7B                                                    | LLaMA-3-8B                                                    |
| :------------------------------------ | :------------------------------------------------------------ | :------------------------------------------------------------ |
| Dropout                               | 0.0                                                           | 0.0                                                           |
| Optimizer                             | AdamW                                                         | AdamW                                                         |
| LR                                    | 3e-4                                                          | 2e-4                                                          |
| LR Scheduler                          | Linear                                                        | Linear                                                        |
| Weight decay                          | 0.0                                                           | 0.0                                                           |
| Batch size (per device × grad-accum) | 8 × 2 = 16                                                   | 8 × 2 = 16                                                   |
| Warmup ratio                          | 0.03                                                          | 0.03                                                          |
| Epochs                                | 3                                                             | 3                                                             |
| Max sequence length                   | 2048                                                          | 2048                                                          |
| Mixed precision                       | bf16                                                          | bf16                                                          |
| Gradient checkpointing                | on                                                            | on                                                            |
| Where                                 | Q, K, V, O, Up, Down, Gate (all 7 linear projections / layer) | Q, K, V, O, Up, Down, Gate (all 7 linear projections / layer) |

## Table 2 — FuRA on RL (GRPO) for math reasoning (Qwen3-1.7B & Qwen2.5-7B)

GRPO RL on `qwedsacf/competition_math`, evaluated on MATH-500 / AMC23 / AIME-24 /
AIME-25. Source: `run_rl.py` + `run_rl.sh::run_blocktt`. Headline rows per `rl.md`.

| Hyperparameters (FuRA)       | Qwen3-1.7B                                      | Qwen2.5-7B             |
| :--------------------------- | :---------------------------------------------- | :--------------------- |
| Optimizer                    | AdamW                                           | AdamW                  |
| LR                           | 1e-4                                            | 1e-4                   |
| LR Scheduler                 | none (constant)                                 | none (constant)        |
| Warmup ratio                 | 0.0                                             | 0.0                    |
| Weight decay                 | 0.0                                             | 0.0                    |
| GRPO steps                   | 50                                              | 50                     |
| Prompts per step             | 32                                              | 32                     |
| Group size (rollouts/prompt) | 8                                               | 8                      |
| Epochs per step              | 1                                               | 1                      |
| Max model length (vLLM)      | 2048                                            | 2048                   |
| Eval max tokens              | 2048                                            | 2048                   |
| Train rollout temperature    | 1.0                                             | 1.0                    |
| Eval temperature             | 0.6 (AIME-24/25), 0.0 (MATH-500/AMC23)          | 0.6 / 0.0              |
| Seed                         | 42                                              | 42                     |
| Where                        | All linear projections (`trainable_type=all`) | All linear projections |

## Table 3 — FuRA for visual instruction tuning (LLaVA-1.5-7B)

Trained on `llava_v1_5_mix665k` (5,197 steps = 1 epoch) with the LLaVA-1.5 + DoRA
fine-tune recipe (`ref/DoRA/visual_instruction_tuning/Dora_7b.sh`), with FuRA
(BlockTT-full output_one_block, small core, `keep_trainable`) substituted in
place of DoRA. Best LR = 3e-4 in our sweep (`docs/exp_results/vlm.md`).

| Hyperparameters (FuRA)                | LLaVA-1.5-7B                                                  |
| :------------------------------------ | :------------------------------------------------------------ |
| Dropout                               | 0.05                                                          |
| Optimizer                             | AdamW                                                         |
| LR                                    | 3e-4 (best of {2e-4, 3e-4, 4e-4} sweep)                       |
| `mm_projector_lr`                   | 2e-5                                                          |
| LR Scheduler                          | Cosine decay                                                  |
| Weight decay                          | 0.0                                                           |
| Batch size (per device × grad-accum) | 4 × 4 = 16                                                   |
| Warmup ratio                          | 0.03                                                          |
| Epochs                                | 1                                                             |
| Model max length                      | 2048                                                          |
| Mixed precision                       | bf16                                                          |
| Gradient checkpointing                | on                                                            |
| Where                                 | Q, K, V, O, Up, Down, Gate (all 7 linear projections / layer) |

## Table 4 — qFuRA on MetaMathQA-100K (LLaMA-3-70B, single H100)

Single-GPU 70B fine-tune on `metamath_100k.json` (100K examples, 1 epoch). Mirrors
the QPiSSA recipe (`lr=2e-5, batch=1×128, seq=512`) but substitutes qFuRA's
quantized BlockTT decomposition for PiSSA's rank-r SVD adapter. The 70B model
never fully materialises in bf16 on GPU at any point — `--load_strategy=layer_stream`
streams layers in. Source script: `ref/LIFT/bash_scripts/finetune_metamath_qfura_70b.sh`.

| Hyperparameters (qFuRA)               | LLaMA-3-70B                |
| :------------------------------------ | :------------------------- |
| Optimizer                             | Paged AdamW 8-bit          |
| LR                                    | 2e-5                       |
| LR Scheduler                          | Cosine decay               |
| Weight decay                          | 0.0                        |
| Batch size (per device × grad-accum) | 1 × 128 = 128             |
| Warmup ratio                          | 0.03                       |
| Steps                                 | 100                        |
| Max sequence length                   | 512                        |
| Mixed precision                       | bf16                       |
| Trainable param dtype                 | bf16                       |
| Seed                                  | 42                         |
| Where                                 | Q, K, V, O, Up, Down, Gate |

---

Notes:

- All four settings use the project-wide FuRA defaults (`decomp_mode=output_one_block`,
  `blocktt_rank=full`, `train_position=small`); the only setting-specific knob is
  `s_merged_to` (`frozen` only for the Qwen3-1.7B RL paper row, `keep_trainable`
  elsewhere). Table 4 is qFuRA = FuRA + 8-bit-quantized base weights via the
  `quant_block_layout=flat` storage layout; the BlockTT cores themselves remain
  bf16 trainable.
- "Where" matches DoRA's column for the analogous table — all linear projections
  inside each transformer block are converted to BlockTT layers; embeddings,
  LM head, layer norms, and (for LLaVA) the vision tower / projector remain
  unchanged.
- Param-counting for LLaVA-1.5-7B: with the recipe above FuRA trains ~96.6 M
  parameters = **1.37 %** of the 7.06 B-param model — ~3× fewer trainables than
  DoRA r=128 (4.63 %) at comparable average accuracy. See `vlm.md` for the
  derivation.
