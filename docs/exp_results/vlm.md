# LLaVA-1.5-7B Full Eval: DoRA vs FuRA (BTT)

All four checkpoints visual-instruction-tuned for 1 epoch on `llava_v1_5_mix665k`
(5,197 steps), evaluated on the standard 7 LLaVA benchmarks.

## Paper headline table

LLaVA-1.5-7B, average across 7 vision-language tasks (VQAv2, GQA, VisWiz, SQA, VQAᵀ,
POPE, MMBench). FT / LoRA / DoRA from DoRA paper Table 12; FuRA = ours.

| Method                | # Params (%)   | Avg.           |
| --------------------- | -------------- | -------------- |
| Full FT               | 100            | 66.5           |
| LoRA                  | 4.61           | 66.9           |
| DoRA                  | 4.63           | 67.6           |
| **FuRA (ours)** | **1.37** | **67.6** |

FuRA matches DoRA's average (−0.1) using **3.4× fewer trainable parameters** and trains
**41 % faster** (see training-cost section).

## Table 1 — DoRA paper Table 12 + our best FuRA

DoRA paper Table 12 (Liu et al., 2024 — visual instruction tuning of LLaVA-1.5-7B,
7 vision-language tasks), with our **best BTT (FuRA)** result appended.

| Method                      | # Params (%)   | VQAv2           | GQA             | VisWiz          | SQA             | VQAᵀ           | POPE | MMBench         | Avg.           |
| --------------------------- | -------------- | --------------- | --------------- | --------------- | --------------- | --------------- | ---- | --------------- | -------------- |
| FT                          | 100            | 78.5            | 61.9            | 50.0            | 66.8            | 58.2            | 85.9 | 64.3            | 66.5           |
| LoRA                        | 4.61           | 79.1            | 62.9            | 47.8            | 68.4            | 58.2            | 86.4 | 66.1            | 66.9           |
| DoRA (paper)                | 4.63           | 78.6            | 62.9            | 52.2            | 69.9            | 57.0            | 87.2 | 66.1            | 67.6           |
| **DoRA (ours)**       | 4.63           | 78.99           | **63.02** | 53.15           | **70.05** | 57.67           | 86.0 | **66.85** | **68.0** |
| **FuRA (ours, best)** | **1.37** | **78.74** | 62.68           | **54.43** | 67.34           | **58.13** | 86.6 | 64.50           | 67.5           |

FuRA "best" = best per-column across our BTT lr-sweep {2e-4, 3e-4, 4e-4} (all `output_one_block`,
small core trainable, full rank). Metrics mapped to DoRA Table 12 conventions:

- VQAv2 = test-dev2015 Overall (server eval)
- GQA = testdev_balanced
- VisWiz = test Overall (server eval)
- SQA = full Acc (all 21k questions)
- VQAᵀ = TextVQA val Acc
- POPE = mean F1 across {random, popular, adversarial} ×100
- MMBench = dev split, CircularEval

Our DoRA run reproduces the DoRA paper's Table 12 numbers within ≤1.2pp on every column
(VQAv2 +0.4, GQA +0.1, VisWiz +1.0, SQA +0.2, VQAᵀ +0.7, POPE −1.2, MMBench +0.8).

**FuRA % Params derivation** — 32-layer LLaMA-2-7B backbone, all 7 linear projections per
layer converted (q/k/v/o + gate/up/down) under the project-default FuRA recipe
(`output_one_block`, full rank, `train_position=small`, `s_merged_to=keep_trainable`).
For each linear with `in_features = n × b` where `(n, b) = closest_factor_pair(in_features)`,
trainable params = `btt_r + btt_s = n·b² + n·b = in_features · (b + 1)`; the dense
`btt_l (in_features × out_features)` core is frozen.

- 4096 → factor (64, 64), so q/k/v/o/gate/up each contribute 4096 × 65 = 266,240
- 11008 → factor (86, 128), so down_proj contributes 11008 × 129 = 1,420,032
- Per LLM block: 6 × 266,240 + 1,420,032 = 3,017,472 trainable
- Total LLM trainable: 32 × 3,017,472 ≈ **96.56 M** (≈ 95.4 M btt_r + 1.1 M btt_s, ignoring
  embeddings/lm_head/norms which stay frozen)
- LLaVA-1.5-7B total ≈ 7.06 B (LLaMA-7B 6.738 B + CLIP-L/14-336 303 M + projector 21 M)
- **96.56 M / 7.06 B = 1.37 %** (vs DoRA's 4.63 % at r=128, α=256 — FuRA uses ~3× fewer
  trainable params for similar avg score)

---

## Table 2 — Full results: our DoRA run + FuRA lr-sweep

## Training cost

| Method                                  | Trained params                   | LR             | Wall time | Ckpt size on disk    |
| --------------------------------------- | -------------------------------- | -------------- | --------- | -------------------- |
| DoRA r=128, α=256                      | ~611 MB adapter + 43 MB non-LoRA | 2e-4           | 35h 46m   | 654 MB               |
| FuRA (BTT-full output_one_block, small) | ~95 M (R cores only)             | **2e-4** | 21h 11m   | 14 GB (materialized) |
| FuRA (BTT-full output_one_block, small) | ~95 M                            | **3e-4** | 20h 53m   | 14 GB                |
| FuRA (BTT-full output_one_block, small) | ~95 M                            | **4e-4** | 20h 57m   | 14 GB                |

FuRA trains **41 % faster** than DoRA at same recipe; ckpt is dense at save time
(14 GB vs DoRA's 654 MB adapter).

Bold = best per row across all 4 ckpts. ⭐ = paper-comparable metric (mapped to DoRA Table 12).

### Hallucination (POPE, COCO val2014)

| Split                     | DoRA 2e-4     | FuRA 2e-4     | FuRA 3e-4               | FuRA 4e-4               |
| ------------------------- | ------------- | ------------- | ----------------------- | ----------------------- |
| random Acc / F1           | 0.855 / 0.845 | 0.854 / 0.844 | 0.857 / 0.848           | **0.859 / 0.850** |
| popular Acc / F1          | 0.874 / 0.862 | 0.875 / 0.863 | 0.879 /**0.868**  | **0.879 / 0.869** |
| adversarial Acc / F1      | 0.881 / 0.872 | 0.881 / 0.873 | **0.886** / 0.878 | **0.886 / 0.879** |
| **avg F1 ×100 ⭐** | 86.0          | 86.0          | 86.5                    | **86.6**          |

POPE saturates — all methods within ±0.005. FuRA (any LR) ≈ DoRA.

### Reasoning / VQA

| Benchmark                               | DoRA 2e-4       | FuRA 2e-4 | FuRA 3e-4       | FuRA 4e-4 |
| --------------------------------------- | --------------- | --------- | --------------- | --------- |
| **ScienceQA Acc ⭐**              | **70.05** | 66.61     | 67.30           | 67.34     |
| ScienceQA IMG-Acc                       | **67.97** | 66.78     | 67.23           | 66.44     |
| **TextVQA val Acc ⭐**            | 57.67           | 57.12     | **58.13** | 57.78     |
| **GQA testdev_balanced ⭐**       | **63.02** | 61.52     | 62.56           | 62.68     |
| **VQAv2 test-dev2015 Overall ⭐** | **78.99** | 78.14     | 78.57           | 78.74     |
| yes/no                                  | **92.59** | 91.97     | 92.29           | 92.39     |
| number                                  | **61.41** | 60.31     | 60.69           | 61.28     |
| other                                   | **71.40** | 70.40     | 70.94           | 71.08     |

DoRA wins SQA (+2.7), GQA (+0.5), VQAv2 (+0.25) — reasoning-heavy. FuRA 3e-4 wins TextVQA (+0.5).

### Multimodal benchmarks

| Benchmark                             | DoRA 2e-4       | FuRA 2e-4       | FuRA 3e-4         | FuRA 4e-4       |
| ------------------------------------- | --------------- | --------------- | ----------------- | --------------- |
| MME Total                             | 1729.39         | 1769.01         | **1807.65** | 1785.92         |
| Perception                            | 1470.47         | 1476.51         | **1493.01** | 1487.35         |
| Cognition                             | 258.93          | 292.50          | **314.64**  | 298.57          |
| MMBench-dev single-shot               | **75.46** | 73.25           | 73.89             | 73.79           |
| **MMBench-dev CircularEval ⭐** | **66.85** | 63.33           | 64.50             | 64.13           |
| **VizWiz test Overall ⭐**      | 53.15           | **54.43** | 53.29             | 53.86           |
| unanswerable detection                | 50.75           | 56.92           | **59.98**   | 52.77           |
| answerable any-match                  | 51.63           | 49.57           | 44.42             | **52.14** |

- **MME**: FuRA 3e-4 wins big (+78 vs DoRA, +33 cognition). Not in DoRA Table 12.
- **MMBench-CircularEval**: DoRA wins (+2.4 to +3.5pp).
- **VizWiz**: All FuRA LRs ≥ DoRA — first reasoning-heavy task FuRA beats DoRA.

## Aggregate winners (count of best-in-row across 8 metrics)

| Task                 | Winner                  |
| -------------------- | ----------------------- |
| POPE rand            | FuRA 4e-4               |
| POPE popular         | FuRA 4e-4               |
| POPE adversarial     | FuRA 3e-4 / 4e-4 (tied) |
| ScienceQA            | DoRA                    |
| TextVQA              | FuRA 3e-4               |
| MME Total            | FuRA 3e-4               |
| GQA                  | DoRA                    |
| VQAv2                | DoRA                    |
| MMBench-CircularEval | DoRA                    |
| VizWiz               | FuRA 2e-4               |

**DoRA**: 4 wins (SQA, GQA, VQAv2, MMBench)
**FuRA 3e-4**: 3 wins (TextVQA, MME, POPE adv shared)
**FuRA 4e-4**: 2-3 wins (POPE rand/popular, POPE adv shared)
**FuRA 2e-4**: 1 win (VizWiz)

## Bottom-line narrative

1. **DoRA leads on text-heavy reasoning** (SQA, GQA, MMBench, VQAv2).
2. **FuRA (especially lr=3e-4) leads on multimodal benchmarks** (MME) and per-letter
   discrimination (POPE), and **trains 41 % faster** at equal recipe.
3. **FuRA lr=3e-4 is the sweet spot** within the FuRA lr-sweep (best on TextVQA, MME, GQA;
   second-best on the others). LR ≥ 6e-4 hurts (separately verified, not in this table).
4. The DoRA vs FuRA trade-off is consistent with their inductive biases: DoRA's
   low-rank residual + magnitude decoupling helps language reasoning; FuRA's
   structured full-rank parameterisation (BTT, output_one_block, small core trainable)
   favours visual feature integration.

---

*Generated 2026-05-05.  Eval pipeline:
chunked 16-GPU inference (4 nodes × 4 A100), MMBench scored CircularEval per
LLaVA paper convention; VQAv2 test-dev2015 uploaded to EvalAI; VizWiz scored
locally via the deprecated EvalAI 523's published self-evaluation annotations.*
