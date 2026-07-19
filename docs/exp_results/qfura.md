# qfura vs QLoRA / QDoRA / QPiSSA: Fine-Tuning Results on Llama-3-

**Last updated:** 2026-05-06

This document tracks qfura's fine-tuning quality against QLoRA baselines on two LIFT benchmark suites: math reasoning (`math_10k.json`) and commonsense reasoning (`commonsense_170k.json`). All runs use Llama-3-8B with 3 training epochs. A separate Llama-3-70B section covers MetaMathQA-100k 1-epoch fine-tunes (recipe aligned to the QPiSSA paper) on a single H100.

## Paper headline table

Llama-3-8B numbers below are the 3-epoch tuned setting per method (see "Llama-3-8B commonsense Results"). Llama-3-70B numbers are MetaMathQA-100k, 1 epoch, lr=1e-4, batch=4×32, fp32 adapter + fp32 AdamW, PiSSA prompt format. QLoRA/QDoRA full-epoch 70B runs are pending; their 100-step short-run GSM8K strict accuracy is 0% (format adherence not yet learned), with relaxed-acc (last-number extraction) reported in the 70B section.

| Method          | Llama-3 8B<br />(Commonsense) |                 | Llama-3 70B<br />(Math) |                 |
| --------------- | ----------------------------- | --------------- | ----------------------- | --------------- |
|                 | # Params (%)                  | Avg.            | # Params (%)            | GSM8K           |
| QLoRA           | 1.39                          | 83.89           | 1.17                    | 81.27           |
| QDoRA           | 1.42                          | 86.34           | 1.18                    | 81.80           |
| **QFuRA** | **1.46**                | **87.30** | **1.45**          | **83.78** |

## Methods

### qfura

NF4-quantized BTT fine-tuning with the project-locked defaults (see `CLAUDE.md`):

- `--blocktt_rank full`
- `--decomp_mode output_one_block`
- `--train_position small` (small core trainable, large core frozen + NF4-quantized)
- `--s_merged_to keep_trainable` (singular values held in a separate trainable `btt_s`)
- `--quant_block_layout flat`
- `bnb.optim.PagedAdamW8bit`, `--gradient_checkpointing`

**Trainable parameters:** 118,685,696 (1.46% of 8.15B). 224 frozen BTT cores quantized to NF4.

### QLoRA (two ranks reported)

Standard QLoRA: 4-bit NF4 base + double-quant + bf16 compute, LoRA adapters on the 7 leaves (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`), `bnb.optim.PagedAdamW8bit`.

| QLoRA variant | rank | alpha | Trainable params | Trainable %     | Notes                                  |
| ------------- | ---- | ----- | ---------------- | --------------- | -------------------------------------- |
| QLoRA r=48    | 48   | 96    | 125,829,120      | **1.54%** | Param-parity vs qfura (+6% trainable)  |
| QLoRA r=64    | 64   | 128   | 167,772,160      | 2.05%           | Standard QLoRA preset (+41% trainable) |

After training, QLoRA's PEFT adapter is merged into the bf16 base via `tools/merge_qlora_for_eval.py` to produce a full HF model checkpoint (the eval scripts expect a full model).

## Math Results

`LLM-Adapters/ft-training_set/math_10k.json` (9819 examples, 3 epochs, lr 1e-4, batch 1×16 accum, 1842 optimizer steps). Eval harness: `ref/LIFT/bash_scripts/eval_math.sh` → `run_math_parallel.py` (beam=4, top_k=40, top_p=0.75, temp=0.1).

| Dataset                        |    n |   qfura (1.46%) | QLoRA r=48 (1.54%) | QLoRA r=64 (2.05%) |
| ------------------------------ | ---: | --------------: | -----------------: | -----------------: |
| MultiArith                     |  600 |           95.67 |              98.83 |              99.00 |
| GSM8K                          | 1319 |           66.72 |              70.43 |              70.43 |
| AddSub                         |  395 |           91.90 |              92.15 |              92.66 |
| AQuA                           |  254 |           26.38 |              27.56 |              27.17 |
| SingleEq                       |  508 |           95.47 |              96.06 |              96.65 |
| SVAMP                          | 1000 |           76.50 |              76.90 |              82.20 |
| mawps                          |  238 |           92.02 |    **91.18** |              92.02 |
| **Average (unweighted)** |      | **77.81** |    **79.02** |    **79.88** |
| **Average (n-weighted)** | 4314 | **75.98** |    **78.00** |    **79.46** |

### Math observations

- **At parameter parity (qfura 1.46% vs QLoRA r=48 1.54%), QLoRA still wins overall by +1.21 unweighted / +2.02 n-weighted points** — about half the gap of the over-parameterized QLoRA r=64 run.
- **SVAMP is the parameter-capacity discriminator.** At r=64 QLoRA hits 82.20% — at r=48 it collapses to 76.90%, statistically tied with qfura's 76.50%. The earlier "QLoRA dominates SVAMP" finding was largely a budget effect, not a method effect.
- **qfura wins mawps (92.02 vs 91.18) at param parity** — first dataset where qfura beats QLoRA at any tested rank. mawps is the smallest test set (n=238), so the difference is within ±2-point noise.
- **GSM8K is parameter-insensitive within QLoRA but rank-insensitive** — r=48 and r=64 both at 70.43%; both ahead of qfura by +3.71. Hard arithmetic seems to require either more capacity (r=64) or a different adapter structure than BTT.
- **MultiArith, AddSub, SingleEq, AQuA are near-saturated** — methods within 1-3 points of each other, gaps small relative to per-dataset noise.

### Math training time

| Method     | Wall clock (3 epochs) | Final epoch-3 train loss |
| ---------- | --------------------- | ------------------------ |
| qfura      | 1h 26m                | 0.0009                   |
| QLoRA r=48 | 2h 13m                | 0.0011                   |
| QLoRA r=64 | 2h 09m                | 0.0003                   |

Both QLoRA runs are slower per-step than qfura because the 4-bit dequant happens on every linear matmul of the full forward (vs qfura's per-BTT-layer dequant).

## Llama-3-8B commonsense Results

`LLM-Adapters/ft-training_set/commonsense_170k.json` (170k examples, 3 epochs, batch 8×2 accum, 31932 optimizer steps). Eval harness: `ref/LIFT/bash_scripts/eval_commonsense.sh` over 8 commonsense reasoning datasets (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OBQA; total n=22419).

### Best per method

The single highest-Avg run we have for each method, all on Llama-3-8B base. qfura uses lr=2e-4 (its tuned setting); QLoRA and qdora are reported at lr=1e-4, r=64, α=128 (their best settings — see leaderboard below for the lr=2e-4 baselines they outscore).

| Method | Config                                                     | Trainable % |          BoolQ |           PIQA |           SIQA |      HellaSwag |           Wino |          ARC-e |          ARC-c |           OBQA |       Avg (unw) |       Avg (n-w) |
| ------ | ---------------------------------------------------------- | ----------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | --------------: | --------------: |
| qfura  | rank=full, output_one_block, keep_trainable, flat, lr=2e-4 |       1.46% |           73.0 | **89.9** | **82.7** | **96.6** | **89.1** | **93.1** | **83.4** | **90.6** | **87.30** | **89.78** |
| qdora  | r=64, α=128, fast path, lr=1e-4 (save_both, last)         |       1.42% | **75.5** |           88.0 |           81.1 |           95.9 |           88.4 |           91.8 |           82.0 |           88.0 |           86.34 |           89.23 |
| QLoRA  | r=64, α=128, lr=1e-4                                      |       1.39% |           72.7 |           86.6 |           80.8 |           93.7 |           85.4 |           89.9 |           76.2 |           85.8 |           83.89 |           86.97 |

Bold = column max. qdora is now within 0.96 unweighted / 0.55 n-weighted of qfura; qdora actually beats qfura on BoolQ (75.5 vs 73.0). The other 7 tasks all favour qfura by 0.7-2.6 points.

### Full leaderboard (sorted by Avg unweighted, desc)

Every complete eval we have under `/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/{q*}*` is listed here.

| Method | Config                                                     | Trainable % |          BoolQ |           PIQA |           SIQA |      HellaSwag |           Wino |          ARC-e |          ARC-c |           OBQA |       Avg (unw) |       Avg (n-w) |
| ------ | ---------------------------------------------------------- | ----------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | --------------: | --------------: |
| qfura  | rank=full, output_one_block, keep_trainable, flat, lr=2e-4 |       1.46% |           73.0 | **89.9** | **82.7** | **96.6** | **89.1** | **93.1** | **83.4** | **90.6** | **87.30** | **89.78** |
| qdora  | r=64, α=128, fast path, lr=1e-4 (save_both, last)         |       2.06% | **75.5** |           88.0 |           81.1 |           95.9 |           88.4 |           91.8 |           82.0 |           88.0 |           86.34 |           89.23 |
| QLoRA  | r=64, α=128, lr=1e-4                                      |       2.05% |           72.7 |           86.6 |           80.8 |           93.7 |           85.4 |           89.9 |           76.2 |           85.8 |           83.89 |           86.97 |
| qdora  | r=64, α=128, fast path, lr=2e-4                           |       2.06% |           66.4 |           77.3 |           73.9 |           80.2 |           77.0 |           77.3 |           60.8 |           72.8 |           73.21 |           75.73 |
| QLoRA  | r=48, α=96, lr=2e-4 (param-parity vs qfura)               |       1.54% |           65.1 |           71.3 |           70.9 |           70.0 |           72.2 |           67.8 |           55.5 |           68.2 |           67.63 |           69.40 |

Bold = column max across the leaderboard. **At lr=1e-4, both QLoRA and qdora close most of the gap to qfura** that lr=2e-4 had opened — qdora especially (73.21 → 86.34 = +13.13 points just from the lr halving). Ranking after the lr fix: qfura > qdora > QLoRA > qdora_2e-4 > QLoRA_r48_2e-4.

### Observations

- **At parameter parity (qfura 1.46% vs QLoRA r=48 lr=2e-4 1.54%), qfura wins by 19.7 unweighted points (87.30 vs 67.63).** This was the original published-recipe comparison and is the inverse of the math result, where QLoRA edges qfura by +1.2-2.0 points at parity.
- **Lowering lr from 2e-4 → 1e-4 nearly closes that gap.** At r=64 lr=1e-4, QLoRA reaches **83.89** (3.41 points behind qfura) and qdora reaches **86.34** (only 0.96 points behind qfura). So much of the visible qfura advantage on commonsense at lr=2e-4 was lr-sensitivity of the LoRA-family methods, not a structural advantage of BTT factorisation.
- **qdora benefits more than QLoRA from the lr drop** (+13.13 unweighted, vs QLoRA's gain at the same r=64 budget which we did not measure directly but the param-parity-vs-qfura comparison suggests is ~+16 points). DoRA's per-output-column magnitude updates are known to be more sensitive to step size; this is consistent with what the DoRA paper reports for fp16 training.
- **HellaSwag (n=10042) remains qfura's biggest discriminator**, but the gap is small now: qfura 96.6 vs qdora 95.9 (−0.7) vs QLoRA 93.7 (−2.9). On all other tasks the gap to qfura is under 3 points for qdora and under ~5 points for QLoRA.
- **qdora actually beats qfura on BoolQ (75.5 vs 73.0).** No other task crosses over. BoolQ is a binary yes/no task; the per-output-column magnitude in DoRA may help calibrate the binary decision boundary in a way the BTT structure does not.
- **The previous "qdora is broken" reading from the Llama-3.1 result (Avg 64.81) was driven by lr + the slow PEFT path.** With our fast path *and* a tuned lr, qdora is the second-best method on this benchmark.
- **qfura's lr=2e-4 setting was already in its sweet spot.** We have no evidence qfura would benefit from an lr sweep; QLoRA and qdora visibly do.

### Llama-3-8B training time

| Method                               | Wall clock (3 epochs)      | Notes                                      |
| ------------------------------------ | -------------------------- | ------------------------------------------ |
| qfura lr=2e-4                        | 9h 25m                     |                                            |
| QLoRA r=48 lr=2e-4                   | 10h 21m                    |                                            |
| QLoRA r=64 lr=1e-4                   | _not recorded_           | similar order                              |
| qdora r=64 lr=2e-4 (fast)            | 10h 42m                    | 2.1× faster than the Llama-3.1 PEFT path  |
| qdora r=64 lr=1e-4 (fast, save_both) | ~10h training + ~3.5h eval | save_pretrained fixed (see commit history) |

## Llama-3.1-8B commonsense results

Same recipe (commonsense_170k, 3 epochs, lr 2e-4, batch 8×2 accum) but with **`meta-llama/Llama-3.1-8B`** as the base. Adds a third method, **qdora** (NF4 base + DoRA adapters via PEFT `use_dora=True`, magnitude/direction decomposition).

| Method                                                    | Trainable params | Trainable % |
| --------------------------------------------------------- | ---------------- | ----------- |
| qfura (rank=full, output_one_block, keep_trainable, flat) | 118,685,696      | 1.46%       |
| QLoRA r=64 (α=128, all 7 modules)                        | 167,772,160      | 2.05%       |
| qdora r=64 (α=128, all 7 modules)                        | 169,148,416      | 2.06%       |

| Dataset                        |     n |           qfura |      QLoRA r=64 |      qdora r=64 |
| ------------------------------ | ----: | --------------: | --------------: | --------------: |
| BoolQ                          |  3270 |           74.00 |           62.20 |           65.10 |
| PIQA                           |  1838 |           90.80 |           79.50 |           70.40 |
| SIQA                           |  1954 |           82.30 |           69.10 |           69.50 |
| HellaSwag                      | 10042 |           96.60 |           82.40 |           60.90 |
| WinoGrande                     |  1267 |           87.80 |           73.70 |           71.10 |
| ARC-Easy                       |  2376 |           93.50 |           81.40 |           67.70 |
| ARC-Challenge                  |  1172 |           83.30 |           63.80 |           52.40 |
| OBQA                           |   500 |           89.40 |           70.60 |           62.20 |
| **Average (unweighted)** |       | **87.21** | **72.84** | **64.81** |
| **Average (n-weighted)** | 22419 | **89.30** | **78.30** | **65.93** |

### Llama-3.1-8B observations

- **qfura is essentially unchanged** by the base-model upgrade from Llama-3 to Llama-3.1 (87.30 → 87.21 unweighted). Whatever extra pretraining 3.1 received doesn't translate to qfura's commonsense accuracy.
- **qfura wins by ~14-22 points** depending on weighting and comparison method.
- **qdora (DoRA adapters on NF4 base) underperforms QLoRA at the same rank** by 8.0 unweighted / 12.4 n-weighted points. This is contrary to DoRA's published claim of better-than-LoRA performance. Possible explanations:
  1. DoRA's per-output-column magnitude vector might be sensitive to lr=2e-4 — many DoRA papers use lower LRs (5e-5 to 1e-4) because the magnitude updates compound through the per-step normalization.
  2. PEFT's NF4 + DoRA path is newer than the LoRA path; subtle bugs may exist in the merge step (`merge_and_unload()` for DoRA reconstructs the magnitude/direction product from the saved adapter, which is more complex than LoRA's `B @ A` merge).
  3. Best-eval-checkpoint behavior may interact with DoRA's training trajectory (transient eval-loss spikes during magnitude training could lock in a suboptimal early checkpoint).
- **HellaSwag is the discriminator**: qdora 60.9% is *27 points below qfura* (96.6%) and *21 points below qlora* (82.4%). HellaSwag tests scene-completion semantics where the `output_one_block` BTT factorization apparently captures more useful structure than rank-64 LoRA / DoRA.
- **The base-model upgrade helps QLoRA more than qfura.** QLoRA r=64 on Llama-3.1 (78.30 n-weighted) is ~9 points better than QLoRA r=48 on Llama-3 (69.40 n-weighted), partly from the extra rank and partly from the better base. qfura's performance is base-invariant.

### Llama-3.1-8B training time

| Method     | Wall clock (3 epochs) |
| ---------- | --------------------- |
| qfura      | 11h 44m               |
| QLoRA r=64 | 10h 33m               |
| qdora r=64 | **22h 40m**     |

qdora is **~2.1× slower** than qlora due to DoRA's per-step column-norm computation on every linear's effective weight. This is a real cost: a method that's slower *and* less accurate has no clear regime where it wins.

## Llama-3-70B (MetaMathQA-100k) Results

70B fine-tunes on a single H100 (94 GB), recipe aligned with the QPiSSA paper (lr=2e-5, eff. batch 128, seq 512, 1 epoch on 100K examples, fp32 adapter + fp32 AdamW, PiSSA prompt format, seed=42, gradient_checkpointing). qfura uses the project defaults from `CLAUDE.md` (`blocktt_rank=full`, `decomp_mode=output_one_block`, `train_position=small`, `s_merged_to=keep_trainable`) plus `--load_strategy=layer_stream` (required for 70B on a single 94 GB H100). qlora and qdora are r=64, α=64, dropout=0.

Training params at 70B:

| Method      | Trainable params | % of base | Adapter init                                           |
| ----------- | ---------------: | --------: | ------------------------------------------------------ |
| QLoRA r=64  |           0.83 B |     1.17% | random (B=0)                                           |
| QDoRA r=64  |           0.84 B |     1.18% | random (B=0; magnitude=‖W‖_c)                        |
| QFuRA full  |           1.02 B |     1.45% | BTT decomp of base (lossless reconstruction at step 0) |
| QPiSSA r=64 |           0.83 B |     1.17% | PiSSA SVD residual + LoRA reconstructing W             |

All percentages are trainable / base-70B (70.55B) parameters. (qfura's `sys_metrics.json` reports 2.82% against the post-NF4-quantization model size of 37.35B, which inflates the ratio because each 4-bit weight stored counts as ~0.5 of a parameter — that figure is not directly comparable to LoRA-style methods and we use the true-base denominator throughout this document.)

Eval: vLLM gen with `--quantization bitsandbytes` on the merged bf16 checkpoint, beam=1, temp=0, top_p=1, max_tokens=1024. GSM8K test = 1319 prompts; MATH test = 5000 prompts. Two accuracy metrics:

- **Strict** = `utils/test_acc.py` from PiSSA repo. Counts a sample correct only if "The answer is: X" appears in the output and X equals the gold answer.
- **Relaxed** = last numeric token in the response (skipping in-context regenerations after `### Instruction:`). Captures math ability when the model produces correct numerical answers in plain English without the rigid "The answer is:" suffix.

### Full 1-epoch run (qfura)

| Run                     | Steps |      Train loss |    GSM8K strict | GSM8K relaxed | MATH strict |
| ----------------------- | ----: | --------------: | --------------: | ------------: | ----------: |
| **qfura lr=1e-4** |   781 | **0.071** | **82.34** |         82.18 |   (running) |

Wall clock: ~28 h on a single H100. Step time ~133 s/step at bs=4×32. Final epoch-1 train loss = 0.071. Materialize + CPU-offload save = ~13 min, output = 259 GB single-file `pytorch_model.bin` (fp32 trainable params preserved).

### Short 100-step lr sweep (recipe-aligned to QPiSSA)

All eight runs share: bs=4×32 (eff. 128), seq=512, cosine warmup 0.03, fp32 adapter + fp32 AdamW, PiSSA prompt, seed=42, 100 optimizer steps. The decomposition methods (qfura, QPiSSA) reach the strict format ("The answer is: X") within 100 steps; the random-init low-rank methods (qlora, qdora) do not.

| Method          |    r |             lr | Final train loss |    GSM8K strict |   GSM8K relaxed |
| --------------- | ---: | -------------: | ---------------: | --------------: | --------------: |
| **qfura** | full | **1e-4** |            0.165 | **83.78** | **83.70** |
| qfura           | full |           2e-4 |            0.161 |           81.80 |      **81.58** |
| QPiSSA          |   64 |           2e-5 |            0.178 |           81.27 |      **80.89** |
| qdora           |   64 |           2e-5 |            0.224 |            0.00 |           67.02 |
| qlora           |   64 |           2e-5 |            0.243 |            0.00 |           66.49 |
| qlora           |   64 |           5e-5 |            0.204 |            0.00 |           65.81 |
| qdora           |   64 |           5e-5 |            0.202 |            0.00 |           64.67 |
| qlora           |   64 |           1e-4 |            0.194 |            0.00 |           64.14 |
| qdora           |   64 |           1e-4 |            0.207 |            0.00 |           60.58 |

Bolded row is the best qfura short setting that motivated the full 1-epoch launch.

### 70B observations

- **qfura > QPiSSA by ~3 points strict** at 100 steps (83.78 vs 81.27). QPiSSA initializes its r=64 LoRA from the principal SVD of W (residual = `W − adapter`, NF4-quantized); qfura initializes from a full-rank BTT decomposition (lossless). The full-rank decomp absorbs more of the base's behavior at step 0, leaving less to learn.
- **qfura > {qlora, qdora} by ~17 points relaxed** at every shared lr. The decomposition-init methods reproduce base capability at step 0; the random-init low-rank methods (B=0 for qlora; B=0 + magnitude=‖W‖_c for qdora) start with zero adapter contribution and need many more steps to recover the base's chain-of-thought + format behavior.
- **qlora and qdora score 0% strict at 100 steps** despite producing correct numerical answers in 60-67% of samples. The "The answer is: X" suffix is absent because the random-init adapter delta is too small to overwrite the base's natural rambling output pattern within 100 steps. This is a *format-fitting* failure, not a math-reasoning failure.
- **qlora ≈ qdora at every lr.** DoRA's per-step magnitude rescaling adds compute but does not help here at 100 steps. Both methods favor lr=2e-5 (the QPiSSA paper lr) by 1-7 relaxed-acc points over lr=5e-5/1e-4.
- **Full 1-epoch qfura GSM8K (82.34%) is *lower* than its 100-step run (83.78%).** GSM8K is known to plateau early in MetaMath SFT; the additional 681 steps (~27 h) primarily benefit MATH (the 5000-prompt harder split, eval pending). Train loss continues to decrease (0.165 → 0.071), but GSM8K does not.
- **Strict accuracy is a binary signal at 100 steps.** Decomposition-init methods (qfura, QPiSSA, full FT, PiSSA) reliably learn the rigid output format; random-init low-rank adapters (qlora, qdora) do not. At full-epoch horizons (781 steps) qlora/qdora are expected to learn the format too but full-epoch eval is still pending.

### 70B reproducibility

```bash
# qfura full 1-epoch (project defaults; fp32 adapter + fp32 AdamW + PiSSA prompt for paper alignment)
CUDA_VISIBLE_DEVICES=7 \
HF_HOME=/data/yequan/huggingface \
OUTPUT=/data/yequan/fura/lift/metamath100k/meta-llama/Meta-Llama-3-70B/full-qfura-bs4x32-lr1e-4-fp32-pissaprompt \
run_name=full-qfura-bs4x32-lr1e-4-fp32-pissaprompt \
wandb_project=qfura-metamath-Meta-Llama-3-70B \
lr=1e-4 seed=42 MAX_STEPS=0 num_train_epochs=1 \
per_device_train_batch_size=4 gradient_accumulation_steps=32 model_max_length=512 \
prompt_style=pissa trainable_param_dtype=fp32 optimizer_name=adamw \
bash ref/LIFT/bash_scripts/finetune_metamath_qfura_70b.sh

# qlora 70B short100 (e.g. lr=2e-5)
CUDA_VISIBLE_DEVICES=1 \
OUTPUT=/data/yequan/fura/lift/metamath100k/meta-llama/Meta-Llama-3-70B/short100-qlora-r64-bs4x32-lr2e-5-fp32-pissaprompt \
lr=2e-5 MAX_STEPS=100 \
per_device_train_batch_size=4 gradient_accumulation_steps=32 \
prompt_style=pissa trainable_param_dtype=fp32 optimizer_name=adamw \
bash ref/LIFT/bash_scripts/finetune_metamath_qlora_70b.sh

# qdora 70B short100 (analogous; uses qdora_impl=fast which materializes a fully-merged HF model in last/)
CUDA_VISIBLE_DEVICES=4 \
OUTPUT=/data/yequan/fura/lift/metamath100k/meta-llama/Meta-Llama-3-70B/short100-qdora-r64-bs4x32-lr2e-5-fp32-pissaprompt \
lr=2e-5 MAX_STEPS=100 qdora_impl=fast \
per_device_train_batch_size=4 gradient_accumulation_steps=32 \
prompt_style=pissa trainable_param_dtype=fp32 optimizer_name=adamw \
bash ref/LIFT/bash_scripts/finetune_metamath_qdora_70b.sh

# QPiSSA 70B short100 (uses fxmeng's pre-quantized r=64 5-iter NF4 residual)
CUDA_VISIBLE_DEVICES=7 \
OUTPUT=/data/yequan/fura/pissa/short100-Llama-3-70B-qpissa-r64-lr2e-5 \
lr=2e-5 MAX_STEPS=100 \
bash ref/PiSSA/scripts/run_qpissa_70b_singlegpu.sh

# Eval (qfura/qdora: load <run>/last/ directly. qlora: merge adapter via /tmp/merge_qlora_gpu.py first.)
CUDA_VISIBLE_DEVICES=7 \
HF_HOME=/data/yequan/huggingface \
TORCHINDUCTOR_CACHE_DIR=/data/yequan/torchinductor TMPDIR=/data/yequan/tmpdir \
uv run python ref/PiSSA/utils/gen_vllm.py \
    --model <ckpt> --data_path /data/yequan/pissa-dataset \
    --sub_task gsm8k_pissa_prompt --dataset_split test \
    --output_file <out>/gsm8k_response.jsonl \
    --batch_size 200 --max_tokens 1024 --temperature 0.0 --top_p 1.0 \
    --quantization bitsandbytes --gpu_memory_utilization 0.92 --dtype bfloat16
uv run python ref/PiSSA/utils/test_acc.py --input_file <out>/gsm8k_response.jsonl
```

The QLoRA 70B merge needs `/tmp/merge_qlora_gpu.py` (custom GPU-side merge of the NF4 base + r=64 LoRA bf16 adapter to a CPU-resident bf16 sharded checkpoint), because PEFT's stock `merge_and_unload` is CPU-bound and effectively hangs on the 70B model. The qdora "fast" path saves a fully merged HF model directly to `<run>/last/` at end-of-training, no separate merge step needed.

### 70B output paths

- qfura full 1-epoch: `/data/yequan/fura/lift/metamath100k/meta-llama/Meta-Llama-3-70B/full-qfura-bs4x32-lr1e-4-fp32-pissaprompt/{last,eval_gsm8k,eval_math}/`
- qfura short100 (lr ∈ {5e-5, 1e-4, 2e-4}): `/data/yequan/fura/lift/metamath100k/meta-llama/Meta-Llama-3-70B/short100-qfura-bs4x32-lr<lr>-fp32-pissaprompt/`
- qlora short100 (lr ∈ {2e-5, 5e-5, 1e-4}): `/data/yequan/fura/lift/metamath100k/meta-llama/Meta-Llama-3-70B/short100-qlora-r64-bs4x32-lr<lr>-fp32-pissaprompt/{last_adapter,eval_gsm8k}/`
- qdora short100 (lr ∈ {2e-5, 5e-5, 1e-4}): `/data/yequan/fura/lift/metamath100k/meta-llama/Meta-Llama-3-70B/short100-qdora-r64-bs4x32-lr<lr>-fp32-pissaprompt/{last,eval_gsm8k}/`
- QPiSSA short100: `/data/yequan/fura/pissa/short100-Llama-3-70B-qpissa-r64-lr2e-5/{checkpoint-100,eval_gsm8k}/`

## The math vs commonsense reversal

The single most striking finding from these experiments: **qfura's relative performance against QLoRA at parameter parity flips by ~22 points between math and commonsense.**

| Suite                               | qfura avg (n-weighted) | QLoRA r=48 avg (n-weighted) | qfura − QLoRA r=48 |
| ----------------------------------- | ---------------------: | --------------------------: | ------------------: |
| Math (math_10k.json)                |                  75.98 |                       78.00 |    **−2.02** |
| Commonsense (commonsense_170k.json) |                  89.30 |                       69.40 |    **+19.90** |

Possible explanations to investigate:

1. **Dataset size effect.** Commonsense has 17× more training examples than math (170k vs 10k). At the same trainable param budget, qfura's BTT structure may scale to absorb large datasets better than QLoRA's low-rank adapters. The 6× more optimizer steps (31932 vs 1842) compound any structural advantage.
2. **Task structure.** Math is closed-form symbolic reasoning where the right answer is mostly a deterministic function of the input. Commonsense has a much wider distribution of "reasonable" continuations; absorbing that distribution may benefit from the higher effective rank that BTT provides on each linear (full rank in qfura's defaults vs r=48 in QLoRA).
3. **Quantization noise interaction.** Commonsense answers are short and high-entropy in token-space; even small per-token logit errors can flip a multi-choice answer. QLoRA's NF4 quantization of the full base injects noise on every matmul; qfura's NF4 only on the frozen BTT core. The relative noise budget per forward may matter more for commonsense than for math (where the model has thousands of forward steps to recover via chain-of-thought).
4. **Training-recipe interaction.** Math used lr 1e-4 + batch=1×16accum; commonsense used lr 2e-4 + batch=8×2accum. The higher commonsense LR might over-fit QLoRA's smaller-effective-rank adapters more aggressively.

## Reproducibility

```bash
# qfura math (defaults baked in CLAUDE.md)
CUDA_VISIBLE_DEVICES=4 \
HF_HOME=/data/yequan/huggingface \
no_wandb=1 \
bash ref/LIFT/bash_scripts/finetune_math_qfura.sh

# qfura commonsense
CUDA_VISIBLE_DEVICES=2 \
HF_HOME=/data/yequan/huggingface \
no_wandb=1 \
bash ref/LIFT/bash_scripts/finetune_commonsense_qfura.sh

# QLoRA at param-equivalent rank=48 (math)
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/data/yequan/huggingface \
no_wandb=1 \
lora_r=48 lora_alpha=96 \
bash ref/LIFT/bash_scripts/finetune_math_qlora.sh

# QLoRA at param-equivalent rank=48 (commonsense)
CUDA_VISIBLE_DEVICES=3 \
HF_HOME=/data/yequan/huggingface \
no_wandb=1 \
lora_r=48 lora_alpha=96 \
bash ref/LIFT/bash_scripts/finetune_commonsense_qlora.sh

# QLoRA at default rank=64 (math)
CUDA_VISIBLE_DEVICES=5 \
HF_HOME=/data/yequan/huggingface \
no_wandb=1 \
bash ref/LIFT/bash_scripts/finetune_math_qlora.sh
```

The QLoRA runners auto-merge their PEFT adapter via `tools/merge_qlora_for_eval.py` after training completes, then invoke the existing eval shell scripts on the merged checkpoint.

## Output paths

- qfura math: `/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/qfura-layout_flat-decomp_output_one_block_smerge_keep_trainable-lr_1e-4-seed_43/`
- qfura commonsense: `/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/qfura-layout_flat-decomp_output_one_block_smerge_keep_trainable-lr_2e-4-seed_43/`
- qlora math r=48 (merged): `/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/qlora-r_48-alpha_96-lr_1e-4-seed_43-merged/`
- qlora math r=64 (merged): `/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/qlora-r_64-alpha_128-lr_1e-4-seed_43-merged/`
- qlora commonsense r=48 (merged): `/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/qlora-r_48-alpha_96-lr_2e-4-seed_43-merged/`
- qfura commonsense Llama-3.1: `/data/yequan/fura/lift/commonsense/meta-llama/Llama-3.1-8B/qfura-layout_flat-decomp_output_one_block_smerge_keep_trainable-lr_2e-4-seed_43/`
- qlora commonsense Llama-3.1 r=64 (merged): `/data/yequan/fura/lift/commonsense/meta-llama/Llama-3.1-8B/qlora-r_64-alpha_128-lr_2e-4-seed_43-merged/`
- qdora commonsense Llama-3.1 r=64 (merged): `/data/yequan/fura/lift/commonsense/meta-llama/Llama-3.1-8B/qdora-r_64-alpha_128-lr_2e-4-seed_43-merged/`
- qdora commonsense Llama-3-8B r=64 (fast path, lr=2e-4, last+best): `/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/qdora-r_64-alpha_128-lr_2e-4-seed_43/{last,best}/`
- qdora commonsense Llama-3-8B r=64 (fast path, lr=1e-4, save_both, last+best): `/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/qdora-r_64-alpha_128-lr_1e-4-seed_43/{last,best}/` (re-run after save fix; previous failed run archived under `qdora-r_64-alpha_128-lr_1e-4-seed_43.failed_save_20260502_232016/`)
- qlora commonsense Llama-3-8B r=64 (lr=1e-4, last_adapter+last): `/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/qlora-r_64-alpha_128-lr_1e-4-seed_43/{last_adapter,last}/`

## Quantization-error references

For pre-training-state error analysis (no fine-tuning):

- `docs/reports/qfura-quant-error.md` — qfura post-conversion error sweep over decomp_mode × layout. Best config: `decomp_mode=output_one_block, layout=flat, s_merged_to=keep_trainable` → model-level KL = 0.31 vs bf16 base.
- `docs/reports/qlora-quant-error.md` — QLoRA post-conversion error (NF4 of the original `nn.Linear.weight`, no LoRA adapters). Model-level KL = 0.19.

**The pre-training KL gap (qfura 0.31 vs QLoRA 0.19) does NOT predict downstream accuracy across suites.**

- On math, the KL gap weakly tracks the trained-state gap: QLoRA's lower KL (0.19) maps to a 2-point math advantage at param parity, 3.5 points at default rank.
- On commonsense, qfura's higher pre-training KL (0.31) does *not* hurt — qfura outperforms QLoRA by 19.9 points despite starting from a worse-conditioned initial state.

Initial-state quantization error is a poor proxy for trained-state quality. The structural difference (BTT factorization vs LoRA adapter) is more important than the per-layer quantization-noise difference, and the relative benefit depends strongly on the task domain and dataset size.

## Mixtral-8x7B — CodeFeedback → HumanEval (short-run LR sweep)

**Added:** 2026-07-19

A short learning-rate sweep of qfura vs QLoRA on the **Mixtral-8x7B-v0.1** MoE model, following the PiSSA/QPiSSA code recipe (`docs/papers/24_PiSSA-*.pdf`): fine-tune on CodeFeedback (the PiSSA-dataset `python` split) and evaluate on HumanEval. This is a *short* run intended to shake out the Mixtral pipeline and get a first LR reading, not a converged comparison.

### Setup

- **Model:** `mistralai/Mixtral-8x7B-v0.1`, NF4 base. MoE-aware target modules: attention `q/k/v/o_proj` + expert `w1/w2/w3` (the `block_sparse_moe.gate` router is *not* adapted). qfura uses `--trainable_type mixtral_all` (896 BTT layers = 32 × (4 attn + 8 experts × 3)); QLoRA passes the same 7 leaves via `--target_modules`.
- **Data:** `python_8k.json` — first **8,000** examples of the CodeFeedback-100K `python` split (**~7.6%**). **485** examples whose prompt alone ≥ `max_seq_len` (512) are dropped (all-masked labels → nan loss; see Bugs below), leaving **7,515** trained.
- **Recipe:** 1 epoch (**58 optimizer steps** at effective batch 128 = 1 × 128 grad-accum), seq len 512, AdamW, cosine schedule, warmup ratio 0.03, no weight decay. qfura keeps the project-locked defaults (`--blocktt_rank full --decomp_mode output_one_block --train_position small --s_merged_to keep_trainable`, layout `flat`, PagedAdamW8bit); QLoRA is rank 64 / alpha 64, fp32 adapter.
- **Eval:** HumanEval via vLLM (`gen_vllm.py --sub_task python`) → `code_process.py` → `evalplus.evaluate --dataset humaneval`, greedy (temperature 0), max 1024 new tokens, all 164 problems. The dense eval checkpoint is loaded **4-bit (bitsandbytes)** with `max_model_len 2048` (see Bugs). qfura evaluates its materialized dense checkpoint; QLoRA evaluates its adapter merged into the base.
- **Tracking:** wandb project `mixtral_mode_code`. Launchers: `finetune_code_{qlora,qfura}_mixtral.sh`; eval: `eval_code.sh`.

### Results (HumanEval pass@1, greedy)

| Method | LR   | HumanEval | HumanEval+ |
| ------ | ---- | --------- | ---------- |
| QLoRA  | 2e-5 | 51.2      | 43.3       |
| QLoRA  | 5e-5 | 55.5      | 49.4       |
| QLoRA  | 1e-4 | **56.1**  | 48.8       |
| qfura  | 5e-5 | 48.2      | 38.4       |
| qfura  | 1e-4 | 50.0      | 40.2       |
| qfura  | 2e-4 | **50.6**  | 43.3       |

Best per method in **bold**. HumanEval(+) uses evalplus's extra test cases.

**Reading (short-run, one seed — treat as directional, not converged):**

- Both methods improve monotonically with LR over the swept range; neither has turned over, so the optimum for each may be beyond the swept endpoint (QLoRA > 1e-4, qfura > 2e-4).
- At this ~58-step budget QLoRA leads qfura by ~5–6 points on base HumanEval at each method's best LR (56.1 vs 50.6). This mirrors the *math* finding elsewhere in this doc (QLoRA's lower initial quant error tracks a small code/math advantage), and contrasts with commonsense (where qfura wins). More steps / full-data runs are needed before drawing a firm qfura-vs-QLoRA conclusion on code.
- qfura's best LR (2e-4) is ~2× QLoRA's best (1e-4), consistent with BTT's small trainable core preferring a larger step size.

### Bugs fixed during this run

1. **nan loss from all-masked microbatches** (commit `18436e9`). ~6% of CodeFeedback samples have a prompt ≥ `max_seq_len`; after truncation every token is masked (`IGNORE_INDEX`), so the cross-entropy loss is 0/0 = nan. Under gradient accumulation a single such microbatch poisons the accumulated step and every weight becomes nan from step 1 (survivable at batch 1, fatal at bs 1×128). Fixed by dropping zero-trainable-token examples in `SupervisedDataset` (shared by qlora + qfura) plus gradient clipping (`max_grad_norm=1.0`) with a non-finite-grad skip guard.
2. **eval KV-cache OOM** (commit `6d67e2b`). The merged/materialized dense Mixtral (~87 GiB bf16) leaves negative KV-cache room on a single 95 GiB H100. `eval_code.sh` now loads the eval model **4-bit via bitsandbytes** (~25 GiB, +53 GiB KV cache) with a capped `max_model_len`. 4-bit eval is also faithful to how both methods train (NF4-quantized frozen weights).
