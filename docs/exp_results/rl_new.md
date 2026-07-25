# Qwen2.5-7B GRPO: this repo vs SimpleRL-Zoo

_Created 2026-07-17. Compares the Qwen2.5-7B RL recipe in this repo
(`run_rl_7B.sh` + `run_rl.py`, results in `rl.md`) against the GRPO recipe from
**SimpleRL-Zoo** ("Investigating and Taming Zero Reinforcement Learning for Open
Base Models in the Wild"), whose code is vendored under `ref/simpleRL-reason`._

## TL;DR

The two are **not the same GRPO**. This repo runs a deliberately lightweight,
clip-free / KL-free single-epoch policy gradient on **one H100 for 50 steps**, to
compare PEFT decomposition *methods* (full / LoRA / DoRA / FuRA) quickly and
cheaply. SimpleRL-Zoo runs **canonical GRPO** (PPO clip + KL-to-reference +
std-normalized group advantage) on the full verl stack at ~20–30× the training
budget, to reproduce "zero-RL from a base model."

Because both the **objective** and the **scale** differ, the absolute numbers are
not directly comparable — `rl.md` already flags this for the base-model rows.

## Sources

- This repo: `run_rl.py` (GRPO loop `run_rl.py:1968-2090`, advantage
  `run_rl.py:1980-1983`, loss `run_rl.py:2034-2048`), launcher `run_rl_7B.sh`,
  CLI defaults `run_rl.py:93-108,252-262`.
- SimpleRL-Zoo: `ref/simpleRL-reason/README.md:154-156` (the published
  Qwen-2.5-7B command), `ref/simpleRL-reason/train_grpo_math_tune_ray.sh`
  (defaults + verl launch), `ref/simpleRL-reason/verl/trainer/config/ppo_trainer.yaml`
  (verl defaults), `ref/simpleRL-reason/verl/trainer/ppo/core_algos.py:111-151`
  (GRPO advantage), `:186-192` (clipped PG loss).

## 1. Algorithmic differences (the important part)

The objective is different on **three axes**. This repo's update is effectively a
single-epoch REINFORCE with a group-mean baseline; SimpleRL-Zoo's is textbook GRPO.

| Axis | This repo (`run_rl.py`) | SimpleRL-Zoo (verl GRPO) |
| :--- | :--- | :--- |
| **PPO clipping** | **None.** Loss is `-ratio · advantage` (`run_rl.py:2035`). `ratio` is recomputed but never clipped. | **Yes**, `clip_ratio=0.2` — `torch.max(pg_losses, pg_losses2)` (`core_algos.py:191`). |
| **KL penalty** | **None in the loss.** KL is only *logged* as a diagnostic (`run_rl.py:2038-2040`). | **Yes**, `use_kl_loss=True`, `kl_loss_coef=1e-4`, `kl_loss_type=low_var_kl`. |
| **Advantage normalization** | Group **mean-centered only**: `adv = r − mean(r)` (`run_rl.py:1980-1982`). No division by std. | Group **mean AND std**: `adv = (r − μ)/(σ + 1e-6)` (`core_algos.py:151`). |
| **Entropy bonus** | None. | `entropy_coeff=0.001`. |
| **Epochs per batch** | 1 (`--epochs-per-step 1`). | 1 (`ppo_mini_batch_size == train_batch_size`). Same. |
| **Loss aggregation** | Per-prompt token-mean, then batch-mean (`run_rl.py:2045-2048`). | verl masked token-mean. |

## 2. Scale / hyperparameter differences

| Hyperparameter | This repo (Qwen2.5-7B) | SimpleRL-Zoo (Qwen-2.5-7B) |
| :--- | :--- | :--- |
| Base model | `Qwen/Qwen2.5-7B` | `Qwen-2.5-7B` (same base) |
| Learning rate | `1e-5` (full-FT winner; PEFT rows `1e-4`) | `5e-7` (fixed; ~20× smaller) |
| Prompts / step | 32 | `train_batch_size=1024` (**32×**) |
| Group / rollout size | 8 (`--group-size 8`) | `rollout_n=8` (same) |
| Samples / step | 32 × 8 = **256** | 1024 × 8 = **8192** (32×) |
| Rollout temperature | 1.0 (`run_rl.py:1163,1217,1263`) | 1.0 (same) |
| Max prompt length | part of `max_model_len=2048` | `max_prompt_length=1024` |
| Max response length | gen capped at **1024** tokens (`run_rl.py:1130,1207,1763`) | `max_response_length=8192` (**8×**) |
| Training length | `n_grpo_steps=50` (fixed) | `total_epochs=20` over the 8K set → ~hundreds of steps (~100 steps / ~15 h on 2×8 H100 per their README) |
| Dataset | `qwedsacf/competition_math`, `train[:7500]` (`run_rl.py:791`) | `simplelr_math_35` (GSM8K + MATH, ~8K; the default the published 7B command uses — `train_grpo_math_tune_ray.sh:40`). The `simplelr_qwen_level3to5` split is an alternative from the README's data-download example. |
| LR schedule | `none` (constant); full-FT stable only at lr ≤ 1e-5 | verl default (warmup + constant) |
| Optimizer / offload | adamw / adamw8bit, single H100, `gacc=256`, `gpu_util=0.25`, grad-ckpt | FSDP + Ray + vLLM, `rollout_tp=2`, 2×8 H100, `gpu_util=0.75` |
| KL-to-ref control | n/a (no ref model) | `kl_ctrl.kl_coef=1e-3` (adaptive controller) + the in-loss KL loss above |

## 3. Why it matters

- **Objective mismatch dominates.** Even if you matched batch size and steps, the
  clip-free / KL-free / no-std-norm update optimizes a different objective than
  GRPO. In particular, dropping the std normalization changes the effective
  per-group step size (groups with low reward variance get larger updates here),
  and dropping the KL term removes the anchor to the base model that SimpleRL-Zoo
  relies on for stable "zero-RL."
- **Response length is the biggest scale gap.** SimpleRL-Zoo's headline finding is
  that response length grows during zero-RL; that behavior needs the 8192-token
  budget. This repo caps generation at 1024 tokens, so long-CoT emergence cannot
  appear regardless of method.
- **50 steps vs ~hundreds.** This repo's 50-step budget is enough to rank PEFT
  methods against each other but far short of SimpleRL-Zoo's convergence horizon.

**Do not** compare this repo's Qwen2.5-7B absolute scores to the SimpleRL-Zoo
paper's Qwen2.5-7B numbers as if they were the same run. To actually reproduce
SimpleRL-Zoo on Qwen2.5-7B, use `scripts/reproduce_simplerl_qwen2_5_7b.sh`
(see next section).

## 4. Reproduction script

`scripts/reproduce_simplerl_qwen2_5_7b.sh` wraps the exact published command
(`ref/simpleRL-reason/README.md:156`) with the data download, Ray cluster launch,
and eval invocation. It runs SimpleRL-Zoo's own verl stack under
`ref/simpleRL-reason`, **not** this repo's `run_rl.py` — the whole point is that
`run_rl.py` cannot reproduce that recipe (see §1). See the header of that script
for the hardware requirement (2×8 H100-80G per the SimpleRL README) and env vars
that must be filled in.
