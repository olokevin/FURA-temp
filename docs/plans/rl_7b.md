# Plan: Reproduce SimpleRL-Zoo GRPO on Qwen2.5-7B via `run_rl.py`

## Context

We want to reproduce SimpleRL-Zoo's Qwen2.5-7B GRPO recipe using **this repo's
`run_rl.py`** training path, not the vendored verl code under
`ref/simpleRL-reason`. The motivating question: is this config-only, or does it
need new features?

**Answer: it needs new features.** `run_rl.py`'s current "GRPO" is a stripped
REINFORCE-with-group-baseline. It is missing the three algorithmic pillars of
canonical GRPO — group-**std** advantage normalization, a **PPO-clip** surrogate,
and a **KL-to-frozen-reference** term — plus an entropy bonus. Config alone
covers only the easy axes (lr, batch sizes, group size, temperature, step count,
optimizer betas/wd/constant-LR). The rest must be implemented.

Scope decided with the user:
- **KL for ALL train modes**, including full-FT (frozen bf16 reference copy).
- **Add a selectable SimpleRL data path** (`simplelr_math_35` parquet + qwen-boxed
  prompt) while keeping the current `competition_math` + `boxed.prompt` path alive;
  prompt is chosen to match the selected dataset.
- **Verification = both paths, manual eyeball** (no automated verl-formula test):
  a short automated smoke run on the `run_rl.py` side, plus a documented one-time
  manual procedure to run the reference verl path on 1 GPU and compare logged
  stats by eye.

## Key facts established during exploration

Current `run_rl.py` (all line numbers current):
- Advantage = `reward − group_mean` only, no std (`run_rl.py:1980-1982`), inline in `main()`.
- Loss = `-ratio · advantage`, no clip, no KL-in-loss, no entropy (`:2034-2048`).
- `old_logprobs` cached once/step from the live model (`:2000-2009`); no frozen reference.
- Only one model loaded (`:1685`).
- Rollout `max_tokens=1024` hardcoded in 3 generator builders (`:1130, :1207, :1252`);
  train temp=1.0, eval temp=0.0. vLLM `max_num_batched_tokens=4096` (`:1195, :1235`).
- Dataset hardcoded `qwedsacf/competition_math` (`:791`); prompt `boxed.prompt`.
- Reward binary 0/1 via `is_equiv` on last `\boxed{}` (`:1973-1977`).
- `--epochs-per-step` default 1 + mode-default `grad_accum` ⇒ whole rollout consumed
  in ONE optimizer step ⇒ `ratio ≡ 1` ⇒ clip is a no-op unless we take >1 optimizer
  step per rollout (`compute_num_training_steps`, `:1502-1507`).
- Optimizer `adamw` path uses `torch.optim.AdamW` with PyTorch-default betas
  `(0.9,0.999)` — already matches verl; `--weight-decay` (default 0.0) and
  `--lr-scheduler none` (constant, 0 warmup) already reachable. Only SparseAdamW/Muon
  hardcode `(0.9,0.95)` (`:1393, :1413`), not the plain-AdamW path.
- `PeftModel.disable_adapter()` is available ⇒ free frozen reference for adapter modes.

Donor already in-repo — `legacy/run_rl_dapo.py`:
- `compute_policy_loss` with `torch.clamp`+`torch.minimum` clip + clip-fraction
  (`legacy/run_rl_dapo.py:929-952`), and a `--loss-type {grpo,dapo}` selector
  (`:198-216`) with `--clip-ratio-low/high`. Clip donor (but still mean-only adv, no KL).

SimpleRL-Zoo (verl) target — Qwen2.5-7B (all under `ref/simpleRL-reason/`):
- Advantage: `(score − group_mean)/(group_std + 1e-6)`, unbiased std, broadcast over
  tokens; single-sample group ⇒ mean 0/std 1 (`verl/trainer/ppo/core_algos.py:111-154`).
- Loss = `PPO-clip(0.2)` − `0.001·entropy` + `1e-4·low_var_kl(cur‖ref)`, all aggregated
  with **flat masked_mean over response tokens** (`core_algos.py:162-193`).
  - low_var_kl (k3): `kld = clamp(exp(ref−cur) − (ref−cur) − 1, −10, 10)` (`:263-267`).
  - entropy: `logsumexp(logits) − Σ softmax(logits)·logits` (`torch_functional.py:107-111`).
- Reference = frozen bf16 copy of the initial policy, no optimizer (`fsdp_workers.py`).
- Optim: AdamW, lr **5e-7 constant** (0 warmup), wd **1e-2**, betas **(0.9,0.999)**, grad_clip 1.0.
- Data flow: 1024 prompts × 8 rollouts = 8192/step → 32 mini-batch optimizer updates of
  256, single pass. The in-pass drift is what makes clip bite (not ppo_epochs>1).
- max_prompt 1024 / max_response **8192**, temperature 1.0, total_epochs 20.
- Reward: binary 0/1 boxed correctness (matches `is_equiv` closely).
- Dataset: `simplelr_math_35` parquet.

verl's functions **cannot be imported** here (`verl/__init__` → `protocol` → `tensordict`,
which isn't installed; even `torch_functional.py` needs it). Not a blocker since the
verl-comparison test is out of scope; noted so nobody wastes time trying to import it.

## Implementation

All edits in `run_rl.py` unless noted. Every new flag defaults to **current behavior**,
so existing runs are bit-for-bit unchanged; the `simplerl` preset opts into the new recipe.

### A. Extract the inline GRPO math into module-level functions (enables reuse + smoke test)
The advantage/loss math is currently inline in `main()`. Extract:
- `compute_advantages(raw_reward_tensor, norm)` — `mean` (today) or `mean_std`
  (`(r−μ)/(σ+1e-6)`, unbiased std, group-size-1 guard → 0). Replaces `:1980-1982`.
- `compute_policy_loss(...)` — ported from `legacy/run_rl_dapo.py:929-952` but
  **aligned to verl's flat masked_mean** aggregation; returns `(pg_loss, clip_fraction, ppo_kl)`.
- `get_response_logprobs_and_entropy(model, input_ids, labels)` — new helper next to
  `get_response_log_probs` (`:858`); reuses the single forward, also returns per-token
  entropy via the verl formula. Avoids a second forward pass.
- `get_ref_log_probs(model, ...)` — reference logprobs under `torch.inference_mode()` (see D).

### B. Advantage std-norm
- New arg `--advantage-norm {mean,mean_std}` (default `mean`). Preset → `mean_std`.

### C. PPO-clip surrogate + entropy
- New args `--loss-type {grpo_simple,grpo_clip}` (default `grpo_simple`),
  `--clip-ratio-low 0.2`, `--clip-ratio-high 0.2`, `--entropy-coeff 0.0`.
- Slot into the micro-batch loss block (`:2031-2048`): when `grpo_clip`, use the new
  `compute_policy_loss` (flat masked_mean) and subtract `entropy_coeff·masked_mean(entropy)`;
  when `grpo_simple`, keep today's exact per-sequence-mean loss (legacy preserved).

### D. KL-to-reference — ALL modes
- New args `--kl-loss-coef 0.0` (preset `1e-4`), `--kl-penalty-type {low_var_kl,kl}`
  (default `low_var_kl`), `--kl-ref-model {auto,disable_adapter,frozen_copy}` (default `auto`).
- Reference logprobs per mode (`auto` resolves):
  - **Adapter modes** (`lora`, `dora`, `pissa`, `milora`, `randlora`): wrap the reference
    forward in `with model.disable_adapter():` — 0 extra memory.
  - **`full`, `blocktt`, `svd`, `lora_full`**: load a **second frozen bf16 copy** of the
    original model at `:1685` *before* any factorization/conversion, `requires_grad_(False)`,
    `eval()`, not in the optimizer. For `blocktt`/`svd` the pre-conversion base IS the correct
    reference. Memory: ~+14 GB for 7B (weights only, no optimizer/grad state).
- KL term: `kld = clamp(exp(ref_lp − cur_lp) − (ref_lp − cur_lp) − 1, −10, 10)`;
  `loss += kl_loss_coef · masked_mean(kld, mask)`. Add to the `grpo_clip` loss path.
- Log `train/kl_to_ref` alongside the existing `train/approx_kl`.

### E. Multiple optimizer updates per rollout (make clip meaningful)
- No new mechanism required — this is `grad_accum` such that
  `optimizer_steps_per_epoch = micro_batches_per_epoch // grad_accum > 1`
  (`:1502-1507`), single epoch. Add optional convenience `--ppo-mini-batch-size` that
  derives `grad_accum = ppo_mini_batch // micro_batch`. Add a validation **warning** when
  `--loss-type grpo_clip` but `optimizer_steps_per_epoch == 1` (clip would be inert).

### F. Configurable rollout length
- New arg `--rollout-max-tokens 1024` (preset `8192`). Replace the hardcoded `1024` at
  `:1130, :1207, :1252`. Validate `rollout_max_tokens < max_model_len`; raise
  `max_num_batched_tokens` (`:1195, :1235`) to cover prompt+response when large.

### G. Optimizer parity (mostly config)
- Reachable today: `--lr 5e-7`, `--weight-decay 1e-2`, `--lr-scheduler none`, grad-clip 1.0.
- Plain-AdamW betas already `(0.9,0.999)`. Add explicit `--adam-beta1 0.9 --adam-beta2 0.999`
  only for clarity/override; wire into the `adamw` builder (`:1395-1400`).

### H. Selectable SimpleRL data path (keep current alive)
- New arg `--dataset {competition_math,simplelr_math_35}` (default `competition_math`).
- In `load_datasets_and_tokenizer` (`:790-818`), branch:
  - `competition_math` → today's `load_dataset("qwedsacf/competition_math", …)` + `boxed.prompt`.
  - `simplelr_math_35` → load the SimpleRL parquet (train/test) and use the **qwen-boxed**
    prompt template (add `qwen_boxed.prompt` mirroring SimpleRL's eval template). Map the
    parquet's question/answer fields; keep the binary `is_equiv` reward.
- Prompt auto-selected from `--dataset` unless `--prompt-template` explicitly overrides.

### I. `--grpo-variant {default,simplerl}` preset (delivery shape)
- Applied in `apply_mode_defaults` (`:510`) AFTER mode defaults. `simplerl` sets:
  `loss_type=grpo_clip, advantage_norm=mean_std, clip_ratio_low/high=0.2,
  entropy_coeff=0.001, kl_loss_coef=1e-4, kl_penalty_type=low_var_kl, lr=5e-7,
  weight_decay=1e-2, lr_scheduler=none, warmup_ratio=0.0, adam betas (0.9,0.999),
  grad-clip 1.0, rollout_max_tokens=8192, temperature already 1.0, dataset=simplelr_math_35`,
  and derives `grad_accum` for ~32 updates/rollout. Individual flags still override.
- Extend `validate_mode_specific_flags` so the new flags are accepted/validated coherently.

### Files
- **Modify:** `run_rl.py` (args near `:210-311`; `apply_mode_defaults` `:510`; advantage
  `:1980-1983`; new helpers near `:858`; loss loop `:2031-2048`; model load `:1685`;
  rollout `max_tokens` `:1130/:1207/:1252`; dataset loader `:790-818`; optimizer `:1395-1400`).
- **New:** `qwen_boxed.prompt` (SimpleRL prompt template).
- **Donor (read-only):** `legacy/run_rl_dapo.py:929-952`.
- **Launcher:** extend `run_rl_7B.sh` with a `--grpo-variant simplerl` example invocation
  (single H100 will need reduced batch vs verl's 2×8 — document the memory tradeoff and
  the second frozen ref copy for full-FT).

## Verification (both paths, manual eyeball)

### run_rl.py side — automated short smoke (new: `tests/test_run_rl_simplerl_cli.py`)
- Reuse the tiny harness in `tests/test_run_rl_cli.py:9-78` (`_TinyDataset`, `_DummyModel`,
  `_DummyTokenizer`, `_build_fake_generators`); extend `_DummyModel` with a no-op
  `disable_adapter()` context manager and per-token logits so entropy/KL paths execute.
  Add a fake generator yielding a MIX of correct/incorrect boxed answers so group std > 0
  and advantages are non-trivial.
- Patch `AutoModelForCausalLM.from_pretrained`, `maybe_init_wandb`/`wandb.log`, and the vLLM
  generator builders; drive `run_rl.main([...,"--grpo-variant","simplerl","--train-mode",
  "lora","--n-grpo-steps","2", tiny batch ...])`. This is the first test to drive `main()`
  end-to-end.
- **Assert observables** (captured from the patched `wandb.log`): advantage mean ≈ 0;
  with `mean_std`, advantage std ≈ 1 on variance>0 groups; `clip_fraction` finite in [0,1];
  `kl_to_ref` ≥ 0; entropy > 0; loss finite; and `optimizer_steps_per_epoch > 1` (clip path
  actually exercised). Also run a `--train-mode full` variant to exercise the frozen-copy KL path.

### Reference (verl) side — documented one-time manual eyeball (not CI)
- Add `docs/exp_results/rl_new.md` "Manual cross-check" subsection (or a `scripts/`
  README note) describing: in a SEPARATE env with verl deps installed
  (`cd ref/simpleRL-reason && bash install.sh`), run the verl GRPO on a **tiny model**
  (e.g. Qwen2.5-0.5B) for a few steps on 1 GPU with `rollout_n=8`, and eyeball that its
  logged `critic/advantages`, `actor/pg_clipfrac`, `actor/kl_loss` (or `ppo_kl`), and
  entropy are in the **same ballpark** as the run_rl.py `simplerl` run's
  `advantage_*`, `clip_fraction`, `kl_to_ref`, entropy on a comparably-sized run.
- Explicitly document why bitwise equality is NOT expected: different dataset/tokenizer,
  vLLM sampling nondeterminism, FSDP vs single-GPU, and aggregation details. The check is
  "same sign, same rough magnitude, same qualitative trend across steps", done by a human once.

### Also
- `python -m py_compile run_rl.py` and run existing `tests/test_run_rl_cli.py` +
  `tests/test_run_rl_optimizer.py` to confirm no regressions to the default path.

## Out of scope / deferred
- Automated numerical equivalence test against verl formulas (user chose to skip).
- Exact multi-node scale (verl uses 2×8 H100 / 8192 samples-per-step); on a single H100 the
  `simplerl` preset will use a smaller derived batch — a documented scale deviation, not an
  algorithm deviation.
