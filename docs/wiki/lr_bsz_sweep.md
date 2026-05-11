# Math-10K LR + batch-size sweep: protocol, results, gotchas

This page is a session-handoff guide for re-running or extending the LR / batch-size
sweep on LLaMA-3-8B SFT on Math-10K, evaluated on GSM8K. The sweep was developed
across 2026-04-30 to 2026-05-06 and informed the appendix of the FuRA NeurIPS 2026
submission (`docs/26_nips_fura_paper/neurips_2026.tex`,
`\subsection{LLaMA-3-8B Math-10K SFT: LR and batch-size sweep}` under
`\section{Extended experimental results}`).

If you are starting a fresh sweep on a different model / dataset, the operational
playbook in §5 is the part that transfers; the headline numbers in §3 do not.

---

## 1. Scope

Three methods, all on `meta-llama/Meta-Llama-3-8B`, 3 epochs on
`LLM-Adapters/ft-training_set/math_10k.json` (10K examples, seed 43, max sequence
length 2048, bf16, linear LR decay, warmup ratio 0.03):

- **Full FT** — `ref/LIFT/bash_scripts/finetune_math_full.sh`.
- **LoRA** — `ref/LIFT/bash_scripts/finetune_math_lora.sh`, `r=64, α=128`,
  target_modules `q,k,v,o,gate,up,down` (the math-script default, which is
  **7-module** — distinct from the commonsense LoRA script's 5-module default).
- **FuRA (BlockTT)** — `ref/LIFT/bash_scripts/finetune_math_blocktt.sh`,
  `decomp_mode=output_one_block, train_position=small, blocktt_rank=full,
  s_merged_to=keep_trainable, no calibration`.

Eval: GSM8K-only via `EVAL_DATASETS="gsm8k"` (a session-added env override on
`eval_math.sh` / `eval_math_lora.sh`; default eval list is the full 7-task math
suite). One pass takes ~12 min vs ~75 min for the full 7-task suite.

Two sweep axes:

- **LR sweep**: fixed effective batch size $=16$ (script defaults), vary LR
  per-method.
- **Batch-size sweep**: $\{16, 64, 256\}$, with LR picked best per
  (method, bsz). At fixed bsz the best LR is determined by an explicit
  short LR scan around the $\sqrt{\text{bsz}/16}$-rescaled prediction.

---

## 2. Sweep coverage actually run

Final swept-LR sets (after a few interactive course-corrections — see §4):

| Method  | LR set at bsz $=16$ |
|---------|---|
| Full FT | $\{8{\times}10^{-6},\, 1{\times}10^{-5},\, 2{\times}10^{-5},\, 3{\times}10^{-5}\}$ (an earlier $5{\times}10^{-5}$ point collapsed to 53.6 GSM8K and is excluded from figures) |
| LoRA    | $\{3, 6, 8, 10, 20\}{\times}10^{-5}$ (paper plots use $\{3,6,10,20\}{\times}10^{-5}$) |
| FuRA    | $\{8, 10\}{\times}10^{-5}$, $\{2, 3, 4, 6, 8\}{\times}10^{-4}$ (paper plots use $\{2,3,4,6\}{\times}10^{-4}$) |

Batch-size sweep (best LR per cell):

| bsz | Full FT | LoRA | FuRA |
|-----|---|---|---|
| 16  | $\eta=1{\times}10^{-5}$ | $\eta=6{\times}10^{-5}$ | $\eta=3{\times}10^{-4}$ |
| 64  | $\eta=2{\times}10^{-5}$ | $\eta=2{\times}10^{-4}$ | $\eta=6{\times}10^{-4}$ |
| 256 | $\eta=2{\times}10^{-5}$ | $\eta=6{\times}10^{-4}$ | $\eta=8{\times}10^{-4}$ |

---

## 3. Headline GSM8K results (single seed)

Source-of-truth tables live in `docs/exp_results/lift_math.md`. Highlights:

- Best overall: **Full FT, bsz $=64$, $\eta=2{\times}10^{-5}$ → 72.56**.
- Best PEFT: **FuRA, bsz $=64$, $\eta=6{\times}10^{-4}$ → 72.18**, a hair behind
  Full FT.
- LoRA caps around 70.8 across batch sizes (flat plateau).
- Each method has a different favourable LR range (Full FT $\sim 10^{-5}$,
  LoRA $\sim 10^{-4}$, FuRA $\sim 10^{-4}$–$10^{-3}$).
- At bsz $=256$, every method drops; Full FT drops the most, LoRA the least.
  See the discussion paragraph at the bottom of `lift_math.md` for the
  qualitative summary.

Figures in `docs/exp_results/figs/`:
- `math_lr_sweep_bsz16.{png,pdf}` — 3-curve LR sweep, GSM8K only, with the
  best-LR-per-method labelled in larger text on each peak.
- `math_bsz_sweep.{png,pdf}` — 3-curve batch-size sweep, GSM8K only.

Both also live in `docs/26_nips_fura_paper/figs/` (PDFs only) for the paper.

To regenerate after editing data: `uv run python analysis/plot_math_sweeps.py`.
The data points are hardcoded in lists at the top of that script (`LR_SWEEP_*`,
`BSZ_SWEEP_*`) — update them and re-run; the PNGs and PDFs both rewrite.

---

## 4. Script-state notes (env-overrides added during this session)

The math scripts originally hardcoded almost everything. The following env
overrides were added on `ref/LIFT/bash_scripts/finetune_math_{full,lora,blocktt}.sh`
during this sweep and **are not committed yet** (`git status` will show them as
modified). If a future session pulls a fresh branch, re-apply these patches
or this entire sweep recipe will break:

- `OUTPUT="${OUTPUT:-...}"` instead of `OUTPUT=...` on line ~35-52 of each
  finetune script. Without this, exporting `OUTPUT` from the launcher gets
  silently clobbered.
- `--per_device_train_batch_size ${PER_DEVICE_TRAIN_BS:-N}` and
  `--gradient_accumulation_steps ${GRAD_ACC_STEPS:-N}` (defaults: full=2/8,
  lora=1/16, blocktt=2/8).
- `--num_train_epochs ${num_train_epochs:-3}` on blocktt + lora.
- `${GRAD_CKPT:+--gradient_checkpointing}` on `finetune_math_full.sh` — full FT
  defaults to no checkpointing; set `GRAD_CKPT=1` if you push `per_device > 2`
  (see §5 OOM section).

Eval-side patches (also uncommitted):

- `eval_math_lora.sh` got the same dual-save resolution (`last/` → `best/` → flat)
  that `eval_math.sh` already had. Without this, eval crashes when training saved
  the model into a `last/` subfolder.
- Both `eval_math*.sh` accept `EVAL_DATASETS` (space-separated) to override the
  default 7-task list. Used as `EVAL_DATASETS="gsm8k"` throughout this sweep.
- `ref/LIFT/src/eval/run_math_parallel.py` now logs `eval/<dataset>/accuracy`
  and `eval/<dataset>/total` to wandb (`wandb.init(resume="must")`) and accepts
  `--wandb_project / --wandb_run_name / --wandb_run_id / --no_wandb`. Pre-fix,
  math evals never reached wandb — they only landed in `eval.log` on disk. If a
  prior math run shows training metrics in wandb but no eval, that's why.

---

## 5. Operational playbook (re-runnable recipe)

### 5a. Launch one config

```bash
nohup bash -c '
set -e
export CUDA_VISIBLE_DEVICES=N                                # one GPU
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export HF_HOME=${HF_HOME:-/data/yequan/huggingface/cache}
export EVAL_DATASETS="gsm8k"                                 # speed up auto-eval
export MODEL=meta-llama/Meta-Llama-3-8B
export lr=2e-4                                               # per-run knob

# Sweep-specific (only override if non-default):
# export PER_DEVICE_TRAIN_BS=...
# export GRAD_ACC_STEPS=...

# For Full FT, optionally turn on grad ckpt if per_device > 2 (avoids OOM):
# export GRAD_CKPT=1

# Unique output dir per (method, lr, bsz). Required because the script's
# default OUTPUT path embeds only lr — bsz>16 runs collide otherwise.
# export OUTPUT="/data/yequan/fura/lift/math/$MODEL/lora-lr_${lr}-rank_64-alpha_128-bsz64-seed_43"
# export run_name="$(basename $OUTPUT)"

adapter_name=lora lora_r=64 lora_alpha=128 \
    bash ref/LIFT/bash_scripts/finetune_math_lora.sh
' > /data/yequan/fura/lift/launch_logs/gpuN_<short_tag>.log 2>&1 &
echo "PID: $!"
disown
```

Swap the bash script for `finetune_math_full.sh` / `finetune_math_blocktt.sh`
as needed. For blocktt also export `decomp_mode=output_one_block` and
`s_merged_to=keep_trainable` (the project defaults — included for clarity).

### 5b. Chain runs

For sequential runs on one GPU (recommended over launching many in parallel
on shared GPUs), wrap a `for lr_val in ...; do export lr=$lr_val; bash …; done`
inside the `nohup` block. The session has worked examples under
`/data/yequan/fura/lift/launch_logs/`:

- `gpu5_math_full_lr_sweep_default.log` — full FT bsz=16 chain (3 LRs).
- `gpu3_math_lr_sweep.log` — LR sweep on GPU 3.
- `gpu2_math_lora_blocktt_full_sweep.log` — 6-run mixed-method chain
  (lora + blocktt + full).
- `gpu3_blocktt_lr_then_bsz_sweep.log` — adaptive 2-stage chain that picks
  best LR after stage 1 then launches scaled-LR bsz runs (script:
  `tools/run_gpu3_blocktt_lr_then_bsz_sweep.sh` — keep as a template).

### 5c. Sentinel sanity checks after launch

```bash
LOG=/data/yequan/fura/lift/launch_logs/gpuN_<tag>.log
# Wait for first training step or OOM:
until grep -qE "Step: [0-9]+|out of memory|Total optimization|OutOfMemoryError|Traceback" $LOG; do sleep 10; done
grep -E "Step: [0-9]+|out of memory|Total optimization|Traceback" $LOG | head -3
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -i N
```

Total opt steps should match the bsz: 1842 (bsz=16) / 462 (bsz=64) / 117 (bsz=256)
for math_10k × 3 epochs. If the count is wrong the bsz override didn't take.

Warmup-LR sanity check at step 10: `LR_observed ≈ lr × 10 / (0.03 × total_steps)`.
If observed LR is 10× off, the script picked up the wrong `lr` env var.

### 5d. Harvest results

```bash
M_BASE=/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B
for run in <list_of_runs>; do
    echo "--- $run ---"
    grep "Step:" "$M_BASE/$run/training.log" | tail -1     # final loss
    for log in "$M_BASE/$run/last/math/gsm8k/eval.log" "$M_BASE/$run/math/gsm8k/eval.log"; do
        [ -f "$log" ] && grep -oE 'Result [0-9]+(\.[0-9]+)?' "$log" | tail -1 && break
    done
done
```

The dual path is necessary because some runs save flat (parent dir, no `last/`)
and some use the `last/` subfolder; see §6.

---

## 6. Hard-won lessons / gotchas

### 6.1. Stale `pytorch_model.bin` in the parent dir poisons eval

The eval scripts auto-resolve `${CKPT}` with the priority:

```
${CKPT}/config.json (flat layout)  →  ${CKPT}/last/config.json  →  ${CKPT}/best/config.json
```

If you re-train into a dir that previously held a flat-layout checkpoint, the
new model lands at `last/pytorch_model.bin` but the old `config.json` /
`pytorch_model.bin` at the parent are not deleted by the training script. Eval
then resolves to the **parent** (priority 1) and silently evaluates the **old**
model. The result is a "completed" eval that has the wrong number.

**How to detect**: compare timestamps of `<run>/training.log` vs
`<run>/math/<task>/eval.log`. If the eval log predates training, the eval used
a stale checkpoint.

**Fix**: before re-running into a reused dir name, `rm <run>/config.json
<run>/pytorch_model.bin` (or move the whole dir aside as a `.bak` suffix), or
pass `CKPT="<run>/last"` directly to the eval script to bypass resolution.

### 6.2. Math finetune scripts hardcoded `OUTPUT=…` until 2026-05-05

Before the patch (§4), `export OUTPUT=…-bsz64` from the launcher had no effect.
Multi-bsz sweeps that reused the same lr value collided on disk. The bsz=64
LoRA result was first reported as a stale Apr-14 number because of this; only
after re-eval against the actual `last/` did it land at 70.36. Always check
that `OUTPUT="${OUTPUT:-…}"` is in the script before assuming the launcher's
`OUTPUT=` propagated.

### 6.3. Full FT bsz tuning is tight on a 95 GB H100 NVL

Full LLaMA-3-8B fits with `per_device_bsz=2, accum=8` at ~83 GB; this is the
script default and is the **safest** setting. Two failed attempts on per-device
larger:

- `per_device=8, accum=2` → OOM at step 1 (87 GB allocated, 1.8 GB request
  overflows 95 GB cap).
- `per_device=4, accum=4` → trained for ~70 steps then OOM on a long-sequence
  batch (91 GB / 95 GB headroom, killed by a length-spike).

With `GRAD_CKPT=1`, `per_device=4, accum=4` works but loses the throughput win
the larger per-device was meant to deliver. Net: **leave Full FT at
`per_device=2, accum=8` unless you really need the speedup and have GRAD_CKPT
on**. The other methods (LoRA, FuRA) have plenty of headroom at any per-device
size you'd actually want.

### 6.4. `set -e` plus a flaky training step ⇒ chain abort, possibly mid-eval

The chain scripts use `set -e` so that one failing config doesn't silently
continue. The downside: when a training process crashes, the auto-eval block
in the bash script still fires once (eval scripts return 0 even on the
"no usable checkpoint" path), but the *next* chain stage doesn't get to.
A SIGKILL'd training also kicks off a useless eval against missing weights —
the eval generates a few empty `eval.log` files and then crashes; if you
care about a clean re-run, delete the partial run dir before relaunching.

### 6.5. `${LOAD_LAST_MODEL:-…}` does NOT do what you think for the empty case

In the lora finetune script the flag was made env-overridable as
`${LOAD_LAST_MODEL:-"--load_last_model"}`. The `:-` operator falls back to the
default when the variable is **unset or empty**. So `export LOAD_LAST_MODEL=""`
to "drop the flag" actually leaves the flag in place. To override, use
`${LOAD_LAST_MODEL-"--load_last_model"}` (single dash, falls back only when
unset), or pass a non-empty placeholder for `LOAD_LAST_MODEL`. We hit this once
and it cost an eval phase.

### 6.6. Save layouts differ across methods

Recent training writes the model to `${OUTPUT}/last/`. Some prior runs from
April 2026 wrote flat at `${OUTPUT}/`. The eval scripts' auto-resolution
(§6.1) handles both, but the **harvest scripts** must check both paths. The
canonical harvest snippet is in §5d.

---

## 7. Picking LRs for an extension sweep

Practical recipe that worked on this model/dataset:

1. **Find the small-batch (bsz=16) sweet spot per method.** Method-specific
   ballparks on LLaMA-3-8B:

   - Full FT: $\sim 10^{-5}$. Plateau is narrow. Anything $\geq 3{\times}10^{-5}$
     drops 6+ pts on GSM8K because the model overfits the Math-10K training
     distribution (final loss drops below 0.1 while GSM8K accuracy collapses).
   - LoRA: $\sim (3$–$10){\times}10^{-5}$. Plateau is very flat.
   - FuRA: $\sim (2$–$4){\times}10^{-4}$ (broader than Full FT, narrower than
     LoRA). Unimodal — falls off symmetrically.

2. **For each new batch size, start with $\sqrt{\text{bsz}/16}$ rescaling.**
   It worked decently for bsz=64 across all three methods. For bsz=256 it
   over-shoots for FuRA and Full FT but is reasonable for LoRA.

3. **Always sweep at least 2 LRs per (method, bsz)** even when you trust the
   rescaling. Single-config single-seed results swing by ~1 pt; without a
   neighbour you can't tell signal from noise.

4. **Use GSM8K-only eval during the sweep.** ~12 min vs ~75 min for the full
   suite. Only re-eval the winner on the full suite if needed for a paper.

5. **Watch the training loss-vs-GSM8K signal for Full FT.** Lowest training
   loss is *not* highest GSM8K — there's a sharp inflection around `loss ≈ 0.15`
   below which generalization breaks. For LoRA / FuRA the same effect exists
   but the cliff is gentler.

---

## 8. File pointers

Source-of-truth result tables:
- `docs/exp_results/lift_math.md` — full sweep + per-method best tables + the
  rebalanced 3-method LR / bsz figures and the discussion paragraph.

Paper integration:
- `docs/26_nips_fura_paper/neurips_2026.tex`:
  - `\section{Hyperparameters}` → `Table~\ref{tab:fura-hparams-math10k}`
    (Full FT / LoRA / FuRA columns, shared schedule + per-method LR set).
  - `\section{Extended experimental results}` →
    `\subsection{LLaMA-3-8B Math-10K SFT: LR and batch-size sweep}` →
    `Figure~\ref{fig:math10k_sweep}` (two PDFs side by side).

Plot regeneration:
- `analysis/plot_math_sweeps.py` (PNG + PDF; data is hardcoded near the top).

Launchers / chain scripts:
- `tools/run_gpu3_blocktt_lr_then_bsz_sweep.sh` — adaptive 2-stage chain that
  picks best LR after stage 1 then scales for stage 2. Worth keeping as a
  template for future blocktt sweeps.

Logs:
- `/data/yequan/fura/lift/launch_logs/gpu*_math_*.log` — all chain logs
  preserved.

---

## 9. RL sweep lessons — Qwen2.5-7B / Qwen3-1.7B / R1-Distill-Qwen-7B (2026-05-01…05-06)

Companion to the SFT sweep above. This section captures what we learned about
**RL (GRPO) LR sweeps and seed sweeps** across `run_rl.py`-driven experiments on
three 7B-class models plus Qwen3-1.7B. The infrastructure differences vs SFT are
substantial — RL has rollouts, vLLM-in-process, GRPO step budget, and
checkpoint-merging quirks — so the operational lessons mostly do **not** transfer
from §1–8 verbatim.

If you are starting a fresh RL LR sweep on a new model, §9.5 (operational playbook)
is the transferable part; the headline numbers in §9.4 (LR ranges) are
model-specific.

### 9.1. Scope

Three Q-family models, all `qwedsacf/competition_math`, 50 GRPO steps,
boxed-prompt rollouts via `run_rl.py`:

- **Qwen3-1.7B** (smaller, runs at run_rl.py defaults: mbs=2 gacc=128 gpu_util=0.4
  max_model_len=2048) — used for 6-method Paper Table sweep.
- **Qwen2.5-7B** (base) — required the 7B mem recipe; full FT required new
  `adamw8bit + gradient checkpointing` plumbing.
- **DeepSeek-R1-Distill-Qwen-7B** (reasoning model) — required
  `max_model_len=4096` (training) and `--math-verify-max-tokens 8192` (eval) to
  avoid truncating chain-of-thought before the `\boxed{}` answer.

Eval: math-verify on `MATH-500 / AMC23 / AIME-24 / AIME-25` (Minerva deprecated
2026-05-02 for noisy run-to-run variance). AIME at avg@8 T=0.6, others greedy@1
T=0. Full eval suite runs in ~3 min after training; MATH-500-only in <1 min.

### 9.2. Memory recipes for 7B-class RL

Single H100 NVL (93 GB usable). `run_rl.py` defaults
(`mbs=2 gacc=128 gpu_util=0.4 max_model_len=2048`) **OOM at step 1** on 7B
models for several method × seed combinations. The "always-safe" 7B recipe is:

| Flag | Value | Saves vs default |
|---|---|---|
| `--micro-batch-size` | `1` (vs default 2) | ~5–8 GB activations |
| `--gradient-accumulation-steps` | `256` (vs default 128) | (preserves effective batch) |
| `--gpu-memory-utilization` | `0.25` (vs default 0.4) | ~14 GB vLLM KV-cache |
| `--max-model-len` | `2048` (vs default 2048) | (unchanged for non-reasoning models) |

This is encoded in `run_rl_7B.sh` (created this session) as `MEM_FLAGS`. The
recipe leaves ~25 GB of headroom on 7B base models; works for `lora / dora /
blocktt / full` on Qwen2.5-7B.

**Method-specific exceptions:**

- **Qwen2.5-7B LoRA** fits at the run_rl.py *default* recipe — but barely. One
  Paper-Table seed sweep (seed=44) OOM'd at step 2 with the default recipe while
  seed=42/43 fit fine. **Run-to-run vLLM allocator nondeterminism** can push
  peak memory ±2 GB across seeds — when peak is already at 91 GB, that's
  enough to OOM. Lesson: for any 7B model, default to the safe recipe unless
  GPU pressure is below 70 GB.
- **R1-Distill needs `max_model_len=4096`** for training (otherwise reasoning
  rollouts are truncated mid-thought and the GRPO reward is uniformly low) and
  `--math-verify-max-tokens 8192` for eval. Default 2048 was sufficient for
  non-reasoning models.
- **DoRA peak memory > LoRA peak** by ~5–8 GB at the same rank (DoRA materializes
  extra per-layer base-output tensors). DoRA always needs the safe recipe on 7B.
- **BlockTT peak memory > LoRA peak** by ~3–5 GB (frozen large core stays on
  GPU). BlockTT needs the safe recipe on 7B.

### 9.3. Full FT on 7B with single H100 — `--optimizer adamw8bit --gradient-checkpointing`

Naive full FT of Qwen2.5-7B blows out a single H100 NVL: ~15 GB weights + 15 GB
bf16 grads + 60 GB fp32 AdamW state + ~10 GB activations + ~23 GB vLLM ≈ 120 GB
peak. Two opt-in flags added to `run_rl.py` this session bring it down to ~67
GB peak:

- **`--optimizer adamw8bit`** — `bitsandbytes.optim.AdamW8bit`, drops optimizer
  state from ~60 GB (fp32 m+v) to ~15 GB. Saves ~45 GB. No measurable accuracy
  hit per the bitsandbytes paper.
- **`--gradient-checkpointing`** — calls
  `model.gradient_checkpointing_enable(use_reentrant=False)` on the base model
  (with `enable_input_require_grads()` for PEFT compat). Saves ~5–10 GB
  activations at ~30% step-time cost (~91 s/step at 7B vs ~70 s without GC).

Both flags default **off**; only enable for the specific full-FT-7B-single-GPU
path. Other train modes keep the prior behavior.

### 9.4. LR ranges by model × method (50-step GRPO horizon)

What we learned from the LR sweeps in this session. **All ranges are for 50
GRPO steps; longer horizons may shift them.**

| Model | Method | Stable LR | Best LR | Failure mode |
|---|---|---|---|---|
| Qwen3-1.7B | FuRA `s_to=frozen` | 1e-5 — 1e-4 | **1e-4** | (no collapse observed) |
| Qwen3-1.7B | RandLoRA | ~1e-4 | **1e-4** | (no collapse observed) |
| Qwen2.5-7B | Full FT | ≤1e-5 | **1e-5** | 3e-5 partial / 5e-5 catastrophic |
| Qwen2.5-7B | LoRA | 1e-4 (only LR tried) | **1e-4** | (no collapse at 1e-4) |
| Qwen2.5-7B | DoRA | 1e-4 (only LR tried) | **1e-4** | seed-dependent late collapse (1/3 seeds) |
| Qwen2.5-7B | FuRA `s_to=keep_trainable` | 1e-4 (only LR tried) | **1e-4** | (no collapse at 1e-4) |
| R1-Distill-Qwen-7B | LoRA | 1e-4 | **1e-4** | (no collapse) |
| R1-Distill-Qwen-7B | FuRA `s_to=keep_trainable` | 8e-5 — 1e-4 | **1e-4** | (no collapse) |
| R1-Distill-Qwen-7B | FuRA `s_to=trainable` | 1e-4 | **1e-4** (best of any FuRA config tested on R1-Distill) | (no collapse) |

**Generalizations to remember when picking LRs for a new model:**

1. **PEFT methods tolerate ~10× the LR of Full FT** at the same model size.
   Qwen2.5-7B Full FT collapses at 5e-5 but LoRA/DoRA/FuRA all survived at
   1e-4. Reason: rank-restricted updates can't move parameters as far per
   step, so the optimizer naturally "self-regulates" into a stable region.
2. **Larger base model ⇒ narrower stable LR window for full FT.** Qwen3-1.7B
   full FT was stable at 2e-5; Qwen2.5-7B full FT is stable only at ≤1e-5.
3. **`s_merged_to` choice within BlockTT changes optimal LR by ~2×.** On
   R1-Distill at lr=1e-4 fixed: `s_to=keep_trainable` mean-4 33.96, `s_to=trainable`
   mean-4 37.33 (+3.4 pp). Always sweep `s_merged_to` alongside LR on a new
   model.
4. **Reasoning models have different LR sensitivity from base models** —
   R1-Distill's stable LR for BlockTT looked similar to Qwen2.5-7B base (1e-4),
   but the rollout-length sensitivity is dramatically higher (`max_model_len`
   bumped 2048→4096 changes everything).
5. **Linear LR schedule at 50 GRPO steps is a net negative.** R1-Distill
   FuRA at lr=1e-4 fixed → mean-4 33.96; same setting + linear schedule with
   warmup_ratio=0.03 → mean-4 31.96 (−2 pp). Reason: linear decay from peak
   to 0 over 50 steps with warmup=2 means the average effective LR is
   ~`lr/2` — i.e. the scheduled run trained at half the integral LR of the
   fixed run. Linear schedules pay off only when step count is large enough
   that the warmup-fraction shrinks; for 50-step RL, **use fixed LR**.

### 9.5. Operational playbook (sweep recipe)

What we converged on as the right protocol:

**LR sweep with cheap eval, then re-eval winner.** For a fresh model:

1. Pick 4 candidate LRs spanning an order of magnitude (e.g., 8e-6 / 1e-5 /
   3e-5 / 5e-5 for full FT on 7B; 8e-5 / 1e-4 / 2e-4 / 4e-4 for PEFT).
2. Run them sequentially on **one GPU** as a chain script (see
   `logs/chain_gpu1_qwen2_5_7B_full_lrsweep.sh` for the canonical template),
   each with `CFG_SUFFIX="--math-verify-datasets MATH-500"` to restrict
   eval to MATH-500-only (cuts ~2 min off each run).
3. Pick the winner by MATH-500 (the most stable single signal in the suite).
4. Re-eval the winner offline on the remaining datasets via `eval_rl.py
   --checkpoint <path>/step=50 --math-verify-datasets "AMC23,AIME-24,AIME-25"
   --output-json <path>/step=50/eval_results_extended.json`. Use a separate
   output filename to preserve the training-time `eval_results.json`.

**Seed sweep after LR is picked.** For headline rows that will appear in a
paper:

1. Re-run the winning config with `--seed 43 --seed 44` (the existing run is
   typically seed=42, the `run_rl.py` default).
2. Use the same memory recipe for all seeds — if one seed OOMs because of
   vLLM allocator nondeterminism, **relaunch only that seed with the safer
   recipe** (mbs=1 gacc=256 gpu_util=0.25) rather than re-running everything.
   The effective batch is preserved, so the comparison is still valid.
3. Aggregate via mean ± sample std (n=3). Report `mean-4 = unweighted mean
   across MATH-500/AMC23/AIME-24/AIME-25` per (model, method).
4. **Watch for discrete failure modes** (catastrophic late-stage collapse).
   1 collapsed seed out of 3 inflates std by ~10×; mean is also pulled by
   ~10 pp. Report n=2 trimmed mean if a clear collapse occurs, with the
   collapse rate documented separately (see §9.6 lesson on DoRA).

**Chain script structure** (see `logs/chain_gpu*_*_sweep.sh` for examples):

```bash
#!/usr/bin/env bash
set -u
cd /home/yequan/Project/lora/lora-without-regret

COMMON_ENV=(
  TRAIN_MODE=blocktt DEVICE=3
  MODEL_ID=<model_id>
  WANDB_PROJECT=<project>
  CFG_SUFFIX="--math-verify-datasets MATH-500"  # cheap eval during sweep
)
run() {
  echo "=== [$(date)] starting $* ==="
  env "${COMMON_ENV[@]}" "$@" bash run_rl_7B.sh
  echo "=== [$(date)] rc=$? ==="
}

run LR=8e-6
run LR=1e-5
run LR=3e-5
run LR=5e-5
echo "=== [$(date)] all done ==="
```

Run as `nohup bash logs/chain.sh > logs/chain.out 2>&1 &`. Use the printed
`=== starting ===` / `=== rc= ===` markers for grep-based result extraction
(see one-liner in §9.7).

### 9.6. Hard-won lessons / gotchas

1. **Default `run_rl.py` recipe OOMs on 7B for blocktt/dora and is fragile for
   lora.** Always use the safe 7B recipe via `run_rl_7B.sh` or `MEM_FLAGS`. The
   "lora barely fits" case bit us when one seed=44 OOM'd that seed=42 and seed=43
   hadn't. **Default to the safe recipe for any 7B run.**

2. **`run_rl.py` only saves `step=50/eval_results.json` from the training-time
   math-verify hook.** When extending eval offline, write to a different
   filename (e.g., `eval_results_extended.json`) — `eval_rl.py`'s default
   `--output-json` overwrites the existing file.

3. **In-process vLLM falls back to local rollout when no `vllm serve` HTTP
   server is running** (`resolve_lora_rollout_backend` probes :8000 and falls
   back). For 7B work we used in-process everywhere — no separate `vllm serve`
   needed. The probe is cheap (2s timeout); not worth gating.

4. **The `run_rl_7B.sh` `MEM_FLAGS` array hardcodes `--max-model-len 2048`. For
   R1-Distill (reasoning) you must override via `CFG_SUFFIX="--max-model-len
   4096 --math-verify-max-tokens 8192"`** — argparse last-wins so the
   `CFG_SUFFIX` `--max-model-len 4096` overrides the `MEM_FLAGS` 2048. Don't
   try to edit `MEM_FLAGS` per-run; use `CFG_SUFFIX`.

5. **PEFT wrapper intercepts `gradient_checkpointing_enable`.** When wiring
   gradient checkpointing into `run_rl.py`, reach `model.base_model` if it
   exists (PEFT case) rather than calling on the wrapper directly. Also call
   `enable_input_require_grads()` for PEFT compat. See
   `run_rl.py:if args.gradient_checkpointing` block (added 2026-05-05).

6. **AMC23 is noisy** (40-problem set, 1 problem = 2.5 pp). Many "method A
   beats B by 5 pp on AMC23" claims earlier in `docs/exp_results/rl.md` are
   within seed noise. AIME-25 is the second-noisiest. MATH-500 (500 problems)
   is the most stable single signal — **use MATH-500 for LR selection**, not
   AMC23 or AIME-25.

7. **Seed sweep on DoRA / 7B revealed a binary collapse failure mode.**
   Qwen2.5-7B DoRA seed=43 trained healthily through step 42 (train acc 69.5%)
   then crashed to 9.8% over steps 43–45 — all four math-verify scores landed
   near zero. 1 of 3 seeds. **DoRA at 7B needs ≥5 seeds before reporting a
   headline**; mean ± std with n=3 is misleading because the variance comes
   from a discrete event.

8. **`--lr-scheduler linear --warmup-ratio 0.03` is *worse* than fixed LR at
   50 GRPO steps** (R1-Distill BlockTT lr=1e-4: 33.96 fixed vs 31.96
   scheduled). The LIFT linear-decay recipe was designed for ~600-step SFT
   epochs, not short RL. For 50-step RL: **use fixed LR**. If you must use a
   schedule, compensate by setting initial LR ~2× higher so the integral
   matches the fixed-lr equivalent.

9. **`run_rl_7B.sh` env var pattern: `CFG_SUFFIX` appends, last-wins.** Use
   `CFG_SUFFIX="--flag1 v1 --flag2 v2"` to add or override any flag. Use
   `NAME_SUFFIX="-seed_43"` to disambiguate run names. `LR_SCHEDULER` and
   `WARMUP_RATIO` were added 2026-05-04 as native env vars (default off →
   fixed LR).

10. **Files survive better than wandb runs.** All eval JSONs are written to
    `<run_dir>/step=50/eval_results.json` (or `eval_results_extended.json`
    for offline re-evals). When summarizing many runs, parse the JSONs
    directly — don't rely on the wandb summary, which can lose runs to
    network hiccups.

### 9.7. Result extraction one-liner

For chain logs in the structure used this session, the per-run scores can be
pulled with:

```python
import re
log = open('logs/chain_gpu3_seed_sweep.out').read()
chunks = re.split(r'=== \[[^\]]+\] starting (.+?) ===', log)[1:]
for i in range(0, len(chunks), 2):
    name = chunks[i].strip()
    body = chunks[i+1]
    res = {ds: float(m.group(1)) for ds in ('MATH-500','AIME-24','AIME-25','AMC23')
           for m in [re.search(rf'{ds}: ([\d.]+)%', body)] if m}
    print(name, res)
```

This works because the chain scripts emit `=== [date] starting <label> ===`
and the math-verify block always ends with `MATH-500: x.xx% / AIME-24: ... /
AIME-25: ... / AMC23: ...` lines.

### 9.8. File pointers

Results doc: `docs/exp_results/rl.md` (Paper Table, BlockTT ablation, Seed
Sweep section, Paper Table Methods Seed Sweep section with raw per-seed,
averaged, and mean±std tables).

Launchers added this session:
- `run_rl_7B.sh` — env-var-driven 7B launcher with `MEM_FLAGS`, `SCHED_FLAGS`
  for LR_SCHEDULER / WARMUP_RATIO / MIN_LR_RATIO, and dispatcher accepting
  `TRAIN_MODE=full|lora|dora|blocktt`.
- `logs/chain_gpu*_*_sweep.sh` — chain scripts for: LR sweep
  (chain_gpu1_qwen2_5_7B_full_lrsweep.sh), seed sweep
  (chain_gpu{2,3}_*_seed_sweep.sh), R1-Distill blocktt sweep
  (chain_gpu{4,5}_r1distill_sweep.sh).

Code changes to `run_rl.py` (2026-05-05):
- `--optimizer adamw8bit` choice (default `adamw`).
- `--gradient-checkpointing` BooleanOptionalAction (default off).
- `gradient_checkpointing_enable(use_reentrant=False)` + `enable_input_require_grads()`
  wired after `validate_trainable_params`, reaching `model.base_model` if
  present.

Run dirs reorganized to `/data/yequan/fura/rl_runs/<Model-Name>/<method>/`
(formerly mixed by method only). Qwen2.5-7B / Qwen3-1.7B / DeepSeek-R1-Distill-Qwen-7B
all have their own subdirs as of this session.

Chain logs: `logs/chain_gpu*_*.out` and per-run logs at
`logs/{r1distill_7B_,qwen2_5_7B_}*.out`.
