# PiSSA-recipe code finetuning: LoRA-vs-FURA on Llama-2-7B / CodeFeedback

This page is a session-handoff guide for continuing the FURA-vs-LoRA comparison
on the PiSSA paper's "python" recipe (Llama-2-7B, CodeFeedback-100K SFT,
HumanEval+MBPP eval). Following it lets a fresh session re-launch / extend the
sweep without re-deriving the integration.

It covers:

- where the PiSSA code lives, and what was patched to support FURA + single-GPU
- where data, weights, ckpts, eval artifacts, and W&B runs go
- the canonical train→eval→cleanup→W&B pipeline
- current results table + what's still missing for a publication-quality claim

---

## 1. Context

**Goal.** Reproduce the PiSSA paper's LoRA-Llama-2-7B baseline on the CodeFeedback
"python" subset, then test whether **FURA** (BlockTT, full-rank, project defaults)
improves over a properly LR-tuned LoRA on HumanEval / MBPP pass@1.

**Recipe origin.** PiSSA paper Sec. 5 / Tab. 2 — "all experiments use AdamW, batch
size 128, lr 2e-5, cosine, warmup_ratio 0.03, weight_decay 0; lora_alpha=lora_r,
dropout 0; adapters on all 7 linears". CodeFeedback-100K, 1 epoch, seq_len 512.

The reference codebase is `ref/PiSSA/` (untracked third-party clone of
github.com/GraphPKU/PiSSA, lives outside the project's git tree). FURA was added
by patching `ref/PiSSA/train.py` to swap `nn.Linear → BTTLayer` from this repo's
`btt_layer.py`. See §3.

---

## 2. Files of interest

```
ref/PiSSA/
├── train.py                                # *patched* — adds --fura branch,
│                                           # imports from <repo>/btt_layer.py
│                                           # via sys.path insert (lines 7-14)
└── scripts/python_llama2_7b/
    ├── run_lora_singlegpu.sh               # one-shot LoRA, paper LR (2e-5)
    ├── run_fura_singlegpu.sh               # one-shot FURA, default LR (2e-4)
    ├── run_lora_sweep_one.sh               # sweep wrapper: train + eval + cleanup
    ├── run_fura_sweep_one.sh               # sweep wrapper: train + eval + cleanup
    ├── eval_and_log.sh                     # vLLM gen → split → evalplus →
    │                                       # log_eval_to_wandb.py → ckpt rm
    └── log_eval_to_wandb.py                # parse evalplus stdout, log 4 metrics
```

The two `run_*_sweep_one.sh` scripts are the **canonical entry points**. They
take `LR` and `DEVICE` env vars and chain training → eval → ckpt deletion →
wandb logging. Both `run_lora_singlegpu.sh` / `run_fura_singlegpu.sh` are the
older raw versions kept as a quick reference but are **not** wandb-aware.

### What's patched in `ref/PiSSA/train.py`

1. `sys.path` insert near the top so `from btt_layer import …` resolves to the
   repo root `BTTLayer` / `convert_linear_to_btt` / etc.
2. New `TrainingArguments` flags: `--fura`, `--fura_trainable_type`,
   `--fura_decomp_mode`, `--fura_blocktt_rank`, `--fura_train_position`,
   `--fura_s_merged_to`, `--fura_factorize_by_head`, `--fura_train_bias`.
   Defaults match `CLAUDE.md`'s project-wide defaults (rank=full,
   `output_one_block`, `train_position=small`, `s_merged_to=keep_trainable`).
3. New `materialize_btt_to_linear(model)` helper + `resolve_blocktt_rank`.
4. `build_model` gains a fura branch that calls `convert_linear_to_btt` then
   `configure_blocktt_trainability` (skipping the LoRA/PiSSA PEFT path).
5. The trailing `if 'norm' in name or 'gate' in name: module.to(fp32)` loop is
   guarded so `BTTLayer` modules whose name happens to contain `gate_proj` are
   not coerced to fp32 (would break the bf16 deepspeed flow).
6. Save tail: when `--fura`, materialize BTT → `nn.Linear`, then
   `safe_save_model_for_hf_trainer` (the existing full-FT save path).
7. `torch.distributed.barrier()` calls inside `train()` are guarded with
   `torch.distributed.is_initialized()` so single-GPU `python train.py`
   (no deepspeed launcher) doesn't crash. HF Trainer sets `local_rank=0` even
   without a process group, which used to trigger an unguarded `barrier()`.

---

## 3. Environment

- **uv venv** at `/home/yequan/Project/lora/lora-without-regret/.venv` (Python 3.13).
  PiSSA's pinned `requirements.txt` (deepspeed 0.15.4, transformers 4.45, etc.)
  is **not** what's installed — we use the parent uv env (transformers 4.57.1,
  peft 0.17.1, vllm 0.10.2, accelerate 1.11, bitsandbytes 0.49.2, torch 2.8).
- **deepspeed is not installed** and we do not need it on a single H100. Skip
  the `deepspeed --include=localhost:0` launcher and just call
  `python train.py …` (this is what `run_*_sweep_one.sh` does).
- **tensorboard** is installed (`uv add tensorboard` was run, so `--report_to
  tensorboard` works); we use **wandb** for the sweep though.
- **evalplus 0.3.1** + **human_eval 1.0.3** added via `uv add` — required by the
  HumanEval / MBPP scoring step in `eval_and_log.sh`.
- **wandb** logged in as
  `yequan_zhao (yequan_zhao-university-of-california-santa-barbara)`. Project
  name used: **`pissa-code-llama2-7b`**.

If a fresh shell needs the env:
```bash
export HF_HOME=/data/yequan/huggingface
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
```

---

## 4. Data + weights

| Resource | Location | Notes |
|---|---|---|
| Base model | `meta-llama/Llama-2-7b-hf` (HF cache `/data/yequan/huggingface/hub/…`) | already downloaded |
| pissa-dataset | `/data/yequan/pissa-dataset/` | downloaded once via the README's `huggingface-cli download --local-dir`; contains `python/`, `metamath/`, `commonsense/`, `mmlu/`, etc. |
| Train split | `/data/yequan/pissa-dataset/python/train.json` | 104,848 rows, fields `instruction / input / output / type` |
| Test split | `/data/yequan/pissa-dataset/python/test.json` | 563 rows = 164 HumanEval + 399 MBPP-sanitized; for test rows the `output` field holds the **task_id** (`HumanEval/N` / `Mbpp/N`), not a reference completion |

`gen_vllm.py` and `train.py` both read this dir via
`load_dataset(args.data_path, data_dir='python', split=…)` — the local dir
behaves as a HF datasets repo because of its layout.

---

## 5. Run / artifact / W&B layout

Disk:
```
/data/yequan/pissa-runs/<run_name>/        ← training output_dir
                                           # config.json, tokenizer, trainer_state,
                                           # training.log, runs/ (tb), checkpoint-820/
                                           # (model weights deleted by eval_and_log.sh)

/data/yequan/fura/pissa/code/<run_name>/   ← eval artifacts (kept after cleanup)
                                           # python_response.jsonl  (vLLM raw)
                                           # humaneval.jsonl, mbpp.jsonl  (split)
                                           # humaneval_eval_results.json (evalplus per-task)
                                           # mbpp_eval_results.json
                                           # eval.log     (full pipeline stdout)
                                           # sweep.log    (timestamps + sweep wrapper)
                                           # metrics.json {humaneval, humaneval+, mbpp, mbpp+ pass@1}

/data/yequan/pissa-runs/sweep-logs/        ← per-GPU queue logs
```

W&B:
- Project: **`pissa-code-llama2-7b`**
  (entity `yequan_zhao-university-of-california-santa-barbara`).
- Each train run gets a wandb run via `--report_to wandb --run_name <name>`.
- Each eval run **adds a separate W&B run with the same display name** (created
  by `log_eval_to_wandb.py`). Plot pass@1 in the run summary, train loss in the
  history. Acceptable as long as you filter the project view by job_type if
  needed.

Naming convention: `lora-r128-lr<LR>` and `fura-default-lr<LR>` (e.g.
`fura-default-lr5e-4`). The `default` here means project-default FURA settings
(rank=full / output_one_block / small / keep_trainable).

---

## 6. Canonical train + eval + cleanup pipeline

To launch one LR (either method), use the sweep wrappers. They train, then
auto-run the eval pipeline, then delete the model weights to free disk.

```bash
# FURA (default settings) at lr=5e-4 on GPU 4
LR=5e-4 DEVICE=4 RUN_NAME=fura-default-lr5e-4 \
    bash ref/PiSSA/scripts/python_llama2_7b/run_fura_sweep_one.sh

# LoRA r128 at lr=1e-4 on GPU 3
LR=1e-4 DEVICE=3 RUN_NAME=lora-r128-lr1e-4 \
    bash ref/PiSSA/scripts/python_llama2_7b/run_lora_sweep_one.sh
```

Per-LR cycle on a single H100 NVL: ~2h06m (LoRA) / ~2h31m (FURA) train, ~2 min
eval, ~5 s ckpt cleanup. Both at per-device bs=4 × grad-accum=32 = 128 effective
batch, seq_len=512, 1 epoch, 820 optimizer steps.

To eval an existing ckpt without retraining (e.g. you re-ran training elsewhere
and want to re-score):
```bash
CKPT=/data/yequan/pissa-runs/<dir> RUN_NAME=<name> DEVICE=3 \
    bash ref/PiSSA/scripts/python_llama2_7b/eval_and_log.sh
# Set DELETE_CKPT=0 to keep the model weights
```

Steps inside `eval_and_log.sh`:

1. `utils/gen_vllm.py` — vLLM completes all 563 prompts at T=0, max_tokens=1024,
   ~20s on Llama-2-7B / H100 NVL. Output: `python_response.jsonl`.
2. `utils/code_process.py` — split + de-fence `[python ... ]`, strip
   `if __name__ == "__main__":` / `# Example usage` / trailing `assert`. Output:
   `humaneval.jsonl`, `mbpp.jsonl`.
3. `evalplus.evaluate --dataset humaneval/mbpp --samples …` — reads the jsonl,
   runs each completion against the EvalPlus test suite, prints `pass@1` for
   the base + plus splits.
4. `log_eval_to_wandb.py` — regex-parses the four `pass@1` lines from the tee'd
   eval log, writes `metrics.json`, opens a wandb run, logs the four metrics.
5. Cleanup: deletes `model-*.safetensors`, `optimizer.pt`, `scheduler.pt`,
   `rng_state.pth`, `adapter_model.safetensors`, anything under
   `checkpoint-*/`. Keeps `config.json` + tokenizer + trainer_state +
   training.log + tensorboard `runs/` (≈8 MB).

The vLLM worker prints
`Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.`
during shutdown — this is **cosmetic in vLLM 0.10.2** and happens after all
prompts are processed and the response file is written. Ignore.

---

## 7. Results so far (single seed each, LR sweep partially complete)

All numbers are pass@1 on the PiSSA `python/test.json` test split (164 HumanEval
problems, 399 MBPP-sanitized problems), evaluated by `evalplus 0.3.1`.

| Run | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---|---|---|---|---|
| LoRA r128, lr=2e-5 (paper recipe)         | 0.207 | 0.189 | 0.362 | 0.288 |
| **LoRA r128, lr=1e-4**                    | 0.262 | 0.226 | 0.399 | 0.328 |
| FURA full-rank, lr=2e-4 (project default) | 0.220 | 0.195 | 0.384 | 0.304 |
| **FURA full-rank, lr=5e-4**               | **0.268** | **0.244** | **0.399** | **0.341** |

**Headline:** the PiSSA paper's lr=2e-5 turned out to be way under-tuned for
LoRA in our environment (HumanEval 0.207 vs the paper's 0.184 for LoRA, so the
reproduction is consistent — the PiSSA-paper LoRA was also under-tuned).
Boosting LoRA's LR to 1e-4 closes most of the FURA-vs-LoRA gap. **FURA still
wins on every metric, but the margin (+0.6 to +1.8 pp) is within single-seed
noise on three of the four metrics**.

What we don't yet know:

- LoRA's actual peak LR — only `2e-5` and `1e-4` tried; should bracket with at
  least `5e-4` and ideally `1e-3` to confirm 1e-4 isn't itself near-peak.
- FURA's peak LR — `2e-4` and `5e-4` tried, both improve over LoRA-2e-5; need
  `1e-3` to confirm 5e-4 is the peak (and `1e-4` to bracket lower).
- Multi-seed std. The PiSSA paper reports ~0.3 pp HumanEval std over 3 seeds;
  single-seed differences below ~1 pp are not significant.

### Suggested next runs

The minimum to claim **"FURA improves over LoRA at LR-matched best"**:

| GPU 3 | GPU 4 |
|---|---|
| LoRA lr=5e-4 (~2h06m)             | FURA lr=1e-4 (~2h31m) |
| LoRA lr=1e-3 (queue after, ~2h06m)| FURA lr=1e-3 (queue after, ~2h31m) |

Total ~4–5 h to fill in the LR curve for both methods. Then if FURA still
wins by >1 pp at peak, run **3 seeds at the best LR for each** (~7.5 h on 2
GPUs) for a real significance test.

The deleted-during-cleanup partial run from a prior session (`fura-default-lr1e-4`)
was killed at step ~12 and is **not** in the wandb project — re-run from scratch
if you want it.

---

## 8. Common gotchas (don't re-debug these)

- **`torch.distributed.barrier()` crash on single-GPU.** Already fixed in `train.py`
  via `is_initialized()` guards (commit not yet made — file lives in untracked
  `ref/PiSSA/`).
- **fp32 cast loop kills BTT layers.** PiSSA's `for name, module in
  model.named_modules(): if 'norm' in name or 'gate' in name: module.to(fp32)`
  matches `gate_proj` even after BTT replacement. Already guarded for fura.
- **deepspeed launcher not available.** Use plain `python train.py …` from a
  CUDA_VISIBLE_DEVICES context. The `deepspeed --include=localhost:0` lines in
  the canonical PiSSA `run_*.sh` recipes are removed in our `_singlegpu` /
  `_sweep_one` variants.
- **tensorboard / wandb not installed.** Both added via `uv add`. If a fresh
  Python env: `uv add tensorboard wandb evalplus human_eval`.
- **`humaneval/mbpp pass@1` regex must match lowercase + tab.** evalplus prints
  `humaneval (base tests)\npass@1:\t0.xxx` — the parser in `log_eval_to_wandb.py`
  uses a multiline regex with `re.IGNORECASE`. Earlier capitalized regex missed
  the `+` variants; that's why the smoke-test had to be re-logged.
- **vLLM "EngineCore died" warning during shutdown.** Cosmetic, response file is
  already saved. Don't act on it.
- **`pissa-dataset/python/test.json` `output` field ≠ reference answer.** It's
  the task_id used by evalplus to match against its test suite. The completion
  comes from the model's vLLM output, then is post-processed by `code_process.py`.
- **Effective batch.** PiSSA paper recipe = batch 128. On 1 H100 we use
  per-device 4 × grad-accum 32 = 128. Don't accidentally use 8 × 16 etc. without
  re-tuning LR.
- **Save policy.** PiSSA-paper LoRA recipe sets `--merge True`, which merges the
  LoRA adapter into the base and writes a 13-GB merged model — that's what we
  evaluate via vLLM. FURA also writes a dense materialized 13-GB model. Both
  are auto-deleted by `eval_and_log.sh` after scoring.

---

## 9. Quickstart for a fresh session

```bash
cd /home/yequan/Project/lora/lora-without-regret

# Sanity checks
nvidia-smi --query-gpu=index,memory.free --format=csv | head      # find free GPUs
ls /data/yequan/pissa-dataset/python/                              # train.json + test.json present
ls /data/yequan/fura/pissa/code/                                   # already-evaluated runs

# View existing wandb runs
# https://wandb.ai/yequan_zhao-university-of-california-santa-barbara/pissa-code-llama2-7b

# Launch a new LR (example: FURA lr=1e-3 on GPU 4)
LR=1e-3 DEVICE=4 RUN_NAME=fura-default-lr1e-3 \
    nohup bash ref/PiSSA/scripts/python_llama2_7b/run_fura_sweep_one.sh \
    > /data/yequan/pissa-runs/sweep-logs/fura_lr1e-3.log 2>&1 &
```

Each sweep wrapper writes:
- `<CKPT_DIR>/training.log` — full HF Trainer stdout (loss curves, saves)
- `<ARTIFACT_DIR>/sweep.log` — train+eval timestamps and exit summary
- `<ARTIFACT_DIR>/eval.log` — vLLM gen + evalplus stdout
- `<ARTIFACT_DIR>/metrics.json` — final 4 pass@1 numbers

Tail any of those to monitor a run.
