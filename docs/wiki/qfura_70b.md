# Llama-3-70B QFuRA / QLoRA / QDoRA / QPiSSA: launch and eval playbook

Session-handoff guide for running 4-bit-base PEFT fine-tunes on Llama-3-70B on a
single H100 (94 GB) and evaluating on GSM8K. Developed during the 2026-05-04 to
2026-05-06 sessions. The final headline results live in
`docs/exp_results/qfura.md` → "Llama-3-70B (MetaMathQA-100k) Results"; the
findings are not repeated here. This page documents *how to launch each method
end-to-end* and the gotchas that bit us, so the next session can pick up without
re-discovering them.

---

## 1. What got built

Four 70B fine-tune recipes, all aligned with the QPiSSA paper recipe
(MetaMathQA-100K, 1 epoch, lr=2e-5, eff. batch 128, seq 512, fp32 adapter + fp32
AdamW, PiSSA prompt format, seed=42, gradient_checkpointing). On a 94 GB H100
none of these 70B fits in bf16 (~141 GB), so every recipe needs a custom load
path:

| Method | Shell script | Load path | Save format |
|---|---|---|---|
| QFuRA | `ref/LIFT/bash_scripts/finetune_metamath_qfura_70b.sh` | `--load_strategy=layer_stream`: load CPU bf16, walk each Linear, BTT-decompose + NF4-quantize, move to GPU one at a time | merged fp32 single-file `pytorch_model.bin` (~259 GB) |
| QLoRA | `ref/LIFT/bash_scripts/finetune_metamath_qlora_70b.sh` | NF4 base via `BitsAndBytesConfig` (40 GB on GPU); LoRA adapter on top | PEFT adapter (`last_adapter/`, ~11 GB) — merge step required for eval |
| QDoRA | `ref/LIFT/bash_scripts/finetune_metamath_qdora_70b.sh` | NF4 base + Qdora4bitLinear adapters (fast path) | fully merged HF model in `last/` (~136 GB) — eval can load directly |
| QPiSSA | `ref/PiSSA/scripts/run_qpissa_70b_singlegpu.sh` | Pre-quantized residual snapshot `fxmeng/PiSSA-Llama-3-70B-4bit-r64-5iter` + PEFT LoRA | PEFT adapter (`checkpoint-100/`, ~3 GB) — merge step required for eval |

All four scripts accept env overrides (`lr`, `seed`, `MAX_STEPS`,
`per_device_train_batch_size`, `gradient_accumulation_steps`, `model_max_length`,
`prompt_style`, `trainable_param_dtype`, `optimizer_name`, `OUTPUT`,
`wandb_project`, `wandb_run_name`). Recipe-alignment defaults: `prompt_style=pissa`,
`trainable_param_dtype=fp32`, `optimizer_name=adamw` — these flags were added to
`src/finetune_qfura.py`, `src/finetune_qlora.py`, and `src/finetune_qdora.py` in
this session; older revisions don't have them.

## 2. Launch templates (verbatim, paper-aligned)

### QFuRA — full 1-epoch (long, ~28 h)

```bash
cd /home/yequan/Project/lora/lora-without-regret/ref/LIFT
nohup env CUDA_VISIBLE_DEVICES=7 \
    OUTPUT=/data/yequan/fura/lift/metamath100k/meta-llama/Meta-Llama-3-70B/full-qfura-bs4x32-lr1e-4-fp32-pissaprompt \
    run_name=full-qfura-bs4x32-lr1e-4-fp32-pissaprompt \
    wandb_project=qfura-metamath-Meta-Llama-3-70B \
    lr=1e-4 seed=42 MAX_STEPS=0 num_train_epochs=1 \
    per_device_train_batch_size=4 gradient_accumulation_steps=32 model_max_length=512 \
    prompt_style=pissa trainable_param_dtype=fp32 optimizer_name=adamw \
    bash bash_scripts/finetune_metamath_qfura_70b.sh \
    </dev/null > $OUTPUT/launch.log 2>&1 &
```

### QFuRA — 100-step short (smoke / lr sweep, ~2.5 h)

Identical command, set `MAX_STEPS=100`, `OUTPUT=...short100-qfura-bs4x32-lr<lr>-fp32-pissaprompt`.

### QLoRA / QDoRA — 100-step short

```bash
nohup env CUDA_VISIBLE_DEVICES=<gpu> \
    OUTPUT=/data/yequan/fura/lift/metamath100k/meta-llama/Meta-Llama-3-70B/short100-qlora-r64-bs4x32-lr<lr>-fp32-pissaprompt \
    lr=<lr> seed=42 MAX_STEPS=100 \
    per_device_train_batch_size=4 gradient_accumulation_steps=32 \
    prompt_style=pissa trainable_param_dtype=fp32 optimizer_name=adamw \
    bash bash_scripts/finetune_metamath_qlora_70b.sh \
    </dev/null > $OUTPUT/launch.log 2>&1 &
```

For qdora, swap `qlora` → `qdora` in script name + output dir, add `qdora_impl=fast dora_norm_cache_steps=16`.

### QPiSSA — 100-step short

```bash
cd /home/yequan/Project/lora/lora-without-regret/ref/PiSSA
nohup env CUDA_VISIBLE_DEVICES=<gpu> \
    OUTPUT=/data/yequan/fura/pissa/short100-Llama-3-70B-qpissa-r64-lr2e-5 \
    run_name=short100-Llama-3-70B-qpissa-r64-lr2e-5 \
    wandb_project=qfura-metamath-Meta-Llama-3-70B \
    lr=2e-5 seed=42 MAX_STEPS=100 \
    bash scripts/run_qpissa_70b_singlegpu.sh \
    </dev/null > $OUTPUT/launch.log 2>&1 &
```

QPiSSA uses bs=1×128 per-device by default and pulls the residual from
`fxmeng/PiSSA-Llama-3-70B-4bit-r64-5iter` rather than redoing PiSSA init locally
(the latter OOMs at the bf16 70B load step).

## 3. Step-time and disk budget

| Method | Step time (bs=4×32, fp32 adapter) | Train peak GPU | Save artifact |
|---|---:|---:|---:|
| QFuRA full (1 epoch, 781 steps) | 133 s/step | 71 GB | 259 GB (`pytorch_model.bin`, fp32 trainable preserved) |
| QFuRA short (100 steps) | ~135 s/step | 71 GB | — (we don't save short runs) |
| QLoRA (100 steps) | 77 s/step | 72 GB | 11 GB (adapter only) |
| QDoRA (100 steps) | 105 s/step | 72 GB | 136 GB (fully merged HF model in `last/`) |
| QPiSSA (100 steps, bs=1×128) | 111 s/step | 66 GB | 3 GB (adapter only) |

QDoRA's per-step cost is dominated by the column-norm cache refresh every 16
steps (`--dora_norm_cache_steps 16`). QDoRA's per-step is between QLoRA and QFuRA
because the cache amortizes; setting `dora_norm_cache_steps=1` (PEFT-faithful)
roughly doubles the step time.

Disk budget for a full 4-method sweep at lr=2e-5 + (qfura/qlora/qdora at
lr={5e-5, 1e-4, 2e-4}) is ~3 TB if you keep every `last/` ckpt. We routinely
cleaned up `last_safetensors_bf16/` and `merged_safetensors_bf16/` after eval to
stay under 25 TB total on `/data`.

## 4. GSM8K eval pipeline

vLLM with `--quantization bitsandbytes` on the merged bf16 checkpoint. PiSSA
test_acc requires `"The answer is: X"` suffix; we also use a relaxed extractor at
`/tmp/relaxed_gsm8k_acc.py` (last-numeric-token from response, skipping
in-context regenerations). See "gotcha #7" below.

### QFuRA full ckpt → vLLM

The 259 GB single-file `pytorch_model.bin` loads via bnb-on-the-fly quantization
(bnb reads fp32 weights, quantizes to NF4, writes ~40 GB to GPU). Load takes
~10-15 min, vLLM gen ~10 min.

```bash
CUDA_VISIBLE_DEVICES=7 \
HF_HOME=/data/yequan/huggingface \
TORCHINDUCTOR_CACHE_DIR=/data/yequan/torchinductor \
VLLM_CACHE_ROOT=/data/yequan/vllm_compile_cache \
TRITON_CACHE_DIR=/data/yequan/triton_cache \
TMPDIR=/data/yequan/tmpdir \
uv run --project /home/yequan/Project/lora/lora-without-regret python \
    /home/yequan/Project/lora/lora-without-regret/ref/PiSSA/utils/gen_vllm.py \
    --model "$MODEL_DIR" \
    --data_path /data/yequan/pissa-dataset \
    --sub_task gsm8k_pissa_prompt --dataset_split test \
    --output_file "$OUT/gsm8k_response.jsonl" \
    --batch_size 200 --max_tokens 1024 \
    --temperature 0.0 --top_p 1.0 \
    --quantization bitsandbytes \
    --gpu_memory_utilization 0.92 \
    --dtype bfloat16
```

### QLoRA → merge → vLLM

QLoRA's `last_adapter/` cannot be evaluated directly; vLLM doesn't accept a PEFT
adapter alongside an NF4 base. PEFT's stock `merge_and_unload` on 70B is CPU-bound
and effectively hangs (~10+ hours in our test). We wrote a GPU-side custom merge
at `/tmp/merge_qlora_gpu.py` (see "merge script" subsection below). Wall time
~14 min for a 70B r=64 merge.

```bash
uv run python /tmp/merge_qlora_gpu.py \
    /data/yequan/huggingface/hub/models--meta-llama--Meta-Llama-3-70B/snapshots/<sha> \
    $RUN_DIR/last_adapter \
    $RUN_DIR/merged_safetensors_bf16
```

Then point `gen_vllm.py` at `$RUN_DIR/merged_safetensors_bf16`.

### QDoRA → vLLM directly

QDoRA's "fast" path materializes Qdora4bitLinear into `nn.Linear` on CPU at
end-of-training and writes a fully merged HF model in `<run>/last/`. Point vLLM
at `<run>/last/` — no separate merge.

### QPiSSA → custom merge → vLLM

PEFT's `merge_and_unload` is also too slow on the QPiSSA pre-quantized residual.
We wrote `/tmp/merge_qpissa_gpu7.py` (a near-clone of the qlora merge — load
NF4 residual on GPU, apply adapter, dequant+merge each LoraLayer to CPU bf16,
save sharded). Wall time ~10-12 min. Then eval as for QLoRA.

## 5. Gotchas that bit us this session

### #1 — Adapter dtype: `prepare_model_for_kbit_training` already upcasts to fp32

For QLoRA / QDoRA, PEFT's `prepare_model_for_kbit_training` (called inside the
training script) auto-casts trainable params to fp32. Our explicit
`--trainable_param_dtype fp32` upcast loop runs after, finds 0 params still in
bf16, and prints `upcast 0 qlora params to fp32` — this is **not a bug**. The
params are already fp32. Same goes for the QFuRA log — the loop finds 1120 BTT
trainable params still in bf16 (because qfura's streaming converter doesn't go
through `prepare_model_for_kbit_training`), so the loop has work to do there.

### #2 — Tokenizer pad token bumps vocab from 128256 → 128264

LIFT's `get_tokenizer` adds `'[PAD]'` as a new special token for any Llama
tokenizer. `model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) /
8.0)))` then resizes to a multiple of 8 = 128264. The QLoRA `last_adapter/` was
trained on this resized model; PEFT's `from_pretrained` recreates the base at
128256 vocab → `size mismatch for embed_tokens` crash on adapter load.

Fix (already applied in `/tmp/merge_qlora_gpu.py`): mirror the resize before
loading the adapter:

```python
from transformers import AutoTokenizer
import math
tok = AutoTokenizer.from_pretrained(SNAP)
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": "[PAD]"})
target_vocab = int(8 * math.ceil(len(tok) / 8.0))
m.resize_token_embeddings(target_vocab)
m = PeftModel.from_pretrained(m, ADAPTER)
```

QPiSSA does NOT add `[PAD]` (sets `pad_token = eos_token`), so its merge script
doesn't need this — and the QPiSSA-trained adapter's vocab is 128256.

### #3 — QFuRA save uses CPU offload for the merged bf16; QDoRA the same

`materialize_btt_to_linear` / `materialize_qdora_to_linear` are called with
`target_device="cpu"` at save time. Without that, the per-layer merged bf16
weights accumulate on GPU and OOM around module 160 of 560 on a 94 GB H100.
The CPU spill is mandatory for 70B; for 8B it's a no-op cost (~few seconds).

This was added during the session — make sure `finetune_qfura.py` and
`finetune_qdora.py` both pass `target_device="cpu"` to their materialize call.

### #4 — `save_pretrained` chokes on `torch.dtype` in `model.config`

After materialize, the model's `config.torch_dtype` (and sometimes
`config._pre_quantization_dtype`) are still `torch.dtype` objects. `save_pretrained`
calls `to_json_string`, which fails with `TypeError: Object of type dtype is not
JSON serializable`. The qdora_fast helper `materialize_qdora_to_linear` already
strips these; the custom merge scripts (`/tmp/merge_qlora_gpu.py`,
`/tmp/merge_qpissa_gpu7.py`) walk every config attr at the end and replace any
`torch.dtype` with its string form.

### #5 — Supervisor loops must wait for `model.safetensors.index.json`, not `config.json`

In `/tmp/eval_lr2e5.sh` we initially had:

```bash
until [ -f "$QDORA/last/config.json" ]; do sleep 30; done
```

`save_pretrained` writes `config.json` first (~immediately) but the 31 model
shards take ~5 min and the tokenizer files come at the very end. The supervisor
fired vLLM the moment `config.json` appeared, vLLM tried to load partial shards,
fell back to the slow tokenizer, and crashed in 12 s with `TypeError: not a
string`.

Fix: wait for the last-written file. For qdora `last/`:

```bash
until [ -f "$QDORA/last/model.safetensors.index.json" ] && [ -f "$QDORA/last/tokenizer.json" ]; do sleep 30; done
```

### #6 — Disk `/` fills with vLLM / torch.compile caches

70B model load through bnb on-the-fly quantization writes large compile artifacts
to `~/.cache/vllm`, `/tmp/torchinductor_yequan`, and `~/.cache/triton`. Root
partition is only 1.8 TB and shared across users. We hit "No space left on
device" once and had to clean 14 GB of stale caches mid-run.

Redirect all compile caches to `/data` for every eval:

```bash
TORCHINDUCTOR_CACHE_DIR=/data/yequan/torchinductor \
VLLM_CACHE_ROOT=/data/yequan/vllm_compile_cache \
TRITON_CACHE_DIR=/data/yequan/triton_cache \
TMPDIR=/data/yequan/tmpdir
```

Also worth deleting between runs: `/tmp/fura_*` directories, old
`~/.cache/vllm/torch_compile_cache/*` (compile cache is per-model and grows
unbounded).

### #7 — QLoRA / QDoRA strict-acc = 0% at 100 steps is *format-fitting*, not math

`utils/test_acc.py` only counts a sample correct if `"The answer is: X"` appears
in the response. At 100 steps, qfura/QPiSSA learn the format (their decomposition
init has the model reproducing base behavior at step 0, leaving room for the
small format-fitting update); qlora/qdora don't (random LoRA delta is zero at
step 0, and 100 steps isn't enough to fight off the base's free-form output
habit). Their actual GSM8K math accuracy is ~65% — visible via the relaxed
extractor at `/tmp/relaxed_gsm8k_acc.py`. Don't conclude "qlora/qdora can't do
math" from strict 0%; conclude "100 steps isn't enough to fit the format".

### #8 — GPU contention with shared users

Memory headroom is tight. QDoRA at 70B peaks around 47 GB, but a co-located user
with 50+ GB on the same GPU will OOM you at model load. We had one OOM on GPU 3
when another user pushed memory up to 54 GB. Check `nvidia-smi` before launch
and prefer GPUs <20 GB used; if forced to share, lower
`--gpu_memory_utilization` to 0.70 for eval (training shells don't have a flag
for this — just move to a free GPU).

### #9 — `WANDB_RUN_ID` leaks between sequential runs

If you reuse a shell to launch multiple training runs in sequence (e.g. the
`/tmp/short_qlora_qdora.sh` supervisor), make sure `wandb_run_id` is regenerated
per run, otherwise wandb logs every step to the first run's URL. The shell
scripts call `wandb.util.generate_id()` at the top — invoke them fresh, don't
`source` them.

## 6. Cleanup checklist

After a full sweep, before handing off:

- Delete `<run>/merged_safetensors_bf16/` (qlora) after eval — 132 GB each.
- Delete `<run>/last/` for qdora short runs (136 GB each) — but **keep** the full
  qfura `last/` (259 GB) since that's the headline checkpoint.
- Keep `<run>/last_adapter/` (qlora, ~11 GB) and `<run>/checkpoint-100/` (qpissa,
  ~3 GB) — they're small and let us re-eval later.
- Keep `<run>/eval_gsm8k/gsm8k_response.jsonl` — needed to recompute relaxed-acc.

## 7. Operational summary

Per-run wall clock budget on a single H100:

| Stage | QFuRA full | QLoRA short | QDoRA short | QPiSSA short |
|---|---:|---:|---:|---:|
| Model load | 3 min | 3 min | 3 min | 3 min |
| 100 training steps | — | 2 h 10 m | 2 h 55 m | 1 h 51 m |
| 781 training steps (1 epoch) | 28 h | — | — | — |
| Save | 30 min | <1 min | 13 min | <1 min |
| Merge (CPU bf16, sharded) | (saved as fp32 dense) | 14 min | (saved as merged) | 11 min |
| vLLM eval (GSM8K 1319) | 22 min | 12 min | 12 min | 12 min |
| **Total** | **~29 h** | **~2 h 36 m** | **~3 h 21 m** | **~2 h 14 m** |

A 4-method short-sweep (one lr per method) takes ~11 GPU-hours sequentially or
~3.5 wall hours if run on 4 separate GPUs in parallel.

## 8. Pointers

- Headline results: `docs/exp_results/qfura.md`, section "Llama-3-70B
  (MetaMathQA-100k) Results".
- 70B shell scripts: `ref/LIFT/bash_scripts/finetune_metamath_{qlora,qdora,qfura}_70b.sh`,
  `ref/PiSSA/scripts/run_qpissa_70b_singlegpu.sh`.
- Custom merge scripts (not in repo, kept under `/tmp/` between sessions; copy
  to a stable location if you plan to re-use):
  `/tmp/merge_qlora_gpu.py`, `/tmp/merge_qpissa_gpu7.py`.
- Relaxed GSM8K extractor: `/tmp/relaxed_gsm8k_acc.py`.
- wandb projects:
  `qfura-metamath-Meta-Llama-3-70B`, `qlora-metamath-Meta-Llama-3-70B`,
  `qdora-metamath-Meta-Llama-3-70B`.
- Base model HF snapshot:
  `/data/yequan/huggingface/hub/models--meta-llama--Meta-Llama-3-70B/snapshots/<sha>`.
- QPiSSA NF4 residual snapshot:
  `/data/yequan/huggingface/hub/models--fxmeng--PiSSA-Llama-3-70B-4bit-r64-5iter/snapshots/<sha>`.
