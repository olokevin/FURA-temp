# FuRA Initialization and End-to-End Training Cost

_Response material for the reviewer question: "Can the authors report end-to-end training cost, including block-SVD initialization, for short runs, larger models, and distributed settings? It would also be useful to clarify whether the decomposition can be cached and reused across tasks."_

All numbers below are FuRA / qfura only. End-to-end wall-clock comes from the existing runs; the block-SVD initialization cost was re-measured per model with `tools/bench_fura_init.py` under each run's own configuration (`--decomp_mode output_one_block --s_merged_to keep_trainable --train_position small`, 7 target modules = `q,k,v,o,gate,up,down`), because the training logs record the conversion but not its wall time.

## 1. Measured initialization cost

`tools/bench_fura_init.py`, one H100 NVL (95.8 GB), bf16, cold process.

| Model | Path | Modules converted | Load (s) | Block-SVD convert (s) | Residual→GPU (s) | **Init total (s)** | Peak GPU during init |
| ---------------- | -------------- | ----------------: | -------: | --------------------: | ---------------: | -----------------: | -------------------: |
| Qwen3-1.7B       | FuRA (dense)   |               196 |     3.52 |                 20.95 |                — |          **24.47** |             6.08 GiB |
| Qwen2.5-7B       | FuRA (dense)   |               196 |    11.40 |                 78.87 |                — |          **90.27** |            27.52 GiB |
| Meta-Llama-3-8B  | FuRA (dense)   |               224 |     4.58 |                 79.14 |                — |          **83.71** |            28.95 GiB |
| Meta-Llama-3-70B | qfura (stream) |               560 |     0.91 |                743.56 |             8.57 |         **753.04** |            34.89 GiB |

Notes:

- The 70B row uses the layer-streaming BTT+NF4 path (`convert_and_quantize_linear_to_qbtt_streaming`) that `finetune_qfura.py` selects for ≥30B models. Its `load` figure is not meaningful on its own — `low_cpu_mem_usage=True` mmaps the shards, so the disk read is paged in lazily during conversion; read `load + convert = 744.5 s` as one number.
- Initialization scales with total factorized weight volume, not layer count: 1.7B → 7B is 3.8× the convert time on the same 196 modules.
- Peak GPU during init stays bounded: 34.9 GiB for 70B, because the streaming converter factorizes and NF4-quantizes one Linear at a time.

## 2. End-to-end training cost

Training wall-clock comes from each run's `sys_metrics.json` (`total_wall_s`, the training loop) or W&B `_runtime` (RL, which starts before model load and therefore already includes initialization).

| Setting | Model | Steps | Init | Training | **End-to-end** | Init share |
| --------------------------------- | --------------- | --------: | -------: | ---------: | -------------: | ---------: |
| SFT, Commonsense-170K, 1 epoch     | Llama-3-8B      |    10,645 |   83.7 s |     2.89 h |     **2.92 h** |  **0.80 %** |
| SFT, MetaMath-100K, 1 epoch, qfura | Llama-3-70B     |       781 |    753 s |    20.19 h |    **20.66 h** |  **1.01 %** |
| RLVR (GRPO)                        | Qwen3-1.7B      |        54 |   24.5 s |          — |     **0.93 h** |  **0.73 %** |
| RLVR (GRPO)                        | Qwen2.5-7B      |        53 |   90.3 s |          — |     **1.60 h** |  **1.57 %** |

Run provenance:

- 8B SFT — `lift/commonsense/meta-llama/Meta-Llama-3-8B/blocktt-calib_none-lr_2e-4-decomp_output_one_block_pos_small_smerge_keep_trainable-1ep-seed_43`; `total_wall_s = 10,412 s`, 224 modules converted.
- 70B SFT — `lift/metamath100k/meta-llama/Meta-Llama-3-70B/qfura-r_full-decomp_output_one_block-smerge_keep_trainable-bs8x16-lr_2e-5-seed_42`; `total_wall_s = 72,673 s`, 560 modules. End-to-end is directly observed, not inferred: the launcher stamps `Sun May 3 09:43:50 PM PDT 2026` and `training.log` closes at `2026-05-04 18:23:33`, i.e. **20.66 h**. The 1,710 s gap against the training loop covers model load, streaming conversion (753 s measured), dataset tokenization and the final checkpoint save.
- RL 1.7B — `rl_runs/Qwen3-1.7B/blocktt/blocktt-adamw-lr_1e-4-output_one_block-s_to_keep_trainable-train_small-0419-185333` (0.93 h / 54 steps; sibling runs 0.88–0.93 h).
- RL 7B — `rl_runs/Qwen2.5-7B/blocktt/blocktt-adamw-lr_1e-4-output_one_block-s_to_keep_trainable-train_small-seed_43-0505-151820` (1.60 h / 53 steps; seeds 43/44 and the mbs1 variant span 1.56–1.60 h).

**Short runs.** Initialization is a fixed cost, so its share is set by run length, not by model size. For Llama-3-8B SFT at 0.978 s/optimizer-step (measured, 1-epoch run), the 83.7 s init is ≈22 % of a 300-step debug run, ≈7.9 % of a 1,000-step run, ≈2.8 % of a 3,000-step run, and 0.80 % of the full epoch. This is the honest worst case for the method: FuRA's initialization is only visible in runs short enough that no fine-tuning conclusion could be drawn from them anyway.

**Larger models and distributed settings.** The 70B result is single-GPU: the streaming converter caps peak GPU memory at 34.9 GiB, so no model or tensor parallelism is needed to initialize a 70B FuRA/qfura model on one 94 GB H100, and 753 s of initialization amortizes to 1 % of a 20-hour epoch. We have not measured multi-GPU initialization — every run reported here is `--num_processes 1`. Structurally, conversion is an independent per-`nn.Linear` SVD with no cross-layer dependency, so under FSDP or tensor parallelism each rank factorizes only its own shard and the wall time should fall roughly linearly with rank count; we report this as an expectation, not a measurement.

## 3. Can the decomposition be cached and reused across tasks?

Yes. In the default recipe (`init_mode=default`, no calibration) the factorization is a pure function of the frozen pretrained weight matrix: no training data, task, learning rate, or random seed enters `convert_linear_to_btt`. Verified directly — factorizing Qwen3-1.7B twice under different global RNG seeds gives **588/588 BTT tensors bitwise identical** (1.45 B parameters), so the same cores are recomputed identically on every launch across every task, LR and seed in this paper.

Consequences:

- The conversion is a one-time, offline, task-independent preprocessing step. It could be run once per base model and shipped as an ordinary checkpoint; loading it back is a normal `state_dict` load (≈4.6 s for a Llama-3-8B-sized checkpoint from page cache, versus 79.1 s to re-factorize). The current code recomputes it at every launch — caching is a checkpoint save/load away, not an algorithmic change.
- Within RL, factorization already happens exactly once at setup. The per-rollout path calls `materialize_dense_weight()` (a product of the existing cores) to hand dense weights to vLLM, and never re-factorizes.
- Caveat: the optional **calibrated** variant (`calib_v2*`, `apply_calibrated_btt`) does consume a calibration batch and is therefore data-dependent. It is not part of the default recipe and none of the results above use it; a cache built for the default path is not valid for the calibrated one.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=6 uv run tools/bench_fura_init.py \
  --base_model meta-llama/Meta-Llama-3-8B --methods fura \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --decomp_mode output_one_block --s_merged_to keep_trainable --train_position small \
  --out /tmp/init_8b.json

# 70B uses the streaming BTT+NF4 path
CUDA_VISIBLE_DEVICES=6 uv run tools/bench_fura_init.py \
  --base_model meta-llama/Meta-Llama-3-70B --methods qfura_stream ... --out /tmp/init_70b.json
```
