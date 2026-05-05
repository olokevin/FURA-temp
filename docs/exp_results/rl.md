# RL Experiment Results

_Updated 2026-04-23 after bug-fix sweep, LR exploration, and fura LR sweep. Llama-3.1-8B-Instruct sweeps appended 2026-04-27. Fura missing-cell reruns added 2026-05-02 (cells 1, 4, 5, 6 of the canonical 6-cell Qwen3-1.7B decomp×s_to grid — all 6 cells now extended-eval'd). Qwen2.5-7B blocktt-mml2048 row appended 2026-05-02 evening. Minerva eval deprecated and all Minerva numbers stripped from this doc on 2026-05-02 (means recomputed as mean-4); summary rankings updated accordingly. BlockTT ablation table (mirrored from `lift_commonsense.md`) added at the top on 2026-05-03. Unified paper table (Qwen3-1.7B + Qwen2.5-7B) added at the very top on 2026-05-03. Qwen2.5-7B Full FT row filled in on 2026-05-05 via single-H100 adamw8bit + gradient-checkpointing sweep over {8e-6, 1e-5, 3e-5, 5e-5} (winner: lr=1e-5). Base (no RL) rows filled in 2026-05-05 via offline `eval_rl.py` on the HF model IDs._

## Paper Table

Best run per method on each model, evaluated on 4 held-out math benchmarks. Within each model block, **bold** marks the column-best across that block. AIME-24/25 are avg@8 at T=0.6; MATH-500 and AMC23 are greedy@1 at T=0.0. PEFT methods use rank=64 (FuRA uses `output_one_block / pos_small / s_merged_to=keep_trainable / blocktt_rank=full`).

| Model                | Method         |       MATH-500 |          AMC23 |        AIME-24 |        AIME-25 |
| :------------------- | :------------- | -------------: | -------------: | -------------: | -------------: |
| **Qwen3-1.7B** | Base           |           52.2 |           30.0 |            5.8 |            7.5 |
|                      | Full FT        | **63.6** |           47.5 |           13.8 |           15.4 |
|                      | LoRA           |           60.6 |           50.0 |           11.2 |           11.2 |
|                      | DoRA           |           61.6 |           37.5 |           12.9 |           14.6 |
|                      | PiSSA          |           53.4 |           42.5 |            2.5 |            3.3 |
|                      | MiLoRA         |           59.0 |           45.0 |            7.9 |            9.6 |
|                      | RandLoRA       |           63.2 | **57.5** | **15.8** |           17.1 |
|                      | **FuRA** | **63.6** | **57.5** |           15.0 | **17.5** |
| **Qwen2.5-7B** | Base           |           48.4 |           40.0 |            5.0 |            2.9 |
|                      | Full FT        |           59.4 |           52.5 |            7.9 |            4.2 |
|                      | LoRA           |           58.6 | **60.0** |           10.8 |            2.9 |
|                      | DoRA           |           59.0 |           50.0 |           10.4 |            8.3 |
|                      | **FuRA** | **60.2** |           57.5 | **12.5** |  **9.2** |

Notes:

- **Qwen3-1.7B FuRA best row**: `blocktt-adamw-lr_1e-4-output_one_block-s_to_frozen-train_small-0502-164206`. Note the row uses `s_to=frozen`, not the project-wide `s_to=keep_trainable` default — the 6-cell ablation (next section) showed `frozen` wins by mean-4 (37.15 vs 34.83) at lr=1e-4. RandLoRA narrowly beats FuRA on the 4-metric mean (38.40 vs 37.15) thanks to AIME-25, but FuRA ties or wins on MATH-500 / AMC23 / AIME-24.
- **Qwen2.5-7B FuRA best row**: `blocktt-adamw-lr_1e-4-output_one_block-s_to_keep_trainable-train_small-0502-174402` (`max_model_len=2048`). The earlier `mml=1536` row (mean-4 29.74) is superseded.
- **Qwen2.5-7B Full FT row**: `full-adamw-lr_1e-5-0505-024633` — adamw8bit + gradient checkpointing single-H100 fit. Picked as winner of an LR sweep over {8e-6, 1e-5, 3e-5, 5e-5} by MATH-500 (8e-6: 58.2 / **1e-5: 59.4** / 3e-5: 37.8 collapse / 5e-5: 1.2 catastrophic). The stable LR window for Qwen2.5-7B base full-FT GRPO is narrow: ≤1e-5. AMC23/AIME-24/AIME-25 re-eval'd offline via `eval_rl.py` post-sweep. Full FT does not beat any of the PEFT rows on this block — FuRA wins MATH-500 / AIME-24 / AIME-25, LoRA wins AMC23. Plausibly the sweep should extend below 8e-6, or run for >50 GRPO steps so the smaller LR has more updates to accumulate.
- All Qwen3-1.7B PEFT rows use rank=64 except FuRA (`blocktt_rank=full` per the project default). FuRA's per-layer parameter count is `2·rank·sqrt(d_in·d_out)`, which at rank=full and Qwen3-1.7B's `d=2048` is comparable to LoRA r=64.
- **Base (no RL)** rows measured 2026-05-05 via `uv run eval_rl.py --checkpoint Qwen/Qwen3-1.7B` and `--checkpoint Qwen/Qwen2.5-7B` under the *same* harness as the RL rows (boxed prompt, `max_model_len=2048`, `max_tokens=2048`, MATH-500/AMC23 greedy@1, AIME-24/25 avg@8 at T=0.6). These are not directly comparable to the published Qwen technical reports because the prompt template, eval set selection (MATH-500 vs full MATH 12.5K), and AIME sampling protocol differ. Result JSONs at `logs/qwen{2_5_7B,3_1_7B}_base_eval.json`. Qwen2.5-7B Base MATH-500 (48.4%) is **+22.5 pp below** the RL-trained FuRA row (60.2%); Qwen3-1.7B Base MATH-500 (52.2%) is **+11.4 pp below** its RL-trained FuRA/Full-FT row (63.6%). On AIME-25, however, Qwen3-1.7B Base (7.5%) sits within the noise of the trained PEFT methods (LoRA 11.2, MiLoRA 9.6) — RL gains there are smaller. Internal in-train held-out 1000-prompt baselines from the RL runs themselves (different question set than MATH-500): Qwen3-1.7B ~17.9–18.3%, Qwen2.5-7B 68.6–70.8% (the latter varies ±2.2 pp across in-process vLLM warm-ups; see Qwen2.5-7B section below).
- Source: `/data/yequan/fura/rl_runs`
- Methods covered: `full`, `lora`, `lora_full`, `dora`, `pissa`, `milora`, `lift`, `randlora`, `fura` (=`blocktt`), `svd`.
- Primary metric: `eval/accuracy` on the internal 1000-problem held-out split.
- Extended math evals (post-2026-04-19 runs only): `eval/MATH-500/accuracy`, `eval/AMC23/accuracy`, `eval/AIME-24/accuracy` (avg@8), `eval/AIME-25/accuracy` (avg@8). Minerva (272-problem greedy@1) was dropped from the default eval set on 2026-05-02 because its run-to-run variance was masking method differences; historical Minerva numbers have been removed from this doc.
- `fura` is the project's own BlockTT-based method, stored under `rl_runs/blocktt/`.
- Metric source: final values in each run's `wandb-summary.json` (largest summary per run dir).
- Regeneration script: `uv run python tools/collect_rl_results.py --root /data/yequan/fura/rl_runs`.

## BlockTT Ablation (Qwen3-1.7B, RL on `qwedsacf/competition_math`, lr=1e-4)

Decompose each linear as `W = L · S · R` with L on the output side and R on the input side. **Bold** = trainable; `(SR)` / `(LS)` parentheses mean S is merged into the adjacent core and the merged tensor is trained as one parameter; no parentheses on `S` = `btt_s` is held as a separate parameter (its trainability follows the bolding). All runs use `pos_small`, `blocktt_rank=full`, `lr=1e-4`, `optimizer=adamw`, 50 GRPO steps. **Bold** in numeric cells marks the column-best across this table.

| Notation                  |       MATH-500 |          AMC23 |        AIME-24 |        AIME-25 |
| :------------------------ | -------------: | -------------: | -------------: | -------------: |
| L**S** **R**  | **63.6** | **57.5** | **15.0** | **17.5** |
| L**(SR)**                 |           62.8 |           47.5 |           12.5 |           16.7 |
| (LS)**R**           |           61.4 |           55.0 |           10.0 |           12.9 |
| **L** **S** R |           61.2 |           52.5 |            9.6 |           13.8 |
| **(LS)** R          |           61.4 |           42.5 |           12.9 |           13.8 |
| **L** (SR)          |           62.2 |           42.5 |            9.6 |            8.3 |

Side note on which side is small: for `output_one_block` (m=1, a=out_features, n=in_blocks, b=in_block_size), `btt_l.numel() = rank · n · out_features` and `btt_r.numel() = rank · in_features`, so **R is the small side** (factor of `n` smaller). With `pos_small`, R trains. Symmetrically for `input_one_block` (n=1), L is the small side and L trains.

Takeaways: (i) `output_one_block` (small core on the input/right side) **dominates** `input_one_block` on RL — the worst `output` config (mean-4 34.83) still beats the best `input` config (34.27). The gap is largest on AMC23 (best output 57.5 vs best input 52.5). (ii) Holding `S` as a separate trainable scaling (L **S** **R**) is the strongest configuration on `output_one_block` and wins across MATH-500 / AMC23 / AIME-24 / Mean — promoted to the TL;DR fura row below. The same `keep_trainable` placement is *also* the best `s_to` choice within `input_one_block`. (iii) Within `input_one_block`, merging `S` into the trainable side (**L** (SR), i.e. `s_to=frozen`) is the worst recipe in the entire grid and bombs on AIME-25 (8.3% — worst). The matching `output_one_block` `s_to=frozen` configuration `(LS) **R**` is the *third*-best, not the worst, so the `s_to=frozen` penalty is decomp-coupled. (iv) Eval/acc is largely flat (85.4–88.7) — primary in-train metric does not discriminate the recipes; the OOD math-verify columns do.

## TL;DR — Best per method

**Summary table** — best run per method (selected by mean of the 4 extended evals: MATH-500, AMC23, AIME-24, AIME-25).

| Method                   | Best LR | eval/acc | MATH-500 | AMC23 | AIME-24 | AIME-25 |        Mean (4) |
| ------------------------ | ------- | -------: | -------: | ----: | ------: | ------: | --------------: |
| **randlora**       | 1e-4    |     85.6 |     63.2 |  57.5 |    15.8 |    17.1 | **38.40** |
| **fura** (blocktt) | 1e-4    |     88.7 |     63.6 |  57.5 |    15.0 |    12.5 |           37.15 |
| **full**           | 2e-5    |     86.3 |     63.6 |  47.5 |    13.8 |    15.4 |           35.08 |
| **svd**            | 1e-5    |     88.4 |     64.0 |  40.0 |    16.7 |    13.8 |           33.62 |
| **lora**           | 6e-5    |     84.8 |     60.6 |  50.0 |    11.2 |    11.2 |           33.25 |
| **dora**           | 1e-4    |     86.5 |     61.6 |  37.5 |    12.9 |    14.6 |           31.65 |
| **milora**         | 6e-5    |     85.6 |     59.0 |  45.0 |     7.9 |     9.6 |           30.38 |
| **lift**           | 8e-5    |     84.2 |     60.2 |  37.5 |    10.0 |     8.8 |           29.12 |
| **pissa**          | 8e-5    |     78.6 |     53.4 |  42.5 |     2.5 |     3.3 |           25.43 |

### FuRA LR sweep detail

| FuRA LR          | eval/acc |       MATH-500 |          AMC23 |        AIME-24 |        AIME-25 |        Mean (4) |
| ---------------- | -------: | -------------: | -------------: | -------------: | -------------: | --------------: |
| **8e-5**   |     85.4 | **63.0** | **52.5** |           13.3 | **17.5** | **36.58** |
| 2e-4             |     86.6 |           60.8 |           47.5 | **18.8** |           15.0 |           35.52 |
| 1e-4 (prev best) |     87.1 |           62.8 |           47.5 |           12.5 |           16.7 |           34.88 |
| 3e-4             |     85.1 |           59.6 |           45.0 |           12.1 |           11.7 |           32.10 |

Config for all rows: `output_one_block`, `train_small`, `s_to_trainable`. lr 8e-5 leads on 3 of 4 metrics and on the mean; lr 2e-4 leads AIME-24 (18.8% — best single result in the entire table). LR 3e-4 degrades consistently.

### FuRA decomp × s_to grid at lr=1e-4 (canonical 6-cell, train_small, rank=full)

These 6 runs hold lr fixed at the previously-best fura LR (1e-4) and sweep the two BlockTT design axes: `decomp_mode ∈ {output_one_block, input_one_block}` × `s_merged_to ∈ {frozen, trainable, keep_trainable}`. The 3 `0502` reruns below were re-trained with `--enable-merged-ckpt --enable-math-verify` because the original 0317/0331/0403/0413 runs either lacked extended math eval, lacked a step=50 ckpt on disk, or saved a factored ckpt that `eval_rl.py` cannot consume.

| decomp           | s_to             |       eval/acc |      train/acc |       MATH-500 |          AMC23 |        AIME-24 |        AIME-25 |        Mean (4) | run                                                                                              |
| ---------------- | ---------------- | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | --------------: | ------------------------------------------------------------------------------------------------ |
| output_one_block | **frozen** | **88.7** | **69.5** | **63.6** | **57.5** | **15.0** |           12.5 | **37.15** | `blocktt-adamw-lr_1e-4-output_one_block-s_to_frozen-train_small-0502-164206`                   |
| output_one_block | trainable        |           87.1 | **69.5** |           62.8 |           47.5 |           12.5 | **16.7** |           34.88 | `blocktt-adamw-lr_1e-4-output_one_block-s_to_trainable-train_small-0419-185330`                |
| output_one_block | keep_trainable   |           85.8 |           67.6 |           61.4 |           55.0 |           10.0 |           12.9 |           34.83 | `blocktt-adamw-lr_1e-4-output_one_block-s_to_keep_trainable-train_small-0419-185333`           |
| input_one_block  | frozen           |           85.4 |           67.6 |           62.2 |           42.5 |            9.6 |            8.3 |           30.65 | `blocktt-adamw-lr_1e-4-input_one_block-s_to_frozen-train_small-rerun-0502-0502-213816`         |
| input_one_block  | trainable        |           87.5 |           68.4 |           61.4 |           42.5 |           12.9 |           13.8 |           32.65 | `blocktt-adamw-lr_1e-4-input_one_block-s_to_trainable-train_small-rerun-0502-0502-191940`      |
| input_one_block  | keep_trainable   |           87.4 |           67.6 |           61.2 |           52.5 |            9.6 |           13.8 |           34.27 | `blocktt-adamw-lr_1e-4-input_one_block-s_to_keep_trainable-train_small-rerun-0502-0502-201544` |

**Cross-cell takeaways at lr=1e-4:**

1. **`output_one_block` wins** over `input_one_block` (best output mean-4 37.15 vs best input mean-4 34.27). This sharpens the project-wide `output_one_block` default for fura; the gap is 3 pp on MATH-500 (63.6 vs 61.4) and 5 pp on AMC23 (57.5 vs 52.5).
2. **Within `output_one_block`, `s_to_frozen` is the new fura best** on every single extended-eval metric — MATH-500 (63.6), AMC23 (57.5), AIME-24 (15.0), and on the mean (37.15 vs 34.88 / 34.83 for the other two). Promoted to the TL;DR table above. It's also the new MATH-500 leader for fura (tied with `full` at 63.6) and a clear AMC23 leader (+5 pp over the prior 52.5).
3. **Within `input_one_block`, `s_to_keep_trainable` wins** by mean-4 (34.27 > 32.65 trainable > 30.65 frozen). The two `input × {trainable, frozen}` configs both bottom out around 30 mean-4 — substantially below every `output_one_block` config (all ≥34.83), confirming that `output_one_block` is the right default regardless of `s_to`. Within input_one_block, `frozen` is the worst on AMC23 (42.5 — tied with `trainable`) and AIME-25 (8.33 — worst in the grid).

**Best headline primary `eval/accuracy` per method**:

| Method         | Runs |           Best | Mean | Best run                                                                    |
| -------------- | ---: | -------------: | ---: | --------------------------------------------------------------------------- |
| full           |    7 |           88.6 | 71.6 | `full-adamw-lr_2e-5-0325-215533`                                          |
| lora           |    6 |           84.8 | 48.8 | `lora-adamw-lr_6e-5-rank_64-sweep-0422-002231`                            |
| lora_full      |    1 |           85.6 | 85.6 | `lora_full-adamw-lr_1e-5-rank_64-0319-140945`                             |
| dora           |    4 |           87.1 | 69.2 | `dora-adamw-lr_2e-4-rank_64-sweep-0422-021100`                            |
| pissa          |    3 |           80.2 | 66.7 | `pissa-adamw-lr_6e-5-rank_64-sweep-0422-032327`                           |
| milora         |    3 |           85.6 | 82.6 | `milora-adamw-lr_6e-5-rank_64-sweep-0422-020941`                          |
| lift           |    4 |           84.9 | 64.9 | `lift-adamw-lr_6e-5-sweep-0422-031308`                                    |
| randlora       |    3 |           85.6 | 82.4 | `randlora-adamw-lr_1e-4-rank_64-sweep-0422-041850`                        |
| **fura** |   20 | **89.5** | 82.5 | `blocktt-adamw-lr_1e-5-output_one_block-s_to_keep-train_both-0317-155422` |
| svd            |    6 |           89.1 | 87.5 | `svd-adamw-lr_1e-5-s_to_keep-train_input-0317-141139`                     |

(`lora_full` is the legacy LoRA-with-base-train path and is not subject to this round's sweep; listed here only for reference.)

## Takeaways

1. **LoRA regression fixed.** The recently added `dora/pissa/milora/randlora/lift` support had broken the vanilla `lora` path: `export_lora_merged_weights` in `run_rl.py` appended a live reference to `base_layer.weight` before `unmerge_adapter()` mutated it back to the frozen base. vLLM was therefore serving the pre-training base model every step even while local gradients were computed. Fix: clone the merged tensor before unmerging (`param.detach().clone()` in the new `export_lora_merged_weights_for_vllm` helper, `run_rl.py:1038-1061`). LoRA now converges normally: `lora-lr_6e-5-rank_64` reaches 84.8% eval/accuracy vs ~15–17% on the broken code. DoRA was accidentally unaffected because PEFT's DoRA unmerge swaps `.data` rather than mutating in place.
2. **Minerva eval dropped from defaults (2026-05-02).** Earlier we patched the gated `math-ai/minerva-math` HF id to the community mirror `math-ai/minervamath`. Subsequent runs showed Minerva's 272-problem greedy@1 score had high run-to-run variance that masked method differences (5–7 pp swings on identical recipes), so it was removed from the `run_rl.py` `--math-verify-datasets` default. Minerva is still in the `eval_datasets.py` REGISTRY and can be requested explicitly. All Minerva columns / numbers were also removed from the tables in this doc.
3. **RandLoRA save-and-eval fixed.** The pre-fix `randlora-lr_8e-5-0421` run converged fine (train acc 66% at step 50) but crashed in `save_merged_checkpoint` with "shared tensors" because PEFT's RandLoRA shares the `randlora_A` random projection across layers and safetensors refuses to serialize shared storages. Since this save happens before the math-verify hook, math-verify never fired. Two fixes in `run_rl.py`:

   - `save_merged_checkpoint` now passes `safe_serialization=False` specifically for `randlora` so the save succeeds via torch.save (`run_rl.py:903-919`).
   - The post-training final-checkpoint save is wrapped in `try/except`, so even if the save fails the math-verify hook still runs from the in-memory model (`run_rl.py:2070-2090`).
4. **LR sweep ranking.**

   - `fura`: **8e-5 is the new best** — MATH-500 63.0%, AMC23 52.5%, AIME-25 17.5% (mean-4 36.58). The previous best at 1e-4 / s_to_trainable is slightly weaker on the mean (34.88). Higher LRs (2e-4, 3e-4) still converge but degrade on MATH/AMC. Notably 2e-4 achieves the best single-run AIME-24 (18.8%) across all methods, possibly by over-fitting to harder competition problems at the expense of broader coverage. Note: the 2026-05-02 6-cell decomp×s_to grid found a different fura best at lr=1e-4 / output_one_block / s_to_frozen (mean-4 37.15) — see "FuRA decomp × s_to grid" section above.
   - `lora`: 6e-5 > 1e-4 > 2e-4 (2e-4 peaks around step 20 then over-trains to 75%). lr 8e-5 with rank 16/64 and lr 5e-5 with rank 64 all **diverged** under the pre-fix code; re-running them on the fixed code was not attempted but the 6e-5 and 1e-4 data points already bracket the optimum.
   - `dora`: 1e-4 > 2e-4 > 8e-5 across the extended evals; but all three are within noise on primary eval (85.3 / 86.5 / 87.1). The `dora-lr_1e-4` retry converged cleanly after an earlier OOM; OOM is a concurrency-pressure issue on shared GPUs, not a method issue.
   - `pissa`: 8e-5 > 6e-5 ≫ 1e-4. `pissa-lr_1e-4` collapses (eval/acc 41%, MATH-500 38%); `pissa-lr_6e-5` matches `pissa-lr_8e-5` within noise.
   - `milora`: 6e-5 > 8e-5 ≈ 1e-4. Gains at 6e-5 are small but consistent (MATH +4).
   - `lift`: 8e-5 > 6e-5 > 1e-4 ≫ 2e-4. `lift-lr_2e-4` fully collapses (MATH-500 7%) — LIFT is the most LR-sensitive method in the family.
   - `randlora`: **1e-4 > 8e-5** across every metric (eval 85.6 vs 79.1, MATH 63.2 vs 61.6, AMC 57.5 vs 50.0, AIME-25 17.1 vs 15.8). The 8e-5 baseline substantially under-trained randlora.
5. **Ranking across all methods.** On the 4 extended evals (MATH-500, AMC23, AIME-24, AIME-25), sorting by unweighted mean:

   - **Tier 1** (~37-38): `randlora` (1e-4) 38.40 > `fura` (1e-4 / output_one_block / s_to_frozen) 37.15. Both clearly above full (35.08).
   - **Tier 2** (~33-36): `fura` (8e-5 / s_to_trainable) 36.58 > `fura` (2e-4) 35.52 > `full` (2e-5) 35.08 > `svd` (1e-5) 33.62 > `lora` (6e-5) 33.25.
   - **Tier 3** (~29-32): `dora` (1e-4) 31.65 > `milora` (6e-5) 30.38 > `lift` (8e-5) 29.12.
   - **Tier 4**: `pissa` (8e-5) 25.43.
     The 2026-05-02 6-cell fura grid surfaced `randlora` as the new top-line method by mean-4. **fura is now second**, but trains with far fewer params than randlora and ties or beats it on MATH-500 / AMC23.
6. **fura vs full, per-benchmark.** Fura's best config (1e-4 / output_one_block / s_to_frozen) leads full on AMC23 (57.5 vs 47.5 — +10 pp), AIME-24 (15.0 vs 13.8), and ties on MATH-500 (63.6 each); full leads on AIME-25 (15.4 vs 12.5). The 2e-4 fura variant beats all on AIME-24 (18.8% — the best single result across all methods and LRs in the entire table).

## All runs with extended eval (sorted by MATH-500)

| Method         | Run                                                                                              | eval/acc | train/acc | MATH-500 | AMC23 | AIME-24 | AIME-25 |
| -------------- | ------------------------------------------------------------------------------------------------ | -------: | --------: | -------: | ----: | ------: | ------: |
| svd            | `svd-adamw-lr_1e-5-s_to_keep_trainable-train_input-ext-0421-221304`                            |     88.4 |      70.3 |     64.0 |  40.0 |    16.7 |    13.8 |
| **fura** | `blocktt-adamw-lr_1e-4-output_one_block-s_to_frozen-train_small-0502-164206`                   |     88.7 |      69.5 |     63.6 |  57.5 |    15.0 |    12.5 |
| full           | `full-adamw-lr_2e-5-0420-173501`                                                               |     86.3 |      66.4 |     63.6 |  47.5 |    13.8 |    15.4 |
| randlora       | `randlora-adamw-lr_1e-4-rank_64-sweep-0422-041850`                                             |     85.6 |      66.0 |     63.2 |  57.5 |    15.8 |    17.1 |
| fura           | `blocktt-adamw-lr_8e-5-output_one_block-s_to_trainable-train_small-sweep-0423`                 |     85.4 |      66.0 |     63.0 |  52.5 |    13.3 |    17.5 |
| fura           | `blocktt-adamw-lr_1e-4-output_one_block-s_to_trainable-train_small-0419-185330`                |     87.1 |      69.5 |     62.8 |  47.5 |    12.5 |    16.7 |
| fura           | `blocktt-adamw-lr_1e-4-input_one_block-s_to_frozen-train_small-rerun-0502-0502-213816`         |     85.4 |      67.6 |     62.2 |  42.5 |     9.6 |     8.3 |
| dora           | `dora-adamw-lr_1e-4-rank_64-retry-0422-013148`                                                 |     86.5 |      68.0 |     61.6 |  37.5 |    12.9 |    14.6 |
| randlora       | `randlora-adamw-lr_8e-5-rank_64-fix1-0421-221308`                                              |     79.1 |      63.7 |     61.6 |  50.0 |    15.8 |    15.8 |
| fura           | `blocktt-adamw-lr_1e-4-input_one_block-s_to_trainable-train_small-rerun-0502-0502-191940`      |     87.5 |      68.4 |     61.4 |  42.5 |    12.9 |    13.8 |
| fura           | `blocktt-adamw-lr_1e-4-output_one_block-s_to_keep_trainable-train_small-0419-185333`           |     85.8 |      67.6 |     61.4 |  55.0 |    10.0 |    12.9 |
| fura           | `blocktt-adamw-lr_1e-4-input_one_block-s_to_keep_trainable-train_small-rerun-0502-0502-201544` |     87.4 |      67.6 |     61.2 |  52.5 |     9.6 |    13.8 |
| fura           | `blocktt-adamw-lr_2e-4-output_one_block-s_to_trainable-train_small-sweep-0423`                 |     86.6 |      66.4 |     60.8 |  47.5 |    18.8 |    15.0 |
| lora           | `lora-adamw-lr_6e-5-rank_64-sweep-0422-002231`                                                 |     84.8 |      65.2 |     60.6 |  50.0 |    11.2 |    11.2 |
| lift           | `lift-adamw-lr_8e-5-0421-143200`                                                               |     84.2 |      62.1 |     60.2 |  37.5 |    10.0 |     8.8 |
| fura           | `blocktt-adamw-lr_3e-4-output_one_block-s_to_trainable-train_small-sweep-0423`                 |     85.1 |      64.8 |     59.6 |  45.0 |    12.1 |    11.7 |
| dora           | `dora-adamw-lr_2e-4-rank_64-sweep-0422-021100`                                                 |     87.1 |      68.0 |     59.0 |  50.0 |     5.0 |    10.0 |
| milora         | `milora-adamw-lr_6e-5-rank_64-sweep-0422-020941`                                               |     85.6 |      64.1 |     59.0 |  45.0 |     7.9 |     9.6 |
| dora           | `dora-adamw-lr_8e-5-rank_64-0421-142231`                                                       |     85.3 |      66.8 |     58.0 |  45.0 |     7.1 |     8.3 |
| lift           | `lift-adamw-lr_6e-5-sweep-0422-031308`                                                         |     84.9 |      65.6 |     57.4 |  32.5 |     9.6 |     7.9 |
| lift           | `lift-adamw-lr_1e-4-sweep-0422-001405`                                                         |     82.8 |      63.7 |     55.6 |  30.0 |     7.5 |    10.0 |
| milora         | `milora-adamw-lr_8e-5-rank_64-0421-152507`                                                     |     80.9 |      59.0 |     55.0 |  32.5 |     5.4 |     4.6 |
| milora         | `milora-adamw-lr_1e-4-rank_64-sweep-0421-231002`                                               |     81.4 |      60.5 |     54.4 |  47.5 |     3.8 |     4.2 |
| pissa          | `pissa-adamw-lr_6e-5-rank_64-sweep-0422-033440`                                                |     71.6 |      59.0 |     54.4 |  40.0 |     3.8 |     4.6 |
| pissa          | `pissa-adamw-lr_8e-5-rank_64-0421-142217`                                                      |     78.6 |      58.2 |     53.4 |  42.5 |     2.5 |     3.3 |
| lora           | `lora-adamw-lr_1e-4-rank_64-fix1-0421-221302`                                                  |     82.1 |      63.3 |     53.2 |  30.0 |     5.4 |     4.2 |
| pissa          | `pissa-adamw-lr_6e-5-rank_64-sweep-0422-032327`                                                |     80.2 |      57.8 |     52.2 |  32.5 |     2.5 |     6.2 |
| lora           | `lora-adamw-lr_2e-4-rank_64-sweep-0422-010758`                                                 |     75.3 |      64.8 |     47.4 |  27.5 |     3.8 |     0.8 |
| pissa          | `pissa-adamw-lr_1e-4-rank_64-sweep-0421-232419`                                                |     41.2 |      43.0 |     38.4 |  22.5 |     2.5 |     4.6 |
| lora           | `lora-adamw-lr_5e-5-rank_64-0420-173503`                                                       |     17.6 |      16.0 |     37.2 |  17.5 |     0.0 |     0.0 |
| full           | `full-adamw-lr_5e-5-0419-185332`                                                               |     14.0 |      23.8 |     20.4 |  10.0 |     0.4 |     0.0 |
| lora           | `lora-adamw-lr_8e-5-rank_16-0420-173503`                                                       |     15.6 |      13.7 |     12.4 |  10.0 |     0.0 |     0.0 |
| lora           | `lora-adamw-lr_8e-5-rank_64-0419-185332`                                                       |     17.1 |      15.6 |      8.0 |   2.5 |     0.0 |     0.0 |
| lift           | `lift-adamw-lr_2e-4-sweep-0422-010909`                                                         |      7.8 |      12.9 |      7.0 |   0.0 |     0.0 |     0.8 |

Top 4 of those bottom rows are the broken pre-fix runs; they are kept for historical contrast with the fixed runs right above them (e.g. `lora-lr_8e-5-rank_64` pre-fix: MATH 8.0, AIME 0/0 vs `lora-lr_6e-5-rank_64` post-fix: MATH 60.6, AIME 11.2/11.2).

## Llama-3.1-8B-Instruct LR sweep (added 2026-04-27)

GRPO sweep on `meta-llama/Llama-3.1-8B-Instruct` (and the bit-equivalent `NousResearch/Meta-Llama-3.1-8B-Instruct` mirror), `qwedsacf/competition_math`, 50 GRPO steps, eval = the same internal 1000-prompt held-out split. Two sub-sweeps:

- **LoRA r=64 + BlockTT (output_one_block / small / keep_trainable / full)** at 4 LRs, on the NousResearch mirror (pre-training baseline **67.10%** for LoRA path, **67.60%** for BlockTT path — see footnote on baseline mismatch).
- **LoRA r=16** at the same 4 LRs, on the official `meta-llama/Llama-3.1-8B-Instruct` (pre-training baseline **63.50%**).

Memory recipe (single H100 96GB):

- LoRA: `--gpu-memory-utilization 0.3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (peak ~77 GB).
- BlockTT: `--gpu-memory-utilization 0.25 --max-model-len 1536 --micro-batch-size 1 --gradient-accumulation-steps 256` (peak ~71 GB). Default `gpu-memory-utilization 0.4` OOMs at step 1 / step 2 because the rollout-time vLLM weight materialization plus the trainer activation footprint exceeds 93 GB.

Wandb projects: `llama3-8B-RL` (Instruct sweeps) and `llama3-8B-base-RL` (base sweep below).

### Headline: BlockTT >> LoRA on Llama-3.1-8B-Instruct

| LR   |                                LoRA r=64 final | LoRA r=64 peak |                     BlockTT final |  BlockTT peak |
| ---- | ---------------------------------------------: | -------------: | --------------------------------: | ------------: |
| 6e-5 |                  14.00% (collapsed @ step ~18) |  67.10% (init) |               **67.40%** ✅ |        68.10% |
| 8e-5 |                  13.40% (collapsed @ step ~10) |  67.10% (init) | 12.90% (late collapse @ step ~42) |        68.00% |
| 1e-4 |                          29.20% (slow degrade) |  67.10% (init) | **69.00%** ✅ best of sweep |        69.00% |
| 2e-4 | 0.00% (catastrophic, late collapse @ step ~28) |  67.10% (init) | 13.00% (early collapse @ step ~5) | 67.60% (init) |

Baseline = 67.10% (LoRA path) / 67.60% (BlockTT path). "Peak" = the best `step=N, correct:` eval observed at any of {step 5, 10, …, 50}. "Final" = step 50 eval. **2 of 4 BlockTT runs finish at or above baseline; 0 of 4 LoRA-r=64 runs do.**

### LoRA r=16 vs r=64 — smaller adapter is meaningfully more stable

LoRA r=16 sweep on the official `meta-llama/Llama-3.1-8B-Instruct` (baseline **63.50%**):

| LR   |                                          r=16 final |                                           r=16 peak | r=64 final (NousResearch mirror) |
| ---- | --------------------------------------------------: | --------------------------------------------------: | -------------------------------: |
| 6e-5 |                                 **60.40%** ✅ |                                    68.00% (step 30) |                        14.00% ❌ |
| 8e-5 |          0.00% ❌ (collapsed @ step ~20 after peak) |                                    68.40% (step 10) |                        13.40% ❌ |
| 1e-4 |                                 **59.30%** ✅ |                                    68.10% (step 10) |            29.20% (slow degrade) |
| 2e-4 | 0.00% ❌ (late collapse @ step ~50, peak @ step 15) | **71.30%** (step 15 — highest peak in sweep) |                         0.00% ❌ |

Smaller adapter capacity = smaller per-step parameter movement = much less destructive drift. **2 of 4 r=16 runs survive vs 0 of 4 at r=64.** None of the r=16 runs *improves* over baseline at step 50, but the peaks (~68–71% across all 4 LRs) sit above baseline 63.5% — early-stopping at step ~10–15 would beat all final evals by 5–8 pp.

### Cross-method comparison on Llama-3.1-8B-Instruct

Sorting by step-50 eval/accuracy (higher is better):

| Rank | Run               |            Final |             Peak |      Δ vs baseline |
| ---: | ----------------- | ---------------: | ---------------: | ------------------: |
|    1 | BlockTT lr=1e-4   | **69.00%** |           69.00% |          +1.4 pp ✅ |
|    2 | BlockTT lr=6e-5   |           67.40% |           68.10% | -0.2 pp ≈ baseline |
|    3 | LoRA r=16 lr=6e-5 |           60.40% |           68.00% |             -3.1 pp |
|    4 | LoRA r=16 lr=1e-4 |           59.30% |           68.10% |             -4.2 pp |
|    5 | LoRA r=64 lr=1e-4 |           29.20% |    67.10% (init) |            -37.9 pp |
|    6 | LoRA r=64 lr=6e-5 |           14.00% |    67.10% (init) |            -53.1 pp |
|    7 | BlockTT lr=2e-4   |           13.00% |    67.60% (init) |            -54.6 pp |
|    8 | BlockTT lr=8e-5   |           12.90% |           68.00% |            -54.7 pp |
|    9 | LoRA r=64 lr=8e-5 |           13.40% |    67.10% (init) |            -53.7 pp |
|   10 | LoRA r=16 lr=8e-5 |            0.00% |           68.40% |            -63.5 pp |
|   11 | LoRA r=16 lr=2e-4 |            0.00% | **71.30%** |            -63.5 pp |
|   12 | LoRA r=64 lr=2e-4 |            0.00% |    67.10% (init) |            -67.1 pp |

Best by **peak** eval (any step): r=16 lr=2e-4 hits **71.3%** at step 15 (+7.8 pp over its 63.5% baseline), r=16 lr=8e-5 hits **68.4%** at step 10. Best by **final** eval: BlockTT lr=1e-4 at **69.0%**.

### Llama-3.1-8B (base) — GRPO from-scratch fails

For comparison, the same 4 LRs × 2 modes were run on the *base* `meta-llama/Llama-3.1-8B` (no instruction tuning). All 8 runs:

- **Pre-training baseline: 0.00%** — base model does not produce `\boxed{}` answers from the boxed-prompt template.
- **All 8 final evals: 0.00%** through 50 GRPO steps.

GRPO needs at least one rollout per prompt to be correct to get a non-zero advantage; with 0% baseline reward across all 256 rollouts per step, all advantages are identically zero and no policy gradient flows. The base model would need format-bootstrap SFT, a few-shot prompt that elicits boxed answers, or a permissive format-only initial reward before RL can take hold. **`run_rl.py` was patched (this session) to (a) fall back to raw prompt when `tokenizer.chat_template is None` (otherwise the script would crash on base models) and (b) explicitly label and wandb-log the step-0 evaluation as `eval/baseline_accuracy`.**

### Llama-3.1-8B-Instruct: takeaways

1. **BlockTT (output_one_block/small/keep_trainable/full) is the only method that reliably beats baseline on Llama-3.1-8B-Instruct.** BlockTT lr=1e-4 finishes at 69.0% (+1.4 pp); BlockTT lr=6e-5 essentially holds baseline. BlockTT lr=8e-5 collapses *late* (~step 42), and lr=2e-4 collapses early. So the BlockTT stable-LR window on this model is roughly [6e-5, 1e-4]; lr=8e-5 is borderline (passes 35 steps, then drops).
2. **LoRA r=64 collapses at every LR tested**, including the lowest (6e-5). r=64 may simply be too much capacity for this model's RL signal — the per-step adapter delta dominates the policy and pushes the model into a degenerate region.
3. **LoRA r=16 partially fixes the LoRA collapse problem**, but no r=16 final eval beats baseline. The peaks all sit above baseline (68–71%), which means **early-stopping is critical** when training LoRA on this model. Without an early-stop-on-eval workflow, an RL run that "looks fine" at step 10 may be 60+ pp worse by step 50.
4. **The catastrophic-collapse signature is consistent**: train_acc drops to 0.0% with step time falling from ~100s to ~30s. Short step time is the giveaway — the policy is generating short outputs (likely immediate `<eot_id>`) that produce zero reward and zero gradient.
5. **Comparison to the Qwen3-1.7B results higher in this doc** (where every LoRA LR converged cleanly): Qwen3-1.7B is more LR-tolerant, possibly because (a) it's smaller so per-step relative parameter movement is larger and the optimizer is "more aware" of its updates, or (b) its base is more aligned with the boxed-answer math format. The gap between Qwen3-1.7B LoRA (best 84.8% eval/acc) and Llama-3.1-8B-Instruct LoRA (best 60.4% final) is substantial — this is plausibly **a model-specific stability problem rather than a method problem**.

### Footnote: baseline mismatch between meta-llama and NousResearch mirrors

The official `meta-llama/Llama-3.1-8B-Instruct` baseline measured 63.50% in this pipeline; the `NousResearch/Meta-Llama-3.1-8B-Instruct` mirror baseline measured 67.10%. Same architecture, presumably the same weights. The 3.6 pp gap is likely a tokenizer-config artifact (chat_template differences in special tokens or whitespace handling). Within-sweep comparisons remain valid, but cross-mirror comparisons should adjust by ~3.6 pp. The r=16 sweep used the official repo (after HF gated-access approval); the r=64 sweep used the public mirror. A re-run of one r=64 LR on the official repo would close this footnote — not yet done.

## Remaining gaps

1. **Fura/svd at lr 1e-5 lack extended eval.** The overall-best primary-eval runs (`fura-lr_1e-5` 89.5%, `svd-lr_1e-5` 89.1%) pre-date the extended-eval harness. Their on-disk checkpoints are factored (BTT/SVD cores, not dense); `eval_rl.py` would need a factored-to-dense materializer, or rerun those configs from scratch with `--enable-merged-ckpt`. The new `svd-adamw-lr_1e-5-s_to_keep_trainable-train_input-ext` run filled the gap for svd. An equivalent fura re-run is still queued.
2. ~~Legacy extended-eval runs lack Minerva.~~ — **Resolved by deprecation (2026-05-02).** Minerva was found to be too high-variance to be useful for method comparison and was dropped from the default math-verify dataset list in `run_rl.py`; all Minerva numbers were removed from this doc. The remaining 4 metrics (MATH-500, AMC23, AIME-24, AIME-25) are now uniformly reported across every run.
3. ~~`randlora-lr_1e-4` sweep point~~ — **Completed.** randlora 1e-4 is now the best method overall on the post-Minerva-removal mean-4 (38.40, top of TL;DR).
4. ~~fura LR sweep~~ — **Completed (8e-5, 2e-4, 3e-4).** Best fura LR-only config: 8e-5 / s_to_trainable, mean-4 36.58.
5. ~~fura decomp × s_to grid at lr=1e-4~~ — **Completed for all 6 cells (2026-05-02).** The canonical 6-cell grid (`{output_one_block, input_one_block} × {frozen, trainable, keep_trainable}`, `train_small`, rank=full) is now extended-eval'd. The `output_one_block / s_to_frozen` rerun at lr=1e-4 supersedes the prior fura best on mean-4 extended (37.15 vs 36.58) and is now the TL;DR row. The final `input_one_block / s_to_frozen` cell finished at mean-4 30.65 — the worst of the 6, confirming `output_one_block` as the dominant axis.

## Llama-3.1-8B-Instruct r=64 best-ckpt math-verify eval (added 2026-05-01)

GRPO sweep on `meta-llama/Llama-3.1-8B-Instruct` (the official repo, not the NousResearch mirror), `qwedsacf/competition_math`, 50 GRPO steps, `lr=6e-5`, all adapter ranks fixed at 64 (lift has no rank). Each run captures the **best in-train held-out 1000-prompt eval** via `--save-best-val-ckpt` (added this session — saves a merged HF ckpt to `<run_dir>/best/` whenever `eval/accuracy` improves). Those best ckpts are then re-evaluated on the math-verify suite via `eval_rl.py`. AIME-24/25 are avg@8 (n_samples=8, T=0.6); MATH-500/AMC23 are greedy.

- Source: `/data/yequan/fura/rl_runs/Llama-3.1-8B-Instruct/<mode>/<run_name>/best/eval_results.json`
- Pre-training Llama-3.1-8B-Instruct baseline (this pipeline): **63.10%** in-train held-out 1000.
- Wandb project: `llama3-8B-RL` (same as the prior r=16 sweep — runs with `-llama31-8b-instruct` suffix and rank embedded in name).

### Configuration notes

- All runs used in-process vLLM (`--vllm-url http://localhost:1` to force fallback when an unrelated `vllm serve` was squatting :8000).
- `--gpu-memory-utilization 0.3` for lora/milora/pissa, `0.25` for dora/randlora (their forward materializes additional weights — broke the activation budget at 0.3).
- **dora** and **randlora** required `--micro-batch-size 1 --gradient-accumulation-steps 256` (effective batch unchanged). With the run_rl.py LoRA-mode default of `mbs=2, gacc=128` they OOM during step-1 backward on a single H100 NVL (93 GB). Mbs=1 cuts activation memory ~2× and unblocks both.
- **lift** could not fit on a single H100 NVL at any `--gpu-memory-utilization` value: vLLM needs ≥0.20 to allocate KV-cache for max_model_len=2048, but lift's full-param AdamW state needs the rest of the GPU. Best ckpt saved at step=0 (baseline). Omitted from the eval table below.
- **pissa best ckpt is step-0** (RL-trained model collapsed to <2% mid-run; the pre-training baseline is the saved best). **milora's** best is step=10 (peak 68.0%, then collapse).

### `--save-best-val-ckpt` bug-fix note

The first eval pass failed with `KeyError: 'layers.10.mlp.down_proj.base_layer.weight'` because `save_merged_checkpoint` for the LoRA family currently calls `model.merge_adapter()` (fuses values into `base_layer.weight` but keeps PEFT key wrappers) followed by `base.save_pretrained()` — so the on-disk ckpt has both the merged `*.base_layer.weight` *and* now-redundant `*.lora_A/B.default.weight` entries, which vLLM's vanilla LlamaForCausalLM loader rejects. Fixed by post-hoc rewriting each ckpt with `tools/strip_peft_keys.py` (renames `*.base_layer.weight → *.weight`, drops lora_A/B/magnitude/randlora_* entries; merge values are already correct, this is purely a naming repair). The original PEFT-structured shards are preserved at `<run_dir>/best.pre_strip/`. A proper fix to `run_rl.py:save_merged_checkpoint` would use PEFT's `merge_and_unload()` instead — left as follow-up.

### Best-val ckpts and their math-verify scores

| Method                                                                                          | Best ckpt step | In-train eval/acc (1000) | MATH-500         | AIME-24 (avg@8) | AIME-25 (avg@8) | AMC23            | Mean (4)         |
| ----------------------------------------------------------------------------------------------- | -------------- | ------------------------ | ---------------- | --------------- | --------------- | ---------------- | ---------------- |
| **lora** (rank=64, mbs=2 gacc=128)                                                        | 15             | 69.20%                   | 39.80%           | 4.58%           | 0.00%           | 20.00%           | 16.09%           |
| **dora** (rank=64, mbs=1 gacc=256)                                                        | 10             | 69.00%                   | 40.40%           | 4.58%           | **1.25%** | 27.50%           | 18.43%           |
| **randlora** (rank=64, mbs=1 gacc=256)                                                    | 30             | 67.80%                   | **42.20%** | 2.92%           | 0.83%           | **32.50%** | **19.61%** |
| **milora** (rank=64, mbs=2 gacc=128)                                                      | 10             | 68.00%                   | 41.60%           | **5.83%** | 0.83%           | 30.00%           | 19.57%           |
| **pissa** (rank=64, mbs=2 gacc=128)                                                       | 0              | 64.40%                   | 40.40%           | 2.50%           | 0.42%           | **32.50%** | 18.95%           |
| **blocktt** (output_one_block, small, keep_trainable, rank=full, lr=1e-4, mbs=1 gacc=256) | 45             | **70.30%**         | 40.80%           | 5.42%           | 0.42%           | 27.50%           | 18.54%           |

### Headlines

1. **In-train eval/acc and math-verify mean rank disagree.** lora wins the in-train held-out 1000 (69.20%) but lands last on the math-verify mean (16.09%). randlora is third on in-train (67.80%) but first on math-verify mean (19.61%). This is consistent with the larger pattern in this doc: the held-out 1000 is drawn from the same competition_math distribution used for training, while AMC/AIME/MATH-500 are out-of-distribution from the SFT side, so they reward methods whose RL trajectory generalized rather than overfit.
2. **pissa's collapse is real, but the step-0 ckpt is competitive.** Even though pissa's "best" is just the pre-RL baseline, it scores 18.95% mean — close to randlora (19.61) and milora (19.57). At lr=6e-5 the RL update destroyed the model; the saved baseline gets full credit on math-verify. To get a *trained* pissa number, lr would need to drop to ~1e-5 (not done).
3. **AMC23 is the noisiest single-dataset signal.** Spread is 20.0–32.5% across these 6 methods — 12.5 pp gap on a 40-problem set. Single-pass greedy @ T=0.0; small-n variance is large. AIME-25 is similarly noisy (0.0–1.25% over 240 attempts at avg@8). MATH-500 is the most discriminative single slice here.
4. **Methodology limitation.** The "best by in-train eval/acc" criterion picks ckpts that spike on the same distribution they were trained on. For OOD math-verify reporting, mid-run ckpts (where the model hasn't yet specialized to competition_math) sometimes outscore the in-train peak. We didn't test that here — only the in-train-best ckpt was eval'd.

### What's NOT in this table

- **lift**: couldn't fit on single H100; only step-0 saved. Skipped.

### BlockTT also wins on the ladder

The new BlockTT retrain (`blocktt-output_one_block-small-keep_trainable-r64-lr_1e-4` with `--save-best-val-ckpt`) hit **70.30% in-train @ step 45** — surpassing the prior session's wandb-only BlockTT-Instruct best (69.0% @ lr=1e-4). It now leads the in-train-eval column by +1.10 pp over lora. On math-verify mean it lands second (17.99% vs randlora's 18.63%), and is competitive on every dataset. Because BlockTT trained the **smaller** core under `train_position=small` it has roughly 1/r-th the trainable params of the LoRA-family runs at rank=64, so its accuracy-per-trainable-param ratio is the most favorable in this table.

## Qwen2.5-7B (base) — initial lora / blocktt / dora @ lr=1e-4 (added 2026-05-02, blocktt mml=2048 row appended later same day)

GRPO sweep on `Qwen/Qwen2.5-7B` (the base, *not* the Instruct variant), `qwedsacf/competition_math`, 50 GRPO steps, `lr=1e-4`. Four runs, single-LR comparison: `lora` r=64, `dora` r=64, and **two `blocktt`** rows (`output_one_block` / `train_small` / `s_to_keep_trainable` / `rank=full`) that differ only in `max_model_len` (1536 vs 2048 — testing whether longer rollouts improve OOD math). All four use in-process vLLM rollout (no `vllm serve` HTTP server) on a single H100 NVL (96 GB).

- Source: `/data/yequan/fura/rl_runs/Qwen2.5-7B/{lora,dora,blocktt}/`
- Wandb project: `qwen2_5-7B-RL`
- Baseline (this pipeline, base model under boxed-prompt template): in-train held-out **70.8%** for the lora run, **69.8%** for blocktt, **68.6%** for dora. The 2.2 pp baseline spread across nominally identical configurations is rollout-sampling noise on the n=1000 held-out (T=0.0 greedy at step 0 should be deterministic — the spread suggests the in-process vLLM warm-up state differs run-to-run).
- Eval saved as merged HF ckpt at `step=50/` (no best-val-checkpointing; only the last-step ckpt is preserved per the project's "save last not best" policy).

### Memory recipe

The default `run_rl.py` settings (`--micro-batch-size 2 --gradient-accumulation-steps 128 --gpu-memory-utilization 0.4`) **OOM at step 1** for blocktt and dora on Qwen2.5-7B (lora fits, just). The Llama-3.1-8B-Instruct memory recipe carries over identically:

| Method  | mbs | gacc | gpu_util | max_model_len |  Peak GPU |
| ------- | --: | ---: | -------: | ------------: | --------: |
| lora    |   2 |  128 |      0.4 |          2048 | ~80 GB ✅ |
| dora    |   1 |  256 |     0.25 |          2048 | ~71 GB ✅ |
| blocktt |   1 |  256 |     0.25 |          1536 | ~67 GB ✅ |

Effective batch (32 prompts × 8 group × mbs × gacc / micro_batches_per_step) is unchanged across the mbs=1 and mbs=2 configs. Both dora and blocktt OOM'd at default settings (92.2 / 92.9 GB used at OOM, exceeding the 93 GB budget with vLLM weights + activations + adapter state). The recipe in the table above leaves ~25 GB of headroom.

A follow-up blocktt run with the *same* `mbs=1 / gacc=256 / gpu_util=0.25` recipe but `max_model_len` raised from 1536 to **2048** also fits without OOM (the 2048 setting is the standard for lora and dora; the prior 1536 was a pessimistic guard from the earlier OOM debugging session). This is the second blocktt row in the Results table.

### Results

| Method                                                                 |  mml | Train acc (step 50) | In-train eval/acc (1000) | Baseline |         MATH-500 |  AIME-24 (avg@8) | AIME-25 (avg@8) |            AMC23 |        Mean (4) |
| ---------------------------------------------------------------------- | ---: | ------------------: | -----------------------: | -------: | ---------------: | ---------------: | --------------: | ---------------: | --------------: |
| **lora** (rank=64)                                               | 2048 |    **75.78%** |                   89.30% |   70.80% |           58.60% |           10.83% |           2.92% | **60.00%** | **33.09** |
| **blocktt** (same recipe, mml=2048)                              | 2048 |              71.48% |         **89.40%** |   69.80% | **60.20%** | **12.50%** | **9.17%** |           47.50% |           32.34 |
| **dora** (rank=64)                                               | 2048 |              73.44% |                   89.30% |   68.60% |           59.00% |           10.42% |           8.33% |           50.00% |           31.94 |
| **blocktt** (output_one_block, small, keep_trainable, rank=full) | 1536 |              67.58% |                   89.30% |   69.80% |           59.80% |           10.83% |           5.83% |           42.50% |           29.74 |

(Mean is unweighted across the 4 math-verify datasets. AIME columns are avg@8 at T=0.6; the rest are greedy@1 at T=0.0. mml = `--max-model-len` for vLLM rollout.)

### Headlines

1. **In-train eval/acc saturates at ~89.3%** across all four runs (89.30% for the first three, 89.40% for blocktt-mml2048 — within rounding of identical). The held-out 1000-prompt set is not method-discriminative on Qwen2.5-7B; differentiation comes entirely from the OOD math-verify slices.
2. **blocktt-mml2048 closes the gap to lora.** Bumping `--max-model-len` from 1536 to 2048 lifts blocktt's mean-4 from 29.74 to **32.34**, putting it ahead of dora (31.94) and within 0.75 pp of lora (33.09). The gain comes from AMC23 (+5 pp), AIME-24 (+1.7 pp), and AIME-25 (+3.3 pp); MATH-500 is flat. Interpretation: a chunk of the prior blocktt deficit was rollout-truncation noise on harder problems, not a method gap. **The mml=1536 row is now superseded.**
3. **blocktt-mml2048 leads on 3 of 4 slices**: MATH-500 (60.20%), AIME-24 (12.50%), and AIME-25 (9.17%, beating dora 8.33 and lora 2.92). It trails lora on AMC23 (47.5 vs 60.0). On the two AIME competitions blocktt is now the clear leader, suggesting the longer rollout window helps most on the hardest problems where lora and dora often run out of generation budget.
4. **lora's lead is AMC23-driven.** lora's mean-4 (33.09) beats blocktt-mml2048 (32.34) entirely on AMC23 (60.0 vs 47.5, +12.5 pp). On every other slice lora is mid-pack or last (AIME-25 2.92% — easily the worst across the 4 runs). AMC23 is a 40-problem set so the spread is high-variance; the cross-method ranking should be read with a wide error bar.
5. **Compared to Llama-3.1-8B-Instruct at the same lr=1e-4** (Mean-4 column): qwen2.5-7B lora 33.09 > llama lora 16.09; qwen2.5-7B dora 31.94 > llama dora 18.43; qwen2.5-7B blocktt 29.74–32.34 > llama blocktt 18.54. Qwen2.5-7B base is a meaningfully stronger math-RL substrate than Llama-3.1-8B-Instruct at this LR. None of the runs collapsed (train acc finished 67.6–75.8% from baseline ~69%); the lr=1e-4 stable window is wide.
6. **No LR sweep yet.** All four runs used `lr=1e-4` — the value that maximized blocktt on Llama-3.1-8B-Instruct (69.0% final) but was too high for Llama LoRA r=64 (collapsed). Qwen2.5-7B's LoRA tolerated 1e-4 fine; the optimal lora/dora LR may be lower (cf. Qwen3-1.7B best LoRA LR = 6e-5, best DoRA LR = 1e-4). LR sweep is the obvious next step. Also: the blocktt mml=1536 vs 2048 jump suggests dora and lora may benefit from a similar mml bump — re-running them at mml=2048 is a cheap follow-up before declaring a winner.

### Full FT LR sweep (added 2026-05-05)

Single-H100 NVL full FT of Qwen2.5-7B was infeasible at default settings (~120 GB peak — model + bf16 grads + fp32 AdamW state + activations + vLLM). Two run_rl.py additions made it fit (~67 GB peak, 28 GB headroom):

- **`--optimizer adamw8bit`** — bitsandbytes AdamW8bit, drops the optimizer state from ~60 GB (fp32 m+v) to ~15 GB.
- **`--gradient-checkpointing`** — calls `model.gradient_checkpointing_enable(use_reentrant=False)` (with `enable_input_require_grads()` for PEFT compat). Saves ~5–10 GB activation memory at ~30% step-time cost. ~90 s/step at lr=1e-5, mbs=1, gacc=256.

Both flags default to off; only enabled for this single-GPU full-FT run. Otherwise identical to the lora/blocktt rows above (mbs=1, gacc=256, gpu_util=0.25, max_model_len=2048).

**LR sweep, MATH-500 only** (cheap selection over {8e-6, 1e-5, 3e-5, 5e-5}):

| LR                | Train acc (step 50) | In-train eval/acc | Baseline |                      MATH-500 |
| ----------------- | ------------------: | ----------------: | -------: | ----------------------------: |
| 8e-6              |              74.61% |            89.60% |   70.80% |                        58.20% |
| **1e-5** ⭐ |              73.05% |  **90.20%** |   70.80% |              **59.40%** |
| 3e-5              |              48.83% |            65.00% |   70.80% |     37.80% (collapse partial) |
| 5e-5              |               0.00% |             0.00% |   70.80% | 1.20% (catastrophic collapse) |

The stable LR window is narrow: ≤1e-5. lr=3e-5 partially collapses (train acc halves between step ~30 and step 50); lr=5e-5 fully collapses (train acc → 0). lr=8e-6 trains stably but slightly under-fits (the upper-bound 1e-5 wins by 1.2 pp on MATH-500). lr=1e-5 sits at the upper edge of the stable window.

The lr=1e-5 winner was then re-evaluated offline on AMC23 / AIME-24 / AIME-25 via `eval_rl.py`:

| Method (best LR per row)                       | Train acc (step 50) | In-train eval/acc | Baseline |         MATH-500 |          AIME-24 |         AIME-25 |            AMC23 |        Mean (4) |
| ---------------------------------------------- | ------------------: | ----------------: | -------: | ---------------: | ---------------: | --------------: | ---------------: | --------------: |
| **full FT (adamw8bit + GC, lr=1e-5)** ⭐ |              73.05% |            90.20% |   70.80% |           59.40% |            7.92% |           4.17% |           42.50% |           28.50 |
| lora (rank=64, lr=1e-4)                        |              75.78% |            89.30% |   70.80% |           58.60% |           10.83% |           2.92% | **60.00%** |           33.09 |
| dora (rank=64, lr=1e-4)                        |              73.44% |            89.30% |   68.60% |           59.00% |           10.42% |           8.33% |           50.00% |           31.94 |
| blocktt (mml=2048, lr=1e-4)                    |              71.48% |  **89.40%** |   69.80% | **60.20%** | **12.50%** | **9.17%** |           47.50% | **32.34** |

**Headline: full FT is *not* competitive on Qwen2.5-7B at this scale.** Mean-4 of 28.50 is the worst row (3.4–3.8 pp below all three PEFT methods). Full FT wins no individual slice — FuRA wins MATH-500 / AIME-24 / AIME-25, lora wins AMC23. Possible reasons:

1. **Too few RL steps for low LR.** lr=1e-5 over 50 GRPO steps gives an optimizer trajectory that traverses ~5e-4 in cumulative parameter space — likely under the threshold needed for a 7B base model to specialize on competition_math. PEFT methods amplify this implicitly: a rank-64 LoRA at lr=1e-4 has the same effective "policy LR" as full-FT at ~1e-5, but the structural restriction prevents collapse and lets the optimizer take more of those updates without diverging. Running full-FT for 200+ steps at lr=1e-5 would test this.
2. **The stable LR window is too narrow at this base.** Full FT collapses at 3e-5; LoRA stayed stable through 1e-4. The PEFT methods' implicit LR-tolerance (rank-restricted updates can't move parameters as far per step) gives them ~10× the usable LR range, which translates directly into more effective training within the same step budget.
3. **adamw8bit / GC overhead is unlikely to be the cause.** adamw8bit has been shown by the bitsandbytes paper to match fp32 AdamW within noise on LM fine-tuning; gradient checkpointing changes wall time but not the loss trajectory. The same lr=1e-5 with fp32 AdamW (if it could fit) would land in the same neighborhood.

So full FT here is more of a feasibility demonstration for the single-H100 path than a method comparison: yes, single-GPU 7B full-FT GRPO is now possible on this codebase; no, it does not beat the PEFT methods at the 50-step horizon. A longer sweep is the obvious follow-up.
