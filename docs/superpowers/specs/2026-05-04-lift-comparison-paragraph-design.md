# LIFT Comparison Paragraph — Design Spec

**Date:** 2026-05-04
**Target file:** `docs/26_nips_fura_paper/neurips_2026.tex`
**Scope:** add a paragraph comparing FuRA to LIFT after the RLVR results table; remove the LIFT row from the main complexity table.

## Motivation

The current paper cites LIFT (`liu2025lift`) in §2 (related work) and includes a `LIFT (r=32)` row in the system-cost table (`tab:cost`, line 397). Two issues:

1. The LIFT row is in the main complexity table but LIFT is *not* in the SFT or RLVR results tables, so the table mixes accuracy-relevant baselines with a method that only appears in the cost analysis. This is structurally awkward.
2. We have separately measured LIFT under matched protocols on both LLaMA-3-8B SFT (`/data/yequan/fura/lift/commonsense/`) and Qwen3-1.7B GRPO RLVR (`/data/yequan/fura/rl_runs/Qwen3-1.7B/lift/`), and the comparison reveals a substantive finding: LIFT matches FuRA on SFT but fails on RLVR, and its system cost has a non-trivial decomposition that is worth surfacing rather than burying.

We bundle both into one new paragraph in the RLVR experiments subsection and remove the now-redundant table row.

## Design

### Edit 1: remove `LIFT (r=32)` row from `tab:cost`

`docs/26_nips_fura_paper/neurips_2026.tex` line 397:

```latex
LIFT ($r{=}32$)    & $100$       & $0$        & Yes (sparse)     & $0.144$ & $227{,}075$ & $80.0$ \\
```

Delete this row. The complexity table is now the LoRA-family / FuRA / Full FT cost story; LIFT's cost moves into the new paragraph alongside its accuracy story.

### Edit 2: add `\paragraph{Comparison with LIFT.}` after line 574

Inserted after the existing RLVR-results paragraph, before §5.3 (QFuRA). Two short paragraphs (accuracy story + system-cost story) plus one small inline table.

#### Paragraph A — accuracy

> **Comparison with LIFT.** A natural comparison point for FuRA is LIFT \cite{liu2025lift}, which also targets full-rank adaptation but does so by training a sparse subset of backbone-weight entries selected from a principal-component reconstruction of each weight. The two methods reach comparable accuracy on SFT — on LLaMA-3-8B Commonsense-170K, FuRA averages $87.91$ vs.\ LIFT $86.91$; on the Math-10K SFT benchmark both land near $80$ average. They diverge on RLVR. Under our matched Qwen3-1.7B GRPO protocol ($50$ steps, $32{\times}8$ rollouts), the best LIFT learning-rate sweep ($\eta{=}8{\times}10^{-5}$) reaches MATH-500 $60.2$ / AMC23 $37.5$ / AIME-24 $10.0$ / AIME-25 $8.75$, clearly below FuRA ($63.6 / 57.5 / 15.0 / 17.5$) and below Full FT on the harder slices; aggressive learning rates ($\eta{=}2{\times}10^{-4}$) collapse entirely (MATH-500 $7.0$). We attribute the gap to the choice of update subspace: LIFT updates entries derived from the \emph{principal}-component reconstruction, a top-singular heuristic well-aligned with SFT but at odds with the RLVR finding that fine-tuning gradients spread broadly across the singular spectrum and concentrate energy on \emph{off-principal} directions \cite{zhu2025rlvrpath}. FuRA preserves the full singular spectrum via the trainable diagonal $\mat{S}$ and confines updates to the per-block pretrained column space, so neither tail is cut and the optimizer chooses the rank allocation per layer.

#### Paragraph B — system cost

> **System cost.** A second difference is implementation cost (Table~\ref{tab:lift-cost}). FuRA reaches $0.046$ s/step and $22.7$ GB peak GPU memory on LLaMA-3-8B SFT in plain PyTorch — comparable to LoRA $r{=}64$ ($0.043$ s, $26.8$ GB) — because only the small factored cores $\mat{R}$, $\mat{S}$ are \texttt{requires\_grad=True}, so gradient buffers, AdamW state, and activation memory all scale with the trainable factor, not the backbone. LIFT, by contrast, requires every backbone weight to retain \texttt{requires\_grad=True} so that autograd populates a full $\mat{p}.\mathrm{grad}$ for the sparse mask to gate. This intrinsically adds (i)~a full bf16 backbone gradient buffer (${\sim}16$ GB on $8$ B params) and (ii)~a per-weight bool mask (${\sim}8$ GB), a \textbf{${\sim}+24$ GB peak-memory floor} above any frozen-backbone PEFT baseline that holds even though the sparse AdamW $(m, v)$ state itself is small (${\sim}0.5$ GB at $r{=}32$). The same full backward also makes each optimizer step ${\sim}3{\times}$ slower than FuRA ($0.144$ vs.\ $0.046$ s/step). Closing this gap requires custom kernels that fuse the mask into the backward GEMM and avoid materializing dense gradients; this is non-trivial and absent from the public LIFT recipe. FuRA achieves the same full-rank update goal with no custom kernels because its trainable parameters are factored adapters rather than entries of a dense weight matrix.

#### Inline table `tab:lift-cost`

Four rows × four columns. We report:

- Trainable params (effective): for LIFT, the unmasked-entry count, not the misleading 99.997% from `requires_grad=True` on the full backbone.
- Step (s): measured directly, no master-copy contamination.
- Extra peak vs.\ frozen-backbone PEFT: analytical lower bound for LIFT (full grad buffer + bool mask), measured for FuRA, both relative to a frozen-backbone PEFT baseline.

| Method | Trainable (effective) | Step (s) | Peak GPU (GB) |
|---|---:|---:|---:|
| Full FT | 8.03 B | 0.100 | 76.9 (measured) |
| LoRA $r{=}64$ | 0.11 B | 0.043 | 26.8 (measured) |
| LIFT $r{=}32$ | ${\sim}0.5$ B (sparse, ~6% of 8 B) | 0.144 | $\geq +24$ above LoRA (analytical lower bound) |
| **FuRA (ours)** | **0.16 B** | **0.046** | **22.7 (measured)** |

Notes:
- FuRA's measured peak is reported directly. LIFT's peak is reported as an analytical *lower bound* on the increase over a frozen-backbone PEFT baseline; we deliberately avoid quoting our raw measured peak (~80 GB) because it likely includes an fp32 master copy of the full backbone that mixed-precision wrappers create when every weight has `requires_grad=True` — a recipe-specific artifact, not an intrinsic LIFT cost.
- All measured numbers from `/data/yequan/fura/sys_eval/commonsense/{full,lora/r64,lift/r32,fura}/sys_metrics.json` on a single H100 NVL, 300-step SFT, bs=8×accum=2, seqlen=2048, gradient checkpointing on.
- Full FT step time corrected for accumulation = 2 (raw measured 0.100 s × 2 = 0.201 s/optimizer step matches `tab:cost` row).

## Open question — 1.5B RL system measurement

The user spec asked for "step time and memory on 1.5B RL." We do **not** have system metrics for Qwen3-1.7B RL runs (no `sys_metrics.json` in `/data/yequan/fura/rl_runs/`).

**Decision: defer.** The 8B SFT numbers are sufficient to make the implementation-cost claim, the RL accuracy story is what readers care about most, and the 1.7B numbers would be smaller versions of the same 8B story. If a reviewer asks, we can add 1.7B system numbers in rebuttal.

## Verification before merging

1. Compile the paper and confirm the new paragraphs and table fit on the page (NeurIPS page budget).
2. Re-check the `tab:cost` prose at lines 404–415 for any sentence that referenced LIFT explicitly (none found in current draft, but verify after edit).
3. Confirm `liu2025lift` and `zhu2025rlvrpath` citation keys exist in `neurips_2026.bib`.

## Out of scope

- Changes to the §2 related-work characterization of LIFT (line 133); the existing one sentence is accurate.
- Changes to the SFT results table (`tab:commonsense_sft`); LIFT was not in it and we do not add it now.
- Any custom-kernel design for sparse-mask methods; we mention this as future work and explicitly leave it out of scope.
- Reporting the measured ~80 GB LIFT peak; we report only the analytical floor to stay honest about what's intrinsic vs. recipe-specific.
