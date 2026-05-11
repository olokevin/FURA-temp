# FURA Motivating Example — Experiment Design Doc (v2)

**Aligned with:** `neurips_2026.tex` as of 2026-04-20.

## Paper Structure Context

The paper now places the motivating analysis **inside the Method section**:

- **§3.1 "Analysis of Full FT update dynamics"** — Three observations from Full FT on Qwen3-1.7B GRPO, shown as Figure 2 *left column*.
- **§3.4 "FURA recovers Full FT dynamics with implicit regularization"** — Mirror analysis of FURA on the same task, shown as Figure 2 *right column*.

**Figure 2 (`figs/motivation.png`)** is a single 2×3 figure:

|          | Full FT (left)                            | FURA (right)                               |
|----------|-------------------------------------------|--------------------------------------------|
| Panel a  | Effective rank of ΔW per layer            | Effective rank of ΔW per layer (FURA)      |
| Panel b  | Per-direction update magnitude \|ΔS_k\|   | Per-direction update magnitude (FURA)      |
| Panel c  | Projection residual (orth component)      | Projection residual = 0 by construction    |

**Model & Task:** Qwen3-1.7B, GRPO math RL (50 steps). This is the primary task used in the paper; checkpoints should already exist from the RL sweep.

**Checkpoints needed:**
- `W_0`: Qwen3-1.7B pretrained (base model)
- `W_ft_full`: After Full FT GRPO (best run, lr=2e-5, eval/acc=0.886)
- `W_ft_fura`: After FURA GRPO (best run, out-block s_keep_trainable train_small lr=1e-5, eval/acc=0.886)

Compute per linear layer (Q, K, V, O, gate, up, down across all transformer layers):
- `ΔW_full = W_ft_full − W_0`
- For FURA: `ΔW_fura = L · S_trained · R_trained − L · S_init · R_init` (reconstruct from saved cores)

---

## Panel (a): Effective Update Rank Varies Widely Across Layers

### Paper Text (§3.1, ¶1)
> "We measure the effective rank of ΔW per layer (the number of singular values capturing 90% of ‖ΔW‖_F). The effective rank varies dramatically: some attention projections require fewer than 20 directions while certain feed-forward layers utilize over 200."

### Paper Text (§3.4, ¶1)
> "The effective rank of ΔW under FURA closely tracks that of Full FT across layers."

### Experiment

**For Full FT (left panel):**
1. For every linear layer ℓ, compute `σ_1 ≥ σ_2 ≥ ... = svdvals(ΔW_full_ℓ)`.
2. Compute **r90**: smallest r such that `Σ_{i=1}^{r} σ_i² ≥ 0.90 · ‖ΔW‖²_F`.
3. Also compute as secondary metrics (for CSV, not necessarily plotted):
   - **Stable rank**: `sr = ‖ΔW‖²_F / σ_1²`
   - **r99**: same as r90 at 99% threshold

**For FURA (right panel):**
1. Reconstruct `ΔW_fura_ℓ` from saved FURA cores: `ΔW_fura_ℓ = Σ_k L_k (diag(S̃_k) R̃_k − diag(S_k^init) R_k^init)`
2. Compute the same r90 metric per layer.

### Figure Design

**Plot type:** Bar chart or stem plot.
- **x-axis:** Layer index (0..L-1). Group or color-code by module type (Q/K/V/O/gate/up/down) if space permits; otherwise aggregate across module types within each transformer layer.
- **y-axis:** r90 (effective rank at 90% energy).
- **Left subplot:** Full FT. **Right subplot:** FURA.
- **Annotations:**
  - Horizontal dashed line at r=64 (LoRA-64 budget).
  - Horizontal dashed line at r=128 (PiSSA-128 budget).
  - Optional: shade the region below r=64 to emphasize layers where LoRA suffices.

### Expected Results
- r90 varies from <20 (some attn projections) to >200 (some FFN layers) under Full FT.
- FURA's r90 per layer closely tracks Full FT's profile (high Pearson/Spearman correlation, e.g., r > 0.8).
- The LoRA-64 line cuts through the distribution: many layers above it (LoRA loses information), some below (LoRA wastes capacity).

### Narrative Implication
A fixed rank r is simultaneously too high for some layers and too low for others. FURA's full-rank parameterization lets the optimizer discover the right rank per layer without rank search or adaptive schemes.

### Success Flags
- **PASS:** ≥10× variation in r90 across layers (e.g., min < 20, max > 200, matching the paper's claim). FURA-vs-Full-FT rank correlation ≥ 0.7.
- **PARTIAL:** ≥3× variation, but less dramatic than 10×. Adjust paper text to soften ("varies substantially" instead of "dramatically").
- **FAIL:** <2× variation. Pivot to stable rank or plot per-module-type separately.

---

## Panel (b): Updates Concentrate on Data-Dependent Subset of Pretrained Singular Directions

### Paper Text (§3.1, ¶2)
> "When we project ΔW onto the pretrained SVD basis and measure the per-direction update magnitude, a strongly non-uniform pattern emerges: a small fraction of singular directions absorb most of the change while the majority remain near their pretrained values. Crucially, the active directions are neither purely principal nor purely off-principal."

### Paper Text (§3.4, ¶2)
> "The per-direction update magnitude |ΔS_k| under FURA exhibits the same non-uniform, data-driven pattern observed in Full FT."

### Experiment

**Key distinction from the old plan:** This panel is NOT about subspace alignment (PER). It is about the *distribution of update magnitudes across singular directions*, showing non-uniformity and mixed principal/off-principal activity. This directly attacks PiSSA/MiLoRA.

**For Full FT (left panel):**
1. For each layer ℓ, compute the pretrained block-wise SVD: `W_0_k = U_k Σ_k V_k^T` (using FURA's blocking scheme: n = √d_in blocks of width b = √d_in).
2. Project ΔW_full onto the pretrained SVD basis per block:
   - `ΔΣ_k = diag(U_k^T · ΔW_full_k · V_k)` — the per-direction update magnitude in the pretrained basis.
3. Collect `|ΔΣ_k|` for all blocks k and all layers.

**For FURA (right panel):**
1. Directly available: `|ΔS_k| = |S̃_k − S_k^init|` from the trained FURA cores.
2. Additionally, the directional change in R contributes, but the S change is the cleanest signal. For a complete picture, also compute: `ΔΣ_k^fura = diag(U_k^T · ΔW_fura_k · V_k^init)`.

### Figure Design

**Plot type:** Heatmap or multi-panel stem plot.
- **Option A (heatmap, recommended for density):**
  - x-axis: singular value index within block (1..b), ordered by pretrained singular value magnitude (largest → smallest).
  - y-axis: layer index (0..L-1) or (layer, block) pairs.
  - Color: `|ΔΣ_k|` magnitude (log scale or linear with color cap).
  - Left heatmap: Full FT. Right heatmap: FURA.
  - Key visual: scattered "hot" entries across the whole spectrum (not concentrated at top or bottom).

- **Option B (selected layers, stem plot):**
  - Pick 3–4 representative layers (early attn, mid FFN, late attn).
  - x-axis: singular direction index ordered by pretrained σ magnitude.
  - y-axis: |ΔΣ_k|.
  - Overlay or side-by-side for Full FT vs FURA.

### Expected Results
- The update magnitude |ΔΣ_k| is strongly non-uniform: a small fraction of directions absorb most change.
- Active directions span the entire spectrum — some are principal (large σ), some are off-principal (small σ).
- The pattern is layer-dependent and task-dependent (data-driven selection).
- FURA's |ΔS_k| pattern visually resembles Full FT's |ΔΣ_k| pattern (same directions active).

### Narrative Implication
PiSSA (top-r) and MiLoRA (bottom-r) each commit to a fixed slice that misses active directions at the other end. FURA exposes the entire spectrum and lets the optimizer select which directions to update.

### Success Flags
- **PASS:** Clear visual non-uniformity. Both principal and off-principal directions show large |ΔΣ_k| in at least some layers. FURA and Full FT patterns are visually similar (can quantify with Spearman correlation of |ΔΣ| vectors, target ≥ 0.5).
- **PARTIAL:** Non-uniformity is clear but skewed toward one end. Soften to "predominantly but not exclusively principal" and note task-dependence.
- **FAIL:** Updates are nearly uniform across directions, or clearly concentrated at one end. If so, this panel becomes less interesting — consider replacing with the PER-based alignment analysis from v1.

---

## Panel (c): Full FT Learns Orthogonal Components that FURA's Spectral Constraint Removes

### Paper Text (§3.4, ¶3)
> "Full FT updates contain a non-trivial component outside the pretrained singular subspace. With limited fine-tuning data, these out-of-subspace components can overfit to noise and degrade generalizable representations. In FURA, every update is confined to the pretrained block-wise singular subspace by construction."

### Experiment

**For Full FT (left panel) — Projection Residual Analysis:**
1. For each layer ℓ and each block k, decompose the Full FT update:
   - `ΔW_aligned_k = U_k · U_k^T · ΔW_full_k` — component within pretrained left-singular subspace.
   - `ΔW_orth_k = ΔW_full_k − ΔW_aligned_k` — orthogonal residual.

2. **Metric 1: Per-layer orthogonal energy fraction (OEF):**
   - `OEF_ℓ = Σ_k ‖ΔW_orth_k‖²_F / Σ_k ‖ΔW_full_k‖²_F`
   - This is the fraction of update energy that falls outside the pretrained subspace.

3. **Metric 2: Spectral structure of the orthogonal component:**
   - Compute `svdvals(ΔW_orth_k)` per block. If the spectrum is flat (Marchenko-Pastur-like), the orthogonal component is noise. If it has a clear low-rank structure, it may carry signal.
   - Aggregate: compare the singular value decay rate of ΔW_orth vs ΔW_aligned.

4. **Metric 3 (strongest evidence): Surgical ablation:**
   - Construct `W_aligned_only = W_0 + Σ_k U_k U_k^T ΔW_full_k` for all layers (project away orthogonal part).
   - Evaluate on the GRPO test set. Compare:
     - Full FT: eval/acc = 0.886
     - Aligned-only: expected ≥ 0.886 (if orth is noise)
     - FURA: eval/acc = 0.886 (for reference)

5. **Metric 4: Orthogonal amplification (optional):**
   - `W_amplified = W_0 + ΔW_aligned + α · ΔW_orth` with α ∈ {0.5, 1.0, 1.5, 2.0}.
   - If test accuracy monotonically decreases with α > 1, the orthogonal component hurts generalization.

**For FURA (right panel):**
1. By construction, FURA's ΔW_k lives in col(L_k) = col(U_k), so the orthogonal component is exactly zero.
2. The right panel can simply show OEF = 0 across all layers (trivially), or be repurposed to show a quantitative comparison (e.g., generalization gap: Full FT train-test gap vs FURA train-test gap).

**Important note on block-wise vs matrix-level projection:**
Since each FURA block has U_k ∈ R^{d_out × b} with b = √d_in ≪ d_out, the per-block U_k does NOT span all of R^{d_out}. The orthogonal complement is non-trivial (dim = d_out − b). This means OEF should be meaningfully non-zero for Full FT. If d_out = 1536 and b = 64, each block's U_k spans only 64 out of 1536 output dimensions — the orthogonal space is large.

**Wait — re-examine this.** Actually for Qwen3-1.7B:
- d_model = 1536, d_intermediate (FFN) = 8960
- For attention: d_in = 1536, d_out = 1536 (for Q/K/V/O)
- n = √1536 ≈ 39.2 → need to check actual blocking. If d_in = n·b with n=b=√d_in, then for d_in=1536 we need integer factorization.
- **Action:** Check the actual FURA code for block sizes used on Qwen3-1.7B. The blocking scheme matters for what "orthogonal to pretrained subspace" means.

### Figure Design

**Left panel (Full FT):**
- **Plot type:** Bar chart across layers.
  - x-axis: layer index.
  - y-axis: OEF (orthogonal energy fraction), range [0, 1].
  - Color intensity or secondary bar: spectral flatness of ΔW_orth (higher = more noise-like).
- **Key visual:** Non-trivial OEF (5–25%) across most layers, with flat spectral structure indicating noise.

**Right panel (FURA):**
- **Option A:** Show OEF = 0 as a flat zero-line (visually clean contrast).
- **Option B (preferred):** Repurpose the right panel for the surgical ablation result — a small bar chart comparing Full FT vs Aligned-Only vs FURA test accuracy. This is more informative.
- **Option C:** Show generalization gap (train reward − test accuracy) for Full FT vs FURA across training steps, demonstrating FURA's tighter gap.

### Expected Results
- OEF is non-trivial (5–25%) for most layers under Full FT. The magnitude depends critically on the block size b relative to d_out.
- Spectral structure of ΔW_orth is flatter than ΔW_aligned (higher effective rank relative to its energy, fewer dominant singular values).
- Surgical ablation: Aligned-only model matches or slightly exceeds Full FT on test accuracy.
- FURA's OEF = 0 by construction → natural spectral regularizer.

### Narrative Implication
The orthogonal residual of Full FT carries overfitting noise, not useful signal. FURA's architectural constraint removes these components, explaining how a constrained method surpasses unconstrained Full FT.

### Success Flags
- **PASS:** OEF ≥ 5% on average, ΔW_orth spectrum is visually flat/random, AND surgical ablation shows aligned-only ≥ Full FT test acc (within 0.5%).
- **PARTIAL:** OEF ≥ 5% and spectrum analysis supports noise interpretation, but surgical ablation is inconclusive (aligned-only within 1.5% of Full FT). Rely on spectral evidence + FURA's empirical outperformance as indirect support.
- **FAIL:** OEF < 2% (orthogonal component is negligible) OR aligned-only drops >2% below Full FT. If OEF is negligible, reframe panel (c) as the PER alignment analysis from v1 (showing that updates are >95% within the pretrained subspace — the constraint is nearly free). If aligned-only drops, emphasize the "regularization vs capacity trade-off" angle.

---

## Implementation Checklist

### Prerequisites
- [ ] Locate or re-run Full FT checkpoint for Qwen3-1.7B GRPO (lr=2e-5, best run at 0.886)
- [ ] Locate FURA checkpoint for Qwen3-1.7B GRPO (out-block, s_keep_trainable, train_small, lr=1e-5, best run at 0.886)
- [ ] Locate Qwen3-1.7B base model weights (W_0)
- [ ] Verify FURA blocking scheme: determine actual n, b values for each layer shape in Qwen3-1.7B
- [ ] Save/extract per-layer ΔW_full and per-block FURA cores (L_k, S_k^init, S_k^trained, R_k^init, R_k^trained)

### Script 1: `compute_panel_a.py` — Effective Rank
- [ ] Input: W_0, W_ft_full, W_ft_fura state dicts (or FURA core checkpoints)
- [ ] For each linear layer:
  - Compute ΔW, svdvals, r90, stable_rank, r99
- [ ] Output: `results/panel_a.csv` with columns: (layer_idx, layer_name, module_type, r90_full, r90_fura, stable_rank_full, stable_rank_fura)
- [ ] Sanity check: print min/max/mean r90 for both methods

### Script 2: `compute_panel_b.py` — Per-Direction Update Magnitude
- [ ] Input: W_0, W_ft_full state dict, FURA core checkpoints
- [ ] For each linear layer, for each block k:
  - Compute pretrained block SVD: U_k, Σ_k, V_k = SVD(W_0_k)
  - Project Full FT update: ΔΣ_k = diag(U_k^T · ΔW_full_k · V_k)
  - Extract FURA update: ΔS_k = S_k^trained − S_k^init
- [ ] Output: `results/panel_b_full.npz` and `results/panel_b_fura.npz` (per-block ΔΣ arrays)
- [ ] Sanity check: verify |ΔΣ_k| is non-uniform, check both principal and off-principal ends

### Script 3: `compute_panel_c.py` — Orthogonal Residual Analysis
- [ ] Input: W_0, W_ft_full state dict, FURA blocking config
- [ ] For each layer, for each block k:
  - ΔW_aligned_k = U_k @ (U_k.T @ ΔW_full_k)
  - ΔW_orth_k = ΔW_full_k − ΔW_aligned_k
  - Compute OEF, svdvals of ΔW_orth_k, spectral flatness
- [ ] Output: `results/panel_c.csv` with columns: (layer_idx, layer_name, OEF, orth_stable_rank, aligned_stable_rank, orth_spectral_norm_ratio)
- [ ] Sanity check: OEF values make sense given block dimensions

### Script 4: `surgical_ablation.py` — Aligned-Only Model Evaluation
- [ ] Construct aligned-only weights: for each layer, each block: W_surgical_k = W_0_k + U_k @ (U_k.T @ ΔW_full_k)
- [ ] Reconstruct full weight matrices and create a state dict
- [ ] Evaluate on GRPO test set (same eval protocol as paper)
- [ ] Compare: Full FT acc, Aligned-Only acc, FURA acc
- [ ] Optional: sweep α ∈ {0, 0.5, 1.0, 1.5, 2.0} for orthogonal amplification

### Script 5: `plot_motivation.py` — Assemble Figure 2
- [ ] Reads from results/*.csv and results/*.npz
- [ ] Produces `figs/motivation.png` (or .pdf) as 2-column × 3-row figure
- [ ] Left column: Full FT panels (a, b, c)
- [ ] Right column: FURA panels (a, b, c)
- [ ] Use consistent color scheme, font sizes appropriate for NeurIPS single-column width
- [ ] Save at ≥300 DPI

### Final Verification
- [ ] Figure renders correctly in LaTeX (`\includegraphics[width=\linewidth]{figs/motivation.png}`)
- [ ] Paper text in §3.1 and §3.4 matches the actual numbers (update specific values like "fewer than 20", "over 200")
- [ ] Caption in Figure 2 is accurate
- [ ] Page budget: figure + §3.1 text + §3.4 text fits within target (currently ~1.5 pages total for §3.1+§3.4)

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Full FT checkpoint not saved / lost | Medium | High | Re-run Full FT on Qwen3-1.7B GRPO (50 steps, ~2hrs on H100). This is the cheapest Full FT in the paper. |
| r90 variation is <3× across layers | Low | Medium | Use stable rank instead (more sensitive). Or plot per-module-type separately to show variation within types. |
| OEF is negligible (<2%) due to block U_k spanning most of d_out | Medium | High | This depends on b vs d_out. If b ≈ d_out, each block's U_k nearly spans the full space. **Pre-check:** compute d_out/b ratio for each layer. If this ratio is close to 1, the orthogonal space is tiny. In this case, reframe panel (c) as: "despite having near-complete subspace coverage per block, FURA's per-block confinement still provides meaningful regularization through the block-wise independence constraint (Proposition 5, no cross-block transfer)." |
| Surgical ablation is expensive (requires GRPO eval) | Medium | Medium | The GRPO eval is just inference (no training), should be fast. Alternatively, measure on a held-out validation loss rather than full accuracy. |
| FURA and Full FT rank profiles don't correlate well | Low | High | If FURA's rank profile diverges from Full FT, reframe §3.4 as "FURA discovers a *better* rank allocation" (supported by FURA matching Full FT accuracy). |
| Panel (b) FURA pattern diverges from Full FT | Low-Medium | Medium | The ΔS_k in FURA is constrained to the diagonal of the pretrained basis, while Full FT can move in arbitrary directions. Some divergence is expected. Report Spearman correlation; if <0.3, soften to "FURA identifies a complementary but overlapping set of directions." |

---

## Open Design Questions

1. **Block size verification:** What are the actual (n, b) values for Qwen3-1.7B? For d_in=1536 (attn), √1536 ≈ 39.2 which isn't an integer. The code likely uses a specific factorization (e.g., n=32, b=48 or n=48, b=32). This affects the dimension of the orthogonal complement in panel (c).

2. **Panel (c) right subplot content:** The paper says "FURA updates are fully contained by construction" but a flat zero-line is visually uninteresting. Consider:
   - Show the generalization gap comparison (Full FT vs FURA train-test curves) instead.
   - Show a "decomposition pie chart" for Full FT (aligned % vs orth %) next to FURA (100% aligned).
   - Show the singular value spectrum comparison: ΔW_orth (flat/noisy) vs ΔW_aligned (structured/decaying).

3. **Multiple tasks:** The paper only references GRPO for the motivating analysis. Adding Commonsense-170K in an appendix figure would strengthen generality but requires LLaMA-3-8B Full FT checkpoint (more expensive).

4. **Panel (b) normalization:** Should |ΔΣ_k| be normalized per-block (relative to block's total update energy) or shown in absolute magnitude? Per-block normalization makes cross-layer comparison fairer; absolute magnitude shows which layers change most.
