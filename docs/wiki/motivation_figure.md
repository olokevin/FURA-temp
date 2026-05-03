# Motivation figure (Figure 2): how the 6-panel `motivation.png` is plotted

`docs/exp_results/figs/motivation.png` (and `.pdf`) is the 2×3 motivating
figure that opens the FURA method section. This page documents how it is
generated so it can be re-rendered for new checkpoints, new tasks, or paper
revisions without re-deriving the layout.

The single entry point is **`analysis/plot_motivation_figure.py`**. It reads
three checkpoints + one base model, computes the four spectral quantities
needed by the figure, scrapes two log files for the learning curve, and
assembles the 6-panel figure with `matplotlib.gridspec`.

---

## 1. Panel layout

```
        ┌─────────────────────┬─────────────────────┬──────────────────────┐
row 0   │ (a) Full FT         │ (b) Full FT         │ (c) Math RL eval     │
        │     eff. rank ΔW    │     |V'-V₀| heatmap │     accuracy         │
        │     vs layer idx    │     (one layer)     │     Full FT vs SVD   │
        ├─────────────────────┼─────────────────────┼──────────────────────┤
row 1   │ (d) FuRA            │ (e) FuRA            │ (f) Source-domain    │
        │     eff. rank ΔW    │     |R'-R₀| heatmap │     preservation     │
        │     vs layer idx    │     (one layer)     │     Base/FuRA/LoRA   │
        └─────────────────────┴─────────────────────┴──────────────────────┘
```

Two curves on each rank panel: `q_proj` (blue, `Attn Q`) and `up_proj`
(orange, `MLP Up`). Panels (a) and (d) share the same y-limit so the FuRA-vs-Full
rank shrinkage is visually comparable.

Figure size: `figsize=(14, 9)` with `width_ratios=[1.2, 1.2, 1]`,
`height_ratios=[1, 1.5]`, `hspace=0.40`, `wspace=0.40`.
DPI: `savefig.dpi=300`. Output goes to both `.pdf` (vector) and `.png`
(rasterized at 300 DPI).

---

## 2. What each panel computes

| Panel | Function | Quantity | Inputs |
|-------|----------|----------|--------|
| (a) | `compute_rank_curves` → `plot_rank_curves` | per-layer `r₉₀(ΔW)` for `q_proj`, `up_proj` | base model + Full FT checkpoint |
| (b) | `compute_V_update_heatmap` → `plot_V_heatmap` | `\|V' − V₀\|` from thin SVD of `W₀` and `W'` (sign-aligned per row) | base model + Full FT checkpoint, single layer |
| (c) | `extract_rl_curve` / `extract_sft_curve` → `plot_learning_curves` | step → eval-accuracy (RL) or step → train-loss (SFT) | run dirs with `output.log` (RL) or `training.log` (SFT) |
| (d) | same as (a) | per-layer `r₉₀(ΔW)` | base model + FuRA checkpoint |
| (e) | `compute_R_update_heatmap` → `plot_R_heatmap` | `\|R' − R₀\|` after BlockTT decomp of `W₀` and `W'` (no model change — the layer is decomposed offline via `BTTLayer.init_from_linear_weight`) | base model + FuRA checkpoint, single layer |
| (f) | `plot_forgetting_bar` | scalar avg-commonsense accuracy per method | `--forgetting-json` (optional) or hard-coded fallback `{Base: 37.6, FuRA: 45.5, LoRA: 35.6}` |

### Effective-rank definition

```python
def compute_effective_rank(delta_W, threshold=0.90):
    s = torch.linalg.svdvals(delta_W.float())
    cumulative = torch.cumsum(s ** 2, dim=0)
    k = int((cumulative / cumulative[-1] < threshold).sum().item()) + 1
    return min(k, len(s))
```

i.e. the smallest `k` such that the top-`k` squared singular values capture
≥90 % of `‖ΔW‖²_F`. Threshold is fixed at 0.90 in the panel (a)/(d) function;
not exposed as a CLI flag.

### V-heatmap (panel b)

For every layer:

1. Thin SVD of `W₀` and `W'` (`full_matrices=False`).
2. Sign-align row-by-row: `Vt₁_aligned[i] = sign(⟨Vt₀[i], Vt₁[i]⟩) · Vt₁[i]`
   (zero is mapped to `+1`).
3. Store `|Vt₁_aligned − Vt₀|` as a 2-D array of shape `(rank, d_in)`.
4. The heatmap shows `log₁₀(|V' − V₀| + 1e-10)` for one selected layer.

Sign alignment is necessary because `V` is only defined up to a column sign.
The near-degenerate-σ caveat from
[`projection_energy.md`](projection_energy.md#sign-ambiguity) applies — for a
cleaner subspace metric use principal angles instead.

### R-heatmap (panel e)

For every layer:

1. Build a fresh `BTTLayer(in_features, out_features, rank="full",
   decomp_mode="output_one_block")` (no parameters trained — just used as a
   decomposition primitive).
2. `init_from_linear_weight(W, s_merged_to="trainable")` to obtain the right
   core `R` of shape `(n, b, m·rank)`. Singular values are absorbed into the
   trainable factor (matches the project default
   `--s_merged_to keep_trainable`).
3. Decompose both `W₀` and `W'` to get `R₀` and `R'`, store
   `|R' − R₀|` of shape `(n, b, m·rank)`.
4. Plot is `log₁₀(|R' − R₀| + 1e-10)` flattened to `(n·b, m·rank)`,
   `aspect="equal"`, with white horizontal lines at every block boundary
   (`y = i·b − 0.5` for `i = 1..n-1`) so the block structure is visible.
5. y-tick labels mark block centers (one tick per `n//8` blocks).

Note: `decomp_mode` is overridable via `--decomp-mode`, but it should match
the FuRA training run's `decomp_mode` — otherwise the cores aren't comparable.

### Learning-curve scrapers

- **RL accuracy (`extract_rl_curve`):** finds the first `output.log` under
  `run_dir` (preferring `latest-run` paths), then parses lines matching
  `step=(\d+),\s*correct:\s*(\d+)\s*/\s*(\d+)` with the regex baked in.
- **SFT loss (`extract_sft_curve`):** reads `<run_dir>/training.log`, parses
  lines matching `Step:\s*(\d+),.*Loss:\s*([\d.]+)`.

The CLI auto-detects which scraper to use (`--curve-mode auto`): it tries
`extract_rl_curve` on the checkpoint dir and on its parent before falling
back to `extract_sft_curve`. Force a mode with `--curve-mode accuracy|loss`.

### Source-domain preservation (panel f)

Default values are hard-coded inside the script (Llama-3-8B math SFT):
`{Base: 37.6, FuRA: 45.5, LoRA: 35.6}`. Override by passing
`--forgetting-json path/to/file.json` containing a flat
`{"<method>": <number>}` dict.

The bar colors are mapped from a small dictionary; methods named exactly
`Base`, `Full FT`, `FuRA`, `LoRA`, `SVD FT` get the project's standard
colors, anything else falls back to grey `#888888`.

---

## 3. Reproduce (current `motivation.png`)

The published figure on `main` shows Qwen3-1.7B GRPO math RL (50 steps).

```bash
CUDA_VISIBLE_DEVICES=0 uv run python analysis/plot_motivation_figure.py \
    --base-model Qwen/Qwen3-1.7B \
    --full-ft-ckpt /data/yequan/fura/rl_runs/full/full-adamw-lr_2e-5-.../step=50 \
    --fura-ckpt /data/yequan/fura/rl_runs/blocktt/blocktt-adamw-lr_8e-5-.../step=50 \
    --svd-run   /data/yequan/fura/rl_runs/svd/svd-adamw-lr_1e-5-... \
    --output    docs/exp_results/figs/motivation.pdf
```

Run-level paths (vs `/step=50` checkpoint paths) are needed for the
RL-accuracy scrape, since `output.log` lives at the run level. The CLI
auto-walks one level up if it can't find a log next to the checkpoint.

For the Math SFT variant (Llama-3-8B):

```bash
CUDA_VISIBLE_DEVICES=0 uv run python analysis/plot_motivation_figure.py \
    --base-model meta-llama/Meta-Llama-3-8B \
    --full-ft-ckpt /data/yequan/fura/lift/math/.../full-lr_5e-5-seed_43 \
    --fura-ckpt    /data/yequan/fura/lift/math/.../blocktt-...-seed_43 \
    --svd-ckpt     /data/yequan/fura/lift/math/.../svd-...-seed_43 \
    --output       docs/26_nips_fura_paper/figs/motivation_math.pdf
```

---

## 4. CLI flags

```
--base-model            HF id or local dir with *.safetensors      (required)
--full-ft-ckpt          ckpt dir with *.safetensors|pytorch_model.bin (required)
--fura-ckpt             ckpt dir (BlockTT-decomposable weights)    (required)
--svd-ckpt              SVD-FT checkpoint dir (for SFT curve mode)
--svd-run               SVD-FT run dir (for RL curve mode — has output.log)
--full-ft-run           override run dir for Full FT curve scrape
--fura-run              override run dir for FuRA curve scrape
--heatmap-layer  INT    layer for panels (b) & (e); default = num_layers // 2
--heatmap-module STR    default q_proj
--decomp-mode    STR    BlockTT mode for panel (e); default output_one_block
--curve-mode     STR    auto | accuracy | loss
--forgetting-json PATH  override panel (f) values
--device         STR    default cuda:0
--output         PATH   defaults to docs/exp_results/figs/motivation.pdf
                        (a sibling .png is always emitted alongside)
```

---

## 5. Inputs the script expects on disk

1. **Base model**: a HF id (auto-`snapshot_download`) or a local dir with
   `*.safetensors`. Used as the source for `W₀`.
2. **Two checkpoints** (Full FT + FuRA) as either:
   - directory of `*.safetensors` (preferred), or
   - directory containing `pytorch_model.bin`.
   Their state-dict keys must follow the HF convention
   `model.layers.<L>.self_attn.<q|k|v|o>_proj.weight` and
   `model.layers.<L>.mlp.<gate|up|down>_proj.weight`.
3. **Run dirs** for the learning curve:
   - RL: a dir whose `rglob("output.log")` returns a wandb-style log with
     `step=N, correct: K / M` lines.
   - SFT: a dir containing `training.log` with `Step: N, Loss: F` lines.

The number of layers is detected automatically from the base model
(`detect_num_layers` scans state-dict keys).

---

## 6. Styling conventions reused across the codebase

`plot_motivation_figure.py` is also the *styling source of truth* for other
analysis figures. Specifically `analysis/spectral_projection.py` and the
in-training projection-energy recorder import:

- The rcParams block (serif, 8 pt body, 9 pt axis labels, 0.6 pt axes lines)
- The color constants
  `COLOR_FULL = "#2171B5"`, `COLOR_SVD = "#6A51A3"`,
  `COLOR_FURA = "#D94801"`, `COLOR_LORA = "#969696"`,
  `COLOR_Q = "#2171B5"`, `COLOR_UP = "#D94801"`
- Loaders: `load_safetensors_index`, `load_checkpoint`, `get_weight`,
  `resolve_base_model`, `detect_num_layers`

When adding new analysis figures, import from `plot_motivation_figure.py`
rather than re-defining colors / loaders, so the paper's figures stay
visually consistent.

---

## 7. Common pitfalls

- **`decomp_mode` mismatch in panel (e):** The default
  `output_one_block` matches the project-wide FuRA default
  ([`fura_qfura_defaults`](../../.claude/projects/-home-yequan-Project-lora-lora-without-regret/memory/fura_qfura_defaults.md)).
  If you train a FuRA checkpoint with a different mode, override
  `--decomp-mode` to match — the panel decomposes `W'` *fresh* via BlockTT,
  it does NOT read the trained `btt_r` parameter from the checkpoint, so
  using a non-matching mode silently produces a meaningless heatmap.
- **Best-eval vs last-step checkpoints:** Always use last-step checkpoints
  per the project policy. A best-eval checkpoint locks in W' from before
  the LR-decay-to-zero phase, which makes ΔW look smaller than it should
  and shifts the rank curves down.
- **RL run dir vs ckpt dir:** The auto-detection walks up one level
  from `--full-ft-ckpt` to find `output.log`, but only one level — pass
  `--full-ft-run` / `--fura-run` / `--svd-run` explicitly if your logs live
  elsewhere.
- **Sign-flip noise in panel (b):** Per-row sign alignment removes most
  spurious flips, but near-degenerate `σ₀` eigenspaces still show up as
  bright stripes. This is a basis-rotation artifact, not an actual update;
  if it bothers a reviewer, switch to principal-angle plots.
- **Hard-coded threshold 0.90:** If a reviewer asks for r₉₉ or stable rank
  instead, edit `compute_effective_rank` directly (the threshold is a
  function default, not a CLI flag).

---

## 8. Cross-references

- Existing source: `analysis/plot_motivation_figure.py`
- Spectral primitives reused: `analysis/analyze_weights.py`,
  `btt_layer.py` (`BTTLayer.init_from_linear_weight`,
  `normalize_blocktt_decomp_mode`)
- Sister analysis (different metrics, same styling):
  [`projection_energy.md`](projection_energy.md)
- Original design doc (panel rationale, success/failure criteria, paper text):
  `docs/26_nips_fura_paper/docs/motivating_example_design.md`
- Paper inclusion: `docs/26_nips_fura_paper/neurips_2026.tex`
  (`\includegraphics{figs/motivation.pdf}` around line 199).
