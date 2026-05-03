"""Spectral-projection analysis: pretrained W0 vs gradient G, trained W', update dW.

For every linear module (q/k/v/o_proj, gate/up/down_proj) in every transformer
layer, computes five metrics with respect to the pretrained SVD
W0 = U0 S0 V0^T:

  (1) U-projection energy:  ||U0 U0^T W||_F / ||W||_F        (cheap form: ||U0^T W||_F / ||W||_F)
  (2) V-projection energy:  ||W V0 V0^T||_F / ||W||_F        (cheap form: ||W V0||_F / ||W||_F)
  (3) singular-value diff:  S - S0  (vector + L2 norm)
  (4) per-column inner product |<U0[:,i], U[:,i]>|
  (5) per-row    inner product |<V0[i,:], V[i,:]>|

Per user spec, (1) and (2) are trivially 1.0 whenever U0 (resp. V0) spans
the full output (input) basis. We only PLOT them where they are informative:
metric (1) for {up_proj, gate_proj}, metric (2) for {down_proj}. Metrics
(3)-(5) are plotted for all 7 modules.

Outputs (default paths):
  - analysis_results/spectral_projection__<tag>.npz   raw numerics cache
  - docs/exp_results/figs/motivating_uproj_vs_layer.png       (1x3: G, W', dW)
  - docs/exp_results/figs/motivating_vproj_vs_layer.png       (1x3: G, W', dW)
  - docs/exp_results/figs/motivating_detail_layer{0,15,31}_{G,Wp,dW}.png
        (9 files, each 7x3: rows = modules, cols = {S-S0, |<U0,U>|, |<V0,V>|})
  - docs/exp_results/motivating.md                            (the report)

Reuses loaders from analysis/analyze_weights.py and analysis/plot_motivation_figure.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.analyze_weights import (  # noqa: E402
    load_safetensors_index,
    load_tensor,
    detect_num_layers,
    get_base_weight_key,
    TARGET_MODULES,
)
from analysis.plot_motivation_figure import (  # noqa: E402
    load_checkpoint,
    get_weight,
    resolve_base_model,
)


# ---------------------------------------------------------------------------
# Plot styling (mirrors plot_motivation_figure.py)
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

QUANTITIES = ("G", "Wp", "dW")
QUANTITY_LABEL = {"G": "G  (first-step gradient)",
                  "Wp": "W' (trained)",
                  "dW": "ΔW = W' − W₀"}

MODULE_COLOR = {
    "q_proj":   "#2171B5",
    "k_proj":   "#6BAED6",
    "v_proj":   "#9ECAE1",
    "o_proj":   "#08519C",
    "gate_proj": "#D94801",
    "up_proj":   "#FD8D3C",
    "down_proj": "#A63603",
}

UPROJ_MODULES = ("gate_proj", "up_proj")
VPROJ_MODULES = ("down_proj",)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def _per_module_metrics(
    W0: torch.Tensor, W: torch.Tensor, mod: str,
    U0: torch.Tensor, S0: torch.Tensor, Vt0: torch.Tensor,
) -> dict:
    """Compute the five metrics for a single (W0, W, mod) triple."""
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    den = torch.linalg.norm(W)

    if mod in UPROJ_MODULES and den > 0:
        uproj = float(torch.linalg.norm(U0.T @ W) / den)
    else:
        uproj = float("nan")
    if mod in VPROJ_MODULES and den > 0:
        vproj = float(torch.linalg.norm(W @ Vt0.T) / den)
    else:
        vproj = float("nan")

    sdiff_vec = (S - S0).cpu().numpy()
    sdiff_norm = float(torch.linalg.norm(S - S0))

    udot = (U0 * U).sum(dim=0).abs().cpu().numpy()         # length r
    vdot = (Vt0 * Vt).sum(dim=1).abs().cpu().numpy()       # length r

    assert udot.shape[0] == vdot.shape[0] == min(W.shape), \
        f"shape sanity failed for mod={mod}: udot {udot.shape} vdot {vdot.shape} W {W.shape}"

    return dict(uproj=uproj, vproj=vproj,
                sdiff_vec=sdiff_vec, sdiff_norm=sdiff_norm,
                udot=udot, vdot=vdot,
                shape=tuple(W.shape))


def compute_all_metrics(
    base_index, ckpt_weights, ckpt_fmt, grads,
    num_layers: int, modules: tuple[str, ...],
    device: torch.device,
    rep_layers: set[int],
) -> dict:
    """Iterate every (layer, module). Return nested dict.

    Structure:
      results[quantity][module][layer] = {
        'uproj': float|nan, 'vproj': float|nan, 'sdiff_norm': float,
        'sdiff_vec': np.ndarray, 'udot': np.ndarray, 'vdot': np.ndarray,
        'shape': (out, in)
      }
    Only stores per-vector arrays (sdiff_vec, udot, vdot) for layer in rep_layers
    (memory savings — they are length 4096 to 14336).
    """
    results = {q: {mod: {} for mod in modules} for q in QUANTITIES}
    audit = {  # for verification appendix
        "identity_max_dev": 0.0,
        "square_attn_uproj": [],   # list of (layer, mod, val) for sanity
        "random_uproj_gate": None,
        "random_vproj_down": None,
    }

    rng = torch.Generator(device=device).manual_seed(0)

    t0 = time.time()
    for L in range(num_layers):
        for mod in modules:
            key = get_base_weight_key(L, mod)
            W0_cpu = load_tensor(base_index, key)
            Wp_cpu = get_weight(ckpt_weights, key, ckpt_fmt)
            G_cpu = grads.get(key)
            if W0_cpu is None or Wp_cpu is None or G_cpu is None:
                print(f"[skip] missing tensor for {key} (W0={W0_cpu is None}, "
                      f"Wp={Wp_cpu is None}, G={G_cpu is None})")
                continue

            W0 = W0_cpu.to(device, torch.float32)
            Wp = Wp_cpu.to(device, torch.float32)
            G = G_cpu.to(device, torch.float32)
            dW = Wp - W0

            U0, S0, Vt0 = torch.linalg.svd(W0, full_matrices=False)

            # Identity sanity check on the first encountered module of layer 0
            if L == 0 and mod == modules[0]:
                ident = _per_module_metrics(W0, W0, mod, U0, S0, Vt0)
                u_dev = float(np.max(np.abs(ident["udot"] - 1.0)))
                v_dev = float(np.max(np.abs(ident["vdot"] - 1.0)))
                s_dev = float(np.max(np.abs(ident["sdiff_vec"])))
                audit["identity_max_dev"] = max(u_dev, v_dev, s_dev)
                print(f"[audit]  identity check: max dev = {audit['identity_max_dev']:.2e}")

            # Square-attn trivial check: u/v_proj of any quantity (e.g. dW)
            # against a SQUARE module: should be ~1.0 even if not plotted.
            if mod in ("q_proj", "k_proj", "v_proj", "o_proj") and W0.shape[0] == W0.shape[1]:
                den = torch.linalg.norm(dW)
                if den > 0:
                    u_full = float(torch.linalg.norm(U0.T @ dW) / den)
                    v_full = float(torch.linalg.norm(dW @ Vt0.T) / den)
                    audit["square_attn_uproj"].append((L, mod, u_full, v_full))

            # Random-matrix baseline once per module type (only on layer 0)
            if L == 0 and mod == "gate_proj":
                W_rand = torch.randn(W0.shape, generator=rng, device=device, dtype=torch.float32)
                den = torch.linalg.norm(W_rand)
                audit["random_uproj_gate"] = float(torch.linalg.norm(U0.T @ W_rand) / den)
            if L == 0 and mod == "down_proj":
                W_rand = torch.randn(W0.shape, generator=rng, device=device, dtype=torch.float32)
                den = torch.linalg.norm(W_rand)
                audit["random_vproj_down"] = float(torch.linalg.norm(W_rand @ Vt0.T) / den)

            for label, W in [("G", G), ("Wp", Wp), ("dW", dW)]:
                m = _per_module_metrics(W0, W, mod, U0, S0, Vt0)
                # Drop heavy arrays for non-rep layers to save RAM
                if L not in rep_layers:
                    m = {k: (v if k not in ("sdiff_vec", "udot", "vdot") else None)
                         for k, v in m.items()}
                results[label][mod][L] = m

            del W0, Wp, G, dW, U0, S0, Vt0
        if L % 4 == 0:
            torch.cuda.empty_cache()
            print(f"[layer {L:>2d}/{num_layers - 1}]  ({time.time() - t0:.1f}s)")

    print(f"[done]  total {time.time() - t0:.1f}s")
    return {"metrics": results, "audit": audit}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_proj_vs_layer(results, num_layers, modules, kind: str,
                        out_path: Path):
    """Layer-trend plot: 1 row x 3 cols (G, W', dW). One curve per module."""
    field = "uproj" if kind == "u" else "vproj"
    title_base = "U-projection energy ‖U₀ᵀW‖_F / ‖W‖_F" if kind == "u" \
        else "V-projection energy ‖WV₀‖_F / ‖W‖_F"

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
    layers = np.arange(num_layers)
    for ax, q in zip(axes, QUANTITIES):
        for mod in modules:
            vals = np.array([results[q][mod].get(L, {}).get(field, np.nan)
                             for L in layers])
            if np.all(np.isnan(vals)):
                continue
            ax.plot(layers, vals, marker=".", markersize=4, linewidth=1.3,
                    color=MODULE_COLOR.get(mod, "gray"), label=mod)
        ax.set_xlabel("Layer index")
        ax.set_ylabel(field)
        ax.set_title(QUANTITY_LABEL[q], fontsize=9)
        ax.grid(True, alpha=0.2, linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", framealpha=0.9, edgecolor="gray")
    fig.suptitle(title_base, fontsize=10, y=1.04)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path))
    print(f"[plot]   {out_path}")
    plt.close(fig)


def plot_detail_grid(results_for_q, layer_idx: int, modules: tuple[str, ...],
                      quantity_label: str, out_path: Path):
    """7x3 grid for one (layer, quantity) pair.

    rows = modules, cols = {S - S0, |<U0,U>|, |<V0,V>|}.
    """
    nrows = len(modules)
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 1.6 * nrows),
                              constrained_layout=True)
    if nrows == 1:
        axes = axes[None, :]

    col_titles = ["S − S₀", "|⟨U₀[:,i], U[:,i]⟩|", "|⟨V₀[i,:], V[i,:]⟩|"]
    for ci, t in enumerate(col_titles):
        axes[0, ci].set_title(t, fontsize=9)

    for ri, mod in enumerate(modules):
        m = results_for_q[mod].get(layer_idx)
        color = MODULE_COLOR.get(mod, "gray")
        for ci, key in enumerate(("sdiff_vec", "udot", "vdot")):
            ax = axes[ri, ci]
            if m is None or m.get(key) is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
            else:
                arr = m[key]
                x = np.arange(len(arr))
                ax.plot(x, arr, linewidth=0.8, color=color)
                if key in ("udot", "vdot"):
                    ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.2, linewidth=0.4)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            if ci == 0:
                ax.set_ylabel(mod, fontsize=8)
            if ri == nrows - 1:
                ax.set_xlabel("index")

    fig.suptitle(f"{quantity_label}  —  layer {layer_idx}", fontsize=10, y=1.01)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path))
    print(f"[plot]   {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def save_cache(payload: dict, path: Path):
    """Save the metrics dict + audit + meta as a single .npz of pickled objects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path),
                        metrics=np.array(payload["metrics"], dtype=object),
                        audit=np.array(payload["audit"], dtype=object),
                        meta=np.array(payload["meta"], dtype=object))
    print(f"[cache]  {path}")


def load_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    z = np.load(str(path), allow_pickle=True)
    return {"metrics": z["metrics"].item(),
            "audit":   z["audit"].item(),
            "meta":    z["meta"].item()}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(payload: dict, args, fig_dir: Path, report_path: Path):
    audit = payload["audit"]
    meta = payload["meta"]

    rep_layers = meta["rep_layers"]
    fig_rel = fig_dir.name  # 'figs' typically; for embedding paths

    # square-attn audit summary
    if audit["square_attn_uproj"]:
        u_vals = np.array([t[2] for t in audit["square_attn_uproj"]])
        v_vals = np.array([t[3] for t in audit["square_attn_uproj"]])
        square_summary = (f"u_proj range [{u_vals.min():.6f}, {u_vals.max():.6f}], "
                          f"v_proj range [{v_vals.min():.6f}, {v_vals.max():.6f}] "
                          f"across {len(u_vals)} square attn (layer, module) entries.")
    else:
        square_summary = "no square attn modules sampled."

    body = []
    body.append("# Spectral relationship: pretrained weights vs gradient, trained weights, update\n")
    body.append("## Setup\n")
    body.append(f"- Base model: `{meta['base_model']}` ({meta['num_layers']} transformer layers)")
    body.append(f"- Trained checkpoint: `{meta['checkpoint']}`")
    body.append(f"- First-step gradient G: `{meta['gradient']}`")
    body.append(f"- Modules analyzed: {', '.join(meta['modules'])}")
    body.append(f"- Representative layers: {rep_layers}")
    body.append("- All SVDs computed in fp32 with `full_matrices=False`.")
    body.append("")

    body.append("## Metrics\n")
    body.append("Let W ∈ {G, W′, ΔW = W′ − W₀} with thin SVD W = U S Vᵀ, and "
                "W₀ = U₀ S₀ V₀ᵀ. Five per-(layer, module) quantities:")
    body.append("")
    body.append("1. **U-projection energy**  `‖U₀U₀ᵀW‖_F / ‖W‖_F  =  ‖U₀ᵀW‖_F / ‖W‖_F`  "
                "(orthonormal U₀)")
    body.append("2. **V-projection energy**  `‖WV₀V₀ᵀ‖_F / ‖W‖_F  =  ‖WV₀‖_F / ‖W‖_F`")
    body.append("3. **Singular-value diff** `S − S₀` (vector and L2 norm)")
    body.append("4. **Per-column overlap** `|⟨U₀[:,i], U[:,i]⟩|`, i = 0..r−1")
    body.append("5. **Per-row overlap**    `|⟨V₀[i,:], V[i,:]⟩|`, i = 0..r−1")
    body.append("")

    body.append("## Interpretation note (scope of metrics 1 & 2)\n")
    body.append("With the **full** U₀ used in the spec, metric (1) collapses to 1.0 whenever "
                "U₀ spans the full output space — i.e. whenever `out_features ≤ in_features`. "
                "For Llama-3-8B that means metric (1) is trivially 1.0 for the four attention "
                "projections (square 4096×4096) and for `down_proj` (4096×14336). It is informative "
                "only for `gate_proj` and `up_proj` (14336×4096). Metric (2) is the symmetric "
                "case: trivial except on `down_proj`. We therefore plot metric (1) only for "
                "{`gate_proj`, `up_proj`}, and metric (2) only for `down_proj`. Metrics (3)–(5) "
                "are informative for all module shapes and are plotted for all 7 modules.")
    body.append("")

    body.append("## Results\n")
    body.append("### U-projection energy (rectangular `gate_proj` / `up_proj` only)\n")
    body.append(f"![](./{fig_rel}/motivating_uproj_vs_layer.png)\n")

    body.append("### V-projection energy (rectangular `down_proj` only)\n")
    body.append(f"![](./{fig_rel}/motivating_vproj_vs_layer.png)\n")

    body.append("### Singular-value & singular-vector overlap (representative layers)\n")
    body.append("Each figure is a 7×3 grid. Rows = module type "
                "(`q,k,v,o,gate,up,down_proj`). Columns = (S − S₀, "
                "|⟨U₀[:,i],U[:,i]⟩|, |⟨V₀[i,:],V[i,:]⟩|). x-axis is the singular index "
                "i = 0..min(out, in)−1. Note: under near-degenerate σ₀ the per-column dot "
                "product can swing within a degenerate eigenspace even when the *subspace* "
                "is preserved; for a cleaner subspace metric see "
                "`compute_principal_angles` in `analysis/analyze_weights.py`.")
    body.append("")

    for q in QUANTITIES:
        body.append(f"#### {QUANTITY_LABEL[q]}\n")
        for L in rep_layers:
            body.append(f"![Layer {L}](./{fig_rel}/motivating_detail_layer{L}_{q}.png)\n")

    body.append("## Verification appendix\n")
    body.append(f"- **Identity check** (W = W₀): max deviation across "
                f"`u_dot−1`, `v_dot−1`, `S−S₀` was **{audit['identity_max_dev']:.2e}** "
                f"(should be O(1e-5) in fp32).")
    body.append(f"- **Square-attn trivial check** (u_proj, v_proj of ΔW on q/k/v/o_proj): "
                f"{square_summary} Should be ≈ 1.0 — confirms the skip-condition.")
    if audit["random_uproj_gate"] is not None:
        d_in = 14336
        d_out = 4096
        expected = (d_out / d_in) ** 0.5
        body.append(f"- **Random-Gaussian baseline for `gate_proj` u_proj** "
                    f"(layer 0 W₀, random W of same shape): "
                    f"observed = **{audit['random_uproj_gate']:.4f}**, "
                    f"expected ≈ √(out/in) = √({d_out}/{d_in}) ≈ {expected:.4f}.")
    if audit["random_vproj_down"] is not None:
        d_in = 14336
        d_out = 4096
        expected = (d_out / d_in) ** 0.5
        body.append(f"- **Random-Gaussian baseline for `down_proj` v_proj**: "
                    f"observed = **{audit['random_vproj_down']:.4f}**, "
                    f"expected ≈ √(out/in) = √({d_out}/{d_in}) ≈ {expected:.4f}.")
    body.append("")

    body.append("## Reproduction\n")
    body.append("```bash")
    body.append("# 1. One-time gradient collection (~3 min, single H100)")
    body.append("CUDA_VISIBLE_DEVICES=0 uv run python analysis/compute_first_step_grad.py \\")
    body.append(f"    --base-model {meta['base_model']} \\")
    body.append("    --data-path ref/LIFT/LLM-Adapters/ft-training_set/commonsense_170k.json \\")
    body.append(f"    --seed 43 \\")
    body.append(f"    --output {meta['gradient']}")
    body.append("")
    body.append("# 2. Spectral analysis (~10 min, single H100)")
    body.append("CUDA_VISIBLE_DEVICES=0 uv run python analysis/spectral_projection.py \\")
    body.append(f"    --base-model {meta['base_model']} \\")
    body.append(f"    --checkpoint {meta['checkpoint']} \\")
    body.append(f"    --gradient {meta['gradient']}")
    body.append("```")
    body.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(body))
    print(f"[report] {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="Trained checkpoint dir (safetensors or pytorch_model.bin)")
    p.add_argument("--gradient", required=True,
                   help=".pt produced by compute_first_step_grad.py")
    p.add_argument("--output-dir", default="docs/exp_results/figs",
                   help="Where to write PNG figures")
    p.add_argument("--report-md", default="docs/exp_results/motivating.md")
    p.add_argument("--results-cache", default=None,
                   help=".npz path for raw numerics (default: derived from checkpoint name)")
    p.add_argument("--rep-layers", default="0,15,31",
                   help="Comma-separated layer indices for detail panels")
    p.add_argument("--modules", default=",".join(TARGET_MODULES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--skip-report", action="store_true")
    p.add_argument("--reuse-cache", action="store_true",
                   help="If cache file exists, skip recomputation and just plot/report")
    return p.parse_args()


def main():
    args = parse_args()

    rep_layers = sorted(int(x) for x in args.rep_layers.split(","))
    modules = tuple(args.modules.split(","))
    for mod in modules:
        if mod not in TARGET_MODULES:
            raise ValueError(f"Unknown module {mod}; allowed = {TARGET_MODULES}")

    fig_dir = Path(args.output_dir)
    report_path = Path(args.report_md)
    cache_path = Path(args.results_cache) if args.results_cache else \
        Path("analysis_results") / (
            f"spectral_projection__{Path(args.checkpoint).parent.name}.npz"
        )

    payload = None
    if args.reuse_cache:
        payload = load_cache(cache_path)
        if payload is not None:
            print(f"[cache]  reusing {cache_path}")

    if payload is None:
        device = torch.device(args.device)
        print(f"[init]   device = {device}")

        base_path = resolve_base_model(args.base_model)
        base_index = load_safetensors_index(base_path)
        num_layers = detect_num_layers(base_index)
        print(f"[init]   base = {args.base_model} ({num_layers} layers)")

        ckpt_weights, ckpt_fmt = load_checkpoint(args.checkpoint)
        print(f"[init]   ckpt = {args.checkpoint} (fmt={ckpt_fmt})")

        grads = torch.load(args.gradient, map_location="cpu", weights_only=True)
        print(f"[init]   grads = {args.gradient}  ({len(grads)} tensors)")

        for L in rep_layers:
            if L >= num_layers:
                raise ValueError(f"--rep-layers {L} out of range (model has {num_layers} layers)")

        out = compute_all_metrics(
            base_index, ckpt_weights, ckpt_fmt, grads,
            num_layers=num_layers, modules=modules,
            device=device, rep_layers=set(rep_layers),
        )
        payload = {
            "metrics": out["metrics"],
            "audit": out["audit"],
            "meta": {
                "base_model": args.base_model,
                "checkpoint": args.checkpoint,
                "gradient": args.gradient,
                "num_layers": num_layers,
                "rep_layers": rep_layers,
                "modules": list(modules),
            },
        }
        save_cache(payload, cache_path)

    if not args.skip_plots:
        results = payload["metrics"]
        num_layers = payload["meta"]["num_layers"]
        rep_layers = payload["meta"]["rep_layers"]

        plot_proj_vs_layer(results, num_layers, UPROJ_MODULES, "u",
                            fig_dir / "motivating_uproj_vs_layer.png")
        plot_proj_vs_layer(results, num_layers, VPROJ_MODULES, "v",
                            fig_dir / "motivating_vproj_vs_layer.png")
        for q in QUANTITIES:
            for L in rep_layers:
                plot_detail_grid(results[q], L, tuple(payload["meta"]["modules"]),
                                  QUANTITY_LABEL[q],
                                  fig_dir / f"motivating_detail_layer{L}_{q}.png")

    if not args.skip_report:
        write_report(payload, args, fig_dir, report_path)

    # Print audit summary
    audit = payload["audit"]
    print("\n=== AUDIT SUMMARY ===")
    print(f"identity max dev       : {audit['identity_max_dev']:.2e}")
    if audit['square_attn_uproj']:
        u_vals = np.array([t[2] for t in audit['square_attn_uproj']])
        v_vals = np.array([t[3] for t in audit['square_attn_uproj']])
        print(f"square-attn u_proj     : range [{u_vals.min():.6f}, {u_vals.max():.6f}]")
        print(f"square-attn v_proj     : range [{v_vals.min():.6f}, {v_vals.max():.6f}]")
    print(f"random gate u_proj     : {audit['random_uproj_gate']}")
    print(f"random down v_proj     : {audit['random_vproj_down']}")


if __name__ == "__main__":
    main()
