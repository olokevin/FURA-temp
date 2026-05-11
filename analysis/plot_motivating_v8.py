"""Generate the v8 6-panel motivating-example figure (motivating_v8.png).

Panels (2x3, first row a/b/c, second row d/e/f):
  (a) L15 up_proj: u-projection energy of G (gradient) and W' over training step,
      with random Gaussian baseline √(out/in) ≈ 0.5345 dashed line.
      Source: proj_energy.csv from full-lr_1e-5-seed_43-projE run.
  (b) Layer-15 up_proj: ΔS = S' - S₀ AND |⟨U₀[:,i], U[:,i]⟩| similarity, two stacked sub-axes
      computed from full-FT math projE checkpoint vs base model.
  (c) Bar chart: target-domain math avg (left group) vs source-domain commonsense avg
      (right group), bars for {Full FT, SVD FT}, horizontal dashed line for pretrained.
  (d) Effective rank of ΔW vs layer index — three curves: Full FT, FuRA, LoRA.
  (e) FuRA core-R update heatmap |R'-R₀|, but only 4 evenly-spaced block slices
      (indices 0, n/3, 2n/3, n-1) instead of all n blocks.
  (f) Bar chart: math avg vs commonsense avg, bars for {Full FT, LoRA, FuRA},
      horizontal dashed pretrained line.

All checkpoint paths and bar values are configurable via the CONFIG dict at the top.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.plot_motivation_figure import (  # noqa: E402
    COLOR_Q,
    COLOR_UP,
    compute_effective_rank,
    compute_R_update_heatmap,
    detect_num_layers,
    get_weight,
    load_checkpoint,
    load_safetensors_index,
    load_tensor_sf,
    resolve_base_model,
    _module_prefix,
)

# Local color overrides (red for Full FT, grey for LoRA, blue for SVD / FuRA).
COLOR_FULL = "#D62728"  # red
COLOR_LORA = "#9E9E9E"  # grey
COLOR_SVD  = "#1F77B4"  # blue (also used for SVD-FT bars)
COLOR_FURA = "#1F77B4"  # blue (FuRA shares the blue family)


# ---------------------------------------------------------------------------
# CONFIG — checkpoint paths and panel-specific values. Edit here, not in code.
# ---------------------------------------------------------------------------

CONFIG = {
    "base_model": "meta-llama/Meta-Llama-3-8B",
    # Checkpoints (last-step models, see CLAUDE.md policy).
    "ckpts": {
        "full_ft": "/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/full-lr_1e-5-seed_43-projE/last",
        "fura":    "/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/blocktt-lr_2e-4-decomp_output_one_block_pos_small-rank_full-smerge_trainable-type_all-seed_43",
        "lora":    "/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/lora-lr_2e-4-rank_64-alpha_128-seed_43",
    },
    # Online recorder run dir for panel (a).
    "proj_energy_csv": "/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/full-lr_1e-5-seed_43-projE/proj_energy.csv",
    # Panel (a)/(b) shape parameters for the random baseline.
    "rect_out": 4096,    # up_proj out_features
    "rect_in": 14336,    # up_proj in_features
    # Panel (b) target.
    "panel_b_layer": 15,
    "panel_b_module": "up_proj",
    # Panel (e) target.
    "panel_e_layer": 15,
    "panel_e_module": "q_proj",
    "panel_e_decomp_mode": "output_one_block",
    "panel_e_n_slices": 4,
    # Panel (d) target.
    "panel_d_modules": ["q_proj"],
    # Bar values for panels (c) and (f). Math = in-domain (target).
    # Commonsense = source-domain forgetting probe.
    # Sources:
    #   docs/exp_results/lift_math.md "Source-Domain Forgetting" table for FuRA/LoRA/Base commonsense.
    #   docs/exp_results/lift_math.md "Full Eval Sweep" table for math averages.
    #   Full FT and SVD FT have no clean 8-task commonsense Avg from a math run -- left as None.
    "bars": {
        # Target = Math-10K avg accuracy (in-domain).
        # Source = Commonsense avg accuracy (out-of-domain, source-domain forgetting).
        # Values per user-supplied table (2026-05-03, updated).
        "math": {
            "Pretrained": 41.3,        # shared pretrain baseline (target side)
            "Full FT": 71.2,
            "SVD FT": 72.0,
            "LoRA": 70.8,
            "FuRA": 71.6,
        },
        "commonsense": {
            "Pretrained": 41.3,        # shared pretrain baseline (source side)
            "Full FT": 40.6,
            "SVD FT": 47.70,
            "LoRA": 39.7,
            "FuRA": 45.5,
        },
    },
    # Outputs (two separate figures).
    "output_svdft": "docs/exp_results/figs/motivating_svdft.png",
    "output_fura":  "docs/exp_results/figs/motivating_fura.png",
    # Cache directory for precomputed panel (d) and panel (e) data.
    # Caches are versioned by their inputs (ckpt path + module list / layer).
    "cache_dir": "analysis_results/motivating_v8_cache",
    "device": "cuda:0",
}


# ---------------------------------------------------------------------------
# Plot styling (mirrors plot_motivation_figure.py)
# ---------------------------------------------------------------------------

FONT_BASE = 13
FONT_LABEL = 18
FONT_TITLE = 20
FONT_TICK = 16
FONT_LEGEND = 16
FONT_PANEL_LABEL = 17
FONT_BAR_NUMBER = 16   # numeric labels above bars in panel (c) of each figure

plt.rcParams.update({
    "font.family": "serif",
    "font.size": FONT_BASE,
    "axes.labelsize": FONT_LABEL,
    "axes.titlesize": FONT_TITLE,
    "axes.titleweight": "bold",
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_LEGEND,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})


# Panel labels go in the upper-left corner of each panel cell, using *figure*
# coords so they line up across rows and columns regardless of per-axis layout.
def _panel_label_fig(fig, x_fig: float, y_fig: float, label: str):
    fig.text(x_fig, y_fig, f"({label})",
             fontsize=FONT_PANEL_LABEL, fontweight="bold",
             ha="left", va="top")


def _ckpt_tag(ckpt_path: str) -> str:
    """Stable, filesystem-safe tag for a checkpoint dir.

    If the leaf is a generic name like ``last`` / ``best`` / ``checkpoint`` /
    ``step=...``, climb one parent so the tag identifies the run, not the
    sub-checkpoint.
    """
    p = Path(ckpt_path)
    leaf = p.name
    parent = p.parent.name
    generic = {"last", "best", "checkpoint"}
    if leaf in generic or leaf.startswith("step=") or leaf.startswith("checkpoint-"):
        return f"{parent}__{leaf}".replace("/", "_")
    return leaf.replace("/", "_")


def _panel_d_cache_path(cache_dir: Path, ckpts: dict, modules: list[str]) -> Path:
    """Cache key for panel (d) is (full_ft, fura, lora ckpts) × modules list."""
    parts = [_ckpt_tag(ckpts[k]) for k in ("full_ft", "fura", "lora")]
    mod_tag = "_".join(modules)
    fname = "panel_d__" + "__".join(parts) + f"__{mod_tag}.npz"
    return cache_dir / fname


def _panel_e_cache_path(cache_dir: Path, fura_ckpt: str, layer: int,
                        module: str, decomp_mode: str) -> Path:
    """Cache key for panel (e) is the FuRA ckpt + (layer, module, decomp)."""
    fname = (f"panel_e__{_ckpt_tag(fura_ckpt)}__L{layer}__{module}"
             f"__{decomp_mode}.npz")
    return cache_dir / fname


def _save_panel_d_cache(path: Path, ranks_per_method: dict[str, dict[str, np.ndarray]]):
    """Save {method: {module: array}} as a flat npz."""
    flat = {}
    for method, per_mod in ranks_per_method.items():
        for mod, arr in per_mod.items():
            flat[f"{method}__{mod}"] = arr
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **flat)
    print(f"[cache] saved panel-d → {path}")


def _load_panel_d_cache(path: Path) -> dict[str, dict[str, np.ndarray]]:
    z = np.load(str(path))
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in z.files:
        method, mod = key.split("__", 1)
        out.setdefault(method, {})[mod] = z[key]
    print(f"[cache] loaded panel-d ← {path}")
    return out


def _save_panel_e_cache(path: Path, r_info: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path),
                        delta=r_info["delta"],
                        n=np.int64(r_info["n"]),
                        b=np.int64(r_info["b"]),
                        m=np.int64(r_info["m"]),
                        rank=np.int64(r_info["rank"]))
    print(f"[cache] saved panel-e → {path}")


def _load_panel_e_cache(path: Path) -> dict:
    z = np.load(str(path))
    out = {"delta": z["delta"],
           "n": int(z["n"]), "b": int(z["b"]),
           "m": int(z["m"]), "rank": int(z["rank"])}
    print(f"[cache] loaded panel-e ← {path}")
    return out


def _render_panel_titles(fig, gs, titles: list[str], y_fig: float):
    """Place three panel titles centered above each gridspec cell at the same y.

    ``titles[i]`` is centered horizontally over column ``i`` of ``gs`` (gs is a
    1-row GridSpec). All titles share the same figure-y for clean alignment.
    """
    for i, title in enumerate(titles):
        bbox = gs[0, i].get_position(fig)
        x_center = 0.5 * (bbox.x0 + bbox.x1)
        fig.text(x_center, y_fig, title,
                 fontsize=FONT_TITLE, fontweight="bold",
                 ha="center", va="bottom")


# ---------------------------------------------------------------------------
# Panel (a): u-projection energy vs training step for L15 up_proj
# ---------------------------------------------------------------------------

def plot_panel_a(ax, csv_path: str, rect_out: int, rect_in: int,
                 panel_prefix: str = "", show_annotations: bool = True):
    df = pd.read_csv(csv_path)
    steps = df["step"].values
    g = df["L15.up_proj.G.u_proj"].values
    w = df["L15.up_proj.W.u_proj"].values
    baseline = (rect_out / rect_in) ** 0.5

    color_g = "#D94801"
    color_w = "#2171B5"

    # Single y-axis with both curves and the random baseline.
    ax.plot(steps, w, color=color_w, linewidth=3,
            label="Weight W'")
    ax.plot(steps, g, color=color_g, linewidth=3, alpha=0.9,
            label="Gradient G")
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=2.5,
               label="Random Matrix")

    if show_annotations:
        # Annotate W's final value so it doesn't read as "1.0".
        w_final = float(w[-1])
        ax.annotate(
            f"W' final = {w_final:.4f}",
            xy=(steps[-1], w_final),
            xytext=(steps[-1] * 0.40, w_final - 0.22),
            fontsize=FONT_LABEL + 5,
            color="black",
            arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
        )

    ax.set_xlabel("Training step", fontsize=FONT_LABEL + 4)
    # No y-label per user spec.
    ax.set_ylim(0.0, 1.10)
    # Legend in-plot at the lower right.
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="gray",
              fontsize=FONT_LEGEND, handlelength=1.6,
              borderpad=0.4)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Panel (b): ΔS and |<U0,U>| for L15 up_proj
# ---------------------------------------------------------------------------

def compute_S_U_overlap(W0: torch.Tensor, Wp: torch.Tensor):
    """Return (S0, S, sdiff, udot) for one (W0, Wp) pair."""
    W0f = W0.float()
    Wpf = Wp.float()
    U0, S0, _ = torch.linalg.svd(W0f, full_matrices=False)
    U, S, _ = torch.linalg.svd(Wpf, full_matrices=False)
    sdiff = (S - S0).cpu().numpy()
    udot = (U0 * U).sum(dim=0).abs().cpu().numpy()
    return S0.cpu().numpy(), S.cpu().numpy(), sdiff, udot


def plot_panel_b(ax_top, ax_bot, base_index, full_weights, full_fmt,
                 layer: int, module: str, device: torch.device,
                 panel_prefix: str = "", show_subtitle: bool = True):
    key = f"{_module_prefix(layer, module)}.weight"
    W0 = load_tensor_sf(base_index, key)
    Wp = get_weight(full_weights, key, full_fmt)
    if W0 is None or Wp is None:
        for ax in (ax_top, ax_bot):
            ax.text(0.5, 0.5, "missing weight", ha="center", va="center",
                    transform=ax.transAxes)
        return
    W0g = W0.to(device)
    Wpg = Wp.to(device)
    S0, S, sdiff, udot = compute_S_U_overlap(W0g, Wpg)
    idx = np.arange(len(sdiff))

    # Top: S - S0 (scatter)
    ax_top.scatter(idx, sdiff, s=4, color=COLOR_FULL, alpha=0.6,
                   edgecolors="none")
    ax_top.axhline(0, color="gray", linewidth=0.4, linestyle=":")
    # No y-label on (b)-upper per user spec.
    # No axes-title here — panel title placed by orchestrator at a uniform y.
    ax_top.grid(True, alpha=0.2, linewidth=0.4)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(labelbottom=False)

    # Bottom: |<U0, U>| (scatter)
    ax_bot.scatter(idx, udot, s=4, color=COLOR_UP, alpha=0.6,
                   edgecolors="none")
    ax_bot.axhline(1.0, color="gray", linewidth=0.4, linestyle=":")
    ax_bot.set_ylim(-0.05, 1.10)
    # Sub-title with extra top padding so it doesn't collide with the upper subplot.
    if show_subtitle:
        ax_bot.set_title(r"Singular vector rotation $\mathbf{U}_i'$ - $\mathbf{U}_i$",
                         fontsize=FONT_TITLE, pad=8)
    ax_bot.set_xlabel("Singular index i", fontsize=FONT_LABEL + 4)
    # No y-label on the lower scatter (panel-(b) lower).
    ax_bot.grid(True, alpha=0.2, linewidth=0.4)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Panel (c)/(f): grouped bars with pretrained dashed line
# ---------------------------------------------------------------------------

def plot_grouped_bars(ax, methods: list[str], math_vals: dict[str, float | None],
                      cs_vals: dict[str, float | None],
                      pretrained_cs: float | None,
                      pretrained_math: float | None,
                      title: str):
    color_map = {
        "Pretrained": "#BDBDBD",
        "Full FT": COLOR_FULL,
        "SVD FT": COLOR_SVD,
        "FuRA": COLOR_FURA,
        "LoRA": COLOR_LORA,
    }
    n = len(methods)
    group_centers = np.array([0.0, 1.0])
    group_width = 0.8
    bar_width = group_width / n
    offsets = (np.arange(n) - (n - 1) / 2) * bar_width

    # Identify the best (max) value per group so we can bold only the winner.
    target_vals = {m: math_vals.get(m) for m in methods if math_vals.get(m) is not None}
    source_vals = {m: cs_vals.get(m) for m in methods if cs_vals.get(m) is not None}
    best_target = max(target_vals, key=target_vals.get) if target_vals else None
    best_source = max(source_vals, key=source_vals.get) if source_vals else None

    for j, m in enumerate(methods):
        x_positions = group_centers + offsets[j]
        vals = [math_vals.get(m), cs_vals.get(m)]
        is_best = [m == best_target, m == best_source]
        # Drop missing bars but keep the slot — show as light hatched bar.
        for x, v, best in zip(x_positions, vals, is_best):
            if v is None:
                ax.bar(x, 0.5, width=bar_width, color="#EEEEEE",
                       edgecolor="#999999", linewidth=0.5, hatch="///")
                ax.text(x, 0.6, "N/A", ha="center", va="bottom",
                        fontsize=FONT_BASE - 1, color="#666")
            else:
                ax.bar(x, v, width=bar_width, color=color_map.get(m, "#888"),
                       edgecolor="white", linewidth=0.5, label=m if x == x_positions[0] else None)
                ax.text(x, v + 0.8, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=FONT_BAR_NUMBER,
                        fontweight=("bold" if best else "normal"))

    # Pretrained baseline as a dashed line over the Source group only.
    # (No dashed line on the Target side.)
    pretrain_handle = None
    if pretrained_cs is not None:
        line, = ax.plot(
            [group_centers[1] - group_width / 2,
             group_centers[1] + group_width / 2],
            [pretrained_cs, pretrained_cs],
            color="black", linestyle="--", linewidth=2.5,
            label="Pretrain",
        )
        pretrain_handle = line

    ax.set_xticks(group_centers)
    ax.set_xticklabels(["Target", "Source"], fontsize=FONT_LABEL + 4)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL + 4)
    # Panel title placed by orchestrator at uniform y.
    # y-axis up to a reasonable max
    all_vals = [v for v in list(math_vals.values()) + list(cs_vals.values())
                if v is not None]
    if pretrained_cs is not None:
        all_vals.append(pretrained_cs)
    if pretrained_math is not None:
        all_vals.append(pretrained_math)
    ymax = max(all_vals) * 1.18 if all_vals else 100
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Custom legend (one entry per method, plus Pretrain baseline if present)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map.get(m, "#888"))
               for m in methods]
    labels = list(methods)
    if pretrain_handle is not None:
        handles.append(pretrain_handle)
        labels.append("Pretrain")
    ax.legend(handles, labels, loc="upper right", framealpha=0.9,
              edgecolor="gray", fontsize=FONT_LEGEND, ncol=1)


# ---------------------------------------------------------------------------
# Panel (d): rank curves vs layer for {full, fura, lora}
# ---------------------------------------------------------------------------

def compute_per_layer_rank(base_index, ckpt_weights, ckpt_fmt,
                            num_layers: int, modules: list[str],
                            device: torch.device, label: str) -> dict[str, np.ndarray]:
    """{module: rank_per_layer}. Skips missing tensors (leaves 0)."""
    out = {mod: np.zeros(num_layers) for mod in modules}
    for L in range(num_layers):
        for mod in modules:
            key = f"{_module_prefix(L, mod)}.weight"
            W0 = load_tensor_sf(base_index, key)
            W1 = get_weight(ckpt_weights, key, ckpt_fmt)
            if W0 is None or W1 is None:
                continue
            delta = W1.float().to(device) - W0.float().to(device)
            out[mod][L] = compute_effective_rank(delta)
            del W0, W1, delta
        if L % 8 == 0:
            print(f"  [{label}] rank: layer {L}/{num_layers - 1}")
    return out


def plot_panel_d(ax, ranks_per_method: dict[str, dict[str, np.ndarray]],
                 modules: list[str], panel_prefix: str = ""):
    """Plot 3 methods, each one curve per module (typically up_proj only).

    Solid line for full FT, dashed for FuRA, dotted for LoRA.
    Legend labels are just the method names — no module suffix.
    """
    method_colors = {"Full FT": COLOR_FULL, "FuRA": COLOR_FURA, "LoRA": COLOR_LORA}
    method_styles = {"Full FT": "-", "FuRA": "--", "LoRA": ":"}
    module_markers = {"q_proj": "o", "up_proj": "s"}

    for method, per_mod in ranks_per_method.items():
        for mod in modules:
            vals = per_mod.get(mod)
            if vals is None:
                continue
            layers = np.arange(len(vals))
            ax.plot(layers, vals,
                    color=method_colors.get(method, "gray"),
                    linestyle=method_styles.get(method, "-"),
                    marker=module_markers.get(mod, "."),
                    markersize=4,
                    linewidth=2.5,
                    label=method)

    ax.set_xlabel("Layer index", fontsize=FONT_LABEL + 4)
    # ax.set_ylabel("Effective rank of ΔW")
    # In-plot legend at lower right.
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="gray",
              fontsize=FONT_LEGEND, handlelength=1.8, borderpad=0.4)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Panel (e): FuRA R-update heatmap, 4 evenly-spaced slices only
# ---------------------------------------------------------------------------

def plot_panel_e_2x2(fig, gs_cell, r_info: dict, layer: int, module: str,
                     panel_prefix: str = ""):
    """Plot |R'-R0| as 4 square heatmaps in a 2x2 layout sharing one colorbar.

    Slices picked evenly across n: [0, n/3, 2n/3, n-1].
    """
    if r_info is None:
        ax = fig.add_subplot(gs_cell)
        ax.text(0.5, 0.5, "missing", ha="center", va="center",
                transform=ax.transAxes)
        return ax

    delta = r_info["delta"]   # (n, b, m*rank)
    n, b, mr = delta.shape
    n_slices = 4
    slice_idx = [int(round(k * (n - 1) / (n_slices - 1))) for k in range(n_slices)]
    seen = set()
    slice_idx = [s for s in slice_idx if not (s in seen or seen.add(s))]

    eps = 1e-10
    log_slices = [np.log10(delta[s] + eps) for s in slice_idx]
    vmin = min(arr.min() for arr in log_slices)
    vmax = max(arr.max() for arr in log_slices)

    # 2x2 sub-grid inside gs_cell, with a thin colorbar column on the LEFT.
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=gs_cell,
        width_ratios=[0.06, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        hspace=0.30, wspace=0.40,
    )
    cax = fig.add_subplot(inner[:, 0])
    axes_2x2 = [
        fig.add_subplot(inner[0, 1]),
        fig.add_subplot(inner[0, 2]),
        fig.add_subplot(inner[1, 1]),
        fig.add_subplot(inner[1, 2]),
    ]

    im = None
    for ax_i, sidx, arr in zip(axes_2x2, slice_idx, log_slices):
        im = ax_i.imshow(arr, aspect="equal", cmap="magma",
                         interpolation="nearest", vmin=vmin, vmax=vmax)
        ax_i.set_title(f"block {sidx}", fontsize=FONT_BASE, pad=3)
        # Show 3 ticks along each axis at the matrix coords (0, mid, end-1).
        h, w = arr.shape
        xticks = [0, w // 2, w - 1]
        yticks = [0, h // 2, h - 1]
        ax_i.set_xticks(xticks)
        ax_i.set_xticklabels([str(t) for t in xticks])
        ax_i.set_yticks(yticks)
        ax_i.set_yticklabels([str(t) for t in yticks])
        ax_i.tick_params(axis="both", labelsize=FONT_TICK - 1, length=2.5)

    cbar = fig.colorbar(im, cax=cax, label="log₁₀|R'−R₀|")
    # Move ticks and the label to the LEFT side of the colorbar.
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")
    cax.tick_params(labelsize=FONT_TICK)
    cax.yaxis.label.set_size(FONT_LABEL + 4)
    # Panel title handled by orchestrator; per-block titles already drawn.
    return axes_2x2[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-svdft", default=CONFIG["output_svdft"])
    parser.add_argument("--output-fura", default=CONFIG["output_fura"])
    parser.add_argument("--skip-fura", action="store_true",
                        help="Skip the FuRA figure entirely (panels d/e/f).")
    parser.add_argument("--skip-svdft", action="store_true",
                        help="Skip the SVD-FT figure entirely (panels a/b/c).")
    parser.add_argument("--skip-d", action="store_true",
                        help="Skip panel (d) effective-rank computation.")
    parser.add_argument("--skip-e", action="store_true",
                        help="Skip panel (e) BlockTT R-update computation.")
    parser.add_argument("--cache-dir", default=CONFIG["cache_dir"],
                        help="Directory to read/write panel (d)/(e) caches.")
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore cached panel (d)/(e) data and recompute.")
    parser.add_argument("--no-titles", action="store_true",
                        help=("Suppress all subplot titles AND the W' final "
                              "/ random-baseline annotations on panel (a). "
                              "The figure name gets a `_notitles` suffix."))
    args = parser.parse_args()

    device = torch.device(CONFIG["device"])
    print(f"Device: {device}")

    # ---- Load base model + checkpoints ----
    base_path = resolve_base_model(CONFIG["base_model"])
    base_index = load_safetensors_index(base_path)
    num_layers = detect_num_layers(base_index)
    print(f"Base: {CONFIG['base_model']} ({num_layers} layers)")

    cache_dir = Path(args.cache_dir)

    # ---- Lazy ckpt loaders. Panels a/b need full_w; d needs all three;
    # e needs fura_w. We only load each one if its consumer actually runs.
    _ckpt_cache = {}

    def _get_ckpt(name: str):
        if name not in _ckpt_cache:
            print(f"Loading {name} ckpt...")
            _ckpt_cache[name] = load_checkpoint(CONFIG["ckpts"][name])
        return _ckpt_cache[name]

    # ---- Panel (d): rank curves (FuRA figure) ----
    panel_d_path = _panel_d_cache_path(cache_dir, CONFIG["ckpts"],
                                       CONFIG["panel_d_modules"])
    ranks_per_method: dict[str, dict[str, np.ndarray]] = {}
    if not args.skip_fura and not args.skip_d:
        if panel_d_path.exists() and not args.recompute:
            ranks_per_method = _load_panel_d_cache(panel_d_path)
        else:
            full_w, full_fmt = _get_ckpt("full_ft")
            fura_w, fura_fmt = _get_ckpt("fura")
            lora_w, lora_fmt = _get_ckpt("lora")
            print("Computing per-layer rank curves...")
            ranks_per_method = {
                "Full FT": compute_per_layer_rank(
                    base_index, full_w, full_fmt, num_layers,
                    CONFIG["panel_d_modules"], device, "full"),
                "FuRA": compute_per_layer_rank(
                    base_index, fura_w, fura_fmt, num_layers,
                    CONFIG["panel_d_modules"], device, "fura"),
                "LoRA": compute_per_layer_rank(
                    base_index, lora_w, lora_fmt, num_layers,
                    CONFIG["panel_d_modules"], device, "lora"),
            }
            _save_panel_d_cache(panel_d_path, ranks_per_method)

    # ---- Panel (e): FuRA R-heatmap on a single layer ----
    panel_e_path = _panel_e_cache_path(cache_dir, CONFIG["ckpts"]["fura"],
                                       CONFIG["panel_e_layer"],
                                       CONFIG["panel_e_module"],
                                       CONFIG["panel_e_decomp_mode"])
    r_info = None
    if not args.skip_fura and not args.skip_e:
        if panel_e_path.exists() and not args.recompute:
            r_info = _load_panel_e_cache(panel_e_path)
        else:
            fura_w, fura_fmt = _get_ckpt("fura")
            print(f"Computing FuRA R-heatmap on layer {CONFIG['panel_e_layer']}...")
            from analysis.plot_motivation_figure import decompose_to_btt_R
            L = CONFIG["panel_e_layer"]
            key = f"{_module_prefix(L, CONFIG['panel_e_module'])}.weight"
            W0 = load_tensor_sf(base_index, key).float().to(device)
            Wp = get_weight(fura_w, key, fura_fmt).float().to(device)
            R0, n, b, m, rank = decompose_to_btt_R(W0, CONFIG["panel_e_decomp_mode"])
            R1, _, _, _, _ = decompose_to_btt_R(Wp, CONFIG["panel_e_decomp_mode"])
            delta_R = (R1 - R0).abs().cpu().numpy()
            r_info = {"delta": delta_R, "n": n, "b": b, "m": m, "rank": rank}
            del W0, Wp, R0, R1
            _save_panel_e_cache(panel_e_path, r_info)

    # Panels (a) and (b) need the full-FT weights. Load lazily here.
    if not args.skip_svdft:
        full_w, full_fmt = _get_ckpt("full_ft")
    else:
        full_w, full_fmt = None, None

    # =========================================================================
    # Figure 1 (motivating_svdft): panels a / b / c
    # =========================================================================
    if not args.skip_svdft:
        print("Assembling motivating_svdft...")
        fig1 = plt.figure(figsize=(17, 5.5))
        gs1 = gridspec.GridSpec(1, 3, figure=fig1,
                                width_ratios=[1.1, 1.1, 1.0],
                                wspace=0.40,
                                top=0.84, bottom=0.13, left=0.05, right=0.97)

        ax_a = fig1.add_subplot(gs1[0, 0])
        gs_b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs1[0, 1],
                                                height_ratios=[1, 1.5],
                                                hspace=0.45)
        ax_b_top = fig1.add_subplot(gs_b[0, 0])
        ax_b_bot = fig1.add_subplot(gs_b[1, 0])
        ax_c = fig1.add_subplot(gs1[0, 2])

        plot_panel_a(ax_a, CONFIG["proj_energy_csv"],
                     CONFIG["rect_out"], CONFIG["rect_in"],
                     panel_prefix="(a) ",
                     show_annotations=not args.no_titles)

        plot_panel_b(ax_b_top, ax_b_bot, base_index, full_w, full_fmt,
                     CONFIG["panel_b_layer"], CONFIG["panel_b_module"], device,
                     panel_prefix="(b) ",
                     show_subtitle=not args.no_titles)

        methods_c = ["Full FT", "SVD FT"]
        plot_grouped_bars(
            ax_c,
            methods=methods_c,
            math_vals={m: CONFIG["bars"]["math"].get(m) for m in methods_c},
            cs_vals={m: CONFIG["bars"]["commonsense"].get(m) for m in methods_c},
            pretrained_cs=CONFIG["bars"]["commonsense"]["Pretrained"],
            pretrained_math=CONFIG["bars"]["math"].get("Pretrained"),
            title="",
        )

        # Aligned panel titles at uniform figure-y (suppressed with --no-titles).
        if not args.no_titles:
            _render_panel_titles(fig1, gs1, [
                "(a) Fraction in col(U)",
                r"(b) Singular value change $\mathbf{S}_i'$ - $\mathbf{S}_i$",
                "(c) MATH-10K Training",
            ], y_fig=0.94)

        out = Path(args.output_svdft)
        if args.no_titles:
            out = out.with_name(out.stem + "_notitles" + out.suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(str(out))
        fig1.savefig(str(out.with_suffix(".pdf" if out.suffix == ".png" else ".png")))
        print(f"Saved: {out}")
        plt.close(fig1)

    # =========================================================================
    # Figure 2 (motivating_fura): panels d / e / f relabeled as a / b / c
    # =========================================================================
    if not args.skip_fura:
        print("Assembling motivating_fura...")
        fig2 = plt.figure(figsize=(17, 5.5))
        gs2 = gridspec.GridSpec(1, 3, figure=fig2,
                                width_ratios=[1.1, 1.1, 1.0],
                                wspace=0.40,
                                top=0.84, bottom=0.13, left=0.05, right=0.97)

        ax_d = fig2.add_subplot(gs2[0, 0])
        ax_f = fig2.add_subplot(gs2[0, 2])

        if not args.skip_d:
            plot_panel_d(ax_d, ranks_per_method, CONFIG["panel_d_modules"],
                         panel_prefix="(a) ")
        else:
            ax_d.set_title("(a) Update effective rank", pad=28)
            ax_d.text(0.5, 0.5, "skipped", ha="center", va="center",
                      transform=ax_d.transAxes)

        if not args.skip_e:
            plot_panel_e_2x2(fig2, gs2[0, 1], r_info,
                             CONFIG["panel_e_layer"],
                             CONFIG["panel_e_module"],
                             panel_prefix="(b) ")
        else:
            ax_e_skip = fig2.add_subplot(gs2[0, 1])
            ax_e_skip.set_title("(b) FuRA update pattern", pad=28)
            ax_e_skip.text(0.5, 0.5, "skipped", ha="center", va="center",
                           transform=ax_e_skip.transAxes)

        methods_f = ["Full FT", "LoRA", "FuRA"]
        plot_grouped_bars(
            ax_f,
            methods=methods_f,
            math_vals={m: CONFIG["bars"]["math"].get(m) for m in methods_f},
            cs_vals={m: CONFIG["bars"]["commonsense"].get(m) for m in methods_f},
            pretrained_cs=CONFIG["bars"]["commonsense"]["Pretrained"],
            pretrained_math=CONFIG["bars"]["math"].get("Pretrained"),
            title="",
        )

        # Aligned panel titles at uniform figure-y (suppressed with --no-titles).
        if not args.no_titles:
            _render_panel_titles(fig2, gs2, [
            "(a) Update effective rank",
            "(b) FuRA update pattern",
            "(c) MATH-10K Training",
        ], y_fig=0.94)

        out = Path(args.output_fura)
        if args.no_titles:
            out = out.with_name(out.stem + "_notitles" + out.suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(str(out))
        fig2.savefig(str(out.with_suffix(".pdf" if out.suffix == ".png" else ".png")))
        print(f"Saved: {out}")
        plt.close(fig2)


if __name__ == "__main__":
    main()
