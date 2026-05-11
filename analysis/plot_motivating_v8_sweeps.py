"""Sweep variants of the motivating figures across layers / modules.

Three outputs (each a separate file):

  motivating_svdft_a_sweep.png
    Panel (a) variant: u/v-proj fraction vs training step.
    Layout: 2 rows (L15, L31) × 3 cols (gate_proj u-proj, up_proj u-proj,
    down_proj v-proj). Each sub-axes plots G + W' with the random baseline.
    Source: proj_energy.csv from the full-lr_1e-5 projE run.

  motivating_svdft_b_sweep.png
    Panel (b) variant: per-layer S′−S₀ and |⟨U_i′, U_i⟩| scatter, all 7 modules.
    Layout: 4 rows × 7 cols.
        rows 0-1 = layer 15 (S, then U)
        rows 2-3 = layer 31 (S, then U)
        cols     = q, k, v, o, gate, up, down (left → right)
    Source: full-FT projE checkpoint vs base model (offline SVD).

  motivating_fura_a_sweep.png
    Panel (a) of the FuRA figure (effective rank of ΔW vs layer index)
    swept across all 7 module types. Layout: 2 rows × 4 cols (last cell
    blank). Each subplot plots Full FT / FuRA / LoRA curves for ONE module.
    Cached under analysis_results/motivating_v8_cache/panel_d_sweep__*.npz.

CONFIG (paths, font sizes, colors) inherits from analysis/plot_motivating_v8.py.
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

from analysis.plot_motivating_v8 import (  # noqa: E402
    CONFIG,
    COLOR_FULL,
    COLOR_FURA,
    COLOR_LORA,
    COLOR_UP,
    FONT_BASE,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TICK,
    FONT_TITLE,
    compute_S_U_overlap,
    compute_per_layer_rank,
    _module_prefix,
)
from analysis.plot_motivation_figure import (  # noqa: E402
    detect_num_layers,
    get_weight,
    load_checkpoint,
    load_safetensors_index,
    load_tensor_sf,
    resolve_base_model,
)


TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------------------
# Sweep 1: svdft (a) for L15, L31 × {gate u-proj, up u-proj, down v-proj}
# ---------------------------------------------------------------------------

def plot_svdft_a_sweep(csv_path: str, rect_out: int, rect_in: int,
                       output: Path):
    df = pd.read_csv(csv_path)
    steps = df["step"].values
    baseline = (rect_out / rect_in) ** 0.5

    color_g = "#D94801"
    color_w = "#2171B5"

    layers = [15, 31]
    cols = [
        ("gate_proj", "u_proj", "gate_proj  (col(U) energy)"),
        ("up_proj",   "u_proj", "up_proj  (col(U) energy)"),
        ("down_proj", "v_proj", "down_proj  (row(V) energy)"),
    ]

    fig, axes = plt.subplots(len(layers), len(cols),
                              figsize=(16, 7), sharex=True)
    for i, L in enumerate(layers):
        for j, (mod, metric, title) in enumerate(cols):
            ax = axes[i, j]
            g = df[f"L{L}.{mod}.G.{metric}"].values
            w = df[f"L{L}.{mod}.W.{metric}"].values
            ax.plot(steps, w, color=color_w, linewidth=2.4,
                    label="Weight W'")
            ax.plot(steps, g, color=color_g, linewidth=2.4, alpha=0.9,
                    label="Gradient G")
            ax.axhline(baseline, color="gray", linestyle="--", linewidth=1.6,
                       label="Random Matrix")
            ax.set_ylim(0.0, 1.10)
            if i == len(layers) - 1:
                ax.set_xlabel("Training step", fontsize=FONT_LABEL + 2)
            if j == 0:
                ax.set_ylabel(f"L{L}", fontsize=FONT_LABEL + 4,
                              fontweight="bold")
            if i == 0:
                ax.set_title(title, fontsize=FONT_TITLE - 2, fontweight="bold",
                             pad=6)
            ax.grid(True, alpha=0.2, linewidth=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if i == 0 and j == 0:
                ax.legend(loc="lower right", fontsize=FONT_LEGEND - 2,
                          framealpha=0.9, edgecolor="gray")

    fig.suptitle("Fraction in col(U) / row(V) vs training step",
                 fontsize=FONT_TITLE, fontweight="bold", y=0.995)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output))
    fig.savefig(str(output.with_suffix(".pdf")))
    print(f"Saved: {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep 2: svdft (b) for L15, L31 × all 7 modules
# ---------------------------------------------------------------------------

def plot_svdft_b_sweep(base_index, full_w, full_fmt, layers: list[int],
                       device: torch.device, output: Path):
    n_layers = len(layers)
    n_cols = len(TARGET_MODULES)
    # 4 rows × 7 cols: per layer two stacked rows (S then U).
    fig = plt.figure(figsize=(22, 11))
    gs = gridspec.GridSpec(2 * n_layers, n_cols, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.93, bottom=0.07, left=0.05, right=0.99)

    for li, L in enumerate(layers):
        for mi, mod in enumerate(TARGET_MODULES):
            key = f"{_module_prefix(L, mod)}.weight"
            W0 = load_tensor_sf(base_index, key)
            Wp = get_weight(full_w, key, full_fmt)

            ax_top = fig.add_subplot(gs[2 * li, mi])
            ax_bot = fig.add_subplot(gs[2 * li + 1, mi])

            if W0 is None or Wp is None:
                for ax in (ax_top, ax_bot):
                    ax.text(0.5, 0.5, "missing", ha="center", va="center",
                            transform=ax.transAxes)
                continue

            S0, S, sdiff, udot = compute_S_U_overlap(W0.to(device), Wp.to(device))
            idx = np.arange(len(sdiff))

            ax_top.scatter(idx, sdiff, s=4, color=COLOR_FULL, alpha=0.6,
                           edgecolors="none")
            ax_top.axhline(0, color="gray", linewidth=0.4, linestyle=":")
            ax_top.tick_params(labelbottom=False, labelsize=FONT_TICK - 2)
            ax_top.spines["top"].set_visible(False)
            ax_top.spines["right"].set_visible(False)
            ax_top.grid(True, alpha=0.2, linewidth=0.4)
            if li == 0:
                ax_top.set_title(mod, fontsize=FONT_TITLE - 2,
                                 fontweight="bold", pad=4)
            if mi == 0:
                ax_top.set_ylabel(f"L{L}\n$S_i'$ - $S_i$",
                                   fontsize=FONT_LABEL,
                                   fontweight="bold")

            ax_bot.scatter(idx, udot, s=4, color=COLOR_UP, alpha=0.6,
                           edgecolors="none")
            ax_bot.axhline(1.0, color="gray", linewidth=0.4, linestyle=":")
            ax_bot.set_ylim(-0.05, 1.10)
            ax_bot.tick_params(labelsize=FONT_TICK - 2)
            ax_bot.spines["top"].set_visible(False)
            ax_bot.spines["right"].set_visible(False)
            ax_bot.grid(True, alpha=0.2, linewidth=0.4)
            ax_bot.set_xlabel("singular index i", fontsize=FONT_LABEL - 2)
            if mi == 0:
                ax_bot.set_ylabel(r"$\mathbf{U}_i'$ - $\mathbf{U}_i$",
                                   fontsize=FONT_LABEL,
                                   fontweight="bold")

    fig.suptitle("Per-module singular-spectrum / vector update for L15 and L31",
                 fontsize=FONT_TITLE, fontweight="bold", y=0.995)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output))
    fig.savefig(str(output.with_suffix(".pdf")))
    print(f"Saved: {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep 3: fura (a) effective-rank, one subplot per module
# ---------------------------------------------------------------------------

def _fura_sweep_cache_path(cache_dir: Path, ckpts: dict) -> Path:
    from analysis.plot_motivating_v8 import _ckpt_tag
    parts = [_ckpt_tag(ckpts[k]) for k in ("full_ft", "fura", "lora")]
    return cache_dir / ("panel_d_sweep__" + "__".join(parts) + "__all7.npz")


def _save_d_sweep_cache(path: Path,
                         ranks: dict[str, dict[str, np.ndarray]]):
    flat = {}
    for method, per_mod in ranks.items():
        for mod, arr in per_mod.items():
            flat[f"{method}__{mod}"] = arr
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **flat)
    print(f"[cache] saved fura-(a)-sweep → {path}")


def _load_d_sweep_cache(path: Path) -> dict[str, dict[str, np.ndarray]]:
    z = np.load(str(path))
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in z.files:
        method, mod = key.split("__", 1)
        out.setdefault(method, {})[mod] = z[key]
    print(f"[cache] loaded fura-(a)-sweep ← {path}")
    return out


def plot_fura_a_sweep(base_index, ckpts: dict, num_layers: int,
                      device: torch.device, cache_dir: Path,
                      recompute: bool, output: Path):
    cache_path = _fura_sweep_cache_path(cache_dir, ckpts)

    modules = list(TARGET_MODULES)

    if cache_path.exists() and not recompute:
        ranks = _load_d_sweep_cache(cache_path)
    else:
        full_w, full_fmt = load_checkpoint(ckpts["full_ft"])
        fura_w, fura_fmt = load_checkpoint(ckpts["fura"])
        lora_w, lora_fmt = load_checkpoint(ckpts["lora"])
        print("Computing per-layer rank curves for ALL 7 modules × 3 methods...")
        ranks = {
            "Full FT": compute_per_layer_rank(
                base_index, full_w, full_fmt, num_layers,
                modules, device, "full"),
            "FuRA": compute_per_layer_rank(
                base_index, fura_w, fura_fmt, num_layers,
                modules, device, "fura"),
            "LoRA": compute_per_layer_rank(
                base_index, lora_w, lora_fmt, num_layers,
                modules, device, "lora"),
        }
        _save_d_sweep_cache(cache_path, ranks)

    method_colors = {"Full FT": COLOR_FULL, "FuRA": COLOR_FURA, "LoRA": COLOR_LORA}
    method_styles = {"Full FT": "-", "FuRA": "--", "LoRA": ":"}

    n = len(modules)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 9), sharex=True)
    axes = axes.flatten()

    for i, mod in enumerate(modules):
        ax = axes[i]
        for method, per_mod in ranks.items():
            vals = per_mod.get(mod)
            if vals is None:
                continue
            layers = np.arange(len(vals))
            ax.plot(layers, vals,
                    color=method_colors.get(method, "gray"),
                    linestyle=method_styles.get(method, "-"),
                    marker="o", markersize=3, linewidth=1.5,
                    label=method)
        ax.set_title(mod, fontsize=FONT_TITLE - 2, fontweight="bold")
        ax.set_xlabel("Layer index", fontsize=FONT_LABEL)
        ax.grid(True, alpha=0.2, linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.legend(loc="best", fontsize=FONT_LEGEND - 2,
                      framealpha=0.9, edgecolor="gray")

    # Hide unused cells.
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Update effective rank — sweep over all module types",
                 fontsize=FONT_TITLE, fontweight="bold", y=0.995)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output))
    fig.savefig(str(output.with_suffix(".pdf")))
    print(f"Saved: {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-svdft-a", default="docs/exp_results/figs/motivating_svdft_a_sweep.png")
    p.add_argument("--out-svdft-b", default="docs/exp_results/figs/motivating_svdft_b_sweep.png")
    p.add_argument("--out-fura-a",  default="docs/exp_results/figs/motivating_fura_a_sweep.png")
    p.add_argument("--cache-dir", default=CONFIG["cache_dir"])
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--only", choices=["svdft_a", "svdft_b", "fura_a"], default=None,
                   help="Build just one figure")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(CONFIG["device"])

    # svdft_a needs only the proj_energy.csv (no model loads).
    if args.only in (None, "svdft_a"):
        print("=== Building svdft (a) sweep (L15, L31 × gate/up/down) ===")
        plot_svdft_a_sweep(CONFIG["proj_energy_csv"],
                           CONFIG["rect_out"], CONFIG["rect_in"],
                           Path(args.out_svdft_a))

    # svdft_b and fura_a both need base model + full FT ckpt at minimum.
    base_index = None
    num_layers = None
    full_w = full_fmt = None

    if args.only in (None, "svdft_b", "fura_a"):
        print(f"Resolving base model: {CONFIG['base_model']}")
        base_path = resolve_base_model(CONFIG["base_model"])
        base_index = load_safetensors_index(base_path)
        num_layers = detect_num_layers(base_index)
        print(f"Base: {CONFIG['base_model']} ({num_layers} layers)")

    if args.only in (None, "svdft_b"):
        print("=== Building svdft (b) sweep (L15, L31 × all 7 modules) ===")
        if full_w is None:
            print("Loading full FT ckpt...")
            full_w, full_fmt = load_checkpoint(CONFIG["ckpts"]["full_ft"])
        plot_svdft_b_sweep(base_index, full_w, full_fmt,
                           layers=[15, 31], device=device,
                           output=Path(args.out_svdft_b))

    if args.only in (None, "fura_a"):
        print("=== Building fura (a) sweep (rank vs layer × 7 modules) ===")
        plot_fura_a_sweep(base_index, CONFIG["ckpts"], num_layers, device,
                          Path(args.cache_dir), args.recompute,
                          Path(args.out_fura_a))


if __name__ == "__main__":
    main()
