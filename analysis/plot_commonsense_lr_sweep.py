"""Plot LR sweep curves for commonsense_170k 1-epoch on Meta-Llama-3-8B.

Single-panel figure: x = learning rate (log scale), y = mean accuracy across
the 8 commonsense tasks (BoolQ / PIQA / SIQA / ARC-C / ARC-E / OBQA /
HellaSwag / Wino). Five series:
  - Full FT (all params)
  - LoRA 5-mod (q/k/v/up/down — LoRA paper default)
  - LoRA 7-mod (LoRA + o_proj + gate_proj)
  - FuRA 7-mod (default BTT — all 7 linear types)
  - FuRA 5-mod_lora (BTT restricted to LoRA's 5-mod set)

Data hard-coded from the "LR Sweep — 1-Epoch Full 8-Task Eval" section of
docs/exp_results/lift_commonsense.md so the script doesn't need to re-parse
eval logs every time. Update both this file and the markdown together.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "exp_results" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- LR sweep, 1 epoch, commonsense_170k, Llama-3-8B, seed=43 -----
# entries: (lr, avg_acc) — Avg = mean over 8 tasks.
LR_SWEEP_FULL = [
    (7e-6, 87.88),
    (1e-5, 88.53),
    (3e-5, 83.79),
    (5e-5, 37.01),  # collapse: near-random on PIQA/SIQA/ARC-C/ARC-E/HellaSwag.
]
LR_SWEEP_LORA_5MOD = [
    (5e-5, 88.24),
    (7e-5, 88.43),
    (1e-4, 87.60),
    (2e-4, 82.89),
    # 3e-4: training killed at step 700 — no checkpoint.
]
LR_SWEEP_LORA_7MOD = [
    (5e-5, 88.87),
    (7e-5, 88.50),
]
LR_SWEEP_FURA_7MOD = [
    (7e-5, 82.89),
    (1e-4, 85.96),
    (2e-4, 88.06),
    (4e-4, 86.36),
    (6e-4, 84.56),
    (8e-4, 79.53),
]
LR_SWEEP_FURA_5MOD = [
    (1e-4, 85.26),
    (2e-4, 87.48),
    (3e-4, 87.75),
    (4e-4, 87.10),
    (5e-4, 87.20),
]

STYLE = {
    "lora_5mod": dict(color="#1f77b4", marker="o", linestyle="-",  label="LoRA 5-mod"),
    "lora_7mod": dict(color="#1f77b4", marker="o", linestyle="--", label="LoRA 7-mod", markerfacecolor="white"),
    "fura_7mod": dict(color="#d62728", marker="s", linestyle="-",  label="FuRA 7-mod"),
    "fura_5mod": dict(color="#d62728", marker="s", linestyle="--", label="FuRA 5-mod_lora", markerfacecolor="white"),
    "full":      dict(color="#2ca02c", marker="^", linestyle="-",  label="Full FT"),
}


def plot_lr_sweep() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    series = [
        (LR_SWEEP_FULL, "full"),
        (LR_SWEEP_LORA_5MOD, "lora_5mod"),
        (LR_SWEEP_LORA_7MOD, "lora_7mod"),
        (LR_SWEEP_FURA_7MOD, "fura_7mod"),
        (LR_SWEEP_FURA_5MOD, "fura_5mod"),
    ]
    # Annotate each series' peak with the Avg value.
    label_offsets = {
        "full":      (0, 10),
        "lora_5mod": (0, 10),
        "lora_7mod": (0, 12),
        "fura_7mod": (0, 10),
        "fura_5mod": (0, -16),
    }
    for entries, key in series:
        lrs = [e[0] for e in entries]
        acc = [e[1] for e in entries]
        ax.plot(lrs, acc, **STYLE[key])

        best_idx = max(range(len(entries)), key=lambda i: entries[i][1])
        peak_lr, peak_acc = entries[best_idx]
        ax.annotate(
            f"{peak_acc:.2f}",
            xy=(peak_lr, peak_acc),
            xytext=label_offsets[key],
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color=STYLE[key]["color"],
            ha="center",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Commonsense Avg accuracy (%)")
    ax.set_title("LR sweep — 1 epoch on commonsense_170k (Llama-3-8B)")
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False, loc="lower left", fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "commonsense_lr_sweep_1ep.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_lr_sweep()
