"""Plot LR sweep (bsz=16) and batch-size sweep curves for math_10k.

Two single-panel figures, both GSM8K-only:
- math_lr_sweep_bsz16.png  — LR vs gsm8k acc (full FT, LoRA, FuRA).
- math_bsz_sweep.png       — bsz vs gsm8k acc, best LR per (method, bsz).

Data hard-coded from docs/exp_results/lift_math.md so the script doesn't need
to re-parse eval logs every time. Update both this file and the markdown together.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "exp_results" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- LR sweep at bsz=16 -----
# entries: (lr, gsm8k_acc)
LR_SWEEP_LORA = [
    (3e-5, 70.20),
    (6e-5, 70.81),
    (1e-4, 69.83),
    (2e-4, 66.41),
]
LR_SWEEP_FURA = [
    (2e-4, 69.52),
    (3e-4, 71.57),
    (4e-4, 70.43),
    (6e-4, 66.34),
]
LR_SWEEP_FULL = [
    (8e-6, 70.72),
    (1e-5, 71.19),
    (2e-5, 67.70),
    (3e-5, 64.97),
]

# ----- bsz sweep (best LR per bsz) -----
# entries: (bsz, gsm8k_acc)
BSZ_SWEEP_FULL = [
    (16,  71.19),  # bsz=16 best is now lr=1e-5 (71.19) since lr=8e-6 dropped to 70.72
    (64,  72.56),
    (256, 69.07),
]
BSZ_SWEEP_LORA = [
    (16,  70.81),
    (64,  70.36),
    (256, 70.20),
]
BSZ_SWEEP_FURA = [
    (16,  71.57),
    (64,  72.18),
    (256, 71.04),
]

STYLE = {
    "lora": dict(color="#1f77b4", marker="o", linestyle="-",  label="LoRA"),
    "fura": dict(color="#d62728", marker="s", linestyle="-",  label="FuRA"),
    "full": dict(color="#2ca02c", marker="^", linestyle="-",  label="Full FT"),
}


def plot_lr_sweep() -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    for entries, key in [
        (LR_SWEEP_FULL, "full"),
        (LR_SWEEP_LORA, "lora"),
        (LR_SWEEP_FURA, "fura"),
    ]:
        lrs = [e[0] for e in entries]
        acc = [e[1] for e in entries]
        ax.plot(lrs, acc, **STYLE[key])

        # Mark the peak (best GSM8K per method) with a larger annotation.
        best_idx = max(range(len(entries)), key=lambda i: entries[i][1])
        peak_lr, peak_acc = entries[best_idx]
        ax.annotate(
            f"{peak_acc:.2f}",
            xy=(peak_lr, peak_acc),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color=STYLE[key]["color"],
            ha="center",
        )

    ax.set_xscale("log")
    ax.set_ylim(63, 73)
    ax.set_xlabel("Learning rate", fontsize=14)
    ax.set_ylabel("GSM8K accuracy (%)", fontsize=14)
    ax.set_title("LR sweep at effective batch size = 16", fontsize=16)
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False, loc="lower left", fontsize=14)

    fig.tight_layout()
    out_png = OUT_DIR / "math_lr_sweep_bsz16.png"
    out_pdf = OUT_DIR / "math_lr_sweep_bsz16.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    plt.close(fig)


def plot_bsz_sweep() -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    for entries, key in [
        (BSZ_SWEEP_FULL, "full"),
        (BSZ_SWEEP_LORA, "lora"),
        (BSZ_SWEEP_FURA, "fura"),
    ]:
        bsz = [e[0] for e in entries]
        acc = [e[1] for e in entries]
        ax.plot(bsz, acc, **STYLE[key])

    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 64, 256])
    ax.set_xticklabels(["16", "64", "256"])
    ax.set_ylim(63, 73)
    ax.set_xlabel("Effective batch size", fontsize=14)
    ax.set_ylabel("GSM8K accuracy (%)", fontsize=14)
    ax.set_title("Batch-size sweep — best LR per (method, bsz)", fontsize=16)
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False, loc="best", fontsize=14)

    fig.tight_layout()
    out_png = OUT_DIR / "math_bsz_sweep.png"
    out_pdf = OUT_DIR / "math_bsz_sweep.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    plot_lr_sweep()
    plot_bsz_sweep()
