"""Plot LR sweep (bsz=16) and batch-size sweep curves for math_10k.

Two figure files written:
- math_lr_sweep_bsz16.png  — 1×2 panel: (a) LR vs final loss, (b) LR vs gsm8k acc.
- math_bsz_sweep.png       — 1×2 panel: (a) bsz vs final loss, (b) bsz vs gsm8k acc.
   For bsz sweep we pick the best (gsm8k-maximizing) LR per (method, bsz).

Data is hard-coded from docs/exp_results/lift_math.md so the script doesn't need
to re-parse eval logs every time. Update both this file and the markdown together.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "exp_results" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- LR sweep at bsz=16 -----
# entries: (lr, final_loss, gsm8k_acc)
LR_SWEEP_LORA = [
    (6e-5, 0.155, 70.81),
    (8e-5, 0.135, 69.83),
    (1e-4, 0.122, 69.83),
]
LR_SWEEP_BLOCKTT = [
    (8e-5, 0.308, 66.41),
    (1e-4, 0.290, 67.70),
    (2e-4, 0.220, 69.52),
    (3e-4, 0.165, 71.57),
    (4e-4, 0.130, 70.43),
    (6e-4, 0.095, 66.34),
    (8e-4, 0.084, 62.85),
]

# ----- bsz sweep (best LR per bsz) -----
# entries: (bsz, lr_used, final_loss, gsm8k_acc)
BSZ_SWEEP_FULL = [
    (16,  1e-5, 0.176, 71.19),
    (64,  2e-5, 0.135, 72.56),
    (256, 2e-5, 0.237, 69.07),
]
BSZ_SWEEP_LORA = [
    (16,  6e-5, 0.155, 70.81),
    (64,  2e-4, 0.132, 70.36),
    (256, 6e-4, 0.167, 70.20),
]
BSZ_SWEEP_BLOCKTT = [
    (16,  3e-4, 0.165, 71.57),
    (64,  6e-4, 0.127, 72.18),
    (256, 8e-4, 0.212, 71.04),
]

STYLE = {
    "lora":    dict(color="#1f77b4", marker="o", linestyle="-",  label="LoRA"),
    "blocktt": dict(color="#d62728", marker="s", linestyle="-",  label="FuRA (BlockTT)"),
    "full":    dict(color="#2ca02c", marker="^", linestyle="-",  label="Full FT"),
}


def plot_lr_sweep() -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2))

    for entries, key in [(LR_SWEEP_LORA, "lora"), (LR_SWEEP_BLOCKTT, "blocktt")]:
        lrs   = [e[0] for e in entries]
        loss  = [e[1] for e in entries]
        acc   = [e[2] for e in entries]
        ax_loss.plot(lrs, loss, **STYLE[key])
        ax_acc.plot(lrs, acc, **STYLE[key])

    for ax in (ax_loss, ax_acc):
        ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
        ax.legend(frameon=False, loc="best")

    ax_loss.set_ylabel("Final training loss")
    ax_loss.set_title("(a) LR vs final loss (bsz=16)")
    ax_acc.set_ylabel("GSM8K accuracy (%)")
    ax_acc.set_title("(b) LR vs GSM8K (bsz=16)")

    fig.suptitle(
        "Math-10K LR sweep at effective batch size = 16 — "
        "Meta-Llama-3-8B, 3 epochs, seed=43",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "math_lr_sweep_bsz16.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def plot_bsz_sweep() -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2))

    for entries, key in [
        (BSZ_SWEEP_FULL,    "full"),
        (BSZ_SWEEP_LORA,    "lora"),
        (BSZ_SWEEP_BLOCKTT, "blocktt"),
    ]:
        bsz  = [e[0] for e in entries]
        loss = [e[2] for e in entries]
        acc  = [e[3] for e in entries]
        ax_loss.plot(bsz, loss, **STYLE[key])
        ax_acc.plot(bsz, acc, **STYLE[key])

        # Annotate each point with the LR that produced it.
        for e in entries:
            lr_str = f"lr={e[1]:.0e}".replace("e-0", "e-")
            ax_loss.annotate(lr_str, (e[0], e[2]), textcoords="offset points",
                             xytext=(6, -2), fontsize=7, alpha=0.7)
            ax_acc.annotate(lr_str, (e[0], e[3]), textcoords="offset points",
                            xytext=(6, -2), fontsize=7, alpha=0.7)

    for ax in (ax_loss, ax_acc):
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256])
        ax.set_xticklabels(["16", "64", "256"])
        ax.set_xlabel("Effective batch size")
        ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
        ax.legend(frameon=False, loc="best")

    ax_loss.set_ylabel("Final training loss")
    ax_loss.set_title("(a) Batch size vs final loss")
    ax_acc.set_ylabel("GSM8K accuracy (%)")
    ax_acc.set_title("(b) Batch size vs GSM8K")

    fig.suptitle(
        "Math-10K batch-size sweep — best LR per (method, bsz) — "
        "Meta-Llama-3-8B, 3 epochs, seed=43",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "math_bsz_sweep.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_lr_sweep()
    plot_bsz_sweep()
