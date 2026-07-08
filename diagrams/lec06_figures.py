"""Generate figures for Lecture 6: Regularization in Deep Learning.

Palette + rcParams copied from lec00_figures.py so these new matplotlib
figures sit beside the hand-authored lec06 SVGs without clashing.
Outputs SVG into figures/lec06/svg/.

These COMPLEMENT the existing lec06 figures (dropout_network, dropout_masks,
bn_vs_ln_vs_rms, double_descent, label_smoothing_bars, ...) — they do not
duplicate them:
  * reg_menu       — the data -> architectural -> classical "menu" (the spine).
  * dropout_effect — effect-first: what a layer OUTPUTS at train vs eval.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from pathlib import Path

# ---- palette (from anthropic-theme.css / lec00_figures.py) ------------------
PAPER = "#F7F3E9"
INK = "#161513"
MUTED = "#5F5C54"
RUST = "#B85A3E"
SAGE = "#5F8573"
SLATE = "#37535F"
SAGE_FILL = "#9FB8AC"

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["EB Garamond", "Georgia", "DejaVu Serif"],
    "font.size": 13,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "savefig.bbox": "tight",
    "mathtext.fontset": "cm",
})

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec06" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name, tight=True):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {name}")


# ---- 1. the regularization menu: data -> architectural -> classical ---------
def reg_menu():
    fig, ax = plt.subplots(figsize=(12.4, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 6.2)

    families = [
        dict(x=0.35, color=RUST, title="1 · DATA-CENTRIC",
             tag="new for DL",
             items=["Augmentation", " (flip, crop, RandAugment)",
                    "Mixup / CutMix", "Label smoothing"],
             note="rewrite the data"),
        dict(x=4.35, color=SAGE, title="2 · ARCHITECTURAL",
             tag="new for DL",
             items=["Dropout", "BatchNorm / LayerNorm", "RMSNorm",
                    "(pre-norm placement)"],
             note="rewire the network"),
        dict(x=8.35, color=SLATE, title="3 · CLASSICAL / PENALTY",
             tag="you know (ES 335)",
             items=["L2 = weight decay", "L1  (rare in DL)", "Early stopping"],
             note=r"add a $\lambda\|\theta\|$ term"),
    ]
    cw = 3.7

    ax.text(6.2, 5.95, "The regularization menu — three families, one spine",
            ha="center", va="center", fontsize=15.5, color=INK, style="italic")

    for f in families:
        x = f["x"]
        # header band
        ax.add_patch(FancyBboxPatch((x, 4.55), cw, 0.72,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     fc=f["color"], ec="none"))
        ax.text(x + cw / 2, 4.91, f["title"], ha="center", va="center",
                color=PAPER, fontsize=13.5, weight="bold")
        # body
        ax.add_patch(FancyBboxPatch((x, 1.35), cw, 3.05,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     fc="#EFE7D4", ec=f["color"], lw=1.4))
        # items
        yy = 3.98
        for it in f["items"]:
            if it.startswith(" "):  # continuation / sub-line
                ax.text(x + 0.55, yy + 0.12, it.strip(), ha="left", va="center",
                        color=MUTED, fontsize=10.5, style="italic")
                yy -= 0.5
            else:
                ax.text(x + 0.3, yy, "•", ha="left", va="center",
                        color=f["color"], fontsize=15)
                ax.text(x + 0.6, yy, it, ha="left", va="center",
                        color=INK, fontsize=12.5)
                yy -= 0.62
        # tag + note
        ax.text(x + cw / 2, 1.62, f["tag"], ha="center", va="center",
                color=f["color"], fontsize=10.5, weight="bold")
        ax.text(x + cw / 2, 5.45, f["note"], ha="center", va="center",
                color=MUTED, fontsize=11, style="italic")

    # bottom "reach-for order" arrow spanning all three
    ax.add_patch(FancyArrowPatch((0.55, 0.72), (11.8, 0.72),
                 arrowstyle="-|>", mutation_scale=18, color=MUTED, lw=1.6))
    ax.text(6.2, 0.34, "reach for data-centric first,  then architectural,  then classical",
            ha="center", va="center", color=INK, fontsize=11.5)

    save(fig, "reg_menu.svg")


# ---- 2. dropout effect-first: what a layer OUTPUTS (train vs eval) -----------
def dropout_effect():
    h = np.array([2.0, 1.5, 0.5, 3.0])
    mask = np.array([1, 0, 1, 0])
    p = 0.5
    train = h * mask / p          # [4.0, 0, 1.0, 0]
    idx = np.arange(4)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), layout="constrained",
                             sharey=True)
    labels = ["u1", "u2", "u3", "u4"]

    titles = [
        "INPUT activations  h",
        r"TRAIN output  $(h\odot m)/p$",
        "EVAL output  (unchanged)",
    ]
    data = [h, train, h]
    colors = [SLATE, RUST, SAGE]

    for ax, title, d, c in zip(axes, titles, data, colors):
        bars = ax.bar(idx, d, color=c, edgecolor=MUTED, lw=0.7, width=0.66)
        ax.set_title(title, fontsize=12.5, color=INK, pad=8)
        ax.set_xticks(idx)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 4.5)
        for xi, v in zip(idx, d):
            if v == 0:
                ax.text(xi, 0.12, "×", ha="center", va="bottom",
                        color=MUTED, fontsize=16, weight="bold")
            else:
                ax.text(xi, v + 0.1, f"{v:g}", ha="center", va="bottom",
                        color=INK, fontsize=11)
        _clean(ax)

    # annotate the amplification on the train panel
    axes[1].annotate("survivors scaled\n×1/p = ×2", xy=(0, 4.0), xytext=(1.15, 3.9),
                     ha="left", va="center", color=RUST, fontsize=10.5,
                     arrowprops=dict(arrowstyle="-|>", color=RUST, lw=1.2))
    axes[0].set_ylabel("activation", fontsize=11)

    fig.suptitle(r"Dropout, effect-first: mask + rescale at train, full pass at eval  —  "
                 r"same expected output ($E[\,\cdot\,]=h$) both ways",
                 fontsize=13, color=INK)
    save(fig, "dropout_effect.svg", tight=False)


if __name__ == "__main__":
    print("Generating lec06 figures...")
    reg_menu()
    dropout_effect()
    print("Done.")
