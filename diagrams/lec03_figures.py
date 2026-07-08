"""Generate figures for Lecture 3: Training Deep Networks in Practice.

Palette + rcParams copied from lec00_figures.py so new matplotlib figures sit
beside the hand-authored lec03 SVGs without clashing (warm parchment + ink).
Outputs SVG into figures/lec03/svg/.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# ---- palette (from anthropic-theme.css) -------------------------------------
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

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec03" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name, tight=True):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {name}")


# ---- learning-curve diagnosis · bias vs variance vs good fit ------------------
def learning_curve_diagnosis():
    """Three side-by-side loss-vs-epoch panels: the two numbers (train, val)
    tell you the fix. Object-first diagnostic in the spirit of Andrew Ng."""
    ep = np.linspace(0, 1, 100)

    # --- high bias / underfit: both curves plateau HIGH and close together
    tr_bias = 0.60 + 0.34 * np.exp(-6 * ep)
    va_bias = 0.66 + 0.32 * np.exp(-6 * ep)

    # --- high variance / overfit: train dives low, val bottoms then climbs
    tr_var = 0.08 + 0.82 * np.exp(-5 * ep)
    va_var = 0.45 + 0.45 * np.exp(-7 * ep) + 0.42 * (ep ** 2)

    # --- good fit: both drop low, small persistent gap
    tr_good = 0.12 + 0.78 * np.exp(-5 * ep)
    va_good = 0.20 + 0.75 * np.exp(-4.6 * ep)

    panels = [
        ("HIGH BIAS", "underfit", tr_bias, va_bias,
         "both HIGH & flat",
         "add capacity ·\ntrain longer · lower reg", RUST),
        ("HIGH VARIANCE", "overfit", tr_var, va_var,
         "big GAP · val climbs",
         "more data · aug ·\nregularize · early-stop", RUST),
        ("GOOD FIT", "ship it", tr_good, va_good,
         "both LOW, small gap",
         "tune / stop here", SAGE),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.7))
    for ax, (name, tag, tr, va, read, fix, accent) in zip(axes, panels):
        ax.plot(ep, tr, color=SLATE, lw=2.6, label="train", zorder=3)
        ax.plot(ep, va, color=RUST, lw=2.6, label="val", zorder=3)

        # shade the train-val gap so the "read" is visual
        ax.fill_between(ep, tr, va, color=accent, alpha=0.14, zorder=1)

        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("epochs", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)

        # panel name (bold) + fit tag (muted), stacked above the axes
        ax.text(0.0, 1.20, name, transform=ax.transAxes,
                fontsize=13.5, color=INK, ha="left", fontweight="bold")
        ax.text(0.0, 1.05, f"({tag})", transform=ax.transAxes,
                fontsize=11, color=MUTED, ha="left")
        # signal read-off (the "what the plot tells you")
        ax.text(0.5, 0.88, read, transform=ax.transAxes, fontsize=11.5,
                color=INK, ha="center", style="italic")
        # the matching fix, in an accent chip near the bottom
        ax.text(0.5, 0.055, fix, transform=ax.transAxes, fontsize=10.5,
                color=INK, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", fc=PAPER,
                          ec=accent, lw=1.4))

    axes[0].set_ylabel("loss", fontsize=12)
    # high-bias panel keeps its curves up high → the lower strip is free
    axes[0].legend(frameon=False, loc="lower left", fontsize=10.5, ncol=2,
                   bbox_to_anchor=(0.01, 0.30), columnspacing=1.2,
                   handlelength=1.4)

    fig.suptitle("Read the two numbers first — train vs val tells you the fix",
                 fontsize=15, x=0.02, ha="left", y=1.06, color=INK,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "learning_curve_diagnosis.svg", tight=False)


if __name__ == "__main__":
    learning_curve_diagnosis()
    print("done")
