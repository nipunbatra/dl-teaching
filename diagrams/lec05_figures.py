"""Generate figures for Lecture 5: Adam, AdamW & LR Schedules.

Matches the lec00 CREAM palette (warm parchment + ink, rust/sage/slate) so new
matplotlib figures sit beside the hand-authored lec05 SVGs without clashing.
Outputs SVG into figures/lec05/svg/.

Only adds figures that fill a genuine gap; it must NOT duplicate the existing
hand-authored set (optimizer_family_tree, adam_components, adam_bias_correction,
adagrad_decay, adamw_vs_adam, optimizer_trajectories, lr_schedule_shapes,
lr_schedules, warmup_necessity, gradient_clipping).
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

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec05" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name, tight=True):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {name}")


# ---- per-coordinate step size: steep vs flat --------------------------------
def per_coordinate_stepsize():
    """Why one global LR is wrong, and how dividing by sqrt(v) fixes it.

    Left  : gradient magnitude |g| per direction (steep is big, flat is small).
    Right : Adam's per-coordinate step size eta/sqrt(v) — inverted vs |g|:
            the steep direction is damped, the flat direction amplified.
            Dashed line marks the single global LR that SGD is stuck with.
    """
    eta = 0.1
    labels = ["Steep\ndirection", "Flat\ndirection"]
    g = np.array([4.0, 0.5])            # gradient magnitude per coordinate
    v = g ** 2                          # RMSProp scale settles near g^2
    eff = eta / np.sqrt(v)              # per-coordinate step size eta/sqrt(v)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    x = np.arange(2)
    cols = [RUST, SLATE]

    # ---- left: how big are the gradients ----
    barsL = axL.bar(x, g, width=0.55, color=cols, edgecolor=INK, lw=0.8)
    for xi, gi in zip(x, g):
        axL.text(xi, gi + 0.12, f"|g| = {gi:g}", ha="center", va="bottom",
                 fontsize=12, color=INK)
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=12)
    axL.set_ylim(0, 4.9)
    axL.set_ylabel("gradient magnitude  |g|")
    axL.set_title("1 · How big are the gradients?", fontsize=13, loc="left",
                  color=INK, pad=8)
    _clean(axL)

    # ---- right: the per-coordinate step size Adam actually takes ----
    axR.bar(x, eff, width=0.55, color=cols, edgecolor=INK, lw=0.8)
    for xi, ei in zip(x, eff):
        axR.text(xi, ei + 0.006, r"$\eta/\sqrt{v}$" + f" = {ei:.3g}",
                 ha="center", va="bottom", fontsize=12, color=INK)
    axR.axhline(eta, ls="--", lw=1.6, color=MUTED)
    axR.text(-0.34, eta + 0.005, "one global η\n(SGD is stuck here)", ha="left",
             va="bottom", fontsize=10.5, color=MUTED)
    axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=12)
    axR.set_ylim(0, 0.235)
    axR.set_ylabel(r"step size  $\eta/\sqrt{v}$")
    axR.set_title("2 · Adam's step size — inverted", fontsize=13, loc="left",
                  color=INK, pad=8)
    _clean(axR)

    fig.suptitle("Big gradients get a SMALL step; small gradients get a BIG step",
                 fontsize=14.5, color=INK, y=1.03)
    fig.tight_layout()
    save(fig, "per_coordinate_stepsize.svg")


if __name__ == "__main__":
    per_coordinate_stepsize()
    print("done.")
