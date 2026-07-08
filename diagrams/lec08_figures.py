"""Generate figures for Lecture 8: Modern CNNs & Transfer Learning.

Matches the lec00 CREAM palette (warm parchment + ink, rust/sage/slate) so these
new matplotlib figures sit beside the hand-authored lec08 SVGs without clashing.
Outputs SVG into figures/lec08/svg/.

Three high-value teaching figures for the Ng-style build-up:
  1. degradation_curve   — deeper PLAIN nets get WORSE (the problem to fix)
  2. residual_block_output — output-first: a block outputs x + F(x); F=0 is free identity
  3. transfer_reuse      — learn phi once, reuse it, swap a cheap head per task
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from pathlib import Path

# ---- palette (from anthropic-theme.css) -------------------------------------
PAPER = "#F7F3E9"
INK = "#161513"
MUTED = "#5F5C54"
RUST = "#B85A3E"
SAGE = "#5F8573"
SLATE = "#37535F"
SAGE_FILL = "#9FB8AC"
SLATE_FILL = "#DCE3E6"
RUST_FILL = "#EAD3C5"
OCHRE = "#C9A14A"

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

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec08" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name, tight=True):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {name}")


def _box(ax, x, y, w, h, fc, ec, text, tc=INK, fs=12, weight="normal", pad=0.12):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=0.08",
        fc=fc, ec=ec, lw=1.6, mutation_aspect=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight=weight, zorder=3)


# ---- 1. the degradation problem ---------------------------------------------
def degradation_curve():
    """He et al. 2015 Fig 1 in the CREAM palette: deeper PLAIN nets get WORSE
    on the TRAINING set (not overfitting) — but deeper ResNets get better."""
    t = np.linspace(0, 6, 300)  # iterations, x1e4

    # plain nets: the 56-layer floor sits ABOVE the 20-layer floor
    plain20 = 7.6 + 13.0 * np.exp(-t / 1.05)
    plain56 = 10.9 + 13.0 * np.exp(-t / 0.95)
    # resnets: the 56-layer floor now sits BELOW the 20-layer floor
    res20 = 7.3 + 13.0 * np.exp(-t / 1.05)
    res56 = 5.6 + 13.0 * np.exp(-t / 1.0)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True,
                                   layout="constrained")

    for ax, (c20, c56, title) in zip(
            (axL, axR),
            [(plain20, plain56, "Plain nets  ·  deeper is WORSE"),
             (res20, res56, "ResNets  ·  deeper is BETTER")]):
        ax.plot(t, c20, color=SLATE, lw=2.6, label="20-layer")
        ax.plot(t, c56, color=RUST, lw=2.6, label="56-layer")
        ax.set_title(title, fontsize=13, color=INK, pad=8)
        ax.set_xlabel(r"iterations  ($\times 10^4$)", fontsize=11)
        ax.set_xlim(0, 6)
        ax.set_ylim(3, 22)
        ax.legend(frameon=False, fontsize=11, loc="upper right")
        _clean(ax)

    axL.set_ylabel("training error (%)", fontsize=11)
    axL.annotate("56-layer sits ABOVE 20-layer\nhigher error even on TRAINING data\nnot overfitting — an optimization failure",
                 xy=(5.6, plain56[-1]), xytext=(1.55, 16.5),
                 color=RUST, fontsize=9.5, ha="left",
                 arrowprops=dict(arrowstyle="-|>", color=RUST, lw=1.3))
    axR.annotate("skips let 56-layer\ngo BELOW 20-layer",
                 xy=(5.6, res56[-1]), xytext=(2.4, 12.6),
                 color=SAGE, fontsize=9.5, ha="left",
                 arrowprops=dict(arrowstyle="-|>", color=SAGE, lw=1.3))

    fig.suptitle("The degradation problem — extra depth hurt, until skip connections fixed it",
                 fontsize=13.5, color=INK)
    save(fig, "degradation_curve.svg", tight=False)


# ---- 2. output-first residual block -----------------------------------------
def residual_block_output():
    """Output-first framing: a residual block outputs x + F(x); if F(x)=0 the
    output is x exactly — a free identity. That is why depth stops hurting."""
    fig, ax = plt.subplots(figsize=(9.8, 3.5))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0.55, 4.5)

    ymain = 3.0
    # input
    ax.add_patch(Circle((0.85, ymain), 0.32, fc=PAPER, ec=INK, lw=1.6, zorder=3))
    ax.text(0.85, ymain, r"$x$", ha="center", va="center", fontsize=15, zorder=4)
    ax.text(0.85, ymain + 0.55, "input", ha="center", color=MUTED, fontsize=10)

    # F(x) path: weight -> ReLU -> weight
    _box(ax, 2.1, ymain - 0.42, 1.7, 0.84, SLATE_FILL, SLATE, "weight\nlayer", fs=11)
    _box(ax, 4.15, ymain - 0.36, 1.0, 0.72, "#F7EED6", OCHRE, "ReLU", fs=11)
    _box(ax, 5.55, ymain - 0.42, 1.7, 0.84, SLATE_FILL, SLATE, "weight\nlayer", fs=11)
    ax.text(4.65, ymain - 0.78, r"$F(x)$  — the two weight layers (what the block computes)",
            ha="center", color=SLATE, fontsize=11, fontstyle="italic")

    # add node
    ax.add_patch(Circle((8.35, ymain), 0.34, fc=RUST_FILL, ec=RUST, lw=1.8, zorder=3))
    ax.text(8.35, ymain, "+", ha="center", va="center", fontsize=18, zorder=4)

    # output
    _box(ax, 9.35, ymain - 0.42, 2.3, 0.84, SAGE_FILL, SAGE,
         r"output $= x + F(x)$", fs=12.5, weight="bold")

    # main-path arrows
    arr = dict(arrowstyle="-|>", mutation_scale=14, lw=1.6, color=INK, zorder=2)
    ax.add_patch(FancyArrowPatch((1.17, ymain), (2.05, ymain), **arr))
    ax.add_patch(FancyArrowPatch((3.8, ymain), (4.1, ymain), **arr))
    ax.add_patch(FancyArrowPatch((5.15, ymain), (5.5, ymain), **arr))
    ax.add_patch(FancyArrowPatch((7.25, ymain), (8.0, ymain), **arr))
    ax.add_patch(FancyArrowPatch((8.7, ymain), (9.3, ymain), **arr))

    # identity skip (rust highway)
    ax.add_patch(FancyArrowPatch(
        (0.85, ymain - 0.34), (8.35, ymain - 0.34),
        connectionstyle="arc3,rad=-0.32", arrowstyle="-|>", mutation_scale=15,
        lw=2.4, color=RUST, zorder=1))
    ax.text(4.6, 4.28, "identity skip  ·  carry  $x$  through unchanged",
            ha="center", color=RUST, fontsize=11.5, fontstyle="italic")

    # the punchline callout
    ax.add_patch(FancyBboxPatch(
        (1.6, 0.95), 8.8, 0.62, boxstyle="round,pad=0.06,rounding_size=0.1",
        fc="#F6D4CC", ec=RUST, lw=1.4, zorder=2))
    ax.text(6.0, 1.26,
            r"Set  $F(x)=0$  $\Rightarrow$  output $= x$ exactly."
            "   A block can become a no-op for FREE — identity is the default fallback.",
            ha="center", va="center", color=INK, fontsize=11.2, zorder=3)

    save(fig, "residual_block_output.svg")


# ---- 3. transfer learning: learn phi once, reuse it -------------------------
def transfer_reuse():
    """Reuse the learned features phi. One frozen ImageNet backbone feeds many
    small, cheap, task-specific heads — that is transfer learning in one picture."""
    fig, ax = plt.subplots(figsize=(10.4, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0.45, 4.25)

    # frozen backbone (big slate box)
    _box(ax, 0.4, 1.15, 3.9, 2.2, SLATE_FILL, SLATE, "", fs=12)
    ax.text(2.35, 3.02, "FROZEN backbone  $\\phi$", ha="center", color=SLATE,
            fontsize=13, fontweight="bold")
    ax.text(2.35, 2.55, "pretrained on 1.2M ImageNet images", ha="center",
            color=MUTED, fontsize=9.5, fontstyle="italic")
    ax.text(2.35, 2.0, r"edges $\to$ textures $\to$ parts", ha="center", color=INK,
            fontsize=11)
    ax.text(2.35, 1.5, "generic — works for almost any image", ha="center",
            color=MUTED, fontsize=9.5)
    ax.text(2.35, 0.72, "learned ONCE · never retrained", ha="center",
            color=SLATE, fontsize=10, fontstyle="italic")

    # feature bus
    ax.add_patch(FancyArrowPatch((4.35, 2.25), (5.35, 2.25),
                 arrowstyle="-|>", mutation_scale=16, lw=2.0, color=INK))
    ax.text(4.85, 2.55, "features", ha="center", color=MUTED, fontsize=9.5)
    ax.plot([5.5, 5.5], [0.95, 3.55], color=MUTED, lw=1.4, zorder=1)

    # swappable heads (sage)
    heads = [
        (3.55, r"head $\to$ flowers (102)"),
        (2.25, r"head $\to$ plant disease (5)"),
        (0.95, r"head $\to$ your task (N)"),
    ]
    for cy, lab in heads:
        ax.add_patch(FancyArrowPatch((5.5, cy), (6.35, cy),
                     arrowstyle="-|>", mutation_scale=13, lw=1.5, color=SAGE))
        _box(ax, 6.4, cy - 0.34, 4.9, 0.68, SAGE_FILL, SAGE, lab, fs=11.5)

    ax.text(8.85, 4.18, "train ONLY these — small & cheap", ha="center",
            color=SAGE, fontsize=10.5, fontstyle="italic")

    fig.suptitle("Transfer learning — learn $\\phi$ once, reuse it, swap a fresh head per task",
                 fontsize=13.5, color=INK, y=1.0)
    save(fig, "transfer_reuse.svg", tight=False)


if __name__ == "__main__":
    print("Generating lec08 figures...")
    degradation_curve()
    residual_block_output()
    transfer_reuse()
    print("Done.")
