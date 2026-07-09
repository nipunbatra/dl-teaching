"""Generate four missing figures for the new decks (L02, L03, L08, L13).

Reuses the exact cream/parchment palette + rcParams from lec00_figures.py so the
new figures sit beside the existing ones without clashing.
Outputs SVG into figures/Lnew/svg/.

  1. sigmoid.svg                 — logistic curve with asymptotes + 0.5 crossing (L02)
  2. xor_not_separable.svg       — XOR points + failed separating lines       (L02/L03)
  3. bias_variance_dartboard.svg — the classic 2x2 bias/variance dartboard    (L08)
  4. embedding_geometry.svg      — king - man + woman ~ queen parallelogram   (L13)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Polygon
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

OUT = Path(__file__).resolve().parent.parent / "figures" / "Lnew" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name, tight=True):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {name}")


# ---- 1. the logistic (sigmoid) curve ----------------------------------------
def sigmoid():
    z = np.linspace(-6, 6, 400)
    s = 1 / (1 + np.exp(-z))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    # asymptotes at 0 and 1
    ax.axhline(1.0, color=SAGE, ls="--", lw=1.3, alpha=0.9)
    ax.axhline(0.0, color=SAGE, ls="--", lw=1.3, alpha=0.9)
    ax.text(-5.9, 1.03, r"asymptote  $\sigma\to 1$", color=SAGE, fontsize=11, va="bottom")
    ax.text(-5.9, -0.03, r"asymptote  $\sigma\to 0$", color=SAGE, fontsize=11, va="top")

    # the curve
    ax.plot(z, s, color=RUST, lw=2.8, zorder=3)

    # the 0.5 crossing at z = 0
    ax.plot([0, 0], [0, 0.5], ls=":", color=MUTED, lw=1.4, zorder=2)
    ax.plot([-6, 0], [0.5, 0.5], ls=":", color=MUTED, lw=1.4, zorder=2)
    ax.scatter([0], [0.5], s=90, color=SLATE, edgecolor=PAPER, lw=1.4, zorder=4)
    ax.annotate(r"$\sigma(0)=0.5$", xy=(0, 0.5), xytext=(1.1, 0.36),
                color=INK, fontsize=12,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

    ax.text(3.4, 0.90, r"$\sigma(z)=\dfrac{1}{1+e^{-z}}$", color=RUST, fontsize=14)

    ax.set_xlabel(r"$z=\theta^\top x$")
    ax.set_ylabel(r"$\sigma(z)$")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.12, 1.12)
    ax.set_yticks([0, 0.5, 1])
    ax.set_title("The logistic (sigmoid) squash — a number becomes a probability",
                 fontsize=13, loc="left", color=INK, pad=10)
    _clean(ax)
    save(fig, "sigmoid.svg")


# ---- 2. XOR is not linearly separable ---------------------------------------
def xor_not_separable():
    fig, ax = plt.subplots(figsize=(5.6, 5.2))

    # points
    classA = np.array([[0, 0], [1, 1]])   # y = 0
    classB = np.array([[0, 1], [1, 0]])   # y = 1

    # a couple of FAILED separating lines
    xs = np.linspace(-0.55, 1.55, 50)
    ax.plot(xs, 0.5 + 0.0 * xs, color=MUTED, ls="--", lw=1.6, alpha=0.75)      # horizontal
    ax.plot(xs, xs - 0.15, color=MUTED, ls="--", lw=1.6, alpha=0.75)          # diagonal
    ax.plot(0.5 + 0.0 * xs, xs, color=MUTED, ls="--", lw=1.6, alpha=0.75)      # vertical
    ax.text(1.16, 0.83, "every line\nmisclassifies", color=MUTED, fontsize=11,
            style="italic", va="center", ha="left")

    # the points on top
    ax.scatter(classA[:, 0], classA[:, 1], s=340, color=SLATE, marker="o",
               edgecolor=PAPER, lw=2, zorder=5, label=r"class $0$  ($x_1{=}x_2$)")
    ax.scatter(classB[:, 0], classB[:, 1], s=360, color=RUST, marker="X",
               edgecolor=PAPER, lw=1.6, zorder=5, label=r"class $1$  ($x_1{\neq}x_2$)")

    # corner coordinate labels
    for (px, py), lab in [((0, 0), "(0,0)"), ((1, 1), "(1,1)"),
                          ((0, 1), "(0,1)"), ((1, 0), "(1,0)")]:
        dy = 0.13 if py == 0 else -0.17
        ax.text(px, py + dy, lab, ha="center", va="center", color=INK, fontsize=11)

    ax.set_xlim(-0.55, 1.75)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left", fontsize=10.5,
              bbox_to_anchor=(-0.02, 1.0))
    ax.set_title("XOR: the two classes sit on opposite corners —\nno straight line separates them",
                 fontsize=12.5, loc="left", color=INK, pad=10)
    _clean(ax)
    save(fig, "xor_not_separable.svg")


# ---- 3. the bias/variance dartboard (2 x 2) ---------------------------------
def bias_variance_dartboard():
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 8.0))
    rng = np.random.default_rng(7)

    # (row, col): center offset, scatter spread, panel label
    specs = {
        (0, 0): ((0.0, 0.0), 0.16, "low bias · low variance"),      # bullseye
        (0, 1): ((0.0, 0.0), 0.52, "low bias · high variance"),     # centred, spread
        (1, 0): ((0.55, 0.55), 0.16, "high bias · low variance"),   # off-centre, tight
        (1, 1): ((0.6, 0.55), 0.52, "high bias · high variance"),   # off + spread
    }
    rings = [1.0, 0.66, 0.33]
    ring_cols = ["#E6D6CB", "#EFE7D4", PAPER]

    for (r, c), (center, spread, label) in specs.items():
        ax = axes[r, c]
        ax.set_aspect("equal")
        ax.axis("off")

        # dartboard rings
        for rad, col in zip(rings, ring_cols):
            ax.add_patch(Circle((0, 0), rad, facecolor=col, edgecolor=MUTED, lw=1.1, zorder=1))
        ax.add_patch(Circle((0, 0), 0.09, facecolor=SAGE, edgecolor=INK, lw=0.8, zorder=2))
        # crosshair
        ax.plot([-1, 1], [0, 0], color=MUTED, lw=0.7, alpha=0.5, zorder=1)
        ax.plot([0, 0], [-1, 1], color=MUTED, lw=0.7, alpha=0.5, zorder=1)

        # darts
        darts = rng.normal(center, spread, (10, 2))
        ax.scatter(darts[:, 0], darts[:, 1], s=70, color=RUST, marker="^",
                   edgecolor=INK, lw=0.6, zorder=4)

        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.28, 1.15)
        ax.text(0, -1.24, label, ha="center", va="top", color=INK, fontsize=12)

    # column / row super-labels
    axes[0, 0].set_title("Low variance", fontsize=13, color=SLATE, pad=8)
    axes[0, 1].set_title("High variance", fontsize=13, color=SLATE, pad=8)
    axes[0, 0].text(-1.35, 0, "Low bias", rotation=90, ha="center", va="center",
                    color=SLATE, fontsize=13)
    axes[1, 0].text(-1.35, 0, "High bias", rotation=90, ha="center", va="center",
                    color=SLATE, fontsize=13)

    fig.suptitle("Bias vs variance — every dart is one retrain; the bullseye is the truth",
                 fontsize=13, color=INK, y=0.98)
    fig.subplots_adjust(wspace=0.05, hspace=0.18, top=0.92)
    save(fig, "bias_variance_dartboard.svg", tight=False)


# ---- 4. word-embedding analogy geometry -------------------------------------
def embedding_geometry():
    # coords chosen so king - man + woman == queen exactly
    man = np.array([0.8, 0.6])
    king = np.array([2.6, 1.2])          # man -> king is the "royalty" direction
    woman = np.array([1.2, 1.9])
    queen = woman + (king - man)         # = (3.0, 2.5)

    words = {"man": man, "king": king, "woman": woman, "queen": queen}
    cols = {"man": SLATE, "king": RUST, "woman": SLATE, "queen": RUST}

    fig, ax = plt.subplots(figsize=(6.8, 5.4))

    # parallelogram man -> king -> queen -> woman
    poly = Polygon([man, king, queen, woman], closed=True, fill=True,
                   facecolor=SAGE_FILL, alpha=0.22, edgecolor=SAGE, lw=1.6,
                   ls="--", zorder=1)
    ax.add_patch(poly)

    # vectors from the origin to each word
    for w, p in words.items():
        ax.add_patch(FancyArrowPatch((0, 0), tuple(p), arrowstyle="-|>",
                     mutation_scale=14, color=cols[w], lw=2.0, alpha=0.9, zorder=3))
        ax.scatter(*p, s=80, color=cols[w], edgecolor=PAPER, lw=1.4, zorder=4)
        off = (0.06, 0.10) if w in ("king", "queen") else (-0.10, 0.10)
        ha = "left" if w in ("king", "queen") else "right"
        ax.text(p[0] + off[0], p[1] + off[1], w, color=cols[w], fontsize=14,
                ha=ha, va="bottom", fontweight="bold")

    # the "royalty" direction annotations on the parallel edges
    mid_mk = (man + king) / 2
    mid_wq = (woman + queen) / 2
    ax.annotate("", xy=tuple(king), xytext=tuple(man),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.annotate("", xy=tuple(queen), xytext=tuple(woman),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(mid_mk[0] + 0.05, mid_mk[1] - 0.28, r"$+$ royalty", color=INK,
            fontsize=11, style="italic")
    ax.text(mid_wq[0] - 0.15, mid_wq[1] + 0.12, r"$+$ royalty", color=INK,
            fontsize=11, style="italic")

    ax.text(0.15, 3.15, r"$v_{\mathrm{king}} - v_{\mathrm{man}} + v_{\mathrm{woman}} \approx v_{\mathrm{queen}}$",
            color=INK, fontsize=14)

    ax.set_xlim(0, 3.6)
    ax.set_ylim(0, 3.4)
    ax.set_xlabel("embedding dim 1")
    ax.set_ylabel("embedding dim 2")
    ax.set_title("Meaning is geometry — the analogy closes a parallelogram",
                 fontsize=13, loc="left", color=INK, pad=10)
    _clean(ax)
    save(fig, "embedding_geometry.svg")


if __name__ == "__main__":
    print("Generating Lnew figures...")
    sigmoid()
    xor_not_separable()
    bias_variance_dartboard()
    embedding_geometry()
    print("Done.")
