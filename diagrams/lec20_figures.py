"""Generate figures for Lecture 20: GANs.

Palette + rcParams copied from lec00_figures.py (warm parchment + ink,
rust/sage/slate) so new matplotlib figures sit beside the hand-authored
lec20 SVGs without clashing. Outputs SVG into figures/lec20/svg/.

Two figures (complement, do NOT duplicate, the existing hand-SVGs):
  1. js_vs_wasserstein.svg — JS saturates to a flat log 2 (zero gradient)
     while Wasserstein slopes linearly, for two disjoint distributions.
  2. gan_1d_toy.svg — a 1D bimodal target and G's distribution converging
     to it over training steps 0 / 100 / 500 / 1000.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# ---- palette (from anthropic-theme.css / lec00_figures.py) ------------------
PAPER = "#F7F3E9"
INK = "#161513"
MUTED = "#5F5C54"
RUST = "#B85A3E"
SAGE = "#5F8573"
SLATE = "#37535F"
WINE = "#8E2A3B"
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

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec20" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ---- 1. JS saturates vs Wasserstein slopes ----------------------------------
def js_vs_wasserstein():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.3),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # -- Left: two disjoint bumps a distance theta apart ----------------------
    xs = np.linspace(-1.2, 4.2, 400)

    def bump(c, s=0.28):
        return np.exp(-0.5 * ((xs - c) / s) ** 2)

    real = bump(0.0)
    theta = 2.6
    fake = bump(theta)
    axL.fill_between(xs, real, color=SAGE, alpha=0.55, lw=0)
    axL.plot(xs, real, color=SAGE, lw=1.8)
    axL.fill_between(xs, fake, color=RUST, alpha=0.55, lw=0)
    axL.plot(xs, fake, color=RUST, lw=1.8)
    axL.text(0.0, 1.09, r"$p_\mathrm{data}$", color=SAGE, ha="center", fontsize=13)
    axL.text(theta, 1.09, r"$p_G$", color=RUST, ha="center", fontsize=13)

    # distance arrow theta
    yb = -0.14
    axL.annotate("", xy=(theta, yb), xytext=(0.0, yb),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    axL.text(theta / 2, yb - 0.11, r"separation $\theta$",
             ha="center", va="top", color=INK, fontsize=12)
    axL.text(theta / 2, 0.5, "no\noverlap", ha="center", va="center",
             color=MUTED, fontsize=11, style="italic")

    axL.set_ylim(-0.34, 1.28)
    axL.set_xlim(-1.2, 4.2)
    axL.set_yticks([])
    axL.set_xticks([])
    axL.spines[["top", "right", "left"]].set_visible(False)
    axL.spines["bottom"].set_visible(False)
    axL.set_title("Two distributions that don't overlap",
                  fontsize=13, loc="center", color=INK, pad=8)

    # -- Right: distance as a function of theta -------------------------------
    t = np.linspace(0, 3.0, 300)
    js = np.full_like(t, np.log(2))     # constant log 2 for theta > 0
    js[0] = 0.0                          # only 0 when perfectly overlapping
    W = t                               # Wasserstein-1 = theta

    axR.plot(t[1:], js[1:], color=SLATE, lw=2.6, label=r"JS divergence")
    axR.plot([0], [0], "o", color=SLATE, mfc=PAPER, ms=6)  # open dot at origin
    axR.plot(t, W, color=RUST, lw=2.6, label=r"Wasserstein $W$")

    axR.axhline(np.log(2), color=SLATE, lw=0.8, ls=":", alpha=0.7)
    axR.text(2.98, np.log(2) + 0.05, r"$\log 2$", color=SLATE, ha="right", fontsize=11)

    # gradient annotations
    axR.annotate("flat: gradient = 0\n(G gets no signal)",
                 xy=(1.5, np.log(2)), xytext=(1.35, 1.35),
                 color=SLATE, fontsize=11, ha="center",
                 arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.2))
    axR.annotate("slope 1: gradient\neverywhere",
                 xy=(2.3, 2.3), xytext=(2.35, 1.55),
                 color=RUST, fontsize=11, ha="center",
                 arrowprops=dict(arrowstyle="->", color=RUST, lw=1.2))

    axR.set_xlabel(r"separation $\theta$ between the two distributions")
    axR.set_ylabel("distance between them")
    axR.set_xlim(0, 3.0)
    axR.set_ylim(0, 3.0)
    axR.set_title("Same separation, two different distances",
                  fontsize=13, loc="center", color=INK, pad=8)
    axR.legend(frameon=False, loc="upper left", fontsize=11)
    _clean(axR)

    fig.suptitle("Why WGAN helps: JS goes flat when supports are disjoint, Wasserstein keeps sloping",
                 fontsize=14.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "js_vs_wasserstein.svg")


# ---- 2. 1D bimodal toy: G converges to a two-mode target --------------------
def gan_1d_toy():
    xs = np.linspace(-5, 5, 600)

    def gauss(c, s):
        return np.exp(-0.5 * ((xs - c) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    def mix(c, s):
        return 0.5 * gauss(-c, s) + 0.5 * gauss(c, s)

    p_data = mix(2.0, 0.5)   # target: two modes at +/- 2

    # G's distribution at four training snapshots
    snaps = [
        ("Step 0", gauss(0.0, 1.5), "one wide blob at 0"),
        ("Step 100", mix(1.0, 0.9), "two bumps emerging"),
        ("Step 500", mix(1.8, 0.6), "closing on the modes"),
        ("Step 1000", mix(2.0, 0.5), r"$p_G = p_\mathrm{data}$"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.4), sharey=True)
    ymax = p_data.max() * 1.25
    for ax, (label, pg, note) in zip(axes, snaps):
        ax.fill_between(xs, p_data, color=SAGE, alpha=0.45, lw=0)
        ax.plot(xs, p_data, color=SAGE, lw=1.6)
        ax.plot(xs, pg, color=RUST, lw=2.4)
        ax.set_title(label, fontsize=13, color=INK, pad=6)
        ax.text(0, ymax * 0.93, note, ha="center", va="top",
                color=MUTED, fontsize=10.5, style="italic")
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, ymax)
        ax.set_yticks([])
        ax.set_xticks([-2, 0, 2])
        _clean(ax)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("x", fontsize=11)

    # a single figure-level legend at the top (avoids crowding per-panel notes)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=SAGE, lw=3, alpha=0.6, label=r"$p_\mathrm{data}$ (target)"),
               Line2D([], [], color=RUST, lw=2.4, label=r"$p_G$ (generator)")]
    fig.legend(handles=handles, frameon=False, ncol=2, fontsize=11,
               loc="upper center", bbox_to_anchor=(0.5, 0.99))

    fig.suptitle(r"A 1D toy: G's distribution $p_G$ slides onto a two-mode target $p_\mathrm{data}$",
                 fontsize=14.5, color=INK, y=1.12)
    fig.tight_layout()
    save(fig, "gan_1d_toy.svg")


if __name__ == "__main__":
    print("lec20 figures →", OUT)
    js_vs_wasserstein()
    gan_1d_toy()
    print("done.")
