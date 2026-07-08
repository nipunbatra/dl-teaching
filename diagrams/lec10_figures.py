"""Generate NEW figures for Lecture 10: RNNs, LSTMs & GRUs.

These COMPLEMENT the existing hand-authored SVGs (rnn_unrolled, rnn_three_patterns,
bptt_vanishing, lstm_cell, lstm_annotated) — they do not duplicate them.

Palette + rcParams copied from diagrams/lec00_figures.py so the new matplotlib
figures sit beside the parchment/ink hand-drawn SVGs without clashing.
Outputs SVG into figures/lec10/svg/.

New figures:
  1. rnn_cell_output          — OBJECT FIRST: a cell eats (h_{t-1}, x_t) -> new memory h_t.
  2. gradient_survival        — quantitative: LSTM 0.99^t survives, RNN 0.5^t vanishes.
  3. additive_vs_multiplicative — the cell-state highway (add) vs the vanilla path (multiply).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from pathlib import Path

# ---- palette (from anthropic-theme.css / lec00_figures.py) -------------------
PAPER = "#F7F3E9"
INK = "#161513"
MUTED = "#5F5C54"
RUST = "#B85A3E"
SAGE = "#5F8573"
SLATE = "#37535F"
SAGE_FILL = "#9FB8AC"
SAND = "#EFE7D4"
ROSE = "#E6D6CB"

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

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec10" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name, tight=True):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {name}")


def _rbox(ax, xy, w, h, fc, ec, lw=1.6, r=0.06):
    ax.add_patch(FancyBboxPatch(xy, w, h,
                 boxstyle=f"round,pad=0.01,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, mutation_aspect=1))


# ---- 1. OBJECT FIRST: what does an RNN cell output? --------------------------
def rnn_cell_output():
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    ax.axis("off")

    # the cell
    cx, cy, cw, ch = 4.2, 1.5, 1.9, 1.5
    _rbox(ax, (cx, cy), cw, ch, ROSE, RUST, lw=2.2)
    ax.text(cx + cw / 2, cy + ch / 2 + 0.18, "RNN cell", ha="center", va="center",
            color=INK, fontsize=15, weight="bold")
    ax.text(cx + cw / 2, cy + ch / 2 - 0.32,
            r"$\tanh(Wx_t + Uh_{t-1})$", ha="center", va="center",
            color=MUTED, fontsize=11)

    # incoming: previous memory h_{t-1}
    _rbox(ax, (0.5, 2.55), 2.35, 0.7, SAGE_FILL, SAGE, lw=1.4)
    ax.text(1.67, 2.9, r"$h_{t-1}=[\,0.10,\ 0.29,\ 0.46\,]$",
            ha="center", va="center", color=INK, fontsize=11.5)
    ax.text(1.67, 3.5, "old memory", ha="center", color=SAGE, fontsize=11, style="italic")

    # incoming: new word x_t
    _rbox(ax, (0.9, 0.9), 1.55, 0.7, SAND, MUTED, lw=1.4)
    ax.text(1.67, 1.25, r'$x_t$ = "love"', ha="center", va="center", color=INK, fontsize=12)
    ax.text(1.67, 0.45, "new word", ha="center", color=MUTED, fontsize=11, style="italic")

    ax.add_patch(FancyArrowPatch((2.9, 2.75), (cx - 0.05, cy + ch - 0.35),
                 arrowstyle="-|>", mutation_scale=15, color=SAGE, lw=1.8,
                 connectionstyle="arc3,rad=-0.12"))
    ax.add_patch(FancyArrowPatch((2.5, 1.25), (cx - 0.05, cy + 0.4),
                 arrowstyle="-|>", mutation_scale=15, color=MUTED, lw=1.8,
                 connectionstyle="arc3,rad=0.12"))

    # outgoing: NEW memory h_t (the object)
    ax.add_patch(FancyArrowPatch((cx + cw + 0.05, cy + ch / 2), (7.35, cy + ch / 2),
                 arrowstyle="-|>", mutation_scale=17, color=RUST, lw=2.4))
    _rbox(ax, (7.4, 1.55), 2.05, 1.4, "#F3E2D2", RUST, lw=2.2)
    ax.text(8.42, 2.55, r"$h_t$", ha="center", va="center", color=RUST,
            fontsize=17, weight="bold")
    ax.text(8.42, 2.05, r"$[\,0.39,\ 0.70,\ 0.87\,]$", ha="center", va="center",
            color=INK, fontsize=11.5)

    ax.text(8.42, 1.15, "NEW memory —", ha="center", color=RUST, fontsize=11.5, weight="bold")
    ax.text(8.42, 0.78, "a running summary of\nthe sequence so far",
            ha="center", va="top", color=MUTED, fontsize=10.5, style="italic")

    ax.set_xlim(0, 9.7)
    ax.set_ylim(0.25, 3.75)
    save(fig, "rnn_cell_output.svg")


# ---- 2. quantitative gradient survival: LSTM vs vanilla RNN ------------------
def gradient_survival():
    t = np.arange(0, 101)
    lstm = 0.99 ** t
    rnn = 0.5 ** t

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.semilogy(t, lstm, color=SAGE, lw=2.8,
                label=r"LSTM cell-state path  $(f_t\approx 0.99)^t$")
    ax.semilogy(t, rnn, color=RUST, lw=2.8,
                label=r"vanilla RNN  $(0.5)^t$")

    # endpoint markers + annotations
    ax.scatter([100], [lstm[-1]], s=70, color=SAGE, zorder=5, edgecolor=PAPER, lw=1)
    ax.scatter([100], [rnn[-1]], s=70, color=RUST, zorder=5, edgecolor=PAPER, lw=1)
    ax.annotate(r"$0.99^{100}\approx 0.37$" + "\nstill usable",
                xy=(100, lstm[-1]), xytext=(66, 3e-2),
                color=SAGE, fontsize=11.5,
                arrowprops=dict(arrowstyle="-|>", color=SAGE, lw=1.4))
    ax.annotate(r"$0.5^{100}\approx 8\times10^{-31}$" + "\neffectively zero",
                xy=(100, rnn[-1]), xytext=(40, 2e-27),
                color=RUST, fontsize=11.5,
                arrowprops=dict(arrowstyle="-|>", color=RUST, lw=1.4))

    ax.set_xlabel("timesteps the gradient travels back", fontsize=12)
    ax.set_ylabel("gradient magnitude  (log scale)", fontsize=12)
    ax.set_title("Same gradient, 100 steps back — the additive path survives",
                 fontsize=14.5, loc="left", color=INK, pad=10)
    ax.set_xlim(0, 104)
    ax.set_ylim(1e-32, 5)
    ax.legend(frameon=False, fontsize=11.5, loc="lower left")
    _clean(ax)
    save(fig, "gradient_survival.svg")


# ---- 3. additive highway vs multiplicative path -----------------------------
def additive_vs_multiplicative():
    fig, ax = plt.subplots(figsize=(9.8, 4.7))
    ax.axis("off")

    xs = [0.7, 3.0, 5.3, 7.6]           # box left-x for 4 states
    bw, bh = 1.35, 0.72

    # ---------------- TOP: vanilla RNN, multiplicative (shrinks) --------------
    top_y = 3.15
    ax.text(0.2, 4.35, "Vanilla RNN — gradient is MULTIPLIED by $W$ every hop",
            color=RUST, fontsize=13, weight="bold")
    labs = [r"$h_{t-3}$", r"$h_{t-2}$", r"$h_{t-1}$", r"$h_t$"]
    for x, lab in zip(xs, labs):
        _rbox(ax, (x, top_y), bw, bh, SAND, MUTED, lw=1.4)
        ax.text(x + bw / 2, top_y + bh / 2, lab, ha="center", va="center",
                color=INK, fontsize=12.5)
    # arrows with x W and shrinking gradient bars underneath
    grads_top = [1.0, 0.5, 0.25, 0.125]
    for i in range(3):
        x0 = xs[i] + bw
        x1 = xs[i + 1]
        ax.add_patch(FancyArrowPatch((x0, top_y + bh / 2), (x1, top_y + bh / 2),
                     arrowstyle="-|>", mutation_scale=14, color=RUST, lw=2))
        ax.text((x0 + x1) / 2, top_y + bh + 0.02, r"$\times W$", ha="center",
                color=RUST, fontsize=11)
    # shrinking bars (below the chain)
    for x, g in zip(xs, grads_top):
        cxb = x + bw / 2
        ax.add_patch(Rectangle((cxb - 0.16, top_y - 0.28 - g * 0.5), 0.32, g * 0.5,
                     fc=RUST, ec="none", alpha=0.85))
    ax.text(xs[-1] + bw + 0.15, top_y + bh / 2, "shrinks fast\n(vanishes)",
            va="center", color=RUST, fontsize=11, style="italic")

    # ---------------- BOTTOM: LSTM cell state, additive (survives) ------------
    bot_y = 0.75
    ax.text(0.2, 1.95, "LSTM cell state — gradient is ADDED along, scaled by $f_t\\approx 1$",
            color=SAGE, fontsize=13, weight="bold")
    labs2 = [r"$c_{t-3}$", r"$c_{t-2}$", r"$c_{t-1}$", r"$c_t$"]
    for x, lab in zip(xs, labs2):
        _rbox(ax, (x, bot_y), bw, bh, SAGE_FILL, SAGE, lw=1.6)
        ax.text(x + bw / 2, bot_y + bh / 2, lab, ha="center", va="center",
                color=INK, fontsize=12.5)
    grads_bot = [1.0, 0.99, 0.98, 0.97]
    for i in range(3):
        x0 = xs[i] + bw
        x1 = xs[i + 1]
        ax.add_patch(FancyArrowPatch((x0, bot_y + bh / 2), (x1, bot_y + bh / 2),
                     arrowstyle="-|>", mutation_scale=14, color=SAGE, lw=2.4))
        ax.text((x0 + x1) / 2, bot_y + bh + 0.02, r"$+\ (\times f_t)$", ha="center",
                color=SAGE, fontsize=11)
    for x, g in zip(xs, grads_bot):
        cxb = x + bw / 2
        ax.add_patch(Rectangle((cxb - 0.16, bot_y - 0.28 - g * 0.5), 0.32, g * 0.5,
                     fc=SAGE, ec="none", alpha=0.85))
    ax.text(xs[-1] + bw + 0.15, bot_y + bh / 2, "stays tall\n(survives)",
            va="center", color=SAGE, fontsize=11, style="italic")

    ax.text(0.2, -0.15,
            r"bars = gradient magnitude carried back.  Product of $<1$ terms decays;"
            r" the $+$ keeps a clean path open.",
            color=MUTED, fontsize=10.5, style="italic")

    ax.set_xlim(0, 10.4)
    ax.set_ylim(-0.4, 4.6)
    save(fig, "additive_vs_multiplicative.svg")


if __name__ == "__main__":
    print("Generating lec10 figures...")
    rnn_cell_output()
    gradient_survival()
    additive_vs_multiplicative()
    print("Done.")
