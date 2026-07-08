"""Generate figures for Lecture 13: The Transformer.

New matplotlib figures that support the "Andrew Ng" output-first build-up for
attention. Palette + rcParams copied from lec00_figures.py so these sit beside
the hand-authored lec13 SVGs (transformer_block, multi_head_*, etc.) without
clashing. Outputs SVG into figures/lec13/svg/.

Three figures, each mapping to one step of the build-up:
  1. attention_weighted_average.svg  — THE OUTPUT FIRST: a head outputs a
     weighted average of value vectors (3-token toy example, weights as bars).
  2. qk_to_weights.svg               — where the weights come from: q·k scores
     → softmax → those same weights.
  3. causal_mask.svg                 — the causal mask as a heatmap (no peeking).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ---- palette (from anthropic-theme.css / lec00_figures.py) -------------------
PAPER = "#F7F3E9"
PAPER_ALT = "#EFEADA"
INK = "#161513"
MUTED = "#5F5C54"
RUST = "#B85A3E"
SAGE = "#5F8573"
SLATE = "#37535F"
GOLD = "#C9A14A"
WINE = "#A8324B"
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

# cream -> rust sequential colormap, kept on-palette for heatmaps
CREAM_RUST = LinearSegmentedColormap.from_list(
    "cream_rust", [PAPER, "#E8CBBB", "#D98E6E", RUST, "#8F3F28"]
)

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec13" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ---- 1. attention output-first: weighted average of value vectors -----------
def attention_weighted_average():
    tokens = ["The", "cat", "sat"]
    weights = np.array([0.1, 0.6, 0.3])
    colors = [SLATE, RUST, SAGE]

    # 2-D value vectors (toy) and the resulting weighted-average output
    V = np.array([[2.5, 0.5], [1.0, 2.5], [2.0, 1.0]])
    out = weights @ V  # (1.45, 1.85)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 4.5))

    # -- left: attention weights as bars --------------------------------------
    bars = axL.bar(tokens, weights, color=colors, edgecolor=INK, linewidth=1.1,
                   width=0.6, zorder=3)
    for b, w in zip(bars, weights):
        axL.text(b.get_x() + b.get_width() / 2, w + 0.02, f"{w:.1f}",
                 ha="center", va="bottom", fontsize=13, color=INK)
    axL.set_ylim(0, 0.72)
    axL.set_ylabel("attention weight")
    axL.set_title("weights for query  “sat”  (sum = 1)",
                  fontsize=13.5, loc="left", color=INK, pad=8)
    axL.spines[["top", "right"]].set_visible(False)
    axL.tick_params(length=3)

    # -- right: value vectors + the weighted-average output -------------------
    axR.axhline(0, color=MUTED, lw=0.8, zorder=1)
    axR.axvline(0, color=MUTED, lw=0.8, zorder=1)
    for (vx, vy), tok, c, w in zip(V, tokens, colors, weights):
        axR.annotate("", xy=(vx, vy), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6,
                                     alpha=0.45 + 0.55 * w))
        axR.text(vx + 0.08, vy + 0.06, f"$v_{{\\rm {tok}}}$", color=c,
                 fontsize=13)
    # output vector (thick)
    axR.annotate("", xy=(out[0], out[1]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=3.2))
    axR.scatter([out[0]], [out[1]], s=70, color=INK, zorder=5)
    axR.text(out[0] + 0.08, out[1] - 0.28,
             "output\n= 0.1$v_{\\rm The}$+0.6$v_{\\rm cat}$+0.3$v_{\\rm sat}$",
             color=INK, fontsize=11.5, va="top")
    axR.set_xlim(-0.2, 3.0)
    axR.set_ylim(-0.2, 3.0)
    axR.set_aspect("equal")
    axR.set_title("output = weighted average of value vectors",
                  fontsize=13.5, loc="left", color=INK, pad=8)
    axR.spines[["top", "right"]].set_visible(False)
    axR.tick_params(length=3)

    fig.suptitle("One attention head, output-first · a smart weighted average",
                 fontsize=15, x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "attention_weighted_average.svg")


# ---- 2. where the weights come from: q.k scores -> softmax -> weights --------
def qk_to_weights():
    tokens = ["The", "cat", "sat"]
    scores = np.array([0.0, 1.8, 1.1])            # q . k / sqrt(d_k)
    weights = np.exp(scores) / np.exp(scores).sum()  # -> [0.10, 0.60, 0.30]
    colors = [SLATE, RUST, SAGE]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                   gridspec_kw={"wspace": 0.6})

    b1 = axL.bar(tokens, scores, color=PAPER_ALT, edgecolor=INK, linewidth=1.1,
                 width=0.6, zorder=3)
    for b, s in zip(b1, scores):
        axL.text(b.get_x() + b.get_width() / 2, s + 0.04, f"{s:.1f}",
                 ha="center", va="bottom", fontsize=12.5, color=INK)
    axL.set_ylim(0, 2.2)
    axL.set_ylabel(r"similarity  $q\cdot k/\sqrt{d_k}$")
    axL.set_title("1 · raw scores: query “sat” dotted with each key",
                  fontsize=12.5, loc="left", color=INK, pad=8)
    axL.spines[["top", "right"]].set_visible(False)
    axL.tick_params(length=3)

    b2 = axR.bar(tokens, weights, color=colors, edgecolor=INK, linewidth=1.1,
                 width=0.6, zorder=3)
    for b, w in zip(b2, weights):
        axR.text(b.get_x() + b.get_width() / 2, w + 0.015, f"{w:.2f}",
                 ha="center", va="bottom", fontsize=12.5, color=INK)
    axR.set_ylim(0, 0.72)
    axR.set_ylabel("attention weight")
    axR.set_title("2 · softmax $\\rightarrow$ weights (bigger dot, bigger weight)",
                  fontsize=12.5, loc="left", color=INK, pad=8)
    axR.spines[["top", "right"]].set_visible(False)
    axR.tick_params(length=3)

    # softmax arrow between the panels
    fig.text(0.492, 0.5, "softmax", ha="center", va="center", fontsize=12,
             style="italic", color=RUST,
             bbox=dict(boxstyle="rarrow,pad=0.3", fc=PAPER, ec=RUST, lw=1.3))

    fig.suptitle("Where the weights come from · dot-product then softmax",
                 fontsize=15, x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "qk_to_weights.svg")


# ---- 3. causal mask as a heatmap --------------------------------------------
def causal_mask():
    toks = ["The", "quick", "brown", "fox"]
    n = len(toks)
    rng = np.random.default_rng(3)
    raw = rng.uniform(0.2, 1.0, size=(n, n))      # arbitrary pre-softmax scores

    # causal mask: keep j <= i, softmax over allowed keys per row
    masked = np.where(np.tril(np.ones((n, n), bool)), raw, -np.inf)
    weights = np.exp(masked - np.nanmax(np.where(np.isfinite(masked), masked, -np.inf),
                                        axis=1, keepdims=True))
    weights = np.where(np.isfinite(masked), weights, 0.0)
    weights = weights / weights.sum(axis=1, keepdims=True)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 4.6),
                                   gridspec_kw={"wspace": 0.35})

    # left: scores with future positions struck out
    disp = np.where(np.tril(np.ones((n, n), bool)), raw, np.nan)
    axL.imshow(disp, cmap=CREAM_RUST, vmin=0, vmax=1.1, aspect="equal")
    for i in range(n):
        for j in range(n):
            if j <= i:
                axL.text(j, i, f"{raw[i, j]:.1f}", ha="center", va="center",
                         fontsize=11, color=INK)
            else:
                axL.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=True, fc="#E4DECB", ec=PAPER))
                axL.text(j, i, r"$-\infty$", ha="center", va="center",
                         fontsize=11, color=MUTED)
    axL.set_title("1 · add $-\\infty$ above the diagonal",
                  fontsize=12.5, loc="left", color=INK, pad=8)
    _grid_labels(axL, toks)

    # right: resulting weights (lower-triangular, rows sum to 1)
    im = axR.imshow(np.where(weights > 0, weights, np.nan), cmap=CREAM_RUST,
                    vmin=0, vmax=1.0, aspect="equal")
    for i in range(n):
        for j in range(n):
            if j <= i:
                axR.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center",
                         fontsize=10.5,
                         color=INK if weights[i, j] < 0.6 else PAPER)
            else:
                axR.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=True, fc="#E4DECB", ec=PAPER))
                axR.text(j, i, "0", ha="center", va="center", fontsize=11,
                         color=MUTED)
    axR.set_title("2 · softmax $\\rightarrow$ future gets exactly 0",
                  fontsize=12.5, loc="left", color=INK, pad=8)
    _grid_labels(axR, toks)

    fig.suptitle("Causal mask · token $i$ can attend only to $j\\leq i$ (no peeking ahead)",
                 fontsize=14.5, x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "causal_mask.svg")


def _grid_labels(ax, toks):
    n = len(toks)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(toks, fontsize=11); ax.set_yticklabels(toks, fontsize=11)
    ax.set_xlabel("key  $j$  (attend to)", fontsize=11.5)
    ax.set_ylabel("query  $i$  (from)", fontsize=11.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=PAPER, linewidth=1.5)


if __name__ == "__main__":
    attention_weighted_average()
    qk_to_weights()
    causal_mask()
    print("lec13 figures done.")
