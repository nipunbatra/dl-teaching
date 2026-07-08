"""Generate figures for Lecture 15: Large Language Models.

Three approachable, "object-first" diagrams that COMPLEMENT the existing
hand-authored SVGs (chinchilla_scaling, chinchilla_sweet_spot, gqa_variants,
rope_rotation, distributed_3d, emergent_abilities) rather than duplicate them:

  1. next_token_pipeline  — the concrete object: tokens -> Transformer ->
                            softmax -> sample. Anchors "an LLM = next-token
                            predictor at scale".
  2. loss_vs_compute      — the scaling LAW itself: loss falls as a straight
                            line on log-log compute (predictable), with the
                            compute-optimal frontier. Complements the existing
                            loss-vs-PARAMS iso-FLOP chart.
  3. kv_cache_bars        — the OBJECT that GQA shrinks: KV-cache memory in GB
                            for MHA vs GQA vs MQA. Complements the gqa_variants
                            head-sharing schematic with the actual size.

Palette + rcParams match diagrams/lec00_figures.py so these sit beside the
existing cream/parchment figures without clashing. Outputs SVG into
figures/lec15/svg/.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from pathlib import Path

# ---- palette (from anthropic-theme.css / lec00_figures.py) -------------------
PAPER = "#F7F3E9"
PANEL = "#EEEBDF"
INK = "#161513"
MUTED = "#5F5C54"
RUST = "#B85A3E"
SAGE = "#5F8573"
SLATE = "#37535F"
SAGE_FILL = "#9FB8AC"
GRID = "#C9C4B5"

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

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec15" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name, tight=True):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {name}")


# ---- 1. the whole object: a next-token predictor ----------------------------
def next_token_pipeline():
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)

    def box(x, y, w, h, text, fc, ec=INK, fs=12, tc=INK, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight=weight)

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                     mutation_scale=15, color=MUTED, lw=1.8))

    ymid = 2.5

    # 1. input tokens
    box(0.15, ymid - 0.45, 2.15, 0.9, "The cat sat\non the ___", "#EFE7D4",
        fs=12.5)
    ax.text(1.22, ymid - 0.85, "tokens (context)", ha="center", color=MUTED,
            fontsize=10.5, style="italic")

    arrow(2.35, 3.05, ymid)

    # 2. transformer
    box(3.05, ymid - 0.6, 2.3, 1.2, "decoder-only\nTransformer", SLATE,
        ec=INK, fs=12.5, tc=PAPER, weight="bold")
    ax.text(4.2, ymid - 1.0, "the recipe from L13/L14", ha="center",
            color=MUTED, fontsize=10.5, style="italic")

    arrow(5.4, 6.05, ymid)

    # 3. softmax distribution over the vocabulary (mini bar chart)
    cand = ["mat", "rug", "floor", "sofa", "..."]
    probs = [0.61, 0.18, 0.09, 0.06, 0.06]
    bx, by, bw, bh = 6.1, 1.35, 2.35, 2.3
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 fc="#FBF8F0", ec=INK, lw=1.4))
    ax.text(bx + bw / 2, by + bh - 0.22, r"softmax  $\rightarrow$  P(next token)",
            ha="center", color=INK, fontsize=11)
    n = len(cand)
    barw = bw / (n + 1.4)
    x0 = bx + barw * 0.6
    base = by + 0.55
    maxh = 1.25
    for i, (c, p) in enumerate(zip(cand, probs)):
        xx = x0 + i * (barw * 1.15)
        col = RUST if i == 0 else "#C9BCA2"
        ax.add_patch(Rectangle((xx, base), barw * 0.8, p * maxh,
                     fc=col, ec=MUTED, lw=0.6))
        ax.text(xx + barw * 0.4, base - 0.17, c, ha="center", color=MUTED,
                fontsize=9.5, rotation=0)
    ax.text(x0 + 0.05, base + 0.61 * maxh + 0.14, "0.61", color=RUST,
            fontsize=10, ha="left")

    arrow(8.5, 9.15, ymid)

    # 4. sampled token
    box(9.0, ymid - 0.42, 0.9, 0.84, "mat", RUST, ec=INK, fs=13, tc=PAPER,
        weight="bold")
    ax.text(9.45, ymid - 0.75, "sample", ha="center", color=MUTED,
            fontsize=10.5, style="italic")

    fig.suptitle("An LLM does exactly one thing — predict the next token; "
                 "scale is what changed",
                 fontsize=13.5, style="italic", weight="bold", color=INK, y=0.99)
    ax.text(5.0, 0.35,
            r"append the sampled token to the context, repeat  $\rightarrow$  "
            "the same code path runs GPT-2 and Llama-3",
            ha="center", color=MUTED, fontsize=10.5, style="italic")
    save(fig, "next_token_pipeline.svg")


# ---- 2. the scaling LAW: loss falls predictably with compute -----------------
def loss_vs_compute():
    fig, ax = plt.subplots(figsize=(8.6, 4.3))

    # reducible loss ~ a straight line on log-log; plus an irreducible floor.
    C = np.logspace(19, 25, 200)
    floor = 1.69
    L = floor + 14.0 * C ** (-0.050)

    ax.plot(C, L, color=RUST, lw=2.8, zorder=3,
            label="compute-optimal frontier")
    ax.axhline(floor, color=MUTED, ls=":", lw=1.3)
    ax.text(1.4e19, floor + 0.02, "irreducible loss (entropy of language)",
            color=MUTED, fontsize=10.5, va="bottom")

    # model dots sitting on the frontier
    pts = [
        (1.5e21, "GPT-2", SLATE, 10, -14),
        (3.1e23, "GPT-3", SLATE, 10, 12),
        (5.7e23, "Chinchilla", SAGE, -8, -16),
        (5e24, "frontier", RUST, -6, 14),
    ]
    for c, name, col, dx, dy in pts:
        l = floor + 14.0 * c ** (-0.050)
        ax.scatter([c], [l], s=70, color=col, edgecolor=INK, lw=0.8, zorder=4)
        ax.annotate(name, xy=(c, l), xytext=(dx, dy),
                    textcoords="offset points", color=col, fontsize=11,
                    ha="center")

    ax.set_xscale("log")
    ax.set_xlabel(r"training compute  C  (FLOPs, log scale)  $\rightarrow$")
    ax.set_ylabel(r"test loss  $\downarrow$")
    ax.set_xlim(1e19, 3e25)
    ax.set_ylim(floor - 0.15, 3.7)
    ax.set_title("Scaling law · more compute means lower loss, along a "
                 "straight line you can predict",
                 fontsize=13.5, style="italic", loc="left", color=INK, pad=10)

    ax.annotate("straight-ish on log-log:\npredictable before you train",
                xy=(4e22, floor + 14.0 * (4e22) ** (-0.050)),
                xytext=(6e22, 3.15), color=INK, fontsize=11,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    _clean(ax)
    save(fig, "loss_vs_compute.svg")


# ---- 3. the object GQA shrinks: KV-cache memory ------------------------------
def kv_cache_bars():
    fig, ax = plt.subplots(figsize=(8.6, 3.9))

    labels = ["MHA\n64 KV heads", "GQA\n8 KV heads", "MQA\n1 KV head"]
    sizes = [84.0, 10.5, 1.3]
    cols = [RUST, SAGE, SLATE]

    y = np.arange(len(labels))[::-1]
    ax.barh(y, sizes, color=cols, edgecolor=INK, lw=0.8, height=0.6)
    for yi, s in zip(y, sizes):
        ax.text(s + 1.5, yi, f"{s:g} GB", va="center", color=INK,
                fontsize=12.5, weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11.5)
    ax.set_xlim(0, 98)
    ax.set_xlabel("KV-cache memory for ONE 32k-token sequence  (70B model, fp16)")
    ax.set_title("What GQA shrinks · the KV-cache — store K,V for every past "
                 "token × layer × head",
                 fontsize=13.5, style="italic", loc="left", color=INK, pad=10)

    # 8x annotation between MHA and GQA
    ax.annotate("", xy=(84, y[0] - 0.34), xytext=(10.5, y[0] - 0.34),
                arrowprops=dict(arrowstyle="<|-|>", color=MUTED, lw=1.3))
    ax.text((84 + 10.5) / 2, y[0] - 0.55, "8× smaller, ~same quality",
            ha="center", color=MUTED, fontsize=11, style="italic")

    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    save(fig, "kv_cache_bars.svg")


if __name__ == "__main__":
    print("Generating lec15 figures...")
    next_token_pipeline()
    loss_vs_compute()
    kv_cache_bars()
    print("Done.")
