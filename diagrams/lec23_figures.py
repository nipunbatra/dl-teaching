"""Generate figures for Lecture 23: Efficient Inference.

This deck is figure-light. Six hand-authored SVGs already exist (kv_cache,
kv_cache_growth, quantization_ladder, quantization_worked,
flash_attention_tiles, speculative_decoding). This script fills the three
genuine gaps, keeping the lec00 cream palette so everything sits together:

  1. memory_bound.svg      — Part 1 has no figure; this is the lecture's thesis
                             (one decode step is dominated by data movement,
                             not math → memory-bound; prefill is compute-bound).
  2. serving_menu.svg      — the "now it clicks" synthesis: every trick either
                             moves fewer bytes or reuses work.
  3. distillation_soft.svg — soft vs hard labels: temperature reveals the
                             teacher's dark knowledge (reuses the deck's numbers).

Outputs SVG into figures/lec23/svg/.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from pathlib import Path

# ---- palette (from anthropic-theme.css) -------------------------------------
PAPER = "#F7F3E9"
INK = "#161513"
MUTED = "#5F5C54"
RUST = "#B85A3E"
SAGE = "#5F8573"
SLATE = "#37535F"
SAGE_FILL = "#9FB8AC"
RUST_FILL = "#E5D2C6"
SLATE_FILL = "#AEC2CB"
CARD = "#EFE7D4"

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

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec23" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name, tight=True):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {name}")


# ---- 1. memory-bound: one decode step is data movement, not math ------------
def memory_bound():
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.6, 4.4),
        gridspec_kw={"width_ratios": [1.15, 1.0]}, layout="constrained")

    # LEFT · time budget of one decode step
    labels = ["Move weights + KV\nfrom GPU memory (HBM)", "Do the math\n(matrix × vector)"]
    times = [100, 0.9]        # ms · illustrative; compute is a sliver
    ypos = [1, 0]
    axL.barh(ypos, times, color=[RUST, SAGE], edgecolor=INK, height=0.46)
    axL.set_yticks(ypos)
    axL.set_yticklabels(labels, fontsize=10.5)
    axL.set_xlabel("time per generated token  (ms · illustrative)", fontsize=10.5)
    axL.set_xlim(0, 122)
    axL.text(102, 1, "≈ 100 ms", va="center", ha="left", color=RUST, fontsize=12, fontweight="bold")
    axL.text(4.5, 0, r"$\ll$ 1 ms", va="center", ha="left", color=SAGE, fontsize=12, fontweight="bold")
    axL.set_title("One decode step · where the time goes", fontsize=12.5,
                  loc="left", color=INK, pad=8)
    axL.text(0, -0.52, "≈150 GB moved at 1.5 TB/s ≈ 0.1 s\ncores idle almost the whole step",
             fontsize=9.5, color=MUTED, va="top")
    axL.set_ylim(-0.75, 1.6)
    _clean(axL)
    axL.spines["left"].set_visible(False)

    # RIGHT · the shape of the matmul explains it
    axR.axis("off")
    axR.set_title("Why · the shape of the matmul", fontsize=12.5, loc="left", color=INK, pad=8)

    def block(x, y, w, h, fc, ec, label, ncols=1):
        axR.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.4))
        if ncols > 1:
            for c in range(1, ncols):
                axR.add_patch(Rectangle((x + c * w / ncols, y), 0, h, ec=ec, lw=0.7))
                axR.plot([x + c * w / ncols, x + c * w / ncols], [y, y + h],
                         color=ec, lw=0.6, alpha=0.5)
        axR.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                 color=INK, fontsize=10)

    # prefill row
    axR.text(0.15, 3.45, "PREFILL — whole prompt at once", color=SLATE, fontsize=11, fontweight="bold")
    block(0.2, 2.35, 0.85, 0.9, SLATE_FILL, SLATE, "W")
    axR.text(1.22, 2.8, "×", ha="center", va="center", fontsize=16, color=MUTED)
    block(1.45, 2.35, 1.15, 0.9, CARD, SLATE, "tokens", ncols=5)
    axR.text(2.85, 2.8, "matrix × matrix\ncores saturated\ncompute-bound",
             va="center", ha="left", fontsize=9.5, color=SLATE)

    # decode row
    axR.text(0.15, 1.55, "DECODE — one token at a time", color=RUST, fontsize=11, fontweight="bold")
    block(0.2, 0.45, 0.85, 0.9, SLATE_FILL, SLATE, "W")
    axR.text(1.22, 0.9, "×", ha="center", va="center", fontsize=16, color=MUTED)
    block(1.45, 0.45, 0.22, 0.9, RUST_FILL, RUST, "")
    axR.text(1.56, 0.9, "1", ha="center", va="center", fontsize=9, color=INK)
    axR.text(2.0, 0.9, "matrix × vector\ncores wait for W\nmemory-bound",
             va="center", ha="left", fontsize=9.5, color=RUST)

    axR.set_xlim(0, 5.0)
    axR.set_ylim(0, 3.8)

    fig.suptitle("LLM inference is memory-bound · the GPU waits on data, not math",
                 fontsize=13.5, color=INK)
    save(fig, "memory_bound.svg", tight=False)


# ---- 2. the serving menu: fewer bytes, or reuse work ------------------------
def serving_menu():
    fig, ax = plt.subplots(figsize=(12.4, 4.9))
    ax.axis("off")

    cats = [
        ("Fewer bytes\nper weight", SLATE,
         ["Quantization\nINT8 / INT4", "Distillation\nto smaller model"]),
        ("Reuse work\nacross tokens", SAGE,
         ["KV-cache", "Paged attention\n(vLLM)"]),
        ("Fewer big-model\nsteps", RUST,
         ["Speculative\ndecoding"]),
        ("Cheaper exact\nattention", SLATE,
         ["FlashAttention\ntiles in SRAM"]),
        ("Pack the\nhardware", SAGE,
         ["Batching +\ncontinuous batching"]),
    ]

    n = len(cats)
    cw, gap, x0 = 2.25, 0.17, 0.15
    head_top, head_h = 3.55, 0.62
    item_h, item_gap = 0.66, 0.14

    for i, (head, color, items) in enumerate(cats):
        x = x0 + i * (cw + gap)
        ax.add_patch(Rectangle((x, head_top), cw, head_h, fc=color, ec="none"))
        ax.text(x + cw / 2, head_top + head_h / 2, head, ha="center", va="center",
                color=PAPER, fontsize=11, fontweight="bold")
        yy = head_top - item_gap
        for it in items:
            ax.add_patch(Rectangle((x, yy - item_h), cw, item_h, fc=CARD, ec=color, lw=1.3))
            ax.text(x + cw / 2, yy - item_h / 2, it, ha="center", va="center",
                    color=INK, fontsize=10)
            yy -= (item_h + item_gap)

    total_w = x0 + n * cw + (n - 1) * gap
    # synthesis banner — sit just below the lowest card, no wasted gap
    ax.add_patch(Rectangle((x0, 0.95), total_w - x0, 0.7, fc=SAGE_FILL, ec=SAGE, lw=1.2, alpha=0.55))
    ax.text((x0 + total_w) / 2, 1.3,
            "Serving = do less work per token  +  reuse work across tokens",
            ha="center", va="center", color=INK, fontsize=12.5, fontweight="bold")

    ax.text(x0, 4.55, "The menu · every trick moves fewer bytes or reuses work",
            ha="left", va="center", color=INK, fontsize=13.5, fontweight="bold")

    ax.set_xlim(0, total_w + 0.15)
    ax.set_ylim(0.8, 4.85)
    save(fig, "serving_menu.svg")


# ---- 3. distillation: temperature reveals dark knowledge --------------------
def distillation_soft():
    classes = ["Cat", "Dog", "Car"]
    t1 = [0.999, 0.0003, 0.0001]
    t4 = [0.81, 0.11, 0.08]
    x = np.arange(3)
    cols = [RUST, SLATE, SAGE]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2), layout="constrained")

    a1.bar(x, t1, color=cols, edgecolor=INK, lw=0.7)
    a1.set_title("Hard target · T = 1\none spike — no dark knowledge", fontsize=12, color=INK)
    a1.set_xticks(x); a1.set_xticklabels(classes)
    a1.set_ylim(0, 1.08)
    a1.set_ylabel("teacher probability")
    a1.text(0, 1.005, "0.999", ha="center", va="bottom", color=INK, fontsize=10)
    a1.text(1, 0.03, "≈0", ha="center", color=MUTED, fontsize=11)
    a1.text(2, 0.03, "≈0", ha="center", color=MUTED, fontsize=11)
    _clean(a1)

    a2.bar(x, t4, color=cols, edgecolor=INK, lw=0.7)
    a2.set_title("Softened · T = 4\ndark knowledge: Dog > Car", fontsize=12, color=INK)
    a2.set_xticks(x); a2.set_xticklabels(classes)
    a2.set_ylim(0, 1.08)
    for xi, v in zip(x, t4):
        a2.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", color=INK, fontsize=10.5)
    a2.annotate("Dog looks more like Cat\nthan Car does",
                xy=(1.32, 0.11), xytext=(1.42, 0.62), fontsize=10, color=SLATE,
                ha="left", arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.2))
    _clean(a2)

    fig.suptitle("Temperature reveals what the teacher knows about the wrong classes",
                 fontsize=13, color=INK)
    save(fig, "distillation_soft.svg", tight=False)


if __name__ == "__main__":
    print("Generating lec23 figures...")
    memory_bound()
    serving_menu()
    distillation_soft()
    print("Done.")
