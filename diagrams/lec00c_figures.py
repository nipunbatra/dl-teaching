"""Figures for Lecture 0C: Information Theory for ML.
Anthropic parchment palette (matches lec00/lec00b). SVG -> figures/lec00c/svg/.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from pathlib import Path

PAPER = "#F7F3E9"; INK = "#161513"; MUTED = "#5F5C54"
RUST = "#B85A3E"; SAGE = "#5F8573"; SLATE = "#37535F"; SAGE_FILL = "#9FB8AC"

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["EB Garamond", "Georgia", "DejaVu Serif"],
    "font.size": 13, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
    "mathtext.fontset": "cm",
})

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec00c" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name, tight=True):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {name}")


# ---- 1. information content I(p) = -log2 p ----------------------------------
def info_content():
    p = np.linspace(0.001, 1, 400)
    I = -np.log2(p)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(p, I, color=RUST, lw=2.8)
    pts = [(1.0, r"certain $\to$ 0 bits"), (0.5, r"fair coin $\to$ 1 bit"),
           (1/1024, r"1-in-1024 $\to$ 10 bits")]
    for pv, lbl in pts:
        ax.scatter([pv], [-np.log2(pv)], color=SLATE, s=55, zorder=5)
        ax.annotate(lbl, (pv, -np.log2(pv)),
                    xytext=(pv + 0.06, -np.log2(pv) + 0.4), color=INK, fontsize=11)
    ax.set_xlabel("probability of the outcome,  p")
    ax.set_ylabel(r"surprise  $I(p) = -\log_2 p$  (bits)")
    ax.set_title("Information = surprise · rare outcomes carry more bits",
                 fontsize=14, loc="left", color=INK, pad=10)
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 10.5)
    _clean(ax)
    save(fig, "info_content.svg")


# ---- 2. optimal prefix code as a binary tree (weather) ---------------------
def coding_tree():
    fig, ax = plt.subplots(figsize=(8.6, 4.6)); ax.axis("off")
    # nodes
    def node(x, y, fc="#EFE7D4"):
        ax.add_patch(plt.Circle((x, y), 0.16, facecolor=fc, edgecolor=SLATE, lw=1.4, zorder=3))
    def edge(x1, y1, x2, y2, lbl):
        ax.plot([x1, x2], [y1, y2], color=MUTED, lw=1.6, zorder=1)
        ax.text((x1+x2)/2 - 0.07, (y1+y2)/2 + 0.04, lbl, color=RUST, fontsize=13, fontweight="bold")
    def leaf(x, y, text):
        ax.add_patch(Rectangle((x-0.55, y-0.22), 1.1, 0.44, facecolor=SAGE_FILL,
                     edgecolor=SLATE, lw=1.2, zorder=3))
        ax.text(x, y, text, ha="center", va="center", color=INK, fontsize=11)

    node(2.5, 3.2)
    edge(2.5, 3.05, 1.2, 2.1, "0"); leaf(1.2, 1.85, "Sunny  ½\ncode '0'  · 1 bit")
    edge(2.5, 3.05, 3.8, 2.1, "1"); node(3.8, 2.0)
    edge(3.8, 1.85, 3.0, 1.0, "0"); leaf(3.0, 0.75, "Cloudy  ¼\n'10' · 2 bits")
    edge(3.8, 1.85, 4.8, 1.0, "1"); leaf(4.8, 0.75, "Rainy  ¼\n'11' · 2 bits")

    ax.text(6.4, 2.4,
            r"frequent $\to$ short code" "\n" r"rare $\to$ long code" "\n\n"
            r"avg length"
            "\n= ½·1 + ¼·2 + ¼·2\n= 1.5 bits = H(P)",
            ha="left", va="center", color=INK, fontsize=12)
    ax.set_title("The optimal code · short codes for frequent symbols",
                 fontsize=14, loc="left", color=INK)
    ax.set_xlim(0.2, 9.2); ax.set_ylim(0.3, 3.7)
    save(fig, "coding_tree.svg", tight=False)


# ---- 3. cross-entropy = entropy + KL (extra bits) stacked bars --------------
def cross_entropy_extrabits():
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    # bar 1: optimal code for P
    ax.bar(0, 1.5, width=0.5, color=SAGE, edgecolor=MUTED, lw=0.8)
    ax.text(0, 0.75, "H(P)\n1.5", ha="center", va="center", color="white", fontsize=12)
    ax.text(0, -0.18, "optimal code\n(built for P)", ha="center", va="top", color=INK, fontsize=11)
    # bar 2: code for Q used on P = entropy + KL
    ax.bar(1, 1.5, width=0.5, color=SAGE, edgecolor=MUTED, lw=0.8)
    ax.bar(1, 0.25, width=0.5, bottom=1.5, color=RUST, edgecolor=MUTED, lw=0.8)
    ax.text(1, 0.75, "H(P)\n1.5", ha="center", va="center", color="white", fontsize=12)
    ax.text(1, 1.625, "KL 0.25", ha="center", va="center", color="white", fontsize=10)
    ax.text(1, -0.18, "wrong code\n(built for Q)", ha="center", va="top", color=INK, fontsize=11)

    ax.annotate("cross-entropy\nH(P,Q) = 1.75", xy=(1.27, 1.75), xytext=(1.5, 1.4),
                color=INK, fontsize=11, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.annotate("extra bits\n= KL(P" r"$\,\|\,$" "Q)", xy=(1.27, 1.62), xytext=(1.5, 0.55),
                color=RUST, fontsize=11, va="center",
                arrowprops=dict(arrowstyle="-|>", color=RUST, lw=1.2))
    ax.set_xticks([]); ax.set_ylabel("bits per symbol")
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 2.1)
    ax.set_title("Cross-entropy = entropy + KL · you can never beat the optimal code",
                 fontsize=13, loc="left", color=INK, pad=10)
    _clean(ax)
    save(fig, "cross_entropy_extrabits.svg")


# ---- 4. Serrano dice game · scoring candidate distributions ----------------
def dice_scoring():
    P = np.array([0.4, 0.2, 0.1, 0.1, 0.2])     # true die
    Q1 = np.array([0.4, 0.1, 0.2, 0.2, 0.1])    # close
    Q2 = np.array([0.1, 0.2, 0.4, 0.2, 0.1])    # far
    def ce(p, q): return -np.sum(p * np.log2(q))
    H = ce(P, P)
    faces = np.arange(1, 6)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    data = [(P, "true die  P", SAGE, f"H(P) = {H:.2f} bits\n(best possible score)"),
            (Q1, "model $Q_1$ (close)", SLATE, f"H(P,$Q_1$) = {ce(P,Q1):.2f}\nKL = {ce(P,Q1)-H:.2f}"),
            (Q2, "model $Q_2$ (far)", RUST, f"H(P,$Q_2$) = {ce(P,Q2):.2f}\nKL = {ce(P,Q2)-H:.2f}")]
    for ax, (d, ttl, col, note) in zip(axes, data):
        ax.bar(faces, d, color=col, edgecolor=MUTED, lw=0.6)
        ax.set_title(ttl, fontsize=12.5, color=INK)
        ax.set_xticks(faces); ax.set_ylim(0, 0.5)
        ax.set_xlabel("die face")
        ax.text(0.5, 0.93, note, transform=ax.transAxes, ha="left", va="top",
                color=INK, fontsize=10.5)
        _clean(ax)
    axes[0].set_ylabel("probability")
    fig.suptitle("Score a model by surprise on real rolls · the wrong die pays a KL penalty",
                 fontsize=13.5, color=INK)
    save(fig, "dice_scoring.svg", tight=False)


if __name__ == "__main__":
    print("Generating lec00c figures...")
    info_content()
    coding_tree()
    cross_entropy_extrabits()
    dice_scoring()
    print("Done.")
