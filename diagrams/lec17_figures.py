"""Generate figures for Lecture 17: Self-Supervised & Contrastive Learning.

Matches the hand-authored lec00/lec17 SVG palette (warm parchment + ink,
rust/sage/slate) so new matplotlib figures sit beside the existing ones.
Outputs SVG into figures/lec17/svg/.
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, FancyArrowPatch
from pathlib import Path

# ---- palette (from anthropic-theme.css / lec00_figures.py) ------------------
PAPER = "#F7F3E9"
INK = "#161513"
MUTED = "#5F5C54"
RUST = "#B85A3E"
SAGE = "#5F8573"
SLATE = "#37535F"
SAGE_FILL = "#9FB8AC"
RUST_FILL = "#D89A85"
SLATE_FILL = "#8FA6AF"

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

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec17" / "svg"
OUT.mkdir(parents=True, exist_ok=True)
ARROW = r"$\rightarrow$"


def save(fig, name):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def _tile(ax, cx, cy, s, motif, fill):
    """A little 'image' tile with a simple motif."""
    ax.add_patch(FancyBboxPatch(
        (cx - s / 2, cy - s / 2), s, s,
        boxstyle="round,pad=0.01,rounding_size=0.08",
        facecolor=PAPER, edgecolor=INK, lw=1.6, zorder=3))
    if motif == "circle":
        ax.add_patch(Circle((cx, cy), 0.22, facecolor=fill,
                            edgecolor=INK, lw=1.0, zorder=4))
    elif motif == "circle_crop":  # zoomed / off-centre → looks cropped
        ax.add_patch(Circle((cx + 0.10, cy + 0.06), 0.33, facecolor=fill,
                            edgecolor=INK, lw=1.0, zorder=4))
    elif motif == "tri":
        ax.add_patch(Polygon([(cx, cy + 0.24), (cx - 0.23, cy - 0.18),
                              (cx + 0.23, cy - 0.18)], closed=True,
                             facecolor=fill, edgecolor=INK, lw=1.0, zorder=4))
    elif motif == "tri_crop":
        ax.add_patch(Polygon([(cx + 0.08, cy + 0.34), (cx - 0.30, cy - 0.26),
                              (cx + 0.40, cy - 0.26)], closed=True,
                             facecolor=fill, edgecolor=INK, lw=1.0, zorder=4))


def _arrow(ax, p0, p1, color=MUTED, lw=1.6, style="-|>", dashed=False):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=14,
        color=color, lw=lw, linestyle="--" if dashed else "-",
        shrinkA=1, shrinkB=1, zorder=2))


def positive_pairs():
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.set_xlim(0, 9.6)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    s = 0.78

    # ---- ROW A (top) : image A -> two views = a POSITIVE pair --------------
    ax.text(1.0, 5.02, "image A", ha="center", fontsize=11, color=MUTED)
    _tile(ax, 1.0, 4.30, s, "circle", SAGE_FILL)
    _arrow(ax, (1.5, 4.45), (2.55, 4.78))
    _arrow(ax, (1.5, 4.15), (2.55, 3.72))
    ax.text(2.0, 4.82, "augment", ha="center", fontsize=9, color=MUTED, style="italic")
    _tile(ax, 3.0, 4.78, s, "circle_crop", SAGE_FILL)
    _tile(ax, 3.0, 3.72, s, "circle", RUST_FILL)
    _arrow(ax, (3.55, 4.70), (3.55, 3.80), color=SAGE, lw=2.6, style="<|-|>")
    ax.text(5.55, 4.30, "POSITIVE pair", ha="center", va="center",
            fontsize=13, color=SAGE, weight="bold")
    ax.text(5.55, 3.95, "two views of image A " + ARROW + " pull together",
            ha="center", va="center", fontsize=10, color=MUTED, style="italic")

    # ---- ROW B (bottom) : image B -> two views = another positive pair ------
    ax.text(1.0, 2.10, "image B", ha="center", fontsize=11, color=MUTED)
    _tile(ax, 1.0, 1.38, s, "tri", SLATE_FILL)
    _arrow(ax, (1.5, 1.53), (2.55, 1.86))
    _arrow(ax, (1.5, 1.23), (2.55, 0.80))
    ax.text(2.0, 1.90, "augment", ha="center", fontsize=9, color=MUTED, style="italic")
    _tile(ax, 3.0, 1.86, s, "tri_crop", SLATE_FILL)
    _tile(ax, 3.0, 0.80, s, "tri", RUST_FILL)
    _arrow(ax, (3.55, 1.78), (3.55, 0.88), color=SAGE, lw=2.6, style="<|-|>")
    ax.text(5.55, 1.38, "POSITIVE pair", ha="center", va="center",
            fontsize=13, color=SAGE, weight="bold")
    ax.text(5.55, 1.03, "two views of image B " + ARROW + " pull together",
            ha="center", va="center", fontsize=10, color=MUTED, style="italic")

    # ---- NEGATIVE link across the two images --------------------------------
    _arrow(ax, (3.0, 3.30), (3.0, 2.32), color=RUST, lw=2.2,
           style="<|-|>", dashed=True)
    ax.text(7.7, 2.95, "NEGATIVE pair", ha="center", va="center",
            fontsize=13, color=RUST, weight="bold")
    ax.text(7.7, 2.60, "any view of A  vs  any view of B", ha="center",
            va="center", fontsize=10, color=MUTED, style="italic")
    ax.text(7.7, 2.30, "different images " + ARROW + " push apart", ha="center",
            va="center", fontsize=10, color=MUTED, style="italic")

    # colour / size legend + banner
    ax.text(4.8, 0.18, "colour = colour-jitter    ·    zoom = random crop",
            ha="center", va="center", fontsize=9, color=MUTED, style="italic")

    save(fig, "positive_pairs.svg")


if __name__ == "__main__":
    print("Generating lec17 figures...")
    positive_pairs()
    print("Done.")
