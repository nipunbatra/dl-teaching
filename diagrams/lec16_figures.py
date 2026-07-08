"""Generate figures for Lecture 16: Alignment & Fine-tuning.

Matches the hand-authored lec16 / lec00 SVG palette (warm parchment + ink,
rust/sage/slate) so new matplotlib figures sit beside the existing hand-SVGs
without clashing. Outputs SVG into figures/lec16/svg/.

Palette + rcParams copied from diagrams/lec00_figures.py (single source of truth
for the cream theme) so this file stands alone.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ---- palette (from anthropic-theme.css) -------------------------------------
PAPER = "#F7F3E9"
INK = "#161513"
MUTED = "#5F5C54"
RUST = "#B85A3E"
SAGE = "#5F8573"
SLATE = "#37535F"
SAGE_FILL = "#9FB8AC"
RUST_FILL = "#EAD3C5"
SLATE_FILL = "#DCE3E6"

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

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec16" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name, tight=True):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {name}")


# ---- reward model · output-first --------------------------------------------
def reward_model_output():
    """OUTPUT FIRST: a reward model maps (prompt, response) -> ONE scalar score.
    Higher score = a human would prefer this response. Two responses to the same
    prompt get scored; the higher one is 'chosen', the lower one 'rejected'.
    Numbers match Practice Problem P2 (r_w = 1.2, r_l = 0.5).
    """
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(0.15, 5.68, "The reward model outputs ONE number per response",
            fontsize=15, fontweight="bold", color=INK)
    ax.text(0.15, 5.28, r"$r_\phi(\mathrm{prompt},\ \mathrm{response}) \to$ a scalar 'how much a human would prefer this'",
            fontsize=11.5, color=MUTED, style="italic")

    # shared prompt box
    prompt = FancyBboxPatch((0.15, 4.05), 3.05, 0.78,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc=PAPER, ec=INK, lw=1.3)
    ax.add_patch(prompt)
    ax.text(1.675, 4.55, "PROMPT", ha="center", fontsize=8.5,
            color=MUTED, family="sans-serif")
    ax.text(1.675, 4.27, "Explain gravity to a 5-year-old",
            ha="center", fontsize=10.5, color=INK)

    # two candidate responses
    def response(y, tag, text, fill, edge):
        box = FancyBboxPatch((0.15, y), 3.05, 1.06,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             fc=fill, ec=edge, lw=1.4)
        ax.add_patch(box)
        ax.text(0.30, y + 0.82, tag, ha="left", fontsize=8.5,
                color=edge, family="sans-serif", fontweight="bold")
        ax.text(1.675, y + 0.34, text, ha="center", fontsize=9.6, color=INK)
        return box

    response(2.62, "RESPONSE A",
             '"The earth is a giant ball that\ngently pulls things to its middle."', "#F0F5EF", SAGE)
    response(0.92, "RESPONSE B",
             '"Gravity is the curvature of\nspacetime per general relativity."', "#FBEDE7", RUST)

    # reward model box in the middle
    rm = FancyBboxPatch((3.95, 2.15), 1.9, 2.0,
                        boxstyle="round,pad=0.02,rounding_size=0.1",
                        fc=SLATE_FILL, ec=SLATE, lw=1.6)
    ax.add_patch(rm)
    ax.text(4.90, 3.42, "Reward", ha="center", fontsize=12.5,
            color=SLATE, fontweight="bold")
    ax.text(4.90, 3.10, "model", ha="center", fontsize=12.5,
            color=SLATE, fontweight="bold")
    ax.text(4.90, 2.70, r"$r_\phi$", ha="center", fontsize=15, color=SLATE)
    ax.text(4.90, 2.38, "frozen after training", ha="center",
            fontsize=8, color=MUTED, family="sans-serif", style="italic")

    # arrows prompt+responses -> RM
    for y0 in (4.05, 3.03, 1.33):
        ax.add_patch(FancyArrowPatch((3.25, y0 + 0.20), (3.9, 3.15),
                     arrowstyle="-|>", mutation_scale=11,
                     color=MUTED, lw=1.2, connectionstyle="arc3,rad=0.04"))

    # scores as horizontal bars on the right (THE OUTPUT)
    bar_x0 = 6.55
    bar_max = 3.0          # 10 - 6.55 ~ 3.45 available
    rmax = 1.4
    def scorebar(y, score, label, color, tag):
        w = bar_max * (score / rmax)
        ax.add_patch(FancyArrowPatch((6.20, y + 0.22), (bar_x0 - 0.05, y + 0.22),
                     arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.2))
        bar = FancyBboxPatch((bar_x0, y), w, 0.44,
                             boxstyle="round,pad=0.0,rounding_size=0.03",
                             fc=color, ec=INK, lw=1.0)
        ax.add_patch(bar)
        ax.text(bar_x0 + w + 0.12, y + 0.22, f"{score:.1f}", va="center",
                fontsize=13, fontweight="bold", color=INK)
        ax.text(bar_x0, y + 0.62, label, fontsize=9.2, color=color,
                family="sans-serif", fontweight="bold")
        ax.text(bar_x0, y - 0.24, tag, fontsize=8.6, color=MUTED,
                family="sans-serif", style="italic")

    scorebar(3.15, 1.2, "chosen  →  higher score", SAGE, r"the preferred answer $y_w$")
    scorebar(1.35, 0.5, "rejected  →  lower score", RUST, r"the worse answer $y_l$")

    # Bradley-Terry footnote strip
    ax.add_patch(FancyBboxPatch((0.15, 0.02), 9.7, 0.55,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 fc="#EEEBDF", ec="#C9C4B5", lw=1.0))
    ax.text(0.32, 0.30,
            r"Train it so preferred beats rejected:  $P(A \succ B) = \sigma(r_A - r_B) = \sigma(1.2 - 0.5) = \sigma(0.7) \approx 0.67$   (Bradley–Terry)",
            fontsize=10.2, color=INK, va="center")

    save(fig, "reward_model_io.svg")


if __name__ == "__main__":
    reward_model_output()
    print("done.")
