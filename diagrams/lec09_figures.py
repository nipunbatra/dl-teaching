"""Generate figures for Lecture 9: Detection & Segmentation.

Matches the hand-authored lec00 SVG palette (warm parchment + ink, rust/sage/slate)
so new matplotlib figures sit beside the existing lec09 SVGs without clashing.
Outputs SVG into figures/lec09/svg/.

Three "output-first" (Ng-style) schematics:
  1. output_ladder      — classification -> localization -> detection -> segmentation,
                          drawn as a ladder of OUTPUT representations.
  2. detector_output    — the per-cell / per-anchor output vector, number by number.
  3. anchor_delta_decode — anchor box + predicted deltas -> decoded box.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from pathlib import Path

# ---- palette (from anthropic-theme.css) -------------------------------------
PAPER = "#F7F3E9"
INK = "#161513"
MUTED = "#5F5C54"
RUST = "#B85A3E"
SAGE = "#5F8573"
SLATE = "#37535F"
SAGE_FILL = "#9FB8AC"
PAPER_ALT = "#EFEADA"
RUST_FILL = "#EAD3C5"
SLATE_FILL = "#D8E3DE"

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

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec09" / "svg"
OUT.mkdir(parents=True, exist_ok=True)

MONO = {"family": "monospace"}


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name, tight=True):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {name}")


def _cellrow(ax, x0, y0, labels, w=0.62, h=0.62, fc=PAPER, ec=MUTED,
             fontsize=12, textcolor=INK, lw=1.1, gap=0.0):
    """Draw a row of labelled cells starting at (x0, y0); return right edge x."""
    x = x0
    for lab in labels:
        ax.add_patch(Rectangle((x, y0), w, h, fill=True, fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y0 + h / 2, lab, ha="center", va="center",
                fontsize=fontsize, color=textcolor)
        x += w + gap
    return x


# ---- 1. the ladder of outputs -----------------------------------------------
def output_ladder():
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.axis("off")
    ax.set_xlim(0, 15.4)
    ax.set_ylim(0, 9.2)

    rows = [
        # (y, task, subtitle)
        (7.6, "Classification", "one label"),
        (5.4, "+ Localization", "one label + one box"),
        (3.2, "Detection", "many boxes, variable count"),
        (1.0, "Segmentation", "a label on every pixel"),
    ]

    # column x anchors
    x_task = 0.2
    x_out = 4.5      # output schematic starts here
    x_says = 12.7    # plain-English "reads as"

    ax.text(x_task, 8.9, "TASK", fontsize=11, color=MUTED, style="italic")
    ax.text(x_out, 8.9, "WHAT THE NETWORK OUTPUTS", fontsize=11, color=MUTED, style="italic")
    ax.text(x_says, 8.9, "READS AS", fontsize=11, color=MUTED, style="italic")

    for (y, task, sub) in rows:
        ax.text(x_task, y + 0.42, task, fontsize=14.5, color=INK, weight="bold")
        ax.text(x_task, y - 0.02, sub, fontsize=10.5, color=MUTED, style="italic")

    # --- row 1: classification -> K class scores
    y = 7.6
    _cellrow(ax, x_out, y, ["c₁", "c₂", "c₃", "…", "c_K"], fc=SLATE_FILL, ec=SLATE)
    ax.text(x_out + 1.7, y - 0.55, "K class scores", ha="center", fontsize=10, color=SLATE)
    ax.text(x_says, y + 0.2, "“cat”", fontsize=13, color=INK)

    # --- row 2: localization -> class scores + box
    y = 5.4
    xr = _cellrow(ax, x_out, y, ["c₁", "…", "c_K"], fc=SLATE_FILL, ec=SLATE)
    _cellrow(ax, xr + 0.35, y, ["x", "y", "w", "h"], fc=RUST_FILL, ec=RUST)
    ax.text(x_out + 0.9, y - 0.55, "class", ha="center", fontsize=10, color=SLATE)
    ax.text(xr + 0.35 + 1.24, y - 0.55, "one box (4 numbers)", ha="center", fontsize=10, color=RUST)
    ax.text(x_says, y + 0.2, "“cat @ box”", fontsize=13, color=INK)

    # --- row 3: detection -> N copies of [obj, box, class]
    y = 3.2
    xc = x_out
    for k in range(2):
        _cellrow(ax, xc, y, ["p"], w=0.5, fc=SAGE_FILL, ec=SAGE, fontsize=10)
        _cellrow(ax, xc + 0.5, y, ["x", "y", "w", "h"], w=0.5, fc=RUST_FILL, ec=RUST, fontsize=10)
        _cellrow(ax, xc + 2.5, y, ["c…"], w=0.6, fc=SLATE_FILL, ec=SLATE, fontsize=10)
        xc += 3.55
    ax.text(xc + 0.05, y + 0.31, "…", fontsize=16, color=MUTED)
    ax.text(x_out + 3.4, y - 0.55, "N × [ obj · box · class ]  (one per grid cell / anchor, variable N)",
            ha="center", fontsize=10, color=MUTED)
    ax.text(x_says, y + 0.2, "“many objects”", fontsize=13, color=INK)

    # --- row 4: segmentation -> HxW grid of per-pixel labels
    y = 1.0
    grid = np.array([
        [0, 0, 1, 1, 1, 2],
        [0, 0, 1, 1, 2, 2],
        [0, 1, 1, 2, 2, 2],
    ])
    cmap = {0: PAPER_ALT, 1: RUST_FILL, 2: SAGE_FILL}
    ecmap = {0: MUTED, 1: RUST, 2: SAGE}
    cw = 0.52
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax.add_patch(Rectangle((x_out + j * cw, y + (2 - i) * cw), cw, cw,
                         fill=True, fc=cmap[v], ec=ecmap[v], lw=0.8))
    ax.text(x_out + 3 * cw, y - 0.55, "a class label at every (i, j) pixel",
            ha="center", fontsize=10, color=MUTED)
    ax.text(x_says, y + 0.2, "“which pixel\nis what”", fontsize=12.5, color=INK, va="center")

    # descending brace-ish arrow on the far left to show "outputs grow"
    ax.add_patch(FancyArrowPatch((0.05, 8.2), (0.05, 0.7), arrowstyle="-|>",
                 mutation_scale=14, color=RUST, lw=1.6))
    ax.text(-0.15, 4.5, "richer output", rotation=90, va="center", ha="center",
            fontsize=11, color=RUST, style="italic")

    ax.set_title("The ladder of outputs — same backbone, the OUTPUT is what changes",
                 fontsize=15, color=INK, loc="left", pad=8)
    save(fig, "output_ladder.svg")


# ---- 2. a detector's output, number by number -------------------------------
def detector_output():
    fig, ax = plt.subplots(figsize=(11.6, 4.7))
    ax.axis("off")
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 6.8)

    ax.set_title("A detector's output, number by number — one grid cell / anchor",
                 fontsize=15, color=INK, loc="left", pad=8)

    w = 1.05
    y = 4.6
    x = 0.4

    # objectness (1)
    ax.add_patch(Rectangle((x, y), w, w, fc=SAGE_FILL, ec=SAGE, lw=1.4))
    ax.text(x + w / 2, y + w / 2, r"$p_{obj}$", ha="center", va="center", fontsize=14, color=INK)
    obj_x0 = x
    x += w + 0.12

    # box (4)
    box_x0 = x
    for lab in ["x", "y", "w", "h"]:
        ax.add_patch(Rectangle((x, y), w, w, fc=RUST_FILL, ec=RUST, lw=1.4))
        ax.text(x + w / 2, y + w / 2, lab, ha="center", va="center", fontsize=14, color=INK)
        x += w + 0.12
    box_x1 = x - 0.12

    # class scores (K)
    cls_x0 = x
    for lab in ["c₁", "c₂", "c₃", "…", "c_K"]:
        ax.add_patch(Rectangle((x, y), w, w, fc=SLATE_FILL, ec=SLATE, lw=1.4))
        ax.text(x + w / 2, y + w / 2, lab, ha="center", va="center", fontsize=13, color=INK)
        x += w + 0.12
    cls_x1 = x - 0.12

    # group braces + labels underneath
    def brace(x0, x1, text, color):
        yb = y - 0.28
        ax.plot([x0, x0, x1, x1], [yb, yb - 0.16, yb - 0.16, yb], color=color, lw=1.3)
        ax.text((x0 + x1) / 2, yb - 0.5, text, ha="center", va="top", fontsize=11.5, color=color)

    brace(obj_x0, obj_x0 + w, "objectness (1)\n“is anything here?”", SAGE)
    brace(box_x0, box_x1, "box (4)\n“where & how big?”", RUST)
    brace(cls_x0, cls_x1, "class scores (K)\n“what is it?”", SLATE)

    # concrete example row
    ye = 1.35
    ax.text(0.4, ye + w + 0.25, "concrete example —  a cell that saw a dog (class 3):",
            fontsize=12, color=MUTED, style="italic")
    vals = ["0.9", "0.5", "0.5", "0.2", "0.3", "0.1", "0.2", "0.6", "0.1", "…"]
    cols = [SAGE_FILL] + [RUST_FILL] * 4 + [SLATE_FILL] * 5
    ecs = [SAGE] + [RUST] * 4 + [SLATE] * 5
    xe = 0.4
    for v, fc, ec in zip(vals, cols, ecs):
        ax.add_patch(Rectangle((xe, ye), w, w, fc=fc, ec=ec, lw=1.2))
        ax.text(xe + w / 2, ye + w / 2, v, ha="center", va="center", fontsize=13, color=INK,
                weight="bold" if v == "0.6" else "normal")
        xe += w + 0.12

    ax.annotate("90% sure\nsomething's here", xy=(0.4 + w / 2, ye), xytext=(0.4 + w / 2, ye - 1.0),
                ha="center", va="top", fontsize=10, color=SAGE,
                arrowprops=dict(arrowstyle="-|>", color=SAGE, lw=1.1))
    ax.annotate("centred in cell,\n0.2×0.3 of image", xy=(box_x0 + 2 * w, ye),
                xytext=(box_x0 + 2 * w, ye - 1.0), ha="center", va="top", fontsize=10, color=RUST,
                arrowprops=dict(arrowstyle="-|>", color=RUST, lw=1.1))
    ax.annotate("argmax = class 3\n(“dog”)", xy=(cls_x0 + 2.2 * w, ye),
                xytext=(cls_x0 + 2.2 * w, ye - 1.0), ha="center", va="top", fontsize=10, color=SLATE,
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.1))

    ax.text(cls_x1 + 0.5, y + w / 2, "full tensor =\nS × S × (B·5 + K)",
            va="center", ha="left", fontsize=11.5, color=INK, style="italic")

    save(fig, "detector_output.svg")


# ---- 3. anchor + deltas -> decoded box --------------------------------------
def anchor_delta_decode():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.6),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    # --- LEFT: picture in image coords (y down) ---
    ax = axL
    ax.set_xlim(40, 340)
    ax.set_ylim(320, 150)   # inverted -> image-like (origin top-left)
    ax.set_aspect("equal")
    ax.set_title(r"anchor  $\rightarrow$  decoded box", fontsize=13, color=INK, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_color(MUTED)

    # anchor: centre (120,240), w80 h100 -> corners (80,190)-(160,290)
    ax.add_patch(Rectangle((80, 190), 80, 100, fill=False, ec=SLATE, lw=2.0,
                 linestyle=(0, (5, 3))))
    ax.plot(120, 240, "o", color=SLATE, ms=6)
    ax.text(120, 300, "anchor\n(120, 240, 80, 100)", ha="center", va="top",
            fontsize=10.5, color=SLATE)

    # decoded box: centre (128,220), w108 h90 -> corners (74,175)-(182,265)
    ax.add_patch(Rectangle((74, 175), 108, 90, fill=False, ec=RUST, lw=2.4))
    ax.plot(128, 220, "*", color=RUST, ms=12)
    ax.text(215, 178, "decoded box\n(128, 220, 108, 90)", ha="left", va="center",
            fontsize=10.5, color=RUST)

    # nudge arrow centre->centre
    ax.add_patch(FancyArrowPatch((120, 240), (128, 220), arrowstyle="-|>",
                 mutation_scale=13, color=INK, lw=1.4))
    ax.text(150, 250, "small nudge", fontsize=10, color=MUTED, style="italic")

    # --- RIGHT: the decode equations ---
    ax = axR
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0, 9.4, "network predicts deltas  $(t_x, t_y, t_w, t_h) = (0.1, -0.2, 0.3, -0.1)$",
            fontsize=11.5, color=INK)

    lines = [
        (r"$b_x = x_a + t_x\, w_a$", r"$= 120 + 0.1\cdot 80 = 128$", RUST),
        (r"$b_y = y_a + t_y\, h_a$", r"$= 240 - 0.2\cdot 100 = 220$", RUST),
        (r"$b_w = w_a\, e^{t_w}$", r"$= 80\cdot e^{0.3}\approx 108$", SLATE),
        (r"$b_h = h_a\, e^{t_h}$", r"$= 100\cdot e^{-0.1}\approx 90$", SLATE),
    ]
    yy = 7.9
    for form, val, c in lines:
        ax.text(0.2, yy, form, fontsize=13.5, color=c)
        ax.text(4.4, yy, val, fontsize=12, color=INK)
        yy -= 1.55

    ax.text(0.2, 1.0, "centre: shift, scaled by anchor size\nsize: log-space, so width stays positive",
            fontsize=10.5, color=MUTED, style="italic")

    fig.suptitle("Decode: the box is the anchor, gently corrected",
                 fontsize=15, color=INK, x=0.02, ha="left", y=1.02)
    save(fig, "anchor_delta_decode.svg", tight=True)


if __name__ == "__main__":
    print("Generating lec09 figures...")
    output_ladder()
    detector_output()
    anchor_delta_decode()
    print("Done.")
