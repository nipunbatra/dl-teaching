"""Generate figures for Lecture 7: CNN Deep Dive & Classic Architectures.

Matches the lec00 CREAM palette (warm parchment + ink, rust/sage/slate) so the
new matplotlib figures sit beside the existing hand-authored lec07 SVGs without
clashing. Outputs SVG into figures/lec07/svg/.

Three "Andrew-Ng, output-first" figures:
  1. conv_numeric      — one 3x3 filter slides over a patch, the actual
                         multiply-add produces ONE number, and the whole
                         feature map "lights up" on the edge (concrete numbers).
  2. feature_maps_stack — what a conv layer *produces*: a STACK of feature
                         maps, one per filter (the output, first).
  3. feature_hierarchy  — edges -> textures -> parts -> objects.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, Ellipse
from pathlib import Path

# ---- palette (from anthropic-theme.css) -------------------------------------
PAPER = "#F7F3E9"
PAPER_ALT = "#EFE7D4"
INK = "#161513"
MUTED = "#5F5C54"
RUST = "#B85A3E"
SAGE = "#5F8573"
SLATE = "#37535F"
SAGE_FILL = "#9FB8AC"
RUST_FILL = "#EAD3C5"

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

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec07" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name, tight=True):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {name}")


# ---- 1. one filter -> one number -> one feature map (OUTPUT FIRST) ----------
def conv_numeric():
    fig, ax = plt.subplots(figsize=(11.6, 4.9))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(8.25, 6.95,
            r"One filter  $\rightarrow$  one number  $\rightarrow$  one feature map",
            ha="center", va="top", fontsize=16, color=INK)

    # ---- INPUT 5x5 (a vertical edge: dark left, bright right) ----
    inp = np.array([[0, 0, 9, 9, 9]] * 5)
    x0, ytop, c = 0.5, 5.9, 0.62
    ax.text(x0 + 2.5 * c, ytop + 0.42, "INPUT  5×5", ha="center",
            color=MUTED, fontsize=11.5)
    for r in range(5):
        for cc in range(5):
            v = inp[r, cc]
            fill = SLATE if v == 0 else PAPER_ALT
            tcol = PAPER if v == 0 else INK
            ax.add_patch(Rectangle((x0 + cc * c, ytop - (r + 1) * c), c, c,
                         fc=fill, ec=MUTED, lw=0.7))
            ax.text(x0 + cc * c + c / 2, ytop - (r + 0.5) * c, str(v),
                    ha="center", va="center", fontsize=11, color=tcol)
    # highlight the top-left 3x3 window
    ax.add_patch(Rectangle((x0, ytop - 3 * c), 3 * c, 3 * c, fill=False,
                 ec=RUST, lw=2.6))
    ax.text(x0 + 2.5 * c + 0.15, ytop - 5 * c - 0.28, "one window",
            ha="right", color=RUST, fontsize=10.5)

    # ---- KERNEL 3x3 (vertical-edge detector) ----
    ker = np.array([[-1, 0, 1]] * 3)
    kx, kt, kc = 4.9, 5.5, 0.66
    ax.text(kx + 1.5 * kc, kt + 0.42, "FILTER  3×3", ha="center",
            color=SAGE, fontsize=11.5)
    for r in range(3):
        for cc in range(3):
            v = ker[r, cc]
            fill = RUST_FILL if v < 0 else (PAPER if v == 0 else "#D8E3DE")
            ax.add_patch(Rectangle((kx + cc * kc, kt - (r + 1) * kc), kc, kc,
                         fc=fill, ec=SAGE, lw=0.9))
            ax.text(kx + cc * kc + kc / 2, kt - (r + 0.5) * kc,
                    f"{v:+d}", ha="center", va="center", fontsize=12, color=INK)
    ax.text(kx + 1.5 * kc, kt - 3 * kc - 0.28, "vertical-edge detector",
            ha="center", color=SAGE, fontsize=10.5)

    # ---- the arithmetic ----
    ex, ey = 3.55, 1.35
    ax.text(ex, ey + 0.55, "dot product of window × filter, summed:",
            ha="left", color=MUTED, fontsize=11.5)
    ax.text(ex, ey,
            r"$(0{\cdot}{-}1 + 0{\cdot}0 + 9{\cdot}1)\times 3\ \mathrm{rows} = \mathbf{27}$",
            ha="left", color=INK, fontsize=14)

    # ---- OUTPUT 3x3 feature map ----
    out = np.array([[27, 27, 0]] * 3)
    ox, ot, oc = 11.7, 5.4, 0.82
    ax.text(ox + 1.5 * oc, ot + 0.42, "FEATURE MAP  3×3", ha="center",
            color=RUST, fontsize=11.5)
    for r in range(3):
        for cc in range(3):
            v = out[r, cc]
            # bright (edge) -> sage fill, dark (0) -> pale
            fill = SAGE if v == 27 else PAPER
            tcol = PAPER if v == 27 else MUTED
            ax.add_patch(Rectangle((ox + cc * oc, ot - (r + 1) * oc), oc, oc,
                         fc=fill, ec=MUTED, lw=0.8))
            ax.text(ox + cc * oc + oc / 2, ot - (r + 0.5) * oc, str(v),
                    ha="center", va="center", fontsize=12, color=tcol)
    # highlight the cell we just computed (top-left = 27)
    ax.add_patch(Rectangle((ox, ot - oc), oc, oc, fill=False, ec=RUST, lw=2.6))
    ax.text(ox + 1.5 * oc, ot - 3 * oc - 0.28,
            "bright = edge here · dark = no edge", ha="center",
            color=MUTED, fontsize=10.5)

    # ---- arrows ----
    ax.add_patch(FancyArrowPatch((x0 + 3 * c + 0.05, ytop - 1.5 * c),
                 (kx - 0.15, kt - 1.5 * kc), arrowstyle="-|>",
                 mutation_scale=13, color=MUTED, lw=1.4))
    ax.add_patch(FancyArrowPatch((ex + 3.1, ey + 0.1), (ox - 0.15, ot - 0.5 * oc),
                 arrowstyle="-|>", mutation_scale=13, color=RUST, lw=1.6,
                 connectionstyle="arc3,rad=-0.15"))

    ax.text(8.25, 0.28,
            "Slide the SAME 3×3 filter over every window — each output "
            "cell is one dot-product. The map lights up where the edge is.",
            ha="center", color=MUTED, fontsize=11, style="italic")

    save(fig, "conv_numeric.svg")


# ---- 2. a conv layer produces a STACK of feature maps (OUTPUT FIRST) --------
def _conv2d_valid(img, ker):
    H, W = img.shape
    kh, kw = ker.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * ker)
    return out


def feature_maps_stack():
    # synthetic input scene with a vertical edge, a horizontal bar, and a blob
    img = np.zeros((16, 16))
    img[:, 8:] = 1.0
    img[3:6, :] = 1.0
    yy, xx = np.ogrid[:16, :16]
    img[((xx - 11) ** 2 + (yy - 11) ** 2) < 6] = 1.0

    vert = np.array([[-1, 0, 1]] * 3)
    horz = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    blob = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernels = [vert, horz, blob]
    labels = ["vertical\nedge", "horizontal\nedge", "blob"]
    maps = [np.abs(_conv2d_valid(img, k)) for k in kernels]

    warm = matplotlib.colors.LinearSegmentedColormap.from_list(
        "warm", [PAPER, "#E7CBBB", RUST])

    fig = plt.figure(figsize=(11.6, 4.7))
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off")
    bg.set_xlim(0, 1); bg.set_ylim(0, 1)
    bg.text(0.5, 0.95,
            "A conv layer outputs a STACK of feature maps — one per filter",
            ha="center", va="top", fontsize=16, color=INK, style="italic")

    # input image
    axin = fig.add_axes([0.03, 0.30, 0.20, 0.46])
    axin.imshow(img, cmap="bone_r"); axin.set_xticks([]); axin.set_yticks([])
    for s in axin.spines.values():
        s.set_edgecolor(SLATE); s.set_linewidth(1.4)
    bg.text(0.13, 0.235, "input image\n(H×W×3)", ha="center",
            va="top", color=SLATE, fontsize=11)

    # filters column
    for i, (k, lab) in enumerate(zip(kernels, labels)):
        axk = fig.add_axes([0.335, 0.60 - i * 0.235, 0.075, 0.16])
        axk.imshow(k, cmap="PuOr", vmin=-4, vmax=4)
        axk.set_xticks([]); axk.set_yticks([])
        for s in axk.spines.values():
            s.set_edgecolor(SAGE); s.set_linewidth(1.1)
        bg.text(0.305, 0.68 - i * 0.235, lab, ha="right", va="center",
                color=SAGE, fontsize=10)
    bg.text(0.37, 0.865, "K filters", ha="center", color=SAGE, fontsize=11.5)

    # stacked feature maps (offset "deck of cards")
    for i, m in enumerate(maps):
        off = (2 - i) * 0.035
        axm = fig.add_axes([0.62 + off, 0.28 + off, 0.28, 0.50])
        axm.imshow(m, cmap=warm); axm.set_xticks([]); axm.set_yticks([])
        for s in axm.spines.values():
            s.set_edgecolor(RUST); s.set_linewidth(1.3)
    bg.text(0.80, 0.235, "K feature maps  (H′×W′×K)\n"
            "each filter lights up on its own feature", ha="center", va="top",
            color=RUST, fontsize=11)

    # arrows
    bg.add_patch(FancyArrowPatch((0.245, 0.53), (0.30, 0.53),
                 arrowstyle="-|>", mutation_scale=15, color=MUTED, lw=1.6))
    bg.add_patch(FancyArrowPatch((0.435, 0.53), (0.60, 0.53),
                 arrowstyle="-|>", mutation_scale=15, color=MUTED, lw=1.6))

    save(fig, "feature_maps_stack.svg", tight=False)


# ---- 3. feature hierarchy: edges -> textures -> parts -> objects ------------
def feature_hierarchy():
    fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.7))
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_aspect("equal")
        ax.add_patch(Rectangle((0.04, 0.04), 0.92, 0.92, fc=PAPER,
                     ec=MUTED, lw=1.0))

    titles = ["Conv1 · edges", "Conv2–3 · textures",
              "Conv4 · parts", "Conv5 · objects"]
    cols = [SLATE, SAGE, RUST, INK]

    # (a) edges: oriented line segments (Gabor-like)
    ax = axes[0]
    rng = np.random.default_rng(2)
    for gx in np.linspace(0.20, 0.80, 3):
        for gy in np.linspace(0.22, 0.78, 3):
            th = rng.uniform(0, np.pi)
            dx, dy = 0.11 * np.cos(th), 0.11 * np.sin(th)
            ax.plot([gx - dx, gx + dx], [gy - dy, gy + dy],
                    color=SLATE, lw=2.4, solid_capstyle="round")

    # (b) textures: repeated motif (stripes / crosshatch)
    ax = axes[1]
    for gy in np.linspace(0.18, 0.82, 6):
        ax.plot([0.14, 0.86], [gy, gy], color=SAGE, lw=1.8)
    for gx in np.linspace(0.18, 0.82, 6):
        ax.plot([gx, gx], [0.14, 0.86], color=SAGE, lw=1.0, alpha=0.5)

    # (c) parts: an eye + a wheel
    ax = axes[2]
    ax.add_patch(Ellipse((0.5, 0.66), 0.62, 0.26, fc="none", ec=RUST, lw=2.2))
    ax.add_patch(Circle((0.5, 0.66), 0.085, fc=RUST, ec=RUST))
    ax.add_patch(Circle((0.5, 0.28), 0.15, fc="none", ec=RUST, lw=2.2))
    for th in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        ax.plot([0.5, 0.5 + 0.15 * np.cos(th)], [0.28, 0.28 + 0.15 * np.sin(th)],
                color=RUST, lw=1.3)

    # (d) objects: a simple face assembled from parts
    ax = axes[3]
    ax.add_patch(Circle((0.5, 0.5), 0.34, fc="none", ec=INK, lw=2.4))
    ax.add_patch(Circle((0.38, 0.60), 0.05, fc=INK))
    ax.add_patch(Circle((0.62, 0.60), 0.05, fc=INK))
    th = np.linspace(0.15 * np.pi, 0.85 * np.pi, 40)
    ax.plot(0.5 + 0.16 * np.cos(-th) * 1.0, 0.40 + 0.10 * np.sin(-th),
            color=INK, lw=2.2)

    for ax, t, cparts in zip(axes, titles, cols):
        ax.set_title(t, fontsize=12.5, color=cparts, pad=6)

    # "compose ->" arrows between panels
    for xf in [0.285, 0.51, 0.735]:
        fig.patches.append(FancyArrowPatch(
            (xf - 0.012, 0.5), (xf + 0.012, 0.5), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=16, color=MUTED, lw=1.8))

    fig.suptitle("Feature hierarchy — deep layers COMPOSE simple features "
                 "into complex ones", fontsize=14.5, color=INK, y=1.02)
    fig.text(0.5, -0.02, "each layer's larger receptive field lets it assemble "
             "the previous layer's features", ha="center", color=MUTED,
             fontsize=11, style="italic")
    save(fig, "feature_hierarchy.svg", tight=True)


if __name__ == "__main__":
    print("Generating lec07 figures...")
    conv_numeric()
    feature_maps_stack()
    feature_hierarchy()
    print("Done.")
