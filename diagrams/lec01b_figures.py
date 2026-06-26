"""Replacement / new figures for Lecture 1, in the anthropic parchment palette
(matches lec00 figures and blends with the cream slide background).
Outputs SVG into figures/lec01/svg/, overwriting the old white-bg versions.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Polygon
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

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec01" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name, tight=True):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {name}")


# ---- 1. linear separable vs curved -----------------------------------------
def linear_vs_nonlinear_data():
    rng = np.random.default_rng(1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")

    # left: two separable blobs + a line that works
    A = rng.normal([-1.2, -0.6], 0.45, (40, 2))
    B = rng.normal([1.2, 0.8], 0.45, (40, 2))
    a1.scatter(*A.T, s=28, color=SLATE, label="class A")
    a1.scatter(*B.T, s=28, color=RUST, label="class B")
    xs = np.linspace(-2.6, 2.6, 10)
    a1.plot(xs, -0.9*xs + 0.2, color=SAGE, lw=2.6)
    a1.set_title(r"linearly separable  $\to$  a straight line works", fontsize=13, color=INK)
    a1.legend(frameon=False, fontsize=10, loc="lower right")

    # right: concentric rings, no line works
    th = rng.uniform(0, 2*np.pi, 70)
    inner = np.c_[0.7*np.cos(th[:35]), 0.7*np.sin(th[:35])] + rng.normal(0, 0.08, (35, 2))
    outer = np.c_[2.0*np.cos(th[35:]), 2.0*np.sin(th[35:])] + rng.normal(0, 0.12, (35, 2))
    a2.scatter(*inner.T, s=28, color=SLATE)
    a2.scatter(*outer.T, s=28, color=RUST)
    a2.plot(xs, 0.3*xs + 0.4, color=SAGE, lw=2.6, ls="--")
    a2.text(0, -2.5, "any line splits each ring in half", ha="center", color=MUTED, fontsize=11)
    a2.set_title(r"curved  $\to$  no straight line can separate", fontsize=13, color=INK)

    for a in (a1, a2):
        a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal"); _clean(a)
    save(fig, "linear_vs_nonlinear_data.svg", tight=False)


# ---- 2. pixel shift breaks a linear classifier -----------------------------
def pixel_shift_fail():
    def blob(shift):
        g = np.zeros((8, 8))
        r, c = 1 + shift, 1 + shift
        g[r:r+3, c:c+4] = 1.0
        g[r+3, c+1:c+3] = 1.0   # little tail
        return g
    A, B = blob(0), blob(3)

    fig = plt.figure(figsize=(11, 4.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1])

    for col, (img, name) in enumerate([(A, "image A · object top-left"),
                                       (B, "image B · same object, shifted")]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img, cmap="bone_r", vmin=0, vmax=1)
        ax.set_title(name, fontsize=12, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        axv = fig.add_subplot(gs[1, col])
        axv.imshow(img.reshape(1, -1), cmap="bone_r", aspect="auto", vmin=0, vmax=1)
        axv.set_yticks([]); axv.set_xticks([])
        axv.set_xlabel("flattened 64-pixel vector", fontsize=10)

    ax = fig.add_subplot(gs[:, 2])
    ax.axis("off")
    ax.text(0.5, 0.86, "the linear classifier sees", ha="center", color=INK, fontsize=13)
    ax.text(0.5, 0.66, r"$\|\,x_A - x_B\,\| $  is large", ha="center", color=RUST, fontsize=16)
    ax.text(0.5, 0.45,
            "one weight per pixel · no\nnotion that B is just A moved.\nIt must relearn the object at\nevery position, from scratch.",
            ha="center", va="top", color=MUTED, fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle("Why a linear classifier over raw pixels fails — no translation prior",
                 fontsize=14, color=INK)
    save(fig, "pixel_shift_fail.svg", tight=False)


# ---- 3. three eras of deep learning ----------------------------------------
def dl_timeline():
    fig, ax = plt.subplots(figsize=(13.8, 4.7))
    ax.axis("off")
    bands = [
        (1956, 1986, "#EADDCB", "ERA 1 · symbolic & perceptrons", "hand-written rules · single-layer nets"),
        (1986, 2012, "#D9E0DA", "ERA 2 · shallow learning", "hand-engineered features + SVM / backprop"),
        (2012, 2026, "#EAD3C5", "ERA 3 · deep learning", "learned representations · AlexNet to LLMs"),
    ]
    x0, x1 = 1956, 2026
    def X(y): return (y - x0) / (x1 - x0)
    for (s, e, col, title, sub) in bands:
        ax.add_patch(Rectangle((X(s), 0.36), X(e)-X(s), 0.34, facecolor=col,
                     edgecolor="white", lw=2))
        xm = (X(s)+X(e))/2
        ax.text(xm, 0.605, title, ha="center", color=INK, fontsize=13, fontweight="bold")
        ax.text(xm, 0.46, sub, ha="center", color=MUTED, fontsize=11)
        ax.text(X(s)+0.004, 0.31, f"{s}", ha="left", color=MUTED, fontsize=10)
    ax.text(X(2026), 0.31, "2026", ha="right", color=MUTED, fontsize=10)

    # milestones — stagger above/below the band to avoid crowding
    miles = [(1958, "Perceptron", -1), (1986, "Backprop", -1), (1998, "LeNet", +1),
             (2012, "AlexNet", -1), (2017, "Transformer", +1), (2020, "GPT-3", -1),
             (2024, "LLMs + diffusion", +1)]
    for (yr, name, d) in miles:
        ax.plot([X(yr)], [0.36], marker="o", ms=8, color=RUST, zorder=5)
        if d > 0:
            ax.plot([X(yr), X(yr)], [0.70, 0.76], color=MUTED, lw=0.8)
            ax.text(X(yr), 0.785, name, ha="center", va="bottom", color=INK, fontsize=10)
        else:
            ax.plot([X(yr), X(yr)], [0.36, 0.28], color=MUTED, lw=0.8)
            ax.text(X(yr), 0.255, name, ha="center", va="top", color=INK, fontsize=10)

    ax.text(0.5, 0.04,
            "Takeaway · in 2012 learned representations replaced hand-engineered features — and never gave the lead back.",
            ha="center", color=INK, fontsize=12, style="italic")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1)
    save(fig, "dl_timeline.svg", tight=False)


# ---- 4. stacked linear collapses · nonlinear data, still a straight line ----
def stacked_linear_collapses():
    rng = np.random.default_rng(2)
    n = 70
    t1 = np.pi * rng.random(n)
    moon_a = np.c_[np.cos(t1), np.sin(t1)] + rng.normal(0, 0.10, (n, 2))
    t2 = np.pi * rng.random(n)
    moon_b = np.c_[1 - np.cos(t2), 1 - np.sin(t2) - 0.5] + rng.normal(0, 0.10, (n, 2))
    xs = np.linspace(-1.6, 2.6, 10); line = -0.6 * xs + 0.55   # one straight boundary

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.5), layout="constrained")
    for ax, ttl in [(a1, "1 linear layer"), (a2, "5 stacked linear layers")]:
        ax.scatter(*moon_a.T, s=24, color=SLATE)
        ax.scatter(*moon_b.T, s=24, color=RUST)
        ax.plot(xs, line, color=SAGE, lw=2.6)
        ax.set_title(ttl, fontsize=13, color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); _clean(ax)
    a2.text(0.5, -1.15, "exactly the same straight line", ha="center", color=RUST, fontsize=12)
    fig.suptitle(r"Curved data, but no non-linearity $\to$ every depth gives the same straight line (and misses the curve)",
                 fontsize=12.5, color=INK)
    save(fig, "stacked_linear_collapses.svg", tight=False)


# ---- 5. feature-space transformation makes data separable ------------------
def feature_transform():
    rng = np.random.default_rng(4)
    th = rng.uniform(0, 2*np.pi, 120)
    r_in = 0.8 + rng.normal(0, 0.08, 60)
    r_out = 2.0 + rng.normal(0, 0.10, 60)
    inner = np.c_[r_in*np.cos(th[:60]), r_in*np.sin(th[:60])]
    outer = np.c_[r_out*np.cos(th[60:]), r_out*np.sin(th[60:])]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
    a1.scatter(*inner.T, s=26, color=SLATE); a1.scatter(*outer.T, s=26, color=RUST)
    a1.set_title("input space · no straight line separates", fontsize=12.5, color=INK)
    a1.set_xlabel(r"$x_1$"); a1.set_ylabel(r"$x_2$"); a1.set_aspect("equal"); _clean(a1)

    # transform: (radius, angle) — classes become two horizontal bands
    def feat(P):
        r = np.sqrt((P**2).sum(1)); a = np.arctan2(P[:, 1], P[:, 0])
        return a, r
    ai, ri = feat(inner); ao, ro = feat(outer)
    a2.scatter(ai, ri, s=26, color=SLATE); a2.scatter(ao, ro, s=26, color=RUST)
    a2.axhline(1.4, color=SAGE, lw=2.6)
    a2.text(0, 1.5, "a straight line now separates", ha="center", color=SAGE, fontsize=11)
    a2.set_title("learned feature space · linearly separable", fontsize=12.5, color=INK)
    a2.set_xlabel("angle"); a2.set_ylabel("radius"); _clean(a2)
    fig.suptitle("A hidden layer reshapes the data so the final linear layer's job becomes trivial",
                 fontsize=13, color=INK)
    save(fig, "feature_transform.svg", tight=False)


# ---- 6. linear model -> neuron (steps) -------------------------------------
def linear_to_neuron():
    fig, ax = plt.subplots(figsize=(11, 3.4)); ax.axis("off")
    steps = [
        ("linear model", r"$z = \mathbf{w}^\top\mathbf{x}$", "weighted sum of inputs"),
        ("+ bias", r"$z = \mathbf{w}^\top\mathbf{x} + b$", "shift the threshold"),
        ("+ non-linearity", r"$a = \sigma(z)$", "bend / squash"),
        ("= a neuron", r"$a = \sigma(\mathbf{w}^\top\mathbf{x}+b)$", "the whole building block"),
    ]
    w = 0.215; gap = (1 - 4*w) / 3
    for i, (title, eq, sub) in enumerate(steps):
        x = i*(w+gap)
        col = SAGE_FILL if i < 3 else "#EAD3C5"
        ax.add_patch(Rectangle((x, 0.25), w, 0.5, facecolor=col, edgecolor=MUTED, lw=1))
        ax.text(x+w/2, 0.66, title, ha="center", color=INK, fontsize=12.5, fontweight="bold")
        ax.text(x+w/2, 0.50, eq, ha="center", color=INK, fontsize=14)
        ax.text(x+w/2, 0.33, sub, ha="center", color=MUTED, fontsize=10)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x+w, 0.5), (x+w+gap, 0.5),
                         arrowstyle="-|>", mutation_scale=14, color=RUST, lw=1.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "linear_to_neuron.svg", tight=False)


# ---- 7. linear keeps lines straight; a non-linearity bends them -------------
def magnifying_glass():
    gl = np.linspace(-1.4, 1.4, 8)
    t = np.linspace(-1.4, 1.4, 80)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.7), layout="constrained")

    # left: a linear (shear) map — every grid line stays straight
    for c in gl:
        a1.plot(c + 0.45 * t, t, color=SLATE, lw=1)        # transformed verticals
        a1.plot(t + 0.45 * c, c + 0 * t, color=SLATE, lw=1)  # transformed horizontals
    a1.set_title(r"stack of linear layers $\to$ lines stay straight", fontsize=12.5, color=INK)
    a1.text(0, -2.15, "compose linear maps and you still have a linear map —\nthe boundary can never curve",
            ha="center", color=MUTED, fontsize=10.5)

    # right: a non-linear warp — grid lines bend into curves
    def warp(x, y):
        return x + 0.5 * np.tanh(1.6 * y), y + 0.5 * np.tanh(1.6 * x)
    for c in gl:
        X, Y = warp(c + 0 * t, t); a2.plot(X, Y, color=RUST, lw=1)
        X, Y = warp(t, c + 0 * t); a2.plot(X, Y, color=RUST, lw=1)
    a2.set_title(r"add a non-linearity $\to$ lines bend", fontsize=12.5, color=INK)
    a2.text(0, -2.15, "σ warps the space so the next linear layer\ncan carve curved boundaries",
            ha="center", color=MUTED, fontsize=10.5)

    for a in (a1, a2):
        a.set_xlim(-2.2, 2.2); a.set_ylim(-2.4, 2.2)
        a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal"); _clean(a)
    save(fig, "magnifying_glass.svg", tight=False)


# ---- 8. activation functions grid (curve + formula each) -------------------
def activation_grid():
    z = np.linspace(-4, 4, 300)
    sig = 1 / (1 + np.exp(-z)); tanh = np.tanh(z); relu = np.maximum(0, z)
    lrelu = np.where(z > 0, z, 0.1 * z)
    gelu = z * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))
    silu = z * sig
    acts = [(sig, "Sigmoid", r"$\sigma(z)=\dfrac{1}{1+e^{-z}}$"),
            (tanh, "Tanh", r"$\tanh(z)$"),
            (relu, "ReLU", r"$\max(0,z)$"),
            (lrelu, "Leaky ReLU", r"$\max(0.1z,\,z)$"),
            (gelu, "GELU", r"$z\,\Phi(z)$"),
            (silu, "SiLU / Swish", r"$z\,\sigma(z)$")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout="constrained")
    for ax, (y, name, formula) in zip(axes.flat, acts):
        ax.axhline(0, color="#C9C4B5", lw=0.8); ax.axvline(0, color="#C9C4B5", lw=0.8)
        ax.plot(z, y, color=RUST, lw=2.6)
        ax.set_title(f"{name}", fontsize=13, color=INK)
        ax.text(0.04, 0.92, formula, transform=ax.transAxes, va="top", fontsize=13, color=SLATE)
        ax.set_xlim(-4, 4); ax.set_ylim(-1.5, 4); _clean(ax)
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "activation_grid.svg", tight=False)


# ---- 9. MLP architecture with explicit node counts -------------------------
def mlp_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.0)); ax.axis("off")
    layers = [("input", 784, 6), ("hidden 1", 256, 5), ("hidden 2", 256, 5), ("output", 10, 4)]
    xs = [0, 1, 2, 3]
    pos = {}
    for li, (name, count, shown) in enumerate(layers):
        ys = np.linspace(0.78, 0.22, shown)
        pos[li] = [(xs[li], y) for y in ys]
        col = SAGE_FILL if 0 < li < 3 else ("#EAD3C5" if li == 3 else "#E6D9C8")
        for (x, y) in pos[li]:
            ax.add_patch(plt.Circle((x, y), 0.045, facecolor=col, edgecolor=SLATE, lw=1.2, zorder=3))
        ax.text(xs[li], 0.88, name, ha="center", color=INK, fontsize=12.5, fontweight="bold")
        ax.text(xs[li], 0.10, f"{count} units", ha="center", color=RUST, fontsize=12)
        if shown < count:
            ax.text(xs[li], 0.13, r"$\vdots$", ha="center", color=MUTED, fontsize=16)
    # edges between consecutive layers
    for li in range(3):
        for (x1, y1) in pos[li]:
            for (x2, y2) in pos[li+1]:
                ax.plot([x1, x2], [y1, y2], color=MUTED, lw=0.35, alpha=0.5, zorder=1)
    # weight labels
    for li, lbl in zip(range(3), [r"$W_1{:}\,784\times256$", r"$W_2{:}\,256\times256$", r"$W_3{:}\,256\times10$"]):
        ax.text((xs[li]+xs[li+1])/2, 0.50, lbl, ha="center", color=INK, fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.2", fc=PAPER, ec=MUTED, lw=0.6))
    ax.set_title(r"MLP for MNIST · 784 $\to$ 256 $\to$ 256 $\to$ 10", fontsize=14, loc="left", color=INK)
    ax.set_xlim(-0.3, 3.3); ax.set_ylim(0.05, 0.95)
    save(fig, "mlp_architecture.svg", tight=False)


if __name__ == "__main__":
    print("Generating lec01 replacement figures...")
    linear_vs_nonlinear_data()
    pixel_shift_fail()
    dl_timeline()
    stacked_linear_collapses()
    feature_transform()
    linear_to_neuron()
    magnifying_glass()
    activation_grid()
    mlp_architecture()
    print("Done.")
