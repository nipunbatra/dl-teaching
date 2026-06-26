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
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.axis("off")
    bands = [
        (1956, 1986, "#EADDCB", "ERA 1 · symbolic & perceptrons", "hand rules · 1-layer nets"),
        (1986, 2012, "#D9E0DA", "ERA 2 · shallow learning", "feature engineering + SVM / backprop"),
        (2012, 2026, "#EAD3C5", "ERA 3 · deep learning", r"learned features · AlexNet $\to$ LLMs"),
    ]
    x0, x1 = 1956, 2026
    def X(y): return (y - x0) / (x1 - x0)
    for (s, e, col, title, sub) in bands:
        ax.add_patch(Rectangle((X(s), 0.30), X(e)-X(s), 0.40, facecolor=col,
                     edgecolor="white", lw=2))
        xm = (X(s)+X(e))/2
        ax.text(xm, 0.605, title, ha="center", color=INK, fontsize=12.5, fontweight="bold")
        ax.text(xm, 0.45, sub, ha="center", color=MUTED, fontsize=10.5)
        ax.text(X(s)+0.005, 0.255, f"{s}", ha="left", color=MUTED, fontsize=10)
    ax.text(X(2026), 0.255, "2026", ha="right", color=MUTED, fontsize=10)

    # milestones
    miles = [(1958, "Perceptron"), (1986, "Backprop"), (1998, "LeNet"),
             (2012, "AlexNet"), (2017, "Transformer"), (2020, "GPT-3"), (2024, "LLMs+diffusion")]
    for (yr, name) in miles:
        ax.plot([X(yr)], [0.30], marker="o", ms=7, color=RUST, zorder=5)
        ax.text(X(yr), 0.20, name, ha="center", color=INK, fontsize=9.5, rotation=0)

    ax.annotate("", xy=(X(2012), 0.74), xytext=(X(2012), 0.86),
                arrowprops=dict(arrowstyle="-|>", color=RUST, lw=2))
    ax.text(X(2012), 0.90, "2012 · the turning point", ha="center", color=RUST, fontsize=12, fontweight="bold")
    ax.text(0.5, 0.05, "Takeaway · in 2012, learned representations replaced hand-engineered features — and never gave the lead back.",
            ha="center", color=INK, fontsize=12, style="italic")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1)
    save(fig, "dl_timeline.svg", tight=False)


# ---- 4. stacked linear collapses to one line -------------------------------
def stacked_linear_collapses():
    rng = np.random.default_rng(2)
    A = rng.normal([-1, -0.5], 0.5, (40, 2))
    B = rng.normal([1.1, 0.7], 0.5, (40, 2))
    xs = np.linspace(-2.6, 2.6, 10); line = -0.9*xs + 0.1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for ax, ttl in [(a1, "1 linear layer"), (a2, "5 stacked linear layers")]:
        ax.scatter(*A.T, s=26, color=SLATE); ax.scatter(*B.T, s=26, color=RUST)
        ax.plot(xs, line, color=SAGE, lw=2.6)
        ax.set_title(ttl, fontsize=13, color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); _clean(ax)
    a2.text(0, -2.6, "exactly the same line", ha="center", color=RUST, fontsize=12)
    fig.suptitle("Without a non-linearity, depth buys nothing — the boundary is always one straight line",
                 fontsize=13, color=INK)
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


# ---- 7. magnifying-glass analogy -------------------------------------------
def magnifying_glass():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0), layout="constrained")
    for a in (a1, a2):
        a.axis("off"); a.set_xlim(0, 1); a.set_ylim(0, 1)

    # left: two lenses in series = bigger but still linear
    a1.set_title("two linear layers = one (bigger) linear map", fontsize=12.5, color=INK)
    for cx in (0.30, 0.55):
        a1.add_patch(Polygon([(cx-0.04, 0.5), (cx, 0.74), (cx+0.04, 0.5),
                              (cx, 0.26)], closed=True, facecolor=SAGE_FILL,
                              edgecolor=SLATE, lw=1.4))
    a1.annotate("", xy=(0.22, 0.5), xytext=(0.06, 0.5),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    a1.annotate("", xy=(0.92, 0.5), xytext=(0.63, 0.5),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    a1.text(0.5, 0.12, "still just a straight-through scaling — no new patterns",
            ha="center", color=MUTED, fontsize=11)

    # right: a prism bends the rays = new features
    a2.set_title("linear + non-linearity = bends the space", fontsize=12.5, color=INK)
    a2.add_patch(Polygon([(0.42, 0.30), (0.58, 0.30), (0.50, 0.72)], closed=True,
                 facecolor="#EAD3C5", edgecolor=RUST, lw=1.6))
    a2.annotate("", xy=(0.46, 0.5), xytext=(0.10, 0.5),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    for dy, col in [(0.16, RUST), (0.02, SAGE), (-0.12, SLATE)]:
        a2.annotate("", xy=(0.92, 0.5+dy), xytext=(0.54, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5))
    a2.text(0.5, 0.12, "splits the input into new directions a linear stack can't reach",
            ha="center", color=MUTED, fontsize=11)
    save(fig, "magnifying_glass.svg", tight=False)


if __name__ == "__main__":
    print("Generating lec01 replacement figures...")
    linear_vs_nonlinear_data()
    pixel_shift_fail()
    dl_timeline()
    stacked_linear_collapses()
    feature_transform()
    linear_to_neuron()
    magnifying_glass()
    print("Done.")
