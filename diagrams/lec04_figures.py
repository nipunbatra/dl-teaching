"""Generate figures for Lecture 4: SGD, Momentum, Nesterov.

These COMPLEMENT the hand-authored lec04 SVGs (ravine_zigzag, momentum_ema,
nesterov_lookahead, ...). They fill the Andrew-Ng build-up at the FRONT of the
deck — the part before the loss-landscape geometry:

  (1) full-batch is too slow  -> sgd_vs_fullbatch_progress.svg
  (2) one SGD step = a noisy gradient from a small random minibatch
                              -> minibatch_noise.svg
  (3) that noise is OK because it is unbiased (bias vs variance)
                              -> gradient_variance.svg

Palette copied from diagrams/lec00_figures.py so these sit beside the existing
cream-theme figures without clashing. Outputs SVG into figures/lec04/svg/.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# ---- palette (from anthropic-theme.css / lec00_figures.py) ------------------
PAPER = "#F7F3E9"
INK = "#161513"
MUTED = "#5F5C54"
RUST = "#B85A3E"
SAGE = "#5F8573"
SLATE = "#37535F"
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

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec04" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name, tight=True):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {name}")


# ---- 1. the object: full-batch smooth path vs SGD noisy path ----------------
def minibatch_noise():
    """One SGD step uses a NOISY gradient from a small minibatch.
    Full-batch glides smoothly downhill; SGD jitters but gets there too."""
    # gently elongated bowl so both paths visibly curve toward the min
    a, b = 2.6, 0.55
    def grad(p):
        return np.array([a * p[0], b * p[1]])

    start = np.array([-2.7, 2.3])
    eta = 0.13
    rng = np.random.default_rng(1)

    # full-batch: exact gradient -> smooth descent
    fb = [start.copy()]
    p = start.copy()
    for _ in range(28):
        p = p - eta * grad(p)
        fb.append(p.copy())
    fb = np.array(fb)

    # SGD: gradient + minibatch noise -> jittery descent, same destination
    sg = [start.copy()]
    p = start.copy()
    for _ in range(28):
        noise = rng.normal(0, 0.85, 2)
        p = p - eta * (grad(p) + noise)
        sg.append(p.copy())
    sg = np.array(sg)

    fig, ax = plt.subplots(figsize=(9.2, 4.3))

    # contours of the bowl
    xs = np.linspace(-3.2, 3.2, 240)
    ys = np.linspace(-3.0, 3.0, 240)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.5 * (a * X**2 + b * Y**2)
    ax.contour(X, Y, Z, levels=10, colors="#CFCBBE", linewidths=0.9, zorder=1)

    ax.plot(fb[:, 0], fb[:, 1], "-", color=SLATE, lw=2.6, zorder=3,
            label="full-batch GD · exact gradient")
    ax.plot(sg[:, 0], sg[:, 1], "-", color=RUST, lw=1.6, alpha=0.9, zorder=2,
            label="SGD · one minibatch per step")
    ax.scatter(sg[:, 0], sg[:, 1], s=14, color=RUST, zorder=4, alpha=0.7)
    ax.scatter([start[0]], [start[1]], s=80, color=INK, zorder=5)
    ax.text(start[0] - 0.05, start[1] + 0.18, "start", color=INK, fontsize=11,
            ha="center")
    ax.scatter([0], [0], marker="*", s=280, color=SAGE, edgecolor=INK,
               lw=0.8, zorder=6)
    ax.text(0.28, -0.42, "min", color=SAGE, fontsize=11)

    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.0, 3.0)
    ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
    ax.set_title("One step of SGD uses a noisy gradient — cheap, jittery, still downhill",
                 fontsize=13.5, loc="left", color=INK, pad=10)
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    _clean(ax)
    save(fig, "minibatch_noise.svg")


# ---- 2. why the noise is OK: unbiased, variance shrinks with batch size ------
def gradient_variance():
    """Minibatch gradient = true gradient + zero-mean noise.
    Small batch: wide scatter (high variance). Large batch: tight (low variance).
    In BOTH the average points the true way -> unbiased."""
    rng = np.random.default_rng(7)
    g_true = np.array([2.3, 1.5])          # the full-batch gradient
    origin = np.array([0.0, 0.0])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), layout="constrained")

    for ax, (B, sd, tag) in zip(
        axes,
        [(8, 1.15, r"small batch  $B=8$"), (128, 0.30, r"large batch  $B=128$")],
    ):
        # a fan of minibatch gradient estimates around the true gradient
        est = g_true + rng.normal(0, sd, (16, 2))
        for e in est:
            ax.add_patch(FancyArrowPatch(origin, e, arrowstyle="-|>",
                         mutation_scale=8, color=RUST, lw=1.0, alpha=0.45,
                         zorder=2))
        # the true (full-batch) gradient
        ax.add_patch(FancyArrowPatch(origin, g_true, arrowstyle="-|>",
                     mutation_scale=16, color=SLATE, lw=3.0, zorder=4))
        ax.scatter([origin[0]], [origin[1]], s=40, color=INK, zorder=5)
        ax.text(g_true[0] + 0.15, g_true[1] + 0.12, "full-batch\ngradient",
                color=SLATE, fontsize=10.5, va="center")

        ax.set_title(tag, fontsize=12.5, color=INK)
        ax.set_xlim(-1.6, 4.4); ax.set_ylim(-1.6, 4.0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        _clean(ax)

    axes[0].text(-1.4, -1.35, "wide scatter = high variance", color=RUST,
                 fontsize=10.5)
    axes[1].text(-1.4, -1.35, "tight = low variance", color=RUST, fontsize=10.5)
    fig.suptitle(r"Minibatch gradient is unbiased:  $\mathbb{E}[\,g_{\mathrm{batch}}\,]=g_{\mathrm{full}}$"
                 "  — bigger batch only shrinks the noise, never the aim",
                 fontsize=13, color=INK)
    save(fig, "gradient_variance.svg", tight=False)


# ---- 3. why full-batch is too slow: progress per unit of compute -------------
def sgd_vs_fullbatch_progress():
    """Same compute budget (examples touched). Full-batch spends a whole pass on
    ONE update; SGD makes many cheap updates and drops the loss fast early on."""
    N = 5000          # dataset size
    B = 50            # minibatch size
    epochs = 6
    rng = np.random.default_rng(3)

    # full-batch: one update per epoch -> a coarse staircase in loss
    fb_x = [0.0]
    fb_y = [1.0]
    L = 1.0
    for e in range(epochs):
        L *= 0.72                     # one good step per full pass
        fb_x.append((e + 1) * N)
        fb_y.append(L)

    # SGD: N/B updates per epoch -> rapid early drop, then a noisy floor
    steps = epochs * (N // B)
    sg_x = np.arange(1, steps + 1) * B
    base = 1.0 * (0.9955 ** np.arange(1, steps + 1))
    sg_y = base * (1 + rng.normal(0, 0.02, steps))
    sg_y = np.clip(sg_y, 0.02, None)

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.plot(sg_x / N, sg_y, "-", color=RUST, lw=1.4, alpha=0.85,
            label=f"SGD · {N//B} cheap steps / pass")
    ax.step(np.array(fb_x) / N, fb_y, where="post", color=SLATE, lw=2.6,
            label="full-batch · 1 step / pass")
    ax.scatter(np.array(fb_x) / N, fb_y, s=40, color=SLATE, zorder=5)

    ax.annotate("SGD already low\nafter <1 pass",
                xy=(0.55, sg_y[int(0.55 * N / B)]), xytext=(2.7, 0.52),
                color=RUST, fontsize=11, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=RUST, lw=1.2,
                                connectionstyle="arc3,rad=0.2"))
    ax.annotate("full-batch: only\n1 step by here",
                xy=(1.0, fb_y[1]), xytext=(2.9, 0.88),
                color=SLATE, fontsize=11, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.2))

    ax.set_xlabel("compute  (passes over the data)")
    ax.set_ylabel("training loss")
    ax.set_title("Same compute, far more progress — why we go stochastic",
                 fontsize=13.5, loc="left", color=INK, pad=10)
    ax.set_xlim(0, epochs); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    _clean(ax)
    save(fig, "sgd_vs_fullbatch_progress.svg")


if __name__ == "__main__":
    print("Generating lec04 figures...")
    sgd_vs_fullbatch_progress()
    minibatch_noise()
    gradient_variance()
    print("Done.")
