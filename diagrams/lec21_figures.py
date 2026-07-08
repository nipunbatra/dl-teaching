"""Generate figures for Lecture 21: Diffusion Models — Theory.

Matches the hand-authored lec21 SVG palette (warm parchment + ink, rust/sage/slate)
so these matplotlib figures sit beside the existing hand-drawn SVGs without clashing.

These three COMPLEMENT the existing SVGs (they do not duplicate them):
  - existing diffusion_timeline.svg  · conceptual image strip, forward + reverse
  - existing alpha_schedule.svg       · a-bar_t vs t, linear vs cosine
  - existing forward_reverse.svg      · image squares noising / denoising
Here we add:
  1. training_loop     · the DDPM training procedure as a clean flow
  2. signal_vs_noise   · the mixing WEIGHTS sqrt(a-bar) and sqrt(1-a-bar) vs t
  3. reverse_sampling  · the sampling loop (algorithm) + a 2D data-emerges strip

Outputs SVG into figures/lec21/svg/.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ---- palette (from anthropic-theme.css, identical to lec00_figures.py) ------
PAPER = "#F7F3E9"
INK = "#161513"
MUTED = "#5F5C54"
RUST = "#B85A3E"
SAGE = "#5F8573"
SLATE = "#37535F"
SAGE_FILL = "#9FB8AC"
# soft tints used by the hand-drawn lec21 SVGs (keep everything on-palette)
SLATE_FILL = "#DCE3E6"
RUST_FILL = "#EAD3C5"
SAGE_TINT = "#D8E3DE"
CREAM_BOX = "#F6F0DF"
NEUTRAL = "#EEEBDF"

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

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec21" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name, tight=True):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {name}")


# ---- shared linear DDPM schedule (beta 1e-4 -> 0.02 over T=1000) ------------
def linear_alpha_bar(T=1000):
    betas = np.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    return np.cumprod(alphas)


def _box(ax, x, y, w, h, text, fc, ec, tc=INK, fs=12, mono=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.10",
        fc=fc, ec=ec, lw=1.6, mutation_aspect=0.55))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, family="monospace" if mono else "serif")


def _arrow(ax, p0, p1, color=INK, ls="-", rad=0.0, lw=1.6):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=15, color=color,
        lw=lw, linestyle=ls, connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2))


# ---- 1. the DDPM training loop, as a clean left-to-right flow ---------------
def training_loop():
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    ax.axis("off")
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 5.2)

    yc = 2.2          # main-row centre
    h = 1.15

    # main row boxes
    _box(ax, 0.2, yc - h / 2, 2.3, h,
         "clean image\n$x_0$", SLATE_FILL, SLATE, tc=SLATE, fs=13)
    _box(ax, 4.3, yc - h / 2, 3.9, h,
         r"add noise (closed form)" "\n" r"$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$",
         CREAM_BOX, RUST, tc=INK, fs=12.5)
    _box(ax, 9.9, yc - h / 2, 2.4, h,
         "U-Net\n" r"$\epsilon_\theta(x_t,\,t)$", RUST_FILL, RUST, tc=RUST, fs=13)
    _box(ax, 13.0, yc - h / 2, 2.0, h,
         r"MSE loss" "\n" r"$\|\epsilon-\hat\epsilon\|^2$", SAGE_TINT, SAGE, tc=SAGE, fs=12.5)

    # feeder boxes (t and epsilon) above the "add noise" box
    _box(ax, 4.3, 4.05, 1.75, 0.85,
         r"$t\sim\{1..T\}$", NEUTRAL, MUTED, tc=MUTED, fs=12)
    _box(ax, 6.35, 4.05, 1.85, 0.85,
         r"$\epsilon\sim\mathcal{N}(0,I)$", NEUTRAL, MUTED, tc=MUTED, fs=12)

    # arrows along the main row
    _arrow(ax, (2.5, yc), (4.3, yc), color=INK)
    _arrow(ax, (8.2, yc), (9.9, yc), color=INK)
    _arrow(ax, (12.3, yc), (13.0, yc), color=INK)
    # predicted-noise label on U-Net -> loss arrow
    ax.text(12.65, yc + 0.42, r"$\hat\epsilon$", ha="center", color=RUST, fontsize=13)

    # feeders drop into the add-noise box
    _arrow(ax, (5.15, 4.05), (5.15, yc + h / 2), color=MUTED, ls=(0, (4, 2)), lw=1.3)
    _arrow(ax, (7.25, 4.05), (6.6, yc + h / 2), color=MUTED, ls=(0, (4, 2)), lw=1.3)

    # true epsilon also feeds the loss (dashed, routed over the top)
    ax.add_patch(FancyArrowPatch(
        (8.2, 4.48), (14.0, yc + h / 2),
        arrowstyle="-|>", mutation_scale=14, color=SAGE, lw=1.4,
        linestyle=(0, (4, 2)),
        connectionstyle="arc3,rad=-0.28", shrinkA=2, shrinkB=2))
    ax.text(11.1, 4.72, r"true $\epsilon$ (the target)", ha="center",
            color=SAGE, fontsize=11, style="italic")

    ax.text(0.2, 0.35,
            "One clean image + one random step $t$ + one noise sample $\\epsilon$  =  one training step. "
            "No adversary, no second network.",
            color=MUTED, fontsize=11.5, style="italic")
    ax.set_title("DDPM training · make a noisy image, then ask the net to name the noise",
                 fontsize=14, loc="left", color=INK, pad=8)
    save(fig, "training_loop.svg")


# ---- 2. signal vs noise mixing weights vs t ---------------------------------
def signal_vs_noise():
    T = 1000
    ab = linear_alpha_bar(T)
    t = np.arange(T)
    sig = np.sqrt(ab)             # weight on x0
    noi = np.sqrt(1.0 - ab)       # weight on epsilon

    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    ax.plot(t, sig, color=RUST, lw=2.8, label=r"signal weight  $\sqrt{\bar\alpha_t}$  (on $x_0$)")
    ax.plot(t, noi, color=SLATE, lw=2.8, ls="--",
            label=r"noise weight  $\sqrt{1-\bar\alpha_t}$  (on $\epsilon$)")

    ax.fill_between(t, 0, sig, color=RUST, alpha=0.10)
    ax.fill_between(t, 0, noi, color=SLATE, alpha=0.08)

    # crossover: where signal == noise (a-bar = 0.5)
    xc = int(np.argmin(np.abs(ab - 0.5)))
    ax.axvline(xc, color=MUTED, ls=":", lw=1.3)
    ax.scatter([xc], [sig[xc]], s=60, color=INK, zorder=5)
    ax.annotate(f"signal = noise\n(t $\\approx$ {xc}, $\\bar\\alpha$=0.5)",
                xy=(xc, sig[xc]), xytext=(xc + 60, 0.78),
                color=INK, fontsize=11,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

    ax.text(60, 0.20, "mostly signal", color=RUST, fontsize=12, style="italic")
    ax.text(760, 0.20, "mostly noise", color=SLATE, fontsize=12, style="italic",
            ha="right")

    ax.text(500, 1.045,
            r"$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$"
            r"   $\Rightarrow$   weights$^2$ sum to 1 at every $t$",
            ha="center", color=MUTED, fontsize=12)

    ax.set_xlabel("timestep  t   (0 = clean,  1000 = pure noise)")
    ax.set_ylabel("mixing weight")
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.12)
    ax.set_title("As $t$ grows, signal fades and noise takes over",
                 fontsize=14, loc="left", color=INK, pad=10)
    ax.legend(frameon=False, fontsize=11, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)
    save(fig, "signal_vs_noise.svg")


# ---- 3. reverse sampling loop (algorithm) + 2D data-emerges strip -----------
def reverse_sampling():
    fig = plt.figure(figsize=(9.6, 4.5))
    gs = GridSpec(2, 4, height_ratios=[1.25, 1.0], hspace=0.42, wspace=0.28,
                  left=0.03, right=0.97, top=0.9, bottom=0.08)

    # --- top: the algorithmic flow (single wide axis) ---
    top = fig.add_subplot(gs[0, :])
    top.axis("off")
    top.set_xlim(0, 15.2)
    top.set_ylim(0, 4.0)
    yc, h = 1.5, 1.35

    _box(top, 0.2, yc - h / 2, 2.6, h,
         "start:\n" r"$x_T\sim\mathcal{N}(0,I)$", SLATE_FILL, SLATE, tc=SLATE, fs=13)
    _box(top, 4.6, yc - h / 2, 6.1, h,
         r"predict $\hat\epsilon=\epsilon_\theta(x_t,t)$,  then" "\n"
         r"$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\!\left(x_t-c_t\,\hat\epsilon\right)+\sigma_t z$",
         RUST_FILL, RUST, tc=INK, fs=12.5)
    _box(top, 12.4, yc - h / 2, 2.6, h,
         "result:\n" r"$x_0$ sample", SAGE_TINT, SAGE, tc=SAGE, fs=13)

    _arrow(top, (2.8, yc), (4.6, yc), color=INK)
    _arrow(top, (10.7, yc), (12.4, yc), color=INK)

    # loop-back arrow over the central box
    top.add_patch(FancyArrowPatch(
        (10.4, yc + h / 2), (4.9, yc + h / 2),
        arrowstyle="-|>", mutation_scale=15, color=RUST, lw=1.6,
        connectionstyle="arc3,rad=-0.55", shrinkA=3, shrinkB=3))
    top.text(7.65, 3.55, r"repeat for $t = T,\ T{-}1,\ \dots,\ 1$",
             ha="center", color=RUST, fontsize=12, style="italic")
    top.set_title("Sampling · start from noise, subtract a little predicted noise, repeat",
                  fontsize=14, loc="left", color=INK, pad=6)

    # --- bottom: 2D data emerging from noise (two Gaussian blobs) ---
    rng = np.random.default_rng(7)
    n = 300
    blob = np.vstack([
        rng.normal([-1.1, 0.0], 0.32, (n // 2, 2)),
        rng.normal([1.1, 0.4], 0.32, (n // 2, 2)),
    ])
    ab_panels = [0.03, 0.30, 0.65, 1.0]          # noisy -> clean (reverse order)
    labels = ["t = T\n(noise)", "t = 2T/3", "t = T/3", "t = 0\n(data)"]
    for k, (abv, lab) in enumerate(zip(ab_panels, labels)):
        a = fig.add_subplot(gs[1, k])
        eps = rng.normal(0, 1, blob.shape)
        pts = np.sqrt(abv) * blob + np.sqrt(1 - abv) * eps
        a.scatter(pts[:, 0], pts[:, 1], s=7,
                  color=SAGE if k == 3 else SLATE, alpha=0.6, edgecolors="none")
        a.set_xlim(-3, 3)
        a.set_ylim(-3, 3)
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor("#C9C4B5")
        a.set_title(lab, fontsize=10.5, color=MUTED)
        if k < 3:
            # small connecting arrow between panels (in figure coords via annotate)
            a.annotate("", xy=(1.16, 0.5), xytext=(1.02, 0.5),
                       xycoords="axes fraction",
                       arrowprops=dict(arrowstyle="-|>", color=RUST, lw=1.6))

    save(fig, "reverse_sampling.svg", tight=False)


if __name__ == "__main__":
    print("Generating lec21 figures...")
    training_loop()
    signal_vs_noise()
    reverse_sampling()
    print("Done.")
