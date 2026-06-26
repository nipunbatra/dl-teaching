"""Generate figures for Lecture 0: A Probabilistic View of ML.

Matches the hand-authored lec00 SVG palette (warm parchment + ink, rust/sage/slate)
so new matplotlib figures sit beside the existing ones without clashing.
Outputs SVG into figures/lec00/svg/.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from pathlib import Path

# ---- palette (from anthropic-theme.css) -------------------------------------
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

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec00" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name, tight=True):
    # constrained_layout figures must NOT also be bbox-tight cropped — the two
    # interact and silently clip the right-most axes. Pass tight=False for them.
    fig.savefig(OUT / name, format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {name}")


# ---- 1. linear regression, frequentist view ---------------------------------
def linreg_frequentist():
    rng = np.random.default_rng(0)
    x = np.linspace(0.3, 6, 9)
    y = 1.0 + 0.7 * x + rng.normal(0, 0.45, x.size)
    xs = np.linspace(0, 6.3, 100)
    line = 1.0 + 0.7 * xs

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(xs, line, color=RUST, lw=2.6, label=r"fitted line  $\hat{y}=\theta^\top x$", zorder=2)
    ax.scatter(x, y, s=80, color=SLATE, edgecolor=PAPER, lw=1.2, zorder=3, label="data")

    # one query point: drop a line, mark the single predicted number
    xq = 4.3
    yq = 1.0 + 0.7 * xq
    ax.plot([xq, xq], [0, yq], ls=":", color=MUTED, lw=1.3, zorder=1)
    ax.plot([0, xq], [yq, yq], ls=":", color=MUTED, lw=1.3, zorder=1)
    ax.scatter([xq], [yq], s=120, marker="*", color=RUST, edgecolor=INK, lw=0.8, zorder=4)
    ax.annotate(r"$\hat{y}=\theta^\top x_\star$  — one number",
                xy=(xq, yq), xytext=(xq + 0.15, yq - 1.15),
                color=INK, fontsize=12,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.text(0.15, yq + 0.05, r"$x_\star$", color=MUTED, fontsize=12)

    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xlim(0, 6.4); ax.set_ylim(0, 6.2)
    ax.set_title("Linear regression · the frequentist view — predict a point",
                 fontsize=14, loc="left", color=INK, pad=10)
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    _clean(ax)
    save(fig, "linreg_frequentist.svg")


# ---- 2. random variable as a function Omega -> R -----------------------------
def random_variable_map():
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.axis("off")

    outcomes = ["HH", "HT", "TH", "TT"]
    oy = [3.2, 2.4, 1.6, 0.8]
    # sample space box
    ax.add_patch(Rectangle((0.3, 0.4), 1.7, 3.2, fill=False, ec=SLATE, lw=1.6))
    ax.text(1.15, 3.85, r"sample space  $\Omega$", ha="center", color=SLATE, fontsize=13)
    for o, y in zip(outcomes, oy):
        ax.text(1.15, y, o, ha="center", va="center", color=INK, fontsize=14,
                family="monospace")

    # number line targets
    vals = {0: 0.8, 1: 2.0, 2: 3.2}
    ax.plot([6.3, 6.3], [0.4, 3.6], color=MUTED, lw=1.4)
    for v, y in vals.items():
        ax.plot([6.2, 6.4], [y, y], color=MUTED, lw=1.4)
        ax.text(6.65, y, f"{v}", va="center", color=INK, fontsize=14)
    ax.text(6.3, 3.95, r"$\mathbb{R}$  (number of heads)", ha="center", color=MUTED, fontsize=12)

    # mapping arrows X: outcome -> count of heads
    mapping = {"HH": 2, "HT": 1, "TH": 1, "TT": 0}
    for o, y in zip(outcomes, oy):
        ax.add_patch(FancyArrowPatch((2.1, y), (6.05, vals[mapping[o]]),
                     arrowstyle="-|>", mutation_scale=12, color=SAGE, lw=1.4,
                     connectionstyle="arc3,rad=0.06", alpha=0.9))
    ax.text(4.0, 3.75, r"$X:\ \Omega \to \mathbb{R}$", ha="center", color=SAGE, fontsize=15)
    ax.text(4.0, 0.15, "a random variable is a function from outcomes to numbers",
            ha="center", color=MUTED, fontsize=11, style="italic")

    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.2)
    save(fig, "random_variable_map.svg")


# ---- 3. IID = the ML data layout (X rows, y column) --------------------------
def iid_matrix():
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.axis("off")
    N, d = 5, 4
    cw, ch, x0, y0 = 0.9, 0.62, 0.6, 3.4

    # X matrix
    for i in range(N):
        for j in range(d):
            ax.add_patch(Rectangle((x0 + j * cw, y0 - i * ch), cw, ch,
                         fill=True, fc="#EFE7D4", ec=MUTED, lw=0.8))
    # highlight one row as "one draw"
    ax.add_patch(Rectangle((x0, y0 - 2 * ch), d * cw, ch, fill=False, ec=RUST, lw=2.2))
    ax.text(x0 + d * cw + 0.15, y0 - 2 * ch + ch / 2, "one independent draw",
            va="center", color=RUST, fontsize=11)
    ax.text(x0 + d * cw / 2, y0 + 0.7, r"design matrix  $X$  ($N\times d$)",
            ha="center", color=INK, fontsize=13)
    ax.text(x0 + d * cw / 2, y0 - N * ch - 0.18, "rows = examples · columns = features",
            ha="center", color=MUTED, fontsize=10.5)

    # y column
    yx = x0 + d * cw + 2.6
    for i in range(N):
        ax.add_patch(Rectangle((yx, y0 - i * ch), cw, ch,
                     fill=True, fc="#E6D6CB", ec=MUTED, lw=0.8))
    ax.text(yx + cw / 2, y0 + 0.7, r"targets  $y$", ha="center", color=INK, fontsize=13)

    ax.text(x0 + 0.1, -0.65,
            r"identically distributed: every row drawn from the same $p(\cdot\mid\theta)$"
            "\n"
            r"independent: row $i$ tells you nothing about row $j$",
            color=MUTED, fontsize=11)

    ax.set_xlim(0, 9.4); ax.set_ylim(-1.1, 4.5)
    save(fig, "iid_matrix.svg")


# ---- 4. when IID fails -------------------------------------------------------
def iid_fails():
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), layout="constrained")

    # (a) time series — autocorrelated random walk
    ax = axes[0]
    t = np.arange(40)
    y = np.zeros(40); y[0] = 0
    for k in range(1, 40):
        y[k] = 0.85 * y[k - 1] + rng.normal(0, 0.5)
    ax.plot(t, y, color=SLATE, lw=2)
    ax.scatter([10, 11], [y[10], y[11]], s=55, color=RUST, zorder=3)
    ax.annotate(r"$y_{t}$ depends on $y_{t-1}$", xy=(11, y[11]),
                xytext=(14, y.max() - 0.2), color=RUST, fontsize=11,
                arrowprops=dict(arrowstyle="-|>", color=RUST, lw=1.2))
    ax.set_title("time series", fontsize=12, color=INK)
    ax.set_xlabel("t", fontsize=10)

    # (b) video frames — near-identical consecutive frames
    ax = axes[1]
    ax.axis("off")
    ax.set_title("video frames", fontsize=12, color=INK)
    for k, dx in enumerate([0.0, 0.18, 0.36]):
        base = 0.15 + k * 1.05
        ax.add_patch(Rectangle((base, 0.9), 0.9, 0.9, fill=True,
                     fc="#EFE7D4", ec=MUTED, lw=1))
        circ = plt.Circle((base + 0.4 + dx * 0.4, 1.35), 0.16, color=SAGE)
        ax.add_patch(circ)
        ax.text(base + 0.45, 0.65, f"frame {k+1}", ha="center", color=MUTED, fontsize=9)
    ax.text(1.65, 0.2, "consecutive frames\nalmost identical", ha="center",
            color=RUST, fontsize=10.5)
    ax.set_xlim(0, 3.4); ax.set_ylim(0, 2.1)

    # (c) one-device sensor — shared offset
    ax = axes[2]
    g1 = rng.normal([1.0, 1.0], 0.18, (12, 2))
    g2 = rng.normal([2.4, 2.3], 0.18, (12, 2))
    ax.scatter(g1[:, 0], g1[:, 1], s=40, color=SLATE, label="device A")
    ax.scatter(g2[:, 0], g2[:, 1], s=40, color=RUST, label="device B")
    ax.set_title("sensor logs per device", fontsize=12, color=INK)
    ax.text(1.7, 0.1, "each device adds its own offset", ha="center",
            color=MUTED, fontsize=10, transform=ax.transData)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xticks([]); ax.set_yticks([])
    _clean(ax)

    fig.suptitle("When IID breaks — you need different math",
                 fontsize=13, color=INK)
    _clean(axes[0])
    save(fig, "iid_fails.svg", tight=False)


# ---- 5. MNIST softmax = Categorical -----------------------------------------
def mnist_categorical():
    from sklearn.datasets import load_digits
    digits = load_digits()
    # find an example of a "2"
    idx = np.where(digits.target == 2)[0][1]
    img = digits.images[idx]

    pi = np.array([0.02, 0.03, 0.70, 0.05, 0.02, 0.04, 0.03, 0.06, 0.03, 0.02])

    fig, (axim, axbar) = plt.subplots(1, 2, figsize=(9.8, 4.0),
                                      gridspec_kw={"width_ratios": [1, 2.1]},
                                      layout="constrained")
    axim.imshow(img, cmap="bone_r")
    axim.set_title("input image  (true class = 2)", fontsize=12, color=INK)
    axim.set_xticks([]); axim.set_yticks([])

    cols = [SAGE if k == 2 else "#C9BCA2" for k in range(10)]
    axbar.bar(range(10), pi, color=cols, edgecolor=MUTED, lw=0.6)
    axbar.set_xticks(range(10))
    axbar.set_xlabel("class k", fontsize=11)
    axbar.set_ylabel(r"$\hat{\pi}_k$", fontsize=12)
    axbar.set_title(r"softmax output $\hat{\pi}$ — a Categorical over 10 classes",
                    fontsize=12, color=INK)
    axbar.text(2, 0.72, r"$\hat{\pi}_2 = 0.70$", ha="center", color=SAGE, fontsize=12)
    axbar.set_ylim(0, 0.82)
    _clean(axbar)
    save(fig, "mnist_categorical.svg", tight=False)


# ---- 6. the Normal bell curve -----------------------------------------------
def normal_bellcurve():
    x = np.linspace(-4.2, 4.2, 400)
    pdf = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    wide = np.exp(-x**2 / (2 * 1.8**2)) / (1.8 * np.sqrt(2 * np.pi))

    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    ax.plot(x, pdf, color=RUST, lw=2.6, label=r"$\sigma = 1$")
    ax.plot(x, wide, color=SLATE, lw=2.0, ls="--", label=r"$\sigma = 1.8$ (wider)")

    # shade +/-1 sigma, +/-2 sigma on the main curve
    m1 = (x >= -1) & (x <= 1)
    m2 = (x >= -2) & (x <= 2)
    ax.fill_between(x[m2], pdf[m2], color=SAGE_FILL, alpha=0.35)
    ax.fill_between(x[m1], pdf[m1], color=SAGE_FILL, alpha=0.6)
    ax.axvline(0, color=MUTED, lw=1, ls=":")
    ax.text(0, 0.43, r"$\mu$", ha="center", color=INK, fontsize=13)
    ax.text(0, 0.16, "68%", ha="center", color=INK, fontsize=11)
    ax.text(1.45, 0.05, "95%", ha="center", color=INK, fontsize=11)

    ax.set_xlabel("y"); ax.set_ylabel("density  p(y)")
    ax.set_title("The Normal · centred at μ, spread set by σ", fontsize=14,
                 loc="left", color=INK, pad=10)
    ax.set_ylim(0, 0.46)
    ax.legend(frameon=False, fontsize=11)
    _clean(ax)
    save(fig, "normal_bellcurve.svg")


# ---- 7. the conditional view: model outputs a distribution ------------------
def conditional_view():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")

    # (a) regression: line + a Gaussian bell on y at a query x
    ax = axes[0]
    xs = np.linspace(0, 6, 100); line = 1 + 0.6 * xs
    ax.plot(xs, line, color=RUST, lw=2.4)
    xq = 4.0; mu = 1 + 0.6 * xq
    yy = np.linspace(mu - 2.2, mu + 2.2, 120)
    bell = np.exp(-(yy - mu)**2 / (2 * 0.55**2))
    ax.fill_betweenx(yy, xq, xq + bell * 1.1, color=SAGE_FILL, alpha=0.55)
    ax.plot(xq + bell * 1.1, yy, color=SAGE, lw=1.8)
    ax.plot([xq, xq], [0, mu], ls=":", color=MUTED, lw=1)
    ax.scatter([xq], [mu], color=RUST, s=35, zorder=4)
    ax.set_title(r"regression: $\,Y\mid x \sim \mathcal{N}(\mu_\theta(x),\sigma^2)$",
                 fontsize=11.5, color=INK)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xlim(0, 7); ax.set_ylim(0, 6)
    _clean(ax)

    # (b) logistic: sigmoid + two-bar Bernoulli
    ax = axes[1]
    z = np.linspace(-6, 6, 200); sig = 1 / (1 + np.exp(-z))
    ax.plot(z, sig, color=RUST, lw=2.4)
    zq = 1.2; pq = 1 / (1 + np.exp(-zq))
    ax.plot([zq, zq], [0, pq], ls=":", color=MUTED, lw=1)
    ax.scatter([zq], [pq], color=SAGE, s=60, zorder=3)
    ax.text(zq + 0.3, pq - 0.12, fr"$\hat p={pq:.2f}$", color=SAGE, fontsize=11)
    ax.set_title(r"logistic: $\,Y\mid x \sim \mathrm{Bernoulli}(\hat p_\theta(x))$",
                 fontsize=11.5, color=INK)
    ax.set_xlabel(r"$\theta^\top x$"); ax.set_ylabel(r"$P(Y{=}1\mid x)$")
    _clean(ax)

    # (c) softmax bars
    ax = axes[2]
    pi = [0.1, 0.62, 0.18, 0.1]
    cols = [SAGE if k == 1 else "#C9BCA2" for k in range(4)]
    ax.bar(range(4), pi, color=cols, edgecolor=MUTED, lw=0.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(["c1", "c2", "c3", "c4"])
    ax.set_title(r"softmax: $\,Y\mid x \sim \mathrm{Categorical}(\hat\pi_\theta(x))$",
                 fontsize=11.5, color=INK)
    ax.set_ylabel(r"$\hat\pi_k$")
    ax.set_ylim(0, 0.75)
    _clean(ax)

    fig.suptitle("The model outputs the parameters of a distribution — not a number",
                 fontsize=13, color=INK)
    save(fig, "conditional_view.svg", tight=False)


# ---- 8. logistic regression MLE picture -------------------------------------
def logistic_mle_picture():
    rng = np.random.default_rng(5)
    x0 = rng.normal(-1.6, 1.0, 14)
    x1 = rng.normal(1.6, 1.0, 14)
    xs = np.linspace(-5, 5, 200)
    sig = 1 / (1 + np.exp(-1.4 * xs))

    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    ax.plot(xs, sig, color=RUST, lw=2.6)
    ax.text(3.0, 0.86, r"$\hat p(x)=\sigma(\theta^\top x)$", color=RUST, fontsize=12)
    ax.scatter(x0, np.zeros_like(x0), s=55, color=SLATE, marker="|", lw=2)
    ax.scatter(x1, np.ones_like(x1), s=55, color=SAGE, marker="|", lw=2)
    ax.text(-4.7, 0.11, "y = 0 examples", color=SLATE, fontsize=10)
    ax.text(-4.7, 0.95, "y = 1 examples", color=SAGE, fontsize=10)

    xq = 0.8; pq = 1 / (1 + np.exp(-1.4 * xq))
    ax.plot([xq, xq], [0, pq], ls=":", color=MUTED, lw=1.2)
    ax.plot([-5, xq], [pq, pq], ls=":", color=MUTED, lw=1.2)
    ax.scatter([xq], [pq], color=RUST, s=70, zorder=4, edgecolor=INK, lw=0.6)
    ax.annotate(r"label $\sim \mathrm{Bernoulli}(\hat p(x))$",
                xy=(xq, pq), xytext=(1.4, 0.40), color=INK, fontsize=11,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

    ax.set_xlabel("x (feature)"); ax.set_ylabel(r"$P(Y{=}1\mid x)$")
    ax.set_title("Logistic regression · the probabilistic view", fontsize=14,
                 loc="left", color=INK, pad=10)
    ax.set_ylim(-0.08, 1.12)
    _clean(ax)
    save(fig, "logistic_mle_picture.svg")


# ---- 9. log is monotonic: likelihood vs log-likelihood, same peak -----------
def log_monotonic():
    p = np.linspace(1e-3, 1 - 1e-3, 400)
    L = p**6 * (1 - p)**4
    logL = 6 * np.log(p) + 4 * np.log(1 - p)
    pk = 0.6

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    a1.plot(p, L, color=RUST, lw=2.6)
    a1.axvline(pk, color=MUTED, ls=":", lw=1.3)
    a1.scatter([pk], [pk**6 * (1 - pk)**4], color=RUST, s=55, zorder=5)
    a1.set_title(r"likelihood  $\mathcal{L}(p)=p^6(1-p)^4$", fontsize=12, color=INK)
    a1.set_xlabel("p"); a1.set_ylabel(r"$\mathcal{L}(p)$")

    a2.plot(p, logL, color=SLATE, lw=2.6)
    a2.axvline(pk, color=MUTED, ls=":", lw=1.3)
    a2.scatter([pk], [6*np.log(pk) + 4*np.log(1-pk)], color=SLATE, s=55, zorder=5)
    a2.set_title(r"log-likelihood  $\ell(p)=\log\mathcal{L}(p)$", fontsize=12, color=INK)
    a2.set_xlabel("p"); a2.set_ylabel(r"$\ell(p)$")

    for a in (a1, a2):
        _clean(a)
    fig.suptitle("log is monotonic — the peak stays at the same p (only the y-axis rescales)",
                 fontsize=13, color=INK)
    save(fig, "log_monotonic.svg", tight=False)


if __name__ == "__main__":
    print("Generating lec00 figures...")
    log_monotonic()
    linreg_frequentist()
    random_variable_map()
    iid_matrix()
    iid_fails()
    mnist_categorical()
    normal_bellcurve()
    conditional_view()
    logistic_mle_picture()
    print("Done.")
