"""Every data figure and every number quoted on the rebuilt slides comes from this script.

    PY=/Users/nipun/git/clip-lab/.venv/bin/python
    $PY scripts/make_figures.py            # analytic figures + model figures (if checkpoints exist)
    $PY scripts/make_figures.py analytic   # only figures that need no trained model

Writes figures/data/*.svg|png and figures/data/numbers.json.
Display convention (same as Lecture I): ink on paper, x = +1 is ink, x = -1 is paper.
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_models import (TinyUNet, ConvAE, LatentDenoiser, Classifier, load_mnist, schedule,  # noqa: E402
                          time_features, T)

MODELS = ROOT / "models" / "v2"
FIG = ROOT / "figures" / "data"
FIG.mkdir(parents=True, exist_ok=True)
NUMFILE = FIG / "numbers.json"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
NUM = json.loads(NUMFILE.read_text()) if NUMFILE.exists() else {}

C = dict(x="#245EDB", x_t="#E4ECFF", enc="#8B2CDE", enc_t="#F1E5FC", dec="#0F766E", dec_t="#D9F2EF",
         z="#AA4E08", z_t="#FCEFD9", loss="#BE123C", loss_t="#FDE2E7", xt="#334155", xt_t="#E2E8F0",
         ink="#14171F", ink2="#4A5160", ink3="#6B7280", line="#D9DDE5", grid="#ECEEF2")

plt.rcParams.update({
    "font.family": ["Avenir", "DejaVu Sans"], "font.size": 15, "svg.fonttype": "path",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9AA1AD",
    "axes.labelcolor": C["ink2"], "axes.labelsize": 16, "xtick.color": C["ink3"], "ytick.color": C["ink3"],
    "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.linewidth": 1.0,
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    "legend.frameon": False, "legend.fontsize": 15, "lines.linewidth": 2.6,
    "mathtext.fontset": "cm",
})

from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
STRIPE = LinearSegmentedColormap.from_list("stripe", ["#334155", "#FFFFFF", "#0F766E"])

# the running example: MNIST test digit 0 (a 7), pixel row 12, column 19 (ink, value 255)
DIGIT_I, PIX = 0, (12, 19)
SEED = 667


def save(fig, name):
    path = FIG / f"{name}.svg"
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    svg = path.read_text()
    svg = svg.replace("'Avenir'", "'Avenir Next', 'Avenir', 'Segoe UI', sans-serif")
    svg = svg.replace("font-family:Avenir", "font-family:'Avenir Next', 'Avenir', 'Segoe UI', sans-serif")
    svg = re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.S)
    path.write_text(svg)


def r(v, k=3):
    return float(round(float(v), k))


# ----------------------------------------------------------------------------- image helpers
INK = np.array([0x14, 0x17, 0x1F]) / 255.0
PAPER = np.array([1.0, 1.0, 1.0])


def to_rgb(x):
    """x in [-1, 1] (values outside are clipped for display only): +1 ink, -1 paper."""
    a = (np.clip(np.asarray(x, dtype=float), -1, 1) + 1) / 2
    return PAPER * (1 - a[..., None]) + INK * a[..., None]


def signed_rgb(x, lim=None):
    """Diverging display for noise / differences: amber positive, blue negative, white zero."""
    x = np.asarray(x, dtype=float)
    lim = lim or max(1e-6, np.abs(x).max())
    a = np.clip(x / lim, -1, 1)[..., None]
    pos, neg = np.array([0xAA, 0x4E, 0x08]) / 255, np.array([0x24, 0x5E, 0xDB]) / 255
    return np.where(a > 0, PAPER * (1 - a) + pos * a, PAPER * (1 + a) - neg * a)


def tile(x, name, scale=10, signed=False, lim=None):
    img = signed_rgb(x, lim) if signed else to_rgb(x)
    img = np.repeat(np.repeat(img, scale, 0), scale, 1)
    plt.imsave(FIG / f"{name}.png", np.clip(img, 0, 1))


def grid(imgs, rows, cols, name, scale=4, pad=2, signed=False, lim=None):
    imgs = np.asarray(imgs).reshape(-1, *np.asarray(imgs).shape[-2:])
    h, w = imgs.shape[-2:]
    H, W = rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad
    out = np.ones((H, W, 3)) * np.array([0.90, 0.91, 0.93])
    for i in range(min(len(imgs), rows * cols)):
        rr, cc = divmod(i, cols)
        y0, x0 = pad + rr * (h + pad), pad + cc * (w + pad)
        out[y0:y0 + h, x0:x0 + w] = signed_rgb(imgs[i], lim) if signed else to_rgb(imgs[i])
    out = np.repeat(np.repeat(out, scale, 0), scale, 1)
    plt.imsave(FIG / f"{name}.png", np.clip(out, 0, 1))


def sched_np():
    beta, alpha, ab, var = (v.numpy().astype(np.float64) for v in schedule("cpu"))
    return beta, alpha, ab, var


# ============================================================================= analytic figures
def toy_numbers():
    b, a = 0.2, 0.8
    ab1, ab2 = 0.8, 0.64
    x0, eta1, eps, eh = 1.0, -0.5, -0.5, -0.4
    x1 = math.sqrt(a) * x0 + math.sqrt(b) * eta1
    x2 = math.sqrt(ab2) * x0 + math.sqrt(1 - ab2) * eps
    eta2 = (x2 - math.sqrt(a) * x1) / math.sqrt(b)
    eps_agg = (math.sqrt(a * b) * eta1 + math.sqrt(b) * eta2) / math.sqrt(1 - ab2)
    x0h = (x2 - math.sqrt(1 - ab2) * eh) / math.sqrt(ab2)
    c = b / math.sqrt(1 - ab2)
    mu = (x2 - c * eh) / math.sqrt(a)
    mu_t = (x2 - c * eps) / math.sqrt(a)
    mu_t2 = math.sqrt(ab1) * b / (1 - ab2) * x0 + math.sqrt(a) * (1 - ab1) / (1 - ab2) * x2
    bt = b * (1 - ab1) / (1 - ab2)
    z = 0.3
    x1s = mu + math.sqrt(bt) * z
    w2 = b ** 2 / (2 * bt * a * (1 - ab2))
    kl = (mu_t - mu) ** 2 / (2 * bt)
    # two explanations of x2 = 0.5: ink (x0 = +1) or paper (x0 = -1), equal prior
    lik = [math.exp(-(x2 - math.sqrt(ab2) * s) ** 2 / (2 * (1 - ab2))) for s in (1, -1)]
    p_ink = lik[0] / sum(lik)
    eps_paper = (x2 + math.sqrt(ab2)) / math.sqrt(1 - ab2)
    E_eps = p_ink * eps + (1 - p_ink) * eps_paper
    E_x0 = p_ink * 1 + (1 - p_ink) * (-1)
    assert abs(mu_t - mu_t2) < 1e-9 and abs(eps_agg - eps) < 1e-9 and abs(kl - w2 * (eh - eps) ** 2) < 1e-12
    NUM["toy"] = dict(
        beta=b, alpha=a, ab1=ab1, ab2=ab2, sa=r(math.sqrt(a)), sb=r(math.sqrt(b)), x0=x0, eta1=eta1, x1=r(x1),
        eta2=r(eta2), eps=eps, x2=r(x2, 2), ehat=eh, loss=r((eh - eps) ** 2, 2), x0hat=r(x0h),
        x0hat_err=r(1 - x0h), amp=r(math.sqrt(1 - ab2) / math.sqrt(ab2), 2), c=r(c), c_frac="1/3",
        num=r(x2 - c * eh), mu=r(mu), mu_t=r(mu_t), gap=r(mu_t - mu), gap4=r(mu_t - mu, 4), bt=r(bt), sigma=r(math.sqrt(bt)), z=z,
        sz=r(math.sqrt(bt) * z), x1s=r(x1s), w2=r(w2), kl=r(kl, 5), var_sum=r(a * b + b, 2), ab_var=r(a * b, 2),
        sab=r(math.sqrt(a * b), 1), p_ink=f"{p_ink:.2f}", p_paper=f"{1 - p_ink:.2f}", eps_paper=f"{eps_paper:.2f}",
        E_eps=f"{E_eps:.2f}", E_x0=f"{E_x0:.2f}", lik_ratio=r(lik[0] / lik[1], 1))


def schedule_numbers():
    beta, alpha, ab, var = sched_np()
    ts = [1, 2, 7, 50, 100, 250, 300, 350, 500, 600, 700, 800, 920, 1000]
    NUM["sched"] = {str(t): dict(beta=r(beta[t], 5), ab=r(ab[t], 5), abd=f"{ab[t]:.3f}" if ab[t] >= 0.01 else f"{ab[t]:.5f}",
                                 sab=r(math.sqrt(ab[t]), 3),
                                 s1ab=r(math.sqrt(1 - ab[t]), 3), snr=r(ab[t] / (1 - ab[t]), 4),
                                 amp=r(math.sqrt(1 - ab[t]) / math.sqrt(ab[t]), 1)) for t in ts}
    NUM["sched"]["t_signal_below_0p1"] = int(np.argmax(np.sqrt(ab[1:]) < 0.1) + 1)
    NUM["sched"]["t_half"] = int(np.argmax(ab[1:] < 0.5) + 1)
    NUM["sched"]["abT"] = f"{ab[T]:.1e}".replace("e-05", " × 10⁻⁵")
    NUM["sched"]["beta_max_ratio"] = r(0.2 / 0.02, 0)
    w = np.zeros(T + 1)
    w[2:] = beta[2:] ** 2 / (2 * var[2:] * alpha[2:] * (1 - ab[2:]))
    NUM["weights"] = dict(w2=r(w[2], 2), w10=r(w[10], 3), w100=r(w[100], 4), w500=r(w[500], 4), w1000=r(w[1000], 4),
                          wmin=r(w[2:].min(), 4), tmin=int(np.argmin(w[2:]) + 2), ratio=int(round(w[2] / w[500])))
    return beta, alpha, ab, var, w


def fig_schedule(ab):
    t = np.arange(0, T + 1)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.fill_between(t, 0, ab, color=C["x"], alpha=.18, lw=0)
    ax.fill_between(t, ab, 1, color=C["z"], alpha=.16, lw=0)
    ax.plot(t, ab, color=C["x"], lw=2.8)
    ax.text(30, 0.22, r"signal variance $\bar{\alpha}_t$", color=C["x"], fontsize=16, fontweight=600)
    ax.text(700, 0.40, r"noise variance $1-\bar{\alpha}_t$", color=C["z"], fontsize=16, fontweight=600)
    for tt in (100, 300, 500, 700, 1000):
        ax.plot([tt], [ab[tt]], "o", color=C["x"], ms=7, mec="white", mew=1.5, zorder=5)
    t01 = NUM["sched"]["t_signal_below_0p1"]
    ax.axvline(t01, color=C["ink3"], lw=1.2, ls=(0, (4, 3)))
    ax.text(t01 - 14, 0.95, f"t = {t01}: signal amplitude\n" + r"$\sqrt{\bar{\alpha}_t}$ falls below 0.1", fontsize=14, color=C["ink2"], va="top", ha="right")
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1)
    ax.set_xlabel("step t")
    ax.set_yticks([0, .5, 1])
    save(fig, "schedule")


def fig_two_noise_hist():
    rng = np.random.default_rng(SEED)
    e1, e2 = rng.standard_normal(200000), rng.standard_normal(200000)
    s = 0.4 * e1 + math.sqrt(0.2) * e2
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.hist(s, bins=90, range=(-2.5, 2.5), density=True, color=C["z"], alpha=.35, lw=0)
    xs = np.linspace(-2.5, 2.5, 300)
    ax.plot(xs, np.exp(-xs ** 2 / (2 * .36)) / math.sqrt(2 * math.pi * .36), color=C["z"], lw=2.8)
    ax.text(0.0, 0.95, "200,000 draws of\n" + r"$0.4\,\eta_1 + 0.447\,\eta_2$", transform=ax.transAxes, ha="left", va="top",
            fontsize=14, color=C["ink2"])
    ax.text(1.0, 0.95, "curve:\n" + r"$\mathcal{N}(0,\ 0.36)$", transform=ax.transAxes, ha="right", va="top", fontsize=14, color=C["z"], fontweight=600)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    NUM["hist_var"] = r(s.var(), 3)
    save(fig, "two-noise-hist")


def fig_two_explanations():
    """Two clean values could produce the same noisy value x2 = 0.5 (toy schedule, abar = 0.64)."""
    toy = NUM["toy"]
    xs = np.linspace(-3, 3, 600)
    g = lambda m, v: np.exp(-(xs - m) ** 2 / (2 * v)) / math.sqrt(2 * math.pi * v)
    for name, abar, label in (("explain-t2", 0.64, "toy step 2:  ᾱ = 0.64"),
                              ("explain-t500", NUM["sched"]["500"]["ab"], f"step 500:  ᾱ = {NUM['sched']['500']['ab']:.3f}")):
        fig, ax = plt.subplots(figsize=(7.4, 3.3))
        m = math.sqrt(abar)
        ax.fill_between(xs, g(m, 1 - abar), color=C["x"], alpha=.14, lw=0)
        ax.fill_between(xs, g(-m, 1 - abar), color=C["enc"], alpha=.12, lw=0)
        ax.plot(xs, g(m, 1 - abar), color=C["x"], lw=2.6)
        ax.plot(xs, g(-m, 1 - abar), color=C["enc"], lw=2.6)
        ax.axvline(0.5, color=C["xt"], lw=2, ls=(0, (5, 3)))
        ax.text(0.55, g(m, 1 - abar).max() * 1.08, "observed 0.5", ha="left", va="bottom", fontsize=15,
                color=C["xt"], fontweight=600)
        top = g(m, 1 - abar).max()
        lk = [math.exp(-(0.5 - s * m) ** 2 / (2 * (1 - abar))) for s in (1, -1)]
        pi = lk[0] / sum(lk)
        ax.text(2.95, top * .95, "pixel was ink\n" + r"$x_0=+1$" + f"\nP = {pi:.2f}", color=C["x"], fontsize=14, fontweight=600, ha="right", va="top")
        ax.text(-2.95, top * .95, "pixel was paper\n" + r"$x_0=-1$" + f"\nP = {1 - pi:.2f}", color=C["enc"], fontsize=14, fontweight=600, ha="left", va="top")
        ax.text(0.5, -top * .02, "", fontsize=1)
        ax.set_title(label, fontsize=14, color=C["ink3"], loc="left", pad=22)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(-3, 3)
        ax.set_xlabel("noisy value of our pixel")
        lk = [math.exp(-(0.5 - s * m) ** 2 / (2 * (1 - abar))) for s in (1, -1)]
        NUM.setdefault("explain", {})[name] = dict(p_ink=r(lk[0] / sum(lk), 2), p_paper=r(lk[1] / sum(lk), 2))
        save(fig, name)


def fig_clusters(ab):
    rng = np.random.default_rng(SEED)
    ang = np.arange(8) * 2 * math.pi / 8
    mu = 2 * np.stack([np.cos(ang), np.sin(ang)], 1)
    k = rng.integers(0, 8, 800)
    x0 = mu[k] + 0.18 * rng.standard_normal((800, 2))
    eps = rng.standard_normal((800, 2))
    ts = (0, 100, 250, 1000)
    fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.6))
    for ax, t in zip(axs, ts):
        a = 1.0 if t == 0 else ab[t]
        xt = math.sqrt(a) * x0 + math.sqrt(1 - a) * eps
        ax.scatter(xt[:, 0], xt[:, 1], s=6, color=C["x"] if t == 0 else C["xt"], alpha=.55, lw=0)
        ax.set_xlim(-3.4, 3.4)
        ax.set_ylim(-3.4, 3.4)
        ax.set_aspect("equal")
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.set_title(f"t = {t}    " + r"$\bar{\alpha}_t$" + f" = {a:.3f}" if t and a > 1e-3 else (f"t = {t}    " + r"$\bar{\alpha}_t$" + " = 0.00004" if t else "t = 0   (data)"), fontsize=16, color=C["ink2"], pad=8)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#C7CCD4")
    save(fig, "clusters")
    NUM["clusters"] = dict(radius=2, tau=0.18, n=800)
    return mu


def fig_weights(w):
    t = np.arange(2, T + 1)
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.semilogy(t, w[2:], color=C["loss"], lw=2.8)
    ax.axhline(1, color=C["dec"], lw=2.4, ls=(0, (6, 3)))
    ax.text(990, 1.25, "DDPM's simple loss: weight 1 for every t", ha="right", color=C["dec"], fontsize=15, fontweight=600)
    ax.text(40, w[2] * 0.6, f"t = 2:  w = {w[2]:.2f}", color=C["loss"], fontsize=14, fontweight=600)
    ax.text(500, w[500] * 2.2, f"t = 500:  w = {w[500]:.4f}", color=C["loss"], fontsize=14, ha="center")
    ax.text(700, 0.022, r"the bound's weight $w_t$", color=C["loss"], fontsize=15, ha="center")
    ax.set_xlim(0, T)
    ax.set_ylim(3e-3, 3)
    ax.set_xlabel("step t")
    ax.set_ylabel(r"weight on $\|\epsilon-\hat\epsilon\|^2$")
    save(fig, "weights")


def fig_amplification(ab):
    t = np.arange(1, T + 1)
    amp = np.sqrt(1 - ab[1:]) / np.sqrt(ab[1:])
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.semilogy(t, amp, color=C["x"], lw=2.8)
    for tt in (100, 350, 920):
        ax.plot([tt], [amp[tt - 1]], "o", color=C["x"], ms=7, mec="white", mew=1.5)
        ax.text(tt + 30 if tt < 500 else tt - 25, amp[tt - 1] * (0.45 if tt < 500 else 1.9), f"t = {tt}:  ×{amp[tt - 1]:.1f}" if amp[tt - 1] < 10 else f"t = {tt}:  ×{amp[tt - 1]:.0f}",
                ha="left" if tt < 500 else "right", fontsize=14, color=C["ink2"])
    ax.set_xlim(0, T)
    ax.set_xlabel("step t")
    ax.set_ylabel(r"$\sqrt{1-\bar\alpha_t}\,/\sqrt{\bar\alpha_t}$")
    save(fig, "amplification")


def fig_kl_gaussians():
    toy = NUM["toy"]
    xs = np.linspace(-0.4, 1.9, 500)
    v = toy["bt"]
    g = lambda m: np.exp(-(xs - m) ** 2 / (2 * v)) / math.sqrt(2 * math.pi * v)
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.fill_between(xs, g(toy["mu_t"]), color=C["x"], alpha=.12, lw=0)
    ax.plot(xs, g(toy["mu_t"]), color=C["x"], lw=2.8)
    ax.plot(xs, g(toy["mu"]), color=C["dec"], lw=2.8, ls=(0, (6, 3)))
    top = g(toy["mu"]).max()
    ax.plot([toy["mu"]] * 2, [0, top * 1.12], color=C["dec"], lw=1.6)
    ax.plot([toy["mu_t"]] * 2, [0, top * 1.12], color=C["x"], lw=1.6)
    ax.annotate(f"means differ by {toy['gap']:.3f}", xy=((toy["mu"] + toy["mu_t"]) / 2, top * 1.1), xytext=(1.05, top * 1.18),
                color=C["loss"], fontsize=15, fontweight=600, va="center",
                arrowprops=dict(arrowstyle="->", color=C["loss"], lw=1.6))
    ax.text(1.9, top * 0.55, "target  " + r"$q(x_1\mid x_2, x_0)$" + f"\nmean {toy['mu_t']:.3f}", color=C["x"], fontsize=14, fontweight=600, ha="right")
    ax.text(-0.38, top * 0.55, "model  " + r"$p_\theta(x_1\mid x_2)$" + f"\nmean {toy['mu']:.3f}", color=C["dec"], fontsize=14, fontweight=600)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(0, top * 1.3)
    ax.set_xlabel(r"value of $x_1$")
    save(fig, "kl-gaussians")


def fig_reverse_line():
    toy = NUM["toy"]
    xs = np.linspace(-0.3, 1.8, 500)
    v = toy["bt"]
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    gy = np.exp(-(xs - toy["mu"]) ** 2 / (2 * v)) / math.sqrt(2 * math.pi * v)
    ax.fill_between(xs, gy, color=C["dec"], alpha=.12, lw=0)
    ax.plot(xs, gy, color=C["dec"], lw=2.4)
    top = gy.max()
    marks = [(toy["x2"], "x₂ = 0.5\nnoisy input", C["xt"], 1.30),
             (toy["mu"], f"μ_θ = {toy['mu']:.3f}\nmean", C["dec"], 1.62),
             (toy["x1s"], f"x₁ = {toy['x1s']:.3f}\nsampled", C["z"], 1.30),
             (toy["x0hat"], f"x̂₀ = {toy['x0hat']:.3f}\nguess of clean", C["x"], 1.62)]
    for x, lab, col, h in marks:
        ax.plot([x, x], [0, top * (h - .12)], color=col, lw=2.2)
        ax.plot([x], [0], "o", color=col, ms=8, mec="white", mew=1.5, zorder=5)
        ax.text(x, top * h, lab, ha="center", va="bottom", fontsize=14, color=col, fontweight=600)
    ax.annotate("", xy=(toy["x1s"], top * .22), xytext=(toy["mu"], top * .22),
                arrowprops=dict(arrowstyle="->", color=C["z"], lw=2))
    ax.text((toy["mu"] + toy["x1s"]) / 2, top * .28, "σz = 0.1", ha="center", fontsize=13, color=C["z"])
    ax.set_ylim(0, top * 2.15)
    ax.set_xlim(-0.3, 1.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    save(fig, "reverse-line")


def fig_time_stripes():
    t = torch.tensor([7, 350, 920])
    f = time_features(t).numpy()
    fig, axs = plt.subplots(3, 1, figsize=(7.4, 1.9))
    for ax, row, tt in zip(axs, f, (7, 350, 920)):
        ax.imshow(row[None, :], aspect="auto", cmap=STRIPE, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(f"t = {tt}", rotation=0, ha="right", va="center", fontsize=22, color=C["ink2"])
        for s in ax.spines.values():
            s.set_visible(False)
    save(fig, "time-stripes")


def fig_schedules(ab):
    t = np.arange(0, T + 1)
    f = lambda tt: np.cos(((tt / T) + 0.008) / 1.008 * math.pi / 2) ** 2
    abc = f(t) / f(0)
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.plot(t, ab, color=C["x"], lw=2.8, label="linear β (DDPM, our model)")
    ax.plot(t, abc, color=C["dec"], lw=2.8, ls=(0, (6, 3)), label="cosine (Nichol & Dhariwal)")
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("step t")
    ax.set_ylabel(r"$\bar{\alpha}_t$")
    ax.legend(loc="upper right")
    save(fig, "schedules")


def mixture_eps(x, a, mu, tau=0.18):
    """Exact E[eps | x_t = x] for the 8-cluster mixture at signal level a = abar_t."""
    m = math.sqrt(a) * mu                      # [8, 2]
    v = a * tau ** 2 + 1 - a
    d2 = ((x[:, None, :] - m[None]) ** 2).sum(-1)
    lw = -d2 / (2 * v)
    w = np.exp(lw - lw.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    score = (w[..., None] * (m[None] - x[:, None, :])).sum(1) / v
    return -math.sqrt(1 - a) * score, score


def fig_score(ab, var, beta, alpha):
    ang = np.arange(8) * 2 * math.pi / 8
    mu = 2 * np.stack([np.cos(ang), np.sin(ang)], 1)
    tshow = 150
    g1 = np.linspace(-3, 3, 17)
    X, Y = np.meshgrid(g1, g1)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    _, sc = mixture_eps(pts, ab[tshow], mu)
    n = np.linalg.norm(sc, axis=1, keepdims=True) + 1e-9
    dirs = sc / n * np.minimum(n, 3) / 3
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((400, 2))
    paths = [x[:8].copy()]
    ts = np.linspace(T, 1, 200).round().astype(int)
    for i, t in enumerate(ts):          # deterministic DDIM steps with the exact noise prediction
        s_ = int(ts[i + 1]) if i + 1 < len(ts) else 0
        e, _ = mixture_eps(x, ab[t], mu)
        x0 = (x - math.sqrt(1 - ab[t]) * e) / math.sqrt(ab[t])
        x = math.sqrt(ab[s_]) * x0 + math.sqrt(1 - ab[s_]) * e
        paths.append(x[:8].copy())
    P = np.stack(paths)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.quiver(pts[:, 0], pts[:, 1], dirs[:, 0], dirs[:, 1], color="#9AA1AD", scale=22, width=0.004)
    for k in range(8):
        ax.plot(P[:, k, 0], P[:, k, 1], color=C["xt"], lw=1.8, alpha=.85)
        ax.plot(P[0, k, 0], P[0, k, 1], "o", color=C["xt"], ms=5)
    ax.scatter(x[:, 0], x[:, 1], s=9, color=C["x"], zorder=5)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    save(fig, "score-field")
    NUM["score"] = dict(t=tshow)


def fig_flow():
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    noise = np.array([[-2.2, 1.0], [-2.2, -1.0]])
    data = np.array([[2.2, -1.2], [2.2, 1.2]])
    for (a, b) in zip(noise, data):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=C["dec"], lw=2.2))
    ax.scatter(noise[:, 0], noise[:, 1], s=90, color=C["z"], zorder=5)
    ax.scatter(data[:, 0], data[:, 1], s=90, color=C["x"], zorder=5)
    ax.text(-2.3, 1.35, "noise", color=C["z"], fontsize=15, ha="center")
    ax.text(2.3, 1.55, "data", color=C["x"], fontsize=15, ha="center")
    ax.annotate("", xy=(0.9, 0.0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C["loss"], lw=3))
    ax.text(0.45, -1.7, "learned direction at the crossing:\nthe average of the two", color=C["loss"], fontsize=14, ha="center")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.9, 1.9)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "flow")


def digit_figures(x_test, beta, ab):
    d = x_test[DIGIT_I, 0].numpy()
    two = x_test[1, 0].numpy()
    tile(d, "digit7")
    tile(two, "digit2")
    r0, c0 = PIX
    raw = np.round((d + 1) * 127.5).astype(int)
    NUM["pixel"] = dict(row=r0, col=c0, raw=int(raw[r0, c0]), x0=r(d[r0, c0], 2), in2=r(two[r0, c0], 2))
    # zoom of the 7 x 7 neighbourhood around the running pixel with raw uint8 values
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    sub = raw[r0 - 3:r0 + 4, c0 - 3:c0 + 4]
    ax.imshow(to_rgb(sub / 127.5 - 1), interpolation="nearest")
    for i in range(7):
        for j in range(7):
            v = sub[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=14 if v < 100 else 13,
                    color="white" if v > 130 else C["ink2"], family="Menlo")
    ax.add_patch(plt.Rectangle((2.5, 2.5), 1, 1, fill=False, ec=C["z"], lw=3.5))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "pixel-zoom")
    # the full digit with the zoom window and the pixel marked
    img = np.repeat(np.repeat(to_rgb(d), 10, 0), 10, 1)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    ax.add_patch(plt.Rectangle(((c0 - 3) * 10 - .5, (r0 - 3) * 10 - .5), 70, 70, fill=False, ec=C["z"], lw=2, ls="--"))
    ax.add_patch(plt.Rectangle((c0 * 10 - .5, r0 * 10 - .5), 10, 10, fill=False, ec=C["z"], lw=3))
    ax.axis("off")
    save(fig, "digit7-marked")

    # one toy step with a fresh noise image (beta = 0.2 so it is visible)
    g = torch.Generator().manual_seed(SEED)
    eta = torch.randn(28, 28, generator=g).numpy()
    tile(eta, "eta1", signed=True, lim=2.5)
    tile(math.sqrt(0.8) * d, "digit7-scaled")
    tile(math.sqrt(0.8) * d + math.sqrt(0.2) * eta, "x1-toy")

    # a recursive forward chain on the real schedule, one fresh draw per step
    g = torch.Generator().manual_seed(SEED)
    x = torch.from_numpy(d).double()
    keep = {0: d.copy()}
    for t in range(1, T + 1):
        x = math.sqrt(1 - beta[t]) * x + math.sqrt(beta[t]) * torch.randn(28, 28, generator=g).double()
        if t in (100, 300, 500, 700, 1000):
            keep[t] = x.numpy().copy()
    for t, v in keep.items():
        tile(v, f"fwd-{t}")

    # the direct jump with one fixed epsilon image (also used by the explorer widget)
    g = torch.Generator().manual_seed(SEED + 1)
    eps = torch.randn(28, 28, generator=g).numpy()
    tile(eps, "eps", signed=True, lim=2.5)
    for t in (7, 100, 350, 600, 920):
        tile(math.sqrt(ab[t]) * d + math.sqrt(1 - ab[t]) * eps, f"jump-{t}")
    NUM["eps_p"] = r(eps[PIX], 2)
    (FIG / "explorer.json").write_text(json.dumps(dict(
        x0=[round(float(v), 3) for v in d.ravel()], eps=[round(float(v), 3) for v in eps.ravel()],
        ab=[round(float(v), 7) for v in ab], pix=PIX)))
    return d, eps


def analytic():
    beta, alpha, ab, var, w = schedule_numbers()
    toy_numbers()
    fig_schedule(ab)
    fig_two_noise_hist()
    fig_two_explanations()
    fig_clusters(ab)
    fig_weights(w)
    fig_amplification(ab)
    fig_kl_gaussians()
    fig_time_stripes()
    fig_schedules(ab)
    fig_score(ab, var, beta, alpha)
    fig_flow()
    x_test = load_mnist(False)
    digit_figures(x_test, beta, ab)


# ============================================================================= model figures
def load_unet(name, n_classes=0):
    ck = torch.load(MODELS / f"{name}.pt", map_location="cpu")
    m = TinyUNet(n_classes)
    m.load_state_dict(ck["model"])
    return m.to(DEV).eval(), ck


@torch.no_grad()
def ddpm_sample(model, n, seed, c=None, s=1.0, mean_only=False, record=(), x_init=None, t_start=T,
                known=None, mask=None):
    """DDPM ancestral sampler with posterior variance. Optional: classifier-free guidance (c, s),
    mean-only (no fresh noise), starting from a given x at t_start (SDEdit), RePaint-style known pixels."""
    beta, alpha, ab, var = (v.to(DEV) for v in schedule(DEV))
    g = torch.Generator().manual_seed(seed)
    rnd = lambda: torch.randn(n, 1, 28, 28, generator=g).to(DEV)
    x = rnd() if x_init is None else x_init.to(DEV)
    frames, ests = {}, {}
    for step in range(t_start, 0, -1):
        t = torch.full((n,), step, device=DEV, dtype=torch.long)
        if c is None:
            e = model(x, t)
        else:
            cu = torch.full((n,), 10, device=DEV, dtype=torch.long)
            cc = torch.full((n,), c, device=DEV, dtype=torch.long)
            eu, ec = model(x, t, cu), model(x, t, cc)
            e = eu + s * (ec - eu)
        if step in record:
            frames[step] = x.cpu().numpy()
            ests[step] = ((x - (1 - ab[step]).sqrt() * e) / ab[step].sqrt()).cpu().numpy()
        mean = (x - beta[step] * e / (1 - ab[step]).sqrt()) / alpha[step].sqrt()
        noise = rnd()
        x = mean + (var[step].sqrt() * noise if (step > 1 and not mean_only) else 0)
        if known is not None and step > 1:     # RePaint without resampling: re-noise the known pixels to t-1
            kn = ab[step - 1].sqrt() * known + (1 - ab[step - 1]).sqrt() * rnd()
            x = mask * kn + (1 - mask) * x
    if known is not None:
        x = mask * known + (1 - mask) * x
    return x.cpu().numpy(), frames, ests


@torch.no_grad()
def model_figures():
    beta, alpha, ab, var = sched_np()
    x_test = load_mnist(False)
    d = x_test[DIGIT_I:DIGIT_I + 1].to(DEV)
    U, ck = load_unet("uncond")
    lp = MODELS / "uncond-log.json"
    log = json.loads(lp.read_text()) if lp.exists() else dict(   # preview while training is still running
        steps=ck["steps"], batch=ck["batch"], params=sum(p.numel() for p in U.parameters()), seconds=0,
        history=[dict(step=ck["steps"], smooth=float("nan"))])
    NUM["uncond"] = dict(steps=log["steps"], batch=log["batch"], params=f"{log['params']:,}",
                         minutes=max(1, round(log["seconds"] / 60)), final=r(log["history"][-1]["smooth"], 4),
                         epochs=round(log["steps"] * log["batch"] / 60000))
    # ---- loss curve
    h = log["history"]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.plot([e["step"] for e in h], [e["smooth"] for e in h], color=C["loss"], lw=2.4)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("noise MSE")
    save(fig, "loss-curve")

    # ---- one training example at t = 350 with a fixed epsilon (the same eps image as the slides)
    g = torch.Generator().manual_seed(SEED + 1)
    eps = torch.randn(1, 1, 28, 28, generator=g).to(DEV)
    t = 350
    a = float(ab[t])
    xt = math.sqrt(a) * d + math.sqrt(1 - a) * eps
    eh = U(xt, torch.tensor([t], device=DEV))
    x0h = (xt - math.sqrt(1 - a) * eh) / math.sqrt(a)
    v = math.sqrt(a) * eps - math.sqrt(1 - a) * d
    tile(xt[0, 0].cpu(), "ex-xt")
    tile(eh[0, 0].cpu(), "ex-ehat", signed=True, lim=2.5)
    tile(((eh - eps) ** 2)[0, 0].cpu(), "ex-err", signed=True, lim=1.0)
    tile(x0h[0, 0].cpu(), "ex-x0hat")
    tile(v[0, 0].cpu(), "ex-v", signed=True, lim=2.5)
    r0, c0 = PIX
    pts = [(r0, c0), (8, 10), (20, 15), (3, 3)]
    four = [dict(where=f"({p[0]},{p[1]})", eps=f"{float(eps[0, 0, p[0], p[1]]):.2f}".replace("-", "−"),
                 ehat=f"{float(eh[0, 0, p[0], p[1]]):.2f}".replace("-", "−"),
                 sq=f"{float((eh[0, 0, p[0], p[1]] - eps[0, 0, p[0], p[1]]) ** 2):.3f}") for p in pts]
    NUM["example"] = dict(t=t, ab=r(a, 3), sab=r(math.sqrt(a), 3), s1ab=r(math.sqrt(1 - a), 3),
                          mse=f"{float(((eh - eps) ** 2).mean()):.3f}", four=four,
                          four_mean=f"{np.mean([float(f['sq']) for f in four]):.3f}",
                          x0hat_mse=r(((x0h - d) ** 2).mean(), 3))
    # ---- what the model believes the clean image is, at several noise levels (same eps)
    for tt in (920, 700, 500, 350, 100):
        a_ = float(ab[tt])
        xt_ = math.sqrt(a_) * d + math.sqrt(1 - a_) * eps
        e_ = U(xt_, torch.tensor([tt], device=DEV))
        tile(((xt_ - math.sqrt(1 - a_) * e_) / math.sqrt(a_))[0, 0].cpu(), f"guess-{tt}")
        tile(xt_[0, 0].cpu(), f"guess-in-{tt}")
    # ---- batch with three noise levels
    for tt in (7, 350, 920):
        tile((math.sqrt(ab[tt]) * d + math.sqrt(1 - ab[tt]) * eps)[0, 0].cpu(), f"batch-{tt}")
    # ---- error vs noise level on 512 test digits
    xs = x_test[:512].to(DEV)
    gg = torch.Generator().manual_seed(1)
    mse_t = []
    tgrid = [1, 5, 10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for tt in tgrid:
        e = torch.randn(xs.shape, generator=gg).to(DEV)
        a_ = float(ab[tt])
        p = U(math.sqrt(a_) * xs + math.sqrt(1 - a_) * e, torch.full((len(xs),), tt, device=DEV))
        mse_t.append(float(((p - e) ** 2).mean()))
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.semilogy(tgrid, mse_t, "o-", color=C["dec"], ms=6)
    ax.set_xlabel("step t")
    ax.set_ylabel("noise MSE on test digits")
    save(fig, "mse-vs-t")
    NUM["mse_t"] = {str(k): np.format_float_positional(v, precision=2, fractional=False, trim="-") for k, v in zip(tgrid, mse_t)}

    # ---- samples: 32 unselected, seed 667, and their trajectories
    rec = (1000, 800, 600, 400, 200, 100, 50, 1)
    xs_, fr, es = ddpm_sample(U, 32, SEED, record=rec)
    grid(xs_[:, 0], 4, 8, "samples-32", scale=4)
    np.save(MODELS / "samples-32.npy", xs_)
    k = 0
    for tt in rec:
        tile(fr[tt][k, 0], f"traj-x-{tt}")
        tile(es[tt][k, 0], f"traj-e-{tt}")
    tile(xs_[k, 0], "traj-x-0")
    NUM["traj_index"] = k
    # scrubber sprites: 6 seeds, state and clean guess every 20 steps (row = seed, column = frame)
    ts_s = list(range(1000, 0, -20)) + [1]
    xsS, frS, esS = ddpm_sample(U, 6, 2024, record=tuple(ts_s))
    def sprite(frames, name):
        cols = [frames[t] for t in ts_s] + ([xsS] if name == "scrub-x" else [frames[1]])
        img = np.concatenate([np.concatenate([c[k, 0] for c in cols], 1) for k in range(6)], 0)
        plt.imsave(FIG / f"{name}.png", np.clip(to_rgb(img), 0, 1))
    sprite(frS, "scrub-x")
    sprite(esS, "scrub-e")
    (FIG / "scrub.json").write_text(json.dumps(dict(ts=ts_s + [0], seeds=6)))
    # nearest training neighbours for the first 6 samples
    xtr = load_mnist(True).reshape(60000, -1)
    sm = torch.from_numpy(xs_[:6].reshape(6, -1)).float()
    dist = torch.cdist(sm, xtr)
    nn_i = dist.argmin(1)
    grid(np.concatenate([xs_[:6, 0], xtr[nn_i].reshape(6, 28, 28).numpy()]), 2, 6, "nearest", scale=4)
    NUM["nearest"] = dict(dist=[r(v, 1) for v in dist.min(1).values], idx=[int(i) for i in nn_i])
    # mean-only sampling (no fresh noise) vs normal sampling, same seeds
    a8, _, _ = ddpm_sample(U, 8, 11)
    b8, _, _ = ddpm_sample(U, 8, 11, mean_only=True)
    grid(a8[:, 0], 1, 8, "with-noise", scale=4)
    grid(b8[:, 0], 1, 8, "mean-only", scale=4)
    pair = lambda z: float(torch.pdist(torch.from_numpy(z.reshape(8, -1))).mean())
    NUM["noise_ablation"] = dict(div_noise=r(pair(a8), 1), div_mean=r(pair(b8), 1))

    # ---- U-Net feature maps for the t = 350 example
    keep = {}
    U(xt, torch.tensor([t], device=DEV), keep=keep)
    for k_ in ("e1", "e2", "m", "d1", "d2"):
        f = keep[k_][0].cpu().numpy()
        ref = torch.nn.functional.adaptive_avg_pool2d(d, f.shape[-1])[0, 0].cpu().numpy().ravel()
        cor = [abs(np.corrcoef(f[c].ravel(), ref)[0, 1]) if f[c].std() > 1e-6 else 0 for c in range(len(f))]
        ch = np.argsort(cor)[::-1][:2]
        for j, cidx in enumerate(ch):
            fm = f[cidx]
            fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-6) * 2 - 1
            tile(fm, f"fmap-{k_}-{j}", scale=max(2, 280 // fm.shape[0] // 2))
    NUM["unet_params"] = f"{sum(p.numel() for p in U.parameters()):,}"

    # ---- inpainting with the unconditional model: keep the top half of our 7
    mask = torch.zeros(1, 1, 28, 28, device=DEV)
    mask[..., :13, :] = 1
    known = d.repeat(6, 1, 1, 1)
    inp, _, _ = ddpm_sample(U, 6, 21, known=known, mask=mask.repeat(6, 1, 1, 1))
    shown = d[0, 0].cpu().numpy().copy()
    shown[13:, :] = np.nan
    vis = to_rgb(np.nan_to_num(shown, nan=-1))
    vis[13:, :] = np.array([0.98, 0.93, 0.83])
    plt.imsave(FIG / "inpaint-input.png", np.repeat(np.repeat(vis, 10, 0), 10, 1))
    grid(inp[:, 0], 1, 6, "inpaint-out", scale=5)

    # ---- SDEdit: a blocky painted 7, noised to t0, then denoised
    blocky = torch.nn.functional.interpolate(torch.nn.functional.avg_pool2d(d, 4), scale_factor=4, mode="nearest")
    blocky = torch.where(blocky > -0.8, torch.ones_like(blocky), -torch.ones_like(blocky))
    tile(blocky[0, 0].cpu(), "sdedit-input")
    NUM["sdedit"] = {}
    for t0 in (200, 400, 600, 800):
        g2 = torch.Generator().manual_seed(5)
        x0s = blocky.repeat(3, 1, 1, 1)
        e0 = torch.randn(x0s.shape, generator=g2).to(DEV)
        xinit = math.sqrt(ab[t0]) * x0s + math.sqrt(1 - ab[t0]) * e0
        out, _, _ = ddpm_sample(U, 3, 30 + t0, x_init=xinit, t_start=t0)
        grid(out[:, 0], 3, 1, f"sdedit-{t0}", scale=4)
        tile(xinit[0, 0].cpu(), f"sdedit-start-{t0}")
        NUM["sdedit"][str(t0)] = dict(dist=r(float(((torch.from_numpy(out) - blocky.cpu()) ** 2).mean()), 3))

    # ---- DDIM with 50 steps vs DDPM with 1000, same starting noise
    ddim = ddim_sample(U, 8, 11, 50)
    grid(ddim[:, 0], 1, 8, "ddim-50", scale=4)

    # ---- class-conditional model + classifier-free guidance
    if (MODELS / "cond.pt").exists():
        Cm, _ = load_unet("cond", 10)
        clf = Classifier().to(DEV).eval()
        clf.load_state_dict(torch.load(MODELS / "clf.pt", map_location=DEV))
        NUM["cfg"] = {}
        for s in (0, 1, 3, 7):
            out, _, _ = ddpm_sample(Cm, 64, 3, c=7, s=s)
            grid(out[:8, 0], 1, 8, f"cfg-{s}", scale=4)
            o = torch.from_numpy(out).to(DEV)
            pred = clf(o).argmax(1)
            NUM["cfg"][str(s)] = dict(frac7=int(round(100 * float((pred == 7).float().mean()))),
                                     div=r(float(torch.pdist(o.reshape(64, -1).cpu()).mean()), 1))
        # the two predictions for one noisy input
        tt = 600
        xin = math.sqrt(ab[tt]) * torch.zeros_like(d) + math.sqrt(1 - ab[tt]) * eps   # a blank page made noisy
        tn = torch.tensor([tt], device=DEV)
        eu = Cm(xin, tn, torch.tensor([10], device=DEV))
        ec = Cm(xin, tn, torch.tensor([7], device=DEV))
        lim = 2.5
        tile(xin[0, 0].cpu(), "cfg-in")
        tile(eu[0, 0].cpu(), "cfg-eu", signed=True, lim=lim)
        tile(ec[0, 0].cpu(), "cfg-ec", signed=True, lim=lim)
        tile((ec - eu)[0, 0].cpu(), "cfg-diff", signed=True, lim=0.6)
        tile((eu + 3 * (ec - eu))[0, 0].cpu(), "cfg-e3", signed=True, lim=lim)
        for name, e in (("u", eu), ("c", ec), ("3", eu + 3 * (ec - eu))):
            tile(((xin - math.sqrt(1 - ab[tt]) * e) / math.sqrt(ab[tt]))[0, 0].cpu(), f"cfg-x0-{name}")
        NUM["cond"] = dict(steps=torch.load(MODELS / "cond.pt", map_location="cpu")["steps"], p_drop=0.1, null=10)

    # ---- latent diffusion on the small autoencoder
    if (MODELS / "latent.pt").exists():
        ckae = torch.load(MODELS / "ae.pt", map_location=DEV)
        ae = ConvAE().to(DEV).eval()
        ae.load_state_dict(ckae["model"])
        z = ae.enc(d)
        rec = ae.dec(z)
        tile(rec[0, 0].cpu(), "ae-recon")
        zz = (z[0] / ckae["latent_std"]).cpu().numpy()
        for j in range(4):
            tile(zz[j], f"latent-{j}", scale=30, signed=True, lim=2.5)
        NUM["ae"] = dict(latent_std=r(ckae["latent_std"], 3), mse=r(float(((rec - d) ** 2).mean()), 4))
        ld = LatentDenoiser().to(DEV).eval()
        ld.load_state_dict(torch.load(MODELS / "latent.pt", map_location=DEV)["model"])
        zs = latent_sample(ld, 8, 4)
        dec = ae.dec(torch.from_numpy(zs).to(DEV) * ckae["latent_std"]).cpu().numpy()
        grid(dec[:, 0], 1, 8, "latent-samples", scale=4)
        tile(dec[0, 0], "latent-one")
        for j in range(4):
            tile(zs[0, j], f"latent-z0-{j}", scale=30, signed=True, lim=2.5)
        g3 = torch.Generator().manual_seed(4)
        zT = torch.randn(1, 4, 7, 7, generator=g3).numpy()[0]
        for j in range(4):
            tile(zT[j], f"latent-zT-{j}", scale=30, signed=True, lim=2.5)


@torch.no_grad()
def ddim_sample(model, n, seed, steps):
    beta, alpha, ab, var = (v.to(DEV) for v in schedule(DEV))
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=g).to(DEV)
    ts = np.linspace(T, 1, steps).round().astype(int)
    for i, t in enumerate(ts):
        s_ = int(ts[i + 1]) if i + 1 < len(ts) else 0
        e = model(x, torch.full((n,), int(t), device=DEV, dtype=torch.long))
        x0 = (x - (1 - ab[t]).sqrt() * e) / ab[t].sqrt()
        x = ab[s_].sqrt() * x0 + (1 - ab[s_]).sqrt() * e
    return x.cpu().numpy()


@torch.no_grad()
def latent_sample(model, n, seed):
    beta, alpha, ab, var = (v.to(DEV) for v in schedule(DEV))
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 4, 7, 7, generator=g).to(DEV)
    for step in range(T, 0, -1):
        e = model(z, torch.full((n,), step, device=DEV, dtype=torch.long))
        mean = (z - beta[step] * e / (1 - ab[step]).sqrt()) / alpha[step].sqrt()
        z = mean + (var[step].sqrt() * torch.randn(z.shape, generator=g).to(DEV) if step > 1 else 0)
    return z.cpu().numpy()


if __name__ == "__main__":
    analytic()
    if len(sys.argv) == 1 and (MODELS / "uncond.pt").exists():
        model_figures()
    NUMFILE.write_text(json.dumps(NUM, indent=1, ensure_ascii=False))
    print("numbers:", ", ".join(NUM))
