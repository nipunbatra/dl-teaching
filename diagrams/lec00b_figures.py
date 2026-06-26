"""Figures for Lecture 0B: Bayes, MAP, Regularization & KL.

Same parchment/ink palette as lec00. SVG out to figures/lec00/svg/
(lec00b shares the lec00 figure directory).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

OUT = Path(__file__).resolve().parent.parent / "figures" / "lec00" / "svg"
OUT.mkdir(parents=True, exist_ok=True)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name, tight=True):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  wrote {name}")


def _beta_pdf(x, a, b):
    from math import lgamma
    logB = lgamma(a) + lgamma(b) - lgamma(a + b)
    return np.exp((a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - logB)


# ---- 1. Bayesian updating: prior sharpens as data arrives --------------------
def bayesian_updating():
    x = np.linspace(1e-3, 1 - 1e-3, 400)
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    stages = [
        (2, 2, "prior  Beta(2,2)", MUTED, "--"),
        (5, 3, r"after 3H,1T $\to$ Beta(5,3)", SAGE, "-"),
        (17, 9, r"after 15H,7T $\to$ Beta(17,9)", RUST, "-"),
    ]
    for a, b, lbl, c, ls in stages:
        ax.plot(x, _beta_pdf(x, a, b), color=c, lw=2.4, ls=ls, label=lbl)
    ax.axvline(0.66, color=INK, lw=0.8, ls=":")
    ax.set_xlabel("p (coin bias)"); ax.set_ylabel("density")
    ax.set_title("Bayesian updating · today's posterior is tomorrow's prior",
                 fontsize=14, loc="left", color=INK, pad=10)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.set_xlim(0, 1)
    _clean(ax)
    save(fig, "bayesian_updating.svg")


# ---- 2. MAP geometry: real tiny dataset, SSE ellipses, prior circles, path --
def map_geometry_lambda():
    # concrete dataset (bias + 1 feature): (x,y) = (1,2),(2,2),(3,4)
    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([2.0, 2.0, 4.0])
    A = X.T @ X                      # [[3,6],[6,14]]
    Xty = X.T @ y                    # [8, 18]
    theta_ols = np.linalg.solve(A, Xty)   # [0.667, 1.0]

    g0 = np.linspace(-0.6, 1.6, 320)
    g1 = np.linspace(-0.3, 1.6, 320)
    T0, T1 = np.meshgrid(g0, g1)
    d0, d1 = T0 - theta_ols[0], T1 - theta_ols[1]
    sse = A[0, 0]*d0*d0 + 2*A[0, 1]*d0*d1 + A[1, 1]*d1*d1   # SSE - SSE_min
    prior = T0**2 + T1**2

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.contour(T0, T1, sse, levels=[0.4, 1.5, 4, 9], colors=[SLATE],
               linewidths=1.1, alpha=0.85)
    ax.contour(T0, T1, prior, levels=[0.25, 1, 2.25], colors=[SAGE],
               linewidths=1.0, linestyles="--", alpha=0.85)

    lams = np.concatenate([[0], np.geomspace(0.05, 200, 40)])
    path = np.array([np.linalg.solve(A + l*np.eye(2), Xty) for l in lams])
    ax.plot(path[:, 0], path[:, 1], color=RUST, lw=2.6, zorder=4)

    ax.scatter(*theta_ols, color=SLATE, s=90, zorder=5)
    ax.annotate(r"$\hat\theta_{\rm MLE}$ (OLS, $\lambda=0$)", theta_ols,
                xytext=(theta_ols[0]-0.05, theta_ols[1]+0.22), color=SLATE, fontsize=11)
    ax.scatter(0, 0, color=SAGE, s=70, zorder=5)
    ax.annotate(r"$\lambda\to\infty$", (0, 0),
                xytext=(0.05, -0.22), color=SAGE, fontsize=11)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color=SLATE, lw=1.4, label="data loss SSE — ellipses"),
        Line2D([0], [0], color=SAGE, lw=1.4, ls="--", label=r"prior $\theta_0^2+\theta_1^2$ — circles"),
        Line2D([0], [0], color=RUST, lw=2.4, label=r"MAP path as $\lambda$ grows"),
    ], frameon=False, fontsize=10.5, loc="lower right")

    ax.set_xlabel(r"$\theta_0$ (bias)"); ax.set_ylabel(r"$\theta_1$ (slope)")
    ax.set_title("MAP geometry · data ellipses vs prior circles, for a 3-point dataset",
                 fontsize=12.5, loc="left", color=INK, pad=10)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.3, 1.6)
    ax.set_aspect("equal")
    _clean(ax)
    save(fig, "map_geometry_lambda.svg")


# ---- 3. Gaussian prior in 2D: independence = isotropic circles ---------------
def gaussian_prior_2d():
    g = np.linspace(-3, 3, 200)
    T1, T2 = np.meshgrid(g, g)
    dens = np.exp(-(T1**2 + T2**2) / 2)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.contourf(T1, T2, dens, levels=12, cmap="BuGn", alpha=0.55)
    ax.contour(T1, T2, dens, levels=6, colors=[SAGE], linewidths=0.8, alpha=0.7)
    ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8)
    ax.scatter(0, 0, color=RUST, s=50, zorder=5)
    ax.text(0.1, -0.45, "0", color=RUST, fontsize=12)
    ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
    ax.set_title("Gaussian prior $p(\\theta)=\\prod_j \\mathcal{N}(\\theta_j;0,\\sigma_p^2)$",
                 fontsize=13, loc="left", color=INK, pad=10)
    ax.text(-2.8, 2.4, r"independent dims $\Rightarrow$ circular" "\n" r"contours (no preferred" "\n" r"direction, same $\sigma_p$ each axis)",
            color=INK, fontsize=11)
    ax.set_aspect("equal")
    _clean(ax)
    save(fig, "gaussian_prior_2d.svg")


# ---- 4. Laplace vs Gaussian densities (matched variance) --------------------
def laplace_vs_gaussian():
    x = np.linspace(-4, 4, 500)
    # match variance = 1: Gaussian sigma=1; Laplace b=1/sqrt(2)
    gauss = np.exp(-x**2 / 2) / np.sqrt(2*np.pi)
    b = 1/np.sqrt(2)
    lap = np.exp(-np.abs(x)/b) / (2*b)

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(x, gauss, color=SLATE, lw=2.4, label=r"Gaussian ($\to$ L2)")
    ax.plot(x, lap, color=RUST, lw=2.4, label=r"Laplace ($\to$ L1)")
    ax.fill_between(x, lap, color=RUST, alpha=0.08)
    ax.annotate("sharp peak at 0\n" r"$\to$ favors exact zeros", xy=(0, lap[250]),
                xytext=(0.6, 0.62), color=RUST, fontsize=11,
                arrowprops=dict(arrowstyle="-|>", color=RUST, lw=1.2))
    ax.annotate("heavier tails", xy=(2.6, lap[int((2.6+4)/8*500)]),
                xytext=(2.1, 0.22), color=RUST, fontsize=10)
    ax.set_xlabel(r"$\theta_j$"); ax.set_ylabel("prior density")
    ax.set_title("Laplace vs Gaussian prior · same variance", fontsize=14,
                 loc="left", color=INK, pad=10)
    ax.legend(frameon=False, fontsize=11)
    ax.set_ylim(0, 0.75)
    _clean(ax)
    save(fig, "laplace_vs_gaussian.svg")


# ---- 5. Gaussian prior cases: 3D surface, isotropic, diagonal, correlated ----
def gaussian_prior_cases():
    import numpy as np
    g = np.linspace(-3, 3, 120)
    T0, T1 = np.meshgrid(g, g)

    def dens(S):
        Si = np.linalg.inv(S)
        q = Si[0, 0]*T0**2 + 2*Si[0, 1]*T0*T1 + Si[1, 1]*T1**2
        return np.exp(-0.5 * q)

    fig = plt.figure(figsize=(11, 9))

    # (a) 3D surface, isotropic
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    Z = dens(np.eye(2))
    ax.plot_surface(T0, T1, Z, cmap="BuGn", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_title("isotropic  $\\sigma^2 I$  ·  the 3-D bump", fontsize=12, color=INK)
    ax.set_xlabel(r"$\theta_0$"); ax.set_ylabel(r"$\theta_1$")
    ax.set_zticks([])
    ax.view_init(elev=38, azim=-60)

    # (b) isotropic contours -> circles
    ax = fig.add_subplot(2, 2, 2)
    ax.contour(T0, T1, dens(np.eye(2)), levels=6, colors=[SAGE], linewidths=1.1)
    ax.set_title(r"isotropic $\to$ circles (independent, equal $\sigma$)", fontsize=12, color=INK)
    ax.set_aspect("equal"); _clean(ax)
    ax.set_xlabel(r"$\theta_0$"); ax.set_ylabel(r"$\theta_1$")

    # (c) diagonal, unequal -> axis-aligned ellipse
    ax = fig.add_subplot(2, 2, 3)
    ax.contour(T0, T1, dens(np.diag([2.2, 0.4])), levels=6, colors=[SLATE], linewidths=1.1)
    ax.set_title(r"diagonal $\Sigma \to$ axis-aligned ellipse" "\n" r"(independent, different $\sigma$)",
                 fontsize=12, color=INK)
    ax.set_aspect("equal"); _clean(ax)
    ax.set_xlabel(r"$\theta_0$"); ax.set_ylabel(r"$\theta_1$")

    # (d) full covariance -> tilted ellipse
    ax = fig.add_subplot(2, 2, 4)
    ax.contour(T0, T1, dens(np.array([[1.4, 0.9], [0.9, 1.0]])), levels=6,
               colors=[RUST], linewidths=1.1)
    ax.set_title(r"full $\Sigma \to$ tilted ellipse" "\n" r"(correlated weights)", fontsize=12, color=INK)
    ax.set_aspect("equal"); _clean(ax)
    ax.set_xlabel(r"$\theta_0$"); ax.set_ylabel(r"$\theta_1$")

    fig.suptitle("Gaussian prior · the covariance sets the shape (we use the simplest: isotropic)",
                 fontsize=13.5, color=INK)
    fig.tight_layout()
    save(fig, "gaussian_prior_cases.svg", tight=False)


# ---- 6. KL "extra bits" weather example (Jia-Bin Huang style) ---------------
def kl_weather():
    labels = ["Sunny", "Cloudy", "Rainy"]
    P = [0.5, 0.25, 0.25]
    Q = [0.25, 0.25, 0.5]
    x = np.arange(3)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.0), layout="constrained")
    a1.bar(x, P, color=SAGE, edgecolor=MUTED, lw=0.6)
    a1.set_title("true weather  P  ·  optimal code = 1.5 bits", fontsize=12, color=INK)
    for i, (p, code) in enumerate(zip(P, ["0", "10", "11"])):
        a1.text(i, p + 0.02, f"'{code}'", ha="center", color=INK, fontsize=11)
    a2.bar(x, Q, color=RUST, edgecolor=MUTED, lw=0.6)
    a2.set_title("forecast model  Q  ·  its code costs 1.75 bits on P", fontsize=12, color=INK)
    for i, (q, code) in enumerate(zip(Q, ["10", "11", "0"])):
        a2.text(i, q + 0.02, f"'{code}'", ha="center", color=INK, fontsize=11)
    for a in (a1, a2):
        a.set_xticks(x); a.set_xticklabels(labels)
        a.set_ylim(0, 0.62); a.set_ylabel("probability"); _clean(a)
    fig.suptitle(r"KL(P $\|$ Q) = 1.75 - 1.5 = 0.25 extra bits/day for using the wrong code",
                 fontsize=13, color=INK)
    save(fig, "kl_weather.svg", tight=False)


if __name__ == "__main__":
    print("Generating lec00b figures...")
    bayesian_updating()
    map_geometry_lambda()
    gaussian_prior_2d()
    gaussian_prior_cases()
    laplace_vs_gaussian()
    kl_weather()
    print("Done.")
