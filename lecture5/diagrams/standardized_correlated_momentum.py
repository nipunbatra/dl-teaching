"""Exact four-row example: standardized, correlated features still form a ravine.

Both feature columns have population mean zero and population variance one,
but their correlation is 4/5. The resulting least-squares curvature matrix has
eigenvalues 1/5 and 9/5, so the condition number is 9 and equal-loss contours
are three times longer in the shallow direction.

The optimizer comparison uses the same normalized EWMA momentum convention as
the lecture and the same conceptual learning rate as plain GD.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


INK = "#23373B"
ACC = "#EB811B"
TEAL = "#2C7A7B"
MUTED = "#6E7F82"
RED = "#D64550"
BLUE = "#2B6CB0"

OUT = Path("lecture5/figures")
GEOMETRY_ASSET = "story_standardized_correlated_geometry"
PATHS_ASSET = "story_standardized_correlated_paths"

# Four exact rows. We use the training-set/population convention (divide by N).
X = np.array(
    [
        [-1.0, -7.0 / 5.0],
        [-1.0, -1.0 / 5.0],
        [1.0, 1.0 / 5.0],
        [1.0, 7.0 / 5.0],
    ]
)
N = len(X)
THETA_STAR = np.array([1.0, 0.0])
Y = X[:, 0].copy()  # y = x_1, hence theta* = (1, 0)
THETA0 = np.array([0.0, 0.0])
HESSIAN = X.T @ X / N
EIGENVALUES, EIGENVECTORS = np.linalg.eigh(HESSIAN)
CONDITION_NUMBER = EIGENVALUES[-1] / EIGENVALUES[0]

# beta=.25 is chosen so the first few values remain easy to verify by hand.
# Both algorithms use the same conceptual learning rate.
GD_ETA = 1.0
BETA = 0.25
MOMENTUM_ETA = GD_ETA
STEPS = 18


mpl.rcParams.update(
    {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans", "Arial"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.6,
        "lines.solid_capstyle": "round",
        "svg.hashsalt": "dl-teaching-l5-exact-standardized-correlated",
    }
)


def loss(theta):
    residual = X @ theta - Y
    return 0.5 * np.mean(residual**2)


def gradient(theta):
    residual = X @ theta - Y
    return X.T @ residual / N


def gd_path(steps=STEPS):
    theta = THETA0.copy()
    path = [theta.copy()]
    for _ in range(steps):
        theta = theta - GD_ETA * gradient(theta)
        path.append(theta.copy())
    return np.asarray(path)


def momentum_path(steps=STEPS):
    theta = THETA0.copy()
    memory = np.zeros_like(theta)
    path = [theta.copy()]
    for _ in range(steps):
        memory = BETA * memory + (1 - BETA) * gradient(theta)
        theta = theta - MOMENTUM_ETA * memory
        path.append(theta.copy())
    return np.asarray(path)


def losses(path):
    return np.asarray([loss(theta) for theta in path])


def loss_grid(xlim=(-0.55, 2.1), ylim=(-1.35, 1.35), count=180):
    theta1 = np.linspace(*xlim, count)
    theta2 = np.linspace(*ylim, count)
    theta1_grid, theta2_grid = np.meshgrid(theta1, theta2)
    error1 = theta1_grid - THETA_STAR[0]
    error2 = theta2_grid - THETA_STAR[1]
    values = 0.5 * (
        HESSIAN[0, 0] * error1**2
        + 2.0 * HESSIAN[0, 1] * error1 * error2
        + HESSIAN[1, 1] * error2**2
    )
    return theta1_grid, theta2_grid, values


def clean(ax, *, grid=False):
    if grid:
        ax.grid(alpha=0.16, linewidth=0.7)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)


def validate(gd, momentum, gd_losses, momentum_losses):
    assert np.allclose(X.mean(axis=0), [0.0, 0.0])
    assert np.allclose(np.mean(X**2, axis=0), [1.0, 1.0])
    assert np.isclose(np.mean(X[:, 0] * X[:, 1]), 4.0 / 5.0)
    assert np.allclose(HESSIAN, [[1.0, 4.0 / 5.0], [4.0 / 5.0, 1.0]])
    assert np.allclose(EIGENVALUES, [1.0 / 5.0, 9.0 / 5.0])
    assert np.isclose(CONDITION_NUMBER, 9.0)

    same_direction = THETA_STAR + np.array([1.0, 1.0])
    opposite_direction = THETA_STAR + np.array([1.0, -1.0])
    assert np.isclose(loss(same_direction), 9.0 / 5.0)
    assert np.isclose(loss(opposite_direction), 1.0 / 5.0)
    assert np.isclose(loss(same_direction) / loss(opposite_direction), 9.0)

    assert np.allclose(gd[1], [1.0, 0.8])
    assert np.allclose(momentum[1], [0.75, 0.6])
    assert np.all(np.diff(gd_losses) < 0.0)
    assert np.all(np.diff(momentum_losses) < 0.0)
    assert np.all(momentum_losses[2:] < gd_losses[2:])
    assert momentum_losses[-1] < 0.06 * gd_losses[-1]


def build_geometry_figure():
    theta1_grid, theta2_grid, values = loss_grid()
    cmap = LinearSegmentedColormap.from_list(
        "lecture5_teal", ["#F7FBFB", "#B9DEDC", TEAL, INK]
    )

    fig = plt.figure(figsize=(13.333, 7.5))
    ax3d = fig.add_axes([0.035, 0.13, 0.47, 0.76], projection="3d")
    ax2d = fig.add_axes([0.565, 0.19, 0.39, 0.67])

    ax3d.plot_surface(
        theta1_grid,
        theta2_grid,
        values,
        cmap=cmap,
        rstride=4,
        cstride=4,
        linewidth=0.25,
        edgecolor=(0.14, 0.22, 0.23, 0.22),
        antialiased=True,
        alpha=0.95,
    )
    ax3d.scatter(
        [THETA_STAR[0]], [THETA_STAR[1]], [0], marker="*", s=150, color=RED,
        depthshade=False, zorder=10,
    )
    ax3d.set(
        xlabel=r"parameter $\theta_1$",
        ylabel=r"parameter $\theta_2$",
        zlabel=r"loss $\mathcal{L}$",
        xlim=(-0.55, 2.1),
        ylim=(-1.35, 1.35),
        zlim=(0, 4.2),
    )
    ax3d.set_title("The exact four-row loss surface", weight="bold", pad=10)
    ax3d.view_init(elev=28, azim=-55)
    ax3d.grid(alpha=0.14)
    ax3d.xaxis.pane.set_alpha(0.0)
    ax3d.yaxis.pane.set_alpha(0.0)
    ax3d.zaxis.pane.set_alpha(0.0)

    levels = np.geomspace(0.02, 2.6, 11)
    ax2d.contour(
        theta1_grid, theta2_grid, values, levels=levels,
        colors=[TEAL], linewidths=1.05, alpha=0.62,
    )
    ax2d.scatter(*THETA_STAR, marker="*", s=180, color=RED, zorder=8)
    ax2d.annotate(
        "same-sign move\nsteep: loss = 1.8",
        xy=THETA_STAR + np.array([0.72, 0.72]),
        xytext=THETA_STAR + np.array([0.30, 1.05]),
        ha="center", color=ACC, fontsize=12.5, weight="bold",
        arrowprops=dict(arrowstyle="->", color=ACC, lw=2.1),
    )
    ax2d.annotate(
        "opposite-sign move\nshallow: loss = 0.2",
        xy=THETA_STAR + np.array([0.72, -0.72]),
        xytext=THETA_STAR + np.array([0.74, -1.13]),
        ha="center", color=BLUE, fontsize=12.5, weight="bold",
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=2.1),
    )
    ax2d.set(
        xlabel=r"parameter $\theta_1$",
        ylabel=r"parameter $\theta_2$",
        xlim=(-0.55, 2.1),
        ylim=(-1.35, 1.35),
    )
    ax2d.set_aspect("equal")
    ax2d.set_title("Equal-length moves: 9× different loss", weight="bold", pad=12)
    clean(ax2d)

    fig.text(
        0.5,
        0.055,
        "eigenvalues 0.2 and 1.8  ·  condition number 9  ·  contour-axis ratio 3",
        ha="center",
        color=INK,
        fontsize=15.0,
        weight="bold",
    )
    return fig


def build_paths_figure(gd, momentum, gd_losses, momentum_losses):
    theta1_grid, theta2_grid, values = loss_grid()
    fig = plt.figure(figsize=(13.333, 7.5))
    ax_path = fig.add_axes([0.075, 0.18, 0.40, 0.69])
    ax_loss = fig.add_axes([0.575, 0.21, 0.37, 0.63])

    levels = np.geomspace(0.002, 2.6, 12)
    ax_path.contour(
        theta1_grid, theta2_grid, values, levels=levels,
        colors=[TEAL], linewidths=0.9, alpha=0.48,
    )
    ax_path.plot(
        gd[:, 0], gd[:, 1], color=ACC, marker="o", markersize=3.4,
        markevery=2, label="plain GD",
    )
    ax_path.plot(
        momentum[:, 0], momentum[:, 1], color=TEAL, marker="o", markersize=3.4,
        markevery=2, label="momentum",
    )
    ax_path.scatter(*THETA0, s=62, color=INK, zorder=8)
    ax_path.scatter(*THETA_STAR, marker="*", s=180, color=RED, zorder=9)
    ax_path.annotate("start", xy=THETA0, xytext=(-0.35, -0.18), color=INK, weight="bold")
    ax_path.annotate(
        r"optimum $\theta^*=(1,0)$", xy=THETA_STAR, xytext=(1.18, -0.30),
        color=RED, weight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
    )
    ax_path.set(
        xlabel=r"parameter $\theta_1$",
        ylabel=r"parameter $\theta_2$",
        xlim=(-0.55, 2.1),
        ylim=(-1.35, 1.35),
    )
    ax_path.set_aspect("equal")
    ax_path.set_title("Same start and learning rate", weight="bold", pad=12)
    ax_path.legend(frameon=False, loc="upper right")
    clean(ax_path)

    updates = np.arange(STEPS + 1)
    ax_loss.semilogy(updates, gd_losses, color=ACC, label="plain GD")
    ax_loss.semilogy(updates, momentum_losses, color=TEAL, label=rf"momentum  $\beta={BETA}$")
    ax_loss.scatter([STEPS], [gd_losses[-1]], color=ACC, s=42, zorder=5)
    ax_loss.scatter([STEPS], [momentum_losses[-1]], color=TEAL, s=42, zorder=5)
    ax_loss.text(
        STEPS - 0.4, gd_losses[-1] * 1.45, f"{gd_losses[-1]:.4f}",
        ha="right", va="bottom", color=ACC, fontsize=11.5, weight="bold",
    )
    ax_loss.text(
        STEPS - 0.4, momentum_losses[-1] * 1.45, f"{momentum_losses[-1]:.4f}",
        ha="right", va="bottom", color=TEAL, fontsize=11.5, weight="bold",
    )
    ax_loss.set(xlabel="exact full-batch updates", ylabel="full loss")
    ax_loss.set_title("Momentum damps the zigzag", weight="bold", pad=12)
    ax_loss.legend(frameon=False, loc="upper right")
    clean(ax_loss, grid=True)

    fig.text(
        0.5,
        0.085,
        r"same four rows · same start · same learning rate $\eta=1$",
        ha="center",
        color=INK,
        fontsize=14.5,
        weight="bold",
    )
    fig.text(
        0.5,
        0.042,
        r"normalized momentum: $\beta=.25,\ \eta=1$ · same learning rate as GD · both losses decrease",
        ha="center",
        color=MUTED,
        fontsize=11.2,
    )
    return fig


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"{stem}.svg"
    png_path = OUT / f"{stem}.png"
    fig.savefig(svg_path, transparent=True, metadata={"Date": None})
    fig.savefig(png_path, transparent=True, dpi=200, metadata={"Date": None})
    plt.close(fig)
    print(f"wrote {svg_path}")
    print(f"wrote {png_path}")


def main():
    gd = gd_path()
    momentum = momentum_path()
    gd_losses = losses(gd)
    momentum_losses = losses(momentum)
    validate(gd, momentum, gd_losses, momentum_losses)

    save(build_geometry_figure(), GEOMETRY_ASSET)
    save(build_paths_figure(gd, momentum, gd_losses, momentum_losses), PATHS_ASSET)

    print(f"feature means: {X.mean(axis=0)}")
    print(f"feature variances: {np.mean(X**2, axis=0)}")
    print(f"feature correlation: {np.mean(X[:, 0] * X[:, 1]):.3f}")
    print(f"Hessian:\n{HESSIAN}")
    print(f"Hessian eigenvalues: {EIGENVALUES}")
    print(f"condition number: {CONDITION_NUMBER:.1f}")
    print(f"contour-axis ratio: {np.sqrt(CONDITION_NUMBER):.1f}")
    print(f"GD loss after {STEPS}: {gd_losses[-1]:.8f}")
    print(f"momentum loss after {STEPS}: {momentum_losses[-1]:.8f}")


if __name__ == "__main__":
    main()
