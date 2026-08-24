"""Computed teaching figures for Lecture 5: Optimization for Deep Learning.

The figures deliberately reuse two small empirical objectives:

1. A four-point line fit for full-batch, stochastic, and minibatch updates.
2. A two-feature linear regression whose empirical loss is an exact ravine.

Nothing in the contour/path figures is hand-drawn. Run from the repository root:

    uv run --with matplotlib --with torch python lecture5/diagrams/l5_figs.py
"""

import copy
import random
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


INK = "#23373B"
ACC = "#EB811B"
TEAL = "#2C7A7B"
GREEN = "#14A66A"
MUTED = "#6E7F82"
RED = "#D64550"
BLUE = "#2B6CB0"
PURPLE = "#7B61A8"

mpl.rcParams.update(
    {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans", "Arial"],
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 1.0,
        "font.size": 13,
        "axes.titlesize": 13.5,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "svg.hashsalt": "dl-teaching-lecture5",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.4,
        "lines.solid_capstyle": "round",
    }
)

OUT = Path("lecture5/figures")
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(
        OUT / f"{name}.svg", bbox_inches="tight", transparent=True,
        metadata={"Date": None},
    )
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", transparent=True, dpi=200)
    plt.close(fig)


def clean(ax, *, grid=False):
    if grid:
        ax.grid(alpha=0.16, linewidth=0.7)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)


# ---------------------------------------------------------------------------
# Example A: four-point line fitting
# ---------------------------------------------------------------------------
X1 = np.array([-1.0, 0.0, 1.0, 2.0])
Y1 = np.array([-1.0, 1.0, 3.0, 6.0])
A1 = np.column_stack([np.ones_like(X1), X1])
THETA0 = np.array([2.0, -1.0])  # (bias, slope)
THETA_STAR = np.linalg.lstsq(A1, Y1, rcond=None)[0]


def line_loss(theta):
    residual = A1 @ theta - Y1
    return 0.5 * np.mean(residual**2)


def line_grad(theta, indices=None):
    if indices is None:
        indices = np.arange(len(X1))
    indices = np.asarray(indices, dtype=int)
    residual = A1[indices] @ theta - Y1[indices]
    return np.mean(residual[:, None] * A1[indices], axis=0)


def line_loss_grid(b_lim=(-0.8, 3.8), w_lim=(-1.8, 3.8), n=240):
    b = np.linspace(*b_lim, n)
    w = np.linspace(*w_lim, n)
    B, W = np.meshgrid(b, w)
    pred = B[..., None] + W[..., None] * X1
    loss = 0.5 * np.mean((pred - Y1) ** 2, axis=-1)
    return B, W, loss


def draw_line_data(ax, theta, title, *, highlight=None):
    xx = np.linspace(-1.35, 2.35, 100)
    ax.scatter(X1, Y1, s=58, color=BLUE, edgecolor="white", linewidth=0.8, zorder=5)
    if highlight is not None:
        ax.scatter(
            [X1[highlight]],
            [Y1[highlight]],
            s=150,
            facecolor=ACC,
            edgecolor=INK,
            linewidth=1.1,
            zorder=6,
        )
    ax.plot(xx, theta[0] + theta[1] * xx, color=INK, linewidth=2.7)
    ax.plot(xx, THETA_STAR[0] + THETA_STAR[1] * xx, color=TEAL, linewidth=1.8, linestyle="--")
    ax.set_xlim(-1.35, 2.35)
    ax.set_ylim(-2.2, 7.1)
    ax.set_xlabel("input $x$")
    ax.set_ylabel("target / prediction")
    ax.set_title(title, fontsize=12.5)
    clean(ax, grid=True)


def f_line_fit_revision():
    eta = 0.1
    gradient = line_grad(THETA0)
    theta1 = THETA0 - eta * gradient
    assert np.allclose(gradient, [-0.75, -4.5])
    assert line_loss(theta1) < line_loss(THETA0)

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.45))
    B, W, loss = line_loss_grid()
    levels = [0.05, 0.15, 0.4, 0.9, 1.8, 3.2, 5.0, 7.0, 10.0, 15.0]
    axes[0].contour(B, W, loss, levels=levels, colors=[TEAL], linewidths=0.85, alpha=0.72)
    axes[0].annotate(
        "",
        xy=theta1,
        xytext=THETA0,
        arrowprops=dict(arrowstyle="-|>", color=ACC, lw=2.6, mutation_scale=14),
    )
    axes[0].scatter(*THETA0, s=55, color=INK, zorder=5, label=rf"start $L={line_loss(THETA0):.3f}$")
    axes[0].scatter(*theta1, s=62, color=ACC, zorder=6, label=rf"after step $L={line_loss(theta1):.3f}$")
    axes[0].scatter(*THETA_STAR, marker="*", s=145, color=RED, zorder=6, label="least-squares optimum")
    axes[0].set(xlabel="bias $b$", ylabel="slope $w$", title=rf"parameter space · $\eta={eta}$")
    axes[0].set_aspect("equal")
    axes[0].legend(frameon=False, fontsize=8.6, loc="upper left")
    clean(axes[0])

    draw_line_data(
        axes[1],
        THETA0,
        rf"before · $b={THETA0[0]:.0f},\ w={THETA0[1]:.0f}$",
    )
    draw_line_data(
        axes[2],
        theta1,
        rf"after · $b={theta1[0]:.3f},\ w={theta1[1]:.2f}$",
    )
    axes[2].set_ylabel("")
    axes[1].text(-1.27, 6.55, "solid: current model", color=INK, fontsize=9.5)
    axes[1].text(-1.27, 5.95, "dashed: least-squares fit", color=TEAL, fontsize=9.5)
    fig.tight_layout()
    save(fig, "story_line_fit_revision")


def f_per_example_contours():
    """Show the four stochastic objectives that underlie Example A.

    Every panel uses the same parameter window, contour levels, aspect ratio,
    and displayed step size.  Arrow lengths are therefore directly
    comparable: each is exactly ``-eta * grad ell_i(theta_0)`` in parameter
    coordinates rather than a normalized direction.
    """
    b = np.linspace(-1.25, 4.50, 320)
    w = np.linspace(-2.00, 4.50, 320)
    B, W = np.meshgrid(b, w)
    eta_display = 0.10
    assert np.allclose(THETA0 - eta_display * line_grad(THETA0, [1]), [1.9, -1.0])
    levels = [0.0, 0.125, 0.5, 2.0, 4.5, 8.0, 12.5, 18.0, 24.5, 32.0]
    band_colors = [
        "#F3FAF9", "#E5F4F2", "#D3ECE9", "#BCE1DD", "#A0D2CE",
        "#82C0BC", "#62AAA7", "#438F8E", "#2C7A7B",
    ]
    equations = [
        r"\ell_1(b,w)=\frac{1}{2}(b-w+1)^2",
        r"\ell_2(b,w)=\frac{1}{2}(b-1)^2",
        r"\ell_3(b,w)=\frac{1}{2}(b+w-3)^2",
        r"\ell_4(b,w)=\frac{1}{2}(b+2w-6)^2",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.4), sharex=True, sharey=True)
    for i, (ax, x_i, y_i, equation) in enumerate(zip(axes.ravel(), X1, Y1, equations)):
        is_sampled = i == 1  # I_0 = 2 in the running stochastic update.
        residual = B + W * x_i - y_i
        loss = 0.5 * residual**2
        gradient = line_grad(THETA0, [i])
        endpoint = THETA0 - eta_display * gradient

        expected_gradient = (THETA0[0] + THETA0[1] * x_i - y_i) * np.array([1.0, x_i])
        assert np.allclose(gradient, expected_gradient)
        assert 0.5 * (endpoint[0] + endpoint[1] * x_i - y_i) ** 2 < 0.5 * (
            THETA0[0] + THETA0[1] * x_i - y_i
        ) ** 2

        if is_sampled:
            ax.set_facecolor("#FFF3E8")
        ax.contourf(B, W, loss, levels=levels, colors=band_colors, extend="max", alpha=0.52)
        ax.contour(B, W, loss, levels=levels[1:], colors=[TEAL], linewidths=0.60, alpha=0.58)

        # The zero-loss set is a line because one example supplies one linear
        # equation for the two parameters (b, w).
        if np.isclose(x_i, 0.0):
            ax.axvline(y_i, color=GREEN, linestyle="--", linewidth=2.0)
        else:
            ax.plot(b, (y_i - b) / x_i, color=GREEN, linestyle="--", linewidth=2.0)

        arrow_color = ACC if is_sampled else MUTED
        endpoint_color = ACC if is_sampled else TEAL
        proposal_alpha = 1.0 if is_sampled else 0.58
        ax.annotate(
            "",
            xy=endpoint,
            xytext=THETA0,
            arrowprops=dict(
                arrowstyle="-|>",
                color=arrow_color,
                lw=3.0 if is_sampled else 1.8,
                mutation_scale=16 if is_sampled else 13,
                alpha=proposal_alpha,
            ),
            zorder=8,
        )
        ax.scatter(*THETA0, s=58, color=INK, edgecolor="white", linewidth=0.8, zorder=9)
        ax.scatter(
            *endpoint,
            s=62 if is_sampled else 42,
            color=endpoint_color,
            edgecolor="white",
            linewidth=0.9,
            alpha=proposal_alpha,
            zorder=9,
        )
        if is_sampled:
            ax.annotate(
                "sampled this iteration\n" r"$\boldsymbol{\theta}_1=(1.9,-1)$",
                xy=endpoint,
                xytext=(11, 10),
                textcoords="offset points",
                color=ACC,
                fontsize=11.2,
                weight="bold",
            )
        ax.text(
            0.03,
            0.045,
            rf"$\mathbf{{g}}_{i + 1}(\boldsymbol{{\theta}}_0)=({gradient[0]:.0f},{gradient[1]:.0f})$",
            transform=ax.transAxes,
            fontsize=11.2,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.82),
        )
        ax.set_title(
            rf"$(x_{i + 1},y_{i + 1})=({x_i:.0f},{y_i:.0f})$" + "\n" + rf"${equation}$",
            fontsize=13.0,
            pad=3,
        )
        ax.set(xlim=(b.min(), b.max()), ylim=(w.min(), w.max()))
        # Identical wide axes keep arrows comparable across examples while
        # fitting the 2x2 story cleanly on a landscape lecture slide.
        ax.set_aspect("auto")
        clean(ax)
        if is_sampled:
            ax.add_patch(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=ax.transAxes,
                    fill=False,
                    edgecolor=ACC,
                    linewidth=2.4,
                    alpha=0.82,
                    clip_on=False,
                    zorder=20,
                )
            )

    for ax in axes[-1, :]:
        ax.set_xlabel("bias $b$")
    for ax in axes[:, 0]:
        ax.set_ylabel("slope $w$")
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.105,
        top=0.965,
        wspace=0.13,
        hspace=0.47,
    )
    save(fig, "story_per_example_contours")


def f_bad_sample_step():
    """Trace why a valid one-sample step can raise the full objective.

    The figure is deliberately a causal ledger rather than another optimizer
    path.  It shows (1) which example generated the step, (2) how every
    per-example loss changed after applying that step, and (3) the exact
    arithmetic that turns those changes into an increase in the mean loss.
    """
    eta = 0.1
    sampled_index = 1  # x=0, y=1
    gradient = line_grad(THETA0, [sampled_index])
    theta1 = THETA0 - eta * gradient
    assert np.allclose(gradient, [1.0, 0.0])
    assert line_loss(theta1) > line_loss(THETA0)

    before_contrib = 0.5 * (A1 @ THETA0 - Y1) ** 2
    after_contrib = 0.5 * (A1 @ theta1 - Y1) ** 2
    assert np.isclose(before_contrib[sampled_index], 0.5)
    assert np.isclose(after_contrib[sampled_index], 0.405)
    assert np.isclose(before_contrib.mean(), line_loss(THETA0))
    assert np.isclose(after_contrib.mean(), line_loss(theta1))

    delta = after_contrib - before_contrib
    sampled_delta = delta[sampled_index]
    other_delta = delta.sum() - sampled_delta
    assert np.allclose(before_contrib, [8.0, 0.5, 2.0, 18.0])
    assert np.allclose(after_contrib, [7.605, 0.405, 2.205, 18.605])
    assert np.allclose(delta, [-0.395, -0.095, 0.205, 0.605])
    assert np.isclose(sampled_delta, -0.095)
    assert np.isclose(other_delta, 0.415)
    assert np.isclose(delta.sum(), 0.320)
    assert np.isclose(delta.mean(), 0.080)

    # A wide causal story: sampled calculation -> all-example ledger -> mean.
    fig = plt.figure(figsize=(12.4, 4.65))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.17, 1.52, 0.98], wspace=0.30)
    step_ax = fig.add_subplot(grid[0, 0])
    ledger_ax = fig.add_subplot(grid[0, 1])
    mean_ax = fig.add_subplot(grid[0, 2])

    # 1. The one example that determines this update.
    step_ax.set_title("1  One draw chooses the step", loc="left", fontsize=15.0, weight="bold", pad=12)
    step_ax.set_xlim(0, 1)
    step_ax.set_ylim(0, 1)
    step_ax.axis("off")
    step_boxes = [
        (
            0.825,
            r"draw $I_0=2$: $(x_2,y_2)=(0,1)$" + "\n" + r"$\hat y_2=2+(-1)(0)=2$",
            ACC,
            "#FFF4E8",
        ),
        (
            0.570,
            r"$\widehat{\mathbf{g}}_0=(2-1)(1,0)=(1,0)$",
            TEAL,
            "white",
        ),
        (
            0.335,
            r"$\boldsymbol{\theta}_1=(2,-1)-0.1(1,0)$" + "\n" + r"$=(1.9,-1)$",
            TEAL,
            "white",
        ),
        (
            0.095,
            r"sampled $\ell_2$:  $0.500\;\rightarrow\;0.405\;\downarrow$",
            GREEN,
            "#ECF8F3",
        ),
    ]
    for y_pos, label, edge, fill in step_boxes:
        step_ax.text(
            0.50,
            y_pos,
            label,
            ha="center",
            va="center",
            fontsize=12.5,
            color=INK,
            linespacing=1.32,
            bbox=dict(boxstyle="round,pad=0.52", facecolor=fill, edgecolor=edge, linewidth=1.6),
        )
    for start_y, end_y in [(0.735, 0.665), (0.490, 0.425), (0.245, 0.185)]:
        step_ax.annotate(
            "",
            xy=(0.50, end_y),
            xytext=(0.50, start_y),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.7, mutation_scale=13),
        )

    # 2. Re-evaluate every example after taking that one-example step.
    rows = np.arange(4)
    bar_colors = [TEAL, ACC, RED, RED]
    ledger_ax.axhspan(0.56, 1.44, color="#FFF4E8", alpha=0.92, zorder=0)
    ledger_ax.axvline(0, color=INK, linewidth=1.15, alpha=0.75, zorder=1)
    ledger_ax.barh(rows, delta, height=0.52, color=bar_colors, alpha=0.92, zorder=3)
    ledger_ax.set_yticks(
        rows,
        [
            r"$i=1$   $x=-1$",
            r"$i=2$   $x=0$  sampled",
            r"$i=3$   $x=1$",
            r"$i=4$   $x=2$",
        ],
    )
    ledger_ax.invert_yaxis()
    ledger_ax.set_xlim(-0.58, 1.68)
    ledger_ax.set_xticks([-0.4, 0.0, 0.4, 0.8])
    ledger_ax.set_xlabel(r"change in loss  $\Delta\ell_i=\ell_i(\theta_1)-\ell_i(\theta_0)$", labelpad=8)
    ledger_ax.set_title("2  Re-score all four examples", loc="left", fontsize=15.0, weight="bold", pad=12)
    ledger_ax.text(
        1.64,
        -0.70,
        "before → after",
        ha="right",
        va="center",
        fontsize=10.8,
        color=MUTED,
        weight="bold",
    )
    for row, (before, after, change) in enumerate(zip(before_contrib, after_contrib, delta)):
        change_label = f"{change:+.3f}"
        if change <= -0.20:
            change_x, change_ha, change_color = change / 2, "center", "white"
        elif change < 0:
            change_x, change_ha, change_color = change - 0.035, "right", INK
        elif change >= 0.40:
            change_x, change_ha, change_color = change - 0.035, "right", "white"
        else:
            change_x, change_ha, change_color = change + 0.035, "left", INK
        ledger_ax.text(
            change_x,
            row,
            change_label,
            ha=change_ha,
            va="center",
            fontsize=11.8,
            color=change_color,
            weight="bold",
            zorder=5,
        )
        ledger_ax.text(
            1.64,
            row,
            f"{before:.3f} → {after:.3f}",
            ha="right",
            va="center",
            fontsize=11.3,
            color=ACC if row == sampled_index else INK,
            weight="bold" if row == sampled_index else "normal",
            zorder=5,
        )
    ledger_ax.grid(axis="x", alpha=0.17, linewidth=0.8)
    ledger_ax.spines["left"].set_visible(False)
    ledger_ax.spines["bottom"].set_color(MUTED)
    ledger_ax.tick_params(axis="y", length=0, pad=6)

    # 3. The exact aggregation that makes the full loss rise.
    mean_ax.set_title("3  Average the changes", loc="left", fontsize=15.0, weight="bold", pad=12)
    mean_ax.set_xlim(0, 1)
    mean_ax.set_ylim(0, 1)
    mean_ax.axis("off")
    mean_ax.text(0.04, 0.82, r"sampled:  $\Delta\ell_2=-0.095$", fontsize=13.0, color=GREEN)
    mean_ax.text(0.04, 0.68, r"other three:  $+0.415$", fontsize=13.0, color=RED)
    mean_ax.plot([0.04, 0.96], [0.605, 0.605], color=MUTED, lw=1.0, alpha=0.65)
    mean_ax.text(0.04, 0.50, r"$\sum_i\Delta\ell_i=+0.320$", fontsize=14.2, color=INK, weight="bold")
    mean_ax.text(0.04, 0.36, r"$\Delta\mathcal{L}=+0.320/4=+0.080$", fontsize=14.2, color=INK, weight="bold")
    mean_ax.text(
        0.50,
        0.185,
        r"full $\mathcal{L}$:  $7.125\;\rightarrow\;7.205\;\uparrow$",
        ha="center",
        va="center",
        fontsize=14.0,
        color=RED,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#FDEBEC", edgecolor=RED, linewidth=1.7),
    )
    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.17, top=0.89)
    save(fig, "story_bad_sample_step")


def f_gradient_estimates():
    """Show how the chosen examples determine both surface and gradient.

    The shared parameter window, start, and step size make every arrow an
    apples-to-apples update from ``THETA0``.  The bottom row uses complementary
    batches {1, 2} and {3, 4}, so their two gradients average exactly to the
    full gradient.  This gives the lecture one readable construction while the
    notebook remains the place to enumerate all six two-example batches.
    """
    display_eta = 0.10
    b = np.linspace(-0.40, 3.40, 320)
    w = np.linspace(-1.40, 3.50, 320)
    B, W = np.meshgrid(b, w)
    residuals = B[..., None] + W[..., None] * X1 - Y1
    individual_losses = 0.5 * residuals**2
    full_loss = individual_losses.mean(axis=-1)
    per_example_gradients = np.array([line_grad(THETA0, [i]) for i in range(4)])
    gradient_12 = line_grad(THETA0, [0, 1])
    gradient_34 = line_grad(THETA0, [2, 3])
    full_gradient = line_grad(THETA0)
    endpoint_12 = THETA0 - display_eta * gradient_12
    endpoint_34 = THETA0 - display_eta * gradient_34
    full_endpoint = THETA0 - display_eta * full_gradient
    assert np.allclose(per_example_gradients.mean(axis=0), full_gradient)
    assert np.allclose(0.5 * (gradient_12 + gradient_34), full_gradient)
    assert np.allclose(0.5 * (endpoint_12 + endpoint_34), full_endpoint)

    pair_12_loss = 0.5 * (individual_losses[..., 0] + individual_losses[..., 1])
    pair_34_loss = 0.5 * (individual_losses[..., 2] + individual_losses[..., 3])
    assert np.allclose(0.5 * (pair_12_loss + pair_34_loss), full_loss)

    levels = [0.10, 0.40, 1.0, 2.0, 4.0, 7.0, 11.0, 17.0]
    surface_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "l5_gradient_surface", ["#F6FBFA", "#D4ECE9", "#8CC5C2", TEAL]
    )
    equations = [
        r"$\ell_1=\frac{1}{2}(b-w+1)^2$",
        r"$\ell_2=\frac{1}{2}(b-1)^2$",
        r"$\ell_3=\frac{1}{2}(b+w-3)^2$",
        r"$\ell_4=\frac{1}{2}(b+2w-6)^2$",
    ]

    def draw_surface(ax, surface, gradient, *, arrow_color=ACC, featured=False):
        """Draw the displayed objective and its own gradient at theta_0."""
        ax.contourf(
            B,
            W,
            surface,
            levels=[0.0, *levels, 28.0],
            cmap=surface_cmap,
            extend="max",
            alpha=0.70,
        )
        ax.contour(
            B,
            W,
            surface,
            levels=levels,
            colors=[TEAL],
            linewidths=1.05 if featured else 0.78,
            alpha=0.76,
        )
        endpoint = THETA0 - display_eta * gradient
        ax.annotate(
            "",
            xy=endpoint,
            xytext=THETA0,
            arrowprops=dict(
                arrowstyle="-|>", color=arrow_color, lw=3.1, mutation_scale=16
            ),
            zorder=9,
        )
        ax.scatter(
            *endpoint,
            s=27,
            color=arrow_color,
            edgecolor="white",
            linewidth=0.6,
            zorder=10,
        )
        ax.scatter(
            *THETA0,
            s=43,
            color=INK,
            edgecolor="white",
            linewidth=0.8,
            zorder=10,
        )
        ax.set(xlim=(b.min(), b.max()), ylim=(w.min(), w.max()))
        ax.set_box_aspect(0.80)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([-1, 0, 1, 2, 3])
        clean(ax)
        return endpoint

    fig = plt.figure(figsize=(13.6, 7.30))
    grid = fig.add_gridspec(2, 24, hspace=0.38)
    top_axes = [fig.add_subplot(grid[0, start : start + 6]) for start in (0, 6, 12, 18)]
    batch_12_ax = fig.add_subplot(grid[1, 3:9])
    batch_34_ax = fig.add_subplot(grid[1, 9:15])
    full_ax = fig.add_subplot(grid[1, 15:21])

    # One example supplies a rank-one objective: its minimum is a line.
    for i, (ax, equation) in enumerate(zip(top_axes, equations)):
        endpoint = draw_surface(ax, individual_losses[..., i], per_example_gradients[i])
        if np.isclose(X1[i], 0.0):
            ax.axvline(Y1[i], color=GREEN, linestyle="--", linewidth=2.0, zorder=7)
        else:
            ax.plot(
                b,
                (Y1[i] - b) / X1[i],
                color=GREEN,
                linestyle="--",
                linewidth=2.0,
                zorder=7,
            )
        loss_before = 0.5 * (A1[i] @ THETA0 - Y1[i]) ** 2
        loss_after = 0.5 * (A1[i] @ endpoint - Y1[i]) ** 2
        assert loss_after < loss_before
        ax.set_title(f"Example {i + 1}\n{equation}", fontsize=12.8, pad=4, linespacing=1.15)

    # The two complementary minibatches and their average full objective.
    pair_specs = [
        (
            batch_12_ax,
            pair_12_loss,
            gradient_12,
            BLUE,
            "Examples {1, 2}\n" + r"$\mathcal{L}_{12}=\frac{\ell_1+\ell_2}{2}$",
            [0, 1],
        ),
        (
            batch_34_ax,
            pair_34_loss,
            gradient_34,
            PURPLE,
            "Examples {3, 4}\n" + r"$\mathcal{L}_{34}=\frac{\ell_3+\ell_4}{2}$",
            [2, 3],
        ),
    ]
    for ax, surface, gradient, color, title, indices in pair_specs:
        draw_surface(ax, surface, gradient, arrow_color=color, featured=True)
        optimum = np.linalg.lstsq(A1[indices], Y1[indices], rcond=None)[0]
        assert np.allclose(A1[indices] @ optimum, Y1[indices])
        ax.scatter(
            *optimum,
            marker="*",
            s=135,
            color=GREEN,
            edgecolor="white",
            linewidth=0.8,
            zorder=10,
        )
        ax.set_title(title, fontsize=13.4, pad=4, linespacing=1.15)

    draw_surface(full_ax, full_loss, full_gradient, arrow_color=ACC, featured=True)
    full_ax.scatter(
        *THETA_STAR,
        marker="*",
        s=145,
        color=GREEN,
        edgecolor="white",
        linewidth=0.8,
        zorder=10,
    )
    # Repeat the two minibatch vectors faintly: the full endpoint is exactly
    # their midpoint, which makes the gradient-average identity visible.
    for endpoint, color in ((endpoint_12, BLUE), (endpoint_34, PURPLE)):
        full_ax.annotate(
            "",
            xy=endpoint,
            xytext=THETA0,
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.8,
                linestyle="--",
                mutation_scale=13,
                alpha=0.75,
            ),
            zorder=7,
        )
        full_ax.scatter(*endpoint, s=27, color=color, alpha=0.82, zorder=8)
    full_ax.plot(
        [endpoint_12[0], endpoint_34[0]],
        [endpoint_12[1], endpoint_34[1]],
        color=MUTED,
        linestyle=":",
        linewidth=1.6,
        zorder=6,
    )
    full_ax.scatter(
        *full_endpoint,
        s=58,
        color=ACC,
        edgecolor="white",
        linewidth=0.8,
        zorder=11,
    )
    full_ax.annotate(
        "midpoint",
        xy=full_endpoint,
        xytext=(9, 8),
        textcoords="offset points",
        color=ACC,
        fontsize=10.3,
        weight="bold",
        zorder=12,
    )
    full_ax.set_title(
        "All four examples\n" + r"$\mathcal{L}=\frac{\mathcal{L}_{12}+\mathcal{L}_{34}}{2}$",
        fontsize=13.4,
        weight="bold",
        pad=4,
        linespacing=1.15,
    )

    # Reduce repeated furniture but keep a common, readable coordinate system.
    for ax in top_axes:
        ax.tick_params(labelbottom=False)
    for ax in top_axes[1:] + [batch_34_ax, full_ax]:
        ax.tick_params(labelleft=False)
    top_axes[0].set_ylabel("slope $w$")
    batch_12_ax.set_ylabel("slope $w$")
    for ax in (batch_12_ax, batch_34_ax, full_ax):
        ax.set_xlabel("bias $b$")

    fig.text(
        0.5,
        0.008,
        r"$\nabla\mathcal{L}(\boldsymbol{\theta}_0)="
        r"\frac{1}{2}[\nabla\mathcal{L}_{12}(\boldsymbol{\theta}_0)+"
        r"\nabla\mathcal{L}_{34}(\boldsymbol{\theta}_0)]$"
        "     ·     every arrow uses the displayed surface     ·     green: zero-loss set / minimizer",
        ha="center",
        va="bottom",
        fontsize=10.8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.145, top=0.93)
    save(fig, "story_gradient_surfaces_batch_family")


# ---------------------------------------------------------------------------
# Estimator-first reveal sequence for Example A
# ---------------------------------------------------------------------------


def _example_a_surface_components():
    """Return one shared grid and the exact Example A component losses."""
    b = np.linspace(-0.40, 3.40, 320)
    w = np.linspace(-1.40, 3.50, 320)
    B, W = np.meshgrid(b, w)
    residuals = B[..., None] + W[..., None] * X1 - Y1
    individual = 0.5 * residuals**2
    return b, w, B, W, individual


_EXAMPLE_A_LEVELS = [0.125, 0.5, 1.25, 2.5, 4.5, 8.0, 13.0, 20.0]
_EXAMPLE_A_SURFACE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "l5_estimator_surface", ["#F7FBFA", "#D9EFEC", "#91C8C4", TEAL]
)
_EXAMPLE_A_EQUATIONS = [
    r"$\ell_1(b,w)=\frac{1}{2}(b-w+1)^2$",
    r"$\ell_2(b,w)=\frac{1}{2}(b-1)^2$",
    r"$\ell_3(b,w)=\frac{1}{2}(b+w-3)^2$",
    r"$\ell_4(b,w)=\frac{1}{2}(b+2w-6)^2$",
]


def _draw_example_a_surface(
    ax,
    B,
    W,
    surface,
    *,
    zero_loss_index=None,
    update=None,
    update_color=ACC,
    show_theta_label=True,
    show_ticks=True,
):
    """Draw a loss surface and, optionally, its exact update from theta_0."""
    ax.contourf(
        B,
        W,
        surface,
        levels=[0.0, *_EXAMPLE_A_LEVELS, 36.0],
        cmap=_EXAMPLE_A_SURFACE_CMAP,
        extend="max",
        alpha=0.72,
    )
    ax.contour(
        B,
        W,
        surface,
        levels=_EXAMPLE_A_LEVELS,
        colors=[TEAL],
        linewidths=0.86,
        alpha=0.76,
    )

    # A single linear example constrains (b, w) to a line, not a point.
    if zero_loss_index is not None:
        b = B[0]
        x_i = X1[zero_loss_index]
        y_i = Y1[zero_loss_index]
        if np.isclose(x_i, 0.0):
            ax.axvline(y_i, color=GREEN, linestyle="--", linewidth=2.1, zorder=7)
        else:
            ax.plot(
                b,
                (y_i - b) / x_i,
                color=GREEN,
                linestyle="--",
                linewidth=2.1,
                zorder=7,
            )

    ax.scatter(
        *THETA0,
        s=62,
        color=INK,
        edgecolor="white",
        linewidth=0.9,
        zorder=10,
    )
    if show_theta_label:
        ax.annotate(
            r"$\boldsymbol{\theta}_0$",
            xy=THETA0,
            xytext=(-8, -15),
            textcoords="offset points",
            ha="right",
            color=INK,
            fontsize=11.0,
            weight="bold",
            zorder=11,
        )

    if update is not None:
        endpoint = THETA0 + update
        ax.annotate(
            "",
            xy=endpoint,
            xytext=THETA0,
            arrowprops=dict(
                arrowstyle="-|>",
                color=update_color,
                lw=3.1,
                mutation_scale=17,
            ),
            zorder=12,
        )
        ax.scatter(
            *endpoint,
            s=52,
            color=update_color,
            edgecolor="white",
            linewidth=0.8,
            zorder=13,
        )

    ax.set(
        xlim=(B.min(), B.max()),
        ylim=(W.min(), W.max()),
        xticks=[0, 1, 2, 3],
        yticks=[-1, 0, 1, 2, 3],
    )
    ax.set_box_aspect(0.80)
    if not show_ticks:
        ax.tick_params(labelbottom=False, labelleft=False)
    clean(ax)


def _format_signed_pair(vector, places=2):
    # Avoid pedagogically distracting ``-0.00`` from floating-point signed zero.
    vector = np.where(np.isclose(vector, 0.0), 0.0, vector)
    return rf"({vector[0]:+.{places}f},\,{vector[1]:+.{places}f})"


def f_individual_losses():
    """Four one-example objectives at the same parameter; deliberately no arrows."""
    _, _, B, W, individual = _example_a_surface_components()

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.75), sharex=True, sharey=True)
    for i, ax in enumerate(axes.ravel()):
        _draw_example_a_surface(
            ax,
            B,
            W,
            individual[..., i],
            zero_loss_index=i,
            show_theta_label=True,
        )
        ax.set_title(
            f"Example {i + 1}:  $(x_{i + 1},y_{i + 1})=({X1[i]:.0f},{Y1[i]:.0f})$\n"
            + _EXAMPLE_A_EQUATIONS[i],
            fontsize=14.0,
            pad=5,
            linespacing=1.22,
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("bias $b$")
    for ax in axes[:, 0]:
        ax.set_ylabel("slope $w$")
    fig.text(
        0.50,
        0.014,
        r"same model $\boldsymbol{\theta}_0=(2,-1)$ in every panel"
        r"  $\cdot$  dashed green: zero-loss line for that one example",
        ha="center",
        va="bottom",
        fontsize=11.3,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.94, wspace=0.13, hspace=0.42)
    save(fig, "story_individual_losses")


def f_individual_updates():
    """Add each loss surface's exact descent update to the previous visual."""
    eta = 0.10
    _, _, B, W, individual = _example_a_surface_components()
    gradients = np.array([line_grad(THETA0, [i]) for i in range(len(X1))])
    updates = -eta * gradients
    expected = np.array([[-0.4, 0.4], [-0.1, 0.0], [0.2, 0.2], [0.6, 1.2]])
    assert np.allclose(updates, expected)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.75), sharex=True, sharey=True)
    for i, ax in enumerate(axes.ravel()):
        _draw_example_a_surface(
            ax,
            B,
            W,
            individual[..., i],
            zero_loss_index=i,
            update=updates[i],
            show_theta_label=True,
        )
        endpoint = THETA0 + updates[i]
        before = 0.5 * (A1[i] @ THETA0 - Y1[i]) ** 2
        after = 0.5 * (A1[i] @ endpoint - Y1[i]) ** 2
        assert after < before
        ax.set_title(
            f"Example {i + 1}:  " + _EXAMPLE_A_EQUATIONS[i] + "\n"
            + rf"$\Delta\boldsymbol{{\theta}}_{i + 1}=-\eta\nabla\ell_{i + 1}="
            + _format_signed_pair(updates[i])
            + r"$",
            fontsize=13.4,
            pad=5,
            linespacing=1.22,
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("bias $b$")
    for ax in axes[:, 0]:
        ax.set_ylabel("slope $w$")
    fig.text(
        0.50,
        0.014,
        r"orange: the exact parameter update for that panel, with $\eta=0.10$"
        r"  $\cdot$  coordinates are $(\Delta b,\Delta w)$",
        ha="center",
        va="bottom",
        fontsize=11.3,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.94, wspace=0.13, hspace=0.45)
    save(fig, "story_individual_updates")


def f_batch_loss_surfaces():
    """Use nested selections to show how averaging changes surface and update."""
    eta = 0.10
    _, _, B, W, individual = _example_a_surface_components()
    selections = [
        ([1], "the worked SGD sample", r"$B=1:\quad \widehat{\mathcal{L}}=\ell_2$", ACC),
        ([0, 1], "the worked minibatch", r"$B=2:\quad \widehat{\mathcal{L}}=\frac{\ell_1+\ell_2}{2}$", TEAL),
        ([0, 1, 2, 3], "the full dataset", r"$B=4:\quad \widehat{\mathcal{L}}=\mathcal{L}$", INK),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.1, 4.55), sharex=True, sharey=True)
    for ax, (indices, description, equation, color) in zip(axes, selections):
        surface = individual[..., indices].mean(axis=-1)
        gradient = line_grad(THETA0, indices)
        update = -eta * gradient
        _draw_example_a_surface(
            ax,
            B,
            W,
            surface,
            zero_loss_index=indices[0] if len(indices) == 1 else None,
            update=update,
            update_color=color,
            show_theta_label=True,
        )
        before = 0.5 * np.mean((A1[indices] @ THETA0 - Y1[indices]) ** 2)
        endpoint = THETA0 + update
        after = 0.5 * np.mean((A1[indices] @ endpoint - Y1[indices]) ** 2)
        assert after < before
        ax.set_title(
            description + "\n" + equation + "\n" + r"$\Delta\boldsymbol{\theta}_B="
            + _format_signed_pair(update, places=3)
            + r"$",
            fontsize=14.0,
            pad=6,
            linespacing=1.28,
            color=color,
        )
        ax.set_xlabel("bias $b$")
    axes[0].set_ylabel("slope $w$")
    fig.text(
        0.50,
        0.012,
        r"same $\boldsymbol{\theta}_0$ and $\eta=0.10$"
        r"  $\cdot$  averaging examples changes both the loss surface and its gradient",
        ha="center",
        va="bottom",
        fontsize=11.4,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.16, top=0.90, wspace=0.13)
    save(fig, "story_batch_loss_surfaces")


def _draw_update_vector_average(*, show_average):
    """Common canvas for a reveal from four updates to their exact mean."""
    eta = 0.10
    updates = np.array([-eta * line_grad(THETA0, [i]) for i in range(len(X1))])
    full_update = -eta * line_grad(THETA0)
    assert np.allclose(updates.mean(axis=0), full_update)
    colors = [BLUE, TEAL, PURPLE, GREEN]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.axhline(0, color=MUTED, linewidth=0.9, alpha=0.45)
    ax.axvline(0, color=MUTED, linewidth=0.9, alpha=0.45)
    for i, (update, color) in enumerate(zip(updates, colors)):
        ax.annotate(
            "",
            xy=update,
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=2.7,
                mutation_scale=16,
                alpha=0.90,
            ),
            zorder=5,
        )
        ax.scatter(*update, s=48, color=color, edgecolor="white", linewidth=0.7, zorder=6)
        offset = [(-12, 7), (-26, 10), (8, -14), (8, 5)][i]
        ax.annotate(
            rf"$\Delta\boldsymbol{{\theta}}_{i + 1}$",
            xy=update,
            xytext=offset,
            textcoords="offset points",
            fontsize=12.5,
            color=color,
            weight="bold",
        )

    if show_average:
        # The orange endpoint is the coordinate-wise centroid of the four
        # proposal endpoints and exactly the full-batch update.
        ax.annotate(
            "",
            xy=full_update,
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=ACC,
                lw=4.3,
                mutation_scale=20,
            ),
            zorder=8,
        )
        ax.scatter(
            *full_update,
            marker="D",
            s=105,
            color=ACC,
            edgecolor="white",
            linewidth=1.0,
            zorder=9,
        )
        ax.annotate(
            r"mean $=$ full-batch update",
            xy=full_update,
            xytext=(0.43, 0.45),
            textcoords="data",
            fontsize=13.2,
            color=ACC,
            weight="bold",
            va="center",
            bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="none", alpha=0.88),
            arrowprops=dict(arrowstyle="->", color=ACC, lw=1.5, shrinkA=3, shrinkB=7),
        )
        statement = None
    else:
        statement = "Each sampled example proposes a different update from the same model."

    ax.scatter(0, 0, s=78, color=INK, edgecolor="white", linewidth=0.9, zorder=10)
    ax.set(
        xlim=(-0.55, 1.15),
        ylim=(-0.14, 1.38),
        xlabel=r"change in bias  $\Delta b$",
        ylabel=r"change in slope  $\Delta w$",
    )
    ax.set_aspect("equal")
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    clean(ax, grid=True)
    if statement is not None:
        fig.text(0.50, 0.022, statement, ha="center", va="bottom", fontsize=12.6, color=INK)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.19 if statement is not None else 0.14,
        top=0.97,
    )
    return fig


def f_update_vector_fan():
    """Progressive base: the four possible one-example updates only."""
    save(_draw_update_vector_average(show_average=False), "story_update_vector_fan")


def f_update_vector_average():
    """Final reveal: individual update centroid equals the full-batch update."""
    save(_draw_update_vector_average(show_average=True), "story_update_vector_average")


def f_estimator_clouds():
    """Exact gradient-estimator clouds for uniform subsets of each batch size."""
    pairs = list(combinations(range(len(X1)), 2))
    gradient_sets = [
        np.array([line_grad(THETA0, [i]) for i in range(len(X1))]),
        np.array([line_grad(THETA0, pair) for pair in pairs]),
        np.array([line_grad(THETA0)]),
    ]
    full_gradient = line_grad(THETA0)
    for gradients in gradient_sets:
        assert np.allclose(gradients.mean(axis=0), full_gradient)

    panels = [
        (r"$B=1$ · choose one row", "4 possible gradients", BLUE,
         [f"row {i + 1}" for i in range(len(X1))],
         [(5, 5), (6, 5), (6, 5), (6, 2)]),
        (r"$B=2$ · choose two rows", "6 possible averages", TEAL,
         [rf"${{{i + 1},{j + 1}}}$" for i, j in pairs],
         [(6, 5), (6, 5), (6, 4), (6, -14), (6, 4), (6, 4)]),
        (r"$B=4$ · use every row", "1 exact gradient", INK,
         [r"$\{1,2,3,4\}$"], [(9, 10)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.9, 4.05), sharex=True, sharey=True)
    for ax, gradients, (batch_label, description, color, labels, offsets) in zip(axes, gradient_sets, panels):
        ax.axhline(full_gradient[1], color=ACC, linewidth=1.1, linestyle=":", alpha=0.58)
        ax.axvline(full_gradient[0], color=ACC, linewidth=1.1, linestyle=":", alpha=0.58)
        ax.axhline(0, color=MUTED, linewidth=0.8, alpha=0.34)
        ax.axvline(0, color=MUTED, linewidth=0.8, alpha=0.34)
        for gradient in gradients:
            ax.annotate(
                "",
                xy=gradient,
                xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=2.0,
                    mutation_scale=13,
                    alpha=0.42 if len(gradients) > 1 else 0.86,
                ),
                zorder=3,
            )
        ax.scatter(
            gradients[:, 0],
            gradients[:, 1],
            s=69,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        for gradient, label, offset in zip(gradients, labels, offsets):
            ax.annotate(
                label,
                xy=gradient,
                xytext=offset,
                textcoords="offset points",
                fontsize=9.8,
                color=color,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.78),
                zorder=7,
            )
        ax.scatter(
            *full_gradient,
            marker="D",
            s=105,
            color=ACC,
            edgecolor="white",
            linewidth=1.0,
            zorder=8,
        )
        ax.set_title(
            batch_label + "\n" + description,
            fontsize=16.0,
            color=color,
            weight="bold",
            pad=8,
        )
        ax.set(
            xlim=(-6.7, 4.7),
            ylim=(-12.7, 1.2),
            xlabel=r"gradient component  $\widehat g_b$",
        )
        ax.set_aspect("equal")
        # This slide is about centre and spread, not numerical covariance.
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
        clean(ax)
    axes[0].set_ylabel(r"gradient component  $\widehat g_w$")
    fig.text(
        0.50,
        0.018,
        r"labels identify the selected row(s) $\cdot$ orange diamond: the same mean"
        r" $=\boldsymbol{g}_{\mathrm{full}}$ in every panel",
        ha="center",
        va="bottom",
        fontsize=12.1,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.16, top=0.90, wspace=0.12)
    save(fig, "story_estimator_clouds")


MATCHED_EPOCH_STEP = 0.075


def decimal_label(value, places=5):
    return f"{value:.{places}f}".rstrip("0").rstrip(".")


def matched_batch_eta(batch_size):
    """Match the nominal sum of learning rates within one N=4 epoch."""
    return MATCHED_EPOCH_STEP * batch_size / len(X1)


def run_shuffled_epochs(batch_size, *, epochs=16, eta=None, seed=5):
    if eta is None:
        eta = matched_batch_eta(batch_size)
    rng = np.random.default_rng(seed)
    theta = THETA0.copy()
    path = [theta.copy()]
    seen = [0]
    losses = [line_loss(theta)]
    total_seen = 0
    for _ in range(epochs):
        order = rng.permutation(len(X1))
        for start in range(0, len(X1), batch_size):
            indices = order[start : start + batch_size]
            theta = theta - eta * line_grad(theta, indices)
            total_seen += len(indices)
            path.append(theta.copy())
            seen.append(total_seen)
            losses.append(line_loss(theta))
    return np.asarray(path), np.asarray(seen), np.asarray(losses)


def f_batch_paths():
    """Compare actual Example A update directions at one fixed parameter.

    The displayed arrows are the normalized negative gradients for all
    possible one-example draws, all two-example averages, or the exact full
    average at ``THETA0``.  Normalizing their lengths keeps the diagram about
    direction disagreement rather than magnitude or training speed.
    """
    individual_updates = np.array([-line_grad(THETA0, [i]) for i in range(len(X1))])
    pair_updates = np.array([-line_grad(THETA0, pair) for pair in combinations(range(len(X1)), 2)])
    full_update = -line_grad(THETA0)
    assert np.allclose(individual_updates.mean(axis=0), full_update)
    assert np.allclose(pair_updates.mean(axis=0), full_update)

    def unit_directions(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        assert np.all(norms > 0)
        return vectors / norms

    exact_direction = unit_directions(full_update[None, :])[0]
    direction_sets = [
        exact_direction[None, :],
        unit_directions(pair_updates),
        unit_directions(individual_updates),
    ]

    fig, ax = plt.subplots(figsize=(12.8, 5.25))
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 5.25)
    ax.axis("off")

    ax.text(
        0.45,
        5.01,
        "SAME MODEL  ·  LOSS  ·  DATASET  ·  STARTING POINT θ₀",
        ha="left",
        va="center",
        fontsize=12.9,
        color=TEAL,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#EAF5F4", edgecolor="none"),
    )
    ax.text(
        12.35,
        5.01,
        "MORE AVERAGING  →  TIGHTER FAN",
        ha="right",
        va="center",
        fontsize=13.2,
        color=INK,
        weight="bold",
    )

    for x_pos, label in [
        (2.72, "ONE GRADIENT ESTIMATE USES"),
        (7.05, "POSSIBLE UPDATE DIRECTIONS AT θ₀"),
        (10.90, "WHAT TO NOTICE"),
    ]:
        ax.text(
            x_pos,
            4.57,
            label,
            ha="center",
            va="center",
            fontsize=11.4,
            color=MUTED,
            weight="bold",
        )

    lanes = [
        {
            "y": 3.70,
            "title": "Full batch",
            "selection": "average all examples",
            "selected": {0, 1, 2, 3},
            "node": "mean",
            "notice": "exact direction",
            "color": INK,
            "tint": "#F3F6F6",
        },
        {
            "y": 2.48,
            "title": "Minibatch",
            "selection": "average a few",
            "selected": {0, 3},
            "node": "mean",
            "notice": "some disagreement cancels",
            "color": TEAL,
            "tint": "#EEF7F6",
        },
        {
            "y": 1.26,
            "title": "SGD",
            "selection": "use one example",
            "selected": {2},
            "node": "one",
            "notice": "varies most",
            "color": ACC,
            "tint": "#FFF5EA",
        },
    ]

    example_offsets = [(-0.31, 0.18), (0.31, 0.18), (-0.31, -0.18), (0.31, -0.18)]
    fan_origin_x = 6.55
    fan_length = 0.78
    for lane, directions in zip(lanes, direction_sets):
        y_pos = lane["y"]
        color = lane["color"]
        ax.add_patch(
            mpl.patches.FancyBboxPatch(
                (0.36, y_pos - 0.50),
                12.08,
                1.00,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                linewidth=0.8,
                edgecolor="#D7E0E1",
                facecolor=lane["tint"],
                zorder=0,
            )
        )

        ax.text(
            0.72,
            y_pos + 0.14,
            lane["title"],
            ha="left",
            va="center",
            fontsize=17.0,
            color=color,
            weight="bold",
        )
        ax.text(
            0.72,
            y_pos - 0.19,
            lane["selection"],
            ha="left",
            va="center",
            fontsize=12.4,
            color=INK,
        )

        # Repeating all four dots makes the invariant dataset visible.  Filled
        # dots are one representative selection; the fan shows every possible
        # selection of that size.
        dot_center_x = 2.78
        merge_x = 3.78
        for index, (dx, dy) in enumerate(example_offsets):
            selected = index in lane["selected"]
            dot_x = dot_center_x + dx
            dot_y = y_pos + dy
            ax.plot(
                [dot_x + 0.10, merge_x - 0.31],
                [dot_y, y_pos],
                color=color if selected else "#CAD4D5",
                linewidth=1.55 if selected else 0.9,
                alpha=0.82 if selected else 0.40,
                zorder=1,
            )
            ax.scatter(
                dot_x,
                dot_y,
                s=115,
                facecolor=color if selected else "#E3E9E9",
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )
        ax.add_patch(
            mpl.patches.FancyBboxPatch(
                (merge_x - 0.29, y_pos - 0.17),
                0.58,
                0.34,
                boxstyle="round,pad=0.02,rounding_size=0.14",
                linewidth=1.8,
                edgecolor=color,
                facecolor="white",
                zorder=4,
            )
        )
        ax.text(
            merge_x,
            y_pos,
            lane["node"],
            ha="center",
            va="center",
            fontsize=9.4,
            color=color,
            weight="bold",
            zorder=5,
        )
        ax.annotate(
            "",
            xy=(4.78, y_pos),
            xytext=(4.18, y_pos),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6, mutation_scale=13),
        )

        origin_y = y_pos - 0.34
        # The faint dashed arrow is the exact full-gradient reference in every
        # lane.  A common origin and common arrow length isolate direction.
        ref_end = (
            fan_origin_x + fan_length * exact_direction[0],
            origin_y + fan_length * exact_direction[1],
        )
        ax.annotate(
            "",
            xy=ref_end,
            xytext=(fan_origin_x, origin_y),
            arrowprops=dict(
                arrowstyle="-|>",
                color=MUTED,
                lw=1.6,
                linestyle="--",
                alpha=0.72,
                mutation_scale=13,
            ),
            zorder=2,
        )
        for direction in directions:
            endpoint = (
                fan_origin_x + fan_length * direction[0],
                origin_y + fan_length * direction[1],
            )
            is_full = lane["title"] == "Full batch"
            ax.annotate(
                "",
                xy=endpoint,
                xytext=(fan_origin_x, origin_y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=3.2 if is_full else 2.25,
                    alpha=0.98 if is_full else 0.58,
                    mutation_scale=15,
                ),
                zorder=3,
            )
        ax.scatter(
            fan_origin_x,
            origin_y,
            s=70,
            color=BLUE,
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )
        ax.text(
            fan_origin_x - 0.14,
            origin_y,
            "θ₀",
            ha="right",
            va="center",
            fontsize=11.2,
            color=BLUE,
            weight="bold",
        )

        ax.annotate(
            "",
            xy=(9.28, y_pos),
            xytext=(8.68, y_pos),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6, mutation_scale=13),
        )
        ax.text(
            10.82,
            y_pos,
            lane["notice"],
            ha="center",
            va="center",
            fontsize=14.5,
            color=color,
            weight="bold",
        )

    # Compact keys: filled/pale examples and the common exact reference.
    ax.scatter(0.58, 0.47, s=95, facecolor=INK, edgecolor="white", linewidth=1.0)
    ax.scatter(0.95, 0.47, s=95, facecolor="#E3E9E9", edgecolor="white", linewidth=1.0)
    ax.text(
        1.17,
        0.47,
        "used now     used later",
        ha="left",
        va="center",
        fontsize=11.0,
        color=MUTED,
    )
    ax.annotate(
        "",
        xy=(4.28, 0.47),
        xytext=(3.60, 0.47),
        arrowprops=dict(
            arrowstyle="-|>",
            color=MUTED,
            lw=1.6,
            linestyle="--",
            mutation_scale=12,
        ),
    )
    ax.text(
        4.48,
        0.47,
        "exact full-gradient reference",
        ha="left",
        va="center",
        fontsize=11.0,
        color=MUTED,
    )
    ax.text(
        12.38,
        0.47,
        "COMPUTED FROM EXAMPLE A · DIRECTION-ONLY SCHEMATIC",
        ha="right",
        va="center",
        fontsize=10.8,
        color=MUTED,
        weight="bold",
    )

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    save(fig, "story_batch_paths")


def f_batch_loss():
    specs = [(4, "full batch", INK), (2, "minibatch", TEAL), (1, "stochastic", ACC)]
    fig, ax = plt.subplots(figsize=(7.5, 3.25))
    optimum = line_loss(THETA_STAR)
    for batch_size, label, color in specs:
        eta = matched_batch_eta(batch_size)
        _, seen, losses = run_shuffled_epochs(batch_size, epochs=28, eta=eta)
        ax.plot(
            seen, losses - optimum, "o-", color=color, markersize=2.6, linewidth=2,
            label=rf"{label} · $B={batch_size}$ · $\eta={decimal_label(eta)}$",
        )
    ax.set_yscale("log")
    ax.set_xlabel("training examples processed (data exposure)")
    ax.set_ylabel(r"full-loss gap $\mathcal{L}(\boldsymbol{\theta})-\mathcal{L}_{\min}$")
    ax.set_title(
        rf"matched nominal $\sum\eta={decimal_label(MATCHED_EPOCH_STEP)}$ per epoch · seed 5"
    )
    ax.legend(frameon=False, ncol=1, fontsize=9.6, loc="upper right")
    clean(ax, grid=True)
    fig.tight_layout()
    save(fig, "story_batch_loss")


def f_estimator_spread():
    per_example = np.array([line_grad(THETA0, [i]) for i in range(4)])
    pair_means = np.array([line_grad(THETA0, pair) for pair in combinations(range(4), 2)])
    full = line_grad(THETA0)
    clouds = [per_example, pair_means, full[None, :]]
    batch_sizes = [1, 2, 4]
    colors = [BLUE, TEAL, INK]
    fig, axes = plt.subplots(1, 3, figsize=(10.9, 3.35), sharex=True, sharey=True)
    for ax, values, batch_size, color in zip(axes, clouds, batch_sizes, colors):
        centroid = values.mean(axis=0)
        trace_cov = np.mean(np.sum((values - centroid) ** 2, axis=1))
        assert np.allclose(centroid, full)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.quiver(
            np.zeros(len(values)), np.zeros(len(values)), values[:, 0], values[:, 1],
            angles="xy", scale_units="xy", scale=1, color=color, alpha=0.42, width=0.008,
        )
        ax.scatter(values[:, 0], values[:, 1], s=72, color=color, alpha=0.84, edgecolor="white", linewidth=0.7)
        ax.scatter(*centroid, marker="X", s=125, color=RED, edgecolor="white", linewidth=0.7, zorder=6)
        ax.set_title(rf"$B={batch_size}$ · $\mathrm{{tr}}(\mathrm{{Cov}})={trace_cov:.2f}$")
        ax.set(xlabel=r"bias component $\hat g_b$", xlim=(-7.0, 5.0), ylim=(-13.0, 1.8))
        ax.set_aspect("equal")
        clean(ax)
    axes[0].set_ylabel(r"slope component $\hat g_w$")
    fig.text(0.5, 0.01, r"red X: vector centroid $=(-0.75,-4.5)$ in every panel", ha="center", color=RED, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save(fig, "story_estimator_spread")


# ---------------------------------------------------------------------------
# Example B: a real ill-conditioned linear regression
# X^T X / N = diag(1, 100). In the original parameter coordinates,
# L(theta) = 1/2 (theta_1 - 1)^2 + 50 (theta_2 + 0.5)^2.
# ---------------------------------------------------------------------------
X2 = np.array([[-1.0, -10.0], [-1.0, 10.0], [1.0, -10.0], [1.0, 10.0]])
THETA2_STAR = np.array([1.0, -0.5])
Y2 = X2 @ THETA2_STAR
THETA2_0 = np.array([-1.4, 0.35])


def ravine_loss(theta):
    """Full empirical loss in the model's native parameter coordinates."""
    residual = X2 @ theta - Y2
    return 0.5 * np.mean(residual**2)


def ravine_grad(theta):
    """Exact full-batch gradient with respect to (theta_1, theta_2)."""
    residual = X2 @ theta - Y2
    return X2.T @ residual / len(X2)


assert np.allclose(X2.T @ X2 / len(X2), np.diag([1.0, 100.0]))
assert np.isclose(
    ravine_loss(THETA2_0),
    0.5 * (THETA2_0[0] - THETA2_STAR[0]) ** 2
    + 50 * (THETA2_0[1] - THETA2_STAR[1]) ** 2,
)


def ravine_grid(xlim=(-1.65, 1.45), ylim=(-1.52, 0.52), n=260):
    theta1 = np.linspace(*xlim, n)
    theta2 = np.linspace(*ylim, n)
    THETA1, THETA2 = np.meshgrid(theta1, theta2)
    error1 = THETA1 - THETA2_STAR[0]
    error2 = THETA2 - THETA2_STAR[1]
    return THETA1, THETA2, 0.5 * error1**2 + 50 * error2**2


def gd_path(eta, steps):
    theta = THETA2_0.copy()
    out = [theta.copy()]
    for _ in range(steps):
        theta = theta - eta * ravine_grad(theta)
        out.append(theta.copy())
    return np.asarray(out)


def momentum_run(eta, beta, steps):
    theta = THETA2_0.copy()
    memory = np.zeros(2)
    path = [theta.copy()]
    memories = [memory.copy()]
    losses = [ravine_loss(theta)]
    for _ in range(steps):
        gradient = ravine_grad(theta)
        memory = beta * memory + (1 - beta) * gradient
        theta = theta - eta * memory
        path.append(theta.copy())
        memories.append(memory.copy())
        losses.append(ravine_loss(theta))
    return np.asarray(path), np.asarray(memories), np.asarray(losses)


def momentum_path(eta, beta, steps):
    return momentum_run(eta, beta, steps)[0]


def rmsprop_run(eta, rho, steps, eps=1e-8):
    theta = THETA2_0.copy()
    scale = np.zeros(2)
    path = [theta.copy()]
    scales = [scale.copy()]
    normalized = []
    losses = [ravine_loss(theta)]
    for _ in range(steps):
        gradient = ravine_grad(theta)
        scale = rho * scale + (1 - rho) * gradient**2
        normalized_gradient = gradient / (np.sqrt(scale) + eps)
        theta = theta - eta * normalized_gradient
        path.append(theta.copy())
        scales.append(scale.copy())
        normalized.append(normalized_gradient.copy())
        losses.append(ravine_loss(theta))
    return np.asarray(path), np.asarray(scales), np.asarray(normalized), np.asarray(losses)


def rmsprop_path(eta, rho, steps, eps=1e-8):
    return rmsprop_run(eta, rho, steps, eps)[0]


def adam_path(eta, beta1, beta2, steps, eps=1e-8):
    theta = THETA2_0.copy()
    first = np.zeros(2)
    second = np.zeros(2)
    out = [theta.copy()]
    for t in range(1, steps + 1):
        gradient = ravine_grad(theta)
        first = beta1 * first + (1 - beta1) * gradient
        second = beta2 * second + (1 - beta2) * gradient**2
        first_hat = first / (1 - beta1**t)
        second_hat = second / (1 - beta2**t)
        theta = theta - eta * first_hat / (np.sqrt(second_hat) + eps)
        out.append(theta.copy())
    return np.asarray(out)


def draw_ravine(ax):
    THETA1, THETA2, loss = ravine_grid()
    levels = [0.05, 0.15, 0.4, 0.9, 1.8, 3.5, 7, 14, 25, 40]
    ax.contour(THETA1, THETA2, loss, levels=levels, colors=[TEAL], linewidths=0.75, alpha=0.62)
    ax.scatter(*THETA2_STAR, marker="*", s=130, color=RED, zorder=7)
    ax.set_xlim(-1.65, 1.45)
    ax.set_ylim(-1.52, 0.52)
    ax.set_xlabel(r"parameter $\theta_1$")
    # Equal data aspect preserves the Euclidean geometry used by the gradient.
    ax.set_aspect("equal")
    clean(ax)


def f_ravine_geometry():
    """Define the ravine vocabulary with one uncluttered contour plot."""
    fig, ax = plt.subplots(figsize=(8.7, 4.45))
    draw_ravine(ax)
    ax.scatter(*THETA2_0, s=78, color=INK, edgecolor="white", linewidth=1.0, zorder=9)
    ax.annotate(
        r"start  $\boldsymbol{\theta}_0=(-1.4,.35)$",
        xy=THETA2_0,
        xytext=(-1.55, 0.08),
        fontsize=11.0,
        color=INK,
        weight="bold",
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.2),
    )
    ax.annotate(
        r"optimum  $\boldsymbol{\theta}^*=(1,-.5)$",
        xy=THETA2_STAR,
        xytext=(0.22, -0.72),
        fontsize=11.0,
        color=RED,
        weight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
    )
    ax.annotate(
        "",
        xy=(0.45, -1.06),
        xytext=(-1.20, -1.06),
        arrowprops=dict(arrowstyle="<->", color=BLUE, lw=2.4),
    )
    ax.text(
        -0.37,
        -1.20,
        "along the valley\nshallow: loss changes slowly",
        ha="center",
        va="top",
        fontsize=11.2,
        color=BLUE,
        weight="bold",
    )
    ax.annotate(
        "",
        xy=(0.68, 0.15),
        xytext=(0.68, -0.98),
        arrowprops=dict(arrowstyle="<->", color=ACC, lw=2.4),
    )
    ax.text(
        0.82,
        -0.02,
        "across the valley\nsteep: loss changes quickly",
        ha="left",
        va="center",
        fontsize=11.2,
        color=ACC,
        weight="bold",
    )
    ax.set_title("a ravine is a long, narrow low-loss valley", fontsize=14.0, weight="bold", pad=8)
    ax.set_xlabel(r"parameter  $\theta_1$")
    ax.set_ylabel(r"parameter  $\theta_2$")

    fig.text(
        0.5,
        0.012,
        r"computed from the four displayed rows  $\cdot$  each contour joins points with the same full-batch loss",
        ha="center",
        color=MUTED,
        fontsize=10.8,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save(fig, "story_ravine_geometry")


def f_ravine_first_step():
    """Show one correct step overshooting across the ravine."""
    eta = 0.019
    gradient = ravine_grad(THETA2_0)
    delta = -eta * gradient
    theta_after_1 = THETA2_0 + delta
    assert np.allclose(gradient, [-2.4, 85.0])
    assert np.allclose(delta, [0.0456, -1.615])
    assert np.allclose(theta_after_1, [-1.3544, -1.265])

    fig, ax = plt.subplots(figsize=(7.8, 4.25))
    draw_ravine(ax)
    ax.scatter(*THETA2_0, s=75, color=INK, edgecolor="white", linewidth=1.0, zorder=9)
    ax.scatter(*theta_after_1, s=75, color=ACC, edgecolor="white", linewidth=1.0, zorder=9)
    ax.annotate(
        "",
        xy=theta_after_1,
        xytext=THETA2_0,
        arrowprops=dict(arrowstyle="-|>", color=ACC, lw=3.0, mutation_scale=17),
        zorder=8,
    )
    ax.annotate(r"start  $\boldsymbol{\theta}_0$", xy=THETA2_0, xytext=(-1.48, 0.08), color=INK,
                fontsize=11.0, weight="bold", arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax.annotate(r"after one step  $\boldsymbol{\theta}_1$", xy=theta_after_1, xytext=(-0.78, -1.36), color=ACC,
                fontsize=11.0, weight="bold", arrowprops=dict(arrowstyle="->", color=ACC, lw=1.1))
    ax.axhline(THETA2_STAR[1], color=MUTED, lw=0.9, linestyle="--", alpha=0.8)
    ax.text(-1.57, -0.45, r"valley centre  $\theta_2^*=-.5$", color=MUTED, fontsize=10.5, va="bottom")
    ax.set_title("one exact-gradient step crosses to the other side", fontsize=13.6, weight="bold", pad=8)
    ax.set_xlabel(r"parameter  $\theta_1$")
    ax.set_ylabel(r"parameter  $\theta_2$")
    fig.tight_layout()
    save(fig, "story_ravine_first_step")


def f_ravine_learning_rates():
    specs = [
        (0.005, 32, r"small rate $\eta=.005$" + "\n" + "little oscillation, little progress", BLUE),
        (0.019, 32, r"larger rate $\eta=.019$" + "\n" + "more progress, repeated crossings", ACC),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharex=True, sharey=True)
    for ax, (eta, steps, title, color) in zip(axes, specs):
        draw_ravine(ax)
        path = gd_path(eta, steps)
        # Keep the visible band centred on the optimum in native parameter space.
        visible = np.abs(path[:, 1] - THETA2_STAR[1]) <= 1.03
        first_hidden = np.where(~visible)[0]
        stop_at = first_hidden[0] if len(first_hidden) else len(path)
        ax.plot(path[:stop_at, 0], path[:stop_at, 1], "o-", color=color, markersize=2.9, linewidth=1.7)
        ax.scatter(*THETA2_0, s=38, color=INK, zorder=7)
        ax.set_title(title, fontsize=10.8)
        ax.set_xlabel(r"parameter $\theta_1$")
    axes[0].set_ylabel(r"parameter $\theta_2$")
    fig.text(
        0.5, 0.01,
        r"computed full-batch GD $\cdot$ same objective and start $\cdot$ only the learning rate changes",
        ha="center", color=MUTED, fontsize=10.2,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save(fig, "story_ravine_learning_rates")


def f_feature_scaling_fix():
    """Show that standardizing the constructed feature removes this ravine."""
    raw = gd_path(0.019, 10)
    # With z_2 = x_2 / 10, the corresponding coefficient is phi_2 = 10 theta_2.
    # Transform both the start and optimum, so both panels represent the same model.
    scaled_star = np.array([THETA2_STAR[0], 10.0 * THETA2_STAR[1]])
    scaled_start = np.array([THETA2_0[0], 10.0 * THETA2_0[1]])
    scaled_eta = 0.35
    scaled = [scaled_start.copy()]
    phi = scaled_start.copy()
    for _ in range(10):
        phi = phi - scaled_eta * (phi - scaled_star)
        scaled.append(phi.copy())
    scaled = np.asarray(scaled)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    draw_ravine(axes[0])
    axes[0].plot(raw[:, 0], raw[:, 1], "o-", color=ACC, lw=1.8, ms=2.7)
    axes[0].scatter(*THETA2_0, color=INK, s=42, zorder=8)
    axes[0].set_title(r"raw $x_2$: curvature $(1,100)$, $\eta=.019$", fontsize=11.4, weight="bold")
    axes[0].set_xlabel(r"parameter $\theta_1$")
    axes[0].set_ylabel(r"parameter $\theta_2$")

    phi1 = np.linspace(-8.3, 10.3, 320)
    phi2 = np.linspace(-14.3, 4.3, 320)
    S1, S2 = np.meshgrid(phi1, phi2)
    round_loss = 0.5 * ((S1 - scaled_star[0]) ** 2 + (S2 - scaled_star[1]) ** 2)
    levels = [0.5, 2, 5, 10, 20, 40, 80]
    axes[1].contour(S1, S2, round_loss, levels=levels, colors=[TEAL], linewidths=0.8, alpha=0.65)
    axes[1].plot(scaled[:, 0], scaled[:, 1], "o-", color=BLUE, lw=2.2, ms=3.0)
    axes[1].scatter(*scaled_start, color=INK, s=42, zorder=8)
    axes[1].scatter(*scaled_star, marker="*", s=130, color=RED, zorder=7)
    axes[1].set_xlim(-8.2, 10.2)
    axes[1].set_ylim(-14.2, 4.2)
    axes[1].set_aspect("equal")
    axes[1].set_title(r"standardized $z_2$: curvature $(1,1)$, $\eta=.35$", fontsize=11.4, weight="bold")
    axes[1].set_xlabel(r"parameter $\phi_1=\theta_1$")
    axes[1].set_ylabel(r"parameter $\phi_2=10\theta_2$")
    clean(axes[1])

    fig.text(
        0.5,
        0.01,
        r"10 updates each $\cdot$ same starting predictor: $\theta_0=(-1.4,.35)$ maps to $\phi_0=(-1.4,3.5)$",
        ha="center",
        color=MUTED,
        fontsize=10.1,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save(fig, "story_feature_scaling_fix")


def f_ravine_rate_traces():
    eta = 0.019
    path = gd_path(eta, 28)
    t = np.arange(len(path))
    losses = np.array([ravine_loss(theta) for theta in path])
    exact_theta1 = THETA2_STAR[0] + (THETA2_0[0] - THETA2_STAR[0]) * (1 - eta) ** t
    exact_theta2 = THETA2_STAR[1] + (THETA2_0[1] - THETA2_STAR[1]) * (1 - 100 * eta) ** t
    assert np.allclose(path[:, 0], exact_theta1)
    assert np.allclose(path[:, 1], exact_theta2)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.65))
    axes[0].axhline(THETA2_STAR[0], color=MUTED, lw=0.8)
    axes[0].plot(t, path[:, 0], "o-", color=BLUE, markersize=3.5)
    axes[0].set(
        title=r"first parameter approaches $\theta_1^*=1$",
        xlabel="update $t$",
        ylabel=r"parameter $\theta_1$",
    )
    axes[0].text(0.04, 0.08, "same side; 98.1% remains each step", transform=axes[0].transAxes,
                 fontsize=10.2, color=BLUE, weight="bold")

    axes[1].axhline(THETA2_STAR[1], color=MUTED, lw=0.8)
    axes[1].plot(t, path[:, 1], "o-", color=ACC, markersize=3.5)
    axes[1].set(
        title=r"second parameter crosses $\theta_2^*=-.5$",
        xlabel="update $t$",
        ylabel=r"parameter $\theta_2$",
    )
    axes[1].text(0.04, 0.08, "crosses sides; 90% remains each step", transform=axes[1].transAxes,
                 fontsize=10.2, color=ACC, weight="bold")

    axes[2].semilogy(t, losses, "o-", color=TEAL, markersize=3.5)
    axes[2].set(
        title="full loss still decreases",
        xlabel="update $t$",
        ylabel="full loss (log scale)",
    )
    axes[2].text(0.04, 0.08, f"{losses[0]:.2f} → {losses[-1]:.2f}", transform=axes[2].transAxes,
                 fontsize=10.5, color=TEAL, weight="bold")
    for ax in axes:
        clean(ax, grid=True)
    fig.text(
        0.5,
        0.015,
        r"one near-limit run, $\eta=.019$ $\cdot$ exact full-batch gradient at every update",
        ha="center",
        color=MUTED,
        fontsize=10.6,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save(fig, "story_ravine_rate_traces")


def f_ravine_gradients():
    path = gd_path(0.019, 14)
    gradients = np.array([ravine_grad(theta) for theta in path[:-1]])
    t = np.arange(len(gradients))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.05), sharex=True)
    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].plot(t, gradients[:, 0], "o-", color=BLUE, markersize=4)
    axes[0].set_title(r"first-parameter gradient: same sign", fontsize=12.2)
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].plot(t, gradients[:, 1], "o-", color=ACC, markersize=4)
    axes[1].set_title(r"second-parameter gradient: sign flips", fontsize=12.2)
    for ax in axes:
        ax.set_xlabel("update $t$")
        clean(ax, grid=True)
    axes[0].set_ylabel("gradient component")
    fig.text(
        0.5, 0.01,
        r"full-batch gradients along the actual $\eta=.019$ path · $\mathbf{g}_t=(\theta_{t,1}-1,\ 100(\theta_{t,2}+.5))$",
        ha="center", color=MUTED, fontsize=10.3,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "story_ravine_gradients")


def ewma(values, beta):
    state = 0.0
    out = []
    for value in values:
        state = beta * state + (1 - beta) * value
        out.append(state)
    return np.asarray(out)


def ewma_initialized_at_first(values, beta):
    """EWMA used only for a clean smoothing demonstration, with m_1=x_1."""
    values = np.asarray(values, dtype=float)
    state = values[0]
    out = [state]
    for value in values[1:]:
        state = beta * state + (1 - beta) * value
        out.append(state)
    return np.asarray(out)


def temperature_signal():
    """Deterministic noisy signal used to teach smoothing before gradients."""
    day = np.arange(1, 121)
    trend = 25.0 + 7.5 * np.sin(2 * np.pi * (day - 24) / 120)
    rng = np.random.default_rng(12)
    values = trend + rng.normal(0.0, 4.4, size=len(day))
    return day, trend, values


def draw_temperature_raw(ax):
    day, _trend, values = temperature_signal()
    ax.scatter(day, values, s=15, color=MUTED, alpha=0.72, label="daily reading")
    ax.plot(day, values, color=MUTED, lw=0.65, alpha=0.35)
    ax.set(xlim=(1, 120), ylim=(9, 42), xlabel="day", ylabel="temperature (°C)")
    clean(ax, grid=True)


def f_ewma_raw():
    fig, ax = plt.subplots(figsize=(8.8, 3.7))
    draw_temperature_raw(ax)
    ax.set_title("daily readings are noisy; the slower pattern is hard to see", fontsize=13.5, weight="bold")
    ax.annotate(
        "a one-day jump may be noise",
        xy=(47, temperature_signal()[2][46]),
        xytext=(60, 39),
        fontsize=11.0,
        color=INK,
        weight="bold",
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.2),
    )
    fig.tight_layout()
    save(fig, "story_ewma_raw")


def make_ewma_beta_figure(beta, color, name):
    day, trend, values = temperature_signal()
    smoothed = ewma_initialized_at_first(values, beta)
    assert np.isclose(smoothed[0], values[0])
    fig, ax = plt.subplots(figsize=(8.8, 3.7))
    ax.scatter(day, values, s=13, color=MUTED, alpha=0.40, label="daily readings")
    ax.plot(day, trend, color=INK, lw=1.5, ls="--", alpha=0.75, label="slow pattern used to generate the data")
    ax.plot(day, smoothed, color=color, lw=3.0, label=rf"EWMA  $\beta={beta:g}$")
    rough = int(round(1 / (1 - beta)))
    ax.text(
        0.025,
        0.94,
        rf"keep {100*beta:.0f}% of the previous summary; add {100*(1-beta):.0f}% of today"
        + "\n"
        + rf"rough memory: {rough} readings",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=color,
        fontsize=11.0,
        weight="bold",
        bbox=dict(boxstyle="round,pad=.28", facecolor="white", edgecolor=color, alpha=.94),
    )
    ax.set(xlim=(1, 120), ylim=(9, 42), xlabel="day", ylabel="temperature (°C)")
    ax.legend(frameon=False, ncol=3, fontsize=9.6, loc="lower center")
    clean(ax, grid=True)
    fig.tight_layout()
    save(fig, name)


def f_ewma_beta05():
    make_ewma_beta_figure(0.5, ACC, "story_ewma_beta05")


def f_ewma_beta09():
    make_ewma_beta_figure(0.9, TEAL, "story_ewma_beta09")


def f_ewma_beta098():
    make_ewma_beta_figure(0.98, PURPLE, "story_ewma_beta098")


def f_ewma_bias():
    values = np.array([29, 31, 30, 32, 36, 35, 34, 38, 37, 39, 40, 38], dtype=float)
    beta = 0.9
    t = np.arange(1, len(values) + 1)
    raw = ewma(values, beta)
    corrected = raw / (1 - beta**t)
    assert np.isclose(raw[0], (1 - beta) * values[0])
    assert np.isclose(corrected[0], values[0])

    fig, ax = plt.subplots(figsize=(7.7, 3.25))
    ax.plot(t, values, "o--", color=MUTED, lw=1.5, markersize=5, label="daily readings")
    ax.plot(t, raw, "o-", color=ACC, markersize=3.5, label="raw zero-start state")
    ax.plot(t, corrected, "o-", color=TEAL, markersize=3.5, label="missing-mass corrected")
    ax.annotate(
        rf"day 1: ${raw[0]:.1f}\rightarrow{corrected[0]:.0f}$",
        xy=(1, corrected[0]), xytext=(2.2, 20), color=TEAL, fontsize=10.5,
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2),
    )
    ax.set(xlabel="day $t$", ylabel="temperature (°C)", title="starting from zero pulls the early estimate down")
    ax.legend(frameon=False, ncol=3, fontsize=9.2, loc="lower right")
    clean(ax, grid=True)
    fig.tight_layout()
    save(fig, "story_ewma_bias")


def f_ewma_weights():
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.15), sharey=True)
    for ax, beta, color in zip(axes, [0.5, 0.9], [ACC, TEAL]):
        last_age = int(np.ceil(np.log(0.01) / np.log(beta)) - 1)
        age = np.arange(0, last_age + 1)
        weights = (1 - beta) * beta**age
        shown_mass = weights.sum()
        assert shown_mass >= 0.99
        ax.bar(age, weights, color=color, alpha=0.88)
        label = "short memory" if beta == 0.5 else "long memory"
        ax.set_title(rf"{label} · $\beta={beta}$", fontsize=11.2)
        ax.set_xlabel("observation age $k$ · 0 = current")
        rough = int(round(1 / (1 - beta)))
        ax.text(0.98, 0.92, rf"rough timescale: ${rough}$ readings",
                transform=ax.transAxes, ha="right", va="top", color=color,
                fontsize=10.0, weight="bold")
        clean(ax, grid=True)
    axes[0].set_ylabel(r"coefficient $(1-\beta)\beta^k$")
    axes[1].set_ylabel(r"coefficient $(1-\beta)\beta^k$")
    fig.tight_layout()
    save(fig, "story_ewma_weights")


def f_gradient_filter():
    """Show, on Example B, exactly how memory addresses ravine oscillation."""
    beta = 0.8
    eta = 0.095
    gradient_0 = ravine_grad(THETA2_0)
    memory_0 = np.zeros(2)
    memory_1 = beta * memory_0 + (1 - beta) * gradient_0
    theta_after_1 = THETA2_0 - eta * memory_1
    gradient_1 = ravine_grad(theta_after_1)
    old_contribution = beta * memory_1
    new_contribution = (1 - beta) * gradient_1
    memory_2 = old_contribution + new_contribution
    theta_after_2 = theta_after_1 - eta * memory_2

    assert np.allclose(gradient_0, [-2.4, 85.0])
    assert np.allclose(memory_1, [-0.48, 17.0])
    assert np.allclose(theta_after_1, [-1.3544, -1.265])
    assert np.allclose(gradient_1, [-2.3544, -76.5])
    assert np.allclose(old_contribution, [-0.384, 13.6])
    assert np.allclose(new_contribution, [-0.47088, -15.3])
    assert np.allclose(memory_2, [-0.85488, -1.7])
    assert np.allclose(theta_after_2, [-1.2731864, -1.1035])

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 3.85),
        gridspec_kw={"width_ratios": [1.05, 1.28, 0.88]},
    )

    # 1. Repeat the concrete failure that motivates memory.
    ax = axes[0]
    draw_ravine(ax)
    ax.scatter(*THETA2_0, s=58, color=INK, edgecolor="white", linewidth=0.9, zorder=9)
    ax.scatter(*theta_after_1, s=58, color=ACC, edgecolor="white", linewidth=0.9, zorder=9)
    ax.annotate(
        "",
        xy=theta_after_1,
        xytext=THETA2_0,
        arrowprops=dict(arrowstyle="-|>", color=ACC, lw=2.8, mutation_scale=15),
        zorder=8,
    )
    ax.text(
        0.34,
        0.97,
        r"$\mathbf{g}_0=(-2.4,\ 85)$" "\n" r"$\mathbf{m}_1=.2\mathbf{g}_0=(-.48,\ 17)$",
        transform=ax.transAxes,
        va="top",
        fontsize=9.7,
        color=INK,
        bbox=dict(boxstyle="round,pad=.25", facecolor="white", edgecolor="none"),
        zorder=10,
    )
    ax.text(THETA2_0[0] + 0.08, THETA2_0[1] - 0.03, "start", color=INK, fontsize=9.8,
            weight="bold", va="top", zorder=10)
    ax.text(theta_after_1[0] + 0.08, theta_after_1[1] - 0.03, "after 1", color=ACC,
            fontsize=9.8, weight="bold", va="top", zorder=10)
    ax.set_ylabel(r"parameter  $\theta_2$")
    ax.set_title("1 · the same first step crosses", fontsize=12.5, weight="bold", pad=7)

    # 2. Draw the two component calculations as arrows, not a dense table.
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("2 · memory combines old + new", fontsize=12.5, weight="bold", pad=7)
    ax.text(
        0.5,
        0.94,
        r"after crossing:  $\mathbf{g}_1=(-2.3544,\ -76.5)$",
        ha="center",
        va="center",
        fontsize=10.6,
        color=INK,
    )

    along_box = mpl.patches.FancyBboxPatch(
        (0.025, 0.49), 0.95, 0.37,
        boxstyle="round,pad=0.018,rounding_size=0.022",
        linewidth=1.2, edgecolor=BLUE, facecolor="#EFF5FB",
    )
    across_box = mpl.patches.FancyBboxPatch(
        (0.025, 0.05), 0.95, 0.34,
        boxstyle="round,pad=0.018,rounding_size=0.022",
        linewidth=1.2, edgecolor=ACC, facecolor="#FFF4E8",
    )
    ax.add_patch(along_box)
    ax.add_patch(across_box)

    ax.text(0.07, 0.81, r"first parameter  $(\theta_1)$", color=BLUE, fontsize=10.7, weight="bold")
    ax.annotate("", xy=(0.17, 0.67), xytext=(0.45, 0.67),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.4, mutation_scale=13))
    ax.annotate("", xy=(0.17, 0.57), xytext=(0.45, 0.57),
                arrowprops=dict(arrowstyle="-|>", color=ACC, lw=2.4, mutation_scale=13))
    ax.text(0.48, 0.67, r"$.8m_{1,1}=-.384$", va="center", fontsize=10.0, color=TEAL)
    ax.text(0.48, 0.57, r"$.2g_{1,1}=-.471$", va="center", fontsize=10.0, color=ACC)
    ax.text(0.50, 0.50, r"same sign $\Rightarrow\ m_{2,1}=-.855$", ha="center",
            va="bottom", fontsize=10.4, color=BLUE, weight="bold")

    ax.text(0.07, 0.34, r"second parameter  $(\theta_2)$", color=ACC, fontsize=10.7, weight="bold")
    ax.annotate("", xy=(0.29, 0.30), xytext=(0.29, 0.13),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.4, mutation_scale=13))
    ax.annotate("", xy=(0.43, 0.13), xytext=(0.43, 0.30),
                arrowprops=dict(arrowstyle="-|>", color=ACC, lw=2.4, mutation_scale=13))
    ax.text(0.49, 0.27, r"$.8m_{1,2}=+13.6$", va="center", fontsize=10.0, color=TEAL)
    ax.text(0.49, 0.17, r"$.2g_{1,2}=-15.3$", va="center", fontsize=10.0, color=ACC)
    ax.text(0.50, 0.065, r"opposite signs $\Rightarrow\ m_{2,2}=-1.7$", ha="center",
            va="bottom", fontsize=10.4, color=RED, weight="bold")

    # 3. Make the answer to the motivating problem visually explicit.
    ax = axes[2]
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.axis("off")
    ax.set_title("3 · the next second-parameter move is much smaller", fontsize=12.5, weight="bold", pad=7)
    ax.add_patch(
        mpl.patches.Rectangle((-1.0, -0.15), 2.0, 0.30, facecolor="#EAF5F5",
                              edgecolor="none", zorder=0)
    )
    ax.annotate("", xy=(0.92, 0), xytext=(-0.92, 0),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
    ax.annotate("", xy=(0, 0.92), xytext=(0, -0.92),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
    ax.scatter([0], [0], s=66, color=ACC, edgecolor="white", linewidth=0.9, zorder=5)
    ax.annotate(
        "",
        xy=(0.62, 0.34),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=4.0, mutation_scale=17),
    )
    ax.text(0.34, 0.25, r"update  $-\eta\mathbf{m}_2$", ha="center", color=BLUE,
            fontsize=10.5, weight="bold")
    ax.text(-0.10, 0.62, r"$\theta_2$" "\nupdate", ha="right", color=MUTED, fontsize=10.0)
    ax.text(0.72, -0.16, r"$\theta_1$ update", ha="center", va="top", color=MUTED, fontsize=10.0)
    ax.text(0.0, -0.46, r"$\mathbf{m}_2=(-.855,\ -1.7)$", ha="center", fontsize=12.0,
            color=INK, weight="bold")
    ax.text(0.0, -0.68, "second-parameter move: +.162\nplain GD would move +1.454", ha="center",
            va="top", fontsize=10.8, color=TEAL, weight="bold")

    fig.text(
        0.5,
        0.008,
        r"computed from Example B · first displacement matches GD $\eta=.019$ · normalized memory $\beta=.8$, $\eta=.095$",
        ha="center",
        color=MUTED,
        fontsize=10.1,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.2)
    save(fig, "story_gradient_filter")


def f_momentum_ravine():
    steps = 55
    gd_eta = 0.019
    beta = 0.80
    mom_eta = 0.095
    pytorch_lr = mom_eta * (1 - beta)
    gd = gd_path(gd_eta, steps)
    momentum, memory, momentum_loss = momentum_run(mom_eta, beta, steps)
    gd_loss = np.array([ravine_loss(theta) for theta in gd])

    transverse_gd = np.abs(np.diff(gd[:, 1])).sum()
    transverse_momentum = np.abs(np.diff(momentum[:, 1])).sum()
    gd_error2 = gd[:, 1] - THETA2_STAR[1]
    momentum_error2 = momentum[:, 1] - THETA2_STAR[1]
    crossings_gd = np.count_nonzero(np.signbit(gd_error2[1:]) != np.signbit(gd_error2[:-1]))
    crossings_momentum = np.count_nonzero(
        np.signbit(momentum_error2[1:]) != np.signbit(momentum_error2[:-1])
    )
    assert transverse_momentum < transverse_gd
    assert crossings_momentum < crossings_gd
    assert momentum_loss[-1] < gd_loss[-1]
    assert np.isclose(pytorch_lr, gd_eta)
    assert np.isclose(ravine_grad(THETA2_0)[1], 85.0)
    assert np.isclose(memory[1, 1], 17.0)
    assert np.isclose(momentum[1, 1], -1.265)
    assert np.isclose(ravine_grad(momentum[1])[1], -76.5)
    assert np.isclose(memory[2, 1], -1.7)
    assert np.isclose(momentum[2, 1], -1.1035)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.05), sharex=True, sharey=True)
    draw_ravine(axes[0])
    axes[0].plot(gd[:, 0], gd[:, 1], "o-", color=ACC, linewidth=1.6, markersize=2.3, label=rf"GD · $\eta={gd_eta}$")
    axes[0].scatter(*THETA2_0, color=INK, s=38, zorder=7)
    axes[0].set(ylabel=r"parameter $\theta_2$",
                xlabel=r"parameter $\theta_1$",
                title=rf"plain GD · {crossings_gd} crossings" "\n" rf"$L_{{55}}={gd_loss[-1]:.3f}$")
    axes[0].legend(frameon=False, fontsize=9.8, loc="lower right")

    draw_ravine(axes[1])
    axes[1].plot(
        momentum[:, 0], momentum[:, 1], "o-", color=TEAL, linewidth=1.8, markersize=2.3,
        label=rf"momentum · $\beta={beta}$",
    )
    axes[1].scatter(*THETA2_0, color=INK, s=38, zorder=7)
    axes[1].set(ylabel="", xlabel=r"parameter $\theta_1$",
                title=rf"momentum · {crossings_momentum} crossings" "\n" rf"$L_{{55}}={momentum_loss[-1]:.1e}$")
    axes[1].legend(frameon=False, fontsize=9.8, loc="lower right")

    fig.text(
        0.5, 0.01,
        rf"55 exact full-batch gradient evaluations each · same start and first displacement · GD $\eta=.019$ · normalized momentum $\eta=.095,\ \beta=.8$",
        ha="center", color=MUTED, fontsize=10.3,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "story_momentum_ravine")


def f_momentum_loss():
    """Make momentum's improvement visible on the full empirical loss."""
    steps = 55
    gd = gd_path(0.019, steps)
    momentum, _memory, momentum_loss = momentum_run(0.095, 0.8, steps)
    gd_loss = np.array([ravine_loss(theta) for theta in gd])
    update = np.arange(steps + 1)
    threshold = 0.1
    permanently_below = int(next(
        i for i in range(len(momentum_loss)) if np.all(momentum_loss[i:] < threshold)
    ))
    rise_updates = np.flatnonzero(np.diff(momentum_loss) > 0) + 1
    overshoot_update = 3
    assert permanently_below == 26
    assert not np.any(gd_loss < threshold)
    assert np.all(np.diff(gd_loss) < 0)
    assert overshoot_update in rise_updates

    fig, ax = plt.subplots(figsize=(8.6, 3.75))
    ax.semilogy(update, gd_loss, color=ACC, lw=2.6, label="plain GD · monotone here")
    ax.semilogy(update, momentum_loss, color=TEAL, lw=3.0,
                label=r"momentum  $\beta=.8$ · stored motion")
    ax.scatter([overshoot_update], [momentum_loss[overshoot_update]],
               s=54, color=TEAL, zorder=7)
    ax.annotate(
        rf"stored motion overshoots:  $L_2={momentum_loss[2]:.1f}\rightarrow "
        rf"L_3={momentum_loss[3]:.1f}$",
        xy=(overshoot_update, momentum_loss[overshoot_update]),
        xytext=(8.0, 17),
        color=TEAL,
        fontsize=10.5,
        weight="bold",
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.3),
    )
    ax.text(54, gd_loss[-1] * 1.15, rf"GD: ${gd_loss[-1]:.3f}$", color=ACC,
            ha="right", va="bottom", fontsize=11.0, weight="bold")
    ax.text(54, momentum_loss[-1] * 1.25, rf"momentum: ${momentum_loss[-1]:.1e}$", color=TEAL,
            ha="right", va="bottom", fontsize=11.0, weight="bold")
    ax.set(
        xlabel="exact full-batch gradient evaluations",
        ylabel="full empirical loss (log scale)",
        title="full-batch loss after every update · no sampling noise",
        xlim=(0, 55),
    )
    ax.legend(frameon=False, loc="upper right")
    clean(ax, grid=True)
    fig.tight_layout()
    save(fig, "story_momentum_loss")


def f_rmsprop_ruler():
    """Turn Example B's unequal gradient coordinates into two local rulers."""
    rho = 0.90
    gradient = ravine_grad(THETA2_0)
    raw_magnitude = np.abs(gradient)
    second_moment = (1 - rho) * gradient**2
    ruler = np.sqrt(second_moment)
    # Omit epsilon only in this arithmetic illustration, as is customary when
    # explaining the mechanism. The actual optimizer below uses epsilon=1e-8.
    rescaled_magnitude = raw_magnitude / ruler

    assert np.allclose(gradient, [-2.4, 85.0])
    assert np.allclose(second_moment, [0.576, 722.5])
    assert np.allclose(ruler, [np.sqrt(0.576), np.sqrt(722.5)])
    assert np.allclose(rescaled_magnitude, [np.sqrt(10), np.sqrt(10)])

    labels = [r"parameter $\theta_1$", r"parameter $\theta_2$"]
    colors = [BLUE, ACC]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.15))

    bars = axes[0].bar(labels, raw_magnitude, color=colors, width=0.58, alpha=0.92)
    axes[0].bar_label(bars, labels=["2.4", "85"], padding=4, fontsize=12.0, weight="bold")
    axes[0].set_ylim(0, 96)
    axes[0].set_ylabel(r"current magnitude $|g_{0,j}|$")
    axes[0].set_title("1 · one gradient\nvery different coordinate sizes", weight="bold", pad=9)
    axes[0].text(
        0.04, 0.81, r"$|g_{0,2}|/|g_{0,1}|=35.4$",
        transform=axes[0].transAxes, color=RED, fontsize=11.0, weight="bold",
    )
    clean(axes[0], grid=True)

    bars = axes[1].bar(labels, ruler, color=colors, width=0.58, alpha=0.92)
    axes[1].bar_label(
        bars,
        labels=[r"$\sqrt{0.576}=0.759$", r"$\sqrt{722.5}=26.88$"],
        padding=4,
        fontsize=10.8,
        weight="bold",
    )
    axes[1].set_ylim(0, 31)
    axes[1].set_ylabel(r"local ruler $\sqrt{s_{1,j}}$")
    axes[1].set_title("2 · remember squared magnitude\nwith one ruler per coordinate", weight="bold", pad=9)
    axes[1].text(
        0.04, 0.62, r"$s_1=.9s_0+.1g_0^2$",
        transform=axes[1].transAxes, ha="left", color=TEAL, fontsize=11.0, weight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.92),
    )
    clean(axes[1], grid=True)

    bars = axes[2].bar(labels, rescaled_magnitude, color=colors, width=0.58, alpha=0.92)
    axes[2].bar_label(
        bars, labels=["3.162", "3.162"], padding=4, fontsize=12.0, weight="bold"
    )
    axes[2].set_ylim(0, 3.75)
    axes[2].set_ylabel(r"rescaled magnitude $|g_{0,j}|/\sqrt{s_{1,j}}$")
    axes[2].set_title("3 · divide each coordinate\nby its own recent scale", weight="bold", pad=9)
    clean(axes[2], grid=True)

    fig.text(
        0.5,
        0.090,
        "Equal first magnitudes are a zero-start effect: later gradients update each ruler, so the scaled coordinates need not remain equal.",
        ha="center",
        va="center",
        color=INK,
        fontsize=11.1,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F4F8F7", edgecolor=TEAL, linewidth=1.0),
    )
    fig.text(
        0.5,
        0.008,
        r"computed from Example B at $\theta_0=(-1.4,.35)$ · $\rho=.9$ · $\epsilon$ omitted only from this arithmetic illustration",
        ha="center",
        va="bottom",
        color=MUTED,
        fontsize=9.7,
    )
    fig.tight_layout(rect=(0, 0.23, 1, 1), w_pad=2.2)
    save(fig, "story_rmsprop_ruler")


def f_rmsprop_ravine():
    steps = 40
    gd_eta = 0.019
    eta = 0.08
    rho = 0.90
    eps = 1e-8
    gd = gd_path(gd_eta, steps)
    path, scales, normalized, losses = rmsprop_run(eta, rho, steps, eps)
    gd_losses = np.array([ravine_loss(theta) for theta in gd])
    gd_error2 = gd[:, 1] - THETA2_STAR[1]
    rmsprop_error2 = path[:, 1] - THETA2_STAR[1]
    def material_crossings(error, tolerance=1e-6):
        changed_side = np.signbit(error[1:]) != np.signbit(error[:-1])
        away_from_roundoff = (
            np.maximum(np.abs(error[1:]), np.abs(error[:-1])) > tolerance
        )
        return int(np.count_nonzero(changed_side & away_from_roundoff))

    gd_crossings = material_crossings(gd_error2)
    rmsprop_crossings = material_crossings(rmsprop_error2)
    assert np.allclose(scales[1], [0.576, 722.5])
    assert np.allclose(np.abs(normalized[0]), [np.sqrt(10), np.sqrt(10)], atol=1e-6)
    expected_scale_2 = rho * scales[1] + (1 - rho) * ravine_grad(path[1]) ** 2
    assert np.allclose(scales[2], expected_scale_2)
    assert gd_crossings == 40
    assert rmsprop_crossings == 0
    assert np.isclose(gd_losses[-1], 0.6286371741979371)
    assert np.isclose(losses[-1], 0.0008152220927773283)
    assert losses[-1] < gd_losses[-1]
    assert np.all(np.diff(gd_losses) < 0)
    assert np.all(np.diff(losses) < 0)
    assert np.isfinite(path).all() and np.isfinite(losses).all()
    rmsprop_first_below_tenth = int(np.flatnonzero(losses < 0.1)[0])
    assert rmsprop_first_below_tenth == 24
    assert not np.any(gd_losses < 0.1)

    fig, axes = plt.subplots(
        1, 2, figsize=(11.8, 4.15), gridspec_kw={"width_ratios": [1.08, 0.92]}
    )
    draw_ravine(axes[0])
    axes[0].plot(
        gd[:, 0], gd[:, 1], "o-", color=ACC, linewidth=1.5, markersize=2.2,
        label=rf"GD · one scale · $\eta={gd_eta}$",
    )
    axes[0].plot(
        path[:, 0], path[:, 1], "o-", color=GREEN, linewidth=2.0, markersize=2.5,
        label=rf"RMSProp · local scales · $\eta={eta}$",
    )
    axes[0].scatter(*THETA2_0, color=INK, s=38, zorder=7)
    axes[0].set(
        ylabel=r"parameter $\theta_2$",
        title="same actual ravine · 40 exact full-batch gradient evaluations each",
    )
    axes[0].legend(frameon=False, fontsize=10.1, loc="lower right")
    axes[0].text(
        0.98, 0.96, f"GD: {gd_crossings} crossings",
        transform=axes[0].transAxes, ha="right", va="top", color=ACC,
        fontsize=10.5, weight="bold",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.90),
    )
    axes[0].text(
        0.98, 0.89, f"RMSProp: {rmsprop_crossings} crossings",
        transform=axes[0].transAxes, ha="right", va="top", color=GREEN,
        fontsize=10.5, weight="bold",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.90),
    )

    t = np.arange(steps + 1)
    axes[1].semilogy(
        t, gd_losses, color=ACC, lw=1.9,
        label=rf"GD · $L_{{40}}={gd_losses[-1]:.3f}$",
    )
    axes[1].semilogy(
        t, losses, color=GREEN, lw=2.4,
        label=rf"RMSProp · $L_{{40}}={losses[-1]:.2e}$",
    )
    axes[1].scatter([steps], [gd_losses[-1]], color=ACC, s=40, zorder=6)
    axes[1].scatter([steps], [losses[-1]], color=GREEN, s=40, zorder=6)
    axes[1].set(
        xlabel="optimizer update",
        ylabel="full empirical loss (log scale)",
        title="computed loss from the same start",
    )
    axes[1].legend(frameon=False, fontsize=10.5, loc="upper right")
    axes[1].text(
        0.04, 0.07,
        rf"RMSProp reaches $L<.1$ at update {rmsprop_first_below_tenth};"
        "\nGD does not within this 40-update view.",
        transform=axes[1].transAxes, ha="left", va="bottom",
        color=GREEN, fontsize=10.5, weight="bold",
    )
    clean(axes[1], grid=True)

    fig.text(
        0.5,
        0.012,
        rf"computed Example B · 40 exact full-batch gradients · GD $\eta={gd_eta}$ · RMSProp $\eta={eta},\ \rho={rho},\ \epsilon={eps:.0e}$ · method-specific rates, not a benchmark",
        ha="center",
        color=MUTED,
        fontsize=10.0,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 1), w_pad=2.2)
    save(fig, "story_rmsprop_ravine")


def f_optimizer_paths():
    steps = 55
    specs = [
        (gd_path(0.019, steps), "GD", r"$\eta=.019$", ACC),
        (momentum_path(0.095, 0.8, steps), "momentum", r"$\eta_{\rm EWMA}=.095,\ \beta=.8$" "\n" r"(PyTorch lr $=.019$)", TEAL),
        (rmsprop_path(0.08, 0.9, steps), "RMSProp", r"$\eta=.08,\ \rho=.9$", GREEN),
        (adam_path(0.10, 0.9, 0.999, steps), "Adam", r"$\eta=.10,\ \beta=(.9,.999)$", PURPLE),
    ]
    loss_series = {
        method: np.array([ravine_loss(theta) for theta in path])
        for path, method, _settings, _color in specs
    }
    expected_endpoints = {
        "GD": 0.34946145743566004,
        "momentum": 0.00013618968018459663,
        "RMSProp": 1.7248121651950545e-08,
        "Adam": 0.06260063530642346,
    }

    def material_crossings(path, tolerance=1e-6):
        theta2_error = path[:, 1] - THETA2_STAR[1]
        changed_sign = np.signbit(theta2_error[1:]) != np.signbit(theta2_error[:-1])
        away_from_roundoff = (
            np.maximum(np.abs(theta2_error[1:]), np.abs(theta2_error[:-1]))
            > tolerance
        )
        return int(np.count_nonzero(changed_sign & away_from_roundoff))

    crossings = {
        method: material_crossings(path)
        for path, method, _settings, _color in specs
    }
    raw_crossings = {
        method: int(np.count_nonzero(
            np.signbit(path[1:, 1] - THETA2_STAR[1])
            != np.signbit(path[:-1, 1] - THETA2_STAR[1])
        ))
        for path, method, _settings, _color in specs
    }
    assert crossings == {"GD": 55, "momentum": 29, "RMSProp": 0, "Adam": 3}
    # RMSProp's extra raw centre crossings occur only after the theta_2 error has reached
    # numerical roundoff; report material crossings in the teaching visual.
    assert raw_crossings == {"GD": 55, "momentum": 29, "RMSProp": 4, "Adam": 3}
    assert all(len(path) == steps + 1 for path, *_ in specs)
    assert all(np.allclose(path[0], THETA2_0) for path, *_ in specs)
    assert all(np.isfinite(path).all() for path, *_ in specs)
    assert all(np.isfinite(losses).all() for losses in loss_series.values())
    assert all(np.isclose(loss_series[name][-1], value) for name, value in expected_endpoints.items())
    assert (
        loss_series["RMSProp"][-1]
        < loss_series["momentum"][-1]
        < loss_series["Adam"][-1]
        < loss_series["GD"][-1]
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.35),
        gridspec_kw={"width_ratios": [1.65, 0.75]},
    )
    loss_ax, geometry_ax = axes
    update = np.arange(steps + 1)
    for path, method, _settings, color in specs:
        losses = loss_series[method]
        loss_ax.semilogy(update, losses, color=color, linewidth=2.35, label=method)
        loss_ax.scatter([steps], [losses[-1]], s=48, color=color, zorder=7)

    loss_ax.axvline(steps, color=MUTED, linewidth=1.0, linestyle="--", alpha=0.8)
    loss_ax.set(
        xlim=(0, 64),
        ylim=(5e-9, 80),
        xlabel="optimizer update",
        ylabel="full empirical loss (log scale)",
        title="same ravine · same start · 55 exact full-batch updates each",
    )
    endpoint_labels = [
        ("GD", 0.54, r"GD  $0.349$"),
        ("momentum", 2.2e-4, r"momentum  $1.36\times10^{-4}$"),
        ("Adam", 0.043, r"Adam  $0.0626$"),
        ("RMSProp", 1.8e-8, r"RMSProp  $1.72\times10^{-8}$"),
    ]
    colors = {method: color for _path, method, _settings, color in specs}
    for method, label_y, label in endpoint_labels:
        loss_ax.annotate(
            label,
            xy=(steps, loss_series[method][-1]),
            xytext=(57.2, label_y),
            color=colors[method],
            fontsize=10.8,
            weight="bold",
            va="center",
            arrowprops=dict(arrowstyle="-", color=colors[method], lw=1.2),
        )
    loss_ax.text(
        0.025,
        0.055,
        r"Final loss here:  RMSProp  $<$  momentum  $<$  Adam  $<$  GD",
        transform=loss_ax.transAxes,
        color=INK,
        fontsize=11.0,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F4F8F7", edgecolor=TEAL, alpha=0.96),
    )
    clean(loss_ax, grid=True)

    draw_ravine(geometry_ax)
    gd = specs[0][0]
    rmsprop = specs[2][0]
    geometry_ax.plot(
        gd[:, 0], gd[:, 1], color=ACC, linewidth=1.55, marker="o",
        markersize=2.2, markevery=3, label="GD · repeated crossings",
    )
    geometry_ax.plot(
        rmsprop[:, 0], rmsprop[:, 1], color=GREEN, linewidth=2.2, marker="o",
        markersize=2.5, markevery=3, label="RMSProp · follows valley",
    )
    geometry_ax.scatter(*THETA2_0, color=INK, s=40, zorder=8)
    geometry_ax.set(
        ylabel=r"parameter $\theta_2$",
        title="why coordinate scale helps here",
    )
    geometry_ax.legend(frameon=False, fontsize=9.6, loc="lower right")
    geometry_ax.text(
        0.97,
        0.96,
        "one global scale\nvs. one ruler per coordinate",
        transform=geometry_ax.transAxes,
        ha="right",
        va="top",
        color=INK,
        fontsize=9.8,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.90),
    )

    fig.text(
        0.5,
        0.038,
        r"method-specific settings: GD $\eta=.019$ · normalized momentum $\eta=.095,\beta=.8$ (PyTorch lr $=.019$) · RMSProp $\eta=.08,\rho=.9$ · Adam $\eta=.10,\beta=(.9,.999)$",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    fig.text(
        0.5,
        0.006,
        "controlled Example B mechanism comparison · method-specific rates · not a universal optimizer ranking",
        ha="center",
        color=RED,
        fontsize=9.8,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.6)
    save(fig, "story_optimizer_paths")


def sgd_stream(indices, schedule):
    theta = THETA0.copy()
    losses = []
    path = [theta.copy()]
    for t, index in enumerate(indices):
        theta = theta - schedule(t) * line_grad(theta, [index])
        losses.append(line_loss(theta))
        path.append(theta.copy())
    return np.asarray(path), np.asarray(losses)


def f_decay_noise():
    rng = np.random.default_rng(12)
    indices = rng.integers(0, len(X1), 360)
    eta0 = 0.10
    decay_rate = 0.02
    fixed_path, fixed_loss = sgd_stream(indices, lambda _t: eta0)
    decay_path, decay_loss = sgd_stream(
        indices, lambda t: eta0 / (1 + decay_rate * t)
    )
    updates = np.arange(len(indices))
    fixed_schedule = np.full_like(updates, eta0, dtype=float)
    decay_schedule = eta0 / (1 + decay_rate * updates)
    fixed_steps = np.linalg.norm(np.diff(fixed_path, axis=0), axis=1)
    decay_steps = np.linalg.norm(np.diff(decay_path, axis=0), axis=1)
    # Final 100 path points contain 99 actual parameter transitions.
    fixed_late_distance = float(fixed_steps[-99:].sum())
    decay_late_distance = float(decay_steps[-99:].sum())
    distance_ratio = fixed_late_distance / decay_late_distance

    assert np.isclose(decay_schedule[0], eta0)
    assert np.isclose(decay_schedule[-1], 0.012224938875305624)
    assert np.all(np.diff(decay_schedule) < 0)
    assert np.isclose(fixed_late_distance, 3.794479821649366)
    assert np.isclose(decay_late_distance, 0.520387236914856)
    assert np.isclose(distance_ratio, 7.291646590229894)
    assert fixed_late_distance > 7 * decay_late_distance
    assert np.allclose(fixed_path[0], decay_path[0])
    assert np.isfinite(fixed_path).all() and np.isfinite(decay_path).all()
    assert np.isfinite(fixed_loss).all() and np.isfinite(decay_loss).all()
    fixed_extent = np.ptp(fixed_path[-100:], axis=0)
    decay_extent = np.ptp(decay_path[-100:], axis=0)
    assert np.all(fixed_extent > 5 * decay_extent)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.8, 3.75),
        gridspec_kw={"width_ratios": [1.0, 0.72, 1.18]},
    )

    axes[0].plot(updates, fixed_schedule, color=ACC, linewidth=2.5)
    axes[0].plot(updates, decay_schedule, color=TEAL, linewidth=2.5)
    axes[0].scatter([updates[-1]], [fixed_schedule[-1]], color=ACC, s=42, zorder=6)
    axes[0].scatter([updates[-1]], [decay_schedule[-1]], color=TEAL, s=42, zorder=6)
    axes[0].text(
        348, 0.104, r"fixed: $.100$", color=ACC, ha="right", va="bottom",
        fontsize=10.4, weight="bold",
        bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.90),
    )
    axes[0].text(
        348, 0.018, r"decay: $.100\rightarrow.0122$", color=TEAL,
        ha="right", va="bottom", fontsize=10.4, weight="bold",
        bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.90),
    )
    axes[0].set(
        xlim=(0, len(indices) - 1),
        ylim=(0, 0.116),
        xlabel="optimizer update",
        ylabel=r"chosen learning rate $\eta_t$",
        title="1 · WHAT WE CHOOSE\nrate over training",
    )
    clean(axes[0], grid=True)

    bars = axes[1].barh(
        [0, 1],
        [fixed_late_distance, decay_late_distance],
        color=[ACC, TEAL],
        height=0.52,
        alpha=0.92,
    )
    axes[1].invert_yaxis()
    axes[1].set_yticks([0, 1], [r"fixed $.10$", "decay"])
    axes[1].bar_label(
        bars,
        labels=[f"{fixed_late_distance:.3f}", f"{decay_late_distance:.3f}"],
        padding=5,
        fontsize=11.0,
        weight="bold",
    )
    axes[1].text(
        0.96,
        0.08,
        f"decay = {100 / distance_ratio:.0f}%\nof fixed motion",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        color=TEAL,
        fontsize=11.0,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.90),
    )
    axes[1].set(
        xlim=(0, 4.25),
        xlabel="total parameter distance\n(final 99 transitions)",
        title="2 · WHAT IT CHANGES\nactual update distance",
    )
    clean(axes[1], grid=True)

    B, W, loss = line_loss_grid(b_lim=(0.95, 1.25), w_lim=(2.14, 2.42))
    optimum = line_loss(THETA_STAR)
    levels = optimum + np.array([0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03])
    axes[2].contour(
        B, W, loss, levels=levels, colors=[MUTED], linewidths=0.75, alpha=0.48
    )
    axes[2].plot(
        fixed_path[-100:, 0],
        fixed_path[-100:, 1],
        color=ACC,
        linewidth=1.8,
        marker="o",
        markersize=2.3,
        markevery=9,
        alpha=0.86,
        label="fixed · roaming",
    )
    axes[2].plot(
        decay_path[-100:, 0],
        decay_path[-100:, 1],
        color=TEAL,
        linewidth=2.2,
        marker="o",
        markersize=2.5,
        markevery=9,
        alpha=0.94,
        label="decay · settling",
    )
    axes[2].scatter(*THETA_STAR, marker="*", s=165, color=RED, zorder=8)
    axes[2].annotate(
        "least-squares optimum",
        xy=THETA_STAR,
        xytext=(0.965, 2.405),
        color=RED,
        fontsize=9.7,
        weight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.0),
    )
    axes[2].set(
        xlim=(0.95, 1.25),
        ylim=(2.14, 2.42),
        xlabel="bias $b$",
        ylabel="slope $w$",
        title="3 · WHAT WE SEE\ndecayed path settles",
    )
    axes[2].set_aspect("equal")
    axes[2].legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.90,
        fontsize=9.5,
        loc="lower right",
    )
    clean(axes[2])

    fig.tight_layout(w_pad=1.6)
    save(fig, "story_decay_noise")


def f_schedule_two_jobs():
    """Use one seeded SGD stream to show travel early and refinement late."""
    rng = np.random.default_rng(12)
    indices = rng.integers(0, len(X1), 360)
    large_eta = 0.10
    small_eta = 0.02
    decay_rate = 0.02
    large_path, large_loss = sgd_stream(indices, lambda _t: large_eta)
    small_path, small_loss = sgd_stream(indices, lambda _t: small_eta)
    decay_path, decay_loss = sgd_stream(
        indices, lambda t: large_eta / (1 + decay_rate * t)
    )
    optimum_loss = line_loss(THETA_STAR)
    gaps = {
        "large fixed": large_loss - optimum_loss,
        "small fixed": small_loss - optimum_loss,
        "decay": decay_loss - optimum_loss,
    }
    paths = {
        "large fixed": large_path,
        "small fixed": small_path,
        "decay": decay_path,
    }
    colors = {"large fixed": ACC, "small fixed": BLUE, "decay": TEAL}

    travel_target = 0.5

    def first_below(values, threshold):
        hits = np.flatnonzero(values < threshold)
        assert len(hits) > 0
        return int(hits[0] + 1)  # losses[0] is after update 1

    travel_updates = {
        name: first_below(values, travel_target) for name, values in gaps.items()
    }
    late_mean_gap = {name: float(values[-100:].mean()) for name, values in gaps.items()}
    late_motion = {
        name: float(np.linalg.norm(np.diff(path[-100:], axis=0), axis=1).sum())
        for name, path in paths.items()
    }
    assert travel_updates == {"large fixed": 8, "small fixed": 56, "decay": 8}
    assert np.isclose(late_mean_gap["large fixed"], 0.004099611448621285)
    assert np.isclose(late_mean_gap["small fixed"], 0.0012871016156542003)
    assert np.isclose(late_mean_gap["decay"], 0.00020140237311844306)
    assert late_mean_gap["decay"] < late_mean_gap["small fixed"] < late_mean_gap["large fixed"]
    assert np.isclose(late_motion["large fixed"], 3.794479821649366)
    assert np.isclose(late_motion["small fixed"], 0.7391207330467251)
    assert np.isclose(late_motion["decay"], 0.520387236914856)
    assert late_motion["decay"] < late_motion["small fixed"] < late_motion["large fixed"]
    assert all(np.allclose(path[0], THETA0) for path in paths.values())
    assert all(np.isfinite(path).all() for path in paths.values())
    assert all(np.isfinite(values).all() and np.all(values >= 0) for values in gaps.values())

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))

    early_horizon = 80
    early_updates = np.arange(early_horizon + 1)
    start_gap = line_loss(THETA0) - optimum_loss
    for name in ["large fixed", "small fixed", "decay"]:
        early_gap = np.r_[start_gap, gaps[name][:early_horizon]]
        label = {
            "large fixed": r"fixed $\eta=.10$",
            "small fixed": r"fixed $\eta=.02$",
            "decay": r"decay from $\eta=.10$",
        }[name]
        axes[0].semilogy(
            early_updates, early_gap, color=colors[name], linewidth=2.35, label=label
        )
        hit = travel_updates[name]
        axes[0].scatter(
            [hit], [gaps[name][hit - 1]], color=colors[name], s=45, zorder=7
        )
    axes[0].axhline(travel_target, color=MUTED, linewidth=1.0, linestyle="--")
    axes[0].text(
        78,
        travel_target * 1.13,
        r"travel target: loss gap $<.5$",
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=10.0,
    )
    axes[0].annotate(
        "large + decay: 8 updates",
        xy=(8, travel_target),
        xytext=(16, 1.25),
        color=TEAL,
        fontsize=10.4,
        weight="bold",
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.1),
    )
    axes[0].annotate(
        "small: 56 updates",
        xy=(56, gaps["small fixed"][55]),
        xytext=(42, 2.1),
        color=BLUE,
        fontsize=10.4,
        weight="bold",
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1),
    )
    axes[0].set(
        xlim=(0, early_horizon),
        ylim=(3e-4, 10),
        xlabel="stochastic update",
        ylabel="full-loss gap (log scale)",
        title="EARLY · travel into a useful region",
    )
    axes[0].legend(frameon=False, fontsize=10.0, loc="lower left")
    clean(axes[0], grid=True)

    B, W, loss = line_loss_grid(b_lim=(0.95, 1.25), w_lim=(2.14, 2.42))
    contour_levels = optimum_loss + np.array([0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03])
    axes[1].contour(
        B, W, loss, levels=contour_levels, colors=[MUTED], linewidths=0.75, alpha=0.48
    )
    for name in ["large fixed", "small fixed", "decay"]:
        late = paths[name][-100:]
        label = {
            "large fixed": rf"$\eta=.10$ · mean gap {late_mean_gap[name]:.4f}",
            "small fixed": rf"$\eta=.02$ · mean gap {late_mean_gap[name]:.4f}",
            "decay": rf"decay · mean gap {late_mean_gap[name]:.5f}",
        }[name]
        axes[1].plot(
            late[:, 0],
            late[:, 1],
            color=colors[name],
            linewidth=1.75 if name != "decay" else 2.15,
            marker="o",
            markersize=2.4,
            markevery=9,
            alpha=0.88,
            label=label,
        )
    axes[1].scatter(*THETA_STAR, marker="*", s=165, color=RED, zorder=8)
    axes[1].set(
        xlim=(0.95, 1.25),
        ylim=(2.14, 2.42),
        xlabel="bias $b$",
        ylabel="slope $w$",
        title="LATE · last 100 updates refine the fit",
    )
    axes[1].set_aspect("equal")
    axes[1].legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.90,
        fontsize=9.5,
        loc="upper right",
    )
    axes[1].text(
        0.035,
        0.055,
        "large rate keeps roaming;\nsmall and decayed rates stay closer",
        transform=axes[1].transAxes,
        color=INK,
        fontsize=10.1,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.90),
    )
    clean(axes[1])

    # The slide supplies the provenance and takeaway below the figure; keep the
    # asset itself focused on the two computed comparisons.
    fig.tight_layout(rect=(0, 0.02, 1, 1), w_pad=1.6)
    save(fig, "story_schedule_two_jobs")


def f_schedule_shapes():
    horizon = 100
    eta0 = 0.10
    decay = 0.02
    update = np.arange(horizon + 1)
    inverse_time = eta0 / (1 + decay * update)
    exponential = eta0 * np.exp(-decay * update)
    cosine = 0.5 * eta0 * (1 + np.cos(np.pi * update / horizon))
    schedules = [
        (inverse_time, rf"inverse-time $\eta_t={eta0:.2f}/(1+{decay:.2f}t)$", ACC),
        (exponential, rf"exponential $\eta_t={eta0:.2f}e^{{-{decay:.2f}t}}$", TEAL),
        (cosine, rf"cosine $\eta_t=\frac{{{eta0:.2f}}}{{2}}[1+\cos(\pi t/{horizon})]$", PURPLE),
    ]
    for values, _label, _color in schedules:
        assert np.isclose(values[0], eta0)
        assert np.all(np.diff(values) <= 1e-12)
        assert np.all(values >= 0)
    assert np.isclose(cosine[-1], 0.0)

    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    for values, label, color in schedules:
        ax.plot(update, values, color=color, lw=2.4, label=label)
        ax.scatter([horizon], [values[-1]], color=color, s=28, zorder=5)
    ax.set(
        xlim=(0, horizon), ylim=(-0.003, eta0 * 1.06),
        xlabel="optimizer update $t$", ylabel=r"learning rate $\eta_t$",
        title=rf"same start $\eta_0={eta0:.2f}$ and same {horizon}-update horizon",
    )
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    clean(ax, grid=True)
    fig.text(
        0.5, 0.01,
        "schedule shapes only · curves do not imply optimizer performance · inverse-time is applied once per optimizer update",
        ha="center", color=RED, fontsize=9.6, weight="bold",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "story_schedule_shapes")


def xor_optimizer_results():
    """Recompute the controlled experiment in notebook 04 exactly."""
    import torch

    seed = 2026
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])

    def make_model():
        return torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 1),
        )

    base_model = make_model()
    initial_state = {
        name: value.detach().clone() for name, value in base_model.state_dict().items()
    }
    generator = torch.Generator().manual_seed(seed + 1)
    batch_schedule = []
    for _ in range(200):
        order = torch.randperm(len(x), generator=generator)
        batch_schedule.extend([order[:2], order[2:]])
    assert len(batch_schedule) == 400
    assert [indices.tolist() for indices in batch_schedule[:4]] == [[1, 2], [3, 0], [3, 0], [1, 2]]

    specs = [
        ("plain SGD", lambda params: torch.optim.SGD(params, lr=0.25), ACC, r"$\eta=.25$"),
        (
            "momentum SGD",
            lambda params: torch.optim.SGD(params, lr=0.12, momentum=0.9),
            TEAL,
            r"$\eta=.12,\ \beta=.9$",
        ),
        (
            "RMSProp",
            lambda params: torch.optim.RMSprop(params, lr=0.03, alpha=0.9, eps=1e-8),
            GREEN,
            r"$\eta=.03,\ \rho=.9$",
        ),
        (
            "Adam",
            lambda params: torch.optim.Adam(params, lr=0.03, betas=(0.9, 0.999), eps=1e-8),
            PURPLE,
            r"$\eta=.03,\ \beta=(.9,.999)$",
        ),
    ]
    criterion = torch.nn.BCEWithLogitsLoss()
    grid_axis = np.linspace(-0.35, 1.35, 180)
    g1, g2 = np.meshgrid(grid_axis, grid_axis)
    grid = torch.tensor(np.column_stack([g1.ravel(), g2.ravel()]), dtype=torch.float32)
    results = []

    for name, make_optimizer, color, rate_label in specs:
        model = make_model()
        model.load_state_dict(copy.deepcopy(initial_state))
        for state_name, value in model.state_dict().items():
            assert torch.equal(value, initial_state[state_name])
        optimizer = make_optimizer(model.parameters())
        losses = []
        for indices in batch_schedule:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[indices]).squeeze(-1)
            criterion(logits, y[indices]).backward()
            optimizer.step()
            with torch.no_grad():
                losses.append(criterion(model(x).squeeze(-1), y).item())
        with torch.no_grad():
            probabilities = torch.sigmoid(model(x).squeeze(-1)).numpy()
            probability_grid = torch.sigmoid(model(grid).squeeze(-1)).reshape(g1.shape).numpy()
        accuracy = np.mean((probabilities >= 0.5) == y.numpy().astype(bool))
        assert accuracy == 1.0
        results.append(
            {
                "name": name,
                "rate_label": rate_label,
                "color": color,
                "losses": np.asarray(losses),
                "probabilities": probabilities,
                "probability_grid": probability_grid,
            }
        )

    expected = {
        "plain SGD": (9.424e-2, [0.082, 0.885, 0.902, 0.064]),
        "momentum SGD": (6.151e-3, [0.005, 0.993, 0.992, 0.005]),
        "RMSProp": (5.770e-3, [0.001, 0.995, 0.991, 0.009]),
        "Adam": (3.998e-2, [0.012, 0.961, 0.940, 0.045]),
    }
    for result in results:
        expected_loss, expected_probabilities = expected[result["name"]]
        assert np.isclose(result["losses"][-1], expected_loss, rtol=5e-4)
        assert np.allclose(result["probabilities"], expected_probabilities, atol=6e-4)
    return x.numpy(), y.numpy(), g1, g2, results


def f_xor_optimizer_lab():
    _x, _y, _g1, _g2, results = xor_optimizer_results()
    fig, ax = plt.subplots(figsize=(9.0, 4.35))
    update = np.arange(1, 401)
    for result in results:
        ax.semilogy(
            update, result["losses"], color=result["color"], lw=2.1,
            label=(
                rf"{result['name']} · {result['rate_label']} · "
                rf"$L_{{400}}={result['losses'][-1]:.2e}$"
            ),
        )
    ax.set(xlim=(1, 400), xlabel="optimizer update", ylabel="full-data XOR BCE")
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle("same initialization · same scheduled size-2 minibatches · 400 updates", y=0.98)
    fig.legend(handles, labels, frameon=False, fontsize=10.3, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 0.88))
    clean(ax, grid=True)
    fig.text(
        0.5, 0.01,
        "seeded mechanism demo · seed 2026 · method-specific rates · not a benchmark or universal ranking",
        ha="center", color=RED, fontsize=9.7, weight="bold",
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.66)
    save(fig, "story_xor_optimizer_lab")


def f_xor_decision_boundaries():
    x, y, g1, g2, results = xor_optimizer_results()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.15), sharex=True, sharey=True)
    filled = None
    for panel, (ax, result) in enumerate(zip(axes.ravel(), results)):
        filled = ax.contourf(
            g1, g2, result["probability_grid"], levels=np.linspace(0, 1, 11),
            cmap="RdBu_r", alpha=0.74, vmin=0, vmax=1,
        )
        ax.contour(g1, g2, result["probability_grid"], levels=[0.5], colors=INK, linewidths=2.0)
        ax.scatter(
            x[:, 0], x[:, 1], c=y, cmap="RdBu_r", vmin=0, vmax=1,
            s=92, edgecolor="white", linewidth=1.2, zorder=5,
        )
        for point, target in zip(x, y):
            ax.text(
                point[0], point[1], str(int(target)), color="white", fontsize=9.0,
                weight="bold", ha="center", va="center", zorder=6,
            )
        ax.set(
            xlim=(-0.35, 1.35), ylim=(-0.35, 1.35), aspect="equal",
            xlabel="$x_1$", ylabel="$x_2$",
        )
        ax.set_title(
            rf"{result['name']} · {result['rate_label']}" "\n"
            rf"$L_{{400}}={result['losses'][-1]:.2e}$",
            fontsize=10.3, pad=5,
        )
        if panel < 2:
            ax.set_xlabel("")
        if panel % 2 == 1:
            ax.set_ylabel("")
        clean(ax)
    assert filled is not None
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.80, hspace=0.45, wspace=0.18)
    colorbar_ax = fig.add_axes([0.20, 0.105, 0.60, 0.028])
    colorbar = fig.colorbar(filled, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_label(r"learned probability $p(y=1\mid x)$; dark contour is $p=.5$")
    colorbar.set_ticks([0, 0.5, 1])
    fig.suptitle("final XOR decision boundaries · same 400-update budget", y=0.985, fontsize=14)
    fig.text(
        0.5, 0.925, "all four classify the four XOR training points correctly",
        ha="center", color=MUTED, fontsize=10.5,
    )
    fig.text(
        0.5, 0.01,
        "seeded mechanism demo · seed 2026 · shared initialization and minibatch order · not a benchmark",
        ha="center", color=RED, fontsize=9.6, weight="bold",
    )
    save(fig, "story_xor_decision_boundaries")


def f_constraint_failure():
    """Show one exact unconstrained step making a Gaussian scale invalid."""
    scale_0 = 0.10
    eta = 0.02
    gradient_0 = 1 / scale_0
    scale_1 = scale_0 - eta * gradient_0

    assert np.isclose(gradient_0, 10.0)
    assert np.isclose(scale_1, -0.10)
    assert scale_0 > 0
    assert scale_1 < 0
    assert scale_0 - scale_1 == 0.20

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.8, 3.25),
        gridspec_kw={"width_ratios": [0.85, 1.75, 0.95]},
    )

    start_ax = axes[0]
    start_ax.axis("off")
    start_ax.text(
        0.5, 0.88, "VALID START", ha="center", va="center",
        color=TEAL, fontsize=12.0, weight="bold",
    )
    start_ax.text(
        0.5, 0.70, r"one observation: $y=\mu$", ha="center", va="center",
        color=INK, fontsize=12.0,
    )
    start_ax.text(
        0.5, 0.52, r"$\mathcal{L}(s)=\log s$", ha="center", va="center",
        color=INK, fontsize=17.0, weight="bold",
    )
    start_ax.text(
        0.5, 0.25,
        r"$s_0=.10$" "\n" r"$\left.\dfrac{\partial\mathcal{L}}{\partial s}\right|_{s_0}=10$",
        ha="center", va="center", color=INK, fontsize=13.3,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F4F8F7", edgecolor=TEAL, linewidth=1.1),
    )
    start_ax.text(
        0.5, 0.015, "(up to an additive constant)", ha="center", va="center",
        color=MUTED, fontsize=9.6,
    )

    step_ax = axes[1]
    step_ax.set_xlim(-0.16, 0.16)
    step_ax.set_ylim(-0.72, 0.85)
    step_ax.axvspan(-0.16, 0, color="#FCEDEE", zorder=0)
    step_ax.axvspan(0, 0.16, color="#EAF7F0", zorder=0)
    step_ax.axhline(0, color=INK, linewidth=1.7)
    step_ax.axvline(0, color=RED, linewidth=2.0, linestyle="--")
    step_ax.annotate(
        "",
        xy=(scale_1, 0),
        xytext=(scale_0, 0),
        arrowprops=dict(arrowstyle="-|>", color=ACC, lw=3.2, mutation_scale=18),
        zorder=5,
    )
    step_ax.scatter([scale_0], [0], color=TEAL, s=105, zorder=7)
    step_ax.scatter(
        [scale_1], [0], color=RED, marker="X", s=145,
        edgecolor="white", linewidth=1.2, zorder=7,
    )
    step_ax.text(
        0.08, 0.60, "FEASIBLE", ha="center", va="center",
        color=GREEN, fontsize=11.0, weight="bold",
    )
    step_ax.text(
        0.08, 0.43, r"$s\in(0,\infty)$", ha="center", va="center",
        color=INK, fontsize=13.0,
    )
    step_ax.text(
        -0.08, 0.60, "INVALID", ha="center", va="center",
        color=RED, fontsize=11.0, weight="bold",
    )
    step_ax.text(
        -0.08, 0.43, r"$s<0$", ha="center", va="center",
        color=INK, fontsize=13.0,
    )
    step_ax.text(
        0, 0.14, "boundary 0", ha="center", va="bottom",
        color=RED, fontsize=10.0, weight="bold",
        bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.94),
    )
    step_ax.text(
        scale_0, -0.20, r"current $s_0=.10$", ha="center", va="top",
        color=TEAL, fontsize=10.5, weight="bold",
        bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.94),
    )
    step_ax.text(
        scale_1, -0.20, r"proposal $s_1=-.10$", ha="center", va="top",
        color=RED, fontsize=10.5, weight="bold",
        bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.94),
    )
    step_ax.text(
        0, -0.52, r"$s_1=.10-.02(10)=-.10$", ha="center", va="center",
        color=INK, fontsize=14.0, weight="bold",
    )
    step_ax.set_title("ONE UNCONSTRAINED GRADIENT STEP", fontsize=12.0, weight="bold", pad=8)
    step_ax.set_xticks([])
    step_ax.set_yticks([])
    for spine in step_ax.spines.values():
        spine.set_visible(False)

    failure_ax = axes[2]
    failure_ax.axis("off")
    failure_ax.text(
        0.5, 0.86, "INVALID FORWARD", ha="center", va="center",
        color=RED, fontsize=12.0, weight="bold",
    )
    failure_ax.text(
        0.5, 0.63, "×", ha="center", va="center",
        color=RED, fontsize=40.0, weight="bold",
    )
    failure_ax.text(
        0.5, 0.40, r"$\log(s_1)$ is undefined", ha="center", va="center",
        color=INK, fontsize=13.0, weight="bold",
    )
    failure_ax.text(
        0.5, 0.20, r"Normal$(\mu,\ \mathrm{scale}=s_1)$" "\n" "rejects a negative scale",
        ha="center", va="center", color=INK, fontsize=11.2,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FCEDEE", edgecolor=RED, linewidth=1.1),
    )

    fig.tight_layout(w_pad=1.35)
    save(fig, "story_constraint_failure")


def f_constraint_maps():
    raw = np.linspace(-6, 6, 400)
    exponential = np.exp(raw)
    softplus = np.logaddexp(0, raw)
    sigmoid = 1 / (1 + np.exp(-raw))
    projected = np.maximum(raw, 0)
    assert np.all(exponential > 0) and np.all(softplus > 0)
    assert projected.min() == 0 and np.all(projected >= 0)
    assert np.all((sigmoid > 0) & (sigmoid < 1))

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.7))
    axes[0].semilogy(raw, exponential, color=BLUE, lw=2.2, label=r"$s=e^r$ · $r=\log s$")
    axes[0].semilogy(raw, softplus, color=TEAL, lw=2.2, label=r"$s=\operatorname{softplus}(r)$")
    axes[0].set(xlabel=r"unconstrained raw value $r$", ylabel=r"strictly positive $s$")
    axes[0].set_title(r"$s>0$" "\n" "reparameterize: interior only", fontsize=11.3, pad=7)
    axes[0].legend(frameon=False, fontsize=10.4, loc="upper left")
    clean(axes[0], grid=True)

    axes[1].plot(raw, projected, color=GREEN, lw=2.3)
    axes[1].axhline(0, color=RED, linewidth=1, linestyle="--")
    axes[1].axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
    axes[1].scatter([0], [0], color=RED, s=48, zorder=5)
    axes[1].set(xlabel=r"unconstrained proposal $r$", ylabel=r"nonnegative value $s$")
    axes[1].set_title(r"$s\geq0$" "\n" "project: boundary allowed", fontsize=11.3, pad=7)
    axes[1].text(0.04, 0.88, "kink at 0", transform=axes[1].transAxes, color=RED, fontsize=10.5)
    clean(axes[1], grid=True)

    axes[2].plot(raw, sigmoid, color=ACC, lw=2.3)
    axes[2].axhline(0, color=RED, linewidth=1, linestyle="--")
    axes[2].axhline(1, color=RED, linewidth=1, linestyle="--")
    axes[2].set(xlabel=r"unconstrained logit $a$", ylabel=r"probability $p$")
    axes[2].set_title(r"$p=\sigma(a)\in(0,1)$" "\n" "reparameterize: interior only", fontsize=11.3, pad=7)
    axes[2].text(
        0.5, 0.08, "finite precision may saturate at 0 or 1",
        transform=axes[2].transAxes, ha="center", color=RED, fontsize=10.3,
    )
    clean(axes[2], grid=True)
    fig.suptitle("smooth interior maps and an exact closed-boundary projection", y=0.995, fontsize=13.0)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save(fig, "story_constraint_maps")


FIGURES = [
    f_line_fit_revision,
    f_per_example_contours,
    f_bad_sample_step,
    f_gradient_estimates,
    f_individual_losses,
    f_individual_updates,
    f_batch_loss_surfaces,
    f_update_vector_fan,
    f_update_vector_average,
    f_estimator_clouds,
    f_batch_paths,
    f_batch_loss,
    f_estimator_spread,
    f_ravine_geometry,
    f_ravine_first_step,
    f_ravine_learning_rates,
    f_feature_scaling_fix,
    f_ravine_rate_traces,
    f_ravine_gradients,
    f_ewma_raw,
    f_ewma_beta05,
    f_ewma_beta09,
    f_ewma_beta098,
    f_ewma_bias,
    f_ewma_weights,
    f_gradient_filter,
    f_momentum_ravine,
    f_momentum_loss,
    f_rmsprop_ruler,
    f_rmsprop_ravine,
    f_optimizer_paths,
    f_decay_noise,
    f_schedule_two_jobs,
    f_schedule_shapes,
    f_xor_optimizer_lab,
    f_xor_decision_boundaries,
    f_constraint_failure,
    f_constraint_maps,
]


if __name__ == "__main__":
    for make_figure in FIGURES:
        make_figure()
        print("ok", make_figure.__name__)
    print("done ->", OUT)
