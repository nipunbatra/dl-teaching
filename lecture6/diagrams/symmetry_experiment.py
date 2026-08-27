#!/usr/bin/env python3
"""Reproduce the Lecture 6 hidden-unit symmetry experiment.

The controlled comparison uses the same XOR data, architecture, optimizer, and
base random draw twice.  In the ``cloned`` run, the four hidden rows, hidden
biases, and four outgoing coefficients are copied from unit 1.  They remain
separate tensor coordinates, so any equality after training is a consequence
of equal gradients rather than parameter tying.

Run from the repository root:

    uv run --with torch --with matplotlib --with numpy \
      python lecture6/diagrams/symmetry_experiment.py

Outputs:

* ``lecture6/figures/symmetry_decision_boundaries.svg``
* ``lecture6/figures/symmetry_decision_boundaries.png``
* ``lecture6/evidence/symmetry_experiment.json``
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "lecture6" / "figures"
EVIDENCE = ROOT / "lecture6" / "evidence"
FIGURES.mkdir(parents=True, exist_ok=True)
EVIDENCE.mkdir(parents=True, exist_ok=True)

INK = "#23373B"
ACC = "#EB811B"
TEAL = "#2C7A7B"
GREEN = "#14B03D"
BLUE = "#2B6CB0"
MUTED = "#6E7F82"
PAPER = "#FBFAF7"

DATA_SEED = 7
WEIGHT_SEED = 1
POINTS_PER_BLOB = 80
SIGMA = 0.18
LEARNING_RATE = 0.03
STEPS = 300

mpl.rcParams.update(
    {
        "figure.facecolor": "none",
        "axes.facecolor": PAPER,
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans", "Arial"],
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "path",
    }
)

# The experiment is tiny; one CPU thread plus deterministic algorithms makes
# the reported values stable across repeated runs on the same PyTorch build.
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def make_xor_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Return four balanced Gaussian blobs with quadrant-XOR labels."""

    generator = torch.Generator().manual_seed(DATA_SEED)
    blobs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    # Class 1 means that the two coordinate signs differ.
    for center, label in (
        ((-1.0, -1.0), 0.0),
        ((-1.0, +1.0), 1.0),
        ((+1.0, -1.0), 1.0),
        ((+1.0, +1.0), 0.0),
    ):
        noise = SIGMA * torch.randn(POINTS_PER_BLOB, 2, generator=generator)
        blobs.append(torch.tensor(center, dtype=torch.float32) + noise)
        labels.append(torch.full((POINTS_PER_BLOB,), label, dtype=torch.float32))

    x = torch.cat(blobs)
    y = torch.cat(labels)
    assert x.shape == (4 * POINTS_PER_BLOB, 2)
    assert torch.bincount(y.to(torch.long)).tolist() == [160, 160]
    return x, y


class TinyXORMLP(nn.Module):
    """A 2 -> 4 ReLU -> 1 binary classifier."""

    def __init__(self, *, cloned: bool) -> None:
        super().__init__()

        # Seed before constructing the modules.  Linear's constructors consume
        # random numbers; the explicit initializers below then consume the next
        # draws.  Keeping this sequence explicit reproduces the audited result.
        torch.manual_seed(WEIGHT_SEED)
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)
        nn.init.kaiming_normal_(
            self.hidden.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        if cloned:
            with torch.no_grad():
                self.hidden.weight.copy_(self.hidden.weight[0].expand_as(self.hidden.weight))
                self.hidden.bias.copy_(self.hidden.bias[0].expand_as(self.hidden.bias))
                self.output.weight.copy_(
                    self.output.weight[0, 0].expand_as(self.output.weight)
                )

    def hidden_activations(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.hidden(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden_activations(x)).squeeze(-1)


def clone_parameter_spread(model: TinyXORMLP) -> float:
    """Maximum difference from hidden unit 1 across cloned coordinates."""

    spreads = (
        (model.hidden.weight - model.hidden.weight[0]).abs().max(),
        (model.hidden.bias - model.hidden.bias[0]).abs().max(),
        (model.output.weight - model.output.weight[0, 0]).abs().max(),
    )
    return max(value.item() for value in spreads)


@dataclass(frozen=True)
class RunMetrics:
    loss: float
    accuracy: float
    activation_rank: int
    initial_parameter_spread: float
    final_parameter_spread: float


def train_one(
    x: torch.Tensor, y: torch.Tensor, *, cloned: bool
) -> tuple[TinyXORMLP, RunMetrics]:
    model = TinyXORMLP(cloned=cloned)
    initial_spread = clone_parameter_spread(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(x)
        activations = model.hidden_activations(x)
        final_loss = F.binary_cross_entropy_with_logits(logits, y).item()
        predictions = logits >= 0.0
        accuracy = (predictions == y.bool()).float().mean().item()
        rank = int(torch.linalg.matrix_rank(activations).item())
        final_spread = clone_parameter_spread(model)

    return model, RunMetrics(
        loss=final_loss,
        accuracy=accuracy,
        activation_rank=rank,
        initial_parameter_spread=initial_spread,
        final_parameter_spread=final_spread,
    )


def assert_audited_results(
    cloned: RunMetrics, independent: RunMetrics
) -> None:
    """Fail loudly if a dependency or code change alters the teaching result."""

    assert cloned.initial_parameter_spread == 0.0
    assert cloned.final_parameter_spread == 0.0
    assert cloned.activation_rank == 1
    assert cloned.accuracy == 0.75
    assert abs(cloned.loss - 0.47758022) < 2e-5

    assert independent.activation_rank == 4
    assert independent.accuracy == 1.0
    assert abs(independent.loss - 0.000909758) < 2e-5


def hinge_segment(
    weight: np.ndarray,
    bias: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the visible segment of w1*x1 + w2*x2 + b = 0."""

    w1, w2 = (float(weight[0]), float(weight[1]))
    candidates: list[tuple[float, float]] = []
    tolerance = 1e-7

    if abs(w2) > tolerance:
        for x1 in xlim:
            x2 = -(w1 * x1 + bias) / w2
            if ylim[0] - tolerance <= x2 <= ylim[1] + tolerance:
                candidates.append((x1, x2))
    if abs(w1) > tolerance:
        for x2 in ylim:
            x1 = -(w2 * x2 + bias) / w1
            if xlim[0] - tolerance <= x1 <= xlim[1] + tolerance:
                candidates.append((x1, x2))

    unique: list[tuple[float, float]] = []
    for point in candidates:
        if not any(np.linalg.norm(np.subtract(point, other)) < 1e-6 for other in unique):
            unique.append(point)
    if len(unique) < 2:
        raise RuntimeError("hidden hinge does not intersect the plotting window")

    return np.asarray([unique[0][0], unique[1][0]]), np.asarray(
        [unique[0][1], unique[1][1]]
    )


def decision_probability(
    model: TinyXORMLP, xx: np.ndarray, yy: np.ndarray
) -> np.ndarray:
    grid = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32))
    with torch.no_grad():
        probability = torch.sigmoid(model(grid)).reshape(xx.shape)
    return probability.numpy()


def draw_panel(
    ax: plt.Axes,
    *,
    model: TinyXORMLP,
    metrics: RunMetrics,
    x: torch.Tensor,
    y: torch.Tensor,
    cloned: bool,
    xx: np.ndarray,
    yy: np.ndarray,
    cmap: LinearSegmentedColormap,
) -> mpl.contour.QuadContourSet:
    probability = decision_probability(model, xx, yy)
    fill = ax.contourf(
        xx,
        yy,
        probability,
        levels=np.linspace(0.0, 1.0, 17),
        cmap=cmap,
        antialiased=True,
    )
    ax.contour(
        xx,
        yy,
        probability,
        levels=[0.5],
        colors=[INK],
        linewidths=2.7,
        zorder=4,
    )

    weights = model.hidden.weight.detach().cpu().numpy()
    biases = model.hidden.bias.detach().cpu().numpy()
    hinge_colors = (TEAL, GREEN, BLUE, ACC)
    if cloned:
        hx, hy = hinge_segment(weights[0], float(biases[0]), (-1.65, 1.65), (-1.65, 1.65))
        ax.plot(hx, hy, color="white", linewidth=5.0, alpha=0.82, zorder=3)
        ax.plot(
            hx,
            hy,
            color=TEAL,
            linewidth=2.5,
            linestyle=(0, (4, 2.5)),
            zorder=5,
        )
        midpoint = (float(np.mean(hx)), float(np.mean(hy)))
        ax.annotate(
            "4 hidden hinges\noverlap exactly",
            xy=midpoint,
            xytext=(-1.52, 1.48),
            color=TEAL,
            fontsize=10.2,
            fontweight=600,
            ha="left",
            va="top",
            arrowprops={
                "arrowstyle": "->",
                "color": TEAL,
                "linewidth": 1.2,
                "shrinkA": 3,
                "shrinkB": 4,
            },
            zorder=8,
        )
    else:
        for index, (weight, bias, color) in enumerate(
            zip(weights, biases, hinge_colors, strict=True), start=1
        ):
            hx, hy = hinge_segment(weight, float(bias), (-1.65, 1.65), (-1.65, 1.65))
            ax.plot(
                hx,
                hy,
                color=color,
                linewidth=1.8,
                linestyle=(0, (4, 2.5)),
                alpha=0.92,
                label=f"hidden hinge {index}",
                zorder=5,
            )

    x_np = x.numpy()
    y_np = y.numpy().astype(bool)
    ax.scatter(
        x_np[~y_np, 0],
        x_np[~y_np, 1],
        s=20,
        marker="o",
        facecolor=BLUE,
        edgecolor="white",
        linewidth=0.55,
        alpha=0.9,
        zorder=7,
    )
    ax.scatter(
        x_np[y_np, 0],
        x_np[y_np, 1],
        s=23,
        marker="^",
        facecolor=ACC,
        edgecolor="white",
        linewidth=0.55,
        alpha=0.9,
        zorder=7,
    )

    condition = "Cloned hidden units" if cloned else "Independent hidden units"
    subtitle = (
        "one feature, copied 4 times"
        if cloned
        else "four independently learned features"
    )
    ax.set_title(condition, fontsize=16.5, fontweight=600, pad=24, color=INK)
    ax.text(
        0.5,
        1.035,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=TEAL if cloned else GREEN,
        fontweight=600,
    )
    ax.text(
        0.5,
        -0.17,
        (
            f"accuracy {100 * metrics.accuracy:.1f}%   |   "
            f"BCE {metrics.loss:.5f}   |   rank(H) = {metrics.activation_rank}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.6,
        color=INK,
        fontweight=600,
    )

    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.65, 1.65)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"input $x_1$", fontsize=11.5)
    ax.set_ylabel(r"input $x_2$", fontsize=11.5)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(labelsize=9.5, length=3)
    ax.grid(color=MUTED, alpha=0.14, linewidth=0.65)
    return fill


def make_figure(
    x: torch.Tensor,
    y: torch.Tensor,
    cloned_model: TinyXORMLP,
    cloned_metrics: RunMetrics,
    independent_model: TinyXORMLP,
    independent_metrics: RunMetrics,
) -> None:
    axis = np.linspace(-1.65, 1.65, 281, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    probability_cmap = LinearSegmentedColormap.from_list(
        "lecture6_probability",
        ("#C6DDF2", "#F7F5EF", "#F6D1AE"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.85))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.82, bottom=0.22, wspace=0.18)
    fill = draw_panel(
        axes[0],
        model=cloned_model,
        metrics=cloned_metrics,
        x=x,
        y=y,
        cloned=True,
        xx=xx,
        yy=yy,
        cmap=probability_cmap,
    )
    draw_panel(
        axes[1],
        model=independent_model,
        metrics=independent_metrics,
        x=x,
        y=y,
        cloned=False,
        xx=xx,
        yy=yy,
        cmap=probability_cmap,
    )

    probability_axis = fig.add_axes((0.38, 0.055, 0.24, 0.026))
    colorbar = fig.colorbar(fill, cax=probability_axis, orientation="horizontal")
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_ticklabels(["0", "0.5", "1"])
    colorbar.ax.tick_params(labelsize=8.5, length=2, pad=2)
    colorbar.outline.set_linewidth(0.6)
    colorbar.set_label(r"background: predicted $p(y=1\mid x)$", fontsize=9.5, labelpad=2)

    legend_handles = [
        Line2D(
            [0], [0], color=INK, linewidth=2.7, label=r"decision boundary  $p=0.5$"
        ),
        Line2D(
            [0],
            [0],
            color=TEAL,
            linewidth=2.0,
            linestyle=(0, (4, 2.5)),
            label=r"hidden hinge  $w_i^T x+b_i=0$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=BLUE,
            markeredgecolor="white",
            markersize=6.5,
            label="class 0",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor=ACC,
            markeredgecolor="white",
            markersize=7,
            label="class 1",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        columnspacing=1.55,
        handlelength=2.7,
        fontsize=9.4,
    )

    svg_path = FIGURES / "symmetry_decision_boundaries.svg"
    png_path = FIGURES / "symmetry_decision_boundaries.png"
    fig.savefig(svg_path, bbox_inches="tight", transparent=True)
    fig.savefig(png_path, bbox_inches="tight", transparent=True, dpi=220)
    plt.close(fig)

    # A vector-safe SVG must not embed a rasterized copy of the probability map.
    assert "<image" not in svg_path.read_text(encoding="utf-8")


def parameter_snapshot(model: TinyXORMLP) -> dict[str, object]:
    return {
        "hidden_weight": model.hidden.weight.detach().tolist(),
        "hidden_bias": model.hidden.bias.detach().tolist(),
        "output_weight": model.output.weight.detach().tolist(),
        "output_bias": model.output.bias.detach().tolist(),
    }


def write_evidence(
    cloned_model: TinyXORMLP,
    cloned_metrics: RunMetrics,
    independent_model: TinyXORMLP,
    independent_metrics: RunMetrics,
) -> None:
    evidence = {
        "experiment": "hidden-unit symmetry on four-Gaussian XOR",
        "configuration": {
            "centers": [[-1, -1], [-1, 1], [1, -1], [1, 1]],
            "labels": [0, 1, 1, 0],
            "points_per_blob": POINTS_PER_BLOB,
            "sigma": SIGMA,
            "data_seed": DATA_SEED,
            "architecture": "2 -> 4 ReLU -> 1",
            "hidden_initialization": "kaiming_normal_(mode='fan_in', nonlinearity='relu')",
            "output_initialization": "xavier_normal_",
            "bias_initialization": "zeros",
            "weight_seed": WEIGHT_SEED,
            "loss": "BCEWithLogits (full batch)",
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "updates": STEPS,
        },
        "cloned": {
            "metrics": asdict(cloned_metrics),
            "final_parameters": parameter_snapshot(cloned_model),
        },
        "independent": {
            "metrics": asdict(independent_metrics),
            "final_parameters": parameter_snapshot(independent_model),
        },
    }
    path = EVIDENCE / "symmetry_experiment.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_metrics(name: str, metrics: RunMetrics) -> None:
    print(
        f"{name:11s}  accuracy={100 * metrics.accuracy:6.2f}%  "
        f"BCE={metrics.loss:.8f}  activation_rank={metrics.activation_rank}  "
        f"initial_spread={metrics.initial_parameter_spread:.1f}  "
        f"final_spread={metrics.final_parameter_spread:.1f}"
    )


def main() -> None:
    x, y = make_xor_data()
    cloned_model, cloned_metrics = train_one(x, y, cloned=True)
    independent_model, independent_metrics = train_one(x, y, cloned=False)
    assert_audited_results(cloned_metrics, independent_metrics)

    make_figure(
        x,
        y,
        cloned_model,
        cloned_metrics,
        independent_model,
        independent_metrics,
    )
    write_evidence(
        cloned_model,
        cloned_metrics,
        independent_model,
        independent_metrics,
    )

    print_metrics("cloned", cloned_metrics)
    print_metrics("independent", independent_metrics)
    print(f"figure  {FIGURES / 'symmetry_decision_boundaries.svg'}")
    print(f"figure  {FIGURES / 'symmetry_decision_boundaries.png'}")
    print(f"evidence {EVIDENCE / 'symmetry_experiment.json'}")


if __name__ == "__main__":
    main()
