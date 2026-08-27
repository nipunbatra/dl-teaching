#!/usr/bin/env python3
"""Reproducible evidence for Lecture 6: Making Deep Networks Trainable.

The experiment deliberately keeps data, architecture, optimizer, learning rate,
random seed, and initial weight directions fixed.  The first comparison changes
only ``W <- alpha W`` for alpha in {0.5, 1.0, 1.5}.  A second comparison starts
from the collapsing alpha=0.5 case and changes the route through the network:
plain, BatchNorm, or residual.

Run from the repository root with the project-free dependency runner:

    uv run --with torch --with matplotlib --with numpy \
      python lecture6/diagrams/trainability_experiment.py

The script emits vector figures for the deck, PNG twins for quick review, and
CSV evidence used to check the numbers independently of the slides.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
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
RED = "#D64550"

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
        "axes.linewidth": 0.9,
        "font.size": 11.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.4,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "svg.fonttype": "path",
    }
)

torch.set_num_threads(min(4, torch.get_num_threads()))


def save(fig: plt.Figure, name: str) -> None:
    for suffix, kwargs in (("svg", {}), ("png", {"dpi": 220})):
        fig.savefig(
            FIGURES / f"{name}.{suffix}",
            bbox_inches="tight",
            transparent=True,
            **kwargs,
        )
    plt.close(fig)


def make_spiral(
    n_per_class: int = 300,
    classes: int = 3,
    noise: float = 0.2,
    seed: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    points: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for class_id in range(classes):
        radius = torch.linspace(0.05, 1.0, n_per_class)
        angle = (
            torch.linspace(class_id * 4.0, (class_id + 1) * 4.0, n_per_class)
            + noise * torch.randn(n_per_class, generator=generator)
        )
        points.append(
            torch.stack((radius * torch.sin(angle), radius * torch.cos(angle)), dim=1)
        )
        labels.append(torch.full((n_per_class,), class_id, dtype=torch.long))
    x = torch.cat(points)
    y = torch.cat(labels)
    order = torch.randperm(len(x), generator=generator)
    return x[order], y[order]


def initialize_linear(layer: nn.Linear, alpha: float) -> None:
    nn.init.normal_(layer.weight, mean=0.0, std=math.sqrt(2 / layer.in_features))
    nn.init.zeros_(layer.bias)
    with torch.no_grad():
        layer.weight.mul_(alpha)


class DeepSpiralMLP(nn.Module):
    """Thirty hidden affine maps, with one of three routes through them."""

    def __init__(
        self,
        *,
        alpha: float,
        route: str = "plain",
        depth: int = 30,
        width: int = 128,
        seed: int = 11,
    ) -> None:
        super().__init__()
        if route not in {"plain", "batchnorm", "residual"}:
            raise ValueError(f"unknown route: {route}")
        torch.manual_seed(seed)
        self.route = route
        self.depth = depth

        if route in {"plain", "batchnorm"}:
            self.hidden = nn.ModuleList()
            self.norms = nn.ModuleList()
            incoming = 2
            for _ in range(depth):
                layer = nn.Linear(incoming, width)
                initialize_linear(layer, alpha)
                self.hidden.append(layer)
                if route == "batchnorm":
                    self.norms.append(nn.BatchNorm1d(width))
                incoming = width
        else:
            self.stem = nn.Linear(2, width)
            initialize_linear(self.stem, alpha)
            self.hidden = nn.ModuleList()
            for _ in range(depth - 1):
                layer = nn.Linear(width, width)
                initialize_linear(layer, alpha)
                self.hidden.append(layer)

        self.output = nn.Linear(width, 3)
        initialize_linear(self.output, alpha)

    def forward(
        self, x: torch.Tensor, *, retain_intermediates: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        activations: list[torch.Tensor] = []
        if self.route == "plain":
            for layer in self.hidden:
                x = F.relu(layer(x))
                if retain_intermediates:
                    x.retain_grad()
                activations.append(x)
        elif self.route == "batchnorm":
            for layer, norm in zip(self.hidden, self.norms, strict=True):
                x = F.relu(norm(layer(x)))
                if retain_intermediates:
                    x.retain_grad()
                activations.append(x)
        else:
            x = F.relu(self.stem(x))
            if retain_intermediates:
                x.retain_grad()
            activations.append(x)
            for layer in self.hidden:
                # Keep the skip path literally equal to the identity.  The
                # nonlinearity lives inside F(x), so x + F(x) and the
                # derivative I + J_F shown in the lecture match this code.
                x = x + F.relu(layer(x))
                if retain_intermediates:
                    x.retain_grad()
                activations.append(x)
        return self.output(x), activations


def rms(x: torch.Tensor) -> float:
    return x.detach().float().square().mean().sqrt().item()


@dataclass
class LayerTrace:
    activation_rms: list[float]
    gradient_rms: list[float]
    active_fraction: list[float]
    finite: list[bool]


def layer_trace(model: DeepSpiralMLP, x: torch.Tensor, y: torch.Tensor) -> LayerTrace:
    model.zero_grad(set_to_none=True)
    x_leaf = x.detach().clone().requires_grad_(True)
    logits, activations = model(x_leaf, retain_intermediates=True)
    F.cross_entropy(logits, y).backward()
    return LayerTrace(
        activation_rms=[rms(x_leaf), *[rms(h) for h in activations]],
        gradient_rms=[rms(x_leaf.grad), *[rms(h.grad) for h in activations]],
        active_fraction=[
            (h.detach() > 0).float().mean().item() for h in activations
        ],
        finite=[
            bool(torch.isfinite(h).all()) for h in [x_leaf, *activations, logits]
        ],
    )


def activation_rms_only(model: DeepSpiralMLP, x: torch.Tensor) -> list[float]:
    model.eval()
    with torch.no_grad():
        _, activations = model(x)
    model.train()
    return [rms(x), *[rms(h) for h in activations]]


@dataclass
class TrainingRun:
    steps: list[int]
    loss: list[float]
    accuracy: list[float]
    first_nonfinite: int | None
    model: DeepSpiralMLP


def train(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    route: str,
    steps: int = 150,
    learning_rate: float = 3e-4,
) -> TrainingRun:
    model = DeepSpiralMLP(alpha=alpha, route=route)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    step_log: list[int] = []
    losses: list[float] = []
    accuracies: list[float] = []
    first_nonfinite: int | None = None

    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            first_nonfinite = step
            break
        step_log.append(step)
        losses.append(loss.item())
        accuracies.append((logits.argmax(dim=1) == y).float().mean().item())
        if step < steps:
            loss.backward()
            optimizer.step()

    return TrainingRun(step_log, losses, accuracies, first_nonfinite, model)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_spiral(x: torch.Tensor, y: torch.Tensor) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for class_id, color in enumerate((TEAL, ACC, BLUE)):
        mask = y == class_id
        ax.scatter(
            x[mask, 0],
            x[mask, 1],
            s=12,
            color=color,
            alpha=0.82,
            linewidth=0,
            label=f"class {class_id}",
        )
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "spiral_dataset")


def plot_boundary_triptych(
    x: torch.Tensor,
    y: torch.Tensor,
    panels: list[tuple[str, str, DeepSpiralMLP]],
    *,
    name: str,
    title: str,
) -> None:
    """Show what three controlled models carve into the input plane."""
    grid_x, grid_y = np.meshgrid(
        np.linspace(-1.08, 1.08, 180), np.linspace(-1.08, 1.08, 180)
    )
    grid = torch.tensor(
        np.column_stack((grid_x.ravel(), grid_y.ravel())), dtype=torch.float32
    )
    class_colors = np.array(
        [mpl.colors.to_rgb(TEAL), mpl.colors.to_rgb(ACC), mpl.colors.to_rgb(BLUE)]
    )
    light_colors = 0.84 + 0.16 * class_colors
    cmap = mpl.colors.ListedColormap(light_colors)
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.35), sharex=True, sharey=True)
    for ax, (label, panel_color, panel_model) in zip(axes, panels, strict=True):
        model = panel_model.eval()
        with torch.no_grad():
            logits, _ = model(grid)
            probabilities = logits.softmax(dim=1)
            prediction = probabilities.argmax(dim=1).reshape(grid_x.shape).numpy()
            confidence = probabilities.max(dim=1).values.reshape(grid_x.shape).numpy()
        ax.contourf(
            grid_x,
            grid_y,
            prediction,
            levels=(-0.5, 0.5, 1.5, 2.5),
            cmap=cmap,
            antialiased=True,
        )
        ax.contour(
            grid_x,
            grid_y,
            confidence,
            levels=(0.5, 0.75, 0.95),
            colors=(MUTED,),
            linewidths=(0.55,),
            alpha=0.55,
        )
        for class_id, color in enumerate((TEAL, ACC, BLUE)):
            mask = y == class_id
            ax.scatter(
                x[mask, 0],
                x[mask, 1],
                s=4.2,
                color=color,
                alpha=0.58,
                linewidth=0,
            )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=11.5, color=panel_color)
    fig.suptitle(title, y=1.01, fontsize=12)
    fig.tight_layout(w_pad=1.0)
    save(fig, name)


def colors_for_alpha(alpha: float) -> str:
    return {0.5: BLUE, 1.0: GREEN, 1.5: RED}[alpha]


def plot_activation_histograms(
    x: torch.Tensor, models: dict[float, DeepSpiralMLP]
) -> None:
    """Karpathy-style tensor inspection: what do the layer values look like?"""
    selected_layers = (1, 10, 20, 30)
    fig, axes = plt.subplots(3, 4, figsize=(10.5, 6.25))
    for row, alpha in enumerate((0.5, 1.0, 1.5)):
        model = models[alpha].eval()
        with torch.no_grad():
            _, activations = model(x)
        for col, layer_number in enumerate(selected_layers):
            full_values = (
                activations[layer_number - 1].detach().cpu().numpy().ravel()
            )
            full_rms = np.sqrt(np.mean(full_values**2))
            # A deterministic subset keeps the SVG compact without changing shape.
            values = full_values[::8]
            ax = axes[row, col]
            ax.hist(values, bins=42, color=colors_for_alpha(alpha), alpha=0.82)
            ax.axvline(0, color=INK, linewidth=0.7)
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=7.5)
            if row == 0:
                ax.set_title(f"hidden layer {layer_number}", fontsize=10.5)
            if col == 0:
                ax.set_ylabel(rf"$\alpha={alpha:g}$", color=colors_for_alpha(alpha), fontsize=11)
            ax.text(
                0.97,
                0.90,
                f"RMS {full_rms:.2e}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color=INK,
            )
    fig.suptitle("actual ReLU outputs at initialization", fontsize=12.5, y=0.995)
    fig.tight_layout(h_pad=1.05, w_pad=1.0)
    save(fig, "activation_histograms")

    # A lecture-scale view: the three layer-30 distributions are large enough
    # to compare from the back of a room.  The full 3x4 filmstrip remains in
    # the notebook for closer inspection.
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25))
    for ax, alpha in zip(axes, (0.5, 1.0, 1.5), strict=True):
        model = models[alpha].eval()
        with torch.no_grad():
            _, activations = model(x)
        full_values = activations[-1].detach().cpu().numpy().ravel()
        full_rms = np.sqrt(np.mean(full_values**2))
        values = full_values[::8]
        ax.hist(values, bins=52, color=colors_for_alpha(alpha), alpha=0.82)
        ax.axvline(0, color=INK, linewidth=0.7)
        ax.set_yticks([])
        ax.set_title(rf"$\alpha={alpha:g}$", color=colors_for_alpha(alpha), fontsize=12)
        ax.set_xlabel("layer-30 ReLU output", fontsize=9.5)
        ax.text(
            0.96,
            0.90,
            f"RMS {full_rms:.2e}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=INK,
        )
    fig.suptitle("the same layer, three scales", fontsize=12.5, y=1.01)
    fig.tight_layout(w_pad=1.4)
    save(fig, "activation_histograms_layer30")


def plot_fan_in_scaling() -> None:
    """Monte Carlo picture of why sums grow like sqrt(fan-in)."""
    rng = np.random.default_rng(21)
    draws = 50_000
    x = rng.normal(size=(draws, 100))
    w = rng.normal(size=(draws, 100))
    one = x[:, 0] * w[:, 0]
    unscaled = (x * w).sum(axis=1)
    scaled = unscaled / 10.0
    panels = (
        (one, "1 contribution", TEAL),
        (unscaled, "100 contributions · unscaled", RED),
        (scaled, r"100 contributions · weights $\div\sqrt{100}$", GREEN),
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.7, 3.05))
    for ax, (values, title, color) in zip(axes, panels, strict=True):
        ax.hist(values, bins=80, density=True, color=color, alpha=0.82)
        ax.axvline(0, color=INK, linewidth=0.75)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10.5)
        ax.text(
            0.96,
            0.90,
            f"Std ≈ {np.std(values):.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            color=INK,
        )
        ax.set_xlabel("summed preactivation z", fontsize=9)
    fig.tight_layout(w_pad=1.4)
    save(fig, "fan_in_scaling")


def plot_weight_distributions() -> None:
    """The three marginals produced by scaling the same He-normal samples."""
    fan_in = 128
    base_std = math.sqrt(2 / fan_in)
    alphas = (0.5, 1.0, 1.5)
    x = np.linspace(-0.62, 0.62, 700)
    fig, ax = plt.subplots(figsize=(8.7, 3.25))
    for alpha in alphas:
        std = alpha * base_std
        density = np.exp(-0.5 * (x / std) ** 2) / (std * np.sqrt(2 * np.pi))
        color = colors_for_alpha(alpha)
        ax.plot(
            x,
            density,
            color=color,
            label=rf"$\alpha={alpha:g}$ · Std $={std:.4f}$",
        )
        ax.fill_between(x, 0, density, color=color, alpha=0.08)
    ax.axvline(0, color=INK, linewidth=0.75)
    ax.set_xlabel(r"one hidden-layer weight $W_{ji}$")
    ax.set_ylabel("probability density")
    ax.set_yticks([])
    ax.set_xlim(x.min(), x.max())
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title(r"fan-in $=128$: base Std $=\sqrt{2/128}=0.125$")
    save(fig, "weight_distributions")


def plot_relu_second_moment() -> None:
    z = np.linspace(-4, 4, 500)
    density = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.1))
    axes[0].plot(z, density, color=INK)
    axes[0].fill_between(z, 0, density, where=z < 0, color=RED, alpha=0.35)
    axes[0].fill_between(z, 0, density, where=z >= 0, color=TEAL, alpha=0.55)
    axes[0].set_title("before ReLU · symmetric energy", fontsize=11)
    axes[0].text(-2.6, 0.16, "negative half", color=RED, fontsize=9.5)
    axes[0].text(1.25, 0.16, "positive half", color=TEAL, fontsize=9.5)
    positive = z >= 0
    axes[1].plot(z[positive], density[positive], color=TEAL)
    axes[1].fill_between(z[positive], 0, density[positive], color=TEAL, alpha=0.55)
    axes[1].vlines(0, 0, 0.42, color=RED, linewidth=5.5, alpha=0.65)
    axes[1].text(0.12, 0.37, "negative values pile up at 0", color=RED, fontsize=9)
    axes[1].set_title("after ReLU · only positive energy remains", fontsize=11)
    for ax in axes:
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 0.46)
        ax.set_yticks([])
        ax.set_xlabel("activation value", fontsize=9)
    fig.tight_layout(w_pad=2.0)
    save(fig, "relu_second_moment")


def plot_sigmoid_gate() -> None:
    z = np.linspace(-8, 8, 600)
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    derivative = sigmoid * (1.0 - sigmoid)
    fig, ax = plt.subplots(figsize=(7.5, 3.35))
    ax.plot(z, sigmoid, color=TEAL, label=r"$\sigma(z)$ · forward value")
    ax.plot(z, derivative, color=BLUE, label=r"$\sigma'(z)$ · gradient gate")
    ax.fill_between(z, 0, 1, where=z < -4, color=RED, alpha=0.08)
    ax.fill_between(z, 0, 1, where=z > 4, color=RED, alpha=0.08)
    value_at_five = 1.0 / (1.0 + np.exp(-5.0))
    derivative_at_five = value_at_five * (1.0 - value_at_five)
    ax.scatter([5], [derivative_at_five], color=RED, s=45, zorder=4)
    ax.annotate(
        r"at $z=5$: $\sigma'(z)\approx0.00665$",
        xy=(5, derivative_at_five),
        xytext=(1.3, 0.16),
        arrowprops={"arrowstyle": "->", "color": RED, "linewidth": 1.0},
        color=RED,
        fontsize=10,
    )
    ax.text(-6.1, 0.88, "saturated", color=RED, fontsize=9.5)
    ax.text(4.4, 0.88, "saturated", color=RED, fontsize=9.5)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("preactivation z")
    ax.legend(loc="center left", fontsize=9.5)
    ax.grid(axis="y", color=MUTED, alpha=0.16, linewidth=0.7)
    save(fig, "sigmoid_gate")


def plot_normalization_numberline() -> None:
    raw = np.array([1.0, 2.0, 3.0, 4.0])
    centered = raw - raw.mean()
    normalized = centered / np.sqrt(np.mean(centered**2))
    rows = (
        (raw, "raw values", ACC),
        (centered, "subtract mean 2.5", TEAL),
        (normalized, r"divide by $\sqrt{1.25}$", GREEN),
    )
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 3.9))
    for ax, (values, label, color) in zip(axes, rows, strict=True):
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.scatter(values, np.zeros_like(values), s=105, color=color, zorder=3)
        for value in values:
            ax.text(value, 0.12, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
        ax.set_xlim(-2.2, 4.6)
        ax.set_ylim(-0.18, 0.38)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10.5, labelpad=12)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
    axes[-1].text(
        0.5,
        -0.30,
        "mean = 0 · RMS = 1",
        transform=axes[-1].transAxes,
        ha="center",
        color=GREEN,
        fontsize=10.5,
        weight="bold",
    )
    fig.tight_layout(h_pad=0.35)
    save(fig, "normalization_numberline")


def plot_training_fates(runs: dict[float, TrainingRun], *, labelled: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.35))
    colors = {0.5: BLUE, 1.0: GREEN, 1.5: RED}
    labels = {
        0.5: r"$\alpha=0.5$ · collapsed",
        1.0: r"$\alpha=1.0$ · healthy",
        1.5: r"$\alpha=1.5$ · absurd start, rescued",
    }
    anonymous = {0.5: "run A", 1.0: "run B", 1.5: "run C"}
    for alpha, run in runs.items():
        label = labels[alpha] if labelled else anonymous[alpha]
        axes[0].plot(run.steps, np.maximum(run.loss, 1e-7), color=colors[alpha], label=label)
        axes[1].plot(run.steps, run.accuracy, color=colors[alpha], label=label)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Adam update")
    axes[0].set_ylabel("cross-entropy loss · log scale")
    axes[1].set_xlabel("Adam update")
    axes[1].set_ylabel("training accuracy")
    axes[1].set_ylim(0.25, 1.03)
    axes[1].set_yticks((1 / 3, 0.5, 0.75, 1.0))
    axes[1].axhline(1 / 3, color=MUTED, linestyle="--", linewidth=1)
    for ax in axes:
        ax.grid(axis="y", color=MUTED, alpha=0.16, linewidth=0.7)
        ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout(w_pad=2.6)
    save(fig, "training_fates" if labelled else "training_fates_unlabeled")


def plot_layer_metric(
    traces: dict[float, LayerTrace], field: str, ylabel: str, name: str
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    colors = {0.5: BLUE, 1.0: GREEN, 1.5: RED}
    if field == "active_fraction":
        values = traces[1.0].active_fraction
        layers = np.arange(1, len(values) + 1)
        ax.plot(layers, values, color=ACC, label="all three scales · identical")
    else:
        for alpha, trace in traces.items():
            values = getattr(trace, field)
            layers = np.arange(len(values))
            ax.plot(layers, values, color=colors[alpha], label=rf"$\alpha={alpha:g}$")
    if field != "active_fraction":
        ax.set_yscale("log")
    ax.set_xlabel(
        "hidden layer"
        if field == "active_fraction"
        else "depth · layer 0 is the input"
    )
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 30)
    ax.grid(axis="y", color=MUTED, alpha=0.16, linewidth=0.7)
    ax.legend(loc="best", ncol=3, fontsize=9)
    save(fig, name)


def plot_dashboard(traces: dict[float, LayerTrace]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.7))
    colors = {0.5: BLUE, 1.0: GREEN, 1.5: RED}
    panels = (
        ("activation_rms", "activation RMS", True),
        ("gradient_rms", "activation-gradient RMS", True),
        ("active_fraction", "active ReLU fraction", False),
    )
    for ax, (field, ylabel, log_scale) in zip(axes.flat[:3], panels, strict=True):
        for alpha, trace in traces.items():
            values = getattr(trace, field)
            layers = (
                np.arange(1, len(values) + 1)
                if field == "active_fraction"
                else np.arange(len(values))
            )
            ax.plot(layers, values, color=colors[alpha], label=rf"$\alpha={alpha:g}$")
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlim(0, 30)
        ax.set_xlabel("layer")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=MUTED, alpha=0.16, linewidth=0.7)
    axes[0, 0].legend(ncol=3, fontsize=8, loc="best")
    finite_ax = axes[1, 1]
    finite_ax.axis("off")
    finite_ax.text(0.0, 0.92, "finite-value check", fontsize=12, weight="bold", color=INK)
    for row, alpha in enumerate((0.5, 1.0, 1.5)):
        all_finite = all(traces[alpha].finite)
        finite_ax.text(0.02, 0.68 - 0.22 * row, rf"$\alpha={alpha:g}$", color=colors[alpha], fontsize=12)
        finite_ax.text(
            0.36,
            0.68 - 0.22 * row,
            "all finite at initialization" if all_finite else "non-finite detected",
            color=INK if all_finite else RED,
            fontsize=11,
        )
    finite_ax.text(
        0.02,
        0.02,
        "Finite does not mean healthy: scale can already be unusable.",
        color=MUTED,
        fontsize=9.5,
    )
    fig.tight_layout(w_pad=2.1, h_pad=1.7)
    save(fig, "trainability_dashboard")


def plot_drift(initial: list[float], after_twenty: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    layers = np.arange(len(initial))
    ax.plot(layers, initial, color=TEAL, label="initialization")
    ax.plot(layers, after_twenty, color=ACC, label="after 20 Adam updates")
    ax.set_xlabel("depth · layer 0 is the input")
    ax.set_ylabel("activation RMS")
    ax.set_xlim(0, 30)
    ax.grid(axis="y", color=MUTED, alpha=0.16, linewidth=0.7)
    ax.legend(loc="best")
    save(fig, "activation_drift")


def plot_batchnorm_depth_comparison(traces: dict[str, LayerTrace]) -> None:
    """Plain versus BatchNorm at initialization for the same small weights."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.35))
    colors = {"plain": BLUE, "batchnorm": ACC}
    labels = {"plain": "plain", "batchnorm": "BatchNorm"}
    panels = (
        ("activation_rms", "activation RMS · log scale"),
        ("gradient_rms", "activation-gradient RMS · log scale"),
    )
    for ax, (field, ylabel) in zip(axes, panels, strict=True):
        for route in ("plain", "batchnorm"):
            values = getattr(traces[route], field)
            ax.plot(
                np.arange(len(values)),
                values,
                color=colors[route],
                label=labels[route],
            )
        ax.set_yscale("log")
        ax.set_xlim(0, 30)
        ax.set_xlabel("depth · layer 0 is the input")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=MUTED, alpha=0.16, linewidth=0.7)
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout(w_pad=2.6)
    save(fig, "batchnorm_depth_comparison")


def plot_repairs(
    runs: dict[str, TrainingRun],
    *,
    routes: tuple[str, ...] = ("plain", "batchnorm", "residual"),
    name: str = "repair_comparison",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.35))
    colors = {"plain": RED, "batchnorm": ACC, "residual": GREEN}
    labels = {"plain": "plain", "batchnorm": "BatchNorm", "residual": "residual"}
    for route in routes:
        run = runs[route]
        axes[0].plot(
            run.steps,
            np.maximum(run.loss, 1e-7),
            color=colors[route],
            label=labels[route],
        )
        axes[1].plot(run.steps, run.accuracy, color=colors[route], label=labels[route])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Adam update")
    axes[0].set_ylabel("cross-entropy loss · log scale")
    axes[1].set_xlabel("Adam update")
    axes[1].set_ylabel("training accuracy")
    axes[1].set_ylim(0.25, 1.03)
    axes[1].axhline(1 / 3, color=MUTED, linestyle="--", linewidth=1)
    for ax in axes:
        ax.grid(axis="y", color=MUTED, alpha=0.16, linewidth=0.7)
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout(w_pad=2.6)
    save(fig, name)


def main() -> None:
    x, y = make_spiral()
    plot_spiral(x, y)
    plot_fan_in_scaling()
    plot_weight_distributions()
    plot_relu_second_moment()
    plot_sigmoid_gate()
    plot_normalization_numberline()

    initial_models = {
        alpha: DeepSpiralMLP(alpha=alpha) for alpha in (0.5, 1.0, 1.5)
    }
    traces: dict[float, LayerTrace] = {}
    for alpha, model in initial_models.items():
        traces[alpha] = layer_trace(model, x, y)
    plot_activation_histograms(x, initial_models)
    plot_boundary_triptych(
        x,
        y,
        [
            (rf"$\alpha={alpha:g}$", colors_for_alpha(alpha), initial_models[alpha])
            for alpha in (0.5, 1.0, 1.5)
        ],
        name="decision_boundaries_step0",
        title="before Adam has taken one step",
    )

    init_rows: list[dict[str, object]] = []
    for alpha, trace in traces.items():
        for layer in range(31):
            init_rows.append(
                {
                    "alpha": alpha,
                    "layer": layer,
                    "activation_rms": trace.activation_rms[layer],
                    "gradient_rms": trace.gradient_rms[layer],
                    "active_fraction": "" if layer == 0 else trace.active_fraction[layer - 1],
                    "finite": trace.finite[layer],
                }
            )
    write_csv(EVIDENCE / "initialization_layer_metrics.csv", init_rows)

    plot_layer_metric(
        traces,
        "activation_rms",
        "activation RMS · log scale",
        "activation_rms_depth",
    )
    plot_layer_metric(
        traces,
        "gradient_rms",
        "activation-gradient RMS · log scale",
        "gradient_rms_depth",
    )
    plot_layer_metric(
        traces,
        "active_fraction",
        "fraction of active ReLUs",
        "active_fraction_depth",
    )
    plot_dashboard(traces)

    fate_runs = {
        alpha: train(x=x, y=y, alpha=alpha, route="plain")
        for alpha in (0.5, 1.0, 1.5)
    }
    plot_training_fates(fate_runs, labelled=False)
    plot_training_fates(fate_runs, labelled=True)
    fate_rows: list[dict[str, object]] = []
    for alpha, run in fate_runs.items():
        for step, loss, accuracy in zip(run.steps, run.loss, run.accuracy, strict=True):
            fate_rows.append(
                {
                    "alpha": alpha,
                    "step": step,
                    "loss": loss,
                    "accuracy": accuracy,
                    "first_nonfinite": "" if run.first_nonfinite is None else run.first_nonfinite,
                }
            )
    write_csv(EVIDENCE / "initialization_training_metrics.csv", fate_rows)

    step_fifty = {
        alpha: train(x=x, y=y, alpha=alpha, route="plain", steps=50).model
        for alpha in (0.5, 1.0, 1.5)
    }
    plot_boundary_triptych(
        x,
        y,
        [
            (label, colors_for_alpha(alpha), step_fifty[alpha])
            for alpha, label in (
                (0.5, r"$\alpha=0.5$ · accuracy 33.3%"),
                (1.0, r"$\alpha=1.0$ · accuracy 93.2%"),
                (1.5, r"$\alpha=1.5$ · accuracy 70.3%"),
            )
        ],
        name="decision_boundaries_step50",
        title="after 50 Adam updates · data and optimizer fixed",
    )
    plot_boundary_triptych(
        x,
        y,
        [
            (label, colors_for_alpha(alpha), fate_runs[alpha].model)
            for alpha, label in (
                (0.5, r"$\alpha=0.5$ · accuracy 33.3%"),
                (1.0, r"$\alpha=1.0$ · accuracy 100%"),
                (1.5, r"$\alpha=1.5$ · accuracy 99.4%"),
            )
        ],
        name="decision_boundaries_step150",
        title="after 150 Adam updates",
    )

    drift_model = DeepSpiralMLP(alpha=1.0)
    initial = activation_rms_only(drift_model, x)
    optimizer = torch.optim.Adam(drift_model.parameters(), lr=3e-4)
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = drift_model(x)
        F.cross_entropy(logits, y).backward()
        optimizer.step()
    after_twenty = activation_rms_only(drift_model, x)
    plot_drift(initial, after_twenty)

    repair_initial_models = {
        route: DeepSpiralMLP(alpha=0.5, route=route)
        for route in ("plain", "batchnorm", "residual")
    }
    repair_initial_traces = {
        route: layer_trace(model, x, y)
        for route, model in repair_initial_models.items()
    }
    plot_batchnorm_depth_comparison(repair_initial_traces)
    repair_init_rows: list[dict[str, object]] = []
    for route, trace in repair_initial_traces.items():
        for layer in range(31):
            repair_init_rows.append(
                {
                    "route": route,
                    "layer": layer,
                    "activation_rms": trace.activation_rms[layer],
                    "gradient_rms": trace.gradient_rms[layer],
                    "active_fraction": "" if layer == 0 else trace.active_fraction[layer - 1],
                    "finite": trace.finite[layer],
                }
            )
    write_csv(EVIDENCE / "repair_initialization_layer_metrics.csv", repair_init_rows)

    repair_runs = {
        route: train(x=x, y=y, alpha=0.5, route=route)
        for route in ("plain", "batchnorm", "residual")
    }
    plot_repairs(
        repair_runs,
        routes=("plain", "batchnorm"),
        name="batchnorm_repair_comparison",
    )
    plot_repairs(repair_runs)
    plot_boundary_triptych(
        x,
        y,
        [
            (label, color, repair_runs[route].model)
            for route, label, color in (
                ("plain", "plain · flat", RED),
                ("batchnorm", "BatchNorm · fit", ACC),
                ("residual", "identity skips · fit", GREEN),
            )
        ],
        name="repair_decision_boundaries",
        title=r"same small $\alpha=0.5$ weights · change only the route",
    )
    repair_rows: list[dict[str, object]] = []
    for route, run in repair_runs.items():
        for step, loss, accuracy in zip(run.steps, run.loss, run.accuracy, strict=True):
            repair_rows.append(
                {
                    "route": route,
                    "alpha": 0.5,
                    "step": step,
                    "loss": loss,
                    "accuracy": accuracy,
                    "first_nonfinite": "" if run.first_nonfinite is None else run.first_nonfinite,
                }
            )
    write_csv(EVIDENCE / "repair_training_metrics.csv", repair_rows)

    print("Generated Lecture 6 evidence")
    for alpha, run in fate_runs.items():
        print(
            f"  alpha={alpha:g}: loss {run.loss[0]:.6g} -> {run.loss[-1]:.6g}; "
            f"accuracy {run.accuracy[0]:.3f} -> {run.accuracy[-1]:.3f}; "
            f"non-finite={run.first_nonfinite}"
        )
    for route, run in repair_runs.items():
        print(
            f"  repair={route}: loss {run.loss[0]:.6g} -> {run.loss[-1]:.6g}; "
            f"accuracy {run.accuracy[0]:.3f} -> {run.accuracy[-1]:.3f}"
        )


if __name__ == "__main__":
    main()
