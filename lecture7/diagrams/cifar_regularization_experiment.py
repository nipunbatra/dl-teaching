#!/usr/bin/env python3
"""Small, reproducible CIFAR-10 regularization comparison for Lecture 7.

The experiment is intentionally designed as a teaching example, not as a
state-of-the-art benchmark.  It uses only 2,000 stratified training images so
that a fixed CNN visibly overfits, reserves 5,000 *different* images from the
official training set for validation, and leaves the official test set
untouched until a checkpoint has been selected by minimum validation loss.

Five conditions share the same split, initial weights, minibatch order,
architecture, learning rate, and number of updates.  Only one ingredient is
changed at a time:

* baseline: no explicit regularizer;
* weight_decay: decoupled AdamW weight decay (matrix/kernel weights only);
* augmentation: reflection-padded random crop plus horizontal flip;
* dropout: spatial dropout in the convolutional trunk and dropout in the MLP;
* label_smoothing: cross-entropy with epsilon=0.1 during training only.

Run from the repository root:

    uv run --with 'torch>=2.2' --with 'matplotlib>=3.8' --with 'numpy>=1.26' \
      python lecture7/diagrams/cifar_regularization_experiment.py

Outputs:

* lecture7/evidence/cifar_regularization_config.json
* lecture7/evidence/cifar_regularization_epoch_metrics.csv
* lecture7/evidence/cifar_regularization_summary.{json,csv}
* lecture7/evidence/cifar_regularization_split_indices.npz
* lecture7/figures/cifar_regularization_{curves,summary}.{svg,png}

The plots and summary explicitly identify this as a single-seed descriptive
experiment.  Test accuracy is never used for tuning or checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path(
    "/Users/nipun/git/stt-ai-teaching/lecture-demos/week05/data/"
    "cifar-10-batches-py"
)
DEFAULT_FIGURES = ROOT / "lecture7" / "figures"
DEFAULT_EVIDENCE = ROOT / "lecture7" / "evidence"

# Published CIFAR-10 channel statistics.  Augmentation is applied to [0, 1]
# images first; normalization is then identical in every condition.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

SPLIT_SEED = 20260827
MODEL_SEED = 1701
ORDER_SEED = 314159
AUGMENT_SEED = 271828

# Lecture palette (shared with lecture7/diagrams/l7_figs.py).
INK = "#23373B"
ACC = "#EB811B"
TEAL = "#2C7A7B"
GREEN = "#14B03D"
BLUE = "#2B6CB0"
MUTED = "#6E7F82"
RED = "#D64550"

CONDITION_COLORS = {
    "baseline": INK,
    "weight_decay": BLUE,
    "augmentation": GREEN,
    "dropout": ACC,
    "label_smoothing": RED,
}

CONDITION_LABELS = {
    "baseline": "Baseline",
    "weight_decay": "AdamW decay",
    "augmentation": "Crop + flip",
    "dropout": "Dropout",
    "label_smoothing": "Label smoothing",
}

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
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11.5,
        "lines.linewidth": 2.3,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "svg.fonttype": "none",
    }
)


@dataclass(frozen=True)
class Condition:
    name: str
    weight_decay: float = 0.0
    augmentation: bool = False
    conv_dropout: float = 0.0
    classifier_dropout: float = 0.0
    label_smoothing: float = 0.0


def condition_table(args: argparse.Namespace) -> dict[str, Condition]:
    return {
        "baseline": Condition(name="baseline"),
        "weight_decay": Condition(
            name="weight_decay", weight_decay=args.weight_decay
        ),
        "augmentation": Condition(name="augmentation", augmentation=True),
        "dropout": Condition(
            name="dropout",
            conv_dropout=args.conv_dropout,
            classifier_dropout=args.classifier_dropout,
        ),
        "label_smoothing": Condition(
            name="label_smoothing", label_smoothing=args.label_smoothing
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--conv-dropout", type=float, default=0.15)
    parser.add_argument("--classifier-dropout", type=float, default=0.50)
    parser.add_argument("--label-smoothing", type=float, default=0.10)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITION_LABELS),
        choices=list(CONDITION_LABELS),
        help="Subset of conditions to run (main figures are clearest with all five).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="auto prefers Apple MPS when it is available.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="Print progress every N epochs; epoch-level CSV is always complete.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.train_per_class < 1 or args.val_per_class < 1:
        parser.error("split sizes must be positive")
    return args


def select_device(requested: str) -> torch.device:
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if requested == "mps" and not mps_available:
        raise RuntimeError("--device mps requested, but torch reports MPS unavailable")
    if requested == "auto":
        requested = "mps" if mps_available else "cpu"
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="bytes")
    images = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"mismatched image/label counts in {path}")
    return images, labels


def load_cifar10(
    data_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[Path]]:
    training_files = [data_dir / f"data_batch_{index}" for index in range(1, 6)]
    test_file = data_dir / "test_batch"
    required = [*training_files, test_file]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing CIFAR-10 Python batches: " + ", ".join(missing))

    train_batches = [load_cifar_batch(path) for path in training_files]
    train_images = np.concatenate([batch[0] for batch in train_batches])
    train_labels = np.concatenate([batch[1] for batch in train_batches])
    test_images, test_labels = load_cifar_batch(test_file)
    return (
        torch.from_numpy(train_images),
        torch.from_numpy(train_labels),
        torch.from_numpy(test_images),
        torch.from_numpy(test_labels),
        required,
    )


def stratified_split(
    labels: torch.Tensor,
    train_per_class: int,
    val_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels_np = labels.numpy()
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []
    for class_id in range(len(CLASS_NAMES)):
        candidates = np.flatnonzero(labels_np == class_id)
        rng.shuffle(candidates)
        required = train_per_class + val_per_class
        if len(candidates) < required:
            raise ValueError(
                f"class {class_id} has {len(candidates)} examples; need {required}"
            )
        train_indices.append(candidates[:train_per_class])
        val_indices.append(candidates[train_per_class:required])
    train = np.concatenate(train_indices)
    val = np.concatenate(val_indices)
    rng.shuffle(train)
    rng.shuffle(val)
    if set(train.tolist()) & set(val.tolist()):
        raise AssertionError("training and validation splits overlap")
    return train, val


class SmallCifarCNN(nn.Module):
    """A 667k-parameter CNN with dropout modules present in every condition."""

    def __init__(self, conv_dropout: float, classifier_dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(conv_dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(conv_dropout),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(conv_dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, len(CLASS_NAMES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def make_optimizer(
    model: nn.Module, learning_rate: float, weight_decay: float
) -> torch.optim.AdamW:
    # Kernels and matrices decay; biases do not.  There are no normalization
    # parameters in this intentionally plain model.
    decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def normalize(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def random_crop_and_flip(
    images: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Reflection-pad by 4, crop 32x32 independently, then flip with p=0.5."""

    batch_size = images.shape[0]
    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    patches = padded.unfold(2, 32, 1).unfold(3, 32, 1)
    rows = torch.randint(0, 9, (batch_size,), generator=generator).to(device)
    cols = torch.randint(0, 9, (batch_size,), generator=generator).to(device)
    batch_indices = torch.arange(batch_size, device=device)
    cropped = patches[batch_indices, :, rows, cols, :, :]
    flip = (torch.rand(batch_size, generator=generator) < 0.5).to(device)
    return torch.where(flip[:, None, None, None], cropped.flip(-1), cropped)


def prepare_batch(
    raw_images: torch.Tensor,
    device: torch.device,
    *,
    augment: bool,
    augment_generator: torch.Generator | None,
) -> torch.Tensor:
    images = raw_images.to(device=device, dtype=torch.float32).div_(255.0)
    if augment:
        if augment_generator is None:
            raise ValueError("augmentation requested without a generator")
        images = random_crop_and_flip(images, augment_generator, device)
    return normalize(images, device)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        x = prepare_batch(
            images[start:stop], device, augment=False, augment_generator=None
        )
        y = labels[start:stop].to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
    return {
        "loss": total_loss / len(images),
        "accuracy": total_correct / len(images),
    }


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    condition: Condition,
    device: torch.device,
    batch_size: int,
    order_generator: torch.Generator,
    augment_generator: torch.Generator,
) -> dict[str, float]:
    model.train()
    order = torch.randperm(len(images), generator=order_generator)
    objective_sum = 0.0
    correct = 0
    for start in range(0, len(images), batch_size):
        index = order[start : start + batch_size]
        x = prepare_batch(
            images[index],
            device,
            augment=condition.augmentation,
            augment_generator=augment_generator,
        )
        y = labels[index].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(
            logits, y, label_smoothing=condition.label_smoothing
        )
        loss.backward()
        optimizer.step()
        objective_sum += loss.detach().item() * len(index)
        correct += (logits.detach().argmax(dim=1) == y).sum().item()
    return {
        # This is the stochastic training objective.  Cross-condition plots use
        # clean evaluation loss below because augmentation and label smoothing
        # otherwise change what "training loss" means.
        "objective": objective_sum / len(images),
        "stochastic_accuracy": correct / len(images),
    }


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


@torch.inference_mode()
def decayed_parameter_l2(model: nn.Module) -> float:
    """L2 norm of the matrices/kernels to which AdamW decay is applied."""

    squared = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        if parameter.ndim >= 2:
            squared += parameter.float().square().sum()
    return squared.sqrt().item()


def run_condition(
    *,
    condition: Condition,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed_everything(MODEL_SEED)
    model = SmallCifarCNN(
        conv_dropout=condition.conv_dropout,
        classifier_dropout=condition.classifier_dropout,
    ).to(device)
    initial_hash = state_sha256(cpu_state_dict(model))
    optimizer = make_optimizer(model, args.learning_rate, condition.weight_decay)
    order_generator = torch.Generator().manual_seed(ORDER_SEED)
    augment_generator = torch.Generator().manual_seed(AUGMENT_SEED)

    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    run_start = time.perf_counter()

    for epoch in range(args.epochs + 1):
        epoch_start = time.perf_counter()
        if epoch == 0:
            stochastic = {"objective": float("nan"), "stochastic_accuracy": float("nan")}
        else:
            stochastic = train_one_epoch(
                model,
                optimizer,
                train_images,
                train_labels,
                condition,
                device,
                args.batch_size,
                order_generator,
                augment_generator,
            )

        # Every reported train metric is a deterministic clean-input eval with
        # dropout disabled and hard targets.  That keeps the curves comparable.
        train_clean = evaluate(
            model, train_images, train_labels, device, args.eval_batch_size
        )
        validation = evaluate(
            model, val_images, val_labels, device, args.eval_batch_size
        )
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "condition": condition.name,
            "epoch": epoch,
            "train_objective": stochastic["objective"],
            "train_stochastic_accuracy": stochastic["stochastic_accuracy"],
            "train_clean_loss": train_clean["loss"],
            "train_clean_accuracy": train_clean["accuracy"],
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "decayed_parameter_l2": decayed_parameter_l2(model),
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)

        if validation["loss"] < best_val_loss:
            best_epoch = epoch
            best_val_loss = validation["loss"]
            best_state = cpu_state_dict(model)

        if (
            epoch == 0
            or epoch == 1
            or epoch == args.epochs
            or epoch % args.log_every == 0
        ):
            print(
                f"[{condition.name:15s}] epoch {epoch:02d}/{args.epochs}: "
                f"clean train {100 * train_clean['accuracy']:5.1f}% / "
                f"val {100 * validation['accuracy']:5.1f}% / "
                f"val CE {validation['loss']:.3f} / {epoch_seconds:.1f}s",
                flush=True,
            )

    if best_state is None:
        raise AssertionError("no validation checkpoint was recorded")
    final_state = cpu_state_dict(model)
    final_parameter_l2 = decayed_parameter_l2(model)
    model.load_state_dict(best_state)
    selected_parameter_l2 = decayed_parameter_l2(model)
    selected_train = evaluate(
        model, train_images, train_labels, device, args.eval_batch_size
    )
    selected_val = evaluate(model, val_images, val_labels, device, args.eval_batch_size)
    # This is deliberately the only test-set access in the experiment.
    selected_test = evaluate(
        model, test_images, test_labels, device, args.eval_batch_size
    )
    model.load_state_dict(final_state)

    final = history[-1]
    run_seconds = time.perf_counter() - run_start
    summary = {
        "condition": condition.name,
        "condition_label": CONDITION_LABELS[condition.name],
        "settings": asdict(condition),
        "initial_state_sha256": initial_hash,
        "best_epoch_by_val_loss": best_epoch,
        "selected_train_loss": selected_train["loss"],
        "selected_train_accuracy": selected_train["accuracy"],
        "selected_val_loss": selected_val["loss"],
        "selected_val_accuracy": selected_val["accuracy"],
        "selected_test_loss": selected_test["loss"],
        "selected_test_accuracy": selected_test["accuracy"],
        "selected_decayed_parameter_l2": selected_parameter_l2,
        "selected_train_val_accuracy_gap": (
            selected_train["accuracy"] - selected_val["accuracy"]
        ),
        "final_train_loss": final["train_clean_loss"],
        "final_train_accuracy": final["train_clean_accuracy"],
        "final_val_loss": final["val_loss"],
        "final_val_accuracy": final["val_accuracy"],
        "final_decayed_parameter_l2": final_parameter_l2,
        "final_train_val_accuracy_gap": (
            final["train_clean_accuracy"] - final["val_accuracy"]
        ),
        "validation_loss_rise_after_best": final["val_loss"] - selected_val["loss"],
        "epochs_after_best": args.epochs - best_epoch,
        "runtime_seconds": run_seconds,
        "test_accesses": 1,
        "checkpoint_rule": "minimum validation cross-entropy",
    }
    return history, summary


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, figures_dir: Path, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.svg", bbox_inches="tight", transparent=True)
    fig.savefig(
        figures_dir / f"{name}.png",
        bbox_inches="tight",
        transparent=True,
        dpi=220,
    )
    plt.close(fig)


def plot_curves(
    histories: dict[str, list[dict[str, Any]]],
    summaries: dict[str, dict[str, Any]],
    figures_dir: Path,
) -> None:
    if "baseline" not in histories:
        baseline_name = next(iter(histories))
    else:
        baseline_name = "baseline"
    baseline = histories[baseline_name]
    baseline_summary = summaries[baseline_name]
    epochs = np.asarray([row["epoch"] for row in baseline])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.85))
    ax = axes[0]
    ax.plot(
        epochs,
        [row["train_clean_loss"] for row in baseline],
        color=TEAL,
        label="clean training loss",
    )
    ax.plot(
        epochs,
        [row["val_loss"] for row in baseline],
        color=ACC,
        label="validation loss",
    )
    best_epoch = baseline_summary["best_epoch_by_val_loss"]
    best_loss = baseline_summary["selected_val_loss"]
    ax.axvline(best_epoch, color=RED, linestyle="--", linewidth=1.6)
    ax.scatter([best_epoch], [best_loss], s=50, color=RED, zorder=5)
    ax.annotate(
        f"early-stop checkpoint\nepoch {best_epoch}",
        xy=(best_epoch, best_loss),
        xytext=(6, 23),
        textcoords="offset points",
        color=RED,
        fontsize=10.5,
        arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.2},
    )
    ax.set_title("A   Baseline overfits", loc="left", fontweight="bold")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy (clean inputs)")
    ax.set_xlim(0, epochs[-1])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=10.2)

    ax = axes[1]
    for name, history in histories.items():
        x = [row["epoch"] for row in history]
        y = [100 * row["val_accuracy"] for row in history]
        color = CONDITION_COLORS[name]
        ax.plot(x, y, color=color, label=CONDITION_LABELS[name], alpha=0.94)
        best = summaries[name]
        selected_epoch = best["best_epoch_by_val_loss"]
        selected_y = 100 * best["selected_val_accuracy"]
        ax.scatter(
            [selected_epoch],
            [selected_y],
            color=color,
            edgecolor="white",
            linewidth=0.7,
            s=38,
            zorder=5,
        )
    ax.set_title("B   One controlled change per run", loc="left", fontweight="bold")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_xlim(0, max(len(history) - 1 for history in histories.values()))
    ax.legend(loc="lower right", fontsize=9.3)
    ax.grid(axis="y", color=MUTED, alpha=0.14, linewidth=0.8)
    fig.text(
        0.995,
        -0.01,
        "single fixed seed · dots = checkpoint chosen by validation loss · test set not shown",
        ha="right",
        va="top",
        color=MUTED,
        fontsize=9.2,
    )
    fig.tight_layout(w_pad=2.7)
    save_figure(fig, figures_dir, "cifar_regularization_curves")


def plot_summary(
    summaries: dict[str, dict[str, Any]], figures_dir: Path
) -> None:
    order = [name for name in CONDITION_LABELS if name in summaries]
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(
        1, 2, figsize=(11.2, 3.95), gridspec_kw={"width_ratios": (1.18, 1.0)}
    )

    ax = axes[0]
    for position, name in zip(y, order, strict=True):
        result = summaries[name]
        train = 100 * result["selected_train_accuracy"]
        val = 100 * result["selected_val_accuracy"]
        color = CONDITION_COLORS[name]
        ax.plot([val, train], [position, position], color=color, alpha=0.42, lw=3.0)
        ax.scatter([train], [position], facecolor="white", edgecolor=color, s=64, lw=1.8)
        ax.scatter([val], [position], color=color, s=64, zorder=4)
    ax.set_yticks(y, [CONDITION_LABELS[name] for name in order])
    ax.set_xlabel("accuracy at validation-selected checkpoint (%)")
    ax.set_title("A   Train–validation gap at selection", loc="left", fontweight="bold")
    ax.grid(axis="x", color=MUTED, alpha=0.14, linewidth=0.8)
    ax.scatter([], [], facecolor="white", edgecolor=INK, label="clean train")
    ax.scatter([], [], color=INK, label="validation")
    ax.legend(loc="upper right", fontsize=9.5)

    ax = axes[1]
    test = np.asarray([100 * summaries[name]["selected_test_accuracy"] for name in order])
    baseline = 100 * summaries["baseline"]["selected_test_accuracy"] if "baseline" in summaries else test[0]
    colors = [CONDITION_COLORS[name] for name in order]
    bars = ax.barh(y, test, color=colors, height=0.58, alpha=0.92)
    ax.axvline(baseline, color=INK, linestyle="--", lw=1.25, alpha=0.58)
    low = max(0.0, float(test.min()) - 4.0)
    high = min(100.0, float(test.max()) + 5.0)
    ax.set_xlim(low, high)
    ax.set_yticks(y, [CONDITION_LABELS[name] for name in order])
    ax.set_xlabel("test accuracy (%)")
    ax.set_title("B   Test once, after selection", loc="left", fontweight="bold")
    ax.grid(axis="x", color=MUTED, alpha=0.14, linewidth=0.8)
    for bar, accuracy, name in zip(bars, test, order, strict=True):
        epoch = summaries[name]["best_epoch_by_val_loss"]
        ax.text(
            accuracy + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"{accuracy:.1f}%  ·  e{epoch}",
            va="center",
            ha="left",
            fontsize=9.2,
            color=INK,
        )
    fig.text(
        0.995,
        -0.01,
        "2,000 train · 5,000 validation · 10,000 test · one seed (descriptive, not a benchmark)",
        ha="right",
        va="top",
        color=MUTED,
        fontsize=9.2,
    )
    fig.tight_layout(w_pad=2.4)
    save_figure(fig, figures_dir, "cifar_regularization_summary")


def assess_pedagogy(
    summaries: dict[str, dict[str, Any]], epochs: int
) -> dict[str, Any]:
    baseline = summaries.get("baseline")
    if baseline is None:
        return {
            "verdict": "not_assessed",
            "reason": "baseline was not included",
        }

    baseline_overfits = bool(
        baseline["best_epoch_by_val_loss"] <= epochs - 5
        and baseline["validation_loss_rise_after_best"] >= 0.10
        and baseline["final_train_val_accuracy_gap"] >= 0.20
    )
    comparisons: dict[str, Any] = {}
    helpful_count = 0
    for name, result in summaries.items():
        if name == "baseline":
            continue
        val_accuracy_delta = (
            result["selected_val_accuracy"] - baseline["selected_val_accuracy"]
        )
        test_accuracy_delta = (
            result["selected_test_accuracy"] - baseline["selected_test_accuracy"]
        )
        final_gap_reduction = (
            baseline["final_train_val_accuracy_gap"]
            - result["final_train_val_accuracy_gap"]
        )
        # Count a run as pedagogically helpful on validation/generalization
        # evidence only.  The test delta is reported, never used for the verdict.
        helpful = bool(val_accuracy_delta >= 0.01 or final_gap_reduction >= 0.05)
        helpful_count += int(helpful)
        comparisons[name] = {
            "selected_val_accuracy_delta": val_accuracy_delta,
            "selected_test_accuracy_delta_descriptive_only": test_accuracy_delta,
            "final_gap_reduction": final_gap_reduction,
            "helpful_by_predeclared_validation_or_gap_rule": helpful,
        }

    expected_comparisons = max(0, len(summaries) - 1)
    enough_helpful = helpful_count >= min(3, expected_comparisons)
    clean = baseline_overfits and enough_helpful
    return {
        "verdict": "clean_for_teaching" if clean else "mixed_not_clean_enough",
        "predeclared_rule": (
            "baseline best epoch at least 5 epochs early, validation CE rises >=0.10, "
            "and final train-validation accuracy gap >=20 pp; at least 3 regularizers "
            "improve selected validation accuracy >=1 pp or reduce final gap >=5 pp"
        ),
        "baseline_visibly_overfits": baseline_overfits,
        "helpful_regularizer_count": helpful_count,
        "regularizer_count": expected_comparisons,
        "comparisons": comparisons,
        "caveat": (
            "One fixed seed is appropriate for a lecture illustration but cannot "
            "support benchmark-level claims or uncertainty estimates."
        ),
    }


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    # Ask PyTorch to surface nondeterministic operations without turning a
    # supported MPS run into a hard failure.
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed_everything(MODEL_SEED)

    total_start = time.perf_counter()
    print(f"Loading CIFAR-10 Python batches from {args.data_dir}", flush=True)
    all_images, all_labels, test_images, test_labels, source_files = load_cifar10(
        args.data_dir
    )
    train_indices, val_indices = stratified_split(
        all_labels,
        args.train_per_class,
        args.val_per_class,
        SPLIT_SEED,
    )
    train_images = all_images[torch.from_numpy(train_indices)]
    train_labels = all_labels[torch.from_numpy(train_indices)]
    val_images = all_images[torch.from_numpy(val_indices)]
    val_labels = all_labels[torch.from_numpy(val_indices)]
    del all_images, all_labels

    np.savez_compressed(
        args.evidence_dir / "cifar_regularization_split_indices.npz",
        train_indices=train_indices,
        val_indices=val_indices,
    )

    table = condition_table(args)
    selected_conditions = [table[name] for name in args.conditions]
    probe = SmallCifarCNN(0.0, 0.0)
    parameter_count = sum(parameter.numel() for parameter in probe.parameters())
    del probe

    config: dict[str, Any] = {
        "purpose": "single-seed controlled lecture demonstration, not a benchmark",
        "script": str(Path(__file__).resolve()),
        "script_sha256_at_start": sha256_file(Path(__file__).resolve()),
        "source_data_dir": str(args.data_dir.resolve()),
        "source_files_sha256": {
            path.name: sha256_file(path) for path in source_files
        },
        "split": {
            "source": "official CIFAR-10 training batches only",
            "method": "stratified without replacement",
            "seed": SPLIT_SEED,
            "train_size": len(train_images),
            "validation_size": len(val_images),
            "train_per_class": args.train_per_class,
            "validation_per_class": args.val_per_class,
            "train_class_counts": torch.bincount(train_labels, minlength=10).tolist(),
            "validation_class_counts": torch.bincount(val_labels, minlength=10).tolist(),
            "indices_file": "cifar_regularization_split_indices.npz",
        },
        "test": {
            "source": "official CIFAR-10 test batch",
            "size": len(test_images),
            "class_counts": torch.bincount(test_labels, minlength=10).tolist(),
            "policy": (
                "evaluate exactly once per condition after checkpoint selection; "
                "never use for tuning"
            ),
        },
        "classes": list(CLASS_NAMES),
        "normalization": {"mean": CIFAR_MEAN, "std": CIFAR_STD},
        "architecture": {
            "name": "SmallCifarCNN",
            "parameter_count": parameter_count,
            "notes": (
                "No batch normalization; dropout modules have p=0 outside the "
                "dropout condition, so all conditions have identical parameters."
            ),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "scheduler": None,
            "checkpoint_rule": "minimum validation cross-entropy",
            "evaluation_loss": "hard-label cross-entropy for every condition",
            "model_seed": MODEL_SEED,
            "minibatch_order_seed": ORDER_SEED,
            "augmentation_seed": AUGMENT_SEED,
            "hyperparameter_selection": (
                "Conventional teaching-scale settings, checked against validation "
                "behavior only; test metrics were not used to choose settings."
            ),
        },
        "conditions": [asdict(condition) for condition in selected_conditions],
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "mps_built": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
            ),
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
            "deterministic_algorithms": "enabled with warn_only=True",
        },
    }
    json_dump(args.evidence_dir / "cifar_regularization_config.json", config)

    print(
        f"Device: {device}; train={len(train_images)}, val={len(val_images)}, "
        f"test={len(test_images)}; parameters={parameter_count:,}",
        flush=True,
    )
    histories: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for condition in selected_conditions:
        print(f"\n--- {CONDITION_LABELS[condition.name]} ---", flush=True)
        history, summary = run_condition(
            condition=condition,
            train_images=train_images,
            train_labels=train_labels,
            val_images=val_images,
            val_labels=val_labels,
            test_images=test_images,
            test_labels=test_labels,
            args=args,
            device=device,
        )
        histories[condition.name] = history
        summaries[condition.name] = summary

        # Keep an interrupted long run resumable at the level of evidence.
        partial_rows = [row for rows in histories.values() for row in rows]
        write_csv(
            args.evidence_dir / "cifar_regularization_epoch_metrics.csv",
            partial_rows,
            list(partial_rows[0]),
        )
        json_dump(
            args.evidence_dir / "cifar_regularization_summary.json",
            {"status": "in_progress", "results": summaries},
        )

    total_seconds = time.perf_counter() - total_start
    assessment = assess_pedagogy(summaries, args.epochs)
    summary_document = {
        "status": "complete",
        "selection_and_test_policy": (
            "Each checkpoint was chosen by minimum validation cross-entropy. "
            "The official test set was evaluated exactly once per selected checkpoint."
        ),
        "single_seed_warning": (
            "Results are descriptive teaching evidence; no error bars or benchmark "
            "claims are justified from one seed."
        ),
        "total_runtime_seconds": total_seconds,
        "results": summaries,
        "pedagogical_assessment": assessment,
    }
    json_dump(
        args.evidence_dir / "cifar_regularization_summary.json", summary_document
    )
    summary_rows = list(summaries.values())
    flat_summary_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        flat = {key: value for key, value in row.items() if key != "settings"}
        flat.update({f"setting_{key}": value for key, value in row["settings"].items()})
        flat_summary_rows.append(flat)
    write_csv(
        args.evidence_dir / "cifar_regularization_summary.csv",
        flat_summary_rows,
        list(flat_summary_rows[0]),
    )
    plot_curves(histories, summaries, args.figures_dir)
    plot_summary(summaries, args.figures_dir)

    print("\nFinal selected-checkpoint results", flush=True)
    for name in args.conditions:
        result = summaries[name]
        print(
            f"  {CONDITION_LABELS[name]:16s} e{result['best_epoch_by_val_loss']:02d}: "
            f"val={100 * result['selected_val_accuracy']:.1f}%  "
            f"test={100 * result['selected_test_accuracy']:.1f}%  "
            f"gap={100 * result['selected_train_val_accuracy_gap']:.1f} pp",
            flush=True,
        )
    print(
        f"Pedagogical verdict: {assessment['verdict']}; "
        f"total runtime {total_seconds:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
