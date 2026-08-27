"""Practical visual evidence for Lecture 7.

The two augmentation figures use real 32x32 CIFAR-10 examples from the local
copy already used by ``stt-ai-teaching``.  The low source resolution is
deliberate: students can see that augmentation changes the *training view*, not
the class.  The MC-dropout figure is explicitly a schematic bridge, not an
empirical calibration claim.

Run from the ``dl-teaching`` repository root:

    python3 lecture7/diagrams/l7_practical_figs.py
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


INK = "#23373B"
ACC = "#EB811B"
TEAL = "#2C7A7B"
GREEN = "#14B03D"
MUTED = "#6E7F82"
RED = "#D64550"
BLUE = "#2B6CB0"
CREAM = "#F7F5F0"

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "lecture7" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_CIFAR = Path(
    "/Users/nipun/git/stt-ai-teaching/lecture-demos/week05/data/"
    "cifar-10-batches-py/data_batch_1"
)
CIFAR_BATCH = Path(os.environ.get("L7_CIFAR_BATCH", DEFAULT_CIFAR))

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
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 1.0,
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.solid_capstyle": "round",
    }
)


def save(fig: plt.Figure, name: str, *, raster_only: bool = False) -> None:
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=220, transparent=True)
    if not raster_only:
        fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


def load_cifar_pair() -> tuple[np.ndarray, np.ndarray]:
    """Return one cat (class 3) and one dog (class 5), CHW -> HWC uint8."""
    if not CIFAR_BATCH.exists():
        raise FileNotFoundError(
            f"CIFAR batch not found at {CIFAR_BATCH}. Set L7_CIFAR_BATCH to "
            "a CIFAR-10 Python data_batch file."
        )
    with CIFAR_BATCH.open("rb") as handle:
        batch = pickle.load(handle, encoding="bytes")
    data = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.asarray(batch[b"labels"])
    # Fixed indices within each class make the generated figures reproducible.
    cat_i = np.flatnonzero(labels == 3)[7]
    dog_i = np.flatnonzero(labels == 5)[3]
    cat = data[cat_i].transpose(1, 2, 0)
    dog = data[dog_i].transpose(1, 2, 0)
    return cat, dog


def padded_crop(image: Image.Image, *, left: int, top: int) -> Image.Image:
    padded = ImageOps.expand(image, border=4, fill=(127, 127, 127))
    return padded.crop((left, top, left + 32, top + 32))


def f_augmentation_gallery(cat: np.ndarray) -> None:
    base = Image.fromarray(cat)
    rng = np.random.default_rng(21)
    noise = np.clip(cat.astype(float) + rng.normal(0, 12, cat.shape), 0, 255).astype(np.uint8)
    views = [
        ("original", base),
        ("crop + shift", padded_crop(base, left=1, top=6)),
        ("horizontal flip", ImageOps.mirror(base)),
        ("colour jitter", ImageEnhance.Color(ImageEnhance.Contrast(base).enhance(1.25)).enhance(0.65)),
        ("mild blur", base.filter(ImageFilter.GaussianBlur(radius=0.65))),
        ("sensor noise", Image.fromarray(noise)),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(12.8, 2.65))
    for ax, (label, view) in zip(axes, views):
        ax.imshow(view, interpolation="nearest")
        ax.set_title(label, fontsize=11.5, color=TEAL if label == "original" else INK, pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.4 if label == "original" else 0.8)
            spine.set_color(TEAL if label == "original" else MUTED)
    fig.suptitle("six training views · one label: cat", fontsize=15, weight=600, y=1.04)
    fig.text(
        0.5,
        -0.02,
        "CIFAR-10 · 32×32 source image · every transform must preserve the task label",
        ha="center",
        color=MUTED,
        fontsize=10.5,
    )
    fig.tight_layout(w_pad=0.65)
    save(fig, "cifar10_augmentation_gallery", raster_only=True)


def f_mixup_cutmix(cat: np.ndarray, dog: np.ndarray) -> None:
    lam = 0.70
    mix = np.clip(lam * cat.astype(float) + (1 - lam) * dog.astype(float), 0, 255).astype(np.uint8)

    cut = cat.copy()
    # A 16x16 patch occupies 25% of the image, so the label weights are 0.75/0.25.
    cut[8:24, 15:31] = dog[8:24, 15:31]

    panels = [
        (cat, "cat", TEAL),
        (dog, "dog", ACC),
        (mix, "MixUp\n0.70 cat + 0.30 dog", BLUE),
        (cut, "CutMix\n0.75 cat + 0.25 dog", GREEN),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 3.15))
    for ax, (arr, label, colour) in zip(axes, panels):
        ax.imshow(arr, interpolation="nearest")
        ax.set_title(label, color=colour, fontsize=12, weight=600, pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_color(colour)
    fig.suptitle("mix the supervision whenever you mix the image", fontsize=15, weight=600, y=1.03)
    fig.tight_layout(w_pad=1.1)
    save(fig, "cifar10_mixup_cutmix", raster_only=True)


def f_mc_dropout_bridge() -> None:
    """Schematic stochastic functions: agreement near data, disagreement away."""
    rng = np.random.default_rng(7)
    x = np.linspace(-3.25, 3.25, 420)
    train_x = np.linspace(-1.75, 1.75, 11)
    train_y = np.sin(1.35 * train_x)
    base = np.sin(1.35 * x)
    outside = np.maximum(np.abs(x) - 1.65, 0.0)

    samples = []
    for _ in range(18):
        a, b, c = rng.normal(0, [0.12, 0.06, 0.025])
        smooth_noise = a + b * x + c * (x**2 - 2.0)
        samples.append(base + (0.045 + 0.62 * outside**1.35) * smooth_noise)
    samples = np.asarray(samples)

    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    ax.axvspan(-3.25, -1.75, color=ACC, alpha=0.08, lw=0)
    ax.axvspan(1.75, 3.25, color=ACC, alpha=0.08, lw=0)
    for curve in samples:
        ax.plot(x, curve, color=BLUE, alpha=0.22, lw=1.2)
    ax.plot(x, samples.mean(axis=0), color=INK, lw=2.6, label="mean of T stochastic passes")
    ax.scatter(train_x, train_y, s=32, color=TEAL, edgecolor="white", linewidth=0.5, zorder=5)
    ax.axvline(-1.75, color=MUTED, lw=1.0, ls="--")
    ax.axvline(1.75, color=MUTED, lw=1.0, ls="--")
    ax.text(0, -1.38, "training support → masks mostly agree", ha="center", color=TEAL, fontsize=11)
    ax.text(2.52, 1.20, "away from data\npasses disagree", ha="center", color=ACC, fontsize=11)
    ax.set_xlabel("input x")
    ax.set_ylabel("prediction")
    ax.set_xticks([-3, -1.75, 0, 1.75, 3])
    ax.set_xticklabels(["−3", "edge", "0", "edge", "3"])
    ax.set_yticks([])
    ax.set_ylim(-1.55, 1.55)
    ax.legend(frameon=False, loc="upper left", fontsize=10.5)
    ax.set_title("MC dropout bridge · keep masks stochastic and measure disagreement", fontsize=13)
    fig.text(
        0.995,
        0.01,
        "schematic — uncertainty must be validated, not assumed",
        ha="right",
        color=MUTED,
        fontsize=9.5,
    )
    save(fig, "mc_dropout_bridge")


def main() -> None:
    cat, dog = load_cifar_pair()
    f_augmentation_gallery(cat)
    f_mixup_cutmix(cat, dog)
    f_mc_dropout_bridge()
    print(f"wrote practical Lecture 7 figures to {OUT}")


if __name__ == "__main__":
    main()
