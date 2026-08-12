#!/usr/bin/env python3
"""Build the real-image transfer-learning evidence used by public Lecture 9.

The experiment is deliberately small and pedagogical, not a benchmark.  It
uses six Oxford-IIIT Pet cat breeds, preserves the dataset's official
trainval/test boundary, and compares three ResNet-18 adaptation regimes under
one fixed protocol.  The official test subset is evaluated exactly once,
after validation has selected a regime and epoch.

The first run bootstraps ``selection-manifest.csv`` from JPEGs available under
``--images-dir``.  Later runs reuse and hash-check that committed manifest.
Pass ``--refresh-selection`` only when intentionally defining a new evidence
case; doing so invalidates all previous numeric results.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import json
import math
import platform
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import PIL
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.models import ResNet18_Weights, resnet18


SEED = 20260812
BREEDS = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
]
TRAIN_PER_BREED = 12
VAL_PER_BREED = 6
TEST_PER_BREED = 6
EPOCHS = 15
BATCH_SIZE = 12
WEIGHT_DECAY = 1e-4
HEAD_LR = 3e-3
LATE_LR = 3e-4
ALL_LR = 3e-5
EXPECTED_WEIGHTS_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)
WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
DATASET_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/"
IMAGES_ARCHIVE_URL = "https://thor.robots.ox.ac.uk/pets/images.tar.gz"
ANNOTATIONS_ARCHIVE_URL = (
    "https://thor.robots.ox.ac.uk/pets/annotations.tar.gz"
)

TEAL = (44, 122, 123)
ORANGE = (235, 129, 27)
BLUE = (43, 108, 176)
GREEN = (44, 142, 80)
RED = (205, 74, 74)
INK = (35, 55, 59)
MUTED = (101, 117, 120)
PAPER = (247, 248, 246)
WHITE = (255, 255, 255)
REGIME_COLORS = {"probe": "#2c7a7b", "late": "#eb811b", "all": "#2b6cb0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing official Oxford-IIIT Pet JPEGs.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        required=True,
        help="Directory containing official trainval.txt and test.txt.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Cached ResNet18 IMAGENET1K_V1 state dict.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Evidence output directory (default: directory of this script).",
    )
    parser.add_argument(
        "--shared-image",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Abyssinian_1.jpg",
        help="Shared real image used only for preprocessing/activation figures.",
    )
    parser.add_argument(
        "--refresh-selection",
        action="store_true",
        help="Replace the fixed manifest from currently available official JPEGs.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Refresh preprocessing/activation figures without training or opening test.",
    )
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(official_split: str, image_id: str) -> str:
    return hashlib.sha256(
        f"{SEED}:{official_split}:{image_id}".encode("utf-8")
    ).hexdigest()


def breed_for(image_id: str) -> str | None:
    for breed in BREEDS:
        if image_id.startswith(breed + "_"):
            return breed
    return None


def read_official_split(path: Path, official_split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"{path}:{line_number}: expected four columns")
        image_id, class_id, species_id, breed_id = parts
        breed = breed_for(image_id)
        if breed is None:
            continue
        rows.append(
            {
                "image_id": image_id,
                "filename": f"{image_id}.jpg",
                "breed": breed,
                "label": BREEDS.index(breed),
                "official_split": official_split,
                "official_class_id": int(class_id),
                "official_species_id": int(species_id),
                "official_breed_id": int(breed_id),
            }
        )
    return rows


def verify_jpeg(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        if width < 64 or height < 64:
            raise ValueError(f"unexpectedly small image: {path} ({width}x{height})")
        image.convert("RGB").load()
    return width, height


MANIFEST_FIELDS = [
    "manifest_index",
    "filename",
    "image_id",
    "breed",
    "label",
    "official_split",
    "teaching_split",
    "official_class_id",
    "official_species_id",
    "official_breed_id",
    "selection_rank_sha256",
    "jpeg_sha256",
    "width",
    "height",
]


def bootstrap_manifest(
    images_dir: Path, annotations_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    source_rows = {
        "trainval": read_official_split(
            annotations_dir / "trainval.txt", "trainval"
        ),
        "test": read_official_split(annotations_dir / "test.txt", "test"),
    }
    availability: dict[str, dict[str, int]] = {}
    rows: list[dict[str, Any]] = []

    for breed in BREEDS:
        present_trainval = [
            row
            for row in source_rows["trainval"]
            if row["breed"] == breed and (images_dir / row["filename"]).is_file()
        ]
        present_test = [
            row
            for row in source_rows["test"]
            if row["breed"] == breed and (images_dir / row["filename"]).is_file()
        ]
        available_trainval: list[dict[str, Any]] = []
        available_test: list[dict[str, Any]] = []
        for row in present_trainval:
            try:
                verify_jpeg(images_dir / row["filename"])
            except (OSError, ValueError):
                continue
            available_trainval.append(row)
        for row in present_test:
            try:
                verify_jpeg(images_dir / row["filename"])
            except (OSError, ValueError):
                continue
            available_test.append(row)
        availability[breed] = {
            "trainval_present": len(present_trainval),
            "trainval_available": len(available_trainval),
            "trainval_rejected_invalid": len(present_trainval)
            - len(available_trainval),
            "test_present": len(present_test),
            "test_available": len(available_test),
            "test_rejected_invalid": len(present_test) - len(available_test),
        }
        if len(available_trainval) < TRAIN_PER_BREED + VAL_PER_BREED:
            raise ValueError(f"not enough available trainval JPEGs for {breed}")
        if len(available_test) < TEST_PER_BREED:
            raise ValueError(f"not enough available test JPEGs for {breed}")

        available_trainval.sort(
            key=lambda row: stable_rank("trainval", row["image_id"])
        )
        available_test.sort(key=lambda row: stable_rank("test", row["image_id"]))
        chosen_trainval = available_trainval[: TRAIN_PER_BREED + VAL_PER_BREED]
        chosen_test = available_test[:TEST_PER_BREED]

        for index, row in enumerate(chosen_trainval):
            selected = dict(row)
            selected["teaching_split"] = (
                "train" if index < TRAIN_PER_BREED else "val"
            )
            rows.append(selected)
        for row in chosen_test:
            selected = dict(row)
            selected["teaching_split"] = "test"
            rows.append(selected)

    split_order = {"train": 0, "val": 1, "test": 2}
    rows.sort(
        key=lambda row: (
            split_order[row["teaching_split"]],
            row["label"],
            stable_rank(row["official_split"], row["image_id"]),
        )
    )
    for manifest_index, row in enumerate(rows):
        path = images_dir / row["filename"]
        width, height = verify_jpeg(path)
        row.update(
            {
                "manifest_index": manifest_index,
                "selection_rank_sha256": stable_rank(
                    row["official_split"], row["image_id"]
                ),
                "jpeg_sha256": sha256(path),
                "width": width,
                "height": height,
                "path": path,
            }
        )
    return rows, availability


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path, images_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MANIFEST_FIELDS:
            raise ValueError(f"unexpected manifest schema in {path}")
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for key in [
                "manifest_index",
                "label",
                "official_class_id",
                "official_species_id",
                "official_breed_id",
                "width",
                "height",
            ]:
                row[key] = int(row[key])
            row["path"] = images_dir / row["filename"]
            rows.append(row)

    expected = {
        "train": TRAIN_PER_BREED * len(BREEDS),
        "val": VAL_PER_BREED * len(BREEDS),
        "test": TEST_PER_BREED * len(BREEDS),
    }
    if Counter(row["teaching_split"] for row in rows) != Counter(expected):
        raise ValueError("manifest split counts do not match the fixed protocol")
    for split, count_per_breed in [
        ("train", TRAIN_PER_BREED),
        ("val", VAL_PER_BREED),
        ("test", TEST_PER_BREED),
    ]:
        counts = Counter(
            row["breed"] for row in rows if row["teaching_split"] == split
        )
        if counts != Counter({breed: count_per_breed for breed in BREEDS}):
            raise ValueError(f"manifest is not breed-balanced in {split}")
    for row in rows:
        if not row["path"].is_file():
            raise FileNotFoundError(row["path"])
        if sha256(row["path"]) != row["jpeg_sha256"]:
            raise ValueError(f"JPEG hash mismatch: {row['path']}")
        if verify_jpeg(row["path"]) != (row["width"], row["height"]):
            raise ValueError(f"JPEG dimensions changed: {row['path']}")
    return rows


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class Pets(Dataset):
    def __init__(
        self, rows: list[dict[str, Any]], split: str, transform: Any
    ) -> None:
        self.rows = [row for row in rows if row["teaching_split"] == split]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"]), int(row["manifest_index"])


class TransferModel(nn.Module):
    def __init__(
        self, weights_path: Path, head_state: dict[str, torch.Tensor]
    ) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(512, len(BREEDS))
        self.head.load_state_dict(head_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def configure(model: TransferModel, regime: str) -> list[dict[str, Any]]:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    groups: list[dict[str, Any]] = [
        {"params": model.head.parameters(), "lr": HEAD_LR, "name": "head"}
    ]
    if regime == "late":
        for parameter in model.backbone.layer4.parameters():
            parameter.requires_grad = True
        groups.insert(
            0,
            {
                "params": model.backbone.layer4.parameters(),
                "lr": LATE_LR,
                "name": "layer4",
            },
        )
    elif regime == "all":
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
        groups.insert(
            0,
            {
                "params": model.backbone.parameters(),
                "lr": ALL_LR,
                "name": "backbone",
            },
        )
    elif regime != "probe":
        raise ValueError(regime)
    return groups


@torch.inference_mode()
def evaluate_validation(
    model: TransferModel, loader: DataLoader
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    for x, y, _ in loader:
        logits = model(x)
        total_loss += nn.functional.cross_entropy(
            logits, y, reduction="sum"
        ).item()
        correct += int((logits.argmax(1) == y).sum())
        count += len(y)
    return {"loss": total_loss / count, "accuracy": correct / count}


def train_regime(
    rows: list[dict[str, Any]],
    transform: Any,
    weights_path: Path,
    head_state: dict[str, torch.Tensor],
    regime: str,
) -> dict[str, Any]:
    set_seed()
    model = TransferModel(weights_path, head_state)
    groups = configure(model, regime)
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    train_loader = DataLoader(
        Pets(rows, "train", transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )
    val_loader = DataLoader(
        Pets(rows, "val", transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    history: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, int], dict[str, torch.Tensor], dict[str, Any]] | None = None

    for epoch in range(1, EPOCHS + 1):
        model.head.train()
        # Fixed buffers are part of the comparison contract.  BN affine
        # parameters still receive gradients when their layer is unfrozen.
        model.backbone.eval()
        running_loss = 0.0
        correct = 0
        count = 0
        for x, y, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)
            correct += int((logits.argmax(1) == y).sum())
            count += len(y)

        validation = evaluate_validation(model, val_loader)
        record = {
            "regime": regime,
            "epoch": epoch,
            "train_loss": running_loss / count,
            "train_accuracy": correct / count,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
        }
        history.append(record)
        epoch_key = (
            record["val_accuracy"],
            -record["val_loss"],
            -epoch,
        )
        if best is None or epoch_key > best[0]:
            best = (epoch_key, copy.deepcopy(model.state_dict()), dict(record))
        print(
            f"{regime:5s} epoch {epoch:02d}/{EPOCHS}: "
            f"train {record['train_accuracy']:.3f}, "
            f"val {record['val_accuracy']:.3f}, "
            f"val loss {record['val_loss']:.4f}",
            flush=True,
        )

    assert best is not None
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "regime": regime,
        "trainable_parameters": trainable_parameters,
        "best": best[2],
        "history": history,
        "state": best[1],
    }


@torch.inference_mode()
def evaluate_sealed_test_once(
    model: TransferModel,
    rows: list[dict[str, Any]],
    transform: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the one permitted selected-checkpoint pass over official test rows."""
    test_rows = {
        int(row["manifest_index"]): row
        for row in rows
        if row["teaching_split"] == "test"
    }
    loader = DataLoader(
        Pets(rows, "test", transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    prediction_rows: list[dict[str, Any]] = []
    total_loss = 0.0
    correct = 0
    confusion = np.zeros((len(BREEDS), len(BREEDS)), dtype=np.int64)

    for x, y, manifest_indices in loader:
        logits = model(x)
        probabilities = logits.softmax(dim=1)
        predictions = logits.argmax(dim=1)
        total_loss += nn.functional.cross_entropy(
            logits, y, reduction="sum"
        ).item()
        correct += int((predictions == y).sum())
        for item in range(len(y)):
            true_label = int(y[item])
            predicted_label = int(predictions[item])
            confusion[true_label, predicted_label] += 1
            manifest_index = int(manifest_indices[item])
            source = test_rows[manifest_index]
            record: dict[str, Any] = {
                "manifest_index": manifest_index,
                "filename": source["filename"],
                "true_label": true_label,
                "true_breed": BREEDS[true_label],
                "predicted_label": predicted_label,
                "predicted_breed": BREEDS[predicted_label],
                "correct": int(true_label == predicted_label),
                "confidence": float(probabilities[item, predicted_label]),
            }
            for label, breed in enumerate(BREEDS):
                record[f"prob_{breed}"] = float(probabilities[item, label])
            prediction_rows.append(record)

    count = len(prediction_rows)
    return (
        {
            "loss": total_loss / count,
            "accuracy": correct / count,
            "correct": correct,
            "count": count,
            "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        },
        prediction_rows,
    )


def pretrained_backbone(weights_path: Path) -> nn.Module:
    model = resnet18(weights=None)
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    model.fc = nn.Identity()
    model.eval()
    return model


@torch.inference_mode()
def save_trainval_features(
    rows: list[dict[str, Any]],
    transform: Any,
    weights_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Save only training/validation features; sealed test remains untouched."""
    model = pretrained_backbone(weights_path)
    subset_rows = [row for row in rows if row["teaching_split"] != "test"]
    features: list[np.ndarray] = []
    labels: list[int] = []
    indices: list[int] = []
    splits: list[str] = []
    filenames: list[str] = []
    for split in ["train", "val"]:
        loader = DataLoader(
            Pets(rows, split, transform),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        for x, y, manifest_indices in loader:
            features.append(model(x).cpu().numpy().astype(np.float32))
            labels.extend(int(value) for value in y)
            indices.extend(int(value) for value in manifest_indices)
    by_index = {int(row["manifest_index"]): row for row in subset_rows}
    for manifest_index in indices:
        row = by_index[manifest_index]
        splits.append(row["teaching_split"])
        filenames.append(row["filename"])
    matrix = np.concatenate(features, axis=0)
    np.savez_compressed(
        output_path,
        features=matrix,
        labels=np.asarray(labels, dtype=np.int64),
        manifest_indices=np.asarray(indices, dtype=np.int64),
        teaching_splits=np.asarray(splits),
        filenames=np.asarray(filenames),
        breeds=np.asarray(BREEDS),
    )
    return {
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "dtype": str(matrix.dtype),
        "contains_test": False,
    }


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def contain(image: Image.Image, size: tuple[int, int], background=PAPER) -> Image.Image:
    canvas = Image.new("RGB", size, background)
    fitted = image.convert("RGB").copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def build_preprocessing_figure(
    image_path: Path, transform: Any, output_path: Path
) -> dict[str, Any]:
    with Image.open(image_path) as handle:
        original = handle.convert("RGB")
    resized = TF.resize(
        original,
        transform.resize_size,
        interpolation=transform.interpolation,
        antialias=transform.antialias,
    )
    cropped = TF.center_crop(resized, transform.crop_size)
    tensor = transform(original)
    channel_stats = []
    for channel, name in enumerate(["R", "G", "B"]):
        values = tensor[channel].numpy()
        channel_stats.append(
            {
                "channel": name,
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
        )

    panel_width = 410
    canvas = Image.new("RGB", (panel_width * 4, 570), PAPER)
    draw = ImageDraw.Draw(canvas)
    title_font = font(25, bold=True)
    body_font = font(18)
    mono_font = font(18, bold=True)
    panels = [
        ("1  ORIGINAL", original, f"RGB  {original.width} x {original.height}"),
        ("2  RESIZE SHORT SIDE", resized, f"Bilinear  {resized.width} x {resized.height}"),
        ("3  CENTER CROP", cropped, f"Tensor source  {cropped.width} x {cropped.height}"),
    ]
    for index, (title, image, subtitle) in enumerate(panels):
        left = index * panel_width
        draw.text((left + 20, 18), title, font=title_font, fill=INK)
        draw.text((left + 20, 54), subtitle, font=body_font, fill=MUTED)
        canvas.paste(contain(image, (370, 390), WHITE), (left + 20, 92))

    left = panel_width * 3
    draw.text((left + 20, 18), "4  NORMALIZE", font=title_font, fill=INK)
    draw.text((left + 20, 54), "Official IMAGENET1K_V1 contract", font=body_font, fill=MUTED)
    draw.rounded_rectangle((left + 20, 92, left + 390, 482), radius=10, fill=WHITE, outline=(210, 216, 213), width=2)
    draw.text((left + 42, 118), "x' = (x - mean) / std", font=mono_font, fill=INK)
    means = transform.mean
    stds = transform.std
    colors = [RED, GREEN, BLUE]
    for index, stats in enumerate(channel_stats):
        y = 174 + index * 88
        draw.ellipse((left + 42, y, left + 70, y + 28), fill=colors[index])
        draw.text(
            (left + 82, y - 4),
            f"{stats['channel']}   mean {means[index]:.3f}  std {stds[index]:.3f}",
            font=body_font,
            fill=INK,
        )
        draw.text(
            (left + 82, y + 26),
            f"observed [{stats['min']:+.2f}, {stats['max']:+.2f}]",
            font=body_font,
            fill=MUTED,
        )
    draw.text((left + 42, 444), "output shape  3 x 224 x 224", font=mono_font, fill=INK)
    draw.text(
        (20, 520),
        "Observed transformation of Abyssinian_1.jpg; no augmentation and no model prediction.",
        font=body_font,
        fill=MUTED,
    )
    canvas.save(output_path, optimize=True)
    return {
        "original_size_wh": [original.width, original.height],
        "resized_size_wh": [resized.width, resized.height],
        "crop_size_wh": [cropped.width, cropped.height],
        "tensor_shape_chw": list(tensor.shape),
        "channel_stats_after_normalization": channel_stats,
    }


def activation_color(values: np.ndarray) -> Image.Image:
    lo, hi = np.percentile(values, [1, 99])
    scaled = np.clip((values - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0)
    stops = np.asarray(
        [
            [244, 247, 245],
            [86, 140, 144],
            [22, 71, 80],
            [235, 129, 27],
            [255, 236, 184],
        ],
        dtype=np.float64,
    )
    position = scaled * (len(stops) - 1)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = position - lower
    rgb = stops[lower] * (1 - fraction[..., None]) + stops[upper] * fraction[..., None]
    return Image.fromarray(np.uint8(np.round(rgb)), mode="RGB")


@torch.inference_mode()
def build_activation_figure(
    image_path: Path,
    transform: Any,
    weights_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    with Image.open(image_path) as handle:
        original = handle.convert("RGB")
    input_tensor = transform(original).unsqueeze(0)
    model = resnet18(weights=None)
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    x = model.conv1(input_tensor)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    stages: list[tuple[str, torch.Tensor]] = []
    x = model.layer1(x)
    stages.append(("layer1", x.detach().cpu()))
    x = model.layer2(x)
    stages.append(("layer2", x.detach().cpu()))
    x = model.layer3(x)
    x = model.layer4(x)
    stages.append(("layer4", x.detach().cpu()))

    canvas = Image.new("RGB", (1660, 780), PAPER)
    draw = ImageDraw.Draw(canvas)
    title_font = font(25, bold=True)
    body_font = font(17)
    small_font = font(15, bold=True)
    crop = TF.center_crop(
        TF.resize(
            original,
            transform.resize_size,
            interpolation=transform.interpolation,
            antialias=transform.antialias,
        ),
        transform.crop_size,
    )
    canvas.paste(contain(crop, (286, 286), WHITE), (24, 76))
    draw.text((24, 26), "PRETRAINED RESNET-18", font=title_font, fill=INK)
    draw.text((24, 382), "actual center crop", font=body_font, fill=INK)
    draw.text((24, 414), "computed activations", font=body_font, fill=MUTED)
    draw.text((24, 448), "not saliency", font=small_font, fill=ORANGE)

    metadata: dict[str, Any] = {}
    tile = 116
    gap = 12
    # Leave a clear gutter after the global title/input column so the layer1
    # heading cannot visually concatenate with "RESNET-18".
    base_x = 380
    for row_index, (name, activation) in enumerate(stages):
        array = activation[0].numpy()
        energy = np.mean(np.abs(array), axis=(1, 2))
        selected = np.argsort(-energy, kind="stable")[:8]
        top = 34 + row_index * 242
        draw.text(
            (base_x, top),
            f"{name}   {array.shape[0]} x {array.shape[1]} x {array.shape[2]}",
            font=title_font,
            fill=INK,
        )
        draw.text(
            (base_x + 430, top + 5),
            "top channels by mean |activation|; each tile uses its own 1-99% scale",
            font=body_font,
            fill=MUTED,
        )
        for column, channel in enumerate(selected):
            left = base_x + column * (tile + gap)
            image = activation_color(array[int(channel)]).resize(
                (tile, tile), Image.Resampling.NEAREST
            )
            canvas.paste(image, (left, top + 50))
            draw.rectangle((left, top + 50, left + tile, top + 50 + tile), outline=(185, 196, 193), width=1)
            draw.text(
                (left + 4, top + 171),
                f"ch {int(channel)}",
                font=small_font,
                fill=INK,
            )
        metadata[name] = {
            "shape_chw": list(array.shape),
            "selected_channels_by_mean_absolute_activation": [
                int(channel) for channel in selected
            ],
            "display_percentiles_per_channel": [1, 99],
        }
    canvas.save(output_path, optimize=True)
    return metadata


def build_contact_sheet(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    output_path: Path,
) -> list[int]:
    by_index = {int(row["manifest_index"]): row for row in rows}
    chosen: list[dict[str, Any]] = []
    for breed in BREEDS:
        candidates = [row for row in predictions if row["true_breed"] == breed]
        candidates.sort(key=lambda row: int(row["manifest_index"]))
        chosen.append(candidates[0])

    panel_width, panel_height = 420, 350
    header_height = 42
    canvas = Image.new(
        "RGB", (panel_width * 3, header_height + panel_height * 2), PAPER
    )
    draw = ImageDraw.Draw(canvas)
    title_font = font(18, bold=True)
    body_font = font(16)
    draw.text(
        (20, 11),
        "SEALED TEST: FIRST MANIFEST IMAGE PER BREED (NOT CHOSEN FOR CORRECTNESS)",
        font=body_font,
        fill=MUTED,
    )
    for index, prediction in enumerate(chosen):
        left = (index % 3) * panel_width
        top = header_height + (index // 3) * panel_height
        source = by_index[int(prediction["manifest_index"])]
        with Image.open(source["path"]) as handle:
            image = contain(handle.convert("RGB"), (380, 250), WHITE)
        canvas.paste(image, (left + 20, top + 18))
        correct = bool(prediction["correct"])
        color = GREEN if correct else RED
        draw.text(
            (left + 20, top + 280),
            f"GT  {prediction['true_breed'].replace('_', ' ')}",
            font=title_font,
            fill=INK,
        )
        draw.text(
            (left + 20, top + 310),
            f"PRED  {prediction['predicted_breed'].replace('_', ' ')}  "
            f"{prediction['confidence']:.0%}",
            font=body_font,
            fill=color,
        )
    canvas.save(output_path, quality=88, optimize=True, subsampling=1)
    return [int(row["manifest_index"]) for row in chosen]


def svg_polyline(
    values: list[float],
    left: float,
    top: float,
    width: float,
    height: float,
    ymin: float,
    ymax: float,
) -> str:
    points = []
    for index, value in enumerate(values):
        x = left + width * index / max(len(values) - 1, 1)
        y = top + height * (ymax - value) / max(ymax - ymin, 1e-12)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def build_curves_svg(
    results: list[dict[str, Any]],
    selected: dict[str, Any],
    test: dict[str, Any],
    output_path: Path,
) -> None:
    width, height = 1280, 650
    left_a, left_b, top = 90, 700, 130
    plot_width, plot_height = 480, 355
    selected_regime = selected["regime"]
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f8f6"/>',
        # These sizes are intentionally large: the SVG is embedded at roughly
        # 177 mm in the deck, and the strict slide audit requires >= 7.5 pt
        # after scaling.  Keep the generated asset and builder in lockstep.
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#23373b}.title{font-size:30px;font-weight:700}.label{font-size:25px}.small{font-size:25px;fill:#657578}.grid{stroke:#d9dfdc;stroke-width:1}.axis{stroke:#52666a;stroke-width:1.5}</style>',
        '<text x="50" y="45" class="title">FIXED VALIDATION SELECTS THE ADAPTATION REGIME</text>',
        f'<text x="50" y="72" class="small">6 breeds; 72 train / 36 validation / 36 sealed official-test images; seed {SEED}</text>',
        f'<text x="{left_a}" y="{top - 22}" class="title">Validation accuracy</text>',
        f'<text x="{left_b}" y="{top - 22}" class="title">Selected: {html.escape(selected_regime)}</text>',
    ]
    for panel_left in [left_a, left_b]:
        for tick in range(6):
            y = top + plot_height * tick / 5
            pieces.append(
                f'<line x1="{panel_left}" y1="{y:.1f}" x2="{panel_left + plot_width}" y2="{y:.1f}" class="grid"/>'
            )
            pieces.append(
                f'<text x="{panel_left - 12}" y="{y + 5:.1f}" text-anchor="end" class="small">{1 - tick / 5:.1f}</text>'
            )
        pieces.append(
            f'<line x1="{panel_left}" y1="{top}" x2="{panel_left}" y2="{top + plot_height}" class="axis"/>'
        )
        pieces.append(
            f'<line x1="{panel_left}" y1="{top + plot_height}" x2="{panel_left + plot_width}" y2="{top + plot_height}" class="axis"/>'
        )
        for epoch in [1, 5, 10, 15]:
            x = panel_left + plot_width * (epoch - 1) / (EPOCHS - 1)
            pieces.append(
                f'<text x="{x:.1f}" y="{top + plot_height + 26}" text-anchor="middle" class="small">{epoch}</text>'
            )
    for index, result in enumerate(results):
        regime = result["regime"]
        values = [row["val_accuracy"] for row in result["history"]]
        points = svg_polyline(values, left_a, top, plot_width, plot_height, 0, 1)
        pieces.append(
            f'<polyline points="{points}" fill="none" stroke="{REGIME_COLORS[regime]}" stroke-width="4" stroke-linejoin="round"/>'
        )
        legend_y = 545 + index * 24
        pieces.append(
            f'<line x1="{left_a}" y1="{legend_y}" x2="{left_a + 28}" y2="{legend_y}" stroke="{REGIME_COLORS[regime]}" stroke-width="4"/>'
        )
        pieces.append(
            f'<text x="{left_a + 38}" y="{legend_y + 5}" class="label">{html.escape(regime)}  best {result["best"]["val_accuracy"]:.1%}</text>'
        )

    selected_result = next(
        result for result in results if result["regime"] == selected_regime
    )
    train_values = [row["train_accuracy"] for row in selected_result["history"]]
    val_values = [row["val_accuracy"] for row in selected_result["history"]]
    for values, color, dash, label, legend_y in [
        (train_values, "#52666a", "8 7", "train", 545),
        (val_values, REGIME_COLORS[selected_regime], "", "validation", 569),
    ]:
        points = svg_polyline(values, left_b, top, plot_width, plot_height, 0, 1)
        pieces.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4" stroke-dasharray="{dash}" stroke-linejoin="round"/>'
        )
        pieces.append(
            f'<line x1="{left_b}" y1="{legend_y}" x2="{left_b + 28}" y2="{legend_y}" stroke="{color}" stroke-width="4" stroke-dasharray="{dash}"/>'
        )
        pieces.append(
            f'<text x="{left_b + 38}" y="{legend_y + 5}" class="label">{label}</text>'
        )
    pieces.append(
        f'<rect x="{left_b + 185}" y="530" width="290" height="75" rx="8" fill="#ffffff" stroke="#2c7a7b" stroke-width="2"/>'
    )
    pieces.append(
        f'<text x="{left_b + 205}" y="557" class="label">test opened once</text>'
    )
    pieces.append(
        f'<text x="{left_b + 205}" y="590" class="title">{test["correct"]}/{test["count"]} = {test["accuracy"]:.1%}</text>'
    )
    pieces.append("</svg>")
    output_path.write_text("\n".join(pieces))


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "regime": result["regime"],
        "trainable_parameters": result["trainable_parameters"],
        "best": result["best"],
        "history": result["history"],
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    set_seed()

    if sha256(args.weights) != EXPECTED_WEIGHTS_SHA256:
        raise ValueError(
            "checkpoint is not torchvision ResNet18 IMAGENET1K_V1: "
            f"{args.weights}"
        )
    trainval_txt = args.annotations_dir / "trainval.txt"
    test_txt = args.annotations_dir / "test.txt"
    for path in [trainval_txt, test_txt, args.shared_image]:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest_path = args.output_dir / "selection-manifest.csv"
    availability: dict[str, dict[str, int]] | None = None
    if args.refresh_selection or not manifest_path.exists():
        rows, availability = bootstrap_manifest(args.images_dir, args.annotations_dir)
        write_csv(manifest_path, rows, MANIFEST_FIELDS)
        print(f"wrote new fixed selection: {manifest_path}", flush=True)
    rows = load_manifest(manifest_path, args.images_dir)
    existing_results_path = args.output_dir / "results.json"
    if availability is None and existing_results_path.is_file():
        existing_metadata = json.loads(existing_results_path.read_text())
        availability = existing_metadata.get("dataset", {}).get(
            "availability_at_bootstrap"
        )
    print(
        "locked split: "
        + ", ".join(
            f"{split}={sum(row['teaching_split'] == split for row in rows)}"
            for split in ["train", "val", "test"]
        ),
        flush=True,
    )

    weights_enum = ResNet18_Weights.IMAGENET1K_V1
    transform = weights_enum.transforms()
    preprocessing = build_preprocessing_figure(
        args.shared_image, transform, args.output_dir / "preprocessing-contract.png"
    )
    activations = build_activation_figure(
        args.shared_image,
        transform,
        args.weights,
        args.output_dir / "resnet18-activations.png",
    )

    if args.figures_only:
        results_path = args.output_dir / "results.json"
        if not results_path.is_file():
            raise FileNotFoundError(
                "--figures-only requires an existing measured results.json"
            )
        metadata = json.loads(results_path.read_text())
        metadata["preprocessing"]["observed_shared_image"] = preprocessing
        metadata["activation_figure"] = {
            "source_image": args.shared_image.name,
            "source_image_sha256": sha256(args.shared_image),
            "semantics": "computed pretrained activations; not saliency",
            "stages": activations,
        }
        selected_existing = next(
            result
            for result in metadata["regimes"]
            if result["regime"]
            == metadata["sealed_test_contract"]["selected_regime"]
        )
        build_curves_svg(
            metadata["regimes"],
            selected_existing,
            metadata["selected_test"],
            args.output_dir / "transfer-curves.svg",
        )
        for name in [
            "preprocessing-contract.png",
            "resnet18-activations.png",
            "transfer-curves.svg",
        ]:
            path = args.output_dir / name
            metadata["artifacts"][name] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        predictions_path = args.output_dir / "sealed-test-predictions.csv"
        if predictions_path.is_file():
            with predictions_path.open(newline="") as handle:
                recorded_predictions: list[dict[str, Any]] = []
                for raw in csv.DictReader(handle):
                    recorded: dict[str, Any] = dict(raw)
                    for key in [
                        "manifest_index",
                        "true_label",
                        "predicted_label",
                        "correct",
                    ]:
                        recorded[key] = int(recorded[key])
                    recorded["confidence"] = float(recorded["confidence"])
                    recorded_predictions.append(recorded)
            contact_indices = build_contact_sheet(
                rows,
                recorded_predictions,
                args.output_dir / "sealed-test-examples.jpg",
            )
            metadata["sealed_test_example_manifest_indices"] = contact_indices
            contact_path = args.output_dir / "sealed-test-examples.jpg"
            metadata["artifacts"][contact_path.name] = {
                "sha256": sha256(contact_path),
                "bytes": contact_path.stat().st_size,
            }
        metadata["build"]["builder_sha256"] = sha256(Path(__file__))
        metadata["build"]["last_figure_refresh"] = {
            "test_evaluations": 0,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        }
        results_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            "refreshed preprocessing/activation figures; did not train or open test",
            flush=True,
        )
        return

    set_seed()
    shared_head = nn.Linear(512, len(BREEDS))
    shared_head_state = copy.deepcopy(shared_head.state_dict())
    head_initialization_sha256 = hashlib.sha256(
        b"".join(
            tensor.detach().cpu().numpy().tobytes()
            for _, tensor in sorted(shared_head_state.items())
        )
    ).hexdigest()
    results = [
        train_regime(rows, transform, args.weights, shared_head_state, regime)
        for regime in ["probe", "late", "all"]
    ]

    # Regime selection is validation-only: accuracy first, then parsimony,
    # then lower loss and earlier epoch.  No test image has been decoded by the
    # experiment before this decision point.
    selected = max(
        results,
        key=lambda result: (
            result["best"]["val_accuracy"],
            -result["trainable_parameters"],
            -result["best"]["val_loss"],
            -result["best"]["epoch"],
        ),
    )
    print(
        f"validation selected {selected['regime']} at epoch "
        f"{selected['best']['epoch']}; opening sealed test once",
        flush=True,
    )
    selected_model = TransferModel(args.weights, shared_head_state)
    configure(selected_model, selected["regime"])
    selected_model.load_state_dict(selected["state"])
    test_result, predictions = evaluate_sealed_test_once(
        selected_model, rows, transform
    )

    history_rows = [row for result in results for row in result["history"]]
    write_csv(
        args.output_dir / "training-history.csv",
        history_rows,
        [
            "regime",
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ],
    )
    prediction_fields = [
        "manifest_index",
        "filename",
        "true_label",
        "true_breed",
        "predicted_label",
        "predicted_breed",
        "correct",
        "confidence",
    ] + [f"prob_{breed}" for breed in BREEDS]
    write_csv(
        args.output_dir / "sealed-test-predictions.csv",
        predictions,
        prediction_fields,
    )
    features = save_trainval_features(
        rows,
        transform,
        args.weights,
        args.output_dir / "pretrained-trainval-features.npz",
    )
    contact_indices = build_contact_sheet(
        rows, predictions, args.output_dir / "sealed-test-examples.jpg"
    )
    summarized = [summarize_result(result) for result in results]
    selected_summary = next(
        result for result in summarized if result["regime"] == selected["regime"]
    )
    build_curves_svg(
        summarized,
        selected_summary,
        test_result,
        args.output_dir / "transfer-curves.svg",
    )

    generated_paths = [
        manifest_path,
        args.output_dir / "training-history.csv",
        args.output_dir / "sealed-test-predictions.csv",
        args.output_dir / "pretrained-trainval-features.npz",
        args.output_dir / "preprocessing-contract.png",
        args.output_dir / "resnet18-activations.png",
        args.output_dir / "sealed-test-examples.jpg",
        args.output_dir / "transfer-curves.svg",
    ]
    elapsed_seconds = time.perf_counter() - started
    metadata = {
        "evidence_status": "measured fixed-subset teaching experiment; not a benchmark",
        "sealed_test_contract": {
            "regime_and_epoch_selected_from": "validation only",
            "selection_order": [
                "maximum validation accuracy",
                "fewest trainable parameters",
                "minimum validation loss",
                "earliest epoch",
            ],
            "selected_regime": selected["regime"],
            "selected_epoch": selected["best"]["epoch"],
            "test_evaluations": 1,
            "test_evaluated_after_selection": True,
        },
        "dataset": {
            "name": "Oxford-IIIT Pet",
            "dataset_page": DATASET_URL,
            "images_archive": IMAGES_ARCHIVE_URL,
            "annotations_archive": ANNOTATIONS_ARCHIVE_URL,
            "license": "CC BY-SA 4.0; copyright remains with original image owners",
            "breeds": BREEDS,
            "official_boundary": "train and validation are disjoint subselects of official trainval; sealed test is a subselect of official test",
            "selection_population": "official split rows whose JPEG was present under --images-dir when selection-manifest.csv was bootstrapped",
            "selection_rank": f"SHA256('{SEED}:<official_split>:<image_id>')",
            "counts": dict(Counter(row["teaching_split"] for row in rows)),
            "availability_at_bootstrap": availability,
            "official_split_hashes": {
                "trainval.txt": sha256(trainval_txt),
                "test.txt": sha256(test_txt),
            },
            "manifest_sha256": sha256(manifest_path),
        },
        "model": {
            "architecture": "torchvision ResNet-18",
            "weights_enum": "ResNet18_Weights.IMAGENET1K_V1",
            "weights_url": WEIGHTS_URL,
            "weights_sha256": EXPECTED_WEIGHTS_SHA256,
            "feature_dimension": 512,
            "head": "Linear(512, 6)",
            "head_parameters": 512 * len(BREEDS) + len(BREEDS),
            "shared_head_initialization_sha256": head_initialization_sha256,
        },
        "preprocessing": {
            "source": "ResNet18_Weights.IMAGENET1K_V1.transforms()",
            "resize_size": list(transform.resize_size),
            "crop_size": list(transform.crop_size),
            "interpolation": str(transform.interpolation),
            "antialias": bool(transform.antialias),
            "mean": list(transform.mean),
            "std": list(transform.std),
            "augmentation": "none",
            "observed_shared_image": preprocessing,
        },
        "protocol": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "learning_rates": {
                "head_all_regimes": HEAD_LR,
                "late_layer4": LATE_LR,
                "all_backbone": ALL_LR,
            },
            "batch_norm": "backbone kept in eval mode; buffers fixed; affine parameters train only in unfrozen layers",
            "same_training_order_across_regimes": True,
            "same_head_initialization_across_regimes": True,
            "epoch_selection": "max val accuracy, then min val loss, then earliest epoch",
        },
        "regimes": summarized,
        "selected_test": test_result,
        "sealed_test_example_manifest_indices": contact_indices,
        "pretrained_trainval_features": features,
        "activation_figure": {
            "source_image": args.shared_image.name,
            "source_image_sha256": sha256(args.shared_image),
            "semantics": "computed pretrained activations; not saliency",
            "stages": activations,
        },
        "artifacts": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in generated_paths
        },
        "build": {
            "command_template": "python build_transfer_evidence.py --images-dir <official-images> --annotations-dir <official-annotations> --weights <resnet18-f37072fd.pth>",
            "builder_sha256": sha256(Path(__file__)),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "threads": args.threads,
            "deterministic_algorithms": True,
            "elapsed_seconds": elapsed_seconds,
            "numeric_reproducibility_note": "protocol and selection are fixed; floating-point last bits can differ across PyTorch/platform builds",
        },
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"sealed test: {test_result['correct']}/{test_result['count']} = "
        f"{test_result['accuracy']:.3f}; wrote {results_path}; "
        f"elapsed {elapsed_seconds:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
