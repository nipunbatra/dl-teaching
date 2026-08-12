#!/usr/bin/env python3
"""Build licensed real-mask teaching evidence for public Lecture 11.

OBSERVED evidence is the Oxford-IIIT Pet photograph. GT evidence is its
official trimap. The probability field below is a declared deterministic
teaching construction from GT; it is not a trained model output. All masks,
overlays, counts, losses, and metrics are deterministic computations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFilter, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMAGE_PATH = ROOT / "Abyssinian_1.jpg"
TRIMAP_PATH = ROOT / "Abyssinian_1-trimap.png"

EXPECTED_SOURCE_SHA256 = {
    "Abyssinian_1.jpg": "2533197401eebe9410ea4d063f86c43fbd2666f3e8165a38aca155c0d09c21be",
    "Abyssinian_1-trimap.png": "a39ce8ec0363178918cb257844125e3aa7773a3acd4ca8b84780d62c1fa6f220",
}

W, H = 1600, 900
WHITE = (255, 255, 255)
PAPER = (245, 247, 246)
INK = (35, 55, 59)
MUTED = (94, 109, 111)
TEAL = (44, 122, 123)
BLUE = (43, 108, 176)
ORANGE = (235, 129, 27)
GREEN = (20, 176, 61)
RED = (214, 69, 80)
VIOLET = (126, 84, 170)

TRIMAP_VALUES = {1: "foreground", 2: "background", 3: "boundary / ignore"}
THRESHOLD = 0.50
SHIFT_X_PX = 16
FOREGROUND_PROBABILITY = 0.90
BACKGROUND_PROBABILITY = 0.10
BOUNDARY_TOLERANCE_PX = 5.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def font_identity(bold: bool = False) -> str:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    return next((p for p in candidates if Path(p).exists()), "Pillow.load_default()")


def txt(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    body: str,
    *,
    size: int = 28,
    bold: bool = False,
    color: tuple[int, int, int] = INK,
    anchor: str | None = None,
) -> None:
    draw.text(xy, body, font=font(size, bold), fill=color, anchor=anchor)


def rounded_label(
    image: Image.Image,
    xy: tuple[int, int],
    body: str,
    *,
    color: tuple[int, int, int],
) -> None:
    draw = ImageDraw.Draw(image)
    fnt = font(25, True)
    box = draw.textbbox(xy, body, font=fnt)
    pad = 10
    draw.rounded_rectangle(
        (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
        radius=7,
        fill=WHITE,
        outline=color,
        width=3,
    )
    draw.text(xy, body, font=fnt, fill=color)


def title(canvas: Image.Image, headline: str, kicker: str) -> None:
    draw = ImageDraw.Draw(canvas)
    txt(draw, (48, 28), kicker.upper(), size=19, bold=True, color=TEAL)
    txt(draw, (48, 62), headline, size=36, bold=True)
    draw.line((48, 118, W - 48, 118), fill=(190, 199, 197), width=2)


def footer(canvas: Image.Image, body: str) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.line((48, H - 58, W - 48, H - 58), fill=(206, 213, 211), width=2)
    txt(draw, (48, H - 43), body, size=18, color=MUTED)


def cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    x = (resized.width - target_w) // 2
    y = (resized.height - target_h) // 2
    return resized.crop((x, y, x + target_w, y + target_h))


def fit(image: Image.Image, size: tuple[int, int], *, background=WHITE) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, background)
    panel.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return panel


def panel(
    image: Image.Image,
    heading: str,
    subheading: str,
    *,
    size: tuple[int, int] = (470, 610),
    stripe: tuple[int, int, int] = TEAL,
) -> Image.Image:
    canvas = Image.new("RGB", size, WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((1, 1, size[0] - 2, size[1] - 2), radius=10, outline=(190, 199, 197), width=2)
    draw.rectangle((1, 1, 12, size[1] - 2), fill=stripe)
    txt(draw, (28, 20), heading, size=24, bold=True, color=stripe)
    txt(draw, (28, 55), subheading, size=17, color=MUTED)
    content = fit(image, (size[0] - 42, size[1] - 118), background=PAPER)
    canvas.paste(content, (22, 100))
    return canvas


def overlay_rgba(
    photo: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: int,
) -> Image.Image:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, alpha)
    return Image.alpha_composite(photo.convert("RGBA"), Image.fromarray(rgba, "RGBA")).convert("RGB")


def mask_rgb(trimap: np.ndarray) -> Image.Image:
    rgb = np.empty((*trimap.shape, 3), dtype=np.uint8)
    rgb[trimap == 1] = TEAL
    rgb[trimap == 2] = (232, 236, 235)
    rgb[trimap == 3] = ORANGE
    return Image.fromarray(rgb, "RGB")


def probability_rgb(probability: np.ndarray) -> Image.Image:
    low = np.array(WHITE, dtype=np.float64)
    high = np.array(BLUE, dtype=np.float64)
    rgb = (1 - probability[..., None]) * low + probability[..., None] * high
    return Image.fromarray(np.uint8(np.round(rgb)), "RGB")


def nearest_true_distance(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Exact nearest Euclidean distance from every source pixel to target.

    This bounded implementation is intentionally transparent. It is fast for
    thin contours and avoids an undeclared SciPy dependency.
    """
    syx = np.argwhere(source)
    tyx = np.argwhere(target)
    if len(syx) == 0:
        return np.empty(0, dtype=np.float64)
    if len(tyx) == 0:
        return np.full(len(syx), np.inf, dtype=np.float64)
    best_sq = np.full(len(syx), np.inf, dtype=np.float64)
    block = 256
    for start in range(0, len(tyx), block):
        targets = tyx[start:start + block]
        diff = syx[:, None, :] - targets[None, :, :]
        best_sq = np.minimum(best_sq, np.sum(diff * diff, axis=2).min(axis=1))
    return np.sqrt(best_sq)


def constructed_probability(gt: np.ndarray) -> np.ndarray:
    shifted = np.zeros_like(gt)
    shifted[:, SHIFT_X_PX:] = gt[:, :-SHIFT_X_PX]
    return np.where(shifted, FOREGROUND_PROBABILITY, BACKGROUND_PROBABILITY)


def boundary_from_binary(mask: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.uint8(mask) * 255, "L")
    eroded = np.asarray(image.filter(ImageFilter.MinFilter(3))) > 0
    return mask & ~eroded


def build() -> dict[str, object]:
    for source, expected in ((IMAGE_PATH, EXPECTED_SOURCE_SHA256[IMAGE_PATH.name]), (TRIMAP_PATH, EXPECTED_SOURCE_SHA256[TRIMAP_PATH.name])):
        actual = sha256(source)
        assert actual == expected, f"source hash mismatch: {source.name}: {actual}"

    photo = Image.open(IMAGE_PATH).convert("RGB")
    trimap = np.asarray(Image.open(TRIMAP_PATH), dtype=np.uint8)
    assert tuple(photo.size) == (600, 400)
    assert trimap.shape == (400, 600)
    assert set(np.unique(trimap)) == {1, 2, 3}

    gt = trimap == 1
    background = trimap == 2
    ignored = trimap == 3
    valid = ~ignored
    probability = constructed_probability(gt)
    prediction = probability >= THRESHOLD

    tp_mask = valid & gt & prediction
    fp_mask = valid & ~gt & prediction
    fn_mask = valid & gt & ~prediction
    tn_mask = valid & ~gt & ~prediction
    tp, fp, fn, tn = (int(mask.sum()) for mask in (tp_mask, fp_mask, fn_mask, tn_mask))
    assert tp + fp + fn + tn == int(valid.sum())

    eps = np.finfo(np.float64).tiny
    pv = np.clip(probability[valid], eps, 1 - np.finfo(np.float64).eps)
    gv = gt[valid]
    bce = float(-np.mean(gv * np.log(pv) + (~gv) * np.log(1 - pv)))
    soft_dice = float(2 * np.sum(pv * gv) / (np.sum(pv) + np.sum(gv)))
    iou = float(tp / (tp + fp + fn))
    hard_dice = float(2 * tp / (2 * tp + fp + fn))
    accuracy = float((tp + tn) / valid.sum())

    # Counterfactual protocol error: collapse ignored boundary into background.
    all_valid = np.ones_like(valid, dtype=bool)
    naive_tp = int((all_valid & gt & prediction).sum())
    naive_fp = int((all_valid & ~gt & prediction).sum())
    naive_fn = int((all_valid & gt & ~prediction).sum())
    naive_tn = int((all_valid & ~gt & ~prediction).sum())
    naive_iou = float(naive_tp / (naive_tp + naive_fp + naive_fn))
    naive_dice = float(2 * naive_tp / (2 * naive_tp + naive_fp + naive_fn))

    gt_boundary = ignored
    pred_boundary = boundary_from_binary(prediction)
    pred_to_gt = nearest_true_distance(pred_boundary, gt_boundary)
    gt_to_pred = nearest_true_distance(gt_boundary, pred_boundary)
    boundary_precision = float(np.mean(pred_to_gt <= BOUNDARY_TOLERANCE_PX))
    boundary_recall = float(np.mean(gt_to_pred <= BOUNDARY_TOLERANCE_PX))
    boundary_f1 = float(2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall))
    assd = float((pred_to_gt.mean() + gt_to_pred.mean()) / 2)
    hd95 = float(max(np.percentile(pred_to_gt, 95), np.percentile(gt_to_pred, 95)))

    # Plate 1: actual pixels and official trimap contract.
    trimap_overlay = overlay_rgba(photo, gt, GREEN, 105)
    trimap_overlay = overlay_rgba(trimap_overlay, ignored, ORANGE, 210)
    trimap_plate = Image.new("RGB", (W, H), PAPER)
    title(trimap_plate, "One real photograph, one official pixel contract", "OBSERVED pixels + GT annotation")
    p1 = panel(photo, "OBSERVED", "Oxford-IIIT Pet RGB photograph", stripe=INK)
    p2 = panel(mask_rgb(trimap), "GT VALUES", "1 foreground · 2 background · 3 boundary", stripe=TEAL)
    p3 = panel(trimap_overlay, "GT OVERLAY", "value 3 is boundary / ignore", stripe=ORANGE)
    for x, item in zip((48, 565, 1082), (p1, p2, p3)):
        trimap_plate.paste(item, (x, 150))
    footer(trimap_plate, "Region losses/metrics use only values 1 and 2. Value 3 is excluded from every confusion count and region denominator.")
    trimap_plate.save(HERE / "real-trimap-contract.png", optimize=True)

    # Plate 2: deterministic construction, not a model output.
    probability_image = probability_rgb(probability)
    hard_rgb = np.full((*prediction.shape, 3), WHITE, dtype=np.uint8)
    hard_rgb[prediction] = BLUE
    hard_image = Image.fromarray(hard_rgb, "RGB")
    constructed_plate = Image.new("RGB", (W, H), PAPER)
    title(constructed_plate, "A declared construction crosses TRAIN, INFER, and EVAL", "CONSTRUCTED probability + COMPUTED mask")
    gt_panel = panel(mask_rgb(trimap), "GT", "official trimap; boundary ignored", stripe=TEAL)
    prob_panel = panel(probability_image, "CONSTRUCTED p", "official foreground shifted +16 px", stripe=BLUE)
    pred_panel = panel(hard_image, "COMPUTED P", "P = 1[p >= 0.50]", stripe=ORANGE)
    for x, item in zip((48, 565, 1082), (gt_panel, prob_panel, pred_panel)):
        constructed_plate.paste(item, (x, 150))
    footer(constructed_plate, "No checkpoint was executed. This field is pedagogical construction, not measured performance or model output.")
    constructed_plate.save(HERE / "constructed-probability-mask.png", optimize=True)

    # Plate 3: error overlay on actual pixels.
    error_rgba = np.zeros((*gt.shape, 4), dtype=np.uint8)
    error_rgba[tp_mask] = (*GREEN, 112)
    error_rgba[fp_mask] = (*RED, 220)
    error_rgba[fn_mask] = (*VIOLET, 225)
    error_rgba[ignored] = (*ORANGE, 175)
    error_overlay = Image.alpha_composite(photo.convert("RGBA"), Image.fromarray(error_rgba, "RGBA")).convert("RGB")
    error_plate = Image.new("RGB", (W, H), PAPER)
    title(error_plate, "Region counts become visible on the real photograph", "COMPUTED overlay")
    large = cover(error_overlay, (1080, 650))
    error_plate.paste(large, (48, 150))
    draw = ImageDraw.Draw(error_plate)
    draw.rounded_rectangle((76, 174, 640, 352), radius=9, fill=(255, 255, 255, 238), outline=INK, width=2)
    txt(draw, (98, 193), "PIXEL STATUS", size=19, bold=True, color=INK)
    overlay_legend = [
        (GREEN, "TP · valid foreground retained"),
        (RED, "FP · foreground on valid background"),
        (VIOLET, "FN · valid foreground missed"),
        (ORANGE, "IGNORE · official boundary excluded"),
    ]
    for i, (color, body) in enumerate(overlay_legend):
        y = 232 + i * 28
        draw.rectangle((98, y + 2, 118, y + 20), fill=color, outline=INK, width=1)
        txt(draw, (129, y), body, size=17, bold=True, color=INK)
    draw.rounded_rectangle((1160, 150, 1552, 800), radius=10, fill=WHITE, outline=(190, 199, 197), width=2)
    txt(draw, (1190, 175), "VALID PIXELS", size=20, bold=True, color=TEAL)
    legend = [
        (GREEN, "TP", f"{tp:,}"),
        (RED, "FP", f"{fp:,}"),
        (VIOLET, "FN", f"{fn:,}"),
        (INK, "TN", f"{tn:,}"),
    ]
    for i, (color, name, value) in enumerate(legend):
        y = 225 + i * 52
        draw.rectangle((1190, y + 3, 1214, y + 27), fill=color, outline=INK, width=1)
        txt(draw, (1227, y), name, size=24, bold=True, color=color)
        txt(draw, (1520, y), value, size=28, bold=True, color=color, anchor="ra")
    draw.line((1190, 445, 1522, 445), fill=(206, 213, 211), width=2)
    txt(draw, (1190, 475), f"IoU  {iou:.3f}", size=29, bold=True)
    txt(draw, (1190, 525), f"Dice  {hard_dice:.3f}", size=29, bold=True)
    txt(draw, (1190, 595), f"Ignored  {int(ignored.sum()):,}", size=20, bold=True, color=ORANGE)
    txt(draw, (1190, 632), "not TN; not scored", size=18, color=MUTED)
    txt(draw, (1190, 704), "green TP · red FP", size=17, color=MUTED)
    txt(draw, (1190, 734), "violet FN · orange ignore", size=17, color=MUTED)
    footer(error_plate, "GT and photo are real. Probability, prediction, error colors, counts, and metrics are computed from the declared construction.")
    error_plate.save(HERE / "real-error-overlay.png", optimize=True)

    # Plate 4: region and boundary metrics use distinct contracts.
    audit = Image.new("RGB", (W, H), PAPER)
    title(audit, "Region overlap can be strong while boundaries still shift", "COMPUTED audit")
    draw = ImageDraw.Draw(audit)
    cards = [
        (48, 160, 755, 610, TEAL, "REGION · valid pixels only"),
        (845, 160, 1552, 610, ORANGE, "BOUNDARY · contour distances"),
    ]
    for x0, y0, x1, y1, color, heading in cards:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=13, fill=WHITE, outline=color, width=3)
        txt(draw, (x0 + 28, y0 + 25), heading, size=23, bold=True, color=color)
    region_rows = [
        ("BCE", f"{bce:.4f}"),
        ("Soft Dice", f"{soft_dice:.4f}"),
        ("IoU", f"{iou:.4f}"),
        ("Hard Dice", f"{hard_dice:.4f}"),
        ("Accuracy", f"{accuracy:.4f}"),
    ]
    boundary_rows = [
        (f"Precision @ {BOUNDARY_TOLERANCE_PX:.0f}px", f"{boundary_precision:.4f}"),
        (f"Recall @ {BOUNDARY_TOLERANCE_PX:.0f}px", f"{boundary_recall:.4f}"),
        (f"Boundary F1 @ {BOUNDARY_TOLERANCE_PX:.0f}px", f"{boundary_f1:.4f}"),
        ("ASSD (px)", f"{assd:.3f}"),
        ("HD95 (px)", f"{hd95:.3f}"),
    ]
    for rows, x0 in ((region_rows, 48), (boundary_rows, 845)):
        for i, (name, value) in enumerate(rows):
            y = 235 + i * 65
            txt(draw, (x0 + 30, y), name, size=22, color=MUTED)
            txt(draw, (x0 + 665, y), value, size=27, bold=True, anchor="ra")
            if i < len(rows) - 1:
                draw.line((x0 + 30, y + 44, x0 + 675, y + 44), fill=(224, 228, 227), width=2)
    draw.rounded_rectangle((48, 650, 1552, 790), radius=12, fill=WHITE, outline=INK, width=2)
    txt(draw, (78, 680), "Two contracts, one lesson", size=24, bold=True)
    txt(draw, (78, 725), "IoU/Dice reduce valid-region confusion. Boundary F1/ASSD/HD95 compare contours; the ignored GT band is the reference contour.", size=23)
    footer(audit, "Boundary metrics are a diagnostic extension; they are not part of the deck's exact 4 x 4 numerical spine.")
    audit.save(HERE / "region-boundary-audit.png", optimize=True)

    metrics = {
        "valid_count": int(valid.sum()),
        "foreground_count": int(gt.sum()),
        "background_count": int(background.sum()),
        "boundary_ignore_count": int(ignored.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": THRESHOLD,
        "bce": bce,
        "soft_dice": soft_dice,
        "iou": iou,
        "hard_dice": hard_dice,
        "accuracy": accuracy,
        "boundary_tolerance_px": BOUNDARY_TOLERANCE_PX,
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": boundary_f1,
        "assd_px": assd,
        "hd95_px": hd95,
        "naive_boundary_as_background_tp": naive_tp,
        "naive_boundary_as_background_fp": naive_fp,
        "naive_boundary_as_background_fn": naive_fn,
        "naive_boundary_as_background_tn": naive_tn,
        "naive_boundary_as_background_iou": naive_iou,
        "naive_boundary_as_background_dice": naive_dice,
    }
    # Store arrays as a deterministic, uncompressed NumPy zip. NumPy's default
    # ZIP timestamps vary between runs, so normalize every member timestamp.
    npz_path = HERE / "real-mask-arrays.npz"
    np.savez(
        npz_path,
        trimap=trimap.astype(np.uint8),
        valid=valid.astype(np.bool_),
        gt=gt.astype(np.bool_),
        probability=probability.astype(np.float64),
        prediction=prediction.astype(np.bool_),
    )
    import zipfile
    deterministic_npz = npz_path.with_suffix(".normalized.npz")
    with zipfile.ZipFile(npz_path, "r") as source_zip, zipfile.ZipFile(deterministic_npz, "w", compression=zipfile.ZIP_STORED) as target_zip:
        for member in sorted(source_zip.namelist()):
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            target_zip.writestr(info, source_zip.read(member))
    deterministic_npz.replace(npz_path)
    with (HERE / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value", "scope"))
        for name, value in metrics.items():
            scope = "boundary diagnostic" if name.startswith("boundary_") or name.endswith("_px") else "valid region"
            writer.writerow((name, value, scope))

    derived_paths = [
        HERE / "real-trimap-contract.png",
        HERE / "constructed-probability-mask.png",
        HERE / "real-error-overlay.png",
        HERE / "region-boundary-audit.png",
        HERE / "metrics.csv",
        HERE / "real-mask-arrays.npz",
    ]
    metadata: dict[str, object] = {
        "schema_version": 1,
        "source": {
            "dataset": "Oxford-IIIT Pet",
            "image_id": "Abyssinian_1",
            "image_path": "../Abyssinian_1.jpg",
            "trimap_path": "../Abyssinian_1-trimap.png",
            "image_size_wh": list(photo.size),
            "license": "CC BY-SA 4.0; copyright remains with the original image owner",
            "dataset_page": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
            "source_sha256": {path.name: sha256(path) for path in (IMAGE_PATH, TRIMAP_PATH)},
        },
        "taxonomy": {
            "observed": ["original Oxford-IIIT Pet RGB pixels"],
            "ground_truth": ["official trimap values and masks"],
            "constructed": ["probability field: official foreground shifted 16 px right with zero fill"],
            "computed": ["hard mask", "overlays", "confusion counts", "losses", "region metrics", "boundary diagnostics"],
            "model_output": [],
            "measured_performance": [],
        },
        "mask_policy": {
            "official_values": {str(key): value for key, value in TRIMAP_VALUES.items()},
            "valid_values": [1, 2],
            "ignored_values": [3],
            "foreground_definition": "trimap == 1",
            "background_definition": "trimap == 2",
            "ignore_definition": "trimap == 3",
            "region_rule": "exclude ignored pixels from BCE, Soft Dice, confusion, accuracy, IoU, and hard Dice",
            "boundary_rule": "use official value-3 band as GT reference contour only for boundary-distance diagnostics",
        },
        "construction": {
            "input": "binary GT foreground mask including all official value-1 pixels",
            "shift_xy_px": [SHIFT_X_PX, 0],
            "zero_fill_after_shift": True,
            "foreground_probability": FOREGROUND_PROBABILITY,
            "background_probability": BACKGROUND_PROBABILITY,
            "threshold": THRESHOLD,
            "threshold_comparison": ">=",
            "claim": "deterministic pedagogical construction; not model output",
        },
        "metrics": metrics,
        "boundary_metric_policy": {
            "predicted_contour": "one-pixel inner boundary: P AND NOT min_filter_3x3(P)",
            "ground_truth_contour": "all official trimap value-3 pixels",
            "distance": "exact Euclidean pixel-center distance",
            "f1_tolerance_px": BOUNDARY_TOLERANCE_PX,
            "assd": "mean of directed pred-to-GT mean and GT-to-pred mean",
            "hd95": "maximum of the two directed 95th-percentile distances",
        },
        "deck_contract": {
            "exact_4x4_spine": "separate constructed case in lecture source; not derived from this real image",
            "role": "this bundle grounds masks, ignore policy, region errors, and boundary diagnostics in actual pixels/GT",
        },
        "arrays": {
            "path": "real-mask-arrays.npz",
            "compressed_npz": False,
            "deterministic_zip_timestamp": "1980-01-01T00:00:00",
            "members": {
                "trimap": {"dtype": "uint8", "shape": list(trimap.shape), "meaning": "official values 1/2/3"},
                "valid": {"dtype": "bool", "shape": list(valid.shape), "meaning": "trimap != 3"},
                "gt": {"dtype": "bool", "shape": list(gt.shape), "meaning": "trimap == 1"},
                "probability": {"dtype": "float64", "shape": list(probability.shape), "meaning": "constructed foreground probability"},
                "prediction": {"dtype": "bool", "shape": list(prediction.shape), "meaning": "probability >= 0.50"},
            },
        },
        "build": {
            "command": "uv run --with numpy==2.4.6 --with pillow==12.3.0 python shared/vision-evidence/oxford-iiit-pet/l10/build_segmentation_evidence.py",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "font_regular": font_identity(False),
            "font_bold": font_identity(True),
            "builder_sha256": sha256(Path(__file__)),
            "derived_sha256": {path.name: sha256(path) for path in derived_paths},
        },
    }
    (HERE / "evidence.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    result = build()
    print(json.dumps(result["metrics"], indent=2))
