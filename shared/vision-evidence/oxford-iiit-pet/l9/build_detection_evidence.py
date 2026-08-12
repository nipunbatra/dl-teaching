#!/usr/bin/env python3
"""Build a real two-image detection mini-batch for public Lecture 10.

Observed evidence: two Oxford-IIIT Pet photographs and their official XML head
ROIs. Pedagogical construction: candidates A--E, scores, crops, and affine
placement. Computed evidence: IoU, per-image NMS, matching, AP, and overlays.
No candidate is a detector/model output.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree

import PIL
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SAMPLES = {
    "I1": {
        "stem": "Abyssinian_1",
        "breed": "Abyssinian",
        "image": ROOT / "Abyssinian_1.jpg",
        "xml": ROOT / "Abyssinian_1.xml",
        "image_sha256": "2533197401eebe9410ea4d063f86c43fbd2666f3e8165a38aca155c0d09c21be",
        "xml_sha256": "bea13ddcf1dc171e39da5ce2c134b1dc8c247a9f1bb21cc2056f37f7a423a51f",
        "expected_size": (600, 400),
        "expected_gt": (332, 71, 425, 158),
        "gt_id": "G1",
        "canonical_gt": (10, 10, 50, 50),
        "view": (300, 40, 545, 230),
    },
    "I2": {
        "stem": "Bengal_10",
        "breed": "Bengal",
        "image": HERE / "Bengal_10.jpg",
        "xml": HERE / "Bengal_10.xml",
        "image_sha256": "4cc958fc6bb47738488fdc83a3053f01ce2937c9fbcaf5db2e649a1d612f8725",
        "xml_sha256": "9a6e7d513d999b23cde839a6efce908e2e1472d14d69f13d671e9db9cdb09ae6",
        "expected_size": (500, 375),
        "expected_gt": (314, 59, 446, 219),
        "gt_id": "G2",
        "canonical_gt": (60, 15, 90, 55),
        "view": (255, 40, 500, 230),
    },
}

CANDIDATES = {
    "A": {"image": "I1", "score": Fraction(95, 100), "box": (10, 10, 50, 50)},
    "B": {"image": "I1", "score": Fraction(90, 100), "box": (12, 12, 52, 52)},
    "E": {"image": "I1", "score": Fraction(89, 100), "box": (35, 5, 55, 25)},
    "C": {"image": "I2", "score": Fraction(88, 100), "box": (60, 15, 90, 55)},
    "D": {"image": "I2", "score": Fraction(75, 100), "box": (63, 17, 91, 56)},
}
RAW_ORDER = ("A", "B", "E", "C", "D")
NMS_THRESHOLD = Fraction(1, 2)
EVAL_THRESHOLD = Fraction(1, 2)

BG = (244, 242, 238)
WHITE = (255, 255, 255)
INK = (35, 55, 59)
MUTED = (82, 102, 106)
TEAL = (44, 122, 123)
ORANGE = (235, 129, 27)
BLUE = (43, 108, 176)
GREEN = (20, 176, 61)
RED = (214, 69, 80)
PURPLE = (128, 90, 213)
LIGHT_TEAL = (229, 241, 239)
LIGHT_RED = (252, 235, 237)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
        if bold
        else Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default(size=size)


def font_identity(*, bold: bool = False) -> str:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
        if bold
        else Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "Pillow.load_default()"


def txt(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    body: str,
    *,
    size: int,
    color: tuple[int, int, int] = INK,
    bold: bool = False,
    anchor: str | None = None,
) -> None:
    draw.text(xy, body, font=font(size, bold=bold), fill=color, anchor=anchor)


def wrap(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    body: str,
    *,
    width: int,
    size: int,
    color: tuple[int, int, int] = INK,
    gap: int = 7,
) -> int:
    words = body.split()
    lines: list[str] = []
    current = ""
    fnt = font(size)
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), trial, font=fnt)[2] <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=color)
        y += size + gap
    return y


def card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int] = WHITE,
    outline: tuple[int, int, int] = (215, 220, 218),
    radius: int = 18,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def tag(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    body: str,
    *,
    color: tuple[int, int, int],
    size: int = 22,
) -> None:
    fnt = font(size, bold=True)
    bounds = draw.textbbox(xy, body, font=fnt)
    draw.rounded_rectangle(
        (bounds[0] - 8, bounds[1] - 4, bounds[2] + 8, bounds[3] + 4),
        radius=7,
        fill=WHITE,
        outline=color,
        width=3,
    )
    draw.text(xy, body, font=fnt, fill=color)


def dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int],
    width: int,
    dash: int = 18,
    gap: int = 11,
) -> None:
    x0, y0, x1, y1 = box
    for start in range(x0, x1, dash + gap):
        draw.line((start, y0, min(start + dash, x1), y0), fill=color, width=width)
        draw.line((start, y1, min(start + dash, x1), y1), fill=color, width=width)
    for start in range(y0, y1, dash + gap):
        draw.line((x0, start, x0, min(start + dash, y1)), fill=color, width=width)
        draw.line((x1, start, x1, min(start + dash, y1)), fill=color, width=width)


def cross(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], *, width: int = 5) -> None:
    x0, y0, x1, y1 = box
    draw.line((x0, y0, x1, y1), fill=RED, width=width)
    draw.line((x0, y1, x1, y0), fill=RED, width=width)


def load_xml_box(path: Path) -> tuple[int, int, int, int]:
    """Convert VOC 1-based inclusive coordinates to 0-based half-open xyxy."""
    node = ElementTree.parse(path).getroot().find("object/bndbox")
    assert node is not None
    xmin, ymin, xmax, ymax = (
        int(node.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")
    )
    return xmin - 1, ymin - 1, xmax, ymax


def area(box: tuple[int, int, int, int]) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection_area(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> Fraction:
    inter = intersection_area(a, b)
    union = area(a) + area(b) - inter
    return Fraction(inter, union) if union else Fraction(0)


def exact(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def dec(value: Fraction, digits: int = 6) -> float:
    return round(float(value), digits)


def source_box(
    canonical_box: tuple[int, int, int, int],
    canonical_gt: tuple[int, int, int, int],
    source_gt: tuple[int, int, int, int],
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    cw = canonical_gt[2] - canonical_gt[0]
    ch = canonical_gt[3] - canonical_gt[1]
    sx = Fraction(source_gt[2] - source_gt[0], cw)
    sy = Fraction(source_gt[3] - source_gt[1], ch)
    x0, y0, x1, y1 = canonical_box
    return (
        Fraction(source_gt[0]) + (x0 - canonical_gt[0]) * sx,
        Fraction(source_gt[1]) + (y0 - canonical_gt[1]) * sy,
        Fraction(source_gt[0]) + (x1 - canonical_gt[0]) * sx,
        Fraction(source_gt[1]) + (y1 - canonical_gt[1]) * sy,
    )


def nms() -> tuple[list[str], list[dict[str, object]]]:
    kept: list[str] = []
    trace: list[dict[str, object]] = []
    for image_id in ("I1", "I2"):
        pending = [name for name in RAW_ORDER if CANDIDATES[name]["image"] == image_id]
        while pending:
            current = pending.pop(0)
            kept.append(current)
            trace.append(
                {
                    "image": image_id,
                    "action": "keep",
                    "candidate": current,
                    "reference": None,
                    "iou": None,
                    "reason": "highest score remaining in this image",
                }
            )
            survivors: list[str] = []
            for other in pending:
                overlap = iou(CANDIDATES[current]["box"], CANDIDATES[other]["box"])
                if overlap > NMS_THRESHOLD:
                    trace.append(
                        {
                            "image": image_id,
                            "action": "suppress",
                            "candidate": other,
                            "reference": current,
                            "iou": overlap,
                            "reason": "same image/class and IoU > 0.50",
                        }
                    )
                else:
                    survivors.append(other)
            pending = survivors
    kept.sort(key=lambda name: CANDIDATES[name]["score"], reverse=True)
    return kept, trace


def evaluate(kept: list[str]) -> tuple[list[dict[str, object]], Fraction]:
    available = {"I1": True, "I2": True}
    rows: list[dict[str, object]] = []
    tp = 0
    precisions_at_tp: list[Fraction] = []
    for rank, name in enumerate(kept, start=1):
        image_id = CANDIDATES[name]["image"]
        gt = SAMPLES[image_id]["canonical_gt"]
        overlap = iou(CANDIDATES[name]["box"], gt)
        is_tp = bool(available[image_id] and overlap >= EVAL_THRESHOLD)
        if is_tp:
            available[image_id] = False
            tp += 1
            precisions_at_tp.append(Fraction(tp, rank))
        rows.append(
            {
                "rank": rank,
                "candidate": name,
                "image": image_id,
                "gt": SAMPLES[image_id]["gt_id"],
                "iou": overlap,
                "result": "TP" if is_tp else "FP",
                "cumulative_tp": tp,
                "cumulative_fp": rank - tp,
                "precision": Fraction(tp, rank),
                "recall": Fraction(tp, 2),
            }
        )
    ap = sum(precisions_at_tp, start=Fraction(0)) / 2
    return rows, ap


def load_sources() -> tuple[dict[str, Image.Image], dict[str, tuple[int, int, int, int]]]:
    photos: dict[str, Image.Image] = {}
    gts: dict[str, tuple[int, int, int, int]] = {}
    for image_id, sample in SAMPLES.items():
        image_path, xml_path = sample["image"], sample["xml"]
        assert sha256(image_path) == sample["image_sha256"], f"unexpected {image_id} JPEG"
        assert sha256(xml_path) == sample["xml_sha256"], f"unexpected {image_id} XML"
        photos[image_id] = Image.open(image_path).convert("RGB")
        assert photos[image_id].size == sample["expected_size"]
        gts[image_id] = load_xml_box(xml_path)
        assert gts[image_id] == sample["expected_gt"]
    return photos, gts


def source_boxes(
    gts: dict[str, tuple[int, int, int, int]]
) -> dict[str, tuple[Fraction, Fraction, Fraction, Fraction]]:
    result = {}
    for name, candidate in CANDIDATES.items():
        image_id = candidate["image"]
        result[name] = source_box(
            candidate["box"], SAMPLES[image_id]["canonical_gt"], gts[image_id]
        )
    return result


def crop_photo(photo: Image.Image, image_id: str, size: tuple[int, int]) -> Image.Image:
    return photo.crop(SAMPLES[image_id]["view"]).resize(size, Image.Resampling.LANCZOS)


def crop_coords(
    image_id: str,
    box: tuple[Fraction, Fraction, Fraction, Fraction],
    size: tuple[int, int],
    offset: tuple[int, int],
) -> tuple[int, int, int, int]:
    view = SAMPLES[image_id]["view"]
    sx = Fraction(size[0], view[2] - view[0])
    sy = Fraction(size[1], view[3] - view[1])
    return tuple(
        round(float((value - origin) * scale + shift))
        for value, origin, scale, shift in (
            (box[0], view[0], sx, offset[0]),
            (box[1], view[1], sy, offset[1]),
            (box[2], view[0], sx, offset[0]),
            (box[3], view[1], sy, offset[1]),
        )
    )  # type: ignore[return-value]


def fit_photo(
    canvas: Image.Image,
    photo: Image.Image,
    box: tuple[int, int, int, int],
) -> tuple[float, int, int]:
    max_w, max_h = box[2] - box[0], box[3] - box[1]
    scale = min(max_w / photo.width, max_h / photo.height)
    size = (round(photo.width * scale), round(photo.height * scale))
    shown = photo.resize(size, Image.Resampling.LANCZOS)
    x = box[0] + (max_w - size[0]) // 2
    y = box[1] + (max_h - size[1]) // 2
    canvas.paste(shown, (x, y))
    return scale, x, y


def build_targets_plate(
    photos: dict[str, Image.Image], gts: dict[str, tuple[int, int, int, int]]
) -> Path:
    canvas = Image.new("RGB", (1600, 900), BG)
    draw = ImageDraw.Draw(canvas)
    txt(draw, (45, 35), "A REAL TWO-IMAGE MINI-BATCH", size=43, bold=True)
    txt(draw, (47, 92), "Two photographs · two official XML head boxes", size=24, color=MUTED)
    for x, image_id in ((35, "I1"), (815, "I2")):
        sample = SAMPLES[image_id]
        card(draw, (x, 140, x + 750, 850))
        txt(draw, (x + 28, 170), f"{image_id} · {sample['stem']}", size=28, bold=True)
        txt(draw, (x + 28, 210), "OBSERVED PHOTO + XML GT", size=19, color=TEAL, bold=True)
        scale, px, py = fit_photo(canvas, photos[image_id], (x + 25, 255, x + 725, 705))
        gt = gts[image_id]
        display = (
            round(gt[0] * scale + px),
            round(gt[1] * scale + py),
            round(gt[2] * scale + px),
            round(gt[3] * scale + py),
        )
        dashed_rectangle(draw, display, color=TEAL, width=8, dash=22, gap=13)
        tag(draw, (display[0] + 10, max(py + 10, display[1] - 38)), sample["gt_id"], color=TEAL)
        voc = [gt[0] + 1, gt[1] + 1, gt[2], gt[3]]
        txt(draw, (x + 30, 742), f"VOC XML  ({voc[0]}, {voc[1]})–({voc[2]}, {voc[3]})", size=20, color=MUTED)
        txt(draw, (x + 30, 778), f"course xyxy  {list(gt)}", size=22, bold=True)
        txt(
            draw,
            (x + 30, 815),
            f"{gt[2]-gt[0]} × {gt[3]-gt[1]} source pixels · ground truth, not model output",
            size=18,
            color=TEAL,
        )
    output = HERE / "real-head-targets.png"
    canvas.save(output, optimize=True)
    return output


def build_candidates_plate(
    photos: dict[str, Image.Image],
    gts: dict[str, tuple[int, int, int, int]],
    boxes: dict[str, tuple[Fraction, Fraction, Fraction, Fraction]],
) -> Path:
    canvas = Image.new("RGB", (1600, 900), BG)
    draw = ImageDraw.Draw(canvas)
    txt(draw, (42, 30), "THE SAME A–E SPINE, NOW ON TWO REAL PHOTOS", size=41, bold=True)
    txt(draw, (44, 82), "Candidates/scores are constructed; XML boxes are observed", size=23, color=MUTED)
    colors = {"A": GREEN, "B": ORANGE, "E": BLUE, "C": PURPLE, "D": RED}
    assignments = {"I1": ("A", "B", "E"), "I2": ("C", "D")}
    for x, image_id in ((30, "I1"), (810, "I2")):
        card(draw, (x, 125, x + 760, 855))
        txt(draw, (x + 25, 155), f"{image_id} · {SAMPLES[image_id]['stem']}", size=26, bold=True)
        txt(draw, (x + 25, 193), f"{SAMPLES[image_id]['gt_id']} + {'/'.join(assignments[image_id])}", size=19, color=TEAL)
        size, offset = (710, 551), (x + 25, 235)
        canvas.paste(crop_photo(photos[image_id], image_id, size), offset)
        for name in reversed(assignments[image_id]):
            display = crop_coords(image_id, boxes[name], size, offset)
            draw.rectangle(display, outline=colors[name], width=7)
        display_gt = crop_coords(
            image_id, tuple(Fraction(v) for v in gts[image_id]), size, offset
        )
        dashed_rectangle(draw, display_gt, color=TEAL, width=10, dash=24, gap=15)
        for index, name in enumerate(assignments[image_id]):
            tag(
                draw,
                (x + 45 + index * 155, 805),
                f"{name}  {float(CANDIDATES[name]['score']):.2f}",
                color=colors[name],
                size=20,
            )
    txt(
        draw,
        (45, 870),
        "A=G1 and C=G2. Affine maps preserve IoU; rendering rounds boundaries only for display.",
        size=18,
        color=MUTED,
    )
    output = HERE / "real-candidates.png"
    canvas.save(output, optimize=True)
    return output


def nms_panel(
    canvas: Image.Image,
    photos: dict[str, Image.Image],
    gts: dict[str, tuple[int, int, int, int]],
    boxes: dict[str, tuple[Fraction, Fraction, Fraction, Fraction]],
    *,
    x: int,
    image_id: str,
    title: str,
    candidates: tuple[str, ...],
    kept: set[str],
    suppressed: set[str],
) -> None:
    draw = ImageDraw.Draw(canvas)
    card(draw, (x, 145, x + 365, 760))
    txt(draw, (x + 20, 175), title, size=23, bold=True)
    txt(draw, (x + 20, 210), SAMPLES[image_id]["stem"], size=17, color=MUTED)
    size, offset = (325, 252), (x + 20, 260)
    canvas.paste(crop_photo(photos[image_id], image_id, size), offset)
    for name in candidates:
        display = crop_coords(image_id, boxes[name], size, offset)
        color = GREEN if name in kept else RED if name in suppressed else MUTED
        draw.rectangle(display, outline=color, width=5 if name in kept else 4)
        if name in suppressed:
            cross(draw, display, width=4)
    # Deliberately do not draw the XML truth here: NMS compares predictions
    # within (image_id, class) and has no access to ground truth.
    y = 550
    for label, values, color in (
        ("KEEP", ", ".join(n for n in candidates if n in kept) or "—", GREEN),
        ("SUPPRESS", ", ".join(n for n in candidates if n in suppressed) or "—", RED),
        ("PENDING", ", ".join(n for n in candidates if n not in kept | suppressed) or "—", MUTED),
    ):
        txt(draw, (x + 22, y), label, size=16, color=color, bold=True)
        txt(draw, (x + 145, y), values, size=18)
        y += 48


def build_nms_plate(
    photos: dict[str, Image.Image],
    gts: dict[str, tuple[int, int, int, int]],
    boxes: dict[str, tuple[Fraction, Fraction, Fraction, Fraction]],
) -> Path:
    canvas = Image.new("RGB", (1600, 900), BG)
    draw = ImageDraw.Draw(canvas)
    txt(draw, (42, 32), "NMS RUNS WITHIN EACH IMAGE", size=43, bold=True)
    txt(draw, (44, 87), "Same class · score order · IoU > 0.50 · ground truth never participates", size=22, color=MUTED)
    nms_panel(canvas, photos, gts, boxes, x=35, image_id="I1", title="I1 · RAW A/B/E", candidates=("A", "B", "E"), kept=set(), suppressed=set())
    nms_panel(canvas, photos, gts, boxes, x=430, image_id="I1", title="I1 · RETURN A/E", candidates=("A", "B", "E"), kept={"A", "E"}, suppressed={"B"})
    nms_panel(canvas, photos, gts, boxes, x=825, image_id="I2", title="I2 · RAW C/D", candidates=("C", "D"), kept=set(), suppressed=set())
    nms_panel(canvas, photos, gts, boxes, x=1220, image_id="I2", title="I2 · RETURN C", candidates=("C", "D"), kept={"C"}, suppressed={"D"})
    txt(draw, (50, 800), "IoU(A,B) = 1444/1756 = 0.822", size=22, bold=True)
    txt(draw, (620, 800), "IoU(C,D) = 1026/1266 = 0.810", size=22, bold=True)
    txt(draw, (1190, 800), "BATCH → A, E, C", size=23, color=GREEN, bold=True)
    wrap(
        draw,
        (50, 840),
        "No cross-image suppression: A can never suppress C. E survives because it is isolated, not because it is correct.",
        width=1480,
        size=18,
        color=MUTED,
        gap=4,
    )
    output = HERE / "real-nms-trace.png"
    canvas.save(output, optimize=True)
    return output


def build_eval_plate(
    photos: dict[str, Image.Image],
    gts: dict[str, tuple[int, int, int, int]],
    boxes: dict[str, tuple[Fraction, Fraction, Fraction, Fraction]],
    rows: list[dict[str, object]],
    ap: Fraction,
) -> Path:
    canvas = Image.new("RGB", (1600, 900), BG)
    draw = ImageDraw.Draw(canvas)
    txt(draw, (42, 30), "MATCH THE RETURNED BATCH TO BOTH REAL TRUTHS", size=40, bold=True)
    txt(draw, (44, 82), "Global score order A, E, C · one claim per XML box", size=23, color=MUTED)
    for x, image_id, names in ((30, "I1", ("A", "E")), (555, "I2", ("C",))):
        card(draw, (x, 130, x + 500, 665))
        txt(draw, (x + 22, 158), f"{image_id} · {SAMPLES[image_id]['gt_id']}", size=24, bold=True)
        size, offset = (460, 357), (x + 20, 205)
        canvas.paste(crop_photo(photos[image_id], image_id, size), offset)
        for name in names:
            display = crop_coords(image_id, boxes[name], size, offset)
            color = GREEN if name in ("A", "C") else RED
            draw.rectangle(display, outline=color, width=7)
        display_gt = crop_coords(image_id, tuple(Fraction(v) for v in gts[image_id]), size, offset)
        dashed_rectangle(draw, display_gt, color=TEAL, width=9, dash=20, gap=12)
        footer = "A = TP · E = FP" if image_id == "I1" else "C = TP"
        txt(draw, (x + 25, 610), footer, size=23, color=GREEN if image_id == "I2" else INK, bold=True)

    card(draw, (1080, 130, 1570, 860))
    txt(draw, (1115, 165), "RANKED MATCHING", size=23, bold=True)
    headers = ("rank", "box", "IoU", "result")
    xs = (1115, 1195, 1320, 1450)
    for x, header in zip(xs, headers):
        txt(draw, (x, 220), header.upper(), size=15, color=MUTED, bold=True)
    draw.line((1110, 252, 1535, 252), fill=(210, 216, 214), width=2)
    y = 280
    for row in rows:
        values = (str(row["rank"]), f"{row['candidate']} {float(CANDIDATES[row['candidate']]['score']):.2f}", f"{float(row['iou']):.3f}")
        for x, value in zip(xs[:3], values):
            txt(draw, (x, y), value, size=20)
        color = GREEN if row["result"] == "TP" else RED
        txt(draw, (xs[3], y), row["result"], size=20, color=color, bold=True)
        y += 67
    draw.rounded_rectangle((1110, 505, 1535, 635), radius=15, fill=LIGHT_TEAL)
    txt(draw, (1140, 532), "AVERAGE PRECISION", size=17, color=TEAL, bold=True)
    txt(draw, (1140, 575), "AP = (1 + 2/3) / 2", size=24, bold=True)
    txt(draw, (1140, 609), f"= {exact(ap)} ≈ {float(ap):.3f}", size=24, color=TEAL, bold=True)
    draw.rounded_rectangle((1110, 675, 1535, 785), radius=15, fill=LIGHT_RED)
    txt(draw, (1140, 700), "FINAL COUNTS", size=17, color=RED, bold=True)
    txt(draw, (1140, 744), "2 TP · 1 FP · 0 FN", size=25, bold=True)
    txt(draw, (1115, 812), "Two-image teaching batch; not a benchmark.", size=17, color=MUTED)
    txt(draw, (45, 710), "E overlaps G1 by only 9/71 ≈ 0.127; NMS could not know it was wrong.", size=21, color=RED, bold=True)
    wrap(
        draw,
        (45, 755),
        "Evidence taxonomy: photos/XML = observed · candidates/scores/crops = constructed · IoU/NMS/matching/AP = computed · model output = none.",
        width=990,
        size=19,
        color=MUTED,
    )
    output = HERE / "real-eval-matching.png"
    canvas.save(output, optimize=True)
    return output


def write_candidates_csv(
    gts: dict[str, tuple[int, int, int, int]],
    boxes: dict[str, tuple[Fraction, Fraction, Fraction, Fraction]],
) -> Path:
    output = HERE / "candidates.csv"
    fields = [
        "id", "image", "gt", "score", "class", "evidence_type",
        "canonical_x0", "canonical_y0", "canonical_x1", "canonical_y1",
        "source_x0_exact", "source_y0_exact", "source_x1_exact", "source_y1_exact",
        "source_x0", "source_y0", "source_x1", "source_y1",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in RAW_ORDER:
            candidate, mapped = CANDIDATES[name], boxes[name]
            image_id = candidate["image"]
            canonical = candidate["box"]
            writer.writerow(
                {
                    "id": name, "image": image_id, "gt": SAMPLES[image_id]["gt_id"],
                    "score": f"{float(candidate['score']):.2f}", "class": "head",
                    "evidence_type": "constructed; not model output",
                    "canonical_x0": canonical[0], "canonical_y0": canonical[1],
                    "canonical_x1": canonical[2], "canonical_y1": canonical[3],
                    "source_x0_exact": exact(mapped[0]), "source_y0_exact": exact(mapped[1]),
                    "source_x1_exact": exact(mapped[2]), "source_y1_exact": exact(mapped[3]),
                    "source_x0": dec(mapped[0], 3), "source_y0": dec(mapped[1], 3),
                    "source_x1": dec(mapped[2], 3), "source_y1": dec(mapped[3], 3),
                }
            )
    return output


def write_iou_csv() -> Path:
    output = HERE / "pairwise-iou.csv"
    fields = ["scope", "image", "a", "b", "intersection", "union", "iou_fraction", "iou_decimal"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image_id in ("I1", "I2"):
            names = [name for name in RAW_ORDER if CANDIDATES[name]["image"] == image_id]
            for index, a_name in enumerate(names):
                for b_name in names[index + 1 :]:
                    a, b = CANDIDATES[a_name]["box"], CANDIDATES[b_name]["box"]
                    inter = intersection_area(a, b)
                    union = area(a) + area(b) - inter
                    overlap = Fraction(inter, union)
                    writer.writerow({"scope": "candidate-candidate", "image": image_id, "a": a_name, "b": b_name, "intersection": inter, "union": union, "iou_fraction": exact(overlap), "iou_decimal": f"{float(overlap):.9f}"})
            gt = SAMPLES[image_id]["canonical_gt"]
            for name in names:
                box = CANDIDATES[name]["box"]
                inter = intersection_area(box, gt)
                union = area(box) + area(gt) - inter
                overlap = Fraction(inter, union)
                writer.writerow({"scope": "candidate-ground-truth", "image": image_id, "a": name, "b": SAMPLES[image_id]["gt_id"], "intersection": inter, "union": union, "iou_fraction": exact(overlap), "iou_decimal": f"{float(overlap):.9f}"})
    return output


def write_nms_csv(trace: list[dict[str, object]]) -> Path:
    output = HERE / "nms-trace.csv"
    fields = ["step", "image", "action", "candidate", "reference", "iou_fraction", "iou_decimal", "reason"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step, row in enumerate(trace, start=1):
            overlap = row["iou"]
            writer.writerow({"step": step, "image": row["image"], "action": row["action"], "candidate": row["candidate"], "reference": row["reference"] or "", "iou_fraction": exact(overlap) if isinstance(overlap, Fraction) else "", "iou_decimal": f"{float(overlap):.9f}" if isinstance(overlap, Fraction) else "", "reason": row["reason"]})
    return output


def write_eval_csv(rows: list[dict[str, object]], ap: Fraction) -> Path:
    output = HERE / "evaluation.csv"
    fields = ["rank", "candidate", "image", "gt", "iou_fraction", "iou_decimal", "result", "cumulative_tp", "cumulative_fp", "precision", "recall", "ap"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"rank": row["rank"], "candidate": row["candidate"], "image": row["image"], "gt": row["gt"], "iou_fraction": exact(row["iou"]), "iou_decimal": f"{float(row['iou']):.9f}", "result": row["result"], "cumulative_tp": row["cumulative_tp"], "cumulative_fp": row["cumulative_fp"], "precision": exact(row["precision"]), "recall": exact(row["recall"]), "ap": exact(ap)})
    return output


def build() -> dict[str, object]:
    photos, gts = load_sources()
    boxes = source_boxes(gts)
    assert boxes["A"] == tuple(Fraction(v) for v in gts["I1"])
    assert boxes["C"] == tuple(Fraction(v) for v in gts["I2"])
    assert iou(CANDIDATES["A"]["box"], CANDIDATES["B"]["box"]) == Fraction(361, 439)
    assert iou(CANDIDATES["C"]["box"], CANDIDATES["D"]["box"]) == Fraction(171, 211)
    assert iou(CANDIDATES["A"]["box"], CANDIDATES["E"]["box"]) == Fraction(9, 71)
    kept, trace = nms()
    assert kept == ["A", "E", "C"]
    rows, ap = evaluate(kept)
    assert [row["result"] for row in rows] == ["TP", "FP", "TP"]
    assert ap == Fraction(5, 6)

    artifacts = [
        build_targets_plate(photos, gts),
        build_candidates_plate(photos, gts, boxes),
        build_nms_plate(photos, gts, boxes),
        build_eval_plate(photos, gts, boxes, rows, ap),
        write_candidates_csv(gts, boxes),
        write_iou_csv(),
        write_nms_csv(trace),
        write_eval_csv(rows, ap),
    ]
    metadata: dict[str, object] = {
        "schema_version": 2,
        "evidence_taxonomy": {
            "observed": ["two original Oxford-IIIT Pet photographs", "two official tight-head XML ROIs"],
            "constructed": ["two display crops", "candidates A-E", "scores", "affine placement"],
            "computed": ["IoU", "per-image NMS", "one-to-one matching", "AP", "overlays"],
            "model_output": [],
            "warning": "A-E and their scores are pedagogical constructions; no detector was run.",
        },
        "source": {
            "dataset": "Oxford-IIIT Pet",
            "dataset_page": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
            "paper": "Parkhi et al., Cats and Dogs, CVPR 2012",
            "license": "CC BY-SA 4.0; copyright remains with the original image owners",
            "samples": {
                image_id: {
                    "stem": sample["stem"], "breed": sample["breed"],
                    "image_path": str(Path(sample["image"]).relative_to(HERE) if Path(sample["image"]).is_relative_to(HERE) else Path("..") / Path(sample["image"]).name),
                    "image_sha256": sha256(sample["image"]), "image_size": list(photos[image_id].size),
                    "xml_path": str(Path(sample["xml"]).relative_to(HERE) if Path(sample["xml"]).is_relative_to(HERE) else Path("..") / Path(sample["xml"]).name),
                    "xml_sha256": sha256(sample["xml"]), "gt_id": sample["gt_id"],
                    "xml_voc_1_based_inclusive": [gts[image_id][0] + 1, gts[image_id][1] + 1, gts[image_id][2], gts[image_id][3]],
                    "gt_source_xyxy_half_open": list(gts[image_id]), "gt_canonical_xyxy": list(sample["canonical_gt"]),
                }
                for image_id, sample in SAMPLES.items()
            },
        },
        "coordinate_contract": {
            "format": "xyxy", "boundary": "continuous / 0-based half-open", "area": "(x1-x0)*(y1-y0); no +1",
            "maps": {
                "I1": {"x": "332 + (x-10)*93/40", "y": "71 + (y-10)*87/40"},
                "I2": {"x": "314 + (x-60)*132/30", "y": "59 + (y-15)*160/40"},
            },
            "invariance": "independent positive x/y scaling preserves IoU",
            "rendering": "PNG boundaries are rounded for display only; decisions use exact canonical rational geometry.",
        },
        "candidates": {
            name: {"image": candidate["image"], "score": dec(candidate["score"], 2), "class": "head", "canonical_box": list(candidate["box"]), "source_box_exact": [exact(v) for v in boxes[name]], "source_box": [dec(v, 3) for v in boxes[name]], "evidence_type": "constructed; not model output"}
            for name, candidate in CANDIDATES.items()
        },
        "nms": {"score_threshold": 0.70, "iou_threshold": 0.50, "scope": "within image and class", "raw_global_order": list(RAW_ORDER), "kept_global_order": kept, "suppressed": {"B": {"by": "A", "iou": dec(Fraction(361, 439))}, "D": {"by": "C", "iou": dec(Fraction(171, 211))}}},
        "evaluation": {"iou_threshold": 0.50, "ground_truth_count": 2, "ranked_results": [{"rank": row["rank"], "candidate": row["candidate"], "image": row["image"], "gt": row["gt"], "iou_fraction": exact(row["iou"]), "iou": dec(row["iou"]), "result": row["result"], "precision": exact(row["precision"]), "recall": exact(row["recall"])} for row in rows], "counts": {"tp": 2, "fp": 1, "fn": 0}, "ap_fraction": exact(ap), "ap": dec(ap), "ap_definition": "mean precision at the two TP ranks; equals the all-point precision-envelope area for this sequence", "teaching_scope": "fixed two-image mini-batch; not a benchmark"},
        "build": {"command": "uv run --with pillow==12.2.0 python shared/vision-evidence/oxford-iiit-pet/l9/build_detection_evidence.py", "python": platform.python_version(), "pillow": PIL.__version__, "font_regular": font_identity(), "font_bold": font_identity(bold=True), "builder_sha256": sha256(Path(__file__)), "artifacts": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in artifacts}},
    }
    (HERE / "evidence.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    result = build()
    print(json.dumps({"kept": result["nms"]["kept_global_order"], "counts": result["evaluation"]["counts"], "ap": result["evaluation"]["ap_fraction"], "output": str(HERE)}, indent=2))
