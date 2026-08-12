#!/usr/bin/env python3
"""Build reproducible teaching evidence from one Oxford-IIIT Pet sample.

The source photograph, trimap, and XML annotation are ground truth. Every
output written by this script is either a direct annotation overlay/crop or a
deterministic computation over the photograph; none is a model prediction.
"""

from __future__ import annotations

import json
import hashlib
import platform
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "Abyssinian_1.jpg"
TRIMAP_PATH = ROOT / "Abyssinian_1-trimap.png"
XML_PATH = ROOT / "Abyssinian_1.xml"
OUT = ROOT / "derived"

TEAL = (44, 122, 123)
ORANGE = (235, 129, 27)
BLUE = (43, 108, 176)
GREEN = (20, 176, 61)
RED = (214, 69, 80)
INK = (35, 55, 59)
WHITE = (255, 255, 255)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def font_identity(bold: bool = False) -> str:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return "Pillow.load_default()"


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, color=INK) -> None:
    fnt = font(20, bold=True)
    box = draw.textbbox(xy, text, font=fnt, stroke_width=0)
    pad = 5
    draw.rounded_rectangle(
        (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
        radius=4,
        fill=WHITE,
        outline=color,
        width=2,
    )
    draw.text(xy, text, font=fnt, fill=color)


def load_box() -> tuple[int, int, int, int]:
    """Return the VOC 1-based inclusive XML box as 0-based half-open xyxy."""
    node = ElementTree.parse(XML_PATH).getroot().find("object/bndbox")
    assert node is not None
    xmin, ymin, xmax, ymax = tuple(int(node.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax"))
    return xmin - 1, ymin - 1, xmax, ymax


def cross_correlate(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    windows = np.lib.stride_tricks.sliding_window_view(gray, (kh, kw))
    return np.einsum("ijuv,uv->ij", windows, kernel, optimize=True)


def max_pool_2x2_stride2(values: np.ndarray) -> np.ndarray:
    height = values.shape[0] // 2 * 2
    width = values.shape[1] // 2 * 2
    windows = np.lib.stride_tricks.sliding_window_view(values[:height, :width], (2, 2))[::2, ::2]
    return windows.max(axis=(-2, -1))


def normalize_map(values: np.ndarray, *, signed: bool, scale: float | None = None) -> Image.Image:
    values = np.asarray(values, dtype=np.float64)
    if signed:
        scale = float(np.percentile(np.abs(values), 99)) if scale is None else scale
        z = np.clip(values / max(scale, 1e-9), -1, 1)
        rgb = np.zeros((*z.shape, 3), dtype=np.float64)
        positive = z >= 0
        rgb[positive] = (1 - z[positive, None]) * np.array([1.0, 1.0, 1.0]) + z[positive, None] * np.array(ORANGE) / 255
        magnitude = -z[~positive]
        rgb[~positive] = (1 - magnitude[:, None]) * np.array([1.0, 1.0, 1.0]) + magnitude[:, None] * np.array(BLUE) / 255
    else:
        lo, hi = np.percentile(values, [1, 99])
        z = np.clip((values - lo) / max(hi - lo, 1e-9), 0, 1)
        rgb = (1 - z[..., None]) * np.array([1.0, 1.0, 1.0]) + z[..., None] * np.array(GREEN) / 255
    return Image.fromarray(np.uint8(np.round(rgb * 255)), mode="RGB")


def panel_with_title(image: Image.Image, title: str, *, width: int = 480, height: int = 320) -> Image.Image:
    panel = Image.new("RGB", (width, height + 48), WHITE)
    fitted = image.copy()
    fitted.thumbnail((width, height), Image.Resampling.LANCZOS)
    x = (width - fitted.width) // 2
    y = 48 + (height - fitted.height) // 2
    panel.paste(fitted, (x, y))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 10), title, font=font(22, bold=True), fill=INK)
    return panel


def build() -> dict[str, object]:
    OUT.mkdir(exist_ok=True)
    photo = Image.open(IMAGE_PATH).convert("RGB")
    trimap = np.asarray(Image.open(TRIMAP_PATH), dtype=np.uint8)
    image = np.asarray(photo, dtype=np.float64)
    box = load_box()

    # Ground-truth annotation overlay and crop.
    overlay = photo.copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline=TEAL, width=6)
    label(draw, (box[0] + 7, max(8, box[1] - 32)), "ground-truth head ROI", color=TEAL)
    overlay.save(OUT / "photo-head-roi.png", optimize=True)
    photo.crop(box).resize((368, 344), Image.Resampling.LANCZOS).save(OUT / "head-roi-crop.png", optimize=True)

    # Ground-truth trimap overlay: 1 foreground, 2 background, 3 boundary.
    trimap_rgb = np.zeros((*trimap.shape, 4), dtype=np.uint8)
    trimap_rgb[trimap == 1] = (*GREEN, 118)
    trimap_rgb[trimap == 3] = (*ORANGE, 210)
    trimap_layer = Image.fromarray(trimap_rgb, mode="RGBA")
    trimap_overlay = Image.alpha_composite(photo.convert("RGBA"), trimap_layer)
    trimap_overlay = trimap_overlay.convert("RGB")
    trimap_draw = ImageDraw.Draw(trimap_overlay)
    trimap_draw.rounded_rectangle((10, 10, 330, 92), radius=6, fill=WHITE, outline=INK, width=2)
    trimap_draw.rectangle((24, 24, 56, 50), fill=GREEN, outline=INK, width=2)
    trimap_draw.text((68, 24), "foreground", font=font(18, bold=True), fill=INK)
    trimap_draw.rectangle((24, 60, 56, 84), fill=ORANGE, outline=INK, width=2)
    trimap_draw.line((24, 84, 56, 60), fill=INK, width=3)
    trimap_draw.text((68, 58), "boundary / ignore", font=font(18, bold=True), fill=INK)
    trimap_overlay.save(OUT / "photo-trimap-overlay.png", optimize=True)

    # A declared 45x45 real crop produces the same five horizontal binary bands
    # as the deck's exact X after row-wise averaging and thresholding.
    crop_box = (328, 120, 373, 165)
    crop = image[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
    gray_crop = 0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]
    row_bands = np.array([part.mean() for part in np.array_split(gray_crop, 5, axis=0)])
    threshold = 128.0
    binary_rows = (row_bands >= threshold).astype(int)
    binary_grid = np.repeat(binary_rows[:, None], 5, axis=1)
    assert binary_grid.tolist() == [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    crop_img = Image.fromarray(np.uint8(crop), mode="RGB").resize((360, 360), Image.Resampling.NEAREST)
    crop_draw = ImageDraw.Draw(crop_img)
    for i in range(1, 5):
        crop_draw.line((0, i * 72, 360, i * 72), fill=WHITE, width=2)
    crop_img.save(OUT / "real-crop-five-bands.png", optimize=True)

    # An actual pixel from the same head ROI supplies the RGB channel example.
    pixel_xy = (373, 137)
    pixel_rgb = np.asarray(photo)[pixel_xy[1], pixel_xy[0]].astype(int)
    pixel_norm = pixel_rgb / 255.0
    kernel_rgb = np.array([0.5, -1.0, 0.4])
    pixel_response = float(pixel_norm @ kernel_rgb)
    pixel_patch = photo.crop((pixel_xy[0] - 5, pixel_xy[1] - 5, pixel_xy[0] + 6, pixel_xy[1] + 6))
    pixel_patch = pixel_patch.resize((330, 330), Image.Resampling.NEAREST)
    pixel_draw = ImageDraw.Draw(pixel_patch)
    pixel_draw.rectangle((150, 150, 179, 179), outline=GREEN, width=5)
    label(pixel_draw, (8, 8), "center pixel", color=GREEN)
    pixel_patch.save(OUT / "actual-rgb-pixel.png", optimize=True)

    # Computed Sobel response, a shifted response, and max-pooled magnitudes.
    work = photo.resize((300, 200), Image.Resampling.LANCZOS)
    work_array = np.asarray(work, dtype=np.float64)
    gray = (0.299 * work_array[..., 0] + 0.587 * work_array[..., 1] + 0.114 * work_array[..., 2]) / 255.0
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    response = cross_correlate(gray, sobel_x)
    shifted = np.zeros_like(gray)
    shifted[:, 12:] = gray[:, :-12]
    shifted_response = cross_correlate(shifted, sobel_x)
    pooled = max_pool_2x2_stride2(np.abs(response))
    signed_scale = float(np.percentile(np.abs(np.concatenate((response.ravel(), shifted_response.ravel()))), 99))
    interior_a = shifted_response[:, 12:]
    interior_b = response[:, :-12]
    interior_max_abs_error = float(np.max(np.abs(interior_a - interior_b)))
    pooled_lo, pooled_hi = (float(v) for v in np.percentile(pooled, [1, 99]))

    panels = [
        panel_with_title(work, "REAL IMAGE (300 x 200)"),
        panel_with_title(normalize_map(response, signed=True, scale=signed_scale).resize((480, 320), Image.Resampling.NEAREST), "COMPUTED SOBEL RESPONSE"),
        panel_with_title(normalize_map(shifted_response, signed=True, scale=signed_scale).resize((480, 320), Image.Resampling.NEAREST), "SOBEL RESPONSE TO +12 PX INPUT SHIFT"),
        panel_with_title(normalize_map(pooled, signed=False).resize((480, 320), Image.Resampling.NEAREST), "ORIGINAL |RESPONSE|: 2 x 2 MAX POOL"),
    ]
    composite = Image.new("RGB", (960, 736), (243, 246, 245))
    for idx, panel in enumerate(panels):
        composite.paste(panel, ((idx % 2) * 480, (idx // 2) * 368))
    composite.save(OUT / "computed-conv-shift-pool.png", optimize=True)

    # RF geometry overlay: these are theoretical support boxes, not saliency.
    rf = photo.copy()
    rf_draw = ImageDraw.Draw(rf)
    center = (381, 115)
    scales = [(48, BLUE, "r = 3"), (64, ORANGE, "r = 4"), (128, TEAL, "r = 8")]
    for side, color, text in scales:
        xy = (center[0] - side // 2, center[1] - side // 2, center[0] + side // 2, center[1] + side // 2)
        rf_draw.rectangle(xy, outline=color, width=4)
    for index, (_, color, text) in enumerate(scales):
        label(rf_draw, (445, 62 + 38 * index), text, color=color)
    label(rf_draw, (15, 15), "theoretical input support; not saliency", color=INK)
    rf.save(OUT / "photo-rf-overlay.png", optimize=True)

    generated_pngs = sorted(OUT.glob("*.png"))
    metadata = {
        "build": {
            "command": "uv run --with numpy==2.3.5 --with pillow==12.2.0 python shared/vision-evidence/oxford-iiit-pet/build_evidence.py",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "font_regular": font_identity(False),
            "font_bold": font_identity(True),
            "builder_sha256": sha256(Path(__file__)),
            "derived_sha256": {path.name: sha256(path) for path in generated_pngs},
        },
        "source": {
            "image": IMAGE_PATH.name,
            "trimap": TRIMAP_PATH.name,
            "xml": XML_PATH.name,
            "license": "CC BY-SA 4.0; copyright remains with the original image owner",
        },
        "ground_truth": {
            "head_roi_xml_voc_1_based_inclusive": [333, 72, 425, 158],
            "head_roi_xyxy_0_based_half_open": box,
            "trimap_values": {"1": "foreground", "2": "background", "3": "boundary"},
            "trimap_counts": {str(v): int((trimap == v).sum()) for v in (1, 2, 3)},
        },
        "lecture8": {
            "crop_xyxy": crop_box,
            "crop_definition": "45x45 real RGB crop; grayscale bands are means of five consecutive 9-row strips",
            "grayscale_band_means_0_255": [round(float(v), 6) for v in row_bands],
            "quantization_threshold_0_255": threshold,
            "quantized_binary_rows": binary_rows.tolist(),
            "pixel_xy": pixel_xy,
            "pixel_rgb_0_255": pixel_rgb.tolist(),
            "pixel_rgb_0_1": [round(float(v), 6) for v in pixel_norm],
            "rgb_kernel": kernel_rgb.tolist(),
            "rgb_response": round(pixel_response, 6),
            "computed_map": {
                "resize": [300, 200],
                "grayscale": "0.299 R + 0.587 G + 0.114 B, divided by 255",
                "kernel": sobel_x.tolist(),
                "operator": "valid cross-correlation",
                "signed_display_scale": {
                    "definition": "shared p99 absolute magnitude across original and shifted response; values clipped to [-1,1] for display",
                    "p99_absolute": signed_scale,
                    "clipped_original_samples": int((np.abs(response) > signed_scale).sum()),
                    "clipped_shifted_samples": int((np.abs(shifted_response) > signed_scale).sum()),
                    "samples_per_response": int(response.size),
                },
                "shift": "zero-filled +12 pixels horizontally",
                "common_interior_check": {
                    "comparison": "shifted_response[:, 12:] versus response[:, :-12]",
                    "shape": list(interior_a.shape),
                    "max_abs_error_floating_point": interior_max_abs_error,
                    "expected_exact_linear_operator_equality": True,
                    "allclose_atol_1e_12": bool(np.allclose(interior_a, interior_b, rtol=0, atol=1e-12)),
                },
                "pool": "2x2 max pool, stride 2, applied to absolute response",
                "pooled_display_scale": {
                    "definition": "linear p1-p99 range, clipped to [0,1] for display",
                    "p1": pooled_lo,
                    "p99": pooled_hi,
                    "clipped_low_samples": int((pooled < pooled_lo).sum()),
                    "clipped_high_samples": int((pooled > pooled_hi).sum()),
                    "samples": int(pooled.size),
                },
                "response_shape": list(response.shape),
                "pooled_shape": list(pooled.shape),
            },
            "rf_overlay": "theoretical support widths 3, 4, and 8, drawn with one 16-display-pixels-per-input-pixel scale as 48, 64, and 128 pixels; not saliency",
        },
    }
    (OUT / "evidence.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    result = build()
    print(json.dumps(result["lecture8"], indent=2))
