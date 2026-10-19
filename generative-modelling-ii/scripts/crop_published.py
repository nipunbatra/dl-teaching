"""Crop the published figures used on the slides from the original files in assets/.

    python scripts/crop_published.py      # writes figures/data/pub-*.png

The originals were downloaded by scripts/acquire_assets.py from each project's own page
(URLs in asset-manifest.json). Every crop below is one panel of a figure, kept at native
resolution; nothing is retouched. Boxes are (left, top, right, bottom) in original pixels,
then near-white / near-grey margins are trimmed.
"""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "assets"
OUT = ROOT / "figures" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def trim(im, tol=18):
    a = np.asarray(im.convert("RGB")).astype(int)
    bg = a[0, 0]
    diff = np.abs(a - bg).sum(2) > tol
    ys, xs = np.where(diff)
    if len(xs) == 0:
        return im
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def crop(src, box, name, do_trim=True):
    im = Image.open(A / src).convert("RGB").crop(box)
    if do_trim:
        im = trim(im)
    im.save(OUT / f"pub-{name}.png")
    return im.size


def tiles_by_gaps(src, box, axis, n, name, thresh=245, do_trim=True):
    """Split a strip into n tiles at the white gaps along an axis (0 = rows, 1 = columns)."""
    im = Image.open(A / src).convert("RGB").crop(box)
    a = np.asarray(im).astype(int).mean(2)
    prof = a.mean(axis=1 - axis) if axis == 1 else a.mean(axis=1)
    content = prof < thresh
    runs, start = [], None
    for i, c in enumerate(content):
        if c and start is None:
            start = i
        if not c and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(content)))
    runs = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:n]
    runs.sort()
    out = []
    for k, (lo, hi) in enumerate(runs):
        t = im.crop((lo, 0, hi, im.height) if axis == 1 else (0, lo, im.width, hi))
        if do_trim:
            t = trim(t)
        t.save(OUT / f"pub-{name}-{k + 1}.png")
        out.append(t.size)
    return out


SIZES = {}
# DDPM: four unconditional CelebA-HQ samples (top-left 2 x 2 block of the project mosaic)
SIZES["ddpm-faces"] = crop("ddpm.png", (0, 0, 511, 512), "ddpm-faces", do_trim=False)
# LDM Figure 17: the "A basket full of cherries" column, four samples
for k, (y0, y1) in enumerate(((104, 228), (228, 352), (352, 476), (476, 599))):   # the four samples touch
    SIZES[f"cherries-{k + 1}"] = crop("ldm.png", (1, y0, 124, y1), f"cherries-{k + 1}", do_trim=False)
# RePaint: the face row; input with the blue missing region, then five sampled fillings
SIZES["repaint-in"] = crop("repaint.png", (0, 0, 180, 182), "repaint-in")
SIZES["repaint-out"] = tiles_by_gaps("repaint.png", (746, 0, 1655, 182), 1, 5, "repaint")
# SR3: Bicubic / Regression / SR3 / Reference, full face and the eye zoom
SIZES["sr3"] = tiles_by_gaps("sr3.png", (0, 56, 1460, 400), 1, 4, "sr3-face")
SIZES["sr3-eye"] = tiles_by_gaps("sr3.png", (0, 417, 1460, 760), 1, 4, "sr3-eye")
# SDEdit: CelebA stroke painting and two outputs
SIZES["sdedit"] = crop("sdedit.jpg", (818, 3, 918, 103), "sdedit-in", do_trim=False)
SIZES["sdedit-out1"] = crop("sdedit.jpg", (818, 112, 918, 212), "sdedit-out1", do_trim=False)
SIZES["sdedit-out2"] = crop("sdedit.jpg", (818, 219, 918, 319), "sdedit-out2", do_trim=False)
# ControlNet: input photo, its Canny edge map and five generated birds (inner portrait regions)
SIZES["control-photo"] = crop("control.png", (302, 60, 458, 294), "control-photo", do_trim=False)
cols = [(848, 1070), (1210, 1436)]
rows = [(66, 418), (432, 784), (798, 1152)]
k = 0
for (y0, y1) in rows:
    for (x0, x1) in cols:
        k += 1
        crop("control.png", (x0, y0, x1, y1), f"control-{k}", do_trim=False)
# GLIGEN: layout boxes and three of the five samples for "Space view of a planet and its sun"
SIZES["gligen-layout"] = crop("gligen.png", (36, 10, 478, 455), "gligen-layout", do_trim=False)
SIZES["gligen"] = tiles_by_gaps("gligen.png", (505, 10, 2880, 455), 1, 5, "gligen")
# DreamBooth: input subject images, and two generated contexts with their prompts
SIZES["db-in"] = crop("dreambooth.jpg", (0, 0, 420, 400), "dreambooth-in")
SIZES["db-acropolis"] = crop("dreambooth.jpg", (476, 0, 872, 462), "dreambooth-acropolis")
SIZES["db-haircut"] = crop("dreambooth.jpg", (1254, 0, 1650, 462), "dreambooth-haircut")
# DiffusionSat super-resolution, left panel, fifth row: Sentinel-2 RGB, HR reference, DiffusionSat
for name, (x0, x1) in (("rgb", (539, 808)), ("hr", (809, 1078)), ("ours", (1106, 1376)), ("sd", (1383, 1653)),
                       ("dbpn", (1660, 1930))):
    SIZES[f"sat-{name}"] = crop("sat-sr.png", (x0, 1211, x1, 1481), f"sat-{name}", do_trim=False)

if __name__ == "__main__":
    for k_, v in SIZES.items():
        print(k_, v)

# the SR3 reference column carries the figure's right border; match the other columns' width
for nm in ("sr3-face-4", "sr3-eye-4"):
    im = Image.open(OUT / f"pub-{nm}.png")
    im.crop((0, 0, 347, im.height)).save(OUT / f"pub-{nm}.png")
