# Lecture 11 real segmentation evidence

This bundle grounds the public semantic/instance-segmentation lecture in one
licensed photograph and its official pixel annotation. It does **not** claim
to show a trained model. The deck's exact `4 x 4` probability/mask ledger
remains a separate pedagogical construction.

## Evidence contract

- **OBSERVED:** the original `Abyssinian_1.jpg` pixels.
- **GROUND TRUTH (GT):** the official Oxford-IIIT Pet trimap.
- **CONSTRUCTED:** one declared probability field made by shifting the GT
  foreground `16 px` right with zero fill.
- **COMPUTED:** the thresholded mask, overlays, confusion counts, BCE,
  Soft Dice, IoU, hard Dice, and boundary diagnostics.
- **MODEL OUTPUT:** none. No checkpoint is loaded or executed.
- **MEASURED PERFORMANCE:** none. The numbers diagnose a deterministic
  teaching construction; they are not a benchmark result.

## Source, license, and annotation

- Dataset: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- Sample: `Abyssinian_1`, image size `600 x 400`
- License: CC BY-SA 4.0; copyright remains with the original image owner.
- Source photograph: `../Abyssinian_1.jpg`
- Official trimap: `../Abyssinian_1-trimap.png`
- Trimap value convention:
  - `1`: foreground
  - `2`: background
  - `3`: boundary / ignore

The photograph was retrieved from the Hugging Face mirror
`timm/oxford-iiit-pet` on 2026-08-12. The trimap came from Oxford's official
`annotations.tar.gz` archive. Their hashes are asserted by the builder and
recorded in `evidence.json`.

## Mask and metric policy

Values `1` and `2` are valid region pixels. Value `3` is excluded from BCE,
Soft Dice, TP/FP/FN/TN, accuracy, IoU, and hard Dice. This means ignored pixels
are neither background nor free true negatives; they are removed from every
region numerator and denominator.

For the separate boundary diagnostic, the official value-`3` band is the GT
reference contour. The predicted contour is the one-pixel inner boundary of
the hard mask. Boundary precision/recall/F1 use exact Euclidean pixel-center
distance with a declared `5 px` tolerance. ASSD is the mean of the two directed
mean distances; HD95 is the maximum directed 95th-percentile distance. These
are diagnostics, not part of the lecture's exact `4 x 4` arithmetic spine.

## Deterministic constructed probability

Starting from `G = 1[trimap == 1]`, the builder shifts `G` `16 px` right with
zero fill, assigns probability `.90` to the shifted foreground and `.10` to
the remaining pixels, and computes `P = 1[p >= 0.50]`.

Every parameter is repeated in `evidence.json`. This construction intentionally
creates a spatially controlled boundary displacement. It is a failure case,
not an inference result.

## Files

- `real-trimap-contract.png`: observed photo, official trimap values, and GT
  overlay.
- `constructed-probability-mask.png`: GT -> constructed probability -> hard
  mask across the three clocks.
- `real-error-overlay.png`: TP/FP/FN/ignore overlay on actual pixels, with exact
  region counts and IoU/Dice.
- `region-boundary-audit.png`: region losses/metrics beside boundary
  diagnostics.
- `metrics.csv`: one-row-per-metric machine-readable audit.
- `real-mask-arrays.npz`: compact `trimap`, `valid`, `gt`, constructed
  `probability`, and hard `prediction` arrays for exact notebook recomputation.
- `evidence.json`: full source, taxonomy, mask policy, construction, metric,
  environment, and SHA-256 manifest.
- `build_segmentation_evidence.py`: reproducible builder.

## Rebuild and verify

Run from the repository root:

```bash
uv run --with numpy==2.4.6 --with pillow==12.3.0 python \
  shared/vision-evidence/oxford-iiit-pet/l10/build_segmentation_evidence.py
```

The builder checks the two source hashes before doing any work and writes the
builder/output hashes into `evidence.json`. Text uses an available system font
with a Pillow fallback; numerical results and mask pixels do not depend on the
font.

## Safe lecture claims

- The image and trimap are real, licensed evidence.
- Region metrics exclude the official ignore band consistently.
- A plausible-looking hard mask can still show spatially structured boundary
  and false-positive errors.
- IoU/Dice and boundary diagnostics reduce different aspects of the same
  declared hard-mask error.
- A one-pet semantic mask does not demonstrate instance separation by itself;
  the generated two-tree scene remains an explicitly generated teaching case.
