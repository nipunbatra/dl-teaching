# Lecture 10 real two-image detection evidence

This bundle grounds the public detection lecture's exact A–E box ledger in a
two-image Oxford-IIIT Pet mini-batch. Each real photograph has its own official
XML tight-head ROI, so the existing two-truth NMS, matching, and AP spine
remains exact without inventing a second object in one photo.

## Four explicit contracts

1. **Separate official truth per photograph.** Image `I1` is
   `Abyssinian_1.jpg` with XML truth `G1`; image `I2` is `Bengal_10.jpg` with
   XML truth `G2`.
2. **Separate positive affine maps.** Constructed candidates A/B/E and their
   declared scores belong to I1; constructed C/D and their scores belong to
   I2. Each image has its own positive x/y affine map from canonical lecture
   coordinates to source pixels. A equals G1 and C equals G2 exactly.
3. **NMS is grouped by `(image_id, class)`.** There is no cross-image
   suppression. At IoU threshold `0.50`, A suppresses B in I1 and C suppresses
   D in I2; isolated E survives. The globally score-sorted returned batch is
   A, E, C.
4. **Evaluation matches within `image_id`, then ranks globally.** At evaluation
   IoU threshold `0.50`, A=TP, E=FP, C=TP. Precision at the two TP ranks is
   `1` and `2/3`, hence `AP=(1+2/3)/2=5/6≈0.833` with two TPs, one FP, and no
   FNs.

This is a fixed two-image teaching case, not a benchmark.

## Evidence taxonomy

- **OBSERVED:** original Oxford pixels and each official XML tight-head ROI.
- **CONSTRUCTED:** display crops, A–E boxes, scores, shared `head` class, and
  affine mappings. These are pedagogical constructions, not predictions.
- **COMPUTED:** IoU, per-image NMS, one-to-one matching, AP, CSV ledgers, and
  overlays.
- **MODEL OUTPUT:** none. No detector was run.

## Source and coordinates

Both samples are from Oxford-IIIT Pet (Parkhi et al., *Cats and Dogs*, CVPR
2012), licensed CC BY-SA 4.0 with copyright retained by the original image
owners. `Bengal_10` is in the official `trainval.txt` split (`Bengal_10 6 1
2`). Its JPEG was retrieved from the same `timm/oxford-iiit-pet` mirror used by
the shared evidence case; its XML came from Oxford's official
`annotations.tar.gz` archive.

All boxes use continuous / 0-based half-open `xyxy`; area is
`(x1-x0)(y1-y0)` with no `+1`.

| image | official VOC XML | course source box | canonical truth | affine map |
|---|---|---|---|---|
| I1 · Abyssinian_1 | `(333,72)–(425,158)` | `[332,71,425,158)` | `G1=[10,10,50,50]` | `x'=332+(x-10)93/40`; `y'=71+(y-10)87/40` |
| I2 · Bengal_10 | `(315,60)–(446,219)` | `[314,59,446,219)` | `G2=[60,15,90,55]` | `x'=314+(x-60)132/30`; `y'=59+(y-15)160/40` |

Independent positive scaling of x and y preserves IoU, so the public
lecture's arithmetic is unchanged:

- `IoU(A,B)=1444/1756=361/439≈0.822323`
- `IoU(C,D)=1026/1266=171/211≈0.810427`
- `IoU(A,E)=200/1575=9/71≈0.126761`

## Visual assets

All plates are `1600×900` (16:9) combined images for direct slide placement.

- `real-head-targets.png`: reveals both official XML truths on the original
  photographs.
- `real-candidates.png`: reveals XML truth with constructed A–E candidates,
  scores, and two real crops.
- `real-nms-trace.png`: deliberately **hides all truth**; shows raw/surviving
  candidates within each image and the two suppressions.
- `real-eval-matching.png`: reveals truth again for ranked A/E/C matching and
  AP.

Machine-readable outputs:

- `candidates.csv`: image identity, canonical boxes, and exact rational plus
  decimal source coordinates.
- `pairwise-iou.csv`: within-image candidate IoUs and candidate-to-GT IoUs.
- `nms-trace.csv`: every per-image keep/suppress decision.
- `evaluation.csv`: global survivor rank, TP/FP, cumulative precision/recall,
  and AP.
- `evidence.json`: full taxonomy, provenance, coordinate contract, results,
  build environment, and artifact hashes.

## Deterministic rebuild

From the repository root:

```bash
uv run --with pillow==12.2.0 python \
  shared/vision-evidence/oxford-iiit-pet/l9/build_detection_evidence.py
```

The builder asserts the two source JPEG/XML hashes, exact IoUs, grouped-NMS
survivors, matching sequence, and AP before writing outputs.

Source hashes:

- `Abyssinian_1.jpg`:
  `2533197401eebe9410ea4d063f86c43fbd2666f3e8165a38aca155c0d09c21be`
- `Abyssinian_1.xml`:
  `bea13ddcf1dc171e39da5ce2c134b1dc8c247a9f1bb21cc2056f37f7a423a51f`
- `Bengal_10.jpg`:
  `4cc958fc6bb47738488fdc83a3053f01ce2937c9fbcaf5db2e649a1d612f8725`
- `Bengal_10.xml`:
  `9a6e7d513d999b23cde839a6efce908e2e1472d14d69f13d671e9db9cdb09ae6`
