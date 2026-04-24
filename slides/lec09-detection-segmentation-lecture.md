---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Detection &amp; Segmentation

## Lecture 9 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. Extend a classifier to **localization + detection** with multiple heads.
2. Compute **IoU** and explain **NMS**.
3. Contrast **R-CNN family vs YOLO vs DETR**.
4. Pick **anchor boxes** and explain delta parameterization.
5. Use a **U-Net** for per-pixel segmentation; choose appropriate loss.
6. Prompt **SAM** for zero-shot segmentation.

---

# Recap · where we are

Module 4 so far:
- **L7** · CNN mechanics, receptive field, classic architectures
- **L8** · Modern CNNs (ResNet bottleneck, MobileNet, EfficientNet) + transfer learning

All of that was **classification** — one label per image.

<div class="paper">

Today is different: UDL does NOT cover detection / segmentation. Read Bishop Ch 10 or CS231n OD notes alongside these slides.

</div>

Today's jump: **one label per pixel, per object, per region.**

---

# Four questions

1. How do we go from classification to **bounding boxes**?
2. How do we compare boxes — what is **IoU**? How do we deduplicate predictions — **NMS**?
3. How does **YOLO** do all of it in one forward pass?
4. How do we predict a **mask per pixel** — **U-Net** and beyond?

---

<!-- _class: section-divider -->

### PART 1

# Classification → Localization → Detection

Three increasing levels of spatial specificity

---

# The spectrum of "what's in this image"

| Task | Output | Example |
|------|--------|---------|
| **Classification** | 1 label | "cat" |
| **Classification + localization** | 1 label + 1 bbox | "cat @ (60, 80, 200, 180)" |
| **Object detection** | many labels + many bboxes | "cat @ ... dog @ ... car @ ..." |
| **Semantic segmentation** | label per pixel | every pixel → class (no instance) |
| **Instance segmentation** | label + instance per pixel | "cat 1 vs cat 2 as separate masks" |

Today we cover the last four.

---

# Classification + localization · the simplest jump

Keep the CNN. Add **two heads** on top:

```python
class ClsLocHead(nn.Module):
    def __init__(self, feat_dim, n_classes):
        super().__init__()
        self.cls = nn.Linear(feat_dim, n_classes)   # class scores
        self.box = nn.Linear(feat_dim, 4)           # (x, y, w, h)

    def forward(self, feat):
        return self.cls(feat), self.box(feat)
```

Multi-task loss:

$$\mathcal{L} = \mathcal{L}_\text{class}(\text{logits}, y) + \lambda \cdot \mathcal{L}_\text{box}(\hat{\mathbf{b}}, \mathbf{b})$$

Typical $\mathcal{L}_\text{box}$: **Smooth L1** — L2 near 0 (smooth), L1 far (robust).

---

<!-- _class: section-divider -->

### PART 2

# IoU and NMS

Two primitives every detector needs

---

# IoU · the metric · NMS · the cleanup

![w:920px](figures/lec09/svg/iou_nms.svg)

<div class="realworld">

▶ Interactive: draw boxes on a canvas, see IoU + NMS live — [object-detection](https://nipunbatra.github.io/interactive-articles/object-detection/).

</div>

---

# IoU · visual examples

![w:920px](figures/lec09/svg/iou_visual_examples.svg)

---

# IoU · a quick worked example

Predicted box · $(x_1, y_1, x_2, y_2) = (50, 50, 150, 150)$ · area = 100 × 100 = 10,000.
Ground-truth box · $(100, 100, 200, 200)$ · area = 10,000.

Intersection · $(100, 100, 150, 150)$ · area = 50 × 50 = 2,500.
Union · 10,000 + 10,000 − 2,500 = 17,500.

$$\text{IoU} = 2{,}500 / 17{,}500 \approx 0.14$$

<div class="insight">

An IoU below 0.5 is almost always considered a miss. Detectors report mAP at multiple IoU thresholds because a box that's 90% right (IoU 0.9) is much more useful than one that's 60% right (IoU 0.6) — the metric penalizes imprecision.

</div>

---

# NMS · by pseudocode

```python
def nms(boxes, scores, iou_threshold=0.5):
    idx = scores.argsort(descending=True)
    keep = []
    while len(idx):
        i = idx[0]
        keep.append(i)
        # drop any later box that overlaps >threshold with this one
        idx = [j for j in idx[1:] if iou(boxes[i], boxes[j]) < iou_threshold]
    return keep
```

<div class="keypoint">

**Greedy** — the highest-confidence box wins in its neighborhood. Lower-confidence duplicates get suppressed. A fundamental primitive · every detector uses some form of NMS (Faster R-CNN, YOLO) or its learned replacement (DETR's Hungarian matching).

</div>

---

# mAP · mean Average Precision

The headline metric for detection:

1. For each class, sort predictions by confidence.
2. Sweep the confidence threshold from high to low.
3. For each threshold, compute precision and recall.
4. **Average Precision** = area under the precision-recall curve.
5. **mAP** = average over classes.

<div class="math-box">

**mAP@0.5** — IoU threshold of 0.5 for a hit.
**mAP@[0.5:0.95]** — average mAP at 10 IoU thresholds from 0.5 to 0.95. The COCO standard.

</div>

Higher is better. Papers usually report both.

---

<!-- _class: section-divider -->

### PART 3

# One-stage vs two-stage detectors

R-CNN family · YOLO · DETR

---

# R-CNN family · two-stage (brief)

**R-CNN (2014)** → **Fast R-CNN (2015)** → **Faster R-CNN (2015)**

Two-stage idea:
1. **Propose** candidate regions (where might objects be?).
2. **Classify** each region (what's in it?).

Evolution:

| | How regions proposed | Inference time |
|--|---------------------|----------------|
| R-CNN | selective search (outside CNN) | ~50 s |
| Fast R-CNN | selective search + shared backbone | ~2 s |
| Faster R-CNN | **Region Proposal Network (RPN)** inside CNN | ~0.1 s |

<div class="insight">

Two-stage detectors still win on accuracy for small objects. They are slower and more complex — often replaced by YOLO in production.

</div>

---

# YOLO · you only look once

![w:920px](figures/lec09/svg/yolo_grid.svg)

---

# YOLO loss · three terms

<div class="math-box">

$$\mathcal{L} = \lambda_\text{coord} \sum \mathcal{L}_\text{box} + \sum \mathcal{L}_\text{conf} + \sum \mathcal{L}_\text{class}$$

1. **Box regression** — MSE on $(x, y, \sqrt{w}, \sqrt{h})$ offsets (sqrt so small boxes matter).
2. **Object confidence** — how likely is there *any* object in this anchor? BCE with logits.
3. **Classification** — softmax CE (or sigmoid per-class for multi-label).

</div>

Anchor boxes are predefined aspect ratios — the model predicts *deltas* from them, not absolute coordinates.

---

# Why predict deltas, not absolute boxes?

<div class="keypoint">

A conv layer is **translation equivariant** — the same filter at different spatial positions produces the same feature. If you asked it to predict **absolute** $(x, y)$, every spatial location would have to re-derive its own position. Instead we predict **$\Delta$ from a known anchor** at each location, and the conv output is the offset.

</div>

- `anchor = (x_a, y_a, w_a, h_a)` · known, per-cell
- `predict (tx, ty, tw, th)` · 4 scalars from the conv
- `box = (x_a + tx·w_a, y_a + ty·h_a, w_a · exp(tw), h_a · exp(th))`

That exponential makes width/height positive; the anchor does the heavy lifting so the network only has to learn a small correction.

---

# Speed vs accuracy · a detector comparison

| Detector | mAP (COCO) | FPS (V100) | Notes |
|:-:|:-:|:-:|:-:|
| Faster R-CNN | 42 | ~5 | accuracy king, slow |
| YOLOv8-m | 50 | ~250 | production default |
| DETR | 44 | ~30 | elegant, data-hungry |
| RT-DETR | 53 | ~100 | real-time Transformer-based |

<div class="insight">

Choose by constraint · real-time camera feed → YOLO. Labeled-data poor → DETR with good augmentations. Highest accuracy → large backbone + Faster R-CNN variant. There is no universally-best detector.

</div>

---

# The YOLO lineage

| Version | Year | Key contribution |
|---------|------|------------------|
| YOLOv1 | 2015 | grid formulation, one shot |
| YOLOv3 | 2018 | multi-scale predictions, anchor clustering |
| YOLOv5 | 2020 | mosaic augmentation, practical toolkit |
| YOLOv8 | 2023 | anchor-free, efficient |
| **YOLOv11** | **2024** | current production default |

<div class="realworld">

For any real-time detection task in 2026, start with `ultralytics` YOLOv11. `pip install ultralytics` → model downloads + runs in 10 lines.

</div>

---

# DETR · detection as set prediction (2020)

End-to-end with a Transformer:

1. CNN backbone → feature map.
2. Transformer encoder/decoder processes it.
3. Decoder outputs a **set** of $(x, y, w, h, \text{class})$ tuples.
4. Hungarian matching pairs predictions with ground truth.

**No** anchor boxes. **No** NMS. Just a fixed-size set.

<div class="insight">

DETR cleans up detection conceptually but is slow and data-hungry. YOLO still wins on speed; DETR wins on elegance.

</div>

---

<!-- _class: section-divider -->

### PART 4

# Semantic segmentation · U-Net

Pixel-level classification

---

# From detection to segmentation

Detection gives boxes around objects. Segmentation gives a **label per pixel**.

Key architectural change: we need to go *back up* in spatial resolution — the feature map shrinks through convs/pooling, but the output must match the input size.

**Solution**: encoder-decoder with upsampling.

---

# U-Net architecture

![w:920px](figures/lec09/svg/unet_architecture.svg)

---

# U-Net · with channels and sizes

![w:920px](figures/lec09/svg/unet_annotated.svg)

<div class="realworld">

▶ Interactive: click an image region, see segmentation fill in — [image-segmentation](https://nipunbatra.github.io/interactive-articles/image-segmentation/).

</div>

---

# Why skip connections are essential

The encoder compresses spatial info into richer features. But **spatial precision is lost** — a 16×16 feature map can't localize edges accurately.

Skip connections let the decoder *concatenate* encoder features at matching resolution:

```python
def forward(self, x):
    e1 = self.enc1(x)         # 128×128
    e2 = self.enc2(pool(e1))  #  64×64
    e3 = self.enc3(pool(e2))  #  32×32
    b  = self.bridge(pool(e3))

    d3 = self.dec3(cat([up(b),  e3]))    # ← skip from e3
    d2 = self.dec2(cat([up(d3), e2]))    # ← skip from e2
    d1 = self.dec1(cat([up(d2), e1]))    # ← skip from e1
    return self.final(d1)
```

Every modern segmentation net (DeepLab, SegFormer) uses this pattern.

---

# Segmentation loss functions

- **Pixel-wise cross-entropy** · default. Per-pixel softmax over $C$ classes.
- **Dice loss** · $1 - \frac{2|P \cap T|}{|P| + |T|}$ — robust to class imbalance (small objects).
- **Focal loss** · down-weights easy pixels so the model focuses on boundaries.
- **Boundary loss** · emphasize pixel accuracy near object edges.

<div class="warning">

**Class imbalance** is the #1 problem in segmentation. A medical image with 99% background pixels will optimize to "predict background always" under plain CE. Dice loss (or weighted CE) is what you reach for first.

</div>

---

# Why U-Net caught on in medical imaging

Ronneberger 2015 targeted electron microscopy cell segmentation. Two reasons it spread:

1. **Small data tolerance** · medical datasets are in the hundreds. U-Net's strong spatial prior (encoder-decoder + skip) makes this workable.
2. **Permissive license + simple architecture** · anyone could port it to any framework in a day.

By 2020, U-Net was the default segmentation network not just in medicine but in satellite imagery, materials science, audio-spectrogram analysis, and later **diffusion models** (L21 / L22) — where the same encoder-decoder-with-skips handles noise-to-image mapping.

---

# Instance segmentation · Mask R-CNN (brief)

Built on Faster R-CNN. Adds a third head:

- **Class** head (from Faster R-CNN)
- **Bbox** head (from Faster R-CNN)
- **Mask** head — a small FCN producing a pixel mask per region

<div class="paper">

He et al. 2017 · Mask R-CNN — cleanly combines detection and segmentation. Standard baseline for instance tasks.

</div>

---

<!-- _class: section-divider -->

### PART 5

# 2026 frontier · SAM

Zero-shot segmentation by prompting

---

# Segment Anything · SAM (Meta, 2023)

A **foundation model for segmentation**:

- Trained on 1.1 billion masks across 11M images.
- Takes a prompt — a point, a box, or text — and returns a mask.
- **Zero-shot** on unseen categories and domains.

<div class="keypoint">

SAM changed segmentation the way CLIP changed classification — you don't need to train for your specific dataset; you just prompt a pretrained model.

</div>

<div class="realworld">

In 2026: for most segmentation tasks, start with SAM-2 and fine-tune only if the domain is truly specialized (medical, satellite).

</div>

---

# The open-vocabulary shift

Classical detectors and segmenters had a **fixed class list** — if you trained for COCO's 80 classes, you couldn't detect "a purple thermos" without retraining.

<div class="keypoint">

**Open-vocabulary** models (OWLv2, GroundingDINO, SAM with text prompts) take a *text query* and localize it. Built by combining detector architectures with CLIP text embeddings.

</div>

- `"find the red bicycle"` → bounding box
- `"segment everything that looks like a tree"` → masks

The 2024–2026 frontier is fully prompt-driven vision: you describe what you want, the model localizes and segments it.

---

<!-- _class: summary-slide -->

# Lecture 9 — summary

- **Detection** = classification + localization at multiple objects.
- **IoU** quantifies bbox quality; **NMS** deduplicates; **mAP** is the headline metric.
- **R-CNN family** — propose regions then classify (two-stage, accurate, slow).
- **YOLO** — one forward pass on a grid (one-stage, fast, production standard).
- **DETR** — Transformer-based set prediction; no NMS.
- **U-Net** — encoder-decoder with skip connections · canonical for segmentation.
- **SAM** — zero-shot segmentation via prompting (2023+).

### Read before Lecture 10

Bishop Ch 12 · sequences and recurrence.

### Next lecture

**RNNs, LSTMs, GRUs** — why MLPs fail on sequences, BPTT, gating mechanisms.

<div class="notebook">

**Notebook 9** · `09-yolo-unet.ipynb` — run pretrained YOLOv11 on sample images; train a small U-Net on a toy segmentation task; measure IoU per class.

</div>
