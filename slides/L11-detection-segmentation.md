---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Detection & Segmentation

## Lecture 11 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

Your classifier answers *what is in this image?* with **one** label. A self-driving frame breaks that in one glance — seven pedestrians, three cars, and you need to know **where each one is**.

<div class="keypoint">

**Same backbone every time — only the *output shape* changes.** Classification → localization → detection → per-pixel segmentation is a **ladder of outputs**. For each rung we ask one question *first*: **what does the network emit, number by number?** Once you can name every number, **the loss almost writes itself.**

</div>

You met the CNN backbone and cross-entropy already. Today we bolt **new heads** onto that backbone and read the loss straight off the outputs.

$$\underbrace{K}_{\text{classify}} \rightarrow \underbrace{K+4}_{\text{localize}} \rightarrow \underbrace{(5+C)\text{ per cell}}_{\text{detect (YOLO)}} \rightarrow \underbrace{H\times W\times C}_{\text{segment}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**Built on what you already have:**
- the **CNN backbone** (last two lectures) — untouched, we only add heads
- **cross-entropy** from L1 — still the class loss, now *per box* and *per pixel*
- **ResNet skip connections** — U-Net reuses the exact same idea, but **concatenates** instead of adds

</div>
<div>

**Reused later:**
- the **U-Net** encoder-decoder returns as the denoiser in **diffusion (L21/L22)**
- the **multi-task loss** (weighted sum of heads) reappears whenever a net predicts more than one thing

</div>
</div>

<div class="insight">

Everything is the **negative log-likelihood** story from L1 — the only thing that changes is the **support** of the random variable: one label → one box → a box per cell → a label per pixel.

</div>

---

<!-- _class: section-divider -->

## Part 1 · The output ladder — classification to detection

---

# The spectrum of "what's in this image"

| Task | Output | Example |
|------|--------|---------|
| **Classification** | 1 label | "cat" |
| **Classification + localization** | 1 label + 1 box | "cat @ (60, 80, 200, 180)" |
| **Object detection** | many labels + many boxes | "cat @ … dog @ … car @ …" |
| **Semantic segmentation** | label per pixel | every pixel → class (no instance) |
| **Instance segmentation** | label + instance per pixel | "cat 1 vs cat 2 as separate masks" |

The naive fixes fail on **one property** of a real frame: the number of objects is **not fixed**. A 10-output classifier assumes 10 objects; a box regressor can't grow. A **detector** predicts a *variable* number of class+box pairs — that's the target for Parts 1–3.

---

# The ladder of outputs — the through-line for today

**Same backbone every time — only the *output shape* changes.** So for each new task we ask one question **first**: *what does the network emit, number by number?* Once you can name every number, the loss almost writes itself.

![w:900px](figures/lec09/svg/output_ladder.svg)

---

# Name every number — then the loss almost writes itself

This is the one habit for the whole lecture. Before touching a loss, **list the numbers the head emits and the *support* of each** — is it a probability, a real coordinate, a yes/no? The support forces the distribution, and the loss is its negative log-likelihood (straight from L1).

<div class="columns">
<div>

| Output group | Support | Distribution | Loss |
|--|--|--|--|
| class scores | one-of-$K$ | Categorical | cross-entropy |
| box $x,y,w,h$ | real | Gaussian | Smooth L1 / MSE |
| objectness | yes/no | Bernoulli | BCE |
| pixel label | one-of-$C$ | Categorical | per-pixel CE |

</div>
<div>

Every task today is just a **different pile of these rows**. Localize $=$ class $+$ box. Detect $=$ objectness $+$ box $+$ class, per cell. Segment $=$ one class row *per pixel*.

</div>
</div>

<div class="keypoint">

**The loss is never invented.** Read the output groups, match each to its distribution, take $-\log$, sum. That is the entire recipe — L1's NLL story wearing a vision hat.

</div>

---

# Worked numeric · count the ladder's numbers

Take a toy detector: $C = 3$ classes (cat/dog/car), a $7\times7$ grid, $B = 2$ anchors per cell.

<div class="math-box">

- **Classify** · $C = \mathbf{3}$ numbers (one softmax).
- **Classify + localize** · $C + 4 = \mathbf{7}$ numbers (add one box).
- **Detect** · per anchor $5 + C = 8$; per cell $B(5{+}C) = 16$; whole grid $7 \times 7 \times 16 = \mathbf{784}$ numbers.
- **Segment** (image $224\times224$) · $224 \times 224 \times 3 \approx \mathbf{150{,}000}$ numbers — a full softmax *per pixel*.

</div>

<div class="insight">

Same backbone, four wildly different output sizes — from **3** numbers to **150k**. The count explodes; the recipe doesn't. Each number still belongs to exactly one distribution and one loss term. **"Name every number" scales — memorising architectures doesn't.**

</div>

---

# Classification + localization · output first, then loss

**Name every number.** Keep the CNN, add **two heads**. For one object the network emits exactly **$K + 4$ numbers**: $K$ class scores, then 4 box numbers $[x, y, w, h]$. Nothing else.

```python
class ClsLocHead(nn.Module):
    def __init__(self, feat_dim, n_classes):
        super().__init__()
        self.cls = nn.Linear(feat_dim, n_classes)   # K class scores
        self.box = nn.Linear(feat_dim, 4)           # x, y, w, h
```

<div class="insight">

**Analogy · grading a two-section exam.** Section 1 is multiple-choice (the class); Section 2 is a drawing (the box). The final score adds the two — and if the drawing matters more, you weight it. Our loss does exactly this.

</div>

---

# The multi-task loss writes itself

Two heads → **two loss terms, added up** (true class $y$, true box $\mathbf{b}$):

<div class="math-box">

$$\mathcal{L} = \underbrace{\mathcal{L}_\text{class}(\text{logits}, y)}_{\text{cross-entropy on the } K \text{ scores}} \;+\; \lambda \cdot \underbrace{\mathcal{L}_\text{box}(\hat{\mathbf{b}}, \mathbf{b})}_{\text{Smooth L1 on the 4 numbers}}$$

</div>

- $\lambda$ **balances** the two heads — the "weight the drawing" knob from the exam analogy.
- $\mathcal{L}_\text{box}$ is **Smooth L1**: quadratic near 0 (precise), linear far away (robust to outlier boxes).

<div class="keypoint">

There is no new machinery here — a multi-output network just **sums one loss per output group**. Every detector below is a longer version of this same sum.

</div>

---

# Worked numeric · multi-task loss

One image of a cat. True class index 1. True box $[100, 120, 50, 80]$.

Model predicts:
- Class logits $[0.1, 2.5]$ → confidently "cat" → say $\mathcal{L}_\text{class} = 0.08$.
- Box $\hat{\mathbf{b}} = [102, 119, 55, 81]$.

**Box loss (squared error for illustration).**
$\mathcal{L}_\text{box} = (102{-}100)^2 + (119{-}120)^2 + (55{-}50)^2 + (81{-}80)^2 = 4 + 1 + 25 + 1 = 31$

**Total ($\lambda = 0.1$).**
$\mathcal{L} = 0.08 + 0.1 \cdot 31 = 0.08 + 3.1 = \mathbf{3.18}$

This single number `3.18` is what backprop sees → it updates the class head **and** the box head simultaneously.

---

<!-- _class: section-divider -->

## Part 2 · IoU and non-max suppression

---

# IoU · how good is a box?

A box is right if it **overlaps** the ground truth. Quantify overlap as **intersection over union**:

$$\text{IoU} = \frac{|\,\text{prediction} \cap \text{truth}\,|}{|\,\text{prediction} \cup \text{truth}\,|} \in [0, 1]$$

![w:620px](figures/lec09/svg/iou_nms.svg)

$\text{IoU}=1$ is a perfect box; below **0.5** is almost always scored a miss.

---

# IoU · visual examples

![w:900px](figures/lec09/svg/iou_visual_examples.svg)

<div class="insight">

Detectors report **mAP at several IoU thresholds** because a box that is 90% right (IoU 0.9) is far more useful than one that is 60% right (IoU 0.6) — the metric *penalizes imprecision*, not just misses.

</div>

---

# Intuition · IoU is overlap *over union*, not just overlap

Why divide by the **union** at all — why not just measure the shared area?

<div class="columns">
<div>

**Overlap alone can be cheated.** Draw one giant box covering the whole image and it overlaps every object perfectly — huge intersection, but a useless "detection."

</div>
<div>

**Union is the honest denominator.** It grows whenever the box is too big *or* misplaced, so IoU punishes a sloppy oversized box as hard as a missed one. Only a box of the *right size and place* scores near 1.

</div>
</div>

<div class="insight">

Read it as a **fraction of agreement**: of all the pixels *either* box claims, what share do they *agree* on? $\text{IoU}=1$ only when the two boxes are identical. That single ratio is why it became the field's standard overlap score — and why it returns, unchanged, to grade *masks* in Part 4.

</div>

---

# IoU · a quick worked example

Predicted box · $(x_1, y_1, x_2, y_2) = (50, 50, 150, 150)$ · area $= 100 \times 100 = 10{,}000$.
Ground-truth box · $(100, 100, 200, 200)$ · area $= 10{,}000$.

Intersection · $(100, 100, 150, 150)$ · area $= 50 \times 50 = 2{,}500$.
Union · $10{,}000 + 10{,}000 - 2{,}500 = 17{,}500$.

$$\text{IoU} = \frac{2{,}500}{17{,}500} \approx 0.14$$

Barely overlapping → this would be scored a **miss**. The recipe: build the intersection rectangle, then use $|A \cup B| = |A| + |B| - |A \cap B|$.

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** Predicted box $(10, 10, 50, 50)$ and ground-truth box $(15, 15, 55, 55)$, format $(x_1, y_1, x_2, y_2)$. Compute the **IoU**. Is this a **true positive** at threshold $0.5$?

</div>

*Try it before the next slide — build the intersection rectangle first.*

---

# Solution · practice problem 1

Both boxes are $40 \times 40$, so each area $= 1{,}600$.

**Intersection** · $x \in [15, 50] \to 35$, $\;y \in [15, 50] \to 35$ · area $= 35 \times 35 = 1{,}225$.
**Union** · $1{,}600 + 1{,}600 - 1{,}225 = 1{,}975$.

$$\text{IoU} = \frac{1{,}225}{1{,}975} \approx \mathbf{0.62}$$

<div class="keypoint">

$0.62 > 0.5$ → a **true positive**. Note the intersection uses $\max$ of the top-left corners and $\min$ of the bottom-right corners — the same clamp you'd write in three lines of code.

</div>

---

# NMS · deleting the duplicates

After prediction you get a **pile of overlapping boxes** on the same object — like several people shouting the same answer.

<div class="keypoint">

**Non-max suppression** · keep the most confident shout, silence the rest. Sort boxes by confidence, keep the best, discard any later box that overlaps it (IoU above threshold). Repeat until the pool is empty.

</div>

<div class="notebook">

**🎛 Interactive · draw boxes, watch IoU + NMS live** — drag two boxes on a canvas and see the intersection, union, and which duplicates get suppressed. *(Interactive Lab · `interactive/articles/object-detection` · [object-detection](https://nipunbatra.github.io/interactive-articles/object-detection/))*

</div>

---

# Intuition · the loudest detection silences its neighbours

Picture a crowded room where several people spot the same cat and all shout a box. NMS is the **chairperson's rule**: believe the loudest voice, then tell everyone standing *near* that person to be quiet — they're describing the same cat.

<div class="columns">
<div>

**"Loudest"** $=$ highest confidence. That box is kept; it becomes the accepted answer for its patch of image.

</div>
<div>

**"Near"** $=$ high IoU with the kept box. Overlap above the threshold means "same object" → suppressed. Low overlap means a *different* cat, so it survives to shout next.

</div>
</div>

<div class="keypoint">

It is **greedy and per-object**: the winner silences only its own neighbourhood, never a distant detection. The threshold is the knob — too low deletes real nearby objects (two cats hugging), too high leaves duplicates. That trade-off is why **crowded scenes are the classic NMS failure case**.

</div>

---

# NMS · step-by-step example

5 predicted boxes for one car, IoU threshold 0.5.
Scores · **A** 0.95 · **B** 0.90 · **C** 0.80 · **D** 0.75 · **E** 0.70.

**Step 1 · pick A** (highest). Add to `keep`.
IoU(A, others): IoU(A,B)=0.8 → drop B; IoU(A,C)=0.2 → keep; IoU(A,D)=0.7 → drop D; IoU(A,E)=0.1 → keep.
Pool now `[C, E]`.

**Step 2 · pick C.** IoU(C, E)=0.15 → keep E. Pool now `[E]`.

**Step 3 · pick E.** Pool empty. Stop.

**Final · `keep = [A, C, E]`** — 5 boxes collapse to 3 clean detections. Greedy: the highest-confidence box wins its neighborhood.

---

# How do we grade a detector? · precision and recall

<div class="insight">

**Analogy · a search engine for "cat."** You return 10 results.
- **Precision** = of the answers you gave, how many were right? (8/10 → 80%.)
- **Recall** = of all the right answers out there, how many did you find? (8 of 100 → 8%.)

</div>

They trade off: return everything → recall 100%, precision terrible; return only your one surest box → precision 100%, recall terrible. A prediction counts as a **true positive** only if its **IoU** with a ground-truth box clears the threshold — so the two ideas from Part 2 feed straight into the metric.

**mAP** (mean average precision) summarizes the whole precision–recall trade-off in one number — the headline score for every detector.

---

# Computing AP · a worked example

3 ground-truth cats. 5 predictions, sorted by confidence. IoU $> 0.5$ → True Positive.

| Rank | Conf | TP/FP | TP cum | FP cum | Recall (TP/3) | Precision (TP/(TP+FP)) |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 1 | 0.98 | TP | 1 | 0 | 0.33 | 1.00 |
| 2 | 0.95 | TP | 2 | 0 | 0.67 | 1.00 |
| 3 | 0.88 | FP | 2 | 1 | 0.67 | 0.67 |
| 4 | 0.75 | TP | 3 | 1 | 1.00 | 0.75 |
| 5 | 0.60 | FP | 3 | 2 | 1.00 | 0.60 |

- **AP** = area under the precision–recall curve traced by these points.
- **mAP** = average AP over all classes. **mAP@0.5** uses IoU threshold 0.5; **mAP@[0.5:0.95]** averages over 10 thresholds (the COCO standard).

---

<!-- _class: section-divider -->

## Part 3 · YOLO — detection in one forward pass

---

# R-CNN family · two-stage (brief)

Before YOLO, detectors were **two-stage**: (1) **propose** candidate regions, then (2) **classify** each region.

| | How regions proposed | Inference |
|--|---------------------|----------------|
| R-CNN (2014) | selective search (outside the CNN) | ~50 s |
| Fast R-CNN (2015) | selective search + shared backbone | ~2 s |
| Faster R-CNN (2015) | **Region Proposal Network** inside the CNN | ~0.1 s |

<div class="insight">

Two-stage detectors are accurate (still strong on small objects) but slower and more complex. The next idea collapses both stages into **one pass**.

</div>

---

# YOLO · you only look once

**Big idea, two sentences.** Instead of *propose regions, then classify*, YOLO lays an $S \times S$ **grid** on the image and makes **every cell predict its own boxes and classes in one forward pass**. One look, one output tensor, done.

![w:880px](figures/lec09/svg/yolo_grid.svg)

---

# A detector's output, number by number

**Ask the question again: what does one cell emit?** Each grid cell (per anchor) outputs **one vector** — **1** objectness + **4** box numbers + **C** class scores:

![w:880px](figures/lec09/svg/detector_output.svg)

Three groups of numbers go in → three loss terms come out. That mapping *is* the whole YOLO loss.

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** A YOLO head uses an $S \times S$ grid with $B$ anchor boxes per cell on a $C$-class dataset. Write the shape of the **output tensor** (name every number a cell emits). Then compute the **total number of predicted values** for $S = 13$, $B = 3$, $C = 80$.

</div>

*Try it before the next slide — start from one anchor.*

---

# Solution · practice problem 2

**One anchor emits** $\;\underbrace{4}_{\text{box } x,y,w,h} + \underbrace{1}_{\text{objectness}} + \underbrace{C}_{\text{class scores}} = 5 + C$ numbers.

<div class="math-box">

$$\text{output tensor shape} = S \times S \times B \times (5 + C)$$

</div>

Plugging in $S=13$, $B=3$, $C=80$:

$$5 + C = 85, \quad B(5+C) = 255, \quad \underbrace{13 \times 13}_{169 \text{ cells}} \times 255 = \mathbf{43{,}095 \text{ numbers}}$$

Shape $(13, 13, 255)$ — a fixed-size tensor no matter how many objects are actually present. **That fixed shape is exactly how a one-pass net handles a *variable* object count.**

---

# YOLO loss · one term per output group

The output vector has three groups, so the loss has **three matching terms**, summed over every cell:

$$\mathcal{L} = \lambda_\text{coord}\sum\mathcal{L}_\text{box} \;+\; \sum\mathcal{L}_\text{obj} \;+\; \sum\mathcal{L}_\text{class}$$

- **$p_\text{obj}$ → objectness** · active **always** — BCE pushes toward 1 if an object is centred here, 0 if not.
- **$x,y,w,h$ → box** · active **only** where an object lives. MSE on $(x, y, \sqrt{w}, \sqrt{h})$ — the $\sqrt{\cdot}$ makes small boxes matter as much as large ones.
- **$c_1..c_C$ → class** · active **only** where an object lives — cross-entropy.

<div class="insight">

Every number in the output vector is graded by exactly **one** term. Objectness is checked in *every* cell (most are empty); box and class only where an object actually sits.

</div>

---

# Worked numeric · YOLO loss for one cell

Cell holds a **dog** (class index 2). **Ground truth** · box $[0.5, 0.5, 0.2, 0.3]$, objectness 1, class one-hot $[0,0,1,0,\ldots]$.
**Prediction** · $\hat{\mathbf{b}} = [0.6, 0.4, 0.25, 0.31]$, objectness 0.85, class probs $[0.1, 0.2, 0.6, 0.1,\ldots]$.

$$\mathcal{L}_\text{box} = (0.6{-}0.5)^2 + (0.4{-}0.5)^2 + (\sqrt{0.25}{-}\sqrt{0.2})^2 + (\sqrt{0.31}{-}\sqrt{0.3})^2 \approx \mathbf{0.023}$$

- $\mathcal{L}_\text{obj} = -\log(0.85) \approx 0.16$
- $\mathcal{L}_\text{class} = -\log(0.6) \approx 0.51$

**Total per cell** (no weighting): $0.023 + 0.16 + 0.51 \approx \mathbf{0.69}$. Sum over all cells = the full-image loss.

<div class="notebook">

**📓 Notebook · run YOLO end-to-end** — `pip install ultralytics`, run a pretrained YOLO on sample images, and print the raw output tensor to see the $(5+C)$ groups per cell. *(ES 667 · `notebooks/11-yolo-unet.ipynb`)*

</div>

---

# Why predict deltas, not absolute boxes · anchor boxes

<div class="insight">

**Analogy · giving directions.** *Absolute:* "go to GPS (40.7128, −74.0060)" — hard, different from every starting point. *Relative:* "50 m forward, 10 m right" — easy, works from any corner.

</div>

**Anchor boxes** are the starting corners: a few fixed box shapes tiled at every cell. The network predicts a **small correction** to the nearest anchor, not the box from scratch.

<div class="keypoint">

Convolutions are **translation-equivariant** — the same filter runs at every location, so it should learn the *same correction style* everywhere, not different absolute coordinates. Anchors + deltas make that possible. *(Decode algebra in the appendix.)*

</div>

---

# Intuition · anchor boxes are *priors on shape*

Where have we seen "hand the model a sensible starting belief, let it learn a small correction" before? **L1 — the MAP prior.** Anchors are that idea, applied to box geometry.

<div class="columns">
<div>

**Objects have characteristic shapes.** A standing pedestrian is *tall and thin*; a car is *wide and short*; a plate is *square*. These are priors on the box's aspect ratio.

</div>
<div>

**Anchors encode those priors.** We tile a few fixed shapes at every cell. The network never guesses a box from nothing — it nudges the *closest* prior (the deltas of the appendix).

</div>
</div>

<div class="keypoint">

Just as a Gaussian prior pulls a weight gently toward zero, an anchor pulls a prediction toward a **plausible shape** — so the net spends its capacity on the small correction, not on rediscovering that pedestrians are tall. Good anchors (clustered from the data, YOLOv3) are simply a *well-chosen prior*.

</div>

---

# Worked numeric · which anchor is responsible?

An object with normalized box size $(w, h) = (0.2, 0.4)$ — **tall**. Three anchors tiled at its cell, centres aligned. Which one "owns" it? → the highest **shape IoU**.

<div class="math-box">

| Anchor | $(w,h)$ | $\cap = \min w \cdot \min h$ | $\cup = a_\text{gt}{+}a_\text{anc}{-}\cap$ | IoU |
|--|--|--|--|--|
| A1 tall | $(0.2, 0.4)$ | $0.2\cdot0.4=0.08$ | $0.08$ | $\mathbf{1.00}$ |
| A2 square | $(0.3, 0.3)$ | $0.2\cdot0.3=0.06$ | $0.08{+}0.09{-}0.06=0.11$ | $0.55$ |
| A3 wide | $(0.5, 0.2)$ | $0.2\cdot0.2=0.04$ | $0.08{+}0.10{-}0.04=0.14$ | $0.29$ |

</div>

**A1 wins** → it is the anchor made *responsible* for this object; its box + class terms switch on, the other two stay background.

<div class="insight">

This is exactly how the "active only where an object lives" gate is *decided* in training: match each ground-truth box to its best-IoU anchor, and only that anchor pays the box and class loss. Anchors turn a variable list of objects into a **fixed set of slots**.

</div>

---

# One-stage vs two-stage · speed vs accuracy

| Detector | mAP (COCO) | FPS | Notes |
|:-:|:-:|:-:|:-:|
| Faster R-CNN | 42 | ~5 | two-stage baseline, accurate, slow |
| YOLOv8-m | 50 | ~250 | one-stage production default |
| DETR | 44 | ~30 | Transformer set-prediction, no NMS, data-hungry |
| RT-DETR | 53 | ~100 | real-time Transformer |

<div class="insight">

No universally-best detector. **Real-time camera feed → YOLO. Labeled-data poor → a *pretrained* YOLO / Faster R-CNN. Highest accuracy → a large-backbone detector.** DETR replaces NMS with **Hungarian matching** (learned one-to-one assignment) — elegant, but slower to train.

</div>

---

# The YOLO lineage

| Version | Year | Key contribution |
|---------|------|------------------|
| YOLOv1 | 2015 | grid formulation, one shot |
| YOLOv3 | 2018 | multi-scale predictions, anchor clustering |
| YOLOv5 | 2020 | mosaic augmentation, a practical toolkit |
| YOLOv8 | 2023 | **anchor-free**, efficient |
| **YOLOv11** | **2024** | widely-used production default |

<div class="insight">

The formulation you just learned (grid + anchors + three-term loss) is YOLOv1. Later versions predict at **multiple scales** and, from v8 on, go **anchor-free** (predict box centres directly) — but the "name every number, one loss term per group" story is unchanged. For any real-time task today, start with `ultralytics` YOLOv11: `pip install ultralytics` and it runs in ~10 lines.

</div>

---

<!-- _class: section-divider -->

## Part 4 · Semantic segmentation — U-Net

---

# From detection to segmentation · output, pixel by pixel

**The same question, one more rung up the ladder — what does the network emit?** Now it is **one class label per pixel**: an $H \times W \times C$ tensor where *every* pixel carries its own softmax over $C$ classes — a tiny classifier at each location.

- Detection output = a few boxes. Segmentation output = a full **label map** the size of the image.
- Concrete: for a $2\times2$ patch the network emits 4 per-pixel labels.

To emit a full-resolution map, the feature map (shrunk by pooling) has to climb **back up** in resolution — an **encoder-decoder with upsampling**: the U-Net.

---

# U-Net architecture

![w:900px](figures/lec09/svg/unet_architecture.svg)

The **encoder** (left) contracts — pool down, get richer features. The **decoder** (right) expands — upsample back to full resolution. The horizontal arrows are the **skip connections**, the crux of the whole design.

---

# U-Net · with channels and sizes

![w:900px](figures/lec09/svg/unet_annotated.svg)

<div class="notebook">

**🎛 Interactive · watch skips restore detail** — toggle the skip connections on and off and see the mask boundaries sharpen or blur. *(Interactive Lab · `interactive/articles/unet` · [unet](https://nipunbatra.github.io/interactive-articles/unet/) · click-to-fill: [image-segmentation](https://nipunbatra.github.io/interactive-articles/image-segmentation/))*

</div>

---

# U-Net skips · the same idea as ResNet, but concatenate

<div class="columns">
<div>

**ResNet skip** — *add* the input back:
$$\text{out} = F(x) + x$$
keeps gradients flowing, learns a residual.

</div>
<div>

**U-Net skip** — *concatenate* the encoder feature:
$$\text{out} = [\,\text{up}(d);\; e\,]$$
hands the decoder the original crisp detail.

</div>
</div>

<div class="keypoint">

Think of the encoder as a **rich but blurry summary** — "a kidney here, a vessel there," but exact edges smudged by pooling. The skip is a **cheat sheet** from the matching-resolution encoder layer: high-level summary for *what*, crisp pixels for *exactly where the boundary is*.

</div>

```python
d3 = self.dec3(cat([up(b),  e3]))   # ← skip from encoder level 3
d2 = self.dec2(cat([up(d3), e2]))   # ← skip from encoder level 2
d1 = self.dec1(cat([up(d2), e1]))   # ← skip from encoder level 1
```

---

# Intuition · the decoder phones the encoder for the lost detail

Downsampling is **lossy on purpose**: pooling trades *where* for *what*. By the bottleneck the net knows "this blob is a kidney" but has forgotten the exact edge — those pixels were averaged away.

<div class="keypoint">

A skip connection is the decoder **phoning back to the encoder layer at its own resolution**: *"You still remember the sharp edges here — send them over."* The decoder supplies the **semantics** (what), the encoder's saved feature map supplies the **geometry** (exactly where). Concatenate the two and the label comes out both correct *and* crisply placed.

</div>

<div class="insight">

Detail lost to pooling is **recalled, not reinvented** — the encoder still holds it. That is the whole reason a U-Net beats a plain encoder-decoder: without the phone line the decoder would have to *hallucinate* boundaries from a blurry summary.

</div>

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** A U-Net encoder downsamples $5\times$ (a $256\times256$ image becomes an $8\times8$ feature map), then the decoder upsamples $5\times$ back to $256\times256$. **Explain why the skip connections are what give *sharp* masks.** What happens to boundary sharpness if you delete them?

</div>

*Try it before the next slide — think about what an $8\times8$ map can and cannot localize.*

---

# Solution · practice problem 3

Pooling throws away **spatial precision**: an $8\times8$ feature map has rich semantics ("this region is kidney") but can localize an edge only to within $\sim32$ pixels — upsampling a coarse map just produces **blocky, blurry** boundaries.

<div class="keypoint">

The skip connections **concatenate the high-resolution encoder features** (which still hold the crisp $256\times256$ edge detail) into the decoder at each level. The decoder fuses *semantics from the bottleneck* with *spatial detail from the skips* → **sharp, accurate masks**. Delete them and the class is right but boundaries turn to mush — the network simply has no source of fine spatial detail left.

</div>

Detail lost in downsampling is **restored** by the skip, not reinvented — the same reason every modern segmentation net (DeepLab, SegFormer) keeps this pattern.

---

# Segmentation loss · from IoU to Dice

**Output first, loss second:** the output is a per-pixel mask $P$; the loss scores how well $P$ overlaps the true mask $T$. For boxes we used IoU; for masks we can too:

$$\text{IoU} = \frac{|P \cap T|}{|P \cup T|} \qquad\qquad \text{Dice} = \frac{2\,|P \cap T|}{|P| + |T|}$$

IoU is a fine *metric* but its gradient is sharp — awkward for SGD. **Dice** is the smoother cousin, and optimizers minimize, so:

$$\mathcal{L}_\text{Dice} = 1 - \text{Dice} \qquad (\text{perfect} \to 0, \;\; \text{no overlap} \to 1)$$

<div class="warning">

**Class imbalance is the #1 segmentation trap.** A medical image that is 99% background optimizes to "predict background everywhere" under plain cross-entropy. Dice only counts pixels in $P$ or $T$ — the vast background of true negatives can't dominate. Reach for **Dice (or weighted CE)** first.

</div>

---

# Worked numeric · Dice loss on a 2×2 image

Task · segment the top-left pixel.

Ground truth $T = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$, prediction (probabilities) $P = \begin{pmatrix} 0.9 & 0.2 \\ 0.1 & 0.3 \end{pmatrix}$.

1. Intersection $|P \cap T| = \sum P \odot T = 0.9\cdot 1 + 0.2\cdot 0 + 0.1\cdot 0 + 0.3\cdot 0 = \mathbf{0.9}$
2. $|P| = 0.9 + 0.2 + 0.1 + 0.3 = 1.5$
3. $|T| = 1.0$
4. $\text{Dice} = \dfrac{2 \cdot 0.9}{1.5 + 1.0} = \dfrac{1.8}{2.5} = \mathbf{0.72}$
5. $\mathcal{L}_\text{Dice} = 1 - 0.72 = \mathbf{0.28}$

The model backprops this `0.28` — and, as the appendix shows, that gradient depends on **all four** pixels, not just the top-left.

---

# Practice problem 4

<div class="popquiz">

**Practice problem 4.** A $3\times3$ segmentation. The thresholded prediction $P$ and truth $T$ (both an L-shape of foreground) are

$$T=\begin{pmatrix}1&0&0\\1&0&0\\1&1&0\end{pmatrix}\qquad P=\begin{pmatrix}1&0&0\\1&0&0\\0&1&1\end{pmatrix}$$

Compute the **Dice score**, the **IoU**, and the **Dice loss** $1-\text{Dice}$. Is Dice larger or smaller than IoU here?

</div>

*Try it before the next slide — count $|P\cap T|$, $|P|$, $|T|$ first.*

---

# Solution · practice problem 4

Foreground pixels: $T$ has 4, $P$ has 4; they agree on 3 (both top-left column pixels and $(2,1)$).

$$|P\cap T| = 3, \qquad |P| = 4, \qquad |T| = 4$$

$$\text{Dice} = \frac{2\cdot 3}{4+4} = \frac{6}{8} = \mathbf{0.75}, \qquad \text{IoU} = \frac{3}{4+4-3} = \frac{3}{5} = \mathbf{0.60}$$

$$\mathcal{L}_\text{Dice} = 1 - 0.75 = \mathbf{0.25}$$

<div class="keypoint">

**Dice $\ge$ IoU always** (here $0.75 > 0.60$) — Dice double-counts the intersection in its numerator, so it is the *gentler* score with smoother gradients for SGD. They agree only at the extremes: both are $1$ for a perfect mask and $0$ for no overlap. Report IoU as the metric, optimise Dice as the loss.

</div>

---

# Other segmentation losses · when Dice isn't enough

| Loss | What it does | Reach for it when |
|------|--------------|-------------------|
| **Pixel-wise cross-entropy** | per-pixel softmax over $C$ classes | balanced classes, the default |
| **Dice / IoU loss** | overlap-based, ignores easy background | heavy foreground/background imbalance |
| **Weighted CE** | up-weight rare classes | mild imbalance, still want per-pixel CE |
| **Focal loss** | down-weight *easy* pixels | boundaries drowned out by flat interiors |
| **Boundary loss** | penalize errors near edges | edge accuracy is the goal |

<div class="insight">

In practice, **Dice + CE** (summed) is a strong default: CE gives clean per-pixel gradients everywhere, Dice keeps the rare foreground from being ignored. Same "sum one loss per objective" move as the detection multi-task loss.

</div>

<div class="notebook">

**📓 Notebook · train a tiny U-Net** — segment a toy dataset, compare **CE vs Dice** under 95% background, and measure IoU per class. *(ES 667 · `notebooks/11-yolo-unet.ipynb`)*

</div>

---

# Practice problem 5

<div class="popquiz">

**Practice problem 5.** A $10\times10$ medical scan has **4** foreground (tumour) pixels; the other 96 are background. A lazy model predicts **background everywhere**. Compute its **pixel accuracy** and its **Dice score** on the foreground. What does the gap tell you about which loss to train with?

</div>

*Try it before the next slide — accuracy counts *all* pixels; Dice counts only the foreground.*

---

# Solution · practice problem 5

**Pixel accuracy** · 96 of 100 pixels are correctly called background:

$$\text{acc} = \frac{96}{100} = \mathbf{96\%}$$

**Dice** · the model predicts *no* foreground, so $|P\cap T| = 0$:

$$\text{Dice} = \frac{2\cdot 0}{0 + 4} = \mathbf{0}$$

<div class="warning">

A model scores **96% accuracy while finding nothing** — accuracy and plain CE are dominated by the 96 easy background pixels. **Dice sees the disaster (0) because it never counts the true-negative background** — *the* reason to use Dice (or weighted CE) on imbalanced masks.

</div>

<div class="notebook">

**📓 Notebook · feel the imbalance trap** — retrain the toy U-Net with CE vs Dice on this split; CE flatlines at "all background," Dice recovers the foreground. *(ES 667 · `notebooks/11-yolo-unet.ipynb`)*

</div>

---

# Why U-Net spread — and where it goes next

Ronneberger 2015 targeted electron-microscopy **cell segmentation**. It escaped medicine for two reasons:

1. **Small-data tolerance** · medical datasets are in the hundreds; U-Net's strong spatial prior (encoder-decoder + skips) makes that workable.
2. **Simple + permissive** · anyone could port it to any framework in a day.

<div class="insight">

By 2020 U-Net was the default in satellite imagery, materials science, and audio spectrograms — and then in **diffusion models (L21/L22)**, where the *same* encoder-decoder-with-skips maps noise to image. The architecture you learned for masks is the one that will draw pictures later.

</div>

**Instance** segmentation (separate mask per object) adds a mask head onto Faster R-CNN — **Mask R-CNN** (He et al. 2017), enabled by RoIAlign (bilinear crop, no rounding).

---

<!-- _class: section-divider -->

## Part 5 · 2026 frontier — segment anything

---

# Frontier · SAM and open-vocabulary vision

<div class="columns">
<div>

**SAM — Segment Anything** (Meta, 2023)
- trained on **1.1B masks** / 11M images
- prompt with a point, box, or text → returns a mask
- **zero-shot** on unseen categories

Changed segmentation the way CLIP changed classification: **prompt a pretrained model** instead of training your own.

</div>
<div>

**Open-vocabulary detection**
- a classic detector has a **fixed dictionary** — train on cat/dog/car and it can *only* find those
- **CLIP** embeds image regions and text in one space
- match a **text prompt** ("a red bicycle") to image regions → box the matches
- OWLv2, GroundingDINO, SAM-with-text

</div>
</div>

<div class="keypoint">

The frontier is **fully prompt-driven** vision. Two caveats for SAM: it returns **masks, not labels** (pair with CLIP for semantics), and quality is **prompt-sensitive** on niche domains — fine-tune only when the domain is truly specialized (medical, satellite).

</div>

---

# Intuition · prompting beats a fixed dictionary

A classic detector ships with a **closed dictionary** — train it on {cat, dog, car} and it is *blind* to "zebra" forever; a new class means relabelling and retraining.

<div class="keypoint">

**CLIP breaks the dictionary open.** It embeds image regions and free text in **one shared space**, so "find the zebra" becomes *"which region's embedding is closest to the text 'a zebra'?"* — no retraining, any word you can type. Open-vocabulary detection (OWLv2, GroundingDINO) and SAM-with-text are this trick bolted onto a detector or segmenter.

</div>

<div class="notebook">

**🎛 Interactive · match images to text prompts in CLIP space** — type a caption and watch the cosine similarity to each image rise and fall; this is exactly the scoring an open-vocabulary detector runs per region. *(Interactive Lab · `interactive/articles/clip-zero-shot` · [clip-zero-shot](https://nipunbatra.github.io/interactive-articles/clip-zero-shot/))*

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Detection and segmentation don't replace the classifier — they bolt new heads onto the same backbone; name every number the head emits, and the loss is its negative log-likelihood.**

</div>

- **Ladder of outputs** · classify ($K$) → localize ($K{+}4$) → detect ($5{+}C$ per cell) → segment ($H\times W\times C$).
- **IoU** grades a box; **NMS** deletes duplicates; **mAP** is the headline metric.
- **YOLO** · one grid, one pass, a **fixed-shape tensor** handling a *variable* object count; three-term loss.
- **U-Net** · encoder-decoder with **skip connections** (ResNet's idea, concatenated) → sharp masks; **Dice** loss beats CE under class imbalance.

**Next (L12):** images hold still, sequences don't — RNNs and the problem of memory.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's own ES 667 course (IIT Gandhinagar) and standard references:

- **Output-ladder framing, IoU / NMS, YOLO loss, U-Net, Dice** — *ES 667 Deep Learning* detection & segmentation materials, N. Batra.
- **Pedagogical structure & pacing** — A. Ng, *Deep Learning Specialization*, **Course 4 (CNNs), Week 3** (detection & segmentation).
- **YOLO** — Redmon, Divvala, Girshick & Farhadi, *You Only Look Once: Unified, Real-Time Object Detection*, CVPR 2016.
- **U-Net** — Ronneberger, Fischer & Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.
- **Open-vocabulary / prompt-driven frontier** — Radford et al., *Learning Transferable Visual Models from Natural Language Supervision* (CLIP), ICML 2021; Kirillov et al., *Segment Anything* (SAM), ICCV 2023.
- **Interactives** — Interactive Lab (`~/git/interactive`): `object-detection`, `image-segmentation`, `unet`, `clip-zero-shot`.

Figures adapted from the ES 667 figure library (`figures/lec09/`). All source material © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · decoding the box from an anchor

Given an anchor $(x_a, y_a, w_a, h_a)$ and the network's four predicted deltas $(t_x, t_y, t_w, t_h)$, the box is the **anchor, gently corrected**:

$$x = x_a + w_a\,t_x, \quad y = y_a + h_a\,t_y, \quad w = w_a\,e^{t_w}, \quad h = h_a\,e^{t_h}$$

![w:900px](figures/lec09/svg/anchor_delta_decode.svg)

- Centre: a **shift** scaled by the anchor size, so the same $t_x$ means "a fraction of a box over" at every scale.
- Size: a **log-space stretch**, $e^{t_w}$, which stays **positive** for any real $t_w$ — the net never has to output a negative width.

**Numeric.** Anchor $(50, 50, 20, 40)$, deltas $(0.1, -0.2, 0.0, 0.3)$: $x = 50 + 20(0.1) = 52$, $y = 50 + 40(-0.2) = 42$, $w = 20\,e^0 = 20$, $h = 40\,e^{0.3} \approx 54$. The network only learns **small corrections**; the anchor does the heavy lifting.

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · why Dice's gradient is non-local

Write the soft Dice loss with $I = \sum_i P_i T_i$ (intersection) and $U = \sum_i P_i + \sum_i T_i$ (sum of sizes):

$$\mathcal{L}_\text{Dice} = 1 - \frac{2I}{U} \;\Longrightarrow\; \frac{\partial \mathcal{L}_\text{Dice}}{\partial P_j} = -\,\frac{2\,T_j\,U - 2\,I}{U^2} = -\,\frac{2\,(T_j\,U - I)}{U^2}$$

<div class="insight">

The gradient at pixel $j$ contains the **global** sums $I$ and $U$ — it depends on **every** pixel in the mask, not just $P_j$. That coupling is exactly what lets Dice ignore a huge easy background and chase the small foreground.

</div>

Contrast **pixel-wise BCE**, whose gradient is fully **local**:

$$\mathcal{L}_\text{BCE} = -\sum_i \big[T_i \log P_i + (1{-}T_i)\log(1{-}P_i)\big] \;\Longrightarrow\; \frac{\partial \mathcal{L}_\text{BCE}}{\partial P_j} = \frac{P_j - T_j}{P_j(1 - P_j)}$$

depends on $P_j, T_j$ **alone**. Locality is why plain BCE lets 99%-background images collapse to "predict background" — no single pixel's gradient ever sees the class imbalance.
