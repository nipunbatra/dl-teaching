---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# CNN Deep Dive &amp; Classic Architectures

## Lecture 7 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. State the **output-size formula** and compute it.
2. Explain **receptive field** and its growth with depth.
3. Apply the **VGG insight** · 3 stacked 3×3 ≈ one 7×7.
4. Describe **1×1 conv** as channel mixing / bottleneck.
5. Name the **three inductive biases** of conv and when they fail.
6. Place classic architectures · LeNet → AlexNet → VGG → GoogLeNet.

---

# Recap · where we are

- **Training stack**: PyTorch recipe, debug ladder, error analysis (L3).
- **Optimizer**: AdamW + warmup + cosine (L4–L5).
- **Regularization**: weight decay, aug, Mixup, dropout, BN/LN/RMSNorm (L6).

<div class="paper">

Today maps to **UDL Ch 10** · *Convolutional networks* (early sections).

ES 654 covered LeNet and CNN basics. We skim those and spend time on **receptive fields, the classic architecture progression, and inductive biases**.

</div>

---

# Four questions

1. What does convolution *actually compute*? (brisk — prereq knows)
2. How does receptive field grow with depth?
3. Why did architectures converge on stacked 3×3 convs?
4. What inductive biases does convolution bake in — and why do they help?

---

<!-- _class: section-divider -->

### PART 1

# Convolution mechanics

A brisk recap with new diagrams

---

# Convolution · the feature-detector view

<div class="keypoint">

A convolution kernel is a **learned feature detector**.

- Kernel A · responds to vertical edges
- Kernel B · responds to horizontal edges
- Kernel C · responds to red-green color transitions

Slide each detector across the entire image · the output map "lights up" wherever its specific feature appears.

</div>

In a CNN we don't hand-design these kernels · the network *learns* the most useful detectors during training. Early layers learn edges and textures; deeper layers compose those into parts and objects.

---

# Convolution — sliding window, shared weights

![w:920px](figures/lec07/svg/convolution_mechanics.svg)

<div class="realworld">

▶ Interactive: drag the kernel across an image, see the feature map fill in live — [convolution-visualizer](https://nipunbatra.github.io/interactive-articles/convolution-visualizer/).

</div>

---

# The output-size formula

For input size $W$, kernel $K$, padding $P$, stride $S$:

$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

- **padding** lets output preserve input size (with $P = (K-1)/2$ and $S = 1$)
- **stride** downsamples — halves spatial resolution at $S = 2$
- **dilation** spaces kernel elements apart — enlarges RF without more params

`nn.Conv2d(64, 128, kernel_size=3, padding=1)` — the line you'll write a thousand times.

---

# A small numeric check

Input `(3, 224, 224)` (RGB ImageNet image). Apply `Conv2d(3, 64, kernel_size=7, stride=2, padding=3)`:

$$O = \lfloor (224 - 7 + 6)/2 \rfloor + 1 = 112$$

Output shape: `(64, 112, 112)`. Compare to MLP · same layer would be $224^2 \cdot 3 \cdot (64 \cdot 112^2) \approx 10^{11}$ ops. The conv is $7^2 \cdot 3 \cdot 64 \cdot 112^2 \approx 1.2 \cdot 10^8$ ops — nearly 1000× cheaper.

<div class="insight">

Parameters are *shared* across positions, and each output depends on a small region. That's what makes the conv *orders of magnitude* cheaper than an equivalent MLP — not a marginal win.

</div>

---

# The four hyperparameters, in one picture

For every conv layer, think in this order:

1. **kernel size** — the *field of view* (3 for modern nets, 7 only at the stem).
2. **stride** — how fast the filter steps (1 to preserve, 2 to halve).
3. **padding** — how much zero-boundary to add (so output size matches or shrinks predictably).
4. **dilation** — gap between kernel taps (used in segmentation to widen RF without more params).

<div class="keypoint">

99% of production CNNs only use (1, 2). Segmentation (L9) uses (4). Advanced audio / video sometimes uses (4). For images, reach for `kernel_size=3, padding=1, stride=1 or 2` — nearly every other choice is a specific research idea.

</div>

---

# Max-pool · worked numeric example

<div class="math-box">

Input · 4 × 4 matrix
$$\begin{bmatrix} 1 & 2 & 8 & 3 \\ 4 & 6 & 5 & 1 \\ 9 & 7 & 2 & 3 \\ 5 & 3 & 1 & 0 \end{bmatrix}$$

2×2 max-pool, stride 2 · slide a 2×2 window non-overlapping.

| window | values | max |
|:-:|:-:|:-:|
| top-left | 1, 2, 4, 6 | **6** |
| top-right | 8, 3, 5, 1 | **8** |
| bottom-left | 9, 7, 5, 3 | **9** |
| bottom-right | 2, 3, 1, 0 | **3** |

Output · $\begin{bmatrix} 6 & 8 \\ 9 & 3 \end{bmatrix}$

</div>

Halved spatial size · kept the strongest activation per region · gained translation invariance.

---

# Pooling · the other downsample

$$\text{MaxPool}(x) = \max_{i, j \in \text{window}}\, x_{ij}$$

- **Max pooling** keeps the strongest activation → small translation invariance.
- **Average pooling** smooths — used in the global-average-pooling head for modern architectures.

<div class="insight">

Convolution is **translation equivariant** (shift input → shift output).
Pooling adds **translation invariance** (shift input → same output).

</div>

In 2026, stride-2 convolutions often replace pooling entirely (avoid info loss).

---

# Why pooling works · the invariance argument

Imagine a cat in the corner of an image vs centered. Should the network's answer depend on *where* the cat is?

- For classification · **no** (it's still a cat).
- For segmentation · **yes** (we need the pixels back).

<div class="math-box">

Pooling creates a small *"I don't care exactly where"* window. Stacks of pool+conv compound this: by layer 5, the network cares about the cat's presence, not its exact pixel position.

</div>

This is the **invariance gradient** a classification CNN builds — conv1 nearly equivariant (tells you where), final global pool fully invariant (tells you what).

---

<!-- _class: section-divider -->

### PART 2

# Receptive field

How deep features "see" far-away pixels

---

# RF grows with depth

![w:920px](figures/lec07/svg/receptive_field.svg)

<div class="realworld">

▶ Interactive: drag depth/kernel/stride and watch RF grow live — [receptive-field-grower](https://nipunbatra.github.io/interactive-articles/receptive-field-grower/).

</div>

---

# Receptive field grows with depth · picture

![w:920px](figures/lec07/svg/conv_receptive_field_grow.svg)

---

# The VGG insight · stack 3×3, not 7×7

**Claim.** Three stacked $3 \times 3$ convs have the same receptive field as one $7 \times 7$ conv — but with fewer parameters and *more non-linearities*.

<div class="math-box">

| | 1× (7 × 7) | 3× (3 × 3) stacked |
|--|-----------|--------------------|
| RF | 7 | 7 |
| Params (C → C) | $49 C^2$ | $27 C^2$ |
| Non-linearities | 1 ReLU | 3 ReLUs |

</div>

Fewer params + more non-linearities → richer function class at lower cost.

**VGG (2014)** built its whole architecture around this observation.

---

# Worked RF calculation · a 5-layer CNN

Imagine five conv layers, all $3 \times 3$, stride 1. Per layer, RF grows by $(K - 1) = 2$.

| Layer | Receptive field | What it "sees" |
|:-:|:-:|:-:|
| Input | 1 | single pixel |
| Conv1 | 3 | tiny patch |
| Conv2 | 5 | edge-length corner |
| Conv3 | 7 | small feature |
| Conv4 | 9 | object part |
| Conv5 | 11 | object |

Add stride-2 anywhere and the downstream RF **doubles** per stride. A ResNet-50 has RF ≈ 500 in the final block — it can see the whole image.

<div class="insight">

Next time you read an architecture diagram, compute RF mentally: deeper nets = larger RF = more *context* per pixel. This is the capacity that lets CNNs understand objects, not just textures.

</div>

---

# Effective receptive field

<div class="insight">

Theoretical RF grows linearly with depth.
**Effective** RF (Luo et al. 2016) is more **Gaussian-shaped** — pixels near the centre contribute much more than edge pixels.

</div>

Two consequences:

1. Modern architectures often need **dilated convs** or **attention** to get truly global RF.
2. CNNs and Vision Transformers differ here — ViT has uniform global RF from layer 1 (coming in L18).

---

<!-- _class: section-divider -->

### PART 3

# Classic architecture evolution

LeNet → AlexNet → VGG → Inception → ResNet → MobileNet → EfficientNet

---

# The progression

![w:920px](figures/lec07/svg/architecture_timeline.svg)

---

# What each era got right

| Year | Model | Key contribution |
|------|-------|------------------|
| 1998 | **LeNet-5** | first successful CNN (digits, MNIST) |
| 2012 | **AlexNet** | GPU training, ReLU, dropout, big data (ImageNet) |
| 2014 | **VGG** | *"depth with small kernels"* — 3×3 only, stacked |
| 2014 | **GoogLeNet** | 1×1 bottlenecks, parallel branches (Inception) |
| 2015 | **ResNet** | skip connections → 152 layers trainable |
| 2017 | **MobileNet** | depthwise separable convs (edge devices) |
| 2019 | **EfficientNet** | compound scaling (depth × width × resolution) |

---

# 1×1 conv · the recipe-mixer

<div class="keypoint">

A 1×1 convolution sounds useless · it only sees one pixel! But the power is in the **depth dimension**.

At each pixel · you have 256 channel values (your "ingredients"). The 1×1 conv learns the best **recipes** to mix them down into 64 new "flavors" (output channels).

</div>

It's an extremely cheap way to · reduce channels (bottleneck) · expand them after a 3×3 · or remix a feature map without spatial mixing.

Used everywhere in modern CNNs and Transformers (the "output projection" of attention is a 1×1 conv applied to the channel axis).

---

# 1×1 convolutions · the unsung hero

A $1 \times 1$ convolution looks trivial — but it's **channel mixing**.

Takes $C_\text{in}$ channels at each spatial location, produces $C_\text{out}$ channels:

$$y_{i,j,c'} = \sum_{c=1}^{C_\text{in}} w_{c,c'}\, x_{i,j,c}$$

**Uses everywhere in modern networks:**
- Reduce channels before expensive $3 \times 3$ conv (GoogLeNet bottleneck)
- Expand channels after (ResNet bottleneck: $1 \times 1 \to 3 \times 3 \to 1 \times 1$)
- As the "attention output projection" in Transformers

---

# 1×1 conv · worked example

Input tensor `(256, 14, 14)` — 256 channels at 14×14 spatial resolution.
Apply `Conv2d(256, 64, kernel_size=1)` → `(64, 14, 14)`.

<div class="math-box">

- Parameters: $256 \cdot 64 = 16{,}384$ (plus bias).
- FLOPs: $256 \cdot 64 \cdot 14 \cdot 14 \approx 3.2 \text{M}$ per forward pass.

</div>

What did we just do? Took **256 channels at each spatial position, mixed them linearly to 64 channels.** Spatial structure preserved. Channel structure compressed.

A 3×3 conv immediately after now runs 4× cheaper because the depth is 4× smaller. That's the **bottleneck trick**: sandwich 3×3 convs between 1×1 compressions.

---

# AlexNet → VGG · the "just add depth" years

Between 2012 and 2014, the field converged on a recipe:

1. Keep convolution kernels **small** (3×3) and **many**.
2. **Stack deeper** until you run out of memory or accuracy plateaus.
3. Use **ReLU + dropout + batch-norm** to make deeper nets trainable.

<div class="insight">

By 2015, people tried to go deeper than 25 layers and networks **stopped learning**. Adding layers made *training* loss worse — not a generalization issue. This pointed at the *optimization* problem that ResNet (next lecture) solved with skip connections.

</div>

---

<!-- _class: section-divider -->

### PART 4

# Inductive biases

Why convolution beats MLP on images

---

# Three biases baked into convolution

![w:920px](figures/lec07/svg/inductive_bias.svg)

---

# In words · what each bias does

<div class="columns">
<div>

### 1. Locality

Each output sees only a small window. Forces the network to first extract local features (edges, textures) before combining them.

**Why correct for images** · nearby pixels are semantically related (same edge, same object).

</div>
<div>

### 2. Translation equivariance

Shift input → shift output. The same feature detector runs at every position.

**Why correct for images** · a cat is a cat whether it's in the corner or centre.

</div>
</div>

### 3. Hierarchy of scales

Stacking convs builds larger receptive fields. Deep networks *compose* features at each scale.

**Why correct for images** · visual world is hierarchical (edge → texture → part → object).

---

---

# What inductive bias buys you

An MLP on a 224×224 image: 150k inputs × 4k hidden units = **600M parameters** for *one layer*.

A CNN with a 3×3 kernel and 64 channels: **576 parameters** for *one layer* (regardless of image size).

<div class="keypoint">

The inductive bias is the prior. With the right prior you need less data and less capacity. With the wrong prior (e.g., MLP on images) you need both in huge quantity.

</div>

<div class="insight">

Vision Transformers (L18) give up most of this inductive bias — they need pretraining on far more data to compensate.

</div>

---

# Inductive bias · the data-efficiency plot

<div class="columns">
<div>

### Small data (≤ 10⁴ images)

- **CNN wins** — the prior does the heavy lifting.
- MLP barely learns anything; ViT is worse than CNN.

</div>
<div>

### Huge data (≥ 10⁸ images)

- **ViT matches or beats CNN** — enough data to overcome the missing prior.
- The "ImageNet-21k threshold" from Dosovitskiy 2020.

</div>
</div>

<div class="keypoint">

**The bias is a free data multiplier.** A CNN at 50k images behaves like a ViT at 500k. If your dataset is small, use a CNN (or start from a pretrained CNN).

</div>

---

# A CNN block in PyTorch

```python
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
```

Every classic CNN is a stack of these (plus pooling or stride-2 for downsampling). VGG is literally this block repeated.

---

# Feature visualization · what each layer learns

| Layer | Typical features (for a trained CNN on natural images) |
|-------|--------------------------------------------------------|
| Conv1 | oriented edges, colour blobs (~like Gabor filters) |
| Conv2 | junctions, simple textures |
| Conv3 | repeated patterns (fur, grid, stripes) |
| Conv4 | object parts (eyes, wheels) |
| Conv5 | whole objects / object arrangements |

<div class="paper">

Zeiler &amp; Fergus 2014 · *Visualizing and Understanding Convolutional Networks* — canonical reference for layer visualization.

</div>

---

# What's next

<div class="insight">

L7 covered the "classic CNN" era — the 1998–2014 progression that ended with VGG.

**L8 (next lecture)** picks up where we paused: Inception modules, ResNet in CNNs, MobileNet's depthwise separable convs, EfficientNet scaling, and transfer learning — the one practical skill you'll use most often.

</div>

---

# Common questions · FAQ

**Q. What's a typical kernel size in 2026?**
A. 3 everywhere, except stem layer uses 7×7 (more RF for first layer). Very occasionally 5×5 for specific blocks.

**Q. When do I need big-kernel convs (ConvNeXt 7×7)?**
A. When replicating Transformer-style long-range mixing in CNNs. ConvNeXt showed 7×7 depthwise conv can match attention on some tasks.

**Q. How do I choose number of channels?**
A. Double channels every time you halve spatial (32→64→128→256→512). Keeps params-per-layer roughly constant. VGG, ResNet, EfficientNet all follow this.

**Q. Padding · 'same' vs 'valid'?**
A. `padding='same'` (PyTorch) keeps output size = input size. Default for most blocks. `'valid'` (no padding) shrinks · used when downsampling is the point.

---

<!-- _class: summary-slide -->

# Lecture 7 — summary

- **Convolution** = sliding window with shared weights · translation-equivariant · 3 biases (sparse, shared, local).
- **Output size** · $O = (W − K + 2P)/S + 1$; pad to preserve, stride to downsample.
- **Receptive field** grows with depth — 3 stacked $3 \times 3$ convs ≈ one $7 \times 7$ with fewer params.
- **Stacked 3×3 (VGG)** beat single $7 \times 7$ — fewer params, more non-linearities.
- **1×1 convs** mix channels — they're everywhere.
- **LeNet → AlexNet → VGG** is the classic era; ResNet (next) finally solved depth.

### Read before Lecture 8

**Prince** — Ch 10 (advanced), Ch 11 (residual in CNNs).

### Next lecture

**Modern CNNs + Transfer Learning** — GoogLeNet bottlenecks, ResNet-in-CNN, MobileNet, EfficientNet, fine-tuning pretrained backbones.

<div class="notebook">

**Notebook 7** · `07-cnn-from-scratch.ipynb` — build a VGG-style mini-CNN for CIFAR-10; print tensor shapes at each layer; compute receptive field per layer.

</div>
