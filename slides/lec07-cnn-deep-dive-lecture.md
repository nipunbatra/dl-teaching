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

<!-- _class: section-divider -->

### PART 2

# Receptive field

How deep features "see" far-away pixels

---

# RF grows with depth

![w:920px](figures/lec07/svg/receptive_field.svg)

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

# 1×1 convolutions · the unsung hero

A $1 \times 1$ convolution looks trivial — but it's **channel mixing**.

Takes $C_\text{in}$ channels at each spatial location, produces $C_\text{out}$ channels:

$$y_{i,j,c'} = \sum_{c=1}^{C_\text{in}} w_{c,c'}\, x_{i,j,c}$$

**Uses everywhere in modern networks:**
- Reduce channels before expensive $3 \times 3$ conv (GoogLeNet bottleneck)
- Expand channels after (ResNet bottleneck: $1 \times 1 \to 3 \times 3 \to 1 \times 1$)
- As the "attention output projection" in Transformers

---

<!-- _class: section-divider -->

### PART 4

# Inductive biases

Why convolution beats MLP on images

---

# Three biases baked into convolution

![w:920px](figures/lec07/svg/inductive_bias.svg)

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
