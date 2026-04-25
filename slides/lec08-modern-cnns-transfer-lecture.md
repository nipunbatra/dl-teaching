---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Modern CNNs &amp; Transfer Learning

## Lecture 8 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. Explain the **Inception** multi-branch idea and 1×1 bottlenecks.
2. Describe **ResNet basic / bottleneck** blocks with skip connections.
3. Apply **depthwise separable convolutions** (MobileNet) for efficiency.
4. Use **EfficientNet's compound scaling** rule.
5. Execute the **three transfer-learning recipes** correctly.
6. Pick a backbone from `timm` for a given dataset size.

---

# Recap · where we are

- **Classic CNN era** (L7): LeNet, AlexNet, VGG — stacked 3×3 convs work.
- **Receptive field** grows with depth.
- **Inductive biases** — sparse connectivity, weight sharing, equivariance.

<div class="paper">

Today maps to **UDL Ch 10** (advanced sections) and **Ch 11** (residual / skip connections in CNNs).

</div>

Two halves today:
1. The architectures that came after VGG — Inception, ResNet, MobileNet, EfficientNet.
2. Transfer learning — the single most practical CNN skill.

---

<!-- _class: section-divider -->

### PART 1

# Inception · parallel kernels

Let SGD pick the right receptive field

---

# Inception · the buffet idea

<div class="keypoint">

VGG taught us · stacking 3×3 convs is powerful. But every layer has the **same fixed view** of the world.

What if some features are tiny (need 1×1) and others are large (need 5×5)?

Inception's answer · **don't pick** · offer a buffet of kernel sizes (1×1, 3×3, 5×5, pool) at every layer · let SGD pick the right ratio.

</div>

The buffet is concatenated along the channel axis · the next layer sees all kernel sizes' outputs simultaneously.

---

# The Inception module (Szegedy 2014)

![w:920px](figures/lec08/svg/inception_module.svg)

---

# Why 1×1 convolutions matter

Two jobs:

1. **Channel mixing** · $C_\text{in}$ → $C_\text{out}$ at every spatial location.
2. **Dimensionality reduction** · squeeze before an expensive 3×3 or 5×5.

<div class="math-box">

Cost of a direct 3×3 with 256 → 256 channels: $3 \cdot 3 \cdot 256 \cdot 256 = 589{,}824$ params.

With a 1×1 bottleneck to 64 first:
$256 \cdot 64 + 3 \cdot 3 \cdot 64 \cdot 64 + 64 \cdot 256 = 69{,}632$ params · **8.5× cheaper**.

</div>

1×1 convs are in every modern architecture — as reduction in GoogLeNet, as bottlenecks in ResNet, as the FFN in Transformers.

---

<!-- _class: section-divider -->

### PART 2

# ResNet in CNNs

Skip connections · bottleneck blocks

---

# Why skip connections work · the gradient argument

Ordinary layer · $h_{l+1} = F(h_l)$. Gradient must pass through $F$'s Jacobian at every level. 152 layers of that → product of 152 Jacobians → vanishing or exploding.

Residual layer · $h_{l+1} = h_l + F(h_l)$. Gradient:
$$\frac{\partial \mathcal{L}}{\partial h_l} = \frac{\partial \mathcal{L}}{\partial h_{l+1}} \cdot \left(I + \frac{\partial F}{\partial h_l}\right)$$

<div class="keypoint">

The identity path sends the gradient through **unchanged**. Deep gradients stop vanishing at construction time, not through training luck. This is why 152 layers is easy but 34 plain layers was impossible.

</div>

---

# Why skip connections also help forward pass

When a new block isn't useful yet, residual = 0 is the easiest thing to learn — the identity mapping just passes $h_l$ through.

<div class="insight">

**Adding blocks can only improve or at worst no-op**, never hurt. Before ResNet, adding layers to a working network often made it worse (degradation problem). After ResNet, deeper ≥ shallower — you just keep adding.

</div>

Same idea shows up as LSTM's cell state (L10) and Transformer's residual stream (L13). Skip connections are the single most load-bearing design in modern deep learning.

---

# The ResNet-CNN block

![w:920px](figures/lec08/svg/resnet_bottleneck.svg)

---

# Basic vs bottleneck · annotated

![w:920px](figures/lec08/svg/resnet_skip_closeup.svg)

---

# Projection shortcuts · when dimensions change

When you need to change the number of channels or downsample (stride 2), identity can't match shapes directly.

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{W}_s\, \mathbf{x}$$

$\mathbf{W}_s$ is a learned **1×1 convolution with the same stride** as the main branch. Everything else stays residual.

```python
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, c_in, c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c, 1)
        self.conv2 = nn.Conv2d(c, c, 3, stride, padding=1)
        self.conv3 = nn.Conv2d(c, c * self.expansion, 1)
        self.bn1, self.bn2, self.bn3 = [nn.BatchNorm2d(x) for x in (c, c, c*self.expansion)]
        self.shortcut = nn.Conv2d(c_in, c*self.expansion, 1, stride) if stride != 1 or c_in != c*self.expansion else nn.Identity()
```

---

# The ResNet family by depth

| Model | Depth | Params | ImageNet top-1 |
|-------|-------|--------|----------------|
| ResNet-18 | 18 | 12 M | 69.8% |
| ResNet-34 | 34 | 22 M | 73.3% |
| ResNet-50 | 50 | 26 M | 76.1% |
| ResNet-101 | 101 | 45 M | 77.4% |
| ResNet-152 | 152 | 60 M | 78.3% |

<div class="realworld">

ResNet-50 is the workhorse. Unless you have a specific reason, start there for any CNN task you face in 2026.

</div>

---

<!-- _class: section-divider -->

### PART 3

# MobileNet · efficient CNNs

Depthwise separable convolutions

---

# Depthwise separable · split the work

![w:920px](figures/lec08/svg/depthwise_separable.svg)

---

# Why depthwise separable works

A standard 3×3 convolution does two things at once:

1. **Spatial mixing** — combine a $3 \times 3$ neighbourhood.
2. **Channel mixing** — combine information across input channels.

Depthwise separable splits these:

- **Depthwise (3×3 per channel)** — only spatial mixing.
- **Pointwise (1×1)** — only channel mixing.

<div class="math-box">

Cost · $D_K^2 \cdot C + C^2$   vs   standard · $D_K^2 \cdot C^2$.

For $C = 128, D_K = 3$: $17{,}024$ params vs $147{,}456$ params. **~9× cheaper.**

</div>

Accuracy drop: ~1%. Speed-up: ~8–10×. Ships in every mobile / edge model since 2017.

---

# Why edge models care about FLOPs

A phone has ~10 W total power; a datacenter GPU draws 300 W just to idle. Efficient nets let you run:

| Model | FLOPs | iPhone inference |
|:-:|:-:|:-:|
| ResNet-50 | 4.1 G | ~90 ms |
| MobileNetV2 | 0.3 G | ~15 ms |
| MobileNetV3-S | 0.06 G | ~4 ms |

<div class="keypoint">

Real-time on-device tasks (camera AR, live caption, wake-word) need ≤30 ms budget. MobileNet-style splits are the reason such apps exist.

</div>

---

# MobileNet variants · a decade of improvements

| Model | Year | Key idea |
|-------|------|----------|
| MobileNet v1 | 2017 | Depthwise separable + width multiplier |
| MobileNet v2 | 2018 | Inverted residuals + linear bottlenecks |
| MobileNet v3 | 2019 | Neural-architecture-search + SiLU |
| EfficientNet-B0 | 2019 | Compound scaling foundation |

---

<!-- _class: section-divider -->

### PART 4

# EfficientNet · compound scaling

Scale depth, width, and resolution together

---

# Compound scaling · tuning a car engine

<div class="keypoint">

You can scale up an engine by · making it **bigger** (depth) · using **wider** pistons (width) · running on **higher-octane** fuel (resolution).

Any one alone helps a bit. Doing all three together · in balance · gives the best engine.

</div>

EfficientNet's insight · the same is true of neural networks. Depth, width, and input resolution should grow *together* for a given compute budget · not one at a time.

---

# The compound-scaling principle

Previously, researchers scaled up nets by picking ONE dimension:

- VGG-11 → VGG-19 · **depth** only
- WideResNet · **width** only
- ProGAN / high-res nets · **resolution** only

**Tan &amp; Le 2019** (EfficientNet) showed you should scale all three *together* under a fixed budget.

<div class="math-box">

$$\text{depth } d = \alpha^\phi \quad \text{width } w = \beta^\phi \quad \text{resolution } r = \gamma^\phi$$

subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$.

Choose $\phi$ as your compute budget; $\alpha, \beta, \gamma$ are tuned by grid search once.

</div>

---

# EfficientNet scale-up at a glance

| Model | Depth | Width | Res | Params | ImageNet top-1 |
|-------|-------|-------|-----|--------|----------------|
| B0 | 1.0 | 1.0 | 224 | 5.3 M | 77.3% |
| B1 | 1.1 | 1.0 | 240 | 7.8 M | 79.2% |
| B3 | 1.4 | 1.2 | 300 | 12 M | 81.6% |
| B5 | 1.6 | 1.6 | 456 | 30 M | 83.6% |
| B7 | 2.0 | 2.0 | 600 | 66 M | 84.3% |

<div class="realworld">

EfficientNet set the accuracy/param Pareto frontier for 2019–2021 before Vision Transformers took over.

</div>

---

<!-- _class: section-divider -->

### PART 5

# Transfer learning

The skill you'll use 90% of the time

---

# The premise

## What transfer learning *is*, in one line

Start from a model that already knows something relevant, adapt it to your new task with a fraction of the data and compute.

<div class="keypoint">

The pretrained network is a **learned prior**. You're not training from random init — you're fine-tuning a near-working initialization. In most vision tasks this is worth ≈ 1–2 orders of magnitude of training data.

</div>

---

# The premise

ImageNet pretraining gives you a **generic vision stack**:

- Early layers detect edges, textures — *universal*, works for any image domain.
- Mid layers detect parts — *mostly transferable* across domains.
- Late layers detect ImageNet-specific object categories — *domain-specific*, usually replaced.

<div class="keypoint">

**Transfer learning rule** — more data in your new domain → unfreeze more layers.

</div>

---

# Three recipes

![w:920px](figures/lec08/svg/transfer_learning.svg)

---

# Transfer learning · by data size

<div class="math-box">

| Your data | Recommended recipe | What to freeze |
|:-:|:-:|:-:|
| < 100 labels | Linear probe | everything except head |
| 100 - 1k | Fine-tune top layers | freeze conv1-3 |
| 1k - 10k | Fine-tune whole backbone | nothing (with discriminative LR) |
| 10k - 100k | Fine-tune + LR scheduling | nothing, bigger LR |
| > 100k | Probably train from scratch | — |

</div>

<div class="insight">

Smaller data → more frozen. Larger data → more trainable. If you have 1M labels in your domain, you likely don't need transfer at all (but it rarely hurts as a warm start).

</div>

---

# The bitter truth · negative transfer

<div class="warning">

Sometimes transfer learning **hurts**. Medical MRI → ImageNet-pretrained ResNet. Natural photos teach features that don't transfer to grayscale medical modalities.

</div>

Symptoms:
- Val loss starts higher with pretraining than from scratch
- Fine-tuned performance plateaus
- Early layers still detect ImageNet-like edges; later layers never adapt

Fix · try from-scratch, or use **domain-specific pretraining** (RadImageNet for medical, SatCLIP for satellite). Generic features aren't universal.

---

# Discriminative (layer-wise) learning rates

When you do unfreeze early layers, they should learn more *slowly* than late layers — early layers are already good.

```python
# PyTorch — different LR per param group
params = [
    {"params": model.conv1.parameters(), "lr": 1e-5},  # early   · slow
    {"params": model.conv2.parameters(), "lr": 1e-5},
    {"params": model.conv3.parameters(), "lr": 3e-5},
    {"params": model.conv4.parameters(), "lr": 1e-4},
    {"params": model.conv5.parameters(), "lr": 3e-4},  # late    · fast
    {"params": model.fc.parameters(),    "lr": 1e-3},  # new     · fastest
]
opt = torch.optim.AdamW(params, weight_decay=0.01)
```

<div class="realworld">

fastai popularized "1cycle + discriminative LRs" for transfer learning — often the right defaults for small-data fine-tuning.

</div>

---

# When transfer learning fails

<div class="warning">

**Large domain gap.** ImageNet (natural photos) → medical X-rays, satellite imagery, microscopy. Early-layer features may still transfer; late-layer features definitely won't.

**Very different image sizes.** ImageNet is 224×224; medical imaging may be 1024+. You may need to resize or re-pretrain.

**Very small target dataset** (≪ 100 examples). Even linear probing won't save you — consider self-supervised pretraining on your domain first (L17).

</div>

In those cases — pre-train on a closer domain, or use self-supervised methods (coming in L17).

---

# Pick-the-right-backbone table

| Scenario | Backbone | Why |
|:-:|:-:|:-:|
| ImageNet-like natural photos | ResNet-50 or ConvNeXt | strong + well-supported |
| Very small labeled data | CLIP ViT features | cross-domain generalization |
| Edge / real-time | MobileNet-V3 | latency budget |
| High-res medical / satellite | ConvNeXt-large or DINOv2 | captures fine detail |
| Arbitrary RGB, unknown domain | DINOv2 frozen features | best general-purpose SSL |

<div class="realworld">

In 2026, **DINOv2** (self-supervised on 142M images) is often the starting point for vision-backbone features — even better than ImageNet-supervised ResNets for downstream tasks.

</div>

---

# Loading a pretrained backbone · PyTorch

```python
import torch, torch.nn as nn
import torchvision.models as M

# 1. Load pretrained weights
model = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2)

# 2. Freeze everything
for p in model.parameters():
    p.requires_grad = False

# 3. Replace the 1000-way classifier with N-way
n_classes = 10
model.fc = nn.Linear(model.fc.in_features, n_classes)   # new, trainable by default

# 4. Train only the new fc
opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)

# 5. Later, unfreeze progressively
for p in model.layer4.parameters(): p.requires_grad = True
```

---

# The `timm` ecosystem

For 2026, stop hand-rolling architectures:

```python
import timm

# 500+ pretrained vision models in one line
model = timm.create_model('resnet50',        pretrained=True, num_classes=10)
model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=10)
model = timm.create_model('convnext_base',   pretrained=True, num_classes=10)
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
```

<div class="realworld">

`timm` by Ross Wightman · the de facto vision model zoo. Every competitive Kaggle vision solution uses it.

</div>

---

<!-- _class: summary-slide -->

# Lecture 8 — summary

- **Inception module** — parallel branches of 1×1, 3×3, 5×5, pool; let SGD pick the receptive field.
- **1×1 convolutions** — channel mixing + cheap bottlenecks; in every modern architecture.
- **ResNet bottleneck** — 1×1 → 3×3 → 1×1 + skip; 17× fewer params per block than basic residual.
- **Depthwise separable** — split spatial from channel mixing; MobileNet's foundation.
- **Compound scaling** (EfficientNet) — scale depth × width × resolution together.
- **Transfer learning** recipes — feature-extract · fine-tune top · fine-tune all. Match to data size.
- **Practically** · start from `timm.create_model('resnet50', pretrained=True)` and go from there.

### Read before Lecture 9

Bishop Ch 10 + CS231n OD notes (UDL doesn't cover detection/segmentation).

### Next lecture

**Detection &amp; Segmentation** — R-CNN → Faster R-CNN, YOLO, IoU, NMS, U-Net, Mask R-CNN, zero-shot SAM.

<div class="notebook">

**Notebook 8** · `08-transfer-learning.ipynb` — fine-tune ResNet-50 on Flowers-102 with discriminative LRs, measure effect of freezing depth.

</div>
