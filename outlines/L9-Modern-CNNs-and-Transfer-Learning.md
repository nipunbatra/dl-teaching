---
title: "Lecture 9: Modern CNN Pipelines and Transfer Learning"
source: "https://chatgpt.com/share/6a50728a-9720-83ee-a33d-5f2e3ff744fe"
status: "Imported from the finalized shared-course discussion"
---

# Lecture 9: Modern CNN Pipelines and Transfer Learning

Core story:

\[
\boxed{
\text{CNN basics become useful backbones through blocks, scaling, residuals and transfer.}
}
\]

VGG showed the power of stacking small \(3\times3\) convolutions; ResNet made very deep CNNs trainable through residual learning; Inception/MobileNet/EfficientNet focus on efficient computation; ConvNeXt modernizes CNNs using design ideas that compete with Transformer-era vision backbones.

---

# Slide-level outline

## Part I — Why another CNN lecture?

### 1. Title  
**Modern CNN Pipelines**

> VGG, \(1\times1\) convs, ResNet, efficient CNNs, transfer learning

---

### 2. Recall: CNN primitive

\[
X\in\mathbb R^{H\times W\times C}
\]

\[
Y=\operatorname{Conv}(X;W)
\]

Basic block:

\[
\text{Conv}\rightarrow\text{Norm}\rightarrow\text{Activation}
\]

---

### 3. Today’s question

CNN basics answer:

\[
\text{How does convolution work?}
\]

Today answers:

\[
\text{How do we design CNN systems that actually work?}
\]

---

## Part II — VGG: simple depth

### 4. VGG design principle

Use many small convolutions:

\[
3\times3,\quad s=1,\quad p=1.
\]

Stack them repeatedly.

VGG evaluated depth systematically and showed strong results with 16–19 weight layers using small \(3\times3\) filters.

---

### 5. Why \(3\times3\)?

Two \(3\times3\) convs give effective receptive field:

\[
5\times5.
\]

Three \(3\times3\) convs give:

\[
7\times7.
\]

But with more nonlinearities.

---

### 6. Parameter comparison

Assume same channels \(C\).

One \(5\times5\):

\[
25C^2
\]

Two \(3\times3\):

\[
2\cdot9C^2=18C^2
\]

So:

\[
18C^2<25C^2.
\]

---

### 7. VGG-style stage

```text
Conv 3x3 → ReLU
Conv 3x3 → ReLU
MaxPool 2x2
```

Then double channels:

\[
64\rightarrow128\rightarrow256\rightarrow512.
\]

---

### 8. VGG lesson

\[
\boxed{
\text{Depth + small kernels + simple repeated blocks can work very well.}
}
\]

But VGG is parameter-heavy, especially due to fully connected heads.

---

## Part III — \(1\times1\) convolution

### 9. What is a \(1\times1\) convolution?

For each spatial location \((h,w)\):

\[
y_{hw}=Wx_{hw}+b.
\]

It mixes channels, not spatial neighbours.

\[
x_{hw}\in\mathbb R^{C_{\text{in}}}
\]

\[
y_{hw}\in\mathbb R^{C_{\text{out}}}
\]

---

### 10. \(1\times1\) conv as per-pixel MLP

At every pixel/location:

\[
\mathbb R^{C_{\text{in}}}\rightarrow\mathbb R^{C_{\text{out}}}.
\]

Same weights shared over all spatial positions.

Use cases:

- change channel dimension;
- bottleneck;
- feature mixing;
- projection in residual blocks;
- segmentation heads.

---

### 11. Worked \(1\times1\) parameter count

Input:

\[
H\times W\times256.
\]

Output:

\[
H\times W\times64.
\]

Parameters:

\[
1\cdot1\cdot256\cdot64+64
=
16{,}448.
\]

A \(3\times3\) conv would use:

\[
3\cdot3\cdot256\cdot64+64
=
147{,}520.
\]

---

### 12. Bottleneck idea

Instead of:

\[
3\times3: 256\rightarrow256
\]

use:

\[
1\times1:256\rightarrow64
\]

\[
3\times3:64\rightarrow64
\]

\[
1\times1:64\rightarrow256
\]

This reduces computation while preserving representational power.

---

## Part IV — Inception: multi-scale and factorized computation

### 13. Inception motivation

Objects/features appear at multiple scales.

So process in parallel:

```text
1x1 branch
3x3 branch
5x5 branch
pool branch
concat channels
```

Inception explored computationally efficient scaling using factorized convolutions and regularization.

---

### 14. Inception module diagram

```text
           ┌→ 1x1 ──────────┐
input ─────┼→ 1x1 → 3x3 ────┼→ concat
           ├→ 1x1 → 5x5 ────┤
           └→ pool → 1x1 ───┘
```

---

### 15. Why \(1\times1\) before expensive convs?

Dimensionality reduction.

Example:

Before \(5\times5\):

\[
C=256.
\]

Direct:

\[
5\cdot5\cdot256\cdot128
=
819{,}200.
\]

Reduce first:

\[
1\times1:256\rightarrow32
\]

\[
5\times5:32\rightarrow128
\]

Cost:

\[
256\cdot32+25\cdot32\cdot128
=
8{,}192+102{,}400
=
110{,}592.
\]

Huge reduction.

---

### 16. Inception lesson

\[
\boxed{
\text{Use computation where it matters; reduce channels before expensive operations.}
}
\]

---

## Part V — ResNet: residual learning as default backbone

### 17. Plain deep networks degrade

More layers should not hurt training error in principle.

But very deep plain networks can be hard to optimize.

ResNet reformulated layers as residual functions and showed that residual networks are easier to optimize and can benefit from greatly increased depth.

---

### 18. Residual block

\[
x_{\ell+1}=x_\ell+F_\ell(x_\ell).
\]

Diagram:

```text
x ─→ conv → norm → ReLU → conv → norm ─→ + → ReLU
│                                           ↑
└──────────────── identity ─────────────────┘
```

---

### 19. Gradient through residual block

\[
\frac{\partial L}{\partial x_\ell}
=
\frac{\partial L}{\partial x_{\ell+1}}
\left(
I+
\frac{\partial F_\ell}{\partial x_\ell}
\right).
\]

There is a direct identity path:

\[
\frac{\partial L}{\partial x_{\ell+1}}.
\]

---

### 20. Basic block vs bottleneck block

Basic block:

```text
3x3 → 3x3
```

Bottleneck block:

```text
1x1 reduce → 3x3 process → 1x1 expand
```

Used in deeper ResNets.

---

### 21. Projection shortcut

If shape changes:

\[
x\in\mathbb R^{H\times W\times C_1}
\]

\[
F(x)\in\mathbb R^{H/2\times W/2\times C_2}
\]

identity cannot be directly added.

Use:

\[
P(x)=1\times1\text{ conv with stride }2.
\]

Then:

\[
x_{\ell+1}=P(x_\ell)+F(x_\ell).
\]

---

### 22. ResNet as backbone

A ResNet backbone gives feature maps at multiple resolutions:

\[
C2,\ C3,\ C4,\ C5.
\]

These become inputs to:

- classification head;
- detection head;
- segmentation decoder;
- feature pyramid network.

---

## Part VI — Efficient CNNs

### 23. Standard convolution cost

\[
k^2C_{\text{in}}C_{\text{out}}HW.
\]

This becomes expensive for mobile/edge devices.

---

### 24. Depthwise separable convolution

Split standard conv into:

1. depthwise spatial conv;
2. pointwise \(1\times1\) conv.

Depthwise:

\[
k^2C_{\text{in}}HW.
\]

Pointwise:

\[
C_{\text{in}}C_{\text{out}}HW.
\]

MobileNet is built around depthwise separable convolutions for efficient mobile and embedded vision.

---

### 25. Worked cost comparison

Let:

\[
k=3,\quad C_{\text{in}}=C_{\text{out}}=128.
\]

Standard conv per location:

\[
3^2\cdot128\cdot128
=
147{,}456.
\]

Depthwise separable:

\[
3^2\cdot128+128\cdot128
=
1{,}152+16{,}384
=
17{,}536.
\]

Reduction:

\[
\approx8.4\times.
\]

---

### 26. EfficientNet: scaling rule

Instead of scaling only depth or width:

\[
d,\quad w,\quad r
\]

scale all three:

- depth;
- width;
- input resolution.

EfficientNet proposed compound scaling to balance depth, width and resolution.

---

### 27. ConvNeXt: modern CNN design

ConvNeXt revisits ResNet and modernizes design choices inspired partly by Transformer-era architectures while remaining a pure ConvNet.

Mention only at high level:

- larger kernels;
- inverted bottlenecks;
- fewer activation/norm placements;
- LayerNorm-style choices;
- modern training recipe.

---

### 28. Efficiency axes

| Architecture idea | Saves/improves |
|---|---|
| \(1\times1\) bottleneck | parameters/FLOPs |
| depthwise separable conv | FLOPs |
| global average pooling | parameters |
| residuals | trainability |
| compound scaling | accuracy/FLOP tradeoff |
| transfer learning | data efficiency |

---

## Part VII — Classification pipeline

### 29. Modern classification pipeline

```text
Dataset
→ transforms/augmentation
→ pretrained backbone
→ classifier head
→ loss
→ optimizer
→ scheduler
→ validation
→ checkpoint
```

---

### 30. Image preprocessing

Typical:

\[
x'=\frac{x-\mu}{\sigma}.
\]

For pretrained ImageNet models, use the expected normalization.

Wrong normalization can badly hurt transfer.

---

### 31. Classification head

Backbone output:

\[
F\in\mathbb R^{H'\times W'\times C}.
\]

Global average pooling:

\[
h_c=\frac1{H'W'}\sum_{i,j}F_{ijc}.
\]

Linear classifier:

\[
z=Wh+b.
\]

Softmax:

\[
p=\operatorname{softmax}(z).
\]

Loss:

\[
L=-\log p_y.
\]

Global average pooling was introduced in Network in Network as a structural alternative to fully connected classifier heads.

---

### 32. Worked shape pipeline

Input:

\[
224\times224\times3.
\]

Backbone final feature map:

\[
7\times7\times2048.
\]

Global average pool:

\[
2048.
\]

Classifier for \(K=10\):

\[
2048\rightarrow10.
\]

Head parameters:

\[
2048\cdot10+10=20{,}490.
\]

---

## Part VIII — Transfer learning

### 33. Why transfer learning works

Early/mid CNN layers often learn reusable features:

- edges;
- textures;
- parts;
- object-level patterns.

Use pretrained backbone:

\[
\theta_{\text{pretrained}}
\]

then adapt to new task.

PyTorch’s official transfer-learning tutorial covers two common modes: finetuning the whole network and using the pretrained CNN as a fixed feature extractor.

---

### 34. Feature extractor mode

Freeze backbone:

\[
\theta_{\text{backbone}}\ \text{fixed}.
\]

Train only head:

\[
\theta_{\text{head}}.
\]

Objective:

\[
\min_{\theta_{\text{head}}}
L(y, g_{\theta_{\text{head}}}(f_{\text{frozen}}(x))).
\]

Good when:

- small dataset;
- classes similar to pretraining;
- limited compute.

---

### 35. Finetuning mode

Initialize from pretrained weights.

Train:

\[
\theta_{\text{backbone}},\theta_{\text{head}}.
\]

Usually use smaller LR for backbone:

\[
\eta_{\text{backbone}}<\eta_{\text{head}}.
\]

Good when:

- dataset is larger;
- domain differs;
- need better final performance.

---

### 36. Linear probing then finetuning

Practical two-stage recipe:

1. freeze backbone, train head;
2. unfreeze later blocks;
3. optionally unfreeze full model;
4. use discriminative learning rates.

---

### 37. Worked transfer example

Dataset:

\[
5{,}000\text{ images},\quad K=6.
\]

Backbone:

\[
\text{ResNet-50 pretrained}.
\]

Replace head:

\[
2048\rightarrow1000
\]

with:

\[
2048\rightarrow6.
\]

New head parameters:

\[
2048\cdot6+6=12{,}294.
\]

Compared to training all \(\sim\)millions of parameters, this is far safer for small data.

---

### 38. Transfer pitfalls

- wrong input normalization;
- too high LR while finetuning;
- forgetting `model.train()` / `model.eval()`;
- augmentations inconsistent with task;
- class imbalance;
- domain shift;
- frozen BatchNorm issues.

---

## Part IX — From classification backbones to dense prediction

### 39. Why this lecture belongs before OD

Detection and segmentation reuse CNN backbones.

Classification:

\[
\text{image}\rightarrow\text{single vector}\rightarrow\text{class}.
\]

Detection/segmentation:

\[
\text{image}\rightarrow\text{spatial feature maps}\rightarrow\text{boxes/masks}.
\]

So we need to understand:

\[
\boxed{\text{backbone + head}}
\]

---

### 40. Backbone/head abstraction

\[
F=f_{\text{backbone}}(x)
\]

Then:

Classification:

\[
z=f_{\text{cls-head}}(F)
\]

Detection:

\[
\{b_i,c_i,s_i\}=f_{\text{det-head}}(F)
\]

Segmentation:

\[
M=f_{\text{seg-head}}(F)
\]

---

### 41. Feature Pyramid Network preview

Objects appear at different scales.

Need multi-resolution features:

\[
P2,\ P3,\ P4,\ P5.
\]

Use only as preview before OD.

---

## Part X — Summary

### 42. Architecture table

| Architecture | Key idea |
|---|---|
| VGG | stack small \(3\times3\) convs |
| Inception | parallel multi-scale branches + \(1\times1\) reductions |
| ResNet | residual learning |
| MobileNet | depthwise separable conv |
| EfficientNet | compound scaling |
| ConvNeXt | modernized ConvNet design |

---

### 43. Design principles

\[
\boxed{\text{Use local spatial structure}}
\]

\[
\boxed{\text{Share weights}}
\]

\[
\boxed{\text{Preserve trainability with residuals/norm}}
\]

\[
\boxed{\text{Control compute with bottlenecks/separable convs}}
\]

\[
\boxed{\text{Reuse pretrained backbones}}
\]

---

### 44. Next lecture

**Object Detection**

\[
\text{classification: what?}
\]

\[
\text{detection: what and where?}
\]

Topics:

- bounding boxes;
- IoU;
- R-CNN family;
- YOLO;
- detection losses;
- NMS.

---

# Interactives

1. `vgg-receptive-field.html`  
   Stack \(3\times3\) convs and show receptive field growth.

2. `one-by-one-conv.html`  
   Channel mixing at each spatial location.

3. `inception-cost-calculator.html`  
   Compare direct conv vs \(1\times1\) bottleneck branches.

4. `resnet-block-flow.html`  
   Plain vs residual gradient path.

5. `depthwise-separable-cost.html`  
   Standard vs depthwise separable FLOPs.

6. `transfer-learning-playground.html`  
   Freeze/unfreeze backbone, compare train/val curves.

---

# Notebooks

1. `01_vgg_blocks_shapes_params.ipynb`
2. `02_1x1_conv_and_bottlenecks.ipynb`
3. `03_inception_module_costs.ipynb`
4. `04_resnet_block_from_scratch.ipynb`
5. `05_depthwise_separable_conv.ipynb`
6. `06_transfer_learning_resnet.ipynb`
7. `07_feature_extraction_vs_finetuning.ipynb`

---

# Suggested course flow after this

| Lecture | Topic |
|---:|---|
| 8 | CNN basics |
| **9** | **Modern CNN pipelines + transfer learning** |
| 10 | Object detection: localization, R-CNN, YOLO, losses |
| 11 | Segmentation: FCN, U-Net, Mask R-CNN, Dice/IoU losses |
| 12 | Sequence Models I: RNNs, LSTMs, GRUs |
| 13 | Sequence Models II: Seq2Seq, attention, encoder-decoder |
| 14 | Transformers I: self-attention, positional encoding, encoder blocks |
| 15 | Transformers II: GPT/BERT-style pretraining, finetuning, prompting |
| 16 | Vision Transformers and multimodal models |
| 17 | Autoencoders, VAEs and representation learning |
| 18 | Diffusion models / generative deep learning |
| 19 | Self-supervised learning and contrastive learning |
| 20 | Deployment, compression, quantization, distillation |

For your plan, I’d make **OD and segmentation separate**. Detection has enough maths: IoU, box parameterization, anchor assignment, focal loss, NMS, mAP. Segmentation has different maths: pixel CE, Dice, IoU/Jaccard loss, upsampling, skip connections, masks.
