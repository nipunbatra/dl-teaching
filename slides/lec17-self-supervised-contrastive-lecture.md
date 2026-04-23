---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Self-Supervised &amp; Contrastive Learning

## Lecture 17 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. State the **labeling bottleneck** and why self-supervision matters.
2. Describe **pretext tasks** and give 3 examples (rotation, jigsaw, colorization).
3. Write the **SimCLR pipeline** end-to-end · augmentations, projection head, InfoNCE.
4. Explain **InfoNCE** as a soft classification problem.
5. Contrast **SimCLR vs BYOL** and articulate why BYOL doesn't collapse.
6. Describe **MAE** (masked autoencoding) and when it beats contrastive.
7. Pick an SSL method for a given dataset scale and downstream task.

---

# Where we are

Everything we've seen used labels — classification, MT, LLM pretraining on curated corpora. But labels are **expensive, finite, biased.**

Meanwhile, the internet has **unlimited unlabeled data**. Can we learn from it?

<div class="paper">

Today maps to **UDL Ch 14** (unsupervised / contrastive). Papers: Chen 2020 (SimCLR), Grill 2020 (BYOL), He 2021 (MAE), Oquab 2023 (DINOv2).

</div>

Four questions:
1. What is self-supervised learning, formally?
2. How does **SimCLR** use augmentations as supervision?
3. Why does **BYOL** work **without negatives**?
4. How does **MAE** (masked autoencoding) compare to contrastive?

---

# The labeling bottleneck · in numbers

<div class="math-box">

| Task | Typical labeling cost | Typical dataset size |
|:-:|:-:|:-:|
| ImageNet class | 0.5 USD per image (crowdsource) | 14M images |
| Detection bbox | 5-20 USD per image | 200k images (COCO) |
| Segmentation mask | 30-100 USD per image | 10k images |
| Medical annotation | 50-500 USD per image | ~1-10k images |

</div>

At 14M × $0.5, ImageNet cost ~$7M to label. Segmentation at that scale would be ~$500M. **Labels don't scale.**

Meanwhile · Common Crawl has 10⁹+ web pages, Flickr has billions of photos, YouTube has zetabytes of video. All unlabeled.

---

<!-- _class: section-divider -->

### PART 1

# The labeling bottleneck

Why self-supervision scaled

---

# Two facts about modern ML

1. **Labeled data is scarce.** ImageNet's 14M labels took thousands of human-hours. Medical imaging datasets struggle to reach 10k labeled cases.

2. **Unlabeled data is free.** YouTube uploads 500 hours / minute. The web has exabytes of text, images, video.

<div class="keypoint">

Self-supervised learning invents a "label" from the data itself — a surrogate task — and uses it to learn representations that transfer to real (labeled) downstream tasks.

</div>

Language already won with SSL — every LLM is pretrained with next-token prediction (which needs no labels). Vision caught up in 2020–2023.

---

# What a surrogate task looks like

Given just an image (no label):

- **Predict the next word** (works for text, but images?)
- **Colorize a grayscale image** (labels are the original color channels)
- **Predict rotation** (0°/90°/180°/270°)
- **Fill in missing patches** (MAE)
- **Group augmented versions of the same image** (SimCLR, BYOL)

All of these train an encoder to produce **useful representations** — features that later transfer to classification, detection, segmentation.

---

<!-- _class: section-divider -->

### PART 2

# Contrastive learning · SimCLR

Augmentations as implicit labels

---

# The SimCLR framework

![w:920px](figures/lec17/svg/simclr_framework.svg)

---

# SimCLR · batch as an 8×8 matrix

![w:920px](figures/lec17/svg/simclr_batch_matrix.svg)

---

# How SimCLR works, step-by-step

1. Sample a minibatch of $N$ images.
2. Apply **two different augmentations** to each → $2N$ views.
3. The $(i, j)$ pair from the same image = **positive**. All $2N - 2$ others = **negatives**.
4. Pass all through a **shared encoder** (ResNet / ViT).
5. Pass through a small **projection head** (2-layer MLP).
6. Compute similarity (cosine) in projection space.
7. Apply **NT-Xent loss** — pull positive together, push negatives apart.

<div class="math-box">

$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k \ne i} \exp(\text{sim}(z_i, z_k) / \tau)}$$

</div>

---

# InfoNCE · geometrically

![w:900px](figures/lec17/svg/infonce_geometry.svg)

---

# InfoNCE in plain English

Read the loss $\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\text{sim}(z_i, z_k)/\tau)}$ one piece at a time:

- **Numerator** — similarity to the *one* positive. We want this **high**.
- **Denominator** — sum of similarities to *all* $2N-1$ items in the batch. We want this **low** (except for the one positive already in the sum).
- **Temperature $\tau$** — sharpens or softens the softmax. Smaller $\tau$ = sharper contrast, harder negatives matter more.

<div class="keypoint">

It's a **softmax classification** problem · "given $z_i$, which of the $2N-1$ candidates is its positive partner?" Cross-entropy with $2N-1$ classes, no labels needed — the positive is the *other augmentation of the same image*.

</div>

---

# Why SimCLR works

Three ingredients (Chen et al. 2020 ablations):

1. **Strong augmentations** — especially color jitter + random crop. Weaker augs give much worse representations.
2. **Projection head** — throw it away after pretraining; use encoder features for downstream tasks. The projection exists to absorb augmentation invariances.
3. **Large batch size** — more negatives → sharper contrast. SimCLR used batch 8192 on a TPU pod.

<div class="realworld">

Pretrained SimCLR features, fine-tuned, match or beat supervised ImageNet on many downstream tasks. This was surprising in 2020 — still foundational today.

</div>

---

# Temperature · the forgotten hyperparameter

The InfoNCE softmax uses a temperature $\tau$ inside: $\text{sim}(z_i, z_j) / \tau$.

- **Small $\tau$ (≈ 0.07)** · sharper softmax → hard negatives dominate gradient. SimCLR's choice.
- **Large $\tau$ (≥ 1.0)** · all negatives contribute equally → gradient is softer.

<div class="keypoint">

$\tau$ controls *which negatives the model pays attention to*. Too small and training is noisy / unstable. Too large and training is slow. It's also what CLIP learns as a trainable scalar (L18).

</div>

---

# Augmentation matters · *hugely*

Chen et al. swept augmentation pairs. Accuracy (linear-probe on ImageNet):

| Augmentation pair | Accuracy |
|:-:|:-:|
| crop only | 40% |
| color-jitter only | 28% |
| crop + color-jitter | 56% |
| crop + color + blur | 64% |

<div class="insight">

Contrastive learning is as much about **what invariances you pick** as about the loss. You're telling the model · "ignore crops, ignore color shifts, ignore blur — but pay attention to content." Those choices **become** the downstream invariances of the representation.

</div>

---

<!-- _class: section-divider -->

### PART 3

# BYOL · self-distillation without negatives

Two networks chase each other

---

# The BYOL surprise

Every contrastive method needed **negatives**. BYOL (Grill et al. 2020) dropped negatives entirely and *still* learned useful features. How?

- **Online network** (with predictor head) — takes view 1.
- **Target network** — slow-moving EMA of the online network; takes view 2.
- Online network predicts target's output.
- Target is **stop-gradient** — gradients only flow through online.

<div class="math-box">

Online updates: $\theta \leftarrow \text{SGD}(\mathcal{L})$
Target updates: $\xi \leftarrow m\,\xi + (1-m)\,\theta$ with $m \approx 0.996$

</div>

The asymmetry (predictor + EMA + stop-grad) prevents collapse without needing negatives.

---

# Why doesn't BYOL collapse?

Without negatives, the obvious failure mode is $f(x) = \text{const}$ — both networks output the same vector, loss is zero. Why doesn't this happen?

<div class="keypoint">

Three forces prevent collapse:

1. **Predictor head** introduces asymmetry — online must *predict* target, not match it directly.
2. **Stop-gradient** on the target prevents the target from moving to meet the online's output.
3. **EMA update** gives the target a momentum that trails the online; the online can never "catch" its target exactly.

</div>

Grill et al. had to run the training for 300 epochs to verify it really didn't collapse — nobody initially believed it.

---

# MoCo, SwAV, and the contrastive zoo

The 2019–2021 era had many flavors:

| Method | Key idea |
|--------|----------|
| **MoCo** | FIFO queue of negatives; momentum encoder |
| **SimCLR** | large batch negatives; projection head |
| **SwAV** | online clustering (prototype assignments) |
| **BYOL** | no negatives; predictor + EMA |
| **SimSiam** | BYOL minus EMA — even simpler |
| **Barlow Twins** | decorrelate representations across views |

By 2023 the community mostly converged on masked autoencoding (MAE) and self-distillation (DINO).

---

# The pattern across all these methods

Zoom out. Every SSL method is a variation on **"create a task the model can only solve if it learns features."**

<div class="keypoint">

- **Predict the future** (GPT, word2vec) — temporal / sequential structure.
- **Predict the missing part** (BERT, MAE) — contextual structure.
- **Match two views of the same thing** (SimCLR, MoCo) — invariance to nuisance.
- **Predict what a teacher thinks** (BYOL, DINO, knowledge distillation) — relational structure.

</div>

The architecture and loss differ, but the meta-idea is the same · **make the data supervise itself**.

---

<!-- _class: section-divider -->

### PART 4

# MAE · BERT for pixels

Masked autoencoding for images

---

# MAE · mask 75%, reconstruct the rest

![w:920px](figures/lec17/svg/mae_masking.svg)

---

# Why MAE beat contrastive (for many tasks)

<div class="columns">
<div>

### Contrastive (SimCLR)

- Needs massive batches for negatives
- Sensitive to augmentation choice
- Strong on classification
- Weak on dense tasks (segmentation)

</div>
<div>

### MAE

- Any batch size works
- Minimal augmentation
- Strong on detection / segmentation
- Slightly weaker on pure classification

</div>
</div>

<div class="keypoint">

**MAE's asymmetric encoder-decoder is the key.** The heavy encoder only sees visible patches (25%) → 4× cheaper than processing the full image.

</div>

He et al. 2021 · ViT-Huge MAE pretraining → state-of-the-art for many downstream vision tasks.

---

# DINO and DINOv2 · self-distillation at scale

**DINO (Caron et al. 2021)** combined MAE-style patches with BYOL-style self-distillation for ViTs. Emergent properties:

- **Zero-shot object segmentation** appears in attention maps without any training on masks.
- Features transfer across domains without fine-tuning.

**DINOv2 (Oquab et al. 2023)** scaled this up to ViT-g (1B params) on 142M curated images.

<div class="realworld">

DINOv2 features are the *de facto* general-purpose vision representation in 2026 — ship it for any vision task where you can't afford full fine-tuning.

</div>

---

<!-- _class: section-divider -->

### PART 5

# Where self-supervision lives in 2026

---

# The landscape

| Modality | Dominant SSL approach |
|----------|-----------------------|
| **Text** | Next-token prediction (every LLM) |
| **Vision** | MAE + DINO-style distillation |
| **Speech** | Wav2Vec 2.0 / HuBERT (masked frame prediction) |
| **Video** | MAE extended to spacetime patches |
| **Multimodal** | CLIP-style contrastive image-text (next lecture) |

<div class="insight">

Self-supervision created the **foundation model** era. Every modality now has its own canonical SSL recipe. Supervised learning survives only at the end of the pipeline — fine-tuning on small labeled data.

</div>

---

<!-- _class: summary-slide -->

# Lecture 17 — summary

- **Self-supervision** turns unlabeled data into training signal via surrogate tasks.
- **SimCLR** · pull two augmentations of the same image together, push all others apart. Needs large batches (for negatives).
- **BYOL** · two networks with EMA + stop-gradient; no negatives needed. Still works.
- **MAE** · mask 75% of patches; reconstruct; asymmetric encoder-decoder; the 2022 winner.
- **DINO(v2)** · self-distillation for ViTs; emergent segmentation in attention; the 2026 general-purpose vision representation.

### Read before Lecture 18

Prince Ch 12 §12.5 (ViT) + CLIP paper (Radford 2021).

### Next lecture

**Vision-Language Models** — ViT, CLIP, LLaVA, multimodal LLMs.

<div class="notebook">

**Notebook 17** · `17-simclr-mini.ipynb` — implement NT-Xent from scratch; pretrain on CIFAR-10; t-SNE the embeddings to see class clustering without labels.

</div>
