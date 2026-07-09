---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Self-Supervised &amp; Contrastive Learning

## Lecture 19 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

Every model so far needed a human to write down $y$ for each $\mathbf x$. Labels are the scarcest, most expensive thing we own. Today we stop asking for them.

<div class="keypoint">

**Make the data supervise itself.** Hide part of an input, or show the model two versions of the same input, and turn "match them back up" into the loss. No human ever labels a thing — yet the features you learn beat ImageNet-supervised on most downstream tasks.

</div>

You met the cross-entropy loss in L1 and the ViT/transformer encoder in L7/L12. Today we keep the encoder and the loss, and throw away the label:

$$\text{unlabeled data} \rightarrow \underbrace{\text{a made-up task}}_{\text{augment / mask}} \rightarrow \underbrace{\text{contrastive (SimCLR)} \;/\; \text{predictive (MAE)}}_{\text{one loss each}} \rightarrow \text{reusable features}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**Where today's ideas came from:**
- **L1** — cross-entropy / softmax; InfoNCE *is* softmax cross-entropy
- **L7 / L12** — the CNN & ViT encoders we now pretrain
- **L16** — BERT's masked-token trick; MAE is "BERT for pixels"

</div>
<div>

**Where they go next:**
- **L20** — CLIP: the *second view* of an image becomes its **caption** (contrastive across modalities)
- **L21** — SSL features seed generative & diffusion pretraining

</div>
</div>

<div class="insight">

This is the opener of the **vision / representation** arc: how to learn *what things look like* before anyone tells you *what they are called*. Framing follows **A. Ng** — introduce the loss where it's used, one idea per slide.

</div>

---

<!-- _class: section-divider -->

## Part 1 · Make the data supervise itself

---

# Labels don't scale — unlabeled data is free

<div class="math-box">

| Task | Cost per image | Realistic dataset |
|:-:|:-:|:-:|
| ImageNet class tag | ~$0.5 (crowdsourced) | 14M images |
| Detection bounding box | $5–20 | 200k (COCO) |
| Segmentation mask | $30–100 | ~10k |
| Medical annotation | $50–500 | ~1–10k |

</div>

ImageNet's labels cost **~$7M**; segmentation at that scale would run **~$500M**. Meanwhile Common Crawl has $10^9$ pages, Flickr billions of photos, YouTube uploads 500 hours **per minute** — all unlabeled.

<div class="keypoint">

Compute is getting cheaper every year; **human labels are not.** If labels cost more than compute, learn from the unlabeled pile instead.

</div>

---

# Text already won this way

Language never had a labeling bottleneck — the supervision was hiding in the text itself.

- **word2vec (2013)** — predict a word from its neighbours.
- **BERT (L16)** — mask 15% of tokens, predict them back.
- **GPT** — predict the next token, on a trillion-token corpus.

<div class="insight">

Every LLM you have used was pretrained with **zero human labels** (before any RLHF). The "label" is just *the rest of the sentence*. Vision only caught up around 2020 — this lecture is how. The meta-move is identical: **invent a task the model can only solve by learning good features.**

</div>

---

# The fix: invent the label from the data

Given just an image and no tag, manufacture a supervised task:

- **Predict rotation** — was it turned 0° / 90° / 180° / 270°?
- **Colorize** — the label is the original color you removed.
- **Fill in missing patches** — mask, then reconstruct (**MAE**, Part 4).
- **Match two views of the same image** — pull them together (**SimCLR**, Part 2).

<div class="keypoint">

These are **pretext tasks** — the answer is derivable from the raw data, so the label is free. We don't care about the pretext answer itself; we keep the **encoder** it forces the model to build.

</div>

---

# What we're really after: reusable features

The point of pretraining is a *frozen* encoder whose feature space is already organized — so a downstream task needs almost no labels.

<div class="insight">

**Linear probe.** Freeze the encoder, train a **single linear layer** on top from a handful of labels. If a one-layer classifier works, the features must already place similar images near each other. **Pretrain once on the unlabeled pile; reuse everywhere from a few labels.**

</div>

This is the whole business case for SSL: turn an expensive-label problem into a cheap-label problem. Now — how do we build such an encoder without any labels? Two answers: **contrastive** and **predictive**.

---

<!-- _class: section-divider -->

## Part 2 · Contrastive learning — SimCLR

---

# Two augmented views = a positive pair

Take one image. Augment it **twice** (random crop, color-jitter, blur) → two views that *look* different but *are* the same content.

![w:760px](figures/lec17/svg/positive_pairs.svg)

<div class="keypoint">

- **Positive pair** — two augmentations of the **same** image → pull **together**.
- **Negatives** — every **other** image in the batch → push **apart**.

The augmentations *manufacture the label*. Nobody tagged a thing — the supervision is "which view is the other crop of me?"

</div>

---

# The SimCLR pipeline

![w:900px](figures/lec17/svg/simclr_framework.svg)

Each view flows through a **shared encoder** $f$ (ResNet / ViT) → features $h$, then a small **projection head** $g$ (2-layer MLP) → $z$. We contrast in $z$-space, but keep $h$ for downstream use.

---

# Build the batch: one positive, the rest negative

Sample $N$ images, augment each twice → $2N$ views. For anchor view $i$, exactly **one** other view $j$ is its positive; the other $2N-2$ are negatives.

![w:820px](figures/lec17/svg/simclr_batch_matrix.svg)

The $2N \times 2N$ similarity matrix *is* the supervision: light the diagonal-pair cells, darken the rest.

---

# InfoNCE: "which one is my partner?"

Give each view a score against every other view — **cosine similarity** in $z$-space:

$$\text{sim}(z_i, z_k) = \frac{z_i \cdot z_k}{\lVert z_i\rVert\,\lVert z_k\rVert}$$

Now turn "find my positive partner $j$" into a **softmax classification** over the $2N-1$ candidates, with a **temperature** $\tau$:

$$P(\text{partner}=k \mid z_i) = \frac{\exp\big(\text{sim}(z_i,z_k)/\tau\big)}{\sum_{k'\ne i}\exp\big(\text{sim}(z_i,z_{k'})/\tau\big)}$$

The "classes" are **batch positions**; the label is the position of the other augmentation.

---

# InfoNCE: the full loss

Cross-entropy against the true partner $j$ — substitute the softmax and you get the whole loss:

<div class="math-box">

$$\mathcal L_{i,j} = -\log \frac{\exp\big(\text{sim}(z_i,z_j)/\tau\big)}{\displaystyle\sum_{k\ne i}\exp\big(\text{sim}(z_i,z_k)/\tau\big)}$$

</div>

Read it one piece at a time:
- **Numerator** — similarity to the *one* positive. Want it **high**.
- **Denominator** — similarity to *everything* in the batch. Want it **low** (bar the positive already inside).

<div class="insight">

It's exactly the L1 cross-entropy loss — same recipe, no human labels. The positive is just "the other crop of me."

</div>

---

# InfoNCE, geometrically

![w:820px](figures/lec17/svg/infonce_geometry.svg)

On the unit hypersphere: the anchor pulls its positive **closer** (numerator up) and pushes all negatives **away** (denominator down). The learned geometry — similar content clustered together — is precisely what makes the linear probe work.

---

# Why more negatives help

The InfoNCE denominator sums over **every** negative in the batch. More negatives = "find my partner among more distractors" = a harder task = a stronger signal and a more uniformly-spread embedding space.

That is why SimCLR needed **batch 8192**, and why **MoCo** added a FIFO **queue** of past embeddings — thousands of negatives without a giant batch.

<div class="keypoint">

More negatives → tighter contrast → better features, up to a saturating point (Appendix: exactly the $\log N$ term in the mutual-information bound). This batch-size hunger is precisely what **BYOL and MAE escape** — Part 3 drops negatives entirely.

</div>

---

# Temperature $\tau$ — the sharpness knob

$\tau$ rescales the logits *before* the softmax: small $\tau$ ⇒ sharp distribution ⇒ the **hardest** (closest) negatives dominate the gradient.

Same scores $[0.9,\ 0.2,\ -0.1]$ (one positive, two negatives), two temperatures:

<div class="columns">
<div>

**$\tau = 1.0$** (soft)
$\exp[0.9,0.2,-0.1]=[2.46,1.22,0.90]$
Probs $=[0.54,\ 0.27,\ 0.20]$
Negatives still carry weight.

</div>
<div>

**$\tau = 0.1$** (SimCLR)
$\exp[9,2,-1]=[8103,\ 7.4,\ 0.4]$
Probs $=[0.999,\ 0.001,\ \sim0]$
Positive dominates; hard negatives bite.

</div>
</div>

SimCLR uses $\tau\approx0.1$. Next lecture, **CLIP learns $\tau$** as a trainable scalar.

---

# Augmentations are the whole game

Chen et al. swept augmentation pairs — linear-probe accuracy on ImageNet:

| Augmentation | Acc. |
|:-:|:-:|
| crop only | 40% |
| color-jitter only | 28% |
| crop + color-jitter | 56% |
| crop + color + blur | **64%** |

<div class="keypoint">

The invariances you **augment away become the invariances of the representation.** You are telling the model: *ignore crop, ignore color shift, ignore blur — attend to content.* Pick weak augmentations and the model cheats on color histograms instead of learning semantics.

</div>

---

# The projection head — don't throw away good features

InfoNCE forces two augmentations to map to the **same** $z$. But a downstream task might need what it discards (color *does* signal ripeness).

<div class="insight">

**Crumple-zone analogy.** The encoder $f$ is the passenger cabin — keep it rich ($h$ has color, texture, shape). The projection head $g$ is the crumple zone — it *absorbs the crash* of the harsh contrastive task, squeezing $h\to z$. **After pretraining, discard $g$ and use $h$.** The encoder never had to delete useful information to win the contrastive game.

</div>

---

# See it / play with it

<div class="notebook">

**🎛 Interactive · contrastive embedding space** — animate positive pairs pulling together and negatives pushing apart on the sphere; drag $\tau$ and watch the softmax sharpen. *(Interactive Lab · `interactive/articles/vision-ssl`)*

</div>

<div class="notebook">

**📓 Notebook · NT-Xent from scratch** — implement the InfoNCE loss, pretrain a small encoder on CIFAR-10 with no labels, then t-SNE the embeddings and *see* the classes cluster. *(ES 667 · `notebooks/17-contrastive-toy.ipynb`)*

</div>

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** Anchor $z_1$ has cosine similarity **0.8** to its positive, and **0.3** and **0.1** to two negatives. With temperature $\tau=0.2$, compute the InfoNCE loss $\mathcal L$ for $z_1$. *(Scale by $1/\tau$, exponentiate, normalize, take $-\log$ of the positive's share.)*

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

Scale logits by $1/\tau = 5$: positive $\to 4.0$, negatives $\to 1.5,\ 0.5$.

$$e^{4.0}=54.60,\quad e^{1.5}=4.48,\quad e^{0.5}=1.65 \;\Rightarrow\; \text{denominator}=60.73$$

$$\mathcal L = -\log\frac{54.60}{60.73} = -\log(0.899) \approx \mathbf{0.11}$$

<div class="insight">

At $\tau=1$ (no sharpening) the *same* scores give $\mathcal L\approx0.74$ — nearly **7× larger**. Temperature turns a similarity gap of 0.5 into a logit gap of 2.5, so a modest positive lead becomes near-certainty. Small $\tau$ = the model must make the positive *dramatically* closer than any negative.

</div>

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** SimCLR's accuracy collapses if you use only *weak* augmentations (say, a tiny random crop and nothing else). Explain **why strong augmentations are essential** — what does the encoder learn to exploit when the two views are nearly identical?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

If the two views are nearly identical, "matching them" is trivial via **low-level shortcuts** — color histogram, brightness, JPEG artifacts, patch position. The encoder can win InfoNCE **without learning any semantics**, so the features don't transfer.

<div class="keypoint">

Strong, overlapping-but-different views (aggressive crop + color-jitter + blur) *destroy* those shortcuts, forcing the encoder to rely on **content** — the object that survives every augmentation. The augmentation set literally *defines* which nuisances the representation becomes invariant to (recall the ablation table: crop+color+blur → 64%, crop-only → 40%).

</div>

---

<!-- _class: section-divider -->

## Part 3 · Drop the negatives — BYOL

---

# The collapse trap

SimCLR needs big batches purely to have enough negatives (SimCLR used batch **8192** on a TPU pod). Can we keep *only* the "pull positives together" force and drop negatives?

<div class="keypoint">

**Danger: collapse.** With only an attractive force, the trivial optimum is a **constant** — $f(\mathbf x)=[c,c,\dots]$ for every image. Loss is exactly zero, features are useless. The negatives were the *repulsive* force keeping the space spread out. Remove them naively and everything falls to a point.

</div>

BYOL removes negatives and *still* doesn't collapse — through asymmetry.

---

# BYOL: two networks, a predictor, an EMA target

<div class="columns">
<div>

**Online network** ($\theta$) — trained by SGD; has an **extra predictor** head $q$.
**Target network** ($\xi$) — **no gradients**; its weights are a slow EMA of the online:
$$\xi \leftarrow m\,\xi + (1-m)\,\theta,\quad m\approx 0.99$$

</div>
<div>

**The game.** Online sees view 1, must *predict* the target's output for view 2:
$$\mathcal L = \big\lVert\, q(\text{online}_1) - \text{sg}(\text{target}_2)\,\big\rVert^2$$
$\text{sg}$ = stop-gradient — no gradient flows into the target.

</div>
</div>

The EMA makes the target a **stable, slow-moving teacher**; the online is chasing a lagged version of itself.

---

# BYOL's EMA target — a worked trace

The target trails the online with momentum $m$. Take $m=0.9$, target initialized to the online $\xi_0=\theta_0=[10,\,2]$:

<div class="math-box">

**Step 1** — online jumps to $\theta_1=[12,\,3]$:
$$\xi_1 = 0.9\,[10,2] + 0.1\,[12,3] = [10.2,\ 2.1]$$

**Step 2** — online jumps to $\theta_2=[11,\,5]$:
$$\xi_2 = 0.9\,[10.2,2.1] + 0.1\,[11,5] = [10.28,\ 2.39]$$

</div>

The teacher lags *far* behind each jump of the student — a stable reference that drifts only 10% of the way each step. That lag is exactly what gives the online a non-trivial, always-slightly-ahead target to predict, instead of a mirror it could match with a constant.

---

# Why BYOL doesn't collapse

<div class="keypoint">

Three forces break the symmetry that would allow a constant solution:

1. **Predictor head** $q$ — the online must *predict* the target, not equal it. This extra map has to be learned, and it can't reduce to identity-onto-a-constant.
2. **Stop-gradient** on the target — the target can't move its output to meet the online. A constant is not a *fixed point the loss can slide into*.
3. **EMA target** — the target trails the online with momentum; the online can never instantaneously catch it and freeze.

</div>

Grill et al. ran 300 epochs to convince a skeptical field it truly doesn't collapse. **SimSiam** later showed even the EMA is optional — predictor + stop-gradient alone suffice.

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** BYOL has **no negatives**, yet the encoder does *not* collapse to $f(\mathbf x)=\text{const}$. Name the architectural ingredients that prevent collapse, and explain **why stop-gradient specifically** is what blocks the constant solution.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

The ingredients: **predictor head** + **stop-gradient on the target** + **EMA target** (asymmetry between online and target).

<div class="insight">

**Why stop-gradient is the key.** The constant $f(\mathbf x)=c$ gives zero loss only if *both* networks emit $c$. With stop-gradient, gradients update **only the online** — the target is a fixed reference each step. So there is no gradient path that drags the target *down* to meet a collapsing online; the online must instead *predict* a moving, non-trivial teacher. Remove stop-gradient and both sides can slide to the same constant together → collapse. (The predictor + EMA make that teacher rich enough that the online can't win by outputting a constant.)

</div>

---

# The contrastive zoo, in one table

| Method | Negatives? | Key trick |
|:-:|:-:|:-:|
| **SimCLR** | yes (in-batch) | InfoNCE + projection head |
| **MoCo** | yes (FIFO queue) | momentum encoder |
| **SwAV** | no (clustering) | online prototype assignments |
| **BYOL** | **no** | predictor + EMA + stop-grad |
| **SimSiam** | **no** | BYOL minus EMA |
| **Barlow Twins** | **no** | decorrelate features across views |

Every row writes a *free* supervised signal out of the same "two views of one image" idea. By 2023 the field mostly converged on **self-distillation (DINO)** and **masked prediction (MAE)**.

---

# Evaluating an SSL encoder

How good is a *frozen* encoder, really? Two protocols dominate.

<div class="columns">
<div>

**Linear probe** — freeze the encoder, train one linear layer. *Isolates* inherent feature quality. The cleanest measure.

</div>
<div>

**Fine-tune** — unfreeze everything (low LR). Tests the **ceiling**, but a good fine-tuner can hide a mediocre encoder.

</div>
</div>

<div class="insight">

**Knife analogy.** Linear probe = hand the knife to a beginner: a clean cut means the *blade* is sharp. Fine-tune = hand it to a master chef: great results, but skill masks the blade. Also common: **k-NN** (no classifier at all) and **few-shot** (tiny labeled set).

</div>

---

<!-- _class: section-divider -->

## Part 4 · Masked prediction — MAE

---

# MAE = BERT for pixels

Different recipe, same spirit: don't *contrast* views — **delete** most of the image and ask for it back.

![w:820px](figures/lec17/svg/mae_masking.svg)

Split the image into patches, **mask 75%**, feed the visible 25% to the encoder, reconstruct the missing patches. To fill a face or a tree branch from a quarter of the pixels, the encoder must build a real *visual world model* — exactly BERT's masked-token trick (L16), moved to pixels.

---

# The asymmetric encoder / decoder

![w:820px](figures/lec17/svg/mae_pipeline.svg)

<div class="keypoint">

**The heavy encoder sees only the visible 25%.** A large ViT processes a quarter of the patches → **~4× cheaper** pretraining. A **lightweight decoder** then takes the encoded visible patches plus mask tokens and reconstructs everything. Encoder = expensive expert on a few pages; decoder = cheap intern writing the full document. After pretraining, keep the encoder, discard the decoder.

</div>

---

# MAE vs contrastive — who wins what

<div class="columns">
<div>

### Contrastive (SimCLR)

- Needs **large batches** for negatives
- Very sensitive to augmentation choice
- Strong on **classification** (global features)
- Weaker on dense / segmentation tasks

</div>
<div>

### Masked (MAE)

- **Any batch size** works
- Minimal augmentation needed
- Strong on **detection / segmentation** (local features)
- Slightly weaker on pure classification

</div>
</div>

<div class="insight">

Contrastive learns *invariances* (throw nuisances away); masked prediction learns *everything reconstructable* (keep local detail). That's why they win different downstream tasks — and why DINOv2 blends both ideas.

</div>

---

# When should you reach for SSL?

<div class="math-box">

| Scenario | Use SSL? |
|:-:|:-:|
| Plenty of labels, one task | No — just train supervised |
| Big unlabeled pool, few labels | **Yes** — SSL pretrain + fine-tune |
| Need generic features for many tasks | **Yes** — start from DINOv2 |
| Novel domain (medical, satellite) | **Yes** — in-domain SSL, then fine-tune |
| Tight deploy budget, no pretraining | Use a **frozen** DINOv2 / CLIP |

</div>

Rule of thumb: **if labels cost more than compute, use SSL.** In most real settings, labels *are* the bottleneck.

---

<!-- _class: section-divider -->

## Part 5 · What won, and what's next

---

# DINOv2 — the practical default

Self-distillation (BYOL-style, no negatives) applied to ViTs *at scale* — 142M curated images, up to 1B params.

<div class="math-box">

| Backbone | Linear-probe (ImageNet) | Fine-tune (detection) |
|:-:|:-:|:-:|
| Supervised ResNet-50 | 76.1 | 38 mAP |
| SimCLR ResNet-50 | 69 | 36 |
| MAE ViT-H | 76 | **54 mAP** |
| **DINOv2 ViT-L** | **86** | 52 |

</div>

DINOv2 dominates frozen-feature evaluation, with **emergent segmentation** in its attention maps — never trained on a single mask.

<div class="keypoint">

Practical advice: for almost any vision task where you can't afford full fine-tuning, **start from frozen DINOv2 features** and put a small head on top.

</div>

---

# The meta-pattern behind every SSL method

Zoom out — every method is one sentence with the blanks filled differently:

<div class="keypoint">

*"Create a task the model can only solve by learning features."*

- **Predict the next thing** — GPT, word2vec (sequential structure)
- **Predict the missing thing** — BERT, MAE (contextual structure)
- **Match two views of one thing** — SimCLR, MoCo (invariance to nuisances)
- **Predict what a teacher thinks** — BYOL, DINO (relational structure)

</div>

Architectures and losses differ; the move is identical — **make the data supervise itself.**

---

# The SSL landscape, by modality

| Modality | Dominant SSL recipe |
|:-:|:-:|
| **Text** | next-token prediction (every LLM) |
| **Vision** | MAE + DINO-style self-distillation |
| **Speech** | Wav2Vec 2.0 / HuBERT (masked frame prediction) |
| **Video** | MAE over spacetime patches |
| **Multimodal** | CLIP-style image–text contrastive → **next lecture** |

Self-supervision created the **foundation-model** era. Supervised learning now survives mainly at the very end — fine-tuning on a little labeled data.

---

# Next: CLIP — the second view is the caption

Everything today used *two augmentations of one image* as a positive pair. **L20 changes the second view.**

<div class="insight">

CLIP keeps the exact InfoNCE loss — but the positive pair is now **(image, its caption)**. Contrast across **modalities** instead of augmentations, on 400M image–text pairs from the web (captions are the free label). The payoff: **zero-shot** classification — name any class in words and the model finds it, no training. Same contrastive machinery you learned today, one modality wider.

</div>

<div class="notebook">

**🎛 Interactive · CLIP zero-shot** — preview next lecture: score an image against text prompts and watch the softmax pick a class. *(Interactive Lab · `interactive/articles/clip-zero-shot`)*

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Make the data supervise itself: a made-up task (match two views, or reconstruct masked patches) is a free label, and the encoder it forces you to build transfers everywhere.**

</div>

- **Contrastive (SimCLR)** — two augmentations = a positive pair; InfoNCE pulls positives close, pushes negatives apart; temperature sharpens; **augmentations define the invariances.**
- **No-negatives (BYOL)** — predictor + stop-gradient + EMA target prevent collapse without negatives.
- **Masked (MAE)** — mask 75%, reconstruct; asymmetric heavy-encoder / light-decoder; "BERT for pixels."
- **DINOv2** — the practical default frozen encoder today.

**Next (L20):** swap the second view for a caption — that's **CLIP**.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts the instructor's ES 667 SSL materials (IIT Gandhinagar) and standard references:

- **SimCLR — positive pairs, NT-Xent/InfoNCE, augmentation ablations** — Chen et al. 2020, *A Simple Framework for Contrastive Learning of Visual Representations.*
- **BYOL — no negatives, predictor + EMA + stop-gradient** — Grill et al. 2020, *Bootstrap Your Own Latent.*
- **MAE — mask 75%, asymmetric encoder/decoder** — He et al. 2022, *Masked Autoencoders Are Scalable Vision Learners.*
- **DINOv2 — self-distillation at scale, general-purpose features** — Oquab et al. 2023, *DINOv2.*
- **Survey framing** — L. Weng, *Self-Supervised Representation Learning* (blog).
- **Pedagogical framing** — A. Ng, *Deep Learning Specialization* (ideas introduced where they're used).

Figures reused from the ES 667 `lec17` figure library. All source materials © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · InfoNCE is a lower bound on mutual information

For a batch of $N$ candidates (one positive $j$, $N-1$ negatives) with an optimal critic, the InfoNCE objective bounds the mutual information between the two views $z_i, z_j$:

$$I(z_i;\, z_j) \;\ge\; \log N \;-\; \mathcal L_{\text{InfoNCE}}$$

So **minimizing** InfoNCE **maximizes** a lower bound on $I(z_i; z_j)$ — the model is squeezing out as much shared information between the two augmentations as it can. Two consequences fall out immediately: bigger $N$ (more negatives) → tighter bound → *why large batches / memory queues help*; and the bound saturates at $\log N$, so returns diminish. This is the original **CPC** (van den Oord et al. 2018) view; SimCLR is the augmentation-based special case. *(Interactive: `interactive/articles/info-theory`.)*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · why stop-gradient beats collapse, formally

Write the BYOL loss as $\mathcal L(\theta) = \lVert q_\theta(f_\theta(x_1)) - \text{sg}(f_\xi(x_2))\rVert^2$. Stop-gradient means we optimize $\theta$ treating the target $f_\xi(x_2)$ as a **constant** — this is an *alternating* optimization (predict a fixed teacher, then slowly move the teacher via EMA), not a joint minimization.

The joint problem $\min_{\theta,\xi}$ *does* have the trivial global minimum $f\equiv c$. The stop-gradient / EMA scheme never optimizes that joint objective, so it need not fall into that minimum. Tian et al. (2021) show the predictor $q$ aligns with the covariance's eigen-structure of the online features, and the resulting dynamics have the collapsed solution as an **unstable** point — pushed away, not toward. Hence: **no negatives, no collapse** — the asymmetry, not a repulsive force, does the work.
