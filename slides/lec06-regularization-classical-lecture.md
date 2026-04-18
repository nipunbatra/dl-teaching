---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Regularization I · Classical &amp; Data-Centric

## Lecture 6 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Recap · the training triad so far

- **Architecture** — ResNets, He init, ReLU (L2).
- **Optimizer** — AdamW with warmup + cosine (L4–L5).
- **Recipe** — debug ladder, error analysis (L3).

<div class="paper">

Today maps to **UDL Ch 9** · regularization. **Brisk** on L1/L2 (you know ridge/LASSO from ES 654) — focus on what's new: double descent, data augmentation, Mixup, label smoothing.

</div>

---

# What's new in DL regularization vs classical ML

<div class="columns">
<div>

### You already know (ES 654)

- L2 / ridge
- L1 / LASSO
- Cross-validation
- Bias-variance tradeoff

We **skim** these.

</div>
<div>

### New for DL

- Double descent
- Data augmentation (images, text)
- Mixup / CutMix
- Label smoothing
- Early stopping as regularization
- Dropout, BatchNorm — next lecture

We spend time here.

</div>
</div>

---

<!-- _class: section-divider -->

### PART 1

# Double descent

Classical bias-variance, revisited

---

# The classical textbook picture

From ES 654 you know the U-curve:

- **Too simple** → underfit (high bias)
- **Too complex** → overfit (high variance)
- **Sweet spot** in the middle

For 50 years ML chose "the middle" via cross-validation. Done.

**Q.** But modern nets have $10^6$–$10^{12}$ parameters for datasets of $10^6$ examples. We should be in catastrophic-overfit territory. Why aren't we?

---

# Double descent · the 2019 surprise

![w:900px](figures/lec06/svg/double_descent.svg)

---

# What's actually happening

Three regimes:

1. **Classical underparameterized** (params ≪ data): U-curve as expected.
2. **Interpolation threshold** (params ≈ data): test error **spikes**.
3. **Modern overparameterized** (params ≫ data): test error **drops again**.

<div class="keypoint">

More parameters past the threshold *does not hurt*. Implicit regularization from SGD + overparameterization finds flat, generalizing minima.

</div>

This is one of the big open questions in DL theory. Prince Ch 20 — *"Why does deep learning work?"*

---

# Practical implication

You don't need to shrink a model to generalize. You can just **go bigger** and rely on:

- **Implicit regularization** from SGD
- **Explicit regularization** (today's topic)
- **Massive data** (when available)

ResNet-50 has 25M params; modern LLMs have 10¹¹. Both generalize fine *because* they are past the interpolation threshold.

---

<!-- _class: section-divider -->

### PART 2

# L2, L1, early stopping

Brisk — you know these from ES 654

---

# L2 / weight decay · 30-second recap

Add $\frac{\lambda}{2}\|\theta\|^2$ to the loss → gradient contribution $\lambda \theta$ → weights shrink every step.

<div class="math-box">

**Bayesian view** · equivalent to a Gaussian prior $\theta \sim \mathcal{N}(0, \sigma^2 I)$ on weights; the optimum becomes MAP instead of MLE.

</div>

In PyTorch · `AdamW(..., weight_decay=0.1)` — the one line you need.

---

# L1 · the sparsity-inducing sibling

Add $\lambda \|\theta\|_1$ → encourages *many* weights to be exactly 0.

| | L1 | L2 |
|---|----|----|
| Prior | Laplace | Gaussian |
| Solution | sparse (many zeros) | small (everything shrinks) |
| Use in DL | rare | ubiquitous |

DL rarely uses L1 — features are *distributed* across many weights, not localized in a few. L1's sparsity breaks distributed representations.

---

# Early stopping as implicit regularization

**Q.** If val loss starts rising at epoch 30, why stop training?

Because continuing means:
- More optimization steps → more capacity effectively used
- Same function class, but walking further into the parameter landscape
- Eventually you'll memorize training noise

<div class="keypoint">

Early stopping = an implicit form of capacity control. It's **free** and almost always helps. Every serious training script checkpoints on val loss.

</div>

Covered in Lecture 1 — the curve with the "best val" marker.

---

<!-- _class: section-divider -->

### PART 3

# Data augmentation

The single highest-value regularizer for vision

---

# Why augmentation is powerful

<div class="insight">

Classical regularization constrains the *model*. Data augmentation constrains the *data* — by telling the model what invariances it should respect.

</div>

A flipped cat is still a cat. A color-jittered cat is still a cat. By training on the augmented versions, the model **learns** that these transformations should not change the prediction.

---

# Standard vision augmentations

![w:920px](figures/lec06/svg/aug_examples.svg)

---

# Which augmentations · which problem

| Domain | Useful | Avoid |
|--------|--------|-------|
| Natural images | flip · crop · color jitter · rotate | vertical flip (changes sky/ground) |
| Medical imaging | small rotations · mild intensity | flips (mirrors anatomy) |
| MNIST / digits | small rotations · elastic | flip (6 ↔ 9) |
| Satellite imagery | all rotations (no "up") · flip | color jitter (semantic) |
| Text | synonym · back-translation · mask | random char shuffle |

**Rule** · augmentation must **preserve the label**. If not, it's noise, not signal.

---

# Advanced · RandAugment, AutoAugment

Instead of hand-picking augmentations:

- **AutoAugment** (2018) — learn the augmentation policy from data.
- **RandAugment** (2020) — pick $N$ augmentations uniformly; pick magnitude $M$ uniformly. Two knobs.

In `torchvision`:

```python
transforms.RandAugment(num_ops=2, magnitude=9)
```

<div class="realworld">

RandAugment is the 2026 default for vision pre-training. Two-line change, consistent +1–3% accuracy.

</div>

---

<!-- _class: section-divider -->

### PART 4

# Mixup &amp; CutMix

Augment the *label*, not just the input

---

# The idea

Standard augmentation: one image → one transformed image, same label.

**Mixup and CutMix** go further: *combine two images* and *interpolate their labels* correspondingly.

---

# Mixup and CutMix in one picture

![w:920px](figures/lec06/svg/mixup_cutmix.svg)

---

# Why Mixup and CutMix work

Three observations:

1. **Decision boundary smoothing.** The model sees "half-cat half-dog" examples with mixed labels → boundary becomes smoother, not piecewise-constant.
2. **Implicit regularizer.** Forces calibration — the output distribution has to reflect the interpolation.
3. **Free data.** No manual annotation; the mixing happens inside the training loop.

Empirically: Mixup/CutMix adds ~1–2% CIFAR-10 accuracy. Essentially a free win for vision.

---

# Mixup in PyTorch · 10 lines

```python
def mixup_batch(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mix, y_a, y_b, lam

# in training step
x_mix, y_a, y_b, lam = mixup_batch(x, y)
logits = model(x_mix)
loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
```

---

<!-- _class: section-divider -->

### PART 5

# Label smoothing

Because "1.0 for the right class" is a lie

---

# Hard vs soft targets

![w:920px](figures/lec06/svg/label_smoothing.svg)

---

# Why soften the labels?

<div class="math-box">

$$y_\text{smooth} = (1 - \alpha)\, y_\text{hard} + \frac{\alpha}{K}$$

For $\alpha = 0.1, K = 10$: correct class gets 0.91, each wrong class gets 0.01.

</div>

Three reasons this helps:

1. **Prevents overconfidence.** Hard labels push logits to $\pm \infty$ — the model becomes miscalibrated.
2. **Regularizes the output layer.** Softer target → smaller logit magnitudes.
3. **Label noise robustness.** Some labels are wrong anyway — smoothing acknowledges uncertainty.

<div class="realworld">

Used in most modern vision and LLM training. `$α = 0.1$` is standard. One flag in PyTorch: `CrossEntropyLoss(label_smoothing=0.1)`.

</div>

---

<!-- _class: section-divider -->

### PART 6

# What actually ships in 2026

---

# The regularization stack for a real vision run

```python
# 1. Architecture regularization (next lecture)
#    — BN + Dropout

# 2. Optimizer regularization
opt = AdamW(model.parameters(), weight_decay=0.05)

# 3. Data augmentation
train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25),
    transforms.Normalize(MEAN, STD),
])

# 4. Mixup inside training loop (conditional)
# 5. Label smoothing in the loss
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 6. Early stopping — checkpoint on val loss
```

---

<!-- _class: summary-slide -->

# Lecture 6 — summary

- **Double descent** explains why overparameterized DL generalizes at all.
- **L2 / L1 / early stopping** — you already know; use AdamW's `weight_decay`, checkpoint on val.
- **Data augmentation** — single highest-value regularizer for vision. Must preserve labels.
- **Mixup / CutMix** — interpolate inputs *and* labels. Free ~1–2% accuracy.
- **Label smoothing** — softens hard targets; better calibration. One flag.
- **In practice** · AdamW weight decay + RandAugment + Mixup + label smoothing is the modern vision default.

### Read before Lecture 7

**Prince** — Ch 9 · *Regularization*. Free at [udlbook.github.io](https://udlbook.github.io/udlbook/).

### Next lecture

**Dropout &amp; Normalization** — Dropout as implicit ensemble, BatchNorm mechanics and the ICS debate, LayerNorm, RMSNorm, and where each one belongs.

<div class="notebook">

**Notebook 6** · `06-regularization.ipynb` — sweep weight decay · add/remove RandAugment · turn label smoothing on/off. Measure val accuracy on CIFAR-10.

</div>
