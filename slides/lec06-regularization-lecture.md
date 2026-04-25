---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Regularization in Deep Learning

## Lecture 6 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

*This lecture is comprehensive — Part 1 covers classical + data-centric regularization; Part 2 covers dropout and normalization. Intended to span two sessions.*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. Explain **double descent** and when it happens.
2. Pick among **L2 / early stopping / augmentation** by regime.
3. Apply **Mixup / CutMix / label smoothing** correctly.
4. Implement **dropout** (including inverted scaling) from scratch.
5. Choose among **BN / LN / GN / RMSNorm** by architecture.
6. Explain **pre-norm vs post-norm** trade-offs for depth.

---

# Recap · where we are

- **Architecture** — ResNets, He init, ReLU (L2).
- **Optimizer** — AdamW with warmup + cosine (L4–L5).
- **Recipe** — debug ladder, error analysis (L3).

<div class="paper">

Today maps to **UDL Ch 9** (Regularization) and the BatchNorm parts of **Ch 11** (Residual Networks). ES 654 covered ridge/LASSO — we skim those and focus on what's new.

</div>

---

# Plan for the two sessions

**Session 1 · classical &amp; data-centric**
1. Double descent — the modern bias-variance picture
2. L2, L1, early stopping (brisk — prereq covered)
3. Data augmentation
4. Mixup &amp; CutMix
5. Label smoothing

**Session 2 · architectural**
6. Dropout
7. BatchNorm — mechanics, ICS debate
8. LayerNorm — why sequences differ
9. RMSNorm — the modern simplification
10. Pre-norm vs post-norm placement

---

# Two students · the regularization story

<div class="keypoint">

**Student A · memorizes** the 100 practice problems. Aces the practice test. Fails the real exam (different problems).

**Student B · learns the method.** Doesn't memorize · understands. Does fine on both.

Regularization is how we force our model to be Student B.

</div>

Without it · a deep network has more than enough capacity to memorize the training set perfectly while learning nothing transferable. With it · the model is encouraged to find patterns that hold beyond the training data.

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
- Dropout, BatchNorm, LayerNorm, RMSNorm

We spend time here.

</div>
</div>

---

<!-- _class: section-divider -->

### SESSION 1 · PART 1

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

### SESSION 1 · PART 2

# L2, L1, early stopping

Brisk — you know these from ES 654

---

# L2 worked numeric · single-weight update

<div class="math-box">

A single weight · $w = 2.0$. Loss gradient · $dL/dw = 1.5$. LR · $\eta = 0.1$. Decay · $\lambda = 0.01$.

**Without L2** · $w_\text{new} = 2.0 - 0.1 \cdot 1.5 = 1.85$

**With L2** · the gradient becomes $1.5 + \lambda w = 1.52$
$w_\text{new} = 2.0 - 0.1 \cdot 1.52 = 1.848$

</div>

The weight ends up slightly smaller · "decayed" toward zero. Repeat over thousands of steps · big weights shrink, the loss only lets them grow if they really earn their keep.

---

# L2 / weight decay · 30-second recap

Add $\frac{\lambda}{2}\|\theta\|^2$ to the loss → gradient contribution $\lambda \theta$ → weights shrink every step.

<div class="math-box">

**Bayesian view** · equivalent to a Gaussian prior $\theta \sim \mathcal{N}(0, \sigma^2 I)$ on weights; the optimum becomes MAP instead of MLE.

</div>

In PyTorch · `AdamW(..., weight_decay=0.1)` — the one line you need.

---

# L1 · the sparsity-inducing sibling

Add $\lambda \|\theta\|_1$ → encourages many weights to be exactly 0.

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

### SESSION 1 · PART 3

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

### SESSION 1 · PART 4

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

### SESSION 1 · PART 5

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

One flag in PyTorch: `CrossEntropyLoss(label_smoothing=0.1)`.

</div>

---

<!-- _class: section-divider -->

### SESSION 2

# Architectural regularization

Dropout + Normalization

---

# Session 2 · what we will cover

Two architectural regularizers that ship inside the network:

1. **Dropout** — randomly silence neurons during training.
2. **Normalization** (BatchNorm / LayerNorm / RMSNorm) — re-center and re-scale activations.
3. **Placement** — pre-norm vs post-norm.

All 2026-relevant. All in every modern architecture.

---

<!-- _class: section-divider -->

### SESSION 2 · PART 6

# Dropout

An implicit ensemble, one line of code

---

# The idea (Hinton 2012)

Every forward pass during training:

1. Sample a random binary mask $\mathbf{m}_i \sim \text{Bernoulli}(p)$ for each hidden unit $i$.
2. Multiply: $\mathbf{h}_\text{drop} = \mathbf{h} \odot \mathbf{m}$.
3. Scale surviving units by $1/p$ to keep expected activation the same.

At eval time, **turn dropout off** — use all units.

---

# Dropout in one picture

![w:920px](figures/lec06/svg/dropout_network.svg)

<div class="realworld">

▶ Interactive: slide <code>p</code> and watch a small network flicker; toggle train/eval mode — [dropout-playground](https://nipunbatra.github.io/interactive-articles/dropout-playground/).

</div>

---

# Two intuitions for why it helps

<div class="columns">
<div>

### Ensemble view

Each mini-batch trains a *different* sub-network (a subset of units).

Over many batches you are implicitly training an **ensemble of $2^N$ thinned networks**, all sharing weights.

At test time, no mask → like averaging the ensemble.

</div>
<div>

### Co-adaptation view

Without dropout, units **co-adapt** — neuron $j$ relies on neuron $k$ being alive to do its job.

Dropout forces every unit to be useful *on its own* → more distributed representation.

</div>
</div>

---

# Dropout · different masks per pass

![w:920px](figures/lec06/svg/dropout_masks.svg)

---

# Inverted dropout — why divide by $p$

<div class="math-box">

$$\mathbf{h}_\text{drop} = \frac{\mathbf{h} \odot \mathbf{m}}{p}, \quad \mathbf{m} \sim \text{Bernoulli}(p)$$

During training, $E[\mathbf{h}_\text{drop}] = \mathbf{h}$ — same expectation as the full forward pass.

At eval, set $\mathbf{m} = \mathbf{1}$, no scaling — output matches expectation.

</div>

Without the $1/p$ rescale, training-time and eval-time activations would differ. This is the "inverted" part — the form PyTorch uses.

---

# Dropout · worked numeric example

Suppose hidden activations · `h = [2.0, 1.5, 0.5, 3.0]` and we use `p = 0.5` keep-prob.

<div class="math-box">

**Train pass.** Sample mask `m = [1, 0, 1, 0]` (Bernoulli p=0.5).
$\mathbf{h}_\text{drop} = (h \odot m) / p = [2.0, 0, 0.5, 0] / 0.5 = [\mathbf{4.0}, 0, \mathbf{1.0}, 0]$
(the kept units are *amplified* to compensate)

**Expected value** · $E[\mathbf{h}_\text{drop}] = p \cdot (h/p) + (1-p) \cdot 0 = h = [2.0, 1.5, 0.5, 3.0]$
↑ same as the no-dropout output.

**Eval pass.** $\mathbf{h}_\text{eval} = h = [2.0, 1.5, 0.5, 3.0]$. No mask, no scaling.

</div>

The training-time scaling-up by $1/p$ is what lets us drop the mask at eval time without changing the network's expected output.

---

# Dropout · the basketball-team analogy

<div class="keypoint">

Imagine training a basketball team where, in any given practice drill, some players randomly sit out. No one can rely too much on the star player · she might not be there.

Result · everyone becomes more versatile. The team performs more reliably with any subset on the court.

</div>

That's what dropout does to neurons · it prevents them from **co-adapting** (relying too heavily on a few specific neighbors). Each neuron has to become individually useful.

---

# Dropout in PyTorch

```python
self.drop = nn.Dropout(p=0.1)     # typical: 0.1 for Transformers
                                   #          0.5 for MLP hidden layers
                                   #          0.0 for CNNs (usually)

def forward(self, x):
    h = F.relu(self.fc1(x))
    h = self.drop(h)               # apply after the activation
    h = self.fc2(h)
    return h
```

<div class="realworld">

`model.train()` and `model.eval()` toggle it automatically. Forgetting the mode switch is a classic bug.

</div>

<div class="warning">

**Convention mismatch warning** · in lecture math, $p$ often denotes **keep** probability (Bernoulli$(p)$). PyTorch's `nn.Dropout(p)` uses $p$ as the **drop** probability. Keep-prob 0.8 ↔ `nn.Dropout(p=0.2)`. Always double-check.

</div>

---

# Where dropout lives in 2026

| Architecture | Typical use |
|--------------|-------------|
| Large MLPs | `p = 0.5` between hidden layers |
| CNNs | usually absent (BN + aug is enough) |
| RNNs / LSTMs | variational dropout (same mask per timestep) |
| Transformers | `p = 0.1` after attention and FFN |
| Fine-tuning an LLM | `p = 0.0` or very small — data is scarce |

<div class="insight">

Dropout was the biggest regularization breakthrough of 2012. Today it is overshadowed by BN + LN + augmentation for many tasks, but still in every Transformer.

</div>

---

<!-- _class: section-divider -->

### SESSION 2 · PART 7

# Normalization

Same family, three flavours, one knob at a time

---

# Hiker in a canyon · why normalization matters

<div class="keypoint">

Imagine a hiker descending a long, narrow, steep-sided canyon. They bounce side-to-side, making slow progress along the canyon's length.

A round bowl is much easier · the hiker walks straight to the bottom.

Normalization **reshapes** the loss landscape from a canyon into a bowl · same minimum, much easier optimizer trajectory.

</div>

Concretely · BN/LN keep activations centered and unit-scale, which means the loss's curvature in different directions is roughly equal. The optimizer takes confident, direct steps.

---

# Why normalize at all?

Two problems that normalization fixes:

1. **Scale drift across layers.** Activations grow or shrink with depth. He init addresses this at *init*; normalization does it at *every step*.
2. **Internal covariate shift (original claim).** Distribution of layer inputs changes during training. This explanation turned out to be partly wrong — see next slide.

---

# BN · LN · RMSNorm · the axes

![w:920px](figures/lec06/svg/bn_vs_ln_vs_rms.svg)

---

# BatchNorm · train vs eval modes

![w:920px](figures/lec06/svg/bn_train_eval.svg)

---

# BatchNorm · worked numeric example

<div class="math-box">

Mini-batch of 4 activations · $x = [1, 3, 5, 7]$.

**Step 1** · mean $\mu = (1 + 3 + 5 + 7) / 4 = 4.0$
**Step 2** · variance $\sigma^2 = \frac{1}{4}((-3)^2 + (-1)^2 + 1^2 + 3^2) = 5.0$ → $\sigma \approx 2.236$
**Step 3** · normalize · $\hat x = (x - \mu) / \sigma = [-1.34, -0.45, 0.45, 1.34]$
**Step 4** · scale + shift with learned $\gamma, \beta$ · suppose $\gamma = 2.0, \beta = 0.5$

$$y = \gamma \hat x + \beta = [-2.18, -0.40, 1.40, 3.18]$$

</div>

At eval time · use the running mean/var collected during training, not the batch statistics. This is why `model.eval()` matters · it switches to running stats.

---

# The ICS debate

Ioffe &amp; Szegedy 2015 · BN helps by reducing **internal covariate shift** (ICS) — the changing distribution of layer inputs during training.

**Santurkar et al. 2018** — showed ICS was largely a red herring:

<div class="insight">

BN's real benefit is that it **smooths the loss landscape** — makes gradients more predictable, enabling larger learning rates.

You don't need to remember this. You *do* need to remember: BN works, but the *reason* it works is more subtle than the original paper claimed.

</div>

---

# BatchNorm in PyTorch

```python
# For 2D / FC: nn.BatchNorm1d
# For 4D / conv: nn.BatchNorm2d

layer = nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),          # ← after the conv, before ReLU
    nn.ReLU(),
)
```

- Two learnable parameters per channel · $\gamma$ (scale), $\beta$ (shift)
- Two buffers per channel · `running_mean`, `running_var`
- Initialized to identity · $\gamma = 1, \beta = 0$

---

# When BatchNorm fails

<div class="warning">

**Small batch sizes** — stats are noisy, BN hurts more than it helps. `batch_size < 32` → prefer GroupNorm or LayerNorm.

**Sequence models** — variable-length sequences have inconsistent statistics along the batch axis.

**Online / streaming** — can't collect meaningful batch stats.

**Distributed training** — each replica computes its own batch stats unless you use SyncBN.

</div>

These are exactly the situations that birthed LayerNorm.

---

# LayerNorm · fix for sequences

Normalize across the **feature dimension** instead of the batch dimension.

$$\hat{x}_i = \frac{x_i - \mu_\text{sample}}{\sqrt{\sigma^2_\text{sample} + \epsilon}}$$

No dependence on batch size or other samples.

```python
norm = nn.LayerNorm(d_model)    # applied at every Transformer block
```

<div class="keypoint">

Every modern Transformer (BERT, GPT, Llama, Claude) uses LayerNorm or its cheaper cousin RMSNorm — *not* BatchNorm.

</div>

---

# RMSNorm · the cheap modern cousin

Drop the mean subtraction — keep only the scale:

<div class="math-box">

$$\hat{x}_i = \frac{x_i}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$$

</div>

- ~15–30% cheaper than LayerNorm.
- Empirically no worse at convergence.
- Used in Llama, Mistral, PaLM, and most open LLMs from 2023 onward.

```python
# PyTorch 2.4+
norm = nn.RMSNorm(d_model)
```

---

<!-- _class: section-divider -->

### SESSION 2 · PART 8

# Where to put the norm

Pre-norm vs post-norm

---

# The placement matters

![w:920px](figures/lec06/svg/pre_vs_post_norm.svg)

---

# Why pre-norm won

Post-norm (original Transformer) puts `LayerNorm` **after** the residual addition. The gradient flowing through the skip is modulated by the norm — gradient can vanish at depth → needs aggressive warmup + careful tuning.

Pre-norm puts `LayerNorm` **before** the sublayer, leaving the skip connection untouched. Gradient has a clean residual highway.

<div class="realworld">

Pre-norm is the default for every modern Transformer (GPT-2 onwards). If you're building a new Transformer in 2026, use pre-norm.

</div>

---

# Decision table · which norm · which model

| Architecture | Normalization | Why |
|--------------|---------------|-----|
| CNN for images | **BatchNorm** | large batches, fixed shapes |
| Small-batch CNN (`< 32`) | **GroupNorm** | batch-independent |
| Transformer | **LayerNorm** (pre-norm) | batch/seq-length independent |
| Modern LLM (Llama, Mistral) | **RMSNorm** (pre-norm) | cheaper, no loss of quality |
| RNN / LSTM | **LayerNorm** across features | same reason as Transformer |

---

# Decision rule in two sentences

<div class="keypoint">

**Fixed-size dense data with big batches → BatchNorm.**

**Anything else → LayerNorm (or RMSNorm if you want cheap).**

</div>

---

<!-- _class: section-divider -->

# Putting it together

The full regularization stack for 2026

---

# The stack for a real vision training run

```python
# 1. Architecture regularization
model = ResNet50(dropout=0.0)      # BN already in ResNet

# 2. Optimizer regularization (from L5)
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

# 4. Mixup inside the training loop (conditional)
# 5. Label smoothing in the loss
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 6. Early stopping — checkpoint on val loss
```

---

<!-- _class: summary-slide -->

# Lecture 6 — summary

### Session 1 · classical + data-centric

- **Double descent** — more params past the threshold can help; generalization in the overparameterized regime is the modern story.
- **L2 / L1 / early stopping** — you know these. Use `weight_decay` in AdamW; checkpoint on val.
- **Data augmentation** — single highest-value regularizer for vision. Must preserve labels.
- **Mixup / CutMix** — interpolate inputs *and* labels. Free ~1–2% accuracy.
- **Label smoothing** — softens hard targets; better calibration. One flag.

### Session 2 · architectural

- **Dropout** — implicit ensemble; inverted rescaling. `p = 0.1` for Transformers, `0.5` for MLPs.
- **BN · LN · RMSNorm** — same family, three axes. BN for CNNs, LN for Transformers, RMSNorm for LLMs.
- **Pre-norm &gt; post-norm** for deep Transformers.

---

<!-- _class: summary-slide -->

# Lecture 6 — what's next

### Read before Lecture 7

**Prince** — Ch 10 · *Convolutional networks*.

### Next lecture

**CNN deep dive** — brisk on convolution mechanics (prereq covered LeNet); deep on receptive fields, modern architectures, and inductive-bias framing.

<div class="notebook">

**Notebook 6a** · `06a-regularization-stack.ipynb` — sweep weight decay, RandAugment, label smoothing on CIFAR-10.
**Notebook 6b** · `06b-batchnorm-by-hand.ipynb` — implement BatchNorm1d forward + backward manually (Karpathy-style); verify against `nn.BatchNorm1d`.

</div>
