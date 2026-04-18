---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Dropout &amp; Normalization

## Lecture 7 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Recap · where we are

- **Classical regularization** · L2, L1, early stopping (L6 · brisk).
- **Data regularization** · augmentation, Mixup, label smoothing (L6).

Today, two **architectural** regularizers that ship inside the network:

1. **Dropout** — randomly silence neurons during training.
2. **Normalization** (BatchNorm / LayerNorm / RMSNorm) — re-center and re-scale activations.

<div class="paper">

Today maps to **UDL Ch 9** (dropout) and **Ch 11** (residuals + BatchNorm).

</div>

---

<!-- _class: section-divider -->

### PART 1

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

![w:920px](figures/lec07/svg/dropout_network.svg)

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

# Inverted dropout — why divide by $p$

<div class="math-box">

$$\mathbf{h}_\text{drop} = \frac{\mathbf{h} \odot \mathbf{m}}{p}, \quad \mathbf{m} \sim \text{Bernoulli}(p)$$

During training, $E[\mathbf{h}_\text{drop}] = \mathbf{h}$ — same expectation as the full forward pass.

At eval, set $\mathbf{m} = \mathbf{1}$, no scaling — output matches expectation.

</div>

Without the $1/p$ rescale, training-time and eval-time activations would differ, and the model would produce different outputs. This is the "inverted" part of inverted dropout — it is the form PyTorch uses.

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

### PART 2

# Normalization

Same family, three flavours, one knob at a time

---

# Why normalize at all?

Two problems that normalization fixes:

1. **Scale drift across layers.** Activations grow or shrink with depth. He init addresses this at *init*; normalization does it at *every step*.
2. **Internal covariate shift (original claim).** Distribution of layer inputs changes during training. This explanation turned out to be partly wrong — see next slide.

---

# BN · LN · RMSNorm · the axes

![w:920px](figures/lec07/svg/bn_vs_ln_vs_rms.svg)

---

# BatchNorm · train vs eval modes

![w:920px](figures/lec07/svg/bn_train_eval.svg)

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

### PART 3

# Where to put the norm

Pre-norm vs post-norm

---

# The placement matters

![w:920px](figures/lec07/svg/pre_vs_post_norm.svg)

---

# Why pre-norm won

Post-norm (original Transformer) puts `LayerNorm` **after** the residual addition. The gradient flowing through the skip is modulated by the norm — gradient can vanish at depth → needs aggressive warmup + careful tuning.

Pre-norm puts `LayerNorm` **before** the sublayer, leaving the skip connection untouched. Gradient has a clean residual highway.

<div class="realworld">

Pre-norm is the default for every modern Transformer (GPT-2 onwards). If you're building a new Transformer in 2026, use pre-norm.

</div>

---

<!-- _class: section-divider -->

### PART 4

# Decision table

Which normalization · which model

---

# When to use what

| Architecture | Normalization | Why |
|--------------|---------------|-----|
| CNN for images | **BatchNorm** | large batches, fixed shapes — BN thrives |
| Small-batch CNN (`< 32`) | **GroupNorm** | batch-independent alternative |
| Transformer (encoder or decoder) | **LayerNorm** (pre-norm) | batch/seq-length independent |
| Modern LLM (Llama, Mistral) | **RMSNorm** (pre-norm) | cheaper, no loss of quality |
| RNN / LSTM | **LayerNorm** across features | same reason as Transformer |

---

# Decision rule in two sentences

<div class="keypoint">

**Fixed-size dense data with big batches → BatchNorm.**

**Anything else → LayerNorm (or RMSNorm if you want cheap).**

</div>

---

<!-- _class: summary-slide -->

# Lecture 7 — summary

- **Dropout** — random mask during training; inverted rescaling keeps expectation constant. Still alive in every Transformer.
- **BatchNorm** — per-feature, stats over batch dimension. Two modes (train/eval) — forget to switch = bug.
- **LayerNorm** — per-sample, stats over feature dimension. Batch-independent. Standard for Transformers.
- **RMSNorm** — no mean subtraction. Cheaper. Standard for modern LLMs.
- **Pre-norm &gt; post-norm** for deep Transformers — clean gradient highway.
- **Decision rule** — fixed dense data + big batch → BN; anything else → LN / RMSNorm.

### Read before Lecture 8

**Prince** — Ch 10 · *Convolutional networks*.

### Next lecture

**CNN deep dive** — convolution mechanics, receptive fields, LeNet → AlexNet → VGG (brisk — prereq covered LeNet).

<div class="notebook">

**Notebook 7** · `07-dropout-batchnorm.ipynb` — implement BatchNorm1d forward + backward by hand (Karpathy-style); verify against `nn.BatchNorm1d`; compare Dropout vs no Dropout on overfitting.

</div>
