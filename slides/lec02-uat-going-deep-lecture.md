---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Universal Approximation & Going Deep

## Lecture 2 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. State the **Universal Approximation Theorem** and cite its caveats.
2. Explain why **depth beats width** in practice despite theoretical equivalence.
3. Diagnose **vanishing / exploding gradients** in deep nets.
4. Apply **residual connections** to train 100+ layer networks.
5. Pick **weight init** (Xavier / He) based on activation.
6. Articulate three **practical limits** UAT does not address.

---

# Recap · where we left off

- **MLPs** are stacks of affine + non-linearity.
- **Backprop** is chain rule run right-to-left.
- **Sigmoids vanish;** ReLU un-blocks depth.
- **One hidden layer can approximate anything** — we need to make that precise.

<div class="paper">

Today maps to **UDL Ch 4** (deep networks), **Ch 7** (gradients &amp; init), **Ch 11** (residual networks). Read these three chapters before or after — whichever works for you.

</div>

Three questions for today:

1. What does *"can approximate anything"* actually mean?
2. If one layer suffices, why are SOTA models 100+ layers deep?
3. How do we train deep nets without the gradient evaporating?

---

<!-- _class: section-divider -->

### PART 1

# Universal approximation

What a single hidden layer can — and can't — do

---

# Build a bump from two ReLUs

![w:900px](figures/lec02/svg/uat_bump_construction.svg)

<div class="realworld">

▶ Interactive: grow a 1-hidden-layer net and watch it fit a target curve — [universal-approximation](https://nipunbatra.github.io/interactive-articles/universal-approximation/).

</div>

---

# UAT · the formal statement

<div class="math-box">

**Theorem** (Cybenko 1989 · Hornik 1991 · Leshno 1993)

For any continuous $f: [0,1]^d \to \mathbb{R}$ and any $\epsilon > 0$, there exist $N$, weights $\{\mathbf{w}_i, b_i, \alpha_i\}$ such that

$$\left| f(\mathbf{x}) - \sum_{i=1}^{N} \alpha_i \, \sigma(\mathbf{w}_i^\top \mathbf{x} + b_i) \right| < \epsilon$$

for any non-polynomial activation $\sigma$ — including ReLU.

</div>

One hidden layer suffices. The catch hides in one word: **exist.**

---

# The price of width — curse of dimensionality

Piecewise-linear approximation of $f$ to error $\epsilon$:

- 1D: $N \approx O(1 / \sqrt{\epsilon})$
- $d$D: $N \approx O(1 / \epsilon^{d/2})$

| $d$ | $\epsilon = 0.01$ | neurons |
|-----|-------------------|---------|
| 1   |  | ~10 |
| 10  |  | $10^{10}$ |
| 100 |  | $10^{100}$ |

<div class="warning">

UAT says good weights *exist*. Not that SGD finds them. Not that the network generalizes. Not that $N$ is reasonable.

</div>

---

# Three things UAT does *not* promise

<div class="columns">
<div>

**Learnability.**
Existence of good weights ≠ SGD finding them.

**Width.**
The bound on $N$ can be astronomical.

</div>
<div>

**Generalization.**
A network that memorizes $n$ training points also satisfies UAT. Works on train, fails on test.

</div>
</div>

<div class="popquiz">

**Pop quiz.** True or false: UAT guarantees a 1-hidden-layer net will perform well on unseen data, given enough neurons and data.

</div>

---

# Pop quiz · answer

**False** — for three independent reasons.

1. Existence ≠ findability.
2. "Enough neurons" can mean exponentially many.
3. UAT says *nothing* about generalization.

In practice, depth is far more parameter-efficient than width.

---

<!-- _class: section-divider -->

### PART 2

# Depth vs width

Why deeper is (usually) better

---

# Shallow enumerates, deep reuses

![w:900px](figures/lec02/svg/depth_compositionality.svg)

---

# A parameter-budget exercise

**Q.** On a $784 \to \,?\, \to 10$ classifier:

- **Wide shallow:** 1 hidden layer of 2048 units
- **Deep narrow:** 8 hidden layers of 128 units

Which has more parameters?

---

# Answer — and the twist

| Architecture | Parameters |
|--------------|-----------|
| $784 \to 2048 \to 10$ | ~1.63 M |
| $784 \to 128^{\times 8} \to 10$ | ~0.22 M |

Shallow-wide has **7× more** parameters, but deep-narrow typically wins on natural data.

<div class="insight">

Parameter count is a crude proxy for capacity. What depth gives you, width alone cannot: **reusable hierarchy.**

</div>

---

# Parity — the canonical case

![w:900px](figures/lec02/svg/parity_tree.svg)

---

# Why depth helps for parity

Parity is **recursive** — XOR of pairs, then XOR of pair-pairs, then XOR of those.

Depth matches the structure of the problem:

- **$\log n$ layers**, $O(n)$ gates.
- Shallow has no hierarchy to exploit — it must enumerate every pattern.

<div class="paper">

Formal proof: Telgarsky, *"Benefits of Depth in Neural Networks,"* COLT 2016. We take the statement on faith today.

</div>

---

# Enough theory — can we just stack layers?

**Q.** If depth is so great, should we train a 500-layer plain MLP?

Give an honest answer; then read on.

---

# Two things break

<div class="warning">

**Problem 1 — vanishing gradients** (from L1).
Product of many sub-1 Jacobians → zero.

**Problem 2 — the degradation problem.**
Something strictly more surprising. Next section.

</div>

---

<!-- _class: section-divider -->

### PART 3

# Vanishing gradients · the full picture

Before we fix depth, let's see it break

---

# The chain rule is a product

![w:900px](figures/lec02/svg/chain_rule_product.svg)

---

# Sigmoid's fatal ceiling · 0.25

![w:900px](figures/lec02/svg/sigmoid_gradient_stack.svg)

<div class="realworld">

▶ Interactive: stack sigmoids and watch the gradient evaporate with depth — [vanishing-gradients](https://nipunbatra.github.io/interactive-articles/vanishing-gradients/).

</div>

---

# Fix #1 · the ReLU family

![w:900px](figures/lec02/svg/relu_family.svg)

---

<!-- _class: section-divider -->

### PART 4

# The degradation problem & ResNets

The most important architectural idea since backprop

---

# The He et al. (2015) experiment

Train plain CNNs of increasing depth on CIFAR-10 — same optimizer, same init, just more layers.

**Q.** What do you expect? More layers → more capacity → better, right?

---

# The surprise

![w:920px](figures/lec02/svg/degradation_problem.svg)

---

# Why this should bother you

<div class="insight">

If this were overfitting, training error would **drop** with depth (more capacity to memorize) and test error would **rise** (poor generalization).

Instead, training error went *up*. The deeper net cannot even fit the training data.

This is an **optimization** problem, not a capacity problem.

</div>

---

# The thought experiment

![w:920px](figures/lec02/svg/identity_thought_exp.svg)

---

# ResNet · the key insight

<div class="keypoint">

**He et al. 2015** — don't ask the block to learn the full mapping $H(\mathbf{x})$. Ask it to learn the **residual**:

$$\mathcal{F}(\mathbf{x}) = H(\mathbf{x}) - \mathbf{x} \;\Longrightarrow\; H(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

If the optimum is close to identity, we only need $\mathcal{F} \approx \mathbf{0}$ — which SGD finds trivially.

</div>

---

# Why residuals are easier

- **Weight decay** pushes $\mathcal{F}$ toward zero.
- **Init near zero** starts $\mathcal{F} \approx \mathbf{0}$.
- **Small SGD updates** keep it there unless signal says otherwise.

Coordinating non-linear layers to produce *exact* identity is the opposite: a delicate balance with no prior.

**The skip connection turns a hard default into a free default.**

---

# The residual block

![w:900px](figures/lec02/svg/resnet_block.svg)

---

# Skip connections fix gradient flow

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x} \;\Longrightarrow\; \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I}$$

Even if $\partial \mathcal{F}/\partial \mathbf{x}$ collapses to zero, the identity $\mathbf{I}$ survives — a direct path back to every early layer.

---

# Gradient highway · plain vs ResNet

![w:920px](figures/lec02/svg/gradient_highway.svg)

---

# ResNet in PyTorch · 12 lines

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)    # ← the skip
```

**Q.** What constraint does `self.block(x) + x` impose on dimensions?

---

# Empirical impact

| Year | Model | Depth | ImageNet top-5 |
|------|-------|-------|----------------|
| 2012 | AlexNet | 8 | 16.4% |
| 2014 | VGG-19 | 19 | 7.3% |
| 2014 | GoogLeNet | 22 | 6.7% |
| **2015** | **ResNet-152** | **152** | **3.6%** |

Skip connections are now in virtually every modern architecture — CNNs, Transformers, diffusion U-Nets.

---

<!-- _class: section-divider -->

### PART 5

# Initialization

From first principles

---

# The goal

Keep activations — and gradients — at roughly **constant variance** across layers.

- Variance grows → exploding activations.
- Variance shrinks → vanishing activations.

![w:900px](figures/lec02/svg/init_landscape.svg)

---

# Forward-pass variance

Layer: $y = \sum_{i=1}^{n_\text{in}} w_i\, x_i$. Assume $w_i, x_i$ independent, zero-mean.

For independent zero-mean $A, B$:

$$\text{Var}(AB) = \text{Var}(A)\, \text{Var}(B)$$

Therefore:

$$\text{Var}(y) = n_\text{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$

To preserve variance we need $n_\text{in} \cdot \text{Var}(w) = 1$.

---

# Xavier · for sigmoid/tanh

Forward: $\text{Var}(w) = 1/n_\text{in}$.
Backward: $\text{Var}(w) = 1/n_\text{out}$.

<div class="math-box">

**Xavier / Glorot (2010)** — compromise:

$$W \sim \mathcal{N}\!\left(0, \frac{2}{n_\text{in} + n_\text{out}}\right)$$

</div>

---

# He · for ReLU

ReLU zeros half the pre-activations:

$$E[h^2] = \tfrac{1}{2}\, \text{Var}(y)$$

To keep this constant across layers:

<div class="math-box">

**He / Kaiming (2015)**:

$$W \sim \mathcal{N}\!\left(0, \frac{2}{n_\text{in}}\right)$$

Factor of 2 compensates the ReLU halving.

</div>

---

# Initialization in PyTorch

```python
# Default for nn.Linear — Kaiming uniform (sensible for ReLU)
model = nn.Linear(784, 256)

# Explicit He init
nn.init.kaiming_normal_(model.weight, mode='fan_in', nonlinearity='relu')
nn.init.zeros_(model.bias)

# Explicit Xavier (for sigmoid/tanh)
nn.init.xavier_normal_(model.weight)
```

<div class="insight">

**Rule of thumb** · ReLU family → He, sigmoid / tanh → Xavier.

</div>

---

<div class="popquiz">

**Pop quiz.** You build a 10-layer MLP with **Tanh** activations and **He** initialization. Loss oscillates, activations saturate. Why?

</div>

---

# Pop quiz · answer

He doubles variance to compensate for ReLU's halving. Tanh doesn't halve — He gives **too large** a variance → activations saturate at $\pm 1$ → gradients die.

**Fix:** Xavier.

| Activation | Init |
|-----------|------|
| ReLU, Leaky | He |
| Sigmoid, Tanh | Xavier |
| GELU, SiLU | He (convention) |

---

<!-- _class: summary-slide -->

# Lecture 2 — summary

- **UAT** is an existence theorem. Width can be exponential; depth is the practical knob.
- **Compositionality** (and parity) shows depth can replace exponential width.
- **Vanishing gradients** come from products of sub-1 Jacobians; ReLU unlocks depth by giving gradient 1 on the active side.
- **Degradation problem** — plain deep nets *train* worse.
- **ResNets** — $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$. Identity-in-the-Jacobian gradient highway + smoother landscape.
- **Xavier / He** — both derived from variance preservation.

### Read before Lecture 3

**Prince** — Ch 4, Ch 11. Free at [udlbook.github.io](https://udlbook.github.io/udlbook/).

### Next lecture

Tensors, autograd, `nn.Module`, `DataLoader`, the full training recipe, debugging ladder, error analysis.

<div class="notebook">

**Notebook 2** · `02-depth-and-resnets.ipynb` — shallow-wide vs deep-narrow on spirals; build a residual block; visualize gradient norms across depth.

</div>
