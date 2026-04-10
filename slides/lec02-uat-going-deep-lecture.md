---
marp: true
theme: dl-theme
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Universal Approximation & Going Deep

## Lecture 2 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

<!-- _class: section-divider -->

# Part 1: Universal Approximation Theorem

What can a single hidden layer do?

---

# Recall from Last Lecture

We said: a single hidden layer MLP can approximate *any* continuous function.

**Today we'll make this precise:**
- What does the theorem actually say?
- How does the approximation work (mechanically)?
- What are the limitations?

---

# UAT: Formal Statement

<div class="math-box">

**Theorem** (Cybenko, 1989; Hornik, 1991):

Let $\sigma$ be a non-constant, bounded, continuous activation function. For any continuous function $f: [0,1]^d \to \mathbb{R}$ and any $\epsilon > 0$, there exist $N$, weights $w_i$, biases $b_i$, and coefficients $\alpha_i$ such that:

$$\left| f(\mathbf{x}) - \sum_{i=1}^{N} \alpha_i \, \sigma(\mathbf{w}_i^\top \mathbf{x} + b_i) \right| < \epsilon \quad \forall \mathbf{x} \in [0,1]^d$$

</div>

In words: one hidden layer with $N$ neurons, for large enough $N$, can get $\epsilon$-close to any continuous function.

---

# Let's Unpack This

The network is:

$$g(\mathbf{x}) = \sum_{i=1}^{N} \alpha_i \, \sigma(\mathbf{w}_i^\top \mathbf{x} + b_i)$$

**Q.** What does each term $\sigma(\mathbf{w}_i^\top \mathbf{x} + b_i)$ look like geometrically, in 1D?

---

# Each Neuron is a Shifted, Scaled Activation

In 1D: $\sigma(w \cdot x + b)$

- $w$ controls the **slope** (steepness)
- $b$ controls the **shift** (position)
- $\alpha$ controls the **weight** (how much it contributes)

For sigmoid: each neuron is a smooth step, positioned somewhere along the $x$-axis.

**Q.** How do we build a "bump" (localized peak) from these steps?

---

# Building Bumps from Steps

A bump = two shifted steps with opposite signs:

$$\text{bump}(x) = \alpha \left[\sigma(w(x - a)) - \sigma(w(x - b))\right]$$

This creates a localized peak between $a$ and $b$.

**Many bumps** at different positions → approximate any smooth curve.

---

# How ReLU Networks Approximate

With ReLU, each neuron gives a "kink":

$\max(0, wx + b)$

Two ReLUs → a tent/bump. Many bumps → piecewise-linear approximation.

![w:900px](figures/lec02/uat_relu_bumps.png)

---

# Approximation with Increasing Neurons

![w:950px](figures/lec02/uat_approximation.png)

More neurons → finer bumps → better fit. UAT guarantees we can get $\epsilon$-close.

---

# A Concrete Example

**Q.** Suppose we want to approximate $f(x) = \sin(x)$ on $[0, 2\pi]$ with a 1-hidden-layer ReLU network. How many neurons?

---

# A Concrete Example

Piecewise-linear approximation of $\sin(x)$ with error $< \epsilon$:

$$N \approx O\left(\frac{1}{\sqrt{\epsilon}}\right)$$

For $\epsilon = 0.01$: $N \approx 10$ neurons. Manageable!

**But in $d$ dimensions:**

$$N \approx O\left(\frac{1}{\epsilon^{d/2}}\right)$$

For $d = 100$, $\epsilon = 0.01$: $N \approx 10^{100}$.

This is the **curse of dimensionality** — the number of neurons explodes exponentially with dimension.

---

# Three Things UAT Does NOT Tell You

1. **How to find the weights.**

   UAT says good weights *exist*. It doesn't say gradient descent will find them.

---

# Three Things UAT Does NOT Tell You

2. **How many neurons you need.**

   Could be astronomically many. The theorem gives existence, not a bound.

---

# Three Things UAT Does NOT Tell You

3. **Anything about generalization.**

   A network that perfectly memorizes $n$ training points also satisfies UAT. But it may fail completely on new data.

---

<div class="popquiz">

**Pop Quiz**: True or False?

*"The Universal Approximation Theorem guarantees that a 1-hidden-layer network will perform well if given enough data and enough neurons."*

</div>

---

# Pop Quiz Answer

**False.**

- UAT guarantees *approximation ability*, not *learnability via gradient descent*
- More neurons doesn't mean gradient descent will find good weights
- "Enough data" is not part of the UAT statement at all
- In practice, **depth is far more efficient than width** (Part 2)

---

<!-- _class: section-divider -->

# Part 2: Depth vs Width

Why deeper is (usually) better

---

# The Depth Advantage

UAT says: width alone suffices (in theory).

**Q.** Then why would we ever want more layers?

---

# Compositionality

<div class="insight">

**Key idea**: Many real-world functions have **hierarchical structure**. Deep networks exploit this; shallow networks cannot.

</div>

Consider recognizing "a dog sitting" in an image:

---

# Compositionality: Shallow vs Deep

![w:950px](figures/lec02/depth_compositionality.png)

Deep: learn 11 reusable parts. Shallow: learn 9 separate detectors.

Now add "cat flying" — deep network reuses existing parts; shallow needs a brand new detector.

---

# Counting Parameters

**Q.** Which has more parameters: 1 layer of 512 neurons, or 8 layers of 64 neurons?

(Input: 784, Output: 10)

---

# Counting Parameters

![w:800px](figures/lec02/param_count_depth_width.png)

The shallow-wide network has **far more** parameters, yet deep-narrow networks often perform better. Parameters alone don't determine quality.

---

# Depth Separation: A Formal Result

<div class="math-box">

**Theorem** (Telgarsky, 2016): There exist functions computable by a depth-$k$ network with polynomial width, that require **exponential width** to approximate with depth $O(k^{1/3})$.

</div>

In other words: for some functions, removing depth costs you exponentially in width.

<div class="paper">

Telgarsky, *"Benefits of Depth in Neural Networks"*, COLT 2016

</div>

---

# A Classic Example: Parity

$f(x_1, \ldots, x_n) = x_1 \oplus x_2 \oplus \cdots \oplus x_n$ (XOR over $n$ bits)

- **Shallow**: needs $O(2^n)$ neurons
- **Deep**: needs $O(n)$ neurons in $O(\log n)$ layers

**Q.** Why does depth help for parity?

---

# Why Depth Helps for Parity

Parity is **recursive**:

$$x_1 \oplus x_2 \oplus \cdots \oplus x_n = (x_1 \oplus x_2) \oplus (x_3 \oplus x_4) \oplus \cdots$$

A deep network computes XOR of *pairs* in layer 1, XOR of *pairs of pairs* in layer 2, etc. Binary tree structure → $O(\log n)$ depth, $O(n)$ neurons.

A shallow network cannot decompose the problem — it must enumerate all $2^n$ input patterns.

---

# The Circuit Complexity View

Think of a network as a **circuit** (Boolean or arithmetic):

| Function | Shallow (1 layer) | Deep |
|----------|-------------------|------|
| AND of $n$ variables | $O(2^n)$ gates | $O(n)$ gates |
| Parity of $n$ bits | $O(2^n)$ gates | $O(n)$ gates, $O(\log n)$ depth |
| Degree-$k$ polynomial | $O(n^k)$ neurons | $O(k \log n)$ neurons |

Depth enables compositional computation. This is the fundamental reason deep learning works.

---

# Empirical Evidence

![w:900px](figures/lec02/depth_vs_width.png)

With similar parameter budgets, deeper networks capture multi-scale structure that shallow networks miss.

---

# So: Just Add More Layers?

**Q.** If depth is so great, why not use 500 layers with a plain MLP?

---

# So: Just Add More Layers?

**Q.** What goes wrong with very deep plain networks?

Two problems:
1. **Vanishing gradients** (Lecture 1) — gradients decay exponentially
2. **Degradation problem** (next part) — something more surprising

---

<!-- _class: section-divider -->

# Part 3: The Degradation Problem & ResNets

The most important idea in deep network design

---

# An Experiment (He et al., 2015)

Train plain networks of increasing depth on CIFAR-10.

**Q.** What do you expect?

Intuition: deeper = more parameters = more capacity = better. Right?

---

# The Surprise

![w:900px](figures/lec02/resnet_vs_plain.png)

**Both training and test error increase** for the deeper plain network.

---

# Why This Is Surprising

<div class="warning">

If it were overfitting, we'd see:
- Training error: lower (deeper has more capacity)
- Test error: higher (memorizes training data)

But training error **also goes up**. The deeper network can't even fit the training data as well!

</div>

This is **not** a capacity problem. It's an **optimization** problem.

---

# A Thought Experiment

**Q.** A 20-layer network achieves 5% error. We add 36 identity layers on top (total: 56 layers). What should happen?

---

# A Thought Experiment

**Q.** What should happen with 36 added identity layers?

The 56-layer network should get **at most 5% error** — the extra layers just pass data through.

$$\mathbf{h}_{l+1} = \mathbf{h}_l \quad \text{(identity mapping)}$$

**But in practice**, learning the identity through nonlinear layers is hard. Standard networks don't find this solution.

---

# ResNet: The Key Insight

<div class="keypoint">

**He et al. (2015)**: Instead of learning $H(\mathbf{x})$ directly, learn the **residual**:

$$\mathcal{F}(\mathbf{x}) = H(\mathbf{x}) - \mathbf{x}$$

Then: $H(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$

If the optimal mapping is close to identity, we only need $\mathcal{F}(\mathbf{x}) \approx \mathbf{0}$.

</div>

**Q.** Why is learning $\mathcal{F} \approx \mathbf{0}$ easier than learning $H \approx \text{identity}$?

---

# Why Residuals Are Easier

**Q.** Why is $\mathcal{F} \approx \mathbf{0}$ easier than $H \approx \text{identity}$?

Pushing weights toward zero is natural — weight decay, initialization near zero, and SGD dynamics all favor small weights. Learning the identity mapping $H(\mathbf{x}) = \mathbf{x}$ through several nonlinear layers requires careful coordination of weights.

---

# The Residual Block

![w:850px](figures/lec02/resnet_block.png)

---

# Skip Connections Fix Gradient Flow

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

**Q.** What is $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$?

---

# Skip Connections Fix Gradient Flow

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I}$$

The gradient through the skip path is always $\mathbf{I}$ — it cannot vanish!

Even if $\frac{\partial \mathcal{F}}{\partial \mathbf{x}} \to 0$, the total gradient $\to \mathbf{I}$.

This creates a **gradient highway** from the loss all the way back to early layers.

---

# Gradient Flow: Plain vs ResNet

![w:950px](figures/lec02/gradient_flow_comparison.png)

---

# Loss Landscape Smoothing

![w:950px](figures/lec02/loss_landscape_skip.png)

Skip connections dramatically smooth the optimization landscape.

<div class="paper">

Li et al., *"Visualizing the Loss Landscape of Neural Nets"*, NeurIPS 2018

</div>

---

# ResNet in PyTorch

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)  # <-- skip connection
```

**Q.** What constraint does the skip connection $+ \mathbf{x}$ impose on the layer dimensions?

---

# ResNet in PyTorch

**Q.** What constraint does $+ \mathbf{x}$ impose?

The input and output of the residual block must have the **same dimension**. Otherwise $\mathcal{F}(\mathbf{x}) + \mathbf{x}$ is not defined.

When dimensions change (e.g., 256 → 512), we use a **projection shortcut**:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{W}_s \mathbf{x}$$

where $\mathbf{W}_s$ is a linear projection (1×1 convolution in CNNs).

---

# Impact of ResNets

| Year | Architecture | Depth | ImageNet Top-5 |
|------|-------------|-------|----------------|
| 2012 | AlexNet | 8 | 16.4% |
| 2014 | VGG | 19 | 7.3% |
| 2014 | GoogLeNet | 22 | 6.7% |
| **2015** | **ResNet** | **152** | **3.6%** |

ResNets made **hundreds of layers** trainable. The skip connection is used in virtually every modern architecture — CNNs, Transformers, diffusion models.

<div class="paper">

He et al., *"Deep Residual Learning for Image Recognition"*, CVPR 2016 (12,000+ citations)

</div>

---

# Other Skip Connection Designs

<div class="columns">
<div>

### DenseNet (2017)

Each layer receives features from **all** previous layers:

$\mathbf{h}_l = f([\mathbf{h}_0, \mathbf{h}_1, \ldots, \mathbf{h}_{l-1}])$

Concatenation, not addition.

Advantage: maximum feature reuse.

</div>
<div>

### Highway Networks (2015)

Learned gating:

$\mathbf{y} = T \odot H(\mathbf{x}) + (1 - T) \odot \mathbf{x}$

where $T = \sigma(\mathbf{W}_T \mathbf{x} + \mathbf{b}_T)$.

Predates ResNet! But ResNet's simpler addition works as well or better.

</div>
</div>

---

<div class="popquiz">

**Pop Quiz**: You're training a 50-layer plain MLP and the loss barely decreases. What do you try first?

(a) Use sigmoid instead of ReLU
(b) Add skip connections
(c) Use a smaller learning rate
(d) Add more layers

</div>

---

# Pop Quiz Answer

**(b) Add skip connections.**

- (a) Makes it worse — sigmoid has max gradient 0.25
- (b) Correct — provides gradient highway to early layers
- (c) Might help stability, but doesn't fix vanishing gradients
- (d) Makes the degradation problem worse

---

<!-- _class: section-divider -->

# Part 4: Initialization — The Full Story

Deriving Xavier and He from first principles

---

# The Goal

We want activations to have roughly **constant variance** across layers.

**Q.** If layer $l$ has variance $V$, what is the variance after passing through layer $l+1$?

Let's work this out.

---

# Variance Analysis: Forward Pass

Layer: $y = \sum_{i=1}^{n_{in}} w_i x_i$ (no bias, no activation)

Assume $w_i$ and $x_i$ are independent, zero mean.

$$\text{Var}(y) = \sum_{i=1}^{n_{in}} \text{Var}(w_i x_i)$$

**Q.** For independent zero-mean variables, $\text{Var}(AB) = ?$

---

# Variance of a Product

For independent zero-mean random variables:

$$\text{Var}(AB) = \text{Var}(A) \cdot \text{Var}(B)$$

(This follows from $E[A^2 B^2] = E[A^2]E[B^2]$ and $E[AB] = E[A]E[B] = 0$.)

Therefore:

$$\text{Var}(y) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$

---

# Xavier Initialization

To preserve variance: $\text{Var}(y) = \text{Var}(x)$

$$n_{in} \cdot \text{Var}(w) = 1 \implies \text{Var}(w) = \frac{1}{n_{in}}$$

Backward pass analysis gives: $\text{Var}(w) = \frac{1}{n_{out}}$

<div class="math-box">

**Xavier (Glorot & Bengio, 2010)**: Compromise:

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

</div>

---

# He Initialization

With ReLU: $h = \max(0, y)$

For $y \sim \mathcal{N}(0, \sigma^2)$, half the values are zeroed:

$$\text{Var}(\text{ReLU}(y)) = \frac{1}{2} \text{Var}(y)$$

To compensate for this halving:

$$n_{in} \cdot \text{Var}(w) \cdot \frac{1}{2} = 1 \implies \text{Var}(w) = \frac{2}{n_{in}}$$

<div class="math-box">

**He (Kaiming) initialization**: $W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$

</div>

<div class="paper">

He et al., *"Delving Deep into Rectifiers"*, ICCV 2015

</div>

---

# The Effect of Initialization

![w:950px](figures/lec01/weight_init_activations.png)

Too small → collapse. Too large → explode. Xavier/He → stable across layers.

---

# Initialization in PyTorch

```python
# PyTorch default for nn.Linear: Kaiming uniform (good for ReLU!)
model = nn.Linear(784, 256)  # Already sensible

# Explicit He init:
nn.init.kaiming_normal_(model.weight, mode='fan_in', nonlinearity='relu')
nn.init.zeros_(model.bias)

# Xavier (for sigmoid/tanh):
nn.init.xavier_normal_(model.weight)
```

<div class="insight">

Rule of thumb: **ReLU → He, Sigmoid/Tanh → Xavier**. In practice, PyTorch defaults work well. But knowing *why* helps you debug when things go wrong.

</div>

---

<div class="popquiz">

**Pop Quiz**: You build a 10-layer MLP with Tanh and **He** initialization. Loss oscillates wildly. What's wrong?

</div>

---

# Pop Quiz Answer

He init assumes ReLU (compensates for 50% zeroed outputs). Tanh doesn't zero any outputs — He gives **too large** a variance → activations saturate.

**Fix**: Use Xavier initialization.

| Activation | Init | Why |
|-----------|------|-----|
| ReLU | He ($\frac{2}{n_{in}}$) | Compensates for 50% zero outputs |
| Sigmoid, Tanh | Xavier ($\frac{2}{n_{in}+n_{out}}$) | No compensation needed |
| GELU, SiLU | He (convention) | Similar to ReLU in practice |

---

<!-- _class: section-divider -->

# Recap

---

# What We Covered Today

| Topic | Key Takeaway |
|-------|-------------|
| **UAT** | 1 hidden layer suffices in theory; exponential width in practice |
| **Depth vs Width** | Depth enables compositionality; exponential parameter savings |
| **Degradation** | Deeper plain nets are *harder* to optimize (not just overfit) |
| **ResNets** | $H(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$; gradient always $\geq \mathbf{I}$ |
| **Xavier** | $\text{Var}(w) = \frac{2}{n_{in}+n_{out}}$ for sigmoid/tanh |
| **He** | $\text{Var}(w) = \frac{2}{n_{in}}$ for ReLU (compensates halving) |

---

<!-- _class: summary-slide -->

# Lecture 2: Summary

- **UAT**: existence result only — says nothing about learnability, efficiency, or generalization
- **Depth wins**: compositionality gives exponential savings; deep = reusable hierarchical features
- **Degradation problem**: deeper plain networks optimize *worse* — adding layers hurts training error
- **ResNets**: skip connections → gradient highway ($\nabla \geq \mathbf{I}$) → smooth loss landscape
- **Initialization**: Xavier for sigmoid/tanh, He for ReLU — derived from variance preservation across layers

### Next lecture

**Lecture 3**: Training Deep Networks in Practice — batch size, learning rate, data pipelines, GPU training, and the full PyTorch workflow end to end.

<div class="notebook">

**Notebook**: [02-depth-and-resnets.ipynb](https://colab.research.google.com/) — Compare shallow vs deep MLPs; build ResNet blocks; visualize gradient flow and activation distributions

</div>
