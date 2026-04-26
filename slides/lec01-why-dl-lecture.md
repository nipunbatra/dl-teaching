---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Why Deep Learning?

## Lecture 1 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

<!-- _class: section-divider -->

### READING · Prince UDL · Ch 1 · Ch 3

# Lecture 1

This lecture mirrors UDL **Ch 1** (introduction) and **Ch 3** (shallow networks). Read both — slides go brisk on the parts the book covers well, deep on what it doesn't.

---

<!-- _class: section-divider -->

### PART 1

# The big picture

What is deep learning, and why now?

---

# A question to open the semester

You already built classifiers in ES 654 — logistic regression, SVMs, decision trees.

**Q.** Point any of them at a raw $224 \times 224$ colour photo.

<div class="popquiz">

Input dimension: $224 \times 224 \times 3 = 150{,}528$ features.

Think before the next slide: *why* would that fail?

</div>

---

# Linear · works for separable data, fails for curved data

![w:920px](figures/lec01/svg/linear_vs_nonlinear_data.svg)

---

# Why raw pixels break a linear classifier

![w:900px](figures/lec01/svg/pixel_shift_fail.svg)

---

# Classical ML vs deep learning

![w:920px](figures/lec01/svg/ml_vs_dl_pipeline.svg)

---

# Deep learning, in one sentence

<div class="insight">

**Deep learning = representation learning** with differentiable, composable modules, trained end-to-end by gradient descent.

</div>

Every word matters. We will unpack all of them this semester.

---

# Three eras of deep learning

![w:900px](figures/lec01/svg/dl_timeline.svg)

---

# ImageNet · the turning point (2012)

| Year | Winner | Top-5 error | Method |
|------|--------|-------------|--------|
| 2010 | NEC-UIUC | 28.2% | SIFT + Fisher |
| 2011 | XRCE | 25.8% | Hand-crafted |
| **2012** | **AlexNet** | **16.4%** | **8-layer CNN** |
| 2015 | ResNet | **3.6%** | 152-layer CNN |

Human top-5 error: ~5.1%. **AlexNet cut error by ~10 points in one year.**

---

# Why now? Three ingredients compounded

![w:900px](figures/lec01/svg/three_ingredients.svg)

---

# Learning outcomes · for this lecture

By the end of this lecture you will be able to:

1. Articulate **why DL works now but didn't earlier** (compute + data + algorithms).
2. Describe the **representation learning** vs hand-crafted feature contrast.
3. Reason about the **ImageNet 2012 moment** and its consequences.
4. Place the course in context · what you'll learn by L24.
5. Set up the **compute and environment** for assignments.
6. State **three real risks** of modern DL (energy, alignment, misinformation).

---

# Course roadmap

<div class="realworld">

**24-lecture arc**

Foundations → optimization → regularization → **CNNs → detection** → sequences → **attention → Transformers → LLMs → vision-language** → VAEs → GANs → diffusion → efficient inference.

</div>

- **Framework:** PyTorch, exclusively.
- **Primary textbook:** Prince, *Understanding Deep Learning* (2023) — free PDF.
- **Style:** math + code + intuition + examples.
- **Assessment:** 4 quizzes · 4 assignments · attendance · bonus.

---

<!-- _class: section-divider -->

### PART 2

# MLP recap

The building block you already know — tightened up

---

# Our running example

![w:880px](figures/lec01/svg/mnist_samples.svg)

---

# From linear models to neurons

You already know **linear regression / classification** from ES 654:

$$\hat y = \mathbf{w}^\top \mathbf{x} + b$$

A neuron is just this · plus a non-linear "squashing" function:

$$\hat y = \sigma(\mathbf{w}^\top \mathbf{x} + b)$$

<div class="keypoint">

That's the only new ingredient. Everything in this course builds on top · stack many of these neurons, learn the weights, you get a deep network.

</div>

---

# Worked example · one neuron forward pass

<div class="math-box">

Input · $x = [0.5, -1.0]$ · Weights · $w = [0.8, 0.2]$ · Bias · $b = 0.1$

**Pre-activation** · $z = (0.8)(0.5) + (0.2)(-1.0) + 0.1 = 0.4 - 0.2 + 0.1 = 0.3$

**Activation (sigmoid)** · $\sigma(0.3) = 1 / (1 + e^{-0.3}) \approx 0.574$

</div>

This 1-neuron, 2-input setup is exactly what we just had in linear regression · plus the sigmoid making it 0–1 instead of any real number. Stack 500 of these and you have an MLP layer.

---

# The single neuron — anatomy

![w:900px](figures/lec01/svg/neuron_anatomy.svg)

---

# In vector form

$$y = \sigma\big(\underbrace{\mathbf{w}^\top \mathbf{x} + b}_{\text{pre-activation }z}\big)$$

- $\mathbf{x} \in \mathbb{R}^d$ — inputs
- $\mathbf{w} \in \mathbb{R}^d$ — learned weights (how much each input matters)
- $b \in \mathbb{R}$ — learned bias (threshold shift)
- $\sigma(\cdot)$ — non-linearity (squash)

**Q.** Why the non-linearity? What breaks without it?

---

# Why we need a non-linearity · the magnifying-glass analogy

<div class="keypoint">

Stack two magnifying glasses · you get a bigger image. Still just a *bigger linear* version of the original.

Stack two linear layers · same story. The composition of linear maps is just another linear map. No new patterns become learnable.

A non-linearity is like adding a **prism** · it bends the input in a way no linear stack can replicate. Each layer can learn a new *kind* of feature.

</div>

This is why every deep network has activation functions between linear layers. Without them, depth is wasted.

---

# Let's prove it · stacking linear layers collapses

A tiny 2-layer network with no nonlinearity:

<div class="math-box">

**Layer 1** · $h = W_1 x + b_1$
**Layer 2** · $y = W_2 h + b_2$

Substitute h into the second equation:
$y = W_2 (W_1 x + b_1) + b_2$

Distribute $W_2$:
$y = (W_2 W_1) x + (W_2 b_1 + b_2)$

Define · $W_\text{eff} = W_2 W_1$ (just another matrix) and $b_\text{eff} = W_2 b_1 + b_2$ (just another vector):

$$y = W_\text{eff} x + b_\text{eff}$$

</div>

**The 2-layer network is exactly one linear layer.** The depth was useless.

---

# Worked numeric example · the collapse

<div class="math-box">

$x = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$ · $W_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ · $b_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ · $W_2 = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ · $b_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$

**Forward through the 2 layers:**
- $h = W_1 x + b_1 = \begin{bmatrix} 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 3 \\ 3 \end{bmatrix}$
- $y = W_2 h + b_2 = \begin{bmatrix} 9 \\ 9 \end{bmatrix} + \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 9 \\ 10 \end{bmatrix}$

**Equivalent single layer:**
- $W_\text{eff} = W_2 W_1 = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$
- $b_\text{eff} = W_2 b_1 + b_2 = \begin{bmatrix} 2 \\ 1 \end{bmatrix} + \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$
- Check · $W_\text{eff} x + b_\text{eff} = \begin{bmatrix} 7 \\ 8 \end{bmatrix} + \begin{bmatrix} 2 \\ 2 \end{bmatrix} = \begin{bmatrix} 9 \\ 10 \end{bmatrix}$ · **same!**

</div>

---

# Without σ · depth gives nothing

![w:920px](figures/lec01/svg/stacked_linear_collapses.svg)

---

# Activation functions at a glance

![w:900px](figures/lec01/activation_functions.png)

| Name | Formula | Where you see it |
|------|---------|------------------|
| Sigmoid | $1/(1+e^{-z})$ | gates |
| Tanh | $\tanh(z)$ | RNNs |
| ReLU | $\max(0, z)$ | most CNNs |
| GELU / SiLU | $z\,\Phi(z)$ / $z\,\sigma(z)$ | Transformers, LLMs |

---

# Stacking neurons → MLP

![w:820px](figures/lec01/svg/mlp_architecture.svg)

---

# Parameter count — do this in your head

For MNIST with hidden sizes 256, 256:

$$\underbrace{256 \times 784}_{W_1} + \underbrace{256}_{b_1} + \underbrace{256 \times 256}_{W_2} + \underbrace{256}_{b_2} + \underbrace{10 \times 256}_{W_3} + \underbrace{10}_{b_3} = \boxed{269{,}322}$$

A tiny MNIST model has ~270k parameters. GPT-3 has 175 billion — $6 \times 10^5 \times$ more.

---

# MLP in PyTorch

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in=784, d_h=256, d_out=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(),
            nn.Linear(d_h,  d_h), nn.ReLU(),
            nn.Linear(d_h,  d_out),     # raw logits — no softmax here
        )

    def forward(self, x):
        return self.net(x)
```

**Q.** Why no activation after the last `Linear`?

---

# The last layer is bare · the #1 beginner bug

For classification:
- Output should be $K$ raw scores (logits).
- `nn.CrossEntropyLoss` internally applies `log_softmax`.
- Adding your own softmax → **double softmax** → frozen loss near $\log K$.

<div class="warning">

Symptom to memorize: training loss stuck at ~2.30 (= $\log 10$) and refusing to move.
Fix: remove the extra softmax.

</div>

---

<!-- _class: section-divider -->

### PART 3

# Losses and backprop

The math that makes learning possible

---

# From scores to probabilities · the goal

The network outputs raw "scores" (logits) for each class · arbitrary real numbers like $[2.0, 1.0, 0.1]$.

For classification, we need a **valid probability distribution** · all values $\ge 0$, summing to 1.

<div class="keypoint">

Two problems to solve:
1. **Make values positive** · `exp(·)` does this for any real input.
2. **Make them sum to 1** · divide by the total.

Together · the **softmax** function.

</div>

---

# Softmax · worked numeric example

<div class="math-box">

Logits · $z = [2.0, 1.0, 0.1]$.

**Step 1 · exponentiate** · $e^z = [e^{2.0}, e^{1.0}, e^{0.1}] = [7.39, 2.72, 1.11]$
**Step 2 · sum** · $7.39 + 2.72 + 1.11 = 11.22$
**Step 3 · normalize** · $\hat y = [7.39/11.22, 2.72/11.22, 1.11/11.22] = [0.66, 0.24, 0.10]$

</div>

Note · the relative ranking is preserved (the largest logit becomes the largest probability) · but the values are now interpretable as probabilities. The softmax is the standard last layer for classification.

---

# Softmax · three acts

![w:920px](figures/lec01/svg/softmax_visual.svg)

<div class="realworld">

▶ Interactive: drag the temperature slider, watch the distribution morph — [softmax-temperature](https://nipunbatra.github.io/interactive-articles/softmax-temperature/).

</div>

---

# Why exponentiate?

1. Logits can be **negative**; raw ratios misbehave. $e^{z_k}$ is always positive.
2. Softmax **amplifies** differences — biggest logit dominates smoothly.
3. It falls out of **maximum likelihood** for categorical outputs (next).

---

# Cross-entropy from MLE

One example $(\mathbf{x}, y)$, true class $c$. Maximize data likelihood → minimize negative log-likelihood:

$$\mathcal{L}(\theta) = -\log P(y = c \mid \mathbf{x}; \theta) = -\log \hat{y}_c$$

With one-hot $\mathbf{y}$:

$$\mathcal{L} = -\sum_{k=1}^{K} y_k \log \hat{y}_k$$

<div class="math-box">

This **is** cross-entropy. MLE hands it to us for free — we did not invent it.

</div>

---

# Push-pull intuition · the gradient

Logits $z = [z_1, ..., z_K]$. Loss $\mathcal{L}$. We need $\partial \mathcal{L} / \partial z_k$ · "how should we change $z_k$ to lower the loss?"

<div class="keypoint">

- If $k$ is the **correct** class · we want its probability to be 1 · gradient should **push $z_k$ up**.
- If $k$ is a **wrong** class · we want its probability to be 0 · gradient should **push $z_k$ down**.

</div>

The next slide derives the formula. The result · $\partial \mathcal{L}/\partial z_k = \hat y_k - y_k$ · does exactly the push-pull above.

---

# Deriving · softmax + CE gradient · step by step

<div class="math-box">

**Setup** · $\mathcal{L} = -\sum_j y_j \log \hat y_j$ · $\hat y_j = e^{z_j} / \sum_i e^{z_i}$.

Want · $\partial \mathcal{L}/\partial z_k$. Chain rule · $\partial \mathcal{L}/\partial z_k = \sum_j (\partial \mathcal{L}/\partial \hat y_j)(\partial \hat y_j/\partial z_k)$.

**Part 1** · $\partial \mathcal{L}/\partial \hat y_j = -y_j/\hat y_j$.

**Part 2 · case $j = k$** (own logit, quotient rule):
$\partial \hat y_k/\partial z_k = \hat y_k(1 - \hat y_k)$

**Part 2 · case $j \ne k$**:
$\partial \hat y_j/\partial z_k = -\hat y_j \hat y_k$

**Combine** · $\partial \mathcal{L}/\partial z_k = -y_k(1-\hat y_k) + \sum_{j \ne k} y_j \hat y_k = -y_k + \hat y_k \sum_j y_j$

Since $\sum_j y_j = 1$ for one-hot · $\boxed{\partial \mathcal{L}/\partial z_k = \hat y_k - y_k}$

</div>

---

# Worked numeric · the gradient

<div class="math-box">

Logits $z = [2.0, 1.0, 0.1]$. Softmax · $\hat y = [0.66, 0.24, 0.10]$. True label · class 0, $y = [1, 0, 0]$.

$$\partial \mathcal{L}/\partial z = \hat y - y = [0.66 - 1, 0.24 - 0, 0.10 - 0] = [-0.34, 0.24, 0.10]$$

**SGD step** · $z \leftarrow z - \eta \cdot (\hat y - y)$
- $z_0$ has negative gradient · SGD pushes it **up** (good, correct class).
- $z_1, z_2$ have positive gradient · SGD pushes them **down** (good, wrong classes).

</div>

The gradient is bounded between -1 and 1 per logit · stable. No exploding gradients from the loss itself.

---

# The elegant softmax + CE gradient

$$\boxed{\dfrac{\partial \mathcal{L}}{\partial z_k} = \hat{y}_k - y_k}$$

**Prediction minus target.** Same form as logistic regression — no accident.

---

# What that gradient actually looks like

![w:920px](figures/lec01/svg/ce_gradient_visual.svg)

---

# Backprop · the blame game

The forward pass computes the loss. **Backprop figures out who to blame.**

<div class="keypoint">

For each weight, ask · *how much did this weight contribute to the final error?*

Walk backwards from the loss through the network · at each layer, distribute "blame" proportionally to how much each weight affected the layer's output. Adjust weights to reduce that blame.

</div>

That's the entire idea. The math (chain rule) is mechanical · the *concept* is "trace responsibility backwards."

---

# Backpropagation · the computational view

Every network is a DAG of differentiable ops. Forward computes values; backward computes gradients in reverse order.

![w:900px](figures/lec01/svg/computational_graph.svg)

---

# Backprop · the blame distributor

For a layer $z = Wx + b$, suppose the next layer sends back · "your output's error signal is $\delta = \partial \mathcal{L}/\partial z$."

<div class="keypoint">

We must figure out three things:
- How much was **W**'s fault? · $\partial \mathcal{L}/\partial W$
- How much was **b**'s fault? · $\partial \mathcal{L}/\partial b$
- How much should we blame the **previous layer's** output $x$? · $\partial \mathcal{L}/\partial x$ · this is the message we pass back.

</div>

The next slide does these one at a time, with full chain rule.

---

# Deriving · the linear-layer backward pass

For one element · $z_i = \sum_j W_{ij} x_j + b_i$.

<div class="math-box">

**1. Bias** · $\partial z_i/\partial b_i = 1$ → $\partial \mathcal{L}/\partial b_i = \delta_i$ → $\partial \mathcal{L}/\partial b = \delta$

**2. Weight** · $\partial z_i/\partial W_{ij} = x_j$ → $\partial \mathcal{L}/\partial W_{ij} = \delta_i x_j$ → $\partial \mathcal{L}/\partial W = \delta\, x^\top$ (outer product)

**3. Input** · $x_j$ affects every output through $W_{ij}$ · sum over outputs:
$\partial \mathcal{L}/\partial x_j = \sum_i \delta_i \cdot W_{ij}$ → $\partial \mathcal{L}/\partial x = W^\top \delta$

</div>

These three lines are the entire backward pass for a `Linear` layer in PyTorch.

---

# Worked numeric · linear-layer backward

<div class="math-box">

$x = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$ · $W = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \end{bmatrix}$ · upstream gradient $\delta = \begin{bmatrix} 0.5 \\ -0.1 \end{bmatrix}$.

**Weight gradient** · $\delta x^\top = \begin{bmatrix} 0.5 \\ -0.1 \end{bmatrix} \begin{bmatrix} 2 & 3 \end{bmatrix} = \begin{bmatrix} 1.0 & 1.5 \\ -0.2 & -0.3 \end{bmatrix}$

**Bias gradient** · $\delta = \begin{bmatrix} 0.5 \\ -0.1 \end{bmatrix}$

**Input gradient** (passed to previous layer) · $W^\top \delta = \begin{bmatrix} 0.1 & 0.3 \\ 0.2 & 0.4 \end{bmatrix} \begin{bmatrix} 0.5 \\ -0.1 \end{bmatrix} = \begin{bmatrix} 0.02 \\ 0.06 \end{bmatrix}$

</div>

These three numbers are what `loss.backward()` computes for one Linear layer · in pure NumPy you could write it in 3 lines.

---

# The local-gradient rule · three lines

For $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$ with upstream $\boldsymbol{\delta} = \partial \mathcal{L} / \partial \mathbf{z}$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \boldsymbol{\delta}\, \mathbf{x}^\top, \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \boldsymbol{\delta}, \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \mathbf{W}^\top \boldsymbol{\delta}$$

<div class="keypoint">

These three lines are the **entire** backward pass of a linear layer. Everything else is repeating this rule through activations and stacking.

</div>

---

<!-- _class: section-divider -->

### PART 4

# Why go deep?

A teaser for Lecture 2

---

# One layer is enough, in principle

A single hidden layer can approximate any continuous function (UAT — next lecture).

**Q.** So why do we ever use more than one?

Give an honest answer before turning the page.

---

# Depth ⇒ hierarchical features

![w:900px](figures/lec01/svg/feature_hierarchy.svg)

---

# Biology does the same thing

Hubel & Wiesel (Nobel 1981) — the cat visual cortex is hierarchical.

| Brain region | Roughly analogous layer | Detects |
|--------------|------------------------|---------|
| V1 | early layer | oriented edges |
| V2 | next layer | textures, junctions |
| V4 | mid layer | shapes, parts |
| IT | late layer | objects, faces |

Biology inspired the architecture; the optimizer and data are engineered.

---

# Backprop as broken telephone

<div class="keypoint">

Think of backprop as a "telephone game" played backwards from the loss.

Each layer has to pass the **error signal** to the layer before it. If each layer multiplies the message by something less than 1 (e.g., 0.25 for sigmoid), then by the time the signal reaches the early layers it's a faint mumble.

</div>

Those early layers stop learning. This is the **vanishing gradient** problem · the topic of the next slide and a recurring theme through the course (RNN, deep MLPs, deep Transformers).

---

# Why σ′ shrinks · let's compute it

For sigmoid · $\sigma(z) = 1/(1 + e^{-z})$.

<div class="math-box">

$\sigma'(z) = \dfrac{e^{-z}}{(1+e^{-z})^2} = \sigma(z) \cdot (1 - \sigma(z))$

Maximum value · at $z = 0$, $\sigma(0) = 0.5$, so $\sigma'(0) = 0.5 \cdot 0.5 = \mathbf{0.25}$.

For $|z| \ge 3$ · $\sigma'(z) \le 0.045$ (saturated regions).

</div>

**Every layer** with sigmoid multiplies the backward-flowing gradient by something $\le 0.25$. After 5 layers, the gradient is shrunk by at least $0.25^5 \approx 0.001$. After 10 layers · $\approx 10^{-6}$. The earliest layers stop learning.

---

# Worked numeric · the gradient vanishes

A 5-layer sigmoid net. Assume all weights = 1 and inputs are in saturating regions so $\sigma'(z) \approx 0.1$. Gradient at output = 1.

<div class="math-box">

| Layer | Local factor $W \cdot \sigma'$ | Gradient signal |
|:-:|:-:|:-:|
| Output | — | 1.0 |
| L4 | $1 \cdot 0.1 = 0.1$ | 0.1 |
| L3 | $0.1$ | 0.01 |
| L2 | $0.1$ | 0.001 |
| L1 | $0.1$ | **0.0001** |

</div>

The first layer gets $10^{-4}$ of the signal · its weights barely move. **Learning stalls in the early layers.** This is exactly why ReLU (derivative 0 or 1) replaced sigmoid in deep nets · and why ResNet's skip connections exist.

---

# Depth has a cost · vanishing gradients

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \prod_{l=1}^{L} \frac{\partial \mathbf{h}_l}{\partial \mathbf{h}_{l-1}} \cdot \frac{\partial \mathcal{L}}{\partial \hat{y}}$$

Each sigmoid factor ≤ 0.25:

| Depth $L$ | upper bound on gradient magnitude |
|----|----|
| 5  | $10^{-3}$ |
| 10 | $10^{-6}$ |
| 20 | $10^{-12}$ |

Early layers effectively stop learning. **This blocked depth for 20 years.**

---

# The fix · ReLU

$$\text{ReLU}(z) = \max(0, z), \quad \text{ReLU}'(z) \in \{0, 1\}$$

For active neurons the gradient is exactly 1 — no shrinkage.

We'll come back to ResNets (the *real* fix for very deep nets) in Lecture 2.

---

<!-- _class: section-divider -->

### PART 5

# The training loop

The five lines you'll type every day this semester

---

# The training cycle

![w:880px](figures/lec01/svg/training_cycle.svg)

---

# The PyTorch training loop · code

```python
model     = MLP(784, 256, 10).to('cuda')
criterion = nn.CrossEntropyLoss()
optim     = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(10):
    for x, y in loader:
        x, y = x.view(-1, 784).to('cuda'), y.to('cuda')
        logits = model(x)                   # 1. forward
        loss   = criterion(logits, y)       # 2. loss
        optim.zero_grad()                   # 3. zero grads
        loss.backward()                     # 4. backward
        optim.step()                        # 5. update
```

Every real training script is a variation on this.

---

# One common bug · `zero_grad`

**Q.** What if you forget `optimizer.zero_grad()`?

<div class="warning">

PyTorch **accumulates** gradients by default. Without zeroing, each backward adds to the previous `.grad`. After $k$ batches you are stepping $k\times$ the real gradient — updates explode silently.

The single most common PyTorch bug.

</div>

---

# Train / val / test

![w:900px](figures/lec01/svg/train_val_test_split.svg)

---

# Training loss ↓ ≠ model better

![w:920px](figures/lec01/svg/loss_curves_annotated.svg)

---

# Overfitting · the rule

- Training loss monotonically drops.
- Validation loss drops, then rises.
- The gap **is** overfitting.

<div class="warning">

**Never tune on the test set.**

Test = final exam you take once.
Validation = practice exams, take many times.
Training = studying.

</div>

---

<!-- _class: summary-slide -->

# Lecture 1 — summary

- **Deep learning = representation learning.** Features learned, not designed.
- **Why now:** data + compute + algorithms compounded 2009–2017.
- **Neuron = sum + squash.** Stack them and non-linearity keeps depth meaningful.
- **Softmax + CE from MLE:** $\partial \mathcal{L} / \partial \mathbf{z} = \hat{\mathbf{y}} - \mathbf{y}$.
- **Backprop** = three lines per layer, repeated.
- **Training loop:** forward → loss → zero_grad → backward → step.

### Read before Lecture 2

**Prince · Understanding Deep Learning** — Ch 1, Ch 3. Free PDF at [udlbook.github.io](https://udlbook.github.io/udlbook/).

### Next lecture

Why depth, ResNets, Xavier / He initialization derived from first principles.

<div class="notebook">

**1a** · `01a-micrograd.ipynb` — scalar autograd engine from scratch (Karpathy-style).
**1b** · `01b-mlp-mnist.ipynb` — train this MLP on MNIST end-to-end.

</div>
