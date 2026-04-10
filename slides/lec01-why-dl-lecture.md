---
marp: true
theme: dl-theme
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

# Part 1: The Big Picture

What is deep learning, and why now?

---

# A Question

You have built classifiers in your ML course — SVMs, decision trees, logistic regression.

They work well on tabular data with hand-crafted features.

**Q.** What happens when you try to classify raw images (say 224×224 pixels)?

---

# A Question

**Q.** What happens when you try to classify raw images?

- Input dimension: $224 \times 224 \times 3 = 150{,}528$ features
- SVM / Logistic Regression on raw pixels?

**Very poor performance.** Why?

---

# A Question

**Q.** Why do raw pixels fail for traditional ML?

- Pixels are not meaningful features
- A 1-pixel shift changes *every* feature value
- No notion of locality, edges, or shapes
- The **representation** is wrong

---

# The Core Idea of Deep Learning

<div class="keypoint">

**Deep Learning** = Representation Learning

Instead of hand-crafting features, learn them from data.

</div>

---

# Machine Learning vs Deep Learning

<div class="columns">
<div>

### ML (what you know)

- **You** design features
- Shallow models (SVM, trees)
- Structured / tabular data
- Representation is fixed

</div>
<div>

### DL (what we'll learn)

- **Model** learns features
- Deep models (many layers)
- Images, text, audio, video
- Representation is learned

</div>
</div>

---

# The Deep Learning Revolution

![w:900px](figures/lec01/dl_timeline.png)

---

# ImageNet: The Turning Point (2012)

| Year | Winner | Top-5 Error | Method |
|------|--------|-------------|--------|
| 2010 | NEC-UIUC | 28.2% | Hand-crafted (SIFT + Fisher) |
| 2011 | XRCE | 25.8% | Hand-crafted (Fisher) |
| **2012** | **AlexNet** | **16.4%** | **CNN (deep learning!)** |
| 2013 | ZFNet | 11.7% | CNN |
| 2014 | GoogLeNet | 6.7% | CNN (22 layers) |
| 2015 | ResNet | **3.6%** | CNN (**152 layers**) |

**Human performance**: ~5.1%

AlexNet cut the error rate by **10 percentage points** in one year. Everything changed.

---

# Why Now? Three Ingredients

<div class="columns3">
<div>

### Data

- ImageNet (14M images)
- Common Crawl (PBs of text)
- YouTube, Wikipedia
- Synthetic data

</div>
<div>

### Compute

- GPUs (NVIDIA, 2007+)
- TPUs (Google, 2016+)
- Cloud computing
- Training: days → hours

</div>
<div>

### Algorithms

- Better activations (ReLU)
- Better optimizers (Adam)
- Better architectures (ResNet, Transformer)
- Better regularization

</div>
</div>

---

# What Can DL Do Today?

<div class="columns">
<div>

### Vision
- Image classification
- Object detection (self-driving)
- Medical image segmentation
- Image generation (Stable Diffusion)

### Language
- Translation (Google Translate)
- Chatbots (ChatGPT, Claude)
- Code generation (Copilot)

</div>
<div>

### Multimodal
- Visual question answering
- Text-to-image (DALL-E)
- Text-to-video (Sora)

### Science
- Protein folding (AlphaFold)
- Weather forecasting (GenCast)
- Drug discovery
- Climate modeling

</div>
</div>

---

# Course Roadmap

<div class="realworld">

**Our 24-lecture journey:**

Foundations → Optimization → Regularization → **CNNs → Object Detection** → Sequences → **Attention → Transformers → LLMs → VLMs** → VAEs → GANs → Diffusion → Frontiers

</div>

- **Framework**: PyTorch (exclusively)
- **Style**: Math + code + intuition
- **Assessment**: 4 quizzes (48%), 4-5 assignments (40%), attendance (6%), bonus (6%)

---

<!-- _class: section-divider -->

# Part 2: MLP Recap

Building blocks you already know

---

# Recall: The Single Neuron

A single neuron computes:

$$y = \sigma\left(\sum_{i=1}^{d} w_i x_i + b\right)$$

**Q.** Write this in vector notation.

---

# Recall: The Single Neuron

$$y = \sigma(\mathbf{w}^\top \mathbf{x} + b)$$

- **Inputs**: $\mathbf{x} \in \mathbb{R}^d$
- **Weights**: $\mathbf{w} \in \mathbb{R}^d$
- **Bias**: $b \in \mathbb{R}$
- **Activation**: $\sigma(\cdot)$

**Q.** What does $\sigma$ do? Why do we need it?

---

# Recall: The Single Neuron

**Q.** Why do we need the activation function $\sigma$?

Without it: $y = \mathbf{w}^\top \mathbf{x} + b$ — just linear regression!

Stacking linear layers gives: $\mathbf{W}_2(\mathbf{W}_1 \mathbf{x}) = (\mathbf{W}_2 \mathbf{W}_1)\mathbf{x} = \mathbf{W}' \mathbf{x}$

**Still linear.** Non-linearity is essential.

---

# Activation Functions

![w:950px](figures/lec01/activation_functions.png)

---

# From Neuron to MLP

**Multi-Layer Perceptron**: Stack layers with non-linearities between them.

$$\mathbf{h}_1 = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{h}_2 = \sigma(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)$$
$$\hat{y} = \mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3$$

**Q.** How many learnable parameters does this have, if $\mathbf{x} \in \mathbb{R}^{784}$, $\mathbf{h}_1, \mathbf{h}_2 \in \mathbb{R}^{256}$, $\hat{y} \in \mathbb{R}^{10}$?

---

# Parameter Count

$$\mathbf{W}_1 \in \mathbb{R}^{256 \times 784}, \quad \mathbf{b}_1 \in \mathbb{R}^{256}$$
$$\mathbf{W}_2 \in \mathbb{R}^{256 \times 256}, \quad \mathbf{b}_2 \in \mathbb{R}^{256}$$
$$\mathbf{W}_3 \in \mathbb{R}^{10 \times 256}, \quad \mathbf{b}_3 \in \mathbb{R}^{10}$$

Total: $256 \times 784 + 256 + 256 \times 256 + 256 + 10 \times 256 + 10$

$= 200{,}704 + 65{,}536 + 2{,}560 + 522 = \mathbf{269{,}322}$ parameters

Even a small MLP has **~270K parameters**. Modern LLMs have **billions**.

---

# MLP Architecture

![w:800px](figures/lec01/mlp_architecture.png)

---

# MLP in PyTorch

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
```

**Q.** How many `nn.Linear` layers does this have? How many activation functions?

---

# MLP in PyTorch

**Q.** How many `nn.Linear` layers? **3**. How many activations? **2**.

Note: No activation after the last layer.

**Q.** Why no activation after the final layer?

---

# MLP in PyTorch

**Q.** Why no activation after the final layer?

- For **classification**: we output raw logits; `CrossEntropyLoss` applies softmax internally
- For **regression**: we want unbounded output $\hat{y} \in (-\infty, \infty)$

<div class="warning">

Common mistake: applying softmax *and* using `CrossEntropyLoss` — this double-applies softmax!

</div>

---

<!-- _class: section-divider -->

# Part 3: Why Go Deep?

Depth as a computational resource

---

# The Central Question

We know shallow networks (1 hidden layer) work.

**Q.** Why would we ever want *more* layers?

Let us think about this carefully.

---

# Can a Single Layer Learn Anything?

<div class="popquiz">

**Pop Quiz**: Can a single hidden layer MLP approximate *any* continuous function?

</div>

---

# Universal Approximation Theorem

**Yes.** (Cybenko 1989, Hornik 1991)

<div class="math-box">

**Theorem (informal)**: A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of $\mathbb{R}^n$, to arbitrary accuracy.

</div>

So... problem solved? Just use 1 hidden layer?

---

# The Catch

The UAT guarantees **existence**, not **efficiency**.

A single hidden layer *can* approximate any function, but it may require **exponentially many neurons**.

**Q.** Can you think of a function that is easy to represent with depth, but hard with width?

---

# The Catch: An Example

Consider the function: $f(x_1, \ldots, x_n) = x_1 \oplus x_2 \oplus \cdots \oplus x_n$ (parity / XOR over $n$ bits)

- **Shallow network**: Needs $O(2^n)$ hidden neurons
- **Deep network**: Needs $O(n)$ neurons in $O(\log n)$ layers

Depth gives **exponential** savings for some function families.

<div class="paper">

Telgarsky, *"Benefits of Depth in Neural Networks"*, COLT 2016 — Proves formal depth separation results.

</div>

---

# Depth Enables Hierarchical Features

![w:950px](figures/lec01/feature_hierarchy.png)

Deep networks learn a **hierarchy** of representations:

Pixels → Edges → Textures → Parts → Objects

---

# Aside: How the Visual Cortex Works

The brain processes visual information hierarchically too:

| Brain Region | Roughly Corresponds To | Detects |
|-------------|----------------------|---------|
| V1 | Layer 1 | Oriented edges, bars |
| V2 | Layer 2 | Textures, simple shapes |
| V4 | Layer 3 | Complex shapes, parts |
| IT | Layer 4+ | Objects, faces |

<div class="insight">

Hubel & Wiesel (Nobel Prize, 1981) showed this hierarchical processing in cat visual cortex — inspiring neural network architectures decades later.

</div>

---

# Computational Graphs

![w:900px](figures/lec01/computational_graph.png)

---

# The Chain Rule: Backbone of DL

**Forward pass**: compute output from input (left → right)

**Backward pass**: compute gradients (right → left) via the chain rule

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \mathbf{h}_2} \cdot \frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1} \cdot \frac{\partial \mathbf{h}_1}{\partial \mathbf{W}_1}$$

**Q.** This is a product of $L$ terms for an $L$-layer network. What can go wrong?

---

# The Chain Rule: What Can Go Wrong?

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \prod_{l=1}^{L} \frac{\partial \mathbf{h}_{l}}{\partial \mathbf{h}_{l-1}} \cdot \frac{\partial \mathcal{L}}{\partial \hat{y}}$$

If each factor is slightly $< 1$ → the product $\to 0$ (**vanishing gradients**)

If each factor is slightly $> 1$ → the product $\to \infty$ (**exploding gradients**)

---

# Vanishing Gradients: The Sigmoid Problem

**Q.** What is the maximum value of $\sigma'(z)$ for the sigmoid function?

---

# Vanishing Gradients: The Sigmoid Problem

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

Maximum at $z = 0$: $\sigma'(0) = 0.25$

![w:900px](figures/lec01/activation_gradients.png)

---

# Vanishing Gradients: The Sigmoid Problem

With sigmoid, each layer multiplies the gradient by *at most* 0.25.

After 10 layers: $0.25^{10} = 9.5 \times 10^{-7}$

After 20 layers: $0.25^{20} \approx 10^{-12}$

**The gradient essentially disappears.** Early layers cannot learn.

---

# Vanishing Gradients: Visualized

![w:900px](figures/lec01/vanishing_gradient.png)

---

# Solution 1: ReLU

$$\text{ReLU}(z) = \max(0, z)$$

**Q.** What is the gradient of ReLU for $z > 0$?

---

# Solution 1: ReLU

$$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}$$

Gradient is exactly **1** for active neurons — no vanishing!

**Why ReLU dominates:**
- Gradient doesn't shrink with depth (for active neurons)
- Computationally cheap — just a threshold
- Induces sparsity — many neurons output 0
- Empirically works much better than sigmoid/tanh for deep networks

---

# ReLU Variants

| Name | Formula | Used in |
|------|---------|---------|
| ReLU | $\max(0, z)$ | Most CNNs |
| Leaky ReLU | $\max(0.01z, z)$ | GANs |
| GELU | $z \cdot \Phi(z)$ | **Transformers** (BERT, GPT) |
| SiLU / Swish | $z \cdot \sigma(z)$ | **Modern LLMs** (Llama) |

**Q.** What problem does Leaky ReLU solve that standard ReLU doesn't?

---

# ReLU Variants

**Q.** What does Leaky ReLU fix?

**Dead neurons.** If a ReLU neuron's input is always negative, its gradient is always 0 — it can never recover. Leaky ReLU ensures a small gradient even for negative inputs.

---

# Solution 2: Weight Initialization

<div class="warning">

**Q.** What happens if we initialize all weights to zero?

</div>

---

# Solution 2: Weight Initialization

**Q.** What happens if all weights are zero?

All neurons compute the same output → same gradient → same update.

**Symmetry is never broken.** The network effectively has 1 neuron per layer.

---

# Weight Initialization: The Goal

We want activations to maintain roughly **constant variance** across layers.

If variance grows per layer → exploding activations
If variance shrinks per layer → vanishing activations

![w:950px](figures/lec01/weight_init_activations.png)

---

# Xavier and He Initialization

| Method | Formula | When to use |
|--------|---------|-------------|
| **Xavier / Glorot** | $W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$ | Sigmoid, Tanh |
| **He / Kaiming** | $W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$ | ReLU, Leaky ReLU |

<div class="math-box">

**Intuition (He init):** ReLU kills ~50% of activations (those $< 0$). So we need $\text{Var}(W) = \frac{2}{n_{in}}$ instead of $\frac{1}{n_{in}}$ to compensate.

</div>

In PyTorch: `nn.Linear` uses Kaiming uniform by default — sensible out of the box.

---

<!-- _class: section-divider -->

# Part 4: The Training Loop

The 5 lines you'll write every day this semester

---

# The Loss Surface

<div class="columns">
<div>

![w:480px](figures/lec01/loss_surface_3d.png)

</div>
<div>

Neural network optimization is **non-convex**:

- Multiple local minima
- Saddle points (common in high-D)
- Plateaus and ravines

**But**: most local minima are *good enough* in practice.

</div>
</div>

---

# The Loss Surface: Contour View

![w:700px](figures/lec01/loss_surface_contour.png)

<div class="paper">

Li et al., *"Visualizing the Loss Landscape of Neural Nets"*, NeurIPS 2018

</div>

---

# The PyTorch Training Loop

<!-- _class: code-heavy -->

```python
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Data
transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 2. Model + Loss + Optimizer
model = MLP(784, 256, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3. Training loop
for epoch in range(10):
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.view(-1, 784)    # Flatten
        logits = model(batch_x)             # Forward
        loss = criterion(logits, batch_y)   # Loss
        optimizer.zero_grad()               # Zero gradients
        loss.backward()                     # Backward
        optimizer.step()                    # Update
```

---

# Anatomy of the Training Loop

| Step | Code | What happens |
|------|------|-------------|
| **Forward** | `model(x)` | Input flows through layers |
| **Loss** | `criterion(logits, y)` | Scalar measuring error |
| **Zero grad** | `optimizer.zero_grad()` | Clear old gradients |
| **Backward** | `loss.backward()` | Compute $\frac{\partial \mathcal{L}}{\partial \theta}$ via chain rule |
| **Step** | `optimizer.step()` | $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$ |

**Q.** What happens if you forget `optimizer.zero_grad()`?

---

# A Common Bug

**Q.** What happens if you forget `optimizer.zero_grad()`?

PyTorch **accumulates** gradients by default.

Without zeroing, each `.backward()` *adds* to existing gradients.

After $k$ batches: effective gradient $= \sum_{i=1}^{k} \nabla_\theta \mathcal{L}_i$

Your updates become huge and wrong.

<div class="warning">

This is the single most common PyTorch bug for beginners. Always zero gradients before each backward pass.

</div>

---

# Train / Validation / Test

![w:900px](figures/lec01/train_val_test_split.png)

---

# Train / Validation / Test

**Q.** Training loss is decreasing every epoch. Is the model getting better?

---

# Train / Validation / Test

**Q.** Training loss decreasing = better model?

**Not necessarily.** Training loss *always* decreases (given enough capacity).

What matters is **validation loss**:

![w:900px](figures/lec01/training_curves.png)

---

# Overfitting: The Gap

<div class="columns">
<div>

- Training loss ↓ always
- Validation loss ↓ then ↑
- The gap = **overfitting**

**Use validation to decide:**
- When to stop (early stopping)
- Which hyperparameters are best
- Whether model is too complex

</div>
<div>

<div class="warning">

**Never** tune hyperparameters on the test set.

Test set = final exam you take **once**.

Validation set = practice exams you can take many times.

</div>

</div>
</div>

---

# Evaluation in PyTorch

```python
model.eval()                          # Switch to eval mode (disables dropout, etc.)
with torch.no_grad():                 # No gradient computation needed
    correct = 0
    total = 0
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.view(-1, 784)
        logits = model(batch_x)
        preds = logits.argmax(dim=1)  # Predicted class
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

print(f"Validation accuracy: {correct/total:.4f}")
model.train()                         # Switch back to train mode
```

**Q.** Why do we use `torch.no_grad()`? Why not just skip `loss.backward()`?

---

# Evaluation in PyTorch

**Q.** Why `torch.no_grad()`?

Even without calling `.backward()`, PyTorch still **builds the computation graph** during the forward pass (allocating memory for potential gradients).

`torch.no_grad()` skips graph construction entirely → faster + less memory.

---

<!-- _class: section-divider -->

# Part 5: Looking Ahead

What this course will cover

---

# The Landscape of Deep Learning

| Architecture | Input | Lecture |
|-------------|-------|---------|
| MLP | Tabular, vectors | 1-3 |
| CNN | Images, grids | 8-10 |
| RNN / LSTM | Sequences | 11-12 |
| Transformer | Anything (!) | 13-15 |
| LLM | Text | 16-17 |
| VLM | Image + Text | 18 |
| VAE / GAN / Diffusion | Generation | 19-22 |

Each architecture is designed for a specific **inductive bias** — an assumption about the structure of the data.

---

# Textbooks

<div class="columns">
<div>

### Primary
1. Bishop & Bishop, *Deep Learning: Foundations and Concepts* (2024)
2. Prince, *Understanding Deep Learning* (2023)
3. Goodfellow et al., *Deep Learning* (2016)

</div>
<div>

### Online
4. Zhang et al., *Dive into Deep Learning* (d2l.ai)
5. Karpathy, *Neural Networks: Zero to Hero* (YouTube)
6. Andrew Ng, *Deep Learning Specialization* (Coursera)

</div>
</div>

---

<!-- _class: summary-slide -->

# Lecture 1: Summary

- **Deep Learning** = learn representations, not hand-craft them
- **Why now**: data (ImageNet) + compute (GPUs) + algorithms (ReLU, Adam, ResNet)
- **MLP recap**: layers, activations, parameter counting
- **Why depth**: hierarchical features, but watch out for vanishing gradients
- **ReLU** prevents vanishing gradients; **He init** maintains variance across layers
- **Training loop**: forward → loss → zero_grad → backward → step
- **Validation** detects overfitting; test set used **once**

### Next lecture

**Lecture 2**: Universal Approximation Theorem in depth, depth vs width tradeoffs, residual connections, and why ResNets changed everything.

<div class="notebook">

**Notebook**: [01-mlp-mnist.ipynb](https://colab.research.google.com/) — Build an MLP, train on MNIST, visualize loss curves and learned representations

</div>
