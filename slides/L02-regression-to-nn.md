---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# From Regression to Neural Networks

## Lecture 2 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

A neural network is not a new animal. It is **logistic regression, stacked** — the moment you stop hand-picking features and let the model learn its own.

<div class="keypoint">

**One neuron = one logistic regression.** A line, squashed by a sigmoid, is already a classifier. It only fails when the true boundary is a *curve* — and the fix is not a cleverer line, it's a **hidden layer that learns the features** that make the problem linear again.

</div>

You built $\hat y = \theta^\top x$ and the cost $J(\theta)$ in **ES 335**. Today we watch that exact model grow a hidden layer and become a network.

$$\underbrace{\hat y=\theta^\top x}_{\text{linear reg.}} \;\rightarrow\; \underbrace{\sigma(\theta^\top x)}_{\text{logistic = 1 neuron}} \;\rightarrow\; \underbrace{\sigma\big(W_2\,\sigma(W_1 x)\big)}_{\text{MLP: features are learned}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**Today unlocks the rest of the course:**
- **L3** — if a hidden layer can bend any boundary, *how* expressive is it? (universal approximation)
- **L4–L5** — training this stack: SGD, momentum, Adam
- **L7–L9** — the same neuron, wired for images (CNNs)
- **L13+** — and wired for sequences (Transformers, LLMs)

</div>
<div>

**Borrowed from your own courses** (build on, not repeat):
- linear regression, $\theta$, $J(\theta)$ — **ML (ES 335)**
- logistic regression, basis $\phi(x)$ — **ML (ES 335)**
- MSE / cross-entropy as NLL — **L1 (this course)**

</div>
</div>

<div class="insight">

The pattern of the whole semester: **pick a distribution, minimise its negative log-likelihood** (L1) — today we make the *model* behind that likelihood deeper.

</div>

---

<!-- _class: section-divider -->

## Part 1 · The model you already have: linear regression

---

# Linear regression, in one line

Given inputs $x\in\mathbb R^d$ (with a $1$ folded in for the bias), the model predicts a **number**:

<div class="math-box">

$$\hat y = \theta^\top x = \theta_0 + \theta_1 x_1 + \dots + \theta_d x_d$$

</div>

![w:720px](figures/lec01/svg/linear_to_neuron.svg)

Each $\theta_j$ says *how much input $j$ pushes the prediction up or down*. That single weighted sum $z=\theta^\top x$ is the atom we build everything from today.

---

# The cost $J(\theta)$ — and where it comes from

You fit $\theta$ by minimising the **mean squared error** over $n$ examples:

$$J(\theta) = \frac{1}{n}\sum_{i=1}^{n}\big(y_i - \theta^\top x_i\big)^2$$

<div class="insight">

In **ES 335** this was "the natural thing to minimise." In **L1** we saw *why*: assume $y \mid x \sim \mathcal N(\theta^\top x,\sigma^2)$ and MSE is exactly the **negative log-likelihood**. Least squares *is* Gaussian maximum likelihood — the loss was never arbitrary.

</div>

Minimise it by the normal equations (closed form) or by **gradient descent** $\theta \leftarrow \theta - \eta\,\nabla_\theta J$ — the update we will run all semester.

---

# But most questions are yes / no, not how-much

"Is this email spam?" · "orange or tomato?" · "which digit?" — the target is a **label**, not a real number.

<div class="popquiz">

**Warm-up.** Feed a house-price regression $\hat y = \theta^\top x$ and read off $\hat y = 1.7$. If we wanted a *probability* of "spam," what is wrong with reporting $1.7$ directly?

</div>

*Answer:* a probability must live in $[0,1]$; $\theta^\top x$ ranges over all of $\mathbb R$. We need to **squash** the line's output into $[0,1]$ — that squash is the whole difference between regression and classification.

---

<!-- _class: section-divider -->

## Part 2 · Logistic regression *is* a single neuron

---

# The squash: the logistic (sigmoid) function

Take the linear score $z=\theta^\top x$ and pass it through the **sigmoid**:

<div class="math-box">

$$\sigma(z) = \frac{1}{1+e^{-z}}\in(0,1), \qquad \hat y = P(y{=}1\mid x)=\sigma(\theta^\top x)$$

</div>

- $z\to+\infty \Rightarrow \sigma\to 1$ · $z\to-\infty \Rightarrow \sigma\to 0$ · $z=0 \Rightarrow \sigma=0.5$
- monotone, smooth, differentiable — so gradient descent still works.

![w:520px](figures/Lnew/svg/sigmoid.svg)

The model is unchanged from linear regression *except* for this one nonlinear cap. That cap turns a number into a probability.

---

# That picture has a name: a **neuron**

$$\hat y = \sigma\big(\underbrace{\mathbf w^\top \mathbf x + b}_{\text{pre-activation }z}\big)$$

![w:820px](figures/lec01/svg/neuron_anatomy.svg)

<div class="keypoint">

**Logistic regression and a single artificial neuron are the same equation.** Weighted sum, add a bias, squash. Everything in this course is this atom, wired together thousands of times.

</div>

---

# Reading the neuron: log-odds are linear

Rearrange $\hat y = \sigma(\theta^\top x)$ and the linear model reappears — on the **log-odds** scale:

$$\log\frac{P(y{=}1\mid x)}{P(y{=}0\mid x)} = \theta^\top x$$

<div class="insight">

Logistic regression is a **linear model of the log-odds**. Each weight $\theta_j$ is *"how many log-odds units does class-1 gain per unit of feature $j$."* The nonlinearity only lives at the very last step — which is exactly why the *decision boundary* stays a straight line.

</div>

---

# The decision boundary is a straight line

We predict class 1 when $\hat y > 0.5$, i.e. when $\sigma(\theta^\top x)>0.5$, i.e.

$$\theta^\top x > 0.$$

<div class="math-box">

The boundary $\theta^\top x = 0$ is a **line** (in 2-D), a **plane** (in 3-D), a **hyperplane** in general. On one side class 1, on the other class 0 — the sigmoid only sets *how confident* we are as we move away from it.

</div>

So a single neuron can only ever draw a **straight** separator. Hold that thought — it is the crack the rest of the lecture pries open.

---

# Worked numeric · one neuron classifies an email

<div class="math-box">

A spam neuron with two features — $x_1=$ "contains 'free'", $x_2=$ "num. links" — weights $\mathbf w=[2.0,\,0.5]$, bias $b=-3$.

An email with $x=[1,\,4]$:
**Pre-activation** · $z = 2.0(1) + 0.5(4) - 3 = 2.0 + 2.0 - 3 = 1.0$
**Probability** · $\hat y = \sigma(1.0) = \dfrac{1}{1+e^{-1.0}} \approx 0.73$

</div>

$0.73 > 0.5$, so **flag as spam**. Notice: this is *identical* arithmetic to the linear-regression forward pass — plus one sigmoid. That is the entire step from ES 335 regression to a neuron.

<div class="notebook">

**📓 Notebook · logistic regression on real data** — fit a single neuron to separate **oranges vs apples** and watch the straight boundary form. *(ML ES 335 · `notebooks/logistic-apple-oranges.ipynb`)*

</div>

---

# A neuron is one line of PyTorch

```python
import torch, torch.nn as nn

neuron = nn.Linear(2, 1)        # holds w (2×1) and b
logit  = neuron(x)              # z = θᵀx
p      = torch.sigmoid(logit)   # ŷ = σ(θᵀx) ∈ (0,1)
```

Logistic regression is literally `nn.Linear(d, 1)` fed to a sigmoid. In practice we skip the explicit sigmoid and hand **raw logits** to `nn.BCEWithLogitsLoss`, which fuses sigmoid + cross-entropy for numerical stability. Same model — the neuron you just drew, now trainable by `loss.backward()`.

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** A logistic neuron has $\hat y = \sigma(2x_1 + 3x_2 - 6)$.
(a) Write the equation of its decision boundary and sketch it. (b) Classify the point $x=(3,\,1)$. (c) In which direction do you become *more* confident it is class 1?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

**(a)** Boundary is $\theta^\top x = 0$: $\;2x_1+3x_2-6=0 \Rightarrow x_2 = 2 - \tfrac23 x_1$ — a straight line through $(3,0)$ and $(0,2)$.

**(b)** $z = 2(3)+3(1)-6 = 3 > 0 \Rightarrow \hat y = \sigma(3)\approx 0.95$ → **class 1**.

**(c)** Confidence grows along the weight vector $\theta=(2,3)$ — the direction **perpendicular** to the boundary.

<div class="keypoint">

One neuron ⇒ one straight boundary, and the weights point "uphill" in probability. To bend that boundary we will need more than one neuron.

</div>

---

<!-- _class: section-divider -->

## Part 3 · Where a straight line fails

---

# Some data no line can split

On the left, one line separates the classes. On the right, the classes **curl around each other** — no straight boundary can do better than chance.

![w:680px](figures/lec01/svg/linear_vs_nonlinear_data.svg)

<div class="insight">

The technical name for a model too rigid to fit the true boundary is **high bias** (underfitting). A single neuron is high-bias the instant the answer is a curve.

</div>

---

# The ES 335 canon: oranges vs tomatoes

The running example from your ML course: two fruits, features like **radius** and **colour**. Small tomatoes sit *inside* a ring of oranges — a straight line in the raw feature plane strands one class on both sides.

<div class="columns">
<div>

**What a line sees**
No half-plane holds the inner cluster and excludes the outer ring. Accuracy caps out well below what the data allows.

</div>
<div>

**The classical fix**
Add the feature $r^2 = x_1^2 + x_2^2$ (distance from centre). Now a *threshold on $r$* — a linear rule in the new coordinate — separates them: the boundary is a **circle** back in the original space.

</div>
</div>

You had to *know* to add $r^2$. The whole promise of a hidden layer is that it would **discover** that feature for you.

---

# The canonical break: XOR

Two binary inputs, output $= x_1 \oplus x_2$:

<div class="columns">
<div>

| $x_1$ | $x_2$ | $y$ |
|:-:|:-:|:-:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

</div>
<div>

![w:380px](figures/Lnew/svg/xor_not_separable.svg)

*You can't.* The positive class sits on **opposite corners** — no half-plane contains exactly those two.

</div>
</div>

<div class="keypoint">

XOR is the 1969 Minsky–Papert result that **broke the perceptron** and froze the field for a decade. A single neuron literally cannot compute the simplest "same-or-different."

</div>

---

# XOR · a line can't, a curve can

![w:760px](figures/lec01/svg/xor_decision.svg)

**Left** · every single line strands two same-class points on opposite sides. **Right** · a curved region — which we will build from *two* neurons plus a nonlinearity — catches all four correctly.

---

# ES 335's fix: hand-craft the features

Your ES 335 escape hatch was **basis expansion** — invent richer features so a *linear* model can bend:

$$\phi(x) = \big[\,1,\; x_1,\; x_2,\; x_1^2,\; x_2^2,\; x_1 x_2,\;\dots\,\big], \qquad \hat y = \sigma\big(\theta^\top \phi(x)\big)$$

For the **circular** oranges-vs-tomatoes data, adding $x_1^2, x_2^2$ turns a ring that no line can split into something a plane *can* — the boundary becomes a **circle** in the original space.

<div class="notebook">

**📓 Notebook · feature transforms for a curved boundary** — logistic regression fails on concentric-ring data, then succeeds once you append $x_1^2,x_2^2$. *(ML ES 335 · `notebooks/logistic-circular.ipynb`)*

</div>

---

# The catch with hand-crafted $\phi(x)$

It works — *if you can guess the right $\phi$.*

<div class="columns">
<div>

**Easy to guess**
- oranges vs tomatoes → add radius $x_1^2+x_2^2$
- a parabola → add $x^2$

You *know* the physics, so you *know* the basis.

</div>
<div>

**Hopeless to guess**
- a $224\times224\times3$ photo → **150,528** raw pixels
- audio, text, protein sequences

What polynomial of pixels means "cat"? Nobody can write it down.

</div>
</div>

<div class="insight">

This is the through-line of the entire course: **stop guessing $\phi$. Learn it.** The next part builds the machine that does exactly that.

</div>

---

<!-- _class: section-divider -->

## Part 4 · The fix: a nonlinearity, then the MLP

---

# The one missing ingredient

We want to *stack* neurons so later ones work on the features earlier ones produce. The obvious move — stack two **linear** layers — does nothing. The whole game is one nonlinearity in between.

![w:860px](figures/lec01/svg/magnifying_glass.svg)

<div class="keypoint">

A linear map can **stretch and rotate** space, but never **bend** it — grid lines stay straight, so the boundary stays a line. A nonlinearity *bends* the space, so the next linear layer can carve a curve.

</div>

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** Stack two linear layers with *no* activation:
$$h = W_1 x + b_1, \qquad y = W_2 h + b_2.$$
Show that $y$ is still a **single** linear function of $x$ — write the effective $W_\text{eff}$ and $b_\text{eff}$. What does this prove about depth without nonlinearity?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

Substitute $h$ into the second layer and distribute $W_2$:

$$y = W_2(W_1 x + b_1) + b_2 = \underbrace{(W_2 W_1)}_{W_\text{eff}}\,x + \underbrace{(W_2 b_1 + b_2)}_{b_\text{eff}}$$

<div class="keypoint">

$W_\text{eff}=W_2 W_1$ is just another matrix; $b_\text{eff}$ just another vector. **The two-layer network collapses to one linear layer — the depth bought nothing.** Ten stacked linear layers are still one line. *Nonlinearity between layers is not optional; it is the only reason depth exists.*

</div>

![w:560px](figures/lec01/svg/stacked_linear_collapses.svg)

---

# Perceptron → multi-layer perceptron

- **Perceptron** (1958) — a *single* neuron with a hard threshold. Trainable, but (Minsky–Papert) can only draw a line: XOR defeats it.
- **Multi-layer perceptron (MLP)** — stack neurons into **layers**, each layer's outputs feeding the next, with a nonlinearity between them.

<div class="math-box">

$$\mathbf h = \sigma\!\big(W_1 \mathbf x + \mathbf b_1\big), \qquad \hat y = \sigma\!\big(W_2 \mathbf h + \mathbf b_2\big)$$

The hidden vector $\mathbf h$ is a **new representation** of $x$ — computed, not hand-designed.

</div>

<div class="notebook">

**🎛 Interactive · the perceptron learning rule** — watch a single neuron rotate its boundary example-by-example, and see it stall on XOR. *(ML ES 335 · `notebooks/perceptron-learning.ipynb`)*

</div>

---

# The MLP, drawn

![w:720px](figures/lec01/svg/mlp_architecture.svg)

Input $\to$ hidden $\to$ hidden $\to$ output. Every arrow bundle is a weight matrix $W_\ell$ plus a bias $\mathbf b_\ell$; every hidden layer applies a nonlinearity. Learning = finding all the $W_\ell,\mathbf b_\ell$ by gradient descent (backprop, L3–L4).

---

# The MLP is a few more lines of PyTorch

```python
net = nn.Sequential(
    nn.Linear(3, 4), nn.ReLU(),   # hidden layer — learns φ(x)
    nn.Linear(4, 2),              # output: raw logits
)
logits = net(x)                   # loss adds the softmax
```

Three linear layers with a ReLU between them — the XOR-solving machine of the last slide, now general. Note the last layer stays **bare**: `nn.CrossEntropyLoss` applies `log_softmax` internally, so adding your own softmax means squashing **twice** (the classic beginner bug — loss stuck near $\ln 10 \approx 2.30$).

---

# XOR, solved by hand with two neurons

A 2-input, 2-hidden, 1-output MLP with ReLU $\big(\mathrm{ReLU}(z)=\max(0,z)\big)$:

<div class="math-box">

$W_1 = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix},\; \mathbf b_1 = \begin{pmatrix} 0 \\ -1 \end{pmatrix},\; \mathbf w_2 = \begin{pmatrix} 1 \\ -2 \end{pmatrix},\; b_2 = 0$

| $\mathbf x$ | $W_1\mathbf x+\mathbf b_1$ | $\mathbf h=\mathrm{ReLU}(\cdot)$ | $\mathbf w_2^\top\mathbf h$ |
|:-:|:-:|:-:|:-:|
| $(0,0)$ | $(0,-1)$ | $(0,0)$ | $0$ |
| $(0,1)$ | $(1,0)$ | $(1,0)$ | $1$ |
| $(1,0)$ | $(1,0)$ | $(1,0)$ | $1$ |
| $(1,1)$ | $(2,1)$ | $(2,1)$ | $2-2=0$ |

</div>

Outputs $0,1,1,0$ — **exactly XOR.** The ReLU "kink" is the bend that no line had. Two neurons and one nonlinearity are the smallest machine that solves it.

---

# The nonlinearity has choices

![w:860px](figures/lec01/svg/activation_grid.svg)

Sigmoid (gates, probabilities) · Tanh (older RNNs) · **ReLU** (the modern default — cheap, no saturation for $z>0$) · GELU / SiLU (Transformers). Any of them bends the space; they differ in how gradients flow back, which we take up in L3–L4.

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** Count the parameters of a small MLP with layer sizes $\mathbf{3 \to 4 \to 2}$ (one hidden layer of 4 units, output of 2), each linear layer having weights **and** a bias. Give a formula, then the number.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

A layer from $m$ inputs to $n$ outputs has $n\times m$ weights and $n$ biases.

$$\underbrace{(4\times 3 + 4)}_{\text{layer 1}} \;+\; \underbrace{(2\times 4 + 2)}_{\text{layer 2}} \;=\; 16 + 10 = \boxed{26}$$

<div class="keypoint">

General rule: $\sum_\ell (n_\ell \cdot n_{\ell-1} + n_\ell)$. The same count for a tiny MNIST net $784\to256\to10$ is $\approx 203{,}530$ — and GPT-3 is $1.75\times10^{11}$. **Same atom, more of it.**

</div>

---

<!-- _class: section-divider -->

## Part 5 · Representation learning, and why now

---

# A hidden layer *learns* $\phi(x)$

In ES 335 *you* wrote $\phi(x)=[1,x,x^2,\dots]$. In an MLP the hidden layer **is** $\phi(x)$ — but its entries are learned from data.

![w:800px](figures/lec01/svg/feature_transform.svg)

<div class="keypoint">

The hidden layer reshapes the data into a coordinate system where the classes are **linearly separable**; the output neuron is then just logistic regression on those learned features. Rings no line could split become bands one line splits cleanly.

</div>

---

# Hand-crafted vs learned features

<div class="columns">
<div>

**ES 335 — classical ML**
$$\text{raw} \rightarrow \underbrace{\phi(x)}_{\text{you design}} \rightarrow \text{classifier}$$
You pick the basis; the model only draws the boundary. New domain ⇒ new hand-crafted features.

</div>
<div>

**ES 667 — deep learning**
$$\text{raw} \rightarrow \underbrace{\text{learned layers}}_{\text{gradient descent designs}} \rightarrow \text{prediction}$$
The features and the boundary are learned **together**, end-to-end.

</div>
</div>

![w:900px](figures/lec01/svg/ml_vs_dl_pipeline.svg)

---

# Deep learning = representation learning

<div class="insight">

**Deep learning is representation learning.** Instead of engineering $\phi(x)$, the network *learns* it — each layer outputs a new representation of the input, tuned to make the final decision easy.

</div>

Stack the layers and the learned features become **hierarchical**: pixels → edges → parts → objects. Each level is built from the one below, and nobody wrote the rules.

![w:820px](figures/lec01/svg/feature_hierarchy.svg)

---

# What makes a representation *good*

A hidden layer may compute any features it likes. The useful ones share two properties:

<div class="columns">
<div>

**Keep** task-relevant variation
— whatever actually distinguishes the classes (shape of the digit, meaning of the word).

</div>
<div>

**Suppress** nuisance variation
— lighting, position, phrasing, background — things that change the input but not the label.

</div>
</div>

<div class="keypoint">

A good $\phi(x)$ makes the final linear decision **trivial**. That is the real work training does — which is why "just add parameters" misses the point. Depth helps because *better features* let the last layer be simple.

</div>

---

# Why this exploded *now*, not in 1990

The idea (backprop, MLPs) is from the 1980s. Three curves had to cross first:

![w:900px](figures/lec01/svg/three_ingredients.svg)

<div class="columns">
<div>

- **Data** — ImageNet's 1.2M labelled images gave the features something to learn *from*.

</div>
<div>

- **Compute** — GPUs made training a big net hours, not months.
- **Algorithms** — ReLU, better init, dropout, Adam made depth trainable.

</div>
</div>

---

# ImageNet 2012 · learned features win

**The task:** classify each photo into one of **1000** categories, trained on **1.2M** labelled images. "Top-5 error" = the true label is not in the model's top 5 guesses.

| Year | Winner | Top-5 error | Features |
|:-:|:-:|:-:|:-:|
| 2011 | XRCE | 25.8% | **hand-crafted** (SIFT + Fisher) |
| **2012** | **AlexNet** | **16.4%** | **learned** (8-layer CNN) |
| 2015 | ResNet | **3.6%** | learned (152 layers) |

Human top-5 error $\approx 5.1\%$. In a single year, a network that **learned** its features cut error by ~10 points over the best hand-engineered pipeline — the moment the field turned to deep learning.

---

# Scale is the story

![w:820px](figures/lec01/svg/compute_scaling.svg)

<div class="keypoint">

Deep learning did not suddenly get *smarter* — the world got *faster and bigger*, and around **2012 (AlexNet)** all three curves crossed at once. ImageNet top-5 error fell from ~26% (hand-crafted features) to 16% (an 8-layer CNN) to 3.6% (ResNet) in three years. Learned features beat engineered ones, decisively.

</div>

---

# See it in code

<div class="notebook">

**📓 Notebook · single neuron on real data** — fit logistic regression to oranges vs apples; the boundary is the line from Part 2. *(ML ES 335 · `notebooks/logistic-apple-oranges.ipynb`)*

**📓 Notebook · learning a curved boundary** — the circular data a line cannot split, solved with $x^2$ features (hand-crafted), then reflect on what a hidden layer would learn instead. *(ML ES 335 · `notebooks/logistic-circular.ipynb`)*

**🎛 Interactive Lab · "why XOR needs a curve"** — a scroll-driven explainer showing the single line failing and the two-neuron bend succeeding *(to be authored · `interactive/articles/xor-needs-a-curve`)*.

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**A neuron is logistic regression; an MLP is neurons stacked with a nonlinearity between them, so the hidden layer learns the features $\phi(x)$ you used to hand-craft.**

</div>

- **Linear regression** $\hat y=\theta^\top x$ → **logistic** $\sigma(\theta^\top x)$ is a **single neuron** with a straight decision boundary.
- One neuron fails when the boundary is a **curve** (XOR, rings).
- **Nonlinearity is essential** — stacked linear layers collapse to one line.
- An **MLP** bends the space: its hidden layer *learns* $\phi(x)$ instead of you designing it.
- Deep learning = **representation learning**, and data + compute + algorithms made it work around 2012.

**Next (L3):** if one hidden layer can bend any boundary — *exactly how expressive is it?* The universal approximation theorem, and why we still go deep.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's own courses (IIT Gandhinagar) and standard references:

- **Linear & logistic regression, the sigmoid, decision boundary, basis expansion $\phi(x)$, softmax cost, oranges-vs-tomatoes** — *Machine Learning (ES 335)*, N. Batra · `github.com/nipunbatra/ml-teaching` (`linear-regression`, `logistic-regression` slides; notebooks `logistic-apple-oranges.ipynb`, `logistic-circular.ipynb`, `perceptron-learning.ipynb`)
- **Representation-learning framing** — *ML (ES 335)* neural-networks notes, N. Batra
- **MSE / cross-entropy as negative log-likelihood** — *L1 · Probability & the Language of Losses*, this course
- **Pedagogical framing (regression → logistic → neural nets)** — A. Ng, *Deep Learning Specialization*, Course 1, Week 2; Prince, *Understanding Deep Learning*, Ch. 3–4.

Figures adapted from the ES 667 figure library (`figures/lec01`). All source courses © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · the logistic cost *is* cross-entropy

The neuron outputs a **Bernoulli**: $P(y\mid x)=\hat y^{\,y}(1-\hat y)^{1-y}$ with $\hat y=\sigma(\theta^\top x)$. Its negative log-likelihood over the data is

$$J(\theta) = -\sum_{i=1}^{n}\Big[\,y_i\log\hat y_i + (1-y_i)\log(1-\hat y_i)\,\Big]$$

— **binary cross-entropy**, exactly the L1 recipe with a Bernoulli plugged in. Differentiating gives the clean gradient $\partial J/\partial z_i = \hat y_i - y_i$ (prediction minus target), as in linear regression. For $K$ classes, use **softmax** $\hat y_k = e^{z_k}/\sum_j e^{z_j}$; the loss becomes $-\log\hat y_{\text{true}}$.

![w:420px](figures/lec01/svg/softmax_visual.svg)

*(Full derivation: ML ES 335 `logistic-regression.tex`, "Cross Entropy Cost Function".)*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · why not squared loss for a neuron?

Tempting: reuse MSE, $J(\theta)=\sum_i\big(\sigma(\theta^\top x_i)-y_i\big)^2$. **Don't** — it is **non-convex** in $\theta$.

$$\text{squared error} \;\circ\; \underbrace{\sigma(\cdot)}_{\text{nonlinear}} \;\Rightarrow\; \text{a bumpy surface with multiple local minima.}$$

Gradient descent can then stall in a bad basin. Cross-entropy paired with the sigmoid is **convex** for a single neuron, so the optimum is unique and reachable — one more reason every classifier in this course uses cross-entropy, not MSE. (Once we stack hidden layers the overall surface is non-convex regardless, but the *loss choice* still governs how gradients behave; more in L4.)

*(ML ES 335 `logistic-regression.tex`, "Learning Parameters / cost convexity".)*
