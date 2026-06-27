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
7. Separate **expressivity**, **optimization**, and **generalization** claims.
8. Explain projection shortcuts and pre-activation residual blocks.

---

# Recap · where we left off

- **MLPs** are stacks of affine + non-linearity (L1).
- **Backprop** is chain rule run right-to-left (L1).
- **Logistic / softmax classification = 1-layer MLP** under categorical NLL (L0 + L1).
- **Sigmoids vanish; ReLU un-blocks depth** (L1, end).
- **One hidden layer can approximate anything** (UAT) — we need to make that precise today.

<div class="paper">

Today maps to **UDL Ch 4** (deep networks), **Ch 7** (gradients &amp; init), **Ch 11** (residual networks). Read these three chapters before or after — whichever works for you.

</div>

<div class="keypoint">

Today's question · *if one hidden layer is universal, why do we ever go deeper?* Three answers · **width is exponential**, **depth is hierarchical**, and **depth is trainable** (with the right tools).

</div>

---

# Three axes we must not mix up

Deep learning progress depends on three different questions.

| Axis | Question | L02 answer |
|------|----------|------------|
| **Expressivity** | Can this architecture represent the function? | UAT and depth separation |
| **Optimization** | Can SGD find useful weights? | ReLU, residuals, initialization |
| **Generalization** | Will it work on unseen data? | not guaranteed by UAT |

<div class="warning">

A network can be expressive but untrainable. It can be trainable but overfit. It can fit train/val and still fail under distribution shift. Keep these axes separate in every DL paper you read.

</div>

---

# ▶ Interactives for L02

<div class="paper">

Play along while you read · all run in the browser ·

- [universal-approximation](https://nipunbatra.github.io/interactive-articles/universal-approximation/) · build any function from ReLU bumps.
- [vanishing-gradients](https://nipunbatra.github.io/interactive-articles/vanishing-gradients/) · sigmoid vs ReLU, live.
- [resnet](https://nipunbatra.github.io/interactive-articles/resnet/) · skip-connection gradient highway.

</div>

---

# Pop quiz · two architectures, same parameter budget

You have ~$15\,000$ parameters to spend on a regression task with 1-D input.

<div class="popquiz">

(a) **Wide-and-shallow** · 1 hidden layer with 5,000 ReLU units (≈ 15,001 params).
(b) **Tall-and-thin** · 50 hidden layers with 17 ReLU units each (≈ 15,046 params).

Which one would you bet on for fitting a *complex* function like $\sin(50 x)$ on $[0, 1]$?

Stop and decide. We'll come back to this exact question when we hit Telgarsky's separation — your gut answer should change once you've seen the result.

</div>

This is L02's central tension · **width** is universal but expensive, **depth** is exponentially more efficient *when it can be trained*. Today we earn both halves of that sentence.

---

# Bridge from ES 335 · you already know this

| From ES 335 | What changes today |
|---|---|
| You hand-picked the basis $\phi(x)=[1,x,x^2,\dots]$ to bend a boundary | **one wide hidden layer** of ReLUs *learns* a basis rich enough for **any** function (UAT) |
| Gradient descent derived from first-order Taylor expansion | that same gradient must now survive a **product of 50 Jacobians** |
| Weight init = "small random numbers" | init **derived** from variance preservation (Xavier / He) |
| One model = one function class | depth vs width · same budget, exponentially different reach |

<div class="insight">

In ES 335 you bent the boundary by **writing $\phi(x)=[1,x,x^2]$ by hand**. A wide hidden layer makes that automatic — it can approximate **any** function (UAT). The catch · matching a hard function this way needs **exponentially many** units. The fix that buys the same reach cheaply is **depth** — and that is the whole lecture.

</div>

---

<!-- _class: section-divider -->

### PART 1

# Universal approximation

What a single hidden layer can — and can't — do

---

# Build a bump from ReLU kinks · the intuition

![w:900px](figures/lec02/svg/uat_bump_construction.svg)

One bump = $\text{relu}(x-a) - 2\,\text{relu}(x-b) + \text{relu}(x-c)$ · stack many, weight by $\alpha_i$, approximate any curve.

<div class="realworld">

▶ Interactive: grow a 1-hidden-layer net and watch it fit a target curve — [universal-approximation](https://nipunbatra.github.io/interactive-articles/universal-approximation/).

</div>

---

# The "LEGO brick" idea · UAT in plain English

<div class="keypoint">

Imagine an unlimited supply of LEGO bricks. Can you build a sculpture of *anything*? A car, a house, the Eiffel Tower? **Yes** · if your bricks are small enough, you can approximate any shape.

UAT says · a neural network with one hidden layer can do the same for **mathematical functions**. Its "LEGO bricks" are simple functions built from neurons.

</div>

---

# UAT · unpacking the statement

<div class="math-box">

**"For any continuous function $f(\mathbf{x})$..."**
The "true" relationship in our data · e.g., `price = f(house_features)`.

**"...and any small error $\epsilon > 0$..."**
How close we want our approximation. Set $\epsilon = 0.01$ or whatever you need.

**"...there exists a network..."**
With one hidden layer of $N$ neurons. The theorem guarantees · for any $\epsilon$, some $N$ exists.

**"...whose output is within $\epsilon$ of $f$."**
$|\,f(\mathbf{x}) - \sum_{i=1}^N \alpha_i\, \sigma(\mathbf{w}_i^\top \mathbf{x} + b_i)| < \epsilon$

</div>

A weighted sum of neuron outputs · each neuron is a "LEGO brick."

---

# ⭐⭐⭐ Optional · UAT · the formal statement

<div class="math-box">

**Theorem** (Cybenko 1989 · Hornik 1991 · Leshno 1993)

For any continuous $f: [0,1]^d \to \mathbb{R}$ and any $\epsilon > 0$, there exist $N$, weights $\{\mathbf{w}_i, b_i, \alpha_i\}$ such that

$$\left| f(\mathbf{x}) - \sum_{i=1}^{N} \alpha_i \, \sigma(\mathbf{w}_i^\top \mathbf{x} + b_i) \right| < \epsilon$$

for any non-polynomial activation $\sigma$ — including ReLU.

</div>

One hidden layer suffices. The catch hides in one word: **exist.**

---

# ⭐⭐⭐ Optional · UAT · the proof in three moves (1/2)

We won't write a full proof — but the structure is short and worth knowing.

<div class="math-box">

**Move 1 · continuous $f$ → step function.** Any continuous $f$ on $[0,1]^d$ is *uniformly continuous* (Heine–Cantor). Partition the cube into cells small enough that $f$ varies by $<\epsilon/2$ inside each; replace $f$ by its average on each cell → a step function within $\epsilon/2$ of $f$.

**Move 2 · step function → sum of sigmoids.** A sigmoid $\sigma(w(x-b))$ with large $w$ is essentially a step at $b$; a *bump* is a difference of two near-steps. Each cell ⇒ one bump · $N$ cells → $2N$ hidden units. *(This clean count is the **1-D** picture; in $d$ dimensions one layer needs many more units per cell — the curse of dimensionality, ahead.)*

</div>

---

# ⭐⭐⭐ Optional · UAT · the proof in three moves (2/2)

<div class="math-box">

**Move 3 · density.** The space of finite sums $\sum_i \alpha_i\,\sigma(\mathbf{w}_i^\top \mathbf{x} + b_i)$ is **dense** in $C([0,1]^d)$ — a functional-analysis theorem (Stone–Weierstrass / Hahn–Banach). With Moves 1–2 this gives $\|f - \hat f\|_\infty < \epsilon$.

</div>

**Bottom line** · UAT is *not* a constructive recipe — it is a density theorem. It says good weights *exist*; it says nothing about $N$, generalization, or whether SGD finds them.

---

# Worked example · approximate `f(x) = x²` with 4 ReLUs

![w:920px](figures/lec02/svg/uat_relu_bumps.svg)

---

# Worked example · approximate `f(x) = x²` with 4 ReLUs · numbers

<div class="math-box">

Pick 4 ReLU **ramps** (one-sided hinges) at $x = 0.0, 0.25, 0.5, 0.75$ on $[0, 1]$. Each is `relu(w·(x − b))` for slope $w = 1$.

| ReLU $i$ | $b_i$ | $\alpha_i$ | turns on at $x =$ |
|:-:|:-:|:-:|:-:|
| 1 | 0.0  | 0.25 | 0.0 |
| 2 | 0.25 | 0.50 | 0.25 |
| 3 | 0.50 | 0.50 | 0.50 |
| 4 | 0.75 | 0.50 | 0.75 |

</div>

Each $\alpha_i$ adds to the running slope ($0.25, 0.75, 1.25, 1.75$), so the output **matches $x^2$ exactly at every knot**. Between knots the interpolation error is at most $\tfrac{h^2}{8}\max|f''| = \tfrac{0.25^2}{8}\cdot 2 = \tfrac{1}{64} \approx 0.016$ — and shrinks as you add ramps.

**That's UAT in numbers.** A weighted sum of ReLU ramps approximates any 1D continuous function.

---

# Two ReLUs · a soft step

A single ReLU is a ramp going up forever. The **difference of two ReLUs** flattens the ramp into a **soft step** ·

<div class="math-box">

$$\text{step}(x;\, a, b) = \text{relu}(x - a) - \text{relu}(x - b), \quad a < b$$

Zero before $a$ · rises linearly on $[a, b]$ · then **plateaus** at height $b - a$ forever after. (For $x > b$ the two ramps cancel each other's slope, not each other's value.)

</div>

Shifted, scaled soft steps stack into any **staircase** — exactly the step functions from Move 1 of the proof. But a step never comes back down. For a *localized* bump we need one more ReLU · next slide.

---

# Three ReLUs · a triangle bump

To go up **and come back down to zero**, add a third ReLU ·

<div class="math-box">

$f(x) = \text{relu}(x - 1) - 2 \cdot \text{relu}(x - 2) + \text{relu}(x - 3)$

| $x$ | relu($x{-}1$) | $-2\cdot$relu($x{-}2$) | relu($x{-}3$) | sum |
|:-:|:-:|:-:|:-:|:-:|
| 0   | 0   | 0  | 0 | **0** |
| 1.5 | 0.5 | 0  | 0 | **0.5** |
| 2   | 1   | 0  | 0 | **1.0** (peak) |
| 2.5 | 1.5 | -1 | 0 | **0.5** |
| 4   | 3   | -4 | 1 | **0** |

</div>

A perfect triangular bump centred at $x=2$. Tile the axis with such bumps, weight each by the function value there — and you can approximate any continuous function. **This is what UAT proves.** *Existence* is given by the construction; *training* is the open problem.

---

# The price of width — curse of dimensionality

Piecewise-linear approximation of $f$ to error $\epsilon$ — for *generic* (worst-case-smooth) functions the neuron count can grow **exponentially** in dimension:

- 1D: $N \approx O(1 / \sqrt{\epsilon})$
- $d$D: $N \approx O(1 / \epsilon^{d/2})$ *(illustrative — exact rate depends on the function class)*

| $d$ | linear pieces (grid cells) for $\epsilon = 0.01$ |
|:-:|:-:|
| 1   | ~10 |
| 10  | $10^{10}$ |
| 100 | $10^{100}$ |

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

# When depth helps — and when it doesn't

Depth helps when the target function has reusable substructure.

| Domain | Reusable structure | What deeper layers reuse |
|--------|--------------------|--------------------------|
| images | edges → textures → parts → objects | local visual motifs |
| language | characters → words → phrases → discourse | compositional meaning |
| audio | samples → phonemes → syllables → words | local temporal patterns |
| tabular data | often weaker hierarchy | depth may help less |

<div class="keypoint">

Depth is not magic. It is a strong inductive bias for **compositional** problems. If the data do not have reusable structure, a deeper model can simply be harder to optimize.

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

# Depth-vs-width · the formal separation

<div class="paper">

**Theorem (Telgarsky 2016, simplified)** · There exists a function $f : [0, 1] \to [0, 1]$ representable by a ReLU network of depth $2k$ and width $\le 3$, such that **any** ReLU network of depth $\le k$ approximating $f$ within constant error needs **at least $2^{k}$** units.

</div>

The witness function is the $k$-fold composition of the **sawtooth** ·
$$T(x) = \begin{cases} 2x & x \le 1/2 \\ 2 - 2x & x > 1/2\end{cases}$$

Each composition doubles the number of "teeth" — depth $k$ gives $2^k$ teeth using $O(k)$ ReLUs. A shallow net **must enumerate** every tooth · exponential width.

<div class="keypoint">

**Depth is exponentially more parameter-efficient than width** for problems with compositional / recursive structure. Real-world data (images, language) is richly compositional · so depth pays off in practice.

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

# Backprop · term-by-term, no Jacobians

A 4-layer scalar network · `y = w_4 · w_3 · w_2 · w_1 · x` (ignoring activations).

We want gradient of loss $L$ with respect to the **first** weight $w_1$. Chain rule, one link at a time:

<div class="math-box">

$\dfrac{\partial L}{\partial w_1} = \dfrac{\partial L}{\partial y} \cdot \dfrac{\partial y}{\partial w_1}$

Expand $\partial y/\partial w_1$:
- $\partial y/\partial(\text{layer 3 out}) = w_4$
- $\partial(\text{layer 3 out})/\partial(\text{layer 2 out}) = w_3$
- $\partial(\text{layer 2 out})/\partial(\text{layer 1 out}) = w_2$
- $\partial(\text{layer 1 out})/\partial w_1 = x$

So · $\dfrac{\partial L}{\partial w_1} = \dfrac{\partial L}{\partial y} \cdot (w_4 \cdot w_3 \cdot w_2) \cdot x$

</div>

The key is the **product** $w_4 \cdot w_3 \cdot w_2$. If any of these is small (< 1), the product shrinks fast.

---

# Numeric · vanishing with sigmoids

Sigmoid derivative · $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ · max value **0.25**.

Each backward step multiplies by $w_l \cdot \sigma'$. Assume weights ≈ 1.0:

<div class="math-box">

| Layer (counting back) | Cumulative factor |
|:-:|:-:|
| L | 0.25 |
| L−1 | $0.25^2 = 0.0625$ |
| L−2 | $0.25^3 = 0.0156$ |
| L−5 | $0.25^5 \approx 10^{-3}$ |
| L−10 | $0.25^{10} \approx 10^{-6}$ |

</div>

After 10 sigmoid layers, the gradient signal at the **earliest** weight is shrunk by a million. **The first layer barely updates · learning stalls.**

This is the core problem ResNets and ReLU were invented to solve.

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

# Degradation is not overfitting

Compare the signatures:

| Failure mode | Training error | Validation / test error | Main diagnosis |
|--------------|----------------|--------------------------|----------------|
| Underfitting | high | high | model or training too weak |
| Overfitting | low | high | generalization failure |
| Degradation | higher for deeper model | higher for deeper model | optimization failure |

<div class="keypoint">

The ResNet paper mattered because it exposed a surprising fact: simply adding layers can make the **training objective itself** harder, even though the deeper model has more representational capacity.

</div>

---

# The thought experiment

![w:920px](figures/lec02/svg/identity_thought_exp.svg)

---

# Why "learning the change" is easier · steering analogy

<div class="keypoint">

Imagine steering a car by giving the wheel an *absolute* angle (e.g., "set wheel to 27 degrees from zero"). Hard to do.

Instead, you say "**turn a little right**" or "**stay straight**." Much easier.

</div>

ResNets do the same · they reframe each layer's job from *"output the right thing"* to *"output a small change to your input."* The default action — change nothing, pass input through — is now trivial. The network only learns *deviations* from identity, which is much easier for SGD.

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

# BatchNorm in the ResNet story

The original ResNet block was not just "add a skip connection." It used a stack like:

$$\text{Conv} \rightarrow \text{BatchNorm} \rightarrow \text{ReLU} \rightarrow \text{Conv} \rightarrow \text{BatchNorm} \rightarrow +x \rightarrow \text{ReLU}$$

| Component | What it helps with |
|-----------|--------------------|
| Skip connection | direct signal and gradient path |
| BatchNorm | stable activation scale across layers |
| ReLU | nonlinearity with active-side gradient 1 |
| He initialization | variance scale matched to ReLU |

<div class="keypoint">

Do not learn the wrong lesson: ResNets train because several engineering choices work together. The skip connection is the central idea, but scale control is part of the system.

</div>

---

# How the skip-connection makes a gradient highway

Forward · $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$.

We want the gradient flowing back through the block · $\partial L/\partial \mathbf{x}$.

<div class="math-box">

By the chain rule · $\dfrac{\partial L}{\partial \mathbf{x}} = \dfrac{\partial L}{\partial \mathbf{y}} \cdot \dfrac{\partial \mathbf{y}}{\partial \mathbf{x}}$

Compute the local gradient:
$\dfrac{\partial \mathbf{y}}{\partial \mathbf{x}} = \dfrac{\partial}{\partial \mathbf{x}}(\mathcal{F}(\mathbf{x}) + \mathbf{x}) = \dfrac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I}$

</div>

So · $\dfrac{\partial L}{\partial \mathbf{x}} = \dfrac{\partial L}{\partial \mathbf{y}} \cdot \big( \dfrac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I} \big) = \dfrac{\partial L}{\partial \mathbf{y}}\,\dfrac{\partial \mathcal{F}}{\partial \mathbf{x}} \;+\; \underbrace{\dfrac{\partial L}{\partial \mathbf{y}}}_{\text{the highway!}}$

The "+I" gives an **uninterrupted express lane** for the gradient · the second term is the original signal **passing straight through** even if the first term vanishes. The early layers always get a clean signal.

---

# Numeric · gradient flow with vs without skip

Same 5-layer net. Each layer's $\partial \mathcal{F}/\partial \mathbf{x}$ has tiny norm 0.1 (saturated regime).

<div class="math-box">

| Layer | Plain (multiplicative) | ResNet (additive) |
|:-:|:-:|:-:|
| L (output) | 1.0 | 1.0 |
| L−1 | 0.1 | 1.1 |
| L−2 | 0.01 | 1.21 |
| L−3 | 0.001 | 1.33 |
| L−4 | **0.0001** | **1.46** |

</div>

The plain net's signal vanishes; the ResNet's $+\mathbf{I}$ keeps a **direct path** open, so the gradient still flows even when each block's Jacobian is tiny. (Real blocks add matrices, ReLUs and projections — a strong *help*, not a literal guarantee.) **Hundred-layer ResNets train; 100-layer plain nets don't.**

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

# Projection shortcuts when shapes change

The addition requires matching shapes. If the block changes width, channels, or spatial resolution, the skip path must also transform $x$.

```python
class ResidualBlock(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(d_in, d_out), nn.ReLU(),
            nn.Linear(d_out, d_out),
        )
        self.skip = nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out)

    def forward(self, x):
        return torch.relu(self.block(x) + self.skip(x))
```

In CNNs this is often a $1 \times 1$ convolution, sometimes with stride 2, so both branches land on the same shape before addition.

---

# Pre-activation ResNets

Later ResNets moved normalization and activation **before** the weight layers:

| Block style | Formula sketch | Why it matters |
|-------------|----------------|----------------|
| Post-activation | $y = \mathrm{ReLU}(x + F(x))$ | original ResNet |
| Pre-activation | $y = x + F(\mathrm{BN/ReLU}(x))$ | cleaner identity path |

<div class="keypoint">

Pre-activation keeps the skip path as close to a pure identity as possible. This makes very deep residual networks easier to optimize because the gradient highway is less obstructed.

</div>

Transformers use the same idea in another form: residual stream plus normalization around each block.

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

# What to check in a real model

Initialization theory is useful because it gives a concrete diagnostic: **activation statistics by layer**.

For one batch, log:

| Quantity | Healthy early signal | Red flag |
|----------|----------------------|----------|
| activation mean | near 0 for normalized layers | large drift |
| activation std | roughly stable across depth | shrinks to 0 or explodes |
| gradient norm | nonzero in early layers | first layers near 0 |
| fraction ReLU active | neither 0% nor 100% | many dead units |

```python
for name, p in model.named_parameters():
    if p.grad is not None:
        print(name, p.grad.norm().item())
```

<div class="keypoint">

Before changing architecture, check whether the signal survives the forward pass and the gradient survives the backward pass.

</div>

---

# Why initialization matters · the failure mode

If we initialize weights from $\mathcal{N}(0, 1)$ in a deep ReLU net:

<div class="columns">
<div>

### Activations

Each layer multiplies by a matrix of N(0,1) values. Activation magnitudes either **grow exponentially** with depth (explode) or **shrink exponentially** (vanish) depending on shape.

</div>
<div>

### Gradients

Same product, going backwards. A single bad scale → all early layers train at $10^{-30}$ effective rate · they never move.

</div>
</div>

<div class="warning">

**Symptom · loss is NaN at step 1**, or loss flat with all weights stuck. Often an init problem — but first rule out a too-high LR, unnormalized inputs, a logits/loss mismatch, or AMP overflow.

</div>

The variance argument on the next slide gives a one-line fix · scale init so activation RMS is preserved — $\sqrt{1/n_\text{in}}$ (Xavier, for tanh) or $\sqrt{2/n_\text{in}}$ (He, for ReLU).

---

# Variance · three regimes across layers

![w:920px](figures/lec02/svg/variance_flow.svg)

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

# Worked numeric · variance flow over 10 layers

10-layer fully-connected ReLU net with $n_\text{in} = 512$ everywhere. Initial activation variance $\text{Var}(x) = 1$.

<div class="math-box">

**Naive init** $W \sim \mathcal{N}(0, 1)$ (no scaling) ·
- Each layer · $\text{Var}(z) = n_\text{in} \cdot \text{Var}(W) \cdot \text{Var}(x) = 512 \cdot 1 \cdot 1 = 512$.
- After ReLU · $\approx 256$.
- Pass through layer 2 · $\text{Var}(z) = 512 \cdot 1 \cdot 256 = 131{,}072$. **Explodes.**
- After 10 layers · $(256)^{10} \approx 10^{24}$. NaN at step 1.

**He init** $W \sim \mathcal{N}(0,\, 2/512)$ ·
- Each layer · $\text{Var}(z) = 512 \cdot (2/512) \cdot 1 = 2$.
- After ReLU · $\approx 1$.
- Stable for 10, 100, or 1000 layers. ✓

</div>

This is **why initialization is not optional**. Bad init → loss is NaN at step 1, or weights are stuck at $10^{-30}$ scale and never move. Good init keeps signal magnitude constant across depth.

---

# Initialization in PyTorch

```python
# nn.Linear's default is Kaiming-uniform with a=sqrt(5) — a legacy choice,
# NOT textbook He-for-ReLU. For deep ReLU nets, set it explicitly:
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

# Pop quiz · the mismatched init

<div class="popquiz">

**Pop quiz.** You build a 10-layer MLP with **Tanh** activations and **He** initialization. Loss oscillates, activations saturate. Why?

*Stop and reason before the next slide.*

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

# Putting it all together · the L02 master sentence

<div class="math-box">

**Depth is mathematically expressive (UAT, Telgarsky), but practically fragile.**
Three forces conspire against naive deep nets · vanishing/exploding **gradients**, vanishing/exploding **activations**, and the **degradation problem** even when both are tame.

</div>

| Symptom | Root cause | Fix introduced |
|:-:|:-:|:-:|
| Gradient → 0 in early layers | $\sigma'$ ceiling 0.25 + product of small Jacobians | ReLU family · skip connections |
| Activation variance blows up / shrinks | Wrong init scale per layer | Xavier (tanh) · He (ReLU) |
| Deeper net is *worse* at training loss | Optimization, not capacity | Residual blocks $\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})$ |

The single insight underneath all three fixes · **make every layer easy to leave alone.** Identity-friendly initialization, identity skip-connections, and activations that pass gradients through unchanged in their linear regime. That's what made depth practical in 2015 and onward.

---

# What breaks · symptom → suspect → test

| Symptom | Suspect | Fastest test |
|---|---|---|
| Loss flat from step 0 | vanishing gradients · bad init | print per-layer gradient norms after one backward |
| Loss `NaN` at step 1 | exploding activations · init variance too large | log activation std per layer on one forward pass |
| Deeper net trains *worse* than shallow | degradation (optimization, not capacity) | compare **training** curves · add skip connections |
| Tanh net saturates, loss oscillates | He init on a non-ReLU activation | histogram activations · switch to Xavier |
| Many units output 0 forever | dead ReLUs (LR too high · bad init) | log fraction of active ReLUs per layer |

---

# Practice problems

<div class="math-box">

**P1.** UAT says $f(x) = x^2$ on $[0, 1]$ can be approximated to error $\epsilon$ by a sum of $N$ ReLUs. Estimate $N$ as a function of $\epsilon$. (Hint · piecewise-linear with $N$ breakpoints has error $\sim 1/N^2$ for smooth $f$.)

**P2.** A 5-layer plain MLP with sigmoid activations is failing to train. Without changing the architecture, name **two** changes that would help and explain why each works.

**P3.** Show that for a ResNet block $\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})$, $\partial \mathbf{y}/\partial \mathbf{x} = I + \partial \mathcal{F}/\partial \mathbf{x}$. Use this to argue that even if $\mathcal{F}$ has tiny Jacobian, gradient through the residual block does not vanish.

**P4.** A 100-layer ReLU MLP with $n_\text{in} = 256$ everywhere uses Xavier init $W \sim \mathcal{N}(0, 1/n_\text{in})$. Will the activation variance grow, shrink, or stay constant? Why is this wrong for ReLU? What's the fix?

**P5.** Telgarsky's separation says depth-$2k$ ReLU nets need $\ge 2^k$ units to be matched by depth-$k$ nets. Plug in $k = 10$ · how many shallow units? Why does this argue for going deep?

**P6.** You replace ReLU with leaky ReLU ($\max(0.01 z, z)$) in a 50-layer net. (a) What changes for forward-pass variance? (b) What changes for vanishing gradients on initially-negative pre-activations?

</div>

---

<!-- _class: summary-slide -->

# Lecture 2 — summary

- **UAT** is an existence theorem. Width can be exponential; depth is the practical knob.
- **Compositionality** (and parity) shows depth can replace exponential width.
- **Vanishing gradients** come from products of sub-1 Jacobians; ReLU unlocks depth by giving gradient 1 on the active side.
- **Degradation problem** — plain deep nets *train* worse.
- **ResNets** — $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$. Identity-in-the-Jacobian gradient highway + smoother landscape.
- **ResNet practice** — BatchNorm, projection shortcuts, and pre-activation blocks keep the identity path usable.
- **Xavier / He** — both derived from variance preservation.
- **Always separate axes:** expressivity, optimization, and generalization are different claims.

### Read before Lecture 3

**Prince** — Ch 4, Ch 11. Free at [udlbook.github.io](https://udlbook.github.io/udlbook/).

### Next lecture

Tensors, autograd, `nn.Module`, `DataLoader`, the full training recipe, debugging ladder, error analysis.

<div class="notebook">

**Notebook 2** · `02-depth-and-resnets.ipynb` — shallow-wide vs deep-narrow on spirals; build a residual block; visualize gradient norms across depth.

</div>

---

# The one-sentence takeaway

<div class="insight">

**Depth buys exponential feature reuse — if gradients survive the trip.**

</div>

*Next (L03) · the training loop in practice — where silent failure is the default.*
