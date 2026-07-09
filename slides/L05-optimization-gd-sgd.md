---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Optimization I: Gradient Descent to SGD

## Lecture 5 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

Every model in this course is trained by one move, repeated: **stand on the loss surface, feel the downhill direction, take a step.**

<div class="keypoint">

**Gradient descent is that step; SGD is the same step done on a cheap, noisy estimate of the gradient.**
The surprising part is that the *noise is free* — a random minibatch gradient points the right way *on average*, so we get hundreds of cheap steps for the price of one exact one. That trade is what makes training billion-parameter models possible.

</div>

You wrote down losses $J(\theta)$ in L1 and built the models that produce them in L2–L4. Today we answer: **how do we actually minimize $J(\theta)$ when there's no formula for the answer?**

$$\text{steepest descent} \rightarrow \underbrace{\text{the GD step}}_{\theta \leftarrow \theta-\eta\nabla J} \rightarrow \underbrace{\text{batch}\to\text{mini}\to\text{stochastic}}_{\text{compute argument}} \rightarrow \underbrace{\mathbb E[\hat g]=\nabla J}_{\text{noise is unbiased}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**You'll reuse today's ideas in:**
- **L6** — momentum & Adam: same step, smarter direction and step size
- **L7** — weight decay is just a gradient on the regularizer
- **L8+** — every CNN, RNN, Transformer is trained by minibatch SGD
- **L14–L18** — LLM pretraining is SGD on cross-entropy, at scale

</div>
<div>

**Borrowed from your own courses** (we build on, not repeat):
- gradient, Taylor series, convexity — **Maths for ML**
- GD for linear regression, the normal equation — **ML (ES 335)**
- MSE / cross-entropy as the loss $J(\theta)$ — **L1 of this course**

</div>
</div>

<div class="insight">

This is **Optimization I**: today is the *engine* (plain GD, then SGD). L6 is the *tuning* (momentum, adaptive rates, schedules) that makes it fast and less fiddly.

</div>

---

<!-- _class: section-divider -->

## Part 1 · Gradient descent — the update rule

---

# Optimization is the whole game

Training *is* minimizing a cost: $\;\theta^\* = \arg\min_\theta J(\theta)$.

<div class="columns">
<div>

- Linear regression: minimize $\lVert y-\mathbf X\theta\rVert^2$
- Logistic regression: minimize cross-entropy
- A neural net: minimize the same losses, composed through many layers

</div>
<div>

<div class="keypoint">

**The catch.** Linear regression has a closed form ($\theta=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top y$). Almost nothing else does. Neural nets have **no formula** for the minimum — so we search for it, iteratively.

</div>

</div>
</div>

The search is **gradient descent**: repeatedly step in the steepest-downhill direction. The rest of Part 1 derives *why that direction is $-\nabla J$*.

---

# Gradient intuition: hiking in fog

You're on a hillside in dense fog and want the valley. You can't see the map — you can only feel the **slope under your feet**. Strategy: always step in the steepest *downhill* direction.

![w:820px](figures/lec04/svg/gd_intuition.svg)

The **gradient** $\nabla f$ points in the direction of steepest *ascent*; so $-\nabla f$ points steepest *descent*. That single fact is the whole algorithm — we just have to prove it.

---

# Why the gradient is perpendicular to the level sets

A **level set** is $\{x : f(x)=c\}$ — a contour of constant height. Walk *along* a contour and $f$ never changes.

<div class="math-box">

Parameterize the contour as $x(t)$, so $f(x(t))=c$. Differentiate:
$$\frac{d}{dt}f(x(t)) = \nabla f(x)^\top x'(t) = 0$$

</div>

So $\nabla f \perp x'(t)$ for *every* direction $x'(t)$ tangent to the contour. The gradient always points **straight across** the contours — the most direct way uphill, and $-\nabla f$ the most direct way down.

---

# From Taylor series to the step direction

We can't solve $\min f$ directly, so approximate $f$ **locally** with its first-order Taylor expansion around $x_0$:

$$f(x_0+\Delta x) \;\approx\; f(x_0) + \nabla f(x_0)^\top \Delta x$$

For the value to *decrease* we need $\nabla f(x_0)^\top \Delta x < 0$. Using $\;a^\top b = \lVert a\rVert\lVert b\rVert\cos\theta$:

<div class="insight">

This dot product is **most negative** when $\cos\theta=-1$ — i.e. $\Delta x$ points *exactly opposite* the gradient. The steepest-descent direction is $\Delta x = -\eta\,\nabla f(x_0)$, for a step size $\eta>0$.

</div>

---

# The gradient descent update rule

<div class="math-box">

**Gradient descent**
1. Initialize $\theta_0$.
2. Repeat: $\quad\theta_{t+1} = \theta_t - \eta\,\nabla J(\theta_t)$
3. Stop when $\lVert\nabla J\rVert$ is tiny (or you run out of patience).

</div>

Two things worth noticing:

- The steps **shrink automatically** near a minimum, because $\lVert\nabla J\rVert\to 0$ there — you glide in, you don't slam in.
- The only knob is the **learning rate** $\eta$. Everything hard about GD is choosing it.

---

# The learning rate is the whole ballgame

For $f(x)=x^2$ started far from $0$, the same algorithm does four completely different things:

| Learning rate $\eta$ | Behavior | Why |
|---|---|---|
| **too small** (0.01) | crawls down, stable | steps a fraction of the slope; needs many iterations |
| **just right** (0.1) | fast *and* stable | big enough to move, small enough not to overshoot |
| **too large** (0.8) | overshoots, oscillates | jumps past the minimum, bounces across the valley |
| **way too large** ($>1$) | **diverges** | each step lands *farther* out; loss explodes |

<div class="keypoint">

**Goldilocks.** Start around $\eta\in[0.01,\,0.1]$ and read your loss curve: exploding $\Rightarrow$ lower it; barely moving $\Rightarrow$ raise it.

</div>

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** Let $f(x)=(x-3)^2$, start at $x_0=0$ with learning rate $\eta=0.1$.

(a) Compute $f'(x_0)$.
(b) Take one GD step — what is $x_1$?
(c) Did $f$ decrease?
(d) *Bonus:* what single $\eta$ would land you **exactly** at the minimum in one step?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

$f'(x)=2(x-3)$, so $f'(0) = -6$.

$$x_1 = x_0 - \eta f'(x_0) = 0 - 0.1(-6) = \mathbf{0.6}$$

(c) $f(0)=9 \to f(0.6)=(-2.4)^2 = 5.76$ — **yes, it decreased**, and $0.6$ is closer to the minimum $x^\*=3$.

<div class="keypoint">

**Bonus.** For a quadratic, the perfect one-step rate is $\eta=1/f''=1/2=0.5$: then $x_1 = 0-0.5(-6)=3$ exactly. That "divide by the curvature" idea is exactly what Newton's method — and, loosely, Adam (L6) — is reaching for.

</div>

---

<!-- _class: section-divider -->

## Part 2 · The loss landscape

---

# The easy case: convex bowls

A function is **convex** if the chord between any two points lies *on or above* the curve — equivalently, $f''(x)\ge 0$ (Hessian positive semidefinite in higher dimensions).

<div class="columns">
<div>

$x^2$, $|x|$, $e^x$ are convex; $\log x$ and $x^3$ are not (on all of $\mathbb R$).

</div>
<div>

<div class="keypoint">

**Why we care.** On a convex loss, *every* local minimum is the global minimum, and GD is guaranteed to find it. Linear and logistic regression are convex. Neural nets are **not** — which is where the trouble starts.

</div>

</div>
</div>

---

# Three kinds of critical points

Where $\nabla f = 0$, GD stalls. But not all stalls are equal:

![w:900px](figures/lec04/svg/saddle_minima.svg)

In high dimensions almost every critical point is a **saddle**, not a minimum — flat in some directions, downhill in others. Escaping saddles, not avoiding local minima, is the real challenge (and where SGD's noise will earn its keep in Part 6).

---

# Not all bowls are round: the condition number

Rescale one axis and the round bowl becomes a long, narrow **ravine**. The **condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$ of the Hessian measures how stretched it is.

![w:900px](figures/lec04/svg/condition_number.svg)

Big $\kappa$ means the surface is steep across the valley but nearly flat along it — and a single global $\eta$ can't be right for both directions at once.

---

# Why plain GD zig-zags on a ravine

$\eta$ small enough to be stable *across* the steep walls is far too small to make progress *along* the gentle floor. So GD bounces wall-to-wall, creeping forward:

![w:900px](figures/lec04/svg/ravine_zigzag.svg)

<div class="insight">

This is the single most common failure of vanilla GD/SGD on real losses. **Momentum (L6)** fixes it by averaging out the sideways bounces and building speed along the floor.

</div>

---

# A practical fix you already know: scale your features

Ravines often come from features on wildly different scales (age in $[0,100]$, income in $[0,10^6]$). Standardizing each feature — $x \leftarrow (x-\mu)/\sigma$ — makes the bowl rounder, shrinks $\kappa$, and lets a single $\eta$ work.

![w:820px](figures/lec04/svg/loss_landscape_ravine.svg)

**Same optimizer, better-conditioned surface, far fewer steps.** Preprocessing is optimization.

---

# See the geometry move

<div class="notebook">

**📓 Notebook · GD on a 2-D surface** — watch the iterates crawl down a contour plot as you change $\eta$; see the zig-zag appear when the bowl is stretched. *(ML ES 335 · `notebooks/gradient-descent-2d.ipynb`)*

</div>

<div class="notebook">

**🎛 Interactive · SGD on a ravine** — a scroll-driven explainer: drag the condition number $\kappa$ and the step size, and watch plain GD zig-zag while momentum glides. *(Interactive Lab · to be authored at `~/git/interactive/articles/sgd-ravine`)*

</div>

---

# Practice problem 2 · read the loss curves

<div class="popquiz">

**Practice problem 2.** You run the same model at four learning rates and plot **loss vs. iteration**:

- **A** — shoots upward to $\infty$ within a few steps
- **B** — drops fast, then flattens at a *low* value
- **C** — drifts down very slowly, still far from flat at the end
- **D** — drops, then bounces up and down around a middling value forever

Match each curve to: *too small · just right · too large · way too large.* Which $\eta$ do you ship?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

| Curve | Diagnosis | $\eta$ |
|---|---|---|
| **A** explodes | **way too large** — diverging | drop it hard |
| **D** bounces forever | **too large** — overshooting the valley | lower it |
| **C** crawls | **too small** — stable but wasteful | raise it |
| **B** fast to a low floor | **just right** | **ship this one** |

<div class="keypoint">

You can diagnose a training run from the loss curve alone: **explode $\to$ diverging, oscillate $\to$ too big, crawl $\to$ too small, smooth descent to a low floor $\to$ just right.** This single skill saves more GPU-hours than any other.

</div>

---

<!-- _class: section-divider -->

## Part 3 · A gradient descent step, worked by hand

---

# Setup: gradient descent for linear regression

Fit $\;\hat y = \theta_0 + \theta_1 x\;$ to three points $(1,1),(2,2),(3,3)$ — true line is $y=x$.

<div class="columns">
<div>

**Cost (MSE):**
$$J(\theta) = \frac1n\sum_{i=1}^n (y_i-\theta_0-\theta_1 x_i)^2$$

</div>
<div>

**Gradients** (with residual $\epsilon_i = y_i-\hat y_i$):
$$\frac{\partial J}{\partial\theta_0} = -\frac2n\sum_i \epsilon_i,\quad
\frac{\partial J}{\partial\theta_1} = -\frac2n\sum_i \epsilon_i x_i$$

</div>
</div>

Start at $\theta_0=4,\ \theta_1=0$ (a deliberately bad guess), learning rate $\eta=0.1$.

---

# Worked step: predictions and residuals

With $\theta_0=4,\ \theta_1=0$, every prediction is $\hat y = 4+0\cdot x = 4$:

| $x_i$ | $y_i$ | $\hat y_i$ | $\epsilon_i = y_i-\hat y_i$ |
|---|---|---|---|
| 1 | 1 | 4 | $-3$ |
| 2 | 2 | 4 | $-2$ |
| 3 | 3 | 4 | $-1$ |

Current loss: $J = \tfrac13(9+4+1) = \tfrac{14}{3} \approx 4.67$. Everything below flows from the residual column $(-3,-2,-1)$.

---

# Worked step: gradients and the update

$$\frac{\partial J}{\partial\theta_0} = -\frac23(-3-2-1) = -\frac23(-6) = 4$$
$$\frac{\partial J}{\partial\theta_1} = -\frac23\big((-3)(1)+(-2)(2)+(-1)(3)\big) = -\frac23(-10) = 6.67$$

Apply $\theta\leftarrow\theta-\eta\nabla J$:

$$\theta_0 = 4 - 0.1(4) = \mathbf{3.6}, \qquad \theta_1 = 0 - 0.1(6.67) = \mathbf{-0.67}$$

<div class="keypoint">

New loss at $(3.6,-0.67)$ is $\approx 1.93$ — **down from 4.67 in one step.** (Notice $\theta_1$ even moved *away* from its target first; GD only promises the loss drops, not that every coordinate marches straight to its answer.)

</div>

---

# See it in code

<div class="notebook">

**📓 Notebook · fit a line by gradient descent** — run these exact updates in a loop and watch $(\theta_0,\theta_1)$ spiral into the minimum over a contour plot of $J$. *(ML ES 335 · `notebooks/gradient-descent-fit-contour.ipynb`)*

</div>

<div class="insight">

For *this* problem the normal equation $\theta=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top y$ gives the answer in one shot. We do it iteratively anyway — because the moment the model is nonlinear (any neural net), the closed form is gone and iteration is *all we have*. (Cost comparison in the Appendix.)

</div>

---

<!-- _class: section-divider -->

## Part 4 · Batch → mini-batch → stochastic

---

# The compute wall: why full-batch won't scale

The batch gradient sums over **every** example before taking **one** step:

$$\nabla J(\theta) = \frac1n\sum_{i=1}^n \nabla \ell\big(f(x_i;\theta),\,y_i\big)$$

<div class="keypoint">

On ImageNet ($n\approx 1.3\times 10^6$) or web-scale text ($n$ in the billions), computing that full sum is one forward+backward pass over the *entire dataset* — just to move the parameters *once*. You'd take a handful of steps a day. **We need a cheaper estimate of the gradient.**

</div>

The fix: don't wait for all $n$. Estimate $\nabla J$ from a small random sample and step **now**.

---

# The gradient descent family

![w:860px](figures/lec04/svg/sgd_zoo.svg)

| Method | Data per update | Updates / epoch | Path |
|---|---|---|---|
| **Batch GD** | all $n$ | 1 | smooth, expensive |
| **Stochastic (SGD)** | 1 sample | $n$ | noisy, cheap |
| **Mini-batch** | a batch of $b$ (32–256) | $n/b$ | balanced — **the standard** |

---

# One vocabulary slide (so we're precise later)

| Term | Meaning |
|---|---|
| **Iteration / step** | one parameter update, using one (mini)batch |
| **Batch size $b$** | how many examples per update |
| **Epoch** | one full pass through all $n$ examples $=n/b$ iterations |

<div class="insight">

Batch GD does **1 update per epoch**; SGD with $b=1$ does **$n$ updates per epoch**. Same data seen, but SGD has already stepped thousands of times before batch GD finishes its first — that's why it's so much faster in wall-clock terms.

</div>

---

# The payoff, in one picture

Full-batch spends an entire pass to take one step. SGD spends that same pass taking *hundreds* of cheap steps — and is far down the loss before full-batch has moved once.

![w:780px](figures/lec04/svg/sgd_vs_fullbatch_progress.svg)

The catch: those cheap steps are **noisy**. Part 5 proves the noise is harmless (unbiased), and Part 6 shows it's actually *helpful*.

---

# The same step, worked stochastically

Same data, same start $\theta_0=4,\theta_1=0,\ \eta=0.1$ — but update on **one** sample $(x_1,y_1)=(1,1)$ using the per-example loss $\ell=(y-\hat y)^2$:

$$\hat y_1 = 4,\quad \epsilon_1 = 1-4 = -3$$
$$\frac{\partial\ell_1}{\partial\theta_0} = -2\epsilon_1 = 6,\qquad \frac{\partial\ell_1}{\partial\theta_1} = -2\epsilon_1 x_1 = 6$$
$$\theta_0 = 4-0.1(6) = \mathbf{3.4},\qquad \theta_1 = 0-0.1(6) = \mathbf{-0.6}$$

<div class="keypoint">

Batch GD went to $(3.6,-0.67)$; SGD went to $(3.4,-0.6)$ — a **different path** from a **different (cheaper) gradient**. Both head downhill; SGD just wobbles.

</div>

---

# Mini-batch: the sweet spot

A minibatch gradient is a **noisy estimate** of the true one — cheap enough to step thousands of times, accurate enough to point roughly right.

![w:720px](figures/lec04/svg/minibatch_noise.svg)

<div class="insight">

Why $b=32$–$256$, not $b=1$? A batch of $32$ is not $32\times$ slower on a GPU — it's *nearly free* thanks to parallel hardware, and it cuts the gradient noise. Mini-batch buys stability **and** GPU throughput at once.

</div>

---

<!-- _class: section-divider -->

## Part 5 · Why the noise is safe — the unbiased-estimator proof

---

# First, the key fact: the gradient is linear

Because $\nabla$ is a linear operator, the gradient of the *average loss* is the *average of the gradients*:

<div class="math-box">

$$\nabla J(\theta) = \nabla\!\left(\frac1n\sum_{i=1}^n \ell_i(\theta)\right) = \frac1n\sum_{i=1}^n \nabla \ell_i(\theta)$$

</div>

So the true gradient is literally the **mean** of the $n$ per-example gradients. That reframes the question: *if the answer is a mean, can we estimate it by sampling?* Yes — that's exactly what a minibatch is.

---

# Practice problem 3 · prove SGD is unbiased

<div class="popquiz">

**Practice problem 3 (guided).** Pick index $j$ **uniformly at random** from $\{1,\dots,n\}$ and use $\hat g = \nabla\ell_j(\theta)$ as your gradient estimate. Show that

$$\mathbb E[\hat g] = \nabla J(\theta).$$

*Hints:* (1) write $\mathbb E$ as a sum over which sample was drawn; (2) what is $P(\text{draw } i)$? (3) use linearity from the previous slide.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

<div class="math-box">

$$
\begin{aligned}
\mathbb E[\hat g] &= \mathbb E\big[\nabla\ell_j(\theta)\big] \\
&= \sum_{i=1}^n P(\text{draw } i)\,\nabla\ell_i(\theta) &&\text{(expectation over the random draw)}\\
&= \sum_{i=1}^n \frac1n\,\nabla\ell_i(\theta) &&\big(P(\text{draw } i)=\tfrac1n\big)\\
&= \frac1n\sum_{i=1}^n \nabla\ell_i(\theta) &&\text{(linearity)}\\
&= \nabla J(\theta). &&\blacksquare
\end{aligned}
$$

</div>

**On average, the one-sample gradient points exactly where the full gradient points.** A minibatch of size $b$ is the average of $b$ such draws, so it's unbiased too.

---

# Unbiased, and the noise shrinks with batch size

Individual gradients scatter around the true direction; their **average** *is* the true direction. Bigger batches average away more of the scatter:

![w:800px](figures/lec04/svg/gradient_variance.svg)

<div class="keypoint">

**Bias vs. variance of the estimate.** $\mathbb E[\hat g]=\nabla J$ (zero bias, any batch size). And for $b$ iid samples, $\operatorname{Var}(\hat g) \propto \tfrac1b$ — doubling the batch halves the noise but **never** changes the aim. This is the entire theoretical license for training on minibatches.

</div>

---

# The picture behind the proof

<div class="columns">
<div>

Ask six random people for the direction to the station. Each is a bit off — but if no one is *systematically* biased, their **average** points true.

</div>
<div>

<div class="insight">

SGD does exactly this with gradients. Any *single* minibatch step may head slightly wrong, but across many steps the errors are random, not systematic, so they cancel — and the parameters march downhill.

</div>

</div>
</div>

So the noise costs us nothing in expectation. Next: it actually **helps**.

---

<!-- _class: section-divider -->

## Part 6 · Why the noise even helps

---

# Noise escapes saddles and shallow traps

Batch GD is *deterministic*: land on a saddle where $\nabla J=0$ and it sits there forever. SGD almost never sees an exact zero — its minibatch gradient jitters, nudging the parameters off the saddle and back downhill.

![w:720px](figures/lec04/svg/saddle_minima.svg)

<div class="keypoint">

In the million-dimensional, non-convex landscapes of deep nets — which are *dominated* by saddles — this stochastic jitter is not a bug to be tolerated but a **feature that keeps training moving.** The very noise Part 5 proved harmless turns out to be useful.

</div>

---

# Where this goes next

<div class="columns">
<div>

**Today you can:**
- derive the GD step from Taylor series
- diagnose a run from its loss curve
- explain why we use minibatches, and prove the noise is unbiased

</div>
<div>

**Still fiddly:**
- one global $\eta$ can't fit a ravine
- vanilla SGD zig-zags and crawls
- picking $\eta$ by hand is painful

</div>
</div>

<div class="keypoint">

**Next (L6) — momentum & Adam.** Add a *velocity* that averages out the zig-zag, and a *per-parameter* step size that adapts to each direction's curvature. Same downhill step; far faster, far less hand-tuning.

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Step opposite the gradient; when the dataset is huge, estimate that gradient from a random minibatch — the estimate is unbiased, its noise shrinks like $1/b$, and that noise even helps you escape saddles.**

</div>

- **GD**: $\theta\leftarrow\theta-\eta\nabla J$, the steepest-descent step from a Taylor expansion.
- **The landscape**: convex bowls are easy; ravines ($\kappa\gg1$) and saddles are the enemy.
- **SGD**: minibatch gradients are cheap and unbiased ($\mathbb E[\hat g]=\nabla J$, $\operatorname{Var}\propto 1/b$).
- **The learning rate** is the one knob — and you can read it straight off the loss curve.

**Next (L6):** momentum & Adam — making SGD fast and less fiddly.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's own courses (IIT Gandhinagar) and standard references:

- **Taylor-series derivation, gradient $\perp$ level sets, GD & SGD worked examples, the unbiased-estimator proof, complexity comparison** — *Machine Learning (ES 335)*, optimization slides & notebooks, N. Batra · `github.com/nipunbatra/ml-teaching` (`optimization/slides/gradient-descent.tex`, `convexity.tex`; `notebooks/gradient-descent-2d.ipynb`, `gradient-descent-fit-contour.ipynb`)
- **Pedagogical framing (batch → mini-batch → stochastic, "compute argument")** — A. Ng, *Deep Learning Specialization*, Course 2 (Improving Deep Neural Networks), Week 2.
- **Loss-landscape, ravine, saddle & variance figures** — ES 667 figure library (`figures/lec04`).

All source courses © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · least squares is convex, so GD can't get stuck

For $f(\theta)=\lVert y-\mathbf X\theta\rVert^2$, differentiate twice:

$$\frac{df}{d\theta} = -2\mathbf X^\top y + 2\mathbf X^\top\mathbf X\theta \quad\Longrightarrow\quad \nabla^2 f = \mathbf H = 2\,\mathbf X^\top\mathbf X$$

For any real $\mathbf X$, $\mathbf X^\top\mathbf X$ is **positive semidefinite** ($v^\top\mathbf X^\top\mathbf X v = \lVert\mathbf X v\rVert^2\ge 0$), so $\mathbf H\succeq 0$ and $f$ is convex. That's the guarantee behind Part 3: on this bowl GD has *one* minimum to find. Add an L2 penalty $\lVert\theta\rVert^2$ (also convex) and the sum stays convex — the mathematics behind weight decay in L7.

*(Full derivation: ML ES 335 `convexity.tex`.)*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · why iterate when a formula exists?

The normal equation is exact, but its cost is set by the number of features $d$:

| | Normal equation | Gradient descent |
|---|---|---|
| formula | $\theta=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top y$ | $\theta_{t+1}=\theta_t-\eta\tfrac2n\mathbf X^\top(\mathbf X\theta_t-y)$ |
| time | $\mathcal O(d^2 n + d^3)$ | $\mathcal O(T\cdot nd)$ for $T$ iters |
| space | $\mathcal O(d^2)$ | $\mathcal O(nd)$ |
| solution | exact, one shot | approximate, iterative |

The $d^3$ matrix inversion is fatal when $d$ is large — and it only exists for *linear* least squares. For high-dimensional or nonlinear models (every neural net), gradient descent is not a convenience, it's the **only** option.

*(Step-by-step complexity: ML ES 335 `gradient-descent.tex`, "Computational Complexity".)*
