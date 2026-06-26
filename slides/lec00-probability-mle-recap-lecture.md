---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# A Probabilistic View of ML

## Distributions · Likelihood · MLE

### Lecture 0 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Why this lecture

You already know how to solve regression and classification. You wrote down a loss, took a gradient, ran SGD. It works.

But two questions you may never have asked ·

1. **Where does the loss come from?** Why MSE for regression and cross-entropy for classification — and not something else?
2. **What is L1 / L2 regularization, really?** They're not just "add this term, weights stay small." They have a meaning.

<div class="paper">

Today's promise · one principle — **maximum likelihood under a probabilistic model** — gives every loss in this course, and a tiny twist on it gives every regularizer. From this lecture onward, "loss design" stops being a bag of tricks and becomes a modeling choice.

</div>

---

# Learning outcomes

By the end of this lecture you can ·

<div class="math-box">

1. **State** the Bernoulli, Categorical, and Normal distributions and the `~` notation.
2. **Read a plate-notation** graphical model and recognize the supervised setup.
3. Explain three reasons why **the Normal shows up everywhere** (CLT, max entropy, closed-under-linear).
4. **Derive MSE / BCE / categorical CE** as NLL under Gaussian / Bernoulli / Categorical outputs.
5. State **the MLE recipe** in 3 steps and apply it to coin, linear, logistic, multiclass.

L00B (next lecture) takes this and adds priors → MAP → L1/L2 → KL.

</div>

---

# Bridge from ES 335 · you already know this

| From ES 335 | What changes today |
|---|---|
| You were *told* cross-entropy is the loss for logistic regression | today you **derive** it from a Bernoulli model — the recipe for *every* loss |
| MSE justified as "penalize big errors more" | MSE = NLL under Gaussian noise — a *modeling choice* |
| Each task came with its loss, handed down | losses are **derived** · pick the distribution, the loss falls out |
| The model outputs a number | the model outputs a **distribution** over $y$ |

<div class="keypoint">

In ES 335 cross-entropy was handed to you as "the classification loss" — *stated*, never derived. Today you derive it from a Bernoulli model, and that one derivation becomes the organizing principle of the whole course.

</div>

---

<!-- _class: section-divider -->

### PART 0

# Revision · linear & logistic regression

The loss functions we used without asking why

---

# Linear regression · the model

Given $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$ with targets $y_i \in \mathbb{R}$, fit a line and **read off a prediction** — no probability anywhere in sight yet.

$$\hat y_i = \boldsymbol\theta^\top \mathbf{x}_i \qquad\text{— a single number, a *point estimate*}$$

![w:660px](figures/lec00/svg/linreg_frequentist.svg)

---

# Linear regression · loss & training

<div class="math-box">

**Loss · mean squared error**
$$L(\boldsymbol\theta) = \frac{1}{N}\sum_{i=1}^N (y_i - \boldsymbol\theta^\top \mathbf{x}_i)^2$$

**Train · two equivalent options** ·
- Solve $\nabla_\theta L = 0$ in closed form · $\hat{\boldsymbol\theta} = (X^\top X)^{-1} X^\top \mathbf{y}$.
- Or run gradient descent · $\boldsymbol\theta \leftarrow \boldsymbol\theta - \eta \nabla_\theta L$.

</div>

**Open question** · you probably justified MSE as *"penalize big errors more than small ones."* True — but **why squared and not absolute or cubed?** We'll see today.

---

# Logistic regression · the model

Same setup as linear regression, but now $y_i \in \{0, 1\}$ (binary classification). The output should be a *probability* between 0 and 1.

<div class="math-box">

$$\hat p_i = \sigma(\boldsymbol\theta^\top \mathbf{x}_i),\qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

</div>

The **sigmoid** maps any real number to $(0, 1)$ — large positive logits give probabilities near 1; large negative logits give probabilities near 0; zero logit gives $0.5$.

The model output $\hat p_i$ is *interpreted* as $P(Y = 1 \mid \mathbf{x}_i)$. (We'll make that interpretation precise in Part 1.)

---

# Logistic regression · loss & training

<div class="math-box">

**Loss · binary cross-entropy (BCE)**
$$L(\boldsymbol\theta) = -\frac{1}{N}\sum_{i=1}^N \bigl[y_i \log\hat p_i + (1 - y_i)\log(1 - \hat p_i)\bigr]$$

**Train** · gradient descent (no closed form because of the sigmoid).

</div>

**Open question** · you justified BCE as *"big penalty when confidently wrong."* True again — but **why this exact form**, not $-(y - \hat p)^2$ or $-|y - \hat p|$? Same question as MSE: where do these specific losses come from?

---

# Regularization · the other half-mystery

When the model overfits, you added a penalty term:

<div class="math-box">

**L2 (ridge)** · $L(\boldsymbol\theta) + \lambda \|\boldsymbol\theta\|_2^2 = L(\boldsymbol\theta) + \lambda \sum_j \theta_j^2$

**L1 (lasso)** · $L(\boldsymbol\theta) + \lambda \|\boldsymbol\theta\|_1\, = L(\boldsymbol\theta) + \lambda \sum_j |\theta_j|$

</div>

You probably learned ·
- **L2** keeps weights small (smooth shrinkage).
- **L1** drives many weights to **exactly zero** (sparsity).

But **why** does L1 hit zero and L2 doesn't? Why is the penalty *squared* in L2 and *absolute* in L1? We'll derive both from first principles.

---

# Today's question · the four mysteries

<div class="math-box">

| Object | Recipe | Where does it come from? |
|:-:|:-:|:-:|
| MSE | $\sum (y - \hat y)^2$ | ? |
| BCE | $-\sum [y\log\hat p + (1-y)\log(1-\hat p)]$ | ? |
| L2 | $\lambda \sum_j \theta_j^2$ | ? |
| L1 | $\lambda \,\lVert \boldsymbol\theta \rVert_1$ | ? |

</div>

You've used all four — but none was ever *derived* from anything. They were handed to you with the magic words *"this is the loss for regression"*.

Today we replace the magic with a single principle.

---

# Today's promise · one principle, two answers

<div class="keypoint">

In L00 (today) we answer **two** of the mysteries ·

1. **The model defines a probability distribution** over $y$ given $x$ — not a single number.
2. **Pick parameters that make the data most likely** (MLE).

</div>

That's the whole lecture in two bullets. **MSE and BCE/CE** fall out of MLE as instances of one recipe.

The other two mysteries — **L1 / L2 regularization** — fall out of MAP (MLE + prior) in L00B.

---

# Bonus · the dividend in later lectures

The same machinery gives us **KL divergence** — the natural distance between distributions.

KL becomes the central object in ·

- **VAEs (L19)** · the ELBO is a KL between approximate and true posterior.
- **Diffusion (L21)** · score matching ≡ KL minimization between noisy data and model.
- **RLHF / DPO (L16)** · the reward objective is regularized by KL to a reference policy.
- **Distillation (L23)** · student matches teacher distribution by minimizing KL.

One framework today, ten lectures of dividends.

---

# ▶ Interactives for L00

<div class="paper">

Play along while you read · all run in the browser ·

- [mle-map-coin](https://nipunbatra.github.io/interactive-articles/mle-map-coin/) · slide α, β, true p, N — watch MLE drift toward MAP.
- [bayesian-posterior](https://nipunbatra.github.io/interactive-articles/bayesian-posterior/) · drag the prior, watch the posterior update.
- [info-theory](https://nipunbatra.github.io/interactive-articles/info-theory/) · entropy + KL with sliders.
- [multivariate-normal](https://nipunbatra.github.io/interactive-articles/multivariate-normal/) · 2D Gaussian shape vs covariance.
- [demystifying-p-values](https://nipunbatra.github.io/interactive-articles/demystifying-p-values/) · companion for the disease-test slide.

</div>

---

<!-- _class: section-divider -->

### PART 1

# A probabilistic view of ML

The model doesn't predict a number — it predicts a **distribution**

---

# Random variable · the definition from first-year probability

In your probability course, a **random variable** $X$ was a *function* that maps each outcome in the sample space $\Omega$ to a number.

![w:600px](figures/lec00/svg/random_variable_map.svg)

Flip two coins · $X =$ number of heads sends each outcome to $\{0, 1, 2\}$.

---

# Random variable · the view we'll use today

For ML we care less about $\Omega$ and more about the **distribution** of the values $Y$ takes ·

<div class="math-box">

$$Y \sim p$$

Read · *"$Y$ is **distributed as** $p$"* — the central notation of this lecture.

</div>

- **Discrete** $Y \in \{y_1, y_2, \ldots\}$ — probability **mass** $P(Y = y)$, sums to $1$.
- **Continuous** $Y \in \mathbb{R}$ — probability **density** $p(y)$, integrates to $1$.

---

# Distributions usually have parameters

Most distributions have **parameters** — knobs that shape the distribution. We collect them into a single symbol $\theta$.

<div class="math-box">

$$Y \sim p(\cdot \mid \theta)$$

Read · *"$Y$ is distributed as $p$, **given** parameters $\theta$."*

</div>

The vertical bar "$\mid$" means **"given"** — the same conditional notation as in $P(A \mid B)$ from your probability course.

---

# Parameters · three concrete examples

The parameter symbol $\theta$ is just a placeholder. For the three distributions we'll meet today ·

<div class="math-box">

- **Coin** · one parameter, the bias. Call it $p \in [0, 1]$ (so here $\theta = p$).
$Y \sim \text{Bernoulli}(p)$.

- **Normal** · two parameters · $\theta = (\mu, \sigma^2)$.
$Y \sim \mathcal{N}(\mu, \sigma^2)$.

- **Categorical** · $K$-vector of probabilities $\boldsymbol\pi$ (with $\sum_k \pi_k = 1$), so $\theta = \boldsymbol\pi$.
$Y \sim \text{Categorical}(\boldsymbol\pi)$.

</div>

**In ML, $\theta$ ends up being the model's weights** — the things we estimate from data via MLE / MAP later. For now, treat $\theta$ as known.

---

# The three distributions · one picture

![w:920px](figures/lec00/svg/three_distributions.svg)

Binary → Bernoulli · multi-class → Categorical · continuous → Normal. Each will hand us its loss in Part 3.

---

# IID · start with the picture

Every supervised dataset is a table · $N$ rows, each one an (input, target) example.

![w:720px](figures/lec00/svg/iid_matrix.svg)

Two questions about those rows lead to the **IID** assumption →

---

# IID · the assumption, formally

<div class="math-box">

$$Y_i \stackrel{\text{iid}}{\sim} p(\cdot \mid \theta)$$

- **Identically distributed** · every row comes from the *same* $p(\cdot\mid\theta)$.
- **Independent** · knowing $Y_i$ tells you *nothing* about $Y_j$ ($i \ne j$).

</div>

Examples first, then the algebra · together they give the **product factorization** ·
$$P(\mathcal{D} \mid \theta) = \prod_{i=1}^N P(Y_i \mid \theta)$$

---

# IID · why it matters

The product factorization becomes a **sum after taking logs** — and that sum is the **summed loss over a dataset** in every training loop.

<div class="keypoint">

IID is the formal license to add up per-example losses. Shuffle the batch order and the summed loss is unchanged — that is independence at work.

</div>

---

# IID · and when it breaks

![w:1000px](figures/lec00/svg/iid_fails.svg)

Time series, video frames, one-device sensors · here $y_t$ depends on $y_{t-1}$, so the product factorization is simply wrong. Then you need different math — autoregressive or state-space models. For this course, treat *shuffled* batches as IID.

<div class="notebook">

▶ **Notebook · [`lec00-iid-demo.ipynb`](../notebooks/lec00-iid-demo.ipynb)** — four cases (IID / indep-not-identical / identical-not-indep / neither) with sequence, histogram and lag-1 scatter plots; shows `sum(log p)` is *blind* to autocorrelation.

</div>

---

# Bernoulli · the coin (the simplest distribution)

A **Bernoulli** models a single yes/no outcome $Y \in \{0, 1\}$. One parameter $p \in [0,1]$ = the probability of a "1" (call it "heads").

<div class="math-box">

$$Y \sim \text{Bernoulli}(p), \qquad P(Y=1\mid p) = p,\quad P(Y=0\mid p) = 1-p$$

</div>

Anything binary is a Bernoulli ·

- **spam** · an email is spam ($Y=1$) or not ($Y=0$)
- **diagnosis** · a patient has the disease or does not
- **segmentation** · a pixel is foreground or background

Two outcomes, two probabilities summing to 1 — the simplest non-trivial distribution, and the seed of binary cross-entropy.

---

# Bernoulli · why we need a *compact* form

The two-case PMF works, but the two cases are awkward to differentiate ·

$$P(Y = y \mid p) = \begin{cases} p & y = 1 \\ 1 - p & y = 0 \end{cases}$$

Problems · the formula **branches** on $y$, you can't write $\nabla_p \log P$ with one expression, and stacking $N$ flips gives an *if-else tree* that's hopeless for code or maths.

We want **one expression** that ·
1. evaluates to $p$ when $y = 1$ and to $1 - p$ when $y = 0$,
2. is differentiable in $p$,
3. multiplies cleanly across $N$ samples.

---

# Bernoulli · the compact form we'll reuse

The form that satisfies all three ·

<div class="math-box">

$$P(Y = y \mid p) = p^y\,(1 - p)^{1 - y}$$

</div>

**Sanity check** ·
- $y = 1 \;\Rightarrow\; p^1 (1-p)^0 = p$ ✓
- $y = 0 \;\Rightarrow\; p^0 (1-p)^1 = 1 - p$ ✓

The trick is using $y$ and $1 - y$ **as exponents** · only one of them is non-zero at a time, the other factor collapses to 1.

---

# Why not just $p\cdot y + (1-p)(1-y)$?

A *linear* interpolation between $p$ and $1-p$ also gives the right answer at $y \in \{0, 1\}$ ·

$$f(y) = p y + (1-p)(1-y)$$

Evaluate · $f(1) = p$ ✓, $f(0) = 1-p$ ✓.

But this form **fails the third requirement** · the product over $N$ samples doesn't collapse to a clean expression in $p$ and the counts of heads / tails. The exponential form $p^y (1-p)^{1-y}$ does ·

$$\prod_{i=1}^N p^{y_i}(1-p)^{1 - y_i} = p^{\sum y_i}\,(1-p)^{N - \sum y_i} = p^{\#H}\,(1-p)^{\#T}$$

**Only counts matter** — a clean *sufficient statistic*. This collapse is what makes the MLE derivation in Part 3 a one-liner.

---

# The compact form is the seed of BCE

Take logs of the compact form ·

<div class="math-box">

$$\log P(Y = y \mid p) = y \log p + (1 - y) \log(1 - p)$$

</div>

Negate and average over a dataset and you get **binary cross-entropy** — *every* binary-classifier loss in this course.

$$\text{BCE}(p, y) = -\bigl[y \log p + (1 - y) \log(1 - p)\bigr]$$

The compact form turned a branching PMF into a single, differentiable expression that becomes the standard BCE loss. We'll use it every time.

---

# Bernoulli · moments

<div class="math-box">

**Mean** · $\mathbb{E}[Y] = 0 \cdot (1-p) + 1 \cdot p = p$

**Variance** · $\text{Var}[Y] = \mathbb{E}[Y^2] - \mathbb{E}[Y]^2 = p - p^2 = p(1 - p)$

</div>

Variance is largest at $p = 0.5$ (most uncertain) and zero at $p \in \{0, 1\}$ (deterministic).

This will be reused when we derive **logistic regression's gradient** — it has a $\hat p (1 - \hat p)$ term that is exactly the Bernoulli variance at the predicted probability.

---

# Bernoulli · IID worked example

**Setup** · coin with $p = 0.7$. Three flips give $H, T, H$ (i.e. $Y_1, Y_2, Y_3 = 1, 0, 1$).

<div class="math-box">

Under the IID assumption ·

$$P(\mathcal{D} \mid p) = P(Y_1 = 1) \cdot P(Y_2 = 0) \cdot P(Y_3 = 1)$$

$$= 0.7 \cdot 0.3 \cdot 0.7 = \mathbf{0.147}$$

</div>

The **product over independent observations** is the heart of likelihood — coming up in Part 2 when we ask *"which $p$ makes the observed data most likely?"*

---

# Plate notation · graphical-model conventions

We now have one concrete distribution (Bernoulli). A clean way to *draw* a probabilistic model is **plate notation** — the standard for the rest of this course.

<div class="math-box">

| Symbol | Meaning |
|:-:|:-:|
| ○ | a random variable (uncertain) |
| ● (filled) | an **observed** random variable (we see its value) |
| arrow $A \to B$ | $A$ generates $B$ (i.e. $B$ depends on $A$) |
| rectangle (plate) labelled $i = 1\ldots N$ | the contents are repeated $N$ times — independence across $i$ |

</div>

These four symbols compose every probabilistic model in this course. **Bayesian networks**, **HMMs**, **VAEs**, **diffusion models** — all drawn with these conventions.

---

# Plate notation · IID Bernoulli example

Apply the conventions to the simplest model · $N$ IID Bernoulli observations.

A single Bernoulli observation · $p \to \bullet\,Y$.

For $N$ observations, draw a plate around the repeated part ·

![w:380px](figures/lec00/svg/plate_iid_bernoulli.svg)

The plate says *"draw a fresh $Y_i$ for each $i$, all from the same Bernoulli($p$)."* The single $p$ outside the plate is **shared across all observations** — that is what makes the dataset *identically distributed*.

---

# Bernoulli · what we'd compute in code

We now know the three things we *do* with any distribution. For Bernoulli($p = 0.6$) ·

<div class="math-box">

| Operation | Math | PyTorch |
|:-:|:-:|:-:|
| Sample | $Y \sim \text{Bernoulli}(p)$ | `dist.Bernoulli(0.6).sample((10,))` |
| Score | $\log P(Y \mid p)$ | `d.log_prob(y)` |
| Moments | $\mathbb{E}, \text{Var}$ | `d.mean, d.variance` |

</div>

Sample · score · know-the-moments. **These three operations are the entire interface** to any distribution we use in this course — Bernoulli, Categorical, Normal, Beta, Laplace, … all the same three.

<div class="notebook">

▶ **Notebook · [`lec00-bernoulli-playground.ipynb`](../notebooks/lec00-bernoulli-playground.ipynb)** — sample, score, IID factorisation, BCE = NLL of Bernoulli (all three operations from this slide, hands-on).

</div>

---

# Categorical · the K-sided die

A **Categorical** models one outcome out of $K$ classes — think a (possibly loaded) $K$-sided die. Parameter $\boldsymbol\pi = (\pi_1, \ldots, \pi_K)$ with $\pi_k \ge 0$, $\sum_k \pi_k = 1$.

<div class="math-box">

$$Y \sim \text{Categorical}(\boldsymbol\pi), \qquad P(Y = k \mid \boldsymbol\pi) = \pi_k$$

</div>

**Bernoulli is the special case $K = 2$.**

<div class="math-box">

**What the "mean" means** · encode the outcome one-hot as $\mathbf{y}\in\{0,1\}^K$. Average many rolls and each coordinate converges to that class's frequency · $\mathbb{E}[\mathbf{y}] = \boldsymbol\pi$.

**Example** · a loaded 3-sided die with $\boldsymbol\pi = (0.2, 0.5, 0.3)$ — over many rolls the one-hot average $\to (0.2, 0.5, 0.3)$, literally the class probabilities.

</div>

---

# Categorical · the one-hot compact form

Encode the label as a **one-hot** vector $\mathbf{y} \in \{0,1\}^K$ — a single $1$ in the true class, $0$ everywhere else.

<div class="math-box">

$$P(Y \mid \boldsymbol\pi) = \prod_{k=1}^K \pi_k^{\,y_k}$$

</div>

**Why this works** · say $K = 4$ and the true class is $Y = 3$, so $\mathbf{y} = (0,0,1,0)$ ·

$$\pi_1^{\,0}\,\pi_2^{\,0}\,\pi_3^{\,1}\,\pi_4^{\,0} = \pi_3$$

Every factor with exponent $0$ collapses to $1$ — only the true class survives. Same exponent trick as Bernoulli, and the seed of categorical cross-entropy.

---

# Categorical · a worked example (MNIST)

An MNIST classifier outputs a 10-way softmax for one image. Here the true digit is **2** ·

![w:780px](figures/lec00/svg/mnist_categorical.svg)

The model put $\hat\pi_2 = 0.70$ on the truth (a perfect model would put all mass on class 2). **The softmax output of *any* classifier IS a Categorical** — treat it that way and cross-entropy falls out automatically.

---

# Normal (Gaussian) · the bell curve

Continuous $Y \in \mathbb{R}$ with mean $\mu$, variance $\sigma^2$ · $Y \sim \mathcal{N}(\mu, \sigma^2)$.

![w:600px](figures/lec00/svg/normal_bellcurve.svg)

$$p(y \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\,\exp\!\left(-\frac{(y - \mu)^2}{2\sigma^2}\right)$$

The most important continuous distribution — and the seed of the MSE loss.

---

# Normal · three properties to memorize

<div class="math-box">

1. **Centred at $\mu$**, spread controlled by $\sigma$.
2. Density falls off **exponentially in the squared distance** $(y - \mu)^2$.
3. The squared exponent will be the **seed of MSE** in Part 3.

</div>

Property 2 is what we'll lean on most. Let's unpack it.

---

# Normal · how fast the bell decays

Property 2 says density falls off **exponentially** in $(y - \mu)^2$. How fast?

<div class="math-box">

| Distance from mean | Exponent | Density factor |
|:-:|:-:|:-:|
| $1\sigma$ | $1/2$ | $e^{-0.5} \approx 0.61$ |
| $2\sigma$ | $2$ | $e^{-2} \approx 0.14$ |
| $4\sigma$ | $8$ | $e^{-8} \approx 3 \times 10^{-4}$ |

</div>

This **squared, exponential decay** is what makes Gaussians "tightly concentrated" — almost all the mass sits within a few $\sigma$ of the mean.

**Empirical rule** · 68% within $1\sigma$, 95% within $2\sigma$, 99.7% within $3\sigma$.

---

# Normal · a worked numeric example

**House prices** modelled as $\mathcal{N}(50, 5^2)$ lakh.

<div class="math-box">

| Sample value $y$ | Density $p(y)$ | Distance from mean |
|:-:|:-:|:-:|
| $50$ (the mean) | $1 / \sqrt{2\pi \cdot 25} \approx 0.0798$ | $0\sigma$ |
| $55$ | $\approx 0.0484$ | $1\sigma$ |
| $60$ | $\approx 0.0108$ | $2\sigma$ |
| $70$ | $\approx 2.7 \times 10^{-5}$ | $4\sigma$ |

</div>

A house priced at the mean ($Y = 50$) is the most likely. A house priced at $Y = 70$ — four standard deviations away — is vanishingly unlikely under this model.

This **squared-distance penalty** $(y - \mu)^2$ is the exact form that becomes MSE when we maximize the likelihood over data — covered in Part 3.

---

# Why does the Normal show up everywhere?

Three deep reasons — each comes back later in the course.

<div class="math-box">

1. **Central Limit Theorem (CLT)** · the sum of many small independent things is Normal. So measurement noise, sensor jitter, biological variation all *empirically* look Gaussian.
2. **Maximum entropy** · among all distributions with given mean and variance, Normal has the largest entropy → "the most agnostic choice when all you know is mean & variance."
3. **Closed under linear operations** · sum, scaling, and conditioning on linear maps of Gaussians stay Gaussian. Bayesian updates with Gaussian prior + Gaussian likelihood give a Gaussian posterior — *conjugacy*.

</div>

These three properties together explain why the Gaussian dominates classical statistics, signal processing, **diffusion models**, Kalman filters, and Bayesian neural nets.

---

# The conditional view · the model outputs a distribution

In supervised learning the model outputs the **parameters of a distribution** over $Y$ given $\mathbf{x}$ — never just a number.

![w:1000px](figures/lec00/svg/conditional_view.svg)

Training asks · *under these distributions, how likely are the labels we actually saw?* Maximize that — everything else follows.

---

# The supervised graphical model · one picture

Every supervised learning setup we'll see in this course shares the same plate diagram ·

![w:560px](figures/lec00/svg/plate_supervised.svg)

<div class="math-box">

- $\theta$ — model parameters (we will estimate by MLE / MAP).
- $\mathbf{x}_i$ — input, observed (filled).
- $Y_i$ — output, observed during training (filled), unknown at test time.
- The plate says · "$N$ IID samples, all sharing the same $\theta$."

</div>

Whether you are doing logistic regression, an MLP, a Transformer, or a diffusion model — the *outermost* graphical model is **always this**. Only the conditional distribution $p(y \mid \mathbf{x}, \theta)$ inside the plate changes.

---

# Bayes' rule in ML · why we start with the likelihood

We really want the parameters **given** the data — the *posterior* $p(\theta \mid \mathcal{D})$. Bayes' rule splits it into two pieces ·

<div class="math-box">

$$\underbrace{p(\theta \mid \mathcal{D})}_{\text{posterior}} \;\propto\; \underbrace{p(\mathcal{D} \mid \theta)}_{\text{likelihood}}\;\cdot\;\underbrace{p(\theta)}_{\text{prior}}$$

</div>

- **Likelihood** $p(\mathcal{D}\mid\theta)$ — how well $\theta$ explains the data. Maximize it alone → **MLE**. *That is the whole of today.*
- **Prior** $p(\theta)$ — what we believed before any data. Multiply it back in → **MAP**, and every regularizer — *next lecture (L00B)*.

<div class="keypoint">

We spend L00 entirely on the **likelihood** term. The prior is a one-line addition in L00B — and L2 / L1 fall straight out of it.

</div>

---

<!-- _class: section-divider -->

### PART 2

# Likelihood

The probability of the data, viewed as a function of $\theta$

---

# Coin toss · setup

You're handed a coin with **unknown** bias $p$. You flip it $N = 10$ times and observe ·

$$\mathcal{D} = \texttt{H, H, T, H, T, T, H, H, T, H}$$

Six heads, four tails.

<div class="keypoint">

Question · what value of $p$ is *most consistent* with this data?

</div>

Intuition says $p \approx 0.6$. We'll make that intuition rigorous and, in doing so, write down the entire framework for everything that follows.

---

# Likelihood · definition

The **likelihood** of a parameter $\theta$ given data $\mathcal{D}$ is the probability of observing $\mathcal{D}$ under that parameter ·

<div class="math-box">

$$\mathcal{L}(\theta) := P(\mathcal{D} \mid \theta)$$

</div>

It is **not** a probability over $\theta$ — that would be the *posterior*, from the Bayes'-rule slide we just saw. It is the data's probability, viewed as a function of $\theta$.

For the coin, IID assumption gives ·
$$\mathcal{L}(p) = \prod_{i=1}^{N} P(y_i \mid p) = \prod_{i=1}^{N} p^{y_i}(1 - p)^{1 - y_i} = p^{\#H}\,(1 - p)^{\#T}$$

For our data · $\mathcal{L}(p) = p^6 (1 - p)^4$.

---

# Likelihood · plotted

![w:780px](figures/lec00/svg/coin_likelihood.svg)

<div class="math-box">

$\mathcal{L}(p) = p^6 (1-p)^4$, plotted on $p \in [0, 1]$.

Maximum at $p = 0.6$ — which matches our intuition (6 heads out of 10).

</div>

---

# Two problems with the raw likelihood

The raw likelihood $\mathcal{L}(\theta) = \prod_{i=1}^N P(y_i \mid \theta)$ is a *product*. Two practical issues follow ·

<div class="warning">

**Problem 1 · numerical underflow.** With $N = 1000$ and each factor $\sim 0.5$ ·

$$\mathcal{L} \approx 0.5^{1000} \approx 10^{-301}$$

This is *below* the smallest representable double-precision float ($\approx 10^{-308}$). On a computer, $\mathcal{L}$ becomes literally zero.

**Problem 2 · hard to differentiate.** The product rule on $N$ factors produces $N$ terms — algebraically and computationally messy.

</div>

We need the same answer in a form that doesn't underflow and is easy to differentiate.

<div class="notebook">

▶ **Notebook · [`lec00-mle-map.ipynb`](../notebooks/lec00-mle-map.ipynb)** — watch $0.5^{1000}$ underflow to exactly `0.0` in float64 while the log-likelihood stays finite.

</div>

---

# The fix · take the log

The logarithm turns the product into a **sum** ·

<div class="math-box">

$$\ell(\theta) := \log \mathcal{L}(\theta) = \log \prod_{i=1}^N P(y_i \mid \theta) = \sum_{i=1}^N \log P(y_i \mid \theta)$$

</div>

The dataset's "score" is now a **sum of $N$ moderate numbers** — numerically stable and easy to differentiate term-by-term.

---

# Why taking the log is safe · monotonicity

$\log$ is **monotonic increasing** · if $a > b$ then $\log a > \log b$. So whatever $\theta$ maximizes $\mathcal{L}$ also maximizes $\log\mathcal{L}$ — **the location of the peak doesn't move**, only the $y$-axis rescales.

![w:880px](figures/lec00/svg/log_monotonic.svg)

Both curves peak at the same $p = 0.6$. We optimize the friendlier (log) curve and get the identical answer.

---

# NLL · the convention we'll always use

We minimize the **negative** log-likelihood (NLL) so "loss" is something we drive *down* with gradient descent ·

<div class="math-box">

$$\hat\theta_{\text{MLE}} \;=\; \arg\min_\theta \,\bigl[-\ell(\theta)\bigr] \;=\; \arg\min_\theta \,\sum_i \bigl[-\log P(y_i \mid \theta)\bigr]$$

</div>

We will *always* work with log-likelihood from this point onward.

<div class="keypoint">

**Every loss in this course is an NLL.** MSE, BCE, cross-entropy, ELBO, diffusion loss — all are just NLLs of carefully chosen distributions.

</div>

---

<!-- _class: section-divider -->

### PART 3

# Maximum Likelihood Estimation

Concrete derivations · coin · linear regression · logistic regression

---

# Pop quiz · which course did this student come from?

Three sections of the same exam · grades modelled as Normal ·

<div class="popquiz">

| Course | Mean $\mu$ | Std $\sigma$ |
|---|---|---|
| C1 | 80 | 10 |
| C2 | 70 | 10 |
| C3 | 90 | 5 |

A student's marks $y = 82$. Which course is this student most likely from?

*Stop and guess before the next slide.*

</div>

---

# Answer · pick the distribution that best explains the data

Evaluate the density $\mathcal{N}(82 \mid \mu, \sigma^2)$ for each course ·

$$p_1(82) \approx 0.0391,\quad p_2(82) \approx 0.0194,\quad p_3(82) \approx 0.0222$$

Course **C1** wins · its bell is centred close to 82 with reasonable spread.

<div class="math-box">

**MLE intuition** · among candidate distributions, choose the one under which the **observed data has the highest probability**.

</div>

This is the entire idea. Everything below is just doing it carefully.

---

# MLE for the coin · setup

Data · $\#H = h = 6$ heads out of $N = 10$ flips. Likelihood ·
$\mathcal{L}(p) = p^h (1 - p)^{N - h}$

Take the log ·
<div class="math-box">

$$\ell(p) = h \log p + (N - h) \log(1 - p)$$

</div>

**Step 1 · differentiate.**
$$\frac{d\ell}{dp} = \frac{h}{p} - \frac{N - h}{1 - p}$$

---

# MLE for the coin · solve

**Step 2 · set derivative to zero.**
$$\frac{h}{p} - \frac{N - h}{1 - p} = 0 \;\Longrightarrow\; h(1 - p) = (N - h)\,p$$

Expand · $h - hp = Np - hp \Longrightarrow h = Np$.

<div class="math-box">

$$\boxed{\;\hat p_{\text{MLE}} = \frac{h}{N} = \frac{\#H}{\#H + \#T}\;}$$

</div>

For our data $h = 6, N = 10$ · $\hat p = 0.6$. **Exactly** the empirical frequency.

**Step 3 · confirm it's a maximum (second-derivative test).**
$$\frac{d^2\ell}{dp^2} = -\frac{h}{p^2} - \frac{N-h}{(1-p)^2} < 0 \quad\text{for all } p$$

$\ell$ is **concave**, so the critical point is a *maximum* — not a minimum or saddle. Your first MLE derivation, done properly.

---

# The MLE recipe · 3 steps

To find the MLE for any model ·

<div class="math-box">

1. **Pick a probabilistic model** — choose a distribution $p(y \mid \theta)$ that matches your output type (Bernoulli for binary, Normal for continuous, …).
2. **Write the log-likelihood** of the dataset under that model · sum the per-example $\log p(y_i \mid \theta)$.
3. **Maximize over $\theta$** — by setting derivative to zero analytically, or by gradient ascent (equivalently, gradient descent on the negative log-likelihood).

</div>

We will now apply this recipe to **linear regression** (where the answer pops out as MSE) and **logistic regression** (where it pops out as BCE).

---

# MLE for linear regression · the assumption

Modelling choice · the target is a linear function plus Gaussian noise ·

$$y = \boldsymbol\theta^\top \mathbf{x} + \epsilon,\qquad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

Equivalently · the conditional distribution of $Y$ given $\mathbf{x}$ is

<div class="math-box">

$$p(y \mid \mathbf{x}, \boldsymbol\theta) = \mathcal{N}(y \mid \boldsymbol\theta^\top \mathbf{x}, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(y - \boldsymbol\theta^\top \mathbf{x})^2}{2\sigma^2}\right)$$

</div>

The model's prediction $\hat\mu = \boldsymbol\theta^\top \mathbf{x}$ is the **mean** of the Gaussian. Real $y$ scatters around it with variance $\sigma^2$.

---

# Linear regression as MLE · the picture

![w:680px](figures/lec00/svg/mle_linear_regression.svg)

Each point is a **draw** from a Gaussian centred on the line. MLE asks · *which line makes the observed $y$'s most probable?* — equivalently, *which line minimizes squared distance to the points.* That equivalence is the whole derivation; everything else is algebra.

---

# MLE for linear regression · log-likelihood

For a single example,
$\log p(y_i \mid \mathbf{x}_i, \boldsymbol\theta) = -\tfrac{1}{2}\log(2\pi\sigma^2) - \dfrac{(y_i - \boldsymbol\theta^\top \mathbf{x}_i)^2}{2\sigma^2}$

Sum over the dataset · log of a product of IID terms is a sum.

<div class="math-box">

$$\ell(\boldsymbol\theta) = \sum_{i=1}^N \log p(y_i \mid \mathbf{x}_i, \boldsymbol\theta) = \underbrace{-\frac{N}{2}\log(2\pi\sigma^2)}_{\text{constant in }\theta} \;-\; \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - \boldsymbol\theta^\top \mathbf{x}_i)^2$$

</div>

Only the second term depends on $\boldsymbol\theta$.

---

# MLE for linear regression · MSE pops out

Maximizing $\ell$ over $\boldsymbol\theta$ ·

$$\hat{\boldsymbol\theta}_{\text{MLE}} = \arg\max_\theta \left[-\frac{1}{2\sigma^2}\sum_i (y_i - \boldsymbol\theta^\top \mathbf{x}_i)^2\right]$$

The factor $-1/(2\sigma^2)$ is a negative constant — flip the sign and minimize ·

<div class="math-box">

$$\boxed{\;\hat{\boldsymbol\theta}_{\text{MLE}} = \arg\min_\theta \sum_{i=1}^N (y_i - \boldsymbol\theta^\top \mathbf{x}_i)^2\;}$$

</div>

**This is exactly MSE.** MSE is not a heuristic — it is the **MLE under Gaussian noise**.

**Loss design = noise model.** Change the noise distribution and a *different* loss falls out — let's do Laplace noise next.

---

# Worked · Laplace noise → MAE

Same model $y = \boldsymbol\theta^\top\mathbf{x} + \epsilon$, but now $\epsilon \sim \text{Laplace}(0, b)$ ·

<div class="math-box">

$$p(y \mid \mathbf{x}, \boldsymbol\theta) = \frac{1}{2b}\exp\!\left(-\frac{|y - \boldsymbol\theta^\top\mathbf{x}|}{b}\right)$$

Log, sum, negate — the $1/b$ and $\log 2b$ are constants in $\boldsymbol\theta$ ·
$$L_{\text{NLL}}(\boldsymbol\theta) = \frac{1}{b}\sum_{i=1}^N |y_i - \boldsymbol\theta^\top\mathbf{x}_i| + \text{const}$$

$$\boxed{\;\hat{\boldsymbol\theta}_{\text{MLE}} = \arg\min_\theta \sum_{i=1}^N |y_i - \boldsymbol\theta^\top\mathbf{x}_i|\;}$$

</div>

**Exactly MAE.** Gaussian noise → squared error; Laplace noise → absolute error. Laplace's heavier tails make MAE **robust to outliers** — same recipe, different distribution.

---

# MLE for logistic regression · the assumption

Now $y \in \{0, 1\}$. Model each label as a **Bernoulli** whose parameter is the sigmoid of a linear score ·

$$Y \mid \mathbf{x} \sim \text{Bernoulli}\bigl(\hat p_\theta(\mathbf{x})\bigr),\qquad \hat p_\theta(\mathbf{x}) = \sigma(\boldsymbol\theta^\top \mathbf{x})$$

![w:600px](figures/lec00/svg/logistic_mle_picture.svg)

Same compact Bernoulli form as the coin — $p(y\mid\mathbf{x},\boldsymbol\theta) = \hat p^{\,y}(1-\hat p)^{1-y}$ — except $\hat p$ now depends on $\mathbf{x}$.

---

# MLE for logistic regression · log-likelihood

Take the log ·
$\log p(y_i \mid \mathbf{x}_i, \boldsymbol\theta) = y_i \log \hat p_i + (1 - y_i) \log(1 - \hat p_i)$
where $\hat p_i = \sigma(\boldsymbol\theta^\top \mathbf{x}_i)$.

Sum over the dataset and **negate** to get NLL ·

<div class="math-box">

$$L_{\text{NLL}}(\boldsymbol\theta) = -\sum_{i=1}^N \bigl[y_i \log \hat p_i + (1 - y_i) \log(1 - \hat p_i)\bigr]$$

</div>

**This is exactly binary cross-entropy.** Same story · BCE is not invented — it's the **MLE under a Bernoulli output**.

---

# MLE for logistic regression · gradient

Differentiate $L_{\text{NLL}}$ w.r.t. $\boldsymbol\theta$. Using $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ and the chain rule (derivation in any ML textbook) ·

<div class="math-box">

$$\nabla_\theta L_{\text{NLL}} = \sum_{i=1}^N (\hat p_i - y_i)\,\mathbf{x}_i$$

</div>

The gradient is "**prediction minus truth**, weighted by input." This is the exact same form as the linear regression gradient $(\hat y_i - y_i)\,\mathbf{x}_i$.

That is **not** a coincidence — it's a feature of *generalized linear models*, all derived from MLE.

---

# Multiclass · the softmax

Now $y \in \{1, \ldots, K\}$. Learn one weight vector $\boldsymbol\theta_k$ per class; the **logit** for class $k$ is $\textcolor{#37535F}{z_k} = \boldsymbol\theta_k^\top \mathbf{x}$.

<div class="math-box">

**Softmax** turns logits into probabilities ·
$$\textcolor{#5F8573}{\hat\pi_k(\mathbf{x})} = \frac{\exp(\textcolor{#37535F}{z_k})}{\sum_{j=1}^K \exp(\textcolor{#37535F}{z_j})}$$

</div>

Reading guide · <span style="color:#37535F">**slate = logits** $z_k$</span> (any real number) → <span style="color:#5F8573">**green = probabilities** $\hat\pi_k$</span> (all positive, sum to 1).

---

# Multiclass · the modelling assumption

<div class="math-box">

$$Y \mid \mathbf{x} \sim \text{Categorical}\bigl(\hat{\boldsymbol\pi}_\theta(\mathbf{x})\bigr)$$

</div>

Same recipe as before · the model outputs the parameters $\hat{\boldsymbol\pi}$ of a Categorical, and we maximize the probability it assigns to the true labels.

**Binary logistic is the special case $K = 2$** — the softmax collapses to a sigmoid.

---

# Multiclass · the per-example log-likelihood

The probability the model assigns to the **true** class $\textcolor{#B85A3E}{y_i}$ is just the softmax entry at that position ·

<div class="math-box">

$$p(\textcolor{#B85A3E}{y_i} \mid \mathbf{x}_i, \theta) = \textcolor{#5F8573}{\hat\pi_{i,\,y_i}} \;=\; \prod_{k=1}^K \textcolor{#5F8573}{\hat\pi_{i, k}}^{\,\textcolor{#B85A3E}{y_{i, k}}}$$

Take logs — only the true-class term survives the one-hot ·
$$\log p(\textcolor{#B85A3E}{y_i} \mid \mathbf{x}_i, \theta) = \sum_{k=1}^K \textcolor{#B85A3E}{y_{i, k}}\,\log \textcolor{#5F8573}{\hat\pi_{i, k}} = \log \textcolor{#5F8573}{\hat\pi_{i,\, y_i}}$$

</div>

<span style="color:#B85A3E">**rust = the truth** (one-hot $y_i$)</span> · <span style="color:#5F8573">**green = the model's probabilities** $\hat\pi$</span>. The $0$ exponents kill every term except the true class.

---

# Multiclass · NLL = categorical cross-entropy

Sum over the dataset and negate ·

<div class="math-box">

$$L_{\text{NLL}}(\theta) = -\sum_{i=1}^N \log \hat\pi_{i,\,y_i} \;=\; -\sum_{i=1}^N \sum_{k=1}^K y_{i,k}\,\log \hat\pi_{i,k}$$

</div>

The right-hand form is the textbook **categorical cross-entropy** — true distribution $\mathbf{y}_i$ (one-hot) against predicted distribution $\hat{\boldsymbol\pi}_i$.

This loss powers every multiclass classifier in this course · CIFAR-10 (K = 10), ImageNet (K = 1000), and the **next-token loss in every LLM** (K = vocab size, often 50,000+) in L13–L15.

---

# Multiclass · the elegant gradient

Differentiate $L_{\text{NLL}}$ w.r.t. the logit $z_{i, k}$ (derivation uses the softmax-Jacobian trick) ·

<div class="math-box">

$$\frac{\partial L_{\text{NLL}}}{\partial z_{i, k}} = \hat\pi_{i, k} - y_{i, k}$$

</div>

**Prediction minus truth, per logit.** Same elegant form as binary logistic ($\hat p - y$) and linear regression ($\hat y - y$). All three are instances of the same generalized-linear-model identity.

This is why **softmax + cross-entropy** are *always* implemented as one fused op (`F.cross_entropy(logits, target)`) — both for numerical stability (log-sum-exp trick) and because the gradient simplifies dramatically when you do them together.

---

# Multiclass · worked numeric

3 classes (cat, dog, car). Image with true class **cat** (index 0). Model produces logits $\mathbf{z} = [2.0,\, 1.0,\, 0.1]$.

<div class="math-box">

**Softmax** (subtract max for stability) · $\mathbf{z} - 2.0 = [0,\, -1,\, -1.9]$.
$\exp = [1.000,\, 0.368,\, 0.150]$ · sum $= 1.518$.
$\hat{\boldsymbol\pi} = [\,0.659,\, 0.242,\, 0.099\,]$.

**Per-example loss** · NLL of the true class.
$L = -\log \hat\pi_{\text{cat}} = -\log 0.659 = \mathbf{0.417}$

**Gradient on logits** · $\hat{\boldsymbol\pi} - \mathbf{y} = [0.659, 0.242, 0.099] - [1, 0, 0] = [-0.341,\, 0.242,\, 0.099]$.

</div>

The gradient says · *push the cat-logit up* (negative gradient → the optimizer subtracts it, so cat-logit increases), *push the dog and car logits down*. Exactly what you want.

---

# Summary so far · the pattern

<div class="math-box">

| Output | Distribution chosen | NLL turns out to be |
|:-:|:-:|:-:|
| Continuous $y \in \mathbb{R}$ | Normal | **MSE** |
| Binary $y \in \{0,1\}$ | Bernoulli | **BCE** |
| $K$-class $y \in \{1,\ldots,K\}$ | Categorical | **CE** |

</div>

<div class="keypoint">

Every loss = **NLL under an assumed conditional distribution** $p(y \mid \mathbf{x}, \theta)$.

Pick the distribution to match your data. The loss falls out automatically. No more "memorize MSE for regression and CE for classification" — both come from the same place.

</div>

---

# A worked numeric · BCE on one prediction

Cat-vs-dog classifier · model output $\hat p = 0.8$ for one image.

| true class $y$ | log-likelihood | NLL = loss | model "happy"? |
|:-:|:-:|:-:|:-:|
| 1 (cat) | $\log 0.8 = -0.223$ | $0.223$ | yes — small loss |
| 0 (dog) | $\log 0.2 = -1.609$ | $1.609$ | no — big loss |

<div class="keypoint">

The loss is small **iff the model assigned high probability to the true class**. That's all cross-entropy is doing — and that's all "maximizing log-likelihood" means once you write it out.

</div>

---

<!-- _class: section-divider -->

# Putting L00 together · the master sentence

<div class="math-box">

1. A model defines a **conditional distribution** $p(y \mid x; \boldsymbol\theta)$.
2. **Likelihood of a dataset** = $\prod_i p(y_i \mid x_i; \boldsymbol\theta)$ · take $\log$ + sum.
3. **MLE** = $\arg\max_\theta \sum_i \log p(y_i \mid x_i; \boldsymbol\theta) = \arg\min_\theta \text{NLL}$.
4. Plug in the right distribution → the right loss falls out ·

   - Normal → **MSE**
   - Bernoulli → **BCE**
   - Categorical → **Cross-entropy**

</div>

**Every loss you have used in any DL training run is one of these four lines.** Next lecture (L00B) adds a *prior* on $\boldsymbol\theta$ → MAP → L1/L2 → KL.

---

# L00 → L00B · what's coming next

| L00 (today) | L00B (next lecture) |
|:-:|:-:|
| MLE = best-fit-to-data | MAP = MLE + prior |
| MSE, BCE, CE losses | L2 + L1 regularizers |
| NLL of one model | KL between two distributions |
| Where the loss came from | The unifying language for VAE · diffusion · RLHF · distillation |

Today gives you **half** the answer to the four mysteries. L00B gives the other half — and the bridge into the rest of the course.

---

# In-class problem · single-feature linear regression

Model $Y \mid x \sim \mathcal{N}(\theta x, \sigma^2)$ with **known** $\sigma^2$ (one feature, no bias).

<div class="popquiz">

Derive the MLE for $\theta$ from scratch · write the log-likelihood, differentiate, set it to zero.

*Try it now — 3 minutes. Solution on the next slide.*

</div>

---

# Solution · single-feature linear regression MLE

<div class="math-box">

Per-example · $\log p(y_i \mid x_i, \theta) = \text{const} - \dfrac{(y_i - \theta x_i)^2}{2\sigma^2}$.

Sum and drop constants → minimize $\displaystyle\sum_i (y_i - \theta x_i)^2$.

Differentiate · $\dfrac{d}{d\theta}\displaystyle\sum_i (y_i - \theta x_i)^2 = -2\sum_i x_i(y_i - \theta x_i) = 0$.

Solve · $\sum_i x_i y_i = \theta \sum_i x_i^2 \;\Longrightarrow\; \boxed{\;\hat\theta = \dfrac{\sum_i x_i y_i}{\sum_i x_i^2}\;}$

</div>

The single-feature OLS estimate — and a template for the full normal equation $\hat{\boldsymbol\theta} = (X^\top X)^{-1}X^\top \mathbf{y}$.

---

# Practice problems

Try these on paper; answers worked through in the notebook (`lec00-mle-map.ipynb`).

<div class="math-box">

**P1.** A coin gives 12 heads in 20 flips. Compute the MLE for $p$ and the negative log-likelihood at that estimate.

**P2.** Write down the conditional distribution and the per-example log-likelihood for **Poisson regression** ($Y \mid x \sim \text{Poisson}(\exp(\theta^\top x))$). What is the resulting NLL loss?

**P3.** For a 3-class softmax with logits $\mathbf{z} = [1, 2, 0]$ and true class $y = 1$, compute (a) the predicted probabilities, (b) the cross-entropy loss, (c) the gradient on each logit.

**P4.** A coin's true bias is $p = 0.3$. You see 0 heads in 5 flips. What is the MLE? Why is the answer absurd, and what would a $\text{Beta}(2, 2)$ prior change?

</div>

---

# The one-sentence takeaway

<div class="insight">

**Every loss function is a negative log-likelihood in disguise.**

</div>

*Next (L00B) · put a prior on $\boldsymbol\theta$ — and every regularizer turns out to be a prior in disguise.*

---

<!-- _class: section-divider -->

# Appendix · optional depth

*Why the Normal is everywhere — CLT, max-entropy, closed-under-linear. Reference material; not examinable.*

---

# Why Normal · the Central Limit Theorem

**Statement (informal)** · let $X_1, X_2, \ldots, X_N$ be IID with mean $\mu$ and variance $\sigma^2$. Define the standardized sum ·

<div class="math-box">

$$Z_N = \frac{1}{\sqrt{N}}\sum_{i=1}^N \frac{X_i - \mu}{\sigma}$$

Then as $N \to \infty$, $Z_N \to \mathcal{N}(0, 1)$ — **regardless of the original distribution** of $X_i$.

</div>

**Implication** · any quantity arising as the *aggregate of many small effects* looks Gaussian. Sensor noise, human height, daily temperature deviation — all approximately Normal because the underlying causes are sums of many small contributions.

---

# CLT · a worked numeric

A clean way to see CLT in action ·

<div class="math-box">

Sum of 12 IID $\text{Uniform}[0, 1]$ samples ·

$$S = \sum_{i=1}^{12} U_i,\quad U_i \stackrel{\text{iid}}{\sim} \text{Uniform}[0,1]$$

- **Mean** of $S$ · $12 \cdot 0.5 = 6$.
- **Variance** of $S$ · $12 \cdot 1/12 = 1$.
- **Distribution** of $S$ · approximately $\mathcal{N}(6, 1)$.

</div>

Historically used to *generate* Gaussian samples before better algorithms (Box-Muller) existed · subtract 6, you get a draw from $\mathcal{N}(0, 1)$.

---

# CLT in pictures · sum of N uniforms

![w:920px](figures/lec00/svg/clt_demo.svg)

<div class="math-box">

Sum of $N$ IID $\text{Uniform}[0, 1]$ samples, repeated 5000 times. **One uniform** is uniform; **two uniforms summed** form a triangular density; by **N = 30** the sum is essentially indistinguishable from a Gaussian. The CLT is not a special property of any one distribution — it's an *attractor* that almost any IID sum flows to.

</div>

---

# Why Normal · maximum entropy

**Entropy** quantifies "how spread-out / uncertain" a distribution is. Among all distributions with a *given* mean $\mu$ and variance $\sigma^2$ ·

<div class="math-box">

$$\boxed{\;\arg\max_{p}\; H(p) \;\text{ s.t. }\; \mathbb{E}_p[Y] = \mu,\; \mathbb{E}_p[(Y - \mu)^2] = \sigma^2 \;\;\Longrightarrow\;\; p = \mathcal{N}(\mu, \sigma^2)\;}$$

</div>

**Reading** · *"if all you know about a quantity is its first two moments, the least committal probability model is Gaussian."* This is **Occam's razor for distributions** — don't bake in assumptions you can't justify.

---

# Why Normal · closed under linear operations

Two structural facts make Gaussians uniquely well-behaved under linear maps ·

<div class="math-box">

**Sum of independent Gaussians is Gaussian.** If $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$ are independent ·
$$X + Y \sim \mathcal{N}\bigl(\mu_1 + \mu_2,\; \sigma_1^2 + \sigma_2^2\bigr)$$

**Affine transform of a Gaussian is Gaussian.**
$$aY + b \sim \mathcal{N}\bigl(a\mu + b,\; a^2 \sigma^2\bigr)$$

</div>

**No other common distribution behaves this nicely.** Sum of two Bernoullis isn't Bernoulli; sum of two uniforms isn't uniform. The Gaussian is the *fixed point* of summing.

This is why Gaussians compound cleanly under repeated additive operations.

---

# Why Normal · where this pays off (1/2)

Two consequences you'll see in the next few weeks ·

<div class="math-box">

**Diffusion (L21)** · the forward process adds Gaussian noise at every step. The closed-form jump

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon \sim \mathcal{N}(0, I)$$

exists **exactly because sums of independent Gaussians are Gaussian** — no integral needed.

</div>

<div class="math-box">

**Kalman filter** · linear-Gaussian state-space models have a closed-form posterior at every time step. Used in robotics, control, and signal processing — and a stepping-stone to L19's variational inference.

</div>

---

# Why Normal · where this pays off (2/2)

<div class="math-box">

**Reparameterization trick (L19/L21)** · sampling from a non-standard Normal is just an affine transform of a standard one ·

$$z = \mu + \sigma \cdot \epsilon,\quad \epsilon \sim \mathcal{N}(0, 1)$$

</div>

This is the **single most important sampling trick in deep learning** ·

- VAEs (L19) · sample latent codes through it.
- Diffusion (L21) · every denoising step uses it.
- Bayesian neural nets · weights are sampled this way.

All three rely on Gaussians being closed under affine maps. Each lecture above is a direct consequence of the two structural properties we just stated.