---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# A Probabilistic View of ML

## Distributions · Sampling · Likelihood · MLE

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
4. **Sample** from Bernoulli, Categorical, and Normal — and recognize the **reparameterization trick** as an affine map of a base sample.
5. **Derive MSE / BCE / categorical CE** as NLL under Gaussian / Bernoulli / Categorical outputs.
6. State **the MLE recipe** in 3 steps and apply it to coin, linear, logistic, multiclass.

L00B (next lecture) takes this and adds priors → MAP → L1/L2 → KL.

</div>

---

<!-- _class: section-divider -->

### PART 0

# Revision · linear & logistic regression

The loss functions we used without asking why

---

# Linear regression · the model

Given a dataset $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$ with $\mathbf{x}_i \in \mathbb{R}^d$ and continuous targets $y_i \in \mathbb{R}$, fit a linear model ·

<div class="math-box">

$$\hat y_i = \boldsymbol\theta^\top \mathbf{x}_i$$

(absorb the bias into $\boldsymbol\theta$ by appending a $1$ to each $\mathbf{x}_i$).

</div>

The prediction is a single real number — a *point estimate* of $y$.

So far, no probability anywhere in sight. We pick weights, we predict a number, we measure how wrong we are.

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

# Random variable · the basic object

A **random variable** $Y$ is a quantity whose value is uncertain. It follows a distribution $p$.

<div class="math-box">

$$Y \sim p$$

Read · *"$Y$ is **distributed as** $p$"*. The "$\sim$" is the central notation of this lecture.

</div>

Two flavours, depending on the type of value $Y$ takes ·

- **Discrete** $Y \in \{y_1, y_2, \ldots\}$ — described by a probability **mass** function $P(Y = y)$ summing to $1$.
- **Continuous** $Y \in \mathbb{R}$ — described by a probability **density** $p(y)$ integrating to $1$.

We'll use both. The notation is mostly the same.

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

# IID · the assumption that makes everything work

A dataset $\mathcal{D} = \{Y_1, \ldots, Y_N\}$ is **independent and identically distributed** if ·

<div class="math-box">

$$Y_i \stackrel{\text{iid}}{\sim} p(\cdot \mid \theta)$$

- **Identically distributed** · every $Y_i$ comes from the *same* distribution.
- **Independent** · knowing $Y_i$ tells you *nothing* about $Y_j$ for $i \ne j$.

</div>

These two assumptions together give us the **product factorization** ·
$$P(\mathcal{D} \mid \theta) = \prod_{i=1}^N P(Y_i \mid \theta)$$

This product is what becomes a sum after taking logs — and what becomes the **summed loss over a dataset** in every training loop. IID is the formal license to add up per-example losses.

When IID fails (time series, video frames, sensor logs from one device) we need different math · autoregressive models, state-space models, etc. For this course, treat batches as IID.

<div class="notebook">

▶ **Notebook · [`lec00-iid-demo.ipynb`](../notebooks/lec00-iid-demo.ipynb)** — `torch.distributions` walks through four cases (IID · independent-not-identical · identical-not-independent · neither). Includes sequence + histogram + lag-1 scatter plots, and shows how `sum(log p)` is *blind* to autocorrelation — so shuffling the batch matters.

</div>

---

# Bernoulli · the coin

Outcome $Y \in \{0, 1\}$, parameter $p \in [0, 1]$ = probability of "heads."

$$Y \sim \text{Bernoulli}(p)$$

<div class="math-box">

**Probability mass function** ·

$$P(Y = 1 \mid p) = p$$
$$P(Y = 0 \mid p) = 1 - p$$

</div>

Two outcomes, two probabilities, summing to 1. This is the simplest non-trivial distribution.

**Examples** · email is spam (Y=1) or not (Y=0) · patient has disease or not · pixel is foreground or background.

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

**Only counts matter** — a clean *sufficient statistic*. This collapse is what makes the MLE derivation in Part 4 a one-liner.

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

# Categorical · the K-sided die (formal)

Outcome $Y \in \{1, 2, \ldots, K\}$, parameter vector $\boldsymbol\pi = (\pi_1, \ldots, \pi_K)$ with $\pi_k \ge 0$ and $\sum_k \pi_k = 1$.

$$Y \sim \text{Categorical}(\boldsymbol\pi)$$

<div class="math-box">

**Probability mass function** · $P(Y = k \mid \boldsymbol\pi) = \pi_k$

**One-hot compact form** · let $\mathbf{y} \in \{0,1\}^K$ with $y_k = 1$ if $Y = k$, else 0. Then ·
$$P(Y \mid \boldsymbol\pi) = \prod_{k=1}^K \pi_k^{\,y_k}$$

</div>

The product collapses · only one $y_k = 1$, so only one factor survives.

**Mean** · $\mathbb{E}[\mathbf{y}] = \boldsymbol\pi$. **Bernoulli is the special case $K = 2$.**

---

# Categorical · a worked example

**MNIST classifier** outputs $\boldsymbol\pi = (0.05, 0.7, 0.1, 0.02, \ldots, 0.05)$ for one image (10 components, summing to 1).

<div class="math-box">

The model says $P(\text{class }k)$ for each digit.

If the true label is $Y = 2$ (digit "2"), then the probability the model assigned to **the truth** is
$$\pi_{Y_{\text{true}}} = \pi_2 = 0.7$$

</div>

A perfect model would put all mass on class 2 (i.e. $\boldsymbol\pi = (0, 0, 1, 0, \ldots, 0)$). The further the prediction is from a one-hot truth, the *less likely* the data is under it — and the larger the cross-entropy loss.

**The softmax output of *any* classifier IS a Categorical distribution.** Treat it that way and the loss falls out automatically.

---

# Normal (Gaussian) · the bell curve

Continuous $Y \in \mathbb{R}$ with mean $\mu$ and variance $\sigma^2$.

$$Y \sim \mathcal{N}(\mu, \sigma^2)$$

<div class="math-box">

**Probability density function (PDF)** ·

$$p(y \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\,\exp\!\left(-\frac{(y - \mu)^2}{2\sigma^2}\right)$$

**Mean & variance** · $\mathbb{E}[Y] = \mu, \quad \text{Var}[Y] = \sigma^2$

</div>

The most important continuous distribution in all of statistics — and the seed of the MSE loss.

---

# Normal · three properties to memorize

<div class="math-box">

1. **Centred at $\mu$**, spread controlled by $\sigma$.
2. Density falls off **exponentially in the squared distance** $(y - \mu)^2$.
3. The squared exponent will be the **seed of MSE** in Part 4.

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

This **squared-distance penalty** $(y - \mu)^2$ is the exact form that becomes MSE when we maximize the likelihood over data — covered in Part 4.

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

**Reparameterization trick (next!)** · sampling from a non-standard Normal is just an affine transform of a standard one ·

$$z = \mu + \sigma \cdot \epsilon,\quad \epsilon \sim \mathcal{N}(0, 1)$$

</div>

This is the **single most important sampling trick in deep learning** ·

- VAEs (L19) · sample latent codes through it.
- Diffusion (L21) · every denoising step uses it.
- Bayesian neural nets · weights are sampled this way.

All three rely on Gaussians being closed under affine maps. Each lecture above is a direct consequence of the two structural properties we just stated.

---

# The conditional view · model outputs a distribution

In supervised learning the model **does not output a number**. It outputs the *parameters of a distribution* over $Y$ given the input $\mathbf{x}$.

<div class="math-box">

| Task | Model output | Conditional distribution |
|:-:|:-:|:-:|
| Linear regression | $\hat\mu_\theta(\mathbf{x}) = \boldsymbol\theta^\top \mathbf{x}$ | $Y \mid \mathbf{x} \sim \mathcal{N}(\hat\mu_\theta(\mathbf{x}),\,\sigma^2)$ |
| Logistic regression | $\hat p_\theta(\mathbf{x}) = \sigma(\boldsymbol\theta^\top \mathbf{x})$ | $Y \mid \mathbf{x} \sim \text{Bernoulli}(\hat p_\theta(\mathbf{x}))$ |
| $K$-class softmax | $\hat{\boldsymbol\pi}_\theta(\mathbf{x})$ | $Y \mid \mathbf{x} \sim \text{Categorical}(\hat{\boldsymbol\pi}_\theta(\mathbf{x}))$ |

</div>

Training asks · *under these conditional distributions, how likely are the labels we actually saw?* Maximize that — the rest follows.

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

<!-- _class: section-divider -->

### PART 1.5

# Sampling

How we draw from distributions — and why every generative model needs it

---

# Why we sample · two roles

Sampling appears all over deep learning, in two distinct roles ·

<div class="math-box">

| Role | What it means | Examples |
|:-:|:-:|:-:|
| **Generation** | Produce new instances from a learned distribution | LLM next-token, VAE images, diffusion, GAN |
| **Monte Carlo estimation** | Approximate an expectation we can't compute analytically | mini-batch SGD, REINFORCE, dropout averaging, evaluating ELBOs |

</div>

These two uses are technically the same operation — *draw $y \sim p$* — but conceptually different. Today we set up the primitives. The advanced uses (reparameterization in VAEs, ancestral sampling in diffusion, nucleus sampling in LLMs) all reduce to combinations of what's on the next four slides.

---

# ⭐⭐⭐ Optional · the master sampling primitive · inverse CDF

For any 1-D distribution with CDF $F(y) = P(Y \le y)$ ·

<div class="math-box">

1. Draw $u \sim \text{Uniform}[0, 1]$.
2. Return $y = F^{-1}(u)$.

</div>

**Why it works** · $P(Y \le y) = P(F^{-1}(u) \le y) = P(u \le F(y)) = F(y)$. ✓

![w:560px](figures/lec00/svg/inverse_cdf.svg)

---

# ⭐⭐⭐ Optional · Inverse CDF · worked on a Bernoulli

To sample $Y \sim \text{Bernoulli}(p)$ ·

<div class="math-box">

The CDF jumps from $0$ to $1-p$ at $y = 0$, then from $1-p$ to $1$ at $y = 1$.

Inverting ·

```python
return 0 if u < 1 - p else 1
```

</div>

One uniform draw + one comparison · this is what `torch.bernoulli` does internally. **Same algorithm extends to any 1-D distribution** as long as you can compute the CDF.

---

# Sampling from a Categorical · the algorithm

Categorical with probabilities $\boldsymbol\pi = (\pi_1, \ldots, \pi_K)$. The CDF is the **stick-breaking cumulative** ·

<div class="math-box">

Build $c_k = \sum_{j=1}^k \pi_j$ (so $c_K = 1$). Draw $u \sim \text{Uniform}[0, 1]$. Return the smallest $k$ such that $c_k \ge u$.

</div>

**Worked** · $\boldsymbol\pi = (0.1, 0.4, 0.3, 0.2)$ → cumulative $(0.1, 0.5, 0.8, 1.0)$.

Draw $u = 0.55$ · smallest $k$ with $c_k \ge 0.55$ is $k = 3$. Return class **3**.

---

# Categorical sampling · why it matters

This is what `torch.multinomial` does · one uniform + one cumulative scan.

<div class="keypoint">

**Every LLM samples its next token with exactly this algorithm.**

The model produces a vector of $K$ probabilities (where $K \approx$ vocab size, often 50,000+) and we run a single Categorical draw.

</div>

You will see this primitive in ·
- L13–L15 · LLM token generation
- L19 · VAE discrete latents
- L21 · diffusion class-conditional sampling

Knowing it once means knowing it everywhere.

---

# Sampling from a Normal · the affine trick

To sample $Y \sim \mathcal{N}(\mu, \sigma^2)$ ·

<div class="math-box">

1. Sample $\epsilon \sim \mathcal{N}(0, 1)$ (e.g. via Box-Muller, or just `torch.randn(...)`).
2. Return $y = \mu + \sigma \cdot \epsilon$.

</div>

**Why this works** · the affine transform of a Gaussian is a Gaussian (the "closed under linear ops" property from earlier). If $\epsilon \sim \mathcal{N}(0, 1)$, then $\mu + \sigma\epsilon \sim \mathcal{N}(\mu, \sigma^2)$ — exactly what we wanted.

So we only ever need a routine to draw from $\mathcal{N}(0, 1)$. Everything else is multiplication and addition.

---

# The reparameterization trick · preview of L19

The same affine trick is one of the most important ideas in modern DL.

<div class="keypoint">

Let the model output $\mu_\theta(\mathbf{x})$ and $\sigma_\theta(\mathbf{x})$ as deterministic functions of an input. To sample a **stochastic** output ·

$$y = \mu_\theta(\mathbf{x}) + \sigma_\theta(\mathbf{x}) \cdot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, 1)$$

The randomness $\epsilon$ is **outside** the gradient path. $\mu_\theta, \sigma_\theta$ are deterministic ⇒ we can **backprop through the sample**.

</div>

This is what makes **VAEs trainable** (L19) and powers the entire **diffusion** stack (L21–L22). You'll see this exact form repeatedly — and you now know where it comes from.

---

# LLM token sampling · preview of L14

You now know the primitive (sampling from a Categorical). LLM generation is just Categorical sampling at every step — but with a few tweaks to control diversity ·

<div class="math-box">

| Strategy | What it does | When to use |
|:-:|:-:|:-:|
| **Greedy** ($\arg\max$) | Pick the most likely token | Deterministic; safe but boring |
| **Temperature** $T$ | Sample from $\tilde\pi_k \propto \pi_k^{1/T}$ | $T \to 0$ = greedy; $T = 1$ = unchanged; $T > 1$ = more random |
| **Top-$k$** | Keep top $k$ logits, renormalize, sample | Caps diversity at the top |
| **Top-$p$ (nucleus)** | Keep smallest set with cumulative prob $\ge p$ | Adapts to the model's confidence |

</div>

L14 covers the full story. The point today · the underlying operation is *sampling from a Categorical* — exactly the inverse-CDF primitive from two slides ago.

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

It is **not** a probability over $\theta$ (we'll get that from Bayes' rule next). It is the data's probability, viewed as a function of $\theta$.

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

---

# The fix · take the log

The logarithm turns products into sums and is monotonic — so maxima are preserved.

<div class="math-box">

$$\ell(\theta) := \log \mathcal{L}(\theta) = \log \prod_{i=1}^N P(y_i \mid \theta) = \sum_{i=1}^N \log P(y_i \mid \theta)$$

</div>

Now the dataset's "score" is a **sum of $N$ moderate negative numbers** — numerically stable and easy to differentiate term-by-term.

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

### PART 4

# Maximum Likelihood Estimation

Concrete derivations · coin · linear regression · logistic regression

---

# Pop quiz · which course did this student come from?

Three sections of the same exam · grades modelled as Normal ·

| Course | Mean $\mu$ | Std $\sigma$ |
|---|---|---|
| C1 | 80 | 10 |
| C2 | 70 | 10 |
| C3 | 90 | 5 |

A student's marks $y = 82$. Which course is this student most likely from?

*Stop and guess before the next slide.*

---

# Answer · pick the distribution that best explains the data

Evaluate the density $\mathcal{N}(82 \mid \mu, \sigma^2)$ for each course ·

$$p_1(82) \approx 0.0388,\quad p_2(82) \approx 0.0194,\quad p_3(82) \approx 0.0299$$

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

This is your first MLE derivation. Same recipe applies to everything below.

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

![w:780px](figures/lec00/svg/mle_linear_regression.svg)

<div class="math-box">

Each data point is one **draw** from a Gaussian whose mean lies on the regression line. MLE asks · *which line $\boldsymbol\theta^\top \mathbf{x}$ makes the observed $y$'s most probable?*

Equivalently · *which line minimizes the squared distance from each point to the line* — exactly the OLS objective. That equivalence is the whole derivation.

</div>

This is the only modelling choice. Everything else is algebra.

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

If the noise had been Laplace ($\epsilon \sim \text{Laplace}$), the same derivation would give MAE (mean absolute error) instead. Loss design = noise model.

---

# MLE for logistic regression · the assumption

Now $y \in \{0, 1\}$. Model ·

$$Y \mid \mathbf{x} \sim \text{Bernoulli}\bigl(\,\hat p_\theta(\mathbf{x})\,\bigr),\qquad \hat p_\theta(\mathbf{x}) = \sigma(\boldsymbol\theta^\top \mathbf{x})$$

Per-example probability ·

<div class="math-box">

$$p(y \mid \mathbf{x}, \boldsymbol\theta) = \hat p_\theta(\mathbf{x})^{\,y}\,\bigl(1 - \hat p_\theta(\mathbf{x})\bigr)^{1 - y}$$

</div>

This is the same compact Bernoulli form as the coin — except $\hat p$ now depends on $\mathbf{x}$.

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

# Multiclass · the assumption

Now $y \in \{1, 2, \ldots, K\}$ — one of $K$ mutually exclusive classes. We need the model to output a full **probability distribution** over the $K$ classes. We use the **softmax**.

<div class="math-box">

For each class $k$, learn a weight vector $\boldsymbol\theta_k \in \mathbb{R}^d$. Compute logits $z_k = \boldsymbol\theta_k^\top \mathbf{x}$ for $k = 1, \ldots, K$.

**Softmax** turns logits into probabilities ·
$$\hat\pi_k(\mathbf{x}) = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)} = \frac{\exp(\boldsymbol\theta_k^\top \mathbf{x})}{\sum_{j=1}^K \exp(\boldsymbol\theta_j^\top \mathbf{x})}$$

Two properties · $\hat\pi_k > 0$ (because $\exp > 0$) and $\sum_k \hat\pi_k = 1$. So $\hat{\boldsymbol\pi}$ is a valid distribution.

</div>

Modelling assumption · $Y \mid \mathbf{x} \sim \text{Categorical}(\hat{\boldsymbol\pi}_\theta(\mathbf{x}))$. Binary logistic is the special case $K = 2$ (with the softmax collapsing to a sigmoid).

---

# Multiclass · the per-example log-likelihood

If example $i$ has true class $y_i \in \{1, \ldots, K\}$, the probability the model assigns to that label is $\hat\pi_{i,\,y_i}$ — i.e. the entry of the softmax at position $y_i$.

<div class="math-box">

$$p(y_i \mid \mathbf{x}_i, \theta) = \hat\pi_{i,\,y_i}$$

Equivalently with one-hot encoding $\mathbf{y}_i \in \{0, 1\}^K$ ·
$$p(y_i \mid \mathbf{x}_i, \theta) = \prod_{k=1}^K \hat\pi_{i, k}^{\,y_{i, k}}$$

Take logs ·
$$\log p(y_i \mid \mathbf{x}_i, \theta) = \sum_{k=1}^K y_{i, k}\,\log \hat\pi_{i, k} = \log \hat\pi_{i,\, y_i}$$

</div>

Same compact-Bernoulli trick as before — only one term in the sum survives because the one-hot vector has a single 1.

---

# Multiclass · NLL = categorical cross-entropy

Sum over the dataset and negate ·

<div class="math-box">

$$L_{\text{NLL}}(\theta) = -\sum_{i=1}^N \log \hat\pi_{i,\,y_i} \;=\; -\sum_{i=1}^N \sum_{k=1}^K y_{i,k}\,\log \hat\pi_{i,k}$$

</div>

The right-hand form is the textbook **categorical cross-entropy** — true distribution $\mathbf{y}_i$ (one-hot) against predicted distribution $\hat{\boldsymbol\pi}_i$.

This loss powers every multiclass classifier in this course · CIFAR-10 ($K=10$), ImageNet ($K=1000$), and the **next-token loss in every LLM** ($K = $ vocab size, often 50,000+) in L13–L15.

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
$L = -\log \hat\pi_{\text{cat}} = -\log 0.659 = \mathbf{0.418}$

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

# Practice problems

Try these on paper; answers worked through in the notebook (`lec00-mle-map.ipynb`).

<div class="math-box">

**P1.** A coin gives 12 heads in 20 flips. Compute the MLE for $p$ and the negative log-likelihood at that estimate.

**P2.** Show that for $Y \mid x \sim \mathcal{N}(\theta x, \sigma^2)$ with **known** $\sigma^2$, the MLE for $\theta$ is $\hat\theta = \sum_i x_i y_i / \sum_i x_i^2$. (Hint · single-feature case of OLS.)

**P3.** Write down the conditional distribution and the per-example log-likelihood for **Poisson regression** ($Y \mid x \sim \text{Poisson}(\exp(\theta^\top x))$). What is the resulting NLL loss?

**P4.** For a 3-class softmax with logits $\mathbf{z} = [1, 2, 0]$ and true class $y = 1$, compute (a) the predicted probabilities, (b) the cross-entropy loss, (c) the gradient on each logit.

**P5.** A coin's true bias is $p = 0.3$. You see 0 heads in 5 flips. What is the MLE? Why is the answer absurd, and what would a $\text{Beta}(2, 2)$ prior change?

</div>
