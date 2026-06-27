---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Bayes, MAP, Regularization &amp; KL

## A Probabilistic View of ML · Part 2

### Lecture 0B · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Recap from L00

In L00 we built the probabilistic framework ·

<div class="math-box">

1. A model defines a **conditional distribution** $p(y \mid x; \theta)$.
2. **Likelihood** of a dataset = $\prod_i p(y_i \mid x_i; \theta)$ · take log + sum.
3. **MLE** = $\arg\max_\theta \sum_i \log p(y_i \mid x_i; \theta) = \arg\min_\theta \text{NLL}$.
4. Plug in the right distribution → the right loss falls out · **Normal → MSE · Bernoulli → BCE · Categorical → CE**.

</div>

Today · what changes when we add a **prior** on $\theta$ — and why **KL divergence** is the right language for every regularizer and generative model ahead.

---

# Today's question

You learned in ES 335 that L2 / L1 regularization "shrinks weights." We never said *why*.

<div class="popquiz">

(a) L2 is squared, L1 is absolute · they just happen to work.
(b) They're arbitrary penalties · pick whichever generalizes on val data.
(c) Each is the **negative log of a prior distribution on $\boldsymbol\theta$** — L2 ⟺ Gaussian prior · L1 ⟺ Laplace prior.

Stop and guess — and hold your answer. By the end of L00B you won't just know which it is; you'll be able to **derive** it from one machinery (MAP).

</div>

---

# Learning outcomes

By the end of this lecture you will be able to ·

<div class="math-box">

1. **Apply Bayes' rule** and identify prior / likelihood / posterior / evidence.
2. **Derive MAP** as MLE + log-prior · explain when it differs from MLE.
3. **Derive L2** as MAP with a Gaussian prior on weights · **L1** as MAP with a Laplace prior · explain L1's sparsity geometrically.
4. State and use **KL divergence** · recognize **cross-entropy = entropy + KL** · and that cross-entropy loss = NLL of the Categorical for one-hot labels.
5. Distinguish **forward vs reverse KL** and predict their failure modes (mode-covering vs mode-seeking).
6. **Connect** these foundations to VAE / diffusion / RLHF / distillation throughout the rest of the course.

</div>

---

# ▶ Interactives for L00B

<div class="paper">

- [bayesian-posterior](https://nipunbatra.github.io/interactive-articles/bayesian-posterior/) · drag the prior, watch the posterior update.
- [mle-map-coin](https://nipunbatra.github.io/interactive-articles/mle-map-coin/) · slide α, β, true p, N — watch MLE drift toward MAP.
- [info-theory](https://nipunbatra.github.io/interactive-articles/info-theory/) · entropy + KL with sliders.
- [demystifying-p-values](https://nipunbatra.github.io/interactive-articles/demystifying-p-values/) · companion for the disease-test slide.

</div>

---

# Bridge from ES 335 · you already know this

| From ES 335 | What changes today |
|---|---|
| Ridge penalty $\lambda \lVert \boldsymbol\theta \rVert_2^2$, used as a knob | revealed as a **Gaussian prior** on the weights |
| Lasso penalty $\lambda \lVert \boldsymbol\theta \rVert_1$, used as a knob | revealed as a **Laplace prior** on the weights |
| "L1 gives sparsity" — memorized | **derived** — from the prior's kink at zero |
| Cross-entropy as a loss formula | re-read as entropy + KL · the course's central distance |

<div class="keypoint">

The **weight-decay / ridge** you ran in ES 335 **= a Gaussian prior** on $\boldsymbol\theta$; the **lasso = a Laplace prior**. You used them as penalty knobs; today you *derive* them — and MAP turns out to be just **MLE + a prior**.

</div>

---

<!-- _class: section-divider -->

### PART 1

# Bayes' rule

A quick refresher (you met the ML form in L00) — then onto MAP

---

# Conditional probability · refresher

For two events $A, B$ with $P(B) > 0$ ·

<div class="math-box">

$$P(A \mid B) = \frac{P(A, B)}{P(B)}$$

Read · "given that $B$ happened, the probability $A$ also happened."

</div>

Cross-multiply and you get the **product rule** ·
$$P(A, B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A)$$

---

# Bayes' rule · the formula

From the two ways to factor $P(A, B)$ ·

$$P(A \mid B)\,P(B) = P(B \mid A)\,P(A)$$

Divide by $P(B)$ ·

<div class="math-box">

$$\boxed{\,P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}\,}$$

</div>

This **flips the conditional** · if you know $P(B \mid A)$ but want $P(A \mid B)$, Bayes is the bridge.

---

# Bayes worked example · disease test (setup)

Three terms first — most people mix these up ·

- **Prevalence** · how common the disease is · $P(D)$.
- **Sensitivity** · if you *have* it, the test catches it · $P(+ \mid D)$.
- **Specificity** · if you're *healthy*, the test stays negative · $P(- \mid \bar D)$.

Here · prevalence $1\%$, sensitivity $95\%$, specificity $95\%$. **You test positive — how likely are you to actually have the disease?**

<div class="math-box">

$D$ = have disease, $+$ = test positive ·
$P(D) = 0.01$ (prior) · $P(+\mid D) = 0.95$ (sensitivity) · $P(+\mid \bar D) = 0.05$ (= 1 − specificity)

</div>

*Stop and guess before the next slide.*

---

# Bayes worked example · the answer

<div class="math-box">

**Evidence (total probability of testing positive)** ·
$$P(+) = 0.95 \cdot 0.01 + 0.05 \cdot 0.99 = 0.0095 + 0.0495 = 0.059$$

**Posterior** ·
$$P(D \mid +) = \frac{P(+ \mid D)\,P(D)}{P(+)} = \frac{0.95 \cdot 0.01}{0.059} \approx \mathbf{0.16}$$

</div>

Despite a "95% accurate" test, a positive result gives only **16%** chance of disease.

This is the **base-rate fallacy** — and it's the same maths we'll apply to ML when the *prior* on $\theta$ is strong but the *likelihood* of any single data point is weak.

---

# Bayes for ML · flip onto $\theta$

In ML, $A$ becomes the parameter $\theta$ and $B$ becomes the data $\mathcal{D}$. You previewed *posterior ∝ likelihood × prior* in L00; here is the full four-term form.

<div class="math-box">

$$\boxed{\,p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\,p(\theta)}{p(\mathcal{D})}\,}$$

</div>

This is the **central equation of probabilistic ML** · how to update our belief about $\theta$ after seeing data $\mathcal{D}$. Each term has a name and a role (next slide).

---

# The four terms · names, roles, colors

<div class="math-box">

$$\underbrace{\color{#B85A3E}{p(\theta \mid \mathcal{D})}}_{\color{#B85A3E}{\text{posterior}}} \;=\; \frac{\overbrace{\color{#7B9E89}{p(\mathcal{D} \mid \theta)}}^{\color{#7B9E89}{\text{likelihood}}}\;\,\overbrace{\color{#4A6670}{p(\theta)}}^{\color{#4A6670}{\text{prior}}}}{\underbrace{\color{#7B6E5A}{p(\mathcal{D})}}_{\color{#7B6E5A}{\text{evidence}}}}$$

</div>

| Term | What it is | Where it comes from |
|:-:|:-:|:-:|
| <span style="color:#7B9E89">**Likelihood**</span> | how plausible is the data under $\theta$ | the model |
| <span style="color:#4A6670">**Prior**</span> | belief about $\theta$ before seeing data | choice / domain knowledge |
| <span style="color:#B85A3E">**Posterior**</span> | updated belief about $\theta$ after data | what we compute |
| <span style="color:#7B6E5A">**Evidence**</span> | $\int p(\mathcal{D} \mid \theta)\,p(\theta)\,d\theta$ — a normalizer | usually intractable, often ignored |

We will keep these colors consistent for the rest of the course.

---

# Why we usually ignore the evidence

The evidence $p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta)\,p(\theta)\,d\theta$ is a constant **with respect to $\theta$**. It does not change as we vary $\theta$.

<div class="keypoint">

For finding the *single best* $\theta$ (MLE or MAP), the evidence is irrelevant ·

$$\arg\max_\theta\, p(\theta \mid \mathcal{D}) = \arg\max_\theta\, p(\mathcal{D} \mid \theta)\,p(\theta)$$

We only need ·  **posterior $\propto$ likelihood $\times$ prior.**

</div>

The evidence becomes important only when we want a *full posterior* — Bayesian neural nets, model comparison, ELBO in VAEs (L19). For today, MLE + MAP, we drop it.

---

# Bayesian updating · the dynamic view

Bayes' rule is not one-shot · as data arrives, **today's posterior becomes tomorrow's prior**.

![w:620px](figures/lec00/svg/bayesian_updating.svg)

$$p(\theta \mid \mathcal{D}_1, \mathcal{D}_2) \propto p(\mathcal{D}_2 \mid \theta)\,\underbrace{p(\theta \mid \mathcal{D}_1)}_{\text{previous posterior = new prior}}$$

▶ Watch it update live · [bayesian-posterior](https://nipunbatra.github.io/interactive-articles/bayesian-posterior/).

---

# The Beta distribution · a prior over a probability

To put a prior on a coin's bias $p$ we need a distribution *over* $[0,1]$. The **Beta** is the natural choice ·

<div class="math-box">

$$p \sim \text{Beta}(\alpha, \beta), \qquad \text{mean} = \frac{\alpha}{\alpha+\beta}$$

Read $\alpha-1$ as "prior heads" and $\beta-1$ as "prior tails." $\text{Beta}(1,1)$ is **uniform** (no opinion); $\text{Beta}(2,2)$ is a gentle bump at $0.5$ ("probably fair").

</div>

---

# Conjugate updating · Beta prior, in pictures

![w:680px](figures/lec00/svg/beta_binomial_update.svg)

Beta prior + Bernoulli likelihood → Beta posterior · pure arithmetic · $\text{Beta}(\alpha, \beta) \to \text{Beta}(\alpha + \#H,\, \beta + \#T)$. More flips ⇒ sharper posterior, weaker prior.

---

# Estimator #1 · Maximum Likelihood (MLE)

The simplest possible estimator · **ignore the prior**, just maximize the likelihood.

<div class="math-box">

$$\hat\theta_{\text{MLE}} = \arg\max_\theta\, p(\mathcal{D} \mid \theta)$$

</div>

Read · *"what value of $\theta$ best explains the data we actually saw?"*

This was all of L00 · coin → linear regression → logistic regression → multiclass — each derived its loss as the negative log-likelihood under MLE.

---

# Estimator #2 · Maximum a Posteriori (MAP)

The Bayesian-flavoured estimator · **use the prior**, maximize the full posterior.

<div class="math-box">

$$\hat\theta_{\text{MAP}} = \arg\max_\theta\, p(\mathcal{D} \mid \theta)\,p(\theta)$$

</div>

Read · *"given my prior belief and the data, what's the most probable $\theta$?"*

This is Part 2 — the heart of today. MAP = MLE *plus* a regularizer that comes from the log-prior. With a Gaussian prior we'll recover **L2**; with a Laplace prior we'll recover **L1**. Same machinery, different prior.

---

# MLE vs MAP · the master sentence

<div class="keypoint">

**MAP = MLE + a prior on $\theta$.**

That single sentence is what makes **regularization fall out of the same machinery as the loss**. There is no separate "loss" and "regularizer" theory — it's all one Bayesian story, and L2/L1 are just choices of prior.

</div>

When the data is plentiful, the likelihood dominates and MAP $\to$ MLE. When data is scarce, the prior matters — and that's exactly when overfitting bites and regularization helps. Bayes' rule formalizes this *automatically*.

---

<!-- _class: section-divider -->

### PART 2

# MAP and the meaning of regularization

L2 from a Gaussian prior · L1 from a Laplace prior

---

# Why we need a prior · MLE's trap

You flip a coin **3 times** and see **3 heads**. MLE says $\hat p_{\text{MLE}} = 3/3 = 1.0$.

<div class="warning">

According to MLE, this coin is **certainly biased to always land heads**. Future tails impossible.

This is absurd. Three flips is not enough evidence to make such an extreme claim. You *know* most coins are roughly fair.

</div>

The fix · **encode that prior knowledge** into the inference. That gives MAP.

In ML, the analogous trap is overfitting · with finite data, MLE drives weights to whatever value most exactly fits the training set, even if those values are wildly extreme. A prior pulls them back.

---

# MAP · maximum a posteriori

By Bayes, $p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)\,p(\theta)$.

<div class="math-box">

$$\hat\theta_{\text{MAP}} = \arg\max_\theta\, p(\theta \mid \mathcal{D}) = \arg\max_\theta \bigl[\,p(\mathcal{D} \mid \theta)\,p(\theta)\,\bigr]$$

In log space ·

$$\hat\theta_{\text{MAP}} = \arg\max_\theta \bigl[\underbrace{\log p(\mathcal{D} \mid \theta)}_{\text{log-likelihood}} + \underbrace{\log p(\theta)}_{\text{log-prior}}\bigr]$$

</div>

Equivalently, **minimize the negative** ·
$$\hat\theta_{\text{MAP}} = \arg\min_\theta \bigl[\,L_{\text{NLL}}(\theta) - \log p(\theta)\,\bigr]$$

The first term is your usual loss. The second term is whatever the prior gives. **That second term is what we will recognize as L1 or L2.**

---

# MAP · the geometric picture (setup)

A 3-point dataset · $(x,y) = (1,2),(2,2),(3,4)$, model $\hat y = \theta_0 + \theta_1 x$. The data loss is an explicit quadratic in $(\theta_0, \theta_1)$ ·

<div class="math-box">

$$\text{SSE}(\theta_0,\theta_1) = (2-\theta_0-\theta_1)^2 + (2-\theta_0-2\theta_1)^2 + (4-\theta_0-3\theta_1)^2$$

</div>

Its contours are **ellipses** around the OLS solution $(\theta_0,\theta_1) = (\tfrac{2}{3}, 1)$; the prior $\theta_0^2 + \theta_1^2$ adds **circles** around 0.

---

# MAP · the geometric picture

![w:620px](figures/lec00/svg/map_geometry_lambda.svg)

MAP sits where data ellipses meet prior circles. Crank $\lambda$ up and the solution slides along the rust path from OLS toward 0 — **that path is ridge regression.**

---

# Gaussian prior · the idea

Each weight is *a priori* small, centred at zero · $p(\theta_j)=\mathcal{N}(0,\sigma_p^2)$ independently. The **covariance sets the shape** — we use the simplest, **isotropic** ($\Sigma=\sigma_p^2 I$).

![w:680px](figures/lec00/svg/gaussian_prior_cases.svg)

---

# Gaussian prior · setup (the algebra)

In $\sim$ notation · $\theta_j \sim \mathcal{N}(0, \sigma_p^2)$ for each weight, independently. Now the math ·

<div class="math-box">

Start from the Gaussian density ·
$$p(\theta_j) = \frac{1}{\sqrt{2\pi\sigma_p^2}}\,\exp\!\left(-\frac{\theta_j^2}{2\sigma_p^2}\right)$$

Take the log of both sides ·
$$\log p(\theta_j) = -\log\sqrt{2\pi\sigma_p^2} - \frac{\theta_j^2}{2\sigma_p^2}$$

The first term doesn't depend on $\theta_j$ — call it a constant ·
$$\log p(\theta_j) = -\frac{\theta_j^2}{2\sigma_p^2} + \text{const}$$

</div>

---

# Gaussian prior · stack across all weights

Independence across the $d$ weights means the joint factorises — and log turns the product into a sum ·

<div class="math-box">

$$p(\boldsymbol\theta) = \prod_{j=1}^d p(\theta_j)$$

$$\log p(\boldsymbol\theta) = \sum_{j=1}^d \log p(\theta_j) = \sum_{j=1}^d \left(-\frac{\theta_j^2}{2\sigma_p^2}\right) + \text{const}$$

</div>

---

# Gaussian prior · collect the sum into a norm

Pull the constant $-1/(2\sigma_p^2)$ out of the sum ·

<div class="math-box">

$$\log p(\boldsymbol\theta) = -\frac{1}{2\sigma_p^2}\underbrace{\sum_{j=1}^d \theta_j^2}_{=\,\|\boldsymbol\theta\|_2^2} + \text{const} \;=\; -\frac{1}{2\sigma_p^2}\,\|\boldsymbol\theta\|_2^2 + \text{const}$$

</div>

The constant drops out of any $\arg\min$. **Keep $-\dfrac{1}{2\sigma_p^2}\|\boldsymbol\theta\|_2^2$** — the only $\boldsymbol\theta$-dependent piece, and the seed of L2.

---

# Gaussian prior · L2 pops out (step by step)

**Start** · MAP objective ·
$$\hat{\boldsymbol\theta}_{\text{MAP}} = \arg\min_{\boldsymbol\theta} \bigl[\,L_{\text{NLL}}(\boldsymbol\theta) - \log p(\boldsymbol\theta)\,\bigr]$$

**Substitute** the Gaussian log-prior we just derived ·
$$= \arg\min_{\boldsymbol\theta} \bigl[\,L_{\text{NLL}}(\boldsymbol\theta) - \bigl(-\tfrac{1}{2\sigma_p^2}\|\boldsymbol\theta\|_2^2 + \text{const}\bigr)\,\bigr]$$

**Distribute the minus** and drop the constant ·
$$= \arg\min_{\boldsymbol\theta} \bigl[\,L_{\text{NLL}}(\boldsymbol\theta) + \tfrac{1}{2\sigma_p^2}\|\boldsymbol\theta\|_2^2\,\bigr]$$

**Rename** $\lambda := \dfrac{1}{2\sigma_p^2}$ to match standard ML notation ·

<div class="math-box">

$$\boxed{\;\hat{\boldsymbol\theta}_{\text{MAP}} = \arg\min_{\boldsymbol\theta} \bigl[\,L_{\text{NLL}}(\boldsymbol\theta) + \lambda\,\|\boldsymbol\theta\|_2^2\,\bigr]\;}$$

</div>

**This is exactly L2 regularization (a.k.a. ridge / weight decay).**

---

# L2 · reading $\lambda$ · the weight-decay connection

Recall $\lambda = \dfrac{1}{2\sigma_p^2}$ — the prior's *width* sets the penalty's *strength* ·

- **Strong belief that weights are small** (small $\sigma_p^2$) ⇒ **large $\lambda$** ⇒ heavy penalty ⇒ pulled far along the rust path toward 0.
- **Weak belief** (large $\sigma_p^2$) ⇒ **small $\lambda$** ⇒ MAP $\to$ MLE (stays near OLS).

<div class="keypoint">

**DL linkage** · this is exactly the `weight_decay` argument in `torch.optim.SGD` / `AdamW`. Every `weight_decay=1e-4` you've ever set *is* a Gaussian prior on every weight. You've been doing Bayes the whole time.

</div>

---

# Weight decay · code and effect

```python
# PyTorch — weight_decay IS the Gaussian-prior shrink
opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01)
```

With data gradient $g \approx 0$ (an idle weight), $\eta = 0.1$ ·

| per step | **no decay** ($\lambda=0$) | **with decay** ($\lambda=0.01$) |
|---|---|---|
| shrink factor | $\times 1.000$ | $\times 0.998$ |
| $\theta=[3,-4]$ after 1 step | $[3,\,-4]$ | $[2.994,\,-3.992]$ |
| after 100 steps | $[3,\,-4]$ | $[2.46,\,-3.27]$ |

Without decay an idle weight stays put; with decay it **drifts toward 0** every step — the Gaussian prior quietly pulling it in.

---

# Worked numeric · MAP with L2 (loss value)

Tiny dataset, 2 weights · $(\mathbf{x}_1, y_1) = ([1,1],\,0)$ and $(\mathbf{x}_2, y_2) = ([1,-1],\,7)$. Candidate $\boldsymbol\theta = [3, -4]$, noise $\sigma^2 = 1$.

<div class="math-box">

**Predict** · $\hat y_1 = 3(1){+}(-4)(1) = -1$ · $\hat y_2 = 3(1){+}(-4)(-1) = 7$.
**Residuals** · $y_1 - \hat y_1 = 1$ · $y_2 - \hat y_2 = 0$.
**Data NLL** (Gaussian) · $L_{\text{NLL}} = \tfrac{1}{2}\sum_i (y_i - \hat y_i)^2 = \tfrac{1}{2}(1^2 + 0^2) = 0.5$.

**L2 term** · $\|\boldsymbol\theta\|_2^2 = 9 + 16 = 25$; take $\lambda = 0.01$ ·
$$L_{\text{MAP}} = L_{\text{NLL}} + \lambda \|\boldsymbol\theta\|_2^2 = 0.5 + 0.01(25) = \mathbf{0.75}$$

</div>

The optimizer now pays a $0.25$ "tax" for these large weights.

---

# Worked numeric · MAP with L2 (the gradient)

Same setup, one SGD step with $\eta = 0.1$. The L2 term adds a very simple gradient ·

<div class="math-box">

$$\nabla_{\boldsymbol\theta}\bigl(\lambda\|\boldsymbol\theta\|_2^2\bigr) = 2\lambda\,\boldsymbol\theta$$

Combine with the data gradient $\mathbf{g}$ and rearrange ·
$$\boldsymbol\theta \leftarrow \boldsymbol\theta - \eta(\mathbf{g} + 2\lambda\boldsymbol\theta) = \underbrace{(1 - 2\eta\lambda)}_{\text{shrink}}\,\boldsymbol\theta \;-\; \eta\,\mathbf{g}$$

</div>

So the prior contributes a term that **multiplies $\boldsymbol\theta$ by a factor just below 1** *before* the data step — it literally shrinks the weights every iteration.

---

# Why it's called "weight decay"

With $\eta = 0.1$, $\lambda = 0.01$ · the shrink factor is $1 - 2\eta\lambda = 1 - 0.002 = 0.998$ ·

<div class="math-box">

$$\boldsymbol\theta_{\text{after shrink}} = 0.998 \cdot [3, -4] = [2.994, -3.992]$$

</div>

This is *literally* the `weight_decay` step in `torch.optim.SGD` — one line that multiplies $\boldsymbol\theta$ by $(1 - 2\eta\lambda)$ each iteration.

<div class="keypoint">

If you've trained with `weight_decay` and never realized it was a **Gaussian prior** quietly decaying your weights — that's exactly the point of today. The same pattern returns in **AdamW** (L05).

</div>

---

# Laplace prior · the idea

Swap the Gaussian for a **Laplace** prior · $p(\theta_j) = \dfrac{1}{2b}\exp\!\left(-\dfrac{|\theta_j|}{b}\right)$ — same "weights are small" idea, but a **sharp peak at 0** and **heavier tails**.

![w:660px](figures/lec00/svg/laplace_vs_gaussian.svg)

That spike at zero is the whole story · it puts real prior mass *right at* 0, so MAP parks weak weights *exactly* there.

---

# Laplace prior · setup (the algebra)

<div class="math-box">

Start from the Laplace density ·
$$p(\theta_j) = \frac{1}{2b}\,\exp\!\left(-\frac{|\theta_j|}{b}\right)$$

Take the log ·
$$\log p(\theta_j) = -\log(2b) - \frac{|\theta_j|}{b}$$

The first term doesn't depend on $\theta_j$ ·
$$\log p(\theta_j) = -\frac{|\theta_j|}{b} + \text{const}$$

</div>

Notice · the exponent is $|\theta_j|$ — **linear** in the absolute value. Compare to Gaussian's $\theta_j^2$ — **quadratic**. This single difference produces all of L1's distinctive behaviour.

---

# Laplace prior · stack across all weights

Independent prior on each weight $\Rightarrow$ joint factorises ·

<div class="math-box">

$$\log p(\boldsymbol\theta) = \sum_{j=1}^d \log p(\theta_j) = \sum_{j=1}^d \left(-\frac{|\theta_j|}{b}\right) + \text{const}$$

Pull out $-1/b$ ·
$$\log p(\boldsymbol\theta) = -\frac{1}{b}\underbrace{\sum_{j=1}^d |\theta_j|}_{=\,\|\boldsymbol\theta\|_1} + \text{const} \;=\; -\frac{1}{b}\,\|\boldsymbol\theta\|_1 + \text{const}$$

</div>

Same structural moves as the Gaussian case · only the *form* of the per-weight penalty differs.

---

# Laplace prior · L1 pops out (step by step)

**Start** · MAP objective ·
$$\hat{\boldsymbol\theta}_{\text{MAP}} = \arg\min_{\boldsymbol\theta} \bigl[\,L_{\text{NLL}}(\boldsymbol\theta) - \log p(\boldsymbol\theta)\,\bigr]$$

**Substitute** Laplace log-prior ·
$$= \arg\min_{\boldsymbol\theta} \bigl[\,L_{\text{NLL}}(\boldsymbol\theta) - \bigl(-\tfrac{1}{b}\|\boldsymbol\theta\|_1 + \text{const}\bigr)\,\bigr]$$

**Distribute minus**, drop constant ·
$$= \arg\min_{\boldsymbol\theta} \bigl[\,L_{\text{NLL}}(\boldsymbol\theta) + \tfrac{1}{b}\|\boldsymbol\theta\|_1\,\bigr]$$

**Rename** $\lambda := 1/b$ ·

<div class="math-box">

$$\boxed{\;\hat{\boldsymbol\theta}_{\text{MAP}} = \arg\min_{\boldsymbol\theta} \bigl[\,L_{\text{NLL}}(\boldsymbol\theta) + \lambda\,\|\boldsymbol\theta\|_1\,\bigr]\;}$$

</div>

**This is exactly L1 regularization (a.k.a. lasso).**

---

# L1 · how to use it in practice

<div class="keypoint">

**DL linkage** · L1 isn't typically a built-in optimizer option (it has a kink at 0 — see ahead for why that matters). To use it you add `lambda * w.abs().sum()` to the loss manually, or use **proximal gradient (ISTA)** which has a clean soft-thresholding step that produces *exact* zeros. We do this in [`lec00-mle-map.ipynb`](../notebooks/lec00-mle-map.ipynb).

</div>

---

# Worked numeric · L1 zeroes a weight, L2 doesn't

Same data gradient $g = 0.3$ acts on a tiny weight $\theta_j = 0.05$. $\eta = 1$, $\lambda = 0.5$.

<div class="math-box">

**L2 update** · $\theta_j \leftarrow \theta_j - \eta(g + 2\lambda \theta_j) = 0.05 - (0.3 + 2 \cdot 0.5 \cdot 0.05) = 0.05 - 0.35 = \mathbf{-0.30}$.
*Crossed zero, didn't land on it.*

**L1 (proximal / ISTA) update.**

Step 1 · gradient step on data term · $\theta'_j = 0.05 - 0.3 = -0.25$.
Step 2 · soft-threshold with $\eta\lambda = 0.5$ · $\theta_j \leftarrow \text{sign}(-0.25) \cdot \max(0.25 - 0.5, 0) = -1 \cdot 0 = \mathbf{0}$.

</div>

**L1 produced an exact zero; L2 did not.** That's the entire reason lasso induces *feature selection* and ridge does not.

**DL linkage** · this is **why early stopping + L2 weight decay** is the dominant DL recipe (we don't usually need exact sparsity in neural networks), but **L1 is the right tool for interpretable linear models** and for *learning a sparse mask* (think · MoE routing, structured pruning · L23).

---

# L1 vs L2 · summary table

| | **L2 (Ridge)** | **L1 (Lasso)** |
|---|---|---|
| Prior | $\mathcal{N}(0, \sigma_p^2)$ | $\text{Laplace}(0, b)$ |
| Log-prior penalty | $\lambda \sum_j \theta_j^2$ | $\lambda \,\lVert \boldsymbol\theta \rVert_1$ |
| Geometry | Circle | Diamond |
| Solution | small everywhere | many exactly zero |
| Gradient of penalty | $2\lambda \theta_j$ (smooth) | $\lambda\,\text{sign}(\theta_j)$ (kink at 0) |
| When to use | dense problems, "shrink everything" | feature selection, "kill weak features" |

<div class="keypoint">

L1 and L2 are **the same MAP machinery** — they differ only in the prior over $\theta$. Pick your prior, get your regularizer.

</div>

---

# A coin · MLE vs MAP

Back to the coin · 3 heads, 0 tails. MLE says $\hat p = 1.0$ (absurd).

Suppose your prior is $p \sim \text{Beta}(2, 2)$ — *"coin is probably near 0.5"*. MAP combines this with the likelihood ·

<div class="math-box">

Posterior ∝ Likelihood × Prior · $p^3 (1-p)^0 \cdot p^{2-1} (1-p)^{2-1} = p^4 (1-p)^1$

Maximize · $\log\text{post} = 4\log p + \log(1-p)$. Set derivative · $4/p - 1/(1-p) = 0$ ⇒ $p = 4/5 = 0.8$.

</div>

MAP says $\hat p = 0.8$ — **sensible**. Three heads is *some* evidence the coin is biased, but the prior keeps us from going to 1.0.

This is the same regularization story as L1/L2 on weights — just with a different distribution.

---

<!-- _class: section-divider -->

### PART 3

# Entropy & KL divergence

What is the right *distance* between two distributions — and why is every loss ahead "fit + don't drift"?

*A condensed tour — the full intuition-first treatment (surprise → coding → cross-entropy → KL) is **Lecture 0C**.*

---

# Every loss is an NLL — the master picture

![w:1000px](figures/lec00/svg/nll_master_poster.svg)

The whole course keeps instantiating this one recipe — each new model just changes **which distribution** is assumed.

---

# Information content · the surprise of a single outcome

Before defining entropy, define the **information content** (or "surprise") of one outcome ·

<div class="math-box">

$$I(y) := -\log p(y)$$

</div>

**Why this exact formula?** Three axioms we want $I$ to satisfy ·

1. $I$ depends **only on $p(y)$** (not on $y$ itself).
2. $I$ is **decreasing** in $p$ — rare events are more informative.
3. $I$ is **additive** for independent events · $I(y_1, y_2) = I(y_1) + I(y_2)$.

The **only** function satisfying all three is $I(y) = -c \log p(y)$ for $c > 0$ (Shannon 1948). With $c = 1$ and $\log = \log_2$, the unit is **bits**.

---

# Surprise · "snowing in Kashmir vs Gandhinagar"

Same headline · *"It snowed today."* Two locations ·

- **Kashmir in January** · $p(\text{snow}) \approx 0.5$ · $I = -\log_2 0.5 = 1$ bit. Mild surprise.
- **Gandhinagar in January** · $p(\text{snow}) \approx 10^{-6}$ · $I = -\log_2 10^{-6} \approx 20$ bits. Front-page news.

<div class="keypoint">

The **same event** carries different surprise depending on the distribution generating it. That is *exactly* what $I(y) = -\log p(y)$ formalizes.

</div>

This is why **log-likelihood scores a model** · if the model gives high probability to the data that *actually* happened, each observation is unsurprising under it — a small $-\log p$. A good model = low average surprise on real data = low loss.

---

# Information content · worked examples

<div class="math-box">

| Event | $p$ | $I = -\log_2 p$ | Interpretation |
|:-:|:-:|:-:|:-:|
| Fair coin lands heads | $0.5$ | $1$ bit | one yes/no question |
| Roll a 6 on a fair die | $1/6$ | $\approx 2.58$ bits | "less than 3 yes/no questions worth" |
| Win a 1-in-1024 lottery | $1/1024$ | $10$ bits | extremely surprising |
| The sun rises tomorrow | $\approx 1$ | $\approx 0$ bits | no information (already certain) |
| Sample a specific token from 50k vocab | $1/50000$ | $\approx 15.6$ bits | one token = one short word in English |

</div>

**Reading** · large $I(y)$ = "I learned a lot from observing $y$." Small $I(y)$ = "I expected this, no information gained."

This is **the per-sample log-loss** in disguise. Cross-entropy is just the *average* of these surprises.

---

# Entropy · the definition

The **entropy** of a distribution $p$ is the **expected information** of a draw from $p$ ·

<div class="math-box">

$$H(p) := \mathbb{E}_{Y \sim p}[I(Y)] = -\mathbb{E}_{Y \sim p}[\log p(Y)] = -\sum_y p(y)\,\log p(y)$$

</div>

In base 2, $H$ is in **bits** · roughly *the average number of yes/no questions needed to pin down a draw from $p$* — the average surprise per sample.

So entropy is what you get when you average the per-sample surprise from the previous slides. Big entropy = on average, draws from $p$ are surprising. Small entropy = mostly predictable.

---

# Entropy · worked numerics in bits

| Distribution | $H$ (bits) | Computation |
|:-:|:-:|:-:|
| Fair coin | $1$ | $-2 \cdot 0.5 \log_2 0.5 = 1$ |
| Biased coin $p = 0.99$ | $0.08$ | $-0.99 \log_2 0.99 - 0.01 \log_2 0.01$ |
| Uniform over 8 | $3$ | $-8 \cdot \tfrac{1}{8} \log_2 \tfrac{1}{8} = \log_2 8$ |
| One-hot (deterministic) | $0$ | nothing to encode |

<div class="keypoint">

**Entropy peaks for uniform** distributions and is **zero** for deterministic ones. A fair coin needs 1 bit per flip; a near-deterministic coin needs ~0 bits per flip; a uniform-over-K distribution needs $\log_2 K$ bits.

</div>

This explains why low-entropy classifier outputs (confident) need fewer bits to encode than high-entropy outputs (uncertain) — and why temperature in LLMs (which controls output entropy) controls "creativity vs determinism."

---

# Entropy of a Bernoulli · in pictures

![w:620px](figures/lec00/svg/entropy_bernoulli.svg)

$H(p) = -p \log_2 p - (1-p)\log_2(1-p)$ — symmetric, **peaks at 1 bit** for a fair coin, and falls to $0$ as the coin becomes deterministic.

---

# Entropy worked · Categorical (3-class)

Let $\boldsymbol\pi$ vary over a 3-class Categorical and compute $H$ in bits ·

<div class="math-box">

| Distribution $\boldsymbol\pi$ | $H(\boldsymbol\pi)$ (bits) | Interpretation |
|:-:|:-:|:-:|
| $(1, 0, 0)$ — deterministic | $0$ | already certain |
| $(0.7, 0.2, 0.1)$ — peaked | $-0.7\log_2 0.7 - 0.2\log_2 0.2 - 0.1\log_2 0.1 \approx 1.16$ | confident model |
| $(0.5, 0.3, 0.2)$ — moderate | $\approx 1.49$ | mixed evidence |
| $(\tfrac{1}{3}, \tfrac{1}{3}, \tfrac{1}{3})$ — uniform | $\log_2 3 \approx 1.58$ | maximum |

</div>

The maximum entropy of a $K$-Categorical is $\log_2 K$, achieved by the uniform. **A confident classifier has low-entropy predictions; a confused one has high-entropy predictions.** This connects directly to the **temperature** in LLM sampling — high $T$ pushes the categorical toward uniform (high entropy), low $T$ sharpens it (low entropy).

---

# KL divergence · the intuition (a scoring game)

Data really comes from $p$; you model it with $q$. Score each sample by its **surprise** $-\log q(y)$ and average ·

- Score with the **true** $p$ → the *best possible* average = **entropy** $H(p)$. You can't beat it.
- Score with a **wrong** $q$ → a worse average = **cross-entropy** $H(p, q)$.

<div class="keypoint">

**KL is the penalty** — the *extra* score for modelling $p$ with the wrong $q$ ·
$$\text{KL}(p\,\Vert\,q) = H(p, q) - H(p) \;\ge\; 0, \qquad = 0 \text{ iff } q = p.$$

</div>

*Intuition after L. Serrano's dice game and J.-B. Huang's weather example.*

---

# KL · counting the extra bits (weather)

True weather $p = (\tfrac12, \tfrac14, \tfrac14)$ has an optimal code costing $H(p) = 1.5$ bits/day. Build a code for the *wrong* forecast $q = (\tfrac14, \tfrac14, \tfrac12)$ and use it on real weather ·

![w:760px](figures/lec00/svg/kl_weather.svg)

$$H(p, q) = 1.75 \text{ bits} \;\Rightarrow\; \text{KL}(p\,\Vert\,q) = 1.75 - 1.5 = \mathbf{0.25}\text{ extra bits/day.}$$

---

# KL · the formula

Both pictures — "penalty" and "extra bits" — are the **same average extra surprise** ·

<div class="math-box">

$$\text{KL}(p \,\Vert\, q) = \mathbb{E}_{Y \sim p}\!\left[\log \frac{p(Y)}{q(Y)}\right] = \sum_y p(y)\,\log\frac{p(y)}{q(y)}$$

</div>

Zero iff $q = p$; **directed** — a gap *from $p$ to $q$*, not a symmetric distance. For the weather numbers, $\sum_x p\log_2(p/q)$ gives exactly $0.25$ bits, matching the count.

---

# KL · three properties

<div class="math-box">

1. $\text{KL}(p \| q) \ge 0$ for all $p, q$. (Gibbs' inequality — proof in two slides.)
2. $\text{KL}(p \| q) = 0$ **if and only if** $p = q$ everywhere.
3. **Asymmetric** · in general $\text{KL}(p \| q) \ne \text{KL}(q \| p)$.

</div>

Property 1 is what makes KL a sensible *loss-like* quantity — minimize it and you get to zero only when distributions match.

Property 2 says KL = 0 is the unique minimum. So **minimizing KL is the right objective** for matching distributions.

Property 3 is the surprising one. KL is not a metric — *which direction* you take matters. We'll see this matters a lot in "forward vs reverse KL" later.

---

# KL worked · two Bernoullis

Let $p = (0.7, 0.3)$ (true) and $q = (0.5, 0.5)$ (model).

<div class="math-box">

$$\text{KL}(p \,\Vert\, q) = \sum_y p(y)\log\frac{p(y)}{q(y)} = 0.7 \log\frac{0.7}{0.5} + 0.3 \log\frac{0.3}{0.5}$$

In nats (natural log) · $0.7 \cdot 0.336 + 0.3 \cdot (-0.511) = 0.235 - 0.153 = \mathbf{0.082}$ nats.

In bits · divide by $\ln 2 \approx 0.693$, giving $\approx 0.119$ bits.

</div>

**Asymmetry check** · $\text{KL}(q \,\Vert\, p) = 0.5 \log\frac{0.5}{0.7} + 0.5 \log\frac{0.5}{0.3} = -0.168 + 0.255 = 0.087$ nats — close but **different**. KL is not a metric.

---

# KL worked · two Categoricals (3-class)

True $p = (0.5, 0.3, 0.2)$. Two candidate models ·

<div class="math-box">

**Model A** · $q_A = (0.4, 0.4, 0.2)$ — close to truth.
$\text{KL}(p \,\Vert\, q_A) = 0.5 \log\tfrac{0.5}{0.4} + 0.3 \log\tfrac{0.3}{0.4} + 0.2 \log\tfrac{0.2}{0.2}$
$= 0.5 \cdot 0.223 + 0.3 \cdot (-0.288) + 0 = 0.112 - 0.086 = \mathbf{0.026}$ nats.

**Model B** · $q_B = (0.1, 0.1, 0.8)$ — very wrong on class 3.
$\text{KL}(p \,\Vert\, q_B) = 0.5 \log\tfrac{0.5}{0.1} + 0.3 \log\tfrac{0.3}{0.1} + 0.2 \log\tfrac{0.2}{0.8}$
$= 0.5 \cdot 1.609 + 0.3 \cdot 1.099 + 0.2 \cdot (-1.386)$
$= 0.805 + 0.330 - 0.277 = \mathbf{0.858}$ nats.

</div>

A roughly-aligned model has KL $\approx 0.03$; a wildly mismatched one has KL $\approx 0.86$. **KL ≈ 0 ⇒ the two distributions agree** ; large KL ⇒ they disagree, especially in directions where $p$ has mass but $q$ doesn't (mode-covering penalty).

---

# ⭐⭐⭐ Optional · Why KL ≥ 0 · Jensen's inequality (proof)

**Jensen's inequality** · for any *concave* function $\varphi$ and random variable $X$,
$$\mathbb{E}[\varphi(X)] \le \varphi(\mathbb{E}[X])$$

Since $\log$ is concave ·

<div class="math-box">

$$-\text{KL}(p \| q) = \mathbb{E}_p\!\left[\log\tfrac{q(Y)}{p(Y)}\right] \;\le\; \log\,\mathbb{E}_p\!\left[\tfrac{q(Y)}{p(Y)}\right] = \log\!\sum_y p(y)\,\tfrac{q(y)}{p(y)} = \log\!\sum_y q(y) = \log 1 = 0$$

So $-\text{KL} \le 0$, i.e. $\text{KL}(p \| q) \ge 0$. Equality iff $q(y)/p(y)$ is constant — i.e. $p = q$. ∎

</div>

This is the full proof of **Gibbs' inequality**. Jensen's inequality is the same tool used to derive the **ELBO** in VAEs (L19). Same trick, same place — once you see it once, you see it everywhere.

---

# Cross-entropy = entropy + KL · derive it

**Definition** ·
$$H(p, q) := -\mathbb{E}_{Y \sim p}[\log q(Y)] = -\sum_y p(y) \log q(y)$$

**Algebra · add and subtract $\log p(Y)$ inside the expectation** ·

$$H(p, q) = -\mathbb{E}_p[\log q(Y)]$$

$$= -\mathbb{E}_p\bigl[\log p(Y) - \log p(Y) + \log q(Y)\bigr]$$

$$= \underbrace{-\mathbb{E}_p[\log p(Y)]}_{=\,H(p)} \;+\; \underbrace{\mathbb{E}_p\!\left[\log p(Y) - \log q(Y)\right]}_{=\,\mathbb{E}_p[\log(p(Y)/q(Y))] \,=\, \text{KL}(p \,\|\, q)}$$

<div class="math-box">

$$\boxed{\;H(p, q) \;=\; H(p) \;+\; \text{KL}(p \,\|\, q)\;}$$

</div>

---

# Cross-entropy = entropy + KL · reading the terms

<div class="math-box">

$$H(p, q) \;=\; H(p) \;+\; \text{KL}(p \,\|\, q)$$

</div>

**Reading each term ·**
- **Cross-entropy $H(p, q)$** · expected bits to encode samples from $p$ if we used a code optimized for $q$.
- **Entropy $H(p)$** · irreducible cost. Independent of $q$.
- **KL** · the *extra* bits we waste because $q \ne p$. Always $\ge 0$.

**Why this matters for classification** · the **true label** distribution $p$ is one-hot. $H(\text{one-hot}) = 0$ (no uncertainty). So in that case · **cross-entropy = KL**. Minimising the cross-entropy loss *is* minimising KL to the true label distribution.

---

# Cross-entropy worked · the one-hot collapse

In classification, the truth $\mathbf{y}$ is **one-hot** and the model output $\hat{\boldsymbol\pi}$ is a softmax.

<div class="math-box">

For any one-hot, $H(\mathbf{y}) = 0$ — there's no uncertainty in the truth. So the cross-entropy formula collapses ·

$$H(\mathbf{y}, \hat{\boldsymbol\pi}) = \underbrace{H(\mathbf{y})}_{=\,0} + \text{KL}(\mathbf{y} \,\Vert\, \hat{\boldsymbol\pi}) = -\log\hat\pi_{\,y_{\text{true}}}$$

</div>

**Cross-entropy of a one-hot truth against a softmax model is exactly the NLL of the model on the true class.** This is the standard classification loss.

It's also exactly what we derived in L00 from the Bernoulli/Categorical assumption — same answer through two different lenses (NLL or KL). On the next slide we tabulate it across confidence levels.

---

# Cross-entropy · table across confidence levels

3 classes, true class is 1 (so $\mathbf{y} = (1, 0, 0)$). Vary the model output and read off the loss ·

<div class="math-box">

| Model $\hat{\boldsymbol\pi}$ | $-\log \hat\pi_1$ (nats) | "Sentiment" |
|:-:|:-:|:-:|
| $(0.99, 0.005, 0.005)$ | $0.010$ | confidently right · tiny loss |
| $(0.7, 0.2, 0.1)$ | $0.357$ | mostly right |
| $(0.34, 0.33, 0.33)$ | $1.079$ | uncertain (≈ uniform) |
| $(0.05, 0.5, 0.45)$ | $2.996$ | confidently wrong · big loss |

</div>

The loss is small **iff the model assigned high probability to the true class** and grows steeply when the model is *confidently wrong*. This asymmetric penalty is what makes cross-entropy a **proper scoring rule** — it rewards calibration, not just accuracy.

The classifier's training loss is just the average of this column over the dataset.

---

# MLE through the KL lens · setup

Define the **empirical data distribution** as the histogram of the training set, treated as a distribution ·

<div class="math-box">

$$\hat p_{\text{data}}(y) := \frac{1}{N}\sum_{i=1}^N \mathbb{1}[Y_i = y]$$

</div>

This is just a discrete probability mass with weight $1/N$ on each observed point.

Now the average log-likelihood becomes an **expectation under $\hat p_{\text{data}}$** ·
$$\frac{1}{N}\,\ell(\theta) = \frac{1}{N}\sum_{i=1}^N \log p_\theta(y_i) = \mathbb{E}_{Y \sim \hat p_{\text{data}}}[\log p_\theta(Y)]$$

So the score we already maximize is the *negative cross-entropy* between the empirical distribution and the model.

---

# MLE through the KL lens · result (step by step)

**Step 1 · normalize log-likelihood.** Divide by $N$ ·
$$\frac{1}{N}\ell(\boldsymbol\theta) = \frac{1}{N}\sum_{i=1}^N \log p_{\boldsymbol\theta}(y_i) = \mathbb{E}_{Y \sim \hat p_{\text{data}}}\bigl[\log p_{\boldsymbol\theta}(Y)\bigr]$$

**Step 2 · recognise the right-hand side** as the **negative cross-entropy** between $\hat p_{\text{data}}$ and $p_{\boldsymbol\theta}$ ·
$$\mathbb{E}_{\hat p_{\text{data}}}[\log p_{\boldsymbol\theta}] \;=\; -H(\hat p_{\text{data}}, p_{\boldsymbol\theta})$$

**Step 3 · apply the identity** from the previous derivation ·
$$H(\hat p_{\text{data}}, p_{\boldsymbol\theta}) \;=\; H(\hat p_{\text{data}}) + \text{KL}(\hat p_{\text{data}} \,\|\, p_{\boldsymbol\theta})$$

**Step 4 · drop the constant.** $H(\hat p_{\text{data}})$ doesn't depend on $\boldsymbol\theta$, so it doesn't affect the $\arg\max$ ·

<div class="math-box">

$$\arg\max_{\boldsymbol\theta}\, \ell(\boldsymbol\theta) \;=\; \arg\min_{\boldsymbol\theta}\, H(\hat p_{\text{data}}, p_{\boldsymbol\theta}) \;=\; \arg\min_{\boldsymbol\theta}\, \text{KL}(\hat p_{\text{data}} \,\|\, p_{\boldsymbol\theta})$$

</div>

**MLE = make the model distribution as KL-close as possible to the empirical distribution.**

---

# MLE = KL minimization · what it buys

<div class="keypoint">

**DL linkage** · *every* classifier you have ever trained with cross-entropy loss has been **silently minimizing this KL** — to the one-hot empirical distribution of class labels. The "summed per-example loss" you write in code is *exactly* the KL up to a constant.

</div>

This is also why MLE is **mode-covering** (forward KL) — the upcoming slides make the failure mode visible.

---

# MAP through the KL lens

Adding the log-prior to the previous slide ·

<div class="math-box">

$$\hat\theta_{\text{MAP}} = \arg\min_\theta\,\Bigl[\,\underbrace{\text{KL}(\hat p_{\text{data}} \,\|\, p_\theta)}_{\text{fit}}\; - \;\underbrace{\tfrac{1}{N}\log p(\theta)}_{\text{prior penalty}}\,\Bigr]$$

</div>

**L2 regularization** through this lens · "minimize KL-to-empirical, plus pay $\lambda \|\theta\|^2$ for straying from a Gaussian prior."

This is the **fit + don't drift in KL** template that recurs throughout the course.

---

# KL regularization · everywhere in DL

The same "fit + don't drift in KL" pattern appears in many places ·

<div class="math-box">

| Method | Loss structure | Reading |
|:-:|:-:|:-:|
| **L2 regularization** | NLL $+ \lambda \lVert\theta\rVert^2$ | don't drift from prior |
| **VAE (L19)** | reconstruction $+ \text{KL}(q_\phi(z\mid x) \Vert p(z))$ | encoder posterior close to standard normal |
| **DPO / RLHF (L16)** | reward $- \beta\,\text{KL}(\pi_\theta \Vert \pi_{\text{ref}})$ | policy close to base model |
| **Distillation (L23)** | $\text{KL}(p_{\text{teacher}} \Vert p_{\text{student}})$ | student matches teacher |

</div>

Every regularizer in modern DL is some form of **"don't drift too far in KL"** from a reference distribution. Once you see the pattern, you stop memorizing losses and start *deriving* them.

---

# KL is asymmetric · two directions, two problems

We've already seen that $\text{KL}(p \| q) \ne \text{KL}(q \| p)$. So *which direction* you minimize is itself a modelling choice — and the two directions optimize for very different things.

<div class="keypoint">

- **Forward KL** · $\text{KL}(p \,\|\, q_\theta)$ — "average over the truth $p$."
- **Reverse KL** · $\text{KL}(q_\theta \,\|\, p)$ — "average over the model $q_\theta$."

</div>

Both are valid. Both are used in DL. Knowing which one your loss optimizes tells you what failure mode to expect. Next two slides unpack each.

---

# Forward KL · mode-covering

<div class="math-box">

$$\text{KL}(p \,\|\, q_\theta) = \sum_y p(y) \log\frac{p(y)}{q_\theta(y)}$$

Read · "average the log-ratio over the **truth** $p$."

</div>

**Failure mode** · the expectation is taken over $p$, so a region where $p(y) > 0$ but $q_\theta(y) \approx 0$ contributes a *huge* penalty (log of a tiny number).

**Consequence · mode-covering.** $q_\theta$ is *forced* to put mass everywhere $p$ does. If $p$ has two modes, $q_\theta$ has to span both. Single-Gaussian fit to a bimodal target → blurry middle.

**This is what MLE optimizes** — recall MLE = $\arg\min_\theta \text{KL}(\hat p_{\text{data}} \| p_\theta)$. So MLE-trained models are *mode-covering* by construction.

---

# Reverse KL · mode-seeking

<div class="math-box">

$$\text{KL}(q_\theta \,\|\, p) = \sum_y q_\theta(y) \log\frac{q_\theta(y)}{p(y)}$$

Read · "average the log-ratio over the **model** $q_\theta$."

</div>

**Failure mode** · the expectation is taken over $q_\theta$, so a region where $q_\theta(y) > 0$ but $p(y) \approx 0$ contributes a huge penalty. The safest move for $q_\theta$ is to put mass *only* where $p$ is large.

**Consequence · mode-seeking.** $q_\theta$ concentrates on a single high-density region. Two-mode target → $q_\theta$ picks one mode and ignores the other.

**This is what reverse-KL variational inference, GAN-style training, and RL policy distillation optimize.** Sharper, more confident output — but mode collapse is a real risk. (Whole-model VAE samples are *blurry*, not sharp — that comes from the forward-KL reconstruction term, two slides on; the reverse KL here is the VAE's *latent* term $\text{KL}(q_\phi\|p(z))$.)

---

# Forward vs reverse KL · the picture

![w:920px](figures/lec00/svg/forward_reverse_kl.svg)

<div class="math-box">

Same bimodal target $p$ (grey shaded). Fit a single Gaussian by minimizing each direction of KL ·

- **Forward KL** spreads the Gaussian across both modes — covers all data, but spends mass on the gap between modes (blurry samples). Why **VAE samples are blurry**.
- **Reverse KL** concentrates on one mode — sharp samples but ignores the other half of the distribution. Why **GANs mode-collapse**.

The two errors are *opposite failure modes*. Knowing which KL direction your loss optimizes tells you which failure mode to expect.

</div>

---

# KL across the rest of the course

Once you see KL as the underlying object, every advanced model becomes a **specific KL minimization** ·

<div class="math-box">

| Model | KL it minimizes | Lecture |
|:-:|:-:|:-:|
| Classifier (CE loss) | $\text{KL}(\hat p_{\text{data}} \Vert p_\theta)$ | L1, every classifier |
| **VAE** | recon-NLL $+ \text{KL}(q_\phi(z\mid x) \,\Vert\, p(z))$ | L19 |
| **Diffusion (variational view)** | $\sum_t \text{KL}(q(x_{t-1}\mid x_t, x_0) \,\Vert\, p_\theta(x_{t-1}\mid x_t))$ | L21 |
| **GAN** | (approximately) Jensen-Shannon — symmetrized KL | L20 |
| **DPO / RLHF KL term** | $\text{KL}(\pi_\theta \,\Vert\, \pi_{\text{ref}})$ | L16 |
| **Knowledge distillation** | $\text{KL}(p_{\text{teacher}} \,\Vert\, p_{\text{student}})$ | L23 |

</div>

You will see KL terms in every generative or alignment loss for the next 24 lectures. Each is an instance of *what we just derived*.

---

# Practice problems

Try these on paper; the notebook has plotting and verification.

<div class="math-box">

**P1.** Re-derive Bayes' rule from $P(A, B) = P(A \mid B) P(B) = P(B \mid A) P(A)$. State which step is the "flip."

**P2.** A disease has prevalence 0.5%. A test has sensitivity 99% and specificity 95%. You test positive. What is $P(\text{disease} \mid +)$? Why is it not ≥ 99%?

**P3.** For linear regression with NLL = 0.4 and weights $\boldsymbol\theta = [-2, 3, 0.1]$, compute the MAP loss with (a) L2 penalty $\lambda = 0.5$, (b) L1 penalty $\lambda = 0.5$. Which weight does L1 push hardest toward zero?

**P4.** Compute $\text{KL}\bigl(\mathcal{N}(2, 0.25) \,\Vert\, \mathcal{N}(0, 1)\bigr)$ using the closed form. Interpret the answer.

**P5.** Show that for one-hot label $\mathbf{y}_i$ and softmax prediction $\hat{\boldsymbol\pi}_i$, cross-entropy equals $\text{KL}(\mathbf{y}_i \,\Vert\, \hat{\boldsymbol\pi}_i)$. (Hint · entropy of a one-hot is zero.)

**P6.** A bimodal target has modes at $\pm 2$ with equal mass. You fit a single Gaussian $q$. Sketch the optimum under (a) forward KL, (b) reverse KL. Which one is mode-covering?

</div>

---

# Let's do one · P2 worked (disease test)

Prevalence $0.5\%$, sensitivity $99\%$, specificity $95\%$. You test positive ·

<div class="math-box">

$$P(+) = \underbrace{0.99(0.005)}_{\text{true pos}} + \underbrace{0.05(0.995)}_{\text{false pos}} = 0.00495 + 0.04975 = 0.0547$$

$$P(D \mid +) = \frac{0.99(0.005)}{0.0547} \approx \mathbf{0.09}$$

</div>

Only ~9% — *not* ≥99%. The disease is so rare that false positives from the huge healthy majority swamp the true positives. **Base rates dominate** — the same lesson as the first disease-test slide.

---

# Foreshadow · how this lecture powers DL

Every advanced model in this course uses MLE (or MAP) under a specific distribution ·

<div class="math-box">

| Model | Output distribution | Loss = NLL of … | Lecture |
|:-:|:-:|:-:|:-:|
| LLM (next-token) | Categorical over vocab | $-\log p_\theta(y_{t+1} \mid y_{1:t})$ | L13–L15 |
| VAE | Normal pixel decoder + Gaussian latent | reconstruction NLL + KL to prior | L19 |
| GAN | implicit generator | min–max over $\log D, \log(1-D)$ | L20 |
| Diffusion | Gaussian forward process | MSE on predicted noise | L21–L22 |
| RLHF / DPO | Bradley–Terry preference pair | log-sigmoid of reward gap | L16 |

</div>

You now know what *all* of these losses are. They're NLLs.

---

# Notebook · MLE & MAP in PyTorch

<div class="notebook">

▶ **[`lec00-mle-map.ipynb`](../notebooks/lec00-mle-map.ipynb)** walks through everything we derived today ·

1. **MLE for a coin** with `torch.distributions.Bernoulli`.
2. **MLE for linear regression** with `torch.distributions.Normal` — recovers OLS.
3. **MLE for logistic regression** — `BCEWithLogitsLoss` equals NLL of Bernoulli.
4. **MAP with Gaussian prior** — recovers ridge regression.
5. **MAP with Laplace prior** — recovers lasso · sparsity emerges as $\lambda$ grows (ISTA / proximal step).
6. **Visualise** likelihood + posterior surfaces in 2D.

</div>

Same code skeleton; three lines change between MLE and MAP.

---

# Putting L00B together · the master sentence

<div class="math-box">

1. **Bayes' rule** lets us flip likelihood + prior → posterior.
2. **MAP** = $\arg\max_\theta \log P(\mathcal{D} \mid \theta) + \log P(\theta) = $ MLE + log-prior.
3. **L2 (ridge)** = MAP with **Gaussian** prior $\mathcal{N}(0, \sigma_p^2)$ on weights.
4. **L1 (lasso)** = MAP with **Laplace** prior on weights — *kink at 0* drives sparsity.
5. **KL divergence** = expected extra surprise from using $q$ when truth is $p$. Always $\ge 0$, zero iff $p = q$.
6. **Cross-entropy** = entropy + KL. For one-hot truth, cross-entropy *is* KL — every classifier minimizes KL.

</div>

Every regularizer in this course is a prior. Every generative-model loss is a KL.

---

# Common questions · FAQ

**Q.** Do we always have to pick a distribution before training?
**A.** Yes — implicitly or explicitly. When you write MSE, you have *implicitly* assumed Gaussian noise. When you write BCE, Bernoulli. The conscious choice is what we're advocating today.

**Q.** Why minimize NLL instead of maximize LL?
**A.** Pure convention. ML libraries minimize. Negate, minimize, same answer.

**Q.** What if my output isn't Bernoulli/Categorical/Gaussian?
**A.** Pick whatever distribution fits — Poisson for counts, Beta for probabilities, mixture-of-Gaussians for multimodal data. The recipe (NLL = loss) is universal.

**Q.** How does this connect to Bayesian deep learning?
**A.** Bayesian DL keeps the *full posterior* over $\theta$ instead of a single point estimate. We won't go there in this course, but everything we do today is the entry point.

---

<!-- _class: summary-slide -->

# Lecture 0B — summary

We made the probabilistic framework concrete ·

- **Conditional view** · the model outputs a distribution $p(y \mid \mathbf{x}, \theta)$, not a number.
- **Bayes' rule for parameters** · $p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)\,p(\theta)$ — four named terms.
- **MLE** · ignore the prior, maximize log-likelihood.
  - MSE = MLE under Gaussian noise.
  - BCE / CE = MLE under Bernoulli / Categorical outputs.
- **MAP** · MLE + a prior on $\theta$, log-prior becomes a regularizer.
  - L2 = MAP with Gaussian prior on weights.
  - L1 = MAP with Laplace prior — sparsity from diamond geometry.
- **NLL is the loss everywhere.** Every advanced model in this course will be an instance of this recipe.

### Read before Lecture 1

Strang Ch 1 (vectors / matrices) · Bishop & Bishop §2.1–2.3 (probability primer) · §4.1–4.3 (linear regression as MLE) · §5.4 (logistic regression).

### Next lecture

**Why deep learning** — why these tools alone aren't enough at scale, and what depth and non-linearity buy us.

---

# The one-sentence takeaway

<div class="insight">

**Every regularizer is a prior in disguise.**

</div>

*Next (L01) · the same probabilistic machinery — but now the features themselves are learned.*
