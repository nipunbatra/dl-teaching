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

You learned in ES 654 that L2 / L1 regularization "shrinks weights." We never said *why*.

<div class="popquiz">

(a) L2 is squared, L1 is absolute · they just happen to work.
(b) They're arbitrary penalties · pick whichever generalizes on val data.
(c) Each is the **negative log of a prior distribution on $\boldsymbol\theta$** — L2 ⟺ Gaussian prior · L1 ⟺ Laplace prior.

Stop and guess. By the end of L00B the answer is **(c)** — derived from one machinery (MAP).

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


<!-- _class: section-divider -->

### PART 3

# Bayes' rule

Inverting conditional probabilities — the foundation of MAP

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

A disease has prevalence 1%. A test has sensitivity 95% and specificity 95%.

**You test positive. How likely are you to have the disease?**

<div class="math-box">

Let $D$ = "have disease", $+$ = "test positive".

- **Prior** · $P(D) = 0.01$, $P(\bar D) = 0.99$
- **Likelihood** · $P(+ \mid D) = 0.95$, $P(+ \mid \bar D) = 0.05$

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

In ML, $A$ becomes the parameter $\theta$ and $B$ becomes the data $\mathcal{D}$.

<div class="math-box">

$$\boxed{\,p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\,p(\theta)}{p(\mathcal{D})}\,}$$

</div>

This is the **central equation of probabilistic ML**. It tells us how to update our belief about $\theta$ after seeing data $\mathcal{D}$. Each term has a name and a role.

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

Bayes' rule is not a one-shot operation. As more data arrives, **today's posterior becomes tomorrow's prior** ·

<div class="math-box">

After dataset $\mathcal{D}_1$ ·
$$p(\theta \mid \mathcal{D}_1) \propto p(\mathcal{D}_1 \mid \theta)\,p(\theta)$$

Now observe more data $\mathcal{D}_2$, independent of $\mathcal{D}_1$ given $\theta$ ·
$$p(\theta \mid \mathcal{D}_1, \mathcal{D}_2) \propto p(\mathcal{D}_2 \mid \theta)\,p(\theta \mid \mathcal{D}_1)$$

The previous posterior $p(\theta \mid \mathcal{D}_1)$ now plays the role of prior. Bayesian inference is **iterative belief updating**.

</div>

Practical issue · for general distributions, the posterior may not be in the same family as the prior, so each update changes the *shape* of the formula. **Conjugate priors** (next slide) avoid this · prior and posterior stay in the same family, and updates are just parameter arithmetic.

---

# Estimator #1 · Maximum Likelihood (MLE)

The simplest possible estimator · **ignore the prior**, just maximize the likelihood.

<div class="math-box">

$$\hat\theta_{\text{MLE}} = \arg\max_\theta\, p(\mathcal{D} \mid \theta)$$

</div>

Read · *"what value of $\theta$ best explains the data we actually saw?"*

This is what we'll spend Part 4 of this lecture on. Coin → linear regression → logistic regression → multiclass — all derive their loss as the negative log-likelihood under MLE.

---

# Estimator #2 · Maximum a Posteriori (MAP)

The Bayesian-flavoured estimator · **use the prior**, maximize the full posterior.

<div class="math-box">

$$\hat\theta_{\text{MAP}} = \arg\max_\theta\, p(\mathcal{D} \mid \theta)\,p(\theta)$$

</div>

Read · *"given my prior belief and the data, what's the most probable $\theta$?"*

This is Part 5 (Session 2). MAP = MLE *plus* a regularizer that comes from the log-prior. With a Gaussian prior we'll recover **L2**; with a Laplace prior we'll recover **L1**. Same machinery, different prior.

---

# MLE vs MAP · the master sentence

<div class="keypoint">

**MAP = MLE + a prior on $\theta$.**

That single sentence is what makes **regularization fall out of the same machinery as the loss**. There is no separate "loss" and "regularizer" theory — it's all one Bayesian story, and L2/L1 are just choices of prior.

</div>

When the data is plentiful, the likelihood dominates and MAP $\to$ MLE. When data is scarce, the prior matters — and that's exactly when overfitting bites and regularization helps. Bayes' rule formalizes this *automatically*.

---

<!-- _class: section-divider -->

### PART 5

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

# MAP · the geometric picture

![w:780px](figures/lec00/svg/map_geometry.svg)

<div class="math-box">

The MAP estimate is the point in $\theta$-space that **best balances** the data's preferences (likelihood) against your prior beliefs (prior).

</div>

---

# Gaussian prior · setup

We choose a prior $p(\theta_j) = \mathcal{N}(0, \sigma_p^2)$ for every weight, independently. In words · *"a priori, weights are small and centred at zero."*

<div class="math-box">

For a single weight ·
$\log p(\theta_j) = -\dfrac{\theta_j^2}{2\sigma_p^2} + \text{const}$

For the whole vector $\boldsymbol\theta \in \mathbb{R}^d$ (independent prior on each component) ·
$\log p(\boldsymbol\theta) = \sum_j \log p(\theta_j) = -\dfrac{1}{2\sigma_p^2}\sum_j \theta_j^2 + \text{const} = -\dfrac{1}{2\sigma_p^2}\|\boldsymbol\theta\|_2^2 + \text{const}$

</div>

The constant doesn't depend on $\boldsymbol\theta$, so it drops out of the optimization.

---

# Gaussian prior · L2 pops out

Plug into the MAP objective ·

$$L_{\text{MAP}}(\boldsymbol\theta) = L_{\text{NLL}}(\boldsymbol\theta) - \log p(\boldsymbol\theta)$$
$$= L_{\text{NLL}}(\boldsymbol\theta) + \frac{1}{2\sigma_p^2}\|\boldsymbol\theta\|_2^2 + \text{const}$$

<div class="math-box">

$$\boxed{\;L_{\text{MAP}}(\boldsymbol\theta) = L_{\text{NLL}}(\boldsymbol\theta) + \lambda\,\|\boldsymbol\theta\|_2^2,\qquad \lambda := \frac{1}{2\sigma_p^2}\;}$$

</div>

**This is exactly L2 regularization (a.k.a. ridge, weight decay).**

- Strong prior (small $\sigma_p^2$) ⇒ large $\lambda$ ⇒ heavy penalty ⇒ weights pulled hard to zero.
- Weak prior (large $\sigma_p^2$) ⇒ small $\lambda$ ⇒ MAP $\to$ MLE.

---

# Worked numeric · MAP with L2

Linear regression. Suppose at the MLE estimate, the NLL is $L_{\text{NLL}} = 0.50$. Weight vector $\boldsymbol\theta = [3, -4]$ — so $\|\boldsymbol\theta\|_2^2 = 9 + 16 = 25$.

Pick $\lambda = 0.01$ (corresponds to $\sigma_p^2 = 50$ — a fairly weak prior).

<div class="math-box">

$L_{\text{MAP}} = 0.50 + 0.01 \cdot 25 = 0.50 + 0.25 = \mathbf{0.75}$

</div>

The optimizer now pays a price for large weights. Since the gradient of $\|\theta\|^2$ is $2\theta$, every step **shrinks weights by a factor $(1 - 2\eta\lambda)$** before the data update. This is exactly **weight decay** — and you'll see this exact form again in Adam vs AdamW (L5).

---

# Laplace prior · setup

Now choose a prior $p(\theta_j) = \text{Laplace}(0, b)$ — same idea (centred at zero) but **heavier tails and a sharper peak at zero**.

<div class="math-box">

The Laplace density ·
$p(\theta_j) = \dfrac{1}{2b}\exp\!\left(-\dfrac{|\theta_j|}{b}\right)$

For one weight ·
$\log p(\theta_j) = -\dfrac{|\theta_j|}{b} + \text{const}$

For the whole vector with independent components ·
$\log p(\boldsymbol\theta) = -\dfrac{1}{b}\sum_j |\theta_j| + \text{const} = -\dfrac{1}{b}\|\boldsymbol\theta\|_1 + \text{const}$

</div>

Note the **absolute value** in the log — that's the structural difference from Gaussian's squared term.

---

# Laplace prior · L1 pops out

Plug into MAP ·

$$L_{\text{MAP}}(\boldsymbol\theta) = L_{\text{NLL}}(\boldsymbol\theta) + \frac{1}{b}\|\boldsymbol\theta\|_1 + \text{const}$$

<div class="math-box">

$$\boxed{\;L_{\text{MAP}}(\boldsymbol\theta) = L_{\text{NLL}}(\boldsymbol\theta) + \lambda\,\|\boldsymbol\theta\|_1,\qquad \lambda := \frac{1}{b}\;}$$

</div>

**This is exactly L1 regularization (a.k.a. lasso).**

- Same machinery (MAP), different prior, different penalty.
- The Laplace density is "sharply peaked at zero with heavy tails" — and that geometry is what makes L1 produce **sparse solutions**.

---

# Why L1 produces sparse solutions · the geometry

![w:920px](figures/lec00/svg/l1_l2_geometry.svg)

<div class="math-box">

Think of MAP as minimizing the loss subject to the prior. In 2D ·

- **L2 ball** — circle. The data-loss contour can touch it anywhere on the circle. Generic touchpoints have *both* coordinates non-zero. Solutions are **shrunk but rarely zero**.
- **L1 ball** — diamond. Corners stick out **on the axes**. Generic data-loss contours hit the diamond *at a corner*, where one or more coordinates **are exactly zero**. Solutions are **sparse**.

</div>

L1's sparsity is a direct consequence of the **diamond geometry** of the Laplace prior — corners that lie exactly on the coordinate axes are the most likely touchpoints. Move to high dimensions and there are exponentially many corners → many features get zeroed simultaneously.

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

### PART 6

# Where this matters · the rest of the course

NLL is the loss everywhere · KL connects · VAE/diffusion/GANs are MAP++

---

# Every loss is an NLL — the master table

<div class="math-box">

| Output | Distribution | Loss = NLL | Lecture |
|:-:|:-:|:-:|:-:|
| Real-valued | Normal | MSE | L1 (recap), L19 (VAE recon) |
| Binary | Bernoulli | BCE | L1 (recap) |
| K classes | Categorical | Cross-entropy | L7+ (vision), L13–15 (LLMs) |
| Pixels | per-pixel Normal | per-pixel MSE | L19 (VAE), L21 (diffusion) |
| Tokens | Categorical | next-token CE | L13–L15 (LLMs) |
| Image patch given noise | Normal in pixel/score space | MSE on noise | L21 (diffusion) |
| Latent variable model | Normal + KL prior | ELBO = recon + KL | L19 (VAE) |
| Two distributions match | KL minimization | DPO, distillation | L16, L23 |

</div>

The whole course will keep instantiating the same NLL recipe. Each new model just changes **which distribution** is being assumed.

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

This is also why log-likelihood works as a model-quality signal · a model that places high probability on the data has *low* per-sample surprise.

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

In base 2, $H$ is in **bits** · *the average number of bits the optimal code spends per sample from $p$.*

So entropy is what you get when you average the per-sample surprise from the previous slides. Big entropy = on average, draws from $p$ are surprising. Small entropy = mostly predictable.

---

# Entropy as code length · the Huffman lens

Imagine you must send samples from $p$ over a wire using a **prefix-free binary code**. Shannon's source-coding theorem says ·

<div class="math-box">

$$\underbrace{H(p)}_{\text{lower bound}} \;\le\; \mathbb{E}_p[\text{bits per symbol}] \;\le\; H(p) + 1$$

</div>

**Concrete** · alphabet $\{A, B, C, D\}$ with $p = (0.5, 0.25, 0.125, 0.125)$ ·

| Symbol | $p$ | Optimal code | Code length |
|:-:|:-:|:-:|:-:|
| A | $0.5$ | `0` | 1 |
| B | $0.25$ | `10` | 2 |
| C | $0.125$ | `110` | 3 |
| D | $0.125$ | `111` | 3 |

Average length $= 0.5(1) + 0.25(2) + 0.125(3) + 0.125(3) = 1.75$ bits $= H(p)$ exactly.

**Entropy = the cost of describing samples from $p$ optimally.** That's the right physical interpretation.

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

![w:680px](figures/lec00/svg/entropy_bernoulli.svg)

<div class="math-box">

$H(p) = -p \log_2 p - (1 - p) \log_2 (1 - p)$ — concave, symmetric around $p = 0.5$, peaking at $1$ bit.

A *fair* coin needs **one bit** to encode each flip. A coin with $p = 0.99$ is almost deterministic — encoded with arithmetic coding it costs ~$0.08$ bits/flip on average.

</div>

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

# KL divergence · the definition

KL divergence measures how different one distribution $p$ is from another distribution $q$ over the same support.

<div class="math-box">

$$\text{KL}(p \,\Vert\, q) := \mathbb{E}_{Y \sim p}\!\left[\log \frac{p(Y)}{q(Y)}\right] = \sum_y p(y)\,\log\frac{p(y)}{q(y)}$$

(Replace $\sum$ with $\int$ for continuous distributions.)

</div>

**Read** · *"the average log-ratio when the data really comes from $p$ but you used $q$ to model it."*

It's the **average extra surprise** you experience by encoding samples-from-$p$ using a code optimized for $q$.

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

# KL as wasted code length · the Huffman lens

Same alphabet $\{A,B,C,D\}$. **True** $p = (0.5, 0.25, 0.125, 0.125)$ — Huffman code lengths $(1, 2, 3, 3)$ · optimal cost $H(p) = 1.75$ bits.

Suppose we mistakenly built our code for **wrong** $q = (0.125, 0.125, 0.25, 0.5)$ — Huffman lengths $(3, 3, 2, 1)$.

When we encode samples from $p$ with the code for $q$ ·

$$H(p, q) = 0.5(3) + 0.25(3) + 0.125(2) + 0.125(1) = \mathbf{2.625}\text{ bits}$$

<div class="math-box">

$$\text{KL}(p \,\Vert\, q) \;=\; H(p, q) - H(p) \;=\; 2.625 - 1.75 \;=\; \mathbf{0.875}\text{ bits/symbol of waste}$$

</div>

**Reading** · KL is the literal *number of extra bits per sample* you pay for using the wrong code. Modelling = compression · this is why "**better model = better compression**" is not a metaphor.

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

# Cross-entropy = entropy + KL

Algebra ·
$$H(p, q) := -\mathbb{E}_p[\log q(Y)] = -\mathbb{E}_p[\log p(Y)] + \mathbb{E}_p\!\left[\log\tfrac{p(Y)}{q(Y)}\right] = H(p) + \text{KL}(p \| q)$$

<div class="math-box">

$$\boxed{\;H(p, q) \;=\; H(p) \;+\; \text{KL}(p \,\|\, q)\;}$$

</div>

- **Cross-entropy $H(p, q)$** · "expected bits to encode samples from $p$ if we used a code optimized for $q$."
- **Entropy $H(p)$** · "the irreducible cost of encoding $p$." Independent of $q$.
- **KL** · the *extra* bits we waste because $q \ne p$.

For classification with a **one-hot true label** $p$, $H(p) = 0$ — so cross-entropy *is* KL. That's why we say "the classifier loss minimizes KL to the true label."

---

# Cross-entropy worked · the one-hot collapse

In classification, the truth $\mathbf{y}$ is **one-hot** and the model output $\hat{\boldsymbol\pi}$ is a softmax.

<div class="math-box">

For any one-hot, $H(\mathbf{y}) = 0$ — there's no uncertainty in the truth. So the cross-entropy formula collapses ·

$$H(\mathbf{y}, \hat{\boldsymbol\pi}) = \underbrace{H(\mathbf{y})}_{=\,0} + \text{KL}(\mathbf{y} \,\Vert\, \hat{\boldsymbol\pi}) = -\log\hat\pi_{\,y_{\text{true}}}$$

</div>

**Cross-entropy of a one-hot truth against a softmax model is exactly the NLL of the model on the true class.** This is the standard classification loss.

It's also exactly what L1 derived from a Bernoulli/Categorical assumption — same answer through two different lenses (NLL or KL). On the next slide we tabulate it across confidence levels.

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

# MLE through the KL lens · result

Use $H(p, q) = H(p) + \text{KL}(p\|q)$ and drop $H(\hat p_{\text{data}})$ as constant in $\theta$ ·

<div class="math-box">

$$\arg\max_\theta \,\ell(\theta) \;=\; \arg\min_\theta \,H(\hat p_{\text{data}}, p_\theta) \;=\; \arg\min_\theta \,\text{KL}(\hat p_{\text{data}} \,\|\, p_\theta)$$

</div>

**MLE = make the model distribution as KL-close as possible to the empirical distribution.**

Same machinery, two-distribution view. Every classifier you've trained with cross-entropy has been silently minimizing this KL — to the one-hot empirical distribution of class labels.

This is also why MLE is **mode-covering** (forward KL) — covered in the "forward vs reverse KL" slide.

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
| **L2 regularization** | NLL $+ \lambda \|\theta\|^2$ | don't drift from prior |
| **VAE (L19)** | reconstruction $+ \text{KL}(q_\phi(z\|x) \| p(z))$ | encoder posterior close to standard normal |
| **DPO / RLHF (L16)** | reward $- \beta\,\text{KL}(\pi_\theta \| \pi_{\text{ref}})$ | policy close to base model |
| **Distillation (L23)** | $\text{KL}(p_{\text{teacher}} \| p_{\text{student}})$ | student matches teacher |

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

**This is approximately what VAE encoders, GAN training, and policy distillation optimize.** Sharper samples, but mode collapse is a real risk.

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
| Classifier (CE loss) | $\text{KL}(\hat p_{\text{data}} \| p_\theta)$ | L1, every classifier |
| **VAE** | recon-NLL $+ \text{KL}(q_\phi(z\|x) \,\|\, p(z))$ | L19 |
| **Diffusion (variational view)** | $\sum_t \text{KL}(q(x_{t-1}\|x_t, x_0) \,\|\, p_\theta(x_{t-1}\|x_t))$ | L21 |
| **GAN** | (approximately) Jensen-Shannon — symmetrized KL | L20 |
| **DPO / RLHF KL term** | $\text{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ | L16 |
| **Knowledge distillation** | $\text{KL}(p_{\text{teacher}} \,\|\, p_{\text{student}})$ | L23 |

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
