---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# A Probabilistic View of ML
## Distributions · sampling · MLE · MAP · KL

### Lecture 0 · ES 667: Deep Learning · *spans 2 sessions*

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

By the end of these **two sessions** you will be able to:

**Session 1 · framework + likelihood**

1. **State** the Bernoulli, Categorical, and Normal distributions, the `~` notation, and IID sampling.
2. **Read a plate-notation** graphical model and recognize the supervised setup.
3. Explain three reasons why **the Normal shows up everywhere** (CLT, max entropy, closed-under-linear).
4. **Sample** from Bernoulli, Categorical, and Normal — and recognize the **reparameterization trick** as an affine map of a base sample.
5. **Apply Bayes' rule** and identify prior, likelihood, evidence, and posterior.
6. **Derive MSE / BCE / categorical CE** as NLL under Gaussian / Bernoulli / Categorical outputs.

**Session 2 · MAP + KL + course spine**

7. **Derive L2** as MAP with a Gaussian prior, **L1** as MAP with a Laplace prior, and explain L1's sparsity geometrically.
8. State and use **KL divergence**, recognize cross-entropy as KL up to a constant, and re-derive MLE/MAP through the KL lens.
9. Distinguish **forward vs reverse KL** and predict their respective failure modes (mode-covering vs mode-seeking).
10. **Connect** these foundations to VAEs, diffusion, RLHF, distillation, and the rest of the course.

---

<!-- _class: section-divider -->

### PART 0

# Revision · linear & logistic regression

The loss functions we used without asking why

---

# Linear regression · the recap

Given dataset $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$ with $\mathbf{x}_i \in \mathbb{R}^d$ and continuous $y_i \in \mathbb{R}$, fit:

$$\hat y_i = \boldsymbol\theta^\top \mathbf{x}_i$$

(absorb the bias into $\boldsymbol\theta$ by appending 1 to $\mathbf{x}_i$).

<div class="math-box">

**Loss** · mean squared error
$$L(\boldsymbol\theta) = \frac{1}{N}\sum_{i=1}^N (y_i - \boldsymbol\theta^\top \mathbf{x}_i)^2$$

**Train** · solve $\nabla_\theta L = 0$ in closed form ($\hat\theta = (X^\top X)^{-1} X^\top y$) or run gradient descent.

</div>

You probably justified MSE as *"penalize big errors more than small ones."* True — but **why squared and not absolute or cubed?** We'll see today.

---

# Logistic regression · the recap

Same setup, but $y_i \in \{0, 1\}$ (binary classification). Output is a probability via the sigmoid:

$$\hat p_i = \sigma(\boldsymbol\theta^\top \mathbf{x}_i),\qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

<div class="math-box">

**Loss** · binary cross-entropy (BCE)
$$L(\boldsymbol\theta) = -\frac{1}{N}\sum_{i=1}^N \bigl[y_i \log\hat p_i + (1 - y_i)\log(1 - \hat p_i)\bigr]$$

**Train** · gradient descent. No closed form.

</div>

You justified BCE as *"big penalty when confidently wrong."* Again true — but **why this exact form**, not $-(y - \hat p)^2$ or $-|y - \hat p|$? Same question.

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

# Today's question, on one slide

Three loose threads from your last ML course ·

| Object | Recipe | Where does it come from? |
|:-:|:-:|:-:|
| MSE | $\sum (y - \hat y)^2$ | ? |
| BCE | $-\sum [y\log\hat p + (1-y)\log(1-\hat p)]$ | ? |
| L2 | $\lambda \sum \theta_j^2$ | ? |
| L1 | $\lambda \sum |\theta_j|$ | ? |

<div class="keypoint">

By the end of this lecture, all four are **derived consequences** of two ideas:

1. The model defines a probability distribution over $y$.
2. We pick parameters that make the data most likely (MLE), optionally tempered by a prior on $\theta$ (MAP).

</div>

---

<!-- _class: section-divider -->

### PART 1

# A probabilistic view of ML

The model doesn't predict a number — it predicts a **distribution**

---

# Random variable · the notation we'll use

A **random variable** $Y$ is a quantity whose value is uncertain. It follows a distribution $p(\cdot)$ that we read as "$Y$ is **distributed as** $p$":

<div class="math-box">

$$Y \sim p(\cdot \mid \theta)$$

The "$\sim$" is the central piece of notation in this lecture. Read it as **"distributed as"**.

- **Discrete** $Y \in \{y_1, y_2, \ldots\}$ — described by a probability **mass** function $P(Y = y \mid \theta)$ summing to 1.
- **Continuous** $Y \in \mathbb{R}$ — described by a probability **density** $p(y \mid \theta)$ integrating to 1.

</div>

**IID assumption** · the dataset $\mathcal{D} = \{Y_1, \ldots, Y_N\}$ consists of *independent and identically distributed* samples · $Y_i \stackrel{\text{iid}}{\sim} p(\cdot \mid \theta)$. Every $Y_i$ comes from the same distribution and knowing $Y_i$ tells you nothing about $Y_j$.

This is the assumption that lets us write $P(\mathcal{D} \mid \theta) = \prod_i P(Y_i \mid \theta)$ — and turn likelihood into a product.

---

# Plate notation · the picture for IID

Throughout this course, generative models are drawn as **directed graphical models** with **plate notation**. Two conventions ·

<div class="math-box">

| Symbol | Meaning |
|:-:|:-:|
| ○ | a random variable (uncertain) |
| ● (filled) | an **observed** random variable (we see its value) |
| arrow $A \to B$ | $A$ generates $B$ (i.e. $B$ depends on $A$) |
| rectangle (plate) labelled $i = 1\ldots N$ | the contents are repeated $N$ times — independence across $i$ |

</div>

A single Bernoulli observation · $p \to \bullet\,Y$.

A dataset of $N$ IID Bernoulli observations ·

![w:380px](figures/lec00/svg/plate_iid_bernoulli.svg)

The plate says *"draw a fresh $Y_i$ for each $i$, all from the same Bernoulli($p$)."*

---

# Bernoulli · the coin (formal)

Outcome $Y \in \{0, 1\}$, parameter $p \in [0, 1]$ = probability of "heads."

$$Y \sim \text{Bernoulli}(p)$$

<div class="math-box">

**Probability mass function (PMF)** ·
$$P(Y = 1 \mid p) = p,\qquad P(Y = 0 \mid p) = 1 - p$$

**Compact form** · $P(Y = y \mid p) = p^y\,(1 - p)^{1 - y}$
- $y = 1 \Rightarrow p^1 (1-p)^0 = p$ ✓
- $y = 0 \Rightarrow p^0 (1-p)^1 = 1 - p$ ✓

**Mean & variance** · $\mathbb{E}[Y] = p,\quad \text{Var}[Y] = p(1 - p)$

</div>

**Worked** · Coin with $p = 0.7$, three flips $H, T, H$ (i.e. $Y_1, Y_2, Y_3 = 1, 0, 1$). With IID assumption ·
$P(\mathcal{D} \mid p) = 0.7 \cdot 0.3 \cdot 0.7 = 0.147$

The **product over independent observations** is the heart of likelihood.

---

# Categorical · the K-sided die (formal)

Outcome $Y \in \{1, 2, \ldots, K\}$, parameter vector $\boldsymbol\pi = (\pi_1, \ldots, \pi_K)$ with $\pi_k \ge 0$ and $\sum_k \pi_k = 1$.

$$Y \sim \text{Categorical}(\boldsymbol\pi)$$

<div class="math-box">

**PMF** · $P(Y = k \mid \boldsymbol\pi) = \pi_k$

**One-hot compact form** · with $\mathbf{y} \in \{0,1\}^K$ s.t. $y_k = 1$ if $Y = k$ else 0,
$$P(Y \mid \boldsymbol\pi) = \prod_{k=1}^K \pi_k^{\,y_k}$$

**Mean** · $\mathbb{E}[\mathbf{y}] = \boldsymbol\pi$. The Bernoulli is the special case $K = 2$.

</div>

**Worked** · MNIST classifier outputs $\boldsymbol\pi = (0.05, 0.7, 0.1, \ldots, 0.05)$ for one image. The probability of class 2 (digit "2") is $\pi_2 = 0.7$. If the true label is $Y = 2$, the model assigned probability **0.7** to the truth.

The softmax output of *any* classifier IS a Categorical distribution.

---

# Normal (Gaussian) · the bell curve (formal)

Continuous $Y \in \mathbb{R}$ with mean $\mu$ and variance $\sigma^2$.

$$Y \sim \mathcal{N}(\mu, \sigma^2)$$

<div class="math-box">

**Probability density (PDF)** ·
$$p(y \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\,\exp\!\left(-\frac{(y - \mu)^2}{2\sigma^2}\right)$$

**Mean & variance** · $\mathbb{E}[Y] = \mu,\quad \text{Var}[Y] = \sigma^2$

**Three properties to memorize** ·
1. Centred at $\mu$, spread controlled by $\sigma$.
2. Density falls off **exponentially in the squared distance** $(y - \mu)^2$.
3. The squared exponent will be the seed of MSE.

</div>

**Worked** · House prices $\sim \mathcal{N}(50, 5^2)$ lakh. $p(50) \approx 0.0798$, $p(70) \approx 2.7 \times 10^{-5}$ — a 4σ-away house is vanishingly unlikely under this model.

---

# Multivariate Normal · briefly

For $\mathbf{Y} \in \mathbb{R}^d$ ·

$$\mathbf{Y} \sim \mathcal{N}(\boldsymbol\mu, \Sigma)$$

<div class="math-box">

$$p(\mathbf{y} \mid \boldsymbol\mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^d \,|\Sigma|}}\,\exp\!\left(-\tfrac{1}{2}(\mathbf{y} - \boldsymbol\mu)^\top \Sigma^{-1} (\mathbf{y} - \boldsymbol\mu)\right)$$

- $\boldsymbol\mu \in \mathbb{R}^d$ — mean vector.
- $\Sigma \in \mathbb{R}^{d \times d}$ — covariance matrix (symmetric, positive definite).

</div>

If $\Sigma = \sigma^2 I$ (isotropic), the components are independent. **This is the case in diffusion (L21)** — every noise step samples isotropic Gaussian noise. We'll come back to this when we get there.

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

**Statement (informal)** · let $X_1, X_2, \ldots, X_N$ be IID with mean $\mu$ and variance $\sigma^2$. Define the standardized sum

$$Z_N = \frac{1}{\sqrt{N}}\sum_{i=1}^N \frac{X_i - \mu}{\sigma}$$

Then as $N \to \infty$, $Z_N \to \mathcal{N}(0, 1)$ regardless of the original distribution of $X_i$.

<div class="math-box">

**Worked** · Sum of 12 IID $\text{Uniform}[0, 1]$ has mean $6$ and variance $1$. The sum is approximately $\mathcal{N}(6, 1)$ — Box-Muller's original observation, used historically to *generate* Gaussian samples before better algorithms existed.

</div>

**Implication** · any quantity arising as the **aggregate of many small effects** tends to look Gaussian. Sensor noise, height of humans, daily temperature deviation — all approximately Normal because the underlying causes are sums of many small contributions.

---

# Why Normal · maximum entropy

**Entropy** quantifies "how spread-out / uncertain" a distribution is. Among all distributions with a *given* mean $\mu$ and variance $\sigma^2$ ·

<div class="math-box">

$$\boxed{\;\arg\max_{p}\; H(p) \;\text{ s.t. }\; \mathbb{E}_p[Y] = \mu,\; \mathbb{E}_p[(Y - \mu)^2] = \sigma^2 \;\;\Longrightarrow\;\; p = \mathcal{N}(\mu, \sigma^2)\;}$$

</div>

**Reading** · "if all you know about a quantity is its first two moments, the *least committal* probability model is Gaussian."

This is **Occam's razor for distributions** · don't bake in assumptions you can't justify. The Gaussian is the answer to *"what's the safest default?"* given mean and variance.

Other max-entropy distributions you'll meet · **Bernoulli** is max-ent on $\{0, 1\}$ given the mean; **Uniform** is max-ent on a bounded interval; **Exponential** is max-ent on $[0, \infty)$ given a fixed mean.

---

# Why Normal · closed under linear ops + conjugacy

**Sum of Gaussians is Gaussian** · if $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$ are independent,
$$X + Y \sim \mathcal{N}\bigl(\mu_1 + \mu_2,\; \sigma_1^2 + \sigma_2^2\bigr)$$

**Affine transform of a Gaussian is Gaussian** · $aY + b \sim \mathcal{N}(a\mu + b, a^2 \sigma^2)$.

<div class="math-box">

**Conjugacy.** Gaussian prior + Gaussian likelihood ⇒ **Gaussian posterior**, with closed-form mean and variance. No integral, no MCMC.

</div>

These properties matter:

- **Diffusion (L21)** · the forward process adds Gaussian noise at every step, and the closed-form jump $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ exists *exactly because* sums of Gaussians are Gaussian.
- **Kalman filter** · linear-Gaussian state-space models have closed-form posterior at every time step.
- **Reparameterization trick (next!)** · $z = \mu + \sigma \cdot \epsilon$, $\epsilon \sim \mathcal{N}(0, 1)$ — sampling from a non-standard Normal is just an affine transform of a standard one.

---

# A small zoo of distributions you'll meet

| Name | Notation | Type | Used for |
|:-:|:-:|:-:|:-:|
| Bernoulli | $\text{Bernoulli}(p)$ | discrete, binary | binary classification |
| Categorical | $\text{Categorical}(\boldsymbol\pi)$ | discrete, $K$-way | multiclass classification |
| Normal | $\mathcal{N}(\mu, \sigma^2)$ | continuous | regression, Gaussian noise |
| Laplace | $\text{Laplace}(\mu, b)$ | continuous, heavy-tail | L1 regularizer prior |
| Beta | $\text{Beta}(\alpha, \beta)$ | continuous on $[0,1]$ | prior over a probability |
| Multinomial | $\text{Multinomial}(n, \boldsymbol\pi)$ | discrete | counts in $n$ trials |

You'll need the first three today. **Laplace** comes back when we derive L1; **Beta** when we add a prior to the coin.

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

# The master sampling primitive · inverse CDF

For any 1-D distribution with CDF $F(y) = P(Y \le y)$, the algorithm is ·

<div class="math-box">

1. Draw $u \sim \text{Uniform}[0, 1]$.
2. Return $y = F^{-1}(u)$.

</div>

**Why it works** · $P(Y \le y) = P(F^{-1}(u) \le y) = P(u \le F(y)) = F(y)$. ✓

**Worked · sampling from $\text{Bernoulli}(p)$** · the CDF jumps from $0$ to $1-p$ at $0$ and from $1-p$ to $1$ at $1$. Inverting ·
- if $u < 1 - p$ return $0$
- else return $1$

Equivalently · `return 1 if u < p else 0` (after a relabeling). One uniform draw, one comparison · that's the entire Bernoulli sampler.

---

# Sampling from a Categorical · the LLM token primitive

Categorical with probabilities $\boldsymbol\pi = (\pi_1, \ldots, \pi_K)$. The CDF is the **stick-breaking** cumulative ·

<div class="math-box">

Build $c_k = \sum_{j=1}^k \pi_j$ (so $c_K = 1$). Draw $u \sim \text{Uniform}[0, 1]$. Return the smallest $k$ such that $c_k \ge u$.

</div>

**Worked** · $\boldsymbol\pi = (0.1, 0.4, 0.3, 0.2)$ → cumulative $(0.1, 0.5, 0.8, 1.0)$. Draw $u = 0.55$ · smallest $k$ with $c_k \ge 0.55$ is $k = 3$. Return class 3.

This **is what `torch.multinomial` does**. And — bigger picture — **this is how every LLM samples its next token** · the model produces a vector of $K$ probabilities (where $K \approx$ vocab size, often 50,000+), and we run exactly this algorithm to pick a token.

---

# Sampling from a Normal · and the reparameterization trick

To sample $Y \sim \mathcal{N}(\mu, \sigma^2)$ ·

1. Sample $\epsilon \sim \mathcal{N}(0, 1)$ (e.g. via Box-Muller, or just `torch.randn(...)`).
2. Return $y = \mu + \sigma \cdot \epsilon$.

This *works* because the affine transform of a Gaussian is a Gaussian (last lecture's "closed under linear ops").

<div class="keypoint">

This is the **reparameterization trick**, and it is one of the most important ideas in modern deep learning ·

$$y = \mu_\theta(\mathbf{x}) + \sigma_\theta(\mathbf{x}) \cdot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, 1)$$

The randomness $\epsilon$ is **outside** the path we want gradients through. $\mu_\theta$ and $\sigma_\theta$ are deterministic functions of the input. So we can backprop through the sample.

</div>

This is what makes **VAEs trainable** (L19) and what powers the entire **diffusion** stack (L21–L22). You'll see it again, and you now know exactly where it comes from.

---

# Monte Carlo expectation · the workhorse

For an integral / sum we can't compute analytically ·

<div class="math-box">

$$\mathbb{E}_{Y \sim p}[f(Y)] = \int f(y)\,p(y)\,dy \;\;\approx\;\; \frac{1}{N}\sum_{i=1}^N f(y_i),\qquad y_i \sim p$$

</div>

**Law of large numbers** · the estimator is unbiased and its variance scales as $1/N$ — quadrupling samples halves the standard error.

**Hidden everywhere in DL** ·
- **Mini-batch SGD** is a Monte Carlo estimate of the *expected gradient* over the data distribution. Each batch is one MC sample.
- **VAE ELBO** uses $\mathbb{E}_{q(z\mid x)}[\log p(x\mid z)]$ — estimated from one $z$ sample per training step.
- **Diffusion** averages over a randomly chosen timestep $t$ and noise $\epsilon$ — both Monte Carlo.
- **REINFORCE / RLHF** averages over policy rollouts.

Almost every "loss" you'll write is secretly an expectation that's being approximated by a single sample.

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

# Why we take logs · IID + numerical reality

For $N$ IID observations · $\mathcal{L}(\theta) = \prod_{i=1}^N P(y_i \mid \theta)$.

Two problems with the product ·
1. **Underflow** · with $N = 1000$ and each factor $\sim 0.5$, we get $0.5^{1000} \approx 10^{-301}$ — below floating-point precision.
2. **Hard to differentiate** · product rule generates $N$ terms.

<div class="math-box">

**Fix · take the log.** Products become sums. Maxima are preserved (log is monotonic).

$$\ell(\theta) := \log \mathcal{L}(\theta) = \sum_{i=1}^N \log P(y_i \mid \theta)$$

</div>

We will *always* work with log-likelihood. By convention we minimize the **negative** log-likelihood (NLL) so that "loss" is something we drive down.

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

# Bayes worked example · disease test

A disease has prevalence 1%. A test has sensitivity 95% (true positive rate) and specificity 95% (true negative rate). You test positive. How likely are you to have the disease?

<div class="math-box">

Let $D$ = "have disease", $+$ = "test positive".
- Prior · $P(D) = 0.01$, $P(\bar D) = 0.99$
- Likelihood · $P(+ \mid D) = 0.95$, $P(+ \mid \bar D) = 0.05$
- Total · $P(+) = 0.95 \cdot 0.01 + 0.05 \cdot 0.99 = 0.0095 + 0.0495 = 0.059$

$$P(D \mid +) = \frac{0.95 \cdot 0.01}{0.059} \approx 0.16$$

</div>

Despite a "95% accurate" test, a positive result gives only **16%** chance of disease. This is the **base-rate fallacy**, and it's the same maths we'll apply to ML.

---

# Bayes for ML · flip onto $\theta$

In ML, $A$ becomes the parameter $\theta$ and $B$ becomes the data $\mathcal{D}$.

<div class="math-box">

$$\boxed{\,p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\,p(\theta)}{p(\mathcal{D})}\,}$$

</div>

This is the **central equation of probabilistic ML**. It tells us how to update our belief about $\theta$ after seeing data $\mathcal{D}$. Each term has a name and a role.

---

# The four terms · names and roles

<div class="math-box">

$$\underbrace{p(\theta \mid \mathcal{D})}_{\text{posterior}} \;=\; \frac{\overbrace{p(\mathcal{D} \mid \theta)}^{\text{likelihood}}\;\,\overbrace{p(\theta)}^{\text{prior}}}{\underbrace{p(\mathcal{D})}_{\text{evidence}}}$$

</div>

| Term | What it is | Where it comes from |
|:-:|:-:|:-:|
| **Likelihood** | how plausible is the data under $\theta$ | the model |
| **Prior** | belief about $\theta$ before seeing data | choice / domain knowledge |
| **Posterior** | updated belief about $\theta$ after data | what we compute |
| **Evidence** | $\int p(\mathcal{D} \mid \theta)\,p(\theta)\,d\theta$ — a normalizer | usually intractable, often ignored |

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

# Two estimators that fall out of Bayes

<div class="columns">
<div>

### **MLE**

Ignore the prior. Just maximize the likelihood ·

$$\hat\theta_{\text{MLE}} = \arg\max_\theta\, p(\mathcal{D} \mid \theta)$$

"What value of $\theta$ best explains the data?"

</div>
<div>

### **MAP**

Use the prior. Maximize the posterior ·

$$\hat\theta_{\text{MAP}} = \arg\max_\theta\, p(\mathcal{D} \mid \theta)\,p(\theta)$$

"Given my prior belief and the data, what's the most probable $\theta$?"

</div>
</div>

<div class="keypoint">

**MAP = MLE + a prior on $\theta$.** That single sentence is what makes regularization fall out of the same machinery as the loss.

</div>

---

<!-- _class: section-divider -->

### PART 4

# Maximum Likelihood Estimation

Concrete derivations · coin · linear regression · logistic regression

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

# Closed-form OLS · for completeness

The MSE objective is quadratic in $\boldsymbol\theta$, so we can set the gradient to zero analytically.

Stack data · $X \in \mathbb{R}^{N \times d}$, $\mathbf{y} \in \mathbb{R}^N$. Then $\sum (y_i - \boldsymbol\theta^\top \mathbf{x}_i)^2 = \|\mathbf{y} - X\boldsymbol\theta\|^2$.

<div class="math-box">

$$\nabla_\theta \|\mathbf{y} - X\boldsymbol\theta\|^2 = -2 X^\top (\mathbf{y} - X\boldsymbol\theta) = 0$$

$$\boxed{\;\hat{\boldsymbol\theta}_{\text{MLE}} = (X^\top X)^{-1} X^\top \mathbf{y}\;}$$

</div>

The familiar **normal equation** is the closed-form MLE under Gaussian noise. SGD / gradient descent gives the same answer iteratively.

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

# End of Session 1

So far · **probabilistic framework + MLE.** Bernoulli, Categorical, Normal · likelihood and log-likelihood · sampling primitives · Bayes' rule with its four named terms · MLE for the coin, linear regression, logistic regression, and multiclass — all derived as NLL of a chosen distribution.

**Session 2 picks up from here** · MAP and regularization (L2 from Gaussian prior, L1 from Laplace), KL divergence as the unifying lens, and how it all fans out into VAEs, diffusion, RLHF, and the rest of the course.

---

<!-- _class: section-divider -->

# Welcome to Session 2

Recap of session 1 · the model defines a distribution, and MLE = NLL minimization. **Today** · what changes when we add a *prior* — and why KL divergence is the right language for everything that comes next.

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

# Why L1 produces sparse solutions

![w:880px](figures/lec00/svg/l1_l2_geometry.svg)

<div class="math-box">

Think of MAP as minimizing the loss subject to the prior. In 2D ·

- **L2 ball** — circle. The data-loss contour can touch it anywhere on the circle. Generic touchpoints have *both* coordinates non-zero. Solutions are **shrunk but rarely zero**.
- **L1 ball** — diamond. Corners stick out **on the axes**. Generic data-loss contours hit the diamond *at a corner*, where one or more coordinates **are exactly zero**. Solutions are **sparse**.

</div>

L1's sparsity isn't a magic property — it's a direct consequence of the diamond geometry of the Laplace prior.

---

# L1 vs L2 · summary table

| | **L2 (Ridge)** | **L1 (Lasso)** |
|---|---|---|
| Prior | $\mathcal{N}(0, \sigma_p^2)$ | $\text{Laplace}(0, b)$ |
| Log-prior penalty | $\lambda \sum \theta_j^2$ | $\lambda \sum |\theta_j|$ |
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

# KL divergence · definition

KL divergence measures how different one distribution $p$ is from another distribution $q$ over the same support.

<div class="math-box">

$$\text{KL}(p \,\Vert\, q) := \mathbb{E}_{Y \sim p}\!\left[\log \frac{p(Y)}{q(Y)}\right] = \sum_y p(y)\,\log\frac{p(y)}{q(y)}$$

(Replace $\sum$ with $\int$ for continuous distributions.)

</div>

**Read** · *"the average log-ratio when the data really comes from $p$ but you used $q$ to model it."*

**Three properties** ·
1. $\text{KL}(p \| q) \ge 0$ for all $p, q$ (Gibbs' inequality).
2. $\text{KL}(p \| q) = 0$ if and only if $p = q$ everywhere.
3. **Asymmetric** · $\text{KL}(p \| q) \ne \text{KL}(q \| p)$ in general. This asymmetry will matter — see two slides on.

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

# MLE through the KL lens

Define the **empirical data distribution** $\hat p_{\text{data}}(y) = \frac{1}{N}\sum_{i=1}^N \mathbb{1}[Y_i = y]$ — a histogram of the training set, treated as a distribution.

Then ·

<div class="math-box">

$$\frac{1}{N}\,\ell(\theta) = \frac{1}{N}\sum_{i=1}^N \log p_\theta(y_i) = \mathbb{E}_{Y \sim \hat p_{\text{data}}}[\log p_\theta(Y)] = -H(\hat p_{\text{data}}, p_\theta)$$

So MLE ·
$$\arg\max_\theta \,\ell(\theta) \;=\; \arg\min_\theta \,H(\hat p_{\text{data}}, p_\theta) \;=\; \arg\min_\theta \,\text{KL}(\hat p_{\text{data}} \,\|\, p_\theta)$$

(The last step uses $H(p, q) = H(p) + \text{KL}(p\|q)$ and drops $H(\hat p_{\text{data}})$ as constant in $\theta$.)

</div>

**MLE = make the model distribution as KL-close as possible to the empirical distribution.** Same machinery, two-distribution view.

---

# MAP through the KL lens

Adding the log-prior to the previous slide ·

<div class="math-box">

$$\hat\theta_{\text{MAP}} = \arg\min_\theta\,\Bigl[\,\underbrace{\text{KL}(\hat p_{\text{data}} \,\|\, p_\theta)}_{\text{fit}}\; - \;\underbrace{\tfrac{1}{N}\log p(\theta)}_{\text{prior penalty}}\,\Bigr]$$

</div>

So **L2 regularization** is "minimize KL-to-empirical, plus pay $\lambda \|\theta\|^2$ for straying from a Gaussian prior."

This same KL-plus-regularizer pattern appears throughout DL ·
- **VAE** (L19) · ELBO has explicit $-\text{KL}(q(z|x) \| p(z))$ pulling the posterior toward a standard-normal prior.
- **DPO** (L16) · explicit $\beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$ keeping the fine-tuned LM close to its starting point.
- **Distillation** (L23) · student's distribution should be KL-close to teacher's.

Every regularizer in modern DL is some form of "**don't drift too far in KL**."

---

# Forward vs reverse KL · two faces of the same number

KL is asymmetric — and the two directions give very different optimization problems.

<div class="columns">
<div>

### Forward KL · $\text{KL}(p \,\|\, q_\theta)$

- "Average over $p$, evaluate at $q_\theta$."
- Penalty whenever $p(y) > 0$ but $q_\theta(y) \approx 0$.
- ⇒ **mode-covering** · $q_\theta$ must put mass *everywhere* $p$ does.
- **MLE = forward KL** to empirical $p$.

</div>
<div>

### Reverse KL · $\text{KL}(q_\theta \,\|\, p)$

- "Average over $q_\theta$, evaluate at $p$."
- Penalty whenever $q_\theta(y) > 0$ but $p(y) \approx 0$.
- ⇒ **mode-seeking** · $q_\theta$ concentrates on a *single* high-density region.
- **VAE encoder, GAN-like methods** approximate this.

</div>
</div>

<div class="keypoint">

**Picture · bimodal target, single Gaussian fit.**

- Forward KL · the Gaussian *spreads across both modes* — blurry but covers all data. Why VAE samples are blurry.
- Reverse KL · the Gaussian *picks one mode* — sharp but ignores the other. Why GAN samples are sharp but mode-collapse.

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

# Notebook teaser · MLE & MAP in PyTorch

We will pair this lecture with a notebook (`lec00-mle-map.ipynb`) that walks through ·

<div class="math-box">

1. **MLE for a coin** with `torch.distributions.Bernoulli`.
2. **MLE for linear regression** with `torch.distributions.Normal` — recover OLS.
3. **MLE for logistic regression** with `BCEWithLogitsLoss` — show it equals NLL of Bernoulli.
4. **MAP for linear regression with Gaussian prior** — recover ridge regression.
5. **MAP for linear regression with Laplace prior** — recover lasso, see sparsity emerge as $\lambda$ grows.
6. **Visualize** likelihood and posterior surfaces in 2D for a tiny example.

</div>

Same code skeleton, three lines change between MLE and MAP. That's the punchline.

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

# Lecture 0 — summary

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
