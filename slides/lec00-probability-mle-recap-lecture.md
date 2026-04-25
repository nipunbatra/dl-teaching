---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Probability & MLE · the loss-functions-from-first-principles primer

## Lecture 0 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Why this lecture

The whole course assumes · MSE for regression · cross-entropy for classification · KL for VAEs.

But **why** these losses? Where do they come from?

<div class="paper">

This lecture derives every loss we'll ever use from one principle: **maximum likelihood estimation** under a probabilistic model. Once you see the pattern, every loss in the course becomes obvious.

</div>

If you took a probability course (ES 654), this is a refresher. If not, this is your first-principles foundation.

---

# Learning outcomes

By the end of this lecture you will be able to:

1. State the **Bernoulli, Categorical, and Normal** distributions and their parameters.
2. Compute **likelihood** and **log-likelihood** of a dataset under a model.
3. Derive **MSE** as the MLE for regression with Gaussian noise.
4. Derive **cross-entropy** as the MLE for classification with categorical / Bernoulli output.
5. Articulate why we always **maximize log-likelihood** rather than likelihood directly.
6. Connect MLE to **negative-log-loss** — the loss we'll use everywhere.

---

<!-- _class: section-divider -->

### PART 1

# Three distributions you must know

Bernoulli · Categorical · Normal

---

# Random variable · in 30 seconds

A **random variable** $Y$ is a quantity whose value is uncertain · it follows a distribution.

<div class="math-box">

- **Discrete** · finite or countable values · spam/ham, digit class, dice roll. Described by a probability **mass** function $P(Y = y)$.
- **Continuous** · real-valued · height, weight, sensor reading. Described by a probability **density** function $p(y)$.

For continuous · $\int p(y) \, dy = 1$, but $p(y_0)$ at a single point can exceed 1 (it's a density, not a probability).

</div>

We'll abuse notation and write $p(y)$ for both. Context tells you which.

---

# The three distributions · in pictures

![w:920px](figures/lec00/svg/three_distributions.svg)

---

# Three properties of any distribution

<div class="math-box">

| Property | Definition | What it tells you |
|:-:|:-:|:-:|
| **Mean** | $\mathbb{E}[Y] = \int y \, p(y) \, dy$ | center of mass |
| **Variance** | $\mathbb{E}[(Y - \mu)^2]$ | how spread out |
| **Mode** | $\arg\max_y p(y)$ | most likely value |

</div>

**Examples** for a Normal $\mathcal{N}(5, 2^2)$ · mean = 5 · variance = 4 · mode = 5 (Gaussians are unimodal).

For a Bernoulli with $p = 0.7$ · mean = 0.7 · variance = $p(1-p) = 0.21$ · mode = 1.

---

# Bernoulli · the coin

A single binary outcome ($Y \in \{0, 1\}$) parameterized by $p \in [0,1]$ · the probability of "heads."

<div class="math-box">

$$P(Y = 1) = p, \quad P(Y = 0) = 1 - p$$

Compactly · $P(Y = y) = p^y (1-p)^{1-y}$

</div>

**Examples** · email is spam (Y=1) or not (Y=0) · patient has disease or not · pixel is foreground or background.

---

# Bernoulli · worked numeric

A coin · $p = 0.7$.

<div class="math-box">

$P(Y = 1) = 0.7$ · "heads" · 70% chance.

$P(Y = 0) = 1 - 0.7 = 0.3$ · "tails" · 30% chance.

For 3 independent flips, what's $P(Y_1 = 1, Y_2 = 0, Y_3 = 1)$?

$P = 0.7 \cdot 0.3 \cdot 0.7 = 0.147$

</div>

This **product of per-sample probabilities** is the key idea behind likelihood · we'll use it all the time.

---

# Categorical · the K-sided die

Generalization of Bernoulli to $K$ classes. Outcome $Y \in \{1, \ldots, K\}$, parameters $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_K)$ with $\sum \pi_k = 1$.

<div class="math-box">

$$P(Y = k) = \pi_k$$

Compactly · $P(Y) = \prod_{k=1}^K \pi_k^{[Y = k]}$

</div>

**Examples** · 10 digit classes for MNIST · 1000 ImageNet classes · 50,000 token vocabulary.

The softmax output of any classifier IS a categorical distribution over classes.

---

# Categorical · worked numeric

Image classifier · 3 classes · model outputs $\boldsymbol{\pi} = (0.1, 0.7, 0.2)$ for one input.

<div class="math-box">

The model says · $P(\text{class 1}) = 0.1$ · $P(\text{class 2}) = 0.7$ · $P(\text{class 3}) = 0.2$.

If the **true class** is class 2, then the probability the model assigned to the truth is **0.7**.

</div>

A perfect model would put all mass on class 2 (i.e., $\boldsymbol{\pi} = (0, 1, 0)$). The further the model's prediction is from a one-hot truth, the less likely the data is under it.

---

# Normal (Gaussian) · the bell curve

Continuous outcome $Y \in \mathbb{R}$ parameterized by mean $\mu$ and variance $\sigma^2$.

<div class="math-box">

$$p(Y = y) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\!\left(-\frac{(y - \mu)^2}{2 \sigma^2}\right)$$

</div>

**Examples** · house prices · sensor readings · activations after batch-norm · noise added to images in diffusion (L21).

The bell shape · centered at $\mu$ · spread by $\sigma$ · falls off exponentially in $(y - \mu)^2$.

---

# Multivariate normal · the d-dimensional bell

For $\mathbf{y} \in \mathbb{R}^d$ ·

<div class="math-box">

$$p(\mathbf{y}) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} \exp\!\left(-\frac{1}{2} (\mathbf{y} - \boldsymbol\mu)^\top \Sigma^{-1} (\mathbf{y} - \boldsymbol\mu)\right)$$

</div>

- $\boldsymbol\mu \in \mathbb{R}^d$ · mean vector
- $\Sigma \in \mathbb{R}^{d \times d}$ · covariance matrix (symmetric, positive definite)

If $\Sigma = \sigma^2 I$ (isotropic) · independent Gaussian per coordinate. This is what diffusion uses · "add isotropic Gaussian noise."

The exponent $(y - \mu)^\top \Sigma^{-1} (y - \mu)$ is the **Mahalanobis distance** · squared distance, scaled by inverse covariance.

---

# A few more distributions you'll meet

<div class="math-box">

| Name | Type | Used for |
|:-:|:-:|:-:|
| **Beta(α, β)** | continuous, [0,1] | prior over probabilities · A/B testing |
| **Poisson(λ)** | discrete, ≥0 | event counts (clicks, bus arrivals) |
| **Dirichlet(α)** | continuous, simplex | prior over categorical parameters |
| **Exponential(λ)** | continuous, ≥0 | waiting times |
| **Multinomial(n, π)** | discrete | "how many heads in n flips" |

</div>

You'll mostly use **Bernoulli, Categorical, Normal**. The others appear in specific recipes (e.g., Beta in Bayesian methods, Poisson for count regression).

---

# Normal · worked numeric

Suppose the true house-price model is $\mu = 50$ lakh, $\sigma = 5$ lakh.

<div class="math-box">

$p(Y = 50) = \frac{1}{\sqrt{2\pi \cdot 25}} \exp(0) \approx 0.0798$

$p(Y = 55) = \frac{1}{\sqrt{2\pi \cdot 25}} \exp(-25/50) \approx 0.0484$

$p(Y = 70) = \frac{1}{\sqrt{2\pi \cdot 25}} \exp(-400/50) \approx 2.7 \times 10^{-5}$

</div>

A house at the mean ($Y = 50$) is the most likely. A house at $Y = 70$ (4σ away) is **vanishingly unlikely** under this model.

This squared-distance penalty — $(y - \mu)^2$ — is the seed of MSE.

---

<!-- _class: section-divider -->

### PART 2

# Conditional probability and sampling

The plumbing that makes everything work

---

# Conditional probability · the basics

<div class="math-box">

$$P(A \mid B) = \frac{P(A, B)}{P(B)}$$

Given $B$ happened, what's the probability $A$ also happened?

</div>

**Example** · $P(\text{disease} \mid \text{positive test}) = ?$
- $P(\text{disease}) = 0.01$ (prior)
- $P(\text{positive} \mid \text{disease}) = 0.95$ (test sensitivity)
- $P(\text{positive} \mid \text{healthy}) = 0.05$ (false positive rate)

Combine with **Bayes' rule** (next slide) to flip the conditional.

---

# Bayes' rule · 90% of probability

<div class="math-box">

$$P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}$$

</div>

For the disease test ·

$P(\text{disease} \mid +) = \frac{0.95 \cdot 0.01}{0.95 \cdot 0.01 + 0.05 \cdot 0.99} \approx 0.16$

<div class="warning">

**Counterintuitive** · even with 95% sensitivity and 95% specificity, a positive test only gives 16% disease probability when the prior is 1%. The base-rate fallacy. This is the entire reason "false positive paradoxes" exist in medical testing and ML systems.

</div>

---

# Expectation and the law of large numbers

<div class="math-box">

$$\mathbb{E}_{Y \sim p}[f(Y)] = \int f(y) \, p(y) \, dy$$

We can't usually compute this integral analytically · so we **sample**:

$$\mathbb{E}_p[f(Y)] \approx \frac{1}{N} \sum_{i=1}^N f(y_i), \quad y_i \sim p$$

</div>

This is **Monte Carlo estimation** · law of large numbers says the average converges to the true expectation as $N \to \infty$.

VAE training (L19) · expectation over $z \sim q(z|x)$ · estimated by 1 sample. Diffusion (L21) · expectation over noise · estimated by 1 sample. Almost every loss in DL is a sample-mean approximation.

---

# Sampling · how do we draw from a distribution?

<div class="math-box">

| Distribution | Sampling recipe |
|:-:|:-:|
| Uniform[0,1] | most languages have `rand()` built in |
| Bernoulli($p$) | $u \sim U[0,1]$, return 1 if $u < p$ else 0 |
| Categorical($\boldsymbol\pi$) | inverse-CDF · pick the smallest $k$ s.t. $\sum_{j \le k} \pi_j > u$ |
| Normal $\mathcal{N}(0, 1)$ | Box-Muller · `randn()` in NumPy / PyTorch |
| $\mathcal{N}(\mu, \sigma^2)$ | $z \sim \mathcal{N}(0,1)$, return $\mu + \sigma z$ |

</div>

The last row is the **reparameterization trick** (L19) · sample from a simple base distribution, then transform.

---

# Worked example · sampling categorical

Probabilities · $\boldsymbol\pi = (0.1, 0.3, 0.4, 0.2)$. Cumulative · $(0.1, 0.4, 0.8, 1.0)$.

<div class="math-box">

Draw $u \sim U[0, 1]$. Suppose $u = 0.55$.

- 0.55 > 0.1 · skip class 1
- 0.55 > 0.4 · skip class 2
- 0.55 ≤ 0.8 · **return class 3** ✓

</div>

That's all "sampling a softmax output" is doing internally · convert to cumulative, draw one uniform, find where it lands. Fast.

---

# Independent samples vs IID

<div class="math-box">

- **Independent** · knowing one sample tells you nothing about the next.
- **Identically distributed** · all samples come from the same distribution.

**IID** · both. The standard assumption for batches of training data.

</div>

When IID **fails** · time-series (today's stock depends on yesterday's), images from a single video, examples that share a common cause. Be careful · IID assumption is what lets us sum log-likelihoods. Violations need different math (autoregressive models, state-space models, etc.).

---

<!-- _class: section-divider -->

### PART 3

# Information theory · in two slides

Why log-probabilities are the natural unit

---

# Entropy · uncertainty in bits

<div class="math-box">

$$H(p) = -\sum_y p(y) \log p(y)$$

The **expected number of bits** (when log is base 2) needed to encode a sample from $p$.

</div>

**Examples** ·
- A fair coin · $H = -2 \cdot 0.5 \log 0.5 = 1$ bit. One bit for each flip.
- A biased coin $p = 0.99$ · $H \approx 0.08$ bits. Almost no info per flip.
- Uniform over 8 outcomes · $H = 3$ bits. Maximum uncertainty.

Entropy peaks for uniform distributions (most uncertain) and is 0 for deterministic ones (no uncertainty).

---

# KL divergence and cross-entropy

<div class="math-box">

$$\text{KL}(p \| q) = \sum_y p(y) \log \frac{p(y)}{q(y)} = H(p, q) - H(p)$$

where $H(p, q) = -\sum p(y) \log q(y)$ is **cross-entropy**.

</div>

In words ·
- **Cross-entropy $H(p, q)$** · expected bits to encode samples from $p$ using a code optimized for $q$.
- **KL $\text{KL}(p \| q)$** · the *extra* bits you waste because $q \ne p$. Always $\ge 0$, zero iff $p = q$.

The classifier loss · cross-entropy of the one-hot truth $p$ against the model's prediction $q$. Same idea, dressed up.

---

<!-- _class: section-divider -->

### PART 4

# Numerical tricks for log-probabilities

Why your loss should never be a product of probabilities

---

# Log-sum-exp · the trick everyone uses

The softmax · $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$.

For large $z_i$ (e.g., $z = 1000$) · $e^{1000}$ overflows.

<div class="math-box">

**Trick** · subtract $z_\max = \max_j z_j$ from every term:

$$\sigma(\mathbf{z})_i = \frac{e^{z_i - z_\max}}{\sum_j e^{z_j - z_\max}}$$

Same answer (the $e^{z_\max}$ cancels), but now the largest exponent is 0 and nothing overflows.

</div>

For log-softmax · $\log \sigma(\mathbf{z})_i = z_i - z_\max - \log \sum_j e^{z_j - z_\max}$.

This is `torch.logsumexp` · use it whenever you compute log-probabilities.

---

# Log-likelihood gradients · the elegant fact

For categorical NLL with softmax output:

<div class="math-box">

$$\frac{\partial \mathcal{L}}{\partial z_i} = \hat\pi_i - y_i$$

The gradient of cross-entropy w.r.t. the logits is just **prediction minus truth**.

</div>

Two consequences ·
1. The "softmax + cross-entropy" pair is numerically stable when computed together (`F.cross_entropy(logits, target)` does this fused).
2. The gradient is bounded · between -1 and 1 per logit · no exploding gradients from the loss itself.

This is why classification training is so well-behaved compared to GANs / RL.

---

<!-- _class: section-divider -->

### PART 5

# MAP and the Bayesian view

Why L2 regularization is also MLE-flavored

---

# MAP · maximum a posteriori

Add a **prior** $p(\theta)$ over parameters · use Bayes' rule:

<div class="math-box">

$$p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) \, p(\theta)$$

$\hat\theta_\text{MAP} = \arg\max_\theta \log p(\mathcal{D} \mid \theta) + \log p(\theta)$

</div>

The first term is the log-likelihood · the second term is a **regularizer** that pulls $\theta$ toward whatever the prior favors.

**Example** · prior $p(\theta) = \mathcal{N}(0, \sigma_p^2 I)$ · log-prior $= -\|\theta\|^2 / (2 \sigma_p^2) + \text{const}$.

This is exactly **L2 regularization (weight decay)**! L2 = MAP under a Gaussian prior on weights. We'll see this again in L6.

---

# Putting it all together

A neural network's training loop is a *Monte Carlo MLE* algorithm:

<div class="math-box">

1. Pick a distribution that matches your output.
2. NLL of that distribution = your loss.
3. Sample a batch from your training data.
4. Estimate the gradient · sample-mean over the batch.
5. Step in that direction.
6. Repeat · law of large numbers takes you toward the MLE.

</div>

Add weight decay → MAP. Add KL to a prior on latent variables → ELBO (VAE). Add denoising noise → score matching (diffusion). Same skeleton, different probabilistic interpretation.

---

<!-- _class: section-divider -->

### PART 6

# Linear algebra · the bare minimum

Tensors, gradients, shapes

---

# Vectors and matrices · refresher

<div class="math-box">

- $\mathbf{x} \in \mathbb{R}^d$ · a column vector with $d$ entries.
- $W \in \mathbb{R}^{m \times n}$ · a matrix with $m$ rows, $n$ columns.
- **Matrix-vector product** · $W \mathbf{x} \in \mathbb{R}^m$ · valid when $W$ is $m \times n$ and $\mathbf{x}$ is $n$-dim.

</div>

A neural network layer · `y = W @ x + b` · is exactly this. The whole forward pass is a sequence of matrix-vector products with non-linearities sprinkled in.

**Shape mistake** · the #1 source of bugs. Always print shapes.

---

# Worked example · MLP forward pass

Input · $\mathbf{x} \in \mathbb{R}^2$ · `[1.0, 2.0]`. Hidden 3 · output 1.

<div class="math-box">

$W_1 \in \mathbb{R}^{3 \times 2}$ · $\begin{bmatrix} 0.5 & -0.1 \\ 0.2 & 0.3 \\ -0.4 & 0.6 \end{bmatrix}$. $b_1 \in \mathbb{R}^3$ · $[0.1, 0.0, -0.2]$.

$\mathbf{h}_1 = W_1 \mathbf{x} + b_1 = [0.5 \cdot 1 + (-0.1) \cdot 2 + 0.1, 0.2 \cdot 1 + 0.3 \cdot 2 + 0.0, -0.4 \cdot 1 + 0.6 \cdot 2 - 0.2]$
$= [0.4, 0.8, 0.6]$

After ReLU · same (all positive).

$W_2 \in \mathbb{R}^{1 \times 3}$ · $[0.3, -0.2, 0.5]$. $b_2 = 0.1$.
$y = 0.3 \cdot 0.4 + (-0.2) \cdot 0.8 + 0.5 \cdot 0.6 + 0.1 = 0.36$

</div>

That's it · matrix multiply, add bias, activation, repeat. The whole course is variations on this loop.

---

# Gradient · in one line

For a scalar function $f(\theta)$ where $\theta \in \mathbb{R}^d$ ·

<div class="math-box">

$$\nabla_\theta f = \begin{bmatrix} \partial f / \partial \theta_1 \\ \vdots \\ \partial f / \partial \theta_d \end{bmatrix}$$

A vector pointing in the direction of steepest **increase**. To minimize, step in the **opposite** direction.

</div>

Gradient descent · $\theta \leftarrow \theta - \eta \nabla_\theta f$. The whole optimization story is "compute this gradient, take a small step."

Backprop is just an efficient algorithm for computing this gradient through a deep computation graph.

---

# Chain rule · in pictures

If $z = f(y)$ and $y = g(x)$ ·

<div class="math-box">

$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}$$

</div>

For deep networks · gradient at layer 1 = $\prod_{l = L}^{1} \frac{\partial \text{output}_l}{\partial \text{input}_l}$.

Each factor is a Jacobian. If they're $< 1$ on average · gradient **vanishes**. If $> 1$ · **explodes**. We'll see this exact failure mode in L02 (deep MLPs) and L10 (RNNs).

---

# Likelihood · in one line

Given a dataset $\mathcal{D} = \{y_1, \ldots, y_N\}$ and a model with parameters $\theta$ that defines $p(y \mid \theta)$ ·

<div class="math-box">

$$\mathcal{L}(\theta) = p(\mathcal{D} \mid \theta) = \prod_{i=1}^N p(y_i \mid \theta)$$

</div>

The **likelihood** of $\theta$ is the probability of observing the data we actually saw.

**Independence assumption** · we factor across data points. This is the IID assumption every ML course makes.

---

# Why log-likelihood

Likelihood is a **product of N small numbers**. For $N = 1000$ and each $p_i \approx 0.5$ ·
$\mathcal{L} \approx 0.5^{1000} \approx 10^{-301}$

That's smaller than any float can represent. Underflow.

<div class="keypoint">

**Fix · take logs.** Products become sums:

$$\log \mathcal{L} = \sum_{i=1}^N \log p(y_i \mid \theta)$$

Now we have a sum of N moderate negative numbers · numerically stable.

</div>

We always **maximize log-likelihood** in practice, never raw likelihood.

---

# Maximum likelihood estimation (MLE)

<div class="math-box">

$$\hat\theta_\text{MLE} = \arg\max_\theta \sum_{i=1}^N \log p(y_i \mid \theta)$$

</div>

In words · "find the parameters $\theta$ that make the observed data most probable."

**By convention · we *minimize* the negative log-likelihood (NLL).**

$$\hat\theta_\text{MLE} = \arg\min_\theta -\sum_{i=1}^N \log p(y_i \mid \theta)$$

Every loss in DL is an NLL under some assumed distribution. Watch.

---

<!-- _class: section-divider -->

### PART 3

# Deriving the losses

MSE for regression · cross-entropy for classification · all from MLE

---

# Linear regression · the assumption

Assume · the target $y$ is generated by

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

In words · the model's prediction is the *mean* of a Gaussian; observations differ from it by Gaussian noise of fixed variance.

<div class="math-box">

Equivalently · $p(y \mid \mathbf{x}, \mathbf{w}, b) = \mathcal{N}(\mathbf{w}^\top \mathbf{x} + b, \sigma^2)$

</div>

Given a dataset, we want $\mathbf{w}, b$ that **maximize the likelihood** of the observed $y$'s.

---

# MSE · pop out from MLE

Take the log-likelihood for one datapoint and simplify:

<div class="math-box">

$$\log p(y_i \mid \mathbf{x}_i) = -\frac{1}{2} \log(2\pi \sigma^2) - \frac{(y_i - \mathbf{w}^\top \mathbf{x}_i)^2}{2\sigma^2}$$

Sum over $N$, drop constants in $\theta$ and $\sigma$:

$$\log \mathcal{L} = - \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 + \text{const}$$

</div>

**Maximizing this = minimizing $\sum (y_i - \hat y_i)^2 = $ MSE.**

So MSE for regression isn't an arbitrary choice · it's MLE under Gaussian noise.

---

# Logistic regression · the assumption

Binary classification. Given $\mathbf{x}$, the target $y \in \{0, 1\}$ is **Bernoulli** with parameter $p = \sigma(\mathbf{w}^\top \mathbf{x} + b)$.

<div class="math-box">

$$p(y \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x})^y \cdot (1 - \sigma(\mathbf{w}^\top \mathbf{x}))^{1-y}$$

</div>

The sigmoid squashes the linear output to $[0, 1]$ · interpretable as a probability for class 1.

---

# Cross-entropy · pop out from MLE

Take log:

<div class="math-box">

$$\log p(y_i \mid \mathbf{x}_i) = y_i \log \hat p_i + (1 - y_i) \log (1 - \hat p_i)$$

where $\hat p_i = \sigma(\mathbf{w}^\top \mathbf{x}_i)$.

Sum over $N$ and **negate** to get NLL:

$$-\log \mathcal{L} = -\sum_{i=1}^N y_i \log \hat p_i + (1 - y_i) \log(1 - \hat p_i)$$

</div>

This is **binary cross-entropy** · the standard loss for binary classification. It came directly from "assume Bernoulli, take MLE."

---

# Multiclass · same story

For $K$-way classification, the network's softmax output is a **Categorical** distribution.

<div class="math-box">

For one example with true class $y_i \in \{1, \ldots, K\}$:
$\log p(y_i \mid \mathbf{x}_i) = \log \hat\pi_{i, y_i}$

NLL · $-\sum_i \log \hat\pi_{i, y_i}$

</div>

This is **categorical cross-entropy** · the loss every classifier in this course will use. Same idea · different distribution · same recipe.

---

# Worked example · bin-CE on one prediction

Network outputs · $\hat p = 0.8$ for the cat-vs-dog classifier.

<div class="math-box">

| true class y | NLL = -log p(y) | model "happy"? |
|:-:|:-:|:-:|
| 1 (cat) | $-\log(0.8) = 0.223$ | yes (small loss) |
| 0 (dog) | $-\log(0.2) = 1.609$ | no (big loss) |

</div>

The loss is **small when the model assigned high probability to the true class** and large otherwise. That's the entire idea behind cross-entropy.

---

<!-- _class: section-divider -->

### PART 4

# Why this matters for DL

Recap · every loss = NLL under an assumed distribution

---

# The pattern

<div class="math-box">

| Output type | Distribution | Loss = NLL |
|:-:|:-:|:-:|
| Real number | Gaussian | **MSE** |
| Binary | Bernoulli | **Binary cross-entropy** |
| K classes | Categorical | **Categorical cross-entropy** |
| Image pixels (continuous) | Gaussian per pixel | per-pixel MSE |
| Discrete tokens | Categorical | next-token CE (LLMs · L13-15) |
| Latent variables | Gaussian + KL | ELBO (VAE · L19) |

</div>

Every loss in this course follows the recipe:
1. Pick a distribution that matches your output.
2. Write down NLL.
3. That's your loss.

---

# Why log scales matter · numeric

A network with 50% confidence on the wrong class:
$-\log(0.5) = 0.693$

A network with 99% confidence on the wrong class:
$-\log(0.01) = 4.605$ · ~7× larger penalty.

A network with 99.99% confidence on the wrong class:
$-\log(0.0001) = 9.21$ · ~13× larger penalty.

<div class="keypoint">

Cross-entropy **strongly penalizes overconfident wrong predictions**. This forces the model to be both **accurate** and **calibrated** · not just accurate.

</div>

---

# Connecting to KL divergence

KL divergence between two distributions $p$ and $q$:

<div class="math-box">

$$\text{KL}(p \| q) = \sum_y p(y) \log \frac{p(y)}{q(y)}$$

For a one-hot true label $p$ and softmax model $q$ ·
$\text{KL}(p \| q) = -\log q(y_\text{true}) + \text{const}$

</div>

So **cross-entropy is KL divergence to the one-hot true label**, up to a constant. This connects the same loss to information theory · "how many extra bits do I need to encode the truth using my model's predicted distribution?"

---

# Common questions · FAQ

**Q. Do we always use MLE?**
A. No · we sometimes use MAP (maximum a posteriori) which adds a prior · same recipe but with a regularization term. L2 weight decay is exactly MAP under a Gaussian weight prior (we'll see this in L6).

**Q. Why minimize negative log-likelihood instead of maximizing log-likelihood?**
A. Pure convention · ML libraries minimize · `loss.backward()` only goes downhill. Negate, minimize, same answer.

**Q. What if my output isn't Bernoulli/Categorical/Gaussian?**
A. Pick whatever distribution matches. Poisson for counts. Beta for probabilities. Mixture-of-Gaussians for multimodal. The recipe (NLL = loss) is universal.

---

<!-- _class: summary-slide -->

# Lecture 0 — summary

- **Bernoulli, Categorical, Normal** · the three distributions you'll see most.
- **Likelihood** · probability of the data under the model.
- **Log-likelihood** · sum-version, numerically stable.
- **MLE** · choose params to maximize log-likelihood.
- **MSE** · MLE under Gaussian noise (regression).
- **Cross-entropy** · MLE under Bernoulli/Categorical (classification).
- **Negative log-likelihood = the loss** everywhere in DL.

### Read before Lecture 1

- Strang's "Linear Algebra" Ch 1 · vectors, matrices, dot products.
- Bishop's *Pattern Recognition and ML* Ch 1 (free PDF) · prob theory.

### Next lecture

**Lecture 1 · Why deep learning?** · we'll use cross-entropy from day 1, now you know where it comes from.
