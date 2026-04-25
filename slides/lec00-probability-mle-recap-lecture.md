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

# Likelihood and log-likelihood

The number that says how well your model fits the data

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
