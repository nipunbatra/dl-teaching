---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Probability & the Language of Losses

## Lecture 1 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

A neural network never really outputs a *number*. It outputs a **distribution** — and training it means making your data **most likely** under that distribution.

<div class="keypoint">

**Every loss in this course is a negative log-likelihood.**
Choose the distribution your model outputs, and the loss writes itself — MSE, cross-entropy, all the same recipe. A **prior** on the weights turns out to be exactly a **regularizer**.

</div>

You met $\theta$, $\hat y$, and the cost $J(\theta)$ in ES 335. Today we answer the question that course left open: **where does $J(\theta)$ come from?**

$$\text{distributions} \rightarrow \text{likelihood} \rightarrow \underbrace{\text{MLE}}_{\text{MSE, BCE, CE}} \rightarrow \underbrace{\text{MAP}}_{\text{L2, L1}} \rightarrow \underbrace{\text{entropy / KL}}_{\text{why cross-entropy}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**You'll reuse today's ideas in:**
- **L2–L3** — MSE & cross-entropy as the outputs of linear/logistic/neural nets
- **L7** — L2 & L1 regularization = Gaussian & Laplace priors
- **L16–L18** — cross-entropy is *the* LLM training loss
- **L21** — the VAE's KL term, straight from Part 5

</div>
<div>

**Borrowed from your own courses** (we'll build on, not repeat):
- distributions & demos — **PSDV**
- MLE / MAP / info-theory — **Prob. ML**
- MLE→cross-entropy — **ML (ES 335)**

</div>
</div>

<div class="insight">

This is a **primer**: fast where you've seen it, slow where it becomes the language of deep learning.

</div>

---

<!-- _class: section-divider -->

## Part 1 · A model outputs a distribution, not a number

---

# The real output of a classifier is $p(y \mid x)$

When a model "predicts cat," it emits $P(\text{cat}\mid x)=0.9$ — a full distribution over labels. For regression, it emits a Gaussian centered at $\hat y$.

![w:600px](figures/lec00/svg/conditional_view.svg)

<div class="insight">

The single $\hat y$ you read off is just a *summary* of that distribution — its mean, or its argmax. The **loss compares the whole predicted distribution to the truth.** So first we need the vocabulary of distributions.

</div>

---

# A random variable, and its two summaries

A **random variable** $Y$ maps outcomes to numbers; a distribution $p(y)$ says how probable each is.

![w:520px](figures/lec00/svg/random_variable_map.svg)

Two summaries we lean on constantly:

$$\mathbb E[Y]=\sum_y y\,p(y) \quad(\text{the mean}) \qquad \text{Var}(Y)=\mathbb E\big[(Y-\mathbb E[Y])^2\big]\quad(\text{the spread})$$

**Linearity** $\mathbb E[aX+bY]=a\,\mathbb E[X]+b\,\mathbb E[Y]$ holds *always* — no independence needed. We'll use it in the SGD unbiasedness proof (L5).

---

# Variance = expected squared deviation

<div class="notebook">

**🎛 Interactive · variance as a weighted average** — for a Binomial, watch $\text{Var}=\mathbb E[(X-\mu)^2]$ built as three stacked bars: the PMF, the squared deviation $(x-\mu)^2$, and their product. *(PSDV · `apps/variance.py`)*

</div>

<div class="popquiz">

**Warm-up.** A fair die: $\mathbb E[X]=3.5$. Is $\text{Var}(X)$ closer to **1**, **3**, or **9**? (Which outcomes contribute the biggest squared deviations?)

</div>

*Answer:* $\approx 2.92$ — the 1s and 6s (deviation $2.5$, squared $6.25$) dominate. Spread lives in the tails.

---

# Three distributions cover almost everything

![w:880px](figures/lec00/svg/three_distributions.svg)

- **Bernoulli** — one yes/no ("spam?"): parameter $p$
- **Categorical** — one of $K$ classes ("which digit?"): the **softmax** output $(p_1,\dots,p_K)$
- **Gaussian** — a real number ("house price"): mean $\hat y$, variance $\sigma^2$

Each is pinned down by *exactly the quantity your network predicts.*

---

# Bernoulli — the seed of cross-entropy

One coin flip, written in one compact line:

<div class="math-box">

$$P(y \mid p) = p^{\,y}\,(1-p)^{1-y}, \qquad y \in \{0,1\}$$

</div>

The **10-flip** running example from ES 335 — $(H,H,T,T,T,H,H,T,T,T)$ → 4 heads:

$$P(\text{data}\mid \theta) = \theta^{4}(1-\theta)^{6}$$

Hold onto $p^{y}(1-p)^{1-y}$ — take its $-\log$ and you have **binary cross-entropy** (Part 3).

---

# Categorical — one of $K$ classes

For a one-hot label $y$ over $K$ classes, with model probabilities $\mathbf p=(p_1,\dots,p_K)$:

$$P(y\mid \mathbf p)=\prod_{k=1}^{K} p_k^{\,[y=k]}$$

This is exactly what **softmax** produces — one probability per class, summing to 1.

![w:560px](figures/lec00/svg/mnist_categorical.svg)

MNIST digit "3" → the model outputs a 10-way categorical; the loss will ask "how much mass did you put on the true class?"

---

# Gaussian — a real-valued target

The network predicts the **mean** $\hat y$; the Gaussian says nearby values are plausible, far ones aren't:

$$p(y\mid \hat y)=\frac{1}{\sqrt{2\pi}\,\sigma}\,\exp\!\Big(\!-\frac{(y-\hat y)^2}{2\sigma^2}\Big)$$

![w:520px](figures/lec00/svg/normal_bellcurve.svg)

That $(y-\hat y)^2$ in the exponent is about to become **mean squared error**.

---

# Counts: Binomial & Poisson (a quick tour)

<div class="columns">
<div>

**Binomial** — $n$ Bernoulli trials, count the successes:
$$P(k)=\binom{n}{k}p^k(1-p)^{n-k}$$
"How many of 100 emails are spam?"

</div>
<div>

**Poisson** — counts of rare events per interval:
$$P(k)=\frac{\lambda^k e^{-\lambda}}{k!}$$
"How many requests hit the server this second?"

</div>
</div>

<div class="notebook">

**📓 Notebook · the distribution zoo** — every distribution above, plotted and fit from data with `torch.distributions`. *(PSDV · `notebooks/pmf-discrete.ipynb`, `pdf-continuous.ipynb`)*

</div>

---

# Explore them yourself

<div class="notebook">

**🎛 Interactive · distribution explorer** — drag the parameters of Normal / Exponential / Beta / Gamma and watch the density move. Build intuition before the algebra. *(PSDV · `apps/distributions-plot.py`)*

</div>

<div class="notebook">

**🎛 Interactive · Seeing the Multivariate Normal** — a scroll-driven explainer: drag the covariance $\Sigma$ and watch the samples, the $1\sigma$ ellipse, and the eigenvectors rotate. We'll reuse this "rotate-then-stretch" picture for the VAE reparameterization in L21. *(Interactive Lab · `interactive/articles/multivariate-normal` · also PSDV `apps/normal-shape.py`)*

</div>

---

# One assumption ties the data together: IID

To turn *many* examples into *one* likelihood, we assume they are **independent and identically distributed** — so the joint probability is a **product**:

$$P(y_1,\dots,y_n\mid\theta)=\prod_{i=1}^{n}p(y_i\mid\theta)$$

![w:640px](figures/lec00/svg/iid_matrix.svg)

Every loss we write sums over the batch precisely *because* of this product-of-IID assumption.

---

# When IID breaks

<div class="popquiz">

**Work it out.** For which of these is IID *wrong*? (a) shuffled MNIST digits · (b) a company's daily stock price · (c) frames of a video · (d) rolls of a die.

</div>

![w:560px](figures/lec00/svg/iid_fails.svg)

*Answer:* **(b) and (c)** — today's value depends on yesterday's; the samples are correlated in time. We'll return to this exact failure when we hit **sequence models (L12)**; for now, shuffled mini-batches keep us honestly IID.

---

<!-- _class: section-divider -->

## Part 2 · Likelihood → negative log-likelihood

---

# Likelihood: which parameter makes the data most plausible?

Flip the question: fix the data, and vary $\theta$. The **likelihood** $\mathcal L(\theta)=\theta^4(1-\theta)^6$ scores each candidate $\theta$; its peak is the value the data "votes" for.

![w:600px](figures/lec00/svg/coin_likelihood.svg)

---

# Work it out: is $\theta=0.4$ really better than $0.5$?

<div class="popquiz">

**Compute.** With 4 heads in 10 flips, compare $\mathcal L(0.4)$ vs $\mathcal L(0.5)$.

</div>

$$\mathcal L(0.4)=0.4^4\,0.6^6 = 0.00119\qquad \mathcal L(0.5)=0.5^{10}=0.00098$$

<div class="insight">

$\theta=0.4$ wins — the observed frequency is the most likely parameter. Notice both numbers are *tiny* (products of probabilities), and this is only 10 flips. With a million examples they underflow to exactly $0$. That's the first of two problems.

</div>

---

# Two problems, one fix: take the log

Raw likelihood is a **product** of many tiny numbers (underflows), and products are awkward to differentiate. $\log$ fixes both — it turns products into sums and is **monotonic**, so it never moves the maximum.

![w:560px](figures/lec00/svg/log_monotonic.svg)

$$\arg\max_\theta \prod_i p(y_i\mid\theta) = \arg\max_\theta \sum_i \log p(y_i\mid\theta) = \arg\min_\theta \underbrace{-\sum_i \log p(y_i\mid\theta)}_{\text{negative log-likelihood}}$$

---

# The negative log-likelihood *is* the loss

<div class="keypoint">

Minimizing the **NLL** is the same as maximizing likelihood. Every loss in the course is this one line with a different distribution plugged in:

$$J(\theta) = -\sum_i \log p(y_i \mid x_i, \theta)$$

</div>

The rest of Part 3 is just: *plug in Gaussian, Bernoulli, Categorical — read off MSE, BCE, cross-entropy.*

---

<!-- _class: section-divider -->

## Part 3 · MLE — every loss is a negative log-likelihood

---

# The recipe

<div class="math-box">

**Maximum-likelihood estimation, in three steps**
1. Write $p(y\mid \text{model})$ for one example.
2. Sum $-\log$ over the dataset → the NLL.
3. Minimize (set gradient $=0$, or run gradient descent).

</div>

Everything below is this recipe, three times.

---

# Worked example: the coin

$$J(\theta)=-\log\big[\theta^4(1-\theta)^6\big] = -4\log\theta - 6\log(1-\theta)$$

Set $\dfrac{dJ}{d\theta}=0$:

$$-\frac{4}{\theta}+\frac{6}{1-\theta}=0 \;\Longrightarrow\; 4(1-\theta)=6\theta \;\Longrightarrow\; \hat\theta=\frac{4}{10}=0.4$$

<div class="insight">

The answer you'd have *guessed* (the head-frequency) is now **derived** — and it matched the likelihood-plot peak from Part 2.

</div>

---

# Your turn

<div class="popquiz">

**Work it out.** A different coin lands **7 heads in 10 flips**. What is $\hat\theta_{\text{MLE}}$? Redo the derivative — what's the general formula for $h$ heads in $n$ flips?

</div>

*Answer:* $\hat\theta = \dfrac{7}{10}=0.7$, and in general $\hat\theta=\dfrac{h}{n}$ — MLE for a Bernoulli is just the sample mean. Clean, and it generalizes to every distribution's "natural" estimate.

---

# Gaussian output → mean squared error

Assume $y \sim \mathcal N(\hat y,\sigma^2)$. The NLL of one example:

$$-\log p(y\mid \hat y) = \frac{(y-\hat y)^2}{2\sigma^2} + \underbrace{\log\sqrt{2\pi}\sigma}_{\text{constant in }\hat y}$$

![w:560px](figures/lec00/svg/mle_linear_regression.svg)

<div class="keypoint">

Drop the constant: minimizing NLL $=$ minimizing $\sum_i (y_i-\hat y_i)^2$. **Least squares is Gaussian maximum likelihood.**

</div>

---

# Bernoulli output → binary cross-entropy

For a Bernoulli prediction $\hat p$, the NLL of one example is

$$-\big[\,y\log\hat p + (1-y)\log(1-\hat p)\,\big]$$

— that *is* **binary cross-entropy**.

![w:480px](figures/lec00/svg/logistic_mle_picture.svg)

The logistic-regression loss you used in ES 335 was never arbitrary — it's the NLL of the Bernoulli that the sigmoid outputs.

---

# Categorical output → cross-entropy

For $K$ classes with one-hot $y$ and softmax probabilities $\hat p_k$:

$$-\sum_{k} [y{=}k]\log \hat p_k \;=\; -\log \hat p_{\text{true class}}$$

<div class="insight">

Cross-entropy only ever looks at the probability you assigned to the **true** class — put more mass there, pay less. This is the single most-used loss in the entire course (every classifier, every LLM).

</div>

---

# The payoff, on one slide

![w:740px](figures/lec00/svg/nll_master_poster.svg)

<div class="keypoint">

Same recipe, three distributions: **Gaussian → MSE · Bernoulli → BCE · Categorical → cross-entropy.** Pick what the model outputs; the loss is its negative log-likelihood.

</div>

---

# See it in code

<div class="notebook">

**📓 Notebook · MLE by hand** — fit a Gaussian and a Bernoulli by maximizing log-likelihood, and check that `torch.nn.MSELoss` / `BCELoss` reproduce your derivation exactly. *(Prob. ML · `notebooks/mle-univariate.ipynb`, `mle_bernoulli.ipynb`)*

</div>

<div class="popquiz">

**Discuss.** Cross-entropy punishes a *confidently wrong* prediction ($\hat p_{\text{true}}\to 0$) with loss $\to\infty$. Why is that a feature, not a bug, for training?

</div>

---

<!-- _class: section-divider -->

## Part 4 · MAP — a prior is a regularizer

---

# Bayes' rule: fold in what you knew before the data

$$\underbrace{p(\theta\mid \text{data})}_{\text{posterior}} \;\propto\; \underbrace{p(\text{data}\mid\theta)}_{\text{likelihood}}\;\underbrace{p(\theta)}_{\text{prior}}$$

![w:560px](figures/lec00/svg/bayesian_updating.svg)

As data arrives, the posterior tightens around the truth — **today's posterior is tomorrow's prior.** MLE throws the prior away; **MAP** keeps it.

---

# The base-rate trap (why priors matter)

<div class="popquiz">

**Work it out.** A test is 99% accurate. The disease affects 1 in 10,000. You test positive. Roughly what's the chance you actually have it — **99%**, **50%**, or **1%**?

</div>

Of 10,000 people: ~1 true positive, but ~100 false positives (1% of 9,999). So $P(\text{sick}\mid +)\approx \tfrac{1}{101}\approx \mathbf{1\%}$.

<div class="insight">

The **prior** (rarity) dominates a strong likelihood. Ignore priors and you're badly miscalibrated — the same failure a model makes when it overfits with no regularization.

</div>

---

# Explore Bayes yourself

<div class="notebook">

**🎛 Interactive · Bayes calculator** — set the prior, sensitivity, and specificity; watch the posterior swing. The contour plot shows how the "false alarm" region depends on the base rate. *(PSDV · `apps/covid-testing.py`)*

</div>

<div class="notebook">

**📓 Notebook · Bayesian updating** — start from a Beta prior on a coin and update it flip-by-flip toward the truth. *(Prob. ML · `notebooks/MAP.ipynb`)*

</div>

---

# MAP = MLE + a penalty on the weights

Maximize the posterior → add $-\log p(\theta)$ to the NLL:

$$\hat\theta_{\text{MAP}} = \arg\min_\theta \Big[\underbrace{-\log p(\text{data}\mid\theta)}_{\text{your loss}} \;\;\underbrace{-\log p(\theta)}_{\text{regularizer}}\Big]$$

<div class="columns">
<div>

**Gaussian prior** $\theta\sim\mathcal N(0,\tau^2)$
$\Rightarrow -\log p(\theta) \propto \lambda\lVert\theta\rVert_2^2$
$\Rightarrow$ **L2 / weight decay (ridge)**

</div>
<div>

**Laplace prior** $\theta\sim\text{Laplace}(0,b)$
$\Rightarrow -\log p(\theta) \propto \lambda\lVert\theta\rVert_1$
$\Rightarrow$ **L1 (lasso), sparsity**

</div>
</div>

---

# Why L1 gives sparsity, in one picture

The prior's *shape* is the whole story: L2's round penalty shrinks weights smoothly; L1's diamond has corners on the axes, so the optimum often lands *exactly* at zero.

![w:560px](figures/lec00/svg/l1_l2_geometry.svg)

<div class="keypoint">

**Regularization isn't a hack — it's a prior belief that weights are small.** We'll reuse this exact derivation for weight decay in L5–L7.

</div>

---

# Worked example: MAP for the coin

Put a $\text{Beta}(\alpha,\beta)$ prior on $\theta$. The posterior stays Beta (conjugacy), and the MAP estimate is:

$$\hat\theta_{\text{MAP}}=\frac{h+\alpha-1}{n+\alpha+\beta-2}$$

![w:520px](figures/lec00/svg/beta_binomial_update.svg)

With 4/10 heads and a gentle $\text{Beta}(2,2)$ prior: $\hat\theta_{\text{MAP}}=\tfrac{4+1}{10+2}=0.417$ — pulled slightly toward $0.5$ by the prior. **Small data → prior matters; big data → likelihood wins.**

---

<!-- _class: section-divider -->

## Part 5 · Entropy & KL — why cross-entropy *is* the loss

---

# Surprise = $-\log p$

A rare event is **surprising** (carries a lot of information); a certain event carries none. Define the information content of an outcome as $-\log p$.

![w:560px](figures/lec00c/svg/info_content.svg)

$p=1 \Rightarrow$ surprise $0$; $\;p\to 0 \Rightarrow$ surprise $\to\infty$. (Base-2 log → **bits**.)

---

# Entropy = expected surprise

Average the surprise over the distribution: $\;H(p)=\mathbb E_p[-\log p]=-\sum_y p(y)\log p(y)$ — the **uncertainty** in $p$.

<div class="columns">
<div>

![w:340px](figures/lec00/svg/entropy_bernoulli.svg)

</div>
<div>

![w:360px](figures/lec00c/svg/entropy_samples.svg)

</div>
</div>

A fair coin has **maximum** entropy (1 bit); a biased coin, less; a two-headed coin, zero.

---

# Entropy is also the optimal code length

Shannon: the fewest bits to encode outcomes from $p$ is $H(p)$ — frequent symbols get short codes, rare ones long codes.

<div class="columns">
<div>

![w:360px](figures/lec00c/svg/coding_tree.svg)

</div>
<div>

![w:360px](figures/lec00c/svg/naive_vs_optimal_code.svg)

</div>
</div>

This coding picture is what makes the next two quantities — cross-entropy and KL — concrete.

---

# Cross-entropy: coding with the *wrong* distribution

If the truth is $p$ but you build your code for your model $q$, your average message length is the **cross-entropy** $H(p,q)=\mathbb E_p[-\log q]$ — always $\ge H(p)$.

![w:520px](figures/lec00c/svg/cross_entropy_extrabits.svg)

That gap — the extra bits you pay for using $q$ instead of $p$ — is exactly the KL divergence.

---

# KL divergence = the extra bits

$$D_{\text{KL}}(p\Vert q)=\sum_y p(y)\log\frac{p(y)}{q(y)} = H(p,q)-H(p) \;\ge 0,\quad =0 \iff q=p$$

![w:520px](figures/lec00/svg/kl_weather.svg)

It's a *directed* "distance" between distributions (not symmetric). Bigger KL → your model is further from the truth.

---

# Work it out: a tiny KL

<div class="popquiz">

**Compute.** Truth $p=(\tfrac12,\tfrac12)$, model $q=(\tfrac34,\tfrac14)$. Find $D_{\text{KL}}(p\Vert q)$ in bits.

</div>

$$D_{\text{KL}} = \tfrac12\log_2\tfrac{1/2}{3/4} + \tfrac12\log_2\tfrac{1/2}{1/4} = \tfrac12(-0.585) + \tfrac12(1) \approx \mathbf{0.208\ \text{bits}}$$

<div class="insight">

You'd waste ~0.2 bits per symbol coding a fair coin as if it were biased. Training a model = shrinking exactly this quantity.

</div>

---

# The loop closes: cross-entropy = KL = NLL

<div class="math-box">

$$H(p,q) = \underbrace{H(p)}_{\text{fixed}} + \underbrace{D_{\text{KL}}(p\,\Vert\,q)}_{\text{what we minimize}}$$

</div>

For classification, $p$ is one-hot, so $H(p)=0$ and **cross-entropy $=$ KL to the truth $=$ NLL.**

![w:400px](figures/lec00c/svg/mle_kl_fit.svg)

<div class="keypoint">

Minimizing cross-entropy, maximizing likelihood, and minimizing KL to the data are **the same thing** — from three different starting points.

</div>

---

# See it in code

<div class="notebook">

**📓 Notebook · information theory** — compute entropy, cross-entropy, and KL on toy distributions; verify `F.cross_entropy` equals your hand-derived NLL. *(Prob. ML · `notebooks/information-theory.ipynb`, `entropy.ipynb`)*

</div>

<div class="popquiz">

**Looking ahead.** In L21 the VAE minimizes reconstruction error **plus** $D_{\text{KL}}\big(q(z\mid x)\,\Vert\,\mathcal N(0,1)\big)$. Given today, what is that KL term *doing* to the encoder?

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Pick the distribution your model outputs; the loss is its negative log-likelihood; a prior on the weights is your regularizer.**

</div>

- A model outputs a **distribution** — Bernoulli / Categorical / Gaussian.
- **MLE**: Gaussian → MSE, Bernoulli/Categorical → cross-entropy.
- **MAP**: Gaussian prior → L2, Laplace prior → L1.
- **Cross-entropy = KL to the truth = NLL** — the loss you'll use all semester.

**Next (L2):** from these *losses* to the *models* — linear → logistic → neural networks, and why we need a nonlinearity.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's own courses (IIT Gandhinagar) and standard references:

- **Distributions, interactive demos, Bayes base-rate** — *Probability, Statistics & Data Visualization (PSDV)*, N. Batra · `github.com/nipunbatra/psdv-teaching`
- **MLE, MAP, information theory (entropy / cross-entropy / KL)** — *Probabilistic Machine Learning*, N. Batra · `github.com/nipunbatra/pml-teaching`
- **MLE → cross-entropy, the 10-coin-flip example** — *Machine Learning (ES 335)*, N. Batra · `github.com/nipunbatra/ml-teaching`
- **Pedagogical framing** — A. Ng, *Deep Learning Specialization* (losses introduced in-line, Course 1); C. Bishop, *PRML* Ch. 1–2.

Figures adapted from the ES 667 figure library. All source courses © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · MLE can be biased

The Gaussian variance MLE $\hat\sigma^2_{\text{MLE}}=\tfrac1n\sum_i(x_i-\bar x)^2$ **underestimates** $\sigma^2$: $\;\mathbb E[\hat\sigma^2_{\text{MLE}}]=\tfrac{n-1}{n}\sigma^2$. The $n{-}1$ (Bessel's correction) fixes it. Takeaway: maximum likelihood is consistent but not always unbiased — a caveat, not a crisis.

*(Full derivation: Prob. ML `MLE.tex`, "bias of $\sigma^2_{\text{MLE}}$".)*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · KL between two Gaussians

For $p=\mathcal N(\mu_1,\sigma_1^2)$, $q=\mathcal N(\mu_2,\sigma_2^2)$:

$$D_{\text{KL}}(p\Vert q)=\log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac12$$

We'll need exactly this closed form when the **VAE** (L21) pushes its encoder's posterior toward $\mathcal N(0,1)$. *(Step-by-step: Prob. ML `Variational-Inference.tex`.)*
