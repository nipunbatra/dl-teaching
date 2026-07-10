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

# Intuition · a distribution is a more *honest* answer

A weather app never says "it will rain tomorrow." It says **"70% chance of rain"** — and that hedge is the whole point: it tells you *how sure* the model is.

<div class="insight">

A model that emits only $\hat y$ is a forecaster who refuses to hedge. One that emits $p(y\mid x)$ reports its **best guess** *and* its **confidence** — and confidence is exactly what the loss will grade.

</div>

<div class="columns">
<div>

**"It's a cat."**
A bare label — no way to say how sure, and no way to be *penalized* for false bravado.

</div>
<div>

**"$P(\text{cat})=0.9$."**
A bet. Say $0.99$ and be wrong, you pay far more than if you had hedged at $0.6$.

</div>
</div>

That "pay more when confidently wrong" is **cross-entropy** — four parts from now, it falls out for free.

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

# Checkpoint · which distribution does each output need?

<div class="popquiz">

**Match each task to the distribution its output layer should emit:**
(1) "Is this email spam?" · (2) "Which of 10 digits?" · (3) "Tomorrow's temperature in °C" · (4) "How many buses arrive in the next hour?"

</div>

*Answer:* (1) **Bernoulli** — one sigmoid · (2) **Categorical** — a 10-way softmax · (3) **Gaussian** — a mean $\hat y$ · (4) **Poisson** — a non-negative rate $\lambda$.

<div class="insight">

The rule of thumb: **look at the shape of the target, then pick the distribution whose support matches it** — $\{0,1\}$, one-of-$K$, all of $\mathbb R$, or the counts $0,1,2,\dots$. The output nonlinearity (sigmoid / softmax / linear / softplus) is just whatever *guarantees* that support.

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

# Intuition · the underflow picture

Why *minimize the negative log* instead of just *maximizing the product*? Watch what the product does on real data.

<div class="math-box">

$1{,}000$ examples, each with probability $\approx 0.1$:
$$\underbrace{0.1 \times 0.1 \times \cdots}_{1000} = 10^{-1000} \;\xrightarrow{\text{float32}}\; \mathbf{0.0}$$
The product **underflows to exactly zero** — every candidate $\theta$ now looks equally hopeless. The log stays perfectly civilized:
$$\sum_i \log(0.1) = 1000 \times (-2.30) = -2302 \quad(\text{a fine, comparable number})$$

</div>

<div class="insight">

**Gotcha — likelihood is *not* a probability over $\theta$.** $\mathcal L(\theta)$ says how well *each* $\theta$ explains the *fixed* data; it does **not** integrate to $1$ over $\theta$ (that object is the *posterior*, Part 4). "Most likely $\theta$" means "best explanation," not "most probable value."

</div>

<div class="notebook">

**🎛 Interactive · numerical tricks of the trade** — slide a single logit toward $700$ and watch naive softmax return `+Inf` while `log_softmax` and BCE-with-logits stay finite. This is *why* every framework computes losses in log-space. *(Interactive Lab · `interactive/articles/numerical-tricks`)*

</div>

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

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** A coin lands **7 heads in 10 flips**. Write the NLL, set its derivative to zero, and find $\hat\theta_{\text{MLE}}$. Then give the general formula for $h$ heads in $n$ flips.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

$$J(\theta) = -7\log\theta - 3\log(1-\theta)$$

$$\frac{dJ}{d\theta} = -\frac{7}{\theta} + \frac{3}{1-\theta} = 0 \;\Longrightarrow\; 7(1-\theta)=3\theta \;\Longrightarrow\; \hat\theta=\frac{7}{10}=0.7$$

<div class="keypoint">

In general $\hat\theta=\dfrac{h}{n}$ — **MLE for a Bernoulli is just the sample mean.** The same "set the log-likelihood derivative to zero" move gives every distribution's natural estimate.

</div>

---

# Gaussian output → mean squared error

Assume $y \sim \mathcal N(\hat y,\sigma^2)$. The NLL of one example:

$$-\log p(y\mid \hat y) = \frac{(y-\hat y)^2}{2\sigma^2} + \underbrace{\log\sqrt{2\pi}\sigma}_{\text{constant in }\hat y}$$

![w:560px](figures/lec00/svg/mle_linear_regression.svg)

<div class="keypoint">

Drop the constant: minimizing NLL $=$ minimizing $\sum_i (y_i-\hat y_i)^2$. **Least squares is Gaussian maximum likelihood.**

</div>

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** A model predicts a single constant $\hat y=\mu$ for three points $y=(2,4,9)$, assuming $y\sim\mathcal N(\mu,\sigma^2)$ with $\sigma$ fixed. Write the NLL, drop the $\sigma$-constant, and minimize over $\mu$. Which value wins — and what does that reveal about the mean?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

Stack the three per-example NLLs and drop the constant:

$$J(\mu) = \frac{1}{2\sigma^2}\Big[(2-\mu)^2+(4-\mu)^2+(9-\mu)^2\Big]$$

$$\frac{dJ}{d\mu}= -\frac{1}{\sigma^2}\big[(2-\mu)+(4-\mu)+(9-\mu)\big]=0 \;\Longrightarrow\; \hat\mu=\frac{2+4+9}{3}=5$$

<div class="keypoint">

Minimizing a Gaussian NLL **is** minimizing MSE — and its optimum is the **sample mean**. Notice $\sigma$ never touched the *location* of the minimum: it scales the loss, not its $\arg\min$. That is exactly why plain MSE quietly assumes *constant* noise (homoscedastic Gaussian).

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

# Intuition · why cross-entropy punishes *confident* wrongness

The bill for the true class is $-\log \hat p_{\text{true}}$. Read it straight off the curve:

<div class="columns">
<div>

| $\hat p_{\text{true}}$ | loss $-\log_2 \hat p$ |
|:--:|:--:|
| $0.9$ | $0.15$ bits |
| $0.5$ | $1.0$ bit |
| $0.1$ | $3.3$ bits |
| $0.01$ | $6.6$ bits |
| $\to 0$ | $\to \infty$ |

</div>
<div>

Halving your confidence in the truth adds a roughly *fixed* toll each time; being **confidently wrong** ($\hat p_{\text{true}}\to 0$) is charged an *unbounded* one.

</div>
</div>

<div class="insight">

That is a **feature**, not a bug: the gradient blows up as $\hat p_{\text{true}}\to 0$, yanking the model hard away from arrogant mistakes — far faster than squared error ever would. Confidence is cheap only when it is correct.

</div>

<div class="notebook">

**🎛 Interactive · softmax & temperature** — edit logits, slide temperature from peaky to uniform, and watch the entropy read-out (and a live sample) move. The same knob drives LLM sampling, distillation, and calibration. *(Interactive Lab · `interactive/articles/softmax-temperature`)*

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

**📓 Notebook · MLE by hand** — fit a Gaussian and a Bernoulli by maximizing log-likelihood, watch the log-likelihood *surface* rise to its peak, and confirm `torch.nn.MSELoss` / `BCELoss` reproduce your derivation exactly. *(Prob. ML · `notebooks/mle-univariate.ipynb`, `mle_bernoulli.ipynb`, `log-likelihood-linreg.ipynb`)* · **💡 worth building:** *"the likelihood landscape"* — a live 1-D/2-D surface you climb to the MLE peak.

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

# Practice problem 3 · the base-rate trap

<div class="popquiz">

**Practice problem 3.** A test is 99% accurate. The disease affects 1 in 10,000. You test positive. Roughly what's the chance you actually have it — **99%**, **50%**, or **1%**? *(Count people, don't just multiply.)*

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

Imagine 10,000 people. About **1** is truly sick (a true positive), but about **100** healthy people also test positive (1% of 9,999). So

$$P(\text{sick}\mid +)\approx \frac{1}{1+100}\approx \mathbf{1\%}$$

<div class="insight">

The **prior** (rarity) dominates even a strong likelihood. Ignore priors and you're badly miscalibrated — the same failure a model makes when it overfits with no regularization.

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

# Intuition · the prior as a rubber band

MAP is a **tug-of-war**. The likelihood pulls the weights toward whatever fits the data; the prior pulls them back toward $0$. Where they balance is $\hat\theta_{\text{MAP}}$.

<div class="columns">
<div>

**$\lambda$ is the stiffness of the band.**
- $\lambda\to 0$ (loose) → MAP $\to$ MLE, pure data fit.
- $\lambda$ large (stiff) → weights dragged toward $0$, the model stays simple.

</div>
<div>

**More data $\Rightarrow$ weaker pull.**
The likelihood is a sum over $n$ points; the prior is one fixed term. As $n$ grows the data out-votes the prior — the band relaxes on its own.

</div>
</div>

<div class="insight">

**Common misconception:** "regularization is a hack bolted on to stop overfitting." No — it is a **prior belief** that weights are small, folded in by Bayes' rule. The knob $\lambda$ is just *how strongly* you hold that belief.

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

# Practice problem 4

<div class="popquiz">

**Practice problem 4.** You measure one noisy value $y=10$ of a quantity $\mu$, with known noise $\sigma=2$. You also hold a prior $\mu\sim\mathcal N(0,\tau^2)$ with $\tau=2$. The MAP estimate is the precision-weighted blend
$$\hat\mu_{\text{MAP}}=\frac{\tau^2}{\sigma^2+\tau^2}\,y .$$
Plug in the numbers. Where does $\hat\mu$ land, and which way did the prior pull it?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 4

$$\hat\mu_{\text{MAP}}=\frac{\tau^2}{\sigma^2+\tau^2}\,y=\frac{4}{4+4}\times 10 = \tfrac12\times 10 = \mathbf{5}$$

The data alone (the MLE) says $\hat\mu=10$; the prior says $0$; the MAP lands **halfway** — because here data and prior are equally trustworthy ($\sigma=\tau$).

<div class="keypoint">

MAP with a Gaussian prior is **shrinkage** — a convex blend of the data estimate and the prior mean, weighted by relative precision ($1/\sigma^2$ vs $1/\tau^2$). Trust the data more ($\sigma\!\downarrow$) and $\hat\mu\to y$; trust the prior more ($\tau\!\downarrow$) and $\hat\mu\to 0$. **This is exactly weight decay tugging every weight a little toward zero.**

</div>

<div class="notebook">

**🎛 Interactive · MLE vs MAP on a coin** — slide a $\text{Beta}(\alpha,\beta)$ prior, flip $1\!\to\!1000$ coins, and watch the estimate travel from MAP toward MLE as data accumulates: the whole of Parts 3–4 in one picture. *(Interactive Lab · `interactive/articles/mle-map-coin`)*

</div>

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

# Explore entropy, cross-entropy & KL yourself

<div class="notebook">

**🎛 Interactive · information theory by hand** — drag a categorical distribution and watch entropy fall as it sharpens; drop a model $q$ beside the data $p$ and see cross-entropy split *visually* into $H(p)+D_{\text{KL}}(p\Vert q)$; toggle forward vs reverse KL on a bimodal target to feel **mode-covering vs mode-seeking** — the exact choice a VAE makes in L21. *(Interactive Lab · `interactive/articles/info-theory`)*

</div>

<div class="notebook">

**💡 Build-your-own · "cross-entropy as a code"** — a high-value explainer to add: draw a Huffman code for the true $p$, then *reuse that same code* on a mismatched $q$ and watch the wasted bits tick up — accumulating $H(p,q)=H(p)+D_{\text{KL}}$ one symbol at a time. Makes "extra bits" literally countable.

</div>

---

# Practice problem 5 · a tiny KL

<div class="popquiz">

**Practice problem 5.** Truth $p=(\tfrac12,\tfrac12)$, model $q=(\tfrac34,\tfrac14)$. Compute $D_{\text{KL}}(p\Vert q)$ in bits.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 5

$$D_{\text{KL}} = \tfrac12\log_2\frac{1/2}{3/4} + \tfrac12\log_2\frac{1/2}{1/4} = \tfrac12(-0.585) + \tfrac12(1) \approx \mathbf{0.208\ \text{bits}}$$

<div class="insight">

You'd waste ~0.2 bits per symbol coding a fair coin as if it were biased. Training a model = shrinking exactly this quantity.

</div>

---

# Practice problem 6

<div class="popquiz">

**Practice problem 6.** A biased coin lands heads with $p=0.9$. Compute its entropy $H=-p\log_2 p-(1-p)\log_2(1-p)$ in bits. Is it closer to a **fair** coin ($1$ bit) or a **two-headed** coin ($0$ bits)? Then: which stream costs *more* bits to transmit — these biased flips, or fair flips?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 6

$$H = -0.9\log_2 0.9 - 0.1\log_2 0.1 = 0.9(0.152)+0.1(3.32) \approx \mathbf{0.47\ \text{bits}}$$

Well below the fair coin's $1$ bit — the outcome is *mostly* predictable, so each flip carries under half a bit of surprise.

<div class="insight">

**Fewer bits.** A predictable source is cheap to transmit: a good compressor spends $\approx 0.47$ bits per biased flip but a full $1$ bit per fair flip. **Low entropy = low surprise = short codes.** A model that has learned its data well drives its own predictive entropy down — which is exactly why a language model's *bits-per-token* is a headline metric (L16–L18).

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
- **Interactive explainers** (numerical tricks, softmax & temperature, MLE-vs-MAP coin, information theory, multivariate normal) — *ES 667 Interactive Lab*, N. Batra
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
