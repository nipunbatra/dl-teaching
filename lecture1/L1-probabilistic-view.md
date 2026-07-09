---
marp: true
theme: metropolis
paginate: true
math: mathjax
footer: 'ES 667 · Deep Learning · N. Batra'
---

<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: '' -->

# Why These Losses?

## A probabilistic refresher for Deep Learning

<div class="meta">

**Lecture 1 · ES 667: Deep Learning**
Prof. Nipun Batra · IIT Gandhinagar

</div>

![w:300px](figures/net_to_distribution.svg)

<!-- A neural network ending in a probability distribution, not merely a scalar. Time: 1 min -->

---

# Three familiar equations

<div class="big-eq">

$$\mathcal L_{\text{reg}} = \frac1n\sum_i (y_i-\hat y_i)^2$$

$$\mathcal L_{\text{class}} = -\frac1n\sum_i \log p_\theta(y_i\mid x_i)$$

$$\mathcal L_{\text{total}} = \mathcal L_{\text{data}} + \lambda\lVert\theta\rVert_2^2$$

</div>

<div class="box">

**Why exactly these three expressions?**

</div>

<!-- Time: 1.5 min -->

---

# Were these arbitrary choices? <span class="tag q">Q</span>

Which is it?

- Historical convention?
- Convenient gradients?
- Empirically successful?
- **Derived from probability?**
- All of the above?

*(Show of hands / quick poll.)*

<div class="alert">

**Reveal.** They are primarily **consequences of probabilistic assumptions.**

</div>

<!-- Time: 1.5 min -->

---

# The map for today <span class="tag v">V</span>

<div class="cols">
<div>

**Assumption about outputs**
$$\downarrow$$
$$p_\theta(y\mid x)$$
$$\downarrow \quad \text{likelihood of the data}$$
$$\downarrow \quad \text{negative log-likelihood}$$
$$\textbf{training loss}$$

</div>
<div>

**Assumption about parameters**
$$\downarrow$$
$$p(\theta)$$
$$\downarrow \quad \text{Bayes' rule}$$
$$\downarrow \quad \text{MAP}$$
$$\textbf{regularization}$$

</div>
</div>

<!-- Time: 1 min -->

---

<!-- _class: section -->

<div class="kicker">Part II</div>

# Minimal probability revision

*Each concept appears only because it immediately produces a DL objective.*

---

# A model should describe uncertainty

Contrast a point prediction with a **distribution**:

<div class="cols">
<div>

$$\hat y = f_\theta(x)$$

a single number

</div>
<div>

$$Y\mid X{=}x \sim p_\theta(y\mid x)$$

a full distribution

</div>
</div>

- regression network → a **Gaussian mean**;
- classifier → **categorical** probabilities;
- language model → a categorical distribution over **tokens**.

<div class="result">

A neural network parameterizes a probability distribution.

</div>

<!-- Time: 2 min -->

---

# Density is not probability <span class="tag v">V</span>

For a continuous $Y$, probability comes from **area**:

$$P(a\le Y\le b) = \int_a^b p(y)\,dy$$

<div class="alert">

$$p(y) \neq P(Y=y)$$

A density can **exceed 1** — only *integrals* are probabilities.

</div>

![w:560px](figures/density_area.svg)

<!-- No CDFs, no measure theory. Time: 1.5 min -->

---

# Uniform distribution: support matters <span class="tag v">V</span>

<div class="cols wide-left">
<div>

$$Y\sim \mathcal U(a,b)$$

$$p(y)=\begin{cases}\dfrac{1}{b-a}, & a\le y\le b,\\[4pt] 0, & \text{otherwise.}\end{cases}$$

**Q.** What is the NLL *outside* $[a,b]$?
**A.** $-\log 0 = +\infty$.

</div>
<div>

![w:360px](figures/uniform.svg)

</div>
</div>

<div class="box">

A likelihood can impose a **hard constraint**.

</div>

<!-- Time: 1.5 min -->

---

# Gaussian distribution <span class="tag v">V</span>

$$Y\sim\mathcal N(\mu,\sigma^2)\qquad p(y)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\Big[-\frac{(y-\mu)^2}{2\sigma^2}\Big]$$

![w:640px](figures/gaussian_params.svg)

Only two knobs: **location** $\mu$ and **scale** $\sigma$.

<!-- Time: 1.5 min -->

---

# Why the Gaussian produces a square <span class="tag d">D</span>

Take the negative log:

$$-\log p(y) = \underbrace{\frac{(y-\mu)^2}{2\sigma^2}}_{\text{depends on }\mu} + \underbrace{\tfrac12\log(2\pi\sigma^2)}_{\text{constant in }\mu}$$

The only $\mu$-dependent term is the **squared error** $(y-\mu)^2$.

![w:620px](figures/gaussian_to_square.svg)

*Gaussian density → log → negative log → squared error.*

<!-- Time: 2 min -->

---

# Multivariate Gaussian geometry <span class="tag v">V</span>

$$\theta\sim\mathcal N(\mu,\Sigma)\qquad -\log p(\theta)=\tfrac12(\theta-\mu)^\top\Sigma^{-1}(\theta-\mu)+C$$

![w:840px](figures/mvn_triptych.svg)

<div class="box">

Covariance says **which directions in parameter space are plausible.** (We'll reuse this for structured regularizers.)

</div>

<!-- Σ=τ²I circles · diagonal Σ axis-aligned ellipses · full Σ rotated ellipses. Time: 2 min -->

---

# Independence turns products into sums

Conditionally independent observations:

$$p(\mathcal D\mid\theta)=\prod_{i=1}^{n}p_\theta(y_i\mid x_i)$$

Take logs — the product becomes a **sum**:

$$\log p(\mathcal D\mid\theta)=\sum_{i=1}^{n}\log p_\theta(y_i\mid x_i)$$

<div class="box">

This is why a loss is an **average over examples** — and why minibatches work: $\;\dfrac1{|B|}\sum_{i\in B}\ell_i$.

</div>

<!-- Time: 1.5 min -->

---

<!-- _class: section -->

<div class="kicker">Part III</div>

# Maximum likelihood → training losses

---

# Likelihood: same expression, two viewpoints <span class="tag q">Q</span>

The one model $p_\theta(y\mid x)$, read two ways:

<div class="cols">
<div>

**Fix $\theta$, vary $y$**
→ a **probability distribution** over outcomes.

</div>
<div>

**Fix observed $y$, vary $\theta$**
→ a **likelihood** over parameters.

</div>
</div>

<div class="alert">

The likelihood is *not* "the probability that $\theta$ is correct."

</div>

<!-- Time: 1.5 min -->

---

# Maximum likelihood estimation

$$\hat\theta_{\text{MLE}}=\arg\max_\theta\; p(\mathcal D\mid\theta)$$

<div class="box">

**Interpretation.** Choose the parameters under which the observed dataset would be **least surprising**.

</div>

<!-- Time: 1.5 min -->

---

# From MLE to negative log-likelihood <span class="tag d">D</span>

$$\arg\max_\theta \prod_i p_\theta(y_i\mid x_i)
= \arg\max_\theta \sum_i \log p_\theta(y_i\mid x_i)$$

$$= \arg\min_\theta\; -\sum_i \log p_\theta(y_i\mid x_i)$$

<div class="result">

$\ell_i(\theta) = -\log p_\theta(y_i\mid x_i)$

</div>

<!-- Time: 2 min -->

---

# NLL is empirical risk

**ML notation** — empirical risk:

$$\widehat R(\theta)=\frac1n\sum_i \ell\big(y_i, f_\theta(x_i)\big)$$

**Probabilistic reading** of the per-example loss:

$$\ell\big(y_i, f_\theta(x_i)\big) = -\log p_\theta(y_i\mid x_i)$$

<div class="result">

Choosing a loss = choosing a probabilistic observation model.

</div>

<!-- Time: 2 min -->

---

<!-- _class: section -->

<div class="kicker">Part IV</div>

# Regression — where does MSE come from?

---

# Regression as signal plus noise

$$y_i = f_\theta(x_i) + \epsilon_i,\qquad \epsilon_i\sim\mathcal N(0,\sigma^2)$$

Therefore the conditional output distribution is

$$Y_i\mid x_i,\theta \sim \mathcal N\big(f_\theta(x_i),\,\sigma^2\big)$$

<!-- Time: 1.5 min -->

---

# What the assumption means visually <span class="tag v">V</span>

![w:720px](figures/regression_conditionals.svg)

<div class="box">

The Gaussian noise lives in the **output direction**, conditional on $x$ — a vertical bell at every input.

</div>

<!-- Time: 1.5 min -->

---

# Gaussian likelihood <span class="tag d">D</span>

$$p_\theta(y_i\mid x_i)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\Big[-\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}\Big]$$

Negative log of a single example:

$$-\log p_\theta(y_i\mid x_i)=\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}+C$$

<!-- Time: 2 min -->

---

# MSE is Gaussian MLE

Sum over the dataset:

$$-\log p(\mathcal D\mid\theta)=\frac{1}{2\sigma^2}\sum_i\big(y_i-f_\theta(x_i)\big)^2 + C$$

With $\sigma$ fixed, the minimizer is

$$\hat\theta_{\text{MLE}}=\arg\min_\theta\sum_i\big(y_i-f_\theta(x_i)\big)^2$$

<div class="result">

MSE = Gaussian NLL, up to constants.

</div>

<!-- Time: 2 min -->

---

# What role does $\sigma^2$ play? <span class="tag opt">optional</span>

$$\ell_i=\frac{(y_i-\mu_i)^2}{2\sigma^2}+\tfrac12\log\sigma^2 + C$$

- small $\sigma$: errors penalized **strongly**;
- large $\sigma$: errors deemed **less surprising**;
- a predicted $\sigma_i$ needs the $\tfrac12\log\sigma_i^2$ term — it stops the model from claiming infinite uncertainty.

<div class="box">

**Heteroscedastic regression:** let the network output both, $(\mu_i,\log\sigma_i^2)=f_\theta(x_i)$.

</div>

<!-- Time: 2 min -->

---

# Different noise, different loss <span class="tag v">V</span>

<div class="cols wide-right">
<div>

| Noise model | NLL |
|---|---|
| Gaussian | $r^2$ |
| Laplace | $\lvert r\rvert$ |
| Uniform | hard interval |
| Student-$t$ | robust, slow-growing |

$r = y-\hat y$

</div>
<div>

![w:400px](figures/residual_losses.svg)

</div>
</div>

<!-- Time: 1.5 min -->

---

# Interactive: likelihood → loss <span class="tag i">I</span>

<div class="interactive">

**🎛 `likelihood-to-loss.html`** — three synchronized panels: the **noise density** $p(r)$, the **loss** $-\log p(r)$, and **data with a fitted line**.
*Controls:* Gaussian / Laplace / Uniform / Student-$t$ · noise scale · slope & intercept · **add an outlier** · per-point contributions.

</div>

**Ask:** Which model reacts most to an outlier? Why does Uniform give an infinite barrier? Which loss is robust?

<!-- Time: 3 min -->

---

<!-- _class: section -->

<div class="kicker">Part V</div>

# Classification and cross-entropy

---

# Classification should output probabilities

For $K$ classes:

$$p_\theta(y=k\mid x)\ge 0,\qquad \sum_{k=1}^{K}p_\theta(y=k\mid x)=1$$

![w:640px](figures/softmax_pipeline.svg)

**logits** $z_k\in\mathbb R$ → **probabilities** $p_k\in[0,1]$.

<!-- Time: 1 min -->

---

# Binary classification: Bernoulli model

$$Y\mid x\sim\text{Bernoulli}\big(p_\theta(x)\big)$$

Compact one-line form:

$$p(y\mid x,\theta)=p^{\,y}(1-p)^{1-y}$$

- $y=1\Rightarrow p(y)=p$
- $y=0\Rightarrow p(y)=1-p$

<!-- Time: 1.5 min -->

---

# Bernoulli NLL gives BCE <span class="tag d">D</span>

$$-\log p(y\mid x,\theta) = -\log\big[p^{\,y}(1-p)^{1-y}\big]$$

$$= -\,y\log p-(1-y)\log(1-p)$$

<div class="result">

Binary cross-entropy = Bernoulli NLL.

</div>

<!-- Time: 2 min -->

---

# Why use logits?

The network emits an unbounded score $z=f_\theta(x)\in\mathbb R$; the **sigmoid** maps it to a probability:

$$p=\sigma(z)=\frac{1}{1+e^{-z}}$$

<div class="cols wide-right">
<div>

$z\to-\infty:\ p\to 0$
$z=0:\ p=0.5$
$z\to+\infty:\ p\to 1$

</div>
<div>

![w:360px](figures/sigmoid.svg)

</div>
</div>

*In practice sigmoid + BCE are fused for numerical stability.*

<!-- Time: 1.5 min -->

---

# Multi-class: categorical model

$$Y\mid x\sim\text{Categorical}(p_1,\dots,p_K)$$

For a one-hot target $y$:

$$p(y\mid x,\theta)=\prod_{k=1}^{K} p_k^{\,y_k}$$

<!-- Time: 1.5 min -->

---

# From logits to softmax

$$p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}$$

Two guaranteed properties: $\;p_k>0\;$ and $\;\sum_k p_k=1$.

![w:520px](figures/softmax_bars.svg)

**Shift invariance:** $\text{softmax}(z)=\text{softmax}(z+c\mathbf 1)$.

<!-- Time: 1.5 min -->

---

# Categorical NLL gives cross-entropy <span class="tag d">D</span>

$$-\log p(y\mid x,\theta)=-\log\prod_k p_k^{\,y_k}=-\sum_k y_k\log p_k$$

Because $y$ is one-hot, only the true class survives:

$$\mathcal L = -\log p_{\text{true class}}$$

<!-- Time: 2 min -->

---

# Confident mistakes are expensive <span class="tag v">V</span>

$$\ell(p_y)=-\log p_y$$

<div class="cols wide-right">
<div>

$p_y=0.9\Rightarrow \ell\approx 0.105$
$p_y=0.1\Rightarrow \ell\approx 2.303$
$p_y=0.001\Rightarrow \ell\approx 6.908$

</div>
<div>

![w:380px](figures/neglog_curve.svg)

</div>
</div>

**Q.** Why should a *confidently wrong* prediction be punished so hard?

<!-- Time: 1.5 min -->

---

# A remarkably simple gradient <span class="tag d">D</span>

For softmax followed by cross-entropy:

$$\frac{\partial\mathcal L}{\partial z_k}=p_k-y_k,\qquad \nabla_z\mathcal L = p - y$$

![w:560px](figures/gradient_bars.svg)

*Predicted $p$, target $y$, and their difference — a preview of backprop.*

<!-- Time: 2 min -->

---

# Interactive: logits, softmax, cross-entropy <span class="tag i">I</span>

<div class="interactive">

**🎛 `softmax-cross-entropy.html`** — controls $z_1,z_2,z_3$, the true class, and temperature $T$; presets for **correct & confident / correct & unsure / wrong & unsure / wrong & confident**.
Shows the chain $\;z\to\text{softmax}(z/T)\to -\log p_y\to p-y$.

</div>

**Ask students to predict the loss before it is revealed.**

<!-- Time: 3 min -->

---

# Why not MSE for classification? <span class="tag opt">optional</span>

MSE is not *impossible* — it corresponds to the **wrong observation model** for class indicators.

<div class="cols">
<div>

**MSE assumes**
$$Y_k=p_k+\epsilon_k,\ \epsilon_k\sim\mathcal N(0,\sigma^2)$$

</div>
<div>

**Reality is**
$$Y\sim\text{Categorical}(p)$$

</div>
</div>

<div class="box">

Cross-entropy also gives **stronger gradients** on confident errors (where MSE's gradient vanishes).

</div>

<!-- Time: 1.5 min -->

---

<!-- _class: section -->

<div class="kicker">Part VI</div>

# Bayes' rule, directly in classification

---

# Bayes' rule as a classifier

$$p(y=k\mid x)=\frac{p(x\mid y=k)\,p(y=k)}{p(x)}$$

Prediction ignores the constant denominator:

$$\hat y=\arg\max_k\; p(x\mid y=k)\,p(y=k)$$

- $p(y=k)$ — class **prior**
- $p(x\mid y=k)$ — class-conditional **likelihood**
- $p(y=k\mid x)$ — **posterior** class probability

<!-- Time: 2 min -->

---

# A 1-D generative classifier <span class="tag v">V</span>

$$X\mid Y{=}0\sim\mathcal N(\mu_0,\sigma_0^2),\qquad X\mid Y{=}1\sim\mathcal N(\mu_1,\sigma_1^2)$$

![w:660px](figures/bayes_classifier.svg)

*Two class-conditional densities, a test point $x^\star$, and its likelihood under each class (equal priors to start).*

<!-- Time: 2 min -->

---

# Priors move the decision boundary <span class="tag v">V</span>

<div class="cols">
<div>

**Balanced** $\;p(Y{=}1)=0.5$

</div>
<div>

**Rare** $\;p(Y{=}1)=0.1$

</div>
</div>

![w:620px](figures/priors_shift_boundary.svg)

<div class="result">

posterior $\propto$ likelihood $\times$ prior

</div>

*Imbalanced classes · disease prevalence · spam · rare-event detection.*

<!-- Time: 2 min -->

---

# Interactive: Bayes classifier <span class="tag i">I</span>

<div class="interactive">

**🎛 `bayes-classifier.html`** — controls the means $\mu_0,\mu_1$, the variances, the **prior** $p(Y{=}1)$, and a test point $x^\star$.
Shows $p(x\mid Y{=}0)$, $p(x\mid Y{=}1)$, the posterior $p(Y{=}1\mid x)$, and the decision boundary together.

</div>

**Try:** start balanced → drop the positive prior to $0.1$ → why does the boundary move? Then widen one variance.

<!-- The class-prior slider is the most important control. Time: 3 min -->

---

<!-- _class: section -->

<div class="kicker">Part VII</div>

# Priors, MAP and regularization

---

# The limitation of MLE

MLE asks only: *how well does $\theta$ explain the data?*

![w:520px](figures/overfit_polynomial.svg)

<div class="box">

Many parameters fit a small dataset. **Which should we prefer?**

</div>

<!-- Time: 1 min -->

---

# Put a prior over parameters

$$\theta\sim p(\theta)$$

*Before seeing this dataset, which parameter values are plausible?*

- small weights;
- sparse weights;
- smooth functions;
- correlated parameters.

<div class="box">

A prior is an **inductive bias**.

</div>

<!-- Time: 1.5 min -->

---

# Bayes' rule over parameters

$$p(\theta\mid\mathcal D)=\frac{p(\mathcal D\mid\theta)\,p(\theta)}{p(\mathcal D)}$$

![w:600px](figures/prior_likelihood_posterior.svg)

$$p(\mathcal D)=\int p(\mathcal D\mid\theta)\,p(\theta)\,d\theta \quad\text{(the evidence — we stop here).}$$

<!-- Time: 1.5 min -->

---

# MAP estimation <span class="tag d">D</span>

$$\hat\theta_{\text{MAP}}=\arg\max_\theta p(\theta\mid\mathcal D)=\arg\max_\theta p(\mathcal D\mid\theta)\,p(\theta)$$

Take negative logs:

$$\hat\theta_{\text{MAP}}=\arg\min_\theta\big[-\log p(\mathcal D\mid\theta)-\log p(\theta)\big]$$

<div class="result">

data loss $+$ regularization

</div>

<!-- Time: 2 min -->

---

# Gaussian prior produces $L_2$ <span class="tag d">D</span>

Assume $\theta\sim\mathcal N(0,\tau^2 I)$. Then

$$-\log p(\theta)=\frac{1}{2\tau^2}\theta^\top\theta + C$$

$$\Rightarrow\quad \mathcal L_{\text{MAP}}=-\log p(\mathcal D\mid\theta)+\lambda\lVert\theta\rVert_2^2,\qquad \lambda\propto\frac{1}{\tau^2}$$

<div class="box">

**narrow** prior ⇒ **stronger** regularization; **broad** prior ⇒ **weaker**.

</div>

<!-- Time: 2 min -->

---

# A general Gaussian prior → structured penalty <span class="tag opt">optional</span>

For $\theta\sim\mathcal N(0,\Sigma)$ the penalty is $\;\tfrac12\theta^\top\Sigma^{-1}\theta$.

![w:640px](figures/mvn_triptych.svg)

- isotropic $\Sigma$ → ordinary $L_2$;
- unequal variances → penalize parameters **differently**;
- correlations → penalize particular **directions**.

*This is the payoff for revising the multivariate Gaussian.*

<!-- Time: 1.5 min -->

---

# Laplace prior produces $L_1$ <span class="tag opt">optional</span>

$$p(\theta_j)=\frac{1}{2b}\exp\!\Big(-\frac{\lvert\theta_j\rvert}{b}\Big)\ \Rightarrow\ -\log p(\theta)=\frac1b\sum_j\lvert\theta_j\rvert + C$$

<div class="result">

Laplace prior ⇒ $L_1$

</div>

Heavier tails at zero → **sparsity** (the lasso geometry, if time permits).

<!-- Time: 1 min -->

---

# Interactive: likelihood, prior, MAP <span class="tag i">I</span>

<div class="interactive">

**🎛 `map-regularization.html`** — a two-parameter linear model so contours are visible. Shows **likelihood**, **prior**, and **posterior** contours with the **MLE**, **MAP**, and zero points.
*Controls:* amount of data · noise · prior variance · Gaussian vs Laplace · prior correlation.

</div>

**Ask:** more data? narrower prior? why does MAP → MLE as data grows? a correlated prior?

<!-- Time: 3 min -->

---

<!-- _class: section -->

<div class="kicker">Part VIII</div>

# Full Bayes and the DL objective

---

# MAP is still one parameter vector

<div class="cols">
<div>

**MLE** $\hat\theta_{\text{MLE}}$
one point

</div>
<div>

**MAP** $\hat\theta_{\text{MAP}}$
one point

</div>
</div>

**Full Bayes** keeps the whole posterior $p(\theta\mid\mathcal D)$ — a **distribution**, not a point.

<!-- Time: 1 min -->

---

# Posterior predictive distribution

$$p(y^\star\mid x^\star,\mathcal D)=\int p(y^\star\mid x^\star,\theta)\,p(\theta\mid\mathcal D)\,d\theta$$

![w:600px](figures/posterior_samples.svg)

<div class="box">

Bayesian prediction **averages over plausible models** — and reports uncertainty.

</div>

<!-- Time: 1.5 min -->

---

# Why DL usually uses point estimates <span class="tag opt">optional</span>

A modern network has millions–billions of parameters, so computing and integrating $p(\theta\mid\mathcal D)$ is hard.

Common practical choices:

- MLE · MAP;
- deep **ensembles**;
- **dropout**-based approximations;
- **variational** inference.

<!-- Keep conceptual. Time: 1 min -->

---

# The standard supervised DL objective

<div class="big-eq">

$$\min_\theta\ \underbrace{\frac{1}{|B|}\sum_{i\in B}-\log p_\theta(y_i\mid x_i)}_{\text{likelihood / data fit}}\ +\ \underbrace{\lambda\lVert\theta\rVert_2^2}_{\text{prior / regularization}}$$

</div>

$$\text{probabilistic model} \;+\; \text{neural network} \;+\; \text{gradient optimization}$$

<div class="box">

Deep learning doesn't discard probabilistic ML — it **scales** it with expressive functions and gradient descent.

</div>

<!-- Time: 1.5 min -->

---

# Retrieval summary <span class="tag q">Q</span>

Complete each:

<div class="cols">
<div>

Gaussian likelihood ⇒ ?
Categorical likelihood ⇒ ?
Gaussian prior ⇒ ?
Laplace prior ⇒ ?
MLE + prior ⇒ ?

</div>
<div>

⇒ **MSE**
⇒ **cross-entropy**
⇒ **$L_2$**
⇒ **$L_1$**
⇒ **MAP**

</div>
</div>

<!-- Time: 1.5 min -->

---

<!-- _class: section -->

# Final mental model

<div style="font-size:26px; line-height:1.6;">

See a **loss** → ask *"what likelihood produced it?"*
See a **regularizer** → ask *"what prior produced it?"*
See a **prediction** → ask *"point estimate or posterior?"*

</div>

**Next lecture:** computation graphs, gradients, and backpropagation.

<!-- Time: 1 min -->
