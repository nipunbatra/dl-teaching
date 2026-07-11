---
title: "Lecture 1: Detailed slide-level plan"
source: "https://chatgpt.com/share/6a50728a-9720-83ee-a33d-5f2e3ff744fe"
status: "Imported from the finalized shared-course discussion"
---

# Lecture 1: Detailed slide-level plan

## Title

**Why These Losses? A Probabilistic View of Deep Learning**

Central narrative:

\[
\boxed{
\text{distribution}
\rightarrow
\text{likelihood}
\rightarrow
\text{loss}
}
\]

and

\[
\boxed{
\text{prior}
\rightarrow
\text{MAP}
\rightarrow
\text{regularization}
}
\]

Target: **47 slides, 80 minutes**.

Legend:

- **[Q]** audience question
- **[V]** visual slide
- **[D]** derivation
- **[I]** interactive

---

## Part I — Why are we revisiting probability?

### Slide 1 — Title

**Why These Losses?**

Subtitle:

> A probabilistic refresher for Deep Learning

Visual: neural network ending in a probability distribution, rather than merely a scalar prediction.

**Time:** 1 min

---

### Slide 2 — Three familiar equations

Show only:

\[
\mathcal L_{\text{reg}}
=
\frac1n\sum_i(y_i-\hat y_i)^2
\]

\[
\mathcal L_{\text{class}}
=
-\frac1n\sum_i\log p_\theta(y_i\mid x_i)
\]

\[
\mathcal L_{\text{total}}
=
\mathcal L_{\text{data}}
+
\lambda\|\theta\|_2^2
\]

Question:

> Why exactly these three expressions?

**Time:** 1.5 min

---

### Slide 3 — Were these arbitrary choices? [Q]

Show possible answers:

- Historical convention?
- Convenient gradients?
- Empirically successful?
- Derived from probability?
- All of the above?

Conduct a quick show-of-hands or online poll.

Reveal:

> They are primarily consequences of probabilistic assumptions.

**Time:** 1.5 min

---

### Slide 4 — The map for today [V]

Use a flow diagram:

```text
Assumption about outputs
          ↓
 pθ(y | x)
          ↓
 Likelihood of observed data
          ↓
 Negative log-likelihood
          ↓
 Training loss
```

Beside it:

```text
Assumption about parameters
          ↓
       p(θ)
          ↓
      Bayes rule
          ↓
         MAP
          ↓
   Regularization
```

**Time:** 1 min

---

# Part II — Minimal probability revision

Do not call this a general probability revision. Introduce each concept only because it will immediately produce a DL objective.

---

### Slide 5 — A model should describe uncertainty

Contrast:

\[
\hat y=f_\theta(x)
\]

with

\[
Y\mid X=x
\sim p_\theta(y\mid x)
\]

Examples:

- regression network predicts a Gaussian mean;
- classifier predicts categorical probabilities;
- language model predicts a categorical distribution over tokens.

Main message:

> A neural network typically parameterizes a probability distribution.

**Time:** 2 min

---

### Slide 6 — Density is not probability [V]

For continuous \(Y\):

\[
P(a\le Y\le b)
=
\int_a^b p(y)\,dy
\]

Emphasize:

\[
p(y)\
eq P(Y=y).
\]

Diagram: density curve with shaded interval.

Do not discuss CDFs or measure theory.

**Time:** 1.5 min

---

### Slide 7 — Uniform distribution: support matters [V]

\[
Y\sim\mathcal U(a,b)
\]

\[
p(y)=
\begin{cases}
\dfrac1{b-a}, & a\le y\le b,\\
0, & \text{otherwise}.
\end{cases}
\]

Ask:

> What happens to the negative log-likelihood outside the interval?

Answer:

\[
-\log 0=+\infty.
\]

Message:

> A likelihood can impose a hard constraint.

**Time:** 1.5 min

---

### Slide 8 — Gaussian distribution [V]

\[
Y\sim\mathcal N(\mu,\sigma^2)
\]

\[
p(y)
=
\frac1{\sqrt{2\pi\sigma^2}}
\exp\left[
-\frac{(y-\mu)^2}{2\sigma^2}
\right].
\]

Diagram: three curves demonstrating:

- changing \(\mu\);
- changing \(\sigma\).

Focus only on location and scale.

**Time:** 1.5 min

---

### Slide 9 — Why the Gaussian produces a square [D]

Take the negative log:

\[
-\log p(y)
=
\frac{(y-\mu)^2}{2\sigma^2}
+
\frac12\log(2\pi\sigma^2).
\]

Highlight the only term depending on \(\mu\):

\[
(y-\mu)^2.
\]

Animation/reveal sequence:

1. Gaussian density;
2. logarithm;
3. negative logarithm;
4. squared error.

**Time:** 2 min

---

### Slide 10 — Multivariate Gaussian geometry [V]

\[
\theta
\sim
\mathcal N(\mu,\Sigma)
\]

\[
-\log p(\theta)
=
\frac12
(\theta-\mu)^\top
\Sigma^{-1}
(\theta-\mu)
+C.
\]

Show three contour diagrams:

1. \(\Sigma=\tau^2I\): circles;
2. diagonal \(\Sigma\): axis-aligned ellipses;
3. full \(\Sigma\): rotated ellipses.

Main message:

> Covariance specifies which directions in parameter space are plausible.

**Time:** 2 min

---

### Slide 11 — Independence turns products into sums

Assume conditionally independent observations:

\[
p(\mathcal D\mid\theta)
=
\prod_{i=1}^{n}
p_\theta(y_i\mid x_i).
\]

Taking logs:

\[
\log p(\mathcal D\mid\theta)
=
\sum_{i=1}^{n}
\log p_\theta(y_i\mid x_i).
\]

Connect this to minibatch losses:

\[
\frac1B\sum_{i\in\mathcal B}\ell_i.
\]

**Time:** 1.5 min

---

# Part III — Maximum likelihood and training losses

### Slide 12 — Likelihood: same expression, different viewpoint

Probability model:

\[
p_\theta(y\mid x)
\]

- Fix \(\theta\), vary \(y\): probability distribution.
- Fix observed \(y\), vary \(\theta\): likelihood.

Use a two-column diagram.

This distinction is worth making explicitly.

**Time:** 1.5 min

---

### Slide 13 — Maximum likelihood estimation

\[
\hat\theta_{\text{MLE}}
=
\arg\max_\theta
p(\mathcal D\mid\theta)
\]

Interpretation:

> Choose parameters under which the observed dataset would not be surprising.

Avoid saying that likelihood is “the probability that the parameter is correct.”

**Time:** 1.5 min

---

### Slide 14 — From MLE to negative log-likelihood [D]

\[
\arg\max_\theta
\prod_i p_\theta(y_i\mid x_i)
\]

\[
=
\arg\max_\theta
\sum_i\log p_\theta(y_i\mid x_i)
\]

\[
=
\arg\min_\theta
-\sum_i\log p_\theta(y_i\mid x_i).
\]

Highlight:

\[
\boxed{
\ell_i(\theta)
=
-\log p_\theta(y_i\mid x_i)
}
\]

**Time:** 2 min

---

### Slide 15 — NLL is empirical risk

Standard ML notation:

\[
\hat R(\theta)
=
\frac1n
\sum_i
\ell(y_i,f_\theta(x_i)).
\]

Probabilistic interpretation:

\[
\ell(y_i,f_\theta(x_i))
=
-\log p_\theta(y_i\mid x_i).
\]

Message:

> Selecting a loss means selecting a probabilistic observation model.

**Time:** 2 min

---

# Part IV — Regression: where does MSE come from?

### Slide 16 — Regression as signal plus noise

\[
y_i
=
f_\theta(x_i)+\epsilon_i
\]

Assume:

\[
\epsilon_i\sim\mathcal N(0,\sigma^2).
\]

Therefore:

\[
Y_i\mid x_i,\theta
\sim
\mathcal N(f_\theta(x_i),\sigma^2).
\]

**Time:** 1.5 min

---

### Slide 17 — What the assumption means visually [V]

Diagram:

- nonlinear regression curve \(f_\theta(x)\);
- vertical Gaussian bells at four \(x\)-locations;
- observations scattered vertically around the curve.

Explicitly say:

> Gaussian noise is in the output direction, conditional on \(x\).

**Time:** 1.5 min

---

### Slide 18 — Gaussian likelihood [D]

\[
p_\theta(y_i\mid x_i)
=
\frac1{\sqrt{2\pi\sigma^2}}
\exp\left[
-\frac{
(y_i-f_\theta(x_i))^2
}{
2\sigma^2
}
\right].
\]

Then:

\[
-\log p_\theta(y_i\mid x_i)
=
\frac{
(y_i-f_\theta(x_i))^2
}{
2\sigma^2
}
+C.
\]

**Time:** 2 min

---

### Slide 19 — MSE is Gaussian MLE

Summing over examples:

\[
-\log p(\mathcal D\mid\theta)
=
\frac1{2\sigma^2}
\sum_i
(y_i-f_\theta(x_i))^2
+C.
\]

If \(\sigma\) is fixed:

\[
\hat\theta_{\text{MLE}}
=
\arg\min_\theta
\sum_i
(y_i-f_\theta(x_i))^2.
\]

Large boxed conclusion:

\[
\boxed{\text{MSE}=\text{Gaussian NLL up to constants}}
\]

**Time:** 2 min

---

### Slide 20 — What role does \(\sigma^2\) play?

\[
\ell_i
=
\frac{(y_i-\mu_i)^2}{2\sigma^2}
+
\frac12\log\sigma^2
+C.
\]

Discuss:

- small \(\sigma\): errors are penalized strongly;
- large \(\sigma\): errors are considered less surprising;
- if the model predicts \(\sigma_i\), the \(\log\sigma_i^2\) term prevents infinite uncertainty.

Briefly introduce heteroscedastic regression:

\[
(\mu_i,\log\sigma_i^2)=f_\theta(x_i).
\]

**Time:** 2 min

---

### Slide 21 — Different noise, different loss [V]

Place distributions over residual \(r=y-\hat y\) above their losses.

| Noise model | Negative log-likelihood |
|---|---|
| Gaussian | \(r^2\) |
| Laplace | \(|r|\) |
| Uniform | hard interval constraint |
| Student-\(t\) | slowly growing robust loss |

Plot all loss curves on the same axes.

**Time:** 1.5 min

---

### Slide 22 — Interactive: likelihood to loss [I]

Demo layout: three synchronized panels.

1. **Noise density** \(p(r)\)
2. **Loss** \(-\log p(r)\)
3. **Data and fitted regression line**

Controls:

- Gaussian / Laplace / Uniform / Student-\(t\);
- noise scale;
- slope and intercept;
- add an outlier;
- show individual point contributions.

Questions to ask:

1. Which model reacts most to an outlier?
2. Why does Uniform produce an infinite barrier?
3. Which loss would be robust?

**Time:** 3 min

---

# Part V — Classification and cross-entropy

### Slide 23 — Classification should output probabilities

For \(K\) classes:

\[
p_\theta(y=k\mid x),
\qquad
\sum_{k=1}^{K}p_\theta(y=k\mid x)=1.
\]

Diagram:

```text
input → neural network → logits → probabilities
```

Distinguish:

- logits \(z_k\in\mathbb R\);
- probabilities \(p_k\in[0,1]\).

**Time:** 1 min

---

### Slide 24 — Binary classification: Bernoulli model

\[
Y\mid x
\sim
\operatorname{Bernoulli}(p_\theta(x)).
\]

Compact representation:

\[
p(y\mid x,\theta)
=
p^y(1-p)^{1-y}.
\]

Verify:

- \(y=1\Rightarrow p(y)=p\);
- \(y=0\Rightarrow p(y)=1-p\).

**Time:** 1.5 min

---

### Slide 25 — Bernoulli NLL gives BCE [D]

\[
-\log p(y\mid x,\theta)
=
-\log\left[p^y(1-p)^{1-y}\right]
\]

\[
=
-y\log p-(1-y)\log(1-p).
\]

Box:

\[
\boxed{\text{Binary cross-entropy}=\text{Bernoulli NLL}}
\]

**Time:** 2 min

---

### Slide 26 — Why use logits?

Network produces:

\[
z=f_\theta(x)\in\mathbb R.
\]

Convert to probability:

\[
p=\sigma(z)=\frac1{1+e^{-z}}.
\]

Diagram showing:

\[
z=-\infty\rightarrow p=0,
\qquad
z=0\rightarrow p=0.5,
\qquad
z=\infty\rightarrow p=1.
\]

Mention that implementations combine sigmoid and BCE for numerical stability.

**Time:** 1.5 min

---

### Slide 27 — Multi-class classification: categorical model

\[
Y\mid x
\sim
\operatorname{Categorical}(p_1,\ldots,p_K).
\]

For one-hot target \(\mathbf y\):

\[
p(\mathbf y\mid x,\theta)
=
\prod_{k=1}^{K}p_k^{y_k}.
\]

**Time:** 1.5 min

---

### Slide 28 — From logits to softmax

\[
p_k
=
\frac{e^{z_k}}
{\sum_j e^{z_j}}.
\]

Explain two properties:

\[
p_k>0,
\qquad
\sum_kp_k=1.
\]

Visual: three logit bars transformed into three probability bars.

Also show shift invariance:

\[
\operatorname{softmax}(z)
=
\operatorname{softmax}(z+c\mathbf1).
\]

**Time:** 1.5 min

---

### Slide 29 — Categorical NLL gives cross-entropy [D]

\[
-\log p(\mathbf y\mid x,\theta)
=
-\log
\prod_k p_k^{y_k}
\]

\[
=
-\sum_k y_k\log p_k.
\]

Because \(\mathbf y\) is one-hot:

\[
\mathcal L=-\log p_{\text{true class}}.
\]

**Time:** 2 min

---

### Slide 30 — Confident mistakes are expensive [V]

Plot:

\[
\ell(p_y)=-\log p_y.
\]

Mark:

\[
\begin{aligned}
p_y=0.9 &\Rightarrow \ell\approx0.105,\\
p_y=0.1 &\Rightarrow \ell\approx2.303,\\
p_y=0.001 &\Rightarrow \ell\approx6.908.
\end{aligned}
\]

Ask:

> Why should a confidently wrong prediction receive a large penalty?

**Time:** 1.5 min

---

### Slide 31 — A remarkably simple gradient [D]

For softmax followed by CE:

\[
\frac{\partial\mathcal L}{\partial z_k}
=
p_k-y_k.
\]

Vector form:

\[
\
abla_z\mathcal L
=
\mathbf p-\mathbf y.
\]

Visualize two bar charts:

- predicted probabilities \(\mathbf p\);
- target \(\mathbf y\);

and their difference.

This also previews backpropagation.

**Time:** 2 min

---

### Slide 32 — Interactive: logits, softmax and CE [I]

Controls:

- \(z_1,z_2,z_3\);
- select true class;
- temperature \(T\);
- preset buttons.

Presets:

- correct and confident;
- correct but uncertain;
- wrong but uncertain;
- wrong and confident.

Display:

\[
z
\rightarrow
\operatorname{softmax}(z/T)
\rightarrow
-\log p_y
\rightarrow
p-y.
\]

Ask students to predict the loss before showing it.

**Time:** 3 min

---

### Slide 33 — Why not MSE for classification?

Avoid claiming MSE is impossible.

Say:

> MSE can be used, but it corresponds to an inappropriate Gaussian observation model for class indicators.

Contrast:

\[
Y_k=p_k+\epsilon_k,
\qquad
\epsilon_k\sim\mathcal N(0,\sigma^2)
\]

with:

\[
Y\sim\operatorname{Categorical}(\mathbf p).
\]

Also mention that CE generally supplies more useful gradients for confident classification errors.

**Time:** 1.5 min

---

# Part VI — Bayes rule directly in classification

### Slide 34 — Bayes rule as a classifier

\[
p(y=k\mid x)
=
\frac{
p(x\mid y=k)p(y=k)
}{
p(x)
}.
\]

For prediction:

\[
\hat y
=
\arg\max_k
p(x\mid y=k)p(y=k).
\]

Label:

- \(p(y=k)\): class prior;
- \(p(x\mid y=k)\): class-conditional likelihood;
- \(p(y=k\mid x)\): posterior class probability.

**Time:** 2 min

---

### Slide 35 — A one-dimensional generative classifier [V]

Two classes:

\[
X\mid Y=0\sim\mathcal N(\mu_0,\sigma_0^2)
\]

\[
X\mid Y=1\sim\mathcal N(\mu_1,\sigma_1^2).
\]

Show:

- two class-conditional density curves;
- an observed \(x^\star\);
- likelihood of \(x^\star\) under each class.

Initially use equal priors.

**Time:** 2 min

---

### Slide 36 — Priors move the decision boundary [V]

Compare:

\[
p(Y=0)=p(Y=1)=0.5
\]

with:

\[
p(Y=1)=0.1.
\]

Show that the same likelihoods can produce a different posterior and decision boundary.

Main message:

\[
\text{posterior}
\propto
\text{likelihood}
\times
\text{prior}.
\]

Connect to:

- imbalanced classes;
- disease prevalence;
- spam detection;
- rare-event detection.

**Time:** 2 min

---

### Slide 37 — Interactive: Bayes classifier [I]

Controls:

- class means \(\mu_0,\mu_1\);
- class variances;
- prior \(p(Y=1)\);
- test point \(x^\star\).

Display simultaneously:

1. \(p(x\mid Y=0)\);
2. \(p(x\mid Y=1)\);
3. \(p(Y=1\mid x)\);
4. decision boundary.

Suggested interaction:

1. begin with equal priors;
2. reduce positive-class prior to \(0.1\);
3. ask why the boundary moves;
4. increase one class variance.

**Time:** 3 min

---

# Part VII — Priors, MAP and regularization

### Slide 38 — The limitation of MLE

MLE asks only:

\[
\text{How well does }\theta\text{ explain the observed data?}
\]

Show a high-degree polynomial interpolating a small dataset.

Question:

> Among many parameters that fit the data, which should we prefer?

**Time:** 1 min

---

### Slide 39 — Put a prior over parameters

\[
\theta\sim p(\theta).
\]

Interpretation:

> Before observing this dataset, which parameter values are plausible?

Examples:

- small weights;
- sparse weights;
- smooth functions;
- correlated parameters.

Clarify that a prior is an inductive bias.

**Time:** 1.5 min

---

### Slide 40 — Bayes rule over parameters

\[
p(\theta\mid\mathcal D)
=
\frac{
p(\mathcal D\mid\theta)p(\theta)
}{
p(\mathcal D)
}.
\]

Use a diagram:

```text
prior p(θ)
     ×
likelihood p(D|θ)
     ↓
posterior p(θ|D)
```

Mention:

\[
p(\mathcal D)
=
\int p(\mathcal D\mid\theta)p(\theta)\,d\theta.
\]

Do not derive the evidence further.

**Time:** 1.5 min

---

### Slide 41 — MAP estimation [D]

\[
\hat\theta_{\text{MAP}}
=
\arg\max_\theta
p(\theta\mid\mathcal D)
\]

\[
=
\arg\max_\theta
p(\mathcal D\mid\theta)p(\theta).
\]

Taking negative logarithms:

\[
\hat\theta_{\text{MAP}}
=
\arg\min_\theta
\left[
-\log p(\mathcal D\mid\theta)
-\log p(\theta)
\right].
\]

Box:

\[
\boxed{
\text{data loss}+\text{regularization}
}
\]

**Time:** 2 min

---

### Slide 42 — Gaussian prior produces \(L_2\) [D]

Assume:

\[
\theta\sim\mathcal N(0,\tau^2I).
\]

Then:

\[
-\log p(\theta)
=
\frac1{2\tau^2}\theta^\top\theta+C.
\]

Therefore:

\[
\mathcal L_{\text{MAP}}
=
-\log p(\mathcal D\mid\theta)
+
\lambda\|\theta\|_2^2.
\]

Connection:

\[
\lambda\propto\frac1{\tau^2}.
\]

Interpret:

- narrow prior \(\Rightarrow\) stronger regularization;
- broad prior \(\Rightarrow\) weaker regularization.

**Time:** 2 min

---

### Slide 43 — General Gaussian prior gives a structured penalty

For:

\[
\theta\sim\mathcal N(0,\Sigma),
\]

the penalty is:

\[
\frac12\theta^\top\Sigma^{-1}\theta.
\]

Use the MVN contour diagrams from Slide 10.

Interpretation:

- isotropic covariance gives ordinary \(L_2\);
- unequal variances penalize parameters differently;
- correlations penalize particular directions.

This gives a reason for having revised the multivariate Gaussian.

**Time:** 1.5 min

---

### Slide 44 — Laplace prior produces \(L_1\)

\[
p(\theta_j)
=
\frac1{2b}
\exp\left(-\frac{|\theta_j|}{b}\right).
\]

Therefore:

\[
-\log p(\theta)
=
\frac1b\sum_j|\theta_j|+C.
\]

Thus:

\[
\boxed{\text{Laplace prior}\Rightarrow L_1}
\]

Mention sparsity, but do not derive the geometry of the lasso unless time permits.

**Time:** 1 min

---

### Slide 45 — Interactive: likelihood, prior and MAP [I]

Use a two-parameter linear model so contours are visible.

Display:

- likelihood contours;
- prior contours;
- posterior contours;
- MLE point;
- MAP point;
- zero vector.

Controls:

- data amount;
- observation noise;
- prior variance;
- prior type: Gaussian/Laplace;
- prior correlation.

Questions:

1. What happens as data increases?
2. What happens as the prior becomes narrower?
3. Why does MAP approach MLE with abundant data?
4. What changes with a correlated prior?

**Time:** 3 min

---

# Part VIII — Full Bayes and the DL objective

### Slide 46 — MAP is still only one parameter vector

MLE:

\[
\hat\theta_{\text{MLE}}
\]

MAP:

\[
\hat\theta_{\text{MAP}}
\]

Full Bayes:

\[
p(\theta\mid\mathcal D).
\]

Visual:

- MLE: one point;
- MAP: one point;
- posterior: an entire distribution.

**Time:** 1 min

---

### Slide 47 — Posterior predictive distribution

\[
p(y_\star\mid x_\star,\mathcal D)
=
\int
p(y_\star\mid x_\star,\theta)
p(\theta\mid\mathcal D)
\,d\theta.
\]

Visual: several plausible regression functions sampled from the posterior and an uncertainty band.

Message:

> Bayesian prediction averages over plausible models.

**Time:** 1.5 min

---

### Slide 48 — Why deep learning usually uses point estimates

For a modern neural network, \(\theta\) may contain millions or billions of parameters.

Computing:

\[
p(\theta\mid\mathcal D)
\]

and integrating over it is difficult.

Common practical choices:

- MLE;
- MAP;
- ensembles;
- dropout-based approximations;
- variational inference.

Keep this conceptual.

**Time:** 1 min

---

### Slide 49 — The standard supervised DL objective

\[
\min_\theta
\underbrace{
\frac1B
\sum_{i\in\mathcal B}
-\log p_\theta(y_i\mid x_i)
}_{\text{likelihood / data fit}}
+
\underbrace{
\lambda\|\theta\|_2^2
}_{\text{prior / regularization}}.
\]

Below it:

\[
\text{probabilistic model}
+
\text{neural network}
+
\text{optimization}.
\]

Say:

> Deep learning does not discard probabilistic ML. It scales it using expressive functions and gradient-based optimization.

**Time:** 1.5 min

---

### Slide 50 — Retrieval summary [Q]

Ask students to complete:

\[
\text{Gaussian likelihood}\Rightarrow \underline{\hspace{1.5cm}}
\]

\[
\text{Categorical likelihood}\Rightarrow \underline{\hspace{1.5cm}}
\]

\[
\text{Gaussian prior}\Rightarrow \underline{\hspace{1.5cm}}
\]

\[
\text{Laplace prior}\Rightarrow \underline{\hspace{1.5cm}}
\]

\[
\text{MLE}+\text{prior}\Rightarrow \underline{\hspace{1.5cm}}
\]

Answers:

- MSE;
- cross-entropy;
- \(L_2\);
- \(L_1\);
- MAP.

**Time:** 1.5 min

---

### Slide 51 — Final mental model

\[
\boxed{
\text{When you see a loss, ask:
“What likelihood produced it?”}
}
\]

\[
\boxed{
\text{When you see a regularizer, ask:
“What prior produced it?”}
}
\]

\[
\boxed{
\text{When you see a prediction, ask:
“Point estimate or posterior?”}
}
\]

Next lecture:

> Computation graphs, gradients and backpropagation.

**Time:** 1 min

---

# Timing summary

| Section | Slides | Time |
|---|---:|---:|
| Motivation | 1–4 | 5 min |
| Probability revision | 5–11 | 12 min |
| MLE and NLL | 12–15 | 7 min |
| Regression | 16–22 | 13 min |
| Classification | 23–33 | 18 min |
| Bayes classifier | 34–37 | 9 min |
| MAP and regularization | 38–45 | 14 min |
| Full Bayes and DL bridge | 46–51 | 7 min |
| **Total** | **51** | **85 min nominal** |

For an actual 80-minute lecture, mark Slides **20, 33, 43, 44 and 48** as optional/skimmable. The core deck then runs approximately **77–80 minutes**.

---

# Exact interactive set

## 1. `likelihood-to-loss.html`

Linked plots:

\[
p(r),\qquad -\log p(r),\qquad y=f_\theta(x)+r.
\]

Must include Gaussian, Laplace, Uniform and Student-\(t\).

---

## 2. `softmax-cross-entropy.html`

Linked quantities:

\[
z,\qquad p=\operatorname{softmax}(z),\qquad
\mathcal L=-\log p_y,\qquad
\
abla_z\mathcal L=p-y.
\]

---

## 3. `bayes-classifier.html`

Linked quantities:

\[
p(x\mid y),\qquad p(y),\qquad p(y\mid x).
\]

The class-prior slider is the most important element.

---

## 4. `map-regularization.html`

Linked views:

- parameter-space contours;
- fitted model;
- MLE versus MAP;
- Gaussian versus Laplace prior;
- prior scale.

---

## Figures Claude should create

Ask Claude to create these as native Typst/SVG where possible:

1. assumption \(\rightarrow\) likelihood \(\rightarrow\) loss pipeline;
2. continuous density with shaded probability area;
3. Gaussian mean/variance comparison;
4. multivariate Gaussian contour triptych;
5. regression curve with vertical conditional Gaussians;
6. residual density above corresponding loss curve;
7. logits \(\rightarrow\) softmax \(\rightarrow\) CE pipeline;
8. \(-\log p_y\) curve;
9. class-conditional Bayes classifier diagram;
10. prior × likelihood \(\rightarrow\) posterior;
11. MLE/MAP contour diagram;
12. posterior samples of regression functions.

For the implementation, tell Claude to use **Typst + Touying**, generate a PDF-first deck, and place all four JS interactives under a GitHub Pages `demos/` directory.
