# Lecture 1 · Where Deep Learning Losses Come From · Teaching Guide
*First-time-instructor companion. Audience: undergraduates beginning the public deep-learning sequence after one introductory ML course.*

## The spine (say this in two sentences)

A supervised loss is the negative log-likelihood of an explicit observation model: Gaussian outputs give squared error, Bernoulli outputs give binary cross-entropy, and categorical outputs give multiclass cross-entropy. A regularizer is the negative log of a parameter prior: Gaussian priors give $L_2$, Laplace priors give $L_1$, and minimizing data NLL plus negative log-prior is MAP estimation.

The line to repeat is:

> **Choose what outcomes can happen; that choice determines the loss. Choose which parameters are plausible; that choice determines the regularizer.**

## Where it sits

This is public **Lecture 1**, the probabilistic foundations lecture. Students should already recognize linear regression, logistic regression, MSE, and cross-entropy as algorithms or formulas. The purpose here is to give those formulas one probabilistic origin and a vocabulary that remains stable throughout the course.

Public **Lecture 2, From Linear Models to Neural Networks**, keeps the loss contract fixed and expands the prediction rule from a weighted sum to neurons and MLPs. Close Lecture 1 with the exact handoff:

> **Today we chose and justified the objective; next time we expand the function that supplies its parameters.**

## Evidence and reproducibility contract

Lecture 1 contains analytical derivations and small constructed computations. It does not contain benchmark claims.

| Label | Meaning in this lecture |
|---|---|
| **ANALYTICAL** | follows symbolically from a named distribution, factorization, or prior |
| **SYNTHETIC · COMPUTED** | deterministic or seeded values calculated from a displayed teaching case |
| **MEASURED PERFORMANCE** | none |
| **REAL DATA / MODEL OUTPUT** | none |

Keep three numerical lanes distinct:

1. The main notebook uses seeded synthetic samples to connect `torch.distributions`, `log_prob`, sampling, MLE, classification, and MAP.
2. The robust notebook’s main experiment uses 25 seeded observations around $y=1+2x$ plus two inserted errors.
3. The deck carries a separate exact 12-point robust-regression case. Its coordinates and every displayed coefficient are reproduced in **Appendix A** of the executed robust notebook.

Do not describe the 12-point coefficients as “results on a dataset.” Say:

> **SYNTHETIC · COMPUTED: these are deterministic fits to the exact points shown, included so every coefficient can be audited.**

Current execution checkpoint:

- `notebooks/L01/00_likelihood_iid_worked_examples.ipynb`: 69 code cells, execution counts 1–69, zero error outputs;
- `notebooks/L01/03_robust_linear_regression.ipynb`: 20 code cells, execution counts 1–20, zero error outputs, including the exact deck audit;
- focused companions `01_gaussian_nll_is_mse.ipynb` and `02_softmax_cross_entropy.ipynb`: already executed, with zero error outputs.

## Learning contract

By the end, students should be able to:

1. distinguish probability mass from probability density and obtain continuous probability from an integral;
2. distinguish a predictive distribution $p_\theta(y\mid x)$ from a likelihood $L(\theta)$ by naming what is fixed and what varies;
3. separate independence from identical distribution and state what each contributes to an i.i.d. likelihood;
4. turn a product likelihood into a sum log-likelihood without changing the maximizing parameter;
5. explain why training with average NLL is maximum likelihood under conditional independence;
6. derive MSE from a Gaussian observation model;
7. explain how Gaussian, Laplace, and Student-$t$ residual models give different loss and gradient influence for an outlier;
8. derive BCE from Bernoulli NLL and multiclass cross-entropy from categorical NLL;
9. name prior, likelihood, evidence, and posterior in Bayes’ rule for parameters;
10. derive the MAP objective as data NLL plus negative log-prior;
11. map a Gaussian prior to $L_2$ and a Laplace prior to $L_1$;
12. explain why ridge shrinks continuously while $L_1$ can threshold a coefficient exactly to zero.

## Three persistent board cases

Use these cases as a ledger. Do not introduce new numbers when an existing case already supports the next concept.

### Case A · ten coin flips for MLE

Observed sequence:

\[
H,H,T,T,T,H,H,T,T,T,
\qquad n_H=4,\quad n_T=6.
\]

For a Bernoulli coin with $P(H\mid\theta)=\theta$:

\[
L(\theta)=\theta^4(1-\theta)^6,
\qquad
\log L(\theta)=4\log\theta+6\log(1-\theta).
\]

The derivative gives

\[
\frac{d}{d\theta}\log L(\theta)
=\frac{4}{\theta}-\frac{6}{1-\theta}=0
\quad\Rightarrow\quad
\hat\theta_{\text{MLE}}=\frac{4}{10}=0.4.
\]

The second derivative is negative on $(0,1)$:

\[
-\frac{4}{\theta^2}-\frac{6}{(1-\theta)^2}<0.
\]

Use this same sequence to connect product, log, derivative, curvature, NLL, and empirical risk. The main notebook independently verifies a grid MLE of $0.4008$, a closed-form MLE of $0.4000$, and an optimizer estimate of $0.4000$.

### Case B · exact 12-point robust-regression audit

The deck’s typical observations are

\[
\begin{aligned}
&(-3.0,-2.8),(-2.4,-2.2),(-1.8,-1.9),(-1.2,-0.9),(-0.6,-0.5),\\
&(0.0,0.2),(0.7,0.8),(1.4,1.2),(2.0,2.2),(2.6,2.4),
\end{aligned}
\]

and the two declared outliers are

\[
(-2.2,3.3),\qquad(2.2,-3.2).
\]

Every value below is **SYNTHETIC · COMPUTED**:

| Fit | Scale / degrees of freedom | $(\theta_0,\theta_1)$ in $\hat y=\theta_0+\theta_1x$ |
|---|---|---:|
| OLS on ten typical points | Gaussian | $(0.070,0.955)$ |
| Gaussian on all twelve | $\sigma=1$ | $(-0.040,0.397)$ |
| Laplace on all twelve | $s_L=1$ | $(0.044,0.906)$ |
| Student-$t$ | $(\nu,s)=(1,1)$ | $(0.064,0.928)$ |
| Student-$t$ | $(\nu,s)=(3,0.5)$ | $(0.065,0.934)$ |
| Student-$t$ | $(\nu,s)=(3,1)$ | $(0.053,0.882)$ |
| Student-$t$ | $(\nu,s)=(3,1.5)$ | $(0.036,0.807)$ |
| Student-$t$ | $(\nu,s)=(30,1)$ | $(-0.014,0.581)$ |

The notebook trace uses closed-form least squares for Gaussian, the exact LAD vertex for Laplace, and deterministic full-batch L-BFGS for Student-$t$. It asserts every three-decimal pair shown above. The main robust-notebook experiment remains separate; do not mix its 25-point coefficients into this table.

### Case C · HHH for Bayes and MAP

Compare two discrete hypotheses:

- fair coin $F$: $P(H\mid F)=0.5$;
- biased coin $B$: $P(H\mid B)=0.9$.

For $\mathcal D=(H,H,H)$:

\[
P(\mathcal D\mid F)=0.5^3=0.125,
\qquad
P(\mathcal D\mid B)=0.9^3=0.729.
\]

The likelihood ratio is $0.729/0.125=5.832$. With $P(F)=0.90$ and $P(B)=0.10$:

\[
w_F=0.1125,\quad w_B=0.0729,\quad
P(\mathcal D)=0.1854,
\]

so

\[
P(F\mid\mathcal D)\approx0.607,
\qquad
P(B\mid\mathcal D)\approx0.393.
\]

The data favours $B$, but the strong fair-coin prior still wins. This is the cleanest correction to “three heads prove the coin is biased.”

## Exact 80-minute route

### ⭐ Core · exactly 55 minutes

- Opening losses, probabilistic map, and density-versus-probability contract: 8 min.
- Prediction versus likelihood, independence versus identical distribution, and i.i.d. factorization: 10 min.
- Ten-flip likelihood, logs, MLE, NLL, and empirical risk: 12 min.
- Gaussian regression to MSE, then the exact robust case and gradient influence: 10 min.
- Bernoulli/BCE and categorical/cross-entropy: 7 min.
- HHH Bayes update, MAP, and Gaussian/Laplace prior map: 8 min.

### ⭐⭐ Should cover · exactly 15 minutes

- Uniform support as an infinite NLL barrier and heteroscedastic uncertainty: 4 min.
- Student-$t$ influence and the $\nu,s$ audit: 4 min.
- Ridge shrinkage and the three-case soft-threshold derivation: 5 min.
- Full-Bayes posterior predictive versus MLE/MAP point estimates: 2 min.

### ⭐⭐⭐ Optional · exactly 10 minutes

- Poisson likelihood/log-likelihood comparison and floating-point underflow numeric: 3 min.
- Multivariate Gaussian prior geometry and correlated directions: 3 min.
- Interactive likelihood-to-loss or likelihood-prior-MAP exploration: 2 min.
- Focused companion-notebook extension: 2 min.

The labels sum to **55 core + 15 should-cover + 10 optional = 80 minutes**.

### Suggested clock

| Time | Deck route | Teaching move |
|---:|---|---|
| 0–4 | “Three familiar losses” → “map for today” | Ask what assumption could produce each displayed term. Do not answer all three yet. |
| 4–8 | probability primer | Contrast Bernoulli mass with Gaussian density; make students say “area.” |
| 8–14 | model notation → prediction versus likelihood | Hold $\theta$ fixed and vary $y$; then hold observed $y$ fixed and vary $\theta$. |
| 14–18 | independence / identical distribution | Use the two failure examples; write the two “I”s separately. |
| 18–30 | i.i.d. factorization → coin MLE → NLL | Carry Case A from product to stationary point and curvature. |
| 30–35 | empirical risk / support | Explain why $1/n$ rescales but does not move the optimum; use Uniform as a hard support check. |
| 35–43 | Gaussian regression → MSE | Derive only the parameter-dependent terms; state what happens to $\sigma^2$. |
| 43–48 | robust residual models | Reveal Case B’s three main fits and compare gradient influence at $r=4$. |
| 48–55 | Bernoulli/BCE → categorical/CE | Match outcome support first, then take NLL. |
| 55–64 | Bayes HHH update | Keep likelihood fixed, change only the prior, normalize explicitly. |
| 64–72 | MAP → Gaussian/Laplace priors | Translate negative log-prior into $L_2/L_1$ and derive shrink versus threshold. |
| 72–76 | point estimates versus full Bayes | One parameter vector versus averaging predictions over a posterior. |
| 76–80 | standard DL objective and retrieval | Reconstruct the full objective from observation model plus prior; hand off to L2. |

If arithmetic runs long, omit the optional Poisson and multivariate-prior slides. Never save time by dropping the prediction/likelihood distinction, the two “I”s, the support of the outcome model, or the evidence category on the robust case.

## Teach it like this

The lecture is one repeated transformation:

1. **Name the random outcome.** What values can $Y$ take?
2. **Choose a normalized distribution.** Bernoulli, categorical, Gaussian, Laplace, Student-$t$, or Uniform.
3. **Fix the observation and vary parameters.** The predictive distribution becomes a likelihood.
4. **Factor the dataset carefully.** Independence justifies the product; a shared conditional model gives repeated factor form.
5. **Take logs.** Ordering is preserved and products become stable sums.
6. **Negate.** Maximum likelihood becomes minimization of NLL.
7. **Add a prior if desired.** Its negative log is the regularization term; the point estimate is MAP.
8. **Return to the standard DL objective.** Nothing new was added at the end; its two terms came from two declared assumptions.

Use **commit → calculate → reveal → interpret** throughout:

- “Can a continuous density be larger than one?” before the area explanation.
- “What varies in $p_\theta(y\mid x)$? What varies in $L(\theta)$?” before the axis swap.
- “Which ‘I’ allows multiplication? Which allows the same factor form?” before the i.i.d. equation.
- “Should $0.4$ or $0.7$ better explain four heads in ten flips?” before the likelihood values.
- “Why does logging preserve the best parameter?” before invoking monotonicity.
- “Which residual model gives the largest gradient factor at $r=4$?” before revealing $4$, $1$, and $16/19$.
- “What outcomes does a Gaussian allow that a Bernoulli does not?” before BCE versus MSE.
- “Do three heads make $P(B\mid HHH)=1$?” before the prior-weight-normalize ledger.
- “Which prior has a corner at zero?” before the soft-threshold cases.

## Exact mathematics and board work

### 1 · Density is not point probability

For continuous $Y$,

\[
P(a\le Y\le b)=\int_a^b p_Y(y)\,dy,
\qquad
P(Y=y)=0.
\]

A density can exceed $1$ when concentrated over a narrow support; only integrated area must lie in $[0,1]$. Do not call the vertical height of a Gaussian curve “the probability of $y$.”

### 2 · Prediction and likelihood swap the varying axis

Predictive view:

\[
p_\theta(y\mid x),
\qquad \theta,x\text{ fixed; }y\text{ varies}.
\]

Likelihood view after observing $(x_i,y_i)$:

\[
L_i(\theta)=p_\theta(y_i\mid x_i),
\qquad x_i,y_i\text{ fixed; }\theta\text{ varies}.
\]

Likelihood is a score over parameter values. It is not generally a probability distribution over $\theta$ and need not integrate to one over parameter space.

### 3 · Independence and identical distribution do different work

Conditional independence yields

\[
p(\mathcal D\mid\theta)=\prod_{i=1}^n p_\theta(y_i\mid x_i).
\]

Independence justifies multiplication. The shared conditional model and parameters justify writing each factor with the same functional form. An independent but non-identically distributed sequence still factors, but its factors differ. An identically distributed but dependent sequence reuses marginal rules but does not factor into their product.

### 4 · Logs preserve the optimum and stabilize products

Because $\log$ is strictly increasing,

\[
\arg\max_\theta L(\theta)=\arg\max_\theta\log L(\theta).
\]

Under the factorization,

\[
\log L(\theta)=\sum_i\log p_\theta(y_i\mid x_i),
\]

and

\[
\hat\theta_{\text{MLE}}
=\arg\min_\theta\left[-\sum_i\log p_\theta(y_i\mid x_i)\right].
\]

The mean rather than sum changes scale, not the minimizer, when there is no additional term. Once a regularizer is added, the convention affects the numerical value of its coefficient $\lambda$.

### 5 · Gaussian observation noise gives MSE

Assume

\[
Y_i\mid X_i=x_i\sim\mathcal N(f_\theta(x_i),\sigma^2).
\]

Then

\[
-\log p_\theta(y_i\mid x_i)
=\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}
+\log\sigma+\frac12\log(2\pi).
\]

For fixed $\sigma$, the last two terms are constant in $\theta$ and the positive factor $1/(2\sigma^2)$ does not move the minimizer. Therefore Gaussian MLE for the mean parameters is least squares.

Do not say “MSE assumes the targets are Gaussian.” The precise statement is that **conditional residuals around the predicted mean are Gaussian with the declared variance model**.

### 6 · The residual model controls gradient influence

Let $r_i=y_i-f_\theta(x_i)$ and $\ell(r)=-\log p_R(r)$. Then

\[
\nabla_\theta\ell_i
=-\ell'(r_i)\nabla_\theta f_\theta(x_i).
\]

At unit scale:

| Residual model | NLL shape up to constants | $|\ell'(r)|$ at $r=4$ |
|---|---|---:|
| Gaussian | $r^2/2$ | $4$ |
| Laplace | $|r|$ | $1$ |
| Student-$t$, $\nu=3$ | $2\log(1+r^2/3)$ | $16/19\approx0.84$ |

Robust fitting does not automatically identify or delete a bad point. It changes how quickly that point’s loss and gradient influence grow under a declared noise model.

### 7 · Match classification loss to outcome support

For binary $Y\in\{0,1\}$:

\[
p_\theta(y\mid x)=p^y(1-p)^{1-y}
\]

gives

\[
-\log p_\theta(y\mid x)
=-y\log p-(1-y)\log(1-p),
\]

the binary cross-entropy. The logit $z\in\mathbb R$ maps to $p=\sigma(z)\in(0,1)$; in software, use a fused logits-plus-loss implementation for numerical stability.

For one-hot categorical $\mathbf y$:

\[
p_\theta(\mathbf y\mid x)=\prod_k p_k^{y_k}
\quad\Rightarrow\quad
-\log p_\theta(\mathbf y\mid x)=-\sum_k y_k\log p_k=-\log p_{\text{true}}.
\]

### 8 · Bayes and MAP

\[
p(\theta\mid\mathcal D)
=\frac{p(\mathcal D\mid\theta)p(\theta)}{p(\mathcal D)}.
\]

The evidence $p(\mathcal D)$ normalizes the posterior. It is constant with respect to $\theta$, so it drops from a MAP `argmax`, not from the definition of the posterior:

\[
\hat\theta_{\text{MAP}}
=\arg\min_\theta
\left[-\log p(\mathcal D\mid\theta)-\log p(\theta)\right].
\]

### 9 · Gaussian prior gives $L_2$; Laplace prior gives $L_1$

For $\theta\sim\mathcal N(0,\tau^2I)$:

\[
-\log p(\theta)=\frac{1}{2\tau^2}\|\theta\|_2^2+C.
\]

For independent Laplace coordinates with scale $a$:

\[
-\log p(\theta)=\frac1a\|\theta\|_1+C.
\]

The scalar $L_2$ quadratic produces continuous shrinkage. The $L_1$ corner yields the soft-threshold rule

\[
\hat\theta_{\text{MAP}}
=\operatorname{sign}(z)\max(|z|-\lambda,0),
\]

so an interval of weak likelihood centres maps exactly to zero.

## Notebook choreography

### Main notebook · `00_likelihood_iid_worked_examples.ipynb`

Do not run all 69 cells live without a plan. Use these six prediction checkpoints:

| Executed checkpoint | Ask before reveal | Retained output to inspect |
|---:|---|---|
| 10 | At $y=6$, which model has more density: Normal or Student-$t$? | Student-$t$/Normal density ratio $3.58\times10^5$. |
| 19 | What is the joint log-probability of twelve independent flips? | explicit flip ledger and sum $-7.64555$. |
| 25–28 | Where should the coin MLE land? | grid $0.4008$, closed form $0.4000$, optimizer $0.4000$. |
| 39–43 | Do Gaussian NLL and MSE really agree numerically? | identity error $5.97\times10^{-15}$ and identical fit $(1.9404,3.0356)$. |
| 50–54 | Does Bernoulli `log_prob` equal fused BCE? | both $0.319280198231$; exact equality check passes. |
| 63–69 | How does a prior move a weak estimate? | MLE/MAP line comparison and Laplace MAP printed as $-0.000$, numerically zero. |

The notebook is a synthetic mechanism audit. Seeds make the sampled values reproducible; they do not convert the experiment into real-world evidence.

### Robust notebook · `03_robust_linear_regression.ipynb`

Use the main body to compare Gaussian, Laplace, and Student-$t$ on the seeded $y=1+2x$ experiment. Its retained main-case estimates are:

| Model | Main-case $(\theta_0,\theta_1)$ |
|---|---:|
| Gaussian | $(1.143,1.568)$ |
| Laplace | $(1.181,1.965)$ |
| Student-$t$ | $(1.130,1.946)$ |

Then jump to **Appendix A** only when auditing the deck. It prints all eight expected coefficient pairs for the exact 12-point case, asserts equality at three decimals, and renders the same three fitted lines. Say explicitly that the main case and deck audit are different synthetic constructions serving different jobs.

### Focused companions

- `01_gaussian_nll_is_mse.ipynb`: use when students need a shorter regression-only proof.
- `02_softmax_cross_entropy.ipynb`: use when students need the softmax, $-\log p_{\text{true}}$, and $p-y$ gradient in isolation.

## Heads-up for the instructor

- **A likelihood is not a posterior.** $L(\theta)=p(\mathcal D\mid\theta)$ scores parameter values after data are fixed. It is not normalized over $\theta$.
- **Density is not probability.** A continuous density height may exceed one; probability is integrated area.
- **Conditional independence is the training factorization.** Inputs are treated as observed/fixed when writing $\prod_i p_\theta(y_i\mid x_i)$.
- **“Identically distributed” does not mean identical labels.** It means the same conditional sampling rule and shared parameters apply to each example.
- **Logs preserve order only for positive values.** Likelihoods from valid observed events/densities are nonnegative; zero likelihood maps to $-\infty$ log-likelihood and an infinite NLL barrier.
- **Average versus sum matters once regularization is present.** Dividing only the data term by $n$ changes its balance with a fixed regularizer unless $\lambda$ is rescaled.
- **Gaussian NLL is MSE only under the stated variance treatment.** A learned input-dependent variance adds a log-variance term and can no longer be reduced to plain unweighted MSE.
- **Laplace optimization uses subgradients.** At $r=0$, $|r|$ has no unique derivative; its subgradient set is $[-1,1]$ at unit scale.
- **Robust does not mean immune.** Student-$t$ and Laplace reduce extreme-point influence under their assumptions; validation and residual checks still determine whether the model is appropriate.
- **Cross-entropy should consume logits in common frameworks.** Fused losses perform log-softmax or sigmoid-plus-log stably.
- **The evidence term is not “unimportant.”** It is irrelevant to a point-estimate `argmax` over $\theta$, but essential for posterior normalization and model evidence.
- **A prior is a modelling choice, not a moral preference.** Zero-centred isotropic Gaussian priors encode shrinkage toward zero equally in all directions; other covariance structures encode other geometry.

## Where students stumble

- **“The network predicts a likelihood.”** It parameterizes a predictive distribution. Only after an observation is fixed do we view that density or mass as a function of parameters and call it likelihood.
- **“Independence and identical distribution are one assumption.”** Ask which word permits multiplication and which permits reusing the same factor form.
- **“The log changes the best parameter.”** Draw two positive likelihood values, apply a monotone log, and preserve their ordering.
- **“A low likelihood is a probability that the parameter is wrong.”** Likelihood does not assign posterior probability to a parameter without a prior and normalization.
- **“MSE is always wrong for classification.”** It can be optimized, but its usual Gaussian observation model mismatches binary/categorical support. BCE/CE directly matches the label outcome space.
- **“The Gaussian fit found the outliers.”** It did not. In Case B it bent toward them because squared residual influence grows with magnitude.
- **“The Student-$t$ coefficient is a measured improvement.”** No. It is a deterministic fit on a constructed case, useful for mechanism inspection only.
- **“MAP is full Bayes.”** MAP is still one parameter vector. Full Bayesian prediction integrates or approximates an average over the posterior.
- **“$L_1$ is sparse because its density is narrow.”** The exact-zero mechanism is the corner in the negative log-prior, made explicit by the subgradient interval at zero.

## If a student asks…

- **“Can a density exceed one?”** Yes. A Uniform distribution on $[0,0.1]$ has density $10$; its total area is still $10\times0.1=1$.
- **“Why not maximize the raw product?”** Logs preserve the maximizer, turn products into sums, and prevent underflow.
- **“What if examples are dependent?”** The simple product is no longer justified. Model the joint structure, condition on the relevant history, or use a likelihood appropriate to the dependence.
- **“Why is $\sigma$ absent from ordinary least squares?”** With fixed shared $\sigma$, it only rescales the squared-error term and adds a constant. If $\sigma$ is learned or depends on $x$, its terms must remain.
- **“Is MAE always more robust than MSE?”** It caps residual-gradient magnitude at fixed scale, but whether that assumption improves prediction is an empirical validation question.
- **“Why use Student-$t$ instead of deleting outliers?”** It retains all observations while smoothly discounting extreme residuals according to $\nu$ and scale. Deletion needs an independent data-quality justification.
- **“Why does a confident error receive a huge CE loss?”** The model assigned the observed true class probability near zero, so $-\log p_{\text{true}}$ becomes large.
- **“Do priors disappear with lots of data?”** Their relative effect often decreases because log-likelihood contributions accumulate with data, but the conclusion depends on identifiability, prior scale, and model structure.
- **“Does weight decay always equal Bayesian MAP?”** Exact equivalence depends on the optimizer, scaling, parameter groups, and implementation convention. The objective-level Gaussian-prior-to-$L_2$ mapping is the statement established here.
- **“Why exclude bias from regularization sometimes?”** That is an additional modelling/optimization convention. Lecture 1’s isotropic derivation penalizes every included coordinate; later implementations may choose parameter groups deliberately.

## If you are short on time

For a 55-minute version:

- keep density-versus-probability and prediction-versus-likelihood;
- keep the independence/identical distinction and Case A from product through NLL;
- derive Gaussian-to-MSE and show only the three main Case B fits plus the $r=4$ influence table;
- derive Bernoulli-to-BCE and categorical-to-CE;
- run the HHH prior-weight-normalize calculation;
- show MAP as NLL plus negative log-prior and state Gaussian $\to L_2$, Laplace $\to L_1$;
- close on the standard supervised DL objective and L2 handoff.

Cut Poisson, the long underflow numeric, heteroscedastic detail, the Student-$t$ hyperparameter table, correlated Gaussian-prior geometry, full posterior-predictive detail, and both interactives. Do not cut the exact evidence label on the robust case or blur its values with the main robust-notebook experiment.

## Closing retrieval

Ask students to complete all five mappings without looking:

| Assumption | Objective consequence |
|---|---|
| Gaussian observation model | MSE / Gaussian NLL |
| Bernoulli or categorical observation model | BCE or multiclass CE |
| conditional independence | sum of per-example log-probabilities |
| Gaussian parameter prior | $L_2$ penalty |
| Laplace parameter prior | $L_1$ penalty |

Then end with:

> **A deep network changes the function inside the probability model. The logic of likelihood, loss, prior, and regularization stays the same.**
