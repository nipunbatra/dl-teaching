// Why These Losses? — Lecture 1 · ES 667 Deep Learning
// Typst + shared metropolis theme. Compile from the repo root:
//   typst compile --root . lecture1/L1-probabilistic-view.typ
//   typst compile --root . --input handout=true lecture1/L1-probabilistic-view.typ
// All theme/palette/helpers/diagrams live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Why These Losses?],
  subtitle: [A probabilistic refresher for Deep Learning],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"

#title-slide()

// ══════════════════════════════ PART I ══════════════════════════════
== Three familiar equations

You have already met these losses:
#pause
$ cal(L)_"reg" = 1/n sum_i (y_i - hat(y)_i)^2 quad #text(fill: MUTED, size: 15pt)[(regression)] $
#pause
$ cal(L)_"class" = -1/n sum_i log p_theta (y_i | x_i) quad #text(fill: MUTED, size: 15pt)[(classification)] $
#pause
$ cal(L)_"total" = cal(L)_"data" + lambda norm(theta)_2^2 quad #text(fill: MUTED, size: 15pt)[(regularized)] $
#pause
#notebox[*Why exactly these three expressions?* Today: they are not arbitrary.]

== The map for today
#stag(V)

Two probabilistic assumptions, two halves of the objective:
#pause
#v(10pt)
#align(center, diagram(
  spacing: (17mm, 9mm), node-stroke: 0.9pt,
  {
    node((0,0), [outputs\ $p_theta (y|x)$], shape: fletcher.shapes.rect, fill: rgb("#EFEEEB"), stroke: INK, inset: 7pt)
    node((2,0), [parameters\ $p(theta)$], shape: fletcher.shapes.rect, fill: rgb("#EFEEEB"), stroke: INK, inset: 7pt)
    node((0,1), [training loss], shape: fletcher.shapes.rect, fill: rgb("#FDECD6"), stroke: ACC, inset: 7pt)
    node((2,1), [regularizer], shape: fletcher.shapes.rect, fill: rgb("#FDECD6"), stroke: ACC, inset: 7pt)
    edge((0,0),(0,1), text(size: 15pt)[negative log-likelihood], "-|>", stroke: 0.8pt + MUTED, label-side: left)
    edge((2,0),(2,1), text(size: 15pt)[Bayes' rule $arrow.r$ MAP], "-|>", stroke: 0.8pt + MUTED, label-side: right)
    node((1,2), text(fill: white, weight: 600)[DL objective], shape: fletcher.shapes.rect, fill: TEAL, stroke: none, inset: 8pt)
    edge((0,1),(1,2), "-|>", stroke: 0.8pt + MUTED)
    edge((2,1),(1,2), "-|>", stroke: 0.8pt + MUTED)
  }
))

// ══════════════════════════════ PART II ══════════════════════════════
= Minimal probability revision

== A model should describe uncertainty

#two(
  [$ hat(y) = f_theta (x) $ #align(center, text(fill: MUTED)[a single number])],
  [$ Y | X=x tilde p_theta (dot | x) $ #align(center, text(fill: MUTED)[a full distribution])],
)
#pause
- regression network → a *Gaussian mean*
#pause
- classifier → *categorical* probabilities
#pause
- language model → categorical distribution over *tokens*
#pause
#result[A neural network parameterizes a probability distribution.]

== Density is not probability
#stag(V)

For a continuous $Y$, probability comes from *area*:
$ P(a <= Y <= b) = integral_a^b p(y) dif y $
#pause
#alertbox[$p(y) != P(Y=y)$ — a density can *exceed 1*; only _integrals_ are probabilities.]
#fig("/lecture1/figures/density_area.svg", w: 42%)

== Uniform distribution: support matters
#stag(V)

#two(r: (1.15fr, 1fr),
  [$ Y tilde cal(U)(a, b) quad p(y) = cases(1\/(b-a) & quad a <= y <= b, 0 & quad "otherwise") $
   #v(4pt)
   #notebox[A likelihood can impose a *hard constraint*.]],
  fig("/lecture1/figures/uniform.svg", w: 96%),
)

== Checkpoint: uniform support #Q

#mcq(
  [For a Uniform$(a,b)$ observation model, what is the NLL if the target lies outside $[a,b]$?],
  [$0$],
  [A fixed finite penalty],
  [$-log 0 = +infinity$],
  [It depends only on $sigma$],
)

== Answer: uniform support #A

#mcq-answer([C], [$-log 0 = +infinity$], [The model assigns zero density outside its support, so such a target is impossible under the assumption.])

== Gaussian distribution
#stag(V)

$ Y tilde cal(N)(mu, sigma^2) quad quad p(y) = 1/sqrt(2 pi sigma^2) exp(-(y-mu)^2/(2 sigma^2)) $
#pause
#fig("/lecture1/figures/gaussian_params.svg", w: 56%)
#align(center)[Only two knobs: *location* $mu$ and *scale* $sigma$.]

== Multivariate Gaussian geometry
#stag(V)

$ theta tilde cal(N)(mu, Sigma) quad -log p(theta) = 1/2 (theta - mu)^top Sigma^(-1) (theta - mu) + C $
#pause
#fig("/lecture1/figures/mvn_triptych.svg", w: 74%)
#align(center, text(size: 17pt)[Covariance says *which directions in parameter space are plausible.*])

== Independence turns products into sums

Conditionally independent observations:
$ p(cal(D) | theta) = product_(i=1)^n p_theta (y_i | x_i) $
#pause
Take logs — the product becomes a *sum*:
$ log p(cal(D) | theta) = sum_(i=1)^n log p_theta (y_i | x_i) $
#pause
#notebox[Why a loss is an *average over examples*, and why minibatches work: $1/(|B|) sum_(i in B) ell_i$.]

// ══════════════════════════════ PART III ══════════════════════════════
= Bayes' rule → likelihood → loss

== Bayes' rule and its four pieces
#stag(D)

Turn the product rule $p(cal(D), theta) = p(cal(D) | theta) thin p(theta) = p(theta | cal(D)) thin p(cal(D))$ around:
#pause
$ p(theta | cal(D)) = (p(cal(D) | theta) thin p(theta)) / p(cal(D)) $
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 5pt), align: (center, left),
  [$p(theta | cal(D))$], [*posterior* — belief about $theta$ _after_ seeing data],
  [$p(cal(D) | theta)$], [*likelihood* — how well $theta$ explains the data],
  [$p(theta)$], [*prior* — belief about $theta$ _before_ data],
  [$p(cal(D))$], [*evidence* — the normaliser $integral p(cal(D) | theta) p(theta) dif theta$],
))
#pause
#result[*MLE* (now): keep only the *likelihood*. *MAP* (later): likelihood $times$ *prior* → regularisation.]

// ══════════════════════════════ PART III·B ══════════════════════════════
= Maximum likelihood → training losses

== Likelihood: one expression, two viewpoints
#stag(Q)

The one model $p_theta (y | x)$, read two ways:
#pause
#two(
  notebox[*Fix $theta$, vary $y$* \ → a *distribution* over outcomes.],
  notebox[*Fix observed $y$, vary $theta$* \ → a *likelihood* over parameters.],
)
#pause
#alertbox[The likelihood is _not_ "the probability that $theta$ is correct."]

== Coin flips: a likelihood you can compute
#stag(D)

Ten flips come up $H, H, T, T, T, H, H, T, T, T$ — so $n_H = 4$, $n_T = 6$.
#pause
Model each flip as Bernoulli$(theta)$: $p(H) = theta$, $p(T) = 1 - theta$.
#pause
The flips are i.i.d., so the *likelihood* of this exact sequence is
$ L(theta) = product_i p(x_i | theta) = theta^(n_H) (1 - theta)^(n_T) = theta^4 (1 - theta)^6 $
#pause
#notebox[Different $theta$ make the *observed* sequence more or less probable. Which $theta$ makes it most probable?]

== Coin flips: maximise the (log-)likelihood
#stag(D)

Logs turn the product into a sum:
$ cal(L)(theta) = log L(theta) = n_H log theta + n_T log(1 - theta) $
#pause
Set the derivative to zero:
$ (dif cal(L))/(dif theta) = n_H / theta - n_T / (1 - theta) = 0 $
#pause
$ ==> theta_"MLE" = n_H / (n_H + n_T) = 4/10 = 0.4 $
#pause
#result[The intuitive "fraction of heads" *is* the maximum-likelihood estimate.]

== Maximum likelihood estimation

$ hat(theta)_"MLE" = arg max_theta p(cal(D) | theta) $
#pause
#notebox[*Interpretation.* Choose the parameters under which the observed dataset would be *least surprising*.]

== From MLE to negative log-likelihood
#stag(D)

$ arg max_theta product_i p_theta (y_i | x_i) = arg max_theta sum_i log p_theta (y_i | x_i) $
#pause
$ = arg min_theta (-sum_i log p_theta (y_i | x_i)) $
#pause
#result[$ell_i (theta) = -log p_theta (y_i | x_i)$]

== NLL is empirical risk

*ML notation* — empirical risk:
$ hat(R)(theta) = 1/n sum_i ell(y_i, f_theta (x_i)) $
#pause
*Probabilistic reading* of the per-example loss:
$ ell(y_i, f_theta (x_i)) = -log p_theta (y_i | x_i) $
#pause
#result[Choosing a loss = choosing a probabilistic observation model.]

// ══════════════════════════════ PART IV ══════════════════════════════
= Regression — where does MSE come from?

== Regression as signal plus noise

$ y_i = f_theta (x_i) + epsilon_i, quad quad epsilon_i tilde cal(N)(0, sigma^2) $
#pause
Therefore the conditional output distribution is
$ Y_i | x_i, theta tilde cal(N)(f_theta (x_i), sigma^2) $

== What the assumption means visually
#stag(V)

#fig("/lecture1/figures/regression_conditionals.svg", w: 52%)
#pause
#notebox[Gaussian noise lives in the *output direction*, conditional on $x$ — a vertical bell at every input.]

== Gaussian likelihood
#stag(D)

$ p_theta (y_i | x_i) = 1/sqrt(2 pi sigma^2) exp(-(y_i - f_theta (x_i))^2/(2 sigma^2)) $
#pause
Negative log of a single example:
$ -log p_theta (y_i | x_i) = (y_i - f_theta (x_i))^2/(2 sigma^2) + C $

== MSE is Gaussian MLE

Sum over the dataset:
$ -log p(cal(D) | theta) = 1/(2 sigma^2) sum_i (y_i - f_theta (x_i))^2 + C $
#pause
With $sigma$ fixed, the minimizer is
$ hat(theta)_"MLE" = arg min_theta sum_i (y_i - f_theta (x_i))^2 $
#pause
#result[MSE = Gaussian NLL, up to constants.]

== What role does $sigma^2$ play?
#stag(OPT)

$ ell_i = (y_i - mu_i)^2/(2 sigma^2) + 1/2 log sigma^2 + C $
#pause
- small $sigma$: errors penalized *strongly*
#pause
- large $sigma$: errors deemed *less surprising*
#pause
- a predicted $sigma_i$ needs the $1/2 log sigma_i^2$ term — it stops the model claiming infinite uncertainty
#pause
#notebox[*Heteroscedastic regression:* let the network output both $(mu_i, log sigma_i^2) = f_theta (x_i)$.]

== Different noise, different loss
#stag(V)

#two(r: (1fr, 1.25fr),
  [#table(columns: 2, stroke: 0.5pt + MUTED, inset: 7pt, align: left,
    table.header([*Noise model*], [*NLL*]),
    [#text(fill: ACC, weight: 600)[Gaussian]], [$r^2$],
    [#text(fill: TEAL, weight: 600)[Laplace]], [$|r|$],
    [#text(fill: MUTED, weight: 600)[Uniform]], [hard interval],
    [#text(fill: GREEN, weight: 600)[Student-$t$]], [robust])
   #v(3pt) #text(fill: MUTED)[$r = y - hat(y)$ · colours match the curves]],
  align(center, lines(
    // Gaussian r², Laplace |r|, Student-t ln(1+r²) — each the closed-form NLL, sampled
    fn: (r => r * r, r => 1.6 * calc.abs(r), r => 2.2 * calc.ln(1 + r * r)),
    domain: (-3, 3), samples: 60,
    colors: (ACC, TEAL, GREEN),
    markers: false,
    // Uniform "hard interval" walls at r = ±2.2 (the fn/series data channels are
    // mutually exclusive, so the walls ride along as vertical reference lines)
    vlines: ((-2.2, none, MUTED), (2.2, none, MUTED)),
    x-label: [residual $r$], y-label: [$-log p(r)$],
    size: (64mm, 44mm),
  )),
)

== Interactive: likelihood → loss
#stag(I)

#interbox(link-to: IA + "likelihood-to-loss")[*`likelihood-to-loss.html`* — three synchronized panels: the *noise density* $p(r)$, the *loss* $-log p(r)$, and *data with a fitted line*. \
_Controls:_ Gaussian / Laplace / Uniform / Student-$t$ · noise scale · slope & intercept · *add an outlier* · per-point contributions.]
#pause
*Ask:* which model reacts most to an outlier? why does Uniform give an infinite barrier? which loss is robust?

// ══════════════════════════════ PART V ══════════════════════════════
= Classification and cross-entropy

== Classification should output probabilities

For $K$ classes:
$ p_theta (y=k | x) >= 0, quad quad sum_(k=1)^K p_theta (y=k | x) = 1 $
#pause
#fig("/lecture1/figures/softmax_pipeline.svg", w: 58%)
#align(center)[*logits* $z_k in RR$ → *probabilities* $p_k in [0,1]$.]

== Binary classification: Bernoulli model

$ Y | x tilde "Bernoulli"(p_theta (x)) $
#pause
Compact one-line form:
$ p(y | x, theta) = p^y (1-p)^(1-y) $
#pause
#two(notebox[$y=1 => p(y)=p$], notebox[$y=0 => p(y)=1-p$])

== Bernoulli NLL gives BCE
#stag(D)

$ -log p(y | x, theta) = -log(p^y (1-p)^(1-y)) $
#pause
$ = -y log p - (1-y) log(1-p) $
#pause
#result[Binary cross-entropy = Bernoulli NLL.]
#pause
#notebox[*Check.* True label $y=1$: if $p=0.8$ the loss is $-log 0.8 approx 0.22$; if $p=0.2$ it is $-log 0.2 approx 1.61$.]

== Why use logits?
#stag(V)

The network emits an unbounded score $z = f_theta (x) in RR$; the *sigmoid* maps it to a probability:
$ p = sigma(z) = 1/(1 + e^(-z)) $
#pause
#two(r: (1fr, 1.2fr),
  [$z -> -infinity: p -> 0$ \ $z = 0: p = 0.5$ \ $z -> +infinity: p -> 1$
   #v(4pt) #text(size: 16pt, fill: MUTED)[In practice sigmoid + BCE are fused for numerical stability.]],
  fig("/lecture1/figures/sigmoid.svg", w: 80%),
)

== Logistic regression = Bernoulli MLE
#stag(D)

Predict each label with the sigmoid of a linear score: $p_i = sigma(w^T x_i)$, and $Y_i | x_i tilde "Bernoulli"(p_i)$.
#pause
Likelihood of the whole labelled dataset (i.i.d.):
$ L(w) = product_(i=1)^n p_i^(y_i) (1 - p_i)^(1 - y_i) $
#pause
Its negative log — the objective we minimise:
$ J(w) = -sum_(i=1)^n [y_i log sigma(w^T x_i) + (1 - y_i) log(1 - sigma(w^T x_i))] $
#pause
#notebox[Exactly the per-example *BCE from two slides ago, summed over the data*. Now differentiate it in $w$.]

== The logistic-regression gradient
#stag(D)

One lemma — the sigmoid's own derivative:
$ sigma'(z) = sigma(z) (1 - sigma(z)) $
#pause
Differentiate $J$ through $log sigma$ and $log(1 - sigma)$; the $sigma(1 - sigma)$ cancels the chain factors and everything collapses to
$ (partial J)/(partial w_j) = sum_(i=1)^n (sigma(w^T x_i) - y_i) thin x_i^j $
#pause
$ ==> nabla_w J = sum_(i=1)^n (sigma(w^T x_i) - y_i) thin x_i = X^T (sigma(X w) - y) $
#pause
#result[The gradient is (*prediction* $-$ *target*) $times$ *input* — the same $p - y$ signal we will meet again for softmax.]

== Multi-class: categorical model

$ Y | x tilde "Categorical"(p_1, dots, p_K) $
#pause
For a one-hot target $y$:
$ p(y | x, theta) = product_(k=1)^K p_k^(y_k) $

== From logits to softmax

$ p_k = e^(z_k) / (sum_j e^(z_j)) $
#pause
Two guaranteed properties: $p_k > 0$ and $sum_k p_k = 1$.
#pause
#align(center, bars((2, 1, -0.5), labels: ("1", "2", "3"), softmax: true, title: [softmax probabilities]))
#align(center, text(size: 17pt)[*Shift invariance:* $"softmax"(z) = "softmax"(z + c bold(1))$.])

== Categorical NLL gives cross-entropy
#stag(D)

$ -log p(y | x, theta) = -log product_k p_k^(y_k) = -sum_k y_k log p_k $
#pause
Because $y$ is one-hot, only the true class survives:
$ cal(L) = -log p_"true class" $

== Confident mistakes are expensive
#stag(V)

$ ell(p_y) = -log p_y $
#pause
Read off the loss at three confidences for the *true* class:
#pause
#two(r: (1fr, 1.3fr),
  [$p_y = 0.9 => ell approx 0.105$ \ $p_y = 0.1 => ell approx 2.303$ \ $p_y = 0.001 => ell approx 6.908$],
  lines(
    fn: p => -calc.ln(p), domain: (0.001, 1.0), samples: 100,
    markers: false, x-label: [$p_y$], y-label: [$-log p_y$],
    points: ((0.9, -calc.ln(0.9), [0.11]), (0.1, -calc.ln(0.1), [2.30]), (0.001, -calc.ln(0.001), [6.91])),
    size: (56mm, 40mm)),
)
#pause
#align(center, text(size: 17pt)[The negative log makes a confident wrong prediction a large loss.])

== Checkpoint: confident mistakes #Q

#mcq(
  [Why does cross-entropy penalize a confidently wrong prediction strongly?],
  [It makes all class probabilities equal],
  [The true-class probability is near zero, so $-log p_y$ is large],
  [It adds an $L_2$ penalty to the logits],
  [The gradient is always zero],
)

== Answer: confident mistakes #A

#mcq-answer([B], [The true-class probability is near zero], [The negative log grows without bound as $p_y$ approaches zero, matching the fact that the model declared the observed class nearly impossible.])

== A remarkably simple gradient
#stag(D)

For softmax followed by cross-entropy:
$ (partial cal(L))/(partial z_k) = p_k - y_k, quad quad nabla_z cal(L) = p - y $
#pause
#align(center, bars((0.55, -0.70, 0.15), labels: ("1", "2", "3"), baseline: 0, title: [$p - y$ (gradient on the logits)]))
#align(center, text(size: 16pt, fill: MUTED)[For $p = (0.55, 0.30, 0.15)$, one-hot $y = (0, 1, 0)$: the signal $p - y$ is negative only on the true class — a preview of backprop.])

== Interactive: logits → softmax → cross-entropy
#stag(I)

#interbox(link-to: IA + "softmax-cross-entropy")[*`softmax-cross-entropy.html`* — controls $z_1, z_2, z_3$, the true class, and temperature $T$; presets for *correct & confident / correct & unsure / wrong & unsure / wrong & confident*. Shows $z -> "softmax"(z\/T) -> -log p_y -> p - y$.]
#pause
*Ask students to predict the loss before it is revealed.*

== Why not MSE for classification?
#stag(OPT)

MSE is not _impossible_ — it corresponds to the *wrong observation model* for class indicators.
#pause
#two(
  notebox[*MSE assumes* $Y_k = p_k + epsilon_k, epsilon_k tilde cal(N)(0, sigma^2)$],
  notebox[*Reality is* $Y tilde "Categorical"(p)$],
)
#pause
#align(center, text(size: 17pt)[Cross-entropy also gives *stronger gradients* on confident errors (where MSE's gradient vanishes).])

// ══════════════════════════════ PART VI ══════════════════════════════
= Bayes' rule, directly in classification

== Bayes' rule as a classifier

$ p(y=k | x) = (p(x | y=k) thin p(y=k)) / p(x) $
#pause
Prediction ignores the constant denominator:
$ hat(y) = arg max_k p(x | y=k) thin p(y=k) $
#pause
- $p(y=k)$ — class *prior*
#pause
- $p(x | y=k)$ — class-conditional *likelihood*
#pause
- $p(y=k | x)$ — *posterior* class probability

== A 1-D generative classifier
#stag(V)

$ X | Y=0 tilde cal(N)(mu_0, sigma_0^2), quad X | Y=1 tilde cal(N)(mu_1, sigma_1^2) $
#pause
#fig("/lecture1/figures/bayes_classifier.svg", w: 50%)
#align(center, text(size: 16pt, fill: MUTED)[Two class-conditional densities, a test point $x^star$, and its likelihood under each class.])

== Priors move the decision boundary
#stag(V)

#two(notebox[*Balanced* $p(Y=1) = 0.5$], notebox[*Rare* $p(Y=1) = 0.1$])
#fig("/lecture1/figures/priors_shift_boundary.svg", w: 54%)
#result[posterior $prop$ likelihood $times$ prior]

== Interactive: Bayes classifier
#stag(I)

#interbox(link-to: IA + "bayes-classifier")[*`bayes-classifier.html`* — controls the means $mu_0, mu_1$, the variances, the *prior* $p(Y=1)$, and a test point $x^star$. Shows $p(x | Y=0)$, $p(x | Y=1)$, the posterior $p(Y=1 | x)$, and the decision boundary together.]
#pause
*Try:* balanced → drop the positive prior to $0.1$ → why does the boundary move? then widen one variance.

// ══════════════════════════════ PART VII ══════════════════════════════
= Priors, MAP and regularization

== The limitation of MLE

MLE asks only: _how well does $theta$ explain the data?_
#pause
#fig("/lecture1/figures/overfit_polynomial.svg", w: 40%)
#notebox[Many parameters fit a small dataset. *Which should we prefer?*]

== Put a prior over parameters

$ theta tilde p(theta) $
_Before seeing this dataset, which parameter values are plausible?_
#pause
- small weights · sparse weights · smooth functions · correlated parameters
#pause
#notebox[A prior is an *inductive bias*.]

== Bayes' rule over parameters

$ p(theta | cal(D)) = (p(cal(D) | theta) thin p(theta)) / p(cal(D)) $
#pause
#fig("/lecture1/figures/prior_likelihood_posterior.svg", w: 34%)
#align(center, text(size: 17pt)[$p(cal(D)) = integral p(cal(D) | theta) p(theta) dif theta$ — the evidence (we stop here).])

== MAP estimation
#stag(D)

$ hat(theta)_"MAP" = arg max_theta p(theta | cal(D)) = arg max_theta p(cal(D) | theta) thin p(theta) $
#pause
Take negative logs:
$ hat(theta)_"MAP" = arg min_theta (-log p(cal(D) | theta) - log p(theta)) $
#pause
#result[data loss $+$ regularization]

== Gaussian prior produces $L_2$
#stag(D)

Assume $theta tilde cal(N)(0, tau^2 I)$. Then
$ -log p(theta) = 1/(2 tau^2) theta^top theta + C $
#pause
$ => quad cal(L)_"MAP" = -log p(cal(D) | theta) + lambda norm(theta)_2^2, quad lambda prop 1/tau^2 $
#pause
#notebox[*narrow* prior ⇒ *stronger* regularization; *broad* prior ⇒ *weaker*.]

== A general Gaussian prior → structured penalty
#stag(OPT)

For $theta tilde cal(N)(0, Sigma)$ the penalty is $1/2 theta^top Sigma^(-1) theta$.
#pause
- isotropic $Sigma$ → ordinary $L_2$
#pause
- unequal variances → penalize parameters *differently*
#pause
- correlations → penalize particular *directions*

== Laplace prior produces $L_1$
#stag(OPT)

$ p(theta_j) = 1/(2b) exp(-|theta_j|/b) => -log p(theta) = 1/b sum_j |theta_j| + C $
#pause
#result[Laplace prior ⇒ $L_1$]
#align(center)[Heavier tails at zero → *sparsity* (the lasso geometry, if time permits).]

== Interactive: likelihood, prior, MAP
#stag(I)

#interbox(link-to: IA + "map-regularization")[*`map-regularization.html`* — a two-parameter linear model so contours are visible. Shows *likelihood*, *prior*, and *posterior* contours with the *MLE*, *MAP*, and zero points. \
_Controls:_ amount of data · noise · prior variance · Gaussian vs Laplace · prior correlation.]
#pause
*Ask:* more data? narrower prior? why does MAP → MLE as data grows? a correlated prior?

// ══════════════════════════════ PART VIII ══════════════════════════════
= Full Bayes and the deep-learning objective

== MAP is still one parameter vector

#two(notebox[*MLE* $hat(theta)_"MLE"$ — one point], notebox[*MAP* $hat(theta)_"MAP"$ — one point])
#pause
#align(center)[*Full Bayes* keeps the whole posterior $p(theta | cal(D))$ — a *distribution*, not a point.]

== Posterior predictive distribution

$ p(y^star | x^star, cal(D)) = integral p(y^star | x^star, theta) thin p(theta | cal(D)) dif theta $
#pause
#fig("/lecture1/figures/posterior_samples.svg", w: 46%)
#notebox[Bayesian prediction *averages over plausible models* — and reports uncertainty.]

== Why DL usually uses point estimates
#stag(OPT)

A modern network has millions–billions of parameters, so integrating $p(theta | cal(D))$ is hard.
#pause
Common practical choices:
- MLE · MAP
#pause
- deep *ensembles*
#pause
- *dropout*-based approximations
#pause
- *variational* inference

== The standard supervised DL objective

#align(center)[
  $ min_theta underbrace(1/(|B|) sum_(i in B) -log p_theta (y_i | x_i), "likelihood / data fit") + underbrace(lambda norm(theta)_2^2, "prior / regularization") $
]
#pause
#align(center, text(size: 17pt)[probabilistic model $+$ neural network $+$ gradient optimization])
#pause
#notebox[DL doesn't discard probabilistic ML — it *scales* it with expressive functions and gradient descent.]

== Retrieval summary
#stag(V)

#two(
  [Gaussian likelihood \ Categorical likelihood \ Gaussian prior \ Laplace prior \ MLE + prior],
  [⇒ *MSE* \ ⇒ *cross-entropy* \ ⇒ *$L_2$* \ ⇒ *$L_1$* \ ⇒ *MAP*],
)

// final mental model — a metropolis "standout" (dark, centered)
#focus-slide[
  Final mental model
  #v(14pt)
  #set text(size: 22pt)
  #align(left)[
    See a *loss* → ask _"what likelihood produced it?"_ \
    See a *regularizer* → ask _"what prior produced it?"_ \
    See a *prediction* → ask _"point estimate or posterior?"_
  ]
  #v(16pt)
  #text(size: 16pt, fill: rgb("#B9C6C8"))[Next: computation graphs, gradients, and backpropagation.]
]
