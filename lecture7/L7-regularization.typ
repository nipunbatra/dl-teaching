// Generalization and Regularization — Lecture 7 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture7/L7-regularization.typ /tmp/L7.pdf
//   typst compile --root . --input handout=true lecture7/L7-regularization.typ /tmp/L7h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Generalization and Regularization],
  subtitle: [Making deep networks perform well on unseen data],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"

// native ml-plot: double descent — test error (bias + variance spike at the
// interpolation threshold c=1) and train error vs capacity (mirrors l7_figs.py
// f_double_descent). Reused on two slides, so bound once here.
#let dd-test = c => {
  let bias = 0.35 * calc.exp(-2.5 * c)
  let vf = if c < 1.0 { 1.0 } else { 0.55 }
  let vari = 0.28 / (calc.abs(c - 1.0) + 0.14) * vf
  calc.min(1.4, bias + vari * 0.35 + 0.12)
}
#let dd-train = c => calc.min(1.0, 0.9 * calc.exp(-2.2 * c))
#let dd-c31 = 0.05 + 31 * (2.95 / 99)   // capacity at old sample index 31 (interp.-threshold read-off)
#let double-descent-plot(size: (66mm, 42mm)) = lines(
  fn: (dd-test, dd-train),
  domain: (0.05, 3.0), samples: 99,
  colors: (ACC, TEAL),
  labels: ([test], [train]),
  markers: false,
  points: ((dd-c31, dd-test(dd-c31), [interp. threshold]),),
  x-label: [model size / capacity], y-label: [error],
  size: size,
)

// ── dropout: an MLP with two hidden units zeroed (a sampled subnetwork) ──
#let dropoutnet = align(center, diagram(spacing: (20mm, 4.5mm), {
  let L(li, i, n) = (li, i - (n - 1)/2)
  let dropped = (1, 3)
  // edges only through surviving hidden units
  for i in range(3) { for j in range(5) {
    if not dropped.contains(j) { edge(L(0,i,3), L(1,j,5), stroke: 0.4pt + MUTED.lighten(20%)) }
  }}
  for j in range(5) { for k in range(2) {
    if not dropped.contains(j) { edge(L(1,j,5), L(2,k,2), stroke: 0.4pt + MUTED.lighten(20%)) }
  }}
  // inputs
  for i in range(3) { node(L(0,i,3), none, radius: 3mm, fill: INK, stroke: none) }
  // hidden (dropped = dashed grey with ×)
  for j in range(5) {
    if dropped.contains(j) {
      node(L(1,j,5), text(size: 10pt, fill: MUTED)[×], radius: 3.2mm, fill: rgb("#EFEEEB"),
        stroke: (paint: MUTED, thickness: 0.7pt, dash: "dashed"))
    } else {
      node(L(1,j,5), none, radius: 3.2mm, fill: TEAL, stroke: none)
    }
  }
  // outputs
  for k in range(2) { node(L(2,k,2), none, radius: 3mm, fill: ACC, stroke: none) }
}))

#title-slide()

// ═══════════════════════════ PART I — The generalization problem ═══════════════════════════
= The generalization problem

== Where we are

The last lectures gave us a network that *trains*:
#pause
$ f_theta (x) -> hat(y), quad cal(L)(theta) = 1/n sum_(i=1)^n ell(f_theta (x_i), y_i) $
#pause
$ g_t = nabla_theta cal(L)(theta_t), quad quad theta_(t+1) = theta_t - eta thin g_t $
#pause
#notebox[Backprop gives the gradient; the optimizer drives $cal(L)_"train"$ down. We can now make the *training loss* small.]

== But is low training loss the goal? #Q

#mcq(
  [Does low training loss alone guarantee that a model will perform well on new data?],
  [Yes; training loss is the only metric that matters],
  [Yes, if the model has enough parameters],
  [No; it may have memorized the training set],
  [No; because gradient descent cannot reduce loss],
)

== Answer: training is not the goal #A

#mcq-answer([C], [It may have memorized the training set], [We care about expected performance on unseen data. A small training loss can coexist with a large generalization gap.])

== Train loss vs test loss #D

*Training loss* — an empirical average over the data we happened to collect:
#pause
$ cal(L)_"train"(theta) = 1/n sum_(i=1)^n ell(f_theta (x_i), y_i) $
#pause
*Test loss* — the expectation over the *true* data distribution $p_"data"$:
#pause
$ cal(L)_"test"(theta) = EE_((x,y) ~ p_"data") [ell(f_theta (x), y)] $
#pause
#notebox[We only ever *see finite data*; $cal(L)_"train"$ is a noisy estimate of the quantity we actually want, $cal(L)_"test"$.]

== The generalization gap

$ underbrace(cal(L)_"test" - cal(L)_"train", "generalization gap") $
#pause
#v(4pt)
*Overfitting*: training loss keeps falling while validation loss *turns up*.
#pause
#fig("/lecture7/figures/train_val.svg", w: 60%)

== Overfitting can memorize

Give a high-capacity net random labels $y -> y_pi$ (a fixed permutation): it can still drive $cal(L)_"train" -> 0$.
#pause
#alertbox[Validation remains at chance: fitting the training set is *not* evidence of a learned, general rule.]

== The plan for this lecture

We have networks that train. Now we must stop them from *memorizing*.
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Lever*], [*What it controls*]),
  [splits + early stopping], [when to stop],
  [weight decay / $ell_2$], [size of the weights],
  [dropout], [co-adaptation of units],
  [data augmentation / mixup], [invariance, linearity],
  [label smoothing], [overconfidence],
  [implicit regularization], [SGD noise, architecture, init],
))

// ═══════════════════════════ PART II — Splits & early stopping ═══════════════════════════
= Data splits and early stopping

== Three splits, three jobs

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left, left),
  table.header([*Split*], [*Used to*], [*Touches weights?*]),
  [*train*], [fit parameters $theta$], [yes — via gradients],
  [*validation*], [select model / hyperparameters], [no — only choices],
  [*test*], [final, unbiased estimate], [no — used *once*],
))
#pause
#alertbox[*Never* tune on the test set. Every peek leaks information and inflates your reported number.]

== Early stopping

Monitor validation loss each epoch; keep the *best* checkpoint, not the last:
$ t^star = arg min_t thin cal(L)_"val"(theta_t) $
#pause
#fig("/lecture7/figures/train_val.svg", w: 55%)
#align(center, text(size: 16pt, fill: MUTED)[train keeps falling; validation is U-shaped — stop at the bottom.])

== Worked example: early stopping #D

Six epochs of training. Which checkpoint do we keep?
#pause
#align(center, table(
  columns: 7, stroke: 0.5pt + MUTED, inset: (x: 9pt, y: 6pt), align: center,
  table.header([*epoch*], [1], [2], [3], [4], [5], [6]),
  [$cal(L)_"train"$], [1.20], [0.80], [0.45], [0.32], [0.28], [0.25],
  [$cal(L)_"val"$], [1.25], [0.98], [0.82], [#text(fill: GREEN, weight: 700)[0.78]], [0.84], [0.97],
))
#pause
#v(3pt)
- Training loss is *lowest at epoch 6* — but that is not what we want;
#pause
- Validation loss bottoms out at *epoch 4* ($0.78$), then rises → overfitting.
#pause
#result[keep $theta_4$ — the *last epoch is the wrong choice*]

== Early stopping as implicit regularization

Why does stopping early help?
#pause
- Weights *grow* over training; stopping early keeps them *small*;
#pause
- fewer effective updates ⇒ a *simpler* function is explored;
#pause
#notebox[Early stopping limits *effective capacity* without changing the architecture — a free, always-on regularizer. It is the cheapest one you have.]

== Interactive: train / val / overfit #I

#interbox(link-to: IA + "double-descent")[
  Sweep model capacity and training time and watch the train and validation curves separate — find the checkpoint where validation is lowest before the gap opens.
]
#pause
Find the checkpoint with the lowest *validation* loss rather than the last or lowest-training-loss checkpoint.

// ═══════════════════════════ PART III — Capacity, bias, variance ═══════════════════════════
= Capacity, bias and variance

== Three regimes

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 7pt), align: (left, center, center, left),
  table.header([*Regime*], [$cal(L)_"train"$], [$cal(L)_"val"$], [*Diagnosis*]),
  [underfit], [high], [high], [too little capacity],
  [good fit], [low], [low], [capacity \~ right],
  [overfit], [low], [high], [memorizing train],
))
#pause
#fig("/lecture7/figures/bias_variance_poly.svg", w: 74%)

== Bias–variance decomposition

Expected test error splits into three pieces:
$ EE[("error")^2] approx underbrace("bias"^2, "too simple") + underbrace("variance", "too sensitive") + underbrace(sigma^2, "irreducible noise") $
#pause
- *high bias* — the model is too simple; misses the true pattern (underfit);
#pause
- *high variance* — the model is too sensitive to the particular sample (overfit).
#pause
#notebox[More capacity ↓ bias but ↑ variance. Classically we trade one against the other.]

== Worked example: polynomial fits #V

Fit $y = sin(2 pi x) + epsilon$ from noisy samples with polynomials of degree $d$:
#pause
#fig("/lecture7/figures/bias_variance_poly.svg", w: 82%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[$d = 1$ underfits (high bias) · $d = 5$ fits · $d = 15$ overfits (high variance)])

== The deep-learning twist

Classical wisdom: *bigger model ⇒ more overfitting*. Modern practice complicates this.
#pause
- huge over-parameterized nets often generalize *well*, not worse;
#pause
- what governs generalization: *optimization, data, architecture, implicit bias* — not just parameter count;
#pause
#align(center, double-descent-plot(size: (66mm, 40mm)))

== Double descent

#two(
  double-descent-plot(size: (72mm, 46mm)),
  [
    Past the interpolation threshold, test error can *fall again* — the modern "second descent".
    #v(6pt)
    #notebox[Parameter count is a poor proxy for capacity in deep nets. #text(fill: MUTED)[(active research)]]
  ],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[But *validation discipline is still non-negotiable* — it is how we know what is happening.])

// ═══════════════════════════ PART IV — Weight decay & L2 ═══════════════════════════
= Weight decay and $ell_2$ regularization

== Penalize large weights

Add a penalty on the *size* of the weights to the data loss:
#pause
$ cal(L)_"reg"(theta) = cal(L)_"data"(theta) + lambda norm(theta)^2 $
#pause
- $lambda > 0$ — the *regularization strength* (a hyperparameter);
#pause
- large weights ⇒ sharp, wiggly functions; the penalty prefers *smooth* ones.
#pause
#notebox[$ell_2$ regularization $<=>$ a *Gaussian prior* on the weights: MAP estimation with $theta ~ cal(N)(0, sigma^2 I)$ gives exactly this penalty.]

== The gradient of the penalty #D

Differentiate $cal(L)_"reg" = cal(L)_"data" + lambda norm(theta)^2$:
$ nabla cal(L)_"reg" = nabla cal(L)_"data" + 2 lambda thin theta $
#pause
Substitute into the SGD step:
$ theta_(t+1) = theta_t - eta (nabla cal(L)_"data" + 2 lambda thin theta_t) $
#pause
$ theta_(t+1) = (1 - 2 eta lambda) thin theta_t - eta thin nabla cal(L)_"data" $
#pause
#result[a *multiplicative shrinkage* $(1 - 2 eta lambda)$ before the gradient step]

== Why "weight decay"?

Each step first *shrinks* the weight toward zero, then applies the gradient:
$ theta_t arrow.r.bar (1 - eta lambda') thin theta_t $
#pause
#notebox[Absent any gradient, weights *decay* geometrically toward $0$ — hence the name. $ell_2$ regularization and weight decay are the same mechanism (for SGD).]

== Worked example: one decay step #D

$theta = 5$, gradient $g = 3$, learning rate $eta = 0.1$, decay $lambda' = 0.2$:
#pause
*Without* weight decay — a plain gradient step:
$ theta^+ = theta - eta thin g = 5 - 0.1 dot 3 = 4.7 $
#pause
*With* weight decay, shrink first, then step: $theta^+ = (1 - eta lambda') theta - eta g$.
#pause
Step 1 — the shrink factor: #h(0.5em) $1 - eta lambda' = 1 - 0.1 dot 0.2 = 0.98$.
#pause
Step 2 — shrink the weight: #h(0.5em) $0.98 dot 5 = 4.9$.
#pause
Step 3 — subtract the gradient step: #h(0.5em) $4.9 - 0.1 dot 3 = 4.9 - 0.3 = 4.6$.
#pause
#align(center, text(size: 16pt, fill: MUTED)[the extra $-0.1$ ($4.7$ vs $4.6$) is the shrinkage pulling the weight toward zero.])

== Adam ≠ Adam + $ell_2$: AdamW #D

For *SGD*, adding $lambda norm(theta)^2$ to the loss $equiv$ weight decay (up to scaling).
#pause
For *Adam* they *differ*: the $2 lambda theta$ term gets divided by $sqrt(hat(v))$ like any gradient — so decay is *distorted* per coordinate.
#pause
Loshchilov & Hutter *decoupled* it — apply decay directly to the weights:
$ theta arrow.r.bar "AdamStep"(theta), quad quad theta arrow.r.bar (1 - eta lambda) thin theta $
#pause
#result[use *AdamW*, not Adam with naive $ell_2$]

== When (and when not) to decay

- Use it almost everywhere: *MLPs, CNNs, Transformers*, large models;
#pause
- typical $lambda$: $10^(-4)$ to $10^(-2)$ (tune on validation);
#pause
- *do not* decay *biases*, *norm* scale/shift, and often not *embeddings* — decaying them helps nothing and can hurt.
#pause
#notebox[Rule of thumb: decay the *weight matrices*, leave the *one-dimensional* parameters alone.]

== Weight decay shrinks the weights #V

Weight norm $norm(theta)$ over training for several $lambda$:
#pause
// native ml-plot: ‖θ‖ = 3(1−e^(−0.10 s))/(1+8λ) + 0.15 (mirrors l7_figs.py f_weight_decay)
// closed-form settling of the weight norm; wd-series returns the fn of the step s
#let wd-series(lam) = s => 3.0 * (1 - calc.exp(-0.10 * s)) * (1.0 / (1.0 + 8.0 * lam)) + 0.15
#align(center, lines(
  fn: (wd-series(0.0), wd-series(0.01), wd-series(0.05), wd-series(0.15)),
  domain: (0, 59), samples: 59,
  colors: (INK, BLUE, TEAL, ACC),
  markers: false,
  x-label: [training step], y-label: [weight norm $norm(theta)$],
  size: (82mm, 46mm),
))
#align(center, text(size: 15pt)[#text(fill: INK)[$lambda = 0$] · #text(fill: BLUE)[$lambda = 0.01$] · #text(fill: TEAL)[$lambda = 0.05$] · #text(fill: ACC)[$lambda = 0.15$]])
#align(center, text(size: 16pt, fill: MUTED)[larger $lambda$ ⇒ smaller final weights ⇒ smoother function.])

== Interactive: weight decay #I

#interbox(link-to: IA + "weight-decay")[
  sweep $lambda$ and watch the weight norm shrink and the decision boundary smooth out; too large and the model underfits.
]
#pause
Compare the unchanged shrink factor with the doubled data-gradient step when $eta$ changes.

// ═══════════════════════════ PART V — Dropout ═══════════════════════════
= Dropout

== The idea

During training, randomly *drop* each hidden unit with probability $1 - p$:
#pause
$ m_j ~ "Bernoulli"(p), quad quad tilde(h) = m dot.o h $
#pause
- a different random *subnetwork* is trained on every minibatch;
#pause
- units cannot rely on any single partner ⇒ less *co-adaptation*.
#pause
#fig("/lecture7/figures/dropout_mask.svg", w: 52%)

== Inverted dropout: keep the scale #D

Dropping units shrinks the expected activation. *Rescale* by $1 / p$ during training:
$ tilde(h) = (m dot.o h) / p $
#pause
Then the expected activation is *unchanged*:
$ EE[tilde(h)_j] = 1/p dot (p dot h_j + (1-p) dot 0) = h_j $
#pause
#result[no test-time rescaling needed — at test time just use the full network]

== Worked example: inverted dropout #D

$h = [2, 4, 6, 8]$, keep prob $p = 0.5$, sampled mask $m = [1, 0, 1, 0]$:
#pause
Step 1 — apply the mask (units 2 and 4 are dropped): #h(0.4em) $m dot.o h = [2, 0, 6, 0]$.
#pause
Step 2 — rescale survivors by $1 \/ p = 2$:
$ tilde(h) = (m dot.o h) / p = ([2, 0, 6, 0]) / 0.5 = [4, 0, 12, 0] $
#pause
Check the expectation (each unit kept w.p. $0.5$, scaled by $2$):
$ EE[tilde(h)] = 0.5 dot (h / 0.5) = h = [2, 4, 6, 8] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[surviving units are *doubled*, so the layer's expected output is preserved.])

== Why it helps: model averaging

Each mask trains a *different subnetwork* — dashed units are dropped this step:
#pause
#dropoutnet
#pause
We fit exponentially many subnetworks; at test the full net *approximates their average*.
#pause
#notebox[An *ensemble-like* effect from one model — but an intuition, not an exact Bayesian average. Do not overclaim it.]

== Dropout caveats

- Works best on *fully-connected* layers and *classifier heads*;
#pause
- less useful alongside strong *augmentation* and *normalization* (which already regularize);
#pause
- adds gradient noise ⇒ can *slow optimization*, needs more epochs;
#pause
- tune the *keep* rate $p approx 0.5$–$0.8$ (i.e. drop $20$–$50%$).

== Interactive: dropout #I

#interbox(link-to: IA + "dropout-playground")[
  Toggle dropout on a small MLP, sweep the rate, and watch training vs validation loss — see how too much drop underfits and too little overfits.
]

// ═══════════════════════════ PART VI — Data augmentation ═══════════════════════════
= Data augmentation

== Regularize by enlarging the data

Generate new *label-preserving* examples from the ones you have:
$ (x, y) arrow.r.bar (T(x), y), quad quad T "preserves the label" $
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Domain*], [*Transforms $T$*]),
  [images], [crop, flip, rotate, color jitter, noise],
  [audio / time series], [time-shift, mask spans, add noise],
  [text], [paraphrase, synonym swap, back-translation],
))

== Augmentation teaches invariance #V

If $T$ preserves the label, we want the model to be *invariant* to it:
$ f_theta (x) approx f_theta (T(x)) $
#pause
#fig("/lecture7/figures/augmentation.svg", w: 82%)
#align(center, text(size: 16pt, fill: MUTED)[same object, many views — the network learns to ignore the nuisance transform.])

== Bad augmentation encodes wrong knowledge

Augmentation *injects domain knowledge* — the wrong $T$ injects the wrong knowledge:
#pause
- flipping a *6* into a *9* — the label changed;
#pause
- *time-reversing* speech or a heartbeat — no longer valid;
#pause
- *color* jitter when *color is the label* (ripe vs unripe fruit);
#pause
- *cropping out* the object you are trying to classify.
#pause
#alertbox[$T$ must preserve the label. Choosing $T$ *is* a modelling decision about what should not matter.]

== Mixup: interpolate examples

Blend two examples *and* their labels with $lambda ~ "Beta"(alpha, alpha)$:
$ tilde(x) = lambda x_i + (1 - lambda) x_j, quad quad tilde(y) = lambda y_i + (1 - lambda) y_j $
#pause
#fig("/lecture7/figures/mixup.svg", w: 46%)
#align(center, text(size: 16pt, fill: MUTED)[convex combinations encourage *linear* behaviour between examples.])

== Worked example: mixup #D

$x_A$ = cat, $y_A = [1, 0]$; $x_B$ = dog, $y_B = [0, 1]$; mix $lambda = 0.7$:
#pause
The complementary weight is $1 - lambda = 0.3$. Blend the inputs:
$ tilde(x) = 0.7 thin x_A + 0.3 thin x_B $
#pause
Blend the labels the *same* way:
$ tilde(y) = 0.7 [1, 0] + 0.3 [0, 1] = [0.7, 0.3] $
#pause
Train with the *soft* cross-entropy against this target:
$ ell = - sum_k tilde(y)_k log p_k = -0.7 log p_1 - 0.3 log p_2 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the network is asked to be $70%$ cat, $30%$ dog — no hard boundary jump.])

== Interactive: augmentation + mixup #I

#interbox(link-to: IA + "augmentation-mixup")[
  apply transforms and mixup to a 2-D dataset and watch the decision boundary become smoother and more invariant.
]
#pause
Observe how soft targets discourage infinite-margin, overconfident logits.

// ═══════════════════════════ PART VII — Label smoothing ═══════════════════════════
= Label smoothing

== Hard labels push to extremes

With a one-hot target, cross-entropy keeps pushing $p_y -> 1$:
$ ell = - log p_y quad ==> quad "logits" -> plus.minus infinity $
#pause
#notebox[The model is rewarded for *ever more extreme* confidence — even on ambiguous or mislabelled examples. This encourages *overconfidence*.]

== Softening the target #D

Move a little mass $epsilon$ off the correct class onto the rest:
$ y_k^"smooth" = cases(1 - epsilon & "if " k = y, epsilon \/ (K - 1) & "otherwise") $
#pause
Equivalently, mix the one-hot with a uniform distribution over the *other* classes.
#pause
#fig("/lecture7/figures/label_smoothing.svg", w: 60%)

== Worked example: label smoothing #D

$K = 5$ classes, smoothing $epsilon = 0.1$, correct class $= 2$:
#pause
Correct class keeps most of the mass: #h(0.4em) $1 - epsilon = 0.9$.
#pause
The remaining $epsilon = 0.1$ is split over the $K - 1 = 4$ others: #h(0.4em) $epsilon/(K-1) = 0.1/4 = 0.025$.
#pause
Assemble the target vector:
$ y^"smooth" = [0.025, 0.025, thick 0.9, thick 0.025, 0.025] $
#pause
Verify it is a distribution: #h(0.4em) $0.9 + 4 dot 0.025 = 1$.
#pause
#align(center, text(size: 16pt, fill: MUTED)[still sums to $1$; the target is achievable with *finite* logits.])

== Why it helps (and when it hurts)

- *discourages extreme confidence* — logits stay bounded;
#pause
- improves *calibration* (predicted probabilities match accuracy);
#pause
- adds *robustness to noisy labels* and often improves generalization;
#pause
#alertbox[Can *hurt* when you need exact confidences — e.g. *knowledge distillation* or when downstream code reads calibrated probabilities.]

== Interactive: label smoothing #I

#interbox(link-to: IA + "calibration")[
  Sweep $epsilon$ and watch the reliability diagram — see over-confident predictions pull back toward the diagonal as smoothing increases.
]

// ═══════════════════════════ PART VIII — Noise & implicit regularization ═══════════════════════════
= Noise and implicit regularization

== Noise is a regularizer

Many effective regularizers work by *injecting noise* during training:
#pause
- *SGD* minibatch sampling — noisy gradients;
#pause
- *dropout* — noisy activations;
#pause
- *augmentation* — noisy inputs;
#pause
- *stochastic depth*, *label / input noise*.
#pause
#notebox[Noise prevents the network from building *brittle*, precise dependencies — it must be robust to the perturbation.]

== Explicit vs implicit regularization

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left),
  table.header([*Explicit* (a term / operation)], [*Implicit* (a side effect)]),
  [$+ lambda norm(theta)^2$ penalty], [SGD gradient noise],
  [dropout], [architecture / optimizer bias],
  [data augmentation], [early stopping],
  [label smoothing], [normalization, initialization],
))
#pause
#align(center, text(size: 17pt, fill: INK)[Regularization is *not only a penalty term*.])

== Regularization changes the learned function #V

Two models can both fit the training set to $cal(L)_"train" approx 0$ — yet differ:
#pause
#fig("/lecture7/figures/decision_boundary_reg.svg", w: 76%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[regularization *picks* the smoother boundary — smaller norm, more invariance, less confidence, better val.])

// ═══════════════════════════ PART IX — Practical workflow ═══════════════════════════
= A practical workflow

== First: debug before you regularize #Q

#mcq(
  [Both training and validation loss are high. What should be tried before adding dropout?],
  [Add stronger regularization immediately],
  [Improve the fit: capacity, optimization, data checks, or training time],
  [Stop training earlier],
  [Increase label smoothing until training loss vanishes],
)

== Answer: diagnose underfitting first #A

#mcq-answer([B], [Improve the fit first], [High training loss signals underfitting or a pipeline issue. More regularization would make fitting even harder.])

Fix the fit first: *bigger model*, *train longer*, *better LR*, *less augmentation*, *check the labels*.
#pause
#alertbox[Adding regularization to an underfit model makes it *worse*. Regularize only once you can *overfit*.]

== Then: regularize an overfit model

Training loss *low*, validation loss *high* ⇒ overfitting. Now reach for:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Lever*], [*First choice when...*]),
  [more data / augmentation], [cheap to collect more or transform],
  [weight decay], [almost always — near-free],
  [dropout], [MLP / classifier head],
  [early stopping], [always on],
  [smaller model], [data is truly limited],
  [label smoothing], [over-confident / noisy labels],
))

== A regularization dashboard

Track more than the loss — watch *what* is being controlled:
#pause
- $cal(L)_"train"$ *and* $cal(L)_"val"$ (and the gap between them);
#pause
- $"acc"_"train"$ *and* $"acc"_"val"$;
#pause
- weight norm $norm(theta)$;
#pause
- confidence $max_k p_k$ and *calibration error*.
#pause
#notebox[If the *gap* widens while train keeps improving, you are overfitting — turn a regularization knob.]

== Worked diagnosis #D

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 7pt), align: (center, center, center, left),
  table.header([*Model*], [$"acc"_"train"$], [$"acc"_"val"$], [*Action*]),
  [A], [65%], [63%], [underfit → *add capacity*, train longer],
  [B], [99%], [72%], [overfit → *regularize* / augment / early-stop],
))
#pause
#v(3pt)
The gap is $"acc"_"train" - "acc"_"val"$: #h(0.5em) A $-> 2%$, #h(0.4em) B $-> 27%$.
#pause
- *A*: small gap, both low ⇒ the model *cannot fit* — do *not* regularize;
#pause
- *B*: large gap ⇒ the model *memorized* — rein it in.

== Practical defaults

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Setting*], [*Default recipe*]),
  [images (CNN)], [augmentation + weight decay + early stopping],
  [MLP / tabular], [weight decay + dropout + early stopping],
  [Transformer], [AdamW + dropout + (label smoothing) + aug],
  [small data], [transfer learning + strong validation],
))
#pause
#notebox[Start simple, add one knob at a time, and let *validation* decide whether it helped.]

// ═══════════════════════════ PART X — Summary ═══════════════════════════
= Summary

== One table: every method

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left),
  table.header([*Method*], [*What it controls*]),
  [early stopping], [effective complexity — when to stop],
  [weight decay], [parameter norm $norm(theta)$],
  [dropout], [co-adaptation / activation noise],
  [data augmentation], [invariance to nuisance transforms],
  [mixup], [linearity between examples],
  [label smoothing], [overconfidence / calibration],
  [validation split], [model selection — the judge],
))

== Final mental model — one idea

#align(center, text(size: 22pt)[
  Optimization *reduces training loss*. \
  Regularization *improves unseen-data performance*. \
  #v(4pt)
  #text(size: 18pt, fill: MUTED)[validation tells us whether it is actually helping.]
])

#focus-slide[
  Optimization reduces train loss; regularization improves generalization; validation is the judge.
  #v(12pt)
  #set text(size: 22pt)
  Next: *CNNs* — why fully-connected nets are inefficient for images.
]
