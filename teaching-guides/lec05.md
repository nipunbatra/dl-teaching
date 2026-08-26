# Lecture 5 · Optimization for Deep Learning · Teaching Guide

*Instructor companion for the rebuilt public Lecture 5 deck and its three exact notebooks.*

## The spine

Backpropagation gives a gradient; optimization decides **which data measure that
gradient, what history to remember, how strongly to act, and which coordinates
the model is allowed to use**.

Teach the lecture as one progressive diagnosis, not an optimizer catalogue:

1. retrieve vanilla full-batch gradient descent on one four-point line fit;
2. change only the data used per update: one example, then a minibatch;
3. explain the centre and spread of that stochastic gradient estimator;
4. derive a genuine ravine from a second, scaled-feature regression problem;
5. introduce EWMA from a noisy-signal analogy, contrast first-reading and
   zero-start initialization on those same observations, then carry the idea of
   empty memory into EWMA-form momentum and verify its effect on the ravine;
6. answer the feature-standardization objection, then use a separate
   standardized-but-correlated regression to show what can remain;
7. build RMSProp from per-coordinate recent magnitude and Adam from first- and
   second-raw-moment memories, normalizing Adam's two zero-start memories only
   after the temperature and ordinary-momentum cases are understood;
8. reduce late stochastic wandering with learning-rate decay; and
9. motivate positive scales, bounded coefficients, Bernoulli probabilities, and
   simplex weights from their models, then enforce them by changing coordinates
   or, for simple closed sets, projecting.

The two recurring model examples are deliberately small enough to calculate
and plot exactly. The synthetic temperature sequence exists only to make EWMA
intuitive before the signal becomes a gradient.

## Where it sits

- **Retrieves Lectures 2–3:** the loss surface, gradient geometry, and the vanilla
  update $\hat{\boldsymbol g}_t=\nabla_{\boldsymbol\theta}
  \mathcal L_{\mathcal B_t}(\boldsymbol\theta_t)$,
  $\boldsymbol\theta_{t+1}=\boldsymbol\theta_t-\eta\hat{\boldsymbol g}_t$.
- **Builds directly on Lecture 4:** backprop computes the current batch gradient;
  this lecture turns that measurement into an update.
- **Sets up Lecture 6:** initialization, activations, normalization, and residual
  paths make useful activations and gradients survive depth.
- **Leaves optimizer regularization to Lecture 7:** keep that separate from this
  lecture's direction, scale, and step-size mechanisms.

## Notation contract

Keep the notation inherited from Lectures 1–3 visible throughout:

- bold lowercase symbols are vectors: $\boldsymbol x_i$, $\boldsymbol\theta$,
  $\hat{\boldsymbol g}$, $\boldsymbol m$, and $\boldsymbol v$;
  $y_i$, $\hat y_i$, $\ell_i$, and $\mathcal L$ are scalars.
  Example A intentionally uses a single scalar feature $x_i$;
- bold uppercase symbols are matrices, such as the design matrix $\mathbf X$;
- $\hat{\boldsymbol g}_t=\nabla_{\boldsymbol\theta}
  \mathcal L_{\mathcal B_t}(\boldsymbol\theta_t)$ is the current gradient
  measurement supplied to the optimizer. It may be full-batch or minibatch;
  when $\mathcal B_t$ is the complete training set it equals the exact empirical
  gradient $\boldsymbol g_t=\nabla_{\boldsymbol\theta}
  \mathcal L(\boldsymbol\theta_t)$;
- for conceptual momentum, $\boldsymbol m_t$ is the normalized EWMA state
  $\boldsymbol m_{t+1}=\beta\boldsymbol m_t+
  (1-\beta)\hat{\boldsymbol g}_t$, exactly matching the temperature recurrence.
  PyTorch instead stores the unnormalized buffer
  $\widetilde{\boldsymbol m}_{t+1}=\beta
  \widetilde{\boldsymbol m}_t+\hat{\boldsymbol g}_t$, so
  $\boldsymbol m=(1-\beta)\widetilde{\boldsymbol m}$ and its learning rate is
  scaled accordingly. Adam also uses the conventional symbol
  $\boldsymbol m_t$ for its signed first raw moment. $\boldsymbol v_t$ is
  squared-gradient, second-raw-moment scale memory, used by RMSProp and Adam;
  when Adam combines both memories, standalone momentum's $\beta$ is written
  $\beta_1$, while RMSProp's $\beta_2$ carries over. The subscripts identify
  the two roles; they do not imply equal numerical values;
- the optimizer clock is zero-based and advances once per executed
  `optimizer.step()`: parameters $\boldsymbol\theta_t$ produce measurement
  $\hat{\boldsymbol g}_t$, then state $\boldsymbol m_{t+1}$ and/or
  $\boldsymbol v_{t+1}$ and parameters at $t+1$. This is an update-clock
  convention: an executed step still counts when a zero gradient produces no
  numerical parameter change;
- $\epsilon_{\mathrm{opt}}$ stabilizes an optimizer denominator; it is unrelated
  to the primary positive-parameter convention $a=\log\sigma$,
  $\sigma=\exp(a)$. Any small floor added to an alternative `softplus` map is a
  separate modelling choice; and
- the standalone temperature EWMA uses the standard one-based signal notation
  $x_t\mapsto m_t$. The smoothing curves initialize at the first reading
  $m_1=x_1$; the cold-start bridge deliberately reuses the same readings with
  $m_0=0$, for which the observed-weight mass after day $t$ is $1-\beta^t$ and
  $\widehat m_t=m_t/(1-\beta^t)$ removes only the artificial pull toward zero;
- ordinary momentum carries its uncorrected EWMA state
  $\boldsymbol m_{t+1}$ as optimizer dynamics. Although this zero-start state
  initially has less than unit gradient weight, the optimizer deliberately
  does not divide it by the missing mass. Hats on
  $\widehat{\boldsymbol m}_{t+1}$ and
  $\widehat{\boldsymbol v}_{t+1}$ are reserved for Adam's zero-start
  corrections; they are not new stochastic-gradient estimators; and
- once the scalar memory is coupled to an optimizer, return to the zero-based
  clock above: measurement $\hat{\boldsymbol g}_t$ creates state $t+1$, so its
  accumulated mass is $1-\beta^{t+1}$.

## What students should leave able to do

Students should be able to:

- perform one full-batch, one-sample, and minibatch update on the same loss;
- compare full-batch GD, SGD, and minibatch GD in update frequency, gradient
  noise, parallelism, and late motion;
- distinguish batch size, iteration, optimizer update, and epoch;
- state the sampling assumption under which an SGD gradient estimator is
  unbiased;
- explain why unbiasedness neither guarantees a downhill step nor low variance;
- explain why minibatch averaging preserves the mean and reduces covariance;
- derive the ravine loss in the original parameter coordinates from the
  displayed design matrix; distinguish its shallow and steep parameter
  directions, overshoot, and oscillation; explain what feature standardization
  repairs; and explain why unit variance does not remove feature correlation;
- expand an EWMA, explain the smoothness-versus-lag trade-off across
  $\beta=.5,.9,.98$, and calculate the missing weight mass created by
  $m_0=0$ on the same temperature sequence;
- implement EWMA-form momentum from scratch with the same conceptual learning
  rate as GD, recognize PyTorch's unnormalized-buffer rate mapping as an
  implementation detail, and distinguish ordinary momentum's uncorrected
  dynamical state from Adam's explicitly corrected moment estimates;
- explain RMSProp as an EWMA of squared gradients initialized with
  $\boldsymbol v_0=\boldsymbol0$, with `square_avg` storing
  $\boldsymbol v$ and ruler $\sqrt{\boldsymbol v}+\epsilon_{\mathrm{opt}}$;
- explain Adam's signed first raw moment, squared second raw moment, and the
  zero-start bias corrections before combining direction and scale;
- inspect `momentum_buffer`, `square_avg`, `exp_avg`, and `exp_avg_sq`, and
  verify the displayed RMSProp/Adam recurrences against PyTorch;
- state which clock advances a learning-rate schedule;
- turn positive, box-constrained, probability, and simplex problems into
  unconstrained raw-parameter problems using log scale plus `exp`, affine
  `sigmoid`, logits, and `softmax`; and choose projection instead when a simple
  closed boundary is part of the model, including the optimizer-state caveat.

## Evidence language

| Material | Honest label | How to describe it |
|---|---|---|
| Four-point line fit, gradients, losses, contours, and decay plots | **Constructed data; computed exactly** | “Every number and trajectory comes from the four displayed examples.” |
| Batch-size averaging diagram | **Exact enumeration at one fixed parameter value** | “The arrows are computed from Example A, but the picture suppresses numbers to isolate the averaging intuition.” |
| Individual and minibatch loss-surface sequence | **Exact losses and gradients at one shared parameter value** | “The deck separates surfaces from arrows, then shows the worked $B=1,2,4$ family; Notebook 00 exposes the underlying calculations.” |
| Scaled-feature ravine and standardized-correlated control | **Constructed regressions; computed empirical losses** | “The first isolates unequal units; the second uses four exact rows whose feature variances are one but whose correlation is $4/5$, giving condition number $9$ and contour-axis ratio $3$.” |
| Temperature sequence | **Fixed-seed synthetic signal; computed recurrence** | “The same 120-day observations isolate EWMA's memory. First only $\beta$ changes; then the first 12 readings are reused to compare $m_1=x_1$ with a raw and missing-mass-corrected zero start.” |
| Estimator arrows and batch-spread plots | **Exact enumeration at one fixed parameter value** | “The centre and spread are computed over all displayed samples or subsets.” |
| Optimizer-path comparison | **Mechanism demonstration** | “Rates are method-specific; this is not a leaderboard or universal ranking.” |
| Throughput and hardware statements | **Qualitative systems guidance** | “Actual wall-clock performance depends on vectorization, memory, and hardware.” |
| Feasible-set picture, constraint maps, and projected path | **Conceptual illustration plus computed toy examples** | “The picture introduces legal versus illegal values; every scalar transform and optimizer path is then computed. None is a convergence guarantee for arbitrary constrained problems.” |

Use the deck tags as teaching cues: `[Q]` requires a prediction, `[D]` deserves at
least one line of live mathematics, `[V]` should be read from the picture before
the caption, and `[I]` is a notebook or code checkpoint.

## Before class

1. Open the handout and presentation PDFs.
2. Open these exact local notebooks and restart/run all cells in order:
   - `notebooks/L05/00_full_batch_sgd_minibatch.ipynb`
   - `notebooks/L05/02_ewma_momentum.ipynb`
   - `notebooks/L05/03_adaptive_optimizers_constraints.ipynb`
3. Keep one board panel for the four-point line-fit gradients and one for the
   ravine loss, directions, and first overshooting step. Do not erase them when
   the optimizer names change.
4. The course page uses Colab URLs on the public `master` branch. Until this
   worktree is published to that repository, use the local notebooks and do not
   tell students that the three new Colab targets are live.

## 80-minute canonical route through the 110-page handout

This is the default live route. The labels implement the course timing contract:

- **[CORE]** is non-negotiable: it carries the causal story or a mastery check.
- **[SHOULD-COVER]** is taught briefly when the room is on pace and otherwise
  assigned as immediate follow-along.
- **[OPTIONAL]** is enrichment for the 110-minute route or after class.

The page numbers below are physical PDF pages and count the title page.
Separators are used only to pose the next failure; do not spend time reading them.

The synchronized handout has 110 physical pages, and the rebuilt presentation
has 128 states because a few handout pages reveal in stages. Navigate by slide
title, and finish all states carrying that title before advancing. The final
physical-page/state synchronization is:

| Handout pages | Presentation states | Arc |
|---|---:|---|
| 1 | 1 | Title |
| 2–4 | 2–6 | Bridge, outline, and examples |
| 5–16 | 7–33 | Full batch, SGD, and minibatches |
| 17–19 | 34–36 | Batch-size comparison and PyTorch |
| 20–30 | 37–48 | Loss surfaces and estimator reasoning |
| 31–37 | 49–55 | Exact-gradient ravine |
| 38–57 | 56–75 | EWMA and momentum |
| 58–62 | 76–80 | Standardization and correlated control |
| 63–75 | 81–93 | RMSProp |
| 76–85 | 94–103 | Adam |
| 86–93 | 104–111 | Learning-rate schedules |
| 94–108 | 112–126 | Constrained parameters |
| 109–110 | 127–128 | Retrieval and synthesis |

### 0–5 min · Pages 2–4 · bridge and recurring examples

- **[CORE] Page 2, “Backprop measured the slope; today we decide how to
  move.”** Separate the measured gradient from the optimizer's update state.
- **[CORE] Page 3, “Outline.”** Read only the five verbs: choose, understand,
  expose, add, finish.
- **[CORE] Page 4, “Two examples will keep the mechanisms concrete.”**
  Establish Example A for sampling and Example B for geometry.

### 5–16 min · Pages 5–16 · full batch → SGD → minibatch

- **[CORE] Page 5, “How much data should define one update?”** Ask for the two
  extremes before naming them.
- **[SHOULD-COVER] Page 6, “What optimization means in this course.”** Fix the
  empirical mean convention and the forward/backward/optimizer boundary.
  Re-establish that $\boldsymbol x_i$ and $\boldsymbol\theta$ are vectors while
  $y_i$, $\hat y_i$, $\ell_i$, and $\mathcal L$ are scalars, and retain the
  prediction-first loss order $\ell(\hat y_i,y_i)$ used by PyTorch APIs.
- **[CORE] Pages 7–9, “Revision: vanilla gradient descent is a four-line loop,”
  “Example A: compute one full-batch gradient,” and “Average, then update.”**
  Keep parameter order $(b,w)$. At $\boldsymbol\theta_0=(2,-1)$ the residuals
  are $(4,1,-2,-6)$, the full gradient is
  $\hat{\boldsymbol g}_0=(-.75,-4.5)$, and $\eta=.10$ gives
  $\boldsymbol\theta_1=(2.075,-.55)$ with loss
  $7.125\rightarrow5.2153$.
- **[SHOULD-COVER] Page 10, “One gradient step changes the fitted model.”**
  Point out that the fitted line and contour point encode the same parameters.
- **[CORE] Page 11, “Batch gradient descent is exact—but every update must
  wait.”** Make the clock concrete: the first update follows $N$ example
  contributions; one full-batch epoch is one iteration and one update; $E$
  epochs process $E N$ examples and make $E$ updates. Distinguish example-work
  from wall-clock time because hardware can evaluate examples in parallel.
- **[CORE] Page 12, “The training clock: examples, iterations, updates,
  epochs.”** Define the terms before reusing them. For $N=4,B=2$, show two
  iterations per epoch and normally two updates. State the general count
  $K=\lceil N/B\rceil$, then recover full batch ($K=1$) and SGD ($K=N$).
- **[CORE] Pages 13–15, “Full batch waits for all $N$; SGD updates from one
  random example,” “Each example proposes a different update,” and “Worked
  check: $\ell_2$ decreases, but $\mathcal L$ increases.”**
  Start from the full-batch delay before defining *stochastic*: the random
  variable is the sampled index $I_t$, while the model and objective stay
  fixed. Under the displayed uniform sampling scheme, inspect all four exact
  per-example losses and their different gradient proposals at the shared
  $\boldsymbol\theta_0=(2,-1)$. Keep the sampled $I_0=2$ proposal distinct
  from the three alternatives. Its estimator is
  $\hat{\boldsymbol g}_0=(1,0)$ and
  $\boldsymbol\theta_1=(1.9,-1)$; its sampled loss falls
  $0.500\rightarrow0.405$. Then account for all four changes
  $(-0.395,-0.095,+0.205,+0.605)$ to show
  $\Delta\mathcal L=+0.080$ and hence
  $7.125\rightarrow7.205$.
- **[CORE] Page 16, “A minibatch is the useful middle ground.”** For the first
  two examples, calculate $\hat{\boldsymbol g}_0=(2.5,-2)$ and identify $B=1$
  and $B=N$ as endpoints of the same family.

### 16–23 min · Pages 17–19 · compare batch sizes and connect to PyTorch

- **[CORE] Page 17, “Why minibatches help: averaging cancels some
  disagreement.”** Fix $\boldsymbol\theta_0$ before reading the arrows.
  One-example estimates fan out; averaging a few examples clusters the possible
  directions; averaging all examples leaves one exact full-data direction. Do
  not discuss rates, trajectories, wall-clock speed, or convergence here.
- **[SHOULD-COVER] Page 18, “Batch, stochastic, and minibatch
  GD—qualitatively.”** Summarize the trade-offs only after the fixed-parameter
  averaging intuition: update frequency, estimator noise, vectorization, and
  late motion depend on batch size, while actual throughput depends on hardware
  and implementation.
- **[CORE] Page 19, “Choosing the batch size in PyTorch.”** Trace
  `zero_grad → backward → step`. With gradient accumulation, a
  microbatch/iteration need not execute an optimizer step. An executed
  `optimizer.step()` counts as an optimizer update even when a zero gradient
  produces no numerical change; if the optimizer step itself is skipped, as on
  an AMP overflow, there is no optimizer update.

### 23–33 min · Pages 20–30 · surfaces → directions → estimator

- **[CORE] Pages 20–21, “How can one example estimate the full gradient?” and
  “From data rows to the full loss.”** Start with the concrete construction
  $(\boldsymbol x_i,y_i)\to\hat y_i\to\ell_i\to\mathcal L$. Establish the
  linearity bridge $\nabla\mathcal L=N^{-1}\sum_i\nabla\ell_i$ before using the
  word *estimator*.
- **[CORE] Pages 22–23, “Each example defines its own loss surface” and “Same
  parameters, different descent directions.”** Read the four computed troughs
  first, then reveal the four updates from the shared
  $\boldsymbol\theta_0$. The arrows are updates
  $\Delta\boldsymbol\theta_i=-\eta\nabla\ell_i$, not gradients pointing in the
  same direction.
- **[CORE] Pages 24–25, “A minibatch averages the selected loss surfaces” and
  “Average the one-example updates.”** Compare representative
  $B=1,2,4$ surfaces, then translate the four update vectors to a common origin.
  Their arithmetic mean is the full-batch update.
- **[CORE] Page 26, “An estimator maps random data to an estimate.”** Keep the
  first pass generic. For a discrete random variable $X$ with probability mass
  function $p$, write
  $\mathbb E[f(X)]=\sum_x p(x)f(x)$. Define $f(X)$ as an unbiased estimator of
  a target $a$ when $\mathbb E[f(X)]=a$. Students should identify the random
  input, the function, and the target before any gradient symbols appear.
- **[CORE] Page 27, “Why uniformly sampled SGD is unbiased.”** Now substitute
  $X=I_t$, $p(i)=1/N$, and
  $f(i)=\nabla_{\boldsymbol\theta}\ell_i(\boldsymbol\theta_t)$. Define
  $\hat{\boldsymbol g}_t=f(I_t)=
  \nabla_{\boldsymbol\theta}\ell_{I_t}(\boldsymbol\theta_t)$ *before* the
  proof, then condition on the fixed current parameters and sum to
  $\nabla\mathcal L(\boldsymbol\theta_t)$. Uniform sampling is part of the
  claim.
- **[CORE] Page 28, “Minibatches preserve the mean and reduce the spread.”** Read
  the row and row-pair labels before stating the result. Equal-weight averaging
  preserves the mean, while the possible estimates contract toward the same
  full-gradient centre. Keep the independent-draw $\boldsymbol\Sigma/B$ rule,
  exact subset-spread values, and finite-population correction in Notebook 00.
- **[CORE] Page 29, “Unbiased means correct on average—not downhill every
  time.”** Reuse the row-2 counterexample: $\ell_2$ falls
  $.500\to.405$ while the full loss rises $7.125\to7.205$. This separates a
  statement about repeated estimates at fixed parameters from a per-step
  guarantee.
- **[CORE] Page 30, “Notebook 00: reproduce the surfaces, arrows, and
  averages.”** Run the 90-second Notebook 00 checkpoint listed below; leave
  nonuniform sampling and exact covariance variants for follow-along.

### 33–41 min · Pages 31–37 · exact gradient and ravine geometry

- **[CORE] Page 31, “An exact gradient can still be slow.”** Close the sampling
  story: every update in Example B now uses the exact full-batch gradient. The
  remaining question is whether one scalar learning rate can choose a useful
  distance in every parameter direction.
- **[CORE] Page 32, “Example B: one feature is ten times larger.”** Read the
  feature scales $\pm1$ and $\pm10$ before doing algebra. Stay in the model's
  original parameter coordinates: the optimum is
  $\boldsymbol\theta^\star=(1,-.5)$ and the displayed start is
  $\boldsymbol\theta_0=(-1.4,.35)$.
- **[CORE] Page 33, “A ravine is a long, narrow valley with steep walls.”**
  Establish the physical picture before showing an equation: travel along the
  valley is gradual, while a small sideways move climbs a steep wall. This is
  the geometric problem that one global learning rate must handle.
- **[CORE] Page 34, “The $10\times$ feature turns the loss into a ravine.”**
  Read the exact three-dimensional loss surface first, then obtain
  $\mathcal L(\boldsymbol\theta)=
  \tfrac12(\theta_1-1)^2+50(\theta_2+.5)^2$ and
  $\nabla_{\boldsymbol\theta}\mathcal L=
  (\theta_1-1,100(\theta_2+.5))$. Define *curvature* before using it: it is how
  quickly the slope changes as a parameter moves. The feature factor $10$ is
  squared by the loss, so the $\theta_2$ direction is $100\times$ more
  sensitive. Notebook 02 reproduces this exact 3-D surface rather than a
  decorative surrogate.
- **[CORE] Page 35, “Contours show the same surface from above.”** Connect
  height on Page 34's surface to equal-loss contour lines, then name the
  geometry directly on the equal-aspect contour in parameter space. Moving
  horizontally in $\theta_1$ is the shallow direction; moving vertically in
  $\theta_2$ is the steep direction. “Along the valley” and “across the valley”
  are optional geometric descriptions, not additional coordinates students
  must track.
- **[CORE] Page 36, “One correct step can overshoot across the valley.”** Work
  $\hat{\boldsymbol g}_0=(-2.4,85)$,
  $\Delta\boldsymbol\theta_0=-.0195\hat{\boldsymbol g}_0=(.0468,-1.6575)$, and
  $\boldsymbol\theta_1=(-1.3532,-1.3075)$. The gradient points toward increasing
  loss and the update $-\eta\hat{\boldsymbol g}_0$ points toward local decrease; its
  second component carries $\theta_2$ past the optimum $-.5$. Call this one
  crossing *overshoot*, not yet oscillation.
- **[CORE] Page 37, “The trade-off is visible in the computed paths.”** Compare
  the actual full-batch paths at $\eta=.005$ and $.0195$ under the same start
  and update budget. The smaller rate avoids the first crossing but moves
  $\theta_1$ by only $.012$; the larger rate moves $\theta_1$ by $.0468$ while
  overshooting in the steep $\theta_2$ direction.
  One global rate creates the trade-off. Define
  *oscillation* here as repeated side-to-side crossings of the valley—not merely
  one overshoot. The evidence is computed on Example B, not a decorative path.

### 41–55 min · Pages 38–57 · noisy signal → EWMA → momentum

- **[CORE] Page 38, “How can we smooth a noisy sequence?”** Use the separator
  only to pose the new problem. Do not introduce gradient notation yet.
- **[CORE] Page 39, “Daily temperature readings are noisy.”** Read the seeded
  120-day temperature signal before introducing a formula. One day may be
  unusually warm or cool; the target is the slower changing trend. Define
  *memory* as one running average carried from one observation to the next—it
  does not store every reading. If time permits, open the
  [EWMA + momentum explorer](https://nipunbatra.github.io/dl-teaching/interactives/ewma-explorer.html):
  move the day control first, then compare $\beta=.5,.9,.98$ while keeping the
  observations fixed. Use the sudden-change control only after students can
  name the basic smoothness–lag trade-off.
- **[CORE] Page 40, “Short memory reacts quickly—and follows the noise.”** Before
  reading the curve, define $\beta$ as the fraction of the *previous summary*
  retained. With $\beta=.5$, old summary and new observation receive equal
  weight, so the estimate reacts quickly but remains noisy.
- **[CORE] Page 41, “More memory gives a useful compromise.”** Change only
  $\beta$ to $.9$: retain $90\%$ of the previous summary and add $10\%$ of the
  new reading. Ask students to identify both the reduced noise and the lag when
  the trend turns.
- **[CORE] Page 42, “Too much memory becomes slow to respond.”** At
  $\beta=.98$, the curve is smoother but visibly lags. Larger $\beta$ means
  longer memory, not automatically a better estimate.
- **[CORE] Page 43, “EWMA keeps one running memory $m_t$.”** Define $m_t$
  explicitly as the running memory—the smoothed value after processing
  observation $x_t$. Only after the three curves write
  $m_t=\beta m_{t-1}+(1-\beta)x_t$. Work
  $.9(30)+.1(36)=30.6$, then unroll just enough to see weights
  $(1-\beta),(1-\beta)\beta,(1-\beta)\beta^2,\ldots$ and earn the name
  exponentially weighted moving average (EWMA).
- **[CORE] Page 44, “EWMA weights come from repeated substitution.”** Write the
  recurrence at $t$ and $t-1$, then substitute the second expression into the
  first. Point to the resulting weights
  $(1-\beta)$, $(1-\beta)\beta$, and $(1-\beta)\beta^2$ before stating the
  general weight $(1-\beta)\beta^k$ on $x_{t-k}$. For $\beta=.9$, read the
  concrete sequence `.100, .090, .081, .0729, ...`: each additional step into
  the past multiplies the weight by $.9$. This geometric decay earns the word
  *exponential*.
- **[CORE] Page 45, “Starting from zero makes the first temperature estimate too
  small.”** Keep the exact same fixed-seed temperature readings and $\beta=.9$;
  change only the initialization. The earlier smoothing curves used $m_1=x_1$.
  Now set $m_0=0$ and work the actual first reading:
  $x_1=17.968^\circ\mathrm C$ gives
  $m_1=.9(0)+.1(17.968)=1.7968^\circ\mathrm C\approx1.80^\circ\mathrm C$. Say explicitly that this is not a
  plausible average of the readings: only $.1$ observation weight has arrived,
  while $.9$ remains on the artificial initial zero.
- **[CORE] Page 46, “Divide by the weight that has arrived.”** Sum the geometric
  weights, $(1-\beta)\sum_{k=0}^{t-1}\beta^k=1-\beta^t$, then obtain
  $\widehat m_1=1.7968/(1-.9)=17.968^\circ\mathrm C\approx17.97^\circ\mathrm C$. Call this *missing-mass
  correction*: it removes initialization shrinkage, not observation noise.
- **[CORE] Page 47, “Momentum: complete update rule.”** Replace the scalar
  observation $x_t$ by the measured gradient $\boldsymbol g_t$ and return to
  the optimizer clock. Write the full EWMA-form rule before substituting:
  $\boldsymbol m_{t+1}=\beta\boldsymbol m_t+
  (1-\beta)\boldsymbol g_t$,
  $\boldsymbol\theta_{t+1}=\boldsymbol\theta_t-\eta\boldsymbol m_{t+1}$, and
  $\boldsymbol m_0=\boldsymbol0$. Use the same conceptual rate as plain GD,
  $\eta=.0195$, and choose $\beta=.25$. Then substitute
  $\boldsymbol g_0=(-2.4,85)$ to obtain
  $\boldsymbol m_1=.75\boldsymbol g_0=(-1.8,63.75)$ and
  $-\eta\boldsymbol m_1=(.0351,-1.243125)$. With an empty memory and no bias
  correction, the first momentum move is therefore $75\%$ of the GD move at the
  same $\eta$; this is a consequence to explain, not a difference to hide by
  rescaling the rate. Adam later corrects two zero-start memories that are used
  explicitly as moment estimates.
- **[CORE] Pages 48–49, “First update: the same gradient, two update rules” and
  “Updates 2 and 3: repeat the same three equations.”** Read the two update
  rules before entering the table, then compare each row
  horizontally. Both use $\eta=.0195$; momentum additionally uses $\beta=.25$.
  Plain GD reaches $(-1.3532,-1.3075)$ with loss `35.371588`, while momentum
  reaches $(-1.3649,-.893125)$ with loss `10.523739`. After update 2, plain GD is
  at $(-1.307313,.267125)$ with loss `32.085884`, while momentum is at
  $(-1.321538,-.628961)$ with loss `3.526316`. After update 3 the respective
  points are $(-1.262320,-1.228769)$ and $(-1.276745,-.374315)$ with losses
  `29.114240` and `3.381626`. Both loss sequences decrease at every row; the
  first points differ because ordinary momentum uses its uncorrected zero-start
  state.
- **[CORE] Page 50, “Updates 4–6: the same learning rate, different moves.”** Each
  method now evaluates a different current gradient because its parameter path
  has already diverged. Read horizontally rather than turning the table into a
  second derivation: plain GD's losses decrease
  `26.426279 → 23.994575 → 21.794368`, while momentum's decrease
  `2.493000 → 2.447849 → 2.303186`. Momentum is already lower at update 1
  and remains lower through the displayed 55-update run. The comparison isolates
  faster progress while preserving strict descent for both methods.
- **[CORE] Page 51, “The same learning rate produces very different paths.”**
  Read the computed contour only in the model's original
  $(\theta_1,\theta_2)$ coordinates. The paths share only the starting point:
  with the same $\eta$, zero-start momentum makes a smaller first move and then
  carries its running memory. Use the numbered arrows to connect the rows on
  Pages 48–50 to actual parameter movement—this is not a gradient-space plot.
- **[CORE] Page 52, “Momentum update 2 in $\theta_1$: both pushes point
  right.”** At the
  momentum point $\boldsymbol\theta_1=(-1.3649,-.893125)$, isolate the first
  coordinate: $m_{1,1}=-1.8$ and $g_{1,1}=-2.3649$. Convert the two terms of
  $m_{2,1}=.25m_{1,1}+.75g_{1,1}$ into parameter moves. Stored history contributes
  `+.0088`; the new gradient contributes `+.0346`; together they give `+.0434`
  and hence $\theta_{2,1}=-1.321538$. The two pushes agree, so the memory keeps a
  persistent rightward direction.
- **[CORE] Page 53, “Update 2 in $\theta_2$: the old and new pushes nearly
  cancel.”** Now isolate $m_{1,2}=63.75$ and $g_{1,2}=-39.3125$. Stored history
  moves `-.3108`, while the reversed current gradient moves `+.5750`; the net is
  only `+.2642`, giving $\theta_{2,2}=-.628961$. This move is about six times
  smaller than plain GD's update-2 move in the steep coordinate. Combine the two
  derived coordinates to recover
  momentum's $\boldsymbol\theta_2=(-1.321538,-.628961)$, compare with the
  Page 49 update-2 row, and reconnect both methods to Page 51's contour.
- **[CORE] Page 54, “Momentum damps the steep-direction oscillation.”** Read
  the computed paths under the same start, same $\eta=.0195$, and 55 exact
  gradient evaluations. With displacements below $10^{-6}$ treated as numerical
  jitter, plain GD makes 55 visible crossings and momentum makes 12. Momentum
  also reduces across-valley travel from `31.1763` to `1.9751` while retaining
  steady motion in the shallow $\theta_1$ direction. Treat these as mechanism
  clues; the full loss on Page 55 is the outcome that matters.
- **[CORE] Page 55, “Both losses fall; momentum remains lower.”** Both methods
  use exact full-batch gradients and the same $\eta=.0195$. Both full-loss curves
  decrease strictly on every displayed update, and momentum stays lower from
  update 1. On the identical 55-update budget, plain GD ends at approximately
  `.458144`, whereas momentum ends at `.329775`. This is controlled evidence on
  Example B, not a universal guarantee.
- **[SHOULD-COVER] Page 56, “Momentum is GD plus one running memory.”** Read the
  two pure-Python functions side by side. Both functions name their step-size
  argument `eta`, and this comparison passes the same value `.0195` to both.
  Plain GD returns only the next parameters. Momentum
  first blends the old memory with the current gradient, then returns both the
  next parameters and the next memory. Initialize
  `memory = [0.0] * len(theta)` once and reuse the returned list; do not reset it
  inside the training loop. The gradient itself is still recomputed at every
  update. Keep the order explicit:
  $\boldsymbol g_t\to
  \boldsymbol m_{t+1}=\beta\boldsymbol m_t+(1-\beta)
  \boldsymbol g_t\to
  \boldsymbol\theta_{t+1}=\boldsymbol\theta_t-\eta\boldsymbol m_{t+1}$.
- **[CORE] Page 57, “PyTorch momentum uses the same training loop.”** Show
  `SGD(lr=.014625, momentum=.25)` only after the conceptual recurrence is secure.
  Define PyTorch's stored `momentum_buffer` as
  $\widetilde{\boldsymbol m}_{t+1}=\beta
  \widetilde{\boldsymbol m}_t+\boldsymbol g_t$. Then
  $\boldsymbol m=(1-\beta)\widetilde{\boldsymbol m}$ and
  $\eta_{\rm PT}=\eta(1-\beta)=.014625$. Present this as a library-convention
  mapping, not as the conceptual comparison's motivation. Use Notebook 02 to
  compare $\beta=.25,.5,.9$, reproduce the first six table rows, and inspect the
  actual `momentum_buffer`.

### 55–59 min · Pages 58–62 · what standardization fixes—and what remains

- **[CORE] Page 58, “Could feature standardization fix this example? Yes.”**
  Ask the objection only after momentum has solved the raw-feature ravine. In
  this deliberately scale-only example, $z_2=x_2/10$ changes curvature from
  $(1,100)$ to $(1,1)$; plain GD no longer needs momentum for this failure.
- **[CORE] Pages 59–62, exact four-row correlated control.** On “Four rows:
  standardized, but still correlated,” display
  $\mathbf X=[(-1,-7/5),(-1,-1/5),(1,1/5),(1,7/5)]$, use
  $\boldsymbol\theta^\star=(1,0)$ and $y=x_1$, and verify that both columns
  have mean zero and variance one while their correlation is $4/5=.8$. On
  “Equal-sized moves change this loss by a factor of nine,” calculate
  $\mathbf H=\mathbf X^\top\mathbf X/4=
  \begin{bmatrix}1&4/5\\4/5&1\end{bmatrix}$ and compare
  $\mathcal L(a,a)=\tfrac95a^2$ with
  $\mathcal L(a,-a)=\tfrac15a^2$. Then use “The $9\times$ curvature difference
  produces a tilted ravine” to read eigenvalues $.2$ and $1.8$, condition
  number $9$, and contour-axis ratio $\sqrt9=3$. Finally, on “The same four
  rows confirm that momentum can help,” use the same conceptual rate $\eta=1$
  for GD and normalized-EWMA momentum, with $\beta=.25$. Compare both methods
  over 18 exact updates; their first displacements intentionally differ because
  momentum starts from empty memory. Their final losses are approximately
  `1.62259e-4` and `8.35231e-6`,
  respectively; both decrease strictly. The claim is narrow: standardization
  fixes units, not correlation, minibatch fluctuation, or every changing
  deep-network direction.

### 59–67 min · Pages 63–75 · build RMSProp from the remaining scale problem

- **[CORE] Pages 63–66, separator through “Recent magnitude should control each
  component's step.”** Return to the original unscaled Example B solely to
  isolate the mechanism. On Page 64, retrieve the four rows and recompute
  $\hat{\boldsymbol g}_0=(-2.4,85)$ from the displayed loss; derive the
  $35.4\times$ component ratio and the corresponding update magnitudes at
  $\eta=.0195$. Page 65 makes explicit that multiplying both components by one
  learning rate cannot change this ratio. Then read the first five signed
  gradients on Page 66: alternating signs can cancel in an ordinary average,
  while squares retain evidence that a coordinate has been large.
- **[CORE] Pages 67–68, “RMSProp uses a recent root-mean-square scale” and
  “RMSProp: complete update rule.”** Earn each word in the name:
  square, exponentially weighted mean, root, then propagate the scaled update.
  Initialize $\boldsymbol v_0=\boldsymbol0$ and keep the operations elementwise:
  $\boldsymbol v_{t+1}=\beta_2\boldsymbol v_t+(1-\beta_2)\hat{\boldsymbol g}_t^2$,
  ruler $\boldsymbol r_{t+1}=\sqrt{\boldsymbol v_{t+1}}+\epsilon_{\rm opt}$,
  then divide by $\boldsymbol r_{t+1}$. Name $\boldsymbol v$ as the
  squared-gradient memory and $\boldsymbol r$ as the corresponding RMS scale;
  defer the numerical meaning of the division to the scalar calculation next.
- **[CORE] Pages 69–70, the scalar loss bridge.** Use
  $\mathcal L(\theta)=\tfrac12\theta^2$, $\theta_0=2$, and $g_0=2$ to work one
  complete update for plain GD, EWMA momentum, and RMSProp. Keep the loss,
  starting point, current gradient, and $\eta=.10$ fixed. Read every
  intermediate state—$m_1$, $v_1$, $r_1$, and $g_0/r_1$—before comparing the
  new parameters and losses. This is a formula check, not an optimizer ranking.
- **[CORE] Pages 71–72, the “First update” stages.** Build
  $\boldsymbol v_1=(.576,722.5)$ and rulers $(.759,26.879)$ before dividing.
  With $\eta=.08$, obtain the rescaled gradient $(-3.162,3.162)$ and
  $\boldsymbol\theta_1=(-1.147,.097)$. Stress that equal first-step magnitudes
  are a cold-start consequence of $\boldsymbol v_0=\boldsymbol0$, not RMSProp's general
  behavior; standard PyTorch RMSProp does not bias-correct `square_avg`.
- **[CORE] Page 73, “Second update: the history now matters.”** Recompute
  $\hat{\boldsymbol g}_1=(-2.147,59.702)$, then obtain
  $\boldsymbol v_2=(.979,1006.680)$, rulers $(.990,31.728)$, and unequal
  rescaled components $(-2.170,1.882)$. This is the first displayed update in
  which the scale contains two gradients.
- **[SHOULD-COVER] Pages 74–75.** Read the vectorized GD/RMSProp functions side
  by side. In PyTorch, `square_avg` stores $\boldsymbol v$ and the denominator is
  `square_avg.sqrt() + eps`; PyTorch's argument `alpha` is the same retention
  coefficient denoted $\beta_2$ in the lecture. Judge the method only on the computed path after
  understanding the carried state; the method-specific rate is mechanism
  evidence, not a leaderboard.

### 67–73 min · Pages 76–85 · combine the two memories in Adam

- **[CORE] Pages 76–78, separator through “What does ‘moment’ mean here?”**
  Ordinary momentum retains signed gradient history in a normalized EWMA;
  RMSProp retains a recent mean square. Define the first raw moment
  $\mathbb E[G]$ and second raw moment $\mathbb E[G^2]$ before saying that Adam
  means *adaptive moment estimation*. In Adam, rename standalone momentum's
  $\beta$ to $\beta_1$ and keep RMSProp's $\beta_2$. Adam combines a signed EWMA with a
  squared-gradient EWMA, then explicitly corrects both for their missing
  zero-start mass. Do not call the second raw moment a variance.
- **[CORE] Pages 79–80, “Adam normalizes both zero-start memories” and “Adam:
  complete update rule.”** Retrieve the temperature calculation before
  writing Adam's correction. The temperature clock counted observations
  $1,\ldots,t$, hence mass $1-\beta^t$; optimizer measurement
  $\hat{\boldsymbol g}_t$ creates state $t+1$, hence mass
  $1-\beta^{t+1}$. Initialize
  $\boldsymbol m_0=\boldsymbol v_0=\boldsymbol0$, then divide by
  $1-\beta_1^{t+1}$ and $1-\beta_2^{t+1}$. At the first update,
  $\boldsymbol m_1=.1\boldsymbol g_0\to
  \widehat{\boldsymbol m}_1=\boldsymbol g_0$ and
  $\boldsymbol v_1=.001\boldsymbol g_0^2\to
  \widehat{\boldsymbol v}_1=\boldsymbol g_0^2$. Adam corrects both because its
  two memories use different $\beta$ values before they are combined. Ordinary
  momentum above still uses its uncorrected state. This is initialization
  correction—not correction of gradient-sampling noise or predictive-model
  bias. Read the PyTorch tuple positionally: `betas=(.9, .999)` means
  $(\beta_1,\beta_2)$, so the first value retains signed-gradient memory
  $\boldsymbol m$ and the second retains squared-gradient memory
  $\boldsymbol v$.
- **[CORE] Pages 81–82, the two worked Adam updates.** On update 1, obtain
  $\boldsymbol m_1=(-.24,8.5)$, $\boldsymbol v_1=(.00576,7.225)$ and the
  corrected sign-scaled move $(.10,-.10)$. On update 2, both memories contain
  history and the move is approximately $(.0999,-.0995)$.
- **[SHOULD-COVER] Pages 83–85.** Trace the vectorized function with `step=1`
  on its first call. Map `exp_avg` to $\boldsymbol m$ and `exp_avg_sq` to
  $\boldsymbol v$. Use the four-path figure only to explain stored-state
  mechanisms under method-specific rates, then use Notebook 03 to check every
  parameter and state history against PyTorch to numerical precision.

### 73–80 min · Pages 86–93 · why and how the learning rate changes

Budget seven minutes in the canonical class and eight to nine minutes in an
extended delivery; spend any extra time on the six-shape comparison and the
Notebook 03 scheduler trace, not by dropping the verbal close.

- **[CORE] Pages 86–87, separator and “One multiplier, every optimizer.”** Write
  $\boldsymbol\theta_{t+1}=\boldsymbol\theta_t-\eta_t\boldsymbol d_t$ once.
  The direction $\boldsymbol d_t$ may come from GD, momentum, RMSProp, or Adam;
  the schedule changes the multiplier, so scheduling is not restricted to SGD.
- **[CORE] Page 88, “Why decay is especially useful with stochastic
  gradients.”** At the full-batch optimum the exact gradient is zero, but the
  individual example gradients need not be. Full-batch GD therefore has no
  persistent sampling noise at that point; a constant rate may be adequate
  when it is stable. A stochastic update can keep moving, so its late step scale
  matters directly.
- **[CORE] Page 89, “A fixed learning rate forces a choice.”** Compare only the
  two fixed-rate runs first. With the same seed-12 sampled-index stream, the
  $.10$ run first crosses a full-loss gap of $.5$ after 8 updates, whereas the
  $.02$ run takes 56. Say *first crossing*: neither curve is promised to remain
  below the marker.
- **[CORE] Pages 90–91, introduce decay and then inspect its late behaviour.**
  Work $\eta_t=.10/(1+.02t)$ at $t=0,50,350$. The schedule begins with the same
  $.10$ multiplier, so it preserves the 8-update first crossing. Then compare
  only fixed $.10$ with decay over the final 99 transitions: `.520` versus
  `3.794`, or about `14%` as much motion. Pair that shorter path with the lower
  mean late full-loss gap, `.000201` versus `.004100`; a short path alone would
  not establish useful refinement. The sampled-index stream and update
  equations are fixed by design. The learning-rate policy is the intervention;
  later gradients and optimizer states differ because the parameter paths
  differ.
  On Page 91, use the
  [learning-rate schedule explorer](https://nipunbatra.github.io/dl-teaching/interactives/lr-schedule-explorer.html)
  in two passes: **Travel early** compares the two fixed rates, then **Settle
  late** compares fixed $.10$ with decay. Both views reuse Example A and the
  same seed-12 sampled-index stream.
- **[SHOULD-COVER] Page 92, “Six common schedule shapes.”** Name inverse-time,
  exponential, linear, milestone, cosine, and warmup-plus-cosine as families,
  not as a universal ranking or a requirement to use all six.
- **[CORE] Page 93, “PyTorch: state the clock explicitly.”** In this deck
  $t$ counts executed optimizer updates. Each executed `opt.step()` uses
  $\eta_t$ and advances the clock even when a zero gradient causes no numerical
  parameter change; the following `sched.step()` prepares $\eta_{t+1}$.
  Accumulated microbatches do not advance this clock. If `opt.step()` is skipped,
  for example after an AMP overflow, skip `sched.step()` too.
- **[CORE, 20-second verbal close]** Even when the coda is not projected, end
  the live class by retrieving the six decisions aloud: batch size controls
  noise; momentum remembers direction; RMSProp remembers scale; Adam remembers
  both; a schedule controls time; and a parameterization controls feasible
  values.

### Post-80 follow-along coda · Pages 94–110 · constraints and synthesis

- **[OPTIONAL] Pages 94–100, why the constraints exist.** Page 94 poses the
  question and Page 95 shows an unconstrained proposal leaving the feasible set.
  Page 96 motivates a strictly positive Gaussian noise scale for both a shared
  homoskedastic model and an input-dependent heteroskedastic model. On Page 97,
  work the actual Gaussian-NLL failure: residual $.05$, $\sigma_0=.10$,
  gradient $7.5$, and $\eta=.02$ produce the invalid proposal
  $\widetilde\sigma_1=-.05$. Pages 98–100 motivate, respectively, a blend
  coefficient in a box, a Bernoulli probability, and coupled simplex weights.
- **[OPTIONAL] Pages 101–108, four reparameterizations and a projection coda.**
  Page 101 gives the common raw-parameter → transform → valid-model pattern.
  Pages 102–105 then apply it without skipping a family: learn
  $a=\log\sigma$ and use $\sigma=\exp(a)$; map a raw value into a box with an
  affine sigmoid; train Bernoulli logits with a logits-based loss; and map one
  logit per component through softmax. Pages 106–107 reserve projection for
  simple closed constraints and show the PyTorch `clamp_` pattern, including the
  fact that momentum/Adam state is not projected. Page 108 makes the
  reparameterization-versus-projection choice explicit. Treat `softplus` only as
  an alternative positive map, not the primary convention.
- **[OPTIONAL] Pages 109–110, retrieval and synthesis.** Page 109 is only the
  final separator. Page 110 is the written version of the 20-second verbal
  retrieval used to close the core class.

## 110-minute extended route through the same 110-page handout

The extended route keeps the canonical order and spends the extra time on the
worked states and notebook checks.

### 0–41 min · Pages 2–30 · gradient source, clocks, and estimators

Use the canonical route, but calculate every Example A row, derive
$\mathbb E[f(X)]=\sum_x p(x)f(x)$ before substituting the SGD estimator, and
run Notebook 00 for three minutes. Keep the uniform-sampling assumption explicit.

### 41–68 min · Pages 31–57 · ravine, EWMA, and momentum

Derive the exact Example B loss from the four rows, connect its 3-D surface to
the contour view, then use the same temperature observations to contrast
$m_1=x_1$ with raw zero start $m_0=0$ and correction by $1-\beta^t$. Map that
empty scalar memory to uncorrected optimizer momentum before working the first
six GD/momentum updates. Preserve $\boldsymbol m_0=\boldsymbol0$, use the same
conceptual $\eta=.0195$ for both methods with $\beta=.25$, and make the smaller
first momentum move explicit before the coordinate-by-coordinate
reinforcement/cancellation explanation. Finish with the 55-update computed
evidence and the brief PyTorch `momentum_buffer` mapping.

### 68–74 min · Pages 58–62 · standardization counterfactual and correlated control

First show that $z_2=x_2/10$ removes the scale-only $100{:}1$ curvature.
Then switch to the exact four-row control: both new feature variances are one,
correlation is $.8$, and
$\mathbf H=\begin{bmatrix}1&.8\\.8&1\end{bmatrix}$ has eigenvalues $.2$ and
$1.8$, condition number $9$, and contour-axis ratio $3$. Compare 18 exact
updates of GD and EWMA-form momentum at the same conceptual $\eta=1$, with
$\beta=.25$ for momentum. Then require students to distinguish *standardized*
from *uncorrelated*.

### 74–88 min · Pages 63–75 · RMSProp from problem to evidence

Take Pages 63–75 in order: the Example B data-and-gradient recap, magnitude
mismatch, why signed averaging is
insufficient, square → mean → root, the scalar $\tfrac12\theta^2$ formula
check, $\boldsymbol v_0=\boldsymbol0$, first
scale build, first rescaled step, second update with history, vectorized
implementation, and the computed ravine. In Notebook 03 inspect `square_avg`
and verify the complete parameter/state histories to numerical precision.
Explicitly say that `square_avg` stores $\boldsymbol v$, while the ruler is
$\sqrt{\boldsymbol v}+\epsilon_{\rm opt}$.

### 88–99 min · Pages 76–85 · Adam's two raw moments

Define first and second raw moments before the optimizer name. Derive the two
zero-start corrections by retrieving the temperature mass $1-\beta^t$, then
changing clocks to optimizer state $t+1$ and mass $1-\beta^{t+1}$. Contrast
Adam's two explicit corrections with ordinary momentum's uncorrected dynamics,
then work both displayed updates. Map `exp_avg` and `exp_avg_sq` to the
uncorrected memories and verify the full vectorized recurrence against PyTorch
to numerical precision. Use the four-path panel to explain mechanisms, not to
rank optimizers universally.

### 99–107 min · Pages 86–93 · scheduling

Take the eight-page sequence in order. First separate the optimizer direction
$\boldsymbol d_t$ from its multiplier $\eta_t$ and state that schedules can wrap
GD, momentum, RMSProp, or Adam. Contrast the exact full-batch optimum with the
nonzero per-example gradients that keep stochastic updates moving. Predict the
early/late trade-off, derive $\eta_t=\eta_0/(1+dt)$, read both the early-threshold
and last-99-transition comparisons, name the six schedule families without
ranking them, and trace `opt.step(); sched.step()`. Here the displayed clock is
executed optimizer updates: gradient accumulation does not advance it, while an
executed `opt.step()` does even if a zero gradient leaves the parameter values
unchanged. If AMP overflow or another guard skips `opt.step()`, skip
`sched.step()` as well.

### 107–110 min · Pages 94–108 · four constraints, one reusable pattern

Define $\mathcal C$ on Page 95, then name the model reason before each formula:
Gaussian likelihood requires $\sigma>0$; an interpolation coefficient must stay
in its box; a Bernoulli parameter must be a probability; and mixture/class
weights must be nonnegative and sum to one. Work Page 97's invalid direct-scale
step, then use Pages 101–105 to state the reusable repair: optimize an
unrestricted raw value and reconstruct the valid model parameter inside every
forward pass. Live-derive $a=\log\sigma$, $\sigma=\exp(a)$; assign the affine
sigmoid, logits, and softmax calculations as immediate follow-along if time is
tight. Close with Pages 106–108: projection is a simple alternative for closed
boxes or nonnegative sets, but not the default for strict Gaussian scales or a
simplex, and projected parameters do not reset stored optimizer state.

### Post-110 follow-along coda · Pages 109–110 · synthesis

Read the final six-part synthesis as the written counterpart of the canonical
route's 20-second verbal retrieval.

## Current numeric control ledger

Use these values when narrating the candidate deck; do not substitute a nearby
rate from an older offering or from another notebook panel.

1. **Example A · Pages 8–17:** data
   $(-1,-1),(0,1),(1,3),(2,6)$;
   $\boldsymbol\theta_0=(2,-1)$; full-step
   $\eta=.10$; sampled point $(0,1)$. The Page 17 batch-size picture compares
   possible estimators at one fixed $\boldsymbol\theta_0$; it is not an
   optimizer-path or throughput comparison.
2. **Estimator geometry and spread · Pages 21–28:** every surface, direction,
   and estimate is evaluated at the fixed $\boldsymbol\theta_0$. The deck keeps
   the comparison qualitative; Notebook 00 verifies that the uniform-subset
   mean squared distances are
   $34.4375,11.4791667,3.8263889,0$ for $B=1,2,3,4$. Independent
   with-replacement draws instead give $34.4375/B$, so $B=N$ need not be exact
   under that sampler.
3. **Example B geometry and preprocessing · Pages 32–37 and 58–62:**
   $\boldsymbol\theta^\star=(1,-.5)$,
   $\boldsymbol\theta_0=(-1.4,.35)$,
   $\mathcal L(\boldsymbol\theta)=
   \tfrac12(\theta_1-1)^2+50(\theta_2+.5)^2$, initial loss
   $\mathcal L(\boldsymbol\theta_0)=39.005$, and
   $\mathbf X^\top\mathbf X/N=\operatorname{diag}(1,100)$. The GD prediction
   controls shown in the deck are $\eta=.005,.0195$; Page 58 later standardizes
   with $z_2=x_2/10$, producing curvature $(1,1)$ in the rescaled coordinates.
   Pages 59–62 then switch to the exact four-row standardized-but-correlated
   control
   $\mathbf X=[(-1,-7/5),(-1,-1/5),(1,1/5),(1,7/5)]$ with
   $\boldsymbol\theta^\star=(1,0)$ and $y=x_1$. Both feature variances are one,
   correlation is $.8$, and
   $\mathbf H=\begin{bmatrix}1&.8\\.8&1\end{bmatrix}$ has eigenvalues $.2$ and
   $1.8$, condition number $9$, and contour-axis ratio $3$. With the same
   conceptual $\eta=1$ for GD and EWMA-form momentum, $\beta=.25$ for momentum,
   and an 18-update budget, final losses are approximately `1.62259e-4` for GD
   and `8.35231e-6` for momentum; both curves decrease strictly.
4. **EWMA and momentum · Pages 38–57:** Page 39 introduces the fixed noisy
   signal; Pages 40–42 compare $\beta=.5,.9,.98$; Page 43 introduces the EWMA
   recurrence; Page 44 derives its weights by repeated substitution. Pages
   45–46 then reuse the first 12 readings at $\beta=.9$: the first reading is
   $17.96810897$, raw zero-start state is $1.796810897$, observed weight mass is
   $.1$, and the corrected value returns to $17.96810897$. The following
   Page 47 transfers the idea of empty memory to momentum. Ordinary momentum
   uses the same normalized EWMA form
   $\boldsymbol m_{t+1}=.25\boldsymbol m_t+.75\boldsymbol g_t$ with
   $\boldsymbol m_0=\boldsymbol0$. Both methods use the same conceptual
   $\eta=.0195$. Consequently, the first momentum state is
   $\boldsymbol m_1=.75\boldsymbol g_0=(-1.8,63.75)$ and the first momentum move
   is $75\%$ of the plain-GD move; no matched-step rescaling is used.
   Pages 48–49 compare the first three plain-GD and momentum updates numerically;
   Page 50 extends the table through update 6, with plain-GD losses
   `26.426279, 23.994575, 21.794368` and momentum losses
   `2.493000, 2.447849, 2.303186`; Page 51 plots points 0–6 on the exact contour;
   Page 52 shows reinforcement in $\theta_1$; and Page 53 shows cancellation in
   $\theta_2$. EWMA-form momentum uses $\beta=.25$ and $\eta=.0195$ for 55
   updates; the coupled sequence gives
   $\boldsymbol m_1=(-1.8,63.75)$ and
   $\boldsymbol m_2=(-2.223675,-13.546875)$ after remeasuring at
   $\boldsymbol\theta_1=(-1.3649,-.893125)$. The second gradient components nearly
   cancel rather than cancelling exactly. The third momentum state is
   $\boldsymbol m_3=(-2.297073,-13.058789)$ and reaches
   $\boldsymbol\theta_3=(-1.276745,-.374315)$. Both loss sequences are strictly
   decreasing; momentum is lower from update 1. Across 55 updates, the path
   figure shows 55 versus 12 visible/material crossings and across-valley travel
   `31.1763` versus `1.9751` for GD and momentum, respectively. Page 55 reports
   final losses `.458144` for GD and `.329775` for momentum.
   Notebook 02 checks the EWMA recurrence, the PyTorch buffer/rate mapping,
   parameter path, and stored `momentum_buffer`.
5. **RMSProp and Adam · Pages 63–85:** Page 63 deliberately returns to the
   original unscaled ravine to isolate the optimizer mechanism; it is not a
   preprocessing recommendation. Page 64 retrieves the four Example B rows,
   recomputes $\boldsymbol g_0=(-2.4,85)$, and derives the $35.4\times$
   coordinate-scale mismatch; Page 65 visualizes why one global rate preserves
   that mismatch. Pages 67–68 define RMSProp, Pages 69–70 compare one scalar
   GD, momentum, and RMSProp update without skipping intermediate quantities,
   Pages 71–73 work the first two two-coordinate updates from
   $\boldsymbol v_0=\boldsymbol0$, and Page 74 maps the
   state to vectorized code and PyTorch `square_avg`. RMSProp uses
   $\eta=.08,\beta_2=.9,\epsilon_{\mathrm{opt}}=10^{-8}$; Page 75 shows 40
   updates. Pages 76–83 build Adam's two raw moments, retrieve the same
   missing-mass argument with the optimizer's $t+1$ state clock, apply the two
   bias corrections, and then show two worked updates and vectorized
   implementation. The four-method Page 84
   control uses 55 updates: GD $.0195$; EWMA-form momentum
   $.0195,\beta=.25$ (PyTorch `lr=.014625`); RMSProp $.08,\beta_2=.9$; Adam
   $.10,\beta_1=.9,\beta_2=.999$. Final losses are
   `.458144, .329775, 1.7248e-08, .062601` in method order GD, momentum,
   RMSProp, Adam, giving the order
   RMSProp $<$ Adam $<$ momentum $<$ GD. Notebook 03 uses these same four
   controls. These
   method-specific rates support a mechanism comparison, not a universal rank.
6. **Schedule · Pages 86–93:** Page 87 writes every optimizer as a direction
   $\boldsymbol d_t$ multiplied by $\eta_t$; Page 88 distinguishes an exact
   full-batch optimum from persistent stochastic motion. Page 89 uses seed 12
   and the identical stream of 360 sampled indices to isolate the fixed-rate
   trade-off: the $.10$ and $.02$ runs first cross a full-loss gap of $.5$ after
   `8` and `56` updates. Page 90 introduces
   $\eta_t=.10/(1+.02t)$ and evaluates it at $t=0,50,350$. Page 91 then isolates
   fixed $.10$ versus decay. Both first cross after 8 updates, but their
   final-100 mean gaps are `.00409961145` and `.000201402373`, and their
   final-99-transition path lengths are `3.79448` and `.520387`. The rate ends
   at `.0122249`; fixed summed late motion is `7.29165×` the decayed motion.
   The sampled indices and update equations are shared, but later gradients and
   stored states differ because the parameter paths differ. Page 92 compares six schedule shapes without
   ranking them. Page 93 and Notebook 03 advance $t$ once per executed
   `optimizer.step()`, including a zero-gradient step with no numerical parameter
   change; when the optimizer step is skipped, the scheduler step is skipped too.
7. **Constraints · Pages 94–108:** Page 95 shows an unconstrained proposal
   leaving the feasible set. Page 96 motivates $\sigma>0$ from homoskedastic and
   heteroskedastic Gaussian likelihoods. Page 97 uses residual `.05`,
   $\sigma_0=.10$, gradient `7.5`, and $\eta=.02$, giving the invalid direct
   proposal $\widetilde\sigma_1=-.05$. Pages 98–100 motivate a blend coefficient
   in $[0,1]$, a Bernoulli probability, and simplex-valued class or mixture
   weights. Page 101 states the shared raw-coordinate pattern. Page 102 makes
   the primary positive convention explicit: learn $a=\log\sigma$ and rebuild
   $\sigma=\exp(a)$ in the forward pass; `softplus` is only an alternative.
   Pages 103–105 use an affine sigmoid for a box, logits with
   `binary_cross_entropy_with_logits` for Bernoulli data, and `softmax`/
   `log_softmax` for simplex weights. Pages 106–107 give the ordinary proposal,
   simple nonnegative/box projection, PyTorch `clamp_`, and the caveat that
   projection leaves optimizer state untouched. Page 108 chooses between a
   smooth interior and an exact closed boundary. Notebook 03 mirrors all four
   cases: its Gaussian fit learns `mu=.0750`, `sigma=.7980`; its box fit recovers
   `c=.7000`; its Bernoulli fit learns logit `1.0986` and probability `.7500`;
   and its simplex fit returns approximately `[.2,.6,.2]` with sum one. In the
   35-step box-boundary comparison at $\eta=.40$, projection reaches `1.0` while
   the sigmoid parameterization remains at `.781698`. Pages 109–110 are the
   final separator and six-part synthesis.

## Exact notebook checkpoints

### Notebook 00 · `00_full_batch_sgd_minibatch.ipynb`

Colab: [`notebooks/L05/00_full_batch_sgd_minibatch.ipynb`](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L05/00_full_batch_sgd_minibatch.ipynb)

Use at Page 30 for 90 seconds in the canonical route or three minutes in the
extended route.

1. **“Start with the full-data loss and gradient.”** Expect predictions
   `[3, 2, 1, 0]`, residuals
   `[4, 1, -2, -6]`, full gradient
   $\hat{\boldsymbol g}_0=$ `[-0.75, -4.5]`, new parameters
   $\boldsymbol\theta_1=$ `[2.075, -0.55]`, and loss
   `7.1250 → 5.2153`.
2. **“Each example defines its own loss surface” and “One sampled SGD
   update.”** Inspect all four exact $\ell_i$ troughs at the shared
   $\boldsymbol\theta_0$. For the displayed draw $I_0=2$, verify
   $\widehat y_2=2$, $\hat{\boldsymbol g}_0=(1,0)$, and
   $\boldsymbol\theta_1=(1.9,-1)$.
3. **“The stochastic gradient is an estimator.”** Average the four vectors
   exactly, then compare with the deterministic 20,000-draw check. The seeded
   empirical mean is approximately `[-0.7439, -4.5297]`; it approaches the
   exact target `[-0.75, -4.5]` rather than equalling it in a finite run.
4. **“Minibatches preserve the mean and reduce spread.”** Enumerate every
   uniform subset at the fixed $\boldsymbol\theta_0$. For $B=1,2,4$, all three
   means equal the exact full-batch target $\boldsymbol g_0$ and the RMS distances contract
   `5.868 → 3.388 → 0.000`. No optimizer update occurs inside this estimator
   comparison.
5. **“One valid stochastic step need not lower the full loss.”** Verify
   $\ell_2: .500\to.405$ while
   $\mathcal L:7.125\to7.205$, with full-loss change `+0.080`. Read the four
   per-example change bars before the mean.
6. **“Batch, iteration, update, and epoch” and “The same mechanics in
   PyTorch.”** For $B=4,2,1$, verify `1,2,4` iterations/updates per epoch. The
   float64 autograd gradient exactly matches the analytical gradient. The
   longer loop reports `80 epochs × 2 minibatches = 160 updates` and final full
   loss about `.03753`. Treat the nonuniform importance-weighting cell as
   optional sampler detail.

### Notebook 02 · `02_ewma_momentum.ipynb`

Colab: [`notebooks/L05/02_ewma_momentum.ipynb`](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L05/02_ewma_momentum.ipynb)

Use at Page 57 for 90 seconds in the canonical route or three minutes in the
extended route. In deck order, run the EWMA and momentum cells first; return to
the scale-only standardization checkpoint at Page 58 and the separate
standardized-correlated experiment at Pages 59–62.

1. **“The ravine comes directly from the four rows.”** Read the executed table
   before the matrix summary. At $\boldsymbol\theta_0=(-1.4,.35)$ the predictions
   are `[-2.1, 4.9, -4.9, 2.1]`, the residuals are
   `[-6.1, 10.9, -10.9, 6.1]`. Verify
   $\mathbf X(\boldsymbol\theta_0-\boldsymbol\theta^\star)$ reproduces those
   residuals,
   $\mathbf X^\top\mathbf X/N=\operatorname{diag}(1,100)$,
   $\boldsymbol g_0=$ `[-2.4, 85]`, and the full loss is `39.005`.
   Then inspect the notebook's exact 3-D plot of
   $\mathcal L(\theta_1,\theta_2)$ and connect its height to the contour levels
   on Page 35; both views come from this same analytic objective.
2. **“Build EWMA from three views of the same noisy signal.”** On the fixed-seed
   120-day temperature sequence, compare $\beta=.5,.9,.98$ with all smoothers
   initialized at the first reading. Predict responsiveness and lag before each
   reveal, then verify the worked $.9(30)+.1(36)=30.6$ update and the first
   exponential weights $(1-\beta),(1-\beta)\beta,(1-\beta)\beta^2$. Reuse those
   observations—do not generate a second signal—for the cold-start cell. With
   $m_0=0$, $\beta=.9$, and $x_1=17.96810897$, verify
   $m_1=1.796810897$, mass $1-\beta=.1$, and
   $\widehat m_1=m_1/(1-\beta)=17.96810897$. Compare the raw and corrected
   curves over the first 12 readings and state why neither operation removes
   measurement noise. The
   [browser explorer](https://nipunbatra.github.io/dl-teaching/interactives/ewma-explorer.html)
   reproduces those same observations, exposes the two numerical terms in each
   selected update, and then applies the same memory to the computed ravine.
3. **“Couple the momentum state to the path it actually creates.”** Use the full
   recurrence $\boldsymbol m_{t+1}=\beta\boldsymbol m_t+
   (1-\beta)\boldsymbol g_t$ with $\boldsymbol m_0=\boldsymbol0$. Before
   inserting numbers, set the same conceptual $\eta=.0195$ for GD and momentum,
   then choose $\beta=.25$. Thus
   $\boldsymbol m_1=.75\boldsymbol g_0=(-1.8,63.75)$ and
   $-\eta\boldsymbol m_1=(.0351,-1.243125)$: because the uncorrected memory is
   initially empty, the first momentum displacement is $75\%$ of plain GD's.
   Contrast this uncorrected optimizer state with the missing-mass-corrected
   temperature average. Check that every next gradient is evaluated at the
   parameters produced by the preceding memory. Read each row as
   $\boldsymbol\theta_t\to\boldsymbol g_t\to\boldsymbol m_{t+1}\to
   \boldsymbol\theta_{t+1}$. The exact first-six momentum states are
   $\boldsymbol m_1=(-1.8,63.75)$,
   $\boldsymbol m_2=(-2.223675,-13.546875)$,
   $\boldsymbol m_3=(-2.297073,-13.058789)$,
   $\boldsymbol m_4=(-2.281827,6.161711)$,
   $\boldsymbol m_5=(-2.244644,1.955334)$, and
   $\boldsymbol m_6=(-2.202520,-1.955936)$. They produce
   $\boldsymbol\theta_1=(-1.3649,-.893125)$,
   $\boldsymbol\theta_2=(-1.321538,-.628961)$,
   $\boldsymbol\theta_3=(-1.276745,-.374315)$,
   $\boldsymbol\theta_4=(-1.232250,-.494468)$,
   $\boldsymbol\theta_5=(-1.188479,-.532597)$, and
   $\boldsymbol\theta_6=(-1.145530,-.494456)$, with strictly decreasing losses
   `10.523739, 3.526316, 3.381626, 2.493000, 2.447849, 2.303186`.
   For plain GD, the corresponding first-six parameter points are
   `(-1.3532,-1.3075)`, `(-1.307313,.267125)`,
   `(-1.262320,-1.228769)`, `(-1.218205,.192330)`,
   `(-1.174950,-1.157714)`, and `(-1.132538,.124828)`, with losses
   `35.371588, 32.085884, 29.114240, 26.426279, 23.994575, 21.794368`.
4. **“Judge momentum on the full objective.”** After 55 exact gradient
   evaluations at the same $\eta=.0195$, expect final losses `0.458144` for GD
   and `0.329775` for EWMA-form momentum with $\beta=.25$. Both losses decrease
   strictly, and momentum is lower from update 1. The path check reports 55
   versus 12 material crossings at tolerance $10^{-6}$ and across-valley travel
   `31.176297` versus `1.975095`; use those as mechanism clues, while the full
   objective remains the outcome.
5. **“Map the EWMA state to PyTorch's buffer.”** Define PyTorch's
   `momentum_buffer` by
   $\widetilde{\boldsymbol m}_{t+1}=\beta
   \widetilde{\boldsymbol m}_t+\boldsymbol g_t$. Then
   $\boldsymbol m=(1-\beta)\widetilde{\boldsymbol m}$ and
   `lr_torch = eta_EWMA * (1 - beta) = .014625`. Keep this as a brief
   implementation note after the same-$\eta$ conceptual comparison. Verify the
   full parameter path and mapped buffer state to numerical precision rather
   than comparing only endpoints.
6. **“Now test the preprocessing counterfactual.”** For the identical Example B
   start, predict whether $\theta_2$ crosses its optimum value $-.5$ under
   $\eta=.005$ and $.0195$, then compare the computed paths. Standardize with
   $z_2=x_2/10$ and verify that the curvature changes from $(1,100)$ to $(1,1)$;
   remember that the corresponding learned coefficient is
   $\phi_2=10\theta_2$. This establishes why preprocessing fixes this particular
   ravine without claiming that every deep-network loss becomes round.
7. **“Standardized does not mean uncorrelated.”** Switch to the separate exact
   four-row control
   $\mathbf X=[(-1,-7/5),(-1,-1/5),(1,1/5),(1,7/5)]$ with
   $\boldsymbol\theta^\star=(1,0)$ and $y=x_1$. Verify zero means, unit
   variances, correlation $.8$,
   $\mathbf H=\begin{bmatrix}1&.8\\.8&1\end{bmatrix}$, eigenvalues `.2` and
   `1.8`, condition number `9`, and contour-axis ratio `3`. Compare GD and
   EWMA-form momentum at the same conceptual $\eta=1$, with $\beta=.25$ for
   momentum. Over 18 exact updates both losses decrease strictly; final losses
   are approximately `1.62259e-4` and `8.35231e-6`. Their first moves differ
   because momentum uses its uncorrected zero-start memory. Do not mix these
   parameters with raw-feature Example B.

### Notebook 03 · `03_adaptive_optimizers_constraints.ipynb`

Colab: [`notebooks/L05/03_adaptive_optimizers_constraints.ipynb`](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L05/03_adaptive_optimizers_constraints.ipynb)

Use at Page 85 for adaptive-state checks, Page 93 for the schedule clock, and
Pages 97–108 for the worked constraint repairs. Pages 95–100 motivate all four
constraint families before opening the corresponding notebook sections.

1. **RMSProp initialization and two updates.** Start from
   $\boldsymbol v_0=\boldsymbol0$. Reproduce
   $\boldsymbol v_1=(.576,722.5)$, rulers
   $\sqrt{\boldsymbol v_1}=(.759,26.879)$, and the first rescaled gradient.
   Then execute update 2 and confirm that its unequal rescaled magnitudes reflect
   two-gradient history. State explicitly: PyTorch `square_avg` is
   the stored $\boldsymbol v$; the denominator ruler is
   `square_avg.sqrt() + eps`; the equal first-step magnitudes are a zero-start
   effect only. Compare the complete scratch and PyTorch parameter/state
   histories to numerical precision rather than citing a brittle exact
   floating-point difference.
2. **Adam's two raw moments.** Begin by retrieving the Notebook 02 temperature
   result: $t$ one-based observations have mass $1-\beta^t$. Then return to the
   zero-based optimizer clock, where measurement $\hat{\boldsymbol g}_t$ creates
   state $t+1$ and mass $1-\beta^{t+1}$. With
   $\boldsymbol m_0=\boldsymbol v_0=\boldsymbol0$, trace update 1 through
   `exp_avg`, `exp_avg_sq`, both missing-mass corrections, and the sign-sized
   first move. Explicitly contrast this with Notebook 02's uncorrected ordinary
   momentum state. On update 2, show that both corrected memories now contain
   history. Check the complete vectorized and PyTorch paths and both stored-state
   histories to numerical precision.
3. **Mechanism comparison.** Name the state before reading each trajectory: GD
   none, momentum signed direction, RMSProp squared coordinate scale, Adam both.
   Page 84 and the notebook share 55 exact-gradient updates and method-specific
   rates; their final-loss order on this one ravine is mechanism evidence, not a
   universal optimizer ranking. When discussing crossings, distinguish raw sign
   flips from material crossings above the displayed numerical tolerance.
4. **Learning-rate scheduling.** Reproduce the identical seed-12 sampled-index
   stream and all three policies: fixed $.10$, fixed $.02$, and inverse-time
   $\eta_t=.10/(1+.02t)$. At Page 88, verify why the exact full-batch method
   stops at the optimum while individual sample gradients remain nonzero. At
   Page 89, recover the two fixed-rate first crossings, `8` and `56`; Page 90
   adds the decayed policy, whose first crossing is also `8`. At Page 91,
   recover the fixed/decayed final-99 distances `3.79448` and `.520387` (the
   notebook also retains the small-fixed diagnostic `.739121`). The learning-rate policy
   is the controlled intervention, but later gradients and optimizer states are
   path-dependent. At Page 93, inspect the scheduler rates and use the same
   explicit clock as the deck: when `optimizer.step()` executes with $\eta_t$,
   count one update and then call `scheduler.step()` to prepare $\eta_{t+1}$,
   even if a zero gradient caused no numerical parameter change. If the optimizer
   step is skipped, as on an AMP overflow, skip the scheduler step too.
5. **Positive Gaussian scale.** Reproduce Page 97's direct-scale failure from
   the actual Gaussian NLL: residual `.05`, $\sigma_0=.10$, gradient `7.5`, and
   $\eta=.02$ give $\widetilde\sigma_1=-.05$. Then optimize the unrestricted
   log-scale $a$ and rebuild $\sigma=\exp(a)$ inside every forward pass. The
   tiny homoskedastic fit should learn `mu=.0750` and `sigma=.7980`, matching the
   closed-form check; explain that a heteroskedastic network produces one
   $a(x_i)$ and therefore one positive $\sigma_i$ per input. Mention `softplus`
   only as an alternative positive map.
6. **Box, probability, and simplex maps.** Match Pages 103–105. The affine
   sigmoid experiment should keep all 160 displayed blend coefficients in
   $(0,1)$ and recover `c=.7000`. The Bernoulli experiment should learn logit
   `1.0986`, whose sigmoid is `.7500`; emphasize that training uses
   `BCEWithLogitsLoss`. The three-class experiment should return weights close
   to `[.2,.6,.2]` and sum to one; categorical training uses `log_softmax` rather
   than taking `log(softmax(...))` manually. Then run the separate
   mixture-density cell: logits `(0, log 2, 0)` give weights
   `[.25,.50,.25]`, and
   `logsumexp(log_weights + component_log_prob)` gives finite NLL `1.0679` on
   the displayed observations.
7. **Projection versus a smooth interior.** For Pages 106–108, form the ordinary
   proposal first and project second for the closed interval $[0,1]$. With
   $\eta=.40$ for 35 updates, projection reaches the boundary optimum `1.0`,
   while the affine-sigmoid parameterization remains at `.781698`. Clamp only
   after `optimizer.step()` under `torch.no_grad()`, and note that momentum/Adam
   state is not projected or reset. Use projection only for simple closed convex
   sets here; a strict Gaussian scale and simplex are clearer through `exp` and
   `softmax`.

## Board derivations worth preserving

1. **[CORE] Line-fit gradient:** residual times $(1,x_i)$ for all four examples,
   then average.
2. **[CORE · primary reproducible derivation] Ravine from data:**
   $\mathbf X^\top\mathbf X/N=\operatorname{diag}(1,100)$,
   $\mathcal L(\boldsymbol\theta)=
   \tfrac12(\theta_1-1)^2+50(\theta_2+.5)^2$, and
   $\nabla_{\boldsymbol\theta}\mathcal L=
   (\theta_1-1,100(\theta_2+.5))$. Define curvature and identify the shallow
   and steep parameter directions on the contour before computing the first
   overshooting GD step.
3. **[SHOULD-COVER] Unbiasedness:** the one-line conditional expectation under
   a fresh uniform draw at fixed $\boldsymbol\theta_t$:
   $\mathbb E[\hat{\boldsymbol g}_t\mid\boldsymbol\theta_t]=
   \boldsymbol g_t$.
4. **[SHOULD-COVER] EWMA and missing mass:** the concrete
   $.9(30)+.1(36)=30.6$ blend followed by the three-term exponential expansion.
   On the same fixed-seed readings, switch from $m_1=x_1$ to $m_0=0$ and derive
   $(1-\beta)\sum_{k=0}^{t-1}\beta^k=1-\beta^t$; verify
   $1.7968/(1-.9)=17.9681$ on day 1.
5. **[SHOULD-COVER] Uncorrected momentum:** under the deck control
   first write
   $\boldsymbol m_{t+1}=\beta\boldsymbol m_t+
   (1-\beta)\boldsymbol g_t$ and
   $\boldsymbol\theta_{t+1}=\boldsymbol\theta_t-\eta\boldsymbol m_{t+1}$.
   Use the same conceptual $\eta=.0195$ as GD, set $\beta=.25$, and compute
   $\boldsymbol m_1=(-1.8,63.75)$ and
   $\boldsymbol\theta_1=(-1.3649,-.893125)$. After remeasuring
   $\boldsymbol g_1=(-2.3649,-39.3125)$, compute
   $\boldsymbol m_2=(-2.223675,-13.546875)$ and
   $\boldsymbol\theta_2=(-1.321538,-.628961)$. State that ordinary momentum
   applies the uncorrected EWMA state directly, so its first move is $75\%$ of
   GD's at the same $\eta$; no Adam-style bias correction or matched-step
   rescaling is used.
6. **[SHOULD-COVER] RMSProp:** one complete elementwise scale calculation,
   $\boldsymbol\theta_0\to\boldsymbol g_0\to\boldsymbol v_1
   \to\boldsymbol\theta_1$, including the $\sqrt{10}$ cold-start magnitude,
   followed by $\boldsymbol v_2$ to show how the second update contains genuine
   history and no longer equalizes the rescaled magnitudes.
7. **[SHOULD-COVER] Adam:** translate the temperature mass $1-\beta^t$ to the
   optimizer's state clock $1-\beta^{t+1}$, then perform the bias-corrected first
   update and its restricted sign-sized interpretation. Correct
   $\boldsymbol m_{t+1}$ and $\boldsymbol v_{t+1}$ separately because
   $\beta_1\ne\beta_2$.
8. **[OPTIONAL] Sampler covariance:** independent draws versus the
   finite-population correction.
9. **[OPTIONAL] Constraint chain rule:** with $a=\log\sigma$ and
   $\sigma=\exp(a)$, show
   $\partial\mathcal L/\partial a=(\partial\mathcal L/\partial\sigma)\sigma$;
   compare this valid raw-coordinate update with Page 97's invalid direct-scale
   proposal.

## Common misconceptions and repairs

- **“An exact gradient is the optimal update.”** It is the locally steepest
  direction under the chosen coordinates; one global rate may still zigzag in a
  ravine.
- **“Feature standardization makes momentum unnecessary.”** It removes the
  deliberately created $100{:}1$ input-scale imbalance in Example B. The
  separate Pages 59–62 exact four-row control has unit feature variances but
  correlation $.8$; its curvature eigenvalues are $.2$ and $1.8$, condition
  number $9$, and contour-axis ratio $3$, so standardization has not made that
  loss round.
  It also does not remove changing deep-network geometry or minibatch
  fluctuation; normalization and momentum address different parts of the
  problem.
- **“A larger $\beta$ is always better.”** It smooths more history but responds
  more slowly when the underlying direction changes. The $.5,.9,.98$
  temperature sequence makes that lag visible before gradients are involved.
- **“Setting $m_0=0$ says the initial temperature or gradient is zero.”** Zero is
  an empty memory, not an observed value. On the temperature example it leaves
  only $1-\beta^t$ total weight on readings after day $t$; dividing by that mass
  removes the artificial attraction to zero but does not denoise the readings.
- **“Ordinary momentum should use Adam's bias correction.”** In this lecture,
  ordinary momentum deliberately applies the uncorrected EWMA state
  $\boldsymbol m_{t+1}=\beta\boldsymbol m_t+
  (1-\beta)\boldsymbol g_t$ as optimizer dynamics. The conceptual comparison
  gives GD and momentum the same $\eta=.0195$; the smaller first momentum move
  is the honest effect of empty memory. Adam instead treats two differently
  attenuated zero-start memories as moment estimates and normalizes each before
  taking their ratio.
- **“`torch.optim.SGD` means batch size one.”** The `DataLoader` and reduction
  define the gradient's data; the optimizer transforms the gradient it receives.
- **“Unbiased means every sample points downhill.”** It describes the centre of
  repeated draws at one fixed $\boldsymbol\theta_t$ under the stated sampler:
  $\mathbb E[\hat{\boldsymbol g}_t\mid\boldsymbol\theta_t]=\boldsymbol g_t$.
- **“A shuffled epoch gives fresh independent samples.”** Later batches are
  constrained by what has already appeared in that epoch.
- **“Minibatches remove noise.”** They reduce spread. A uniform subset containing
  every example is exact; $N$ independent draws with replacement can repeat and
  omit examples and therefore need not have zero variance.
- **“Iteration, update, and epoch are synonyms.”** They coincide only in a simple
  loop without gradient accumulation or skipped steps.
- **“Adam's moving-average correction makes SGD unbiased.”** It is the same
  missing-weight-mass repair first shown on the zero-start temperature average,
  now applied separately to $\boldsymbol m$ and $\boldsymbol v$. It repairs
  initialization shrinkage. Gradient-estimator unbiasedness is a separate claim
  about the sampling distribution.
- **“PyTorch stores the same normalized EWMA shown in the conceptual rule.”**
  PyTorch stores the unnormalized buffer
  $\widetilde{\boldsymbol m}_{t+1}=\beta
  \widetilde{\boldsymbol m}_t+\boldsymbol g_t$. Since
  $\boldsymbol m=(1-\beta)\widetilde{\boldsymbol m}$, the equivalent PyTorch
  learning rate is $\eta(1-\beta)$. Derive this mapping only after the
  normalized EWMA-form optimizer is understood.
- **“RMSProp bias-corrects its square average like Adam.”** Standard PyTorch
  RMSProp does not. With $\beta_2=.9$ and zero state, the first normalized
  magnitude is inflated to $\sqrt{10}$ for every nonzero coordinate.
- **“RMSProp/Adam use one scalar scale.”** Their squares, roots, and divisions are
  elementwise tensors.
- **“Adam's $\boldsymbol v_t$ is the gradient variance.”** It is an EWMA of
  squared gradients, a second raw moment.
- **“Adam is sign descent.”** That approximation describes only the first
  bias-corrected nonzero update from zero state, ignoring
  $\epsilon_{\mathrm{opt}}$.
- **“Adam always beats SGD.”** The deck compares mechanisms at method-specific
  rates; real comparisons require tuned rates, equal budgets, and held-out
  metrics.
- **“A scheduler naturally counts epochs.”** A schedule may be driven by
  optimizer updates, epochs, or a validation event such as a plateau check; the
  code and API call site define the clock. This deck deliberately uses optimizer
  updates: each executed `optimizer.step()` uses $\eta_t$ and advances $t$, even
  if a zero gradient leaves the parameters numerically unchanged; the following
  `scheduler.step()` prepares $\eta_{t+1}$. Accumulated microbatches do not
  advance this clock. If `optimizer.step()` is skipped, for example after an AMP
  overflow, `scheduler.step()` must also be skipped.
- **“Sigmoid represents the closed interval $[0,1]$ exactly.”** Finite logits
  mathematically represent $(0,1)$; exact endpoints may need projection or
  another model. In finite precision, extreme logits can still round to 0 or 1,
  so prefer logits-based losses.
- **“We can optimize a Gaussian scale directly and clamp it if needed.”** A
  Gaussian likelihood requires $\sigma>0$; zero is not a valid scale. Learning
  $a=\log\sigma$ and rebuilding $\sigma=\exp(a)$ makes every forward pass valid
  for both shared and input-dependent noise. `softplus` is a possible positive
  alternative, not this lecture's primary convention.
- **“Clipping each mixture weight makes a simplex.”** Independent clipping can
  make every coordinate nonnegative without making the coordinates sum to one.
  Softmax couples the weights and enforces both requirements together.
- **“Clamping after Adam is harmless.”** The parameter is projected but stored
  moments are not automatically reconciled with that projection.

## If the room is quiet: prediction prompts

1. Which one-sample update raises the full line-fit loss?
2. If $B$ doubles under independent sampling, what happens to covariance?
3. At the shared rate $\eta=.0195$, why does the steep $\theta_2$
   component alternate signs?
4. Which $\beta$ gives more smoothing, and what does it sacrifice?
5. With the same first temperature reading and $\beta=.9$, why does $m_0=0$
   produce $m_1=1.80^\circ\mathrm C$, and which denominator restores the
   observed-weight mass to one?
6. Why does ordinary momentum use its uncorrected EWMA state as optimizer
   dynamics, while Adam corrects two zero-start moment estimates with different
   $\beta$ values?
7. Why do the first two $\theta_2$ gradient contributions nearly cancel in the
   $\beta=.25$ momentum state while the $\theta_1$ contributions reinforce?
8. Why does replacing $x_2$ by $z_2=x_2/10$ remove this example's $100{:}1$
   curvature imbalance, and what does it not repair in a deep network?
9. Why does RMSProp square before averaging?
10. Which Adam state remembers sign, and which remembers scale? Why do the
    correction exponents use $t+1$ rather than the temperature slide's $t$?
11. Does decay change the sampled data, optimizer state definition, or only the
   applied step scale in the displayed experiment?
12. Which raw variables and transforms would you use for a shared or
    input-dependent Gaussian scale, a blend coefficient in $[0,1]$, a Bernoulli
    probability, and simplex mixture weights? Which of these models needs a
    strict interior, and when is projection to an exact boundary appropriate?

## Follow-along after the 80-minute route

- Read the post-80 coda on Pages 94–110: constrained parameter families,
  repairs, and the written six-decision synthesis. The live class has
  already closed with the 20-second verbal version of that retrieval.
- In Notebook 00, compare uniform subsets with independent draws with
  replacement and state which sampler each variance claim assumes.
- In Notebook 02, inspect every EWMA-form momentum state and its mapped PyTorch
  `momentum_buffer`; verify $\boldsymbol m=(1-\beta)
  \widetilde{\boldsymbol m}$ and
  `lr_torch = eta_EWMA * (1 - beta) = .014625` for the displayed
  $\eta=.0195,\beta=.25$ control. Then verify exactly what the scale-only
  standardization counterfactual changes, and why the exact four-row
  standardized-correlated control still has correlation $.8$, condition number
  $9$, and contour-axis ratio $3$.
- In Notebook 03, finish the scheduler trace and all four constraint maps, then
  run the projected-box comparison; explain mathematical interiors,
  finite-precision saturation, logits-based losses, and projected optimizer
  state.

## Closing line

“Backprop measures a gradient. Batch size controls its noise; momentum remembers
direction; RMSProp remembers scale; Adam remembers both; the schedule controls
time; and a parameterization controls which values the model can express.”

## Primary references

- Andrew Ng, [*Improving Deep Neural Networks: Hyperparameter Tuning,
  Regularization and Optimization — Week 2*](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc),
  C2W2L01–L09—the approachable minibatch → EWMA → momentum → RMSProp → Adam →
  decay order adapted here.
- DeepLearning.AI, [*Learning Rate
  Decay*](https://www.youtube.com/watch?v=QzulmoOg2JE)—the large-early,
  small-late motivation and schedule intuition adapted in Pages 86–93.
- *Dive into Deep Learning*, [*Stochastic Gradient
  Descent*](https://classic.d2l.ai/chapter_optimization/sgd.html) and [*Learning
  Rate Scheduling*](https://d2l.ai/chapter_optimization/lr-scheduler.html)—the
  stochastic late-motion motivation and common schedule families.
- Mitesh M. Khapra, [*Deep Learning (CS7015): Lec 5.4 Momentum Based Gradient
  Descent*](https://www.youtube.com/watch?v=ewN0vFYFJ7A)—the problem → repeated-direction
  intuition → zero-initialized update expansion → actual contour behavior
  sequence adapted in the momentum section and browser stepper.
- Sourish Kundu, [*Who's Adam and What's He Optimizing?*](https://www.youtube.com/watch?v=MD2fYip6QsQ),
  especially 5:39–7:09—the shrinking-step and rolling-ball intuitions. The
  lecture pairs those with the actual ravine's sign reversals and states the
  analogy's limits explicitly.
- Florian Wilhelm, [*Estimators*](https://florian.github.io/estimators/)—the
  estimator framing and expectation calculation; this guide states the sampling
  assumptions explicitly.
- Kingma & Ba (2015), [*Adam: A Method for Stochastic
  Optimization*](https://arxiv.org/abs/1412.6980).
- PyTorch, [`torch.optim.SGD`](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html),
  [`RMSprop`](https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html),
  and [`Adam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)—
  implementation conventions and exposed optimizer state.
- PyTorch, [*How to adjust learning
  rate*](https://docs.pytorch.org/docs/main/optim.html#how-to-adjust-learning-rate)—
  scheduler composition and call-order conventions.

All optimizer trajectories in the deck are generated from the displayed losses
and update rules. Treat the notebooks as deterministic mechanism checks, not
benchmark evidence.
