# Lecture 5 · Optimization for Deep Learning · Teaching Guide

*Instructor companion for the rebuilt public Lecture 5 deck and its four exact notebooks.*

## The spine

Backpropagation gives a gradient; optimization decides **which data measure that
gradient, what history to remember, how strongly to act, and which coordinates
the model is allowed to use**.

Teach the lecture as one progressive diagnosis, not an optimizer catalogue:

1. retrieve vanilla full-batch gradient descent on one four-point line fit;
2. change only the data used per update: one example, then a minibatch;
3. explain the centre and spread of that stochastic gradient estimator;
4. derive a genuine ravine from a second, scaled-feature regression problem;
5. introduce EWMA and momentum from a noisy-signal analogy, then verify the
   repair on the actual ravine;
6. ask whether feature standardization could remove this constructed failure,
   state what momentum still addresses, and then introduce RMSProp and Adam;
7. reduce late stochastic wandering with learning-rate decay;
8. hold data, model, initialization, minibatches, and budget fixed on XOR while
   changing only the optimizer; and
9. handle positive, bounded, and simplex parameters by changing coordinates or
   projecting.

The three recurring model examples are deliberately small enough to calculate
and plot exactly. The synthetic temperature sequence exists only to make EWMA
intuitive before the signal becomes a gradient.

## Where it sits

- **Retrieves Lectures 2–3:** the loss surface, gradient geometry, and the vanilla
  update $\boldsymbol g_t=\nabla_{\boldsymbol\theta}\mathcal L(\boldsymbol\theta_t)$,
  $\boldsymbol\theta_{t+1}=\boldsymbol\theta_t-\eta\boldsymbol g_t$.
- **Builds directly on Lecture 4:** backprop computes the current batch gradient;
  this lecture turns that measurement into an update.
- **Sets up Lecture 6:** initialization, activations, normalization, and residual
  paths make useful activations and gradients survive depth.
- **Leaves optimizer regularization to Lecture 7:** keep that separate from this
  lecture's direction, scale, and step-size mechanisms.

## Notation contract

Keep the notation inherited from Lectures 1–3 visible throughout:

- bold lowercase symbols are vectors: $\boldsymbol x_i$, $\boldsymbol\theta$,
  $\boldsymbol g$, $\boldsymbol m$, $\boldsymbol s$, and
  $\boldsymbol v$; $y_i$, $\hat y_i$, $\ell_i$, and $\mathcal L$ are scalars.
  Example A intentionally uses a single scalar feature $x_i$;
- bold uppercase symbols are matrices, such as the design matrix $\mathbf X$;
- $\boldsymbol g_t=\nabla_{\boldsymbol\theta}\mathcal L(\boldsymbol\theta_t)$
  is the exact full gradient, whereas $\hat{\boldsymbol g}_t$ is a sampled or
  minibatch estimator evaluated at the same current parameters;
- the optimizer clock is zero-based: parameters $\boldsymbol\theta_t$ produce
  measurement $\boldsymbol g_t$ or
  $\hat{\boldsymbol g}_t$, then state $\boldsymbol m_{t+1}$,
  $\boldsymbol s_{t+1}$, or $\boldsymbol v_{t+1}$ and parameters at $t+1$;
- $\epsilon_{\mathrm{opt}}$ stabilizes an optimizer denominator, whereas
  $s_{\min}$ is a modelling floor in a positive parameterization; and
- the standalone temperature EWMA uses the standard one-based signal notation
  $x_t\mapsto m_t$. Once that memory is coupled to an optimizer, return to the
  zero-based measurement/next-state convention above.

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
  directions, overshoot, and oscillation; and explain what feature
  standardization repairs;
- expand an EWMA and explain the smoothness-versus-lag trade-off across
  $\beta=.5,.9,.98$;
- implement normalized momentum from scratch and recognize PyTorch's different
  momentum-buffer convention;
- explain RMSProp as coordinate-scale memory and Adam as direction plus scale
  memory;
- inspect `momentum_buffer`, `exp_avg`, and `exp_avg_sq`, and verify the displayed
  RMSProp/Adam recurrences against PyTorch;
- state which clock advances a learning-rate schedule;
- design a controlled optimizer comparison that shares initialization,
  minibatch order, model, loss, and update budget; and
- choose reparameterization or projection for common constraints, including the
  boundary trade-off.

## Evidence language

| Material | Honest label | How to describe it |
|---|---|---|
| Four-point line fit, gradients, losses, contours, and decay plots | **Constructed data; computed exactly** | “Every number and trajectory comes from the four displayed examples.” |
| Batch-size averaging diagram | **Exact enumeration at one fixed parameter value** | “The arrows are computed from Example A, but the picture suppresses numbers to isolate the averaging intuition.” |
| Individual and minibatch loss-surface sequence | **Exact losses and gradients at one shared parameter value** | “The deck separates surfaces from arrows, then shows the worked $B=1,2,4$ family; Notebook 00 exposes the underlying calculations.” |
| Scaled-feature ravine | **Constructed regression; computed empirical loss** | “The ellipse is the loss of the displayed design matrix, not a decorative quadratic or a real-data benchmark.” |
| Temperature sequence | **Fixed-seed synthetic signal** | “The same 120-day observations isolate EWMA's memory; only $\beta$ changes before we apply the idea to gradients.” |
| Estimator arrows and batch-spread plots | **Exact enumeration at one fixed parameter value** | “The centre and spread are computed over all displayed samples or subsets.” |
| Optimizer-path comparison | **Mechanism demonstration** | “Rates are method-specific; this is not a leaderboard or universal ranking.” |
| XOR loss curves and decision boundaries | **Seeded controlled mechanism demonstration** | “The four runs share initialization, minibatch order, model, loss, and 400-update budget; training accuracy is not generalization evidence.” |
| Throughput and hardware statements | **Qualitative systems guidance** | “Actual wall-clock performance depends on vectorization, memory, and hardware.” |
| Constraint maps and projected path | **Computed toy examples** | “They explain feasibility and chain-rule effects, not convergence guarantees for arbitrary constrained problems.” |

Use the deck tags as teaching cues: `[Q]` requires a prediction, `[D]` deserves at
least one line of live mathematics, `[V]` should be read from the picture before
the caption, and `[I]` is a notebook or code checkpoint.

## Before class

1. Open the handout and presentation PDFs.
2. Open these exact local notebooks and restart/run all cells in order:
   - `notebooks/L05/00_full_batch_sgd_minibatch.ipynb`
   - `notebooks/L05/02_ewma_momentum.ipynb`
   - `notebooks/L05/03_adaptive_optimizers_constraints.ipynb`
   - `notebooks/L05/04_xor_optimizer_lab.ipynb`
3. Keep one board panel for the four-point line-fit gradients and one for the
   ravine loss, directions, and first overshooting step. Do not erase them when
   the optimizer names change.
4. The course page uses Colab URLs on the public `master` branch. Until this
   worktree is published to that repository, use the local notebooks and do not
   tell students that the four new Colab targets are live.

## 80-minute canonical route through the 89-page handout

This is the default live route. The labels implement the course timing contract:

- **[CORE]** is non-negotiable: it carries the causal story or a mastery check.
- **[SHOULD-COVER]** is taught briefly when the room is on pace and otherwise
  assigned as immediate follow-along.
- **[OPTIONAL]** is enrichment for the 110-minute route or after class.

The page numbers below are physical PDF pages and count the title page.
Separators are used only to pose the next failure; do not spend time reading them.

The presentation PDF has 115 states because a few handout pages reveal in
stages. Navigate by slide title, and finish all states carrying that title
before advancing. The current physical-page/state synchronization is:

| Handout pages | Presentation states | Arc |
|---|---:|---|
| 1 | 1 | Title |
| 2–4 | 2–6 | Bridge, outline, and examples |
| 5–16 | 7–33 | Full batch, SGD, and minibatches |
| 17–19 | 34–36 | Batch-size comparison and PyTorch |
| 20–30 | 37–48 | Loss surfaces and estimator reasoning |
| 31–36 | 49–54 | Exact-gradient ravine |
| 37–51 | 55–69 | EWMA and momentum |
| 52–53 | 70–71 | Feature standardization |
| 54–68 | 72–90 | RMSProp and Adam |
| 69–74 | 91–98 | Learning-rate schedules |
| 75–83 | 99–109 | XOR implementation bridge |
| 84–87 | 110–113 | Constrained parameters |
| 88–89 | 114–115 | Retrieval and synthesis |

### 0–5 min · Pages 2–4 · bridge and recurring examples

- **[CORE] Page 2, “Backprop measured the slope; today we decide how to
  move.”** Separate the measured gradient from the optimizer's update state.
- **[CORE] Page 3, “Outline.”** Read only the five verbs: choose, understand,
  expose, add, finish.
- **[CORE] Page 4, “Three examples will keep the mechanisms concrete.”**
  Establish Example A for sampling, Example B for geometry, and Example C for
  the neural-network implementation check.

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
  $\boldsymbol g_0=(-.75,-4.5)$, and $\eta=.10$ gives
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
  `zero_grad → backward → step`. With gradient accumulation or a skipped step,
  a microbatch/iteration need not be an optimizer update.

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

### 33–41 min · Pages 31–36 · exact gradient and ravine geometry

- **[CORE] Page 31, “An exact gradient can still be slow.”** Close the sampling
  story: every update in Example B now uses the exact full-batch gradient. The
  remaining question is whether one scalar learning rate can choose a useful
  distance in every parameter direction.
- **[CORE] Page 32, “Example B: one feature is ten times larger.”** Read the
  feature scales $\pm1$ and $\pm10$ before doing algebra. Stay in the model's
  original parameter coordinates: the optimum is
  $\boldsymbol\theta^\star=(1,-.5)$ and the displayed start is
  $\boldsymbol\theta_0=(-1.4,.35)$.
- **[CORE] Page 33, “The $10\times$ feature creates a $100\times$ sensitive
  direction.”** Obtain
  $\mathcal L(\boldsymbol\theta)=
  \tfrac12(\theta_1-1)^2+50(\theta_2+.5)^2$ and
  $\nabla_{\boldsymbol\theta}\mathcal L=
  (\theta_1-1,100(\theta_2+.5))$. Define *curvature* before using it: it is how
  quickly the slope changes as a parameter moves. The feature factor $10$ is
  squared by the loss, so the $\theta_2$ direction is $100\times$ more
  sensitive.
- **[CORE] Page 34, “A ravine is a long, narrow low-loss valley.”** Name the
  geometry directly on the equal-aspect contour in parameter space. Moving
  horizontally in $\theta_1$ is the shallow direction; moving vertically in
  $\theta_2$ is the steep direction. “Along the valley” and “across the valley”
  are optional geometric descriptions, not additional coordinates students
  must track.
- **[CORE] Page 35, “One correct step can overshoot across the valley.”** Work
  $\boldsymbol g_0=(-2.4,85)$,
  $\Delta\boldsymbol\theta_0=(.0456,-1.615)$, and
  $\boldsymbol\theta_1=(-1.3544,-1.265)$ at $\eta=.019$. The gradient points
  uphill and the update $-\eta\boldsymbol g_0$ points downhill; its second
  component carries $\theta_2$ past the optimum $-.5$. Call this one crossing
  *overshoot*, not yet oscillation.
- **[CORE] Page 36, “The trade-off is visible in the computed paths.”** Compare
  the actual full-batch paths at $\eta=.005$ and $.019$ under the same start
  and update budget. The smaller rate avoids the first crossing but moves
  $\theta_1$ by only $.012$; the larger rate moves farther in the shallow
  $\theta_1$ direction while overshooting in the steep $\theta_2$ direction.
  One global rate creates the trade-off. Define
  *oscillation* here as repeated side-to-side crossings of the valley—not merely
  one overshoot. The evidence is computed on Example B, not a decorative path.

### 41–55 min · Pages 37–51 · noisy signal → EWMA → momentum

- **[CORE] Page 37, “How can we smooth a noisy sequence?”** Use the separator
  only to pose the new problem. Do not introduce gradient notation yet.
- **[CORE] Page 38, “Start with the noisy observations—not the formula.”** Read
  the seeded 120-day temperature signal first. Define *memory* as one running
  summary carried from one observation to the next; it does not store every
  reading. The target is the changing trend, not tomorrow's forecast. If time
  permits, open the [EWMA + momentum explorer](https://nipunbatra.github.io/dl-teaching/interactives/ewma-explorer.html):
  move the day control first, then compare $\beta=.5,.9,.98$ while keeping the
  observations fixed. Use the sudden-change control only after students can
  name the basic smoothness–lag trade-off.
- **[CORE] Page 39, “Short memory reacts quickly—and follows the noise.”** Before
  reading the curve, define $\beta$ as the fraction of the *previous summary*
  retained. With $\beta=.5$, old summary and new observation receive equal
  weight, so the estimate reacts quickly but remains noisy.
- **[CORE] Page 40, “More memory gives a useful compromise.”** Change only
  $\beta$ to $.9$: retain $90\%$ of the previous summary and add $10\%$ of the
  new reading. Ask students to identify both the reduced noise and the lag when
  the trend turns.
- **[CORE] Page 41, “Too much memory becomes slow to respond.”** At
  $\beta=.98$, the curve is smoother but visibly lags. Larger $\beta$ means
  longer memory, not automatically a better estimate.
- **[CORE] Page 42, “EWMA keeps one running memory $m_t$.”** Define $m_t$
  explicitly as the running memory—the smoothed value after processing
  observation $x_t$. Only after the three curves write
  $m_t=\beta m_{t-1}+(1-\beta)x_t$. Work
  $.9(30)+.1(36)=30.6$, then unroll just enough to see weights
  $(1-\beta),(1-\beta)\beta,(1-\beta)\beta^2,\ldots$ and earn the name
  exponentially weighted moving average (EWMA).
- **[CORE] Page 43, “Why ‘exponential’? Unroll the memory.”** Substitute the
  recurrence for $m_{t-1}$ and then $m_{t-2}$. Point to the resulting weights
  $(1-\beta)$, $(1-\beta)\beta$, and $(1-\beta)\beta^2$ before stating the
  general weight $(1-\beta)\beta^k$ on $x_{t-k}$. For $\beta=.9$, read the
  concrete sequence `.100, .090, .081, .0729, ...`: each additional step into
  the past multiplies the weight by $.9$. This geometric decay earns the word
  *exponential*.
- **[CORE] Page 44, “Same-direction pushes build speed; reversals cancel.”**
  First use the shallow parameter component to make the slowdown concrete:
  $g_{t,1}=\theta_{t,1}-1$, hence
  $|\Delta\theta_{t,1}|=\eta|\theta_{t,1}-1|$. As the parameter approaches its
  optimum value $1$, plain GD's next step shrinks because it uses only the
  current gradient. Then read the computed two-panel component diagram from the
  actual $\eta=.019$ full-batch path. The first gradient component keeps the
  same sign, so retained measurements reinforce; the second component reverses
  sign, so retained and current measurements partly cancel. Present the
  rolling-ball analogy as intuition for accumulated direction, not a literal
  force law or a guarantee of a downhill step. State that the displayed GD and
  normalized-momentum rates have been matched to give the same first
  displacement; otherwise “larger steps” would mix the memory mechanism with
  an arbitrary learning-rate change. Map the temperature rule to
  $\boldsymbol m_{t+1}=\beta\boldsymbol m_t+
  (1-\beta)\boldsymbol g_t$ for this exact-gradient example. The same optimizer
  can instead receive a sampled $\hat{\boldsymbol g}_t$ from a minibatch.
- **[CORE] Page 45, “Momentum update 1: there is no history yet.”** Keep the
  first update entirely separate. Define $\boldsymbol m_t$ as the optimizer's
  running memory of earlier gradients. Explain that
  $\boldsymbol m_0=\boldsymbol0$ is a deliberate initial condition: before any
  gradient has been observed there is no history to average and zero adds no
  preferred direction. It is neither a measured gradient nor a learned
  parameter. State the choices $\beta=.8$ and EWMA-form $\eta=.095$, but defer
  the PyTorch conversion to Page 51. Begin from
  $\boldsymbol\theta_0=(-1.4,.35)$, then read the
  three boxes in order: measure
  $\boldsymbol g_0=(-2.4,85)$; remember
  $\boldsymbol m_1=.8\boldsymbol m_0+.2\boldsymbol g_0
  =.2\boldsymbol g_0=(-.48,17)$; move to
  $\boldsymbol\theta_1=(-1.3544,-1.265)$. Emphasize the immediate consequence:
  $\beta$ cannot retain old history on update 1 because none exists. From
  update 2 onward, $\boldsymbol m_t$ carries earlier gradients.
- **[CORE] Page 46, “Momentum update 2: old and new directions can disagree.”**
  Keep $\boldsymbol m_1$, remeasure
  $\boldsymbol g_1=(-2.3544,-76.5)$ at the new point, and compute the two memory
  components separately. In the first parameter component,
  $-.384-.4709=-.8549$ reinforces; in the second parameter component,
  $13.6-15.3=-1.7$ mostly cancels. Only then move to
  $\boldsymbol\theta_2=(-1.2732,-1.1035)$. Close by naming the reusable loop:
  *measure → remember → move*. The cancellation is substantial here, not a
  guarantee of exact cancellation on every step. Reopen the browser explorer's
  momentum view if students need to scrub these rows while watching the contour
  path and memory arrow change together.
- **[CORE] Page 47, “What changed in the second momentum step?”** Read the
  three-panel visual left to right: the first move crosses the valley; the old
  and new first-parameter contributions reinforce while the second-parameter
  contributions nearly cancel; the resulting second-parameter move is much
  smaller than plain GD's proposal at the same point. This is the graphical
  version of Page 46's arithmetic.
- **[CORE] Page 48, “Momentum travels farther in the same 55 updates.”** Read
  the computed paths under the same start, first displacement, and 55 exact
  gradient evaluations. Momentum still crosses the valley, but makes much more
  progress in the shallow $\theta_1$ direction. Use crossings only as a
  mechanism clue; the full loss on Page 49 is the outcome that matters.
- **[CORE] Page 49, “Momentum is faster here—not downhill at every update.”**
  First resolve the apparent surprise in the curves: both use exact
  full-batch gradients, so momentum's sawtooth is not sampling noise. For plain
  GD at $\eta=.019$, the two coordinates update as
  $\theta_{t+1,1}-1=.981(\theta_{t,1}-1)$ and
  $\theta_{t+1,2}+.5=-.9(\theta_{t,2}+.5)$; both squared errors shrink, so the
  loss decreases on every displayed update. Momentum retains direction,
  can cross the valley with stored motion, and therefore can increase the loss
  on an individual update. Read the first concrete rise directly from the
  plot: $\mathcal L_2=20.7943\rightarrow\mathcal L_3=24.9464$ as the steep
  parameter moves from $\theta_{2,2}=-1.1035$ past the optimum to
  $\theta_{3,2}=.17235$. Judge the falling envelope and final objective, not
  stepwise monotonicity. On the identical 55-update
  budget, plain GD ends at approximately `.349461`, whereas momentum ends at
  `.00013619`. Momentum
  remains below $\mathcal L=.1$ from update 26 onward. This is controlled
  evidence on Example B, not a universal guarantee.
- **[SHOULD-COVER] Page 50, “Under the hood: one memory tensor per parameter.”**
  Read the short implementation as *measure → remember → move*. The current
  gradient is recomputed; only `state` persists between updates.
- **[CORE] Page 51, “PyTorch momentum uses the same training loop.”** Show
  `SGD(lr=.019, momentum=.8)`, then connect PyTorch's buffer to the normalized
  recurrence. Define $\boldsymbol v$ as PyTorch's stored `momentum_buffer`;
  then $\boldsymbol m=(1-\beta)\boldsymbol v$ and
  $\eta_{\mathrm{PT}}=(1-\beta)\eta=.019$. Use Notebook 02 to compare
  $\beta=.5,.9,.98$, reproduce the two worked momentum updates, and inspect the
  actual `momentum_buffer`.

### 55–57 min · Pages 52–53 · what preprocessing fixes—and what remains

- **[CORE] Page 52, “Could feature standardization fix this example? Yes.”**
  Ask this only after students have seen momentum work. Use $z_2=x_2/10$ to
  show the before/after geometry. In this constructed linear problem,
  standardization removes the $100{:}1$ scale imbalance. Preserve the same
  starting predictor: because $\phi_2=10\theta_2$, the parameter error relative
  to the optimum changes from $(-2.4,.85)$ in $\boldsymbol\theta$ coordinates
  to $(-2.4,8.5)$ in $\boldsymbol\phi$ coordinates. The figure gives both
  plain-GD paths ten updates and labels their stable rates ($.019$ before,
  $.35$ after); plain GD no longer needs momentum to repair this particular
  feature-scaling failure.
- **[CORE] Page 53, “Feature scaling helps; it does not make momentum
  obsolete.”** Give the direct answer: normalize strongly mismatched input
  features first. Momentum remains useful because deep networks can have
  coupled, changing parameter-space directions, and minibatch gradients still
  fluctuate after input scaling. Normalization changes geometry; momentum
  changes how successive gradients are combined. They are complementary, not
  competing repairs. State the experimental reset explicitly: Page 54 returns
  to the original unscaled ravine only so RMSProp's coordinate-scale mechanism
  remains visible.

### 57–66 min · Pages 54–68 · diagnose scale, then build RMSProp and Adam

- **[CORE] Page 54, “One learning rate cannot fit both directions”.** Return
  deliberately to the original unscaled Example B—not as recommended
  preprocessing, but as a controlled way to isolate the optimizer mechanism.
  The unresolved problem is one scalar $\eta$: a value large enough to move
  along the shallow direction is unsafe across the steep direction.
- **[CORE] Page 55, “The problem: one coordinate moves 35× farther”.** Use the
  exact starting gradient $\boldsymbol g_0=(-2.4,85)$ and compute
  $|g_{0,2}|/|g_{0,1}|=85/2.4\approx35.4$. Ask which coordinate should be
  reduced more before proposing the mechanism. This is a coordinate-scale
  diagnosis on the displayed loss, not a claim about every model.
- **[CORE] Page 56, “RMSProp intuition: give each coordinate its own ruler”.**
  Stay qualitative first: remember recent gradient *magnitude* separately in
  each coordinate, then divide by that local ruler. Read the picture before any
  recurrence: a historically large coordinate receives a smaller effective
  step; a smaller one is not suppressed by the same amount.
- **[CORE] Page 57, “The rule: remember squared magnitude, then rescale”.** Only
  now write
  $\boldsymbol s_{t+1}=\rho\boldsymbol s_t+(1-\rho)\hat{\boldsymbol g}_t^2$
  and the elementwise update. Explain why squaring removes sign while retaining
  scale, why the fading memory can adapt, and why
  $\epsilon_{\mathrm{opt}}$ is a numerical denominator stabilizer.
- **[CORE] Page 58, “Work one RMSProp update from start to finish”.** Follow the
  exact Example B chain
  $\boldsymbol\theta_0\to\boldsymbol g_0\to\boldsymbol s_1
  \to\boldsymbol\theta_1$.
  With $\rho=.9$, obtain $\boldsymbol s_1=(.576,722.5)$ and
  $\boldsymbol g_0/\sqrt{\boldsymbol s_1}\approx(-3.162,3.162)$. Name the
  $\sqrt{10}$ magnitude as zero-start inflation; with $\eta=.08$, finish at
  $\boldsymbol\theta_1\approx(-1.1470,.0970)$. Standard PyTorch RMSProp does not
  bias-correct this state.
- **[CORE] Page 59, “Computed outcome: the ravine path straightens”.** Read the
  computed path as evidence only after the diagnosis, intuition, and update.
  The displayed run uses the same Example B loss with
  $\eta=.08,\rho=.9,\epsilon_{\mathrm{opt}}=10^{-8}$ for 40 updates. Explain
  the repaired geometry, while stating that the method-specific rate makes this
  a mechanism demonstration rather than a universal superiority claim.
- **[SHOULD-COVER] Page 60, “PyTorch stores the ruler as square_avg”.** Map
  `square_avg` to $\boldsymbol s_{t+1}$ and inspect both that tensor and the
  parameter update; matching only the final path is insufficient.
- **[CORE] Page 61, “Adam: remember direction and scale”.** Use the separator
  to pose the next repair: RMSProp addressed unequal coordinate scale, while
  momentum addressed persistent versus alternating direction. Can one update
  retain both memories?
- **[CORE] Page 62, “Why combine them? the failures are different”.** Contrast
  the two failures on Example B before introducing Adam: momentum smooths sign
  but retains one global scale; RMSProp rescales coordinates but does not store
  a signed direction. Avoid implying that either failure appears in every task.
- **[CORE] Page 63, “Adam runs two memories in parallel”.** Map
  $\hat{\boldsymbol g}_t$ to a signed first-moment state
  $\boldsymbol m_{t+1}$ and a squared-magnitude second raw moment
  $\boldsymbol v_{t+1}$. Call $\boldsymbol v$ a second raw moment, not a
  variance, and keep the measurement/state/update clock explicit.
- **[CORE] Page 64, “Zero-start correction makes early memories comparable”.**
  Derive directly that zero-initialized EWMA weights have accumulated masses
  $1-\beta_1^{t+1}$ and $1-\beta_2^{t+1}$. The sign-sized approximation is restricted to the
  first corrected nonzero update from zero state, up to
  $\epsilon_{\mathrm{opt}}$; it is not Adam's general behavior. For the actual
  first Example B update shown, set $\eta=.10$ and omit $\epsilon_{\mathrm{opt}}$
  only for the arithmetic: $\Delta\boldsymbol\theta_0=(.10,-.10)$ and
  $\boldsymbol\theta_1=(-1.30,.25)$.
- **[CORE] Page 65, “Same ravine, four optimizers”.** Read the shared
  55-update evidence on exact Example B. Final losses are GD `.349461`,
  normalized momentum `.000136190`, RMSProp `1.7248e-08`, and Adam `.062601`,
  so the displayed order is
  RMSProp $\ll$ momentum $<$ Adam $<$ GD. The method-specific settings remain
  visible: GD $\eta=.019$; normalized momentum
  $\eta=.095,\beta=.8$ (restricted PyTorch `lr=.019`); RMSProp
  $\eta=.08,\rho=.9$; and Adam $\eta=.10,\beta=(.9,.999)$. Ask students to
  explain each curve from the state its method stores, then state the limit of
  the comparison: these selected rates and this controlled ravine illustrate
  mechanisms; they do not establish a universal optimizer ranking.
- **[SHOULD-COVER] Page 66, “PyTorch exposes exp_avg and exp_avg_sq”.** Map
  `exp_avg` and `exp_avg_sq` to the two raw memories and inspect their correction
  clock. Do not introduce optimizer variants here.
- **[CORE] Page 67, “Choose state to match the failure”.** Require a symptom
  before a method: no state for a cheap baseline, direction memory for rapid
  alternation, scale memory for unequal coordinate magnitudes, and both only
  when both diagnoses are supported. State what optimizer memory cannot repair:
  a wrong objective, broken data, or a poor model.
- **[CORE] Page 68, “Notebook 03: inspect every adaptive state”.** Run the
  RMSProp/Adam checkpoint below. Students should reproduce the analytic first
  update, inspect every carried state, and then read the computed paths.

### 66–70 min · Pages 69–74 · schedule and its clock

- **[CORE] Page 69, “Let the step size change with training.”** Use the
  separator only to pose the conflict: early training needs travel, while late
  training needs careful refinement.
- **[CORE] Page 70, “One schedule has two jobs: travel early, refine late”.**
  Teach the computed visual as two predictions, not a result table. First show
  fixed large $\eta=.10$ and fixed small $\eta=.02$ and ask, “Which reaches a
  full-loss gap below $.5$ first?” Reveal `8` updates for the large rate versus
  `56` for the small rate; inverse-time decay
  $\eta_t=.10/(1+.02t)$ also takes `8` because it starts at $.10$. Then ask,
  “Which choice should refine most tightly late in training?” Over the final
  100 updates, reveal mean full-loss gaps `.0040996` (large), `.0012871`
  (small), and `.0002014` (decay). The corresponding final-99-transition path
  lengths are `3.79448`, `.739121`, and `.520387`: fixed $.10$ keeps roaming,
  fixed $.02$ refines more gently, and the decayed rate is tightest here. All
  runs use Example A and the identical 360 sampled indices from seed 12. The
  lesson is that decay gets both displayed behaviors—a large initial rate and a
  smaller late rate—not that it universally beats every fixed schedule.
- **[CORE] Page 71, “Decay shrinks late updates, so the path settles”.** Read
  the visual strictly left to right as a causal chain. *What we choose:* fixed
  $\eta=.10$ versus $\eta_t=.10/(1+.02t)$, which falls from `.100` to
  `.0122249` over updates 0–359. Before revealing the middle panel, ask,
  “What should change in the distance the parameters actually move late in
  training?” *What it changes:* over the final 99 transitions, the summed
  actual distances $\sum_t\lVert\Delta\boldsymbol\theta_t\rVert$ are
  `3.79447982` (fixed) and `.520387237` (decay); the fixed distance is
  `7.29165×` the decayed distance. *What we see:* connect that measured
  reduction to the contour zoom:
  the fixed-rate path keeps roaming while the decayed path settles more tightly
  near the least-squares optimum. Both runs share Example A,
  $\boldsymbol\theta_0$, seed 12, and the identical 360 sampled indices; only
  the update-scale rule is changed. The resulting parameters and gradients can
  then differ. This is controlled mechanism evidence, not a schedule benchmark.
- **[CORE] Page 72, “One simple schedule is enough to understand the idea.”**
  Derive the inverse-time form after students have seen its effect. Keep the
  clock on optimizer updates.
- **[CORE] Page 73, “PyTorch: implement the same update-clock schedule.”**
  `optimizer.step()` uses $\eta_t$; `scheduler.step()` prepares
  $\eta_{t+1}$. Under accumulation, advance only after an optimizer update.
- **[SHOULD-COVER] Page 74, “Different schedules change the shape, not the
  clock.”** Inverse-time, exponential, and cosine are shapes; the event counted
  by $t$ must still be stated.

### 70–76 min · Pages 75–83 · controlled XOR implementation bridge

- **[CORE] Pages 75–77, “Optimizers on the XOR network,” “Example C: return
  to Lecture 2's XOR task,” and “A controlled
  mechanism demo—not a leaderboard.”** Fix the $2\rightarrow4$ tanh
  $\rightarrow1$-logit network, BCE loss, initialization, minibatch sequence,
  and 400-update budget. Only the optimizer changes.
- **[CORE] Pages 78 and 80, “Same initialization, same 400 minibatches” and
  “The PyTorch loop changes in one line.”** Use seed 2026 and shared size-2
  batches. Rates are SGD $.25$, momentum SGD $.12$ with $\beta=.9$, RMSProp $.03$
  with $\rho=.9$, and Adam $.03$ with $\beta=(.9,.999)$.
- **[SHOULD-COVER] Page 79, “Same 100% training accuracy—different learned
  boundaries.”** Read the boundaries as different fitted functions, not
  generalization evidence.
- **[CORE] Pages 81–82, “Checkpoint: what does the XOR result establish?” and
  “Answer: the implementation bridge is the evidence.”** Answer C: each stored
  state fits the same backward-pass loop.
- **[CORE] Page 83, “Notebook 04 makes every control explicit.”** Run the
  90-second shared-control checkpoint below.

### 76–79 min · Pages 84–87 · keep constrained models valid

- **[CORE] Page 84, “Constrained parameters: keep every forward pass valid”.**
  Use the separator only to change the question: optimization chooses a move,
  but the model decides which parameter values are legal.
- **[CORE] Page 85, “A valid optimizer step can produce an invalid model”.**
  Work the Gaussian-scale counterexample before offering a repair. A scale must
  satisfy $s>0$, yet from $s_0=.10$, gradient $10$, and $\eta=.02$, the ordinary
  proposal is $\tilde s_1=.10-.02(10)=-.10$. The subtraction is a valid
  optimizer operation; the proposed Gaussian is invalid. Ask students where
  validity should be enforced: in the coordinates or after the proposal.
- **[CORE] Page 86, “Reparameterize: optimize a raw value”.** Optimize an
  unrestricted raw parameter and transform it inside every forward pass:
  `softplus(raw_scale)+s_min` for a positive scale, `sigmoid(raw_logit)` for an
  interior probability, and `softmax(raw_logits)` for interior simplex weights.
  Autograd applies the chain rule back to the raw value. Finite raw values give
  smooth interiors, not exact boundaries; saturation can make raw-coordinate
  gradients small. Keep the modelling floor $s_{\min}$ distinct from
  $\epsilon_{\mathrm{opt}}$, and read the minimal PyTorch code as
  *raw parameter → transform → valid forward*.
- **[CORE] Page 87, “Project when the boundary itself is valid”.** Use
  projection only when a boundary such as zero is genuinely permitted. First
  form the optimizer proposal
  $\tilde{\boldsymbol\theta}_{t+1}=\boldsymbol\theta_t+
  \Delta\boldsymbol\theta_t$, then apply
  $\boldsymbol\theta_{t+1}=\Pi_{\mathcal C}
  (\tilde{\boldsymbol\theta}_{t+1})$; `clamp_(min=0)` is the nonnegative scalar
  case. Projection can reach the boundary but is nonsmooth, and the clamp does
  not clear or rotate momentum/Adam state. Do not clamp a Gaussian scale to zero
  when zero itself is invalid.

### 79–80 min · Pages 88–89 · choose, compress, hand off

- **[CORE] Page 88, “Choose the repair that matches the symptom”.** Use the
  separator as the final retrieval prompt: name the observed failure before
  naming an optimizer or schedule.
- **[CORE] Page 89, “Optimization in one view”.** Reconstruct the six decisions
  aloud: examples per gradient, direction memory, coordinate scale, step size
  over time, valid parameters, and fair evidence. End on the operating rule:
  *start simple → diagnose one failure → add one mechanism → measure
  again*. This is the sole final slide; do not append an optimizer catalogue or
  reference-page tour.

## 110-minute extended route through the same 89-page handout

The extended route keeps every **[CORE]** item, teaches all
**[SHOULD-COVER]** items live, and uses the additional 30 minutes for fuller
derivations and notebook evidence. The timings below are cumulative and total
exactly 110 minutes.

### 0–5 min · Pages 2–4 · bridge, outline, and three anchors

Use the canonical opening unchanged. On Page 4, ask students to predict which
example will answer sampling, geometry, and implementation questions.

### 5–18 min · Pages 5–16 · define the clocks and derive all three data-per-update cases

Work Pages 7–9 line by line and establish the clock vocabulary on Page 12. On
Page 13 define what is random and state that the update follows $\ell_2$, not
yet the average. On Page 14 contrast the highlighted sampled proposal with the
three alternatives, then use Page 15 to total the four loss changes. Calculate
the Page 16 two-example gradient. Use Page 10 to connect parameter space and
prediction space.

### 18–28 min · Pages 17–19 · averaging intuition, systems caveat, and PyTorch

Use Page 17 only for the fixed-parameter averaging intuition: directions fan
out, then cluster, then collapse to the exact full-data direction. Use Page 18's
qualitative table to compare update frequency, noise, and vectorization without
turning the qualitative comparison into a wall-clock claim. On Page 19, add gradient
accumulation only after the Page 12 epoch/iteration/update definitions are
secure.

### 28–41 min · Pages 20–30 · estimator geometry, theorem, and Notebook 00

Use Pages 21–25 to move from rows to individual loss surfaces, update arrows,
minibatch surfaces, and the arithmetic mean of update vectors. Only then define
the estimator generically on Page 26:
$\mathbb E[f(X)]=\sum_x p(x)f(x)$, with unbiasedness defined relative to a
target. On Page 27, substitute $X=I_t$, $p(i)=1/N$, and
$f(i)=\nabla_{\boldsymbol\theta}\ell_i(\boldsymbol\theta_t)$; define
$\hat{\boldsymbol g}_t=f(I_t)=
\nabla_{\boldsymbol\theta}\ell_{I_t}(\boldsymbol\theta_t)$ before deriving the
uniform-sampling result.
Read the qualitative spread diagram on Page 28 and revisit the uphill sampled
step on Page 29. Spend three minutes at Page 30 executing Notebook 00;
keep nonuniform sampling, exact subset spreads, and covariance variants there as
optional extensions.

### 41–53 min · Pages 31–36 · define the ravine and its learning-rate trade-off

Page 31 closes the sampling question: the full-batch gradient is exact, yet one
global learning rate can still make poor progress. On Page 32, read the deliberate
$\pm1$ versus $\pm10$ feature scales and stay in the original coordinates with
$\boldsymbol\theta^\star=(1,-.5)$ and
$\boldsymbol\theta_0=(-1.4,.35)$. On Page 33, derive
$\mathcal L(\boldsymbol\theta)=
\tfrac12(\theta_1-1)^2+50(\theta_2+.5)^2$ and
$\nabla_{\boldsymbol\theta}\mathcal L=
(\theta_1-1,100(\theta_2+.5))$, then define curvature in plain language as how
quickly the slope changes as a parameter moves.

Page 34 supplies the geometry vocabulary before any later use: a *ravine* is a
long, narrow low-loss valley; $\theta_1$ is the shallow parameter direction,
while $\theta_2$ is the steep parameter direction. On Page 35, distinguish one
*overshoot* from repeated oscillation while computing the exact $.019$ update.
Page 36 compares the computed $.005$ and $.019$ paths directly: calming the
steep $\theta_2$ move also slows progress in the shallow $\theta_1$ direction.
Define *oscillation* as repeated side-to-side crossings under the shared update
budget.

### 53–68 min · Pages 37–51 · noisy temperature → EWMA → momentum and Notebook 02

Page 37 is the section question. On Page 38, read the seeded noisy temperature
observations before any formula and define memory as one running summary carried
between observations. On Page 39, define $\beta$ as the fraction of the previous
summary retained and predict why $\beta=.5$ reacts quickly but stays noisy.
Pages 40–41 change only $\beta$: $.9$ is a useful noise/lag compromise on this
signal, while $.98$ is very smooth but responds too slowly. Students should
articulate the trade-off before seeing the recurrence.

Page 42 finally writes
$m_t=\beta m_{t-1}+(1-\beta)x_t$ and works
$.9(30)+.1(36)=30.6$. Page 43 gets the full unrolling: substitute twice, identify
the weight $(1-\beta)\beta^k$ on $x_{t-k}$, and read the concrete $.9$ sequence.
Keep these pages focused on smoothing; Adam introduces its initialization
correction later.

On Page 44, transfer the idea to the current gradient components: the first
component keeps its sign and reinforces, whereas the second component changes
sign and partly cancels. Page 45 performs only the zero-memory first update.
Page 46 then
remeasures at $\boldsymbol\theta_1=(-1.3544,-1.265)$ and computes reinforcement
and cancellation component by component before naming
*measure → remember → move*. The smaller second-component memory is specific
computed evidence, not a promise of exact cancellation. Page 47 turns the same
two updates into a graphical three-panel explanation before the longer-run
comparison.

Use Page 48's actual 55-gradient paths to inspect
geometry, then Page 49's full-loss trace to judge success: momentum finishes at
approximately `.00013619` versus GD `.349461` and remains below $.1$ after
update 26. These are controlled Example B results, not a general superiority
claim. On Page 50, trace the from-scratch memory tensor; on Page 51, map
normalized $\eta=.095,\beta=.8$ to restricted PyTorch `lr=.019`, then spend
three minutes in Notebook 02 comparing
$\beta=.5,.9,.98$, reproducing both momentum updates, and inspecting the real
`momentum_buffer` and path agreement.

### 68–71 min · Pages 52–53 · test the preprocessing counterfactual

Only now use Pages 52–53 to answer the preprocessing objection. Standardizing
$x_2$ as $z_2=x_2/10$ removes the deliberately created $100{:}1$ imbalance, so
plain GD no longer needs momentum for this particular linear feature-scaling
failure. Then state why momentum is still relevant: deep networks can create
coupled, changing parameter-space geometry, and minibatch gradients fluctuate
even after input normalization. Preprocess poorly scaled inputs first; use
optimizer memory for the remaining successive-gradient behavior. Page 54 then
deliberately returns to the original unscaled ravine as a controlled mechanism
experiment; do not present that reset as a preprocessing recommendation.

### 71–85 min · Pages 54–68 · RMSProp, Adam, and Notebook 03

Use Pages 54–55 to restate the failure before naming RMSProp: one global rate
acts on an exact starting gradient whose coordinate magnitudes differ by
$35.4\times$. On Page 56, have students propose a separate recent-magnitude
“ruler” for each coordinate and read the intuition visual before Page 57's
elementwise recurrence.

Work the complete Example B chain
$\boldsymbol\theta_0\to\boldsymbol g_0\to\boldsymbol s_1
\to\boldsymbol\theta_1$ on
Page 58, including the $\sqrt{10}$ zero-start inflation. Then read Page 59's
actual 40-update ravine path as evidence that this repair straightens this
computed trajectory; do not present the method-specific rate as a fair
leaderboard. Inspect `square_avg` on Page 60.

Page 61 poses Adam only after both ingredients are understood. On Page 62,
students must state the distinct failures: momentum remembers signed direction
but not coordinate scale; RMSProp remembers coordinate scale but not signed
direction. Page 63 introduces the two raw memories, and Page 64 derives the
accumulated-mass correction for both zero-initialized memories directly. With
$\eta=.10$ and $\epsilon_{\mathrm{opt}}$ omitted only for this arithmetic, the
first corrected Adam update is $\Delta\boldsymbol\theta_0=(.10,-.10)$ and
$\boldsymbol\theta_1=(-1.30,.25)$.

Use Page 65's shared 55-update Example B evidence to explain all four paths from
the state each method stores. Read final losses
`.349461, .000136190, 1.7248e-08, .062601` in method order
GD, normalized momentum, RMSProp, Adam; the resulting order is
RMSProp $\ll$ momentum $<$ Adam $<$ GD. Use the method-specific controls shown
in the figure—especially normalized momentum $\eta=.095,\beta=.8$—and say
explicitly that this controlled ravine illustrates mechanisms rather than
ranking optimizers universally. Inspect `exp_avg` and `exp_avg_sq` on
Page 66, use Page 67 to choose state only after a diagnosed failure, and spend
three minutes at Page 68 reproducing the analytic states and framework checks
in Notebook 03. The full progression is therefore problem → intuition →
mathematics → worked update → computed evidence.

### 85–91 min · Pages 69–74 · decay and explicit schedule clock

On Page 70, ask the early-travel question before revealing `8, 56, 8` updates to
reach a full-loss gap below $.5$ for fixed $.10$, fixed $.02$, and decay. Then
ask the late-refinement question before revealing final-100 mean gaps
`.0040996, .0012871, .0002014`. Use the final-99 path lengths
`3.79448, .739121, .520387` only as supporting evidence for roaming versus
tighter refinement. On Page 71, reveal the chain in order: the selected decay
falls from `.100` to `.0122249`; the final-99 summed update distance falls from
`3.79447982` to `.520387237` (the fixed path travels `7.29165×` as far); and the resulting late path
settles more tightly in the contour zoom. Ask students to predict the middle
panel before revealing it. The two runs share Example A, initialization, seed
12, and all 360 sampled indices, but their later parameters and gradients can
differ. Derive the inverse-time form on Page 72 and trace the exact
`optimizer.step(); scheduler.step()` boundary on Page 73. Page 74 is a
schedule-shape comparison, not optimizer-performance evidence.

### 91–100 min · Pages 75–83 · controlled XOR lab

State all controls on Pages 76–78, read the different Page 79 boundaries, and
map the Page 80 optimizer constructors to the shared loop. Require the Page 81
vote and Page 82 explanation. At Page 83, run three minutes of Notebook 04:
verify the shared first four minibatches, all four 100% training accuracies, and
one decision-boundary plot.

### 100–106 min · Pages 84–87 · valid forwards by construction or projection

Page 84 changes the question from *how to move* to *which values are legal*.
On Page 85, calculate the Gaussian-scale proposal aloud:
$s_0=.10$, $\nabla_s\mathcal L=10$, and $\eta=.02$ produce
$\tilde s_1=-.10$. Require students to say why the optimizer arithmetic is
valid but the model is not.

On Page 86, trace *raw parameter → smooth transform → valid forward* for
softplus, sigmoid, and softmax. State that these maps represent mathematical
interiors for finite raw values and that autograd supplies the raw-coordinate
chain rule; extreme finite-precision logits may still saturate numerically. On
Page 87, use *proposal → projection* only when the boundary is part of the
feasible set. Read the `clamp_` line, then state the stored-state caveat:
projection changes the parameter but does not reconcile momentum or Adam
buffers. Return to Notebook 03 at Page 87 to compare smooth positivity with an
exact projected boundary and verify feasibility after every update.

### 106–110 min · Pages 88–89 · retrieve the diagnosis, then synthesize

On Page 88, ask pairs to name one observed symptom and the smallest matching
repair without looking back. On Page 89, have the room rebuild all six cards:
data per gradient, direction memory, coordinate scale, schedule, constraints,
and controlled evidence. Finish by applying
*start simple → diagnose → add one mechanism → measure again* to a new
training symptom. There is no optional-extension or references block in the
handout tail; keep named extensions in the guide or follow-along resources.

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
3. **Example B geometry and preprocessing · Pages 32–36 and 52–53:**
   $\boldsymbol\theta^\star=(1,-.5)$,
   $\boldsymbol\theta_0=(-1.4,.35)$,
   $\mathcal L(\boldsymbol\theta)=
   \tfrac12(\theta_1-1)^2+50(\theta_2+.5)^2$, initial loss
   $\mathcal L(\boldsymbol\theta_0)=39.005$, and
   $\mathbf X^\top\mathbf X/N=\operatorname{diag}(1,100)$. The GD prediction
   controls shown in the deck are $\eta=.005,.019$; Page 52 later standardizes
   with $z_2=x_2/10$, producing curvature $(1,1)$ in the rescaled coordinates.
4. **EWMA and momentum · Pages 37–51:** Page 38 introduces the fixed noisy
   signal; Pages 39–41 compare $\beta=.5,.9,.98$; Page 42 introduces the EWMA
   recurrence; Page 43 unrolls it; Pages 44–47 transfer that memory to gradients,
   work the first two ravine updates, and visualize their reinforcement and
   cancellation. Normalized momentum uses $\beta=.8$
   and $\eta=.095$ for 55
   updates; the equivalent restricted PyTorch setting is `lr=.019`, so the
   first parameter displacement matches plain GD. The coupled sequence gives
   $\boldsymbol m_1=(-.48,17)$ and
   $\boldsymbol m_2=(-.85488,-1.7)$ after remeasuring at
   $\boldsymbol\theta_1=(-1.3544,-1.265)$. The second gradient components nearly
   cancel rather than cancelling exactly. Page 49 reports final loss
   `.000136190` and shows that momentum remains below $\mathcal L=.1$ from
   update 26 onward. Notebook 02 checks the PyTorch conversion and stored state.
5. **RMSProp and Adam · Pages 54–68:** Page 54 deliberately returns to the
   original unscaled ravine to isolate the optimizer mechanism; it is not a
   preprocessing recommendation. Page 55 quantifies the $35.4\times$
   coordinate-scale mismatch before Page 56 introduces a separate ruler per
   coordinate. RMSProp uses
   $\eta=.08,\rho=.9,\epsilon_{\mathrm{opt}}=10^{-8}$; Page 59 shows 40
   updates. The four-method Page 65 control uses 55 updates: GD $.019$;
   normalized momentum
   $.095,\beta=.8$ (restricted PyTorch `lr=.019`); RMSProp $.08,\rho=.9$; Adam
   $.10,\beta=(.9,.999)$. Final losses are
   `.349461, .000136190, 1.7248e-08, .062601` in method order GD, momentum,
   RMSProp, Adam, giving the order
   RMSProp $\ll$ momentum $<$ Adam $<$ GD. Notebook 03 uses these same four
   controls. These
   method-specific rates support a mechanism comparison, not a universal rank.
6. **Schedule · Pages 70–74:** seed 12; the identical stream of 360 sampled
   indices; fixed $\eta=.10$, fixed $\eta=.02$, and
   $\eta_t=.10/(1+.02t)$. Updates to a full-loss gap below $.5$ are
   `8, 56, 8`; final-100 mean gaps are
   `.00409961145, .00128710162, .000201402373`; and final-99-transition path
   lengths are `3.79448, .739121, .520387`, all in large/small/decay order.
   Page 71 isolates fixed $.10$ versus decay: the rate ends at `.0122249`, and
   fixed summed final-99 update distance is `7.29165×` the decayed distance.
   The candidate and Notebook 03 advance $t$ once per optimizer update.
7. **XOR · Pages 76–83:** seed 2026; $2\rightarrow4$ tanh
   $\rightarrow1$ logit; one shared initialization; one shared sequence of 400
   size-2 minibatches. Settings are SGD $.25$; momentum SGD $.12,\beta=.9$;
   RMSProp $.03,\rho=.9,\epsilon_{\mathrm{opt}}=10^{-8}$; Adam
   $.03,\beta=(.9,.999),\epsilon_{\mathrm{opt}}=10^{-8}$. Final full-data BCE
   values are
   $.09424,.006151,.005770,.03998$ respectively; every run reaches 100%
   training accuracy. These are reproduction checks, not a ranking.
8. **Constraints · Pages 85–87:** Page 85 uses a Gaussian scale
   $s_0=.10$, gradient $10$, and $\eta=.02$, giving the invalid unconstrained
   proposal $\tilde s_1=-.10$. Page 86's structural snippet maps with
   `softplus(raw_scale)+1e-6`, `sigmoid(raw_probability)`, and
   `softmax(simplex_logits)`; `1e-6` is the modelling floor $s_{\min}$, not
   $\epsilon_{\mathrm{opt}}$. Page 87 forms the optimizer proposal before a
   nonnegative `clamp_` and leaves stored optimizer state untouched. Notebook 03
   checks one raw SGD step from $s=.10$ to `.098204`, the exact sigmoid/softmax
   values, and a 35-step boundary comparison at $\eta=.25$: projection ends at
   `0`, while softplus ends at `.109119`.

## Exact notebook checkpoints

### Notebook 00 · `00_full_batch_sgd_minibatch.ipynb`

Colab: [`notebooks/L05/00_full_batch_sgd_minibatch.ipynb`](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L05/00_full_batch_sgd_minibatch.ipynb)

Use at Page 30 for 90 seconds in the canonical route or three minutes in the
extended route.

1. **“Start with the full-data loss and gradient.”** Expect predictions
   `[3, 2, 1, 0]`, residuals
   `[4, 1, -2, -6]`, full gradient
   $\boldsymbol g_0=$ `[-0.75, -4.5]`, new parameters
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
   means equal $\boldsymbol g_0$ and the RMS distances contract
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

Use at Page 51 for 90 seconds in the canonical route or three minutes in the
extended route. In deck order, run the EWMA and momentum cells first; return to
the notebook's feature-standardization checkpoint when the deck reaches Pages
52–53.

1. **“The ravine comes directly from the four rows.”** Read the executed table
   before the matrix summary. At $\boldsymbol\theta_0=(-1.4,.35)$ the predictions
   are `[-2.1, 4.9, -4.9, 2.1]`, the residuals are
   `[-6.1, 10.9, -10.9, 6.1]`. Verify
   $\mathbf X(\boldsymbol\theta_0-\boldsymbol\theta^\star)$ reproduces those
   residuals,
   $\mathbf X^\top\mathbf X/N=\operatorname{diag}(1,100)$,
   $\boldsymbol g_0=$ `[-2.4, 85]`, and the full loss is `39.005`.
2. **“Build EWMA from three views of the same noisy signal.”** On the fixed-seed
   120-day temperature sequence, compare $\beta=.5,.9,.98$ with all smoothers
   initialized at the first reading. Predict responsiveness and lag before each
   reveal, then verify the worked $.9(30)+.1(36)=30.6$ update and the first
   exponential weights $(1-\beta),(1-\beta)\beta,(1-\beta)\beta^2$. The
   [browser explorer](https://nipunbatra.github.io/dl-teaching/interactives/ewma-explorer.html)
   reproduces those same observations, exposes the two numerical terms in each
   selected update, and then applies the same memory to the computed ravine.
3. **“Couple the momentum state to the path it actually creates.”** Use
   $\beta=.8$ and EWMA-form $\eta=.095$; explain later why
   $(1-\beta)\eta=.019$ matches the first displacement of plain GD and
   restricted PyTorch momentum. Check that every next gradient is evaluated at
   the parameters produced by the preceding memory. The first two states are
   $\boldsymbol m_1=$ `[-.48, 17.]` and
   $\boldsymbol m_2=$ `[-.85488, -1.7]`, with
   $\boldsymbol\theta_1=$ `[-1.3544, -1.265]` and
   $\boldsymbol\theta_2=$ `[-1.2731864, -1.1035]`. Read each row as
   $\boldsymbol\theta_t\to\boldsymbol g_t\to\boldsymbol m_{t+1}\to
   \boldsymbol\theta_{t+1}$.
4. **“Judge momentum on the full objective.”** After 55 exact gradient
   evaluations, expect final losses `0.349461` for GD at $.019$ and
   `0.000136190` for normalized momentum at $.095,\beta=.8$. Momentum remains
   below $\mathcal L=.1$ from update 26 onward. Plain GD is monotone on this
   quadratic at the displayed rate; momentum's temporary loss increases come
   from stored motion and overshoot, not minibatch noise. Do not replace this
   objective evidence with a crossing count or a requirement that every update
   lower the loss.
5. **“Why PyTorch's `lr` looks different.”** Define PyTorch's unnormalized
   `momentum_buffer`, then use
   `lr_torch = lr_EWMA * (1 - beta) = .019`. Verify the full parameter path and
   normalized buffer to numerical precision, and inspect `momentum_buffer`
   rather than comparing only endpoints.
6. **“Now test the preprocessing counterfactual.”** For the identical Example B
   start, predict whether $\theta_2$ crosses its optimum value $-.5$ under
   $\eta=.005$ and $.019$, then compare the computed paths. Standardize with
   $z_2=x_2/10$ and verify that the curvature changes from $(1,100)$ to $(1,1)$;
   remember that the corresponding learned coefficient is
   $\phi_2=10\theta_2$. This establishes why preprocessing fixes this particular
   ravine without claiming that every deep-network loss becomes round.

### Notebook 03 · `03_adaptive_optimizers_constraints.ipynb`

Colab: [`notebooks/L05/03_adaptive_optimizers_constraints.ipynb`](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L05/03_adaptive_optimizers_constraints.ipynb)

Use at Page 68 for adaptive state, Page 73 for the schedule clock, and Page 87
for constraints.

1. **RMSProp first step.** Verify
   $\boldsymbol s_1=$ `[0.576, 722.5]`,
   $\boldsymbol g_0/\sqrt{\boldsymbol s_1}=$ `[-3.1623, 3.1623]`, and PyTorch
   path agreement at `1.11e-16`. The scale-history error is `2.27e-13`.
2. **Adam first step.** With $\eta=.10$ and $\epsilon_{\mathrm{opt}}$ omitted
   only for the displayed arithmetic, predict the sign-sized update
   $\Delta\boldsymbol\theta_0=$ `[0.10, -0.10]`, then obtain
   $\boldsymbol\theta_1=$ `[-1.3, 0.25]`; the from-scratch and PyTorch paths agree
   at approximately `8.33e-17`. Trace
   $\hat{\boldsymbol g}_0\to(\boldsymbol m_1,\boldsymbol v_1)$ and the correction
   denominators $1-\beta_1^1$, $1-\beta_2^1$. Inspect both `exp_avg` and
   `exp_avg_sq`.
3. **Mechanism comparison.** Read the four trajectories only after naming the
   state each method stores. The notebook and Page 65 deck panel share 55
   updates and report final losses `.349461, .000136190, 1.7248e-08, .062601`
   in method order GD, normalized momentum, RMSProp, Adam, so
   RMSProp $\ll$ momentum $<$ Adam $<$ GD on this run. The rates are
   method-specific, including normalized momentum $.095,\beta=.8$ (restricted
   PyTorch `lr=.019`), so neither source supports a universal ranking.
4. **Decay.** With the same 360-index stochastic stream, predict and verify the
   early threshold counts `8, 56, 8` for fixed $.10$, fixed $.02$, and
   $.10/(1+.02t)$. Then verify final-100 mean full-loss gaps
   `.00409961145, .00128710162, .000201402373` and final-99-transition path
   lengths `3.79448, .739121, .520387`. For the Page 71 fixed-versus-decay
   chain, verify the final decay rate `.0122249`, exact late distances
   `3.79447982` and `.520387237`, and ratio `7.29165`. Check the first eight
   scheduler rates and that the clock advances after `optimizer.step()`.
5. **Constraint maps and boundary example.** Reproduce the Page 85 Gaussian
   checkpoint $y=\mu$, $\mathcal L(s)=\log s$: from $s_0=.10$, gradient `10`,
   and $\eta=.02$, obtain the invalid proposal `-.10`. Initialize the softplus
   raw parameter to represent $s=.10$; one raw SGD step at `.02` yields
   `s=.098204`, still positive. Also verify
   `sigmoid(.4)=.598688` and
   `softmax([.2,-.1,.7])=[.295025,.218560,.486415]`.
6. **Projection versus a smooth interior.** For
   $\min_{s\ge0}\tfrac12(s+1)^2$ from $s=2$, run 35 SGD updates at $\eta=.25$:
   projection finishes at `0`, while the softplus parameterization finishes at
   `.109119`. Name the projected sequence explicitly: optimizer proposal
   $\tilde{\boldsymbol\theta}_{t+1}$ first, then
   $\boldsymbol\theta_{t+1}=\Pi_{\mathcal C}
   (\tilde{\boldsymbol\theta}_{t+1})$. Clamp only after `optimizer.step()`;
   projection does not clear or rotate stored momentum/Adam state.

### Notebook 04 · `04_xor_optimizer_lab.ipynb`

Colab: [`notebooks/L05/04_xor_optimizer_lab.ipynb`](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L05/04_xor_optimizer_lab.ipynb)

Use at Page 83 for 90 seconds in the canonical route or three minutes in the
extended route.

1. **Shared controls.** Verify seed `2026`, one copied initial state, 400 size-2
   minibatches, and first four batches `[[1,2],[3,0],[3,0],[1,2]]`.
2. **Controlled training runs.** Confirm the four displayed optimizer
   constructors and the final losses `9.424e-2, 6.151e-3, 5.770e-3, 3.998e-2`.
   All four runs reach 100% training accuracy.
3. **Final decision boundaries.** Check that every boundary is evaluated on the
   same grid. Different boundaries at identical training accuracy show different
   fitted functions, not comparative generalization.
4. **Read the evidence carefully.** The lab establishes that the same backward
   pass can feed four update mechanisms under controlled conditions. It does not
   establish that Adam, RMSProp, or momentum wins on another model, dataset,
   budget, or validation metric.

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
4. **[SHOULD-COVER] EWMA:** the concrete
   $.9(30)+.1(36)=30.6$ blend followed by the three-term exponential expansion.
5. **[SHOULD-COVER] Momentum:** under the deck control
   $\beta=.8,\eta=.095$, compute
   $\boldsymbol m_1=(-.48,17)$ and
   $\boldsymbol m_2=(-.85488,-1.7)$ after remeasuring
   $\boldsymbol g_1$ at $\boldsymbol\theta_1=(-1.3544,-1.265)$.
6. **[SHOULD-COVER] RMSProp:** one complete elementwise scale calculation,
   $\boldsymbol\theta_0\to\boldsymbol g_0\to\boldsymbol s_1
   \to\boldsymbol\theta_1$, including the
   $\sqrt{10}$ cold-start magnitude.
7. **[SHOULD-COVER] Adam:** the bias-corrected first update and its restricted
   sign-sized interpretation, with $\boldsymbol m_{t+1}$ and
   $\boldsymbol v_{t+1}$ corrected by exponents $t+1$.
8. **[OPTIONAL] Sampler covariance:** independent draws versus the
   finite-population correction.
9. **[OPTIONAL] Constraint chain rule:** one softplus raw-coordinate gradient.

## Common misconceptions and repairs

- **“An exact gradient is the optimal update.”** It is the locally steepest
  direction under the chosen coordinates; one global rate may still zigzag in a
  ravine.
- **“Feature standardization makes momentum unnecessary.”** It removes the
  deliberately created $100{:}1$ input-scale imbalance in Example B. It does not
  remove all coupled, changing parameter-space geometry or minibatch
  fluctuation in a deep network; normalization and momentum address different
  parts of the problem.
- **“A larger $\beta$ is always better.”** It smooths more history but responds
  more slowly when the underlying direction changes. The $.5,.9,.98$
  temperature sequence makes that lag visible before gradients are involved.
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
- **“Adam's moving-average correction makes SGD unbiased.”** It repairs the
  artificial early shrinkage caused by zero-initialized moment states. Gradient-
  estimator unbiasedness is a separate claim about the sampling distribution.
- **“PyTorch momentum stores the normalized EWMA from the slide.”** It stores
  the unnormalized buffer $\boldsymbol v$; the relation
  $\boldsymbol m=(1-\beta)\boldsymbol v$ explains why the displayed PyTorch
  learning rate is $(1-\beta)\eta$.
- **“RMSProp bias-corrects its square average like Adam.”** Standard PyTorch
  RMSProp does not. With $\rho=.9$ and zero state, the first normalized
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
- **“A scheduler naturally counts epochs.”** Its clock is whatever boundary the
  code uses. State whether it counts epochs or optimizer updates.
- **“Sigmoid represents the closed interval $[0,1]$ exactly.”** Finite logits
  mathematically represent $(0,1)$; exact endpoints may need projection or
  another model. In finite precision, extreme logits can still round to 0 or 1,
  so prefer logits-based losses.
- **“Clamping after Adam is harmless.”** The parameter is projected but stored
  moments are not automatically reconciled with that projection.

## If the room is quiet: prediction prompts

1. Which one-sample update raises the full line-fit loss?
2. If $B$ doubles under independent sampling, what happens to covariance?
3. At $\eta=.019$, why does the steep $\theta_2$ component alternate signs?
4. Which $\beta$ gives more smoothing, and what does it sacrifice?
5. Why do the first two $\theta_2$ gradient contributions nearly cancel in the
   $\beta=.8$ momentum state while the $\theta_1$ contributions reinforce?
6. Why does replacing $x_2$ by $z_2=x_2/10$ remove this example's $100{:}1$
   curvature imbalance, and what does it not repair in a deep network?
7. Why does RMSProp square before averaging?
8. Which Adam state remembers sign, and which remembers scale?
9. Does decay change the sampled data, optimizer state definition, or only the
   applied step scale in the displayed experiment?
10. Which transformation would you use for a variance, a probability, and a
    mixture-weight vector? Which can reach its boundary exactly?

## Follow-along after the 80-minute route

- In Notebook 00, compare uniform subsets with independent draws with
  replacement and state which sampler each variance claim assumes.
- In Notebook 02, inspect every normalized/PyTorch momentum state and write down
  the assumptions behind the learning-rate conversion; then verify exactly
  what the standardization counterfactual changes and leaves unchanged.
- In Notebook 03, finish the scheduler trace and both constraint experiments;
  explain mathematical interiors, finite-precision saturation, and projected
  optimizer state.
- In Notebook 04, compare all four decision boundaries while keeping the shared
  controls visible.

## Closing line

“Backprop measures a gradient. Batch size controls its noise; momentum remembers
direction; RMSProp remembers scale; Adam remembers both; the schedule controls
time; and a parameterization controls which values the model can express.”

## Primary references

- Andrew Ng, [*Improving Deep Neural Networks: Hyperparameter Tuning,
  Regularization and Optimization — Week 2*](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc),
  C2W2L01–L09—the approachable minibatch → EWMA → momentum → RMSProp → Adam →
  decay order adapted here.
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

All optimizer trajectories in the deck are generated from the displayed losses
and update rules. Treat the notebooks as deterministic mechanism checks, not
benchmark evidence.
