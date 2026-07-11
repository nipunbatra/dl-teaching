// Optimization for Deep Learning — Lecture 5 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture5/L5-optimization.typ
//   typst compile --root . --input handout=true lecture5/L5-optimization.typ
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Optimization for Deep Learning],
  subtitle: [SGD, Momentum, Adam, and Learning-Rate Schedules],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
// the update loop: gradient -> optimizer -> new parameters
#let updateloop = align(center, diagram(spacing: (23mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), $theta_t$, radius: 7mm, stroke: 0.9pt + INK)
  node((1,0), text(size: 14pt)[backprop], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 7pt, fill: rgb("#EFEEEB"))
  node((2,0), $g_t$, radius: 7mm, stroke: 0.9pt + TEAL)
  node((3,0), text(size: 14pt)[optimizer], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 7pt, fill: rgb("#FDECD6"), stroke: 0.9pt + ACC)
  node((4,0), $theta_(t+1)$, radius: 7mm, stroke: 0.9pt + ACC)
  edge((0,0),(1,0), "-|>", stroke: 0.8pt + MUTED)
  edge((1,0),(2,0), "-|>", stroke: 0.8pt + MUTED)
  edge((2,0),(3,0), "-|>", stroke: 0.8pt + MUTED)
  edge((3,0),(4,0), "-|>", stroke: 0.8pt + MUTED)
}))

#title-slide()

// ═══════════════════════════ PART I — From backprop to optimization ═══════════════════════════
= From backprop to optimization

== Where we are

Backprop gave us one thing: the *gradient* of the loss.
#pause
$ g_t = nabla_theta cal(L)(theta_t) $
#pause
But a gradient is only a *direction*. Training still has to decide:
#pause
#notebox[*How* do we turn a stream of gradients into good parameters? That is the job of the *optimizer*.]

== The spine of this lecture

#align(center, text(size: 23pt, fill: INK)[
  Backprop gives gradients; \
  *optimization decides how to use them.*
])
#pause
#v(6pt)
#updateloop
#pause
#align(center, text(size: 16pt, fill: MUTED)[each step: current params → gradient → update rule → new params])

== The optimizer is a small function

Every method in this lecture has the same skeleton:
$ theta_(t+1) = "update"(theta_t, g_t) $
#pause
They differ only in *what state they keep* and *how they scale the step*:
#pause
- plain GD keeps *nothing* — just $-eta g_t$;
#pause
- momentum keeps a *running direction*;
#pause
- Adam keeps a running *direction* and a running *magnitude*.

== The loss landscape #V

Training = descending a surface $cal(L)(theta)$ over millions of parameters.
#pause
#fig("/lecture5/figures/landscape.svg", w: 74%)
#align(center, text(size: 16pt, fill: MUTED)[the gradient points *uphill*; we step along $-nabla cal(L)$, perpendicular to contours])

== Gradient descent from a Taylor series #D

Expand the loss around the current point $theta$ for a small step $Delta$:
$ cal(L)(theta + Delta) approx cal(L)(theta) + nabla cal(L)(theta)^top Delta + 1/2 Delta^top H Delta $
#pause
Keep only the *first-order* term (small steps ⇒ curvature negligible):
$ cal(L)(theta + Delta) approx cal(L)(theta) + nabla cal(L)(theta)^top Delta $

== Steepest descent falls out #D

Minimize the linear model over a *trust region* $norm(Delta) <= r$:
$ min_(norm(Delta) <= r) thin nabla cal(L)(theta)^top Delta $
#pause
By Cauchy–Schwarz this is most negative when $Delta$ is *anti-parallel* to the gradient:
$ Delta^star = -r thin (nabla cal(L)(theta))/norm(nabla cal(L)(theta)) quad prop quad -nabla cal(L)(theta) $
#pause
#result[$theta_(t+1) = theta_t - eta thin nabla cal(L)(theta_t)$]

== What is the learning rate, really? #D

Folding the trust-region radius into a single step size gives $eta$.
#pause
Compare with the *second-order* (Newton) step that also uses curvature:
$ Delta_"Newton" = -H^(-1) nabla cal(L) $
#pause
#notebox[$eta$ plays the role of an *inverse-curvature* / trust-region size. A scalar $eta$ is a crude stand-in for $H^(-1)$ — too big overshoots sharp directions, too small crawls on flat ones.]

== Basic gradient descent

#result[$theta_(t+1) = theta_t - eta thin nabla cal(L)(theta_t)$]
#pause
- $eta$ — the *learning rate* (step size);
#pause
- $nabla cal(L)$ — the direction of steepest ascent, so $-nabla cal(L)$ descends;
#pause
- one hyperparameter, no memory — the simplest optimizer there is.

== Worked example: 1-D quadratic #D

Minimum is at $theta = 3$. Set up the loss, gradient, and starting point:
#pause
$ cal(L)(theta) = (theta - 3)^2, quad nabla cal(L) = 2(theta - 3), quad theta_0 = 0, quad eta = 0.1 $
#pause
*Step 1* — plug $theta_0 = 0$ into the update rule:
$ theta_1 = 0 - 0.1 dot 2(0 - 3) $
#pause
$ theta_1 = 0 - 0.1 dot (-6) = 0.6 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the gradient $-6$ points left of the min, so the step moves us *right*, toward $3$.])

== Worked example: 1-D quadratic (cont.) #D

Recall $theta_1 = 0.6$, still with $eta = 0.1$ on $cal(L) = (theta - 3)^2$.
#pause
*Step 2* — gradient at $0.6$ is $2(0.6 - 3) = -4.8$:
$ theta_2 = 0.6 - 0.1 dot (-4.8) = 1.08 $
#pause
*Step 3* — gradient at $1.08$ is $2(1.08 - 3) = -3.84$:
$ theta_3 = 1.08 - 0.1 dot (-3.84) = 1.464 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[$0 -> 0.6 -> 1.08 -> 1.464 -> dots -> 3$ — each step covers $80%$ of the remaining gap.])
#pause
#notebox[The distance to the minimum obeys $(theta_t - 3) -> (1 - 2 eta)(theta_t - 3)$; here $1 - 2 eta = 0.8$, a clean geometric decay.]

== The learning rate matters #V

Same curve $cal(L) = (theta - 3)^2$, four choices of $eta$:
#pause
// native: the ACTUAL GD iterates at each η (ml-plot) — the diverging case really diverges
#let _para(t) = calc.pow(t - 3.0, 2)
#let _gd1(eta) = {
  let t = 0.0
  let pts = ((t, _para(t)),)
  for _ in range(12) {
    t = t - eta * 2.0 * (t - 3.0)
    let tc = calc.max(-1.5, calc.min(7.5, t))   // clip to the frame like the curve
    pts.push((tc, _para(tc)))
  }
  pts
}
#let _bowl = range(0, 61).map(i => { let t = -1.5 + 9.0 * i / 60.0; (t, _para(t)) })
#let _cfgs = (
  (0.05, [$eta = 0.05$ · too small], TEAL),
  (0.4, [$eta = 0.4$ · good], GREEN),
  (0.9, [$eta = 0.9$ · oscillates], ACC),
  (1.05, [$eta = 1.05$ · diverges], RED),
)
#align(center, stack(dir: ltr, spacing: 4mm,
  .._cfgs.map(cf => lines((_bowl, _gd1(cf.at(0))),
    colors: (INK, cf.at(2)), markers: (false, true), y-ticks: false,
    size: (37mm, 30mm), title: cf.at(1))),
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[too small crawls · good converges fast · large oscillates · $eta > 1$ diverges])

== Interactive: optimizer race #I

#interbox(link-to: IA + "optimizer-race")[
  Drop a start point on a loss surface and watch gradient descent trace its path as you sweep the learning rate — see crawl, converge, oscillate, and diverge live.
]
#pause
Try values around the stability boundary and watch the error multiplier $1-2eta$.

== Checkpoint: learning-rate stability #Q

#mcq(
  [For $cal(L)=(theta-3)^2$, gradient descent diverges for which learning rates?],
  [$eta < 0$ only],
  [$0 < eta < 0.5$],
  [$0.5 < eta < 1$],
  [$eta > 1$],
)

== Answer: learning-rate stability #A

#mcq-answer([D], [$eta>1$], [The error is multiplied by $1-2eta$ each step. Convergence requires its magnitude to be below one; beyond $eta=1$ it grows.])

// ═══════════════════════════ PART II — Batch vs SGD ═══════════════════════════
= Full-batch vs stochastic gradient descent

== The loss is an average

Training loss is an *empirical risk* — a mean over $n$ examples:
$ cal(L)(theta) = 1/n sum_(i=1)^n ell_i (theta), quad quad nabla cal(L)(theta) = 1/n sum_(i=1)^n nabla ell_i (theta) $
#pause
#notebox[The true gradient needs a pass over the *entire dataset*. For $n = 10^6$ that is a million backprops per step.]

== Full-batch gradient descent

$ theta_(t+1) = theta_t - eta dot 1/n sum_(i=1)^n nabla ell_i (theta_t) $
#pause
#two(
  notebox[*Pro* — the exact gradient; smooth, deterministic descent.],
  alertbox[*Con* — one step costs a *full epoch*; far too slow for large $n$.],
)

== Stochastic gradient descent (SGD)

Sample one example $I ~ "Uniform"(1, dots, n)$ and step on *its* gradient:
$ g_t = nabla ell_I (theta_t), quad quad theta_(t+1) = theta_t - eta thin g_t $
#pause
#result[one example per step — up to $n times$ more updates per epoch]

== SGD is an unbiased estimator #D

Take the expectation over the random index $I$:
$ EE_I [g_t] = sum_(i=1)^n Pr(I = i) thin nabla ell_i (theta_t) = sum_(i=1)^n 1/n nabla ell_i (theta_t) $
#pause
$ EE_I [g_t] = 1/n sum_(i=1)^n nabla ell_i (theta_t) = nabla cal(L)(theta_t) $
#pause
#result[on average, SGD steps in the *true* gradient direction]

== Unbiased, but noisy

Write the stochastic gradient as the truth plus zero-mean noise:
$ g_t = nabla cal(L)(theta_t) + epsilon_t, quad EE[epsilon_t] = 0, quad "Var"(epsilon_t) > 0 $
#pause
#notebox[Each step is right *on average* but wrong *individually*. The path jitters — that noise both hurts (never fully settles) and helps (escapes bad regions).]

== Minibatch SGD — the practical middle

Average the gradient over a *batch* $B_t$ of $B$ examples:
$ g_t = 1/B sum_(i in B_t) nabla ell_i (theta_t) $
#pause
Still unbiased, $EE[g_t] = nabla cal(L)$, but the noise shrinks:
$ "Var"(g_t) approx 1/B "Var"(nabla ell_i) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[larger batch ⇒ $sqrt(B)$ times less gradient noise])

== The batch-size tradeoff

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 7pt), align: (left, left, left),
  table.header([*Batch size*], [*Gradient*], [*Behaviour*]),
  [small (1–32)], [very noisy], [cheap steps, explores, can escape saddles],
  [medium (64–512)], [moderate], [the usual sweet spot],
  [large (4k+)], [near-exact], [stable but costly; needs LR tuning],
))
#pause
#notebox[Batch size is set as much by *GPU memory and throughput* as by optimization.]

== Batch size shapes the noise #V

Same bowl, three noise levels (full / mini / tiny batch):
#pause
// native: SGD with seeded gradient noise — one loss, gradient via grad2d
#let bowl(x, y) = x * x + 3.0 * y * y
#let _batches = (([full batch], 0.0), ([minibatch], 2.2), ([tiny batch], 6.0))
#align(center, stack(dir: ltr, spacing: 6mm,
  .._batches.map(cf => contour(bowl, xlim: (-2.6, 2.6), ylim: (-2.6, 2.6),
    samples: 46, levels: 9, size: (37mm, 37mm), color: TEAL,
    paths: (sgd(grad2d(bowl), (-2.2, 2.0), lr: 0.10, noise: cf.at(1), seed: 3, steps: 40),),
    marks: ((0, 0, [min], RED),), title: cf.at(0))),
))
#align(center, text(size: 16pt, fill: MUTED)[full batch glides · minibatch jitters toward the minimum · tiny batch bounces])

== Interactive: SGD noise #I

#interbox(link-to: IA + "optimizer-race")[
  Switch between full-batch and stochastic updates and watch the trajectory go from a smooth glide to a noisy walk as the batch size shrinks.
]
#pause
Compare a smooth full-batch path with the small random deviations of minibatches.

// ═══════════════════════════ PART III — Momentum ═══════════════════════════
= Momentum

== The problem: ravines #V

Ill-conditioned loss $cal(L) = x^2 + 100 y^2$: steep across, shallow along.
#pause
// native: the loss is written ONCE — it draws the contour AND (via grad2d) the
// gradient for the ACTUAL GD iterates, so they can't disagree.
#let ravine(x, y) = x * x + 100.0 * y * y
#align(center, contour(ravine, xlim: (-2.6, 2.6), ylim: (-0.55, 0.55),
  samples: 60, levels: 10, size: (108mm, 32mm), color: TEAL,
  paths: (gd(grad2d(ravine), (-2.3, 0.45), lr: 0.0090, steps: 34),),
  marks: ((0, 0, [min], RED),),
))
#align(center, text(size: 16pt, fill: MUTED)[GD must use a *tiny* $eta$ to stay stable in $y$ — so it *zigzags* and crawls in $x$])

== The idea: keep a velocity

Accumulate past gradients into a *velocity*, then step along it:
$ v_t = beta thin v_(t-1) + g_t, quad quad theta_(t+1) = theta_t - eta thin v_t $
#pause
- $beta in [0, 1)$ — the *momentum* coefficient (typically $0.9$);
#pause
- $v_t$ is a running memory of *where we have been going*.

== Momentum as a vector sum #V

The update is a *vector addition*: the decayed velocity plus this step's gradient.
#pause
#fig("/lecture5/figures/momentum_vector.svg", w: 54%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[the long #text(fill: TEAL)[$beta v_(t-1)$] dominates; the cross-valley #text(fill: ACC)[$g_t$] is largely cancelled step to step])

== Momentum is an exponential moving average #D

Unroll the recursion $v_t = beta v_(t-1) + g_t$:
$ v_t = g_t + beta g_(t-1) + beta^2 g_(t-2) + beta^3 g_(t-3) + dots $
#pause
#notebox[A *geometrically weighted sum* of all past gradients. Recent gradients dominate; old ones fade by $beta$ each step. It *remembers the direction*.]

== Why momentum helps #V

// native: same ravine, GD vs momentum — one loss, gradient via grad2d
#let ravine(x, y) = x * x + 100.0 * y * y
#let _rav(title, path) = contour(ravine, xlim: (-2.6, 2.6), ylim: (-0.55, 0.55),
  samples: 60, levels: 10, size: (72mm, 24mm), color: TEAL,
  paths: (path,), marks: ((0, 0, [min], RED),), title: title,
)
#align(center, stack(dir: ltr, spacing: 9mm,
  _rav([plain GD], gd(grad2d(ravine), (-2.3, 0.45), lr: 0.0090, steps: 34)),
  _rav([with momentum], momentum(grad2d(ravine), (-2.3, 0.45), lr: 0.0035, steps: 34)),
))
#pause
- *oscillating* directions ($y$) → gradients flip sign → they *cancel* in $v_t$;
#pause
- *consistent* directions ($x$) → gradients agree → they *accumulate* and accelerate;
#pause
- averaging also *smooths* stochastic noise.

== Worked example: the EMA #D

Use the *normalized* form $m_t = beta m_(t-1) + (1 - beta) g_t$ with $beta = 0.9$, start $m_0 = 0$, gradients $g = (10, 8, 11)$:
#pause
Step 1 — take $10%$ of the new gradient, keep $90%$ of the old average:
$ m_1 = 0.9 dot 0 + 0.1 dot 10 = 1.0 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[one gradient of $10$ only nudges the average to $1.0$ — the memory of $m_0 = 0$ still dominates.])

== Worked example: the EMA (cont.) #D

Carry $m_1 = 1.0$ forward; $beta = 0.9$, next gradients $8$ then $11$.
#pause
*Step 2* — carry $0.9 dot 1.0$ forward, add $0.1 dot 8$:
$ m_2 = 0.9 dot 1.0 + 0.1 dot 8 = 0.9 + 0.8 = 1.7 $
#pause
*Step 3* — carry $0.9 dot 1.7$, add $0.1 dot 11$:
$ m_3 = 0.9 dot 1.7 + 0.1 dot 11 = 1.53 + 1.1 = 2.63 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[raw gradients average $~9.7$, but starting from $0$ the EMA is still climbing at $m_3 = 2.63$ — early averages are biased *low* (Adam will correct this).])

== How much does momentum remember?

The weights $beta^k$ decay geometrically; the *effective memory* is
$ 1/(1 - beta) quad "steps." $
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (center, center),
  table.header([$beta$], [effective window]),
  [$0.9$], [$~10$ steps],
  [$0.99$], [$~100$ steps],
  [$0.5$], [$~2$ steps],
))
#pause
#notebox[Higher $beta$ = longer memory = more smoothing, but slower to react to a real change of direction.]

== Interactive: momentum #I

#interbox(link-to: IA + "optimizer-race")[
  Turn momentum on and off in the ravine and sweep $beta$ — watch zigzags collapse into a smooth, accelerating glide down the valley.
]

// ═══════════════════════════ PART IV — Adaptive methods ═══════════════════════════
= Adaptive learning rates

== One global $eta$ is too blunt

Different parameters live on different scales and see different gradient sizes.
#pause
- a rarely-active feature gets *tiny, infrequent* gradients — wants a *big* step;
#pause
- a hot feature gets *large, frequent* gradients — wants a *small* step.
#pause
#result[give every coordinate its *own* effective learning rate]

== The adaptive form

Scale each coordinate $j$ by the size of *its own* recent gradients:
$ theta_(t+1, j) = theta_(t, j) - eta/(sqrt(s_(t, j)) + epsilon) thin g_(t, j) $
#pause
- $s_(t,j)$ — a running estimate of the *squared* gradient in coordinate $j$;
#pause
- $epsilon$ — a small constant ($~10^(-8)$) to avoid dividing by zero.

== RMSProp: an EMA of squared gradients

Keep an exponential average of $g^2$, *elementwise*:
$ s_t = rho thin s_(t-1) + (1 - rho) thin g_t^2 $
#pause
$sqrt(s_t)$ is the *root-mean-square* of recent gradients — hence the name *RMSProp*.
#pause
#notebox[Big recent gradients ⇒ large $sqrt(s_t)$ ⇒ *smaller* step. Persistently small gradients ⇒ *larger* step. The step self-normalizes.]

== Why divide by the RMS?

Dividing by $sqrt(s_t)$ makes the update *scale-invariant* per coordinate:
$ (g_(t,j))/(sqrt(s_(t,j))) approx plus.minus 1 $
#pause
#notebox[Steep and shallow directions get *comparable* step sizes — exactly what the ravine needed, without hand-tuning per axis.]

== Worked example: RMSProp on two axes #D

Steep axis sees $g = 10$, shallow axis sees $g = 0.1$; take $rho = 0.9$, $s_0 = 0$:
#pause
$ s_"steep" = 0.9 dot 0 + 0.1 dot 10^2 = 10, quad quad s_"shallow" = 0.9 dot 0 + 0.1 dot 0.1^2 = 0.001 $
#pause
Divide each gradient by its own $sqrt(s)$ (ignore $epsilon$):
$ g_"steep"/sqrt(s_"steep") = 10/sqrt(10) approx 3.16, quad quad g_"shallow"/sqrt(s_"shallow") = 0.1/sqrt(0.001) approx 3.16 $
#pause
#result[a $100 times$ gap in raw gradient becomes the *same* effective step]
#pause
#align(center, text(size: 16pt, fill: MUTED)[self-normalization: both axes move at rate $~eta$, so the ravine no longer zigzags.])

// ═══════════════════════════ PART V — Adam ═══════════════════════════
= Adam

== Adam = momentum + adaptivity

Keep *two* moving averages of the gradient:
$ m_t = beta_1 m_(t-1) + (1 - beta_1) g_t quad quad "(1st moment — direction)" $
$ v_t = beta_2 v_(t-1) + (1 - beta_2) g_t^2 quad quad "(2nd moment — magnitude)" $
#pause
#result[$m_t$ = smoothed *direction* · $v_t$ = smoothed *scale*]

== A subtle bug: initialization bias #D

We start $m_0 = 0$ and $v_0 = 0$. After one step with constant $g$:
$ m_1 = (1 - beta_1) g = 0.1 g quad quad ("way below " g) $
#pause
#alertbox[Early moving averages are biased *toward zero* — worst at $t = 1$, since there is nothing yet to average.]

== Bias correction

Divide out the shrinkage factor $1 - beta^t$:
$ hat(m)_t = m_t/(1 - beta_1^t), quad quad hat(v)_t = v_t/(1 - beta_2^t) $
#pause
Then take the RMSProp-style step with the *corrected* moments:
$ theta_(t+1) = theta_t - eta thin hat(m)_t/(sqrt(hat(v)_t) + epsilon) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[as $t -> infinity$, $beta^t -> 0$ and the correction fades away.])

== Worked example: Adam step 1 #D

$g_1 = 4$, $quad beta_1 = 0.9$, $beta_2 = 0.999$, $eta = 0.001$, start $m_0 = v_0 = 0$:
#pause
*1st moment* — EMA of the gradient (direction):
$ m_1 = 0.9 dot 0 + 0.1 dot 4 = 0.4 $
#pause
*2nd moment* — EMA of the squared gradient (magnitude):
$ v_1 = 0.999 dot 0 + 0.001 dot 16 = 0.016 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[both are far *below* the true scale ($g = 4$, $g^2 = 16$) — biased toward $0$ by the cold start.])

== Worked example: Adam step 1 (cont.) #D

Undo the bias by dividing out $1 - beta^t$ (here $t = 1$):
#pause
$ hat(m)_1 = 0.4/(1 - 0.9) = 4 $
#pause
$ hat(v)_1 = 0.016/(1 - 0.999) = 16 $
#pause
Now take the RMSProp-style step with the corrected moments:
$ Delta theta_1 = -eta thin hat(m)_1/sqrt(hat(v)_1) = -0.001 dot 4/sqrt(16) = -0.001 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[bias correction rescues $m_1, v_1$ back to the true gradient scale — the step is exactly $-eta$.])

== Worked example: Adam step 2 #D

$g_2 = 6$ (now $t = 2$), carrying $m_1 = 0.4$, $v_1 = 0.016$ from before:
#pause
*1st moment* — decay the old average, add $10%$ of the new gradient:
$ m_2 = 0.9 dot 0.4 + 0.1 dot 6 = 0.96 $
#pause
*2nd moment* — same, with the squared gradient $36$:
$ v_2 = 0.999 dot 0.016 + 0.001 dot 36 = 0.051984 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the bias factor is now $1 - beta^2$, weaker than at $t = 1$ — the correction is already fading.])

== Worked example: Adam step 2 (cont.) #D

Bias-correct with $t = 2$, so divide by $1 - beta^2$:
#pause
$ hat(m)_2 = 0.96/(1 - 0.9^2) = 0.96/0.19 approx 5.05 $
#pause
$ hat(v)_2 = 0.051984/(1 - 0.999^2) approx 26.0 $
#pause
$ Delta theta_2 = -0.001 dot 5.05/sqrt(26.0) approx -0.000991 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[each coordinate moves by roughly $plus.minus eta$, regardless of the raw gradient size.])

== Adam intuition

$ theta_(t+1) = theta_t - eta thin underbrace(hat(m)_t, "which way") / underbrace((sqrt(hat(v)_t) + epsilon), "how big") $
#pause
- numerator $hat(m)_t$ — a *momentum-smoothed direction*;
#pause
- denominator $sqrt(hat(v)_t)$ — a *per-coordinate magnitude* that self-normalizes;
#pause
- together: a smoothed, scale-free step in every coordinate.

== Adam defaults

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left),
  table.header([*Hyperparameter*], [*Default*]),
  [$beta_1$ (momentum)], [$0.9$],
  [$beta_2$ (RMS)], [$0.999$],
  [$epsilon$], [$10^(-8)$],
  [$eta$ (learning rate)], [$10^(-3)$ (still tune this)],
))
#pause
#notebox[These defaults "just work" on a huge range of problems — a big reason Adam is the default optimizer.]

== Checkpoint: Adam's two memories #Q

#mcq(
  [What extra information does Adam keep beyond plain gradient descent?],
  [Only the current loss value],
  [A first-moment direction and a second-moment magnitude estimate],
  [A full Hessian matrix],
  [The complete training dataset],
)

== Answer: Adam's two memories #A

#mcq-answer([B], [A first and a second moment], [The first moment smooths direction like momentum; the second tracks per-coordinate gradient scale for adaptive steps.])

== Interactive: Adam vs the rest #I

#interbox(link-to: IA + "optimizer-race")[
  Race GD, SGD, momentum and Adam on the same surface. Watch Adam's per-coordinate scaling walk straight down a ravine that makes plain GD zigzag.
]

// ═══════════════════════════ PART VI — Comparison ═══════════════════════════
= Putting the optimizers side by side

== Same loss, four optimizers #V

$cal(L) = x^2 + 50 y^2$, identical start point:
#pause
// native: four real optimizer trajectories on one loss surface — loss + start
// written once, gradient via grad2d
#let loss(x, y) = x * x + 50.0 * y * y
#let g = grad2d(loss)
#let x0 = (-2.4, 0.85)
#align(center, contour(loss, xlim: (-2.7, 2.7), ylim: (-1.0, 1.0),
  samples: 60, levels: 12, size: (100mm, 38mm), color: MUTED, marks: ((0, 0, [·], RED),),
  paths: (
    gd(g, x0, lr: 0.014, steps: 60),                       // green — crawls
    sgd(g, x0, lr: 0.014, noise: 6.0, seed: 0, steps: 60), // teal — jitters
    momentum(g, x0, lr: 0.006, steps: 60),                 // orange — overshoots
    adam(g, x0, lr: 0.16, steps: 60),                      // red — cuts across
  ),
))
#align(center, text(size: 16pt)[
  #text(fill: GREEN, weight: 600)[GD] crawls · #text(fill: TEAL, weight: 600)[SGD] jitters · #text(fill: ACC, weight: 600)[momentum] overshoots then settles · #text(fill: RED, weight: 600)[Adam] cuts across
])

== Summary table

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 6pt), align: (left, center, center, left),
  table.header([*Optimizer*], [*Memory*], [*Adaptive?*], [*Strength*]),
  [GD], [none], [no], [exact gradient, stable],
  [SGD], [none], [no], [cheap, noisy, explores],
  [Momentum], [1st moment], [no], [smooths valleys, accelerates],
  [Adam], [1st + 2nd], [yes], [robust default, per-coord scaling],
))

== When to use what

#pause
- *Adam / AdamW* — the default. Fast, robust, forgiving of LR. Transformers, most research.
#pause
- *SGD + momentum* — often the *best final generalization*, especially in vision (ResNets, ConvNets) — but needs more LR tuning.
#pause
- *Either way, the LR schedule matters* as much as the optimizer choice.
#pause
#notebox[Pragmatic rule: start with AdamW + a cosine schedule; reach for SGD+momentum when you are chasing the last bit of test accuracy.]

== AdamW: decoupled weight decay #OPT

Classic Adam folds $ell_2$ regularization into the gradient — which then gets *rescaled* by $sqrt(hat(v))$, distorting it.
#pause
AdamW instead applies weight decay *directly* to the parameters:
$ theta_(t+1) = theta_t - eta thin hat(m)_t/(sqrt(hat(v)_t) + epsilon) - eta lambda thin theta_t $
#pause
#result[decoupled decay ⇒ the modern default for training large models]

// ═══════════════════════════ PART VII — Learning-rate schedules ═══════════════════════════
= Learning-rate schedules

== A fixed learning rate is rarely ideal

#pause
- *early* training: far from any minimum → want *large* steps to make progress;
#pause
- *late* training: near a minimum → want *small* steps to settle, not bounce.
#pause
#result[so *decay* the learning rate over training]

== Common schedules #V

// native ml-plot: five schedules on t∈[0,100), e0=1 (mirrors l5_figs.py f_lr_schedules)
// each schedule is a closed-form fn of the step, sampled at every integer step 0..lr-T-1
#let lr-T = 100
#let sched-step = t => calc.pow(0.5, calc.floor(t / 30))
#let sched-expo = t => calc.exp(-0.03 * t)
#let sched-cos = t => 0.5 * (1 + calc.cos(calc.pi * t / lr-T))
#let sched-wc = t => if t < 10 { t / 10 } else { 0.5 * (1 + calc.cos(calc.pi * (t - 10) / 90)) }
#let sched-oc = t => calc.max(0, calc.min(1.05,
  if t < 25 { 0.25 + 0.75 * t / 25 }
  else if t < 90 { 1 - 0.85 * (t - 25) / 65 }
  else { 0.15 * (1 - (t - 90) / 10) }))
#align(center, lines(
  fn: (sched-step, sched-expo, sched-cos, sched-wc, sched-oc),
  domain: (0, lr-T - 1), samples: lr-T - 1,
  colors: (TEAL, BLUE, GREEN, ACC, RED),
  markers: false,
  x-label: [training step], y-label: [learning rate $eta_t$],
  size: (90mm, 46mm),
))
#pause
#align(center, text(size: 16pt)[#text(fill: TEAL)[step] · #text(fill: BLUE)[exponential] · #text(fill: GREEN)[cosine] · #text(fill: ACC)[warmup+cosine] · #text(fill: RED)[one-cycle]])

== Warmup

Start from *near zero* and ramp up over the first $T_"warmup"$ steps, then decay:
$ eta_t = eta_max dot t/T_"warmup" quad "for" t <= T_"warmup" $
#pause
#notebox[Early gradients are large and unreliable (moments not yet warmed up); a big LR then can *blow up* training. Warmup is near-essential for Transformers.]

== Cosine decay + warmup — the modern default

$ eta_t = 1/2 eta_max (1 + cos(pi thin (t - T_"warmup")/(T - T_"warmup"))) $
#pause
- smooth decay from $eta_max$ down to $~0$;
#pause
- no sharp drops (unlike step decay);
#pause
- pairs naturally with a short linear warmup.

== Interactive: LR schedules #I

#interbox(link-to: IA + "lr-schedule-visualizer")[
  Compose warmup length, peak LR, and decay shape (step / exponential / cosine / one-cycle) and watch the resulting $eta_t$ curve — and how it changes the loss trajectory.
]

// ═══════════════════════════ PART VIII — Pathologies ═══════════════════════════
= Optimization pathologies

== Too much gradient noise

SGD noise helps *explore* — but past a point it *prevents convergence*.
#pause
$ theta_(t+1) = theta_t - eta(nabla cal(L) + epsilon_t) $
#pause
#notebox[If $eta thin "Var"(epsilon_t)$ stays large, the iterate *bounces around* the minimum forever. Remedies: decay $eta$, grow the batch, or use momentum to average the noise out.]

== Vanishing and exploding gradients

Backprop multiplies Jacobians across layers:
$ nabla_(theta_1) cal(L) = product_(ell) J_ell dot (dots) $
#pause
#two(
  alertbox[*Vanishing* — factors $< 1$ ⇒ product $-> 0$ ⇒ early layers stop learning.],
  alertbox[*Exploding* — factors $> 1$ ⇒ product blows up ⇒ NaNs / divergence.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[fixes live in *architecture*: activation choice, init, normalization, residual connections, gradient clipping.])

== Saddle points

In high dimensions, most points with $nabla cal(L) = 0$ are *saddles*, not minima — the Hessian has *mixed-sign* eigenvalues.
#pause
#notebox[Plain gradient descent can *stall* on the flat directions of a saddle. SGD's noise is a feature here: it *kicks the iterate off* the saddle and lets training continue.]

== Sharp vs flat minima #V #OPT

#fig("/lecture5/figures/sharp_flat.svg", w: 56%)
#pause
#notebox[Flat minima are widely associated with *better generalization*: a small shift in parameters or data barely changes the loss. Small-batch SGD tends to prefer flatter minima. #text(fill: MUTED)[(an active, subtle research area)]]

// ═══════════════════════════ PART IX — End-to-end + summary ═══════════════════════════
= The training loop, end to end

== One optimization step in PyTorch #I

#codebox[```python
for x, y in loader:                 # minibatch
    optimizer.zero_grad()           # clear old gradients
    logits = model(x)               # forward pass
    loss   = loss_fn(logits, y)     # scalar loss
    loss.backward()                 # backprop -> param.grad
    optimizer.step()                # update rule uses grads
```]
#pause
#align(center, text(size: 16pt, fill: MUTED)[forward → loss → *backprop (gradients)* → *optimizer (update)* — every lecture so far, in six lines.])

== The step, three ways #V

#codebox(size: 13pt)[```python
# SGD
p -= lr * p.grad

# SGD + momentum
v = beta * v + p.grad
p -= lr * v

# Adam (per parameter)
m = b1*m + (1-b1)*p.grad
v = b2*v + (1-b2)*p.grad**2
mhat = m/(1-b1**t); vhat = v/(1-b2**t)
p -= lr * mhat / (vhat.sqrt() + eps)
```]
#pause
#align(center, text(size: 15pt, fill: MUTED)[same loop — swap only what `optimizer.step()` does.])

== Choosing an optimizer in one line

#codebox[```python
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
```]
#pause
#notebox[The optimizer and the schedule are *two knobs*; both matter. Defaults above are sane starting points.]

== One-slide summary

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Method*], [*Core idea*]),
  [GD], [full-dataset gradient — exact but slow],
  [SGD], [sample a batch — unbiased, noisy, cheap],
  [Momentum], [average gradients — smooth, accelerate],
  [RMSProp], [divide by RMS gradient — per-coord scale],
  [Adam], [momentum + adaptive scale + bias correction],
  [LR schedule], [big early, small late (warmup + decay)],
))

== Final mental model — one idea

#align(center, text(size: 22pt)[
  Backprop answers: *what direction?* \
  Optimization answers: \
  *how far, how smoothly, how adaptively?* \
  #v(4pt)
  #text(size: 18pt, fill: MUTED)[gradient = direction · optimizer = the art of the step]
])

#focus-slide[
  Backprop gives the gradient; the optimizer turns it into learning.
  #v(12pt)
  #set text(size: 22pt)
  Next: *regularization and generalization* — why the network that fits the training set also works on data it has never seen.
]
