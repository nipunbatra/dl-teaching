// Optimization for Deep Learning - Lecture 5 · ES 667 Deep Learning
// Compile from the repository root:
//   typst compile --root . --input handout=true lecture5/L5-optimization.typ slides-pdf/L5.pdf
//   typst compile --root . lecture5/L5-optimization.typ slides-pdf/L5-presentation.pdf

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Optimization for Deep Learning],
  subtitle: [How gradients become reliable parameter updates],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let NB = "https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L05/"

// Semantic colour grammar used throughout the lecture.
// BLUE = measured gradient · TEAL = direction memory · GREEN = scale memory
// ACC = applied update / learning-rate clock · RED = failure or warning.
#let note(body, color: TEAL) = block(
  width: 100%, inset: (left: 10pt, right: 2pt, top: 4pt, bottom: 4pt),
  stroke: (left: 1.2pt + color), [#body],
)
#let warning(body) = note(body, color: RED)
#let punch(body) = align(center, text(size: 18pt, weight: 650)[#body])
#let caption(body) = align(center, text(size: 14.5pt, fill: MUTED)[#body])
#let hairline(title, body, color: TEAL) = block(width: 100%, inset: 2pt, [
  #text(size: 14pt, weight: 700, fill: color)[#title]
  #v(2pt)
  #line(length: 100%, stroke: 0.7pt + color)
  #v(5pt)
  #body
])
#let swatch(color, body) = box(width: 12pt, height: 3pt, fill: color, radius: 1pt) + h(4pt) + body

#let semantic-legend = align(center, text(size: 13pt)[
  #swatch(BLUE, [measured $g_t$]) #h(15pt)
  #swatch(TEAL, [direction $m_t$]) #h(15pt)
  #swatch(GREEN, [scale $v_t$ or $G_t$]) #h(15pt)
  #swatch(ACC, [applied $Delta theta_t$])
])

// Geometry uses a separate, stable grammar: teal contours are the loss field,
// amber paths are applied parameter motion, and red marks a target or failure.
#let geometry-theme = theme(base: dl-theme, cycle: (INK, ACC, TEAL, GREEN, RED))

#let update-loop = align(center, diagram(
  spacing: (17mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    node((0, 0), $theta_t$, radius: 6.2mm, stroke: 1pt + INK)
    node((1, 0), text(size: 13pt)[forward + loss], shape: fletcher.shapes.rect,
      corner-radius: 3pt, inset: 6pt, fill: rgb("#F1F4F4"), stroke: 0.9pt + MUTED)
    node((2, 0), text(size: 13pt)[backprop], shape: fletcher.shapes.rect,
      corner-radius: 3pt, inset: 6pt, fill: BLUE.lighten(92%), stroke: 1pt + BLUE)
    node((3, 0), $g_t$, radius: 6.2mm, stroke: 1pt + BLUE)
    node((4, 0), text(size: 13pt)[state + rule], shape: fletcher.shapes.rect,
      corner-radius: 3pt, inset: 6pt, fill: TEAL.lighten(92%), stroke: 1pt + TEAL)
    node((5, 0), $theta_(t+1)$, radius: 7mm, stroke: 1.1pt + ACC)
    for i in range(5) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.9pt + MUTED) }
  }
))

#let adam-pipeline = align(center, diagram(
  spacing: (25mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    node((0, 0), $g_t$, radius: 6mm, stroke: 1pt + BLUE)
    node((1, -0.75), $m_t$, radius: 6mm, stroke: 1pt + TEAL)
    node((1, 0.75), $v_t$, radius: 6mm, stroke: 1pt + GREEN)
    node((2, 0), $Delta theta_t$, radius: 7mm, stroke: 1.1pt + ACC)
    edge((0, 0), (1, -0.75), "-|>", stroke: 1pt + TEAL,
      label: text(size: 10pt, fill: TEAL)[smooth sign + direction], label-sep: 5pt)
    edge((0, 0), (1, 0.75), "-|>", stroke: 1pt + GREEN,
      label: text(size: 10pt, fill: GREEN)[track squared size], label-sep: 5pt)
    edge((1, -0.75), (2, 0), "-|>", stroke: 1pt + TEAL)
    edge((1, 0.75), (2, 0), "-|>", stroke: 1pt + GREEN)
  }
))

#title-slide()

// ═══════════════════════════ OPENING ═══════════════════════════
== You already know the two ends of a training step

From backprop, you know how to compute every sensitivity

$ g_t = nabla_theta cal(L)_(cal(B)_t)(theta_t). $

#pause
From gradient descent, you know the simplest parameter update

$ theta_(t+1) = theta_t - eta g_t. $

#pause
Our first new idea—momentum—will remember recent gradients instead of trusting
one minibatch at a time.

#note[
  Today's question is the missing middle: *what state should an optimizer remember, and how should that state shape the next step?*
]

== Commit: one parameter is seen often; one is seen rarely #Q

Two coordinates receive this minibatch-gradient history:

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 16pt, y: 8pt),
  align: (left, center, center, center),
  table.header([coordinate], [$g_1$], [$g_2$], [$g_3$]),
  [#text(fill: BLUE, weight: 650)[dense $d$]], [$4$], [$4$], [$4$],
  [#text(fill: GREEN, weight: 650)[rare $r$]], [$0$], [$0$], [$0.4$],
))

#v(8pt)
#mcq(
  [At step 3, which information could tell us that $r$ is rare?],
  [only the current gradient $g_3$],
  [the signs in $g_1,g_2,g_3$],
  [past squared gradients per coordinate],
  [a larger global learning rate],
)

== Keep your choice; we will earn the answer

We start at $theta_0=(1,1)$ and repeatedly reuse the same stream:

$ g_1=(4,0), quad g_2=(4,0), quad g_3=(4,0.4). $

#pause
#align(center, diagram(spacing: (28mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), [current gradient], shape: fletcher.shapes.rect, corner-radius: 3pt,
    inset: 7pt, stroke: 1pt + BLUE)
  node((1, -0.65), [direction history], shape: fletcher.shapes.rect, corner-radius: 3pt,
    inset: 7pt, stroke: 1pt + TEAL)
  node((1, 0.65), [scale history], shape: fletcher.shapes.rect, corner-radius: 3pt,
    inset: 7pt, stroke: 1pt + GREEN)
  node((2, 0), [parameter update], shape: fletcher.shapes.rect, corner-radius: 3pt,
    inset: 7pt, stroke: 1pt + ACC)
  edge((0, 0), (1, -0.65), "-|>", stroke: 0.9pt + TEAL)
  edge((0, 0), (1, 0.65), "-|>", stroke: 0.9pt + GREEN)
  edge((1, -0.65), (2, 0), "-|>", stroke: 0.9pt + TEAL)
  edge((1, 0.65), (2, 0), "-|>", stroke: 0.9pt + GREEN)
}))

#pause
#punch[Every method today will be one new answer to: “what should the past change about this step?”]

== The route for today #V

#align(center, diagram(spacing: (23mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), [SGD], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 7pt, stroke: 1pt + INK)
  node((1, 0), [momentum], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 7pt, stroke: 1pt + TEAL)
  node((2, 0), [AdaGrad], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 7pt, stroke: 1pt + GREEN)
  node((3, 0), [RMSProp], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 7pt, stroke: 1pt + GREEN)
  node((4, 0), [AdamW], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 7pt, stroke: 1pt + ACC)
  for i in range(4) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.8pt + MUTED) }
}))

#v(10pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([CORE · 55 min], [why each stored quantity exists], color: TEAL),
  hairline([SHOULD COVER · 15 min], [calculate, predict, then run], color: GREEN),
  hairline([OPTIONAL · 10 min], [geometry, diagnostics, sources], color: MUTED),
))

#semantic-legend

// ═════════════════════ FROM GRADIENT TO UPDATE ═════════════════════
= The optimizer is the middle of the loop

== Backprop measures; the optimizer acts #V

#update-loop

#pause
#v(8pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 20pt,
  hairline([Backprop · #text(fill: BLUE)[$g_t$]], [What would a tiny parameter change do to this minibatch loss?], color: BLUE),
  hairline([Optimizer · #text(fill: ACC)[$Delta theta_t$]], [Given this measurement and stored history, what change should we apply?], color: ACC),
))

#pause
#punch[$theta_(t+1)=theta_t+Delta theta_t$ — the gradient and the update are not the same object.]

== A learning rate converts sensitivity into distance

For plain minibatch SGD,

$ Delta theta_t=-eta g_t. $

#pause
On our recurring stream, with $eta=0.1$:

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 7pt),
  align: (center, center, center, center),
  table.header([step], [$g_t$], [$Delta theta_t$], [$theta_t$]),
  [$0$], [—], [—], [$(1,1)$],
  [$1$], [$(4,0)$], [$(-0.4,0)$], [$(0.6,1)$],
  [$2$], [$(4,0)$], [$(-0.4,0)$], [$(0.2,1)$],
  [$3$], [$(4,0.4)$], [$(-0.4,-0.04)$], [$(-0.2,0.96)$],
))

#note[One global $eta$ preserves the $10 times$ magnitude gap in $g_3$. It knows nothing about how often each coordinate has appeared.]

== Why the negative gradient is locally downhill #D #OPT

For a small proposed step $Delta$, the first-order model is

$ cal(L)(theta+Delta) approx cal(L)(theta)+g^top Delta. $

#pause
Among steps with $norm(Delta)<=r$, the smallest predicted change occurs at

$ Delta^star=-r g/norm(g). $

#pause
#note[
  The result is local and Euclidean: after moving we must measure a new gradient, and a change of parameterization changes what “steepest” means.
]

== One quadratic makes learning-rate behaviour predictable #D

Let $cal(L)(theta)=(theta-3)^2$ and $e_t=theta_t-3$. Then

$ e_(t+1)=(1-2eta)e_t. $

#pause
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([$0<eta<0.5$], [positive multiplier: approach from one side], color: TEAL),
  hairline([$0.5<eta<1$], [negative multiplier: alternate and shrink], color: ACC),
  hairline([$eta>1$], [magnitude above one: diverge], color: RED),
))

#pause
#punch[The loss surface stays fixed; the discrete step can crawl, settle, oscillate, or diverge.]

== Predict before revealing the trajectory #Q

For $cal(L)(theta)=(theta-3)^2$, which rate makes the signed error alternate while shrinking by half?

#mcq(
  [Solve $e_(t+1)=(1-2eta)e_t$ mentally.],
  [$eta=0.25$],
  [$eta=0.50$],
  [$eta=0.75$],
  [$eta=1.25$],
)

== Answer: the multiplier must be $-0.5$ #A

$1-2eta=-0.5 quad arrow.r quad eta=0.75.$

#align(center, text(size: 22pt, fill: ACC)[
  $e_0 arrow.r -0.5e_0 arrow.r 0.25e_0 arrow.r -0.125e_0 arrow.r dots$
])

#mcq-answer([C], [$eta=0.75$], [the sign creates alternation; the magnitude $0.5$ creates geometric decay.])

== Follow along: see the multiplier become a path #I

#interbox(link-to: NB + "01_gd_quadratic.ipynb")[]

Before each run:

- predict the sign and magnitude of $1-2eta_t$;
- calculate the next parameter value;
- inspect both signed error and loss;
- explain any disagreement between prediction and plot.

#note[Complete the fixed-rate section now. Return to the warmup–cosine section after schedules are introduced.]

// ═════════════════════════ DIRECTION MEMORY ═════════════════════════
= Momentum remembers direction

== Minibatches make a useful gradient noisy

For a uniformly sampled minibatch $cal(B)_t$,

$ g_t=1/abs(cal(B)_t) sum_(i in cal(B)_t) nabla ell_i(theta_t), quad EE[g_t | theta_t]=nabla cal(L)(theta_t). $

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([One update], [may point away from the full-batch downhill direction], color: RED),
  hairline([Repeated sampling], [recovers the full gradient in expectation], color: TEAL),
))

#pause
#note[“Unbiased” is an average statement, not a promise that every minibatch step lowers training loss.]

== A ravine makes gradients alternate across one axis #V

#let ravine = ad.expr("x^2 + 50*y^2", ("x", "y"))
#let ravine-grad = ad.grad-fn(ravine)
#let ravine-start = (-2.4, 0.85)

#align(center, contour(ad.fn2(ravine),
  xlim: (-2.7, 2.7), ylim: (-1.0, 1.0), samples: 58, levels: 11,
  size: (142mm, 54mm), color: TEAL,
  paths: (gd(ravine-grad, ravine-start, lr: 0.014, steps: 38),),
  marks: ((-2.4, 0.85, [start], BLUE), (0, 0, [minimum], RED)),
  x-label: [shallow coordinate], y-label: [steep coordinate],
  theme: geometry-theme,
))

#caption[The plotted path is computed from the displayed loss. Stability along steep $y$ forces slow progress along shallow $x$.]

== Momentum averages direction before stepping

Using the normalized moving-average convention,

$ m_t=beta m_(t-1)+(1-beta)g_t, quad Delta theta_t=-eta m_t. $

#pause
#align(center, diagram(spacing: (25mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, -0.65), $m_(t-1)$, radius: 7mm, stroke: 1pt + TEAL)
  node((0, 0.65), $g_t$, radius: 6mm, stroke: 1pt + BLUE)
  node((1, 0), $m_t$, radius: 7mm, stroke: 1.1pt + TEAL)
  node((2, 0), $Delta theta_t$, radius: 7mm, stroke: 1.1pt + ACC)
  edge((0, -0.65), (1, 0), "-|>", stroke: 1pt + TEAL, label: [$beta$], label-sep: 5pt)
  edge((0, 0.65), (1, 0), "-|>", stroke: 1pt + BLUE, label: [$1-beta$], label-sep: 5pt)
  edge((1, 0), (2, 0), "-|>", stroke: 1pt + ACC)
}))

#pause
#punch[Alternating components cancel; persistent components accumulate.]

== Work the memory before seeing its effect #D

Take $beta=0.9$, $m_0=0$, and one coordinate with gradients $10,8,11$.

#pause
#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt),
  align: (center, center, left, center),
  table.header([$t$], [$g_t$], [$m_t=0.9m_(t-1)+0.1g_t$], [$m_t$]),
  [$1$], [$10$], [$0.9(0)+0.1(10)$], [$1.00$],
  [$2$], [$8$], [$0.9(1.00)+0.1(8)$], [$1.70$],
  [$3$], [$11$], [$0.9(1.70)+0.1(11)$], [$2.63$],
))

#pause
#note[The average starts low because its unobserved history is represented by zeros. This cold-start bias returns when Adam combines two moving averages.]

== Momentum changes direction, not coordinate scale

On our recurring stream, both coordinates use the same $beta$:

$ m_t=beta m_(t-1)+(1-beta)g_t. $

#pause
It can smooth the dense coordinate's direction, but when the rare coordinate finally appears, its small gradient is still small.

#align(center, text(size: 22pt)[
  #text(fill: TEAL)[$m_t$ remembers *sign and direction*]
  #h(18pt) $!=$ #h(18pt)
  #text(fill: GREEN)[a per-coordinate scale]
])

#pause
#punch[To recognize rarity, the optimizer must remember magnitude separately for every coordinate.]

// ═══════════════════════ ADAPTIVE SCALE ═══════════════════════
= AdaGrad remembers all squared gradients

== Squared gradients form a per-coordinate activity ledger

AdaGrad accumulates

$ G_t=G_(t-1)+g_t^2 $

elementwise, then, with small $epsilon>0$ preventing a zero denominator, updates

$ Delta theta_t=-eta g_t/(sqrt(G_t)+epsilon). $

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([Frequently active], [$G_(t,j)$ grows; future steps shrink], color: BLUE),
  hairline([Rarely active], [$G_(t,j)$ stays small; a rare signal keeps leverage], color: GREEN),
))

#pause
#punch[AdaGrad turns gradient history into a separate effective rate for every coordinate.]

== Build the ledger for our two coordinates #D

$ g_1=(4,0), quad g_2=(4,0), quad g_3=(4,0.4). $

#pause
#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 7pt),
  align: (center, center, center, center),
  table.header([$t$], [$g_t$], [$G_t=sum_(tau<=t)g_tau^2$], [$g_t/(sqrt(G_t)+epsilon)$]),
  [$1$], [$(4,0)$], [$(16,0)$], [$(1,0)$],
  [$2$], [$(4,0)$], [$(32,0)$], [$(0.707,0)$],
  [$3$], [$(4,0.4)$], [$(48,0.16)$], [$(0.577,1)$],
))

#pause
#note[At its first appearance, the rare coordinate receives normalized magnitude $1$ even though its raw gradient is $10 times$ smaller.]

== AdaGrad answers the opening question #A

With $eta=0.1$ and $theta_0=(1,1)$:

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 14pt, y: 7pt),
  align: (center, center, center, center),
  table.header([step], [dense update], [rare update], [$theta_t$]),
  [$1$], [$-0.100$], [$0$], [$(0.900,1)$],
  [$2$], [$-0.071$], [$0$], [$(0.829,1)$],
  [$3$], [$-0.058$], [$-0.100$], [$(0.772,0.900)$],
))

#mcq-answer([C], [past squared gradients per coordinate], [only a coordinate-wise history distinguishes “small because rare” from “small on every batch.”])

== The strength of AdaGrad becomes its failure

$G_t$ only increases.

#pause
#align(center, grid(columns: (1fr, 18mm, 1fr), row-gutter: 8pt, align: horizon,
  hairline([1 · Observe], [more nonzero gradients], color: BLUE),
  align(center, text(size: 21pt, fill: MUTED)[$arrow.r$]),
  hairline([2 · Accumulate], [larger $G_t=sum g_t^2$], color: GREEN),
  hairline([4 · Consequence], [eventual near-freeze], color: RED),
  align(center, text(size: 21pt, fill: MUTED)[$arrow.l$]),
  hairline([3 · Normalize], [smaller effective rate $eta/sqrt(G_t)$], color: ACC),
))

#pause
#warning[Old gradients keep equal voting rights forever. On a long nonstationary training run, useful step sizes can decay too aggressively.]

// ═══════════════════════ RMSPROP ═══════════════════════
= RMSProp lets old scale evidence fade

== Replace the sum with a moving average

RMSProp stores

$ v_t=rho v_(t-1)+(1-rho)g_t^2 $

and applies

$ Delta theta_t=-eta g_t/(sqrt(v_t)+epsilon). $

#pause
#align(center, text(size: 21pt)[
  AdaGrad: #text(fill: GREEN)[all squared gradients]
  #h(18pt) $arrow.r$ #h(18pt)
  RMSProp: #text(fill: GREEN)[recent squared gradients]
])

#pause
#note[The approximate memory window is $1/(1-rho)$ steps: about $10$ for $rho=0.9$, about $100$ for $rho=0.99$.]

== Recompute step 3 with a fading ledger #D

For our stream, let $rho=0.9$ and $v_0=(0,0)$:

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 14pt, y: 7pt),
  align: (center, center, center),
  table.header([$t$], [$v_t$], [$g_t/(sqrt(v_t)+epsilon)$]),
  [$1$], [$(1.600,0)$], [$(3.162,0)$],
  [$2$], [$(3.040,0)$], [$(2.294,0)$],
  [$3$], [$(4.336,0.016)$], [$(1.921,3.162)$],
))

#pause
#note[The rare coordinate is still amplified relative to the dense one, but evidence now fades instead of accumulating forever.]

== A zero start creates an artificially small denominator

At the first nonzero gradient,

$ v_1=(1-rho)g_1^2, quad g_1/sqrt(v_1)="sign"(g_1)/sqrt(1-rho). $

#pause
For $rho=0.9$, the normalized magnitude is $1/sqrt(0.1) approx 3.16$.

#pause
#warning[This is a cold-start effect, not evidence that the first gradient is exceptionally trustworthy. Adam will correct this missing moving-average mass explicitly.]

== What $epsilon$ does—and does not do

RMSProp and Adam divide by $sqrt(v_(t,j))+epsilon$.

#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([$sqrt(v_(t,j)) \gg epsilon$], [$epsilon$ has almost no effect], color: TEAL),
  hairline([$sqrt(v_(t,j)) approx 0$], [the denominator stays nonzero and the step finite], color: RED),
))

#pause
#note[$epsilon$ is numerical protection, but it can also change effective rates in extremely low-variance coordinates. It is part of the optimizer definition, not decorative notation.]

// ═══════════════════════════ ADAM ═══════════════════════════
= Adam combines direction and scale

== Two memories solve two different problems #V

#adam-pipeline

#v(8pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([#text(fill: TEAL)[$m_t$ · first moment]], [smooth noisy signs and preserve persistent direction], color: TEAL),
  hairline([#text(fill: GREEN)[$v_t$ · second raw moment]], [normalize coordinate-wise gradient scale], color: GREEN),
))

#pause
#punch[Adam is momentum in the numerator and RMSProp-like scaling in the denominator.]

== Write Adam as state, correction, update

#align(center, table(
  columns: (31mm, 1fr), stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt),
  align: (left, left),
  [#text(fill: TEAL, weight: 650)[direction]], [$m_t=beta_1m_(t-1)+(1-beta_1)g_t$],
  [#text(fill: GREEN, weight: 650)[scale]], [$v_t=beta_2v_(t-1)+(1-beta_2)g_t^2$],
  [#text(fill: INK, weight: 650)[correct]], [$hat(m)_t=m_t/(1-beta_1^t), quad hat(v)_t=v_t/(1-beta_2^t)$],
  [#text(fill: ACC, weight: 650)[update]], [$Delta theta_t=-eta hat(m)_t/(sqrt(hat(v)_t)+epsilon)$],
))

#pause
#semantic-legend

== Why bias correction has exactly that denominator #D

If $g_t=g$ is constant and $m_0=0$, unrolling gives

$ m_t=(1-beta_1^t)g. $

Similarly,

$ v_t=(1-beta_2^t)g^2. $

#pause
Dividing by the missing mass restores the constant moments:

$ hat(m)_t=g, quad hat(v)_t=g^2. $

#pause
#note[The correction removes initialization bias under a constant-gradient thought experiment; it does not make a noisy gradient statistically unbiased.]

== Predict Adam's first update #Q

Let $g_1=(4,0.4)$, $m_0=v_0=0$, $beta_1=0.9$, $beta_2=0.999$.

#mcq(
  [Ignoring $epsilon$, what is $hat(m)_1/sqrt(hat(v)_1)$?],
  [$(4,0.4)$],
  [$(1,1)$],
  [$(0.1,0.001)$],
  [$(0.4,0.00016)$],
)

== Answer: correction restores both scales #A

$m_1=(0.4,0.04), quad v_1=(0.016,0.00016).$

#pause
$hat(m)_1=(4,0.4), quad hat(v)_1=(16,0.16).$

#pause
$hat(m)_1/sqrt(hat(v)_1)=(1,1).$

#mcq-answer([B], [$(1,1)$], [on the first nonzero observation, bias-corrected Adam retains sign while normalizing coordinate magnitude.])

== Carry our recurring stream through Adam #D

For $g_1=(4,0), g_2=(4,0), g_3=(4,0.4)$:

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt),
  align: (left, center, center),
  table.header([quantity at $t=3$], [dense $d$], [rare $r$]),
  [$m_3$], [$1.084$], [$0.040$],
  [$hat(m)_3$], [$4.000$], [$0.148$],
  [$v_3$], [$0.047952$], [$0.000160$],
  [$hat(v)_3$], [$16.000$], [$0.05339$],
  [$hat(m)_3/sqrt(hat(v)_3)$], [$1.000$], [$0.639$],
))

#pause
#note[Scale normalization lifts the rare coordinate; direction memory tempers it because two previous batches contained zero evidence for that coordinate.]

== AdamW keeps weight decay out of the adaptive denominator

Adam supplies an adaptive data-gradient step. AdamW adds shrinkage separately:

$ theta_(t+1)=theta_t - eta hat(m)_t/(sqrt(hat(v)_t)+epsilon) - eta lambda theta_t. $

#pause
#align(center, diagram(spacing: (28mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, -0.6), [data-gradient update], shape: fletcher.shapes.rect, corner-radius: 3pt,
    inset: 7pt, stroke: 1pt + TEAL)
  node((0, 0.6), [parameter shrinkage], shape: fletcher.shapes.rect, corner-radius: 3pt,
    inset: 7pt, stroke: 1pt + ACC)
  node((1, 0), $theta_(t+1)$, radius: 9mm, stroke: 1.1pt + INK)
  edge((0, -0.6), (1, 0), "-|>", stroke: 1pt + TEAL)
  edge((0, 0.6), (1, 0), "-|>", stroke: 1pt + ACC)
}))

#pause
#note[Decoupling makes the intended shrinkage independent of Adam's coordinate-wise rescaling. Bias parameters and normalization gains are commonly excluded, but the correct choice is model- and recipe-dependent.]

== Predict: should equal parameters shrink unequally? #Q

Return to dense $d$ and rare $r$. Suppose

$ theta=(1,1), quad hat(v)=(16,0.16), quad eta=0.01, quad lambda=0.1. $

Hold $hat(v)$ fixed to isolate the geometry. If an L2 gradient $lambda theta$ is sent through Adam's adaptive denominator, its approximate contribution is

$ -eta lambda theta / sqrt(hat(v)). $

#mcq(
  [How do the two coordinates' shrinkage contributions compare?],
  [both are $-0.001$],
  [dense shrinks $10 times$ more],
  [rare shrinks $10 times$ more],
  [both are $-0.01$],
)

== Answer: preconditioning changes the intended penalty #A

#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([L2 inside adaptive update], [
    dense: $-0.01(0.1)/4=-0.00025$
    #v(4pt)
    rare: $-0.01(0.1)/0.4=-0.0025$
  ], color: GREEN),
  hairline([Decoupled AdamW], [
    dense: $-eta lambda theta_d=-0.001$
    #v(4pt)
    rare: $-eta lambda theta_r=-0.001$
  ], color: ACC),
))

#mcq-answer([C], [the rare coordinate shrinks $10 times$ more], [coupled L2 is filtered by the coordinate-wise preconditioner; AdamW applies the intended proportional shrinkage separately.])

#caption[The coupled calculation holds $hat(v)$ fixed for interpretation. In an actual coupled optimizer, adding $lambda theta$ also changes the moment estimates.]

== Compare mechanisms, not brand names

#align(center, table(
  columns: 4, stroke: 0.45pt + MUTED, inset: (x: 9pt, y: 6pt),
  align: (left, center, center, left),
  table.header([method], [direction memory], [scale memory], [mechanism]),
  [SGD], [—], [—], [current minibatch gradient],
  [momentum], [$m_t$], [—], [smooth direction],
  [AdaGrad], [—], [$sum g_t^2$], [protect rare coordinates; rate only shrinks],
  [RMSProp], [—], [EMA of $g_t^2$], [adapt to recent scale],
  [Adam / AdamW], [$m_t$], [$v_t$], [combine both; optionally decouple decay],
))

#pause
#warning[There is no universally best optimizer. Any comparison also changes learning rate, schedule, update budget, batch size, and often regularization.]

== Same ravine, four explicitly different recipes #V

#let path-panel(title, path) = contour(ad.fn2(ravine),
  xlim: (-2.7, 2.7), ylim: (-1.0, 1.0), samples: 52, levels: 10,
  size: (39mm, 39mm), color: MUTED, paths: (path,),
  marks: ((0, 0, [], RED),), x-label: [$x$], y-label: [$y$], title: title,
  theme: geometry-theme,
)
#align(center, stack(dir: ltr, spacing: 4.5mm,
  path-panel([GD · $eta=0.014$], gd(ravine-grad, ravine-start, lr: 0.014, steps: 55)),
  path-panel([Momentum · norm. $eta=0.06$], momentum(ravine-grad, ravine-start, lr: 0.006, steps: 55)),
  path-panel([RMSProp · $eta=0.06$], rmsprop(ravine-grad, ravine-start, lr: 0.06, steps: 55)),
  path-panel([Adam · $eta=0.10$], adam(ravine-grad, ravine-start, lr: 0.10, steps: 55)),
))

#caption[Same loss and start. Momentum's normalized $eta=0.06$ equals buffer
rate $0.006$. Method-specific rates make this a mechanism illustration—not a ranking.]

== Starting recipes depend on the model regime

#set text(size: 15.5pt)
#align(center, table(
  columns: (38mm, 57mm, 1fr), stroke: 0.45pt + MUTED, inset: (x: 6pt, y: 3pt),
  align: (left, left, left),
  table.header([regime], [reasonable first baseline], [what to test next]),
  [CNN from scratch], [established SGD + momentum or AdamW recipe], [sweep rate/schedule; compare equal update budgets],
  [Transformer / sparse model], [established AdamW, normalization, and schedule recipe], [test peak rate, warmup, decay, exclusions],
  [pretrained fine-tune], [task or adapter recipe; lower rate than pretraining], [test forgetting and layer/adapter rates],
))

#note[Starting hypotheses, not universal defaults: a validated architecture recipe plus a controlled rate sweep outranks a generic number.]

== Make an optimizer comparison interpretable

#align(center, table(
  columns: 2, stroke: 0.45pt + MUTED, inset: (x: 12pt, y: 6pt),
  align: (left, left),
  table.header([discipline], [record or test]),
  [record the complete recipe], [optimizer, base rate, betas, $epsilon$, decay, schedule, batch, update count, seeds],
  [inspect the mechanism], [learning rate, gradient norm, update norm, train loss, validation metric],
  [verify before scaling], [one-batch overfit and a controlled learning-rate sweep],
))

#pause
#warning[Changing optimizer and learning-rate schedule together does not identify which change caused the result.]

// ═════════════════════ LEARNING-RATE CLOCK ═════════════════════
= The schedule changes the step scale over time

== A fixed rate asks one number to do two jobs

#align(center, diagram(spacing: (31mm, 14mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), [early training], shape: fletcher.shapes.rect, corner-radius: 3pt,
    inset: 7pt, stroke: 1pt + TEAL)
  node((1, 0), [late training], shape: fletcher.shapes.rect, corner-radius: 3pt,
    inset: 7pt, stroke: 1pt + ACC)
  node((0, 0.75), text(size: 11pt)[make useful progress], stroke: none, fill: none)
  node((1, 0.75), text(size: 11pt)[refine without bouncing], stroke: none, fill: none)
  edge((0, 0), (1, 0), "-|>", stroke: 0.9pt + MUTED, label: [training updates], label-sep: 6pt)
}))

#pause
#punch[The optimizer chooses a direction and normalization; the schedule multiplies the overall step scale.]

== Warmup limits abrupt early updates

For $T_w$ warmup updates, one simple convention is

$ eta_t=eta_max (t+1)/T_w, quad t=0,dots,T_w-1. $

#pause
#align(center, lines(
  fn: (t => calc.min((t + 1)/10, 1.0),),
  domain: (0, 30), samples: 61, colors: (ACC,), markers: false,
  x-label: [optimizer update], y-label: [relative rate], size: (112mm, 46mm),
))

#pause
#warning[Warmup reduces the initial applied scale; it does not repair a fundamentally unstable peak rate, bad data, or exploding activations.]

== Cosine decay turns progress into refinement

After warmup, let $u=(t-T_w)/(T-T_w)$ be progress from $0$ to $1$:

$ eta_t=eta_min + 1/2(eta_max-eta_min)(1+cos(pi u)). $

#pause
#let sched-total = 100
#let sched-warm = 10
#let sched-warmcos = t => if t < sched-warm { calc.min((t + 1) / sched-warm, 1.0) } else {
  0.5 * (1 + calc.cos(calc.pi * (t - sched-warm) / (sched-total - sched-warm)))
}
#align(center, lines(
  fn: (sched-warmcos,), domain: (0, 100), samples: 101, colors: (ACC,), markers: false,
  hlines: ((1, [$eta_max$], MUTED),),
  x-label: [optimizer update], y-label: [relative rate], size: (120mm, 45mm),
))

#caption[Warmup changes the beginning; cosine decay changes the remainder. Neither replaces optimizer state.]

== Calculate the clock before using it #D

Let $eta_max=0.1$, $eta_min=0$, $T_w=10$, and $T=100$.

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 14pt, y: 7pt),
  align: (center, left, center),
  table.header([update], [phase], [rate]),
  [$t=0$], [first warmup update], [$0.1(1/10)=0.01$],
  [$t=4$], [fifth warmup update], [$0.1(5/10)=0.05$],
  [$t=9$], [last warmup update], [$0.10$],
  [$t=55$], [halfway through decay], [$0.05$],
  [$t=100$], [unexecuted endpoint], [$0$],
))

#pause
#note[With this convention, a 100-update run applies $eta_0,dots,eta_99$;
$eta_100=0$ is the unexecuted endpoint. An off-by-one in warmup can silently
change the first update from a small nonzero rate to no update at all.]

== The schedule must count the same update boundary as the optimizer

With gradient accumulation over $K$ microbatches:

#align(center, diagram(spacing: (22mm, 13mm), node-stroke: 0.9pt + INK, node-fill: white, {
  for i in range(3) {
    node((0, i - 1), [microbatch #str(i+1)], shape: fletcher.shapes.rect,
      corner-radius: 3pt, inset: 5pt, stroke: 0.9pt + INK)
    node((1, i - 1), $g_#(i+1)$, radius: 5mm, stroke: 0.9pt + BLUE)
    edge((0, i - 1), (1, i - 1), "-|>", stroke: 0.8pt + BLUE)
    edge((1, i - 1), (2, 0), "-|>", stroke: 0.8pt + MUTED)
  }
  node((2, 0), [accumulate], shape: fletcher.shapes.rect, corner-radius: 3pt,
    inset: 7pt, stroke: 1pt + TEAL)
  node((3, 0), [one optimizer step], shape: fletcher.shapes.rect, corner-radius: 3pt,
    inset: 7pt, stroke: 1.1pt + ACC)
  edge((2, 0), (3, 0), "-|>", stroke: 1pt + ACC)
}))

#pause
#warning[Advancing an update-based scheduler on every microbatch makes its clock run $K times$ too fast.]

== One reliable PyTorch update boundary #I

#codebox(size: 18pt)[```python
for x, y in loader:
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
    scheduler.step()       # for an update-based schedule
```]

#pause
#caption[Log gradient/update norms at this boundary; clip only under a stated diagnostic or stability policy.]

== Return to the quadratic: schedule and path together #I

#interbox(link-to: NB + "01_gd_quadratic.ipynb")[]

Now complete the warmup–cosine section:

- predict the time-varying multiplier $1-2eta_t$;
- compare signed error and loss, not loss alone;
- test an overly large peak rate;
- distinguish transient growth from eventual recovery under decay.

== Inspect optimizer state, not only a final loss #I

#interbox(link-to: NB + "06_optimizer_comparison_pytorch.ipynb")[]

Use the common ravine and start point to:

- calculate AdaGrad, RMSProp, and Adam state by hand;
- inspect $m_t$, $v_t$, and actual parameter updates separately;
- compare paths only with each method's rate visible;
- reproduce the mechanisms with deterministic PyTorch tensors.

// ═══════════════════════ DIAGNOSTICS ═══════════════════════
= What breaks: symptom → suspect → test

== Read the symptom before changing the optimizer

#align(center, table(
  columns: (35mm, 1fr, 1fr), stroke: 0.45pt + MUTED, inset: (x: 8pt, y: 6pt),
  align: (left, left, left),
  table.header([symptom], [first suspects], [controlled test]),
  [loss becomes non-finite], [rate, bad batch, overflow, exploding gradients], [log batch ID, rate, gradient norm; rerun same batch],
  [bounded zigzag], [rate too high for a steep direction], [lower only the rate; inspect signed updates],
  [no progress], [rate too low, dead path, no gradients, saturated units], [overfit one batch; inspect per-layer gradient norms],
  [train improves, validation worsens], [generalization—not necessarily optimization], [hold optimizer fixed; test regularization or data],
))

#pause
#punch[Change one suspected cause at a time. “Try Adam” is an experiment, not a diagnosis.]

== Gradient clipping caps a dangerous update #D #OPT

Global-norm clipping replaces $g$ by

$ tilde(g)=g min(1,c/norm(g)). $

#pause
For $g=(6,8)$, $norm(g)=10$. With $c=5$:

$ tilde(g)=(6,8)(5/10)=(3,4), quad norm(tilde(g))=5. $

#pause
#warning[Clipping preserves direction and caps magnitude. It can prevent a single catastrophic step; it does not make an unstable learning rate appropriate or repair chronic exploding gradients.]

== A loss spike deserves cause-relevant measurements

#align(center, diagram(spacing: (25mm, 15mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), [loss spike], shape: fletcher.shapes.rect, corner-radius: 3pt,
    inset: 7pt, stroke: 1.1pt + RED)
  node((1, -1.0), [batch ID + data], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 5pt, stroke: 0.9pt + INK)
  node((1, 0), [gradient norm], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 5pt, stroke: 0.9pt + BLUE)
  node((1, 1.0), [learning rate], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 5pt, stroke: 0.9pt + ACC)
  node((2, -0.5), [activation scale], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 5pt, stroke: 0.9pt + TEAL)
  node((2, 0.5), [optimizer state], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 5pt, stroke: 0.9pt + GREEN)
  for p in ((1,-1.0),(1,0),(1,1.0),(2,-0.5),(2,0.5)) { edge((0,0), p, "-|>", stroke: 0.75pt + MUTED) }
}))

#pause
#note[For claims about generalization, use held-out performance. Raw “sharp versus flat” pictures depend on parameterization and do not establish generalization by themselves.]

== Checkpoint: choose the smallest informative intervention #Q

A run is stable for 2,000 updates. The loss then spikes on one minibatch, gradient norm jumps $40 times$, and the current rate is unchanged.

#mcq(
  [What should you do first?],
  [switch immediately from AdamW to SGD],
  [rerun and inspect that batch plus gradient/activation scales],
  [increase weight decay],
  [restart with a larger batch and change nothing else],
)

== Answer: preserve evidence before changing the algorithm #A

#mcq-answer([B], [inspect and reproduce the triggering batch], [the correlated gradient-norm jump points to data or forward/backward scale; changing optimizer first destroys the cleanest evidence.])

#pause
#note[If the batch is valid and the spike is reproducible, then test one intervention—rate, clipping threshold, normalization, or architecture—while holding the rest fixed.]

// ═══════════════════════ SYNTHESIS ═══════════════════════
= Revisit, compress, hand off

== Revisit the opening stream

$ g_1=(4,0), quad g_2=(4,0), quad g_3=(4,0.4). $

#align(center, table(
  columns: (32mm, 1fr, 1fr), stroke: 0.45pt + MUTED, inset: (x: 9pt, y: 6pt),
  align: (left, left, left),
  table.header([method], [what history says], [step-3 consequence]),
  [SGD], [nothing], [raw $10 times$ scale gap remains],
  [momentum], [direction has been persistent for $d$], [smooths signs; does not identify rarity],
  [AdaGrad], [$d$ has accumulated much more squared activity], [rare coordinate gets the larger normalized step],
  [RMSProp], [recent squared activity], [same idea without permanent accumulation],
  [Adam], [direction and scale histories], [rare signal is lifted, then tempered by its direction history],
))

#pause
#punch[The opening answer was not “Adam.” It was the information Adam needs: coordinate-wise squared-gradient history.]

== Every optimizer answers three questions

#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([Direction], [current $g_t$ or smoothed $m_t$?], color: TEAL),
  hairline([Scale], [global $eta$ or coordinate-wise $v_t$?], color: GREEN),
  hairline([Clock], [fixed, warmup, decay—and which updates count?], color: ACC),
))

#pause
#v(12pt)
#align(center, text(size: 21pt, weight: 650)[
  #text(fill: BLUE)[measure $g_t$]
  #h(8pt) $arrow.r$ #h(8pt)
  #text(fill: TEAL)[remember direction]
  #h(8pt) $+$ #h(8pt)
  #text(fill: GREEN)[remember scale]
  #h(8pt) $arrow.r$ #h(8pt)
  #text(fill: ACC)[apply a scheduled step]
])

== Practical exit ticket

Before trusting a training run, can you answer all five?

#align(center, grid(columns: (1fr, 1fr), gutter: 14pt,
  note([*1.* What exactly does backprop measure?], color: BLUE),
  note([*2.* What state does the optimizer store?], color: TEAL),
  note([*3.* What is each coordinate's effective scale?], color: GREEN),
  note([*4.* Which event advances the schedule?], color: ACC),
  note([*5.* Which logged measurement would falsify your diagnosis?], color: RED),
  note([*Reproduce:* seed, batch, update count, complete recipe.], color: INK),
))

#pause
#punch[If you cannot answer these, the loss curve is a symptom without a mechanism.]

== Primary references and provenance #OPT

#set text(size: 16pt)

- Duchi, Hazan & Singer (2011), #link("https://jmlr.org/papers/v12/duchi11a.html")[*Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*].
- Tieleman & Hinton (2012), *Lecture 6.5—RMSProp*, Neural Networks for Machine Learning.
- Kingma & Ba (2015), #link("https://arxiv.org/abs/1412.6980")[*Adam: A Method for Stochastic Optimization*].
- Loshchilov & Hutter (2019), #link("https://arxiv.org/abs/1711.05101")[*Decoupled Weight Decay Regularization*].
- Vaswani et al. (2017), #link("https://arxiv.org/abs/1706.03762")[*Attention Is All You Need*]—a prominent warmup-and-decay recipe, not a claim that warmup is universally required.

#v(8pt)
#note[All trajectories in this deck are generated from the loss and update rules shown beside them. Notebook comparisons are deterministic mechanism tests, not benchmark claims.]

#focus-slide[
  Adam gives every parameter a history-dependent effective learning rate.
  #v(12pt)
  #set text(size: 20pt)
  Next: make gradients and activations survive depth—initialization, normalization, and residual paths.
]
