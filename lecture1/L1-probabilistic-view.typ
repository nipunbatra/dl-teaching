// Where Deep Learning Losses Come From — Lecture 1 · ES 667 Deep Learning
// Typst + shared metropolis theme. Compile from the repo root:
//   typst compile --root . lecture1/L1-probabilistic-view.typ
//   typst compile --root . --input handout=true lecture1/L1-probabilistic-view.typ
// All theme/palette/helpers/diagrams live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Where Deep Learning Losses Come From],
  subtitle: [Likelihood, cross-entropy, and regularization from probability],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"

#title-slide()

// ══════════════════════════════ PART I ══════════════════════════════
== Three familiar losses, one probabilistic origin

You have already met these losses:
#pause
$ cal(L)_"reg" = 1/n sum_i (y_i - hat(y)_i)^2 quad #text(fill: MUTED, size: 15pt)[(regression)] $
#pause
$ cal(L)_"BCE" = -1/n sum_i [y_i log p_i + (1-y_i) log(1-p_i)] quad #text(fill: MUTED, size: 15pt)[(binary classification)] $
#pause
$ cal(L)_"total" = cal(L)_"data" + lambda norm(bold(theta))_2^2 quad #text(fill: MUTED, size: 15pt)[(regularized)] $
#pause
#notebox[*Objective.* Derive each term from an explicit likelihood or prior.]

== The map for today
#stag(V)

Two probabilistic assumptions determine the two halves of the objective.
#v(4pt)
#align(center, text(size: 17pt, fill: MUTED)[
  We will define the predictive distribution $p_(bold(theta))(y|bold(x))$, likelihood, log-likelihood, and NLL; \
  then the prior $p(bold(theta))$ and MAP.
])
#pause
#v(6pt)
#align(center, diagram(
  spacing: (17mm, 9mm), node-stroke: 0.9pt,
  {
    node((0,0), [outputs\ $p_(bold(theta))(y|bold(x))$], shape: fletcher.shapes.rect, fill: rgb("#EFEEEB"), stroke: INK, inset: 7pt)
    node((2,0), [parameters\ $p(bold(theta))$], shape: fletcher.shapes.rect, fill: rgb("#EFEEEB"), stroke: INK, inset: 7pt)
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
= Probability primer

== A point prediction summarizes a distribution
#stag(V)

#grid(
  columns: (1fr, 1fr), column-gutter: 28pt,
  [
    #text(font: "IBM Plex Mono", size: 12pt, fill: TEAL, weight: 700)[REGRESSION · $Y in RR$]
    #v(4pt)
    #text(size: 14.5pt, fill: MUTED)[Point prediction: mean of $Y | bold(X)=bold(x)$]
    #v(1pt)
    #text(size: 18pt)[$hat(y)(bold(x)) = bb(E)_(bold(theta))[Y | bold(X)=bold(x)] = mu_(bold(theta))(bold(x))$]
    #v(5pt)
    #text(size: 14.5pt, fill: MUTED)[Full predictive distribution]
    #v(1pt)
    #align(center, text(size: 16.5pt)[
      $Y | bold(X) = bold(x)$ \
      $tilde cal(N)(mu_(bold(theta))(bold(x)), sigma_(bold(theta))^2(bold(x)))$
    ])
    #v(4pt)
    #align(center, lines(
      fn: y => dist.pdf(dist.normal(mu: 0.0, sigma: 1.0), y),
      domain: (-3.2, 3.2), samples: 100, markers: false, colors: (ACC,),
      fill-under: (0, ACC.transparentize(82%)), y-ticks: false,
      vlines: ((0.0, [$mu_(bold(theta))(bold(x))$], MUTED),),
      x-label: [$y$], size: (51mm, 20mm),
    ))
  ],
  [
    #text(font: "IBM Plex Mono", size: 12pt, fill: TEAL, weight: 700)[BINARY CLASSIFICATION · $Y in {0,1}$]
    #v(4pt)
    #text(size: 14.5pt, fill: MUTED)[Point prediction: mode of $Y | bold(X)=bold(x)$]
    #v(1pt)
    #text(size: 18pt)[$hat(y)(bold(x)) = bold(1)[p_(bold(theta))(bold(x)) >= 0.5]$]
    #v(5pt)
    #text(size: 14.5pt, fill: MUTED)[Full predictive distribution]
    #v(1pt)
    #align(center, text(size: 16.5pt)[
      $Y | bold(X) = bold(x)$ \
      $tilde "Bernoulli"(p_(bold(theta))(bold(x)))$
    ])
    #v(4pt)
    #align(center, bars(
      (0.25, 0.75), labels: ([$Y=0$], [$Y=1$]), highlight: 1,
      size: (48mm, 20mm), title: [example: $p_(bold(theta))(bold(x)) = 0.75$],
    ))
  ],
)

#place(bottom + center, dy: -2pt, block(
  fill: rgb("#FDECD6"), inset: (x: 12pt, y: 6pt), radius: 4pt,
  stroke: 1.2pt + ACC,
  text(size: 17pt, weight: 600)[Neural network: $bold(x) arrow.r$ distribution parameters. Prediction: $hat(y) =$ mean or mode.],
))

== Discrete distributions assign probability to outcomes
#stag(V)

#two(r: (1.2fr, 1fr),
  [
    The *support* $cal(S)_Y$ is the set of values that $Y$ can take.
    #v(5pt)
    #text(size: 17pt, fill: MUTED)[Probability mass function (PMF)]
    #v(2pt)
    $ p_Y(y) = P(Y = y), quad y in cal(S)_Y $
    #v(4pt)
    $ p_Y(y) >= 0, quad sum_(y in cal(S)_Y) p_Y(y) = 1 $
    #v(8pt)
    #notebox[
      *Examples.* Bernoulli: ${0,1}$ · categorical: ${1, dots, K}$ · Poisson: ${0,1,2,dots}$
    ]
  ],
  [
    #align(center, bars(
      (0.25, 0.75), labels: ([$Y=0$], [$Y=1$]), highlight: 1,
      size: (53mm, 37mm), title: [Bernoulli$(0.75)$ PMF],
    ))
    #v(5pt)
    #align(center, text(size: 15pt, fill: MUTED)[each bar is an outcome probability])
  ],
)
#v(7pt)
#result[Exact values can have nonzero probability: here $P(Y=1)=0.75$.]

== Continuous distributions assign density across a support
#stag(V)

#two(r: (1.12fr, 1fr),
  [
    The *support* may be bounded or unbounded:
    #v(3pt)
    $ cal(S)_Y = [a,b] $ for a Uniform variable; $cal(S)_Y = RR$ for a Gaussian.
    #v(7pt)
    #text(size: 17pt, fill: MUTED)[Probability density function (PDF)]
    #v(2pt)
    $ p_Y(y) >= 0, quad integral_(cal(S)_Y) p_Y(y) dif y = 1 $
  ],
  [
    #align(center, lines(
      fn: y => dist.pdf(dist.normal(mu: 0.0, sigma: 1.0), y),
      domain: (-4, 4), samples: 120, markers: false, colors: (INK,),
      fill-under: (0, ACC.transparentize(70%)), y-ticks: false,
      x-label: [$y$], y-label: [$p_Y(y)$], size: (59mm, 39mm),
    ))
    #v(4pt)
    #align(center, text(size: 15pt, fill: MUTED)[the total area under a PDF is 1])
  ],
)
#place(bottom + center, dy: -2pt,
  result[For continuous $Y$, probability comes from *area* under the density.])

== Density is not probability
#stag(V)

For a continuous $Y$, probability comes from *area*:
$ P(a <= Y <= b) = integral_a^b p_Y(y) dif y $
#pause
#alertbox[$p_Y(y) != P(Y=y)$ — a density can *exceed 1*; only _integrals_ are probabilities.]
// native: a real N(0,1) density; the interval [a,b] shaded to show the area
#let _dg = dist.normal(mu: 0.0, sigma: 1.0)
#align(center, lines(
  fn: x => dist.pdf(_dg, x), domain: (-4, 4), samples: 120,
  fill-under: (from: -1, to: 1, color: ACC.transparentize(60%)),
  markers: false, colors: (INK,), y-ticks: false,
  vlines: ((-1.0, [$a$], MUTED), (1.0, [$b$], MUTED)),
  x-label: [$y$], y-label: [$p_Y(y)$],
  size: (68mm, 30mm),
))
#align(center, text(size: 15pt, fill: MUTED)[the shaded area is $P(a <= Y <= b)$])

== Uniform distribution: support matters
#stag(V)

#two(r: (1.15fr, 1fr),
  [$ Y tilde cal(U)(a, b) quad p_Y(y) = cases(1\/(b-a) & quad a <= y <= b, 0 & quad "otherwise") $
   #v(4pt)
   #notebox[The support matters: outside $[a,b]$, the density is exactly $0$.]],
  align(center, lines(
    // the TRUE Uniform(a,b) density — a box, zero outside its support
    fn: x => dist.pdf(dist.uniform(a: -1.0, b: 2.0), x),
    domain: (-3, 4), samples: 140, markers: false, colors: (INK,),
    vlines: ((-1.0, [$a$], MUTED), (2.0, [$b$], MUTED)), y-ticks: false,
    x-label: [$y$], y-label: [$p_Y(y)$],
    size: (58mm, 40mm),
  )),
)

== Gaussian distribution
#stag(V)

$ Y tilde cal(N)(mu, sigma^2) quad quad p(y) = 1/sqrt(2 pi sigma^2) exp(-(y-mu)^2/(2 sigma^2)) $
#pause
// native: three real Normal densities — μ shifts location, σ sets the scale/height
#align(center, lines(
  fn: (x => dist.pdf(dist.normal(mu: 0.0, sigma: 1.0), x),
       x => dist.pdf(dist.normal(mu: 0.0, sigma: 1.8), x),
       x => dist.pdf(dist.normal(mu: -1.5, sigma: 1.0), x)),
  domain: (-6, 6), samples: 120, markers: false,
  colors: (INK, ACC, TEAL),
  labels: ([$mu=0, sigma=1$], [$mu=0, sigma=1.8$], [$mu=-1.5, sigma=1$]),
  legend: "tr", y-ticks: false,
  annotations: ((-3.0, 0.018, [$-3$]), (0.0, 0.018, [$0$]), (3.0, 0.018, [$3$])),
  x-label: [$y$], y-label: [$p(y)$],
  size: (82mm, 46mm),
))
#align(center)[Only two knobs: *location* $mu$ and *scale* $sigma$.]

// ══════════════════════════════ PART III ══════════════════════════════
= From probability to likelihood

== Read the model notation one symbol at a time
#stag(Q)

#align(center, text(size: 28pt, weight: 600)[$p_(bold(theta))(y | bold(x))$])
#v(7pt)
#align(center, table(
  columns: (31mm, 1fr), stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 4pt),
  align: (center, left),
  [$bold(X)$ / $bold(x)$], [$bold(X)$ is the random input vector; $bold(x)$ is one observed input],
  [$Y$ / $y$], [$Y$ is the random target; $y$ is one possible or observed value],
  [$bold(theta)$], [the *parameter vector* — the model's learned weights and biases],
  [$p_(bold(theta))(y | bold(x))$], [the probability (discrete $Y$) or density (continuous $Y$) assigned to $y$, *given* $bold(x)$],
))
#place(bottom + center, dy: -2pt,
  result[Read aloud: “model $bold(theta)$'s probability or density for $y$, given $bold(x)$.”])

== Prediction fixes the model and varies the outcome
#stag(V)

Example: $Y in {0,1}$. Fix the input $bold(x)^star$ and one parameter setting $bold(theta)_A$.
#v(8pt)
#two(r: (1.1fr, 1fr),
  [
    $ p_(bold(theta)_A)(Y=0 | bold(x)^star) = 0.20 $
    #v(5pt)
    $ p_(bold(theta)_A)(Y=1 | bold(x)^star) = 0.80 $
    #v(8pt)
    $ 0.20 + 0.80 = 1 $
    #v(8pt)
    #notebox[We vary the possible value of $Y$ while keeping $bold(theta)_A$ and $bold(x)^star$ fixed.]
  ],
  [
    #align(center, bars(
      (0.20, 0.80), labels: ([$Y=0$], [$Y=1$]), highlight: 1,
      size: (55mm, 39mm), title: [predictive distribution],
    ))
  ],
)
#place(bottom + center, dy: -2pt,
  result[$hat(y)(bold(x)^star) = arg max_y p_(bold(theta)_A)(y | bold(x)^star) = 1$])

== Likelihood fixes data and varies the model
#stag(D)

Now observe $y^star = 1$ for the input $bold(x)^star$, so $cal(D) = {(bold(x)^star, 1)}$.
#v(5pt)
$ L(bold(theta); cal(D)) = p_(bold(theta))(y^star = 1 | bold(x)^star) $
#v(7pt)
#two(r: (1.08fr, 1fr),
  [
    *Candidate A:* $L(bold(theta)_A; cal(D)) = 0.80$ \
    *Candidate B:* $L(bold(theta)_B; cal(D)) = 0.30$
    #v(10pt)
    $ L(bold(theta)_A; cal(D)) > L(bold(theta)_B; cal(D)) $
    #v(7pt)
    #notebox[$bold(theta)_A$ makes the observed label $y^star=1$ less surprising.]
  ],
  [
    #align(center, bars(
      (0.80, 0.30), labels: ([$bold(theta)_A$], [$bold(theta)_B$]), highlight: 0,
      size: (55mm, 35mm), title: [likelihood of the fixed observation],
    ))
  ],
)
#place(bottom + center, dy: -2pt,
  alertbox[Likelihood is a *score as a function of* $bold(theta)$; the scores need not sum to $1$ over parameter settings.])

== Coin flips: a likelihood you can compute
#stag(D)

#let _coin-like(theta) = calc.pow(theta, 4) * calc.pow(1 - theta, 6)
#two(r: (1.08fr, 1fr),
  [
    *Observed:* $H, H, T, T, T, H, H, T, T, T$ \
    Hence $n_H = 4$ and $n_T = 6$.
    #v(7pt)
    *Model:* $p(H | theta) = theta$, $p(T | theta) = 1 - theta$.
    #v(7pt)
    Independence gives
    $ L(theta) = product_i p(x_i | theta) $
    $ = theta^(n_H) (1 - theta)^(n_T) $
    $ = theta^4 (1 - theta)^6 $
  ],
  [
    #align(center, lines(
      fn: _coin-like, domain: (0.0, 1.0), samples: 160,
      markers: false, colors: (ACC,), fill-under: (0, ACC.transparentize(84%)),
      points: ((0.4, _coin-like(0.4)), (0.7, _coin-like(0.7))),
      vlines: ((0.4, [$0.4$], ACC), (0.7, [$0.7$], MUTED)),
      x-label: [$theta$], y-label: [$L(theta)$], y-ticks: false,
      size: (59mm, 37mm),
    ))
    #v(4pt)
    #align(center, text(size: 14.5pt)[
      $L(0.4) approx 0.00119$ \
      $L(0.7) approx 0.000175$ \
      #text(fill: MUTED)[about $6.8 times$ larger at $theta=0.4$]
    ])
  ],
)
#place(bottom + center, dy: -2pt,
  result[The likelihood peaks at $theta = 0.4$ — the observed head fraction $4/10$.])

== Coin flips: calculate the log-likelihood
#stag(D)

The *log-likelihood* is the logarithm of the likelihood — not a different model:
#v(4pt)
$ ell(theta) := log L(theta) = log [theta^4 (1-theta)^6] = 4 log theta + 6 log(1-theta) $
#v(9pt)
#two(
  notebox[
    *Candidate $theta=0.4$* \
    $L(0.4) approx 0.00119$ \
    $ell(0.4) = 4 log 0.4 + 6 log 0.6 approx -6.73$
  ],
  notebox[
    *Candidate $theta=0.7$* \
    $L(0.7) approx 0.000175$ \
    $ell(0.7) = 4 log 0.7 + 6 log 0.3 approx -8.65$
  ],
)
#v(7pt)
#align(center, text(size: 15pt, fill: MUTED)[Likelihoods below $1$ have negative logs; among them, $-6.73$ is the larger value.])
#place(bottom + center, dy: -2pt,
  result[The observed sequence has higher log-likelihood at $theta=0.4$.])

== Why logs? The maximizing parameter does not change
#stag(D)

The logarithm is *strictly increasing* on positive numbers:
$ a > b > 0 quad ==> quad log a > log b $
#v(6pt)
Therefore it preserves the ordering of all positive likelihood values:
#v(7pt)
#align(center, block(
  fill: rgb("#EFEEEB"), stroke: 0.8pt + INK, radius: 4pt,
  inset: (x: 18pt, y: 9pt),
  align(center, [
    $L(0.4) > L(0.7)$ \
    #text(size: 14pt, fill: MUTED)[apply the increasing function $log$] \
    $log L(0.4) > log L(0.7)$
  ]),
))
#v(8pt)
$ arg max_theta L(theta) = arg max_theta log L(theta) $
#place(bottom + center, dy: -2pt,
  result[Both likelihood and log-likelihood rank $theta=0.4$ above $theta=0.7$.])

== Poisson example: logging preserves the peak
#stag(V)

#let _pois-like(lambda) = calc.exp(-lambda) * calc.pow(lambda, 3) / 6
#let _pois-loglike(lambda) = 3 * calc.ln(lambda) - lambda - calc.ln(6)

Suppose $Y tilde "Poisson"(lambda)$ and the observed count is $y=3$.
#v(4pt)
#two(r: (1fr, 1fr),
  [
    #align(center, text(size: 16pt, fill: ACC, weight: 700)[LIKELIHOOD])
    #align(center, text(size: 17pt)[$L(lambda) = e^(-lambda) lambda^3 / 3!$])
    #v(3pt)
    #align(center, lines(
      fn: _pois-like, domain: (0.05, 8.0), samples: 160,
      markers: false, colors: (ACC,), fill-under: (0, ACC.transparentize(84%)),
      points: ((3.0, _pois-like(3.0)),),
      vlines: ((3.0, [$lambda=3$], ACC),),
      x-label: [$lambda$], y-label: [$L(lambda)$], y-ticks: false,
      size: (61mm, 38mm),
    ))
  ],
  [
    #align(center, text(size: 16pt, fill: TEAL, weight: 700)[LOG-LIKELIHOOD])
    #align(center, text(size: 17pt)[$log L(lambda) = 3 log lambda - lambda - log 3!$])
    #v(3pt)
    #align(center, lines(
      fn: _pois-loglike, domain: (0.05, 8.0), samples: 160,
      markers: false, colors: (TEAL,),
      points: ((3.0, _pois-loglike(3.0)),),
      vlines: ((3.0, [$lambda=3$], TEAL),),
      x-label: [$lambda$], y-label: [$log L(lambda)$], y-ticks: false,
      size: (61mm, 38mm),
    ))
  ],
)
#place(bottom + center, dy: -2pt,
  result[Both curves peak at $hat(lambda)_"MLE"=3$: the log changes the vertical scale, not the maximizing parameter.])

== Why logs? Products become stable sums
#stag(D)

For independent observations,
$ L(bold(theta)) = product_i p_(bold(theta))(y_i | bold(x)_i) quad ==> quad log L(bold(theta)) = sum_i log p_(bold(theta))(y_i | bold(x)_i) $
#v(10pt)
*Worked numerical example.* Suppose 200 observed outcomes each receive probability $0.01$.
#v(5pt)
#align(center, table(
  columns: (37mm, 1fr), stroke: 0.5pt + MUTED,
  inset: (x: 10pt, y: 6pt), align: (center, left),
  [*Direct product*], [$L = 0.01^200 = 10^(-400) < 4.94 times 10^(-324)$ → underflows to $0$],
  [*Log space*], [$log L = 200 log(0.01) approx -921.03$ → finite and representable],
))
#v(5pt)
#align(center, text(size: 13.5pt, fill: MUTED)[
  Float64 range: $plus.minus 1.80 times 10^308$ · smallest positive normal $2.23 times 10^(-308)$ · smallest subnormal $4.94 times 10^(-324)$.
])
#place(bottom + center, dy: -2pt,
  result[Logs make optimization easier *and* prevent numerical underflow.])

== Coin flips: solve for the stationary point
#stag(D)

For $L(theta) = theta^(n_H)(1-theta)^(n_T)$, the log-likelihood is
$ log L(theta) = n_H log theta + n_T log(1 - theta) $
#pause
Set the derivative to zero:
$ (dif)/(dif theta) log L(theta) = n_H / theta - n_T / (1 - theta) = 0 $
#pause
$ ==> theta^star = n_H / (n_H + n_T) = 4/10 = 0.4 $
#pause
#result[$theta^star=0.4$ is a *candidate* optimum. Next, check the curvature.]

== The second derivative confirms a maximum
#stag(D)

Differentiate the log-likelihood again:
$ (dif^2)/(dif theta^2) log L(theta) = -n_H/theta^2 - n_T/(1-theta)^2 $
#pause
For $n_H,n_T > 0$ and $0 < theta < 1$, both terms are negative:
$ (dif^2)/(dif theta^2) log L(theta) < 0 $
#pause
At the stationary point $theta^star=0.4$,
$ -4/(0.4)^2 - 6/(0.6)^2 = -25 - 16.67 approx -41.67 < 0 $
#place(bottom + center, dy: -2pt,
  result[Strict concavity $+$ stationary point $arrow.r$ unique global maximum: $hat(theta)_"MLE" = theta^star = 4/10 = 0.4$.])

== Maximum likelihood estimation

$ hat(bold(theta))_"MLE" = arg max_(bold(theta)) p(cal(D) | bold(theta)) $
#pause
#notebox[*Interpretation.* Choose the parameter values that assign the *highest likelihood* to the observed dataset.]

== Maximizing likelihood = minimizing NLL
#stag(D)

$ arg max_(bold(theta)) product_i p_(bold(theta))(y_i | bold(x)_i) = arg max_(bold(theta)) sum_i log p_(bold(theta))(y_i | bold(x)_i) $
#pause
$ = arg min_(bold(theta)) (-sum_i log p_(bold(theta))(y_i | bold(x)_i)) $
#pause
#result[$ell_i(bold(theta)) = -log p_(bold(theta))(y_i | bold(x)_i)$]

== Empirical risk = average training loss
#stag(D)

In ML, *risk* means expected loss. Because the data-generating distribution is unknown, we estimate it from the observed training set.
#v(6pt)
#notebox[*Empirical risk* = average loss on $cal(D) = {(bold(x)_i, y_i)}_(i=1)^n$.]
#v(7pt)
#align(center, table(
  columns: (37mm, 1fr), stroke: 0.5pt + MUTED,
  inset: (x: 9pt, y: 6pt), align: (center, left),
  [*Per example*], [$ell_i(bold(theta)) = -log p_(bold(theta))(y_i | bold(x)_i)$],
  [*Training set*], [$hat(R)(bold(theta)) = 1/n sum_i ell_i(bold(theta)) = -1/n log p(cal(D) | bold(theta))$],
))
#v(6pt)
#align(center, text(size: 14pt, fill: MUTED)[The final equality uses conditional independence: $p(cal(D)|bold(theta)) = product_i p_(bold(theta))(y_i|bold(x)_i)$.])
#place(bottom + center, dy: -2pt,
  result[Minimizing empirical NLL is maximum likelihood; $1/n$ only rescales the objective.])

== Worked example: support becomes a hard loss
#stag(D)

Recall the Uniform observation model:
$ p_(bold(theta))(y_i | bold(x)_i) = cases(1\/(b-a) & quad a <= y_i <= b, 0 & quad "otherwise") $
#pause
If an observed target lies outside its support, then
$ p_(bold(theta))(y_i | bold(x)_i) = 0 $
#pause
Its contribution to the negative log-likelihood is
$ ell_i(bold(theta)) = -log p_(bold(theta))(y_i | bold(x)_i) = -log 0 = +infinity $
#pause
#result[Zero likelihood becomes an infinite training penalty.]

== Checkpoint: uniform support #Q

#mcq(
  [For a Uniform$(a,b)$ observation model, what is the NLL if the target lies outside $[a,b]$?],
  [$0$],
  [A fixed finite penalty],
  [$-log 0 = +infinity$],
  [It depends only on $sigma$],
)

== Answer: uniform support #A

#mcq-answer([C], [$-log 0 = +infinity$], [The model assigns zero density outside its support, so the observation is impossible under the assumption.])

// ══════════════════════════════ PART IV ══════════════════════════════
= Regression: why MSE?

== Regression as signal plus noise

$ y_i = f_(bold(theta))(bold(x)_i) + epsilon_i, quad quad epsilon_i tilde cal(N)(0, sigma^2) $
#pause
Therefore the conditional output distribution is
$ Y_i | bold(X)_i = bold(x)_i tilde cal(N)(f_(bold(theta))(bold(x)_i), sigma^2) $

== What the assumption means visually
#stag(V)

#let _reg-mean(x) = 1.2 + 0.28 * x
#let _reg-xs = range(26).map(i => 0.3 + 9.4 * i / 25)
#let _reg-data = _reg-xs.enumerate().map(((i, x)) =>
  (x, _reg-mean(x) + 0.48 * rnd.randn(1, i)))
#let _reg-line = range(81).map(i => {
  let x = 10.0 * i / 80
  (x, _reg-mean(x))
})
// At four fixed inputs, draw p(y|x) sideways: horizontal width is density,
// while the bell's vertical centre is the conditional mean.
#let _reg-bells = (1.5, 4.0, 6.5, 9.0).map(x =>
  range(41).map(j => {
    let t = -2.6 + 5.2 * j / 40
    (x + 0.48 * calc.exp(-0.5 * t * t), _reg-mean(x) + 0.55 * t)
  }))
#let _reg-series = (_reg-line,) + _reg-bells + _reg-data.map(p => (p,))
#align(center, lines(
  _reg-series,
  colors: (ACC,) + range(_reg-bells.len()).map(_ => TEAL) + range(_reg-data.len()).map(_ => MUTED),
  markers: (false,) + range(_reg-bells.len()).map(_ => false) + range(_reg-data.len()).map(_ => true),
  y-ticks: false, x-label: [$x$], y-label: [$y$],
  annotations: ((8.1, 4.65, [$f_(bold(theta))(bold(x))$]),),
  size: (91mm, 45mm),
))
#pause
#notebox[Gaussian noise lives in the *output direction*, conditional on $bold(x)$ — a vertical bell at every input.]

== Gaussian likelihood
#stag(D)

$ p_(bold(theta))(y_i | bold(x)_i) = 1/sqrt(2 pi sigma^2) exp(-(y_i - f_(bold(theta))(bold(x)_i))^2/(2 sigma^2)) $
#pause
Negative log of a single example:
$ -log p_(bold(theta))(y_i | bold(x)_i) = (y_i - f_(bold(theta))(bold(x)_i))^2/(2 sigma^2) + C $

== MSE is Gaussian MLE

Sum over the dataset:
$ -log p(cal(D) | bold(theta)) = 1/(2 sigma^2) sum_i (y_i - f_(bold(theta))(bold(x)_i))^2 + C $
#pause
With $sigma$ fixed, the minimizer is
$ hat(bold(theta))_"MLE" = arg min_(bold(theta)) sum_i (y_i - f_(bold(theta))(bold(x)_i))^2 $
#pause
#result[MSE = Gaussian NLL, up to constants.]

== Linear regression makes $f_(bold(theta))$ explicit
#stag(D)

For the one-dimensional picture,
$ hat(y)_i = theta_0 + theta_1 x_i $
#pause
For $d$ features, prepend a constant $1$ so the intercept is part of the same parameter vector:
$ tilde(bold(x))_i = (1, bold(x)_i), quad bold(theta) = (theta_0, theta_1, dots, theta_d) $
#pause
Then the same notation used throughout the lecture becomes
$ hat(y)_i = f_(bold(theta))(bold(x)_i) = bold(theta)^top tilde(bold(x))_i $
#place(bottom + center, dy: -2pt,
  result[$theta_0$ is the intercept; $theta_1, dots, theta_d$ are the slopes.])

== Gaussian MLE becomes ordinary least squares
#stag(D)

Substitute the linear mean into the observation model:
$ Y_i | bold(X)_i = bold(x)_i tilde cal(N)(bold(theta)^top tilde(bold(x))_i, sigma^2) $
#pause
The vertical residual of point $i$ is
$ r_i = y_i - hat(y)_i = y_i - bold(theta)^top tilde(bold(x))_i $
#pause
With $sigma$ fixed, maximum likelihood therefore gives
$ hat(bold(theta))_"MLE" = arg min_(bold(theta)) sum_i underbrace(r_i^2, "squared vertical error") $
#place(bottom + center, dy: -2pt,
  result[Ordinary least squares is Gaussian maximum likelihood for a linear mean.])

== The normal equations solve linear regression
#stag(D)

Stack the augmented inputs into $tilde(bold(X)) = [bold(1) quad bold(X)]$. Then
$ hat(bold(y)) = tilde(bold(X)) bold(theta), quad J(bold(theta)) = norm(bold(y) - tilde(bold(X)) bold(theta))_2^2 $
#pause
Differentiate and set the gradient to zero:
#align(center, text(size: 17.5pt)[
  $nabla_(bold(theta)) J = 2 tilde(bold(X))^top (tilde(bold(X)) bold(theta) - bold(y)) = bold(0)$
])
#pause
This gives the *normal equations*:
$ tilde(bold(X))^top tilde(bold(X)) hat(bold(theta)) = tilde(bold(X))^top bold(y) $
#pause
If $tilde(bold(X))^top tilde(bold(X))$ is invertible,
#align(center, text(size: 17.5pt)[
  $hat(bold(theta)) = (tilde(bold(X))^top tilde(bold(X)))^(-1) tilde(bold(X))^top bold(y)$
])
#place(bottom + center, dy: -2pt,
  notebox[Linear regression has a closed form; neural networks keep the MSE objective but use iterative optimization.])

== What role does $sigma^2$ play?
#stag(OPT)

$ ell_i = (y_i - mu_i)^2/(2 sigma^2) + 1/2 log sigma^2 + C $
#pause
- one global $sigma^2$ for every example → *homoscedastic* regression
#pause
- small $sigma$: errors penalized *strongly*
#pause
- large $sigma$: errors deemed *less surprising*
#place(bottom + center, dy: -2pt,
  notebox[With fixed $sigma^2$, every input has the same predicted noise width.])

== Homoscedastic vs input-dependent uncertainty
#stag(V)

#grid(
  columns: (1fr, 1fr), column-gutter: 28pt,
  [
    #align(center, text(font: "IBM Plex Mono", size: 12pt, fill: TEAL, weight: 700)[HOMOSCEDASTIC])
    #v(5pt)
    #align(center, diagram(
      spacing: (14mm, 7mm), node-stroke: 0.9pt,
      {
        node((0,1), [$bold(x)_i$], shape: fletcher.shapes.rect, fill: rgb("#EFEEEB"), stroke: INK, inset: 6pt)
        node((1,1), [$f_(bold(theta))$], shape: fletcher.shapes.rect, fill: rgb("#EFEEEB"), stroke: INK, inset: 6pt)
        node((2,1), [$mu_i$], shape: fletcher.shapes.rect, fill: rgb("#FDECD6"), stroke: ACC, inset: 6pt)
        node((3,1), [$cal(N)(mu_i, sigma^2)$], shape: fletcher.shapes.rect, fill: white, stroke: TEAL, inset: 6pt)
        edge((0,1),(1,1), "-|>", stroke: 0.9pt + MUTED)
        edge((1,1),(2,1), "-|>", stroke: 0.9pt + MUTED)
        edge((2,1),(3,1), "-|>", stroke: 0.9pt + MUTED)
      }
    ))
    #v(7pt)
    #align(center, text(size: 16pt)[
      Network output: $mu_i$ \
      Shared variance: $sigma^2$ \
      #text(fill: MUTED)[same uncertainty at every $bold(x)_i$]
    ])
  ],
  [
    #align(center, text(font: "IBM Plex Mono", size: 12pt, fill: ACC, weight: 700)[HETEROSCEDASTIC])
    #v(5pt)
    #align(center, diagram(
      spacing: (14mm, 7mm), node-stroke: 0.9pt,
      {
        node((0,1), [$bold(x)_i$], shape: fletcher.shapes.rect, fill: rgb("#EFEEEB"), stroke: INK, inset: 6pt)
        node((1,1), [$f_(bold(theta))$], shape: fletcher.shapes.rect, fill: rgb("#EFEEEB"), stroke: INK, inset: 6pt)
        node((2,0), [$mu_i$], shape: fletcher.shapes.rect, fill: rgb("#FDECD6"), stroke: ACC, inset: 6pt)
        node((2,2), [$s_i$], shape: fletcher.shapes.rect, fill: rgb("#FDECD6"), stroke: ACC, inset: 6pt)
        node((3,1), [$cal(N)(mu_i, e^(s_i))$], shape: fletcher.shapes.rect, fill: white, stroke: TEAL, inset: 6pt)
        edge((0,1),(1,1), "-|>", stroke: 0.9pt + MUTED)
        edge((1,1),(2,0), "-|>", stroke: 0.9pt + MUTED)
        edge((1,1),(2,2), "-|>", stroke: 0.9pt + MUTED)
        edge((2,0),(3,1), "-|>", stroke: 0.9pt + MUTED)
        edge((2,2),(3,1), "-|>", stroke: 0.9pt + MUTED)
      }
    ))
    #v(7pt)
    #align(center, text(size: 16pt)[
      Network outputs: $mu_i$ and $s_i$ \
      $s_i = log sigma_i^2, quad sigma_i^2 = e^(s_i) > 0$ \
      #text(fill: MUTED)[uncertainty changes with $bold(x)_i$]
    ])
  ],
)
#place(bottom + center, dy: -2pt,
  result[Predict *log-variance* so every real $s_i$ maps to a positive variance $e^(s_i)$.])

== Different noise, different loss
#stag(V)

#two(r: (1fr, 1.25fr),
  [#table(
    columns: (1.05fr, 1fr), stroke: none,
    inset: (x: 8pt, y: 6pt), align: (left, center),
    table.header(
      [#text(fill: INK, weight: 700)[Noise model]],
      [#text(fill: INK, weight: 700)[Negative log-likelihood]],
    ),
    table.hline(stroke: 1.5pt + INK),
    [#box(width: 12pt, height: 3pt, fill: ACC, radius: 1pt) #h(6pt) #text(fill: INK, weight: 600)[Gaussian]], [$r^2$],
    table.hline(stroke: 0.45pt + MUTED.transparentize(45%)),
    [#box(width: 12pt, height: 3pt, fill: TEAL, radius: 1pt) #h(6pt) #text(fill: INK, weight: 600)[Laplace]], [$|r|$],
    table.hline(stroke: 0.45pt + MUTED.transparentize(45%)),
    [#box(width: 12pt, height: 3pt, fill: MUTED, radius: 1pt) #h(6pt) #text(fill: INK, weight: 600)[Uniform]], [$0$ inside; $+infinity$ outside],
    table.hline(stroke: 0.45pt + MUTED.transparentize(45%)),
    [#box(width: 12pt, height: 3pt, fill: GREEN, radius: 1pt) #h(6pt) #text(fill: INK, weight: 600)[Student-$t$]], [$log(1 + r^2/nu)$],
    table.hline(stroke: 0.8pt + INK),
  )
   #v(6pt) #text(size: 15pt, fill: MUTED)[$r = y - hat(y)$ · swatches match plot]],
  align(center, lines(
    // the TRUE negative log-likelihood of each noise model (ml-dist), shifted to
    // min 0: Normal(σ=1/√2)→r², Laplace(b=1)→|r|, Student-t(ν=2)→robust/tail-flat.
    fn: (r => dist.nll0(dist.normal(sigma: 1 / calc.sqrt(2)), r),
         r => dist.nll0(dist.laplace(b: 1.0), r),
         r => dist.nll0(dist.student-t(nu: 2.0), r)),
    domain: (-3, 3), samples: 80,
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
= Classification: cross-entropy

== Classification should output probabilities

For $K$ classes:
$ p_(bold(theta))(Y=k | bold(x)) >= 0, quad quad sum_(k=1)^K p_(bold(theta))(Y=k | bold(x)) = 1 $
#pause
#let _logits = (1.4, 0.6, -0.8)
#align(center, grid(
  columns: (auto, auto, auto, auto, auto),
  column-gutter: 9pt, align: horizon,
  block(fill: rgb("#EFEEEB"), stroke: 0.9pt + INK, radius: 4pt, inset: 8pt,
    text(size: 16pt)[$bold(x)$]),
  text(size: 20pt, fill: MUTED)[$arrow.r$],
  block(fill: TEAL, radius: 4pt, inset: 8pt,
    text(size: 15pt, fill: white, weight: 650)[neural network $f_(bold(theta))$]),
  text(size: 20pt, fill: MUTED)[$arrow.r$],
  grid(
    columns: (1fr, auto, 1fr), column-gutter: 9pt, align: horizon,
    align(center, bars(
      _logits, labels: ([$z_1$], [$z_2$], [$z_3$]), baseline: -1.0,
      size: (43mm, 28mm), title: [logits $bold(z)$],
    )),
    align(center, [#text(size: 13pt, fill: MUTED)[softmax] $arrow.r$]),
    align(center, bars(
      _logits, labels: ([$p_1$], [$p_2$], [$p_3$]), softmax: true,
      highlight: 0, size: (43mm, 28mm), title: [probabilities $bold(p)$],
    )),
  ),
))
#align(center)[*logits* $z_k in RR$ → *probabilities* $p_k in [0,1]$.]

== Binary classification: Bernoulli model

$ Y | bold(X) = bold(x) tilde "Bernoulli"(p_(bold(theta))(bold(x))) $
#pause
Compact one-line form:
$ p_(bold(theta))(y | bold(x)) = p^y (1-p)^(1-y) $
#pause
#two(notebox[$y=1 => p(y)=p$], notebox[$y=0 => p(y)=1-p$])

== Bernoulli NLL gives BCE
#stag(D)

$ -log p_(bold(theta))(y | bold(x)) = -log(p^y (1-p)^(1-y)) $
#pause
$ = -y log p - (1-y) log(1-p) $
#pause
#result[Binary cross-entropy = Bernoulli NLL.]
#pause
#notebox[*Check.* True label $y=1$: if $p=0.8$ the loss is $-log 0.8 approx 0.22$; if $p=0.2$ it is $-log 0.2 approx 1.61$.]

== Why use logits?
#stag(V)

The network emits an unbounded score $z = f_(bold(theta))(bold(x)) in RR$; the *sigmoid* maps it to a probability:
$ p = sigma(z) = 1/(1 + e^(-z)) $
#pause
#two(r: (1fr, 1.2fr),
  [$z -> -infinity: p -> 0$ \ $z = 0: p = 0.5$ \ $z -> +infinity: p -> 1$
   #v(4pt) #text(size: 16pt, fill: MUTED)[In practice sigmoid + BCE are fused for numerical stability.]],
  align(center, lines(
    // the TRUE logistic sigmoid; the dot reads off σ(0)=0.5, dashed lines are the 0/1 asymptotes
    fn: z => dist.sigmoid(z), domain: (-6, 6), samples: 120, markers: false, colors: (ACC,),
    hlines: ((0.0, none, MUTED), (1.0, none, MUTED)),
    points: ((0.0, 0.5, [$0.5$]),),
    x-label: [$z$], y-label: [$sigma(z)$],
    size: (66mm, 46mm),
  )),
)

== Logistic regression = Bernoulli MLE
#stag(D)

Use an augmented input $tilde(bold(x))_i$ so $bold(theta)$ contains both weights and the intercept:
$ p_i = sigma(bold(theta)^top tilde(bold(x))_i), quad Y_i | bold(X)_i = bold(x)_i tilde "Bernoulli"(p_i) $.
#pause
Likelihood of the whole labelled dataset (i.i.d.):
$ L(bold(theta)) = product_(i=1)^n p_i^(y_i) (1 - p_i)^(1 - y_i) $
#pause
Its negative log — the objective we minimise:
$ J(bold(theta)) = -sum_(i=1)^n [y_i log p_i + (1-y_i) log(1-p_i)] $
#pause
#notebox[Logistic regression is Bernoulli maximum likelihood; its training objective is *binary cross-entropy*.]

== Multi-class model and softmax
#stag(V)

#two(
  [
    #align(center, text(font: "IBM Plex Mono", size: 12pt, fill: TEAL, weight: 700)[CATEGORICAL OBSERVATION])
    #v(7pt)
    $ Y | bold(X)=bold(x) tilde "Categorical"(p_1, dots, p_K) $
    #v(8pt)
    A *one-hot* target has one $1$ at the true class and $0$ everywhere else.
    #v(3pt)
    #align(center, text(size: 14.5pt, fill: MUTED)[Example: class 2 of 3 $arrow.r bold(y)=(0,1,0)$])
    #v(4pt)
    $ p_(bold(theta))(bold(y) | bold(x)) = product_(k=1)^K p_k^(y_k) $
  ],
  [
    #align(center, text(font: "IBM Plex Mono", size: 12pt, fill: ACC, weight: 700)[NETWORK PARAMETERIZATION])
    #v(7pt)
    $ bold(z) = f_(bold(theta))(bold(x)), quad p_k = e^(z_k)/(sum_j e^(z_j)) $
    #v(5pt)
    #align(center, bars((2, 1, -0.5), labels: ("1", "2", "3"), softmax: true, title: [softmax probabilities], size: (58mm, 36mm)))
  ],
)
#place(bottom + center, dy: -2pt,
  result[Softmax guarantees $p_k>0$ and $sum_k p_k=1$, as a categorical model requires.])

== Categorical NLL gives cross-entropy
#stag(D)

$ -log p_(bold(theta))(bold(y) | bold(x)) = -log product_k p_k^(y_k) = -sum_k y_k log p_k $
#pause
Because $bold(y)$ is one-hot, only the true class survives:
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

== Interactive: logits → softmax → cross-entropy
#stag(I)

#interbox(link-to: IA + "softmax-cross-entropy")[*`softmax-cross-entropy.html`* — controls $z_1, z_2, z_3$, the true class, and temperature $T$; presets for *correct & confident / correct & unsure / wrong & unsure / wrong & confident*. Shows $bold(z) -> "softmax"(bold(z)\/T) -> -log p_y$.]
#pause
*Ask students to predict the loss before it is revealed.*

== Why not MSE for classification?
#stag(OPT)

MSE is not _impossible_, but it usually corresponds to a *mismatched observation model* for class indicators.
#pause
#two(
  notebox[*Gaussian view* $Y_k = p_k + epsilon_k, epsilon_k tilde cal(N)(0, sigma^2)$],
  notebox[*Categorical view* $Y tilde "Categorical"(bold(p))$],
)
#pause
#align(center, text(size: 17pt)[Cross-entropy also gives *stronger gradients* on confident errors (where MSE's gradient vanishes).])

// ══════════════════════════════ PART VI ══════════════════════════════
= Bayes' rule for learning

== Bayes' rule: from events to model parameters
#stag(V)

#grid(
  columns: (1fr, 1fr), column-gutter: 30pt,
  [
    #align(center, text(font: "IBM Plex Mono", size: 12pt, fill: TEAL, weight: 700)[PROBABILITY LECTURE])
    #v(7pt)
    #align(center, text(size: 23pt)[$P(A | B) = (P(B | A) P(A)) / P(B)$])
    #v(8pt)
    $A$ — event or hypothesis to infer \
    $B$ — observed evidence
    #v(8pt)
    #text(size: 16pt, fill: MUTED)[Examples: disease given a test; urn given a sampled colour.]
  ],
  [
    #align(center, text(font: "IBM Plex Mono", size: 12pt, fill: ACC, weight: 700)[MACHINE / DEEP LEARNING])
    #v(7pt)
    #align(center, text(size: 23pt)[$p(bold(theta) | cal(D)) = (p(cal(D) | bold(theta)) p(bold(theta))) / p(cal(D))$])
    #v(8pt)
    $bold(theta)$ — model parameters to infer \
    $cal(D)$ — observed training dataset
    #v(8pt)
    #text(size: 16pt, fill: MUTED)[Question: which parameter settings remain plausible after seeing the data?]
  ],
)
#place(bottom + center, dy: -2pt,
  result[Replace $A$ by $bold(theta)$ and $B$ by $cal(D)$: the algebra is unchanged; the interpretation changes.])

== Three heads do not prove $theta = 1$
#stag(Q)

Let $theta = P(H)$ and suppose the first three flips are $cal(D) = (H,H,H)$.
#v(4pt)
#two(r: (1.05fr, 0.95fr),
  [
    #align(center, text(size: 35pt, weight: 700, fill: ACC)[H · H · H])
    #v(3pt)
    #align(center, lines(
      fn: t => t * t * t,
      domain: (0, 1), samples: 100, markers: false,
      colors: (ACC,), labels: ([likelihood $L(theta)=theta^3$],), legend: "tl",
      x-label: [$theta$], y-label: [$L(theta)$], y-ticks: false,
      vlines: ((1, [MLE], ACC),), size: (66mm, 43mm),
    ))
  ],
  [
    $ L(theta; cal(D)) = p(cal(D) | theta) = theta^3 $
    #v(9pt)
    $ hat(theta)_"MLE" = arg max_(0 <= theta <= 1) theta^3 = 1 $
    #v(13pt)
    #notebox[Did we learn that the coin *always* lands heads—or did a merely head-biased coin produce a lucky run?]
  ],
)
#place(bottom + center, dy: -2pt,
  result[MLE identifies the best-fitting $theta$; it does *not* make $theta=1$ certain.])

== A prior tempers the conclusion from three flips
#stag(V)

Take a mild prior centred on a fair coin: $theta tilde "Beta"(2,2)$.
#two(r: (0.92fr, 1.08fr),
  [
    *Prior* \
    $p(theta) = 6 theta(1-theta)$
    #v(8pt)
    *Likelihood of $H,H,H$* \
    $p(cal(D) | theta) = theta^3$
    #v(8pt)
    *Posterior* \
    $p(theta | cal(D)) prop theta^3 p(theta)$ \
    $quad = "Beta"(5,2)$
    #v(8pt)
    $hat(theta)_"MLE" = 1, quad hat(theta)_"MAP" = 4/5 = 0.8$
  ],
  [
    #align(center, lines(
      fn: (
        t => 6 * t * (1 - t),
        t => t * t * t,
        t => 30 * t * t * t * t * (1 - t),
      ),
      domain: (0, 1), samples: 140, markers: false,
      colors: (BLUE, ACC, GREEN),
      labels: ([prior], [likelihood], [posterior]), legend: "tl",
      dashes: ("dashed", "dotted", "solid"),
      x-label: [$theta = P(H)$], y-label: [relative support], y-ticks: false,
      vlines: ((0.8, [MAP], GREEN),), size: (70mm, 45mm),
    ))
    #align(center, text(size: 15pt, fill: MUTED)[The posterior moves toward heads, but remains spread over many possible coins.])
  ],
)
#place(bottom + center, dy: -2pt,
  result[With little data, the prior matters; as observations accumulate, the likelihood increasingly dominates.])

// ══════════════════════════════ PART VII ══════════════════════════════
= Priors, MAP and regularization

== Every learner has an inductive bias
#stag(V)

When several predictors fit the training data, the learner still needs a preference for *unseen inputs*. That preference is its *inductive bias*.
#v(6pt)
#align(center, text(size: 16pt)[
  #table(
    columns: (38mm, 128mm), stroke: none,
    inset: (x: 8pt, y: 3.5pt), align: (left, left),
    table.header(
      [#text(fill: INK, weight: 700)[Model]],
      [#text(fill: INK, weight: 700)[Built-in preference $arrow.r$ consequence]],
    ),
    table.hline(stroke: 1.5pt + INK),
    [#text(weight: 650)[Linear regression]], [responses vary linearly with features $arrow.r$ line or plane extrapolation],
    table.hline(stroke: 0.45pt + MUTED.transparentize(45%)),
    [#text(weight: 650)[$k$-NN]], [nearby inputs tend to have similar outputs $arrow.r$ local prediction],
    table.hline(stroke: 0.45pt + MUTED.transparentize(45%)),
    [#text(weight: 650)[Decision tree]], [a few axis-aligned rules partition the space $arrow.r$ piecewise-constant regions],
    table.hline(stroke: 0.45pt + MUTED.transparentize(45%)),
    [#text(weight: 650)[CNN]], [useful local patterns repeat across positions $arrow.r$ shared filters recognize translations],
    table.hline(stroke: 0.8pt + INK),
  )
])
#place(bottom + center, dy: -2pt,
  result[A prior is one explicit inductive bias; the model class, features, architecture, and optimization add others.])

== A neural-network prior ranks plausible parameter vectors
#stag(V)

#let _prior-narrow = dist.normal(mu: 0.0, sigma: 0.55)
#let _prior-broad = dist.normal(mu: 0.0, sigma: 1.55)
#two(r: (1.05fr, 1fr),
  [
    $ bold(theta) tilde p(bold(theta)) $
    #v(7pt)
    For a neural network, $bold(theta)$ contains all weights and biases. Before using the current dataset, a prior states which settings are more plausible.
    #v(8pt)
    It can prefer small or sparse weights, smooth functions, and correlated parameters.
  ],
  [
    #align(center, lines(
      fn: (t => dist.pdf(_prior-narrow, t), t => dist.pdf(_prior-broad, t)),
      domain: (-4, 4), samples: 140, markers: false,
      colors: (ACC, BLUE), labels: ([narrow prior], [broad prior]), legend: "tr",
      x-label: [$theta_j$], y-label: [$p(theta_j)$], y-ticks: false,
      size: (62mm, 43mm),
    ))
    #align(center, text(size: 15pt, fill: MUTED)[narrow = stronger preference near $0$])
  ],
)
#place(bottom + center, dy: -2pt,
  result[A prior makes part of the model's inductive bias explicit as a probability distribution.])

== Parameter priors have visible consequences
#stag(V)

#grid(
  columns: (1fr, 1fr), column-gutter: 30pt,
  [
    #align(center, text(font: "IBM Plex Mono", size: 12pt, fill: TEAL, weight: 700)[REGRESSION])
    #v(4pt)
    #align(center, text(size: 16pt)[$Y_i | bold(X)_i = bold(x)_i tilde cal(N)(bold(theta)^top tilde(bold(x))_i, sigma^2)$])
    #v(7pt)
    #align(center, diagram(
      spacing: (18mm, 10mm), node-stroke: 0.9pt,
      {
        node((0,0), text(size: 14pt)[$p(bold(theta))$], shape: fletcher.shapes.rect, fill: rgb("#EFEEEB"), stroke: INK, inset: 5pt)
        node((0,1), text(size: 14pt)[small $norm(bold(theta))$], shape: fletcher.shapes.rect, fill: rgb("#FDECD6"), stroke: ACC, inset: 5pt)
        node((0,2), text(size: 14pt)[smaller slopes], shape: fletcher.shapes.rect, fill: white, stroke: TEAL, inset: 5pt)
        edge((0,0),(0,1), "-|>", stroke: 0.9pt + MUTED)
        edge((0,1),(0,2), "-|>", stroke: 0.9pt + MUTED)
      }
    ))
    #v(7pt)
    #align(center, text(size: 16pt, fill: MUTED)[Before seeing data, flatter linear predictions are more plausible.])
  ],
  [
    #align(center, text(font: "IBM Plex Mono", size: 12pt, fill: ACC, weight: 700)[BINARY CLASSIFICATION])
    #v(4pt)
    #align(center, text(size: 16pt)[$p_(bold(theta))(Y=1 | bold(x)) = sigma(bold(theta)^top tilde(bold(x)))$])
    #v(7pt)
    #align(center, diagram(
      spacing: (18mm, 10mm), node-stroke: 0.9pt,
      {
        node((0,0), text(size: 14pt)[$p(bold(theta))$], shape: fletcher.shapes.rect, fill: rgb("#EFEEEB"), stroke: INK, inset: 5pt)
        node((0,1), text(size: 14pt)[small logits], shape: fletcher.shapes.rect, fill: rgb("#FDECD6"), stroke: ACC, inset: 5pt)
        node((0,2), text(size: 14pt)[$p$ nearer $0.5$], shape: fletcher.shapes.rect, fill: white, stroke: TEAL, inset: 5pt)
        edge((0,0),(0,1), "-|>", stroke: 0.9pt + MUTED)
        edge((0,1),(0,2), "-|>", stroke: 0.9pt + MUTED)
      }
    ))
    #v(7pt)
    #align(center, text(size: 16pt, fill: MUTED)[Before seeing data, extremely confident predictions are less plausible.])
  ],
)
#place(bottom + center, dy: -2pt,
  result[The prior acts on parameters; its visible effect depends on how those parameters enter the model.])

== Model complexity: underfitting to overfitting

Fit the same eight observations by least squares; only the polynomial degree changes.
#v(2pt)
#align(center, text(size: 12.5pt, fill: MUTED)[dashed = underlying trend · orange = fitted polynomial · red dot = one noisy observation])
#v(3pt)
#let _poly-features(x, degree) = range(1, degree + 1).map(k => calc.pow(x, k))
#let _over-features(x, degree) = {
  let t = (x - 0.5) / 0.4
  range(1, degree + 1).map(k => calc.pow(t, k))
}
#let _over-xs = range(8).map(i => 0.1 + 0.8 * i / 7.0)
// A deliberately small, noisy sample. Degree 1 is too rigid, degree 3 captures
// the broad trend, and degree 7 interpolates the sample then swings at the edges.
#let _over-ys = (0.36, 0.60, 0.81, 1.00, -0.03, -0.85, -1.13, -0.81)
#let _over-true = range(101).map(i => {
  let x = 0.06 + 0.88 * i / 100.0
  (x, calc.sin(2 * calc.pi * x))
})
#let _over-data = _over-xs.zip(_over-ys).map(p => (p,))
#let _over-point-colors = range(_over-data.len()).map(i => if i == 3 { RED } else { INK })
#let _over-bounds = ((0.06, -3.0), (0.94, 2.0))
#let _over-panel(degree, verdict) = {
  let w = linreg-fit(
    _over-xs.map(x => _over-features(x, degree)), _over-ys, ridge: 1e-12)
  let fit = range(101).map(i => {
    let x = 0.06 + 0.88 * i / 100.0
    (x, linreg-predict(w, _over-features(x, degree)))
  })
  let series = (_over-true, fit, _over-bounds) + _over-data
  [
    #align(center, text(size: 15pt, weight: 700)[degree #degree])
    #v(2pt)
    #align(center, lines(
      series,
      colors: (MUTED, ACC, white.transparentize(100%)) + _over-point-colors,
      markers: (false, false, false) + range(_over-data.len()).map(_ => true),
      dashes: ("dashed", "solid", "solid") + range(_over-data.len()).map(_ => "solid"),
      x-label: [$x$], y-ticks: false, size: (48mm, 30mm),
    ))
    #v(2pt)
    #align(center, text(size: 13pt, fill: if degree == 3 { GREEN } else { ACC }, weight: 650)[#verdict])
  ]
}
#grid(
  columns: (1fr, 1fr, 1fr), column-gutter: 9pt,
  _over-panel(1, [underfit]),
  _over-panel(3, [captures the main trend]),
  _over-panel(7, [overfit]),
)
#place(bottom + center, dy: -2pt,
  notebox[Higher degree lowers training error, but degree 7 chases the noisy observation and becomes unstable. Priors and regularization can prefer smoother solutions.])

== Bayes' rule over parameters

$ p(bold(theta) | cal(D)) = (p(cal(D) | bold(theta)) thin p(bold(theta))) / p(cal(D)) $
#pause
// native: real Gaussian likelihood × prior → posterior contours (dist.gaussian-2d).
// posterior sits between the MLE and the prior mean 0 = MAP = regularisation.
#let _lik   = dist.gaussian-2d(mu: (2.0, 1.5), sigma: ((0.64, 0), (0, 0.64)))   // σ = 0.8
#let _prior = dist.gaussian-2d(mu: (0, 0), sigma: ((1.44, 0), (0, 1.44)))       // σ = 1.2
#v(3pt)
#align(center, contour(
  (_lik, _prior, (t1, t2) => _lik(t1, t2) * _prior(t1, t2)),
  xlim: (-2.5, 4), ylim: (-2, 3.5), samples: 64, levels: 4,
  colors: (ACC, BLUE, GREEN),
  // Empty labels keep the three point markers clear; the legend below names them.
  marks: ((2.0, 1.5, [], ACC), (1.4, 1.04, [], GREEN), (0.0, 0.0, [], BLUE)),
  size: (92mm, 54mm), x-label: [$theta_1$], y-label: [$theta_2$],
))
#v(4pt)
#align(center, text(size: 14pt)[
  Contours: #text(fill: ACC, weight: 650)[likelihood] #h(11pt)
  #text(fill: BLUE, weight: 650)[prior] #h(11pt)
  #text(fill: GREEN, weight: 650)[posterior]
])
#align(center, text(size: 14pt)[
  #text(fill: ACC, weight: 700)[●] MLE #h(16pt)
  #text(fill: GREEN, weight: 700)[●] MAP #h(16pt)
  #text(fill: BLUE, weight: 700)[●] prior mean
])
#v(3pt)
#align(center, text(size: 13.5pt, fill: MUTED)[The posterior combines likelihood and prior; $p(cal(D))$ normalizes it, and its mode is the MAP.])

== MAP estimation
#stag(D)

$ hat(bold(theta))_"MAP" = arg max_(bold(theta)) p(bold(theta) | cal(D)) = arg max_(bold(theta)) p(cal(D) | bold(theta)) thin p(bold(theta)) $
#pause
Take negative logs:
$ hat(bold(theta))_"MAP" = arg min_(bold(theta)) (underbrace(-log p(cal(D) | bold(theta)), "NLL") + underbrace(-log p(bold(theta)), "negative log-prior")) $
#pause
#result[MAP objective: NLL $+$ negative log-prior]
#align(center, text(size: 14pt, fill: MUTED)[In ML language: data loss $+$ regularization.])

== A Gaussian prior is $L_2$ regularization
#stag(D)

Assume the parameters are centred at zero before seeing the data:
$ bold(theta) tilde cal(N)(bold(0), tau^2 bold(I)). $
#v(4pt)
#align(center, text(size: 20pt)[
  $underbrace(-log p(cal(D) | bold(theta)), "NLL") + underbrace(-log p(bold(theta)), "negative log-prior")$
])
#pause
$ -log p(bold(theta)) = 1/(2 tau^2) norm(bold(theta))_2^2 + C $
#pause
$ => quad cal(L)_"MAP" = "NLL" + underbrace(1/(2 tau^2), lambda) norm(bold(theta))_2^2 + C $
#v(6pt)
#result[$lambda = 1/(2 tau^2)$: smaller $tau$ $arrow.r$ larger $lambda$ $arrow.r$ stronger shrinkage toward $bold(0)$.]

== Prior width controls how far MAP moves
#stag(V)

#align(center, text(size: 16.5pt)[
  Same likelihood: $L(theta) prop cal(N)(theta; hat(theta)_"MLE"=2, s^2)$ with $s=0.6$; prior: $p(theta)=cal(N)(0,tau^2)$.
])
#align(center, text(size: 17pt)[
  $hat(theta)_"MAP" = (tau^2)/(tau^2+s^2) hat(theta)_"MLE"$ #h(8pt)
  #text(fill: MUTED)[(a shrinkage factor between 0 and 1)]
])
#align(center, text(size: 13.5pt)[
  #text(fill: ACC, weight: 650)[··· likelihood] #h(14pt)
  #text(fill: BLUE, weight: 650)[– – prior] #h(14pt)
  #text(fill: GREEN, weight: 650)[— posterior]
])
#let _prior-map-panel(heading, tau, map-value, map-label) = {
  let s = 0.6
  let post-sigma = calc.sqrt((s*s*tau*tau)/(s*s+tau*tau))
  block(breakable: false)[
    #align(center, text(size: 16pt, weight: 700)[#heading])
    #align(center, text(size: 13.5pt)[
      #text(fill: ACC, weight: 650)[MLE $= 2$] $arrow.r$
      #text(fill: GREEN, weight: 650)[#map-label]
    ])
    #align(center, lines(
      fn: (
        t => calc.exp(-0.5 * ((t - 2.0) / s) * ((t - 2.0) / s)),
        t => calc.exp(-0.5 * (t / tau) * (t / tau)),
        t => calc.exp(-0.5 * ((t - map-value) / post-sigma) * ((t - map-value) / post-sigma)),
      ),
      domain: (-2.5, 3.2), samples: 150, markers: false,
      colors: (ACC, BLUE, GREEN), dashes: ("dotted", "dashed", "solid"),
      vlines: ((2.0, [], ACC), (map-value, [], GREEN)),
      x-label: [$theta$], y-ticks: false,
      size: (65mm, 28mm),
    ))
  ]
}
#v(1pt)
#grid(
  columns: (1fr, 1fr), column-gutter: 16pt,
  _prior-map-panel([Broad prior: $tau=2, lambda=0.125$], 2.0, 1.83, [$"MAP" approx 1.83$]),
  _prior-map-panel([Narrow prior: $tau=0.5, lambda=2$], 0.5, 0.82, [$"MAP" approx 0.82$]),
)
#place(bottom + center, dy: -2pt,
  result[The narrower prior gives the larger $lambda$ and pulls MAP farther from the MLE toward zero.])

== A multivariate Gaussian prior has geometry
#stag(V)

$ bold(theta) tilde cal(N)(bold(mu), bold(Sigma)) quad -log p(bold(theta)) = 1/2 (bold(theta) - bold(mu))^top bold(Sigma)^(-1) (bold(theta) - bold(mu)) + C $
#pause
// native: each panel is the real N(0, Σ) density — dist.gaussian-2d inverts Σ
#let _covs = (
  ([$bold(Sigma) = tau^2 bold(I)$], ((1.0, 0.0), (0.0, 1.0))),
  ([diagonal $bold(Sigma)$], ((2.2, 0.0), (0.0, 0.55))),
  ([full $bold(Sigma)$], ((1.4, 0.95), (0.95, 1.0))),
)
#align(center, stack(dir: ltr, spacing: 6mm,
  .._covs.map(cv => contour(dist.gaussian-2d(sigma: cv.at(1)), xlim: (-3, 3), ylim: (-3, 3),
    samples: 50, levels: 6, fill: true,
    ramp: (rgb("#FFF7ED"), rgb("#FDBA74"), ACC), color: ACC,
    size: (37mm, 37mm), x-label: [$theta_1$], y-label: [$theta_2$], title: cv.at(0))),
))
#v(5pt)
#align(center, text(size: 14pt, fill: MUTED)[
  Axes: $theta_1$ and $theta_2$ are two coordinates of the parameter vector $bold(theta)$.
])
#align(center, text(size: 14pt)[
  #text(fill: ACC, weight: 650)[Colour = prior density $p(bold(theta))$:] pale outer regions are less plausible; darker centres are more plausible.
])
#v(4pt)
#align(center, text(size: 17pt)[Covariance says *which directions in parameter space are plausible.*])

== A Laplace prior is $L_1$ regularization
#stag(OPT)

Assume the parameter components are independent, with
$ p(theta_j) = 1/(2b) exp(-abs(theta_j)/b). $
#pause
$ p(bold(theta)) = product_j p(theta_j). $
#pause
Taking negative logs gives
$ -log p(bold(theta)) = 1/b sum_j abs(theta_j) + C = underbrace(1/b, lambda) norm(bold(theta))_1 + C. $
#place(bottom + center, dy: -2pt,
  result[$lambda = 1/b$: a narrower Laplace prior gives stronger $L_1$ regularization; its sharp peak at zero promotes *sparsity*.])

== Prior strength controls underfitting and overfitting
#stag(V)

For the summed-NLL convention used above, $lambda = 1/(2 tau^2)$:
#v(6pt)
#align(center, diagram(
  spacing: (39mm, 8mm), node-stroke: 0.9pt,
  {
    node((0,0), [weak prior\ $lambda approx 0$], shape: fletcher.shapes.rect,
      fill: rgb("#FDECD6"), stroke: ACC, inset: 7pt)
    node((1,0), [balanced], shape: fletcher.shapes.rect,
      fill: rgb("#E7F3EA"), stroke: GREEN, inset: 7pt)
    node((2,0), [strong prior\ large $lambda$], shape: fletcher.shapes.rect,
      fill: rgb("#E8EEF6"), stroke: BLUE, inset: 7pt)
    edge((0,0),(1,0), "-|>", stroke: 1pt + MUTED)
    edge((1,0),(2,0), "-|>", stroke: 1pt + MUTED)
  }
))
#v(12pt)
#grid(
  columns: (1fr, 1fr, 1fr), column-gutter: 20pt,
  [
    #align(center, text(font: "IBM Plex Mono", size: 12pt, fill: ACC, weight: 700)[DATA FIT DOMINATES])
    #v(6pt)
    #align(center)[
      flexible model
      #linebreak()
      #text(fill: RED, weight: 650)[high variance · overfit]
    ]
  ],
  [
    #align(center, text(font: "IBM Plex Mono", size: 12pt, fill: GREEN, weight: 700)[LIKELIHOOD + PRIOR])
    #v(6pt)
    #align(center)[
      enough flexibility
      #linebreak()
      #text(fill: GREEN, weight: 650)[better generalization]
    ]
  ],
  [
    #align(center, text(font: "IBM Plex Mono", size: 12pt, fill: BLUE, weight: 700)[PRIOR DOMINATES])
    #v(6pt)
    #align(center)[
      weights forced small
      #linebreak()
      #text(fill: BLUE, weight: 650)[high bias · underfit]
    ]
  ],
)
#place(bottom + center, dy: -2pt,
  result[Regularization trades variance for bias; *more is not always better*.])

== Interactive: likelihood, prior, MAP
#stag(I)

#interbox(link-to: IA + "map-regularization")[*`map-regularization.html`* — a two-parameter linear model so contours are visible. Shows *likelihood*, *prior*, and *posterior* contours with the *MLE*, *MAP*, and zero points. \
_Controls:_ amount of data · noise · prior variance · Gaussian vs Laplace · prior correlation.]
#pause
*Ask:* more data? narrower prior? why does MAP → MLE as data grows? a correlated prior?

// ══════════════════════════════ PART VIII ══════════════════════════════
= Full Bayes and deep learning

== MAP is still one parameter vector

#two(notebox[*MLE* $hat(bold(theta))_"MLE"$ — one point], notebox[*MAP* $hat(bold(theta))_"MAP"$ — one point])
#pause
#align(center)[*Full Bayes* keeps the whole posterior $p(bold(theta) | cal(D))$ — a *distribution*, not a point.]

== Posterior predictive distribution

$ p(y^star | bold(x)^star, cal(D)) = integral p_(bold(theta))(y^star | bold(x)^star) thin p(bold(theta) | cal(D)) dif bold(theta) $
#pause
#let _post-xs = range(15).map(i => i / 14.0)
#let _post-ys = _post-xs.enumerate().map(((i, x)) =>
  calc.sin(2 * calc.pi * x) + 0.20 * rnd.randn(5, i))
#let _post-curves = range(6).map(s => {
  let ys = _post-ys.enumerate().map(((i, y)) => y + 0.12 * rnd.randn(20 + s, i))
  let w = linreg-fit(_post-xs.map(x => _poly-features(x, 5)), ys, ridge: 0.002)
  range(101).map(i => {
    let x = i / 100.0
    (x, linreg-predict(w, _poly-features(x, 5)))
  })
})
#let _post-data = _post-xs.zip(_post-ys).map(p => (p,))
#align(center, lines(
  _post-curves + _post-data,
  colors: range(_post-curves.len()).map(i =>
    (TEAL, GREEN, BLUE, ACC, TEAL.lighten(25%), ACC.lighten(20%)).at(i))
    + range(_post-data.len()).map(_ => INK),
  markers: range(_post-curves.len()).map(_ => false) + range(_post-data.len()).map(_ => true),
  annotations: ((0.78, 1.1, [posterior model samples]),),
  x-label: [$x^star$], y-label: [$y^star$], y-ticks: false, size: (82mm, 43mm),
))
#notebox[Bayesian prediction *averages over plausible models* — and reports uncertainty.]

== Why DL usually uses point estimates
#stag(OPT)

A modern network has millions to billions of parameters, so integrating $p(bold(theta) | cal(D))$ is hard.
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

#align(center, text(size: 21pt)[
  $ min_(bold(theta)) underbrace(1/(|B|) sum_(i in B) -log p_(bold(theta))(y_i | bold(x)_i), "likelihood / data fit") + underbrace(lambda norm(bold(theta))_2^2, "prior / regularization") $
])
#v(8pt)
#align(center, text(size: 14pt, fill: MUTED)[With an averaged NLL, $lambda$ absorbs the dataset or minibatch scaling convention.])
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

== Every supervised objective begins with two assumptions
#stag(V)

#v(8pt)
#grid(
  columns: (0.9fr, auto, 1.35fr, auto, 0.9fr), column-gutter: 11pt,
  align: horizon,
  [
    #align(center, text(font: "IBM Plex Mono", size: 12pt, fill: TEAL, weight: 700)[MODEL])
    #v(9pt)
    #align(center, text(size: 17pt, weight: 650)[observation model])
    #align(center, text(size: 20pt)[$p_(bold(theta))(y | bold(x))$])
    #v(8pt)
    #align(center, text(size: 17pt, weight: 650)[parameter prior])
    #align(center, text(size: 20pt)[$p(bold(theta))$])
  ],
  align(center, text(size: 31pt, fill: MUTED)[$arrow.r$]),
  [
    #align(center, text(font: "IBM Plex Mono", size: 12pt, fill: ACC, weight: 700)[TRAIN])
    #v(9pt)
    #align(center, text(size: 18pt)[
      $cal(L)_"MAP"(bold(theta))$ \
      $= underbrace("NLL"(cal(D); bold(theta)), "data term")$ \
      $quad + underbrace(Omega(bold(theta)), "parameter term")$
    ])
    #align(center, text(size: 14pt, fill: MUTED)[$Omega(bold(theta)) = -log p(bold(theta))$])
    #v(10pt)
    #align(center, text(size: 18pt, weight: 650)[$hat(bold(theta))_"MAP" = arg min_(bold(theta)) cal(L)_"MAP"$])
  ],
  align(center, text(size: 31pt, fill: MUTED)[$arrow.r$]),
  [
    #align(center, text(font: "IBM Plex Mono", size: 12pt, fill: GREEN, weight: 700)[PREDICT])
    #v(9pt)
    #align(center, text(size: 17pt, weight: 650)[plug-in: fix $bold(theta)=hat(bold(theta))$])
    #align(center, text(size: 18pt)[$p_(hat(bold(theta)))(y^star | bold(x)^star)$])
    #v(8pt)
    #align(center, text(size: 17pt, weight: 650)[or average over the posterior])
    #align(center, text(size: 18pt)[$p(y^star | bold(x)^star, cal(D))$])
  ],
)
#v(15pt)
#result[Loss $arrow.l.r$ likelihood; regularizer $arrow.l.r$ prior; prediction $arrow.l.r$ plug-in or posterior predictive.]
#place(bottom + center, dy: -2pt,
  text(size: 15pt, fill: MUTED)[Next: from linear models to neural networks — *neurons, nonlinearities, and multilayer perceptrons.*])
