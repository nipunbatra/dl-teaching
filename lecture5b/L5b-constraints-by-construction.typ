// Constraints by Construction - Lecture 5B · ES 667 Deep Learning
// Compile from the repository root:
//   typst compile --root . --input handout=true lecture5b/L5b-constraints-by-construction.typ slides-pdf/L5B.pdf
//   typst compile --root . lecture5b/L5b-constraints-by-construction.typ slides-pdf/L5B-presentation.pdf

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *

#show: metropolis-deck.with(
  title: [Constraints by Construction],
  subtitle: [Lecture 5B · Parameterization is part of model design],
)

#let PT-SOFTPLUS = "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html"
#let PT-BCE = "https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.BCEWithLogitsLoss.html"
#let PT-CE = "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html"
#let PT-LOGSOFTMAX = "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html"
#let PT-MVN = "https://docs.pytorch.org/docs/stable/distributions.html#multivariatenormal"

// Semantic grammar for this mini-lecture:
// MUTED/INK = raw coordinates · TEAL/GREEN = valid transformed parameters
// ACC = transformation · RED = invalid or brittle behaviour.
#let caption(body) = align(center, text(size: 13pt, fill: MUTED)[#body])
#let punch(body) = align(center, text(size: 20pt, weight: 650, fill: INK)[#body])
#let tiny-source(body) = text(size: 10.5pt, fill: MUTED)[#body]
#let swatch(color, body) = box(width: 11pt, height: 3pt, fill: color, radius: 1pt) + h(4pt) + body
#let hairline(title, body, color: TEAL) = block(width: 100%, inset: 3pt, [
  #text(size: 14pt, weight: 700, fill: color)[#title]
  #v(2pt)
  #line(length: 100%, stroke: 0.7pt + color)
  #v(5pt)
  #body
])

#let softplus-num(z) = calc.ln(1 + calc.exp(z))

#let transform-flow(left: [$z in RR^d$], middle: [$phi$], right: [$theta in cal(C)$]) = align(center,
  diagram(spacing: (31mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
    node((0, 0), left, shape: fletcher.shapes.rect, corner-radius: 4pt,
      inset: 9pt, stroke: 1pt + INK, fill: rgb("#F3F2EE"))
    node((1, 0), middle, radius: 7.5mm, stroke: 1.2pt + ACC, fill: ACC.lighten(88%))
    node((2, 0), right, shape: fletcher.shapes.rect, corner-radius: 4pt,
      inset: 9pt, stroke: 1.2pt + TEAL, fill: TEAL.lighten(91%))
    edge((0, 0), (1, 0), "-|>", stroke: 1pt + MUTED)
    edge((1, 0), (2, 0), "-|>", stroke: 1.1pt + ACC)
  })
)

#let simplex-picture = align(center, diagram(
  spacing: (36mm, 28mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    // Edges first so every interior point remains unobscured.
    edge((0, 1.35), (1, 0), stroke: 1.1pt + INK)
    edge((1, 0), (2, 1.35), stroke: 1.1pt + INK)
    edge((2, 1.35), (0, 1.35), stroke: 1.1pt + INK)
    node((0, 1.35), none, radius: 2.2mm, fill: BLUE, stroke: none)
    node((1, 0), none, radius: 2.2mm, fill: TEAL, stroke: none)
    node((2, 1.35), none, radius: 2.2mm, fill: ACC, stroke: none)
    node((0, 1.62), text(size: 12pt)[$p_1=1$], stroke: none)
    node((1, -0.25), text(size: 12pt)[$p_3=1$], stroke: none)
    node((1.72, 1.62), text(size: 12pt)[$p_2=1$], stroke: none)
    node((0.65, 0.93), none, radius: 1.8mm, fill: ACC, stroke: none)
    node((1.06, 0.72), none, radius: 1.8mm, fill: GREEN, stroke: none)
    node((1.37, 1.05), none, radius: 1.8mm, fill: BLUE, stroke: none)
    node((0.97, 1.16), none, radius: 1.8mm, fill: TEAL, stroke: none)
  }
))

// A deterministic heteroskedastic dataset: the plotted uncertainty and the
// observations share the same sigma(x), so the picture cannot contradict the model.
#let h-mean(x) = -0.7 + 0.42 * x + 0.20 * calc.sin(1.15 * x)
#let h-sigma(x) = 0.14 + 0.075 * x
#let h-xs = range(28).map(i => 0.25 + 7.5 * i / 27)
#let h-data = h-xs.enumerate().map(((i, x)) =>
  (x, h-mean(x) + h-sigma(x) * rnd.randn(53, i)))
#let h-grid = range(101).map(i => 8.0 * i / 100)
#let h-mid = h-grid.map(x => (x, h-mean(x)))
#let h-lo = h-grid.map(x => (x, h-mean(x) - 2 * h-sigma(x)))
#let h-hi = h-grid.map(x => (x, h-mean(x) + 2 * h-sigma(x)))
#let h-series = (h-lo, h-hi, h-mid) + h-data.map(p => (p,))
#let hetero-picture = align(center, lines(
  h-series,
  colors: (BLUE, BLUE, ACC) + range(h-data.len()).map(_ => MUTED),
  markers: (false, false, false) + range(h-data.len()).map(_ => true),
  y-ticks: false, x-label: [$x$], y-label: [$y$],
  annotations: (
    (5.8, h-mean(5.8) + 0.13, [$mu(x)$]),
    (6.2, h-mean(6.2) + 2 * h-sigma(6.2) + 0.18, [$mu(x)+2sigma(x)$]),
  ),
  size: (99mm, 54mm),
))

#title-slide()

// ═══════════════════════════ OPENING ═══════════════════════════
== A neural network can output values the model cannot use #V

#grid(columns: (1fr, 1fr, 1fr), gutter: 16pt,
  align(center, [
    #text(size: 14pt, weight: 700, fill: INK)[Neural-network output]
    #v(2pt)
    #text(size: 19pt)[$z=f_w(x) in RR^d$]
  ]),
  align(center, [
    #text(size: 14pt, weight: 700, fill: ACC)[Smooth map]
    #v(2pt)
    #text(size: 19pt)[$phi: RR^d arrow.r cal(C)$]
  ]),
  align(center, [
    #text(size: 14pt, weight: 700, fill: TEAL)[Valid model parameter]
    #v(2pt)
    #text(size: 19pt)[$theta=phi(z) in cal(C)$]
  ]),
)

#v(2pt)
#align(center, image("figures/constraints-by-construction-pipeline-v4-wide.png",
  width: 252mm, height: 72mm, fit: "cover"))

#v(2pt)
#notebox[The network may emit any $z in RR^d$; $phi$ ensures the model only sees a valid $theta in cal(C)$.]

== Before optimizing: what does scale do? #D

For one Gaussian observation, let $r=y-mu$ and work inside the valid domain $sigma>0$:

#align(center, text(size: 22pt)[
  $ y | x tilde cal(N)(mu, sigma^2), quad
    cal(L)=underbrace(r^2/(2 sigma^2), "fit the residual")
    + underbrace(log sigma, "pay for uncertainty") + C. $
])

#v(8pt)
#grid(columns: (1fr, 1fr, 1fr), gutter: 15pt,
  hairline([$sigma<0$], [
    Not a valid standard deviation. The probabilistic model is undefined.
  ], color: RED),
  hairline([$sigma arrow 0$], [
    If $r != 0$, the residual term explodes. Tiny scales demand tiny errors.
  ], color: ACC),
  hairline([$sigma$ is large], [
    The $log sigma$ term grows. The model cannot inflate uncertainty for free.
  ], color: TEAL),
)

== What will one gradient step produce? #Q

Suppose a Gaussian model learns its scale directly. At one update,

$ sigma_0=0.1, quad (partial cal(L))/(partial sigma)=5, quad eta=0.1. $

#v(8pt)
#mcq(
  [What is $sigma_1 = sigma_0 - eta (partial cal(L))/(partial sigma)$?],
  [$0.6$: the step increases uncertainty],
  [$0.4$: take the absolute value],
  [$-0.4$: the step crosses zero],
  [$0.1$: the model blocks invalid moves],
)

== A gradient step can make $sigma$ invalid #A

#grid(columns: (1.05fr, 0.95fr), gutter: 26pt,
  [
    $ sigma_1 = 0.1 - 0.1(5) = -0.4. $
    #v(10pt)
    #align(center, diagram(
      spacing: (14mm, 9mm), node-stroke: none, node-fill: none,
      {
        // Candidate answers are placed approximately to scale.
        edge((0, 1), (5.1, 1), stroke: 1.1pt + INK)

        node((0.50, 1), none, radius: 2.3mm, fill: RED)
        node((0.50, 0.48), text(size: 13pt, weight: 750, fill: RED)[C], stroke: none)
        node((0.50, 1.52), text(size: 12pt, fill: RED)[$-0.4$], stroke: none)

        node((2.08, 1), none, radius: 1.2mm, fill: INK)
        node((2.08, 1.52), text(size: 12pt)[$0$], stroke: none)

        node((2.50, 1), none, radius: 2.0mm, fill: TEAL)
        node((2.50, 0.48), text(size: 13pt, weight: 750, fill: TEAL)[D], stroke: none)
        node((2.50, 1.52), text(size: 12pt, fill: TEAL)[$0.1$], stroke: none)

        node((3.75, 1), none, radius: 2.0mm, fill: BLUE)
        node((3.75, 0.48), text(size: 13pt, weight: 750, fill: BLUE)[B], stroke: none)
        node((3.75, 1.52), text(size: 12pt, fill: BLUE)[$0.4$], stroke: none)

        node((4.58, 1), none, radius: 2.0mm, fill: ACC)
        node((4.58, 0.48), text(size: 13pt, weight: 750, fill: ACC)[A], stroke: none)
        node((4.58, 1.52), text(size: 12pt, fill: ACC)[$0.6$], stroke: none)

        node((-0.22, 1.52), text(size: 13pt)[$-infinity$], stroke: none)
        node((5.34, 1.52), text(size: 13pt)[$+infinity$], stroke: none)
      }
    ))
    #v(9pt)
    #mcq-answer([C], [$-0.4$], [Gradient descent follows the local slope, even when the step makes $sigma$ invalid.])
  ],
  [
    #hairline([What failed?], [
      The update followed the usual gradient-descent rule:
      #v(5pt)
      $ "new" = "old" - eta times "gradient". $
      #v(7pt)
      The problem is the coordinate: $sigma$ is valid only on $(0,infinity)$, but direct optimization treats it as an unconstrained real number.
    ], color: RED)
  ],
)

// ═════════════════════ CONSTRAINTS BY CONSTRUCTION ═════════════════════
== Correct fix: make invalid values unreachable #D

#transform-flow()

#v(10pt)
#align(center, table(
  columns: (72mm, 1fr),
  stroke: (x: none, y: 0.45pt + MUTED),
  inset: (x: 7pt, y: 7pt),
  align: (left + horizon, left + horizon),
  text(size: 13.5pt, weight: 700, fill: TEAL)[Positive],
  text(size: 15pt)[$s in RR arrow.r sigma=e^s$ or $sigma="softplus"(s)+epsilon$, $epsilon>0$; either gives $sigma>0$],
  text(size: 13.5pt, weight: 700, fill: ACC)[Probability],
  text(size: 15.5pt)[$z in RR arrow.r p="sigmoid"(z) in (0,1)$],
  text(size: 13.5pt, weight: 700, fill: BLUE)[Probability vector (simplex)],
  text(size: 15.5pt)[$bold(p)="softmax"(bold(z)): quad p_k>0, quad sum_k p_k=1$],
  text(size: 13.5pt, weight: 700, fill: GREEN)[Positive-definite covariance],
  text(size: 15pt)[raw vector $arrow.r$ lower-triangular factor $arrow.r Sigma succ 0$],
))

#v(5pt)
#notebox[
  #set text(size: 13.3pt)
  *$z in RR^d$:* unrestricted network output; $d$ is its length. #h(12pt)
  *$phi$ (phi):* constraint map.
  #linebreak()
  *$cal(C)$:* allowed parameter set. #h(12pt)
  *$theta=phi(z) in cal(C)$:* valid parameter used by the model.
]

== Softplus smoothly turns any real number positive #V

#grid(columns: (0.98fr, 1.02fr), gutter: 24pt,
  [
    #set text(size: 16pt)
    Take any real-valued raw coordinate $u$ and define
    #v(5pt)
    #align(center, text(size: 22pt, weight: 650, fill: ACC)[
      $ "softplus"(u)=log(1+e^u). $
    ])
    #v(6pt)
    #hairline([Constraint guarantee], [
      $sigma="softplus"(u)+epsilon>epsilon$ for every finite $u$.
    ], color: TEAL)
    #v(6pt)
    #hairline([Learning signal], [
      Softplus is increasing, and its slope stays below $1$.
    ], color: ACC)
  ],
  [
    #align(center, lines(
      fn: (u => calc.max(0, u), u => softplus-num(u)),
      domain: (-4.0, 4.0), samples: 140, markers: false,
      colors: (MUTED, TEAL), labels: ([$max(0,u)$], [$"softplus"(u)$]),
      legend: "tl", y-ticks: false, x-label: [$u$], y-label: [$f(u)$],
      size: (78mm, 45mm),
    ))
    #caption[Softplus rounds off the ReLU corner while remaining strictly above zero.]
    #v(6pt)
    #align(center, text(size: 14.5pt)[
      $u=-4 arrow.r 0.018$ #h(10pt)
      $u=0 arrow.r log 2 approx 0.693$ #h(10pt)
      $u=3 arrow.r 3.05$
    ])
  ],
)

== Can you differentiate softplus? #Q

#set text(size: 16pt)
Let $a(u)=1+e^u$. Then $"softplus"(u)=log(a(u))$.

#pause

#v(7pt)
#grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([1. Inner derivative], [
    $a'(u)=0+e^u=e^u.$
  ], color: TEAL),
  hairline([2. Outer derivative], [
    $ (partial)/(partial a) log a = 1/a. $
  ], color: ACC),
)

#v(8pt)
#hairline([3. Apply the chain rule], [
  #align(center, text(size: 19pt)[
    $ (partial "softplus"(u))/(partial u)
      = 1/(a(u)) times a'(u) $
    #v(4pt)
    $ = 1/(1+e^u) times e^u
      = e^u/(1+e^u). $
  ])
], color: INK)

== The derivative of softplus is a sigmoid #D

#set text(size: 16pt)
Divide the numerator and denominator of $e^u/(1+e^u)$ by $e^u$.

#v(6pt)
#align(center, block(width: 142mm, inset: (x: 12pt, y: 8pt), radius: 4pt,
  fill: rgb("#F3F2EE"), stroke: 0.7pt + MUTED, [
  #align(center, text(size: 20.5pt, weight: 650, fill: ACC)[
    $ e^u/(1+e^u) = (e^u/e^u)/((1+e^u)/e^u) $
    #v(5pt)
    $ = 1/(e^(-u)+1) = 1/(1+e^(-u)) $
    #v(5pt)
    $ = "sigmoid"(u). $
  ])
]))

#v(7pt)
#grid(columns: (1fr, 1fr, 1fr), gutter: 16pt,
  hairline([Large negative $u$], [
    The slope is near $0$.
  ], color: MUTED),
  hairline([$u=0$], [
    The slope is exactly $1/2$.
  ], color: ACC),
  hairline([Large positive $u$], [
    The slope is near $1$.
  ], color: TEAL),
)

#v(4pt)
#align(center, text(size: 13pt, fill: MUTED)[For finite $u$ in exact arithmetic, $0<"sigmoid"(u)<1$; softplus has no exactly flat branch.])

== The same raw step is gentler under softplus #V

#grid(columns: (1.2fr, 0.8fr), gutter: 28pt,
  [
    #align(center, lines(
      fn: (u => calc.exp(u), u => softplus-num(u)),
      domain: (-4.0, 3.0), samples: 120, markers: false,
      colors: (ACC, TEAL), labels: ([Exponential], [Softplus]), legend: "tl",
      y-ticks: false, x-label: [raw coordinate], y-label: [$sigma$], size: (108mm, 61mm),
    ))
  ],
  [
    #text(size: 16pt, weight: 650)[Starting from $sigma=10$, take the same raw step $Delta=+0.1$.]
    #v(12pt)
    #table(
      columns: (38mm, 1fr),
      stroke: (x: none, y: 0.5pt + MUTED),
      inset: (x: 5pt, y: 7pt),
      align: (left + horizon, left + horizon),
      text(size: 14.5pt, weight: 700, fill: ACC)[Exponential],
      text(size: 17pt)[$10 arrow.r 11.05$],
      text(size: 14.5pt, weight: 700, fill: TEAL)[Softplus],
      text(size: 17pt)[$10 arrow.r 10.10$],
    )
    #v(14pt)
    #notebox[
      #set text(size: 14.2pt)
      *Here we use softplus:* large predicted scales move gently.
    ]
  ],
)

== For Gaussian regression, we use softplus #V

#grid(columns: (1.12fr, 0.88fr), gutter: 24pt,
  [
    #codebox(size: 12.4pt)[```python
mu, raw_scale = net(x).unbind(-1)  # net(x): [N, 2]
assert y.ndim == 2 and y.shape[-1] == 1
y = y[:, 0]                        # [N, 1] -> [N]
assert y.shape == mu.shape

# Chosen here: positive scale + a small numerical floor
sigma = F.softplus(raw_scale) + 1e-4

# Also valid, but not used here:
# sigma = raw_scale.exp()

loss = -Normal(mu, sigma).log_prob(y).mean()
```]
    #v(5pt)
    #tiny-source[#link(PT-SOFTPLUS)[PyTorch `softplus` documentation]]
  ],
  [
    #set text(size: 14pt)
    #hairline([Which map?], [
      We use *softplus* from here on. `exp` would also keep $sigma$ positive, but it is a different map. Pick one.
    ], color: TEAL)
    #v(5pt)
    #hairline([What does PyTorch handle?], [
      `Normal.log_prob(y)` evaluates the Gaussian likelihood; autograd differentiates through softplus. No hand derivation is needed.
    ], color: ACC)
    #v(5pt)
    #hairline([Why assert shapes?], [
      Keep $mu$ and $y$ both `[N]`; otherwise broadcasting can silently produce `[N, N]` log probabilities.
    ], color: RED)
  ],
)

== Heteroskedastic regression makes the scale visible #V

#grid(columns: (1.2fr, 0.8fr), gutter: 20pt,
  [
    #hetero-picture
    #caption[#swatch(ACC, [predicted mean $mu(x)$]) #h(14pt) #swatch(BLUE, [$mu(x) plus.minus 2sigma(x)$]) #h(14pt) noise widens with $x$]
  ],
  [
    #align(center, diagram(spacing: (15mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
      node((0, 0.5), [$x$], radius: 5mm)
      node((1, 0.5), [$f_w$], shape: fletcher.shapes.rect, corner-radius: 3pt,
        inset: 6pt, stroke: 1pt + INK)
      node((2, 0), [$mu$], radius: 5.5mm, stroke: 1pt + ACC)
      node((2, 1), [$u$], radius: 5.5mm, stroke: 1pt + INK)
      node((3, 1), [$sigma$], radius: 5.5mm, stroke: 1pt + TEAL)
      edge((0, 0.5), (1, 0.5), "-|>", stroke: 0.9pt + MUTED)
      edge((1, 0.5), (2, 0), "-|>", stroke: 0.9pt + ACC)
      edge((1, 0.5), (2, 1), "-|>", stroke: 0.9pt + MUTED)
      edge((2, 1), (3, 1), "-|>", stroke: 1pt + ACC,
        label: text(size: 9pt, fill: ACC)[softplus])
    }))
    #v(9pt)
    #hairline([Two outputs, two meanings], [
      $mu(x)$ is the conditional mean. Softplus maps the raw $u(x)$ to a valid conditional scale.
    ], color: TEAL)
    #v(6pt)
    #text(size: 12.5pt, fill: MUTED)[Under the Gaussian model, $sigma(x)$ is the conditional residual, or aleatoric, scale. It does not measure epistemic uncertainty about the network.]
  ],
)

// ═══════════════════════════ PROBABILITIES ═══════════════════════════
== Sigmoid maps one logit to a valid probability #V

#grid(columns: (1.08fr, 0.92fr), gutter: 26pt,
  [
    $ p="sigmoid"(z)=1/(1+e^(-z)), quad z in RR. $
    #align(center, lines(
      fn: z => dist.sigmoid(z), domain: (-6.0, 6.0), samples: 120,
      markers: false, colors: (ACC,), y-ticks: false,
      hlines: ((0.0, none, MUTED), (1.0, none, MUTED)),
      points: ((0.0, 0.5, [$0.5$]),),
      x-label: [$z$], y-label: [$p$], size: (67mm, 45mm),
    ))
    #caption[In exact arithmetic, finite logits map into the open interval $(0,1)$.]
  ],
  [
    #align(center, table(
      columns: (26mm, 36mm),
      stroke: 0.5pt + MUTED, inset: (x: 8pt, y: 5pt),
      align: center,
      table.header([$z$], [$p="sigmoid"(z)$]),
      [$-2$], [$0.119$],
      [$0$], [$0.500$],
      [$2$], [$0.881$],
    ))
    #v(8pt)
    #set text(size: 16.5pt)
    Negative logits favor class $0$; positive logits favor class $1$.
    Running the map backward gives the *log-odds*; let us unpack that scale next.
    #v(4pt)
    #text(size: 12pt, fill: MUTED)[Finite precision can saturate sigmoid to exactly $0$ or $1$ for extreme logits.]
  ],
)

== Odds compare class 1 with class 0 #V

#align(center, text(size: 22pt, weight: 650, fill: ACC)[
  $"odds"=p/(1-p).$
])

#v(5pt)
#align(center, text(size: 15.8pt)[
  In binary classification, $p$ is the probability of class $1$ and $1-p$ is the probability of class $0$.
])

#v(14pt)
#grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([$p=0.01$], [
    #align(center, text(size: 17pt)[
      $"odds"=0.01/0.99$
      #v(6pt)
      $=1/99 approx 0.010$
    ])
    #v(6pt)
    #caption[Class 0 is strongly favored.]
  ], color: MUTED),
  hairline([$p=0.50$], [
    #align(center, text(size: 17pt)[
      $"odds"=0.50/0.50$
      #v(6pt)
      $=1$
    ])
    #v(6pt)
    #caption[Neither class is favored.]
  ], color: ACC),
  hairline([$p=0.99$], [
    #align(center, text(size: 17pt)[
      $"odds"=0.99/0.01$
      #v(6pt)
      $=99$
    ])
    #v(6pt)
    #caption[Class 1 is strongly favored.]
  ], color: TEAL),
)

#v(14pt)
#notebox[
  #grid(columns: (1fr, 1fr, 1fr), gutter: 12pt, align: center + horizon,
    text(size: 15.5pt, weight: 650, fill: MUTED)[Odds $<1$: class 0],
    text(size: 15.5pt, weight: 650, fill: ACC)[Odds $=1$: tie],
    text(size: 15.5pt, weight: 650, fill: TEAL)[Odds $>1$: class 1],
  )
]

== Log-odds stretch probabilities across the real line #V

#align(center, text(size: 21pt, weight: 650, fill: ACC)[
  $z="logit"(p)=log("odds")=log(p/(1-p)).$
])

#v(7pt)
#align(center, text(size: 15.5pt)[
  Taking the logarithm sends odds below $1$ negative, odds of $1$ to zero, and odds above $1$ positive.
])

#v(14pt)
#grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([$p=0.01$], [
    #align(center, text(size: 18pt)[
      $z=log(1/99)$
      #v(6pt)
      $approx -4.60$
    ])
  ], color: MUTED),
  hairline([$p=0.50$], [
    #align(center, text(size: 18pt)[
      $z=log(1)$
      #v(6pt)
      $=0$
    ])
  ], color: ACC),
  hairline([$p=0.99$], [
    #align(center, text(size: 18pt)[
      $z=log(99)$
      #v(6pt)
      $approx +4.60$
    ])
  ], color: TEAL),
)

#v(15pt)
#grid(columns: (1fr, 1fr, 1fr), gutter: 12pt, align: center + horizon,
  text(size: 17pt, weight: 650)[$p arrow 0^+ quad => quad z arrow -infinity$],
  text(size: 17pt, weight: 650)[$p=0.5 quad => quad z=0$],
  text(size: 17pt, weight: 650)[$p arrow 1^- quad => quad z arrow +infinity$],
)

#v(13pt)
#notebox[
  #set text(size: 15pt)
  *Inverse maps:* $p="sigmoid"(z)$ if and only if $z="logit"(p)$.
]

== Can you rewrite BCE using $"softplus"(z)$? #Q

#align(center, text(size: 15.5pt)[Let $y in {0,1}$, $p="sigmoid"(z)$, and
$ell(z,y)="BCE"("sigmoid"(z),y)$.])

#v(4pt)
#align(center, text(size: 18pt, weight: 650)[
  $ell(z,y)=-y log p-(1-y)log(1-p).$
])

#pause

#v(4pt)
#grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([1. Rewrite $log p$], [
    #align(center, text(size: 16.5pt)[
      $p=e^z/(1+e^z)$
      #v(4pt)
      $log p=z-log(1+e^z)$
    ])
  ], color: ACC),
  hairline([2. Rewrite $log(1-p)$], [
    #align(center, text(size: 16.5pt)[
      $1-p=1/(1+e^z)$
      #v(4pt)
      $log(1-p)=-log(1+e^z)$
    ])
  ], color: TEAL),
)

#v(5pt)
#hairline([3. Substitute and collect the common term], [
  #align(center, text(size: 16.2pt)[
    $ell(z,y)=-y (z-log(1+e^z))+(1-y)log(1+e^z)$
    #v(4pt)
    $=-y z+[y+(1-y)]log(1+e^z)="softplus"(z)-y z.$
  ])
], color: INK)

== A correct formula can still overflow on a computer #D

#align(center, text(size: 17.5pt, weight: 650)[
  $"softplus"(z)=log(1+e^z)$ is mathematically correct.
])

#v(7pt)
#grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([A modest input: $z=2$], [
    #align(center, text(size: 16.5pt)[
      $e^2 approx 7.389$
      #v(4pt)
      $log(1+e^2) approx 2.1269.$
    ])
    #v(7pt)
    #text(size: 13.5pt)[The direct computation creates no unusually large number, so it works.]
  ], color: TEAL),
  hairline([A large input: $z=1000$], [
    #align(center, text(size: 16.5pt)[
      $e^1000 arrow infinity$ in float32
      #v(4pt)
      $log(1+infinity)=infinity.$
    ])
    #v(7pt)
    #text(size: 13.5pt)[But the true value is approximately $1000$, not infinity. The computer overflowed before taking the logarithm.]
  ], color: RED),
)

#v(9pt)
#notebox[
  #set text(size: 15pt)
  *What we need:* the same mathematical function, evaluated in an order that never constructs a huge exponential.
]

== Can we evaluate softplus without a large exponential? #Q

#align(center, text(size: 15.5pt)[Start from $"softplus"(z)=log(1+e^z)$ and separate the two signs of $z$.])

#pause

#v(3pt)
#grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([If $z >= 0$: move the large term outside], [
    #align(center, text(size: 15.6pt)[
      $log(1+e^z)$
      #v(2pt)
      $=log(e^z(1+e^(-z)))$
      #v(2pt)
      $=z+log(1+e^(-z)).$
    ])
    #v(3pt)
    #text(size: 12.4pt)[Now $-z <= 0$, so the exponential is at most $1$.]
  ], color: ACC),
  hairline([If $z < 0$: the original form is already safe], [
    #align(center, text(size: 15.6pt)[
      $0<e^z<1$
      #v(2pt)
      $log(1+e^z)$
      #v(2pt)
      $=0+log(1+e^(-abs(z))).$
    ])
    #v(3pt)
    #text(size: 12.4pt)[Here $-abs(z)=z$, so we have not changed the value.]
  ], color: TEAL),
)

#v(4pt)
#hairline([One expression covers both cases], [
  #align(center, text(size: 17.4pt, weight: 650)[
    $"softplus"(z)=max(z,0)+log(1+e^(-abs(z))).$
  ])
  #v(2pt)
  #align(center, text(size: 12.6pt, fill: GREEN)[Because $-abs(z) <= 0$, the remaining exponential satisfies $e^(-abs(z)) <= 1$ and cannot overflow.])
], color: GREEN)

== Same answer normally; stable answer at extremes #D

#grid(columns: (1.12fr, 0.88fr), gutter: 24pt,
  [
    #hairline([Compare float32 outputs], [
      #table(
        columns: (0.48fr, 1.30fr, 1.08fr),
        inset: (x: 7pt, y: 8pt),
        align: (center, center, center),
        stroke: (x, y) => if y == 0 { (bottom: 0.8pt + INK) } else { (bottom: 0.35pt + MUTED.lighten(45%)) },
        table.header(
          [*$z$*],
          [*Direct formula*],
          [*Stable route*],
        ),
        [$-1000$], [`0`], [`0`],
        [$2$], [`2.1269`], [`2.1269`],
        [$1000$], [#text(fill: RED)[`inf`]], [#text(fill: GREEN)[`1000`]],
      )
    ], color: INK)
    #v(8pt)
    #text(size: 13.5pt)[At ordinary inputs the answers agree. At $z=1000$, only the stable evaluation returns the finite answer.]
  ],
  [
    #hairline([Check it in PyTorch], [
      #codebox(size: 11.4pt)[```python
z = torch.tensor([-1000., 2., 1000.])

naive = torch.log1p(torch.exp(z))
stable = F.softplus(z)

# naive:  tensor([0., 2.1269, inf])
# stable: tensor([0., 2.1269, 1000.])
```]
    ], color: ACC)
    #v(8pt)
    #alertbox[
      `log1p` improves accuracy when its input is small, but it cannot rescue an `exp(z)` that has already overflowed.
    ]
  ],
)

#v(7pt)
#notebox[
  #set text(size: 14.2pt)
  *In practice:* use `F.softplus`. For binary classification, use the fused BCE-with-logits operation introduced after the gradient derivation.
]

== Is underflow a problem too? #Q

#only("1")[
  #align(center, text(size: 17pt, weight: 650)[
    At $z=+1000$ or $z=-1000$, floating point rounds $e^(-abs(z))$ to $0$.
  ])
  #v(5pt)
  #align(center, text(size: 17pt, fill: ACC)[
    Does that break the stable formula?
  ])
]

#pause

#only("2-")[
  #grid(columns: (1fr, 1fr), gutter: 26pt,
    hairline([$z=+1000$], [
      #align(center, text(size: 17pt)[
        $e^(-1000) arrow.r 0$
        #v(2pt)
        $"softplus"(z)=1000+log(1+0)=1000.$
      ])
      #v(2pt)
      #text(size: 13.2pt)[Only an unrepresentably tiny correction was discarded.]
    ], color: GREEN),
    hairline([$z=-1000$], [
      #align(center, text(size: 17pt)[
        $e^(-1000) arrow.r 0$
        #v(2pt)
        $"softplus"(z)=0+log(1+0)=0.$
      ])
      #v(2pt)
      #text(size: 13.2pt)[The exact answer is positive, about $e^(-1000)$, but too small to store.]
    ], color: TEAL),
  )

  #v(8pt)
  #notebox[
    #set text(size: 13.7pt)
    *Forward value:* overflow turned a useful finite answer into $infinity$; underflow here drops only a tiny correction. For a scale, $sigma="softplus"(u)+epsilon$ still gives $sigma>epsilon$.
    #v(4pt)
    *Learning caveat:* $"softplus"'(z)="sigmoid"(z)$ also rounds to $0$ at $z=-1000$, so an extremely negative raw scale can learn very slowly.
  ]
]

== What is the BCE gradient with respect to the logit? #Q

#align(center, text(size: 16pt)[Start from the smooth expression we just derived:])

#v(2pt)
#align(center, text(size: 20pt, weight: 650)[
  $ell(z,y)="softplus"(z)-y z.$
])

#pause

#v(5pt)
#hairline([Differentiate with respect to the logit], [
  #align(center, text(size: 18pt)[
    $(partial ell)/(partial z)="sigmoid"(z)-y=p-y.$
  ])
  #v(3pt)
  #align(center, text(size: 13.2pt, fill: MUTED)[We used $"softplus"'(z)="sigmoid"(z)=p$.])
], color: ACC)

#v(5pt)
#grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([If $p-y<0$], [
    #text(size: 14.5pt)[The loss has a *negative slope* at the current $z$. Moving $z$ upward makes the loss fall.]
  ], color: GREEN),
  hairline([If $p-y>0$], [
    #text(size: 14.5pt)[The loss has a *positive slope* at the current $z$. Moving $z$ downward makes the loss fall.]
  ], color: RED),
)

#v(6pt)
#align(center, text(size: 14.2pt, weight: 650, fill: ACC)[The crucial next step: gradient descent *subtracts* this slope.])

== Which way will the logit move? #Q

#align(center, text(size: 23pt, weight: 650, fill: ACC)[
  $"new " z = "old " z - eta (p-y).$
])

#v(12pt)
#align(center, text(size: 17pt, weight: 650)[
  Target $y=1$, current probability $p=0.2$, learning rate $eta=0.1$.
])

#pause

#v(12pt)
#hairline([One example], [
  #align(center, text(size: 21pt)[
    $p-y=0.2-1=-0.8$
    #v(12pt)
    $"new " z="old " z-0.1(-0.8)$
    #v(5pt)
    $="old " z+0.08.$
  ])
], color: GREEN)

#v(16pt)
#notebox[
  #set text(size: 16pt)
  *Subtracting a negative slope raises $z$.* A larger logit gives a larger probability $p$, moving the prediction toward the target $y=1$.
]

== The same $p-y$ signal trains the whole network #D

#align(center, text(size: 15.5pt)[Now return to the real network: the optimizer updates $theta$. For $z=f_(theta)(x)$, apply the chain rule:])

#v(3pt)
#hairline([Error signal $times$ sensitivity of the logit], [
  #align(center, text(size: 20pt, weight: 650)[
    $nabla_theta ell=(p-y)nabla_theta z.$
  ])
  #v(3pt)
  #align(center, text(size: 13.2pt, fill: MUTED)[The product combines the prediction error with each parameter's effect on the logit.])
], color: ACC)

#v(5pt)
#grid(columns: (1fr, 1fr), gutter: 26pt,
  hairline([For $z=bold(w)^T bold(h)+b$], [
    #align(center, text(size: 16.5pt)[
      $(partial ell)/(partial bold(w))=(p-y)bold(h)$
      #v(4pt)
      $(partial ell)/(partial b)=p-y$
      #v(4pt)
      $(partial ell)/(partial bold(h))=(p-y)bold(w).$
    ])
  ], color: TEAL),
  hairline([What the two factors mean], [
    #set text(size: 14.5pt)
    $p-y$ says whether the prediction should move *up or down*, and how strongly.
    #v(7pt)
    $nabla_theta z$ says how each parameter changes that prediction.
  ], color: INK),
)

== Train with logits #D

#grid(columns: (0.95fr, 1.05fr), gutter: 24pt,
  [
    *Why not form $p$ first?*
    #v(5pt)
    #codebox(size: 11.6pt)[```python
p = torch.sigmoid(z)
loss = -(y * p.log()
         + (1 - y) * torch.log1p(-p)).mean()
```]
    #v(7pt)
    With $z=100$ and $y=0$, float32 sigmoid may round $p$ to $1$, so
    $-log(1-p) arrow infinity$.
    #v(7pt)
    #alertbox[Forming $p$ first can make the naive formula infinite.]
  ],
  [
    *Use the fused primitive*
    #v(5pt)
    #codebox(size: 11.8pt)[```python
assert y.dtype.is_floating_point
assert y.shape == z.shape
assert ((0 <= y) & (y <= 1)).all()

loss = F.binary_cross_entropy_with_logits(z, y)
p = z.sigmoid()  # interpretation/reporting only
```]
    #v(6pt)
    For $z=100,y=0$, the fused loss is approximately $100$, not infinity.
    #v(5pt)
    #tiny-source[#link(PT-BCE)[PyTorch `BCEWithLogitsLoss`: fused sigmoid and BCE]]
  ],
)

== Softmax maps logits into the simplex #V

#grid(columns: (0.95fr, 1.05fr), gutter: 24pt,
  [
    #simplex-picture
    #v(4pt)
    #caption[For $K=3$, the simplex is this triangle of valid probability vectors. An edge means at least one class has probability $0$.]
  ],
  [
    $ p_k=e^(z_k)/(sum_j e^(z_j)), quad bold(z) in RR^K. $
    #v(7pt)
    The simplex is the set of vectors with $p_k >= 0$ and $sum_k p_k=1$.
    #v(7pt)
    In exact arithmetic, finite logits guarantee
    $ p_k>0, quad sum_k p_k=1. $
    #v(9pt)
    Thus softmax lands strictly *inside* the simplex, never exactly on an edge. Boundary distributions are approached as logit differences grow.
  ],
)

== What does naive softmax compute at extreme logits? #Q

#align(center, text(size: 19pt, weight: 650)[
  $p_k=e^(z_k)/(sum_j e^(z_j)).$
])

#v(3pt)
#align(center, text(size: 14.5pt)[
  Try $bold(z)=(1000,999,998)$ and $bold(z)=(-1000,-1001,-1002)$ in float32.
])

#pause

#v(5pt)
#grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([Overflow: exponentials become infinite], [
    #align(center, text(size: 16.5pt)[
      $bold(z)=(1000,999,998)$
      #v(5pt)
      $"float32:" quad e^bold(z) arrow (infinity,infinity,infinity)$
      #v(5pt)
      $p_k=infinity/infinity arrow "NaN".$
    ])
  ], color: RED),
  hairline([Underflow: exponentials become zero], [
    #align(center, text(size: 16.5pt)[
      $bold(z)=(-1000,-1001,-1002)$
      #v(5pt)
      $"float32:" quad e^bold(z) arrow (0,0,0)$
      #v(5pt)
      $p_k=0/0 arrow "NaN".$
    ])
  ], color: MUTED),
)

#v(7pt)
#notebox[
  #set text(size: 14.5pt)
  Both vectors have the same logit gaps and should give
  $bold(p) approx (0.665,0.245,0.090)$.
  The probability distribution exists; the direct floating-point algorithm failed before it could normalize.
]

== Does adding the same constant change softmax? #Q

#align(center, text(size: 15.5pt)[Let $c$ be the same scalar added to every logit.])

#v(4pt)
#align(center, text(size: 18pt, weight: 650)[
  $p'_k = e^(z_k+c)/(sum_j e^(z_j+c)).$
])

#pause

#v(5pt)
#align(center, block(width: 190mm, inset: (x: 14pt, y: 9pt), radius: 4pt,
  fill: rgb("#F3F2EE"), stroke: 0.7pt + MUTED, [
  #align(center, text(size: 18pt)[
    $p'_k = (e^c e^(z_k))/(e^c sum_j e^(z_j))$
    #v(5pt)
    $= e^(z_k)/(sum_j e^(z_j)) = p_k.$
  ])
]))

#v(9pt)
#grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([Only the gaps matter], [
    #align(center, text(size: 16pt)[
      $(z_i+c)-(z_j+c)=z_i-z_j.$
    ])
    The common offset carries no class information.
  ], color: TEAL),
  hairline([Exact consequence], [
    #align(center, text(size: 15.5pt)[
      $(1000,999,998)=(2,1,0)+998(1,1,1).$
    ])
    These vectors have exactly the same softmax in exact arithmetic.
  ], color: ACC),
)

== A small example shows shifting preserves probabilities #D

#align(center, text(size: 14.4pt)[
  #table(
    columns: (39mm, 86mm, 86mm),
    stroke: 0.5pt + MUTED, inset: (x: 8pt, y: 6pt),
    align: (left, center, center),
    table.header([*Quantity*], [*Direct: $bold(z)=(2,1,0)$*], [*Shifted: $bold(r)=(0,-1,-2)$*]),
    [Logits], [$(2,1,0)$], [$(0,-1,-2)$],
    [Exponentials], [$(7.389,2.718,1.000)$], [$(1.000,0.368,0.135)$],
    [Sum], [$11.107$], [$1.503$],
    [Probabilities], [$(0.665,0.245,0.090)$], [$(0.665,0.245,0.090)$],
  )
])

#v(10pt)
#hairline([Why the ratios match], [
  #align(center, text(size: 17pt)[
    $e^(z_j-2)=e^(z_j)/e^2.$
  ])
  Every numerator and the denominator were divided by the same positive number $e^2$, so every ratio is unchanged.
], color: TEAL)

#v(7pt)
#notebox[
  #set text(size: 14pt)
  The loss is unchanged too: for class $1$, $-log(p_1) approx 0.408$ along either route.
  For $bold(z)=(1000,999,998)$, subtracting its maximum also gives $bold(r)=(0,-1,-2)$; next we work that extreme case through.
]

== The same shift rescues $bold(z)=(1000,999,998)$ #D

#grid(columns: (1.08fr, 0.92fr), gutter: 24pt,
  [
    #hairline([Direct fails; shifting first works], [
      #set text(size: 14.7pt)
      *Direct float32:* $e^bold(z) arrow (infinity,infinity,infinity)$, so $bold(p) arrow ("NaN","NaN","NaN")$.
      #v(6pt)
      *Shifted route:*
      #v(3pt)
      *1.* $m=max_j z_j=1000$ \
      *2.* $r_j=z_j-m$, so $bold(r)=(0,-1,-2)$ \
      *3.* $bold(w)=e^bold(r) approx (1,0.368,0.135)$ \
      *4.* $S=sum_j w_j approx 1.503$ \
      *5.* $bold(p)=bold(w)/S approx (0.665,0.245,0.090)$.
    ], color: ACC)
  ],
  [
    #hairline([Why the shifted arithmetic is safe], [
      #set text(size: 15pt)
      $r_j=z_j-m <= 0$ for every class. \
      At least one $r_j=0$. \
      Therefore $0 <= e^(r_j) <= 1$. \
      The denominator contains $e^0=1$, so
      #align(center, text(size: 17pt, weight: 650)[$1 <= sum_j e^(r_j) <= K.$])
      No exponential overflows, and the denominator cannot become zero.
    ], color: GREEN)
  ],
)

#v(8pt)
#align(center, text(size: 12.2pt, fill: MUTED)[Very small probabilities may still round to zero. That is acceptable for reporting, but a training loss should stay in log space.])

== For training, stay in log space #D

#grid(columns: (1.05fr, 0.95fr), gutter: 24pt,
  [
    #hairline([Stable cross-entropy ($m=max_j z_j$)], [
      #align(center, text(size: 16.5pt)[
        $ell(bold(z),t)=-log p_t$
        #v(4pt)
        $=-z_t+log(sum_j e^(z_j))$
        #v(4pt)
        $=-(z_t-m)+log(sum_j e^(z_j-m)).$
      ])
    ], color: TEAL)
    #v(7pt)
    #hairline([Why log space matters], [
      #set text(size: 14.2pt)
      For $bold(z)=(0,1000)$ and *mathematical class 1*, softmax may round to $(0,1)$, so $-log(p_1)=infinity$.
      The stable expression gives
      #align(center, text(size: 15.5pt)[$1000+log(1+e^(-1000)) approx 1000.$])
    ], color: ACC)
  ],
  [
    #hairline([Use the fused primitives], [
      #codebox(size: 11.8pt)[```python
# logits: [N, K]
# target: [N], class indices 0,...,K-1
loss = F.cross_entropy(logits, target)

# only when these values are needed
log_probs = F.log_softmax(logits, dim=-1)
probs = logits.softmax(dim=-1)
```]
      #v(6pt)
      `cross_entropy` accepts unnormalized logits; `log_softmax` avoids computing `log(softmax(...))` as two unstable operations.
      #v(5pt)
      #tiny-source[#link(PT-CE)[PyTorch `cross_entropy`] · #link(PT-LOGSOFTMAX)[PyTorch `log_softmax`]]
      #v(6pt)
      #text(size: 11.8pt, fill: MUTED)[Stable does not make floating-point exact or repair `NaN` or infinite input logits.]
    ], color: GREEN)
  ],
)

// ═══════════════════════════ MATRIX CONSTRAINTS ═══════════════════════════
#let covariance-panel(title, sigma, rho-label, note, color: TEAL) = block(width: 100%, [
  #align(center, text(size: 14pt, weight: 700, fill: color)[#title])
  #v(2pt)
  #align(center, contour(
    dist.gaussian-2d(sigma: sigma),
    xlim: (-3, 3), ylim: (-3, 3), samples: 42, levels: 5,
    fill: true, ramp: (rgb("#FFF7ED"), color.lighten(68%), color), color: color,
    size: (53mm, 35mm), x-label: [$Y_1$], y-label: [$Y_2$],
  ))
  #v(1pt)
  #align(center, text(size: 13.2pt, weight: 650)[#rho-label])
  #align(center, text(size: 12.4pt, fill: MUTED)[#note])
])

== A multivariate Gaussian needs a covariance matrix #V

#grid(columns: (1.08fr, 0.92fr), gutter: 24pt,
  [
    #align(center, text(size: 19pt, weight: 650, fill: INK)[
      $y tilde cal(N)(mu,sigma^2)$
      $quad arrow.r quad$
      $bold(Y) tilde cal(N)(bold(mu),Sigma).$
    ])
    #v(8pt)
    #hairline([The definition], [
      #align(center, text(size: 17pt)[
        $bold(mu)=EE[bold(Y)],$
        #v(3pt)
        $Sigma="Cov"(bold(Y))=EE[(bold(Y)-bold(mu))(bold(Y)-bold(mu))^T].$
      ])
    ], color: ACC)
    #v(8pt)
    #align(center, text(size: 17pt)[
      $Sigma=mat(
        "Var"(Y_1), "Cov"(Y_1,Y_2);
        "Cov"(Y_1,Y_2), "Var"(Y_2)
      ).$
    ])
    #v(11pt)
    #align(center, text(size: 13pt, fill: MUTED)[
      In one dimension, variance sets the width. In several dimensions, covariance sets the widths and tilt of the Gaussian cloud.
    ])
  ],
  [
    #align(center, contour(
      dist.gaussian-2d(sigma: ((1.0, 0.75), (0.75, 1.35))),
      xlim: (-3.2, 3.2), ylim: (-3.2, 3.2), samples: 48, levels: 6,
      fill: true, ramp: (rgb("#FFF7ED"), TEAL.lighten(68%), TEAL), color: TEAL,
      size: (82mm, 54mm), x-label: [$Y_1$], y-label: [$Y_2$],
    ))
    #caption[Equal-density contours form a tilted ellipse around $bold(mu)$.]
    #v(5pt)
    #hairline([Read the entries], [
      #set text(size: 13.8pt)
      *Diagonal:* each output's variance. #linebreak()
      *Off-diagonal:* how the two outputs move together. #linebreak()
      The repeated off-diagonal entry makes $Sigma=Sigma^T$.
    ], color: TEAL)
  ],
)

== Covariance shows how two outputs move together #V

#align(center, text(size: 17pt, weight: 650, fill: INK)[
  $"Cov"(Y_1,Y_2)=EE[(Y_1-mu_1)(Y_2-mu_2)].$
])
#caption[Each observation contributes the product of its two deviations from the means.]

#v(5pt)
#grid(columns: (1fr, 1fr, 1fr), gutter: 14pt,
  covariance-panel(
    [Positive covariance], ((1.0, 0.75), (0.75, 1.0)),
    [$rho=0.75$], [Same-sign deviations reinforce.], color: TEAL,
  ),
  covariance-panel(
    [Zero covariance], ((1.0, 0.0), (0.0, 1.0)),
    [$rho=0$], [Positive and negative products cancel.], color: BLUE,
  ),
  covariance-panel(
    [Negative covariance], ((1.0, -0.75), (-0.75, 1.0)),
    [$rho=-0.75$], [Opposite-sign deviations dominate.], color: ACC,
  ),
)

#v(3pt)
#align(center, text(size: 12.8pt, fill: MUTED)[
  $rho="Cov"(Y_1,Y_2)/("sd"(Y_1)"sd"(Y_2)) in [-1,1]$ is unitless. Covariance carries multiplied units—for example, cm$dot$kg. For a jointly Gaussian pair, $rho=0$ also means independence.
])

== PSD vs PD: can a combination have zero variance? #V

#align(center, text(size: 16pt)[Choose weights $bold(w)=(w_1,w_2)$ and combine the outputs:])
#v(2pt)
#align(center, text(size: 19pt, weight: 650, fill: ACC)[
  $S=w_1 Y_1+w_2 Y_2, quad "Var"(S)=bold(w)^T Sigma bold(w).$
])
#v(2pt)
#align(center, text(size: 13.5pt, fill: MUTED)[$bold(w)$ is simply the pair of weights; $S$ is one weighted sum of the two outputs.])

#v(7pt)
#grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([Positive semidefinite (PSD)], [
    #text(size: 14pt)[Every choice of weights gives $"Var"(S)>=0$; a nonzero choice may still give $0$.]
    #v(7pt)
    #align(center, text(size: 17pt)[$Sigma=mat(1,1; 1,1), quad bold(w)=(1,-1)$])
    #v(3pt)
    #align(center, text(size: 17pt, weight: 650)[$S=Y_1-Y_2, quad "Var"(S)=0.$])
    #v(3pt)
    #text(size: 13.5pt)[The difference is constant: the two outputs are perfectly locked together.]
  ], color: TEAL),
  hairline([Positive definite (PD)], [
    #text(size: 14pt)[Every nonzero choice of weights gives $"Var"(S)>0$; zero is not allowed.]
    #v(7pt)
    #align(center, text(size: 18pt)[$Sigma=I_2$])
    #v(3pt)
    #align(center, text(size: 17pt, weight: 650)[$"Var"(S)=w_1^2+w_2^2>0$ if $bold(w) != bold(0)$.])
    #v(3pt)
    #text(size: 13.5pt)[No nontrivial weighted combination is perfectly fixed.]
  ], color: GREEN),
)

== Why does $Sigma=L L^T$ guarantee positive variance? #Q

#align(center, text(size: 16.5pt)[For a 2-D target, the network emits three unrestricted numbers:])
#v(3pt)
#align(center, text(size: 20pt, weight: 650)[$bold(q)=(q_11,q_21,q_22) in RR^3.$])

#v(9pt)
#grid(columns: (1.22fr, 0.78fr), gutter: 28pt,
  [
    #align(center, text(size: 19pt, weight: 650, fill: ACC)[
      $L(bold(q))=mat(
        "softplus"(q_11)+epsilon, 0;
        q_21, "softplus"(q_22)+epsilon
      )$
    ])
    #v(10pt)
    #align(center, text(size: 23pt, weight: 650, fill: TEAL)[$Sigma=L L^T.$])
  ],
  [
    #hairline([Read the construction], [
      #set text(size: 14.4pt)
      *$bold(q)$:* raw network outputs. #linebreak()
      *$L$:* lower-triangular factor. #linebreak()
      *Softplus:* makes the two diagonal entries positive. #linebreak()
      *$epsilon>0$:* a small numerical floor.
    ], color: TEAL)
  ],
)

#pause

#v(8pt)
#notebox[
  #set text(size: 14.2pt)
  Pick any nonzero weights $bold(w)$. Positive diagonal entries make $L$ invertible, so $L^T bold(w) != bold(0)$ and
  $bold(w)^T Sigma bold(w)=norm(L^T bold(w))^2>0$. Thus every nontrivial weighted combination has positive variance, so $Sigma$ is PD.
]

== In PyTorch, pass the factor directly #V

#grid(columns: (1.2fr, 0.8fr), gutter: 28pt,
  [
    #codebox(size: 12.2pt)[```python
mu, q = model(x)                 # [B, 2], [B, 3]

L = q.new_zeros(q.shape[:-1] + (2, 2))
L[..., 0, 0] = F.softplus(q[..., 0]) + 1e-4
L[..., 1, 0] = q[..., 1]
L[..., 1, 1] = F.softplus(q[..., 2]) + 1e-4

dist = MultivariateNormal(mu, scale_tril=L)
loss = -dist.log_prob(y).mean()
```]
    #v(5pt)
    #tiny-source[#link(PT-MVN)[PyTorch `MultivariateNormal` documentation]]
    #v(9pt)
    #text(size: 13.5pt, fill: MUTED)[`scale_tril=L` passes the factor directly; PyTorch need not form or invert $Sigma=L L^T$.]
  ],
  [
    #hairline([How gradients reach $bold(q)$], [
      #set text(size: 14.2pt)
      Autograd applies the chain rule. For the first diagonal entry,
      #v(6pt)
      #align(center, text(size: 15.8pt)[$
        (partial ell)/(partial q_11)
        =(partial ell)/(partial L_11) "sigmoid"(q_11).
      $])
      #v(5pt)
      This uses $"softplus"'(q)="sigmoid"(q)$; the same rule holds for $q_22$.
      #v(7pt)
      Since $L_21=q_21$, its gradient passes through unchanged.
      #v(7pt)
      #text(size: 13.2pt, fill: ACC)[*Unlike ReLU,* softplus has no exactly flat negative branch, although its slope can become very small for a very negative input.]
    ], color: TEAL)
  ],
)

// ═══════════════════════════ SYNTHESIS ═══════════════════════════
== One recipe: raw outputs in, valid parameters out #V

#align(center, text(size: 16pt)[
  The network may output any real numbers. A fixed transformation gives the model only legal parameters.
])

#v(12pt)
#align(center, table(
  columns: (58mm, 58mm, 1fr),
  stroke: 0.5pt + MUTED,
  inset: (x: 9pt, y: 9pt),
  align: (left + horizon, center + horizon, left + horizon),
  table.header([*Model needs*], [*Raw output*], [*Transformation*]),
  [Positive scale], [$u in RR$], [$sigma="softplus"(u)+epsilon$],
  [Probability], [$z in RR$], [$p="sigmoid"(z)$],
  [Probability vector], [$bold(z) in RR^K$], [$bold(p)="softmax"(bold(z))$],
  [PD covariance], [raw lower-triangle entries], [positive-diagonal $L$; $Sigma=L L^T$],
))

#v(14pt)
#notebox[
  #set text(size: 15.5pt)
  *Main idea:* let the network learn unrestricted coordinates, and enforce the model's domain in the transformation.
]
