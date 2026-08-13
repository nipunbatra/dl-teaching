// From Linear Models to Neural Networks — Lecture 2 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture2/L2-linear-to-mlp.typ
//   typst compile --root . --input handout=true lecture2/L2-linear-to-mlp.typ
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#import "@preview/cetz:0.5.2" as cetz
#show: metropolis-deck.with(
  title: [From Linear Models to Neural Networks],
  subtitle: [Linear & logistic regression, neurons, MLPs, universal approximation],
)

// interactive base + a small local compute graph (fletcher)
#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let _sigmoid(z) = 1 / (1 + calc.exp(-z))
#let _tanh(z) = {
  let e = calc.exp(2 * z)
  (e - 1) / (e + 1)
}
#let _relu(z) = calc.max(0, z)
#let _gelu(z) = z * _sigmoid(1.702 * z)
#let _activation-panel(f, plot-title, color, y-range) = {
  let (y-min, y-max) = y-range
  lines(
    fn: (f, z => y-min, z => y-max), domain: (-4, 4), samples: 140,
    markers: (false, false, false),
    colors: (color, color.transparentize(100%), color.transparentize(100%)),
    hlines: ((0, none, MUTED),),
    x-label: [$z$], y-label: [$phi(z)$], title: plot-title, size: (78mm, 32mm),
  )
}
#let _sin-pwl(x, knots: 7) = {
  let left = -3.14
  let right = 3.14
  let step = (right - left) / (knots - 1)
  let raw-index = (x - left) / step
  let i = calc.max(0, calc.min(knots - 2, calc.floor(raw-index)))
  let x0 = left + i * step
  let x1 = x0 + step
  let y0 = calc.sin(x0)
  let y1 = calc.sin(x1)
  y0 + (x - x0) * (y1 - y0) / step
}
// Local diagram language: document-matched type, clean outlines, sparse arrows,
// and enough white space to read each transformation independently.
#let neat-card(title, body, color: TEAL, width: 54mm, height: auto, body-align: center) = block(
  width: width, height: height, inset: (x: 8pt, y: 7pt),
  fill: white, stroke: 0.9pt + color, radius: 3pt,
  [#text(size: 11.5pt, weight: 700, fill: color)[#upper(title)]
   #v(4pt)
   #align(body-align, body)],
)
#let neat-arrow(label: none, color: MUTED) = align(center, [
  #if label != none { [#text(size: 10.5pt, fill: color)[#label] #h(3pt)] }
  #text(size: 21pt, fill: color)[$arrow.r$]
])
#let neat-caption(body) = align(center, text(size: 14.5pt, fill: MUTED)[#body])
#let decision-score-plot = {
  let w = 103mm
  let h = 39mm
  let x-min = -1.2
  let x-max = 1.2
  let y-min = -1.2
  let y-max = 1.2
  let px(x) = (x - x-min) / (x-max - x-min) * w
  let py(y) = (y - y-min) / (y-max - y-min) * h
  let fmt(v) = if calc.abs(v) < 1e-9 { "0" } else { str(calc.round(v, digits: 1)) }
  cetz.canvas({
    import cetz.draw: line, content
    // Decision classes: class 2 wins on the left, class 3 in the middle,
    // and class 1 on the right. Curves and tie markers remain above the fill.
    for band in (
      (x-min, -0.5, TEAL.lighten(89%)),
      (-0.5, 0.5, ACC.lighten(89%)),
      (0.5, x-max, BLUE.lighten(89%)),
    ) {
      line((px(band.at(0)), 0), (px(band.at(1)), 0),
        (px(band.at(1)), h), (px(band.at(0)), h),
        close: true, fill: band.at(2), stroke: none)
    }
    for f in (0.25, 0.5, 0.75, 1.0) {
      line((0, f * h), (w, f * h), stroke: 0.4pt + MUTED.lighten(40%))
    }
    line((0, 0), (w, 0), stroke: 1.2pt + INK)
    line((0, 0), (0, h), stroke: 1.2pt + INK)
    for spec in (
      (-0.5, [visible tie], MUTED),
      (0, [hidden tie], RED),
      (0.5, [visible tie], MUTED),
    ) {
      line((px(spec.at(0)), 0), (px(spec.at(0)), h),
        stroke: (paint: spec.at(2), dash: "dashed", thickness: 0.8pt))
      content((px(spec.at(0)), h + 0.3em),
        text(size: 8pt, fill: spec.at(2), spec.at(1)), anchor: "south")
    }
    for curve in (
      (x => x, BLUE),
      (x => -x, TEAL),
      (x => 0.5, ACC),
    ) {
      let pts = range(101).map(i => {
        let x = x-min + i * (x-max - x-min) / 100
        (px(x), py(curve.at(0)(x)))
      })
      for i in range(pts.len() - 1) {
        line(pts.at(i), pts.at(i + 1), stroke: 1.4pt + curve.at(1))
      }
    }
    for f in (0.0, 0.5, 1.0) {
      content((-0.4em, f * h), text(size: 7.5pt, fill: MUTED, fmt(y-min + f * (y-max - y-min))), anchor: "east")
    }
    content((0, -0.4em), text(size: 8pt, fill: MUTED, fmt(x-min)), anchor: "north")
    content((w, -0.4em), text(size: 8pt, fill: MUTED, fmt(x-max)), anchor: "north")
    content((w/2, -1.6em), text(size: 9pt, fill: MUTED)[$x$], anchor: "north")
    content((-2.5em, h/2), rotate(-90deg, text(size: 9pt, fill: MUTED)[class score]), anchor: "south")
    let lx = w - 20mm
    for item in (
      ([$z_1=x$], BLUE),
      ([$z_2=-x$], TEAL),
      ([$z_3=0.5$], ACC),
    ).enumerate() {
      let i = item.at(0)
      let pair = item.at(1)
      let yy = h - 1.5mm - i * 1.15em
      line((lx, yy), (lx + 3.5mm, yy), stroke: 2.2pt + pair.at(1))
      content((lx + 4.5mm, yy), text(size: 8pt, fill: INK, pair.at(0)), anchor: "west")
    }
  })
}

// Coordinate grid and activation scale for the two learned-feature fields.
// The colour encodes a ReLU activation value, not a probability.
#let _feature-grid(size) = {
  let (w, h) = size
  cetz.canvas({
    import cetz.draw: line
    for k in range(1, 4) {
      line((k * w / 4, 0mm), (k * w / 4, h),
        stroke: 0.45pt + MUTED.transparentize(48%))
      line((0mm, k * h / 4), (w, k * h / 4),
        stroke: 0.45pt + MUTED.transparentize(48%))
    }
  })
}

#let _scale-cell(color, lightness) = block(
  width: 7mm, height: 3mm,
  fill: color.lighten(lightness), stroke: none,
)

#let _feature-field(fn, color, max-value, caption, marks) = {
  let size = (68mm, 38mm)
  let (w, ht) = size
  align(center, [
    #text(size: 9.5pt, weight: 650, fill: INK)[#caption]
    #v(1pt)
    #grid(
      columns: (8mm, w), column-gutter: 2mm, align: horizon,
      rotate(-90deg, text(size: 9pt, fill: MUTED)[$x_2$]),
      block(width: w, height: ht)[
        #place(top + left, contour(
          fn, xlim: (-0.5, 1.5), ylim: (-0.5, 1.5), samples: 64,
          levels: (0.25, 0.5, 1, 1.5), fill: true,
          ramp: (white, color.lighten(78%), color), color: color,
          marks: marks, size: size,
        ))
        #place(top + left, _feature-grid(size))
      ],
    )
    #grid(
      columns: (8mm, w), column-gutter: 2mm,
      [],
      grid(
        columns: (1fr, 1fr, 1fr, 1fr, 1fr), align: center,
        ..([-0.5], [$0$], [$0.5$], [$1$], [$1.5$]).map(v => text(size: 8.5pt, fill: MUTED, v)),
      ),
    )
    #grid(columns: (8mm, w), column-gutter: 2mm, [], align(center, text(size: 8.5pt, fill: MUTED)[$x_1$]))
    #v(1pt)
    #grid(
      columns: (8mm, w), column-gutter: 2mm, align: horizon,
      [],
      align(center, grid(
        columns: (auto, auto, 42mm, auto), column-gutter: 2pt, align: horizon,
        text(size: 8.5pt, fill: MUTED)[$h_j(bold(x)):$],
        text(size: 8.5pt)[$0$],
        grid(columns: 6, column-gutter: 0pt,
          _scale-cell(color, 92%), _scale-cell(color, 76%), _scale-cell(color, 60%),
          _scale-cell(color, 44%), _scale-cell(color, 26%), _scale-cell(color, 8%)),
        text(size: 8.5pt)[#max-value],
      )),
    )
  ])
}

#title-slide()

// ═══════════════════════════ PART I — Where we left off ═══════════════════════════
== Lecture 1 chose the loss; Lecture 2 expands the prediction rule

#two(r: (1fr, 1fr),
  [*Keep from Lecture 1*
   #v(4pt)
   #notebox[Regression output $arrow.r$ Gaussian model $arrow.r$ MSE]
   #v(5pt)
   #notebox[Class probabilities $arrow.r$ Bernoulli / categorical model $arrow.r$ cross-entropy]],
  [#pause
   *Change in Lecture 2*
   #v(4pt)
   Start with one weighted sum, expose what it cannot represent, then let hidden layers learn nonlinear features.],
)
#pause
#v(7pt)
#align(center, text(size: 21pt)[$bold(x) arrow.r f_(bold(theta))(bold(x)) arrow.r "output parameters" arrow.r "same Lecture 1 loss"$])
#pause
#place(bottom + center, dy: -2pt,
  result[We will not re-derive the losses; today is about making $f_(bold(theta))$ more expressive.])

== Outline

#[
  #set text(size: 22pt)
  #enum(
    [*Linear prediction* — weighted sums, batches, and output shapes],
    [*One neuron* — add an activation to create a nonlinear feature],
    [*The limitation* — see exactly why a linear model cannot solve XOR],
    [*An MLP* — compose learned features across width and depth],
    [*Practice* — trace one forward pass, count parameters, and make one update],
    spacing: 9pt,
  )
]

== One notation covers every prediction rule
#stag(Q)

Write $f_(bold(theta))(bold(x))$ for the model's prediction at input $bold(x)$.
#v(5pt)

#two(r: (1fr, 1.1fr),
  [*$bold(x)$*: the input features \
   *$bold(theta)$*: all trainable weights and biases \
   *$f_(bold(theta))$*: the rule that maps input to a score or prediction],
  [#pause
   *Starting point* — one weighted sum plus a bias:
   $ f_(bold(theta))(bold(x)) = bold(w)^top bold(x) + b $],
)
#pause
#alertbox[We will build the network gradually; the full nested formula comes only after each part is concrete.]

// ═══════════════════════════ PART II — Linear starting point ═══════════════════════════
= The linear starting point

== Start with one weighted sum

For one output, every input node sends a weighted value into one summation node.
#pause
#v(4pt)
#align(center, diagram(spacing: (28mm, 15mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let x1 = (0, -1)
  let x2 = (0, 0)
  let bias = (0, 1)
  let sum-node = (3, 0)
  let out = (4.7, 0)

  edge(x1, sum-node, text(size: 12.5pt, weight: 700, fill: BLUE)[$w_1=1.5$],
    "-|>", stroke: 1.05pt + BLUE, label-side: left)
  edge(x2, sum-node, text(size: 12.5pt, weight: 700, fill: GREEN)[$w_2=0.5$],
    "-|>", stroke: 1.05pt + GREEN, label-side: left)
  edge(bias, sum-node, text(size: 12.5pt, weight: 700, fill: RED)[$b=-0.2$],
    "-|>", stroke: 1.05pt + RED, label-side: right)
  edge(sum-node, out, text(size: 12.5pt, weight: 700, fill: ACC)[$z=2.3$],
    "-|>", stroke: 1.15pt + ACC, label-side: left)

  node(x1, align(center)[$x_1$ \
    #text(size: 11pt, fill: MUTED)[$2$]], radius: 9mm, stroke: 1.1pt + BLUE)
  node(x2, align(center)[$x_2$ \
    #text(size: 11pt, fill: MUTED)[$-1$]], radius: 9mm, stroke: 1.1pt + GREEN)
  node(bias, $1$, radius: 7mm, stroke: 1.1pt + RED)
  node(sum-node, $sum$, radius: 8mm, stroke: 1.2pt + TEAL)
  node(out, $hat(y)$, radius: 8mm, stroke: 1.2pt + ACC)

  node((0, -1.75), text(size: 10.5pt, weight: 650, fill: MUTED)[INPUT NODES], stroke: none)
  node((1.55, -1.75), text(size: 10.5pt, weight: 650, fill: MUTED)[EDGE PARAMETERS], stroke: none)
  node((3, -1.75), text(size: 10.5pt, weight: 650, fill: MUTED)[WEIGHTED SUM], stroke: none)
  node((4.7, -1.75), text(size: 10.5pt, weight: 650, fill: MUTED)[OUTPUT], stroke: none)
}))
#pause
#place(bottom + center, dy: -2pt,
  result[The constant node $1$ turns the bias into one more incoming edge: $b dot 1$.])

== Calculate every contribution entering the node #D

#align(center, grid(
  columns: (45mm, 7mm, 45mm, 7mm, 45mm, 7mm, 52mm), align: horizon,
  neat-card([blue edge], [$w_1x_1$ \
    $=1.5(2)$ \
    $=3.0$], color: BLUE, width: 45mm),
  neat-arrow(),
  neat-card([green edge], [$w_2x_2$ \
    $=0.5(-1)$ \
    $=-0.5$], color: GREEN, width: 45mm),
  neat-arrow(),
  neat-card([bias edge], [$b dot 1$ \
    $=-0.2(1)$ \
    $=-0.2$], color: RED, width: 45mm),
  neat-arrow(),
  neat-card([sum at node], [$3.0-0.5-0.2$ \
    $=2.3$], color: ACC, width: 52mm),
))
#pause
#v(12pt)
#align(center, text(size: 22pt)[
  $hat(y)=underbrace(1.5(2))_(3.0)+underbrace(0.5(-1))_(-0.5)+underbrace((-0.2)(1))_(-0.2)=2.3.$
])
#pause
#place(bottom + center, dy: -2pt,
  result[Each edge contributes “weight × incoming value”; the node adds all contributions.])

== A batch is a labelled data matrix #V

#two(r: (1.28fr, 0.72fr),
  [#align(center, {
    set text(size: 15.5pt)
    table(
      columns: (35mm, 30mm, 30mm, 38mm),
      stroke: 0.6pt + MUTED, inset: (x: 7pt, y: 6pt), align: center,
      table.header([*example*], [*feature $x_1$*], [*feature $x_2$*], [*target $y$*]),
      [$1$], [$2$], [$-1$], [$2.4$],
      [$2$], [$0$], [$3$], [$1.0$],
      [$3$], [$-1$], [$1$], [$-1.1$],
    )
  })],
  [#align(center, [
    #neat-card([data matrix X], [
      $bold(X)=mat(2,-1;0,3;-1,1)$ \
      $3$ rows $=$ examples \
      $2$ columns $=$ features
    ], color: TEAL, width: 79mm)
    #v(8pt)
    #neat-card([target vector y], [
      $bold(y)=mat(2.4;1.0;-1.1)$ \
      one observed target per example
    ], color: ACC, width: 79mm)
  ])],
)
#pause
#v(7pt)
#align(center, text(size: 18pt)[
  Here $n=3$ examples and $d=2$ input features, so $bold(X) in RR^(3 times 2)$ and $bold(y) in RR^3$.
])
#pause
#place(bottom + center, dy: -2pt,
  result[Row $i$ contains the input $bold(x)^((i))$ and its observed answer $y^((i))$.])

== One matrix multiplication predicts all three rows #D

#align(center, text(size: 18pt)[
  Reuse $bold(w)=mat(1.5;0.5)$ and $b=-0.2$ for every example.
])
#pause
#v(6pt)
#align(center, text(size: 20pt)[
  $bold(X)bold(w)+b bold(1)
   = mat(2,-1;0,3;-1,1) mat(1.5;0.5) + (-0.2)mat(1;1;1)$
])
#pause
#v(7pt)
#align(center, text(size: 22pt)[
  $= mat(2.5;1.5;-1.0) + mat(-0.2;-0.2;-0.2)
   = mat(2.3;1.3;-1.2) = hat(bold(y)).$
])
#pause
#v(9pt)
#align(center, grid(
  columns: (62mm, 10mm, 62mm, 10mm, 62mm), align: horizon,
  neat-card([row 1], [$1.5(2)+0.5(-1)-0.2=2.3$], color: BLUE, width: 62mm),
  [],
  neat-card([row 2], [$1.5(0)+0.5(3)-0.2=1.3$], color: TEAL, width: 62mm),
  [],
  neat-card([row 3], [$1.5(-1)+0.5(1)-0.2=-1.2$], color: GREEN, width: 62mm),
))
#pause
#place(bottom + center, dy: -2pt,
  result[$hat(bold(y))$ contains predictions; $bold(y)$ contains observed targets. They are not the same object.])

== The bias node becomes a column of ones #V

#two(r: (1.12fr, 0.88fr),
  [#align(center, [
    #text(size: 18pt, weight: 650, fill: TEAL)[AUGMENT THE DATA MATRIX]
    #v(6pt)
    $tilde(bold(X))
      = mat(2,-1,1;0,3,1;-1,1,1)$
    #v(12pt)
    #text(size: 18pt, weight: 650, fill: ACC)[STACK THE PARAMETERS]
    #v(6pt)
    $bold(theta)=mat(w_1;w_2;b)=mat(1.5;0.5;-0.2)$
  ])],
  [#align(center, [
    #neat-card([$tilde(bold(X))$], [
      original feature columns \
      $+$ one constant column
    ], color: TEAL, width: 84mm)
    #v(8pt)
    #neat-card([$bold(theta)$], [
      all weights \
      $+$ the bias
    ], color: ACC, width: 84mm)
  ])],
)
#pause
#v(10pt)
#align(center, text(size: 23pt)[
  $hat(bold(y))=tilde(bold(X))bold(theta), quad
   (n times (d+1))((d+1) times 1) arrow.r (n times 1).$
])
#pause
#place(bottom + center, dy: -2pt,
  result[The column of ones is the batch version of the constant bias node on the previous network diagram.])

== For regression, one weighted sum gives one flat trend #V

#align(center, text(size: 19pt)[$hat(y)=bold(w)^top bold(x)+b$])
#v(4pt)
#pause
#fig("/lecture2/figures/linreg_geometry.pdf", w: 52%)
#pause
#place(bottom + center, dy: -2pt,
  result[The model can tilt and shift a line or plane, but it cannot bend to follow nonlinear structure.])

// ═══════════════════════════ PART III — From linear score to a neuron ═══════════════════════════
= From a linear score to a neuron

== A neuron has two stages

First compute a weighted sum plus bias; then pass that value through an activation.
#pause
#v(3pt)
#align(center, diagram(spacing: (22mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let x1 = (0, -1.35)
  let x2 = (0, -0.45)
  let xd = (0, 0.45)
  let bias = (0, 1.35)
  let sum-node = (3, 0)
  let act-node = (4.55, 0)
  let out = (6.1, 0)

  edge(x1, sum-node, text(size: 12pt, weight: 700, fill: BLUE)[$w_1$],
    "-|>", stroke: 0.95pt + BLUE, label-side: left)
  edge(x2, sum-node, text(size: 12pt, weight: 700, fill: TEAL)[$w_2$],
    "-|>", stroke: 0.95pt + TEAL, label-side: left)
  edge(xd, sum-node, text(size: 12pt, weight: 700, fill: GREEN)[$w_d$],
    "-|>", stroke: 0.95pt + GREEN, label-side: right)
  edge(bias, sum-node, text(size: 12pt, weight: 700, fill: RED)[$b$],
    "-|>", stroke: 0.95pt + RED, label-side: right)
  edge(sum-node, act-node, text(size: 12pt, weight: 700, fill: TEAL)[$z$],
    "-|>", stroke: 1.1pt + TEAL, label-side: left)
  edge(act-node, out, text(size: 12pt, weight: 700, fill: ACC)[$a$],
    "-|>", stroke: 1.1pt + ACC, label-side: left)

  node(x1, $x_1$, radius: 6.5mm, stroke: 1.05pt + BLUE)
  node(x2, $x_2$, radius: 6.5mm, stroke: 1.05pt + TEAL)
  node(xd, $x_d$, radius: 6.5mm, stroke: 1.05pt + GREEN)
  node(bias, $1$, radius: 6.5mm, stroke: 1.05pt + RED)
  node(sum-node, $sum$, radius: 8mm, stroke: 1.2pt + TEAL)
  node(act-node, $phi$, radius: 8mm, stroke: 1.2pt + ACC)
  node(out, $a$, radius: 7mm, stroke: 1.2pt + ACC)

  node((0, -2.05), text(size: 10.5pt, weight: 650, fill: MUTED)[INPUTS + CONSTANT], stroke: none)
  node((1.5, -2.05), text(size: 10.5pt, weight: 650, fill: MUTED)[WEIGHTS + BIAS], stroke: none)
  node((3, -2.05), text(size: 10.5pt, weight: 650, fill: MUTED)[SUMMATION], stroke: none)
  node((4.55, -2.05), text(size: 10.5pt, weight: 650, fill: MUTED)[ACTIVATION], stroke: none)
  node((6.1, -2.05), text(size: 10.5pt, weight: 650, fill: MUTED)[OUTPUT], stroke: none)
}))
#pause
#place(bottom + center, dy: -2pt,
  result[Weights and bias create $z$ at the $sum$ node; the separate $phi$ node reshapes $z$ into $a$.])

== Trace one concrete neuron through the two stages #D

#align(center, grid(
  columns: (52mm, 12mm, 76mm, 12mm, 58mm), align: horizon,
  neat-card([input], [$bold(x)=(2,-1)$], color: INK, width: 52mm),
  neat-arrow(),
  neat-card([stage 1 · summation], [
    $bold(w)=(1,-1), quad b=0$ \
    $z=1(2)+(-1)(-1)+0=3$
  ], color: TEAL, width: 76mm),
  neat-arrow(),
  neat-card([stage 2 · activation], [$a="ReLU"(3)=3$], color: ACC, width: 58mm),
))
#pause
#v(9pt)
#two(
  notebox[Changing $bold(w)$ or $b$ changes which input patterns make $z$ large.],
  notebox[Changing $phi$ changes how the same pre-activation $z$ is gated or reshaped.],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Topology tells us where values flow; this calculation tells us the number carried along each stage.])

== Bounded activations behave like smooth switches #V

#two(
  [#align(center, text(size: 16.5pt, weight: 650, fill: BLUE)[
     $sigma(z)=frac(1,1+exp(-z))$
   ])
   #v(3pt)
   #align(center, lines(
     fn: _sigmoid, domain: (-4, 4), samples: 140, markers: false, colors: (BLUE,),
     points: ((-4, _sigmoid(-4), [$0.018$]), (0, 0.5, [$0.5$]), (4, _sigmoid(4), [$0.982$])),
     x-label: [$z$], y-label: [$sigma(z)$], title: [sigmoid: off $arrow.r$ on], size: (93mm, 49mm),
   ))],
  [#pause
   #align(center, text(size: 16.5pt, weight: 650, fill: TEAL)[
     $"tanh"(z)=frac(exp(z)-exp(-z),exp(z)+exp(-z))$
   ])
   #v(3pt)
   #align(center, lines(
     fn: _tanh, domain: (-4, 4), samples: 140, markers: false, colors: (TEAL,),
     points: ((-4, _tanh(-4), [$-0.999$]), (0, 0, [$0$]), (4, _tanh(4), [$0.999$])),
     x-label: [$z$], y-label: [$"tanh"(z)$], title: [tanh: negative $arrow.r$ positive], size: (93mm, 49mm),
   ))],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Sigmoid answers “how on?”; tanh answers “negative or positive, and how strongly?”])

== ReLU and GELU behave like ramps #V

#two(
  [#align(center, text(size: 16.5pt, weight: 650, fill: ACC)[
     $"ReLU"(z)=max(0,z)$
   ])
   #v(3pt)
   #align(center, lines(
     fn: _relu, domain: (-4, 4), samples: 140, markers: false, colors: (ACC,),
     points: ((-2, 0, [off]), (0, 0, [hinge]), (2, 2, [linear])),
     x-label: [$z$], y-label: [$"ReLU"(z)$], title: [ReLU: hard gate + ramp], size: (93mm, 49mm),
   ))],
  [#pause
   #align(center, text(size: 16.5pt, weight: 650, fill: GREEN)[
     $"GELU"(z) approx z sigma(1.702z)$
   ])
   #v(3pt)
   #align(center, lines(
     fn: _gelu, domain: (-4, 4), samples: 140, markers: false, colors: (GREEN,),
     points: ((-2, _gelu(-2), [near zero]), (0, 0, [$0$]), (2, _gelu(2), [near linear])),
     x-label: [$z$], y-label: [$"GELU"(z)$], title: [GELU: smooth gate + ramp], size: (93mm, 49mm),
   ))],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Both build ramp-like hidden features; ReLU has a sharp hinge, while GELU eases through the gate.])

== Saturation means the curve has become locally flat #V

#two(
  [#align(center, lines(
     fn: _sigmoid, domain: (-7, 7), samples: 180, markers: false, colors: (BLUE,),
     vlines: ((0, [responsive centre], MUTED), (6, [saturated], RED)),
     points: ((0, 0.5, [$0.5$]), (6, _sigmoid(6), [$0.9975$])),
     x-label: [$z$], y-label: [$sigma(z)$], title: [output], size: (94mm, 52mm),
   ))],
  [#pause
   #align(center, lines(
     fn: z => _sigmoid(z) * (1 - _sigmoid(z)), domain: (-7, 7), samples: 180,
     markers: false, colors: (ACC,),
     points: ((0, 0.25, [$0.25$]), (6, 0.00247, [$0.0025$])),
     x-label: [$z$], y-label: [slope], title: [local sensitivity $sigma'(z)$], size: (94mm, 52mm),
   ))],
)
#pause
#v(4pt)
#align(center, text(size: 17pt)[Near $z=0$, the curve is steep; near $z=6$, it is almost horizontal.])
#pause
#place(bottom + center, dy: -2pt,
  result[A saturated unit still outputs a number, but changing its input barely changes that output.])

== Calculate the effect of saturation #D

#align(center, grid(
  columns: (77mm, 12mm, 77mm), align: horizon,
  neat-card([centre: z = 0], [
    $sigma'(0)=0.25$ \
    for $Delta z=0.1$: \
    $Delta h approx 0.25(0.1)=0.025$
  ], color: BLUE, width: 77mm),
  neat-arrow(label: [same $Delta z$]),
  neat-card([saturated: z = 6], [
    $sigma'(6) approx 0.0025$ \
    for $Delta z=0.1$: \
    $Delta h approx 0.0025(0.1)=0.00025$
  ], color: ACC, width: 77mm),
))
#pause
#v(9pt)
#align(center, text(size: 20pt, weight: 650)[
  $0.025 / 0.00025 = 100$ $arrow.r$ the saturated unit responds *100× less*.
])
#pause
#notebox[For a positive ReLU input, $"ReLU"'(z)=1$, so the same $Delta z=0.1$ gives $Delta h approx 0.1$.]
#pause
#place(bottom + center, dy: -2pt,
  result[The derivative is a local responsiveness meter; Lecture 3 will connect it to the learning signal.])

== Question: do more linear layers create new shapes? #Q

#mcq(
  [Remove every activation. What can a stack of “weighted sum + bias” layers represent?],
  [One equivalent weighted sum + bias],
  [Any curved decision boundary],
  [Only a constant function],
  [A probability distribution automatically],
)

== Answer in nodes: two linear layers collapse into one #A

#align(center, grid(
  columns: (96mm, 18mm, 96mm), align: horizon,
  align(center, [
    #text(size: 18pt, weight: 700, fill: TEAL)[TWO LINEAR LAYERS]
    #v(5pt)
    #mlp-diagram(
      (2, 3, 1),
      labels: ([input $bold(x)$], [linear $bold(h)$], [output $bold(y)$]),
    )
    #v(6pt)
    #text(size: 15pt)[$bold(h)=bold(W)_1bold(x)+bold(b)_1$ \
      $bold(y)=bold(W)_2bold(h)+bold(b)_2$]
  ]),
  align(center, text(size: 31pt, weight: 700, fill: ACC)[$equiv$]),
  align(center, [
    #text(size: 18pt, weight: 700, fill: ACC)[ONE EQUIVALENT LAYER]
    #v(5pt)
    #mlp-diagram(
      (2, 1),
      labels: ([input $bold(x)$], [output $bold(y)$]),
    )
    #v(6pt)
    #text(size: 15pt)[$bold(y)=bold(W)_("eq")bold(x)+bold(b)_("eq")$ \
      no hidden representation is needed]
  ]),
))
#pause
#v(5pt)
#align(center, text(size: 18pt, weight: 650)[
  $bold(W)_("eq")=bold(W)_2bold(W)_1, quad
   bold(b)_("eq")=bold(W)_2bold(b)_1+bold(b)_2$
])
#pause
#place(bottom + center, dy: -2pt,
  result[Without a $phi$ node, the middle column adds parameters but no new bend or feature shape.])

== Interactive: activation functions #I

#interbox(link-to: "https://nipunbatra.github.io/dl-teaching/interactives/activation-explorer.html")[
  Compare activations and their *derivatives*, and see where gradients vanish.
  Controls: activation type · input $z$ · show derivative · compare saturation.
]
#pause
Explore large positive and negative logits, where sigmoid's derivative becomes tiny.

// ═══════════════════════════ PART IV — Where linear scores fail ═══════════════════════════
= Where a single weighted sum fails

== A binary classifier draws one straight boundary #V

#two(r: (0.86fr, 1.14fr),
  [#align(center, [
    #neat-card([linear score], [
      $z=bold(w)^top bold(x)+b$
    ], color: TEAL, width: 78mm)
    #v(8pt)
    #neat-card([threshold], [
      $hat(y)=1$ if $z>=0$ \
      $hat(y)=0$ if $z<0$
    ], color: ACC, width: 78mm)
    #v(8pt)
    #neat-caption([The boundary is where $z=0$.])
  ])],
  [#pause
   #align(center, fig("/lecture2/figures/linear_boundary.pdf", w: 72%))],
)
#pause
#place(bottom + center, dy: -2pt,
  result[$bold(w)^top bold(x)+b=0$ is one line in 2-D, so binary linear classification works only when one line can separate the classes.])

== XOR is the smallest pattern that defeats one line #V

#two(r: (0.95fr, 1.05fr),
  [#align(center, [
    #neat-card([class A], [$(0,0)$ and $(1,1)$], color: TEAL, width: 82mm)
    #v(8pt)
    #neat-card([class B], [$(1,0)$ and $(0,1)$], color: ACC, width: 82mm)
    #v(9pt)
    Every proposed line leaves one same-class corner on the wrong side.
  ])],
  [#pause
   #fig("/lecture2/figures/xor.pdf", w: 80%)],
)
#pause
#place(bottom + center, dy: -2pt,
  result[XOR requires a nonlinear feature transformation before a final straight separator can work.])

== Return to multi-class: several scores create straight-edged regions #V

#two(r: (0.9fr, 1.1fr),
  [For each class $k$, compute one linear score
   $z_k(bold(x))=bold(w)_k^top bold(x)+b_k.$
   #pause
   Predict the largest score:
   $hat(y)(bold(x))=arg max_k z_k(bold(x)).$
   #pause
   #v(8pt)
   #notebox[Each pair of classes can exchange the lead only where their two scores tie.]],
  [#pause
   #align(center, fig("/lecture2/figures/softmax_regions.pdf", w: 72%))],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Multi-class regions can form polygons, but every edge still comes from equality between two linear scores.])

== Question: why are the multi-class boundaries straight? #Q

#two(r: (0.92fr, 1.08fr),
  [#align(center, fig("/lecture2/figures/softmax_regions.pdf", w: 74%))
   #v(4pt)
   #align(center, text(size: 15pt, fill: MUTED)[three linear scores divide the input plane])],
  [The classifier predicts
   $ hat(y)(bold(x)) = arg max_k p_k(bold(x)). $
   #v(6pt)
   *When is a point on the boundary between predicted classes $i$ and $j$?*
   #v(8pt)
   #grid(columns: 1, row-gutter: 8pt,
     [*A.* Whenever $p_i=p_j$],
     [*B.* When $p_i=p_j$ and both are maximal],
     [*C.* Whenever $p_i+p_j=1$],
     [*D.* Only when all $K$ probabilities are equal],
   )
  ],
)

== Softmax preserves the winner #A

#align(center, grid(
  columns: (70mm, 14mm, 70mm), align: horizon,
  neat-card([scores], [$z_i=2, quad z_j=1$], color: INK, width: 70mm),
  neat-arrow(),
  neat-card([softmax probabilities], [$p_i approx 0.731, quad p_j approx 0.269$], color: TEAL, width: 70mm),
))
#pause
#v(9pt)
#align(center, text(size: 20pt)[
  $frac(p_i,p_j)=frac(exp(z_i),exp(z_j))=exp(z_i-z_j)$
])
#pause
#v(7pt)
#align(center, grid(
  columns: (67mm, 10mm, 67mm), align: horizon,
  neat-card([equal scores], [$z_i=z_j arrow.l.r p_i=p_j$], color: BLUE, width: 67mm),
  neat-arrow(),
  neat-card([larger score], [$z_i>z_j arrow.l.r p_i>p_j$], color: ACC, width: 67mm),
))
#pause
#place(bottom + center, dy: -2pt,
  result[Softmax changes score scale, not score order: $arg max_k p_k=arg max_k z_k$.])

== Linear scores can tie only on a flat set #A

#two(r: (1fr, 1fr),
  [#align(center, fig("/lecture2/figures/softmax_regions.pdf", w: 76%))
   #v(5pt)
   #neat-caption([Each coloured region is where one linear score is largest.])],
  [For class $k$,
   $z_k(bold(x))=bold(w)_k^top bold(x)+b_k.$
   #pause
   Two classes tie when
   $z_i(bold(x))=z_j(bold(x)),$
   so
   $(bold(w)_i-bold(w)_j)^top bold(x)+(b_i-b_j)=0.$
   #pause
   #v(7pt)
   #notebox[This is a line in 2-D, a plane in 3-D, and a hyperplane in $d$ dimensions.]],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Equal linear scores create a candidate straight boundary; softmax cannot bend it.])

== A concrete example: three linear scores compete #V

#two(r: (0.78fr, 1.22fr),
  [#align(center, [
    #neat-card([class 1 score], [$z_1(x)=x$], color: BLUE, width: 74mm)
    #v(4pt)
    #neat-card([class 2 score], [$z_2(x)=-x$], color: TEAL, width: 74mm)
    #v(4pt)
    #neat-card([class 3 score], [$z_3(x)=0.5$], color: ACC, width: 74mm)
  ])],
  [#pause
   #align(center, [
     #decision-score-plot
     #v(3pt)
     #text(size: 10.5pt, weight: 700, fill: INK)[PREDICTED CLASS $= arg max_k z_k(x)$ · boundaries at $x=plus.minus 0.5$]
     #v(2pt)
     #block(width: 103mm)[
       #grid(
         columns: (7fr, 10fr, 7fr), column-gutter: 0pt, align: horizon,
         block(height: 10mm, inset: (x: 2pt, y: 1pt),
           fill: TEAL.lighten(84%), stroke: 0.8pt + TEAL,
           align(center + horizon, text(size: 8.5pt, weight: 650, fill: TEAL)[2: $x < -.5$])),
         block(height: 10mm, inset: (x: 2pt, y: 1pt),
           fill: ACC.lighten(87%), stroke: 0.8pt + ACC,
           align(center + horizon, text(size: 8.5pt, weight: 650, fill: ACC)[3: $abs(x) < .5$])),
         block(height: 10mm, inset: (x: 2pt, y: 1pt),
           fill: BLUE.lighten(87%), stroke: 0.8pt + BLUE,
           align(center + horizon, text(size: 8.5pt, weight: 650, fill: BLUE)[1: $x > .5$])),
       )
     ]
   ])],
)
#pause
#place(bottom + center, dy: -2pt,
  result[The learned boundaries are $x=-0.5$ and $x=0.5$; the class-1/class-2 tie at $x=0$ remains hidden.])

== Evaluate five explicit inputs to see which ties are visible #D

#align(center, text(size: 17pt)[Prediction $=arg max_k z_k(x)$; a boundary appears only when the tied scores are maximal.])
#v(6pt)
#align(center, {
  set text(size: 14.5pt)
  table(
    columns: (24mm, 29mm, 29mm, 29mm, 84mm),
    stroke: 0.55pt + MUTED, inset: (x: 6pt, y: 4.5pt), align: center,
    table.header([$bold(x)$], [$z_1=x$], [$z_2=-x$], [$z_3=0.5$], [*what is visible?*]),
    [$-1$], [$-1$], [$1$], [$0.5$], [class 2 wins],
    [$-0.5$], [$-0.5$], [$0.5$], [$0.5$], [*visible boundary 2/3*],
    [$0$], [$0$], [$0$], [$0.5$], [#text(fill: RED, weight: 700)[class 3 wins; 1/2 tie hidden]],
    [$0.5$], [$0.5$], [$-0.5$], [$0.5$], [*visible boundary 1/3*],
    [$1$], [$1$], [$-1$], [$0.5$], [class 1 wins],
  )
})
#pause
#place(bottom + center, dy: -2pt,
  result[The candidate 1/2 tie occurs at $x=0$, but the visible boundaries occur at $x=-0.5$ and $x=0.5$.])

== Candidate boundary versus visible boundary #A

#align(center, grid(
  columns: (83mm, 14mm, 83mm), align: horizon,
  neat-card([candidate pairwise tie], [
    $cal(T)_(i j)={bold(x):z_i=z_j}$ \
    an entire hyperplane
  ], color: MUTED, width: 83mm),
  neat-arrow(label: [keep only winners], color: ACC),
  neat-card([visible class boundary], [
    $cal(B)_(i j)={bold(x):z_i=z_j>=z_l " for every " l}$ \
    the top-scoring portion
  ], color: ACC, width: 83mm),
))
#pause
#v(11pt)
#align(center, text(size: 20pt)[
  In probability language: $p_i=p_j$ *and both are maximal*.
])
#pause
#place(bottom + center, dy: -2pt,
  result[Correct: *B*. Multi-class linear classifiers have piecewise-straight visible boundaries.])

// ═══════════════════════════ PART V — Multilayer perceptron ═══════════════════════════
= The multilayer perceptron

== A hidden layer is a new column of learned features #V

#mlp-diagram(
  (3, 5, 2),
  labels: ([$3$ input features], [$5$ learned features], [$2$ class scores]),
)
#pause
#v(8pt)
#align(center, text(size: 18pt)[
  $bold(h)=phi(bold(W)_1bold(x)+bold(b)_1)$, #h(12pt)
  $bold(z)=bold(W)_2bold(h)+bold(b)_2$.
])
#pause
#place(bottom + center, dy: -2pt,
  result[$bold(W)_1$ stores the $3 times 5=15$ blue-to-teal edges; $bold(W)_2$ stores the $5 times 2=10$ teal-to-orange edges.])

== General layer dimensions must agree #D

#align(center, grid(
  columns: (61mm, 12mm, 76mm, 12mm, 61mm), align: horizon,
  neat-card([input], [$bold(x) in RR^d$], color: INK, width: 61mm),
  neat-arrow(),
  neat-card([hidden layer], [
    $bold(W)_1 in RR^(m times d)$ \
    $bold(b)_1 in RR^m$ \
    $bold(h) in RR^m$
  ], color: TEAL, width: 76mm),
  neat-arrow(),
  neat-card([output layer], [
    $bold(W)_2 in RR^(K times m)$ \
    $bold(b)_2 in RR^K$ \
    $bold(z) in RR^K$
  ], color: ACC, width: 61mm),
))
#pause
#v(11pt)
#align(center, text(size: 20pt)[
  $(m times d)(d times 1) arrow.r (m times 1), quad (K times m)(m times 1) arrow.r (K times 1)$
])
#pause
#place(bottom + center, dy: -2pt,
  result[The inner dimensions cancel; the number of rows determines the next representation size.])

== One XOR network: keep these two features throughout #V

#align(center, text(size: 18.5pt)[
  Both hidden units receive the same pair $bold(x)=(x_1,x_2)$:
])
#v(5pt)
#align(center, text(size: 20pt)[
  #text(fill: BLUE)[$h_1="ReLU"(x_1+x_2)$], #h(14pt)
  #text(fill: GREEN)[$h_2="ReLU"(x_1+x_2-1)$]
])
#pause
#v(5pt)
#align(center, diagram(spacing: (31mm, 18mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let x1 = (0, -0.48)
  let x2 = (0, 0.48)
  let h1 = (2, -0.48)
  let h2 = (2, 0.48)
  let out = (4, 0)

  edge(x1, h1, "-|>", stroke: 1.1pt + BLUE)
  edge(x2, h1, "-|>", stroke: 1.1pt + BLUE)
  edge(x1, h2, "-|>", stroke: 1.1pt + GREEN)
  edge(x2, h2, "-|>", stroke: 1.1pt + GREEN)
  edge(h1, out, text(size: 13pt, weight: 700, fill: BLUE)[$+2$], "-|>", stroke: 1.3pt + BLUE, label-side: left)
  edge(h2, out, text(size: 13pt, weight: 700, fill: GREEN)[$-4$], "-|>", stroke: 1.3pt + GREEN, label-side: right)

  node(x1, $x_1$, radius: 7mm, stroke: 1pt + INK)
  node(x2, $x_2$, radius: 7mm, stroke: 1pt + INK)
  node(h1, align(center, text(size: 12pt)[$h_1$ \
    #text(size: 9pt, fill: BLUE)[bias $0$]]), radius: 9mm, stroke: 1.2pt + BLUE)
  node(h2, align(center, text(size: 12pt)[$h_2$ \
    #text(size: 9pt, fill: GREEN)[bias $-1$]]), radius: 9mm, stroke: 1.2pt + GREEN)
  node(out, align(center, text(size: 13pt)[$z$ \
    #text(size: 9pt, fill: ACC)[bias $-1$]]), radius: 9.5mm, stroke: 1.2pt + ACC)
  node((1, -1.48), text(size: 11pt, weight: 700, fill: MUTED)[all four input weights $=+1$], stroke: none)
}))
#pause
#place(bottom + center, dy: -2pt,
  result[The output node computes $z=2h_1-4h_2-1$; we keep this exact network throughout the XOR example.])

== Verify the XOR network on all four inputs #D

#align(center, {
  set text(size: 13pt)
  table(
    columns: (25mm, 28mm, 30mm, 32mm, 30mm, 31mm, 28mm),
    stroke: 0.5pt + MUTED, inset: (x: 6pt, y: 4pt),
    align: center,
    table.header([*Input*], [$u_1=x_1+x_2$], [$h_1$], [$u_2=x_1+x_2-1$], [$h_2$], [$z=2h_1-4h_2-1$], [*$hat(y)$*]),
    [$(0,0)$], [$0$], [$0$], [$-1$], [$0$], [$-1$], [$0$],
    [$(1,0)$], [$1$], [$1$], [$0$], [$0$], [$1$], [$1$],
    [$(0,1)$], [$1$], [$1$], [$0$], [$0$], [$1$], [$1$],
    [$(1,1)$], [$2$], [$2$], [$1$], [$1$], [$-1$], [$0$],
  )
})
#pause
#v(4pt)
#align(center, text(size: 14.5pt)[
  $bold(W)_1=mat(1,1; 1,1), quad bold(b)_1=mat(0; -1), quad bold(w)_"out"=mat(2;-4), quad b_"out"=-1.$
])
#pause
#place(bottom + center, dy: -2pt,
  result[Predict $1$ when $z >= 0$. One ordinary output node correctly classifies all four XOR inputs.])

== First derive the feature functions for a general input #D

#align(center, text(size: 20pt)[
  Let $bold(x)=mat(x_1;x_2)$. The weights and bias define a *function* before we substitute any particular input.
])
#pause
#v(8pt)
#two(
  [#neat-card([blue hidden unit], [
    $bold(w)_1=mat(1;1), quad b_1=0$ \
    #v(5pt)
    $u_1(bold(x))=bold(w)_1^top bold(x)+b_1$ \
    $=1x_1+1x_2+0=x_1+x_2$ \
    #v(5pt)
    #text(fill: BLUE, weight: 700)[$h_1(bold(x))$] \
    #text(fill: BLUE, weight: 700)[$="ReLU"(x_1+x_2)$]
  ], color: BLUE, width: 98mm, body-align: left)],
  [#pause
   #neat-card([green hidden unit], [
    $bold(w)_2=mat(1;1), quad b_2=-1$ \
    #v(5pt)
    $u_2(bold(x))=bold(w)_2^top bold(x)+b_2$ \
    $=1x_1+1x_2-1=x_1+x_2-1$ \
    #v(5pt)
    #text(fill: GREEN, weight: 700)[$h_2(bold(x))$] \
    #text(fill: GREEN, weight: 700)[$="ReLU"(x_1+x_2-1)$]
  ], color: GREEN, width: 98mm, body-align: left)],
)
#pause
#place(bottom + center, dy: -2pt,
  result[These are exactly the two XOR features already shown in the node diagram; now we evaluate them at one input.])

== Check both hidden calculations in matrix form #D

#align(center, text(size: 20pt)[
  $bold(u)=bold(W)_1 bold(x)+bold(b)_1$
])
#v(5pt)
#align(center, text(size: 18pt)[
  $= mat(1,1;1,1) mat(1;0) + mat(0;-1).$
])
#v(6pt)
#grid(
  columns: (1fr, 1fr), column-gutter: 14pt,
  block(inset: 8pt, fill: BLUE.lighten(93%), stroke: 0.9pt + BLUE, radius: 4pt,
    align(center, text(size: 16.5pt, fill: BLUE, weight: 650)[
      $u_1=1(1)+1(0)+0=1$
    ])),
  block(inset: 8pt, fill: GREEN.lighten(94%), stroke: 0.9pt + GREEN, radius: 4pt,
    align(center, text(size: 16.5pt, fill: GREEN, weight: 650)[
      $u_2=1(1)+1(0)-1=0$
    ])),
)
#pause
#v(6pt)
#align(center, text(size: 21pt)[
  $bold(h)="ReLU"(bold(u))
    = mat(max(0,1); max(0,0))
    = mat(1;0).$
])
#pause
#place(bottom + center, dy: -2pt,
  result[Rows of $bold(W)_1$ are neurons: each row computes one pre-activation, then ReLU acts elementwise.])

== Combine the two features, then compute probability and loss #D

#two(r: (0.78fr, 1.22fr),
  [#align(center, diagram(spacing: (29mm, 18mm), node-stroke: 0.9pt, {
    edge((0, -0.72), (1, 0), text(size: 14pt, weight: 700, fill: BLUE)[$+2$],
      "-|>", stroke: 1.3pt + BLUE, label-side: left)
    edge((0, 0.72), (1, 0), text(size: 14pt, weight: 700, fill: GREEN)[$-4$],
      "-|>", stroke: 1.3pt + GREEN, label-side: right)
    edge((1, 0), (2, 0), text(size: 11.5pt, weight: 700, fill: ACC)[$p=sigma(z)$],
      "-|>", stroke: 1.0pt + ACC, label-side: left)
    node((0, -0.72), align(center, text(size: 13.5pt, fill: white)[$h_1$ \
      $=1$]), radius: 10mm, fill: BLUE, stroke: none)
    node((0, 0.72), align(center, text(size: 13.5pt, fill: white)[$h_2$ \
      $=0$]), radius: 10mm, fill: GREEN, stroke: none)
    node((1, 0), text(size: 12pt, fill: white)[$z=1$], radius: 9mm, fill: ACC, stroke: none)
    node((2, 0), text(fill: white)[$p$], radius: 7mm, fill: ACC, stroke: none)
  }))],
  [#set text(size: 16.5pt)
   *Output logit* \
   $ z = mat(2,-4) mat(1;0)-1 $
   $ quad =2(1)-4(0)-1=1. $
   #pause
   *Probability* \
   $ p=sigma(1)=frac(1,1+exp(-1)) approx 0.731. $
   #pause
   *Binary cross-entropy for $y=1$* \
   $ cal(L)=-log(0.731) approx 0.313. $],
)
#pause
#place(bottom + center, dy: -2pt,
  result[This is the same output node shown in the XOR network: weights $+2,-4$ and bias $-1$. Sigmoid only turns $z$ into $p$.])

== Each ReLU is flat on one side of a line and rises on the other #V

#two(
  [#align(center, text(size: 17pt, weight: 700, fill: BLUE)[
     $h_1(bold(x))="ReLU"(x_1+x_2)$
   ])
   #v(2pt)
   #_feature-field(
     (x1, x2) => _relu(x1 + x2),
     BLUE, [$3$], [active above $x_1+x_2=0$],
     ((0.5, 0.5, [$h_1=1$], ACC), (-0.2, 0.2, [$h_1=0$], INK)),
   )
  ],
  [#pause
   #align(center, text(size: 17pt, weight: 700, fill: GREEN)[
     $h_2(bold(x))="ReLU"(x_1+x_2-1)$
   ])
   #v(2pt)
   #_feature-field(
     (x1, x2) => _relu(x1 + x2 - 1),
     GREEN, [$2$], [active above $x_1+x_2=1$],
     ((0.75, 0.75, [$h_2=0.5$], ACC), (0.6, 0.4, [$h_2=0$], INK)),
   )
  ],
)
#pause
#place(bottom + center, dy: -2pt,
  result[The grid is input space; each colour bar reports a hidden-unit activation value—not a probability.])

== Compute each XOR point's new hidden-space coordinates #V

#align(center, text(size: 16pt)[
  Use the *same map* for every point:
  $T(x_1,x_2)=(h_1,h_2)=("ReLU"(x_1+x_2), "ReLU"(x_1+x_2-1)).$
])
#v(3pt)
#align(center, text(size: 13.2pt, weight: 650)[
  #text(fill: TEAL)[A: $(0,0) arrow.r (0,0)$] #h(18pt)
  #text(fill: ACC)[B: $(1,0) arrow.r (1,0)$] #h(18pt)
  #text(fill: ACC)[C: $(0,1) arrow.r (1,0)$] #h(18pt)
  #text(fill: TEAL)[D: $(1,1) arrow.r (2,1)$]
])
#v(2pt)
#grid(
  columns: (1fr, 25mm, 1.1fr), column-gutter: 6pt, align: horizon,
  [#align(center, [
    #text(size: 15pt, weight: 700, fill: INK)[INPUT SPACE $(x_1,x_2)$]
    #v(3pt)
    #diagram(spacing: (30mm, 20mm), node-stroke: none, {
      edge((-0.12, 0), (1.32, 0), "-|>", stroke: 0.9pt + INK)
      edge((0, 0.12), (0, -1.32), "-|>", stroke: 0.9pt + INK)
      node((0, 0), none, radius: 3.4mm, fill: TEAL, stroke: 0.8pt + white)
      node((1, 0), none, radius: 3.4mm, fill: ACC, stroke: 0.8pt + white)
      node((0, -1), none, radius: 3.4mm, fill: ACC, stroke: 0.8pt + white)
      node((1, -1), none, radius: 3.4mm, fill: TEAL, stroke: 0.8pt + white)
      node((-0.02, 0.23), text(size: 10pt, weight: 650, fill: TEAL)[A: $(0,0)$], stroke: none)
      node((1, 0.23), text(size: 10pt, weight: 650, fill: ACC)[B: $(1,0)$], stroke: none)
      node((-0.02, -1.20), text(size: 10pt, weight: 650, fill: ACC)[C: $(0,1)$], stroke: none)
      node((1, -1.20), text(size: 10pt, weight: 650, fill: TEAL)[D: $(1,1)$], stroke: none)
      node((1.34, 0.10), text(size: 12pt)[$x_1$], stroke: none)
      node((-0.10, -1.34), text(size: 12pt)[$x_2$], stroke: none)
    })
  ])],
  [#pause
   #align(center, [
     #text(size: 10.5pt, weight: 650, fill: MUTED)[APPLY $T$]
     #v(3pt)
     #text(size: 25pt, fill: MUTED)[$arrow.r$]
     #v(3pt)
     #text(size: 11pt, fill: MUTED)[B and C merge]
   ])],
  [#pause
   #align(center, [
    #text(size: 15pt, weight: 700, fill: INK)[HIDDEN SPACE $(h_1,h_2)$]
    #v(3pt)
    #diagram(spacing: (22mm, 20mm), node-stroke: none, {
      edge((-0.12, 0), (2.42, 0), "-|>", stroke: 0.9pt + INK)
      edge((0, 0.12), (0, -1.32), "-|>", stroke: 0.9pt + INK)
      edge((0.5, 0), (2.40, -0.95), stroke: 1.3pt + RED)
      node((0, 0), none, radius: 3.4mm, fill: TEAL, stroke: 0.8pt + white)
      node((1, 0), text(size: 9pt, weight: 700, fill: white)[B,C], radius: 4.2mm, fill: ACC, stroke: 0.8pt + white)
      node((2, -1), none, radius: 3.4mm, fill: TEAL, stroke: 0.8pt + white)
      node((0, 0.23), text(size: 10pt, weight: 650, fill: TEAL)[A: $(0,0)$], stroke: none)
      node((1, 0.23), text(size: 10pt, weight: 650, fill: ACC)[B,C: $(1,0)$], stroke: none)
      node((2, -1.20), text(size: 10pt, weight: 650, fill: TEAL)[D: $(2,1)$], stroke: none)
      node((1.28, -0.78), text(size: 10pt, weight: 650, fill: RED)[$z=0$], stroke: none)
      node((2.44, 0.10), text(size: 12pt)[$h_1$], stroke: none)
      node((-0.10, -1.34), text(size: 12pt)[$h_2$], stroke: none)
    })
  ])],
)
#pause
#place(bottom + center, dy: -2pt,
  result[#text(size: 17.5pt)[B and C both become $(1,0)$. Since $z=2h_1-4h_2-1$, the boundary $z=0 arrow.l.r h_1-2h_2=0.5$ separates them from A and D.]])

== Node view: binary uses one output; $K$-class uses $K$ #V

#two(
  [#align(center, [
    #text(size: 18pt, weight: 700, fill: ACC)[BINARY]
    #v(6pt)
    #mlp-diagram((3, 4, 1), labels: ([$bold(x)$], [$bold(h)$], [$z$]))
  ])],
  [#pause
   #align(center, [
    #text(size: 18pt, weight: 700, fill: ACC)[THREE CLASSES]
    #v(6pt)
    #mlp-diagram((3, 4, 3), labels: ([$bold(x)$], [$bold(h)$], [$z_1,z_2,z_3$]))
  ])],
)
#pause
#place(bottom + center, dy: -2pt,
  result[The hidden representation can stay the same; only the number and interpretation of output nodes changes.])

== Node view: depth adds hidden columns #V

#mlp-diagram(
  (3, 4, 5, 4, 2),
  labels: (
    [input $bold(x)$],
    [hidden $bold(h)^((1))$],
    [hidden $bold(h)^((2))$],
    [hidden $bold(h)^((3))$],
    [output $bold(z)$],
  ),
)
#pause
#v(8pt)
#neat-caption([Every node connects only to the next column; information reaches the output through successive learned transformations.])
#pause
#place(bottom + center, dy: -2pt,
  result[Depth is visible as the number of hidden columns, not the number of nodes within one column.])

// ═══════════════════════════ PART VII — Universal approximation ═══════════════════════════
= Universal approximation

== Universal approximation asks one precise question #Q

#align(center, grid(
  columns: (1fr, auto, 1fr, auto, 1fr), column-gutter: 8pt, align: horizon,
  align(center, [#text(size: 12pt, weight: 700, fill: INK)[TARGET] \
    #text(size: 19pt)[continuous $f$]]),
  neat-arrow(),
  align(center, [#text(size: 12pt, weight: 700, fill: TEAL)[REGION] \
    #text(size: 19pt)[closed, bounded $D$]]),
  neat-arrow(),
  align(center, [#text(size: 12pt, weight: 700, fill: ACC)[TOLERANCE] \
    #text(size: 19pt)[$epsilon>0$]]),
))
#pause
#v(13pt)
#align(center, text(size: 21pt)[
  Can some finite network $g$ make $abs(f(x)-g(x))<epsilon$ for *every* $x in D$?
])
#pause
#v(13pt)
#align(center, text(size: 19pt, weight: 650)[
  With a suitable nonlinear activation, the theorem says: *yes—such a finite one-hidden-layer network exists.*
])
#pause
#place(bottom + center, dy: -2pt,
  result[This is an existence claim about representational capacity; it does not promise that training will find the network.])

== A ReLU contributes one hinge #V

#two(r: (0.9fr, 1.1fr),
  [
    $ phi(z) = max(0, z) $
    #v(7pt)
    #pause
    - $z<0$: output is zero;
    - $z>0$: output grows linearly;
    - at $z=0$: the slope changes—a *hinge*.
  ],
  [#pause
   #align(center, lines(
     fn: z => calc.max(0, z), domain: (-2, 2), samples: 80,
     markers: false, colors: (ACC,), points: ((0, 0, [$0$]),),
     x-label: [$z$], y-label: [$"ReLU"(z)$], size: (70mm, 44mm),
   ))],
)
#pause
#place(bottom + center, dy: -2pt,
  result[A hidden layer combines many shifted hinges into a piecewise-linear function.])

== One input becomes three symbolic hinge features #V

#align(center, text(size: 18pt)[
  Every unit first computes $u_i=w_i x+b_i$, then applies $h_i="ReLU"(u_i)$.
])
#v(6pt)
#align(center, diagram(spacing: (31mm, 14.5mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let inp = (0, 0)
  let us = ((2, -1), (2, 0), (2, 1))
  let hs = ((4, -1), (4, 0), (4, 1))
  let cols = (BLUE, RED, GREEN)
  let params = ([$(w_1,b_1)=(1,0)$], [$(w_2,b_2)=(1,-1)$], [$(w_3,b_3)=(1,-2)$])
  let param-pos = ((1.02, -0.70), (1.02, -0.18), (1.02, 0.70))
  let ulabs = ($u_1$, $u_2$, $u_3$)
  let hlabs = ($h_1$, $h_2$, $h_3$)

  for i in range(3) {
    edge(inp, us.at(i), "-|>", stroke: 1.0pt + cols.at(i))
    edge(us.at(i), hs.at(i), text(size: 10.5pt, weight: 650, fill: cols.at(i))[ReLU],
      "-|>", stroke: 1.0pt + cols.at(i))
  }
  node(inp, text(fill: white)[$x$], radius: 8mm, fill: INK, stroke: none)
  for i in range(3) {
    node(param-pos.at(i), block(fill: white, inset: 2pt,
      text(size: 9.8pt, weight: 650, fill: cols.at(i), params.at(i))), stroke: none)
    node(us.at(i), ulabs.at(i), radius: 7.5mm, stroke: 1.2pt + cols.at(i))
    node(hs.at(i), text(fill: white, hlabs.at(i)), radius: 7.5mm, fill: cols.at(i), stroke: none)
  }
  node((0, -1.72), text(size: 11pt, weight: 650, fill: MUTED)[ONE INPUT], stroke: none)
  node((2, -1.72), text(size: 11pt, weight: 650, fill: MUTED)[AFFINE SUMS], stroke: none)
  node((4, -1.72), text(size: 11pt, weight: 650, fill: MUTED)[RELU FEATURES], stroke: none)
}))
#place(bottom + center, dy: -2pt,
  result[The same $x$ creates $bold(h)(x)=("ReLU"(x),"ReLU"(x-1),"ReLU"(x-2))$; the hinges are at $0,1,2$.])

== Output weights combine the three symbolic hinges #V

#align(center, diagram(spacing: (31mm, 15mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let inp = (0, 0)
  let hs = ((2, -1), (2, 0), (2, 1))
  let cols = (BLUE, RED, GREEN)
  let labs = ($h_1$, $h_2$, $h_3$)
  let out = (4, 0)

  for i in range(3) {
    edge(inp, hs.at(i), "-|>", stroke: 0.85pt + cols.at(i))
  }
  edge(hs.at(0), out, block(fill: white, inset: 2pt,
    text(size: 11.5pt, weight: 700, fill: BLUE)[$g_1(x)=(+1)h_1(x)$]),
    "-|>", stroke: 1.2pt + BLUE, label-side: left)
  edge(hs.at(1), out, block(fill: white, inset: 2pt,
    text(size: 11.5pt, weight: 700, fill: RED)[$g_2(x)=(-2)h_2(x)$]),
    "-|>", stroke: 1.2pt + RED, label-side: left)
  edge(hs.at(2), out, block(fill: white, inset: 2pt,
    text(size: 11.5pt, weight: 700, fill: GREEN)[$g_3(x)=(+1)h_3(x)$]),
    "-|>", stroke: 1.2pt + GREEN, label-side: left)
  node(inp, $x$, radius: 7mm, stroke: 1pt + INK)
  for i in range(3) {
    node(hs.at(i), labs.at(i), radius: 9.5mm, stroke: 1.2pt + cols.at(i))
  }
  node(out, $f(x)$, radius: 8mm, stroke: 1.2pt + ACC)
  node((0, -1.65), text(size: 11pt, weight: 650, fill: MUTED)[INPUT], stroke: none)
  node((1, -1.65), text(size: 10pt, weight: 650, fill: MUTED)[all input weights $=+1$], stroke: none)
  node((2, -1.65), text(size: 11pt, weight: 650, fill: MUTED)[SHIFTED RELUS], stroke: none)
  node((4, -1.65), text(size: 11pt, weight: 650, fill: MUTED)[WEIGHTED SUM], stroke: none)
}))
#place(bottom + center, dy: -2pt,
  result[The output adds $f(x)=g_1(x)+g_2(x)+g_3(x)$—the theorem's “sum of hidden units” is literal.])

== Output weights choose each hinge's slope change #V

#align(center, text(size: 18pt)[
  Start with $h(x)="ReLU"(x-t)$; its output contribution is $g(x)=v h(x)$.
])
#v(4pt)
#two(
  [#align(center, [
    #text(size: 18pt, weight: 650, fill: BLUE)[$v=+1 quad arrow.r quad g(x)=+h(x)$]
    #v(4pt)
    #lines(
      fn: (x => calc.max(0, x - 1), x => -4, x => 2),
      domain: (-1, 3), samples: 120,
      markers: (false, false, false),
      colors: (BLUE, BLUE.transparentize(100%), BLUE.transparentize(100%)),
      points: ((1, 0, [hinge $t$]),),
      x-label: [$x$], y-label: [$g(x)$], size: (88mm, 47mm),
    )
    #v(2pt)
    #text(size: 15pt, weight: 650, fill: BLUE)[slope change $Delta s=+1$]
  ])],
  [#align(center, [
    #text(size: 18pt, weight: 650, fill: RED)[$v=-2 quad arrow.r quad g(x)=-2h(x)$]
    #v(4pt)
    #lines(
      fn: (x => -2 * calc.max(0, x - 1), x => -4, x => 2),
      domain: (-1, 3), samples: 120,
      markers: (false, false, false),
      colors: (RED, RED.transparentize(100%), RED.transparentize(100%)),
      points: ((1, 0, [same hinge $t$]),),
      x-label: [$x$], y-label: [$g(x)$], size: (88mm, 47mm),
    )
    #v(2pt)
    #text(size: 15pt, weight: 650, fill: RED)[slope change $Delta s=-2$]
  ])],
)
#place(bottom + center, dy: -2pt,
  result[The hinge stays at $t$; the sign of $v$ chooses the bend, and $abs(v)$ sets its strength.])

#let _tent-interval-plane(stage) = align(center, diagram(
  spacing: (36mm, 27mm), node-stroke: none, node-fill: none,
  {
    // Fixed axes and hinge locations anchor all four animation states.
    edge((-0.65, 0), (3.15, 0), "->", stroke: 0.8pt + INK)
    edge((-0.65, 0.15), (-0.65, -1.48), "->", stroke: 0.8pt + INK)
    node((3.27, 0.04), $x$, stroke: none)
    node((-0.73, -1.58), $f(x)$, stroke: none)
    node((-0.86, -1), $1$, stroke: none)

    for x in (0, 1, 2) {
      edge((x, -0.055), (x, 0.055), stroke: 0.75pt + INK)
      node((x, 0.22), [$#x$], stroke: none)
    }

    if stage >= 1 {
      edge((-0.62, 0), (0, 0), stroke: 3pt + MUTED)
      node((-0.31, -0.28), text(size: 12pt, weight: 650, fill: MUTED)[$0+0+0=0$], stroke: none)
      node((-0.31, 0.47), text(size: 12pt, weight: 650, fill: MUTED)[$x<0$], stroke: none)
    }
    if stage >= 2 {
      edge((0, 0), (1, -1), stroke: 3pt + BLUE)
      node((0.47, -0.67), text(size: 12pt, weight: 650, fill: BLUE)[$h_1=x arrow.r f=x$], stroke: none)
      node((0.5, 0.47), text(size: 12pt, weight: 650, fill: BLUE)[$0<=x<1$], stroke: none)
    }
    if stage >= 3 {
      edge((1, -1), (2, 0), stroke: 3pt + RED)
      node((1.52, -0.67), text(size: 12pt, weight: 650, fill: RED)[$h_1-2h_2=2-x$], stroke: none)
      node((1.5, 0.47), text(size: 12pt, weight: 650, fill: RED)[$1<=x<2$], stroke: none)
    }
    if stage >= 4 {
      edge((2, 0), (3.28, 0), stroke: 3pt + GREEN)
      node((2.62, -0.28), align(center, text(size: 12pt, weight: 650, fill: GREEN)[
        $h_1-2h_2+h_3$ \
        $=x-2(x-1)+(x-2)=0$
      ]), stroke: none)
      node((2.62, 0.47), text(size: 12pt, weight: 650, fill: GREEN)[$x>=2$], stroke: none)
    }

    // Knots remain visible as each new piece is attached.
    if stage >= 2 { node((0, 0), none, radius: 1.7mm, fill: ACC, stroke: none) }
    if stage >= 3 { node((1, -1), none, radius: 1.7mm, fill: ACC, stroke: none) }
    if stage >= 4 { node((2, 0), none, radius: 1.7mm, fill: ACC, stroke: none) }
  },
))

== Three hinges assemble a tent interval by interval #D

#align(center, text(size: 18.5pt)[
  $f(x) = h_1 - 2h_2 + h_3 = "ReLU"(x) - 2"ReLU"(x-1) + "ReLU"(x-2).$
])
#v(3pt)
#alternatives(
  [#_tent-interval-plane(1)
   #v(2pt)
   #align(center, text(size: 16.5pt, fill: MUTED)[Region 1: every hidden unit is off, so $f(x)=0$.])],
  [#_tent-interval-plane(2)
   #v(2pt)
   #align(center, text(size: 16.5pt, fill: BLUE)[Region 2: $h_1=x$ switches on, so the output rises with slope $+1$.])],
  [#_tent-interval-plane(3)
   #v(2pt)
   #align(center, text(size: 16.5pt, fill: RED)[Region 3: $-2h_2$ switches on, changing the total slope from $+1$ to $-1$.])],
  [#_tent-interval-plane(4)
   #v(2pt)
   #align(center, text(size: 16.5pt, fill: GREEN)[Region 4: $h_3$ switches on and cancels the remaining slope, so $f(x)=0$.])],
)
#uncover("4-")[
  #place(bottom + center, dy: -2pt,
    result[Reading left to right: flat $arrow.r$ rise $arrow.r$ fall $arrow.r$ flat—the weighted sum is a tent.])
]

#let _sin-pwl-breaks(x, m) = _sin-pwl(x, knots: m + 2)
#let _breakpoint-lines(m) = range(1, m + 1).map(i => (
  -3.14 + 6.28 * i / (m + 1), none, MUTED.lighten(48%),
))
#let _constructed-interpolant-panel(m, error) = [
  #align(center, [
    #text(size: 18pt, weight: 700, fill: ACC)[#m hinges]
    #v(1pt)
    #text(size: 14pt, weight: 650, fill: MUTED)[$epsilon_m approx #str(error)$]
    #v(2pt)
    #lines(
      fn: (x => calc.sin(x), x => _sin-pwl-breaks(x, m)),
      domain: (-3.14, 3.14), samples: 240,
      colors: (INK, ACC), dashes: ("solid", "dashed"), markers: (false, false),
      vlines: _breakpoint-lines(m),
      x-label: [$x$], y-label: [output], size: (70mm, 43mm),
    )
  ])
]

== Learned ReLU units are the theorem's building blocks #D

For a one-dimensional input, a trained hidden unit has the form
$ h_j(x)="ReLU"(w_j x+b_j), quad g(x)=c+sum_j v_j h_j(x). $
#pause
#v(7pt)
#align(center, {
  set text(size: 14pt)
  table(
    columns: (18mm, 27mm, 27mm, 27mm, 30mm, 68mm),
    stroke: 0.5pt + MUTED, inset: (x: 6pt, y: 5pt), align: center,
    table.header([$j$], [hinge $t_j$], [$w_j$], [$b_j$], [output $v_j$], [signed contribution $g_j(x)$]),
    [#text(fill: BLUE)[$1$]], [$0$], [$1$], [$0$], [#text(fill: BLUE)[$+1$]], [#text(fill: BLUE)[$+"ReLU"(x)$]],
    [#text(fill: RED)[$2$]], [$1$], [$1$], [$-1$], [#text(fill: RED)[$-2$]], [#text(fill: RED)[$-2"ReLU"(x-1)$]],
    [#text(fill: GREEN)[$3$]], [$2$], [$1$], [$-2$], [#text(fill: GREEN)[$+1$]], [#text(fill: GREEN)[$+"ReLU"(x-2)$]],
  )
})
#pause
#place(bottom + center, dy: -2pt,
  result[Training learns hinge locations and signed contributions; universal approximation says enough such units can make the remaining gap arbitrarily small.])

== In many dimensions, a hinge becomes a hyperplane #V

#two(r: (0.86fr, 1.14fr),
  [A unit reads the whole vector:
   $h(bold(x))="ReLU"(bold(w)^top bold(x)+b).$
   #pause
   The hinge is where the pre-activation is zero:
   $bold(w)^top bold(x)+b=0.$
   #v(8pt)
   #notebox[In 1-D this is a point. In 2-D it is a line. In 3-D it is a plane. In $d$ dimensions it is a hyperplane.]],
  [#pause
   #align(center, lines(
     fn: x => 2 - 2 * x, domain: (-1, 2), samples: 80, markers: false, colors: (ACC,),
     points: ((0,0, [off: $h=0$]), (0,2, [hinge]), (2,-1, [on: $h=0.5$])),
     x-label: [$x_1$], y-label: [$x_2$], title: [$x_1+0.5x_2-1=0$], size: (103mm, 58mm),
   ))],
)
#pause
#place(bottom + center, dy: -2pt,
  result[The one-dimensional breakpoint construction generalizes to learned, oriented folds of a multidimensional input space.])

== Beyond 1-D: approximate class scores, then classify #V

#grid(
  columns: (90mm, 1fr), column-gutter: 12pt, align: horizon,
  [#align(center, scale(58%)[
    #mlp-diagram((3, 5, 3), labels: (
      [$bold(x) in RR^d$],
      [#text(size: 13.5pt)[ReLU features]],
      [#text(size: 13.5pt)[$K$ logits]],
    ))
  ])],
  [#text(size: 15.5pt)[
    *1 · Vector input $arrow.r$ learned folds* \
    $h_j(bold(x))="ReLU"(bold(w)_j^top bold(x)+b_j).$
    #v(5pt)
    *2 · Features $arrow.r$ continuous scores* \
    $z_k(bold(x))=c_k+sum_j v_(k j)h_j(bold(x)).$
    #v(5pt)
    *3 · Scores $arrow.r$ classification* \
    $bold(p)="softmax"(bold(z)), quad hat(y)=arg max_k z_k.$ \
    A class changes where two top scores tie.
  ]],
)
#pause
#place(bottom + center, dy: -2pt,
  result[#text(size: 18pt)[On compact $D subset RR^d$, approximate continuous logits; $arg max$ classifies wherever the winning score has a margin.]])

== One hidden unit has three controls #D

#align(center, diagram(
  spacing: (29mm, 15mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    edge((0, 0), (2.6, 0), text(size: 13pt, weight: 650, fill: BLUE)[$t_j arrow.r b_j=-w_j t_j$],
      "-|>", stroke: 1pt + BLUE, label-side: left)
    edge((2.6, 0), (5.35, 0), text(size: 13pt, weight: 650, fill: ACC)[$v_j$],
      "-|>", stroke: 1pt + ACC, label-side: left)
    node((0, 0), $x$, radius: 7mm, stroke: 1pt + INK)
    node((2.6, 0), [$h_j="ReLU"(w_j(x-t_j))$], shape: fletcher.shapes.rect,
      inset: 9pt, fill: TEAL.lighten(92%), stroke: 1.1pt + TEAL)
    node((5.35, 0), [$g_j(x)=v_j h_j(x)$], shape: fletcher.shapes.rect,
      inset: 9pt, fill: ACC.lighten(94%), stroke: 1.1pt + ACC)
    node((0, 0.9), text(size: 11pt, weight: 650, fill: MUTED)[INPUT], stroke: none)
    node((2.6, 0.9), text(size: 11pt, weight: 650, fill: MUTED)[HIDDEN FEATURE], stroke: none)
    node((5.35, 0.9), text(size: 11pt, weight: 650, fill: MUTED)[SIGNED COMPONENT], stroke: none)
  },
))
#v(14pt)
#align(center, grid(
  columns: (68mm, 68mm, 68mm), gutter: 10pt,
  [#align(center)[#text(size: 13pt, weight: 700, fill: BLUE)[HINGE $t_j$] \
    #text(size: 16.5pt)[moves the corner]]],
  [#align(center)[#text(size: 13pt, weight: 700, fill: TEAL)[HIDDEN WEIGHT $w_j$] \
    #text(size: 16.5pt)[sets orientation and scale]]],
  [#align(center)[#text(size: 13pt, weight: 700, fill: ACC)[OUTPUT WEIGHT $v_j$] \
    #text(size: 16.5pt)[sets sign and slope change]]],
))
#place(bottom + center, dy: -2pt,
  result[The controls are the learned parameters; toggling components reveals how their signed sum creates the curve.])

== Interactive: build functions and a binary classifier #I

#interbox(link-to: "https://nipunbatra.github.io/dl-teaching/interactives/relu-function-builder.html")[
  Follow the same sequence as the slides: *one hinge $arrow.r$ tent $arrow.r$ smooth target $arrow.r$ binary classifier*.
]
#v(13pt)
#align(center, text(size: 18pt)[
  Select one unit, then change $t_j$ (hinge), $w_j$ (hidden slope/orientation), or $v_j$ (signed output contribution).
])
#pause
#v(10pt)
#align(center, text(size: 17pt)[
  In the classifier preset, the weighted sum is the logit $z(x)$; the decision changes where $z=0$, and $p=sigma(z)$ reports confidence.
])
#pause
#place(bottom + center, dy: -2pt,
  result[Inspect individual $v_j h_j(x)$ components first, their sum second, and the class decision last.])

== First fix the target and the input region #V

#two(r: (1.12fr, 0.88fr),
  align(center, lines(
    fn: x => calc.sin(x), domain: (-3.14, 3.14), samples: 220,
    markers: false, colors: (INK,),
    points: ((-3.14, 0, [$-pi$]), (3.14, 0, [$pi$])),
    x-label: [$x$], y-label: [$f(x)$], title: [a continuous target],
    size: (94mm, 57mm),
  )),
  [
    *Continuous* means that the target has no jumps.
    #pause
    #v(10pt)
    In one dimension, a *compact domain* can be a closed, bounded interval such as
    $ D = [-pi, pi]. $
    #pause
    #v(10pt)
    We ask the network to match $f$ *on $D$*; the theorem makes no claim outside that chosen region.
  ],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Approximation begins with one target function on one specified input region.])

== At one input, error is one vertical gap #V

#align(center, text(size: 20pt, weight: 650)[
  $e(x_0)=abs(f(x_0)-g(x_0))$
])
#v(3pt)
#align(center, lines(
  fn: (x => calc.sin(x), x => _sin-pwl(x, knots: 7)),
  domain: (-3.14, 3.14), samples: 220,
  labels: ([$f(x)$ target], [$g(x)$ constructed]),
  colors: (INK, ACC), dashes: ("solid", "dashed"), markers: (false, false),
  legend: "tl", vlines: ((1.57, [], MUTED),),
  points: ((1.57, calc.sin(1.57), []),
    (1.57, _sin-pwl(1.57, knots: 7), [])),
  x-label: [$x$], y-label: [output], title: [vertical slice at $x_0=1.57$],
  size: (126mm, 59mm),
))
#place(bottom + center, dy: -2pt,
  result[Pointwise error is the vertical distance between the target and its constructed approximation at the same input.])

== Calculate the gap at $x_0=1.57$ #D

#align(center, text(size: 19pt)[
  $e(x_0)=abs(f(x_0)-g(x_0))$
])
#v(10pt)
#align(center, diagram(
  spacing: (22mm, 18mm), node-stroke: 0.95pt + INK, node-fill: white,
  {
    edge((0, -0.65), (3, 0), "-|>", stroke: 0.9pt + INK)
    edge((0, 0.65), (3, 0), "-|>", stroke: 0.9pt + ACC)
    edge((3, 0), (5.7, 0), "-|>", stroke: 1.05pt + TEAL)

    node((0, -0.65), align(center)[#text(size: 11pt, weight: 700, fill: INK)[TARGET] \
      $f(1.57) approx 1.000$], shape: fletcher.shapes.rect,
      inset: 9pt, fill: INK.lighten(96%), stroke: 1pt + INK)
    node((0, 0.65), align(center)[#text(size: 11pt, weight: 700, fill: ACC)[CONSTRUCTED] \
      $g(1.57) approx 0.866$], shape: fletcher.shapes.rect,
      inset: 9pt, fill: ACC.lighten(94%), stroke: 1pt + ACC)
    node((3, 0), align(center)[#text(size: 11pt, weight: 700, fill: TEAL)[ABSOLUTE DIFFERENCE] \
      $abs(1.000-0.866)$], shape: fletcher.shapes.rect,
      inset: 10pt, fill: TEAL.lighten(94%), stroke: 1pt + TEAL)
    node((5.7, 0), align(center)[#text(size: 11pt, weight: 700, fill: RED)[POINTWISE ERROR] \
      $e(1.57) approx 0.134$], shape: fletcher.shapes.rect,
      inset: 10pt, fill: RED.lighten(94%), stroke: 1.1pt + RED)
  },
))
#place(bottom + center, dy: -2pt,
  result[One point is not enough: universal approximation asks for the largest gap over the whole region.])

== “sup” means the worst gap over the whole region #V

#two(r: (0.9fr, 1.1fr),
  [#set text(size: 16pt)
   #text(size: 19pt)[$sup_(x in D) abs(f(x)-g(x))$]
   #v(5pt)
   *sup* means *supremum*: the smallest number at least as large as every pointwise error.
   #pause
   #v(5pt)
   Here the error is continuous and $D=[-pi,pi]$ is closed and bounded, so a highest error exists. In this plot, read *sup* as *maximum*.
   #v(6pt)
   *$sup < epsilon$*: no input in $D$ crosses the tolerance.],
  [#pause
   #align(center, lines(
     fn: (x => calc.abs(calc.sin(x) - _sin-pwl(x, knots: 13)), x => 0.06),
     domain: (-3.14, 3.14), samples: 220,
     colors: (ACC, ACC.transparentize(100%)),
     dashes: ("solid", "solid"), markers: (false, false),
     hlines: ((0.05, [$epsilon=0.05$], RED),),
     points: ((1.31, 0.0329, [worst gap $approx 0.033$]),),
     x-label: [$x$], y-label: [$abs(f(x)-g(x))$],
     title: [error only], size: (88mm, 55mm),
   ))],
)
#pause
#place(bottom + center, dy: -2pt,
  result[The entire orange error curve stays below $epsilon=0.05$, so the uniform guarantee is satisfied.])

== More hinges reduce the largest approximation error #V

#align(center, [
  #text(size: 14pt, weight: 750, fill: ACC)[CONSTRUCTED INTERPOLANTS · FIXED KNOTS · NOT TRAINED]
  #h(10pt)
  #text(size: 14pt, fill: INK)[solid: target $sin x$]
  #h(10pt)
  #text(size: 14pt, fill: ACC)[dashed: $g_m(x)$]
])
#v(3pt)
#align(center, text(size: 15pt)[Each $g_m$ matches $sin x$ at fixed knots and is linear between adjacent knots.])
#v(5pt)
#grid(
  columns: (1fr, 1fr, 1fr), column-gutter: 8pt, align: top,
  _constructed-interpolant-panel(2, 0.437),
  _constructed-interpolant-panel(5, 0.134),
  _constructed-interpolant-panel(15, 0.019),
)
#place(bottom + center, dy: -2pt,
  result[More hinges shorten each linear segment, so the worst gap falls from $0.437$ to $0.019$.])

== Read the universal-approximation claim in order #D

#align(center, grid(
  columns: (48mm, 8mm, 48mm, 8mm, 58mm, 8mm, 58mm), align: horizon,
  neat-card([1 · fix target], [$f$ and $D$], color: INK, width: 48mm),
  neat-arrow(),
  neat-card([2 · demand accuracy], [choose any $epsilon>0$], color: ACC, width: 48mm),
  neat-arrow(),
  neat-card([3 · choose network], [some finite MLP $g$ may depend on $epsilon$], color: TEAL, width: 58mm),
  neat-arrow(),
  neat-card([4 · verify], [$sup_(x in D)abs(f(x)-g(x))<epsilon$], color: GREEN, width: 58mm),
))
#pause
#v(12pt)
#align(center, text(size: 21pt)[
  $forall epsilon>0, quad exists " finite MLP " g quad "such that" quad sup_(x in D)abs(f(x)-g(x))<epsilon.$
])
#pause
#two(
  notebox[*For every* $epsilon$: we may request any positive tolerance.],
  notebox[*There exists* $g$: the network may change when the tolerance changes.],
)
#pause
#place(bottom + center, dy: -2pt,
  result[This is a statement about representational capacity—not yet about how to find the network.])

== Existence does not guarantee learnability

Universal approximation does *not* promise:
#pause
- that training will *find* that function;
- that *finite data* is enough;
#pause
- that a *small* network suffices;
- that it will *generalize*.
#pause
#result[expressivity $eq.not$ trainability $eq.not$ generalization]

// ═══════════════════════════ PART VIII — Depth versus width ═══════════════════════════
= Depth versus width

== Node view: width adds rows; depth adds columns #V

#two(
  [#align(center, [
    #text(size: 19pt, weight: 700, fill: ACC)[WIDER]
    #v(5pt)
    #mlp-diagram((2, 6, 1), labels: ([input], [$6$ parallel units], [output]))
  ])],
  [#pause
   #align(center, [
    #text(size: 19pt, weight: 700, fill: TEAL)[DEEPER]
    #v(5pt)
    #mlp-diagram((2, 3, 3, 1), labels: ([input], [stage $1$], [stage $2$], [output]))
  ])],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Count nodes vertically to read width; count hidden columns horizontally to read depth.])

== Layer 1 creates a hinge at $x=-0.5$ #V

#two(r: (0.9fr, 1.1fr),
  [#align(center, [
    #diagram(spacing: (15mm, 16mm), node-stroke: 1pt + INK, node-fill: white, {
      let inp = (0, 0)
      let bias = (0, 1.15)
      let pre = (2.35, 0)
      let out = (4.55, 0)

      edge(inp, pre, text(size: 13pt, weight: 700, fill: TEAL)[$w_1=2$],
        "-|>", stroke: 1.1pt + TEAL, label-side: left)
      edge(bias, pre, text(size: 13pt, weight: 700, fill: RED)[$b_1=+1$],
        "-|>", stroke: 1.0pt + RED, label-side: right)
      edge(pre, out, text(size: 12.5pt, weight: 700, fill: ACC)[ReLU],
        "-|>", stroke: 1.15pt + ACC, label-side: left)

      node(inp, $x$, radius: 8.5mm, stroke: 1.2pt + INK)
      node(bias, $1$, radius: 7mm, stroke: 1.1pt + RED)
      node(pre, $u_1$, radius: 10mm, fill: TEAL.lighten(92%), stroke: 1.2pt + TEAL)
      node(out, $h_1$, radius: 10mm, fill: ACC.lighten(92%), stroke: 1.2pt + ACC)
    })
    #v(8pt)
    #text(size: 18pt)[$u_1=2x+1$ \
      $h_1=f_1(x)="ReLU"(u_1)$]
    #v(8pt)
    #text(size: 16.5pt)[At $x=0.25$: \
      $u_1=1.5 arrow.r h_1=1.5$]
  ])],
  [#pause
   #align(center, lines(
     fn: x => _relu(2 * x + 1), domain: (-1.5, 1.5), samples: 120,
     labels: ([$f_1(x)$],), colors: (TEAL,), markers: false, legend: "tl",
     vlines: ((-0.5, [hinge], MUTED),),
     points: ((0.25, 1.5, [$(0.25,1.5)$]),),
     x-label: [$x$], y-label: [$h_1$], title: [layer 1 output],
     size: (92mm, 55mm),
   ))],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Layer 1 is flat until $2x+1$ becomes positive; then its learned feature grows linearly.])

== The second layer transforms the learned feature #V

#two(r: (0.9fr, 1.1fr),
  [#align(center, [
    #diagram(spacing: (15mm, 16mm), node-stroke: 1pt + INK, node-fill: white, {
      let inp = (0, 0)
      let bias = (0, 1.15)
      let pre = (2.35, 0)
      let out = (4.55, 0)

      edge(inp, pre, text(size: 13pt, weight: 700, fill: TEAL)[$w_2=-1$],
        "-|>", stroke: 1.1pt + TEAL, label-side: left)
      edge(bias, pre, text(size: 13pt, weight: 700, fill: RED)[$b_2=+2$],
        "-|>", stroke: 1.0pt + RED, label-side: right)
      edge(pre, out, text(size: 12.5pt, weight: 700, fill: ACC)[ReLU],
        "-|>", stroke: 1.15pt + ACC, label-side: left)

      node(inp, $h_1$, radius: 10mm, fill: TEAL.lighten(92%), stroke: 1.2pt + TEAL)
      node(bias, $1$, radius: 7mm, stroke: 1.1pt + RED)
      node(pre, $u_2$, radius: 10mm, fill: ACC.lighten(94%), stroke: 1.2pt + ACC)
      node(out, $h_2$, radius: 10mm, fill: ACC.lighten(90%), stroke: 1.2pt + ACC)
    })
    #v(8pt)
    #text(size: 18pt)[$u_2=2-h_1$ \
      $h_2=f_2(h_1)="ReLU"(u_2)$]
    #v(8pt)
    #text(size: 16.5pt)[For $h_1=1.5$: \
      $u_2=0.5 arrow.r h_2=0.5$]
  ])],
  [#pause
   #align(center, lines(
     fn: h => _relu(2 - h), domain: (0, 4), samples: 120,
     labels: ([$f_2(h_1)$],), colors: (ACC,), markers: false, legend: "tl",
     vlines: ((2, [hinge], MUTED),),
     points: ((1.5, 0.5, [$(1.5,0.5)$]),),
     x-label: [$h_1$], y-label: [$h_2$], title: [layer 2 response],
     size: (92mm, 55mm),
   ))],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Layer 2 decreases as $h_1$ grows, then ReLU clips every negative result to zero.])

== Two layers create a clipped nonlinear ramp #V

#align(center, diagram(spacing: (28mm, 16mm), node-stroke: 1pt + INK, node-fill: white, {
  let inp = (0, 0)
  let mid = (2.7, 0)
  let out = (5.4, 0)

  edge(inp, mid, text(size: 13.5pt, weight: 700, fill: TEAL)[$f_1(x)="ReLU"(2x+1)$],
    "-|>", stroke: 1.15pt + TEAL, label-side: left)
  edge(mid, out, text(size: 13.5pt, weight: 700, fill: ACC)[$f_2(h_1)="ReLU"(2-h_1)$],
    "-|>", stroke: 1.15pt + ACC, label-side: left)

  node(inp, $x$, radius: 9mm, stroke: 1.2pt + INK)
  node(mid, $h_1$, radius: 10.5mm, fill: TEAL.lighten(92%), stroke: 1.2pt + TEAL)
  node(out, $h_2$, radius: 10.5mm, fill: ACC.lighten(90%), stroke: 1.2pt + ACC)
}))
#v(3pt)
#align(center, text(size: 18pt, weight: 650)[
  $h_2(x)=f_2(f_1(x))="ReLU"(2-"ReLU"(2x+1))$
])
#pause
#align(center, lines(
  fn: x => _relu(2 - _relu(2 * x + 1)), domain: (-1.5, 1.5), samples: 150,
  labels: ([$h_2(x)$],), colors: (ACC,), markers: false, legend: "tl",
  vlines: ((-0.5, [$-0.5$], MUTED), (0.5, [$0.5$], MUTED)),
  x-label: [$x$], y-label: [$h_2$], title: [overall map from raw input to layer 2],
  size: (126mm, 40mm),
))
#pause
#place(bottom + center, dy: -2pt,
  result[The first hinge turns on at $x=-0.5$; the second layer creates another bend at $x=0.5$.])

== Three inputs reveal the complete composition #D

#two(r: (1.05fr, 0.95fr),
  [#align(center, lines(
    fn: x => _relu(2 - _relu(2 * x + 1)), domain: (-1.5, 1.5), samples: 150,
    labels: ([$h_2(x)$],), colors: (ACC,), markers: false, legend: "tl",
    vlines: ((-0.5, [], MUTED), (0.5, [], MUTED)),
    points: ((-1, 2, [$(-1,2)$]), (0.25, 0.5, [$(0.25,0.5)$]), (1, 0, [$(1,0)$])),
    x-label: [$x$], y-label: [$h_2$], title: [three regions, three behaviours],
    size: (101mm, 58mm),
  ))],
  [#set text(size: 16.5pt)
   #text(size: 18pt, weight: 700, fill: TEAL)[$x=-1$]
   $u_1=-1 arrow.r h_1=0$ \
   $u_2=2 arrow.r h_2=2$
   #v(7pt)
   #text(size: 18pt, weight: 700, fill: ACC)[$x=0.25$]
   $u_1=1.5 arrow.r h_1=1.5$ \
   $u_2=0.5 arrow.r h_2=0.5$
   #v(7pt)
   #text(size: 18pt, weight: 700, fill: GREEN)[$x=1$]
   $u_1=3 arrow.r h_1=3$ \
   $u_2=-1 arrow.r h_2=0$
  ],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Depth means sequential reuse: layer 2 receives the learned value $h_1$ and transforms it again.])

== Depth builds features one stage at a time #V

#align(center, text(size: 18.5pt)[
  Each layer receives the representation produced immediately before it.
])
#v(5pt)
#align(center, text(size: 19pt, weight: 650)[
  pixels $arrow.r$ short strokes $arrow.r$ closed loops $arrow.r$ digit score
])
#v(8pt)
#align(center, diagram(spacing: (27mm, 14mm), node-stroke: 1pt + INK, node-fill: white, {
  let inp = (0, 0)
  let first = (2, 0)
  let second = (4, 0)
  let out = (6, 0)

  edge(inp, first, text(size: 12.5pt, weight: 700, fill: TEAL)[find simple patterns],
    "-|>", stroke: 1.15pt + TEAL, label-side: left)
  edge(first, second, text(size: 12.5pt, weight: 700, fill: BLUE)[combine them],
    "-|>", stroke: 1.15pt + BLUE, label-side: left)
  edge(second, out, text(size: 12.5pt, weight: 700, fill: ACC)[make a prediction],
    "-|>", stroke: 1.15pt + ACC, label-side: left)

  node(inp, $bold(x)$, radius: 10mm, stroke: 1.2pt + INK)
  node(first, $bold(h)^(1)$, radius: 11mm,
    fill: TEAL.lighten(92%), stroke: 1.2pt + TEAL)
  node(second, $bold(h)^(2)$, radius: 11mm,
    fill: BLUE.lighten(92%), stroke: 1.2pt + BLUE)
  node(out, $bold(z)$, radius: 10mm,
    fill: ACC.lighten(92%), stroke: 1.2pt + ACC)

  node((0, -1.25), text(size: 11.5pt, weight: 700, fill: INK)[INPUT], stroke: none)
  node((2, -1.25), text(size: 11.5pt, weight: 700, fill: TEAL)[LAYER 1], stroke: none)
  node((4, -1.25), text(size: 11.5pt, weight: 700, fill: BLUE)[LAYER 2], stroke: none)
  node((6, -1.25), text(size: 11.5pt, weight: 700, fill: ACC)[OUTPUT], stroke: none)

  node((0, 1.35), align(center)[#text(size: 12pt)[raw \
    measurements]], stroke: none)
  node((2, 1.35), align(center)[#text(size: 12pt)[simple learned \
    features]], stroke: none)
  node((4, 1.35), align(center)[#text(size: 12pt)[features made \
    from features]], stroke: none)
  node((6, 1.35), align(center)[#text(size: 12pt)[task-specific \
    score]], stroke: none)
}))
#pause
#place(bottom + center, dy: -2pt,
  result[Later layers build on earlier features; they do not solve the whole task from raw input again.])

== Vision: depth builds strokes, then loops #V

#align(center, text(size: 11.5pt, weight: 700, fill: MUTED)[
  CONCEPTUAL SCHEMATIC · illustrates a possible hierarchy; not measured model activations
])
#v(2pt)
#align(center, image(
  "figures/depth-feature-hierarchy-vision.png",
  width: 244mm, height: 57mm, fit: "cover",
))
#v(4pt)
#grid(columns: (1fr, 1fr, 1fr, 1fr), column-gutter: 5mm,
  [#align(center)[#text(size: 11.5pt, weight: 700, fill: INK)[INPUT] \
    #text(size: 14pt)[pixel intensities]]],
  [#align(center)[#text(size: 11.5pt, weight: 700, fill: TEAL)[LAYER 1] \
    #text(size: 14pt)[curved stroke]]],
  [#align(center)[#text(size: 11.5pt, weight: 700, fill: BLUE)[LAYER 2] \
    #text(size: 14pt)[two closed loops]]],
  [#align(center)[#text(size: 11.5pt, weight: 700, fill: ACC)[OUTPUT] \
    #text(size: 14pt)[evidence for 8]]],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Early units respond to local edges; later units combine them into shapes that support the final class.])

== Audio: depth builds tones, then phonemes #V

#align(center, text(size: 11.5pt, weight: 700, fill: MUTED)[
  CONCEPTUAL SCHEMATIC · illustrates a possible hierarchy; not measured model activations
])
#v(2pt)
#align(center, image(
  "figures/depth-feature-hierarchy-audio.png",
  width: 244mm, height: 57mm, fit: "cover",
))
#v(4pt)
#grid(columns: (1fr, 1fr, 1fr, 1fr), column-gutter: 5mm,
  [#align(center)[#text(size: 11.5pt, weight: 700, fill: INK)[INPUT] \
    #text(size: 14pt)[waveform samples]]],
  [#align(center)[#text(size: 11.5pt, weight: 700, fill: TEAL)[LAYER 1] \
    #text(size: 14pt)[frequency + onset]]],
  [#align(center)[#text(size: 11.5pt, weight: 700, fill: BLUE)[LAYER 2] \
    #text(size: 14pt)[phoneme pattern]]],
  [#align(center)[#text(size: 11.5pt, weight: 700, fill: ACC)[OUTPUT] \
    #text(size: 14pt)[word evidence]]],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Early layers detect local frequency and timing cues; later layers combine them over longer time windows.])

== Text: depth combines words into phrase meaning #V

#align(center, text(size: 11.5pt, weight: 700, fill: MUTED)[
  CONCEPTUAL SCHEMATIC · illustrates a possible hierarchy; not measured model activations
])
#v(2pt)
#align(center, image(
  "figures/depth-feature-hierarchy-text.png",
  width: 244mm, height: 57mm, fit: "cover",
))
#v(4pt)
#grid(columns: (1fr, 1fr, 1fr, 1fr), column-gutter: 5mm,
  [#align(center)[#text(size: 11.5pt, weight: 700, fill: INK)[INPUT] \
    #text(size: 14pt)[tokens “not”, “good”]]],
  [#align(center)[#text(size: 11.5pt, weight: 700, fill: TEAL)[LAYER 1] \
    #text(size: 14pt)[ordered phrase]]],
  [#align(center)[#text(size: 11.5pt, weight: 700, fill: BLUE)[LAYER 2] \
    #text(size: 14pt)[negative evidence]]],
  [#align(center)[#text(size: 11.5pt, weight: 700, fill: ACC)[OUTPUT] \
    #text(size: 14pt)[negative review]]],
)
#pause
#place(bottom + center, dy: -2pt,
  result[A later unit can reverse the evidence from “good” after combining it with the preceding token “not”.])

== Choose width for variety; choose depth for composition #D

#two(
  [#text(size: 20pt, weight: 700, fill: ACC)[ADD WIDTH]
   #v(7pt)
   - adds more features *in parallel*;
   - helps when many patterns live at the same abstraction level;
   - costs wider activations and weight matrices.
   #v(9pt)
   *Example:* add hinge locations to follow a detailed curve.],
  [#pause
   #text(size: 20pt, weight: 700, fill: TEAL)[ADD DEPTH]
   #v(7pt)
   - adds more transformations *in sequence*;
   - helps when the target is compositional or hierarchical;
   - costs longer paths and harder optimization.
   #v(9pt)
   *Example:* strokes $arrow.r$ loops $arrow.r$ digit identity.],
)
#pause
#v(12pt)
#notebox[There is no universally best shape. Compare models under a meaningful parameter, memory, and compute budget.]
#pause
#place(bottom + center, dy: -2pt,
  result[Modern networks usually need both: enough width per stage and enough depth to compose stages.])

== Interactive: compare width and depth on the same task #I

#interbox(link-to: IA + "mlp-decision-boundary")[
  Train a linear classifier or a small MLP on *XOR / circles / spirals / moons*. Compare one versus two hidden layers, then adjust width while watching the learned boundary.
  Controls: dataset · linear/MLP · hidden width · one/two layers · activation · train/reset/resample.
]
#pause
#v(7pt)
#align(center, text(size: 17pt)[
  Hold the dataset fixed. First add width within one layer; then redistribute a similar number of hidden units across two layers. Which boundary appears sooner and which trains more reliably?
])
#pause
#place(bottom + center, dy: -2pt,
  result[The comparison is empirical: width and depth change both representational structure and optimization behaviour.])

// ═══════════════════════════ PART IX — Connecting to practice ═══════════════════════════
= Connecting to deep-learning practice

== Forward pass, step 1: flatten the image into a vector #V

#align(center, text(size: 18pt)[Start with a $2 times 2$ toy image so that every number stays visible.])
#v(10pt)
#align(center, grid(
  columns: (82mm, 38mm, 82mm), align: horizon,
  neat-card([pixel grid], [
    $bold(X)_("image")=mat(1,0;1,1)$ \
    four pixel intensities
  ], color: INK, width: 82mm),
  neat-arrow(label: [read row by row]),
  neat-card([input vector], [
    $bold(x)=mat(1;0;1;1) in RR^4$ \
    the same four values
  ], color: TEAL, width: 82mm),
))
#pause
#v(13pt)
#align(center, neat-card([real image], [
  a $28 times 28$ image becomes $bold(x) in RR^784$ because $28(28)=784$
], color: BLUE, width: 154mm))
#pause
#place(bottom + center, dy: -2pt,
  result[Flattening changes the arrangement of the pixels—not their values or meaning.])

== Forward pass, step 2: one hidden unit tests one pattern #D

#align(center, diagram(spacing: (23mm, 16mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let inputs = ((0,-1.5), (0,-0.5), (0,0.5), (0,1.5))
  let names = ([$x_1$], [$x_2$], [$x_3$], [$x_4$])
  let vals = ([$1$], [$0$], [$1$], [$1$])
  let weights = ([$+1$], [$-1$], [$+0.5$], [$+0.5$])
  let sum-pos = (3.2, 0)
  let act = (5.25, 0)
  for i in range(4) {
    edge(inputs.at(i), sum-pos,
      text(size: 11.5pt, weight: 700, fill: TEAL)[#weights.at(i)],
      "-|>", stroke: 1pt + TEAL, label-side: left)
    node(inputs.at(i), align(center, text(size: 11pt)[#names.at(i) \
      #vals.at(i)]), radius: 7.5mm, stroke: 1pt + INK)
  }
  edge((1.75, 1.75), sum-pos, text(size: 11.5pt, weight: 700, fill: ACC)[$-0.5$],
    "-|>", stroke: 1pt + ACC, label-side: right)
  edge(sum-pos, act, text(size: 11pt, weight: 700, fill: BLUE)[ReLU],
    "-|>", stroke: 1.1pt + BLUE, label-side: left)
  node((1.75, 1.75), [$1$], radius: 6mm, fill: ACC.lighten(90%), stroke: 1pt + ACC)
  node(sum-pos, text(size: 16pt)[$sum$], radius: 9mm, stroke: 1.2pt + TEAL)
  node((3.2, 0.72), text(size: 10pt, weight: 650, fill: TEAL)[$u=1.5$], stroke: none)
  node(act, align(center, text(size: 12pt)[$h$ \
    #text(size: 9pt, fill: BLUE)[$=1.5$]]), radius: 10mm, stroke: 1.2pt + BLUE)
}))
#pause
#place(bottom + center, dy: -2pt,
  result[#text(size: 17pt)[
    $u=1(1)-1(0)+0.5(1)+0.5(1)-0.5=1.5$, then $h=max(0,u)=1.5$.
    The edge weights define the pixel pattern this feature detects.
  ]])

== Forward pass, step 3: 128 units test 128 patterns #V

#grid(
  columns: (105mm, 1fr), column-gutter: 14pt, align: horizon,
  [#block(width: 105mm, height: 76mm, clip: true, inset: 0pt)[
     #align(center + horizon, scale(82%)[
       #mlp-diagram((4, 5), labels: ([784 input pixels], [128 hidden units]))
     ])
   ]
   #v(3pt)
   #neat-caption([schematic: only a few nodes are drawn])],
  [Each receiving unit has its own row of weights:
   #v(6pt)
   $u_j=bold(w)_j^top bold(x)+b_j$ \
   $h_j=max(0,u_j)$ \
   #v(8pt)
   $bold(u)=bold(W)_1 bold(x)+bold(b)_1 in RR^128$ \
   $bold(h)="ReLU"(bold(u)) in RR^128$],
)
#pause
#v(5pt)
#align(center, text(size: 16.5pt)[
  For one trained layer: #text(fill: BLUE)[$h_17=2.1$] $arrow.r$ curved-stroke unit *on*; #h(7pt)
  #text(fill: ACC)[$h_83=0$] $arrow.r$ another unit *off*.
])
#pause
#place(bottom + center, dy: -2pt,
  result[The hidden vector is a list of 128 learned measurements of the same image.])

== Forward pass, step 4: hidden features become class scores #D

#align(center, text(size: 18pt)[Use a three-class toy output so that every logit can be computed explicitly.])
#v(8pt)
#align(center, grid(
  columns: (69mm, 12mm, 106mm), align: horizon,
  neat-card([hidden features], [$bold(h)=(1.5,0.5)^top$], color: BLUE, width: 69mm),
  neat-arrow(),
  neat-card([linear output layer], [
    #set text(size: 16pt)
    $bold(W)_2=mat(1,0;0,1;-1,1), quad bold(b)_2=(0.5,0,0)^top$ \
    $bold(z)=bold(W)_2 bold(h)+bold(b)_2=(2.0,0.5,-1.0)^top$
  ], color: ACC, width: 106mm),
))
#pause
#v(10pt)
#align(center, grid(
  columns: (1fr, 1fr, 1fr), column-gutter: 8pt,
  notebox[$z_A=1(1.5)+0(0.5)+0.5=2.0$],
  notebox[$z_B=0(1.5)+1(0.5)+0=0.5$],
  notebox[$z_C=-1(1.5)+1(0.5)+0=-1.0$],
))
#pause
#place(bottom + center, dy: -2pt,
  result[Logits are unrestricted class scores: larger means preferred, but they are not probabilities yet.])

== Forward pass, step 5: softmax compares every score #D

#align(center, grid(
  columns: (60mm, 10mm, 84mm, 10mm, 70mm), align: horizon,
  neat-card([logits], [$bold(z)=(2.0,0.5,-1.0)^top$], color: ACC, width: 60mm),
  neat-arrow(),
  neat-card([exponentiate and add], [
    $exp(bold(z)) approx (7.389,1.649,0.368)^top$ \
    total $=9.406$
  ], color: TEAL, width: 84mm),
  neat-arrow(),
  neat-card([probabilities], [
    #set text(size: 15pt)
    $bold(p) approx$ \
    $(0.786,0.175,0.039)^top$ \
    sum $=1$
  ], color: BLUE, width: 70mm),
))
#pause
#v(12pt)
#align(center, text(size: 20pt)[
  $p_A=frac(exp(2.0), exp(2.0)+exp(0.5)+exp(-1.0))=frac(7.389,9.406) approx 0.786$
])
#pause
#place(bottom + center, dy: -2pt,
  result[Softmax preserves the score ordering while turning all three scores into one probability distribution.])

== Forward pass, step 6: the label selects the loss #D

#align(center, text(size: 19pt)[Cross-entropy reads the probability assigned to the true class: $cal(L)=-log p_y$.])
#v(12pt)
#align(center, grid(
  columns: (1fr, 1fr, 1fr), column-gutter: 10pt,
  neat-card([if y = A], [$cal(L)=-log(0.786) approx 0.241$], color: GREEN, width: 72mm),
  neat-card([if y = B], [$cal(L)=-log(0.175) approx 1.743$], color: ACC, width: 72mm),
  neat-card([if y = C], [$cal(L)=-log(0.039) approx 3.244$], color: RED, width: 72mm),
))
#pause
#v(13pt)
#two(
  notebox[High probability on the true class gives a small loss.],
  notebox[Confident probability on the wrong class gives a large loss.],
)
#pause
#place(bottom + center, dy: -2pt,
  result[The forward pass ends with one scalar loss; it has evaluated the weights but has not updated them.])

== A minibatch stacks examples as rows #D

#align(center, text(size: 18pt)[Apply the same hidden unit from step 2 to three inputs at once.])
#v(8pt)
#align(center, grid(
  columns: (89mm, 12mm, 105mm), align: horizon,
  neat-card([three input rows], [
    $bold(X)=mat(1,0,1,1;0,1,1,0;1,1,0,0)$ \
    shape $3 times 4$
  ], color: INK, width: 89mm),
  neat-arrow(),
  neat-card([one feature across the batch], [
    $bold(u)=bold(X)bold(w)+b bold(1)=(1.5,-1,-0.5)^top$ \
    $bold(h)="ReLU"(bold(u))=(1.5,0,0)^top$
  ], color: TEAL, width: 105mm),
))
#pause
#v(10pt)
#align(center, text(size: 19pt)[The same $bold(w)$ and $b$ are reused for row 1, row 2, and row 3.])
#pause
#place(bottom + center, dy: -2pt,
  result[Rows are different examples; columns are the features computed for those examples.])

== The full minibatch keeps the same network #D

For $n=64$ images, all $128$ hidden units and all $10$ logits are computed together:
#v(9pt)
#align(center, grid(
  columns: (65mm, 12mm, 76mm, 12mm, 65mm), align: horizon,
  neat-card([batch input], [$bold(X): 64 times 784$ \
    one image per row], color: INK, width: 65mm),
  neat-arrow(),
  neat-card([batch hidden features], [$bold(H)="ReLU"(bold(X)bold(W)_1^top+bold(1)bold(b)_1^top)$ \
    shape $64 times 128$], color: TEAL, width: 76mm),
  neat-arrow(),
  neat-card([batch logits], [$bold(Z):64 times 10$ \
    ten scores per row], color: ACC, width: 65mm),
))
#pause
#v(10pt)
#two(
  notebox[*Shared across rows:* weights and biases.],
  notebox[*Recomputed for this batch:* activations, probabilities, and losses.],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Batching changes the array shapes, not the prediction rule: every row follows the same network.])

== A tiny layer reveals the parameter-count rule #V

#grid(
  columns: (112mm, 1fr), column-gutter: 12pt, align: horizon,
  [#mlp-diagram((4, 3), labels: ([4 incoming values], [3 receiving units]))
   #v(4pt)
   #neat-caption([every connecting edge carries one weight])],
  [*Weights* \
   $4$ inputs $times 3$ units $=12$
   #v(9pt)
   *Biases* \
   one per receiving unit $=3$
   #v(9pt)
   *Total* \
   $12+3=15$ parameters],
)
#pause
#v(9pt)
#align(center, neat-card([general rule], [
  $("output width" times "input width") + "output width"$
], color: GREEN, width: 162mm))
#pause
#place(bottom + center, dy: -2pt,
  result[Count the arrows, then add one bias for every receiving node.])

== Apply the same counting rule to both affine layers #D

#two(r: (0.9fr, 1.1fr),
  [#align(center, [
    #mlp-diagram((4, 5, 3), labels: ([$784$ inputs], [$128$ hidden units], [$10$ outputs]))
    #v(5pt)
    #neat-caption([The drawing is schematic; the labels give the actual widths.])
  ])],
  [#text(size: 18pt, weight: 700, fill: TEAL)[INPUT $arrow.r$ HIDDEN]
   #v(4pt)
   $bold(W)_1:128 times 784 quad arrow.r quad 128(784)="100,352"$ \
   $bold(b)_1 in RR^128 quad arrow.r quad 128$
   #v(4pt)
   *Subtotal:* $"100,352"+128="100,480"$
   #pause
   #v(10pt)
   #text(size: 18pt, weight: 700, fill: ACC)[HIDDEN $arrow.r$ OUTPUT]
   #v(4pt)
   $bold(W)_2:10 times 128 quad arrow.r quad 10(128)="1,280"$ \
   $bold(b)_2 in RR^10 quad arrow.r quad 10$
   #v(4pt)
   *Subtotal:* $"1,280"+10="1,290"$],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Total trainable parameters: $"100,480"+"1,290"="101,770"$.])

== Training changes parameters, not the architecture #D

#align(center, grid(
  columns: (43mm, 7mm, 43mm, 7mm, 43mm, 7mm, 43mm, 7mm, 43mm), align: horizon,
  neat-card([current parameters], [$bold(theta)$], color: INK, width: 43mm),
  neat-arrow(),
  neat-card([forward pass], [activations and prediction], color: TEAL, width: 43mm),
  neat-arrow(),
  neat-card([batch loss], [$cal(L)$], color: ACC, width: 43mm),
  neat-arrow(),
  neat-card([gradient], [$nabla_(bold(theta))cal(L)$], color: BLUE, width: 43mm),
  neat-arrow(),
  neat-card([updated parameters], [$bold(theta)_("new")$], color: GREEN, width: 43mm),
))
#pause
#v(12pt)
#align(center, text(size: 22pt)[$bold(theta)_("new")=bold(theta)_("old")-eta nabla_(bold(theta))cal(L)$])
#pause
#v(10pt)
#two(
  notebox[*Fixed:* layer sizes, connectivity pattern, and activation choices.],
  notebox[*Changed:* the entries of weight matrices and bias vectors.],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Forward propagation produces activations and loss; learning uses that loss to update the shared parameters.])

== One numerical parameter update #D

#align(center, grid(
  columns: (58mm, 10mm, 58mm, 10mm, 58mm), align: horizon,
  neat-card([current value], [$w_("old")=0.8$], color: INK, width: 58mm),
  neat-arrow(),
  neat-card([gradient and step size], [$frac(partial cal(L),partial w)=-0.3$ \
    $eta=0.1$], color: BLUE, width: 58mm),
  neat-arrow(),
  neat-card([updated value], [$w_("new")=0.8-0.1(-0.3)=0.83$], color: GREEN, width: 58mm),
))
#pause
#v(13pt)
#align(center, text(size: 20pt)[The negative gradient says increasing $w$ locally decreases the loss, so the update moves $w$ upward.])
#pause
#notebox[Lecture 3 will explain how backpropagation computes every component of $nabla_(bold(theta))cal(L)$ efficiently.]
#pause
#place(bottom + center, dy: -2pt,
  result[The forward pass evaluates the current network; the update changes the network for the next pass.])

== The decisive change is the learned representation

#align(center, [
  #mlp-diagram(
    (3, 5, 5, 2),
    labels: ([input $bold(x)$], [learned $bold(h)^((1))$], [learned $bold(h)^((2))$], [output $bold(z)$]),
  )
  #v(8pt)
  #neat-caption([The final scores still feed the familiar Lecture 1 output model and loss.])
])
#pause
#place(bottom + center, dy: -2pt,
  result[The output loss can stay the same while hidden layers make the representation nonlinear.])

// closing standout
#focus-slide[
  We now have $bold(x) -> f_(bold(theta))(bold(x)) -> cal(L)$.
  #v(12pt)
  #set text(size: 22pt)
  Lecture 3: how do we compute $display(nabla_(bold(theta)) cal(L))$ efficiently?
  #v(8pt)
  #text(size: 17pt, fill: rgb("#B9C6C8"))[computation graphs · the chain rule · backpropagation · automatic differentiation]
]
