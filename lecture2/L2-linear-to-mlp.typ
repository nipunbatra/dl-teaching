// From Linear Models to Neural Networks — Lecture 2 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture2/L2-linear-to-mlp.typ
//   typst compile --root . --input handout=true lecture2/L2-linear-to-mlp.typ
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
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
#let compgraph = align(center, diagram(spacing: 13mm, node-stroke: 0.9pt + INK, node-fill: white, {
  let labs = ($bold(x)$, $bold(h)^((1))$, $bold(h)^((2))$, $bold(z)$, $bold(p)$, $cal(L)$)
  for (i, l) in labs.enumerate() {
    let c = if i == labs.len() - 1 { ACC } else if i == 0 { INK } else { TEAL }
    node((i, 0), l, radius: 6mm, stroke: 0.9pt + c)
  }
  for i in range(labs.len() - 1) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.8pt + MUTED) }
}))

// stacked-depth schematic: weighted sum + bias → nonlinearity → … → output
#let depth-schematic = align(center, diagram(spacing: (8mm, 8mm), node-stroke: 0.9pt, {
  node((0,0), $bold(x)$, radius: 5mm, stroke: INK)
  node((1,0), [$bold(W)_1 dot + bold(b)_1$], shape: fletcher.shapes.rect, fill: rgb("#E3EDED"), stroke: TEAL, inset: 6pt)
  node((2,0), $phi$, radius: 5mm, fill: rgb("#FDECD6"), stroke: ACC)
  node((3,0), [$bold(W)_2 dot + bold(b)_2$], shape: fletcher.shapes.rect, fill: rgb("#E3EDED"), stroke: TEAL, inset: 6pt)
  node((4,0), $phi$, radius: 5mm, fill: rgb("#FDECD6"), stroke: ACC)
  node((5,0), $dots.h$, stroke: none)
  node((6,0), [$bold(W)_L dot + bold(b)_L$], shape: fletcher.shapes.rect, fill: rgb("#E3EDED"), stroke: TEAL, inset: 6pt)
  node((7,0), $bold(z)$, radius: 5mm, stroke: ACC)
  for i in range(7) { edge((i,0),(i+1,0), "-|>", stroke: 0.8pt + MUTED) }
}))

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

== The map for today
#stag(V)

One question drives the lecture: *what changes when one weighted sum becomes a network?*
#v(8pt)
#pause
#align(center, diagram(spacing: (14mm, 8mm), node-stroke: 0.9pt, {
  let labels = ([linear score], [single neuron], [hidden features], [multilayer network])
  for (i, label) in labels.enumerate() {
    let fill = if i < 2 { rgb("#EFEEEB") } else { rgb("#FDECD6") }
    let stroke = if i < 2 { INK } else { ACC }
    node((i, 0), label, shape: fletcher.shapes.rect, fill: fill, stroke: stroke, inset: 7pt)
    if i > 0 { edge((i - 1, 0), (i, 0), "-|>", stroke: 0.8pt + MUTED) }
  }
}))
#v(8pt)
#pause
#align(center, text(size: 16.5pt, fill: MUTED)[
  weighted sum $arrow.r$ expose the limitation $arrow.r$ learn nonlinear features $arrow.r$ compose them
])
#pause
#notebox[We will calculate every stage for small numerical examples before writing the general matrix form.]

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

For an input vector $bold(x) in RR^d$ and scalar target $y$,
$ hat(y) = bold(w)^top bold(x) + b $
#pause
For $bold(x)=(2,-1)$, $bold(w)=(1.5,0.5)$, and $b=-0.2$:
$ hat(y)=1.5(2)+0.5(-1)-0.2=2.3. $
#pause
#two(
  notebox[Regression: read $2.3$ as the predicted mean.],
  notebox[Classification: read the weighted sum as a score, then convert it to probabilities.],
)
#pause
#result[The same weighted-sum core can feed different output models and losses from Lecture 1.]

== A whole batch is one matrix multiplication

For $bold(X) in RR^(n times d)$ and $bold(y) in RR^n$,
$ hat(bold(y)) = bold(X) bold(w) + b bold(1) = tilde(bold(X)) bold(theta). $
#pause
$ tilde(bold(X)) in RR^(n times (d+1)), quad bold(theta) in RR^(d+1), quad hat(bold(y)) in RR^n. $
#pause
#notebox[Rows are examples; columns are features; matrix multiplication performs all $n$ predictions together.]

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
#neuron-diagram(d: 3)

== One neuron recovers familiar Lecture 1 models

Let $z=bold(w)^top bold(x)+b$. The output interpretation chooses the last step:
#v(7pt)
#grid(
  columns: (1fr, 1fr, 1fr), column-gutter: 8pt,
  notebox[*Regression* \
    identity output \
    $hat(y)=z$],
  [#pause
   #notebox[*Binary class* \
     sigmoid output \
     $p=sigma(z)$]],
  [#pause
   #notebox[*$K$ classes* \
     $K$ scores + softmax \
     $bold(p)="softmax"(bold(z))$]],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Lecture 1 tells us how to interpret the output and choose the loss; Lecture 2 changes the features entering that output.])

== Common activation functions #V

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 9pt,
  align(center, lines(
    fn: _sigmoid, domain: (-4, 4), samples: 140,
    markers: false, colors: (BLUE,), y-ticks: false,
    x-label: [$z$], y-label: [$phi(z)$], title: [sigmoid], size: (50mm, 37mm),
  )),
  align(center, lines(
    fn: _tanh, domain: (-4, 4), samples: 140,
    markers: false, colors: (TEAL,), y-ticks: false,
    hlines: ((0, none, MUTED),),
    x-label: [$z$], y-label: [$phi(z)$], title: [tanh], size: (50mm, 37mm),
  )),
  align(center, lines(
    fn: _relu, domain: (-4, 4), samples: 140,
    markers: false, colors: (ACC,), y-ticks: false,
    x-label: [$z$], y-label: [$phi(z)$], title: [ReLU], size: (50mm, 37mm),
  )),
  align(center, lines(
    fn: _gelu, domain: (-4, 4), samples: 140,
    markers: false, colors: (GREEN,), y-ticks: false,
    hlines: ((0, none, MUTED),),
    x-label: [$z$], y-label: [$phi(z)$], title: [GELU], size: (50mm, 37mm),
  )),
)
#pause
#v(6pt)
#align(center, text(size: 16.5pt)[sigmoid → gates/probabilities · tanh → zero-centred · *ReLU* → common hidden unit · GELU → Transformers])

== Question: do more linear layers create new shapes? #Q

#mcq(
  [Remove every activation. What can a stack of “weighted sum + bias” layers represent?],
  [One equivalent weighted sum + bias],
  [Any curved decision boundary],
  [Only a constant function],
  [A probability distribution automatically],
)

== Answer: line after line is still a line #A

#two(r: (1.18fr, 1fr),
  [
    *Algebraic view*
    #v(5pt)
    $ bold(h)_1 = bold(W)_1 bold(x) + bold(b)_1 $
    $ bold(h)_2 = bold(W)_2 bold(h)_1 + bold(b)_2 $
    #v(4pt)
    $ bold(h)_2 = underbrace(bold(W)_2 bold(W)_1, "one matrix") bold(x)
      + underbrace((bold(W)_2 bold(b)_1 + bold(b)_2), "one bias") $
    #v(7pt)
    #notebox[$bold(W)_2 bold(W)_1$ is one new weight matrix; the remaining term is one new bias vector.]
  ],
  [
    *Geometric view*
    #v(3pt)
    #align(center, text(size: 15.5pt)[$h_1=x+1, quad h_2=2h_1-1=2x+1$])
    #v(2pt)
    #align(center, lines(
      fn: (x => x + 1, x => 2 * x + 1), domain: (-2, 2), samples: 100,
      markers: false, colors: (TEAL, ACC), dashes: ("dashed", "solid"),
      labels: ([$h_1=x+1$], [$h_2=2x+1$]), legend: "tl",
      x-label: [$x$], y-label: [output], size: (64mm, 40mm),
    ))
    #v(3pt)
    #align(center, text(size: 14.5pt, fill: MUTED)[The slope changes; no bend appears.])
  ],
)
#place(bottom + center, dy: -2pt,
  result[Correct: *A*. Without a nonlinear activation, extra layers still reduce to one weighted sum plus bias.])

== Interactive: activation functions #I

#interbox(link-to: "https://nipunbatra.github.io/dl-teaching/interactives/activation-explorer.html")[
  Compare activations and their *derivatives*, and see where gradients vanish.
  Controls: activation type · input $z$ · show derivative · compare saturation.
]
#pause
Explore large positive and negative logits, where sigmoid's derivative becomes tiny.

// ═══════════════════════════ PART IV — Where linear scores fail ═══════════════════════════
= Where a single weighted sum fails

== Single-layer classifiers draw only straight boundaries #V

#two(r: (1fr, 1fr),
  [*Binary:* threshold one score $z=bold(w)^top bold(x)+b$.
   #v(6pt)
   #fig("/lecture2/figures/linear_boundary.pdf", w: 64%)],
  [#pause
   *Multi-class:* compare several linear scores.
   #v(6pt)
   #fig("/lecture2/figures/softmax_regions.pdf", w: 64%)],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Sigmoid and softmax change scores into probabilities; they do not bend the boundaries in input space.])

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

== Answer, part 1: equal probabilities give a flat tie set #A

#two(r: (1fr, 1fr),
  [
    *Softmax preserves score comparisons.*
    #v(5pt)
    $ frac(p_i, p_j) = frac(exp(z_i), exp(z_j)) = exp(z_i-z_j). $
    #v(5pt)
    Therefore
    $ p_i=p_j arrow.l.r z_i=z_j, $
    $ p_i>p_j arrow.l.r z_i>z_j. $
    #v(7pt)
    #notebox[$arg max_k p_k = arg max_k z_k$: softmax changes the scale, not the ranking.]
  ],
  [
    *Linear-score ties are flat.*
    #v(5pt)
    $ z_k(bold(x))=bold(w)_k^top bold(x)+b_k $
    $ z_i=z_j arrow.l.r (bold(w)_i-bold(w)_j)^top bold(x)+(b_i-b_j)=0. $
    #v(5pt)
    This equation is a line in 2-D and a hyperplane in higher dimensions.
    #v(7pt)
    #notebox[It is the *candidate pairwise boundary* between classes $i$ and $j$.]
  ],
)
#place(bottom + center, dy: -2pt,
  result[Pairwise equality explains why every candidate boundary is straight. But is every tie visible?])

== Answer, part 2: the tied classes must also be on top #A

#two(r: (0.9fr, 1.1fr),
  [
    *A pairwise tie alone is not enough.*
    #v(5pt)
    Suppose
    $ bold(z)=(1,1,3). $
    Then
    $ p_1=p_2 < p_3. $
    #v(4pt)
    The prediction is class 3. The point lies on $z_1=z_2$, but *not* on a visible class-1/class-2 boundary.
  ],
  [
    *Decision region for class $i$*
    $ cal(R)_i = { bold(x): z_i >= z_l " for every " l }. $
    #v(6pt)
    *Boundary shared by classes $i$ and $j$*
    $ cal(B)_(i j) = cal(R)_i inter cal(R)_j $
    $ cal(B)_(i j) = { bold(x): z_i=z_j >= z_l " for every " l }. $
    #v(5pt)
    Equivalently, $p_i=p_j$ and both probabilities are maximal.
  ],
)
#place(bottom + center, dy: -2pt,
  result[Correct: *B*. The visible boundary is the top-scoring portion of a pairwise linear tie set.])

== XOR exposes the missing ingredient #V

#two(r: (1.05fr, 1fr),
  [No single line separates the two XOR classes.
   #v(6pt)
   The model first needs a transformation of the input—new features in which the classes become separable.],
  [#pause
   #fig("/lecture2/figures/xor.pdf", w: 80%)],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Hidden nonlinear units learn the transformation instead of asking us to hand-design it.])

// ═══════════════════════════ PART V — Multilayer perceptron ═══════════════════════════
= The multilayer perceptron

== A hidden layer learns features before the output

Instead of $bold(z) = bold(W) bold(x) + bold(b)$, insert a nonlinear layer first:
$ bold(h) = phi(bold(W)_1 bold(x) + bold(b)_1), quad bold(z) = bold(W)_2 bold(h) + bold(b)_2 $
#pause
For classification, $bold(p) = "softmax"(bold(z))$. This is a *one-hidden-layer MLP*.

== Layer dimensions must agree #V

#two(r: (1.25fr, 1fr),
  mlp-diagram(
    (3, 5, 2),
    labels: ([$d$ inputs], [$m$ hidden units], [$K$ output scores]),
  ),
  [#pause
    $bold(x) in RR^d$ \
    $bold(W)_1 in RR^(m times d), quad bold(b)_1 in RR^m$ \
    $bold(h) in RR^m$ \
    $bold(W)_2 in RR^(K times m), quad bold(b)_2 in RR^K$ \
    $bold(z) in RR^K$
  ],
)

== First see the complete $2 arrow.r 2 arrow.r 1$ network #V

#two(r: (1.05fr, 1fr),
  [#align(center, diagram(spacing: (15mm, 13mm), node-stroke: 0.9pt, {
    let pos = (
      (0, -0.65), (0, 0.65),
      (2, -0.65), (2, 0.65),
      (3.25, 0),
    )
    // Connections first, so they remain behind the units.
    for i in range(2) { for j in range(2) {
      edge(pos.at(i), pos.at(2 + j), "-|>", stroke: 0.7pt + MUTED)
    } }
    for j in range(2) {
      edge(pos.at(2 + j), pos.at(4), "-|>", stroke: 0.8pt + MUTED)
    }
    node(pos.at(0), text(fill: white)[$x_1$], radius: 5.5mm, fill: INK, stroke: none)
    node(pos.at(1), text(fill: white)[$x_2$], radius: 5.5mm, fill: INK, stroke: none)
    node(pos.at(2), text(fill: white)[$h_1$], radius: 5.5mm, fill: TEAL, stroke: none)
    node(pos.at(3), text(fill: white)[$h_2$], radius: 5.5mm, fill: TEAL, stroke: none)
    node(pos.at(4), text(fill: white)[$p$], radius: 5.5mm, fill: ACC, stroke: none)

    node((0.2, -1.55), text(size: 10.5pt, weight: 700, fill: INK)[INPUT], stroke: none)
    node((2, -1.55), text(size: 10.5pt, weight: 700, fill: TEAL)[HIDDEN · ReLU], stroke: none)
    node((3.1, -1.55), text(size: 10.5pt, weight: 700, fill: ACC)[OUTPUT · SIGMOID], stroke: none)
    node((0.35, 1.55), text(size: 11pt, fill: MUTED)[$bold(x)=(2,-1)$], stroke: none)
    node((2, 1.55), text(size: 11pt, fill: MUTED)[$bold(h)=(3,0.5)$], stroke: none)
    node((3.0, 1.55), text(size: 10.5pt, fill: MUTED)[$p approx 0.769$], stroke: none)
  }))],
  [#pause
   *Hidden layer*
   $ bold(W)_1 = mat(1, -1; 0.5, 1), quad bold(b)_1=(0,0.5) $
   #v(7pt)
   *Output layer*
   $ bold(w)_2=(0.8,-0.4), quad b_2=-1 $
   #v(7pt)
   *Training example*
   $ y=1, quad cal(L)=-log p $
  ],
)
#pause
#v(7pt)
#align(center, text(size: 17pt)[
  $bold(x) arrow.r bold(u)=bold(W)_1 bold(x)+bold(b)_1 arrow.r bold(h)="ReLU"(bold(u))$
  $arrow.r z=bold(w)_2^top bold(h)+b_2 arrow.r p=sigma(z) arrow.r cal(L)$
])
#pause
#place(bottom + center, dy: -2pt,
  result[Now calculate the same network from left to right.])

== Work a $2 arrow.r 2$ hidden layer by hand
#stag(D)

Let $bold(x)=(2,-1)$ and
$ bold(W)_1 = mat(1, -1; 0.5, 1), quad bold(b)_1=(0,0.5). $
#pause
The hidden pre-activation is
$ bold(u) = bold(W)_1 bold(x)+bold(b)_1 = (3,0.5). $
#pause
Apply ReLU elementwise:
$ bold(h) = "ReLU"(bold(u)) = (3,0.5). $
#pause
#result[The hidden layer converts the raw input into two learned features.]

== The output layer turns hidden features into a prediction
#stag(D)

For binary classification, choose $bold(w)_2=(0.8,-0.4)$ and $b_2=-1$.
#pause
$ z = bold(w)_2^top bold(h)+b_2 = 0.8(3)-0.4(0.5)-1 = 1.2. $
#pause
$ p = sigma(1.2) approx 0.769. $
#pause
If $y=1$, then $cal(L)=-log(0.769) approx 0.262$.
#pause
#result[$bold(x) arrow.r bold(h) arrow.r z arrow.r p arrow.r cal(L)$ is one complete forward pass.]

== Each hidden neuron creates one learned feature

$ h_j = phi(bold(w)_j^top bold(x) + b_j) $
#pause
- the first layer *learns features*;
- the output layer *combines* them.
#pause
For ReLU, $h_j = max(0, bold(w)_j^top bold(x) + b_j)$ — each unit is active on one side of a hyperplane.

== The feature-transformation view #V

#fig("/lecture2/figures/feature_transform.pdf", w: 62%)
#pause
#result[An MLP can learn a representation where classification becomes linear.]

== Binary uses one output node; $K$-class uses $K$ #V

#two(r: (1fr, 1fr),
  [#align(center)[
     #text(size: 19pt, weight: 700, fill: ACC)[BINARY CLASSIFICATION]
     #v(5pt)
     #mlp-diagram(
       (3, 4, 1),
       labels: ([$d$ inputs], [$m$ hidden units], [*1 output node*]),
     )
     #v(6pt)
     $ z in RR arrow.r p=sigma(z) $
   ]],
  [#pause
   #align(center)[
     #text(size: 19pt, weight: 700, fill: ACC)[MULTI-CLASS CLASSIFICATION]
     #v(5pt)
     #mlp-diagram(
       (3, 4, 3),
       labels: ([$d$ inputs], [$m$ hidden units], [*$K$ output nodes*]),
     )
     #v(6pt)
     $ bold(z) in RR^K arrow.r bold(p)="softmax"(bold(z)) $
   ]],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Binary: 1 node gives $p$ and $1-p$. Multi-class: $K$ nodes give $K$ softmax probabilities.])

== Depth stacks several hidden layers #V

$ bold(h)^((1)) = phi(bold(W)_1 bold(x) + bold(b)_1) $
$ bold(h)^((ell)) = phi(bold(W)_ell bold(h)^((ell - 1)) + bold(b)_ell), quad ell = 2, dots, L - 1 $
$ bold(z) = bold(W)_L bold(h)^((L - 1)) + bold(b)_L $
#pause
#v(5pt)
#mlp-diagram(
  (3, 4, 5, 4, 2),
  labels: (
    [input $bold(x)$],
    [hidden $bold(h)^((1))$],
    [hidden $bold(h)^((2))$],
    [hidden $bold(h)^((L - 1))$],
    [output $bold(z)$],
  ),
)
#pause
#place(bottom + center, dy: -2pt,
  result[Each hidden layer transforms the previous representation. The output layer then produces task-specific scores or predictions.])

== Width and depth enlarge a network differently #V

#two(
  [
    #align(center, [
      #text(size: 20pt, weight: 700, fill: ACC)[WIDER]
      #v(3pt)
      #text(size: 16pt)[One hidden layer, *six neurons*]
      #v(7pt)
      #mlp-diagram((2, 6, 1))
    ])
  ],
  [
    #pause
    #align(center, [
      #text(size: 20pt, weight: 700, fill: TEAL)[DEEPER]
      #v(3pt)
      #text(size: 16pt)[Two hidden layers, *three neurons each*]
      #v(7pt)
      #mlp-diagram((2, 3, 3, 1))
    ])
  ],
)
#pause
#place(bottom + center, dy: -2pt,
  result[*Width* adds neurons within a layer. *Depth* adds successive transformations.])

== Interactive: MLP decision boundaries #I

#interbox(link-to: IA + "mlp-decision-boundary")[
  Train a linear classifier or a small MLP on *XOR / circles / spirals / moons* and watch the boundary form.
  Controls: dataset · linear/MLP · hidden width · one/two layers · activation · train/reset/resample.
]
#pause
Compare wider layers (more features at one stage) with deeper layers (more composed stages).

// ═══════════════════════════ PART VII — Universal approximation ═══════════════════════════
= Why MLPs represent complex functions

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

== Step 1: one hidden layer creates three shifted ReLUs #V

#two(r: (1.05fr, 1fr),
  [
    *Unit 1* uses weight $1$ and bias $0$:
    $u_1 = 1x + 0, quad h_1 = "ReLU"(u_1) = "ReLU"(x).$
    #v(7pt)
    #pause
    *Unit 2* uses weight $1$ and bias $-1$:
    $u_2 = 1x - 1, quad h_2 = "ReLU"(u_2) = "ReLU"(x-1).$
    #v(7pt)
    #pause
    *Unit 3* uses weight $1$ and bias $-2$:
    $u_3 = 1x - 2, quad h_3 = "ReLU"(u_3) = "ReLU"(x-2).$
  ],
  [#align(center, diagram(spacing: (25mm, 16mm), node-stroke: 0.9pt, {
    // Connections first, so they remain behind the units.
    edge((0, 0), (1, -1), "-|>", stroke: 0.8pt + MUTED)
    edge((0, 0), (1, 0), "-|>", stroke: 0.8pt + MUTED)
    edge((0, 0), (1, 1), "-|>", stroke: 0.8pt + MUTED)

    node((0, 0), text(fill: white)[$x$], radius: 6mm, fill: INK, stroke: none)
    node((1, -1), [$h_1="ReLU"(x)$], shape: fletcher.shapes.rect,
      fill: BLUE.lighten(83%), stroke: BLUE, inset: 7pt)
    node((1, 0), [$h_2="ReLU"(x-1)$], shape: fletcher.shapes.rect,
      fill: RED.lighten(86%), stroke: RED, inset: 7pt)
    node((1, 1), [$h_3="ReLU"(x-2)$], shape: fletcher.shapes.rect,
      fill: GREEN.lighten(84%), stroke: GREEN, inset: 7pt)

    node((0, -1.65), text(size: 11pt, weight: 700, fill: INK)[INPUT], stroke: none)
    node((1, -1.65), text(size: 11pt, weight: 700, fill: TEAL)[THREE HIDDEN UNITS], stroke: none)
  }))],
)
#pause
#place(bottom + center, dy: -2pt,
  result[The hidden biases move the three hinge locations to $x=0,1,2$. No factor $-2$ has appeared yet.])

== Step 2: the output layer supplies the coefficients #V

#align(center, diagram(spacing: (28mm, 18mm), node-stroke: 0.9pt, {
  // Hidden-layer connections.
  edge((0, 0), (1, -1), "-|>", stroke: 0.8pt + MUTED)
  edge((0, 0), (1, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((0, 0), (1, 1), "-|>", stroke: 0.8pt + MUTED)
  // Output-layer connections. Their labels are the second-layer weights.
  edge((1, -1), (2, 0), text(size: 14pt, weight: 700, fill: BLUE)[$+1$],
    "-|>", stroke: 1.1pt + BLUE, label-side: left)
  edge((1, 0), (2, 0), text(size: 14pt, weight: 700, fill: RED)[$-2$],
    "-|>", stroke: 1.1pt + RED, label-side: left)
  edge((1, 1), (2, 0), text(size: 14pt, weight: 700, fill: GREEN)[$+1$],
    "-|>", stroke: 1.1pt + GREEN, label-side: right)

  node((0, 0), text(fill: white)[$x$], radius: 6mm, fill: INK, stroke: none)
  node((1, -1), [$h_1="ReLU"(x)$], shape: fletcher.shapes.rect,
    fill: BLUE.lighten(83%), stroke: BLUE, inset: 7pt)
  node((1, 0), [$h_2="ReLU"(x-1)$], shape: fletcher.shapes.rect,
    fill: RED.lighten(86%), stroke: RED, inset: 7pt)
  node((1, 1), [$h_3="ReLU"(x-2)$], shape: fletcher.shapes.rect,
    fill: GREEN.lighten(84%), stroke: GREEN, inset: 7pt)
  node((2, 0), text(fill: white)[$f(x)$], radius: 7mm, fill: ACC, stroke: none)

  node((0, -1.65), text(size: 11pt, weight: 700, fill: INK)[INPUT], stroke: none)
  node((1, -1.65), text(size: 11pt, weight: 700, fill: TEAL)[HIDDEN ReLU UNITS], stroke: none)
  node((2, -1.65), text(size: 11pt, weight: 700, fill: ACC)[LINEAR OUTPUT], stroke: none)
}))
#pause
#v(5pt)
#align(center, text(size: 20pt)[
  $bold(h) = "ReLU"(mat(1;1;1)x + mat(0;-1;-2)), quad
   f(x) = mat(1, -2, 1) bold(h).$
])
#pause
#place(bottom + center, dy: -2pt,
  result[The $-2$ is an *output-layer weight*: the second hidden activation is multiplied by $-2$ only after ReLU.])

== Step 3: inspect the output contributions separately #V

#grid(
  columns: (1fr, 1fr, 1fr),
  column-gutter: 12pt,
  [
    #align(center, [
      #text(size: 17pt, weight: 650, fill: BLUE)[$g_1(x) = (+1)h_1$]
      #v(6pt)
      #lines(
        fn: x => calc.max(0, x),
        domain: (-1, 3), samples: 100, markers: false, colors: (BLUE,),
        points: ((0, 0, [$0$]),), y-ticks: false,
        x-label: [$x$], y-label: [$g_1(x)$], size: (47mm, 36mm),
      )
      #v(5pt)
      #text(size: 15pt, fill: MUTED)[hinge at $x=0$]
    ])
  ],
  [
    #pause
    #align(center, [
      #text(size: 17pt, weight: 650, fill: RED)[$g_2(x) = (-2)h_2$]
      #v(6pt)
      #lines(
        fn: x => -2 * calc.max(0, x - 1),
        domain: (-1, 3), samples: 100, markers: false, colors: (RED,),
        points: ((1, 0, [$1$]),), y-ticks: false,
        x-label: [$x$], y-label: [$g_2(x)$], size: (47mm, 36mm),
      )
      #v(5pt)
      #text(size: 15pt, fill: MUTED)[$h_2 >= 0$; output weight $-2$ flips it]
    ])
  ],
  [
    #pause
    #align(center, [
      #text(size: 17pt, weight: 650, fill: GREEN)[$g_3(x) = (+1)h_3$]
      #v(6pt)
      #lines(
        fn: x => calc.max(0, x - 2),
        domain: (-1, 3), samples: 100, markers: false, colors: (GREEN,),
        points: ((2, 0, [$2$]),), y-ticks: false,
        x-label: [$x$], y-label: [$g_3(x)$], size: (47mm, 36mm),
      )
      #v(5pt)
      #text(size: 15pt, fill: MUTED)[hinge at $x=2$]
    ])
  ],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Hidden activations $h_j$ are the features; $g_j = v_j h_j$ are their signed contributions to the output.])

== Step 4: add the active terms on each interval #D

#align(center, text(size: 19pt)[
  $f(x) = h_1 - 2h_2 + h_3 = "ReLU"(x) - 2"ReLU"(x-1) + "ReLU"(x-2).$
])
#pause
#v(7pt)
#align(center, {
  set text(size: 14.5pt)
  table(
    columns: (29mm, 31mm, 36mm, 36mm, 53mm),
    stroke: 0.5pt + MUTED, inset: (x: 7pt, y: 5.5pt),
    align: (center, center, center, center, left),
    table.header([*Input region*], [$h_1$], [$-2h_2$], [$h_3$], [*Sum $f(x)$*]),
    [$x<0$], [$0$], [$0$], [$0$], [$0$],
    [$0 <= x < 1$], [$x$], [$0$], [$0$], [$x$],
    [$1 <= x < 2$], [$x$], [$-2(x-1)$], [$0$], [$2-x$],
    [$x >= 2$], [$x$], [$-2(x-1)$], [$x-2$], [$0$],
  )
})
#pause
#place(bottom + center, dy: -2pt,
  result[The output rises as $x$, falls as $2-x$, then returns to zero: a tent.])

== Step 5: the weighted sum is the tent #V

#two(r: (0.9fr, 1.1fr),
  [
    #text(size: 18pt, weight: 650)[Hidden layer]
    $bold(h)(x) = ("ReLU"(x), "ReLU"(x-1), "ReLU"(x-2)).$
    #pause
    #v(9pt)
    #text(size: 18pt, weight: 650)[Linear output layer]
    $f(x) = mat(1, -2, 1) bold(h)(x).$
    #pause
    #v(9pt)
    The network has *three hidden nodes* and *one output node*. The output node performs no ReLU here; it only takes a weighted sum.
  ],
  [#align(center, lines(
    fn: x => calc.max(0, x) - 2 * calc.max(0, x - 1) + calc.max(0, x - 2),
    domain: (-1, 3), samples: 120, markers: false, colors: (ACC,),
    points: ((0, 0, [$0$]), (1, 1, [$1$]), (2, 0, [$2$])),
    x-label: [$x$], y-label: [$f(x)$], title: [network output], size: (73mm, 48mm),
  ))],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Hidden units create reusable nonlinear features; the output layer chooses their signs and strengths.])

== Five hinges create a two-peak function #V

#two(r: (0.95fr, 1.15fr),
  [A slightly richer weighted sum:
   #v(7pt)
   $ q(x) = "ReLU"(x) - 2"ReLU"(x-1) $
   $ quad + 2"ReLU"(x-2) - 2"ReLU"(x-3) $
   $ quad + "ReLU"(x-4). $
   #pause
   Each shifted unit changes the slope at one breakpoint.],
  align(center, lines(
    fn: x => calc.max(0, x) - 2 * calc.max(0, x - 1)
      + 2 * calc.max(0, x - 2) - 2 * calc.max(0, x - 3)
      + calc.max(0, x - 4),
    domain: (-0.5, 4.5), samples: 160, markers: false, colors: (ACC,),
    points: ((0, 0, [$0$]), (1, 1, [$1$]), (2, 0, [$2$]),
      (3, 1, [$3$]), (4, 0, [$4$])),
    x-label: [$x$], y-label: [$q(x)$], size: (72mm, 48mm),
  )),
)
#pause
#place(bottom + center, dy: -2pt,
  result[More hidden units $arrow.r$ more breakpoints $arrow.r$ a more detailed piecewise-linear function.])

== More breakpoints can follow a smooth curve #V

#align(center, lines(
  fn: (x => calc.sin(x), x => _sin-pwl(x, knots: 7)),
  domain: (-3.14, 3.14), samples: 220,
  labels: ([target: $sin x$], [seven-knot ReLU approximation]),
  colors: (INK, ACC), dashes: ("solid", "dashed"), markers: (false, false),
  legend: "tl", x-label: [$x$], y-label: [$f(x)$], size: (118mm, 58mm),
))
#pause
#align(center, text(size: 16.5pt, fill: MUTED)[
  The orange approximation is still made only of straight segments; its corners are the hinges.
])
#pause
#place(bottom + center, dy: -2pt,
  result[Adding hidden units adds breakpoints and reduces the approximation gap.])

== Interactive: build functions from ReLU hinges #I

#interbox(link-to: "https://nipunbatra.github.io/dl-teaching/interactives/relu-function-builder.html")[
  Recreate the sequence: inspect one hinge, combine three into a tent, add two more for a two-peak function, then increase the number of knots used to approximate $sin x$.
  Controls: preset · hidden-unit count · hinge location · output weight · show components.
]
#pause
#align(center, text(size: 17pt)[Predict which local slope will change *before* moving a hinge or its output weight.])

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

== “Within $epsilon$” means close at every input #V

#grid(
  columns: (1fr, 1fr), column-gutter: 18pt,
  [
    #align(center, lines(
      fn: (x => calc.sin(x), x => _sin-pwl(x, knots: 13)),
      domain: (-3.14, 3.14), samples: 220,
      labels: ([$f(x)$: target], [$g(x)$: network]),
      colors: (INK, ACC), dashes: ("solid", "dashed"), markers: (false, false),
      legend: "tl", x-label: [$x$], y-label: [output],
      title: [target and approximation], size: (84mm, 49mm),
    ))
  ],
  [
    #pause
    #align(center, lines(
      fn: (x => calc.abs(calc.sin(x) - _sin-pwl(x, knots: 13)), x => 0.05),
      domain: (-3.14, 3.14), samples: 220,
      labels: ([$abs(f(x)-g(x))$], [$epsilon=0.05$]),
      colors: (ACC, RED), dashes: ("solid", "dashed"), markers: (false, false),
      legend: "tr", x-label: [$x$], y-label: [error],
      title: [pointwise error], size: (84mm, 49mm),
    ))
  ],
)
#pause
#align(center, text(size: 17pt, fill: MUTED)[
  $sup_(x in D) abs(f(x)-g(x))$ is the *largest vertical gap* anywhere on $D$.
])
#pause
#place(bottom + center, dy: -2pt,
  result[The guarantee is uniform: no input in the interval may exceed the chosen tolerance.])

== More hinges reduce the largest approximation error #V

#grid(
  columns: (1fr, 1fr, 1fr), column-gutter: 12pt,
  [
    #align(center, [
      #text(size: 17pt, weight: 650)[4 knots]
      #v(5pt)
      #lines(
        fn: (x => calc.sin(x), x => _sin-pwl(x, knots: 4)),
        domain: (-3.14, 3.14), samples: 180,
        markers: (false, false), colors: (INK, ACC), dashes: ("solid", "dashed"),
        y-ticks: false, x-label: [$x$], size: (51mm, 36mm),
      )
      #v(5pt)
      #text(size: 15pt, fill: MUTED)[largest error $approx 0.44$]
    ])
  ],
  [
    #pause
    #align(center, [
      #text(size: 17pt, weight: 650)[7 knots]
      #v(5pt)
      #lines(
        fn: (x => calc.sin(x), x => _sin-pwl(x, knots: 7)),
        domain: (-3.14, 3.14), samples: 180,
        markers: (false, false), colors: (INK, ACC), dashes: ("solid", "dashed"),
        y-ticks: false, x-label: [$x$], size: (51mm, 36mm),
      )
      #v(5pt)
      #text(size: 15pt, fill: MUTED)[largest error $approx 0.13$]
    ])
  ],
  [
    #pause
    #align(center, [
      #text(size: 17pt, weight: 650)[13 knots]
      #v(5pt)
      #lines(
        fn: (x => calc.sin(x), x => _sin-pwl(x, knots: 13)),
        domain: (-3.14, 3.14), samples: 180,
        markers: (false, false), colors: (INK, ACC), dashes: ("solid", "dashed"),
        y-ticks: false, x-label: [$x$], size: (51mm, 36mm),
      )
      #v(5pt)
      #text(size: 15pt, fill: MUTED)[largest error $approx 0.033$]
    ])
  ],
)
#pause
#place(bottom + center, dy: -2pt,
  result[More hidden units provide more breakpoints, so a piecewise-linear network can follow finer detail.])

== Read the universal-approximation claim in order #D

#two(r: (1.08fr, 0.92fr),
  align(center, grid(
    columns: (1fr, 9mm, 1fr), row-gutter: 6pt,
    box(width: 100%, inset: 8pt, fill: rgb("#EFEEEB"), stroke: 0.9pt + INK,
      align(center, text(size: 16pt)[#text(fill: ACC, weight: 700)[1] choose $f$ and $D$])),
    align(center, $arrow.r$),
    box(width: 100%, inset: 8pt, fill: rgb("#FDECD6"), stroke: 0.9pt + ACC,
      align(center, text(size: 16pt)[#text(fill: ACC, weight: 700)[2] choose $epsilon > 0$])),
    [], [], align(center, $arrow.b$),
    box(width: 100%, inset: 8pt, fill: rgb("#E8F4EA"), stroke: 0.9pt + GREEN,
      align(center, text(size: 16pt)[#text(fill: ACC, weight: 700)[4] gap is below $epsilon$])),
    align(center, $arrow.l$),
    box(width: 100%, inset: 8pt, fill: rgb("#E3EDED"), stroke: 0.9pt + TEAL,
      align(center, text(size: 16pt)[#text(fill: ACC, weight: 700)[3] some finite MLP $g$ exists])),
  )),
  [
    *The order matters.*
    #pause
    #v(8pt)
    *For every* $epsilon$: we may demand any positive tolerance.
    #pause
    #v(8pt)
    *There exists* $g$: the network can change when we demand a smaller error.
    #pause
    #v(10pt)
    $ forall epsilon > 0, thick exists thin "MLP" g $
    $ "such that" quad sup_(x in D) abs(f(x)-g(x)) < epsilon. $
  ],
)
#pause
#place(bottom + center, dy: -2pt,
  result[This is a statement about representational capacity—not yet about how to find the network.])

== Existence does not guarantee learnability #OPT

Universal approximation does *not* promise:
#pause
- that training will *find* that function;
- that *finite data* is enough;
#pause
- that a *small* network suffices;
- that it will *generalize*.
#pause
#result[expressivity $eq.not$ trainability $eq.not$ generalization]

== Composition means feeding one result into the next #V

Start with two small functions:
$ f_1(x) = 2x + 1, quad f_2(u) = u^2. $
#pause
#v(8pt)
#align(center, diagram(spacing: (13mm, 10mm), node-stroke: 0.9pt, {
  edge((0, 0), (1, 0), "-|>", stroke: 0.9pt + MUTED)
  edge((1, 0), (2, 0), "-|>", stroke: 0.9pt + MUTED)
  edge((2, 0), (3, 0), "-|>", stroke: 0.9pt + MUTED)
  edge((3, 0), (4, 0), "-|>", stroke: 0.9pt + MUTED)

  node((0, 0), text(fill: white)[$x=1$], radius: 7mm, fill: INK, stroke: none)
  node((1, 0), [$f_1(x)=2x+1$], shape: fletcher.shapes.rect,
    fill: TEAL.lighten(84%), stroke: TEAL, inset: 8pt)
  node((2, 0), text(fill: white)[$u=3$], radius: 7mm, fill: TEAL, stroke: none)
  node((3, 0), [$f_2(u)=u^2$], shape: fletcher.shapes.rect,
    fill: ACC.lighten(86%), stroke: ACC, inset: 8pt)
  node((4, 0), text(fill: white)[$y=9$], radius: 7mm, fill: ACC, stroke: none)
}))
#pause
#v(10pt)
#align(center, text(size: 23pt)[
  $y = f_2(f_1(x)) = (2x+1)^2$
])
#pause
#notebox[The value $u$ is an *intermediate feature*: the second stage uses what the first stage computed.]
#pause
#place(bottom + center, dy: -2pt,
  result[Composition is sequential reuse: $f = f_2 compose f_1$.])

== A deep model can build features in stages #V

Consider recognizing a handwritten *8* from an image:
#pause
#v(11pt)
#align(center, diagram(spacing: (18mm, 12mm), node-stroke: 0.9pt, {
  edge((0, 0), (1, 0), "-|>", stroke: 0.9pt + MUTED)
  edge((1, 0), (2, 0), "-|>", stroke: 0.9pt + MUTED)
  edge((2, 0), (3, 0), "-|>", stroke: 0.9pt + MUTED)

  node((0, 0), align(center)[
    *PIXELS* #linebreak()
    #text(size: 15pt, fill: MUTED)[28 $times$ 28 intensities]
  ], shape: fletcher.shapes.rect, fill: INK.lighten(91%), stroke: INK, inset: 10pt)
  node((1, 0), align(center)[
    *STROKES* #linebreak()
    #text(size: 15pt, fill: MUTED)[curves and short edges]
  ], shape: fletcher.shapes.rect, fill: TEAL.lighten(86%), stroke: TEAL, inset: 10pt)
  node((2, 0), align(center)[
    *PARTS* #linebreak()
    #text(size: 15pt, fill: MUTED)[upper and lower loops]
  ], shape: fletcher.shapes.rect, fill: ACC.lighten(87%), stroke: ACC, inset: 10pt)
  node((3, 0), align(center)[
    *CLASS* #linebreak()
    #text(size: 23pt, weight: 750, fill: ACC)[8]
  ], shape: fletcher.shapes.rect, fill: ACC.lighten(92%), stroke: ACC, inset: 10pt)
}))
#pause
#v(12pt)
#grid(
  columns: (1fr, 1fr, 1fr), column-gutter: 12pt,
  notebox[*Layer 1* detects local contrast.],
  notebox[*Layer 2* combines strokes into loops.],
  notebox[*Output* combines parts into a class score.],
)
#pause
#place(bottom + center, dy: -2pt,
  result[Later features are functions of earlier features: depth mirrors a hierarchy.])

== Depth can match a compositional problem #V

Suppose the target itself combines eight inputs in stages:
#pause
#align(center, text(size: 15pt, weight: 650, fill: MUTED)[
  INPUTS $arrow.r$ PAIR FEATURES $arrow.r$ GROUP FEATURES $arrow.r$ OUTPUT
])
#v(4pt)
#align(center, diagram(spacing: (18mm, 9mm), node-stroke: 0.8pt, {
  // Connections first so they stay behind the nodes.
  for i in range(4) {
    edge((0, i), (1, i), "-|>", stroke: 0.65pt + MUTED)
  }
  for i in range(4) {
    edge((1, i), (2, calc.floor(i / 2)), "-|>", stroke: 0.8pt + TEAL)
  }
  edge((2, 0), (3, 0), "-|>", stroke: 0.9pt + ACC)
  edge((2, 1), (3, 0), "-|>", stroke: 0.9pt + ACC)

  node((0, 0), [$(x_1,x_2)$], shape: fletcher.shapes.rect,
    fill: INK.lighten(93%), stroke: INK, inset: 6pt)
  node((0, 1), [$(x_3,x_4)$], shape: fletcher.shapes.rect,
    fill: INK.lighten(93%), stroke: INK, inset: 6pt)
  node((0, 2), [$(x_5,x_6)$], shape: fletcher.shapes.rect,
    fill: INK.lighten(93%), stroke: INK, inset: 6pt)
  node((0, 3), [$(x_7,x_8)$], shape: fletcher.shapes.rect,
    fill: INK.lighten(93%), stroke: INK, inset: 6pt)
  for i in range(4) {
    node((1, i), [$h_#(i+1)$], radius: 4.7mm, fill: TEAL.lighten(82%), stroke: TEAL)
  }
  for i in range(2) {
    node((2, i), [$q_#(i+1)$], radius: 5.2mm, fill: ACC.lighten(86%), stroke: ACC)
  }
  node((3, 0), text(fill: white)[$y$], radius: 6mm, fill: ACC, stroke: none)
}))
#pause
#place(bottom + center, dy: -2pt,
  result[Each node combines two values, yet $y$ depends on all eight. For some compositional functions, a shallow network needs far more units.])

// ═══════════════════════════ PART VIII — Connecting to practice ═══════════════════════════
= Connecting to deep-learning practice

== A forward pass transforms one input, step by step #V

#align(center, text(size: 18pt)[Example: classify one $28 times 28$ image — $784$ pixel values.])
#v(8pt)
#pause
#let pass-card(title, formula, note, color, width: 39mm) = block(
  width: width, height: 46mm, inset: (x: 6pt, y: 7pt),
  fill: color.lighten(91%), stroke: 0.9pt + color, radius: 4pt,
  align(center)[
    #text(size: 13pt, weight: "bold", fill: color)[#title]
    #v(4pt)
    #text(size: 14pt)[#formula]
    #v(3pt)
    #text(size: 10.5pt, fill: MUTED)[#note]
  ],
)
#align(center, grid(
  columns: (31mm, 7mm, 50mm, 7mm, 30mm, 7mm, 39mm, 7mm, 38mm),
  align: horizon,
  pass-card([INPUT], [$bold(x) in RR^784$], [pixel values], INK, width: 31mm),
  align(center, text(size: 22pt, fill: MUTED)[$arrow.r$]),
  pass-card([HIDDEN LAYER], [$bold(u)=bold(W)_1 bold(x)+bold(b)_1$ \ $bold(h)="ReLU"(bold(u))$], [128 learned features], TEAL, width: 50mm),
  align(center, text(size: 22pt, fill: MUTED)[$arrow.r$]),
  pass-card([LOGITS], [$bold(z)$], [10 class scores], ACC, width: 30mm),
  align(center, text(size: 22pt, fill: MUTED)[$arrow.r$]),
  pass-card([SOFTMAX], [$bold(p)$ \ $= "softmax"(bold(z))$], [10 probabilities], TEAL, width: 39mm),
  align(center, text(size: 22pt, fill: MUTED)[$arrow.r$]),
  pass-card([LOSS], [$cal(L) = -log p_y$], [one scalar], ACC, width: 38mm),
))
#pause
#v(9pt)
#align(center, text(size: 16pt, fill: MUTED)[
  The true class $y$ selects its predicted probability $p_y$; a smaller $p_y$ gives a larger loss.
])
#pause
#place(bottom + center, dy: -2pt,
  result[A forward pass computes a prediction and its loss; it does *not* update the weights.])

== Count parameters by following the same network #D

#align(center, text(size: 17pt)[Every connection has a *weight*; every receiving unit has a *bias*.])
#v(7pt)
#align(center, grid(
  columns: (31mm, 10mm, 49mm, 10mm, 35mm, 10mm, 45mm, 10mm, 30mm),
  align: horizon,
  block(width: 31mm, inset: 7pt, fill: INK.lighten(92%), stroke: 0.9pt + INK, radius: 4pt,
    align(center)[*INPUT* \ #text(size: 21pt)[$784$] \ #text(size: 11pt, fill: MUTED)[pixels]]),
  align(center, text(size: 22pt, fill: MUTED)[$arrow.r$]),
  block(width: 49mm, inset: 7pt, fill: TEAL.lighten(91%), stroke: 0.9pt + TEAL, radius: 4pt,
    align(center)[*$bold(W)_1, bold(b)_1$* \ #text(size: 13pt)[$128 times 784$ weights \ $+128$ biases]]),
  align(center, text(size: 22pt, fill: MUTED)[$arrow.r$]),
  block(width: 35mm, inset: 7pt, fill: TEAL.lighten(84%), stroke: 0.9pt + TEAL, radius: 4pt,
    align(center)[*HIDDEN* \ #text(size: 21pt)[$128$] \ #text(size: 11pt, fill: MUTED)[units]]),
  align(center, text(size: 22pt, fill: MUTED)[$arrow.r$]),
  block(width: 45mm, inset: 7pt, fill: ACC.lighten(91%), stroke: 0.9pt + ACC, radius: 4pt,
    align(center)[*$bold(W)_2, bold(b)_2$* \ #text(size: 13pt)[$10 times 128$ weights \ $+10$ biases]]),
  align(center, text(size: 22pt, fill: MUTED)[$arrow.r$]),
  block(width: 30mm, inset: 7pt, fill: ACC.lighten(84%), stroke: 0.9pt + ACC, radius: 4pt,
    align(center)[*OUTPUT* \ #text(size: 21pt)[$10$] \ #text(size: 11pt, fill: MUTED)[classes]]),
))
#pause
#v(2pt)
#two(r: (1fr, 1fr),
  notebox[
    *Input $arrow.r$ hidden* \
    weights: $128 times 784 = "100,352"$ \
    biases: $128$ \
    *subtotal: $"100,480"$*
  ],
  notebox[
    *Hidden $arrow.r$ output* \
    weights: $10 times 128 = "1,280"$ \
    biases: $10$ \
    *subtotal: $"1,290"$*
  ],
)
#pause
#v(0pt)
#align(center, text(size: 15pt, fill: MUTED)[Layer $ell$: $d_ell d_(ell-1)$ weights $+$ $d_ell$ biases.])
#pause
#place(bottom + center, dy: -2pt,
  result[total $= "100,480" + "1,290" = "101,770"$ trainable parameters])

== From one weighted sum to an MLP #V

#align(center, diagram(spacing: (13mm, 8mm), node-stroke: 0.9pt, {
  for i in range(3) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.9pt + MUTED) }

  node((0, 0), text(size: 12.5pt)[#align(center)[
    *WEIGHTED SUM* \
    $z = bold(w)^top bold(x) + b$
  ]], shape: fletcher.shapes.rect, fill: INK.lighten(93%), stroke: INK, inset: 8pt)

  node((1, 0), text(size: 12.5pt)[#align(center)[
    *ONE NEURON* \
    $a = phi(z)$
  ]], shape: fletcher.shapes.rect, fill: TEAL.lighten(91%), stroke: TEAL, inset: 8pt)

  node((2, 0), text(size: 12.5pt)[#align(center)[
    *HIDDEN LAYER* \
    $bold(h) = phi(bold(W) bold(x) + bold(b))$
  ]], shape: fletcher.shapes.rect, fill: ACC.lighten(92%), stroke: ACC, inset: 8pt)

  node((3, 0), text(size: 12.5pt)[#align(center)[
    *DEEP MLP* \
    $bold(h)^((1)) arrow.r bold(h)^((2)) arrow.r dots.h arrow.r bold(z)$
  ]], shape: fletcher.shapes.rect, fill: ACC.lighten(84%), stroke: ACC, inset: 8pt)
}))
#pause
#v(9pt)
#align(center, {
  set text(size: 13.5pt)
  table(
    columns: (38mm, 73mm, 111mm), stroke: 0.5pt + MUTED,
    inset: (x: 9pt, y: 5.5pt), align: (left, left, left),
    table.header([*Stage*], [*Representation*], [*What it adds*]),
    [Weighted sum], [$z = bold(w)^top bold(x) + b$], [One line or plane; captures a global linear trend.],
    [One neuron], [$a = phi(z)$], [One nonlinear feature: a hinge, gate, or score.],
    [Hidden layer], [$bold(h) = phi(bold(W)bold(x)+bold(b))$], [Many learned features can bend or partition the input space.],
    [Deep MLP], [$bold(h)^((ell)) = phi(bold(W)_ell bold(h)^((ell-1)) + bold(b)_ell)$], [Layers compose features into a hierarchical representation.],
  )
})
#pause
#place(bottom + center, dy: -2pt,
  result[The key addition is not a new loss; it is a learned nonlinear representation.])

== The decisive change is the learned representation

#align(center, text(size: 20pt)[
  *Input* $bold(x)$ \
  $arrow.b$ \
  *learned nonlinear features* $bold(h)^((1)), dots, bold(h)^((L-1))$ \
  $arrow.b$ \
  *familiar Lecture 1 output model and loss*
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
