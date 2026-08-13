// Making Deep Networks Trainable — Lecture 6 · ES 667 Deep Learning
// Compile from the repository root:
//   typst compile --root . --input handout=true lecture6/L6-trainability.typ /tmp/L6-handout.pdf
//   typst compile --root . lecture6/L6-trainability.typ /tmp/L6-presentation.pdf

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [Making Deep Networks Trainable],
  subtitle: [Activations → initialization → normalization → residual connections],
)

#let NB = "https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L06/"
#let RESNET = "https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html"

// Semantic grammar: TEAL = forward signal · BLUE = backward gradient
// ACC = scale/gate being chosen · GREEN = preserved/identity path · RED = failure.
#let note(body, color: TEAL) = block(
  width: 100%, inset: (left: 10pt, right: 2pt, top: 4pt, bottom: 4pt),
  stroke: (left: 1.2pt + color), [#body],
)
#let caption(body) = align(center, text(size: 14.5pt, fill: MUTED)[#body])
#let punch(body) = align(center, text(size: 19pt, weight: 650, fill: INK)[#body])
#let hairline(title, body, color: TEAL) = block(width: 100%, inset: 2pt, [
  #text(size: 14pt, weight: 700, fill: color)[#title]
  #v(2pt)
  #line(length: 100%, stroke: 0.7pt + color)
  #v(5pt)
  #body
])
#let swatch(color, body) = box(width: 12pt, height: 3pt, fill: color, radius: 1pt) + h(4pt) + body

#let semantic-legend = align(center, text(size: 13pt)[
  #swatch(TEAL, [forward signal]) #h(16pt)
  #swatch(BLUE, [backward gradient]) #h(16pt)
  #swatch(ACC, [chosen gain / reset]) #h(16pt)
  #swatch(GREEN, [preserved / identity]) #h(16pt)
  #swatch(RED, [failure])
])

#let anchor = block(
  width: 100%, fill: rgb("#F3F6F5"), stroke: (left: 3pt + TEAL),
  inset: (x: 11pt, y: 6pt), radius: 3pt,
  text(size: 14.5pt)[
    *ANCHOR* · five width-$4$ layers · ReLU after every affine map ·
    #text(fill: TEAL)[$q_0=1$] · #text(fill: BLUE)[$g_5=1$]
  ],
)

#let chain = align(center, diagram(
  spacing: (18mm, 10mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    for ell in range(6) {
      let col = if ell == 0 { TEAL } else if ell == 5 { BLUE } else { INK }
      node((ell, 0), [$h_#ell$], radius: 6.2mm, stroke: 1pt + col)
    }
    for ell in range(5) {
      edge((ell, 0), (ell + 1, 0), "-|>", stroke: 0.9pt + MUTED,
        label: text(size: 10pt)[$W_#(ell+1)$ + ReLU], label-sep: 4pt)
    }
    edge((0, 1.25), (5, 1.25), "-|>", stroke: 1.25pt + TEAL,
      label: text(size: 11pt, fill: TEAL, weight: 650)[signal], label-side: left)
    edge((5, 2.25), (0, 2.25), "-|>", stroke: 1.25pt + BLUE,
      label: text(size: 11pt, fill: BLUE, weight: 650)[gradient], label-side: left)
  },
))

#let causal-map = align(center, diagram(
  spacing: (25mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    let labels = ([activation gate], [matched init], [normalization], [residual path])
    for (i, label) in labels.enumerate() {
      node((i, 0), label, shape: fletcher.shapes.rect, corner-radius: 3pt,
        inset: 8pt, stroke: 1.1pt + if i == 3 { GREEN } else { ACC })
    }
    for i in range(3) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.9pt + MUTED) }
  },
))

#let residual-block = align(center, diagram(
  spacing: (18mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    node((0, 0), $x_ell$, radius: 6.5mm)
    node((1.4, 0), $F_ell$, shape: fletcher.shapes.rect, corner-radius: 3pt,
      inset: 9pt, fill: rgb("#FDECD6"), stroke: 1pt + ACC)
    node((2.8, 0), $+$, radius: 5.5mm)
    node((4.0, 0), $x_(ell+1)$, radius: 7mm, stroke: 1pt + TEAL)
    edge((0, 0), (1.4, 0), "-|>", stroke: 1pt + INK)
    edge((1.4, 0), (2.8, 0), "-|>", stroke: 1pt + ACC)
    edge((2.8, 0), (4.0, 0), "-|>", stroke: 1pt + TEAL)
    edge((0, 0), (2.8, 0), "-|>", bend: -48deg, stroke: 1.3pt + GREEN,
      label: text(size: 11pt, fill: GREEN, weight: 650)[identity $I$], label-side: left)
  },
))

#let residual-backward = align(center, diagram(
  spacing: (24mm, 13mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    node((3.2, 0), $nabla_"up"$, radius: 8mm, stroke: 1.1pt + BLUE)
    node((1.7, -0.75), [$I$], shape: fletcher.shapes.rect, corner-radius: 3pt,
      inset: 7pt, stroke: 1.1pt + GREEN)
    node((1.7, 0.75), $J_(F_ell)^T$, shape: fletcher.shapes.rect, corner-radius: 3pt,
      inset: 7pt, stroke: 1.1pt + ACC)
    node((0.5, 0), $+$, radius: 5.5mm)
    node((-0.7, 0), $nabla_(x_ell) cal(L)$, radius: 9mm, stroke: 1.1pt + BLUE)
    edge((3.2, 0), (1.7, -0.75), "-|>", stroke: 1.2pt + GREEN)
    edge((3.2, 0), (1.7, 0.75), "-|>", stroke: 1.1pt + ACC)
    edge((1.7, -0.75), (0.5, 0), "-|>", stroke: 1.2pt + GREEN,
      label: text(size: 10.5pt, fill: GREEN)[$nabla_"up"$], label-side: right)
    edge((1.7, 0.75), (0.5, 0), "-|>", stroke: 1.1pt + ACC,
      label: text(size: 10.5pt, fill: ACC)[$J_F^T nabla_"up"$], label-side: left)
    edge((0.5, 0), (-0.7, 0), "-|>", stroke: 1.1pt + BLUE)
  },
))

#let axes-grid(mode) = {
  let B = 4
  let D = 5
  let cell(r, c) = {
    let selected = if mode == "batch" { c == 2 } else { r == 1 }
    circle(radius: 5pt, fill: if selected { ACC } else { rgb("#C9D4D4") }, stroke: none)
  }
  stack(dir: ttb, spacing: 6pt,
    ..range(B).map(r => stack(dir: ltr, spacing: 6pt, ..range(D).map(c => cell(r, c)))))
}

#let ledger(activation: [pending], init: [pending], norm: [pending], residual: [pending]) = block(
  width: 100%, fill: rgb("#F8F8F6"), inset: 8pt, radius: 3pt,
  stroke: 0.6pt + rgb("#D7DDDC"), [
    #text(size: 11pt, weight: 700, fill: MUTED, tracking: 0.8pt)[GAUGE LEDGER · SAME FIVE-LAYER ANCHOR]
    #v(4pt)
    #set text(size: 13.2pt)
    #table(
      columns: (25%, 75%), stroke: 0.35pt + rgb("#D7DDDC"), inset: (x: 7pt, y: 4pt),
      align: (left, left),
      table.header([control], [evidence / status · squared $q,g$; RMS $r,s$ where shown]),
      [activation], activation,
      [initialization], init,
      [normalization], norm,
      [residual path], residual,
    )
  ],
)

#title-slide()

// ═══════════════════════════ OPENING · CORE ═══════════════════════════
== Backprop found the gradients; depth may erase them

#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([L4 · BACKPROP], [computes $nabla_theta cal(L)$], color: BLUE),
  hairline([L5 · OPTIMIZATION], [chooses the parameter update], color: ACC),
  hairline([L6 · TRAINABILITY], [asks whether signals and gradients survive depth], color: TEAL),
))

#pause
#v(10pt)
#chain

#pause
#result[Today the computation rule stays fixed; we redesign how information travels through it.]

== Commit: which start survives five ReLU layers? #Q

#anchor
#v(8pt)

#mcq(
  [Which zero-mean weight distribution should keep both end-to-end RMS gauges closest to $1$ at initialization?],
  [Xavier normal with $"Var"(w)=1/4$],
  [Xavier uniform with the same variance $1/4$],
  [He normal with $"Var"(w)=1/2$],
  [All weights equal to zero],
)

== Keep your bet; every section updates one ledger

#chain

#pause
#align(center, text(size: 18pt)[We will calculate the same chain, then add one control at a time:])
#v(8pt)
#causal-map

#pause
#note(color: ACC)[Do not change your answer yet. We return to it after the activation gate determines the matching initialization.]

== Outline: keep signals usable through depth #V

#causal-map

#v(10pt)
#punch[Predict the drift → measure both RMS gauges → change the earliest plausible cause → measure again.]

#v(9pt)
#semantic-legend

== One gauge, two directions, no width drift #V

For width $d_ell$, average over coordinates—not a raw vector norm:

$ q_ell := 1/d_ell sum_i (h_(ell,i))^2, quad r_ell:=sqrt(q_ell) $
$ g_ell := 1/d_ell sum_i (overline(h)_(ell,i))^2, quad s_ell:=sqrt(g_ell). $

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([#text(fill: TEAL)[FORWARD]], [$r_ell$ is activation RMS; $q_ell$ is its squared ledger], color: TEAL),
  hairline([#text(fill: BLUE)[BACKWARD]], [$s_ell$ is gradient RMS; $g_ell$ is its squared ledger], color: BLUE),
))

#pause
#anchor

== A harmless local factor becomes a depth failure

If one layer multiplies a *squared* gauge by $c$, then five layers multiply it by $c^5$.

#pause
#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 7pt), align: center,
  table.header([per-layer $c$], [$c^1$], [$c^5$], [RMS factor $sqrt(c^5)$]),
  [$0.5$], [$0.5$], [$0.03125$], [$0.17678$],
  [$1$], [$1$], [$1$], [$1$],
  [$2$], [$2$], [$32$], [$5.65685$],
))

#pause
#result[Depth turns a small scale mismatch into a visible vanishing or exploding trace.]

== The repair order follows causality #V

#causal-map
#v(10pt)

#align(center, table(
  columns: 2, stroke: 0.45pt + MUTED, inset: (x: 13pt, y: 6pt), align: (left, left),
  [activation], [sets the local nonlinear gate],
  [initialization], [matches weight gain to that gate at $t=0$],
  [normalization], [re-establishes a reference scale while training],
  [residual connection], [adds a short signal and gradient route],
))

#pause
#punch[Activation comes first because its gate determines the scale that initialization must match.]

// ═══════════════════════════ ACTIVATIONS · CORE ═══════════════════════════
= The activation is a gradient gate

== Every nonlinearity acts in both passes #V

#align(center, diagram(spacing: (23mm, 12mm), node-stroke: 1pt + INK, node-fill: white, {
  node((0, 0), $u_ell$, radius: 7mm, stroke: 1pt + TEAL)
  node((1, 0), $phi$, radius: 7mm, fill: rgb("#FDECD6"), stroke: 1pt + ACC)
  node((2, 0), $h_ell$, radius: 7mm, stroke: 1pt + TEAL)
  edge((0, 0), (1, 0), "-|>", stroke: 1pt + TEAL)
  edge((1, 0), (2, 0), "-|>", stroke: 1pt + TEAL, label: [$h_ell=phi(u_ell)$])
  edge((2, 1.25), (0, 1.25), "-|>", stroke: 1.25pt + BLUE,
    label: text(fill: BLUE)[$overline(u)_ell=phi'(u_ell)overline(h)_ell$], label-side: left)
}))

#pause
#result[The curve shapes the representation; its derivative gates the returning gradient.]

== Sigmoid becomes nearly flat in either tail

$ sigma(u)=1/(1+e^(-u)), quad sigma'(u)=sigma(u)(1-sigma(u)). $

#pause
#align(center, table(
  columns: 5, stroke: 0.45pt + MUTED, inset: (x: 11pt, y: 6pt), align: center,
  table.header([$u$], [$-5$], [$0$], [$5$], [meaning]),
  [$sigma(u)$], [$0.0067$], [$0.5$], [$0.9933$], [bounded output],
  [$sigma'(u)$], [$0.00665$], [$0.25$], [$0.00665$], [gradient gate],
))

#pause
#note(color: RED)[Even its best slope is only $1/4$; saturation makes the slope far smaller.]

== One saturated sigmoid blocks more than 99% #D

Take $u=5$ and upstream gradient $overline(h)=1$.

#pause
$ h=sigma(5) approx 0.9933 $
#pause
$ sigma'(5)=h(1-h) approx 0.00665 $
#pause
$ overline(u)=overline(h)sigma'(u)=1(0.00665)=0.00665. $

#pause
#result[Only $0.665%$ of the arriving gradient magnitude crosses this activation.]

== ReLU keeps one side linear

$ "ReLU"(u)=max(0,u), quad "ReLU"'(u)=cases(0 & u<0, 1 & u>0). $

#pause
#align(center, table(
  columns: 4, stroke: 0.45pt + MUTED, inset: (x: 13pt, y: 7pt), align: center,
  table.header([$u$], [$overline(h)$], [$h$], [$overline(u)$]),
  [$2.3$], [$0.7$], [$2.3$], [$0.7$],
  [$-2.3$], [$0.7$], [$0$], [$0$],
))

#pause
#result[ReLU passes the active-side gradient unchanged and blocks the inactive side.]

== The anchor is half active in expectation #D

#anchor
#v(6pt)

For a symmetric zero-mean pre-activation $z$, positive and negative values are equally likely:

$ EE["ReLU"(z)^2] approx 1/2 EE[z^2]. $

#pause
The same Bernoulli gate appears in the backward squared gauge:

$ EE[("ReLU"'(z)overline(h))^2] approx 1/2 EE[overline(h)^2]. $

#pause
#punch[The gate contributes $1/2$ to both squared-gauge multipliers.]

== “Dead ReLU” means inactive on every current example

A unit is dead on the current dataset if

$ u_i=w^T x_i+b<0 quad "for every example" i. $

#pause
Then its incoming parameter gradients are zero because every $"ReLU"'(u_i)=0$.

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([SUSPECT], [large negative bias, overly large update, or poor input scale], color: RED),
  hairline([TEST], [count examples active for each unit—not only the global zero fraction], color: BLUE),
))

#pause
#note[Changing upstream representations may revive it; “can never recover” is too strong.]

== Compare the gate, not only the curve #V

#fig("/lecture6/figures/activation_zoo.svg", w: 73%)

#caption[Solid: $phi(u)$. Dashed: $phi'(u)$, the returning-gradient gate. GELU uses the documented sigmoid approximation $u sigma(1.702u)$; the other curves are exact away from their kink.]

#pause
#result[Choose the activation for representation; choose initialization to match its gain.]

== Ledger 1 — the gate is now known #V

#ledger(
  activation: [$c_"gate"=1/2$ in both squared gauges],
  init: [not chosen; weight gain still unknown],
  norm: [not yet added],
  residual: [not yet added],
)

#pause
#anchor

// ═══════════════════════════ INITIALIZATION · CORE ═══════════════════════════
= Initialization must match the gate

== Fan-in gives the reproducible scale derivation #D

For one pre-activation $z=sum_(i=1)^n w_i x_i$, assume independent, zero-mean weights and inputs.

#pause
$ EE[z^2] = EE[(sum_i w_i x_i)^2] $
#pause
$ = sum_i EE[w_i^2]EE[x_i^2] + underbrace(sum_(i != j) EE[w_i]EE[w_j]EE[x_i x_j], 0) $
#pause
$ = n_"in" thin "Var"(w) thin EE[x^2]. $

#pause
#result[An affine layer multiplies the squared signal gauge by $n_"in" "Var"(w)$.]

== The gate tells us which weight variance to choose

#anchor
#v(6pt)

For width $4$ followed by ReLU,

$ c = underbrace(4 "Var"(w), "affine gain") times underbrace(1/2, "ReLU gate") = 2 "Var"(w). $

#pause
#align(center, table(
  columns: 4, stroke: 0.45pt + MUTED, inset: (x: 13pt, y: 7pt), align: center,
  table.header([scheme], [$"Var"(w)$], [one-layer $c$], [five-layer $c^5$]),
  [Xavier], [$1/4$], [$0.5$], [$0.03125$],
  [He], [$2/4=1/2$], [$1$], [$1$],
))

#pause
#punch[The same half-active gate that lost energy supplies the missing factor of $2$.]

== Answer: He keeps the anchor gauges near one #A

#mcq-answer(
  [C],
  [He normal with $"Var"(w)=1/2$],
  [For width $4$, affine gain $4(1/2)=2$ and the ReLU gate contributes $1/2$, so the net squared-gauge factor is $c=1$.],
)

#pause
#note(color: ACC)[Xavier normal and Xavier uniform share a variance, so both give $c=0.5$ here. Their samples are variance-matched alternatives, not identical distributions.]

== Xavier loses half the squared gauge per ReLU layer #D

#anchor
#v(6pt)

With $"Var"(w)=1/4$,

$ q_5=(0.5)^5 q_0=0.03125, quad g_0=(0.5)^5 g_5=0.03125. $

#pause
Read the same ledger in the primary RMS gauge:

$ r_5=sqrt(q_5)=0.17678, quad s_0=sqrt(g_0)=0.17678. $

#pause
#result[Xavier preserves the affine pre-activation scale, but ReLU then removes half the second moment.]

== He preserves the second moment through ReLU #D

#anchor
#v(6pt)

With $"Var"(w)=2/n_"in"=1/2$,

$ c=4(1/2)(1/2)=1. $

#pause
$ q_5=1^5q_0=1, quad g_0=1^5g_5=1, quad r_5=s_0=1. $

#pause
#result[Under the mean-field assumptions, He preserves activation and gradient RMS through the ReLU stack.]

== The computed trace stays on one honest scale #V

#anchor
#v(6pt)

#align(center, table(
  columns: 6, stroke: 0.45pt + MUTED, inset: (x: 10pt, y: 6pt), align: center,
  table.header([squared gauge], [$ell=0$], [$1$], [$2$], [$3$], [$5$]),
  [Xavier $0.5^ell$], [$1$], [$0.5$], [$0.25$], [$0.125$], [$0.03125$],
  [He $1^ell$], [$1$], [$1$], [$1$], [$1$], [$1$],
))

#pause
#caption[Both rows share the same exact $0$–$1$ scale; no trajectory falls below or leaves the shown range.]

== A seeded wider simulation shows the same scale story #V

#fig("/lecture6/figures/signal_flow.svg", w: 68%)

#caption[Constructed diagnostic, seed $0$: width $256$, batch $1024$, $40$ bias-free ReLU layers, Gaussian inputs. Each curve uses newly sampled weights; points are measured activation RMS.]

#pause
#note(color: GREEN)[He keeps the trace order-one in this finite-width draw; Xavier loses roughly one half of the squared gauge per ReLU layer, as predicted.]

== Xavier samplers match variance, not samples

For $n_"in"=n_"out"=4$, Xavier targets $"Var"(w)=2/(4+4)=1/4$.

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([NORMAL ALTERNATIVE], [$w ~ cal(N)(0,1/4)$ · unbounded · most mass near $0$], color: ACC),
  hairline([UNIFORM ALTERNATIVE], [$w ~ cal(U)(-sqrt(3/4),sqrt(3/4))$ · bounded · flat density], color: ACC),
))

#pause
They have the same mean and variance, so the second-moment calculation matches; higher moments and individual draws do not.

#pause
#result[“Variance-matched” is the useful equivalence; the distributions themselves are different.]

== Convolutions use the fan-in of one output

For a convolution,

$ n_"in"=k_h k_w C_"in". $

#pause
A $3 times 3$ kernel with $64$ input channels has

$ n_"in"=3 dot 3 dot 64=576, quad "Std"_"He"=sqrt(2/576) approx 0.0589. $

#pause
#note[Stride changes output resolution, not how many input values feed one output. A projection shortcut may use a $1 times 1$ convolution with stride $>1$ when it must downsample.]

== Initialization controls only the start

The derivation assumes independent coordinates, symmetric pre-activations, and unchanged weight statistics.

#pause
Training breaks these assumptions: correlations grow, weights move, activation means shift, and data are not idealized.

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([PROMISE], [a sensible RMS scale at $t=0$], color: GREEN),
  hairline([NOT A PROMISE], [exact preservation for every layer, example, or training step], color: RED),
))

#pause
#punch[Initialization earns a healthy start; measurement tells us whether it stays healthy.]

== Ledger 2 — the anchor starts balanced #V

#ledger(
  activation: [ReLU gate contributes $1/2$],
  init: [Xavier: $q_5=g_0=.03125$; He: $q_5=g_0=1$],
  norm: [not yet added],
  residual: [not yet added],
)

#pause
#caption[Primary RMS reading: Xavier $r_5=s_0=.17678$; He $r_5=s_0=1$.]

// ═══════════════════════════ NORMALIZATION · CORE ═══════════════════════════
= Normalization resets a reference scale

== A normalization layer defines its own measuring group

For values in one chosen group,

$ mu=1/N sum_i x_i, quad sigma^2=1/N sum_i (x_i-mu)^2 $
$ hat(x)_i=(x_i-mu)/sqrt(sigma^2+epsilon), quad y_i=gamma hat(x)_i+beta. $

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([RESET], [$hat(x)$ has mean near $0$ and variance near $1$ over that group], color: ACC),
  hairline([LEARN], [$gamma,beta$ restore any useful affine scale and shift], color: GREEN),
))

#pause
#result[The key design question is not “normalize?” but “which entries share one statistic?”]

== BatchNorm and LayerNorm choose different axes #V

For $X in RR^(B times D)$, rows are examples and columns are features.

#v(8pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 34pt,
  [#align(center, axes-grid("batch"))
   #v(8pt)
   #align(center, text(size: 16pt, weight: 650)[BatchNorm: down one feature])
   #caption[reduce over the batch axis $B$]],
  [#align(center, axes-grid("layer"))
   #v(8pt)
   #align(center, text(size: 16pt, weight: 650)[LayerNorm: across one example])
   #caption[reduce over that example's $D$ features]],
))

== Commit: normalize the slice before calculating #Q

For one BatchNorm feature across four examples,

$ x=[1,2,3,4], quad gamma=2, quad beta=1, quad epsilon approx 0. $

#v(8pt)
#mcq(
  [What are the batch mean and population variance used on this slide?],
  [$mu=2.5, sigma^2=1.25$],
  [$mu=2.5, sigma^2=5/3$],
  [$mu=10, sigma^2=5$],
  [$mu=0, sigma^2=1$],
)

== Answer: build the whole BatchNorm slice #A

$ mu=(1+2+3+4)/4=2.5. $
#pause
$ sigma^2=((1-2.5)^2+(2-2.5)^2+(3-2.5)^2+(4-2.5)^2)/4=1.25. $
#pause
$ hat(x)=(x-2.5)/sqrt(1.25) approx [-1.342,-0.447,0.447,1.342]. $
#pause
$ y=2hat(x)+1 approx [-1.683,0.106,1.894,3.683]. $

#pause
#result[Correct: *A* — the normalized slice has mean $0$, variance $1$, and RMS $1$ before $gamma,beta$.]

== Learnable scale is allowed to move the gauge

For the normalized slice,

$ 1/4 sum_i hat(x)_i^2 approx 1 quad ⇒ quad "RMS"(hat(x)) approx 1. $

#pause
After $y=2hat(x)+1$,

$ 1/4 sum_i y_i^2 approx 5 quad ⇒ quad "RMS"(y) approx sqrt(5). $

#pause
#note(color: ACC)[Normalization supplies a stable coordinate system; it does not force the learned output RMS to remain $1$.]

== BatchNorm inference must ignore the current batch

During training, BatchNorm uses the current batch and updates running estimates.

#pause
During inference, it uses frozen running mean and variance:

$ y=gamma (x-mu_"run")/sqrt(sigma_"run"^2+epsilon)+beta. $

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([ALLOWED], [inference may process one example or a batch for efficiency], color: GREEN),
  hairline([FORBIDDEN], [an example's prediction changing because current batch-mates changed], color: RED),
))

#pause
#caption[`model.eval()` selects running statistics; “inference is always unbatched” is not the rule.]

== LayerNorm avoids batch dependence

LayerNorm computes one mean and variance from each example's feature vector:

$ mu_i=1/D sum_j x_(i j), quad sigma_i^2=1/D sum_j (x_(i j)-mu_i)^2. $

#pause
#align(center, table(
  columns: 3, stroke: 0.45pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left, left),
  table.header([property], [BatchNorm], [LayerNorm]),
  [normalizes over], [examples per feature], [features per example],
  [depends on batch-mates?], [training: yes], [no],
  [train / inference stats], [batch / running], [same computation],
  [common home], [CNN feature channels], [sequence/token models],
))

#pause
#result[Choose the axis from the invariance you need, not from the layer's name.]

== Ledger 3 — the BN slice supplies a scale reset #V

#ledger(
  activation: [ReLU gate contributes $1/2$],
  init: [He gives $q_5=g_0=1$ at $t=0$],
  norm: [$[1,2,3,4]$: $mu=2.5$, $sigma^2=1.25$, $"RMS"(hat(x))=1$],
  residual: [not yet added],
)

#pause
#caption[The reset is defined over the selected axis; learned $gamma,beta$ may deliberately move the post-normalization RMS.]

// ═══════════════════════════ RESIDUALS · CORE ═══════════════════════════
= Residual connections shorten the route

== Greater depth can make the training error worse #V

In the original ResNet experiment on CIFAR-10, a 56-layer *plain* network had higher training error than a 20-layer plain network.

#pause
#align(center, diagram(spacing: (27mm, 13mm), node-stroke: 1pt + INK, node-fill: white, {
  node((0, 0), [20-layer plain], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 9pt, stroke: 1pt + TEAL)
  node((1, 0), [lower training error], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 9pt, stroke: 1pt + GREEN)
  node((0, 1), [56-layer plain], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 9pt, stroke: 1pt + RED)
  node((1, 1), [higher training error], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 9pt, stroke: 1pt + RED)
  edge((0, 0), (1, 0), "-|>", stroke: 1pt + GREEN)
  edge((0, 1), (1, 1), "-|>", stroke: 1pt + RED)
}))

#pause
#caption[#link(RESNET)[He et al., “Deep Residual Learning for Image Recognition,” CVPR 2016, Fig. 1].]

== Learning a correction makes “do nothing” easy

Reparameterize a desired map $H_ell$ as

$ F_ell(x):=H_ell(x)-x quad ⇒ quad H_ell(x)=x+F_ell(x). $

#pause
#residual-block

#pause
#result[If the useful transformation is close to identity, the branch only has to learn the small correction $F_ell$.]

== Column-gradient convention fixes the transpose #D

Let $J_(F_ell)=partial F_ell(x_ell)/partial x_ell$ and store gradients as column vectors.

#pause
$ x_(ell+1)=x_ell+F_ell(x_ell) $
#pause
$ nabla_(x_ell) cal(L)=(I+J_(F_ell))^T nabla_(x_(ell+1))cal(L) $
#pause
$ =underbrace(nabla_(x_(ell+1))cal(L), "identity contribution") + underbrace(J_(F_ell)^T nabla_(x_(ell+1))cal(L), "residual contribution"). $

#pause
#result[Backward applies the transposed local Jacobian; the two contributions add at the block input.]

== The arriving gradient splits into two contributions #V

#residual-backward

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([#text(fill: GREEN)[IDENTITY]], [$nabla_"up"$ arrives without a branch matrix multiply], color: GREEN),
  hairline([#text(fill: ACC)[RESIDUAL]], [$J_F^T nabla_"up"$ may reinforce or cancel it], color: ACC),
))

#pause
#punch[A shortcut adds a route; it does not guarantee that every total gradient is large.]

== Commit: compare five plain and residual blocks #Q

Let one scalar branch have $F'(x)=-0.1$ and let the top gradient magnitude be $1$.

#v(8pt)
#mcq(
  [After five blocks, which pair of input-gradient magnitudes is correct?],
  [plain $0.5$; residual $1.5$],
  [plain $10^(-5)$; residual $0.9^5$],
  [plain $0.1$; residual $0.9$],
  [plain $0$; residual $1$ exactly],
)

== Answer: the shortcut changes the local multiplier #A

#align(center, table(
  columns: 4, stroke: 0.45pt + MUTED, inset: (x: 12pt, y: 6pt), align: center,
  table.header([path], [one block], [five-block magnitude], [squared gauge]),
  [plain branch $F$], [$|-0.1|$], [$|-0.1|^5=0.00001$], [$10^(-10)$],
  [residual $I+F$], [$|1-0.1|=0.9$], [$0.9^5=0.59049$], [$0.59049^2=0.34868$],
))

#pause
#result[Correct: *B* — the residual RMS magnitude is $0.59049$; its squared ledger is $0.3487$.]

== Residual structure improves the route, not every ingredient

#align(center, table(
  columns: 3, stroke: 0.45pt + MUTED, inset: (x: 10pt, y: 6pt), align: (left, left, left),
  table.header([failure], [why shortcut alone is insufficient], [check]),
  [branch explodes], [$J_F$ can dominate the identity], [branch/output RMS],
  [branch cancels identity], [$J_F approx -I$ in a direction], [identity vs residual contribution],
  [shape changes], [$x$ and $F(x)$ cannot be added], [projection shape/stride],
  [training diverges], [learning rate still controls update size], [update-to-weight RMS],
))

#pause
#note[Initialization, normalization, residual scaling, and learning rate remain coupled design choices.]

== Ledger 4 — the short path completes the anchor #V

#ledger(
  activation: [ReLU gate contributes $1/2$],
  init: [He gives $q_5=g_0=1$ at $t=0$],
  norm: [$mu=2.5$, $sigma^2=1.25$, normalized RMS $=1$],
  residual: [$F'=-.1$: plain RMS $10^(-5)$; residual RMS $.59049$ ($g=.3487$)],
)

#pause
#result[Each intervention answers a different reason the gauges can drift.]

// ═══════════════════════════ SYNTHESIS · SHOULD COVER ═══════════════════════════
= Measure the system, not the slogan

== One ledger connects all four controls #V

#causal-map
#v(6pt)

#align(center, table(
  columns: 4, stroke: 0.45pt + MUTED, inset: (x: 8pt, y: 5.5pt), align: (left, left, left, left),
  table.header([control], [mechanism], [anchor evidence], [what it does not guarantee]),
  [activation], [local gate], [ReLU contributes $1/2$], [no dead units],
  [initialization], [starting gain], [He changes $c:.5→1$], [scale later in training],
  [normalization], [axis-wise reset], [$[1,2,3,4]→"RMS"=1$], [post-$gamma,beta$ RMS],
  [residual], [short additive route], [$10^(-5)→.59049$], [no cancellation/explosion],
))

#pause
#punch[Use the ledger to explain a mechanism, then measure the actual network.]

== Diagnose with symptom → suspect → test

#align(center, table(
  columns: 3, stroke: 0.45pt + MUTED, inset: (x: 9pt, y: 5.5pt), align: (left, left, left),
  table.header([symptom], [first suspect], [smallest informative test]),
  [activation RMS shrinks by depth], [gate/init mismatch], [log pre/post RMS by layer],
  [early gradient RMS is tiny], [saturation or long path], [log gradient RMS and shortcut terms],
  [many inactive ReLU units], [bias/update/input scale], [active-example count per unit],
  [train/eval predictions differ], [BatchNorm statistics], [same input in both modes],
  [first non-finite value], [scale or update explosion], [locate first bad layer and step],
))

#pause
#caption[Change one cause-relevant control at a time so the experiment can identify the cause.]

== Measure the same RMS gauges in code #I

#codebox(size: 13pt)[```python
def rms(x):
    return x.detach().float().square().mean().sqrt().item()

for name, h in activations.items():
    print(name, "activation_rms", rms(h))

for name, p in model.named_parameters():
    if p.grad is not None:
        print(name, "gradient_rms", rms(p.grad),
              "finite", bool(p.grad.isfinite().all()))
```]

#pause
#caption[`square().mean().sqrt()` matches $r_ell,s_ell$; a raw `norm()` would grow with width and break the ledger.]

== Three labs, one predict → run → explain loop #I

#align(center, table(
  columns: (30%, 70%), stroke: 0.45pt + MUTED, inset: (x: 10pt, y: 7pt), align: (left, left),
  [*1 · autopsy*], [#link(NB + "01_signal_gradient_autopsy.ipynb")[01_signal_gradient_autopsy.ipynb]],
  [*2 · propagate*], [#link(NB + "02_variance_propagation.ipynb")[02_variance_propagation.ipynb]],
  [*3 · compare*], [#link(NB + "03_xavier_vs_he.ipynb")[03_xavier_vs_he.ipynb]],
))

#pause
#align(center, text(size: 18pt)[For each notebook: *commit to a trace → run → compare with the ledger → explain any mismatch*.])

#pause
#note(color: GREEN)[The three filenames above are direct Colab links to the public repository.]

== Misconception check: which claim survives? #Q

#mcq(
  [Which statement is defensible without extra assumptions?],
  [He exactly preserves every layer's variance throughout training],
  [BatchNorm inference must contain only one example],
  [Residual shortcuts guarantee gradients cannot vanish],
  [Each control targets a mechanism; the actual gauges must still be measured],
)

== Answer: mechanisms guide; measurements decide #A

#mcq-answer(
  [D],
  [Each control targets a mechanism; the actual gauges must still be measured],
  [He is an initialization approximation, inference may be batched, and a residual branch can reinforce or cancel the identity contribution.],
)

== Revisit your opening bet

#anchor
#v(6pt)

#align(center, table(
  columns: 3, stroke: 0.45pt + MUTED, inset: (x: 13pt, y: 7pt), align: center,
  table.header([choice], [five-layer squared gauges], [five-layer RMS gauges]),
  [Xavier normal / uniform], [$q_5=g_0=.03125$], [$r_5=s_0=.17678$],
  [He normal], [$q_5=g_0=1$], [$r_5=s_0=1$],
  [all-zero weights], [symmetry is unbroken], [units learn the same feature],
))

#pause
#result[The winning bet was He because ReLU's half-active gate required twice Xavier's variance.]

== Exit ticket: prescribe the smallest repair

For a 30-layer ReLU MLP, you observe activation RMS $1.0→0.6→0.35→dots.c$ and early-layer gradient RMS near zero.

#pause
Write three lines:

1. *symptom:* which gauge drifts, and in which direction?
2. *suspect:* which local multiplier could explain it?
3. *test:* which single controlled change or measurement distinguishes the cause?

#pause
#note(color: TEAL)[A strong answer names the activation/init pair first, verifies per-layer RMS, then considers normalization or a shorter residual route.]

#focus-slide[
  Depth is trainable when signal and gradient RMS remain usable along the routes that matter.
  #v(12pt)
  #set text(size: 22pt)
  Next: L7 asks whether a trainable network also *generalizes*—regularization and the train/test gap.
]

// ═══════════════════════════ OPTIONAL APPENDIX · 10 MIN ═══════════════════════════
= Further variants and provenance

== RMSNorm resets magnitude without mean subtraction

For one example with $D$ features,

$ "RMS"(x)=sqrt(1/D sum_(j=1)^D x_j^2+epsilon), quad y_j=gamma_j x_j/"RMS"(x). $

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([KEEPS], [feature-wise learned scale and RMS denominator], color: GREEN),
  hairline([DROPS], [mean subtraction and learned shift in the basic form], color: MUTED),
))

#pause
#note[$epsilon>0$ prevents division by zero and limits amplification when the input RMS is extremely small.]

== Projection shortcuts can change width and resolution

If $F_ell(x_ell)$ and $x_ell$ have different shapes, use

$ x_(ell+1)=P_ell x_ell+F_ell(x_ell). $

#pause
In CNNs, $P_ell$ is often a learned $1 times 1$ convolution:

- stride $1$ can change channel count;
- stride $>1$ can change channels *and* downsample spatial resolution.

#pause
#result[The shortcut must match shape before addition; it need not be a literal identity tensor.]

== Placement decides whether the shortcut is truly clean

#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([POST-ACTIVATION], [$x_(ell+1)=phi(x_ell+F_ell(x_ell))$; the final $phi'$ gates both paths], color: ACC),
  hairline([PRE-ACTIVATION], [norm and activation live in $F_ell$; the additive route stays unobstructed], color: GREEN),
))

#pause
#note[Fixup is a separate carefully scaled initialization strategy that trains deep residual networks without normalization; it is evidence that these controls are coupled, not mandatory in one fixed recipe.]

== BatchNorm for convolution reduces over $B,H,W$

For $X in RR^(B times C times H times W)$, BatchNorm computes one mean and variance per channel:

$ mu_c, sigma_c^2 quad "over" B dot H dot W " values of channel " c. $

#pause
#align(center, table(
  columns: 3, stroke: 0.45pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left, left),
  table.header([axis], [reduced?], [meaning]),
  [batch $B$], [yes], [examples share channel statistics],
  [channel $C$], [no], [one statistic pair per channel],
  [space $H,W$], [yes], [locations act as additional samples],
))

#pause
#caption[At inference the frozen per-channel running statistics are used, regardless of inference batch size.]

== Generalized He gain handles a leaky slope

For $phi(u)=max(u,a u)$ with a symmetric pre-activation,

$ EE[phi(z)^2] approx (1+a^2)/2 EE[z^2]. $

#pause
Matching the affine gain gives

$ "Var"(w)=2/((1+a^2)n_"in"). $

#pause
#align(center, table(
  columns: 3, stroke: 0.45pt + MUTED, inset: (x: 13pt, y: 7pt), align: center,
  table.header([negative slope $a$], [gain factor], [$"Var"(w)$]),
  [$0$], [$1/2$], [$2/n_"in"$],
  [$0.1$], [$0.505$], [$2/(1.01n_"in")$],
  [$1$], [$1$], [$1/n_"in"$],
))

== Primary references and provenance

#set text(size: 15pt)

- Glorot & Bengio (2010), #link("https://proceedings.mlr.press/v9/glorot10a.html")[Understanding the difficulty of training deep feedforward neural networks].
- He et al. (2015), #link("https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html")[Delving Deep into Rectifiers].
- Ioffe & Szegedy (2015), #link("https://proceedings.mlr.press/v37/ioffe15.html")[Batch Normalization].
- Ba, Kiros & Hinton (2016), #link("https://arxiv.org/abs/1607.06450")[Layer Normalization].
- He et al. (2016), #link(RESNET)[Deep Residual Learning for Image Recognition].
- Zhang & Sennrich (2019), #link("https://arxiv.org/abs/1910.07467")[Root Mean Square Layer Normalization].

#v(8pt)
#caption[The activation catalogue and seeded signal-flow trace are generated by `lecture6/diagrams/l6_figs.py`; other diagrams and numeric traces are native Typst vectors computed from the equations shown.]
