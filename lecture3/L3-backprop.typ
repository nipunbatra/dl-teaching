// Computation Graphs, Backpropagation and Autodiff — Lecture 3 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture3/L3-backprop.typ
//   typst compile --root . --input handout=true lecture3/L3-backprop.typ
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [Computation Graphs, Backpropagation and Autodiff],
  subtitle: [How gradients flow through neural networks],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let NB = "https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L03/"

// Semantic gradient colors used throughout the lecture.
// Forward values stay neutral so these three roles remain unambiguous.
#let UPSTREAM = TEAL
#let LOCAL = BLUE
#let DOWNSTREAM = ACC
#let FORWARD = MUTED
#let upstream(body) = text(fill: UPSTREAM)[#body]
#let localterm(body) = text(fill: LOCAL)[#body]
#let downstream(body) = text(fill: DOWNSTREAM)[#body]

// Computation-graph grammar. Gradient colors are reserved for backward arrows
// and formulas; forward graph objects are distinguished primarily by shape.
#let LEAF_FILL = white
#let VALUE_FILL = rgb("#F1F4F4")
#let OP_FILL = LOCAL.lighten(92%)
#let LOSS_FILL = DOWNSTREAM.lighten(91%)

#let graphkey = align(center, diagram(spacing: (20mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), $x$, radius: 5.5mm, fill: LEAF_FILL, stroke: 1pt + INK)
  node((1, 0), $f$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((2, 0), $v$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((3, 0), $ell$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((4, 0), $cal(L)$, radius: 5.8mm, fill: LOSS_FILL, stroke: 1.1pt + DOWNSTREAM)
  for i in range(4) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.9pt + FORWARD) }
  node((0, 0.9), text(size: 9.5pt, fill: MUTED)[leaf value], stroke: none, fill: none)
  node((1, 0.9), text(size: 9.5pt, fill: LOCAL)[operation], stroke: none, fill: none)
  node((2, 0.9), text(size: 9.5pt, fill: MUTED)[stored intermediate], stroke: none, fill: none)
  node((3, 0.9), text(size: 9.5pt, fill: LOCAL)[operation], stroke: none, fill: none)
  node((4, 0.9), text(size: 9.5pt, fill: DOWNSTREAM)[scalar loss], stroke: none, fill: none)
}))

// L3 uses a quiet visual language: diagrams and equations carry the argument.
// Comparison columns use a heading and hairline, while genuine warnings retain boxes.
#let neat-card(title, body, color: TEAL, width: 54mm, height: auto, body-align: center) = block(
  width: width, height: height, inset: (x: 5pt, y: 4pt),
  [#text(size: 12.5pt, weight: 700, fill: color)[#title]
   #v(3pt)
   #line(length: 100%, stroke: 0.7pt + color)
   #v(5pt)
   #align(body-align, body)],
)
#let neat-arrow(label: none, color: MUTED) = align(center, [
  #if label != none { [#text(size: 10.5pt, fill: color)[#label] #h(3pt)] }
  #text(size: 21pt, fill: color)[$arrow.r$]
])
#let neat-caption(body) = align(center, text(size: 14.5pt, fill: MUTED)[#body])
#let notebox(body) = block(
  width: 100%,
  inset: (left: 10pt, right: 2pt, top: 4pt, bottom: 4pt),
  stroke: (left: 1.2pt + TEAL),
  [#body],
)
#let result(body) = align(center, text(size: 16.5pt, weight: 650)[#body])
#let gradient-key(label, detail, formula, color) = block(
  width: 100%,
  inset: (left: 10pt, right: 2pt, top: 4pt, bottom: 4pt),
  stroke: (left: 1.4pt + color),
  [#text(size: 14.5pt, fill: color, weight: 700)[#label]
   #v(2pt)
   #text(size: 13pt, fill: MUTED)[#detail]
   #v(5pt)
   #text(fill: color)[#formula]],
)

// ── local fletcher diagrams ──────────────────────────────────────────
// one local operation: neutral values forward; teal → blue → orange backward
#let localnode = align(center, diagram(spacing: (23mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), $u$, radius: 6mm, fill: LEAF_FILL, stroke: 1pt + INK)
  node((1,0), $f$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((2,0), $v$, radius: 6mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((3,0), $ell$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((4,0), $cal(L)$, radius: 6mm, fill: LOSS_FILL, stroke: 1pt + DOWNSTREAM)
  for i in range(4) { edge((i,0),(i + 1,0), "-|>", stroke: 0.9pt + FORWARD) }
  edge((4,0),(2,0), "-|>", bend: 28deg, stroke: 1.2pt + UPSTREAM)
  edge((2,0),(0,0), "-|>", bend: -28deg, stroke: 1.2pt + DOWNSTREAM)
  node((1,-0.72), text(size: 12pt, fill: FORWARD)[forward #h(3pt) $v = f(u)$], stroke: none, fill: none)
  node((1,0.78), text(size: 11pt, fill: LOCAL)[local #h(3pt) $f'(u)$], stroke: none, fill: none)
}))

// compact recurring motif for slides that refer back to the same local rule
#let localnode-mini = align(center, diagram(spacing: (16mm, 7mm), node-stroke: 0.8pt + INK, node-fill: white, {
  node((0,0), $u$, radius: 4.5mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((1,0), $f$, shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 4pt, fill: OP_FILL, stroke: 0.9pt + LOCAL)
  node((2,0), $v$, radius: 4.5mm, fill: VALUE_FILL, stroke: 0.9pt + INK)
  node((3,0), $ell$, shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 4pt, fill: OP_FILL, stroke: 0.9pt + LOCAL)
  node((4,0), $cal(L)$, radius: 4.5mm, fill: LOSS_FILL, stroke: 0.9pt + DOWNSTREAM)
  for i in range(4) { edge((i,0),(i + 1,0), "-|>", stroke: 0.8pt + FORWARD) }
  edge((4,0),(2,0), "-|>", bend: 28deg, stroke: 1.05pt + UPSTREAM)
  edge((2,0),(0,0), "-|>", bend: -28deg, stroke: 1.05pt + DOWNSTREAM)
  node((1,-0.64), text(size: 9.5pt, fill: FORWARD)[$v = f(u)$], stroke: none, fill: none)
  node((1,0.7), text(size: 8.5pt, fill: LOCAL)[local $f'(u)$], stroke: none, fill: none)
}))

// numeric square-operation instance used in the first worked local-rule example
#let squarenode = align(center, diagram(spacing: (26mm, 10mm), node-stroke: 0.9pt + INK, node-fill: white, {
  // Numeric value labels need more room than a one-symbol node. Keep the
  // circle grammar, but size these nodes so $u=3$ and $v=9$ stay on one line.
  node((0,0), align(center + horizon, box(text(size: 16pt)[$u = 3$])), radius: 11mm, fill: LEAF_FILL, stroke: 1pt + INK)
  node((1,0), [$(dot)^2$], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 7pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((2,0), align(center + horizon, box(text(size: 16pt)[$v = 9$])), radius: 11mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((3,0), $ell$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 7pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((4,0), $cal(L)$, radius: 7mm, fill: LOSS_FILL, stroke: 1pt + DOWNSTREAM)
  for i in range(4) { edge((i,0),(i + 1,0), "-|>", stroke: 0.9pt + FORWARD) }
  edge((4,0),(2,0), text(size: 10pt, fill: UPSTREAM)[arriving $g_v = 7$], "-|>", bend: 28deg, stroke: 1.15pt + UPSTREAM, label-sep: 6pt)
  edge((2,0),(0,0), text(size: 10pt, fill: DOWNSTREAM)[returned $g_u = 42$], "-|>", bend: -28deg, stroke: 1.15pt + DOWNSTREAM, label-sep: 6pt)
  node((1,0.7), text(size: 10pt, fill: LOCAL)[local $2u=6$], stroke: none, fill: none)
}))

// Addition merges two values forward and copies one gradient backward. Keep
// the diagram labels short; the full partial-derivative equations live below.
#let addnode = align(center, diagram(spacing: (23mm, 13mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,-0.7), $a$, radius: 5mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((0,0.7), $b$, radius: 5mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((1,0), $+$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((2,0), $v$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((3,0), $ell$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((4,0), $cal(L)$, radius: 5.5mm, fill: LOSS_FILL, stroke: 1pt + DOWNSTREAM)
  edge((0,-0.7),(1,0), "-|>", stroke: 0.85pt + FORWARD)
  edge((0,0.7),(1,0), "-|>", stroke: 0.85pt + FORWARD)
  edge((1,0),(2,0), "-|>", stroke: 0.85pt + FORWARD)
  edge((2,0),(3,0), "-|>", stroke: 0.85pt + FORWARD)
  edge((3,0),(4,0), "-|>", stroke: 0.85pt + FORWARD)
  edge((4,0),(2,0), text(size: 10pt, fill: UPSTREAM)[arriving $g_v$],
    "-|>", bend: 31deg, stroke: 1.15pt + UPSTREAM, label-sep: 8pt)
  edge((2,0),(0,-0.7), text(size: 10pt)[#downstream[$g_a$] $=$ #upstream[$g_v$]],
    "-|>", bend: -25deg, stroke: 1.15pt + DOWNSTREAM, label-sep: 8pt)
  edge((2,0),(0,0.7), text(size: 10pt)[#downstream[$g_b$] $=$ #upstream[$g_v$]],
    "-|>", bend: 25deg, stroke: 1.15pt + DOWNSTREAM, label-sep: 8pt)
}))

// multiplication sends each input the other input during the backward pass
#let mulnode = align(center, diagram(spacing: (23mm, 13mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,-0.7), $a$, radius: 5mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((0,0.7), $b$, radius: 5mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((1,0), $times$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((2,0), $v$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((3,0), $ell$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((4,0), $cal(L)$, radius: 5.5mm, fill: LOSS_FILL, stroke: 1pt + DOWNSTREAM)
  edge((0,-0.7),(1,0), "-|>", stroke: 0.85pt + FORWARD)
  edge((0,0.7),(1,0), "-|>", stroke: 0.85pt + FORWARD)
  edge((1,0),(2,0), "-|>", stroke: 0.85pt + FORWARD)
  edge((2,0),(3,0), "-|>", stroke: 0.85pt + FORWARD)
  edge((3,0),(4,0), "-|>", stroke: 0.85pt + FORWARD)
  edge((4,0),(2,0), text(size: 10pt, fill: UPSTREAM)[arriving $g_v$],
    "-|>", bend: 31deg, stroke: 1.15pt + UPSTREAM, label-sep: 8pt)
  edge((2,0),(0,-0.7), text(size: 10pt)[#downstream[$g_a$] $=$ #upstream[$g_v$] #localterm[$b$]],
    "-|>", bend: -25deg, stroke: 1.15pt + DOWNSTREAM, label-sep: 8pt)
  edge((2,0),(0,0.7), text(size: 10pt)[#downstream[$g_b$] $=$ #upstream[$g_v$] #localterm[$a$]],
    "-|>", bend: 25deg, stroke: 1.15pt + DOWNSTREAM, label-sep: 8pt)
}))

// elementwise activation as the same one-input local-rule motif
#let actnode = align(center, diagram(spacing: (21mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), $u$, radius: 5.5mm, fill: LEAF_FILL, stroke: 1pt + INK)
  node((1,0), $phi$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((2,0), $v$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((3,0), $ell$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((4,0), $cal(L)$, radius: 5.5mm, fill: LOSS_FILL, stroke: 1pt + DOWNSTREAM)
  for i in range(4) { edge((i,0),(i + 1,0), "-|>", stroke: 0.9pt + FORWARD) }
  node((1,-0.74), text(size: 10.5pt, fill: FORWARD)[$v=phi(u)$], stroke: none, fill: none)
  edge((4,0),(2,0), text(size: 10pt, fill: UPSTREAM)[upstream], "-|>", bend: 28deg, stroke: 1.15pt + UPSTREAM, label-sep: 6pt)
  edge((2,0),(0,0), text(size: 9.5pt)[#downstream[$(partial cal(L))/(partial u)$] $=$ #upstream[$(partial cal(L))/(partial v)$] #localterm[$phi'(u)$]], "-|>", bend: -28deg, stroke: 1.15pt + DOWNSTREAM, label-sep: 6pt)
}))

// branch: x fans to u and v, both to L
#let branchgraph = align(center, diagram(spacing: (17mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), $x$, radius: 6mm, fill: LEAF_FILL, stroke: 1pt + INK)
  node((1,-0.75), $f$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((1,0.75), $g$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((2,-0.75), $u$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((2,0.75), $v$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((3.2,0), $ell$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((4.2,0), $cal(L)$, radius: 6mm, fill: LOSS_FILL, stroke: 1pt + DOWNSTREAM)
  edge((0,0),(1,-0.75), "-|>", stroke: 0.9pt + FORWARD)
  edge((0,0),(1,0.75), "-|>", stroke: 0.9pt + FORWARD)
  edge((1,-0.75),(2,-0.75), "-|>", stroke: 0.9pt + FORWARD)
  edge((1,0.75),(2,0.75), "-|>", stroke: 0.9pt + FORWARD)
  edge((2,-0.75),(3.2,0), "-|>", stroke: 0.9pt + FORWARD)
  edge((2,0.75),(3.2,0), "-|>", stroke: 0.9pt + FORWARD)
  edge((3.2,0),(4.2,0), "-|>", stroke: 0.9pt + FORWARD)
}))

// general fan-out: one variable reaches the loss along k distinct paths
#let multipathgraph = align(center, diagram(spacing: (19mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), $x$, radius: 6mm, fill: LEAF_FILL, stroke: 1pt + INK)
  node((3.2,0), $ell$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((4.2,0), $cal(L)$, radius: 6mm, fill: LOSS_FILL, stroke: 1pt + DOWNSTREAM)
  let rows = ((-1.2, $f_1$, $u_1$), (-0.4, $f_2$, $u_2$), (1.2, $f_k$, $u_k$))
  for (y, f, u) in rows {
    node((1,y), f, shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 4pt, fill: OP_FILL, stroke: 0.9pt + LOCAL)
    node((2,y), u, radius: 5mm, fill: VALUE_FILL, stroke: 0.9pt + INK)
    edge((0,0), (1,y), "-|>", stroke: 0.85pt + FORWARD)
    edge((1,y), (2,y), "-|>", stroke: 0.85pt + FORWARD)
    edge((2,y), (3.2,0), "-|>", stroke: 0.85pt + FORWARD)
  }
  edge((3.2,0), (4.2,0), "-|>", stroke: 0.85pt + FORWARD)
  node((1,0.4), text(size: 15pt, fill: MUTED)[$dots.v$], stroke: none, fill: none)
  node((2,0.4), text(size: 15pt, fill: MUTED)[$dots.v$], stroke: none, fill: none)
}))

// scalar computation graph for L=(wx+b-y)^2. Values and operations alternate.
#let scalargraph = align(center, diagram(spacing: (18mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,1.3), $w$, radius: 5.2mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((0,0.3), $x$, radius: 5.2mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((1,0.8), $times$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((2,0.8), $m$, radius: 5.2mm, fill: VALUE_FILL, stroke: 0.9pt + INK)
  node((2,-0.5), $b$, radius: 5.2mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((3,0.3), $+$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((4,0.3), $a$, radius: 5.2mm, fill: VALUE_FILL, stroke: 0.9pt + INK)
  node((4,-1), $y$, radius: 5.2mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((5,-0.3), $-$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((6,-0.3), $e$, radius: 5.2mm, fill: VALUE_FILL, stroke: 0.9pt + INK)
  node((7,-0.3), [$(dot)^2$], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((8,-0.3), $cal(L)$, radius: 5.5mm, fill: LOSS_FILL, stroke: 1pt + DOWNSTREAM)
  edge((0,1.3),(1,0.8), "-|>", stroke: 0.8pt + FORWARD)
  edge((0,0.3),(1,0.8), "-|>", stroke: 0.8pt + FORWARD)
  edge((1,0.8),(2,0.8), "-|>", stroke: 0.8pt + FORWARD)
  edge((2,0.8),(3,0.3), "-|>", stroke: 0.8pt + FORWARD)
  edge((2,-0.5),(3,0.3), "-|>", stroke: 0.8pt + FORWARD)
  edge((3,0.3),(4,0.3), "-|>", stroke: 0.8pt + FORWARD)
  edge((4,0.3),(5,-0.3), "-|>", stroke: 0.8pt + FORWARD)
  edge((4,-1),(5,-0.3), "-|>", stroke: 0.8pt + FORWARD)
  edge((5,-0.3),(6,-0.3), "-|>", stroke: 0.8pt + FORWARD)
  edge((6,-0.3),(7,-0.3), "-|>", stroke: 0.8pt + FORWARD)
  edge((7,-0.3),(8,-0.3), "-|>", stroke: 0.8pt + FORWARD)
}))

// full scalar graph with one operation boxed for the staged backward derivation
#let scalargraph-focus(stage) = align(center, diagram(spacing: (18mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  // A symbol and its stored number are two distinct visual lines. Explicit
  // sizes and spacing keep their baselines apart inside a projector-sized node.
  let vlabel(sym, num) = stack(
    dir: ttb,
    spacing: 1.5pt,
    align(center, text(size: 14pt)[#sym]),
    align(center, text(size: 9.5pt, fill: MUTED)[#num]),
  )
  let value(pos, label, keys, leaf: false, loss: false) = {
    let active = keys.contains(stage)
    let fill-color = if loss { LOSS_FILL } else if leaf { LEAF_FILL } else { VALUE_FILL }
    let base-stroke = if loss { DOWNSTREAM } else { INK }
    node(pos, label, radius: 7mm, fill: fill-color,
      stroke: if active { 1.7pt + LOCAL } else { 0.9pt + base-stroke })
  }
  let op(pos, label, key) = node(pos, label, shape: fletcher.shapes.rect,
    corner-radius: 4pt, inset: 6pt,
    fill: if stage == key { LOCAL.lighten(82%) } else { OP_FILL },
    stroke: if stage == key { 1.8pt + LOCAL } else { 0.9pt + LOCAL })

  value((0,1.3), vlabel($w$, [2]), ("mul",), leaf: true)
  value((0,0.3), vlabel($x$, [3]), ("mul",), leaf: true)
  op((1,0.8), $times$, "mul")
  value((2,0.8), vlabel($m$, [6]), ("mul", "add"))
  value((2,-0.5), vlabel($b$, [1]), ("add",), leaf: true)
  op((3,0.3), $+$, "add")
  value((4,0.3), vlabel($a$, [7]), ("add", "sub"))
  value((4,-1), vlabel($y$, [10]), ("sub",), leaf: true)
  op((5,-0.3), $-$, "sub")
  value((6,-0.3), vlabel($e$, [-3]), ("sub", "square"))
  op((7,-0.3), [$(dot)^2$], "square")
  value((8,-0.3), vlabel($cal(L)$, [9]), ("square",), loss: true)
  edge((0,1.3),(1,0.8), "-|>", stroke: 0.75pt + FORWARD)
  edge((0,0.3),(1,0.8), "-|>", stroke: 0.75pt + FORWARD)
  edge((1,0.8),(2,0.8), "-|>", stroke: 0.75pt + FORWARD)
  edge((2,0.8),(3,0.3), "-|>", stroke: 0.75pt + FORWARD)
  edge((2,-0.5),(3,0.3), "-|>", stroke: 0.75pt + FORWARD)
  edge((3,0.3),(4,0.3), "-|>", stroke: 0.75pt + FORWARD)
  edge((4,0.3),(5,-0.3), "-|>", stroke: 0.75pt + FORWARD)
  edge((4,-1),(5,-0.3), "-|>", stroke: 0.75pt + FORWARD)
  edge((5,-0.3),(6,-0.3), "-|>", stroke: 0.75pt + FORWARD)
  edge((6,-0.3),(7,-0.3), "-|>", stroke: 0.75pt + FORWARD)
  edge((7,-0.3),(8,-0.3), "-|>", stroke: 0.75pt + FORWARD)
}))

// Recurring circular node-link views. Cards explain a stage; these diagrams keep
// the actual neural/computation graph visible throughout the lecture.
#let mlpforward = align(center, diagram(spacing: (14mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), $bold(x)$, radius: 5.5mm, fill: LEAF_FILL, stroke: 1pt + INK)
  node((1,0), [dense 1], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((2,0), $bold(z)_1$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((3,0), [ReLU], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((4,0), $bold(h)$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((5,0), [dense 2], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((6,0), $bold(hat(y))$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((7,0), [squared loss], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((8,0), $cal(L)$, radius: 5.8mm, fill: LOSS_FILL, stroke: 1pt + DOWNSTREAM)
  for i in range(8) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.9pt + FORWARD) }
}))

#let mlpbackward = align(center, diagram(spacing: (14mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), $bold(x)$, radius: 5.5mm, fill: LEAF_FILL, stroke: 1pt + INK)
  node((1,0), [dense 1], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((2,0), $bold(z)_1$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((3,0), [ReLU], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((4,0), $bold(h)$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((5,0), [dense 2], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((6,0), $bold(hat(y))$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((7,0), [squared loss], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((8,0), $cal(L)$, radius: 5.8mm, fill: LOSS_FILL, stroke: 1pt + DOWNSTREAM)
  for i in range(8) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.75pt + FORWARD) }
  for (right, left) in ((8,6), (6,4), (4,2), (2,0)) {
    edge((right,0), (left,0), "-|>", bend: 25deg, stroke: 1.15pt + DOWNSTREAM)
  }
}))

// Same MLP graph, with only one half of the reverse sweep emphasized.
#let mlpbackward-focus(stage) = align(center, diagram(spacing: (14mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), $bold(x)$, radius: 5.5mm, fill: LEAF_FILL, stroke: 1pt + INK)
  node((1,0), [dense 1], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((2,0), $bold(z)_1$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((3,0), [ReLU], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((4,0), $bold(h)$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((5,0), [dense 2], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((6,0), $bold(hat(y))$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((7,0), [squared loss], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((8,0), $cal(L)$, radius: 5.8mm, fill: LOSS_FILL, stroke: 1pt + DOWNSTREAM)
  for i in range(8) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.75pt + FORWARD) }
  let active = if stage == "loss" {
    ((8,6),)
  } else if stage == "dense2" {
    ((6,4),)
  } else {
    ((4,2), (2,0))
  }
  for (right, left) in active {
    edge((right,0), (left,0), "-|>", bend: 25deg, stroke: 1.5pt + DOWNSTREAM)
  }
}))

#let neuronflow = align(center, diagram(spacing: (20mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), $bold(x)$, radius: 6mm, fill: INK.lighten(95%))
  node((0, 1.1), $bold(w)$, radius: 5.5mm)
  node((0, -1.1), $b$, radius: 5.5mm)
  node((1.3, 0), $z$, radius: 6mm, fill: TEAL.lighten(93%), stroke: 1pt + TEAL)
  node((2.6, 0), $a$, radius: 6mm, fill: TEAL.lighten(93%), stroke: 1pt + TEAL)
  node((2.6, -1.1), $y$, radius: 5.5mm)
  node((3.9, 0), $cal(L)$, radius: 6mm, fill: ACC.lighten(92%), stroke: 1pt + ACC)
  edge((0, 0), (1.3, 0), "-|>", stroke: 1.05pt + TEAL)
  edge((0, 1.1), (1.3, 0), "-|>", stroke: 0.85pt + MUTED)
  edge((0, -1.1), (1.3, 0), "-|>", stroke: 0.85pt + MUTED)
  edge((1.3, 0), (2.6, 0), text(size: 11pt, fill: TEAL)[$phi$], "-|>", stroke: 1.05pt + TEAL, label-sep: 6pt)
  edge((2.6, 0), (3.9, 0), text(size: 11pt, fill: ACC)[$ell$], "-|>", stroke: 1.05pt + ACC, label-sep: 6pt)
  edge((2.6, -1.1), (3.9, 0), "-|>", stroke: 0.85pt + MUTED)
}))

#let densebackgraph = align(center, diagram(spacing: (18mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), $bold(x)$, radius: 5.5mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((0, 1.1), $bold(W)$, radius: 5.5mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((0, -1.1), $bold(b)$, radius: 5.5mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((1.4, 0), [dense], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((2.6, 0), $bold(z)$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((3.8, 0), $ell$, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((5, 0), $cal(L)$, radius: 5.8mm, fill: LOSS_FILL, stroke: 1pt + DOWNSTREAM)
  edge((0, 0), (1.4, 0), "-|>", stroke: 0.85pt + FORWARD)
  edge((0, 1.1), (1.4, 0), "-|>", stroke: 0.85pt + FORWARD)
  edge((0, -1.1), (1.4, 0), "-|>", stroke: 0.85pt + FORWARD)
  edge((1.4, 0), (2.6, 0), "-|>", stroke: 0.85pt + FORWARD)
  edge((2.6, 0), (3.8, 0), "-|>", stroke: 0.85pt + FORWARD)
  edge((3.8, 0), (5, 0), "-|>", stroke: 0.85pt + FORWARD)
  edge((5, 0), (2.6, 0), text(size: 10pt, fill: UPSTREAM)[upstream], "-|>", bend: 28deg, stroke: 1.2pt + UPSTREAM, label-sep: 6pt)
  edge((2.6, 0), (0, 0), "-|>", bend: -28deg, stroke: 1.15pt + DOWNSTREAM)
  edge((2.6, 0), (0, 1.1), "-|>", bend: 24deg, stroke: 1.15pt + DOWNSTREAM)
  edge((2.6, 0), (0, -1.1), "-|>", bend: -24deg, stroke: 1.15pt + DOWNSTREAM)
  node((1.4,0.82), text(size: 10pt, fill: LOCAL)[local dense rule], stroke: none, fill: none)
}))

// A vector affine operation is the first non-scalar extension of the local rule.
#let vectorgraph = align(center, diagram(spacing: (18mm, 10mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 1.15), $bold(x)$, radius: 5.5mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((0, 0), $bold(w)$, radius: 5.5mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((0, -1.15), $b$, radius: 5.5mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((1.4, 0), [affine], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((2.6, 0), $z$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((3.8, 0), [$(dot)^2$], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((5, 0), $cal(L)$, radius: 5.8mm, fill: LOSS_FILL, stroke: 1pt + DOWNSTREAM)
  edge((0, 1.15), (1.4, 0), "-|>", stroke: 0.9pt + FORWARD)
  edge((0, 0), (1.4, 0), "-|>", stroke: 0.9pt + FORWARD)
  edge((0, -1.15), (1.4, 0), "-|>", stroke: 0.9pt + FORWARD)
  edge((1.4, 0), (2.6, 0), "-|>", stroke: 0.9pt + FORWARD)
  edge((2.6, 0), (3.8, 0), "-|>", stroke: 0.9pt + FORWARD)
  edge((3.8, 0), (5, 0), "-|>", stroke: 0.9pt + FORWARD)
  node((1.4, 0.9), text(size: 10.5pt, fill: LOCAL)[$z = bold(w)^top bold(x) + b$], stroke: none, fill: none)
}))

#let vectorseedgraph = align(center, diagram(spacing: (33mm, 13mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let value-label(symbol, value) = stack(
    dir: ttb, spacing: 1pt,
    text(size: 16pt)[#symbol],
    text(size: 8.5pt, fill: MUTED)[value #value],
  )
  node((0, 0), value-label($z$, $-1$), radius: 8.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((1, 0), text(size: 12pt, weight: 650)[square], shape: fletcher.shapes.rect,
    corner-radius: 4pt, inset: 7pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((2, 0), value-label($cal(L)$, $1$), radius: 8.5mm, fill: LOSS_FILL, stroke: 1pt + DOWNSTREAM)
  edge((0, 0), (1, 0), "-|>", stroke: 0.9pt + FORWARD)
  edge((1, 0), (2, 0), "-|>", stroke: 0.9pt + FORWARD)
  node((1, -0.72), text(size: 10pt, fill: LOCAL)[local gradient $partial cal(L)\/partial z=2z=-2$], stroke: none, fill: none)
  node((2, -0.72), text(size: 9.5pt, fill: UPSTREAM)[seed $partial cal(L)\/partial cal(L)=1$], stroke: none, fill: none)
  edge((2, 0), (0, 0), text(size: 10.5pt, fill: DOWNSTREAM)[downstream gradient $partial cal(L)\/partial z=-2$],
    "-|>", bend: 29deg, stroke: 1.25pt + DOWNSTREAM, label-sep: 7pt)
}))

#let vectorbackgraph = align(center, diagram(spacing: (18mm, 10mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 1.15), $bold(x)$, radius: 5.5mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((0, 0), $bold(w)$, radius: 5.5mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((0, -1.15), $b$, radius: 5.5mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((1.4, 0), [affine], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((2.6, 0), $z$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((3.8, 0), [$(dot)^2$], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((5, 0), $cal(L)$, radius: 5.8mm, fill: LOSS_FILL, stroke: 1pt + DOWNSTREAM)
  edge((0, 1.15), (1.4, 0), "-|>", stroke: 0.75pt + FORWARD)
  edge((0, 0), (1.4, 0), "-|>", stroke: 0.75pt + FORWARD)
  edge((0, -1.15), (1.4, 0), "-|>", stroke: 0.75pt + FORWARD)
  edge((1.4, 0), (2.6, 0), "-|>", stroke: 0.75pt + FORWARD)
  edge((2.6, 0), (3.8, 0), "-|>", stroke: 0.75pt + FORWARD)
  edge((3.8, 0), (5, 0), "-|>", stroke: 0.75pt + FORWARD)
  edge((5, 0), (2.6, 0), "-|>", bend: 28deg, stroke: 1.2pt + UPSTREAM)
  edge((2.6, 0), (0, 1.15), "-|>", bend: 24deg, stroke: 1.15pt + DOWNSTREAM)
  edge((2.6, 0), (0, 0), "-|>", bend: -24deg, stroke: 1.15pt + DOWNSTREAM)
  edge((2.6, 0), (0, -1.15), "-|>", bend: -24deg, stroke: 1.15pt + DOWNSTREAM)
  node((1.4, 0.9), text(size: 10.5pt, fill: LOCAL)[local affine rule], stroke: none, fill: none)
}))

// A dense layer is a set of vector dot products evaluated in parallel.
#let denseforwardgraph = align(center, diagram(spacing: (18mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), $bold(x)$, radius: 6mm, fill: LEAF_FILL, stroke: 1pt + INK)
  node((1.3, 0.75), [$bold(w)_1^top dot + b_1$], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((1.3, -0.75), [$bold(w)_2^top dot + b_2$], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((2.6, 0.75), $z_1$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((2.6, -0.75), $z_2$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((3.9, 0), [stack], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((5.1, 0), $bold(z)$, radius: 5.8mm, fill: VALUE_FILL, stroke: 1pt + INK)
  edge((0, 0), (1.3, 0.75), "-|>", stroke: 0.9pt + FORWARD)
  edge((0, 0), (1.3, -0.75), "-|>", stroke: 0.9pt + FORWARD)
  edge((1.3, 0.75), (2.6, 0.75), "-|>", stroke: 0.9pt + FORWARD)
  edge((1.3, -0.75), (2.6, -0.75), "-|>", stroke: 0.9pt + FORWARD)
  edge((2.6, 0.75), (3.9, 0), "-|>", stroke: 0.9pt + FORWARD)
  edge((2.6, -0.75), (3.9, 0), "-|>", stroke: 0.9pt + FORWARD)
  edge((3.9, 0), (5.1, 0), "-|>", stroke: 0.9pt + FORWARD)
}))

// Keep the whole two-output dense graph visible while tracing one scalar
// coordinate. The inactive second-neuron path is deliberately quiet; the
// forward path is dark, the arriving gradient is teal, and the returned
// contribution is orange.
#let dense-neuron1-focus = align(center, diagram(spacing: (16mm, 10mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let value-label(symbol, value, quiet: false) = stack(
    dir: ttb, spacing: 1pt,
    text(size: 13pt, fill: if quiet { MUTED } else { INK })[#symbol],
    text(size: 8.5pt, fill: MUTED)[#value],
  )
  let active-stroke = 1.1pt + INK
  let quiet-stroke = 0.7pt + MUTED.lighten(35%)

  node((0, 0), value-label($bold(x)$, $(2,-1)^top$), radius: 11mm,
    fill: LEAF_FILL, stroke: active-stroke)

  node((1.8, -0.78), stack(
      dir: ttb, spacing: 1pt,
      text(size: 11pt, weight: 650)[neuron 1],
      text(size: 8.5pt, fill: MUTED)[$bold(w)_1^top=(1,3), b_1=0$],
    ), shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 6pt,
    fill: OP_FILL, stroke: 1.2pt + LOCAL)
  node((1.8, 0.78), stack(
      dir: ttb, spacing: 1pt,
      text(size: 10.5pt, fill: MUTED)[neuron 2],
      text(size: 8pt, fill: MUTED)[$bold(w)_2^top=(-2,1), b_2=1$],
    ), shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 6pt,
    fill: white, stroke: quiet-stroke)

  node((3.45, -0.78), value-label($z_1$, $-1$), radius: 9.5mm,
    fill: VALUE_FILL, stroke: active-stroke)
  node((3.45, 0.78), value-label($z_2$, $-4$, quiet: true), radius: 9.5mm,
    fill: white, stroke: quiet-stroke)
  node((4.75, 0), [stack], shape: fletcher.shapes.rect,
    corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((6.05, 0), value-label($bold(z)$, $(-1,-4)^top$), radius: 12mm,
    fill: VALUE_FILL, stroke: active-stroke)
  node((7.45, 0), $cal(L)$, radius: 6mm,
    fill: LOSS_FILL, stroke: 1.1pt + DOWNSTREAM)

  // Forward evaluation: emphasize only the coordinate-1 route.
  edge((0, 0), (1.8, -0.78), "-|>", stroke: 1.25pt + INK)
  edge((1.8, -0.78), (3.45, -0.78), "-|>", stroke: 1.25pt + INK)
  edge((3.45, -0.78), (4.75, 0), "-|>", stroke: 1.25pt + INK)
  edge((4.75, 0), (6.05, 0), "-|>", stroke: 1.25pt + INK)
  edge((6.05, 0), (7.45, 0), "-|>", stroke: 1.25pt + INK)
  edge((0, 0), (1.8, 0.78), "-|>", stroke: quiet-stroke)
  edge((1.8, 0.78), (3.45, 0.78), "-|>", stroke: quiet-stroke)
  edge((3.45, 0.78), (4.75, 0), "-|>", stroke: quiet-stroke)

  // Reverse sweep: select coordinate 1, then apply its scalar affine rule.
  edge((7.45, 0), (6.05, 0), text(size: 9pt, fill: UPSTREAM)[$bold(g)_z=(4,-2)^top$],
    "-|>", bend: 27deg, stroke: 1.2pt + UPSTREAM, label-sep: 6pt)
  edge((6.05, 0), (3.45, -0.78), "-|>", bend: 24deg,
    stroke: 1.25pt + UPSTREAM)
  node((4.72, -0.62), text(size: 9pt, fill: UPSTREAM)[arriving $g_1=4$],
    stroke: none, fill: none)
  edge((3.45, -0.78), (1.8, -0.78), "-|>", bend: -20deg,
    stroke: 1.2pt + DOWNSTREAM)
  edge((1.8, -0.78), (0, 0), "-|>", bend: -25deg,
    stroke: 1.2pt + DOWNSTREAM)
}))

// Expanded scalar view of a 2 x 2 vector map: every output depends on every input.
#let jacobianexamplegraph = align(center, diagram(spacing: (21mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, -0.75), $x_1$, radius: 5.5mm, fill: LEAF_FILL, stroke: 1pt + INK)
  node((0, 0.75), $x_2$, radius: 5.5mm, fill: LEAF_FILL, stroke: 1pt + INK)
  node((1.5, -0.75), [$x_1+x_2$], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((1.5, 0.75), [$2x_1-x_2$], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 5pt, fill: OP_FILL, stroke: 1pt + LOCAL)
  node((3, -0.75), $z_1$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  node((3, 0.75), $z_2$, radius: 5.5mm, fill: VALUE_FILL, stroke: 1pt + INK)
  edge((0, -0.75), (1.5, -0.75), "-|>", stroke: 0.9pt + FORWARD)
  edge((0, 0.75), (1.5, -0.75), "-|>", stroke: 0.9pt + FORWARD)
  edge((0, -0.75), (1.5, 0.75), "-|>", stroke: 0.9pt + FORWARD)
  edge((0, 0.75), (1.5, 0.75), "-|>", stroke: 0.9pt + FORWARD)
  edge((1.5, -0.75), (3, -0.75), "-|>", stroke: 0.9pt + FORWARD)
  edge((1.5, 0.75), (3, 0.75), "-|>", stroke: 0.9pt + FORWARD)
}))

// Why an MLP needs a nonlinearity between affine maps.
#let relumotivationgraph = align(center, diagram(spacing: (14mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, -0.75), $bold(x)$, radius: 5mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((1, -0.75), [dense 1], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 4pt, fill: OP_FILL, stroke: 0.9pt + LOCAL)
  node((2, -0.75), $bold(z)_1$, radius: 5mm, fill: VALUE_FILL, stroke: 0.9pt + INK)
  node((5, -0.75), [dense 2], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 4pt, fill: OP_FILL, stroke: 0.9pt + LOCAL)
  node((6, -0.75), $bold(hat(y))$, radius: 5mm, fill: VALUE_FILL, stroke: 0.9pt + INK)
  edge((0, -0.75), (1, -0.75), "-|>", stroke: 0.85pt + FORWARD)
  edge((1, -0.75), (2, -0.75), "-|>", stroke: 0.85pt + FORWARD)
  edge((2, -0.75), (5, -0.75), "-|>", stroke: 0.85pt + FORWARD)
  edge((5, -0.75), (6, -0.75), "-|>", stroke: 0.85pt + FORWARD)
  node((3.5, -1.35), text(size: 10pt, fill: MUTED)[still one affine map], stroke: none, fill: none)

  node((0, 0.75), $bold(x)$, radius: 5mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((1, 0.75), [dense 1], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 4pt, fill: OP_FILL, stroke: 0.9pt + LOCAL)
  node((2, 0.75), $bold(z)_1$, radius: 5mm, fill: VALUE_FILL, stroke: 0.9pt + INK)
  node((3, 0.75), [ReLU], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 4pt, fill: OP_FILL, stroke: 0.9pt + LOCAL)
  node((4, 0.75), $bold(h)$, radius: 5mm, fill: VALUE_FILL, stroke: 0.9pt + INK)
  node((5, 0.75), [dense 2], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 4pt, fill: OP_FILL, stroke: 0.9pt + LOCAL)
  node((6, 0.75), $bold(hat(y))$, radius: 5mm, fill: VALUE_FILL, stroke: 0.9pt + INK)
  for i in range(6) { edge((i, 0.75), (i + 1, 0.75), "-|>", stroke: 0.85pt + FORWARD) }
  node((3, 1.38), text(size: 10pt, fill: LOCAL)[nonlinear map], stroke: none, fill: none)
}))

// A compact deep chain that makes the repeated local multiplications visible.
#let deepchaingraph = align(center, diagram(spacing: (14mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), $h_0$, radius: 4.8mm, fill: LEAF_FILL, stroke: 0.9pt + INK)
  node((1, 0), $f_1$, shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 4pt, fill: OP_FILL, stroke: 0.9pt + LOCAL)
  node((2, 0), $h_1$, radius: 4.8mm, fill: VALUE_FILL, stroke: 0.9pt + INK)
  node((3, 0), $f_2$, shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 4pt, fill: OP_FILL, stroke: 0.9pt + LOCAL)
  node((4, 0), $h_2$, radius: 4.8mm, fill: VALUE_FILL, stroke: 0.9pt + INK)
  node((5, 0), $dots.c$, stroke: none, fill: none)
  node((6, 0), $f_L$, shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 4pt, fill: OP_FILL, stroke: 0.9pt + LOCAL)
  node((7, 0), $h_L$, radius: 4.8mm, fill: VALUE_FILL, stroke: 0.9pt + INK)
  node((8, 0), $ell$, shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 4pt, fill: OP_FILL, stroke: 0.9pt + LOCAL)
  node((9, 0), $cal(L)$, radius: 5mm, fill: LOSS_FILL, stroke: 1pt + DOWNSTREAM)
  for i in range(9) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.75pt + FORWARD) }
  edge((9, 0), (7, 0), "-|>", bend: 25deg, stroke: 1.1pt + DOWNSTREAM)
  edge((7, 0), (4, 0), "-|>", bend: 22deg, stroke: 1.1pt + DOWNSTREAM)
  edge((4, 0), (2, 0), "-|>", bend: 22deg, stroke: 1.1pt + DOWNSTREAM)
  edge((2, 0), (0, 0), "-|>", bend: 25deg, stroke: 1.1pt + DOWNSTREAM)
  node((1, 0.72), text(size: 9pt, fill: LOCAL)[$f_1'$], stroke: none, fill: none)
  node((3, 0.72), text(size: 9pt, fill: LOCAL)[$f_2'$], stroke: none, fill: none)
  node((6, 0.72), text(size: 9pt, fill: LOCAL)[$f_L'$], stroke: none, fill: none)
}))

#title-slide()

// ═══════════════════════════ PART I — Vocabulary ═══════════════════════════
= Backprop vocabulary

== Lecture 2 built the forward map; today we reverse it #V

Last time, an MLP turned an input into a prediction and then one scalar loss. Today we reuse that *same graph* in reverse:
#pause
#two(
  notebox[
    *General pattern*
    #v(5pt)
    *Forward:* compute values
    $ bold(x) -> f_(bold(theta))(bold(x)) = hat(bold(y)) -> cal(L) $
    #v(5pt)
    *Backward:* compute sensitivities
    loss sensitivity for every parameter $theta_j$
  ],
  notebox[
    *Concrete scalar example*
    #v(5pt)
    *Forward:* with $(w, x, b, y) = (2, 3, 1, 10)$
    $ hat(y) = 2 dot 3 + 1 = 7, quad cal(L) = (7 - 10)^2 = 9 $
    #v(5pt)
    *Backward:*
    $ ((partial cal(L))/(partial w), (partial cal(L))/(partial b)) = (-18, -6) $
  ],
)
#pause
#result[same executed graph · values flow forward, gradients flow backward]

== Read the notation before the algorithm

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*Symbol*], [*Meaning in this lecture*]),
  [$u, v, z$], [scalars],
  [$bold(x), bold(z), bold(theta)$], [vectors (bold lowercase)],
  [$bold(W), bold(X)$], [matrices (bold uppercase)],
  [$partial cal(L) \/ partial u$], [how the loss changes when scalar $u$ changes],
  [$partial cal(L) \/ partial bold(theta)$], [all parameter derivatives, arranged like $bold(theta)$],
))
#pause
#notebox[We keep the full $partial cal(L) \/ partial (dot)$ notation visible throughout: the numerator is always the final loss; the denominator names the stored value whose derivative we want.]

== Our computation-graph convention #V

#graphkey
#pause
#align(center, {
  set text(size: 15pt)
  table(
    columns: (38mm, 172mm), stroke: 0.5pt + MUTED,
    inset: (x: 9pt, y: 3.5pt), align: (left, left),
    [*white circle*], [given value: input, parameter, or target — a graph leaf],
    [*gray circle*], [computed intermediate value, saved for backward when needed],
    [*blue box*], [operation: consumes value(s) and produces a value],
    [*gray arrow*], [forward dependency from a value to the operation that uses it],
    [*orange circle*], [final scalar loss $cal(L)$; seeds backward with $partial cal(L) \/ partial cal(L)=1$],
  )
})
#pause
#result[circles store values · boxes transform them · arrows show dependency]

== One derivative, three routes #D

*Symbolic = formula $arrow.r$ formula.* A person or a computer-algebra system may do the algebra.
$ cal(L) = (w x + b - y)^2, quad (w,x,b,y)=(2,3,1,10). $
#pause
#align(center, table(
  columns: (42mm, 86mm, 88mm), stroke: 0.5pt + MUTED,
  inset: (x: 9pt, y: 4.5pt), align: (left, left, left),
  table.header([*Route*], [*What it does*], [*What it returns here*]),
  [#uncover("2-")[symbolic]], [#uncover("2-")[derive a formula]], [#uncover("2-")[$(partial cal(L))/(partial w)=2(w x+b-y)x$; substitute $arrow.r -18$]],
  [#uncover("3-")[finite difference]], [#uncover("3-")[evaluate $cal(L)(w plus.minus epsilon)$]], [#uncover("3-")[approximate number at $w=2$: $-18$]],
  [#uncover("4-")[reverse-mode AD]], [#uncover("4-")[run local rules backward]], [#uncover("4-")[number for this run: $(-6) dot 3=-18$]],
))
#pause
#pause
#pause
#result[formula · approximate value · reverse sweep through the executed graph]

== Gradient checking: compare one coordinate two ways #I

Reverse-mode AD computes the derivative from local rules, up to floating-point arithmetic. Central difference independently approximates the same value:
$g_("FD") = (cal(L)(theta_j + epsilon) - cal(L)(theta_j - epsilon)) /(2 epsilon).$
#pause
#codebox(size: 11.5pt)[```python
import torch
def loss(w): return (w*3.0 + 1.0 - 10.0)**2
w, eps = 2.0, 1e-5
g_fd = (loss(w + eps) - loss(w - eps)) / (2*eps)
wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
loss(wt).backward()
g_ad = wt.grad.item()
print(g_fd, g_ad)  # both approximately -18
```]
#pause
#notebox[Use float64 and check only a few coordinates. Finite differences debug gradients; they do not train models because every checked coordinate needs two extra forward passes.]

== Why train with reverse-mode autodiff? #D

#align(center, table(
  columns: (42mm, 58mm, 66mm, 50mm), stroke: 0.5pt + MUTED,
  inset: (x: 8pt, y: 7pt), align: (left, left, left, left),
  table.header([*Method*], [*Exact?*], [*Cost for $P$ parameters*], [*Use it for*]),
  [symbolic], [exact formula before numerical evaluation], [expression-dependent; algebra can grow], [deriving / reasoning],
  [#uncover("2-")[finite difference]], [#uncover("2-")[approximate]], [#uncover("2-")[$2P$ loss evaluations]], [#uncover("2-")[checking]],
  [#uncover("3-")[reverse mode]], [#uncover("3-")[exact local rules; floating-point values]], [#uncover("3-")[one reverse sweep for scalar $cal(L)$]], [#uncover("3-")[training]],
))
#pause
#pause
#pause
#result[Symbolic to reason · finite differences to check · reverse mode to train.]

== Backprop returns every parameter sensitivity in one sweep

For each parameter coordinate, gradient descent uses
$ theta_(j,t+1) = theta_(j,t) - eta thin (partial cal(L))/(partial theta_j). $
#pause
$eta$ is the learning rate; the derivative supplies the direction and magnitude of the local change.
#pause
A modern network has $P = 10^7$ to $10^11$ parameters.
#pause
#alertbox[We need all $P$ derivatives $partial cal(L) \/ partial theta_j$. Backprop computes them together in *one reverse sweep*, without perturbing parameters one at a time.]

== The route from one local operation to an entire network #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 7pt), align: (left, left),
  table.header([*Stage*], [*Question we answer*]),
  [1 · local rule], [How does one operation pass a gradient backward?],
  [#uncover("2-")[2 · branches]], [#uncover("2-")[Why do gradient contributions add?]],
  [#uncover("3-")[3 · worked graph]], [#uncover("3-")[Can we reproduce every derivative numerically?]],
  [#uncover("4-")[4 · layers + MLP]], [#uncover("4-")[How do scalar rules become vector and matrix formulas?]],
  [#uncover("5-")[5 · depth + gradient flow]], [#uncover("5-")[What happens when many local gradients multiply?]],
))
#pause
#pause
#pause
#pause

== One local operation

Zoom in on one operation. The value circle $u$ enters the boxed operation $f$; its output is stored in the value circle $v$.
#pause
#localnode
#pause
#result[backward computes #downstream[$(partial cal(L))/(partial u)$] $=$ #upstream[$(partial cal(L))/(partial v)$] $dot$ #localterm[$f'(u)$]]

== Every operation multiplies the arriving gradient by its local gradient #D

For the operation $v = f(u)$, the chain rule is
#localnode-mini
#pause
#align(center, text(size: 28pt, weight: 600)[
  #downstream[$(partial cal(L))/(partial u)$] $=$
  #upstream[$(partial cal(L))/(partial v)$] $dot$
  #localterm[$f'(u)$]
])
#pause
#grid(
  columns: 3, gutter: 12pt,
  gradient-key([Downstream gradient], [sent back to $u$], [$(partial cal(L))/(partial u)$], DOWNSTREAM),
  gradient-key([Upstream gradient], [arriving at $v$], [$(partial cal(L))/(partial v)$], UPSTREAM),
  gradient-key([Local gradient], [this operation's slope], [$f'(u) = (partial v)/(partial u)$], LOCAL),
)

== The chain rule is one reusable local rule #D

For every scalar operation $v=f(u)$:
#localnode-mini
#align(center)[#downstream[$(partial cal(L))/(partial u)$] $=$ #upstream[$(partial cal(L))/(partial v)$] $dot$ #localterm[$(partial v)/(partial u)$].]
#pause
#notebox[The operation multiplies its #upstream[*upstream gradient*] by its #localterm[*local gradient*] to produce the #downstream[*downstream gradient*]. It needs no global formula for the whole network.]

== Example: a square operation #D

For $v = u^2$ at $u = 3$, the forward value is $v = 9$ and the #localterm[local slope is $partial v \/ partial u = 2u = 6$]. Suppose the rest of the graph sends #upstream[$g_v = partial cal(L) \/ partial v = 7$] back to this square.
#pause
#squarenode
#pause
#align(center)[#downstream[$(partial cal(L))/(partial u)$] $=$ #upstream[$(partial cal(L))/(partial v)$] $dot$ #localterm[$2u$] $= 7 dot 6 = 42$.]
#pause
#notebox[The square operation multiplies the #upstream[arriving $partial cal(L) \/ partial v = 7$] by its #localterm[local gradient $2u = 6$] to produce #downstream[$partial cal(L) \/ partial u = 42$].]

== Addition copies the arriving gradient #D

Operation $v = a + b$. Let #upstream[$g_v = partial cal(L) \/ partial v$] arrive. #localterm[Local derivatives: $partial v \/ partial a = 1$, $partial v \/ partial b = 1$.]
#pause
#addnode
#pause
#align(center)[
  #downstream[$(partial cal(L))/(partial a)$] $=$ #upstream[$(partial cal(L))/(partial v)$] #localterm[$dot 1$],
  #h(8mm)
  #downstream[$(partial cal(L))/(partial b)$] $=$ #upstream[$(partial cal(L))/(partial v)$] #localterm[$dot 1$].
]
#pause
#result[$+$ copies the #upstream[arriving derivative] into a #downstream[downstream derivative] for every input.]

== Multiplication scales by the other input #D

Operation $v = a b$. Let #upstream[$g_v = partial cal(L) \/ partial v$] arrive. #localterm[Local derivatives: $partial v \/ partial a = b$, $partial v \/ partial b = a$.]
#pause
#mulnode
#pause
#align(center)[
  #downstream[$(partial cal(L))/(partial a)$] $=$ #upstream[$(partial cal(L))/(partial v)$] #localterm[$dot b$],
  #h(8mm)
  #downstream[$(partial cal(L))/(partial b)$] $=$ #upstream[$(partial cal(L))/(partial v)$] #localterm[$dot a$].
]
#pause
#result[$times$ uses the #localterm[other input as its local multiplier].]
#pause
#notebox[If $a = 2$, $b = 5$, and #upstream[$partial cal(L) \/ partial v = 3$], then #downstream[$partial cal(L) \/ partial a = 15$ and $partial cal(L) \/ partial b = 6$].]

== Activation operation #D

Operation $v = phi(u)$ (elementwise nonlinearity). #localterm[Local derivative: $partial v \/ partial u = phi'(u)$.]
#pause
#actnode
#pause
#align(center)[#downstream[$(partial cal(L))/(partial u)$] $=$ #upstream[$(partial cal(L))/(partial v)$] $dot$ #localterm[$phi'(u)$].]
#pause
#two(
  notebox[*sigmoid* $sigma$: #h(0.3em) $phi'(u) = sigma(u)(1 - sigma(u))$],
  notebox[*ReLU*: #h(0.3em) $phi'(u) = 1$ for $u>0$, and $0$ for $u<0$],
)

== Pop quiz: what crosses an inactive ReLU? #Q

#mcq(
  [Let $v="ReLU"(u)$. If $u=-2$ and the #upstream[*upstream gradient* is $partial cal(L) \/ partial v=5$], what #downstream[*downstream gradient* $partial cal(L) \/ partial u$] is sent back?],
  [$5$],
  [$-10$],
  [$0$],
  [undefined because $u<0$],
)

== Answer: an inactive ReLU blocks the gradient #A

#mcq-answer([C], [$0$], [For $u<0$, the #localterm[local gradient is $partial v \/ partial u=0$]. Therefore #downstream[downstream] $=$ #upstream[upstream] $times$ #localterm[local] $=5 times 0=0$.])
#pause
#two(
  notebox[*Forward:* $u=-2 arrow.r v=0$],
  notebox[*Backward:* $5 times 0 arrow.r partial cal(L) \/ partial u=0$],
)
#pause
#notebox[At exactly $u=0$, ReLU's ordinary derivative is undefined. Autodiff software must choose a convention; commonly it returns $0$.]

== The rule table — memorize this #V

#align(center, table(
  columns: (42mm, 172mm), stroke: 0.5pt + MUTED, inset: (x: 9pt, y: 6pt), align: (left, left),
  table.header([*Forward operation*], [*Contribution sent backward*]),
  [$v = a + b$],   [#downstream[$(partial cal(L))/(partial a)$] += #upstream[$(partial cal(L))/(partial v)$] #localterm[$dot 1$]; #h(0.3em) #downstream[$(partial cal(L))/(partial b)$] += #upstream[$(partial cal(L))/(partial v)$] #localterm[$dot 1$]],
  [$v = a b$],     [#downstream[$(partial cal(L))/(partial a)$] += #upstream[$(partial cal(L))/(partial v)$] #localterm[$dot b$]; #h(0.3em) #downstream[$(partial cal(L))/(partial b)$] += #upstream[$(partial cal(L))/(partial v)$] #localterm[$dot a$]],
  [$v = u^2$],     [#downstream[$(partial cal(L))/(partial u)$] += #upstream[$(partial cal(L))/(partial v)$] #localterm[$dot 2u$]],
  [$v = phi(u)$],  [#downstream[$(partial cal(L))/(partial u)$] += #upstream[$(partial cal(L))/(partial v)$] #localterm[$dot phi'(u)$]],
))
#pause
#alertbox[Use `+=`, not `=` — a value consumed by several operations accumulates gradient from *all* of them.]

// ═══════════════════════════ PART II — Accumulation ═══════════════════════════
= Gradients accumulate at branches

== Why gradients add at a branch #V

Imagine nudging $x$ from $4$ to $4.01$. The *same nudge* travels down both routes to the loss:
#pause
#branchgraph
#pause
#two(
  notebox[
    *Via $u = x^2$:* the local slope is $2x = 8$,
    so $Delta x = 0.01$ changes the loss by about $8 dot 0.01 = 0.08$.
  ],
  notebox[
    *Via $v = 3x$:* the local slope is $3$,
    so the same nudge changes the loss by about $3 dot 0.01 = 0.03$.
  ],
)
#v(7pt)
#result[total slope $=$ total loss change $div$ nudge $= (0.08 + 0.03) / 0.01 = 8 + 3 = 11$]

== The multivariable chain rule adds every path #D

The nudge experiment generalizes. If $x$ influences $cal(L)$ through $u_1, dots, u_k$:
#pause
#multipathgraph
#v(3pt)
#neat-caption[branch $i$ contributes #upstream[upstream $partial cal(L) \/ partial u_i$] $times$ #localterm[local $partial u_i \/ partial x$]]
#pause
#align(center)[#text(size: 25pt, weight: 600)[
  #downstream[$(partial cal(L))/(partial x)$] $= sum_(i=1)^k$
  #upstream[$(partial cal(L))/(partial u_i)$]
  #localterm[$(partial u_i)/(partial x)$].
]]
#pause
#result[each path contributes one chain-rule product; the products add]

== Worked branch example

Let $u = x^2$, #h(0.4em) $v = 3x$, #h(0.4em) $cal(L) = u + v = x^2 + 3x$.
#pause
By symbolic differentiation — derive the formula first:
$ (partial cal(L))/(partial x) = 2x + 3, quad "at" x = 4: quad 2(4) + 3 = 11. $
#pause
Now let us get the *same* answer by backprop.

== Same answer, by backprop #D

*Forward* at $x = 4$: #h(0.5em) $u = 16$, #h(0.3em) $v = 12$, #h(0.3em) $cal(L) = 28$.
#pause
*Backward:* #h(0.5em) seed $partial cal(L) \/ partial cal(L) = 1$; the add operation gives #upstream[$partial cal(L) \/ partial u = 1$ and $partial cal(L) \/ partial v = 1$].
#pause
#align(center)[
  #downstream[$((partial cal(L))/(partial x))_("via" u)$] $=$ #upstream[$1$] $dot$ #localterm[$2x$] $= 8$,
  #h(7mm)
  #downstream[$((partial cal(L))/(partial x))_("via" v)$] $=$ #upstream[$1$] $dot$ #localterm[$3$] $= 3$.
]
#pause
#align(center)[#downstream[$(partial cal(L))/(partial x) = 8 + 3 = 11$]. #h(3mm) $checkmark$]

== PyTorch adds the same two paths #I

#codebox(size: 14pt)[```python
import torch

x = torch.tensor(4.0, requires_grad=True)
u = x**2                 # path 1
v = 3*x                  # path 2
L = u + v
L.backward()

print(x.grad)            # tensor(11.)
```]
#pause
#two(
  notebox[*via $u$:* #localterm[$partial u / partial x = 2x = 8$]],
  notebox[*via $v$:* #localterm[$partial v / partial x = 3$]],
)
#pause
#result[autograd accumulates both #downstream[downstream contributions]: $8 + 3 = 11$]

// ═══════════════════════════ PART III — Central example ═══════════════════════════
= The central worked example

== A scalar linear model #D

Predict $hat(y) = w x + b$, squared-error loss $cal(L) = (hat(y) - y)^2$.
#pause
Concrete numbers:
$ x = 3, quad w = 2, quad b = 1, quad y = 10. $
#pause
#notebox[Compute $partial cal(L) \/ partial w$, $partial cal(L) \/ partial x$, $partial cal(L) \/ partial b$, and $partial cal(L) \/ partial y$ by backpropagation; verify the four values with PyTorch.]

== Break into primitives

Decompose the loss into elementary operations and name each stored result:
$ m = w x, quad a = m + b, quad e = a - y, quad cal(L) = e^2. $
#pause
#result[value $m$ after $times$ · value $a$ after $+$ · value $e$ after $-$ · value $cal(L)$ after $(dot)^2$]
#pause
The symbols $m$, $a$, and $e$ name stored intermediate values. They are circles in the graph; the four operations are boxes.

== The computation graph #V

#scalargraph
#align(center, {
  set text(size: 13pt)
  table(
    columns: 2, stroke: 0.45pt + MUTED.lighten(45%), inset: (x: 18pt, y: 5pt), align: center,
    [#text(weight: 700)[Given values] #h(8pt) $w, x, b, y$],
    [#text(weight: 700, fill: MUTED)[Stored values] #h(8pt) $m, a, e$],
    [#text(weight: 700, fill: LOCAL)[Operations] #h(8pt) $times, +, -, (dot)^2$],
    [#text(weight: 700, fill: DOWNSTREAM)[Scalar loss] #h(8pt) $cal(L)$],
  )
})

== Forward pass #D

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 6pt), align: (left, center, center),
  table.header([*Stored value*], [*Operation that produces it*], [*Numeric value*]),
  [$m$],   [$w times x$],      [$2 dot 3 = 6$],
  [$a$],   [$m + b$],          [$6 + 1 = 7$],
  [$e$],   [$a - y$],          [$7 - 10 = -3$],
  [$cal(L)$], [$e^2$],         [$(-3)^2 = 9$],
))
#pause
#result[$cal(L) = 9$ — now run the graph *backward*]

== Backward: seed and the square operation #D

Seed the output: $partial cal(L) \/ partial cal(L) = 1$.
#pause
#scalargraph-focus("square")
#pause
Square operation $cal(L) = e^2$, local $partial cal(L) \/ partial e = 2e$:
$ (partial cal(L))/(partial e) = 1 dot 2e = 1 dot 2(-3) = -6. $

== Backward: the subtraction operation #D

Subtraction operation $e = a - y$, locals $partial e \/ partial a = 1$, $partial e \/ partial y = -1$:
#pause
#scalargraph-focus("sub")
#pause
#grid(
  columns: 2, gutter: 12mm,
  [$ (partial cal(L))/(partial a) = (partial cal(L))/(partial e) dot 1 = -6 $],
  [#pause $ (partial cal(L))/(partial y) = (partial cal(L))/(partial e) dot (-1) = 6 $],
)
#pause
#notebox[The target $y$ also receives a derivative. We would use $partial cal(L) \/ partial y$ if $y$ were itself a learned quantity.]

== Backward: the addition operation #D

Addition operation $a = m + b$ *copies* its gradient:
#pause
#scalargraph-focus("add")
#pause
$ (partial cal(L))/(partial m) = (partial cal(L))/(partial a) = -6, quad (partial cal(L))/(partial b) = -6. $
#pause
So $partial cal(L) \/ partial b = -6$ is one of our final answers.

== Backward: the multiplication operation #D

Multiplication operation $m = w x$ *sends the other input*:
#pause
#scalargraph-focus("mul")
#pause
#grid(
  columns: 2, gutter: 12mm,
  [$ (partial cal(L))/(partial w) = (partial cal(L))/(partial m) dot x = -6 dot 3 = -18 $],
  [#pause $ (partial cal(L))/(partial x) = (partial cal(L))/(partial m) dot w = -6 dot 2 = -12 $],
)
#pause
#result[$partial cal(L) \/ partial w = -18, quad partial cal(L) \/ partial x = -12$]

== Final gradients #V

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 15pt, y: 7pt), align: center,
  table.header([$partial cal(L) \/ partial w$], [$partial cal(L) \/ partial b$], [$partial cal(L) \/ partial x$], [$partial cal(L) \/ partial y$]),
  [$-18$], [$-6$], [$-12$], [$6$],
))
#pause
#notebox[One backward sweep produced *all four* partial derivatives — no formula re-derivation per parameter.]

== All stored values and their gradients #V

#scalargraph
#align(center, {
  set text(size: 14pt)
  table(
    columns: (34mm, 42mm, 42mm, 42mm, 42mm),
    stroke: 0.45pt + MUTED.lighten(45%), inset: (x: 7pt, y: 4.5pt), align: center,
    table.header([*first half*], [$w$], [$x$], [$m$], [$b$]),
    [#text(weight: 700, fill: MUTED)[forward value]], [$2$], [$3$], [$6$], [$1$],
    [#text(weight: 700, fill: DOWNSTREAM)[gradient]],
    [#downstream[$g_w=-18$]], [#downstream[$g_x=-12$]], [#downstream[$g_m=-6$]], [#downstream[$g_b=-6$]],
  )
  v(8pt)
  table(
    columns: (34mm, 42mm, 42mm, 42mm, 42mm),
    stroke: 0.45pt + MUTED.lighten(45%), inset: (x: 7pt, y: 4.5pt), align: center,
    table.header([*second half*], [$a$], [$y$], [$e$], [$cal(L)$]),
    [#text(weight: 700, fill: MUTED)[forward value]], [$7$], [$10$], [$-3$], [$9$],
    [#text(weight: 700, fill: DOWNSTREAM)[gradient]],
    [#downstream[$g_a=-6$]], [#downstream[$g_y=6$]], [#downstream[$g_e=-6$]], [#downstream[$g_cal(L)=1$]],
  )
})

== Check with symbolic differentiation #D

Keep $w,x,b,y$ symbolic, derive the formulas, and only then substitute $w x + b - y = -3$:
#pause
$ (partial cal(L))/(partial w) = 2(w x + b - y) x = 2(-3)(3) = -18 quad checkmark $
#pause
$ (partial cal(L))/(partial b) = 2(w x + b - y) = 2(-3) = -6 quad checkmark $
#pause
#result[Backprop and the symbolic formulas agree at this point.]

== PyTorch executes the same graph #I

#codebox(size: 12.5pt)[```python
import torch

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
# Targets normally do not track gradients; enable it here to inspect dL/dy.
y = torch.tensor(10.0, requires_grad=True)

m = w * x
a = m + b
e = a - y
L = e**2
for value in (m, a, e, L):
    value.retain_grad()      # keep intermediate grads for inspection

L.backward()
print([t.item() for t in (m, a, e, L)])
# [6.0, 7.0, -3.0, 9.0]
print([t.grad.item() for t in (L, e, a, m, w, x, b, y)])
# [1.0, -6.0, -6.0, -6.0, -18.0, -12.0, -6.0, 6.0]
```]

== Hand computation → PyTorch autograd #V

#align(center, {
  set text(size: 13.5pt)
  table(
    columns: (60mm, 72mm, 82mm), stroke: 0.5pt + MUTED,
    inset: (x: 9pt, y: 5pt), align: (left, left, left),
    table.header([*In our graph*], [*In PyTorch*], [*Meaning*]),
    [stored scalar value], [`Tensor`], [one stored forward value],
    [#localterm[local backward rule]], [`grad_fn`], [operation that produced a non-leaf tensor],
    [saved forward value], [saved by the operation], [operand needed by a local derivative],
    [reverse sweep], [`L.backward()`], [seed at $cal(L)$; visit operations in reverse order],
    [leaf derivative], [`p.grad`], [$partial cal(L) \/ partial p$, accumulated with `+=`],
    [intermediate derivative], [`retain_grad()`], [keep non-leaf `.grad` for inspection],
  )
})
#pause
#result[reverse mode = direction · backprop = algorithm · autograd = engine: record → save → replay]

== Checkpoint: read the sign of a gradient #Q

#mcq(
  [We found $partial cal(L) \/ partial w = -18$. For a small positive learning rate, which way does gradient descent move $w$?],
  [Decrease $w$],
  [Increase $w$],
  [Leave $w$ unchanged],
  [The sign tells us nothing],
)

== Answer: a negative gradient means increase the parameter #A

#mcq-answer([B], [Increase $w$], [Gradient descent subtracts the gradient: $w <- w - eta(-18) = w + 18 eta$.])

== A small gradient step reduces this loss #D

Take $eta = 0.01$ and update:
$ w <- 2 - 0.01(-18) = 2.18, quad quad b <- 1 - 0.01(-6) = 1.06. $
#pause
New prediction: $hat(y) = 2.18 dot 3 + 1.06 = 7.60$.
#pause
#result[loss falls from $9$ to $(7.60 - 10)^2 = 5.76$]
#pause
#notebox[Backprop supplied the direction. The learning rate controls how far we move; a later optimization lecture studies that choice.]

== Interactive: the computation graph #I

#interbox(link-to: IA + "autograd")[
  Build a scalar computation graph, run the forward pass, then click *backward* to watch each $partial cal(L) \/ partial u$ fill in — exactly the $(w x + b - y)^2$ example above.
]
#pause
Trace which local rule changes when the final squared-error operation is replaced with $|e|$.

== Practice the scalar graph three ways #I

#grid(
  columns: 3, gutter: 10pt,
  notebox[
    *See it move*
    #v(5pt)
    Step through values and loss derivatives in the #link(IA + "autograd")[interactive graph ↗].
  ],
  notebox[
    *PyTorch → build autograd*
    #v(5pt)
    Open the #link(NB + "04_scalar_autograd_pytorch_to_scratch.ipynb")[scalar-engine Colab ↓]: use plain PyTorch, trace every edge, then compare fused and atomic sigmoid.
  ],
  notebox[
    *Continue through the lecture*
    #v(5pt)
    Use the #link(NB + "00_backprop_autograd_complete.ipynb")[complete Colab ↓] for branches, vectors, the dense VJP, batches, one update, and the tiny MLP.
  ],
)
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 5pt), align: (left, left),
  [*Easy*], [change one input and recompute the forward pass],
  [*Medium*], [derive all four gradients before calling `backward()`],
  [*Hard*], [replace the square by $|e|$; state what happens at $e = 0$],
))

// ═══════════════════════════ PART IV — Vectors ═══════════════════════════
= From scalars to vectors

== Derivative, gradient, Jacobian: the shape tells us #V

We store gradients as column vectors. The name of the local object depends on the input and output shapes:
#pause
#align(center, {
  set text(size: 16pt)
  table(
    columns: (70mm, 88mm, 56mm), stroke: 0.5pt + MUTED,
    inset: (x: 9pt, y: 6pt), align: (left, center, left),
    table.header([*Operation shape*], [*Local object*], [*Name*]),
    [$u in RR arrow.r v in RR$], [$partial v \/ partial u in RR$], [derivative],
    [$bold(x) in RR^d arrow.r z in RR$], [$partial z \/ partial bold(x) in RR^d$], [gradient],
    [$bold(x) in RR^d arrow.r bold(z) in RR^m$], [$J_(i j) = partial z_i \/ partial x_j$], [Jacobian $bold(J) in RR^(m times d)$],
  )
})
#pause
#result[a Jacobian contains one local derivative for every output–input pair]

== A $2 times 2$ Jacobian is four familiar derivatives #V

Take a vector operation with two inputs and two outputs:
$ z_1=x_1+x_2, quad z_2=2x_1-x_2. $
#pause
#jacobianexamplegraph
#pause
#align(center, table(
  columns: (48mm, 55mm, 55mm), stroke: 0.5pt + MUTED,
  inset: (x: 10pt, y: 6pt), align: center,
  table.header([*Output*], [*Input $x_1$*], [*Input $x_2$*]),
  [$z_1$], [$partial z_1 \/ partial x_1=1$], [$partial z_1 \/ partial x_2=1$],
  [$z_2$], [$partial z_2 \/ partial x_1=2$], [$partial z_2 \/ partial x_2=-1$],
))
#pause
#result[$bold(J)=mat(1,1;2,-1)$ · rows are outputs · columns are inputs]

== Vector chain rule: make the shapes fit #D

For $bold(x) in RR^d arrow.r bold(z) in RR^m$, let the #localterm[local Jacobian] be
$ bold(J)_(i j) = partial z_i \/ partial x_j, quad bold(J) in RR^(m times d). $
#pause
#align(center, text(size: 30pt, weight: 600)[
  #downstream[$(partial cal(L))/(partial bold(x))$] $=$
  #localterm[$bold(J)^top$]
  #upstream[$(partial cal(L))/(partial bold(z))$]
])
#pause
#align(center, text(size: 19pt, fill: MUTED)[
  $(d times 1) quad = quad (d times m) quad (m times 1)$
])
#pause
#result[downstream gradient = local Jacobian transpose × upstream gradient]
#pause
#notebox[When $d=m=1$, $bold(J)$ is one number and this is the scalar chain rule from earlier.]

== The transpose adds every output's contribution #D

For the previous operation, suppose the #upstream[arriving gradient] is
$ bold(g)_z = partial cal(L) \/ partial bold(z) = mat(3;4). $
#pause
#align(center, text(size: 25pt, weight: 600)[
  #downstream[$(partial cal(L))/(partial bold(x))$]
  $=$ #localterm[$bold(J)^top$] #upstream[$bold(g)_z$]
  $= mat(1,2;1,-1) mat(3;4) = mat(11;-1)$
])
#pause
#align(center, table(
  columns: (55mm, 115mm), stroke: 0.5pt + MUTED,
  inset: (x: 11pt, y: 7pt), align: (left, center),
  [#downstream[$partial cal(L) \/ partial x_1$]], [$3(1)+4(2)=11$],
  [#downstream[$partial cal(L) \/ partial x_2$]], [$3(1)+4(-1)=-1$],
))
#pause
#result[the vector rule is the scalar branch rule applied to every input]

== An affine operation adds scalar products and a bias #V

*Affine* means a weighted sum plus a bias:
$ z = bold(w)^top bold(x) + b quad = quad sum_(j=1)^d w_j x_j + b. $

#pause
#vectorgraph

== Change one scalar while holding every other value fixed #D

Start from $z=sum_j w_j x_j+b$. Change only one input by a small amount $Delta$:
#pause
#align(center, {
  set text(size: 16pt)
  table(
    columns: (56mm, 100mm, 62mm), stroke: 0.5pt + MUTED,
    inset: (x: 9pt, y: 7pt), align: (left, center, center),
    table.header([*Change*], [*Resulting change in $z$*], [*Divide by $Delta$*]),
    [$w_i arrow.r w_i+Delta$],
    [$((w_i+Delta) x_i)-w_i x_i = x_i Delta$],
    [$partial z \/ partial w_i=x_i$],
    [$x_i arrow.r x_i+Delta$],
    [$w_i (x_i+Delta)-w_i x_i = w_i Delta$],
    [$partial z \/ partial x_i=w_i$],
    [$b arrow.r b+Delta$],
    [$(b+Delta)-b = Delta$],
    [$partial z \/ partial b=1$],
  )
})
#pause
#result[local derivative = change in the operation's output ÷ change in one input]

== Stack the coordinate derivatives into gradient vectors #D

The derivatives with respect to the coordinates of a vector are stored in one column:
#pause
#align(center, text(size: 23pt)[
  $(partial z)/(partial bold(w))
    := mat(partial z\/partial w_1; dots.v; partial z\/partial w_d)
    quad = quad mat(x_1; dots.v; x_d) = bold(x)$
])
#pause
#align(center, text(size: 23pt)[
  $(partial z)/(partial bold(x))
    := mat(partial z\/partial x_1; dots.v; partial z\/partial x_d)
    quad = quad mat(w_1; dots.v; w_d) = bold(w)$
])
#pause
With $bold(x)=mat(2;-1)$ and $bold(w)=mat(1;3)$:
#align(center, text(size: 20pt)[
  $(partial z)/(partial bold(w))=mat(2;-1), quad
   (partial z)/(partial bold(x))=mat(1;3), quad
   (partial z)/(partial b)=1.$
])
#pause
#result[the vector formulas are just the scalar coordinate derivatives stacked together]

== Affine forward: multiply componentwise, then add #D

Use $bold(x)=mat(2;-1)$, $bold(w)=mat(1;3)$, and $b=0$.
#pause
#align(center, table(
  columns: (52mm, 45mm, 45mm, 75mm), stroke: 0.5pt + MUTED,
  inset: (x: 12pt, y: 6pt), align: center,
  table.header([*Coordinate*], [$x_i$], [$w_i$], [$w_i x_i$]),
  [$1$], [$2$], [$1$], [$1 dot 2=2$],
  [$2$], [$-1$], [$3$], [$3 dot (-1)=-3$],
))
#pause
$ bold(w)^top bold(x) = 2 + (-3) = -1, quad z=-1+0=-1. $
#pause
$ cal(L)=z^2=(-1)^2=1. $
#pause
#result[dot product = multiply matching entries, then sum]

== First backward step: the square operation #D

#vectorseedgraph
#pause
Seed #upstream[$partial cal(L) \/ partial cal(L)=1$]. The square has #localterm[local derivative $2z=-2$], so
#align(center, text(size: 27pt, weight: 600)[
  #downstream[$(partial cal(L))/(partial z)$] $=$ #upstream[$1$] #localterm[$dot 2z$] $= -2$.
])
#pause
#result[the square returns $-2$; it now arrives at the affine operation]

== Affine backward: apply the three local gradients #D

#vectorbackgraph
#pause
Let #upstream[$g_z = partial cal(L) \/ partial z=-2$] arrive at the affine operation. Its #localterm[local gradients] send back:
#pause
#align(center, table(
  columns: (48mm, 72mm, 95mm), stroke: 0.5pt + MUTED,
  inset: (x: 10pt, y: 6pt), align: (left, center, center),
  [#text(fill: INK, weight: 700)[Input]], [#text(fill: LOCAL, weight: 700)[Local gradient]], [#text(fill: DOWNSTREAM, weight: 700)[Downstream gradient]],
  [$bold(w)$], [#localterm[$partial z \/ partial bold(w)=bold(x)$]], [#downstream[$partial cal(L) \/ partial bold(w)=(-2)bold(x)=(-4,2)^top$]],
  [$bold(x)$], [#localterm[$partial z \/ partial bold(x)=bold(w)$]], [#downstream[$partial cal(L) \/ partial bold(x)=(-2)bold(w)=(-2,-6)^top$]],
  [$b$], [#localterm[$partial z \/ partial b=1$]], [#downstream[$partial cal(L) \/ partial b=-2$]],
))
#pause
#result[one arriving gradient $g_z$ produces three separate downstream gradients]

== PyTorch gives the same vector gradients #I

#codebox(size: 13.5pt)[```python
import torch

x = torch.tensor([2., -1.], requires_grad=True)
w = torch.tensor([1.,  3.], requires_grad=True)
b = torch.tensor(0., requires_grad=True)

z = w @ x + b
L = z**2
L.backward()

print(z.item(), L.item())  # -1.0, 1.0
print(w.grad)              # tensor([-4.,  2.])
print(x.grad, b.grad)      # tensor([-2., -6.]), tensor(-2.)
```]
#pause
#result[`@` performs the dot product; `backward()` still applies the same local rules]

== Under the hood: the affine backward rule #D

For this one-output affine node, PyTorch sends the #upstream[upstream gradient]
`grad_z` into a backward rule. A readable teaching equivalent is:

#codebox(size: 12pt)[```python
class Affine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        ctx.save_for_backward(x, w)   # needed later
        return w @ x + b

    @staticmethod
    def backward(ctx, grad_z):        # grad_z = dL/dz
        x, w = ctx.saved_tensors
        grad_x = grad_z * w           # dL/dx = dL/dz * dz/dx
        grad_w = grad_z * x           # dL/dw = dL/dz * dz/dw
        grad_b = grad_z               # dL/db = dL/dz * 1
        return grad_x, grad_w, grad_b # one gradient per input
```]
#pause
#result[
  `backward` receives #upstream[$partial cal(L) / partial z$]
  and returns #downstream[$partial cal(L) / partial bold(x)$],
  #downstream[$partial cal(L) / partial bold(w)$], and
  #downstream[$partial cal(L) / partial b$]
]
#pause
#neat-caption[
  Teaching equivalent. `nn.Linear.forward` calls `F.linear`; production autograd
  uses generated native kernels. #link("https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py")[Linear source ↗]
  · #link("https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml")[derivative rules ↗]
]

// ═══════════════════════════ PART V — Dense layer ═══════════════════════════
= From one scalar output to a vector output

== One neuron still produces one scalar #V

Start with the affine rule we already know:
#align(center, text(size: 26pt, weight: 600)[
  $z=bold(w)^top bold(x)+b.$
])
#pause
#align(center, table(
  columns: (54mm, 54mm, 54mm, 54mm), stroke: 0.5pt + MUTED,
  inset: (x: 10pt, y: 7pt), align: center,
  table.header([*Quantity*], [$bold(w)^top$], [$bold(x)$], [$z$]),
  [*Shape*], [$(1 times d)$], [$(d times 1)$], [$(1 times 1)$],
))
#pause
#align(center, text(size: 23pt)[
  $(1 times d)(d times 1) = (1 times 1)$
])
#pause
#result[one weight row dotted with one input vector produces one number]

== Two neurons produce two scalars #V

Give each neuron its own weight row and bias; both read the same $bold(x)$.
#denseforwardgraph
#pause
#align(center, text(size: 22pt)[
  $z_1=bold(w)_1^top bold(x)+b_1, quad
  z_2=bold(w)_2^top bold(x)+b_2.$
])
#pause
#result[stack the two numbers into $bold(z)=mat(z_1;z_2)$]

== Stacking the two neurons gives one dense layer #D

#align(center, text(size: 22pt)[
  $bold(W)=mat(bold(w)_1^top;bold(w)_2^top), quad
  bold(b)=mat(b_1;b_2), quad
  bold(z)=bold(W)bold(x)+bold(b).$
])
#pause
#align(center, table(
  columns: (47mm, 47mm, 47mm, 47mm), stroke: 0.5pt + MUTED,
  inset: (x: 9pt, y: 7pt), align: center,
  table.header([$bold(W)$], [$bold(x)$], [$bold(b)$], [$bold(z)$]),
  [$(m times d)$], [$(d times 1)$], [$(m times 1)$], [$(m times 1)$],
))
#pause
#align(center, text(size: 22pt, weight: 600)[
  $(m times d)(d times 1)+(m times 1)=(m times 1)$
])
#pause
#notebox[$d$ is the number of input features; $m$ is the number of output neurons.]

== Work out output 1 #D

Use
#align(center, text(size: 22pt)[
  $bold(x)=mat(2;-1), quad bold(w)_1^top=(1,3), quad b_1=0.$
])
#pause
#align(center, text(size: 30pt, weight: 600)[
  $z_1=1(2)+3(-1)+0=-1.$
])
#pause
#align(center, text(size: 19pt, fill: MUTED)[
  $(1 times 2)(2 times 1)+(1 times 1)=(1 times 1)$
])
#pause
#result[multiply matching entries, add them, then add the bias]

== Work out output 2, then stack #D

The second neuron uses a different row:
#align(center, text(size: 22pt)[
  $bold(w)_2^top=(-2,1), quad b_2=1.$
])
#pause
#align(center, text(size: 29pt, weight: 600)[
  $z_2=(-2)(2)+1(-1)+1=-4.$
])
#pause
#align(center, text(size: 24pt)[
  $bold(z)=mat(z_1;z_2)=mat(-1;-4), quad
  bold(W)=mat(1,3;-2,1).$
])
#pause
#result[matrix notation only stacks the two familiar dot products]

== Backward starts with one arrival per output #D

Suppose the later loss returns
#align(center, text(size: 28pt, weight: 600)[
  #upstream[$bold(g)_z = partial cal(L)\/partial bold(z) = mat(4;-2)$].
])
#pause
Read this vector as two scalar messages:
#align(center, table(
  columns: (54mm, 72mm, 72mm), stroke: 0.5pt + MUTED,
  inset: (x: 10pt, y: 7pt), align: center,
  table.header([*Output*], [*Arriving gradient*], [*Meaning*]),
  [$z_1$], [#upstream[$g_1=partial cal(L)\/partial z_1=4$]], [nudge $z_1$ upward; loss rises at rate $4$],
  [$z_2$], [#upstream[$g_2=partial cal(L)\/partial z_2=-2$]], [nudge $z_2$ upward; loss falls at rate $2$],
))
#pause
#result[a vector gradient is simply one arriving number for each output coordinate]

== Backward through neuron 1: trace one scalar path #V

#dense-neuron1-focus
#pause
#align(center, text(size: 17.5pt)[
  $bold(W)_[1,:]=bold(w)_1^top=(1,3)$ #h(8pt) $bold(b)_[1]=b_1=0$
  #h(8pt) $bold(z)_[1]=z_1=-1$ #h(8pt) $bold(g)_z[1]=g_1=4$
])
#pause
#align(center, text(size: 16.5pt)[
  orange path to $bold(x)$: #downstream[$g_1 bold(w)_1 = 4(1,3)^top = (4,12)^top$]
])
#pause
#result[one matrix row and one vector coordinate recover the scalar neuron we already know]

== Neuron 1 returns gradients to its row, bias, and input #D

The highlighted node receives #upstream[$g_1=4$]. Reuse the scalar affine rule:
#pause
#align(center, table(
  columns: (58mm, 70mm, 76mm), stroke: 0.5pt + MUTED,
  inset: (x: 10pt, y: 7pt), align: center,
  table.header([*Recipient*], [*Local derivative*], [*Gradient returned*]),
  [$bold(w)_1$], [#localterm[$partial z_1\/partial bold(w)_1=bold(x)$]], [#downstream[$g_1 bold(x)=4mat(2;-1)=mat(8;-4)$]],
  [$b_1$], [#localterm[$partial z_1\/partial b_1=1$]], [#downstream[$g_1=4$]],
  [$bold(x)$], [#localterm[$partial z_1\/partial bold(x)=bold(w)_1$]], [#downstream[$g_1 bold(w)_1=4mat(1;3)=mat(4;12)$]],
))
#pause
#result[neuron 1 updates only row 1 of $bold(W)$, but it also returns a contribution to the shared input]

== Backward through neuron 2 #D

Neuron 2 receives #upstream[$g_2=-2$]. Apply the same rule again:
#pause
#align(center, table(
  columns: (58mm, 70mm, 76mm), stroke: 0.5pt + MUTED,
  inset: (x: 10pt, y: 7pt), align: center,
  table.header([*Recipient*], [*Local derivative*], [*Gradient returned*]),
  [$bold(w)_2$], [#localterm[$partial z_2\/partial bold(w)_2=bold(x)$]], [#downstream[$g_2 bold(x)=(-2)(2,-1)^top=(-4,2)^top$]],
  [$b_2$], [#localterm[$partial z_2\/partial b_2=1$]], [#downstream[$g_2=-2$]],
  [$bold(x)$], [#localterm[$partial z_2\/partial bold(x)=bold(w)_2$]], [#downstream[$g_2 bold(w)_2=(-2)(-2,1)^top=(4,-2)^top$]],
))
#pause
#result[neuron 2 updates row 2 and returns its own contribution to $bold(x)$]

== The shared input receives both contributions #D

The same $bold(x)$ fed both neurons, so its two backward paths add:
#pause
#align(center, text(size: 27pt, weight: 600)[
  #downstream[$bold(g)_x$]
  $=underbrace(mat(4;12), "from neuron 1")+underbrace(mat(4;-2), "from neuron 2")
  =mat(8;10).$
])
#pause
#align(center, text(size: 22pt)[
  #downstream[$bold(g)_x$] $=$ #localterm[$bold(W)^top$] #upstream[$bold(g)_z$]
])
#pause
#align(center, text(size: 19pt, fill: MUTED)[
  $(2 times 1)=(2 times 2)(2 times 1)$
])
#pause
#notebox[The transpose lines up the two output contributions for each input coordinate; the matrix multiplication performs the required addition.]

== Parameter gradients keep the two rows separate #D

#align(center, text(size: 23pt)[
  #downstream[$bold(g)_W$]
  $=mat(8,-4;-4,2), quad
  #downstream[$bold(g)_b$]=mat(4;-2).$
])
#pause
The compact rules are
#align(center, text(size: 25pt, weight: 600)[
  #downstream[$bold(g)_W$] $=$ #upstream[$bold(g)_z$] #localterm[$bold(x)^top$] #h(24pt)
  #downstream[$bold(g)_b$] $=$ #upstream[$bold(g)_z$].
])
#pause
#align(center, text(size: 19pt, fill: MUTED)[
  $(2 times 2)=(2 times 1)(1 times 2), quad (2 times 1)=(2 times 1)$
])
#pause
#result[one output coordinate creates one row of the weight gradient]

== Dense backward: three rules, with shapes #V

#align(center, table(
  columns: (35mm, 88mm, 78mm), stroke: 0.5pt + MUTED,
  inset: (x: 9pt, y: 7pt), align: (left, center, center),
  table.header([*Returned to*], [*Rule*], [*Shape*]),
  [$bold(x)$], [#downstream[$bold(g)_x$] $=$ #localterm[$bold(W)^top$] #upstream[$bold(g)_z$]], [$(d times 1)=(d times m)(m times 1)$],
  [$bold(W)$], [#downstream[$bold(g)_W$] $=$ #upstream[$bold(g)_z$] #localterm[$bold(x)^top$]], [$(m times d)=(m times 1)(1 times d)$],
  [$bold(b)$], [#downstream[$bold(g)_b$] $=$ #upstream[$bold(g)_z$]], [$(m times 1)=(m times 1)$],
))
#pause
#notebox[Do the scalar reasoning first. Use the shapes as a check that the stacked formula has the correct orientation.]

== PyTorch stores gradients in matching shapes #I

#codebox(size: 12.5pt)[```python
import torch

x = torch.tensor([2., -1.], requires_grad=True)       # (2,)
W = torch.tensor([[1., 3.], [-2., 1.]],
                 requires_grad=True)                  # (2, 2)
b = torch.tensor([0., 1.], requires_grad=True)        # (2,)

z = W @ x + b                                         # (2,)
z.backward(torch.tensor([4., -2.]))                   # arriving g_z

print(x.grad)  # tensor([ 8., 10.])
print(W.grad)  # tensor([[ 8., -4.], [-4.,  2.]])
print(b.grad)  # tensor([ 4., -2.])
```]
#pause
#result[every tensor and its gradient have the same shape]

= From one example to a batch

== A batch is a stack of examples #V

Use two examples so every calculation remains visible:
#align(center, text(size: 25pt, weight: 600)[
  $bold(X)=mat(2,-1;-1,2).$
])
#pause
#align(center, table(
  columns: (42mm, 60mm, 82mm), stroke: 0.5pt + MUTED,
  inset: (x: 10pt, y: 7pt), align: center,
  table.header([*Row*], [*Example*], [*Meaning*]),
  [1], [$bold(x)^(1)^top=(2,-1)$], [first training example],
  [2], [$bold(x)^(2)^top=(-1,2)$], [second training example],
))
#pause
#align(center, text(size: 21pt)[
  $bold(X) in RR^(B times d)=RR^(2 times 2)$
])
#pause
#result[the new first axis counts examples; it does not create new parameters]

== The same dense layer processes both rows #D

Keep the same parameters
#align(center, text(size: 22pt)[
  $bold(W)=mat(1,3;-2,1), quad bold(b)=mat(0;1).$
])
#pause
#align(center, text(size: 24pt, weight: 600)[
  $bold(Z)=bold(X)bold(W)^top+bold(1)_B bold(b)^top
  =mat(-1,-4;5,5).$
])
#pause
#align(center, text(size: 19pt, fill: MUTED)[
  $(B times m)=(B times d)(d times m)+(B times 1)(1 times m)$
])
#pause
#align(center, text(size: 20pt)[
  row 1: $(2,-1)bold(W)^top+bold(b)^top=(-1,-4)$ #linebreak()
  row 2: $(-1,2)bold(W)^top+bold(b)^top=(5,5)$
])
#pause
#result[each row runs the same forward calculation with the same $bold(W),bold(b)$]

== Two outputs mean a length-2 target per example #D

Let $n in {1,2}$ be the *example (row) index*; the parenthesized superscript
$(n)$ is a label, not a power. Here $m=2$, so $bold(y)^(n) in RR^2$.
#align(center, text(size: 22pt)[
  $bold(Y)=mat((bold(y)^(1))^top;(bold(y)^(2))^top)=mat(-5,-2;7,1)
  in RR^(B times m)=RR^(2 times 2).$
])
#pause
#align(center, table(
  columns: (43mm, 58mm, 58mm, 66mm), stroke: 0.5pt + MUTED,
  inset: (x: 8pt, y: 7pt), align: center,
  table.header([*Example index $n$*], [$bold(z)^(n)$], [$bold(y)^(n)$], [$bold(r)^(n)=bold(z)^(n)-bold(y)^(n)$]),
  [1], [$(-1,-4)$], [$(-5,-2)$], [#upstream[$(4,-2)$]],
  [2], [$(5,5)$], [$(7,1)$], [#upstream[$(-2,4)$]],
))
#pause
#two(
  align(center, text(size: 19pt)[
    $n=1: quad ell_1=1/2(4^2+(-2)^2)=10$
  ]),
  align(center, text(size: 19pt)[
    $n=2: quad ell_2=1/2((-2)^2+4^2)=10$
  ]),
)
#pause
#result[each $bold(r)^(n)$ is a vector; squaring and summing its coordinates gives scalar $ell_n$]
#neat-caption[Scalar-output special case: if $m=1$, each target $y^(n)$ is a scalar and $bold(Y)$ has shape $B times 1$.]

== The mean loss scales both arriving gradients #D

#align(center, text(size: 26pt, weight: 600)[
  $cal(L)=1/2(ell_1+ell_2)=10.$
])
#pause
Therefore
#align(center, text(size: 26pt)[
  #upstream[$bold(G)_Z=partial cal(L)\/partial bold(Z)$]
  $=1/2 mat(4,-2;-2,4)=mat(2,-1;-1,2).$
])
#pause
#align(center, text(size: 19pt, fill: MUTED)[
  $bold(G)_Z in RR^(B times m)=RR^(2 times 2)$
])
#pause
#notebox[Rows still correspond to examples; columns still correspond to output neurons. The factor $1/B$ comes only from taking the mean.]

== Example 1 proposes a parameter gradient #D

Before averaging, example 1 has
$bold(r)^(1)=mat(4;-2)$ and $bold(x)^(1)=mat(2;-1)$.
#pause
#align(center, text(size: 25pt, weight: 600)[
  $bold(G)_W^(1)=bold(r)^(1)(bold(x)^(1))^top
  =mat(8,-4;-4,2).$
])
#pause
#align(center, text(size: 19pt, fill: MUTED)[
  $(m times d)=(m times 1)(1 times d)$
])
#pause
#align(center, text(size: 22pt)[
  $bold(g)_b^(1)=bold(r)^(1)=mat(4;-2).$
])
#pause
#result[this is the same single-example outer product derived on the previous slides]

== Example 2 proposes another gradient #D

Example 2 has
$bold(r)^(2)=mat(-2;4)$ and $bold(x)^(2)=mat(-1;2)$.
#pause
#align(center, text(size: 25pt, weight: 600)[
  $bold(G)_W^(2)=bold(r)^(2)(bold(x)^(2))^top
  =mat(2,-4;-4,8).$
])
#pause
Both examples used the same parameters, so their paths add; the mean divides by $2$:
#align(center, text(size: 24pt)[
  #downstream[$bold(g)_W$]
  $=1/2(bold(G)_W^(1)+bold(G)_W^(2))
  =mat(5,-4;-4,5).$
])
#pause
#align(center, text(size: 22pt)[
  #downstream[$bold(g)_b$] $=1/2(mat(4;-2)+mat(-2;4))=mat(1;1).$
])

== Batch matrix multiplication performs the same sum #D

Let $bold(G)_Z=bold(R)/B$. Then
#align(center, text(size: 25pt, weight: 600)[
  #downstream[$bold(g)_W$] $=bold(G)_Z^top bold(X).$
])
#pause
#align(center, text(size: 19pt, fill: MUTED)[
  $(m times d)=(m times B)(B times d)$
])
#pause
#align(center, text(size: 23pt)[
  $mat(2,-1;-1,2) mat(2,-1;-1,2)=mat(5,-4;-4,5).$
])
#pause
#notebox[The middle dimension $B$ disappears because matrix multiplication sums the contribution from every example.]

== Batch gradients have one simple shape story #V

#align(center, table(
  columns: (34mm, 88mm, 79mm), stroke: 0.5pt + MUTED,
  inset: (x: 9pt, y: 7pt), align: (left, center, center),
  table.header([*Returned to*], [*Rule*], [*Shape*]),
  [$bold(W)$], [#downstream[$bold(g)_W$] $=bold(G)_Z^top bold(X)$], [$(m times d)=(m times B)(B times d)$],
  [$bold(b)$], [#downstream[$bold(g)_b$] $=bold(G)_Z^top bold(1)_B$], [$(m times 1)=(m times B)(B times 1)$],
  [$bold(X)$], [#downstream[$bold(g)_X$] $=bold(G)_Z bold(W)$], [$(B times d)=(B times m)(m times d)$],
))
#pause
#align(center, text(size: 22pt)[
  #downstream[$bold(g)_X$] $=mat(4,5;-5,-1).$
])
#pause
#result[shared parameter gradients add across rows; input gradients remain one row per example]

== Sum and mean differ only by scale #D

#align(center, table(
  columns: (54mm, 78mm, 78mm), stroke: 0.5pt + MUTED,
  inset: (x: 10pt, y: 8pt), align: center,
  table.header([*Reduction*], [*Arriving gradient*], [*Weight gradient*]),
  [sum], [$bold(G)_Z=bold(R)$], [$bold(g)_W=bold(R)^top bold(X)$],
  [mean], [$bold(G)_Z=bold(R)/B$], [$bold(g)_W=bold(R)^top bold(X)/B$],
))
#pause
#codebox(size: 15pt)[```python
loss = 0.5 * ((Z - Y)**2).sum(dim=1).mean()
# sum output coordinates inside each example; mean over examples
```]
#pause
#result[the dense-layer rule is unchanged; the final scalar reduction chooses the scale]

== PyTorch reproduces the two-example calculation #I

#codebox(size: 12.2pt)[```python
X = torch.tensor([[ 2., -1.], [-1., 2.]], requires_grad=True)  # (B=2, d=2)
Y = torch.tensor([[-5., -2.], [ 7., 1.]])                     # (2, 2)
W = torch.tensor([[1., 3.], [-2., 1.]], requires_grad=True)   # (m=2, d=2)
b = torch.tensor([0., 1.], requires_grad=True)                # (2,)

Z = X @ W.T + b                                               # (B, m)
loss = 0.5 * ((Z - Y)**2).sum(dim=1).mean()                   # scalar
loss.backward()

print(loss.item())  # 10.0
print(W.grad)        # [[ 5., -4.], [-4., 5.]]
print(b.grad)        # [1., 1.]
print(X.grad)        # [[ 4., 5.], [-5., -1.]]
```]
#pause
#result[`backward()` carries out the same rowwise contributions and additions]

== The training loop repeats this batch calculation #I

#codebox(size: 13pt)[```python
for X, Y in loader:
    optimizer.zero_grad()       # discard gradients from the previous update
    Z = model(X)                # batch forward: (B, d) -> (B, m)
    loss = loss_fn(Z, Y)        # reduce B examples to one scalar
    loss.backward()             # add this batch's parameter contributions
    optimizer.step()            # update parameters once
```]
#pause
#align(center, text(size: 20pt)[
  clear $arrow.r$ forward $arrow.r$ scalar loss $arrow.r$ backward $arrow.r$ update
])
#pause
#notebox[`step()` uses the gradients; it does not compute or clear them. That is why the next iteration begins with `zero_grad()`.]

== Microbatches split one batch without changing the goal #D

#codebox(size: 13.5pt)[```python
optimizer.zero_grad()
(loss_example_1 / 2).backward()   # first half of the mean gradient
(loss_example_2 / 2).backward()   # second half accumulates
optimizer.step()
```]
#pause
#align(center, text(size: 22pt)[
  $bold(W)."grad": quad bold(G)_W^(1)/2
  arrow.r (bold(G)_W^(1)+bold(G)_W^(2))/2$
])
#pause
#notebox[For equivalence, keep parameters fixed, preserve the same overall scaling, and do not call `zero_grad()` or `step()` between pieces.]
#pause
#result[one update window may contain one full batch or several correctly scaled pieces]

// ═══════════════════════════ PART VI — MLP ═══════════════════════════
= A tiny neural network

== ReLU is what prevents two dense layers from collapsing into one #V

Without a nonlinearity, two affine maps are still one affine map:
$ bold(W)_2(bold(W)_1 bold(x)+bold(b)_1)+bold(b)_2 $
$ quad = (bold(W)_2 bold(W)_1)bold(x)+(bold(W)_2 bold(b)_1+bold(b)_2). $
#pause
#relumotivationgraph
#pause
#result[ReLU makes the composition nonlinear; depth can now represent a richer function]

== A tiny MLP composes three vector blocks #V

#mlpforward
#pause
#align(center, table(
  columns: (45mm, 72mm, 76mm), stroke: 0.5pt + MUTED,
  inset: (x: 9pt, y: 6pt), align: center,
  table.header([*Value*], [*Rule*], [*Shape*]),
  [$bold(z)_1$], [$bold(W)_1bold(x)+bold(b)_1$], [$(h times d)(d times 1)+(h times 1)=h times 1$],
  [$bold(h)$], [$"ReLU"(bold(z)_1)$], [$h times 1$],
  [$bold(hat(y))$], [$bold(W)_2bold(h)+bold(b)_2$], [$(m times h)(h times 1)+(m times 1)=m times 1$],
  [$cal(L)$], [$sum_i (hat(y)_i-y_i)^2$], [scalar],
))
#pause
#result[$d$ input features → $h$ hidden features → $m$ output coordinates → one loss]

== MLP forward, one block at a time #D

#align(center, text(size: 24pt, weight: 600)[
  $bold(z)_1=bold(W)_1bold(x)+bold(b)_1 in RR^(h times 1)$
])
#pause
#align(center, text(size: 24pt, weight: 600)[
  $bold(h)="ReLU"(bold(z)_1) in RR^(h times 1)$
])
#pause
#align(center, text(size: 24pt, weight: 600)[
  $bold(hat(y))=bold(W)_2bold(h)+bold(b)_2 in RR^(m times 1)$
])
#pause
#align(center, text(size: 24pt, weight: 600)[
  $cal(L)=sum_i (hat(y)_i-y_i)^2 in RR$
])
#pause
#result[forward changes the representation shape; the loss finally collapses it to one number]

== Backward, part 1: loss → predictions #D

#mlpbackward-focus("loss")
#pause
*1. Squared loss.* Start with the seed $partial cal(L)\/partial cal(L)=1$. The loss operation returns
#align(center, text(size: 21pt)[
  #downstream[$bold(g)_(hat(y))$] $= 2(bold(hat(y))-bold(y)).$
])
#align(center, text(size: 17pt, fill: MUTED)[$bold(g)_(hat(y)) in RR^(m times 1)$])
#pause
#notebox[There is one returned number for each prediction coordinate: positive means increasing that prediction increases the loss; negative means it decreases the loss.]
#pause
#result[$bold(g)_(hat(y))$ now arrives at the second dense layer]

== Backward, part 2: dense 2 → hidden activations #D

#mlpbackward-focus("dense2")
#pause
*2. Second dense layer.* It receives #upstream[$bold(g)_(hat(y)) in RR^(m times 1)$] and returns a gradient to its input $bold(h)$:
#align(center, text(size: 21pt)[
  #downstream[$bold(g)_h$] $=$ #localterm[$bold(W)_2^top$] #upstream[$bold(g)_(hat(y))$].
])
#align(center, text(size: 17pt, fill: MUTED)[$(h times 1)=(h times m)(m times 1)$])
#pause
It also returns parameter gradients
#align(center, text(size: 19pt)[
  $partial cal(L)\/partial bold(W)_2=bold(g)_(hat(y))bold(h)^top in RR^(m times h), quad
  partial cal(L)\/partial bold(b)_2=bold(g)_(hat(y)) in RR^(m times 1).$
])
#pause
#result[the returned $bold(g)_h$ is now the arriving gradient at ReLU]

== Backward, part 3: ReLU → dense 1 → input #D

#mlpbackward-focus("left")
#pause
*3. ReLU.* Apply its local gate coordinate by coordinate:
#align(center, text(size: 21pt)[
  #downstream[$bold(g)_(z_1)$] $=$ #upstream[$bold(g)_h$] #localterm[$dot.o bb(1)[bold(z)_1>0]$].
])
#align(center, text(size: 17pt, fill: MUTED)[$(h times 1)=(h times 1) dot.o (h times 1)$])
#pause
*4. First dense layer.* It receives #upstream[$bold(g)_(z_1)$] and returns
#align(center, text(size: 20pt)[
  #downstream[$bold(g)_x$] $=$ #localterm[$bold(W)_1^top$] #upstream[$bold(g)_(z_1)$], $quad$
  #downstream[$partial cal(L)\/partial bold(W)_1$] $=$ #upstream[$bold(g)_(z_1)$] #localterm[$bold(x)^top$], $quad$
  #downstream[$partial cal(L)\/partial bold(b)_1$] $=$ #upstream[$bold(g)_(z_1)$].
])
#align(center, text(size: 16.5pt, fill: MUTED)[
  $bold(g)_x: d times 1 quad dot quad partial cal(L)\/partial bold(W)_1: h times d quad dot quad partial cal(L)\/partial bold(b)_1: h times 1$
])
#pause
#result[each returned gradient becomes the arriving gradient for the block immediately to its left]

== The same tiny MLP in PyTorch #I

#codebox(size: 12.5pt)[```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2)
)
x = torch.tensor([2., -1.])
y = torch.tensor([1., 0.])

y_hat = model(x)
loss = ((y_hat - y)**2).sum()
loss.backward()

for name, p in model.named_parameters():
    print(name, tuple(p.shape), tuple(p.grad.shape))
```]
#pause
#result[PyTorch records the three blocks and applies their local rules in reverse]

== Interactive: follow one vector gradient #I

#interbox(link-to: IA + "autograd")[
  Step through the forward values, then follow one arriving gradient through a dense operation, a ReLU, and the preceding dense operation.
]
#pause
For practice, predict each gradient's shape before revealing its value in the interactive.

// ═══════════════════════════ PART VIII — Flow through depth ═══════════════════════════
= When gradients cross many layers

== Deep networks repeat the same multiplication #D

For a scalar chain $h_0 arrow.r h_1 arrow.r dots arrow.r h_L arrow.r cal(L)$:
#pause
#deepchaingraph
#pause
$ (partial cal(L))/(partial h_0) = (partial cal(L))/(partial h_L) dot f_L'(h_(L-1)) dots f_1'(h_0). $
#pause
#result[depth creates a product of many local slopes]
#pause
#notebox[Vector layers do the same operation with matrix–vector products. The intuition is unchanged: repeatedly small factors shrink a gradient; repeatedly large factors grow it.]

== In a vector layer, the returned shape matches the previous value #D

For layer $ell$,
#align(center, text(size: 23pt)[
  $bold(z)_ell=bold(W)_ell bold(h)_(ell-1)+bold(b)_ell.$
])
#pause
#align(center, table(
  columns: (50mm, 70mm, 78mm), stroke: 0.5pt + MUTED,
  inset: (x: 9pt, y: 7pt), align: center,
  table.header([*Quantity*], [*Shape*], [*Backward role*]),
  [$bold(g)_(z_ell)$], [$n_ell times 1$], [arrives from the layer to the right],
  [$bold(W)_ell^top$], [$n_(ell-1) times n_ell$], [local linear map],
  [$bold(g)_(h_(ell-1))$], [$n_(ell-1) times 1$], [returns to the previous layer],
))
#pause
#align(center, text(size: 24pt, weight: 600)[
  #downstream[$bold(g)_(h_(ell-1))$]
  $=$ #localterm[$bold(W)_ell^top$] #upstream[$bold(g)_(z_ell)$].
])
#pause
#result[each layer receives a gradient shaped like its output and returns one shaped like its input]

== Ten ordinary-looking slopes can create three regimes #V

Start with an arriving loss derivative of $1$ and multiply the same local slope ten times:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 15pt, y: 8pt), align: center,
  [#text(fill: LOCAL, weight: 700)[Local slope]], [#text(fill: INK, weight: 700)[After 10 layers]], [#text(fill: INK, weight: 700)[Effect]],
  [#localterm[$0.5$]], [$0.5^10 approx 0.001$], [gradient nearly disappears],
  [#localterm[$1.0$]], [$1^10 = 1$], [gradient is preserved],
  [#localterm[$1.5$]], [$1.5^10 approx 57.7$], [gradient grows rapidly],
))
#pause
#result[the chain rule explains both vanishing and exploding gradients]

== Sigmoid saturation #V

$sigma'(z) = sigma(z)(1 - sigma(z)) <= 1/4$, and $approx 0$ for large $|z|$:
#pause
#fig("/lecture3/figures/sigmoid_deriv.svg", w: 40%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[each sigmoid layer multiplies the gradient by at most $1 \/ 4$])
#pause
#notebox[Repeated sigmoid derivatives can shrink the signal quickly. ReLU has slope $1$ on active units and $0$ on inactive units; weights and normalization also shape the full Jacobian.]

== Interactive: gradient flow #I

#interbox(link-to: IA + "vanishing-gradients")[
  Sweep depth and activation function; watch the gradient magnitude at layer $0$ collapse for sigmoid and survive for ReLU.
]
#pause
Sweep depth to see repeated small local slopes rapidly erase an early-layer gradient.

// ═══════════════════════════ PART IX — Summary ═══════════════════════════
= Summary

== Backprop in one sentence

#align(center, text(size: 22pt)[
  Seed $partial cal(L) \/ partial cal(L) = 1$, then walk the graph *backward*, at each operation computing
  #v(6pt)
  #downstream[*downstream gradient*] $=$ #upstream[*upstream gradient*] $times$ #localterm[*local derivative*],
  #v(6pt)
  and *adding* gradients wherever a variable branched.
])
#pause
#align(center, text(size: 16pt, fill: MUTED)[For vector operations, the #localterm[local derivative] can be a vector or matrix; the same color-coded chain rule applies.])

== The formulas to keep #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 7pt), align: (left, left),
  table.header([*Setting*], [*Backward rule*]),
  [scalar operation],   [#downstream[$(partial cal(L))/(partial u)$] $=$ #upstream[$(partial cal(L))/(partial v)$] #localterm[$dot partial v \/ partial u$]],
  [dense layer],        [#downstream[$(partial cal(L))/(partial bold(W))$] $=$ #upstream[$(partial cal(L))/(partial bold(z))$] #localterm[$bold(x)^top$]; #downstream[$(partial cal(L))/(partial bold(x))$] $=$ #localterm[$bold(W)^top$] #upstream[$(partial cal(L))/(partial bold(z))$]],
  [activation],         [#downstream[$(partial cal(L))/(partial bold(z))$] $=$ #upstream[$(partial cal(L))/(partial bold(a))$] #localterm[$dot.o phi'(bold(z))$]],
  [branch point],       [gradients *add*],
))

#focus-slide[
  We can compute $partial cal(L) \/ partial theta_j$ for every parameter.
  #v(12pt)
  #set text(size: 22pt)
  Next: turn those gradients into reliable parameter updates.
  #v(8pt)
  #text(size: 17pt, fill: rgb("#B9C6C8"))[Lecture 5 · Optimization for Deep Learning]
]
