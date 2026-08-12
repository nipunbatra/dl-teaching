// Modern CNN Pipelines and Transfer Learning — Lecture 8B · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture8b/L8b-modern-cnns.typ /tmp/L8b.pdf
//   typst compile --root . --input handout=true lecture8b/L8b-modern-cnns.typ /tmp/L8bh.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Modern CNN Pipelines and Transfer Learning],
  subtitle: [Blocks, Scaling, Residuals, and Reusing Backbones],
)

#let NB1 = "https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L08B/01_depthwise_separable_and_transfer_head.ipynb"
#let VGG = "https://arxiv.org/abs/1409.1556"
#let INCEPTION = "https://arxiv.org/abs/1409.4842"
#let RESNET = "https://arxiv.org/abs/1512.03385"
#let MOBILENET = "https://arxiv.org/abs/1704.04861"
#let EFFICIENTNET = "https://arxiv.org/abs/1905.11946"
#let CONVNEXT = "https://arxiv.org/abs/2201.03545"
#let FPN = "https://arxiv.org/abs/1612.03144"

// Stable semantic grammar:
// INK = held input / known quantity · TEAL = learned operation / parameter
// BLUE = spatial geometry / context · ACC = compute / task adaptation
// GREEN = verified result / within budget · RED = failure / violated contract.
#let swatch(color, body) = box(width: 13pt, height: 4pt, fill: color, radius: 1pt) + h(4pt) + body
#let semantic-legend = align(center, text(size: 12.5pt)[
  #swatch(INK, [held input]) #h(12pt)
  #swatch(TEAL, [learned operation]) #h(12pt)
  #swatch(BLUE, [context / shape]) #h(12pt)
  #swatch(ACC, [compute / adaptation]) #h(12pt)
  #swatch(GREEN, [verified]) #h(12pt)
  #swatch(RED, [failure])
])

#let hairline(title, body, color: TEAL) = block(
  width: 100%, inset: 3pt,
  [#text(size: 13.5pt, weight: 700, fill: color)[#upper(title)]
   #v(2pt)
   #line(length: 100%, stroke: 0.8pt + color)
   #v(5pt)
   #body],
)

#let card(title, body, color: TEAL) = block(
  width: 100%, inset: (x: 10pt, y: 8pt), fill: white,
  stroke: 1pt + color, radius: 3pt,
  [#text(size: 12.5pt, weight: 700, fill: color)[#upper(title)]
   #v(4pt)
   #body],
)

#let note(body, color: TEAL) = block(
  width: 100%, inset: (left: 11pt, right: 4pt, top: 5pt, bottom: 5pt),
  stroke: (left: 2pt + color), fill: color.lighten(93%), radius: 2pt,
  [#body],
)

// This deck's tag body stays above the 7.5pt audit floor while retaining the
// same shared theme grammar.
#let _l8b-chip(lbl, word, col) = box(
  fill: col, inset: (x: 7pt, y: 3.5pt), radius: 4pt, baseline: 0.28em,
  text(font: "IBM Plex Mono", size: 12pt, fill: white, weight: 700, tracking: 0.4pt,
    [#lbl#h(5pt)#text(size: 9pt, weight: 500, tracking: 0.8pt)[#upper(word)]])
)
#let Q = [#h(1fr)#_l8b-chip("Q", "question", BLUE)]
#let A = [#h(1fr)#_l8b-chip("A", "answer", GREEN)]
#let V = [#h(1fr)#_l8b-chip("V", "visual", TEAL)]
#let D = [#h(1fr)#_l8b-chip("D", "derivation", ACC)]
#let I = [#h(1fr)#_l8b-chip("I", "interactive", GREEN)]
#let OPT = [#h(1fr)#_l8b-chip(sym.star.filled, "optional", MUTED)]

#let case-strip(stage, detail) = block(
  width: 100%, inset: (x: 10pt, y: 6pt),
  fill: rgb("#F3F6F5"), stroke: (left: 3pt + TEAL), radius: 3pt,
  [#text(size: 11pt, weight: 700, fill: TEAL, tracking: 0.7pt)[HELD CASE · #upper(stage)]
   #h(8pt)
   #text(size: 13.5pt, fill: INK)[#detail]],
)

#let ledger(rows, note: none) = {
  align(center, table(
    columns: (42mm, 35mm, 21mm, 31mm, 30mm),
    stroke: 0.45pt + MUTED, inset: (x: 7pt, y: 5pt), align: (left, center, center, center, center),
    table.header([*candidate*], [*output*], [*RF*], [*weights*], [*MACs*]),
    ..rows,
  ))
  if note != none { v(5pt); align(center, text(size: 12.5pt, fill: MUTED)[#note]) }
}

// ── the modern conv primitive: Conv -> Norm -> Activation ──
#let convblock = align(center, diagram(spacing: 13mm, node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), text(size: 15pt)[Conv], fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL, corner-radius: 3pt, inset: 8pt)
  node((1,0), text(size: 15pt)[Norm], fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE, corner-radius: 3pt, inset: 8pt)
  node((2,0), text(size: 15pt)[Activation], fill: ACC.lighten(82%), stroke: 0.9pt + ACC, corner-radius: 3pt, inset: 8pt)
  for i in range(2) { edge((i,0),(i+1,0), "-|>", stroke: 0.8pt + MUTED) }
}))

// ── Inception module: input fans to four branches -> concat ──
#let inception = align(center, diagram(spacing: (11mm, 6mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let blk(p, lbl, col) = node(p, text(size: 12pt, lbl), fill: col.lighten(82%), stroke: 0.9pt + col, corner-radius: 3pt, inset: 5pt)
  node((0, 0), text(size: 13pt)[input], radius: 9mm, fill: rgb("#EFEEEB"))
  // four parallel branches at four heights
  blk((2, 3), "1×1", TEAL)
  blk((1.4, 1), "1×1", BLUE); blk((2.9, 1), "3×3", TEAL)
  blk((1.4, -1), "1×1", BLUE); blk((2.9, -1), "5×5", TEAL)
  blk((1.4, -3), "3×3 pool", MUTED); blk((2.9, -3), "1×1", TEAL)
  node((4.3, 0), text(size: 13pt)[concat], radius: 10mm, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  // edges from input
  edge((0,0), (2,3), "-|>", stroke: 0.7pt + MUTED)
  edge((0,0), (1.4,1), "-|>", stroke: 0.7pt + MUTED)
  edge((0,0), (1.4,-1), "-|>", stroke: 0.7pt + MUTED)
  edge((0,0), (1.4,-3), "-|>", stroke: 0.7pt + MUTED)
  // within-branch chains
  edge((1.4,1), (2.9,1), "-|>", stroke: 0.7pt + MUTED)
  edge((1.4,-1), (2.9,-1), "-|>", stroke: 0.7pt + MUTED)
  edge((1.4,-3), (2.9,-3), "-|>", stroke: 0.7pt + MUTED)
  // to concat
  edge((2,3), (4.3,0), "-|>", stroke: 0.7pt + MUTED)
  edge((2.9,1), (4.3,0), "-|>", stroke: 0.7pt + MUTED)
  edge((2.9,-1), (4.3,0), "-|>", stroke: 0.7pt + MUTED)
  edge((2.9,-3), (4.3,0), "-|>", stroke: 0.7pt + MUTED)
}))

// ── ResNet residual block: conv-norm-relu-conv-norm -> (+) -> relu, identity shortcut ──
#let resblock = align(center, diagram(spacing: (10mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let blk(x, lbl, col) = node((x, 0), text(size: 11.5pt, lbl), fill: col.lighten(82%), stroke: 0.9pt + col, corner-radius: 3pt, inset: 5pt)
  node((0, 0), $x_ell$, radius: 6mm, fill: rgb("#EFEEEB"))
  blk(1, "conv", TEAL); blk(2, "norm", BLUE); blk(3, "ReLU", ACC)
  blk(4, "conv", TEAL); blk(5, "norm", BLUE)
  node((6, 0), [$+$], radius: 5mm, fill: white, stroke: 1pt + INK)
  node((7, 0), text(size: 11.5pt)[ReLU], fill: ACC.lighten(82%), stroke: 0.9pt + ACC, corner-radius: 3pt, inset: 5pt)
  node((8, 0), $x_(ell+1)$, radius: 7mm, stroke: 0.9pt + ACC)
  for i in range(6) { edge((i,0),(i+1,0), "-|>", stroke: 0.8pt + MUTED) }
  edge((6,0),(7,0), "-|>", stroke: 0.8pt + MUTED)
  edge((7,0),(8,0), "-|>", stroke: 0.8pt + MUTED)
  // identity shortcut: x_l over the top, into the (+)
  edge((0,0), (0,-1.4), (6,-1.4), (6,0), "-|>", stroke: 1.1pt + GREEN)
  node((3, -1.4), text(size: 11pt, fill: GREEN)[identity  $x_ell$], stroke: none, fill: none)
}))

// ── the modern classification pipeline ──
#let pipeline = align(center, diagram(spacing: (7.5mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let blk(x, lbl, col) = node((x, 0), text(size: 11.5pt, lbl), fill: col.lighten(85%), stroke: 0.9pt + col, corner-radius: 3pt, inset: 5pt)
  blk(0, "data", INK); blk(1, "augment", BLUE); blk(2, "backbone", TEAL)
  blk(3, "head", ACC); blk(4, "loss", RED); blk(5, "optim", GREEN)
  blk(6, "sched", BLUE); blk(7, "val", TEAL); blk(8, "ckpt", INK)
  for i in range(8) { edge((i,0),(i+1,0), "-|>", stroke: 0.7pt + MUTED) }
}))

// ── backbone / head abstraction: one backbone, three task heads ──
#let backboneheads = align(center, diagram(spacing: (16mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), [image $x$], fill: rgb("#EFEEEB"), stroke: 0.9pt + INK, corner-radius: 3pt, inset: 8pt)
  node((1, 0), align(center, [backbone \ #text(size: 11pt, fill: MUTED)[$F = f(x)$]]), fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL, corner-radius: 3pt, inset: 8pt)
  node((2.2, 1.3), text(size: 12pt)[classification], fill: ACC.lighten(85%), stroke: 0.9pt + ACC, corner-radius: 3pt, inset: 6pt)
  node((2.2, 0), text(size: 12pt)[detection], fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE, corner-radius: 3pt, inset: 6pt)
  node((2.2, -1.3), text(size: 12pt)[segmentation], fill: GREEN.lighten(85%), stroke: 0.9pt + GREEN, corner-radius: 3pt, inset: 6pt)
  edge((0,0), (1,0), "-|>", stroke: 0.8pt + MUTED)
  edge((1,0), (2.2,1.3), "-|>", stroke: 0.8pt + MUTED)
  edge((1,0), (2.2,0), "-|>", stroke: 0.8pt + MUTED)
  edge((1,0), (2.2,-1.3), "-|>", stroke: 0.8pt + MUTED)
}))

// ── held-case bottleneck: 64 -> 16 -> 16 -> 128 ──
#let bottleneck = align(center, diagram(spacing: (11mm, 7mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), $x$, radius: 5mm, fill: rgb("#EFEEEB"))
  node((1, 0), align(center, [$1 times 1$ \ #text(size: 10pt, fill: MUTED)[$64 -> 16$]]), fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE, corner-radius: 3pt, inset: 6pt)
  node((2, 0), align(center, [$3 times 3$ \ #text(size: 10pt, fill: MUTED)[$16 -> 16$]]), fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL, corner-radius: 3pt, inset: 6pt)
  node((3, 0), align(center, [$1 times 1$ \ #text(size: 10pt, fill: MUTED)[$16 -> 128$]]), fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE, corner-radius: 3pt, inset: 6pt)
  node((4, 0), $y$, radius: 5mm, stroke: 0.9pt + ACC)
  for i in range(4) { edge((i,0),(i+1,0), "-|>", stroke: 0.8pt + MUTED) }
  node((1, 1.5), text(size: 10pt, fill: MUTED)[squeeze], stroke: none, fill: none)
  node((2, 1.5), text(size: 10pt, fill: MUTED)[cheap spatial], stroke: none, fill: none)
  node((3, 1.5), text(size: 10pt, fill: MUTED)[expand], stroke: none, fill: none)
}))

#title-slide()

#let _small-vs-large = align(center, diagram(spacing: (17mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let blk(p, body, col) = node(p, body, fill: col.lighten(86%), stroke: 0.9pt + col, corner-radius: 3pt, inset: 7pt)
  node((0, 0), [$X$], radius: 6mm, fill: rgb("#EFEEEB"))
  blk((1, 1.2), align(center, [$3 times 3$ \ #text(size: 10pt, fill: MUTED)[ReLU]]), BLUE)
  blk((2, 1.2), align(center, [$3 times 3$ \ #text(size: 10pt, fill: MUTED)[ReLU]]), TEAL)
  blk((1.5, -1.2), [$5 times 5$], ACC)
  node((3, 0), [$Y$], radius: 6mm, stroke: 0.9pt + INK)
  edge((0,0), (1,1.2), "-|>", stroke: 0.9pt + BLUE)
  edge((1,1.2), (2,1.2), "-|>", stroke: 0.9pt + TEAL)
  edge((2,1.2), (3,0), "-|>", stroke: 0.9pt + TEAL)
  edge((0,0), (1.5,-1.2), "-|>", stroke: 0.9pt + ACC)
  edge((1.5,-1.2), (3,0), "-|>", stroke: 0.9pt + ACC)
}))

#let _vgg-stage = align(center, diagram(spacing: (13mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let blk(x, body, col) = node((x, 0), body, fill: col.lighten(86%), stroke: 0.9pt + col, corner-radius: 3pt, inset: 7pt)
  blk(0, align(center, [$28 times 28 times 64$ \ #text(size: 10pt, fill: MUTED)[held input]]), INK)
  blk(1, align(center, [$3 times 3$ \ #text(size: 10pt, fill: BLUE)[$64 -> 64$]]), BLUE)
  blk(2, align(center, [$3 times 3$ \ #text(size: 10pt, fill: BLUE)[$64 -> 64$]]), BLUE)
  blk(3, align(center, [$3 times 3$ \ #text(size: 10pt, fill: TEAL)[$64 -> 128$]]), TEAL)
  blk(4, align(center, [$28 times 28 times 128$ \ #text(size: 10pt, fill: MUTED)[held output]]), ACC)
  for i in range(4) { edge((i,0), (i+1,0), "-|>", stroke: 0.8pt + MUTED) }
}))

#let _pixel-mix = align(center, diagram(spacing: (15mm, 7mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 1), text(size: 14pt)[$x_1=2$], radius: 6mm, stroke: 0.9pt + BLUE)
  node((0, 0), text(size: 14pt)[$x_2=-1$], radius: 6mm, stroke: 0.9pt + TEAL)
  node((0, -1), text(size: 14pt)[$x_3=0.5$], radius: 6mm, stroke: 0.9pt + GREEN)
  node((1.4, 0), align(center, [#text(size: 14pt)[$1 times 1$] \ #text(size: 10pt, fill: MUTED)[$W x+b$]]), fill: ACC.lighten(86%), stroke: 0.9pt + ACC, corner-radius: 3pt, inset: 6pt)
  node((2.8, 0.65), text(size: 14pt)[$y_1=3.5$], radius: 7mm, stroke: 0.9pt + BLUE)
  node((2.8, -0.65), text(size: 14pt)[$y_2=1$], radius: 7mm, stroke: 0.9pt + TEAL)
  edge((0,1), (1.4,0), "-|>", stroke: 0.8pt + BLUE)
  edge((0,0), (1.4,0), "-|>", stroke: 0.8pt + TEAL)
  edge((0,-1), (1.4,0), "-|>", stroke: 0.8pt + GREEN)
  edge((1.4,0), (2.8,0.65), "-|>", stroke: 0.9pt + BLUE)
  edge((1.4,0), (2.8,-0.65), "-|>", stroke: 0.9pt + TEAL)
}))

#let _inception-shaped = align(center, diagram(spacing: (10mm, 5mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let blk(p, body, col) = node(p, text(size: 13pt, body), fill: col.lighten(86%), stroke: 0.9pt + col, corner-radius: 3pt, inset: 4pt)
  blk((0,0), align(center, [$28 times 28 times 64$ \ #text(size: 10pt, fill: MUTED)[held input]]), INK)
  blk((1.6,2.1), align(center, [$1 times 1$ \ #text(size: 10pt, fill: TEAL)[$->32$]]), TEAL)
  blk((1.2,0.7), align(center, [$1 times 1$ \ #text(size: 10pt, fill: BLUE)[$64 -> 8$]]), BLUE)
  blk((2.4,0.7), align(center, [$3 times 3$ \ #text(size: 10pt, fill: TEAL)[$8 -> 48$]]), TEAL)
  blk((1.2,-0.7), align(center, [$1 times 1$ \ #text(size: 10pt, fill: BLUE)[$64 -> 4$]]), BLUE)
  blk((2.4,-0.7), align(center, [$5 times 5$ \ #text(size: 10pt, fill: GREEN)[$4 -> 16$]]), GREEN)
  blk((1.2,-2.1), [pool], MUTED)
  blk((2.4,-2.1), align(center, [$1 times 1$ \ #text(size: 10pt, fill: ACC)[$->32$]]), ACC)
  blk((4,0), align(center, [concat \ #text(size: 10pt, fill: MUTED)[$32+48+16+32$]]), ACC)
  blk((5.3,0), [$28 times 28 times 128$], INK)
  edge((0,0),(1.6,2.1),"-|>",stroke:0.7pt+MUTED)
  edge((0,0),(1.2,0.7),"-|>",stroke:0.7pt+MUTED)
  edge((0,0),(1.2,-0.7),"-|>",stroke:0.7pt+MUTED)
  edge((0,0),(1.2,-2.1),"-|>",stroke:0.7pt+MUTED)
  edge((1.2,0.7),(2.4,0.7),"-|>",stroke:0.8pt+TEAL)
  edge((1.2,-0.7),(2.4,-0.7),"-|>",stroke:0.8pt+GREEN)
  edge((1.2,-2.1),(2.4,-2.1),"-|>",stroke:0.8pt+ACC)
  edge((1.6,2.1),(4,0),"-|>",stroke:0.7pt+MUTED)
  edge((2.4,0.7),(4,0),"-|>",stroke:0.7pt+MUTED)
  edge((2.4,-0.7),(4,0),"-|>",stroke:0.7pt+MUTED)
  edge((2.4,-2.1),(4,0),"-|>",stroke:0.7pt+MUTED)
  edge((4,0),(5.3,0),"-|>",stroke:0.8pt+MUTED)
}))

== L8 gave us the operator; today we spend it deliberately #V

#align(center, grid(
  columns: (55mm, 12mm, 55mm, 12mm, 55mm), gutter: 5pt, align: horizon,
  card([L8 · MECHANICS], [Local/shared cross-correlation; output shapes, receptive fields, weights, and MACs.], color: BLUE),
  text(size: 25pt, fill: MUTED)[$arrow.r$],
  card([TODAY · BLOCKS], [Allocate context, channel mixing, depth, and compute inside one stage.], color: TEAL),
  text(size: 25pt, fill: MUTED)[$arrow.r$],
  card([TRANSFER], [Reuse a learned backbone; validate how much of it must adapt.], color: ACC),
))

#pause
#v(9pt)
#note([L6 already established why residual routes help optimization. L7 fixed the validation/test protocol. We will retrieve both—then use them inside a deployable CNN pipeline.], color: GREEN)

== Commit: which design can meet the stage budget? #Q

#case-strip([deployment contract], [$28 times 28 times 64 arrow.r 28 times 28 times 128$, same spatial grid, receptive field at least $3 times 3$, fewer than $10$ million convolution MACs.])
#v(9pt)

#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([A · STANDARD], [one shared $3 times 3$, $64 arrow.r 128$], color: TEAL),
  hairline([B · FACTORIZED], [separate spatial filtering from channel mixing], color: BLUE),
  hairline([C · EITHER], [sharing alone guarantees the budget], color: ACC),
))

#pause
#note([Commit to A, B, or C. Write the count you would calculate first; keep the choice until the closing revisit.], color: BLUE)

== The student contract and 80-minute route #V

By the end, you should be able to turn a block diagram into a shape/compute ledger, justify one design move, and run a leak-free probe-first transfer experiment.

#v(8pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([CORE · 55 min], [bridge → context + bottleneck → residuals → separable convolution → transfer → revisit], color: TEAL),
  hairline([SHOULD · 15 min], [Inception branches → frozen state + preprocessing → diagnostic tests], color: BLUE),
  hairline([OPTIONAL · 10 min], [groups + scaling + ConvNeXt → FPN teaser → primary provenance], color: MUTED),
))

#v(8pt)
#semantic-legend

== One ledger will carry every design decision #V

#case-strip([ledger convention], [All stage candidates use $H_"out"=W_"out"=28$, stride $1$, stated same padding, and no bias. One MAC means one multiply–accumulate per weight per output location.])
#v(8pt)

#ledger(
  ([standard $3 times 3$], [$28 times 28 times 128$], [$3 times 3$], [$?$], [$?$]),
  note: [Budget: fewer than $10,000,000$ MACs. Shape alone is not enough.],
)

#pause
#align(center, text(size: 18pt, fill: BLUE)[Every candidate must preserve the output contract; only context, factorization, route, and cost may change.])

== Baseline: count the standard $3 times 3$ before optimizing #Q

#case-strip([candidate 0], [One same-padded $3 times 3$ convolution maps $64 arrow.r 128$ channels at $28 times 28$ output locations. Bias is omitted.])
#v(8pt)

#align(center, text(size: 22pt)[$"weights"=3 dot 3 dot 64 dot 128$])
#pause
#align(center, text(size: 22pt)[$"MACs"=H_"out"W_"out" dot "weights"$])

#pause
#note([Predict both exact counts and whether the design meets the $10$M budget.], color: BLUE)

== Baseline answer: sharing is necessary but not sufficient #A

#uncover("2-")[
#ledger(
  ([standard $3 times 3$], [$28 times 28 times 128$], [$3 times 3$], [$73,728$], [$57,802,752$]),
  note: [$57.803$M $>$ $10$M: the output contract passes; the deployment budget fails.],
)]

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([WHAT L8 BOUGHT], [One kernel bank is reused at all $784$ locations.], color: GREEN),
  hairline([WHAT REMAINS], [Reduce work inside the kernel bank without changing the output.], color: RED),
))

#pause
#result[The lecture is now a controlled search for a cheaper stage—not a tour of architecture names.]

// ── VGG: context from small kernels ──
== Design move 1 — grow context with small kernels

#case-strip([context], [Keep the held $28 times 28 times 64 arrow.r 28 times 28 times 128$ output. Now require a $5 times 5$ receptive field.])
#v(5pt)
#align(center, text(size: 19pt)[Should we use one $5 times 5$ convolution—or compose two same-padded $3 times 3$s?])
#v(4pt)
#_small-vs-large
#pause
#v(6pt)
#align(center, text(size: 16pt, fill: MUTED)[Both paths see the same spatial extent. Only the blue–teal path inserts an extra learned nonlinearity.])

== Receptive field: derive it one layer at a time #D

With stride $1$, each $k times k$ layer grows the receptive-field side length by $k-1$:
#v(6pt)
#pause
#align(center, text(size: 22pt)[$r_ell = r_(ell-1) + (k-1)$])
#v(7pt)
#pause
#align(center, text(size: 21pt)[
  $1 arrow.r^(3 times 3) 3 arrow.r^(3 times 3) 5 arrow.r^(3 times 3) 7 arrow.r^(3 times 3) 9$
])
#v(8pt)
#grid(columns: (1fr, 1fr), gutter: 13mm,
  [Four stacked $3 times 3$ layers match one $9 times 9$ receptive field.],
  [They also insert *four* transformations and *four* activations instead of one.]
)
#v(6pt)
#result[Depth grows context gradually—and learns what to keep after each step.]

== Same output and context: count both candidates #D

The stacked path uses $64 arrow.r 64 arrow.r 128$; the direct path uses $64 arrow.r 128$. Biases are omitted.
#v(5pt)
#align(center, table(
  columns: 3, stroke: 0.45pt + MUTED, inset: (x: 13pt, y: 7pt), align: (left, center, center),
  table.header([], [*two $3 times 3$s*], [*one $5 times 5$*]),
  [weights], [$9(64 dot 64+64 dot 128)$], [$25 dot 64 dot 128$],
  [count], [$110,592$], [$204,800$],
  [ReLUs], [$2$], [$1$],
))
#pause
#v(7pt)
$204,800-110,592=94,208$ fewer weights: a $46%$ reduction.
#v(6pt)
#result[Small kernels are not merely aesthetic: they buy equal context, fewer weights, and more nonlinear stages.]

== Ledger: more context still misses the deployment budget #D

#ledger((
  [standard $3 times 3$], [$28 times 28 times 128$], [$3 times 3$], [$73,728$], [$57.803$M],
  [two $3 times 3$s], [$28 times 28 times 128$], [$5 times 5$], [$110,592$], [$86.704$M],
  [one $5 times 5$], [$28 times 28 times 128$], [$5 times 5$], [$204,800$], [$160.563$M],
), note: [All three preserve shape with stated same padding. None meets $10$M.])

#pause
#note([Stacking small kernels is a context decision. It does not, by itself, solve the channel-pair cost.], color: RED)

== Misconception check: does depth always grow spatial context? #Q

#align(center, text(size: 22pt)[Stack ten $1 times 1$ convolutions. What is the receptive field?])
#v(12pt)
#grid(columns: (1fr, 1fr, 1fr), gutter: 8mm,
  [#align(center, text(size: 19pt)[A · $1 times 1$])],
  [#align(center, text(size: 19pt)[B · $10 times 10$])],
  [#align(center, text(size: 19pt)[C · depends on channels])],
)
#pause
#v(12pt)
#result[Answer: A. Depth composes transformations; *kernel size and stride* determine spatial reach.]

// ── 1×1: channel mixing ──
== Design move 2 — mix channels without mixing neighbours

At one spatial location, collect the $C_"in"$ channel values into a vector $x_(h w)$:
#v(6pt)
#pause
#align(center, text(size: 22pt)[$y_(h w) = W x_(h w) + b, quad W in RR^(C_"out" times C_"in")$])
#v(9pt)
#grid(columns: (1fr, 1fr), gutter: 14mm,
  [The *same* matrix $W$ is used at every location.],
  [No value from $(h+1,w)$ or $(h,w+1)$ is read.]
)
#v(7pt)
#result[A $1 times 1$ convolution is a shared, per-pixel linear layer across channels.]

== One pixel, two learned channel combinations #D

#_pixel-mix
#v(3pt)
For $x=(2,-1,0.5)^top$, choose
$W = mat(1,-1,0; 0.5,0,2)$ and $b=(0.5,-1)^top$.
#v(3pt)
#pause
#align(center, text(size: 18pt)[$
y = mat(1,-1,0; 0.5,0,2) mat(2;-1;0.5) + mat(0.5;-1)
thin = mat(3.5;1).
$])
#v(3pt)
#align(center, text(size: 15.5pt, weight: 600, fill: MUTED)[Two output channels = two different learned summaries of the same input channels.])

== The same channel mixer runs over the whole grid #V

#uncover("2-")[
#align(center, table(
  columns: 5, stroke: 0.6pt + MUTED, inset: 8pt, align: center,
  table.header([$x_(h w)$], [$(0,0)$], [$(0,1)$], [$(1,0)$], [$(1,1)$]),
  [three channels], [$mat(2;-1;0.5)$], [$mat(0;1;2)$], [$mat(-1;1;0)$], [$mat(3;0;1)$],
  [$W x+b$], [$mat(3.5;1)$], [$mat(-0.5;3)$], [$mat(-1.5;-1.5)$], [$mat(3.5;2.5)$],
))
]
#pause
#v(10pt)
#align(center, text(size: 18pt)[
  $2 times 2 times 3 arrow.r^"same 1×1 weights" 2 times 2 times 2$
])
#v(8pt)
#result[Spatial coordinates stay aligned; only the channel representation changes.]

== Bottleneck: spend the $3 times 3$ in a narrow channel space #V

#case-strip([candidate 2], [Narrow $64 arrow.r 16$, perform the spatial work at width $16$, then expand $16 arrow.r 128$. Same $28 times 28$ grid throughout.])
#v(5pt)
#bottleneck
#pause
#v(8pt)
#align(center, text(size: 17pt)[
  $64 arrow.r^"choose 16 mixtures" 16 arrow.r^"spatial work" 16 arrow.r^"expand" 128$
])
#v(7pt)
#result[The two $1 times 1$s learn which channel combinations deserve expensive spatial processing.]

== Bottleneck: complete cost comparison #D

For the held $28 times 28$ stage, compare the direct $3 times 3$, $64 arrow.r 128$, with $64 arrow.r 16 arrow.r 16 arrow.r 128$:
#v(5pt)
$"direct weights" = 3 dot 3 dot 64 dot 128 = 73,728.$
#v(4pt)
With the bottleneck:
#v(3pt)
#pause
#align(center, text(size: 18pt)[$
64 dot 16 + 3 dot 3 dot 16 dot 16 + 16 dot 128
thin = 1,024 + 2,304 + 2,048 = 5,376.
$])
#v(5pt)
Each weight is reused at $28^2=784$ locations:
$57.803$ million versus $4.215$ million MACs.
#v(5pt)
#result[$5,376 dot 784=4,214,784$: the first candidate under the $10$M budget.]

== Misconception check: what did the $1 times 1$ save? #A

#align(center, text(size: 20pt)[It did *not* make the final $3 times 3$ kernel spatially smaller.])
#v(9pt)
#align(center, text(size: 21pt)[
  #text(fill: RED)[wrong:] $3 times 3 arrow.r 1 times 1$ spatial window \
  #v(6pt)
  #text(fill: GREEN)[right:] $64 arrow.r 16$ channels *before* the $3 times 3$
])
#pause
#v(10pt)
#result[The saving comes from fewer input–output channel pairs inside the expensive spatial convolution.]

== Ledger: channel compression changes the decision #V

#ledger((
  [standard $3 times 3$], [$28 times 28 times 128$], [$3 times 3$], [$73,728$], [$57.803$M],
  [two $3 times 3$s], [$28 times 28 times 128$], [$5 times 5$], [$110,592$], [$86.704$M],
  [bottleneck $16$], [$28 times 28 times 128$], [$3 times 3$], [$5,376$], [$4.215$M],
), note: [GREEN means the candidate satisfies shape, receptive-field, and compute contracts.])

#pause
#align(center, text(size: 21pt, weight: 700, fill: GREEN)[The bottleneck clears the budget with $5.785$M MACs to spare.])

// ── Inception: multi-scale branches ──
== SHOULD · let one block inspect several scales

#case-strip([candidate 3], [The held stage may need channel evidence, $3 times 3$ texture, $5 times 5$ context, and a pooled summary at the same location.])
#v(5pt)
#align(center, text(size: 19pt)[Can four branches preserve the $28 times 28$ grid, concatenate to $128$ channels, and stay under budget?])

#pause
#v(7pt)
#align(center, text(size: 18pt)[
  $1 times 1$ channel evidence $quad$ $3 times 3$ local pattern $quad$ $5 times 5$ wider context $quad$ pooling summary
])
#v(8pt)
#result[Inception runs alternatives in parallel; the stage output keeps every branch as a channel group.]

== A shaped Inception candidate: every width is visible #V

#_inception-shaped
#pause
#v(4pt)
#align(center, text(size: 16pt)[$32+48+16+32=128$ output channels; height and width remain $28 times 28$.])
#v(4pt)
#align(center, text(size: 15pt, weight: 600, fill: MUTED)[Concatenation stacks channels; it does not average or add branch outputs.])

== Count the multi-scale candidate branch by branch #D

All convolutions use stride $1$. The $3 times 3$, $5 times 5$, and $3 times 3$ pool use padding $1$, $2$, and $1$, so every branch remains $28 times 28$.
#v(5pt)
#pause
#align(center, table(
  columns: (42mm, 95mm, 30mm), stroke: 0.45pt + MUTED,
  inset: (x: 8pt, y: 5pt), align: (left, left, right),
  table.header([branch], [weight expression], [weights]),
  [$1 times 1 arrow.r 32$], [$64 dot 32$], [$2,048$],
  [$1 times 1 arrow.r 8 arrow.r^(3 times 3) 48$], [$64 dot 8+9 dot 8 dot 48$], [$3,968$],
  [$1 times 1 arrow.r 4 arrow.r^(5 times 5) 16$], [$64 dot 4+25 dot 4 dot 16$], [$1,856$],
  [pool $arrow.r^(1 times 1) 32$], [$64 dot 32$], [$2,048$],
))
#v(7pt)
#result[$9,920$ weights and $9,920 dot 784=7,777,280$ convolution MACs. Pooling comparisons are excluded from this ledger.]

== Concatenate only after every spatial shape agrees #D

#align(center, text(size: 20pt)[$
28 times 28 times 32 \; || \;
28 times 28 times 48 \; || \;
28 times 28 times 16 \; || \;
28 times 28 times 32
$])
#pause
#v(8pt)
#align(center, text(size: 22pt)[$"concatenate" arrow.r 28 times 28 times (32+48+16+32) = 28 times 28 times 128$])
#v(8pt)
#result[The candidate meets shape and compute contracts; its widest branch sees a $5 times 5$ neighbourhood.]

== Misconception check: branches are not extra sequential depth #Q

#align(center, text(size: 21pt)[Which statement is true?])
#v(8pt)
#grid(columns: (1fr, 1fr), gutter: 14mm,
  [*A.* Every signal crosses all four branches in sequence.],
  [*B.* Each branch offers an alternative transform at one stage; concatenation preserves all outputs.]
)
#pause
#v(11pt)
#result[Answer: B. Branching widens the choice of scale; stacking modules supplies depth.]

#let _residual-numeric = align(center, diagram(spacing: (17mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), [$x=2$], radius: 7mm, fill: rgb("#EFEEEB"))
  node((1.4,0.8), align(center, [$F(x)=0.1x^2$ \ #text(size: 10pt, fill: TEAL)[$F(2)=0.4$]]), fill: TEAL.lighten(86%), stroke: 0.9pt + TEAL, corner-radius: 3pt, inset: 7pt)
  node((2.8,0), [$+$], radius: 6mm, stroke: 1pt + ACC)
  node((4,0), [$y=2.4$], radius: 8mm, stroke: 1pt + ACC)
  edge((0,0),(1.4,0.8),"-|>",stroke:0.9pt+TEAL)
  edge((1.4,0.8),(2.8,0),"-|>",stroke:0.9pt+TEAL)
  edge((0,0),(0,-1),(2.8,-1),(2.8,0),"-|>",stroke:1.1pt+GREEN)
  edge((2.8,0),(4,0),"-|>",stroke:0.9pt+ACC)
  node((1.4,-1), text(size: 11pt, fill: GREEN)[identity: $+x$], stroke: none, fill: none)
}))

#let _residual-stack = align(center, diagram(spacing: (13mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), [$x_0$], radius: 6mm, stroke: 0.9pt + TEAL)
  node((1,0), [$x_1$], radius: 6mm, stroke: 0.9pt + TEAL)
  node((2,0), [$x_2$], radius: 6mm, stroke: 0.9pt + TEAL)
  node((3,0), [$x_3$], radius: 6mm, stroke: 0.9pt + TEAL)
  node((4,0), [$x_4$], radius: 6mm, stroke: 0.9pt + TEAL)
  node((0.5,1), [$F_1$], fill: TEAL.lighten(87%), stroke:0.8pt+TEAL, corner-radius:3pt, inset:5pt)
  node((1.5,1), [$F_2$], fill: TEAL.lighten(87%), stroke:0.8pt+TEAL, corner-radius:3pt, inset:5pt)
  node((2.5,1), [$F_3$], fill: TEAL.lighten(87%), stroke:0.8pt+TEAL, corner-radius:3pt, inset:5pt)
  node((3.5,1), [$F_4$], fill: TEAL.lighten(87%), stroke:0.8pt+TEAL, corner-radius:3pt, inset:5pt)
  for i in range(4) { edge((i,0),(i+1,0),"-|>",stroke:1.2pt+GREEN) }
  edge((0,0),(0.5,1),"-|>",stroke:0.7pt+TEAL); edge((0.5,1),(1,0),"-|>",stroke:0.7pt+TEAL)
  edge((1,0),(1.5,1),"-|>",stroke:0.7pt+TEAL); edge((1.5,1),(2,0),"-|>",stroke:0.7pt+TEAL)
  edge((2,0),(2.5,1),"-|>",stroke:0.7pt+TEAL); edge((2.5,1),(3,0),"-|>",stroke:0.7pt+TEAL)
  edge((3,0),(3.5,1),"-|>",stroke:0.7pt+TEAL); edge((3.5,1),(4,0),"-|>",stroke:0.7pt+TEAL)
}))

#let _separable = align(center, diagram(spacing: (15mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let blk(x, body, col) = node((x,0), body, fill: col.lighten(86%), stroke: 0.9pt + col, corner-radius: 3pt, inset: 7pt)
  blk(0, align(center, [$28 times 28 times 64$ \ #text(size: 10pt, fill: MUTED)[input]]), INK)
  blk(1, align(center, [depthwise $3 times 3$ \ #text(size: 10pt, fill: BLUE)[one filter/channel]]), BLUE)
  blk(2, align(center, [$28 times 28 times 64$ \ #text(size: 10pt, fill: MUTED)[$576$ weights]]), BLUE)
  blk(3, align(center, [pointwise $1 times 1$ \ #text(size: 10pt, fill: TEAL)[mix channels]]), TEAL)
  blk(4, align(center, [$28 times 28 times 128$ \ #text(size: 10pt, fill: MUTED)[$8,192$ weights]]), ACC)
  for i in range(4) { edge((i,0),(i+1,0),"-|>",stroke:0.8pt+MUTED) }
}))

// ── ResNet: trainable depth ──
== L6 retrieval · learn a correction, not a replacement

Suppose a useful block should mostly preserve its input. A plain block must learn $H(x) approx x$.
#v(7pt)
#pause
A residual block instead writes
#align(center, text(size: 23pt)[$H(x)=x+F(x)$])
#v(7pt)
If “do nothing” is best, the learned branch only needs $F(x) approx 0$.
#v(8pt)
#result[The shortcut makes identity behaviour explicit; the residual path learns only the needed change.]

== Commit: residual arithmetic on one scalar #Q

#case-strip([one scalar block], [At $x=2$, let $F(x)=0.1x^2$. Predict $F(2)$, $y=x+F(x)$, and $d y/d x$.])
#pause
#v(5pt)
#_residual-numeric
#v(5pt)
#pause
#align(center, text(size: 21pt)[$F(2)=0.1(2)^2=0.4, quad y=x+F(x)=2+0.4=2.4.$])
#v(7pt)
#result[The output keeps the original $2$ and adds the learned correction $0.4$.]

== The residual block as a node diagram #V

#resblock
#pause
#v(7pt)
#align(center, text(size: 18pt)[$F(x_ell)$ travels through learned layers; $x_ell$ travels unchanged along the green edge.])
#v(7pt)
#result[The addition node is only legal when the two incoming tensors have identical shapes.]

== Why the gradient has a direct route #D

For $y=x+F(x)$,
#pause
#align(center, text(size: 23pt)[$frac(partial y, partial x)=1+frac(partial F, partial x).$])
#v(6pt)
Continue the scalar example $F(x)=0.1x^2$ at $x=2$:
#v(3pt)
$F'(x)=0.2x$, so $F'(2)=0.4$ and
#pause
#align(center, text(size: 23pt)[$frac(d y,d x)=1+0.4=1.4.$])
#v(8pt)
#result[The $1$ is a direct derivative term. It helps signal flow; it does not guarantee a large total gradient if other terms cancel.]

== Residual depth: the identity path survives many blocks #V

#_residual-stack
#pause
#v(9pt)
#align(center, text(size: 19pt)[$x_4=x_0+F_1(x_0)+F_2(x_1)+F_3(x_2)+F_4(x_3)$])
#v(7pt)
#result[Depth becomes a sequence of refinements around a persistent representation.]

== The held channel change makes an identity shortcut illegal #Q

#case-strip([candidate 4], [$x in RR^(28 times 28 times 64)$ but the required stage output is $28 times 28 times 128$. Can $x+F(x)$ be added directly?])
#v(7pt)
#align(center, text(size: 20pt)[$F(x): 28 times 28 times 64 arrow.r 28 times 28 times 128.$])
#v(7pt)
#pause
The identity tensor cannot be added as-is. Apply a $1 times 1$, stride-$1$ projection:
#align(center, text(size: 20pt)[$P(x): 28 times 28 times 64 arrow.r 28 times 28 times 128.$])
#v(7pt)
Now the addition is well-defined:
#align(center, text(size: 23pt)[$y=F(x)+P(x).$])
#v(7pt)
#result[Projection shortcuts change shape; identity shortcuts preserve it.]

== A projected residual candidate still clears the budget #D

Use $F:64 arrow.r 8 arrow.r 8 arrow.r 128$ with $1 times 1$, $3 times 3$, $1 times 1$; use $P:64 arrow.r 128$ with $1 times 1$. Biases are omitted.
#v(6pt)
#pause
#align(center, text(size: 20pt)[$
"weights"_F=64 dot 8+9 dot 8 dot 8+8 dot 128=2,112,
$])
#align(center, text(size: 20pt)[$
"weights"_P=64 dot 128=8,192.
$])
#v(7pt)
#uncover("3-")[
#ledger((
  [projected residual], [$28 times 28 times 128$], [$3 times 3$], [$10,304$], [$8,078,336$],
), note: [$y=F(x)+P(x)$ is legal; both routes end at the same shape.])
]
#pause
#result[The shortcut is not free when shape changes—but this narrow residual branch remains within budget.]

== Misconception check: is the shortcut “skipping learning”? #A

#align(center, text(size: 21pt)[No. The output still depends on the learned residual $F(x)$.])
#v(8pt)
#align(center, text(size: 22pt)[$x + F(x) = y$])
#v(8pt)
#align(center, text(size: 17pt)[
  #text(fill: GREEN)[preserved representation] $quad + quad$ #text(fill: TEAL)[learned update]
])
#v(9pt)
The shortcut changes the *parameterization* of the target function. It makes preservation cheap while leaving the branch free to learn large or small changes.
#pause
#v(7pt)
#result[Residual learning is controlled refinement—not a bypass around the model.]

// ── Efficient convolutions ──
== Core move · factor spatial filtering from channel mixing

A standard $k times k$ convolution does both jobs at once:
#v(7pt)
#pause
#align(center, text(size: 22pt)[$k^2 dot C_"in" C_"out"$])
#v(4pt)
#align(center, text(size: 17pt)[
  #text(fill: BLUE)[$k^2$ spatial positions] $quad times quad$
  #text(fill: TEAL)[$C_"in" C_"out"$ channel pairs]
])
#v(8pt)
Ignoring bias, its weight count is $k^2 C_"in"C_"out"$; its MAC count is
#align(center, text(size: 22pt)[$H_"out" W_"out" k^2 C_"in" C_"out".$])
#v(7pt)
#result[Depthwise separable convolution performs the two jobs in two explicit steps.]

== Standard convolution: establish the baseline #D

Let $H=W=28$, $k=3$, $C_"in"=64$, and $C_"out"=128$.
#v(6pt)
#pause
#align(center, text(size: 20pt)[$"weights" = 3 dot 3 dot 64 dot 128 = 73,728.$])
#v(5pt)
There are $28 dot 28=784$ output locations:
#align(center, text(size: 20pt)[$"MACs" = 784 dot 73,728 = 57,802,752.$])
#v(8pt)
#alertbox[This is the cost the factorized version must match in output shape—not in arithmetic.]

== Depthwise step: one spatial filter per input channel #D

#align(center, text(size: 19pt)[$28 times 28 times 64 arrow.r^"64 separate 3×3 filters" 28 times 28 times 64$])
#v(7pt)
#pause
#align(center, text(size: 21pt)[$"weights" = 3 dot 3 dot 64 = 576,$])
#align(center, text(size: 21pt)[$"MACs" = 784 dot 576 = 451,584.$])
#v(7pt)
No channel is mixed with another yet.
#v(7pt)
#result[Depthwise convolution answers only: “what spatial pattern occurs within this channel?”]

== Pointwise step: mix the filtered channels #D

#align(center, text(size: 19pt)[$28 times 28 times 64 arrow.r^"1×1, 128 outputs" 28 times 28 times 128$])
#v(7pt)
#pause
#align(center, text(size: 21pt)[$"weights" = 64 dot 128 = 8,192,$])
#align(center, text(size: 21pt)[$"MACs" = 784 dot 8,192 = 6,422,528.$])
#v(7pt)
#result[The pointwise layer answers: “which channel combinations should form the $128$ outputs?”]

== Put the factorized block back together #V

#_separable
#v(8pt)
#pause
#align(center, text(size: 20pt)[$"separable weights"=576+8,192=8,768,$])
#align(center, text(size: 20pt)[$"separable MACs"=451,584+6,422,528=6,874,112.$])
#v(6pt)
#result[$frac("57,802,752", "6,874,112") approx 8.41 times$ fewer MACs than the standard convolution.]

== OPTIONAL · grouped convolution connects the extremes #V

#align(center, text(size: 19pt)[
  one group $arrow.r$ a standard convolution mixes *all* input channels \
  #v(8pt)
  $g$ groups $arrow.r$ each output sees only $C_"in"/g$ input channels \
  #v(8pt)
  $g=C_"in"$ $arrow.r$ depthwise connectivity: each output sees one input channel
])
#pause
#v(11pt)
#result[With depth multiplier $1$, $C_"out"=C_"in"$ and there is one spatial filter per channel; a later $1 times 1$ restores full mixing.]

== OPTIONAL · three scaling knobs have different costs #D

For a representative convolution, MACs scale approximately as
#align(center, text(size: 23pt)[$"MACs" prop r^2 d w^2,$])
where $r$ scales resolution, $d$ depth, and $w$ channel width.
#v(7pt)
#align(center, table(
  columns: 3, stroke: 0.45pt + MUTED, inset: (x: 14pt, y: 7pt), align: center,
  table.header([*Double*], [*What changes*], [*rough MAC multiplier*]),
  [resolution $r$], [$H,W -> 2H,2W$], [$4 times$],
  [depth $d$], [twice as many blocks], [$2 times$],
  [width $w$], [$C_"in",C_"out" -> 2 times$], [$4 times$],
))
#pause
#v(7pt)
#result[EfficientNet's compound scaling grows all three gradually instead of overspending on one axis.]

== OPTIONAL · ConvNeXt recombines familiar moves #V

#align(center, diagram(spacing: (12mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let blk(x, body, col) = node((x,0), body, fill:col.lighten(87%), stroke:0.9pt+col, corner-radius:3pt, inset:7pt)
  node((0,0),[$x$],radius:6mm,stroke:0.9pt+INK)
  blk(1, align(center, [depthwise $7 times 7$ \ #text(size:10pt,fill:MUTED)[spatial context]]), BLUE)
  blk(2, [LayerNorm], TEAL)
  blk(3, align(center, [$1 times 1$ expand \ #text(size:10pt,fill:MUTED)[channel MLP]]), ACC)
  blk(4, [GELU], GREEN)
  blk(5, [$1 times 1$ project], ACC)
  node((6,0),[$+$],radius:6mm,stroke:1pt+INK)
  for i in range(6) { edge((i,0),(i+1,0),"-|>",stroke:0.8pt+MUTED) }
  edge((0,0),(0,-1.3),(6,-1.3),(6,0),"-|>",stroke:1pt+GREEN)
}))
#pause
#v(9pt)
#result[Modern CNNs recombine the same ideas: large-context depthwise filtering, cheap channel mixing, normalization, and a residual path.]

== Diagnose: fewer MACs need not mean lower latency #Q

#align(center, text(size: 22pt)[Does $8.41 times$ fewer MACs guarantee $8.41 times$ lower wall-clock latency?])
#pause
#v(8pt)
#align(center, text(size: 22pt, weight: 700, fill: RED)[Not necessarily.])
#v(6pt)
Wall-clock speed also depends on memory traffic, kernel launch overhead, tensor layout, batch size, and hardware support.
#v(9pt)
#align(center, text(size: 20pt)[$"algorithmic cost" != "measured latency"$])
#v(9pt)
#result[Use MACs and parameters for reasoning; benchmark latency on the deployment device before making the final choice.]

#let _classifier = align(center, diagram(spacing: (10mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let blk(x, body, col) = node((x,0), body, fill: col.lighten(87%), stroke: 0.9pt + col, corner-radius: 3pt, inset: 6pt)
  blk(0, align(center, [image \ #text(size: 10pt, fill: MUTED)[$224 times 224 times 3$]]), INK)
  blk(1, align(center, [backbone \ #text(size: 10pt, fill: TEAL)[$7 times 7 times 512$]]), TEAL)
  blk(2, align(center, [global average \ #text(size: 10pt, fill: MUTED)[$512$]]), BLUE)
  blk(3, align(center, [linear head \ #text(size: 10pt, fill: ACC)[$512 -> K$]]), ACC)
  blk(4, align(center, [logits $z$ \ #text(size: 10pt, fill: MUTED)[$K$ scores]]), ACC)
  blk(5, align(center, [softmax \ #text(size: 10pt, fill: TEAL)[$K$ probabilities]]), TEAL)
  for i in range(5) { edge((i,0),(i+1,0),"-|>",stroke:0.8pt+MUTED) }
}))

#let _transfer-flow = align(center, diagram(spacing: (12mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let blk(p, body, col) = node(p, body, fill: col.lighten(87%), stroke: 0.9pt + col, corner-radius: 3pt, inset: 7pt)
  blk((0,0), [images], INK)
  blk((1.2,0), align(center, [pretrained backbone \ #text(size: 10pt, fill: TEAL)[reuse first]]), TEAL)
  blk((2.6,0), align(center, [new head \ #text(size: 10pt, fill: ACC)[train first]]), ACC)
  blk((3.7,0), [logits], ACC)
  blk((4.8,0), [loss], RED)
  blk((4.8,-1.3), [labels], INK)
  edge((0,0),(1.2,0),"-|>",stroke:0.8pt+MUTED)
  edge((1.2,0),(2.6,0),"-|>",stroke:0.8pt+MUTED)
  edge((2.6,0),(3.7,0),"-|>",stroke:0.8pt+MUTED)
  edge((3.7,0),(4.8,0),"-|>",stroke:0.8pt+MUTED)
  edge((4.8,-1.3),(4.8,0),"-|>",stroke:0.9pt+RED)
}))

#let _fpn = align(center, diagram(spacing: (15mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let feat(p, body, col) = node(p, body, fill: col.lighten(87%), stroke: 0.9pt + col, corner-radius: 3pt, inset: 6pt)
  feat((0,2.1), [$C_2: 56 times 56$], BLUE)
  feat((0,0.7), [$C_3: 28 times 28$], TEAL)
  feat((0,-0.7), [$C_4: 14 times 14$], GREEN)
  feat((0,-2.1), [$C_5: 7 times 7$], ACC)
  feat((2.2,2.1), [$P_2$], BLUE)
  feat((2.2,0.7), [$P_3$], TEAL)
  feat((2.2,-0.7), [$P_4$], GREEN)
  feat((2.2,-2.1), [$P_5$], ACC)
  for y in (2.1,0.7,-0.7,-2.1) { edge((0,y),(2.2,y),"-|>",stroke:0.8pt+MUTED) }
  edge((2.2,-2.1),(2.2,-0.7),"-|>",stroke:1pt+ACC)
  edge((2.2,-0.7),(2.2,0.7),"-|>",stroke:1pt+GREEN)
  edge((2.2,0.7),(2.2,2.1),"-|>",stroke:1pt+TEAL)
  node((1.1,2.7), text(size: 10pt, fill: MUTED)[$1 times 1$ lateral], stroke:none, fill:none)
  node((3.15,0), text(size: 10pt, fill: MUTED)[upsample + add], stroke:none, fill:none)
}))

// ── From a stage to a reusable backbone ──
== The stage ledger now contains architectural choices #V

#align(center, text(size: 11.5pt)[#table(
  columns: (42mm, 32mm, 21mm, 31mm, 30mm),
  stroke: 0.45pt + MUTED, inset: (x: 7pt, y: 3pt), align: (left, center, center, center, center),
  table.header([*candidate*], [*output*], [*RF*], [*weights*], [*MACs*]),
  [standard $3 times 3$], [$28^2 times 128$], [$3 times 3$], [$73,728$], [$57.803$M],
  [two $3 times 3$s], [$28^2 times 128$], [$5 times 5$], [$110,592$], [$86.704$M],
  [bottleneck $16$], [$28^2 times 128$], [$3 times 3$], [$5,376$], [$4.215$M],
  [Inception], [$28^2 times 128$], [up to $5 times 5$], [$9,920$], [$7.777$M],
  [projected residual], [$28^2 times 128$], [$3 times 3$], [$10,304$], [$8.078$M],
  [depthwise + pointwise], [$28^2 times 128$], [$3 times 3$], [$8,768$], [$6.874$M],
)])

#pause
#v(5pt)
#align(center, text(size: 13pt, weight: 600, fill: GREEN)[The best block is conditional: satisfy the task contract, then measure the resource that matters on the target hardware.])

== A stage becomes one repeated part of a backbone #V

#_classifier
#pause
#v(8pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([STAGES], [spatial transforms build a hierarchy], color: TEAL),
  hairline([POOL], [$7 times 7 times 512 arrow.r h in RR^512$], color: BLUE),
  hairline([HEAD], [$h arrow.r z in RR^K$], color: ACC),
))
#pause
#result[The block ledger governs the backbone; transfer asks how much of that learned hierarchy to reuse.]

// ── Transfer learning ──
== Transfer starts with a hypothesis: old features may already separate new labels

#align(center, text(size: 18pt)[
  pixels $arrow.r$ edges and colours $arrow.r$ textures $arrow.r$ parts $arrow.r$ task evidence
])
#pause
#v(8pt)
#_transfer-flow
#v(7pt)
#result[Probe the reused representation first; adapt it only when validation evidence says the fixed features are insufficient.]

== Commit: probe, unfreeze late, or fine-tune all? #Q

A pretrained backbone emits $h in RR^512$ for a new six-class dataset. Training data are limited; the test set is sealed.
#v(9pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([A · PROBE], [freeze the backbone; train only $512 arrow.r 6$], color: BLUE),
  hairline([B · LATE BLOCKS], [train the head and final backbone stage], color: TEAL),
  hairline([C · ALL], [fine-tune every parameter immediately], color: RED),
))
#pause
#note([Commit before seeing target validation curves. The correct choice is empirical, not a slogan.], color: BLUE)

== Count the new six-class head exactly #D

#align(center, text(size: 21pt)[$z=W h+b, quad W in RR^(6 times 512), quad b in RR^6.$])
#pause
#v(8pt)
#align(center, text(size: 25pt, weight: 700, fill: ACC)[$"trainable head parameters"=6 dot 512+6=3,078.$])
#pause
#v(8pt)
If the backbone has $11$ million parameters, a linear probe trains
$frac("3,078", "11,000,000"+"3,078") approx 0.028%$ of the model.
#v(7pt)
#result[The first transfer experiment is intentionally tiny and inspectable.]

== A linear probe computes through the backbone but updates only the head #V

#align(center, diagram(spacing:(15mm,9mm), node-stroke:0.9pt+INK, node-fill:white, {
  node((0,0),[image],radius:7mm,stroke:0.9pt+INK)
  node((1.4,0),align(center,[backbone \ #text(size:10pt,fill:BLUE)[fixed parameters]]),fill:BLUE.lighten(88%),stroke:0.9pt+BLUE,corner-radius:3pt,inset:8pt)
  node((2.8,0),align(center,[new head \ #text(size:10pt,fill:ACC)[trainable]]),fill:ACC.lighten(87%),stroke:0.9pt+ACC,corner-radius:3pt,inset:8pt)
  node((4,0),[loss],radius:7mm,stroke:0.9pt+RED)
  edge((0,0),(1.4,0),"-|>",stroke:0.8pt+MUTED)
  edge((1.4,0),(2.8,0),"-|>",stroke:0.8pt+MUTED)
  edge((2.8,0),(4,0),"-|>",stroke:0.8pt+MUTED)
  edge((4,0),(2.8,-1.1),(2.8,0),"-|>",stroke:1pt+RED)
}))
#pause
#v(8pt)
#result[A probe measures linear separability in pretrained feature space; it does not make backbone computation disappear.]

== SHOULD · freezing parameters does not freeze model state #D

#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([PARAMETERS], [`requires_grad=False`; exclude them from the optimizer.], color: TEAL),
  hairline([BUFFERS], [BatchNorm running means/variances can still change in `train()` mode.], color: RED),
  hairline([STOCHASTIC LAYERS], [Dropout also changes behaviour between `train()` and `eval()`.], color: BLUE),
))
#pause
#v(9pt)
#note([For a truly fixed probe, keep the frozen backbone in `eval()` while the new head trains. Updating BatchNorm statistics deliberately is a different intervention—name and validate it.], color: GREEN)

== SHOULD · match both the preprocessing and tensor-layout contracts #D

#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([BOARD NOTATION], [$H times W times C$ keeps the spatial story readable.], color: BLUE),
  hairline([PYTORCH INPUT], [`Conv2d` receives $N times C times H times W$.], color: TEAL),
))
#pause
#v(9pt)
#align(center, text(size: 21pt)[$x'_c=frac(x_c-mu_c,sigma_c)$])
#v(6pt)
Channel order, input range, resize/crop policy, and $(mu,sigma)$ must match the weight recipe.
#pause
#alertbox[Before blaming transfer, inspect one batch: shape, dtype, range, channel order, normalization, and backbone mode.]

== L7 protocol: validation chooses how much to unfreeze #V

#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([TRAIN], [fit head/backbone parameters for one declared regime], color: TEAL),
  hairline([VALIDATION], [choose probe vs late/all, learning rates, stopping epoch, and checkpoint], color: BLUE),
  hairline([SEALED TEST], [run once after the full decision rule and checkpoint are frozen], color: RED),
))
#pause
#v(9pt)
#result[Repeatedly checking test performance while choosing unfreeze depth turns the test set into validation data.]

== A controlled transfer experiment changes one lever at a time #D

Hold fixed the seed, train/validation split, head initialization, augmentation, optimizer family, epoch budget, and selection metric.
#pause
#v(8pt)
#align(center, table(
  columns: (42mm, 48mm, 48mm, 48mm), stroke: 0.45pt + MUTED,
  inset: (x: 7pt, y: 3pt), align: center,
  table.header([regime], [trainable blocks], [backbone LR], [record]),
  [probe], [head only], [$0$], [best validation; epoch],
  [late], [head + final stage], [small], [same record],
  [all], [entire network], [smaller still], [same record],
))
#pause
#v(5pt)
#align(center, text(size: 13pt, weight: 600, fill: GREEN)[Validation compares interventions fairly; the sealed test evaluates only the selected procedure.])

== Unfreeze progressively in response to evidence #V

#align(center, text(size: 18pt)[
  *probe underfits train and validation* $arrow.r$ reused features may not separate the task \
  #v(7pt)
  *unfreeze late blocks* $arrow.r$ test whether task-specific adaptation closes both gaps \
  #v(7pt)
  *consider broader unfreezing* $arrow.r$ only as another validation-controlled candidate
])
#pause
#v(9pt)
#note([High training accuracy but poor validation is not evidence to unfreeze more. It points first to data size, regularization, augmentation, or shift.], color: RED)

== Parameter groups make the fine-tuning hypothesis explicit #D

#align(center, text(size: 21pt)[$
theta_h arrow.r^(eta_h) theta_h-eta_h nabla_(theta_h) cal(L),
quad
theta_b arrow.r^(eta_b) theta_b-eta_b nabla_(theta_b) cal(L)
$])
#pause
#align(center, text(size: 22pt, weight: 700, fill: ACC)[$0 < eta_b < eta_h$])
#v(8pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([NEW HEAD], [larger step: it starts random and must learn the task], color: ACC),
  hairline([REUSED BACKBONE], [smaller step: preserve useful features while adapting], color: TEAL),
))
#pause
#result[The ratio is a hyperparameter selected on validation—not a universal constant.]

== SHOULD · diagnose transfer failures as symptom → suspect → test #V

#set text(size: 13.5pt)
#align(center, table(
  columns: (58mm, 58mm, 73mm), stroke: 0.45pt + MUTED,
  inset: (x: 7pt, y: 5pt), align: (left, left, left),
  table.header([symptom], [suspect], [smallest discriminating test]),
  [loss stays near random], [input contract mismatch], [log batch shape/range; compare expected transform],
  [frozen features drift], [BN buffers or dropout active], [diff parameters *and buffers* across one step],
  [validation drops after unfreeze], [backbone rate too large], [reduce $eta_b$; log layer-wise update norms],
  [MACs fall; latency does not], [memory/kernel overhead], [benchmark the target device and batch size],
))
#pause
#result[Change the experiment only after the test separates competing causes.]

== Follow along: predict, calculate, run, explain #I

#interbox(link-to: NB1)[
  Open the exact Colab notebook. Predict the 73,728 / 8,768 parameter ratio, verify standard and separable NCHW paths, inspect residual identity gradients, then prove which parameters changed after one optimizer step.
]
#pause
#v(7pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([FACTOR], [$57.803$M $arrow.r 6.874$M MACs], color: BLUE),
  hairline([PRESERVE], [$x+F(x)$ and the direct gradient term], color: GREEN),
  hairline([REUSE], [$512 arrow.r 6$ head: $3,078$ trainable], color: ACC),
))
#pause
#note([The synthetic TinyBackbone verifies freezing mechanics; because it is randomly initialized, it is not evidence that pretrained transfer improves validation.], color: MUTED)

== The transfer decision is small, but the protocol is complete #V

#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([START], [$512 arrow.r 6$ head = $3,078$ trainable parameters], color: ACC),
  hairline([SELECT], [validation chooses frozen/late/all and the checkpoint], color: BLUE),
  hairline([REPORT], [sealed test once; include trainable count and measured latency], color: GREEN),
))
#pause
#v(10pt)
#result[“Transfer learning worked” means a controlled candidate won on validation and survived the one-time test—not merely that training ran.]

== OPTIONAL · one teaser for L10: keep spatial features when the task asks where #V

#_fpn
#pause
#v(6pt)
#align(center, text(size: 16pt)[A feature pyramid laterally projects backbone stages, upsamples coarse features, and adds only tensors with matching height, width, and channels.])
#pause
#result[Today the backbone is the reusable object. L10 will derive the localization heads and pyramid operations needed for detection.]

// ── Synthesis ──
== Architecture names collapse into six reusable moves #V

#align(center, text(size: 18pt)[
  need more context $arrow.r$ stack small kernels *(VGG)* \
  #v(6pt)
  need cheaper channel work $arrow.r$ squeeze with $1 times 1$ *(bottleneck)* \
  #v(6pt)
  need several scales $arrow.r$ branch then concatenate *(Inception)* \
  #v(6pt)
  need trainable depth $arrow.r$ add identity or projected shortcuts *(ResNet)* \
  #v(6pt)
  need mobile efficiency $arrow.r$ factor spatial and channel work *(MobileNet)* \
  #v(6pt)
  need a new task $arrow.r$ reuse backbone, replace head, validate adaptation
])
#pause
#result[Names provide provenance; shape, compute, route, and validation provide the reasoning.]

== Revisit the opening commitment before revealing it #Q

#case-strip([same deployment contract], [$28 times 28 times 64 arrow.r 28 times 28 times 128$, at least $3 times 3$ context, fewer than $10$M convolution MACs.])
#v(9pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([A · STANDARD], [$73,728$ weights; $57.803$M MACs], color: TEAL),
  hairline([B · FACTORIZED], [$8,768$ weights; $6.874$M MACs], color: BLUE),
  hairline([C · EITHER], [sharing alone guarantees budget], color: MUTED),
))
#pause
#note([Defend your original choice using the ledger—not the architecture name.], color: BLUE)

== Answer: B meets every stated contract #A

#align(center, text(size: 25pt, weight: 700, fill: GREEN)[$frac("57,802,752", "6,874,112") approx 8.41 times$ fewer MACs])
#pause
#v(9pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([SHAPE], [$28 times 28 times 128$ on both paths], color: GREEN),
  hairline([CONTEXT], [depthwise $3 times 3$ supplies the required RF], color: GREEN),
  hairline([BUDGET], [$6.874$M $<$ $10$M], color: GREEN),
))
#pause
#result[Factorization won this contract. A bottleneck, Inception, or projected residual candidate may win when context, branching, or optimization is the dominant need.]

== Final design checklist #Q

Given an image task, ask in order:
#v(5pt)
1. What output shape and receptive field must the evidence satisfy?
2. Which operation mixes space; which mixes channels?
3. Are additions and concatenations shape-legal?
4. Which weights and output positions dominate MACs?
5. Does measured device latency agree with the ledger?
6. Can a pretrained backbone be probed before broader adaptation?
7. Which validation rule chooses the regime before the test is opened?
#pause
#v(7pt)
#result[A strong CNN pipeline is a sequence of declared contracts and measured decisions—not a pile of named blocks.]

== L10 handoff: the backbone answers what; detection must retain where #V

#align(center, grid(columns: (1fr, 18mm, 1fr), gutter: 10pt, align: horizon,
  card([CLASSIFICATION], [$7 times 7 times 512 arrow.r^"global average" 512 arrow.r 6$ \ one image-level answer], color: TEAL),
  text(size: 27pt, fill: MUTED)[$arrow.r$],
  card([DETECTION], [$C_2,C_3,C_4,C_5 arrow.r$ spatial heads \ classes *and* boxes at many locations], color: ACC),
))
#pause
#v(10pt)
#result[The next lecture keeps the reusable backbone but changes the head, targets, geometry, and evaluation.]

== OPTIONAL · primary provenance and exact companion #V

#set text(size: 14pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 20pt,
  hairline([BLOCKS], [
    #link(VGG)[Simonyan & Zisserman · VGG] \
    #link(INCEPTION)[Szegedy et al. · Inception] \
    #link(RESNET)[He et al. · ResNet] \
    #link(MOBILENET)[Howard et al. · MobileNet]
  ], color: TEAL),
  hairline([SCALING + PIPELINES], [
    #link(EFFICIENTNET)[Tan & Le · EfficientNet] \
    #link(CONVNEXT)[Liu et al. · ConvNeXt] \
    #link(FPN)[Lin et al. · FPN] \
    #link(NB1)[Exact L08B Colab notebook]
  ], color: BLUE),
))
#pause
#v(9pt)
#align(center, text(size: 16pt, fill: MUTED)[Use the papers for historical claims; use the held ledger and notebook for reproducible arithmetic.])
