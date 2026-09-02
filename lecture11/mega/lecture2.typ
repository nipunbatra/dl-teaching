// L11M, Lecture 2 — 59 conceptual frames.
// Native Typst/Fletcher only: every technical figure remains vector in PDF.
#import "../../common/metropolis.typ": *
#import "../../common/mldiag.typ": *
#import "helpers.typ": *

// Lecture-local drawing helpers. They deliberately live here: Lecture 2 owns
// the complete attention derivation, while the shared visual grammar stays
// small and reusable across all three parts.
#let l2-slot(body, color: TEAL, fill: PALE-TEAL, width: 14mm, height: 9mm,
  size: 11pt) = box(
  width: width,
  height: height,
  fill: fill,
  stroke: 0.8pt + color,
  radius: 2pt,
  align(center + horizon, text(size: size, weight: 650, fill: INK, body)),
)

#let l2-concat-strip(n: 5, color: TEAL, label: [embedding slots]) = align(center, [
  #grid(
    columns: (1fr,) * n,
    gutter: 2pt,
    ..range(n).map(i => l2-slot(
      [$e_#(i+1)$],
      color: color,
      fill: color.lighten(88%),
      width: 100%,
      height: 10mm,
    )),
  )
  #v(3pt)
  #align(center, text(size: 10pt, fill: MUTED, label))
])

#let l2-qkv-card(letter, title, question, color, fill) = card(
  title,
  [
    #align(center, text(size: 28pt, weight: 750, fill: color)[#letter])
    #v(2pt)
    #align(center, text(size: 13pt, fill: INK, question))
  ],
  color: color,
  fill: fill,
)

#let l2-weight-strip(labels, weights, colors: none) = {
  assert(labels.len() == weights.len())
  let cols = if colors == none { labels.map(_ => BLUE) } else { colors }
  align(center, grid(
    columns: labels.map(_ => 20mm),
    gutter: 6pt,
    align: bottom,
    ..range(labels.len()).map(i => {
      let p = weights.at(i)
      let c = cols.at(i)
      stack(dir: ttb, spacing: 2pt,
        align(center, text(size: 10.5pt, weight: 700, fill: c)[#(calc.round(p * 1000) / 1000)]),
        box(width: 18mm, height: 25mm, align(bottom, block(
          width: 18mm,
          height: p * 25mm,
          fill: c.lighten(18%),
          stroke: 0.5pt + c,
          radius: (top: 2pt),
        ))),
        l2-slot(labels.at(i), color: TEAL, fill: PALE-TEAL, width: 18mm, height: 8mm),
      )
    }),
  ))
}

#let l2-role-flow() = align(center, diagram(
  spacing: (20mm, 12mm),
  node-stroke: 0.9pt + INK,
  node-fill: white,
  {
    flow-node((0, 0), [current token], color: BLUE, fill: PALE-BLUE, w: 27mm)
    flow-node((1.5, 0), [$q$\ what I seek], color: BLUE, fill: PALE-BLUE, w: 29mm)
    flow-node((3.3, -1), [$k_j$\ what matches], color: TEAL, fill: PALE-TEAL, w: 29mm)
    flow-node((3.3, 1), [$v_j$\ what is sent], color: ACC, fill: PALE-ACC, w: 29mm)
    flow-node((5.25, 0), [weighted\ return $h$], color: GREEN, fill: PALE-GREEN, w: 30mm)
    flow-arrow((0, 0), (1.5, 0), color: BLUE)
    flow-arrow((1.5, 0), (3.3, -1), color: BLUE, label: [$q^top k_j$])
    flow-arrow((3.3, -1), (5.25, 0), color: BLUE, label: [$alpha_j$])
    flow-arrow((3.3, 1), (5.25, 0), color: ACC, label: [$v_j$])
  },
))

// A stable whiteboard sketch for pooling. Arrow thickness and labels carry the
// only new idea from one build to the next: equal weights, then learned weights.
#let l2-pool-diagram(labels, weights, result: [$h$], colors: none) = {
  assert(labels.len() == weights.len())
  let n = labels.len()
  let cols = if colors == none { labels.map(_ => TEAL) } else { colors }
  align(center, diagram(
    spacing: (22mm, 12mm),
    node-stroke: .9pt + TEAL,
    node-fill: PALE-TEAL,
    {
      for i in range(n) {
        let x = i * 1.15
        let p = calc.round(weights.at(i) * 100) / 100
        let label = labels.at(i)
        flow-node(
          (x, 0),
          stack(
            dir: ttb,
            spacing: 1pt,
            align(center, label),
            align(center, text(size: 9pt, weight: 650, fill: cols.at(i))[α = #p]),
          ),
          color: cols.at(i), fill: cols.at(i).lighten(88%), w: 21mm,
        )
        edge(
          (x, 0), ((n - 1) * .575, 1.65), "-|>",
          stroke: (.45pt + 2.6pt * weights.at(i)) + cols.at(i),
        )
      }
      flow-node(((n - 1) * .575, 1.65), result, color: BLUE, fill: PALE-BLUE, w: 30mm, size: 14pt)
    },
  ))
}

// One token state branches into three learned views. Keeping this picture
// separate from the all-token matrix picture makes the Q/K/V leap gentler.
#let l2-qkv-split() = align(center, diagram(
  spacing: (20mm, 11mm),
  node-stroke: .9pt + INK,
  node-fill: white,
  {
    flow-node((0, 0), [token state $h_i$], color: INK, fill: CREAM, w: 28mm)
    for (y, w, out, color, fill) in (
      (-1.25, [$W_Q$], [$q_i$\ what to seek], BLUE, PALE-BLUE),
      (0, [$W_K$], [$k_i$\ how to match], TEAL, PALE-TEAL),
      (1.25, [$W_V$], [$v_i$\ what to send], ACC, PALE-ACC),
    ) {
      flow-node((1.65, y), w, color: color, fill: white, w: 18mm)
      flow-node((3.25, y), out, color: color, fill: fill, w: 31mm)
      flow-arrow((0, 0), (1.65, y), color: color)
      flow-arrow((1.65, y), (3.25, y), color: color)
    }
  },
))

#let l2-mini-map(labels, mode: "all", cell: 10.5mm) = {
  let n = labels.len()
  let cells = ()
  cells.push(block())
  for label in labels {
    cells.push(align(center, text(size: 9.2pt, weight: 700, fill: TEAL, label)))
  }
  for i in range(n) {
    cells.push(align(right + horizon, text(size: 9.2pt, weight: 700, fill: BLUE, labels.at(i))))
    for j in range(n) {
      let future-leak = mode == "leaky" and j > i
      let allowed = mode == "all" or mode == "leaky" or j <= i
      cells.push(block(
        width: cell,
        height: 7mm,
        fill: if future-leak { PALE-RED } else if allowed { PALE-BLUE } else { PALE-RED },
        stroke: 0.45pt + white,
        align(center + horizon, text(
          size: 10.5pt,
          weight: 700,
          fill: if future-leak { RED } else if allowed { BLUE } else { RED },
          if allowed { [•] } else { [$times$] },
        )),
      ))
    }
  }
  align(center, grid(
    columns: (13mm,) + (cell,) * n,
    rows: (6.8mm,) + (7mm,) * n,
    gutter: 0pt,
    ..cells,
  ))
}

#let l2-parallel-row(inputs, targets) = {
  assert(inputs.len() == targets.len())
  let n = inputs.len()
  align(center, grid(
    columns: (16mm,) + (1fr,) * n,
    rows: (8mm, 8mm, 8mm),
    gutter: 2pt,
    align: center + horizon,
    text(size: 9pt, weight: 700, fill: MUTED)[INPUT],
    ..inputs.map(x => l2-slot(x, width: 100%, height: 8mm)),
    text(size: 9pt, weight: 700, fill: MUTED)[STATE],
    ..range(n).map(i => l2-slot([$h_#(i+1)$], color: BLUE, fill: PALE-BLUE, width: 100%, height: 8mm)),
    text(size: 9pt, weight: 700, fill: MUTED)[TARGET],
    ..targets.map(x => l2-slot(x, color: ACC, fill: PALE-ACC, width: 100%, height: 8mm)),
  ))
}

= Lecture 2: How attention works

// ───────────────────────────── ACT 5 ─────────────────────────────

// Scope local text settings to keep Touying's show-rule nesting bounded.
#[
== The fixed-window pipeline

#simple-flow((
  ([token IDs], TEAL, PALE-TEAL),
  ([embedding\ lookup], TEAL, PALE-TEAL),
  ([concatenate\ $k d$ numbers], INK, CREAM),
  ([MLP], BLUE, PALE-BLUE),
  ([next-token\ softmax], ACC, PALE-ACC),
))

#v(8pt)
#token-row((token([a]), token([a]), token([b])), arrow-to: [i])

#pause
#v(7pt)
#note[
  We still predict the next token. How should we combine the context vectors?
]

== What if the window is five tokens? #Q

#set text(size: 17pt)
#align(center, text(size: 19pt)[Five tokens $arrow.r$ five embedding vectors $arrow.r$ one wider first layer])

#v(4pt)
#align(center, scale(x: 78%, y: 78%, reflow: true,
  neural-net-sketch(
    input-labels: ([$e_1$], [$e_2$], [$e_3$], [$e_4$], [$e_5$]),
    output-labels: ([a], [b], [i], [$dots$], [z]),
    hidden: 4,
    highlight-output: 2,
  )
))

#pause
#v(4pt)
#claim[
  The first layer now takes $5d$ inputs instead of $3d$.
]

== What if the window is 100 tokens? #D

#set text(size: 17pt)
#two(
  [
    #hairline([representation], [
      $h_0 = [e_1; e_2; dots; e_100] in RR^(100d)$
      #v(7pt)
      The input is now *100 times* the embedding width.
    ], color: TEAL)
  ],
  [
    #hairline([first MLP layer], [
      $W_1 in RR^(H times 100d)$
      #v(7pt)
      parameter count: $100 d H + H$
    ], color: BLUE)
  ],
)

#pause
#v(10pt)
#claim[
  If $d=256$ and $H=1024$, this one layer has about $26.2$ million weights.
]

== What if the window is 10,000 tokens? #V

#set text(size: 17pt)
#align(center, [
  #grid(
    columns: (1fr,) * 20,
    gutter: 1.2pt,
    ..range(20).map(i => l2-slot(
      if i == 9 or i == 10 { [⋯] } else { [$e$] },
      width: 100%, height: 8mm, size: 9pt,
      color: if i < 7 { TEAL } else if i > 12 { ACC } else { MUTED },
      fill: if i < 7 { PALE-TEAL } else if i > 12 { PALE-ACC } else { CREAM },
    )),
  )
  #v(3pt)
  #text(size: 11pt, fill: MUTED)[10,000 embedding slots, with most omitted here]
])

#pause
#v(9pt)
#three(
  card([width], [$10,000d$ input coordinates], color: RED, fill: PALE-RED),
  card([parameters], [$10,000 d H$ first-layer weights], color: RED, fill: PALE-RED),
  card([fixed size], [the network expects exactly 10,000 slots], color: RED, fill: PALE-RED),
)

== The window is part of the architecture

#set text(size: 17pt)
#align(center, grid(
  columns: (1fr, auto, 1fr, auto, 1fr),
  gutter: 9pt,
  align: center + horizon,
  card([window $k=3$], [$W_1 in RR^(H times 3d)$], color: TEAL, fill: PALE-TEAL),
  text(size: 24pt, fill: MUTED)[$eq.not$],
  card([window $k=5$], [$W_1 in RR^(H times 5d)$], color: BLUE, fill: PALE-BLUE),
  text(size: 24pt, fill: MUTED)[$eq.not$],
  card([window $k=100$], [$W_1 in RR^(H times 100d)$], color: ACC, fill: PALE-ACC),
))

#pause
#v(10pt)
#claim[
  Change the window, and we must change the network.
]

== Which context words help? #Q

#set text(size: 18pt)
#align(center, text(size: 22pt)[
  `I poured the coffee from the pot into the cup because the` #text(fill: ACC, weight: 700)[`___`]
])

#v(9pt)
#two(
  card([continuation A], [`cup was empty`\ Which noun matters most?], color: BLUE, fill: PALE-BLUE),
  card([continuation B], [`coffee was cold`\ Which noun matters most?], color: ACC, fill: PALE-ACC),
)

#pause
#v(10pt)
#note(color: BLUE)[
  Different predictions need different words from the context.
]

== Useful evidence can be far away #V

#set text(size: 17pt)
#align(center, [
  #grid(
    columns: (auto,) * 9,
    gutter: 4pt,
    align: center + horizon,
    ..([The], [keys], [to], [the], [old], [wooden], [cabinet], [are], [missing]).map(x => token(x, w: 18mm, hgt: 8mm, size: 11pt)),
  )
  #v(-1pt)
  #diagram(spacing: (18mm, 8mm), node-stroke: none, node-fill: none, {
    node((0, 0), [ ], stroke: none)
    node((7, 0), [ ], stroke: none)
    edge((1, 0), (7, 0), "-|>", bend: -32deg, stroke: 1.5pt + GREEN,
      label: text(size: 10.5pt, fill: GREEN)[subject controls verb])
  })
])

#pause
#align(center, text(size: 17pt, weight: 650, fill: GREEN)[`keys` is plural $arrow.r$ `are`. The nearer noun, `cabinet`, is singular.])

== Who does `she` refer to?

#set text(size: 17pt)
#align(center, text(size: 22pt)[
  `Alice texted Priya after` #text(fill: BLUE, weight: 700)[`she`] `finished the report.`
])

#v(10pt)
#two(
  card([candidate 1], [`she` $arrow.r$ Alice], color: TEAL, fill: PALE-TEAL),
  card([candidate 2], [`she` $arrow.r$ Priya], color: ACC, fill: PALE-ACC),
)

#pause
#v(9pt)
#note(color: PURPLE)[
  This sentence alone does not tell us who `she` is. Attention cannot resolve it without more evidence.
]

== A lookup embedding cannot see context #V

#set text(size: 17pt)
#align(center, grid(
  columns: (1fr, auto, 1fr),
  gutter: 16pt,
  align: center + horizon,
  card([river], [`They sat on the` *`bank`* `of the Narmada.`], color: TEAL, fill: PALE-TEAL),
  [
    #l2-slot([$e_"bank"$], color: INK, fill: CREAM, width: 26mm, height: 13mm)
    #v(2pt)
    #align(center, text(size: 11pt, fill: MUTED)[same table row])
  ],
  card([finance], [`She deposited money in the` *`bank`*`.`], color: ACC, fill: PALE-ACC),
))

#pause
#v(10pt)
#claim[
  We want the vector for `bank` to depend on the sentence.
]

== What do we want from a context summary?

#set text(size: 17pt)
#align(center, diagram(
  spacing: (18mm, 8mm),
  node-stroke: 0.9pt + TEAL,
  node-fill: PALE-TEAL,
  {
    for i in range(7) {
      node((i, 0), text(size: 11pt)[$v_#(i+1)$], radius: 5.5mm,
        fill: if i == 2 or i == 5 { PALE-ACC } else { PALE-TEAL },
        stroke: 0.9pt + if i == 2 or i == 5 { ACC } else { TEAL })
      edge((i, 0), (3, 1.55), stroke: 0.55pt + MUTED)
    }
    node((3, 1.55), text(size: 15pt, weight: 700)[$h$], radius: 7mm,
      fill: PALE-BLUE, stroke: 1.2pt + BLUE)
    node((3, 2.35), text(size: 10pt, fill: BLUE)[fixed $d$-vector], stroke: none)
  },
))

#pause
#v(8pt)
#claim[
  Combine any number of token vectors into one $d$-dimensional vector.
]

== First try: average the token vectors

#set text(size: 17pt)
#align(center, text(size: 23pt, weight: 650, fill: BLUE)[
  $h = 1/T sum_(j=1)^T v_j$
])

#v(5pt)
#l2-pool-diagram(
  ([the], [keys], [are]),
  (1/3, 1/3, 1/3),
  result: [mean $h$],
)

#pause
#v(5pt)
#note(color: GREEN)[
  The output has $d$ coordinates, however many vectors we average.
]

== Average one tiny example

#set text(size: 17pt)
#token-row((token([a]), token([a]), token([b])))

#v(8pt)
#align(center, text(size: 21pt)[
  $v_a=mat(2,0), quad v_b=mat(0,2)$
])

#pause
$ h = 1/3 (mat(2,0) + mat(2,0) + mat(0,2)) $

#pause
#claim(color: GREEN)[$h = mat(4/3, 2/3)$]

#align(center, text(size: 13pt, fill: MUTED)[The width stayed 2, even though three vectors went in.])

== The mean gives every token the same weight

#set text(size: 17pt)
#align(center, text(size: 21pt)[`keys to the cabinet are missing`])

#v(9pt)
#l2-weight-strip(
  ([keys], [to], [the], [cabinet], [are], [missing]),
  (1/6, 1/6, 1/6, 1/6, 1/6, 1/6),
)

#pause
#v(6pt)
#alertbox[
  Every word gets $1/6$. We cannot give `keys` more weight than the other words.
]

== The mean also forgets order #D

#set text(size: 18pt)
#two(
  [#token-row((token([dog]), token([bites]), token([man])))],
  [#token-row((token([man]), token([bites]), token([dog])))],
)

#pause
#v(12pt)
#align(center, text(size: 23pt, weight: 650, fill: RED)[
  $"mean"(e_"dog", e_"bites", e_"man") = "mean"(e_"man", e_"bites", e_"dog")$
])

#pause
#v(6pt)
#note([
  Both sequences have the same mean. Self-attention will need position information
  too. We will add it before computing Q, K, and V.
], color: PURPLE)

// ───────────────────────────── ACT 6 ─────────────────────────────

== Next try: use a weighted average

#set text(size: 18pt)
#align(center, text(size: 24pt, weight: 650, fill: BLUE)[
  $h = sum_(j=1)^T alpha_j v_j$
])

#v(4pt)
#l2-pool-diagram(
  ([the], [keys], [cabinet], [are]),
  (0.05, 0.65, 0.10, 0.20),
  result: [weighted $h$],
  colors: (MUTED, GREEN, TEAL, BLUE),
)

#pause
#v(4pt)
#claim[
  Each input contributes a different amount. The output still has $d$ coordinates.
]

== Read the weights as mixing amounts #V

#set text(size: 17pt)
#l2-weight-strip(
  ([the], [keys], [cabinet], [are]),
  (0.05, 0.65, 0.10, 0.20),
  colors: (MUTED, GREEN, TEAL, BLUE),
)

#pause
#v(8pt)
#two(
  note([Here, `keys` supplies most of the mixture.], color: GREEN),
  note([These weights show how one head mixes its values. They do not explain the whole model.], color: PURPLE),
)

== Who should choose the weights? #Q

#set text(size: 18pt)
#align(center, text(size: 23pt, weight: 650)[Where should $alpha_1,dots,alpha_T$ come from?])

#v(12pt)
#three(
  card([A · position], [always prefer nearby tokens], color: TEAL, fill: PALE-TEAL),
  card([B · identity], [give each word a fixed importance], color: ACC, fill: PALE-ACC),
  card([C · query], [score each source for this query], color: BLUE, fill: PALE-BLUE),
)

#pause
#v(10pt)
#claim(color: BLUE)[Use C: compute each weight from the query and the source.]

== One fixed set of weights is not enough

#set text(size: 16.5pt)
#grid(
  columns: (31mm, 1.55fr, .85fr),
  rows: (auto, auto, auto, auto),
  column-gutter: 10pt,
  row-gutter: 13pt,
  align: left + horizon,
  [], text(weight: 700, fill: BLUE)[question or task], text(weight: 700, fill: TEAL)[useful source],
  text(weight: 700)[sentiment], [`The plot was slow but the ending was` *`wonderful`*], [`wonderful`],
  text(weight: 700)[pronoun], [`Alice thanked Priya because` *`she`* `helped`], [`Alice` or `Priya`],
  text(weight: 700)[next token], [`The keys to the cabinet` *`are`*], [`keys`],
)

#pause
#v(8pt)
#alertbox[
  Which word helps depends on the question we are answering.
]

== Compute the weights from a query #V

#set text(size: 17pt)
#align(center, diagram(
  spacing: (20mm, 10mm),
  node-stroke: 0.9pt + INK,
  node-fill: white,
  {
    flow-node((0, 0), [current\ need], color: BLUE, fill: PALE-BLUE, w: 25mm)
    flow-node((1.55, 0), [query $q$], color: BLUE, fill: PALE-BLUE, w: 23mm)
    for (i, y) in ((0, -1.2), (1, 0), (2, 1.2)) {
      flow-node((3.25, y), [$k_#(i+1)$], color: TEAL, fill: PALE-TEAL, w: 18mm)
      flow-arrow((1.55, 0), (3.25, y), color: BLUE, label: [$s_#(i+1)$])
    }
    flow-node((5.0, 0), [softmax\ $alpha_1,alpha_2,alpha_3$], color: BLUE, fill: PALE-BLUE, w: 29mm)
    for y in (-1.2, 0, 1.2) { flow-arrow((3.25, y), (5.0, 0), color: MUTED) }
  },
))

#pause
#claim[
  Compare the query with each key, then apply softmax to the scores.
]

== Attention is a soft lookup

#set text(size: 15pt)
#two(
  [
    #grid(
      columns: (24mm, 5mm, 1fr),
      rows: (9mm,) * 5,
      gutter: 3pt,
      row-gutter: 7pt,
      align: horizon,
      text(weight: 650)[request], text(fill: MUTED)[$arrow.r$], text(weight: 650, fill: BLUE)[query $q$],
      text(weight: 650)[address], text(fill: MUTED)[$arrow.r$], text(weight: 650, fill: TEAL)[key $k_j$],
      text(weight: 650)[match], text(fill: MUTED)[$arrow.r$], text(weight: 650, fill: BLUE)[score $q^top k_j$],
      text(weight: 650)[payload], text(fill: MUTED)[$arrow.r$], text(weight: 650, fill: ACC)[value $v_j$],
      text(weight: 650)[return], text(fill: MUTED)[$arrow.r$], text(weight: 650, fill: PURPLE)[$sum_j alpha_j v_j$],
    )
  ],
  [
    #align(center, image("../figures/soft_lookup_illustration_imagen.png", height: 59mm, fit: "contain"))
    #v(-2pt)
    #grid(
      columns: (.21fr, .51fr, .28fr),
      align: center + horizon,
      text(size: 11pt, weight: 650, fill: BLUE)[query $q$],
      text(size: 11pt, weight: 650, fill: TEAL)[sources $(k_j, v_j)$],
      text(size: 11pt, weight: 650, fill: PURPLE)[weighted return],
    )
  ],
  ratio: (.88fr, 1.12fr),
  gutter: 10pt,
)

#pause
#v(5pt)
#claim(color: BLUE)[$j$ names the source. $alpha_j$ says how much of its value enters the return.]

== Query: what am I looking for?

#set text(size: 17pt)
#two(
  l2-qkv-card([$q$], [query], [what this token is looking for], BLUE, PALE-BLUE),
  [
    #hairline([examples], [
      - Which earlier noun controls this verb?
      - Which entity does this pronoun refer to?
      - What topic helps predict the next token?
    ], color: BLUE)
  ],
  ratio: (0.72fr, 1.28fr),
)

#pause
#v(9pt)
#claim(color: BLUE)[The network makes $q$ from the current token state.]

== Key: when should this source match?

#set text(size: 17pt)
#two(
  l2-qkv-card([$k$], [key], [match features carried by one source], TEAL, PALE-TEAL),
  [
    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 6pt,
      l2-slot([source 1\ $k_1$], color: TEAL, fill: PALE-TEAL, width: 100%, height: 16mm, size: 12pt),
      l2-slot([source 2\ $k_2$], color: TEAL, fill: PALE-TEAL, width: 100%, height: 16mm, size: 12pt),
      l2-slot([source 3\ $k_3$], color: TEAL, fill: PALE-TEAL, width: 100%, height: 16mm, size: 12pt),
    )
    #v(7pt)
    #align(center, text(size: 21pt, weight: 650)[$s_j = q^top k_j$])
    #v(6pt)
    #note(color: TEAL)[The small $j$ is the source number. A larger score means source $j$ matches the query better.]
  ],
  ratio: (0.72fr, 1.28fr),
)

== Value: what should this source send?

#set text(size: 17pt)
#two(
  l2-qkv-card([$v_j$], [value], [payload carried by source $j$], ACC, PALE-ACC),
  [
    #hairline([why separate it from the key?], [
      Matching and sending are different jobs.
      #v(7pt)
      Address: "I am a plural noun."\
      Payload: semantic and syntactic information.
    ], color: ACC)
  ],
  ratio: (0.72fr, 1.28fr),
)

#pause
#align(center, text(size: 17pt, weight: 650, fill: ACC)[Compare queries with keys. Use the resulting weights to mix values.])

== Make a query, key, and value

#set text(size: 17pt)
#l2-qkv-split()

#pause
#v(6pt)
#align(center, text(size: 20pt)[
  $q_i=h_i W_Q, quad k_i=h_i W_K, quad v_i=h_i W_V$
])

#pause
#note[
  $i$ is the token position. Multiply its vector by three learned matrices.
]

== Compute attention for one query #D

#set text(size: 17pt)
#align(center, text(size: 13.5pt, fill: MUTED)[
  For current token $i$, write $q=q_i$. Compare it with sources $j=1,dots,T$.
])
#v(4pt)
#l2-role-flow()

#pause
#v(8pt)
#align(center, grid(
  columns: (1fr, auto, 1fr, auto, 1fr),
  gutter: 7pt,
  align: center + horizon,
  dim-pill([$s_j=q^top k_j$], color: BLUE),
  text(fill: MUTED)[$arrow.r$],
  dim-pill([$alpha="softmax"(s)$], color: BLUE),
  text(fill: MUTED)[$arrow.r$],
  dim-pill([$h=sum_j alpha_j v_j$], color: ACC),
))

== Write the three steps in one line

#set text(size: 18pt)
#align(center, text(size: 25pt, weight: 650, fill: BLUE)[
  $"Attention"(q,K,V) = "softmax"(q K^top) V$
])

#v(9pt)
#two(
  card([one query], [$q$: one row\ for the current token], color: BLUE, fill: PALE-BLUE),
  card([memory], [$K,V$: $T$ rows\ one per source token], color: TEAL, fill: PALE-TEAL),
)

#pause
#v(8pt)
#row-col-note()

#align(center, text(size: 12.5pt, fill: MUTED)[For one query, softmax runs across the $T$ key columns.])

== One query in PyTorch

#set text(size: 16pt)
#torch-code(
  "# q: [d], K: [T, d], V: [T, dv]\nscores = q @ K.T\nweights = scores.softmax(dim=-1)\nout = weights @ V",
  takeaway: [Compare with keys, apply softmax, then mix values.],
)

#pause
#row-col-note()

// ───────────────────────────── ACT 7 ─────────────────────────────

== Attention on the character example

#set text(size: 17pt)
#two(
  [
    #hairline([concatenation], [
      #token-row((token([a]), token([a]), token([b])))
      #v(6pt)
      $[e_a; e_a; e_b] in RR^(3d)$
      #v(6pt)
      Three fixed slots.
    ], color: INK)
  ],
  [
    #hairline([attention], [
      #token-row((token([a]), token([a]), query-token([b])))
      #v(6pt)
      Use the query from `b` and a key and value from each visible token.
      #v(6pt)
      The output is one $d$-dimensional vector.
    ], color: BLUE)
  ],
)

#pause
#claim[We still predict `i` from `a a b`, now using attention to combine the context.]

== Pick tiny vectors we can check by hand #D

#set text(size: 16pt)
#align(center, grid(
  columns: (26mm, 1fr, 1fr, 1fr),
  rows: (auto, auto, auto, auto),
  gutter: 5pt,
  align: center + horizon,
  [], token([a₁]), token([a₂]), query-token([b₃]),
  text(weight: 700, fill: BLUE)[query], [—], [—], [$q_b=mat(0,1)$],
  text(weight: 700, fill: TEAL)[key], [$k_1=mat(1,0)$], [$k_2=mat(1,0)$], [$k_3=mat(0,1)$],
  text(weight: 700, fill: ACC)[value], [$v_1=mat(2,0)$], [$v_2=mat(2,0)$], [$v_3=mat(0,2)$],
))

#pause
#v(7pt)
#note[
  These are chosen numbers for a worked example. Leave out $1/sqrt(d_k)$ for now; we will include it later.
]

== What does the dot product notice?

#set text(size: 16pt)
#two(
  [
    #align(center, image("../figures/dot_product_alignment_imagen_v2.png", width: 100%, height: 66mm, fit: "contain"))
    #v(-2pt)
    #grid(
      columns: (1fr, 1fr),
      gutter: 8pt,
      align: center + horizon,
      text(size: 12pt, weight: 650, fill: TEAL)[perpendicular],
      text(size: 12pt, weight: 650, fill: PURPLE)[same direction],
    )
  ],
  [
    #hairline([query stays fixed], [
      #align(center, text(size: 22pt, weight: 650, fill: BLUE)[$q=mat(0,1)$])
    ], color: BLUE)
    #v(7pt)
    #note(color: TEAL)[
      #grid(
        columns: (1fr, auto),
        gutter: 8pt,
        align: horizon,
        [#text(size: 11.5pt, weight: 700, fill: TEAL)[PERPENDICULAR]\
         #text(size: 17pt, fill: TEAL)[$k_1=mat(1,0)$]],
        text(size: 20pt, weight: 650)[$q^top k_1=0$],
      )
    ]
    #v(7pt)
    #note(color: PURPLE)[
      #grid(
        columns: (1fr, auto),
        gutter: 8pt,
        align: horizon,
        [#text(size: 11.5pt, weight: 700, fill: PURPLE)[SAME DIRECTION]\
         #text(size: 17pt, fill: TEAL)[$k_3=mat(0,1)$]],
        text(size: 20pt, weight: 650, fill: BLUE)[$q^top k_3=1$],
      )
    ]
  ],
  ratio: (1.06fr, .94fr),
  gutter: 12pt,
)

#pause
#v(4pt)
#note(color: BLUE)[For unit vectors, the dot product measures directional agreement. Vector length matters in general.]

]

== Step 1: compare the query with each key #D

#set text(size: 17pt)
#align(center, grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  card([source $a_1$], [
    $s_1=q_b^top k_1$\
    $=mat(0,1) dot mat(1,0)$\
    $=0 times 1+1 times 0=0$
  ], color: TEAL, fill: PALE-TEAL),
  card([source $a_2$], [
    $s_2=q_b^top k_2$\
    $=mat(0,1) dot mat(1,0)$\
    $=0 times 1+1 times 0=0$
  ], color: TEAL, fill: PALE-TEAL),
  card([source $b_3$], [
    $s_3=q_b^top k_3$\
    $=mat(0,1) dot mat(0,1)$\
    $=0 times 0+1 times 1=1$
  ], color: BLUE, fill: PALE-BLUE),
))

#pause
#align(center, text(size: 22pt, weight: 700, fill: BLUE)[$s=mat(0,0,1)$])

== Step 2: turn scores into weights #D

#set text(size: 16.5pt)
$ alpha = "softmax"(0,0,1)
  = 1/(e^0+e^0+e^1) mat(e^0,e^0,e^1)
  = 1/(2+e) mat(1,1,e). $

#pause
#align(center, text(size: 20pt, weight: 650, fill: BLUE)[
  $alpha approx mat(0.212,0.212,0.576)$
])

#v(5pt)
#l2-weight-strip(([a₁], [a₂], [b₃]), (0.212, 0.212, 0.576), colors: (TEAL, TEAL, BLUE))

#pause
#align(center, text(size: 12pt, fill: MUTED)[$0.212+0.212+0.576=1.000$: one distribution across key columns.])

== Step 3: mix the values #D

#set text(size: 16.5pt)
$ h = 0.212 v_1 + 0.212 v_2 + 0.576 v_3 $

$ = 0.212 mat(2,0) + 0.212 mat(2,0) + 0.576 mat(0,2) $

#pause
$ = mat(0.424,0) + mat(0.424,0) + mat(0,1.152) $

#pause
#claim(color: ACC)[$h = mat(0.848,1.152) in RR^2$]

#align(center, text(size: 13pt, fill: MUTED)[This mixture differs from each of the three input values.])

== The context vector goes back to the language model

#set text(size: 17pt)
#simple-flow((
  ([#scale(x: 82%, y: 82%, reflow: true)[$h=mat(0.848,1.152)$]], ACC, PALE-ACC),
  ([MLP /\ LM head], BLUE, PALE-BLUE),
  ([27 logits], BLUE, PALE-BLUE),
  ([softmax], BLUE, PALE-BLUE),
  ([target `i`], ACC, PALE-ACC),
))

#pause
#v(7pt)
#prob-bars((([a], 0.05, MUTED), ([b], 0.08, TEAL), ([d], 0.07, TEAL), ([i], 0.72, ACC), ([other], 0.08, MUTED)), hgt: 22mm)

#pause
#note(color: ACC)[
  We changed the context vector. The loss is still $-log p_(theta)("i" | "a a b")$.
]

== What if we change the query? #D

#set text(size: 16.5pt)
Keep the keys and values. Use $q'=mat(1,0)$.

#v(5pt)
#two(
  card([old query $q_b$], [
    scores $mat(0,0,1)$\
    weights $mat(.212,.212,.576)$\
    output $mat(.848,1.152)$
  ], color: BLUE, fill: PALE-BLUE),
  card([new query $q'$], [
    scores $mat(1,1,0)$\
    weights $mat(.422,.422,.155)$\
    output $mat(1.688,.310)$
  ], color: GREEN, fill: PALE-GREEN),
)

#pause
#v(8pt)
#claim[Changing only the query changed both the weights and the output.]

== Why a soft lookup?

#set text(size: 17pt)
#two(
  card([hard], [
    choose $j^*=arg max_j s_j$\
    return only $v_(j^*)$
    #v(7pt)
    One winner.
  ], color: RED, fill: PALE-RED),
  card([soft], [
    $alpha="softmax"(s)$\
    return $sum_j alpha_j v_j$
    #v(7pt)
    A smooth mixture.
  ], color: GREEN, fill: PALE-GREEN),
)

#pause
#v(9pt)
#two(
  card([hard lookup], [The $arg max$ selects one value.], color: RED, fill: PALE-RED),
  card([soft lookup], [Differentiable weights let several sources contribute, including uncertain matches.], color: GREEN, fill: PALE-GREEN),
)

== Which parts are learned? #D

#set text(size: 17pt)
#align(center, grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  card([query projection], [$q_i=h_i W_Q$\ learns what to seek], color: BLUE, fill: PALE-BLUE),
  card([key projection], [$k_i=h_i W_K$\ learns how to match], color: TEAL, fill: PALE-TEAL),
  card([value projection], [$v_i=h_i W_V$\ learns what to send], color: ACC, fill: PALE-ACC),
))

#pause
#v(9pt)
#claim[
  We learn $W_Q,W_K,W_V$. We recompute scores for every sentence.
]

#pause
#align(center, text(size: 13pt, fill: MUTED)[Backpropagation learns the three projections together with the language model.])

== Shapes for one query #D

#set text(size: 17pt)
#align(center, grid(
  columns: (1fr, auto, 1fr, auto, 1fr),
  gutter: 7pt,
  align: center + horizon,
  card([query], [$q in RR^(d_k)$], color: BLUE, fill: PALE-BLUE),
  text(size: 22pt, fill: MUTED)[$times$],
  card([keys], [$K^top in RR^(d_k times T)$], color: TEAL, fill: PALE-TEAL),
  text(size: 22pt, fill: MUTED)[$arrow.r$],
  card([scores], [$s in RR^T$], color: BLUE, fill: PALE-BLUE),
))

#pause
#v(8pt)
#align(center, grid(
  columns: (1fr, auto, 1fr, auto, 1fr),
  gutter: 7pt,
  align: center + horizon,
  card([weights], [$alpha in RR^T$], color: BLUE, fill: PALE-BLUE),
  text(size: 22pt, fill: MUTED)[$times$],
  card([values], [$V in RR^(T times d_v)$], color: ACC, fill: PALE-ACC),
  text(size: 22pt, fill: MUTED)[$arrow.r$],
  card([output], [$h in RR^(d_v)$], color: GREEN, fill: PALE-GREEN),
))

// ───────────────────────────── ACT 8 ─────────────────────────────

== Why should only the final token ask? #Q

#set text(size: 17pt)
#token-row((query-token([a₁]), query-token([a₂]), query-token([b₃])))

#v(5pt)
#l2-mini-map(([a₁], [a₂], [b₃]), mode: "all", cell: 22mm)

#align(center, text(size: 11.5pt, fill: PURPLE)[each row is one query · each column is one possible source])

#pause
#v(5pt)
#claim(color: BLUE)[Compute a query at every position and compare it with all the keys.]

#pause
#v(6pt)
#align(center, text(size: 15pt, fill: MUTED)[
  When the sequence supplies its own queries, keys, and values, this is *self-attention*.
])

== Let every token make Q, K, and V #D

#set text(size: 16.5pt)
#align(center, diagram(
  spacing: (21mm, 11mm),
  node-stroke: .9pt + INK,
  node-fill: white,
  {
    flow-node((0, 0), [$X$\ $T times d$], color: INK, fill: CREAM, w: 24mm)
    for (y, w, out, shape, color, fill) in (
      (-1.25, [$W_Q$], [$Q$], [$T times d_k$], BLUE, PALE-BLUE),
      (0, [$W_K$], [$K$], [$T times d_k$], TEAL, PALE-TEAL),
      (1.25, [$W_V$], [$V$], [$T times d_v$], ACC, PALE-ACC),
    ) {
      flow-node((1.55, y), w, color: color, fill: white, w: 18mm)
      flow-node((3.05, y), [#out\ #shape], color: color, fill: fill, w: 25mm)
      flow-arrow((0, 0), (1.55, y), color: color)
      flow-arrow((1.55, y), (3.05, y), color: color)
    }
  },
))

#pause
#v(10pt)
#note[
  Each projection changes the vector width. We still have one row per token.
]

== Compare every query with every key #D

#set text(size: 16pt)
For the toy sequence, use
$ Q=K=mat(1,0; 1,0; 0,1). $

#pause
#two(
  [
    $S=Q K^top = mat(1,1,0; 1,1,0; 0,0,1)$
    #v(7pt)
    $S_(i j)=q_i^top k_j$
  ],
  matrix-table(
    ([q₁:a], [q₂:a], [q₃:b]),
    ([k₁:a], [k₂:a], [k₃:b]),
    (([1], [1], [0]), ([1], [1], [0]), ([0], [0], [1])),
    cell-fill: (i, j, v) => if i == j { PALE-BLUE } else { white },
    cell-width: 20mm,
    cell-height: 11mm,
    text-size: 13pt,
    label-size: 12pt,
  ),
  ratio: (0.85fr, 1.15fr),
)

#row-col-note()

== Softmax each query row #D

#set text(size: 16pt)
$ A_(i,:) = "softmax"(S_(i,:)) $

#v(5pt)
#matrix-table(
  ([q₁:a], [q₂:a], [q₃:b]),
  ([k₁:a], [k₂:a], [k₃:b]),
  (([.422], [.422], [.155]), ([.422], [.422], [.155]), ([.212], [.212], [.576])),
  cell-fill: (i, j, v) => if i == 2 { if j == 2 { BLUE.lighten(78%) } else { PALE-BLUE } } else { white },
  cell-width: 20mm,
  cell-height: 11mm,
  text-size: 13pt,
  label-size: 12pt,
)

#pause
#v(6pt)
#align(center, text(size: 16pt, weight: 650, fill: GREEN)[
  row sums: $approx 1.000$ (rounded) #h(20pt) softmax axis: *key columns*
])

== Mix values for every query #D

#set text(size: 16pt)
With $V=mat(2,0;2,0;0,2)$,

$ O=A V
 = mat(.422,.422,.155; .422,.422,.155; .212,.212,.576)
   mat(2,0;2,0;0,2). $

#pause
#claim(color: GREEN)[
  $O approx mat(1.688,.310; 1.688,.310; .848,1.152) in RR^(3 times 2)$
]

#pause
#align(center, text(size: 13pt, fill: MUTED)[Row $i$ of $O$ uses the weights computed for query $i$.])

== The full self-attention formula

#set text(size: 18pt)
#align(center, block(
  width: 100%,
  inset: (x: 18pt, y: 14pt),
  fill: PALE-BLUE,
  stroke: 2pt + BLUE,
  radius: 5pt,
  align(center, text(size: 27pt, weight: 700, fill: INK)[
    $"Attention"(Q,K,V) = "softmax"((Q K^top)/sqrt(d_k)) V$
  ]),
))

#pause
#v(6pt)
#torch-code(
  "Q, K, V = X @ Wq, X @ Wk, X @ Wv\nscores = Q @ K.T / K.size(-1) ** 0.5\nweights = scores.softmax(dim=-1)\nO = weights @ V",
  takeaway: [One sequence, without a batch dimension. Each row is a query; `dim=-1` runs across keys.],
)

#row-col-note()

== Each output vector depends on its context

#set text(size: 16.5pt)
#align(center, grid(
  columns: (1fr, auto, 1fr),
  gutter: 12pt,
  align: center + horizon,
  card([before attention], [
    occurrence 1: $e_a$\
    occurrence 2: $e_a$\
    same vocabulary-table row
  ], color: TEAL, fill: PALE-TEAL),
  text(size: 24pt, fill: MUTED)[$arrow.r$],
  card([after attention], [
    $o_i=sum_j A_(i j)v_j$\
    depends on the query at $i$ and its allowed sources
  ], color: GREEN, fill: PALE-GREEN),
))

#pause
#v(9pt)
#note(color: GREEN)[
  Position information will let two copies of `a` produce different queries. We have not added it yet.
]

== Attention alone does not know order #D

#set text(size: 17pt)
Let $P$ permute the token rows. Without a position signal or fixed mask,

#align(center, text(size: 22pt, weight: 650, fill: RED)[
  $"Attn"(P X) = P "Attn"(X)$
])

#pause
#two(
  [#token-row((token([dog]), token([bites]), token([man])))],
  [#token-row((token([man]), token([bites]), token([dog])))],
)

#pause
#v(8pt)
#alertbox[
  Reordering the tokens only reorders the outputs. The vectors contain no information about which token came first.
]

== Add position before making Q, K, and V #V

#set text(size: 16.5pt)
#align(center, grid(
  columns: (1fr, auto, 1fr, auto, 1fr),
  gutter: 10pt,
  align: center + horizon,
  card([token content], [$e(x_i)$\ "this is `a`"], color: TEAL, fill: PALE-TEAL),
  text(size: 24pt, fill: MUTED)[$plus$],
  card([position], [$p_i$\ "this is slot $i$"], color: PURPLE, fill: PALE-PURPLE),
  text(size: 24pt, fill: MUTED)[$arrow.r$],
  card([input state], [$h_i^((0))=E[x_i]+p_i$], color: INK, fill: CREAM),
))

#pause
#v(9pt)
#align(center, text(size: 20pt)[
  $q_i=h_i^((0)) W_Q, quad k_i=h_i^((0)) W_K, quad v_i=h_i^((0)) W_V$
])

#pause
#note(color: PURPLE)[
  Addition keeps the same width and makes position available to Q, K, and V.
]

== Two simple ways to add position

#set text(size: 15.5pt)
#grid(
  columns: (1fr, 1fr), gutter: 16pt, align: top,
  card([learned table], [
    $p_i=P[i]$
    #v(6pt)
    Learned from data.\
    A table lookup for each position.\
    Only the trained positions have entries.
  ], color: BLUE, fill: PALE-BLUE),
  card([sinusoidal], [
    $ p_(i,2r)=sin(i/10000^(2r/d)) $
    #v(5pt)
    $ p_(i,2r+1)=cos(i/10000^(2r/d)) $
    #v(7pt)
    Deterministic.\
    Defined beyond the training positions.
  ], color: ACC, fill: PALE-ACC),
)

#pause
#v(8pt)
#note[
  Both add order. Later models often use relative or rotary position instead.
]

// ───────────────────────────── ACT 9 ─────────────────────────────

== Can a training token peek at the answer? #Q

#set text(size: 17pt)
#align(center, text(size: 23pt)[
  input prefix: `a a` #h(12pt) target: #text(weight: 700, fill: ACC)[`b`]
])

#v(10pt)
#two(
  card([allowed], [`a₂` reads positions 1 and 2], color: GREEN, fill: PALE-GREEN),
  card([cheating], [`a₂` reads position 3 = target `b`], color: RED, fill: PALE-RED),
)

#pause
#v(10pt)
#claim(color: RED)[If the predictor can already see `b`, it can simply copy the answer.]

== Without a mask, future tokens leak in #V

#set text(size: 14.5pt)
#align(center, text(size: 16pt)[Sequence: `---aabid-`])
#v(4pt)
#l2-mini-map(([1:-], [2:-], [3:-], [4:a], [5:a], [6:b], [7:i], [8:d], [9:-]), mode: "leaky")

#v(4pt)
#align(center, text(size: 11.5pt, fill: PURPLE)[rows = query positions · columns = key positions · row-softmax across columns])

#pause
#v(5pt)
#alertbox[
  Every cell above the diagonal is a leak: query $i$ can read a future token $j>i$.
]

== Draw the causal mask #V

#set text(size: 14.5pt)
#l2-mini-map(([1:-], [2:-], [3:-], [4:a], [5:a], [6:b], [7:i], [8:d], [9:-]), mode: "causal")

#v(5pt)
#align(center, text(size: 11.5pt, fill: PURPLE)[rows = queries · columns = keys · row-softmax runs only over allowed columns])

#pause
#v(5pt)
#two(
  note([$j <= i$: allowed source, mask value $0$], color: GREEN),
  note([$j > i$: future source, mask value $-oo$], color: RED),
)

== Mask future scores before softmax #D

#set text(size: 15.5pt)
For query row 3 in a four-token sequence:

#align(center, grid(
  columns: (1fr, auto, 1fr, auto, 1fr),
  gutter: 7pt,
  align: center + horizon,
  card([scores $S_(3,:)$], [$mat(0.2,1.0,0.4,2.7)$], color: BLUE, fill: PALE-BLUE),
  text(size: 22pt, fill: MUTED)[$plus$],
  card([mask $M_(3,:)$], [$mat(0,0,0,-oo)$], color: RED, fill: PALE-RED),
  text(size: 22pt, fill: MUTED)[$arrow.r$],
  card([masked scores], [$mat(0.2,1.0,0.4,-oo)$], color: INK, fill: CREAM),
))

#pause
#v(7pt)
#claim(color: BLUE)[
  $"softmax"(0.2,1.0,0.4,-oo) approx mat(.225,.500,.275,0)$
]

#align(center, text(size: 12.5pt, fill: MUTED)[$exp(-oo)=0$: forbidden keys receive exactly zero probability.])

== The causal mask in PyTorch

#set text(size: 15.5pt)
#torch-code(
  "scores = Q @ K.T / K.size(-1) ** 0.5\nmask = torch.triu(torch.ones(T, T), diagonal=1).bool()\nscores = scores.masked_fill(mask, float('-inf'))\nweights = scores.softmax(dim=-1)\nout = weights @ V",
  takeaway: [One sequence; batch dimension omitted. Mask scores first, then softmax, then mix values.],
  color: RED,
)

#pause
#row-col-note()

== Read a masked attention matrix

#set text(size: 15.5pt)
#matrix-table(
  ([q₁], [q₂], [q₃], [q₄]),
  ([k₁], [k₂], [k₃], [k₄]),
  (([1.00], [0], [0], [0]),
   ([.45], [.55], [0], [0]),
   ([.225], [.500], [.275], [0]),
   ([.10], [.20], [.30], [.40])),
  cell-fill: (i, j, v) => if j > i { PALE-RED } else { PALE-BLUE },
  cell-width: 20mm,
  cell-height: 11mm,
  text-size: 13pt,
  label-size: 12pt,
)

#pause
#v(6pt)
#three(
  dim-pill([row 1 sees 1], color: BLUE),
  dim-pill([row 2 sees 1 and 2], color: BLUE),
  dim-pill([row $t$ sees 1 to $t$], color: BLUE),
)

#v(5pt)
#align(center, text(size: 12pt, fill: GREEN)[Every row sums to one across its allowed key columns.])

== One forward pass predicts every next token #V

#set text(size: 15.5pt)
#l2-parallel-row(
  ([`-`], [`-`], [`-`], [a], [a], [b], [i], [d]),
  ([`-`], [`-`], [a], [a], [b], [i], [d], [`-`]),
)

#pause
#v(8pt)
#align(center, text(size: 18pt, weight: 650)[
  state $h_t$ predicts shifted target $x_(t+1)$
])

#pause
#note(color: GREEN)[
  Each row sees only its prefix, but all rows are still computed together.
]

== Predict all six targets in parallel

#set text(size: 15pt)
#grid(
  columns: (1fr, auto, 1fr),
  gutter: 12pt,
  align: center + horizon,
  card([fixed-window MLP], [
    `--- → a`\
    `--a → a`\
    `-aa → b`\
    `aab → i`\
    `abi → d`\
    `bid → -`
  ], color: INK, fill: CREAM),
  text(size: 28pt, fill: MUTED)[$arrow.r.double$],
  card([causal self-attention], [
    One sequence: `---aabid`\
    #v(5pt)
    take losses at predictor positions 3 to 8\
    #v(5pt)
    targets: `a a b i d -`
  ], color: GREEN, fill: PALE-GREEN),
)

#pause
#v(8pt)
#claim[The six targets are unchanged. Each row now uses its whole allowed prefix.]

== Why scale the dot product? #D

#set text(size: 15.5pt)
If coordinates are roughly independent with unit variance,

#align(center, text(size: 21pt)[
  $q^top k = sum_(r=1)^(d_k) q_r k_r, quad "Var"(q^top k) approx d_k.$
])

#pause
#v(6pt)
#two(
  card([unscaled], [large $d_k$ $arrow.r$ large logits $arrow.r$ near one-hot softmax $arrow.r$ tiny gradients], color: RED, fill: PALE-RED),
  card([scaled], [$q^top k / sqrt(d_k)$ keeps score variance roughly constant], color: GREEN, fill: PALE-GREEN),
)

#pause
#v(7pt)
#claim(color: BLUE)[Divide by $sqrt(d_k)$ to keep the scores at a manageable scale.]

== Check the attention calculation

#set text(size: 13.5pt)
#grid(
  columns: (1fr, 1fr, 1fr), gutter: 10pt, align: top,
  hairline([1 · row sums], [
    #align(center, grid(
      columns: (15mm, 15mm, 15mm), gutter: 2pt,
      token([.2], w: 15mm, hgt: 9mm, size: 11pt),
      token([.3], w: 15mm, hgt: 9mm, size: 11pt),
      token([.5], w: 15mm, hgt: 9mm, size: 11pt),
    ))
    #v(6pt)
    #align(center, $sum_j A_(i j)=1$)
    #v(5pt)
    #align(center, text(size: 11pt, fill: GREEN)[test `A.sum(dim=-1)`])
  ], color: BLUE),
  hairline([2 · future stays zero], [
    #l2-mini-map(([1], [2], [3]), mode: "causal", cell: 9mm)
    #v(5pt)
    #align(center, text(size: 11pt, fill: RED)[inspect the upper triangle])
  ], color: RED),
  hairline([3 · matrix shapes], [
    #align(center, $underbrace((T times d_k)(d_k times T), Q K^top)$)
    #v(7pt)
    #align(center, $arrow.r T times T$)
    #v(7pt)
    #align(center, $underbrace((T times T)(T times d_v), A V)$)
    #v(7pt)
    #align(center, $arrow.r T times d_v$)
    #v(5pt)
    #align(center, text(size: 11pt, fill: GREEN)[write every multiplication])
  ], color: GREEN),
)

#pause
#v(8pt)
#claim[Rows are queries. Columns are keys. Softmax runs across each row.]

== What we built today

#set text(size: 16.5pt)
#align(center, grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  gutter: 5pt,
  align: center + horizon,
  stage-chip([1], [average], active: true, color: TEAL),
  text(fill: MUTED)[$arrow.r$],
  stage-chip([2], [weighted average], active: true, color: BLUE),
  text(fill: MUTED)[$arrow.r$],
  stage-chip([3], [query-dependent], active: true, color: BLUE),
  text(fill: MUTED)[$arrow.r$],
  stage-chip([4], [Q / K / V], active: true, color: ACC),
  text(fill: MUTED)[$arrow.r$],
  stage-chip([5], [causal self-attention], active: true, color: GREEN),
))

#pause
#v(10pt)
#claim[
  We still predict the next token. Each position now combines its context using attention weights.
]

#pause
#v(10pt)
#align(center, text(size: 22pt, weight: 700, fill: ACC)[
  Next: what if each token asks several different questions?
])
