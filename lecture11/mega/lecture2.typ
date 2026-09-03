// L11M, Lecture 2 — 62 conceptual frames, including the bank training modules.
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

// Keep the prediction target visible while recapping the three attention roles.
#let l2-bank-prediction() = align(center, text(size: 18pt)[
  `She deposited money in the` #text(fill: BLUE, weight: 650)[`bank`] #text(fill: ACC)[`___`]
])

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
    flow-arrow((1.5, 0), (3.3, -1), color: BLUE, label: [$q k_j^top$])
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
    flow-node((0, 0), [embedding $e_i$], color: INK, fill: CREAM, w: 28mm)
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
  ([concatenate\ $w d$ numbers], INK, CREAM),
  ([MLP], BLUE, PALE-BLUE),
  ([next-token\ softmax], ACC, PALE-ACC),
))

#v(8pt)
#token-row((token([a]), token([a]), token([b])), arrow-to: [i])

#pause
#v(7pt)
#note[
  Our window has $w=3$ tokens. What changes if we use five to predict the next one?
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
      $a_0 = [e_1, e_2, dots, e_100] in RR^(1 times 100d)$
      #v(7pt)
      The input is now *100 times* the embedding width.
    ], color: TEAL)
  ],
  [
    #hairline([first MLP layer], [
      $W_1 in RR^(100d times d_h)$
      #v(7pt)
      parameter count: $100 d d_h + d_h$
    ], color: BLUE)
  ],
)

#pause
#v(10pt)
#claim[
  If $d=256$ and hidden width $d_h=1024$, this layer has about $26.2$ million weights.
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
  card([parameters], [$10,000 d d_h$ first-layer weights], color: RED, fill: PALE-RED),
  card([fixed size], [the network expects exactly 10,000 slots], color: RED, fill: PALE-RED),
)

== The window is part of the architecture

#set text(size: 17pt)
#align(center, grid(
  columns: (1fr, auto, 1fr, auto, 1fr),
  gutter: 9pt,
  align: center + horizon,
  card([window $w=3$], [$W_1 in RR^(3d times d_h)$], color: TEAL, fill: PALE-TEAL),
  text(size: 24pt, fill: MUTED)[$eq.not$],
  card([window $w=5$], [$W_1 in RR^(5d times d_h)$], color: BLUE, fill: PALE-BLUE),
  text(size: 24pt, fill: MUTED)[$eq.not$],
  card([window $w=100$], [$W_1 in RR^(100d times d_h)$], color: ACC, fill: PALE-ACC),
))

#pause
#v(10pt)
#claim[
  Change the window, and we must change the network.
]

// One bank story: contrast contexts, then retain the finance prefix.
#let l2-bank-contrast() = grid(columns: (1fr, 1fr), gutter: 20pt,
  hairline([Finance context], [
    `She deposited` #text(fill: ACC, weight: 650)[`money`] `in the` *`bank`* `___`
  ], color: ACC),
  hairline([River context], [
    `She watched` #text(fill: TEAL, weight: 650)[`water`] `by the` *`bank`* `___`
  ], color: TEAL),
)
#let l2-bank-table(columns, ..cells) = grid(
  columns: columns, column-gutter: 14pt, row-gutter: 12pt,
  align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
  ..cells.pos(),
)
#let l2-state-path() = align(center, grid(
  columns: (44mm, 8mm, 48mm, 8mm, 44mm), gutter: 5pt, align: center + horizon,
  [$e_6$\ #text(size: 13pt)[lookup embedding]],
  [$+$],
  text(fill: GREEN)[$Delta e_6$\ #text(size: 13pt)[update from context]],
  [$=$],
  text(fill: BLUE)[$e'_6$\ #text(size: 13pt)[context-enriched embedding]],
))

== Which context would help after `bank`? #Q

#set text(size: 18pt)
#align(center)[Both prefixes end in the same word. What information differs?]
#v(17pt)
#l2-bank-contrast()
#pause
#v(17pt)
#note(color: BLUE)[`bank` is already an input. We want to predict the token #text(weight: 650)[after it].]
#pause
#v(12pt)
#note[The earlier words tell us about different situations. A useful predictor should be able to use that difference.]

== The useful clue may lie outside the window #V

#set text(size: 17pt)
#align(center, text(size: 19pt)[`She deposited` #text(fill: ACC, weight: 650)[`money`] `after a long wait near the bank ___`])
#v(17pt)
#two(
  hairline([A three-token window sees], [
    #token-row((token([near]), token([the]), token([bank], color: BLUE, fill: PALE-BLUE)))
    #v(10pt)
    The finance-related clue `money` is outside this window.
  ], color: RED),
  uncover("2-")[#hairline([What we want to retain], [
    Information from `money` and `deposited`, even when other words intervene.
    #v(10pt)
    Distance alone does not tell us what matters.
  ], color: BLUE)],
)
#v(17pt)
#uncover("3-")[#note[Gather information from the allowed prefix without concatenating every embedding into a wider input.]]

== Context cannot supply a clue that is missing

#set text(size: 18pt)
#align(center, text(size: 23pt)[`She waited near the bank ___`])
#v(20pt)
#two(
  hairline([A financial institution?], [That is possible.], color: ACC),
  hairline([The side of a river?], [That is also possible.], color: TEAL),
)
#pause
#v(22pt)
#note[This prefix does not settle the meaning. A context mechanism can use available evidence; it cannot manufacture missing evidence.]
#pause
#v(12pt)
#note(color: BLUE)[Next, use two prefixes that do provide different clues.]

== Same lookup embedding, different context #V

#set text(size: 17pt)
#l2-bank-contrast()
#v(8pt)
#align(center, text(size: 22pt)[$e_6 = E[t_6] = e_"bank"$ #h(12pt) in both prefixes])
#pause
#v(3pt)
#note[The lookup uses the token ID, not its neighbours. Both occurrences start with the same bank embedding.]
#pause
#v(7pt)
#two(
  hairline([Desired finance-context result], [A representation informed by `money` and `deposited`.], color: ACC),
  hairline([Desired river-context result], [A representation informed by `water` and `watched`.], color: TEAL),
)
#v(7pt)
#align(center, text(size: 12pt, fill: MUTED)[Desired behaviour, not outputs measured from our untrained toy model.])

== Keep the embedding; add context #V

#set text(size: 18pt)
#l2-bank-prediction()
#v(12pt)
#align(center)[$e_6=E[t_6]$: the familiar lookup embedding for `bank` at position 6.]
#pause
#v(18pt)
#l2-state-path()
#pause
#v(16pt)
#note(color: BLUE)[What should we add to `bank`? Information from `money` and `deposited` can help this occurrence reflect its finance context.]
#v(9pt)
#align(center)[$e'_6 arrow.r "vocabulary predictor" arrow.r p("next token")$]
#v(7pt)
#align(center, text(size: 12pt, fill: MUTED)[The prime means after adding context, at the same position. This does not overwrite the table $E$.])

== What should context add to bank's embedding? #V

#set text(size: 17pt)
#block[
#set par(spacing: 5pt)
#l2-bank-prediction()
#v(10pt)
#align(center)[Use a toy embedding with *three coordinates*.]
#v(12pt)
#let update-row(label, entries, color) = align(center, grid(
  columns: (78mm, 33mm, 33mm, 33mm), gutter: 5mm, align: horizon,
  label,
  ..entries.map(entry => l2-slot(entry,
    color: color, fill: color.lighten(92%), width: 100%, height: 10mm, size: 22pt)),
))
#align(center, grid(
  columns: (78mm, 33mm, 33mm, 33mm), gutter: 5mm, align: center,
  [], ..(1, 2, 3).map(i => text(size: 13pt, fill: MUTED)[coordinate #i]),
))
#v(4pt)
#update-row([bank's embedding $e_6$], ([$1$], [$1$], [$0$]), TEAL)
#pause
#v(6pt)
#update-row([$+$ context update $Delta e_6$], ([$?$], [$?$], [$?$]), GREEN)
#v(6pt)
#update-row([$=$ enriched embedding $e'_6$], ([$1+?$], [$1+?$], [$0+?$]), BLUE)
#v(7pt)
#align(center)[We add matching coordinates: *three numbers + three numbers*.]
#v(3pt)
#align(center, text(size: 12pt, fill: MUTED)[Each ? is an update number we still need to compute, not a predicted token.])
#pause
#v(12pt)
#note(color: BLUE)[How can the *six input embeddings* supply these *three update numbers*?]
#v(5pt)
#align(center, text(weight: 650)[First try: average the six input embeddings.])
]

== Average the six input embeddings #D

#set text(size: 16pt)
#align(center)[First attempt: let every source send its original embedding into the update.]
#v(12pt)
#l2-bank-table((36mm, 1fr, 36mm, 1fr),
  [*input*], [*embedding*], [*input*], [*embedding*],
  [1 · She], [$e_1=mat(0,0,1)$], [4 · in], [$e_4=mat(0,1,0)$],
  [2 · deposited], [$e_2=mat(1,0,1)$], [5 · the], [$e_5=mat(0,0,0)$],
  [3 · money], [$e_3=mat(2,0,0)$], [6 · bank], [$e_6=mat(1,1,0)$],
)
#pause
#v(15pt)
#align(center, text(size: 22pt, fill: GREEN)[$Delta e_6 = 1/6 sum_(j=1)^6 e_j = 1/6 mat(4,2,2) = mat(2/3,1/3,1/3)$])
#pause
#v(12pt)
#note[We would add this vector to $e_6$. But can an equal average select the useful clues?]
#v(5pt)
#align(center, text(size: 12pt, fill: MUTED)[Hand-chosen numbers. We will reuse this same embedding table in the complete model.])

== Problem 1: equal weights cannot select a clue

#set text(size: 18pt)
#l2-bank-prediction()
#v(14pt)
#l2-weight-strip(
  ([She], [deposited], [money], [in], [the], [bank]),
  (1/6, 1/6, 1/6, 1/6, 1/6, 1/6),
)
#pause
#v(18pt)
#note(color: RED)[The mean gives `money` and `the` exactly the same multiplier: $1/6$.]
#pause
#v(10pt)
#note(color: BLUE)[If finance-related information is useful here, we want a way to give its sources more weight.]

== Problem 2: an average forgets order #D

#set text(size: 18pt)
#two(
  hairline([Who told whom?], [`Maya told Ravi about the bank`], color: ACC),
  hairline([Who told whom?], [`Ravi told Maya about the bank`], color: TEAL),
)
#pause
#v(19pt)
#align(center, text(size: 20pt)[$"same tokens" arrow.r "same mean" quad "same final token" arrow.r "same " e_"last"$])
#v(12pt)
#note(color: RED)[Same average update and the same final `bank` embedding—but different people are speaking.]
#pause
#v(12pt)
#note[We will add position information before the parallel model. First solve Problem 1: deciding which sources contribute more.]

== Give useful sources more weight

#set text(size: 18pt)
#l2-bank-prediction()
#v(18pt)
#align(center, text(size: 25pt, fill: BLUE)[$Delta e_6 = sum_(j=1)^6 alpha_j e_j$])
#v(12pt)
#align(center)[$alpha_j >= 0, quad sum_j alpha_j=1$]
#pause
#v(16pt)
#grid(columns: (1fr, 1fr), gutter: 16pt, align: top,
  hairline([The old mean], [All six multipliers are $1/6$.], color: TEAL),
  hairline([A weighted update], [The multiplier for `money` can differ from the multiplier for `the`.], color: BLUE),
)
#pause
#v(18pt)
#note[For now, sources send their embeddings unchanged. The weights decide how much each source contributes. How should we choose them?]

== The receiving position needs a say #Q

#set text(size: 18pt)
#l2-bank-prediction()
#v(15pt)
- *Distance alone?* The nearest word is `the`. Nearby does not necessarily mean informative.

#pause
#v(10pt)
- *Fixed importance per word?* Always preferring `money` cannot express every position's information needs.

#pause
#v(10pt)
- *Match the current need.* Use the representation at `bank` to form a request. Compare it with each source's matching features.

#v(15pt)
#note(color: BLUE)[Weights should depend on both the receiving position and the source.]

== What information would help after bank? #Q

#set text(size: 18pt)
#block[
#set par(spacing: 6pt)
#l2-bank-prediction()
#v(10pt)
#align(center)[`bank` is known input 6. We want to predict the token #text(weight: 650)[after it].]
#v(7pt)
#align(center)[First, which clues could help enrich bank's embedding?]
#pause
#v(16pt)
#grid(columns: (1fr, 48mm), column-gutter: 20pt, row-gutter: 12pt,
  [*Possible information to seek*], [*Clue in prefix*],
  [Is there finance-related information?], text(fill: TEAL, weight: 650)[`money`],
  [What action happened?], text(fill: TEAL, weight: 650)[`deposited`],
  [Who is involved?], text(fill: TEAL, weight: 650)[`She`],
)
#pause
#v(19pt)
#note(color: BLUE)[A *query* encodes a request as numbers. For our toy calculation, let $q_6$ seek finance-related clues.]
#v(8pt)
#align(center, text(size: 12pt, fill: MUTED)[These questions explain possible roles. The model computes a vector, not an English question.])
]

== Turn bank's embedding into a query #D

#set text(size: 18pt)
#block[
#set par(spacing: 5pt)
#align(center)[The query comes from the #text(weight: 650)[known bank embedding], not from the blank.]
#v(13pt)
#align(center, text(size: 23pt)[$q_6 = underbrace(mat(1,1,0), e_6 quad (1 times 3)) underbrace(mat(1,0,0;0,0,0;0,0,0), W_Q quad (3 times 3))$])
#v(8pt)
#align(center, text(size: 13pt, fill: MUTED)[Our toy $W_Q$ is shared and trainable. It keeps coordinate 1 only because of the starting entries we chose.])
#pause
#v(12pt)
#grid(columns: (40mm, 1fr), column-gutter: 12pt, row-gutter: 5pt,
  [coordinate 1], [$1 times 1 + 1 times 0 + 0 times 0 = 1$],
  [coordinate 2], [$1 times 0 + 1 times 0 + 0 times 0 = 0$],
  [coordinate 3], [$1 times 0 + 1 times 0 + 0 times 0 = 0$],
)
#v(10pt)
#align(center, text(size: 23pt, fill: BLUE)[$q_6=mat(1,0,0)$ #h(10pt) #text(size: 14pt)[query shape: $1 times 3$]])
#pause
#v(8pt)
#torch-code("e6 = torch.tensor([[1., 1., 0.]])          # [1, 3]\nW_Q = torch.diag(torch.tensor([1., 0., 0.]))\nq6 = e6 @ W_Q                              # [[1., 0., 0.]]")
#v(5pt)
#align(center, text(size: 11.5pt, fill: MUTED)[Row-by-column teaching approach: #link("https://www.youtube.com/watch?v=vkhPtpUiLd8&t=199s")[Visual AI, 3:19]. Bank numbers are our toy example.])
]

== A key describes what a source can match #V

#set text(size: 17pt)
#block[
#set par(spacing: 6pt)
#align(center)[The query describes the request. Each source supplies a *key* to match against it.]
#v(11pt)
#align(center, text(size: 22pt, fill: TEAL)[$k_j=e_j W_K$])
#align(center)[For this toy, $W_K=I$: the identity map leaves all three coordinates unchanged.]
#pause
#v(15pt)
#grid(columns: (43mm, 1fr, 1fr), column-gutter: 17pt, row-gutter: 12pt,
  align: (x,y) => if x == 0 { left + horizon } else { center + horizon },
  [*source*], [*lookup embedding*], text(fill: TEAL)[*matching key*],
  [3 · `money`], [$e_3=mat(2,0,0)$], text(fill: TEAL)[$k_3=mat(2,0,0)$],
  [5 · `the`], [$e_5=mat(0,0,0)$], text(fill: TEAL)[$k_5=mat(0,0,0)$],
  [6 · `bank`], [$e_6=mat(1,1,0)$], text(fill: TEAL)[$k_6=mat(1,1,0)$],
)
#pause
#v(18pt)
#note(color: BLUE)[For intuition, call matching coordinate 1 a *finance-related cue*. Our query $(1,0,0)$ reads this coordinate and ignores the other two.]
#v(8pt)
#align(center, text(size: 12pt, fill: MUTED)[Hand-chosen features, not discovered meanings. $W_K$ is trainable; identity is only our starting choice.])
]

== Match the request, not another copy of bank #D

#set text(size: 17pt)
#block[
#set par(spacing: 6pt)
#align(center, text(size: 21pt, fill: BLUE)[$q_6=mat(1,0,0)$ #h(10pt) seek the first matching cue])
#v(9pt)
#align(center, text(size: 13pt)[Three highlighted sources for now. The later softmax will use #text(weight: 650)[all six].])
#v(18pt)
#grid(columns: (34mm, 34mm, 1fr, 17mm), column-gutter: 11pt, row-gutter: 15pt,
  align: (x,y) => if x == 0 { left + horizon } else { center + horizon },
  [*source*], text(fill: TEAL)[*key*],
  uncover("2-")[#text(fill: BLUE)[$q_6 dot k_j$]], uncover("2-")[*score*],
  [3 · `money`], [$mat(2,0,0)$],
  uncover("2-")[$1 times 2 + 0 times 0 + 0 times 0$], uncover("2-")[#text(fill: BLUE)[$2$]],
  [5 · `the`], [$mat(0,0,0)$],
  uncover("2-")[$1 times 0 + 0 times 0 + 0 times 0$], uncover("2-")[#text(fill: BLUE)[$0$]],
  [6 · `bank`], [$mat(1,1,0)$],
  uncover("2-")[$1 times 1 + 0 times 1 + 0 times 0$], uncover("2-")[#text(fill: BLUE)[$1$]],
)
#v(17pt)
#uncover("3-")[
  #note(color: BLUE)[`money` matches the requested direction more strongly than bank's own key. We compare $q_6$ with $k_6$—not with itself.]
  #v(8pt)
  #align(center, text(size: 13pt)[These are match scores, not attention weights or next-token probabilities.])
]
#v(7pt)
#align(center, text(size: 12pt, fill: MUTED)[Unscaled dot products for now: a three-coordinate query and key give one score.])
]

== The same query can find different evidence #Q

#set text(size: 18pt)
#block[
#set par(spacing: 6pt)
#align(center)[`She` #text(fill: TEAL, weight: 650)[`deposited money`] `in the` #text(fill: BLUE)[`bank`] #text(fill: ACC)[`___`]]
#v(10pt)
#align(center)[`She` #text(fill: TEAL, weight: 650)[`watched water`] `by the` #text(fill: BLUE)[`bank`] #text(fill: ACC)[`___`]]
#v(16pt)
#align(center)[Same word at position 6. Does the query already know which meaning applies?]
#pause
#v(18pt)
#align(center, text(size: 22pt)[$underbrace(e_6=mat(1,1,0), "same lookup") quad arrow.r^("same " W_Q) quad underbrace(q_6=mat(1,0,0), "same query")$])
#v(12pt)
#note(color: BLUE)[Not at this first step: $q_6$ comes from bank's lookup embedding. It does *not* yet summarize the whole prefix.]
#pause
#v(16pt)
#note(color: TEAL)[The source embeddings differ, so their keys can differ. The same query can find different matches in the two prefixes.]
#v(10pt)
#align(center)[Matching decides *where* to get information. Next: what should each source *send*?]
#v(7pt)
#align(center, text(size: 12pt, fill: MUTED)[Qualitative comparison only. We have not trained this toy model to distinguish both meanings.])
]

== Value: what should this source send? #V

#set text(size: 17pt)
#align(center)[Earlier, a source sent its embedding unchanged. Now learn #text(weight: 650)[what it should send].]
#v(12pt)
#align(center, text(size: 23pt)[$e_6=mat(1,1,0)$ #h(10pt) the lookup embedding for `bank`])
#pause
#v(15pt)
#two(
  hairline([Key: help choose an amount], [
    #text(size: 22pt, fill: TEAL)[$k_6=e_6 W_K=mat(1,1,0)$]
    #v(9pt)
    Compare this with a query.
  ], color: TEAL),
  hairline([Value: supply the information], [
    #text(size: 22pt, fill: ACC)[$v_6=e_6 W_V=mat(1,2,0)$]
    #v(9pt)
    Information to add at the receiver.
  ], color: ACC),
)
#pause
#v(13pt)
#note(color: ACC)[$e_j$ represents the source token. $v_j=e_j W_V$ is the information it sends—not its finished context-enriched embedding.]
#v(7pt)
#align(center, text(size: 12pt, fill: MUTED)[Here $W_K=I$ and $W_V="diag"(1,2,1)$. If $W_V=I$, sending $v_j=e_j$ is a valid special case.])

== Match queries and keys; mix the values #V

#set text(size: 17pt)
#l2-bank-prediction()
#v(13pt)
#align(center, diagram(spacing: (14mm, 13mm), {
  flow-node((0,0), [query $q_6$], color: BLUE, fill: PALE-BLUE, w: 28mm)
  flow-node((0,1.3), [source keys $k_j$], color: TEAL, fill: PALE-TEAL, w: 32mm)
  flow-node((2.2,0), [match scores], color: BLUE, fill: PALE-BLUE, w: 30mm)
  flow-node((4.4,0), [softmax\ weights $alpha_(6j)$], color: BLUE, fill: PALE-BLUE, w: 38mm)
  flow-node((2.2,1.8), [source values $v_j$], color: ACC, fill: PALE-ACC, w: 36mm)
  flow-node((4.4,1.8), [mix values\ message $h_6$], color: GREEN, fill: PALE-GREEN, w: 38mm)
  flow-arrow((0,0),(2.2,0),color: BLUE)
  flow-arrow((0,1.3),(2.2,0),color: TEAL)
  flow-arrow((2.2,0),(4.4,0),color: BLUE)
  flow-arrow((4.4,0),(4.4,1.8),color: BLUE)
  flow-arrow((2.2,1.8),(4.4,1.8),color: ACC)
}))
#pause
#v(12pt)
#note[The weight computed using $k_3$ multiplies $v_3$: the key and value come from the #text(weight: 650)[same source position].]
#pause
#v(8pt)
#align(center, text(size: 14pt)[$alpha_(6j)$: receiver 6, source $j$. The same source embedding creates its key and value.])

== Turn the collected message into an embedding update #D

#set text(size: 18pt)
#l2-bank-prediction()
#v(12pt)
#align(center, text(size: 25pt, fill: GREEN)[$h_6 = sum_(j=1)^6 alpha_(6j) v_j$])
#v(8pt)
#align(center, text(size: 15pt)[$h_6$: the message gathered for `bank` using its query—not a new token.])
#pause
#v(12pt)
#note(color: GREEN)[For now, use this message directly as the update: $Delta e_6=h_6$. Later we will learn a map from message to update.]
#pause
#v(12pt)
#l2-state-path()
#v(10pt)
#align(center, text(size: 14pt)[Information from `money` and `deposited` can now enter $e'_6$. The predictor reads this enriched embedding.])

== Same matches can carry different messages #Q

#set text(size: 17pt)
#align(center)[A controlled experiment: keep all inputs, queries and keys fixed.]
#v(5pt)
#two(
  hairline([Original value transformation], [
    Scale coordinates by $(1,2,1)$.
    #v(5pt)
    `money` sends $v_3=mat(2,0,0)$.
  ], color: ACC),
  uncover("2-")[#hairline([Stop sending coordinate 1], [
    Scale coordinates by $(0,2,1)$.
    #v(5pt)
    `money` now sends $tilde(v)_3=mat(0,0,0)$.
  ], color: PURPLE)],
)
#v(5pt)
#uncover("2-")[#align(center)[Do the matches change? Does the message change?]]
#v(7pt)
#uncover("3-")[#note(color: BLUE)[Scores and weights stay the same. Every value's first coordinate becomes zero, so the message's first coordinate becomes zero too.]]
#v(5pt)
#uncover("3-")[#note(color: GREEN)[The original $e_6$ still reaches the addition. Suppressing incoming information does not erase what `bank` already carries.]]
#v(6pt)
#align(center, text(size: 11.5pt, fill: MUTED)[This deliberately changes $W_V$ only. It is not a word replacement or an optimizer step. It is verified in the runnable companion.])

== A contextual update is not a learning update

#set text(size: 17pt)
#two(
  hairline([While reading this prefix], [
    $e_6 arrow.r e'_6$
    #v(13pt)
    Compute a new representation for #text(weight: 650)[this occurrence] of `bank`.
    #v(10pt)
    The stored table $E$ and all matrices stay unchanged.
  ], color: BLUE),
  uncover("2-")[#hairline([While training the model], [
    $E, W_Q, W_K, W_V, dots$
    #v(13pt)
    A loss guides updates to shared #text(weight: 650)[parameters].
    #v(10pt)
    The next forward pass recomputes the intermediate vectors.
  ], color: ACC)],
)
#v(21pt)
#uncover("3-")[#note[The two bank prefixes can start with the same lookup row—and even the same first-layer query. Their different source vectors can produce different messages.]]

== Our `bank` model will predict after adding context

#set text(size: 18pt)
#l2-bank-prediction()
#v(8pt)
#align(center, text(size: 23pt)[$e'_6 = e_6 + Delta e_6$])
#v(8pt)
#uncover("2-")[
  #align(center)[$Delta e_6 = h_6 W_O$]
  #v(10pt)
  #note[Start with $W_O=I$ and three coordinates throughout: the message itself is the update. Later, learn how to write it back.]
]
#v(8pt)
#uncover("3-")[
  #align(center)[$e'_6 arrow.r "vocabulary scores" arrow.r "next-token probabilities"$]
  #v(8pt)
  #note(color: BLUE)[Now put numbers on every step: query, keys, values, weights, message, addition, and prediction.]
]
#v(7pt)
#align(center, text(size: 11.5pt, fill: MUTED)[Return to the original value transformation (scale by 1, 2, 1) for the calculation below.\
This small residual-attention model will become a full Transformer block in Lecture 3.])

#include "bank-worked.typ"
#include "bank-training.typ"
#include "bank-parallel.typ"
]
