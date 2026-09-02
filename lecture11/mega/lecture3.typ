// L11M, Lecture 3 — 54 conceptual frames.
// All figures in this file are native Typst/Fletcher vectors.
#import "../../common/metropolis.typ": *
#import "../../common/mldiag.typ": *
#import "helpers.typ": *

#let head-card(number, question, detail, color: TEAL, fill: PALE-TEAL) = block(
  width: 100%, inset: (x: 4pt, y: 3pt),
  [
    #text(size: 10pt, weight: 750, fill: color)[HEAD #number]
    #v(2pt)
    #box(fill: fill, inset: (x: 3pt, y: 1pt), text(size: 15pt, weight: 650)[#question])
    #v(3pt)
    #line(length: 58%, stroke: 1.2pt + color)
    #v(5pt)
    #text(size: 11pt, fill: MUTED)[#detail]
  ],
)

#let head-output(label, color, fill) = block(
  width: 31mm, height: 14mm, inset: 5pt, fill: fill,
  stroke: 1pt + color, radius: 3pt,
  align(center + horizon, text(size: 11pt, weight: 650, fill: color, label)),
)

#let mini-pipeline(items) = {
  let cells = ()
  for (i, item) in items.enumerate() {
    if i > 0 {
      cells.push(text(size: 15pt, fill: MUTED)[$arrow.r$])
    }
    cells.push(token(
      item.at(0), color: item.at(1), fill: item.at(2),
      w: 18mm, hgt: 9mm, size: 9pt,
    ))
  }
  align(center, grid(
    columns: cells.map(_ => auto), gutter: 2pt, align: horizon, ..cells,
  ))
}

#let weights-row(query, columns, values, emphasis: none, forbidden: none, color: BLUE) = {
  let cells = ()
  cells.push(align(right + horizon, text(size: 12pt, weight: 750, fill: TEAL)[keys $arrow.r$]))
  for column in columns {
    cells.push(align(center, text(size: 11.5pt, weight: 650, fill: TEAL, column)))
  }
  cells.push(align(right + horizon, text(size: 11.5pt, weight: 750, fill: BLUE)[#query $arrow.r$]))
  for (j, value) in values.enumerate() {
    let is-forbidden = forbidden != none and forbidden.contains(j)
    let is-emphasis = emphasis != none and emphasis.contains(j)
    let bg = if is-forbidden { PALE-RED } else if is-emphasis { color.lighten(83%) } else { white }
    let fg = if is-forbidden { RED } else if is-emphasis { color } else { INK }
    cells.push(block(
      width: 100%, height: 14mm, fill: bg, stroke: 0.55pt + rgb("#D9DEDC"),
      align(center + horizon, text(
        size: 13pt, weight: if is-emphasis { 750 } else { 500 }, fill: fg, value,
      )),
    ))
  }
  align(center, grid(
    columns: (25mm,) + (34mm,) * columns.len(),
    rows: (8mm, 14mm), gutter: 0pt, ..cells,
  ))
}

#let residual-diagram(sublayer, color: BLUE) = align(center, diagram(
  spacing: (23mm, 13mm),
  {
    flow-node((0, 0), [$x$], color: TEAL, fill: PALE-TEAL, w: 16mm, size: 15pt)
    flow-node((1.45, 0), sublayer, color: color, fill: color.lighten(88%), w: 28mm)
    flow-node((2.9, 0), [$+$], color: GREEN, fill: PALE-GREEN, w: 14mm, size: 17pt)
    flow-node((4.35, 0), [updated state], color: GREEN, fill: PALE-GREEN, w: 28mm)
    flow-arrow((0, 0), (1.45, 0), color: color)
    flow-arrow((1.45, 0), (2.9, 0), color: color)
    flow-arrow((2.9, 0), (4.35, 0), color: GREEN)
    edge((0, 0), (2.9, 0), "-|>", bend: -48deg, stroke: 1.15pt + GREEN,
      label: text(size: 9.5pt, weight: 650, fill: GREEN)[identity path])
  },
))

#let decoder-block-diagram() = align(center, diagram(
  spacing: (16.5mm, 11mm),
  {
    flow-node((0, 0), [$x$], color: TEAL, fill: PALE-TEAL, w: 13mm, size: 15pt)
    flow-node((1.2, 0), [LN], color: PURPLE, fill: PALE-PURPLE, w: 15mm)
    flow-node((2.4, 0), [causal #linebreak() MHA], color: BLUE, fill: PALE-BLUE, w: 23mm)
    flow-node((3.6, 0), [$+$], color: GREEN, fill: PALE-GREEN, w: 13mm, size: 16pt)
    flow-node((4.8, 0), [LN], color: PURPLE, fill: PALE-PURPLE, w: 15mm)
    flow-node((6, 0), [FFN], color: ACC, fill: PALE-ACC, w: 19mm)
    flow-node((7.2, 0), [$+$], color: GREEN, fill: PALE-GREEN, w: 13mm, size: 16pt)
    flow-node((8.4, 0), [$y$], color: INK, fill: CREAM, w: 13mm, size: 15pt)
    for i in range(7) {
      let col = if i == 2 or i == 5 { GREEN } else { MUTED }
      flow-arrow((i * 1.2, 0), ((i + 1) * 1.2, 0), color: col)
    }
    edge((0, 0), (3.6, 0), "-|>", bend: -42deg, stroke: 1.05pt + GREEN,
      label: text(size: 8.5pt, fill: GREEN)[residual])
    edge((3.6, 0), (7.2, 0), "-|>", bend: -42deg, stroke: 1.05pt + GREEN,
      label: text(size: 8.5pt, fill: GREEN)[residual])
  },
))

// Keep this exact drawing on the board while we reveal one multi-head step at
// a time. Only the part we are discussing changes colour.
#let multihead-diagram(focus: "split") = {
  let active(name, color) = if focus == name { color } else { MUTED }
  let paper(name, color, pale) = if focus == name { pale } else { white }
  align(center, diagram(
    spacing: (18.5mm, 12mm),
    {
      flow-node((0, 0), [$X$], color: TEAL, fill: PALE-TEAL, w: 17mm, size: 15pt)
      for (label, formula, y, color, pale) in (
        ([1], [Q / K / V], -1.15, TEAL, PALE-TEAL),
        ([2], [Q / K / V], 0, BLUE, PALE-BLUE),
        ([H], [Q / K / V], 1.15, ACC, PALE-ACC),
      ) {
        flow-node(
          (1.65, y),
          [#text(size: 7.5pt, weight: 650)[head #label] #linebreak() #text(size: 9pt, formula)],
          color: active("heads", color),
          fill: paper("heads", color, pale),
          w: 31mm,
        )
        flow-arrow((0, 0), (1.65, y), color: if focus == "split" { color } else { MUTED })
        flow-arrow((1.65, y), (3.35, 0), color: active("heads", color))
      }
      flow-node(
        (3.35, 0), [concat],
        color: active("concat", GREEN), fill: paper("concat", GREEN, PALE-GREEN), w: 22mm,
      )
      flow-node(
        (4.75, 0), [$W_O$],
        color: active("mix", ACC), fill: paper("mix", ACC, PALE-ACC), w: 18mm, size: 14pt,
      )
      flow-node(
        (6.15, 0), [$"MHA"(X)$],
        color: active("mix", GREEN), fill: paper("mix", GREEN, PALE-GREEN), w: 25mm, size: 13pt,
      )
      flow-arrow((3.35, 0), (4.75, 0), color: active("concat", GREEN))
      flow-arrow((4.75, 0), (6.15, 0), color: active("mix", GREEN))
    },
  ))
}

// The LM head really is a dense layer. Draw a few representative dimensions
// and vocabulary entries instead of hiding the neural network in a rectangle.
#let lm-head-diagram() = align(center, diagram(
  spacing: (20mm, 9mm),
  {
    let inputs = ([$h_1$], [$h_2$], [$h_3$], [$dots.v$], [$h_d$])
    let outputs = ([a], [b], [i], [$dots.v$], [z])
    for i in range(inputs.len()) {
      let y = i - 2
      neuron-node((0, y), body: inputs.at(i), color: TEAL, fill: PALE-TEAL)
    }
    for j in range(outputs.len()) {
      let y = j - 2
      for i in range(inputs.len()) {
        edge((0, i - 2), (2.3, y), stroke: .32pt + MUTED.lighten(45%))
      }
      neuron-node(
        (2.3, y), body: outputs.at(j),
        color: if j == 2 { ACC } else { BLUE },
        fill: if j == 2 { PALE-ACC } else { PALE-BLUE },
      )
    }
    node((0, -3.05), text(size: 10pt, weight: 650, fill: TEAL)[$d_"model"$ features], stroke: none)
    node((2.3, -3.05), text(size: 10pt, weight: 650, fill: BLUE)[$V$ logits], stroke: none)
  },
))

#let tiny-visibility(causal: false, color: TEAL) = align(center, grid(
  columns: (7mm,) * 4,
  rows: (7mm,) * 4,
  gutter: 1pt,
  ..range(16).map(n => {
    let i = calc.floor(n / 4)
    let j = calc.rem(n, 4)
    let visible = not causal or j <= i
    block(
      width: 7mm, height: 7mm,
      fill: if visible { color.lighten(84%) } else { PALE-RED },
      stroke: .45pt + white,
      align(center + horizon, text(size: 8pt, weight: 700, fill: if visible { color } else { RED }, if visible { [•] } else { [$times$] })),
    )
  }),
))

#let family-card(title, visibility, objective, example, color, fill, visual: none) = block(
  width: 100%, inset: (x: 5pt, y: 4pt),
  [
    #box(fill: fill, inset: (x: 4pt, y: 1.5pt), text(size: 14pt, weight: 750, fill: color)[#title])
    #v(3pt)
    #line(length: 80%, stroke: 1pt + color)
    #if visual != none {
      v(5pt)
      align(center, visual)
    }
    #v(6pt)
    #text(size: 10.5pt, weight: 650)[VISIBILITY]
    #linebreak()
    #text(size: 11pt)[#visibility]
    #v(5pt)
    #text(size: 10.5pt, weight: 650)[COMMON OBJECTIVE]
    #linebreak()
    #text(size: 11pt)[#objective]
    #v(5pt)
    #text(size: 10pt, fill: MUTED)[Example: #example]
  ],
)

= Lecture 3: From attention to the Transformer

// Act 10 · Multi-head attention · 105–116

== Recall one attention head

#claim[Each token gets a weighted combination of the visible values]

#v(8pt)

#align(center, $Q = X W_Q quad K = X W_K quad V = X W_V$)

#v(8pt)

#simple-flow((
  ([token states $X$], TEAL, PALE-TEAL),
  ([$Q K^top / sqrt(d_k)$], BLUE, PALE-BLUE),
  ([row-wise softmax], BLUE, PALE-BLUE),
  ([$A V$], ACC, PALE-ACC),
))

#pause

#note[One head makes one set of Q, K, and V vectors. Each row uses its own weights to combine the values it is allowed to see.]

== Why use more than one head?

#token-row((token([Priya], w: 18mm), token([opened], w: 23mm), token([notebook], w: 30mm), token([;], w: 8mm), token([she]), query-token([smiled], w: 23mm)))

#v(7pt)

#align(center, text(size: 13pt, fill: MUTED)[Several parts of this sentence could help us predict what comes after `smiled`.])

#v(7pt)

#three(
  card([nearby], [What just happened?], color: TEAL, fill: PALE-TEAL),
  card([who?], [Which earlier person does `she` refer to?], color: BLUE, fill: PALE-BLUE),
  card([structure], [Where did one clause end and another begin?], color: ACC, fill: PALE-ACC),
)

#pause

#claim[Can we learn a separate set of attention weights for each?]

== Give each token several attention heads

#three(
  head-card([1], [what is nearby?], [might use nearby tokens], color: TEAL, fill: PALE-TEAL),
  head-card([2], [which person?], [might connect distant tokens], color: BLUE, fill: PALE-BLUE),
  head-card([3], [where is the boundary?], [might use punctuation], color: ACC, fill: PALE-ACC),
)

#pause

#multihead-diagram(focus: "split")

== Each head learns its own projections

#align(center, $Q_h = X W_Q^((h)) comma quad K_h = X W_K^((h)) comma quad V_h = X W_V^((h))$)

#v(5pt)

#multihead-diagram(focus: "heads")

#v(5pt)

#pause

#claim[Every head receives the same $X$ and learns its own weights]

== One head may look nearby

#text(size: 15pt)[In this example, the row for `smiled` puts most of its weight on nearby `she`.]

#v(9pt)

#weights-row(
  [`smiled`],
  ([Priya], [opened], [notebook], [`;`], [she], [smiled]),
  ([.05], [.08], [.07], [.04], [.66], [.10]),
  emphasis: (4,), color: TEAL,
)

#v(9pt)

#row-col-note()

#pause

#note[Softmax runs across the six key columns in this row. The weights sum to $1$.]

== Another head may look back to a person

#text(size: 15pt)[A second head might connect `she` with `Priya`. Here is one possible row of weights.]

#v(9pt)

#weights-row(
  [`she`],
  ([Priya], [opened], [notebook], [`;`], [she], [smiled]),
  ([.61], [.12], [.10], [.07], [.10], [$0$]),
  emphasis: (0,), forbidden: (5,), color: BLUE,
)

#v(9pt)

#row-col-note()

#pause

#note([`smiled` is in the future for this row. The causal mask makes its attention weight exactly zero.], color: RED)

== A third head may notice a boundary

#text(size: 15pt)[For a third head, the row for `smiled` might put most of its weight on the semicolon.]

#v(9pt)

#weights-row(
  [`smiled`],
  ([Priya], [opened], [notebook], [`;`], [she], [smiled]),
  ([.07], [.08], [.09], [.58], [.13], [.05]),
  emphasis: (3,), color: ACC,
)

#v(9pt)

#row-col-note()

#pause

#note([We chose these weights to illustrate possible patterns. Training learns the heads across many examples; a head need not have one clear role.], color: PURPLE)

== The heads do the same calculation separately

#align(center, $"head"_h = "Attention"(Q_h, K_h, V_h) = "softmax"(Q_h K_h^top / sqrt(d_k) + M) V_h$)

#v(5pt)

#multihead-diagram(focus: "heads")

#v(4pt)

#align(center, grid(
  columns: (auto, 12mm, auto, 12mm, auto, 12mm, auto), align: horizon,
  dim-pill([$Q_h: T times d_k$], color: BLUE), [$times$],
  dim-pill([$K_h^top: d_k times T$], color: TEAL), [$arrow.r$],
  dim-pill([$A_h: T times T$], color: BLUE), [$times$],
  dim-pill([$V_h: T times d_v$], color: ACC),
))

#pause

#claim[$"head"_h in RR^(T times d_v)$]

== Put the head outputs side by side

#multihead-diagram(focus: "concat")

#v(8pt)

#claim[$C = "Concat"("head"_1, dots.c, "head"_H) in RR^(T times H d_v)$]

#pause

#note[For each token, write head 1's output, then head 2's output, and so on. Concatenation keeps all the features from every head.]

== Mix the concatenated features with a linear layer

#multihead-diagram(focus: "mix")

#v(10pt)

#align(center, $"MHA"(X) = "Concat"("head"_1, dots.c, "head"_H) W_O$)

#pause

#note[$W_O$ is a learned linear layer. It combines features from the heads and produces width $d_"model"$, which we need for the residual addition.]

== Check the widths before moving on

#two(
  [#card([common width choice], [
    $d_k = d_v = d_"model" / H$
    #v(6pt)
    If $d_"model"=512$ and $H=8$,
    #linebreak()
    then $d_k=d_v=64$.
  ], color: BLUE, fill: PALE-BLUE)],
  [#card([concatenation check], [
    $H d_v = 8 times 64 = 512$
    #v(6pt)
    so $C$ already has width $d_"model"$ before $W_O$ mixes it.
  ], color: GREEN, fill: PALE-GREEN)],
)

#pause

#claim[Each head produces 64 features. Concatenating eight heads gives 512.]

== What can an attention map tell us?

#three(
  card([inspect], [Look at weights, outputs, gradients, and how interventions change behavior.], color: TEAL, fill: PALE-TEAL),
  card([be careful], [A head may change roles across inputs. A large weight alone cannot prove causal importance.], color: RED, fill: PALE-RED),
  card([test], [Remove a component or change the input, then compare the model's behavior.], color: GREEN, fill: PALE-GREEN),
)

#pause

#claim[Attention weights help us inspect a model. Explaining its behavior needs further evidence.]

// Act 11 · Build one Transformer block · 117–129

== Attention mixes rows; the FFN mixes features

#two(
  [#card([attention], [
    Mix information between visible token rows.
  ], color: BLUE, fill: PALE-BLUE)],
  [#card([feed-forward network], [
    Recombine the features inside each row with a nonlinear map.
  ], color: ACC, fill: PALE-ACC)],
)

#pause

#note([A block first mixes information across token rows, then transforms the features within each row.], color: PURPLE)

== Give every token the same small neural network

#two(
  [
    #mlp-diagram((3, 5, 3), labels: ([$d_"model"$], [$d_"ff"$], [$d_"model"$]))
    #v(4pt)
    #align(center, $"FFN"(x) = W_2 phi(W_1 x + b_1) + b_2$)
  ],
  [
    #mini-pipeline((([$x_1$], TEAL, PALE-TEAL), ([same FFN], ACC, PALE-ACC), ([$y_1$], GREEN, PALE-GREEN)))
    #v(9pt)
    #mini-pipeline((([$x_2$], TEAL, PALE-TEAL), ([same FFN], ACC, PALE-ACC), ([$y_2$], GREEN, PALE-GREEN)))
    #v(9pt)
    #mini-pipeline((([$x_T$], TEAL, PALE-TEAL), ([same FFN], ACC, PALE-ACC), ([$y_T$], GREEN, PALE-GREEN)))
  ],
  ratio: (1.15fr, 1fr),
)

#pause

#claim[The same MLP weights are used at every position]

== Across tokens, then within a token

#two(
  [
    #hairline([MHA · across rows], [
      #matrix-table(
        ([query 1], [query 2], [query 3]),
        ([key 1], [key 2], [key 3]),
        (([.7], [.2], [.1]), ([.1], [.7], [.2]), ([.2], [.3], [.5])),
        cell-fill: (i, j, value) => if i == j { PALE-BLUE } else { white },
        cell-width: 13mm,
      )
      #v(5pt)
      #align(center, text(size: 11pt, fill: MUTED)[token positions exchange information])
    ], color: BLUE)
  ],
  [
    #hairline([FFN · within one row], [
      #simple-flow((
        ([$d_"model"$], TEAL, PALE-TEAL),
        ([$d_"ff"$], ACC, PALE-ACC),
        ([$d_"model"$], GREEN, PALE-GREEN),
      ))
      #v(5pt)
      #align(center, text(size: 11pt, fill: MUTED)[a nonlinear transformation of one row's features])
    ], color: ACC)
  ],
)

#pause

#claim[Attention combines rows. The FFN transforms each row separately.]

== Why does the FFN widen in the middle?

#align(center, scale(x: 76%, y: 76%, reflow: true,
  mlp-diagram((3, 7, 3), labels: ([$512$ input features], [$2048$ hidden features], [$512$ output features]))
))

#pause

#claim[$512 arrow.r 2048 arrow.r 512$: compute more nonlinear features, then return to the original width.]

== Bring back the ResNet shortcut

#align(center, text(size: 25pt, weight: 650)[Can a sublayer pass its input through unchanged?])

#pause

#claim[$y = x + F(x)$]

#v(10pt)

#two(
  card([keep the input], [If $F(x) approx 0$, then $y approx x$. The input passes straight through.], color: GREEN, fill: PALE-GREEN),
  card([learn an update], [The sublayer learns the correction $F(x)$ to add to its input.], color: BLUE, fill: PALE-BLUE),
)

== Add the attention output to its input

#residual-diagram([causal MHA], color: BLUE)

#v(8pt)

#claim[$r = x + "MHA"(x)$]

#pause

#note[Shapes must match: both $x$ and $"MHA"(x)$ are $T times d_"model"$.]

== Add the FFN output to its input

#residual-diagram([position-wise FFN], color: ACC)

#v(8pt)

#claim[$y = r + "FFN"(r)$]

#pause

#note([The residual stream keeps width $d_"model"$ through every sublayer and every block.], color: GREEN)

== LayerNorm works across one token's features

#align(center, text(size: 14pt, fill: MUTED)[Start with one token row. Subtract its mean, then divide by its standard deviation.])

#v(7pt)

#simple-flow((
  ([$mat(1,3,5)$], TEAL, PALE-TEAL),
  ([subtract mean $3$], PURPLE, PALE-PURPLE),
  ([$mat(-2,0,2)$], BLUE, PALE-BLUE),
  ([divide by $sqrt(8/3)$], PURPLE, PALE-PURPLE),
  ([#scale(x: 70%, y: 70%, reflow: true)[$approx mat(-1.22,0,1.22)$]], GREEN, PALE-GREEN),
))

#pause
#v(9pt)

#two(
  card([general row $x_i$], [$"LN"(x_i)=gamma dot (x_i-mu_i)/sqrt(sigma_i^2+epsilon)+beta$], color: PURPLE, fill: PALE-PURPLE),
  card([one row at a time], [Each token row has its own feature mean and variance. Learned $gamma,beta$ are shared across positions.], color: TEAL, fill: PALE-TEAL),
)

#pause
#claim[Each token row is normalized separately]

== Where should LayerNorm go?

#two(
  card([pre-norm], [
    $y = x + F("LN"(x))$
    #v(7pt)
    Normalize *before* the sublayer.
  ], color: GREEN, fill: PALE-GREEN),
  card([post-norm], [
    $y = "LN"(x + F(x))$
    #v(7pt)
    Normalize *after* residual addition.
  ], color: MUTED, fill: CREAM),
)

#pause

#claim[We will use pre-norm in our block]

#note[Both versions are used. In the pre-norm drawing, we can follow the residual path around each sublayer.]

== Put the sublayers together

#decoder-block-diagram()

#v(9pt)

#three(
  dim-pill([same $T times d_"model"$ stream], color: TEAL),
  dim-pill([future positions masked], color: RED),
  dim-pill([two residual additions], color: GREEN),
)

#pause

#claim[attention $arrow.r$ add input $arrow.r$ FFN $arrow.r$ add input]

== Write the block as two updates

#v(5pt)

#card([attention sublayer], [
  $u = x + underbrace("MHA"("LN"(x)), "causal retrieval")$
], color: BLUE, fill: PALE-BLUE)

#pause

#v(10pt)

#card([feed-forward sublayer], [
  $y = u + underbrace("FFN"("LN"(u)), "per-token transformation")$
], color: ACC, fill: PALE-ACC)

#pause

#note([$x,u,y in RR^(T times d_"model")$. The two green plus signs in the drawing are exactly the two additions above.], color: GREEN)

== The block in five lines of PyTorch-style code

#torch-code(
  "class Block(nn.Module):\n    def forward(self, x):\n        x = x + self.causal_attn(self.ln1(x))\n        x = x + self.ffn(self.ln2(x))\n        return x",
  takeaway: [For each sublayer: normalize its input, compute the output, and add the input back.],
  color: GREEN,
)

#pause

#note[`self.causal_attn` wraps the attention calculation. We will add the full module, dropout, and initialization in the notebook.]

== Stack several blocks

#align(center, grid(
  columns: (34mm,),
  rows: (11mm, 8mm, 11mm, 8mm, 11mm, 8mm, 11mm),
  align: center + horizon,
  token([$X^((0))$], color: TEAL, fill: PALE-TEAL, w: 31mm),
  text(size: 20pt, fill: MUTED)[$arrow.b$],
  token([block 1], color: BLUE, fill: PALE-BLUE, w: 31mm),
  text(size: 20pt, fill: MUTED)[$arrow.b$],
  token([$dots.v$], color: MUTED, fill: CREAM, w: 31mm),
  text(size: 20pt, fill: MUTED)[$arrow.b$],
  token([block L], color: ACC, fill: PALE-ACC, w: 31mm),
))

#pause

#note[Each block has its own attention, FFN, and LayerNorm parameters. The blocks share the same structure.]

== The next block uses the updated token states

#simple-flow((
  ([gather entity features], BLUE, PALE-BLUE),
  ([transform its features], ACC, PALE-ACC),
  ([attend using the new state], TEAL, PALE-TEAL),
  ([update the features again], GREEN, PALE-GREEN),
))

#pause

#two(
  card([first block], [Might combine information from a name or a nearby verb in a token's state.], color: BLUE, fill: PALE-BLUE),
  card([later block], [Can use the updated state to compute a new set of attention weights.], color: GREEN, fill: PALE-GREEN),
)

#note([Several blocks allow several steps of computation. We cannot assume that each layer corresponds to one human reasoning step.], color: PURPLE)

== Information can pass through an intermediate token

#token-row((
  token([The]), token([trophy]), token([did]), token([not]), token([fit]),
  token([in]), token([the]), token([suitcase], w: 21mm, size: 10pt), token([because], w: 20mm, size: 10pt), token([it]),
  token([was]), query-token([too]),
), arrow-to: [large])

#v(9pt)

#simple-flow((
  ([block 1 at `it`], BLUE, PALE-BLUE),
  ([gather from `trophy`], TEAL, PALE-TEAL),
  ([updated `it` state], GREEN, PALE-GREEN),
  ([later block at `too` gathers `it`], BLUE, PALE-BLUE),
  ([LM head favours `large`], ACC, PALE-ACC),
))

#pause

#note([The later block uses the updated state at `it`. This picture illustrates one possible computation.], color: PURPLE)

// Act 12 · Complete decoder-only language model · 130–142

== Begin with token and position information

#align(center, diagram(
  spacing: (21mm, 13mm),
  {
    flow-node((0, 0), [token ID $x_i$], color: INK, fill: CREAM, w: 25mm)
    flow-node((1.55, -0.7), [embedding #linebreak() $E[x_i]$], color: TEAL, fill: PALE-TEAL, w: 27mm)
    flow-node((1.55, 0.7), [position #linebreak() $p_i$], color: PURPLE, fill: PALE-PURPLE, w: 27mm)
    flow-node((3.05, 0), [$+$], color: GREEN, fill: PALE-GREEN, w: 14mm, size: 18pt)
    flow-node((4.55, 0), [$X_i^((0))$], color: GREEN, fill: PALE-GREEN, w: 23mm, size: 16pt)
    flow-arrow((0, 0), (1.55, -0.7), color: TEAL)
    flow-arrow((1.55, -0.7), (3.05, 0), color: TEAL)
    flow-arrow((1.55, 0.7), (3.05, 0), color: PURPLE)
    flow-arrow((3.05, 0), (4.55, 0), color: GREEN)
  },
))

#v(10pt)

#align(center, $X_i^((0)) = E[x_i] + p_i$)

#pause

#note[Without $p_i$, identical tokens begin with identical rows. Position information tells the model where each copy occurs. Some modern models add position information inside attention instead.]

== Turn the final token state into token scores

#two(
  [#lm-head-diagram()],
  [
    #align(center, $z_t = h_t W_"vocab" + b$)
    #v(9pt)
    #align(center, $p_t = "softmax"(z_t)$)
    #v(12pt)
    #align(center, grid(
      columns: (auto,), row-gutter: 6pt,
      dim-pill([$h_t: d_"model"$], color: BLUE),
      dim-pill([$z_t: V$], color: BLUE),
    ))
    #v(9pt)
    #align(center, text(size: 12.5pt, fill: MUTED)[softmax runs over the $V$ vocabulary scores])
  ],
  ratio: (1.2fr, .8fr),
)

== The decoder-only model from end to end

#token-row((token([deep], w: 20mm), token([learning], w: 28mm), token([is])))

#v(7pt)

#simple-flow((
  ([token IDs], INK, CREAM),
  ([embed $+$ position], PURPLE, PALE-PURPLE),
  ([$L times$ decoder blocks], BLUE, PALE-BLUE),
  ([final LN], PURPLE, PALE-PURPLE),
  ([dense $arrow.r$ $V$ logits], ACC, PALE-ACC),
  ([next-token distributions], GREEN, PALE-GREEN),
))

#pause

#align(center, text(size: 14pt, fill: MUTED)[The output is still a probability distribution over the next token.])

== Slide the targets one place to the left

#align(center, grid(
  columns: (22mm, 30mm, 30mm, 30mm, 30mm),
  rows: (10mm, 12mm, 12mm), gutter: 3pt,
  block(),
  align(center, text(size: 10pt, weight: 700, fill: MUTED)[POSITION 1]),
  align(center, text(size: 10pt, weight: 700, fill: MUTED)[POSITION 2]),
  align(center, text(size: 10pt, weight: 700, fill: MUTED)[POSITION 3]),
  align(center, text(size: 10pt, weight: 700, fill: MUTED)[POSITION 4]),
  align(right + horizon, text(size: 11pt, weight: 700, fill: TEAL)[input]),
  token([`<BOS>`], w: 28mm), token([deep], w: 28mm), token([learning], w: 28mm), token([is], w: 28mm),
  align(right + horizon, text(size: 11pt, weight: 700, fill: ACC)[target]),
  target-token([deep], w: 28mm), target-token([learning], w: 28mm), target-token([is], w: 28mm), target-token([fun], w: 28mm),
))

#pause

#claim[Each row uses its prefix to predict the next token]

== During training, we know the whole sentence

#two(
  [#card([during one training pass], [
    Feed in the true sentence. Row $t$ can use the true prefix $x_(<=t)$.
  ], color: TEAL, fill: PALE-TEAL)],
  [#card([all rows together], [
    Score the next token at every row in the same pass. Sampling and feeding a token back happens during generation.
  ], color: RED, fill: PALE-RED)],
)

#pause

#note[The causal mask hides future tokens from each row, so a row cannot see the answer it must predict.]

== Score every next-token guess

#claim[$cal(L) = - sum_(t=1)^(T-1) m_t log p_(theta)(x_(t+1) | x_(<=t))$]

#v(10pt)

#three(
  card([look only left], [The causal mask keeps future words hidden.], color: RED, fill: PALE-RED),
  card([score real rows], [$m_t$ skips padding or ignored positions.], color: MUTED, fill: CREAM),
  card([score the target], [Cross-entropy uses the probability of the token one step to the right.], color: ACC, fill: PALE-ACC),
)

#pause

#note[Code often averages over valid tokens. The sum above shows one loss term for each next-token prediction.]

== The next-token objective stays the same

#two(
  card([fixed-window character model], [
    context: `a a b`
    #v(5pt)
    target: `i`
    #v(5pt)
    loss: $-log p_(theta)(i | a,a,b)$
  ], color: TEAL, fill: PALE-TEAL),
  card([decoder Transformer], [
    context: $x_(<=t)$
    #v(5pt)
    target: $x_(t+1)$
    #v(5pt)
    loss: $-log p_(theta)(x_(t+1) | x_(<=t))$
  ], color: BLUE, fill: PALE-BLUE),
)

#pause

#claim[Both models minimize the loss on the next token]

== Shift the targets and compute cross-entropy

#torch-code(
  "inputs  = tokens[:, :-1]\ntargets = tokens[:, 1:]\nlogits  = model(inputs)\nloss = F.cross_entropy(\n    logits.flatten(0, 1), targets.flatten())",
  takeaway: [The two slices pair each input position with its next-token target.],
  color: RED,
)

#pause

#note[For padded batches, pass the padding token ID as `ignore_index`. We will add the data loader and masks in the notebook.]

== Training: one pass, many next-token guesses

#clock-strip("train")

#v(12pt)

#simple-flow((
  ([true token sequences], TEAL, PALE-TEAL),
  ([one causal forward pass], BLUE, PALE-BLUE),
  ([CE at valid rows], ACC, PALE-ACC),
  ([backpropagate], RED, PALE-RED),
  ([update parameters], GREEN, PALE-GREEN),
))

#pause

#note[Positions are processed in parallel during the training forward pass. Layers are still evaluated one after another.]

== Generation: one new token, then go again

#clock-strip("generate")

#v(12pt)

#simple-flow((
  ([current prefix], TEAL, PALE-TEAL),
  ([final-row logits], BLUE, PALE-BLUE),
  ([choose / sample], ACC, PALE-ACC),
  ([append token], GREEN, PALE-GREEN),
  ([repeat], INK, CREAM),
))

#pause

#note[Autoregressive generation is sequential because the next model input depends on the token just chosen.]

== Generation, first step

#token-row((token([`<BOS>`], w: 26mm), token([The], w: 22mm)), arrow-to: [?])

#v(10pt)

#prob-bars((([student], .52, GREEN), ([teacher], .23, BLUE), ([door], .08, MUTED), ([other], .17, ACC)), hgt: 25mm)

#pause

#align(center, text(size: 20pt, weight: 650)[choose `student` $arrow.r$ append it to the prefix])

#note[Use the distribution at the final input position to choose the next token.]

== Generation, one step later

#token-row((token([`<BOS>`], w: 25mm), token([The], w: 20mm), token([student], w: 29mm)), arrow-to: [?])

#v(10pt)

#prob-bars((([opened], .47, GREEN), ([studied], .25, BLUE), ([is], .16, ACC), ([other], .12, MUTED)), hgt: 25mm)

#pause

#align(center, text(size: 20pt, weight: 650)[choose `opened` $arrow.r$ continue with a longer prefix])

#note[The prefix now has one more token. We can cache calculations from the earlier positions and reuse them.]

== The whole generation loop

#torch-code(
  "for _ in range(max_new_tokens):\n    logits = model(ids)[:, -1]\n    probs = logits.softmax(dim=-1)\n    next_id = torch.multinomial(probs, 1)\n    ids = torch.cat([ids, next_id], dim=1)",
  takeaway: [Score the prefix, sample one token, append it, and repeat.],
  color: BLUE,
)

#pause

#note[Greedy decoding would use `argmax` instead of sampling. A real implementation also handles stopping tokens and the KV cache.]

== Cache the keys and values from earlier positions

#two(
  [
    #hairline([already cached at each layer], [
      #align(center, grid(
        columns: (auto, auto), gutter: 5pt,
        token([$K_1$], color: TEAL, fill: PALE-TEAL, w: 20mm), target-token([$V_1$], w: 20mm),
        token([$K_2$], color: TEAL, fill: PALE-TEAL, w: 20mm), target-token([$V_2$], w: 20mm),
        token([$K_3$], color: TEAL, fill: PALE-TEAL, w: 20mm), target-token([$V_3$], w: 20mm),
      ))
    ], color: TEAL)
  ],
  [
    #hairline([new token], [
      #align(center, grid(
        columns: (auto, auto, auto), gutter: 5pt,
        query-token([$q_4$], w: 20mm), token([$k_4$], color: TEAL, fill: PALE-TEAL, w: 20mm), target-token([$v_4$], w: 20mm),
      ))
      #v(8pt)
      #text(size: 11pt)[Score $q_4$ against cached $K_(1:3)$ and new $k_4$; append $k_4,v_4$ to the cache.]
    ], color: BLUE)
  ],
)

#pause

#note[In causal incremental decoding, adding a future token leaves earlier hidden rows unchanged. Their projected keys and values stay the same too.]

== What does a longer context cost?

#three(
  hairline([training score grid], [
    #matrix-table(
      ([q₁], [q₂], [q₃], [q₄]), ([k₁], [k₂], [k₃], [k₄]),
      (([•], [×], [×], [×]), ([•], [•], [×], [×]), ([•], [•], [•], [×]), ([•], [•], [•], [•])),
      cell-fill: (i, j, v) => if j <= i { PALE-BLUE } else { PALE-RED },
      cell-width: 8mm, cell-height: 7mm, text-size: 8pt, label-size: 8pt,
    )
    #v(5pt)
    #align(center, text(size: 11pt)[$Theta(T^2)$ pair scores])
  ], color: BLUE),
  hairline([decoding cache], [
    #align(center, grid(
      columns: (auto, auto), gutter: 3pt,
      token([$K_1$], w: 16mm, hgt: 8mm, size: 10pt), target-token([$V_1$], w: 16mm),
      token([$K_2$], w: 16mm, hgt: 8mm, size: 10pt), target-token([$V_2$], w: 16mm),
      token([$dots.v$], w: 16mm, hgt: 8mm, size: 10pt), target-token([$dots.v$], w: 16mm),
      token([$K_T$], w: 16mm, hgt: 8mm, size: 10pt), target-token([$V_T$], w: 16mm),
    ))
    #v(5pt)
    #align(center, text(size: 11pt)[stored rows grow with $T$])
  ], color: TEAL),
  hairline([one new query], [
    #align(center, diagram(spacing: (13mm, 8mm), {
      flow-node((0, 0), [$q_(T+1)$], color: BLUE, fill: PALE-BLUE, w: 19mm)
      for (i, y) in ((0, -1.2), (1, -.4), (2, .4), (3, 1.2)) {
        flow-node((2.0, y), if i == 3 { [$K_T$] } else if i == 2 { [$dots.v$] } else { [$K_#(i+1)$] }, color: TEAL, fill: PALE-TEAL, w: 16mm)
        flow-arrow((0, 0), (2.0, y), color: ACC)
      }
    }))
    #v(5pt)
    #align(center, text(size: 11pt)[compare with all cached keys])
  ], color: ACC),
)

#pause

#note[Optimized kernels can avoid materializing the full $T times T$ matrix, but they do not remove the vanilla all-pairs arithmetic.]

// Act 13 · Model families and return to aabid · 143–154

== Three common ways to arrange Transformer blocks

#three(
  family-card(
    [encoder-only], [all tokens can see both directions], [masked-token / representation learning], [BERT], TEAL, PALE-TEAL,
    visual: tiny-visibility(color: TEAL),
  ),
  family-card(
    [decoder-only], [each token sees only its prefix], [next-token prediction], [GPT-style LM], BLUE, PALE-BLUE,
    visual: tiny-visibility(causal: true, color: BLUE),
  ),
  family-card(
    [encoder-decoder], [source bidirectional; target causal], [conditioned sequence generation], [translation], ACC, PALE-ACC,
    visual: mini-pipeline((
      ([source], TEAL, PALE-TEAL), ([memory], GREEN, PALE-GREEN), ([target], ACC, PALE-ACC),
    )),
  ),
)

== Encoder-only: read the whole input

#weights-row(
  [`bank`],
  ([the], [river], [bank], [was], [wide]),
  ([.12], [.39], [.18], [.10], [.21]),
  emphasis: (1, 4), color: TEAL,
)

#v(10pt)

#two(
  card([visibility], [Every position may attend left and right; there is no causal triangle.], color: TEAL, fill: PALE-TEAL),
  card([typical use], [Build contextual features for classification, tagging, retrieval, or masked-token training.], color: GREEN, fill: PALE-GREEN),
)

#pause

#note[BERT uses an encoder. The next-token model we built uses a causal decoder.]

== Decoder-only: read only the prefix

#weights-row(
  [`sat`],
  ([`<BOS>`], [The], [cat], [sat], [on]),
  ([.08], [.14], [.43], [.35], [$0$]),
  emphasis: (2, 3), forbidden: (4,), color: BLUE,
)

#v(10pt)

#two(
  card([visibility], [Position $t$ attends only to positions $<=t$.], color: RED, fill: PALE-RED),
  card([objective], [Predict $x_(t+1)$ from $x_(<=t)$ at every valid row.], color: BLUE, fill: PALE-BLUE),
)

#pause

#claim[This is the decoder we have been building.]

== Encoder-decoder: read one sequence, write another

#simple-flow((
  ([source sentence], TEAL, PALE-TEAL),
  ([bidirectional encoder], TEAL, PALE-TEAL),
  ([source memory], GREEN, PALE-GREEN),
  ([causal target decoder], BLUE, PALE-BLUE),
  ([target sentence], ACC, PALE-ACC),
))

#pause

#note[
  The decoder uses causal self-attention over the generated target *and*
  cross-attention to the encoded source. Translation is the classic example.
]

== Self-attention and cross-attention use the same calculation

#two(
  card([self-attention], [
    $Q = X W_Q$
    #v(3pt)
    $K = X W_K quad V = X W_V$
    #v(6pt)
    Queries, keys, and values come from the same sequence.
  ], color: BLUE, fill: PALE-BLUE),
  card([cross-attention], [
    $Q = Y_"decoder" W_Q$
    #v(3pt)
    $K = X_"source" W_K quad V = X_"source" W_V$
    #v(6pt)
    Queries come from the decoder. Keys and values come from the encoded source.
  ], color: ACC, fill: PALE-ACC),
)

#pause

#note[The attention calculation stays the same. What changes is the source of the queries, keys, and values.]

== The 2017 Transformer had an encoder and a decoder

#two(
  [
    #align(center, diagram(
      spacing: (24mm, 10mm),
      {
        flow-node((0, 0), [source tokens], color: TEAL, fill: PALE-TEAL, w: 28mm)
        flow-node((0, -1.25), [$N times$ encoder], color: TEAL, fill: PALE-TEAL, w: 30mm)
        flow-node((0, -2.5), [source memory], color: GREEN, fill: PALE-GREEN, w: 30mm)
        flow-node((2, 0), [shifted target], color: BLUE, fill: PALE-BLUE, w: 28mm)
        flow-node((2, -1.25), [$N times$ decoder], color: BLUE, fill: PALE-BLUE, w: 30mm)
        flow-node((2, -2.5), [next target token], color: ACC, fill: PALE-ACC, w: 31mm)
        flow-arrow((0, 0), (0, -1.25), color: TEAL)
        flow-arrow((0, -1.25), (0, -2.5), color: GREEN)
        flow-arrow((2, 0), (2, -1.25), color: BLUE)
        flow-arrow((2, -1.25), (2, -2.5), color: ACC)
        flow-arrow((0, -2.5), (2, -1.25), color: GREEN, label: [cross-attention])
      },
    ))
  ],
  [
    #card([2017], [
      Vaswani et al., *Attention Is All You Need*, introduced the Transformer as an encoder-decoder model for sequence transduction.
      #v(8pt)
      We focused on a decoder-only model because it directly extends next-token prediction.
    ], color: INK, fill: CREAM)
  ],
)

#source-line([Vaswani et al. (2017) · #link("https://arxiv.org/abs/1706.03762")[arXiv:1706.03762]])

== Put the sequence models side by side

#block[
#set text(size: 12pt)
#table(
  columns: (0.8fr, 1.3fr, 1.15fr, 1.05fr, 1.3fr),
  rows: auto,
  inset: 5pt,
  stroke: 0.45pt + rgb("#D5DBD9"),
  fill: (x, y) => if y == 0 { INK } else if calc.rem(y, 2) == 0 { rgb("#F6F5F1") } else { white },
  table.header(
    text(fill: white, weight: 700)[model],
    text(fill: white, weight: 700)[available context],
    text(fill: white, weight: 700)[path to far token],
    text(fill: white, weight: 700)[training across time],
    text(fill: white, weight: 700)[main trade-off],
  ),
  [fixed-window MLP], [exactly $k$ previous tokens], [$1$ within window; unavailable outside], [parallel examples], [input width and parameters tied to $k$],
  [RNN], [prefix compressed into recurrent state], [$O(T)$ recurrent steps], [sequential recurrence], [compact state; long dependency path],
  [temporal convolution], [finite receptive field grows with depth/dilation], [depends on kernel and depth], [parallel], [local bias; depth needed for reach],
  [self-attention], [all visible tokens in the mask], [$1$ attention edge per layer], [parallel masked rows], [vanilla all-pairs score cost],
)

#v(7pt)

#note[The path length counts the steps information must take through the model. A short path does not guarantee that the model learns the dependency.]
]

== Longer memory has a price

#three(
  card([fixed MLP], [Uses a fixed window $k$. A larger context needs a different first layer.], color: TEAL, fill: PALE-TEAL),
  card([RNN], [Roughly linear steps in $T$, but those recurrent steps are sequential.], color: ACC, fill: PALE-ACC),
  card([self-attention], [Vanilla pair scores are $Theta(T^2)$; a causal triangle is still $Theta(T^2)$.], color: BLUE, fill: PALE-BLUE),
)

#pause

#claim[Compare both what context a model can use and what that access costs.]

#note([These complexity terms omit width, layers, constants, kernels, and hardware. Each architecture has trade-offs.], color: PURPLE)

== What does attention give us?

#two(
  [
    #hairline([gives], [
      #card([content-dependent routing], [A query can weight visible sources differently for each token and layer.], color: GREEN, fill: PALE-GREEN)
      #v(6pt)
      #card([short communication paths], [Any visible token can directly contribute within an attention sublayer.], color: GREEN, fill: PALE-GREEN)
    ], color: GREEN)
  ],
  [
    #hairline([does not guarantee], [
      #card([truth or reasoning], [The model may still generate false or poorly justified text.], color: RED, fill: PALE-RED)
      #v(6pt)
      #card([explanation or unlimited memory], [Attention weights are not explanations. Context length and the KV cache remain finite.], color: RED, fill: PALE-RED)
    ], color: RED)
  ],
)

== Four quick checks

#block[
#set text(size: 13pt)
#table(
  columns: (1fr, 1.15fr),
  inset: 7pt,
  stroke: 0.5pt + rgb("#D5DBD9"),
  table.header(
    block(fill: INK, inset: 6pt, width: 100%, text(fill: white, weight: 700)[CLAIM]),
    block(fill: INK, inset: 6pt, width: 100%, text(fill: white, weight: 700)[CORRECTION]),
  ),
  ["Attention weights are explanations."], [We can inspect the weights, but an explanation needs stronger causal evidence.],
  [#pause "One head learns one linguistic rule."], [#pause A pattern may be spread across heads and may change across inputs.],
  [#pause "Masking is only used at generation."], [#pause We also need the causal mask during parallel next-token training.],
  [#pause "A Transformer removes the MLP."], [#pause Every standard block contains a substantial position-wise FFN.],
)
]

== Compare our two language models

#two(
  [
    #hairline([fixed-window MLP], [
      #mini-pipeline((
        ([IDs], INK, CREAM),
        ([embeds], TEAL, PALE-TEAL),
        ([concat 3], BLUE, PALE-BLUE),
        ([MLP → 27], ACC, PALE-ACC),
      ))
      #v(8pt)
      #token-row((token([a]), token([a]), token([b])), arrow-to: [i])
    ], color: TEAL)
  ],
  [
    #hairline([decoder Transformer], [
      #mini-pipeline((
        ([IDs], INK, CREAM),
        ([embed + pos.], TEAL, PALE-TEAL),
        ([causal blocks], BLUE, PALE-BLUE),
        ([head → $V$], ACC, PALE-ACC),
      ))
      #v(8pt)
      #token-row((token([`-`]), token([a]), token([a]), token([b])), arrow-to: [i])
    ], color: BLUE)
  ],
  ratio: (1fr, 1fr), gutter: 11pt,
)

#pause

#claim[Both use IDs $arrow.r$ embeddings $arrow.r$ logits $arrow.r$ cross-entropy $arrow.r$ sampling]

#note[The MLP concatenates three embedding rows. The Transformer keeps one row per token and combines information between rows through causal attention.]

== Can you finish this sentence?

#align(center, block(
  width: 92%, inset: (x: 18pt, y: 16pt), fill: CREAM,
  stroke: 1.2pt + INK, radius: 5pt,
  text(size: 24pt, weight: 600)[
    "A Transformer language model differs from our character MLP mainly because ..."
  ],
))

#pause

#v(12pt)

#claim[
  ...each position can weight earlier tokens differently, combine their values,
  pass the result through an MLP, and repeat this in the next block.
]

#pause

#align(center, text(size: 16pt, weight: 650, fill: TEAL)[We still train it to predict the next token from the previous tokens.])
