// Seven frames: causal training, positions, the matrix view, and the full update.
// The numerical example uses the same six input words and eight-token vocabulary.
#import "../../common/metropolis.typ": *
#import "../../common/mldiag.typ": *
#import "helpers.typ": *

#let bp-words = ([She], [deposited], [money], [in], [the], [bank])
#let bp-targets = ([deposited], [money], [in], [the], [bank], [.])
#let bp-numerical-run = json("bank-training-numbers.json").full_causal_training.before
#let bp-decimal(value) = {
  let rounded = str(calc.round(value * 1000) / 1000)
  let parts = rounded.split(".")
  if parts.len() == 1 { rounded + ".000" }
  else { rounded + "0" * (3 - parts.at(1).len()) }
}

#let bp-cell(body, color: TEAL, fill: PALE-TEAL, width: 100%, height: 11mm, size: 15pt) = box(
  width: width, height: height, radius: 2pt,
  stroke: .8pt + color, fill: fill,
  align(center + horizon, text(size: size, body)),
)

#let bp-mask = {
  show math.equation: set text(size: 10.5pt)
  matrix-table(
  ([1], [2], [3], [4], [5], [6]),
  ([1], [2], [3], [4], [5], [6]),
  range(6).map(i => range(6).map(j => if j <= i { [0] } else { [$-oo$] })),
  cell-fill: (i, j, value) => if j <= i { PALE-BLUE } else { PALE-RED },
  cell-width: 12mm,
  cell-height: 6mm,
  text-size: 10.5pt,
  label-size: 10pt,
  )
}

#[
== Predict every next token without reading ahead

#set text(size: 17pt)
#align(center, text(size: 18pt)[Complete sentence: `She deposited money in the bank .`])
#v(5pt)
#grid(columns: (22mm,) + (1fr,) * 6, column-gutter: 5pt, row-gutter: 6pt,
  align: center + horizon,
  text(size: 12pt, weight: 650, fill: TEAL)[input],
  ..bp-words.map(word => bp-cell(word)),
  text(size: 12pt, weight: 650, fill: ACC)[target],
  ..bp-targets.map(word => bp-cell(word, color: ACC, fill: PALE-ACC)),
)
#v(4pt)
#align(center, text(size: 14pt)[Input position $i$ predicts the target $y_i=t_(i+1)$.])

#v(7pt)
#uncover("2-")[
  #align(center, text(size: 17pt)[At `money` (position 3), the target is `in` (position 4).])
  #v(4pt)
  #grid(columns: (22mm,) + (1fr,) * 6, gutter: 5pt, align: center + horizon,
    text(size: 12pt, weight: 650, fill: BLUE)[query 3],
    ..bp-words.enumerate().map(((i, word)) => bp-cell(
      word,
      color: if i <= 2 { BLUE } else { RED },
      fill: if i <= 2 { PALE-BLUE } else { PALE-RED },
    )),
  )
  #v(3pt)
  #align(center, text(size: 13pt)[Read `She deposited money`. Hide `in the bank` from this query.])
]

#v(6pt)
#uncover("3-")[
  #align(center)[$M_(3,:) = mat(0,0,0,-oo,-oo,-oo)$]
  #v(4pt)
  #text(size: 14pt)[#note(color: RED)[Add the mask before softmax. Positions 4, 5, and 6 then get zero attention weight in row 3.]]
]

== Give each position its own vector

#set text(size: 17pt)
#align(center, text(size: 20pt)[$e_i^((0))=e_i+r_i quad "and" quad e_i^((1))=e_i^((0))+Delta e_i$])
#v(3pt)
#align(center, text(size: 13pt)[Superscripts count stages: (0) before attention, (1) after. The subscript $i$ is still the position, not the token ID.])
#v(7pt)
#grid(columns: (1fr, 9mm, 1fr, 9mm, 1fr), gutter: 8pt, align: center + horizon,
  hairline([lookup embedding], [$e_6=E[t_6]$\ $mat(1,1,0)$], color: TEAL),
  text(size: 23pt)[$+$],
  hairline([position 6], [$r_6$\ $mat(0,0.1,0)$], color: PURPLE),
  text(size: 23pt)[$arrow.r$],
  uncover("2-")[#hairline([stage 0 at position 6], [$e_6^((0))$\ $mat(1,1.1,0)$], color: BLUE)],
)
#v(7pt)
#uncover("2-")[
  #align(center, text(size: 17pt)[$e_6^((0)) W_Q arrow.r q_6 quad e_6^((0)) W_K arrow.r k_6 quad e_6^((0)) W_V arrow.r v_6$])
  #v(4pt)
  #text(size: 14pt)[#note(color: PURPLE)[Row $i$ of $X$ is now $e_i^((0))$. The projections decide how position affects Q, K, and V.]]
]
#v(7pt)
#uncover("3-")[
  #text(size: 13pt)[#note(color: BLUE)[$R$: max-length × $d$ position table; $r_i=R[i-1]$ in code. Restart original parameters with $R=0$; the nonzero $r_6$ above only illustrates addition.]]
  #v(4pt)
  #align(center, text(size: 13pt)[Attention can use word-position pairs. Taking their plain mean still loses order.])
]

== Compute all six query rows together

#set text(size: 16pt)
#set par(spacing: 8pt)
#grid(columns: (1fr, 1fr, 1fr), gutter: 14pt, align: center + horizon,
  hairline([queries], [$Q=X W_Q in RR^(6 times 3)$], color: BLUE),
  hairline([keys], [$K=X W_K in RR^(6 times 3)$], color: TEAL),
  hairline([values], [$V=X W_V in RR^(6 times 3)$], color: ACC),
)
#v(4pt)
#align(center, text(size: 12.5pt)[The same projections act on every row $e_i^((0))$ of $X$. $X'$ will stack the output rows $e_i^((1))$.])
#v(5pt)
#uncover("2-")[
  #grid(columns: (1fr, 1fr), gutter: 15pt, align: center + horizon,
    [
      #align(center)[$S = Q K^top / sqrt(3) in RR^(6 times 6)$]
      #v(11pt)
      #align(center)[$A = "softmax"(S + M) in RR^(6 times 6)$]
      #v(5pt)
      #align(center, text(size: 12pt)[$A_(i j)=alpha_(i j)$: receiver $i$, source $j$.])
      #align(center, text(size: 12pt)[Softmax runs across each row. Earlier $alpha_j=alpha_(6 j)$.])
    ],
    [
      #align(center, text(size: 12pt, weight: 650, fill: RED)[CAUSAL MASK $M$])
      #v(1pt)
      #bp-mask
    ],
  )
]
#v(5pt)
#uncover("3-")[
  #align(center)[$H = A V quad Delta X=H W_O quad X'=X+Delta X$]
  #align(center)[$Z = X' U + b in RR^(6 times 8)$]
  #v(3pt)
  #align(center, text(size: 12pt)[$Delta X$ stacks $Delta e_i$; $X'$ stacks $e_i^((1))=e_i^((0))+Delta e_i$. All three matrices $H,Delta X,X'$ are $6 times 3$.])
  #v(5pt)
  #align(center, text(size: 13pt)[Row 3 predicts `in` using positions 1, 2, and 3. All six rows run in the same forward pass.])
]

== The matrix pass contains our earlier calculation

#set text(size: 15pt)
#set par(spacing: 8pt)
#align(center, text(size: 12.5pt)[1 = She #h(10pt) 2 = deposited #h(10pt) 3 = money #h(10pt) 4 = in #h(10pt) 5 = the #h(10pt) 6 = bank])
#v(5pt)
#grid(columns: (1.45fr, 12mm, 1fr), column-gutter: 6pt, align: center + horizon,
  [
    #align(center, text(size: 14pt, fill: BLUE)[$A$: weights over source positions])
    #v(5pt)
    #matrix-table(
      ([1], [2], [3], [4], [5], [6]),
      ([1], [2], [3], [4], [5], [6]),
      bp-numerical-run.alpha.enumerate().map(((i, row)) => row.enumerate().map(((j, value)) =>
        text(fill: if j > i { RED } else { BLUE }, bp-decimal(value))
      )),
      cell-fill: (i, j, value) => if j > i { PALE-RED } else if i == 5 { PALE-GREEN } else { PALE-BLUE },
      cell-width: 15mm,
      cell-height: 7mm,
      text-size: 13pt,
      label-size: 11pt,
    )
    #v(5pt)
    #align(center, text(size: 12pt)[rows: query positions #h(9pt) columns: source positions])
  ],
  uncover("2-")[#text(size: 23pt, fill: GREEN)[$arrow.r$]],
  uncover("2-")[
    #align(center, text(size: 14pt, fill: GREEN)[$H = A V$: one message per row])
    #v(5pt)
    #matrix-table(
      ([1], [2], [3], [4], [5], [6]),
      ([coord. 1], [coord. 2], [coord. 3]),
      bp-numerical-run.H.map(row => row.map(value => text(fill: GREEN, bp-decimal(value)))),
      cell-fill: (i, j, value) => if i == 5 { PALE-GREEN } else { white },
      cell-width: 17mm,
      cell-height: 7mm,
      text-size: 13pt,
      label-size: 11pt,
    )
    #v(5pt)
    #align(center, text(size: 12pt)[Three coordinates per query position.])
  ],
)
#v(5pt)
#align(center, text(size: 11.5pt, fill: MUTED)[Original parameters, $R=0$. Round only for display: compute $H$ with full-precision weights.])
#v(5pt)
#uncover("3-")[
  #note(color: BLUE)[Row 3 cannot read positions 4, 5, or 6: those weights are zero.]
  #v(3pt)
  #note(color: GREEN)[With $R=0$, $e_6^((0))=e_6$. The same $h_6 approx mat(1.018,0.571,0.286)$ gives $e_6^((1))=e_6^((0))+Delta e_6 approx mat(2.018,1.571,0.286)$.]
]

== The complete forward pass in PyTorch

#set text(size: 14pt)
#align(center, text(size: 13pt)[Original parameters and $R=0$. Rows of X are $e_i^((0))$; rows of X_new are $e_i^((1))$.])
#v(5pt)
#torch-code(
  "def forward(ids):\n    X = E[ids] + R[torch.arange(len(ids))]\n    Q, K, V = X @ W_Q, X @ W_K, X @ W_V",
)
#v(6pt)
#uncover("2-")[
  #torch-code(
    "    scores = Q @ K.T / Q.shape[-1] ** 0.5\n    future = torch.ones_like(scores, dtype=torch.bool).triu(1)\n    A = scores.masked_fill(future, -torch.inf).softmax(-1)",
    color: RED,
  )
]
#v(6pt)
#uncover("3-")[
  #torch-code(
    "    H = A @ V\n    X_new = X + H @ W_O\n    return X_new @ U + b\nall_logits = forward(ids)  # [6, 8]",
    color: GREEN,
  )
]

== Average the six losses, then update the parameters

#set text(size: 14pt)
#align(center, text(size: 13.5pt)[Restart from the original toy parameters, with $R=0$. Targets: `deposited money in the bank .`])
#v(9pt)
#align(center)[$L = -1/6 sum_(i=1)^6 log "softmax"(Z_(i,:))_(y_i)$]
#v(10pt)
#align(center, text(size: 13pt)[Before the update: mean loss $approx 2.339914$.])
#v(5pt)
#uncover("2-")[
  #set text(size: 12.5pt)
  #torch-code(
    "target_ids = torch.tensor([1, 2, 3, 4, 5, 6])\nparams = [E, R, W_Q, W_K, W_V, W_O, U, b]\noptimizer = torch.optim.SGD(params, lr=0.1)",
    color: TEAL,
  )
]
#v(5pt)
#uncover("3-")[
  #set text(size: 12.5pt)
  #torch-code(
    "optimizer.zero_grad()\nloss = F.cross_entropy(all_logits, target_ids)\nloss.backward()\noptimizer.step()",
    color: RED,
  )
  #v(5pt)
  #align(center, text(size: 16pt, fill: GREEN)[Mean loss: $2.339914 arrow.r 2.199261$])
  #v(3pt)
  #align(center, text(size: 12.5pt)[The last row's loss rises: 0.891023 to 0.947535. The average over all six rows falls.])
]

== One complete next-token model

#set text(size: 17pt)
#align(center, diagram(
  spacing: (11mm, 10mm),
  {
    flow-node((0,0), [token + position\ vectors], color: TEAL, fill: PALE-TEAL, w: 35mm, size: 13pt)
    flow-node((2,0), [Q, K, V], color: BLUE, fill: PALE-BLUE, w: 28mm, size: 14pt)
    flow-node((4,0), [scaled, masked\ attention], color: GREEN, fill: PALE-GREEN, w: 35mm, size: 13pt)
    flow-node((6,0), [map message\ + retain $X$], color: GREEN, fill: PALE-GREEN, w: 30mm, size: 13pt)
    flow-node((8,0), [vocabulary\ scores], color: ACC, fill: PALE-ACC, w: 30mm, size: 13pt)
    flow-arrow((0,0), (2,0), color: TEAL)
    flow-arrow((2,0), (4,0), color: BLUE)
    flow-arrow((4,0), (6,0), color: GREEN)
    flow-arrow((6,0), (8,0), color: GREEN)
  },
))
#v(16pt)
#uncover("2-")[
  #grid(columns: (1fr, 1fr), gutter: 20pt, align: top,
    hairline([training], [Known sentence\ $arrow.r$ six next-token losses\ $arrow.r$ one parameter update], color: RED),
    hairline([generation], [Current prefix\ $arrow.r$ next-token probabilities\ $arrow.r$ choose a token and append it], color: BLUE),
  )
  #v(9pt)
  #align(center, text(size: 14pt)[Generation changes the prefix. Training changes the parameter values.])
]
#v(15pt)
#uncover("3-")[
  #note(color: ACC)[Next: combine several attention heads, add normalization and a feed-forward layer, and stack blocks. We will reuse this retained-input idea.]
]
]
