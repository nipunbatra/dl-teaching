// One continuous, trainable bank example. Numbers are checked by bank_training.py.
#import "../../common/metropolis.typ": *
#import "../../common/mldiag.typ": *
#import "helpers.typ": *

#let bt-small(body) = align(center, text(size: 12pt, fill: MUTED, body))
#let bt-prefix = align(center, text(size: 19pt)[`She deposited money in the` #text(fill: BLUE)[`bank`] #text(fill: ACC)[`___`]])
#let bt-table(columns, ..cells) = grid(columns: columns, column-gutter: 14pt, row-gutter: 8pt,
  align: (x, y) => if x == 0 { left + horizon } else { center + horizon }, ..cells.pos())
#let bt-path(items) = align(center, grid(columns: (1fr,) * items.len(), gutter: 10pt,
  ..items.enumerate().map(((i, item)) => [
    #text(size: 15pt, weight: 650, fill: if i == items.len() - 1 { ACC } else { BLUE })[#item]
  ]),
))

// The same layout first shows dependencies, then the reverse-mode traversal.
#let bt-computation-graph(backward: false) = {
  show math.equation: set text(size: 12pt)
  set par(spacing: 0pt, leading: 2pt)
  align(center, diagram(
  {
    // Physical coordinates keep the message path and retained-input path apart.
    let xy(pos) = (pos.at(0) * 1mm, -pos.at(1) * 1mm)
    let graph-node(pos, body, color: INK, fill: white, w: 23mm, h: 9mm, size: 11pt) = node(
      xy(pos),
      box(width: w, height: h, align(center + horizon, text(size: size, body))),
      shape: fletcher.shapes.rect,
      inset: 0pt,
      corner-radius: 2pt,
      stroke: .85pt + color,
      fill: fill,
    )
    let graph-arrow(a, b, ..options) = flow-arrow(xy(a), xy(b), ..options)
    graph-node((0,0), [token IDs], w: 22mm, h: 8mm)
    graph-node((0,18), [$E$], color: ACC, fill: PALE-ACC, w: 22mm, h: 8mm)
    graph-node((0,40), [$e_i=E[t_i]$\ rows of $X$], color: TEAL, fill: PALE-TEAL, w: 25mm, h: 12mm)
    for (y, weight, output) in ((12, [$W_Q$], [$Q$: use $q_6$]), (36, [$W_K$], [$K$]), (60, [$W_V$], [$V$])) {
      graph-node((34,y - 12), weight, color: ACC, fill: PALE-ACC, w: 22mm, h: 8mm)
      graph-node((34,y), output, color: BLUE, fill: PALE-BLUE, w: 25mm, h: 9mm)
    }
    graph-node((66,24), [dot + scale\ $s$], color: BLUE, fill: PALE-BLUE, w: 23mm, h: 12mm)
    graph-node((98,24), [softmax\ $alpha$], color: BLUE, fill: PALE-BLUE, w: 24mm, h: 12mm)
    graph-node((98,57), [mix values\ $h_6$], color: GREEN, fill: PALE-GREEN, w: 24mm, h: 12mm)
    graph-node((130,35), [$W_O$], color: ACC, fill: PALE-ACC, w: 22mm, h: 8mm)
    graph-node((130,57), [map message\ $Delta e_6$], color: GREEN, fill: PALE-GREEN, w: 25mm, h: 12mm)
    graph-node((163,57), [add $e_6$\ $e'_6$], color: GREEN, fill: PALE-GREEN, w: 24mm, h: 12mm)
    graph-node((195,35), [$U,b$], color: ACC, fill: PALE-ACC, w: 22mm, h: 8mm)
    graph-node((195,57), [logits\ $z_6$], color: BLUE, fill: PALE-BLUE, w: 22mm, h: 12mm)
    graph-node((226,35), [target: period], w: 27mm, h: 9mm, size: 10.5pt)
    graph-node((226,57), [cross-entropy\ $L$], color: RED, fill: PALE-RED, w: 29mm, h: 12mm)

    let links = (
      ((0,18),(0,40)),
      ((0,40),(34,12)), ((0,40),(34,36)), ((0,40),(34,60)),
      ((34,0),(34,12)), ((34,24),(34,36)), ((34,48),(34,60)),
      ((34,12),(66,24)), ((34,36),(66,24)), ((66,24),(98,24)),
      ((98,24),(98,57)), ((34,60),(98,57)),
      ((98,57),(130,57)), ((130,35),(130,57)), ((130,57),(163,57)),
      ((163,57),(195,57)), ((195,35),(195,57)), ((195,57),(226,57)),
    )
    for (a,b) in links {
      if backward { graph-arrow(b, a, color: RED, thickness: 1.1pt) }
      else { graph-arrow(a, b, color: MUTED) }
    }
    // Integer IDs and the observed target are data, not trainable parameters.
    graph-arrow((0,0), (0,40), color: MUTED, bend: -40deg)
    graph-arrow((226,35), (226,57), color: MUTED)
    let retained-path = ((0,40), (0,74), (163,74), (163,57)).map(xy)
    edge(
      vertices: if backward { retained-path.rev() } else { retained-path },
      "-|>", stroke: 1.1pt + if backward { RED } else { TEAL },
      label: text(size: 9.5pt, fill: if backward { RED } else { TEAL },
        if backward { [gradient through the retained input] } else { [retain the lookup embedding $e_6$] }),
      label-pos: 55%, corner-radius: 3mm,
    )
  },
  ))
}

#[
== Keep bank, then add what the context contributes

#set text(size: 17pt)
#bt-prefix
#v(13pt)
#grid(columns: (1fr, 1fr), gutter: 20pt,
  hairline([Lookup embedding at bank], [$e_6=mat(1,1,0)$], color: TEAL),
  hairline([Incoming message], [$h_6 approx mat(1.278,0.470,0.235)$], color: GREEN),
)
#bt-small[Keep the unscaled hand calculation for this first addition.]
#pause
#v(17pt)
#align(center)[$Delta e_6=h_6 quad "for now: add the message unchanged"$]
#v(8pt)
#align(center, text(fill: GREEN)[$e'_6=e_6+Delta e_6 approx mat(2.278,1.470,0.235)$])
#pause
#v(14pt)
#note(color: BLUE)[Predict the next token from $e'_6$: bank's embedding *plus* contextual information. If the message is zero, $e_6$ remains.]
#bt-small[This changes the representation for this input. It does not overwrite bank's row in $E$.]

== Start with one embedding row per token

#set text(size: 16pt)
#bt-small[$E$: eight token types × three coordinates. IDs run from 0 to 7; input positions from 1 to 6.]
#v(8pt)
#grid(columns: (1fr, 1fr), gutter: 25pt,
  [#bt-table((9mm, 34mm, 1fr), [*ID*], [*token*], [*embedding row*],
    [0], [`She`], [$mat(0,0,1)$], [1], [`deposited`], [$mat(1,0,1)$],
    [2], [`money`], [$mat(2,0,0)$], [3], [`in`], [$mat(0,1,0)$],
    [4], [`the`], [$mat(0,0,0)$], [5], [`bank`], [$mat(1,1,0)$],
    [6], [period (`.`)], [$mat(0,0,0.5)$], [7], [`and`], [$mat(0.5,1,1)$])],
  [#uncover("2-")[#text(fill: BLUE)[$X = mat(0,0,1;1,0,1;2,0,0;0,1,0;0,0,0;1,1,0)$]
    #v(6pt)
    $e_i=E[t_i]$. Row $i$ of $X$ is $e_i$.\
    $X$: six sequence rows, $6 times 3$.
  ]],
)
#v(10pt)
#uncover("3-")[#torch-code("X = E[ids]                  # rows e_i, [6, 3]\ne6 = X[-1]                  # bank at position 6", takeaway: [Here $t_6=5$: position 6 looks up token ID 5, the row for bank.])]

== Three learned matrices give one vector three jobs

#set text(size: 16pt)
#align(center)[$e_6 = mat(1,1,0)$ #h(10pt) the lookup embedding at `bank`]
#v(13pt)
#bt-table((34mm, 45mm, 1fr),
  [*transform*], [*result*], [*job in this calculation*],
  text(fill: BLUE)[$e_6 W_Q$], text(fill: BLUE)[$q_6$], [What context would help continue after bank?],
  uncover("2-")[#text(fill: TEAL)[$e_6 W_K$]], uncover("2-")[#text(fill: TEAL)[$k_6$]], uncover("2-")[When should another query match this source?],
  uncover("3-")[#text(fill: ACC)[$e_6 W_V$]], uncover("3-")[#text(fill: ACC)[$v_6$]], uncover("3-")[What information should this source contribute?],
)
#v(16pt)
#uncover("3-")[#note[Each $W$ is a shared, learned transformation. Every input position uses the same three matrices. Their outputs are new vectors, not new parameters.]]
#v(5pt)
#bt-small[$W_Q,W_K in RR^(d times d_k)$; $W_V in RR^(d times d_v)$. Here $d=d_k=d_v=3$.]
#bt-small[These questions describe roles; the model computes vectors, not English questions.]

== Use the same query map at every position

#set text(size: 17pt)
#block[
#set par(spacing: 6pt)
#align(center, text(fill: BLUE)[$q_i=e_i W_Q quad W_Q=mat(1,0,0;0,0,0;0,0,0)$])
#v(7pt)
#bt-small[One shared 3 × 3 matrix. It is not a separate learned matrix for each word.]
#pause
#v(15pt)
#bt-table((39mm, 1fr, 1fr),
  [*position*], [*embedding row*], text(fill: BLUE)[*query row*],
  [1 · `She`], [$e_1=mat(0,0,1)$], text(fill: BLUE)[$q_1=mat(0,0,0)$],
  [3 · `money`], [$e_3=mat(2,0,0)$], text(fill: BLUE)[$q_3=mat(2,0,0)$],
  [6 · `bank`], [$e_6=mat(1,1,0)$], text(fill: BLUE)[$q_6=mat(1,0,0)$],
)
#v(12pt)
#align(center)[$underbrace(Q, 6 times 3)=underbrace(X, 6 times 3) underbrace(W_Q, 3 times 3)$]
#pause
#v(10pt)
#torch-code("Q = X @ W_Q                 # [6, 3]\nq6 = Q[-1]                  # [3]", takeaway: [Q stacks all six queries. We use its bank row to predict after this prefix.])
#bt-small[In math queries are rows; PyTorch's Q[-1] is a one-dimensional tensor.]
]

== The other two projections reproduce our keys and values

#set text(size: 16pt)
#align(center)[$W_K = mat(1,0,0;0,1,0;0,0,1) quad W_V = mat(1,0,0;0,2,0;0,0,1)$]
#v(12pt)
#grid(columns: (1fr, 1fr), gutter: 24pt,
  [#text(fill: TEAL)[$K = X W_K = mat(0,0,1;1,0,1;2,0,0;0,1,0;0,0,0;1,1,0)$]],
  [#uncover("2-")[#text(fill: ACC)[$V = X W_V = mat(0,0,1;1,0,1;2,0,0;0,2,0;0,0,0;1,2,0)$]]],
)
#v(10pt)
#uncover("2-")[#bt-small[Rows in both matrices: She · deposited · money · in · the · bank. Both are 6 × 3.]]
#v(10pt)
#uncover("3-")[#torch-code("K = X @ W_K                 # matching vectors\nV = X @ W_V                 # information to mix", takeaway: [These simple initial matrices reproduce every vector in our hand calculation. Training can change every entry.])]

== Is the value just the original embedding?

#set text(size: 17pt)
#grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([Lookup embedding], [#align(center)[$e_6 = mat(1,1,0)$]\ The row read from the table $E$.], color: TEAL),
  hairline([Value supplied to the mixture], [#align(center)[$v_6 = e_6 W_V = mat(1,2,0)$]\ A learned transformation of that row.], color: ACC),
)
#pause
#v(20pt)
#note(color: ACC)[If $W_V=I$, then $v_j=e_j$: using the original embedding as a value is valid. Learning $W_V$ lets the model learn *what information to send*.]
#pause
#v(14pt)
#note(color: BLUE)[A learned $W_V$ selects, combines, or reweights input features. Queries and keys choose the mixing amounts; values supply the information to mix.]
#bt-small[Here all widths are 3. Query/key widths must agree; value width may differ. Coordinates need not have simple meanings.]

== Map the incoming message into the space we are updating

#set text(size: 17pt)
#align(center)[$h_6 in RR^(1 times d_v) quad arrow.r^(W_O) quad Delta e_6 in RR^(1 times d)$]
#v(12pt)
#note(color: GREEN)[So far we passed the message through unchanged. A learned $W_O$ can translate it into the embedding coordinates before addition.]
#pause
#v(13pt)
#align(center)[$W_O in RR^(d_v times d), quad e'_6=e_6+h_6W_O$]
#v(8pt)
#note[Here $d_v=d=3$ and we initialize $W_O=I$. For now the message passes through unchanged. Training can change that map too.]
#pause
#v(13pt)
#torch-code("delta6 = h6 @ W_O           # message in embedding coordinates\ne6_new = X[-1] + delta6     # retain bank + add context")

== Why divide a dot product by the square root of its width?

#set text(size: 17pt)
#align(center)[$s_j = (q_6 dot k_j) / sqrt(d_k), quad d_k = 3$]
#v(10pt)
#note[The dot product adds $d_k$ products. With more coordinates, its typical magnitude grows even without a better match.]
#pause
#v(10pt)
#bt-table((1fr, 1fr, 1fr), [*query/key width*], [*typical score scale*], [*after division*],
  [$d_k = 4$], [$sqrt(4) = 2$], [$2 ÷ 2 = 1$],
  [$d_k = 64$], [$sqrt(64) = 8$], [$8 ÷ 8 = 1$],
)
#bt-small[Under the simple assumption of independent, zero-mean, unit-variance coordinates: variance of the dot product is $d_k$.]
#pause
#v(11pt)
#note(color: BLUE)[Scaling keeps softmax from becoming too sharp just because vectors are wider. It does not normalize each vector or force a self-match to win.]
#bt-small[#link("https://arxiv.org/abs/1706.03762")[Vaswani et al., §3.2.1]. Standard deviation, not variance, gives the square root.]

== Use scaled scores for the rest of the model

#set text(size: 16pt)
#align(center)[$sqrt(3) approx 1.732 quad s_j = (q_6 dot k_j) / sqrt(3)$]
#v(3pt)
#bt-table((37mm, 1fr, 1fr, 1fr), [*input*], [*dot product*], [*scaled score*], [*weight*],
  [`She`], [0], [0.000], [0.103], [`deposited`], [1], [0.577], [0.183],
  [`money`], [2], [1.155], [0.326], [`in`], [0], [0.000], [0.103],
  [`the`], [0], [0.000], [0.103], [`bank`], [1], [0.577], [0.183],
)
#pause
#v(3pt)
#align(center, text(fill: GREEN)[$h_6 = sum_j alpha_j v_j approx mat(1.018,0.571,0.286)$])
#bt-small[Scaled numbers replace the warm-up; displayed weights are rounded. $alpha_j=alpha_(6 j)$: source $j$ to bank.]
#pause
#v(2pt)
#torch-code("s = (Q[-1] @ K.T) / math.sqrt(K.shape[-1])\na = s.softmax(dim=-1)\nh6 = a @ V                 # use full-precision weights")

== Predict from bank plus its contextual update

#set text(size: 16pt)
#align(center, text(fill: GREEN)[$e'_6=e_6+h_6W_O approx mat(2.018,1.571,0.286)$])
#v(5pt)
#align(center)[$z_6 = e'_6 U + b quad (1 times 3)(3 times 8) + (1 times 8)$]
#v(10pt)
#bt-small[Vocabulary column order: She · deposited · money · in · the · bank · period (`.`) · and]
#align(center)[$U = mat(0,0,0,0,0,0,1,0;0,0,0,0,0,0,0,1;0,0,0,0,0,0,0,0), quad b = 0_(1 times 8)$]
#pause
#v(12pt)
#align(center)[$z_(6,"period") = 2.018(1) + 1.571(0) + 0.286(0) approx 2.018$]
#align(center)[$z_(6,"and") approx 1.571$; the other six logits are 0.]
#pause
#v(10pt)
#torch-code("e6_new = X[-1] + h6 @ W_O   # [3]\nlogits = e6_new @ U + b     # [8]")

== Vocabulary softmax answers “what could come next?”

#set text(size: 16pt)
#bt-prefix
#v(12pt)
#bt-table((1fr, 1fr, 1fr), [*candidate token*], [*logit*], [*probability*],
  [`She`, `deposited`, `money`, `in`, `the`, `bank`], [0.000 each], [0.055 each],
  text(fill: ACC)[period (`.`)], [2.018], text(fill: ACC)[0.410],
  [`and`], [1.571], [0.263],
)
#pause
#v(13pt)
#align(center)[$p(".") = exp(2.018) / (6 exp(0) + exp(2.018) + exp(1.571)) approx 0.410$]
#bt-small[The calculation uses unrounded logits. The six zero logits still receive probability mass.]
#bt-small[Probabilities are rounded for display, so the displayed entries need not sum exactly to 1.]
#pause
#v(12pt)
#grid(columns: (1fr, 1fr), gutter: 23pt,
  note(color: BLUE)[Attention softmax: *six source positions*, used to mix vectors.],
  note(color: ACC)[Vocabulary softmax: *eight token types*, used to choose the next token.],
)

== Follow one complete forward pass

#set text(size: 16pt)
#bt-prefix
#v(16pt)
#bt-table((67mm, 50mm, 1fr), [*operation*], [*tensor / shape*], [*one row we followed*],
  [look up embeddings], [$X: 6 times 3$], [$e_6 = mat(1,1,0)$],
  [make matching vectors], [$Q,K: 6 times 3$], [$q_6 = mat(1,0,0)$],
  [make value vectors], [$V: 6 times 3$], [$v_6 = mat(1,2,0)$],
  uncover("2-")[score, scale, softmax], uncover("2-")[$s,alpha: 1 times 6$], uncover("2-")[`money` weight $approx 0.326$],
  uncover("2-")[mix values], uncover("2-")[$h_6: 1 times 3$], uncover("2-")[$mat(1.018,0.571,0.286)$],
  uncover("3-")[map and add the message], uncover("3-")[$e'_6: 1 times 3$], uncover("3-")[$mat(2.018,1.571,0.286)$],
  uncover("3-")[predict next token], uncover("3-")[$z_6,p: 1 times 8$], uncover("3-")[$p(".") approx 0.410$],
)
#v(9pt)
#uncover("3-")[#bt-small[$X$ stacks $e_i$; $Delta X$ stacks $Delta e_i$; $X'$ stacks $e'_i$. Capital $E$ remains the lookup table.]]
#uncover("3-")[#bt-small[In the next snippets, forward_last(ids) runs this calculation and returns the last row's logits. Parameters stay fixed.]]

== Generate by choosing one token from this distribution

#set text(size: 17pt)
#bt-prefix
#v(8pt)
#align(center)[$p(".") approx 0.410, quad p("and") approx 0.263$]
#pause
#v(8pt)
#note(color: BLUE)[Greedy decoding chooses the period token here. Sampling can choose another token with nonzero probability. Suppose this sample is `and`.]
#v(12pt)
#align(center, text(size: 20pt)[`She deposited money in the bank` #text(fill: BLUE)[`and`] #text(fill: ACC)[`___`]])
#pause
#v(8pt)
#torch-code("p = logits.softmax(dim=-1)  # probabilities over vocabulary\nnext_id = torch.multinomial(p, num_samples=1)\nids = torch.cat([ids, next_id])", takeaway: [“and” is one possible sample, not the most probable token or the observed training answer.])

== The appended token supplies the next query

#set text(size: 16pt)
#align(center, text(size: 19pt)[`She deposited money in the bank` #text(fill: BLUE)[`and`] #text(fill: ACC)[`___`]])
#v(13pt)
#bt-table((1fr, 1fr, 1fr), [*generation step*], [*last input vector*], [*last query*],
  [after `bank`, position 6], [$e_6=mat(1,1,0)$], [$q_6 = mat(1,0,0)$],
  uncover("2-")[after `and`, position 7], uncover("2-")[$e_7=mat(0.5,1,1)$], uncover("2-")[$q_7 = mat(0.5,0,0)$],
)
#pause
#v(13pt)
#note(color: TEAL)[The new token gets its own lookup embedding $e_7$, not bank's $e'_6$. Its query collects $h_7$ from seven sources; add the mapped message to $e_7$.]
#pause
#v(13pt)
#torch-code("X = E[ids]                  # now [7, 3]\nQ, K, V = X @ W_Q, X @ W_K, X @ W_V\na = ((Q[-1] @ K.T) / math.sqrt(3)).softmax(-1)\nh7 = a @ V", takeaway: [The matrices are unchanged. The last input and the set of available sources changed.])

== Reuse the same model for every generation step

#set text(size: 16pt)
#align(center)[$"prefix" arrow.r h_("last") arrow.r e'_("last") arrow.r p("next") arrow.r "append"$]
#v(8pt)
#align(center)[$h_7 approx mat(0.791,0.811,0.406)$]
#align(center, text(fill: GREEN)[$e'_7=e_7+h_7W_O approx mat(1.291,1.811,1.406)$])
#bt-small[After the possible sample and: $p(".") approx 0.231$ and $p("and") approx 0.388$.]
#pause
#v(8pt)
#torch-code("p = forward_last(ids).softmax(-1)\nnext_id = torch.multinomial(p, 1)\nids = torch.cat([ids, next_id])")
#pause
#v(8pt)
#note(color: ACC)[Generation changes the inputs, with *fixed parameters*. Training uses observed targets to change the parameters.]
#bt-small[A repeated last word can have the same first-layer query in this content-only toy. Growing context can still change the mixture.]

== The observed next token tells us the prediction error

#set text(size: 17pt)
#align(center, text(size: 20pt)[Training text: `She deposited money in the bank` #text(fill: ACC)[`.`]])
#v(13pt)
#note[Return to the six-token prefix, before the sampled `and`. The target is the period token (`.`). It was never an input to this prediction.]
#pause
#v(16pt)
#align(center, text(fill: RED)[$L = -ln p("." | "She deposited money in the bank") approx -ln(0.410) approx 0.891$])
#bt-small[This is one next-token cross-entropy loss, using the natural logarithm.]
#pause
#v(15pt)
#torch-code("ids = torch.tensor([0, 1, 2, 3, 4, 5])\nlogits = forward_last(ids)\ntarget = torch.tensor(6)     # vocabulary ID for '.'\nloss = F.cross_entropy(logits, target)", takeaway: [Pass logits to cross_entropy. It applies the required log-softmax internally.])

== Which numbers will the optimizer change?

#set text(size: 15pt)
#grid(columns: (1fr, 1fr), gutter: 25pt,
  hairline([Learned parameters], [
    $E$: token embedding table\
    $W_Q, W_K, W_V, W_O$: projection matrices\
    $U, b$: vocabulary predictor
    #v(9pt)
    Stored across examples. Updated by the optimizer.
  ], color: ACC),
  uncover("2-")[#hairline([Recomputed intermediates], [
    $e_i,Q,K,V$: this input's vectors\
    $s,alpha$: scores and attention weights\
    $h,Delta e,e'$: message and contextual embedding\
    $z,p,L$: prediction and loss
    #v(9pt)
    Recomputed from the inputs and current parameters.
  ], color: BLUE)],
)
#pause
#pause
#v(8pt)
#note(color: RED)[Adding context gives $e'_6=e_6+Delta e_6$ for this input. Learning changes the shared parameters, including $E$. These are two different kinds of update.]
#v(7pt)
#uncover("3-")[#torch-code("params = [E, W_Q, W_K, W_V, W_O, U, b]\noptimizer = torch.optim.SGD(params, lr=0.1)")]

== The forward pass is a computational graph

#set text(size: 16pt)
#bt-small[Input: `She deposited money in the bank`. Orange boxes are learned parameters.]
#v(4pt)
#bt-computation-graph()
#pause
#v(6pt)
#note[Follow the arrows to the loss. Each operation records which tensors produced its output.]
#pause
#v(4pt)
#bt-small[The retained input and incoming message meet at addition. The updated state supplies logits for the observed period target.]

== Autograd follows the graph backward

#set text(size: 15pt)
#bt-small[Red arrows carry loss gradients back through the same operations.]
#v(4pt)
#bt-computation-graph(backward: true)
#pause
#v(5pt)
#text(size: 13pt)[#torch-code("loss.backward()\ngrad_WQ = W_Q.grad           # dL / dW_Q\ngrad_E = E.grad             # dL / dE")]
#pause
#v(4pt)
#bt-small[Gradients flow through matching, values, and the retained-input path. Learned tensors require gradients; integer IDs and targets are data.]

== Apply the update to all learned parameters

#set text(size: 17pt)
#align(center)[$theta = {E,W_Q,W_K,W_V,W_O,U,b}, quad theta <- theta - eta nabla_theta L$]
#v(18pt)
#note[Autograd computes the gradients. The optimizer uses them to change the learned parameters. We do not need to derive each gradient by hand.]
#pause
#v(17pt)
#torch-code("optimizer.zero_grad()\nloss = F.cross_entropy(forward_last(ids), target)\nloss.backward()\noptimizer.step()")
#pause
#v(15pt)
#note(color: RED)[Use the original six-token prefix and learning rate $eta=0.1$. The next forward pass recomputes the queries, keys, values, and prediction from the updated parameters.]

== Run the same prefix again after the update

#set text(size: 16pt)
#bt-prefix
#v(8pt)
#bt-table((1fr, 1fr, 1fr), [*computed quantity*], [*before SGD*], [*after SGD*],
  [lookup $e_6$ from table $E$], [$mat(1.000,1.000,0.000)$], [$mat(1.091,0.964,0.000)$],
  [incoming message $h_6$], [$mat(1.018,0.571,0.286)$], [$mat(1.196,0.488,0.272)$],
  [mapped message $Delta e_6$], [$mat(1.018,0.571,0.286)$], [$mat(1.289,0.446,0.272)$],
  [contextual embedding $e'_6$], [$mat(2.018,1.571,0.286)$], [$mat(2.379,1.410,0.272)$],
  [probability of the period token], [0.410], [0.658],
  [loss $-ln p(".")$], [0.891], [0.419],
)
#pause
#v(10pt)
#note(color: GREEN)[Learning changed $E$, so the lookup $e_6$ changed too. $W_O$ is no longer identity: $h_6$ and $Delta e_6$ now differ.]
#pause
#v(8pt)
#note[The model is still tiny and barely trained. Learning on one example does not establish good predictions on unseen text.]
#bt-small[Displayed rows are rounded separately; additions use full precision. Next, train at every position.]

]
