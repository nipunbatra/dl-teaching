// Transformers I: Self-Attention, Positional Encoding and Transformer Blocks
// Lecture 15 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture15/L15-transformers-i.typ /tmp/L15.pdf
//   typst compile --root . --input handout=true lecture15/L15-transformers-i.typ /tmp/L15h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Transformers I: Self-Attention, Positional Encoding and Transformer Blocks],
  subtitle: [Replacing recurrence with attention between all tokens],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let cream = rgb("#EFEEEB")

// ═══════════════════════════ fletcher diagrams (centerpieces) ═══════════════════════════

// ── (A) RNN chain vs self-attention all-to-all: same two-row layout for a fair compare ──
#let rnnvsattn = align(center, diagram(spacing: (10mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  // ---- left cluster: RNN (local + sequential chain) ----
  node((1.35, 2.1), text(size: 13pt, fill: INK)[*RNN — sequential*], stroke: none, fill: none)
  for (i, l) in ((0, $x_1$), (1, $x_2$), (2, $x_3$), (3, $x_4$)) {
    node((i*0.9, 0), text(size: 11pt, l), radius: 5.5mm, fill: cream)
    node((i*0.9, 1.3), text(size: 11pt, fill: white, $h_#(i+1)$), radius: 5.5mm, fill: TEAL, stroke: 0.9pt + TEAL)
    edge((i*0.9, 0), (i*0.9, 1.3), "-|>", stroke: 0.7pt + MUTED)
  }
  for i in range(3) { edge((i*0.9, 1.3), ((i+1)*0.9, 1.3), "-|>", stroke: 1.1pt + TEAL) }
  node((1.35, -0.95), text(size: 10.5pt, fill: MUTED)[path $i -> j$: #h(0.2em) $O(|i - j|)$ steps], stroke: none, fill: none)
  // ---- right cluster: self-attention (all-to-all bipartite) ----
  let bx = 5.0
  node((bx + 1.35, 2.1), text(size: 13pt, fill: INK)[*self-attention — all-to-all*], stroke: none, fill: none)
  for (i, l) in ((0, $x_1$), (1, $x_2$), (2, $x_3$), (3, $x_4$)) {
    node((bx + i*0.9, 0), text(size: 11pt, l), radius: 5.5mm, fill: cream)
    node((bx + i*0.9, 1.3), text(size: 11pt, fill: white, $z_#(i+1)$), radius: 5.5mm, fill: ACC, stroke: 0.9pt + ACC)
  }
  // dense edges: every z_i attends to every x_j
  for i in range(4) { for j in range(4) {
    edge((bx + j*0.9, 0), (bx + i*0.9, 1.3), stroke: 0.4pt + BLUE.lighten(35%))
  } }
  node((bx + 1.35, -0.95), text(size: 10.5pt, fill: MUTED)[path $i -> j$: #h(0.2em) $O(1)$ — one layer], stroke: none, fill: none)
}))

// ── (B) attention as a soft dictionary: q vs keys -> scores -> softmax -> weight values -> o ──
#let qkvlookup = align(center, diagram(spacing: (12mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), $q$, radius: 6mm, fill: BLUE.lighten(80%), stroke: 1pt + BLUE)
  node((0, -1.9), text(size: 10.5pt, fill: BLUE)[query], stroke: none, fill: none)
  // keys
  for (j, yj) in ((1, 1.4), (2, 0), (3, -1.4)) {
    node((1.6, yj), text(size: 12pt, $k_#j$), radius: 5.5mm, fill: TEAL.lighten(80%), stroke: 0.9pt + TEAL)
    node((3.0, yj), text(size: 11pt, $s_#j$), radius: 5mm, fill: cream)
    edge((0, 0), (3.0, yj), "-|>", stroke: 0.5pt + MUTED)
    edge((1.6, yj), (3.0, yj), "-|>", stroke: 0.7pt + TEAL, label: text(size: 9pt, fill: TEAL)[$q dot.c k_#j$], label-side: left)
  }
  node((1.6, 2.35), text(size: 10pt, fill: TEAL)[keys], stroke: none, fill: none)
  // softmax turns scores into weights
  node((4.3, 0), text(size: 11pt)[softmax], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 6pt, fill: rgb("#FDECD6"), stroke: 0.9pt + ACC)
  for yj in (1.4, 0, -1.4) { edge((3.0, yj), (4.3, 0), "-|>", stroke: 0.6pt + MUTED) }
  // weights and values -> weighted sum
  for (j, yj) in ((1, 1.4), (2, 0), (3, -1.4)) {
    node((5.6, yj), text(size: 11pt, fill: GREEN)[$alpha_#j$], radius: 4.5mm, fill: GREEN.lighten(85%), stroke: 0.9pt + GREEN)
    node((7.0, yj), text(size: 12pt, $v_#j$), radius: 5.5mm, fill: ACC.lighten(80%), stroke: 0.9pt + ACC)
    edge((4.3, 0), (5.6, yj), "-|>", stroke: 0.6pt + MUTED)
    edge((5.6, yj), (7.0, yj), "-|>", stroke: 0.7pt + GREEN, label: text(size: 9pt, fill: GREEN)[weight], label-side: left)
    edge((7.0, yj), (8.4, 0), "-|>", stroke: 0.8pt + ACC)
  }
  node((7.0, 2.35), text(size: 10pt, fill: ACC)[values], stroke: none, fill: none)
  node((8.4, 0), text(size: 14pt)[$Sigma$], radius: 5.5mm, fill: cream, stroke: 1pt + INK)
  node((9.7, 0), $o$, radius: 6mm, fill: ACC.lighten(70%), stroke: 1pt + ACC)
  edge((8.4, 0), (9.7, 0), "-|>", stroke: 1.1pt + ACC)
  node((9.7, -1.4), text(size: 10pt, fill: ACC)[$o = sum_j alpha_j v_j$], stroke: none, fill: none)
}))

// ── (C) multi-head attention: X -> H parallel heads -> concat -> W_O ──
#let mhablock = align(center, diagram(spacing: (13mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), $X$, radius: 6mm, fill: cream, stroke: 1pt + INK)
  node((0, -1.85), text(size: 10pt, fill: MUTED)[$T times d$], stroke: none, fill: none)
  // three heads (H of them)
  for (h, yh, c) in ((1, 1.5, TEAL), (2, 0.35, BLUE), (8, -1.5, GREEN)) {
    node((1.7, yh), text(size: 11pt)[head #h], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 6pt, fill: c.lighten(82%), stroke: 0.9pt + c)
    edge((0, 0), (1.7, yh), "-|>", stroke: 0.7pt + MUTED)
    edge((1.7, yh), (3.5, 0), "-|>", stroke: 0.7pt + MUTED)
  }
  node((1.7, -0.6), text(size: 15pt, fill: MUTED)[$dots.v$], stroke: none, fill: none)
  node((3.5, 0), text(size: 11pt)[concat], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 6pt, fill: cream, stroke: 0.9pt + INK)
  node((4.8, 0), text(size: 12pt)[$W_O$], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 6pt, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  node((6.1, 0), text(size: 11pt)[MHA$(X)$], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 6pt, fill: ACC.lighten(70%), stroke: 1pt + ACC)
  edge((3.5, 0), (4.8, 0), "-|>", stroke: 0.9pt + INK)
  edge((4.8, 0), (6.1, 0), "-|>", stroke: 0.9pt + INK)
  node((3.5, -1.85), text(size: 10pt, fill: MUTED)[$T times H d_v$], stroke: none, fill: none)
  node((6.1, -1.85), text(size: 10pt, fill: MUTED)[$T times d$], stroke: none, fill: none)
}))

// ── (D) Transformer ENCODER block: residual stream spine + two sublayers branching up ──
#let encblock = align(center, diagram(spacing: (12.5mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white, {
  // spine (residual stream) — thick horizontal INK line
  node((0, 0), $x$, radius: 5.5mm, fill: cream, stroke: 1pt + INK)
  node((2, 0), text(size: 14pt)[$+$], radius: 5mm, fill: GREEN.lighten(80%), stroke: 1pt + GREEN)
  node((3.3, 0), text(size: 11pt)[LN], radius: 5mm, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((5.3, 0), text(size: 14pt)[$+$], radius: 5mm, fill: GREEN.lighten(80%), stroke: 1pt + GREEN)
  node((6.6, 0), text(size: 11pt)[LN], radius: 5mm, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((7.9, 0), $y$, radius: 5.5mm, fill: ACC.lighten(72%), stroke: 1pt + ACC)
  edge((0, 0), (2, 0), "-|>", stroke: 1.6pt + INK)
  edge((2, 0), (3.3, 0), "-|>", stroke: 1.6pt + INK)
  edge((3.3, 0), (5.3, 0), "-|>", stroke: 1.6pt + INK)
  edge((5.3, 0), (6.6, 0), "-|>", stroke: 1.6pt + INK)
  edge((6.6, 0), (7.9, 0), "-|>", stroke: 1.6pt + INK)
  // sublayer branches above
  node((1, 1.5), align(center, text(size: 10.5pt)[multi-head \ self-attention]), shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 6pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((4.3, 1.5), text(size: 11pt)[FFN], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  edge((0, 0), (1, 1.5), "-|>", stroke: 0.9pt + TEAL)
  edge((1, 1.5), (2, 0), "-|>", stroke: 0.9pt + TEAL)
  edge((3.3, 0), (4.3, 1.5), "-|>", stroke: 0.9pt + ACC)
  edge((4.3, 1.5), (5.3, 0), "-|>", stroke: 0.9pt + ACC)
  node((3.95, -1.0), text(size: 10.5pt, fill: INK)[the horizontal spine is the *residual stream* #h(0.3em) (LN $=$ LayerNorm)], stroke: none, fill: none)
}))

// ── (E) decoder-only (GPT-style) block: causal self-attn + FFN + LM head ──
#let decblock = align(center, diagram(spacing: (13mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), $x$, radius: 5.5mm, fill: cream, stroke: 1pt + INK)
  node((2, 0), text(size: 14pt)[$+$], radius: 5mm, fill: GREEN.lighten(80%), stroke: 1pt + GREEN)
  node((3.3, 0), text(size: 11pt)[LN], radius: 5mm, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((5.3, 0), text(size: 14pt)[$+$], radius: 5mm, fill: GREEN.lighten(80%), stroke: 1pt + GREEN)
  node((6.6, 0), text(size: 11pt)[LN], radius: 5mm, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  edge((0, 0), (2, 0), "-|>", stroke: 1.6pt + INK)
  edge((2, 0), (3.3, 0), "-|>", stroke: 1.6pt + INK)
  edge((3.3, 0), (5.3, 0), "-|>", stroke: 1.6pt + INK)
  edge((5.3, 0), (6.6, 0), "-|>", stroke: 1.6pt + INK)
  node((1, 1.5), align(center, text(size: 10.5pt)[masked (causal) \ self-attention]), shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 6pt, fill: RED.lighten(85%), stroke: 0.9pt + RED)
  node((4.3, 1.5), text(size: 11pt)[FFN], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  edge((0, 0), (1, 1.5), "-|>", stroke: 0.9pt + RED)
  edge((1, 1.5), (2, 0), "-|>", stroke: 0.9pt + RED)
  edge((3.3, 0), (4.3, 1.5), "-|>", stroke: 0.9pt + ACC)
  edge((4.3, 1.5), (5.3, 0), "-|>", stroke: 0.9pt + ACC)
  // LM head
  node((8.0, 0), align(center, text(size: 10pt)[Linear \ softmax]), shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 6pt, fill: cream, stroke: 0.9pt + INK)
  node((9.7, 0), text(size: 11pt)[$p(x_(t+1) | x_(<= t))$], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(72%), stroke: 1pt + ACC)
  edge((6.6, 0), (8.0, 0), "-|>", stroke: 1.6pt + INK, label: text(size: 10pt, fill: MUTED)[$h_t$], label-side: right)
  edge((8.0, 0), (9.7, 0), "-|>", stroke: 0.9pt + ACC)
}))

// ── (F) tiny decoder-only LM pipeline (Part IX) ──
#let tinylm = align(center, diagram(spacing: (13mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(x, lbl, c, fill) = node((x, 0), align(center, text(size: 10.5pt, lbl)), shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 6pt, stroke: 0.9pt + c, fill: fill)
  bx(0, [token ids \ `<BOS> I like`], INK, cream)
  bx(1.35, [embed \ $+ space$ pos], TEAL, TEAL.lighten(82%))
  bx(2.7, [causal \ self-attn], RED, RED.lighten(85%))
  bx(3.9, [FFN], ACC, ACC.lighten(82%))
  bx(5.0, [logits], BLUE, BLUE.lighten(85%))
  bx(6.2, [softmax \ `-> cats`], GREEN, GREEN.lighten(82%))
  for (a, b) in ((0, 1.35), (1.35, 2.7), (2.7, 3.9), (3.9, 5.0), (5.0, 6.2)) {
    edge((a, 0), (b, 0), "-|>", stroke: 0.8pt + MUTED)
  }
}))

#title-slide()

// ═══════════════════════════ PART I — Why Transformers ═══════════════════════════
= Why Transformers?

== Where we are

Last lectures built sequence models on *recurrence* — RNNs, LSTMs, and seq2seq with *attention*:
#pause
$ h_t = f(h_(t-1), x_t) quad quad #text(size: 15pt, fill: MUTED)[(encoder: a state carried step by step)] $
#pause
$ c_u = sum_(t=1)^T alpha_(u t) thin h_t quad quad #text(size: 15pt, fill: MUTED)[(decoder attends over encoder states $h_t$)] $
#pause
#notebox[Attention already lets the decoder look *directly* at any source position. This lecture asks: what if attention were the *whole* model — and we dropped recurrence entirely?]

== The problem with recurrence #D

Attention fixed the *bottleneck*, but the encoder states $h_t$ are still built *sequentially*:
#pause
$ h_1 -> h_2 -> h_3 -> dots -> h_T quad quad #text(size: 15pt, fill: MUTED)[(each $h_t$ waits for $h_(t-1)$)] $
#pause
#alertbox[$h_t = f(h_(t-1), x_t)$ is *inherently serial*: you cannot compute step $t$ before step $t-1$. On modern parallel hardware (GPUs) this wastes the machine — training time grows with sequence length $T$.]
#pause
#result[we want the representation of every token computed *in parallel*]

== The Transformer idea #D

Replace the recurrent update with a function of *all* tokens at once:
#pause
$ underbrace(h_t = f(h_(t-1), x_t), "RNN: depends on the previous state") quad --> quad underbrace(z_i = f(x_1, dots, x_T), "Transformer: attends to all tokens") $
#pause
Each token builds its new representation by *directly attending* to every other token:
#pause
#notebox[*Self-attention* relates positions *within the same sequence*. No state to carry — every $z_i$ is computed independently and in parallel (Vaswani et al. 2017, "Attention Is All You Need").]

== Three ways to mix information #V

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 7pt), align: (left, left, left, left),
  table.header([], [*RNN*], [*CNN*], [*Transformer*]),
  [context mechanism], [hidden state $h_t$], [receptive field], [attention],
  [reach per layer], [all past (decayed)], [fixed kernel $k$], [all positions],
  [parallel over positions?], [#text(fill: RED)[no] — serial], [#text(fill: GREEN)[yes]], [#text(fill: GREEN)[yes]],
  [content-dependent mixing?], [partly], [#text(fill: RED)[no] — fixed], [#text(fill: GREEN)[yes]],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[the Transformer keeps the CNN's parallelism *and* gains content-dependent, full-range mixing])

== Recurrence vs attention, side by side #V

#rnnvsattn
#pause
#align(center, text(size: 15pt, fill: MUTED)[the RNN routes information along a *chain*; self-attention wires *every token to every token* in one layer])

// ═══════════════════════════ PART II — Attention as soft lookup ═══════════════════════════
= Attention as a soft lookup

== Recall: query, keys, values #D

Attention is a *differentiable dictionary lookup*. Given a query $q$ and $T$ key–value pairs:
#pause
$ s_i = q^top k_i quad quad #text(size: 15pt, fill: MUTED)[(score: how well the query matches key $i$)] $
#pause
$ alpha = "softmax"(s) quad quad o = sum_(i=1)^T alpha_i thin v_i $
#pause
#align(center, text(size: 16pt, fill: MUTED)[a *soft* lookup: instead of picking one entry, return a weighted blend of *all* values])

== The Q / K / V intuition #D

Three learned roles, one per vector:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left, left),
  table.header([*Vector*], [*Question it answers*], [*Role*]),
  [query $q$], [what am I looking for?], [probe],
  [key $k_i$], [what do I contain?], [address],
  [value $v_i$], [what do I pass on?], [content],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[match $q$ against every $k_i$ to decide *how much* of each $v_i$ to retrieve — all differentiable])

== Attention as a soft dictionary #V

#qkvlookup
#pause
#align(center, text(size: 14pt, fill: MUTED)[compare query to keys $->$ scores $->$ softmax weights $alpha$ $->$ blend the values into $o$])

== Worked attention: the setup #D

$q = mat(1, 0)$, three keys, three values:
#pause
$ k_1 = mat(1, 0), quad k_2 = mat(0, 1), quad k_3 = mat(1, 1) $
#pause
$ v_1 = mat(10, 0), quad v_2 = mat(0, 10), quad v_3 = mat(10, 10) $
#pause
Scores are dot products of the query with each key:
$ s = (q^top k_1, thin q^top k_2, thin q^top k_3) = (1, thin 0, thin 1) $

== Worked attention: softmax and output #D

Softmax the scores $s = (1, 0, 1)$:
#pause
$ alpha = "softmax"(1, 0, 1) = (e^1, e^0, e^1) / (e^1 + e^0 + e^1) approx (0.422, thin 0.155, thin 0.422) $
#pause
Blend the values with these weights:
$ o = 0.422 thin v_1 + 0.155 thin v_2 + 0.422 thin v_3 $
#pause
$ o approx (8.45, thin 5.78) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[keys $1$ and $3$ match the query, so $o$ leans toward $v_1$ and $v_3$ — a soft blend, not a hard pick])

== Scaled dot-product attention #D

Stack the queries, keys and values into matrices and do it all at once:
#pause
$ "Attention"(Q, K, V) = "softmax"((Q K^top) / sqrt(d_k)) thin V $
#pause
#result[this single expression is the central operation of the Transformer]
#pause
#align(center, text(size: 15pt, fill: MUTED)[$Q K^top$ is every query dotted with every key; softmax normalizes each row; the result weights $V$])

== Why divide by $sqrt(d_k)$? #D

Suppose query and key entries are independent with mean $0$, variance $1$. Then a single score
#pause
$ q^top k = sum_(i=1)^(d_k) q_i thin k_i quad quad => quad quad "Var"(q^top k) = d_k $
#pause
#alertbox[Large $d_k$ $->$ scores of magnitude $~sqrt(d_k)$ $->$ softmax saturates into a near one-hot spike $->$ gradients through it vanish. Learning stalls.]
#pause
#result[divide by $sqrt(d_k)$ to hold the score variance at $approx 1$, whatever $d_k$ is]

== Matrix shapes #V

One self-attention layer, sequence length $T$, model width $d$, head width $d_k$:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, center, left),
  table.header([*Tensor*], [*Shape*], [*How*]),
  [input $X$], [$T times d$], [one row per token],
  [$W_Q, W_K, W_V$], [$d times d_k$], [learned projections],
  [$Q, K, V$], [$T times d_k$], [$Q = X W_Q$, etc.],
  [scores $A = "softmax"(Q K^top \/ sqrt(d_k))$], [$T times T$], [token-to-token weights],
  [output $O = A V$], [$T times d_k$], [attended representations],
))

== Interactive: attention lookup #I

#interbox(link-to: IA + "attention")[
  Drag the query and watch the scores $q^top k_i$, the softmax weights $alpha_i$, and the blended output $o = sum_i alpha_i v_i$ update — the worked example above, made live.
]

// ═══════════════════════════ PART III — Self-attention ═══════════════════════════
= Self-attention

== Queries, keys and values from one sequence #D

In *self*-attention, all three come from the *same* input $X$ — each token is a query, a key, *and* a value:
#pause
$ Q = X W_Q, quad quad K = X W_K, quad quad V = X W_V $
#pause
#notebox[Each token builds a *new representation* by mixing information from the other tokens it finds relevant — a query looks around, keys advertise, values are gathered.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[same operation as before, but the sequence attends *to itself*])

== The attention matrix #V

$A_(i j)$ = how much token $i$ (query) attends to token $j$ (key); every *row sums to 1*:
#pause
#fig("/lecture15/figures/attn_heatmap.svg", w: 43%)
#pause
#align(center, text(size: 15pt, fill: MUTED)[a $T times T$ matrix — each row is a distribution over which tokens to gather from])

== Worked self-attention #D

Take $W_Q = W_K = W_V = I$, so $Q = K = V = X$ with $X = mat(1, 0; 0, 1; 1, 1)$. The scores $Q K^top = X X^top$:
#pause
#fig("/lecture15/figures/attn_matrix.svg", w: 30%)
#pause
#align(center, text(size: 15pt, fill: MUTED)[the diagonal grows with each token's norm; scale by $sqrt(d_k) = sqrt(2)$ before the softmax next])

== Reading a row #D

Scale by $sqrt(2)$; row $1$ (token $1$ as query) before softmax is $mat(1, 0, 1) \/ sqrt(2) = mat(0.71, 0, 0.71)$:
#pause
$ alpha_1 = "softmax"(0.71, 0, 0.71) approx (0.40, thin 0.20, thin 0.40) $
#pause
$ z_1 = 0.40 thin x_1 + 0.20 thin x_2 + 0.40 thin x_3 approx mat(0.80, 0.60) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[token $1$ attends to tokens $1$ and $3$ more than $2$ — its output is a weighted average of their values])

== Self-attention is content-dependent #D

Compare how a CNN and self-attention decide *which positions* interact:
#pause
#two(
  notebox[*CNN*: the receptive field is *fixed by the kernel* — position $i$ always mixes with $i-1, i, i+1$, regardless of content.],
  notebox[*Self-attention*: $A_(i j)$ depends on the *token representations* — each token *decides* which others matter, per input.],
)
#pause
#result[same weights, different wiring for every sentence — the mixing pattern is *learned and dynamic*]

== Interactive: the self-attention matrix #I

#interbox(link-to: IA + "attention")[
  Edit a short sentence and watch the $T times T$ attention matrix light up — each row a softmax distribution over which tokens that position pulls information from.
]

// ═══════════════════════════ PART IV — Masked / causal self-attention ═══════════════════════════
= Masked (causal) self-attention

== The future-leakage problem #D

For a *language model*, position $t$ must predict $x_(t+1)$ from *only* the past:
#pause
$ p(x_(t+1) | x_(<= t)) quad quad #text(size: 15pt, fill: MUTED)[(it may look at $x_1, dots, x_t$ — never ahead)] $
#pause
#alertbox[Plain self-attention lets token $t$ attend to *every* position, including $x_(t+1), x_(t+2), dots$ — it would peek at the answer. That is *illegal* for autoregressive generation.]

== The causal mask #D

Compute the scores, then set the *future* entries to $-oo$ before the softmax:
#pause
$ S = (Q K^top) / sqrt(d_k), quad quad S_(i j) <- -oo #h(0.6em) "if" #h(0.4em) j > i $
#pause
Softmax turns $-oo$ into a weight of $0$:
$ A_(i j) = 0 quad #text(size: 15pt, fill: MUTED)[for all] quad j > i $
#pause
#align(center, text(size: 16pt, fill: MUTED)[each row now normalizes over the *past and present only* — the future contributes nothing])

== The causal mask, drawn #V

$T = 5$: allowed entries (lower triangle) vs masked future ($-oo$, upper triangle):
#pause
#align(center, attn-matrix(("1", "2", "3", "4", "5"), mask: "causal", cell: 11mm,
  q-label: [query position $i$], k-label: [key position $j$]))
#pause
#align(center, text(size: 15pt, fill: MUTED)[query $i$ may attend to key $j$ only when $j <= i$ — a lower-triangular attention pattern])

== Encoder vs decoder attention #D

The *same* self-attention op, two masking regimes:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 7pt), align: (left, center, left),
  table.header([*Setting*], [*Mask*], [*Sees*]),
  [encoder (BERT-style)], [none], [*bidirectional* — the whole sequence],
  [decoder (GPT-style)], [causal], [only the past $x_(<= t)$],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[bidirectional is great for *understanding*; causal is required for *generation*])

== Interactive: causal attention #I

#interbox(link-to: IA + "attention")[
  Toggle the causal mask on a short sequence and watch the upper triangle of the attention matrix go dark — the future becomes unreachable, one row at a time.
]

// ═══════════════════════════ PART V — Multi-head attention ═══════════════════════════
= Multi-head attention

== Why more than one head? #D

A single attention pattern can track *one* kind of relation at a time. Use $H$ of them in parallel:
#pause
$ "head"_i = "Attention"(X W_i^Q, thin X W_i^K, thin X W_i^V) $
#pause
$ "MHA"(X) = mat("head"_1; dots.v; "head"_H) W_O quad quad #text(size: 14pt, fill: MUTED)[(concatenate, then mix with $W_O$)] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[each head projects into its *own* subspace and learns a *different* relation])

== Multi-head shapes #V

The classic setup: $d_"model" = 512$, $H = 8$ heads:
#pause
$ d_k = d_v = d_"model" / H = 512 / 8 = 64 $
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, center, left),
  table.header([*Per head*], [*Shape*], [*Note*]),
  [$W_i^Q, W_i^K, W_i^V$], [$512 times 64$], [projects into a $64$-dim subspace],
  [$"head"_i$], [$T times 64$], [attention within that subspace],
  [concat of $8$ heads], [$T times 512$], [back to full width],
  [$W_O$], [$512 times 512$], [mixes the heads],
))

== What can heads learn? #D

Different heads specialize — some interpretable patterns observed in trained models:
#pause
#two(
  notebox[- attend to the *previous* word
  - *subject–verb* agreement
  - matching *brackets* / quotes],
  notebox[- local *phrase* structure
  - *positional* patterns
  - coarse *semantic* similarity],
)
#pause
#alertbox[*Caution.* Individual-head interpretation is *noisy* and post-hoc — heads are not cleanly labeled functions, and behavior is distributed across many of them.]

== Worked: multi-head parameter count #D

Combining all $H$ heads, the per-projection weights stack into full $d times d$ matrices. With $d = 512$:
#pause
$ W_Q, W_K, W_V, W_O in RR^(512 times 512) $
#pause
$ 4 times 512^2 = 4 times "262,144" = $ #text(fill: ACC, weight: 600)[$"1,048,576"$] #text(size: 15pt, fill: MUTED)[parameters]
#pause
#align(center, text(size: 16pt, fill: MUTED)[$approx 1.05$ M — this is one attention sublayer; the FFN typically adds a few times more])

== The multi-head block #V

#mhablock
#pause
#align(center, text(size: 14pt, fill: MUTED)[$H$ attentions run in parallel on the same $X$, concatenate, then $W_O$ mixes them back to width $d$])

== Interactive: multi-head attention #I

#interbox(link-to: IA + "attention")[
  Switch between heads and see each one attend to a *different* relation on the same sentence — then watch their outputs concatenate and mix through $W_O$.
]

// ═══════════════════════════ PART VI — Positional information ═══════════════════════════
= Positional information

== Self-attention forgets order #D

Attention is a *weighted sum* — it depends on *content*, not position. Shuffle the tokens and:
#pause
$ "SelfAttention"(mat(x_1; x_2; x_3)) quad #text(fill: RED)[is a permutation of] quad "SelfAttention"(mat(x_2; x_1; x_3)) $
#pause
#alertbox[Self-attention is *permutation-equivariant*: with no position signal, the model cannot tell "dog bites man" from "man bites dog". Order is invisible.]
#pause
#result[we must *inject* position information into the token representations]

== Adding a positional encoding #D

Give every position $t$ a vector $p_t$ and *add* it to the token embedding:
#pause
$ r_t = e_t + p_t quad quad #text(size: 15pt, fill: MUTED)[(token content $+$ where it sits)] $
#pause
#notebox[Now two identical tokens at different positions get *different* inputs $r_t$ — attention can use the position signal, and permuting the sequence changes the output.]

== Learned positional embeddings #D

The simplest choice: a *lookup table* of position vectors, trained like any embedding:
#pause
$ P in RR^(T_"max" times d), quad quad p_t = P_(t, :) $
#pause
#two(
  notebox[*Pro* — simple, flexible, learns whatever the data needs (GPT / BERT use this).],
  alertbox[*Con* — undefined for $t > T_"max"$; cannot extrapolate to sequences longer than seen in training.],
)

== Sinusoidal positional encoding #D

The original Transformer uses a *fixed*, deterministic pattern of sines and cosines:
#pause
$ "PE"_(("pos", 2i)) = sin("pos" / 10000^(2i \/ d)) $
#pause
$ "PE"_(("pos", 2i+1)) = cos("pos" / 10000^(2i \/ d)) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[dimension $2i$ oscillates at wavelength $2 pi dot 10000^(2i \/ d)$ — no parameters, defined for *any* position])

== Many frequencies, many scales #V

Low dimensions oscillate *fast*, high dimensions *slow* — position encoded at every scale:
#pause
#fig("/lecture15/figures/pe_waves.svg", w: 74%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[left: sinusoids at different frequencies · right: $P P^top$ — nearby positions have *similar* encodings])

== Worked position vector #D

Take $d = 4$, so $i in {0, 1}$ and the exponents are $10000^0 = 1$ and $10000^(1\/2) = 100$:
#pause
$ "PE"("pos") = mat(sin("pos"/1), thin cos("pos"/1), thin sin("pos"/100), thin cos("pos"/100)) $
#pause
At $"pos" = 0$ every sine is $0$ and every cosine is $1$:
$ "PE"(0) = mat(0, 1, 0, 1) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[a unique, smooth signature per position — added onto the token embedding])

== Interactive: positional encoding #I

#interbox(link-to: IA + "positional-encoding")[
  Slide the position and watch each sinusoidal dimension turn; compare learned vs sinusoidal encodings and see how nearby positions stay similar while distant ones drift apart.
]

// ═══════════════════════════ PART VII — The Transformer block ═══════════════════════════
= The Transformer block

== The components #D

One block stacks two sublayers, each wrapped in a residual connection and a normalization:
#pause
#align(center, text(size: 17pt)[
  multi-head self-attention $-> $ residual $+$ norm $-> $ FFN $-> $ residual $+$ norm
])
#pause
#notebox[*Attention* mixes information *across positions*; the *FFN* then processes *each position* on its own. Communication, then computation.]

== The feed-forward network #D

Applied *independently and identically* to every position — a two-layer MLP:
#pause
$ "FFN"(x) = W_2 thin phi(W_1 x + b_1) + b_2 $
#pause
- same weights $W_1, W_2$ at *every* position (a $1 times 1$ transform over the sequence);
- original sizing: $d_"model" = 512$ expands to $d_"ff" = 2048$, then back to $512$.
#pause
#align(center, text(size: 16pt, fill: MUTED)[$phi$ is usually ReLU or GELU; the hidden layer is $4 times$ wider than the model])

== Communication, then computation #D

Why an FFN *after* attention? The two sublayers do *complementary* jobs:
#pause
#two(
  notebox[*Attention* = *communication*. Tokens exchange information — each position gathers from others.],
  notebox[*FFN* = *computation*. Each token, now informed, is transformed on its own — no mixing.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[attention decides *what to gather*; the FFN decides *what to do* with it, tokenwise])

== Residuals and normalization #D

Each sublayer is wrapped so gradients flow and activations stay well-scaled:
#pause
$ #text(fill: TEAL)[*post-norm*]: quad x' = "LN"(x + "MHA"(x)), quad y = "LN"(x' + "FFN"(x')) $
#pause
$ #text(fill: ACC)[*pre-norm*]: quad x' = x + "MHA"("LN"(x)), quad y = x' + "FFN"("LN"(x')) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[*pre-norm* keeps a clean residual highway — more stable for *deep* stacks (the modern default)])

== The Transformer encoder block #V

#encblock
#pause
#align(center, text(size: 14pt, fill: MUTED)[the residual stream runs straight through; each sublayer *branches off, computes, and adds back*])

== The decoder block #D

A full encoder–decoder decoder layer has *three* sublayers:
#pause
+ *masked* self-attention over the output so far (causal);
+ *cross-attention* — queries from the decoder, keys and values from the *encoder*;
+ a position-wise FFN.
#pause
#align(center, text(size: 16pt, fill: MUTED)[cross-attention is how the decoder reads the source — we unpack encoder–decoder next lecture])

== The decoder-only (GPT-style) block #V

Drop the encoder and cross-attention: *causal self-attention $+$ FFN*, then an LM head:
#pause
#decblock
#pause
#align(center, text(size: 14pt, fill: MUTED)[output $p(x_(t+1) | x_(<= t)) = "softmax"(W_o thin h_t)$ — this is the GPT family])

== Interactive: the Transformer block #I

#interbox[#text(fill: MUTED)[(to build)] — send a sequence through one block and watch the residual stream: attention mixes across positions, the FFN transforms each, and the norms keep it stable.]

// ═══════════════════════════ PART VIII — Complexity & comparison ═══════════════════════════
= Complexity and comparison

== The cost of self-attention #D

The attention matrix is $T times T$ — quadratic in sequence length:
#pause
$ Q K^top: quad O(T^2 d) #text(size: 15pt)[ compute], quad quad A in RR^(T times T): quad O(T^2) #text(size: 15pt)[ memory] $
#pause
#alertbox[Every token attends to every token, so cost grows with $T^2$. This is the *long-context bottleneck* — the price of full-range, content-dependent mixing.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[fine for $T ~ 10^3$; the target of much later work (efficient / sparse attention) for very long $T$])

== Path length: how far is information? #D

How many steps must a signal travel between positions $i$ and $j$?
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, center),
  table.header([*Model*], [*Path length $i -> j$*]),
  [RNN], [$O(|i - j|)$ — along the chain],
  [dilated CNN], [$O(log |i - j|)$ — up the dilation tree],
  [self-attention], [$O(1)$ — one hop, any distance],
))
#pause
#result[short paths $=$ easy gradient flow $=$ long-range dependencies are *learnable*]

== Model comparison #V

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 6pt), align: (left, center, center, center),
  table.header([], [*RNN*], [*TCN (CNN)*], [*Transformer*]),
  [parallel over time], [#text(fill: RED)[low]], [#text(fill: GREEN)[high]], [#text(fill: GREEN)[high]],
  [longest path $i -> j$], [$O(T)$], [$O(log T)$], [$O(1)$],
  [compute per layer], [$O(T d^2)$], [$O(T k d^2)$], [$O(T^2 d)$],
  [context], [state (decays)], [fixed field $k$], [content-dependent, full],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[the Transformer trades $O(T^2)$ cost for *parallelism* and *constant* path length])

== Why Transformers won #D

#pause
- *parallel training* — no sequential dependency, the whole sequence at once;
#pause
- *dynamic context* — content-dependent attention, not a fixed window;
#pause
- *short paths* — $O(1)$ between any two positions, so long-range signals survive;
#pause
- *stackable* — residual $+$ norm blocks compose into very deep, scalable models.
#pause
#result[the same architecture scales from a toy LM to the largest models trained today]

// ═══════════════════════════ PART IX — Worked mini decoder-only LM ═══════════════════════════
= A tiny decoder-only language model

== The whole pipeline #V

A minimal GPT: turn `<BOS> I like` into a prediction of the next token:
#pause
#tinylm
#pause
#align(center, text(size: 14pt, fill: MUTED)[token ids $-> $ embeddings $+$ position $-> $ causal self-attention $-> $ FFN $-> $ logits $-> $ softmax])

== The training loss #D

The logits at position $t$ predict the token at $t+1$ — the *same next-token objective* as before:
#pause
$ cal(L) = - sum_(t=1)^(T-1) log p_theta (x_(t+1) | x_(<= t)) $
#pause
#notebox[Nothing new in the loss — token-level cross-entropy. What changed is *how* $p_theta$ is computed: causal self-attention over the prefix, not a recurrent state.]

== Shifted targets #D

Feed the sequence in; the target is the *same sequence shifted left by one*:
#pause
#align(center, table(
  columns: 5, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (center, center, center, center, center),
  table.header([position $t$], [1], [2], [3], [4]),
  [input $x_t$], [`<BOS>`], [`I`], [`like`], [`cats`],
  [target $x_(t+1)$], [`I`], [`like`], [`cats`], [`<EOS>`],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[the *causal mask* is what makes this safe — position $t$ cannot see its own target $x_(t+1)$])

== Interactive: tiny Transformer LM #I

#interbox[#text(fill: MUTED)[(to build)] — type a short prefix and watch it flow through embeddings, causal attention and the FFN to a next-token distribution; sample a token and feed it back.]

// ═══════════════════════════ PART X — Summary ═══════════════════════════
= Summary

== What changed from RNNs #D

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 8pt), align: (left, left),
  table.header([*RNN*], [*Transformer*]),
  [$h_t = f(h_(t-1), x_t)$], [$H' = "SelfAttention"(H)$],
  [memory carried through *state*], [interactions computed *directly*],
  [serial over time], [parallel over positions],
  [path $i -> j$ is $O(|i-j|)$], [path $i -> j$ is $O(1)$],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[recurrence *routes* information step by step; attention *relates* all positions at once])

== The key equations #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left),
  table.header([*Step*], [*Equation*]),
  [project], [$Q = X W_Q, quad K = X W_K, quad V = X W_V$],
  [attend], [$A = "softmax"(Q K^top \/ sqrt(d_k) + "mask")$],
  [gather], [$O = A V$],
  [multi-head], [$"MHA" = mat("head"_1; dots.v; "head"_H) W_O$],
  [block], [$"MHSA" -> $ residual $+$ norm $-> $ FFN $-> $ residual $+$ norm],
))

== Final mental model — one idea

#align(center, text(size: 22pt)[
  *Self-attention*: every token asks the others for information.
  #v(6pt)
  *Multi-head*: several relation types, in parallel.
  #v(6pt)
  *Positional encoding*: puts the order back in.
  #v(6pt)
  #text(size: 18pt, fill: MUTED)[a Transformer block $=$ attention $+$ tokenwise MLP $+$ residual / norm]
])

#focus-slide[
  We replaced recurrence with *attention between all tokens*.
  #v(12pt)
  #set text(size: 22pt)
  Next — *Transformers II*: encoder-only (BERT), decoder-only (GPT), encoder–decoder; pretraining, finetuning and generation.
]
