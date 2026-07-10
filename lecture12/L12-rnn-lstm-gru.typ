// Sequence Models I: RNNs, BPTT, LSTMs and GRUs — Lecture 12 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture12/L12-rnn-lstm-gru.typ /tmp/L12.pdf
//   typst compile --root . --input handout=true lecture12/L12-rnn-lstm-gru.typ /tmp/L12h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [Sequence Models I: RNNs, BPTT, LSTMs and GRUs],
  subtitle: [Remembering with a hidden state — and gating what to keep],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let cream = rgb("#EFEEEB")

// ═══════════════════════════ fletcher diagrams (centerpieces) ═══════════════════════════

// ── (A) the RNN cell: h_{t-1} and x_t → Σ → φ → h_t, with a recurrent delay ──
#let rnncell = align(center, diagram(spacing: (16mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), $h_(t-1)$, radius: 7mm, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((1.7, 0), [$Sigma$], radius: 6.5mm, fill: cream)
  node((3.1, 0), [$phi$], radius: 6.5mm, fill: rgb("#FDECD6"), stroke: 0.9pt + ACC)
  node((4.7, 0), $h_t$, radius: 7mm, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((1.7, -1.5), $x_t$, radius: 7mm, fill: cream)
  edge((0,0),(1.7,0), "-|>", label: text(size: 12pt, fill: TEAL)[$W_(h h)$], stroke: 0.9pt + INK)
  edge((1.7,-1.5),(1.7,0), "-|>", label: text(size: 12pt, fill: BLUE)[$W_(x h)$], label-side: right, stroke: 0.9pt + INK)
  edge((1.7,0),(3.1,0), "-|>", label: text(size: 11pt, fill: MUTED)[$a_t$], stroke: 0.9pt + INK)
  edge((3.1,0),(4.7,0), "-|>", stroke: 0.9pt + INK)
  // recurrent delay: h_t feeds the next step's h_{t-1}
  edge((4.7,0),(4.7,1.35),(0,1.35),(0,0), "-|>", stroke: (paint: MUTED, dash: "dashed", thickness: 1pt), label: text(size: 11pt, fill: MUTED)[delay — becomes next $h_(t-1)$])
}))

// ── (B) the unrolled RNN: shared cell applied along time, outputs on top ──
#let unrolled = align(center, diagram(spacing: (15mm, 10mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((-0.35, 0), $h_0$, radius: 6mm, fill: cream)
  for i in (1, 2, 3, 4) {
    node((i, 0), [RNN], fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL, shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 7pt)
    node((i, -1.5), [$x_#i$], radius: 6mm, fill: cream)
    node((i, 1.5), [$hat(y)_#i$], radius: 6mm, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  }
  // inputs up, outputs up
  edge((1,-1.5),(1,0), "-|>", label: text(size: 11pt, fill: BLUE)[$W_(x h)$], label-side: right, stroke: 0.8pt + MUTED)
  for i in (2, 3, 4) { edge((i,-1.5),(i,0), "-|>", stroke: 0.8pt + MUTED) }
  edge((1,0),(1,1.5), "-|>", label: text(size: 11pt, fill: ACC)[$W_(h y)$], label-side: right, stroke: 0.8pt + MUTED)
  for i in (2, 3, 4) { edge((i,0),(i,1.5), "-|>", stroke: 0.8pt + MUTED) }
  // recurrent state left→right, shared W_hh (labelled once on the first state edge)
  edge((-0.35,0),(1,0), "-|>", label: text(size: 11pt, fill: TEAL)[$W_(h h)$], label-side: left, stroke: 1pt + TEAL)
  edge((1,0),(2,0), "-|>", label: text(size: 11pt, fill: TEAL)[$h_1$], stroke: 1pt + TEAL)
  edge((2,0),(3,0), "-|>", label: text(size: 11pt, fill: TEAL)[$h_2$], stroke: 1pt + TEAL)
  edge((3,0),(4,0), "-|>", label: text(size: 11pt, fill: TEAL)[$h_3$], stroke: 1pt + TEAL)
  edge((4,0),(5,0), "-|>", label: text(size: 11pt, fill: TEAL)[$h_4$], stroke: 1pt + TEAL)
}))

// ── (C) BPTT: forward chain + backward adjoints, two sources of overline(h)_t ──
#let bpttdiag = align(center, diagram(spacing: (22mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), $h_0$, radius: 6mm, fill: cream)
  for i in (1, 2, 3) {
    node((i, 0), [$h_#i$], radius: 6.5mm, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
    node((i, 1.5), [$cal(L)_#i$], radius: 6.5mm, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  }
  // forward state chain (teal, above the axis)
  for i in (0, 1, 2) { edge((i,0),(i+1,0), "-|>", stroke: 1pt + TEAL) }
  // output path into each hidden state (loss → h_t)  [orange, straight down]
  for i in (1, 2, 3) { edge((i,1.5),(i,0), "-|>", stroke: 1pt + ACC) }
  // future path: adjoint flows right→left, arcing above the nodes (accumulation) — short labels
  edge((2,0),(1,0), "-|>", bend: -40deg, stroke: 1.1pt + RED, label: text(size: 11pt, fill: RED)[$W_(h h)^top overline(a)_2$])
  edge((3,0),(2,0), "-|>", bend: -40deg, stroke: 1.1pt + RED, label: text(size: 11pt, fill: RED)[$W_(h h)^top overline(a)_3$])
}))

// ── (D) LSTM cell — the centerpiece: additive cell highway + 3 gates ──
#let lstmcell = align(center, diagram(spacing: (12mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  // cell-state highway (top, teal, thick)
  node((0, 2.2), $c_(t-1)$, radius: 7mm, fill: TEAL.lighten(82%), stroke: 1pt + TEAL)
  node((1.7, 2.2), text(size: 14pt)[$times$], radius: 5mm, fill: RED.lighten(80%), stroke: 1pt + RED)
  node((3.7, 2.2), text(size: 15pt)[$+$], radius: 5mm, fill: GREEN.lighten(80%), stroke: 1pt + GREEN)
  node((5.4, 2.2), $c_t$, radius: 7mm, fill: TEAL.lighten(82%), stroke: 1pt + TEAL)
  edge((0,2.2),(1.7,2.2), "-|>", stroke: 1.4pt + TEAL)
  edge((1.7,2.2),(3.7,2.2), "-|>", stroke: 1.4pt + TEAL)
  edge((3.7,2.2),(5.4,2.2), "-|>", stroke: 1.4pt + TEAL)
  // forget gate (red) → up into ×
  node((1.7, 0.7), text(size: 13pt)[$sigma$], radius: 5.5mm, fill: RED.lighten(82%), stroke: 0.9pt + RED)
  edge((1.7,0.7),(1.7,2.2), "-|>", stroke: 1pt + RED, label: text(size: 11pt, fill: RED)[$f_t$], label-side: right)
  // write path (green): i and c̃ → × → up into +
  node((3.0, 0.7), text(size: 13pt)[$sigma$], radius: 5.5mm, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  node((4.4, 0.7), text(size: 12pt)[tanh], fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN, shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 4pt)
  node((3.7, 1.45), text(size: 13pt)[$times$], radius: 4.5mm, fill: GREEN.lighten(80%), stroke: 0.9pt + GREEN)
  edge((3.0,0.7),(3.7,1.45), "-|>", stroke: 0.9pt + GREEN, label: text(size: 11pt, fill: GREEN)[$i_t$], label-side: left)
  edge((4.4,0.7),(3.7,1.45), "-|>", stroke: 0.9pt + GREEN, label: text(size: 11pt, fill: GREEN)[$tilde(c)_t$], label-side: right)
  edge((3.7,1.45),(3.7,2.2), "-|>", stroke: 1pt + GREEN)
  // output path (orange): c_t → tanh → × o → h_t
  node((6.6, 1.3), text(size: 12pt)[tanh], fill: ACC.lighten(82%), stroke: 0.9pt + ACC, shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 4pt)
  node((7.8, 1.3), text(size: 13pt)[$times$], radius: 4.5mm, fill: ACC.lighten(80%), stroke: 0.9pt + ACC)
  node((7.8, -0.1), text(size: 13pt)[$sigma$], radius: 5.5mm, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  node((9.0, 1.3), $h_t$, radius: 7mm, fill: BLUE.lighten(82%), stroke: 1pt + BLUE)
  edge((5.4,2.2),(6.6,1.3), "-|>", stroke: 1pt + ACC)
  edge((6.6,1.3),(7.8,1.3), "-|>", stroke: 1pt + ACC)
  edge((7.8,-0.1),(7.8,1.3), "-|>", stroke: 1pt + ACC, label: text(size: 11pt, fill: ACC)[$o_t$], label-side: right)
  edge((7.8,1.3),(9.0,1.3), "-|>", stroke: 1.2pt + BLUE)
  // shared input block feeds all four gates
  node((4.5, -1.5), text(size: 12.5pt)[$u_t = [h_(t-1) space ; space x_t]$], fill: cream, stroke: 0.9pt + INK, shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 6pt)
  edge((4.5,-1.5),(1.7,0.7), "-|>", stroke: 0.6pt + MUTED)
  edge((4.5,-1.5),(3.0,0.7), "-|>", stroke: 0.6pt + MUTED)
  edge((4.5,-1.5),(4.4,0.7), "-|>", stroke: 0.6pt + MUTED)
  edge((4.5,-1.5),(7.8,-0.1), "-|>", stroke: 0.6pt + MUTED)
}))

// legend row for the LSTM paths
#let lstmlegend = align(center, text(size: 13pt)[
  #box(baseline: 2pt, rect(width: 12pt, height: 5pt, fill: RED, stroke: none)) forget #h(1.2em)
  #box(baseline: 2pt, rect(width: 12pt, height: 5pt, fill: GREEN, stroke: none)) write (input) #h(1.2em)
  #box(baseline: 2pt, rect(width: 12pt, height: 5pt, fill: ACC, stroke: none)) output #h(1.2em)
  #box(baseline: 2pt, rect(width: 12pt, height: 5pt, fill: TEAL, stroke: none)) cell-state highway
])

// ── (E) GRU cell — simpler: reset r, update z, convex combine into h ──
#let grucell = align(center, diagram(spacing: (13mm, 10mm), node-stroke: 0.9pt + INK, node-fill: white, {
  // hidden highway
  node((0, 1.7), $h_(t-1)$, radius: 7mm, fill: TEAL.lighten(82%), stroke: 1pt + TEAL)
  node((5.2, 1.7), text(size: 12pt)[$(1-z_t) dot.o h_(t-1)$ \ $+ space z_t dot.o tilde(h)_t$], fill: BLUE.lighten(85%), stroke: 1pt + BLUE, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 6pt)
  node((7.0, 1.7), $h_t$, radius: 7mm, fill: TEAL.lighten(82%), stroke: 1pt + TEAL)
  edge((0,1.7),(5.2,1.7), "-|>", stroke: 1.3pt + TEAL, label: text(size: 11pt, fill: TEAL)[carry $(1-z_t)$], label-side: left)
  edge((5.2,1.7),(7.0,1.7), "-|>", stroke: 1.3pt + TEAL)
  // reset gate & candidate (green write path)
  node((1.7, 0.4), text(size: 13pt)[$sigma$], radius: 5.5mm, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  node((2.9, 0.9), text(size: 13pt)[$times$], radius: 4.5mm, fill: ACC.lighten(80%), stroke: 0.9pt + ACC)
  node((3.9, 0.4), text(size: 12pt)[tanh], fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN, shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 4pt)
  edge((1.7,0.4),(2.9,0.9), "-|>", stroke: 0.9pt + ACC, label: text(size: 11pt, fill: ACC)[$r_t$], label-side: left)
  edge((0,1.7),(2.9,0.9), "-|>", bend: -22deg, stroke: 0.8pt + TEAL)
  edge((2.9,0.9),(3.9,0.4), "-|>", stroke: 0.9pt + GREEN)
  edge((3.9,0.4),(5.2,1.7), "-|>", stroke: 1pt + GREEN, label: text(size: 11pt, fill: GREEN)[$tilde(h)_t$], label-side: right)
  // update gate z controls the combine
  node((5.2, 0.4), text(size: 13pt)[$sigma$], radius: 5.5mm, fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE)
  edge((5.2,0.4),(5.2,1.7), "-|>", stroke: 0.9pt + BLUE, label: text(size: 11pt, fill: BLUE)[$z_t$], label-side: right)
  // shared input x_t feeds gates & candidate
  node((3.4, -1.1), $x_t$, radius: 6mm, fill: cream)
  edge((3.4,-1.1),(1.7,0.4), "-|>", stroke: 0.6pt + MUTED)
  edge((3.4,-1.1),(3.9,0.4), "-|>", stroke: 0.6pt + MUTED)
  edge((3.4,-1.1),(5.2,0.4), "-|>", stroke: 0.6pt + MUTED)
}))

// ── (F) fixed window (MLP) vs recurrent (RNN) ──
#let fixedvsrec = align(center, diagram(spacing: (11mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  // fixed window (left): concat last k tokens into one MLP
  node((1.4, 2.0), text(size: 13pt, fill: INK)[*fixed window (MLP-LM)*], stroke: none, fill: none)
  for (i, l) in ((0, $x_(t-3)$), (1, $x_(t-2)$), (2, $x_(t-1)$)) {
    node((i*1.4, 0.5), text(size: 11pt, l), fill: cream, shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 5pt)
    edge((i*1.4, 0.5), (1.4, -0.7), "-|>", stroke: 0.6pt + MUTED)
  }
  node((1.4, -0.7), text(size: 11pt)[MLP], fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE, shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 5pt)
  node((1.4, -1.7), text(size: 10.5pt, fill: MUTED)[window $k=3$ only], stroke: none, fill: none)
  // recurrent (right): state carries all the past
  node((6.85, 2.0), text(size: 13pt, fill: INK)[*recurrent (RNN)*], stroke: none, fill: none)
  for (i, l) in ((0, $x_1$), (1, $x_2$), (2, $x_3$), (3, $x_4$)) {
    node((5.7 + i*0.9, -0.7), text(size: 11pt, l), radius: 5.5mm, fill: cream)
    node((5.7 + i*0.9, 0.6), text(size: 11pt, fill: white, $h_#(i+1)$), radius: 5.5mm, fill: TEAL, stroke: 0.9pt + TEAL)
    edge((5.7 + i*0.9, -0.7), (5.7 + i*0.9, 0.6), "-|>", stroke: 0.6pt + MUTED)
  }
  for i in range(3) { edge((5.7 + i*0.9, 0.6), (5.7 + (i+1)*0.9, 0.6), "-|>", stroke: 1pt + TEAL) }
  node((7.05, 1.5), text(size: 10.5pt, fill: TEAL)[state carries all of the past], stroke: none, fill: none)
}))

#title-slide()

// ═══════════════════════════ PART I — Motivation ═══════════════════════════
= From fixed windows to recurrent state

== Where we are

Last lecture built a language model over a *fixed window*: predict the next token from the previous $k$.
#pause
$ p_theta (x_(t+1) | x_(<t)) approx p_theta (x_(t+1) | x_(t-k+1 : t)) $
#pause
#notebox[This lecture keeps the same *task* — next-token prediction — but changes *how the model remembers the past*: from a fixed window of inputs to a *running hidden state*.]

== Recall: modeling a sequence #D

A language model factorizes the joint probability by the chain rule:
#pause
$ p(x_(1:T)) = product_(t=1)^T p(x_t | x_(<t)) $
#pause
and we train it by minimizing the negative log-likelihood — token-level cross-entropy:
$ cal(L) = - sum_(t=1)^T log p_theta (x_t | x_(<t)) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[each factor is one $V$-way classification: "which token comes next?"])

== The fixed-window MLP, and its limits

The MLP-LM concatenates the last $k$ embeddings and runs one MLP:
#pause
#two(
  notebox[*fixed context* — anything before $x_(t-k)$ is invisible; long-range dependencies are lost.],
  notebox[*params grow with $k$* — $W_h in RR^(H times k d)$; doubling context doubles the layer.],
)
#pause
#alertbox[*No shared state across time.* The same token at position 1 vs 2 hits *different* columns of $W_h$ — computation is not reused, and the input length is frozen at $k$.]

== What we actually want

A better sequence model should:
#pause
- read *one token at a time*, for a sequence of *any length*;
#pause
- keep a *memory* $h_t$ — a summary of everything seen so far, $x_(1:t)$;
#pause
- *predict* from that memory, $hat(y)_t = g(h_t)$;
#pause
- *share* the same parameters at every step.
#pause
#result[a running state updated by *one* function: $h_t = f_theta (h_(t-1), x_t)$]

== The recurrent idea

Replace the fixed window with a *state* that is updated at every step:
#pause
$ underbrace(p(x_(t+1) | x_(t-k:t)), "MLP: fixed window") quad --> quad underbrace(h_t = f_theta (h_(t-1), x_t), "RNN: recurrent state") $
#pause
One *shared* function $f_theta$ applied over and over gives us, for free:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*Property*], [*Why it follows*]),
  [variable length], [same $f_theta$ runs for as many steps as there are tokens],
  [parameter sharing], [one $f_theta$, not one weight block per position],
  [learned state], [$h_t$ is a trainable summary of $x_(1:t)$],
  [online], [update as each token arrives — no need to see the whole sequence],
))

== Fixed window vs recurrent state #V

#fixedvsrec
#pause
#align(center, text(size: 16pt, fill: MUTED)[the MLP forgets past the window; the RNN *carries* the past forward in $h_t$])

// ═══════════════════════════ PART II — The vanilla RNN ═══════════════════════════
= The vanilla (Elman) RNN

== The equations #D

At each step $t$: look up the token, update the state, read out a prediction.
#pause
$ e_t = E[x_t] quad quad #text(size: 15pt, fill: MUTED)[(embedding lookup)] $
#pause
$ a_t = W_(x h) thin e_t + W_(h h) thin h_(t-1) + b_h, quad quad h_t = phi(a_t) $
#pause
$ z_t = W_(h y) thin h_t + b_y, quad quad p_t = "softmax"(z_t) $
#pause
#align(center, text(size: 15pt, fill: MUTED)[$phi = tanh$ usually; one set of weights $W_(x h), W_(h h), W_(h y)$ reused at *every* step (Elman)])

== The RNN cell #V

One shared cell: the past state $h_(t-1)$ and the new input $x_t$ meet, then $phi$ gives $h_t$.
#pause
#rnncell
#pause
#align(center, text(size: 15pt, fill: MUTED)[$h_t = phi(W_(x h) e_t + W_(h h) h_(t-1) + b_h)$ — the dashed edge is the *recurrence*])

== Unrolling the RNN in time #V

The *same* cell, drawn once per step — this is what backprop actually sees:
#pause
#unrolled
#pause
#align(center, text(size: 15pt, fill: MUTED)[$W_(x h), W_(h h), W_(h y)$ are *shared* across all steps — the diagram repeats one cell])

== The hidden state is memory #D

Unroll the recurrence and the whole past is nested inside $h_t$:
#pause
$ h_t = f(h_(t-1), x_t) = f(f(h_(t-2), x_(t-1)), x_t) = dots $
#pause
$ h_t = f(f(dots f(h_0, x_1) dots, x_(t-1)), x_t) $
#pause
#notebox[$h_t$ is a *learned, fixed-size summary* of the entire prefix $x_(1:t)$ — no matter how long. The window is gone; the state remembers.]

== RNN for language modeling #D

Feed a sentence in; the *target at each step is the next token* (teacher forcing):
#pause
#align(center, table(
  columns: 5, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, center, center, center, center),
  table.header([step $t$], [1], [2], [3], [4]),
  [input $x_t$], [`deep`], [`learning`], [`is`], [`fun`],
  [state], [$h_1$], [$h_2$], [$h_3$], [$h_4$],
  [target $x_(t+1)$], [`learning`], [`is`], [`fun`], [`<eos>`],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[loss $= sum_t$ cross-entropy$(p_t, x_(t+1))$ — every position supervises the step before it])

== RNN task patterns #V

The same cell, wired differently, covers most sequence tasks:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 6pt), align: (left, center, left),
  table.header([*Pattern*], [*Shape*], [*Example*]),
  [many-to-one], [$x_(1:T) -> y$], [sentiment: read all, predict at $h_T$],
  [many-to-many (aligned)], [$x_(1:T) -> y_(1:T)$], [POS tag each token],
  [language model], [$x_(1:T) -> x_(2:T+1)$], [next-token at every step],
  [seq2seq (encoder–decoder)], [$x_(1:T) -> y_(1:U)$], [translation — different length],
))

== Tensor shapes

Track a batch of $B$ sequences, length $T$, embedding $d$, hidden $m$, vocab $V$:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, center, left),
  table.header([*Tensor*], [*Shape*], [*Meaning*]),
  [token ids $X$], [$B times T$], [one id per position],
  [embeddings $E[X]$], [$B times T times d$], [one $d$-vector per token],
  [hidden states $H$], [$B times T times m$], [the state at every step],
  [logits $Z$], [$B times T times V$], [a next-token score at every step],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[time is now an *axis*, not a fixed slot — the model handles any $T$])

// ═══════════════════════════ PART III — Fully-worked forward ═══════════════════════════
= A fully-worked RNN forward pass

== The tiny scalar RNN

Strip it to *one* hidden unit — everything is a scalar so we can do it by hand:
#pause
$ h_t = tanh(w_x thin x_t + w_h thin h_(t-1) + b) $
#pause
$ w_x = 1, quad w_h = 0.5, quad b = 0, quad h_0 = 0, quad x = (1, 2, -1) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[three steps, three states — watch the memory build])

== Step 1 #D

$ a_1 = w_x x_1 + w_h h_0 + b = 1 dot 1 + 0.5 dot 0 = 1 $
#pause
$ h_1 = tanh(1) approx 0.762 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[with no past state ($h_0 = 0$), the first step is just the input through $tanh$])

== Step 2 #D

Now the past enters through $w_h h_1$:
#pause
$ a_2 = w_x x_2 + w_h h_1 = 1 dot 2 + 0.5 dot 0.762 = 2.381 $
#pause
$ h_2 = tanh(2.381) approx 0.983 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the state is being pushed toward the $+1$ saturation of $tanh$])

== Step 3 #D

A negative input, but the state still remembers the positive past:
#pause
$ a_3 = w_x x_3 + w_h h_2 = 1 dot (-1) + 0.5 dot 0.983 = -0.509 $
#pause
$ h_3 = tanh(-0.509) approx -0.469 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the $+0.49$ carried from $h_2$ *softened* the $-1$ input — that is memory at work])

== The forward trace #V

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 16pt, y: 7pt), align: (center, center, center, center),
  table.header([$t$], [$x_t$], [$a_t = x_t + 0.5 h_(t-1)$], [$h_t = tanh(a_t)$]),
  [1], [$+1$], [$1.000$], [$0.762$],
  [2], [$+2$], [$2.381$], [$0.983$],
  [3], [$-1$], [$-0.509$], [$-0.469$],
))
#pause
#fig("/lecture12/figures/rnn_forward.svg", w: 52%)

== Why this is memory #D

Substitute backwards — $h_3$ depends on *every* earlier input:
#pause
$ h_3 = tanh(x_3 + 0.5 thin underbrace(tanh(x_2 + 0.5 thin underbrace(tanh(x_1), h_1)), h_2)) $
#pause
#notebox[$x_1$ reaches $h_3$ through *two* nested $tanh$ and two factors of $w_h = 0.5$. Information persists — but each step *attenuates* it by $w_h$. Hold that thought (Part V).]

== Interactive: RNN forward #I

#interbox[#text(fill: MUTED)[(to build)] — set $w_x, w_h, b$ and type a short sequence; watch $h_t$ update step by step and the state trajectory trace out, exactly like the scalar example above.]

// ═══════════════════════════ PART IV — Backpropagation through time ═══════════════════════════
= Backpropagation through time

== An RNN is a deep net in time #D

Unrolled, the RNN is just a *very deep* feedforward net whose layers *share weights*:
#pause
$ h_0 -> h_1 -> h_2 -> dots -> h_T $
#pause
#result[training it = backprop on the unrolled graph = *backpropagation through time* (BPTT)]
#pause
#align(center, text(size: 16pt, fill: MUTED)[same chain rule as Lecture 3 — only now the "depth" is the sequence length $T$])

== The loss sums over time #D

Every step contributes a term; the total loss is their sum:
#pause
$ cal(L) = sum_(t=1)^T cal(L)_t, quad quad cal(L)_t = - log p_theta (x_(t+1) | x_(<= t)) $
#pause
Because the weights are *shared*, each step's gradient lands on the *same* parameters:
$ nabla_(W_(h h)) cal(L) = sum_(t=1)^T nabla_(W_(h h)) cal(L)_t $
#pause
#align(center, text(size: 16pt, fill: MUTED)[this is exactly Lecture 3's rule: a shared parameter *accumulates* gradient from every use])

== The backward recurrence #D

Write $overline(u) = partial cal(L) \/ partial u$. Walking one step back through the cell:
#pause
$ overline(a)_t = overline(h)_t dot.o phi'(a_t) quad quad #text(size: 14pt, fill: MUTED)[(through the $tanh$)] $
#pause
$ nabla_(W_(x h)) thin "+="thin overline(a)_t thin e_t^top, quad quad nabla_(W_(h h)) thin "+="thin overline(a)_t thin h_(t-1)^top $
#pause
$ overline(h)_(t-1) thin "+="thin W_(h h)^top thin overline(a)_t quad quad #text(size: 14pt, fill: MUTED)[(state gradient to the previous step)] $

== Two sources of $overline(h)_t$ #D

The state $h_t$ is used *twice* — so its gradient is a *sum* of two paths:
#pause
$ overline(h)_t = underbrace((partial cal(L)_t)/(partial h_t), "output path: its own loss") + underbrace((partial cal(L)_(>= t+1))/(partial h_t), "future path: W_(h h)^top overline(a)_(t+1)") $
#pause
#notebox[Exactly the *branch rule*: $h_t$ feeds both this step's readout *and* the next step's state, so gradients from both add. Miss the second term and long-range learning is impossible.]

== BPTT, unrolled #V

Forward state (teal) flows right; the adjoint (orange = own loss, red = future) flows *back*:
#pause
#bpttdiag
#pause
#align(center, text(size: 15pt, fill: MUTED)[each $overline(h)_t$ collects its own loss #text(fill: ACC)[(↓)] plus the future via $W_(h h)^top$ #text(fill: RED)[(←)]])

== Truncated BPTT #D

Full BPTT over a length-$10^4$ sequence is expensive and unstable. So *chunk* it:
#pause
- process the sequence in blocks of length $K$ (say $K = 35$);
- backprop *within* a block; then *carry the state but detach it*:
#pause
$ h_"start"^"(next block)" = "detach"(h_"end"^"(this block)") $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the state keeps flowing forward; the *gradient* stops at the block boundary — bounded cost])

== Interactive: BPTT #I

#interbox[#text(fill: MUTED)[(to build)] — step through the unrolled graph and watch each $overline(h)_t$ accumulate its two contributions; toggle truncation length $K$ and see where the gradient is cut.]

// ═══════════════════════════ PART V — Vanishing / exploding gradients ═══════════════════════════
= Vanishing and exploding gradients

== The long-range gradient is a product #D

To learn a dependency from step $k$ to step $T$, the gradient runs the whole chain:
#pause
$ (partial h_T)/(partial h_k) = product_(j=k+1)^T (partial h_j)/(partial h_(j-1)), quad quad (partial h_j)/(partial h_(j-1)) = D_j thin W_(h h) $
#pause
where $D_j = "diag"(phi'(a_j))$ is the activation slope at step $j$.
#pause
#result[a product of $(T - k)$ matrices — it tends to *vanish* or *explode*]

== The scalar recurrence lays it bare #D

Take one unit, $h_t = tanh(w thin h_(t-1))$. Near $0$, $tanh'(0) = 1$, so:
#pause
$ (partial h_T)/(partial h_0) approx w^T $
#pause
#two(
  alertbox[$|w| < 1$: #h(0.3em) $w^T -> 0$ — *vanishing*. The distant past cannot send a gradient.],
  alertbox[$|w| > 1$: #h(0.3em) $w^T -> infinity$ — *exploding*. Gradients blow up, training diverges.],
)

== Worked: $w^50$ #D

Fifty steps back, with two nearby weights:
#pause
$ w = 0.9: quad 0.9^50 approx 0.005 quad #text(fill: RED)[(vanished)] $
#pause
$ w = 1.1: quad 1.1^50 approx 117 quad #text(fill: RED)[(exploded)] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[a $10%$ change in $w$ flips the gradient from $1/200$ to $times 117$ — over just 50 steps])

== The saturation problem too #V

$tanh'(z) = 1 - tanh^2(z) <= 1$, and $-> 0$ once the unit saturates:
#pause
#fig("/lecture12/figures/tanh_sat.svg", w: 46%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[each $D_j <= 1$ can only *shrink* the signal further — saturation makes vanishing worse])

== Gradient norm vs time-lag #V

#fig("/lecture12/figures/grad_vs_lag.svg", w: 60%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[vanish (decay), explode (blow up), or — with a *gate* holding $f_t approx 1$ — stay *flat*])

== Gradient clipping #D

Explosion has a cheap fix: cap the global gradient *norm* before the update.
#pause
$ "if" norm(g) > c: quad g <- c thin g / norm(g) $
#pause
#notebox[Rescale, don't truncate — the *direction* of $g$ is preserved, only its *length* is capped at $c$ (Pascanu, Mikolov & Bengio 2013).]

== Worked: clipping #D

Threshold $c = 5$, gradient $g = (6, 8)$:
#pause
$ norm(g) = sqrt(6^2 + 8^2) = 10 > 5 $
#pause
$ g <- 5 dot (6, 8) / 10 = (3, 4), quad norm((3,4)) = 5 quad checkmark $
#pause
#align(center, text(size: 16pt, fill: MUTED)[same direction $(3,4) parallel (6,8)$, length now exactly $5$])

== Clipping is only half the cure

#pause
#two(
  notebox[*Fixes exploding* — caps the step size so a single huge gradient cannot blow up training.],
  alertbox[*Does nothing for vanishing* — you cannot rescale a gradient that has already decayed to $approx 0$.],
)
#pause
#result[to carry gradients across long lags we must change the *architecture* → gates]

== Interactive: gradient dynamics #I

#interbox(link-to: IA + "vanishing-gradients")[
  Sweep the recurrent weight $w$ and the lag $T$; watch the long-range gradient collapse for $|w| < 1$, blow up for $|w| > 1$, and stay flat once a gate holds the memory open.
]

// ═══════════════════════════ PART VI — LSTM ═══════════════════════════
= The LSTM

== The core problem #D

The vanilla RNN *overwrites* its state every step: $h_t = phi(W_(h h) h_(t-1) + dots)$.
#pause
#alertbox[Every step multiplies the old state by $W_(h h)$ and squashes it — memory is *rewritten*, and (Part V) the gradient is *multiplied* away. We want to *choose* what to keep.]
#pause
#result[let the network *decide*, per component: what to *forget*, what to *write*, what to *expose*]

== Two states, one additive highway #D

The LSTM (Hochreiter & Schmidhuber, 1997) carries *two* vectors:
#pause
#two(
  notebox[*cell state* $c_t$ — the long-term memory (the "highway").],
  notebox[*hidden state* $h_t$ — the exposed output, a filtered view of $c_t$.],
)
#pause
The cell updates by *addition*, not full overwrite:
$ c_t = f_t dot.o c_(t-1) + i_t dot.o tilde(c)_t $
#pause
#align(center, text(size: 16pt, fill: MUTED)[keep a gated fraction of the old memory, *add* a gated bit of new — this is the whole idea])

== Gates are learned $[0, 1]$ valves #D

A *gate* is a vector in $[0,1]$, computed by a sigmoid from the input and previous state:
#pause
$ "gate" = sigma(W thin u_t + b) in (0, 1)^m, quad quad u_t = [h_(t-1) space ; space x_t] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[$0 =$ block the channel, $1 =$ let it pass — applied *elementwise* with $dot.o$])

== The LSTM equations #D

Shared input $u_t = [h_(t-1); x_t]$; three sigmoid gates and one $tanh$ candidate:
#pause
$ f_t = sigma(W_f u_t + b_f), quad i_t = sigma(W_i u_t + b_i), quad o_t = sigma(W_o u_t + b_o) $
#pause
$ tilde(c)_t = tanh(W_c u_t + b_c) quad quad #text(size: 14pt, fill: MUTED)[(candidate memory)] $
#pause
$ c_t = f_t dot.o c_(t-1) + i_t dot.o tilde(c)_t, quad quad h_t = o_t dot.o tanh(c_t) $

== The LSTM cell #V

#lstmcell
#v(2pt)
#lstmlegend
#pause
#align(center, text(size: 14pt, fill: MUTED)[cell state runs straight across the top; gates tap in to forget, write, and read out])

== Reading the three gates #D

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 6pt), align: (left, center, left),
  table.header([*Gate*], [*Operation*], [*Meaning*]),
  [forget $f_t$], [$f_t dot.o c_(t-1)$], [$1$ = keep this memory, $0$ = erase it],
  [input $i_t$], [$i_t dot.o tilde(c)_t$], [how much of the new candidate to *write*],
  [output $o_t$], [$o_t dot.o tanh(c_t)$], [how much of the memory to *expose* as $h_t$],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[candidate $tilde(c)_t = tanh(dots)$ proposes *what* to write; the gates decide *how much*])

== Why additive memory helps #D

The cell state's step-to-step Jacobian is just the *forget gate* — no weight matrix:
#pause
$ (partial c_t)/(partial c_(t-1)) = "diag"(f_t) $
#pause
So over many steps the gradient along the cell highway is a product of *gates*:
$ (partial c_T)/(partial c_k) = product_(j=k+1)^T "diag"(f_j) $
#pause
#result[if $f_j approx 1$, the gradient flows *undamped* — no repeated $times W_(h h)$]
#pause
#align(center, text(size: 15pt, fill: MUTED)[the central LSTM advantage: an *open* forget gate is a gradient highway across time])

== Worked scalar LSTM #D

One cell, old memory $c_(t-1) = 5$. *Retain* case — forget open, input nearly shut:
#pause
$ f_t = 0.8, thick i_t = 0.2, thick tilde(c)_t = 3: quad c_t = 0.8 dot 5 + 0.2 dot 3 = 4.6 $
#pause
*Replace* case — forget nearly shut, input open, candidate negative:
$ f_t = 0.1, thick i_t = 0.9, thick tilde(c)_t = -2: quad c_t = 0.1 dot 5 + 0.9 dot (-2) = -1.3 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[same equation: gates near $(1, 0)$ *hold* memory ($5 -> 4.6$); near $(0, 1)$ *overwrite* it ($5 -> -1.3$)])

== Interactive: the LSTM cell #I

#interbox(link-to: IA + "lstm-gates")[
  Drag the forget, input and output gates and watch the cell state $c_t$ retain, overwrite, or leak — and see the gradient highway open and close as $f_t$ moves between $0$ and $1$.
]

// ═══════════════════════════ PART VII — GRU ═══════════════════════════
= The GRU

== A simpler gated cell

The LSTM works but has *three* gates, a candidate, and *two* states. Can we do less?
#pause
#notebox[The GRU (Cho et al. 2014, in the RNN encoder–decoder for translation) keeps the gating idea but uses *one* state and *two* gates.]

== The GRU equations #D

Update gate $z_t$, reset gate $r_t$, candidate $tilde(h)_t$, then a *convex combination*:
#pause
$ z_t = sigma(W_z x_t + U_z h_(t-1) + b_z), quad quad r_t = sigma(W_r x_t + U_r h_(t-1) + b_r) $
#pause
$ tilde(h)_t = tanh(W_h x_t + U_h (r_t dot.o h_(t-1)) + b_h) $
#pause
$ h_t = (1 - z_t) dot.o h_(t-1) + z_t dot.o tilde(h)_t $

== The GRU cell #V

#grucell
#pause
#align(center, text(size: 14pt, fill: MUTED)[reset $r_t$ gates the past *into* the candidate; update $z_t$ mixes old state and candidate])

== Reading the two gates #D

#pause
#two(
  notebox[*update* $z_t$: #h(0.3em) $h_t = (1 - z_t) h_(t-1) + z_t tilde(h)_t$. #linebreak() $z = 0 ->$ copy old state; $z = 1 ->$ take candidate.],
  notebox[*reset* $r_t$: #h(0.3em) scales $h_(t-1)$ *inside* the candidate. #linebreak() $r = 0 ->$ ignore the past, propose from $x_t$ alone.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[$z_t$ plays the LSTM's forget + input as *one* knob; $z approx 0$ is the gradient highway])

== Worked scalar GRU #D

One unit, old state $h_(t-1) = 4$, candidate $tilde(h)_t = 0$. *Keep* — update nearly shut:
#pause
$ z_t = 0.25: quad h_t = 0.75 dot 4 + 0.25 dot 0 = 3 $
#pause
*Replace* — update wide open:
$ z_t = 0.9: quad h_t = 0.1 dot 4 + 0.9 dot 0 = 0.4 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[$z$ small holds the state ($4 -> 3$); $z$ large pulls it to the candidate ($4 -> 0.4$)])

== LSTM vs GRU #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left, left),
  table.header([], [*LSTM*], [*GRU*]),
  [states], [two: $h_t$ and $c_t$], [one: $h_t$],
  [gates], [forget, input, output], [update, reset],
  [memory path], [additive cell $c_t$, Jacobian $"diag"(f_t)$], [convex mix, carry $(1 - z_t)$],
  [parameters], [more (4 weight blocks)], [fewer (3 weight blocks)],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[both fix vanishing gradients; GRU is lighter, LSTM sometimes stronger — *try both, it is task-dependent*])

== Interactive: the GRU cell #I

#interbox[#text(fill: MUTED)[(to build)] — move the update and reset gates and watch the state interpolate between "keep the past" and "take the candidate", side by side with the LSTM.]

// ═══════════════════════════ PART VIII — Applications ═══════════════════════════
= Sequence tasks with RNNs

== Char-level RNN language model #D

Feed characters in, predict the *next* character at every step — the makemore idea, recurrent:
#pause
#align(center, table(
  columns: 5, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (center, center, center, center, center),
  table.header([$t$], [1], [2], [3], [4]),
  [input], [`a`], [`l`], [`i`], [`c`],
  [target], [`l`], [`i`], [`c`], [`e`],
))
#pause
$ p_t = "softmax"(W_o h_t + b_o); quad quad #text[generate by *sampling* $hat(x)_t ~ p_t$ and feeding it back] $

== Sequence classification #D

Many-to-one: read the whole sequence, predict *once*.
#pause
#two(
  notebox[*last state*: #h(0.3em) $hat(y) = "softmax"(W h_T + b)$ — the final $h_T$ summarizes everything.],
  notebox[*mean pool*: #h(0.3em) $hat(y) = "softmax"(W thin overline(h) + b)$, #h(0.2em) $overline(h) = 1/T sum_t h_t$ — robust to length.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[sentiment, topic, spam — one label from a variable-length input])

== Bidirectional RNNs #D

For *non-causal* tasks, run two RNNs — forward and backward — and concatenate:
#pause
$ arrow(h)_t = "RNN"_-> (x_(1:t)), quad quad arrow.l(h)_t = "RNN"_<- (x_(t:T)), quad quad h_t = [arrow(h)_t space ; space arrow.l(h)_t] $
#pause
#alertbox[Each $h_t$ now sees the *whole* sequence — great for tagging or classification. But it peeks at the *future*, so it is *illegal for autoregressive generation*.]

== Padding and masking #D

Batches mix lengths — pad to a common $T$, then *mask the padding out of the loss*:
#pause
$ cal(L) = (sum_(i, t) m_(i t) thin (- log p_(i t))) / (sum_(i, t) m_(i t)), quad quad m_(i t) in {0, 1} $
#pause
#notebox[$m_(i t) = 1$ for real tokens, $0$ for `<pad>`. The mask makes padding invisible to the gradient — otherwise the model learns to predict `<pad>`.]

// ═══════════════════════════ PART IX — Practical training ═══════════════════════════
= Training RNNs in practice

== The training loop

Detach for truncated BPTT, mask the padding, clip the gradient:
#pause
#codebox(size: 13pt)[```python
h = model.init_hidden(batch_size)
for x, y, mask in loader:
    h = h.detach()                     # truncated BPTT: cut the graph
    logits, h = model(x, h)            # forward one chunk
    loss = masked_cross_entropy(logits, y, mask)   # ignore <pad>
    optimizer.zero_grad()
    loss.backward()                    # BPTT within the chunk
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # tame explosions
    optimizer.step()
```]
#pause
#align(center, text(size: 15pt, fill: MUTED)[the three RNN-specific lines: `detach`, `masked` loss, `clip_grad_norm_`])

== Common bugs #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 5.5pt), align: (left, left),
  table.header([*Symptom*], [*Likely cause*]),
  [loss goes `NaN`], [exploding gradients — clip; lower the learning rate],
  [predicts `<pad>` a lot], [no loss mask on padded positions],
  [memory grows each step], [state not detached — full graph kept forever],
  [no long-term dependence], [vanilla RNN vanishing — use LSTM / GRU],
  [generation repeats itself], [greedy decoding — sample / temperature / top-$k$],
  [training is slow], [not packing / batching variable lengths],
))

== RNN vs TCN — a preview

The RNN is inherently *sequential*: $h_t$ needs $h_(t-1)$, so time cannot be parallelized.
#pause
#two(
  notebox[*RNN*: $h_t = f(h_(t-1), x_t)$ — a state carried step by step; $O(T)$ sequential.],
  notebox[*TCN*: stacked *dilated causal convolutions* — a fixed receptive field, fully *parallel* over time.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[next lecture: convolutions over time, and then *attention* removes the recurrence entirely])

// ═══════════════════════════ PART X — Summary ═══════════════════════════
= Summary

== One-slide summary #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Idea*], [*Takeaway*]),
  [recurrent state], [$h_t = f_theta (h_(t-1), x_t)$ — shared $f_theta$, any length, memory in $h_t$],
  [BPTT], [unroll in time; shared-weight gradients *accumulate* over steps],
  [vanilla RNN], [$partial h_T \/ partial h_k approx w^T$ — vanishing / exploding gradients],
  [clipping], [caps exploding norm; does *nothing* for vanishing],
  [LSTM], [cell $c_t = f_t dot.o c_(t-1) + i_t dot.o tilde(c)_t$; forget / input / output gates],
  [GRU], [one state, update + reset gates; simpler, often as good],
))

== Final mental model — one idea

#align(center, text(size: 22pt)[
  RNNs *remember* by carrying a hidden state.
  #v(6pt)
  LSTMs and GRUs *learn when* to remember, forget, and update.
  #v(6pt)
  #text(size: 18pt, fill: MUTED)[a gate is a controllable path — for information *and* for gradients]
])

#focus-slide[
  We gave the model a *memory* and gates to control it.
  #v(12pt)
  #set text(size: 22pt)
  Next: *attention and seq2seq* — let every step look *directly* at every other, no recurrence required.
]
