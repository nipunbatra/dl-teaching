// Convolutional Sequence Models — Lecture 13 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture13/L13-conv-sequence.typ /tmp/L13.pdf
//   typst compile --root . --input handout=true lecture13/L13-conv-sequence.typ /tmp/L13h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [Convolutional Sequence Models],
  subtitle: [Causal convolutions, dilation, TCNs and WaveNet],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let cream = rgb("#EFEEEB")

// ═══ native fletcher diagrams (the centerpieces) ═══════════════════════════

// ── causal connectivity: y_4 sees only x_2,x_3,x_4 (k=3) ──
#let causalconn = align(center, diagram(spacing: (13mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  // causal taps (drawn first, behind nodes)
  edge((1, 0), (3, 1.7), "-|>", stroke: 1pt + TEAL)
  edge((2, 0), (3, 1.7), "-|>", stroke: 1pt + TEAL)
  edge((3, 0), (3, 1.7), "-|>", stroke: 1pt + TEAL)
  // now boundary between x_4 and x_5
  edge((3.5, -0.7), (3.5, 0.7), stroke: (paint: RED, dash: "dashed", thickness: 1pt))
  // inputs
  node((0, 0), text(size: 13pt, $x_1$), radius: 6mm, fill: cream)
  node((1, 0), text(size: 13pt, fill: white, $x_2$), radius: 6mm, fill: TEAL, stroke: 0.9pt + TEAL)
  node((2, 0), text(size: 13pt, fill: white, $x_3$), radius: 6mm, fill: TEAL, stroke: 0.9pt + TEAL)
  node((3, 0), text(size: 13pt, fill: white, $x_4$), radius: 6mm, fill: TEAL, stroke: 0.9pt + TEAL)
  node((4, 0), text(size: 13pt, fill: MUTED, $x_5$), radius: 6mm, fill: white, stroke: (paint: MUTED, dash: "dashed"))
  node((5, 0), text(size: 13pt, fill: MUTED, $x_6$), radius: 6mm, fill: white, stroke: (paint: MUTED, dash: "dashed"))
  // output
  node((3, 1.7), text(size: 13pt, fill: white, $y_4$), radius: 6.5mm, fill: ACC, stroke: 0.9pt + ACC)
  // annotations
  node((3.5, 1.05), text(size: 10.5pt, fill: RED)[now], stroke: none, fill: none)
  node((4.7, -0.98), text(size: 10.5pt, fill: MUTED)[future — not seen], stroke: none, fill: none)
}))

// ── ordinary causal receptive-field cone: 3 layers, k=3, R=7 ──
#let rftree = align(center, diagram(spacing: (11mm, 12mm), node-fill: white, {
  // edges first (behind)
  for p in range(2, 7) { for j in (p - 2, p - 1, p) { edge((p, 1), (j, 0), stroke: 0.35pt + MUTED) } }
  for p in range(4, 7) { for j in (p - 2, p - 1, p) { edge((p, 2), (j, 1), stroke: 0.35pt + MUTED) } }
  for j in (4, 5, 6) { edge((6, 3), (j, 2), stroke: 0.7pt + ACC) }
  // nodes
  for i in range(7) { node((i, 0), none, radius: 2.7mm, fill: TEAL.lighten(55%), stroke: 0.6pt + INK) }
  for i in range(2, 7) { node((i, 1), none, radius: 2.7mm, fill: TEAL.lighten(35%), stroke: 0.6pt + INK) }
  for i in range(4, 7) { node((i, 2), none, radius: 2.7mm, fill: TEAL.lighten(15%), stroke: 0.6pt + INK) }
  node((6, 3), none, radius: 3.6mm, fill: ACC, stroke: 0.8pt + ACC)
  // labels
  node((0, -0.72), text(size: 10pt, fill: MUTED)[$x_(t-6)$], stroke: none, fill: none)
  node((6, -0.72), text(size: 10pt, fill: MUTED)[$x_t$], stroke: none, fill: none)
  node((7.7, 3), text(size: 11pt, fill: ACC)[output], stroke: none, fill: none)
  node((7.7, 0), text(size: 11pt, fill: MUTED)[$7$ inputs], stroke: none, fill: none)
  node((-1.15, 0), text(size: 10pt, fill: MUTED)[layer 0], stroke: none, fill: none)
  node((-1.15, 3), text(size: 10pt, fill: MUTED)[layer 3], stroke: none, fill: none)
}))

// ── dilated dependency tree: k=2, d=1,2,4, reach 2^3 = 8 (WaveNet-style) ──
#let diltree = align(center, diagram(spacing: (12mm, 13mm), node-fill: white, {
  // edges first
  edge((3.5, 3), (1.5, 2), stroke: 0.7pt + ACC); edge((3.5, 3), (5.5, 2), stroke: 0.7pt + ACC)
  edge((1.5, 2), (0.5, 1), stroke: 0.5pt + TEAL); edge((1.5, 2), (2.5, 1), stroke: 0.5pt + TEAL)
  edge((5.5, 2), (4.5, 1), stroke: 0.5pt + TEAL); edge((5.5, 2), (6.5, 1), stroke: 0.5pt + TEAL)
  for (a, b) in ((0.5, 0), (0.5, 1), (2.5, 2), (2.5, 3), (4.5, 4), (4.5, 5), (6.5, 6), (6.5, 7)) {
    edge((a, 1), (b, 0), stroke: 0.4pt + MUTED)
  }
  // nodes
  for i in range(8) { node((i, 0), none, radius: 2.6mm, fill: TEAL.lighten(55%), stroke: 0.6pt + INK) }
  for x in (0.5, 2.5, 4.5, 6.5) { node((x, 1), none, radius: 2.9mm, fill: TEAL.lighten(30%), stroke: 0.6pt + INK) }
  for x in (1.5, 5.5) { node((x, 2), none, radius: 3.1mm, fill: TEAL, stroke: 0.6pt + INK) }
  node((3.5, 3), none, radius: 3.8mm, fill: ACC, stroke: 0.8pt + ACC)
  // dilation labels on the right
  node((8.0, 0.5), text(size: 10.5pt, fill: MUTED)[$d = 1$], stroke: none, fill: none)
  node((8.0, 1.5), text(size: 10.5pt, fill: MUTED)[$d = 2$], stroke: none, fill: none)
  node((8.0, 2.5), text(size: 10.5pt, fill: MUTED)[$d = 4$], stroke: none, fill: none)
  node((3.5, 3.75), text(size: 11pt, fill: ACC)[output], stroke: none, fill: none)
  node((3.5, -0.72), text(size: 10.5pt, fill: MUTED)[$8 = 2^3$ inputs, densely covered], stroke: none, fill: none)
}))

// ── dilation fan-in (single output, k=3, spacing d over n positions) ──
#let fanin(d) = {
  let n = 2 * d + 1
  diagram(spacing: (6.2mm, 12mm), node-fill: white, {
    for j in (0, d, 2 * d) { edge((j, 0), (d, 1.2), "-|>", stroke: 0.8pt + ACC) }
    for i in range(n) {
      let tap = calc.rem(i, d) == 0
      node((i, 0), none, radius: 2.4mm,
        fill: if tap { TEAL } else { white },
        stroke: if tap { 0.7pt + TEAL } else { (paint: MUTED, dash: "dotted") })
    }
    node((d, 1.2), none, radius: 3.2mm, fill: ACC, stroke: 0.8pt + ACC)
  })
}

// ── TCN residual block ──
#let tcnblock = align(center, diagram(spacing: (8.5mm, 10mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let blk(x, lbl, col) = node((x, 0), text(size: 10.5pt, lbl), fill: col.lighten(82%), stroke: 0.9pt + col, corner-radius: 3pt, inset: 5pt)
  node((0, 0), $x_ell$, radius: 6mm, fill: cream)
  blk(1, "dilated\ncausal conv", TEAL); blk(2, "ReLU", ACC); blk(3, "dropout", BLUE)
  blk(4, "dilated\ncausal conv", TEAL); blk(5, "ReLU", ACC); blk(6, "dropout", BLUE)
  node((7, 0), $+$, radius: 5mm, fill: white, stroke: 1pt + INK)
  node((8, 0), $x_(ell+1)$, radius: 7mm, stroke: 0.9pt + ACC)
  for i in range(6) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.8pt + MUTED) }
  edge((6, 0), (7, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((7, 0), (8, 0), "-|>", stroke: 0.8pt + MUTED)
  // residual: 1x1 shortcut over the top
  edge((0, 0), (0, -1.6), (7, -1.6), (7, 0), "-|>", stroke: 1.1pt + GREEN)
  node((3.5, -2.05), text(size: 10.5pt, fill: GREEN)[residual  (1×1 if channels differ)], stroke: none, fill: none)
}))

// ── WaveNet gated residual + skip block ──
#let wavenetblock = align(center, diagram(spacing: (13mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let blk(p, lbl, col) = node(p, text(size: 10.5pt, lbl), fill: col.lighten(82%), stroke: 0.9pt + col, corner-radius: 3pt, inset: 5pt)
  node((0, 0), $x$, radius: 6mm, fill: cream)
  blk((1.15, 0), "dilated\ncausal conv", TEAL)
  node((2.3, 0.75), $tanh$, radius: 6.5mm, fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE)
  node((2.3, -0.75), $sigma$, radius: 6.5mm, fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE)
  node((3.4, 0), $dot.o$, radius: 5mm, fill: white, stroke: 1pt + INK)
  blk((4.5, 0), "1×1", ACC)
  node((5.75, 0), $+$, radius: 5mm, fill: white, stroke: 1pt + INK)
  node((6.9, 0), text(size: 10.5pt)[to next \ block], radius: 8mm, stroke: 0.9pt + ACC)
  node((4.5, 1.5), text(size: 10.5pt, fill: GREEN)[skip out], radius: 7mm, fill: GREEN.lighten(85%), stroke: 0.9pt + GREEN)
  // main path
  edge((0, 0), (1.15, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((1.15, 0), (2.3, 0.75), "-|>", stroke: 0.8pt + MUTED)
  edge((1.15, 0), (2.3, -0.75), "-|>", stroke: 0.8pt + MUTED)
  edge((2.3, 0.75), (3.4, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((2.3, -0.75), (3.4, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((3.4, 0), (4.5, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((4.5, 0), (5.75, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((5.75, 0), (6.9, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((4.5, 0), (4.5, 1.5), "-|>", stroke: 1pt + GREEN)
  // residual over the top
  edge((0, 0), (0, -1.8), (5.75, -1.8), (5.75, 0), "-|>", stroke: 1.1pt + ACC)
  node((2.9, -2.15), text(size: 10pt, fill: ACC)[residual], stroke: none, fill: none)
}))

// ── path length: RNN O(T) chain vs dilated CNN O(log T) ──
#let pathrnn = diagram(spacing: (10mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), $x_1$, radius: 5mm, fill: cream)
  for i in (1, 2, 3, 4) { node((i, 0), text(size: 11pt, $h_#i$), radius: 5mm, fill: TEAL.lighten(75%), stroke: 0.9pt + TEAL) }
  node((5, 0), $y_T$, radius: 5mm, fill: ACC.lighten(78%), stroke: 0.9pt + ACC)
  for i in range(5) { edge((i, 0), (i + 1, 0), "-|>", stroke: 1.1pt + RED) }
  node((2.5, 0.8), text(size: 11pt, fill: RED)[$O(T)$ sequential hops], stroke: none, fill: none)
})
#let pathtcn = diagram(spacing: (9mm, 10mm), node-fill: white, {
  edge((3.5, 2), (1.5, 1), stroke: 0.8pt + GREEN); edge((3.5, 2), (5.5, 1), stroke: 0.5pt + MUTED)
  edge((1.5, 1), (0.5, 0), stroke: 0.5pt + MUTED); edge((1.5, 1), (2.5, 0), stroke: 0.8pt + GREEN)
  for x in (0.5, 2.5, 4.5, 6.5) { node((x, 0), none, radius: 2.6mm, fill: TEAL.lighten(55%), stroke: 0.6pt + INK) }
  for x in (1.5, 5.5) { node((x, 1), none, radius: 2.9mm, fill: TEAL.lighten(30%), stroke: 0.6pt + INK) }
  node((3.5, 2), none, radius: 3.6mm, fill: ACC, stroke: 0.8pt + ACC)
  node((2.5, -0.55), text(size: 10.5pt, fill: GREEN)[$x_1$], stroke: none, fill: none)
  node((3.5, 2.6), text(size: 11pt, fill: GREEN)[$y_T$ reached in $O(log T)$ hops], stroke: none, fill: none)
})

#title-slide()

// ═══════════════════════════ PART I — Why another sequence architecture ═══════════════════════════
= Why another sequence architecture?

== Where we are

We have built two families of *next-token* models over a sequence $x_1, dots, x_T$:
#pause
- a *fixed-window MLP* — condition on the last $k$ tokens (Lecture 11);
- a *recurrent network* — carry a running state $h_t$ (Lecture 12).
#pause
#notebox[Both estimate the *same* object: the conditional $p(x_t | x_(<t))$. Today we add a *third* way to compute it — one built from *convolutions* over time.]

== The objective is architecture-agnostic #D

Every language model factorizes the sequence the *same* way — the chain rule:
#pause
$ p(x_1, dots, x_T) = product_(t=1)^T p(x_t | x_(<t)) $
#pause
and is trained by the *same* negative log-likelihood:
$ cal(L) = - sum_(t=1)^T log p_theta (x_t | x_(<t)) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the *objective* is fixed; only the *function* that produces $p(x_t | x_(<t))$ changes])

== Three ways to model one conditional

The architectures differ only in *how* they summarize the past $x_(<t)$:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 7pt), align: (left, center, left),
  table.header([*Model*], [*Conditioning*], [*Mechanism*]),
  [MLP-LM (L11)], [$p(x_t | x_(t-k:t-1))$], [fixed window of $k$ tokens],
  [RNN-LM (L12)], [$h_t = f(h_(t-1), x_t)$], [recurrent state, one step at a time],
  [*Conv-LM (today)*], [$g("CNN"(x_(<= t)))$], [*convolution over time*],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[same makemore names, same loss — a new way to build the feature that feeds the softmax])

== Why convolutions? #V

An RNN computes $h_t$ *from* $h_(t-1)$ — position $t$ must wait for position $t-1$:
#pause
#align(center, diagram(spacing: (11mm, 6mm), node-stroke: 0.9pt + INK, node-fill: white, {
  // RNN: sequential
  node((-1.3, 0.9), text(size: 11pt, fill: RED)[RNN], stroke: none, fill: none)
  for i in range(5) { node((i, 0.9), text(size: 10pt, $h_#(i+1)$), radius: 4.5mm, fill: TEAL.lighten(78%), stroke: 0.9pt + TEAL) }
  for i in range(4) { edge((i, 0.9), (i + 1, 0.9), "-|>", stroke: 1pt + RED) }
  // Conv: parallel
  node((-1.3, -0.6), text(size: 11pt, fill: GREEN)[conv], stroke: none, fill: none)
  for i in range(5) { node((i, -0.6), text(size: 10pt, $y_#(i+1)$), radius: 4.5mm, fill: ACC.lighten(80%), stroke: 0.9pt + ACC) }
  node((5.6, 0.9), text(size: 10pt, fill: RED)[must go in order], stroke: none, fill: none)
  node((5.6, -0.6), text(size: 10pt, fill: GREEN)[all at once], stroke: none, fill: none)
}))
#pause
#align(center, text(size: 16pt, fill: MUTED)[a convolution applies the *same* kernel to *every* position — all outputs computed *in parallel* during training])

== Four questions for today

To turn a CNN into a sequence model we must answer:
#pause
+ how does a *1-D convolution* slide over time?
#pause
+ how do we stop an output at $t$ from *seeing the future* $x_(>t)$?
#pause
+ how do we reach *long context* without a huge, deep stack?
#pause
+ how does the result *compare* to an RNN?
#pause
#result[causal convolution · dilation · TCN / WaveNet]

// ═══════════════════════════ PART II — 1D convolution over sequences ═══════════════════════════
= 1-D convolution over sequences

== A sequence is a tensor

Batch a set of sequences into one tensor — three axes:
#pause
$ X in RR^(B times T times D) $
#pause
- $B$ — examples in the batch;
- $T$ — *time* / position along the sequence;
- $D$ — *channels* (features) at each position — e.g. the embedding dimension.
#pause
#align(center, text(size: 16pt, fill: MUTED)[a 1-D convolution slides *along the $T$ axis*, mixing across the $D$ channels — the image conv of L8, one dimension down])

== 1-D convolution, one channel #D

Slide a length-$k$ kernel $w$ along the sequence, taking a weighted sum:
#pause
$ y_t = sum_(i=0)^(k-1) w_i thin x_(t+i) $
#pause
- the *same* weights $w_0, dots, w_(k-1)$ at *every* position $t$ — weight sharing;
- each output looks at only $k$ neighbouring inputs — locality.
#pause
#align(center, text(size: 16pt, fill: MUTED)[exactly the 1-D conv of L8 — now the axis it slides along is *time*])

== Multiple channels in and out

Real layers map $C_"in"$ channels to $C_"out"$ channels, just like a 2-D conv:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, center, left),
  table.header([*Tensor*], [*Shape*], [*Meaning*]),
  [input $X$], [$T times C_"in"$], [$C_"in"$ features per step],
  [weights $W$], [$k times C_"in" times C_"out"$], [$C_"out"$ temporal kernels],
  [output $Y$], [$T' times C_"out"$], [$C_"out"$ feature streams],
))
#pause
$ Y_(t, c) = sum_(i=0)^(k-1) sum_(c') W_(i, c', c) thin X_(t+i, c') $
#pause
#align(center, text(size: 16pt, fill: MUTED)[each output channel sums over *all* input channels across a $k$-wide temporal window])

== Worked example: a scalar 1-D conv #D

Input $x = [1, 2, 3, 4, 5]$, kernel $w = [2, -1]$ (valid cross-correlation):
#pause
$ y_t = 2 thin x_t - x_(t+1) $
#pause
$ y_1 = 2(1) - 2 = 0, quad y_2 = 2(2) - 3 = 1, quad y_3 = 2(3) - 4 = 2, quad y_4 = 2(4) - 5 = 3 $
#pause
#result[$y = [0, 1, 2, 3]$]

== The same conv, visually #V

#fig("/lecture13/figures/temporal_conv.svg", w: 62%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[a length-2 kernel over 5 inputs → 4 outputs; the *same* $[2, -1]$ is reused at every step])

== What a temporal kernel detects

A learned temporal kernel responds to one *local pattern in time*:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*Kernel*], [*Detects*]),
  [$[-1, 1]$], [a *rise* — value going up],
  [$[1, -2, 1]$], [a local *peak* / curvature],
  [oriented weights], [a short *n-gram* or motif],
  [smoothing weights], [a *trend* / running average],
))
#pause
#notebox[As in vision, these kernels are *not* hand-designed — they are *learned* by gradient descent to detect whatever temporal motifs help predict the next token.]

== Connection to the fixed-window MLP #D

The MLP-LM also read a window of $k$ tokens — but it used a *different* weight block per slot:
#pause
$ underbrace(#text[MLP: separate $W$ per relative position], "position-specific") quad != quad underbrace(#text[conv: one shared kernel $w$], "temporal weight sharing") $
#pause
#alertbox[The MLP relearns each position separately; move a motif one step and it hits *different* weights. A convolution applies the *same* kernel everywhere — the pattern is detected wherever it occurs.]

== Translation equivariance in time #D

Because one kernel is shared across all $t$, *shifting the input shifts the output the same way*:
#pause
$ f(T_Delta thin x) = T_Delta thin f(x) $
#pause
where $T_Delta$ is a shift by $Delta$ steps. A motif that occurs later produces the same response, later.
#pause
#align(center, text(size: 16pt, fill: MUTED)[the temporal analogue of the image equivariance from L8 — one detector, valid at every moment])

== Interactive: temporal convolution #I

#interbox(link-to: IA + "convolution-visualizer")[
  Slide a 1-D kernel over a signal and watch each multiply-add produce one output — change the weights and see how the detected temporal pattern changes.
]
#pause
*Q.* Which kernel fires on a *falling* edge in the signal?

// ═══════════════════════════ PART III — Causal convolution ═══════════════════════════
= Causal convolution

== The future-information problem

Next-token prediction at $t$ must depend *only* on $x_(<= t)$ — never on what comes later.
#pause
A *centered* conv $y_t = sum_(i=-r)^(r) w_i x_(t+i)$ reads $x_(t+1), x_(t+2), dots$ — the *future*:
#pause
#alertbox[If the representation at $t$ peeks at $x_(t+1)$, then "predicting" the next token is *cheating* — the answer has already leaked in. Training loss looks great; generation is impossible.]

== The causal convolution #D

Flip the window so it reaches only *backward* in time:
#pause
$ y_t = sum_(i=0)^(k-1) w_i thin x_(t-i) $
#pause
- $y_t$ depends on $x_t, x_(t-1), dots, x_(t-k+1)$ — the present and the past;
- *nothing* to the right of $t$ enters the sum.
#pause
#align(center, text(size: 16pt, fill: MUTED)[same weights, same cost — only the *window's direction* changed])

== Causal connectivity #V

With $k = 3$, the output $y_4$ is wired to $x_2, x_3, x_4$ only:
#pause
#causalconn
#pause
#align(center, text(size: 16pt, fill: MUTED)[the boundary at *now* is never crossed — every arrow points from the past])

== Left padding keeps the length #D

To keep $y$ the same length as $x$, pad $k-1$ zeros on the *left* (not symmetrically):
#pause
$ [thick underbrace(0\, dots\, 0, k-1 " zeros") thick, x_1, x_2, dots, x_T] $
#pause
For $k = 3$: prepend $[0, 0]$, so $y_1$ sees $[0, 0, x_1]$, $y_2$ sees $[0, x_1, x_2]$, …
#pause
#align(center, text(size: 16pt, fill: MUTED)[padding is *asymmetric* — all on the left — precisely so no output reaches forward in time])

== Shifted targets

Feed the sequence in, ask each position to predict the *next* token:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*input* (positions $1 dots T$)], [*target* (predict next)]),
  [`<BOS> deep learning is`], [`deep learning is useful`],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[the causal representation at position $t$ is trained to predict the token at position $t+1$])

== Strict vs inclusive causality #OPT

Two consistent conventions — both correct, if you shift the targets to match:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, center, left),
  table.header([*Convention*], [*$y_t$ uses*], [*Predicts*]),
  [inclusive], [$x_(<= t)$], [$x_(t+1)$],
  [strict], [$x_(<t)$], [$x_t$],
))
#pause
#notebox[The only requirement is *consistency*: whatever $y_t$ is allowed to see, the target must be a token it has *not* seen. Mismatch here is the classic "my LM has zero loss but generates garbage" bug.]

== Interactive: causal convolution #I

#interbox[#text(fill: MUTED)[(to build)] — drag the kernel across the sequence and watch it *forbidden* from crossing the "now" line; toggle left-padding and see the output length change.]

// ═══════════════════════════ PART IV — Receptive field ═══════════════════════════
= Receptive field

== How far back can one output see?

The *receptive field* $R$ of an output = how many input positions can affect it.
#pause
- a *single* causal conv layer, kernel $k$: $R = k$;
- stack layers and the field *grows* — deeper outputs see *more* history.
#pause
#result[context length is set by the receptive field, not by any fixed window $k$]

== Stacking ordinary causal convs #D

Each stride-1 layer adds $k - 1$ to the receptive field:
#pause
$ R_L = 1 + L(k - 1) $
#pause
For $k = 3$: $R_L = 1 + 2L$. Three layers, $R_0 = 1$:
$ R_1 = 3, quad R_2 = 5, quad R_3 = 7 $
#pause
#rftree

== The problem: linear growth #D

The receptive field grows only *linearly* in depth. To cover $"1,001"$ steps with $k = 3$:
#pause
$ 1 + 2L = "1,001" quad => quad L = 500 quad #text[layers] $
#pause
#alertbox[$500$ layers just to see $1000$ steps back — far too deep to train or run. Stacking ordinary convs cannot reach long context economically.]
#pause
#result[we need the field to grow *faster* than linearly — enter dilation]

// ═══════════════════════════ PART V — Dilated convolution ═══════════════════════════
= Dilated convolution — the key idea

== Skip positions inside the kernel #D

A *dilated* convolution spreads its $k$ taps apart by a factor $d$:
#pause
$ y_t = sum_(i=0)^(k-1) w_i thin x_(t - d i) $
#pause
For $k = 3, d = 2$:
$ y_t = w_0 thin x_t + w_1 thin x_(t-2) + w_2 thin x_(t-4) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[still only $k$ weights — but they *reach* over $d$-spaced positions, skipping those between])

== Dilation, fan by fan #V

The *same three taps* reach exponentially further as $d$ doubles:
#pause
#align(center, grid(columns: 3, gutter: 16pt, align: bottom,
  stack(spacing: 5pt, fanin(1), text(size: 12pt, fill: TEAL)[$d = 1$], text(size: 11pt, fill: MUTED)[$k_"eff" = 3$]),
  stack(spacing: 5pt, fanin(2), text(size: 12pt, fill: BLUE)[$d = 2$], text(size: 11pt, fill: MUTED)[$k_"eff" = 5$]),
  stack(spacing: 5pt, fanin(4), text(size: 12pt, fill: ACC)[$d = 4$], text(size: 11pt, fill: MUTED)[$k_"eff" = 9$]),
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[teal = tapped, dotted = skipped — $3$ weights spanning $3$, then $5$, then $9$ positions])

== Effective kernel width #D

A dilated kernel of size $k$ spans an *effective* width:
#pause
$ k_"eff" = 1 + (k - 1) d $
#pause
For $k = 3, d = 4$:
$ k_"eff" = 1 + 2 dot 4 = 9 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[$3$ parameters, but a $9$-wide temporal footprint — coverage without cost])

== Receptive field of a dilated stack #D

Summing the per-layer contributions:
#pause
$ R_L = 1 + (k - 1) sum_(ell=1)^L d_ell $
#pause
Choose *exponential* dilation $d_ell = 2^(ell - 1)$, so $sum_(ell=1)^L d_ell = 2^L - 1$:
$ R_L = 1 + (k - 1)(2^L - 1) $
#pause
For $k = 2$ this is exactly $R_L = 2^L$ — the field *doubles* with each layer.
#pause
#align(center, text(size: 16pt, fill: MUTED)[linear depth, *exponential* context — the whole point of dilation])

== Worked dilated receptive field #D

Four layers, $k = 3$, dilations $d = [1, 2, 4, 8]$:
#pause
$ R = 1 + (k - 1) sum d_ell = 1 + 2(1 + 2 + 4 + 8) = 1 + 2 dot 15 = 31 $
#pause
Four *ordinary* causal layers ($k = 3$) reach only:
$ R = 1 + 4 dot 2 = 9 $
#pause
#result[$31$ with dilation vs $9$ without — same depth, same parameters]

== Linear vs exponential, side by side #V

#fig("/lecture13/figures/rf_growth.svg", w: 62%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[ordinary $1 + 2L$ crawls; dilated $2^L$ reaches $"1,024"$ in ten layers])

== Sparse per layer, dense in total #V

Each dilated layer is *sparse*, but stacked they *tile* the past with no gaps:
#pause
#diltree
#pause
#align(center, text(size: 15pt, fill: MUTED)[$k = 3, d = [1,2,4,8]$ covers $31$ contiguous positions; $k = 2, d = [1,2,4]$ covers $2^3 = 8$])

== The coverage, counted #V

#fig("/lecture13/figures/dilation_coverage.svg", w: 84%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[the top output reaches $31$ dense past positions; an ordinary $4$-layer stack reaches only $9$])

== The dilation cycle

Stacks usually *cycle* the dilation and repeat:
#pause
$ 1, 2, 4, 8, thick 1, 2, 4, 8, thick 1, 2, 4, 8, dots $
#pause
- each cycle mixes *local* ($d = 1$) with *long-range* ($d = 8$) structure;
- repeating cycles deepen the network while the field keeps growing;
- this is exactly the WaveNet stacking pattern.
#pause
#align(center, text(size: 16pt, fill: MUTED)[reset-and-grow: fine detail is re-examined at every cycle, never lost to the long jumps])

== Interactive: dilated receptive field #I

#interbox(link-to: IA + "receptive-field-grower")[
  Add dilated layers one at a time — $d = 1, 2, 4, 8, dots$ — and watch the receptive field *double* with each layer as it fans back over the input.
]
#pause
*Q.* With $k = 2$, how many layers reach a context of $"1,024"$?

// ═══════════════════════════ PART VI — Temporal Convolutional Network ═══════════════════════════
= The Temporal Convolutional Network

== The TCN block #V

A TCN block = *two dilated causal convs*, each with activation and dropout, plus a *residual*:
#pause
#tcnblock
#pause
#align(center, text(size: 15pt, fill: MUTED)[Bai, Kolter & Koltun (2018) — dilated causal conv is the primitive; the block wraps it])

== Residual keeps deep stacks trainable #D

The block learns a *residual* $F$ added to its input, exactly as in ResNet:
#pause
$ x_(ell+1) = x_ell + F(x_ell), quad quad #text[or] quad x_(ell+1) = P(x_ell) + F(x_ell) $
#pause
- $P$ is a $1 times 1$ conv, used *only* when the channel counts differ;
- the identity path lets gradients skip the dilated convs — deep dilation stacks train.
#pause
#align(center, text(size: 16pt, fill: MUTED)[the same trick that tamed deep MLPs and CNNs (L6, L8b) returns for deep temporal stacks])

== Sequence length is preserved

Causal (left) padding makes every layer map $T$ inputs to $T$ outputs:
#pause
$ T_"in" = T_"out" quad #text[at every layer] $
#pause
- no shrinking with depth — stack as many dilated blocks as you like;
- one output stream per position, ready for a per-position prediction.
#pause
#align(center, text(size: 16pt, fill: MUTED)[unlike a valid conv that shrinks, a TCN holds the timeline fixed end to end])

== TCN as a next-token model #D

Embed, run the dilated causal stack, project each position to logits:
#pause
$ E_t = E[x_t], quad quad H_(1:T) = "TCN"(E_(1:T)) $
#pause
$ z_t = W_o thin h_t + b_o, quad quad p(x_(t+1) | x_(<= t)) = "softmax"(z_t) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[same makemore head as the MLP-LM and RNN-LM — only the feature extractor is a TCN])

== One forward predicts every position

Because the whole stack is convolutional, a *single* forward pass emits all outputs:
#pause
$ z_1, z_2, dots, z_T quad #text[predict] quad x_2, x_3, dots, x_(T+1) quad #text[at once] $
#pause
#notebox[Causal padding guarantees $z_t$ never saw $x_(>t)$, so computing *all* positions in parallel is still correct. This is the training-time advantage over an RNN's step-by-step recurrence.]

== Parameter sharing: across time, not depth #D

Where do the weights get reused?
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*Model*], [*Weights reused …*]),
  [RNN], [across *time steps* — the *same* transition every step],
  [TCN], [across *positions within a layer*; each *layer* has its own kernel],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[a TCN shares a kernel along time, but *different layers* learn different kernels — depth is not tied])

== Effective vs theoretical memory

The receptive field gives an *upper bound* on memory — the model need not use all of it:
#pause
- theoretical context $= R_L$; *effective* context may be shorter;
- TCNs can achieve *long* effective memory and, on some long-range benchmarks, match or beat LSTMs.
#pause
#result[recurrence is *not* required for long-range sequence modeling]

// ═══════════════════════════ PART VII — WaveNet ═══════════════════════════
= WaveNet — dilated causal convs on raw audio

== The objective

WaveNet models a *raw audio waveform* $x_1, dots, x_T$ autoregressively — sample by sample:
#pause
$ p(x_1, dots, x_T) = product_(t=1)^T p(x_t | x_(<t)) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[van den Oord et al. (2016) — the *same* chain rule, now over audio *samples* instead of characters])

== Why audio is hard

Raw audio is a brutal sequence-modeling target:
#pause
- *thousands* of samples per second → sequences of tens of thousands of steps;
- prediction must be *exactly causal* (no future leakage);
- needs *large context* (a phoneme spans thousands of samples) at *fine resolution*.
#pause
#align(center, text(size: 16pt, fill: MUTED)[dilated causal convolutions were designed to hit *all* of these at once])

== Gated activation #D

WaveNet replaces the plain activation with a *gated* one — LSTM-like, inside a conv block:
#pause
$ z = underbrace(tanh(W_f ast x), "candidate") thick dot.o thick underbrace(sigma(W_g ast x), "gate") $
#pause
- $tanh$ proposes a value, $sigma in (0, 1)$ decides how much passes;
- $dot.o$ is element-wise — the multiplicative gate from L12, now convolutional.
#pause
#align(center, text(size: 16pt, fill: MUTED)[gating gives the block fine control over what each dilated filter contributes])

== Residual and skip paths #V

Each block feeds a *residual* forward and a *skip* to the output sum:
#pause
#wavenetblock
#pause
#align(center, text(size: 15pt, fill: MUTED)[gated conv → 1×1 → split: a residual to the next block, a skip to the final sum])

== The WaveNet stack

Assemble the blocks into the full model:
#pause
+ a causal conv on the input;
#pause
+ a stack of *gated, dilated, residual* blocks — dilation $1, 2, 4, dots, 512$, then *repeat*;
#pause
+ *sum all skip outputs* → ReLU → $1 times 1$ → ReLU → $1 times 1$ → logits.
#pause
#align(center, text(size: 15pt, fill: MUTED)[stacked dilation cycles give a receptive field of *thousands* of samples with modest depth])

== Training vs generation

The parallelism cuts one way only:
#pause
#two(
  notebox[*Training* \ the whole waveform is known → *all* positions in *one* parallel forward pass.],
  alertbox[*Generation* \ each sample feeds the next → *sequential*, one step at a time (unless conv states are cached).],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[fast to *train* on known data; still autoregressive — hence slow — to *sample*])

== Beyond audio

The same recipe of *causal conv, dilation, residual, and autoregression* reaches far beyond audio:
#pause
- time-series forecasting; language modeling (ByteNet);
- sensor streams, financial ticks; biological sequences.
#pause
#notebox[WaveNet's contribution is less "audio" than a *general blueprint* for autoregressive sequence models built entirely from dilated causal convolutions.]

== Interactive: WaveNet block #I

#interbox[#text(fill: MUTED)[(to build)] — step through one gated residual block: watch $tanh$ and $sigma$ combine, the skip branch peel off, and the residual rejoin for the next block.]

// ═══════════════════════════ PART VIII — CNN vs RNN ═══════════════════════════
= Convolutional vs recurrent sequence models

== Two ways to reach the past

Both see history — but *how* differs fundamentally:
#pause
#two(
  notebox[*RNN* \ compresses all of $x_(<= t)$ into a fixed-size *recurrent state* $h_t$ — a lossy running summary.],
  notebox[*TCN* \ reads a *finite receptive field* of raw positions directly — no compression, but bounded reach.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[state (unbounded but lossy) vs receptive field (lossless but finite)])

== Computational path length #V

How many steps must a gradient travel between a distant input and the output?
#pause
#two(
  align(center, pathrnn),
  align(center, pathtcn),
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[RNN: $O(T)$ sequential hops from $x_1$ to $y_T$ · dilated CNN: only $O(log T)$ layers — far shorter gradient paths])

== Parallelism, side by side

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, center, center),
  table.header([ ], [*RNN*], [*TCN*]),
  [parallel training], [limited (sequential)], [high (all positions)],
  [generation], [sequential], [sequential],
  [context mechanism], [recurrent state], [receptive field],
  [context length], [unbounded (in principle)], [finite ($R_L$)],
  [weight sharing], [across steps], [across positions],
  [path to distant input], [$O(T)$], [$O(log T)$],
))

== Inductive bias — neither is universal

Each architecture *assumes* something about the sequence:
#pause
- *RNN* — a sequential *state update*: the past matters through an evolving summary;
- *CNN* — *local patterns* composed *hierarchically*: motifs, then motifs of motifs.
#pause
#alertbox[Neither is universally better. The right choice depends on sequence *length*, needed *context*, *latency*, *compute*, and the *task*.]

== Why Transformers still become attractive

The TCN fixes the RNN's three pains — parallel training, short gradient paths, fast field growth. But:
#pause
#alertbox[A convolution's aggregation pattern is *predetermined* by its kernel and dilation. *Which* past positions matter is fixed by the architecture, not by the *content*.]
#pause
#result[attention lets each position *dynamically choose* which past positions to attend to]
#pause
#align(center, text(size: 16pt, fill: MUTED)[content-dependent, all-to-all mixing — the transition we take up next])

// ═══════════════════════════ PART IX — Seq2Seq preview + summary ═══════════════════════════
= Preview and summary

== Convolutional encoder–decoder

Convolutions can build a *full* sequence-to-sequence model, not just a language model:
#pause
$ H = "CNN"_"enc"(x), quad quad s_u = "CNN"_"dec"(y_(<u), H) $
#pause
- *ByteNet* and *ConvSeq2Seq* (Gehring et al.) built translation entirely from convolutions;
- a causal decoder conv over $y_(<u)$, conditioned on the encoder features $H$.
#pause
#align(center, text(size: 16pt, fill: MUTED)[a one-slide preview — the encoder–decoder structure is the topic of the next lecture])

== Final summary

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6.5pt), align: (left, left),
  table.header([*Idea*], [*Takeaway*]),
  [1-D convolution], [detect *local temporal patterns*; weight sharing along time],
  [causal convolution], [left padding → no future leakage; predict the next token],
  [receptive field], [ordinary stacking grows it only *linearly* ($1 + 2L$)],
  [dilation], [$d_ell = 2^(ell-1)$ → context grows *exponentially* with depth],
  [TCN], [causal dilated convs $+$ residual; parallel over time],
  [WaveNet], [gated dilated causal convs $+$ autoregressive generation],
))

== Final mental model — one idea

#align(center, text(size: 22pt)[
  A sequence can be modeled *without recurrence* — \
  #v(4pt)
  by *convolving over time*, *causally*, with *dilation* for reach. \
  #v(4pt)
  #text(size: 18pt, fill: MUTED)[parallel training · short gradient paths · context that grows with depth]
])

#focus-slide[
  Convolutions reach the past through a *fixed* receptive field.
  #v(12pt)
  #set text(size: 22pt)
  Next: *sequence-to-sequence and attention* — where each position *chooses* which others to attend to, dynamically.
]
