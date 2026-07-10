// Sequence Models II: Encoder–Decoder, Teacher Forcing and Attention — Lecture 14 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture14/L14-seq2seq-attention.typ /tmp/L14.pdf
//   typst compile --root . --input handout=true lecture14/L14-seq2seq-attention.typ /tmp/L14h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [Sequence Models II: Encoder–Decoder, Teacher Forcing and Attention],
  subtitle: [Mapping one sequence to another — and letting the decoder look back],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let cream = rgb("#EFEEEB")

// ═══════════════════════════ fletcher diagrams (the centerpieces) ═══════════════════════════

// ── (A) encoder–decoder, no attention: encoder RNN → context c=h_T → decoder RNN → outputs ──
#let encdec = align(center, diagram(spacing: (12mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  // encoder: inputs (bottom) → cells → states flow right
  for (i, l) in ((0, $x_1$), (1, $x_2$), (2, $x_3$)) {
    node((i, -1.35), text(size: 12pt, l), radius: 5mm, fill: cream)
    node((i, 0), text(size: 12pt)[Enc], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 6pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
    edge((i, -1.35), (i, 0), "-|>", stroke: 0.7pt + MUTED)
  }
  edge((0, 0), (1, 0), "-|>", stroke: 1pt + TEAL, label: text(size: 10pt, fill: TEAL)[$h_1$])
  edge((1, 0), (2, 0), "-|>", stroke: 1pt + TEAL, label: text(size: 10pt, fill: TEAL)[$h_2$])
  // context bridge c = h_3
  node((3.05, 0), [$c = h_3$], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(80%), stroke: 1pt + ACC)
  edge((2, 0), (3.05, 0), "-|>", stroke: 1.3pt + ACC)
  // decoder: inputs (bottom) → cells → outputs (top)
  for (i, xin, yout) in ((4.1, [`<BOS>`], $y_1$), (5.1, $y_1$, $y_2$), (6.1, $y_2$, [`<EOS>`])) {
    node((i, -1.35), text(size: 10pt, xin), shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 5pt, fill: cream)
    node((i, 0), text(size: 12pt)[Dec], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 6pt, fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE)
    node((i, 1.35), text(size: 11pt, yout), shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 5pt, fill: ACC.lighten(80%), stroke: 0.9pt + ACC)
    edge((i, -1.35), (i, 0), "-|>", stroke: 0.7pt + MUTED)
    edge((i, 0), (i, 1.35), "-|>", stroke: 0.8pt + ACC)
  }
  edge((3.05, 0), (4.1, 0), "-|>", stroke: 1.3pt + ACC, label: text(size: 9pt, fill: MUTED)[init $s_0$])
  edge((4.1, 0), (5.1, 0), "-|>", stroke: 1pt + BLUE, label: text(size: 10pt, fill: BLUE)[$s_1$])
  edge((5.1, 0), (6.1, 0), "-|>", stroke: 1pt + BLUE, label: text(size: 10pt, fill: BLUE)[$s_2$])
}))

// ── (B) the fixed-vector bottleneck: everything squeezed through one c ──
#let bottleneck = align(center, diagram(spacing: (16mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), [source \ #text(size: 10pt, fill: MUTED)[$x_1, x_2, dots, x_T$]], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 8pt, fill: cream)
  node((1.5, 0), [Encoder], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 7pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((2.75, 0), text(size: 13pt)[$c$], radius: 6mm, fill: RED.lighten(78%), stroke: 1.2pt + RED)
  node((4.0, 0), [Decoder], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 7pt, fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE)
  node((5.5, 0), [target \ #text(size: 10pt, fill: MUTED)[$y_1, y_2, dots, y_U$]], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 8pt, fill: ACC.lighten(80%), stroke: 0.9pt + ACC)
  edge((0, 0), (1.5, 0), "-|>", stroke: 0.9pt + INK)
  edge((1.5, 0), (2.75, 0), "-|>", stroke: 1.4pt + TEAL)
  edge((2.75, 0), (4.0, 0), "-|>", stroke: 1.4pt + BLUE)
  edge((4.0, 0), (5.5, 0), "-|>", stroke: 0.9pt + INK)
  node((2.75, -1.15), text(size: 11pt, fill: RED)[one fixed vector], stroke: none, fill: none)
  node((2.75, 1.15), text(size: 11pt, fill: RED)[the *bottleneck*], stroke: none, fill: none)
}))

// ── (C) the attention mechanism: h_1..h_5 → weights α → weighted sum → context c_u ──
#let attnmech = align(center, diagram(spacing: (16mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let states = ((1, 2, 0.05), (2, 1, 0.10), (3, 0, 0.60), (4, -1, 0.20), (5, -2, 0.05))
  // query on top sets the weights
  node((1.15, 3.05), [query #text(size: 11pt)[$s_(u-1)$]], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 5pt, fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE)
  edge((1.15, 3.05), (1.15, 1.7), "-|>", stroke: (paint: BLUE, dash: "dashed", thickness: 0.9pt), label: text(size: 9pt, fill: BLUE)[sets $alpha_(u t)$ via score], label-side: left)
  // encoder states (left column) and the weighted-sum fan
  node((2.3, 0), text(size: 15pt)[$Sigma$], radius: 6.5mm, fill: cream)
  for (i, y, a) in states {
    let big = a >= 0.5
    node((0, y), $h_#i$, radius: 5.5mm, fill: if big { ACC.lighten(70%) } else { TEAL.lighten(82%) }, stroke: 0.9pt + (if big { ACC } else { TEAL }))
    edge((0, y), (2.3, 0), "-|>", stroke: (thickness: (0.4 + a * 4.2) * 1pt, paint: if big { ACC } else { MUTED.lighten(10%) }), label: text(size: 9.5pt, fill: if big { ACC } else { MUTED })[$alpha_#i$], label-pos: 0.28)
  }
  // context and the downstream prediction
  node((3.7, 0), $c_u$, radius: 6mm, fill: ACC.lighten(80%), stroke: 1pt + ACC)
  edge((2.3, 0), (3.7, 0), "-|>", stroke: 1.3pt + ACC)
  node((5.2, 0), [Dec #text(size: 11pt)[$-> y_u$]], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 5pt, fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE)
  edge((3.7, 0), (5.2, 0), "-|>", stroke: 1pt + INK, label: text(size: 9pt, fill: MUTED)[$[s_u; c_u]$])
}))

// ── (D) attention as a differentiable lookup: query · keys → weights → values → output ──
#let softlookup = align(center, diagram(spacing: (17mm, 8.5mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0.5), [query \ #text(size: 10pt, fill: MUTED)[$q = s_(u-1)$]], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 6pt, fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE)
  // keys and values columns
  for (i, y) in ((1, 1.5), (2, 0.5), (3, -0.5)) {
    node((1.5, y), [$k_#i$], radius: 5mm, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
    node((3.5, y), [$v_#i$], radius: 5mm, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
    edge((0, 0.5), (1.5, y), "-|>", stroke: (paint: BLUE, dash: "dashed", thickness: 0.7pt))
  }
  // scores → softmax → weights, then weight the values into the sum
  node((2.5, -1.9), [$alpha_t = "softmax"_t (q^top k_t)$], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(80%), stroke: 0.9pt + ACC)
  node((5, 0.5), text(size: 14pt)[$Sigma$], radius: 6mm, fill: cream)
  for (i, y) in ((1, 1.5), (2, 0.5), (3, -0.5)) {
    edge((3.5, y), (5, 0.5), "-|>", stroke: 0.9pt + GREEN)
  }
  edge((2.5, -1.9), (5, 0.5), "-|>", bend: -18deg, stroke: (paint: ACC, thickness: 1pt), label: text(size: 9pt, fill: ACC)[weights])
  node((6.4, 0.5), [output \ #text(size: 10pt, fill: MUTED)[$sum_t alpha_t v_t$]], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(80%), stroke: 0.9pt + ACC)
  edge((5, 0.5), (6.4, 0.5), "-|>", stroke: 1.1pt + INK)
}))

// ── (E) beam search tree (B = 2): expand, score, keep top-B ──
#let beamtree = align(center, diagram(spacing: (24mm, 6.5mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let keep(pos, lbl, sc) = node(pos, [#lbl \ #text(size: 9pt, fill: rgb("#1B7A34"))[#sc]], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 5pt, fill: GREEN.lighten(85%), stroke: 0.9pt + GREEN)
  let prune(pos, lbl, sc) = node(pos, [#lbl \ #text(size: 9pt, fill: MUTED)[#sc]], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 5pt, fill: cream, stroke: (paint: MUTED, dash: "dashed"))
  node((0, 0), text(size: 11pt, fill: white)[`<BOS>`], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 5pt, fill: INK, stroke: none)
  // level 1
  keep((1, 1.7), [`the`], [$-0.4$])
  keep((1, 0.1), [`a`], [$-0.9$])
  prune((1, -1.4), [`cat`], [$-1.8$])
  edge((0, 0), (1, 1.7), "-|>", stroke: 1pt + GREEN)
  edge((0, 0), (1, 0.1), "-|>", stroke: 1pt + GREEN)
  edge((0, 0), (1, -1.4), "-|>", stroke: (paint: MUTED, dash: "dashed"))
  // level 2 (expand the two survivors)
  keep((2, 2.5), [`the cat`], [$-0.9$])
  keep((2, 1.2), [`the dog`], [$-1.6$])
  prune((2, -0.1), [`a cat`], [$-1.9$])
  prune((2, -1.3), [`a dog`], [$-2.3$])
  edge((1, 1.7), (2, 2.5), "-|>", stroke: 1pt + GREEN)
  edge((1, 1.7), (2, 1.2), "-|>", stroke: 1pt + GREEN)
  edge((1, 0.1), (2, -0.1), "-|>", stroke: (paint: MUTED, dash: "dashed"))
  edge((1, 0.1), (2, -1.3), "-|>", stroke: (paint: MUTED, dash: "dashed"))
  node((2.95, 2.5), text(size: 12pt, fill: ACC)[★ best], stroke: none, fill: none)
}))

// ── (F) full seq2seq + attention (block view): encoder H → attention ⇄ decoder → y_u ──
#let fullmodel = align(center, diagram(spacing: (16mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  // encoder row (top)
  node((-1.4, 1.5), [source \ #text(size: 10pt, fill: MUTED)[$x_(1:T)$]], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 6pt, fill: cream)
  node((0.2, 1.5), [Encoder RNN], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 7pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((2.1, 1.5), [$H = (h_1, dots, h_T)$], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 7pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  edge((-1.4, 1.5), (0.2, 1.5), "-|>", stroke: 0.9pt + INK)
  edge((0.2, 1.5), (2.1, 1.5), "-|>", stroke: 1pt + TEAL)
  // attention (middle)
  node((2.1, 0), [Attention \ #text(size: 10pt, fill: MUTED)[$alpha_(u t), space c_u = sum_t alpha_(u t) h_t$]], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 7pt, fill: ACC.lighten(80%), stroke: 1pt + ACC)
  edge((2.1, 1.5), (2.1, 0), "-|>", stroke: 1pt + TEAL, label: text(size: 9pt, fill: TEAL)[keys / values], label-side: right)
  // decoder row (bottom)
  node((0.2, -1.5), [prev token \ #text(size: 10pt, fill: MUTED)[$y_(u-1)$]], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 6pt, fill: cream)
  node((2.1, -1.5), [Decoder RNN \ #text(size: 10pt, fill: MUTED)[$s_u$]], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 7pt, fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE)
  edge((0.2, -1.5), (2.1, -1.5), "-|>", stroke: 0.9pt + INK)
  edge((2.1, -1.5), (2.1, 0), "-|>", stroke: (paint: BLUE, dash: "dashed", thickness: 0.9pt), label: text(size: 9pt, fill: BLUE)[query $s_u$], label-side: right)
  // prediction
  node((4.2, -0.75), [softmax \ #text(size: 10pt, fill: MUTED)[$-> y_u$]], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 7pt, fill: ACC.lighten(80%), stroke: 0.9pt + ACC)
  edge((2.1, 0), (4.2, -0.75), "-|>", stroke: 1pt + ACC, label: text(size: 9pt, fill: ACC)[$c_u$], label-pos: 0.32, label-side: left)
  edge((2.1, -1.5), (4.2, -0.75), "-|>", stroke: 0.9pt + BLUE, label: text(size: 9pt, fill: BLUE)[$s_u$], label-pos: 0.72, label-side: right)
}))

#title-slide()

// ═══════════════════════════ PART I — Why seq2seq ═══════════════════════════
= From one sequence to another

== Where we are

The last three lectures modeled *one* sequence — next-token prediction over $x_(1:T)$:
#pause
$ p(x_(1:T)) = product_(t=1)^T p(x_t | x_(<t)) $
#pause
#notebox[This lecture changes the *task*: map an *input* sequence to a *different output* sequence — translation, summarization, speech. Input and output can even have *different lengths*.]

== Tasks that map sequence → sequence #V

Many problems are $x_(1:T) -> y_(1:U)$ — a source sequence in, a target sequence out:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left, left),
  table.header([*Task*], [*Input $x_(1:T)$*], [*Output $y_(1:U)$*]),
  [translation], [English sentence], [French sentence],
  [summarization], [a document], [a short summary],
  [speech recognition], [audio frames], [a transcript],
  [image captioning], [image regions], [a caption],
  [dialogue], [conversation so far], [the next reply],
  [forecasting], [past observations], [future horizon],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[one general shape — *sequence in, sequence out* — behind all of them])

== The lengths differ #D

The defining difficulty: the output length $U$ is *not* the input length $T$, and is *not known in advance*.
#pause
#align(center, text(size: 19pt, fill: INK)[
  "I like cats" #text(fill: MUTED)[($T = 3$)] $quad -> quad$ "j'aime les chats" #text(fill: MUTED)[($U = 3$)]
])
#pause
#alertbox[A token classifier ($x_(1:T) -> y_(1:T)$, one label per input) *cannot* do this: it forces $U = T$ and a fixed alignment. We need a model that *generates* an output of *flexible* length.]

== The conditional sequence model #D

Keep the autoregressive factorization — but now *condition on the whole source* $x_(1:T)$:
#pause
$ p(y_(1:U) | x_(1:T)) = product_(u=1)^U p(y_u | y_(<u), x_(1:T)) $
#pause
#two(
  notebox[*autoregressive* in the target: each $y_u$ depends on the tokens generated *before* it.],
  notebox[*conditioned* on the source: every factor also sees the *entire input* $x_(1:T)$.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[same next-token idea as Lecture 11 — now with a *source* to condition on])

// ═══════════════════════════ PART II — Encoder–decoder (no attention) ═══════════════════════════
= The encoder–decoder

== Two networks, one bridge

Split the job in two (Sutskever, Vinyals & Le 2014; Cho et al. 2014):
#pause
#two(
  notebox[*Encoder* — reads the source $x_(1:T)$ and compresses it into a *context vector* $c$.],
  notebox[*Decoder* — a conditional language model that *generates* $y_(1:U)$ from $c$.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[the context $c$ is the only bridge — everything the decoder knows about the source is in $c$])

== The encoder #D

The encoder is an RNN (LSTM/GRU) run over the source; its *final state* is the context:
#pause
$ h_t = f_"enc" (h_(t-1), e(x_t)), quad quad t = 1, dots, T $
#pause
$ c = h_T quad quad #text(size: 15pt, fill: MUTED)[(the whole source, summarized in one vector)] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[read the source left to right; $h_T$ is a fixed-size summary of *all* of it])

== The decoder #D

A second RNN, *initialized from $c$*, generates the target one token at a time:
#pause
$ s_u = f_"dec" (s_(u-1), e(y_(u-1)), c), quad quad s_0 = c $
#pause
$ p(y_u | y_(<u), x_(1:T)) = "softmax"(W_o thin s_u + b_o) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the decoder is a language model over the target, *conditioned* on the source through $c$])

== The architecture #V

Encode the source into $c$; hand $c$ to the decoder; generate the target autoregressively:
#pause
#encdec
#pause
#align(center, text(size: 15pt, fill: MUTED)[`<BOS>` starts generation; each output feeds the next step; `<EOS>` stops it])

== Start and end tokens #D

Two special tokens make *variable-length* generation work:
#pause
#two(
  notebox[`<BOS>` — the decoder's *first input*, so it can produce $y_1$ with no target history.],
  notebox[`<EOS>` — an ordinary vocabulary token; *emitting it* is how the model *chooses to stop*.],
)
#pause
Training a pair shifts the target by one — *input* vs *target* of the decoder:
#align(center, table(
  columns: 5, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 5.5pt), align: (left, center, center, center, center),
  table.header([decoder input], [`<BOS>`], [`y_1`], [`y_2`], [`y_3`]),
  [decoder target], [`y_1`], [`y_2`], [`y_3`], [`<EOS>`],
))

== The training loss #D

Maximize the conditional log-likelihood — sum token cross-entropy over the target:
#pause
$ cal(L) = - sum_(u=1)^U log p(y_u | y_(<u), x_(1:T)) $
#pause
Batches mix target lengths, so pad and *mask* the padded positions out:
$ cal(L) = (sum_(i, u) m_(i u) thin (- log p_(i u))) / (sum_(i, u) m_(i u)), quad quad m_(i u) in {0, 1} $
#pause
#align(center, text(size: 16pt, fill: MUTED)[$m_(i u) = 0$ on `<PAD>` — padding is invisible to the gradient])

// ═══════════════════════════ PART III — Teacher forcing ═══════════════════════════
= Teacher forcing

== Feeding the true previous token #D

At each decoder step the input is the *previous target token*. Which one — the *true* one, or the model's *guess*?
#pause
#result[*Teacher forcing*: during training, feed the *ground-truth* $y_(u-1)^"true"$, not the prediction $hat(y)_(u-1)$.]
#pause
$ s_u = f_"dec" (s_(u-1), e(y_(u-1)^"true"), c) quad quad #text(size: 15pt, fill: MUTED)[(training)] $

== Why teacher forcing #D

#pause
#two(
  notebox[*Stable, parallel supervision* — every step gets a *correct* input, so all $U$ targets can be scored in one pass.],
  alertbox[*Without it* — an early wrong token becomes the *next* step's input, and the error *compounds* down the sequence.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[teacher forcing decouples the steps: each is trained as a clean next-token classification])

== Train ≠ inference: exposure bias #D

The catch — the two regimes feed *different* previous tokens:
#pause
$ #text(fill: TEAL)[train:] quad s_u = f_"dec" (s_(u-1), e(y_(u-1)^"true"), c) $
$ #text(fill: RED)[infer:] quad s_u = f_"dec" (s_(u-1), e(hat(y)_(u-1)), c) $
#pause
#alertbox[*Exposure bias*: the model only ever saw *true* histories in training, but at inference it must consume its *own* (sometimes wrong) outputs — a distribution it was never trained on.]

== Scheduled sampling #D #OPT

A middle ground (Bengio et al. 2015): *sometimes* feed the model's own prediction during training.
#pause
$ #text[decoder input at step $u$] = cases(
  y_(u-1)^"true" & #text[with prob. ] p,
  hat(y)_(u-1) & #text[with prob. ] 1 - p,
) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[anneal $p: 1 -> 0$ over training — start fully teacher-forced, gradually expose the model to its own outputs])

// ═══════════════════════════ PART IV — Worked seq2seq ═══════════════════════════
= A worked seq2seq pass

== The toy translation

Translate "I like cats" $-> $ "j'aime les chats". Set up the source and the shifted target:
#pause
$ x = [#raw("I"), #raw("like"), #raw("cats"), #raw("<EOS>")] $
#pause
$ #text[decoder input] = [#raw("<BOS>"), #raw("j'aime"), #raw("les"), #raw("chats")] $
$ #text[decoder target] = [#raw("j'aime"), #raw("les"), #raw("chats"), #raw("<EOS>")] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the target is the input shifted by one — teacher forcing supplies the true tokens])

== Encode, then start decoding #D

Run the encoder over the four source tokens; take the final state as the context:
#pause
$ h_1, h_2, h_3, h_4 = f_"enc"(dots), quad quad c = h_4 $
#pause
Initialize the decoder and take its *first* step from `<BOS>`:
$ s_1 = f_"dec" (s_0, e(#raw("<BOS>")), c), quad quad s_0 = c $
#pause
$ p_1 = "softmax"(W_o thin s_1 + b_o) quad quad #text(size: 15pt, fill: MUTED)[(a distribution over the French vocabulary)] $

== Step 1 loss #D

The target for step 1 is `j'aime`; the loss is its negative log-probability:
#pause
$ cal(L)_1 = - log p_1(#raw("j'aime")) $
#pause
Teacher forcing now feeds the *true* `j'aime` (not $hat(y)_1$) as the input to step 2:
$ s_2 = f_"dec" (s_1, e(#raw("j'aime")), c) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[each step: predict, score against the true next token, then feed that true token onward])

== The full sequence loss #D

Sum the per-step cross-entropies over the whole target:
#pause
$ cal(L) = - log p_1(#raw("j'aime")) - log p_2(#raw("les")) - log p_3(#raw("chats")) - log p_4(#raw("<EOS>")) $
#pause
#result[$cal(L) = - sum_(u=1)^4 log p_u(y_u^"true")$ — four next-token classifications, one per target position]
#pause
#align(center, text(size: 16pt, fill: MUTED)[note the last term: the model is *supervised to emit `<EOS>`* — that is how it learns to stop])

// ═══════════════════════════ PART V — Inference & decoding ═══════════════════════════
= Inference and decoding

== Greedy decoding #D

At inference there is no true history — feed the model's own output back. Simplest rule: take the *arg max* each step:
#pause
$ hat(y)_u = arg max_v thin p(v | hat(y)_(<u), x_(1:T)) $
#pause
#alertbox[*Greedy is myopic.* The locally most-probable token need not start the globally most-probable *sequence* — one early commitment can doom the rest.]

== Beam search #D

Keep the $B$ best *partial* sequences (the *beam*) instead of one. Score a hypothesis by its log-probability:
#pause
$ "score"(y_(1:u)) = sum_(j=1)^u log p(y_j | y_(<j), x_(1:T)) $
#pause
- *expand* every beam by all vocabulary tokens;
- *score* each extension; *keep* only the top $B$;
- repeat until every beam has emitted `<EOS>`.
#pause
#align(center, text(size: 16pt, fill: MUTED)[$B = 1$ is greedy; larger $B$ searches wider (more compute) for a better sequence])

== Beam search, drawn #V

$B = 2$: at each level, expand the survivors, score, and keep the best two:
#pause
#beamtree
#pause
#align(center, text(size: 15pt, fill: MUTED)[green = kept in the beam; dashed = pruned; the highest-scoring complete sequence wins])

== The length bias #D

Every factor has $log p <= 0$, so a *longer* sequence has a *more negative* raw score:
#pause
$ "score"(y_(1:U)) = sum_(j=1)^U log p(y_j | dots) quad <= quad 0 quad (#text[shrinks with each token]) $
#pause
#alertbox[Raw log-prob *favors short outputs* — the beam is biased toward emitting `<EOS>` early. Fix it by *length-normalizing* the score:]
#pause
$ "score"_"norm" (y_(1:U)) = 1/U^alpha sum_(j=1)^U log p(y_j | dots), quad quad alpha in [0.6, 1] $

// ═══════════════════════════ PART VI — The fixed-vector bottleneck ═══════════════════════════
= The fixed-vector bottleneck

== Everything through one vector #V

In the plain encoder–decoder, *all* of the source must pass through a *single* $c = h_T$:
#pause
#bottleneck
#pause
#align(center, text(size: 15pt, fill: MUTED)[a 40-word sentence and a 4-word sentence get the *same* fixed-size $c$])

== What goes wrong #D

#pause
- for a *long* source, one vector cannot retain every detail — early tokens get *overwritten*;
- the decoder needs *different* parts of the source at *different* output steps — but $c$ is *fixed* for all of them;
- empirically, translation quality *falls off sharply* as the source grows (Bahdanau et al. 2015).
#pause
#alertbox[The bottleneck is *architectural*, not a training issue: a single fixed vector is simply too small a channel for a long, structured source.]

== What we actually want #D

At *each* output step, the decoder should look at the *relevant* source positions — not a frozen summary:
#pause
#result[replace one fixed $c$ with a *per-step*, *source-dependent* context $c_u$]
#pause
#align(center, text(size: 16pt, fill: MUTED)[when translating `chats`, look at `cats`; when translating `j'aime`, look at `like` — this is *attention*])

// ═══════════════════════════ PART VII — Attention (the core) ═══════════════════════════
= Attention

== The idea #D

Keep *all* encoder states $h_1, dots, h_T$. Build the context as a *weighted average* of them, recomputed every step:
#pause
$ c_u = sum_(t=1)^T alpha_(u t) thin h_t, quad quad alpha_(u t) >= 0, quad sum_(t=1)^T alpha_(u t) = 1 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[$alpha_(u t)$ = how much output step $u$ *attends to* source position $t$ — a soft, differentiable selection])

== Scores and weights #D

The weights come from a *score* between the decoder query and each encoder state, passed through a softmax:
#pause
$ e_(u t) = "score"(s_(u-1), h_t) quad quad #text(size: 15pt, fill: MUTED)[(one scalar per source position)] $
#pause
$ alpha_(u t) = (exp(e_(u t))) / (sum_(j=1)^T exp(e_(u j))) quad quad #text(size: 15pt, fill: MUTED)[(softmax over source positions)] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[high score $-> $ high weight $-> $ that source state dominates the context])

== Scoring functions #D

Three standard ways to score a query $s$ against a key $h$:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, center, left),
  table.header([*Name*], [*score$(s, h)$*], [*Note*]),
  [dot], [$s^top h$], [no parameters; needs $dim s = dim h$ (Luong)],
  [general / bilinear], [$s^top W h$], [a learned interaction matrix $W$ (Luong)],
  [additive / MLP], [$v^top tanh(W_s s + W_h h)$], [a small MLP (Bahdanau)],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[Bahdanau 2015 = *additive*; Luong 2015 = *dot* / *general* — all learned end-to-end])

== From context to prediction #D

Combine the context with the decoder state, then read out the next token:
#pause
$ c_u = sum_(t=1)^T alpha_(u t) thin h_t $
#pause
$ p(y_u | y_(<u), x_(1:T)) = "softmax"(W_o thin [s_u; c_u] + b_o) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the readout sees *both* the decoder state $s_u$ *and* the freshly-attended source context $c_u$])

== The attention mechanism #V

Query the encoder states, softmax the scores into weights, take the weighted sum:
#pause
#attnmech
#pause
#align(center, text(size: 15pt, fill: MUTED)[thicker edge = larger $alpha$; here step $u$ attends mostly to $h_3$, so $c_u approx h_3$])

== Attention is a differentiable lookup #D

Rename the pieces and attention becomes a *soft dictionary lookup* — the language of Transformers:
#pause
#two(
  notebox[*query* $q = s_(u-1)$ — what the decoder is looking for.],
  notebox[*keys* $k_t = h_t$, *values* $v_t = h_t$ — what the source offers.],
)
#pause
$ alpha_t = "softmax"_t (q^top k_t), quad quad "output" = sum_(t=1)^T alpha_t thin v_t $

== The lookup, drawn #V

#softlookup
#pause
#align(center, text(size: 15pt, fill: MUTED)[a *hard* lookup returns one value; attention returns a *soft, weighted* blend — and is differentiable, so it *learns*])

// ═══════════════════════════ PART VIII — Worked attention ═══════════════════════════
= A worked attention step

== Scores, then softmax #D

Scalar encoder states $h = [1, 2, 4]$ and a query $q = 2$. Dot-product scores:
#pause
$ e_t = q dot h_t: quad e = [2, 4, 8] $
#pause
Softmax the scores into weights:
$ alpha = "softmax"([2, 4, 8]) approx [0.0024, 0.0179, 0.9796] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the score gap of $8$ vs $4$ is *exponentiated* — step attends almost entirely to $h_3$])

== The context vector #V

Weighted sum of the states — the context lands near the most-attended one:
#pause
$ c = sum_t alpha_t h_t approx 0.0024(1) + 0.0179(2) + 0.9796(4) approx 3.96 $
#pause
#fig("/lecture14/figures/attention_weights.svg", w: 66%)

== Temperature: sharp vs diffuse #V

Softmax with temperature $tau$: $alpha = "softmax"(e \/ tau)$ — small $tau$ sharpens, large $tau$ spreads:
#pause
#fig("/lecture14/figures/temperature_attention.svg", w: 74%)
#pause
#align(center, text(size: 15pt, fill: MUTED)[Transformers use *scaled* dot-product $q^top k \/ sqrt(d)$ — the $sqrt(d)$ keeps scores from saturating the softmax])

// ═══════════════════════════ PART IX — Alignment & interpretability ═══════════════════════════
= Alignment and interpretability

== The attention matrix #V

Stack the weights over all steps: $A in RR^(U times T)$, with $A_(u t) = alpha_(u t)$. Each row is one output step's distribution over the source:
#pause
#fig("/lecture14/figures/attention_heatmap.svg", w: 47%)

== It learns an alignment #D

The bright cells trace a soft *word alignment* — learned, never supervised:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*target token*], [*attends to source*]),
  [`j'aime`], [`like` (and a little `I`)],
  [`les`], [`cats` (the plural determiner)],
  [`chats`], [`cats`],
  [`<EOS>`], [`<EOS>`],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[attention discovers *which source word each target word corresponds to* — for free])

== Attention is not always explanation #I

#pause
#alertbox[*Caution.* Attention weights are a useful *diagnostic* of what the model looked at — but they are *not always a faithful explanation*. Different weightings can give the same output, and high weight $!=$ causal importance (Jain & Wallace 2019).]
#pause
#interbox(link-to: IA + "attention")[
  Hover the alignment heatmap for "I like cats" $-> $ "j'aime les chats": watch each target row light up the source words it attends to, and edit the sentence to see the alignment shift.
]

// ═══════════════════════════ PART X — Full seq2seq + attention ═══════════════════════════
= The full seq2seq + attention model

== Encoder: keep every state #D

The encoder is unchanged — but now we *keep all* the states, not just the last one:
#pause
$ h_t = f_"enc" (h_(t-1), e(x_t)), quad quad H = (h_1, dots, h_T) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[$H$ is the memory the decoder attends into — the fixed $c = h_T$ is gone])

== Decoder with attention #D

Each decoder step: update the state, attend over $H$, predict from state $+$ context:
#pause
$ s_u = f_"dec" (s_(u-1), e(y_(u-1)), c_(u-1)) $
#pause
$ e_(u t) = "score"(s_u, h_t), quad quad alpha_(u t) = "softmax"_t (e_(u t)), quad quad c_u = sum_(t=1)^T alpha_(u t) h_t $
#pause
$ p(y_u | dot) = "softmax"(W_o thin [s_u; c_u] + b_o) $

== The full model #V

Encoder produces $H$; every decoder step queries it for a fresh context $c_u$:
#pause
#fullmodel
#pause
#align(center, text(size: 15pt, fill: MUTED)[the attention block runs *once per output step* — a new $c_u$ each time, chosen by the query $s_u$])

== One objective, gradients everywhere #D

Same masked cross-entropy as before — but now the gradient reaches *further*:
#pause
$ cal(L) = - sum_(u=1)^U log p(y_u | y_(<u), x_(1:T)) $
#pause
#notebox[Backprop flows: decoder readout $-> $ attention weights $alpha_(u t)$ $-> $ encoder states $h_t$ $-> $ the encoder RNN. Attention is *fully differentiable*, so the alignment is *learned* end-to-end from the translation loss alone.]

// ═══════════════════════════ PART XI — Implementation ═══════════════════════════
= Implementation

== Batching variable lengths #D

Sequences differ in length, so *pad* to a common $T$ and track a *source mask*:
#pause
$ m_t = cases(1 quad x_t != #raw("<PAD>"), 0 quad x_t = #raw("<PAD>")) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the mask marks which source positions are real — attention must not attend to padding])

== Masked attention #D

Padding positions must get *zero* weight. Set their scores to $-oo$ *before* the softmax:
#pause
$ e_(u t) <- cases(e_(u t) quad & m_t = 1, -oo quad & m_t = 0) quad => quad alpha_(u t) = 0 #text[ on padding] $
#pause
#notebox[$exp(-oo) = 0$, so masked positions drop out of the softmax cleanly and the weights still sum to $1$ over the *real* tokens. (The same trick gives the *causal* mask in Transformers.)]

== Teacher-forcing loop #V

Train step: feed the *true* previous target each step, accumulate masked cross-entropy:
#pause
#codebox(size: 12.5pt)[```python
H, enc_state = encoder(src, src_mask)   # keep ALL states H for attention
dec_in = BOS.expand(batch)              # first decoder input
state = init_from(enc_state); loss = 0
for u in range(U):                      # teacher forcing along the target
    logits, state, alpha = decoder(dec_in, state, H, src_mask)
    loss += cross_entropy(logits, tgt[:, u], ignore_index=PAD)
    dec_in = tgt[:, u]                  # feed the TRUE token, not argmax
loss.backward()
```]

== Greedy decode & beam #V

Inference: feed the model's *own* output back; or keep $B$ hypotheses:
#pause
#two(
  codebox(size: 11.5pt)[```python
# greedy
dec_in = BOS; out = []
for _ in range(max_len):
  logits, state, _ = decoder(
      dec_in, state, H, src_mask)
  dec_in = logits.argmax(-1)
  if dec_in == EOS: break
  out.append(dec_in)
```],
  codebox(size: 11.5pt)[```python
# beam search (B hypotheses)
beams = [Hyp(tokens=[BOS], state=s0,
             score=0.0, done=False)]
# each step: expand every beam,
#   score = cum log-prob,
#   keep top-B, stop when all done
```],
)

// ═══════════════════════════ PART XII — Limitations + summary ═══════════════════════════
= Limitations and summary

== What attention fixed — and did not #D

#pause
#two(
  notebox[*Fixed* — the bottleneck. #linebreak() before: $x -> c = h_T -> y$. #linebreak() after: $x -> H$, and $c_u = sum_t alpha_(u t) h_t$ *every step*.],
  alertbox[*Not fixed* — recurrence. The decoder is *still* an RNN: $s_u = f(s_(u-1), y_(u-1), c_(u-1))$ — target generation stays *sequential*.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[attention removed *one* bottleneck; the *sequential* encoder and decoder remain])

== The remaining costs #D

#pause
- the encoder is $O(T)$ *sequential* — cannot parallelize over source time;
- the decoder is $O(U)$ *sequential* — each $s_u$ needs $s_(u-1)$;
- long-range dependencies still travel through many recurrent steps.
#pause
#result[if attention can access any position directly — do we even *need* recurrence?]

== What changed, across three lectures #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 7pt), align: (left, center, left),
  table.header([*Model*], [*Conditional*], [*Context used*]),
  [RNN LM], [$p(y_t | y_(<t))$], [own past only],
  [seq2seq], [$p(y_t | y_(<t), x)$], [one fixed $c = h_T$],
  [seq2seq $+$ attention], [$p(y_t | y_(<t), sum_t alpha h_t)$], [per-step context $c_u$],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[each row adds access: to the source, then to *every source position, dynamically*])

== One-slide summary #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Idea*], [*Takeaway*]),
  [encoder–decoder], [encode source $-> c$; decode target autoregressively from $c$],
  [teacher forcing], [feed true $y_(u-1)$ in training — stable, but *exposure bias*],
  [decoding], [greedy / beam search; length-normalize the score],
  [bottleneck], [one fixed $c = h_T$ loses long, structured sources],
  [attention], [$c_u = sum_t alpha_(u t) h_t$ — per-step, source-dependent context],
  [scoring], [dot / general / additive; softmax; query–key–value lookup],
))

== Final mental model — one idea

#align(center, text(size: 21pt)[
  The *encoder* represents the source; the *decoder* generates the target.
  #v(6pt)
  *Teacher forcing* trains it; *attention* lets each step read the source *dynamically*.
  #v(6pt)
  #text(size: 17pt, fill: MUTED)[attention = a differentiable, weighted lookup over all source states]
])

#focus-slide[
  Attention removed the bottleneck — but recurrence remains.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Transformers* — remove recurrence entirely and make *attention the main computation* (queries, keys, values, self-attention, causal masks, positional encodings).
]
