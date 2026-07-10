// Next-Token Prediction — Lecture 11 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture11/L11-next-token.typ /tmp/L11.pdf
//   typst compile --root . --input handout=true lecture11/L11-next-token.typ /tmp/L11h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [Next-Token Prediction],
  subtitle: [Tokens, embeddings and neural language models],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let cream = rgb("#EFEEEB")

// ── causal / autoregressive: predicting x_t may look only at the past ──
#let causaldiag = align(center, diagram(spacing: (13mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let seen(x, lbl) = node((x, 0), text(size: 14pt, lbl), radius: 6.5mm, fill: cream, stroke: 0.9pt + INK)
  let future(x, lbl) = node((x, 0), text(size: 14pt, fill: MUTED, lbl), radius: 6.5mm, fill: white, stroke: (paint: MUTED, dash: "dashed"))
  seen(0, $x_1$); seen(1, $x_2$); seen(2, $x_3$)
  node((3, 0), text(size: 14pt, fill: white, $x_4$), radius: 6.5mm, fill: ACC, stroke: 0.9pt + ACC)
  future(4, $x_5$); future(5, $x_6$)
  // causal arrows: only the past feeds the prediction of x_4
  for i in (0, 1, 2) { edge((i, 0), (3, 0), "-|>", bend: 32deg, stroke: 0.8pt + TEAL) }
  // the "now" boundary
  edge((2.5, -0.9), (2.5, 0.9), stroke: (paint: RED, dash: "dashed", thickness: 1pt))
  node((2.5, 1.15), text(size: 11pt, fill: RED)[now], stroke: none, fill: none)
  node((4.5, -0.95), text(size: 11pt, fill: MUTED)[future — never seen], stroke: none, fill: none)
}))

// ── token id -> embedding table -> a single row ──
// rounded-rect nodes so the two-line labels sit fully inside
#let lookupdiag = align(center, diagram(spacing: (34mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), [token id \ #text(size: 13pt, fill: MUTED)[$= 5$]], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 8pt, fill: cream)
  node((1, 0), [$E$ \ #text(size: 12pt, fill: MUTED)[$V times d$ table]], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 8pt, fill: TEAL.lighten(80%), stroke: 0.9pt + TEAL)
  node((2, 0), [row 5 \ #text(size: 12pt, fill: MUTED)[$= e_5 in RR^d$]], shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 8pt, fill: ACC.lighten(80%), stroke: 0.9pt + ACC)
  edge((0, 0), (1, 0), "-|>", stroke: 0.8pt + MUTED, label: text(size: 11pt)[index], label-side: center)
  edge((1, 0), (2, 0), "-|>", stroke: 0.8pt + MUTED, label: text(size: 11pt)[select], label-side: center)
}))

// ── the makemore MLP language model (Bengio 2003) architecture ──
#let mlplm = align(center, diagram(spacing: (13mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let idn(y, lbl) = node((0, y), text(size: 12pt, lbl), fill: cream, corner-radius: 3pt, inset: 6pt)
  let emn(y, lbl) = node((1.05, y), text(size: 12pt, lbl), fill: TEAL.lighten(80%), stroke: 0.9pt + TEAL, corner-radius: 3pt, inset: 6pt)
  idn(1, $x_(t-3)$); idn(0, $x_(t-2)$); idn(-1, $x_(t-1)$)
  emn(1, $e_(t-3)$); emn(0, $e_(t-2)$); emn(-1, $e_(t-1)$)
  node((2.25, 0), [concat \ #text(size: 10pt, fill: MUTED)[$c in RR^(k d)$]], shape: fletcher.shapes.rect, fill: cream, corner-radius: 4pt, inset: 7pt)
  node((3.55, 0), [hidden \ #text(size: 10pt, fill: MUTED)[$h = phi(W_h c + b_h)$]], shape: fletcher.shapes.rect, fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE, corner-radius: 4pt, inset: 7pt)
  node((4.95, 0), [logits \ #text(size: 10pt, fill: MUTED)[$z in RR^V$]], shape: fletcher.shapes.rect, fill: ACC.lighten(80%), stroke: 0.9pt + ACC, corner-radius: 4pt, inset: 7pt)
  node((6.1, 0), [softmax \ #text(size: 10pt, fill: MUTED)[$p(x_t | dot)$]], shape: fletcher.shapes.rect, fill: rgb("#FDECD6"), stroke: 0.9pt + ACC, corner-radius: 4pt, inset: 7pt)
  for y in (1, 0, -1) {
    edge((0, y), (1.05, y), "-|>", stroke: 0.7pt + MUTED)
    edge((1.05, y), (2.25, 0), "-|>", stroke: 0.7pt + MUTED)
  }
  edge((2.25, 0), (3.55, 0), "-|>", stroke: 0.9pt + INK)
  edge((3.55, 0), (4.95, 0), "-|>", stroke: 0.9pt + INK)
  edge((4.95, 0), (6.1, 0), "-|>", stroke: 0.9pt + INK)
  node((0, 1.95), text(size: 11pt, fill: MUTED)[token ids], stroke: none, fill: none); node((1.05, 1.95), text(size: 11pt, fill: MUTED)[$E$ lookup], stroke: none, fill: none)
}))

// ── the autoregressive generation loop: context -> model -> p_t -> sample -> append -> repeat ──
#let genloop = align(center, diagram(spacing: (17mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let rn(pos, lbl, c, fill) = node(pos, text(size: 13pt, lbl), shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 7pt, stroke: 0.9pt + c, fill: fill)
  rn((0, 0), [context], INK, cream)
  rn((1, 0), [model], BLUE, BLUE.lighten(82%))
  rn((2, 0), [$p_t$], ACC, ACC.lighten(80%))
  rn((3, 0), [sample $hat(x)_t$], GREEN, GREEN.lighten(80%))
  edge((0, 0), (1, 0), "-|>", stroke: 0.7pt + MUTED, label: text(size: 10pt, fill: MUTED)[run])
  edge((1, 0), (2, 0), "-|>", stroke: 0.7pt + MUTED)
  edge((2, 0), (3, 0), "-|>", stroke: 0.7pt + MUTED, label: text(size: 10pt, fill: MUTED)[sample])
  edge((3, 0), (0, 0), "-|>", bend: -36deg, stroke: 0.9pt + GREEN)
  node((1.5, 1.02), text(size: 10pt, fill: GREEN)[append $hat(x)_t$, drop oldest — *repeat*], stroke: none, fill: white)
}))

#title-slide()

// ═══════════════════════════ PART I — What is sequence modeling ═══════════════════════════
= What is sequence modeling?

== Where we are

The vision lectures took a *fixed-shape* input — an image tensor:
#pause
$ X in RR^(H times W times C) $
#pause
Every example had the *same* shape; a CNN mapped it to one label.
#pause
#notebox[This lecture changes the *input type*. We move from fixed tensors to *sequences* of variable length — the first step toward language models.]

== Images are fixed; sequences are ordered

An image is a fixed grid. A *sequence* is an ordered list of tokens $x_1, x_2, dots, x_T$:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*Sequence*], [*Tokens $x_t$*]),
  [text], [characters, subwords, or words],
  [audio], [waveform samples / frames],
  [sensors], [readings over time],
  [events / logs], [user clicks, actions],
  [DNA], [bases A, C, G, T],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[length $T$ *varies* from example to example — a fixed-size vector no longer fits])

== Order carries meaning

The *same* tokens in a *different order* mean different things:
#pause
#align(center, text(size: 20pt, fill: INK)[
  "dog bites man" $quad != quad$ "man bites dog"
])
#pause
#alertbox[A model $f$ over sequences must *preserve order*. Treating $x_(1:T)$ as an unordered bag throws away exactly the structure that makes it a sequence.]

== The sequence tasks #V

A handful of task shapes cover most of sequence modeling:
#pause
#fig("/lecture11/figures/task_taxonomy.svg", w: 80%)

== The sequence tasks, precisely

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 6pt), align: (left, center, left),
  table.header([*Task*], [*Shape*], [*Example*]),
  [classification], [$x_(1:T) -> y$], [sentiment of a review],
  [token classification], [$x_(1:T) -> y_(1:T)$], [tag each word (POS)],
  [seq2seq], [$x_(1:T) -> y_(1:U)$], [translate a sentence],
  [*next-token*], [$x_(1:t) -> x_(t+1)$], [*predict what comes next*],
))
#pause
#result[this lecture: *next-token prediction* — the task behind language models]

== Our first task

Given a prefix, predict what comes next:
#pause
#align(center, text(size: 20pt, fill: INK)[
  "deep learning is" $quad -->^? quad$ ???
])
#pause
The model outputs a *probability distribution* over the whole vocabulary $V$:
$ p_theta (x_(t+1) = v | x_(1:t)) quad "for every" v in V $
#pause
#align(center, text(size: 16pt, fill: MUTED)[not a single guess — a *score for every possible next token*])

== Running example: makemore names

Our anchor all lecture: a *character-level name generator*.
#pause
- predict the next *character* of a name: `emma`, `olivia`, `ava`, `noah`;
#pause
- tiny vocabulary — 26 letters plus one special token;
#pause
- embeddings small enough to *see*, training fast enough to *play with*.
#pause
#notebox[This is Karpathy's *makemore* (part 2), which is exactly the neural language model of *Bengio et al. 2003, "A Neural Probabilistic Language Model"*. We build up to that model in this lecture.]

// ═══════════════════════════ PART II — Sequence probability ═══════════════════════════
= The probability of a sequence

== A model assigns probability to sequences

A language model is really a *probability distribution* over sequences:
#pause
$ p_theta (x_1, x_2, dots, x_T) $
#pause
How can one model score a sequence of *arbitrary length*? Factor it into pieces.

== The chain rule of probability #D

Any joint distribution factorizes into a product of conditionals:
#pause
$ p(x_1, dots, x_T) = p(x_1) thin p(x_2 | x_1) thin p(x_3 | x_1, x_2) dots p(x_T | x_(1:T-1)) $
#pause
Compactly, with $x_(<t) = x_(1:t-1)$:
$ p(x_(1:T)) = product_(t=1)^T p(x_t | x_(<t)) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[exact — no approximation. This holds for *any* sequence distribution.])

== Why next-token is enough

Read the chain rule the other way around:
#pause
#notebox[If we can model *every* conditional $p(x_t | x_(<t))$, we have modelled the *whole* sequence. Each factor is a next-token prediction.]
#pause
#result[language modeling $=$ repeated classification]
#pause
#align(center, text(size: 16pt, fill: MUTED)[each step: a $V$-way classification of "which token comes next?"])

== The training objective falls out #D

Take the log of the chain-rule product — products become sums:
#pause
$ log p(x_(1:T)) = sum_(t=1)^T log p(x_t | x_(<t)) $
#pause
Maximizing likelihood $=$ minimizing the *negative log-likelihood*:
$ cal(L) = - sum_(t=1)^T log p_theta (x_t | x_(<t)) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[this is exactly *token-level cross-entropy* — the same loss as classification])

== Every position is a training example #D

One name yields *many* supervised (context $->$ target) pairs. For `emma`:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 5.5pt), align: (left, left),
  table.header([*context* $x_(<t)$], [*target* $x_t$]),
  [`.` #text(fill: MUTED)[(start)]], [`e`],
  [`. e`], [`m`],
  [`. e m`], [`m`],
  [`. e m m`], [`a`],
  [`. e m m a`], [`.` #text(fill: MUTED)[(end)]],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[one 4-letter name $-> 5$ training pairs. A corpus gives *millions*.])

== Causal / autoregressive #V

Predicting $x_t$ may look at the *past* $x_(<t)$ — never at $x_t$ or the future:
#pause
#causaldiag
#pause
#align(center, text(size: 16pt, fill: MUTED)[*autoregressive*: each token is predicted from the tokens before it])

// ═══════════════════════════ PART III — Tokens & vocabulary ═══════════════════════════
= Tokens and vocabulary

== Text must become integers

A neural network consumes *numbers*, not strings. So we fix a *vocabulary* and map each token to an integer *id*:
#pause
$ "cat" -> 17, quad "dog" -> 18, quad "." -> 0 $
#pause
- the vocabulary $V$ is the fixed set of tokens the model knows;
#pause
- a sequence of text becomes a sequence of *token ids*.
#pause
#align(center, text(size: 16pt, fill: MUTED)[makemore: $V = { space . space, a, b, dots, z }$ — 27 symbols])

== What counts as a token?

The *granularity* of tokenization is a design choice:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 6pt), align: (left, center, center),
  table.header([*Unit*], [*Vocabulary size*], [*Sequence length*]),
  [character], [tiny (~$10^2$)], [very long],
  [word], [huge (~$10^5$+)], [short],
  [subword], [moderate (~$10^4$)], [moderate],
))
#pause
#notebox[Subword tokenization (e.g. *BPE*) is the modern default — a middle ground. We won't deep-dive it here; makemore just uses *characters*.]

== Special tokens

Beyond ordinary symbols, we reserve a few *control* tokens:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*Token*], [*Role*]),
  [`<BOS>` / `.`], [marks the *start* of a sequence],
  [`<EOS>` / `.`], [marks the *end* — lets the model stop],
  [`<PAD>`], [fills short sequences to a common length],
  [`<UNK>`], [stands in for out-of-vocabulary tokens],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[makemore reuses a single `.` as both start and end marker])

== A token id is not a number

The id is a *name*, not a *quantity*:
#pause
$ "cat" -> 17, quad "dog" -> 18 quad #text(fill: RED)[but] quad "dog" - "cat" != 1 $
#pause
#alertbox[Ids are *categorical* indices. Token 18 is not "one more than" token 17 — the integers carry *no* magnitude or order. Feeding the raw id as a scalar feature would invent a meaningless numeric scale.]
#pause
#result[we need a representation with *no* spurious arithmetic — next]

// ═══════════════════════════ PART IV — One-hot & embeddings ═══════════════════════════
= One-hot vectors and embeddings

== One-hot: the honest encoding

Represent token $i$ as a vector that is *1* in position $i$, *0* elsewhere:
#pause
$ o_i in RR^V, quad (o_i)_j = cases(1 quad j = i, 0 quad "otherwise") $
#pause
- no false ordering — every token is a distinct axis;
#pause
- but *high-dimensional* ($V$ can be $"50,000"$) and *sparse*;
#pause
- and *no notion of similarity* — every pair is equally far apart.
#pause
#align(center, text(size: 16pt, fill: MUTED)[$norm(o_i - o_j)^2 = 2$ for *every* distinct pair — "cat" is as close to "dog" as to "the"])

== One-hot vs dense embedding #V

#fig("/lecture11/figures/one_hot_vs_embed.svg", w: 80%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[replace a long sparse indicator with a short *dense* vector of learned features])

== Embeddings: a learned dense vector

Give each token a *dense* vector $e_i in RR^d$ with $d << V$. Stack them as rows of a matrix:
#pause
$ E in RR^(V times d), quad quad e_i = E_(i, :) quad (#text[the $i$-th row]) $
#pause
- $V = 27, d = 2$ — small enough to *plot* (makemore);
#pause
- $V = "50,000", d = 256$ — a realistic language model.
#pause
#align(center, text(size: 16pt, fill: MUTED)[from a $V$-dim sparse axis to a $d$-dim dense point — $d$ is the *embedding dimension*])

== Lookup is a matrix multiply #D

Selecting row $i$ of $E$ *is* a linear operation on the one-hot vector:
#pause
$ e_i = E^top o_i $
#pause
#lookupdiag
#pause
#align(center, text(size: 16pt, fill: MUTED)[in code it is a *table lookup* — but mathematically it is linear, so gradients flow through it])

== Worked example: embedding lookup #D

Four tokens `[I, like, cats, dogs]`, embedding dimension $d = 2$:
#pause
$ E = mat(1, 0; 0, 1; 1, 1; -1, 1), quad quad e_"I" = mat(1, 0), quad e_"like" = mat(0, 1) $
#pause
Context `"I like"` looks up two rows:
$ (e_"I", e_"like") = ((1, 0), (0, 1)) $
#pause
#align(center, text(size: 16pt, fill: BLUE)[*Check:* what does context `"cats dogs"` look up?])
#pause
#align(center, text(size: 16pt, fill: MUTED)[rows $3$ and $4$: $(e_"cats", e_"dogs") = ((1, 1), (-1, 1))$ — two dense rows out, the whole embedding layer])

== The embedding table E #V

$E$ is just a matrix of parameters; a context selects a few *rows*:
#pause
#fig("/lecture11/figures/embedding_table.svg", w: 42%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[looking up `e` and `o` picks out two rows of $E$ — nothing else is touched])

== Embeddings are parameters #D

$E$ starts as *small random noise* and is *learned* by gradient descent, like any weight:
#pause
$ E <- E - eta thin nabla_E cal(L) $
#pause
#notebox[Only the *rows that were looked up* in a batch receive a gradient — an unseen token's embedding does not move on that step. Frequent tokens get trained the most.]

== Embedding geometry #V

Trained embeddings place *similar-context* tokens *near* each other:
#pause
#fig("/lecture11/figures/embedding_2d.svg", w: 52%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[in makemore, vowels and consonants tend to cluster — measured by *cosine similarity*])

== A caveat on embedding geometry

#pause
- similarity emerges from *usage* — tokens used in like contexts drift together;
#pause
- cosine similarity $cos(e_i, e_j) = (e_i^top e_j) / (norm(e_i) norm(e_j))$ measures it;
#pause
#alertbox[Do *not* over-promise. Not every direction is a clean semantic axis, and small models learn only coarse structure. "Vowels cluster" is a *tendency*, not a law.]

== Interactive: embedding lookup #I

#interbox[#text(fill: MUTED)[(to build)] — click a token id and watch the corresponding *row* of $E$ light up and drop out as the dense embedding vector; nudge a value and see the point move in 2-D.]

// ═══════════════════════════ PART V — Fixed-context MLP language model ═══════════════════════════
= A fixed-context MLP language model

== The full history is too much

Exactly, $p(x_t | x_(<t))$ conditions on *all* previous tokens — an unbounded input.
#pause
Simplify: condition on only the previous $k$ tokens (a *fixed context window*):
$ p_theta (x_t | x_(<t)) approx p_theta (x_t | x_(t-k : t-1)) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[a bounded, fixed-size input $-> $ an ordinary MLP can handle it. This is the Bengio-2003 / makemore model.])

== The sliding-window dataset #D

Slide a window of length $k$ over each name to make (context $->$ target) pairs. `emma`, $k = 3$:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 5.5pt), align: (left, left),
  table.header([*context* ($k = 3$)], [*target*]),
  [`. . .`], [`e`],
  [`. . e`], [`m`],
  [`. e m`], [`m`],
  [`e m m`], [`a`],
  [`m m a`], [`.` #text(fill: MUTED)[(end)]],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[pad the start with `.`; every window is one supervised example])

== The MLP language model #V

#mlplm
#pause
#align(center, text(size: 15pt, fill: MUTED)[look up $-> $ concatenate $-> $ hidden layer $-> $ logits $-> $ softmax over $V$])

== The model, in equations #D

Look up each context token's embedding, then run an MLP:
#pause
$ e_(t-j) = E[x_(t-j)], quad quad c_t = [e_(t-k); dots; e_(t-1)] in RR^(k d) $
#pause
$ h = phi(W_h thin c_t + b_h), quad quad z = W_o thin h + b_o $
#pause
$ p = "softmax"(z) in RR^V $
#pause
#align(center, text(size: 15pt, fill: MUTED)[Bengio et al. 2003 · Karpathy's makemore — embeddings + one hidden layer + softmax])

== Why concatenate, not average? #D

Concatenation *keeps position information*:
#pause
$ [e_"dog"; e_"bites"; e_"man"] quad != quad [e_"man"; e_"bites"; e_"dog"] $
#pause
#alertbox[*Averaging* the embeddings would give $1/k sum_j e_(t-j)$ — identical for any *reordering* of the context. That destroys order, the one thing a sequence model must keep.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[each position lands in its *own* slice of $c_t$, hitting its *own* columns of $W_h$])

== Tensor shapes

Track a batch of $B$ contexts through the model:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 5.5pt), align: (left, center, left),
  table.header([*Tensor*], [*Shape*], [*Meaning*]),
  [context ids $X$], [$B times k$], [$k$ token ids per row],
  [embeddings $E[X]$], [$B times k times d$], [one $d$-vector per token],
  [concat $C$], [$B times k d$], [flatten the $k$ embeddings],
  [hidden $H$], [$B times H_"dim"$], [after $W_h$ and $phi$],
  [logits $Z$], [$B times V$], [one score per vocab token],
))

== Parameter count #D

Three weight blocks — embedding, hidden, output:
#pause
$ underbrace(V d, "embedding") + underbrace(H dot k d + H, "hidden") + underbrace(V dot H + V, "output") $
#pause
where $H$ is the hidden width and $k$ the context length.
#pause
#align(center, text(size: 16pt, fill: MUTED)[note: the *embedding* and *output* blocks both scale with the vocabulary $V$])

== Worked parameter count #D

Take $V = "20,000", thin d = 128, thin k = 4, thin H = 512$:
#pause
*Embedding* — one $d$-vector per vocab item:
$ V d = "20,000" dot 128 = "2,560,000" $
#pause
*Hidden* — an $H times k d$ matrix plus $H$ biases (note $k d = 512$):
$ H dot k d + H = 512 dot 512 + 512 = "262,656" $
#pause
*Output* — a $V times H$ matrix plus $V$ biases:
$ V H + V = "20,000" dot 512 + "20,000" = "10,260,000" $

== Worked parameter count (cont.) #D

Add the three weight blocks:
#pause
$ underbrace("2,560,000", "embed") + underbrace("262,656", "hidden") + underbrace("10,260,000", "output") $
#pause
$ = "13,082,656" $
#pause
#result[total $approx 13.08$ M parameters]
#pause
#align(center, text(size: 16pt, fill: BLUE)[*Check:* which block shrinks most if we halve the vocabulary $V$?])
#pause
#align(center, text(size: 16pt, fill: MUTED)[embedding + output (both $prop V$) dominate — halving $V$ nearly *halves* the model])

// ═══════════════════════════ PART VI — Fully-worked forward + backward ═══════════════════════════
= A fully-worked forward and backward pass

== The tiny setup

Vocabulary `[I, like, cats, dogs]`, context `"I like"`, target `cats`, $d = 2$:
#pause
$ e_"I" = mat(1, 0), quad e_"like" = mat(0, 1), quad c = mat(1, 0, 0, 1)^top $
#pause
$ W_h = mat(1, 0, 0, 1; 0, 1, 1, 0), quad b_h = 0 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[small enough to do *entirely by hand* — and check against the earlier formulas])

== Forward: the hidden layer #D

Multiply row by row, then apply ReLU. Recall $c = mat(1, 0, 0, 1)^top$:
#pause
$ (W_h c)_1 = 1 dot 1 + 0 dot 0 + 0 dot 0 + 1 dot 1 = 2 $
#pause
$ (W_h c)_2 = 0 dot 1 + 1 dot 0 + 1 dot 0 + 0 dot 1 = 0 $
#pause
$ h = "ReLU"(mat(2, 0)) = mat(2, 0) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the first hidden unit fires; the second is switched off])

== Forward: logits and softmax #D

Suppose the output layer produces logits (for `[I, like, cats, dogs]`):
#pause
$ z = mat(-2, 0, 2, 1) $
#pause
Softmax first *exponentiates* each logit:
#pause
$ e^(-2) approx 0.135, quad quad e^0 = 1 $
#pause
$ e^2 approx 7.389, quad quad e^1 approx 2.718 $
#pause
Sum them to get the normalizer $Z$:
$ Z = 0.135 + 1 + 7.389 + 2.718 = 11.243 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[a bigger logit becomes an *exponentially* bigger weight — that is the "soft-max"])

== Forward: softmax probabilities (cont.) #D

Divide each exponential by the normalizer $Z = 11.243$:
#pause
$ p_"I" = 0.135 / 11.243 approx 0.012, quad quad p_"like" = 1 / 11.243 approx 0.089 $
#pause
$ p_"cats" = 7.389 / 11.243 approx 0.657, quad quad p_"dogs" = 2.718 / 11.243 approx 0.242 $
#pause
$ p = mat(0.012, 0.089, 0.657, 0.242), quad quad sum_i p_i = 1 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the model already puts most mass ($0.657$) on the correct token `cats`])

== Forward: the loss #D

Cross-entropy picks out the *target* probability. Target is `cats`:
#pause
$ cal(L) = - log p_"cats" = - log(0.657) approx 0.42 $
#pause
#result[$cal(L) approx 0.42$ nats]
#pause
#align(center, text(size: 16pt, fill: MUTED)[a confident-but-imperfect prediction; a perfect one would give $cal(L) = 0$])

== Backward: the output gradient #D

For softmax + cross-entropy, the gradient w.r.t. the logits is beautifully simple:
#pause
$ nabla_z cal(L) = p - y, quad quad y = mat(0, 0, 1, 0) quad (#text[one-hot target]) $
#pause
Only the target coordinate `cats` has $y = 1$; subtract term by term:
#pause
$ nabla_z cal(L) = mat(0.012 - 0, 0.089 - 0, 0.657 - 1, 0.242 - 0) $
#pause
$ nabla_z cal(L) = mat(0.012, 0.089, -0.343, 0.242) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[*push down* the three wrong logits, *push up* the correct one `cats` — that is all learning is here])

== Backward: down to the embeddings #D

The chain rule walks the gradient back through the MLP. First to the hidden layer:
#pause
$ nabla_h cal(L) = W_o^top thin nabla_z cal(L) quad quad #text(size: 14pt, fill: MUTED)[(pull the logit gradient through $W_o$)] $
#pause
Through the ReLU — only the units that *fired* in the forward pass pass a gradient:
$ nabla_a cal(L) = nabla_h cal(L) dot.o "ReLU"'(W_h c) $
#pause
Then back to the concatenated context $c$:
$ nabla_c cal(L) = W_h^top thin nabla_a cal(L) = mat(-0.234, 0, 0, -0.234) $
#pause
#align(center, text(size: 15pt, fill: MUTED)[the two *zeros* are the coordinates the switched-off hidden unit never touched])

== Backward: the embedding gradients (cont.) #D

Split $c = [e_"I"; e_"like"]$ back into the two rows that were looked up:
#pause
$ nabla_(e_"I") cal(L) = mat(-0.234, 0) quad quad #text(size: 14pt, fill: MUTED)[(first two coordinates of $nabla_c cal(L)$)] $
#pause
$ nabla_(e_"like") cal(L) = mat(0, -0.234) quad quad #text(size: 14pt, fill: MUTED)[(last two coordinates)] $
#pause
#align(center, text(size: 16pt, fill: BLUE)[*Check:* which of the four rows of $E$ move this step?])
#pause
#align(center, text(size: 15pt, fill: MUTED)[only rows `I` and `like` — `cats` and `dogs` were not in the context, so their gradient is $0$])

== Interactive: the MLP language model #I

#interbox[#text(fill: MUTED)[(to build)] — step through this exact example: watch the embeddings look up, the concat form, the softmax bar chart appear, and the $p - y$ gradient flow back and nudge the two used rows.]

// ═══════════════════════════ PART VII — Training objective & evaluation ═══════════════════════════
= Training objective and evaluation

== The dataset loss

Average the per-token NLL over every (context $->$ target) pair in the corpus:
#pause
$ cal(L) = - 1/(T - k) sum_(t) log p_theta (x_t | x_(t-k:t-1)) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the token-level cross-entropy from Part II — now over the whole sliding-window dataset])

== The minibatch loop

Train by the same loop as any classifier — on a minibatch $B$ of contexts:
#pause
$ cal(L)_B = - 1/abs(B) sum_(i in B) log p_theta (y_i | c_i) $
#pause
#align(center, diagram(spacing: 8mm, node-stroke: 0.8pt + INK, node-fill: white, {
  let b(x, lbl, col) = node((x, 0), text(size: 12pt, lbl), fill: col.lighten(82%), stroke: 0.8pt + col, corner-radius: 3pt, inset: 6pt)
  b(0, "contexts", INK); b(1, "embed", TEAL); b(2, "MLP", BLUE); b(3, "logits", ACC); b(4, "cross-entropy", RED); b(5, "backprop", GREEN); b(6, "update", INK)
  for i in range(6) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.7pt + MUTED) }
}))

== Padding and loss masks

Sequences in a batch have *different lengths*. Pad short ones to a common length:
#pause
- introduce a `<PAD>` token so every row is length $k$;
- but *do not* let padding contribute to the loss.
#pause
Use a mask $m_(i t) in {0, 1}$ ($1$ = real token, $0$ = pad):
$ cal(L) = (sum_(i, t) m_(i t) thin (- log p_(i t))) / (sum_(i, t) m_(i t)) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the mask makes padding *invisible* to the gradient — only real tokens count])

== Perplexity #D

Report the *per-token* NLL exponentiated — the *perplexity*:
#pause
$ "PPL" = exp("NLL") = exp(- 1/N sum_t log p_theta (x_t | x_(<t))) $
#pause
#notebox[Interpretation: the *effective number of plausible next tokens*. PPL $= 1$ means perfectly certain; PPL $= V$ means no better than a uniform guess.]

== Perplexity, two ways #V

#fig("/lecture11/figures/perplexity.svg", w: 74%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[a peaked distribution has *low* perplexity; a flat one is *perplexed* among many tokens])

== Perplexity: worked and cautioned #D

Quick sanity checks (natural log):
#pause
$ "NLL" = log 2 approx 0.69 quad => quad "PPL" = e^(log 2) = 2 $
#pause
$ "NLL" = log 10 approx 2.30 quad => quad "PPL" = e^(log 10) = 10 $
#pause
#alertbox[*Caveat.* Perplexity depends on the *tokenization*. A character model and a word model are *not* comparable by PPL — they predict different-sized units.]

// ═══════════════════════════ PART VIII — Autoregressive generation ═══════════════════════════
= Autoregressive generation

== Training vs generation #V

At *training* time the previous tokens are *true*; at *generation* time they are the model's *own* outputs, fed back in:
#pause
#align(center, diagram(spacing: (17mm, 13mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let rn(pos, lbl, c: INK, fill: white) = node(pos, text(size: 13pt, lbl),
    shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 7pt, stroke: 0.9pt + c, fill: fill)
  // generation row (top)
  node((-1.15, -1.4), text(size: 12pt, fill: MUTED)[gen], stroke: none, fill: none)
  rn((0, -1.4), [context], fill: cream)
  rn((1.4, -1.4), [model], c: BLUE, fill: BLUE.lighten(82%))
  rn((2.9, -1.4), [sample $hat(x)_t$], c: GREEN, fill: GREEN.lighten(80%))
  edge((0, -1.4), (1.4, -1.4), "-|>", stroke: 0.7pt + MUTED); edge((1.4, -1.4), (2.9, -1.4), "-|>", stroke: 0.7pt + MUTED)
  edge((2.9, -1.4), (0, -1.4), "-|>", bend: 32deg, stroke: 0.9pt + GREEN)
  node((1.45, -0.72), text(size: 10pt, fill: GREEN)[append, feed back], stroke: none, fill: white)
  // training row (bottom)
  node((-1.15, 0), text(size: 12pt, fill: MUTED)[train], stroke: none, fill: none)
  rn((0, 0), [true $x_(<t)$], fill: cream)
  rn((1.4, 0), [model], c: BLUE, fill: BLUE.lighten(82%))
  rn((2.9, 0), [$p(x_t)$], c: ACC, fill: ACC.lighten(80%))
  edge((0, 0), (1.4, 0), "-|>", stroke: 0.7pt + MUTED); edge((1.4, 0), (2.9, 0), "-|>", stroke: 0.7pt + MUTED)
}))

== The generation algorithm

One token at a time:
#pause
+ start from a context (e.g. just `.` for makemore);
#pause
+ run the model $-> $ next-token distribution $p_t$;
#pause
+ *select* or *sample* a token $hat(x)_t$ from $p_t$;
#pause
+ *append* it; drop the oldest token if the context is fixed-length;
#pause
+ repeat until `<EOS>` or a maximum length.
#pause
#v(6pt)
#genloop

== Greedy decoding

Always take the *most probable* token:
#pause
$ hat(x)_(t+1) = arg max_v thin p_t (v) $
#pause
#two(
  notebox[*Pro* — deterministic, simple, always picks the model's best guess.],
  alertbox[*Con* — repetitive and bland; locally-best $!=$ globally-best sequence.],
)

== Sampling

Draw the next token *at random* from the distribution:
#pause
$ hat(x)_(t+1) ~ "Categorical"(p_t) $
#pause
- respects the model's uncertainty — likely tokens are likely, but not forced;
#pause
- produces *diverse* outputs across runs;
#pause
- makemore sampling character-by-character gives *new* plausible names.
#pause
#align(center, text(size: 16pt, fill: MUTED)[`. -> e -> m -> m -> a -> .` yields `emma`; a different draw yields `elian`, `amara`, …])

== Temperature #V

Rescale the logits by $tau$ before the softmax:
#pause
$ p_i (tau) = "softmax"(z \/ tau)_i $
#pause
#fig("/lecture11/figures/temperature.svg", w: 74%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[$tau < 1$ sharpens (bolder) · $tau > 1$ flattens (more random) · $tau -> 0$ $-> $ greedy])

== Top-k sampling

Keep only the $k$ most probable tokens, *renormalize*, then sample:
#pause
- zero out everything outside the top $k$;
#pause
- rescale the survivors to sum to $1$;
#pause
- prevents rare, low-probability tokens from ever being drawn.
#pause
#align(center, text(size: 16pt, fill: MUTED)[nucleus / top-$p$ sampling is the close cousin — we meet it in the Transformer lecture])

== Interactive: next-token sampling #I

#interbox(link-to: IA + "softmax-temperature")[
  Sweep temperature and top-$k$ on a fixed set of logits and watch the next-token bar chart sharpen or flatten — then sample repeatedly and see the diversity of outputs change.
]

// ═══════════════════════════ PART IX — Where the MLP fails → RNN motivation ═══════════════════════════
= Where the fixed-context MLP fails

== It only sees a fixed window

The MLP conditions on exactly the last $k$ tokens — anything before $x_(t-k)$ is *invisible*:
#pause
#align(center, text(size: 18pt, fill: INK)[
  "The book that I placed on the table near the window $dots$ *was*"
])
#pause
#alertbox[To choose `was` vs `were` the model needs the subject *book* — but with a small window that word has already scrolled out of view. Long-range dependencies are lost.]

== Longer context costs parameters #D

Widening the window grows the hidden weight matrix:
#pause
$ W_h in RR^(H times k d) quad => quad H dot k d quad "parameters" $
#pause
- double the context $k$ $-> $ double the hidden-layer parameters;
#pause
- and the model *still* cannot accept a sequence longer than $k$.
#pause
#align(center, text(size: 16pt, fill: MUTED)[you cannot simply set $k = infinity$ — cost grows with the window, and $k$ is fixed at build time])

== Position-specific parameters #D

Each context slot hits a *different* block of columns of $W_h$:
#pause
$ c_t = [e_(t-k); dots; e_(t-1)], quad quad #text(size: 15pt)[slot $j$ $-> $ its own $d$-column block of $W_h$] $
#pause
#alertbox[The *same* token at position 1 vs position 2 is processed by *different* weights. Order is preserved, but computation is *not shared* across time — the model relearns each position separately.]

== What we actually want

A better sequence model should:
#pause
- accept *variable-length* input (no fixed $k$);
#pause
- *share* parameters across time steps;
#pause
- *carry* information from arbitrarily far back;
#pause
- emit an output at *every* step.
#pause
#result[a recurrent update: $h_t = f_theta (h_(t-1), x_t)$]

== From fixed window to recurrent state #V

#align(center, diagram(spacing: (11mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  // fixed window (left)
  node((1, 1.5), text(size: 13pt, fill: INK)[*fixed window (MLP)*], stroke: none, fill: none)
  for (i, l) in ((0, $x_1$), (1, $x_2$), (2, $x_3$), (3, $x_4$)) {
    let seen = i >= 1
    node((i*0.75, 0), text(size: 11pt, fill: if seen { INK } else { MUTED }, l), radius: 5mm,
      fill: if seen { cream } else { white }, stroke: if seen { 0.8pt + INK } else { (paint: MUTED, dash: "dashed") })
  }
  edge((0.75, -0.75), (2.25, -0.75), stroke: 1.2pt + ACC)
  node((1.5, -1.1), text(size: 10pt, fill: ACC)[window $k = 3$], stroke: none, fill: none)
  node((0, -0.7), text(size: 10pt, fill: MUTED)[dropped], stroke: none, fill: none)
  // recurrent (right)
  node((6, 1.5), text(size: 13pt, fill: INK)[*recurrent (RNN)*], stroke: none, fill: none)
  for (i, l) in ((0, $x_1$), (1, $x_2$), (2, $x_3$), (3, $x_4$)) {
    node((5 + i*0.9, -0.9), text(size: 11pt, l), radius: 5mm, fill: cream)
    node((5 + i*0.9, 0.3), text(size: 11pt, fill: white, $h_#(i+1)$), radius: 5mm, fill: TEAL, stroke: 0.9pt + TEAL)
    edge((5 + i*0.9, -0.9), (5 + i*0.9, 0.3), "-|>", stroke: 0.7pt + MUTED)
  }
  for i in range(3) { edge((5 + i*0.9, 0.3), (5 + (i+1)*0.9, 0.3), "-|>", stroke: 0.9pt + TEAL) }
  node((6.35, 1.0), text(size: 10pt, fill: TEAL)[state carries the past], stroke: none, fill: none)
}))

== The recurrent idea

Replace the fixed window with a *running state* that summarizes the past:
#pause
$ underbrace(p(x_(t+1) | x_(t-k:t)), "MLP: fixed window") quad --> quad underbrace(h_t = f_theta (h_(t-1), e_t), "RNN: recurrent state") $
#pause
$ p = "softmax"(W h_t + b) $
#pause
#notebox[The hidden state $h_t$ is a *learned summary* of everything seen so far — one shared function $f_theta$ applied at every step, for any length.]

== One-slide summary

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Idea*], [*Takeaway*]),
  [chain rule], [$p(x_(1:T)) = product_t p(x_t | x_(<t))$ — model every conditional],
  [language modeling], [repeated next-token classification; loss = token cross-entropy],
  [tokens], [text $-> $ integer ids; ids are categorical, not numeric],
  [embeddings], [learned dense rows of $E$; similar tokens land nearby],
  [MLP-LM], [fixed window $-> $ concat $-> $ MLP $-> $ softmax (Bengio / makemore)],
  [generation], [autoregressive; greedy / sample / temperature / top-$k$],
  [limitation], [fixed window, position-specific weights $-> $ motivates RNNs],
))

== Final mental model — one idea

#align(center, text(size: 22pt)[
  A language model estimates a *distribution over the next token.* \
  #v(4pt)
  Language modeling is *repeated classification* \
  #v(4pt)
  #text(size: 18pt, fill: MUTED)[tokens $-> $ embeddings $-> $ context $-> $ softmax over the vocabulary]
])

#focus-slide[
  We predicted the next token from a *fixed window*.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Sequence Models — RNNs* — hidden state, backprop through time, and LSTMs / GRUs that carry information across a whole sequence.
]
