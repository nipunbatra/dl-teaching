// Efficient Inference: Compute Less, Move Fewer Bytes
// Lecture 23 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture23/L23-efficient-inference.typ /tmp/L23.pdf
//   typst compile --root . --input handout=true lecture23/L23-efficient-inference.typ /tmp/L23h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.
// Schematic figures: python3 lecture23/diagrams/l23_figs.py

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [Efficient Inference: Compute Less, Move Fewer Bytes],
  subtitle: [At inference the model is often waiting on memory, not arithmetic],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let cream = rgb("#EFEEEB")
#let rect = fletcher.shapes.rect

// ═══════════════════════════ fletcher diagrams (centerpieces) ═══════════════════════════

// ── (A) the two phases: prompt -> prefill -> first token -> decode loop -> response ──
#let twophase = align(center, diagram(spacing: (13mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(pos, lbl, c, f) = node(pos, align(center, text(size: 10pt, lbl)), shape: rect, corner-radius: 4pt, inset: 7pt, stroke: 0.9pt + c, fill: f)
  node((0, 0), align(center, text(size: 10pt)[prompt \ $x_1 dots x_n$]), shape: rect, corner-radius: 4pt, inset: 7pt, stroke: 0.9pt + INK, fill: cream)
  bx((1.6, 0), [*prefill* \ all tokens \ in parallel], TEAL, TEAL.lighten(82%))
  bx((3.2, 0), [first \ token $y_1$], ACC, ACC.lighten(80%))
  bx((4.8, 0), [*decode* \ one token \ at a time], RED, RED.lighten(85%))
  node((6.4, 0), align(center, text(size: 10pt)[response \ $y_1 dots y_m$]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  for (a, b) in ((0, 1.6), (1.6, 3.2), (3.2, 4.8), (4.8, 6.4)) { edge((a, 0), (b, 0), "-|>", stroke: 0.8pt + MUTED) }
  edge((4.8, 0), (4.8, 0), "-|>", bend: 130deg, stroke: 0.9pt + RED)
  node((4.8, 1.15), text(size: 9.5pt, fill: RED)[repeat until `<EOS>`], stroke: none, fill: none)
  node((2.4, -1.15), text(size: 10pt, fill: TEAL)[compute-heavy, parallel], stroke: none, fill: none)
  node((5.6, -1.15), text(size: 10pt, fill: RED)[bandwidth-bound, sequential], stroke: none, fill: none)
}))

// ── (B) memory-vs-compute: stream every weight once to make ONE token ──
#let membound = align(center, diagram(spacing: (18mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), align(center, text(size: 10pt)[HBM \ *weights* \ $2P$ bytes]), shape: rect, corner-radius: 4pt, inset: 9pt, stroke: 1pt + BLUE, fill: BLUE.lighten(85%))
  node((2.3, 0), align(center, text(size: 10pt)[compute \ unit \ $tilde 2P$ FLOPs]), shape: rect, corner-radius: 4pt, inset: 9pt, stroke: 1pt + INK, fill: cream)
  node((4.5, 0), align(center, text(size: 10pt)[one \ token]), shape: rect, corner-radius: 4pt, inset: 8pt, stroke: 0.9pt + GREEN, fill: GREEN.lighten(82%))
  edge((0, 0), (2.3, 0), "-|>", stroke: 2pt + BLUE, label: text(size: 10pt, fill: BLUE)[move *all* weights], label-side: left)
  edge((2.3, 0), (4.5, 0), "-|>", stroke: 0.9pt + GREEN)
  node((2.3, 1.15), text(size: 10.5pt, fill: RED)[$approx 1$ FLOP per byte fetched $-> $ stalls on memory], stroke: none, fill: none)
}))

// ── (C) KV cache at decode: append new k,v; read the whole cached history ──
#let kvcache = align(center, diagram(spacing: (15mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  // cached slots (stack)
  node((0, 0.75), align(center, text(size: 9.5pt)[cached $K, V$ \ tokens $1 dots t-1$]), shape: rect, corner-radius: 4pt, inset: 7pt, stroke: 1pt + TEAL, fill: TEAL.lighten(82%))
  // new token
  node((0, -1.1), align(center, text(size: 9.5pt)[new token $x_t$ \ $-> q_t, k_t, v_t$]), shape: rect, corner-radius: 4pt, inset: 7pt, stroke: 1pt + ACC, fill: ACC.lighten(82%))
  // attention
  node((2.5, 0), align(center, text(size: 9.5pt)[attention \ $q_t$ over $K_(1:t), V_(1:t)$]), shape: rect, corner-radius: 4pt, inset: 8pt, stroke: 1pt + BLUE, fill: BLUE.lighten(85%))
  // output
  node((4.7, 0), align(center, text(size: 9.5pt)[$h_t$]), radius: 6mm, fill: INK.lighten(4%), stroke: 1pt + INK)
  node((4.7, 0), align(center, text(size: 11pt, fill: white)[$h_t$]), radius: 6mm, fill: INK.lighten(4%), stroke: 1pt + INK)
  edge((0, 0.75), (2.5, 0), "-|>", stroke: 0.9pt + TEAL, label: text(size: 9pt, fill: TEAL)[read], label-side: left)
  edge((0, -1.1), (2.5, 0), "-|>", stroke: 0.9pt + ACC, label: text(size: 9pt, fill: ACC)[$q_t$], label-side: right)
  edge((2.5, 0), (4.7, 0), "-|>", stroke: 0.9pt + INK)
  // append new k,v back to cache (bow LEFT, clear of the q_t / read edges)
  edge((0, -1.1), (0, 0.75), "-|>", stroke: 0.9pt + GREEN, bend: -52deg, label: text(size: 9pt, fill: GREEN)[append $k_t, v_t$], label-side: right, label-pos: 0.5)
}))

// ── (D) MHA vs GQA vs MQA: how many KV heads back the query heads ──
#let gqa = align(center, diagram(spacing: (6mm, 9mm), node-stroke: 0.7pt + INK, node-fill: white, {
  let q(x, c) = node((x, 1), none, radius: 2.6mm, fill: c, stroke: none)
  let kv(x, c) = node((x, -1), none, radius: 3.4mm, fill: c, stroke: 0.7pt + INK)
  // panel titles
  node((1.5, 2.15), text(size: 10.5pt, fill: TEAL, weight: 600)[MHA · 8 KV], stroke: none, fill: none)
  node((6.0, 2.15), text(size: 10.5pt, fill: BLUE, weight: 600)[GQA · 2 KV], stroke: none, fill: none)
  node((10.0, 2.15), text(size: 10.5pt, fill: ACC, weight: 600)[MQA · 1 KV], stroke: none, fill: none)
  // MHA: 8 query, 8 kv, one-to-one
  for i in range(4) { q(i, TEAL); kv(i, TEAL.lighten(55%)); edge((i, 1), (i, -1), stroke: 0.5pt + MUTED) }
  // GQA: 4 query share 2 kv (groups of 2)
  for i in range(4) { q(5 + i, BLUE) }
  kv(5.5, BLUE.lighten(55%)); kv(7.5, BLUE.lighten(55%))
  for i in range(2) { edge((5 + i, 1), (5.5, -1), stroke: 0.5pt + MUTED) }
  for i in range(2) { edge((7 + i, 1), (7.5, -1), stroke: 0.5pt + MUTED) }
  // MQA: 4 query share 1 kv
  for i in range(4) { q(9 + i, ACC); edge((9 + i, 1), (10.5, -1), stroke: 0.5pt + MUTED) }
  kv(10.5, ACC.lighten(55%))
  node((1.5, -2.1), text(size: 9pt, fill: MUTED)[full flexibility], stroke: none, fill: none)
  node((6.0, -2.1), text(size: 9pt, fill: MUTED)[$4 times$ smaller cache], stroke: none, fill: none)
  node((10.5, -2.1), text(size: 9pt, fill: MUTED)[smallest cache], stroke: none, fill: none)
}))

// ── (E) quantization flow: fp16 block -> scale/zero -> low-bit ints -> dequant ──
#let quantflow = align(center, diagram(spacing: (13mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(pos, lbl, c, f) = node(pos, align(center, text(size: 9.5pt, lbl)), shape: rect, corner-radius: 4pt, inset: 7pt, stroke: 0.9pt + c, fill: f)
  bx((0, 0), [fp16 block \ $W$ (16-bit)], BLUE, BLUE.lighten(85%))
  bx((1.7, 0), [find scale $s$, \ zero-point $z$], TEAL, TEAL.lighten(82%))
  bx((3.4, 0), [round to \ int4 / int8 \ $W_"int"$], ACC, ACC.lighten(80%))
  bx((5.2, 0), [store \ $W_"int" + (s, z)$], GREEN, GREEN.lighten(82%))
  for (a, b) in ((0, 1.7), (1.7, 3.4), (3.4, 5.2)) { edge((a, 0), (b, 0), "-|>", stroke: 0.8pt + MUTED) }
  node((3.4, -1.35), align(center, text(size: 9.5pt)[dequant at use: \ $W approx s (W_"int" - z)$]), shape: rect, corner-radius: 4pt, inset: 6pt, stroke: 0.9pt + INK, fill: cream)
  edge((5.2, 0), (3.4, -1.35), "-|>", stroke: 0.8pt + INK, bend: 22deg)
}))

// ── (F) MoE routing: token -> router -> top-2 of N experts -> weighted sum ──
#let moerouting = align(center, diagram(spacing: (14mm, 6.5mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), align(center, text(size: 9.5pt)[token \ $h$]), radius: 6.5mm, fill: cream, stroke: 0.9pt + INK)
  node((1.6, 0), align(center, text(size: 9.5pt)[router \ (softmax gate)]), shape: rect, corner-radius: 4pt, inset: 7pt, stroke: 1pt + BLUE, fill: BLUE.lighten(85%))
  edge((0, 0), (1.6, 0), "-|>", stroke: 0.8pt + MUTED)
  let ex(y, lbl, on) = node((3.4, y), align(center, text(size: 9pt, lbl)),
    shape: rect, corner-radius: 4pt, inset: 6pt,
    stroke: (if on { 1.4pt + ACC } else { 0.8pt + MUTED }),
    fill: (if on { ACC.lighten(78%) } else { white }))
  ex(2.1, [expert 1], true)
  ex(0.7, [expert 2], false)
  ex(-0.7, [expert 3], true)
  ex(-2.1, [expert 4], false)
  edge((1.6, 0), (3.4, 2.1), "-|>", stroke: 1.2pt + ACC, label: text(size: 8pt, fill: ACC)[0.6], label-side: left)
  edge((1.6, 0), (3.4, 0.7), "-|>", stroke: 0.5pt + MUTED)
  edge((1.6, 0), (3.4, -0.7), "-|>", stroke: 1.2pt + ACC, label: text(size: 8pt, fill: ACC)[0.4], label-side: right)
  edge((1.6, 0), (3.4, -2.1), "-|>", stroke: 0.5pt + MUTED)
  node((5.4, 0), align(center, text(size: 9pt)[weighted \ sum $-> y$]), shape: rect, corner-radius: 4pt, inset: 7pt, stroke: 0.9pt + GREEN, fill: GREEN.lighten(82%))
  edge((3.4, 2.1), (5.4, 0), "-|>", stroke: 1.2pt + ACC)
  edge((3.4, -0.7), (5.4, 0), "-|>", stroke: 1.2pt + ACC)
  node((3.4, -3.1), text(size: 9pt, fill: MUTED)[top-2 of 4 fire $-> $ 25% of expert FLOPs], stroke: none, fill: none)
}))

// ── (G) speculative decoding: draft proposes, target verifies in one pass ──
#let specdec = align(center, diagram(spacing: (11mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), align(center, text(size: 9.5pt)[*draft* model \ (small, cheap)]), shape: rect, corner-radius: 4pt, inset: 7pt, stroke: 1pt + TEAL, fill: TEAL.lighten(82%))
  // proposed chips
  let chip(x, lbl, c) = node((x, 1.6), align(center, text(size: 8.5pt, lbl)), shape: rect, corner-radius: 3pt, inset: 4pt, stroke: 0.9pt + c, fill: c.lighten(80%))
  chip(2.0, [$hat(y)_1$], GREEN); chip(2.8, [$hat(y)_2$], GREEN); chip(3.6, [$hat(y)_3$], GREEN); chip(4.4, [$hat(y)_4$], RED)
  edge((0, 0), (2.0, 1.6), "-|>", stroke: 0.8pt + TEAL, label: text(size: 8.5pt, fill: TEAL)[propose 4], label-side: left)
  node((3.2, 0), align(center, text(size: 9.5pt)[*target* model \ verify all in \ ONE pass]), shape: rect, corner-radius: 4pt, inset: 7pt, stroke: 1pt + INK, fill: cream)
  edge((2.0, 1.6), (3.2, 0), "-|>", stroke: 0.7pt + MUTED)
  node((5.7, 0), align(center, text(size: 9pt)[accept \ $hat(y)_1 hat(y)_2 hat(y)_3$ ✓ \ resample $hat(y)_4$]), shape: rect, corner-radius: 4pt, inset: 7pt, stroke: 0.9pt + GREEN, fill: GREEN.lighten(82%))
  edge((3.2, 0), (5.7, 0), "-|>", stroke: 0.9pt + INK)
}))

// ── (H) FlashAttention: tile K,V; online softmax; never write the T×T score matrix ──
#let flashattn = align(center, diagram(spacing: (13mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(pos, lbl, c, f) = node(pos, align(center, text(size: 9.5pt, lbl)), shape: rect, corner-radius: 4pt, inset: 7pt, stroke: 0.9pt + c, fill: f)
  bx((0, 0), [$Q, K, V$ \ in HBM], BLUE, BLUE.lighten(85%))
  bx((1.8, 0), [stream in \ *blocks*], TEAL, TEAL.lighten(82%))
  bx((3.6, 0), [running \ *online softmax* \ in SRAM], ACC, ACC.lighten(80%))
  bx((5.4, 0), [output \ $O$], GREEN, GREEN.lighten(82%))
  for (a, b) in ((0, 1.8), (1.8, 3.6), (3.6, 5.4)) { edge((a, 0), (b, 0), "-|>", stroke: 0.8pt + MUTED) }
  node((3.0, -1.2), align(center, text(size: 10pt, fill: RED)[the $T times T$ score matrix is *never* written to HBM]), stroke: none, fill: none)
}))

#title-slide()

// ═══════════════════════════ PART I — Diagnose before optimizing ═══════════════════════════
= Diagnose before optimizing

== Why this belongs in a deep-learning course

An accurate model that cannot meet its *latency, memory, or cost* budget is not usable.
#pause
#align(center, block(fill: white, inset: (x: 15pt, y: 11pt), radius: 5pt, stroke: 2pt + ACC,
  text(size: 20pt, weight: 600, fill: INK)[
    Moving weights and cached activations is often the bottleneck — not arithmetic.
  ]))
#pause
#notebox[Architecture, quality, and deployment are *coupled*. A design that ignores inference cost is only half a design. This lecture is the systems half.]
#pause
#align(center, text(size: 15pt, fill: MUTED)[one worked calculation (KV-cache memory) · one essential distinction (prefill vs decode)])

== The two phases of autoregressive inference #V

Generation is *not* one uniform computation:
#pause
#twophase
#pause
#align(center, text(size: 14pt, fill: MUTED)[processing the prompt and producing one token at a time have *different shapes, costs, and optimizations*])

== Prefill — process the prompt in parallel #D

Feed *all* $n$ prompt tokens through the model at once:
#pause
- large *matrix–matrix* products keep the accelerator busy;
- each weight is reused across *many* token vectors — high arithmetic intensity;
- attention cost still grows with sequence length ($O(n^2 d)$).
#pause
#align(center, text(size: 16pt, fill: MUTED)[prefill is typically *compute-bound* — the arithmetic units are the limit])

== Decode — one token at a time #D

Produce the next token (or a small batch), then repeat:
#pause
- each step needs the *full model weights* plus *all prior key/value state*;
- the work is *matrix–vector-like*: little reuse per fetched weight;
- steps are *sequential* — token $t+1$ waits for token $t$.
#pause
#align(center, text(size: 16pt, fill: MUTED)[decode is typically *memory-bound* — the device spends its time fetching weights and cache])

== Compute-bound versus memory-bound #D

Two different resources can be the ceiling:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 8pt), align: (left, left),
  table.header([*Regime*], [*What limits you*]),
  [compute-bound], [arithmetic units saturated; more FLOPs $=$ more time],
  [memory-bound], [device idles waiting to *fetch* weights / activations],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[optimize the *right* thing: shrinking FLOPs does nothing if you are memory-bound])

== The bytes intuition #V

For a single generated token, a huge matrix–vector product does *few operations per byte* of weight fetched:
#pause
#membound
#pause
#align(center, text(size: 15pt, fill: MUTED)[the accelerator can sit *idle* even with abundant FLOPs — it is waiting on the weight fetch])

== The roofline picture #V

Attainable throughput is capped by *bandwidth* until arithmetic intensity crosses the ridge:
#pause
#fig("/lecture23/figures/roofline.png", w: 54%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[*decode* (AI $approx 1$) sits deep in the memory-bound region; *prefill* (AI $approx T$) reaches the compute ceiling])

== The bytes intuition, said plainly #Q

#pause
#result[abundant FLOPs, starved bandwidth $-> $ the accelerator *waits on memory*]
#pause
#align(center, text(size: 17pt, fill: INK)[
  decode reads $approx 2P$ bytes to do $approx 2P$ FLOPs: about *one* FLOP per byte
])
#pause
#alertbox[This is why a device with "enough FLOPs" can still be slow at generation. The fix is rarely "more compute" — it is *move fewer bytes* and *reuse what you already moved*.]

== The inference budget worksheet #Q

Every deployment must name its constraints *before* optimizing:
#pause
#align(center, table(
  columns: 5, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 8pt), align: center,
  table.header([*latency*], [*throughput*], [*VRAM*], [*quality*], [*cost*]),
  [time to token], [tokens / s], [fits the device?], [accuracy], [\$ / token],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[every optimization improves *one or more* of these — with a *trade-off* in the others])

// ═══════════════════════════ PART II — KV cache ═══════════════════════════
= KV cache: reuse what attention already computed

== Recall causal attention #D

To attend from a new token $t$, a layer needs the keys and values of *all* tokens $1, dots, t$:
#pause
$ "Attention"(q_t, K_(1:t), V_(1:t)) = "softmax"(q_t K_(1:t)^top \/ sqrt(d)) thin V_(1:t) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the new query reads the *whole prefix* — so the prefix's keys and values must be available])

== The naive mistake #D

Re-run the entire Transformer on the full prefix after *every* new token:
#pause
#codebox(size: 13pt)[```python
for t in range(1, T):
    # recompute K, V for ALL of tokens 1..t  ← wasteful
    K, V = project_all(x[:t])
    h = attention(q[t], K, V)
```]
#pause
#alertbox[Tokens $1, dots, t-1$ have not changed, yet their key/value projections are recomputed at every step — $O(T)$ redundant work per token, $O(T^2)$ overall.]

== The KV-cache idea #V

Compute each token's $K, V$ *once*, then keep them:
#pause
#kvcache
#pause
#align(center, text(size: 14pt, fill: MUTED)[earlier tokens $-> $ cached $K, V$ · new token $-> $ new $q, k, v$ · append $k_t, v_t$ and read the whole cache])

== Interactive: the KV cache #I

#interbox(link-to: IA + "kv-cache")[
  Step a short prompt through decode: watch each new token compute only its own $q_t, k_t, v_t$, *append* $k_t, v_t$ to the growing cache, and attend over the whole stored history — no recomputation of old projections.
]

== What caching saves — and what it does not #D

#pause
#two(
  notebox[*Saves* — the projections of old tokens. Each step computes only $q_t, k_t, v_t$ for the *new* token.],
  alertbox[*Does not remove* — attention over the growing history. The new query still *reads* all cached keys/values.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[KV cache turns repeated *projection* into a *memory lookup* — the read still grows with context])

== KV-cache shape, per layer #D

At every layer we store *two* tensors:
#pause
$ K, V in RR^(B times T times n_"kv" times d_h) $
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*Symbol*], [*Meaning*]),
  [$B$], [batch size (concurrent sequences)],
  [$T$], [current sequence length],
  [$n_"kv"$], [number of key/value heads],
  [$d_h$], [dimension per head],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[two tensors, at *every* one of the $L$ layers])

== KV-cache memory approximation #D

Sum the two tensors over all $L$ layers, at $b$ bytes per stored value:
#pause
$ M_"KV" approx 2 dot L dot B dot T dot n_"kv" dot d_h dot b $
#pause
#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 6pt), align: center,
  table.header([factor $2$], [$L$ layers], [$n_"kv" dot d_h$], [$b$ bytes]),
  [K and V], [per layer], [head geometry], [fp16 $-> 2$],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[linear in $L$, in $B$, and in $T$ — three ways for the cache to grow])

== Board calculation #D

Take $L = 32$, $B = 1$, $T = 4096$, $n_"kv" = 8$, $d_h = 128$, fp16 so $b = 2$:
#pause
$ M_"KV" approx 2 dot 32 dot 4096 dot 8 dot 128 dot 2 $
#pause
Multiply in stages:
$ 2 dot 32 = 64 quad -> quad dot 4096 = "262,144" quad -> quad dot 8 = "2,097,152" $
#pause
$ dot 128 = "268,435,456" quad -> quad dot 2 = "536,870,912" "bytes" $
#pause
#result[$approx 512$ MiB — for a *single* sequence, before concurrency]

== The key trade-off #V

KV cache buys decode speed by *spending memory*:
#pause
#fig("/lecture23/figures/kv_growth.png", w: 52%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[long contexts and high concurrency make *memory* — not parameters — the dominant capacity limit])

== Architectural response: fewer KV heads #V

Multi-query (MQA) and grouped-query (GQA) attention *share* keys/values across query heads:
#pause
#gqa
#pause
#align(center, text(size: 14pt, fill: MUTED)[$n_"kv" < n_"query"$ shrinks the cache by exactly $n_"query" \/ n_"kv"$, keeping many query perspectives])

// ═══════════════════════════ PART III — The main inference levers ═══════════════════════════
= The main inference levers

== Quantization: store fewer bits #V

Represent weights with *8-bit* or *4-bit* values instead of 16:
#pause
#fig("/lecture23/figures/quant_bits.png", w: 54%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[fewer bytes $-> $ less memory and faster bandwidth-bound decode; approximation can cost quality])

== Quantization is not one thing #D

Distinguish *what* you quantize and *when*:
#pause
- *weight-only* — shrink the parameters (helps the memory-bound decode read);
- *activation / KV-cache* — also quantize the running state (helps long-context memory);
- *post-training (PTQ)* vs *quantization-aware training (QAT)* — calibrate after, or train with, low precision.
#pause
#align(center, text(size: 16pt, fill: MUTED)[each choice trades quality, memory, and engineering effort differently])

== Attention kernels: avoid giant intermediates #V

Standard attention can build a large $T times T$ score matrix. IO-aware kernels compute *exact* attention while moving less:
#pause
#flashattn
#pause
#align(center, text(size: 14pt, fill: MUTED)[*FlashAttention* is the canonical example — same answer, far less high-bandwidth-memory traffic (we do not teach the CUDA)])

== Batching and scheduling #D

Serving many requests *together* raises hardware utilization:
#pause
- one weight fetch is amortized across the whole batch (recall: decode is memory-bound);
- but requests have *different* prompt lengths and generation durations.
#pause
#alertbox[The systems problem: pack varied requests to keep the device busy *without* stalling any one of them. We name the problem — we do not build a serving stack.]

== Paged / block-based KV storage #D

Treat the KV cache as *reusable memory blocks*, not one contiguous reservation per request:
#pause
#two(
  notebox[*Contiguous* — reserve max-length up front $-> $ fragmentation and waste.],
  notebox[*Paged* — allocate small blocks as the sequence grows $-> $ share and reclaim freely.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[the resource-management idea — not the framework internals — is what matters here])

== Speculative decoding #V

Break the sequential decode bottleneck with a cheap *draft* and a trusted *verify*:
#pause
#specdec
#pause
#align(center, text(size: 14pt, fill: MUTED)[the *target* model preserves output quality; speedup depends on how often draft tokens are accepted])

== Why speculative decoding can help #D

It trades *cheap draft computation* for *fewer sequential expensive target steps*:
#pause
- one target forward pass verifies several proposed tokens *in parallel*;
- accept the matching prefix $-> $ many decode steps collapse into one.
#pause
#alertbox[It does *not* make the big model unnecessary — the target still verifies every token, so the output follows the target's distribution.]

== Distillation: change the model itself #D

Train a smaller *student* to imitate a larger *teacher*'s output distribution:
#pause
$ L_"distill" = T^2 thin D_"KL"("softmax"(z_"teacher" \/ T) parallel "softmax"(z_"student" \/ T)) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[this changes the *model*, not only the serving procedure — soft targets carry more signal than one hard label])

== Interactive: knowledge distillation #I

#interbox(link-to: IA + "knowledge-distillation")[
  Sweep the temperature $T$ and watch the teacher's *soft* label distribution flatten — see how the student learns the relative ranking of alternatives, not just the top-1 answer.
]

== Pruning and sparsity #D

Remove weights (or whole structures) that contribute little:
#pause
#two(
  notebox[*Unstructured* — zero out individual small weights. High sparsity, but hardware rarely speeds it up.],
  notebox[*Structured* — drop whole heads / channels / layers. Less sparsity, but a *real* smaller-faster model.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[sparsity helps only when the *hardware* can skip the removed work — structure is what buys speed])

== Interactive: quantization and pruning #I

#interbox(link-to: IA + "quantize-prune")[
  Dial the bit-width down from 16 to 4 and sweep a pruning threshold — watch weight memory shrink, the quantization error grow, and task quality hold then fall as you push the compression too far.
]

== Mixture of experts: conditional compute #V

Route each token to a *few* of many expert sub-networks — grow parameters without growing per-token FLOPs:
#pause
#moerouting
#pause
#align(center, text(size: 14pt, fill: MUTED)[a router picks the top-$k$ experts; only those run — capacity scales, active compute does not])

== MoE worked example #D

*8 experts, top-2 routing*: each token fires $2 \/ 8 = 25%$ of the expert capacity.
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, center, center),
  table.header([Mixtral 8×7B], [*total* params], [*active* per token]),
  [full model], [$approx 47 "B"$], [$approx 13 "B"$],
))
#pause
#result[$approx 3.6 times$ fewer *active* parameters than *total* — capacity you do not pay for every token]

== Interactive: mixture of experts #I

#interbox(link-to: IA + "mixture-of-experts")[
  Send tokens through a router and watch the top-$k$ gate light up a small subset of experts — total parameters climb while the *active* compute per token stays fixed.
]

== A decision table #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 7pt), align: (left, left, left),
  table.header([*Lever*], [*Mainly attacks*], [*Main cost*]),
  [KV cache], [repeated prefix computation], [VRAM],
  [quantization], [weight / cache bandwidth], [approximation, engineering],
  [efficient attention kernel], [intermediate-memory traffic], [specialized implementation],
  [batching / paging], [underutilization, fragmentation], [scheduling complexity],
  [speculative decoding], [sequential target steps], [draft model, variable gain],
  [distillation], [model size / latency], [retraining, possible quality loss],
))

// ═══════════════════════════ PART IV — The systems lens across the course ═══════════════════════════
= Apply the systems lens across the course

== Diffusion connection #D

Diffusion sampling repeatedly invokes the *denoiser* — the same "repeated call" cost structure:
#pause
- *faster samplers* — fewer denoising steps (the sequential bottleneck);
- *latent-space* diffusion — smaller tensors per call;
- *batching* and *quantization* — cheaper individual calls.
#pause
#align(center, text(size: 16pt, fill: MUTED)[every trick here reduces the cost of *repeated invocations* — LLM decode and diffusion sampling rhyme])

== Representation-learning connection #D

A good frozen encoder is also an *engineering asset*:
#pause
- compute features *once*; attach lightweight *linear probes* downstream;
- a strong representation makes the whole downstream system *cheaper*, not only more accurate.
#pause
#align(center, text(size: 16pt, fill: MUTED)[accuracy and efficiency are the same coin: a better representation buys both])

== Case exercise — pick a system #Q

Choose one and reason about its inference profile:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left),
  table.header([*System*], [*Dominant pressure*]),
  [chat, long prompts, many users], [KV-cache memory + concurrency],
  [mobile image classifier], [device VRAM + compute],
  [text-to-image demo, latency cap], [repeated denoiser calls],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[choose *two* optimizations and name the expected trade-off — we return to this at the end])

== The one-sentence synthesis #Q

#pause
#align(center, block(fill: white, inset: (x: 18pt, y: 14pt), radius: 5pt, stroke: 2pt + ACC,
  text(size: 22pt, weight: 600, fill: INK)[
    Efficient inference is mostly about avoiding *repeated work* and *moving fewer bytes*.
  ]))
#pause
#align(center, text(size: 16pt, fill: MUTED)[KV cache, prefix reuse, FlashAttention, quantization, batching — all instances of this one idea])

== Closing bridge

The final lecture stacks *agents, reasoning systems, and interpretability* on top of this same foundation:
#pause
#align(center, text(size: 20pt, fill: INK)[
  models $+$ objectives $+$ optimization $+$ representations $+$ *inference constraints*
])
#pause
#align(center, text(size: 16pt, fill: MUTED)[deployment constraints are *first-class* model-design constraints, not an afterthought])

// ═══════════════════════════ PART V — Detailed systems slides ═══════════════════════════
= Detailed systems slides

== Parameter memory comes first #D

For $P$ parameters, the weights alone need:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 14pt, y: 7pt), align: (left, right),
  table.header([*Storage format*], [*Approx. weight memory*]),
  [fp32], [$4 P$ bytes],
  [fp16 / bf16], [$2 P$ bytes],
  [int8], [$P$ bytes],
  [4-bit], [$P \/ 2$ bytes, plus metadata],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[a first-order figure — runtime also pays for KV cache, activations, and framework overhead])

== Weight-memory example #D

A *7-billion*-parameter model in fp16:
#pause
$ 7 times 10^9 times 2 "bytes" approx 14 "GB" $
#pause
for weights *alone*.
#pause
#alertbox[Why is a "16 GB" device still inadequate? Weights ($14$ GB) leave almost nothing for KV cache, activations, and overhead — long-context generation needs *far* more headroom.]

== Arithmetic intensity intuition #D

The *same weights*, two very different reuse patterns:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, center, left),
  table.header([*Phase*], [*Op shape*], [*Reuse per fetched weight*]),
  [prefill], [matrix–matrix], [high (many token vectors)],
  [decode], [matrix–vector], [low (one token vector)],
))
#pause
$ "less reuse per byte" quad ==> quad "memory bandwidth matters more" $
#pause
#align(center, text(size: 15pt, fill: MUTED)[no hardware-specific roofline number needed — the *shape* already tells the story])

== Causal attention cost, briefly #D

A full attention layer compares token pairs:
#pause
$ O(T^2 d) $
#pause
#notebox[KV caching removes the recomputation of *old projections* during decode. It does *not* remove the new query's need to attend across the stored history — the cache read still scales with $T$.]

== Cache append at one layer #D

At token $t$, extend the cache and attend:
#pause
$ K_(1:t) = [K_(1:t-1) thin ; thin k_t], quad quad V_(1:t) = [V_(1:t-1) thin ; thin v_t] $
#pause
Only $q_t, k_t, v_t$ are freshly computed; the attention for $q_t$ *reads* all cached keys and values.
#pause
#codebox(size: 13pt)[```python
k_t, v_t = project(x_t)          # new token only
K = cat(K_cache, k_t); V = cat(V_cache, v_t)
h_t = attention(q_t, K, V)       # read the whole history
K_cache, V_cache = K, V          # append for next step
```]

== Recompute versus cache #V

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 8pt), align: (left, center, center, center),
  table.header([*Strategy*], [*Compute*], [*Memory*], [*Decode latency*]),
  [recompute full prefix], [very high], [low cache], [poor],
  [store KV cache], [lower], [grows with $T$], [much better],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[this table is the *central systems trade-off* of the lecture — spend memory, avoid repeated compute])

== KV-cache calculation, step by step #D

Start from the formula and substitute:
#pause
$ 2 dot L dot B dot T dot n_"kv" dot d_h dot b, quad L = 32, B = 1, T = 4096, n_"kv" = 8, d_h = 128, b = 2 $
#pause
$ 2 dot 32 dot 4096 dot 8 dot 128 dot 2 = "536,870,912" "bytes" $
#pause
$ "536,870,912" \/ 1024^2 = 512 quad ==> quad approx 512 "MiB" quad #text(size: 14pt, fill: MUTED)[(before concurrency)] $

== Concurrency changes the answer #V

If *32* requests each need this cache:
#pause
$ 32 times 512 "MiB" approx 16 "GiB" $
#pause
#fig("/lecture23/figures/kv_growth.png", w: 46%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[long contexts $times$ many users make cache capacity a *product-level* constraint, not a footnote])

== Grouped-query attention #D

Standard MHA uses one key/value head per query head. GQA uses *fewer*:
#pause
$ n_"kv-heads" < n_"query-heads" $
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, center, center),
  table.header([$n_"kv"$], [cache size], [flexibility]),
  [$= n_"query"$ (MHA)], [$1 times$], [full],
  [group (GQA)], [$n_"query" \/ n_"kv"$ smaller], [most],
  [$= 1$ (MQA)], [smallest], [least],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[a direct knob on the cache term $n_"kv" dot d_h$ — retains multiple query perspectives])

== Quantization is approximate representation #V

Store a real-valued block approximately as a low-bit integer tensor plus metadata:
#pause
#quantflow
#pause
$ W approx s (W_"int" - z) quad #text(size: 14pt, fill: MUTED)[(scale $s$, optional zero-point $z$)] $

== Quantization design choices #D

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left),
  table.header([*Choice*], [*Question it answers*]),
  [per-tensor or per-channel scale], [how much variation must be preserved?],
  [weight-only or activations too], [where is the memory / bandwidth pressure?],
  [8-bit or 4-bit], [what quality loss is acceptable?],
  [post-training or trained-aware], [can we afford to adapt the model?],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[finer scales preserve outliers; coarser scales save more — the central quantization tension])

== Quantization is not free speed #D

Fewer bits *can* reduce bandwidth and fit a bigger model — but the gain is conditional:
#pause
#alertbox[Speedups depend on *hardware and kernel support*. Dequantization, metadata, and irregular operations can eat much of the theoretical saving.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[a useful systems-literacy correction: "smaller weights" $!=$ "automatically faster"])

== FlashAttention as a memory algorithm #D

The attention *equation* is unchanged:
#pause
$ "softmax"(Q K^top \/ sqrt(d)) thin V $
#pause
#notebox[The kernel *tiles* the computation so the large score matrix need not be written to slow high-bandwidth memory. Same mathematics, different *data movement*.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[teach it as "same answer, fewer bytes moved" — an IO-aware rewrite, not a new model])

== Prefix caching #D

Many requests share a *system prompt* or *document prefix*:
#pause
- compute that prefix's KV state *once*;
- reuse it for every later request that shares it.
#pause
#align(center, text(size: 16pt, fill: MUTED)[the serving analogue of dynamic programming — never redo the shared sub-computation])

== Continuous batching #V

Requests arrive and finish at *different* times; a scheduler inserts new ones into free capacity:
#pause
#fig("/lecture23/figures/latency_batch.png", w: 50%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[the target becomes *throughput and tail latency* — not one request's speed in isolation])

== Blocked KV allocation #D

Instead of one contiguous cache region per request, allocate *blocks* as sequences grow:
#pause
- reduces fragmentation;
- makes it easy to *share* and *reclaim* memory across requests.
#pause
#align(center, text(size: 16pt, fill: MUTED)[keep framework names off the slide — the resource-management idea is the lesson])

== Speculative decoding, expanded #D

#pause
+ a small *draft* model proposes several tokens;
+ the large *target* model checks them together in one pass;
+ *accept* a matching prefix;
+ *sample a correction* where the draft diverges.
#pause
#codebox(size: 12.5pt)[```python
draft_tokens = draft.generate(context, k=4)      # cheap, sequential-but-small
logits = target(context + draft_tokens)          # ONE target pass, parallel
accept = longest_matching_prefix(draft_tokens, logits)
context += accept + resample_if_rejected(logits) # target stays the authority
```]

== Acceptance rate determines speedup #D

The whole payoff hinges on how often the draft is *right*:
#pause
- draft guesses *poorly* $-> $ verification rejects early $-> $ little sequential work saved;
- draft predicts a *long correct prefix* $-> $ one target pass replaces many decode steps.
#pause
$ "draft quality" + "cheapness" quad ==> quad "useful speculation" $
#pause
#align(center, text(size: 15pt, fill: MUTED)[a well-matched, fast draft is the entire game — a slow or inaccurate draft can even *lose*])

== Distillation changes the model #D

Train the student against the teacher's temperature-softened distribution:
#pause
$ L_"distill" = T^2 thin D_"KL"("softmax"(z_"teacher" \/ T) parallel "softmax"(z_"student" \/ T)) $
#pause
#notebox[Soft targets convey *relative alternatives* — "mostly cat, a little dog, never car" — far richer than a single hard label. The $T^2$ factor keeps gradient magnitudes stable as $T$ rises.]

== Compression versus serving optimization #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 7pt), align: (left, center, left),
  table.header([*Approach*], [*Retraining?*], [*Main effect*]),
  [quantization], [often no], [fewer bits / lower bandwidth],
  [KV cache], [no], [avoid repeated prefix work],
  [speculative decoding], [no target retraining], [fewer sequential target steps],
  [distillation], [yes], [smaller / faster model],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[some levers *change the model*; others only *change how it is served* — know which you are pulling])

== Case exercise: choose levers #Q

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left),
  table.header([*Scenario*], [*Reasonable pair (name the trade-off)*]),
  [A · long-doc chat, many users], [GQA + paged KV · quality-neutral, scheduling cost],
  [B · offline phone classifier], [distillation + int8 · retrain effort, small quality dip],
  [C · image generator, 2 s budget], [fewer sampler steps + quantization · slight fidelity loss],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[there is no free lever — every choice pays somewhere in quality, cost, or engineering])

== Common misconceptions #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 7pt), align: (left, left),
  table.header([*Misconception*], [*Correction*]),
  ["KV cache makes attention constant-time."], [avoids old projections; cache *reads* still grow with context],
  ["Quantization always improves every metric."], [trades precision; relies on efficient kernels],
  ["FlashAttention changes Transformer math."], [same attention, different *memory movement*],
  ["A draft model replaces the target model."], [the target still *verifies* every output token],
))

== Retrieval questions #Q

#pause
+ Why do prefill and decode stress hardware *differently*?
#pause
+ What *exactly* is stored in a KV cache?
#pause
+ Which variables make cache memory *grow*?
#pause
+ Why can 4-bit weights help a *bandwidth-bound* workload?
#pause
+ Which optimization changes the *underlying model* rather than only serving it?

== Memory-budget worksheet #D

Before loading a model, estimate *all four* terms:
#pause
$ underbrace("weights", "2P") + underbrace("KV cache", "grows with " B T) + "temporary activations" + "runtime overhead" $
#pause
#alertbox[For a *serving* system, multiply the KV-cache term by *expected concurrency* — not by one idealized request. This is where "it fit in the demo" becomes "it OOMs in production".]

== Context-window policy is a systems policy #D

A larger maximum context improves user experience but *raises cache memory and attention work*:
#pause
- truncation · retrieval · summarization · sliding windows;
#pause
- these are *product* choices shaped by the *same* compute constraints.
#pause
#align(center, text(size: 16pt, fill: MUTED)[this is where inference engineering meets retrieval and agent design])

== Model parallelism, one placement slide #D

When one device cannot hold the model, split parameters or layers across devices:
#pause
$ "more devices" quad != quad "linear speedup" $
#pause
#notebox[Distributing the model makes larger models *feasible*, but adds *communication overhead*. Pipeline- and tensor-parallel algorithms stay outside this core lecture.]

== Compilation and fused operations #D

Launching many tiny operations separately wastes time:
#pause
- *compilers* and *fused kernels* combine operations;
- fewer launches, fewer intermediate memory reads/writes.
#pause
#align(center, text(size: 16pt, fill: MUTED)[another instance of the core idea: *avoid unnecessary movement and coordination*])

== Quantization evaluation protocol #D

Never judge a compressed model on average accuracy alone — test:
#pause
- representative *task quality*;
- *long-context* behavior;
- *rare / error-sensitive* cases;
- *latency and memory* on the *actual target hardware*.
#pause
#alertbox[Average benchmark accuracy can hide a damaging quantization failure that only surfaces on the tail.]

== Cost per useful outcome #Q

The relevant quantity is *not* merely tokens per second:
#pause
$ ("cost") / ("useful, correct, safe outcome") $
#pause
#align(center, text(size: 16pt, fill: MUTED)[a slower model can win if it avoids retries; a fast model is wasteful if its outputs need repeated correction])

== Energy and hardware awareness #D

Moving data across the memory hierarchy costs *time and energy*:
#pause
- efficient inference can lower operational cost *and* environmental impact;
- but *demand growth* can offset per-query savings.
#pause
#alertbox[Avoid the simplistic "efficient therefore green" claim — count the *total* workload, not the per-query number.]

== Serving versus research benchmark #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left),
  table.header([*Benchmark focus*], [*Production focus*]),
  [one model, one batch, peak throughput], [variable requests, tail latency, reliability],
  [fixed prompt length], [long and changing contexts],
  [clean hardware state], [memory fragmentation and concurrency],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[question any performance claim that omits its *workload assumptions*])

== Retrieval questions, extended #Q

#pause
+ Why can a model *fit* in memory yet *fail* under concurrent long-context use?
#pause
+ What is the difference between quantizing *weights* and quantizing the *KV cache*?
#pause
+ Which optimization changes *arithmetic scheduling* but not model *output*?
#pause
+ Why does serving *require* workload assumptions?

== Final bridge

#align(center, text(size: 21pt, fill: INK)[
  models $+$ objectives $+$ representations $+$ *inference systems*
])
#pause
#align(center, text(size: 16pt, fill: MUTED)[are all needed to understand a frontier deployment])
#pause
#notebox[The final lecture uses this checklist to demystify *agents, reasoning systems, and interpretability* — they are built on the foundation this course has assembled, inference constraints included.]

// ═══════════════════════════ Summary ═══════════════════════════
= Summary

== Student takeaways #V

#pause
+ Prompt processing (*prefill*) and token-by-token *decode* have different performance profiles.
#pause
+ KV caching trades *memory* for *avoided repeated computation*.
#pause
+ Many inference optimizations reduce *memory traffic* rather than changing neural-network mathematics.
#pause
+ Deployment constraints are *first-class* model-design constraints.

== The levers, one last look #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 6pt), align: (left, left, left),
  table.header([*Lever*], [*Changes*], [*Pay with*]),
  [KV cache · GQA], [serving], [VRAM],
  [quantization], [representation], [precision + kernels],
  [FlashAttention], [data movement], [specialized kernel],
  [batching · paging], [scheduling], [complexity],
  [speculative decoding], [scheduling], [a draft model],
  [distillation · pruning · MoE], [the model], [training effort],
))

#focus-slide[
  At inference, the model is often *waiting on memory*, not arithmetic.
  #v(10pt)
  #set text(size: 22pt)
  Efficient inference $=$ avoid *repeated work* $+$ move *fewer bytes*.
  #v(12pt)
  #text(size: 20pt)[Next: *Frontier Topics & Course Synthesis* — agents, reasoning, interpretability.]
]
