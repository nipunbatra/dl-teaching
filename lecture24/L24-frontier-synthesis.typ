// Frontier Topics and Course Synthesis
// Lecture 24 (finale) · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture24/L24-frontier-synthesis.typ /tmp/L24.pdf
//   typst compile --root . --input handout=true lecture24/L24-frontier-synthesis.typ /tmp/L24h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.
// Trend figures: python3 lecture24/diagrams/l24_figs.py   (all structural diagrams are native fletcher)

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Frontier Topics and the Deep-Learning Toolkit],
  subtitle: [Agents, reasoning, interpretability — and the vocabulary to evaluate what comes next],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let cream = rgb("#EFEEEB")
#let rect = fletcher.shapes.rect

// ═══════════════════════════ fletcher diagrams (centerpieces) ═══════════════════════════

// ── (A) the 24-lecture course in ONE serpentine map ──
#let coursearc = align(center, diagram(spacing: (20mm, 15mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let cbx(pos, lbl, c) = node(pos, align(center, text(size: 8.5pt, lbl)), shape: rect, corner-radius: 4pt, inset: 5pt, stroke: 0.9pt + c, fill: c.lighten(85%))
  // row 0 (left -> right): foundations
  cbx((0, 0), [probability \ likelihood], TEAL)
  cbx((1, 0), [losses \ MLE · MAP], TEAL)
  cbx((2, 0), [MLP], BLUE)
  cbx((3, 0), [backprop], BLUE)
  cbx((4, 0), [optimization \ generalization], BLUE)
  cbx((5, 0), [CNNs \ (vision)], ACC)
  for a in range(5) { edge((a, 0), (a + 1, 0), "-|>", stroke: 0.8pt + MUTED) }
  edge((5, 0), (5, 1), "-|>", stroke: 1.4pt + INK)
  // row 1 (right -> left): sequences to the frontier
  cbx((5, 1), [sequences \ RNNs], ACC)
  cbx((4, 1), [attention \ Transformers], ACC)
  cbx((3, 1), [SSL · \ multimodal], GREEN)
  cbx((2, 1), [generative \ VAE·GAN·diff], GREEN)
  cbx((1, 1), [inference \ systems], GREEN)
  node((0, 1), align(center, text(size: 8.5pt, fill: white, weight: 600)[frontier \ (today)]), shape: rect, corner-radius: 4pt, inset: 5pt, stroke: 1.2pt + INK, fill: INK)
  for (a, b) in ((5, 4), (4, 3), (3, 2), (2, 1), (1, 0)) { edge((a, 1), (b, 1), "-|>", stroke: 0.8pt + MUTED) }
}))

// ── (B) the agent loop: a model inside a software cycle ──
#let agentloop = align(center, diagram(spacing: (34mm, 17mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), align(center, text(size: 9.5pt)[observe \ state · context]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: cream, stroke: 0.9pt + INK)
  node((1, 0), align(center, text(size: 9.5pt)[*LLM* proposes a \ structured action]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: TEAL.lighten(82%), stroke: 1pt + TEAL)
  node((1, 1), align(center, text(size: 9.5pt)[*application* validates \ & executes tool]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: ACC.lighten(82%), stroke: 1pt + ACC)
  node((0, 1), align(center, text(size: 9.5pt)[append result \ to context]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  edge((0, 0), (1, 0), "-|>", stroke: 0.9pt + INK, label: text(size: 8.5pt, fill: MUTED)[text out], label-side: left)
  edge((1, 0), (1, 1), "-|>", stroke: 0.9pt + INK, label: text(size: 8.5pt, fill: MUTED)[schema], label-side: right)
  edge((1, 1), (0, 1), "-|>", stroke: 0.9pt + INK)
  edge((0, 1), (0, 0), "-|>", stroke: 0.9pt + INK, label: text(size: 8.5pt, fill: MUTED)[loop], label-side: left)
  node((0.5, 1.9), text(size: 10pt, fill: INK)[only the *model* box (top-right) is a neural net — the rest is ordinary software], stroke: none, fill: none)
}))

// ── (C) structured action schema -> validation -> execution ──
#let actionschema = align(center, diagram(spacing: (13mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), align(center, text(size: 9.5pt)[NL intent \ "book a table"]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: cream, stroke: 0.9pt + INK)
  node((1.4, 0), align(center, text(size: 9pt)[action schema \ {tool, args, id}]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: TEAL.lighten(82%), stroke: 1pt + TEAL)
  node((2.9, 0), align(center, text(size: 9pt)[validate types \ + permissions]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((4.35, 0), align(center, text(size: 9pt)[execute \ tool / API]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(82%), stroke: 1pt + ACC)
  node((5.7, 0), align(center, text(size: 9pt)[result \ back to model]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  for (a, b) in ((0, 1.4), (1.4, 2.9), (2.9, 4.35), (4.35, 5.7)) { edge((a, 0), (b, 0), "-|>", stroke: 0.7pt + MUTED) }
  node((1.4, -0.95), text(size: 9pt, fill: TEAL)[model], stroke: none, fill: none)
  node((3.9, -0.95), text(size: 9pt, fill: ACC)[software boundary], stroke: none, fill: none)
}))

// ── (D) retrieval-augmentation: external memory feeds the context ──
#let ragflow = align(center, diagram(spacing: (13mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), align(center, text(size: 9.5pt)[query]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: cream, stroke: 0.9pt + INK)
  node((1.4, 0), align(center, text(size: 9pt)[encode \ $-> $ vector]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: TEAL.lighten(82%), stroke: 1pt + TEAL)
  node((2.9, 0), align(center, text(size: 9pt)[search \ vector index]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((4.3, 0), align(center, text(size: 9pt)[top-$k$ \ documents]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: BLUE.lighten(72%), stroke: 0.9pt + BLUE)
  node((5.8, 0), align(center, text(size: 9pt)[LM reads \ query $+$ docs]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: INK, stroke: 0.9pt + INK)
  node((5.8, 0), align(center, text(size: 9pt, fill: white)[LM reads \ query $+$ docs]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: INK, stroke: 0.9pt + INK)
  node((7.2, 0), align(center, text(size: 9pt)[grounded \ answer]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  for (a, b) in ((0, 1.4), (1.4, 2.9), (2.9, 4.3), (4.3, 5.8), (5.8, 7.2)) { edge((a, 0), (b, 0), "-|>", stroke: 0.7pt + MUTED) }
  node((3.6, 0.95), text(size: 10pt, fill: INK)[knowledge lives *outside* the weights — as retrievable *state*], stroke: none, fill: none)
}))

// ── (E) inference-time search loop ──
#let searchloop = align(center, diagram(spacing: (16mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), align(center, text(size: 9.5pt)[model proposes \ candidates]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: TEAL.lighten(82%), stroke: 1pt + TEAL)
  node((0, 1), align(center, text(size: 9.5pt)[verifier · reward · critic \ scores candidates]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((0, 2), align(center, text(size: 9.5pt)[keep · revise · branch · *stop*]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: ACC.lighten(82%), stroke: 1pt + ACC)
  edge((0, 0), (0, 1), "-|>", stroke: 0.9pt + INK)
  edge((0, 1), (0, 2), "-|>", stroke: 0.9pt + INK)
  edge((0, 2), (0, 0), "-|>", stroke: 0.9pt + MUTED, bend: 55deg, label: text(size: 8.5pt, fill: MUTED)[more compute], label-side: right)
}))

// ── (F) the residual stream as an additive shared workspace ──
#let residstream = align(center, diagram(spacing: (13mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), $h_ell$, radius: 6mm, fill: cream, stroke: 1pt + INK)
  node((1.6, 0), text(size: 14pt)[$+$], radius: 5mm, fill: GREEN.lighten(80%), stroke: 1pt + GREEN)
  node((3.4, 0), text(size: 14pt)[$+$], radius: 5mm, fill: GREEN.lighten(80%), stroke: 1pt + GREEN)
  node((4.7, 0), $h_(ell+1)$, radius: 6.5mm, fill: ACC.lighten(72%), stroke: 1pt + ACC)
  edge((0, 0), (1.6, 0), "-|>", stroke: 1.8pt + INK)
  edge((1.6, 0), (3.4, 0), "-|>", stroke: 1.8pt + INK)
  edge((3.4, 0), (4.7, 0), "-|>", stroke: 1.8pt + INK)
  node((1.6, 1.4), align(center, text(size: 9.5pt)[Attn$(h_ell)$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((3.4, 1.4), align(center, text(size: 9.5pt)[MLP$(dot)$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  edge((0, 0), (1.6, 1.4), "-|>", stroke: 0.8pt + TEAL, label: text(size: 8pt, fill: TEAL)[read], label-side: left)
  edge((1.6, 1.4), (1.6, 0), "-|>", stroke: 0.8pt + TEAL, label: text(size: 8pt, fill: TEAL)[add], label-side: right)
  edge((1.6, 0), (3.4, 1.4), "-|>", stroke: 0.8pt + BLUE)
  edge((3.4, 1.4), (3.4, 0), "-|>", stroke: 0.8pt + BLUE, label: text(size: 8pt, fill: BLUE)[add], label-side: right)
  node((2.35, -1.0), text(size: 10pt, fill: INK)[each block *reads* the stream and *adds* an update — it does not overwrite], stroke: none, fill: none)
}))

// ── (G) sparse autoencoder: h -> wide, mostly-zero features -> h-hat ──
#let saediagram = align(center, diagram(spacing: (17mm, 4.5mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 2.5), align(center, text(size: 10pt)[$h$ \ dense, $d$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: cream, stroke: 1pt + INK)
  // wide sparse feature column: a handful active (ACC), most zero (grey)
  let active = (1, 4)
  for i in range(6) {
    let on = active.contains(i)
    node((1.4, i), none, radius: 2.5mm, fill: if on { ACC } else { rgb("#E4E2DD") }, stroke: 0.6pt + MUTED)
    edge((0, 2.5), (1.4, i), stroke: 0.3pt + MUTED.lighten(35%))
  }
  node((1.4, 6.0), text(size: 9pt, fill: ACC)[overcomplete, sparse $a$], stroke: none, fill: none)
  node((3.0, 2.5), align(center, text(size: 10pt)[$hat(h)$ \ recon]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: TEAL.lighten(80%), stroke: 1pt + TEAL)
  for i in range(6) { edge((1.4, i), (3.0, 2.5), stroke: 0.3pt + MUTED.lighten(35%)) }
}))

// ── (H) multimodal / generative agent: composition of familiar parts ──
#let mmagent = align(center, diagram(spacing: (26mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), align(center, text(size: 9.5pt)[text model \ (L15–16)]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((0, 1), align(center, text(size: 9.5pt)[vision encoder \ (L17)]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((0, 2), align(center, text(size: 9.5pt)[tool loop \ (L23 systems)]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  node((2, 1), align(center, text(size: 10pt, fill: white, weight: 600)[multimodal \ agent]), shape: rect, corner-radius: 5pt, inset: 9pt, fill: INK, stroke: 1pt + INK)
  edge((0, 0), (2, 1), "-|>", stroke: 0.8pt + TEAL)
  edge((0, 1), (2, 1), "-|>", stroke: 0.8pt + BLUE)
  edge((0, 2), (2, 1), "-|>", stroke: 0.8pt + ACC)
  node((3.6, 1), align(center, text(size: 9.5pt)[external \ state]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  edge((2, 1), (3.6, 1), "-|>", stroke: 0.8pt + MUTED, bend: 30deg)
  edge((3.6, 1), (2, 1), "-|>", stroke: 0.8pt + MUTED, bend: 30deg)
}))

// ── (I) the systems thread: model life-cycle ──
#let systemsthread = align(center, diagram(spacing: (16mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let sbx(x, lbl, c) = node((x, 0), align(center, text(size: 9.5pt, lbl)), shape: rect, corner-radius: 4pt, inset: 7pt, fill: c.lighten(82%), stroke: 0.9pt + c)
  sbx(0, [training], TEAL)
  sbx(1, [pretraining], TEAL)
  sbx(2, [adaptation], BLUE)
  sbx(3, [inference], ACC)
  sbx(4, [tools · search \ deployment], GREEN)
  for a in range(4) { edge((a, 0), (a + 1, 0), "-|>", stroke: 0.8pt + MUTED) }
}))

// ── (J) the representation thread ──
#let repthread = align(center, diagram(spacing: (18mm, 14mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let rbx(pos, lbl, c) = node(pos, align(center, text(size: 8.5pt, lbl)), shape: rect, corner-radius: 4pt, inset: 5pt, fill: c.lighten(84%), stroke: 0.9pt + c)
  rbx((0, 0), [CNN \ features], ACC)
  rbx((1, 0), [transferred \ features], ACC)
  rbx((2, 0), [token \ reps], BLUE)
  rbx((3, 0), [SSL \ features], BLUE)
  for a in range(3) { edge((a, 0), (a + 1, 0), "-|>", stroke: 0.8pt + MUTED) }
  edge((3, 0), (3, 1), "-|>", stroke: 1.2pt + INK)
  rbx((3, 1), [image–text \ embeddings], GREEN)
  rbx((2, 1), [VAE \ latents], GREEN)
  rbx((1, 1), [diffusion \ latents], TEAL)
  rbx((0, 1), [system \ state], INK)
  for (a, b) in ((3, 2), (2, 1), (1, 0)) { edge((a, 1), (b, 1), "-|>", stroke: 0.8pt + MUTED) }
}))

// ── (K) the L1 objective map, extended to the frontier (full-circle callback) ──
#let l1map = align(center, diagram(spacing: (17mm, 9mm), node-stroke: 0.9pt, {
  node((0, 0), align(center, text(size: 12pt)[outputs \ $p_theta (y|x)$]), shape: rect, fill: cream, stroke: INK, inset: 7pt)
  node((2, 0), align(center, text(size: 12pt)[parameters \ $p(theta)$]), shape: rect, fill: cream, stroke: INK, inset: 7pt)
  node((4, 0), align(center, text(size: 11pt)[preference · reward \ verifier signal]), shape: rect, fill: GREEN.lighten(84%), stroke: GREEN, inset: 6pt)
  node((0, 1), align(center, text(size: 12pt)[data loss]), shape: rect, fill: ACC.lighten(80%), stroke: ACC, inset: 7pt)
  node((2, 1), align(center, text(size: 12pt)[regularizer]), shape: rect, fill: ACC.lighten(80%), stroke: ACC, inset: 7pt)
  edge((0, 0), (0, 1), text(size: 12pt)[NLL], "-|>", stroke: 0.8pt + MUTED, label-side: left)
  edge((2, 0), (2, 1), text(size: 12pt)[MAP prior], "-|>", stroke: 0.8pt + MUTED, label-side: right)
  node((1, 2), text(size: 12pt, fill: white, weight: 600)[the same differentiable objective], shape: rect, fill: TEAL, stroke: none, inset: 8pt)
  edge((0, 1), (1, 2), "-|>", stroke: 0.8pt + MUTED)
  edge((2, 1), (1, 2), "-|>", stroke: 0.8pt + MUTED)
  edge((4, 0), (1, 2), "-|>", stroke: 0.9pt + GREEN, label: text(size: 10pt, fill: GREEN)[still a loss], label-side: right, bend: 20deg)
}))

#title-slide()

// ═══════════════════════════ PART I — Reassemble the course ═══════════════════════════
= Part I · Reassemble the course

== Where the whole course was heading #V

We spent twenty-three lectures building a toolkit. Today we *reassemble* it — and use it to read the frontier.
#pause
#result[Most frontier systems are familiar deep-learning components composed into a larger loop.]
#pause
#notebox[This is not a fourth dense module. It is a *disciplined way to place new claims*: what is the model, the objective, the representation, and what runs at inference?]

== The 24-lecture course in one map #V

#coursearc
#pause
#align(center, text(size: 14pt, fill: MUTED)[probability $->$ differentiable models $->$ optimization $->$ vision & sequences $->$ attention & pretraining $->$ SSL & multimodal $->$ generative $->$ systems $->$ *frontier*])

== Four questions for every new model #Q

Whatever the product name, ask the same four questions:
#pause
+ what *representations* and *architecture* does it use?
#pause
+ what *data* and *objective* train it?
#pause
+ what inference-time *loop or tool system* surrounds it?
#pause
+ what *failure mode* or *evaluation* is being claimed?
#pause
#notebox[Every slide today is an instance of these four questions.]

== One durable course equation #D

Across every topic, the same loop kept returning:
#pause
$ #text(fill: TEAL)[data] -->^"model" #text(fill: ACC)[prediction / sample] -->^"loss" #text(fill: BLUE)[gradient update] $
#pause
#align(center, text(size: 15pt, fill: MUTED)[the *model* maps data to predictions; the *loss* turns predictions into gradients])
#pause
#align(center, text(size: 16pt, fill: MUTED)[topics change the *representation, factorization, objective,* or *inference procedure* — not this basic loop])

== A warning about the word "frontier" #Q

Product names change every few months. The *durable* skill is different:
#pause
#two(
  notebox[*Fragile* — memorizing which named model is currently best at which benchmark.],
  notebox[*Durable* — locating a new system in the map, and naming what is *genuinely new* versus renamed.],
)
#pause
#alertbox[Everything on frontier claims today is hedged: "as of this writing." The *method of evaluation* outlives any specific result.]

== Architecture versus system #D

Students should stop calling every new wrapper "a new neural network":
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left, left),
  table.header([*Item*], [*Example*], [*What is new?*]),
  [architecture], [Transformer, U-Net], [parameterized computation],
  [objective], [NLL, contrastive, reward], [what training prefers],
  [representation], [tokens, patches, latents], [what information is carried],
  [system wrapper], [tool loop, search, cache], [how the model is *used*],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[a new *wrapper* is not a new *network* — and vice versa])

// ═══════════════════════════ PART II — Scaling and emergence ═══════════════════════════
= Part II · Scaling, and reading trend claims

== Two scaling axes, kept separate #D

The single most useful distinction in frontier literacy:
#pause
$ underbrace(#text(fill: TEAL)[training compute · data], "how the model was built") quad "versus" quad underbrace(#text(fill: ACC)[inference-time compute per problem], "how hard it thinks at query time") $
#pause
#alertbox[These are *different budgets* with *different cost curves*. Conflating them is the most common frontier confusion.]

== Scaling laws #V

Empirically, test loss often falls as a *power law* in compute, data, and parameters:
#pause
$ L(C) approx E + (C_c / C)^alpha quad quad #text(size: 15pt, fill: MUTED)[($E$ = irreducible error; $alpha$ small)] $
#pause
#fig("/lecture24/figures/scaling_law.svg", w: 46%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[straight-ish on log-log, flattening toward a floor $E$ — *illustrative*, not fitted to any model])

== Emergent abilities — hedge carefully #V

Some capabilities *appear* to switch on sharply with scale — but the *metric* shapes the story:
#pause
#fig("/lecture24/figures/emergent.svg", w: 60%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[a discontinuous metric can make smooth improvement *look* like a jump — treat "emergence" as *open*])

== Scaling is a claim, not a law of nature #Q

#pause
- power laws are *empirical fits over a range* — extrapolation is not guaranteed;
#pause
- a floor $E$ exists; more compute has *diminishing* returns;
#pause
- "bigger" trades against data quality, inference cost, and evaluation coverage.
#pause
#interbox(link-to: IA + "in-context-learning")[
  In-context learning — how a *frozen* model adapts from examples in its prompt, with no weight update, as scale grows.
]

== Generalization still surprises us #V

#let _test(x) = 0.38 + 0.12 * calc.exp(-x / 0.4) + 0.92 * calc.exp(-calc.pow((x - 1.0) / 0.06, 2))
#let _train(x) = if x < 1.0 { 0.57 * (1.0 - x) } else { 0.0 }
#align(center, lines(
  fn: (
    x => _test(x),
    x => _train(x),
  ),
  domain: (0.05, 3.0),
  samples: 118,
  colors: (ACC, TEAL),
  labels: ([test error], [train error]),
  vlines: ((1.0, none, MUTED),),
  markers: false,
  x-label: [capacity (params / data)], y-label: [error],
  size: (78mm, 48mm),
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[even classical questions are not closed — *double descent*: test error falls, spikes, then falls again])
#pause
#interbox(link-to: IA + "double-descent")[Sweep capacity past the interpolation threshold and watch the *second descent*.]

// ═══════════════════════════ PART III — Agents ═══════════════════════════
= Part III · Agents — a model inside a software loop

== A demystifying definition #D

An "agent" is not a new architecture. It is a *composition*:
#pause
$ #text(fill: INK, weight: 600)[agent] = #text(fill: TEAL)[model] + #text(fill: BLUE)[state / context] + #text(fill: ACC)[tool interface] + #text(fill: GREEN)[control loop] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[three of the four pieces are *ordinary software* — only the model is a neural network])

== The agent loop #V

#agentloop
#pause
#align(center, text(size: 14pt, fill: MUTED)[observe $->$ propose an action $->$ execute $->$ append result $->$ observe again])

== The critical correction #Q

#alertbox[The neural network does *not* execute code or call an API by itself.]
#pause
It emits a *structured request* as text. Ordinary software then:
#pause
- *validates* the request against a schema;
- *checks* permissions;
- *executes* the tool and *returns* the result.
#pause
#result[the model proposes; the *system* disposes]

== Tool calling is a representation problem #D

Natural-language intent must be *represented* as a constrained action:
#pause
#actionschema
#pause
#align(center, text(size: 15pt, fill: MUTED)[validity, permissions and error handling live *outside* the model — natural language is not a safe API contract])

== State is external #D

Debugging an agent means knowing *where each kind of state lives*:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Kind of state*], [*Where it lives / when it is set*]),
  [model parameters], [learned *before* deployment; frozen at run time],
  [context window], [temporary text / token state for this session],
  [tool / database state], [external, mutable *world* state],
  [application memory], [what the system *chooses* to retain],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[most "agent bugs" are confusions about *which* state changed])

== Planning is not a magic module #D

A system can ask the model to propose actions, critique them, call a verifier, or loop. These are *inference-time orchestration* choices:
#pause
$ #text(fill: RED)[more steps] != #text(fill: GREEN)[a guaranteed better plan] $
#pause
#align(center, text(size: 15pt, fill: MUTED)[quality depends on model capability, state quality, tool reliability, and a suitable *stopping rule*])

== Agent evaluation #D

Measure much more than final-answer accuracy:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Dimension*], [*Question*]),
  [correctness], [was the requested task completed under realistic tool failures?],
  [efficiency], [how many tool calls, actions, and tokens? latency?],
  [robustness], [what happens when a tool returns an error?],
  [safety], [were permissions and sensitive actions respected?],
  [calibration], [did the system know when to stop or ask?],
))

== Agent failure modes #Q

#pause
- wrong tool choice; infinite loops; invented / stale state;
#pause
- *prompt injection* through tool output; overconfident *irreversible* actions.
#pause
#alertbox[An agent is a *systems* problem, not only an accuracy metric. Small per-step errors *compound* over a long loop.]

== Prompt injection is a boundary problem #Q

Untrusted web pages, documents, and tool results can carry instructions that *conflict* with the user's goal.
#pause
#alertbox[The defense is not "tell the model to ignore it" alone.]
#pause
It is a *systems* defense: data provenance · action allowlists · sandboxing · confirmation for high-impact actions · output validation.

== Retrieval: knowledge as external state #V

#ragflow
#pause
#align(center)[The most common non-agentic wrapper — *retrieval-augmented generation* (RAG).]
#pause
#align(center, text(size: 14pt, fill: MUTED)[same "state is external" idea — facts live in an *index*; update the index, not the model])

// ═══════════════════════════ PART IV — Reasoning and test-time compute ═══════════════════════════
= Part IV · Reasoning and test-time compute

== Chain-of-thought is an observable artifact #Q

Intermediate text can help a model *decompose* a task. But beware:
#pause
$ #text(fill: TEAL)[plausible explanation] != #text(fill: RED)[faithful causal explanation] $
#pause
#alertbox[A displayed "reasoning trace" is not guaranteed to be a faithful account of the model's internal computation — it may be post-hoc, incomplete, or strategically shaped.]
#pause
#align(center, text(size: 15pt, fill: MUTED)[teach this *before* students meet confident demonstrations of "reasoning"])

== Reasoning-system ingredients #D

A reasoning system is three familiar parts:
#pause
+ a *base model* that can propose candidate solution steps;
#pause
+ a *reward, verifier, or search signal*;
#pause
+ an extra *inference-time budget* for sampling, checking, revising, or branching.
#pause
#align(center, text(size: 15pt, fill: MUTED)[this is *optimization and inference again* — an objective, proposals, and a procedure that allocates compute])

== Verifiers change the problem #D

Some tasks admit *cheap automatic checks*:
#pause
#two(
  notebox[*Verifiable* — unit tests for code, exact arithmetic, constraint satisfaction, a proof checker.],
  alertbox[*Hard to verify* — open-ended judgement, style, "is this a good essay?"],
)
#pause
#align(center, text(size: 15pt, fill: MUTED)[when verification is reliable, inference can *generate many candidates* and select or refine them])

== Search at inference time #V

#searchloop
#pause
#align(center, text(size: 14pt, fill: MUTED)[a *system-level* search loop — related to optimization, but running *after* training, at query time])

== Outcome versus process reward #D

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 6pt), align: (left, left, left, left),
  table.header([*Reward*], [*Checks*], [*Benefit*], [*Difficulty*]),
  [outcome $r_"out"(y)$], [final answer], [simple when verifiable], [sparse feedback],
  [process $r_"proc"(s_1 dots s_k)$], [intermediate steps], [can catch an early wrong step], [hard to label reliably],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[outcome reward is cheap but sparse; process reward guides the path but needs trustworthy step judgements])

== Test-time compute has real trade-offs #V

More samples or longer search can raise reliability on hard tasks — but not for free:
#pause
#fig("/lecture24/figures/test_time.svg", w: 52%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[a *reliable* verifier keeps helping; a *flawed* one caps the gain — and every extra sample costs latency & money])

== Worked: best-of-$N$ with a reliable verifier #D

One sample succeeds with probability $p = 0.4$. With a *perfect* verifier, best-of-$N$ succeeds if *any* sample is correct:
#pause
$ P("success") = 1 - (1 - p)^N $
#pause
$ N = 5: quad 1 - 0.6^5 = 1 - 0.07776 = #text(fill: ACC, weight: 600)[0.922] $
#pause
#alertbox[If instead you took a *majority vote* with $p = 0.4 < 0.5$, more samples make it *worse* (Condorcet) — the verifier is what makes search pay off.]

== Allocating a fixed compute budget #D

For a fixed model, how should you spend a query's compute?
#pause
- one long candidate · many independent candidates · a search tree;
- candidate $+$ verifier · candidate $+$ tool calls.
#pause
#result[there is no universal strategy — the task's *verification structure* decides]

== Reasoning is old friends in new clothing #Q

#pause
This whole part is *optimization and inference*, relabeled:
#pause
#two(
  notebox[*Objective* supplies a signal (reward / verifier).],
  notebox[*Procedure* allocates limited compute to *search* over model proposals.],
)
#pause
#align(center, text(size: 15pt, fill: MUTED)[nothing here escapes the course — it *recombines* the course at inference time])

== Mini activity: what can be verified? #Q

For each system, ask: *what is automatically checkable? where is process feedback risky? what latency is acceptable?*
#pause
#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 6pt), align: (left, left, left, left),
  table.header([*System*], [*Verifiable?*], [*Process risk*], [*Latency*]),
  [Sudoku solver], [yes — check constraints], [low], [tight],
  [code generator], [partly — run tests], [medium], [moderate],
  [medical assistant], [rarely — no cheap oracle], [high], [varies],
))

// ═══════════════════════════ PART V — Interpretability ═══════════════════════════
= Part V · Interpretability — inspect representations with humility

== The residual stream as a shared workspace #D

Every Transformer block *reads* a representation and *adds* to it:
#pause
$ h_(ell+1) = h_ell + "Attn"(h_ell) + "MLP"(h_ell + "Attn"(h_ell)) $
#pause
#residstream
#pause
#align(center, text(size: 14pt, fill: MUTED)[additive updates mean *features can persist* and later components can read what earlier ones wrote])

== Board example: additive updates #D

Let the stream at block 5 be
$ h_5 = [0.2, 0.5, 0.1, -0.9] $
#pause
Attention adds $[0, 0.3, 0.4, 0]$ and the MLP adds $[0.1, -0.1, 0, 0.2]$:
#pause
$ h_6 = [0.2 + 0.1, thin 0.5 + 0.3 - 0.1, thin 0.1 + 0.4, thin -0.9 + 0.2] = #text(fill: ACC, weight: 600)[$[0.3, 0.7, 0.5, -0.7]$] $
#pause
#align(center, text(size: 15pt, fill: MUTED)[coordinate-wise addition — a component can *write into* a feature a later component *reads*])

== From neurons to features #D

A single neuron can join many unrelated behaviours; one behaviour can be spread over many neurons:
#pause
$ #text(fill: RED)[neuron] != #text(fill: TEAL)[human concept] $
#pause
#align(center, text(size: 15pt, fill: MUTED)[so we study *features and circuits* in the residual stream, not one activation in isolation])

== What attention maps can and cannot say #Q

#pause
#two(
  notebox[*Can* — show one *route* by which information is mixed between tokens.],
  alertbox[*Cannot* — by themselves establish the *semantic cause* of a prediction.],
)
#pause
#align(center, text(size: 15pt, fill: MUTED)[attention is *one information-routing signal* — a starting point, not a proof])

== Superposition intuition #D

A $d$-dimensional representation can encode *more than $d$* conceptual features by combining them in *overlapping directions*:
#pause
#align(center, text(size: 16pt)[few coordinates, many concepts — packed into non-orthogonal directions])
#pause
#notebox[This is why reading one coordinate rarely reveals one concept — concepts are *entangled* by design.]

== Sparse autoencoders: one picture #V

An SAE tries to *unpack* superposition into a wide, mostly-zero dictionary:
#pause
#saediagram
#pause
$ L = norm(h - hat(h))_2^2 + lambda norm(a)_1 quad quad #text(size: 14pt, fill: MUTED)[(reconstruction $+$ sparsity)] $
#pause
#align(center, text(size: 14pt, fill: MUTED)[the *hope*: a sparse code separates concepts entangled in $h$ — some become individually inspectable])

== Interpretation needs intervention #D

Finding a *correlated* feature is not enough. Stronger evidence:
#pause
+ *activate, remove, or patch* a representation component;
#pause
+ observe a *predicted* behavioural change;
#pause
+ *rule out* nearby confounds.
#pause
#align(center, text(size: 15pt, fill: MUTED)[this parallels causal reasoning — correlation is a start, not a complete explanation])

== Interpretability's honest limits #Q

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Tempting claim*], [*More honest claim*]),
  ["we read the model's thoughts"], [we found a feature / circuit *correlated* with a behaviour],
  ["attention explains the answer"], [attention is *one* information-routing signal],
  ["one visualization proves safety"], [coverage and *causal* validation remain limited],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[real circuits and features *can* be found — robustly explaining a frontier model is an *open problem*])

// ═══════════════════════════ PART VI — Efficiency, alignment, responsibility ═══════════════════════════
= Part VI · Efficiency, alignment, responsibility

== Efficiency, at a high level #D

None of this is free. Familiar levers make inference affordable:
#pause
#two(
  notebox[*Smaller / faster* — quantization, distillation, pruning, mixture-of-experts (only some parameters fire).],
  notebox[*Reuse compute* — KV caching, batching, speculative decoding, retrieval instead of memorizing.],
)
#pause
#align(center, text(size: 15pt, fill: MUTED)[efficiency is an *engineering objective* — traded against accuracy, latency, and cost (L23)])

== Alignment, at a high level #D

"Alignment" is, mechanically, *another objective plus another signal*:
#pause
- collect *preferences* (which response is better?);
- train a *reward / preference* model, or optimize directly against it;
- fine-tune the base model toward preferred behaviour.
#pause
#alertbox[It is *not* a solved guarantee — a preference model is itself a learned, imperfect approximation that can be gamed.]

== Responsible deployment questions #Q

Before deploying a frontier system, ask:
#pause
- who can be harmed by a false positive or a false *action*?
- what data or tools are *sensitive*?
- what *audit trail* exists?
- when must a *human approve* an action?
- how will failures be *reported and corrected*?
#pause
#align(center, text(size: 15pt, fill: MUTED)[technical design, governance, and product judgement — together, not separately])

// ═══════════════════════════ PART VII — Synthesis ═══════════════════════════
= Part VII · Synthesis — the whole course, recombined

== Multimodal / generative agents are compositions #V

The most impressive systems are the *most familiar parts*, composed:
#pause
#mmagent
#pause
#align(center, text(size: 14pt, fill: MUTED)[L17 representation alignment $+$ L23 inference constraints — new *capabilities*, and new *failure channels*])

== Course map: the representation thread #V

#repthread
#pause
#align(center, text(size: 14pt, fill: MUTED)[not identical representations — but each one *determines what the next computation can efficiently learn or control*])

== Course map: the objective thread #D

Every module was, at heart, a *choice of objective*:
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 5.5pt), align: (left, left),
  table.header([*Module*], [*Central objective*]),
  [classifiers · LLMs], [negative log-likelihood],
  [regularization], [NLL $+$ constraints / priors],
  [self-supervision], [constructed prediction / agreement],
  [VAE], [likelihood lower bound (ELBO)],
  [GAN], [minimax adversarial objective],
  [diffusion], [noise-prediction regression],
  [alignment · reasoning], [preference- or verifier-driven signal],
))

== Course map: the systems thread #V

A model does *not* end when training loss stops falling:
#pause
#systemsthread
#pause
#align(center, text(size: 14pt, fill: MUTED)[training $->$ pretraining $->$ adaptation $->$ inference $->$ tools · search · deployment])

== The whole course, in one table #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 5.5pt), align: (left, left, left),
  table.header([*Course idea*], [*Frontier manifestation*], [*Still just…*]),
  [learned representations], [foundation & multimodal embedding spaces], [features],
  [objectives & optimization], [preference / reward training, verifiers], [a loss $+$ gradients],
  [attention & autoregression], [LLMs, VLMs, tool-action policies], [$p(y_t | y_(<t))$],
  [generative modeling], [diffusion, world-model-like simulation], [sampling from $p_theta$],
  [inference systems], [caching, search, test-time compute], [how the model is used],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[same left column all semester — only the middle column keeps being renamed])

== Read a claim like a researcher #Q

For any new result, annotate six fields:
#pause
$ #text(fill: INK)[claim] thick | thick #text(fill: TEAL)[data] thick | thick #text(fill: BLUE)[metric] thick | thick #text(fill: ACC)[baseline] thick | thick #text(fill: RED)[cost] thick | thick #text(fill: GREEN)[failure case] $
#pause
And ask: is the gain from *architecture, scale, data, objective,* or *inference budget*? Which *baseline* is missing? Which *failure case* is not measured?
#pause
#notebox[Use one short abstract or announcement as a live in-class exercise.]

== Open problems #Q

The course gives a *map*, not a solved field. Still open:
#pause
- reliable generalization under *distribution shift*;
- data *provenance, privacy, and bias*;
- *factuality, calibration,* and tool-use *safety*;
- scalable *evaluation* of generative & agentic systems;
- *efficient, interpretable,* and *controllable* representations.

// ═══════════════════════════ PART VIII — Full circle ═══════════════════════════
= Part VIII · Full circle — back to Lecture 1

== What has changed since Lecture 1? #Q

Lecture 1 asked: how does a *differentiable model* turn inputs into labels? The *same machinery* now also:
#pause
- learns *reusable representations*;
- *generates* samples;
- *aligns* to preferences or conditions;
- operates *inside tools and search loops*.
#pause
#result[the scale changed; the core vocabulary stayed useful]

== What has not changed? #D

The founding concerns of Lecture 1 are still central:
#pause
$ #text(fill: TEAL)[data quality] quad #text(fill: BLUE)[objective design] quad #text(fill: ACC)[evaluation] quad #text(fill: GREEN)[deployment constraints] $
#pause
#alertbox[A larger model does *not* remove the need to define the task, detect distribution shift, or account for errors.]

== Lecture 1 returns — every loss is a likelihood #V

On day one we drew this: a loss *is* a likelihood, a regularizer *is* a prior. The frontier just adds one more signal into the *same* objective:
#pause
#l1map
#pause
#align(center, text(size: 14pt, fill: MUTED)[preference and verifier signals are *new likelihoods* feeding the *same* differentiable objective])

== The Lecture 1 reflex, extended #V

#align(center, text(size: 20pt)[
  See a *loss* $->$ ask _"what likelihood produced it?"_
  #v(5pt)
  See a *regularizer* $->$ ask _"what prior produced it?"_
  #v(5pt)
  See an *agent* $->$ ask _"what model, state, tools, and loop?"_
  #v(5pt)
  See a *reasoning trace* $->$ ask _"is it faithful, or just plausible?"_
])
#pause
#align(center, text(size: 15pt, fill: MUTED)[the same reflex, now pointed at the frontier])

== Closing synthesis #D

#result[Deep learning is a small set of ideas, recombined at scale.]
#pause
#v(4pt)
You can now distinguish a *new architecture*, a *new objective*, a *new dataset*, and a *new systems wrapper* — and ask whether the *evidence matches the claim*.

// ═══════════════════════════ PART IX — Activities and assessment ═══════════════════════════
= Part IX · Activities, and how to keep going

== Final group activity #Q

Each group gets one system — an *image generator*, a *coding assistant*, a *medical triage tool*, or a *retrieval chatbot*. Produce:
#pause
+ architecture / representation;
+ likely training objective;
+ inference / system loop;
+ one evaluation metric;
+ one serious failure mode *and* a mitigation.
#pause
#align(center, text(size: 15pt, fill: MUTED)[use the four questions from Part I — one capability claim, one likely failure])

== A one-page model card #D

Turn responsible-AI talk into an *engineering artifact*. For your group's system, draft:
#pause
+ intended use;
+ data & objective assumptions;
+ evaluation evidence;
+ known limitations;
+ human oversight & escalation path.
#pause
#notebox[A model card is where architecture, objective, evaluation, and deployment meet on one page.]

== How to read a new paper next year #Q

Read in *this order* — do not start with the equations:
#pause
+ abstract & the *claim*;
+ model / representation *diagram*;
+ *objective* and *data*;
+ evaluation *table & baselines*;
+ *limitations* and compute cost;
+ *only then* the detailed method.
#pause
#align(center, text(size: 15pt, fill: MUTED)[know *what question is being answered* before you get lost in *how*])

== A research-question generator #Q

Complete one sentence — it returns you to L17's representation-design skills:
#pause
#result[if the representation were invariant to \_\_\_ but preserved \_\_\_, a useful pretext task might be \_\_\_, evaluated by \_\_\_.]
#pause
#align(center, text(size: 15pt, fill: MUTED)[invariance · what to preserve · a self-supervised signal · an evaluation — every good project fills these four blanks])

== Exam-style synthesis question #Q

#notebox[*"A team proposes a multimodal agent for scientific literature review. Identify its likely representation modules, training objectives, inference loop, two evaluation metrics, and two failure modes."*]
#pause
#align(left, text(size: 18pt)[A strong answer draws on the *whole course* — vision & language encoders, retrieval, a tool loop, contrastive & NLL objectives, and hallucination & injection failures — *not* memorized product facts.])

== The limits of course coverage #Q

The course gives a map, not mastery of every system. Sometimes the rigorous answer is:
#pause
#result[we do not yet have enough evidence to claim this works reliably]
#pause
#align(center, text(size: 15pt, fill: MUTED)[reward this judgement explicitly — calibrated uncertainty is a skill, not a hedge])

== Retrieval questions #Q

#pause
+ Why is an agent *not* a distinct neural-network architecture?
#pause
+ When can test-time *search* beat a *larger* model?
#pause
+ Why is a visible *reasoning trace* not guaranteed to be faithful?
#pause
+ What evidence makes an *interpretability* claim stronger?
#pause
+ Which earlier lecture best explains *VLM-agent* systems, and why?

== The closing claim #D

#result[A model is not a product, a benchmark is not deployment, and a new name is not necessarily a new idea.]
#pause
#v(4pt)
#align(center, text(size: 16pt, fill: MUTED)[the durable outcome: decompose any claim into *data · representations · architecture · objective · inference · evidence*])

#focus-slide[
  Deep learning is a *small set of ideas*, recombined at scale.
  #v(14pt)
  #set text(size: 21pt)
  You now have the vocabulary to place *anything that comes next.*
  #v(16pt)
  #text(size: 16pt, fill: rgb("#B9C6C8"))[Thank you — ES 667 · Deep Learning.]
]
