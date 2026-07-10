// Self-Supervised Learning and Representation Learning
// Lecture 18 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture18/L18-self-supervised.typ /tmp/L18.pdf
//   typst compile --root . --input handout=true lecture18/L18-self-supervised.typ /tmp/L18h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.
// Schematic figures: python3 lecture18/diagrams/l18_figs.py

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Self-Supervised Learning],
  subtitle: [Labels hidden in plain sight — representation learning],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let cream = rgb("#EFEEEB")
#let rect = fletcher.shapes.rect

// ═══════════════════════════ fletcher diagrams (centerpieces) ═══════════════════════════

// ── (map) the course's representation map: a snaking chain, SSL highlighted ──
#let repmap = align(center, diagram(spacing: (14mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(pos, lbl, c, f) = node(pos, align(center, text(size: 9.5pt, lbl)), shape: rect, corner-radius: 4pt, inset: 6pt, stroke: 0.9pt + c, fill: f)
  bx((0, 0.8), [CNN feature \ hierarchies], TEAL, TEAL.lighten(82%))
  bx((1.9, 0.8), [transfer \ learning], TEAL, TEAL.lighten(82%))
  bx((3.8, 0.8), [language-model \ pretraining], BLUE, BLUE.lighten(85%))
  bx((3.8, -0.7), [self-supervision \ *(you are here)*], ACC, ACC.lighten(78%))
  bx((1.9, -0.7), [CLIP / VLMs \ (next lecture)], BLUE, BLUE.lighten(85%))
  bx((0, -0.7), [latent generative \ models], GREEN, GREEN.lighten(82%))
  edge((0, 0.8), (1.9, 0.8), "-|>", stroke: 0.8pt + MUTED)
  edge((1.9, 0.8), (3.8, 0.8), "-|>", stroke: 0.8pt + MUTED)
  edge((3.8, 0.8), (3.8, -0.7), "-|>", stroke: 0.8pt + MUTED)
  edge((3.8, -0.7), (1.9, -0.7), "-|>", stroke: 0.8pt + MUTED)
  edge((1.9, -0.7), (0, -0.7), "-|>", stroke: 0.8pt + MUTED)
}))

// ── (A) SimCLR pipeline: x -> two augmentations -> shared encoder f -> proj head g -> InfoNCE ──
#let simclr = align(center, diagram(spacing: (10mm, 7mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(pos, lbl, c, f) = node(pos, align(center, text(size: 8.5pt, lbl)), shape: rect, corner-radius: 4pt, inset: 5pt, stroke: 0.9pt + c, fill: f)
  node((0, 0), text(size: 10pt)[image $x$], radius: 6mm, fill: cream)
  // top branch
  bx((1.2, 0.95), [augment \ $t_1$], TEAL, TEAL.lighten(85%))
  bx((2.5, 0.95), [encoder \ $f$], TEAL, TEAL.lighten(82%))
  node((3.6, 0.95), $h_1$, radius: 4.5mm, fill: TEAL.lighten(70%), stroke: 0.9pt + TEAL)
  bx((4.7, 0.95), [proj \ head $g$], ACC, ACC.lighten(82%))
  node((5.8, 0.95), $z_1$, radius: 4.5mm, fill: ACC.lighten(72%), stroke: 0.9pt + ACC)
  // bottom branch
  bx((1.2, -0.95), [augment \ $t_2$], TEAL, TEAL.lighten(85%))
  bx((2.5, -0.95), [encoder \ $f$], TEAL, TEAL.lighten(82%))
  node((3.6, -0.95), $h_2$, radius: 4.5mm, fill: TEAL.lighten(70%), stroke: 0.9pt + TEAL)
  bx((4.7, -0.95), [proj \ head $g$], ACC, ACC.lighten(82%))
  node((5.8, -0.95), $z_2$, radius: 4.5mm, fill: ACC.lighten(72%), stroke: 0.9pt + ACC)
  // loss
  bx((7.0, 0), [InfoNCE \ pull $z_1, z_2$ \ together], GREEN, GREEN.lighten(82%))
  edge((0, 0), (1.2, 0.95), "-|>", stroke: 0.7pt + MUTED)
  edge((0, 0), (1.2, -0.95), "-|>", stroke: 0.7pt + MUTED)
  for (a, b) in ((1.2, 2.5), (2.5, 3.6), (3.6, 4.7), (4.7, 5.8)) {
    edge((a, 0.95), (b, 0.95), "-|>", stroke: 0.7pt + MUTED)
    edge((a, -0.95), (b, -0.95), "-|>", stroke: 0.7pt + MUTED)
  }
  edge((5.8, 0.95), (7.0, 0), "-|>", stroke: 0.8pt + GREEN)
  edge((5.8, -0.95), (7.0, 0), "-|>", stroke: 0.8pt + GREEN)
  node((2.5, 0), text(size: 8.5pt, fill: MUTED)[weights *shared*], stroke: none, fill: none)
}))

// ── (B) a contrastive batch, one anchor: 1 positive vs 2N-2 negatives ──
#let cbatch = align(center, diagram(spacing: (16mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), align(center, text(size: 9.5pt)[anchor $z_i$ \ (view A1)]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: TEAL.lighten(78%), stroke: 1pt + TEAL)
  node((2.4, 1.0), align(center, text(size: 9.5pt)[$z_j$ (view A2) \ *positive*]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: GREEN.lighten(80%), stroke: 1pt + GREEN)
  node((2.4, -1.0), align(center, text(size: 9pt)[B1 B2 C1 C2 D1 D2 \ *2N − 2 = 6 negatives*]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: RED.lighten(85%), stroke: 1pt + RED)
  edge((0, 0), (2.4, 1.0), "-|>", stroke: 1pt + GREEN, label: text(size: 8.5pt, fill: GREEN)[pull together], label-pos: 0.5, label-side: left)
  edge((0, 0), (2.4, -1.0), "-|>", stroke: 1pt + RED, label: text(size: 8.5pt, fill: RED)[push apart], label-pos: 0.5, label-side: right)
}))

// ── (C) MAE asymmetric encoder / decoder ──
#let maepipe = align(center, diagram(spacing: (11mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(pos, lbl, c, f) = node(pos, align(center, text(size: 9pt, lbl)), shape: rect, corner-radius: 4pt, inset: 6pt, stroke: 0.9pt + c, fill: f)
  bx((0, 0), [196 image \ patches], INK, cream)
  bx((1.5, 0.85), [49 visible \ patches], TEAL, TEAL.lighten(80%))
  bx((2.9, 0.85), [encoder $f$ \ (heavy)], TEAL, TEAL.lighten(82%))
  bx((2.9, -0.9), [mask tokens \ (147)], MUTED, MUTED.lighten(78%))
  bx((4.5, 0), [decoder \ (light)], ACC, ACC.lighten(82%))
  bx((6.1, 0), [reconstruct \ masked patches], GREEN, GREEN.lighten(82%))
  edge((0, 0), (1.5, 0.85), "-|>", stroke: 0.8pt + MUTED, label: text(size: 8pt, fill: MUTED)[keep 25%], label-side: left)
  edge((1.5, 0.85), (2.9, 0.85), "-|>", stroke: 0.8pt + TEAL)
  edge((2.9, 0.85), (4.5, 0), "-|>", stroke: 0.8pt + TEAL, label: text(size: 8pt, fill: TEAL)[encoded], label-side: left)
  edge((2.9, -0.9), (4.5, 0), "-|>", stroke: 0.8pt + MUTED, label: text(size: 8pt, fill: MUTED)[add masks], label-side: right)
  edge((4.5, 0), (6.1, 0), "-|>", stroke: 0.8pt + GREEN)
}))

// ── (D) BYOL teacher-student with stop-gradient + EMA loop ──
#let byol = align(center, diagram(spacing: (12mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(pos, lbl, c, f) = node(pos, align(center, text(size: 9pt, lbl)), shape: rect, corner-radius: 4pt, inset: 6pt, stroke: 0.9pt + c, fill: f)
  node((0, 1.0), $v_1$, radius: 5mm, fill: cream)
  bx((1.3, 1.0), [student \ encoder], TEAL, TEAL.lighten(82%))
  bx((2.8, 1.0), [predictor \ $q$], ACC, ACC.lighten(82%))
  node((0, -1.0), $v_2$, radius: 5mm, fill: cream)
  bx((1.3, -1.0), [teacher \ encoder], BLUE, BLUE.lighten(85%))
  bx((3.0, -1.0), [$op("sg")[z_xi (v_2)]$ \ stop-grad], BLUE, BLUE.lighten(88%))
  bx((4.6, 0), [match \ (MSE on \ normalized)], GREEN, GREEN.lighten(82%))
  edge((0, 1.0), (1.3, 1.0), "-|>", stroke: 0.8pt + MUTED)
  edge((1.3, 1.0), (2.8, 1.0), "-|>", stroke: 0.8pt + TEAL)
  edge((2.8, 1.0), (4.6, 0), "-|>", stroke: 0.8pt + TEAL)
  edge((0, -1.0), (1.3, -1.0), "-|>", stroke: 0.8pt + MUTED)
  edge((1.3, -1.0), (3.0, -1.0), "-|>", stroke: 0.8pt + BLUE)
  edge((3.0, -1.0), (4.6, 0), "-|>", stroke: 0.8pt + BLUE)
  edge((1.3, 1.0), (1.3, -1.0), "--|>", stroke: 1pt + ACC, label: text(size: 8.5pt, fill: ACC)[EMA update], label-side: left)
}))

#title-slide()

// ═══════════════════════════ PART I — Representation learning is the thread ═══════════════════════════
= Representation learning is the thread

== A network is also a representation map #D

Every classifier you have built already splits into two parts:
#pause
$ x quad ->^(f_theta) quad h quad ->^(g_phi) quad hat(y) $
#pause
#two(
  notebox[the *head* $g_phi$ solves *today's* task — one label set, one loss.],
  notebox[the *representation* $h = f_theta (x)$ is what we hope survives to *tomorrow's* task.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[the whole game of representation learning: make $h$ good *before* we know the downstream task])

== The course's representation map #V

One thread runs through the whole second half of the course:
#pause
#repmap
#pause
#align(center, text(size: 14pt, fill: MUTED)[everything here is about *learning $h$ from structure already in the data* — today's stop is self-supervision])

== Supervised learning's bargain, and the label bottleneck #D

Supervised learning strikes a clean bargain:
#pause
$ (x_i, y_i) quad -> quad "learn " f_theta (x_i) " useful for " y_i $
#pause
Labels make the objective crisp — but they *cost*. Contrast three data pools:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, center, left),
  table.header([*Pool*], [*Scale*], [*Cost*]),
  [curated labeled images], [$10^6$–$10^7$], [expensive human annotation],
  [web images / text / audio / sensors], [$10^9$+], [no manual labels; noisy and costly to curate],
  [what a student / lab can collect], [modest], [cheap, but tiny],
))
#pause
#align(center, text(size: 15pt, fill: RED)[expensive labels *bottleneck* the very scale that makes deep learning work])

== The central pivot #D

The whole field turns on one move:
#pause
#result[instead of *receiving* a human label, *manufacture* a target from the input itself]
#pause
#notebox[The data becomes its own teacher. Hide part of it, transform it, or pair it with a copy — and predict what you hid. Self-supervision turns *raw data into its own training signal*.]

== Three task families #D

Three recurring recipes — memorize the *pattern*, not a list of method names:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 7pt), align: (left, left, left),
  table.header([*Family*], [*Constructed target*], [*Main intuition*]),
  [contrastive / predictive], [another *related view*], [keep related things close],
  [masked reconstruction], [a *hidden part*], [infer context and structure],
  [teacher–student], [a *teacher representation*], [stable targets without labels],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[match related views $quad|quad$ reconstruct hidden parts $quad|quad$ match a slowly-changing teacher])

// ═══════════════════════════ PART II — Contrastive learning ═══════════════════════════
= Contrastive learning: make two views agree

== Hook: one object, two views #Q

Two hard crops of the *same* dog, and a crop of a *different* dog.
#pause
#two(
  notebox[what should the representation treat as the *same*? (the two views of one dog)],
  notebox[what should it treat as *different*? (the other dog)],
)
#pause
#align(center, text(size: 17pt)[which differences should the representation learn to *ignore*?])

== Augmentations define the invariance #D

Two random transforms make two views — each *declares* what does *not* change identity:
#pause
$ x quad ->^(t_1) quad v_1, quad quad quad x quad ->^(t_2) quad v_2 $
#pause
- *random crop* $->$ ignore exact location and framing;
- *colour jitter* $->$ ignore global colour statistics;
- *blur* $->$ ignore fine texture; *small geometric* $->$ ignore small pose changes.
#pause
#align(center, text(size: 15pt, fill: MUTED)[the set of augmentations *is* the set of invariances you ask for])

== A caution: an augmentation is a modeling assumption #D

The invariances are a choice — and a *wrong* choice ruins the representation:
#pause
#alertbox[*rotation* is harmless for natural photos, but *disastrous* for digit recognition (6 vs 9). *time-reversal* is invalid for causal signals. *colour* jitter is fine for species, unacceptable for medical staining.]
#pause
#align(center, text(size: 16pt, fill: RED)[wrong invariance $=>$ wrong representation — SSL moves domain expertise into *objective design*])

== The SimCLR pipeline #V

SimCLR (Chen et al. 2020): two views, a *shared* encoder, a projection head, a contrastive loss.
#pause
#simclr
#pause
#align(center, text(size: 14pt, fill: MUTED)[distinguish the encoder output $h = f(v)$ from the contrastive vector $z = g(h)$])

== Why a projection head? #D

The contrastive loss *discards* nuisance information to make views agree. Let $g$ absorb that pressure:
#pause
#two(
  notebox[*$z = g(h)$* — trained hard for the contrastive task; may throw away detail. *Discarded* after pretraining.],
  notebox[*$h = f(v)$* — kept *richer*, less over-specialized to the pretext task. This is what transfers.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[measure and *use* $h$; optimize *through* $z$])

== A contrastive batch #D

From $N$ source images make $2N$ views (two per image):
#pause
- pick any view as the *anchor* $z_i$;
- exactly *one* other view is its *positive* (the sibling of the same source);
- the remaining $2N - 2$ views act as *negatives*.
#pause
#cbatch

== Similarity: compare by angle, not length #D

Normalize the vectors, then take a cosine similarity:
#pause
$ s(z_i, z_j) = (z_i^top z_j) / (norm(z_i) thin norm(z_j)) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[dividing by the norms means similarity is the *angle* between embeddings, not their magnitude])

== InfoNCE: identify the positive among candidates #D

For anchor $i$ with positive $j$, the loss (van den Oord et al. — InfoNCE):
#pause
$ ell_(i,j) = - log (exp(s(z_i, z_j) \/ tau)) / (sum_(k != i) exp(s(z_i, z_k) \/ tau)) $
#pause
#result[this is just a $(2N - 1)$-way *softmax classification*: "which of the other views is my positive?"]
#pause
#align(center, text(size: 15pt, fill: MUTED)[not a mysterious new loss — cross-entropy where the "classes" are the other views in the batch])

== Board: temperature is a contrast amplifier #D

One anchor, three similarities: $0.9$ (positive), $0.2$, $-0.1$. Softmax the scaled scores $s \/ tau$:
#pause
#fig("/lecture18/figures/temperature_contrast.svg", w: 66%)
#pause
#align(center, text(size: 15pt, fill: MUTED)[smaller $tau$ *sharpens the competition* — the model is forced to beat the *hardest* negatives])

== What prevents trivial constant features? #D

Suppose the encoder collapses — every input maps to the *same* embedding.
#pause
- then $s(z_i, z_"positive") = s(z_i, z_"negative")$ for all pairs;
- the positive can *never* win the softmax — the loss stays high.
#pause
#result[the *denominator* (the negatives) makes collapse a *bad* solution — competition is built in]
#pause
#align(center, text(size: 15pt, fill: MUTED)[contrast between positive and negatives is exactly what rules out the constant shortcut])

== The similarity-matrix view #V

Stack all $2N$ views: entry $(i, j)$ is $s(z_i, z_j) \/ tau$. Each row is one softmax problem.
#pause
#align(center, attn-matrix(
  ("A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"),
  values: (
    (1.00, 0.90, 0.18, 0.12, 0.25, 0.15, 0.20, 0.10),
    (0.90, 1.00, 0.14, 0.22, 0.16, 0.28, 0.12, 0.19),
    (0.18, 0.14, 1.00, 0.90, 0.21, 0.13, 0.17, 0.24),
    (0.12, 0.22, 0.90, 1.00, 0.15, 0.26, 0.20, 0.11),
    (0.25, 0.16, 0.21, 0.15, 1.00, 0.90, 0.19, 0.14),
    (0.15, 0.28, 0.13, 0.26, 0.90, 1.00, 0.23, 0.18),
    (0.20, 0.12, 0.17, 0.20, 0.19, 0.23, 1.00, 0.90),
    (0.10, 0.19, 0.24, 0.11, 0.14, 0.18, 0.90, 1.00),
  ),
  mask: (i, j) => i == j,
  annotate: false,
  boxes: ((0, 1, INK), (1, 0, INK), (2, 3, INK), (3, 2, INK),
          (4, 5, INK), (5, 4, INK), (6, 7, INK), (7, 6, INK)),
  colorbar: true,
  cell: 9mm,
  q-label: [anchor #h(0.25em) $z_i$],
  k-label: [view #h(0.25em) $z_j$],
))
#pause
#align(center, text(size: 14pt, fill: MUTED)[one boxed positive per row, the diagonal ignored — the *same* "correct match in a batch" geometry that powers CLIP])

== Evaluation: feature quality is not head capacity #D

A powerful head can mask a weak representation. Separate the two:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, center, left),
  table.header([*Protocol*], [*Encoder*], [*What it tests*]),
  [linear probe], [frozen], [representation quality (linearly readable?)],
  [fine-tuning], [updated], [downstream ceiling],
  [retrieval / nearest-neighbour], [frozen], [semantic geometry, no new head],
  [dense-task transfer], [often updated], [locality and spatial detail],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[a strong *frozen linear probe* is the real signal that pretraining worked])

== Interactive: contrastive learning #I

#interbox(link-to: IA + "vision-ssl")[
  Build two augmented views, watch them pass through the shared encoder and projection head, and see the InfoNCE softmax pick the positive out of the batch as you drag the temperature $tau$.
]

// ═══════════════════════════ PART III — Masked reconstruction ═══════════════════════════
= Masked reconstruction: infer what is missing

== A different question #D

Contrastive and masked modeling ask *different* questions of the data:
#pause
#two(
  notebox[*contrastive* — "which view *matches* this one?" Pressure toward *invariance*.],
  notebox[*masked* — "what must be *true* about the world for the hidden part to make sense?" Pressure toward *context*.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[no augmentations, no negatives — the supervision is the *hidden content itself*])

== The masked-autoencoder setup #V

MAE (He et al. 2022): hide most of the image, reconstruct it from the little that remains.
#pause
#maepipe
#pause
#align(center, text(size: 14pt, fill: MUTED)[patches $->$ hide a large subset $->$ encoder sees visible only $->$ light decoder fills the rest])

== MAE is asymmetric #D

The two halves do very different amounts of work:
#pause
- the *expensive encoder* processes *only the visible* patches — cheap because most are gone;
- the *lightweight decoder* gets encoded-visible tokens *plus* learned *mask tokens*;
- the decoder is *discarded* after pretraining — only $f$ transfers.
#pause
#notebox[For a $224 times 224$ image with $16 times 16$ patches: $N = (224 \/ 16)^2 = 196$. At 75% masking the heavy encoder sees only $196 times 0.25 = #text(fill: ACC, weight: 600)[49]$ tokens — a learning *and* an efficiency choice.]

== A simple reconstruction objective #D

Mean-squared pixel error, computed *only on the masked patches* $M$:
#pause
$ L_"mask" = 1 / abs(M) sum_(p in M) norm(x_p - hat(x)_p)^2 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[no label anywhere — the target $x_p$ is just the pixels we chose to hide])

// 14×14 = 196 patches; the 49 visible (25%) are a fixed seeded scatter,
// so the masked set below is the MAE-style 75% (147 patches).
#let mae-visible = (18, 23, 24, 25, 27, 28, 32, 34, 42, 43, 45, 58, 60, 62, 64,
  71, 77, 89, 91, 92, 93, 96, 106, 112, 113, 116, 118, 121, 122, 123, 128, 131,
  132, 136, 138, 140, 152, 154, 158, 165, 177, 179, 184, 185, 186, 192, 193, 194, 195)
#let mae-masked = range(196).filter(i => not mae-visible.contains(i))

== Why mask heavily? #V

The mask ratio decides whether the task is *trivial* or *forces semantics*:
#pause
#two(
  align(center, patchify(image: 14, patch: 1, mask: mae-masked,
    numbering: false, cell: 3mm)),
  [
    - a *few* hidden patches $->$ copy local texture from neighbours — *trivial*;
    #v(3pt)
    - *≈75%* hidden $->$ local copying fails $->$ the model must use *global context*.
    #v(6pt)
    #text(size: 13.5pt, fill: MUTED)[No universal number: BERT masks ≈15% (text is dense), MAE ≈75% (images are redundant) — mask ratio is a *modality* choice.]
  ],
  r: (0.9fr, 1.1fr),
)

== Contrastive versus masked #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left, left),
  table.header([], [*Contrastive*], [*Masked reconstruction*]),
  [supervision], [a related view], [hidden content],
  [main pressure], [invariance], [contextual prediction],
  [common strength], [global classification / retrieval], [spatial / dense transfer],
  [key design choice], [augmentations, negatives, $tau$], [mask pattern, reconstruction target],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[two different assumptions about *what makes two inputs related* — often complementary])

== Bridge to language models #D

The masked idea is exactly what pretrains modern language models:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 7pt), align: (left, left),
  table.header([*Model*], [*Predict…*]),
  [BERT], [masked *tokens* from surrounding context],
  [GPT], [the *next token* from preceding context],
  [MAE], [masked image *patches* from visible patches],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[language-model pretraining is *not* separate from SSL — it is its most familiar instance])

== One unifying view #D

Strip away the modality and every recipe here is the same objective:
#pause
#result[learn $quad p("hidden or future part" | "observed context")$]
#pause
#align(center, text(size: 16pt, fill: MUTED)[the *output spaces* differ — tokens, pixels, audio frames — but the self-supervised *logic* is shared])

// ═══════════════════════════ PART IV — Teacher-student consistency ═══════════════════════════
= Teacher–student consistency: learn without negatives

== The apparent problem: collapse #D

Drop the negatives. Train two networks *only to agree* on two views — what breaks?
#pause
#alertbox[Both can output the *same constant* vector $[0, 0, dots, 0]$ for every input. The matching loss is *perfect*, yet the representation says *nothing*. This is *collapse*.]
#pause
#align(center, text(size: 16pt, fill: RED)[agreement alone is *not* enough — we need an asymmetry that makes the constant unreachable])

== BYOL at a glance #V

BYOL (Grill et al. 2020): a student *predicts* a slowly-moving *teacher* — a design that can avoid collapse in practice.
#pause
#byol
#pause
#align(center, text(size: 14pt, fill: MUTED)[student path has an extra *predictor*; the teacher target is *stop-gradient*; teacher weights are an *EMA* of the student])

== The three asymmetries to teach #D

BYOL avoids the trivial solution by breaking the symmetry three ways:
#pause
+ a *predictor* $q$ exists *only* on the student path;
+ the teacher output is *stop-gradient* — no gradient flows into the teacher;
+ the teacher weights are an *exponential moving average* of the student.
#pause
#align(center, text(size: 16pt, fill: MUTED)[the two branches are *not* interchangeable — so copying a constant is not an easy fixed point])

== The EMA update #D

The teacher is never trained directly — it *trails* the student:
#pause
$ theta_"teacher" quad <- quad m thin theta_"teacher" + (1 - m) thin theta_"student" $
#pause
#notebox[Worked step, $m = 0.99$: teacher weight $0.40$, student now $0.70$ $-> quad 0.99 (0.40) + 0.01 (0.70) = #text(fill: ACC, weight: 600)[$0.403$]$. The teacher barely moves — a *stable target* to chase.]

== Why this does not collapse — a careful answer #D

#alertbox[*Do not say "the EMA alone prevents collapse."* That is the common exam mistake.]
#pause
In the working recipe, several ingredients act *together*:
#pause
- *stop-gradient* + *predictor asymmetry* break the student↔teacher symmetry;
- *normalization* keeps outputs from shrinking to zero;
- the *slowly-moving target* gives a stable, non-trivial thing to predict.
#pause
#align(center, text(size: 15pt, fill: MUTED)[remove any one and the argument weakens — it is the *combination* that works])

== DINO and multi-crop #D

DINO (Caron et al.) / DINOv2 (Oquab et al.) scale the teacher–student idea:
#pause
- student and teacher predict *distributions over features* from *differently cropped* views;
- *multi-crop*: two *global* views + several small *local* views — the student must match a global target from local content;
- result: *emergent* semantic regions and strong *frozen* features, with *no labels*.
#pause
#align(center, text(size: 15pt, fill: MUTED)[we keep this high-level — centering / sharpening details are beyond today])

== One-slide landscape, not a zoo #V

Do not memorize methods — file each under a *recipe*:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 7pt), align: (left, left),
  table.header([*If you meet…*], [*It is a variation on…*]),
  [MoCo, SwAV, SimSiam, Barlow Twins], [contrast / anti-collapse design],
  [BYOL, DINO], [teacher–student consistency],
  [MAE, masked language modeling], [reconstruct / predict hidden content],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[three recipes explain a decade of "new" methods])

// ═══════════════════════════ PART V — Across modalities + close ═══════════════════════════
= The same question across modalities

== Cross-modal comparison #V

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 9pt, y: 6pt), align: (left, left, left, left),
  table.header([*Data*], [*Contrastive / predictive*], [*Masked*], [*Consistency*]),
  [images], [augmented views; image–caption], [masked patches], [teacher / student crops],
  [text], [adjacent spans; sentence pairs], [masked / next token], [teacher / student encodings],
  [audio / time series], [nearby or future windows], [masked spans / channels], [noisy-view agreement],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[same three recipes, different output spaces — the objective *family* transfers, not the architecture])

== The transfer question — decide before you train #D

State the *invariance* and the *downstream task* *first*, then choose the objective:
#pause
#two(
  notebox[*"ignore colour"* — ideal for *species* recognition (a robin is a robin in any light).],
  alertbox[*"ignore colour"* — *unacceptable* for *medical* imaging, where stain colour is diagnostic.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[the pretext task must make the *knowledge you want* *necessary* to solve it])

== Common misconceptions #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left),
  table.header([*Misconception*], [*Correction*]),
  ["SSL has no labels."], [it has *targets constructed* from the data],
  ["Contrastive means semantic negatives."], [negatives are often *just batch alternatives*],
  ["BYOL works because of EMA."], [*several asymmetries* work together],
  ["Good reconstruction $=>$ good classifier."], [the *target* decides what features are kept],
))

== Exit ticket: design an SSL recipe #Q

*Wearable-sensor* stream, no labels, later used for activity recognition. In pairs, propose:
#pause
+ *two valid* augmentations (e.g. small time-warp, additive sensor noise);
+ *one invalid* augmentation (e.g. time-reversal — destroys causal order);
+ *one masked-prediction* target (e.g. reconstruct a masked time span);
+ *one evaluation* protocol (e.g. linear probe on a small labeled set).
#pause
#notebox[*One mental model:* choose a transformation or hidden part that makes the *desired knowledge necessary*. Everything else in SSL is an implementation of that decision.]

== Closing bridge to CLIP #D

We built a geometry: *two augmented views of one image* pulled together against a batch of negatives.
#pause
$ "two augmented views of an image" quad ==> quad "an image and its caption" $
#pause
#result[swap the second view for a *caption* — the same contrastive geometry becomes a shared *image–text* space]
#pause
#interbox(link-to: IA + "clip-zero-shot")[
  Next lecture: watch the batch similarity matrix and contrastive softmax reappear — now between images and text — to drive zero-shot classification.
]

#focus-slide[
  Self-supervision turns *raw data into its own training signal*.
  #v(10pt)
  #set text(size: 20pt)
  Three recipes: match related views · reconstruct hidden parts · match a slow teacher.
  #v(10pt)
  Next: *Vision-Language Models* — image views $->$ image–caption pairs $->$ a shared multimodal space.
]
