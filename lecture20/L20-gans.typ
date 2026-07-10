// Generative Adversarial Networks
// Lecture 20 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture20/L20-gans.typ /tmp/L20.pdf
//   typst compile --root . --input handout=true lecture20/L20-gans.typ /tmp/L20h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.
// Schematic figures: python3 lecture20/diagrams/l20_figs.py

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [Generative Adversarial Networks],
  subtitle: [Learn to generate by playing a game against a critic],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let cream = rgb("#EFEEEB")
#let rect = fletcher.shapes.rect
// diagram label: size inline math to match the surrounding small text (the theme forces equations to 19pt)
#let dl(sz, body) = { set text(size: sz); show math.equation: set text(size: sz); body }

// ═══════════════════════════ fletcher diagrams (centerpieces) ═══════════════════════════

// ── (A) the adversarial setup: z -> G -> fake -> D ; real -> D -> score ──
#let advsetup = align(center, diagram(spacing: (13mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 1), align(center, text(size: 10pt)[real image \ $x$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((0, -1), text(size: 11pt)[noise $z$], radius: 6mm, fill: cream)
  node((1.8, -1), align(center, text(size: 9.5pt)[generator \ $G_theta$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(80%), stroke: 1pt + ACC)
  node((3.5, -1), align(center, text(size: 10pt)[fake \ $tilde(x)$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(88%), stroke: 0.9pt + ACC)
  node((5.2, 0), align(center, text(size: 9.5pt)[discriminator \ $D_psi$]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: BLUE.lighten(82%), stroke: 1.2pt + BLUE)
  node((7.0, 0), align(center, text(size: 9.5pt)[real / fake \ score]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  edge((0, 1), (5.2, 0), "-|>", stroke: 0.9pt + BLUE, label: text(size: 8.5pt, fill: BLUE)[want $D -> 1$], label-side: left)
  edge((0, -1), (1.8, -1), "-|>", stroke: 0.8pt + MUTED)
  edge((1.8, -1), (3.5, -1), "-|>", stroke: 0.8pt + MUTED)
  edge((3.5, -1), (5.2, 0), "-|>", stroke: 0.9pt + ACC, label: text(size: 8.5pt, fill: ACC)[want $D -> 0$], label-side: right)
  edge((5.2, 0), (7.0, 0), "-|>", stroke: 0.9pt + INK)
}))

// ── (B) one training iteration as a cyclic loop ──
#let trainloop = align(center, diagram(spacing: (17mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(pos, lbl, c, f) = node(pos, align(center, dl(9.5pt, lbl)), shape: rect, corner-radius: 4pt, inset: 7pt, stroke: 1pt + c, fill: f)
  bx((0, 1), [1 · sample real $x$ \ and noise $z$], INK, cream)
  bx((3, 1), [2 · update $D$: \ real $-> 1$, fake $-> 0$], BLUE, BLUE.lighten(84%))
  bx((3, -1), [3 · sample \ fresh noise $z$], INK, cream)
  bx((0, -1), [4 · freeze $D$, \ update $G$ through $D$], ACC, ACC.lighten(82%))
  edge((0, 1), (3, 1), "-|>", stroke: 1pt + INK)
  edge((3, 1), (3, -1), "-|>", stroke: 1pt + INK)
  edge((3, -1), (0, -1), "-|>", stroke: 1pt + INK)
  edge((0, -1), (0, 1), "-|>", stroke: 1pt + INK, label: text(size: 9pt, fill: MUTED)[repeat], label-side: left)
}))

// ── (C) gradient path: forward z->G->D->L_G, dashed backward gradient into G ──
#let gradpath = align(center, diagram(spacing: (16mm, 13mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), text(size: 12pt)[$z$], radius: 5.5mm, fill: cream)
  node((1.5, 0), align(center, text(size: 10pt)[$G_theta$]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: ACC.lighten(80%), stroke: 1pt + ACC)
  node((3.0, 0), align(center, text(size: 9.5pt)[$D_psi$ \ (frozen ❄)]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: BLUE.lighten(84%), stroke: 1pt + BLUE)
  node((4.6, 0), align(center, text(size: 10pt)[$L_G$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  edge((0, 0), (1.5, 0), "-|>", stroke: 1pt + INK)
  edge((1.5, 0), (3.0, 0), "-|>", stroke: 1pt + INK)
  edge((3.0, 0), (4.6, 0), "-|>", stroke: 1pt + INK)
  edge((4.6, 0), (1.5, 0), "-|>", stroke: (paint: RED, thickness: 1.3pt, dash: "dashed"), bend: -42deg, label: text(size: 9pt, fill: RED)[$nabla_theta L_G$], label-side: right)
}))

// ── (D) minibatch D-update: real -> D, G(z).detach() -> D, no gradient to G ──
#let detachdiag = align(center, diagram(spacing: (12mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 1), align(center, dl(10pt)[real $x$ \ (label 1)]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((0, -1), text(size: 11pt)[$z$], radius: 5.5mm, fill: cream)
  node((1.7, -1), align(center, text(size: 12pt)[$G_theta$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(80%), stroke: 1pt + ACC)
  node((3.4, -1), align(center, dl(9pt)[$G(z)$.detach() \ (label 0)]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: rgb("#F0EDEA"), stroke: 0.9pt + MUTED)
  node((5.2, 0), align(center, text(size: 9.5pt)[$D_psi$]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: BLUE.lighten(82%), stroke: 1.2pt + BLUE)
  node((6.9, 0), align(center, text(size: 10pt)[$L_D$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  edge((0, 1), (5.2, 0), "-|>", stroke: 0.9pt + BLUE)
  edge((0, -1), (1.7, -1), "-|>", stroke: 0.8pt + MUTED)
  edge((1.7, -1), (3.4, -1), "-|>", stroke: (paint: RED, thickness: 1pt, dash: "dashed"), label: text(size: 8pt, fill: RED)[✂ stop-grad], label-side: right)
  edge((3.4, -1), (5.2, 0), "-|>", stroke: 0.9pt + ACC)
  edge((5.2, 0), (6.9, 0), "-|>", stroke: 0.9pt + INK)
}))

// ── (E) mode collapse: healthy G fans out, collapsed G maps everything to one output ──
#let mc_good = diagram(spacing: (12mm, 6mm), node-stroke: 0.8pt + INK, node-fill: white, {
  for (i, y) in (1, 0, -1).enumerate() { node((0, y), text(size: 9pt)[$z_#(i+1)$], radius: 4mm, fill: cream) }
  node((1.5, 0), align(center, text(size: 9.5pt)[$G$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(80%), stroke: 1pt + ACC)
  let outs = ((3, 1, TEAL), (3, 0, BLUE), (3, -1, GREEN))
  for (x, y, c) in outs { node((x, y), none, radius: 4.5mm, fill: c.lighten(55%), stroke: 0.9pt + c) }
  for y in (1, 0, -1) { edge((0, y), (1.5, 0), "-|>", stroke: 0.6pt + MUTED) }
  for (x, y, c) in outs { edge((1.5, 0), (x, y), "-|>", stroke: 0.8pt + c) }
})
#let mc_bad = diagram(spacing: (12mm, 6mm), node-stroke: 0.8pt + INK, node-fill: white, {
  for (i, y) in (1, 0, -1).enumerate() { node((0, y), text(size: 9pt)[$z_#(i+1)$], radius: 4mm, fill: cream) }
  node((1.5, 0), align(center, text(size: 9.5pt)[$G$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(80%), stroke: 1pt + ACC)
  node((3, 0), none, radius: 4.5mm, fill: RED.lighten(50%), stroke: 1.1pt + RED)
  for y in (1, 0, -1) { edge((0, y), (1.5, 0), "-|>", stroke: 0.6pt + MUTED) }
  edge((1.5, 0), (3, 0), "-|>", stroke: 1.1pt + RED)
})

// ── (F) conditional GAN: (z, c) -> G -> fake ; c also enters D ──
#let condgan = align(center, diagram(spacing: (13mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 1), text(size: 11pt)[$z$], radius: 5mm, fill: cream)
  node((0, -1), align(center, text(size: 9.5pt)[condition \ $c$]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: TEAL.lighten(80%), stroke: 0.9pt + TEAL)
  node((1.8, 0), align(center, text(size: 9.5pt)[$G$]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: ACC.lighten(80%), stroke: 1pt + ACC)
  node((3.4, 0), align(center, text(size: 10pt)[$tilde(x)$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(88%), stroke: 0.9pt + ACC)
  node((5.1, 0), align(center, text(size: 9.5pt)[$D$]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: BLUE.lighten(82%), stroke: 1.2pt + BLUE)
  node((6.8, 0), align(center, text(size: 9pt)[real / fake \ under $c$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  edge((0, 1), (1.8, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((0, -1), (1.8, 0), "-|>", stroke: 0.9pt + TEAL)
  edge((1.8, 0), (3.4, 0), "-|>", stroke: 0.9pt + ACC)
  edge((3.4, 0), (5.1, 0), "-|>", stroke: 0.9pt + INK)
  edge((0, -1), (5.1, 0), "-|>", stroke: 0.9pt + TEAL, bend: -32deg, label: text(size: 8.5pt, fill: TEAL)[$c$ to $D$ too], label-side: right)
  edge((5.1, 0), (6.8, 0), "-|>", stroke: 0.9pt + INK)
}))

// ── (G) generator geometry: simple z -> upsampling blocks -> image manifold ──
#let gengeo = align(center, diagram(spacing: (12mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), align(center, text(size: 9.5pt)[$z in RR^d$ \ (simple)]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: cream, stroke: 0.9pt + INK)
  node((1.6, 0), align(center, text(size: 9pt)[upsample \ $4 times 4$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((3.0, 0), align(center, text(size: 9pt)[upsample \ $16 times 16$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: BLUE.lighten(84%), stroke: 0.9pt + BLUE)
  node((4.4, 0), align(center, text(size: 9pt)[upsample \ $64 times 64$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  node((5.9, 0), align(center, text(size: 9.5pt)[image \ (manifold)]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  for (a, b) in ((0, 1.6), (1.6, 3.0), (3.0, 4.4), (4.4, 5.9)) { edge((a, 0), (b, 0), "-|>", stroke: 0.7pt + MUTED) }
}))

// ── (H) image-to-image: source + condition -> G -> translated -> D judges ──
#let img2img = align(center, diagram(spacing: (13mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), align(center, text(size: 9pt)[source image \ $+$ condition]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: cream, stroke: 0.9pt + INK)
  node((1.9, 0), align(center, text(size: 9.5pt)[generator]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: ACC.lighten(80%), stroke: 1pt + ACC)
  node((3.6, 0), align(center, text(size: 9pt)[translated \ image]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(88%), stroke: 0.9pt + ACC)
  node((5.4, 0), align(center, text(size: 9.5pt)[discriminator]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: BLUE.lighten(82%), stroke: 1.2pt + BLUE)
  node((5.4, -1.5), align(center, text(size: 9pt)[judges realism \ under condition]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  edge((0, 0), (1.9, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((1.9, 0), (3.6, 0), "-|>", stroke: 0.9pt + ACC)
  edge((3.6, 0), (5.4, 0), "-|>", stroke: 0.9pt + INK)
  edge((5.4, 0), (5.4, -1.5), "-|>", stroke: 0.8pt + GREEN)
}))

#title-slide()

// ═══════════════════════════ PART I — From noise to a fake image ═══════════════════════════
= From a latent noise vector to a fake image

== Recall the generative task #Q

We want to turn simple noise into a realistic sample:
#pause
$ z ~ p(z) quad -->^(G_theta) quad tilde(x) $
#pause
#notebox[We need a *training signal* that says whether the sample $tilde(x)$ looks like real data — but for a freshly drawn $z$ there is *no target image* to compare against.]
#pause
#result[so where does the loss come from?]

== Why not use only pixel reconstruction? #Q

The obvious idea — minimize pixel distance to some target — fails twice:
#pause
- a newly sampled $z$ has *no* paired target image to reconstruct;
#pause
- even with a target, pixel distance rewards the *average* of plausible images, giving *blurry* results (exactly the VAE symptom from L19).
#pause
#align(center, text(size: 16pt, fill: MUTED)[we need a signal for *realism*, not for pixel-wise proximity])

== The two-player setup #V

Turn generation into a game between two networks:
#pause
#advsetup
#pause
#align(center, text(size: 14pt, fill: MUTED)[the generator makes fakes; the discriminator scores real vs fake — neither loss makes sense alone])

== The two roles #D

Two networks with opposing goals:
#pause
#two(
  notebox[*$G_theta$ — the counterfeiter.* Maps simple noise $z$ to samples $tilde(x)$; wants the detective to call them real.],
  notebox[*$D_psi$ — the detective.* Separates real data from generated samples; wants to catch every fake.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[their *competition* supplies the learning signal — a learned notion of realism])

== A discriminator is a classifier #D

The discriminator is an ordinary binary classifier with a probability output:
#pause
$ D_psi(x) in (0, 1) approx P("real" | x) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[label real examples *1*, generated examples *0* — the one twist is that the negative class keeps changing])

== Discriminator objective #D

Push $D$ up on real data and down on fakes:
#pause
$ max_psi thin EE_(x ~ p_"data") log D_psi(x) + EE_(z ~ p(z)) log(1 - D_psi(G_theta(z))) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the first term rewards calling real data real; the second rewards calling fakes fake])

== The generator's original objective #D

Freeze $D$; change the fakes so the detective calls them real:
#pause
$ min_theta thin EE_(z ~ p(z)) log(1 - D_psi(G_theta(z))) $
#pause
#notebox[Only the *second* term of the discriminator objective depends on $theta$. The generator has no access to real data — it only sees the discriminator's verdict on its own samples.]

== Minimax, in words #D

Stack the two objectives into one value function:
#pause
$ min_G max_D thin V(D, G) $
#pause
$ V(D, G) = EE_(x ~ p_"data") [log D(x)] + EE_(z ~ p(z)) [log(1 - D(G(z)))] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the detective *maximizes*, the counterfeiter *minimizes* — the same scalar])

== Board sketch: alternating steps #I

The game plays out as an *alternation*, never a single descent:
#pause
+ fix $D$, nudge the *fakes* toward regions $D$ currently calls real;
#pause
+ fix $G$, move the *boundary* to reject those fakes again.
#pause
#interbox(link-to: IA + "gan-minimax-dance")[
  Step the minimax dance: watch generated samples drift toward the data while the discriminator boundary chases them — both players keep moving.
]

// ═══════════════════════════ PART II — How GAN training works ═══════════════════════════
= How GAN training actually works

== One training iteration #V

Each step updates the two networks in turn:
#pause
#trainloop
#pause
#align(center, text(size: 14pt, fill: MUTED)[update $D$ on a fresh minibatch, then update $G$ on fresh noise — then repeat])

== The gradient path to the generator #V

The generator never touches real data; its signal arrives *through* the discriminator:
#pause
#gradpath
#pause
$ z -> G_theta(z) -> D_psi(G_theta(z)) -> L_G $
#pause
#align(center, text(size: 14pt, fill: MUTED)[gradients pass through the *frozen* $D$ into $G$; $D$'s weights are held fixed, but its computation stays in the graph])

== The saturation problem #D

Early on, the detective easily spots fakes, so $D(G(z)) approx 0$:
#pause
$ log(1 - D(G(z))) quad #text(fill: MUTED)[is very *flat* near $D = 0$] $
#pause
#alertbox[The original $log(1 - D)$ generator objective gives a *tiny gradient* exactly when the generator is worst and needs the *strongest* push. The signal vanishes early.]

== Non-saturating generator loss #D

Keep the same goal — make fakes look real — but flip to a loss with a healthy gradient:
#pause
$ L_G = - EE_(z ~ p(z)) log D_psi(G_theta(z)) $
#pause
#fig("/lecture20/figures/sat_vs_nonsat.svg", w: 66%)

== Worked scalar example #D

Take a confident detective: $D(G(z)) = 0.01$.
#pause
#two(
  notebox[*Saturating* $log(1 - 0.01) = log 0.99 approx -0.01$. \ Gradient magnitude $1 \/ (1 - D) approx 1.01$ — nearly flat.],
  notebox[*Non-saturating* $-log 0.01 approx 4.61$. \ Gradient magnitude $1 \/ D = 100$ — a strong push.],
)
#pause
#result[$approx 100 times$ stronger gradient at $D = 0.01$ — same direction, far more useful signal]

== What equilibrium would mean #D

If the generated and real distributions matched *perfectly*, no detective could beat a coin flip:
#pause
$ p_G = p_"data" quad ==> quad D(x) = 1/2 quad #text(fill: MUTED)[everywhere] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the detective is reduced to guessing — the strongest possible evidence that the fakes match the data])

== Important correction #Q

A common misreading of $D = 1/2$:
#pause
#alertbox[$D(x) = 1/2$ at equilibrium does *not* mean every generated image is average or blurry. It means the *two distributions are indistinguishable* — the detective cannot separate them, not that samples are bland.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[a statement about *distributions*, not about any single image])

== Latent-space interpolation #V

Sample two codes, interpolate, and generate along the path:
#pause
#fig("/lecture20/figures/latent_interp.svg", w: 82%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[smooth semantic change is evidence that $G$ has *organized* the noise space into a usable map])

// ═══════════════════════════ PART III — Why GANs are hard to train ═══════════════════════════
= Why GANs can be hard to train

== The moving-target problem #D

Unlike ordinary supervised learning, the loss surface itself *keeps moving*:
#pause
#two(
  notebox[*Supervised* — the target is *fixed*; you descend a stationary loss.],
  notebox[*Adversarial* — the opponent *learns too*; the objective shifts under you every step.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[game optimization $!=$ minimizing one fixed function])

== Failure mode: discriminator too strong #D

If the detective becomes *perfect*, it separates real from fake with total confidence:
#pause
$ D(x) approx 1 "on real", quad D(G(z)) approx 0 "on fake" $
#pause
#alertbox[A perfect discriminator leaves the generator with *flat, uninformative* gradients — the game stalls because the critic gives no direction.]

== Failure mode: mode collapse #V

The generator finds a few outputs that fool the *current* detective, and repeats them:
#pause
#fig("/lecture20/figures/toy_2d.svg", w: 66%)

== Coverage versus realism #V

Sharpness and coverage are *different* axes of quality:
#pause
#fig("/lecture20/figures/coverage.svg", w: 82%)
#pause
#align(center, text(size: 13pt, fill: MUTED)[broad-but-blurry, sharp-but-partial, and sharp-and-complete are three genuinely distinct behaviours])

== Failure mode: oscillation #V

The two players can *chase* each other around rather than converge:
#pause
#fig("/lecture20/figures/training_curves.svg", w: 62%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[training curves alone can mislead — a lower $D$ loss is not a better generator])

== Practical stabilizers #D

A grab-bag of tricks that make the game better-behaved:
#pause
- *architecture* — convolutional up/down-sampling (DCGAN);
- *balance* — tuned update schedules and regularization;
- *better critics* — distance-like objectives (Wasserstein intuition);
- *monitor samples* — inspect generated grids, not only losses.
#pause
#align(center, text(size: 16pt, fill: MUTED)[none is a silver bullet — GAN training is a craft as much as an algorithm])

== Wasserstein intuition, at a glance #D

A binary detective saturates; a *critic* that scores "how real" gives a smoother signal:
#pause
#notebox[*WGAN* replaces the saturating classifier with a critic and measures an *earth-mover* distance between distributions. Read it as a response to *unstable geometry*, not a fourth objective to memorize.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[we return to the earth-mover picture in Part V])

== Conditional GANs #V

Feed a *condition* $c$ so generation becomes controllable:
#pause
#condgan
#pause
$ z, c quad -->^G quad tilde(x), quad quad D(x, c) $
#pause
#align(center, text(size: 14pt, fill: MUTED)[$c$ can be a class, a caption, a pose, or a segmentation map])

// ═══════════════════════════ PART IV — The generative-model landscape ═══════════════════════════
= GANs in the generative-model landscape

== VAE versus GAN #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left, left),
  table.header([], [*VAE*], [*GAN*]),
  [training signal], [likelihood bound (ELBO)], [learned adversarial critic],
  [inference encoder], [yes], [not required],
  [typical trade-off], [coverage, smoother samples], [sharpness, instability / collapse],
  [density estimate], [explicit approximate model], [implicit sampler],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[a probabilistic bound versus a learned game — two very different sources of gradient])

== Where GANs remain useful #D

GANs shine where structure is *paired* or *conditional* and a fast single pass matters:
#pause
- *image-to-image translation* (day $->$ night, sketch $->$ photo);
- *super-resolution* and restoration;
- *domain adaptation* with an adversarial alignment loss.
#pause
#align(center, text(size: 16pt, fill: MUTED)[strong conditional signal $+$ one-shot sampling $=$ a natural GAN fit])

== Why diffusion displaced many GAN uses #D

Diffusion (next lecture) trades one adversarial pass for *many* stable denoising steps:
#pause
- *coverage* — a stationary regression target avoids mode collapse;
- *stability* — no moving opponent, so training is far more reliable;
- *cost* — the price is *many* sampling steps instead of one.
#pause
#align(center, text(size: 16pt, fill: MUTED)[better coverage and stability, at a sampling-time cost])

== Quick conceptual check #Q

#align(center, text(size: 18pt)[
  "If samples look *sharp* but all resemble the *same* face, which failure mode is this?"
])
#pause
#v(6pt)
#align(center, text(size: 16pt, fill: MUTED)[and — would a discriminator loss *alone* have diagnosed it?])
#pause
#result[mode collapse — and no, sharp fakes can keep $D$ loss looking healthy]

== Closing line of the first pass #D

One sentence to carry forward:
#pause
#result[GANs learn by *fooling a critic*; diffusion learns by *reversing noise*.]
#pause
#align(center, text(size: 15pt, fill: MUTED)[Part V now derives the game's optimum, the losses per minibatch, and the failure modes in detail])

// ═══════════════════════════ PART V — Detailed adversarial-training slides ═══════════════════════════
= Detailed adversarial training

== Binary cross-entropy from labels #D

For real data label $y = 1$ and generated data label $y = 0$:
#pause
$ L_D = - [ thin y log D(x) + (1 - y) log(1 - D(x)) thin ] $
#pause
Concretely, if $D(x_"real") = 0.9$ and $D(G(z)) = 0.2$:
$ L_D = -(log 0.9 + log 0.8) approx -(-0.105 - 0.223) = #text(fill: ACC, weight: 600)[0.329] $
#pause
#align(center, text(size: 15pt, fill: MUTED)[ordinary classification — the twist is that the *negative class* moves as $G$ changes])

== A minibatch discriminator loss #V

Average the BCE over a batch of $B$ real images and $B$ fakes:
#pause
$ L_D = -1/B sum_(i=1)^B log D(x_i) - 1/B sum_(i=1)^B log(1 - D(G(z_i))) $
#pause
#detachdiag
#pause
#align(center, text(size: 13pt, fill: MUTED)[when updating $D$, the fakes are *detached* — no gradient leaks back into $G$])

== A minibatch generator loss #D

Now update $G$ with the non-saturating loss, discriminator frozen:
#pause
$ L_G = -1/B sum_(i=1)^B log D(G(z_i)) $
#pause
#notebox[The discriminator's weights are *frozen*, but its computation *remains in the graph* — that is how $G$ receives a direction for changing its pixels.]

== The generator is not trained on "fake" labels #D

A subtle but crucial point about what $G$ optimizes:
#pause
#codebox(size: 15pt)[
```text
update D:  real       -> 1 ,   generated -> 0
update G:  generated  -> make D output 1
```
]
#pause
#align(center, text(size: 15pt, fill: MUTED)[$G$ does *not* become a classifier — it changes its images so $D$'s learned features regard them as real])

== The discriminator's boundary moves #V

Two real-ish clusters and a fake cluster: after each update the picture shifts.
#pause
#fig("/lecture20/figures/decision_surf.svg", w: 50%)
#pause
#align(center, text(size: 13pt, fill: MUTED)[this moving-boundary picture is worth more than a formal convergence proof on a first pass])

== Ideal equilibrium, carefully stated #D

For a *fixed* generator, the optimal discriminator has a closed form:
#pause
$ D^*(x) = p_"data"(x) / (p_"data"(x) + p_G(x)) $
#pause
#fig("/lecture20/figures/optimal_d.svg", w: 55%)

== Divergence intuition #D

Substituting $D^*$ back turns the game value into a *divergence* between distributions:
#pause
$ C(G) = -log 4 + 2 dot "JSD"(p_"data" thin || thin p_G) $
#pause
#fig("/lecture20/figures/js_value.svg", w: 52%)

== The classroom message #Q

Strip away the algebra and one idea remains:
#pause
#result[GAN training tries to *match distributions* — not to pair each fake with one real target.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[the minimum $-log 4 approx -1.386$ is reached exactly when $p_G = p_"data"$, where JSD $= 0$])

== Why early generator gradients can fail #D

When the detective assigns $D(G(z)) approx 0$ everywhere:
#pause
- the original minimax generator loss *flattens* — a vanishing gradient;
#pause
- the non-saturating loss changes the *gradient behaviour* while keeping the *same desired direction*.
#pause
#align(center, text(size: 16pt, fill: MUTED)[same target, different curvature — the fix is about gradient magnitude, not about goals])

== Alternating optimization pseudocode #D

The whole training loop in five lines:
#pause
#codebox(size: 14.5pt)[
```text
repeat:
    x <- real minibatch ;  z <- random noise
    update discriminator using x and G(z).detach()
    z <- fresh random noise
    update generator using  -log D(G(z))
```
]
#pause
#align(center, text(size: 15pt, fill: MUTED)[the two updates use *different* colours of parameter — $D$ frozen for one, $G$ frozen for the other])

== Update ratio is a design decision #D

One $D$ step per $G$ step is *common*, not a law:
#pause
- some critic objectives take *several* discriminator updates per generator update, to keep the signal informative;
#pause
- others balance learning rates or add regularization instead.
#pause
#align(center, text(size: 16pt, fill: MUTED)[a first example of why *game* optimization is more delicate than minimizing one fixed loss])

== Conditioning enters both players #D

Supply the condition $c$ to the generator *and* the discriminator:
#pause
$ G(z, c) -> tilde(x), quad quad D(x, c) -> "real / fake" $
#pause
- for *digit generation*, $c$ is a class label;
- for *image translation*, $c$ is an input image, pose, or segmentation map.
#pause
#align(center, text(size: 15pt, fill: MUTED)[$D$ must now judge realism *and* agreement with $c$ — a fake-but-mismatched sample is still rejected])

== Conditional generation versus selection #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 8pt), align: (left, left),
  table.header([*Mechanism*], [*What it does*]),
  [latent $z$], [chooses *one* diverse sample],
  [condition $c$], [specifies the *desired content*],
  [discriminator / critic], [judges *realism* under $c$],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[many different $z$ values can produce different *valid* samples for the *same* $c$])

== Architecture: DCGAN principles #D

The practical pattern that first made GANs train reliably on images:
#pause
- *convolutional* up / down-sampling (no large dense hidden layers);
- careful *normalization* choices;
- *progressively* build spatial resolution.
#pause
#align(center, text(size: 16pt, fill: MUTED)[treat DCGAN as an architectural *stabilization milestone*, not a paper to memorize])

== Generator geometry #V

The generator maps a *simple* connected latent onto a *complicated* data manifold:
#pause
#gengeo
#pause
#align(center, text(size: 14pt, fill: MUTED)[interpolation between two codes traces a *path* along this learned map])

== Mode collapse, expanded #V

Send several $z$ values through $G$ and look at the outputs:
#pause
#two(
  align(center, mc_good),
  align(center, mc_bad),
)
#pause
#align(center, text(size: 14pt, fill: MUTED)[left: a *healthy* generator fans out to varied outputs — right: a *collapsed* one maps everything to a single convincing sample])

== Mode dropping is subtler #Q

Collapse is the obvious failure; *dropping* is the quiet one:
#pause
#alertbox[A generator may cover *several* modes yet *omit rare* regions. The sample grid looks impressive while quietly failing fairness, safety, or scientific-use requirements.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[quality and coverage must be evaluated *separately*])

== Loss curves are not enough #D

GAN losses oscillate and change scale as the opponent changes. Always also inspect:
#pause
- *fixed-noise sample grids* over the course of training;
- *diversity* of the generated set;
- *task-specific* downstream measures.
#pause
#align(center, text(size: 16pt, fill: MUTED)[a lower discriminator loss does *not* monotonically mean better samples])

== Precision and recall for generation #D

Split "quality" into two measurable axes:
#pause
#two(
  notebox[*Precision* — do generated samples lie *near* the data manifold? (fidelity)],
  notebox[*Recall* — does the generator *cover* the manifold? (diversity)],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[sharp, collapsed outputs can score *high precision* and *poor recall*])

== FID, one careful slide #D

Fréchet-style metrics compare *feature statistics* of real and generated images:
#pause
$ "one scalar metric" quad != quad "complete judgment of quality, diversity, bias, memorization" $
#pause
#alertbox[FID is a *useful summary*, not a verdict. A single number hides mode dropping, bias, and near-copying — inspect samples too.]

== Wasserstein intuition #V

If two distributions have *disjoint support*, a binary detective becomes overconfident. A distance-like critic measures *how far mass must move*:
#pause
#fig("/lecture20/figures/earth_mover.svg", w: 58%)

== The Lipschitz constraint, at a glance #D

Wasserstein critics need *controlled sensitivity* — a bounded slope:
#pause
$ abs(f(x) - f(y)) <= K norm(x - y) $
#pause
#notebox[*Weight clipping* and *gradient penalties* are two ways to enforce this approximate 1-Lipschitz property. We name them, and stop short of deriving WGAN-GP.]

== Style-based generation, one placement slide #D

Where the latent is *injected* changes what it controls:
#pause
- separate latent processing can steer *coarse-to-fine* attributes at *different layers*;
- pose and layout at coarse scales, colour and texture at fine scales.
#pause
#align(center, text(size: 16pt, fill: MUTED)[a clean example of *architecture shaping the meaning* of a latent — not a required implementation])

== Image-to-image GANs #V

Condition on a *source image* and translate it:
#pause
#img2img
#pause
#align(center, text(size: 14pt, fill: MUTED)[super-resolution · day-to-night · domain adaptation — contrast with diffusion-based editing later])

== Limited data and memorization #Q

With small or narrow datasets, a generator can *memorize* or nearly copy examples:
#pause
#alertbox[A visually persuasive sample grid does *not* by itself establish generalization. Run *nearest-neighbour* checks against the training set.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[memorization is a *data-governance* issue, not only a quality one])

== GANs and representations #D

The discriminator's intermediate features often become *useful representations*:
#pause
- separating real from fake requires *semantic* and *texture-sensitive* features;
- those features transfer to downstream tasks — echoing L17.
#pause
#align(center, text(size: 16pt, fill: MUTED)[many generative objectives *incidentally* learn representations — but their evaluation targets differ])

== VAE, GAN, diffusion: first comparison #V

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 7pt), align: (left, left, left, left),
  table.header([*Question*], [*VAE*], [*GAN*], [*Diffusion*]),
  [what trains it?], [likelihood bound], [adversarial signal], [known-noise regression],
  [what is hard?], [latent / recon trade-off], [game stability], [iterative sampling],
  [what we emphasize], [probabilistic latent], [realism and coverage], [denoising dynamics],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[three routes to the same goal — draw samples that look like the data])

== Classroom diagnosis #Q

Match each *symptom* to a likely issue and one intervention category:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*Symptom*], [*Likely issue*]),
  [fake samples repeat], [mode collapse — diversify / rebalance],
  [$D$ accuracy near perfect], [$D$ too strong — weaken / critic loss],
  [samples drift around training], [oscillation — schedule / regularize],
  [diverse but implausible], [under-trained $G$ — more capacity / steps],
))
#pause
#align(center, text(size: 14pt, fill: MUTED)[no symptom has a *unique* fix — name a category, not a magic cure])

== Common misconceptions #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left),
  table.header([*Misconception*], [*Correction*]),
  [$D$ is used at generation time], [usually only $G(z)$ is needed after training],
  [$D = 1/2$ means a bad classifier], [it is *ideal* only when distributions match],
  [sharp samples imply good generation], [mode collapse can be sharp and incomplete],
  [GANs maximize likelihood], [standard GANs are *implicit* adversarial models],
))

== Retrieval questions #Q

Test yourself before the next lecture:
#pause
+ Why does the generator need gradients *through* a frozen discriminator?
#pause
+ What exactly changes in the *non-saturating* loss?
#pause
+ How does *mode collapse* differ from poor visual fidelity?
#pause
+ Why do *game* losses resist a simple interpretation?

== Student takeaways #D

#pause
+ A GAN is a *two-player optimization problem*, not a generator plus a conventional label.
#pause
+ The discriminator supplies a *learned realism signal*, and its gradients train the generator.
#pause
+ Sharp samples do *not* guarantee coverage — *mode collapse* is the central warning.
#pause
+ Diffusion solves a *different* training problem: denoising known corruptions, not fooling a moving critic.

#focus-slide[
  A GAN learns a *critic* to tell fake from real; diffusion learns to *reverse a known noise process*.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Diffusion Models* — a simple corruption process turns generation into *supervised denoising*.
]
