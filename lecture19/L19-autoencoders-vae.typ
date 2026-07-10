// Autoencoders and Variational Autoencoders
// Lecture 19 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture19/L19-autoencoders-vae.typ /tmp/L19.pdf
//   typst compile --root . --input handout=true lecture19/L19-autoencoders-vae.typ /tmp/L19h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.
// Schematic figures: python3 lecture19/diagrams/l19_figs.py
// Worked-number checks: python3 lecture19/diagrams/verify_numbers.py

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [Autoencoders and Variational Autoencoders],
  subtitle: [From compressing inputs to sampling a distribution],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let cream = rgb("#EFEEEB")
#let rect = fletcher.shapes.rect

// ═══════════════════════════ fletcher diagrams (centerpieces) ═══════════════════════════

// ── (A) plain autoencoder: x -> encoder -> bottleneck z -> decoder -> x-hat ──
#let aediag = align(center, diagram(spacing: (13mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), text(size: 10pt)[input $x$], radius: 6mm, fill: cream)
  node((1.5, 0), align(center, text(size: 9.5pt)[encoder \ $f_phi$]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((2.75, 0), $z$, radius: 5mm, fill: ACC.lighten(70%), stroke: 1.1pt + ACC)
  node((4.0, 0), align(center, text(size: 9.5pt)[decoder \ $g_theta$]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((5.5, 0), text(size: 10pt)[$hat(x)$], radius: 6mm, fill: cream)
  edge((0, 0), (1.5, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((1.5, 0), (2.75, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((2.75, 0), (4.0, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((4.0, 0), (5.5, 0), "-|>", stroke: 0.8pt + MUTED)
  node((2.75, -1.15), text(size: 9.5pt, fill: ACC)[code $z in RR^d$ (bottleneck)], stroke: none, fill: none)
  edge((5.5, 0.55), (0, 0.55), "..|>", bend: -32deg, stroke: 0.8pt + GREEN, label: text(size: 9pt, fill: GREEN)[minimize $norm(x - hat(x))^2$], label-side: left)
}))

// ── (B) denoising autoencoder: corrupt the input, reconstruct the CLEAN target ──
#let denoise = align(center, diagram(spacing: (11mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), text(size: 10pt)[clean $x$], radius: 6mm, fill: cream)
  node((1.3, 0), align(center, text(size: 9pt)[add \ noise]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: RED.lighten(82%), stroke: 0.9pt + RED)
  node((2.5, 0), text(size: 10pt)[$tilde(x)$], radius: 5.5mm, fill: white)
  node((3.75, 0), align(center, text(size: 9pt)[encoder]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((4.9, 0), $z$, radius: 5mm, fill: ACC.lighten(70%), stroke: 1.1pt + ACC)
  node((6.05, 0), align(center, text(size: 9pt)[decoder]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((7.3, 0), text(size: 10pt)[$hat(x)$], radius: 6mm, fill: cream)
  for (a, b) in ((0, 1.3), (1.3, 2.5), (2.5, 3.75), (3.75, 4.9), (4.9, 6.05), (6.05, 7.3)) {
    edge((a, 0), (b, 0), "-|>", stroke: 0.8pt + MUTED)
  }
  edge((0, 0.55), (7.3, 0.55), "..|>", bend: 30deg, stroke: 0.9pt + GREEN, label: text(size: 9pt, fill: GREEN)[target $=$ the *clean* $x$, not $tilde(x)$], label-pos: 0.5)
}))

// ── (C) VAE forward: encoder -> (mu, log sigma^2) -> sample z -> decoder -> p(x|z) ──
#let vaearch = align(center, diagram(spacing: (12mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), text(size: 10pt)[$x$], radius: 5.5mm, fill: cream)
  node((1.3, 0), align(center, text(size: 9pt)[encoder \ $q_phi$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((2.75, 0.75), text(size: 10.5pt)[$mu(x)$], shape: rect, corner-radius: 4pt, inset: 5pt, fill: ACC.lighten(80%), stroke: 0.9pt + ACC)
  node((2.75, -0.75), text(size: 10.5pt)[$log sigma^2(x)$], shape: rect, corner-radius: 4pt, inset: 5pt, fill: ACC.lighten(80%), stroke: 0.9pt + ACC)
  node((4.35, 0), align(center, text(size: 10pt)[$z = mu + sigma dot.o epsilon$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(68%), stroke: 1.2pt + ACC)
  node((4.35, -1.55), align(center, text(size: 9pt)[$epsilon tilde cal(N)(0, I)$]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: GREEN.lighten(80%), stroke: 0.9pt + GREEN)
  node((6.0, 0), align(center, text(size: 9pt)[decoder \ $p_theta (x | z)$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((7.45, 0), align(center, text(size: 9pt)[dist. \ over $x$]), radius: 6.5mm, fill: cream)
  edge((0, 0), (1.3, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((1.3, 0), (2.75, 0.75), "-|>", stroke: 0.8pt + TEAL)
  edge((1.3, 0), (2.75, -0.75), "-|>", stroke: 0.8pt + TEAL)
  edge((2.75, 0.75), (4.35, 0), "-|>", stroke: 0.8pt + ACC)
  edge((2.75, -0.75), (4.35, 0), "-|>", stroke: 0.8pt + ACC)
  edge((4.35, -1.55), (4.35, 0), "-|>", stroke: 0.9pt + GREEN)
  edge((4.35, 0), (6.0, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((6.0, 0), (7.45, 0), "-|>", stroke: 0.8pt + MUTED)
}))

// ── (D) reparameterization: deterministic path carries the gradient; noise is external ──
#let reparam = align(center, diagram(spacing: (16mm, 10mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), align(center, text(size: 9.5pt)[$mu, sigma$ \ (network)]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((1.6, 0), text(size: 10.5pt)[$z = mu + sigma dot.o epsilon$], shape: rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(68%), stroke: 1.2pt + ACC)
  node((3.1, 0), text(size: 10.5pt)[loss $cal(L)$], shape: rect, corner-radius: 4pt, inset: 6pt, fill: INK, stroke: 1pt + INK)
  node((3.1, 0), text(size: 10.5pt, fill: white)[loss $cal(L)$], shape: rect, corner-radius: 4pt, inset: 6pt, fill: INK, stroke: 1pt + INK)
  node((1.6, -1.5), align(center, text(size: 9pt)[$epsilon tilde cal(N)(0, I)$]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: GREEN.lighten(80%), stroke: 0.9pt + GREEN)
  edge((0, 0), (1.6, 0), "-|>", stroke: 1.2pt + INK)
  edge((1.6, 0), (3.1, 0), "-|>", stroke: 1.2pt + INK)
  edge((1.6, -1.5), (1.6, 0), "-|>", stroke: 0.9pt + GREEN, label: text(size: 8.5pt, fill: GREEN)[external · no grad], label-side: right)
  node((1.55, 1.0), text(size: 9pt, fill: RED)[$nabla$ flows back along this *deterministic* path], stroke: none, fill: none)
}))

// ── (E) VAE graphical model: z -> x, observed x shaded, plate over N examples ──
#let pgm = align(center, diagram(spacing: (10mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node(enclose: ((0, 1.0), (0, -0.4)), inset: 15pt, stroke: 0.9pt + MUTED, fill: none, corner-radius: 5pt, snap: -2)
  node((0, 1.0), $z$, radius: 6.5mm, fill: white, stroke: 1pt + INK)
  node((0, -0.4), $x$, radius: 6.5mm, fill: MUTED.lighten(45%), stroke: 1pt + INK)
  edge((0, 1.0), (0, -0.4), "-|>", stroke: 1pt + INK, label: text(size: 9.5pt)[$p_theta (x | z)$], label-side: right)
  node((-1.6, 1.0), text(size: 9.5pt, fill: ACC)[$z tilde cal(N)(0, I)$], stroke: none, fill: none)
  node((-1.6, -0.4), text(size: 9pt, fill: MUTED)[observed], stroke: none, fill: none)
  node((0.95, -0.95), text(size: 10pt, fill: MUTED)[$N$], stroke: none, fill: none)
}))

// ── (F) ELBO decomposition: log p(x) = ELBO + KL-gap ; ELBO = recon - KL(q||prior) ──
#let elbodecomp = align(center, diagram(spacing: (13mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), align(center, text(size: 10pt)[$log p_theta (x)$ \ (evidence)]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: cream, stroke: 1pt + INK)
  node((2.15, 0.95), align(center, text(size: 9.5pt)[$"ELBO"(x)$]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: GREEN.lighten(80%), stroke: 1pt + GREEN)
  node((2.15, -0.95), align(center, text(size: 8.5pt)[$D_"KL" (q parallel p_theta (z|x))$ \ $>= 0$  (the gap)]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: MUTED.lighten(55%), stroke: 0.9pt + MUTED)
  edge((0, 0), (2.15, 0.95), "-|>", stroke: 0.9pt + GREEN)
  edge((0, 0), (2.15, -0.95), "-|>", stroke: 0.9pt + MUTED)
  node((4.5, 1.55), align(center, text(size: 8.5pt)[$EE_q[log p_theta (x|z)]$ \ reconstruct]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((4.5, 0.35), align(center, text(size: 8.5pt)[$-  D_"KL" (q parallel p(z))$ \ organize latent]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(80%), stroke: 0.9pt + ACC)
  edge((2.15, 0.95), (4.5, 1.55), "-|>", stroke: 0.8pt + TEAL)
  edge((2.15, 0.95), (4.5, 0.35), "-|>", stroke: 0.8pt + ACC)
}))

// ── (G) training loop: forward chain + backprop return arrow ──
#let trainloop = align(center, diagram(spacing: (10mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(x, lbl, c, f) = node((x, 0), align(center, text(size: 8.8pt, lbl)), shape: rect, corner-radius: 4pt, inset: 5.5pt, stroke: 0.9pt + c, fill: f)
  node((0, 0), text(size: 9.5pt)[$x$], radius: 5mm, fill: cream)
  bx(1.15, [$mu, log sigma^2$], TEAL, TEAL.lighten(82%))
  bx(2.5, [$z = mu + sigma dot.o epsilon$], ACC, ACC.lighten(70%))
  bx(3.85, [decoder], BLUE, BLUE.lighten(85%))
  bx(5.25, [recon loss \ $+$ KL loss], INK, cream)
  for (a, b) in ((0, 1.15), (1.15, 2.5), (2.5, 3.85), (3.85, 5.25)) { edge((a, 0), (b, 0), "-|>", stroke: 0.8pt + MUTED) }
  node((2.5, -1.35), align(center, text(size: 9pt)[$epsilon tilde cal(N)(0, I)$]), stroke: none, fill: none)
  edge((2.5, -1.05), (2.5, -0.35), "-|>", stroke: 0.8pt + GREEN)
  edge((5.25, 0.75), (1.15, 0.75), "..|>", bend: -42deg, stroke: 1.1pt + RED, label: text(size: 9pt, fill: RED)[backprop], label-pos: 0.5)
}))

// ── (H) generation loop: sample the prior, decode once ──
#let genloop = align(center, diagram(spacing: (16mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), align(center, text(size: 9.5pt)[$z tilde cal(N)(0, I)$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: GREEN.lighten(80%), stroke: 0.9pt + GREEN)
  node((1.5, 0), align(center, text(size: 9pt)[trained \ decoder $g_theta$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((3.0, 0), align(center, text(size: 9.5pt)[new $x$]), radius: 6.5mm, fill: cream)
  edge((0, 0), (1.5, 0), "-|>", stroke: 0.9pt + INK)
  edge((1.5, 0), (3.0, 0), "-|>", stroke: 0.9pt + INK)
  node((1.5, -1.1), text(size: 9pt, fill: MUTED)[no encoder, no data at generation time], stroke: none, fill: none)
}))

// ── (I) conditional VAE: a condition c feeds BOTH encoder and decoder ──
#let cvae = align(center, diagram(spacing: (12mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), text(size: 10pt)[$x$], radius: 5.5mm, fill: cream)
  node((1.4, 0), align(center, text(size: 8.8pt)[encoder \ $q_phi (z | x, c)$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((2.95, 0), $z$, radius: 5mm, fill: ACC.lighten(70%), stroke: 1.1pt + ACC)
  node((4.5, 0), align(center, text(size: 8.8pt)[decoder \ $p_theta (x | z, c)$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((6.0, 0), text(size: 10pt)[$hat(x)$], radius: 5.5mm, fill: cream)
  for (a, b) in ((0, 1.4), (1.4, 2.95), (2.95, 4.5), (4.5, 6.0)) { edge((a, 0), (b, 0), "-|>", stroke: 0.8pt + MUTED) }
  node((2.95, -1.5), align(center, text(size: 9pt)[condition $c$ \ (label / text)]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: GREEN.lighten(80%), stroke: 0.9pt + GREEN)
  edge((2.95, -1.5), (1.4, -0.35), "-|>", stroke: 0.8pt + GREEN)
  edge((2.95, -1.5), (4.5, -0.35), "-|>", stroke: 0.8pt + GREEN)
}))

// ── (J) latent diffusion: AE compresses to latent, diffusion runs there, decode once ──
#let ldm = align(center, diagram(spacing: (10mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), align(center, text(size: 9pt)[image \ (pixels)]), radius: 6.5mm, fill: cream)
  node((1.35, 0), align(center, text(size: 8.8pt)[AE \ encoder]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((2.6, 0), $z$, radius: 5mm, fill: ACC.lighten(70%), stroke: 1.1pt + ACC)
  node((4.1, 0), align(center, text(size: 8.8pt)[diffusion \ (in latent)]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: BLUE.lighten(82%), stroke: 1.1pt + BLUE)
  node((5.6, 0), $z_0$, radius: 5mm, fill: ACC.lighten(70%), stroke: 1.1pt + ACC)
  node((6.85, 0), align(center, text(size: 8.8pt)[AE \ decoder]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((8.2, 0), align(center, text(size: 9pt)[image]), radius: 6.5mm, fill: cream)
  for (a, b) in ((0, 1.35), (1.35, 2.6), (2.6, 4.1), (4.1, 5.6), (5.6, 6.85), (6.85, 8.2)) { edge((a, 0), (b, 0), "-|>", stroke: 0.8pt + MUTED) }
  node((2.6, -1.15), text(size: 9pt, fill: TEAL)[compress once], stroke: none, fill: none)
  node((5.6, -1.15), text(size: 9pt, fill: TEAL)[decode once], stroke: none, fill: none)
}))

// ── (K) reparameterization computation graph (fuller, log-variance version) ──
#let reparamgraph = align(center, diagram(spacing: (10mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), text(size: 9.5pt)[$x$], radius: 5mm, fill: cream)
  node((1.25, 0), align(center, text(size: 8.8pt)[encoder]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((2.75, 0.7), text(size: 10pt)[$mu$], shape: rect, corner-radius: 4pt, inset: 5pt, fill: ACC.lighten(80%), stroke: 0.9pt + ACC)
  node((2.75, -0.7), text(size: 9.5pt)[$log sigma^2$], shape: rect, corner-radius: 4pt, inset: 5pt, fill: ACC.lighten(80%), stroke: 0.9pt + ACC)
  node((4.3, 0), text(size: 10pt)[$z = mu + sigma dot.o epsilon$], shape: rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(68%), stroke: 1.2pt + ACC)
  node((4.3, -1.65), align(center, text(size: 8.5pt)[$epsilon tilde cal(N)(0, I)$]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: GREEN.lighten(80%), stroke: 0.9pt + GREEN)
  node((5.85, 0), align(center, text(size: 8.8pt)[decoder]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((7.15, 0), align(center, text(size: 8.5pt)[$p_theta (x | z)$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: cream, stroke: 0.9pt + INK)
  node((2.75, -1.5), text(size: 8pt, fill: MUTED)[$sigma = exp(1/2 log sigma^2)$], stroke: none, fill: none)
  edge((0, 0), (1.25, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((1.25, 0), (2.75, 0.7), "-|>", stroke: 0.8pt + TEAL)
  edge((1.25, 0), (2.75, -0.7), "-|>", stroke: 0.8pt + TEAL)
  edge((2.75, 0.7), (4.3, 0), "-|>", stroke: 0.8pt + ACC)
  edge((2.75, -0.7), (4.3, 0), "-|>", stroke: 0.8pt + ACC)
  edge((4.3, -1.65), (4.3, 0), "-|>", stroke: 0.9pt + GREEN)
  edge((4.3, 0), (5.85, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((5.85, 0), (7.15, 0), "-|>", stroke: 0.8pt + MUTED)
}))

#title-slide()

// ═══════════════════════════ PART I — Why reconstruction is not generation ═══════════════════════════
= Why reconstruction is not generation

== The generative-modeling question #Q

A classifier only learns the conditional $p(y | x)$. A *generative* model is more ambitious:
#pause
#result[learn $p_theta (x)$ well enough to *generate* plausible new $x$]
#pause
#two(
  notebox[*Discriminative* — $p(y | x)$: given an image, name it. Never has to imagine an image.],
  notebox[*Generative* — $p_theta (x)$: what does a *plausible image even look like*? Must model the data itself.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[today: build a generative model by *learning a latent space we can sample from*])

== The autoencoder #V

Compress the input to a code $z$, then reconstruct it:
#pause
$ x thick arrow.r.long^(f_phi) thick z thick arrow.r.long^(g_theta) thick hat(x) $
#pause
#aediag
#pause
#align(center, text(size: 14pt, fill: MUTED)[*encoder* $f_phi$ maps $x -> z$; *decoder* $g_theta$ maps $z -> hat(x)$; train to make $hat(x) approx x$])

== The bottleneck intuition #D

Why does a narrow code $z$ force *learning* rather than *copying*?
#pause
#notebox[If $z$ is constrained (few dimensions), the network cannot store every input coordinate. To reconstruct well it must keep the *salient structure* and discard the rest.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[a wide-open code learns the *identity* map and learns nothing useful])

== The reconstruction objective #D

Train encoder and decoder together to minimize squared reconstruction error:
#pause
$ cal(L)_"AE" = sum_i norm(x_i - g_theta (f_phi (x_i)))_2^2 $
#pause
This is not an arbitrary choice. A *Gaussian-output* decoder gives
$ p_theta (x | z) = cal(N)(g_theta (z), sigma_x^2 I) quad arrow.r.double quad -log p_theta (x | z) prop norm(x - g_theta (z))^2 $
#pause
#align(center, text(size: 15pt, fill: MUTED)[squared error $=$ negative Gaussian log-likelihood — a first taste of the probabilistic view])

== What an autoencoder does well #D

Even without a probabilistic story, a trained autoencoder is useful:
#pause
- *denoising* — corrupt the input, reconstruct the clean signal;
- *compression* — a compact code $z$ for storage or transmission;
- *visualization* — a 2-D or 3-D code to plot the data;
- *features* — reuse $z$ (or the encoder) for a downstream task.
#pause
#align(center, text(size: 16pt, fill: MUTED)[all of these use $z$ as a *representation* — none of them require *sampling* new $x$])

== Denoising autoencoder #V

Force the code to capture structure by making the task harder — corrupt, then restore:
#pause
#denoise
#pause
#align(center, text(size: 14pt, fill: MUTED)[the target is the *clean* $x$; the encoder must learn what survives noise, not memorize pixels])

== The sampling failure #D

So can we just feed a random $z$ to the decoder and get a new image?
#pause
#alertbox[*No.* An autoencoder never constrains *where* codes live. Encoded training points occupy *disconnected, irregular* regions. A random $z$ lands *between* them and decodes to nonsense.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[reconstruction says nothing about the *empty* parts of latent space])

== Visual: latent-space islands #V

Encode the data — the codes form islands with wide empty gaps:
#pause
#fig("/lecture19/figures/latent_islands.svg", w: 33%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[samples from $cal(N)(0, I)$ mostly land in the *gap between islands* $->$ the decoder was never trained there])

== The design requirement #Q

For a latent space to be *generative*, sampling the prior must produce plausible data:
#pause
#result[$z tilde p(z) quad arrow.r.double quad g_theta (z)$ should be plausible]
#pause
#notebox[We need two things at once: (1) a *known* distribution $p(z)$ we can sample, and (2) a decoder trained so that *every* likely $z$ maps to something realistic. The VAE engineers exactly this.]

== Interactive: the latent space #I

#interbox(link-to: IA + "vae-latent-explorer")[
  Move a point around a 2-D latent space and watch the decoded output. See the islands of a plain autoencoder, and how prior samples fall into the gaps between them.
]

// ═══════════════════════════ PART II — The VAE probabilistic model ═══════════════════════════
= The VAE probabilistic model

== The generative story #D

A VAE first *defines* how data is generated from a simple latent cause:
#pause
$ z tilde p(z) = cal(N)(0, I), quad quad x tilde p_theta (x | z) $
#pause
+ sample a simple latent $z$ from a fixed *standard Gaussian* prior;
+ *render* an observation $x$ with a neural decoder $p_theta (x | z)$.
#pause
#align(center, text(size: 16pt, fill: MUTED)[the prior is *fixed and sampleable by construction* — that is what fixes the islands problem])

== A decoder is a conditional distribution #D

The decoder does not output "an image" — it outputs the *parameters of a distribution* over $x$:
#pause
$ p_theta (x | z) = cal(N)(g_theta (z), sigma_x^2 I) quad quad #text(size: 15pt, fill: MUTED)[(Gaussian example)] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the familiar "reconstruction image" is just the conditional *mean* $g_theta (z)$])

== Why inference is hard #D

To train, we would like the *posterior* over the latent cause of each $x$:
#pause
$ p_theta (z | x) = (p_theta (x | z) thin p(z)) / p_theta (x) $
#pause
$ p_theta (x) = integral p_theta (x | z) thin p(z) thin d z quad quad #text(size: 15pt, fill: RED)[(intractable integral)] $
#pause
#alertbox[The evidence $p_theta (x)$ integrates over *all* $z$ through a nonlinear decoder. No closed form — so the posterior is out of reach too.]

== Introduce an approximate posterior #D

Sidestep the intractable posterior: let the *encoder* predict a tractable Gaussian approximation.
#pause
$ q_phi (z | x) = cal(N)(mu_phi (x), thin "diag"(sigma_phi (x)^2)) $
#pause
#notebox[The encoder now outputs a *distribution*, not a single point: a mean $mu_phi (x)$ and a per-dimension variance $sigma_phi (x)^2$. Each input maps to a small cloud in latent space.]

== The VAE, drawn #V

The whole forward model — encoder predicts a distribution, we sample, the decoder renders:
#pause
#vaearch
#pause
#align(center, text(size: 14pt, fill: MUTED)[$x ->$ encoder $-> (mu, log sigma^2) ->$ sample $z ->$ decoder $->$ distribution over $x$])

== The learning target #D

We cannot maximize $log p_theta (x)$ directly — but we can maximize a *lower bound* on it:
#pause
$ log p_theta (x) >= underbrace(EE_(q_phi (z | x))[log p_theta (x | z)], "reconstruct") - underbrace(D_"KL" (q_phi (z | x) parallel p(z)), "organize latent space") $
#pause
#align(center, text(size: 16pt, fill: MUTED)[two terms, two jobs — this is the object we actually optimize])

== Name it: the ELBO #D

This bound is the *Evidence Lower BOund*:
#pause
#result[maximize a *tractable lower bound* on the data log-likelihood]
#pause
#notebox[Understand what each term *does* before any algebra: the first term *explains the data*, the second *keeps latent codes near the prior* so we can sample them later.]

== The reconstruction term #D

$ EE_(q_phi (z | x))[log p_theta (x | z)] $
#pause
Read it as a recipe:
#pause
+ take input $x$, encode it, and *sample* a plausible latent explanation $z tilde q_phi (z | x)$;
+ require the decoder to *reconstruct $x$* from that $z$.
#pause
#align(center, text(size: 16pt, fill: MUTED)[high when the sampled latent actually explains the observed input])

== The KL term #D

$ D_"KL" (q_phi (z | x) parallel p(z)) $
#pause
Read it as a constraint:
#pause
#notebox[each example's encoded cloud $q_phi (z | x)$ should stay *close to the shared prior* $cal(N)(0, I)$. This is what pulls every code toward one common, sampleable region.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[without it, the encoder would scatter codes into islands again])

== The Gaussian KL — show, do not derive #D

For a diagonal-Gaussian posterior against the standard-normal prior, the KL is *closed form*:
#pause
$ D_"KL" (cal(N)(mu, sigma^2) parallel cal(N)(0, 1)) = 1/2 sum_j (mu_j^2 + sigma_j^2 - log sigma_j^2 - 1) $
#pause
#fig("/lecture19/figures/gaussian_kl.svg", w: 34%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[minimized at $mu = 0, sigma = 1$ (KL $= 0$) — the KL pushes each code toward the prior])

== The tension #D

The two terms *pull in opposite directions* — the objective finds the balance:
#pause
#notebox[*Reconstruction* wants every input to keep its *own detailed* code; *KL* wants *all* codes to collapse to *one* Gaussian cloud at the origin.]
#pause
#fig("/lecture19/figures/kl_recon.svg", w: 41%)

// ═══════════════════════════ PART III — Backpropagation through sampling ═══════════════════════════
= Backpropagation through sampling

== The apparent obstacle #D

The ELBO needs a *sample* $z tilde cal(N)(mu, sigma^2)$ inside the forward pass. But:
#pause
#alertbox[*Sampling is not differentiable.* A raw draw $z tilde cal(N)(mu, sigma^2)$ seems to block gradients from flowing back into $mu$ and $sigma$ — the encoder cannot learn.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[we need the randomness *without* breaking the chain rule])

== The reparameterization trick #D

Move the randomness to an *external* variable, then transform it deterministically:
#pause
$ epsilon tilde cal(N)(0, I), quad quad z = mu + sigma dot.o epsilon $
#pause
#reparam
#pause
#align(center, text(size: 14pt, fill: MUTED)[$epsilon$ is sampled *outside* the network; the map $mu, sigma -> z$ is a smooth, differentiable function])

== Board calculation #D

Take a 2-D latent with $mu = [1, thin -1]$, $sigma = [0.5, thin 0.2]$, and a draw $epsilon = [-0.4, thin 1.5]$:
#pause
$ z = mu + sigma dot.o epsilon = mat(1 + 0.5 dot (-0.4); -1 + 0.2 dot 1.5) = #text(fill: ACC, weight: 600)[$vec(0.8, -0.7)$] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[grow $sigma$ $->$ the same $epsilon$ moves $z$ *further* from $mu$: more spread, a larger KL])

== The training loop #V

Every step: encode, reparameterize, decode, score both terms, and backprop.
#pause
#trainloop
#pause
#align(center, text(size: 14pt, fill: MUTED)[data *enters* the loop; gradients flow through $z$ into the encoder *and* the decoder])

== The generation loop #V

Generation throws away the encoder — just sample the prior and decode:
#pause
#genloop
#pause
#align(center, text(size: 15pt, fill: MUTED)[*training* begins with data; *generation* begins with $z tilde cal(N)(0, I)$ — nothing else])

== Interpolation as a diagnostic #V

Walk a straight line between two codes and decode along the path:
#pause
#fig("/lecture19/figures/interpolation.svg", w: 62%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[a good latent space yields *gradual semantic change*; a bad one crosses gaps into invalid images])

== Conditional VAE #D

Feed a *condition* $c$ (a class label, a text embedding) into both networks:
#pause
$ q_phi (z | x, c), quad quad p_theta (x | z, c) $
#pause
#cvae
#pause
#align(center, text(size: 14pt, fill: MUTED)[at generation time: *choose* $c$, sample $z tilde p(z)$, decode — $c$ steers *what* is generated])

== The $beta$-VAE trade-off #D

Put a *weight* $beta$ on the KL term and dial the balance explicitly:
#pause
$ cal(L) = -EE_q[log p_theta (x | z)] + beta thin D_"KL" (q_phi (z | x) parallel p(z)) $
#pause
#fig("/lecture19/figures/beta_sweep.svg", w: 46%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[higher $beta$ $->$ more organized latent factors, but *softer / worse* reconstructions])

// ═══════════════════════════ PART IV — Honest limitations and the bridge ═══════════════════════════
= Honest limitations and the bridge onward

== Blurry samples: why? #D

A recurring complaint: VAE samples look *soft*. The cause is the per-pixel likelihood.
#pause
#fig("/lecture19/figures/blurry.svg", w: 74%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[several sharp outputs are all plausible; a per-pixel Gaussian rewards their *average*, which is blurry])

== Posterior collapse #D

The opposite failure — the model *ignores* the latent entirely:
#pause
#fig("/lecture19/figures/posterior_collapse.svg", w: 62%)
#pause
#alertbox[A very powerful decoder can predict $x$ from context alone, so $q_phi (z | x) -> p(z)$ for every $x$. The latent becomes *unused* — not magically regularized.]

== VAE versus deterministic autoencoder #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left, left),
  table.header([], [*Autoencoder*], [*VAE*]),
  [code], [one point], [a distribution],
  [sampling], [unreliable (islands)], [defined by the prior],
  [objective], [reconstruction], [likelihood lower bound],
  [typical output], [useful features], [smooth, often softer samples],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[the KL term is the single change that turns a point-code into a *sampleable* latent space])

== VAE versus GAN — a preview #D

Both use a latent $z$, but their *learning signals* differ radically:
#pause
#two(
  notebox[*VAE* — an *explicit* probabilistic story; maximize a likelihood bound (reconstruction $+$ KL).],
  notebox[*GAN* — *no* explicit likelihood; a generator learns by trying to *fool a discriminator*.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[VAEs optimize a bound on $p(x)$; GANs play a game — sharper samples, no density])

== Connection to diffusion #V

Modern *latent diffusion* leans directly on the autoencoder:
#pause
#ldm
#pause
#align(center, text(size: 14pt, fill: MUTED)[compress pixels to a compact latent, run diffusion *there*, decode once — the AE is an active component])

== Exit ticket #Q

On a small VAE diagram, label the three steps and predict a failure:
#pause
#two(
  notebox[*Label* — (1) the *reconstruction* step, (2) the *KL* step, (3) the *sampling* step.],
  alertbox[*Predict* — what happens to the latent space if we *remove the KL term* entirely?],
)
#pause
#align(center, text(size: 15pt, fill: MUTED)[no KL $->$ the codes drift back into islands $->$ prior samples decode to nonsense again])

== Closing line #Q

#result[VAEs make latent spaces *sampleable* by *regularizing* them]
#pause
#align(center, text(size: 17pt)[
  Next: *GANs* trade the explicit likelihood for an *adversarial game*
  #v(4pt)
  #text(size: 15pt, fill: MUTED)[— giving up density estimation to chase *sharper* samples]
])

// ═══════════════════════════ PART V — Deeper: latent-variable models ═══════════════════════════
= Deeper: latent-variable models

== Autoencoders are nonlinear compression #D

PCA finds the best *linear* low-dimensional reconstruction:
#pause
$ x thick arrow.r.long thick W^top x thick arrow.r.long thick W W^top x $
#pause
An autoencoder replaces the two linear maps with *neural networks*, so it can bend around *nonlinear* manifolds.
#pause
#align(center, text(size: 15pt, fill: MUTED)[an intuition link, *not* a claim that every autoencoder is PCA — the nonlinearity is the point])

== Shape bookkeeping #D

Track the dimensions as data flows through the encoder:
#pause
$ x in RR^(H times W times C) quad arrow.r.long^"encoder" quad z in RR^d $
#pause
A concrete example — a $32 times 32$ grayscale image compressed to a 16-dimensional code:
$ x in RR^("1,024") quad arrow.r.long quad z in RR^16 quad quad #text(size: 15pt, fill: MUTED)[(a $64 times$ squeeze)] $
#pause
#align(center, text(size: 15pt, fill: MUTED)[what must the 16 numbers preserve so the decoder can rebuild all 1,024 pixels?])

== Undercomplete and overcomplete codes #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left),
  table.header([*Code dimension*], [*Pressure on the model*]),
  [$d < "dim"(x)$ (undercomplete)], [must compress salient structure],
  [$d >= "dim"(x)$ (overcomplete)], [may learn an identity shortcut],
  [overcomplete $+$ regularization], [can still learn sparse / robust features],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[the bottleneck can be *dimensional*, *stochastic*, *architectural*, or *regularization*-based])

== Reconstruction likelihood, not just MSE #D

Grounding the familiar loss in probability. If the decoder is Gaussian:
#pause
$ p_theta (x | z) = cal(N)(g_theta (z), sigma_x^2 I) $
#pause
then the negative log-likelihood is *proportional to squared error*:
$ -log p_theta (x | z) = 1 / (2 sigma_x^2) norm(x - g_theta (z))^2 + "const" $
#pause
#align(center, text(size: 15pt, fill: MUTED)[MSE is a *modeling choice* (a Gaussian decoder), not a law of nature])

== Data type determines the decoder output #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left, left),
  table.header([*Data*], [*Decoder distribution*], [*Reconstruction term*]),
  [normalized continuous pixels], [Gaussian], [squared error],
  [binary image], [Bernoulli], [binary cross-entropy],
  [tokens], [categorical], [cross-entropy],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[the decoder predicts *distribution parameters* — match the distribution to the data type])

== The VAE graphical model #V

At generation time the arrow is simple: cause $->$ observation.
#pause
#pgm
#pause
#align(center, text(size: 14pt, fill: MUTED)[training *inverts* it: infer the hidden cause $z$ that plausibly generated each observed $x$])

== Marginal likelihood is an integral #D

The quantity we truly want to maximize:
#pause
$ p_theta (x) = integral p_theta (x | z) thin p(z) thin d z $
#pause
#alertbox[For a nonlinear neural decoder and high-dimensional $z$, this integral has *no closed form* and cannot be evaluated exactly — hence the lower bound.]

== Variational inference: a tractable stand-in #D

Choose a tractable family $q_phi (z | x)$ to *imitate* the true posterior $p_theta (z | x)$:
#pause
#notebox[*Amortized inference.* One encoder network $phi$ handles *every* input — instead of solving a fresh optimization for each image, we *amortize* the cost into shared weights.]
#pause
#align(center, text(size: 15pt, fill: MUTED)[classical VI optimizes per-datapoint; the VAE trains one network to predict $q$ for all of them])

== The exact decomposition #D

The identity behind the bound's *name*:
#pause
$ log p_theta (x) = "ELBO"(x) + D_"KL" (q_phi (z | x) parallel p_theta (z | x)) $
#pause
#elbodecomp
#pause
#align(center, text(size: 14pt, fill: MUTED)[the KL gap is $>= 0$, so ELBO $<=$ $log p_theta (x)$ — a genuine *lower bound*])

== Expand the ELBO #D

Start from the definition and substitute the joint:
#pause
$ "ELBO"(x) = EE_(q_phi (z | x))[log p_theta (x, z) - log q_phi (z | x)] $
#pause
$ p_theta (x, z) = p_theta (x | z) thin p(z) $
#pause
#align(center, text(size: 15pt, fill: MUTED)[substitute the joint, then split the logarithm — the next slide collects the pieces])

== Rearrange into the two familiar terms #D

Grouping the split logarithm gives back the working objective:
#pause
$ "ELBO" = underbrace(EE_q[log p_theta (x | z)], #text(fill: TEAL)[explain the data]) - underbrace(D_"KL" (q_phi (z | x) parallel p(z)), #text(fill: ACC)[organize the latent]) $
#pause
#align(center, text(size: 15pt, fill: MUTED)[same two terms as Part II — now *derived*, not asserted])

== Which KL is which? #D

Two KL expressions appear — students routinely conflate them:
#pause
$ D_"KL" (q(z | x) parallel #text(fill: ACC)[$p(z)$]) quad quad #text(size: 15pt, fill: MUTED)[regularizes each code toward the *prior* (in the loss)] $
#pause
$ D_"KL" (q(z | x) parallel #text(fill: RED)[$p(z | x)$]) quad quad #text(size: 15pt, fill: MUTED)[the *gap* between ELBO and true log-likelihood] $
#pause
#align(center, text(size: 15pt, fill: MUTED)[one is a *term you optimize*; the other is *how loose the bound is*])

== Training loss signs #D

We *maximize* the ELBO, equivalently *minimize* its negative:
#pause
$ cal(L)_"VAE" = underbrace(-EE_q[log p_theta (x | z)], #text(fill: TEAL)["reconstruction loss"]) + underbrace(D_"KL" (q_phi (z | x) parallel p(z)), #text(fill: ACC)["KL loss"]) $
#pause
#align(center, text(size: 15pt, fill: MUTED)[in code the first term is literally the MSE or BCE; the second is the closed-form Gaussian KL])

== Monte Carlo estimate #D

The expectation over $q_phi (z | x)$ is approximated by *sampling* latent vectors:
#pause
$ EE_q[dot] approx 1/K sum_(k=1)^K [dot]_(z^((k))), quad quad z^((k)) = mu + sigma dot.o epsilon^((k)) $
#pause
#align(center, text(size: 15pt, fill: MUTED)[*one* reparameterized sample per input ($K = 1$) is usually enough for the standard objective])

== Why output log-variance? #D

Networks predict $ell = log sigma^2$ rather than $sigma$ directly, then recover:
#pause
$ ell = log sigma^2 quad quad arrow.r.long quad quad sigma = exp(1/2 thin ell) $
#pause
#notebox[$ell$ ranges over *all* of $RR$ — safe for an unconstrained linear output — while $exp(dot)$ keeps the variance *strictly positive*. No clamping, no $log 0$.]
#pause
#align(center, text(size: 15pt, fill: MUTED)[e.g. $ell = -1.386 arrow.r sigma = exp(-0.693) = 0.5$])

== Reparameterization computation graph #V

The full graph with the log-variance parameterization:
#pause
#reparamgraph
#pause
#align(center, text(size: 14pt, fill: MUTED)[$epsilon$ is drawn independently; gradients flow through the *deterministic* transform $mu, sigma -> z$])

== Worked: one training step's loss #D

Take the same encoded point, $mu = [0.8, thin -0.7]$, $sigma = [0.5, thin 0.2]$, with $sigma_x^2 = 1$ and $norm(x - hat(x))^2 = 3.0$:
#pause
$ "KL" = 1/2 [(0.64 + 0.25 + 1.386 - 1) + (0.49 + 0.04 + 3.219 - 1)] = #text(fill: ACC, weight: 600)[$2.01$] $
#pause
$ "recon" = 1/2 norm(x - hat(x))^2 = 1.50, quad quad cal(L)_"VAE" = 1.50 + 2.01 = #text(fill: ACC, weight: 600)[$3.51$] $
#pause
#align(center, text(size: 14pt, fill: MUTED)[both terms are in *nats*; the optimizer trades one against the other])

== Sampling versus reconstruction #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left, left),
  table.header([*Operation*], [*Latent source*], [*Question answered*]),
  [reconstruction], [$z tilde q(z | x)$], [can the model explain this input?],
  [unconditional sample], [$z tilde p(z)$], [what does the model generate?],
  [interpolation], [path between two codes], [is the latent geometry smooth?],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[*where $z$ comes from* changes the meaning of the decoder's output])

== A KL trade-off numeric intuition #D

Two candidate encoders for the *same* data:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 7pt), align: (left, center, center),
  table.header([*Encoder*], [*Reconstruction*], [*KL*]),
  [highly individualized codes], [1], [100],
  [codes near the prior], [10], [1],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[totals $101$ vs $11$: neither is "obviously best" — the *objective* picks the balance, not reconstruction alone])

== Visualize a 2-D latent space #V

Encode a dataset into $(z_1, z_2)$ and decode a *grid* of latent points:
#pause
#fig("/lecture19/figures/latent_manifold.svg", w: 34%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[a good map shows *clusters*, *smooth directions*, *empty regions*, and where *prior samples* land])

== Latent traversal #D

Hold every coordinate fixed and move a *single* latent dimension:
#pause
$ z_j arrow.l z_j + delta, quad quad #text(size: 15pt, fill: MUTED)[(decode several values of $delta$)] $
#pause
#notebox[Interpretable changes along $z_j$ are *interesting evidence* about the learned representation — but *not proof* that the factors are perfectly disentangled.]

== Conditional VAE, expanded #D

For a label or text condition $c$, condition *both* networks:
#pause
$ q_phi (z | x, c), quad quad p_theta (x | z, c) $
#pause
#two(
  notebox[*Controlled by $c$* — the attribute you set (the digit class, the caption).],
  notebox[*Still diverse through $z$* — the style, stroke, and variation *within* that attribute.],
)
#pause
#align(center, text(size: 15pt, fill: MUTED)[sample: choose $c$, draw $z tilde p(z)$, decode $p_theta (x | z, c)$])

== $beta$-VAE and disentanglement claims #D

$ cal(L) = -EE_q[log p_theta (x | z)] + beta thin D_"KL" (q parallel p) $
#pause
Increasing $beta$ strengthens the pressure toward a simple prior.
#pause
#alertbox[It *can* encourage more separated factors on curated data — but "disentangled" is *not* a guaranteed, universal outcome, and reconstruction suffers.]

== Posterior collapse, diagnose it #D

The symptom to watch for:
#pause
$ q_phi (z | x) approx p(z) quad quad #text(size: 15pt, fill: MUTED)[for *every* $x$] $
#pause
#notebox[A powerful decoder predicts $x$ well from context alone, so the KL term drives every posterior to the prior. The latent has become *unused* — inspect per-dimension KL, not just reconstruction.]

== Posterior collapse, high-level responses #D

Design choices, not a new module:
#pause
- *weaken or schedule* the KL pressure early in training (KL annealing);
- *constrain* the decoder's capacity so it cannot ignore $z$;
- *force* more information through the latent (e.g. a free-bits floor);
- *inspect* latent-use statistics, not only reconstructions.
#pause
#align(center, text(size: 15pt, fill: MUTED)[the fixes all *rebalance* how much the model is allowed to rely on the decoder alone])

== VQ-style discrete latents #D

Some autoencoders encode into a *finite codebook* instead of a continuous Gaussian:
#pause
$ #text(fill: TEAL)[continuous VAE latent] quad quad #text(fill: MUTED)[versus] quad quad #text(fill: ACC)[discrete codebook token] $
#pause
#align(center, text(size: 15pt, fill: MUTED)[this previews *tokenized* visual representations — the basis of many image + video generators])
#pause
#interbox(link-to: IA + "vq-codebook")[A codebook widget: snap continuous encodings to their nearest discrete code vectors.]

== Why VAEs matter to modern diffusion #V

Latent diffusion uses the autoencoder to change *what space* the generator works in:
#pause
$ #text(fill: TEAL)[pixels] quad arrow.r.long quad #text(fill: ACC)[compact continuous latents] $
#pause
#ldm
#pause
#align(center, text(size: 14pt, fill: MUTED)[the VAE is not historical background — it is an *active systems component* of latent diffusion])

== How should generative models be evaluated? #Q

No single number suffices — use several lenses:
#pause
- *sample quality and diversity* (are outputs realistic *and* varied?);
- *reconstruction / likelihood* metrics where appropriate;
- *downstream utility* of the learned features;
- *human evaluation* with a transparent protocol;
- *memorization and data-provenance* checks.
#pause
#align(center, text(size: 15pt, fill: MUTED)[a model can score well on one lens and fail badly on another])

== Common misconceptions #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 7pt), align: (left, left),
  table.header([*Misconception*], [*Correction*]),
  [“the encoder outputs one latent point”], [it outputs a *distribution's parameters*],
  [“KL makes samples better directly”], [it *organizes codes* to match the sampling prior],
  [“blurry samples prove VAEs are wrong”], [they reflect likelihood / output trade-offs],
  [“interpolation proves understanding”], [it is a *diagnostic*, not proof],
))

== Retrieval questions #Q

Test yourself before the next lecture:
#pause
+ Why is an ordinary autoencoder *not guaranteed* to generate from a random $z$?
#pause
+ What does the KL term *compare* in a VAE?
#pause
+ Why does $z = mu + sigma dot.o epsilon$ *permit* backpropagation?
#pause
+ What is *posterior collapse*, and how would you *detect* it?

== Next lecture #Q

#align(center, text(size: 18pt)[
  #text(fill: TEAL)[*VAE*]: explain data through a latent *likelihood*
  #v(6pt)
  $arrow.b$
  #v(6pt)
  #text(fill: ACC)[*GAN*]: learn realism through an adversarial *critic*
])
#pause
#align(center, text(size: 15pt, fill: MUTED)[the next model *removes* the explicit reconstruction target and introduces a *game*])

#focus-slide[
  A VAE turns a code into a *sampleable latent variable*: reconstruction $+$ a KL-regularized prior $=$ a generative model.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Generative Adversarial Networks* $->$ diffusion $->$ modern generative models.
]
