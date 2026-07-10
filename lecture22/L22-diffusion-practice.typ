// Diffusion Models — Practice
// Lecture 22 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture22/L22-diffusion-practice.typ /tmp/L22.pdf
//   typst compile --root . --input handout=true lecture22/L22-diffusion-practice.typ /tmp/L22h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.
// Schematic figures: python3 lecture22/diagrams/l22_figs.py

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [Diffusion Models — Practice],
  subtitle: [Latents, text conditioning, and classifier-free guidance],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let cream = rgb("#EFEEEB")
#let rect = fletcher.shapes.rect
// scope inline-math in diagram node labels down from the global 19pt so symbols
// sit proportionally next to the ~9pt node text
#let dgm(body) = align(center, { show math.equation: set text(size: 11pt); body })

// ═══════════════════════════ fletcher diagrams (centerpieces) ═══════════════════════════

// ── (A) system map: noise + condition -> denoiser -> reverse steps -> decoder -> image ──
#let systemmap = dgm(diagram(spacing: (11mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(pos, lbl, c, f) = node(pos, align(center, text(size: 9.5pt, lbl)), shape: rect, corner-radius: 4pt, inset: 6pt, stroke: 0.9pt + c, fill: f)
  node((0, 0), text(size: 10pt)[noise], radius: 6mm, fill: cream)
  bx((1.7, 0), [denoiser \ $epsilon_theta (z_t, t, c)$], INK, cream)
  bx((3.5, 0), [repeated \ reverse steps], TEAL, TEAL.lighten(82%))
  bx((5.1, 0), [VAE \ decoder], BLUE, BLUE.lighten(85%))
  node((6.5, 0), text(size: 10pt)[image], radius: 6mm, fill: GREEN.lighten(80%), stroke: 0.9pt + GREEN)
  for (a, b) in ((0, 1.7), (1.7, 3.5), (3.5, 5.1), (5.1, 6.5)) { edge((a, 0), (b, 0), "-|>", stroke: 0.8pt + MUTED) }
  edge((3.5, 0.55), (1.7, 0.55), "--|>", stroke: 0.8pt + TEAL, bend: -32deg)
  // text-condition branch feeds the denoiser from below
  node((0, -1.6), text(size: 10pt)[prompt], radius: 6mm, fill: cream)
  bx((1.7, -1.6), [text \ encoder], ACC, ACC.lighten(82%))
  edge((0, -1.6), (1.7, -1.6), "-|>", stroke: 0.8pt + MUTED)
  edge((1.7, -1.6), (1.7, 0), "-|>", stroke: 0.9pt + ACC, label: text(size: 8.5pt, fill: ACC)[$c$], label-side: right)
  // time enters from above
  node((1.7, 1.35), text(size: 9pt, fill: BLUE)[time $t$], stroke: none, fill: none)
  edge((1.7, 1.1), (1.7, 0.42), "-|>", stroke: 0.8pt + BLUE)
}))

// ── (B) latent diffusion, end to end — THE centerpiece ──
#let latentdiff = dgm(diagram(spacing: (8.5mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(pos, lbl, c, f) = node(pos, align(center, text(size: 9pt, lbl)), shape: rect, corner-radius: 4pt, inset: 5pt, stroke: 0.9pt + c, fill: f)
  // pixel space in / out
  node((0, 0), align(center, text(size: 9pt)[training \ image $x$]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: cream, stroke: 0.9pt + INK)
  bx((1.35, 0), [VAE \ encoder $E$], TEAL, TEAL.lighten(82%))
  node((2.6, 0), $z_0$, radius: 5mm, fill: TEAL.lighten(70%), stroke: 0.9pt + TEAL)
  bx((3.8, 0), [add \ noise], BLUE, BLUE.lighten(85%))
  node((5.0, 0), $z_t$, radius: 5mm, fill: BLUE.lighten(72%), stroke: 0.9pt + BLUE)
  bx((6.5, 0), [U-Net denoiser \ $epsilon_theta (z_t, t, c)$], INK, cream)
  node((8.1, 0), $hat(z)_0$, radius: 5mm, fill: ACC.lighten(70%), stroke: 0.9pt + ACC)
  bx((9.35, 0), [VAE \ decoder $D$], BLUE, BLUE.lighten(85%))
  node((10.7, 0), align(center, text(size: 9pt)[generated \ image]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: GREEN.lighten(80%), stroke: 0.9pt + GREEN)
  for (a, b) in ((0, 1.35), (1.35, 2.6), (2.6, 3.8), (3.8, 5.0), (5.0, 6.5), (6.5, 8.1), (8.1, 9.35), (9.35, 10.7)) {
    edge((a, 0), (b, 0), "-|>", stroke: 0.75pt + MUTED)
  }
  // text conditioning enters the U-Net from below
  node((5.0, -1.7), align(center, text(size: 9pt)[text prompt $c$]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: cream, stroke: 0.9pt + INK)
  bx((6.5, -1.7), [text \ encoder], ACC, ACC.lighten(82%))
  edge((5.0, -1.7), (6.5, -1.7), "-|>", stroke: 0.75pt + MUTED)
  edge((6.5, -1.7), (6.5, 0), "-|>", stroke: 0.9pt + ACC, label: text(size: 8pt, fill: ACC)[cross-attn], label-side: right)
  // space labels
  node((1.3, 1.15), text(size: 8.5pt, fill: MUTED)[← pixel space], stroke: none, fill: none)
  node((5.0, 1.15), text(size: 8.5pt, fill: BLUE)[latent space (diffusion here)], stroke: none, fill: none)
  node((10.0, 1.15), text(size: 8.5pt, fill: MUTED)[pixel space →], stroke: none, fill: none)
}))

// ── (C) U-Net: encoder down, bottleneck, decoder up, skip connections ──
#let unet = dgm(diagram(spacing: (11mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let res(pos, lbl, c) = node(pos, text(size: 9.5pt, lbl), shape: rect, corner-radius: 4pt, inset: 6pt, stroke: 0.9pt + c, fill: c.lighten(82%))
  // encoder (down the left), decoder (up the right)
  res((0, 0), [$64^2$], TEAL); res((1, -1.2), [$32^2$], TEAL); res((2, -2.4), [$16^2$], TEAL)
  node((3.5, -3.1), align(center, text(size: 9pt)[bottleneck]), shape: rect, corner-radius: 4pt, inset: 6pt, stroke: 1pt + INK, fill: cream)
  res((5, -2.4), [$16^2$], ACC); res((6, -1.2), [$32^2$], ACC); res((7, 0), [$64^2$], ACC)
  // main path
  for (a, b) in (((0,0),(1,-1.2)), ((1,-1.2),(2,-2.4)), ((2,-2.4),(3.5,-3.1)), ((3.5,-3.1),(5,-2.4)), ((5,-2.4),(6,-1.2)), ((6,-1.2),(7,0))) {
    edge(a, b, "-|>", stroke: 0.8pt + MUTED)
  }
  // input / output
  edge((-1, 0), (0, 0), "-|>", stroke: 0.9pt + INK, label: text(size: 8.5pt)[$z_t$], label-side: left)
  edge((7, 0), (8, 0), "-|>", stroke: 0.9pt + INK, label: text(size: 8.5pt)[$hat(epsilon)$], label-side: right)
  // skip connections (dashed, arcing over the top)
  for (a, b) in (((0,0),(7,0)), ((1,-1.2),(6,-1.2)), ((2,-2.4),(5,-2.4))) {
    edge(a, b, "--|>", stroke: (paint: GREEN, dash: "dashed", thickness: 0.8pt), bend: 32deg)
  }
  node((3.5, 0.9), text(size: 9pt, fill: GREEN)[skip connections carry fine detail], stroke: none, fill: none)
  // time + text injected at the bottleneck
  node((3.5, -4.3), align(center, text(size: 8.5pt)[time $t$, condition $c$]), shape: rect, corner-radius: 4pt, inset: 4pt, stroke: 0.9pt + BLUE, fill: BLUE.lighten(85%))
  edge((3.5, -4.3), (3.5, -3.1), "-|>", stroke: 0.8pt + BLUE)
}))

// ── (D) cross-attention conditioning: latent queries, text keys/values ──
#let crossattn = dgm(diagram(spacing: (16mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0.9), align(center, text(size: 9.5pt)[latent features \ (query $Q$)]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL)
  node((0, -0.9), align(center, text(size: 9.5pt)[text embeddings \ (keys $K$, values $V$)]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  node((2, 0), align(center, text(size: 9.5pt)[cross-attention \ $"softmax"(Q K^top \/ sqrt(d)) V$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: BLUE.lighten(85%), stroke: 1pt + BLUE)
  node((3.7, 0), align(center, text(size: 9.5pt)[conditioned \ latent]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  edge((0, 0.9), (2, 0), "-|>", stroke: 0.9pt + TEAL)
  edge((0, -0.9), (2, 0), "-|>", stroke: 0.9pt + ACC)
  edge((2, 0), (3.7, 0), "-|>", stroke: 0.9pt + INK)
}))

// ── (E) classifier-free guidance flow: two passes -> combine -> guided noise ──
#let cfgflow = dgm(diagram(spacing: (13mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), $z_t$, radius: 5.5mm, fill: cream)
  node((1.8, 0.9), align(center, text(size: 9.5pt)[$epsilon_theta (z_t, t, c)$ \ conditional]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE)
  node((1.8, -0.9), align(center, text(size: 9.5pt)[$epsilon_theta (z_t, t, nothing)$ \ unconditional]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: MUTED.lighten(70%), stroke: 0.9pt + MUTED)
  node((4.0, 0), align(center, text(size: 9.5pt)[combine \ $epsilon_nothing + w(epsilon_c - epsilon_nothing)$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: ACC.lighten(80%), stroke: 1.1pt + ACC)
  node((5.9, 0), $hat(epsilon)$, radius: 5mm, fill: ACC.lighten(68%), stroke: 0.9pt + ACC)
  edge((0, 0), (1.8, 0.9), "-|>", stroke: 0.9pt + BLUE)
  edge((0, 0), (1.8, -0.9), "-|>", stroke: 0.9pt + MUTED)
  edge((1.8, 0.9), (4.0, 0), "-|>", stroke: 0.9pt + BLUE)
  edge((1.8, -0.9), (4.0, 0), "-|>", stroke: 0.9pt + MUTED)
  edge((5.9, 0), (7.0, 0), "-|>", stroke: 0.9pt + INK, label: text(size: 8.5pt)[reverse step], label-side: right)
  edge((4.0, 0), (5.9, 0), "-|>", stroke: 0.9pt + ACC)
  node((2.9, 1.7), text(size: 9pt, fill: INK)[*two* forward passes per step, *one* network], stroke: none, fill: none)
}))

// ── (F) DDPM vs DDIM: same denoiser, many stochastic vs few deterministic steps ──
#let samplers = dgm(diagram(spacing: (6mm, 10mm), node-stroke: 0.6pt + INK, {
  // DDPM row (top): many small steps
  node((-0.9, 1), text(size: 9.5pt)[$z_T$], stroke: none)
  let ddpm = (0, 0.55, 1.1, 1.65, 2.2, 2.75, 3.3, 3.85, 4.4, 4.95, 5.5)
  for (i, x) in ddpm.enumerate() { node((x, 1), none, radius: 1.6mm, fill: BLUE, stroke: none) }
  for i in range(ddpm.len() - 1) { edge((ddpm.at(i), 1), (ddpm.at(i+1), 1), "-|>", stroke: 0.6pt + BLUE) }
  node((6.4, 1), text(size: 9.5pt)[$z_0$], stroke: none)
  node((2.75, 1.7), align(center, text(size: 9pt, fill: BLUE)[*DDPM* — many stochastic steps (e.g. 1000)]), stroke: none)
  // DDIM row (bottom): few big deterministic jumps
  node((-0.9, -1), text(size: 9.5pt)[$z_T$], stroke: none)
  let ddim = (0, 1.83, 3.66, 5.5)
  for (i, x) in ddim.enumerate() { node((x, -1), none, radius: 2.2mm, fill: ACC, stroke: none) }
  for i in range(ddim.len() - 1) { edge((ddim.at(i), -1), (ddim.at(i+1), -1), "-|>", stroke: 1.1pt + ACC) }
  node((6.4, -1), text(size: 9.5pt)[$z_0$], stroke: none)
  node((2.75, -1.75), align(center, text(size: 9pt, fill: ACC)[*DDIM* — few deterministic steps (e.g. 20–50)]), stroke: none)
}))

// ── (G) full Stable-Diffusion pipeline: text + seed -> denoise loop -> decode -> safety ──
#let sdpipe = dgm(diagram(spacing: (9mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(pos, lbl, c, f) = node(pos, align(center, text(size: 9pt, lbl)), shape: rect, corner-radius: 4pt, inset: 6pt, stroke: 0.9pt + c, fill: f)
  // conditioning row (top)
  node((0, 1.3), text(size: 9.5pt)[prompt], radius: 6mm, fill: cream)
  bx((1.6, 1.3), [CLIP text \ encoder], ACC, ACC.lighten(82%))
  // noise row (bottom)
  node((0, -1.3), text(size: 9.5pt)[seed], radius: 6mm, fill: cream)
  bx((1.6, -1.3), [noise \ latent $z_T$], MUTED, MUTED.lighten(72%))
  // denoise loop (center)
  bx((3.6, 0), [U-Net denoise $times N$ \ (cross-attn $+$ time)], INK, cream)
  edge((1.6, 1.3), (3.6, 0), "-|>", stroke: 0.9pt + ACC, label: text(size: 8pt, fill: ACC)[$c$], label-side: left)
  edge((1.6, -1.3), (3.6, 0), "-|>", stroke: 0.9pt + MUTED, label: text(size: 8pt, fill: MUTED)[$z_T$], label-side: left)
  edge((4.35, 0.6), (2.85, 0.6), "--|>", stroke: 0.8pt + INK, bend: -40deg, label: text(size: 8pt)[loop], label-side: left)
  // decode + safety
  bx((5.4, 0), [VAE \ decoder], BLUE, BLUE.lighten(85%))
  bx((7.1, 0), [safety \ filter], RED, RED.lighten(85%))
  node((8.6, 0), align(center, text(size: 9pt)[output \ image]), radius: 6.5mm, fill: GREEN.lighten(80%), stroke: 0.9pt + GREEN)
  edge((3.6, 0), (5.4, 0), "-|>", stroke: 0.9pt + INK, label: text(size: 8pt)[$z_0$], label-side: right)
  edge((5.4, 0), (7.1, 0), "-|>", stroke: 0.75pt + MUTED)
  edge((7.1, 0), (8.6, 0), "-|>", stroke: 0.75pt + MUTED)
}))

// ── (H) image-to-image: encode -> partial noise -> denoise with new prompt -> decode ──
#let img2img = dgm(diagram(spacing: (9mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(pos, lbl, c, f) = node(pos, align(center, text(size: 9pt, lbl)), shape: rect, corner-radius: 4pt, inset: 5pt, stroke: 0.9pt + c, fill: f)
  node((0, 0), align(center, text(size: 9pt)[input \ image]), radius: 6mm, fill: cream)
  bx((1.4, 0), [encode], TEAL, TEAL.lighten(82%))
  node((2.6, 0), $z_0$, radius: 5mm, fill: TEAL.lighten(70%), stroke: 0.9pt + TEAL)
  bx((4.0, 0), [add noise \ to level $t^*$], BLUE, BLUE.lighten(85%))
  node((5.4, 0), $z_(t^*)$, radius: 5mm, fill: BLUE.lighten(72%), stroke: 0.9pt + BLUE)
  bx((6.9, 0), [denoise with \ new prompt], INK, cream)
  bx((8.5, 0), [decode], BLUE, BLUE.lighten(85%))
  node((9.8, 0), align(center, text(size: 9pt)[edited \ image]), radius: 6mm, fill: GREEN.lighten(80%), stroke: 0.9pt + GREEN)
  for (a, b) in ((0, 1.4), (1.4, 2.6), (2.6, 4.0), (4.0, 5.4), (5.4, 6.9), (6.9, 8.5), (8.5, 9.8)) {
    edge((a, 0), (b, 0), "-|>", stroke: 0.75pt + MUTED)
  }
  node((4.7, 1.0), text(size: 8.5pt, fill: BLUE)[more initial noise → more freedom to change the input], stroke: none, fill: none)
}))

#title-slide()

// ═══════════════════════════ PART I — Turn theory into a system ═══════════════════════════
= Turn the theory into a working system

== Recall the minimal theory #Q

L21 gave us one denoiser and one reverse update:
#pause
$ (z_t, t) quad -->^(epsilon_theta) quad hat(epsilon) quad -->^"reverse update" quad z_(t-1) $
#pause
#notebox[Everything else in a modern text-to-image system is an *engineering decision* around this loop:]
#pause
#align(center, text(size: 16pt, fill: MUTED)[*representation* (what space?) · *architecture* (which denoiser?) · *condition* (what steers it?) · *sampling* (how fast?)])

== The question for this lecture #Q

Not "how do we memorize Stable Diffusion?" but:
#pause
#result[where does each input — noise level, text, image, mask, control — *enter* the denoising computation?]
#pause
#align(center, text(size: 16pt, fill: MUTED)[one worked mechanism: *classifier-free guidance* · one diagram: *latent diffusion end-to-end*])

== The system map #V

Every input joins the same denoising loop at a specific place:
#pause
#systemmap
#pause
#align(center, text(size: 14pt, fill: MUTED)[noise $->$ denoiser (conditioned on time $+$ text) $->$ repeated reverse steps $->$ decode $->$ image])

== Why not diffuse raw pixels? #D

A single $512 times 512 times 3$ RGB image is a very large tensor:
#pause
$ 512 times 512 times 3 = #text(fill: RED, weight: 600)[$"786,432"$] quad #text(size: 15pt, fill: MUTED)[values] $
#pause
The reverse process calls the denoiser *dozens to hundreds of times*. Denoising a 786k-value tensor at every step is expensive.
#pause
#align(center, text(size: 16pt, fill: MUTED)[can we run the *same* diffusion loop on a much *smaller* tensor?])

== The latent-diffusion idea #D

Compress the image into a compact latent with a VAE (L19), diffuse *there*, decode at the end:
#pause
$ x quad -->^E quad z quad quad #text(size: 15pt, fill: MUTED)[diffuse in $z$-space], quad quad z_0 quad -->^D quad hat(x) $
#pause
#notebox[Reuse the VAE's *continuous, compact* representation. The denoiser never touches raw pixels during the loop — only at the encode/decode ends.]

== Why latents pay off #V

The same $512 times 512$ image, encoded to a $64 times 64 times 4$ latent:
#pause
#fig("/lecture22/figures/latent_compression.svg", w: 52%)
#pause
#align(center, text(size: 15pt, fill: MUTED)[$"786,432" -> "16,384"$ values — about $48 times$ fewer to denoise at *every* reverse step])

== What the autoencoder must preserve #D

The latent is *not* just a JPEG compressor:
#pause
- it should retain *perceptual content* while cutting spatial cost;
#pause
- its *decoder* defines the ceiling on what the diffusion model can ultimately render.
#pause
#alertbox[If the VAE cannot decode a detail, *no amount* of diffusion in latent space can put it back. The decoder is the renderer.]

== Latent diffusion, end to end #V

The full picture — pixel space at the ends, diffusion in the middle, text entering the U-Net:
#pause
#latentdiff
#pause
#align(center, text(size: 13pt, fill: MUTED)[encode $->$ noise $->$ denoise (conditioned on $t$ and $c$) $->$ predicted clean latent $hat(z)_0$ $->$ decode])

== The training loss stays familiar #D

Only the *space* and the *conditioning input* change — the objective is L21's, unchanged:
#pause
$ cal(L) = EE_(z_0, t, epsilon, c) [thin norm(epsilon - epsilon_theta (z_t, t, c))_2^2 thin] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[predict the noise added to the *latent* $z_t$, now also given the condition $c$ — same regression as before])

== Freeze or train the autoencoder? #D

A common latent-diffusion recipe splits the two hard problems:
#pause
+ train (or obtain) an image autoencoder;
#pause
+ *freeze* it;
#pause
+ train the denoiser in the frozen latent space;
#pause
+ decode *only* the final samples.
#pause
#align(center, text(size: 15pt, fill: MUTED)[separates *representation compression* from the expensive *generative* training])

== Latent scaling matters #D

The noise schedule assumes a predictable latent *magnitude*:
#pause
- latents are normalized / rescaled so that data latents and Gaussian noise have *compatible* scales;
#pause
- a fixed scaling factor makes $z_0$ and $epsilon ~ cal(N)(0, I)$ numerically comparable.
#pause
#notebox[The broader point: *interfaces between learned modules need numerical contracts* — the encoder's output distribution must match what the diffusion process expects.]

// ═══════════════════════════ PART II — The denoiser architecture ═══════════════════════════
= The denoiser architecture

== What structure does a denoiser need? #Q

To clean a noisy latent well, the network needs four things:
#pause
- *local detail* at many resolutions;
- a *global view* of the composition;
- a *route* for conditioning information;
- knowledge of the *current noise level* $t$.
#pause
#align(center, text(size: 16pt, fill: MUTED)[the U-Net (and its Transformer successors) is built to provide exactly these])

== The U-Net at a glance #V

A symmetric encoder–decoder with skip connections:
#pause
#unet
#pause
#align(center, text(size: 13pt, fill: MUTED)[downsample for global context $->$ bottleneck $->$ upsample for detail, with skips bridging matched resolutions])

== Why the U shape helps #D

The two halves solve complementary problems:
#pause
#two(
  notebox[*Down* — pooling grows the receptive field, so the bottleneck sees *global structure* (layout, objects).],
  notebox[*Up* — skip connections re-inject *fine spatial detail* the bottleneck blurred away.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[global reasoning *and* local detail — neither alone reconstructs a clean latent])

== The resolution hierarchy #D

The denoiser walks *down* then back *up* a resolution ladder:
#pause
$ 64^2 -> 32^2 -> 16^2 -> #text(fill: INK)[bottleneck] -> 16^2 -> 32^2 -> 64^2 $
#pause
- high resolution keeps *fine texture*; low resolution supplies *global context*;
#pause
- each up-step *concatenates* the matching encoder feature before convolving.
#pause
#align(center, text(size: 16pt, fill: MUTED)[the same ladder a CNN backbone uses — here run *symmetrically* for reconstruction])

== Time embedding #D

The same noisy latent must be denoised *differently* early and late in the process:
#pause
- turn the integer timestep $t$ into a learned vector $e_t$ (sinusoidal $->$ MLP);
#pause
- inject $e_t$ into every residual block.
#pause
#align(center, text(size: 16pt, fill: MUTED)[at $t = 900$ almost everything is noise; at $t = 20$ only a little — the model must know which])

== Residual blocks and timestep injection #D

Each block mixes spatial features $h$ with the timestep embedding $e_t$:
#pause
$ h' = "ResBlock"(h, e_t) quad quad #text(size: 15pt, fill: MUTED)[($e_t$ scales / shifts the block's features)] $
#pause
#notebox[The time embedding *modulates* the computation, so one visual feature can be handled one way at high noise and another way at low noise.]

== Skip connections carry high-frequency detail #D

A deep bottleneck weakens precise location and texture:
#pause
$ underbrace(#text[encoder feature], "at a resolution") quad --> quad underbrace(#text[decoder input], "same resolution") $
#pause
#align(center, text(size: 16pt, fill: MUTED)[skips let *global reasoning* (bottleneck) and *local detail* (early layers) coexist in the output])

== Text is not pasted onto pixels #D

A text encoder turns the prompt into a *sequence of embeddings*:
#pause
$ #text[prompt] quad -->^"tokenize" quad #text[tokens] quad -->^"encoder" quad c = (c_1, dots, c_L) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the denoiser never sees characters — it accesses $c$ through *cross-attention* inside its blocks])

== Cross-attention reminder #D

The attention you already know, with a twist on where $Q, K, V$ come from:
#pause
$ "Attention"(Q, K, V) = "softmax"(Q K^top / sqrt(d)) V $
#pause
#result[queries $Q$ come from the *latent/image* features; keys $K$ and values $V$ come from the *text* tokens]

== Cross-attention shapes #D

Let the latent have $N$ spatial positions and the prompt have $L$ tokens:
#pause
$ Q in RR^(N times d), quad quad K, V in RR^(L times d) $
#pause
$ Q K^top in RR^(N times L) quad quad #text(size: 15pt, fill: MUTED)[(every image position scores every prompt token)] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the $N times L$ attention matrix routes text information to *each spatial location*])

== Cross-attention as conditioning #V

The whole conditioning mechanism in one block:
#pause
#crossattn
#pause
#align(center, text(size: 14pt, fill: MUTED)[at every location the denoiser asks: *which prompt tokens matter for cleaning up this region now?*])

== Interactive: text conditioning #I

#interbox(link-to: IA + "text-diffusion")[
  Type a prompt, watch it tokenize into text embeddings, then see cross-attention route each token to the image regions that attend to it during denoising.
]

== Cross-attention interpretation #D

For "a red bicycle beside a tree":
#pause
- a location on the bicycle can attend more to *bicycle* and *red*;
- background locations may attend more to *tree*.
#pause
#alertbox[This is an *intuition for information flow*, not a proof of exact visual grounding — heads are learned and often diffuse.]

== Text encoder choices #D

The denoiser reuses representation learning from L14–L18:
#pause
- a *pretrained* text encoder (e.g. CLIP text tower) maps tokens to *contextual* embeddings;
#pause
$ #text[prompt] -> #text[tokens] -> (c_1, dots, c_L) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[stronger text encoders $=$ better prompt understanding — the conditioning is only as good as $c$])

== Practical condition types #V

The same cross-attention / concatenation machinery accepts many conditions:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*Condition*], [*Example use*]),
  [text], [describe desired semantics / style],
  [class label], [class-conditional generation],
  [image / low-res image], [image-to-image, super-resolution],
  [mask], [inpainting],
  [pose / edges / depth], [layout and structure control],
))

== Architecture landscape — one slide #D

#pause
#two(
  notebox[*U-Net* — the intuitive teaching architecture; convolutions $+$ cross-attention at each resolution.],
  notebox[*Diffusion Transformer (DiT)* — replaces much of the conv backbone with *token* processing over latent patches.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[architecture changes; the *noise-prediction objective* is the stable idea beneath both])

// ═══════════════════════════ PART III — Classifier-free guidance ═══════════════════════════
= Make a condition matter: classifier-free guidance

== The conditioning problem #Q

A conditional model can produce *plausible* images that only *weakly* follow the prompt:
#pause
- training maximizes likelihood, not prompt adherence;
#pause
- we want *stronger* adherence — *without* a separate external classifier.
#pause
#align(center, text(size: 16pt, fill: MUTED)[how do we amplify the prompt's influence using only the denoiser we already trained?])

== Train with occasionally dropped conditions #D

During training, sometimes replace $c$ with an *empty* condition $nothing$:
#pause
$ epsilon_theta (z_t, t, c) quad quad #text[and] quad quad epsilon_theta (z_t, t, nothing) $
#pause
#notebox[*One* network learns both a *conditional* and an *unconditional* denoiser — the empty condition is a real, learned input (typically dropped ~10% of the time).]

== CFG training procedure #D

For each training example:
#pause
+ choose the condition $c$;
#pause
+ with some probability, *replace* $c$ by the empty condition $nothing$;
#pause
+ sample a timestep $t$ and noise $epsilon$;
#pause
+ train $epsilon_theta (z_t, t, c)$ to predict $epsilon$.
#pause
#align(center, text(size: 15pt, fill: MUTED)[no architecture change — just occasionally blank the condition])

== Condition dropout is not prompt dropout #D

A subtle but important distinction:
#pause
#two(
  alertbox[*Not* this — drop a few *words* from the prompt. That is still a (weaker) conditional model.],
  notebox[*This* — drop the *entire condition representation* $c -> nothing$, giving a genuine *unconditional* estimate.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[the empty prompt must be a *meaningful learned condition* to compare against later])

== Classifier-free guidance #V

At sampling time, run *both* predictions and extrapolate:
#pause
#cfgflow
#pause
$ hat(epsilon)_"guided" = epsilon_theta (z_t, t, nothing) + w [thin epsilon_theta (z_t, t, c) - epsilon_theta (z_t, t, nothing) thin] $

== Read the formula in words #D

$ hat(epsilon) = epsilon_nothing + w (epsilon_c - epsilon_nothing) $
#pause
- start from what the model would create *unconditionally* ($epsilon_nothing$);
#pause
- move *further* in the direction the prompt *changes* the prediction, scaled by $w$.
#pause
#align(center, text(size: 16pt, fill: MUTED)[$w = 0$: ignore the prompt · $w = 1$: plain conditional · $w > 1$: *amplified* prompt influence])

== Board example #D

Take a scalar toy: unconditional $= 2$, conditional $= 1.5$, guidance $w = 4$:
#pause
$ hat(epsilon) = 2 + 4 (1.5 - 2) = 2 + 4 (-0.5) = 2 - 2 = #text(fill: ACC, weight: 700)[0] $
#pause
#notebox[The guided prediction ($0$) is pushed *past* the conditional value ($1.5$). A larger $w$ *amplifies* the conditional–unconditional gap — it is *not* a probability.]

== Guidance is vector extrapolation #V

In two dimensions, $hat(epsilon)$ extrapolates *beyond* the conditional direction:
#pause
#fig("/lecture22/figures/cfg_vectors.svg", w: 40%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[$w = 1$: land on $epsilon_c$ · $w > 1$: shoot past it along the $(epsilon_c - epsilon_nothing)$ direction])

== CFG misuse check #D

A common misreading:
#pause
$ w #text(fill: RED)[$eq.not$] Pr(#text[prompt is correct]) $
#pause
$w$ is a *heuristic scale* on one prediction relative to another — nothing more.
#pause
#alertbox[Very high $w$ over-extrapolates: samples become *less natural*, *oversaturated*, and *less diverse*.]

== Guidance-scale trade-off #V

Sweeping $w$ trades two things against each other:
#pause
#fig("/lecture22/figures/cfg_scale.svg", w: 52%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[low $w$: diverse, weaker adherence · high $w$: strong adherence, can look oversaturated / repetitive])

== Interactive: guidance scale #I

#interbox(link-to: IA + "cfg-scale-visualizer")[
  Drag the guidance scale $w$ and watch $hat(epsilon) = epsilon_nothing + w(epsilon_c - epsilon_nothing)$ extrapolate — see prompt adherence rise and diversity fall as $w$ grows.
]

== Prompt wording is distribution matching #D

A prompt is a *condition*, not a spell:
#pause
- different phrasing $=$ different text embeddings $c$ $=$ a different steering vector;
#pause
- "steering" means *within* the learned training distribution, not conjuring the impossible.
#pause
#align(center, text(size: 16pt, fill: MUTED)[prompt engineering $=$ finding the $c$ whose region of the distribution you actually want])

// ═══════════════════════════ PART IV — Sampling, editing, trade-offs ═══════════════════════════
= Sampling, editing, and the engineering trade-off

== Sampling is an iterative solver #D

The reverse update need *not* use every original training step:
#pause
- training uses a fine schedule (e.g. $T = 1000$);
#pause
- sampling can take a *shorter path* through the *same* denoiser.
#pause
#align(center, text(size: 16pt, fill: MUTED)[faster samplers re-traverse the learned reverse field with *fewer, larger* steps])

== The sampler is an inference algorithm #D

Separate two things that are easy to conflate:
#pause
$ underbrace(#text[model parameters $theta$], "trained once") quad quad #text[vs.] quad quad underbrace(#text[number and type of reverse updates], "chosen at inference") $
#pause
#notebox[The denoiser is trained *once*. Many compatible samplers can reuse its weights, but their schedule, quality, and diversity trade-offs still need to be checked.]

== DDPM vs accelerated sampling #V

#pause
#samplers
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 5pt), align: (left, left, left),
  table.header([], [*Many stochastic steps*], [*Fewer scheduled steps*]),
  [benefit], [faithful, simple formulation], [lower latency],
  [cost], [slow generation], [possible quality / diversity trade-off],
))
#align(center, text(size: 13pt, fill: MUTED)[DDIM is *one* example of a fast, deterministic-style sampler — we do not derive its update here])

== Sampling-step budget #V

More steps $=$ more denoiser calls; the returns diminish:
#pause
#fig("/lecture22/figures/sampler_steps.svg", w: 50%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[fewer steps: fast iteration · more steps: final rendering — *no* single count is universally optimal])

== Interactive: the reverse process #I

#interbox(link-to: IA + "diffusion-denoise")[
  Step through the reverse process one update at a time, and watch how the step schedule and count change the denoising trajectory from $z_T$ to $z_0$.
]

== Seed and reproducibility #D

A seed fixes the *initial latent noise* $z_T$:
#pause
$ #text[same seed] quad => quad #text[same (or nearly same) trajectory] $
#pause
holding model, sampler, steps, guidance, and condition fixed.
#pause
#notebox[For a scientific demonstration, *record the seed*, model version, step count, and guidance — otherwise the result is not reproducible.]

== Noise schedules #V

The schedule sets how fast signal decays into noise — a real design knob:
#pause
#fig("/lecture22/figures/noise_schedule.svg", w: 50%)
#pause
#align(center, text(size: 14pt, fill: MUTED)[cosine schedules keep *signal* $sqrt(overline(alpha)_t)$ longer at high $t$, spending more steps where it matters])

== Image-to-image #V

Start from an *encoded input*, add noise only partway, then denoise under a new prompt:
#pause
#img2img
#pause
$ #text[more initial noise] quad => quad #text[more freedom to change the input] $

== Inpainting #D

Preserve known regions, regenerate only the masked one:
#pause
- at each step, keep a *suitably noised* version of the *known* pixels outside the mask;
- let the denoiser fill the *masked* region, conditioned on visible context $+$ prompt.
#pause
#alertbox[The model must keep *matching boundaries and semantics* across the visible/edited seam — editing is not a clean cut-and-paste.]

== Structural control #V

Text says *what*; a structural condition says *where*:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*Input condition*], [*What it constrains*]),
  [edges], [outline / composition],
  [pose], [body arrangement],
  [depth], [geometry],
  [segmentation], [regions and classes],
))
#align(center, text(size: 13pt, fill: MUTED)[a control signal gives the denoiser an extra *layout* representation — it does not eliminate all errors])

== Diffusion Transformers #D

Patch/token-based backbones swap the conv U-Net for Transformer blocks:
#pause
- inputs are still *noisy latent tokens* $+$ *timestep* $+$ *condition tokens*;
#pause
- attention replaces much of the convolutional mixing.
#pause
#align(center, text(size: 16pt, fill: MUTED)[the architecture changes; L21's *denoising loss remains recognizable*])

== Fine-tuning a diffusion model #D

Parameter-efficient adaptation adjusts a *small* subset of weights:
#pause
- learn a new *style* or *domain* (LoRA-style, as in the LLM module);
#pause
- inherits the same adaptation trade-offs — and raises *data-consent* questions.
#pause
#alertbox[Do not make a first offering depend on a *live* fine-tuning exercise — use pre-computed adapters if you must demo.]

== Adaptation boundaries #D

Fine-tuning can change:
#pause
- a *style* · a new *visual domain* · a *text/image association* · a *safety / refusal* behavior.
#pause
#notebox[Ask: *what data licensed the adaptation*, and does the base model's capability *shift in unintended ways*?]

== The full Stable-Diffusion pipeline #V

Putting every piece together — the system you actually run:
#pause
#sdpipe
#pause
#align(center, text(size: 13pt, fill: MUTED)[text $+$ seed $->$ conditioned denoise loop ($times N$) $->$ decode $->$ safety filter $->$ image])

== Where the time goes #D

Not all stages cost the same:
#pause
- *text encoding* — once;
- *denoiser* — called *every* sampling step ($times N$);
- *VAE decode* — once, at the end.
#pause
#result[step count drives latency far more than prompt length — the loop is the bottleneck]

== Batch generation is not free #D

Generating several images together interacts with the hardware:
#pause
- better accelerator *utilization*, but *more memory*;
#pause
- latent size, denoiser architecture, and step count all interact with throughput.
#pause
#align(center, text(size: 16pt, fill: MUTED)[this is exactly the L23 systems discussion — throughput vs latency vs memory])

== A small reproducible demo #D

Fix the seed and prompt, then change *one* variable at a time:
#pause
+ vary the *guidance scale* $w$;
+ vary the *number of inference steps*;
+ change *one structural condition*.
#pause
#align(center, text(size: 15pt, fill: MUTED)[ask students to identify which variable changed *composition*, *fidelity*, and *latency*])

== A reproducible classroom exercise #D

Prepare a grid *in advance* (seeds/settings recorded):
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 5pt), align: (left, left),
  table.header([*Intervention*], [*Which axis it probes*]),
  [fixed prompt, vary $w$], [meaning vs. diversity],
  [fixed seed, vary steps], [fidelity vs. latency],
  [fixed settings, vary one word], [prompt sensitivity],
  [fixed prompt, add pose / edge], [composition control],
))

== A practical system checklist #D

For a new diffusion application, *document*:
#pause
#two(
  notebox[+ model and VAE *versions* \ + prompt / structural *condition* \ + *seed*],
  notebox[+ sampler and *step budget* \ + *guidance scale* \ + post-processing / *safety filters*],
)
#pause
#align(center, text(size: 15pt, fill: MUTED)[this turns an impressive image into a *reproducible experimental artifact*])

== Failure modes to discuss honestly #D

#pause
- text/image *misalignment* and unwanted objects;
- distorted *text*, *hands*, or *counting*;
- inherited *bias* and training-data *memorization*;
- high *compute and energy* cost.
#pause
#align(center, text(size: 16pt, fill: MUTED)[fluent-looking output $!=$ correct or safe output])

== Failure analysis grid #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*Symptom*], [*Likely lever to inspect*]),
  [ignores prompt], [guidance, text condition, data mismatch],
  [wrong composition], [structural condition, seed, prompt ambiguity],
  [weak detail], [step budget, latent / decoder capacity],
  [repeated artifacts], [data / model limit, excessive guidance],
))
#align(center, text(size: 13pt, fill: MUTED)[a *starting point*, not a deterministic one-to-one diagnosis])

== Common misconceptions #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Misconception*], [*Correction*]),
  [text is painted into the image], [text embeddings condition denoising via learned attention],
  [higher guidance always improves quality], [it trades adherence against naturalness / diversity],
  [a seed is a semantic label], [it selects initial noise, not a named concept],
  [diffusion editing is exact], [it is probabilistic; it may alter unintended regions],
))

== Provenance and consent #D

Technical control is *not* the same as permission. Discuss separately:
#pause
- what *training data* a model may have seen;
- whether output *resembles* a living artist or a training item;
- how users should *label* generated content;
- who *bears the cost* of errors in high-stakes imagery.
#pause
#align(center, text(size: 16pt, fill: MUTED)[capability $!=$ accountability])

== Comparison with autoregressive image tokens #V

Two families, two ways to factor $p(#text[image])$:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left, left),
  table.header([*Family*], [*Generate unit*], [*Main inference pattern*]),
  [autoregressive visual tokens], [one token at a time], [causal decoding],
  [diffusion], [whole latent, repeatedly], [iterative denoising],
))
#align(center, text(size: 13pt, fill: MUTED)[both can use Transformer representations; their probability *factorizations* differ])

// ═══════════════════════════ PART V — Wrap up ═══════════════════════════
= Wrap up

== The systems bridge #D

Every extra sampling step calls a *large* denoiser again:
#pause
$ N #text[ steps] quad times quad #text[one U-Net forward pass] quad = quad #text[the dominant cost] $
#pause
#result[efficient inference is not an afterthought — it changes *which* diffusion applications are practical]

== Retrieval questions #Q

#pause
+ Why is a VAE useful even though diffusion already provides the generative model?
#pause
+ Which tensors supply cross-attention *queries*, *keys*, and *values*?
#pause
+ What does classifier-free guidance *extrapolate*?
#pause
+ Why is a *seed* useful for a scientific demonstration?

== Retrieval questions, extended #Q

#pause
+ Why run diffusion in a *latent* space?
#pause
+ How does a U-Net combine *global context* and *detail*?
#pause
+ What does a cross-attention *query* represent in this setting?
#pause
+ Which input controls composition most directly: a *prompt* or a *pose map*?

== Student takeaways #V

#pause
+ *Latent diffusion* cuts the cost of iterative denoising by working in a compact VAE space.
#pause
+ *Cross-attention* lets spatial denoising features retrieve relevant text information.
#pause
+ *Classifier-free guidance* trades diversity for stronger conditional adherence.
#pause
+ *Editing and control* are variations on conditioning the same reverse-denoising process.

== Closing line #V

#result[diffusion is *controllable denoising* in a learned representation space]
#pause
#align(center, text(size: 16pt, fill: MUTED)[a denoiser called *many* times $=>$ memory movement, caching, compression, and scheduling matter])
#pause
#align(center, text(size: 16pt)[Next (L23): why *efficient inference* decides what is actually usable.])

#focus-slide[
  Every input — noise level, text, image, mask, control — enters the *same* denoising loop.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Efficient Inference* $->$ memory movement, caching, and compression.
]
