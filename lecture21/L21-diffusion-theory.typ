// Diffusion Models — Theory
// Lecture 21 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture21/L21-diffusion-theory.typ /tmp/L21.pdf
//   typst compile --root . --input handout=true lecture21/L21-diffusion-theory.typ /tmp/L21h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.
// Quantitative figures: python3 lecture21/diagrams/l21_figs.py

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Diffusion Models: Theory],
  subtitle: [From a known noising process back to data — predict the noise, then reverse it],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let cream = rgb("#EFEEEB")
#let rect = fletcher.shapes.rect

// ═══════════════════════════ fletcher diagrams (native, vector) ═══════════════════════════

// ── (A) the forward + reverse Markov chain: THE picture of a diffusion model ──
#let chainboth = align(center, diagram(spacing: (20mm, 13mm), {
  let lbls = ($x_0$, $x_1$, $dots.c$, $x_(T-1)$, $x_T$)
  for (i, l) in lbls.enumerate() {
    if i == 2 { node((i, 0), text(size: 13pt, l), stroke: none, fill: none) }
    else {
      let f = if i == 0 { TEAL.lighten(72%) } else if i == 4 { ACC.lighten(70%) } else { cream }
      node((i, 0), l, radius: 7mm, fill: f, stroke: 0.9pt + INK)
    }
  }
  for i in range(4) {
    edge((i, 0), (i + 1, 0), "-|>", bend: 34deg, stroke: 1.3pt + ACC)      // forward (top)
    edge((i + 1, 0), (i, 0), "-|>", bend: 34deg, stroke: 1.3pt + GREEN)    // reverse (bottom)
  }
  node((2, -1.25), text(size: 10.5pt, fill: ACC)[*forward* (fixed): add noise  $q(x_t | x_(t-1))$], stroke: none, fill: none)
  node((2, 1.25),  text(size: 10.5pt, fill: GREEN)[*reverse* (learned): denoise  $p_theta(x_(t-1) | x_t)$], stroke: none, fill: none)
}))

// ── (B) forward chain only, with the factorized product ──
#let fwdchain = align(center, diagram(spacing: (18mm, 10mm), {
  let lbls = ($x_0$, $x_1$, $x_2$, $dots.c$, $x_T$)
  for (i, l) in lbls.enumerate() {
    if i == 3 { node((i, 0), text(size: 13pt, l), stroke: none, fill: none) }
    else {
      let f = if i == 0 { TEAL.lighten(72%) } else if i == 4 { ACC.lighten(70%) } else { cream }
      node((i, 0), l, radius: 7mm, fill: f, stroke: 0.9pt + INK)
    }
  }
  for i in range(4) {
    edge((i, 0), (i + 1, 0), "-|>", bend: 26deg, stroke: 1.2pt + ACC,
      label: if i == 0 { text(size: 9.5pt, fill: ACC)[$q(x_t | x_(t-1))$] } else { none }, label-side: left)
  }
}))

// ── (C) reverse chain only ──
#let revchain = align(center, diagram(spacing: (18mm, 10mm), {
  let lbls = ($x_0$, $x_1$, $x_2$, $dots.c$, $x_T$)
  for (i, l) in lbls.enumerate() {
    if i == 3 { node((i, 0), text(size: 13pt, l), stroke: none, fill: none) }
    else {
      let f = if i == 0 { TEAL.lighten(72%) } else if i == 4 { ACC.lighten(70%) } else { cream }
      node((i, 0), l, radius: 7mm, fill: f, stroke: 0.9pt + INK)
    }
  }
  for i in range(4) {
    edge((i + 1, 0), (i, 0), "-|>", bend: 26deg, stroke: 1.2pt + GREEN,
      label: if i == 3 { text(size: 9.5pt, fill: GREEN)[$p_theta(x_(t-1) | x_t)$] } else { none }, label-side: left)
  }
}))

// ── (D) the noise-prediction network: (x_t, t) -> denoiser (+ time embed) -> eps_theta ──
#let noisenet = align(center, diagram(spacing: (14mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0.7), $x_t$, radius: 6.5mm, fill: cream)
  node((0, -0.9), $t$, radius: 6mm, fill: BLUE.lighten(80%), stroke: 0.9pt + BLUE)
  node((1.6, -0.9), align(center, text(size: 9.5pt)[sinusoidal \ time embedding]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: BLUE.lighten(88%), stroke: 0.9pt + BLUE)
  node((3.2, 0), align(center, text(size: 10.5pt)[denoiser \ (U-Net)]), shape: rect, corner-radius: 5pt, inset: 10pt, fill: cream, stroke: 1.2pt + INK)
  node((5.0, 0), $epsilon_theta (x_t, t)$, shape: rect, corner-radius: 4pt, inset: 8pt, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  edge((0, 0.7), (3.2, 0), "-|>", stroke: 1pt + INK)
  edge((0, -0.9), (1.6, -0.9), "-|>", stroke: 0.8pt + BLUE)
  edge((1.6, -0.9), (3.2, 0), "-|>", stroke: 0.8pt + BLUE)
  edge((3.2, 0), (5.0, 0), "-|>", stroke: 1.1pt + GREEN)
}))

// ── (E) training flow: sample -> build x_t -> predict eps -> MSE ──
#let trainflow = align(center, diagram(spacing: (9mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let bx(x, lbl, c, f) = node((x, 0), align(center, text(size: 9pt, lbl)), shape: rect, corner-radius: 4pt, inset: 6pt, stroke: 0.9pt + c, fill: f)
  bx(0, [sample data \ $x_0 tilde cal(D)$], TEAL, TEAL.lighten(82%))
  bx(1, [sample step \ $t tilde {1, dots, T}$], BLUE, BLUE.lighten(86%))
  bx(2, [sample noise \ $epsilon tilde cal(N)(0, I)$], ACC, ACC.lighten(84%))
  bx(3, [build $x_t$ \ (closed form)], INK, cream)
  bx(4, [predict \ $epsilon_theta (x_t, t)$], GREEN, GREEN.lighten(82%))
  node((5, 0), align(center, text(size: 9pt)[loss \ $norm(epsilon - epsilon_theta)^2$]), shape: rect, corner-radius: 4pt, inset: 6pt, fill: RED.lighten(85%), stroke: 0.9pt + RED)
  for a in range(5) { edge((a, 0), (a + 1, 0), "-|>", stroke: 0.7pt + MUTED) }
}))

// ── (F) sampling loop: x_T ~ N(0,I) -> [predict eps, step] xT times -> x_0 ──
#let sampleflow = align(center, diagram(spacing: (15mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), align(center, text(size: 9.5pt)[$x_T tilde cal(N)(0, I)$]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  node((1.75, 0), align(center, text(size: 9.5pt)[predict \ $epsilon_theta (x_t, t)$]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: cream, stroke: 1pt + INK)
  node((3.5, 0), align(center, text(size: 9.5pt)[reverse step \ $-> x_(t-1)$]), shape: rect, corner-radius: 4pt, inset: 7pt, fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN)
  node((5.1, 0), $x_0$, radius: 6.5mm, fill: TEAL.lighten(72%), stroke: 0.9pt + TEAL)
  edge((0, 0), (1.75, 0), "-|>", stroke: 0.8pt + MUTED)
  edge((1.75, 0), (3.5, 0), "-|>", stroke: 1pt + INK)
  edge((3.5, 0), (5.1, 0), "-|>", stroke: 0.8pt + MUTED, label: text(size: 8.5pt, fill: MUTED)[at $t = 1$], label-side: right)
  edge((3.5, 0), (1.75, 0), "-|>", bend: 42deg, stroke: 1pt + BLUE,
    label: text(size: 9pt, fill: BLUE)[repeat $T$ times], label-side: left)
}))

// ── (G) reverse-mean intuition: what goes into one reverse step ──
#let revmean = align(center, diagram(spacing: (13mm, 6.5mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let inp(y, lbl, c, f) = node((0, y), align(center, text(size: 9.5pt, lbl)), shape: rect, corner-radius: 4pt, inset: 6pt, stroke: 0.9pt + c, fill: f)
  inp(1.5,  [noisy state  $x_t$], INK, cream)
  inp(0.5,  [predicted noise  $epsilon_theta (x_t, t)$], GREEN, GREEN.lighten(82%))
  inp(-0.5, [schedule coeffs  $alpha_t, overline(alpha)_t$], BLUE, BLUE.lighten(86%))
  inp(-1.5, [fresh noise  $z tilde cal(N)(0, I)$], ACC, ACC.lighten(84%))
  node((2.5, 0), align(center, text(size: 10.5pt)[reverse \ update]), shape: rect, corner-radius: 5pt, inset: 9pt, fill: cream, stroke: 1.2pt + INK)
  node((4.1, 0), $x_(t-1)$, radius: 7mm, fill: TEAL.lighten(72%), stroke: 0.9pt + TEAL)
  for y in (1.5, 0.5, -0.5, -1.5) { edge((0, y), (2.5, 0), "-|>", stroke: 0.7pt + MUTED) }
  edge((2.5, 0), (4.1, 0), "-|>", stroke: 1.1pt + INK)
}))

// ── (H) U-Net denoiser backbone: encoder / bottleneck / decoder + skips + time ──
#let unet = align(center, diagram(spacing: (10mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let enc(pos, lbl) = node(pos, align(center, text(size: 8.5pt, lbl)), shape: rect, corner-radius: 4pt, inset: 5pt, fill: TEAL.lighten(80%), stroke: 0.9pt + TEAL)
  let dec(pos, lbl) = node(pos, align(center, text(size: 8.5pt, lbl)), shape: rect, corner-radius: 4pt, inset: 5pt, fill: ACC.lighten(80%), stroke: 0.9pt + ACC)
  node((0, 2.1), $x_t$, radius: 5.5mm, fill: cream)
  enc((1, 1.5), [down 1])
  enc((2, 0.6), [down 2])
  enc((3, -0.3), [down 3])
  node((4, -1.1), align(center, text(size: 8.5pt)[bottleneck]), shape: rect, corner-radius: 4pt, inset: 5pt, fill: INK.lighten(6%), stroke: 0.9pt + INK)
  dec((5, -0.3), [up 3])
  dec((6, 0.6), [up 2])
  dec((7, 1.5), [up 1])
  node((8, 2.1), $epsilon_theta$, radius: 5.5mm, fill: GREEN.lighten(78%), stroke: 0.9pt + GREEN)
  // main path
  edge((0, 2.1), (1, 1.5), "-|>", stroke: 0.8pt + INK)
  edge((1, 1.5), (2, 0.6), "-|>", stroke: 0.8pt + INK)
  edge((2, 0.6), (3, -0.3), "-|>", stroke: 0.8pt + INK)
  edge((3, -0.3), (4, -1.1), "-|>", stroke: 0.8pt + INK)
  edge((4, -1.1), (5, -0.3), "-|>", stroke: 0.8pt + INK)
  edge((5, -0.3), (6, 0.6), "-|>", stroke: 0.8pt + INK)
  edge((6, 0.6), (7, 1.5), "-|>", stroke: 0.8pt + INK)
  edge((7, 1.5), (8, 2.1), "-|>", stroke: 0.9pt + GREEN)
  // skip connections (dashed, same resolution)
  edge((1, 1.5), (7, 1.5), "--", stroke: 1pt + MUTED, label: text(size: 8pt, fill: MUTED)[skip], label-side: right)
  edge((2, 0.6), (6, 0.6), "--", stroke: 1pt + MUTED)
  edge((3, -0.3), (5, -0.3), "--", stroke: 1pt + MUTED)
  // time embedding injected into every block
  node((4, 2.3), text(size: 8.5pt, fill: BLUE)[time embed #text(font: "IBM Plex Mono")[emb(t)]], shape: rect, corner-radius: 4pt, inset: 5pt, fill: BLUE.lighten(88%), stroke: 0.9pt + BLUE)
  for p in ((2, 0.6), (4, -1.1), (6, 0.6)) { edge((4, 2.3), p, "-|>", stroke: 0.6pt + BLUE) }
}))

#title-slide()

// ═══════════════════════════ PART I — the generative problem becomes denoising ═══════════════════════════
= The generative problem becomes denoising

== Why a new family of generative models? #Q

We have met two ways to generate data:
#pause
#two(
  notebox[*VAE* (L19) — organize a latent space, decode a latent in *one* pass. Trained by a likelihood bound.],
  notebox[*GAN* (L20) — learn a generator against an *adversarial critic*. One pass, but unstable minimax training.],
)
#pause
#result[*Diffusion* starts from a *corruption process we design* — then generates by iterating a learned reverse process]
#pause
#align(center, text(size: 16pt, fill: MUTED)[the key move: if we *know* how to add noise, we can train a network to *remove* it])

== The one idea of this lecture

#align(center, text(size: 22pt, fill: INK)[
  a diffusion model gradually turns *data* into *Gaussian noise*, \
  then learns to run that sequence *backwards*, one small denoising step at a time.
])
#pause
#v(6pt)
#chainboth
#pause
#align(center, text(size: 15pt, fill: MUTED)[forward is *fixed and known*; only the *reverse* transitions are *learned*])

== Visual hook: the noising filmstrip #V

Take a data point $x_0$ and add a little Gaussian noise, over and over:
#pause
#fig("/lecture21/figures/forward_filmstrip.svg", w: 92%)
#pause
#align(center, text(size: 15pt, fill: MUTED)[structure dissolves smoothly; by $t = T$ the sample is *indistinguishable from pure noise*])

== The forward process #D

Each forward step is a *fixed* Gaussian that shrinks the signal and injects noise:
#pause
$ q(x_t | x_(t-1)) = cal(N)(sqrt(1 - beta_t) thin x_(t-1), thin beta_t I) $
#pause
- $beta_t in (0, 1)$ is a small *variance schedule* — how much noise this step adds;
- nothing here is learned: the forward chain is *chosen in advance*.
#pause
#align(center, text(size: 16pt, fill: MUTED)[$sqrt(1 - beta_t)$ *scales down* the previous sample; $beta_t I$ *adds* fresh noise])

== A notation that makes the algebra clean #D

Define, for each step and cumulatively:
#pause
$ alpha_t = 1 - beta_t, quad quad overline(alpha)_t = product_(s=1)^t alpha_s $
#pause
- $alpha_t$ is the *fraction of variance kept* in one step;
- $overline(alpha)_t$ is the *fraction of the original signal kept* after $t$ steps.
#pause
#align(center, text(size: 16pt, fill: MUTED)[$overline(alpha)_t$ falls from $approx 1$ (early) toward $approx 0$ (late) — it is the whole story of the schedule])

== One-step intuition #D

Read the forward step as two operations:
#pause
$ x_t = underbrace(sqrt(alpha_t) thin x_(t-1), "shrink the signal") + underbrace(sqrt(beta_t) thin epsilon, "add a little noise"), quad epsilon tilde cal(N)(0, I) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[each step barely changes the sample — many *tiny* steps accumulate into total corruption])

== The closed form that makes it all practical #D

Because Gaussians compose, we can jump straight to *any* noise level:
#pause
$ q(x_t | x_0) = cal(N)(sqrt(overline(alpha)_t) thin x_0, thin (1 - overline(alpha)_t) I) $
#pause
#result[no need to simulate steps $1, 2, dots, t-1$ — one formula reaches $x_t$ directly]
#pause
#align(center, text(size: 16pt, fill: MUTED)[mean $sqrt(overline(alpha)_t) thin x_0$ is the *surviving signal*; variance $(1 - overline(alpha)_t) I$ is the *accumulated noise*])

== Sampling a noisy training input #D

The closed form is a one-line recipe with the *reparameterization trick*:
#pause
$ x_t = sqrt(overline(alpha)_t) thin x_0 + sqrt(1 - overline(alpha)_t) thin epsilon, quad quad epsilon tilde cal(N)(0, I) $
#pause
- draw a clean $x_0$, draw a noise $epsilon$, pick a timestep $t$;
- the pair $(x_t, epsilon)$ is a *ready-made supervised example* — input $x_t$, target $epsilon$.
#pause
#align(center, text(size: 16pt, fill: MUTED)[we *know the noise* because we drew it ourselves — this is the crux of the whole method])

== Board calculation: build $x_t$ by hand #D

Let $x_0 = 1$, $thin overline(alpha)_t = 0.64$, and a drawn noise $epsilon = -0.5$.
#pause
$ x_t = sqrt(0.64) thin (1) + sqrt(1 - 0.64) thin (-0.5) $
#pause
$ x_t = underbrace(0.8, "signal") + underbrace(0.6 dot (-0.5), "noise") = 0.8 - 0.3 = #text(fill: ACC, weight: 700)[$0.5$] $
#pause
#two(
  notebox[*signal* $sqrt(overline(alpha)_t) thin x_0 = 0.8$ — most of the clean value survives here],
  notebox[*noise* $sqrt(1 - overline(alpha)_t) thin epsilon = -0.3$ — a moderate perturbation at this $t$],
)

== At the end of the chain #D

As $t -> T$, the product $overline(alpha)_t$ collapses toward zero:
#pause
$ overline(alpha)_T approx 0 quad ==> quad q(x_T | x_0) approx cal(N)(0, I) $
#pause
#result[$x_T$ is (almost) *standard Gaussian noise* — trivial to sample from]
#pause
#align(center, text(size: 16pt, fill: MUTED)[so generation can *start from pure noise* and only needs to learn the way *back*])

// ═══════════════════════════ PART II — learning the reverse process ═══════════════════════════
= Learning the reverse process

== The direction that looks impossible #Q

From one noisy image, *many* clean images are plausible:
#pause
#align(center, text(size: 17pt, fill: INK)[given a noisy $x_t$, which $x_(t-1)$ did it come from?])
#pause
#alertbox[The reverse of a random corruption is *not a function* — it is a *distribution*. There is genuine uncertainty about what was destroyed.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[so we model the reverse step *probabilistically*, and learn it from data])

== A parameterized reverse step #D

Model each reverse transition as a Gaussian whose mean (and variance) the network predicts:
#pause
$ p_theta(x_(t-1) | x_t) = cal(N)(mu_theta(x_t, t), thin Sigma_theta(x_t, t)) $
#pause
- if steps are small, the *true* reverse step is *approximately Gaussian* too — so this form is well-motivated;
- all the difficulty is pushed into predicting the *mean* $mu_theta$.
#pause
#align(center, text(size: 16pt, fill: MUTED)[the network must turn a slightly-noisier $x_t$ into a slightly-cleaner $x_(t-1)$])

== The task we can actually supervise #D

During training we *know both* $x_0$ and the noise $epsilon$ we injected. So flip the problem:
#pause
#result[don't predict $x_(t-1)$ directly — predict the *noise* $epsilon$ that was added]
#pause
#fig("/lecture21/figures/predict_noise.svg", w: 62%)

== The noise-prediction network #V

A single network $epsilon_theta(x_t, t)$ estimates the injected noise from the noisy input and the timestep:
#pause
#noisenet
#pause
#align(center, text(size: 15pt, fill: MUTED)[it *must* see $t$: "remove a little noise" and "remove almost all noise" are different jobs])

== The practical DDPM objective #D

Train by simple regression of predicted noise onto true noise:
#pause
$ L_"simple" = EE_(x_0, thin epsilon, thin t) [ thin norm(epsilon - epsilon_theta(x_t, t))_2^2 thin ] $
#pause
- expectation over data $x_0$, injected noise $epsilon$, and timestep $t$;
- this is *supervised regression* with *self-generated* targets — no labels, no adversary.
#pause
#align(center, text(size: 16pt, fill: MUTED)[the whole of DDPM training is: *sample noise, predict it, minimize the MSE*])

== A useful reframe: self-supervised denoising #D

Every clean example is an *infinite* source of training pairs:
#pause
#two(
  notebox[*one clean $x_0$* $-> $ pick any $t$, draw any $epsilon$ $-> $ a fresh noisy input $x_t$.],
  notebox[*exact target* — the noise $epsilon$ is *known by construction*, so the label is free and precise.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[diffusion is denoising self-supervision at *every* noise level at once])

== From predicted noise to a clean estimate #D

Invert the closed form: if we can guess $epsilon$, we can guess the clean signal.
#pause
$ hat(x)_0 = (x_t - sqrt(1 - overline(alpha)_t) thin epsilon_theta(x_t, t)) / sqrt(overline(alpha)_t) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[predicting the noise and predicting the clean image are *two views of the same quantity*])

== The reverse sampling loop #V

Start at pure noise; peel off a little noise at a time until a sample emerges:
#pause
#sampleflow
#pause
#codebox(size: 14pt)[```
x_T ~ N(0, I)                          # start from pure Gaussian noise
for t = T, ..., 1:
    eps_hat = eps_theta(x_t, t)        # predict the noise
    x_{t-1} = reverse_step(x_t, eps_hat, t)   # sample / estimate x_{t-1}
return x_0
```]

== Interactive: forward and reverse #I

#interbox(link-to: IA + "diffusion-denoise")[
  Slide the noise level to watch $x_t = sqrt(overline(alpha)_t) thin x_0 + sqrt(1 - overline(alpha)_t) thin epsilon$ corrupt an example, then run the reverse chain step-by-step and see a sample condense out of noise.
]

== Why generation is slow #D

The cost of quality is *many* forward passes:
#pause
#two(
  notebox[*GAN* — one $z$, one generator pass, one image. Fast.],
  notebox[*Diffusion* — the *same* denoiser is called *hundreds to thousands* of times per sample.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[stable training and broad coverage — paid for in *inference* time (L22 attacks this)])

// ═══════════════════════════ PART III — interpreting the geometry ═══════════════════════════
= Interpreting the geometry

== Denoising happens at many scales #D

Different reverse steps do qualitatively different work:
#pause
- *early* reverse steps (near pure noise) — recover *global layout* and coarse structure;
- *late* reverse steps (nearly clean) — restore *fine texture* and detail.
#pause
#align(center, text(size: 16pt, fill: MUTED)[one network, but the *job changes* along the trajectory — hence the time conditioning])

== The noise schedule is a curriculum #D

Training touches *every* difficulty, from trivial to nearly hopeless:
#pause
#let _T = 1000
#let _bmin = 0.0001
#let _bmax = 0.02
#let _beta(s) = _bmin + (s - 1) / (_T - 1) * (_bmax - _bmin)
#let _abar(t) = if t == 0 { 1.0 } else { range(1, t + 1).map(s => 1.0 - _beta(s)).fold(1.0, (a, b) => a * b) }
#two(r: (1fr, 1fr),
  align(center, lines(
    ((1, _beta(1)), (_T, _beta(_T))),
    colors: (RED,),
    markers: false,
    x-label: [timestep $t$], y-label: [$beta_t$],
    title: [$beta_t$: $10^(-4) -> 0.02$ (linear)],
    size: (50mm, 38mm),
  )),
  align(center, lines(
    range(0, _T + 1, step: 40).map(t => (t, _abar(t))),
    colors: (TEAL,),
    markers: false,
    x-label: [timestep $t$], y-label: [$overline(alpha)_t$],
    title: [$overline(alpha)_t$: signal retained],
    size: (50mm, 38mm),
  )),
)
#pause
#align(center, text(size: 15pt, fill: MUTED)[$beta_t$ (how much noise each step adds) sets $overline(alpha)_t$ (how much signal survives)])

== The score connection — one conceptual slide #D

For a noisy distribution $p_t$, the *score* is the gradient of its log-density:
#pause
$ nabla_x log p_t (x) quad quad #text(size: 15pt, fill: MUTED)[(points toward *higher-density* regions)] $
#pause
#notebox[Predicting Gaussian noise is *equivalent* (up to a schedule factor) to estimating this score: $ epsilon_theta(x_t, t) approx -sqrt(1 - overline(alpha)_t) thin nabla_x log p_t (x_t). $ ]
#pause
#align(center, text(size: 15pt, fill: MUTED)[a bridge to *score-based* models — we stop here, no SDE derivation])

== What the network actually learns #Q

Not a lookup table of clean images:
#pause
#result[it learns, from any noisy point, a *local direction* toward where the data density is high]
#pause
#align(center, text(size: 16pt, fill: MUTED)[a *vector field of denoising directions* — generalized across inputs, timesteps, and conditions])

== Conditional generation, previewed #D

Feed the denoiser extra information $c$ so the trajectory follows a desired target:
#pause
- a *class label*; a *text embedding*; a *segmentation mask*; a *low-resolution image*;
#pause
$ epsilon_theta(x_t, t, c) quad quad #text(size: 15pt, fill: MUTED)[(same network, now told what to aim for)] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the denoising *question* is unchanged — only the *conditioning* differs (full story in L22)])

== Three generative families so far #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 8pt), align: (left, left, left),
  table.header([*Family*], [*Training signal*], [*Sampling*]),
  [VAE], [likelihood (ELBO) bound], [one latent decode],
  [GAN], [adversarial critic], [one generator pass],
  [Diffusion], [predict *known* injected noise], [repeated denoising],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[diffusion trades a *hard sampler* for an *easy, stable* learning signal])

== What diffusion gains and what it pays #D

#pause
#two(
  notebox[#text(fill: GREEN)[*gains*] — stable supervised-style training; broad mode coverage; flexible conditioning.],
  alertbox[*pays* — many inference steps; substantial compute per sample.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[the central trade of the method: *training ease* against *sampling cost*])

== Knowledge check #Q

#align(center, text(size: 18pt, fill: INK)[
  If $t$ is *large* (a very noisy input), should the model rely more on \
  the *noisy input* or on its *learned data prior*? Why?
])
#pause
#v(4pt)
#notebox[At large $t$, $x_t$ carries *almost no signal* ($overline(alpha)_t approx 0$). The denoiser must lean on the *prior* it learned — the input mostly tells it *which* broad direction, not the fine details.]

== Closing bridge to practice #D

Theory says: *predict the noise, repeatedly*.
#pause
#align(center, text(size: 17pt, fill: INK)[Practice then asks three questions:])
#pause
+ *what architecture* predicts $epsilon_theta$? (U-Net / diffusion Transformer)
+ *how does text steer it*? (conditioning + guidance)
+ *can we avoid thousands of steps*? (fast samplers, latent space)
#pause
#align(center, text(size: 15pt, fill: MUTED)[L22 answers all three])

// ═══════════════════════════ PART IV — diffusion theory in depth ═══════════════════════════
= Diffusion theory in depth

== The forward chain factorizes #D

The forward process is a *Markov chain* — each step depends only on the last:
#pause
$ q(x_(1:T) | x_0) = product_(t=1)^T q(x_t | x_(t-1)) $
#pause
#fwdchain
#pause
#align(center, text(size: 15pt, fill: MUTED)[a simple, fixed structure — even when the data distribution is wildly complicated])

== The reverse model factorizes too #D

Sampling runs the learned chain from the easy prior downward:
#pause
$ p_theta(x_(0:T)) = p(x_T) product_(t=1)^T p_theta(x_(t-1) | x_t), quad quad p(x_T) = cal(N)(0, I) $
#pause
#revchain
#pause
#align(center, text(size: 15pt, fill: MUTED)[start at $cal(N)(0, I)$, apply $T$ learned reverse transitions, arrive at $x_0$])

== Why the closed form matters computationally #D

Without it, one training pair at timestep $t$ would need $t$ *sequential* noising steps:
#pause
#two(
  alertbox[*naive* — to get $x_t$, simulate $x_1 -> x_2 -> dots -> x_t$. Costly, and different for every $t$.],
  notebox[*closed form* — $x_t = sqrt(overline(alpha)_t) x_0 + sqrt(1 - overline(alpha)_t) epsilon$ in *one* operation, for *any* $t$.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[one random $t$ gives one training example in a *single* step — this is what makes training feasible])

== Training touches every noise difficulty #V

#trainflow
#pause
The five operations of one training step:
#pause
#align(center, text(size: 15pt)[
  sample $x_0$  $-> $  sample $t$  $-> $  sample $epsilon$  $-> $  build $x_t$  $-> $  predict $epsilon_theta(x_t, t)$
])
#pause
#align(center, text(size: 15pt, fill: MUTED)[the same data point spawns a *continuum* of self-supervised denoising tasks])

== Reading the noise schedule #V

Plot the signal and noise coefficients against $t$:
#pause
#fig("/lecture21/figures/signal_noise.svg", w: 62%)
#pause
#align(center, text(size: 15pt, fill: MUTED)[$sqrt(overline(alpha)_t)$ falls from $1$; $sqrt(1 - overline(alpha)_t)$ rises to $1$ — a steadily *falling signal-to-noise ratio*])

== Different timesteps are different tasks #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left, left),
  table.header([*Timestep*], [*Input appearance*], [*Main denoising demand*]),
  [small $t$], [mostly clean], [remove fine perturbations],
  [middle $t$], [partly structured], [recover objects / layout],
  [large $t$], [near Gaussian noise], [rely strongly on the learned prior],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[the time embedding tells the network *which* of these regimes it is in])

== Why predict noise rather than the image? #D

Several equivalent parameterizations exist:
#pause
$ epsilon quad quad (#text[noise]), quad quad x_0 quad quad (#text[clean image]), quad quad v quad quad (#text[velocity]) $
#pause
- *noise prediction* gives a target with a *known, unit-scale* Gaussian across all $t$ — nicely conditioned;
- practical systems may use $x_0$ or $v$; the underlying diffusion principle is *unchanged*.
#pause
#align(center, text(size: 16pt, fill: MUTED)[the parameterization is a *modeling choice*, not the essence of diffusion])

== MSE is a conditional regression loss #D

From an ambiguous $x_t$, the network cannot recover the *exact* unknowable noise draw:
#pause
$ epsilon_theta(x_t, t) quad -> quad EE[thin epsilon | x_t, t thin] $
#pause
#notebox[Minimizing MSE makes $epsilon_theta$ approach the *conditional mean* of the noise given $x_t$. It blends the evidence in $x_t$ with the *data prior* — exactly what a good denoiser should do.]

== Time must enter the network #V

The *same* tensor can occur at many noise levels — the model needs to know which:
#pause
#noisenet
#pause
#align(center, text(size: 15pt, fill: MUTED)[$t -> $ sinusoidal / learned embedding $-> $ injected into the denoiser blocks])

== Reverse-mean intuition #V

One reverse step *combines* four ingredients — no need to memorize the coefficients:
#pause
#revmean
#pause
#align(center, text(size: 15pt, fill: MUTED)[current state, predicted noise, schedule, and (optionally) fresh randomness $-> x_(t-1)$])

== Stochastic vs deterministic sampling #D

Once the denoiser is trained, *how* you step back is a separate choice:
#pause
#two(
  notebox[*stochastic* (e.g. DDPM) — add fresh noise $z$ at each reverse step; more *diverse* samples.],
  notebox[*deterministic* (e.g. DDIM) — no added noise once $x_T$ is fixed; *reproducible*, often *fewer* steps.],
)
#pause
#align(center, text(size: 15pt, fill: MUTED)[this changes diversity, reproducibility, and speed — *not* the learned denoiser itself])

== One sample, many intermediate states #V

Run the reverse chain and watch a value settle onto a data mode:
#pause
#fig("/lecture21/figures/denoise_traj.svg", w: 66%)
#pause
#align(center, text(size: 15pt, fill: MUTED)[global choice ("which mode?") is made *early*; fine position is refined *late* — an observation, not a law])

== Relation to denoising autoencoders #D

A denoising autoencoder learns one corruption level:
#pause
$ #text[corrupted input] quad -> quad #text[clean input] $
#pause
#result[diffusion is a DAE *at every calibrated noise level*, plus a *generative* reverse chain]
#pause
#align(center, text(size: 16pt, fill: MUTED)[the extra ingredients: a *sequence* of levels, *time conditioning*, and *sampling from noise*])

== The variational perspective, placed #D

The original DDPM objective comes from a *variational bound* on the data likelihood:
#pause
$ -log p_theta(x_0) <= EE_q [ thin dots thin ] quad quad #text(size: 14pt, fill: MUTED)[(a sum of per-step KL terms — like L19's ELBO)] $
#pause
#notebox[The simple noise-prediction MSE $L_"simple"$ is a *reweighted simplification* of that likelihood bound — it drops constants and per-timestep weights that hurt sample quality.]
#pause
#align(center, text(size: 15pt, fill: MUTED)[connect to the ELBO from L19 — we do *not* re-derive the whole bound here])

== Score intuition, revisited #V

For a noisy point $x_t$, the score points toward denser regions of $p_t$:
#pause
#fig("/lecture21/figures/reverse_density.svg", w: 52%)
#pause
#align(center, text(size: 15pt, fill: MUTED)[the reverse process rides these directions from one blurry blob back to the data modes])

== Conditioning changes the reverse distribution #D

A condition $c$ reshapes *which* reverse paths are favored:
#pause
$ p_theta(x_(t-1) | x_t, c) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[text, class labels, masks, low-res images — all change the *same* conditional denoising question])
#pause
#interbox(link-to: IA + "classifier-guidance")[A conditioning widget: watch the reverse trajectory bend toward a chosen class.]

== Classifier guidance, previewed #D

An external classifier can push the trajectory toward a target class:
#pause
$ nabla_(x_t) log p(c | x_t) quad quad #text(size: 15pt, fill: MUTED)[(gradient that says "look more like class $c$")] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[add this to the score to *steer* sampling — L22 teaches the practical *classifier-free* version])

== Architectural preview: the denoiser #V

The loss says *what* to predict; the architecture says *how* to compute it:
#pause
#unet
#pause
#align(center, text(size: 15pt, fill: MUTED)[a *U-Net* (or diffusion Transformer): multi-scale, skip connections, time embedding in every block])

== Why diffusion training is comparatively stable #D

Every example has a *fixed, known* target:
#pause
$ #text[target] = epsilon quad quad #text(size: 15pt, fill: MUTED)[(the noise we drew — it never moves)] $
#pause
#alertbox[Unlike a GAN, *no learned opponent* shifts the target between batches — there is no minimax game to diverge.]
#pause
#align(center, text(size: 15pt, fill: MUTED)[optimization is not *trivial*, but the central instability of adversarial training is *gone*])

== Why diffusion sampling is expensive #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 14pt, y: 8pt), align: (left, left),
  table.header([*Model family*], [*Typical generation structure*]),
  [VAE], [one decode],
  [GAN], [one generator pass],
  [diffusion], [*repeated* denoiser calls ($T$ of them)],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[the model trades *inference work* for a *tractable, stable* training signal])

== Common misconceptions #Q

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 7pt), align: (left, left),
  table.header([*Misconception*], [*Correction*]),
  [diffusion memorizes a reverse "noise table"], [a network *generalizes* denoising across inputs and conditions],
  [the forward process is learned], [in standard DDPMs it is *chosen and fixed*],
  [the network only predicts the clean image], [noise, clean image, or velocity can be parameterized],
  [more steps always mean a better image], [schedules and samplers set a quality / latency *trade-off*],
))

== Retrieval questions #Q

#v(4pt)
+ Why can $x_t$ be sampled *directly* from $x_0$?
#pause
+ What target is available during diffusion training *without labels*?
#pause
+ Why *must* the denoiser know $t$?
#pause
+ Why is generation *slower* than GAN sampling?
#pause
#align(center, text(size: 15pt, fill: MUTED)[if all four are quick to answer, the core of the lecture is secure])

== The $beta_t$ schedule is a design choice #D

The schedule sets the *sequence of learning problems*:
#pause
#two(
  alertbox[*too little* noise — $x_T$ stays structured; the prior $cal(N)(0, I)$ is a poor starting point.],
  alertbox[*too abrupt* noise — intermediate denoising tasks become needlessly hard.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[smooth schedules (linear, cosine) spread difficulty *evenly* across timesteps])

== Variance in reverse sampling #D

The learned reverse Gaussian has a mean *and* a variance:
#pause
$ p_theta(x_(t-1) | x_t) = cal(N)(mu_theta, thin Sigma_theta) $
#pause
- $Sigma_theta$ may be *fixed* to a schedule value or *learned*;
- it controls how much randomness each reverse step injects.
#pause
#align(center, text(size: 16pt, fill: MUTED)[variance is a *modeling knob*, not an arbitrary visual-effect dial])

== Loss weighting across timesteps #D

An "MSE loss" still hides modeling choices:
#pause
- not every $t$ need contribute *equally* to the objective;
- parameterization ($epsilon$ / $x_0$ / $v$) and schedule change *which* SNR regimes dominate learning.
#pause
#align(center, text(size: 16pt, fill: MUTED)[reweighting timesteps trades off *coarse structure* against *fine detail* quality])

== Data modalities beyond images #D

The same recipe travels — only the denoiser and conditioning change:
#pause
- *audio* waveforms or spectrograms;  *video* with temporal structure;
- *molecular* coordinates;  *trajectories* and actions;  continuous *latent* representations.
#pause
#align(center, text(size: 16pt, fill: MUTED)[for standard continuous-data DDPMs, forward = Gaussian corruption; the *architecture* must respect the data's structure])

== Why direct pixel diffusion is demanding #D

Noise is added independently to a *very high-dimensional* tensor:
#pause
#result[a good denoiser must fix texture, object identity, global layout, *and* condition alignment at once]
#pause
#align(center, text(size: 16pt, fill: MUTED)[this pressure motivates L22's *latent-space* diffusion and multi-scale architectures])

== A tiny one-dimensional thought experiment #V

Data lives near $-2$ and $+2$ on a line; at high noise the two modes *blend*:
#pause
#fig("/lecture21/figures/reverse_density.svg", w: 48%)
#pause
#align(center, text(size: 15pt, fill: MUTED)[the reverse model must *decide which mode* a noisy point should approach — generation is inherently probabilistic])

== Denoising is not a nearest-neighbour lookup #D

For a noisy point between modes, the model uses *learned structure*, not a database:
#pause
- the result may resemble *interpolation*, *recombination*, or a genuinely *new* sample;
#pause
#alertbox[Separately: *memorization* remains an empirical risk to *test for* on small or narrow datasets.]

== A diffusion evaluation lens #Q

No single grid or scalar settles quality — evaluate along five axes:
#pause
- *fidelity*;  *coverage / diversity*;  *conditioning adherence*;
- *sampling cost*;  *robustness and harmful failure modes*.
#pause
#align(center, text(size: 16pt, fill: MUTED)[FID and a caption match tell you *something* — never *everything*])

== Theory-to-practice checklist #V

Before reading any implementation, locate it with five questions:
#pause
+ data space — *pixels or latents*?
+ *denoiser architecture*?
+ *time / noise parameterization*?
+ *condition interface*?
+ *sampler* and step budget?
#pause
#align(center, text(size: 15pt, fill: MUTED)[these five coordinates place nearly *every* diffusion system in the course map])

== Common equations, one page #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 9pt), align: (left, left),
  table.header([*Idea*], [*Equation*]),
  [forward step], [$q(x_t | x_(t-1)) = cal(N)(sqrt(1 - beta_t) thin x_(t-1), thin beta_t I)$],
  [closed form], [$x_t = sqrt(overline(alpha)_t) thin x_0 + sqrt(1 - overline(alpha)_t) thin epsilon$],
  [training loss], [$L = EE norm(epsilon - epsilon_theta(x_t, t))_2^2$],
  [clean estimate], [$hat(x)_0 = (x_t - sqrt(1 - overline(alpha)_t) thin epsilon_theta) \/ sqrt(overline(alpha)_t)$],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[a final reference — *not* another derivation])

== Retrieval questions, extended #Q

#v(4pt)
+ Which part of diffusion is *fixed* and which part is *learned*?
#pause
+ How does the closed form turn one image into *many* training examples?
#pause
+ Why is the reverse direction *conditional and probabilistic*?
#pause
+ What trade-off does *iterative sampling* introduce?

== Next lecture #D

#align(center, text(size: 18pt, fill: INK)[
  noise-prediction objective
  #v(4pt)
  $+$  latent space, U-Net / Transformer, and conditioning
  #v(4pt)
  $=$  a *practical* diffusion system
])
#pause
#v(6pt)
#align(center, text(size: 16pt, fill: MUTED)[L22 builds that system component by component — including *Stable Diffusion*])

#focus-slide[
  If you know how to *add* noise, you can train a network to *remove* it.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Diffusion in practice* $-> $ latent diffusion, U-Nets, guidance, and Stable Diffusion.
]
