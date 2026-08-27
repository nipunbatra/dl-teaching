// Generalization and Regularization — Lecture 7 · ES 667 Deep Learning
// Compile from the repository root:
//   typst compile --root . --input handout=true lecture7/L7-regularization.typ slides-pdf/L7.pdf
//   typst compile --root . lecture7/L7-regularization.typ slides-pdf/L7-presentation.pdf
//
// Design: one seeded 10-point regression lab gives exact mathematics for
// diagnosis, weight decay, and early stopping. A real CIFAR-10 image pair then
// carries augmentation, MixUp/CutMix, dropout, label smoothing, and PyTorch.
// The companion lab lecture7/diagrams/verify_numbers.typ independently checks
// the computed regression numbers.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#import "../vendor/chalkdust/linalg/lib.typ" as la
#import "@preview/lilaq:0.4.0" as lq

#show: metropolis-deck.with(
  title: [Generalization and Regularization],
  subtitle: [Diagnose the gap · change one lever · measure again],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let NB = "https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L07/"
#let PT = "https://docs.pytorch.org/docs/stable/"
#let PTV = "https://docs.pytorch.org/vision/stable/"
#let MC = "https://proceedings.mlr.press/v48/gal16.html"

// ── semantic colour grammar, used in prose, tables, and every figure ──
// TEAL  = the data-fitting force (training loss, training curves)
// ACC   = the preference for simpler behaviour (the regularizer and its knob)
// BLUE  = validation evidence (the judge that sets every knob)
// RED   = memorization / the failing unregularized fit
// GREEN = the kept model (the cured fit, the restored checkpoint)
// INK   = data and ground truth (neutral)
#let fitc(b) = text(fill: TEAL, weight: 650, b)
#let knobc(b) = text(fill: ACC, weight: 650, b)
#let valc(b) = text(fill: BLUE, weight: 650, b)
#let failc(b) = text(fill: RED, weight: 650, b)
#let keepc(b) = text(fill: GREEN, weight: 650, b)

#let note(body, color: TEAL) = block(
  width: 100%, inset: (left: 10pt, right: 2pt, top: 4pt, bottom: 4pt),
  stroke: (left: 1.2pt + color), [#body],
)
#let warning(body) = note(body, color: RED)
#let punch(body) = align(center, text(size: 18pt, weight: 650)[#body])
#let caption(body) = align(center, text(size: 14.5pt, fill: MUTED)[#body])
#let hairline(title, body, color: TEAL) = block(width: 100%, inset: 2pt, [
  #text(size: 14pt, weight: 700, fill: color)[#title]
  #v(2pt)
  #line(length: 100%, stroke: 0.7pt + color)
  #v(5pt)
  #body
])
#let sw(color, body) = box(width: 12pt, height: 3pt, fill: color, radius: 1pt) + h(4pt) + body
#let semantic-legend = align(center, text(size: 13pt)[
  #sw(TEAL, [training / fit]) #h(13pt)
  #sw(ACC, [preference + knob]) #h(13pt)
  #sw(BLUE, [validation]) #h(13pt)
  #sw(RED, [memorization]) #h(13pt)
  #sw(GREEN, [kept model])
])
#let provenance = align(center, text(size: 11.5pt, weight: 650, fill: MUTED)[
  SYNTHETIC · COMPUTED — seeded 10-point teaching case; every curve and number on this slide is recomputed from it at compile time
])

// ═══════════════════ THE SPINE: one dataset for the whole lecture ═══════════════════
// 10 noisy measurements of a smooth truth, fit by the degree-9 polynomial
// feature map students know from ES 335. Everything downstream — the exact
// interpolant, the ridge family, the gram spectrum, closed-form GD iterates,
// the jitter-augmented refit — derives from these ten numbers.
#let SEED = 21
#let NTR = 10
#let DEG = 9
#let NOISE = 0.25
#let truth(x) = calc.sin(calc.pi * x)
#let xs = range(NTR).map(i => -1.0 + 2.0 * i / (NTR - 1))
#let ys = xs.enumerate().map(((i, x)) => truth(x) + NOISE * rnd.randn(SEED, i))
// validation = the 9 midpoints between training inputs, with fresh noise
#let xv = range(NTR - 1).map(i => (xs.at(i) + xs.at(i + 1)) / 2)
#let yv = xv.enumerate().map(((i, x)) => truth(x) + NOISE * rnd.randn(SEED + 100, i))
// sealed test = a dense, independent draw; it is not used by any selection rule
#let xt = range(101).map(i => -1.0 + 2.0 * i / 100.0)
#let yt = xt.enumerate().map(((i, x)) => truth(x) + NOISE * rnd.randn(SEED + 300, i))

#let featmap(x) = range(DEG + 1).map(k => if k == 0 { 1.0 } else { calc.pow(x, k) })
#let VD = xs.map(featmap)
#let polyval(w, x) = w.rev().fold(0.0, (s, c) => s * x + c)
#let mse(w, xa, ya) = xa.zip(ya).map(((x, y)) => calc.pow(polyval(w, x) - y, 2)).sum() / xa.len()
#let data-loss(w, xa, ya) = mse(w, xa, ya) / 2

// the exact interpolant: a square Vandermonde solve (10 equations, 10 weights)
#let w9 = la.solve(VD, ys)
// ridge family, in the convention
//   L_data = (1/2n)‖Xw−y‖²; J = L_data + (λ/2)‖w‖²  ⇒ X'X + nλI
#let ridge-fit(lm) = ml.linreg(VD, ys, ridge: NTR * lm)
#let lam-grid = (0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0)
#let lam-rows = lam-grid.map(lm => {
  let w = ridge-fit(lm)
  (lm, mse(w, xs, ys), mse(w, xv, yv), la.norm(w))
})
#let lam-star = lam-rows.sorted(key: r => r.at(2)).first().at(0)
#let w-star = ridge-fit(lam-star)

// gram spectrum + exact discrete-GD iterates from w = 0 (quadratic loss ⇒ closed form)
#let AG = la.mscale(la.matmul(la.transpose(VD), VD), 1.0 / NTR)
#let eig = la.eig-sym(AG, sweeps: 600)
#let evals = eig.at(0)
#let evecs = eig.at(1)
#let smax = calc.max(..evals)
#let eta-gd = 0.25 / smax
#let gd-coef = range(DEG + 1).map(j => la.dot(evecs.at(j), w9))
#let wt(t) = {
  let w = range(DEG + 1).map(_ => 0.0)
  for j in range(DEG + 1) {
    let fj = 1.0 - calc.pow(1.0 - eta-gd * evals.at(j), t)
    w = la.vadd(w, la.scale(evecs.at(j), fj * gd-coef.at(j)))
  }
  w
}
// checkpoints every half-decade; validation picks the survivor
#let ckpt-grid = (1, 3, 10, 30, 100, 300, 1000, 2000, 5000, 10000, 30000, 100000, 300000, 1000000, 3000000, 10000000)
#let ckpt-rows = ckpt-grid.map(t => (t, mse(wt(t), xs, ys), mse(wt(t), xv, yv)))
#let t-star = ckpt-rows.sorted(key: r => r.at(2)).first().at(0)
#let w-stop = wt(t-star)

// input-jitter augmentation: 8 jittered copies of every point, same label
#let KAUG = 8
#let JIT = 0.10
#let aug-pairs = range(NTR).map(i =>
  range(KAUG).map(k => (xs.at(i) + JIT * rnd.randn(SEED + 200, i * KAUG + k), ys.at(i)))
).flatten().chunks(2)
#let xa = aug-pairs.map(p => p.at(0))
#let ya = aug-pairs.map(p => p.at(1))
#let w-aug = ml.linreg(xa.map(featmap), ya, ridge: 0.0000000001)

// the second exact interpolant: narrow radial bumps through the same 10 points
#let ELL = 0.04
#let kfn(a, b) = calc.exp(-calc.pow(a - b, 2) / (2 * ELL * ELL))
#let alpha-rbf = la.solve(xs.map(a => xs.map(b => kfn(a, b))), ys)
#let rbfval(x) = range(NTR).map(j => alpha-rbf.at(j) * kfn(x, xs.at(j))).sum()

// number formatting that keeps trailing zeros ("1.70", "0.0000")
#let fnum(v, d: 2) = {
  let neg = v < 0
  let a = calc.abs(calc.round(v, digits: d))
  let ip = calc.floor(a)
  let fr = calc.round((a - ip) * calc.pow(10.0, d))
  let fs = str(int(fr))
  while fs.len() < d { fs = "0" + fs }
  (if neg { sym.minus } else { "" }) + str(int(ip)) + (if d > 0 { "." + fs } else { "" })
}

// ── the shared spine figure: data dots + any set of computed curves ──
#let XG = range(241).map(i => -1.0 + i / 120.0)
#let curve-of(w) = XG.map(x => polyval(w, x))
#let truth-curve = XG.map(truth)
#let truth-stroke = (paint: MUTED, thickness: 1.5pt, dash: "dashed")
#let spine-fig(
  curves: (),                    // ((y-array, stroke), …) drawn back to front
  show-train: true, show-val: false,
  width: 148mm, height: 54mm, ylim: (-2.7, 3.7),
) = align(center, {
  set text(size: 13pt)
  lq.diagram(
    width: width, height: height,
    xlim: (-1.06, 1.06), ylim: ylim, margin: 0%,
    xlabel: [$x$], ylabel: [$y$],
    grid: (stroke: 0.35pt + MUTED.transparentize(78%)),
    xaxis: (ticks: (-1, 0, 1), subticks: none, mirror: false),
    yaxis: (ticks: (-2, 0, 2), subticks: none, mirror: false),
    ..curves.map(c => lq.plot(XG, c.at(0), stroke: c.at(1), mark: none)),
    ..(if show-train { (lq.scatter(xs, ys, color: INK, size: 4.6pt),) } else { () }),
    ..(if show-val { (lq.scatter(xv, yv, color: BLUE, size: 4.6pt),) } else { () }),
  )
})

// ── the regularization map: one pipeline, five injection sites ──
// Re-drawn at every section start with that section's site highlighted, so
// students always know WHERE the current method enters the training pipeline.
#let reg-map(active: none) = {
  let site(key, label) = {
    let on = active == key or active == "all"
    (label, if on { 1.4pt + ACC } else { 0.8pt + MUTED.lighten(25%) },
     if on { ACC.lighten(90%) } else { white },
     if on { INK } else { MUTED })
  }
  let (d-l, d-s, d-f, d-t) = site("data", [data $(x, y)$])
  let (m-l, m-s, m-f, m-t) = site("model", [model $f_theta$ · hidden $h$])
  let (o-l, o-s, o-f, o-t) = site("objective", [penalty $lambda Omega(theta)$])
  let (t-l, t-s, t-f, t-t) = site("targets", [targets $tilde(y)$])
  let (l-l, l-s, l-f, l-t) = site("training", [training loop: update $theta$, stop at $t^star$])
  align(center, diagram(
    spacing: (23mm, 8.5mm), node-stroke: 0.9pt + INK, node-fill: white,
    {
      let box-node(pos, label, stroke, fill, tcol) = node(pos, text(size: 12.5pt, fill: tcol)[#label],
        shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 6.5pt, stroke: stroke, fill: fill)
      box-node((0, 0), d-l, d-s, d-f, d-t)
      box-node((1, 0), m-l, m-s, m-f, m-t)
      node((2, 0), text(size: 12.5pt, fill: MUTED)[prediction $p$], shape: fletcher.shapes.rect,
        corner-radius: 3pt, inset: 6.5pt, stroke: 0.8pt + MUTED.lighten(25%))
      node((3, 0), text(size: 12.5pt)[loss $ell(p, tilde(y))$], shape: fletcher.shapes.rect,
        corner-radius: 3pt, inset: 6.5pt, stroke: 0.9pt + INK)
      box-node((3, -1), t-l, t-s, t-f, t-t)
      box-node((3, 1), o-l, o-s, o-f, o-t)
      box-node((1.5, 1), l-l, l-s, l-f, l-t)
      edge((0, 0), (1, 0), "-|>", stroke: 0.8pt + MUTED)
      edge((1, 0), (2, 0), "-|>", stroke: 0.8pt + MUTED)
      edge((2, 0), (3, 0), "-|>", stroke: 0.8pt + MUTED)
      edge((3, -1), (3, 0), "-|>", stroke: 0.8pt + MUTED)
      edge((3, 1), (3, 0), "-|>", stroke: 0.8pt + MUTED)
      edge((1.5, 1), (1, 0), "-|>", stroke: (paint: MUTED, thickness: 0.8pt, dash: "dashed"))
    }
  ))
}

// Compact one-row variant of the map for mid-section orientation (the full
// reg-map appears only twice: at the concept's introduction and the synthesis).
#let reg-strip(active) = {
  let chip(key, label) = {
    let on = active == key or active == "all"
    box(fill: if on { ACC.lighten(88%) } else { white },
      stroke: if on { 1.2pt + ACC } else { 0.7pt + MUTED.lighten(30%) },
      radius: 3pt, inset: (x: 8pt, y: 4pt),
      text(size: 13pt, fill: if on { INK } else { MUTED },
        weight: if on { 700 } else { 400 }, label))
  }
  align(center, {
    let arrow = text(size: 12pt, fill: MUTED.lighten(20%))[ #sym.arrow.r ]
    chip("data", [data])
    arrow
    chip("model", [model / features])
    arrow
    chip("objective", [objective + penalty])
    arrow
    chip("targets", [targets])
    arrow
    chip("training", [training loop])
  })
}

#title-slide()

// ═══════════════════════ PART I — THE PERFECT FIT THAT FAILS ═══════════════════════
== Four lectures made training loss go down — that was the easy part

#align(center, grid(columns: (1fr, 1fr), gutter: 26pt,
  hairline([Keep from L3–L6], [
    backprop measures $g_t$, AdamW turns it into reliable updates, initialization and normalization keep both alive through depth
  ], color: TEAL),
  hairline([Today's problem], [
    the number those tools minimize is computed on data the model has *already seen*
  ], color: BLUE),
))

#pause
$ min_theta cal(L)_"train" (theta) quad eq.not quad min_theta cal(L)_"new data" (theta) $

#note[
  Today's question: when driving $cal(L)_"train"$ down stops helping, *what else can training express — and where does it enter the pipeline?*
]

== Start here: the two curves disagree #V

#align(center, image("figures/train_val.png", width: 62%))

#pause
#punch[$cal(L)_"train" arrow.b$ throughout, while $cal(L)_"val" arrow.b$ then $arrow.t$: the model keeps fitting — but what it learns stops transferring.]

== Three splits, three different jobs #D

#align(center, table(
  columns: (42mm, 72mm, 68mm), stroke: 0.5pt + MUTED,
  inset: (x: 10pt, y: 6pt), align: (left, left, left),
  table.header([split], [job], [allowed decisions]),
  [#fitc[training]], [fit parameters $theta$], [gradients; optimizer updates],
  [#valc[validation]], [choose the recipe], [$lambda$, stopping time, $p$, augmentation, $epsilon$],
  [#text(fill: INK, weight: 650)[test]], [estimate final generalization], [*none* — spend once after the recipe is frozen],
))

#pause
#warning[Using test accuracy to choose even one regularization knob turns the test set into another validation set. Our toy spine therefore seals a separate dense test draw until the finale.]

== Outline

#[
#set text(size: 22pt)
#enum(
  [*Diagnose* — one tiny dataset, a perfect fit, and the gap that exposes it],
  [*Weight decay* — charge the objective for coefficient magnitude; AdamW],
  [*Early stopping* — treat training time itself as a capacity knob],
  [*Augmentation + Mixup/CutMix* — encode task-valid invariances],
  [*Dropout* — train overlapping subnetworks; prepare MC Dropout],
  [*Label smoothing* — soften targets and temper confidence],
  spacing: 9pt,
)
]

== Ten measurements and a model family you already know #V

We measure a smooth signal at ten inputs; every reading carries noise:

$ y_i = f(x_i) + epsilon_i, quad epsilon_i tilde cal(N)(0, 0.25^2). $

#spine-fig(curves: ((truth-curve, truth-stroke),), height: 50mm, ylim: (-1.8, 1.8))
#align(center, text(size: 13pt)[#sw(INK, [10 training points]) #h(13pt) #sw(MUTED, [the hidden truth $f$])])

#note[
  From ES 335: expand features as $phi(x) = (1, x, x^2, dots, x^9)$ and fit
  $hat(y) = bold(w)^top phi(x)$ — a linear model with 10 weights, flexible enough to bend anywhere.
]

== The training loss reaches zero — exactly #V

Ten equations, ten weights: the least-squares fit passes through *every* point.

#spine-fig(curves: ((truth-curve, truth-stroke), (curve-of(w9), 2.2pt + RED)))
#align(center, text(size: 13pt)[#sw(RED, [degree-9 fit · train MSE #fnum(mse(w9, xs, ys), d: 4)]) #h(13pt) #sw(MUTED, [truth])])

#pause
#punch[A training loss of 0.0000 is a #failc[red flag], not a finish line.]

== Nine points in between expose the failure #V

Measure the same signal at the nine #valc[midpoints] the model never saw:

#spine-fig(curves: ((truth-curve, truth-stroke), (curve-of(w9), 2.2pt + RED)), show-val: true, height: 41mm)
#align(center, text(size: 13pt)[#sw(INK, [train]) #h(13pt) #sw(BLUE, [validation (new)]) #h(13pt) #sw(RED, [the perfect fit])])

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 14pt, y: 5.5pt),
  align: (left, center, center),
  table.header([degree-9 fit], [#fitc[train MSE]], [#valc[validation MSE]]),
  [same ten weights], [$#fnum(mse(w9, xs, ys), d: 4)$], [$#failc[#fnum(mse(w9, xv, yv), d: 2)]$],
))

#pause
#note[Even the true $f$ would score $approx sigma^2 = 0.0625$ here; the perfect memorizer is $#fnum(mse(w9, xv, yv) / (NOISE * NOISE), d: 0) times$ worse — *between* the points it memorized.]

== Learning curves separate three diagnoses #D

The pair $(cal(L)_"train", cal(L)_"val")$ — read levels, gap, and trend:

#align(center, table(
  columns: (44mm, 50mm, 62mm, 42mm), stroke: 0.45pt + MUTED,
  inset: (x: 9pt, y: 6pt), align: (left, left, left, left),
  table.header([training], [validation], [diagnosis], [first move]),
  [high], [high, similar], [underfit or broken optimization], [repair fit (L5–L6)],
  [low], [much higher], [#failc[overfitting]], [#knobc[regularize]],
  [low], [low, similar], [fit transfers to this split], [stress-test, ship],
))

#pause
#note([A small gap alone is not success — both losses can agree because both are bad. Our spine: train $#fnum(mse(w9, xs, ys), d: 2)$, validation $#fnum(mse(w9, xv, yv), d: 2)$ — the second row.], color: BLUE)

== Same recipe, fresh noise: the fit is nervous #V

Re-draw the ten noise values four times and refit each dataset:

#let var-curves = (7, 8, 9, 10).map(s => {
  let yy = xs.enumerate().map(((i, x)) => truth(x) + NOISE * rnd.randn(s, i))
  (la.solve(VD, yy), ml.linreg(xs.map(x => (1.0, x)), yy))
})
#align(center, stack(dir: ltr, spacing: 8mm,
  {
    set text(size: 12.5pt)
    lq.diagram(
      width: 82mm, height: 46mm, xlim: (-1.06, 1.06), ylim: (-2.9, 2.9), margin: 0%,
      xlabel: [$x$], title: [degree 1 · four refits],
      xaxis: (ticks: (-1, 0, 1), subticks: none, mirror: false),
      yaxis: (ticks: (-2, 0, 2), subticks: none, mirror: false),
      lq.plot(XG, truth-curve, stroke: truth-stroke, mark: none),
      ..var-curves.map(c => lq.plot(XG, XG.map(x => c.at(1).at(0) + c.at(1).at(1) * x),
        stroke: 1.4pt + TEAL.transparentize(20%), mark: none)),
    )
  },
  {
    set text(size: 12.5pt)
    lq.diagram(
      width: 82mm, height: 46mm, xlim: (-1.06, 1.06), ylim: (-2.9, 2.9), margin: 0%,
      xlabel: [$x$], title: [degree 9 · four refits],
      xaxis: (ticks: (-1, 0, 1), subticks: none, mirror: false),
      yaxis: (ticks: (-2, 0, 2), subticks: none, mirror: false),
      lq.plot(XG, truth-curve, stroke: truth-stroke, mark: none),
      ..var-curves.map(c => lq.plot(XG, curve-of(c.at(0)), stroke: 1.3pt + RED.transparentize(30%), mark: none)),
    )
  },
))

#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([Rigid model], [barely reacts to the noise — but cannot express the signal (*bias*)], color: TEAL),
  hairline([Flexible model], [swings with every noise draw — the noise *is* in the fit (*variance*)], color: RED),
))

== Both of these fits have zero training loss #V

Narrow radial bumps through the same ten points *also* score $0.0000$:

#spine-fig(curves: (
  (truth-curve, truth-stroke),
  (curve-of(w9), 1.9pt + RED),
  (XG.map(rbfval), 1.9pt + TEAL),
), height: 43mm)
#align(center, text(size: 13pt)[#sw(RED, [polynomial interpolant]) #h(12pt) #sw(TEAL, [narrow-bump interpolant]) #h(12pt) — both: train MSE $= 0$])

#pause
#punch[Training data cannot choose between them. They differ exactly *where the data is silent*.]

#note[Modern deep networks live here: many settings reach zero training error — even on *random labels* (Zhang et al., 2017). Which rule you get is decided by everything else about training.]

== Regularization is a policy for the silent regions #V

#reg-map(active: "all")

#v(6pt)
#pause
#punch[Each method today is one *preference for simpler behaviour*, injected at one site, with one knob — and #valc[validation] sets every knob.]

#v(6pt)
#semantic-legend

// ═══════════════════════ PART II — WEIGHT DECAY ═══════════════════════
= Weight decay: charge the objective for magnitude

== Add the preference directly to the objective #D

#reg-strip("objective")

#v(4pt)
$ cal(L)_"data"(w) = 1/(2n) norm(X w - y)_2^2, quad
  cal(J)(w) = underbrace(cal(L)_"data"(w), #fitc[fit]) + underbrace(lambda_"wd" / 2 norm(w)_2^2, #knobc[shrink]) $

#pause
#note[
  *Convention:* the $1/2$ factors make $nabla cal(J) = X^top(X w - y)/n + lambda_"wd" w$.
  Lecture 1 derived the Gaussian-prior view; today we watch what the same term does to a fit and an optimizer.
  #text(size: 14pt, fill: MUTED)[Revisit: #link(IA + "map-regularization/")[interactive-articles/map-regularization]]
]

== What the perfect fit paid in coefficients #V

The zero-training-loss polynomial needed enormous, cancelling weights:

#align(center, bars(
  w9.map(v => calc.round(v, digits: 1)),
  labels: range(10).map(k => [$w_#k$]),
  annotate: false, bar: 8mm, span: 36mm,
  color: (i, v) => RED.transparentize(12%),
))
#caption[weights of the degree-9 interpolant · largest #fnum(calc.max(..w9.map(calc.abs)), d: 0) in magnitude · $norm(bold(w))_2 = #fnum(la.norm(w9), d: 0)$]

#pause
#punch[Wild wiggles require huge coefficients. #knobc[Penalizing magnitude] is a direct tax on wiggling.]

#note[Caveat: parameter norm is not function smoothness (ReLU layers can rescale freely) — it is a useful preference, and validation decides if it helps.]

== One knob sweeps the family from wiggle to flat #V

Same ten points, same ten features — only $lambda$ changes:

#spine-fig(curves: (
  (truth-curve, truth-stroke),
  (curve-of(w9), 1.8pt + RED.transparentize(25%)),
  (curve-of(ridge-fit(1.0)), 1.8pt + MUTED),
  (curve-of(w-star), 2.3pt + GREEN),
))
#align(center, text(size: 13pt)[
  #sw(RED, [$lambda = 0$ · $norm(bold(w)) = #fnum(la.norm(w9), d: 0)$]) #h(10pt)
  #sw(GREEN, [$lambda = 10^(-3)$ · $norm(bold(w)) = #fnum(la.norm(w-star), d: 1)$]) #h(10pt)
  #sw(MUTED, [$lambda = 1$ · $norm(bold(w)) = #fnum(la.norm(ridge-fit(1.0)), d: 1)$])
])

#pause
#punch[$lambda$ is an exchange rate between the two forces: $0$ returns the memorizer, too much buys back the rigid model.]

== Checkpoint: which $lambda$ do we keep? #Q

The sweep on the spine (all values computed live):

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 5pt),
  align: (center, center, center, center),
  table.header([$lambda$], [#fitc[train MSE]], [#valc[val MSE]], [$norm(bold(w))_2$]),
  ..lam-rows.filter(r => r.at(0) != 0.00001 and r.at(0) != 0.1).map(r => (
    [$10^#str(int(calc.round(calc.log(r.at(0)))))$],
    [$#fnum(r.at(1), d: 3)$], [$#fnum(r.at(2), d: 3)$], [$#fnum(r.at(3), d: 1)$],
  )).flatten()
))

#mcq(
  [Which column is allowed to choose $lambda$?],
  [train MSE — it is the objective we optimize],
  [validation MSE],
  [$norm(bold(w))_2$ — smaller is safer],
  [none; average all three columns],
)

== Answer: validation chooses; this data loss votes $lambda = 0$ #A

#mcq-answer(
  [B],
  [validation MSE picks $lambda^star = 10^(-3)$],
  [in this ridge sweep the *unpenalized* training MSE is minimized at $lambda = 0$; the weight norm is a mechanism check, not a target.],
)

#pause
#punch[This selection loop — sweep the knob, read #valc[validation] — repeats for *every* regularizer today.]

== The U-curve is the pattern to look for — not a law #V

#align(center, lines(
  x: lam-grid.map(l => calc.log(l)),
  y: (lam-rows.map(r => r.at(1)), lam-rows.map(r => r.at(2))),
  labels: ([train], [validation]), colors: (TEAL, BLUE), log-y: true, legend: "tl",
  vlines: ((calc.log(lam-star), [$lambda^star$], ACC),),
  x-label: [$log_10 lambda$], y-label: [MSE], size: (146mm, 44mm),
))

#note[
  Left of $lambda^star$: variance dominates (memorization). Right: bias dominates (back to rigid).
  Other regularization knobs often show a related sweet spot, but real validation curves can be noisy, flat, or multi-modal — sweep and measure.
]

== Under SGD, the penalty becomes multiplicative shrinkage #D

The penalty's gradient is just $lambda theta_t$, so the SGD step factors:

$ theta_(t+1) = theta_t - eta (g_t + lambda theta_t) = underbrace((1 - eta lambda) theta_t, #knobc[shrink first]) - underbrace(eta g_t, #fitc[then fit]) $

One concrete step with $theta_t = 5$, $g_t = 3$, $eta = 0.1$, $lambda = 0.2$:

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 4.5pt),
  align: (left, left, center),
  table.header([piece], [substitution], [value]),
  [#knobc[shrink]], [$(1 - 0.1 dot 0.2) times 5 = 0.98 times 5$], [$4.90$],
  [#fitc[fit]], [$-0.1 times 3$], [$-0.30$],
  [$theta_(t+1)$], [$4.90 - 0.30$], [$4.60$],
))

#pause
#punch[Each step, every weight quietly loses $2 percent$ of itself — hence *weight decay*.]

== Geometry: weakly constrained directions shrink first #V #OPT

Quadratic data loss (curvature $4$ along $theta_1$, $0.6$ along $theta_2$) plus the penalty's circles:

#let wd-data(x, y) = 2.0 * calc.pow(x - 1.5, 2) + 0.3 * calc.pow(y - 1.35, 2)
#let wd-pen(x, y) = x * x + y * y
#let wd-path = range(41).map(k => {
  let lam = calc.pow(10.0, -1.5 + k * 0.075)
  (4.0 / (4.0 + lam) * 1.5, 0.6 / (0.6 + lam) * 1.35)
})
#align(center, contour(
  (wd-data, wd-pen), xlim: (-0.5, 2.3), ylim: (-0.5, 2.0), samples: 52, levels: 6,
  size: (118mm, 56mm), colors: (TEAL, ACC.transparentize(35%)),
  paths: (wd-path,),
  marks: ((1.5, 1.35, [$lambda = 0$], TEAL), (0.0, 0.0, [$lambda arrow.r oo$], ACC),
          (4.0 / 4.3 * 1.5, 0.6 / 0.9 * 1.35, [$lambda = 0.3$], GREEN)),
  x-label: [$theta_1$ (strongly curved: data insists)], y-label: [$theta_2$ (weakly curved)],
))

#note[
  The path bends: the *weakly curved* coordinate collapses first ($theta_2$: $1.35 arrow.r 0.90$ already at $lambda = 0.3$), while the direction the data insists on survives longest. Decay suppresses what the data barely constrains.
]

== Checkpoint: send the same penalty through Adam #Q

Recall Lecture 5. Two weights $theta = (1, 1)$, but Adam's scale memory differs: $hat(v) = (16, 0.16)$. Take a one-step snapshot with $eta = 0.01$, $lambda = 0.1$, and a frozen diagonal preconditioner $hat(v)$.

Ignoring first-moment dynamics and $epsilon$ to isolate the effect: if $lambda theta$ rides through the adaptive update, its contribution is $-eta lambda theta slash sqrt(hat(v))$.

#mcq(
  [How do the two shrinkage contributions compare?],
  [both shrink by $0.001$ — the penalty treats equals equally],
  [the high-$hat(v)$ weight shrinks $10 times$ more],
  [the low-$hat(v)$ weight shrinks $10 times$ more],
  [neither shrinks; Adam cancels the penalty exactly],
)

== Answer: the preconditioner distorts the intended preference #A

#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([$L_2$ inside Adam], [
    $-0.01 (0.1) slash sqrt(16) = -0.00025$
    #v(3pt)
    $-0.01 (0.1) slash sqrt(0.16) = -0.0025$
  ], color: MUTED),
  hairline([Decoupled (AdamW)], [
    $-eta lambda theta_1 = -0.001$
    #v(3pt)
    $-eta lambda theta_2 = -0.001$
  ], color: ACC),
))

#mcq-answer([C], [the low-$hat(v)$ weight shrinks $10 times$ more],
  [dividing by $sqrt(hat(v))$ rescales the penalty per-coordinate — you asked for proportional shrinkage and got something else.])

== AdamW: decay the weights, precondition only the data gradient #D

$ theta_(t+1) = underbrace((1 - eta lambda) theta_t, #knobc[decay, undistorted]) - eta underbrace("AdamUpdate"_t (g), #fitc[adaptive data step]) $

#align(center, table(
  columns: (48mm, 1fr), stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  [plain SGD], [$L_2$ penalty and weight decay are the *same algebra*],
  [adaptive (Adam)], [they *differ*; decoupling restores the intended preference (AdamW)],
))

#pause
#note[
  Practice notes: `weight_decay` in `torch.optim.AdamW` is this decoupled $lambda$; biases and normalization
  gains are commonly excluded from decay — a recipe choice to log, not a theorem.
  #text(size: 14pt)[Loshchilov & Hutter (2019) · #link(PT + "generated/torch.optim.AdamW.html")[PyTorch AdamW API].]
]

== PyTorch: make the decay policy visible in parameter groups

#codebox(size: 12.8pt)[```python
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    # Common recipe: do not decay bias or 1-D scale/shift parameters.
    (no_decay if p.ndim == 1 or name.endswith(".bias") else decay).append(p)

optimizer = torch.optim.AdamW([
    {"params": decay,    "weight_decay": 1e-4},
    {"params": no_decay, "weight_decay": 0.0},
], lr=3e-4)
```]

#note([The exclusion is a *recipe*, not a theorem. Log the groups, learning-rate schedule, and $lambda_"wd"$ together; AdamW's effective shrink per step is $eta_t lambda_"wd"$.], color: ACC)

== $L_1$ is the sparsity-seeking cousin #V

#align(center, table(
  columns: (36mm, 46mm, 56mm, 44mm), stroke: 0.45pt + MUTED,
  inset: (x: 9pt, y: 6pt), align: (left, left, left, left),
  table.header([penalty], [gradient force], [pressure], [role in DL]),
  [$lambda norm(theta)_2^2 slash 2$], [$lambda theta$ — proportional], [shrink everything a little], [default],
  [$lambda norm(theta)_1$], [$lambda "sign"(theta)$ — subgradient], [promote sparse solutions], [specialized],
))

#note[
  From Lecture 1: $L_2$ is a Gaussian prior, $L_1$ a Laplace prior. Ordinary SGD need not land exactly on zero; proximal/thresholding methods or explicit pruning create exact zeros. In standard dense networks, $L_1$ is secondary.
]

#place(bottom + center, dy: -2pt, caption[
  Interactive: drag $lambda$ yourself — #link(IA + "weight-decay/")[interactive-articles/weight-decay]
])

// ═══════════════════════ PART III — EARLY STOPPING ═══════════════════════
= Early stopping: training time is a capacity knob

== Train the spine by gradient descent and watch both losses #V

#reg-strip("training")

#v(4pt)
Same model, *no penalty* — just gradient descent from $bold(w) = 0$, both losses logged:

#align(center, lines(
  x: range(33).map(k => k / 4.0),
  y: (
    range(33).map(k => mse(wt(calc.pow(10.0, k / 4.0)), xs, ys)),
    range(33).map(k => mse(wt(calc.pow(10.0, k / 4.0)), xv, yv)),
  ),
  labels: ([train], [validation]), colors: (TEAL, BLUE), log-y: true, markers: false,
  vlines: ((calc.log(t-star), [$t^star = #str(t-star)$], ACC),),
  x-label: [$log_10$ updates], y-label: [MSE], size: (146mm, 42mm),
))

#pause
#punch[#fitc[Train] falls forever. #valc[Validation] falls, turns at $t^star$, and climbs — the U-curve again, with *time* on the axis.]

== The rule: keep the checkpoint validation preferred #D

$ t^star = op("arg min", limits: #true)_t cal(L)_"val" (bold(w)_t) $

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 5.5pt),
  align: (center, center, center, center),
  table.header([checkpoint $t$], [#fitc[train MSE]], [#valc[val MSE]], [verdict]),
  ..(100, 1000, 5000, 30000, 1000000).map(t => (
    [$#str(t)$], [$#fnum(mse(wt(t), xs, ys), d: 3)$], [$#fnum(mse(wt(t), xv, yv), d: 3)$],
    if t == t-star { keepc[restore this one] } else { text(fill: MUTED)[discard] },
  )).flatten()
))

== PyTorch: stop, save, and restore the validation winner #V

#codebox(size: 11.6pt)[```python
from copy import deepcopy
best = patience_ref = float("inf")
wait, best_state = 0, None

for epoch in range(max_epochs):
    train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)

    if val_loss < best:                  # strict argmin checkpoint
        best = val_loss
        best_state = deepcopy(model.state_dict())

    if val_loss < patience_ref - min_delta:
        patience_ref, wait = val_loss, 0 # meaningful progress
    else:
        wait += 1
        if wait >= patience: break

model.load_state_dict(best_state)  # restore; do not keep the final epoch
```]

#note[Checkpointing tracks the strict $arg min$; `min_delta` only controls patience. To resume training later, also save optimizer, scheduler, scaler, epoch, and RNG states.]

== Why validation turns: directions arrive in order of strength #D

For quadratic loss, gradient descent from $bold(w)_0 = 0$ has an exact per-direction solution. Along eigendirection $bold(v)_j$ of the data matrix (strength $sigma_j^2$):

$ bold(v)_j^top bold(w)_t = (1 - (1 - eta sigma_j^2)^t) dot bold(v)_j^top bold(w)_oo $

#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([Strong directions ($sigma_j^2$ large)], [factor $arrow.r 1$ within a few updates — the *shared, simple* structure], color: TEAL),
  hairline([Weak directions ($sigma_j^2$ tiny)], [need $t approx 1 slash (eta sigma_j^2)$ updates — where the *noise* hides], color: RED),
))

#note[
  Our spine's ten strengths, computed: $sigma^2 = #fnum(evals.at(0), d: 2), #fnum(evals.at(1), d: 2), #fnum(evals.at(2), d: 2), dots, 7 times 10^(-8)$ — a *ten-million-fold* spread. So there is a long window where signal has arrived and noise has not. Exact for linear least squares; "networks learn simple patterns first" is the empirical deep-learning analogue.
]

== Watch the stopping time act as a capacity knob #V

Three checkpoints of the *same run* — no penalty anywhere:

#spine-fig(curves: (
  (truth-curve, truth-stroke),
  (curve-of(wt(100000000)), 1.8pt + RED.transparentize(25%)),
  (curve-of(wt(100)), 1.8pt + MUTED),
  (curve-of(w-stop), 2.3pt + GREEN),
))
#align(center, text(size: 13pt)[
  #sw(MUTED, [$t = 100$ · still arriving]) #h(10pt)
  #sw(GREEN, [$t^star = #str(t-star)$ · kept]) #h(10pt)
  #sw(RED, [$t = 10^8$ · the noise arrived — slide 4's memorizer, reborn])
])

#pause
#punch[Stop early $arrow.r$ the weak, noise-carrying directions simply never arrive.]

== Optional: the clock and the penalty are the same filter #D #OPT

Per direction, the two methods apply strikingly similar suppression factors:

$ underbrace(1 - (1 - eta sigma^2)^t, "early stopping") quad "vs." quad underbrace(sigma^2 / (sigma^2 + lambda), "weight decay") $

#align(center, lines(
  fn: (
    s => 1.0 - calc.pow(1.0 - eta-gd * calc.pow(10.0, s), t-star),
    s => calc.pow(10.0, s) / (calc.pow(10.0, s) + 1.0 / (eta-gd * t-star)),
  ),
  domain: (-8, 0.2), samples: 90, markers: false, legend: "tl",
  labels: ([stop at $t^star$], [decay $lambda = 1 slash (eta t^star)$]), colors: (GREEN, ACC),
  x-label: [$log_10 sigma^2$ (direction strength)], y-label: [kept fraction], size: (134mm, 31mm),
))

#note[
  On our spine the match is uncanny: validation independently chose $lambda^star = 10^(-3)$ and $t^star = #str(t-star)$ — and $1 slash (eta t^star) = #fnum(1.0 / (eta-gd * t-star), d: 4)$. Two knobs, one preference.
]

== Predict, then run the curves notebook #I

#interbox(link-to: NB + "01_overfitting_and_early_stopping.ipynb")[
  A NumPy lab that makes the mechanism inspectable: a flexible model overfits a small dataset while you log both curves and restore the best checkpoint.
]

Before executing each cell, write down:

- which epoch range the validation minimum will fall in;
- what *patience* value would have caught it;
- what happens to the restored-checkpoint test error if you double the noise.

// ═══════════════════ PART IV — AUGMENTATION AND MIXUP ═══════════════════
= Augmentation: teach invariances through the data

== One cat, six views — which changes should the classifier ignore? #V

#align(center, image("figures/cifar10_augmentation_gallery.png", width: 92%))

#pause
#punch[Augmentation is not decoration: each view is a claim that the answer stays #keepc[cat].]

#caption[real CIFAR-10 training image · transforms regenerated locally · inspectability matters more than photorealism at $32 times 32$]

== The strongest practical regularizer edits the data distribution #D

#reg-strip("data")

#v(4pt)
Declare a distribution $cal(T)$ of transformations. For a label-preserving classification transform, $y prime = y$; for structured targets, transform the pair:

$ (x prime, y prime) = T(x, y), quad min_theta thick EE_((x, y)) thick EE_(T tilde cal(T)) [ell (f_theta(x prime), y prime)] $

#pause
#note[
  Every epoch samples a new training view: Monte Carlo over your declared invariances. The stored dataset is finite; the *training distribution* no longer is. In vision this is often the highest-leverage regularizer.
]

== A transform is a claim about the label #V

Pixel letters, class set ${a, b, c, d, e}$ — watch what each transform claims:

#let GB = (
  (1, 0, 0, 0, 0, 0, 0),
  (1, 0, 0, 0, 0, 0, 0),
  (1, 0, 0, 0, 0, 0, 0),
  (1, 1, 1, 1, 1, 0, 0),
  (1, 0, 0, 0, 1, 0, 0),
  (1, 0, 0, 0, 1, 0, 0),
  (1, 1, 1, 1, 1, 0, 0),
  (0, 0, 0, 0, 0, 0, 0),
)
#let shift-right(g) = g.map(r => (0,) + r.slice(0, r.len() - 1))
#let hflip(g) = g.map(r => r.rev())
#let glyph-fill = (v, i, j) => if v > 0.5 { INK } else { rgb("#F3F2EE") }
#let glyph(g, cell: 4.6mm) = grid-map(g, cell: cell, fill: glyph-fill, fmt: none, stroke: 0.3pt + rgb("#DDDBD5"))
#align(center, stack(dir: ltr, spacing: 12mm,
  stack(spacing: 4pt, glyph(GB), text(size: 13pt)[original — label #keepc[“b”]]),
  stack(spacing: 4pt, glyph(shift-right(GB)), text(size: 13pt)[shifted — still #keepc[“b”]]),
  stack(spacing: 4pt, glyph(hflip(GB)), text(size: 13pt)[flipped — now reads #failc[“d”]]),
))

#pause
#punch[Horizontal flip is a *valid* invariance for flowers — and a *label change* for letters.]

== Checkpoint: which transform is safe for the letter task? #Q

#mcq(
  [Which policy states an invariance the letter labels actually possess?],
  [horizontal flip — objects don't care about left and right],
  [rotate by $180 degree$ — orientation is superficial],
  [shift by one or two pixels, keeping the glyph in frame],
  [any transform that makes training loss go up],
)

#note([Commit, and name the counterexample for each rejected option.], color: BLUE)

== Answer: only the shift preserves every letter's label #A

#mcq-answer(
  [C],
  [small shifts keep the glyph and the label],
  [flip turns $b arrow.l.r d$; a $180 degree$ rotation turns $6$ into $9$; "harder training" is not a label-preservation argument.],
)

#align(center, [
  #set text(size: 15pt)
  #table(
    columns: (34mm, 54mm, 76mm), stroke: 0.45pt + MUTED,
    inset: (x: 8pt, y: 4pt), align: (left, left, left),
    table.header([transform], [claims the task ignores…], [breaks when…]),
    [random crop], [position and scale (a little)], [the crop removes the object],
    [horizontal flip], [left–right orientation], [text, digits, anatomy (the heart has a side)],
    [colour jitter], [illumination and tint], [colour is the diagnosis],
  )
])
#caption[detection reminder: geometric transforms must move boxes and masks too — augment the *supervision*]

== The label contract changes with the task #V

#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 15pt,
  hairline([Classification], [crop or flip the image; the class index usually stays fixed], color: TEAL),
  hairline([Detection / segmentation], [move boxes, masks, and keypoints with the pixels], color: ACC),
  hairline([Audio / text], [mask time–frequency bands or edit tokens only when meaning survives], color: BLUE),
))

#pause
#warning[Applying a geometric transform to $x$ but not to a spatial target $y$ creates *wrong labels*, not useful noise. Prefer transform APIs that accept the whole sample.]

== Input jitter imposes local smoothness through the data #V

Impose local smoothness: reuse each measured $y$ at 8 nearby inputs $x + epsilon$. This is a deliberate approximation — not an exact invariance of $sin(pi x)$:

#align(center, {
  set text(size: 13pt)
  lq.diagram(
    width: 146mm, height: 40mm, xlim: (-1.06, 1.06), ylim: (-2.7, 3.7), margin: 0%,
    xlabel: [$x$], ylabel: [$y$],
    grid: (stroke: 0.35pt + MUTED.transparentize(78%)),
    xaxis: (ticks: (-1, 0, 1), subticks: none, mirror: false),
    yaxis: (ticks: (-2, 0, 2), subticks: none, mirror: false),
    lq.plot(XG, truth-curve, stroke: truth-stroke, mark: none),
    lq.plot(XG, curve-of(w9), stroke: 1.6pt + RED.transparentize(45%), mark: none),
    lq.plot(XG, curve-of(w-aug), stroke: 2.3pt + GREEN, mark: none),
    lq.scatter(xa, ya, color: TEAL.transparentize(45%), size: 2.6pt),
    lq.scatter(xs, ys, color: INK, size: 4.6pt),
  )
})
#align(center, text(size: 13pt)[
  #sw(TEAL, [80 jittered copies]) #h(10pt) #sw(GREEN, [fit to them · val MSE #fnum(mse(w-aug, xv, yv), d: 3)]) #h(10pt) #sw(RED, [old interpolant · #fnum(mse(w9, xv, yv), d: 2)])
])

#pause
#note[
  The fit becomes locally smoother, at the cost of some bias; validation judges the trade. Input noise is closely related to a smoothness penalty (Bishop, 1995).
]

== PyTorch: the CIFAR-10 train/eval transform split #V

#codebox(size: 12.3pt)[```python
from torchvision.transforms import v2

train_tf = v2.Compose([
    v2.ToImage(),
    v2.RandomCrop(32, padding=4, padding_mode="reflect"),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean, std),
])

eval_tf = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean, std),
])
```]

#note([Train sees fresh random views; validation/test use a deterministic pipeline. Compute `mean, std` from the training split only. #link(PTV + "transforms.html")[torchvision v2 reference]], color: BLUE)

== Before training, audit an augmented batch #V

#align(center, grid(columns: (72mm, 1fr), gutter: 18pt,
  image("figures/cifar10_augmentation_gallery.png", width: 72mm),
  [
    #set text(size: 16pt)
    *For eight random views, ask:*
    #v(5pt)
    - Is the object still present?
    - Would a domain expert keep the label?
    - Are boxes or masks still aligned?
    - Is the strength plausible at deployment?
    - Is normalization applied *after* pixel transforms?
  ],
))

#codebox(size: 12.5pt)[```python
views = torch.stack([train_tf(raw_image) for _ in range(8)])
grid = torchvision.utils.make_grid(denormalize(views), nrow=4)
show(grid)                       # make this a pre-flight check
```]

== Mixup: train on blends of examples and labels #D

Interpolate *both* the inputs and the one-hot targets, with $lambda tilde "Beta"(alpha, alpha)$:

$ tilde(x) = lambda_"mix" x_i + (1 - lambda_"mix") x_j, quad tilde(y) = lambda_"mix" y_i + (1 - lambda_"mix") y_j $

#align(center, image("figures/cifar10_mixup_cutmix.png", width: 89%))

#pause
#punch[Between examples — exactly where data was silent — the model is now told to behave *linearly*.]

== The mixed target is exactly the mixing weight #D

For $lambda_"mix" = 0.70$ mixing a cat with a dog:

$ tilde(y) = 0.70 thin y_"cat" + 0.30 thin y_"dog" $

The cross-entropy against a soft target splits by linearity:

$ ell(tilde(y), p) = -0.70 log p_"cat" - 0.30 log p_"dog" $

#pause
#note[
  Two checkable consequences: the loss never reaches zero (even a perfect $p = tilde(y)$ scores $#fnum(-0.7 * calc.ln(0.7) - 0.3 * calc.ln(0.3), d: 3)$, the target entropy), and extreme confidence on either class is punished. MixUp regularizes targets *and* inputs.
]

== PyTorch: MixUp and CutMix happen after batching #V

#codebox(size: 12.2pt)[```python
from torchvision.transforms import v2
import torch.nn.functional as F

batch_aug = v2.RandomChoice([
    v2.MixUp(num_classes=K, alpha=0.2),
    v2.CutMix(num_classes=K, alpha=1.0),
])

model.train()
for images, class_ids in train_loader:
    images, soft_targets = batch_aug(images, class_ids)
    logits = model(images)
    loss = F.cross_entropy(logits, soft_targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward(); optimizer.step()
```]

#note([The transform pairs examples within a batch and returns probability targets; `cross_entropy` accepts them directly. Accuracy still uses the original class IDs on an unmodified evaluation set. #link(PTV + "auto_examples/transforms/plot_cutmix_mixup.html")[official example]], color: BLUE)

== CutMix and RandAugment industrialize the idea #V

#align(center, table(
  columns: (52mm, 1fr), stroke: 0.45pt + MUTED, inset: (x: 10pt, y: 6.5pt), align: (left, left),
  table.header([extension], [one-line mechanism]),
  [CutMix], [paste a *patch* of image $j$ onto image $i$; mix labels by pasted-area fraction — keeps local evidence crisp],
  [RandAugment], [sample a couple of transforms and one global magnitude from a small policy space — augmentation search without the search cost],
))

#note[
  Both inherit augmentation's contract: the arithmetic is automatic, but the transformation space and its strength remain validation-selected claims about the task.
]

#place(bottom + center, dy: -2pt, caption[
  Interactive: blend images and labels live — #link(IA + "augmentation-mixup/")[interactive-articles/augmentation-mixup]
])

// ═══════════════════════ PART V — DROPOUT ═══════════════════════
= Dropout: train with randomly deleted features

== A feature that only works in committee is fragile #V

#reg-strip("model")

#v(10pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([Co-adaptation], [with all units always present, a hidden feature can lean on its neighbours — a brittle committee that memorizes together], color: RED),
  hairline([The intervention], [delete a random subset of features at *every* training step, so each must carry evidence on its own], color: ACC),
))

#pause
#punch[If any teammate might vanish, every feature must be individually useful.]

== The mechanism is one random mask #D

Sample a fresh Bernoulli mask each pass; keep units with probability $q$:

$ m_i tilde "Bernoulli"(q), quad tilde(bold(h)) = (bold(m) dot.o bold(h)) / q $

#pause
#let drop-panel(mask, tag) = {
  set text(size: 11pt)
  align(center, diagram(
    spacing: (11mm, 3.6mm), node-stroke: 0.8pt + INK, node-fill: white,
    {
      for i in range(3) { node((0, (i - 1) * 1.6), none, radius: 2.5mm, fill: INK, stroke: none) }
      for (j, keep) in mask.enumerate() {
        let y = (j - 2.5) * 1.05
        if keep == 1 {
          node((1, y), none, radius: 2.5mm, fill: TEAL, stroke: none)
          for i in range(3) { edge((0, (i - 1) * 1.6), (1, y), stroke: 0.4pt + MUTED.lighten(25%)) }
          for o in range(2) { edge((1, y), (2, (o - 0.5) * 1.6), stroke: 0.4pt + MUTED.lighten(25%)) }
        } else {
          node((1, y), none, radius: 2.5mm, fill: white, stroke: (paint: MUTED, thickness: 0.8pt, dash: "dashed"))
        }
      }
      for o in range(2) { node((2, (o - 0.5) * 1.6), none, radius: 2.5mm, fill: ACC, stroke: none) }
      node((1, 3.6), text(size: 10.5pt, fill: MUTED, tag), stroke: none)
    }
  ))
}
#align(center, stack(dir: ltr, spacing: 13mm,
  drop-panel((1, 1, 0, 0, 1, 0), [step 1 · mask 110010]),
  drop-panel((0, 1, 0, 1, 0, 1), [step 2 · mask 010101]),
  drop-panel((0, 1, 1, 0, 1, 1), [step 3 · mask 011011]),
))
#caption[three consecutive training steps, masks drawn at $q = 0.5$ — a *different subnetwork* trains each step, all sharing one weight set]

== Divide by $q$ so the expected signal is unchanged #D

Why the $1 slash q$? Take the expectation over masks:

$ EE[tilde(h)_i] = (EE[m_i] dot h_i) / q = (q dot h_i) / q = h_i quad checkmark $

One concrete pass with $bold(h) = (2, 4, 6, 8)$, mask $(1, 0, 1, 0)$, $q = 0.5$:

$ tilde(bold(h)) = (1 dot 2, thick 0 dot 4, thick 1 dot 6, thick 0 dot 8) / 0.5 = (4, thick 0, thick 12, thick 0) $

#note[Corrected *in expectation* only — the injected variance $((1 - q) slash q) thin h_i^2$ remains: deliberate multiplicative noise.]

== From the equation to a tensor operation #V

#codebox(size: 13pt)[```python
def inverted_dropout(h, p_drop=0.3, training=True):
    if not training or p_drop == 0:
        return h
    if not 0 <= p_drop < 1:
        raise ValueError("p_drop must be in [0, 1)")

    q_keep = 1.0 - p_drop
    keep = torch.rand_like(h) < q_keep
    return h * keep / q_keep

h = torch.tensor([2., 4., 6., 8.])
draws = torch.stack([inverted_dropout(h) for _ in range(20_000)])
draws.mean(0)                    # approximately tensor([2, 4, 6, 8])
```]

#note([The mask has no gradients and no learnable parameters. The same learned weights receive gradients through many sampled subnetworks.], color: TEAL)

== Put dropout where representation redundancy exists #V

#codebox(size: 13pt)[```python
class Classifier(nn.Module):
    def __init__(self, d, K):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),   # p_drop=.3; q_keep=.7
            nn.Linear(256, K),
        )

    def forward(self, features):
        return self.head(features)
```]

#align(center, grid(columns: (1fr, 1fr), gutter: 18pt,
  note([Start modestly: $p approx 0.1$–$0.3$ in modern blocks; larger values are common in wide dense heads.], color: ACC),
  note([Do not drop the final logits. With functional `F.dropout`, pass `training=self.training` explicitly.], color: RED),
))

== One network, $2^d$ overlapping committees #V

#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([Sampling view], [$d$ droppable units define up to $2^d$ subnetworks; each step trains one], color: ACC),
  hairline([Sharing view], [all subnetworks reuse *one* parameter set — no extra memory], color: TEAL),
  hairline([Test-time view], [running the full network $approx$ averaging the committee], color: BLUE),
))

#note[
  The averaging claim is exact for a single linear layer and an *approximation* for deep nonlinear networks — a useful intuition, not an identity (Srivastava et al., 2014).
]

== Checkpoint: what does `Dropout(p = 0.5)` do in `eval()` mode? #Q

PyTorch's `p` is the *drop* probability, so $q = 1 - p$.

#codebox(size: 15pt)[```python
h = torch.tensor([2., 4., 6., 8.])
drop = nn.Dropout(p=0.5)
drop.train(); drop(h)   # one draw gave: tensor([ 4.,  0., 12.,  0.])
drop.eval();  drop(h)   # = ?
```]

#mcq(
  [What does the `eval()` line return?],
  [$(2, 4, 6, 8)$ — the identity],
  [$(1, 2, 3, 4)$ — scaled by $q$],
  [$(4, 8, 12, 16)$ — scaled by $1 slash q$],
  [a fresh random mask, like training],
)

== Answer: evaluation is the untouched full network #A

#mcq-answer(
  [A],
  [the identity — dropout is disabled at test time],
  [*inverted* dropout already paid the $1 slash q$ correction during training, so evaluation needs no rescaling and stays deterministic. Training and evaluation differ in exactly one controlled way.],
)

#pause
#warning[A model accidentally left in `train()` mode at evaluation gives noisy, degraded predictions — a classic silent bug. `model.eval()` before measuring anything.]

== Three controls that students often conflate #V

#align(center, [
  #set text(size: 13.5pt)
  #table(
    columns: (53mm, 40mm, 57mm, 39mm), stroke: 0.45pt + MUTED,
    inset: (x: 7pt, y: 5.5pt), align: (left, left, left, left),
    table.header([control], [dropout], [BatchNorm], [autograd]),
    [`model.train()`], [random masks], [batch statistics; buffers update], [unchanged],
    [`model.eval()`], [identity], [stored statistics], [unchanged],
    [`torch.inference_mode()`], [unchanged], [unchanged], [disabled],
  )
])

#pause
#punch[`eval()` changes module behaviour; `inference_mode()` changes gradient tracking. They solve different problems.]

== Where dropout earns its keep today #V

#align(center, table(
  columns: (62mm, 1fr), stroke: 0.45pt + MUTED, inset: (x: 10pt, y: 6pt), align: (left, left),
  table.header([setting], [current practice]),
  [wide fully-connected heads], [classic sweet spot — where co-adaptation is worst],
  [transformer blocks], [small rates inside attention / FFN and on residuals],
  [modern conv backbones], [often secondary to strong augmentation and weight decay; classifier-head dropout remains common],
  [whole residual branches], [the same idea at branch scale: *stochastic depth / DropPath*],
))

#note([The knob is $p$: sweep it like $lambda$ and read #valc[validation]. Too little may not help; too much destroys useful signal.], color: BLUE)

== Standard inference collapses the masks; MC Dropout keeps sampling #V

#align(center, image("figures/mc_dropout_bridge.png", width: 64%))

#align(center, table(
  columns: (1fr, 1fr), stroke: none, inset: (x: 10pt, y: 3pt),
  [#valc[Standard:] dropout off → one deterministic prediction],
  [#knobc[MC:] dropout on → $T$ probabilities; report mean + spread],
))

#caption[schematic: masks agree near training support and can disagree away from it · spread is a useful signal, not a guarantee of correctness]

== Safe MC Dropout: freeze BatchNorm, enable only dropout #V

#codebox(size: 11.5pt)[```python
DROPOUT = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)

def mc_predict(model, x, T=30):
    model.eval()                       # BatchNorm uses stored statistics
    for module in model.modules():
        if isinstance(module, DROPOUT):
            module.train()             # only dropout stays stochastic

    with torch.inference_mode():
        probs = torch.stack([
            model(x).softmax(dim=-1) for _ in range(T)
        ])

    model.eval()                       # restore deterministic evaluation
    return probs.mean(0), probs.std(0)
```]

#pause
#warning[Do not call `model.train()` globally: BatchNorm would use the inference batch and update its buffers. Average *probabilities*, not logits. Functional dropout needs a separate explicit switch.]

== What this lecture establishes — and what the next one must answer #V

#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([Ready now], [why masks create different subnetworks; inverted scaling; train/eval modes; how to sample safely], color: GREEN),
  hairline([Next lecture], [predictive entropy; mutual information / BALD; Bayesian interpretation; calibration; failure cases], color: BLUE),
))

$ x arrow.r {m^(1), dots, m^(T)} arrow.r {p^(1), dots, p^(T)} arrow.r (underbrace(1/T sum_t p^(t), "mean"), underbrace("spread"(p^(1:T)), "uncertainty signal")) $

#pause
#punch[The mask was a training regularizer. Keeping it stochastic at inference turns it into a *prediction distribution*.]

#place(bottom + center, dy: -2pt, caption[
  Follow-along: #link(NB + "02_dropout_from_scratch.ipynb")[dropout-from-scratch notebook] · #link(PT + "generated/torch.nn.Dropout.html")[PyTorch API] · #link(IA + "dropout-playground/")[playground]
])

// ═══════════════════ PART VI — LABEL SMOOTHING ═══════════════════
= Label smoothing: soften the targets

== One-hot targets demand infinite confidence #D

#reg-strip("targets")

With target $y_b = 1$, cross-entropy wants $p_b = 1$ — which softmax only reaches as the correct logit races to $+oo$:

#align(center, lines(
  fn: (
    m => calc.ln(1.0 + 4.0 * calc.exp(-m)),
    m => 0.92 * calc.ln(1.0 + 4.0 * calc.exp(-m)) + 0.08 * calc.ln(calc.exp(m) + 4.0),
  ),
  domain: (0, 8), samples: 90, markers: false,
  labels: ([one-hot], [smoothed $epsilon = 0.1$]), colors: (RED, ACC),
  vlines: ((calc.ln(46.0), [$m^star = ln 46 approx 3.83$], ACC),),
  x-label: [correct-class logit margin $m$], y-label: [cross-entropy], size: (140mm, 40mm),
))

#pause
#punch[#failc[One-hot]: the gradient never rests. #knobc[Smoothed]: a finite margin is *optimal*.]

== Move $epsilon$ of the target mass off the correct class #D

$ tilde(y) = (1 - epsilon) thin y + epsilon / K bold(1) $

For $K = 5$, $epsilon = 0.1$, correct class “b”:

#align(center, stack(dir: ltr, spacing: 16mm,
  stack(spacing: 5pt,
    bars((0, 1, 0, 0, 0), labels: ([a], [b], [c], [d], [e]), bar: 7.5mm, span: 30mm,
      color: (i, v) => if i == 1 { RED } else { MUTED.lighten(35%) }),
    text(size: 12.5pt)[one-hot $y$]),
  stack(spacing: 5pt,
    bars((0.02, 0.92, 0.02, 0.02, 0.02), labels: ([a], [b], [c], [d], [e]), bar: 7.5mm, span: 30mm,
      color: (i, v) => if i == 1 { ACC } else { ACC.lighten(45%) }),
    text(size: 12.5pt)[smoothed $tilde(y)$]),
))

#note[
  Convention check: PyTorch's `CrossEntropyLoss(label_smoothing=0.1)` mixes with the uniform distribution over *all* $K$ classes, giving $0.92 slash 0.02$ — not the $0.90 slash 0.025$ of "redistribute to others". Name the convention before comparing $epsilon$ values.
  #text(size: 14pt)[#link(PT + "generated/torch.nn.CrossEntropyLoss.html")[PyTorch API reference]]
]

== PyTorch: one argument, but still a validation-selected knob #V

#codebox(size: 13pt)[```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

model.train()
for images, class_ids in train_loader:
    logits = model(images)
    loss = criterion(logits, class_ids)
    optimizer.zero_grad(set_to_none=True)
    loss.backward(); optimizer.step()

model.eval()
with torch.inference_mode():
    val_logits = model(val_images)
    val_nll = F.cross_entropy(val_logits, val_class_ids)  # hard labels
```]

#align(center, grid(columns: (1fr, 1fr), gutter: 20pt,
  note([Tune $epsilon$ on validation NLL or the metric you actually care about — not on training loss.], color: BLUE),
  note([The smoothed training loss has a positive floor; do not compare its raw value with a one-hot run.], color: ACC),
))

== Where the softened optimum comes from #D

Zero gradient forces $p = tilde(y)$, so at the optimum

$ e^(m^star) / (e^(m^star) + 4) = 0.92 quad arrow.r.double quad m^star = ln 46 approx 3.83 $

The mechanism, visible in the logit gradient $nabla_z ell = p - tilde(y)$ at $z = (0.6, 2.8, 1.1, -0.4, 0.3)$:

#let zsm = (0.6, 2.8, 1.1, -0.4, 0.3)
#let psm = {
  let m = calc.max(..zsm)
  let e = zsm.map(z => calc.exp(z - m))
  e.map(v => v / e.sum())
}
#align(center, stack(dir: ltr, spacing: 15mm,
  stack(spacing: 4pt,
    bars(psm.enumerate().map(((i, p)) => calc.round(p - (if i == 1 { 1 } else { 0 }), digits: 3)),
      labels: ([a], [b], [c], [d], [e]), bar: 7.2mm, span: 21mm,
      color: (i, v) => if v < 0 { TEAL } else { RED.lighten(20%) }),
    text(size: 12pt)[$p - y$ (one-hot)]),
  stack(spacing: 4pt,
    bars(psm.enumerate().map(((i, p)) => calc.round(p - (if i == 1 { 0.92 } else { 0.02 }), digits: 3)),
      labels: ([a], [b], [c], [d], [e]), bar: 7.2mm, span: 21mm,
      color: (i, v) => if v < 0 { TEAL } else { ACC }),
    text(size: 12pt)[$p - tilde(y)$ (smoothed)]),
))
#caption[negative bar = logit pushed up · smoothing stops pushing once wrong classes reach $0.02$]

== What smoothing buys — and what it costs #V

#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([Can help], [tempers over-confident logits and can improve validation NLL, accuracy, or calibration — the effect is data- and recipe-dependent], color: GREEN),
  hairline([Costs], [output probabilities are pulled toward uniformity — softer interpretability, and distillation teachers can lose class-similarity structure (Müller et al., 2019)], color: RED),
))

#note([
  The knob is $epsilon$, the judge is #valc[validation] — and for probability quality, measure calibration explicitly rather than admiring confidence.
  #text(size: 14pt, fill: MUTED)[Explore: #link(IA + "calibration/")[interactive-articles/calibration]]
], color: BLUE)

== Soft-target methods interact — do not stack them blindly #V

#align(center, [
  #set text(size: 15pt)
  #table(
    columns: (45mm, 62mm, 80mm), stroke: 0.45pt + MUTED,
    inset: (x: 8pt, y: 5pt), align: (left, left, left),
    table.header([method], [target source], [what becomes softer]),
    [label smoothing], [uniform distribution], [confidence for every example],
    [MixUp / CutMix], [a second training label], [confidence *and* behaviour between examples],
    [both together], [mixed target, then uniform mix], [possibly too much target entropy],
  )
])

#pause
#warning[If MixUp or CutMix already produces soft targets, test label smoothing as a separate ablation. More regularization is not monotonically better.]

// ═══════════════════ PART VII — ONE PREFERENCE, MANY KNOBS ═══════════════════
= One preference, many knobs

== Every method, one map #V

#reg-map(active: "all")

#v(10pt)
#pause
#punch[Different sites, one job: decide what the model does *where the data is silent* — then let #valc[validation] set the knob.]

== The taxonomy, with each method's knob #V

#align(center, [
  #set text(size: 16pt)
  #table(
    columns: (52mm, 92mm, 44mm), stroke: 0.45pt + MUTED,
    inset: (x: 9pt, y: 5.5pt), align: (left, left, left),
    table.header([site], [methods], [knob]),
    [objective], [weight decay ($L_2$), $L_1$, spectral penalties], [$lambda$],
    [training loop], [early stopping, minibatch noise], [$t^star$, batch size],
    [model / features], [dropout, stochastic depth], [$p$],
    [data], [crops · flips · jitter, Mixup, CutMix], [strength, $alpha$],
    [targets], [label smoothing], [$epsilon$],
    [architecture], [convolution, weight sharing, bottlenecks], [design choice],
  )
])

#note([Architecture adds no explicit penalty term, but it is not “free”: it restricts the function family and is still a validation-selected design choice.], color: ACC)

== Training regularizes even when you add nothing #V

#align(center, [
  #set text(size: 16.5pt)
  #table(
    columns: (52mm, 1fr), stroke: 0.45pt + MUTED, inset: (x: 10pt, y: 5.5pt), align: (left, left),
    table.header([implicit source], [selection pressure it exerts]),
    [SGD minibatch noise], [a stochastic trajectory with optimizer- and batch-size-dependent implicit bias],
    [initialization + early phase], [which basin training explores first (Lecture 6's init story)],
    [architecture], [some functions are easy to express, others practically unreachable],
    [BatchNorm's batch statistics], [per-batch jitter in activations — a *side effect*],
  )
])

#note[
  Keep BatchNorm filed under *optimization* (Lecture 6): its job is trainability; mild regularization is a bonus. This is also why changing optimizer or batch size silently moves your *effective* regularization.
]

== A staged working recipe — add evidence, not ingredients #V

#result[
  #set text(size: 19pt)
  reproducible baseline #h(4pt) → #h(4pt) valid augmentation #h(4pt) → #h(4pt) weight decay #h(4pt) → #h(4pt) targeted extras
]

#v(3pt)
#align(center, [
  #set text(size: 15pt)
  #table(
    columns: (15mm, 48mm, 1fr), stroke: 0.45pt + MUTED, inset: (x: 8pt, y: 4pt), align: (left, left, left),
    [1], [establish], [log train/validation curves; restore the best checkpoint],
    [2], [encode], [add domain-valid augmentation; inspect actual batches],
    [3], [control], [sweep AdamW decay; compare at the same compute budget],
    [4], [only if needed], [ablate dropout, MixUp/CutMix, or label smoothing one at a time],
  )
])

== Specialized regularizers answer specialized failures #V

Methods reserved for specific problems:

#align(center, [
  #set text(size: 15.5pt)
  #table(
    columns: (52mm, 1fr), stroke: 0.45pt + MUTED, inset: (x: 9pt, y: 4.6pt), align: (left, left),
    [stochastic depth / DropPath], [dropout at residual-branch scale; standard in deep ViTs],
    [spectral normalization], [constrain layer operator norms (GANs, sensitivity control)],
    [SAM], [optimize the worst loss in a parameter neighbourhood],
    [adversarial training], [augment with worst-case perturbations — robustness, at a cost],
  )
])

#note([One change at a time, or the comparison identifies nothing. Sealed test set, evaluated once, after every knob is frozen.], color: BLUE)

== What breaks: symptom → suspect → test #V

#align(center, [
  #set text(size: 15.5pt)
  #table(
    columns: (56mm, 52mm, 84mm), stroke: 0.45pt + MUTED,
    inset: (x: 8pt, y: 5pt), align: (left, left, left),
    table.header([symptom], [first suspect], [controlled test]),
    [train and val both high], [underfit / optimization], [fix fit first (L5–L6); regularizing makes it *worse*],
    [train $arrow.b$, val turns $arrow.t$], [overfitting over time], [restore best checkpoint; then sweep one knob],
    [val *better* than train], [train-only noise: dropout, aug, penalty in train loss], [re-measure both in `eval()` mode, penalty excluded],
    [great val, poor deployment], [validation no longer represents reality], [audit the split and the data pipeline, not the knobs],
    [confident and wrong], [target pressure], [add smoothing; *measure* calibration],
  )
])

#pause
#punch[Regularization treats a diagnosed disease. It is not a vitamin you add by default in arbitrary doses.]

== A controlled CIFAR-10 run: the baseline memorizes #V

#align(center, image("figures/cifar_regularization_curves.png", width: 94%))

#note([2,000 stratified training examples · one fixed initialization and minibatch order · same 666,538-parameter CNN and 35-epoch budget · dots mark the checkpoint selected by validation loss.], color: INK)

== Same model, one controlled change per run #V

#align(center, image("figures/cifar_regularization_summary.png", width: 92%))

#note([Crop + flip wins; dropout and smoothing help modestly; tested AdamW $lambda_"wd" = 0.01$ is neutral. Suite and rules predeclared; each checkpoint frozen by validation before its reporting-only test. One seed $eq.not$ benchmark.], color: BLUE)

== The spine, cured three independent ways #V

#spine-fig(curves: (
  (truth-curve, truth-stroke),
  (curve-of(w9), 1.5pt + RED.transparentize(45%)),
  (curve-of(w-star), 2.0pt + ACC),
  (curve-of(w-stop), 2.0pt + TEAL),
  (curve-of(w-aug), 2.0pt + GREEN),
), height: 47mm)
#align(center, text(size: 13pt)[
  #sw(RED, [none · #fnum(mse(w9, xv, yv), d: 2)]) #h(9pt)
  #sw(ACC, [decay $lambda^star$ · #fnum(mse(w-star, xv, yv), d: 3)]) #h(9pt)
  #sw(TEAL, [stop $t^star$ · #fnum(mse(w-stop, xv, yv), d: 3)]) #h(9pt)
  #sw(GREEN, [augment · #fnum(mse(w-aug, xv, yv), d: 3)]) — validation MSE
])

#pause
#punch[Three different sites, three different knobs — the *same* preference, and nearly the same cured model.]

== Freeze every choice; only then open the sealed test set #V

#align(center, [
  #set text(size: 14pt)
  #table(
    columns: (38mm, 75mm, 34mm, 34mm), stroke: 0.45pt + MUTED,
    inset: (x: 7pt, y: 5pt), align: (left, left, right, right),
    table.header([model], [choice made without test], [val MSE], [test MSE]),
    [interpolant], [unregularized reference], [#fnum(mse(w9, xv, yv), d: 3)], [#fnum(mse(w9, xt, yt), d: 3)],
    [weight decay], [$lambda^star = #str(lam-star)$ from validation], [#fnum(mse(w-star, xv, yv), d: 3)], [#fnum(mse(w-star, xt, yt), d: 3)],
    [early stop], [$t^star = #str(t-star)$ from validation], [#fnum(mse(w-stop, xv, yv), d: 3)], [#fnum(mse(w-stop, xt, yt), d: 3)],
    [input jitter], [policy fixed before test], [#fnum(mse(w-aug, xv, yv), d: 3)], [#fnum(mse(w-aug, xt, yt), d: 3)],
  )
])

#note([The test column is a report, never a steering wheel. If it changes a knob, it has silently become another validation set.], color: BLUE)

#provenance

== Exit ticket: five questions before you regularize

#align(center, grid(columns: (1fr, 1fr), gutter: 14pt,
  note([*1.* Which learning curves did you plot, and what do they diagnose?], color: BLUE),
  note([*2.* Which single site are you intervening on, and with which knob?], color: ACC),
  note([*3.* What did validation choose — and did train loss get a vote?], color: BLUE),
  note([*4.* Is the test set still sealed, to be spent exactly once?], color: INK),
  note([*5.* Which implicit regularizers moved when you changed optimizer or batch size?], color: TEAL),
  note([*Reproduce:* seed, splits, schedule, and every knob — logged.], color: GREEN),
))

#pause
#punch[If the answer to (1) is "none", the prescription precedes the diagnosis.]

== Primary sources

#[
#set text(size: 15pt)
- Zhang, Bengio, Hardt, Recht, Vinyals (2017), #link("https://arxiv.org/abs/1611.03530")[*Understanding deep learning requires rethinking generalization*] — networks can fit random labels.
- Loshchilov & Hutter (2019), #link("https://arxiv.org/abs/1711.05101")[*Decoupled Weight Decay Regularization*] — AdamW.
- Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov (2014), #link("https://jmlr.org/papers/v15/srivastava14a.html")[*Dropout*].
- Gal & Ghahramani (2016), #link(MC)[*Dropout as a Bayesian Approximation*] — the MC Dropout bridge for the next lecture.
- Zhang, Cissé, Dauphin, Lopez-Paz (2018), #link("https://arxiv.org/abs/1710.09412")[*mixup*] · Yun et al. (2019), #link("https://arxiv.org/abs/1905.04899")[*CutMix*] · Cubuk et al. (2020), #link("https://arxiv.org/abs/1909.13719")[*RandAugment*].
- Szegedy et al. (2016), #link("https://arxiv.org/abs/1512.00567")[label smoothing] · Müller, Kornblith, Hinton (2019), #link("https://arxiv.org/abs/1906.02629")[*When Does Label Smoothing Help?*].
- Bishop (1995), *Training with noise is equivalent to Tikhonov regularization*; Belkin et al. (2019), #link("https://arxiv.org/abs/1812.11118")[double descent] — the modern interpolation picture.
]

#v(4pt)
#note([Seeded-spine figures are recomputed at compile time; the opening schematic, CIFAR panels, and controlled PyTorch ablation have documented provenance and scripts. See `lecture7/figures/PROVENANCE.md`; the companion Typst lab independently checks the regression numbers.], color: INK)

#focus-slide[
  Regularization decides what the model does where the data is silent — you state the preference, validation sets the knob.
  #v(12pt)
  #set text(size: 22pt)
  Next: keep dropout stochastic at inference — turn 30 subnetworks into a predictive mean and an uncertainty signal.
  #v(8pt)
  #text(size: 17pt, fill: rgb("#B9C6C8"))[Next lecture · MC Dropout: uncertainty, calibration, and failure modes]
]
