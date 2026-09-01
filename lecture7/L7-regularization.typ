// Generalization and Regularization — Lecture 7 · ES 667 Deep Learning
// Compile from the repository root:
//   typst compile --root . --input handout=true lecture7/L7-regularization.typ slides-pdf/L7.pdf
//   typst compile --root . lecture7/L7-regularization.typ slides-pdf/L7-presentation.pdf
//
// Design: one seeded 10-point regression lab gives exact mathematics for
// diagnosis, weight decay, and early stopping. High-resolution course-owned
// teaching photographs make augmentation and mixing visible; a controlled CIFAR-10
// run later supplies measured PyTorch evidence.
// The companion lab lecture7/diagrams/verify_numbers.typ independently checks
// the computed regression numbers.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#import "../vendor/chalkdust/linalg/lib.typ" as la
#import "@preview/lilaq:0.4.0" as lq

#show: metropolis-deck.with(
  title: [Generalization and Regularization],
  subtitle: [Learning curves, regularization, and model selection],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let NB = "https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L07/"
#let PT = "https://docs.pytorch.org/docs/stable/"
#let PTV = "https://docs.pytorch.org/vision/stable/"

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
  SYNTHETIC · COMPUTED: seeded 10-point teaching case. Every curve and number on this slide is recomputed at compile time.
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

// ── exact bias–variance quantities for the fixed-design teaching example ──
// Both estimators are linear in the observed labels y. Their mean prediction is
// therefore the same fit applied to the noiseless labels f(x_i). Pointwise
// variance follows exactly from their responses to the ten unit label vectors:
//   Var_D[f_hat(x)] = sigma^2 sum_i h_i(x)^2.
#let bv-truth-labels = xs.map(truth)
#let bv-linear-fit(labels) = ml.linreg(xs.map(x => (1.0, x)), labels)
#let bv-linear-predict(w, points) = points.map(x => w.at(0) + w.at(1) * x)
#let bv-poly-predict(w, points) = points.map(x => polyval(w, x))
#let bv-linear-curve(w) = bv-linear-predict(w, XG)
#let bv-mean-1 = bv-linear-curve(bv-linear-fit(bv-truth-labels))
#let bv-mean-9 = curve-of(la.solve(VD, bv-truth-labels))
#let bv-bias-1 = range(XG.len()).map(i => bv-mean-1.at(i) - truth-curve.at(i))
#let bv-bias-9 = range(XG.len()).map(i => bv-mean-9.at(i) - truth-curve.at(i))

#let bv-unit-labels = range(NTR).map(j =>
  range(NTR).map(i => if i == j { 1.0 } else { 0.0 })
)
#let bv-response-1 = bv-unit-labels.map(labels => bv-linear-curve(bv-linear-fit(labels)))
#let bv-response-9 = bv-unit-labels.map(labels => curve-of(la.solve(VD, labels)))
#let bv-variance(responses) = range(responses.at(0).len()).map(k =>
  NOISE * NOISE * responses.map(r => calc.pow(r.at(k), 2)).sum()
)
#let bv-var-1 = bv-variance(bv-response-1)
#let bv-var-9 = bv-variance(bv-response-9)
#let bv-sd-1 = bv-var-1.map(calc.sqrt)
#let bv-sd-9 = bv-var-9.map(calc.sqrt)
#let bv-lower-1 = range(XG.len()).map(i => bv-mean-1.at(i) - bv-sd-1.at(i))
#let bv-upper-1 = range(XG.len()).map(i => bv-mean-1.at(i) + bv-sd-1.at(i))
#let bv-lower-9 = range(XG.len()).map(i => bv-mean-9.at(i) - bv-sd-9.at(i))
#let bv-upper-9 = range(XG.len()).map(i => bv-mean-9.at(i) + bv-sd-9.at(i))
#let bv-average(values) = values.sum() / values.len()
#let bv-bias2-1 = bv-average(bv-bias-1.map(v => v * v))
#let bv-bias2-9 = bv-average(bv-bias-9.map(v => v * v))
#let bv-varbar-1 = bv-average(bv-var-1)
#let bv-varbar-9 = bv-average(bv-var-9)
#let bv-noise = NOISE * NOISE
#let bv-total-1 = bv-bias2-1 + bv-varbar-1 + bv-noise
#let bv-total-9 = bv-bias2-9 + bv-varbar-9 + bv-noise

// The summary table uses the same nine validation midpoints as the running
// example, so its expected MSE is directly comparable to the observed val MSE.
#let bv-mean-val-1 = bv-linear-predict(bv-linear-fit(bv-truth-labels), xv)
#let bv-mean-val-9 = bv-poly-predict(la.solve(VD, bv-truth-labels), xv)
#let bv-bias-val-1 = range(xv.len()).map(i => bv-mean-val-1.at(i) - truth(xv.at(i)))
#let bv-bias-val-9 = range(xv.len()).map(i => bv-mean-val-9.at(i) - truth(xv.at(i)))
#let bv-response-val-1 = bv-unit-labels.map(labels =>
  bv-linear-predict(bv-linear-fit(labels), xv)
)
#let bv-response-val-9 = bv-unit-labels.map(labels =>
  bv-poly-predict(la.solve(VD, labels), xv)
)
#let bv-bias2-val-1 = bv-average(bv-bias-val-1.map(v => v * v))
#let bv-bias2-val-9 = bv-average(bv-bias-val-9.map(v => v * v))
#let bv-varbar-val-1 = bv-average(bv-variance(bv-response-val-1))
#let bv-varbar-val-9 = bv-average(bv-variance(bv-response-val-9))
#let bv-total-val-1 = bv-bias2-val-1 + bv-varbar-val-1 + bv-noise
#let bv-total-val-9 = bv-bias2-val-9 + bv-varbar-val-9 + bv-noise

// Twelve example refits are shown only to make the expectation tangible; the
// solid mean and every reported bias/variance number above are analytic.
#let bv-refits = range(12).map(s => {
  let labels = xs.enumerate().map(((i, x)) => truth(x) + NOISE * rnd.randn(SEED + 500 + s, i))
  (bv-linear-curve(bv-linear-fit(labels)), curve-of(la.solve(VD, labels)))
})

#let spine-fig(
  curves: (),                    // ((y-array, stroke), …) drawn back to front
  show-train: true, show-val: false,
  train-color: INK, val-color: BLUE,
  train-size: 8pt, val-size: 8pt,
  train-mark: "o", val-mark: "s",
  train-stroke: 0.9pt + white, val-stroke: 0.9pt + white,
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
    ..(if show-train { (lq.scatter(xs, ys, color: train-color, size: train-size,
      mark: train-mark, stroke: train-stroke),) } else { () }),
    ..(if show-val { (lq.scatter(xv, yv, color: val-color, size: val-size,
      mark: val-mark, stroke: val-stroke),) } else { () }),
  )
})

// ── the regularization map: one pipeline, five injection sites ──
// Re-drawn at every section start with that section's site highlighted, so
// students always know WHERE the current method enters the training pipeline.
#let reg-map(active: none, named: false, focus-method: none) = {
  let site(key, base, default-method) = {
    let on = active == key or active == "all" or (active == "mixing" and (key == "data" or key == "targets"))
    let method = if focus-method != none and on and active != "all" { focus-method } else { default-method }
    let tcol = if on { INK } else { MUTED }
    let label = if named {
      block(inset: (top: 1.5pt, bottom: 2pt), stack(
        // Keep the method name clear of subscripts and accents in the site label.
        dir: ttb, spacing: 10pt,
        base,
        text(size: 10pt, weight: 700, fill: if on { ACC } else { MUTED.lighten(10%) }, method),
      ))
    } else { base }
    (label, if on { 1.4pt + ACC } else { 0.8pt + MUTED.lighten(25%) },
     if on { ACC.lighten(90%) } else { white }, tcol)
  }
  let (d-l, d-s, d-f, d-t) = site("data", [data $(x, y)$], [augmentation])
  let (m-l, m-s, m-f, m-t) = site("model", [model $f_theta$ · hidden $h$], [dropout])
  let (o-l, o-s, o-f, o-t) = site("objective", [penalty $lambda Omega(theta)$], [weight decay])
  let (t-l, t-s, t-f, t-t) = site("targets", [targets $tilde(y)$], [label smoothing])
  let (l-l, l-s, l-f, l-t) = site("training", [training loop: update $theta$, stop at $t^star$], [early stopping])
  align(center, diagram(
    spacing: (if named { 35mm } else { 23mm }, if named { 11mm } else { 8.5mm }),
    node-stroke: 0.9pt + INK, node-fill: white,
    {
      let box-node(pos, label, stroke, fill, tcol) = node(pos, text(size: 12.5pt, fill: tcol)[#label],
        shape: fletcher.shapes.rect, corner-radius: 3pt,
        inset: if named { 8.5pt } else { 6.5pt }, stroke: stroke, fill: fill)
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

#title-slide()

// ═══════════════════════ PART I — THE PERFECT FIT THAT FAILS ═══════════════════════
== Training loss does not measure generalization

#align(center, [
  #text(size: 22pt)[Training loss is computed on the examples used to fit $theta$.]
  #v(14pt)
  $ min_theta cal(L)_"train" (theta) quad eq.not quad min_theta cal(L)_"new data" (theta) $
  #v(16pt)
  #grid(columns: (1fr, 1fr), gutter: 26pt,
    hairline([Training loss], [
      How well the model fits examples used during optimization.
    ], color: TEAL),
    hairline([New-data loss], [
      How well the learned rule transfers to unseen examples.
    ], color: BLUE),
  )
])

#pause
#note[Training loss can fall while new-data loss rises. Validation curves expose that gap.]

== Learning curves reveal overfitting #V

#align(center, image("figures/train_val.png", width: 62%))

#pause
#punch[$cal(L)_"train" arrow.b$ throughout, while $cal(L)_"val" arrow.b$ then $arrow.t$. Training continues, but generalization gets worse.]

== Train, validation, and test splits #D

#align(center, table(
  columns: (42mm, 72mm, 68mm), stroke: 0.5pt + MUTED,
  inset: (x: 10pt, y: 6pt), align: (left, left, left),
  table.header([split], [job], [allowed decisions]),
  [#fitc[training]], [fit parameters $theta$], [gradients; optimizer updates],
  [#valc[validation]], [choose the recipe], [$lambda$, stopping time, $p$, augmentation, $epsilon$],
  [#text(fill: INK, weight: 650)[test]], [estimate final generalization], [*none.* #h(2pt) Use once after the recipe is frozen.],
))

#pause
#warning[Using test accuracy to choose even one regularization knob turns the test set into another validation set. The running example keeps a separate dense test draw sealed until final evaluation.]

== Outline

#[
#set text(size: 22pt)
#enum(
  [*Diagnose:* find the gap on a tiny dataset with a perfect training fit],
  [*Weight decay:* penalize coefficient magnitude; tune its strength on validation],
  [*Early stopping:* use training time as a capacity control],
  [*Augmentation + input noise + mixing:* regularize in data space],
  [*Dropout:* mask hidden activations during training; disable it at evaluation],
  [*Label smoothing:* soften targets to limit confidence],
  spacing: 9pt,
)
]

== Running example: degree-9 polynomial regression #V

We measure a smooth signal at ten inputs; every reading carries noise:

$ y_i = f(x_i) + epsilon_i, quad epsilon_i tilde cal(N)(0, 0.25^2). $

#spine-fig(curves: ((truth-curve, truth-stroke),), height: 50mm, ylim: (-1.8, 1.8))
#align(center, text(size: 13pt)[
  #text(size: 17pt, fill: INK)[●] #text(fill: INK, weight: 650)[10 training measurements]
  #h(13pt) #sw(MUTED, [hidden truth $f$])
])

#note[
  From ES 335: expand features as $phi(x) = (1, x, x^2, dots, x^9)$ and fit
  $hat(y) = bold(w)^top phi(x)$. This is a linear model with 10 weights and enough flexibility to bend between samples.
]

== A degree-9 polynomial interpolates the training data #V

Ten equations, ten weights: the least-squares fit passes through *every* point.

#spine-fig(curves: ((truth-curve, truth-stroke), (curve-of(w9), 2.2pt + RED)))
#align(center, text(size: 13pt)[
  #text(size: 17pt, fill: INK)[●] #text(fill: INK, weight: 650)[training data] #h(11pt)
  #sw(RED, [degree-9 fit · train MSE #fnum(mse(w9, xs, ys), d: 4)]) #h(11pt)
  #sw(MUTED, [truth])
])

#pause
#punch[A training loss of 0.0000 shows interpolation of these points. It does not establish generalization.]

== Validation reveals poor interpolation between samples #V

#align(center, text(size: 14.5pt, weight: 600)[
  #text(size: 19pt, fill: TEAL)[●] #fitc[TRAIN]: 10 measured points used to fit
  #h(18pt)
  #text(size: 17pt, fill: BLUE)[■] #valc[VALIDATION]: 9 unseen midpoints used only to evaluate
])

#spine-fig(
  curves: ((truth-curve, truth-stroke), (curve-of(w9), 2.2pt + RED)),
  show-val: true, height: 41mm,
  train-color: TEAL, val-color: BLUE,
  train-size: 8pt, val-size: 8pt,
  train-mark: "o", val-mark: "s",
  train-stroke: 0.9pt + white, val-stroke: 0.9pt + white,
)
#align(center, text(size: 13pt)[#sw(MUTED, [hidden truth]) #h(13pt) #sw(RED, [degree-9 interpolating fit])])

#align(center, text(size: 17pt, weight: 650)[
  Same fitted polynomial:
  #h(12pt) #fitc[train MSE $#fnum(mse(w9, xs, ys), d: 4)$]
  #h(18pt) #valc[validation MSE] #failc[$#fnum(mse(w9, xv, yv), d: 2)$]
])

#pause
#note[Even the true $f$ would score $approx sigma^2 = 0.0625$ here. The interpolating polynomial scores $#fnum(mse(w9, xv, yv) / (NOISE * NOISE), d: 0) times$ worse on these unseen midpoints.]

== Diagnosing learning curves #D

Read the levels, gap, and trend of $(cal(L)_"train", cal(L)_"val")$ together:

#align(center, table(
  columns: (44mm, 50mm, 62mm, 42mm), stroke: 0.45pt + MUTED,
  inset: (x: 9pt, y: 6pt), align: (left, left, left, left),
  table.header([training], [validation], [diagnosis], [first move]),
  [high], [high, similar], [underfit or broken optimization], [repair fit (L5 to L6)],
  [low], [much higher], [#failc[overfitting]], [#knobc[regularize]],
  [low], [low, similar], [fit transfers to this split], [run stress tests],
))

#pause
#note([A small gap is not enough; both losses can be high. The running example has train MSE $#fnum(mse(w9, xs, ys), d: 2)$ and validation MSE $#fnum(mse(w9, xv, yv), d: 2)$, which matches the second row.], color: BLUE)

== Same $x$-locations, new noisy $y$-values, different fits #V

#align(center, stack(spacing: 2pt,
  text(size: 16pt)[*Fixed design:* keep the $x_i$, redraw $epsilon_i$, and get new $y_i = f(x_i) + epsilon_i$. The points move vertically.],
  text(size: 14.5pt, fill: MUTED)[A random-design experiment would redraw the $x_i$ too.],
))

#let var-curves = (7, 8, 9, 10).map(s => {
  let yy = xs.enumerate().map(((i, x)) => truth(x) + NOISE * rnd.randn(s, i))
  (yy, la.solve(VD, yy), ml.linreg(xs.map(x => (1.0, x)), yy))
})
#align(center, stack(dir: ltr, spacing: 8mm,
  {
    set text(size: 12.5pt)
    lq.diagram(
      width: 82mm, height: 42mm, xlim: (-1.06, 1.06), ylim: (-2.9, 2.9), margin: 0%,
      xlabel: [$x$], title: [degree 1 · four refits],
      xaxis: (ticks: (-1, 0, 1), subticks: none, mirror: false),
      yaxis: (ticks: (-2, 0, 2), subticks: none, mirror: false),
      lq.plot(XG, truth-curve, stroke: truth-stroke, mark: none),
      ..var-curves.map(c => lq.plot(XG, XG.map(x => c.at(2).at(0) + c.at(2).at(1) * x),
        stroke: 1.4pt + TEAL.transparentize(20%), mark: none)),
      ..var-curves.map(c => lq.scatter(xs, c.at(0), color: TEAL.transparentize(35%), size: 3.4pt)),
    )
  },
  {
    set text(size: 12.5pt)
    lq.diagram(
      width: 82mm, height: 42mm, xlim: (-1.06, 1.06), ylim: (-2.9, 2.9), margin: 0%,
      xlabel: [$x$], title: [degree 9 · four refits],
      xaxis: (ticks: (-1, 0, 1), subticks: none, mirror: false),
      yaxis: (ticks: (-2, 0, 2), subticks: none, mirror: false),
      lq.plot(XG, truth-curve, stroke: truth-stroke, mark: none),
      ..var-curves.map(c => lq.plot(XG, curve-of(c.at(1)), stroke: 1.3pt + RED.transparentize(30%), mark: none)),
      ..var-curves.map(c => lq.scatter(xs, c.at(0), color: RED.transparentize(35%), size: 3.4pt)),
    )
  },
))

#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([Rigid model], [stable across draws but wrong on average (*bias*)], color: TEAL),
  hairline([Flexible model], [changes sharply across draws (*variance*)], color: RED),
))

== Mean prediction across training sets #D

#align(center, text(size: 17pt)[At each $x$, average the predictions from many training sets:
  $mu_d thin (x) := EE_(cal(D))[hat(f)_(d, cal(D)) thin (x)]$.])

#align(center, stack(dir: ltr, spacing: 8mm,
  {
    set text(size: 12.5pt)
    lq.diagram(
      width: 82mm, height: 42mm, xlim: (-1.06, 1.06), ylim: (-2.5, 2.5), margin: 0%,
      xlabel: [$x$], title: [degree 1],
      grid: (stroke: 0.35pt + MUTED.transparentize(78%)),
      xaxis: (ticks: (-1, 0, 1), subticks: none, mirror: false),
      yaxis: (ticks: (-2, 0, 2), subticks: none, mirror: false),
      ..bv-refits.map(r => lq.plot(XG, r.at(0),
        stroke: 0.9pt + TEAL.transparentize(74%), mark: none, z-index: 1)),
      lq.plot(XG, truth-curve, stroke: truth-stroke, mark: none, z-index: 2),
      lq.plot(XG, bv-mean-1, stroke: 2.6pt + INK, mark: none, z-index: 3),
    )
  },
  {
    set text(size: 12.5pt)
    lq.diagram(
      width: 82mm, height: 42mm, xlim: (-1.06, 1.06), ylim: (-2.5, 2.5), margin: 0%,
      xlabel: [$x$], title: [degree 9],
      grid: (stroke: 0.35pt + MUTED.transparentize(78%)),
      xaxis: (ticks: (-1, 0, 1), subticks: none, mirror: false),
      yaxis: (ticks: (-2, 0, 2), subticks: none, mirror: false),
      ..bv-refits.map(r => lq.plot(XG, r.at(1),
        stroke: 0.8pt + RED.transparentize(82%), mark: none, z-index: 1)),
      lq.plot(XG, truth-curve, stroke: truth-stroke, mark: none, z-index: 2),
      lq.plot(XG, bv-mean-9, stroke: 2.6pt + INK, mark: none, z-index: 3),
    )
  },
))

#align(center, text(size: 13.5pt)[
  #text(fill: MUTED)[faint coloured curves = 12 possible fits] #h(13pt)
  #sw(INK, [exact mean prediction $mu_d$]) #h(13pt)
  #text(fill: MUTED)[dashed = truth $f$]
])

#caption[Here $cal(D)$ changes only through label noise at fixed $x_i$. The input locations stay fixed.]

== Bias is the error of the mean prediction #D

$ b_d thin (x) := mu_d thin (x) - f(x), quad "and squared bias" = (b_d thin (x))^2. $

#let bv-bias-panel(mean-curve, color, panel-title) = {
  set text(size: 12.5pt)
  lq.diagram(
    width: 82mm, height: 44mm, xlim: (-1.06, 1.06), ylim: (-1.35, 1.35), margin: 0%,
    xlabel: [$x$], title: panel-title,
    grid: (stroke: 0.35pt + MUTED.transparentize(78%)),
    xaxis: (ticks: (-1, 0, 1), subticks: none, mirror: false),
    yaxis: (ticks: (-1, 0, 1), subticks: none, mirror: false),
    lq.fill-between(XG, mean-curve, y2: truth-curve,
      fill: ACC.transparentize(82%), stroke: none, z-index: 1),
    lq.plot(XG, truth-curve, stroke: truth-stroke, mark: none, z-index: 2),
    lq.plot(XG, mean-curve, stroke: 2.5pt + color, mark: none, z-index: 3),
  )
}

#align(center, stack(dir: ltr, spacing: 8mm,
  bv-bias-panel(bv-mean-1, TEAL, [degree 1 · high bias]),
  bv-bias-panel(bv-mean-9, RED, [degree 9 · almost no bias]),
))

#align(center, text(size: 13.5pt, grid(columns: (1fr, 1fr), gutter: 22pt,
  [#fitc[degree 1] · mean $b^2$ at the 9 validation midpoints: *#fnum(bv-bias2-val-1, d: 3)*],
  [#failc[degree 9] · mean $b^2$: *$#fnum(bv-bias2-val-9 * calc.pow(10, 10), d: 1) times 10^(-10)$*],
)))

#caption[Orange shading shows signed bias. MSE uses squared bias. The tiny degree-9 bias is specific to this smooth signal and fixed design.]

== Variance measures sensitivity to label noise #D

$ "Var"_cal(D)[hat(f)_d thin (x)] := EE_cal(D)[(hat(f)_d thin (x) - mu_d thin (x))^2]. $

#let bv-band-panel(mean-curve, lower, upper, color, panel-title) = {
  set text(size: 12.5pt)
  lq.diagram(
    width: 82mm, height: 44mm, xlim: (-1.06, 1.06), ylim: (-2.5, 2.5), margin: 0%,
    xlabel: [$x$], title: panel-title,
    grid: (stroke: 0.35pt + MUTED.transparentize(78%)),
    xaxis: (ticks: (-1, 0, 1), subticks: none, mirror: false),
    yaxis: (ticks: (-2, 0, 2), subticks: none, mirror: false),
    lq.fill-between(XG, lower, y2: upper,
      fill: color.transparentize(80%), stroke: none, z-index: 1),
    lq.plot(XG, truth-curve, stroke: truth-stroke, mark: none, z-index: 2),
    lq.plot(XG, mean-curve, stroke: 2.5pt + color, mark: none, z-index: 3),
  )
}

#align(center, stack(dir: ltr, spacing: 8mm,
  bv-band-panel(bv-mean-1, bv-lower-1, bv-upper-1, TEAL, [degree 1 · narrow spread]),
  bv-band-panel(bv-mean-9, bv-lower-9, bv-upper-9, RED, [degree 9 · wide spread]),
))

#align(center, text(size: 13.5pt, grid(columns: (1fr, 1fr), gutter: 22pt,
  [#fitc[degree 1] · mean variance at the 9 validation midpoints: *#fnum(bv-varbar-val-1, d: 3)*],
  [#failc[degree 9] · mean variance: *#fnum(bv-varbar-val-9, d: 3)*],
)))

#caption[Shading shows $mu_d thin (x) plus.minus 1$ SD across training sets. It is not a prediction interval for a new noisy target.]

== Expected test MSE has three components #D

For a fresh target $Y = f(x) + epsilon_"test"$, with independent zero-mean noise:

#align(center, text(size: 18.5pt)[$
  EE_(cal(D), epsilon_"test") [(hat(f)_d thin (x) - Y)^2] =
  underbrace((b_d thin (x))^2, #knobc[squared bias])
  + underbrace("Var"_cal(D)[hat(f)_d thin (x)], #failc[variance])
  + underbrace(sigma^2, #text(fill: MUTED, weight: 650)[noise])
$])

#align(center, [
  #set text(size: 14.5pt)
  #table(
    columns: (36mm, 38mm, 38mm, 32mm, 42mm), stroke: 0.5pt + MUTED,
    inset: (x: 8pt, y: 5.5pt), align: (left, center, center, center, center),
    table.header([model], [#knobc[mean bias²]], [#failc[mean variance]], [noise], [expected MSE]),
    [degree 1], [$#fnum(bv-bias2-val-1, d: 3)$], [$#fnum(bv-varbar-val-1, d: 3)$], [$#fnum(bv-noise, d: 3)$], [$#fnum(bv-total-val-1, d: 3)$],
    [degree 9], [$approx 0$], [$#fnum(bv-varbar-val-9, d: 3)$], [$#fnum(bv-noise, d: 3)$], [$#fnum(bv-total-val-9, d: 3)$],
  )
])

#caption[These are exact expectations at the same nine validation midpoints. Cross-terms vanish under independent zero-mean noise. The observed MSE $1.70$ is one realization.]

#pause
#punch[The degree-9 fit removes bias here, but its high variance raises the expected test error.]

== Many functions can interpolate the training data #V

Narrow radial bumps through the same ten points *also* score $0.0000$:

#spine-fig(curves: (
  (truth-curve, truth-stroke),
  (curve-of(w9), 1.9pt + RED),
  (XG.map(rbfval), 1.9pt + TEAL),
), height: 38mm)
#align(center, text(size: 13pt)[#sw(RED, [polynomial interpolant]) #h(12pt) #sw(TEAL, [narrow-bump interpolant]) #h(12pt) both have train MSE $= 0$])

#pause
#punch[Both fits agree at the 10 training points but make different predictions between them.]

#note[
  #set text(size: 17.5pt)
  Large neural networks can memorize randomly shuffled labels (Zhang et al., 2017).
  Zero training error shows *capacity* but says nothing about generalization.
  Architecture, optimization, augmentation, weight decay, and stopping determine
  which zero-error predictor training finds.
]

== Weight decay changes the objective #V

Regularizers differ mainly in where they enter training. Weight decay adds a preference for smaller parameters to the objective:

#v(3pt)
#reg-map(active: "objective", named: true)

#v(10pt)
#pause
#punch[Make large weights costly, steering training away from coefficient-heavy fits that match noise; #valc[validation] chooses $lambda$.]

// ═══════════════════════ PART II — WEIGHT DECAY ═══════════════════════

== $L_2$ regularization adds a norm penalty #D

$ cal(L)_"data"(w) = 1/(2n) norm(X w - y)_2^2, quad
  cal(J)(w) = underbrace(cal(L)_"data"(w), #fitc[fit]) + underbrace(lambda_"wd" / 2 norm(w)_2^2, #knobc[shrink]) $

#pause
#note[
  *Convention:* the $1/2$ factors make $nabla cal(J) = X^top(X w - y)/n + lambda_"wd" w$.
  Lecture 1 derived the Gaussian-prior view. Here we examine how the same term changes the fitted function and the optimizer update.
  #text(size: 14pt, fill: MUTED)[Revisit: #link(IA + "map-regularization/")[interactive-articles/map-regularization]]
]

== The interpolating polynomial has large coefficients #V

The zero-training-loss polynomial uses large coefficients that cancel each other:

#align(center, bars(
  w9.map(v => calc.round(v, digits: 1)),
  labels: range(10).map(k => [$w_#k$]),
  annotate: false, bar: 8mm, span: 36mm,
  color: (i, v) => RED.transparentize(12%),
))
#caption[Degree-9 interpolant: largest coefficient magnitude $=#fnum(calc.max(..w9.map(calc.abs)), d: 0)$; $norm(bold(w))_2 = #fnum(la.norm(w9), d: 0)$.]

#pause
#punch[In this polynomial basis, the oscillations require large coefficients, so an #knobc[$L_2$ penalty] assigns that fit a large cost.]

== $lambda$ controls the strength of weight decay #V

The data and features stay fixed. The legend reads: chosen $lambda arrow.r$ resulting $norm(bold(w))_2$.

#spine-fig(curves: (
  (truth-curve, truth-stroke),
  (curve-of(w9), 1.8pt + RED.transparentize(25%)),
  (curve-of(ridge-fit(1.0)), 1.8pt + MUTED),
  (curve-of(w-star), 2.3pt + GREEN),
))
#align(center, text(size: 13pt)[
  #sw(RED, [$lambda = 0 arrow.r norm(bold(w))_2 = #fnum(la.norm(w9), d: 0)$]) #h(10pt)
  #sw(GREEN, [$lambda = 10^(-3) arrow.r norm(bold(w))_2 = #fnum(la.norm(w-star), d: 1)$]) #h(10pt)
  #sw(MUTED, [$lambda = 1 arrow.r norm(bold(w))_2 = #fnum(la.norm(ridge-fit(1.0)), d: 1)$])
])

#pause
#punch[Increasing $lambda$ trades data fit against coefficient size. At $lambda=0$ the model interpolates; too much shrinkage underfits.]

== Checkpoint: which metric should choose $lambda$? #Q

The sweep below is computed from the running example:

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
  [train MSE, because it is the objective we optimize],
  [validation MSE],
  [$norm(bold(w))_2$, because smaller seems safer],
  [none; average all three columns],
)

== Answer: choose $lambda$ on validation data #A

#mcq-answer(
  [B],
  [validation MSE picks $lambda^star = 10^(-3)$],
  [in this ridge sweep the *unpenalized* training MSE is minimized at $lambda = 0$; the weight norm is a mechanism check, not a target.],
)

#pause
#punch[Use the same procedure for every regularizer: sweep the knob and select by #valc[validation].]

== Sweep $lambda$ and choose the lowest validation loss #V

#align(center, lines(
  x: lam-grid.map(l => calc.log(l)),
  y: (lam-rows.map(r => r.at(1)), lam-rows.map(r => r.at(2))),
  labels: ([train], [validation]), colors: (TEAL, BLUE), log-y: true, legend: "tl",
  vlines: ((calc.log(lam-star), [$lambda^star$], ACC),),
  x-label: [$log_10 lambda$], y-label: [MSE], size: (146mm, 44mm),
))

#note[
  Too little weight decay leaves a high-variance fit; too much makes the model underfit.
  Search $lambda$ on a log scale and keep the value with the lowest validation loss. Real curves may be noisy or flat.
]

== Weight decay shrinks weights before the data update #V

// Ported from the paired OLS/ridge update geometry in
// ~/git/ml-teaching/supervised/slides/ridge-regression.tex.
#let wd-step-panel(decay: false) = {
  let tx = 2.55
  let ty = 2.35
  // Give the data-gradient step a deliberately different direction from the
  // radial shrink, so the head-to-tail construction is readable at a glance.
  let dx = 1.00
  let dy = -1.15
  // Exaggerate the shrink geometrically so the intermediate and resultant
  // vectors remain legible at lecture distance; the equation above is exact.
  let sx = if decay { 0.56 * tx } else { tx }
  let sy = if decay { 0.56 * ty } else { ty }
  let fx = sx + dx
  let fy = sy + dy

  align(center, lq.diagram(
    width: 78mm, height: 46mm,
    xlim: (-0.15, 3.85), ylim: (-0.15, 2.90), margin: 0%,
    xlabel: [$theta_1$], ylabel: [$theta_2$],
    grid: (stroke: 0.35pt + MUTED.transparentize(80%)),
    xaxis: (ticks: none, subticks: none, mirror: false),
    yaxis: (ticks: none, subticks: none, mirror: false),

    // The parameter vector at the start of the update.
    lq.quiver((0.0,), (0.0,), (x, y) => (tx, ty), pivot: start, scale: 1,
      stroke: 2.0pt + INK.transparentize(25%)),

    ..(if decay {
      (
        // Radial shrink, followed head-to-tail by the same data-gradient vector as at left.
        lq.quiver((tx,), (ty,), (x, y) => (sx - tx, sy - ty), pivot: start, scale: 1,
          stroke: 2.8pt + ACC),
        lq.quiver((sx,), (sy,), (x, y) => (dx, dy), pivot: start, scale: 1,
          stroke: 2.8pt + TEAL),
        // The origin-to-final arrow is the resultant parameter vector.
        lq.quiver((0.0,), (0.0,), (x, y) => (fx, fy), pivot: start, scale: 1,
          stroke: 3.2pt + GREEN),
        lq.scatter((tx,), (ty,), color: INK, size: 8pt, mark: "o", stroke: 1.1pt + white),
        lq.scatter((sx,), (sy,), color: ACC, size: 6pt, mark: "o", stroke: 0.8pt + white),
        lq.scatter((fx,), (fy,), color: GREEN, size: 6pt, mark: "o", stroke: 0.8pt + white),
        lq.place(2.72, 2.55, text(size: 12pt, weight: 650)[$bold(theta)_t$]),
        lq.place(2.25, 2.21, box(fill: white, inset: 2pt,
          text(size: 11.5pt, weight: 650, fill: ACC)[$-eta lambda bold(theta)_t$])),
        lq.place(1.84, 1.86, box(fill: white, inset: 2pt,
          text(size: 11.5pt, weight: 650, fill: ACC)[$tilde(bold(theta))_t$])),
        lq.place(2.08, 0.76, box(fill: white, inset: 2pt,
          text(size: 11.5pt, weight: 650, fill: TEAL)[$-eta bold(g)_t$])),
        lq.place(1.52, 0.35, box(fill: white, inset: 2pt,
          text(size: 12pt, weight: 700, fill: GREEN)[$bold(theta)_(t+1)$])),
      )
    } else {
      (
        lq.quiver((tx,), (ty,), (x, y) => (dx, dy), pivot: start, scale: 1,
          stroke: 2.8pt + TEAL),
        // The origin-to-final arrow completes the vector sum.
        lq.quiver((0.0,), (0.0,), (x, y) => (fx, fy), pivot: start, scale: 1,
          stroke: 3.2pt + GREEN),
        lq.scatter((tx,), (ty,), color: INK, size: 8pt, mark: "o", stroke: 1.1pt + white),
        lq.scatter((fx,), (fy,), color: GREEN, size: 6pt, mark: "o", stroke: 0.8pt + white),
        lq.place(2.72, 2.55, text(size: 12pt, weight: 650)[$bold(theta)_t$]),
        lq.place(3.03, 1.72, box(fill: white, inset: 2pt,
          text(size: 11.5pt, weight: 650, fill: TEAL)[$-eta bold(g)_t$])),
        lq.place(1.92, 0.48, box(fill: white, inset: 2pt,
          text(size: 12pt, weight: 700, fill: GREEN)[$bold(theta)_(t+1)$])),
      )
    }),

    lq.scatter((0.0,), (0.0,), color: INK, size: 5pt, mark: "o", stroke: 0.8pt + white),
    lq.place(0.10, 0.10, text(size: 11pt, fill: MUTED)[$bold(0)$]),
  ))
}

#grid(
  columns: (1fr, 1fr), gutter: 18pt,
  align(center, stack(spacing: 3pt,
    text(size: 17pt, weight: 700, fill: TEAL)[Plain SGD],
    text(size: 15.5pt)[$bold(theta)_(t+1) = bold(theta)_t - eta bold(g)_t$],
    wd-step-panel(),
  )),
  align(center, stack(spacing: 3pt,
    text(size: 17pt, weight: 700, fill: ACC)[$L_2$-regularized SGD],
    text(size: 14.5pt)[$tilde(bold(theta))_t = (1-eta lambda) bold(theta)_t, quad bold(theta)_(t+1) = tilde(bold(theta))_t - eta bold(g)_t$],
    wd-step-panel(decay: true),
  )),
)

#caption[#knobc[Orange retraces grey toward 0]. The same #fitc[teal step] then turns sharply. #keepc[Green is the result].]

#pause
#punch[The regularized update first shrinks $bold(theta)$ toward $bold(0)$, then applies the usual data-gradient move.]

== Deriving the $L_2$-SGD update #D

The penalty's gradient is $lambda bold(theta)_t$, so the vector update factors:

$ bold(theta)_(t+1) = bold(theta)_t - eta (bold(g)_t + lambda bold(theta)_t) = underbrace((1 - eta lambda) bold(theta)_t, #knobc[shrink first]) - underbrace(eta bold(g)_t, #fitc[then fit]) $

For one coordinate, let $theta_t = 5$, $g_t = 3$, $eta = 0.1$, and $lambda = 0.2$:

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 4.5pt),
  align: (left, left, center),
  table.header([piece], [substitution], [value]),
  [#knobc[shrink]], [$(1 - 0.1 dot 0.2) times 5 = 0.98 times 5$], [$4.90$],
  [#fitc[fit]], [$-0.1 times 3$], [$-0.30$],
  [$theta_(t+1)$], [$4.90 - 0.30$], [$4.60$],
))

#pause
#punch[Each coordinate is multiplied by $0.98$, then the data gradient contributes $-0.30$.]

== $L_2$ regularization with PyTorch SGD

#text(size: 17pt)[With `momentum=0`, `weight_decay=λ` is the coupled $L_2$ penalty from the previous slide:]

#align(center, text(size: 18pt)[$p arrow.l p - "lr" (p."grad" + "weight_decay" dot p)$])

#grid(columns: (1fr, 1fr), gutter: 14pt,
  hairline([Reproduce the $5.00 arrow.r 4.60$ step], [
    #codebox(size: 11.8pt)[```python
w = nn.Parameter(torch.tensor([5.0]))
opt = torch.optim.SGD(
    [w], lr=0.1, momentum=0.0,
    weight_decay=0.2,
)

w.grad = torch.tensor([3.0])  # data gradient
opt.step()
w.item()                      # 4.60
```]
  ], color: ACC),
  hairline([Use it in the training loop], [
    #codebox(size: 11.8pt)[```python
opt = torch.optim.SGD(
    model.parameters(), lr=1e-2,
    momentum=0.0, weight_decay=1e-4,
)

for x, y in train_loader:
    opt.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)  # data loss
    loss.backward()
    opt.step()                     # uses grad + λp
```]
  ], color: TEAL),
)

#note[
  #set text(size: 13.5pt)
  Use either optimizer `weight_decay` or add $(lambda slash 2) norm(theta)_2^2$ to the loss. Do not use both.
  With optimizer-side decay, `loss.item()` omits the penalty. Every parameter in that group decays.
  #h(3pt)#link(PT + "generated/torch.optim.SGD.html")[PyTorch SGD API]
]

== $L_1$ regularization and sparsity #V

#align(center, table(
  columns: (36mm, 46mm, 56mm, 44mm), stroke: 0.45pt + MUTED,
  inset: (x: 9pt, y: 6pt), align: (left, left, left, left),
  table.header([penalty], [gradient force], [pressure], [role in DL]),
  [$lambda norm(theta)_2^2 slash 2$], [$lambda theta$ (proportional)], [shrink everything a little], [default],
  [$lambda norm(theta)_1$], [$lambda "sign"(theta)$ (subgradient)], [promote sparse solutions], [specialized],
))

#note[
  From Lecture 1: $L_2$ is a Gaussian prior, $L_1$ a Laplace prior. Ordinary SGD need not land exactly on zero; proximal/thresholding methods or explicit pruning create exact zeros. In standard dense networks, $L_1$ is secondary.
]

#place(bottom + center, dy: -2pt, caption[
  Interactive: adjust $lambda$ at #link(IA + "weight-decay/")[interactive-articles/weight-decay]
])

// ═══════════════════════ PART III — EARLY STOPPING ═══════════════════════
== Early stopping changes the training loop #V

#reg-map(active: "training", named: true)

#v(10pt)
#pause
#punch[Keep the validation winner before later updates overfit the training set; patience decides when to stop.]

== Watch training and validation loss together #V

Same model, *no penalty*. Start gradient descent from $bold(w) = 0$ and log both losses:

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
#punch[#fitc[Training loss] keeps falling. #valc[Validation loss] reaches its minimum at $t^star$ and then rises, giving a U-shaped curve against training time.]

== Keep the checkpoint with lowest validation loss #D

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

== Patience decides when to stop #V

Example: `patience = 3`, `min_delta = 0`:

#align(center, [
  #set text(size: 14.5pt)
  #table(
    columns: (38mm, 31mm, 31mm, 31mm, 31mm),
    stroke: 0.5pt + MUTED, inset: (x: 9pt, y: 7pt), align: center,
    table.header([epoch], [20], [21], [22], [23]),
    [#valc[validation loss]], [#keepc[0.570]], [0.575], [0.574], [#text(fill: RED)[0.580]],
    [`wait`], [0], [1], [2], [#text(fill: RED, weight: "bold")[3]],
    [action], [#keepc[save]], [continue], [continue], [#text(fill: RED, weight: "bold")[stop]],
  )
])

#note[
  `patience = P` allows $P$ consecutive validation checks without enough improvement.
  `min_delta` sets the improvement required to reset `wait`. A positive value ignores tiny fluctuations.
]

#punch[Stop after epoch 23 and restore epoch 20. Patience controls *when training stops*; validation loss still decides which checkpoint to keep.]

== PyTorch: save, wait, restore #V

#codebox(size: 12.2pt)[```python
from copy import deepcopy
patience, min_delta = 5, 0.0
best_val, bad_epochs, best_state = float("inf"), 0, None

for epoch in range(max_epochs):
    train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)

    if val_loss < best_val - min_delta:
        best_val, bad_epochs = val_loss, 0
        best_state = deepcopy(model.state_dict())
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            break

model.load_state_dict(best_state)  # restore; do not keep the final epoch
```]

#note[`patience = 5` allows five consecutive validation checks without sufficient improvement. `min_delta = 0` keeps the strict $arg min$. Use a positive value when tiny gains should not reset the clock.]

== Why validation loss eventually rises #V

#v(8pt)
#align(center, [
  #set text(size: 15pt)
  #table(
    columns: (38mm, 92mm, 1fr),
    stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 10pt), align: (left, left, center),
    table.header([training stage], [what the model is fitting], [what the losses do]),
    [#text(fill: TEAL, weight: "bold")[earlier epochs]],
      [patterns repeated across many examples, which are useful on new data],
      [#fitc[train $arrow.b$] #h(8pt) #valc[validation $arrow.b$]],
    [#text(fill: RED, weight: "bold")[later epochs]],
      [training-specific quirks, rare cases, and noise],
      [#fitc[train $arrow.b$] #h(8pt) #valc[validation $arrow.t$]],
  )
])

#note[This ordering is common but not guaranteed. The validation curve shows whether more training still helps on unseen data.]

#punch[When validation loss starts to rise, restore the checkpoint from its minimum.]

== Early stopping limits what the model can fit #V

Three checkpoints from the *same run*, with no penalty:

#spine-fig(curves: (
  (truth-curve, truth-stroke),
  (curve-of(wt(100000000)), 1.8pt + RED.transparentize(25%)),
  (curve-of(wt(100)), 1.8pt + MUTED),
  (curve-of(w-stop), 2.3pt + GREEN),
))
#align(center, text(size: 13pt)[
  #sw(MUTED, [$t = 100$ · still underfitting]) #h(10pt)
  #sw(GREEN, [$t^star = #str(t-star)$ · kept]) #h(10pt)
  #sw(RED, [$t = 10^8$ · fits the noise])
])

#pause
#punch[At the chosen stopping time, the model has not yet fitted the weak directions that carry much of the noise.]

== Exercise: predict the learning curves #I

#interbox(link-to: NB + "01_overfitting_and_early_stopping.ipynb")[
  In this NumPy lab, a flexible model overfits a small dataset. Log both curves and restore the best checkpoint.
]

Before executing each cell, write down:

- the epoch range containing the validation minimum;
- a *patience* value that would catch it;
- how doubling the noise changes test error at the restored checkpoint.

// ═══════════════════ PART IV — AUGMENTATION AND MIXUP ═══════════════════
== Data augmentation changes the training examples #V

#reg-map(active: "data", named: true)

#v(10pt)
#pause
#punch[Show task-valid versions of each sample, so the model cannot rely on accidental details of one stored form; update targets when needed.]

== TorchVision: original image and random crop #V

#let aug-code-panel(path, label, code, color: TEAL) = block(width: 100%, [
  #align(center, text(size: 15pt, weight: 700, fill: color)[#label])
  #v(2pt)
  #align(center, image(path, width: 54mm))
  #v(4pt)
  #block(width: 100%, fill: rgb("#F4F2EC"), radius: 4pt, inset: 5pt,
    align(center, text(size: 12pt)[#raw(code, lang: "python")]))
])

#align(center, text(size: 11.5pt, fill: MUTED)[
  `from torchvision.transforms import v2`
])
#v(2pt)
#grid(
  columns: (1fr, 1fr), gutter: 18mm,
  aug-code-panel("figures/augmentation_original_v2_hd.png", [original pixels], "v2.Identity()"),
  aug-code-panel("figures/augmentation_crop_v2_hd.png", [one crop draw], "v2.RandomResizedCrop(512, scale=(.45, .45))", color: ACC),
)

#note[
  #set text(size: 13.5pt)
  Cropping changes position and scale. The label remains valid only while the object stays visible.
]

== TorchVision: horizontal flip and colour jitter #V

#align(center, text(size: 11.5pt, fill: MUTED)[
  `from torchvision.transforms import v2`
])
#v(2pt)
#grid(
  columns: (1fr, 1fr), gutter: 18mm,
  aug-code-panel("figures/augmentation_flip_v2_hd.png", [one flip draw], "v2.RandomHorizontalFlip(p=.5)"),
  aug-code-panel("figures/augmentation_colour_v2_hd.png", [one colour-jitter draw], "v2.ColorJitter(brightness=.4, saturation=.6, hue=.2)", color: ACC),
)

#note[
  #set text(size: 13.5pt)
  A flip assumes left and right do not matter. Colour jitter assumes that illumination and tint do not matter.
]

== TorchVision: blur and rotation #V

#align(center, text(size: 11.5pt, fill: MUTED)[
  `from torchvision.transforms import v2`
])
#v(2pt)
#grid(
  columns: (1fr, 1fr), gutter: 18mm,
  aug-code-panel("figures/augmentation_blur_v2_hd.png", [fixed blur], "v2.GaussianBlur(21, sigma=4.)"),
  aug-code-panel("figures/augmentation_rotate_v2_hd.png", [one rotation draw], "v2.RandomRotation(20, fill=(247, 245, 240))", color: ACC),
)

#note[
  #set text(size: 13.5pt)
  Start with mild transforms, tune them on validation data, and inspect a transformed batch.
]

#place(bottom + center, dy: -2pt, caption[
  Each panel was rendered by the TorchVision v2 call shown above.
])

== One stored example becomes many valid training views #V

#punch[Augmentation asks for the right answer across a #knobc[family of task-valid views].]

#v(14pt)
#grid(
  columns: (1fr, 9mm, 1fr, 9mm, 1fr), gutter: 2mm,
  align(center + horizon, stack(spacing: 6pt,
    text(size: 14pt, weight: 750, fill: TEAL)[STORED EXAMPLE],
    text(size: 23pt, fill: TEAL)[$(x_i, y_i)$],
    text(size: 15pt, fill: MUTED)[one file and its target],
  )),
  align(center + horizon, text(size: 25pt, fill: MUTED)[→]),
  align(center + horizon, stack(spacing: 6pt,
    text(size: 14pt, weight: 750, fill: ACC)[FRESH VALID VIEW],
    text(size: 23pt, fill: ACC)[$tau_i tilde cal(T)$],
    text(size: 15pt, fill: MUTED)[crop, flip, jitter, ...],
  )),
  align(center + horizon, text(size: 25pt, fill: MUTED)[→]),
  align(center + horizon, stack(spacing: 6pt,
    text(size: 14pt, weight: 750, fill: TEAL)[TRAIN ON THIS VIEW],
    text(size: 20pt)[$(x_i prime, y_i prime) = tau_i(x_i, y_i)$],
    text(size: 15pt, fill: MUTED)[a new draw on the next load],
  )),
)

#v(18pt)
#note[The files do not multiply. The *stream of views* does: loading the same item again draws a new $tau_i$.]

#place(bottom + center, dy: -2pt, caption[
  For classification, usually $y_i prime = y_i$. For detection and segmentation, move boxes or masks with the image.
])

== A minibatch samples one fresh view per item #D

#punch[Draw independently for each item, compute its transformed loss, then average.]

#v(10pt)
#grid(
  columns: (0.92fr, 1.18fr), gutter: 24pt,
  hairline([FOR EACH $i = 1, dots, B$], [
    #align(center, stack(spacing: 11pt,
      text(size: 20pt, fill: TEAL)[$(x_i, y_i) tilde cal(D)_"train"$],
      text(size: 20pt, fill: ACC)[$tau_i tilde cal(T)$],
      text(size: 18pt)[$(x_i prime, y_i prime) = tau_i(x_i, y_i)$],
    ))
  ], color: ACC),
  hairline([AVERAGE THE $B$ LOSSES], [
    #align(center, text(size: 20pt)[
      #text(fill: TEAL)[$hat(cal(L))_B$]
      $= 1/B sum_(i=1)^B ell(f_theta(x_i prime), y_i prime)$
    ])
    #v(10pt)
    #text(size: 15pt, fill: MUTED)[Each term uses one randomly transformed view of one minibatch item.]
  ], color: TEAL),
)

#v(14pt)
#note[This random minibatch loss is the #knobc[Monte Carlo estimate] used by SGD. A new batch produces a new estimate.]

== Repeated minibatches estimate the augmented objective #D

#punch[Average over #fitc[training examples] and over #knobc[valid transformations].]

#v(6pt)
#align(center, text(size: 19pt)[
  #text(fill: TEAL)[$EE[hat(cal(L))_B]$]
  $=$ $cal(L)_"aug"(theta)$
])
#v(4pt)
#align(center, text(size: 21pt)[
  $cal(L)_"aug"(theta) =$
  #text(fill: TEAL)[$EE_((x, y) tilde cal(D)_"train")$]
  #text(fill: ACC)[$EE_(tau tilde cal(T))$]
  $[ell(f_theta(x prime), y prime)]$
])
#align(center, text(size: 16pt, fill: ACC)[$(x prime, y prime) = tau(x, y)$])

#v(6pt)
#grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([INNER AVERAGE], [
    #text(size: 14pt)[For one example, average its loss over task-valid views $tau tilde cal(T)$.]
  ], color: ACC),
  hairline([OUTER AVERAGE], [
    #text(size: 14pt)[Then average across examples from the training distribution.]
  ], color: TEAL),
)

#place(bottom + center, dy: -2pt, caption[
  We cannot enumerate every valid view. Repeated random minibatches approximate both averages.
])

== Check whether each transform preserves the label #V

For pixel letters with classes ${a, b, c, d, e}$, check what each transform claims:

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
  stack(spacing: 4pt, glyph(GB), text(size: 13pt)[original, label #keepc[“b”]]),
  stack(spacing: 4pt, glyph(shift-right(GB)), text(size: 13pt)[shifted, still #keepc[“b”]]),
  stack(spacing: 4pt, glyph(hflip(GB)), text(size: 13pt)[flipped, now reads #failc[“d”]]),
))

#pause
#punch[Horizontal flip preserves flower classes, but it can change a letter's class.]

== Checkpoint: valid transformations for letters #Q

#mcq(
  [Which transformation preserves the label for every letter?],
  [horizontal flip, because left/right orientation should not affect the class],
  [rotate by $180 degree$, because orientation should not affect the class],
  [shift by one or two pixels, keeping the glyph in frame],
  [any transform that makes training loss go up],
)

#note([Choose an answer, then give a counterexample for each rejected option.], color: BLUE)

== Answer: only the shift preserves every label #A

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
    table.header([transform], [what the task must ignore], [when the label breaks]),
    [random crop], [position and scale (a little)], [the crop removes the object],
    [horizontal flip], [left-right orientation], [text, digits, anatomy (the heart has a side)],
    [colour jitter], [illumination and tint], [the label depends on colour],
  )
])
#caption[For detection, geometric transforms must also move boxes and masks. Transform the *supervision* with the image.]

== Specify what happens to the target #D

#align(center, text(size: 22pt)[
  $ A_tau(x, y) = (g_tau(x), h_tau(y)) $
])
#caption[One sampled transform $tau$ determines both the input map and the supervision map.]

#v(7pt)
#align(center, grid(columns: (47mm, 1fr), gutter: 14pt, row-gutter: 11pt, align: (right, left),
  [#text(size: 15pt, weight: 750, fill: TEAL)[KEEP TARGET]],
  [#text(size: 16pt)[$h_tau(y) = y$. The target is invariant, as with an object class under a safe crop.]],
  [#text(size: 15pt, weight: 750, fill: ACC)[UPDATE TARGET]],
  [#text(size: 16pt)[$h_tau(y) != y$, but the update is known. Coordinates move, timestamps shift, or a signed value flips.]],
  [#text(size: 15pt, weight: 750, fill: RED)[REJECT OR RELABEL]],
  [#text(size: 16pt)[There is no trustworthy $h_tau$. The transform changes the meaning or removes information needed to determine the target.]],
))

#pause
#punch[The *task* defines $h_tau$. An image library cannot infer what the target means.]

== The same flip requires different target updates #V

#let cyclist-path = "figures/sources/label_contract_cyclist_imagen.png"
#let cyclist-panel(flip: false) = block(
  width: 76mm, height: 50mm, clip: true,
  stroke: 0.7pt + MUTED, radius: 2pt,
  [
    #place(top + left, if flip {
      scale(x: -100%, origin: center, image(cyclist-path, width: 76mm, height: 50mm, fit: "cover"))
    } else {
      image(cyclist-path, width: 76mm, height: 50mm, fit: "cover")
    })
  ],
)

#align(center, grid(columns: (76mm, 76mm), gutter: 12mm,
  stack(spacing: 3pt, cyclist-panel(), align(center, text(size: 13pt, fill: MUTED)[as stored, travelling #keepc[right]])),
  stack(spacing: 3pt, cyclist-panel(flip: true), align(center, text(size: 13pt, fill: MUTED)[same pixels flipped, travelling #failc[left]])),
))

#v(5pt)
#align(center, [
  #set text(size: 14.5pt)
  #table(
    columns: (39mm, 51mm, 75mm), stroke: 0.45pt + MUTED,
    inset: (x: 8pt, y: 4pt), align: (left, left, left),
    table.header([task], [target before], [target after the same flip]),
    [object class], [cyclist], [#keepc[cyclist], unchanged],
    [travel direction], [right], [#failc[left], changed],
    [detection box], [$[x_1,y_1,x_2,y_2]$], [$[W-x_2,y_1,W-x_1,y_2]$, moved],
  )
])

#pause
#punch[The pixel operation is identical. The prediction task determines how the target changes.]

== Flip boxes and keypoints with the image #V

#let runner-path = "figures/sources/label_contract_runner_imagen.png"
#let pose-points = (
  (27.7mm, 2.6mm), (35.4mm, 10.8mm), (28.8mm, 14.0mm),
  (41.6mm, 32.0mm), (30.3mm, 34.6mm), (29.5mm, 45.2mm),
  (36.0mm, 38.8mm), (35.4mm, 51.7mm),
)
#let runner-panel = block(
  width: 72mm, height: 58mm, clip: true,
  stroke: 0.7pt + MUTED, radius: 2pt,
  [
    #place(top + left, image(runner-path, width: 72mm, height: 58mm, fit: "cover"))
    #place(top + left, dx: 26mm, dy: 1.5mm, rect(width: 18mm, height: 53mm, stroke: 1.7pt + GREEN, radius: 1pt))
    #for p in pose-points {
      place(top + left, dx: p.at(0) - 1.25mm, dy: p.at(1) - 1.25mm,
        circle(radius: 1.25mm, fill: BLUE, stroke: 0.5pt + white))
    }
  ],
)

#align(center, grid(columns: (76mm, 1fr), gutter: 20pt,
  [
    #align(center, runner-panel)
    #align(center, text(size: 12.5pt, fill: MUTED)[course-generated source with exact overlays added in the deck])
  ],
  [#align(left, [
    #set text(size: 15pt)
    *Boxes with half-open `XYXY` coordinates*
    #v(3pt)
    $ [x_1,y_1,x_2,y_2] arrow.r [W-x_2,y_1,W-x_1,y_2] $
    #v(6pt)
    Crop or rotate: transform the corners, clip to the canvas, then drop invalid boxes *and their matching labels*.
    #v(10pt)
    *Keypoints*
    #v(3pt)
    $ p_j prime prop H_tau p_(pi_tau(j)) $
    #v(4pt)
    $H_tau$ moves coordinates; $pi_tau$ swaps semantic left/right indices when a flip requires it. Update visibility when a point leaves the frame.
  ])],
))

#pause
#warning[A geometric transform applied only to the image produces incorrect supervision.]

#let contract-panel(title, body, color: TEAL) = block(
  width: 100%, radius: 3pt,
  stroke: 0.85pt + color, inset: 9pt,
  [
    #text(size: 15.3pt, weight: 700, fill: color)[#title]
    #v(5pt)
    #body
  ],
)
#let label-pill(body, color: GREEN) = box(
  fill: color, radius: 9pt, inset: (x: 7pt, y: 2.5pt),
  text(size: 10.8pt, weight: 700, fill: white)[#body],
)

== For sentiment, preserve meaning and polarity #V

#punch[A paraphrase may keep the label. A negation changes it.]

#v(10pt)
#align(center, block(
  width: 165mm, radius: 2pt, fill: luma(246), inset: 9pt,
  [#align(center, [#text(size: 17pt, weight: 650)[“The movie was fantastic!”] #h(6pt) #label-pill[POSITIVE]])],
))

#v(8pt)
#align(center, text(size: 19pt, fill: MUTED)[↙ #h(78mm) ↘])
#v(2pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 18pt,
  [#block(width: 100%, height: 48mm, radius: 2pt, stroke: 0.9pt + GREEN, inset: 9pt, [
    #align(center, text(size: 16pt)[#keepc[KEEP LABEL] · paraphrase])
    #v(10pt)
    #align(center, text(size: 17pt, weight: 600)[“The film was wonderful!”])
    #v(10pt)
    #align(center, label-pill([POSITIVE]))
  ])],
  [#block(width: 100%, height: 48mm, radius: 2pt, stroke: 0.9pt + RED, inset: 9pt, [
    #align(center, text(size: 16pt)[#failc[CHANGE LABEL] · negation])
    #v(10pt)
    #align(center, text(size: 17pt, weight: 600)[“The movie was not fantastic.”])
    #v(10pt)
    #align(center, label-pill([NEGATIVE], color: RED))
  ])],
))

#v(12pt)
#note[The transform is valid only if the target is still correct. Otherwise relabel it—or reject it.]

#place(bottom + center, dy: -2pt, caption[Example adapted from STT-AI Week 5: text augmentation.])

== For NER, edits can move the labelled spans #V

#punch[Keep the entity text and type; recompute where each span begins and ends.]

#v(8pt)
#align(center, image("figures/sources/stt_ner_context_core.png", width: 172mm))

#v(12pt)
#align(center, text(size: 20pt, weight: 650)[
  $cal(A) = {(s_i, e_i, c_i)} arrow.r cal(A) prime = {(s_i prime, e_i prime, c_i)}$
])

#v(10pt)
#align(center, text(size: 16pt)[
  #keepc[KEEP] entity strings and types $c_i$
  #h(20mm)
  #knobc[UPDATE] affected offsets $s_i, e_i$
])

#place(bottom + center, dy: -2pt, caption[Example adapted from STT-AI Week 5: NER-safe context editing.])

#let event-box(name, times, color) = box(
  fill: color.lighten(72%), stroke: 0.7pt + color, radius: 2pt,
  inset: (x: 7pt, y: 4pt),
  align(center, [#text(size: 13.5pt, weight: 650)[#name] #linebreak() #text(size: 12.5pt)[#times]]),
)
#let event-row(lead, door-times, bark-times, tail) = grid(
  columns: (10mm, 1fr, 1fr, 10mm), gutter: 4pt, align: center,
  [#text(size: 12.5pt, fill: MUTED)[#lead]],
  [#event-box([door slam], door-times, ACC)],
  [#event-box([dog bark], bark-times, BLUE)],
  [#text(size: 12.5pt, fill: MUTED)[#tail]],
)

== For ASR, masking can keep the transcript fixed #V

#punch[Hide some evidence, but keep the words intelligible.]

#v(10pt)
#align(center, image("figures/sources/stt_specaugment_core.png", width: 174mm))

#v(14pt)
#align(center, text(size: 19pt, weight: 650)[
  #fitc[input changes:] $x prime = tau(x)$
  #h(18mm)
  #keepc[target stays:] $y prime = y$
])

#v(12pt)
#note[Keep the transcript only while the masked speech is still understandable.]

#place(bottom + center, dy: -2pt, caption[Example adapted from STT-AI Week 3: SpecAugment.])

== A time shift must move every event interval #V

#punch[Shift the audio by $Delta$ seconds; shift its timestamp labels by the same $Delta$.]

#v(5pt)
#align(center, text(size: 20pt, weight: 650)[
  $[t_s, t_e] prime = [t_s + Delta, t_e + Delta]$
])

#v(6pt)
#align(center, block(width: 170mm, stroke: 0.8pt + ACC, radius: 3pt, inset: 7pt, [
  #text(size: 15pt, weight: 700)[BEFORE]
  #v(3pt)
  #event-row([0 s], [[2.3, 3.1] s], [[5.0, 8.2] s], [10 s])

  #v(7pt)
  #align(center, text(size: 17pt, weight: 700, fill: ACC)[$Delta = +2 " s"$ ↓])
  #v(3pt)

  #text(size: 15pt, weight: 700)[AFTER]
  #v(3pt)
  #event-row([0 s], [[4.3, 5.1] s], [[7.0, 10.2] s], [12 s])
]))

#place(bottom + center, dy: -2pt, caption[
  After a crop, clip intervals to the new window—or drop events that leave it entirely.
  #linebreak()
  Example adapted from STT-AI Week 5: sound-event timestamps.
])

== Transform complete samples in one PyTorch call #V

#align(center, grid(columns: (105mm, 1fr), gutter: 18pt,
  [
    #codebox(size: 11.4pt)[```python
from torchvision import tv_tensors
from torchvision.transforms import v2

img = tv_tensors.Image(img)
target = {
    "boxes": tv_tensors.BoundingBoxes(
        boxes, format="XYXY", canvas_size=img.shape[-2:]),
    "masks": tv_tensors.Mask(masks),
    "labels": labels,
}

train_aug = v2.Compose([
    v2.RandomPhotometricDistort(p=.5),
    v2.RandomIoUCrop(),
    v2.RandomHorizontalFlip(p=.5),
    v2.SanitizeBoundingBoxes(),
])

img2, target2 = train_aug(img, target)
```]
  ],
  [#align(left, [
    #set text(size: 15pt)
    *One call keeps the sample synchronized*
    - `TVTensor` types dispatch the right transform to each target.
    - Geometry shares one sampled parameter set; photometric changes touch only the image.
    - `SanitizeBoundingBoxes` removes invalid boxes and matching labels or masks after a crop.
    #v(9pt)
    *Before training*
    1. Overlay 32 random samples.
    2. Check box canvas, positive area, and matching instance counts.
    3. Check mask size and allowed IDs.
    4. Unit-test one known flip numerically.
  ])],
))

#note([Keep validation and test deterministic. `KeyPoints` uses the same typed-dispatch idea but remains a beta TorchVision API. #link(PTV + "auto_examples/transforms/plot_transforms_e2e.html")[official v2 detection/segmentation example]], color: BLUE)

// ── a data-space special case: small random input noise ──
== Input noise is a local kind of data augmentation #V

#reg-map(active: "data", named: true, focus-method: [input noise])

#v(10pt)
#punch[Add small random nudges around each input, making sharp local changes costly; #knobc[$sigma$] sets the neighbourhood size.]

== One example becomes a neighbourhood #V

#let jitter-demo = block(
  width: 82mm, height: 50mm, radius: 2pt,
  stroke: 0.7pt + MUTED, fill: rgb("#F7F5F0"),
  [
    #place(top + left, dx: 6mm, dy: 25mm, line(length: 70mm, stroke: 0.9pt + MUTED))
    #for dx in (10mm, 18mm, 25mm, 31mm, 49mm, 55mm, 62mm, 70mm) {
      place(top + left, dx: dx - 1.65mm, dy: 25mm - 1.65mm,
        circle(radius: 1.65mm, fill: TEAL.transparentize(20%), stroke: 0.4pt + white))
    }
    #place(top + left, dx: 40mm - 2.4mm, dy: 25mm - 2.4mm,
      circle(radius: 2.4mm, fill: INK, stroke: 0.6pt + white))
    #place(top + center, dy: 5mm, text(size: 14pt, weight: 700)[$y_i$ is reused for every nearby input])
    #place(top + left, dx: 37.2mm, dy: 31mm, text(size: 13pt, fill: INK)[$x_i$])
    #place(bottom + center, dy: -5mm, text(size: 12.5pt, fill: TEAL)[fresh $x_i + epsilon_k$ each epoch])
  ],
)

#align(center, grid(columns: (84mm, 1fr), gutter: 20pt,
  [#align(center, jitter-demo)],
  [#align(left, [
    #set text(size: 16pt)
    *Each time the example is loaded:*
    #v(5pt)
    #enum(
      [draw fresh noise $epsilon tilde cal(N)(0, sigma^2 I)$;],
      [make a nearby input $x'_i = x_i + epsilon$;],
      [train on $(x'_i, y_i)$ with the *same* target.],
      spacing: 7pt,
    )
    #v(5pt)
    #note([The model must work around $x_i$, not only at the exact stored coordinate.], color: TEAL)
  ])],
))

#pause
#punch[#knobc[$sigma$] controls the radius: small enough to preserve the label, large enough to matter.]

== Why it helps: nearby predictions should not jump #V

#align(center, grid(columns: (126mm, 1fr), gutter: 18pt,
  [
    #set text(size: 12.5pt)
    #lq.diagram(
      width: 124mm, height: 58mm, xlim: (-1.06, 1.06), ylim: (-2.7, 3.7), margin: 0%,
      xlabel: [$x$], ylabel: [$y$],
      grid: (stroke: 0.35pt + MUTED.transparentize(78%)),
      xaxis: (ticks: (-1, 0, 1), subticks: none, mirror: false),
      yaxis: (ticks: (-2, 0, 2), subticks: none, mirror: false),
      lq.plot(XG, truth-curve, stroke: truth-stroke, mark: none),
      lq.plot(XG, curve-of(w9), stroke: 1.8pt + RED.transparentize(32%), mark: none),
      lq.plot(XG, curve-of(w-aug), stroke: 2.4pt + GREEN, mark: none),
      lq.scatter(xa, ya, color: TEAL.transparentize(38%), size: 3.4pt,
        mark: "o", stroke: 0.45pt + white.transparentize(45%)),
      lq.scatter(xs, ys, color: INK, size: 8pt,
        mark: "o", stroke: 0.9pt + white),
    )
    #align(center, text(size: 12.5pt)[
      #text(size: 16pt, fill: INK)[●] #text(fill: INK, weight: 650)[10 measurements] #h(6pt)
      #text(size: 12pt, fill: TEAL.transparentize(38%))[●] 8 nearby copies each
      #h(6pt) #sw(MUTED, [hidden truth])
    ])
  ],
  [#pad(left: 3mm, [#align(left, [
    #set text(size: 14.5pt)
    #failc[Without input noise]
    #v(2pt)
    The red curve fits every stored point, but can swing sharply between them.
    #v(8pt)
    #keepc[With input noise]
    #v(2pt)
    The green curve is tested around every point, so fragile local wiggles become costly. Validation MSE falls from $#fnum(mse(w9, xv, yv), d: 2)$ to $#fnum(mse(w-aug, xv, yv), d: 3)$.
    #v(8pt)
    #knobc[Why this is regularization]
    #v(2pt)
    It prefers models whose predictions are stable near observed data, rather than models that only memorize the stored coordinates.
  ])])],
))

== The noisy objective averages over nearby copies #D

#align(center, text(size: 18pt, weight: 650)[
  #fitc[Stored data] anchors the loss. #knobc[Noise draws] probe nearby inputs. #valc[Validation sets $sigma$].
])

#v(10pt)
#align(center, block(
  width: 178mm, fill: rgb("#F7F5F0"), radius: 4pt,
  inset: (x: 12pt, y: 10pt),
  [
    #align(center, text(size: 22pt)[
      $cal(L)_"noise"(theta) =$
      #text(fill: TEAL)[$1/n sum_i$]
      #text(fill: ACC)[$EE_(z tilde cal(N)(0,I))$]
      $[ell(f_theta(
        #text(fill: TEAL)[$x_i$] +
        #text(fill: BLUE)[$sigma$]
        #text(fill: ACC)[$z$]
      ), #text(fill: TEAL)[$y_i$])]$
    ])
  ],
))

#v(10pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 15pt,
  hairline([#fitc[Examples]], [Average over #fitc[stored pairs] #fitc[$(x_i,y_i)$].], color: TEAL),
  hairline([#knobc[Noise]], [Average over #knobc[fresh draws] of #knobc[$z$].], color: ACC),
  hairline([#valc[Selection]], [Choose #valc[$sigma$] using #valc[validation loss].], color: BLUE),
))

#v(7pt)
#pause
#note([#text(size: 14pt)[For small #valc[$sigma$], a first-order Taylor expansion gives an approximate #knobc[local-sensitivity penalty] $norm(J_(f) thin (x))_F^2$. We add noise to the #fitc[data]. This encourages #knobc[local smoothness].]], color: BLUE)

== CIFAR-10 augmentations in PyTorch #V

#codebox(size: 12.3pt)[```python
import torch
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

#note([The gallery showed the geometry at high resolution. This is the $32 times 32$ CIFAR-10 recipe used in our controlled experiment. Training uses fresh random views, while validation and test remain deterministic. Compute `mean, std` from training only. #link(PTV + "transforms.html")[torchvision v2 reference]], color: BLUE)

== Inspect augmented samples before training #V

#align(center, grid(columns: (94mm, 1fr), gutter: 16pt,
  [
    #image("figures/augmentation_batch_audit_hd.png", width: 94mm)
    #align(center, text(size: 12.5pt, fill: MUTED)[high-resolution stand-in; run the same check on your training data])
  ],
  [
    #align(left, [
      #set text(size: 16pt)
      *For eight random views, ask:*
      #v(5pt)
      - Is the object still present?
      - Would a domain expert keep the label?
      - Are boxes or masks still aligned?
      - Is the strength plausible at deployment?
      - Is normalization applied *after* pixel transforms?
    ])
  ],
))

#codebox(size: 12.5pt)[```python
import torch, torchvision

views = torch.stack([train_tf(raw_image) for _ in range(8)])
grid = torchvision.utils.make_grid(denormalize(views), nrow=4)
show(grid)                       # make this a pre-flight check
```]

== RandAugment has two main settings #V

#align(center, text(size: 15.5pt)[
  For each training image: *sample $N$ operations* $arrow.r$ *apply them in sequence* $arrow.r$ *use magnitude index $M$*
])

#let ra-panel(path, tag) = stack(spacing: 3pt,
  image(path, width: 34mm, height: 34mm, fit: "cover"),
  align(center, text(size: 11.5pt, fill: MUTED, tag)),
)
#v(2pt)
#align(center, grid(columns: (34mm, 34mm, 34mm, 34mm), gutter: 4mm,
  ra-panel("figures/randaugment_original_hd.png", [original]),
  ra-panel("figures/randaugment_rotate_color_hd.png", [`Rotate` $arrow.r$ `Color`]),
  ra-panel("figures/randaugment_solarize_translate_hd.png", [`Solarize` $arrow.r$ `TranslateX`]),
  ra-panel("figures/randaugment_posterize_contrast_hd.png", [`Posterize` $arrow.r$ `Contrast`]),
))
#align(center, text(size: 11.5pt, fill: MUTED)[high-resolution examples from the operation pool; the sequences do not show one fixed value of $M$])

#v(2pt)
#align(center, [
  #set text(size: 13.5pt)
  #table(
    columns: (34mm, 1fr), stroke: 0.4pt + MUTED,
    inset: (x: 7pt, y: 3pt), align: (left, left),
    [#fitc[$N$, `num_ops`]], [number of randomly chosen operations applied in sequence],
    [#knobc[$M$, `magnitude`]], [a bin index mapped to an operation-specific scale, rather than degrees or percent],
  )
])

#pause
#align(center, text(size: 16pt, weight: 650)[RandAugment avoids a separate search for a learned policy. Validation still chooses $N$, $M$, and a task-valid operation space.])

== RandAugment in TorchVision #V

#align(center, grid(columns: (111mm, 1fr), gutter: 18pt,
  [#codebox(size: 11.7pt)[```python
import torch
from torchvision.transforms import (
    InterpolationMode, v2,
)

strong_train_tf = v2.Compose([
    v2.ToImage(),  # keep uint8 here
    v2.RandomCrop(32, padding=4,
                  padding_mode="reflect"),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandAugment(
        num_ops=2,
        magnitude=9,          # bin 9 of 31
        interpolation=InterpolationMode.BILINEAR,
    ),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean, std),
])
```]],
  [#align(left, [
    #set text(size: 14.5pt)
    *What the code does*
    - Two operations are sampled and applied in sequence on each call.
    - `magnitude=9` is a bin index, not “9 degrees.”
    - Convert and normalize *after* RandAugment.
    - Keep it out of validation and test.
    #v(8pt)
    #failc[*Limitation:*] TorchVision's `RandAugment` handles images and videos only. It does not update boxes, masks, or keypoints, so spatial targets need a task-aware policy.
  ])],
))

#note([Horizontal flip is separate because it is not in TorchVision's RandAugment operation pool. Inspect many draws, then validate $N$ and $M$. #link(PTV + "generated/torchvision.transforms.v2.RandAugment.html")[official API]], color: BLUE)

== Mixup and CutMix change examples and targets #V

#reg-map(active: "mixing", named: true, focus-method: [Mixup / CutMix])

#v(10pt)
#pause
#align(center, grid(
  columns: (1fr, 1fr), gutter: 18pt,
  hairline([#fitc[INTUITION]], [
    #text(size: 15.5pt)[Mix two inputs by a fraction; mix their targets by the same fraction.]
  ], color: TEAL),
  hairline([#knobc[WHY IT CAN HELP]], [
    #text(size: 15.5pt)[Training between stored examples makes memorizing each one less useful and encourages smoother predictions.]
  ], color: ACC),
))

== Mixup interpolates examples and labels #D

$ lambda tilde "Beta"(alpha, alpha), quad tilde(x) = lambda x_i + (1-lambda)x_j, quad tilde(y) = lambda e_(y_i) + (1-lambda)e_(y_j) $

#let mix-panel(path, tag, colour) = stack(spacing: 3pt,
  block(stroke: 1pt + colour, radius: 2pt, clip: true,
    image(path, width: 48mm, height: 48mm, fit: "cover")),
  align(center, text(size: 12.5pt, fill: colour, tag)),
)
#v(2pt)
#align(center, grid(columns: (48mm, 48mm, 48mm), gutter: 9mm,
  mix-panel("figures/mixup_cat_hd.png", [endpoint $i$ · cat], TEAL),
  mix-panel("figures/mixup_70_hd.png", [$lambda=0.70$ · soft target 70/30], BLUE),
  mix-panel("figures/mixup_dog_hd.png", [endpoint $j$ · dog], ACC),
))
#caption[Course-generated teaching photographs with exact pixel arithmetic; the CIFAR-10 experiment uses different inputs.]

#pause
#punch[The blended image need not look natural. Mixup specifies how the prediction should change between two stored examples.]

== For a 70/30 blend, predict 70/30 #D

#align(center, grid(columns: (110mm, 1fr), gutter: 20pt,
  [#align(left, [
    #set text(size: 16pt)
    Mixup gives the blended image a *soft target*:
    #v(5pt)
    $ tilde(y) = 0.70 e_"cat" + 0.30 e_"dog" $
    #v(8pt)
    #grid(columns: (7fr, 3fr), gutter: 0pt,
      block(fill: TEAL.lighten(72%), inset: 6pt)[*cat* 0.70],
      block(fill: BLUE.lighten(72%), inset: 6pt)[*dog* 0.30],
    )
    #v(10pt)
    Cross-entropy uses those same numbers as weights:
    #v(4pt)
    $ "CE"(tilde(y),p) = -0.70 log p_"cat" - 0.30 log p_"dog" $
    #v(10pt)
    #note([Best match: $p_"cat"=0.70$ and $p_"dog"=0.30$. The minimum is *positive* because neither class has target probability 1.], color: TEAL)
  ])],
  [#align(left, [
    #set text(size: 15pt)
    *How strongly do examples mix?*
    #v(4pt)
    $ lambda tilde "Beta"(alpha, alpha) $
    #v(4pt)
    - small $alpha$: $lambda$ is usually near 0 or 1;
    - $alpha=1$: all blend ratios are equally likely;
    - large $alpha$: blends stay near 50/50.
    #v(10pt)
    #note([$lambda$ weights both the pixels and the two target entries. It does not claim that the scene is literally “70% cat.”], color: ACC)
    #v(12pt)
    #text(size: 12.5pt, fill: MUTED)[Later in the course: the entropy and KL-divergence view of cross-entropy.]
  ])],
))

== Birdsong + rain can be a real recording #V

#align(center, [
  #set text(size: 15pt)
  A microphone records *sums of pressure waves*.
  #v(3pt)
  #image("figures/sources/mixup_audio_superposition_imagen.png", width: 160mm)
  #v(2pt)
  #grid(columns: (1fr, 1fr, 1fr), gutter: 8mm,
    align(center, text(size: 13.5pt, weight: 700, fill: TEAL)[birdsong $x_i$]),
    align(center, text(size: 13.5pt, weight: 700, fill: BLUE)[mixture $tilde(x)$]),
    align(center, text(size: 13.5pt, weight: 700, fill: ACC)[rain $x_j$]),
  )
  #v(3pt)
  $ tilde(x) = 0.70 x_i + 0.30 x_j quad comma quad tilde(y) = 0.70 e_"bird" + 0.30 e_"rain" $
  #v(4pt)
  #note([The input mixture is physically meaningful. The target is a soft clip-level training signal that still needs validation. For multi-label detection, use multi-hot targets.], color: GREEN)
  #v(2pt)
  #text(size: 11.5pt, fill: MUTED)[Imagen-generated schematic spectrograms; in code, mix the waveforms rather than these displayed pixels]
])

== A left turn + a right turn is not “go straight” #V

#align(center, [
  #image("figures/mixup_driving_exact_blend_wide.png", width: 160mm)
  #v(2pt)
  #grid(columns: (1fr, 1fr, 1fr), gutter: 8mm,
    align(center, text(size: 14pt, weight: 750, fill: TEAL)[steer $-25 degree$]),
    align(center, text(size: 14pt, weight: 750, fill: RED)[raw Mixup $arrow.r$ target $0 degree$]),
    align(center, text(size: 14pt, weight: 750, fill: ACC)[steer $+25 degree$]),
  )
  #v(3pt)
  $ tilde(x) = 0.5 x_"left" + 0.5 x_"right" quad comma quad tilde(y) = 0 degree $
  #v(4pt)
  #note([The blended image is a double exposure, but the averaged target says “drive straight.” Neither source supports that instruction, so the training example is invalid.], color: RED)
])

== Before using Mixup, ask three questions #D

#align(left, grid(
  columns: (14mm, 1fr), row-gutter: 14pt, column-gutter: 10pt,
  text(size: 24pt, weight: 800, fill: TEAL)[1],
  [
    #text(size: 16.5pt, weight: 750)[Is mixing meaningful in this representation?]
    #v(3pt)
    #text(size: 15pt)[#keepc[Often sensible:] pixels, waveforms, and normalized continuous features. #failc[Usually not:] raw token IDs, categories, graphs, and molecules.]
  ],

  text(size: 24pt, weight: 800, fill: ACC)[2],
  [
    #text(size: 16.5pt, weight: 750)[Does the mixed target match the task?]
    #v(3pt)
    #text(size: 15pt)[#keepc[May mix directly:] global class targets. #failc[Needs its own rule:] boxes, masks, timestamps, and actions.]
  ],

  text(size: 24pt, weight: 800, fill: BLUE)[3],
  [
    #text(size: 16.5pt, weight: 750)[Does it help on untouched validation data?]
    #v(3pt)
    #text(size: 15pt)[#valc[Use untouched validation:] choose $alpha$ on the metric that matters and inspect mixed samples for domain failures.]
  ],
))

#v(14pt)
#note([If a check fails, mix later features, define the correct target rule, use a task-aware alternative, or skip mixing.], color: BLUE)
#v(12pt)
#punch[Use raw-input Mixup only when all three checks pass.]

== CutMix: patch-based mixing #V

#align(center, grid(columns: (108mm, 1fr), gutter: 20pt,
  [#align(center, grid(columns: (49mm, 49mm), gutter: 8mm,
    mix-panel("figures/mixup_cat_hd.png", [base image], TEAL),
    mix-panel("figures/cutmix_25_hd.png", [25% patch from dog], GREEN),
  ))],
  [#align(left, [
    #set text(size: 15.5pt)
    Paste a rectangular region from image $j$ into image $i$.
    #v(8pt)
    $ r = "patch area" / "image area" $
    #v(4pt)
    $ tilde(y) = (1-r)e_(y_i) + r e_(y_j) $
    #v(8pt)
    Here $r=0.25$, so the target is $0.75$ cat + $0.25$ dog.
    #v(8pt)
    Pixels within each pasted region remain unblended. The same domain and spatial-target checks still apply.
  ])],
))

#v(5pt)
#punch[CutMix keeps pixels intact inside each patch while still training the prediction to reflect both source images.]

#place(bottom + center, dy: -2pt, caption[
  Blend images and labels in #link(IA + "augmentation-mixup/")[interactive-articles/augmentation-mixup].
])

== MixUp and CutMix in PyTorch #V

#align(center, grid(columns: (112mm, 1fr), gutter: 18pt,
  [#codebox(size: 11.5pt)[```python
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch.nn.functional as F

train_loader = DataLoader(
    train_set, batch_size=B, shuffle=True)

batch_aug = v2.RandomChoice([
    v2.MixUp(num_classes=K, alpha=0.2),
    v2.CutMix(num_classes=K, alpha=1.0),
])

for images, class_ids in train_loader:
    images, soft_targets = batch_aug(
        images, class_ids
    )
    logits = model(images)
    loss = F.cross_entropy(logits, soft_targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward(); optimizer.step()
```]],
  [#align(left, [
    #set text(size: 14.5pt)
    *What changes in the training loop*
    - transforms run *after* batching;
    - labels change from class IDs `(B,)` to probabilities `(B, K)`;
    - `cross_entropy` accepts those soft targets directly;
    - the current implementation pairs adjacent samples, so shuffle the loader.
    #v(9pt)
    Evaluate accuracy on an unmodified deterministic split with the original class IDs.
  ])],
))

#note([MixUp and CutMix run on batches rather than inside the per-image dataset pipeline. #link(PTV + "auto_examples/transforms/plot_cutmix_mixup.html")[official TorchVision example]], color: BLUE)

// ═══════════════════════ PART V — DROPOUT ═══════════════════════
== Dropout changes hidden activations during training #V

#reg-map(active: "model", named: true)

#v(10pt)
#punch[Hide random hidden activations during training, so useful features cannot rely on one fixed partner; use the full network for evaluation.]

== One favourite route is fragile #V

#align(center, grid(columns: (72mm, 1fr), gutter: 12mm,
  [#align(center, image("figures/dropout_route_brittle.png", width: 68mm))],
  [#align(left, [
    #set text(size: 16pt)
    #text(size: 14pt, weight: 750, fill: RED)[THE ROUTINE LOOKS SUCCESSFUL]
    #v(4pt)
    Every trip follows the same familiar route, and the destination is reached.
    #v(12pt)
    #text(size: 14pt, weight: 750, fill: RED)[BUT THERE IS NO BACKUP]
    #v(4pt)
    Nothing forces the traveller to learn another way. Close one critical road and the whole journey fails.
  ])],
))

#v(8pt)
#punch[A system that depends on one pathway can handle routine cases yet remain brittle.]

== Practise with a different closure every trip #V

#align(center, grid(columns: (72mm, 1fr), gutter: 12mm,
  [#align(center, image("figures/dropout_route_practice.png", width: 68mm))],
  [#align(left, [
    #set text(size: 16pt)
    #enum(
      [Close a *fresh set of roads* on each practice trip.],
      [Keep the *start, destination,* and *city map* unchanged.],
      [Make the traveller find a route that still works.],
      spacing: 10pt,
    )
    #v(9pt)
    #note([A fixed closure would merely create a new favourite route. Changing the closures rewards genuine alternatives.], color: TEAL)
  ])],
))

#v(7pt)
#punch[Random disruption changes the available route—not the task being solved.]

== At evaluation, open every road #V

#align(center, grid(columns: (72mm, 1fr), gutter: 12mm,
  [#align(center, image("figures/dropout_route_evaluation.png", width: 68mm))],
  [#align(left, [
    #set text(size: 16pt)
    #enum(
      [Remove the practice barriers.],
      [Make *all roads* available.],
      [Use the several routes learned while practising with closures.],
      spacing: 10pt,
    )
    #v(9pt)
    #note([The traveller is not split into separate copies. One traveller learned from every disrupted trip.], color: GREEN)
  ])],
))

#v(7pt)
#punch[Training creates backup routes; evaluation uses the complete system.]

== Translate the road story into dropout #V

#align(center, [
  #set text(size: 15.5pt)
  #table(
    columns: (67mm, 100mm),
    stroke: 0.5pt + MUTED,
    inset: (x: 10pt, y: 5pt),
    align: (left, left),
    table.header(
      [#text(weight: 750, fill: TEAL)[ROAD STORY]],
      [#text(weight: 750, fill: ACC)[NEURAL NETWORK]],
    ),
    [road or route], [feature pathway],
    [temporary closure], [hidden activation set to zero for *this pass*],
    [one trip], [one training forward pass],
    [fresh closures], [a fresh dropout mask],
    [same city map], [the same stored weights $bold(theta)$],
    [every road open], [dropout disabled for evaluation],
  )
])

#v(8pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 8mm,
  [#note([*Changes for one pass:* which activations are available.], color: TEAL)],
  [#note([*Does not change:* the stored network or its weights.], color: GREEN)],
))

#v(6pt)
#punch[The same weights must succeed through many temporary subnetworks.]

== Back to the network: one worked example #V

#let worked-drop-net(mask, shown, tag) = {
  align(center, diagram(
    spacing: (11.5mm, 8mm), node-stroke: 0.8pt + INK, node-fill: white,
    {
      for i in range(3) {
        let y = (i - 1) * 1.25
        node((0, y), none, radius: 3.2mm, fill: INK, stroke: none)
      }
      for (j, keep) in mask.enumerate() {
        let y = (j - 1.5) * 1.12
        if keep == 1 {
          for i in range(3) {
            edge((0, (i - 1) * 1.25), (1, y), stroke: 0.55pt + MUTED.lighten(20%))
          }
          edge((1, y), (2, 0), stroke: 0.8pt + TEAL)
          node((1, y), text(size: 11pt, weight: 700, fill: white, shown.at(j)),
            radius: 4mm, fill: TEAL, stroke: none)
        } else {
          node((1, y), text(size: 15pt, weight: 800, fill: RED)[$times$],
            radius: 4mm, fill: white, stroke: (paint: RED, thickness: 1pt, dash: "dashed"))
        }
      }
      node((2, 0), text(size: 11pt, weight: 700, fill: white)[$z$],
        radius: 4mm, fill: ACC, stroke: none)
      node((0, 2.55), text(size: 10.5pt, fill: MUTED)[input $bold(x)$], stroke: none)
      node((1, 2.55), text(size: 10.5pt, fill: MUTED)[hidden $bold(h)$], stroke: none)
      node((2, 2.55), text(size: 10.5pt, fill: MUTED)[next layer], stroke: none)
      node((1, -2.65), text(size: 11pt, weight: 650, fill: INK, tag), stroke: none)
    }
  ))
}

#align(center, grid(columns: (96mm, 1fr), gutter: 38pt,
  [#worked-drop-net((1, 1, 1, 1), ([2], [4], [6], [8]), [all four hidden activations are on])],
  [#align(left, [
    #set text(size: 16pt)
    For one input, suppose the hidden layer produces
    #v(7pt)
    #align(center, text(size: 23pt, weight: 700, fill: TEAL)[$bold(h) = (2, 4, 6, 8)$])
    #v(9pt)
    Apply dropout to this hidden layer before it feeds the next layer.
    #v(9pt)
    #note([Use the #keepc[same network and same $bold(h)$] on every following slide.], color: GREEN)
  ])],
))

#punch[Dropout closes activations for one forward pass; it does not delete weights from the stored network.]

== Training pass: flip four coins #V

#align(center, grid(columns: (1fr, 1fr), gutter: 18pt,
  [#align(center, [
    #text(size: 15pt, weight: 750, fill: TEAL)[BEFORE DROPOUT]
    #v(5pt)
    #worked-drop-net((1, 1, 1, 1), ([2], [4], [6], [8]), [$bold(h)=(2,4,6,8)$])
  ])],
  [#align(center, [
    #text(size: 15pt, weight: 750, fill: RED)[THIS TRAINING PASS]
    #v(5pt)
    #worked-drop-net((1, 0, 1, 0), ([2], [0], [6], [0]), [$bold(m)=(1,0,1,0)$])
  ])],
))

#v(7pt)
#align(center, text(size: 22pt)[
  $ underbrace(bold(h), (2,4,6,8)) quad arrow.r^("mask " bold(m)) quad underbrace(bold(m) dot.o bold(h), (2,0,6,0)) $
])

== First: masking lowers the average signal #V

#let act-cell(value, live: true, color: TEAL) = block(
  width: 25mm, height: 14mm, radius: 2pt,
  fill: if live { color.lighten(72%) } else { MUTED.lighten(83%) },
  stroke: 0.8pt + if live { color } else { MUTED },
  align(center + horizon, text(size: 18pt, weight: 750,
    fill: if live { INK } else { MUTED }, value)),
)
#let act-row(values, live, color: TEAL) = stack(dir: ltr, spacing: 4mm,
  ..values.enumerate().map(((i, value)) => act-cell(value, live: live.at(i), color: color)))

#punch[$q_"keep"=0.5$ means each activation survives only half of the training passes.]

#align(center, grid(columns: (42mm, 1fr), row-gutter: 8pt,
  [#text(size: 13.5pt, weight: 700, fill: MUTED)[FULL ACTIVATION]],
  [#act-row(([2], [4], [6], [8]), (true, true, true, true))],
  [#text(size: 13.5pt, weight: 700, fill: RED)[ONE RAW MASK]],
  [#act-row(([2], [0], [6], [0]), (true, false, true, false), color: RED)],
))

#v(10pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 20pt,
  hairline([#failc[ONE PASS]], [Some entries become zero: $(2,4,6,8) arrow.r (2,0,6,0)$.], color: RED),
  hairline([#knobc[MANY PASSES]], [Every coordinate contributes only $50%$ of the time, so its average contribution is halved.], color: ACC),
))

#v(8pt)
#punch[Without a correction, the next layer receives a smaller signal during training than during evaluation.]

== Then: scale the survivors during training #V

#punch[If half survive on average, multiply each survivor by $2 = 1/q_"keep"$. Dropped entries stay zero.]

#align(center, grid(columns: (42mm, 1fr), row-gutter: 7pt,
  [#text(size: 13.5pt, weight: 700, fill: RED)[MASK A · RAW]],
  [#act-row(([2], [0], [6], [0]), (true, false, true, false), color: RED)],
  [#text(size: 13.5pt, weight: 700, fill: GREEN)[MASK A · ×2]],
  [#act-row(([4], [0], [12], [0]), (true, false, true, false), color: GREEN)],
  [#text(size: 13.5pt, weight: 700, fill: RED)[MASK B · RAW]],
  [#act-row(([0], [4], [0], [8]), (false, true, false, true), color: RED)],
  [#text(size: 13.5pt, weight: 700, fill: GREEN)[MASK B · ×2]],
  [#act-row(([0], [8], [0], [16]), (false, true, false, true), color: GREEN)],
))

#v(9pt)
#align(center, note([The correction changes the #knobc[magnitudes] of surviving activations. It does not change which units the mask dropped. Next, zoom in on $h_3=6$.], color: ACC))

== One activation has two possible training-time outputs #D

#punch[Zoom in on #fitc[$h_3=6$], with #knobc[$q_"keep"=0.5$]. A fresh mask is drawn on every forward pass.]

#v(8pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([#keepc[KEPT]], [
    This happens with probability $0.5$.
    #v(7pt)
    #align(center, text(size: 23pt, weight: 750, fill: GREEN)[$6 / 0.5 = 12$])
    #v(5pt)
    #align(center, text(size: 15pt)[The next layer receives *12*.])
  ], color: GREEN),
  hairline([#failc[DROPPED]], [
    This happens with probability $0.5$.
    #v(7pt)
    #align(center, text(size: 23pt, weight: 750, fill: RED)[$0$])
    #v(5pt)
    #align(center, text(size: 15pt)[The next layer receives *0*.])
  ], color: RED),
))

#v(14pt)
#punch[One pass sends either #keepc[$12$] or #failc[$0$]—not $6$. The $6$ reappears only after averaging over masks.]

== Expected value is the long-run average over masks #D

#punch[Repeat the pass many times: about half the masks send #keepc[$12$], and half send #failc[$0$].]

#v(12pt)
#align(center, block(
  width: 176mm, fill: rgb("#F7F5F0"), radius: 4pt,
  inset: (x: 12pt, y: 12pt),
  [#align(center, text(size: 24pt)[
    $EE[tilde(h)_3] =$
    #text(fill: GREEN, weight: 750)[$0.5 dot 12$]
    $+$
    #text(fill: RED, weight: 750)[$0.5 dot 0$]
    $=$
    #text(fill: TEAL, weight: 800)[$6 = h_3$]
  ])],
))

#v(12pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 16pt,
  hairline([#keepc[KEPT CONTRIBUTION]], [probability kept $times$ value when kept], color: GREEN),
  hairline([#failc[DROPPED CONTRIBUTION]], [probability dropped $times$ value when dropped], color: RED),
  hairline([#fitc[LONG-RUN AVERAGE]], [the value received on average], color: TEAL),
))

#v(10pt)
#punch[“Expected” does not mean every pass returns $6$. The random training outputs are centred on $6$.]

== The general rule has the same two branches #D

#punch[Now replace $6$ and $0.5$ with any activation $h_i$ and keep probability $q_"keep"$.]

#v(8pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([#keepc[KEPT BRANCH]], [
    probability $q_"keep"$
    #v(5pt)
    value $h_i/q_"keep"$
  ], color: GREEN),
  hairline([#failc[DROPPED BRANCH]], [
    probability $1-q_"keep"$
    #v(5pt)
    value $0$
  ], color: RED),
))

#v(12pt)
#align(center, block(
  width: 176mm, fill: rgb("#F7F5F0"), radius: 4pt,
  inset: (x: 12pt, y: 11pt),
  [#align(center, text(size: 22pt)[
    $EE[tilde(h)_i] =$
    #text(fill: GREEN, weight: 750)[$q_"keep" (h_i / q_"keep")$]
    $+$
    #text(fill: RED, weight: 750)[$(1-q_"keep") dot 0$]
  ])],
))

#v(14pt)
#punch[Now simplify one coloured contribution at a time.]

== The two contributions simplify separately #D

#align(center, grid(columns: (1fr, 1fr), gutter: 20pt,
  hairline([#keepc[KEPT CONTRIBUTION]], [
    #align(center, text(size: 22pt, weight: 750, fill: GREEN)[$q_"keep" (h_i/q_"keep") = h_i$])
    #v(7pt)
    The keep probability cancels the factor $1/q_"keep"$.
  ], color: GREEN),
  hairline([#failc[DROPPED CONTRIBUTION]], [
    #align(center, text(size: 22pt, weight: 750, fill: RED)[$(1-q_"keep") dot 0 = 0$])
    #v(7pt)
    Anything multiplied by the dropped value $0$ contributes $0$.
  ], color: RED),
))

#v(14pt)
#align(center, block(
  width: 132mm, fill: rgb("#F7F5F0"), radius: 4pt,
  inset: (x: 12pt, y: 12pt),
  [#align(center, text(size: 25pt)[
    $EE[tilde(h)_i] =$
    #text(fill: GREEN, weight: 750)[$h_i$]
    $+$
    #text(fill: RED, weight: 750)[$0$]
    $=$
    #text(fill: TEAL, weight: 800)[$h_i$]
  ])],
))

#v(14pt)
#punch[Training is noisy, but its mean activation matches the value used at evaluation.]

== Evaluation: the same network, all units on #V

#align(center, grid(columns: (1fr, 1fr), gutter: 20pt,
  hairline([#knobc[TRAINING]], [
    draw a fresh $bold(m)$, then send
    $tilde(bold(h)) = (bold(m) dot.o bold(h)) / q_"keep"$
    to the next layer.
    #v(8pt)
    `model.train()`
  ], color: ACC),
  hairline([#keepc[EVALUATION]], [
    use the full activation
    $bold(h) = (2,4,6,8)$
    once—no mask and no extra scaling.
    #v(8pt)
    `model.eval()`
  ], color: GREEN),
))

#v(18pt)
#punch[The $1/q_"keep"$ correction happens during training, so evaluation is simply the identity: $bold(h) mapsto bold(h)$.]

== Why dropout regularizes: distribute the evidence #D

#align(center, grid(columns: (1fr, 1fr), gutter: 20pt,
  hairline([#failc[BRITTLE RELIANCE]], [
    $bold(w) = (2,0,0,0)$
    #v(5pt)
    One feature carries everything. If it is masked, the unit loses all evidence.
    #v(6pt)
    $norm(bold(w))_2^2 = 4$
  ], color: RED),
  hairline([#keepc[DISTRIBUTED RELIANCE]], [
    $bold(w) = (0.5,0.5,0.5,0.5)$
    #v(5pt)
    Several features can support the unit when a partner disappears.
    #v(6pt)
    $norm(bold(w))_2^2 = 1$
  ], color: GREEN),
))

#v(10pt)
#align(center, text(size: 19pt, weight: 700)[
  Any input may vanish $arrow.r$ concentrating all influence in one pathway becomes risky.
])

#note([This is an #knobc[$L_2$-like] pressure, not the same objective as ordinary weight decay. The effective penalty depends on the data and activation scales; the equal weights above are an illustration, not a theorem.], color: BLUE)

== What to do in PyTorch #V

#align(center, grid(columns: (108mm, 1fr), gutter: 20pt,
  [#codebox(size: 12.5pt)[```python
model = nn.Sequential(
    nn.Linear(d, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2),   # drop 20%; keep q=0.8
    nn.Linear(256, K),
)

model.train()   # fresh masks: ON
# ... training loop ...

model.eval()    # dropout: OFF
# ... validation / test ...
```]],
  [#align(left, [
    #set text(size: 15pt)
    #knobc[PLACE IT]
    #v(2pt)
    After selected hidden activations—not on final logits.
    #v(7pt)
    #knobc[START MODESTLY]
    #v(2pt)
    Try $p_"drop" approx 0.1$–$0.3$ only when validation shows overfitting.
    #v(7pt)
    #valc[LET PYTORCH SCALE]
    #v(2pt)
    `nn.Dropout` performs inverted scaling automatically. Do not divide manually.
  ])],
))

#punch[PyTorch's `p` means #failc[drop probability]. Some implementations instead use `keep_prob` $= q_"keep" = 1-p$.]

== What does dropout return in evaluation mode? #Q

PyTorch's argument `p` is $p_"drop"$, so $q_"keep" = 1 - p_"drop"$.

#codebox(size: 15pt)[```python
h = torch.tensor([2., 4., 6., 8.])
drop = nn.Dropout(p=0.5)
drop.train(); drop(h)   # one draw gave: tensor([ 4.,  0., 12.,  0.])
drop.eval();  drop(h)   # = ?
```]

#mcq(
  [What does the `eval()` line return?],
  [$(2, 4, 6, 8)$, the identity],
  [$(1, 2, 3, 4)$, scaled by $q_"keep"$],
  [$(4, 8, 12, 16)$, scaled by $1 slash q_"keep"$],
  [a fresh random mask, like training],
)

== Answer: dropout is disabled during evaluation #A

#mcq-answer(
  [A],
  [the identity, because dropout is disabled at test time],
  [*Inverted* dropout applies the $1 slash q_"keep"$ correction during training. Evaluation therefore needs no rescaling and is deterministic.],
)

#pause
#warning[A model left in `train()` mode during evaluation gives noisy, degraded predictions. Call `model.eval()` before computing validation or test metrics.]

== `train()`, `eval()`, and `inference_mode()` #V

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
#punch[`eval()` changes module behaviour. `inference_mode()` changes gradient tracking.]

== Dropout in modern architectures #V

#align(center, table(
  columns: (62mm, 1fr), stroke: 0.45pt + MUTED, inset: (x: 10pt, y: 6pt), align: (left, left),
  table.header([setting], [current practice]),
  [wide fully-connected heads], [a common location because co-adaptation is strongest there],
  [transformer blocks], [small rates inside attention / FFN and on residuals],
  [modern conv backbones], [often secondary to strong augmentation and weight decay; classifier-head dropout remains common],
  [whole residual branches], [the same idea at branch scale: *stochastic depth / DropPath*],
))

#note([Tune $p_"drop"$ on #valc[validation], just as you tune $lambda$. A small value may have little effect, while a large value can destroy useful signal.], color: BLUE)

// ═══════════════════ PART VI — LABEL SMOOTHING ═══════════════════
== Label smoothing changes the training targets #V

#reg-map(active: "targets", named: true)

#v(10pt)
#pause
#punch[Replace one-hot targets with slightly softer targets, so extreme confidence is no longer rewarded; #valc[validation] chooses $epsilon$.]

== Why one-hot cross-entropy keeps widening the gap #D

#punch[Getting the class right is not the finish line: a one-hot target still rewards more certainty.]

#v(5pt)
#align(center, grid(columns: (74mm, 1fr), gutter: 20pt,
  hairline([#failc[1 · ONE-HOT LOSS]], [
    #set text(size: 14pt)
    Suppose the correct class is “b”:
    #v(3pt)
    #align(center, text(size: 16pt)[#failc[$bold(y)=(0,1,0,0,0)$]])
    #v(3pt)
    #align(center, text(size: 18pt)[
      $ell = -sum_k y_k log p_k = $ #fitc[$-log p_b$]
    ])
    #v(4pt)
    Every wrong-class term disappears because its target weight is $0$.
  ], color: RED),
  hairline([#fitc[2 · LOGIT GRADIENT]], [
    #set text(size: 14pt)
    Cross-entropy changes the logits through
    #v(3pt)
    #align(center, text(size: 16pt)[$frac(partial ell, partial z_k) = p_k - y_k$])
    #v(4pt)
    #align(left, [
      #fitc[$p_b-1<0$] $arrow.r$ gradient descent pushes $z_b$ *up*.
      #v(2pt)
      #failc[$p_j-0>0$] $arrow.r$ it pushes every wrong $z_j$ *down*.
    ])
  ], color: TEAL),
))

#v(9pt)
#align(center, stack(spacing: 3pt,
  text(size: 15pt, weight: 650)[
    #text(fill: MUTED)[In a symmetric score picture:] #h(4pt)
    #fitc[$z_b arrow.r +oo$] #text(fill: MUTED)[$quad$]
    #failc[$z_j arrow.r -oo$]
  ],
  text(size: 18pt, weight: 750)[#knobc[What matters: every gap $z_b-z_j$ has no finite stopping point.]],
))

== Label smoothing says “confident enough” #D

#punch[Ask for high confidence, not absolute certainty.]

#v(4pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 20pt,
  hairline([#failc[ONE-HOT]], [
    #set text(size: 13.5pt)
    #align(center, bars((0, 1, 0, 0, 0), labels: ([a], [b], [c], [d], [e]),
      bar: 7mm, span: 16mm,
      color: (i, v) => if i == 1 { RED } else { MUTED.lighten(35%) }))
    #align(center, text(size: 11.5pt)[#failc[$y=(0,1,0,0,0)$]])
    #v(4pt)
    #align(center, text(size: 16pt)[$frac(partial ell, partial z_b) = p_b - 1$])
    #v(2pt)
    The push stops only as $p_b arrow.r 1$: no finite logit gap is enough.
  ], color: RED),
  hairline([#knobc[SMOOTHED]], [
    #set text(size: 13.5pt)
    #align(center, text(size: 12pt)[$tilde(y)=(1-epsilon)y + epsilon/K bold(1)$])
    #v(2pt)
    #align(center, bars((0.02, 0.92, 0.02, 0.02, 0.02), labels: ([a], [b], [c], [d], [e]),
      bar: 7mm, span: 16mm,
      color: (i, v) => if i == 1 { ACC } else { ACC.lighten(45%) }))
    #align(center, text(size: 11.5pt)[#knobc[$tilde(y)=(.02,.92,.02,.02,.02)$]])
    #v(4pt)
    #align(center, text(size: 16pt)[$frac(partial ell, partial z_b) = p_b - 0.92$])
    #v(2pt)
    The push stops at $p_b=0.92$. This corresponds to the finite correct-vs-wrong logit gap
    $z_b-z_j = ln(0.92/0.02) approx 3.83$.
  ], color: ACC),
))

== Use label smoothing modestly—and validate it #V

#align(center, grid(columns: (80mm, 1fr), gutter: 20pt,
  [
    #hairline([PYTORCH], [
      #set text(size: 13pt)
      #codebox(size: 11.5pt)[```python
epsilon = 0.1
criterion = nn.CrossEntropyLoss(
    label_smoothing=epsilon
)
loss = criterion(logits, class_ids)
```]
      #v(3pt)
      Train with smoothed targets; keep validation and test labels hard.
      #v(4pt)
      The smoothed training loss has a positive floor, so do not compare it directly with a one-hot run.
    ], color: TEAL)
  ],
  [
    #hairline([#keepc[CAN HELP]], [
      #set text(size: 13pt)
      #align(left, list(
        [reduce pressure toward extreme logits;],
        [improve validation accuracy, NLL, or calibration in some recipes.],
        spacing: 3pt,
      ))
    ], color: GREEN)
    #v(6pt)
    #hairline([#failc[COSTS AND LIMITS]], [
      #set text(size: 13pt)
      #align(left, list(
        [pulls every prediction toward uniformity;],
        [does not know that “wolf” is closer to “husky” than “airplane”;],
        [may over-soften targets on top of Mixup or CutMix.],
        spacing: 3pt,
      ))
    ], color: RED)
  ],
))

#v(7pt)
#note([#text(size: 15pt)[This is a regularizing pressure, not a promise. Start small, tune $epsilon$ on untouched #valc[validation], and—if Mixup or CutMix is active—ablate smoothing separately.]], color: BLUE)

// ═══════════════════ PART VII — ONE PREFERENCE, MANY KNOBS ═══════════════════
= Regularization in practice

== Regularization cheat sheet: five places to intervene #V

#reg-map(active: "all", named: true)

#v(10pt)
#pause
#punch[Diagnose first. Change one site at a time, then let #valc[validation] choose the strength.]
#caption[Mixup and CutMix span both the data and target sites.]

== A taxonomy of regularization methods #V

#align(center, [
  #set text(size: 16pt)
  #table(
    columns: (52mm, 92mm, 44mm), stroke: 0.45pt + MUTED,
    inset: (x: 9pt, y: 5.5pt), align: (left, left, left),
    table.header([site], [methods], [knob]),
    [objective], [weight decay ($L_2$), $L_1$, spectral penalties], [$lambda$],
    [training loop], [early stopping, minibatch noise], [$t^star$, batch size],
    [model / features], [dropout, stochastic depth], [$p$],
    [data], [crops · flips · Gaussian input noise, Mixup, CutMix], [strength, $sigma$, $alpha$],
    [targets], [label smoothing], [$epsilon$],
    [architecture], [convolution, weight sharing, bottlenecks], [design choice],
  )
])

#note([Architecture adds no explicit penalty term, but it still restricts the function family. Treat it as a design choice selected by validation.], color: ACC)

== Implicit regularization in training #V

#align(center, [
  #set text(size: 16.5pt)
  #table(
    columns: (52mm, 1fr), stroke: 0.45pt + MUTED, inset: (x: 10pt, y: 5.5pt), align: (left, left),
    table.header([implicit source], [selection pressure it exerts]),
    [SGD minibatch noise], [a stochastic trajectory with optimizer- and batch-size-dependent implicit bias],
    [initialization + early phase], [which basin training explores first (Lecture 6: initialization)],
    [architecture], [some functions are easy to express, others practically unreachable],
    [BatchNorm's batch statistics], [per-batch jitter in activations as a *side effect*],
  )
])

#note[
  Treat BatchNorm primarily as an *optimization* method (Lecture 6). It improves trainability, with mild regularization as a side effect. Changing the optimizer or batch size can also change the model's *effective* regularization.
]

== A practical regularization workflow #V

#result[
  #set text(size: 19pt)
  fixed baseline #h(4pt) → #h(4pt) valid augmentation #h(4pt) → #h(4pt) weight decay #h(4pt) → #h(4pt) one extra method
]

#v(3pt)
#align(center, [
  #set text(size: 15pt)
  #table(
    columns: (15mm, 48mm, 1fr), stroke: 0.45pt + MUTED, inset: (x: 8pt, y: 4pt), align: (left, left, left),
    [1], [baseline], [log train/validation curves; restore the best checkpoint],
    [2], [augmentation], [add domain-valid transforms; inspect actual batches],
    [3], [weight decay], [sweep $lambda$ on a log scale; compare at the same compute budget],
    [4], [other methods], [if needed, ablate dropout, MixUp/CutMix, or label smoothing one at a time],
  )
])

== Specialized regularization methods #V

These methods address more specific problems:

#align(center, [
  #set text(size: 15.5pt)
  #table(
    columns: (52mm, 1fr), stroke: 0.45pt + MUTED, inset: (x: 9pt, y: 4.6pt), align: (left, left),
    [stochastic depth / DropPath], [dropout at residual-branch scale; standard in deep ViTs],
    [spectral normalization], [constrain layer operator norms (GANs, sensitivity control)],
    [SAM], [optimize the worst loss in a parameter neighbourhood],
    [adversarial training], [augment with worst-case perturbations to improve robustness, at a cost],
  )
])

#note([Change one factor at a time so the comparison remains interpretable. Evaluate the sealed test set once, after every knob is fixed.], color: BLUE)

== Diagnosing common failure modes #V

#align(center, [
  #set text(size: 15.5pt)
  #table(
    columns: (56mm, 52mm, 84mm), stroke: 0.45pt + MUTED,
    inset: (x: 8pt, y: 5pt), align: (left, left, left),
    table.header([symptom], [first suspect], [controlled test]),
    [train and val both high], [underfit / optimization], [fix fit first (L5 to L6); regularizing makes it *worse*],
    [train $arrow.b$, val turns $arrow.t$], [overfitting over time], [restore best checkpoint; then sweep one knob],
    [val *better* than train], [train-only noise: dropout, aug, penalty in train loss], [re-measure both in `eval()` mode, penalty excluded],
    [great val, poor deployment], [validation no longer represents reality], [audit the split and the data pipeline, not the knobs],
    [confident and wrong], [target pressure], [add smoothing; *measure* calibration],
  )
])

#pause
#punch[Use regularization after diagnosing the failure mode, and choose its strength by validation. Do not add it by default or choose an arbitrary strength.]

== Baseline overfitting on CIFAR-10 #V

#align(center, image("figures/cifar_regularization_curves.svg", width: 94%))

#note([The run uses 2,000 stratified training examples, one fixed initialization and minibatch order, the same 666,538-parameter CNN, and a 35-epoch budget. Dots mark the checkpoint selected by validation loss.], color: INK)

== Controlled CIFAR-10 ablations #V

#align(center, image("figures/cifar_regularization_summary.svg", width: 92%))

#note([Crop + flip performs best; dropout and smoothing help modestly; the tested weight-decay setting ($lambda_"wd" = 0.01$) is neutral. Choices were predeclared and checkpoints used validation; the one-seed test result is not a benchmark.], color: BLUE)

== Three regularizers on the polynomial example #V

#spine-fig(curves: (
  (truth-curve, truth-stroke),
  (curve-of(w9), 1.5pt + RED.transparentize(45%)),
  (curve-of(w-star), 2.0pt + ACC),
  (curve-of(w-stop), 2.0pt + TEAL),
  (curve-of(w-aug), 2.0pt + GREEN),
), height: 47mm, train-size: 6.5pt)
#align(center, text(size: 13pt)[
  #sw(RED, [none · #fnum(mse(w9, xv, yv), d: 2)]) #h(9pt)
  #sw(ACC, [decay $lambda^star$ · #fnum(mse(w-star, xv, yv), d: 3)]) #h(9pt)
  #sw(TEAL, [stop $t^star$ · #fnum(mse(w-stop, xv, yv), d: 3)]) #h(9pt)
  #sw(GREEN, [augment · #fnum(mse(w-aug, xv, yv), d: 3)]); validation MSE
])

#pause
#punch[Weight decay, early stopping, and input jitter act at different sites and use different knobs. Here they favour a less variable fit and produce nearly identical curves.]

== Final model selection and test evaluation #V

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

#note([Use the test column only for final reporting. If it changes a knob, the test set has become another validation set.], color: BLUE)

#provenance

== Before you regularize

#align(center, grid(columns: (1fr, 1fr), gutter: 14pt,
  note([#align(left)[*1.* Which learning curves did you plot, and what do they diagnose?]], color: BLUE),
  note([#align(left)[*2.* Which single site are you intervening on, and with which knob?]], color: ACC),
  note([#align(left)[*3.* What did validation choose? Did training loss influence the choice?]], color: BLUE),
  note([#align(left)[*4.* Has the test set remained untouched for one final evaluation?]], color: INK),
  note([#align(left)[*5.* Which implicit regularizers moved when you changed optimizer or batch size?]], color: TEAL),
  note([#align(left)[*Reproduce:* log the seed, splits, schedule, and every knob.]], color: GREEN),
))

#pause
#punch[If you have not plotted learning curves, diagnose the problem before choosing a regularizer.]

== Primary sources

#[
#set text(size: 15pt)
- Zhang, Bengio, Hardt, Recht, Vinyals (2017), #link("https://arxiv.org/abs/1611.03530")[*Understanding deep learning requires rethinking generalization*]. #h(2pt) Shows that networks can fit random labels.
- Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov (2014), #link("https://jmlr.org/papers/v15/srivastava14a.html")[*Dropout*].
- Zhang, Cissé, Dauphin, Lopez-Paz (2018), #link("https://arxiv.org/abs/1710.09412")[*mixup*] · Yun et al. (2019), #link("https://arxiv.org/abs/1905.04899")[*CutMix*] · Cubuk et al. (2020), #link("https://arxiv.org/abs/1909.13719")[*RandAugment*].
- Szegedy et al. (2016), #link("https://arxiv.org/abs/1512.00567")[label smoothing] · Müller, Kornblith, Hinton (2019), #link("https://arxiv.org/abs/1906.02629")[*When Does Label Smoothing Help?*].
- Bishop (1995), *Training with noise is equivalent to Tikhonov regularization*; Belkin et al. (2019), #link("https://arxiv.org/abs/1812.11118")[double descent]; the modern interpolation picture.
]

#v(4pt)
#note([The seeded polynomial figures are recomputed at compile time. The opening schematic, CIFAR panels, and controlled PyTorch ablation have documented provenance and scripts in `lecture7/figures/PROVENANCE.md`. The companion Typst lab independently checks the regression numbers.], color: INK)

#focus-slide[
  Regularization supplies a preference where the training data do not determine the model's behaviour. Validation chooses its strength.
  #v(12pt)
  #set text(size: 22pt)
  Diagnose with learning curves. Change one factor, then measure again on validation.
]

// ═══════════════════ OPTIONAL APPENDIX ═══════════════════
= Optional: AdamW

== Adam and AdamW treat decay differently #OPT

Let $cal(A)_t thin (u)$ denote Adam's update computed from $u$ using the current moment state.

#align(center, [
  #set text(size: 14.5pt)
  #table(
    columns: (34mm, 100mm, 1fr), stroke: 0.5pt + MUTED,
    inset: (x: 8pt, y: 6pt), align: (left, left, left),
    table.header([optimizer], [parameter update], [what happens to decay]),
    [plain SGD],
      [#text(size: 13.5pt)[$bold(theta)_(t+1) = bold(theta)_t - eta bold(g)_t$]],
      [none],
    [SGD + $L_2$],
      [#text(size: 13.5pt)[$bold(theta)_(t+1) = (1-eta lambda) bold(theta)_t - eta bold(g)_t$]],
      [proportional shrink],
    [Adam + $L_2$],
      [#text(size: 13.5pt)[$bold(theta)_(t+1) = bold(theta)_t - eta cal(A)_t thin (bold(g)_t + lambda bold(theta)_t)$]],
      [penalty enters Adam's moments],
    [AdamW],
      [#text(size: 13.5pt)[$bold(theta)_(t+1) = (1-eta lambda) bold(theta)_t - eta cal(A)_t thin (bold(g)_t)$]],
      [shrink stays separate],
  )
])

#note[
  For plain SGD, $L_2$ and weight decay are equivalent. With Adam, coupled $L_2$ enters the moment estimates; AdamW keeps proportional shrinkage separate.
  #h(3pt)#link("https://arxiv.org/abs/1711.05101")[Loshchilov & Hutter (2019)]
]

#pause
#punch[With Adam, use AdamW when you want weight decay to remain a separate proportional shrinkage step.]

== PyTorch: AdamW #OPT

#v(12pt)
#codebox(size: 13.5pt)[```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
```]

#v(16pt)
#note[
  #set text(size: 14pt)
  `weight_decay` is $lambda$. Each step applies $bold(theta) <- (1 - eta_t lambda) bold(theta)$ before Adam's data update.
  Do not also add $(lambda slash 2) norm(theta)_2^2$ to the loss. This call decays every parameter; use parameter groups to exempt bias or normalization parameters.
]
