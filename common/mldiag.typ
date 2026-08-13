// ═══════════════════════════════════════════════════════════════════
//  mldiag — deck-themed instances of the chalkdust packages.
//  The packages are vendored in ../vendor/chalkdust so a clean checkout
//  has no @local package dependency. This file binds them to the
//  metropolis palette ONCE so a slide is a
//  one-liner:
//
//    #import "../common/mldiag.typ": *
//    #conv-op(input: X, kernel: K, step: 4)          // already palette-locked
//    #attn-matrix(("the","cat","sat"), mask: "causal")
// ═══════════════════════════════════════════════════════════════════

#import "../vendor/chalkdust/theme/lib.typ": theme
#import "../vendor/chalkdust/convgrid/lib.typ" as tg   // conv/pool/attention grids (was tensor-grid)
#import "../vendor/chalkdust/plot/lib.typ" as mp       // bar & line plots
#import "../vendor/chalkdust/frame/lib.typ" as md      // pandas-lite: md.frame(csv("x.csv")), md.xy(f, "a", "b")
#import "../vendor/chalkdust/dist/lib.typ" as dist     // distributions: dist.nll0(dist.normal(), r) — the TRUE loss
#import "../vendor/chalkdust/field/lib.typ" as fld     // 2-D/3-D fields: contour/heatmap/surface of f(x,y)
#import "../vendor/chalkdust/optim/lib.typ" as opt     // optimizers: opt.gd/momentum/adam(grad, x0) → trajectory
#import "../vendor/chalkdust/rand/lib.typ" as rnd      // seeded PRNG: rnd.randn(seed, i), rnd.shuffle(seed, arr)
#import "../vendor/chalkdust/learn/lib.typ" as ml      // in-Typst fitting: linear/logistic regression, PCA, k-means
#import "../vendor/chalkdust/autodiff/lib.typ" as ad   // reverse-mode autodiff: ad.grad(f, x) exact; ad.grad-fn/fn2 for optim+contour
#import "metropolis.typ": INK, ACC, TEAL, GREEN, BLUE, MUTED, RED

// the deck theme: metropolis palette, teal→paper→orange diverging ramp
#let dl-theme = theme(
  ink: INK, muted: MUTED, accent: ACC, accent2: TEAL,
  positive: GREEN, negative: RED,
  ramp: (TEAL, rgb("#EFEEEB"), ACC),
  cycle: (INK, GREEN, TEAL, ACC, RED),
)

// ── pre-themed figure one-liners ──
#let grid-map        = tg.grid-map.with(theme: dl-theme)
#let conv-op         = tg.conv-op.with(theme: dl-theme)
#let pool-op         = tg.pool-op.with(theme: dl-theme)
#let attn-matrix     = tg.attn-matrix.with(theme: dl-theme)
#let receptive-field = tg.receptive-field.with(theme: dl-theme)
#let patchify        = tg.patchify.with(theme: dl-theme)

// ── plots (ml-plot), palette-locked ──
#let bars  = mp.bars.with(theme: dl-theme)
#let lines = mp.lines.with(theme: dl-theme)

// ── 2-D / 3-D fields (ml-field), palette-locked ──
#let contour = fld.contour.with(theme: dl-theme)
#let heatmap = fld.heatmap.with(theme: dl-theme)
#let surface = fld.surface.with(theme: dl-theme)

// ── optimizers (ml-optim) — pure numerics, no theme; feed the path to contour(paths:) ──
#let minimize = opt.minimize
#let gd       = opt.gd
#let momentum = opt.momentum
#let nesterov = opt.nesterov
#let rmsprop  = opt.rmsprop
#let adam     = opt.adam
#let sgd      = opt.sgd
#let numgrad  = opt.numgrad
#let grad2d   = opt.grad2d   // gradient of a loss f(x,y) → no hand-derived ∇ on the slide

// ── small fitted models (computed in Typst; feed predictions straight to lines) ──
#let linreg-fit     = ml.linreg-fit
#let linreg-predict = ml.linreg-predict

// ── the math helpers (figures + prose can share one computation) ──
#let conv2d        = tg.conv2d
#let conv-out-size = tg.conv-out-size
#let softmax-rows  = tg.softmax-rows
