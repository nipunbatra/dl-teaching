// ═══════════════════════════════════════════════════════════════════
//  mldiag — deck-themed instances of the chalkdust packages.
//  The packages live in ~/git/chalkdust (symlinked into @local);
//  this file binds them to the metropolis palette ONCE so a slide is a
//  one-liner:
//
//    #import "../common/mldiag.typ": *
//    #conv-op(input: X, kernel: K, step: 4)          // already palette-locked
//    #attn-matrix(("the","cat","sat"), mask: "causal")
// ═══════════════════════════════════════════════════════════════════

#import "@local/theme:0.1.0": theme
#import "@local/convgrid:0.1.0" as tg   // conv/pool/attention grids (was tensor-grid)
#import "@local/plot:0.1.0" as mp       // bar & line plots
#import "@local/frame:0.1.0" as md      // pandas-lite: md.frame(csv("x.csv")), md.xy(f, "a", "b")
#import "@local/dist:0.1.0" as dist     // distributions: dist.nll0(dist.normal(), r) — the TRUE loss
#import "@local/field:0.1.0" as fld     // 2-D/3-D fields: contour/heatmap/surface of f(x,y)
#import "@local/optim:0.1.0" as opt     // optimizers: opt.gd/momentum/adam(grad, x0) → trajectory
#import "@local/rand:0.1.0" as rnd      // seeded PRNG: rnd.randn(seed, i), rnd.shuffle(seed, arr)
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

// ── the math helpers (figures + prose can share one computation) ──
#let conv2d        = tg.conv2d
#let conv-out-size = tg.conv-out-size
#let softmax-rows  = tg.softmax-rows
