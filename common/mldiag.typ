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

#import "@local/ml-theme:0.1.0": theme
#import "@local/tensor-grid:0.1.0" as tg
#import "@local/ml-plot:0.1.0" as mp
#import "@local/ml-data:0.1.0" as md    // pandas-lite: md.frame(csv("x.csv")), md.xy(f, "a", "b")
#import "@local/ml-dist:0.1.0" as dist  // distributions: dist.nll0(dist.normal(), r) — the TRUE loss
#import "@local/ml-field:0.1.0" as fld  // 2-D/3-D fields: contour/heatmap/surface of f(x,y)
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
#let descent = fld.descent   // compute a GD/momentum trajectory in-deck → contour(paths:)

// ── the math helpers (figures + prose can share one computation) ──
#let conv2d        = tg.conv2d
#let conv-out-size = tg.conv-out-size
#let softmax-rows  = tg.softmax-rows
