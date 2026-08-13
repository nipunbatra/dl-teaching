// theme — shared design tokens for the ml diagram package family.
//
// Every sibling package (convgrid, plot, nn-arch, plate-note) takes a
// `theme:` parameter holding this dictionary. Colors are SEMANTIC ROLES, not
// raw colors, so one override restyles every figure. Fonts are never set —
// figures inherit the document's text settings.
//
//   #import "vendor/chalkdust/theme/lib.typ": default-theme, theme
//   #let my-theme = theme(ink: rgb("#23373B"), accent: rgb("#EB811B"))

#let default-theme = (
  // ── semantic colors ──
  ink:      rgb("#26343A"),   // primary text and strokes
  muted:    rgb("#8A959A"),   // secondary text, faint grids, padding
  paper:    white,            // empty-cell background
  bg:       rgb("#EFEEEB"),   // light neutral fill
  accent:   rgb("#E8590C"),   // primary highlight: active window, current cell
  accent2:  rgb("#367E7F"),   // secondary highlight: alt window, second series
  positive: rgb("#188A42"),
  negative: rgb("#D64550"),
  // diverging value ramp for heatmaps/grids: low → mid → high
  ramp: (rgb("#367E7F"), rgb("#EFEEEB"), rgb("#E8590C")),
  // color cycle for multi-series overlays (nested boxes, trajectories, …)
  cycle: (rgb("#26343A"), rgb("#188A42"), rgb("#367E7F"), rgb("#E8590C"), rgb("#D64550")),
  // ── stroke weights ──
  grid-stroke:   0.7pt,       // cell edges
  frame-stroke:  1.6pt,       // grid outer frame / separators
  window-stroke: 2.4pt,       // highlight windows
  arrow-stroke:  1.4pt,
  // ── text sizing (fractions of the cell size) ──
  cell-text:  0.36,           // numbers inside cells
  label-text: 0.32,           // grid titles, axis labels
)

// derive a theme: `theme(accent: red)` or `theme(base: other, accent: red)`
#let theme(base: default-theme, ..overrides) = {
  assert(type(base) == dictionary, message: "theme.theme: base must be a dictionary")
  let t = base
  for (k, v) in overrides.named() {
    assert(k in base, message: "theme.theme: unknown token '" + k + "'")
    t.insert(k, v)
  }
  t
}

// clamp v into [lo, hi]
#let clamp(v, lo, hi) = {
  assert(hi >= lo, message: "theme.clamp: require lo <= hi")
  calc.max(lo, calc.min(hi, v))
}

// normalize v from [vmin, vmax] to [0, 1] (safe when vmin == vmax)
#let norm(v, vmin, vmax) = {
  assert(vmax >= vmin, message: "theme.norm: require vmin <= vmax")
  if vmax - vmin == 0 { 0.5 } else { clamp((v - vmin) / (vmax - vmin), 0, 1) }
}

// map t ∈ [0, 1] through a multi-stop color ramp (oklab interpolation)
#let ramp-color(t, ramp) = {
  assert(type(ramp) == array and ramp.len() > 0 and ramp.all(c => type(c) == color),
    message: "theme.ramp-color: ramp must be a non-empty array of colors")
  if ramp.len() == 1 { return ramp.first() }
  let x = clamp(t, 0, 1) * (ramp.len() - 1)
  let i = calc.min(calc.floor(x), ramp.len() - 2)
  let f = x - i
  color.mix((ramp.at(i), (1 - f) * 100%), (ramp.at(i + 1), f * 100%), space: color.oklab)
}

// readable text color on top of a ramp fill: paper-ish when the value is
// near the ramp extremes, ink otherwise
#let contrast-text(t, theme) = if t > 0.82 or t < 0.12 { theme.paper } else { theme.ink }
