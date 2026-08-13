// field — 2-D and 3-D plots of a function f(x, y): heatmaps, iso-contours,
// and surfaces. Native Typst on CeTZ, themed through theme. Everything is
// SAMPLED FROM THE FUNCTION, so a loss landscape or a posterior contour is the
// real field, not a drawing — and a descent path can be overlaid on the same
// coordinates the contour is drawn in.
//
//   #import "vendor/chalkdust/field/lib.typ": *
//   #contour((x, y) => x*x + 3*y*y, xlim: (-3, 3), ylim: (-3, 3),
//            paths: (((-2.5, 2.5), (-1.2, 0.8), (-0.4, 0.2), (0, 0)),),
//            marks: ((0, 0, [min]),))

#import "@preview/cetz:0.5.2"
#import "../theme/lib.typ": default-theme, theme, clamp, norm, ramp-color

#let _is-num(v) = type(v) == int or type(v) == float
#let _limits(lim, name) = {
  assert(type(lim) == array and lim.len() == 2 and lim.at(1) > lim.at(0),
    message: "field." + name + ": limits must be an increasing pair")
}
#let _sampling(xlim, ylim, samples, name) = {
  _limits(xlim, name); _limits(ylim, name)
  assert(type(samples) == int and samples >= 2,
    message: "field." + name + ": samples must be an integer >= 2")
}

// sample f on an nx×ny grid over [x0,x1]×[y0,y1]; z.at(iy).at(ix)
#let _sample(fn, xlim, ylim, nx, ny, name) = {
  assert(type(fn) == function, message: "field." + name + ": fn must be a function")
  let (x0, x1) = xlim
  let (y0, y1) = ylim
  let z = range(ny).map(iy => range(nx).map(ix => {
    let x = x0 + (x1 - x0) * ix / (nx - 1)
    let y = y0 + (y1 - y0) * iy / (ny - 1)
    fn(x, y)
  }))
  assert(z.all(row => row.all(_is-num)),
    message: "field." + name + ": fn must return numeric values")
  z
}
#let _flat-min-max(z) = { let f = z.flatten(); (calc.min(..f), calc.max(..f)) }

// Descent trajectories to overlay via contour(paths:) come from the sibling
// `optim` package (gd / momentum / nesterov / rmsprop / adam / sgd) — this
// package stays pure field-drawing.

// ═══════════════════════════ heatmap ═══════════════════════════
#let heatmap(
  fn, xlim: (-3, 3), ylim: (-3, 3), samples: 44, cell: 1.7mm,
  ramp: auto, contour-lines: 0, theme: default-theme,
) = {
  let t = theme
  _sampling(xlim, ylim, samples, "heatmap")
  assert(type(contour-lines) == int and contour-lines >= 0,
    message: "field.heatmap: contour-lines must be a non-negative integer")
  let (nx, ny) = (samples, samples)
  let z = _sample(fn, xlim, ylim, nx, ny, "heatmap")
  let (zmin, zmax) = _flat-min-max(z)
  let rmp = if ramp == auto { t.ramp } else { ramp }
  cetz.canvas({
    import cetz.draw: rect, line
    for iy in range(ny) {
      for ix in range(nx) {
        rect((ix * cell, iy * cell), ((ix + 1) * cell, (iy + 1) * cell),
          fill: ramp-color(norm(z.at(iy).at(ix), zmin, zmax), rmp), stroke: none)
      }
    }
    // Optional isolines share the same sampled field, so the overlay cannot drift.
    if contour-lines > 0 {
      let lv = range(1, contour-lines + 1).map(k => zmin + (zmax - zmin) * k / (contour-lines + 1))
      for L in lv {
        for iy in range(ny - 1) {
          for ix in range(nx - 1) {
            let bl = z.at(iy).at(ix)
            let br = z.at(iy).at(ix + 1)
            let tr = z.at(iy + 1).at(ix + 1)
            let tl = z.at(iy + 1).at(ix)
            let pts = ()
            if (bl - L) * (br - L) < 0 { let g = (L - bl) / (br - bl); pts.push(((ix + g + 0.5) * cell, (iy + 0.5) * cell)) }
            if (br - L) * (tr - L) < 0 { let g = (L - br) / (tr - br); pts.push(((ix + 1.5) * cell, (iy + g + 0.5) * cell)) }
            if (tr - L) * (tl - L) < 0 { let g = (L - tr) / (tl - tr); pts.push(((ix + 1.5 - g) * cell, (iy + 1.5) * cell)) }
            if (tl - L) * (bl - L) < 0 { let g = (L - tl) / (bl - tl); pts.push(((ix + 0.5) * cell, (iy + 1.5 - g) * cell)) }
            if pts.len() == 2 { line(pts.at(0), pts.at(1), stroke: 0.75pt + t.ink.transparentize(25%)) }
            else if pts.len() == 4 {
              line(pts.at(0), pts.at(1), stroke: 0.75pt + t.ink.transparentize(25%))
              line(pts.at(2), pts.at(3), stroke: 0.75pt + t.ink.transparentize(25%))
            }
          }
        }
      }
    }
  })
}

// ═══════════════════════════ contour ═══════════════════════════
// Iso-contours via marching squares. `levels` may be an int (that many, spread
// between min and max) or an explicit array of values. Overlay `paths` (each a
// list of (x,y) in data coords) and `marks` (x, y[, label[, colour]]).
#let contour(
  fn, xlim: (-3, 3), ylim: (-3, 3), samples: 56, levels: 8,
  size: (44mm, 44mm), color: auto, colors: none, fill: false, ramp: auto,
  paths: (), marks: (), x-label: none, y-label: none, title: none,
  theme: default-theme,
) = {
  let t = theme
  _sampling(xlim, ylim, samples, "contour")
  assert((type(levels) == int and levels > 0) or
    (type(levels) == array and levels.len() > 0 and levels.all(v => type(v) == int or type(v) == float)),
    message: "field.contour: levels must be a positive integer or non-empty numeric array")
  let (nx, ny) = (samples, samples)
  let (x0, x1) = xlim
  let (y0, y1) = ylim
  let (w, h) = size
  let rmp = if ramp == auto { t.ramp } else { ramp }
  // data (x, y) → screen
  let sx(x) = (x - x0) / (x1 - x0) * w
  let sy(y) = (y - y0) / (y1 - y0) * h
  // grid-index → screen
  let gx(ix) = ix / (nx - 1) * w
  let gy(iy) = iy / (ny - 1) * h
  // one function, or several overlaid families (likelihood / prior / posterior)
  let fns = if type(fn) == function { (fn,) } else { fn }
  assert(type(fns) == array and fns.len() > 0 and fns.all(f => type(f) == function),
    message: "field.contour: fn must be a function or non-empty array of functions")
  assert(type(paths) == array and paths.all(p => type(p) == array and
    p.all(q => type(q) == array and q.len() >= 2 and _is-num(q.at(0)) and _is-num(q.at(1)))),
    message: "field.contour: every path must contain numeric (x, y) points")
  assert(type(marks) == array and marks.all(m => type(m) == array and m.len() >= 2 and
    _is-num(m.at(0)) and _is-num(m.at(1))),
    message: "field.contour: marks must contain numeric (x, y) points")

  cetz.canvas({
    import cetz.draw: rect, line, content, circle
    for (fi, f) in fns.enumerate() {
      let z = _sample(f, xlim, ylim, nx, ny, "contour")
      let (zmin, zmax) = _flat-min-max(z)
      let lv = if type(levels) == array { levels } else {
        range(1, levels + 1).map(k => zmin + (zmax - zmin) * k / (levels + 1)) }
      // colour: explicit per-fn, single override, ramp (1 fn) or cycle (many)
      let single = fns.len() == 1
      let fcol = if colors != none and fi < colors.len() { colors.at(fi) }
        else if color != auto { color }
        else if single { none }                                   // per-level ramp below
        else { t.cycle.at(calc.rem(fi, t.cycle.len())) }
      // optional filled bands (only meaningful for a single field)
      if fill and single {
        for iy in range(ny - 1) {
          for ix in range(nx - 1) {
            let v = (z.at(iy).at(ix) + z.at(iy).at(ix + 1) + z.at(iy + 1).at(ix) + z.at(iy + 1).at(ix + 1)) / 4
            rect((gx(ix), gy(iy)), (gx(ix + 1), gy(iy + 1)),
              fill: ramp-color(norm(v, zmin, zmax), rmp).transparentize(35%), stroke: none)
          }
        }
      }
      for L in lv {
        let lc = if fcol != none { fcol } else { ramp-color(norm(L, zmin, zmax), rmp) }
        for iy in range(ny - 1) {
          for ix in range(nx - 1) {
            let bl = z.at(iy).at(ix)
            let br = z.at(iy).at(ix + 1)
            let tr = z.at(iy + 1).at(ix + 1)
            let tl = z.at(iy + 1).at(ix)
            let pts = ()
            if (bl - L) * (br - L) < 0 { let g = (L - bl) / (br - bl); pts.push((gx(ix + g), gy(iy))) }
            if (br - L) * (tr - L) < 0 { let g = (L - br) / (tr - br); pts.push((gx(ix + 1), gy(iy + g))) }
            if (tr - L) * (tl - L) < 0 { let g = (L - tr) / (tl - tr); pts.push((gx(ix + 1 - g), gy(iy + 1))) }
            if (tl - L) * (bl - L) < 0 { let g = (L - tl) / (bl - tl); pts.push((gx(ix), gy(iy + 1 - g))) }
            if pts.len() == 2 { line(pts.at(0), pts.at(1), stroke: 1pt + lc) }
            else if pts.len() == 4 { line(pts.at(0), pts.at(1), stroke: 1pt + lc); line(pts.at(2), pts.at(3), stroke: 1pt + lc) }
          }
        }
      }
    }
    // overlaid descent paths (data coords)
    for (pi, p) in paths.enumerate() {
      let pc = t.cycle.at(calc.rem(pi + 1, t.cycle.len()))
      let sp = p.map(q => (sx(q.at(0)), sy(q.at(1))))
      if sp.len() >= 2 { for k in range(sp.len() - 1) { line(sp.at(k), sp.at(k + 1), stroke: 1.6pt + pc) } }
      for q in sp { circle(q, radius: 1.6pt, fill: pc, stroke: none) }
    }
    // marked points: (x, y) · (x, y, label) · (x, y, label, colour)
    for m in marks {
      let (mx, my) = (sx(m.at(0)), sy(m.at(1)))
      let mcol = if m.len() > 3 { m.at(3) } else { t.accent }
      circle((mx, my), radius: 2.4pt, fill: mcol, stroke: 0.6pt + t.paper)
      if m.len() > 2 { content((mx + 0.4em, my + 0.4em), text(size: 8.5pt, weight: 600, fill: mcol, m.at(2)), anchor: "west") }
    }
    // frame + labels
    rect((0, 0), (w, h), stroke: t.frame-stroke + t.ink, fill: none)
    if x-label != none { content((w / 2, -1.4em), text(size: 9pt, fill: t.muted, x-label), anchor: "north") }
    if y-label != none { content((-1.4em, h / 2), rotate(-90deg, text(size: 9pt, fill: t.muted, y-label)), anchor: "south") }
    if title != none { content((w / 2, h + 1em), text(size: 10pt, fill: t.ink, weight: 600, title), anchor: "south") }
  })
}

// ═══════════════════════════ surface ═══════════════════════════
// A 3-D surface of f(x,y) drawn as filled quads in an oblique projection,
// painted back-to-front (painter's algorithm) and shaded by height.
#let surface(
  fn, xlim: (-3, 3), ylim: (-3, 3), samples: 26,
  cell: 3.4mm, depth: 0.55, height: 26mm, ramp: auto,
  x-label: none, y-label: none, z-label: none, title: none,
  theme: default-theme,
) = {
  let t = theme
  _sampling(xlim, ylim, samples, "surface")
  let n = samples
  let z = _sample(fn, xlim, ylim, n, n, "surface")
  let (zmin, zmax) = _flat-min-max(z)
  let rmp = if ramp == auto { t.ramp } else { ramp }
  // oblique projection: x → right, y → up-right (depth), z → up
  let px(ix, iy) = ix * cell + iy * depth * cell
  let py(ix, iy, v) = iy * depth * cell * 0.5 + norm(v, zmin, zmax) * height
  cetz.canvas({
    import cetz.draw: line, content
    // draw quads from the BACK (high iy) to the FRONT (low iy) so near hides far
    for iy in range(n - 1).rev() {
      for ix in range(n - 1) {
        let v = (z.at(iy).at(ix) + z.at(iy).at(ix + 1) + z.at(iy + 1).at(ix) + z.at(iy + 1).at(ix + 1)) / 4
        let a = (px(ix, iy), py(ix, iy, z.at(iy).at(ix)))
        let b = (px(ix + 1, iy), py(ix + 1, iy, z.at(iy).at(ix + 1)))
        let c = (px(ix + 1, iy + 1), py(ix + 1, iy + 1, z.at(iy + 1).at(ix + 1)))
        let d = (px(ix, iy + 1), py(ix, iy + 1, z.at(iy + 1).at(ix)))
        line(a, b, c, d, close: true,
          fill: ramp-color(norm(v, zmin, zmax), rmp), stroke: 0.3pt + t.ink.transparentize(55%))
      }
    }
    if title != none {
      content((px(n / 2, n / 2), height + 1.2em), text(size: 10pt, fill: t.ink, weight: 600, title), anchor: "south")
    }
    if x-label != none {
      content((px((n - 1) / 2, 0), -0.7em), text(size: 8.5pt, fill: t.muted, x-label), anchor: "north")
    }
    if y-label != none {
      content((px(n - 1, 0) + 0.7em, -0.1em),
        text(size: 8.5pt, fill: t.muted, y-label), anchor: "west")
    }
    if z-label != none {
      content((-0.7em, height / 2), rotate(-90deg, text(size: 8.5pt, fill: t.muted, z-label)), anchor: "south")
    }
  })
}
