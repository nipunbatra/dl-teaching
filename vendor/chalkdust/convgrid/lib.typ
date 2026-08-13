// convgrid — conv arithmetic, annotated grids, pooling, receptive fields,
// patchify and attention heatmaps for ML/DL teaching. Native Typst on CeTZ.
//
// Design rules:
//  · every figure computes its numbers IN TYPST (conv outputs, softmax rows,
//    receptive-field growth) — the picture cannot disagree with the math
//  · `step` is a first-class parameter — subslide loops animate for free
//  · all styling flows through a semantic theme dict (see theme)
//
//   #import "vendor/chalkdust/convgrid/lib.typ": *
//   #conv-op(input: X, kernel: K, step: 0)

#import "@preview/cetz:0.5.2"
#import "../theme/lib.typ": default-theme, theme, clamp, norm, ramp-color, contrast-text

// ═══════════════════════════ data helpers ═══════════════════════════

// (rows, cols) of a spec that is an int (n×n), an (r, c) pair, or a 2-D array
#let _shape(v) = {
  if type(v) == int {
    assert(v > 0, message: "convgrid: grid size must be positive")
    (v, v)
  }
  else if type(v) == array and v.len() > 0 and type(v.first()) == array {
    assert(v.first().len() > 0 and v.all(r => type(r) == array and r.len() == v.first().len()),
      message: "convgrid: value grids must be non-empty and rectangular")
    (v.len(), v.first().len())
  } else {
    assert(type(v) == array and v.len() == 2 and v.all(n => type(n) == int and n > 0),
      message: "convgrid: shape must be a positive integer or (rows, cols)")
    (v.at(0), v.at(1))
  }
}

// 2-D array of values, or none if the spec is shape-only
#let _values(v) = {
  if type(v) == array and v.len() > 0 and type(v.first()) == array { v } else { none }
}

#let _is-num(v) = type(v) == int or type(v) == float

// "1", "-1", "0.25", "3.5" — trimmed number display
#let _fmt-num(v) = {
  if not _is-num(v) { return v }
  if calc.abs(v - calc.round(v)) < 1e-9 { str(int(calc.round(v))) }
  else { str(calc.round(v, digits: 2)) }
}

#let _min-max(values) = {
  let flat = values.flatten().filter(_is-num)
  if flat.len() == 0 { (0, 1) } else { (calc.min(..flat), calc.max(..flat)) }
}

// conv output size:  o = ⌊(n + 2p − d(k−1) − 1) / s⌋ + 1
#let conv-out-size(n, k, stride: 1, padding: 0, dilation: 1) = {
  assert((n, k, stride, padding, dilation).all(v => type(v) == int),
    message: "convgrid.conv-out-size: arguments must be integers")
  assert(n > 0 and k > 0 and stride > 0 and padding >= 0 and dilation > 0,
    message: "convgrid.conv-out-size: require n,k,stride,dilation > 0 and padding >= 0")
  calc.max(0, calc.floor((n + 2 * padding - dilation * (k - 1) - 1) / stride) + 1)
}

// cross-correlate values (2-D) with kernel (2-D); zero padding
#let conv2d(input, kernel, stride: 1, padding: 0, dilation: 1) = {
  let (ir, ic) = _shape(input)
  let (kr, kc) = _shape(kernel)
  assert(input.flatten().all(_is-num) and kernel.flatten().all(_is-num),
    message: "convgrid.conv2d: input and kernel must be numeric matrices")
  let orr = conv-out-size(ir, kr, stride: stride, padding: padding, dilation: dilation)
  let oc = conv-out-size(ic, kc, stride: stride, padding: padding, dilation: dilation)
  assert(orr > 0 and oc > 0, message: "convgrid.conv2d: effective kernel does not fit the padded input")
  let out = ()
  for i in range(orr) {
    let row = ()
    for j in range(oc) {
      let acc = 0
      for a in range(kr) {
        for b in range(kc) {
          let r = i * stride - padding + a * dilation
          let c = j * stride - padding + b * dilation
          if r >= 0 and r < ir and c >= 0 and c < ic {
            acc += input.at(r).at(c) * kernel.at(a).at(b)
          }
        }
      }
      row.push(acc)
    }
    out.push(row)
  }
  out
}

// row-wise softmax
#let softmax-rows(values) = {
  let _ = _shape(values)
  assert(values.flatten().all(_is-num), message: "convgrid.softmax-rows: values must be numeric")
  values.map(row => {
  let m = calc.max(..row)
  let e = row.map(v => calc.exp(v - m))
  let s = e.sum()
  e.map(v => v / s)
  })
}


// text sized for figure internals — also pins math to the surrounding size so
// document-level `show math.equation: set text(..)` rules can't blow up labels
#let _txt(size, body, fill: black, weight: 400) = text(
  size: size, fill: fill, weight: weight, {
    // absolute size: `1em` would resolve against a document-level
    // `show math.equation: set text(..)` rule instead of resetting it
    show math.equation: set text(size: size)
    body
  })

// ═══════════════════════════ core grid renderer ═════════════════════
// Draws one annotated grid inside a cetz canvas, top-left at `origin`,
// row 0 at the top. Returns cetz draw elements — composable.

#let grid-draw(
  origin, values,                 // values: 2-D array of numbers/content/none, or int/(r,c) shape
  cell: 9mm,
  fill: auto,                     // auto → ramp | none | color | fn(v, i, j) → color/none
  vmin: auto, vmax: auto,
  fmt: auto,                      // auto → numbers | none | fn(v, i, j) → content
  text-fill: auto,                // auto → contrast | color | fn(v, i, j) → color
  stroke: auto,                   // cell-edge stroke; auto → theme
  frame: false,                   // heavy outer frame
  highlight: (),                  // ((row, col), …) or (((row, col), color), …)
  divider: none,                  // int n → heavy separators every n cells
  row-labels: none, col-labels: none,
  title: none,                    // label centered above the grid
  theme: default-theme,
) = {
  import cetz.draw: rect, line, content
  let t = theme
  assert(cell > 0pt, message: "convgrid.grid-draw: cell must be positive")
  let shape = _shape(values)
  let (nr, nc) = shape
  let vals = _values(values)
  let (x0, y0) = origin
  let (lo, hi) = if vals != none { _min-max(vals) } else { (0, 1) }
  if vmin != auto { lo = vmin }
  if vmax != auto { hi = vmax }
  let cs = if stroke == auto { t.grid-stroke + t.ink } else { stroke }
  let fsize = t.cell-text * cell
  let lsize = t.label-text * cell

  // cells
  for i in range(nr) {
    for j in range(nc) {
      let v = if vals != none { vals.at(i).at(j) } else { none }
      let f = if fill == none { none }
        else if fill == auto {
          if _is-num(v) { ramp-color(norm(v, lo, hi), t.ramp) } else { t.paper }
        }
        else if type(fill) == function { fill(v, i, j) }
        else { fill }
      let tl = (x0 + j * cell, y0 - i * cell)
      let br = (x0 + (j + 1) * cell, y0 - (i + 1) * cell)
      rect(tl, br, fill: f, stroke: cs)
      // cell text
      let body = if fmt == none { none }
        else if type(fmt) == function { fmt(v, i, j) }
        else if v == none { none }
        else if _is-num(v) { _fmt-num(v) }
        else { v }
      if body != none {
        let tf = if text-fill == auto {
          if _is-num(v) and fill == auto { contrast-text(norm(v, lo, hi), t) } else { t.ink }
        } else if type(text-fill) == function { text-fill(v, i, j) } else { text-fill }
        content((x0 + (j + 0.5) * cell, y0 - (i + 0.5) * cell),
          _txt(fsize, body, fill: tf, weight: 600))
      }
    }
  }

  // heavy separators (pooling quadrants, patch borders)
  if divider != none {
    assert(type(divider) == int and divider > 0, message: "convgrid.grid-draw: divider must be a positive integer")
    let d = t.frame-stroke + t.ink
    for k in range(0, nr + 1, step: divider) {
      line((x0, y0 - k * cell), (x0 + nc * cell, y0 - k * cell), stroke: d)
    }
    for k in range(0, nc + 1, step: divider) {
      line((x0 + k * cell, y0), (x0 + k * cell, y0 - nr * cell), stroke: d)
    }
  }
  if frame {
    rect((x0, y0), (x0 + nc * cell, y0 - nr * cell), stroke: t.frame-stroke + t.ink)
  }

  // highlight windows: ((row, col)) or (((row, col), color))
  for h in highlight {
    let (pos, hcol) = if type(h.first()) == array { (h.first(), h.last()) } else { (h, t.accent) }
    let (i, j) = pos
    rect((x0 + j * cell, y0 - i * cell), (x0 + (j + 1) * cell, y0 - (i + 1) * cell),
      stroke: t.window-stroke + hcol, fill: none)
  }

  // labels
  if col-labels != none {
    assert(col-labels.len() == nc, message: "convgrid.grid-draw: column label count must match columns")
    for (j, l) in col-labels.enumerate() {
      content((x0 + (j + 0.5) * cell, y0 + 0.45 * cell),
        _txt(lsize, l, fill: t.ink))
    }
  }
  if row-labels != none {
    assert(row-labels.len() == nr, message: "convgrid.grid-draw: row label count must match rows")
    for (i, l) in row-labels.enumerate() {
      content((x0 - 0.45 * cell, y0 - (i + 0.5) * cell), anchor: "east",
        _txt(lsize, l, fill: t.ink))
    }
  }
  if title != none {
    let dy = if col-labels != none { 1.25 * cell } else { 0.55 * cell }
    content((x0 + nc * cell / 2, y0 + dy), anchor: "south",
      _txt(lsize, title, fill: t.ink))
  }
}

// a thick rectangle outlining the k×k window whose top-left CELL is (i, j)
#let window-draw(origin, i, j, k, cell: 9mm, color: none, theme: default-theme, dash: none) = {
  import cetz.draw: rect, line, content
  let (x0, y0) = origin
  let c = if color == none { theme.accent } else { color }
  rect((x0 + j * cell, y0 - i * cell), (x0 + (j + k) * cell, y0 - (i + k) * cell),
    stroke: (paint: c, thickness: theme.window-stroke, dash: dash), fill: none)
}

// standalone annotated grid / heatmap / confusion matrix / mask
#let grid-map(values, ..args) = {
  cetz.canvas(grid-draw((0mm, 0mm), values, ..args))
}

// ═══════════════════════════ convolution ════════════════════════════
// One frame of a 2-D convolution: input (with padding ring) ∗ kernel → output.
// `step` selects the window position (int index in row-major output order, or
// (i, j)); numeric inputs compute the true output values in Typst.

#let conv-op(
  input: 5,                       // int | (r, c) | 2-D values
  kernel: 3,                      // int | 2-D values
  stride: 1, padding: 0, dilation: 1,
  step: 0,                        // int | (i, j) | none (no window)
  accumulate: true,               // fill output cells already produced
  show-expr: false,               // multiply–add annotation under the figure (numeric only)
  labels: auto,                   // (input: .., kernel: .., output: ..) or none
  cell: 9mm,
  gap: auto,                      // horizontal gap between the three grids
  theme: default-theme,
) = {
  let t = theme
  assert(cell > 0pt, message: "convgrid.conv-op: cell must be positive")
  let (ir, ic) = _shape(input)
  let (kr, kc) = _shape(kernel)
  let ivals = _values(input)
  let kvals = _values(kernel)
  let orr = conv-out-size(ir, kr, stride: stride, padding: padding, dilation: dilation)
  let oc = conv-out-size(ic, kc, stride: stride, padding: padding, dilation: dilation)
  assert(orr > 0 and oc > 0, message: "convgrid.conv-op: effective kernel does not fit the padded input")
  let ovals = if ivals != none and kvals != none {
    conv2d(ivals, kvals, stride: stride, padding: padding, dilation: dilation)
  } else { none }
  let lbl = if labels == auto {
    (input: [input #h(0.3em) $X$], kernel: [kernel #h(0.3em) $K$], output: [output])
  } else if labels == none { (:) } else { labels }
  let g = if gap == auto { 1.6 * cell } else { gap }

  // step → output position (oi, oj); window top-left in padded input coords
  let (oi, oj) = if step == none { (none, none) }
    else if type(step) == int {
      assert(step >= 0 and step < orr * oc, message: "convgrid.conv-op: step index out of bounds")
      (calc.floor(step / oc), calc.rem(step, oc))
    } else {
      assert(type(step) == array and step.len() == 2 and step.all(v => type(v) == int),
        message: "convgrid.conv-op: step must be an integer, (row, col), or none")
      assert(step.at(0) >= 0 and step.at(0) < orr and step.at(1) >= 0 and step.at(1) < oc,
        message: "convgrid.conv-op: step position out of bounds")
      step
    }
  let step-idx = if oi == none { none } else { oi * oc + oj }

  // padded input values (pad cells hold 0 in numeric mode)
  let pr = ir + 2 * padding
  let pc = ic + 2 * padding
  let pvals = if ivals == none { (pr, pc) } else {
    range(pr).map(i => range(pc).map(j => {
      let (r, c) = (i - padding, j - padding)
      if r >= 0 and r < ir and c >= 0 and c < ic { ivals.at(r).at(c) } else { 0 }
    }))
  }
  let is-pad(i, j) = i < padding or i >= ir + padding or j < padding or j >= ic + padding
  let (ilo, ihi) = if ivals != none { _min-max(ivals) } else { (0, 1) }

  // heights (for vertical centering on a shared midline)
  let hi-in = pr * cell
  let hi-k = kr * cell
  let hi-out = orr * cell
  let ymid = 0mm

  cetz.canvas({
    import cetz.draw: rect, line, content
    // ── input grid (pad cells: muted, dashed edge, no ramp) ──
    let xin = 0mm
    let yin = ymid + hi-in / 2
    grid-draw((xin, yin), pvals, cell: cell, theme: t,
      vmin: ilo, vmax: ihi,
      fill: (v, i, j) => if is-pad(i, j) { t.bg.transparentize(30%) }
        else if _is-num(v) { ramp-color(norm(v, ilo, ihi), t.ramp) } else { t.paper },
      fmt: (v, i, j) => if is-pad(i, j) {
          if _is-num(v) { text(fill: t.muted, str(0)) } else { none }
        } else if _is-num(v) { _fmt-num(v) } else { none },
      stroke: t.grid-stroke + t.ink,
      title: lbl.at("input", default: none))
    if padding > 0 {
      // dashed outline separating pad ring from the true input
      rect((xin + padding * cell, yin - padding * cell),
           (xin + (padding + ic) * cell, yin - (padding + ir) * cell),
        stroke: (paint: t.muted, thickness: t.frame-stroke, dash: "dashed"))
    }
    // window at the current step (kernel footprint incl. dilation)
    if oi != none {
      let wi = oi * stride
      let wj = oj * stride
      let foot = dilation * (kr - 1) + 1
      window-draw((xin, yin), wi, wj, foot, cell: cell, theme: t)
      if dilation > 1 {
        // mark the actual taps
        for a in range(kr) {
          for b in range(kc) {
            window-draw((xin, yin), wi + a * dilation, wj + b * dilation, 1,
              cell: cell, theme: t, color: t.accent)
          }
        }
      }
    }

    // ── ∗ kernel ──
    let xker = xin + pc * cell + g
    content((xker - g / 2, ymid), _txt(0.55 * cell, $ast$, fill: t.ink))
    let yker = ymid + hi-k / 2
    grid-draw((xker, yker), if kvals != none { kvals } else { kernel },
      cell: cell, theme: t, title: lbl.at("kernel", default: none))

    // ── → output ──
    let xarr = xker + kc * cell + 0.35 * cell
    let xout = xarr + g + 0.5 * cell
    line((xarr, ymid), (xout - 0.35 * cell, ymid),
      stroke: t.arrow-stroke + t.ink, mark: (end: "stealth", fill: t.ink))
    let yout = ymid + hi-out / 2
    let (olo, ohi) = if ovals != none { _min-max(ovals) } else { (0, 1) }
    grid-draw((xout, yout), if ovals != none { ovals } else { (orr, oc) },
      cell: cell, theme: t,
      vmin: olo, vmax: ohi,
      fill: (v, i, j) => {
        let idx = i * oc + j
        if step-idx != none and idx > step-idx and accumulate { t.paper }
        else if _is-num(v) { ramp-color(norm(v, olo, ohi), t.ramp) }
        else if step-idx != none and idx <= step-idx { t.bg } else { t.paper }
      },
      fmt: (v, i, j) => {
        let idx = i * oc + j
        if step-idx != none and idx > step-idx and accumulate { none }
        else if _is-num(v) { _fmt-num(v) } else { none }
      },
      highlight: if oi != none { (((oi, oj), t.accent),) } else { () },
      title: lbl.at("output", default: none))

    // ── multiply–add expression under the figure ──
    if show-expr and ovals != none and oi != none {
      let terms = ()
      for a in range(kr) {
        let rowsum = ()
        for b in range(kc) {
          let r = oi * stride + a * dilation
          let c = oj * stride + b * dilation
          rowsum.push(_fmt-num(pvals.at(r).at(c)) + "·" + _fmt-num(kvals.at(a).at(b)))
        }
        terms.push("(" + rowsum.join(" + ") + ")")
      }
      let total = ovals.at(oi).at(oj)
      let xmid = (xout + oc * cell) / 2
      content((xmid, ymid - calc.max(hi-in, hi-out) / 2 - 0.9 * cell), {
        _txt(t.label-text * cell, terms.join("  +  "), fill: t.muted)
        _txt(t.label-text * cell, "  =  " + _fmt-num(total), fill: t.accent, weight: 700)
      })
    }
  })
}

// ═══════════════════════════ pooling ════════════════════════════════
// input → max/avg pooled outputs; quadrant tints when stride == window

#let _pool(vals, window, stride, kind) = {
  let (ir, ic) = _shape(vals)
  assert(vals.flatten().all(_is-num), message: "convgrid.pool: values must be numeric")
  assert(type(window) == int and window > 0 and type(stride) == int and stride > 0,
    message: "convgrid.pool: window and stride must be positive integers")
  assert(window <= ir and window <= ic, message: "convgrid.pool: window must fit the input")
  assert(("max", "avg").contains(kind), message: "convgrid.pool: kind must be 'max' or 'avg'")
  let orr = calc.floor((ir - window) / stride) + 1
  let oc = calc.floor((ic - window) / stride) + 1
  range(orr).map(i => range(oc).map(j => {
    let xs = ()
    for a in range(window) {
      for b in range(window) { xs.push(vals.at(i * stride + a).at(j * stride + b)) }
    }
    if kind == "max" { calc.max(..xs) } else { xs.sum() / xs.len() }
  }))
}

#let pool-op(
  values,                         // 2-D numeric array
  window: 2, stride: auto,        // stride defaults to window (non-overlapping)
  kinds: ("max",),                // ("max",) | ("avg",) | ("max", "avg")
  cell: 9mm,
  labels: auto,
  theme: default-theme,
) = {
  let t = theme
  let s = if stride == auto { window } else { stride }
  let (ir, ic) = _shape(values)
  assert(type(kinds) == array and kinds.len() > 0 and kinds.all(k => ("max", "avg").contains(k)),
    message: "convgrid.pool-op: kinds must contain 'max' and/or 'avg'")
  let (lo, hi) = _min-max(values)
  let outs = kinds.map(k => _pool(values, window, s, k))
  let kcol = (max: t.accent, avg: t.accent2)
  let lbl = if labels == auto {
    (input: [input #h(0.3em) $#ir times #ic$],) } else if labels == none { (:) } else { labels }

  cetz.canvas({
    import cetz.draw: rect, line, content
    let hin = ir * cell
    let yin = hin / 2
    // quadrant tints behind the grid (only for non-overlapping pooling)
    if s == window {
      for i in range(0, ir, step: window) {
        for j in range(0, ic, step: window) {
          let even = calc.even(calc.floor(i / window) + calc.floor(j / window))
          let f = if even { t.accent } else { t.accent2 }
          rect((j * cell, yin - i * cell),
               (calc.min(j + window, ic) * cell, yin - calc.min(i + window, ir) * cell),
            fill: f.transparentize(88%), stroke: none)
        }
      }
    }
    grid-draw((0mm, yin), values, cell: cell, theme: t, fill: none,
      vmin: lo, vmax: hi, title: lbl.at("input", default: none),
      divider: if s == window { window } else { none })

    // one arrow + output grid per kind, stacked vertically
    let xarr = ic * cell + 0.5 * cell
    let xout = xarr + 3.6 * cell
    let n = outs.len()
    for (idx, k) in kinds.enumerate() {
      let out = outs.at(idx)
      let (orr, occ) = _shape(out)
      let block = orr * cell + 1.2 * cell
      let ymid = (n - 1) * block / 2 - idx * block
      let col = kcol.at(k)
      line((xarr, ymid), (xout - 0.5 * cell, ymid),
        stroke: t.arrow-stroke + col, mark: (end: "stealth", fill: col))
      content(((xarr + xout - 0.5 * cell) / 2, ymid + 0.42 * cell),
        _txt(t.label-text * cell, k + " pool", fill: col))
      grid-draw((xout, ymid + orr * cell / 2), out, cell: cell, theme: t,
        vmin: lo, vmax: hi)
    }
  })
}

// ═══════════════════════════ attention matrix ═══════════════════════
// Annotated T×T heatmap in the QKᵀ style: keys on top, queries on the left,
// white cell separators, optional causal mask (✓ / −∞) and row-softmax.

#let attn-matrix(
  tokens,                         // array of token labels, or (q: (..), k: (..))
  values: none,                   // 2-D scores; none + mask → allowed/masked display
  mask: none,                     // none | "causal" | fn(i, j) → bool (true = masked)
  softmax: false,                 // row-softmax values before display (masked → −∞)
  annotate: true,
  min-annotate: 0.0,              // hide annotations below this value
  boxes: (),                      // ((i, j), …) or ((i, j, color), …) — outline cells (e.g. positives)
  mask-style: "inf",              // masked cells: "inf" (grey + −∞) | "grey" (neutral) | "blank" (empty)
  colorbar: false,
  q-label: [query #h(0.25em) $q_i$],
  k-label: [key #h(0.25em) $k_j$],
  cell: 9mm,
  theme: default-theme,
) = {
  let t = theme
  let (qtok, ktok) = if type(tokens) == dictionary { (tokens.q, tokens.k) } else { (tokens, tokens) }
  assert(type(qtok) == array and qtok.len() > 0 and type(ktok) == array and ktok.len() > 0,
    message: "convgrid.attn-matrix: query/key tokens must be non-empty arrays")
  assert(("inf", "grey", "blank").contains(mask-style),
    message: "convgrid.attn-matrix: unknown mask-style")
  let nq = qtok.len()
  let nk = ktok.len()
  let masked(i, j) = if mask == "causal" { j > i }
    else if type(mask) == function { mask(i, j) } else { false }

  let vals = values
  if vals != none {
    let (vr, vc) = _shape(vals)
    assert(vr == nq and vc == nk and vals.flatten().all(_is-num),
      message: "convgrid.attn-matrix: score matrix must be numeric with shape queries x keys")
  }
  if vals != none and softmax {
    // softmax over unmasked entries only
    vals = range(nq).map(i => {
      let row = range(nk).filter(j => not masked(i, j)).map(j => vals.at(i).at(j))
      assert(row.len() > 0, message: "convgrid.attn-matrix: every softmax row needs an unmasked key")
      let m = calc.max(..row)
      let z = row.map(v => calc.exp(v - m)).sum()
      range(nk).map(j => if masked(i, j) { none } else { calc.exp(vals.at(i).at(j) - m) / z })
    })
  }
  let (lo, hi) = if vals != none {
    _min-max(vals.map(r => r.filter(_is-num)).filter(r => r.len() > 0))
  } else { (0, 1) }

  cetz.canvas({
    import cetz.draw: rect, line, content
    let y0 = nq * cell / 2
    grid-draw((0mm, y0), if vals != none { vals } else { (nq, nk) },
      cell: cell, theme: t, vmin: lo, vmax: hi,
      stroke: 1.3pt + t.paper,     // white separators, matplotlib-heatmap style
      fill: (v, i, j) => {
        if masked(i, j) { if mask-style == "blank" { t.paper } else { t.muted.transparentize(72%) } }
        else if _is-num(v) { ramp-color(norm(v, lo, hi), t.ramp) }
        else { t.accent2.transparentize(15%) }
      },
      fmt: (v, i, j) => {
        if masked(i, j) { if mask-style == "inf" { text(fill: t.negative, weight: 700, size: 0.85em, $-oo$) } else { none } }
        else if not annotate { none }
        else if _is-num(v) { if v >= min-annotate { _fmt-num(v) } else { none } }
        else { text(fill: t.paper, weight: 700, sym.checkmark) }
      },
      text-fill: (v, i, j) => if _is-num(v) { contrast-text(norm(v, lo, hi), t) } else { t.ink },
      col-labels: ktok, row-labels: qtok)
    // outline specific cells (e.g. the positive pair per row of a contrastive matrix)
    for b in boxes {
      assert(type(b) == array and b.len() >= 2 and type(b.at(0)) == int and type(b.at(1)) == int and
        b.at(0) >= 0 and b.at(0) < nq and b.at(1) >= 0 and b.at(1) < nk,
        message: "convgrid.attn-matrix: boxed cell is out of bounds")
      let (bi, bj, bc) = if b.len() > 2 { (b.at(0), b.at(1), b.at(2)) } else { (b.at(0), b.at(1), t.accent) }
      rect((bj * cell, y0 - bi * cell), ((bj + 1) * cell, y0 - (bi + 1) * cell),
        stroke: t.window-stroke + bc, fill: none)
    }
    // axis titles
    content((nk * cell / 2, y0 + 1.15 * cell), anchor: "south",
      _txt(t.label-text * cell, k-label, fill: t.ink))
    content((-1.6 * cell, 0mm), anchor: "center", angle: 90deg,
      _txt(t.label-text * cell, q-label, fill: t.ink))
    // colorbar
    if colorbar and vals != none {
      let xcb = nk * cell + 0.8 * cell
      let hcb = nq * cell
      let nseg = 40
      for k in range(nseg) {
        let f0 = k / nseg
        let f1 = (k + 1) / nseg
        rect((xcb, -hcb / 2 + f0 * hcb), (xcb + 0.42 * cell, -hcb / 2 + f1 * hcb),
          fill: ramp-color(f0, t.ramp), stroke: none)
      }
      content((xcb + 0.65 * cell, hcb / 2), anchor: "west",
        _txt(t.label-text * cell, _fmt-num(hi), fill: t.ink))
      content((xcb + 0.65 * cell, -hcb / 2), anchor: "west",
        _txt(t.label-text * cell, _fmt-num(lo), fill: t.ink))
    }
  })
}

// ═══════════════════════════ receptive field ════════════════════════
// Nested centered squares showing receptive-field growth through a conv
// stack:  r_ℓ = r_{ℓ−1} + (k_ℓ − 1)·∏_{m<ℓ} s_m   — computed, then drawn.

#let receptive-field(
  kernels: (3, 3, 3),
  strides: auto,                  // auto → all 1
  cell: 7mm,
  legend: true,
  labels: auto,                   // per-layer legend labels
  caption: auto,                  // formula line under the grid; none to hide
  theme: default-theme,
) = {
  let t = theme
  assert(type(kernels) == array and kernels.len() > 0 and kernels.all(k => type(k) == int and k > 0),
    message: "convgrid.receptive-field: kernels must be positive integers")
  let ss = if strides == auto { kernels.map(_ => 1) } else { strides }
  assert(type(ss) == array and ss.len() == kernels.len() and ss.all(s => type(s) == int and s > 0),
    message: "convgrid.receptive-field: strides must match kernels and be positive")
  // growth walk
  let rs = (1,)
  let jump = 1
  for (l, k) in kernels.enumerate() {
    rs.push(rs.last() + (k - 1) * jump)
    jump *= ss.at(l)
  }
  let N = rs.last()                        // final RF = base grid size
  let cols = (t.ink,) + range(kernels.len()).map(l =>
    t.cycle.at(calc.min(l + 1, t.cycle.len() - 1)))
  let lbls = if labels == auto {
    ([output cell],) + kernels.enumerate().map(((l, k)) =>
      [after conv #(l + 1): #rs.at(l + 1)#sym.times#rs.at(l + 1)])
  } else { labels }

  cetz.canvas({
    import cetz.draw: rect, line, content
    let y0 = N * cell / 2
    grid-draw((0mm, y0), (N, N), cell: cell, theme: t,
      fill: t.bg.transparentize(45%), stroke: 0.5pt + t.muted.transparentize(40%))
    let c = N / 2
    for (idx, r) in rs.enumerate() {
      let off = c - r / 2
      rect((off * cell, y0 - off * cell), ((off + r) * cell, y0 - (off + r) * cell),
        stroke: t.window-stroke + cols.at(idx), fill: none)
    }
    if legend {
      let xl = N * cell + 0.8 * cell
      for (idx, l) in lbls.enumerate() {
        let yy = y0 - 0.5 * cell - idx * 0.95 * cell
        rect((xl, yy + 0.28 * cell), (xl + 0.56 * cell, yy - 0.28 * cell),
          stroke: 1.8pt + cols.at(idx), fill: none)
        content((xl + 0.85 * cell, yy), anchor: "west",
          _txt(t.label-text * cell, l, fill: t.ink))
      }
    }
    let cap = if caption == auto { $r_ell = r_(ell - 1) + (k - 1) dot product_(m < ell) s_m$ }
      else { caption }
    if cap != none {
      content((N * cell / 2, -y0 - 0.75 * cell),
        _txt(t.label-text * cell, cap, fill: t.ink))
    }
  })
}

// ═══════════════════════════ patchify (ViT) ═════════════════════════
// An image grid cut into p×p patches; optionally exploded with a gap and
// numbered in raster order (the patch-token sequence).

#let patchify(
  image: 8,                       // int | (r, c) | 2-D values
  patch: 4,
  gap: 0mm,                       // > 0 → exploded patches
  numbering: true,                // patch index in the center of each patch
  show-values: false,             // render each cell's value (for value-carrying images)
  mask: (),                       // 0-based raster indices of patches to hide (MAE-style)
  cell: 5mm,
  theme: default-theme,
) = {
  let t = theme
  let (ir, ic) = _shape(image)
  let vals = _values(image)
  assert(type(patch) == int and patch > 0, message: "convgrid.patchify: patch must be a positive integer")
  assert(calc.rem(ir, patch) == 0 and calc.rem(ic, patch) == 0,
    message: "convgrid.patchify: image dimensions must be divisible by patch size")
  let np-r = calc.ceil(ir / patch)
  let np-c = calc.ceil(ic / patch)
  assert(type(mask) == array and mask.all(i => type(i) == int and i >= 0 and i < np-r * np-c),
    message: "convgrid.patchify: mask indices must identify existing patches")
  // GLOBAL colour scale across the whole image, so patches stay comparable —
  // without this each patch self-normalizes and an internally-flat patch loses
  // all cross-patch contrast (e.g. a bright corner patch reads the same as a dark one).
  let (gmin, gmax) = if vals != none { _min-max(vals) } else { (0, 1) }

  cetz.canvas({
    import cetz.draw: rect, line, content
    let y0 = (ir * cell + (np-r - 1) * gap) / 2
    for pi in range(np-r) {
      for pj in range(np-c) {
        let x = pj * (patch * cell + gap)
        let y = y0 - pi * (patch * cell + gap)
        let idx = pi * np-c + pj
        if mask.contains(idx) {
          // a hidden patch (masked autoencoder): grey block, dashed frame, no content
          rect((x, y), (x + patch * cell, y - patch * cell), fill: t.muted.transparentize(58%), stroke: none)
          rect((x, y), (x + patch * cell, y - patch * cell), fill: none,
            stroke: (paint: t.muted, dash: "dashed", thickness: t.grid-stroke))
        } else {
          let sub = if vals != none {
            range(patch).map(a => range(patch).map(b => vals.at(pi * patch + a).at(pj * patch + b)))
          } else { (patch, patch) }
          grid-draw((x, y), sub, cell: cell, theme: t,
            fill: if vals != none { auto } else { t.bg.transparentize(50%) },
            vmin: gmin, vmax: gmax,                         // shared scale (see above)
            fmt: if show-values and vals != none { auto } else { none },
            stroke: 0.5pt + t.muted)
          rect((x, y), (x + patch * cell, y - patch * cell),
            stroke: t.frame-stroke + t.accent, fill: none)
          if numbering {
            content((x + patch * cell / 2, y - patch * cell / 2),
              _txt(0.38 * patch * cell, str(idx + 1),
                fill: t.ink.transparentize(35%), weight: 700))
          }
        }
      }
    }
  })
}
