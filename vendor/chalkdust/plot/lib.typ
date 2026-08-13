// plot — general bar & line plots for ML/DL teaching. Native Typst on CeTZ,
// themed through theme (semantic colors, so one override restyles every plot).
//
// Design rules (shared with convgrid):
//  · numbers are computed IN TYPST — a softmax/temperature bar chart derives its
//    own probabilities, so the picture cannot disagree with the math on the slide
//  · everything flows through the semantic theme dict (see theme)
//
//   #import "vendor/chalkdust/plot/lib.typ": *
//   #bars((3.0, 1.0, 0.2), labels: ("cat", "dog", "cow"), softmax: true)
//   #lines(((1,2),(2,4),(3,8),(4,16)), log-y: true)

#import "@preview/cetz:0.5.2"
#import "../theme/lib.typ": default-theme, theme, clamp, norm, ramp-color, contrast-text

// ═══════════════════════════ helpers ═══════════════════════════

#let _is-num(v) = type(v) == int or type(v) == float

#let _fmt-num(v, digits: 2) = {
  if not _is-num(v) { return v }
  if calc.abs(v - calc.round(v)) < 1e-9 { str(int(calc.round(v))) }
  else { str(calc.round(v, digits: digits)) }
}

#let _softmax(xs, temperature: 1.0) = {
  assert(type(xs) == array and xs.len() > 0 and xs.all(_is-num),
    message: "plot: softmax values must be a non-empty numeric array")
  assert(temperature > 0, message: "plot: softmax temperature must be positive")
  let m = calc.max(..xs)
  let e = xs.map(x => calc.exp((x - m) / temperature))
  let z = e.sum()
  e.map(v => v / z)
}

// normalize a bar spec into a list of (label, value, color-or-none)
#let _bar-items(values, labels) = {
  values.enumerate().map(((i, v)) => {
    if type(v) == dictionary {
      (v.at("label", default: if labels != none { labels.at(i) } else { "" }),
       v.value, v.at("color", default: none))
    } else if type(v) == array {
      (v.at(0), v.at(1), if v.len() > 2 { v.at(2) } else { none })
    } else {
      (if labels != none { labels.at(i) } else { "" }, v, none)
    }
  })
}

// which indices are highlighted, and with what color
#let _hi-set(highlight, theme) = {
  if highlight == none { (:) }
  else if type(highlight) == int { ((str(highlight)): theme.accent) }
  else if type(highlight) == array { let d = (:); for i in highlight { d.insert(str(i), theme.accent) }; d }
  else if type(highlight) == dictionary { let d = (:); for (k, c) in highlight { d.insert(str(k), c) }; d }
  else { (:) }
}

// ═══════════════════════════ bars ═══════════════════════════
// A bar chart for distributions, attention weights, cost comparisons, gradients.
//  values: array of numbers | (label, value) pairs | (label:, value:, color:) dicts
#let bars(
  values,
  labels: none,
  horizontal: false,          // false → vertical bars, true → horizontal
  softmax: false,             // treat values as logits → softmax probabilities
  temperature: 1.0,
  highlight: none,            // index | (i, j, …) | (i: color, …) to emphasize
  annotate: true,             // print each value at the bar tip
  baseline: 0,                // value at the axis (supports signed bars, e.g. p−y)
  bar: 10mm,                  // thickness of each bar
  span: 34mm,                 // length of the longest bar
  gap: auto,                  // spacing between bars (auto → bar * 0.55)
  size: none,                 // (width, height) → auto-fit bar thickness & span to this box
  color: auto,                // auto → theme.accent | a color | fn(i, value) → color
  title: none,
  digits: 2,
  theme: default-theme,
) = {
  let t = theme
  assert(type(values) == array and values.len() > 0, message: "plot.bars: values must be non-empty")
  if labels != none { assert(type(labels) == array and labels.len() == values.len(),
    message: "plot.bars: label count must match values") }
  let items = _bar-items(values, labels)
  assert(items.all(it => _is-num(it.at(1))), message: "plot.bars: bar values must be numeric")
  assert(_is-num(baseline), message: "plot.bars: baseline must be numeric")
  if softmax { assert(temperature > 0, message: "plot.bars: temperature must be positive") }
  // fit the whole chart into `size` if given: n bars along the cross axis (each
  // bar + gap ≈ bar·1.55), the value axis spans the other dimension.
  let (bar, span) = if size != none {
    let (w, h) = size
    if horizontal { (h / (items.len() * 1.55), w) } else { (w / (items.len() * 1.55), h) }
  } else { (bar, span) }
  let raw = items.map(it => it.at(1))
  let vals = if softmax { _softmax(raw, temperature: temperature) } else { raw }
  let hi = _hi-set(highlight, t)
  let g = if gap == auto { bar * 0.55 } else { gap }
  let lo = calc.min(baseline, ..vals)
  let h05 = calc.max(baseline, ..vals)
  let range = h05 - lo
  let scale = if range == 0 { 0pt } else { span / range }
  let pos(v) = (v - baseline) * scale                    // signed offset from the baseline
  let has-hi = hi.len() > 0
  let barcolor(i, v) = {
    if str(i) in hi { hi.at(str(i)) }
    else if type(color) == function { color(i, v) }
    else if color == auto {
      // when a highlight is set, dim the other bars so it pops without a manual base colour
      if has-hi { t.muted.lighten(25%) }
      else if v < baseline { t.negative } else { t.accent }
    }
    else { color }
  }

  cetz.canvas({
    import cetz.draw: rect, line, content
    let n = items.len()
    let fs = 0.34 * bar
    let ls = 0.32 * bar
    if horizontal {
      // bars grow rightward; baseline is a vertical line at x = 0
      let base-off = pos(baseline)
      for (i, it) in items.enumerate() {
        let (lbl, _, _) = it
        let v = vals.at(i)
        let y = -i * (bar + g)
        rect((0, y - bar/2), (pos(v), y + bar/2), fill: barcolor(i, v), stroke: none)
        if lbl != "" { content((-0.4em, y), align(right, text(size: ls, fill: t.ink, [#lbl])), anchor: "east") }
        if annotate {
          content((pos(v) + 0.3em * (if v < baseline {-1} else {1}), y),
            text(size: fs, fill: t.muted, _fmt-num(v, digits: digits)),
            anchor: if v < baseline { "east" } else { "west" })
        }
      }
      line((base-off, bar/2 + 1mm), (base-off, -(n - 1) * (bar + g) - bar/2 - 1mm),
        stroke: t.grid-stroke + t.muted)
    } else {
      // vertical bars grow upward from the baseline
      for (i, it) in items.enumerate() {
        let (lbl, _, _) = it
        let v = vals.at(i)
        let x = i * (bar + g)
        rect((x - bar/2, 0), (x + bar/2, pos(v)), fill: barcolor(i, v), stroke: none)
        // clear the category label below the lowest tip; extra room when bars go negative
        let cat-off = if lo < baseline { 1.4em } else { 0.5em }
        if lbl != "" { content((x, pos(lo) - cat-off), text(size: ls, fill: t.ink, [#lbl]), anchor: "north") }
        if annotate {
          content((x, pos(v) + 0.3em * (if v < baseline {-1} else {1})),
            text(size: fs, fill: t.muted, _fmt-num(v, digits: digits)),
            anchor: if v < baseline { "north" } else { "south" })
        }
      }
      line((-bar/2 - 1mm, pos(baseline)), ((n - 1) * (bar + g) + bar/2 + 1mm, pos(baseline)),
        stroke: t.grid-stroke + t.muted)
    }
    if title != none {
      let w = if horizontal { span/2 } else { (n - 1) * (bar + g) / 2 }
      let ty = if horizontal { bar } else { pos(h05) + 1.4em }
      content((w, ty), text(size: 0.4 * bar, fill: t.ink, weight: 600, title), anchor: "south")
    }
  })
}

// ═══════════════════════════ lines ═══════════════════════════
// A multi-series line plot. Each series is an array of (x, y) pairs, or an array
// of y-values (x = index+1). Supports a log-scaled y-axis (receptive fields,
// cost-vs-length, loss curves).
#let lines(
  // Data — give ONE of: a positional `series` (array of (x,y) pairs, or array of
  // series), or `fn` + `domain`, or `x` / `y` arrays.
  fn: none,                   // a function x→y (or array of functions) sampled over `domain`
  domain: none,               // (xmin, xmax) for `fn`
  samples: 80,                // number of samples for `fn`
  x: none, y: none,           // parallel arrays: y = one array or an array of arrays; x optional (else 1..n)
  labels: none,               // per-series legend labels
  colors: none,               // per-series colour override (array); default → theme.cycle
  log-y: false,
  markers: true,             // bool for all series, or a per-series array of bools
  dashes: none,               // per-series stroke style: array of "solid" | "dashed" | "dotted"
  legend: none,               // none | "tr" | "tl" | "br" | "bl" — a legend box (instead of end labels)
  fill-under: none,           // idx | (idx, color) | (from:, to:, color:, series:) — shade under a curve
  points: (),                 // ((x, y, label), …) — read-off markers with droplines to the axes
  vlines: (),                 // ((x, label?, color?), …) — dashed vertical reference lines
  hlines: (),                 // ((y, label?, color?), …) — dashed horizontal reference lines
  annotations: (),            // ((x, y, label), …) — free-floating text
  y-ticks: true,              // draw numeric y-axis tick labels
  x-label: none, y-label: none, title: none,
  size: (62mm, 42mm),
  theme: default-theme,
  ..args,                     // the (optional) positional `series` lands here
) = {
  let t = theme
  let series = args.pos().at(0, default: none)
  let modes = (fn != none, y != none, series != none).filter(v => v).len()
  assert(modes == 1, message: "plot.lines: provide exactly one of fn, y, or positional series")
  assert(type(samples) == int and samples >= 1, message: "plot.lines: samples must be a positive integer")
  assert(not (log-y and fill-under != none), message: "plot.lines: fill-under is not defined for a log-y axis")
  // Build the list of series (each a list of (x,y)) from whichever input was given:
  //   fn + domain  →  sample a formula (the maintainable path: the curve IS the math)
  //   x / y arrays →  parallel columns (data)
  //   series       →  explicit (x,y) tuples
  let sers = if fn != none {
    assert(type(domain) == array and domain.len() == 2 and domain.at(1) > domain.at(0),
      message: "plot.lines: fn mode requires an increasing domain")
    let (a, b) = domain
    let fns = if type(fn) == function { (fn,) } else { fn }
    assert(type(fns) == array and fns.len() > 0 and fns.all(f => type(f) == function),
      message: "plot.lines: fn must be a function or non-empty array of functions")
    fns.map(f => range(samples + 1).map(i => {
      let xx = a + i * (b - a) / samples
      (xx, f(xx))
    }))
  } else if y != none {
    assert(type(y) == array and y.len() > 0, message: "plot.lines: y must be non-empty")
    let cols = if y.len() > 0 and type(y.first()) == array { y } else { (y,) }
    assert(cols.all(c => type(c) == array and c.len() > 0), message: "plot.lines: every y series must be non-empty")
    if x != none {
      assert(type(x) == array and cols.all(c => c.len() == x.len()),
        message: "plot.lines: x and y lengths must match")
    }
    cols.map(col => col.enumerate().map(((i, yv)) => (if x != none { x.at(i) } else { i + 1 }, yv)))
  } else {
    assert(type(series) == array and series.len() > 0, message: "plot.lines: positional series must be non-empty")
    let as-points(s) = if s.len() > 0 and type(s.first()) == array { s } else {
      s.enumerate().map(((i, yv)) => (i + 1, yv)) }
    let multi = series.len() > 0 and type(series.first()) == array and type(series.first().first()) == array
    if multi { series.map(as-points) } else { (as-points(series),) }
  }

  assert(sers.len() > 0 and sers.all(s => type(s) == array and s.len() > 0 and
    s.all(p => type(p) == array and p.len() >= 2 and _is-num(p.at(0)) and _is-num(p.at(1)))),
    message: "plot.lines: each series must contain numeric (x, y) points")
  if labels != none { assert(type(labels) == array and labels.len() == sers.len(),
    message: "plot.lines: label count must match series") }
  if fill-under != none {
    if type(fill-under) == dictionary {
      assert("from" in fill-under and "to" in fill-under and _is-num(fill-under.from) and
        _is-num(fill-under.to) and fill-under.from < fill-under.to,
        message: "plot.lines: fill-under interval must satisfy from < to")
      let fi = fill-under.at("series", default: 0)
      assert(type(fi) == int and fi >= 0 and fi < sers.len(),
        message: "plot.lines: fill-under series index out of bounds")
    } else if type(fill-under) == array {
      assert(fill-under.len() == 2 and type(fill-under.at(0)) == int,
        message: "plot.lines: fill-under pair must be (series, color)")
      let fi = fill-under.at(0)
      assert(fi >= 0 and fi < sers.len(), message: "plot.lines: fill-under series index out of bounds")
    } else {
      assert(type(fill-under) == int and fill-under >= 0 and fill-under < sers.len(),
        message: "plot.lines: fill-under series index out of bounds")
    }
  }
  let xs = sers.map(s => s.map(p => p.at(0))).flatten()
  let ys = sers.map(s => s.map(p => p.at(1))).flatten()
  if log-y { assert(ys.all(y => y > 0), message: "plot.lines: log-y values must be positive") }
  let (xmin, xmax) = (calc.min(..xs), calc.max(..xs))
  let ty(y) = if log-y { calc.log(y) } else { y }
  let yt = ys.map(ty)
  if fill-under != none { yt.push(0.0) }
  let (ymin, ymax) = (calc.min(..yt), calc.max(..yt))
  let (w, h) = size
  let px(x) = if xmax == xmin { 0pt } else { (x - xmin) / (xmax - xmin) * w }
  let py(y) = if ymax == ymin { 0pt } else { (ty(y) - ymin) / (ymax - ymin) * h }

  cetz.canvas({
    import cetz.draw: line, content, circle
    // axes
    line((0, 0), (w, 0), stroke: t.frame-stroke + t.ink)
    line((0, 0), (0, h), stroke: t.frame-stroke + t.ink)
    // faint horizontal gridlines at 0/50/100%
    for f in (0.25, 0.5, 0.75, 1.0) {
      line((0, f * h), (w, f * h), stroke: 0.4pt + t.muted.lighten(40%))
    }
    let sercol(si) = if colors != none and si < colors.len() { colors.at(si) }
      else { t.cycle.at(calc.rem(si, t.cycle.len())) }
    let dashof(si) = if dashes != none and si < dashes.len() and dashes.at(si) != "solid" { dashes.at(si) } else { none }
    // markers: a single bool for all series, or a per-series array of bools
    let markerof(si) = if type(markers) == array { si < markers.len() and markers.at(si) } else { markers }
    // shade the area under a curve. Forms:
    //   idx                              whole series `idx`
    //   (idx, color)                     whole series, custom colour
    //   (from: a, to: b, color:, series:) only the part with a ≤ x ≤ b (e.g. P(a≤Y≤b))
    if fill-under != none {
      let (fi, fc, s) = if type(fill-under) == dictionary {
        let fi = fill-under.at("series", default: 0)
        (fi, fill-under.at("color", default: sercol(fi).transparentize(70%)),
         sers.at(fi).filter(p => p.at(0) >= fill-under.from and p.at(0) <= fill-under.to))
      } else if type(fill-under) == array {
        (fill-under.at(0), fill-under.at(1), sers.at(fill-under.at(0)))
      } else {
        (fill-under, sercol(fill-under).transparentize(80%), sers.at(fill-under))
      }
      if s.len() >= 2 {
        let poly = s.map(p => (px(p.at(0)), py(p.at(1))))
        poly.push((px(s.last().at(0)), py(0.0))); poly.push((px(s.first().at(0)), py(0.0)))
        line(..poly, close: true, fill: fc, stroke: none)
      }
    }
    // series
    for (si, s) in sers.enumerate() {
      let col = sercol(si)
      let pts = s.map(p => (px(p.at(0)), py(p.at(1))))
      let st = if dashof(si) != none { (paint: col, thickness: 1.4pt, dash: dashof(si)) } else { 1.4pt + col }
      for k in range(pts.len() - 1) { line(pts.at(k), pts.at(k + 1), stroke: st) }
      if markerof(si) { for p in pts { circle(p, radius: 1.6pt, fill: col, stroke: none) } }
      // end-of-line label (unless a legend box is drawn instead)
      if labels != none and legend == none and si < labels.len() {
        content((pts.last().at(0) + 0.4em, pts.last().at(1)),
          text(size: 9pt, fill: col, labels.at(si)), anchor: "west")
      }
    }
    // annotated read-off points: dot + dashed droplines to both axes + label
    for pt in points {
      let (xv, yv, lbl) = (pt.at(0), pt.at(1), if pt.len() > 2 { pt.at(2) } else { none })
      let (cx, cy) = (px(xv), py(yv))
      let ds = (paint: t.muted, dash: "dashed", thickness: 0.6pt)
      line((cx, 0), (cx, cy), stroke: ds)
      line((0, cy), (cx, cy), stroke: ds)
      circle((cx, cy), radius: 2pt, fill: t.accent, stroke: none)
      if lbl != none { content((cx + 0.4em, cy + 0.4em), text(size: 8.5pt, fill: t.ink, lbl), anchor: "west") }
    }
    // dashed vertical / horizontal reference lines with optional labels
    for v in vlines {
      let (xv, lbl, vc) = (v.at(0), if v.len() > 1 { v.at(1) } else { none }, if v.len() > 2 { v.at(2) } else { t.muted })
      line((px(xv), 0), (px(xv), h), stroke: (paint: vc, dash: "dashed", thickness: 0.8pt))
      if lbl != none { content((px(xv), h + 0.3em), text(size: 8pt, fill: vc, lbl), anchor: "south") }
    }
    for hl in hlines {
      let (yv, lbl, hc) = (hl.at(0), if hl.len() > 1 { hl.at(1) } else { none }, if hl.len() > 2 { hl.at(2) } else { t.muted })
      line((0, py(yv)), (w, py(yv)), stroke: (paint: hc, dash: "dashed", thickness: 0.8pt))
      if lbl != none { content((w - 0.3em, py(yv) + 0.25em), text(size: 8pt, fill: hc, lbl), anchor: "south-east") }
    }
    // free-floating text annotations
    for an in annotations {
      content((px(an.at(0)), py(an.at(1))), text(size: 8.5pt, fill: t.ink, an.at(2)), anchor: "center")
    }
    // y-axis numeric ticks (linear only; log shows a 'log' tag)
    if y-ticks and not log-y {
      for f in (0.0, 0.5, 1.0) {
        content((-0.4em, f * h), text(size: 7.5pt, fill: t.muted, _fmt-num(ymin + f * (ymax - ymin))), anchor: "east")
      }
    }
    // legend box (swatch + label per series) instead of end labels
    if legend != none and labels != none {
      let rh = 1.15em
      let (lx, ly) = if legend == "tl" { (2mm, h - 1.5mm) }
        else if legend == "br" { (w - 20mm, labels.len() * rh) }
        else if legend == "bl" { (2mm, labels.len() * rh) }
        else { (w - 20mm, h - 1.5mm) }              // "tr" default
      for (i, lb) in labels.enumerate() {
        let yy = ly - i * rh
        line((lx, yy), (lx + 3.5mm, yy), stroke: 2.2pt + sercol(i))
        content((lx + 4.5mm, yy), text(size: 8pt, fill: t.ink, lb), anchor: "west")
      }
    }
    // axis labels
    if x-label != none { content((w/2, -1.6em), text(size: 9pt, fill: t.muted, x-label), anchor: "north") }
    if y-label != none { content((-2.5em, h/2), rotate(-90deg, text(size: 9pt, fill: t.muted, y-label)), anchor: "south") }
    if log-y { content((-0.6em, h + 0.4em), text(size: 8pt, fill: t.muted)[log], anchor: "south-east") }
    if title != none { content((w/2, h + 1.2em), text(size: 10pt, fill: t.ink, weight: 600, title), anchor: "south") }
    // x end-tick labels
    content((0, -0.4em), text(size: 8pt, fill: t.muted, _fmt-num(xmin)), anchor: "north")
    content((w, -0.4em), text(size: 8pt, fill: t.muted, _fmt-num(xmax)), anchor: "north")
  })
}
