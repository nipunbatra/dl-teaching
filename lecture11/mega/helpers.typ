// Diagrams and content helpers for L11M.
// Technical figures are native Typst/Fletcher so the exported PDFs stay vector.

#import "../../common/metropolis.typ": *
#import "../../common/mldiag.typ": *

#let PALE-TEAL = rgb("#DCEBEB")
#let PALE-BLUE = rgb("#E7F0FA")
#let PALE-ACC = rgb("#FDECD6")
#let PALE-GREEN = rgb("#E5F3E9")
#let PALE-RED = rgb("#FBE8E9")
#let CREAM = rgb("#F4F1EA")
#let PURPLE = rgb("#7C5C9E")
#let PALE-PURPLE = rgb("#EEE8F4")

// Theme, type, margins, title bar, and footer come from common/metropolis.typ.
// Keep the teaching tags in source without adding labels to the title bar.
#let Q = []
#let A = []
#let V = []
#let D = []
#let I = []
#let OPT = []

#let swatch(color, body) = box(
  width: 13pt,
  height: 4pt,
  fill: color,
  radius: 1pt,
) + h(4pt) + body

#let mega-legend = align(center, text(size: 10.5pt)[
  #swatch(TEAL, [context / key]) #h(8pt)
  #swatch(BLUE, [query / score / probability]) #h(8pt)
  #swatch(ACC, [target / value]) #h(8pt)
  #swatch(GREEN, [retained / residual]) #h(8pt)
  #swatch(RED, [loss / forbidden])
])

#let card(title, body, color: TEAL, fill: white, inset: (x: 10pt, y: 8pt)) = block(
  width: 100%,
  inset: (x: 5pt, y: 3pt),
  [
    #text(font: "IBM Plex Sans", size: 12pt, weight: 700, fill: color, tracking: .35pt)[#upper(title)]
    #v(-1pt)
    #line(length: 100%, stroke: 3.5pt + color.lighten(68%))
    #v(5pt)
    #body
  ],
)

#let note(body, color: TEAL) = block(
  width: 100%,
  inset: (left: 8pt, right: 4pt, top: 4pt, bottom: 4pt),
  stroke: (left: 2.1pt + color),
  body,
)

// A slide-sized notebook cell: never more than a handful of lines. The full
// runnable treatment belongs in the companion notebook, not on the slide.
#let torch-code(source, takeaway: none, color: BLUE) = block(
  width: 100%,
  inset: (left: 11pt, right: 9pt, top: 7pt, bottom: 7pt),
  fill: rgb("#F7F8F7"),
  stroke: (left: 3pt + color, top: .6pt + rgb("#D8DEDC"), bottom: .6pt + rgb("#D8DEDC")),
  radius: 2pt,
  [
    #text(font: "IBM Plex Mono", size: 9.5pt, weight: 700, fill: color)[PyTorch]
    #v(3pt)
    #raw(source, lang: "python", block: true)
    #if takeaway != none {
      v(4pt)
      line(length: 100%, stroke: .5pt + rgb("#D8DEDC"))
      v(3pt)
      text(font: "IBM Plex Sans", size: 11.5pt, fill: INK, takeaway)
    }
  ],
)

#let claim(body, color: ACC) = align(center, block(
  inset: (x: 10pt, y: 7pt),
  stroke: (bottom: 4.5pt + color.lighten(55%)),
  text(font: "IBM Plex Sans", size: 21pt, weight: 650, fill: INK, body),
))

#let hairline(title, body, color: TEAL) = block(
  width: 100%,
  inset: 3pt,
  [
    #text(font: "IBM Plex Sans", size: 12pt, weight: 700, fill: color, tracking: .35pt)[#upper(title)]
    #v(2pt)
    #line(length: 100%, stroke: 0.8pt + color)
    #v(5pt)
    #body
  ],
)

#let token(body, color: TEAL, fill: PALE-TEAL, w: 15mm, hgt: 10mm, size: 14pt) = box(
  width: w,
  height: hgt,
  fill: fill,
  stroke: 0.9pt + color,
  radius: 2.5pt,
  align(center + horizon, text(size: size, weight: 650, body)),
)

#let target-token(body, w: 15mm) = token(body, color: ACC, fill: PALE-ACC, w: w)
#let query-token(body, w: 15mm) = token(body, color: BLUE, fill: PALE-BLUE, w: w)
#let bad-token(body, w: 15mm) = token(body, color: RED, fill: PALE-RED, w: w)

#let token-row(items, arrow-to: none) = align(center, grid(
  columns: if arrow-to == none { items.map(_ => auto) } else { items.map(_ => auto) + (12mm, auto) },
  gutter: 5pt,
  align: horizon,
  ..items,
  ..if arrow-to == none { () } else { ([#text(size: 22pt, fill: MUTED)[$arrow.r$]], target-token(arrow-to)) },
))

#let stage-chip(number, title, active: false, color: TEAL) = block(
  width: 100%,
  inset: (x: 7pt, y: 5pt),
  [
    #text(font: "IBM Plex Sans", size: 10pt, weight: 750, fill: if active { color } else { MUTED })[#number]
    #h(5pt)
    #text(font: "IBM Plex Sans", size: 10pt, fill: if active { INK } else { MUTED })[#title]
    #v(1pt)
    #line(length: 100%, stroke: if active { 3pt + color.lighten(62%) } else { .6pt + MUTED.lighten(65%) })
  ],
)

#let clock-strip(active) = align(center, grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 8pt,
  stage-chip([TRAIN], [tokens $arrow.r$ logits $arrow.r$ loss $arrow.r$ update], active: active == "train", color: TEAL),
  stage-chip([GENERATE], [prefix $arrow.r$ distribution $arrow.r$ choose $arrow.r$ append], active: active == "generate", color: BLUE),
  stage-chip([EVALUATE], [held-out tokens $arrow.r$ mean NLL / PPL], active: active == "evaluate", color: INK),
))

#let dim-pill(body, color: INK) = box(
  inset: (x: 3pt, y: 1pt),
  text(font: "IBM Plex Mono", size: 9.5pt, weight: 650, fill: color, [(#body)]),
)

#let row-col-note() = note([
  Rows are queries; columns are source tokens and their keys.
], color: PURPLE)

#let two(a, b, ratio: (1fr, 1fr), gutter: 16pt) = grid(
  columns: ratio,
  gutter: gutter,
  align: horizon,
  a,
  b,
)

#let three(a, b, c, gutter: 10pt) = grid(
  columns: (1fr, 1fr, 1fr),
  gutter: gutter,
  align: horizon,
  a,
  b,
  c,
)

#let flow-node(pos, body, color: INK, fill: white, w: 27mm, size: 11pt) = node(
  pos,
  block(width: w, inset: (x: 5pt, y: 5pt), align(center, text(size: size, body))),
  shape: fletcher.shapes.rect,
  corner-radius: 3pt,
  stroke: 0.9pt + color,
  fill: fill,
)

#let flow-arrow(a, b, color: MUTED, label: none, bend: 0deg, thickness: 0.85pt) = edge(
  a,
  b,
  "-|>",
  bend: bend,
  stroke: thickness + color,
  label: if label == none { none } else { text(size: 9.5pt, fill: color, label) },
)

#let neuron-node(pos, body: [], color: BLUE, fill: PALE-BLUE, r: 4.2mm) = node(
  pos,
  box(width: 2 * r, height: 2 * r, align(center + horizon, text(size: 8.8pt, weight: 650, body))),
  shape: fletcher.shapes.circle,
  stroke: .9pt + color,
  fill: fill,
)

// A small, honest neural-network picture. It deliberately shows only a few
// representative output neurons; the label beneath the drawing states the
// real layer widths.
#let neural-net-sketch(
  input-labels: ([e₁], [e₂], [e₃]),
  output-labels: ([a], [b], [i], [$dots$], [z]),
  hidden: 4,
  highlight-output: 2,
) = align(center, diagram(
  spacing: (18mm, 10mm),
  {
    let ni = input-labels.len()
    let no = output-labels.len()
    for i in range(ni) {
      let yi = i - (ni - 1) / 2
      neuron-node((0, yi), body: input-labels.at(i), color: TEAL, fill: PALE-TEAL)
    }
    for j in range(hidden) {
      let yj = j - (hidden - 1) / 2
      neuron-node((2, yj), color: BLUE, fill: PALE-BLUE)
    }
    for k in range(no) {
      let yk = k - (no - 1) / 2
      let active = highlight-output != none and k == highlight-output
      neuron-node(
        (4, yk),
        body: output-labels.at(k),
        color: if active { ACC } else { MUTED },
        fill: if active { PALE-ACC } else { white },
      )
    }
    for i in range(ni) {
      let yi = i - (ni - 1) / 2
      for j in range(hidden) {
        let yj = j - (hidden - 1) / 2
        edge((0, yi), (2, yj), stroke: .35pt + MUTED.lighten(45%))
      }
    }
    for j in range(hidden) {
      let yj = j - (hidden - 1) / 2
      for k in range(no) {
        let yk = k - (no - 1) / 2
        edge((2, yj), (4, yk), stroke: .35pt + MUTED.lighten(45%))
      }
    }
  },
))

#let simple-flow(items) = align(center, diagram(
  spacing: (16mm, 9mm),
  node-stroke: 0.9pt + INK,
  node-fill: white,
  {
    for (i, item) in items.enumerate() {
      let body = item.at(0)
      let color = item.at(1)
      let fill = item.at(2)
      flow-node((i * 1.35, 0), body, color: color, fill: fill, w: 25mm)
      if i > 0 { flow-arrow(((i - 1) * 1.35, 0), (i * 1.35, 0)) }
    }
  },
))

#let matrix-table(
  row-labels,
  col-labels,
  values,
  cell-fill: none,
  cell-width: 17mm,
  cell-height: 9mm,
  text-size: 10.5pt,
  label-size: 10.5pt,
) = {
  let nr = row-labels.len()
  let nc = col-labels.len()
  assert(values.len() == nr and values.all(row => row.len() == nc))
  let cells = ()
  cells.push(block())
  for c in col-labels { cells.push(align(center, text(size: label-size, weight: 700, fill: TEAL, c))) }
  for i in range(nr) {
    cells.push(align(right + horizon, text(size: label-size, weight: 700, fill: BLUE, row-labels.at(i))))
    for j in range(nc) {
      let val = values.at(i).at(j)
      let bg = if cell-fill == none { white } else { cell-fill(i, j, val) }
      cells.push(block(
        width: cell-width,
        height: cell-height,
        fill: bg,
        stroke: 0.55pt + rgb("#D9DEDC"),
        align(center + horizon, text(size: text-size, val)),
      ))
    }
  }
  align(center, grid(
    columns: (20mm,) + (cell-width,) * nc,
    rows: (8mm,) + (cell-height,) * nr,
    gutter: 0pt,
    ..cells,
  ))
}

#let prob-bars(items, hgt: 34mm) = align(center, grid(
  columns: items.map(_ => 15mm),
  gutter: 6pt,
  align: bottom,
  ..items.map(item => {
    let label = item.at(0)
    let p = item.at(1)
    let color = if item.len() > 2 { item.at(2) } else { BLUE }
    stack(dir: ttb, spacing: 2pt,
      text(size: 9pt, weight: 650, fill: color)[#(calc.round(p * 1000) / 1000)],
      box(width: 13mm, height: hgt, align(bottom, block(
        width: 13mm,
        height: p * hgt,
        fill: color.lighten(18%),
        radius: (top: 2pt),
      ))),
      text(size: 10.5pt, weight: 650, label),
    )
  }),
))

#let section-map(active) = align(center, grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 8pt,
  stage-chip([LECTURE 1], [text $arrow.r$ tokens $arrow.r$ next-token prediction], active: active == 1, color: TEAL),
  stage-chip([LECTURE 2], [context $arrow.r$ weights $arrow.r$ attention], active: active == 2, color: BLUE),
  stage-chip([LECTURE 3], [MHA $+$ MLP $+$ residual $arrow.r$ Transformer], active: active == 3, color: ACC),
))

#let source-line(body) = align(right, text(font: "IBM Plex Sans", size: 8.5pt, fill: MUTED, body))
