// ═══════════════════════════════════════════════════════════════════
//  Shared metropolis deck theme for ES 667 Deep Learning (Typst/touying)
//  Import from a lecture:  #import "../common/metropolis.typ": *
//  Then:                   #show: metropolis-deck.with(title: [...], subtitle: [...])
//                          #title-slide()
//  Handout (one page per slide): typst compile --input handout=true <deck>.typ
// ═══════════════════════════════════════════════════════════════════

#import "@preview/touying:0.6.1": *
#import themes.metropolis: *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

// ── palette (matches metropolis defaults: dark-teal + orange) ──
#let INK   = rgb("#23373B")
#let ACC   = rgb("#EB811B")
#let TEAL  = rgb("#2C7A7B")
#let GREEN = rgb("#14B03D")
#let BLUE  = rgb("#2B6CB0")
#let MUTED = rgb("#6E7F82")
#let RED   = rgb("#D64550")

// ── legend pills [Q] [V] [D] [I] [optional], placed top-right of the body ──
#let pill(lbl, col) = box(fill: col, inset: (x: 6pt, y: 3pt), radius: 3pt,
  text(font: "IBM Plex Mono", size: 12pt, fill: white, weight: 600, lbl))
#let Q = pill("Q", BLUE)      // question
#let V = pill("V", TEAL)      // visual
#let D = pill("D", ACC)       // derivation
#let I = pill("I", GREEN)     // interactive
#let OPT = pill("optional", MUTED)
#let stag(t) = place(top + right, dx: 0pt, dy: -2pt, t)   // fixed top-right slot

// ── callout blocks ──
#let bar-block(body, col, bg, head, headcol) = block(width: 100%, fill: bg, inset: 11pt,
  radius: 3pt, stroke: (top: 3pt + col), {
    if head != none { text(font: "IBM Plex Mono", size: 11pt, fill: headcol, tracking: 0.5pt, upper(head)); v(4pt) }
    body
  })
#let notebox(body)  = bar-block(body, INK, rgb("#EFEEEB"), none, MUTED)
#let alertbox(body) = bar-block(body, RED, rgb("#FBEBEC"), none, RED)
#let interbox(body, link-to: none) = bar-block({
    if link-to != none { text(font: "IBM Plex Mono", size: 10.5pt, fill: rgb("#1B7A34"))[↗ #link(link-to)[#link-to.replace("https://", "")]]; v(4pt) }
    body
  }, GREEN, rgb("#EAF6EC"), "Interactive", rgb("#1B7A34"))
#let result(body) = align(center, block(fill: white, inset: (x: 15pt, y: 11pt), radius: 5pt,
  stroke: 2pt + ACC, text(size: 21pt, weight: 600, fill: INK, body)))

// ── figure + layout helpers (Typst reads PNG twins; resvg mangles some mpl SVGs) ──
#let fig(path, w: 58%) = align(center, image(path.replace(".svg", ".png"), width: w))
#let two(a, b, r: (1fr, 1fr)) = grid(columns: r, gutter: 20pt, align(horizon, a), align(horizon, b))

// ── native diagrams (fletcher) — vector, no resvg issues ─────────────
// A single neuron:  inputs → (Σ+b) → φ → a
#let neuron-diagram(d: 3) = align(center, diagram(
  spacing: (11mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    let xy(i) = (0, i - (d - 1)/2)
    for i in range(d) {
      let lbl = if i == d - 1 { $x_d$ } else { [$x_#(i+1)$] }
      node(xy(i), lbl, radius: 7mm)
    }
    node((2, 0), [$Sigma + b$], radius: 9mm, fill: rgb("#EFEEEB"))
    node((3.4, 0), $phi$, radius: 7mm, fill: rgb("#FDECD6"), stroke: 0.9pt + ACC)
    node((4.7, 0), $a$, radius: 6mm)
    for i in range(d) { edge(xy(i), (2, 0), "-|>", stroke: 0.7pt + MUTED) }
    edge((2, 0), (3.4, 0), "-|>", stroke: 0.9pt + INK)
    edge((3.4, 0), (4.7, 0), "-|>", stroke: 0.9pt + INK)
  }))

// A fully-connected MLP for layer sizes like (3, 4, 4, 2).
#let mlp-diagram(sizes, hl: 1) = align(center, diagram(
  spacing: (20mm, 6.5mm),
  {
    let coord(li, i, n) = (li, i - (n - 1)/2)
    // edges first (behind nodes)
    for li in range(sizes.len() - 1) {
      let (n0, n1) = (sizes.at(li), sizes.at(li + 1))
      for i in range(n0) { for j in range(n1) {
        edge(coord(li, i, n0), coord(li + 1, j, n1), stroke: 0.35pt + MUTED.lighten(20%))
      } }
    }
    // nodes
    for (li, n) in sizes.enumerate() {
      let col = if li == 0 { INK } else if li == sizes.len() - 1 { ACC } else { TEAL }
      for i in range(n) { node(coord(li, i, n), none, radius: 3.2mm, fill: col, stroke: none) }
    }
  }))

// ── handout flag + the deck wrapper (theme, fonts, margins, footer) ──
#let HANDOUT = sys.inputs.at("handout", default: "false") == "true"

#let metropolis-deck(
  title: [], subtitle: [],
  author: [Prof. Nipun Batra],
  institution: [ES 667 · Deep Learning · IIT Gandhinagar],
  footer-text: [ES 667 · Deep Learning · N. Batra],
  body,
) = {
  show: metropolis-theme.with(
    aspect-ratio: "16-9",
    config-common(handout: HANDOUT),
    config-page(margin: (top: 1.7em, bottom: 1.4em, left: 2.2em, right: 2.2em)),
    config-info(title: title, subtitle: subtitle, author: author, institution: institution),
    // pad the footer so neither the byline nor the page-number kisses the edge
    footer: self => [#h(0.8em)#footer-text],
    footer-right: context [#utils.slide-counter.display() / #utils.last-slide-number#h(0.8em)],
  )
  set text(font: "IBM Plex Sans", size: 20pt)
  show raw: set text(font: "IBM Plex Mono")
  show heading: set text(font: "IBM Plex Sans")
  set heading(numbering: none)
  show math.equation: set text(size: 19pt)
  body
}
