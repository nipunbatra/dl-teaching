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

// ── discipline tags [Q][V][D][I][opt] — seated in the dark header bar, top-right ──
// A filled chip pops cleanly on the dark ink header; the word rides beside the
// letter so the tag reads without the legend. `stag()` lifts it up into the bar,
// vertically centred with the title, so it never crowds the first body line.
#let _chip(lbl, word, col) = box(fill: col, inset: (x: 7pt, y: 3.5pt), radius: 4pt, baseline: 0.28em,
  text(font: "IBM Plex Mono", size: 12pt, fill: white, weight: 700, tracking: 0.4pt,
    [#lbl#h(5pt)#text(size: 8.5pt, weight: 500, tracking: 1pt)[#upper(word)]]))
// The leading h(1fr) right-pins the chip: in an inline heading it flushes to the
// header's right edge; inside stag()'s place() the 1fr collapses to 0 (harmless).
#let Q = [#h(1fr)#_chip("Q", "question",    BLUE)]
#let A = [#h(1fr)#_chip("A", "answer",      GREEN)]
#let V = [#h(1fr)#_chip("V", "visual",      TEAL)]
#let D = [#h(1fr)#_chip("D", "derivation",  ACC)]
#let I = [#h(1fr)#_chip("I", "interactive", GREEN)]
#let OPT = [#h(1fr)#_chip(sym.star.filled, "optional", MUTED)]
// Legacy separate-line tag (L1): lift the chip up into the header bar, right-pinned.
#let stag(t) = place(top + right, dx: 0.2em, dy: -2.62em, t)

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
#let codebox(body, size: 15pt) = block(fill: rgb("#F3F2EE"), inset: 10pt, radius: 4pt,
  width: 100%, stroke: 0.5pt + rgb("#DAD8D2"), text(size: size, body))

// ── retrieval checkpoints ──────────────────────────────────────────
// Keep a question and its feedback on adjacent slides: students first commit to
// an option, then see the correct choice and the one-sentence reason.
#let mcq(prompt, a, b, c, d) = [
  #text(size: 22pt, weight: 600, fill: INK)[#prompt]
  #v(14pt)
  #grid(columns: 2, gutter: 12pt,
    block(fill: rgb("#EEF4F8"), inset: 10pt, radius: 4pt, stroke: 0.7pt + BLUE)[*A.* #a],
    block(fill: rgb("#EEF4F8"), inset: 10pt, radius: 4pt, stroke: 0.7pt + BLUE)[*B.* #b],
    block(fill: rgb("#EEF4F8"), inset: 10pt, radius: 4pt, stroke: 0.7pt + BLUE)[*C.* #c],
    block(fill: rgb("#EEF4F8"), inset: 10pt, radius: 4pt, stroke: 0.7pt + BLUE)[*D.* #d],
  )
]
#let mcq-answer(letter, choice, why) = [
  #result[Correct: *#letter* — #choice]
  #v(14pt)
  #notebox[*Why:* #why]
]

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
  body,
) = {
  show: metropolis-theme.with(
    aspect-ratio: "16-9",
    align: top,                          // content starts at the SAME vertical position on every slide
    config-common(handout: HANDOUT),
    // taller title bar (bigger top band) + a little side breathing room
    config-page(margin: (top: 3.7em, bottom: 1.35em, left: 2.4em, right: 2.4em)),
    config-info(title: title, subtitle: subtitle, author: author, institution: institution),
    footer: none,                                          // no course/author byline
    footer-right: context utils.slide-counter.display(),   // just the slide number, no "/ total"
  )
  set text(font: "IBM Plex Sans", size: 20pt)
  show raw: set text(font: "IBM Plex Mono")
  // grow the title-bar text a touch so the header reads taller and cleaner
  show heading.where(level: 2): set text(size: 1.06em)
  show heading: set text(font: "IBM Plex Sans")
  set heading(numbering: none)
  show math.equation: set text(size: 19pt)
  // Keep display equations on a stable left edge. Inline mathematics remains
  // inline; only a standalone equation is wrapped by this rule.
  show math.equation.where(block: true): it => align(left, it)
  body
}
