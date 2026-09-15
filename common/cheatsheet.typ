// Shared helpers for readable, full-page A4 lecture cheatsheets.
// Keep this file dependency-free so every sheet can compile with plain Typst.

#let palette = (
  ink: rgb("#183138"),
  muted: rgb("#667b80"),
  line: rgb("#c9d3d3"),
  paper: rgb("#f7f5ef"),
  teal: rgb("#237f82"),
  blue: rgb("#2767b0"),
  orange: rgb("#b95400"),
  red: rgb("#bc343d"),
  green: rgb("#187c38"),
)

#let soft(color, amount: 88%) = color.lighten(amount)

#let sheet-title(kicker, title, subtitle: none) = [
  #block(width: 100%, inset: (bottom: 5pt))[
    #text(size: 8.5pt, weight: "bold", tracking: 0.8pt, fill: palette.orange)[#kicker]
    #linebreak()
    #text(size: 19pt, weight: "bold", fill: palette.ink)[#title]
    #if subtitle != none {
      linebreak()
      text(size: 9.5pt, fill: palette.muted, subtitle)
    }
  ]
  #line(length: 100%, stroke: 1.2pt + palette.orange)
  #v(5pt)
]

#let section-title(title, accent: palette.teal) = [
  #block(width: 100%, inset: (top: 1pt, bottom: 2.5pt))[
    #text(size: 11.3pt, weight: "bold", fill: accent)[#title]
  ]
]

#let card(title: none, accent: palette.teal, fill: white, inset: 7pt, body) = block(
  width: 100%,
  breakable: false,
  fill: fill,
  stroke: (left: 2.2pt + accent, top: 0.45pt + palette.line, right: 0.45pt + palette.line, bottom: 0.45pt + palette.line),
  radius: 3pt,
  inset: inset,
)[
  #if title != none {
    text(size: 10.3pt, weight: "bold", fill: accent, title)
    v(2.8pt)
  }
  #body
]

#let banner(body, accent: palette.orange) = block(
  width: 100%,
  fill: soft(accent, amount: 91%),
  stroke: 0.7pt + accent,
  radius: 4pt,
  inset: (x: 7pt, y: 5pt),
)[
  #align(center, text(size: 10.5pt, weight: "bold", fill: palette.ink, body))
]

#let label(body, color: palette.teal) = box(
  fill: color,
  radius: 10pt,
  inset: (x: 5pt, y: 1.3pt),
)[#text(size: 8pt, weight: "bold", fill: white, body)]

#let keyline(label, body, color: palette.teal) = [
  #text(weight: "bold", fill: color)[#label] #body
]

#let tiny-note(body) = text(size: 8.5pt, fill: palette.muted, body)

#let check(body, color: palette.green) = [
  #text(weight: "bold", fill: color)[✓] #body
]
