// Seven small hand calculations for the same next-token prediction.
// Three progressive builds per slide. All numbers below are teaching examples.
#import "../../common/metropolis.typ": *
#import "../../common/mldiag.typ": *
#import "helpers.typ": *

#let bw-words = ([She], [deposited], [money], [in], [the], [bank])
#let bw-keys = ([$mat(0,0,1)$], [$mat(1,0,1)$], [$mat(2,0,0)$], [$mat(0,1,0)$], [$mat(0,0,0)$], [$mat(1,1,0)$])
#let bw-values = ([$mat(0,0,1)$], [$mat(1,0,1)$], [$mat(2,0,0)$], [$mat(0,2,0)$], [$mat(0,0,0)$], [$mat(1,2,0)$])
#let bw-scores = (0, 1, 2, 0, 0, 1)
#let bw-weights = ("0.063", "0.172", "0.467", "0.063", "0.063", "0.172")

#let bw-cell(body, color: TEAL, fill: PALE-TEAL, width: 100%, height: 10mm) = box(
  width: width, height: height, radius: 2pt,
  stroke: .8pt + color, fill: fill,
  align(center + horizon, body),
)

#let bw-query-line = align(center, text(size: 15pt)[
  Last known token: `bank`, position 6. #h(10pt)
  #text(fill: BLUE)[$q_6 = mat(1,0,0)$]
])

#let bw-number-caption = align(center, text(size: 11.5pt, fill: MUTED)[
  Made-up numbers for this calculation. Real learned coordinates need not have simple meanings.
])

#[
== Keep the query; now inspect all six sources

#set text(size: 17pt)
#align(center, text(size: 15pt)[One word per token. Position $i$ tells us where; token ID $t_i$ tells us which word.])
#v(7pt)
#grid(columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 16mm), gutter: 5pt,
  align: center + horizon,
  ..bw-words.enumerate().map(((i, word)) => [
    #text(size: 11pt, fill: MUTED)[position #(i + 1)]
    #v(3pt)
    #bw-cell(text(size: 15pt, weight: 650, word),
      color: if i == 5 { BLUE } else { TEAL },
      fill: if i == 5 { PALE-BLUE } else { PALE-TEAL })
  ]),
  [#text(size: 11pt, fill: ACC)[predict]\ #bw-cell([?], color: ACC, fill: PALE-ACC)],
)
#v(8pt)
#align(center, text(size: 15pt)[All six words are known. The missing next token is not an input.])

#v(8pt)
#uncover("2-")[
  #align(center, grid(columns: (47mm, 8mm, 43mm, 8mm, 43mm), gutter: 5pt, align: center + horizon,
    bw-cell([embedding $e_6$\ #text(size: 13pt)[$E[t_6]=mat(1,1,0)$]], color: TEAL, fill: PALE-TEAL, height: 19mm),
    text(fill: TEAL)[$arrow.r$],
    bw-cell([apply\ $W_Q$], color: BLUE, fill: PALE-BLUE, height: 19mm),
    text(fill: BLUE)[$arrow.r$],
    bw-cell([query $q_6$\ $mat(1,0,0)$], color: BLUE, fill: PALE-BLUE, height: 19mm),
  ))
  #v(3pt)
  #align(center, text(size: 12pt, fill: MUTED)[$q_6=e_6 W_Q$. We use lookup embeddings now; position information comes later.])
]
#v(8pt)
#uncover("3-")[
  #align(center, text(size: 15pt)[Keep the same query. Now include every known input position in the calculation.])
  #v(5pt)
  #grid(columns: (1fr,) * 6, gutter: 5pt, align: center + horizon,
    ..bw-words.enumerate().map(((i, word)) => bw-cell([
      #text(size: 12pt)[#word]\ $k_#(i + 1)$
    ], height: 16mm)),
  )
]

== Give each input a key and a value

#set text(size: 16pt)
#bw-query-line
#v(3pt)
#align(center, text(size: 12pt)[$k_j=e_j W_K$, $v_j=e_j W_V$. The query, keys, and values each have three coordinates here.])
#v(9pt)
#grid(columns: (16mm, 41mm, 1fr, 1fr), column-gutter: 14pt, row-gutter: 7pt,
  align: (x, y) => if x == 1 { left + horizon } else { center + horizon },
  text(size: 13pt, weight: 650)[position],
  text(weight: 650)[known word],
  text(fill: TEAL, weight: 650)[key: used to match],
  uncover("2-")[#text(fill: ACC, weight: 650)[value: information to mix]],
  ..bw-words.enumerate().fold((), (cells, pair) => {
    let (i, word) = pair
    cells + (
      text(size: 14pt, fill: MUTED)[#(i + 1)],
      text(weight: 650, word),
      text(fill: TEAL)[$k_#(i + 1) =$ #bw-keys.at(i)],
      uncover("2-")[#text(fill: ACC)[$v_#(i + 1) =$ #bw-values.at(i)]],
    )
  }),
)
#v(10pt)
#uncover("3-")[
  #note(color: BLUE)[Compare the query with a key to choose a weight. Use that weight to multiply the value from the same position.]
  #v(5pt)
  #bw-number-caption
]

== Compare the same query with all six keys

#set text(size: 16pt)
#bw-query-line
#v(8pt)
#grid(columns: (37mm, 33mm, 1fr, 17mm), column-gutter: 10pt, row-gutter: 11pt,
  align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
  text(weight: 650)[input position],
  text(fill: TEAL, weight: 650)[key $k_j$],
  uncover("2-")[#text(fill: BLUE)[multiply the coordinates, then add]],
  uncover("2-")[#text(fill: BLUE)[score $s_j$]],
  [1 · She], bw-keys.at(0),
  uncover("2-")[$1 times 0 + 0 times 0 + 0 times 1$], uncover("2-")[#text(fill: BLUE)[$0$]],
  [2 · deposited], bw-keys.at(1),
  uncover("2-")[$1 times 1 + 0 times 0 + 0 times 1$], uncover("2-")[#text(fill: BLUE)[$1$]],
  [3 · money], bw-keys.at(2),
  uncover("2-")[$1 times 2 + 0 times 0 + 0 times 0$], uncover("2-")[#text(fill: BLUE)[$2$]],
  [4 · in], bw-keys.at(3),
  uncover("3-")[$1 times 0 + 0 times 1 + 0 times 0$], uncover("3-")[#text(fill: BLUE)[$0$]],
  [5 · the], bw-keys.at(4),
  uncover("3-")[$1 times 0 + 0 times 0 + 0 times 0$], uncover("3-")[#text(fill: BLUE)[$0$]],
  [6 · bank], bw-keys.at(5),
  uncover("3-")[$1 times 1 + 0 times 1 + 0 times 0$], uncover("3-")[#text(fill: BLUE)[$1$]],
)
#v(10pt)
#uncover("3-")[
  #note(color: BLUE)[
    At `bank`: #text(fill: BLUE)[$q_6 = mat(1,0,0)$], but #text(fill: TEAL)[$k_6 = mat(1,1,0)$]. Different vectors, different jobs.\
    Here, `money` matches this query better than `bank`'s own key.
  ]
  #v(5pt)
  #align(center, text(size: 11.5pt, fill: MUTED)[Unscaled dot products for now. We will add the usual scaling factor later.])
]

== Exponentiate the six match scores

#set text(size: 16pt)
#align(center)[The query at #text(fill: BLUE)[`bank`] gave scores $s = mat(0,1,2,0,0,1)$.]
#v(8pt)
#align(center, grid(columns: (49mm, 24mm, 35mm, 32mm),
  column-gutter: 16pt, row-gutter: 12pt,
  align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
  text(weight: 650)[input position], text(fill: BLUE)[score $s_j$],
  uncover("2-")[#text(fill: BLUE)[exponentiate]],
  uncover("2-")[#text(fill: BLUE)[positive amount]],
  ..bw-words.enumerate().fold((), (cells, pair) => {
    let (i, word) = pair
    let amount = ("1.000", "2.718", "7.389", "1.000", "1.000", "2.718").at(i)
    cells + (
      [#(i + 1) · #word], [#bw-scores.at(i)],
      uncover("2-")[$exp(#str(bw-scores.at(i)))$],
      uncover("2-")[#text(features: ("tnum",), amount)],
    )
  }),
))
#v(11pt)
#uncover("3-")[
  #align(center, text(fill: BLUE)[$D approx 1.000 + 2.718 + 7.389 + 1.000 + 1.000 + 2.718 = 15.825$])
  #v(8pt)
  #note(color: BLUE)[Larger scores give larger positive amounts. These are not weights yet: their total is 15.825, not 1.]
  #v(5pt)
  #align(center, text(size: 11.5pt, fill: MUTED)[For this hand calculation, we round the exponentials first, then add the displayed values.])
]

== Divide every amount by the same total

#set text(size: 16pt)
#align(center)[Same six sources. Same total: #text(fill: BLUE)[$D approx 15.825$].]
#v(3pt)
#align(center, text(size: 12pt, fill: MUTED)[In $alpha_(6j)$, 6 is the receiver and $j$ the source. Here $alpha_j$ abbreviates $alpha_(6j)$.])
#v(6pt)
#align(center, grid(columns: (47mm, 27mm, 10mm, 27mm, 10mm, 31mm),
  column-gutter: 11pt, row-gutter: 9pt,
  align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
  text(weight: 650)[input position], text(fill: BLUE)[amount], [],
  text(fill: BLUE)[total], [], uncover("2-")[#text(fill: BLUE, weight: 650)[weight $alpha_j$]],
  ..bw-words.enumerate().fold((), (cells, pair) => {
    let (i, word) = pair
    let amount = ("1.000", "2.718", "7.389", "1.000", "1.000", "2.718").at(i)
    cells + (
      [#(i + 1) · #word], text(features: ("tnum",), amount),
      [$÷$], text(features: ("tnum",))[15.825], [$approx$],
      uncover("2-")[#text(fill: BLUE, weight: 650, features: ("tnum",), bw-weights.at(i))],
    )
  }),
))
#v(6pt)
#uncover("3-")[
  #align(center)[$alpha_j = exp(s_j) / (sum_(m=1)^6 exp(s_m))$]
  #v(4pt)
  #note(color: BLUE)[Exponentiate, then divide by the total: this is *softmax*.\ Six source positions give six weights; the displayed weights sum to 1.]
  #v(4pt)
  #align(center, text(size: 11.5pt, fill: MUTED)[We will reuse these three-decimal weights when mixing the values.])
]

== Use each weight to mix the corresponding value

#set text(size: 16pt)
#bw-query-line
#v(8pt)
#touying-fn-wrapper((self: none) => align(center, diagram(
  spacing: (6mm, 9mm),
  {
    for i in range(6) {
      node((i, 0), block(width: 31mm, inset: (x: 4pt, y: 5pt), align(center)[
        #text(size: 12pt, weight: 650)[#bw-words.at(i)]\
        #text(fill: ACC)[$v_#(i + 1)$]\
        #bw-values.at(i)
        #v(3pt)
        #if self.subslide >= 2 {
          text(size: 12pt, fill: BLUE, weight: 650)[weight #bw-weights.at(i)]
        } else {
          hide(text(size: 12pt, fill: BLUE, weight: 650)[weight #bw-weights.at(i)])
        }
      ]), name: "bw-value-" + str(i), shape: fletcher.shapes.rect,
        corner-radius: 3pt, stroke: .8pt + ACC, fill: PALE-ACC)
      edge((name: "bw-value-" + str(i), anchor: "south"), (name: "bw-mixture", anchor: "north"), "-|>",
        snap-to: (none, none),
        stroke: (.6pt + 2.3pt * float(bw-weights.at(i))) + if self.subslide >= 2 { BLUE } else { BLUE.transparentize(100%) })
    }
    node((2.5, 1.45), block(width: 40mm, inset: (x: 7pt, y: 6pt), align(center)[
      #if self.subslide >= 2 { [$h_6$\ #text(size: 12pt)[for query at `bank`]] } else { hide[$h_6$\ #text(size: 12pt)[for query at `bank`]] }
    ]), name: "bw-mixture", shape: fletcher.shapes.rect, corner-radius: 3pt,
      stroke: .9pt + if self.subslide >= 2 { GREEN } else { GREEN.transparentize(100%) },
      fill: if self.subslide >= 2 { PALE-GREEN } else { PALE-GREEN.transparentize(100%) })
  },
)), last-subslide: 3)
#v(8pt)
#uncover("3-")[
  #align(center)[$h_6 = 0.063v_1 + 0.172v_2 + 0.467v_3 + 0.063v_4 + 0.063v_5 + 0.172v_6$]
  #v(6pt)
  #note(color: ACC)[Multiply every coordinate of a value by its weight. Then add the six vectors.]
]

== Calculate the three coordinates of the message

#set text(size: 15pt)
#align(center)[$h_(6,c)$ is coordinate $c$ of the message for receiver 6. Use the same weights for each coordinate.]
#v(12pt)
#text(fill: ACC, weight: 650)[First coordinate]
#v(3pt)
#align(center)[$h_(6,1) = 0.063 times 0 + 0.172 times 1 + 0.467 times 2 + 0.063 times 0 + 0.063 times 0 + 0.172 times 1 = 1.278$]
#v(12pt)
#uncover("2-")[
  #text(fill: ACC, weight: 650)[Second coordinate]
  #v(3pt)
  #align(center)[$h_(6,2) = 0.063 times 0 + 0.172 times 0 + 0.467 times 0 + 0.063 times 2 + 0.063 times 0 + 0.172 times 2 = 0.470$]
]
#v(12pt)
#uncover("3-")[
  #text(fill: ACC, weight: 650)[Third coordinate]
  #v(3pt)
  #align(center)[$h_(6,3) = 0.063 times 1 + 0.172 times 1 + 0.467 times 0 + 0.063 times 0 + 0.063 times 0 + 0.172 times 0 = 0.235$]
  #v(12pt)
  #align(center, text(fill: GREEN)[$h_6 approx mat(1.278,0.470,0.235)$])
  #v(6pt)
  #align(center, text(size: 12pt)[Six weights produced a three-coordinate message for this occurrence of `bank`.])
]

]
