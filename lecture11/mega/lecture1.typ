// L11M, Lecture 1: 51 conceptual frames.
// This file is included by ../L11M-from-characters-to-transformers.typ.
#import "../../common/metropolis.typ": *
#import "../../common/mldiag.typ": *
#import "helpers.typ": *

#let nipun-lm = "https://nipunbatra.github.io/ml-teaching/neural-networks/slides/next-token-prediction.pdf"
#let mit-l2 = "https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L2.pdf"
#let stanford-l1 = "https://www.youtube.com/watch?v=Ub3GoFaUcds"
#let bengio-lm = "https://www.jmlr.org/papers/v3/bengio03a.html"

// Lecture-local helpers. Prefixing them keeps this part independent of the
// other two lecture files, which are authored in parallel.
// The source notes are sparse handwritten builds. These two wrappers avoid
// turning every idea into a dashboard tile: only the short label receives a
// highlighter swipe; the explanation sits unboxed on the page.
#let l1-card(title, body, color: TEAL, fill: white) = block(
  width: 100%,
  inset: (x: 4pt, y: 3pt),
  [
    #box(
      fill: color.lighten(86%),
      inset: (x: 5pt, y: 2pt),
      text(size: 10.5pt, weight: 700, fill: color, tracking: 0.45pt, upper(title)),
    )
    #v(4pt)
    #text(size: 14pt, body)
  ],
)

#let l1-note(body, color: TEAL) = block(
  width: 100%,
  inset: (x: 4pt, y: 3pt),
  [
    #line(length: 24mm, stroke: 2.2pt + color.lighten(28%))
    #v(3pt)
    #text(size: 13.5pt, body)
  ],
)

#let l1-label(body, color: MUTED) = text(
  size: 10.5pt,
  weight: 700,
  tracking: 0.55pt,
  fill: color,
  upper(body),
)

#let l1-pair(context-items, target) = token-row(
  context-items.map(x => token(x)),
  arrow-to: target,
)

#let l1-aligned(words, tags) = {
  assert(words.len() == tags.len())
  align(center, grid(
    columns: words.map(_ => auto),
    gutter: 5pt,
    row-gutter: 5pt,
    align: center,
    ..words.map(w => token(w, w: 24mm, size: 11.5pt)),
    ..tags.map(tag => box(
      width: 24mm,
      inset: (x: 2pt, y: 3pt),
      fill: if tag == [O] { CREAM } else { PALE-ACC },
      stroke: 0.6pt + if tag == [O] { MUTED } else { ACC },
      radius: 2pt,
      align(center, text(size: 9pt, weight: 650, tag)),
    )),
  ))
}

#let l1-vocab-strip() = align(center, grid(
  columns: 9,
  gutter: 3pt,
  ..(([-], [a], [b], [c], [dots], [x], [y], [z], [27 classes])).map(it => {
    if it == [27 classes] {
      dim-pill(it, color: BLUE)
    } else if it == [dots] {
      align(center + horizon, text(size: 15pt, fill: MUTED)[$dots$])
    } else {
      token(it, w: 11mm, hgt: 9mm, size: 12pt)
    }
  }),
))

#let l1-window(c1, c2, c3, target, stage) = grid(
  columns: (24mm, 1fr),
  gutter: 8pt,
  align: horizon,
  l1-label(stage, color: TEAL),
  l1-pair((c1, c2, c3), target),
)

// A stable held sketch across the worked `aab → i` derivation. Each frame
// moves one highlighter mark while the geometry itself stays put.
#let l1-held-stage(label, active, color) = box(
  inset: (x: 4pt, y: 2pt),
  fill: if active { color.lighten(84%) } else { none },
  text(size: 9.5pt, weight: if active { 700 } else { 500 }, fill: if active { color } else { MUTED }, label),
)

#let l1-held(active) = align(center, grid(
  columns: (auto, 7mm, auto, 7mm, auto, 7mm, auto, 7mm, auto, 7mm, auto, 7mm, auto),
  gutter: 2pt,
  align: horizon,
  l1-held-stage([`a a b`], active == "tokens", TEAL),
  text(size: 12pt, fill: MUTED)[$arrow.r$],
  l1-held-stage([$E[dot]$], active == "embed", TEAL),
  text(size: 12pt, fill: MUTED)[$arrow.r$],
  l1-held-stage([$[e_a;e_a;e_b]$], active == "join", INK),
  text(size: 12pt, fill: MUTED)[$arrow.r$],
  l1-held-stage([MLP], active == "mlp", BLUE),
  text(size: 12pt, fill: MUTED)[$arrow.r$],
  l1-held-stage([$27$ logits], active == "logits", BLUE),
  text(size: 12pt, fill: MUTED)[$arrow.r$],
  l1-held-stage([softmax], active == "softmax", BLUE),
  text(size: 12pt, fill: MUTED)[$arrow.r$],
  l1-held-stage([$"CE"(i)$], active == "loss", RED),
))

#let l1-continuum-node(title, detail, color) = block(
  width: 45mm,
  inset: (x: 7pt, y: 6pt),
  align(center, [
    #box(fill: color.lighten(86%), inset: (x: 6pt, y: 2pt), [
      #text(size: 12pt, weight: 700, fill: color)[#title]
    ])
    #linebreak()
    #text(size: 10.5pt, fill: INK)[#detail]
  ]),
)

= Lecture 1 · Next-token prediction

// ───────────────────────── ACT 1 · WHY LANGUAGE? ─────────────────────────

== Deep learning for language #V

#align(center, text(size: 27pt, weight: 650, fill: INK)[
  How can a neural network work with *language*?
])
#pause
#v(7pt)
#two(
  [
    #grid(
      columns: (31mm, 1fr),
      rows: (auto,) * 6,
      gutter: 5pt,
      row-gutter: 7pt,
      align: horizon,
      l1-label([CLASSIFY], color: TEAL), [email $arrow.r$ spam?],
      l1-label([UNDERSTAND], color: BLUE), [review $arrow.r$ sentiment],
      l1-label([COMPRESS], color: ACC), [article $arrow.r$ summary],
      l1-label([TRANSLATE], color: PURPLE), [English $arrow.r$ Hindi],
      l1-label([ANSWER], color: GREEN), [context + question $arrow.r$ answer],
      l1-label([GENERATE], color: BLUE), [prompt $arrow.r$ continuation],
    )
  ],
  [#align(center, image("../figures/language_tasks_illustration_imagen.png", height: 77mm, fit: "contain"))],
  ratio: (1fr, 1fr),
  gutter: 10pt,
)
#source-line([Examples: #link(mit-l2)[MIT 6.S191 L2] and #link(stanford-l1)[Stanford CME 295 L1].])

== One input → one label #Q

#l1-card([REVIEW], [“The acting was wonderful, but the ending was disappointing.”], color: TEAL)
#v(11pt)
#align(center, grid(
  columns: (42mm, 15mm, 66mm),
  gutter: 10pt,
  align: horizon,
  l1-card([TEXT INPUT], [$x$], color: TEAL, fill: PALE-TEAL),
  text(size: 24pt, fill: MUTED)[$arrow.r$],
  prob-bars((([positive], .61, BLUE), ([negative], .39, ACC)), hgt: 28mm),
))
#pause
#v(8pt)
#claim[We have seen classification before. What changes when the input is text?]

== Intent classification #V

#grid(
  columns: (1.35fr, 12mm, 1fr),
  gutter: 8pt,
  row-gutter: 8pt,
  align: horizon,
  l1-card([UTTERANCE], [Book a flight to Delhi], color: TEAL),
  text(fill: MUTED)[$arrow.r$],
  l1-card([LABEL], [`BOOK_FLIGHT`], color: ACC, fill: PALE-ACC),
  l1-card([UTTERANCE], [What is my account balance?], color: TEAL),
  text(fill: MUTED)[$arrow.r$],
  l1-card([LABEL], [`CHECK_BALANCE`], color: ACC, fill: PALE-ACC),
  l1-card([UTTERANCE], [Cancel my booking], color: TEAL),
  text(fill: MUTED)[$arrow.r$],
  l1-card([LABEL], [`CANCEL_BOOKING`], color: ACC, fill: PALE-ACC),
)
#pause
#v(9pt)
#claim[$x arrow.r y$: a text input of any length, one class label]

== Token-level prediction: named entities #V

#l1-aligned(
  ([Sundar], [Pichai], [visited], [IIT], [Gandhinagar], [on], [Monday.]),
  ([PERSON], [PERSON], [O], [ORG], [ORG], [O], [DATE]),
)
#pause
#v(12pt)
#two(
  l1-card([INPUT], [a sequence of $T$ tokens], color: TEAL),
  l1-card([OUTPUT], [one class label for each of the $T$ tokens], color: ACC),
)
#v(7pt)
#claim[One token in, one label out.]

== Translation changes the sequence length #Q

#align(center, scale(x: 80%, y: 80%, reflow: true, diagram(
  spacing: (19mm, 9mm),
  node-stroke: 0.9pt + INK,
  node-fill: white,
  {
    flow-node((0, 0), [The lecture starts now.], color: TEAL, fill: PALE-TEAL, w: 47mm)
    flow-node((1.7, 0), [neural model], color: INK, fill: CREAM, w: 31mm)
    flow-node((3.6, 0), [व्याख्यान अब शुरू होता है।], color: ACC, fill: PALE-ACC, w: 55mm)
    flow-arrow((0, 0), (1.7, 0))
    flow-arrow((1.7, 0), (3.6, 0))
  },
)))
#pause
#v(11pt)
#two(
  l1-card([ENGLISH], [4 displayed words], color: TEAL),
  l1-card([HINDI], [5 displayed words], color: ACC),
)
#pause
#l1-note([One input word does not always correspond to one output word.], color: PURPLE)

== Summarize a notice #V

#two(
  l1-card([IITGN NOTICE], [
    The library will remain open until midnight during examinations. #linebreak()
    Extended hours begin Monday. #linebreak()
    Entry requires a valid institute ID. #linebreak()
    The ground-floor reading room stays open. #linebreak()
    Group rooms close at 10 pm. #linebreak()
    Regular timings resume after examinations.
  ], color: TEAL),
  [
    #align(center, text(size: 26pt, fill: MUTED)[$arrow.r$])
    #pause
    #l1-card([ONE-SENTENCE SUMMARY], [During exams, IITGN’s library and ground-floor reading room stay open until midnight from Monday; bring institute ID.], color: ACC, fill: PALE-ACC)
  ],
  ratio: (1.35fr, 1fr),
)
#pause
#v(8pt)
#claim[How does the model decide what matters?]

== Answer a question using the context #V

#l1-card([CONTEXT], [IIT Gandhinagar’s permanent campus is located in Palaj, on the banks of the Sabarmati River.], color: TEAL, fill: PALE-TEAL)
#v(8pt)
#two(
  l1-card([QUESTION], [Where is IITGN’s permanent campus?], color: BLUE, fill: PALE-BLUE),
  l1-card([ANSWER], [Palaj], color: ACC, fill: PALE-ACC),
)
#pause
#v(9pt)
#l1-note([For this question, we need the location. Another question may need a different part of the context.], color: BLUE)

== Represent similar meanings with nearby vectors #V

#align(center, scale(x: 76%, y: 76%, reflow: true, diagram(
  spacing: (24mm, 14mm),
  node-stroke: 0.8pt + INK,
  node-fill: white,
  {
    node((0, 0), [student solved assignment], fill: PALE-TEAL, stroke: 1pt + TEAL)
    node((1.2, .25), [learner completed homework], fill: PALE-TEAL, stroke: 1pt + TEAL)
    node((4, 1.5), [heavy rain flooded the road], fill: CREAM, stroke: 1pt + MUTED)
    edge((0, 0), (1.2, .25), "<->", stroke: 1.5pt + GREEN, label: text(size: 12.5pt, weight: 650, fill: GREEN)[near])
    edge((1.2, .25), (4, 1.5), "<->", stroke: (paint: RED, dash: "dashed", thickness: 1pt), label: text(size: 12.5pt, weight: 650, fill: RED)[far])
  },
)))
#pause
#v(8pt)
#claim[Can similar sentences have nearby vectors?]

== Generation has many valid answers #Q

#two(
  [
    #l1-card([PROMPT], [The student opened the laptop and …], color: TEAL, fill: PALE-TEAL)
    #pause
    #v(7pt)
    #grid(
      columns: (1fr, 1fr),
      gutter: 7pt,
      row-gutter: 7pt,
      l1-card([A], [started coding.], color: BLUE),
      l1-card([B], [checked email.], color: BLUE),
      l1-card([C], [joined class.], color: BLUE),
      l1-card([D], [saw a warning.], color: BLUE),
    )
  ],
  [#align(center, image("../figures/many_continuations_illustration_imagen.png", height: 77mm, fit: "contain"))],
  ratio: (.92fr, 1.08fr),
  gutter: 12pt,
)
#pause
#v(5pt)
#claim[Several answers make sense. What should the model output?]

== Generation is still classification #V

#l1-card([PREFIX], [The student opened the laptop and], color: TEAL)
#v(6pt)
#align(center, text(size: 24pt, fill: MUTED)[$arrow.b$])
#pause
#prob-bars((
  ([started], .36, BLUE),
  ([checked], .27, BLUE),
  ([joined], .21, BLUE),
  ([closed], .10, BLUE),
  ([other], .06, MUTED),
), hgt: 31mm)
#pause
#v(7pt)
#claim[A probability for each possible next token.]

== Generate a name #V

#two(
  [
    #align(center, text(size: 24pt, weight: 650)[Start with a small dataset.])
    #pause
    #v(9pt)
    #claim[Can a neural network generate Indian names?]
  ],
  [#align(center, image("../figures/tiny_character_generator_imagen.png", height: 65mm, fit: "contain"))],
  ratio: (.86fr, 1.14fr),
  gutter: 10pt,
)
#pause
#v(5pt)
#align(center, grid(
  columns: 6,
  gutter: 8pt,
  token([aabid], w: 25mm, color: ACC, fill: PALE-ACC),
  token([aarav], w: 25mm),
  token([avni], w: 25mm),
  token([naveen], w: 25mm),
  token([vani], w: 25mm),
  token([zeel], w: 25mm),
))
#source-line([Worked example: #link(nipun-lm)[Nipun Batra, Next-Token Prediction].])

== What exactly should we predict? #Q

#three(
  l1-card([CHARACTER], [`transformer` $arrow.r$ `t | r | a | …`], color: TEAL, fill: PALE-TEAL),
  l1-card([WORD], [`deep learning` $arrow.r$ `deep | learning`], color: ACC, fill: PALE-ACC),
  l1-card([SUBWORD], [`unbelievable` $arrow.r$ `un | believ | able`], color: PURPLE, fill: PALE-PURPLE),
)
#pause
#v(12pt)
#claim[A *token* is one unit the model predicts.]
#pause
#l1-note([Start with one character per token. We can write out every training example and vector.], color: TEAL)

// ───────────────────────── ACT 2 · REBUILD AABID ─────────────────────────

== The task: generate an Indian name #V

#two(
  [
    #hairline([VOCABULARY], [
      #l1-vocab-strip()
      #v(7pt)
      #align(center, text(size: 17pt)[$cal(V)={-,a,dots,z}, quad abs(cal(V))=27$])
    ], color: TEAL)
  ],
  l1-card([BOUNDARY SYMBOL], [
    Put `-` before a name to fill the context, and after it to mark the end. In this example, the same symbol handles BOS, padding, and EOS.
  ], color: ACC, fill: PALE-ACC),
  ratio: (1.35fr, 1fr),
)
#pause
#v(10pt)
#claim[Learn a probability distribution over 27 possible next characters.]

== Generation is repeated 27-class classification #V

#l1-pair(([a], [a], [b]), [?])
#pause
#v(8pt)
#prob-bars((
  ([a], .03, BLUE),
  ([d], .02, BLUE),
  ([i], .82, ACC),
  ([n], .04, BLUE),
  ([other], .09, MUTED),
), hgt: 31mm)
#pause
#v(8pt)
#claim[$f_(theta)(a,a,b)=p_(theta)(c_"next" | a,a,b)$]

== Choose a context length #Q

#align(center, text(size: 20pt)[How much history does the first model receive?])
#v(8pt)
#three(
  l1-card([$k=1$], [one previous character], color: MUTED),
  l1-card([$k=3$], [three previous characters], color: TEAL, fill: PALE-TEAL),
  l1-card([$k="all"$], [a variable-length prefix], color: MUTED),
)
#pause
#v(12pt)
#claim[Use $k=3$: $p_(theta)(x_t | x_(t-3:t-1))$.]
#l1-note([This gives the MLP a fixed-size input. We will try longer contexts later.], color: BLUE)

== Construct the training examples #V

#align(center, text(size: 13pt)[
  #table(
    columns: 7,
    stroke: 0.55pt + MUTED,
    inset: (x: 8pt, y: 6pt),
    align: center,
    table.header([*pair*], [1], [2], [3], [4], [5], [6]),
    [*context*], [`---`], [`--a`], [`-aa`], [`aab`], [`abi`], [`bid`],
    [*target*], [#text(fill: ACC)[`a`]], [#text(fill: ACC)[`a`]], [#text(fill: ACC)[`b`]], [#text(fill: ACC)[`i`]], [#text(fill: ACC)[`d`]], [#text(fill: ACC)[`-`]],
  )
])
#pause
#v(12pt)
#claim[Six contexts, six next-character targets.]
#pause
#l1-note([Predict the five characters in the name, then predict `-` after `d` to end it.], color: RED)

== Slide the context window #V

#l1-card([NAME WITH BOUNDARY SYMBOLS], [
  #align(center, text(size: 18pt, font: "IBM Plex Mono")[`-  -  -  a  a  b  i  d  -`])
], color: INK, fill: CREAM)
#v(8pt)
#l1-window([-], [-], [-], [a], [WINDOW 1])
#pause
#v(5pt)
#l1-window([-], [-], [a], [a], [SHIFT RIGHT])
#pause
#v(5pt)
#l1-window([-], [a], [a], [b], [SHIFT RIGHT])
#pause
#v(7pt)
#l1-note([Later, causal self-attention will let us predict these targets in parallel during training.], color: GREEN)

== Repeat for every name #D

#align(center, grid(
  columns: (1fr, 14mm, 1fr),
  gutter: 8pt,
  align: horizon,
  l1-card([ONE NAME], [length $L$ #linebreak() $L$ characters + one EOS target], color: TEAL),
  text(size: 24pt, fill: MUTED)[$arrow.r$],
  l1-card([EXAMPLES], [$L+1$ context-target pairs], color: ACC, fill: PALE-ACC),
))
#pause
#v(14pt)
#claim[$N_"examples" = sum_(n=1)^N (L_n+1)$]
#pause
#l1-note([Split the names into training, validation, and test sets *before* making the windows.], color: INK)

== How do we turn characters into numbers? #Q

#l1-held("tokens")
#v(7pt)
#two(
  l1-card([CHARACTERS], [`a`, `b`, `i`: categorical symbols], color: TEAL, fill: PALE-TEAL),
  l1-card([MLP INPUT], [$x in RR^d$: a vector of numbers], color: BLUE, fill: PALE-BLUE),
)
#v(12pt)
#align(center, text(size: 26pt, fill: MUTED)[$? arrow.r$])
#pause
#claim[Give each character a one-hot vector.]

== One-hot encoding #D

#l1-held("tokens")
#v(7pt)
#align(center, text(size: 14pt)[
  #table(
    columns: (22mm, 19mm, 19mm, 19mm, 19mm, 19mm, 19mm),
    stroke: 0.55pt + MUTED,
    inset: (x: 6pt, y: 5pt),
    align: center,
    table.header([*token*], [1], [2], [3], [9], [$dots$], [27]),
    [`a`], [$1$], [$0$], [$0$], [$0$], [$dots$], [$0$],
    [`b`], [$0$], [$1$], [$0$], [$0$], [$dots$], [$0$],
    [`i`], [$0$], [$0$], [$0$], [$1$], [$dots$], [$0$],
  )
])
#pause
#v(9pt)
#claim[$o_a^T o_b=o_a^T o_i=o_b^T o_i=0$]
#pause
#l1-note([Every pair of different characters has dot product zero. These vectors identify characters; they do not tell us which characters are similar.], color: MUTED)

== Learn an embedding for each character #V

#l1-held("embed")
#v(7pt)
#two(
  [
    #align(center, text(size: 14pt)[
      #table(
        columns: (18mm, 24mm, 24mm),
        stroke: 0.55pt + MUTED,
        inset: (x: 7pt, y: 5pt),
        align: center,
        table.header([*row*], [*dim 1*], [*dim 2*]),
        [`-`], [$0.1$], [$-0.4$],
        [`a`], [$1.2$], [$0.1$],
        [`b`], [$0.2$], [$1.1$],
        [$dots$], [$dots$], [$dots$],
        [`z`], [$-0.3$], [$0.6$],
      )
    ])
  ],
  [
    #l1-card([EMBEDDING TABLE], [$E in RR^(27 times 2)$], color: TEAL, fill: PALE-TEAL)
    #v(8pt)
    #l1-card([LOOKUP RULE], [$e_c=E["id"(c)]$], color: BLUE, fill: PALE-BLUE)
    #pause
    #v(8pt)
    #l1-card([LEARNABLE], [Backpropagation updates the rows to help predict the next token.], color: GREEN, fill: PALE-GREEN)
  ],
  ratio: (1fr, 1.1fr),
)

== Look up the embedding #V

#l1-held("embed")
#v(7pt)
#two(
  [
    #align(center, text(size: 13pt)[
      #show math.equation: set text(size: 14pt)
      #table(
        columns: (18mm, 16mm, 51mm),
        stroke: 0.55pt + MUTED,
        inset: (x: 7pt, y: 6pt),
        align: center,
        table.header([*token*], [*id*], [*selected row*]),
        [#text(fill: TEAL)[`a`]], [$1$], [$E[1]=(1.2,0.1)$],
        [#text(fill: TEAL)[`b`]], [$2$], [$E[2]=(0.2,1.1)$],
        [#text(fill: TEAL)[`i`]], [$9$], [$E[9]=(-0.8,0.9)$],
      )
    ])
  ],
  [
    #pause
    #set text(size: 16pt)
    #torch-code(
      "import torch\nembed = torch.nn.Embedding(27, 2)\nids = torch.tensor([1, 1, 2])\nrows = embed(ids)\nassert rows.shape == (3, 2)",
      takeaway: [Each ID selects one trainable row.],
      color: TEAL,
    )
  ],
  ratio: (1.08fr, .92fr),
)

== Check the shapes #D

#l1-held("embed")
#v(7pt)
#l1-pair(([a], [a], [b]), [i])
#v(8pt)
#align(center, grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  l1-card([INTEGER INPUT], [$X=(1,1,2)$ #linebreak() shape: $(3,)$], color: TEAL, fill: PALE-TEAL),
  l1-card([EMBEDDINGS], [
    $E[X]=mat(1.2,0.1; 1.2,0.1; 0.2,1.1)$ #linebreak()
    shape: $3 times 2$
  ], color: BLUE, fill: PALE-BLUE),
))
#pause
#v(8pt)
#claim[One row per token, one column per learned feature.]

== Concatenate the embeddings #D

#l1-held("join")
#v(7pt)
#align(center, text(size: 18pt)[
  $mat(1.2,0.1; 1.2,0.1; 0.2,1.1)$
  #h(10pt)
  $arrow.r$
  #h(10pt)
  $h_0=[1.2,0.1; 1.2,0.1; 0.2,1.1]$
])
#pause
#v(13pt)
#align(center, grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 5pt,
  l1-card([POSITION 1 · `a`], [$[1.2,0.1]$], color: TEAL, fill: PALE-TEAL),
  l1-card([POSITION 2 · `a`], [$[1.2,0.1]$], color: BLUE, fill: PALE-BLUE),
  l1-card([POSITION 3 · `b`], [$[0.2,1.1]$], color: ACC, fill: PALE-ACC),
))
#pause
#v(9pt)
#claim[$h_0 in RR^(k d)=RR^6$]

== Why concatenate? #Q

#l1-held("join")
#v(7pt)
#two(
  l1-card([ORDER 1], [
    `a b i` $arrow.r$ $[e_a; e_b; e_i]$
  ], color: TEAL, fill: PALE-TEAL),
  l1-card([ORDER 2], [
    `i b a` $arrow.r$ $[e_i; e_b; e_a]$
  ], color: ACC, fill: PALE-ACC),
)
#pause
#v(10pt)
#claim[$[e_a;e_b;e_i] != [e_i;e_b;e_a]$]
#pause
#v(8pt)
#l1-note([Adding gives the same result in both orders: $e_a+e_b+e_i=e_i+e_b+e_a$. Concatenation keeps each position in a separate part of the vector.], color: BLUE)

== Pass the vector through an MLP #D

#two(
  [
    #scale(
      x: 82%, y: 82%, reflow: true,
      neural-net-sketch(
        input-labels: ([$h_1$], [$h_2$], [$dots$], [$h_6$]),
        output-labels: ([-], [a], [i], [$dots$], [z]),
        hidden: 5,
        highlight-output: 2,
      ),
    )
  ],
  [
    #pause
    #l1-card([ONE HIDDEN LAYER], [
      $h_1=sigma(W_1 h_0+b_1)$
      #v(7pt)
      $z=W_2h_1+b_2$
      #v(7pt)
      $W_1 in RR^(H times 6), quad W_2 in RR^(27 times H)$
      #v(7pt)
      $6$ input numbers; $27$ output scores
    ], color: BLUE, fill: PALE-BLUE)
  ],
  ratio: (1.12fr, .88fr),
)

== Why exactly 27 outputs? #Q

#l1-held("logits")
#v(7pt)
#l1-vocab-strip()
#v(10pt)
#align(center, text(size: 18pt)[$z=W_2 h_1+b_2 in RR^27$])
#pause
#v(11pt)
#three(
  l1-card([LOGIT $z_-$], [score for boundary], color: BLUE),
  l1-card([LOGIT $z_a$], [score for `a`], color: BLUE),
  l1-card([LOGIT $z_i$], [score for `i`], color: ACC, fill: PALE-ACC),
)
#pause
#v(8pt)
#claim[One score for each possible next character.]

== Turn the 27 scores into probabilities #D

#l1-held("softmax")
#v(7pt)
#two(
  [
    #l1-card([EXAMPLE LOGITS FOR `aab`], [
      $z_i=5, quad z_a=1, quad z_d=0.5,$ #linebreak()
      $z_-=-0.5,$ and every other logit $=-1$.
    ], color: BLUE, fill: PALE-BLUE)
    #v(8pt)
    #align(center, text(size: 18pt)[$p_j=frac(exp(z_j),sum_(r in cal(V)) exp(z_r))$])
  ],
  [
    #pause
    #prob-bars((
      ([`i`], .917, ACC),
      ([`a`], .017, BLUE),
      ([`d`], .010, BLUE),
      ([`-`], .004, BLUE),
      ([23 × other], .052, MUTED),
    ), hgt: 31mm)
  ],
  ratio: (1.05fr, 1fr),
)
#pause
#l1-note([All 27 probabilities are positive and sum to one. The `other` bar groups the remaining 23 characters.], color: BLUE)

== Compute the loss #D

#l1-held("loss")
#v(7pt)
#align(center, text(size: 22pt)[For target `i`: $ell=-log p_(theta)(i | a,a,b)$])
#v(12pt)
#two(
  l1-card([CONFIDENT AND CORRECT], [
    $p(i)=0.80$ #linebreak()
    $ell=-log(0.80) approx 0.223$
  ], color: GREEN, fill: PALE-GREEN),
  l1-card([CONFIDENT AND WRONG], [
    $p(i)=0.01$ #linebreak()
    $ell=-log(0.01) approx 4.605$
  ], color: RED, fill: PALE-RED),
)
#pause
#claim[The more probability we give `i`, the smaller the loss.]

== The loss in PyTorch

#l1-pair(([a], [a], [b]), [i])
#v(9pt)
#torch-code(
  "import torch\nimport torch.nn.functional as F\n# logits has shape [1, 27]\ntarget = torch.tensor([9])      # token i\nloss = F.cross_entropy(logits, target)\nloss.backward()",
  takeaway: [Pass the target ID to cross-entropy. Backpropagation computes gradients for the whole network.],
  color: RED,
)

== Learn the embeddings and MLP weights #V

#l1-held("none")
#v(7pt)
#align(center, diagram(
  spacing: (18mm, 12mm),
  node-stroke: 1pt + INK,
  node-fill: white,
  {
    flow-node((0, 0), [embedding $E$], color: TEAL, fill: PALE-TEAL, w: 31mm)
    flow-node((1.6, 0), [$W_1,b_1$], color: BLUE, fill: PALE-BLUE, w: 28mm)
    flow-node((3.2, 0), [$W_2,b_2$], color: BLUE, fill: PALE-BLUE, w: 28mm)
    flow-node((4.8, 0), [cross-entropy], color: RED, fill: PALE-RED, w: 34mm)
    flow-arrow((0, 0), (1.6, 0))
    flow-arrow((1.6, 0), (3.2, 0))
    flow-arrow((3.2, 0), (4.8, 0))
    edge((4.8, 0), (3.2, 0), "-|>", bend: 35deg, stroke: 1.2pt + RED, label: text(size: 10pt, fill: RED)[gradient])
    edge((3.2, 0), (1.6, 0), "-|>", bend: 35deg, stroke: 1.2pt + RED)
    edge((1.6, 0), (0, 0), "-|>", bend: 35deg, stroke: 1.2pt + RED)
  },
))
#pause
#v(9pt)
#claim[The same loss trains the embedding table and the MLP.]
#l1-note([For `aab → i`, we look up the rows for `a` and `b`, so those rows receive gradients. The target `i` does not cause an embedding lookup.], color: ACC)

== When could two embeddings become similar? #V

#l1-held("embed")
#v(7pt)
#align(center, diagram(
  spacing: (22mm, 13mm),
  node-stroke: 0.8pt + INK,
  node-fill: white,
  {
    node((0, 0), [`a`], radius: 5mm, fill: PALE-TEAL, stroke: 1pt + TEAL)
    node((.8, .2), [`e`], radius: 5mm, fill: PALE-TEAL, stroke: 1pt + TEAL)
    node((1.4, -.15), [`i`], radius: 5mm, fill: PALE-TEAL, stroke: 1pt + TEAL)
    node((3.4, 1.2), [`q`], radius: 5mm, fill: CREAM, stroke: 1pt + MUTED)
    node((4.1, .5), [`x`], radius: 5mm, fill: CREAM, stroke: 1pt + MUTED)
    edge((0, 0), (.8, .2), "-", stroke: 1.3pt + GREEN)
    edge((.8, .2), (1.4, -.15), "-", stroke: 1.3pt + GREEN)
  },
))
#pause
#v(8pt)
#align(center, scale(x: 118%, y: 118%, reflow: true, simple-flow((
  ([similar prediction contexts], TEAL, PALE-TEAL),
  ([similar gradient updates], PURPLE, PALE-PURPLE),
  ([embeddings may move closer], GREEN, PALE-GREEN),
))))

== Sample, append, repeat #V

#l1-held("softmax")
#v(7pt)
#l1-window([-], [-], [-], [a], [START])
#pause
#v(4pt)
#l1-window([-], [-], [a], [a], [SHIFT + SAMPLE])
#pause
#v(4pt)
#l1-window([-], [a], [a], [b], [SHIFT + SAMPLE])
#pause
#v(4pt)
#align(center, text(size: 14pt, fill: MUTED)[
  `aab` $arrow.r$ `i` $arrow.r$ `abi` $arrow.r$ `d` $arrow.r$ `bid` $arrow.r$ `-`
])
#pause
#v(5pt)
#claim[We sampled `-`, so stop. The generated name is `aabid`.]

== Training and generation #V

#grid(
  columns: (27mm, 1fr),
  rows: (auto, auto),
  gutter: 8pt,
  row-gutter: 12pt,
  align: horizon,
  l1-label([TRAIN], color: TEAL),
  simple-flow((
    ([known context + target], TEAL, PALE-TEAL),
    ([model], BLUE, PALE-BLUE),
    ([cross-entropy], RED, PALE-RED),
    ([update weights], GREEN, PALE-GREEN),
  )),
  [#pause #l1-label([GENERATE], color: BLUE)],
  [#pause #simple-flow((
    ([current context], TEAL, PALE-TEAL),
    ([same model], BLUE, PALE-BLUE),
    ([sample], ACC, PALE-ACC),
    ([append + repeat], GREEN, PALE-GREEN),
  ))],
)
#pause
#v(9pt)
#claim[Training uses the observed next token. Generation samples one.]

== Our character language model #V

#align(center, text(size: 28pt, weight: 650)[$p_(theta)(x_t | x_(t-3:t-1))$])
#pause
#v(12pt)
#three(
  l1-card([SIZE], [27 symbols and three context characters], color: MUTED),
  l1-card([CHARACTER-LEVEL], [one character is one token], color: TEAL),
  l1-card([LANGUAGE MODEL], [predicts the next token from previous tokens], color: ACC, fill: PALE-ACC),
)
#pause
#v(10pt)
#claim[Much larger generative models use this same next-token objective.]
#source-line([Neural language model: #link(bengio-lm)[Bengio et al. (2003)] · worked example: #link(nipun-lm)[Next-Token Prediction].])

// ───────────────────────── ACT 3 · WHAT IS A TOKEN? ─────────────────────────

== Why begin with characters? #V

#l1-card([SENTENCE], [`deep learning is amazing`], color: TEAL)
#pause
#v(9pt)
#align(center, text(size: 14pt, font: "IBM Plex Mono", fill: MUTED)[
  `d | e | e | p | ␠ | l | e | a | r | n | i | n | g | ␠ | i | s | …`
])
#pause
#v(12pt)
#two(
  l1-card([SMALL VOCABULARY], [we can write out every lookup and target], color: GREEN, fill: PALE-GREEN),
  l1-card([LONG SEQUENCES], [even an ordinary sentence has many tokens], color: RED, fill: PALE-RED),
)

== Character tokens #V

#align(center, text(size: 19pt, font: "IBM Plex Mono")[
  `transformer` $arrow.r$ `t | r | a | n | s | f | o | r | m | e | r`
])
#v(12pt)
#three(
  l1-card([VOCABULARY], [tiny: letters, digits, punctuation, bytes], color: GREEN, fill: PALE-GREEN),
  l1-card([SEQUENCE], [very long], color: RED, fill: PALE-RED),
  l1-card([UNKNOWN TEXT], [almost none at byte/Unicode-unit level], color: TEAL, fill: PALE-TEAL),
)
#pause
#v(10pt)
#claim[Small units cover many strings. They also produce long sequences.]

== Word tokens #V

#align(center, text(size: 17pt, font: "IBM Plex Mono")[
  `deep learning is useful` $arrow.r$ `deep | learning | is | useful`
])
#v(10pt)
#l1-card([ONE ROOT, MANY VOCABULARY ENTRIES], [
  `walk` · `walks` · `walked` · `walking` · `walker` · `walkability`
], color: ACC, fill: PALE-ACC)
#pause
#v(10pt)
#two(
  l1-card([SHORT SEQUENCES], [fewer tokens; words are easy to interpret], color: GREEN, fill: PALE-GREEN),
  l1-card([LARGE VOCABULARY], [many entries; unseen words need handling], color: RED, fill: PALE-RED),
)

== What happens to an unseen word? #V

#align(center, grid(
  columns: (1fr, 14mm, 1fr),
  gutter: 8pt,
  align: horizon,
  l1-card([TEST STRING], [`hyperhappiness`], color: TEAL, fill: PALE-TEAL),
  text(size: 25pt, fill: MUTED)[$arrow.r$],
  bad-token([`<UNK>`], w: 36mm),
))
#pause
#v(13pt)
#align(center, grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 8pt,
  l1-card([`hyperhappiness`], [`<UNK>`], color: RED, fill: PALE-RED),
  l1-card([`electrojoy`], [`<UNK>`], color: RED, fill: PALE-RED),
  l1-card([`nanobotany`], [`<UNK>`], color: RED, fill: PALE-RED),
))
#pause
#v(8pt)
#claim[These different words all become `<UNK>`.]

== Can we split text at spaces? #Q

#l1-card([MIXED TEXT], [IITGN में आज transformer lecture है :)], color: TEAL, fill: PALE-TEAL)
#v(7pt)
#two(
  l1-card([URL], [`https://iitgn.ac.in/events?id=42`], color: BLUE),
  l1-card([CODE], [`scores[i] += q @ k.T`], color: PURPLE),
)
#pause
#v(10pt)
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 7pt,
  l1-card([SCRIPTS], [Devanagari + Latin], color: INK),
  l1-card([PUNCTUATION], [`://?=[]@`], color: INK),
  l1-card([EMOJI], [`:)` or Unicode], color: INK),
  l1-card([WHITESPACE], [not a universal boundary], color: INK),
)
#pause
#claim[What counts as a word in each example?]

== Subword tokens #V

#grid(
  columns: (37mm, 12mm, 1fr),
  gutter: 7pt,
  row-gutter: 9pt,
  align: horizon,
  token([unbelievable], w: 37mm),
  text(fill: MUTED)[$arrow.r$],
  l1-card([CHUNKS], [`un | believ | able`], color: PURPLE, fill: PALE-PURPLE),
  token([tokenization], w: 37mm),
  text(fill: MUTED)[$arrow.r$],
  l1-card([CHUNKS], [`token | iz | ation`], color: PURPLE, fill: PALE-PURPLE),
  token([electroencephalography], w: 37mm, size: 9pt),
  text(fill: MUTED)[$arrow.r$],
  l1-card([CHUNKS], [`electro | encephalo | graphy`], color: PURPLE, fill: PALE-PURPLE),
)
#pause
#v(10pt)
#l1-note([Reuse common pieces to represent both frequent and rare strings. We will leave the BPE training algorithm for later.], color: PURPLE)

== Choose the token size #V

#align(center, grid(
  columns: (45mm, 20mm, 45mm, 20mm, 45mm),
  gutter: 3pt,
  align: horizon,
  l1-continuum-node([CHARACTERS], [tiny $V$ · long $T$], TEAL),
  text(size: 24pt, fill: MUTED)[$arrow.r$],
  l1-continuum-node([SUBWORDS], [moderate $V$ · moderate $T$], PURPLE),
  text(size: 24pt, fill: MUTED)[$arrow.r$],
  l1-continuum-node([WORDS], [huge $V$ · short $T$], ACC),
))
#pause
#v(14pt)
#align(center, grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  l1-card([LARGER UNITS], [more meaning per displayed token], color: ACC, fill: PALE-ACC),
  l1-card([SMALLER UNITS], [easier to represent unfamiliar strings], color: TEAL, fill: PALE-TEAL),
))
#source-line([Tokenization comparison: #link(stanford-l1)[Stanford CME 295 L1] · #link(mit-l2)[MIT 6.S191 L2].])

== Compare the trade-off #V

#align(center, text(size: 12.5pt)[
  #table(
    columns: (30mm, 35mm, 35mm, 40mm, 46mm),
    stroke: 0.55pt + MUTED,
    inset: (x: 6pt, y: 6pt),
    align: (left, center, center, center, left),
    table.header([*unit*], [*$V$*], [*$T$*], [*unseen text*], [*in practice*]),
    [character], [small], [long], [excellent], [meaning spans many steps],
    [subword], [moderate], [moderate], [strong], [balances $V$ and $T$],
    [word], [huge], [short], [brittle], [easy to read],
  )
])
#pause
#v(9pt)
#claim[Tokenization changes vocabulary size $V$ and sequence length $T$.]
#pause
#l1-note([We will use $T$ again in attention: its score matrix has shape $T times T$.], color: RED)

== Tokens become integer IDs #V

#two(
  [
    #align(center, text(size: 13pt)[
      #table(
        columns: (1fr, 1fr),
        stroke: 0.55pt + MUTED,
        inset: (x: 8pt, y: 5pt),
        align: center,
        table.header([*token*], [*id*]),
        [`The`], [$17$],
        [`cat`], [$42$],
        [`sat`], [$91$],
        [`on`], [$8$],
        [`the`], [$5$],
        [`mat`], [$73$],
      )
    ])
  ],
  [
    #l1-card([TOKEN SEQUENCE], [`The | cat | sat | on | the | mat`], color: TEAL, fill: PALE-TEAL)
    #v(10pt)
    #pause
    #l1-card([ID SEQUENCE], [$[17,42,91,8,5,73]$], color: BLUE, fill: PALE-BLUE)
    #v(10pt)
    #l1-note([An ID labels a row in the learned embedding table.], color: ACC)
  ],
  ratio: (.8fr, 1.2fr),
)

== Use token IDs in the model #D

#align(center, text(size: 27pt, weight: 650)[$x_1,x_2,dots,x_T$])
#v(9pt)
#claim[$x_i in {1,2,dots,V}$]
#pause
#v(11pt)
#three(
  l1-card([DISPLAY], [We will often show whole words.], color: TEAL),
  l1-card([MODEL INPUT], [Each displayed unit has a token ID.], color: BLUE),
  l1-card([OBJECTIVE], [Predict the next character, subword, or word.], color: ACC, fill: PALE-ACC),
)

// ───────────────────────── ACT 4 · NATURAL LANGUAGE ─────────────────────────

== Change the tokens, keep the model #V

#two(
  [
    #l1-label([CHARACTER MODEL], color: TEAL)
    #v(7pt)
    #l1-pair(([a], [a], [b]), [i])
    #v(9pt)
    #align(center, text(size: 13pt, fill: MUTED)[embedding $arrow.r$ concatenate $arrow.r$ MLP $arrow.r$ softmax])
  ],
  [
    #l1-label([TOKEN MODEL], color: BLUE)
    #v(7pt)
    #l1-pair(([cat], [sat], [on]), [the])
    #v(9pt)
    #align(center, text(size: 13pt, fill: MUTED)[embedding $arrow.r$ concatenate $arrow.r$ MLP $arrow.r$ softmax])
  ],
)
#pause
#v(11pt)
#claim[We changed the vocabulary, the tokenizer, and the training data.]

== What could come next? #Q

#l1-card([PREFIX], [`The cat sat on the ___`], color: TEAL, fill: PALE-TEAL)
#pause
#v(9pt)
#prob-bars((
  ([mat], .43, ACC),
  ([floor], .25, BLUE),
  ([chair], .15, BLUE),
  ([sofa], .10, BLUE),
  ([other], .07, MUTED),
), hgt: 31mm)
#pause
#v(8pt)
#l1-note([The training text gives us one target at each position. The model can give probability to several reasonable continuations.], color: BLUE)

== Construct examples from a sentence #V

#align(center, text(size: 13pt)[
  #table(
    columns: (1fr, 22mm),
    stroke: 0.55pt + MUTED,
    inset: (x: 10pt, y: 5pt),
    align: (left, center),
    table.header([*visible prefix*], [*next target*]),
    [`<BOS>`], [#text(fill: ACC)[`deep`]],
    [`deep`], [#text(fill: ACC)[`learning`]],
    [`deep learning`], [#text(fill: ACC)[`is`]],
    [`deep learning is`], [#text(fill: ACC)[`fun`]],
    [`deep learning is fun`], [#text(fill: ACC)[`<EOS>`]],
  )
])
#pause
#v(9pt)
#claim[Keep the whole prefix this time. It grows by one token per row.]
#v(5pt)
#l1-note([Write $x_1=$ `<BOS>` and $x_T=$ `<EOS>`. At position $t$, predict $x_(t+1)$.], color: PURPLE)

== Multiply the next-token probabilities #D

#align(center, text(size: 22pt, weight: 650)[`deep learning is fun`])
#v(8pt)
#align(center, [
  #text(size: 18pt)[$p("deep learning is fun")$]
#pause
  #v(4pt)
  #text(size: 18pt)[$= p("deep")$]
#pause
  #v(4pt)
  #text(size: 18pt)[$quad times p("learning" | "deep")$]
#pause
  #v(4pt)
  #text(size: 18pt)[$quad times p("is" | "deep learning")$]
#pause
  #v(4pt)
  #text(size: 18pt)[$quad times p("fun" | "deep learning is").$]
])
#v(7pt)
#l1-note([The chain rule gives this product exactly. We still need a model to turn each prefix into a prediction.], color: INK)

== How should we represent the context? #D

#align(center, text(size: 25pt, weight: 650)[
  $cal(L)(theta)=-sum_(t=1)^(T-1) log p_(theta)(x_(t+1) | x_(<=t))$
])
#pause
#v(12pt)
#two(
  l1-card([TRAINING OBJECTIVE], [Increase the probability of the observed next token at every position.], color: GREEN, fill: PALE-GREEN),
  l1-card([MODEL DESIGN], [How do we represent a prefix that keeps getting longer?], color: BLUE, fill: PALE-BLUE),
)
#pause
#v(10pt)
#claim[The same loss can train different ways of representing the context.]

== What if the context keeps growing? #Q

#two(
  [
    #l1-label([THREE CHARACTERS], color: TEAL)
    #v(8pt)
    #l1-pair(([a], [a], [b]), [i])
    #v(8pt)
    #align(center, dim-pill([$3 times d$ values $arrow.r$ concatenate], color: TEAL))
  ],
  [
    #l1-label([GROWING PREFIX], color: BLUE)
    #v(8pt)
    #l1-pair(([The], [cat], [sat], [on], [the]), [mat])
    #v(8pt)
    #align(center, dim-pill([length can keep growing], color: BLUE))
  ],
)
#pause
#v(13pt)
#claim[How can the model use a longer context?]
#pause
#v(7pt)
#l1-note([Next, compare concatenation and averaging. Then let the model choose which parts of the context to use for each query.], color: ACC)
