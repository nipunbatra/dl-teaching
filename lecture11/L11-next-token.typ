// Next-Token Prediction - public Lecture 12 · ES 667 Deep Learning
// Compile from the repository root:
//   typst compile --root . --input handout=true lecture11/L11-next-token.typ /private/tmp/L11-handout.pdf
//   typst compile --root . lecture11/L11-next-token.typ /private/tmp/L11-presentation.pdf

#import "../common/metropolis.typ": *
#import "@preview/lilaq:0.4.0" as lq
#show: metropolis-deck.with(
  title: [Next-Token Prediction],
  subtitle: [Tokens, embeddings, and one complete neural language-model calculation],
)

#let NB = "https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L11/01_char_mlp_language_model.ipynb"
#let BENGIO = "https://www.jmlr.org/papers/v3/bengio03a.html"

#let PALE-TEAL = rgb("#DCEBEB")
#let PALE-BLUE = rgb("#E7F0FA")
#let PALE-ACC = rgb("#FDECD6")
#let PALE-GREEN = rgb("#E5F3E9")
#let PALE-RED = rgb("#FBE8E9")
#let CREAM = rgb("#F4F1EA")

// EXECUTED evidence from notebook cell 19. The CSV stores every post-update
// point from the seed-11 run rather than a smoothed or illustrative redraw.
// Rebuild/check it with evidence/build_cell19_curve_data.py, which pins the
// source notebook SHA-256 and verifies the CSV byte-for-byte.
#let nll-curve-data = csv("evidence/cell19_train_heldout_nll.csv", row-type: dictionary)
#let nll-updates = nll-curve-data.map(row => float(row.at("update")))
#let nll-train = nll-curve-data.map(row => float(row.at("train_nll")))
#let nll-heldout = nll-curve-data.map(row => float(row.at("heldout_nll")))
#let nll-uniform = nll-updates.map(_ => calc.ln(6))

// Stable grammar: TEAL = observed/true context and TRAIN; BLUE = model/logits/
// probabilities and INFER; ACC = target/chosen token; GREEN = correct/retained;
// RED = loss/error; INK = EVAL and conventions.
#let swatch(color, body) = box(width: 13pt, height: 4pt, fill: color, radius: 1pt) + h(4pt) + body
#let curve-key(color, body, dashed: false) = {
  let curve-stroke = if dashed {
    (paint: color, dash: "dashed", thickness: 1.2pt)
  } else {
    1.8pt + color
  }
  box(width: 15pt, height: 5pt, align(horizon, line(length: 15pt, stroke: curve-stroke))) + h(4pt) + body
}
#let lm-legend = align(center, text(size: 11.5pt)[
  #swatch(TEAL, [observed context / TRAIN]) #h(10pt)
  #swatch(BLUE, [model score or probability / INFER]) #h(10pt)
  #swatch(ACC, [target or chosen token]) #h(10pt)
  #swatch(GREEN, [correct / retained]) #h(10pt)
  #swatch(RED, [loss or error]) #h(10pt)
  #swatch(INK, [EVAL])
])

#let hairline(title, body, color: TEAL) = block(
  width: 100%, inset: 3pt,
  [#text(size: 13.5pt, weight: 700, fill: color)[#upper(title)]
   #v(2pt)
   #line(length: 100%, stroke: 0.8pt + color)
   #v(5pt)
   #body],
)

#let card(title, body, color: TEAL) = block(
  width: 100%, inset: (x: 10pt, y: 8pt), fill: white,
  stroke: 1pt + color, radius: 3pt,
  [#text(size: 12.5pt, weight: 700, fill: color)[#upper(title)]
   #v(4pt)
   #body],
)

#let note(body, color: TEAL) = block(
  width: 100%, inset: (left: 11pt, right: 5pt, top: 5pt, bottom: 5pt),
  stroke: (left: 2pt + color), fill: color.lighten(93%), radius: 2pt,
  [#body],
)

#let case-strip(stage, detail) = block(
  width: 100%, inset: (x: 10pt, y: 6pt),
  fill: rgb("#F3F6F5"), stroke: (left: 3pt + TEAL), radius: 3pt,
  [#text(size: 11pt, weight: 700, fill: TEAL, tracking: 0.7pt)[HELD CASE · #upper(stage)]
   #h(8pt)
   #text(size: 13.5pt, fill: INK)[#detail]],
)

#let _clock(name, active, col, body) = block(
  width: 100%, inset: (x: 8pt, y: 5pt), radius: 3pt,
  fill: if active { col.lighten(88%) } else { rgb("#F2F1EE") },
  stroke: if active { 1.2pt + col } else { 0.55pt + MUTED },
  [#text(size: 11pt, weight: 750, fill: if active { col } else { MUTED })[#name]
   #h(5pt)
   #text(size: 10.8pt, fill: if active { INK } else { MUTED })[#body]],
)

#let clock-strip(active) = align(center, grid(
  columns: (1fr, 1fr, 1fr), gutter: 9pt,
  _clock([TRAIN], active == "train", TEAL, [true context $arrow.r$ $p$; CE $arrow.r$ update]),
  _clock([INFER], active == "infer", BLUE, [current context $arrow.r$ $p$; choose $arrow.r$ append]),
  _clock([EVAL], active == "eval", INK, [held-out truth $arrow.r$ mean NLL / PPL]),
))

#let token-box(body, color: TEAL, fill: PALE-TEAL, w: 14mm) = box(
  width: w, height: 10mm, fill: fill, stroke: 0.9pt + color, radius: 2.5pt,
  align(center + horizon, text(size: 14pt, weight: 650, body)),
)

#let flow-node(pos, body, color: INK, fill: white, w: 29mm) = node(
  pos,
  block(width: w, inset: (x: 5pt, y: 5pt), align(center, text(size: 12pt, body))),
  shape: fletcher.shapes.rect, corner-radius: 3pt,
  stroke: 0.9pt + color, fill: fill,
)

#let flow-arrow(a, b, color: MUTED, label: none, bend: 0deg) = edge(
  a, b, "-|>", bend: bend, stroke: 0.85pt + color,
  label: if label == none { none } else { text(size: 10.5pt, fill: color, label) },
)

#let context-row(items, target: none) = align(center, grid(
  columns: if target == none { (auto, auto) } else { (auto, auto, 12mm, auto) },
  gutter: 7pt, align: horizon,
  ..items,
  ..if target == none { () } else { ([#text(size: 23pt, fill: MUTED)[$arrow.r$]], token-box(target, color: ACC, fill: PALE-ACC)) },
))

#let prob-base = (
  ([.], .028644, [.029], false),
  ([a], .575333, [.575], true),
  ([e], .077863, [.078], false),
  ([i], .028644, [.029], false),
  ([n], .077863, [.078], false),
  ([v], .211653, [.212], false),
)

#let prob-cold = (
  ([.], .002106, [.002]), ([a], .849672, [.850]), ([e], .015562, [.016]),
  ([i], .002106, [.002]), ([n], .015562, [.016]), ([v], .114991, [.115]),
)

#let prob-hot = (
  ([.], .080017, [.080]), ([a], .358609, [.359]), ([e], .131925, [.132]),
  ([i], .080017, [.080]), ([n], .131925, [.132]), ([v], .217508, [.218]),
)

#let probability-bars(values: prob-base, h: 34mm) = {
  let bar(item) = {
    let col = if item.len() > 3 and item.at(3) { ACC } else { BLUE }
    stack(dir: ttb, spacing: 2pt,
      text(size: 9.5pt, weight: 650, fill: col, item.at(2)),
      box(
        width: 13mm,
        height: h,
        align(bottom, block(
          width: 13mm,
          height: item.at(1) * h,
          fill: col.lighten(18%),
          radius: (top: 2pt),
        )),
      ),
    )
  }
  align(center, grid(
    columns: (12mm, 15mm, 15mm, 15mm, 15mm, 15mm, 15mm),
    rows: (h + 8mm, auto), gutter: 5pt, align: (center, bottom),
    [#box(width: 10mm, height: h)[
       #place(top + right, text(size: 8.5pt, fill: MUTED)[1.0])
       #place(bottom + right, text(size: 8.5pt, fill: MUTED)[0])
       #place(right, line(length: h, angle: 90deg, stroke: 0.7pt + MUTED))
     ]],
    ..values.map(bar),
    [],
    ..values.map(item => text(size: 11pt, weight: 650, item.at(0))),
  ))
}

#let tiny-distribution(title, values, color: BLUE) = card(title, [
  #align(center, grid(
    columns: (8mm, 8mm, 8mm, 8mm, 8mm, 8mm), gutter: 3pt, align: bottom,
    ..values.map(item => stack(dir: ttb, spacing: 1.5pt,
      block(width: 6mm, height: item.at(1) * 25mm, fill: color.lighten(18%), radius: (top: 1.5pt)),
      text(size: 8.5pt, item.at(0)),
    )),
  ))
], color: color)

#let model-pipeline = align(center, diagram(
  spacing: (16mm, 10mm), node-stroke: 1pt + INK, node-fill: white,
  {
    flow-node((0,0), [ids $(0,4)$], color: TEAL, fill: PALE-TEAL, w: 22mm)
    flow-node((1.25,0), [rows $e_.,e_n$], color: TEAL, fill: PALE-TEAL, w: 25mm)
    flow-node((2.5,0), [ordered $c in RR^4$], color: INK, fill: CREAM, w: 27mm)
    flow-node((3.75,0), [ReLU $h in RR^2$], color: BLUE, fill: PALE-BLUE, w: 25mm)
    flow-node((5,0), [logits $z in RR^6$], color: BLUE, fill: PALE-BLUE, w: 25mm)
    flow-node((6.25,0), [softmax $p$], color: BLUE, fill: PALE-BLUE, w: 23mm)
    flow-arrow((0,0),(1.25,0), color: TEAL)
    flow-arrow((1.25,0),(2.5,0), color: TEAL)
    flow-arrow((2.5,0),(3.75,0), color: BLUE)
    flow-arrow((3.75,0),(5,0), color: BLUE)
    flow-arrow((5,0),(6.25,0), color: BLUE)
  },
))

#title-slide()

// ───────────────────────── CORE · 0-55 MIN ─────────────────────────

== L11 classified spatial sites; L12 classifies the next time site #V

#align(center, grid(columns: (1fr, 18mm, 1fr), gutter: 9pt, align: horizon,
  card([PUBLIC L11 · SEGMENTATION], [one categorical distribution at every spatial location #linebreak() output support $H times W$], color: TEAL),
  [#align(center, text(size: 27pt, fill: MUTED)[$arrow.r$])],
  card([PUBLIC L12 · LANGUAGE], [one categorical distribution at the next ordered location #linebreak() output support $V$], color: BLUE),
))
#pause
#v(9pt)
#result[The classifier and cross-entropy stay. Order, causality, and feeding predictions back are new.]

== Commit before the model reveals anything #Q

#case-strip([opening commitment], [Vocabulary `[., a, e, i, n, v]`; visible context `.n`; predict the next token.])
#v(8pt)
#context-row((token-box([.]), token-box([n])), target: [?])
#pause
#v(8pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 10pt,
  card([OUTPUT], [What six numbers must the model return?], color: BLUE),
  card([CHOICE], [Will greedy and sampling always pick the same token?], color: ACC),
  card([LEARNING], [Which embedding rows can this example update?], color: TEAL),
))
#pause
#note([Commit silently. We will derive every answer from one set of weights and revisit all three.], color: BLUE)

== Outline #V

#align(center, grid(columns: (66mm, 1fr), gutter: 16pt, align: horizon,
  [#align(center, text(size: 13pt, weight: 700, fill: MUTED)[ONE CAUSAL EXAMPLE])
   #v(7pt)
   #context-row((token-box([.]), token-box([n])), target: [a])
   #v(10pt)
   #align(center, text(size: 23pt, fill: MUTED)[$arrow.b$])
   #v(6pt)
   #set text(size: 12.5pt)
   #align(center, grid(columns: (1fr, 8mm, 1fr), gutter: 4pt, align: horizon,
     card([CONTEXT], [two learned vectors], color: TEAL),
     [#text(size: 20pt, fill: MUTED)[$arrow.r$]],
     card([OUTPUT], [six next-token probabilities], color: BLUE),
   ))],
  [#grid(columns: (1fr, 1fr), gutter: 9pt, row-gutter: 9pt,
    card([01 · TOKENS], [Add boundaries and extract causal context-target windows.], color: TEAL),
    card([02 · EMBEDDINGS], [Look up and concatenate learned token vectors.], color: ACC),
    card([03 · PREDICT], [Map context through an MLP to vocabulary logits.], color: BLUE),
    card([04 · LEARN], [Use stable softmax, cross-entropy, and $p-y$.], color: RED),
    card([05 · GENERATE], [Choose, append, and repeat under a temperature.], color: GREEN),
    card([06 · EVALUATE], [Track held-out NLL/PPL and expose suffix blindness.], color: INK),
  )],
))

== Three clocks reuse the distribution but change the question #V

#clock-strip("train")
#pause
#clock-strip("infer")
#pause
#clock-strip("eval")
#pause
#v(6pt)
#result[Always ask: are we updating weights, choosing a token, or scoring held-out truth?]

== By the end, one sentence should be operational #V

#align(center, text(size: 22pt, weight: 700)[A language model maps an observed prefix to a distribution over the next token.])
#pause
#v(12pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 10pt,
  card([REPRESENT], [turn categorical ids into learned vectors without inventing numeric order], color: TEAL),
  card([PREDICT], [produce $V$ logits, normalize, and train with token cross-entropy], color: BLUE),
  card([DIAGNOSE], [separate decoding choices from an architecture that cannot see enough history], color: INK),
))
#lm-legend

== One tiny corpus supplies every visible token #V

#align(center, grid(columns: (1fr, 1fr), gutter: 12pt,
  card([TRAIN NAMES], [`naveen` · `navin` · `neev` · `neena` · `vani` #linebreak() `naina` · `avni` · `anvi` · `veena` · `navi`], color: TEAL),
  card([HELD-OUT NAMES], [`navina` · `naveena` · `veenavi` · `anvina`], color: INK),
))
#v(8pt)
#note([Toy convention: `.` is one shared boundary symbol. Repeat it $k=2$ times as left padding; use one `.` target to end a name. It is not punctuation.], color: ACC)

== Tokenization chooses the units the model must predict #V

#align(center, table(
  columns: (31mm, 55mm, 42mm, 45mm), stroke: 0.55pt + MUTED,
  inset: (x: 7pt, y: 6pt), align: (left, left, center, center),
  table.header([*unit*], [*example*], [*vocabulary*], [*sequence*]),
  [character], [#text(size: 13pt)[`learning` #linebreak() $arrow.r$ `l e a r n i n g`]], [small], [long],
  [word], [#text(size: 13pt)[`deep learning` #linebreak() $arrow.r$ `deep | learning`]], [very large], [short],
  [subword], [#text(size: 13pt)[`unpredictable` #linebreak() $arrow.r$ `un | predict | able`]], [moderate], [moderate],
))
#v(5pt)
#set text(size: 13.5pt)
#note([Characters keep every row visible. Production usually separates `<BOS>`, `<EOS>`, `<PAD>`, and `<UNK>`; this toy merges `<BOS>` and `<EOS>` as `.`.], color: TEAL)

== The vocabulary fixes ids and output coordinates #V

#align(center, table(
  columns: 6, stroke: 0.7pt + MUTED, inset: (x: 16pt, y: 8pt), align: center,
  table.header([`.`], [`a`], [`e`], [`i`], [`n`], [`v`]),
  [$0$], [$1$], [$2$], [$3$], [$4$], [$5$],
))
#pause
#v(10pt)
#align(center, text(size: 19pt)[$V=6$ means six logits, six probabilities, and six possible next tokens.])
#case-strip([vocabulary], [The target `a` is coordinate $1$; context `.n` reads ids $(0,4)$.])

== Sliding a two-character window creates seven examples #V

#align(center, table(
  columns: 7, stroke: 0.55pt + MUTED, inset: (x: 9pt, y: 6pt), align: center,
  table.header([`..`], [`.n`], [`na`], [`av`], [`ve`], [`ee`], [`en`]),
  [#text(fill: ACC)[`n`]], [#text(fill: ACC)[`a`]], [#text(fill: ACC)[`v`]], [#text(fill: ACC)[`e`]], [#text(fill: ACC)[`e`]], [#text(fill: ACC)[`n`]], [#text(fill: ACC)[`.`]],
))
#pause
#v(8pt)
#align(center, text(size: 16pt, fill: MUTED)[top = visible context; bottom = next-token target])
#case-strip([training pairs], [`.n` $arrow.r$ `a` is the second example and remains our hand calculation.])

== Causal training shifts the same sequence by one site #V

#clock-strip("train")
#v(6pt)
#align(center, table(
  columns: (24mm, 16mm, 16mm, 16mm, 16mm, 16mm, 16mm, 16mm),
  stroke: 0.5pt + MUTED, inset: (x: 5pt, y: 6pt), align: center,
  [*input*], [`..`], [`.n`], [`na`], [`av`], [`ve`], [`ee`], [`en`],
  [*target*], [`n`], [`a`], [`v`], [`e`], [`e`], [`n`], [`.`],
))
#pause
#v(8pt)
#result[At position $t$, the target token is not part of the input context.]

== The chain rule turns sequence probability into next-token factors #D

Let $B=(.,.)$ denote the fixed left boundary padding and let $x_7=.$ be the end target.
$
p(x_(1:7) | B) = product_(t=1)^7 p(x_t | B, x_(<t)).
$
#pause
#v(8pt)
#result[This is an exact identity. The fixed-window approximation comes next.]

== One ledger will cross probability, loss, and evaluation #V

#case-strip([sequence ledger], [Correct targets for `naveen.` under contexts `..`, `.n`, `na`, `av`, `ve`, `ee`, `en`.])
#v(7pt)
#align(center, table(
  columns: (29mm, 23mm, 23mm, 23mm, 23mm, 23mm, 23mm, 23mm),
  stroke: 0.55pt + MUTED, inset: (x: 5pt, y: 5pt), align: center,
  [*context*], [`..`], [`.n`], [`na`], [`av`], [`ve`], [`ee`], [`en`],
  [*target*], [`n`], [`a`], [`v`], [`e`], [`e`], [`n`], [`.`],
  [*$p$(target)*], [$.30$], [*derive*], [$.70$], [$.60$], [$.50$], [$.40$], [$.80$],
))
#pause
#note([Six probabilities are held illustrative. The orange `.n` $arrow.r$ `a` entry will come from the exact weights.], color: ACC)

== Checkpoint: what did the chain rule assume? #Q

#align(center, grid(columns: (1fr, 1fr), gutter: 10pt,
  card([A], [tokens are independent], color: INK),
  card([B], [the sequence has fixed length], color: INK),
  card([C], [nothing extra; it is an identity], color: INK),
  card([D], [future tokens are visible], color: INK),
))
#pause
#v(9pt)
#result[*C.* Each exact factor conditions on the complete preceding history and boundary.]
#note([The approximation begins only when the model discards all but the latest $k$ tokens.], color: BLUE)

== A fixed-context MLP deliberately truncates the history #D

$
p_theta(x_t | B,x_(<t)) approx p_theta(x_t | x_(t-2:t-1)).
$
#pause
#v(9pt)
#align(center, grid(columns: (1fr, 18mm, 1fr), gutter: 8pt, align: horizon,
  card([EXACT CONDITIONAL], [boundary plus every earlier token #linebreak() variable-length input], color: INK),
  [#align(center, text(size: 24pt, fill: MUTED)[$arrow.r$])],
  card([TOY MODEL], [only two latest tokens #linebreak() fixed-size input for an MLP], color: BLUE),
))
#result[This model learns local spelling patterns; it cannot recover a token that has scrolled out.]

== A token id is a label, not a magnitude

`n` has id $4$ and `v` has id $5$, but

$ 5-4=1 quad #text(fill: RED)[does not mean] quad v-n=1. $
#pause
#v(10pt)
#alertbox[Feeding raw ids as scalar features invents ordering and distance. Ids are categorical addresses.]
The representation must preserve identity without pretending that adjacent ids are similar.

== One-hot vectors give each token an independent axis #D

With vocabulary order `[., a, e, i, n, v]`:

$
o_. = mat(1,0,0,0,0,0)^T,
quad
o_n = mat(0,0,0,0,1,0)^T.
$
#pause
For different tokens $i != j$, $norm(o_i-o_j)^2=2$.
#note([One-hot is faithful but sparse, $V$-dimensional, and has no learned similarity structure.], color: MUTED)

== An embedding is a learned row selected by an id #V

#align(center, table(
  columns: (24mm, 27mm, 27mm), stroke: 0.6pt + MUTED,
  inset: (x: 10pt, y: 5pt), align: center,
  table.header([*token*], [*feature 1*], [*feature 2*]),
  [`.`], [$1$], [$0$],
  [`a`], [$0$], [$1$],
  [`e`], [$1$], [$1$],
  [`i`], [$-1$], [$1$],
  [`n`], [$0$], [$2$],
  [`v`], [$2$], [$0$],
))
#pause
#v(5pt)
$ E in RR^(6 times 2), quad e_i=E[i]=E^T o_i. $
#result[The id addresses a row; gradient descent decides what the two coordinates become.]

== Context `.n` reads exactly two rows #D

#case-strip([lookup], [ids $(0,4)$ select rows for `.` and `n`.])
#v(8pt)
$
e_.=E[0]=(1,0),
quad
e_n=E[4]=(0,2).
$
#pause
#context-row((token-box([.]), token-box([n])))
#note([The target `a` affects the output loss, but its embedding row is not read on this example.], color: ACC)

== Commit: concatenate or average the two rows? #Q

#align(center, grid(columns: (1fr, 1fr), gutter: 14pt,
  card([A · AVERAGE], [$c=(e_.+e_n)/2=(.5,1)$], color: INK),
  card([B · CONCATENATE], [$c=[e_.;e_n]=(1,0,0,2)$], color: INK),
))
#pause
#v(9pt)
#result[*B.* Concatenation gives the first and second context positions different coordinate blocks.]
#pause
$c_(".n")=(1,0,0,2) != (0,2,1,0)=c_("n.").$

== One position is an ordinary six-class classifier #V

#model-pipeline
#pause
#v(8pt)
#align(center, text(size: 14.5pt)[two ids #h(6pt) → #h(6pt) two 2-D rows #h(6pt) → #h(6pt) ordered 4-D context #h(6pt) → #h(6pt) two hidden units #h(6pt) → #h(6pt) six logits])
#result[Language modeling changes how examples are constructed; the local head is familiar classification.]

== The tensor ledger follows the pipeline #D

#clock-strip("train")
#v(5pt)
$
c=[E[x_(t-2)];E[x_(t-1)]] in RR^4,
$
#pause
$
u=W_h c+b_h, quad h="ReLU"(u) in RR^2,
$
#pause
$
z=W_o h+b_o in RR^6, quad p="softmax"(z).
$
#align(center, text(size: 15pt, fill: MUTED)[$W_h in RR^(2 times 4)$; $W_o in RR^(6 times 2)$; rows of $W_o$ follow vocabulary order.])

== Count parameters once, then keep the shapes visible #D

#align(center, table(
  columns: (42mm, 53mm, 45mm), stroke: 0.55pt + MUTED,
  inset: (x: 8pt, y: 5pt), align: (left, center, right),
  table.header([*block*], [*shape*], [*count*]),
  [embedding $E$], [$V times d$], [$V d=12$],
  [hidden $W_h,b_h$], [$H times k d$, $H$], [$H k d+H=10$],
  [output $W_o,b_o$], [$V times H$, $V$], [$V H+V=18$],
))
#pause
#v(7pt)
$ 12+10+18=40 " parameters". $
#note([Increasing context length $k$ grows the hidden block $H k d$ even before we discuss what the model can remember.], color: INK)

== Every weight for `.n` $arrow.r$ `a` is now explicit #D

#case-strip([exact model], [input $x=(.,n)$; target $y=a$; $c=(1,0,0,2)^T$.])
#v(6pt)
$
W_h=mat(0.5,0,0,0.25; 0,1,1,0),
quad b_h=mat(0;-1),
$
#pause
$
W_o=mat(-1,0;2,1;0,-1;-1,1;0,0;1,-1),
quad b_o=0.
$
#note([Rows of $W_o$ are `[., a, e, i, n, v]`. Keep that order for every vector below.], color: BLUE)

== Forward 1: lookup and concatenate #D

#clock-strip("train")
#v(5pt)
#model-pipeline
#pause
$
E[0]=(1,0), quad E[4]=(0,2),
$
#pause
$
c=[E[0];E[4]]=(1,0,0,2)^T.
$
#case-strip([forward · context], [The ordered context occupies four coordinates; no averaging occurs.])

== Forward 2: one hidden unit turns on #D

$
u_1=.5(1)+0(0)+0(0)+.25(2)+0=1,
$
#pause
$
u_2=0(1)+1(0)+1(0)+0(2)-1=-1.
$
#pause
$
h=(max(0,1),max(0,-1))=(1,0).
$
#align(center, grid(columns: (1fr, 1fr), gutter: 12pt,
  card([ACTIVE], [$h_1=1$; gradient may pass], color: BLUE),
  card([OFF], [$h_2=0$; ReLU blocks its gradient here], color: GREEN),
))

== Commit: compute two logits before seeing all six #Q

With $h=(1,0)$, calculate the rows for target `a` and alternative `v`:

#v(8pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 14pt,
  card([TARGET `a`], [$z_a=(2,1) dot (1,0)=?$], color: ACC),
  card([ALTERNATIVE `v`], [$z_v=(1,-1) dot (1,0)=?$], color: BLUE),
))
#pause
#v(10pt)
#align(center, text(size: 23pt, weight: 700)[$z_a=2$ #h(25pt) $z_v=1$])
#pause
#result[A logit is an unconstrained score, not yet a probability.]

== Forward 3: every output row supplies one logit #D

#case-strip([forward · logits], [vocabulary order `[., a, e, i, n, v]`.])
#v(7pt)
$
z=W_o h=(-1,2,0,-1,0,1)^T.
$
#pause
#align(center, table(
  columns: 6, stroke: 0.55pt + MUTED, inset: (x: 12pt, y: 6pt), align: center,
  table.header([`.`], [`a`], [`e`], [`i`], [`n`], [`v`]),
  [$-1$], [#text(fill: ACC, weight: 700)[$2$]], [$0$], [$-1$], [$0$], [$1$],
))
#note([Softmax will preserve this ranking while converting scores into a normalized distribution.], color: BLUE)

== Stable softmax subtracts a constant before exponentiating #D

Naively exponentiating large logits can overflow. Softmax is unchanged by a shared shift:

#pause
$
"softmax"(z)="softmax"(z-m),
quad m=max_j z_j=2.
$
#pause
$
z-m=(-3,0,-2,-3,-2,-1).
$
#result[Subtracting the maximum is an implementation safeguard, not a model change.]

== Forward 4: exponentiate and normalize #D

$
exp(z-m)=(.0498,1,.1353,.0498,.1353,.3679),
$
#pause
$
Z=sum_j exp(z_j-m)=1.7381,
$
#pause
$
p=frac(exp(z-m),Z)
=(.028644,.575333,.077863,.028644,.077863,.211653).
$
#case-strip([forward · probability], [The exact orange ledger entry is now $p(a|.n)=.575333$.])

== The six bars share a true 0-to-1 probability axis #V

#clock-strip("train")
#v(4pt)
#probability-bars()
#pause
#v(5pt)
#align(center, text(size: 16pt, fill: MUTED)[Direct labels follow `[., a, e, i, n, v]`; all six probabilities sum to $1$.])
#result[`a` is most likely, but alternatives retain nonzero probability.]

== Forward 5: cross-entropy reads the target coordinate #D

#clock-strip("train")
#v(5pt)
The target is `a`, so $y=(0,1,0,0,0,0)$.
#pause
$
cal(L)=-sum_j y_j log p_j=-log(.575333)=.552806 " nats".
$
#align(center, grid(columns: (1fr, 1fr), gutter: 12pt,
  card([IF $p_a$ RISES], [the loss falls], color: GREEN),
  card([IF $p_a arrow.r 1$], [$cal(L) arrow.r 0$], color: ACC),
))

== The sequence ledger now contains no second version #V

#align(center, table(
  columns: (29mm, 23mm, 23mm, 23mm, 23mm, 23mm, 23mm, 23mm),
  stroke: 0.55pt + MUTED, inset: (x: 5pt, y: 5pt), align: center,
  [*context*], [`..`], [`.n`], [`na`], [`av`], [`ve`], [`ee`], [`en`],
  [*target*], [`n`], [`a`], [`v`], [`e`], [`e`], [`n`], [`.`],
  [*$p$(target)*], [$.30$], [#text(fill: ACC, weight: 700)[$.575333$]], [$.70$], [$.60$], [$.50$], [$.40$], [$.80$],
))
#pause
#v(7pt)
$
p_theta("naveen."|"..") approx .30(.575333)(.70)(.60)(.50)(.40)(.80)=.0115987.
$
#result[The sequence probability is small because all seven next-token decisions must agree.]

== Logs turn the same product into NLL and perplexity #D

#clock-strip("eval")
#v(4pt)
$
-log(.0115987)=4.45686 " nats",
$
#pause
$
"mean NLL"=4.45686/7=.636694 " nats/token",
$
#pause
$
"PPL"=exp(.636694)=1.89022.
$
#note([This is one illustrative name ledger, not the trained notebook's validation score.], color: INK)

== Commit: what sign must the target-logit gradient have? #Q

#align(center, grid(columns: (1fr, 1fr), gutter: 14pt,
  card([A · POSITIVE], [gradient descent will lower $z_a$], color: INK),
  card([B · NEGATIVE], [gradient descent will raise $z_a$], color: INK),
))
#pause
#v(10pt)
#result[*B.* To reduce $-log p_a$, the update must increase the target logit relative to alternatives.]
#pause
#note([Softmax plus cross-entropy makes the exact sign visible in one vector.], color: RED)

== Backward 1: softmax plus CE gives $p-y$ #D

#clock-strip("train")
#v(4pt)
$
nabla_z cal(L)=p-y
$
#pause
$
=(.028644,-.424667,.077863,.028644,.077863,.211653).
$
- the target coordinate is negative, so gradient descent raises $z_a$;
- every alternative coordinate is positive, so gradient descent lowers it.

== Backward 2: output gradients are outer products #D

Because $z=W_o h+b_o$,

$
nabla_(W_o)cal(L)=(p-y)h^T,
quad
nabla_(b_o)cal(L)=p-y.
$
#pause
With $h=(1,0)$, the first column equals $p-y$ and the second column is zero.
#result[An inactive feature emits no output-weight gradient on this example.]

== Backward 3: ReLU chooses which hidden signal survives #D

$
nabla_h cal(L)=W_o^T(p-y)=(-.694969,-.685539)^T.
$
#pause
The forward pre-activation was $u=(1,-1)$, so the ReLU mask is $(1,0)$.
#pause
$
nabla_u cal(L)=nabla_h cal(L) dot.o (1,0)=(-.694969,0)^T.
$
#note([The second unit had a nonzero upstream signal, but its negative pre-activation blocks that signal.], color: GREEN)

== Backward 4: return to the ordered context #D

Since $u=W_h c+b_h$,

#pause
$
nabla_c cal(L)=W_h^T nabla_u cal(L)
=(-.347485,0,0,-.173742)^T.
$
#pause
#align(center, table(
  columns: (32mm, 28mm, 28mm, 28mm, 28mm), stroke: 0.55pt + MUTED,
  inset: (x: 7pt, y: 6pt), align: center,
  [*block*], [$e_.[1]$], [$e_.[2]$], [$e_n[1]$], [$e_n[2]$],
  [*$nabla_c$*], [$-.347485$], [$0$], [$0$], [$-.173742$],
))
#result[Concatenation makes the split back into two embedding rows unambiguous.]

== Backward 5: split, scatter, and update only rows read #D

$
nabla_(e_.)cal(L)=(-.347485,0),
quad
nabla_(e_n)cal(L)=(0,-.173742).
$
#pause
Scatter these into rows $0$ and $4$ of $E$; every other row receives zero from this lookup.
#pause
With $eta=.1$:
$
e_. <- (1.03475,0),
quad
e_n <- (0,2.01737).
$
#case-strip([opening · learning], [The rows for `.` and `n` move. Target row `a` was not read and does not move through the lookup.])

== One scalar loss teaches every used block #V

#align(center, diagram(spacing: (19mm, 11mm), node-stroke: 1pt + INK, node-fill: white, {
  flow-node((0,0), [rows `.` and `n`], color: TEAL, fill: PALE-TEAL, w: 27mm)
  flow-node((1.5,0), [$W_h,b_h$], color: BLUE, fill: PALE-BLUE, w: 25mm)
  flow-node((3,0), [$W_o,b_o$], color: BLUE, fill: PALE-BLUE, w: 25mm)
  flow-node((4.5,0), [token CE], color: RED, fill: PALE-RED, w: 24mm)
  flow-arrow((0,0),(1.5,0),color:TEAL)
  flow-arrow((1.5,0),(3,0),color:BLUE)
  flow-arrow((3,0),(4.5,0),color:RED)
}))
#pause
#v(8pt)
- output rows learn which hidden features support each token;
- hidden weights learn combinations of ordered context coordinates;
- read embedding rows learn features that improve future predictions.
#result[The path is ordinary backpropagation; language structure entered when we built context-target pairs.]

== TRAIN repeats the calculation over a declared batch #V

#clock-strip("train")
#v(4pt)
$
cal(L)_B=-1/abs(B) sum_(i in B) log p_theta(y_i|c_i).
$
#pause
#align(center, diagram(spacing: (17mm, 9mm), node-stroke: 1pt + INK, node-fill: white, {
  flow-node((0,0), [true contexts], color: TEAL, fill: PALE-TEAL, w: 25mm)
  flow-node((1.3,0), [MLP], color: BLUE, fill: PALE-BLUE, w: 20mm)
  flow-node((2.6,0), [$V$ logits], color: BLUE, fill: PALE-BLUE, w: 22mm)
  flow-node((3.9,0), [token CE], color: RED, fill: PALE-RED, w: 22mm)
  flow-node((5.2,0), [gradients], color: GREEN, fill: PALE-GREEN, w: 23mm)
  flow-node((6.5,0), [update], color: INK, fill: CREAM, w: 21mm)
  flow-arrow((0,0),(1.3,0)); flow-arrow((1.3,0),(2.6,0)); flow-arrow((2.6,0),(3.9,0));
  flow-arrow((3.9,0),(5.2,0)); flow-arrow((5.2,0),(6.5,0));
}))
#note([The notebook uses the entire training set as $B$ on each Adam step: full-batch training, not a minibatch claim.], color: TEAL)

== Split whole names before creating windows #V

#clock-strip("eval")
#v(5pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 14pt,
  card([TRAIN], [10 whole names $arrow.r$ 56 context-target pairs #linebreak() weights receive updates], color: TEAL),
  card([VALIDATION], [4 different whole names $arrow.r$ 30 pairs #linebreak() no updates], color: INK),
))
#pause
#v(8pt)
#alertbox[Do not randomly split windows from the same name. Nearby windows share almost all their evidence.]
#result[The split unit must match the object whose generalization we claim: here, a whole name.]

== EVAL declares tokenizer, targets, and reduction #D

#clock-strip("eval")
#v(4pt)
$
"NLL"=-1/N sum_(i=1)^N log p_theta(y_i|c_i),
quad
"PPL"=exp("NLL").
$
#pause
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 10pt,
  card([TOKENIZER], [same character vocabulary and boundary policy], color: INK),
  card([TARGETS], [$N$ non-padding next-token targets], color: INK),
  card([DATA], [same held-out corpus and token-weighted reduction], color: INK),
))
#result[Perplexities are comparable only when all three contracts agree.]

== The executed notebook shows the full learning trajectory #V

#clock-strip("eval")
#v(2pt)
#align(center, grid(columns: (1.6fr, 1fr), gutter: 12pt, align: top,
  [
    #align(center, [
      #set text(size: 9.4pt)
      #lq.diagram(
        width: 137mm, height: 41mm,
        xlabel: [full-batch Adam update], ylabel: [token NLL],
        xlim: (1, 300), ylim: (0.58, 1.84),
        xaxis: (ticks: (1, 50, 100, 150, 200, 250, 300), subticks: none, mirror: false),
        yaxis: (ticks: (0.6, 0.9, 1.2, 1.5, 1.8), subticks: none, mirror: false),
        legend: none,
        grid: (stroke: 0.35pt + MUTED.lighten(60%)),
        lq.plot(nll-updates, nll-train, stroke: TEAL + 1.8pt, mark: none),
        lq.plot(nll-updates, nll-heldout, stroke: ACC + 1.8pt, mark: none),
        lq.plot(nll-updates, nll-uniform,
          stroke: (paint: MUTED, dash: "dashed", thickness: 1.15pt), mark: none),
      )
    ])
    #v(-3pt)
    #align(center, text(size: 9.4pt)[
      #curve-key(TEAL, [train · 56 targets]) #h(9pt)
      #curve-key(ACC, [held-out · 30 targets]) #h(9pt)
      #curve-key(MUTED, [uniform baseline 1.792], dashed: true)
    ])
  ],
  [
    #card([EXECUTED · SEED 11], [
      #text(size: 12.2pt)[300 full-batch Adam updates #linebreak()
      constructed 10+4-name course corpus]
    ], color: BLUE)
    #v(4pt)
    #card([COMPUTED · UPDATE 300], [
      #text(size: 11.8pt)[*train* · NLL 0.645308 · PPL 1.90657 #linebreak()
      *held-out* · NLL 0.677965 · PPL 1.96986]
    ], color: INK)
  ],
))
#pause
#v(2pt)
#note([Every point evaluates the same post-update parameter snapshot; no smoothing. The train set has 56 targets and the four held-out names have 30 targets.], color: BLUE)
#result[Optimization + in-protocol transfer on a constructed course corpus; not broad name quality.]

== Uniform prediction is the first evaluation sanity check #D

#clock-strip("eval")
#v(5pt)
For $V=6$, a uniform model has $p(y)=1/6$:

#pause
$
"NLL"=-log(1/6)=log 6=1.79176,
quad
"PPL"=6.
$
#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 14pt,
  card([UNIFORM], [$1.79176$ NLL #linebreak() $6.00$ PPL], color: MUTED),
  card([VALIDATION], [$.677965$ NLL #linebreak() $1.96986$ PPL], color: INK),
))
#pause
#result[The trained model predicts held-out next characters far better than uniform guessing under this protocol.]

// ───────────────────────── SHOULD · 55-70 MIN ─────────────────────────

== The same $p$ now crosses from TRAIN to INFER #V

#clock-strip("train")
#pause
#clock-strip("infer")
#pause
#v(7pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 14pt,
  card([TRAIN], [true context $arrow.r$ distribution $arrow.r$ compare with true target $arrow.r$ update], color: TEAL),
  card([INFER], [current context $arrow.r$ distribution $arrow.r$ choose token $arrow.r$ append], color: BLUE),
))
#pause
#result[Weights stay fixed during generation; the chosen token changes the next input context.]

== Autoregressive generation closes a feedback loop #V

#align(center, diagram(spacing: (20mm, 11mm), node-stroke: 1pt + INK, node-fill: white, {
  flow-node((0,0), [current context], color: TEAL, fill: PALE-TEAL, w: 29mm)
  flow-node((1.5,0), [same MLP], color: BLUE, fill: PALE-BLUE, w: 24mm)
  flow-node((3,0), [$p$ over 6 tokens], color: BLUE, fill: PALE-BLUE, w: 29mm)
  flow-node((4.5,0), [choose token], color: ACC, fill: PALE-ACC, w: 26mm)
  flow-node((6,0), [append; shift], color: GREEN, fill: PALE-GREEN, w: 27mm)
  flow-arrow((0,0),(1.5,0)); flow-arrow((1.5,0),(3,0)); flow-arrow((3,0),(4.5,0)); flow-arrow((4.5,0),(6,0));
  flow-arrow((6,0),(0,0),color:GREEN,bend:-34deg,label:[repeat])
}))
#pause
#v(8pt)
Start from `..`; stop when the model emits `.` or reaches a declared maximum length.
#pause
#result[Autoregressive means the model conditions on tokens already present, including its own earlier choices.]

== Commit: must greedy and sampling choose the same token? #Q

#probability-bars()
#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 14pt,
  card([GREEDY], [$hat(x)=arg max_v p(v|.n)$], color: BLUE),
  card([SAMPLING], [$hat(x) tilde "Categorical"(p)$], color: ACC),
))
#pause
#v(7pt)
#result[No. Greedy always chooses `a`; sampling can choose any token with nonzero mass.]

== One uniform draw makes categorical sampling concrete #D

#clock-strip("infer")
#v(4pt)
#align(center, table(
  columns: (31mm, 22mm, 22mm, 22mm, 22mm, 22mm, 22mm),
  stroke: 0.55pt + MUTED, inset: (x: 5pt, y: 5pt), align: center,
  [*token*], [`.`], [`a`], [`e`], [`i`], [`n`], [`v`],
  [*$p$*], [$.029$], [$.575$], [$.078$], [$.029$], [$.078$], [$.212$],
  [*CDF*], [$.029$], [$.604$], [$.682$], [$.710$], [$.788$], [$1.000$],
))
#pause
Draw $u=.650$ from $"Uniform"(0,1)$.
#pause
$ .604 < .650 <= .682 quad => quad #text(fill: ACC, weight: 700)[sample `e`]. $
#pause
#result[The model supplies probabilities; the decoder supplies the decision rule and randomness.]

== Temperature rescales logits before the same softmax #D

#clock-strip("infer")
#v(5pt)
$
p_i(tau)="softmax"(z/tau)_i, quad tau>0.
$
#pause
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 10pt,
  hairline([$tau=.5$], [logit gaps double #linebreak() distribution concentrates], color: ACC),
  hairline([$tau=1$], [learned distribution #linebreak() no rescaling], color: BLUE),
  hairline([$tau=2$], [logit gaps halve #linebreak() distribution flattens], color: TEAL),
))
#pause
#note([Temperature changes neither weights nor token ranking for positive $tau$.], color: BLUE)

== The same six logits produce three exact distributions #V

#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 10pt,
  tiny-distribution([$tau=.5$], prob-cold, color: ACC),
  tiny-distribution([$tau=1$], prob-base, color: BLUE),
  tiny-distribution([$tau=2$], prob-hot, color: TEAL),
))
#pause
#v(6pt)
#set text(size: 12.5pt)
#align(center, table(
  columns: (24mm, 24mm, 24mm, 24mm, 24mm, 24mm, 24mm),
  stroke: 0.45pt + MUTED, inset: (x: 4pt, y: 3pt), align: center,
  [*$tau$*], [`.`], [`a`], [`e`], [`i`], [`n`], [`v`],
  [$.5$], [$.002$], [$.850$], [$.016$], [$.002$], [$.016$], [$.115$],
  [$1$], [$.029$], [$.575$], [$.078$], [$.029$], [$.078$], [$.212$],
  [$2$], [$.080$], [$.359$], [$.132$], [$.080$], [$.132$], [$.218$],
))
#pause
#result[Concentration is measurable; "safer" or "more correct" is not guaranteed.]

== Choose temperature by behavior, not a moral label #Q

#align(center, grid(columns: (1fr, 1fr), gutter: 14pt,
  card([SYMPTOM], [samples collapse to the same spelling], color: INK),
  card([TEST], [hold logits and seed fixed; compare token frequencies across $tau$], color: BLUE),
))
#pause
#v(9pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 14pt,
  card([LOWER $tau$], [more mass on current top tokens; less variation], color: ACC),
  card([HIGHER $tau$], [more mass on alternatives; more variation], color: TEAL),
))
#pause
#note([If the model ranks a bad token highest, lowering temperature can make that error more consistent.], color: RED)

== The trained MLP is blind to prefixes with the same suffix #V

#clock-strip("infer")
#v(4pt)
#align(center, grid(columns: (1fr, 18mm, 1fr), gutter: 8pt, align: horizon,
  card([PREFIX A], [`na` $arrow.r$ visible context `na`], color: TEAL),
  [#align(center, text(size: 25pt, fill: MUTED)[$equiv$])],
  card([PREFIX B], [`veena` $arrow.r$ visible context `na`], color: TEAL),
))
#pause
#v(7pt)
#set text(size: 12.5pt)
$
p=(.418671,
9.91 times 10^(-7),
3.77 times 10^(-6),
.142728,
3.17 times 10^(-5),
.438565)
$
#pause
#result[The two distributions are exactly identical because the model receives exactly the same two ids.]

== Diagnose the architecture before tuning the optimizer #V

#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 11pt,
  hairline([SYMPTOM], [different long prefixes produce identical next-token distributions], color: RED),
  hairline([SUSPECT], [the useful distinction lies outside the fixed window], color: BLUE),
  hairline([TEST], [hold the suffix fixed, change only earlier tokens, compare $p$ exactly], color: GREEN),
))
#pause
#v(9pt)
#result[If $p$ cannot change, more training cannot make this architecture use the hidden prefix.]
#pause
#note([This is an information-access failure, not evidence that Adam, embeddings, or temperature are broken.], color: INK)

== A larger window delays the failure but does not remove it #D

$ W_h in RR^(H times k d) quad => quad #text(fill: ACC)[$H k d$ weights]. $
#pause
- doubling $k$ doubles this block;
- the same token in slot 1 and slot 2 meets different columns;
- the maximum usable history remains fixed at model construction.
#pause
#v(7pt)
#alertbox[Concatenation preserves order, but the MLP does not reuse one position-independent update across time.]

== Revisit: every opening commitment now has evidence #Q

#case-strip([opening revisit], [context `.n`, exact six-token model.])
#v(6pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 10pt,
  card([OUTPUT], [#text(size: 11.5pt)[$p=(.029,.575,.078,$ #linebreak() $.029,.078,.212)$]], color: BLUE),
  card([CHOICE], [greedy $arrow.r$ `a`; sample with $u=.65$ $arrow.r$ `e`], color: ACC),
  card([LEARNING], [lookup updates rows `.` and `n`], color: TEAL),
))
#pause
#v(7pt)
#result[One distribution supports training, decoding, and evaluation only after the clock and policy are declared.]

// ───────────────────────── OPTIONAL · 70-80 MIN ─────────────────────────

== Expose the full output-gradient matrix #D

With $h=(1,0)$,

$
nabla_(W_o)cal(L)=mat(
.028644,0;
-.424667,0;
.077863,0;
.028644,0;
.077863,0;
.211653,0
).
$
#pause
#v(7pt)
#note([Each vocabulary row receives its own logit error; the inactive hidden feature leaves the second column zero.], color: MUTED)

== Use the notebook as a prediction protocol #I

#interbox(link-to: NB)[
  Before running: predict the seven windows, active ReLU unit, sign of the target gradient, two embedding rows that move, and whether `na` and `veena` can differ.
]
#pause
#v(8pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 12pt,
  card([VERIFY], [exact forward/backward arrays; train/validation NLL; generated names], color: TEAL),
  card([EXPLAIN], [why temperature changes diversity and why a shared suffix forces identical $p$], color: BLUE),
))
#pause
#result[Prediction before execution turns the notebook from a demo into evidence.]

== Provenance and extension #I

#align(center, grid(columns: (1fr, 1fr), gutter: 14pt,
  card([PRIMARY SOURCE], [#link(BENGIO)[Bengio et al. (2003)] #linebreak() distributed representations plus a feedforward neural probabilistic language model], color: INK),
  card([FOLLOW-ALONG], [#link(NB)[Exact ES 667 Colab] #linebreak() deterministic NumPy forward/backward, full-batch training, generation, and suffix test], color: TEAL),
))
#pause
#v(9pt)
#note([The paper scales the central idea; our six-token model keeps its mechanics auditable by hand.], color: MUTED)

== Next: replace a fixed window with a shared recurrent state #V

#align(center, grid(columns: (1fr, 18mm, 1fr), gutter: 8pt, align: horizon,
  card([THIS LECTURE], [$c_t=[e_(t-2);e_(t-1)]$ #linebreak() fixed access; position-specific columns], color: BLUE),
  [#align(center, text(size: 26pt, fill: MUTED)[$arrow.r$])],
  card([PUBLIC L13], [$h_t=f_theta(h_(t-1),e_t)$ #linebreak() shared update; variable-length stream], color: TEAL),
))
#pause
#v(10pt)
#align(center, text(size: 22pt, weight: 700)[The next-token objective stays; the mechanism for carrying context changes.])
#focus-slide[
  A language model estimates a distribution over the *next token*.

  #v(9pt)
  #set text(size: 20pt)
  Next: *RNNs, LSTMs, and GRUs* - learn what recurrent state remembers, forgets, and struggles to train.
]
