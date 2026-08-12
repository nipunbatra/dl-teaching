// Semantic and Instance Segmentation — public Lecture 11 · ES 667 Deep Learning
// Compile from the repository root:
//   typst compile --root . --input handout=true lecture10/L10-segmentation.typ /private/tmp/L10-handout.pdf
//   typst compile --root . lecture10/L10-segmentation.typ /private/tmp/L10-presentation.pdf

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [Semantic and Instance Segmentation],
  subtitle: [Pixel evidence becomes a class field—and, when needed, separate object masks],
)

#let NB = "https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L10/01_dice_vs_ce_toy_masks.ipynb"
#let FCN = "https://arxiv.org/abs/1411.4038"
#let UNET = "https://arxiv.org/abs/1505.04597"
#let VNET = "https://arxiv.org/abs/1606.04797"
#let MASKRCNN = "https://arxiv.org/abs/1703.06870"
#let PANOPTIC = "https://arxiv.org/abs/1801.00868"
#let PETS = "https://www.robots.ox.ac.uk/~vgg/data/pets/"
#let ORCHARD = "figures/orchard-two-touching-trees.png"
#let SEG-EVIDENCE = "/shared/vision-evidence/oxford-iiit-pet/l10/"

#let PALE-TEAL = rgb("#DCEBEB")
#let PALE-BLUE = rgb("#E7F0FA")
#let PALE-ACC = rgb("#FDECD6")
#let PALE-GREEN = rgb("#E5F3E9")
#let PALE-RED = rgb("#FBE8E9")
#let CREAM = rgb("#F4F1EA")
#let ID1 = rgb("#E7E0F4")
#let ID2 = rgb("#F8E0CD")

// Stable grammar: TEAL = truth / TRAIN; BLUE = logits, probabilities, INFER;
// ACC = hard prediction; GREEN = correct overlap; RED = errors; INK = EVAL.
#let swatch(color, body) = box(width: 13pt, height: 4pt, fill: color, radius: 1pt) + h(4pt) + body
#let semantic-legend = align(center, text(size: 12pt)[
  #swatch(TEAL, [truth / TRAIN]) #h(10pt)
  #swatch(BLUE, [logit, probability / INFER]) #h(10pt)
  #swatch(ACC, [hard prediction]) #h(10pt)
  #swatch(GREEN, [overlap / TP]) #h(10pt)
  #swatch(RED, [FP or FN]) #h(10pt)
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

#let caption(body) = align(center, text(size: 13.5pt, fill: MUTED)[#body])

#let evidence-image(name, width: auto, height: auto, fit: "contain") = block(
  fill: white, stroke: 0.8pt + MUTED.lighten(35%), radius: 3pt,
  inset: 2pt, clip: true,
  image(SEG-EVIDENCE + name, width: width, height: height, fit: fit),
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
   #text(size: 11pt, fill: if active { INK } else { MUTED })[#body]],
)

#let clock-strip(active) = align(center, grid(
  columns: (1fr, 1fr, 1fr), gutter: 9pt,
  _clock([TRAIN], active == "train", TEAL, [probability ↔ truth; CE / Dice]),
  _clock([INFER], active == "infer", BLUE, [probability → label / object masks]),
  _clock([EVAL], active == "eval", INK, [hard output ↔ truth; IoU / AP]),
))

#let flow-node(pos, body, color: INK, fill: white, w: 35mm) = node(
  pos,
  block(width: w, inset: 5pt, align(center, text(size: 12.5pt, body))),
  shape: fletcher.shapes.rect, corner-radius: 3pt,
  stroke: 0.9pt + color, fill: fill,
)
#let flow-arrow(a, b, color: MUTED, label: none) = edge(
  a, b, "-|>", stroke: 0.85pt + color,
  label: if label == none { none } else { text(size: 10.5pt, fill: color, label) },
)

#let gt = (
  0, 1, 1, 0,
  0, 1, 1, 0,
  0, 1, 1, 0,
  0, 0, 0, 0,
)
#let pred = (
  0, 1, 1, 0,
  1, 1, 1, 0,
  0, 1, 0, 1,
  0, 1, 0, 0,
)
#let probs = (
  (.10, [.10]), (.90, [.90]), (.90, [.90]), (.10, [.10]),
  (.60, [.60]), (.90, [.90]), (.90, [.90]), (.10, [.10]),
  (.10, [.10]), (.90, [.90]), (.30, [.30]), (.60, [.60]),
  (.10, [.10]), (.60, [.60]), (.05, [.05]), (.05, [.05]),
)
#let ids = (
  0, 1, 2, 0,
  0, 1, 2, 0,
  0, 1, 2, 0,
  0, 0, 0, 0,
)

#let mask-cell(item, mode: "truth") = {
  let value = if mode == "prob" { item.at(0) } else { item }
  let label = if mode == "prob" { item.at(1) } else { str(item) }
  let bg = if mode == "truth" {
    if value == 1 { PALE-TEAL } else { white }
  } else if mode == "hard" {
    if value == 1 { PALE-ACC } else { white }
  } else if mode == "prob" {
    if value >= .5 { PALE-BLUE } else { white }
  } else if mode == "id" {
    if value == 1 { ID1 } else if value == 2 { ID2 } else { white }
  } else { white }
  box(
    width: 13mm, height: 13mm, fill: bg, stroke: 0.65pt + MUTED,
    align(center + horizon, text(size: 11.5pt, weight: 650, fill: INK, label)),
  )
}

#let mask-grid(values, mode: "truth") = align(center, grid(
  columns: 4, gutter: 0pt,
  ..values.map(v => mask-cell(v, mode: mode)),
))

#let mask-triplet(show-p: true, show-hard: true) = align(center, grid(
  columns: (1fr, 18mm, 1fr, 18mm, 1fr), gutter: 4pt, align: horizon,
  [#align(center, text(size: 14pt, weight: 700, fill: TEAL)[truth $G$])
   #v(4pt)#mask-grid(gt)],
  [#align(center, text(size: 24pt, fill: MUTED)[$arrow.r$])],
  [#if show-p {
     align(center, text(size: 14pt, weight: 700, fill: BLUE)[foreground $p$])
     v(4pt)
     mask-grid(probs, mode: "prob")
   }],
  [#if show-hard { align(center, text(size: 20pt, fill: MUTED)[$arrow.r$]) }],
  [#if show-hard {
     align(center, text(size: 14pt, weight: 700, fill: ACC)[hard $P$])
     v(4pt)
     mask-grid(pred, mode: "hard")
   }],
))

#let unet-diagram = align(center, scale(x: 82%, y: 82%, reflow: true, diagram(
  spacing: (13mm, 9mm), node-stroke: 1pt + INK, node-fill: white,
  {
    node((-0.8, 0), box(width: 1pt, height: 1pt), stroke: none)
    node((8.2, 0), box(width: 1pt, height: 1pt), stroke: none)
    node((0, 0), align(center)[$256^2$ \ #text(size: 14pt)[32 ch]], fill: PALE-TEAL, stroke: 1.1pt + TEAL)
    node((1.3, 1), align(center)[$128^2$ \ #text(size: 14pt)[64 ch]], fill: PALE-TEAL, stroke: 1.1pt + TEAL)
    node((2.6, 2), align(center)[$64^2$ \ #text(size: 14pt)[128 ch]], fill: PALE-TEAL, stroke: 1.1pt + TEAL)
    node((3.9, 3), align(center)[$32^2$ \ #text(size: 14pt)[256 ch]], fill: CREAM)
    node((5.2, 2), align(center)[$64^2$ \ #text(size: 14pt)[128 ch]], fill: PALE-ACC, stroke: 1.1pt + ACC)
    node((6.5, 1), align(center)[$128^2$ \ #text(size: 14pt)[64 ch]], fill: PALE-ACC, stroke: 1.1pt + ACC)
    node((7.8, 0), align(center)[$256^2$ \ #text(size: 14pt)[$K$ logits]], fill: PALE-ACC, stroke: 1.1pt + ACC)
    edge((0,0),(1.3,1),"-|>",stroke:1pt+TEAL)
    edge((1.3,1),(2.6,2),"-|>",stroke:1pt+TEAL)
    edge((2.6,2),(3.9,3),"-|>",stroke:1pt+TEAL)
    edge((3.9,3),(5.2,2),"-|>",stroke:1pt+ACC)
    edge((5.2,2),(6.5,1),"-|>",stroke:1pt+ACC)
    edge((6.5,1),(7.8,0),"-|>",stroke:1pt+ACC)
    edge((0,0),(7.8,0),"--|>",stroke:1.2pt+GREEN,label:text(size:11pt,fill:GREEN)[concat])
    edge((1.3,1),(6.5,1),"--|>",stroke:1.2pt+GREEN)
    edge((2.6,2),(5.2,2),"--|>",stroke:1.2pt+GREEN)
  },
)))

#let mask-rcnn-diagram = align(center, scale(x: 84%, y: 84%, reflow: true, diagram(
  spacing: (15mm, 12mm), node-stroke: 1pt + INK, node-fill: white,
  {
    node((-0.8, 0), box(width: 1pt, height: 1pt), stroke: none)
    node((7.1, 0), box(width: 1pt, height: 1pt), stroke: none)
    node((0,0), [image], radius: 9mm)
    node((1.5,0), align(center)[backbone \ #text(size:14pt)[feature map]], fill: PALE-TEAL, stroke:1.1pt+TEAL)
    node((3,0), align(center)[RPN \ #text(size:14pt)[proposals]], fill: CREAM)
    node((4.5,0), [RoIAlign], fill: CREAM)
    node((6,-1), align(center)[class \ $hat(y)$], fill: PALE-BLUE, stroke:1pt+BLUE)
    node((6,0), align(center)[box \ $hat(b)$])
    node((6,1), align(center)[$K$ masks \ $hat(M)$], fill: PALE-ACC, stroke:1.2pt+ACC)
    edge((0,0),(1.5,0),"-|>",stroke:1pt+MUTED)
    edge((1.5,0),(3,0),"-|>",stroke:1pt+MUTED)
    edge((3,0),(4.5,0),"-|>",stroke:1pt+MUTED)
    edge((4.5,0),(6,-1),"-|>",stroke:1pt+BLUE)
    edge((4.5,0),(6,0),"-|>",stroke:1pt+INK)
    edge((4.5,0),(6,1),"-|>",stroke:1.2pt+ACC)
  },
)))

#title-slide()

// ───────────────────────── 1 · BRIDGE + COMMIT ─────────────────────────
= Detection drew rectangles; segmentation must assign pixels

== One real photograph changes the supervision from a box to pixels #V

#align(center, evidence-image("real-trimap-contract.png", width: 154mm, height: 86.6mm))
#pause
#v(3pt)
#note([OBSERVED Oxford pixels + official GT trimap. Value 1 is foreground, 2 background, and 3 boundary/ignore; no model output appears here.], color: BLUE)

== Commit: which output can answer both requirements? #Q

#align(center, image(ORCHARD, width: 86mm, height: 30mm, fit: "cover"))
#v(2pt)
#case-strip([generated concept scene], [Two touching crowns form one region. Report total canopy area *and* count two trees.])
#v(3pt)
#align(center, grid(columns: (1fr, 1fr, 1fr, 1fr), gutter: 8pt,
  card([A], [one global class], color: INK),
  card([B], [one class + one box], color: INK),
  card([C], [one $H times W$ foreground map], color: INK),
  card([D], [a scored set of separate object masks], color: INK),
))
#pause
#v(4pt)
#align(center, text(size: 14pt, fill: BLUE)[Commit silently; the same scene returns after semantic and instance outputs.])

== The 80-minute route protects one causal spine #V

#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 12pt,
  hairline([CORE · 55 MIN], [output contract → pixel logits + CE → spatial decoder → held mask → IoU/Dice → Mask R-CNN → evaluation + revisit], color: TEAL),
  hairline([SHOULD · 15 MIN], [void/empty conventions → upsampling + RoIAlign → panoptic task choice], color: BLUE),
  hairline([OPTIONAL · 10 MIN], [primary provenance → architecture variants → mask fusion / PQ extension], color: MUTED),
))
#pause
#v(10pt)
#align(center, text(size: 16pt, fill: MUTED)[The final one-minute checklist and L12 handoff remain CORE even when optional material is skipped.])

== Three clocks prevent a probability from becoming a metric #V

#clock-strip("train")
#pause
#clock-strip("infer")
#pause
#clock-strip("eval")
#pause
#v(7pt)
#result[TRAIN compares with labels; INFER makes an output; EVAL scores that output under a declared protocol.]

== The exact arithmetic case is constructed—not an orchard annotation #V

#align(center, grid(columns: (1.25fr, .8fr, .8fr), gutter: 12pt, align: horizon,
  [#align(center, text(size: 16pt, weight: 700)[generated concept])
   #v(5pt)#image(ORCHARD, width: 88mm, height: 48mm, fit: "cover")
   #v(4pt)#align(center, text(size: 13pt, fill: MUTED)[not a measured orchard sample])],
  [#align(center, text(size: 16pt, weight: 700, fill: TEAL)[constructed truth $G$])
   #v(5pt)#mask-grid(gt)
   #v(5pt)#align(center, text(size: 14pt)[$abs(G)=6$ pixels])],
  [#align(center, text(size: 16pt, weight: 700)[constructed identities])
   #v(5pt)#mask-grid(ids, mode: "id")
   #v(5pt)#align(center, text(size: 14pt, fill: MUTED)[IDs withheld from $G$.])],
))
#pause
#result[The $4 times 4$ ledger isolates the arithmetic: one union supports area; two identities support counting.]

== Semantic, instance, and panoptic outputs answer different questions #V

#align(center, table(
  columns: (42mm, 70mm, 74mm), stroke: 0.5pt + MUTED, inset: (x: 8pt, y: 7pt),
  table.header([*Task*], [*Output contract*], [*Question answered*]),
  [semantic], [$H times W$ class labels], [Which class owns this pixel?],
  [instance], [variable scored set of object masks], [Which pixels belong to object $j$?],
  [panoptic], [$H times W$ class–identity pairs], [What and which object, everywhere?],
))
#pause
#note([“Stuff” such as sky usually needs a class; countable “things” such as trees also need an identity.], color: ACC)

== One prediction will cross all three clocks #V

#align(center, diagram(spacing: (19mm, 13mm), node-stroke: 1pt + INK, node-fill: white, {
  flow-node((0,0), [logits $a$], color: BLUE, fill: PALE-BLUE)
  flow-node((1.8,0), [probabilities $p$], color: BLUE, fill: PALE-BLUE)
  flow-node((3.6,-.6), [CE + Soft Dice \ TRAIN], color: TEAL, fill: PALE-TEAL, w: 40mm)
  flow-node((3.6,.6), [threshold $tau=.50$ \ INFER], color: BLUE, fill: PALE-BLUE, w: 40mm)
  flow-node((5.6,.6), [hard mask $P$ \ EVAL], color: INK, fill: PALE-ACC)
  flow-arrow((0,0),(1.8,0),color:BLUE)
  flow-arrow((1.8,0),(3.6,-.6),color:TEAL)
  flow-arrow((1.8,0),(3.6,.6),color:BLUE)
  flow-arrow((3.6,.6),(5.6,.6),color:INK)
}))
#pause
#case-strip([promise], [The exact $G$, $p$, threshold, and $P$ will remain fixed through loss, inference, and evaluation.])

// ───────────────────────── 2 · TRAIN: PIXEL CLASSIFIER ─────────────────────────
= TRAIN: classify every spatial location

== A semantic head emits one score vector per pixel #V

#clock-strip("train")
#v(7pt)
#align(center, diagram(spacing: (20mm, 13mm), node-stroke: 1pt + INK, node-fill: white, {
  flow-node((0,0), [feature $bold(f)_(h,w) in RR^C$], w: 43mm)
  flow-node((2.2,0), [$K$ logits \ $bold(z)_(h,w)$], color: BLUE, fill: PALE-BLUE)
  flow-node((4.2,0), [$K$ probabilities \ $bold(p)_(h,w)$], color: BLUE, fill: PALE-BLUE)
  flow-arrow((0,0),(2.2,0),color:TEAL,label:[$W bold(f)+b$])
  flow-arrow((2.2,0),(4.2,0),color:BLUE,label:[softmax])
}))
#pause
#result[Segmentation shares one classifier across locations; evidence changes, weights do not. Argmax waits for INFER.]

== Softmax acts across classes at one location—not across pixels #D

#clock-strip("train")
#case-strip([one pixel], [Classes are *background, road, car* and $bold(z)=mat(.2;1.4;-.3)$.])
#pause
#v(6pt)
#align(center, grid(columns:(1fr,1fr),gutter:14pt,
  hairline([EXPONENTIATE], [$exp(bold(z)) approx mat(1.22;4.05;.74)$
    $sum_k exp(z_k) approx 6.01$], color: BLUE),
  hairline([NORMALIZE], [$bold(p) approx mat(.203;.674;.123)$
    $p_"road"=.674$], color: BLUE),
))
#pause
#align(center, text(size: 19pt)[$ell=-log p_"road"=-log .674 approx .395.$])
#pause
#place(bottom + center, dy:-2pt, result[$sum_k p_(h,w,k)=1$ independently at every $(h,w)$.])

== Binary foreground can use one logit and one sigmoid #D

#clock-strip("train")
#align(center, text(size: 20pt)[$p_i=sigma(a_i)=1/(1+e^(-a_i)), quad a_i=log frac(p_i,1-p_i).$])
#pause
#v(8pt)
#align(center, table(
  columns: (34mm, 34mm, 34mm, 34mm, 34mm, 34mm), stroke: 0.5pt + MUTED, inset: 7pt,
  table.header([$p$], [$.90$], [$.60$], [$.30$], [$.10$], [$.05$]),
  [$a="logit"(p)$], [$2.197$], [$.405$], [$-.847$], [$-2.197$], [$-2.944$],
))
#pause
#note([The held binary mask uses one foreground probability. A $K$-class semantic task instead keeps $K$ logits and softmax.], color: BLUE)

== The held foreground probabilities are now fixed #V

#clock-strip("train")
#case-strip([constructed probability field], [$p_i$ is a teaching construction—not model output; blue cells satisfy $p_i >= .50$.])
#v(6pt)
#mask-triplet(show-hard: false)
#pause
#align(center, text(size: 18pt)[$sum_i p_i = 7.2, quad sum_i g_i=6, quad sum_i p_i g_i=4.8.$])

== Binary cross-entropy scores each pixel locally #D

#clock-strip("train")
$ ell_i = -[g_i log p_i + (1-g_i) log(1-p_i)]. $
#pause
#v(7pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 18pt,
  card([FOREGROUND · $g_i=1$], [penalty $=-log p_i$ \ confident $.90 arrow.r .105$ \ missed $.30 arrow.r 1.204$], color: TEAL),
  card([BACKGROUND · $g_i=0$], [penalty $=-log(1-p_i)$ \ safe $.10 arrow.r .105$ \ false alarm $.60 arrow.r .916$], color: BLUE),
))
#pause
#result[CE asks whether each probability is calibrated to that pixel’s label.]

== Group the held pixels to reproduce dense BCE #D

#clock-strip("train")
#case-strip([same $G,p$], [Five foreground pixels have $.90$; one has $.30$. Background has $.60 times 3$, $.10 times 5$, $.05 times 2$.])
#pause
$ 16 cal(L)_"BCE" = -10 log .90 - log .30 - 3 log .40 - 2 log .95 $
#pause
$ approx 1.0536 + 1.2040 + 2.7489 + .1026 approx 5.1090. $
#pause
#align(center, text(size: 24pt, weight: 700, fill: TEAL)[$cal(L)_"BCE" approx 5.109/16 approx .319.$])

== The tensor keeps spatial axes separate from the class axis #D

#clock-strip("train")
#align(center, table(
  columns: (42mm, 48mm, 88mm), stroke: 0.5pt + MUTED, inset: (x: 9pt, y: 7pt),
  table.header([*Axis*], [*Size*], [*Meaning*]),
  [$h$], [$H$], [vertical location],
  [$w$], [$W$], [horizontal location],
  [$k$], [$K$], [class score or probability],
))
#pause
#v(8pt)
#align(center, text(size: 21pt)[$Z[h,w,:] in RR^K arrow.r bold(p)[h,w,:] in [0,1]^K.$])
#pause
#result[Normalize over $k$; reduce the loss over valid $(h,w)$ locations.]

== TRAIN checkpoint: the labels supervise probabilities, not hard masks #Q

#clock-strip("train")
#align(center, text(size: 21pt)[Which object enters the derivative during training?])
#v(10pt)
#align(center, grid(columns: (1fr,1fr,1fr), gutter: 10pt,
  card([A], [the thresholded mask $P$], color: INK),
  card([B], [the differentiable probability $p$], color: INK),
  card([C], [only pixel accuracy], color: INK),
))
#pause
#v(10pt)
#uncover("2-")[#result[*B.* CE and Soft Dice operate on $p$; thresholding belongs to INFER.]]

// ───────────────────────── 3 · TRAIN: SPATIAL ARCHITECTURE ─────────────────────────
= TRAIN: preserve detail while acquiring context

== A classification head destroys the layout segmentation needs #V

#clock-strip("train")
#align(center, diagram(spacing: (23mm, 13mm), node-stroke: 1pt + INK, node-fill: white, {
  flow-node((0,0), [feature map \ $H' times W' times C$], color: TEAL, fill: PALE-TEAL)
  flow-node((2.2,0), [global pool], color: RED, fill: PALE-RED)
  flow-node((4.4,0), [one vector \ $RR^K$], fill: CREAM)
  flow-arrow((0,0),(2.2,0),color:RED)
  flow-arrow((2.2,0),(4.4,0),color:RED)
}))
#pause
#result[After locations collapse, different pixels cannot receive different labels.]

== A $1 times 1$ convolution is the shared pixel classifier #V

#clock-strip("train")
#align(center, diagram(spacing: (23mm, 13mm), node-stroke: 1pt + INK, node-fill: white, {
  flow-node((0,0), [$C$ features \ at $(h,w)$], w: 38mm)
  flow-node((2.2,0), [$K$ logits \ at $(h,w)$], color: BLUE, fill: PALE-BLUE)
  flow-node((4.4,0), [repeat over \ $H' times W'$], color: TEAL, fill: PALE-TEAL)
  flow-arrow((0,0),(2.2,0),color:TEAL,label:[$W bold(f)+b$])
  flow-arrow((2.2,0),(4.4,0),color:TEAL)
}))
#pause
#note([A $1 times 1$ kernel mixes channels at one location. Earlier spatial convolutions already gathered neighborhood evidence.], color: TEAL)

== Encoder–decoder networks exchange coordinates for context #V

#clock-strip("train")
#align(center, grid(columns: (1fr, 12mm, 1fr, 12mm, 1fr, 12mm, 1fr), gutter: 5pt, align: horizon,
  card([INPUT], [$256 times 256 times 3$], color: INK), [#align(center)[$arrow.r$]],
  card([ENCODE], [$32 times 32 times 256$ \ broad context], color: TEAL), [#align(center)[$arrow.r$]],
  card([DECODE], [$256 times 256 times 32$ \ localize], color: ACC), [#align(center)[$arrow.r$]],
  card([LOGITS], [$256 times 256 times K$], color: BLUE),
))
#pause
#result[Downsample to understand *what surrounds a pixel*; upsample to decide *which pixel*.]

== A real boundary audit exposes what region overlap can hide #V

#clock-strip("eval")
#align(center, evidence-image("region-boundary-audit.png", width: 143mm, height: 80.4mm))

== Does a larger grid recreate discarded detail? #Q

#clock-strip("train")
#align(center, text(size: 22pt)[A $2 times 2$ feature map is repeated into a $4 times 4$ grid. Did upsampling recover which source pixels created each value?])
#v(12pt)
#align(center, grid(columns: (1fr, 18mm, 1fr), gutter: 6pt, align: horizon,
  [#align(center, table(columns:2, stroke:.6pt+MUTED, inset:12pt, [1],[2],[3],[4]))],
  [#align(center, text(size:25pt, fill:MUTED)[$arrow.r$])],
  [#align(center, table(columns:4, stroke:.6pt+MUTED, inset:7pt,
    [1],[1],[2],[2], [1],[1],[2],[2], [3],[3],[4],[4], [3],[3],[4],[4]))],
))
#pause
#note([Commit before the next slide: resolution and information are not synonyms.], color: BLUE)

== U-Net routes fine detail around the bottleneck #V

#clock-strip("train")
#uncover("1-")[#unet-diagram]
#uncover("2-")[#align(center, text(size: 16pt)[Teal learns context; orange restores resolution; green skips carry aligned fine features.])]
#uncover("3-")[#place(bottom + center, dy: -2pt, result[Upsampling adds locations; skips restore evidence that the bottleneck discarded.])]

== This ledger is a same-padding teaching U-Net—not the 2015 pixel sizes #D

#clock-strip("train")
#align(center, grid(columns: (1fr,1fr), gutter: 16pt,
  card([TOY LEDGER USED HERE], [same-padding convolutions preserve each stage size; matching encoder and decoder tensors concatenate directly], color: BLUE),
  card([ORIGINAL U-NET], [valid convolutions shrink tensors; the encoder feature is copied *and cropped* before concatenation], color: MUTED),
))
#pause
#v(9pt)
$ U,E in RR^(64 times 64 times 128) quad arrow.r quad "concat"(U,E) in RR^(64 times 64 times 256). $
#pause
#result[The skip operation is concatenation in original U-Net; “skip connection” is not universally synonymous with addition.]

== Boundary failure has a smallest useful diagnostic #V

#clock-strip("train")
#align(center, table(
  columns: (58mm, 64mm, 68mm), stroke: .5pt+MUTED, inset:(x:8pt,y:7pt),
  table.header([*Symptom*], [*Suspect*], [*Smallest test*]),
  [correct class; blurred edge], [coarse stride or missing skip], [overlay boundary; ablate one skip],
  [checkerboard texture], [transposed-conv overlap], [compare resize+conv at same output size],
  [shifted instance mask], [RoI quantization], [compare RoIPool with RoIAlign],
))
#pause
#result[Inspect spatial alignment before changing the loss.]

== Architecture checkpoint: context and coordinates play different roles #Q

#clock-strip("train")
#align(center, text(size: 21pt)[If the deep path is removed but every high-resolution skip remains, what is most at risk?])
#v(10pt)
#align(center, grid(columns:(1fr,1fr,1fr),gutter:10pt,
  card([A], [semantic meaning], color: INK),
  card([B], [number of pixels], color: INK),
  card([C], [softmax normalization], color: INK),
))
#pause
#uncover("2-")[#result[*A.* A crisp local edge does not by itself say whether the region is road, cat, or shadow.]]

// ───────────────────────── 4 · INFER + EVAL ─────────────────────────
= INFER makes a mask; EVAL measures its agreement

== TRAIN ends with probabilities; INFER begins without labels #V

#clock-strip("train")
#v(5pt)
#align(center, text(size: 18pt)[$G,p arrow.r cal(L)_"BCE", cal(L)_"Dice"$])
#pause
#clock-strip("infer")
#v(5pt)
#align(center, text(size: 18pt)[$p, tau arrow.r P$; no truth and no loss])
#pause
#result[The same $p$ crosses a clock boundary; its meaning does not change.]

== Commit: threshold the held probabilities at $.50$ #Q

#clock-strip("infer")
#case-strip([decision rule], [$P_i=1[p_i >= .50]$. Equality is foreground by convention.])
#v(7pt)
#mask-triplet(show-hard: false)
#pause
#note([Before revealing $P$, predict the four counts: TP, FP, FN, TN.], color: BLUE)

== Thresholding yields this exact hard mask #V

#clock-strip("infer")
#mask-triplet()
#pause
#align(center, text(size: 16pt, fill: MUTED)[Thresholding is nondifferentiable; it belongs to inference or metric computation, not the training path.])

== The same clocks operate on actual pixels #V

#align(center, evidence-image("constructed-probability-mask.png", width: 154mm, height: 86.6mm))
#pause
#v(3pt)
#note([GT is official; $p$ is the declared +16 px construction; $P=1[p >= .50]$ is computed. No checkpoint was executed.], color: BLUE)

== Count errors before naming a metric #D

#clock-strip("eval")
#two(
  [#align(center, text(size:15pt,weight:700,fill:TEAL)[truth $G$])#v(4pt)#mask-grid(gt)],
  [#align(center, text(size:15pt,weight:700,fill:ACC)[prediction $P$])#v(4pt)#mask-grid(pred,mode:"hard")],
)
#pause
#align(center, text(size: 21pt)[#uncover("2-")[$"TP"=5$] #h(12pt) #uncover("3-")[$"FP"=3$] #h(12pt) #uncover("4-")[$"FN"=1$] #h(12pt) #uncover("5-")[$"TN"=7$]])
#uncover("6-")[#place(bottom + center, dy:-2pt, result[Every hard metric below is a different reduction of the same four counts.])]

== Pixel accuracy can reward an empty foreground prediction #D

#clock-strip("eval")
#case-strip([held mask], [$"accuracy"=("TP"+"TN")/16=(5+7)/16=.750$.])
#pause
#v(7pt)
Now imagine a $100 times 100$ image with only $100$ tumor pixels.
#pause
$ "predict background everywhere" arrow.r "accuracy"=9900/10000=99%. $
#pause
$ "tumor recall"=0/100=0. $
#pause
#result[Accuracy weights the easy background by its frequency; it may not reflect the foreground task.]

== IoU removes true background from the denominator #D

#clock-strip("eval")
$ "IoU" = frac(abs(P inter G),abs(P union G)) = frac("TP","TP"+"FP"+"FN"). $
#pause
#case-strip([same counts], [$abs(P inter G)=5$ and $abs(P union G)=5+3+1=9$.])
#pause
#align(center, text(size: 25pt, weight:700,fill:INK)[$"IoU"=frac(5,9) approx .556.$])
#pause
#result[Seven true-background pixels cannot inflate foreground IoU.]

== Real pixels reveal where every region count lives #V

#clock-strip("eval")
#v(-1pt)
#align(center, evidence-image("real-error-overlay.png", width: 155mm, height: 87.2mm))

== Dice gives the intersection twice the weight #D

#clock-strip("eval")
$ "Dice" = frac(2 abs(P inter G),abs(P)+abs(G)) = frac(2"TP",2"TP"+"FP"+"FN"). $
#pause
#case-strip([same counts], [$abs(P)=8$, $abs(G)=6$, and $abs(P inter G)=5$.])
#pause
#align(center, text(size:25pt,weight:700,fill:INK)[$"Dice"=frac(10,14) approx .714.$])
#pause
#result[For one binary mask, Dice is the foreground F1 score.]

== Dice is a monotone transform of IoU for one mask pair #D

#clock-strip("eval")
#align(center, text(size:22pt)[$"Dice"=frac(2"IoU",1+"IoU"), quad "IoU"=frac("Dice",2-"Dice").$])
#pause
#v(8pt)
$ frac(2(5/9),1+5/9)=frac(10,14) approx .714. $
#pause
#note([Qualifier: this preserves ranking for a single class and mask pair. Macro-averaging nonlinear scores over classes or images can change a model ranking.], color: RED)

== Soft Dice scores the held probabilities before thresholding #D

#clock-strip("train")
$ "SoftDice" = frac(2 sum_i p_i g_i + epsilon,sum_i p_i + sum_i g_i + epsilon). $
#pause
#case-strip([same $G,p$], [$sum p=7.2$, $sum g=6$, $sum p g=4.8$; omit tiny $epsilon$ in the displayed arithmetic.])
#pause
#align(center, text(size:24pt,weight:700,fill:TEAL)[$"SoftDice"=frac(9.6,13.2) approx .727.$])
#pause
#align(center, text(size:22pt,fill:ACC)[$cal(L)_"Dice"=1-.727=.273.$])

== CE and Soft Dice expose different failure surfaces #V

#clock-strip("train")
#align(center, grid(columns:(1fr,1fr),gutter:16pt,
  card([LOCAL CALIBRATION · CE], [scores every valid pixel; strong gradients identify which probabilities are wrong], color: BLUE),
  card([REGION OVERLAP · DICE], [couples foreground pixels through one region denominator; resists background domination], color: TEAL),
))
#pause
#v(8pt)
#align(center, text(size:22pt)[$cal(L)=cal(L)_"CE"+lambda cal(L)_"Dice".$])
#pause
#result[Combining them does not make either definition optional: declare $lambda$, reduction, and empty-mask policy.]

== Semantic mIoU averages declared per-class IoUs #D

#clock-strip("eval")
For each class $k$,
$ "IoU"_k=frac("TP"_k,"TP"_k+"FP"_k+"FN"_k), quad "mIoU"=frac(1,abs(K_"report")) sum_(k in K_"report") "IoU"_k. $
#pause
#v(7pt)
#align(center, grid(columns:(1fr,1fr,1fr),gutter:10pt,
  hairline([CLASSES], [include or exclude background?], color: TEAL),
  hairline([VOID], [remove ignored pixels from all counts], color: BLUE),
  hairline([REDUCTION], [image mean or dataset-level confusion?], color: INK),
))
#pause
#result[“mIoU” is incomplete until its class set and aggregation rule are named.]

== Diagnose a surprising score on the correct clock #V

#align(center, table(
  columns:(55mm,64mm,72mm),stroke:.5pt+MUTED,inset:(x:7pt,y:6pt),
  table.header([*Symptom*],[*Suspect*],[*Smallest test*]),
  [99% accuracy; IoU $=0$],[rare foreground],[foreground recall; class counts],
  [loss falls; mIoU flat],[threshold or class collapse],[histogram $p$ by class; confusion],
  [IoU $.811$; boundary F1 $.441$],[spatial displacement],[overlay FP/FN; inspect contour distances],
))
#pause
#result[Region and boundary metrics interrogate different geometry; choose the diagnostic on EVAL, then inspect actual pixels.]

// ───────────────────────── 5 · INSTANCE SEGMENTATION ─────────────────────────
= One semantic region is not yet two objects

== A generated touching-object scene isolates the identity question #V

#align(center, grid(columns: (1.25fr, .8fr, .8fr), gutter: 12pt, align: horizon,
  [#align(center, text(size:16pt,weight:700)[generated concept scene])#v(5pt)
   #image(ORCHARD, width:88mm,height:48mm,fit:"cover")],
  [#align(center, text(size:16pt,weight:700)[constructed union])#v(5pt)#mask-grid(gt)
   #v(5pt)#align(center, text(size:14pt,fill:MUTED)[one connected component])],
  [#align(center, text(size:16pt,weight:700)[constructed identities])#v(5pt)#mask-grid(ids,mode:"id")
   #v(5pt)#align(center, text(size:14pt,fill:MUTED)[id 1 vs id 2])],
))
#pause
#result[Oxford supplies real semantic GT; this explicitly generated scene demonstrates why a semantic union cannot count touching instances.]

== Instance segmentation returns a scored set—not one class grid #D

#clock-strip("infer")
$ x arrow.r { (hat(y)_j,hat(b)_j,hat(m)_j,hat(s)_j) }_(j=1)^N. $
#pause
#align(center, table(
  columns:(34mm,52mm,92mm),stroke:.5pt+MUTED,inset:(x:8pt,y:6pt),
  table.header([*Symbol*],[*Shape*],[*Meaning*]),
  [$hat(y)_j$],[scalar],[object class],
  [$hat(b)_j$],[four coordinates],[box that places the mask],
  [$hat(m)_j$],[$m times m$ then resized],[shape in the RoI frame],
  [$hat(s)_j$],[scalar],[ranking confidence],
))

== Mask R-CNN adds a mask branch parallel to class and box #V

#uncover("1-")[#mask-rcnn-diagram]
#uncover("2-")[#align(center, text(size:16pt)[RPN = Region Proposal Network; RoI = Region of Interest.])]
#uncover("3-")[#place(bottom + center,dy:-2pt,result[Detection proposes *which object*; the mask branch predicts *which RoI pixels*.])]

== Separate the RPN loss from the second-stage RoI loss #D

#clock-strip("train")
#align(center, grid(columns:(1fr,1fr),gutter:16pt,
  card([STAGE 1 · RPN], [$cal(L)_"RPN-obj"$ on sampled anchors \ $cal(L)_"RPN-box"$ on positive anchors], color: TEAL),
  card([STAGE 2 · SAMPLED ROIS], [$cal(L)_"cls"$ on positive + negative RoIs \ $cal(L)_"box"+cal(L)_"mask"$ on positive RoIs], color: BLUE),
))
#pause
#v(8pt)
#align(center, text(size:21pt)[$cal(L)_"total"=cal(L)_"RPN"+cal(L)_"RoI".$])
#pause
#result[The familiar three-term Mask R-CNN equation is the *second-stage RoI loss*, not the whole RPN-plus-RoI total.]

== Original Mask R-CNN predicts $K$ independent binary masks #D

#clock-strip("train")
$ hat(M) in RR^(K times m times m), quad hat(M)_(k,u,v)=sigma(a_(k,u,v)). $
#pause
#align(center, text(size:18pt)[For a training RoI with ground-truth class $k$, only channel $k$ contributes to $cal(L)_"mask"$.])
#pause
#v(8pt)
#align(center, grid(columns:(1fr,1fr),gutter:16pt,
  card([CLASS HEAD], [softmax chooses the object category], color: BLUE),
  card([MASK CHANNEL $k$], [sigmoid makes foreground/background decisions inside that RoI], color: ACC),
))
#pause
#note([A class-agnostic one-mask head is a valid later variant; it is not the original $K$-mask formulation.], color: MUTED)

== Calculate one selected-channel mask BCE #D

#clock-strip("train")
$ M=mat(1,1;0,0), quad hat(M)=mat(.8,.6;.3,.1). $
#pause
$ cal(L)_"mask"=-frac(1,4)[log .8+log .6+log(1-.3)+log(1-.1)] $
#pause
$ =frac(1,4)[.223+.511+.357+.105] approx .299. $
#pause
#result[Pixels in this RoI and the selected ground-truth class channel define the mask loss.]

== RoIAlign reduces quantization error for boxes and especially masks #V

#two(
  [#hairline([ROIPOOL], [round RoI and bin coordinates; convenient, but spatially quantized], color: RED)
   #v(8pt)#align(center, box(width:58mm,height:34mm,stroke:.6pt+MUTED)[
     #place(center, rect(width:37mm,height:20mm,stroke:1.5pt+RED))
     #place(center,dx:4pt,dy:-3pt,rect(width:37mm,height:20mm,stroke:1pt+MUTED))
   ])],
  [#hairline([ROIALIGN], [keep fractional coordinates; sample neighboring features bilinearly], color: GREEN)
   #v(8pt)#align(center, box(width:58mm,height:34mm,stroke:.6pt+MUTED)[
     #place(center,dx:4pt,dy:-3pt,rect(width:37mm,height:20mm,stroke:1.5pt+GREEN))
   ])],
)
#pause
#note([The Mask R-CNN paper reports box-AP gains too; the alignment benefit is larger for pixel-accurate masks.], color: INK)

== Instance evaluation reuses L10 AP with mask IoU #D

#clock-strip("eval")
#align(center, diagram(spacing:(18mm,12mm),node-stroke:1pt+INK,node-fill:white,{
  flow-node((0,0),[score-sort \ predicted masks],color:BLUE,fill:PALE-BLUE,w:41mm)
  flow-node((2,0),[same-class \ one-to-one match],color:INK,fill:CREAM,w:42mm)
  flow-node((4,0),[*mask* IoU \ threshold],color:INK,w:36mm)
  flow-node((6,0),[precision–recall \ mask AP],color:INK,w:42mm)
  flow-arrow((0,0),(2,0),color:BLUE)
  flow-arrow((2,0),(4,0),color:INK)
  flow-arrow((4,0),(6,0),color:INK)
}))
#pause
#result[Box AP asks whether rectangles overlap; mask AP asks whether object pixels overlap.]

// ───────────────────────── 6 · SHOULD ─────────────────────────
= SHOULD: policies, alignment, and panoptic output

== SHOULD · void pixels leave both the loss sum and denominator #D

#clock-strip("train")
#align(center, grid(columns: (104mm, 1fr), gutter: 14pt, align: horizon,
  [#evidence-image("real-trimap-contract.png", width: 104mm, height: 58.5mm, fit: "cover")],
  [Let $V$ be the valid labeled pixels:
   $ cal(L)_"CE"=-frac(1,abs(V)) sum_(i in V) log p_(i,y_i). $
   #v(7pt)
   #text(size: 15pt)[Oxford values 1/2 are valid: $abs(V)=221,704$. Value 3 contributes $18,296$ ignored boundary pixels.]],
))
#pause
#result[Remove ignored pixels from CE, confusion counts, IoU, and Dice—not only from the numerator.]

== SHOULD · empty masks and class averaging need a written policy #D

#clock-strip("eval")
#align(center, grid(columns:(1fr,1fr),gutter:16pt,
  card([EMPTY–EMPTY], [raw Dice is $0/0$; choose smoothing, exclusion, or a declared perfect score], color: BLUE),
  card([MULTI-CLASS], [compute per class; declare background inclusion, empty-class handling, and macro or frequency weighting], color: INK),
))
#pause
#v(8pt)
#note([Also declare whether scores are averaged per image or accumulated from a dataset-level confusion matrix.], color: RED)
#pause
#result[The reduction rule is part of the metric—not an implementation footnote.]

== SHOULD · upsampling restores resolution by different mechanisms #V

#align(center, table(
  columns:(52mm,68mm,70mm),stroke:.5pt+MUTED,inset:(x:7pt,y:7pt),
  table.header([*Method*],[*What creates locations?*],[*What learns?*]),
  [nearest / bilinear],[fixed interpolation],[following convolution refines],
  [transposed convolution],[learned overlap pattern],[kernel and mixing jointly],
  [skip + resize],[decoder plus aligned encoder detail],[fusion after concatenation],
))
#pause
#result[Every route needs evidence to refine the larger grid; interpolation alone cannot reverse pooling.]

== SHOULD · bilinear interpolation is a weighted four-neighbor sum #D

Suppose a RoIAlign sample lies halfway in both directions between
$ mat(1,3;5,7). $
#pause
All four weights equal $.25$:
$ f=.25(1)+.25(3)+.25(5)+.25(7)=4. $
#pause
#v(8pt)
#align(center, text(size:18pt)[Moving the fractional sample changes the weights smoothly; no coordinate is rounded.])
#pause
#result[Precise sampling preserves alignment; it does not invent image detail.]

== SHOULD · panoptic output assigns one class–identity record per pixel #V

#align(center, diagram(spacing:(29mm,13mm),node-stroke:1pt+INK,node-fill:white,{
  flow-node((0,-.6),[*stuff* \ class only],color:TEAL,fill:PALE-TEAL)
  flow-node((0,.6),[*things* \ class + id],color:ACC,fill:PALE-ACC)
  flow-node((2.5,0),[one non-overlapping \ class–id map],color:INK,fill:CREAM,w:48mm)
  flow-arrow((0,-.6),(2.5,0),color:TEAL)
  flow-arrow((0,.6),(2.5,0),color:ACC)
}))
#pause
#align(center, text(size:16pt,fill:MUTED)[Overlapping instance hypotheses must be fused; datasets may reserve a void label for uncertain pixels.])

== SHOULD · choose the output and metric from the question #D

#align(center, table(
  columns:(44mm,70mm,72mm),stroke:.5pt+MUTED,inset:(x:7pt,y:7pt),
  table.header([*Need*],[*Output*],[*Evaluation*]),
  [road area],[$H times W$ semantic classes],[per-class IoU / mIoU],
  [count touching trees],[scored set of instance masks],[mask AP under mask IoU],
  [whole labeled scene],[$H times W$ class–id map],[PQ or declared panoptic protocol],
))
#pause
#result[Architecture names follow the output contract; they do not define the task.]

// ───────────────────────── 7 · OPTIONAL + SYNTHESIS ─────────────────────────
== OPTIONAL · primary papers establish the architectural claims #V

#set text(size: 14pt)
#align(center, grid(columns:(1fr,1fr),gutter:10pt,
  hairline([FCN + U-NET], [#link(FCN)[Long et al.] · pixels-to-pixels; #link(UNET)[Ronneberger et al.] · context plus localization], color: TEAL),
  hairline([V-NET + MASK R-CNN], [#link(VNET)[Milletari et al.] · Dice objective; #link(MASKRCNN)[He et al.] · mask branch + RoIAlign], color: BLUE),
  hairline([PANOPTIC + COMPANION], [#link(PANOPTIC)[Kirillov et al.] · stuff + things; #link(NB)[exact L10 Colab] · toy and real ledgers], color: INK),
  hairline([REAL EVIDENCE], [#link(PETS)[Oxford-IIIT Pet] · Abyssinian_1 · official trimap · CC BY-SA 4.0], color: GREEN),
  hairline([REPRODUCIBILITY], [`shared/vision-evidence/oxford-iiit-pet/l10/` · builder + manifest + arrays + metric ledger], color: ACC),
  hairline([HONEST CLAIM], [model output = none · measured performance = none · +16 px probability field = constructed], color: RED),
))
#v(4pt)
#caption[OBSERVED = photo · GT = trimap · CONSTRUCTED = orchard / toy $p$ / shifted $p$ · COMPUTED = masks, errors, metrics]

== OPTIONAL · variants change mechanisms, not the contract #V

#block(width:100%)[
  #set text(size:15pt)
  #align(center, table(
    columns:(55mm,67mm,65mm),stroke:.5pt+MUTED,inset:(x:7pt,y:5pt),
    table.header([*Variant*],[*What changes*],[*What stays explicit*]),
    [class-agnostic mask head],[$1 times m times m$ rather than $K times m times m$],[object class comes from class head],
    [dilated / pyramid decoder],[context without identical downsampling],[output grid and pixel loss],
    [query-based mask sets],[learned queries replace RPN RoIs],[one-to-one identity and mask metric],
    [panoptic fusion],[resolve overlaps and stuff regions],[one class–identity assignment per pixel],
  ))
]
#pause
#note([#text(size:15pt)[Do not turn variants into a catalogue: ask which output, alignment, loss, or matching assumption changed.]], color: MUTED)

== Revisit the orchard commitment before revealing the answer #Q

#align(center, image(ORCHARD, width: 78mm, height: 27mm, fit: "cover"))
#v(2pt)
#case-strip([same generated concept], [Report total canopy area *and* count two touching trees.])
#v(2pt)
#align(center, grid(columns:(1fr,1fr,1fr,1fr),gutter:8pt,
  card([A], [global class], color: INK),
  card([B], [class + box], color: INK),
  card([C], [semantic map], color: INK),
  card([D], [separate object masks], color: INK),
))
#pause
#uncover("2-")[#v(5pt)#result[*D.* Instance masks preserve the two identities; their union gives total area. If every background/stuff pixel also needs a class, use a panoptic class–identity map.]]

== CORE · final checklist and public L12 handoff #V

#align(center, grid(columns:(1fr,1fr),gutter:14pt,
  card([SEGMENTATION CHECKLIST], [
    *1.* What is the output structure? \
    *2.* Which axis holds classes? \
    *3.* What enters TRAIN loss? \
    *4.* Which INFER threshold/fusion applies? \
    *5.* Which EVAL classes, masks, and reductions define the score?
  ], color: TEAL),
  card([PUBLIC L12 · NEXT TOKEN], [
    Pixel prediction shared one classifier over spatial positions $(h,w)$. \
    Next-token prediction shares one classifier over sequence positions $t$. \
    Pixel classes become vocabulary tokens; *causality* becomes the new constraint.
  ], color: BLUE),
))
#pause
#v(8pt)
#result[A dense model is trustworthy only when output support, three clocks, alignment, and metric protocol agree.]
