// Localization and Object Detection — public Lecture 10 · ES 667 Deep Learning
// Compile from the repository root:
//   typst compile --root . --input handout=true lecture9/L9-detection.typ /private/tmp/L9-handout.pdf
//   typst compile --root . lecture9/L9-detection.typ /private/tmp/L9-presentation.pdf

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Localization and Object Detection],
  subtitle: [A fixed dense tensor becomes a scored set of boxes],
)

#let NB-IOU = "https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L09/01_iou_and_bbox_losses.ipynb"
#let NB-NMS = "https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L09/02_nms_from_scratch.ipynb"
#let RCNN = "https://arxiv.org/abs/1311.2524"
#let FAST = "https://arxiv.org/abs/1504.08083"
#let FASTER = "https://arxiv.org/abs/1506.01497"
#let YOLO = "https://arxiv.org/abs/1506.02640"
#let FOCAL = "https://arxiv.org/abs/1708.02002"
#let FPN = "https://arxiv.org/abs/1612.03144"
#let GIOU = "https://arxiv.org/abs/1902.09630"
#let PETS = "https://www.robots.ox.ac.uk/~vgg/data/pets/"
#let DET-EVIDENCE = "/shared/vision-evidence/oxford-iiit-pet/l9/"

// Stable semantic grammar:
// INK = given / evaluation · TEAL = TRAIN / truth · ACC = constructed candidate
// BLUE = INFER / control · GREEN = survivor / TP · RED = suppressed / FP.
#let swatch(color, body) = box(width: 13pt, height: 4pt, fill: color, radius: 1pt) + h(4pt) + body
#let semantic-legend = align(center, text(size: 12.5pt)[
  #swatch(INK, [given / eval]) #h(11pt)
  #swatch(TEAL, [truth / train]) #h(11pt)
  #swatch(ACC, [raw candidate]) #h(11pt)
  #swatch(BLUE, [infer / control]) #h(11pt)
  #swatch(GREEN, [survivor / TP]) #h(11pt)
  #swatch(RED, [suppressed / FP])
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
  image(DET-EVIDENCE + name, width: width, height: height, fit: fit),
)

#let _chip(lbl, word, col) = box(
  fill: col, inset: (x: 7pt, y: 3.5pt), radius: 4pt, baseline: 0.28em,
  text(font: "IBM Plex Mono", size: 12pt, fill: white, weight: 700,
    tracking: 0.4pt, [#lbl#h(5pt)#text(size: 9pt, weight: 500, tracking: 0.8pt)[#upper(word)]])
)
#let Q = [#h(1fr)#_chip("Q", "question", BLUE)]
#let A = [#h(1fr)#_chip("A", "answer", GREEN)]
#let V = [#h(1fr)#_chip("V", "visual", TEAL)]
#let D = [#h(1fr)#_chip("D", "derivation", ACC)]
#let I = [#h(1fr)#_chip("I", "interactive", GREEN)]
#let OPT = [#h(1fr)#_chip(sym.star.filled, "optional", MUTED)]

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
  _clock([TRAIN], active == "train", TEAL, [candidate ↔ truth; losses]),
  _clock([INFER], active == "infer", BLUE, [prediction ↔ prediction; NMS]),
  _clock([EVAL], active == "eval", INK, [prediction ↔ truth; AP]),
))

#let flow-node(pos, body, color: INK, fill: white, w: 34mm) = node(
  pos,
  block(width: w, inset: 5pt, align(center, text(size: 12.5pt, body))),
  shape: fletcher.shapes.rect, corner-radius: 3pt,
  stroke: 0.9pt + color, fill: fill,
)
#let flow-arrow(a, b, color: MUTED, label: none) = edge(
  a, b, "-|>", stroke: 0.85pt + color,
  label: if label == none { none } else { text(size: 10.5pt, fill: color, label) },
)
#let round-node(pos, body, color: INK, fill: white, r: 10mm) = node(
  pos,
  box(width: 2*r, height: 2*r, align(center + horizon, text(size: 12pt, body))),
  shape: fletcher.shapes.circle, stroke: 1pt + color, fill: fill,
)

// The scene uses one scale throughout: 1 x-unit = 1 mm, 1 y-unit = 0.65 mm.
// Truths: G1=[10,10,50,50], G2=[60,15,90,55].
// Predictions: A=G1, B=[12,12,52,52], E=[35,5,55,25], C=G2, D=[63,17,91,56].
#let _pcol(mode, id) = {
  if mode == "raw" { ACC }
  else if mode == "step1" {
    if id == "A" { GREEN } else if id == "B" { RED } else { ACC }
  } else if mode == "step2" {
    if id == "A" or id == "E" { GREEN } else if id == "B" { RED } else { ACC }
  } else if mode == "final" {
    if id == "A" or id == "E" or id == "C" { GREEN } else { RED }
  } else if mode == "eval" {
    if id == "A" or id == "C" { GREEN } else if id == "E" { RED } else { MUTED }
  } else { ACC }
}

#let _plabel(id, mode) = box(
  fill: white, inset: (x: 3pt, y: 1.5pt), radius: 2pt,
  text(size: 10.5pt, weight: 750, fill: _pcol(mode, id), id),
)

#let held-scene(mode: "raw", show-truth: true) = block(
  width: 108mm, height: 49mm, fill: rgb("#F4F2EE"),
  stroke: 0.7pt + MUTED,
  [
    // This strip juxtaposes two image-scoped canonical frames; it is not one
    // scene. During INFER, the official truths and their labels are absent.
    #place(top + left, dx: 6mm, dy: 2mm,
      text(size: 9.5pt, weight: 700, fill: MUTED)[I1 · A/B/E])
    #place(top + left, dx: 65mm, dy: 2mm,
      text(size: 9.5pt, weight: 700, fill: MUTED)[I2 · C/D])
    #place(top + left, dx: 61mm, dy: 3mm,
      rect(width: 0.6pt, height: 42mm, fill: MUTED.lighten(55%)))
    #if show-truth {
      // Truths are thick, so exact candidates A and C remain visible inside.
      place(top + left, dx: 4mm + 10mm, dy: 4mm + 6.5mm,
        rect(width: 40mm, height: 26mm, stroke: 3pt + TEAL))
      place(top + left, dx: 4mm + 60mm, dy: 4mm + 9.75mm,
        rect(width: 30mm, height: 26mm, stroke: 3pt + TEAL))
      place(top + left, dx: 15mm, dy: 7mm,
        box(fill: white, inset: 2pt, text(size: 10pt, weight: 700, fill: TEAL)[G1]))
      place(top + left, dx: 65mm, dy: 10mm,
        box(fill: white, inset: 2pt, text(size: 10pt, weight: 700, fill: TEAL)[G2]))
    } else {
      place(top + right, dx: -3mm, dy: 2mm,
        box(fill: white, inset: 2pt, text(size: 9pt, weight: 700, fill: BLUE)[GT HIDDEN]))
    }

    #if mode != "truth" {
      place(top + left, dx: 14mm, dy: 10.5mm,
        rect(width: 40mm, height: 26mm, stroke: 1.25pt + _pcol(mode, "A")))
      place(top + left, dx: 16mm, dy: 11.8mm,
        rect(width: 40mm, height: 26mm, stroke: 1.25pt + _pcol(mode, "B")))
      place(top + left, dx: 39mm, dy: 7.25mm,
        rect(width: 20mm, height: 13mm, stroke: 1.25pt + _pcol(mode, "E")))
      place(top + left, dx: 64mm, dy: 13.75mm,
        rect(width: 30mm, height: 26mm, stroke: 1.25pt + _pcol(mode, "C")))
      place(top + left, dx: 67mm, dy: 15.05mm,
        rect(width: 28mm, height: 25.35mm, stroke: 1.25pt + _pcol(mode, "D")))

      place(top + left, dx: 16mm, dy: 33mm, _plabel("A", mode))
      place(top + left, dx: 48mm, dy: 36mm, _plabel("B", mode))
      place(top + left, dx: 43mm, dy: 7mm, _plabel("E", mode))
      place(top + left, dx: 65mm, dy: 36mm, _plabel("C", mode))
      place(top + left, dx: 88mm, dy: 38mm, _plabel("D", mode))
    }
  ],
)

#let score-ledger(rows) = align(center, table(
  columns: (16mm, 23mm, 52mm, 42mm, 39mm),
  stroke: 0.45pt + MUTED, inset: (x: 7pt, y: 4pt),
  align: (center, center, center, center, center),
  table.header([*ID*], [*score*], [*box $[x_0,y_0,x_1,y_1]$*], [*status now*], [*next comparison*]),
  ..rows,
))

#title-slide()

// ───────────────────────────── 1 · BRIDGE + COMMIT ─────────────────────────────
= L8B kept spatial features; detection must spend them

== Two real images turn “where” into supervised geometry #V

#align(center, evidence-image("real-head-targets.png", width: 154mm, height: 86.6mm))
#pause
#v(3pt)
#note([OBSERVED pixels + official XML head boxes. Detection reuses the L8B backbone, but changes the head, targets, geometry, and evaluation.], color: BLUE)

== Commit: which operations turn five candidates into trustworthy detections? #Q

#case-strip([opening mini-batch], [CONSTRUCTED, not model output: A/B/E belong to I1; C/D belong to I2.])
#v(3pt)
#align(center, evidence-image("real-candidates.png", width: 100mm, height: 56.3mm))
#pause
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 12pt,
  card([A · THRESHOLD], [score threshold alone], color: INK),
  card([B · GEOMETRY], [decode → score → NMS → match], color: INK),
  card([C · LOSS], [rerun loss at inference], color: INK),
))

== Outline #V

#align(center, grid(columns: (77mm, 1fr), gutter: 15pt, align: horizon,
  [#evidence-image("real-eval-matching.png", width: 77mm, height: 43.3mm)
   #v(3pt)
   #caption[OBSERVED pixels + GT · CONSTRUCTED candidates · COMPUTED matches]],
  [#set text(size: 13pt)
   #grid(columns: (1fr, 1fr), gutter: 9pt, row-gutter: 9pt,
    card([01 · BOX CONTRACT], [Choose coordinates, decode rules, and class scores.], color: TEAL),
    card([02 · OVERLAP], [Use IoU to compare boxes under an explicit convention.], color: BLUE),
    card([03 · TRAIN], [Assign candidates; combine classification and localization losses.], color: ACC),
    card([04 · INFER], [Decode, threshold, and suppress duplicate predictions.], color: BLUE),
    card([05 · EVALUATE], [Match once, rank by score, and accumulate AP.], color: GREEN),
    card([06 · ARCHITECTURES], [Relate two-stage, one-stage, FPN, and anchor-free heads.], color: TEAL),
  )],
))
#pause
#v(6pt)
#semantic-legend

== Three clocks reuse boxes—but never the same comparison #V

#clock-strip("train")
#v(8pt)
#clock-strip("infer")
#v(8pt)
#clock-strip("eval")
#pause
#v(8pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  [#align(center)[#text(fill: TEAL, weight: 700)[TRAIN]
   #v(2pt)
   #text(size: 14.5pt)[assign candidate ↔ truth]]],
  [#align(center)[#text(fill: BLUE, weight: 700)[INFER]
   #v(2pt)
   #text(size: 14.5pt)[compare prediction ↔ prediction]]],
  [#align(center)[#text(fill: INK, weight: 700)[EVAL]
   #v(2pt)
   #text(size: 14.5pt)[match prediction ↔ truth]]],
))
#pause
#result[The clock strip will remain visible whenever “IoU threshold” could otherwise mean three different things.]

// ───────────────────────────── 2 · BOX + HEAD CONTRACT ─────────────────────────────
= A box is four numbers with an explicit convention

== Detection predicts a set, not a longer class vector #V

#align(center, [classifier: $x arrow.r hat(p) in Delta^(K-1)$])
#pause
#v(7pt)
#align(center, [detector: $x arrow.r { (hat(y)_i, hat(b)_i, s_i) }_(i=1)^N$])
#v(7pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([CLASS], [$hat(y)_i$: what is here?], color: ACC),
  hairline([BOX], [$hat(b)_i in RR^4$: where is it?], color: BLUE),
  hairline([SCORE], [$s_i$: how confident?], color: GREEN),
))
#pause
#note([A fixed network tensor may contain many candidate triples; thresholding and NMS make the returned set variable-length.], color: ACC)

== Our coordinate convention prevents the silent “+1” bug #D

#case-strip([real annotation contract], [VOC XML uses one-based inclusive corners; the course converts once to zero-based half-open $[x_0,y_0,x_1,y_1)$.])
#v(4pt)
#align(center, evidence-image("real-head-targets.png", width: 110mm, height: 61.9mm))
#pause
#v(3pt)
#alertbox[Example I1: VOC $(333,72)$–$(425,158)$ becomes $[332,71,425,158)$, so area is $93 dot 87=8,091$. No later component adds $+1$.]

== The canonical truths fix every later area calculation #D

#case-strip([constructed numeric spine], [$G_1=[10,10,50,50]$ in I1; $G_2=[60,15,90,55]$ in I2. Separate positive affine maps preserve IoU.])
#v(6pt)
#two(
  [*$G_1$*\
   $w=50-10=40$\
   $h=50-10=40$\
   $|G_1|=1,600$],
  [*$G_2$*\
   $w=90-60=30$\
   $h=55-15=40$\
   $|G_2|=1,200$],
)
#pause
#v(5pt)
#align(center, held-scene(mode: "truth"))

== A localization head shares evidence, then asks two questions #V

#align(center, grid(columns: (1fr, 18mm, 1fr), gutter: 10pt, align: horizon,
  card([SHARED LOCAL FEATURE], [$h=f_theta(x)$], color: TEAL),
  text(size: 27pt, fill: MUTED)[$arrow.r$],
  grid(columns: 1, gutter: 8pt,
    card([CLASS HEAD], [$z=W_c h+b_c$], color: ACC),
    card([BOX HEAD], [$t=W_b h+b_b in RR^4$], color: BLUE),
  ),
))
#pause
#v(6pt)
#two(
  [#text(size: 14pt, fill: ACC)[what is here?]],
  [#text(size: 14pt, fill: BLUE)[how should the reference move?]],
)
#pause
#note([Detection is multi-task learning repeated at spatial locations—not a classifier with four unexplained extra outputs.], color: ACC)

== Commit: what offsets move this anchor to its matched truth? #Q

#clock-strip("train")
#v(7pt)
#case-strip([positive anchor], [Reference $a=(50,50,40,40)$; matched target $b=(58,46,80,40)$, both in center-size form.])
#v(8pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([TRANSLATE], [$t_x=(x-x_a)/w_a$\ $t_y=(y-y_a)/h_a$], color: BLUE),
  hairline([RESIZE], [$t_w=log(w/w_a)$\ $t_h=log(h/h_a)$], color: TEAL),
))
#pause
#note([Calculate all four before the answer. Which offset must be zero? Which must be $log 2$?], color: BLUE)

== The anchor target is a scaled correction, not an absolute box #A

#clock-strip("train")
#v(7pt)
#two(
  [$t_x=(58-50)/40=0.20$\
   $t_y=(46-50)/40=-0.10$],
  [$t_w=log(80/40)=log 2 approx 0.693$\
   $t_h=log(40/40)=0$],
)
#pause
#v(8pt)
#align(center, text(size: 23pt, weight: 700, fill: ACC)[$t=(0.20,-0.10,0.693,0)$])
#pause
#result[Translations are measured in anchor sizes; log-scale changes keep decoded widths positive and make zero mean “unchanged.”]

== Decoding must recover the same geometry #D

#clock-strip("infer")
#v(7pt)
#two(
  [$hat(x)=x_a+t_x w_a=50+0.20(40)=58$\
   $hat(y)=y_a+t_y h_a=50-0.10(40)=46$],
  [$hat(w)=w_a exp(t_w)=40 exp(0.693) approx 80$\
   $hat(h)=h_a exp(t_h)=40 exp(0)=40$],
)
#pause
#v(8pt)
#align(center, diagram(spacing: (28mm, 9mm), {
  flow-node((0,0), [anchor $a$], color: TEAL)
  flow-node((1.7,0), [offsets $t$], color: ACC)
  flow-node((3.4,0), [decoded $hat(b)$], color: GREEN)
  flow-arrow((0,0),(1.7,0)); flow-arrow((1.7,0),(3.4,0), color: GREEN)
}))
#pause
#result[Encode and decode are inverse contracts; a unit test should recover $b$ before any model is trained.]

== A YOLO-style anchor head emits nine scalars per candidate #V

#clock-strip("train")
#v(7pt)
#case-strip([head convention], [For this teaching ledger only: one objectness logit + $K=4$ class logits + four box offsets.])
#v(8pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([OBJECTNESS], [one logit\ is anything here?], color: BLUE),
  hairline([CLASSES], [$K=4$ logits\ which class?], color: ACC),
  hairline([OFFSETS], [four numbers\ how should the anchor move?], color: GREEN),
))
#pause
#v(6pt)
#align(center, text(size: 22pt, weight: 700)[$1+K+4=1+4+4=9$])
#pause
#align(center, text(size: 13pt, fill: MUTED)[Teaching convention only: other heads may fold background into classes or predict box sides.])

== The dense tensor fixes capacity before the object count is known #D

#clock-strip("train")
#v(7pt)
#case-strip([tensor ledger], [$20 times 20$ locations, $A=3$ anchors per location, $K=4$ classes, nine outputs per anchor.])
#v(6pt)
#align(center, table(
  columns: (55mm, 55mm, 55mm), stroke: 0.45pt + MUTED,
  inset: (x: 10pt, y: 7pt), align: center,
  table.header([*quantity*], [*calculation*], [*count*]),
  [candidate boxes], [$20 dot 20 dot 3$], [$1,200$],
  [output scalars], [$20 dot 20 dot 3 dot 9$], [$10,800$],
))
#pause
#v(8pt)
#result[The head always emits $10,800$ scalars. Labels and post-processing decide which candidate records matter.]

== Toy assignment says which candidates receive which losses #D

#clock-strip("train")
#v(7pt)
For one truth, three anchors have IoUs $0.76$, $0.48$, and $0.12$. Use the declared *toy* policy $tau_+=0.70$, $tau_-=0.30$.
#v(7pt)
#align(center, table(
  columns: (30mm, 44mm, 78mm), stroke: 0.45pt + MUTED,
  inset: (x: 10pt, y: 6pt), align: (center, center, left),
  table.header([*IoU*], [*assignment*], [*loss terms*]),
  [$0.76$], [#text(fill: GREEN)[positive]], [objectness + class + box],
  [$0.48$], [#text(fill: MUTED)[ignore]], [no gradient in this toy policy],
  [$0.12$], [#text(fill: RED)[background]], [objectness only],
))
#pause
#v(7pt)
#note([Assignment is a training rule, not AP matching and not NMS. Real detectors use architecture-specific thresholds or dynamic matchers.], color: ACC)

== One positive candidate produces a fully auditable loss #D

#clock-strip("train")
#v(6pt)
Target $t=(.20,-.10,.693,0)$; prediction $hat(t)=(.25,-.05,.60,.05)$. Let $p_"obj"=.90$, $p_"cup"=.80$ and $lambda=2$.
#v(5pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 16pt,
  [#uncover("2-")[#hairline([OBJECTNESS], [$-log .90 approx .105$], color: BLUE)]],
  [#uncover("3-")[#hairline([CLASS], [$-log .80 approx .223$], color: ACC)]],
  [#uncover("4-")[#hairline([BOX $L_1$], [$|.05|+|.05|+|-.093|+|.05|=.243$], color: GREEN)]],
))
#uncover("5-")[#align(center, text(size: 20pt, weight: 700, fill: ACC)[$cal(L)=.105+.223+2(.243) approx .815$])]
#uncover("6-")[#result[Write the target, parameterization, reduction, and $lambda$. “Box loss” without those contracts is not reproducible.]]

// ───────────────────────────── 3 · IoU ─────────────────────────────
= IoU makes box quality geometric

== IoU asks how much two regions agree #V

#case-strip([same I1 pair], [$A=G_1=[10,10,50,50]$ and $B=[12,12,52,52]$ are affinely placed on the real Abyssinian head ROI.])
#v(4pt)
#align(center, evidence-image("real-candidates.png", width: 126mm, height: 70.9mm))
#pause
#align(center, text(size: 23pt)[$"IoU"(A,B)=frac(|A inter B|,|A union B|) in [0,1]$])
#pause
#v(6pt)
#align(center, text(size: 15pt, fill: MUTED)[$0$: disjoint $quad$ $1$: identical $quad$ invariant to scaling the whole scene])

== The held overlap is $0.822$, not a visual guess #D

#case-strip([same pair], [$A=[10,10,50,50]$, $B=[12,12,52,52]$; both have area $1,600$.])
#v(6pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  [#uncover("2-")[$w_I=min(50,52)-max(10,12)=38$]
   #v(5pt)
   #uncover("3-")[$h_I=min(50,52)-max(10,12)=38$]],
  [#uncover("4-")[$A_I=38 dot 38=1,444$]
   #v(5pt)
   #uncover("5-")[$A_U=1,600+1,600-1,444=1,756$]],
))
#uncover("6-")[#align(center, text(size: 23pt, weight: 700, fill: ACC)[$"IoU"(A,B)=frac(1 444, 1 756) approx 0.822$])]
#uncover("7-")[#result[The same $0.822$ will later justify suppressing B around kept box A.]]

== Safe IoU clamps disjoint intersections to zero #D

#two(
  [For each axis,
   $w_I=max(0,min(x_1^A,x_1^B)-max(x_0^A,x_0^B))$
   $h_I=max(0,min(y_1^A,y_1^B)-max(y_0^A,y_0^B))$],
  [Then,
   $A_I=w_I h_I$
   $A_U=A_A+A_B-A_I$
   $"IoU"=A_I/A_U$],
)
#pause
#v(9pt)
#alertbox[Without $max(0,dot)$, two disjoint boxes can produce two negative side lengths—and a meaningless positive “intersection.”]
#pause
#note([Also assert valid boxes: $x_1 >= x_0$, $y_1 >= y_0$, and define the zero-union case explicitly.], color: BLUE)

== Equal coordinate error can mean unequal overlap #Q

Two predictions move one boundary by $10$ pixels.
#v(8pt)
#two(
  [*Large object:* $200 times 200$\
   a $10$-pixel error is $5%$ of its width.],
  [*Small object:* $20 times 20$\
   the same error is $50%$ of its width.],
)
#pause
#v(10pt)
#alertbox[Coordinate losses see four residuals. IoU sees the region those residuals create.]
#pause
#result[Use coordinate losses for a stable regression signal; report overlap-aware metrics for the geometry the task actually values.]

== Smooth $L_1$ and GIoU repair different failures #D

#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([SMOOTH $L_1$ · $beta=1$], [quadratic near $0$; linear for large residuals\ $0.2 arrow.r 0.02$; $3 arrow.r 2.5$], color: BLUE),
  hairline([GIoU], [$"GIoU"="IoU"-frac(|C \ (A union B)|,|C|)$\ $C$: smallest enclosing box], color: TEAL),
))
#pause
#v(9pt)
#note([Plain $1-"IoU"$ is flat while boxes are disjoint. GIoU changes with the enclosing gap; Smooth $L_1$ instead controls coordinate-residual robustness.], color: GREEN)
#pause
#align(center, text(size: 13pt, fill: MUTED)[#link(GIOU)[Rezatofighi et al. · Generalized IoU]])

// ───────────────────────────── 4 · TRAIN ─────────────────────────────
= TRAIN: dense candidates need selective supervision

== Dense heads create a foreground–background imbalance #V

#clock-strip("train")
#v(7pt)
#align(center, diagram(spacing: (13mm, 10mm), {
  for x in range(8) {
    for y in range(3) {
      round-node((x,y), if x == 4 and y == 1 { [obj] } else { [bg] },
        color: if x == 4 and y == 1 { GREEN } else { MUTED },
        fill: if x == 4 and y == 1 { GREEN.lighten(90%) } else { rgb("#F4F2EE") }, r: 6.3mm)
    }
  }
}))
#pause
#v(7pt)
#alertbox[If thousands of easy background candidates contribute ordinary cross-entropy, their count can drown out the rare positives and hard negatives.]

== Focal loss turns confidence into a gradient-volume control #D

#clock-strip("train")
#v(7pt)
#align(center, text(size: 23pt)[$"FL"(p_t)=-alpha_t(1-p_t)^gamma log p_t$])
#pause
#v(8pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([$p_t$], [probability assigned to the true binary label], color: TEAL),
  hairline([$gamma$], [how strongly easy examples are down-weighted], color: BLUE),
  hairline([$alpha_t$], [optional class-balance weight], color: ACC),
))
#pause
#v(7pt)
#note([The next numeric sets $alpha_t=1$ and $gamma=2$ to isolate the focusing factor.], color: MUTED)

== With $gamma=2$, the hard example is about $978 times$ louder #D

#clock-strip("train")
#v(7pt)
#two(
  [*Easy:* $p_t=.9$\
   $"CE"=-log .9 approx .10536$\
   $(1-p_t)^2=.01$\
   $"FL" approx .0010536$],
  [*Hard:* $p_t=.2$\
   $"CE"=-log .2 approx 1.60944$\
   $(1-p_t)^2=.64$\
   $"FL" approx 1.03004$],
)
#pause
#align(center, text(size: 22pt, weight: 700, fill: ACC)[$1.03004/.0010536 approx 977.6 approx 978 times$])
#pause
#result[Focal loss does not delete easy examples; it makes already-correct examples quiet enough that rare mistakes remain audible.]

== Diagnose training failures as symptom → suspect → test #V

#clock-strip("train")
#v(6pt)
#set text(size: 13.5pt)
#align(center, table(
  columns: (57mm, 59mm, 71mm), stroke: 0.45pt + MUTED,
  inset: (x: 7pt, y: 5pt), align: (left, left, left),
  table.header([*symptom*], [*suspect*], [*smallest discriminating test*]),
  [box loss falls; IoU stays near $0$], [box format / decode mismatch], [round-trip one known anchor and draw decoded boxes],
  [almost no positive anchors], [assignment geometry too strict], [histogram best IoU per truth; inspect $tau_+$],
  [objectness predicts background], [easy negatives dominate], [log pos/neg loss mass; compare focal or sampling],
))
#pause
#result[Inspect geometry and assignment before changing the backbone: a model cannot learn targets that the pipeline encoded incorrectly.]

== Follow along: boxes, IoU, matching, and anchor round-trips #I

#clock-strip("train")
#v(7pt)
#interbox(link-to: NB-IOU)[
  Open the exact Colab. Predict the $frac(1 444, 1 756)$ IoU, implement the clamp, verify the anchor encode/decode round-trip, and build the held matching ledger before reading its outputs.
]
#pause
#v(8pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([BOX], [$[x_0,y_0,x_1,y_1]$; no $+1$], color: TEAL),
  hairline([OVERLAP], [$"IoU"(A,B) approx .822$], color: BLUE),
  hairline([ANCHOR], [$t=(.20,-.10,.693,0)$], color: ACC),
))

== TRAIN closes with targets; inference opens without them #V

#clock-strip("train")
#v(8pt)
#align(center, diagram(spacing: (4mm, 10mm), {
  flow-node((0,0), [$20 times 20 times 3 times 9$\ raw tensor], color: TEAL, w: 40mm)
  flow-node((1.7,0), [assign\ targets], color: ACC, w: 30mm)
  flow-node((3.3,0), [object + class + box\ losses], color: RED, w: 45mm)
  flow-node((5.1,0), [update\ parameters], color: GREEN, w: 30mm)
  flow-arrow((0,0),(1.7,0)); flow-arrow((1.7,0),(3.3,0)); flow-arrow((3.3,0),(5.1,0))
}))
#pause
#v(8pt)
#clock-strip("infer")
#pause
#result[At inference there is no truth, no assignment, and no loss—only decoded boxes, scores, thresholds, and prediction-to-prediction suppression.]

// ───────────────────────────── 5 · INFER ─────────────────────────────
= INFER: score first, then suppress duplicates

== The constructed raw ledger isolates the inference rule #V

#clock-strip("infer")
#v(6pt)
#align(center, held-scene(mode: "raw", show-truth: false))
#v(4pt)
#align(center, text(size: 14pt)[A $.95$ #h(18pt) B $.90$ #h(18pt) E $.89$ #h(18pt) C $.88$ #h(18pt) D $.75$])
#pause
#note([A/B/E share (I1, head); C/D share (I2, head). These scores are pedagogical constructions, not model output. INFER sees neither XML truth nor TP/FP labels.], color: BLUE)

== Score thresholding removes weak evidence—not duplicates #Q

#clock-strip("infer")
#v(7pt)
Use a deliberately permissive score threshold $s >= .70$.
#v(7pt)
#align(center, table(
  columns: (22mm, 22mm, 22mm, 22mm, 22mm), stroke: 0.45pt + MUTED,
  inset: (x: 12pt, y: 8pt), align: center,
  [A $.95$], [B $.90$], [E $.89$], [C $.88$], [D $.75$],
))
#pause
#v(9pt)
#align(center, text(size: 22pt, weight: 700, fill: RED)[All five survive.])
#pause
#result[Thresholding can drop low scores, but it cannot tell that two high-scoring boxes describe the same object.]

== NMS is a greedy prediction-to-prediction comparison #D

#clock-strip("infer")
#v(7pt)
#align(center, diagram(spacing: (4mm, 10mm), {
  flow-node((0,0), [sort by\ score], color: BLUE, w: 29mm)
  flow-node((1.6,0), [keep top\ box], color: GREEN, w: 29mm)
  flow-node((3.4,0), [remove same-class neighbours\ with IoU $>tau_"NMS"$], color: RED, w: 51mm)
  flow-node((5.5,0), [repeat], color: ACC, w: 25mm)
  flow-arrow((0,0),(1.6,0)); flow-arrow((1.6,0),(3.4,0)); flow-arrow((3.4,0),(5.5,0))
}))
#pause
#v(8pt)
#note([No ground truth participates. Standard NMS has no learned parameter and no gradient; here $tau_"NMS"=.50$.], color: ACC)

== Step 1: A claims its high-overlap neighbourhood #D

#clock-strip("infer")
#v(5pt)
#align(center, held-scene(mode: "step1", show-truth: false))
#pause
#align(center, text(size: 19pt)[$"IoU"(A,B)=.822>.50 arrow.r$ keep A, suppress B])
#pause
#v(5pt)
#result[E, C, and D remain because none overlaps A above the NMS threshold.]

== Step 2: E survives because NMS cannot test correctness #D

#clock-strip("infer")
#v(5pt)
#align(center, held-scene(mode: "step2", show-truth: false))
#pause
#align(center, text(size: 19pt)[$E=.89$ is now the highest remaining score; its IoU with every survivor is below $.50$.])
#pause
#v(5pt)
#alertbox[Keep E. NMS removes duplicates around a stronger prediction; it does not remove isolated false alarms.]

== Step 3: C claims the I2 head neighbourhood #D

#clock-strip("infer")
#v(6pt)
#case-strip([last pair], [$C=[60,15,90,55]$ and $D=[63,17,91,56]$.])
#v(6pt)
$A_I=(90-63)(55-17)=27 dot 38=1,026$
#pause
$A_U=1,200+1,092-1,026=1,266$
#pause
#align(center, text(size: 22pt, weight: 700, fill: ACC)[$"IoU"(C,D)=frac(1 026, 1 266) approx .810>.50$])
#pause
#result[Keep C; suppress D. No candidates remain.]

== NMS returns A, E, C—not “the correct boxes” #A

#clock-strip("infer")
#v(3pt)
#align(center, evidence-image("real-nms-trace.png", width: 153mm, height: 86.1mm))
#pause
#note([NMS is grouped by $("image_id", "class")$: A can never suppress C across images. The global returned score order is A, E, C; correctness waits for EVAL.], color: ACC)

== Class-aware NMS prevents a cup from deleting a hand #Q

#clock-strip("infer")
#v(7pt)
#align(center, diagram(spacing: (28mm, 11mm), {
  round-node((0,0), [cup\ $.91$], color: TEAL)
  round-node((1.8,0), [hand\ $.89$], color: ACC)
  flow-arrow((0,0),(1.8,0), color: MUTED, label: [same location])
}))
#pause
#v(8pt)
#result[Usually compare boxes within the same class. Cross-class overlap may describe two real, interacting objects.]
#pause
#note([A low NMS threshold can still merge two nearby objects of the *same* class. Increasing $tau_"NMS"$ often keeps more boxes for a fixed score order, but keep-count is not a global monotonic law across changed scores/classes/pipelines.], color: RED)

== Follow along: print every NMS decision, then break it #I

#clock-strip("infer")
#v(7pt)
#interbox(link-to: NB-NMS)[
  Open the exact Colab. Predict A, E, C; print the keep/suppress trace; verify class-aware NMS; then construct two distinct same-class objects that a low threshold wrongly merges.
]
#pause
#v(8pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([KEEP], [A $.95$, E $.89$, C $.88$], color: GREEN),
  hairline([SUPPRESS], [B by A; D by C], color: RED),
  hairline([LIMIT], [isolated false alarm E survives], color: ACC),
))

// ───────────────────────────── 6 · EVAL ─────────────────────────────
= EVAL: match the returned set to truth, one claim at a time

== Evaluation matching is not training assignment #V

#clock-strip("eval")
#v(7pt)
#align(center, diagram(spacing: (4mm, 10mm), {
  flow-node((0,0), [score-sorted\ detections], color: BLUE, w: 34mm)
  flow-node((1.8,0), [same class +\ IoU $>=tau_"eval"$], color: ACC, w: 41mm)
  flow-node((3.7,0), [claim one\ unmatched truth], color: GREEN, w: 36mm)
  flow-node((5.5,-0.6), [TP], color: GREEN, w: 23mm)
  flow-node((5.5,0.6), [FP], color: RED, w: 23mm)
  flow-arrow((0,0),(1.8,0)); flow-arrow((1.8,0),(3.7,0));
  flow-arrow((3.7,0),(5.5,-0.6), color: GREEN); flow-arrow((3.7,0),(5.5,0.6), color: RED)
}))
#pause
#v(7pt)
#result[At $tau_"eval"=.50$, each truth may be claimed once. A duplicate would be an FP even if its IoU were high.]

== Commit: which of A, E, C are true positives? #Q

#clock-strip("eval")
#v(5pt)
#align(center, held-scene(mode: "final"))
#pause
#align(center, table(
  columns: (30mm, 35mm, 57mm, 43mm), stroke: 0.45pt + MUTED,
  inset: (x: 10pt, y: 6pt), align: center,
  table.header([*detection*], [*score*], [*best same-class IoU*], [*commit*]),
  [A], [$.95$], [$1.000$ with $G_1$], [$?$],
  [E], [$.89$], [$.127$ with $G_1$], [$?$],
  [C], [$.88$], [$1.000$ with $G_2$], [$?$],
))
#pause
#note([Match only within image_id, then concatenate survivors into global score order. A truth claimed by an earlier detection is unavailable to later detections in that image.], color: BLUE)

== Evaluation exposes the false alarm NMS could not see #A

#clock-strip("eval")
#v(3pt)
#align(center, evidence-image("real-eval-matching.png", width: 153mm, height: 86.1mm))
#pause
#note([GT is revealed only now: A claims I1/G1, E is an I1 false alarm at $9/71$, and C claims I2/G2. Two TPs, one FP, zero FNs.], color: ACC)

== Precision and recall evolve with the score ranking #D

#clock-strip("eval")
#v(6pt)
#align(center, table(
  columns: (31mm, 30mm, 30mm, 39mm, 39mm), stroke: 0.45pt + MUTED,
  inset: (x: 9pt, y: 6pt), align: center,
  table.header([*after*], [*cum. TP*], [*cum. FP*], [*precision*], [*recall*]),
  [A], [$1$], [$0$], [$1/1=1.000$], [$1/2=.500$],
  [#uncover("2-")[E]], [#uncover("2-")[$1$]], [#uncover("2-")[$1$]], [#uncover("2-")[$1/2=.500$]], [#uncover("2-")[$1/2=.500$]],
  [#uncover("3-")[C]], [#uncover("3-")[$2$]], [#uncover("3-")[$1$]], [#uncover("3-")[$2/3=.667$]], [#uncover("3-")[$2/2=1.000$]],
))
#uncover("4-")[#v(7pt)
#result[Lowering the score threshold admitted E before C: precision fell, then recall rose when C arrived.]]

== The toy AP convention averages precision at TP ranks #D

#clock-strip("eval")
#v(7pt)
#two(
  [#align(center, lines(
    (
      ((0,1),(0.5,1),(0.5,0.5),(1,0.667)),
      ((0,0),(1,0)),
    ),
    markers: (true, false),
    colors: (INK, INK.transparentize(100%)),
    points: ((0.5,1,[A]), (0.5,0.5,[E]), (1,0.667,[C])),
    size: (74mm,46mm), x-label: [recall], y-label: [precision],
  ))],
  [At the two TP ranks:
   #v(7pt)
   $p_A=1$\
   $p_C=2/3$
   #v(7pt)
   $"AP"_"toy"=frac(1+2/3,2)=5/6 approx .833$],
  r: (1.05fr, .95fr),
)
#pause
#align(center, text(size: 12.5pt, fill: MUTED)[After NMS: A, E, C; B and D were suppressed before evaluation. Axes run from $0$ to $1$.])
#v(5pt)
#result[This TP-rank average is a transparent teaching convention. It is not a claim about every benchmark’s interpolation rule.]

== COCO AP fixes the full evaluation protocol #D

#clock-strip("eval")
#v(7pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 17pt,
  hairline([LOCALIZATION], [10 IoU thresholds $.50,.55,dots,.95$], color: TEAL),
  hairline([INTERPOLATION], [101 recall thresholds $0,.01,dots,1$], color: BLUE),
  hairline([AGGREGATION], [average over classes; declare area + max detections], color: ACC),
))
#pause
#v(8pt)
#note([Say AP50 for one IoU cutoff. “COCO AP” normally means the average across IoU thresholds; the official API also declares area ranges and maxDets.], color: GREEN)
#pause
#align(center, text(size: 13pt, fill: MUTED)[#link("https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py")[Official COCO evaluation implementation]])

== Diagnose metric surprises on the correct clock #V

#clock-strip("eval")
#v(6pt)
#set text(size: 13.5pt)
#align(center, table(
  columns: (58mm, 57mm, 72mm), stroke: 0.45pt + MUTED,
  inset: (x: 7pt, y: 5pt), align: (left, left, left),
  table.header([*symptom*], [*suspect*], [*smallest discriminating test*]),
  [high scores; low AP], [ranking is confidently wrong], [inspect first FP for class, IoU, duplicate status],
  [AP50 high; COCO AP low], [boxes are loose], [compare AP50 vs AP75; histogram matched IoU],
  [recall caps below $1$], [thresholding / missing candidates], [lower score threshold before NMS; count unmatched truths],
))
#pause
#result[Loss, NMS, and AP expose different failure surfaces. Name the clock before changing a threshold.]

// ───────────────────────────── 7 · DETECTOR FAMILIES ─────────────────────────────
= Architectures rearrange the same evidence

== Two-stage means propose, then refine #V

#align(center, diagram(spacing: (4mm, 11mm), {
  flow-node((0,0), [image], fill: rgb("#EFEEEB"), w: 25mm)
  flow-node((1.5,0), [shared\ backbone], color: TEAL, w: 31mm)
  flow-node((3,-0.65), [RPN\ object proposals], color: GREEN, w: 38mm)
  flow-node((3,0.65), [feature map], color: TEAL, w: 30mm)
  flow-node((4.8,0), [RoI\ features], color: BLUE, w: 29mm)
  flow-node((6.3,0), [class +\ refined box], color: ACC, w: 35mm)
  flow-arrow((0,0),(1.5,0)); flow-arrow((1.5,0),(3,-0.65)); flow-arrow((1.5,0),(3,0.65));
  flow-arrow((3,-0.65),(4.8,0)); flow-arrow((3,0.65),(4.8,0)); flow-arrow((4.8,0),(6.3,0))
}))
#pause
#v(8pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([RPN], [Region Proposal Network: predicts objectness + proposal boxes densely.], color: GREEN),
  hairline([RoI], [Region of Interest: samples a fixed-size feature for each proposal.], color: BLUE),
))
#pause
#result[Stage 1 asks “where might an object be?” Stage 2 spends more computation to classify and refine a smaller proposal set.]

== One-stage predicts every candidate directly #V

#align(center, diagram(spacing: (4mm, 11mm), {
  flow-node((0,0), [image], fill: rgb("#EFEEEB"), w: 25mm)
  flow-node((1.7,0), [backbone +\ feature pyramid], color: TEAL, w: 42mm)
  flow-node((3.6,0), [dense heads\ scores + boxes], color: ACC, w: 43mm)
  flow-node((5.3,0), [NMS], color: GREEN, w: 24mm)
  flow-node((6.5,0), [detections], color: BLUE, w: 31mm)
  flow-arrow((0,0),(1.7,0)); flow-arrow((1.7,0),(3.6,0)); flow-arrow((3.6,0),(5.3,0)); flow-arrow((5.3,0),(6.5,0))
}))
#pause
#v(9pt)
#result[There is no learned proposal stage or per-RoI second head: classification and localization happen densely in one pass.]
#pause
#note([One-stage versus two-stage describes where proposal/refinement computation occurs—not a timeless guarantee about speed or accuracy.], color: MUTED)

== FPN gives small and large objects strong features #V

#align(center, diagram(spacing: (16mm, 8mm), {
  let feat(p, body, col) = node(p, body, fill: col.lighten(87%), stroke: 0.9pt + col, corner-radius: 3pt, inset: 6pt)
  feat((0,2.1), [$C_2: 56 times 56$], BLUE)
  feat((0,0.7), [$C_3: 28 times 28$], TEAL)
  feat((0,-0.7), [$C_4: 14 times 14$], GREEN)
  feat((0,-2.1), [$C_5: 7 times 7$], ACC)
  feat((2.2,2.1), [$P_2$], BLUE); feat((2.2,0.7), [$P_3$], TEAL)
  feat((2.2,-0.7), [$P_4$], GREEN); feat((2.2,-2.1), [$P_5$], ACC)
  for y in (2.1,0.7,-0.7,-2.1) { edge((0,y),(2.2,y),"-|>",stroke:0.8pt+MUTED) }
  edge((2.2,-2.1),(2.2,-0.7),"-|>",stroke:1pt+ACC)
  edge((2.2,-0.7),(2.2,0.7),"-|>",stroke:1pt+GREEN)
  edge((2.2,0.7),(2.2,2.1),"-|>",stroke:1pt+TEAL)
}))
#pause
#v(7pt)
#align(center, text(size: 16pt)[FPN = *Feature Pyramid Network*: lateral $1 times 1$ projections + top-down upsampling + same-shape addition.])
#pause
#result[High-resolution levels localize small objects; top-down semantics make those levels useful for recognition.]

== Anchor-free heads change the box parameterization #V

#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([ANCHOR-BASED], [start from $(x_a,y_a,w_a,h_a)$; regress normalized translations and log scales], color: TEAL),
  hairline([ANCHOR-FREE], [predict a center/keypoint or distances to left, top, right, bottom sides], color: BLUE),
))
#pause
#v(10pt)
#result[Both still need spatial features, class evidence, box geometry, a training matcher, and an evaluation protocol.]

== Duplicate handling is a design choice, not a law #V

#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([SOFT-NMS], [decay a neighbour’s score instead of deleting it; nearby objects may survive], color: BLUE),
  hairline([SET PREDICTION], [train a fixed query set with one-to-one assignment; duplicate avoidance moves into learning], color: ACC),
))
#pause
#v(9pt)
#note([These alternatives change inference and training, but the evaluation clock still matches scored predictions to truths under a declared protocol.], color: GREEN)

// ───────────────────────────── 8 · SYNTHESIS ─────────────────────────────
= One pipeline, three clocks, one returned set

== The complete pipeline changes what IoU is comparing #V

#align(center, diagram(spacing: (3mm, 11mm), {
  flow-node((0,0), [TRAIN\ assign + losses], color: TEAL, w: 36mm)
  flow-node((1.7,0), [dense\ tensor], color: INK, w: 28mm)
  flow-node((3.2,0), [INFER\ decode + score + NMS], color: BLUE, w: 45mm)
  flow-node((5.2,0), [A, E, C], color: GREEN, w: 28mm)
  flow-node((6.7,0), [EVAL\ match + AP], color: INK, w: 35mm)
  flow-arrow((0,0),(1.7,0)); flow-arrow((1.7,0),(3.2,0)); flow-arrow((3.2,0),(5.2,0)); flow-arrow((5.2,0),(6.7,0))
}))
#pause
#v(8pt)
#align(center, table(
  columns: (40mm, 53mm, 69mm), stroke: 0.45pt + MUTED,
  inset: (x: 8pt, y: 5pt), align: (center, center, left),
  table.header([*clock*], [*IoU compares*], [*decision*]),
  [TRAIN], [candidate ↔ truth], [which targets and losses apply],
  [INFER], [prediction ↔ prediction], [which neighbours are redundant],
  [EVAL], [prediction ↔ truth], [TP / FP / FN and AP],
))
#pause
#result[The formula is the same; the objects, threshold, and consequence are not.]

== Revisit the opening commitment before revealing it #Q

#case-strip([same five candidates], [A $.95$, B $.90$, E $.89$, C $.88$, D $.75$; $tau_"NMS"=.50$, $tau_"eval"=.50$.])
#v(3pt)
#align(center, evidence-image("real-candidates.png", width: 94mm, height: 52.9mm))
#pause
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 16pt,
  hairline([A · THRESHOLD], [all five pass $.70$], color: INK),
  hairline([B · GEOMETRY], [decode → score → NMS → evaluation match], color: INK),
  hairline([C · LOSS], [requires truths unavailable at inference], color: INK),
))
#pause
#note([Defend your original choice using the exact boxes and clocks—not a detector name.], color: BLUE)

== Answer: B is complete—and still cannot remove E #A

#align(center, grid(columns: (1fr, 16mm, 1fr, 16mm, 1fr), gutter: 6pt, align: horizon,
  card([RAW], [A, B, E, C, D], color: BLUE),
  text(size: 24pt, fill: MUTED)[$arrow.r$],
  card([NMS], [A, E, C], color: GREEN),
  text(size: 24pt, fill: MUTED)[$arrow.r$],
  card([EVAL], [A = TP\ E = FP\ C = TP], color: INK),
))
#pause
#v(9pt)
#align(center, text(size: 21pt, weight: 700)[$"AP"_"toy"=5/6 approx .833$])
#pause
#result[Thresholds control admission, NMS controls redundancy, and matching measures correctness. No one operation can substitute for the others.]

== Final detection checklist #Q

Given a new detector, ask in order:
#v(4pt)
1. What box coordinate and area convention does every component use?
2. What is the dense output shape, and what does each scalar mean?
3. Which TRAIN matcher creates positive, ignored, and background targets?
4. Which loss reduction and weights determine gradient mass?
5. Which INFER score and NMS policies create the returned set?
6. Which EVAL IoU, interpolation, class, area, and max-detection protocol defines AP?
7. Which symptom → suspect → test check will distinguish geometry from learning failure?
#pause
#v(6pt)
#result[A detector is trustworthy only when the tensor, geometry, three clocks, and metric protocol are all explicit.]

== L11 handoff: boxes become a prediction at every pixel #V

#align(center, grid(columns: (1fr, 18mm, 1fr), gutter: 10pt, align: horizon,
  card([DETECTION], [variable set\ class + box + score], color: BLUE),
  text(size: 27pt, fill: MUTED)[$arrow.r$],
  card([SEGMENTATION], [dense spatial output\ class evidence at each pixel], color: ACC),
))
#pause
#v(9pt)
#note([Both reuse the L8B backbone. L11 will replace four-number rectangles with masks, then recover boundaries that pooling made coarse.], color: GREEN)
#pause
#result[Today asked “which objects, and where?” Next asks “which class owns every pixel—and which instance?”]

== Primary provenance and exact companions #V

#set text(size: 13.5pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 20pt,
  hairline([DETECTOR FAMILIES], [
    #link(RCNN)[Girshick et al. · R-CNN]\
    #link(FAST)[Girshick · Fast R-CNN]\
    #link(FASTER)[Ren et al. · Faster R-CNN]\
    #link(YOLO)[Redmon et al. · YOLO]
  ], color: TEAL),
  hairline([LOSSES, PYRAMIDS + CODE], [
    #link(FOCAL)[Lin et al. · Focal Loss / RetinaNet]\
    #link(FPN)[Lin et al. · Feature Pyramid Networks]\
    #link(GIOU)[Rezatofighi et al. · GIoU]\
    #link(NB-IOU)[Exact boxes / IoU Colab] · #link(NB-NMS)[Exact NMS Colab]
  ], color: BLUE),
))
#pause
#v(5pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 20pt,
  hairline([REAL EVIDENCE], [
    #link(PETS)[Oxford-IIIT Pet] · Parkhi et al. (CVPR 2012)\
    I1 Abyssinian_1 · I2 Bengal_10\
    official XML tight-head ROIs · CC BY-SA 4.0
  ], color: GREEN),
  hairline([REPRODUCIBILITY], [
    `shared/vision-evidence/oxford-iiit-pet/l9/`\
    builder + manifest + CSV decision ledgers\
    A–E/scores = constructed · model output = none
  ], color: ACC),
))
#pause
#v(4pt)
#caption[OBSERVED = photos/XML · CONSTRUCTED = crops/candidates/scores · COMPUTED = IoU/NMS/matching/AP]
