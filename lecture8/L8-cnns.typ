// Convolutional Neural Networks — Lecture 8 · ES 667 Deep Learning
// Compile from the repository root:
//   typst compile --root . --input handout=true lecture8/L8-cnns.typ /tmp/L8-handout.pdf
//   typst compile --root . lecture8/L8-cnns.typ /tmp/L8-presentation.pdf

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *

#show: metropolis-deck.with(
  title: [Convolutional Neural Networks],
  subtitle: [One learned local question, asked everywhere],
)

#let NB1 = "https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L08/01_convolution_shapes_params_from_scratch.ipynb"
#let PTCONV = "https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html"
#let CONVARITH = "https://arxiv.org/abs/1603.07285"
#let LENET = "https://ieeexplore.ieee.org/document/726791"

// Stable semantic grammar:
// INK = input / known quantity · TEAL = learned shared parameter / operation
// BLUE = spatial geometry / index · ACC = computed response / prediction
// GREEN = verified conclusion · RED = failure / invalid assumption.
#let swatch(color, body) = box(width: 13pt, height: 4pt, fill: color, radius: 1pt) + h(4pt) + body
#let semantic-legend = align(center, text(size: 12.5pt)[
  #swatch(INK, [input]) #h(12pt)
  #swatch(TEAL, [shared parameter]) #h(12pt)
  #swatch(BLUE, [geometry]) #h(12pt)
  #swatch(ACC, [response]) #h(12pt)
  #swatch(GREEN, [verified]) #h(12pt)
  #swatch(RED, [failure])
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
  width: 100%, inset: (left: 11pt, right: 4pt, top: 5pt, bottom: 5pt),
  stroke: (left: 2pt + color), fill: color.lighten(93%), radius: 2pt,
  [#body],
)

#let caption(body) = align(center, text(size: 14pt, fill: MUTED)[#body])

#let flow-arrow(label: none, color: MUTED) = align(center, [
  #if label != none { text(size: 10.5pt, weight: 600, fill: color, label); linebreak() }
  #text(size: 24pt, fill: color)[$arrow.r$]
])

#let tensor-grid(values, title: none, cell: 9mm, highlight: (), vmin: auto, vmax: auto) = {
  grid-map(values, title: title, cell: cell, highlight: highlight,
    vmin: vmin, vmax: vmax, frame: true)
}

#let case-strip(stage, detail) = block(
  width: 100%, inset: (x: 10pt, y: 6pt),
  fill: rgb("#F3F6F5"), stroke: (left: 3pt + TEAL), radius: 3pt,
  [#text(size: 11pt, weight: 700, fill: TEAL, tracking: 0.7pt)[RUNNING CASE · #upper(stage)]
   #h(8pt)
   #text(size: 13.5pt, fill: INK)[#detail]],
)

// One exact image/kernel pair anchors convolution, geometry, equivariance,
// pooling, and the first receptive-field calculation.
#let X = ((0, 0, 0, 0, 0),
          (0, 0, 0, 0, 0),
          (1, 1, 1, 1, 1),
          (1, 1, 1, 1, 1),
          (0, 0, 0, 0, 0))
#let K = ((1, 1, 1),
          (0, 0, 0),
          (-1, -1, -1))
#let Yvalid = ((-3, -3, -3),
               (-3, -3, -3),
               (3, 3, 3))
#let Ypadded = ((0, 0, 0, 0, 0),
                (-2, -3, -3, -3, -2),
                (-2, -3, -3, -3, -2),
                (2, 3, 3, 3, 2),
                (2, 3, 3, 3, 2))
#let Ystride = ((0, 0, 0),
                (-2, -3, -2),
                (2, 3, 2))
#let Panchor = ((-3, -3), (3, 3))

#title-slide()

// ═════════════════════════ OPENING · CORE ═══════════════════════════
= Architecture can regularize before training begins

== L7 changed the training recipe; L8 changes the hypothesis class #V

#align(center, grid(
  columns: (55mm, 12mm, 55mm, 12mm, 55mm), gutter: 5pt, align: horizon,
  card([L7 · GENERALIZE], [Keep train/validation/test roles fixed; change one intervention.], color: ACC),
  flow-arrow(label: [today], color: BLUE),
  card([IMAGE STRUCTURE], [Nearby pixels interact; useful patterns repeat across location.], color: INK),
  flow-arrow(label: [encode], color: TEAL),
  card([L8 · CNN], [Local connections and shared weights make those rules easy to express.], color: TEAL),
))

#pause
#v(10pt)
#note([
  The loss, optimizer, split protocol, and sealed test stay fixed. The new lever is architectural inductive bias.
  You know convolution mechanics from ES 335; today we audit the assumptions, arithmetic, and gradients.
], color: GREEN)

== Commit: can ten parameters produce 1,024 responses? #Q

A $32 times 32$ grayscale image must produce one $32 times 32$ response map.

#v(8pt)
#mcq(
  [Which claim can be true? Keep your choice until the closing revisit.],
  [Ten parameters can produce only ten output values],
  [Ten shared parameters can produce 1,024 values by reusing one local rule],
  [Any ten-parameter map can reproduce an arbitrary dense layer],
  [Parameter sharing makes the computation independent of image size],
)

#pause
#note([Commit to A, B, C, or D, then write the assumption your choice requires.], color: BLUE)

== The student contract and 80-minute route #V

By the end, given a small image/kernel and a tiny CNN, you should be able to compute its responses, shapes, receptive fields, parameters, MACs, and one shared-weight gradient.

#v(8pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([CORE · 55 min], [local/shared operator → geometry → equivariance/pooling → RF → classifier ledger], color: TEAL),
  hairline([SHOULD COVER · 15 min], [shared-weight backprop → NCHW/mode checks → diagnosis], color: BLUE),
  hairline([OPTIONAL · 10 min], [small-vs-large kernels → im2col → provenance/notebook traces], color: MUTED),
))

#v(8pt)
#semantic-legend

== Flattening preserves values but removes the useful constraint #V

#align(center, grid(
  columns: (54mm, 12mm, 54mm, 12mm, 54mm), gutter: 5pt, align: horizon,
  card([IMAGE], [$X in RR^(32 times 32)$; neighbours are explicit.], color: INK),
  flow-arrow(label: [flatten], color: BLUE),
  card([VECTOR], [$x in RR^1024$; the same values remain.], color: BLUE),
  flow-arrow(label: [dense], color: RED),
  card([POSITION-SPECIFIC], [Every output may learn an unrelated weight for every input.], color: RED),
))

#pause
#v(10pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([LOCALITY], [look at a small neighbourhood], color: BLUE),
  hairline([SHARING], [reuse the same detector everywhere], color: TEAL),
  hairline([HIERARCHY], [compose local evidence through depth], color: ACC),
))

== Compare like with like: include one bias per output #D

Both maps produce $32 times 32=1024$ scalar outputs. The shared $3 times 3$ map therefore uses $p=1$, $s=1$.

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  card([DENSE MAP], [
    weights: $1024 times 1024$ \
    biases: $1024$ \
    total: $"1,049,600"$
  ], color: RED),
  card([SHARED $3 times 3$ MAP], [
    weights: $3 times 3=9$ \
    one shared bias: $1$ \
    total: $10$
  ], color: TEAL),
))

#pause
#align(center, text(size: 23pt, weight: 700, fill: GREEN)[$"1,049,600" / 10 = "104,960" times$ fewer parameters])

#pause
#note([The saving comes from two restrictions together: local connectivity and the same weights at every output location.], color: GREEN)

// ═══════════════════════ CONVOLUTION · CORE ═════════════════════════
= One local question, reused across the image

== Meet the fixed $5 times 5$ image and $3 times 3$ kernel #V

#case-strip([input + parameter], [The image $X$ and kernel $K$ remain fixed through convolution, geometry, pooling, and receptive fields.])
#v(7pt)
#two(
  [#align(center, tensor-grid(X, title: [$X$: input], cell: 10.5mm, vmin: -1, vmax: 1))],
  [#align(center, tensor-grid(K, title: [$K$: shared kernel], cell: 12mm, vmin: -1, vmax: 1))
   #v(6pt)
   #caption[INK = input values · TEAL = the learned kernel]],
)

#pause
#result[The signed kernel compares the row above with the row below.]

== Library “convolution” is cross-correlation #D

For one channel, no padding, and stride $1$,

#pause
$ Y[i,j] = sum_(u=0)^(k_h-1) sum_(v=0)^(k_w-1) K[u,v] X[i+u,j+v]. $

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([CROSS-CORRELATION], [uses $K[u,v]$ in the displayed orientation], color: TEAL),
  hairline([MATHEMATICAL CONVOLUTION], [flips the kernel before sliding], color: MUTED),
))

#pause
#note([PyTorch and other DL libraries implement cross-correlation but call the layer convolution. Because $K$ is learned, either orientation can represent the required detector.], color: GREEN)

== Predict the first response before multiplying #Q

#case-strip([first patch], [$X[0:3,0:3]$ meets the same $K$; predict sign, then exact value.])
#v(8pt)
#two(
  [#align(center, tensor-grid(((0,0,0),(0,0,0),(1,1,1)),
      title: [first patch], cell: 13mm, vmin: -1, vmax: 1))],
  [#align(center, tensor-grid(K, title: [kernel $K$], cell: 13mm, vmin: -1, vmax: 1))],
)

#pause
#align(center, text(size: 21pt, fill: BLUE)[Will $Y[0,0]$ be negative, zero, or positive? What is its exact value?])

== Answer: the bottom row contributes $-3$ #A

#case-strip([first response], [Every displayed number comes from the fixed image and kernel.])
#v(7pt)

#uncover("2-")[
$ Y[0,0] =
  (0 dot 1+0 dot 1+0 dot 1)
  +(0 dot 0+0 dot 0+0 dot 0)
  +(1 dot (-1)+1 dot (-1)+1 dot (-1)). $
]

#pause
#uncover("3-")[#align(center, text(size: 25pt, weight: 700, fill: ACC)[$Y[0,0]=-3$])]

#pause
#result[Negative means bright pixels lie below dark pixels for this kernel orientation.]

== Predict the full valid feature map #Q

#case-strip([slide], [$H=W=5$, $k=3$, $p=0$, $s=1$; the kernel is reused at every legal start.])
#v(8pt)

#align(center, tensor-grid(X, title: [$X$], cell: 9.5mm, vmin: -1, vmax: 1))

#pause
#align(center, text(size: 20pt, fill: BLUE)[
  1. What is the output shape? \
  2. Why are values constant across each output row? \
  3. Which row is positive?
])

== Answer: nine local questions form one feature map #A

#case-strip([feature map], [$Y$ stores one signed response for every legal $3 times 3$ patch.])
#v(7pt)

#align(center, grid(columns: (70mm, 18mm, 70mm), align: horizon,
  tensor-grid(X, title: [$X$: $5 times 5$], cell: 8.5mm, vmin: -1, vmax: 1),
  text(size: 27pt, fill: TEAL)[$arrow.r$],
  uncover("2-")[#tensor-grid(Yvalid, title: [$Y$: $3 times 3$], cell: 11mm, vmin: -3, vmax: 3)],
))

#pause
#result[$Y=mat(-3,-3,-3; -3,-3,-3; 3,3,3)$: the same detector, nine locations.]

== Weight sharing is visible in the computation graph #V

#align(center, diagram(spacing: (27mm, 15mm), node-stroke: 1pt + INK, node-fill: white, {
  let ps = ((0,-1),(0,0),(0,1))
  let ys = ((3,-1),(3,0),(3,1))
  for i in range(3) {
    edge(ps.at(i),(1.5,0), text(size: 10.5pt, fill: TEAL)[$K$], "-|>", stroke: 1.1pt + TEAL)
    edge((1.5,0),ys.at(i),"-|>",stroke: 1pt + ACC)
    node(ps.at(i), text(size: 12pt)[$X_#(i+1)$], radius: 8mm, stroke: 1pt + INK)
    node(ys.at(i), text(size: 12pt)[$Y_#(i+1)$], radius: 8mm, stroke: 1pt + ACC)
  }
  node((1.5,0), align(center)[#text(size: 12pt, weight: 700)[local dot product] \
    #text(size: 10pt, fill: TEAL)[one shared $K$]],
    shape: fletcher.shapes.rect, inset: 10pt, corner-radius: 3pt,
    stroke: 1.3pt + TEAL)
}))

#pause
#result[Outputs are numerous; learned questions are few. Backprop must therefore sum evidence from every use of $K$.]

== A multi-channel kernel spans every input channel #D

At one RGB pixel, let $x=(0.8,0.2,0.5)$ and one $1 times 1$ kernel be $k=(0.5,-1,0.4)$.

#pause
$ y = 0.8(0.5)+0.2(-1)+0.5(0.4) = 0.4-0.2+0.2 = 0.4. $

#pause
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([SPATIAL SUPPORT], [$1 times 1$: one pixel location], color: BLUE),
  hairline([CHANNEL SUPPORT], [$C_"in"=3$: all RGB channels], color: INK),
  hairline([ONE RESPONSE], [channel contributions sum to $0.4$], color: ACC),
))

#pause
#note([A $3 times 3$ RGB kernel has $3 dot 3 dot 3=27$ weights for each output channel.], color: TEAL)

== Tensor shapes name every axis before we count #D

Board notation keeps channels last:

$ X in RR^(H times W times C_"in"), quad
   K in RR^(k_h times k_w times C_"in" times C_"out"), quad
   Y in RR^(H' times W' times C_"out"). $

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([ONE OUTPUT CHANNEL], [one kernel spans all $C_"in"$ input channels], color: TEAL),
  hairline([$C_"out"$ CHANNELS], [a bank of $C_"out"$ kernels produces $C_"out"$ feature maps], color: ACC),
))

#pause
#result[$"parameters"=k_h k_w C_"in" C_"out"+C_"out"$ when each output channel has one bias.]

== Checkpoint: count a kernel bank #Q

A $3 times 3$ convolution maps $64$ input channels to $128$ output channels and uses one bias per output channel.

#v(10pt)
#mcq(
  [How many learned parameters does it have?],
  [$3 dot 3 dot 64+128$],
  [$3 dot 3 dot 64 dot 128$],
  [$3 dot 3 dot 64 dot 128+128$],
  [$H'W' dot 3 dot 3 dot 64 dot 128$],
)

#pause
#note([Commit before calculating. Output height and width do not belong in the parameter formula.], color: BLUE)

== Answer: kernels connect channel pairs; biases index outputs #A

#mcq-answer(
  [C],
  [$3 dot 3 dot 64 dot 128+128="73,856"$],
  [Each of 128 output channels owns a $3 times 3 times 64$ kernel and one bias. Spatial reuse adds computations, not parameters.],
)

#pause
#result[Parameters depend on kernel and channel counts—not on image resolution.]

== Companion notebook: predict, calculate, run, inspect, explain #I

#interbox(link-to: NB1)[
  Open the exact Colab notebook. First predict $Y[0,0]$ and the full $3 times 3$ map; then run the NumPy loops and compare every value.
]

#pause
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([PREDICT], [shape, sign, and legal starts], color: BLUE),
  hairline([RUN], [from-scratch cross-correlation], color: TEAL),
  hairline([EXPLAIN], [where sharing, padding, and stride occur], color: GREEN),
))

#pause
#caption[The notebook retains outputs and later verifies the same gradients with PyTorch and finite differences.]

// ═══════════════════════ GEOMETRY · CORE ════════════════════════════
= Padding and stride count legal starts

== Derive output size by counting kernel starts #D

For one spatial dimension with input length $H$, symmetric padding $p$, kernel size $k$, stride $s$, and dilation $1$:

#pause
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([PADDED LENGTH], [$H+2p$], color: INK),
  hairline([LAST LEGAL START], [$H+2p-k$], color: BLUE),
  hairline([STARTS], [$0,s,2s,dots$ up to that value], color: TEAL),
))

#pause
$ H_"out" = floor((H+2p-k)/s)+1. $

#pause
#result[The floor discards an incomplete final jump; it is not a rounding convention.]

== The running image separates padding from stride #D

#case-strip([geometry], [Same $X$ and $K$; only $p$ and $s$ change.])
#v(6pt)
#set text(size: 13.5pt)
#align(center, grid(columns: (1fr, 1.35fr, 1fr), gutter: 8pt, align: horizon,
  [#align(center, tensor-grid(Yvalid, title: [$p=0,s=1$ · $3 times 3$], cell: 8.4mm, vmin: -3, vmax: 3))],
  [#uncover("2-")[#align(center, tensor-grid(Ypadded, title: [$p=1,s=1$ · $5 times 5$], cell: 8.4mm, vmin: -3, vmax: 3))]],
  [#uncover("3-")[#align(center, tensor-grid(Ystride, title: [$p=1,s=2$ · $3 times 3$], cell: 8.4mm, vmin: -3, vmax: 3))]],
))

#pause
#pause
#note([Padding changes which border patches are legal. Stride keeps only every $s$-th legal start.], color: BLUE)

== “Same” padding is a restricted promise #D

For odd $k$ and stride $s=1$, symmetric

#pause
$ p=(k-1)/2 quad arrow.r quad H_"out"=H. $

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([EXACT SIMPLE CASE], [odd kernel, unit stride, equal padding on both sides], color: GREEN),
  hairline([FRAMEWORK GENERALIZATION], [even kernels or $s>1$ may require asymmetric padding and a stated output convention], color: MUTED),
))

#pause
#note([Padding preserves shape; it does not create new scene evidence. Near a border, some receptive-field entries are padding.], color: RED)

== Floor is not optional #Q

Let $H=8$, $k=3$, $p=0$, and $s=2$.

#v(9pt)
#mcq(
  [What are the legal starts and output length?],
  [$0,2,4,6$ and $H_"out"=4$],
  [$0,2,4$ and $H_"out"=3$],
  [$0,1,2,3,4,5$ and $H_"out"=6$],
  [$0,3,6$ and $H_"out"=3$],
)

#pause
#note([Check whether the length-3 kernel fits completely at each proposed start.], color: BLUE)

== Answer: start $6$ would require an index outside the input #A

#mcq-answer(
  [B],
  [legal starts $0,2,4$; $H_"out"=3$],
  [$floor((8-3)/2)+1=floor(2.5)+1=3$. A start at 6 would use indices $6,7,8$, but index 8 is outside a length-8 input.],
)

#pause
#result[List legal starts when the floor formula feels opaque.]

== Shape arithmetic and parameter arithmetic answer different questions #D

#align(center, table(
  columns: (45mm, 72mm, 72mm), stroke: 0.45pt + MUTED,
  inset: (x: 9pt, y: 6pt), align: (left, left, left),
  table.header([question], [depends on], [does not depend on]),
  [output shape], [$H,W,k,p,s,C_"out"$], [number of training examples],
  [parameter count], [$k_h,k_w,C_"in",C_"out"$ and bias choice], [$H,W,p,s$],
  [convolution MACs], [output positions and kernel volume], [bias count in our convention],
))

#pause
#note([Our MAC ledger counts one multiply-accumulate per kernel weight per output position and excludes bias adds, pooling, and activation costs.], color: MUTED)

// ═════════════════ EQUIVARIANCE + POOLING · CORE ════════════════════
= Sharing moves responses; pooling discards location

== Predict what a one-step shift does #Q

Use $x=(0,1,1,0,0,0)$ and $k=(-1,0,1)$ with valid cross-correlation.

#v(8pt)
#two(
  [#align(center, tensor-grid(((0,1,1,0,0,0),), title: [$x$], cell: 12mm, vmin: -1, vmax: 1))],
  [#align(center, tensor-grid(((-1,0,1),), title: [$k$], cell: 12mm, vmin: -1, vmax: 1))],
)

#pause
#align(center, text(size: 19pt, fill: BLUE)[
  First compute $f(x)$. Then prepend zero and drop the last input value to obtain $T_1x$.
])

#pause
#note([Will $f(T_1x)$ equal the zero-filled shift $T_1f(x)$ at all four displayed output positions?], color: BLUE)

== Answer: the common interior shifts; the cropped boundary differs #A

#set text(size: 14pt)
#align(center, table(
  columns: (42mm, 130mm), stroke: 0.45pt + MUTED,
  inset: (x: 8pt, y: 5pt), align: (left, center),
  table.header([quantity], [exact values]),
  [$x$], [$0,1,1,0,0,0$],
  [$f(x)$], [$1,-1,-1,0$],
  [$T_1x$], [$0,0,1,1,0,0$],
  [$f(T_1x)$], [$1,1,-1,-1$],
  [$T_1f(x)$], [$0,1,-1,-1$],
))

#pause
#note([
  Positions 1–3 agree: $(1,-1,-1)$. Position 0 differs because valid cropping removed the output that would have shifted in from the left.
], color: GREEN)

#pause
#result[The numerical evidence demonstrates interior equivariance and the finite-boundary caveat—both at once.]

== State the equivariance claim with its conditions #D

On an infinite grid, shared stride-1 cross-correlation commutes with integer translation:

#pause
$ f(T_Delta X)=T_Delta f(X). $

#pause
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([BOUNDARY], [crop and padding change what enters from outside], color: RED),
  hairline([STRIDE], [$s>1$ samples one phase; a one-pixel shift changes phase], color: BLUE),
  hairline([POINTWISE RELU], [preserves equivariance away from the same boundaries], color: GREEN),
))

#pause
#note([A stack with total stride $j$ is naturally aligned to translations by multiples of $j$; arbitrary one-pixel shifts need not commute exactly.], color: MUTED)

== Pool the same response map #D

#case-strip([pooling], [Apply $2 times 2$ max pooling with stride $1$ to the exact $Y$ already computed.])
#v(7pt)
#align(center, grid(columns: (72mm, 20mm, 72mm), align: horizon,
  tensor-grid(Yvalid, title: [$Y$: $3 times 3$], cell: 11mm, vmin: -3, vmax: 3),
  text(size: 27pt, fill: BLUE)[$arrow.r$],
  uncover("2-")[#tensor-grid(Panchor, title: [max pool: $2 times 2$], cell: 15mm, vmin: -3, vmax: 3)],
))

#pause
#align(center, text(size: 20pt, fill: ACC)[$P=mat(-3,-3;3,3)$])

#pause
#result[Pooling learns no weights: it summarizes a window and deliberately loses some spatial detail.]

== Equivariance and invariance are different promises #V

#align(center, grid(
  columns: (53mm, 16mm, 53mm, 16mm, 53mm), gutter: 3pt, align: horizon,
  card([INPUT SHIFTS], [Coordinates of the pattern change.], color: INK),
  flow-arrow(label: [shared conv], color: TEAL),
  card([FEATURE SHIFTS], [Response location changes: equivariance.], color: ACC),
  flow-arrow(color: BLUE),
  card([SUMMARY CHANGES LESS], [Some location detail is discarded: partial invariance.], color: GREEN),
))

#pause
#v(8pt)
#note([
  Global average pooling maps $H times W times C arrow.r C$. For our pooled anchor,
  $( -3-3+3+3)/4=0$.
], color: BLUE)

#pause
#result[Pooling and augmentation can promote robustness; neither guarantees invariance to every translation.]

== Checkpoint: what does max pooling guarantee? #Q

#mcq(
  [Choose the strongest defensible statement.],
  [Max pooling makes every CNN exactly translation invariant],
  [Max pooling preserves the precise position of every activation],
  [A max is unchanged by rearrangements within one fixed window, but window boundaries and stride still matter],
  [Pooling learns one detector per window],
)

#pause
#note([Commit before the answer. Separate a local window fact from a whole-network guarantee.], color: BLUE)

== Answer: local robustness is conditional #A

#mcq-answer(
  [C],
  [the max is unchanged by rearrangements within one fixed window],
  [Crossing a window boundary can change two pooled outputs, and strided pooling changes the sampling phase. The layer also discards the winning location.],
)

#pause
#result[Use “partial invariance” or “local robustness,” not “pooling guarantees invariance.”]

// ═════════════════ RF + CLASSIFIER · CORE ═══════════════════════════
= Compose local questions into a classifier

== Receptive field needs both width and jump #D

For layer $ell$, track:

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  card([RECEPTIVE FIELD $r_ell$], [width of original-input region that can influence one unit], color: ACC),
  card([JUMP $j_ell$], [spacing, in input pixels, between adjacent units at this layer], color: BLUE),
))

#pause
#align(center, text(size: 22pt)[$r_0=1, quad j_0=1$])

#pause
#note([Kernel size describes one layer. Receptive field traces influence through every preceding layer.], color: GREEN)

== Derive the receptive-field recursion #D

Assume dilation $1$. A size-$k_ell$ kernel spans $k_ell-1$ gaps from the previous layer, and each gap is $j_(ell-1)$ input pixels.

#pause
$ r_ell = r_(ell-1)+(k_ell-1)j_(ell-1). $

#pause
Stride changes the spacing between adjacent new units:

$ j_ell=j_(ell-1)s_ell. $

#pause
#note([With dilation $d_ell$, replace $(k_ell-1)$ by $(k_ell-1)d_ell$. Padding changes boundary alignment, not this theoretical size.], color: MUTED)

== The running case reaches $r=4$ after convolution and pooling #D

#case-strip([receptive field], [The first $3 times 3$ response uses the anchor kernel; a following $2 times 2$, stride-2 pool combines adjacent responses.])
#v(7pt)
#align(center, table(
  columns: (43mm, 25mm, 25mm, 38mm, 38mm), stroke: 0.45pt + MUTED,
  inset: (x: 8pt, y: 5pt), align: center,
  table.header([layer], [$k$], [$s$], [$r$], [$j$]),
  [input], [$1$], [$1$], [$1$], [$1$],
  [shared conv], [$3$], [$1$], [#uncover("2-")[$3$]], [#uncover("2-")[$1$]],
  [max pool], [$2$], [$2$], [#uncover("3-")[$4$]], [#uncover("3-")[$2$]],
  [next conv], [$3$], [$1$], [#uncover("4-")[$8$]], [#uncover("4-")[$2$]],
))

#pause
#pause
#pause
#result[The first pooled unit can depend on a $4 times 4$ anchor region; the next conv expands context to $8 times 8$.]

== Scale the rule into one $32 times 32$ classifier #V

#align(center, grid(
  columns: (26mm, 8mm, 30mm, 8mm, 30mm, 8mm, 30mm, 8mm, 26mm, 8mm, 25mm),
  gutter: 2pt, align: horizon,
  card([INPUT], [$3 times 32 times 32$], color: INK),
  flow-arrow(color: MUTED),
  card([CONV 1], [$16 times 32 times 32$], color: TEAL),
  flow-arrow(color: MUTED),
  card([POOL], [$16 times 16 times 16$], color: BLUE),
  flow-arrow(color: MUTED),
  card([CONV 2], [$32 times 16 times 16$], color: TEAL),
  flow-arrow(color: MUTED),
  card([GAP], [$32$], color: BLUE),
  flow-arrow(color: MUTED),
  card([LINEAR], [$10$], color: ACC),
))

#pause
#v(8pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([BOARD], [$H times W times C$; kernel $k_h times k_w times C_"in" times C_"out"$], color: BLUE),
  hairline([PYTORCH], [batch $N times C times H times W$; weights $C_"out" times C_"in" times k_h times k_w$], color: TEAL),
))

#pause
#result[Axis order changes between notation and code; the represented operation does not.]

== Ledger row 1: same-padded convolution #D

Input: $N times 3 times 32 times 32$. Layer: $3 times 3$, $p=1$, $s=1$, $C_"out"=16$.

#pause
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([OUTPUT], [$N times 16 times 32 times 32$], color: ACC),
  hairline([PARAMETERS], [$3 dot 3 dot 3 dot 16+16=448$], color: TEAL),
  hairline([RF / JUMP], [$r=3, j=1$], color: BLUE),
))

#pause
$ "MACs" = 32 dot 32 dot 16 dot (3 dot 3 dot 3) = "442,368". $

#pause
#note([Each of 16 kernels is reused at all $32 dot 32$ output positions.], color: GREEN)

== Ledger rows 2–3: pool, then learn 32 feature types #D

#align(center, table(
  columns: (37mm, 58mm, 42mm, 51mm), stroke: 0.45pt + MUTED,
  inset: (x: 8pt, y: 5pt), align: (left, center, center, center),
  table.header([layer], [output], [parameters], [$r,j$]),
  [$2 times 2$ max pool, $s=2$], [$N times 16 times 16 times 16$], [$0$], [#uncover("2-")[$4,2$]],
  [$3 times 3$ conv, $p=1$], [$N times 32 times 16 times 16$], [#uncover("3-")[$3 dot 3 dot 16 dot 32+32="4,640"$]], [#uncover("3-")[$8,2$]],
))

#pause
#pause
#v(8pt)
#uncover("3-")[
$ "conv 2 MACs"=16 dot 16 dot 32 dot (3 dot 3 dot 16)="1,179,648". $
]

#pause
#result[Stride does not add learned parameters; it changes spatial size, compute, jump, and downstream receptive field.]

== Ledger rows 4–5: global average, then ten logits #D

#align(center, grid(
  columns: (55mm, 14mm, 55mm, 14mm, 55mm), gutter: 4pt, align: horizon,
  card([FEATURE MAPS], [$N times 32 times 16 times 16$], color: ACC),
  flow-arrow(label: [GAP], color: BLUE),
  card([SUMMARY], [$N times 32$; zero learned parameters], color: BLUE),
  flow-arrow(label: [linear], color: TEAL),
  card([LOGITS], [$N times 10$; $32 dot 10+10=330$ parameters], color: ACC),
))

#pause
#v(10pt)
$ "linear MACs"=32 dot 10=320. $

#pause
#note([Global average pooling removes the fixed-resolution requirement that a flattened dense head would introduce.], color: GREEN)

== The complete ledger separates capacity, compute, and context #D

#set text(size: 13pt)
#align(center, table(
  columns: (29mm, 50mm, 31mm, 38mm, 25mm), stroke: 0.4pt + MUTED,
  inset: (x: 6pt, y: 4pt), align: (left, center, center, center, center),
  table.header([layer], [output], [parameters], [MACs], [$r,j$]),
  [input], [$N times 3 times 32 times 32$], [$0$], [$0$], [$1,1$],
  [conv 1], [$N times 16 times 32 times 32$], [$448$], [$"442,368"$], [$3,1$],
  [pool], [$N times 16 times 16 times 16$], [$0$], [not counted], [$4,2$],
  [conv 2], [$N times 32 times 16 times 16$], [$"4,640"$], [$"1,179,648"$], [$8,2$],
  [GAP], [$N times 32$], [$0$], [not counted], [global],
  [linear], [$N times 10$], [$330$], [$320$], [global],
  table.hline(),
  [total], [], [$"5,418"$], [$"1,622,336"$], [],
))

#pause
#caption[MAC convention: kernel multiplies only. Bias adds, activations, pooling, and GAP are excluded and stated explicitly.]

== Checkpoint: double height and width #Q

Run the same fully convolutional classifier on $64 times 64$ RGB input.

#v(8pt)
#mcq(
  [What necessarily changes?],
  [Convolution parameters quadruple; MACs stay fixed],
  [Parameters stay fixed; convolution activations and MACs grow by about $4 times$],
  [The number of logits becomes 40],
  [The pre-GAP receptive-field width doubles automatically],
)

#pause
#note([Hold kernel sizes, channels, padding, and stride fixed.], color: BLUE)

== Answer: reuse keeps parameters fixed, not compute #A

#mcq-answer(
  [B],
  [parameters stay fixed; convolution activations and MACs grow by about $4 times$],
  [Doubling both spatial axes creates four times as many output positions. Kernel banks and the GAP-to-linear head are unchanged.],
)

#pause
#align(center, text(size: 19pt)[
  exact MACs: $4("442,368"+"1,179,648")+320="6,488,384"$.
])

#pause
#result[Parameter count measures stored capacity; MAC count measures work at a chosen resolution.]

// ═══════════════════════ SHOULD COVER · 15 MIN ══════════════════════
= Shared gradients and implementation checks

== Forward example: one kernel is reused twice #D

Let $x=(2,1,3)$, $k=(0.5,-1)$, and $b=0$.

#pause
$ y_0=0.5(2)-1(1)=0, quad
   y_1=0.5(1)-1(3)=-2.5. $

#pause
#align(center, diagram(spacing: (27mm, 15mm), node-stroke: 1pt + INK, node-fill: white, {
  node((0,-0.7), [$x_0,x_1$], radius: 11mm, stroke: 1pt + INK)
  node((0,0.7), [$x_1,x_2$], radius: 11mm, stroke: 1pt + INK)
  node((1.6,0), align(center)[$k_0,k_1$ \
    #text(size: 10pt, fill: TEAL)[shared]], radius: 13mm, stroke: 1.3pt + TEAL)
  node((3.2,-0.7), [$y_0=0$], radius: 10mm, stroke: 1pt + ACC)
  node((3.2,0.7), [$y_1=-2.5$], radius: 10mm, stroke: 1pt + ACC)
  edge((0,-0.7),(1.6,0),"-|>",stroke: 1pt + TEAL)
  edge((0,0.7),(1.6,0),"-|>",stroke: 1pt + TEAL)
  edge((1.6,0),(3.2,-0.7),"-|>",stroke: 1pt + ACC)
  edge((1.6,0),(3.2,0.7),"-|>",stroke: 1pt + ACC)
}))

#pause
#result[One parameter value participates in multiple output equations.]

== Predict the shared kernel gradient #Q

An upstream gradient $g=(partial cal(L)/partial y_0, partial cal(L)/partial y_1)=(1,2)$ arrives.

#v(8pt)
#align(center, text(size: 21pt, fill: BLUE)[
  Does $partial cal(L)/partial k_0$ use the first patch, the second patch, or both?
])

#pause
#v(8pt)
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([$k_0$ USES], [$x_0$ in $y_0$ and $x_1$ in $y_1$], color: TEAL),
  hairline([$k_1$ USES], [$x_1$ in $y_0$ and $x_2$ in $y_1$], color: TEAL),
))

#pause
#note([Write both sums before substituting values.], color: BLUE)

== Answer: every spatial use contributes #A

#uncover("2-")[
$ partial cal(L)/partial k_0
   = g_0x_0+g_1x_1
   =1(2)+2(1)=4. $
]

#pause
#uncover("3-")[
$ partial cal(L)/partial k_1
   = g_0x_1+g_1x_2
   =1(1)+2(3)=7. $
]

#pause
#result[$nabla_k cal(L)=(4,7)$: sharing ties the uses together, so their evidence accumulates.]

== Overlapping patches also accumulate input gradients #D

#uncover("2-")[
$ partial cal(L)/partial x_0=g_0k_0=0.5. $
]

#pause
#uncover("3-")[
$ partial cal(L)/partial x_1=g_0k_1+g_1k_0=-1+1=0. $
]

#pause
#uncover("4-")[
$ partial cal(L)/partial x_2=g_1k_1=-2, quad
   partial cal(L)/partial b=g_0+g_1=3. $
]

#pause
#result[$nabla_x cal(L)=(0.5,0,-2)$ and $nabla_b cal(L)=3$. Shared use and patch overlap are different sums.]

== The 2-D kernel gradient is the same sum over locations #D

For a batch, input channel $c$, output channel $o$, and stride $s$:

#pause
$ partial cal(L)/partial K[u,v,c,o]
  = sum_n sum_i sum_j
    G[n,i,j,o]
    X[n,i s+u-p,j s+v-p,c]. $

#pause
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([BATCH], [sum evidence over examples $n$], color: INK),
  hairline([SPACE], [sum every output location $(i,j)$], color: BLUE),
  hairline([SHARING], [all terms update the same $K[u,v,c,o]$], color: TEAL),
))

#pause
#note([Framework autodiff performs these accumulations; the notebook checks them against both PyTorch and finite differences.], color: GREEN)

== Training state has three separate contracts #D

#set text(size: 14.5pt)
#align(center, table(
  columns: (47mm, 68mm, 70mm), stroke: 0.45pt + MUTED,
  inset: (x: 8pt, y: 5pt), align: (left, left, left),
  table.header([mechanism], [training], [validation / inference]),
  [input standardization], [estimate mean/std on training data], [reuse those fixed values unchanged],
  [augmentation], [random, task-valid transforms], [deterministic preprocessing],
  [Dropout], [active], [off],
  [BatchNorm], [batch statistics + running-stat updates], [stored running statistics],
  [autograd], [record graph for updates], [usually disable graph separately],
))

#pause
#note([model.eval() changes Dropout and BatchNorm behavior; it does not itself disable gradient recording.], color: RED)

== Diagnose CNN failures with one targeted test #V

#set text(size: 14pt)
#align(center, table(
  columns: (45mm, 69mm, 72mm), stroke: 0.45pt + MUTED,
  inset: (x: 8pt, y: 5pt), align: (left, left, left),
  table.header([symptom], [suspect], [smallest decisive test]),
  [shape error], [HWC/NCHW order or wrong $C_"in"$], [print/assert $N,C,H,W$ after every layer],
  [map collapses too early], [accumulated stride/pooling], [extend the shape + $r,j$ ledger],
  [loss does not fall], [data, loss, gradient, or update bug], [overfit one tiny batch],
  [same eval input changes], [Dropout active or BatchNorm state moving], [call eval; run two no-grad passes],
  [train good, validation poor], [generalization, split, or mode issue], [return to L7: verify split, then change one lever],
))

#pause
#result[Do not tune the optimizer until the tensor, state, and one-batch contracts pass.]

== Notebook choreography: one trace, six predictions #I

#interbox(link-to: NB1)[
  Use the single L08 notebook from top to bottom; do not skip directly to the PyTorch model.
]

#pause
#align(center, table(
  columns: (17mm, 76mm, 88mm), stroke: 0.4pt + MUTED,
  inset: (x: 7pt, y: 4pt), align: (center, left, left),
  table.header([step], [predict before run], [evidence after run]),
  [1], [$Y[0,0]$ and full $Y$], [$-3$ and the exact $3 times 3$ feature map],
  [2], [legal starts for each $(p,s)$], [$3,5,4,3$ output lengths],
  [3], [parameters, MACs, $r,j$], [$"5,418"$; $"1,622,336"$; final $(8,2)$],
  [4], [which shifted positions agree], [interior equality; boundary mismatch],
  [5], [$nabla_k,nabla_x,nabla_b$], [PyTorch and finite difference agree],
  [6], [every NCHW activation shape], [final $(1,10)$ logits],
))

#pause
#caption[The notebook uses the exact slide numerics and retains deterministic outputs; no dataset download or training is required.]

// ═══════════════════════════ CLOSE ══════════════════════════════════
= Close the loop

== Revisit the opening commitment #A

The position-specific dense map used $"1,049,600"$ parameters; the shared $3 times 3$ map used $10$.

#pause
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([HOW], [reuse the same nine weights and one bias at all $1024$ output locations], color: TEAL),
  hairline([GAIN], [locality, translation structure, and $104,960 times$ fewer parameters], color: GREEN),
  hairline([GIVE UP], [arbitrary position-specific connectivity in this layer], color: RED),
))

#pause
#result[Opening answer: B. With $p=1,s=1$, ten shared parameters can produce 1,024 responses because output count and parameter count are different quantities.]

== Retrieval: which audit is internally consistent? #Q

#mcq(
  [Choose the only fully consistent statement.],
  [Stride changes parameter count because it changes output size],
  [A shared convolution is exactly translation invariant on every finite image],
  [A $3 times 3$ conv can keep shape with $p=1,s=1$; its parameters ignore $H,W$; its MACs do not],
  [Global average pooling learns one weight per pixel],
)

#pause
#note([Commit, then justify your choice with one formula and one caveat.], color: BLUE)

== Answer: geometry, capacity, and compute stay separate #A

#mcq-answer(
  [C],
  [shape can stay fixed while parameters ignore resolution and MACs scale with it],
  [$H_"out"=floor((H+2p-k)/s)+1$ controls geometry; $k_h k_w C_"in" C_"out"+C_"out"$ controls parameters; output positions multiply MACs.],
)

#pause
#result[Translation equivariance is conditional; pooling is fixed; stride changes sampling, not learned parameter count.]

== L9 handoff: make the local pipeline deep, efficient, and reusable #V

#align(center, grid(
  columns: (54mm, 12mm, 54mm, 12mm, 54mm), gutter: 5pt, align: horizon,
  card([L8 · TODAY], [Build and audit one small CNN from exact arithmetic.], color: TEAL),
  flow-arrow(label: [scale], color: BLUE),
  card([NEW PROBLEM], [Depth, compute, and data make design and optimization harder.], color: RED),
  flow-arrow(label: [next], color: ACC),
  card([L9 · MODERN CNNs], [VGG, ResNet, efficient blocks, pretrained backbones, transfer.], color: ACC),
))

#pause
#result[Today explained the local shared operator; next lecture asks how modern pipelines compose and reuse it.]

#focus-slide[
  A convolution is a linear layer that confesses its assumptions.
  #v(12pt)
  #set text(size: 22pt)
  Local connections decide what can interact; shared weights ask the same learned question everywhere.
]

// ═══════════════════════ OPTIONAL APPENDIX · 10 MIN ═════════════════
= Optional appendix · efficiency and provenance

== Two $3 times 3$ layers versus one $5 times 5$ layer #OPT

At constant width $C$ and ignoring biases:

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  card([TWO $3 times 3$], [
    receptive field: $5 times 5$ \
    parameters: $2(3 dot 3 C^2)=18C^2$ \
    two nonlinearities
  ], color: TEAL),
  card([ONE $5 times 5$], [
    receptive field: $5 times 5$ \
    parameters: $25C^2$ \
    one nonlinearity
  ], color: ACC),
))

#pause
#note([The receptive-field sizes match, but the function classes are not identical: the two-layer path factorizes computation and inserts a nonlinearity.], color: MUTED)

== im2col exposes matrix multiplication; sources fix conventions #OPT

For $x=(2,1,3)$ and kernel $k=(0.5,-1)$,

#pause
$ underbrace(mat(2,1;1,3), "patch matrix") quad
  underbrace(vec(0.5,-1), "shared kernel")
  = underbrace(vec(0,-2.5), "responses"). $

#pause
#note([im2col duplicates overlapping input values in a temporary view; it does not create distinct learned kernel copies.], color: TEAL)

#pause
#set text(size: 14pt)
- #link(PTCONV)[PyTorch Conv2d documentation] — operator, shapes, groups, dilation, and framework convention.
- #link(CONVARITH)[Dumoulin & Visin (2016), A guide to convolution arithmetic] — output geometry.
- #link(LENET)[LeCun et al. (1998), Gradient-based learning applied to document recognition] — classic shared-weight CNN.
- #link(NB1)[Exact L08 Colab notebook] — deterministic slide numerics, gradients, and NCHW model.

#v(5pt)
#caption[All lecture diagrams and numeric traces are native Typst vectors; the deck embeds no raster figures.]
