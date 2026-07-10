// Convolutional Neural Networks — Lecture 8 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture8/L8-cnns.typ /tmp/L8.pdf
//   typst compile --root . --input handout=true lecture8/L8-cnns.typ /tmp/L8h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [Convolutional Neural Networks],
  subtitle: [Locality, Weight Sharing, and Hierarchy in Images],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"

// ── the CNN pipeline: image -> (conv act pool) x N -> GAP -> FC -> softmax ──
#let cnnpipe = align(center, diagram(spacing: (9mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let blk(x, lbl, col) = node((x, 0), text(size: 13pt, lbl), fill: col.lighten(82%), stroke: 0.9pt + col, corner-radius: 3pt, inset: 6pt)
  node((0, 0), text(size: 13pt)[image $X$], radius: 8mm, fill: rgb("#EFEEEB"))
  blk(1, "Conv", TEAL); blk(2, "ReLU", ACC); blk(3, "Pool", BLUE)
  blk(4, "Conv", TEAL); blk(5, "ReLU", ACC); blk(6, "Pool", BLUE)
  blk(7, "GAP", GREEN); blk(8, "FC", ACC)
  node((9, 0), text(size: 13pt)[softmax], fill: ACC.lighten(82%), stroke: 0.9pt + ACC, corner-radius: 3pt, inset: 6pt)
  for i in range(9) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.7pt + MUTED) }
}))

// ── the classifier head: feature map -> GAP -> linear -> softmax -> p ──
#let headpipe = align(center, diagram(spacing: 13mm, node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), align(center, [$X$ \ #text(size: 11pt, fill: MUTED)[$H times W times C$]]), fill: rgb("#EFEEEB"), stroke: 0.9pt + INK, corner-radius: 3pt, inset: 7pt)
  node((1,0), text(size: 14pt)[GAP], fill: GREEN.lighten(82%), stroke: 0.9pt + GREEN, corner-radius: 3pt, inset: 7pt)
  node((2,0), text(size: 14pt)[linear], fill: ACC.lighten(82%), stroke: 0.9pt + ACC, corner-radius: 3pt, inset: 7pt)
  node((3,0), text(size: 14pt)[softmax], fill: ACC.lighten(82%), stroke: 0.9pt + ACC, corner-radius: 3pt, inset: 7pt)
  node((4,0), [$p_y$], radius: 7mm, stroke: 0.9pt + INK)
  for i in range(4) { edge((i,0),(i+1,0), "-|>", stroke: 0.8pt + MUTED) }
}))

// ── weight sharing: three input patches all pass through ONE shared kernel ──
#let weightshare = align(center, diagram(spacing: (17mm, 6mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let patch(y, n) = node((0, y), text(size: 12pt)[patch #n], fill: rgb("#EFEEEB"), stroke: 0.9pt + INK, corner-radius: 3pt, inset: 6pt)
  patch(2.2, 1); patch(0, 2); patch(-2.2, 3)
  node((1.6, 0), align(center, [same kernel \ #text(size: 11pt, fill: MUTED)[$K$ — shared]]), fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL, corner-radius: 3pt, inset: 8pt)
  node((3.2, 2.2),  [$y_1$], radius: 6mm, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  node((3.2, 0),    [$y_2$], radius: 6mm, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  node((3.2, -2.2), [$y_3$], radius: 6mm, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  edge((0,2.2),  (1.6,0), "-|>", stroke: 0.7pt + MUTED)
  edge((0,0),    (1.6,0), "-|>", stroke: 0.7pt + MUTED)
  edge((0,-2.2), (1.6,0), "-|>", stroke: 0.7pt + MUTED)
  edge((1.6,0), (3.2,2.2),  "-|>", stroke: 0.7pt + MUTED)
  edge((1.6,0), (3.2,0),    "-|>", stroke: 0.7pt + MUTED)
  edge((1.6,0), (3.2,-2.2), "-|>", stroke: 0.7pt + MUTED)
}))

#title-slide()

// ═══════════════════════════ PART I — Why MLPs fail on images ═══════════════════════════
= Why MLPs struggle with images

== Where we are

So far every network was a *fully-connected* MLP on a flat vector $x in RR^d$.
#pause
$ h = phi(W x + b), quad quad cal(L)(theta) = 1/n sum_(i=1)^n ell(f_theta (x_i), y_i) $
#pause
#notebox[That worked for tabular data. But an *image* is not an unstructured vector — it has *spatial structure* the MLP throws away.]

== An image is a tensor

A colour image is a 3-D array — height, width, and colour channels:
#pause
$ X in RR^(H times W times C) $
#pause
- $H times W$ — a *grid* of pixels with a real *geometry*;
#pause
- $C$ — channels (e.g. $C = 3$ for RGB);
#pause
- neighbouring pixels are *strongly correlated*; far-apart ones usually are not.
#pause
#align(center, text(size: 16pt, fill: MUTED)[a $224 times 224$ RGB image is $224 times 224 times 3 = "150,528"$ numbers])

== Flattening destroys structure #V

To feed an MLP we must *flatten* $X$ into a vector:
$ X in RR^(H times W times C) quad -->^"flatten" quad x in RR^(H W C) $
#pause
#alertbox[Flattening discards *which pixels are neighbours*. Pixel $(i,j)$ and $(i, j+1)$ — adjacent in the image — become two arbitrary, unrelated entries of $x$.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[the MLP would have to *re-learn* the grid geometry from scratch, from data])

== Parameter explosion #D

Connect a flattened $224 times 224 times 3$ image to a modest hidden layer of $1000$ units:
#pause
$ underbrace("150,528", "inputs") times underbrace(1000, "hidden units") = 1.5 times 10^8 quad "weights" $
#pause
#alertbox[*One* layer — 150 million parameters. Deeper nets become impossible: too many weights to store, and far too many to fit without massive overfitting.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[and this is before we add a single second layer])

== Three priors images obey

Images are *not* arbitrary vectors. Three structural facts we should build in:
#pause
+ *Locality* — meaningful patterns (edges, corners) are *local*: a pixel is explained by its *neighbours*, not by pixels across the image.
#pause
+ *Translation equivariance* — a cat is a cat in the *top-left* or the *bottom-right*; the *same* detector should apply everywhere.
#pause
+ *Hierarchy* — edges compose into textures, textures into parts, parts into objects.

== The CNN answer

#align(center, text(size: 22pt, fill: INK)[
  Images have *spatial structure*. \
  CNNs exploit *locality*, *weight sharing*, and *hierarchy*.
])
#pause
#v(6pt)
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left),
  table.header([*Image prior*], [*CNN mechanism*]),
  [locality], [small *local* kernels, not full connections],
  [translation equivariance], [*share* one kernel across all positions],
  [hierarchy], [*stack* conv layers → growing features],
))

// ═══════════════════════════ PART II — Convolution ═══════════════════════════
= Convolution

== 1-D convolution: a sliding weighted sum

Slide a small *kernel* $K$ of weights over a signal $x$, taking a weighted sum at each spot:
#pause
$ y[i] = sum_(u) K[u] thin x[i + u] $
#pause
- the *same* weights $K$ are reused at *every* position $i$ — that is *weight sharing*;
#pause
- each output looks at only a *few* neighbouring inputs — that is *locality*.
#pause
#align(center, text(size: 16pt, fill: MUTED)[a 3-tap kernel $[-1, 0, 1]$ computes a local *difference* — a 1-D edge detector])

== 2-D convolution

For an image, slide a 2-D kernel over both axes:
#pause
$ Y[i, j] = sum_(u, v) K[u, v] thin X[i + u, j + v] $
#pause
- $K$ is a small window (e.g. $3 times 3$) of *learned* weights;
#pause
- one output pixel is a weighted sum of a small *patch* of the input;
#pause
- the *same* $K$ produces *every* output pixel.
#pause
#align(center, text(size: 16pt, fill: MUTED)[few weights, reused everywhere — the opposite of a fully-connected layer])

== Weight sharing, visually #V

The *same* kernel $K$ is applied at *every* spatial location — not a fresh weight set per position:
#pause
#weightshare
#pause
#align(center, text(size: 16pt, fill: MUTED)[one small set of weights serves the whole image — this is what makes convolution cheap and translation-aware])

== Worked example: one conv step #D

Kernel = horizontal-edge detector; patch = top-left $3 times 3$ of the input:
#pause
$ K = mat(1, 1, 1; 0, 0, 0; -1, -1, -1), quad X_"patch" = mat(1, 2, 0; 0, 1, 3; 2, 1, 0) $
#pause
Elementwise-multiply and sum — take it one row at a time:
#pause
- top row: $1 dot 1 + 1 dot 2 + 1 dot 0 = 3$;
#pause
- middle row: $0 dot 0 + 0 dot 1 + 0 dot 3 = 0$ — the zeros erase it;
#pause
- bottom row: $(-1) dot 2 + (-1) dot 1 + (-1) dot 0 = -3$.
#pause
$ Y = 3 + 0 + (-3) = 0 $
#pause
#result[$Y = 0$ — top brightness $=$ bottom brightness, so *no* horizontal edge here]

== Convolution, visually #V

#fig("/lecture8/figures/conv_op.svg", w: 74%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[top row minus bottom row: this kernel *fires* only where brightness changes vertically])

== A kernel is a feature detector

Each kernel *responds* to one local pattern — it lights up where that pattern appears:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*Kernel*], [*Detects*]),
  [$[-1, 0, 1]$ (horizontal)], [vertical edges],
  [$[1,1,1;0,0,0;-1,-1,-1]$], [horizontal edges],
  [centre-surround], [blobs / corners],
  [oriented bars], [textures at an angle],
))
#pause
#notebox[In a CNN these kernels are *not* hand-designed — they are *learned* by gradient descent.]

== Cross-correlation vs convolution #OPT

True mathematical convolution *flips* the kernel before sliding:
$ (K * X)[i,j] = sum_(u,v) K[u,v] thin X[i - u, j - v] $
#pause
Deep-learning libraries skip the flip — they compute *cross-correlation*:
$ Y[i,j] = sum_(u,v) K[u,v] thin X[i + u, j + v] $
#pause
#notebox[Since $K$ is *learned*, the flip is irrelevant — the network just learns the flipped kernel. Everyone calls it "convolution" anyway. We will too.]

== Multiple kernels → channels

One kernel gives one *feature map*. Use $C_"out"$ kernels to detect $C_"out"$ patterns:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, center, left),
  table.header([*Tensor*], [*Shape*], [*Meaning*]),
  [input $X$], [$H times W times C_"in"$], [$C_"in"$ input channels],
  [weights $W$], [$k_h times k_w times C_"in" times C_"out"$], [$C_"out"$ kernels],
  [output $Y$], [$H' times W' times C_"out"$], [$C_"out"$ feature maps],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[each output channel sums over *all* input channels, then across the $k_h times k_w$ window])

== Parameter count of a conv layer #D

Each of the $C_"out"$ kernels has $k_h dot k_w dot C_"in"$ weights, plus one bias:
#pause
$ #text(fill: INK)[params] = k_h dot k_w dot C_"in" dot C_"out" + C_"out" $
#pause
Worked example — a $3 times 3$ conv, $C_"in" = 64$, $C_"out" = 128$:
#pause
$ 3 dot 3 dot 64 = 576 quad "weights in one kernel" $
#pause
$ 576 dot 128 + 128 = "73,728" + 128 = "73,856" $
#pause
#notebox[Independent of image size $H times W$ — a $3 times 3$ kernel costs the same on a $32 times 32$ or a $2000 times 2000$ image. Contrast the MLP's $1.5 times 10^8$.]

== Interactive: convolution playground #I

#interbox(link-to: IA + "convolution-visualizer")[
  Paint an input grid, pick a kernel, and watch the window slide — see each multiply-add produce one output pixel, and how the kernel choice changes the feature map.
]
#pause
*Q.* What kernel would detect a $45degree$ diagonal edge?

// ═══════════════════════════ PART III — Padding, stride, output size ═══════════════════════════
= Padding, stride, and output size

== Convolution shrinks the image

A $k times k$ kernel cannot centre on the border pixels, so the output is *smaller*:
$ H' = H - k + 1 quad (#text[valid convolution, no padding]) $
#pause
- a $3 times 3$ kernel on $32 times 32$ → $30 times 30$;
#pause
- stack many layers and the map *shrinks away*.
#pause
#result[fix: *pad* the border with zeros]

== Padding

Add a ring of $p$ zeros around the input before convolving:
#pause
- $p = 0$ — *valid*: no padding, output shrinks;
#pause
- $p = floor(k slash 2)$ — *same*-ish: output keeps the input size (for stride 1);
#pause
- e.g. $k = 3, p = 1$ or $k = 5, p = 2$ preserve $H times W$.
#pause
#fig("/lecture8/figures/padding_stride.svg", w: 44%)

== Stride

*Stride* $s$ = how far the kernel jumps between outputs. $s > 1$ *downsamples*:
#pause
- $s = 1$ — evaluate at every position (dense);
#pause
- $s = 2$ — skip every other position → output roughly *halved* in each dimension.
#pause
#align(center, text(size: 16pt, fill: MUTED)[stride is a cheap way to shrink the spatial size *inside* the convolution])

== The output-size formula #D

Padding $p$, kernel $k$, and stride $s$ combine into one count:
#pause
$ H_"out" = floor((H + 2 p - k) / s) + 1 $
#pause
#two(
  notebox[*Same* — $H = 32, k = 5, p = 2, s = 1$:
  $ floor((32 + 4 - 5) / 1) + 1 = 31 + 1 = 32 $],
  notebox[*Halved* — $H = 32, k = 3, p = 1, s = 2$:
  $ floor((32 + 2 - 3) / 2) + 1 = 15 + 1 = 16 $],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[*check:* a $7 times 7$ kernel, $s = 1$ — what padding keeps $H times W$? #h(0.4em) ($p = 3$, since $floor(k slash 2) = 3$)])

== Output size at a glance #V

#fig("/lecture8/figures/output_size.svg", w: 66%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[$p = floor(k slash 2), s = 1$ keeps size (same) · $s = 2$ halves it · no padding slowly shrinks])

== Interactive: padding & stride #I

#interbox(link-to: IA + "convolution-visualizer")[
  Sweep padding and stride and watch the output grid grow, hold, or shrink — and confirm the $floor((H + 2p - k) slash s) + 1$ count.
]
#pause
#interbox(link-to: IA + "padding-stride")[a dedicated dial for $p$ and $s$ with the live output-size readout.]

// ═══════════════════════════ PART IV — Equivariance & pooling ═══════════════════════════
= Equivariance and pooling

== Translation equivariance #D

Because one kernel is *shared* across all positions, shifting the input *shifts the output the same way*:
#pause
$ f(T_Delta thin x) = T_Delta thin f(x) $
#pause
where $T_Delta$ is a shift by $Delta$. Move the cat right by 10 pixels → its feature map moves right by 10 pixels.
#pause
#alertbox[This is *equivariance*, not *invariance*. A conv layer does not *ignore* position — it *tracks* it. Invariance ("cat anywhere → same label") is built *later*, by pooling and the final head.]

== Pooling: summarise a neighbourhood

Slide a window over each feature map and replace it by a *single summary* — no learned weights:
#pause
$ y = max_((i,j) in R) x_(i j) quad quad (#text[max pooling]) $
#pause
$ y = 1/abs(R) sum_((i,j) in R) x_(i j) quad quad (#text[average pooling]) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[typically a $2 times 2$ window with stride 2 → halves $H$ and $W$])

== Pooling, worked #V

#fig("/lecture8/figures/pooling.svg", w: 72%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[max keeps the *strongest* response in each region; average keeps the *mean*])

== What pooling buys us

#pause
- *Downsampling* — fewer spatial positions → less compute and memory downstream;
#pause
- *Larger receptive field* — each later neuron now sees more of the input;
#pause
- *Local robustness* — a small shift within a window leaves $max$ unchanged → a step toward *invariance*;
#pause
- but *information loss* — pooling is not invertible; you discard *where* inside the window.

== Global average pooling

At the very end, collapse each feature map to a *single number* — its channel-wise mean:
#pause
$ h_c = 1/(H W) sum_(i, j) X_(i j c) $
#pause
- turns an $H times W times C$ map into a length-$C$ vector;
#pause
- no parameters, and works at *any* input resolution;
#pause
- the modern replacement for a giant flatten-then-dense head.
#pause
#align(center, text(size: 16pt, fill: MUTED)[GAP is where a stack of *equivariant* conv layers finally becomes *invariant* to position])

== Interactive: pooling #I

#interbox(link-to: IA + "pooling-visualizer")[drag a window over a feature map and compare max vs average pooling, and watch a small shift of the input leave the max output unchanged.]

// ═══════════════════════════ PART V — Receptive field ═══════════════════════════
= Receptive field

== What one neuron sees

The *receptive field* of an output unit = the region of the *input* that can affect it.
#pause
- a single $3 times 3$ conv → receptive field $3 times 3$;
#pause
- stack a *second* $3 times 3$ conv → each new unit sees a $3 times 3$ patch of units that *each* saw $3 times 3$ → an effective $5 times 5$ input region.
#pause
#result[depth grows the receptive field — deeper neurons see *more context*]

== Receptive field grows with depth #V

#fig("/lecture8/figures/receptive_field.svg", w: 56%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[three stacked $3 times 3$ convs: receptive field $3 -> 5 -> 7$])

== Two small kernels beat one big one #D

Two stacked $3 times 3$ convs cover the same $5 times 5$ field as one $5 times 5$ conv — but cheaper:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, center, center),
  table.header([ ], [*two $3 times 3$*], [*one $5 times 5$*]),
  [receptive field], [$5 times 5$], [$5 times 5$],
  [parameters], [$2 dot 3 dot 3 C^2 = 18 C^2$], [$5 dot 5 C^2 = 25 C^2$],
  [nonlinearities], [$2$], [$1$],
))
#pause
#notebox[Fewer parameters *and* more nonlinearity → why modern nets stack many small $3 times 3$ kernels.]

== Receptive field recursion #D

For stride-1 layers, the receptive field grows by $k_ell - 1$ at each layer:
#pause
$ r_ell = r_(ell - 1) + (k_ell - 1) $
#pause
Worked example — three $3 times 3$ layers, starting from $r_0 = 1$:
#pause
- after layer 1: $r_1 = 1 + (3 - 1) = 3$;
#pause
- after layer 2: $r_2 = 3 + (3 - 1) = 5$;
#pause
- after layer 3: $r_3 = 5 + (3 - 1) = 7$.
#pause
#align(center, text(size: 16pt, fill: MUTED)[strided or pooled layers grow it *much* faster — the receptive field can cover the whole image within a few blocks])

== Interactive: receptive-field grower #I

#interbox(link-to: IA + "receptive-field-grower")[
  Add conv layers one at a time and watch the highlighted receptive field expand back through the network onto the input image.
]
#pause
*Q.* How many $3 times 3$ stride-1 layers give a receptive field of at least $15 times 15$?

// ═══════════════════════════ PART VI — CNN architecture ═══════════════════════════
= The CNN architecture

== The typical pattern #V

A CNN interleaves *convolution*, *nonlinearity*, and *downsampling*, then classifies:
#pause
#cnnpipe
#pause
#align(center, text(size: 16pt, fill: MUTED)[spatial size *shrinks*, channel count *grows* — trading resolution for semantics])

== A feature hierarchy #V

#fig("/lecture8/figures/feature_hierarchy.svg", w: 82%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[early layers → edges · middle → textures & parts · late → objects & semantics])

== The classification head

Convolutions produce features; a small head turns them into class probabilities:
#pause
#headpipe
#pause
$ p = "softmax"(W h + b), quad quad cal(L) = -log p_y $
#pause
#align(center, text(size: 16pt, fill: MUTED)[same cross-entropy loss as an MLP — only the *feature extractor* changed])

== Worked example: a tiny CNN #D

Track the *shapes* through a small classifier on a $32 times 32$ RGB image:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 5.5pt), align: (left, center, left),
  table.header([*Layer*], [*Output shape*], [*Note*]),
  [input], [$32 times 32 times 3$], [RGB image],
  [conv $3 times 3$, 16, $p = 1$], [$32 times 32 times 16$], [same padding],
  [max-pool $2 times 2$, $s = 2$], [$16 times 16 times 16$], [halve spatial],
  [conv $3 times 3$, 32, $p = 1$], [$16 times 16 times 32$], [more channels],
  [global avg pool], [$32$], [collapse $H,W$],
  [linear → 10], [$10$], [class scores],
))

== Tiny CNN: parameter count #D

#pause
$ #text[conv1] = 3 dot 3 dot 3 dot 16 + 16 = 448 $
#pause
$ #text[conv2] = 3 dot 3 dot 16 dot 32 + 32 = 4640 $
#pause
$ #text[linear] = 32 dot 10 + 10 = 330 $
#pause
#result[total $approx 448 + 4640 + 330 = 5418$ parameters]
#pause
#align(center, text(size: 16pt, fill: MUTED)[a full-image MLP needed $10^8$ — the CNN does more with *four orders of magnitude* fewer weights])

== The modern conv block

In practice each "conv" step is a small *block*:
#pause
#align(center, diagram(spacing: 13mm, node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), text(size: 15pt)[Conv], fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL, corner-radius: 3pt, inset: 8pt)
  node((1,0), text(size: 15pt)[Norm], fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE, corner-radius: 3pt, inset: 8pt)
  node((2,0), text(size: 15pt)[Activation], fill: ACC.lighten(82%), stroke: 0.9pt + ACC, corner-radius: 3pt, inset: 8pt)
  for i in range(2) { edge((i,0),(i+1,0), "-|>", stroke: 0.8pt + MUTED) }
}))
#pause
#notebox[*Normalization* (L6) stabilises training; *residual connections* (L6) let gradients skip layers. The same ideas that tamed deep MLPs return — deep CNNs need them just as much.]

== Historical anchors #OPT

#pause
- *LeNet* (1990s) — small CNNs read handwritten *digits* on cheques and mail.
#pause
- *AlexNet* (2012) — a deep CNN trained on GPUs cut ImageNet error sharply, and kicked off the modern deep-learning wave.
#pause
#notebox[The *ingredients* — conv, pooling, ReLU, depth — are what matter here; the exact architectures are landmarks, not laws. #text(fill: MUTED)[(dates and details are approximate)]]

// ═══════════════════════════ PART VII — Backprop through convolution ═══════════════════════════
= Backprop through convolution

== Convolution is a linear operator

For fixed weights, convolution is *linear* in the input:
$ y = K * x $
#pause
#notebox[Linear ⇒ it has a well-defined *transpose*, and backprop is *automatic*. No new machinery — the same chain rule from L4 applies.]

== The two gradients #D

Given the upstream gradient $partial cal(L) slash partial y$, backprop needs two things:
#pause
- *gradient w.r.t. the kernel* — *correlate* the input with the output gradient (each weight is reused everywhere, so its gradient *sums* over all positions);
#pause
- *gradient w.r.t. the input* — *spread* the output gradient back through the kernel — a convolution with the *flipped* kernel.
#pause
#align(center, text(size: 16pt, fill: MUTED)[weight sharing in the forward pass ⇒ gradient *summing* in the backward pass])

== Conv as matrix multiply #OPT

Convolution can be *unrolled* into a single matrix multiply (`im2col`):
$ y = K_"mat" thin x $
#pause
- lay each input patch out as a column, stack the kernel into a matrix;
#pause
- convolution becomes one big matmul — exactly what GPUs are fastest at.
#pause
#notebox[This is how libraries *implement* conv efficiently. The gradient is then just the transpose $K_"mat"^top$ — standard linear-layer backprop.]

// ═══════════════════════════ PART VIII — Practical CNN training ═══════════════════════════
= Training CNNs in practice

== Data augmentation

Images admit *label-preserving* transforms — a flipped cat is still a cat:
#pause
- random *crop* and *resize*; horizontal *flip*;
#pause
- *colour jitter* (brightness, contrast); small *rotations*;
#pause
- then *normalize* to a standard range.
#pause
#notebox[Augmentation multiplies your effective dataset and bakes in invariances — one of the cheapest, most effective regularisers in vision.]

== Input normalization

Standardise each channel using the *training-set* mean and standard deviation:
$ x' = (x - mu) / sigma $
#pause
- keeps inputs on a consistent scale → healthier gradients, faster training;
#pause
- compute $mu, sigma$ on the *train* set only, then reuse them for val / test.

== Transfer learning

Rarely train from scratch — start from a network *pretrained* on a large dataset:
#pause
- keep the pretrained *backbone* (it already knows edges, textures, parts);
#pause
- replace the *head* with one for your classes;
#pause
- *fine-tune* — train the head, optionally unfreeze the backbone with a small LR.
#pause
#result[strong results from *little* data — reuse features, don't relearn them]

== Common failure modes

#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 5.5pt), align: (left, left),
  table.header([*Symptom*], [*Usual cause*]),
  [shape mismatch error], [wrong tensor shape / forgot a dimension],
  [colours look wrong], [channel order (RGB vs BGR, NCHW vs NHWC)],
  [tiny feature map, weak acc], [too much downsampling too early],
  [overfits fast], [no data augmentation],
  [great train, poor test], [`train()` / `eval()` mode bug (norm, dropout)],
))

== Summary

#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 5.5pt), align: (left, left),
  table.header([*Ingredient*], [*What it buys*]),
  [convolution], [local feature extraction, weight sharing],
  [multiple kernels], [many features → channels],
  [padding / stride], [control output resolution],
  [pooling], [downsample + local robustness],
  [receptive field], [how much context a neuron sees],
  [depth], [edges → textures → parts → objects],
))

== Final mental model — one idea

#align(center, text(size: 22pt)[
  An MLP asks *every pixel about every other*. \
  A CNN asks the *right question*: \
  *what local patterns appear, and where?* \
  #v(4pt)
  #text(size: 18pt, fill: MUTED)[locality + weight sharing + hierarchy = vision that scales]
])

#focus-slide[
  Classification tells us *what* is in the image.
  #v(12pt)
  #set text(size: 22pt)
  Next: *where* is it, and *which pixels* belong to it? — detection and segmentation.
]
