// Modern CNN Pipelines and Transfer Learning — Lecture 8B · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture8b/L8b-modern-cnns.typ /tmp/L8b.pdf
//   typst compile --root . --input handout=true lecture8b/L8b-modern-cnns.typ /tmp/L8bh.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Modern CNN Pipelines and Transfer Learning],
  subtitle: [Blocks, Scaling, Residuals, and Reusing Backbones],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"

// ── the modern conv primitive: Conv -> Norm -> Activation ──
#let convblock = align(center, diagram(spacing: 13mm, node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), text(size: 15pt)[Conv], fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL, corner-radius: 3pt, inset: 8pt)
  node((1,0), text(size: 15pt)[Norm], fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE, corner-radius: 3pt, inset: 8pt)
  node((2,0), text(size: 15pt)[Activation], fill: ACC.lighten(82%), stroke: 0.9pt + ACC, corner-radius: 3pt, inset: 8pt)
  for i in range(2) { edge((i,0),(i+1,0), "-|>", stroke: 0.8pt + MUTED) }
}))

// ── Inception module: input fans to four branches -> concat ──
#let inception = align(center, diagram(spacing: (11mm, 6mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let blk(p, lbl, col) = node(p, text(size: 12pt, lbl), fill: col.lighten(82%), stroke: 0.9pt + col, corner-radius: 3pt, inset: 5pt)
  node((0, 0), text(size: 13pt)[input], radius: 9mm, fill: rgb("#EFEEEB"))
  // four parallel branches at four heights
  blk((2, 3), "1×1", TEAL)
  blk((1.4, 1), "1×1", BLUE); blk((2.9, 1), "3×3", TEAL)
  blk((1.4, -1), "1×1", BLUE); blk((2.9, -1), "5×5", TEAL)
  blk((1.4, -3), "3×3 pool", MUTED); blk((2.9, -3), "1×1", TEAL)
  node((4.3, 0), text(size: 13pt)[concat], radius: 10mm, fill: ACC.lighten(82%), stroke: 0.9pt + ACC)
  // edges from input
  edge((0,0), (2,3), "-|>", stroke: 0.7pt + MUTED)
  edge((0,0), (1.4,1), "-|>", stroke: 0.7pt + MUTED)
  edge((0,0), (1.4,-1), "-|>", stroke: 0.7pt + MUTED)
  edge((0,0), (1.4,-3), "-|>", stroke: 0.7pt + MUTED)
  // within-branch chains
  edge((1.4,1), (2.9,1), "-|>", stroke: 0.7pt + MUTED)
  edge((1.4,-1), (2.9,-1), "-|>", stroke: 0.7pt + MUTED)
  edge((1.4,-3), (2.9,-3), "-|>", stroke: 0.7pt + MUTED)
  // to concat
  edge((2,3), (4.3,0), "-|>", stroke: 0.7pt + MUTED)
  edge((2.9,1), (4.3,0), "-|>", stroke: 0.7pt + MUTED)
  edge((2.9,-1), (4.3,0), "-|>", stroke: 0.7pt + MUTED)
  edge((2.9,-3), (4.3,0), "-|>", stroke: 0.7pt + MUTED)
}))

// ── ResNet residual block: conv-norm-relu-conv-norm -> (+) -> relu, identity shortcut ──
#let resblock = align(center, diagram(spacing: (10mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let blk(x, lbl, col) = node((x, 0), text(size: 11.5pt, lbl), fill: col.lighten(82%), stroke: 0.9pt + col, corner-radius: 3pt, inset: 5pt)
  node((0, 0), $x_ell$, radius: 6mm, fill: rgb("#EFEEEB"))
  blk(1, "conv", TEAL); blk(2, "norm", BLUE); blk(3, "ReLU", ACC)
  blk(4, "conv", TEAL); blk(5, "norm", BLUE)
  node((6, 0), [$+$], radius: 5mm, fill: white, stroke: 1pt + INK)
  node((7, 0), text(size: 11.5pt)[ReLU], fill: ACC.lighten(82%), stroke: 0.9pt + ACC, corner-radius: 3pt, inset: 5pt)
  node((8, 0), $x_(ell+1)$, radius: 7mm, stroke: 0.9pt + ACC)
  for i in range(6) { edge((i,0),(i+1,0), "-|>", stroke: 0.8pt + MUTED) }
  edge((6,0),(7,0), "-|>", stroke: 0.8pt + MUTED)
  edge((7,0),(8,0), "-|>", stroke: 0.8pt + MUTED)
  // identity shortcut: x_l over the top, into the (+)
  edge((0,0), (0,-1.4), (6,-1.4), (6,0), "-|>", stroke: 1.1pt + GREEN)
  node((3, -1.4), text(size: 11pt, fill: GREEN)[identity  $x_ell$], stroke: none, fill: none)
}))

// ── the modern classification pipeline ──
#let pipeline = align(center, diagram(spacing: (7.5mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let blk(x, lbl, col) = node((x, 0), text(size: 11.5pt, lbl), fill: col.lighten(85%), stroke: 0.9pt + col, corner-radius: 3pt, inset: 5pt)
  blk(0, "data", INK); blk(1, "augment", BLUE); blk(2, "backbone", TEAL)
  blk(3, "head", ACC); blk(4, "loss", RED); blk(5, "optim", GREEN)
  blk(6, "sched", BLUE); blk(7, "val", TEAL); blk(8, "ckpt", INK)
  for i in range(8) { edge((i,0),(i+1,0), "-|>", stroke: 0.7pt + MUTED) }
}))

// ── backbone / head abstraction: one backbone, three task heads ──
#let backboneheads = align(center, diagram(spacing: (16mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), [image $x$], fill: rgb("#EFEEEB"), stroke: 0.9pt + INK, corner-radius: 3pt, inset: 8pt)
  node((1, 0), align(center, [backbone \ #text(size: 11pt, fill: MUTED)[$F = f(x)$]]), fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL, corner-radius: 3pt, inset: 8pt)
  node((2.2, 1.3), text(size: 12pt)[classification], fill: ACC.lighten(85%), stroke: 0.9pt + ACC, corner-radius: 3pt, inset: 6pt)
  node((2.2, 0), text(size: 12pt)[detection], fill: BLUE.lighten(85%), stroke: 0.9pt + BLUE, corner-radius: 3pt, inset: 6pt)
  node((2.2, -1.3), text(size: 12pt)[segmentation], fill: GREEN.lighten(85%), stroke: 0.9pt + GREEN, corner-radius: 3pt, inset: 6pt)
  edge((0,0), (1,0), "-|>", stroke: 0.8pt + MUTED)
  edge((1,0), (2.2,1.3), "-|>", stroke: 0.8pt + MUTED)
  edge((1,0), (2.2,0), "-|>", stroke: 0.8pt + MUTED)
  edge((1,0), (2.2,-1.3), "-|>", stroke: 0.8pt + MUTED)
}))

// ── the bottleneck sandwich: 1×1 reduce -> 3×3 spatial -> 1×1 expand ──
#let bottleneck = align(center, diagram(spacing: (11mm, 7mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), $x$, radius: 5mm, fill: rgb("#EFEEEB"))
  node((1, 0), align(center, [$1 times 1$ \ #text(size: 10pt, fill: MUTED)[$256 -> 64$]]), fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE, corner-radius: 3pt, inset: 6pt)
  node((2, 0), align(center, [$3 times 3$ \ #text(size: 10pt, fill: MUTED)[$64 -> 64$]]), fill: TEAL.lighten(82%), stroke: 0.9pt + TEAL, corner-radius: 3pt, inset: 6pt)
  node((3, 0), align(center, [$1 times 1$ \ #text(size: 10pt, fill: MUTED)[$64 -> 256$]]), fill: BLUE.lighten(82%), stroke: 0.9pt + BLUE, corner-radius: 3pt, inset: 6pt)
  node((4, 0), $y$, radius: 5mm, stroke: 0.9pt + ACC)
  for i in range(4) { edge((i,0),(i+1,0), "-|>", stroke: 0.8pt + MUTED) }
  node((1, 1.5), text(size: 10pt, fill: MUTED)[squeeze], stroke: none, fill: none)
  node((2, 1.5), text(size: 10pt, fill: MUTED)[cheap spatial], stroke: none, fill: none)
  node((3, 1.5), text(size: 10pt, fill: MUTED)[expand], stroke: none, fill: none)
}))

#title-slide()

// ═══════════════════════════ PART I — Why another CNN lecture ═══════════════════════════
= Why another CNN lecture?

== Where we are

Last lecture built the *convolution* from scratch: locality, weight sharing, hierarchy.
#pause
Our images are tensors, and the *primitive* we assemble networks from is one small block:
$ X in RR^(H times W times C), quad quad #text[Conv] -> #text[Norm] -> #text[Activation] $
#pause
#convblock
#pause
#align(center, text(size: 16pt, fill: MUTED)[we know *how one block works* — today is about *how to design systems of them*])

== Today's question

#align(center, text(size: 22pt, fill: INK)[
  CNN *basics*: how does a convolution work? \
  #v(4pt)
  Today: how do we design CNN *systems* that actually work?
])
#pause
#v(8pt)
#notebox[The raw conv layer is the transistor. This lecture is the *circuit design*: stacking blocks into *backbones* through depth, multi-scale branches, residual connections, efficient factorizations — then *reusing* those backbones on new tasks.]

== The arc of this lecture

#pause
+ *VGG* — depth from stacking small kernels;
#pause
+ *1×1 conv* — mixing channels cheaply;
#pause
+ *Inception* — parallel multi-scale branches;
#pause
+ *ResNet* — residual connections that keep deep nets trainable;
#pause
+ *Efficient CNNs* — depthwise-separable, compound scaling, ConvNeXt;
#pause
+ *Pipeline + transfer* — reusing pretrained backbones on your data.

// ═══════════════════════════ PART II — VGG ═══════════════════════════
= VGG — depth from simplicity

== One design principle

VGG asked: what if we *only* ever use small $3 times 3$ convs, and just go *deep*?
#pause
- every conv is $3 times 3$, stride $1$, pad $1$ — so spatial size is *preserved*;
#pause
- downsampling is left entirely to $2 times 2$ *max-pooling*;
#pause
- stack $16$–$19$ such layers into a uniform, repetitive network.
#pause
#result[one kernel size, one stride — depth is the only knob]

== Why so many $3 times 3$ kernels? #D

Stacking small kernels *grows the receptive field* while adding *nonlinearities*:
#pause
- two $3 times 3$ convs see a $5 times 5$ input region;
#pause
- three $3 times 3$ convs see a $7 times 7$ region;
#pause
$ r_ell = r_(ell-1) + (k - 1), quad quad 1 ->^(3 times 3) 3 ->^(3 times 3) 5 ->^(3 times 3) 7 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[same field as one big kernel — but with *more* ReLUs in between, so a richer function])

== Two small beat one big #D

Cover a $5 times 5$ field two ways, and count parameters (channels $C$ in and out):
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 7pt), align: (left, center, center),
  table.header([ ], [*two $3 times 3$*], [*one $5 times 5$*]),
  [receptive field], [$5 times 5$], [$5 times 5$],
  [parameters], [$2 dot 3 dot 3 C^2 = 18 C^2$], [$5 dot 5 C^2 = 25 C^2$],
  [nonlinearities], [$2$], [$1$],
))
#pause
#result[$18 C^2 < 25 C^2$ — fewer parameters *and* more nonlinearity]

== The VGG stage #V

A stage = a couple of $3 times 3$ convs, a pool, and *double* the channels:
#pause
#fig("/lecture8b/figures/vgg_stage.svg", w: 82%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[$64 -> 128 -> 256 -> 512$ channels as spatial size halves — resolution traded for semantics])

== What VGG taught us

#pause
- *depth matters* — a deep stack of simple blocks beats a shallow clever one;
#pause
- *small kernels are enough* — no need for large, exotic filters;
#pause
- *uniform blocks* make networks easy to design, scale, and reason about.
#pause
#alertbox[But VGG is *parameter-heavy*: its fully-connected head alone holds over $100$ million weights. The next ideas are largely about *removing that waste*.]

== Interactive: stacking small kernels #I

#interbox(link-to: IA + "receptive-field-grower")[
  Add $3 times 3$ conv layers one at a time and watch the receptive field grow $3 -> 5 -> 7 -> dots$ — the exact mechanism behind VGG's "go deep with small kernels".
]
#pause
*Q.* How many stacked $3 times 3$ convs match the receptive field of one $9 times 9$ kernel?

// ═══════════════════════════ PART III — 1×1 convolution ═══════════════════════════
= The $1 times 1$ convolution

== A convolution with no neighbours

Take a conv whose kernel is a single pixel. At each location $(h, w)$:
#pause
$ y_(h w) = W thin x_(h w) + b, quad quad x_(h w) in RR^(C_"in") -> y_(h w) in RR^(C_"out") $
#pause
- it looks at *one* pixel, across *all* its channels;
#pause
- it *mixes channels*, but touches *no spatial neighbours*;
#pause
- $W$ is just a $C_"out" times C_"in"$ matrix, shared over every position.
#pause
#align(center, text(size: 16pt, fill: MUTED)[a $3 times 3$ asks "what pattern is *around* here?"; a $1 times 1$ asks "what *combination* of channels is here?"])

== $1 times 1$ is a per-pixel MLP #V

#fig("/lecture8b/figures/one_by_one.svg", w: 56%)
#pause
#align(center, text(size: 15pt, fill: MUTED)[the *same* tiny MLP $W$ is applied independently at every spatial location])

== What a $1 times 1$ conv is for

#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*Use*], [*What it does*]),
  [change channel dim], [project $C_"in" -> C_"out"$ (grow or shrink)],
  [bottleneck], [squeeze channels before an expensive conv],
  [feature mixing], [recombine channels into new features],
  [residual projection], [match shapes on a skip connection],
  [segmentation head], [per-pixel class scores],
))

== Worked example: $1 times 1$ vs $3 times 3$ #D

Project a feature map $H times W times 256 -> H times W times 64$. Count the weights:
#pause
$ #text[1×1:] quad 1 dot 1 dot 256 dot 64 + 64 = "16,448" $
#pause
$ #text[3×3:] quad 3 dot 3 dot 256 dot 64 + 64 = "147,520" $
#pause
$ "147,520" \/ "16,448" approx 9 times $
#pause
#result[same channel change, $9 times$ fewer parameters]
#pause
#align(center, text(size: 16pt, fill: MUTED)[if you only need to *re-mix channels*, spending a $3 times 3$ on it is wasteful])

== The bottleneck idea

Instead of a fat $3 times 3$ that maps $256 -> 256$, *sandwich* it between two $1 times 1$s:
#pause
$ underbrace(1 times 1, 256 -> 64) quad -> quad underbrace(3 times 3, 64 -> 64) quad -> quad underbrace(1 times 1, 64 -> 256) $
#pause
- the $1 times 1$s *squeeze* then *restore* the channel count;
#pause
- the expensive $3 times 3$ runs in the *cheap* $64$-channel space;
#pause
- far less compute, with similar representational power.
#pause
#notebox[This *bottleneck* is the engine inside ResNet, Inception, and MobileNet — reduce channels, do the spatial work, expand back.]

== The bottleneck, visually #V

Two $1 times 1$s wrap the costly $3 times 3$, so the spatial work runs in a *thin* channel space:
#pause
#bottleneck
#pause
#align(center, text(size: 15pt, fill: MUTED)[squeeze $256 -> 64$, do the $3 times 3$ in the cheap $64$-channel space, then expand $64 -> 256$])

// ═══════════════════════════ PART IV — Inception ═══════════════════════════
= Inception — multi-scale, factorized

== Which kernel size? All of them

Objects appear at *many scales* — an eye is tiny, a face is large. Why pick one kernel?
#pause
- run several kernel sizes *in parallel*: $1 times 1$, $3 times 3$, $5 times 5$, and a pool;
#pause
- *concatenate* their outputs along the channel axis;
#pause
- let training decide how much to lean on each scale.
#pause
#result[don't choose a receptive field — offer *several* and combine]

== The Inception module #V

#inception
#pause
#align(center, text(size: 15pt, fill: MUTED)[four parallel branches → concat channels · note the $1 times 1$ *reductions* before the costly convs])

== Why the $1 times 1$ reductions? #D

A $5 times 5$ conv straight onto $C = 256$, producing $128$ channels, is expensive:
#pause
$ #text[direct 5×5:] quad 5 dot 5 dot 256 dot 128 = "819,200" $
#pause
Reduce channels *first* with a $1 times 1$ ($256 -> 32$), then do the $5 times 5$ ($32 -> 128$):
#pause
- $1 times 1$ reduce: $256 dot 32 = "8,192"$;
#pause
- $5 times 5$ on the thin map: $25 dot 32 dot 128 = "102,400"$;
#pause
$ "8,192" + "102,400" = "110,592" $
#pause
#result[$approx 7.4 times$ cheaper — same output shape]

== The Inception lesson

#pause
- *multi-scale in parallel* — capture fine and coarse structure in one block;
#pause
- *$1 times 1$ reductions* — put the expensive spatial convs in a low-channel space;
#pause
- *put computation where it matters* — spend FLOPs on spatial work, not channel bulk.
#pause
#notebox[Inception is Part III's bottleneck idea, applied *per branch*. The recurring move: shrink channels, do spatial work, then combine.]

== Interactive: cost of a block #I

#interbox(link-to: IA + "inception-cost")[drag the $1 times 1$ reduction width and watch a branch's FLOPs collapse, then rise again as you widen the reduced space.]
#pause
*Q.* If the $1 times 1$ reduced $256 -> 64$ instead of $256 -> 32$, would the branch be cheaper or dearer than the direct $5 times 5$?

// ═══════════════════════════ PART V — ResNet ═══════════════════════════
= ResNet — the residual backbone

== Deeper should not be worse

Add layers to a working network. In principle the extra layers could learn the *identity* and do no harm.
#pause
#alertbox[Yet very deep *plain* nets trained *worse* — higher training error than shallower ones. This is not overfitting; the optimizer simply *cannot fit* them. Deep plain stacks are hard to optimize.]
#pause
#result[ResNet's fix: make it *easy* for a block to represent "leave the input alone"]

== The residual block #V

Instead of learning a full mapping $H(x)$, learn a *residual* $F(x)$ added to the input:
#pause
$ x_(ell+1) = x_ell + F_ell (x_ell) $
#pause
#resblock
#pause
#align(center, text(size: 15pt, fill: MUTED)[the green *identity* path carries $x_ell$ straight to the sum — the block only learns the *correction*])

== Why residuals train #D

Look at the gradient flowing back through one block. Since $x_(ell+1) = x_ell + F_ell (x_ell)$:
#pause
$ (partial cal(L))/(partial x_ell) = (partial cal(L))/(partial x_(ell+1)) dot (I + (partial F_ell)/(partial x_ell)) $
#pause
- the $I$ term is a *direct highway*: gradient reaches $x_ell$ *undiminished*;
#pause
- even if $partial F_ell slash partial x_ell$ is tiny, the identity keeps the signal alive.
#pause
#result[skip connections defeat vanishing gradients — the same trick that tamed deep MLPs]

== Basic vs bottleneck block

Two flavours of residual block trade depth-per-block against cost:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left, left),
  table.header([ ], [*basic*], [*bottleneck*]),
  [structure], [$3 times 3 -> 3 times 3$], [$1 times 1 -> 3 times 3 -> 1 times 1$],
  [$1 times 1$ role], [—], [reduce then expand channels],
  [used in], [ResNet-18/34], [ResNet-50/101/152],
))
#pause
#notebox[The bottleneck block *is* the Part III sandwich: $1 times 1$ squeeze → $3 times 3$ spatial → $1 times 1$ expand, wrapped in a skip connection.]

== When shapes change: projection shortcut

At a downsampling stage the spatial size halves and channels double — so $x_ell$ and $F(x_ell)$ *no longer match*.
#pause
Fix the shortcut with a $1 times 1$ conv (stride $2$) that reshapes the identity:
$ x_(ell+1) = P(x_ell) + F(x_ell), quad quad P = #text[$1 times 1$ conv, stride $2$] $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the shortcut is *usually* the identity; only at shape changes does it become a learned $1 times 1$ projection])

== ResNet as a backbone #V

A ResNet is a stack of residual stages producing feature maps at *decreasing resolution*:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, center, left),
  table.header([*Stage*], [*Resolution*], [*Feeds*]),
  [C2], [$1 slash 4$], [fine detail — small objects],
  [C3], [$1 slash 8$], [mid-level parts],
  [C4], [$1 slash 16$], [larger context],
  [C5], [$1 slash 32$], [global semantics],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[one backbone → classification, detection, segmentation, FPN heads all tap these maps])

== Interactive: residual flow #I

#interbox(link-to: IA + "resnet")[
  Toggle the skip connection on and off and watch gradients flow (or vanish) through a deep stack — see the identity path keep the signal alive all the way back.
]

// ═══════════════════════════ PART VI — Efficient CNNs ═══════════════════════════
= Efficient CNNs

== The cost of a standard conv

A $k times k$ conv from $C_"in"$ to $C_"out"$ channels over an $H times W$ map costs:
#pause
$ #text[cost] = k^2 dot C_"in" dot C_"out" dot H W $
#pause
- one factor for the *spatial* window ($k^2$);
#pause
- one factor for *every input–output channel pair* ($C_"in" dot C_"out"$);
#pause
- that channel-pair product is where most of the cost hides.
#pause
#result[idea: *separate* the spatial part from the channel-mixing part]

== Depthwise separable convolution #V

Split one conv into two cheaper steps:
#pause
#fig("/lecture8b/figures/depthwise_sep.svg", w: 74%)
#pause
- *depthwise* — one $k times k$ filter *per channel*: spatial only, cost $k^2 dot C_"in" dot H W$;
#pause
- *pointwise* — a $1 times 1$ mixing channels: cost $C_"in" dot C_"out" dot H W$.
#pause
#align(center, text(size: 15pt, fill: MUTED)[MobileNet's core block — the two factors *add* instead of *multiplying*])

== Worked cost: separable vs standard #D

A $3 times 3$ conv with $C_"in" = C_"out" = 128$ (drop the shared $H W$ factor):
#pause
$ #text[standard:] quad 3^2 dot 128 dot 128 = "147,456" $
#pause
- depthwise — one $3 times 3$ per channel: $3^2 dot 128 = "1,152"$;
#pause
- pointwise — a $1 times 1$ mixing channels: $128 dot 128 = "16,384"$;
#pause
$ #text[separable total:] quad "1,152" + "16,384" = "17,536" $
#pause
#result[$"147,456" slash "17,536" approx 8.4 times$ cheaper]

== Cost, side by side #V

#two(
  bars((16448, 147520), labels: ("1×1", "3×3"), horizontal: true,
    color: MUTED, highlight: ("0": GREEN), digits: 0, title: [params: $256 -> 64$ projection]),
  bars((147456, 17536), labels: ("standard", "separable"), horizontal: true,
    color: MUTED, highlight: ("1": GREEN), digits: 0, title: [cost: $3 times 3$, $128 -> 128$]),
)
#pause
#align(center, text(size: 15pt, fill: MUTED)[left: $1 times 1$ vs $3 times 3$ params · right: standard vs depthwise-separable cost])

== EfficientNet: compound scaling

You can make a net bigger three ways — *depth* (more layers), *width* (more channels), *resolution* (bigger input).
#pause
- scaling *one* alone hits diminishing returns quickly;
#pause
- EfficientNet scales all three *together*, in a fixed ratio;
#pause
#result[balanced growth gives more accuracy per FLOP than any single axis]

== ConvNeXt: a modernized ConvNet #OPT

Take a plain ResNet and apply every modern trick, one at a time:
#pause
- *larger* depthwise kernels ($7 times 7$); *inverted* bottlenecks;
#pause
- *fewer* activations and norms per block; *LayerNorm* in place of BatchNorm;
#pause
- modern training recipe (augmentation, schedules, AdamW).
#pause
#notebox[No attention, no self-attention — just a carefully modernized *convolutional* net that competes with Transformer-era backbones. Convolution is far from obsolete.]

== The efficiency axes

Each idea in this lecture buys a *different* kind of efficiency:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 5.5pt), align: (left, left),
  table.header([*Technique*], [*What it saves*]),
  [$1 times 1$ bottleneck], [parameters + FLOPs],
  [depthwise separable], [FLOPs],
  [global average pooling], [parameters (no giant FC head)],
  [residual connections], [trainability (deep nets converge)],
  [compound scaling], [accuracy per FLOP],
  [transfer learning], [data + training time],
))

// ═══════════════════════════ PART VII — Classification pipeline ═══════════════════════════
= The classification pipeline

== The modern pipeline #V

Training an image classifier today is an *assembly line* of standard parts:
#pause
#pipeline
#pause
#align(center, text(size: 15pt, fill: MUTED)[data → augment → *pretrained backbone* → head → loss → optimizer → scheduler → validate → checkpoint])

== Preprocessing: match the backbone

Standardise each channel with a fixed mean and standard deviation:
$ x' = (x - mu) / sigma $
#pause
#alertbox[Use the *same* normalization the backbone was pretrained with (its ImageNet $mu, sigma$). Feeding differently-scaled inputs to a pretrained net silently *breaks* its features.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[a "model expects these exact stats" contract — the single most common transfer bug])

== The classification head

The backbone outputs a feature map; a tiny head turns it into class probabilities:
#pause
$ h_c = 1/(H' W') sum_(i,j) F_(i j c) quad quad (#text[global average pool]) $
#pause
$ z = W h + b, quad quad p = "softmax"(z), quad quad cal(L) = -log p_y $
#pause
#align(center, text(size: 16pt, fill: MUTED)[GAP (from Network-in-Network) collapses $H' times W' times C$ to a length-$C$ vector — no giant flatten])

== Worked example: shapes through the head #D

Track shapes and count the head's parameters ($K = 10$ classes):
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 5.5pt), align: (left, center, left),
  table.header([*Stage*], [*Shape*], [*Note*]),
  [input], [$224 times 224 times 3$], [RGB image],
  [backbone $F$], [$7 times 7 times 2048$], [ResNet-50 output],
  [global avg pool], [$2048$], [collapse $H', W'$],
  [linear → $K$], [$10$], [class scores],
))
#pause
$ #text[head params] = 2048 dot 10 + 10 = "20,480" + 10 = "20,490" $
#pause
#align(center, text(size: 15pt, fill: MUTED)[a $20$k-parameter head on a $25$-million-parameter backbone — the head is *tiny*])

// ═══════════════════════════ PART VIII — Transfer learning ═══════════════════════════
= Transfer learning

== Why it works

Early and middle layers learn *generic* visual features — edges, textures, parts — that recur across *every* vision task.
#pause
$ theta_"pretrained" quad -->^"adapt" quad theta_"your task" $
#pause
- so *start* from weights pretrained on a large dataset (e.g. ImageNet);
#pause
- *keep* the reusable feature extractor; only the task-specific top must change.
#pause
#result[reuse features, don't relearn them — strong results from *little* data]

== Mode 1: feature extractor #V

#two(
  fig("/lecture8b/figures/transfer_modes.svg", w: 100%),
  [
    *Freeze* the backbone, train *only* a new head.
    #v(4pt)
    - backbone weights are fixed — no gradient update;
    - only the small head learns.
    #v(4pt)
    Good for *small data*, *similar classes*, or *limited compute*.
  ],
  r: (1.4fr, 1fr),
)

== Mode 2: finetuning

*Initialise* from pretrained weights, then train *all* layers — but gently:
#pause
$ eta_"backbone" < eta_"head" $
#pause
- the head starts random → needs a *larger* learning rate;
#pause
- the backbone is already good → a *small* rate nudges it without erasing it.
#pause
#align(center, text(size: 16pt, fill: MUTED)[good for *larger* datasets or a *different domain* (medical, satellite, art) where features must shift])

== A safe recipe: probe, then finetune

#pause
+ *freeze* the backbone, train the head to convergence (a *linear probe*);
#pause
+ *unfreeze* the last block or two, continue with a small learning rate;
#pause
+ *optionally* unfreeze everything, using *discriminative* rates — lower for earlier layers.
#pause
#notebox[Training the head first stops large early gradients from a random head *wrecking* the pretrained features on step one.]

== Worked example: how few new weights? #D

$"5,000"$ images, $K = 6$ classes, a ResNet-50 backbone. Replace its $2048 -> 1000$ head:
#pause
$ #text[new head:] quad 2048 dot 6 + 6 = "12,288" + 6 = "12,294" quad #text[parameters] $
#pause
#result[$approx 12$k weights to fit — versus *millions* if you trained the whole net from scratch]
#pause
#align(center, text(size: 15pt, fill: MUTED)[with only $5$k images, fitting $12$k new parameters is usually the lower-variance starting point; end-to-end adaptation of a $25$M pretrained backbone can still work, but needs a smaller LR and careful validation.])

== Transfer pitfalls

#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 5pt), align: (left, left),
  table.header([*Pitfall*], [*Symptom*]),
  [wrong normalization], [pretrained features silently degrade],
  [finetune LR too high], [catastrophic forgetting of good weights],
  [forgot `train()`/`eval()`], [BatchNorm / dropout misbehave],
  [inconsistent augmentation], [train/val mismatch],
  [class imbalance], [head collapses to majority class],
  [frozen BatchNorm stats], [wrong running stats on new domain],
))

== Interactive: transfer playground #I

#interbox(link-to: IA + "transfer-learning-playground")[slide the dataset size and choose freeze vs finetune, then watch accuracy trade off against overfitting.]
#pause
*Q.* You have $200$ labelled images in a domain close to ImageNet. Freeze or finetune?

// ═══════════════════════════ PART IX — From backbones to dense prediction ═══════════════════════════
= From backbones to dense prediction

== Why this belongs before detection

Detection and segmentation do not start over — they *reuse* the very backbones we just built.
#pause
#two(
  notebox[*Classification* \ image → vector → *one* class label. The spatial map is *collapsed* away.],
  notebox[*Dense prediction* \ image → *spatial* feature maps → boxes / masks. The spatial map is *kept*.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[same feature extractor; the difference is entirely in the *head* and what it does with $F$])

== One backbone, many heads #V

#backboneheads
#pause
$ F = f_"backbone"(x); quad z = f_"cls"(F); quad {b_i, c_i, s_i} = f_"det"(F); quad M = f_"seg"(F) $
#pause
#align(center, text(size: 15pt, fill: MUTED)[the backbone is shared infrastructure; swap the *head* to change the task])

== Preview: feature pyramids

Objects come at *wildly* different scales — a distant sign, a nearby truck. One resolution cannot serve both.
#pause
- a Feature Pyramid Network (FPN) taps the backbone at *several* stages;
#pause
- it builds multi-resolution maps $P_2, P_3, P_4, P_5$ — fine to coarse;
#pause
- small objects use the fine maps; large objects use the coarse ones.
#pause
#align(center, text(size: 16pt, fill: MUTED)[a brief preview — the detection lecture builds this properly on top of the C2–C5 maps we saw])

// ═══════════════════════════ PART X — Summary ═══════════════════════════
= Summary

== The architectures at a glance

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*Architecture*], [*Key idea*]),
  [VGG], [stack many small $3 times 3$ convs — go deep],
  [Inception], [parallel multi-scale branches + $1 times 1$ reductions],
  [ResNet], [residual learning — keep deep nets trainable],
  [MobileNet], [depthwise separable convs — cheap FLOPs],
  [EfficientNet], [compound scaling of depth/width/resolution],
  [ConvNeXt], [a modernized ConvNet, no attention needed],
))

== The design principles

#pause
+ *exploit local spatial structure* — convolutions, not full connections;
#pause
+ *share weights* across positions — one detector, used everywhere;
#pause
+ *preserve trainability* — residual connections and normalization;
#pause
+ *control compute* — $1 times 1$ bottlenecks and separable convs;
#pause
+ *reuse pretrained backbones* — transfer beats training from scratch.

== Final mental model — one idea

#align(center, text(size: 22pt)[
  A raw conv is just a *feature detector*. \
  Modern CNNs turn it into a reusable *backbone* \
  through *blocks*, *scaling*, *residuals*, and *transfer*. \
  #v(4pt)
  #text(size: 18pt, fill: MUTED)[design the system, then reuse it — don't train from scratch]
])

#focus-slide[
  Classification answers *what* is in the image.
  #v(12pt)
  #set text(size: 22pt)
  Next: *object detection* — what is in the image, *and where*? — built on exactly these backbones.
]
