// Semantic and Instance Segmentation — Lecture 10 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture10/L10-segmentation.typ
//   typst compile --root . --input handout=true lecture10/L10-segmentation.typ
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [Semantic and Instance Segmentation],
  subtitle: [Per-pixel labels: FCN, U-Net, Dice, and Mask R-CNN],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let enc = rgb("#DCEBEB")
#let dec = rgb("#FDECD6")
#let cream = rgb("#EFEEEB")

// ── the CV task ladder: classification -> detection -> segmentation ──
#let taskladder = align(center, diagram(spacing: (12mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), [classification \ #text(size: 14pt, fill: MUTED)[*what?*]], fill: white)
  node((1,0), [detection \ #text(size: 14pt, fill: MUTED)[*what + where (box)*]], fill: enc, stroke: 0.9pt + TEAL)
  node((2,0), [segmentation \ #text(size: 14pt, fill: MUTED)[*what at each pixel*]], fill: dec, stroke: 1.1pt + ACC)
  edge((0,0),(1,0),"-|>",stroke:0.8pt+MUTED)
  edge((1,0),(2,0),"-|>",stroke:0.8pt+MUTED)
}))

// ── encoder-decoder: image -> context -> upsample -> per-pixel prediction ──
#let encdec = align(center, diagram(spacing: (13mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), [image \ $H times W$], fill: white)
  node((1,0), [encoder \ #text(size: 14pt)[↓ downsample \ build context]], fill: enc, stroke: 0.9pt + TEAL)
  node((2,0), [bottleneck \ #text(size: 14pt)[coarse, semantic]], fill: cream)
  node((3,0), [decoder \ #text(size: 14pt)[↑ upsample \ localize]], fill: dec, stroke: 0.9pt + ACC, shape: fletcher.shapes.rect)
  node((4,0), [prediction \ $H times W times K$], stroke: 0.9pt + ACC)
  edge((0,0),(1,0),"-|>",stroke:0.8pt+MUTED)
  edge((1,0),(2,0),"-|>",stroke:0.8pt+MUTED)
  edge((2,0),(3,0),"-|>",stroke:0.8pt+MUTED)
  edge((3,0),(4,0),"-|>",stroke:0.8pt+MUTED)
}))

// ── U-Net: contracting path down, expanding path up, horizontal skips ──
#let unet = align(center, diagram(spacing: (10mm, 7.5mm), node-stroke: 0.9pt + INK, node-fill: white, {
  // contracting (descending)
  node((0,0), [H×W], fill: enc, stroke: 0.9pt + TEAL)
  node((1,1), [H/2], fill: enc, stroke: 0.9pt + TEAL)
  node((2,2), [H/4], fill: enc, stroke: 0.9pt + TEAL)
  // bottleneck (bottom of the U)
  node((3,3), [H/8 \ #text(size: 13pt)[context]], fill: cream)
  // expanding (ascending)
  node((4,2), [H/4], fill: dec, stroke: 0.9pt + ACC)
  node((5,1), [H/2], fill: dec, stroke: 0.9pt + ACC)
  node((6,0), [H×W], fill: dec, stroke: 0.9pt + ACC)
  node((7,0), [seg map \ #text(size: 13pt)[H×W×K]], stroke: 0.9pt + ACC)
  // contracting path
  edge((0,0),(1,1),"-|>",stroke:0.8pt+TEAL)
  edge((1,1),(2,2),"-|>",stroke:0.8pt+TEAL)
  edge((2,2),(3,3),"-|>",stroke:0.8pt+TEAL)
  // expanding path
  edge((3,3),(4,2),"-|>",stroke:0.8pt+ACC)
  edge((4,2),(5,1),"-|>",stroke:0.8pt+ACC)
  edge((5,1),(6,0),"-|>",stroke:0.8pt+ACC)
  edge((6,0),(7,0),"-|>",stroke:0.8pt+MUTED)
  // skip connections (dashed, green, horizontal)
  edge((0,0),(6,0),"--|>",stroke:1.0pt+GREEN, label: text(fill: GREEN, size: 12pt)[copy])
  edge((1,1),(5,1),"--|>",stroke:1.0pt+GREEN)
  edge((2,2),(4,2),"--|>",stroke:1.0pt+GREEN)
}))

// ── panoptic = semantic (stuff) + instance (things), merged per pixel ──
#let panopticdiag = align(center, diagram(spacing: (13mm, 5mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let rn(pos, lbl, c, fill) = node(pos, lbl, shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 7pt, stroke: 0.9pt + c, fill: fill)
  rn((0, 0), [semantic \ #text(size: 12pt, fill: MUTED)[*stuff*: sky, road]], TEAL, enc)
  rn((0, 1), [instance \ #text(size: 12pt, fill: MUTED)[*things*: car, person]], ACC, dec)
  rn((2, 0.5), [panoptic \ #text(size: 12pt, fill: MUTED)[every pixel: (class, id)]], ACC, cream)
  edge((0, 0), (2, 0.5), "-|>", stroke: 0.9pt + TEAL)
  edge((0, 1), (2, 0.5), "-|>", stroke: 0.9pt + ACC)
}))

// ── Mask R-CNN: backbone -> RPN -> RoIAlign -> {cls, box, mask} heads ──
#let maskrcnn = align(center, diagram(spacing: (12mm, 7mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), [image], fill: white)
  node((1,0), [backbone \ #text(size: 13pt)[CNN]], fill: enc, stroke: 0.9pt + TEAL)
  node((2,0), [RPN \ #text(size: 13pt)[proposals]], fill: cream)
  node((3,0), [RoIAlign], fill: cream)
  node((4,-1), [class $hat(y)$], stroke: 0.9pt + INK)
  node((4,0), [box $hat(b)$], stroke: 0.9pt + INK)
  node((4,1), [mask $hat(m)$], fill: dec, stroke: 1.2pt + ACC)
  edge((0,0),(1,0),"-|>",stroke:0.8pt+MUTED)
  edge((1,0),(2,0),"-|>",stroke:0.8pt+MUTED)
  edge((2,0),(3,0),"-|>",stroke:0.8pt+MUTED)
  edge((3,0),(4,-1),"-|>",stroke:0.8pt+MUTED)
  edge((3,0),(4,0),"-|>",stroke:0.8pt+MUTED)
  edge((3,0),(4,1),"-|>",stroke:1.1pt+ACC)
}))

#title-slide()

// ═══════════════════════════ PART I — From boxes to pixels ═══════════════════════════
= From boxes to pixels

== The computer-vision task ladder

Each task asks *more* of the same image.
#pause
#v(4pt)
#taskladder
#pause
#v(4pt)
#align(center, text(size: 16pt, fill: MUTED)[last lecture ended at *detection*; today we go all the way to *every pixel*.])

== Recap: what detection gave us

For each object, detection returns a *box + class + score*:
$ x quad -> quad { (hat(b)_i, hat(y)_i, hat(s)_i) }_(i=1)^N $
#pause
- $hat(b)_i$ — a *rectangle* $(x, y, w, h)$;
- $hat(y)_i$ — the class;
- $hat(s)_i$ — a confidence, filtered by *NMS*.
#pause
#notebox[A box says *"an object is roughly here."* It never traces the object's actual *shape*.]

== The limit of a box

#pause
- a box is *axis-aligned* — a diagonal or curved object wastes most of it;
- overlapping objects share pixels — which pixel belongs to whom?
- some questions need the *outline*: tumor area, road surface, free space.
#pause
#result[we want a label for *every pixel*, not a rectangle]

== Segmentation: label every pixel #V

#fig("/lecture10/figures/pixel_grid.svg", w: 52%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[the output has the *same spatial size* as the input — one class per pixel.])

== Semantic segmentation — the task

Assign a class to *every* pixel $(h, w)$:
$ x quad -> quad y_(h w) in {1, dots, K} quad forall thin h, w $
#pause
- output is a *label map* of size $H times W$;
- *all* pixels of the same class look identical — two cats merge into one "cat" region;
#pause
#notebox[Semantic segmentation answers *"what class is this pixel?"* — it does *not* count objects.]

== Instance segmentation — the task

Return a *mask per object*, like detection but with pixels:
$ x quad -> quad { (hat(y)_i, hat(b)_i, hat(m)_i, hat(s)_i) }_(i=1)^N $
#pause
- $hat(m)_i$ — a *binary mask* over the object's pixels;
- two cats now become *two separate* masks — objects are *counted and separated*.
#pause
#result[detection's boxes, upgraded to *pixel-accurate masks*]

== Panoptic segmentation — one label for everything

Give *every* pixel a pair — a class *and* an instance id:
$ x quad -> quad (c_(h w), thin "id"_(h w)) quad forall thin h, w $
#pause
#two(
  notebox[*stuff* — amorphous regions (sky, road, grass): class only, *no instance*.],
  notebox[*things* — countable objects (car, person): class *and* a distinct instance id.],
)
#pause
#v(2pt)
#panopticdiag
#align(center, text(size: 15pt, fill: MUTED)[semantic (stuff) *+* instance (things) → panoptic, with *no gaps or overlaps*.])

== Semantic vs instance, side by side #V

#fig("/lecture10/figures/seg_taxonomy.svg", w: 82%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[same scene · *left*: colour by class · *right*: colour by object.])

== The spine of this lecture

#align(center, text(size: 23pt, fill: INK)[
  Detection gives *boxes*; \
  *segmentation labels every pixel.*
])
#pause
#v(8pt)
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 7pt), align: (left, left, left),
  table.header([*Task*], [*Output*], [*Counts objects?*]),
  [semantic seg], [per-pixel class map], [no],
  [instance seg], [per-object mask], [yes],
  [panoptic seg], [per-pixel (class, id)], [yes (things only)],
))

// ═══════════════════════════ PART II — Semantic segmentation ═══════════════════════════
= Semantic segmentation

== Per-pixel classification

Segmentation is *classification, run at every pixel*:
$ p_(h w k) = p(y_(h w) = k mid(|) x) $
#pause
- for each pixel $(h, w)$ the network emits a vector of $K$ class scores;
- a *softmax over the $K$ channels* turns scores into probabilities;
#pause
#notebox[Same loss you already know from image classification — just applied $H times W$ times, once per pixel.]

== The output is a tensor

A classifier outputs *one* vector of length $K$. Segmentation outputs *a grid* of them:
$ "logits" in RR^(H times W times K) quad -> quad "softmax over " K quad -> quad p in RR^(H times W times K) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the *channel* dimension holds class probabilities; $"argmax"_k$ per pixel gives the label map.])

== Pixelwise cross-entropy #D

Sum the ordinary cross-entropy over *all pixels* and average:
$ cal(L) = -1/(H W) sum_(h, w) log p_(h w, thin y_(h w)) $
#pause
- $y_(h w)$ — the *true* class at pixel $(h, w)$;
- $p_(h w, thin y_(h w))$ — the probability the model gave to that *correct* class;
#pause
#result[each pixel contributes its own classification loss]

== Reading the loss

$ cal(L) = -1/(H W) sum_(h, w) underbrace(-log p_(h w, thin y_(h w)), "loss at one pixel") $
#pause
- a *confident, correct* pixel ($p -> 1$) contributes $≈ 0$;
- a *confident, wrong* pixel ($p -> 0$) contributes a *large* penalty;
#pause
#align(center, text(size: 16pt, fill: MUTED)[the $1/(H W)$ makes it a *mean* — it treats every pixel as equally important (we will question this soon).])

== We need dense prediction

#pause
- a classification CNN ends in *pooling + fully-connected* layers → one vector;
- that *destroys spatial layout* — exactly what a pixel map needs to keep.
#pause
#result[how do we turn a "one-label" network into an "$H times W$-label" network?]

== Fully convolutional networks (FCN)

Replace the final *fully-connected* layers with *$1 times 1$ convolutions*:
#pause
- a $1 times 1$ conv is an FC layer applied *at every spatial location*;
- the net now accepts *any input size* and emits a *coarse label map*, not a vector;
#pause
#notebox["Fully convolutional" = *no FC layers at all*. Output is a spatial map of class scores — the first end-to-end dense-prediction net.]

== FCN: coarse semantics meet fine detail

Deep layers know *what*; shallow layers know *where*.
#pause
#two(
  notebox[*Deep* features — rich *semantics*, but *coarse* (heavily downsampled).],
  notebox[*Shallow* features — precise *location / edges*, but weak semantics.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[FCN *fuses* a coarse deep prediction with shallow appearance to sharpen boundaries.])

== Encoder–decoder #V

The general shape of every modern segmentation net:
#pause
#v(4pt)
#encdec
#pause
#v(4pt)
#align(center, text(size: 16pt, fill: MUTED)[*encoder* shrinks space to gather context · *decoder* grows it back to pixel resolution.])

== Upsampling in the decoder

The decoder must *increase* spatial resolution. Two common ways:
#pause
- *transposed convolution* — a learned upsampling ("deconv");
- *bilinear upsample + conv* — cheap interpolation, then refine.
#pause
#notebox[Downsampling was *pooling / stride*; upsampling *undoes* it — but the fine detail that pooling threw away is gone unless we bring it back another way.]

== U-Net #V

Symmetric encoder–decoder with *skip connections* at every level:
#pause
#v(2pt)
#unet
#pause
#align(center, text(size: 15pt, fill: MUTED)[contracting path (teal) captures context · expanding path (orange) localizes · #text(fill: GREEN)[green skips] copy detail across.])

== Why skip connections

The bottleneck knows *what* but has lost *where*. Skips restore the *where*.
#pause
- each skip *copies* high-resolution encoder features to the matching decoder level;
- the decoder *concatenates* them with its upsampled features before predicting;
#pause
#result[coarse semantics (from depth) + fine detail (from skips) = sharp masks]

== U-Net: born in medical imaging #OPT

#pause
- designed for *biomedical* segmentation, where images are large and labels scarce;
- works from *very few* annotated images (heavy augmentation);
- the skip-connected encoder–decoder is now the *default backbone* for dense prediction — including diffusion models.

== Interactive: encoder–decoder / U-Net #I

#interbox(link-to: IA + "unet")[
  Walk an image down the contracting path and back up the expanding path — toggle the skip connections and watch the predicted mask lose, then recover, its fine boundaries.
]
#pause
*Q.* If you *remove* every skip connection, what happens to the sharpness of the output mask, and why?

// ═══════════════════════════ PART III — Losses for dense prediction ═══════════════════════════
= Losses for dense prediction

== The class-imbalance problem

In dense prediction the *background usually dominates* the image.
#pause
- a small tumor may be *< 1%* of the pixels; the rest is "healthy";
- pixelwise CE is a *mean over pixels* → the majority class drowns the minority.
#pause
#alertbox[A network can score *low CE and high pixel accuracy* by predicting *"all background"* — and still be useless.]

== The Dice coefficient

Measure *overlap* between predicted set $P$ and ground-truth set $G$:
$ "Dice" = (2 abs(P inter G))/(abs(P) + abs(G)) $
#pause
- $= 1$ when $P$ and $G$ coincide exactly; $= 0$ when they are disjoint;
- it *ignores* the vast background — it only cares about the *foreground overlap*.
#pause
#align(center, text(size: 16pt, fill: MUTED)[same idea as the F1 score, phrased for masks.])

== Soft Dice loss

Make Dice differentiable: use *probabilities* $p_i in [0,1]$ instead of hard sets, and *minimize* $1 - "Dice"$:
$ cal(L)_"Dice" = 1 - (2 sum_i p_i g_i + epsilon)/(sum_i p_i + sum_i g_i + epsilon) $
#pause
- $p_i$ — predicted foreground probability at pixel $i$; $g_i in {0, 1}$ — ground truth;
- $epsilon$ — a small constant so an empty mask is *stable*, not $0/0$.
#pause
#result[gradient pushes overlap up directly, regardless of how much background there is]

== Worked Dice example #D

#two(
  fig("/lecture10/figures/dice.svg", w: 100%),
  [
    On a $4 times 4$ grid:
    #v(3pt)
    - $abs(G) = 6$ (true region),
    - $abs(P) = 8$ (predicted),
    - $abs(P inter G) = 5$ (overlap).
    #v(3pt)
    #pause
    $ "Dice" = (2 dot 5)/(8 + 6) = 10/14 ≈ 0.714 $
  ],
  r: (1.35fr, 1fr),
)
#pause
#align(center, text(size: 15pt, fill: MUTED)[green = correct (TP) · teal = missed (FN) · orange = extra (FP).])

== When Dice helps #V

Imbalanced scene (5% foreground), two "models":
#pause
#fig("/lecture10/figures/ce_vs_dice.svg", w: 66%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[pixel accuracy calls both *"great"*; Dice *exposes* the trivial all-background predictor.])

== CE vs Dice — use both

They fail in *opposite* ways, so practitioners *combine* them:
$ cal(L) = cal(L)_"CE" + lambda thin cal(L)_"Dice" $
#pause
#two(
  notebox[*Cross-entropy* — smooth, stable gradients per pixel; but *imbalance-blind*.],
  notebox[*Dice* — directly targets *overlap*; but noisier gradients, unstable on empty masks.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[CE gives a clean training signal; Dice keeps the *foreground* honest.])

== Interactive: segmentation losses #I

#interbox(link-to: IA + "image-segmentation")[
  Paint a prediction over a ground-truth mask and watch pixel CE and the Dice score move — see how a mostly-background prediction can keep CE low while Dice collapses to zero.
]
#pause
*Q.* Why does the trivial "predict all background" solution give *Dice $= 0$* even though its pixel *accuracy* is 95%?

// ═══════════════════════════ PART IV — Instance segmentation ═══════════════════════════
= Instance segmentation

== From semantic to instance

Semantic seg cannot *separate* two touching cats — both are just "cat" pixels.
#pause
#result[instance seg = *detect each object*, then *segment inside its box*]
#pause
#align(center, text(size: 16pt, fill: MUTED)[so we bolt a *mask* onto a detector — one mask per detected object.])

== Recap: Faster R-CNN

The two-stage detector we build on:
$ "backbone" -> "RPN (proposals)" -> "RoI features" -> {"class", "box"} $
#pause
- the *RPN* proposes candidate regions;
- each region's features feed *two* heads: a *class* head and a *box-regression* head.
#pause
#notebox[Faster R-CNN already localizes objects — it just stops at boxes.]

== Mask R-CNN #V

Add a *third, parallel* head that predicts a *mask*:
#pause
#v(2pt)
#maskrcnn
#pause
#align(center, text(size: 15pt, fill: MUTED)[same backbone + RPN · three heads: *class*, *box*, and the new #text(fill: ACC)[*mask*] branch.])

== The mask branch

For each RoI, a *small FCN* predicts one binary mask per class:
#pause
- output is $K$ masks of size $m times m$ (e.g. $28 times 28$), one per class;
- the *classification head chooses which* mask to use — the mask branch stays *class-agnostic* in its loss;
#pause
#notebox[Decoupling *"what class"* from *"which pixels"* is the key trick: masks and classes do not compete.]

== Mask loss #D

Per-pixel *binary* cross-entropy, only on the RoI's chosen-class mask:
$ cal(L)_"mask" = -1/m^2 sum_(u, v) [ thin m_(u v) log hat(m)_(u v) + (1 - m_(u v)) log(1 - hat(m)_(u v)) thin ] $
#pause
- $m_(u v) in {0, 1}$ — true mask; $hat(m)_(u v) in [0, 1]$ — predicted (per-pixel *sigmoid*);
- summed only over the $m times m$ RoI, and only for the *ground-truth class*.
#pause
#result[total loss: $cal(L) = cal(L)_"cls" + cal(L)_"box" + cal(L)_"mask"$]

== RoIAlign — why alignment matters

The old *RoIPool* *rounded* region coordinates to the feature grid.
#pause
- rounding shifts features by up to *half a cell* — invisible for a coarse box;
- for a *per-pixel mask* that misalignment *smears the boundary*.
#pause
#alertbox[*RoIAlign* skips rounding: it samples features at *exact* sub-pixel locations with bilinear interpolation — the single change that made masks sharp.]

// ═══════════════════════════ PART V — Summary ═══════════════════════════
= Summary

== Task → loss, at a glance

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 7pt), align: (left, left, left),
  table.header([*Task*], [*Output*], [*Loss*]),
  [semantic seg], [per-pixel class map], [pixel CE (+ Dice)],
  [instance seg], [per-object mask], [detection loss + mask BCE],
  [panoptic seg], [per-pixel (class, id)], [both, fused],
))
#pause
#v(4pt)
#align(center, text(size: 16pt, fill: MUTED)[the *output shape* dictates the *loss*.])

== Segmentation methods

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 7pt), align: (left, left, left),
  table.header([*Method*], [*Core idea*], [*Gives*]),
  [FCN], [FC → $1 times 1$ conv: dense conversion], [coarse class map],
  [U-Net], [encoder–decoder + skip connections], [sharp class map],
  [Mask R-CNN], [detector + parallel mask branch], [per-object masks],
))
#pause
#notebox[FCN made dense prediction possible; U-Net made it *sharp*; Mask R-CNN made it *per-object*.]

== Where segmentation is used #OPT

#pause
- *medical imaging* — tumor / organ delineation (U-Net's home turf);
- *autonomous driving* — drivable surface, lanes, pedestrians (panoptic);
- *photo editing* — background removal, object selection;
- *remote sensing* — land cover, buildings, roads from satellite images.

== Final mental model — one idea

#align(center, text(size: 22pt)[
  Classification asks *what?* \
  Detection asks *what + where (box)?* \
  #text(fill: ACC)[*Segmentation asks: what at each pixel?*] \
  #v(4pt)
  #text(size: 17pt, fill: MUTED)[coarser → finer: one label · a few boxes · a whole grid of labels]
])

#focus-slide[
  Detection gives boxes; segmentation labels every pixel.
  #v(12pt)
  #set text(size: 22pt)
  From grids of pixels to sequences of tokens — next, *sequence models*.
]
