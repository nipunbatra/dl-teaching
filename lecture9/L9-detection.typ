// Localization and Object Detection — Lecture 9 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture9/L9-detection.typ /tmp/L9.pdf
//   typst compile --root . --input handout=true lecture9/L9-detection.typ /tmp/L9h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [Localization and Object Detection],
  subtitle: [From one label to a variable set of boxes],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let OD = IA + "object-detection"

// ── reusable native (fletcher) detector pipelines ───────────────────
#let pnode(pos, lbl, c: INK, fill: white, r: 9mm) = node(pos, lbl, radius: r,
  stroke: 0.9pt + c, fill: fill)
#let parrow(a, b) = edge(a, b, "-|>", stroke: 0.8pt + MUTED)

// R-CNN: proposals -> warp -> CNN (once per region) -> two heads
#let rcnn-pipe = align(center, diagram(spacing: (13mm, 7mm), {
  pnode((0,0), [image], fill: rgb("#EFEEEB"))
  pnode((1,0), [region\ proposals], c: TEAL)
  pnode((2,0), [crop /\ warp], c: TEAL)
  pnode((3,0), [CNN\ per region], c: ACC, fill: rgb("#FDECD6"), r: 11mm)
  pnode((4,0.7), [class], c: BLUE)
  pnode((4,-0.7), [box reg.], c: BLUE)
  parrow((0,0),(1,0)); parrow((1,0),(2,0)); parrow((2,0),(3,0))
  parrow((3,0),(4,0.7)); parrow((3,0),(4,-0.7))
}))

// Fast R-CNN: one CNN over the image, RoI pooling per proposal, shared heads
#let fast-pipe = align(center, diagram(spacing: (13mm, 7mm), {
  pnode((0,0), [image], fill: rgb("#EFEEEB"))
  pnode((1,0), [CNN\ (whole image)], c: ACC, fill: rgb("#FDECD6"))
  pnode((2,0), [RoI\ pooling], c: TEAL)
  pnode((3,0.7), [class], c: BLUE)
  pnode((3,-0.7), [box reg.], c: BLUE)
  node((2,1.15), text(size: 13pt, fill: MUTED)[proposals], stroke: none)
  parrow((0,0),(1,0)); parrow((1,0),(2,0))
  edge((2,1.15),(2,0), "-|>", stroke: (paint: MUTED, dash: "dashed", thickness: 0.7pt))
  parrow((2,0),(3,0.7)); parrow((2,0),(3,-0.7))
}))

// Faster R-CNN: backbone -> RPN proposals -> RoI head, all sharing features
#let faster-pipe = align(center, diagram(spacing: (13mm, 7mm), {
  pnode((0,0), [image], fill: rgb("#EFEEEB"))
  pnode((1,0), [backbone\ CNN], c: ACC, fill: rgb("#FDECD6"))
  pnode((2,0.75), [RPN], c: GREEN, fill: rgb("#EAF6EC"))
  pnode((2,-0.35), [RoI\ pooling], c: TEAL)
  pnode((3.3,-0.35), [RoI head\ cls + box], c: BLUE, r: 11mm)
  parrow((0,0),(1,0)); parrow((1,0),(2,-0.35)); parrow((1,0),(2,0.75))
  edge((2,0.75),(2,-0.35), "-|>", stroke: (paint: GREEN, dash: "dashed", thickness: 0.8pt))
  parrow((2,-0.35),(3.3,-0.35))
}))

// YOLO: one network, dense grid predictions, then NMS
#let yolo-pipe = align(center, diagram(spacing: (14mm, 7mm), {
  pnode((0,0), [image], fill: rgb("#EFEEEB"))
  pnode((1,0), [single\ CNN], c: ACC, fill: rgb("#FDECD6"))
  pnode((2,0), [dense grid:\ boxes + classes], c: TEAL, r: 11mm)
  pnode((3.2,0), [NMS], c: GREEN, fill: rgb("#EAF6EC"))
  pnode((4.3,0), [detections], c: BLUE)
  parrow((0,0),(1,0)); parrow((1,0),(2,0)); parrow((2,0),(3.2,0)); parrow((3.2,0),(4.3,0))
}))

#title-slide()

// ═══════════════════════════ PART I — Task taxonomy ═══════════════════════════
= What task are we solving?

== Where we are

So far a network mapped an image to *one* answer:
#pause
$ f_theta (x) -> hat(y), quad hat(y) = "softmax"(W h) in Delta^(K-1) $
#pause
#notebox[Classification asks *what is in the image?* — a single label. But most real images contain *several* things, each *somewhere*. This lecture is about *what* and *where*, for a variable number of objects.]

== The spine of this lecture

#align(center, text(size: 22pt, fill: INK)[
  Classification predicts *one label*. \
  Detection predicts a *variable set of objects*, \
  each with a *class* and a *box*.
])
#pause
#v(6pt)
#align(center, text(size: 16pt, fill: MUTED)[the output is no longer a fixed-size vector — it is a *set* whose size depends on the image.])

== Classification: $x -> y$

One image in, one categorical label out.
#pause
$ f_theta (x) = hat(y) in {1, dots, K} $
#pause
- output shape is *fixed*: a $K$-way distribution;
- no notion of *location* — the object could be anywhere;
- the whole image is assumed to be *about one thing*.

== Localization: $x -> (y, b)$

Now also report *where* the (single, dominant) object is:
#pause
$ f_theta (x) = (hat(y), hat(b)), quad hat(b) in RR^4 $
#pause
- $hat(y)$ — the class, as before;
- $hat(b)$ — *one* bounding box (four numbers);
- still *exactly one* object per image — the output is fixed-size.
#pause
#result[classification + a box regression head]

== Detection: $x -> {(y_i, b_i, s_i)}_(i=1)^N$

Report *every* object: a *variable-length set*.
#pause
$ f_theta (x) = { (hat(y)_i, hat(b)_i, s_i) }_(i=1)^N $
#pause
- $hat(y)_i$ — class of object $i$; #h(0.5em) $hat(b)_i$ — its box; #h(0.5em) $s_i$ — a confidence score;
- $N$ *varies per image* — 0 objects, 1, or dozens;
- this variable $N$ is what makes detection *fundamentally harder*.

== One more task — but that is next time

#notebox[
*Segmentation* pushes further: a class label for *every pixel*, not just a box.
That is *Lecture 10*. Today we stop at boxes.
]
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, center, left),
  table.header([*Task*], [*Output*], [*Granularity*]),
  [classification], [$hat(y)$], [whole image],
  [localization], [$(hat(y), hat(b))$], [one object],
  [detection], [${(hat(y)_i, hat(b)_i, s_i)}$], [many objects],
  [segmentation], [pixel labels], [every pixel #text(fill: MUTED)[(L10)]],
))

== The same image, three tasks #V

#fig("/lecture9/figures/task_taxonomy.svg", w: 92%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[one label · one box · a *set* of boxes — the output grows richer left to right.])

// ═══════════════════════════ PART II — Bounding boxes & localization ═══════════════════════════
= Bounding boxes and localization

== What is a bounding box? #V

A box is *four numbers* — two equivalent parameterizations:
#pause
#fig("/lecture9/figures/box_params.svg", w: 84%)
#pause
$ b = (x_min, y_min, x_max, y_max) quad "or" quad b = (x_c, y_c, w, h) $

== Normalize the coordinates

Raw pixel coordinates depend on image size. *Divide out* width $W$ and height $H$:
#pause
$ tilde(x) = x/W, quad tilde(y) = y/H, quad tilde(w) = w/W, quad tilde(h) = h/H $
#pause
#notebox[Normalized coordinates live in $[0,1]$ regardless of resolution — so the same network handles images of *any* size, and the regression targets are *well-scaled*.]

== A localization network

Reuse a classification backbone; add a *second head*.
#pause
$ h = f_theta (x) quad quad "(shared backbone features)" $
#pause
$ hat(p) = "softmax"(W_c thin h) quad quad "(classification head)" $
$ hat(b) = W_b thin h quad quad "(box regression head)" $
#pause
#result[one backbone, *two heads* — a class distribution and four box numbers]

== Training with two objectives

The network must be *right about the class* and *tight around the object*:
#pause
$ cal(L) = cal(L)_"cls" + lambda thin cal(L)_"box" $
#pause
- $cal(L)_"cls"$ — cross-entropy on $hat(p)$ (Lecture 1);
- $cal(L)_"box"$ — a *regression* loss on $hat(b)$;
- $lambda$ — a weight balancing the two (they have different units/scales).
#pause
#notebox[This *multi-task loss* — a shared trunk with several weighted heads — recurs throughout detection.]

== The box regression loss

Penalize the difference between predicted and true box coordinates:
#pause
$ cal(L)_"box" = norm(hat(b) - b)_1 = sum_(j=1)^4 |hat(b)_j - b_j| quad quad (L_1) $
#pause
or the squared version:
$ cal(L)_"box" = norm(hat(b) - b)_2^2 = sum_(j=1)^4 (hat(b)_j - b_j)^2 quad quad (L_2) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[$L_2$ punishes large errors hard (sensitive to outliers); $L_1$ is steadier but kinks at 0.])

== Smooth $L_1$: the best of both #D

Quadratic near 0 (stable gradient), linear far out (robust to outliers):
#pause
$ "smooth"_(L_1)(r) = cases(
  1/2 r^2 & "if" |r| < 1,
  |r| - 1/2 & "otherwise"
) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[$r = hat(b)_j - b_j$ is the per-coordinate residual — the pieces meet smoothly at $|r| = 1$.])

== Smooth $L_1$ vs $L_1$ vs $L_2$ #V

#fig("/lecture9/figures/smooth_l1.svg", w: 62%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[near 0 it curves like $L_2$ (no kink) · far out it grows like $L_1$ (no outlier blow-up).])

== Worked example: localization loss #D

True box $b = (0.5, 0.5, 0.4, 0.2)$, prediction $hat(b) = (0.6, 0.45, 0.5, 0.3)$ (normalized).
#pause
Per-coordinate absolute residuals:
$ |0.6 - 0.5| = 0.10, quad |0.45 - 0.5| = 0.05 $
$ |0.5 - 0.4| = 0.10, quad |0.3 - 0.2| = 0.10 $
#pause
$ cal(L)_"box" = 0.10 + 0.05 + 0.10 + 0.10 = 0.35 $
#pause
#result[$L_1 = 0.35$]

== Interactive: box-regression loss #I

#interbox[
  #text(fill: MUTED)[(to build)] — drag the predicted box around a fixed ground-truth box and watch $L_1$, $L_2$, and smooth-$L_1$ update coordinate-by-coordinate, and see where each loss's gradient is large or flat.
]
#pause
*Q.* Coordinate losses treat all four numbers independently. What about the box do they *not* directly measure?

// ═══════════════════════════ PART III — IoU & metrics ═══════════════════════════
= IoU and detection metrics

== Intersection over Union #V

The standard measure of *box overlap* — area of overlap ÷ area of union:
#pause
$ "IoU"(A, B) = (|A inter B|)/(|A union B|) in [0, 1] $
#pause
#fig("/lecture9/figures/iou.svg", w: 46%)

== Worked example: IoU #D

Two boxes, each of area $100$; their overlap has area $25$.
#pause
Union by inclusion–exclusion:
$ |A union B| = |A| + |B| - |A inter B| = 100 + 100 - 25 = 175 $
#pause
$ "IoU" = (|A inter B|)/(|A union B|) = 25/175 approx 0.143 $
#pause
#result[IoU $= 25\/175 approx 0.143$]

== Why coordinate loss is not enough

Smooth-$L_1$ drives the four numbers together — but that is *not the same* as overlap.
#pause
- two boxes can have *equal* coordinate error yet *very different* IoU;
- IoU is *scale-invariant*; a fixed pixel error matters more for a *small* box;
- the metric we *report* (IoU) differs from the loss we *optimize* (coordinates).
#pause
#result[so: optimize an *IoU-based* loss directly]

== The trouble with plain IoU loss

Use $cal(L)_"IoU" = 1 - "IoU"$? It works *when boxes overlap* — but:
#pause
#alertbox[If the boxes *do not overlap*, $"IoU" = 0$ for *all* non-overlapping configurations. The loss is flat — *zero gradient* — so it cannot pull a far-off box toward the target.]

== GIoU: a gradient even with no overlap #OPT

*Generalized IoU* adds a term using the smallest box $C$ enclosing both:
#pause
$ "GIoU" = "IoU" - (|C \\ (A union B)|)/(|C|) in [-1, 1] $
#pause
- when $A, B$ overlap, GIoU $approx$ IoU;
- when they are *apart*, the enclosing-box term *still varies with distance* — a usable gradient.
#pause
#align(center, text(size: 16pt, fill: MUTED)[later variants (DIoU, CIoU) refine this further — all share the same fix: keep a signal when IoU is 0.])

== Precision and recall

To *score a detector*, first count matches (a prediction is a *hit* if IoU with a true box $> tau$):
#pause
$ "precision" = "TP"/("TP" + "FP") quad quad "recall" = "TP"/("TP" + "FN") $
#pause
- *precision* — of the boxes I predicted, how many were *right*?
- *recall* — of the objects that exist, how many did I *find*?
#pause
#notebox[Lowering the score threshold finds more objects (↑ recall) but admits more false alarms (↓ precision). There is a *tradeoff curve*.]

== Average Precision (AP) #V

Sweep the score threshold; plot precision vs recall; AP is the *area under it*:
#pause
#fig("/lecture9/figures/pr_curve.svg", w: 52%)
#pause
$ "AP" = integral_0^1 "precision"(r) thin d r quad in [0, 1] $

== mean Average Precision (mAP)

AP is *per class* at a *given* IoU threshold. Average to summarize a detector:
#pause
$ "mAP" = 1/K sum_(k=1)^K "AP"_k $
#pause
- average over all *classes* $k$ (car, person, dog, …);
- modern benchmarks *also* average over IoU thresholds $tau in {0.5, 0.55, dots, 0.95}$;
- reported as e.g. *mAP\@0.5* or *mAP\@[.5:.95]*.
#pause
#result[mAP — one number summarizing detection quality]

== Interactive: IoU and mAP #I

#interbox(link-to: OD)[
  Drag two boxes to see IoU update live; then move a score threshold along a PR curve and watch precision, recall, and the AP area respond.
]
#pause
*Q.* A detector reports every object with *very low* confidence. What happens to its precision, and to its recall?

// ═══════════════════════════ PART IV — Detection pipeline ═══════════════════════════
= Building a detector

== Why detection is hard

Unlike classification, the output is a *set of unknown size*:
#pause
- *how many* objects? unknown, and it varies per image;
- *where / what scale*? anywhere, any size;
- *class imbalance* — the vast majority of image regions are *background*;
- *duplicates* — one object easily triggers *many* overlapping predictions.
#pause
#notebox[Every design choice below is a response to one of these four problems.]

== The core idea: dense candidates

Rather than guess $N$, *predict at many fixed locations and scales*, then filter.
#pause
- tile the image with a *dense grid* of candidate regions;
- at each, ask: *is there an object here? what class? what exact box?*
- most candidates say "background"; a few fire — then we *clean up duplicates*.
#pause
#result[turn a variable-size set problem into a *fixed, dense* prediction problem]

== Anchor boxes #V

Predefined *reference boxes* of assorted scales/aspect-ratios at every location:
#pause
$ a = (x_a, y_a, w_a, h_a) $
#pause
#fig("/lecture9/figures/anchors.svg", w: 44%)
#align(center, text(size: 15pt, fill: MUTED)[the network *adjusts* an anchor rather than inventing a box from scratch.])

== Predict box offsets, not raw coordinates

The network outputs *corrections* to each anchor — easier to learn and well-scaled:
#pause
$ t_x = (x - x_a)/w_a, quad quad t_y = (y - y_a)/h_a $
$ t_w = log(w/w_a), quad quad t_h = log(h/h_a) $
#pause
#notebox[Offsets are *relative* to the anchor: centers in anchor-widths, sizes in log-scale. A prediction of *all zeros* reproduces the anchor exactly.]

== What the detection head outputs

Per anchor (at each grid location), the head emits:
#pause
- *objectness* $hat(o) in [0,1]$ — is there an object here at all?
- *class logits* — a $K$-way distribution *if* there is;
- *box offsets* $(t_x, t_y, t_w, t_h)$ — how to reshape the anchor.
#pause
#align(center, text(size: 16pt, fill: MUTED)[for $A$ anchors and $K$ classes: $A times (1 + K + 4)$ numbers per location.])

== The detection loss

Sum the three jobs, over *positive* (object) and relevant anchors:
#pause
$ cal(L) = cal(L)_"obj" + cal(L)_"cls" + lambda thin cal(L)_"box" $
#pause
- $cal(L)_"obj"$ — objectness (object vs background), over *all* anchors;
- $cal(L)_"cls"$ — class cross-entropy, on *positive* anchors only;
- $cal(L)_"box"$ — smooth-$L_1$ on offsets, on *positive* anchors only.

== Which anchors are positive? #D

Assign each anchor a label by its IoU with the *ground-truth* boxes:
#pause
$ "IoU"(a, b_"gt") > tau_+ ==> "positive (an object)" $
$ "IoU"(a, b_"gt") < tau_- ==> "negative (background)" $
#pause
- anchors in between ($tau_- <= "IoU" <= tau_+$) are *ignored* (ambiguous);
- a positive anchor's box/class targets come from the matched $b_"gt"$;
- typical thresholds: $tau_+ approx 0.7$, $tau_- approx 0.3$.

== The class-imbalance problem

With thousands of anchors per image, *almost all* are background.
#pause
#alertbox[A sea of easy negatives *drowns out* the gradient from the few real objects. Plain cross-entropy, summed over all anchors, is dominated by easy background.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[foreground : background can be *1 : 1000s* — the loss needs to *ignore the easy majority*.])

== Focal loss #D

Down-weight *easy, well-classified* examples by a factor $(1 - p_t)^gamma$:
#pause
$ "FL"(p_t) = -(1 - p_t)^gamma log p_t $
#pause
where $p_t$ is the predicted probability of the *true* class.
#pause
- $gamma = 0$ recovers *ordinary cross-entropy*;
- large $p_t$ (easy) ⇒ $(1-p_t)^gamma approx 0$ ⇒ *tiny* loss;
- small $p_t$ (hard) ⇒ factor $approx 1$ ⇒ loss *kept*.

== Focal loss down-weights easy examples #V

#fig("/lecture9/figures/focal_loss.svg", w: 58%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[higher $gamma$ ⇒ easy examples ($p_t -> 1$) contribute almost nothing — built for *dense one-stage* detectors.])

== Interactive: anchor assignment #I

#interbox[
  #text(fill: MUTED)[(to build)] — drop a ground-truth box on an anchor grid and watch anchors light up *positive / ignored / negative* as you sweep $tau_+$ and $tau_-$.
]
#pause
*Q.* If you set $tau_+$ very high (say $0.9$), what happens to the number of positive anchors — and to training?

// ═══════════════════════════ PART V — R-CNN family ═══════════════════════════
= Two-stage detectors: the R-CNN family

== R-CNN: propose, then classify

Stage 1 proposes regions (classic *selective search*); stage 2 runs a CNN on *each*.
#pause
#rcnn-pipe
#pause
#alertbox[A CNN forward pass *per region* (thousands of crops) — *slow*, and the multi-stage training (proposals, CNN, SVMs, regressors) is cumbersome.]

== Fast R-CNN: share the convolution

Run the CNN *once* over the whole image; pool features *per proposal* (*RoI pooling*).
#pause
#fast-pipe
#pause
#notebox[The expensive convolutions are computed *once* and *shared* across all proposals — a large speedup, and the heads train *end-to-end*.]

== Faster R-CNN: learn the proposals too

Replace hand-crafted proposals with a *Region Proposal Network* (RPN) on the same features.
#pause
#faster-pipe
#pause
#align(center, text(size: 16pt, fill: MUTED)[the RPN and the detection head *share the backbone* — proposals become *neural*, fast, and trainable.])

== The two losses of Faster R-CNN

Both stages carry an *objectness/class* term and a *box* term:
#pause
$ cal(L)_"RPN" = cal(L)_"obj" + lambda thin cal(L)_"box" quad quad "(is it an object? rough box)" $
#pause
$ cal(L)_"RoI" = cal(L)_"cls" + lambda thin cal(L)_"box" quad quad "(which class? refined box)" $
#pause
#result[trained jointly — one network, two stages, four loss terms]

== The two-stage intuition

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 8pt), align: (left, left, left),
  table.header([*Stage*], [*Question*], [*Output*]),
  [1 — RPN], [where *might* objects be?], [class-agnostic proposals + rough boxes],
  [2 — RoI head], [*what* is it, exactly *where*?], [class + refined box per proposal],
))
#pause
#notebox[Stage 1 casts a wide net (high recall, coarse); stage 2 is precise (class + tight box). *Historically* this two-stage split gave strong localization.]

// ═══════════════════════════ PART VI — YOLO / one-stage + NMS ═══════════════════════════
= One-stage detectors and NMS

== YOLO: detection in a single pass

*You Only Look Once* — one network maps the image *directly* to boxes + classes.
#pause
#yolo-pipe
#pause
#notebox[No separate proposal stage: a *single* forward pass regresses all boxes and classes at once, then NMS removes duplicates. Simple and *fast*.]

== The grid intuition #V

Divide the image into an $S times S$ grid; each cell predicts boxes + classes:
#pause
#fig("/lecture9/figures/yolo_grid.svg", w: 44%)
#align(center, text(size: 15pt, fill: MUTED)[the cell containing an object's *center* is responsible for predicting it.])

== The one-stage loss

The familiar three terms, over the *dense grid* (no proposal stage):
#pause
$ cal(L) = cal(L)_"box" + cal(L)_"obj" + cal(L)_"cls" $
#pause
- $cal(L)_"box"$ — box offset regression (smooth-$L_1$ / IoU-based);
- $cal(L)_"obj"$ — objectness / confidence per cell-anchor;
- $cal(L)_"cls"$ — class prediction for cells that contain an object.
#pause
#align(center, text(size: 16pt, fill: MUTED)[same ingredients as Faster R-CNN — just predicted *all at once*, no region-proposal stage.])

== Speed vs accuracy — carefully

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 7pt), align: (left, left, left),
  table.header([], [*one-stage (YOLO)*], [*two-stage (Faster R-CNN)*]),
  [passes], [single], [propose, then refine],
  [speed], [faster, simpler], [heavier],
  [localization], [historically weaker], [historically stronger],
))
#pause
#notebox[These are *historical* tendencies. Modern one-stage detectors (with focal loss, better assignment, stronger backbones) *close much of the gap* — the real ranking depends on implementation, data, and scale.]

== The duplicate problem

The dense grid fires *many* overlapping boxes for one object:
#pause
#fig("/lecture9/figures/nms.svg", w: 78%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[we want *one* box per object — a post-processing step must *suppress the duplicates*.])

== Non-Maximum Suppression (NMS) #D

A greedy filter run *per class*, on the raw boxes:
#pause
+ *sort* all boxes by confidence score;
+ *keep* the highest-scoring remaining box;
+ *suppress* every other box with $"IoU" > tau$ against it;
+ *repeat* on the boxes that survive.
#pause
#notebox[Each kept box "claims" its neighbourhood; overlapping lower-score boxes are discarded. Typical $tau approx 0.5$. (Soft-NMS *decays* scores instead of hard-removing.)]

== Interactive: NMS #I

#interbox(link-to: OD)[
  Start from a pile of overlapping candidate boxes with scores, then step NMS — sort, keep the top box, suppress its high-IoU neighbours — and sweep $tau$ to see over- vs under-suppression.
]
#pause
*Q.* Two *distinct* objects of the same class stand very close together. What can NMS with a low $tau$ do wrong?

// ═══════════════════════════ PART VII — Summary ═══════════════════════════
= Summary

== Tasks and their losses

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left),
  table.header([*Task*], [*Loss*]),
  [classification], [cross-entropy],
  [localization], [cross-entropy $+ lambda thin$ box],
  [detection], [objectness $+$ class $+ lambda thin$ box],
))
#pause
#notebox[Every step adds a term, never removes one: detection is classification *plus* "is anything here?" *plus* "where exactly?"]

== The detector families

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 7pt), align: (left, left, left),
  table.header([*Detector*], [*Key idea*], [*Stages*]),
  [R-CNN], [region proposals → CNN per crop], [two (slow)],
  [Fast R-CNN], [shared conv + RoI pooling], [two],
  [Faster R-CNN], [neural proposals via RPN], [two],
  [YOLO], [dense prediction in one pass], [one],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[all share: anchors/cells · objectness + class + box · NMS to deduplicate.])

== Final mental model — one idea

#align(center, text(size: 22pt)[
  Classification answers *what?* \
  Detection answers *what + where*, \
  for a *variable set* of objects. \
  #v(4pt)
  #text(size: 18pt, fill: MUTED)[boxes + scores over dense candidates, cleaned up by NMS]
])

#focus-slide[
  Classification = *what*. Detection = *what + where*, as a set of boxes.
  #v(12pt)
  #set text(size: 22pt)
  Next (*Lecture 10*): push to *every pixel* — *semantic and instance segmentation*, where the answer is a label map, not a box.
]
