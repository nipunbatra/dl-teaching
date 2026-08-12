# Lecture 10 · Localization and Object Detection · Teaching Guide

*Public lecture 10; repository file family L09. First-time-instructor companion for an 80-minute class.*

## Synchronization checkpoint

This guide is synchronized to:

- deck source `lecture9/L9-detection.typ`, SHA-256 `664d289a38d1b1ba9ddda7766929429f421b76dbb1d2ab48c2d0008b51946226`;
- `notebooks/L09/01_iou_and_bbox_losses.ipynb`, SHA-256 `30d52418489b0dc059312ee540844226d113817f00721127d03e0bc9b69c4cfc`;
- `notebooks/L09/02_nms_from_scratch.ipynb`, SHA-256 `dc07ad8460628bbf9711d65d6ff6747f5c5ee4ec9376b667b7c2d37b4c55427f`.

If any of those artifacts changes, recheck every number in the held five-box ledger before teaching.

## The student contract

By the end of the core route, a student should be able to:

1. state the box-coordinate and area convention before computing anything;
2. explain how a fixed dense tensor becomes a variable-length set of detections;
3. keep the `TRAIN`, `INFER`, and `EVAL` clocks separate;
4. encode and decode an anchor-relative box and verify the round trip;
5. compute IoU and trace score thresholding plus class-aware NMS by hand;
6. match a ranked detection list one-to-one with ground truth;
7. compute the lecture's one-class rank AP@0.50 and distinguish it from COCO AP;
8. diagnose a detector failure with a symptom → suspect → smallest-test argument.

The lecture spine is:

> A detector trains a fixed set of candidate records, infers a non-redundant scored set, and evaluates that frozen set against truth. The same IoU formula appears on all three clocks, but it compares different objects and makes different decisions.

## Where Lecture 10 sits

### Bridge from L8B

Retrieve one idea before introducing detection: L8B ended with a spatial feature map and then used global average pooling to obtain one image-level label. Global averaging deliberately discards location. Detection reuses the backbone but keeps spatial features long enough to answer both **what** and **where**, repeatedly.

Say this bridge nearly verbatim:

> L8B learned how to reuse a backbone for one label. Today we keep its spatial evidence and replace the head, targets, geometry, inference policy, and metric so the output can be a variable set of labeled boxes.

### Prerequisites to retrieve, not reteach

- From L3/L4: logits, probabilities, cross-entropy, and `-log p`.
- From L7: validation-controlled choices and the rule that the test protocol must remain fixed.
- From L8: spatial feature maps, stride, and why pooling loses precise position.
- From L8B: shared backbones, heads, and feature pyramids as a multi-scale representation.
- From prior ML: precision and recall at a single operating point.
- Mathematical tools: `min`, `max`, logarithms, exponentials, and areas of rectangles.

Use three retrieval questions:

1. “What location information did global average pooling intentionally erase?”
2. “Why can a classifier's fixed $K$-vector not represent an unknown number of objects?”
3. “What does `-log .8` measure?”

If the room cannot answer the third question, write `-log .8 ≈ .223` once and continue. Do not reopen the full cross-entropy lecture.

## Stable teaching grammar

Keep the deck's semantic colors and clock language stable:

- ink: given geometry or evaluation;
- teal: truth or training;
- accent: raw candidate;
- blue: inference control;
- green: survivor, match, or verified result;
- red: suppressed candidate, false positive, or error.

The three clocks are the conceptual guardrail:

| Clock | Available information | IoU compares | Decision |
|---|---|---|---|
| `TRAIN` | predictions and truth | candidate ↔ truth | positive, ignored, or background target; which losses apply |
| `INFER` | predictions only | prediction ↔ prediction | which same-class neighbours are redundant |
| `EVAL` | frozen ranked detections and truth | prediction ↔ truth | TP, FP, FN, precision, recall, and AP |

Whenever anyone says “the IoU threshold,” ask: **which clock?** The training matcher threshold, NMS threshold, and evaluation threshold need not be equal.

## The continuous 55 / 15 / 10 route

The core must close a complete argument by minute 55. Marked `SHOULD` material is an extension, not a prerequisite for the revisit. Marked `OPTIONAL` material is a final widening of the design space.

### Core · minutes 0–55

| Time | Move | Student action | Non-negotiable result |
|---:|---|---|---|
| 0–4 | L8B bridge and held-scene commitment | vote A/B/C before any reveal | preserve opening answers; do not resolve them |
| 4–8 | establish `TRAIN` / `INFER` / `EVAL` | name the comparison made on each clock | same formula, different objects and consequences |
| 8–15 | box contract, held truths, head contract | compute both truth areas | no `+1`; $|G_1|=1600$, $|G_2|=1200$ |
| 15–21 | anchor encode/decode and dense ledger | commit four offsets, then verify inverse | $(.20,-.10,.693,0)$; 1200 candidates; 10,800 scalars |
| 21–27 | held IoU derivation | calculate intersection and union before quotient | $1444/1756\approx.822$ |
| 27–36 | `TRAIN`: assignment, auditable loss, focal imbalance | classify three anchors; compute loss terms | positive/ignore/background; total $\approx.815$; focal ratio ≈978 |
| 36–46 | `INFER`: threshold and exact NMS trace | predict survivor after each step | A suppresses B; E survives; C suppresses D; keep A,E,C |
| 46–52 | `EVAL`: match, precision/recall, rank AP | commit TP/FP before reveal | A=TP, E=FP, C=TP; rank AP@.50=$5/6\approx.833$ |
| 52–55 | revisit, checklist, L11 handoff | defend opening choice using the three clocks | answer B; NMS cannot remove isolated E |

At minute 55, every student must have seen the complete transformation:

`raw A,B,E,C,D → NMS A,E,C → EVAL A=TP,E=FP,C=TP`.

### Should-cover · minutes 55–70

Return to the slides marked `SHOULD` in this order:

| Time | Topic | One claim worth retaining |
|---:|---|---|
| 55–59 | coordinate loss versus overlap; Smooth $L_1$ and GIoU | Smooth $L_1$ controls residual robustness; GIoU supplies a geometric signal for disjoint boxes |
| 59–63 | COCO evaluation protocol | declare IoU thresholds, interpolation, classes, area ranges, and maximum detections |
| 63–66 | two-stage detector | an RPN proposes; a second per-RoI head classifies and refines |
| 66–68 | one-stage detector | dense heads predict class evidence and boxes directly; no learned per-RoI second stage |
| 68–70 | FPN | top-down semantics make high-resolution maps useful for small-object recognition |

Do not teach “two-stage is accurate and slow; one-stage is inaccurate and fast” as a law. The architectural distinction is where proposal and refinement computation happens.

### Optional · minutes 70–80

| Time | Topic | Boundary to state |
|---:|---|---|
| 70–73 | anchor-free heads | parameterization changes; spatial evidence, matching, losses, and evaluation remain |
| 73–77 | Soft-NMS and set prediction | duplicate handling may move from hard deletion to score decay or one-to-one learning |
| 77–79 | primary provenance | identify which claim came from which paper; avoid architecture cataloguing |
| 79–80 | one-sentence revisit and L11 handoff | boxes end here; masks begin in Lecture 11 |

If time slips, protect the minute-52 revisit. A complete causal spine is more valuable than reaching every detector family.

## The held five-box case

Use this case continuously. Do not substitute the old five-car example or a second NMS ledger.

### Truths

- $G_1=[10,10,50,50]$, area $40·40=1600$.
- $G_2=[60,15,90,55]$, area $30·40=1200$.
- Canvas: $100×80$.

### Raw predictions

| ID | Score | Box | Intended role in the worked case |
|---|---:|---|---|
| A | .95 | $[10,10,50,50]$ | exact $G_1$ |
| B | .90 | $[12,12,52,52]$ | duplicate near $G_1$ |
| E | .89 | $[35,5,55,25]$ | isolated false alarm |
| C | .88 | $[60,15,90,55]$ | exact $G_2$ |
| D | .75 | $[63,17,91,56]$ | duplicate near $G_2$ |

Never reveal “duplicate” or “false alarm” while standing on the `INFER` clock; those labels use truth that inference does not have. Phrase them as instructor annotations only.

### Coordinate convention

All boxes are corner form $[x_{min},y_{min},x_{max},y_{max}]$ with continuous boundaries, equivalently half-open pixel regions:

$$
w=x_{max}-x_{min},\qquad h=y_{max}-y_{min},\qquad |B|=wh.
$$

There is no inclusive-pixel `+1`. Also require $x_{max}≥x_{min}$ and $y_{max}≥y_{min}$, clamp intersection side lengths at zero, and define the zero-union behavior.

### Held IoUs

For A and B:

$$
w_I=38,\quad h_I=38,\quad A_I=1444,
$$

$$
A_U=1600+1600-1444=1756,
$$

$$
IoU(A,B)=\frac{1444}{1756}\approx 0.822
$$

For C and D:

$$
A_I=(90-63)(55-17)=27·38=1026,
$$

$$
A_U=1200+1092-1026=1266,
$$

$$
IoU(C,D)=\frac{1026}{1266}\approx 0.810
$$

For E and $G_1$, the intersection is $15×15=225$, E has area $400$, and the union is $1775$:

$$
IoU(E,G_1)=\frac{225}{1775}\approx 0.127
$$

This last number is deliberately unavailable to NMS and decisive during evaluation.

## Commitment-before-reveal choreography

### Commitment 1 · opening scene

Ask:

> Which operation turns five high-scoring candidates into trustworthy detections?

Offer the deck's three choices:

- A: threshold; duplicates disappear automatically;
- B: decode → score → NMS → evaluation matching;
- C: rerun the training loss at inference.

Record votes without commentary. The eventual answer is B, with an important qualification: the pipeline can still return E, so “non-redundant” is not the same as “correct.”

### Commitment 2 · anchor offsets

Given anchor $a=(50,50,40,40)$ and matched target $b=(58,46,80,40)$ in center-size form, have students calculate all four numbers:

$$
t_x=\frac{58-50}{40}=0.20,\qquad
t_y=\frac{46-50}{40}=-0.10
$$

$$
t_w=\log\frac{80}{40}=\log 2\approx 0.693,\qquad
t_h=\log\frac{40}{40}=0
$$

Do not reveal the vector until someone explains why $t_h=0$ means “unchanged” and why exponentiating decoded scale keeps width and height positive.

Decode immediately:

$$
\hat{x}=50+0.20(40)=58,\quad
\hat{y}=50-0.10(40)=46
$$

$$
\hat{w}=40e^{0.693}\approx 80,\quad
\hat{h}=40e^0=40
$$

Treat this round trip as a unit test, not merely algebra.

### Commitment 3 · NMS survivor set

Set $τ_{NMS}=.50$ and use strict suppression `IoU > τ`.

1. A is highest: keep A and suppress B because $.822>.50$.
2. E is next: keep E because it does not overlap the remaining predictions above $.50$.
3. C is next: keep C and suppress D because $.810>.50$.

The returned set is **A, E, C**. Pause on E:

> What information would NMS need to call E wrong?

The answer is ground truth, which is forbidden on the inference clock.

### Commitment 4 · evaluation labels

Move explicitly to `EVAL`. Set $τ_{eval}=.50$ and use `IoU ≥ τ`. Process score order and allow each detection to claim at most one unmatched same-class truth:

| Detection | Best eligible truth | Result |
|---|---|---|
| A (.95) | $G_1$, IoU 1.000 | TP |
| E (.89) | none; best IoU .127 | FP |
| C (.88) | $G_2$, IoU 1.000 | TP |

Only now reveal the labels.

### Revisit

Return to the opening vote and require a causal defense:

- score threshold $.70$: all five remain;
- NMS: A, E, C remain;
- evaluation: A and C are TPs; E is an FP;
- one-class rank AP@.50: $5/6\approx.833$.

The closing line is:

> Thresholds control admission, NMS controls redundancy, and matching measures correctness. None can substitute for the others.

## Exact derivations to put on the board

### Fixed dense output

For the lecture's teaching convention, each anchor emits:

- one objectness logit;
- $K=4$ class logits;
- four box offsets.

That is $1+4+4=9$ scalars. With a $20×20$ grid and three anchors per location:

$$
20·20·3=1200\text{ candidates},
$$

$$
20·20·3·9=10{,}800\text{ output scalars}.
$$

Say “teaching convention.” Modern heads may fold background into class logits, predict box sides, or be anchor-free.

### Toy training assignment

For one truth, suppose three anchors have IoUs $.76$, $.48$, and $.12$. Under the explicitly toy policy $τ_+=.70$ and $τ_-=.30$:

| IoU | Target state | Loss terms |
|---:|---|---|
| .76 | positive | objectness + class + box |
| .48 | ignored | no gradient in this toy policy |
| .12 | background | objectness only |

State the boundary: this is neither NMS nor evaluation matching. Real detectors use architecture-specific or dynamic matching rules.

### Auditable multi-task loss

Use target $t=(.20,-.10,.693,0)$, prediction $\hat{t}=(.25,-.05,.60,.05)$, $p_{obj}=.90$, $p_{cup}=.80$, and $λ=2$.

$$
L_{obj}=-\log .90\approx.105,
$$

$$
L_{class}=-\log .80\approx.223,
$$

$$
L_{box}=|.05|+|.05|+|-.093|+|.05|=.243,
$$

$$
L=.105+.223+2(.243)\approx.815.
$$

Insist on four declarations whenever someone reports “box loss”: parameterization, target assignment, reduction, and weight.

### Focal loss

Use

$$
FL(p_t)=-α_t(1-p_t)^γ\log p_t.
$$

Set $α_t=1$ and $γ=2$ to isolate focusing. For an easy example $p_t=.9$:

$$
FL=.01(-\log .9)\approx.0010536.
$$

For a hard example $p_t=.2$:

$$
FL=.64(-\log .2)\approx1.03004.
$$

The ratio is approximately $977.6$, rounded to **978×**. Focal loss does not delete easy examples; it reduces their aggregate gradient mass.

### Precision, recall, and the lecture AP convention

After NMS, the score order is A, E, C:

| After | Cumulative TP | Cumulative FP | Precision | Recall |
|---|---:|---:|---:|---:|
| A | 1 | 0 | 1.000 | .500 |
| E | 1 | 1 | .500 | .500 |
| C | 2 | 1 | .667 | 1.000 |

The lecture and notebook use a transparent **one-class rank AP@0.50**: average precision at ranks where recall increases.

$$
AP_{\mathrm{rank},0.50}=\frac{1+2/3}{2}=\frac{5}{6}\approx 0.833
$$

Do not call this number “COCO mAP.” COCO evaluation samples 101 recall thresholds, reports results at ten IoU thresholds from .50 to .95, and aggregates over classes while also declaring area range and maximum detections.

## Notebook choreography

Use exactly two notebooks. They are small, deterministic, have no downloads, and use only NumPy and Matplotlib.

### Notebook 1 · boxes, IoU, loss, matching, and AP

[Open the exact Colab](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L09/01_iou_and_bbox_losses.ipynb).

Recommended use:

1. Before the box-conversion cell, ask which **three** normalized entries will equal $.5$.
2. Before the IoU cell, calculate the separate microcase A=`[0,0,10,10]`, B=`[5,5,15,15]` by hand.
3. At the `TRAIN` clock, predict whether Smooth $L_1$ is above or below the $L_1$ sum.
4. At the `EVAL` clock, resume at the held raw list, observe that NMS returned A,E,C, and commit TP/FP labels before execution.
5. Reveal the rank-AP plot only after students calculate $5/6$.
6. End with the anchor encode/decode round trip.

Expected fresh outputs:

- box conversion: `[320,216,320,240]` pixels and `[.50,.45,.50,.50]` normalized;
- separate IoU microcase: intersection 25, union 175, IoU `.143`;
- training residuals `[.05,-.05,.10,-.05]`, $L_1=.25$, Smooth $L_1=.00875$;
- overlap for that same normalized pair: IoU `.691057`;
- classification CE `.223144`; CE plus $2L_1=.723144`;
- NMS-returned evaluation matrix for A,E,C: `[[1,0],[.127,0],[0,1]]`;
- labels: A=TP, E=FP, C=TP;
- precision/recall: `(1,.5)`, `(.5,.5)`, `(.667,1)`;
- rank AP@.50: `.833`;
- anchor offsets `[.20,-.10,.693,0]` and exact decoded target `[58,46,80,40]`.

The notebook includes a regression assertion for a subtle matching bug: with rows `[[.9,0],[.9,.8]]`, the second prediction must be allowed to claim its second-best **unmatched** truth. If an implementation chooses the overall best truth first and only afterward discovers that it was used, it is not implementing the stated rule.

### Notebook 2 · NMS from scratch

[Open the exact Colab](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L09/02_nms_from_scratch.ipynb).

Recommended use:

1. Show the five scores and request the survivor list before running.
2. Print the full trace; do not jump directly to the final list.
3. Read the three-panel state diagram left to right: undecided → A suppresses B → final A,E,C.
4. Run the threshold sweep and stress that its monotonic keep-count is specific to this disjoint-pair scene, not a theorem about arbitrary greedy overlap graphs.
5. Compare class-agnostic and class-aware NMS after adding the overlapping hand box H.
6. End by constructing two distinct nearby same-class objects that a low threshold wrongly merges.

Expected fresh outputs:

- pairwise held IoUs: `IoU(A,B)=.822`, `IoU(C,D)=.810`;
- trace: keep A/suppress B → keep E/suppress none → keep C/suppress D;
- final survivors: `['A','E','C']`;
- threshold sweep: A,E,C at $.3$, $.5$, and $.8$; all five at $.9$;
- class-agnostic output omits H; class-aware output retains H together with A,E,C;
- appended candidate F has $.20×.99=.198$ and fails the $.30$ score threshold;
- the score-plus-NMS pipeline returns A,E,C;
- distinct-object failure pair has IoU `.364`; thresholds `.25` and `.35` keep one, while `.50` and `.70` keep two.

The NMS kernel uses stable descending score order. Equal-score behavior must be declared because other implementations may break ties differently.

### Notebook rhythm

For both notebooks, use:

`predict → calculate → run → inspect → explain`.

Do not live-code from an empty cell. The intellectual work is predicting the state change and explaining the boundary, not watching syntax being typed.

## Diagnostics by clock

### `TRAIN`

| Symptom | Suspect | Smallest discriminating test |
|---|---|---|
| box loss falls; IoU remains near zero | box format or decode mismatch | round-trip one known anchor and draw decoded boxes |
| almost no positive anchors | assignment geometry is too strict or mismatched | histogram best IoU per truth and inspect $τ_+$ |
| objectness collapses to background | easy negatives dominate gradient mass | log positive/negative loss mass; compare focal loss or sampling |
| class learns but boxes do not | box branch receives no/poor targets | count positive box-loss terms and inspect target scales |

### `INFER`

| Symptom | Suspect | Smallest discriminating test |
|---|---|---|
| duplicate boxes remain | wrong decode, class partition, or NMS threshold | print score order and pairwise IoUs for the first cluster |
| nearby real objects merge | $τ_{NMS}$ too low for this geometry | trace the exact suppressing edge; try a higher threshold or Soft-NMS |
| one class deletes another | class-agnostic NMS | run the same boxes independently by class |
| confident isolated errors survive | expecting NMS to test correctness | compare the kept box with truth on the evaluation clock |

### `EVAL`

| Symptom | Suspect | Smallest discriminating test |
|---|---|---|
| high scores but low AP | confidently wrong ranking | inspect the first FP for class, IoU, and duplicate status |
| AP50 high but COCO AP low | boxes are loose | compare AP50 with AP75; histogram matched IoUs |
| recall caps below one | score filtering, NMS, or missing candidates | lower score threshold before NMS and count unmatched truths |
| custom AP disagrees with library | protocol mismatch | print IoU rule, interpolation, class averaging, area, and max-detections settings |

Change one control at a time. In particular, never change the training matcher, score threshold, NMS threshold, and evaluation threshold together; the resulting AP movement would be uninterpretable.

## Common misconceptions and exact repairs

- **“A score threshold removes duplicates.”** It removes weak scores. A and B both survive $.70$.
- **“NMS removes false positives.”** It removes redundant predictions relative to stronger predictions. Isolated E survives.
- **“NMS uses ground truth.”** It runs at inference, where truth is unavailable.
- **“All IoU thresholds are the same.”** Training assignment, NMS, and evaluation live on different clocks and make different decisions.
- **“Boxes count inclusive endpoints, so add one.”** Not in this lecture's continuous/half-open convention.
- **“Assignment and evaluation matching are the same.”** Assignment creates learning targets; evaluation labels a frozen ranked list.
- **“The highest-IoU truth can be checked first, then rejected if already used.”** The stated evaluator chooses the best **unmatched** eligible truth.
- **“Focal loss deletes easy examples.”** It continuously down-weights them.
- **“Higher NMS threshold always means a globally monotone keep count.”** It does in this simple two-cluster scene; greedy interactions do not make that a universal theorem.
- **“One-stage always means faster and less accurate.”** It specifies the computation pattern, not a timeless performance ordering.
- **“AP is one universal formula.”** Always name the IoU cutoff, interpolation, class aggregation, area range, and max detections.
- **“Detection and segmentation belong in one lecture.”** Public Lecture 10 ends at boxes and AP. Public Lecture 11 owns pixel and instance masks.

## If a student asks

### “How can a fixed tensor represent a variable number of objects?”

The network emits a fixed **capacity** of candidate records. Decoding, score filtering, and NMS return a variable-length subset. In the teaching ledger, 1200 candidate records are available even when the image contains two objects.

### “Why offsets instead of absolute pixel coordinates?”

Normalized translations and log-scale changes make the target local to the reference anchor and comparable across positions and object sizes. The same convolutional head can learn “move right by .2 anchor widths” everywhere.

### “Why use log width and height?”

Zero means unchanged scale, positive and negative values represent multiplicative growth and shrinkage symmetrically, and exponentiating at decode keeps sizes positive.

### “Why can E survive NMS?”

NMS only asks whether E is redundant with a stronger same-class prediction. E is isolated. Correctness requires comparison with truth, which happens later during evaluation.

### “Why class-aware NMS?”

Two boxes from different classes can legitimately occupy the same region, such as a hand holding a cup. Same-class partitioning prevents a cup prediction from deleting a hand prediction.

### “Why AP instead of accuracy?”

Detection must judge a ranked, variable-length set while balancing false positives, missed objects, class labels, and localization. A single accuracy denominator does not express those decisions.

### “Are anchors required?”

No. Anchor-free heads may predict centers or distances to four sides. They still require spatial features, target assignment, losses, inference rules, and a declared evaluation protocol.

### “Can NMS be learned or avoided?”

Soft-NMS decays scores instead of deleting neighbours. Set-prediction detectors train a fixed query set with one-to-one matching so duplicate avoidance moves partly into learning. Evaluation still matches scored predictions to truth.

### “What does FPN add?”

It combines top-down semantic information with high-resolution lateral features. High-resolution levels preserve small-object location; top-down features make those locations semantically useful.

## Emergency short route · 35 minutes

Use this only when class time is unexpectedly cut. It preserves the causal argument and drops breadth.

| Time | Keep |
|---:|---|
| 0–3 | L8B bridge and opening vote |
| 3–7 | three clocks and box convention |
| 7–12 | held truth areas and A/B IoU $.822$ |
| 12–16 | fixed candidate capacity plus one-sentence assignment/loss contract |
| 16–24 | exact NMS trace A,B,E,C,D → A,E,C |
| 24–30 | evaluation A=TP,E=FP,C=TP and rank AP $5/6$ |
| 30–33 | one diagnostic per clock |
| 33–35 | opening revisit and L11 handoff |

Cut the focal-loss derivation, Smooth $L_1$/GIoU comparison, detector families, FPN, anchor-free heads, Soft-NMS, and live notebook execution. Show the already-verified notebook outputs only if needed. Never cut the clock distinction, held IoU, NMS trace, evaluation match, or revisit.

## Primary sources and exact companions

Use primary papers for historical or architectural claims:

- [R-CNN — Girshick et al.](https://arxiv.org/abs/1311.2524)
- [Fast R-CNN — Girshick](https://arxiv.org/abs/1504.08083)
- [Faster R-CNN — Ren et al.](https://arxiv.org/abs/1506.01497)
- [YOLO — Redmon et al.](https://arxiv.org/abs/1506.02640)
- [Focal Loss / RetinaNet — Lin et al.](https://arxiv.org/abs/1708.02002)
- [Feature Pyramid Networks — Lin et al.](https://arxiv.org/abs/1612.03144)
- [Generalized IoU — Rezatofighi et al.](https://arxiv.org/abs/1902.09630)
- [Soft-NMS — Bodla et al.](https://arxiv.org/abs/1704.04503)
- [DETR set prediction — Carion et al.](https://arxiv.org/abs/2005.12872)
- [Official COCO evaluation implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)

Exact executable companions:

- [Boxes, IoU, matching, and rank AP Colab](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L09/01_iou_and_bbox_losses.ipynb)
- [NMS from scratch Colab](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L09/02_nms_from_scratch.ipynb)

## L11 handoff · segmentation, not sequences

End with a clean change of output support:

- Lecture 10 detection: a variable set of `(class, box, score)` records;
- Lecture 11 semantic segmentation: class evidence at every pixel;
- Lecture 11 instance segmentation: pixel ownership separated by object instance.

Both reuse the L8B backbone. Lecture 11 replaces four-number rectangles with dense masks and asks how to recover boundaries that downsampling made coarse.

Final line:

> Today asked “which objects, and where?” Lecture 11 asks “which class owns every pixel—and which instance?”
