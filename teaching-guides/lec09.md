# Lecture 10 · Localization and Object Detection · Teaching Guide

*Public lecture 10; repository file family L09. First-time-instructor companion for an 80-minute class.*

## Synchronization checkpoint

This guide is synchronized to:

- deck source `lecture9/L9-detection.typ`, SHA-256 `377f688adbe12f274960e57190260ff9bf3c6fac3ab8e00195c986e53606886b`;
- `notebooks/L09/01_iou_and_bbox_losses.ipynb`, SHA-256 `ea2ef8b54a26094589d3bd41db9ed4e9d273afaae548a965eb33bacba048dd67`;
- `notebooks/L09/02_nms_from_scratch.ipynb`, SHA-256 `5ac6b9648c1eab8ea731715575f5c976e32b0e4bc5200f3dfa1cc9d76d8b1c1c`;
- evidence manifest `shared/vision-evidence/oxford-iiit-pet/l9/evidence.json`, SHA-256 `b539bc41f383d52f95ed2abeeaf764ccf325a1a03b000be7d377b075cac2836a`;
- evidence builder `shared/vision-evidence/oxford-iiit-pet/l9/build_detection_evidence.py`, SHA-256 `1e6d3312df89424974b1fab8ccdbc5e11963dd4b6d6c802d64d33b09380fb3e1`.

If any of those artifacts changes, recheck every number in the held five-box ledger, the two XML conversions, the grouped NMS trace, and both notebook execution sequences before teaching.

## The student contract

By the end of the core route, a student should be able to:

1. state the box-coordinate and area convention before computing anything;
2. explain how a fixed dense tensor becomes a variable-length set of detections;
3. keep the `TRAIN`, `INFER`, and `EVAL` clocks separate;
4. encode and decode an anchor-relative box and verify the round trip;
5. compute IoU and trace score thresholding plus `(image_id, class)`-grouped NMS by hand;
6. match a ranked detection list one-to-one with ground truth within each image, then aggregate the batch ranking;
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

## Real-evidence contract

The held arithmetic is now rendered over a fixed **two-image Oxford-IIIT Pet teaching mini-batch**. Use the following nouns exactly:

| Evidence type | What belongs here |
|---|---|
| **OBSERVED** | original pixels in `Abyssinian_1.jpg` and `Bengal_10.jpg` |
| **GT** | each sample's official XML tight-head ROI, converted from VOC 1-based inclusive to course 0-based half-open `xyxy` |
| **CONSTRUCTED** | display crops, candidates A–E, their scores, shared `head` class, and the two affine maps |
| **COMPUTED** | IoU, per-image/class NMS, one-to-one matching, PR/AP, CSV ledgers, and overlays |
| **MODEL OUTPUT** | none; no detector was run |
| **MEASURED** | none; `5/6` is a computed teaching-case AP, not a benchmark result |

Say the boundary aloud before the opening vote:

> These are two real photographs and two official labels. I constructed five scored candidates over them so every IoU, suppression, match, and AP decision remains exactly auditable. They are not detector predictions.

The production invariant is new but simple: NMS and evaluation matching may compare boxes only when `image_id` and class agree. After each image is processed independently, surviving detections may be globally score-sorted to compute mini-batch precision and recall.

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
| `TRAIN` | predictions and truth | candidate ↔ truth within one image/class | positive, ignored, or background target; which losses apply |
| `INFER` | predictions only | prediction ↔ prediction within one image/class | which neighbours are redundant |
| `EVAL` | frozen ranked detections and truth | prediction ↔ truth within one image/class | TP, FP, FN, precision, recall, and AP |

Whenever anyone says “the IoU threshold,” ask: **which clock?** The training matcher threshold, NMS threshold, and evaluation threshold need not be equal.

## The continuous 55 / 15 / 10 route

The core must close a complete argument by minute 55. Marked `SHOULD` material is an extension, not a prerequisite for the revisit. Marked `OPTIONAL` material is a final widening of the design space.

### Core · minutes 0–55

| Time | Move | Student action | Non-negotiable result |
|---:|---|---|---|
| 0–4 | L8B bridge and real two-image commitment | identify observed/GT/constructed, then vote A/B/C | preserve opening answers; do not resolve them |
| 4–8 | establish `TRAIN` / `INFER` / `EVAL` | name the comparison made on each clock | same formula, different objects and consequences |
| 8–15 | box contract, XML conversion, canonical truths, head contract | convert one VOC ROI; compute both canonical truth areas | no `+1`; $|G_1|=1600$, $|G_2|=1200$ |
| 15–21 | anchor encode/decode and dense ledger | commit four offsets, then verify inverse | $(.20,-.10,.693,0)$; 1200 candidates; 10,800 scalars |
| 21–27 | held IoU derivation | calculate intersection and union before quotient | $1444/1756\approx.822$ |
| 27–36 | `TRAIN`: assignment, auditable loss, focal imbalance | classify three anchors; compute loss terms | positive/ignore/background; total $\approx.815$; focal ratio ≈978 |
| 36–46 | `INFER`: threshold and exact grouped NMS trace | predict survivor inside each image/class | I1: A suppresses B, E survives; I2: C suppresses D; keep A,E,C |
| 46–52 | `EVAL`: image-aware match, global ranking, rank AP | commit TP/FP before reveal | A=TP, E=FP, C=TP; rank AP@.50=$5/6\approx.833$ |
| 52–55 | revisit, checklist, L11 handoff | defend opening choice using the three clocks | answer B; NMS cannot remove isolated E |

At minute 55, every student must have seen the complete transformation:

`I1 raw A,B,E → A,E; I2 raw C,D → C; global rank A,E,C → EVAL A=TP,E=FP,C=TP`.

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

Use this case continuously. The canonical arithmetic and the two photographic renderings are **one case**, not competing ledgers. Do not substitute the old five-car example.

### Official truths and affine maps

| Image | OBSERVED sample | Official VOC XML | Course source GT | Canonical GT | Positive affine map |
|---|---|---|---|---|---|
| I1 | `Abyssinian_1`, $600×400$ | $(333,72)$–$(425,158)$ | $[332,71,425,158)$ | $G_1=[10,10,50,50]$ | $x'=332+(x-10)93/40$; $y'=71+(y-10)87/40$ |
| I2 | `Bengal_10`, $500×375$ | $(315,60)$–$(446,219)$ | $[314,59,446,219)$ | $G_2=[60,15,90,55]$ | $x'=314+(x-60)132/30$; $y'=59+(y-15)160/40$ |

Each image has its own $100×80$ canonical teaching chart. The charts are not a shared image plane. Canonical areas remain $|G_1|=40·40=1600$ and $|G_2|=30·40=1200$. Independent positive scaling of x and y preserves IoU exactly.

### Constructed candidate records

| ID | Image | Score | Class | Canonical box | Source-pixel box | Intended role in the worked case |
|---|---|---:|---|---|---|---|
| A | I1 | .95 | head | $[10,10,50,50]$ | $[332,71,425,158]$ | exact $G_1$ |
| B | I1 | .90 | head | $[12,12,52,52]$ | $[336.65,75.35,429.65,162.35]$ | duplicate candidate near $G_1$ |
| E | I1 | .89 | head | $[35,5,55,25]$ | $[390.125,60.125,436.625,103.625]$ | isolated candidate |
| C | I2 | .88 | head | $[60,15,90,55]$ | $[314,59,446,219]$ | exact $G_2$ |
| D | I2 | .75 | head | $[63,17,91,56]$ | $[327.2,67,450.4,223]$ | duplicate candidate near $G_2$ |

Never reveal “duplicate” or “false alarm” while standing on the `INFER` clock; those labels use truth that inference does not have. Phrase them as instructor annotations only.

### Coordinate convention

All boxes are corner form $[x_{min},y_{min},x_{max},y_{max}]$ with continuous boundaries, equivalently half-open pixel regions:

$$
w=x_{max}-x_{min},\qquad h=y_{max}-y_{min},\qquad |B|=wh.
$$

There is no inclusive-pixel `+1`. Also require $x_{max}≥x_{min}$ and $y_{max}≥y_{min}$, clamp intersection side lengths at zero, and define the zero-union behavior.

Before computing IoU, check identity: a pair is eligible only if its `image_id` and class agree. `IoU(A,C)` is not zero; it is **undefined for this pipeline decision** because A and C belong to different photographs.

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

This last number is deliberately unavailable to NMS as a correctness test and decisive during evaluation. NMS may compute `IoU(A,E)` because both are I1/head predictions; it may not compare either with C or D from I2.

## Commitment-before-reveal choreography

### Commitment 1 · opening mini-batch

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

1. In I1/head, A is highest: keep A and suppress B because $.822>.50$.
2. Still in I1/head, E is next: keep E because it does not overlap A above $.50$.
3. Independently in I2/head, keep C and suppress D because $.810>.50$.
4. Concatenate survivors and sort globally by score: A, E, C.

The returned set is **A, E, C**. Pause on E:

> What information would NMS need to call E wrong?

The answer is ground truth, which is forbidden on the inference clock. Keep every GT rectangle hidden on the photographic NMS trace.

### Commitment 4 · evaluation labels

Move explicitly to `EVAL`. Set $τ_{eval}=.50$ and use `IoU ≥ τ`. Process global score order, but allow each detection to claim only an unmatched truth with the same `image_id` and class:

| Detection | Image | Best eligible truth | Result |
|---|---|---|---|
| A (.95) | I1 | $G_1$, IoU 1.000 | TP |
| E (.89) | I1 | none; IoU .127 with $G_1$ | FP |
| C (.88) | I2 | $G_2$, IoU 1.000 | TP |

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

Use target $t=(.20,-.10,.693,0)$, prediction $\hat{t}=(.25,-.05,.60,.05)$, $p_{obj}=.90$, $p_{head}=.80$, and $λ=2$.

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

Use exactly two notebooks. They are deterministic and dependency-light. Their asset loader first searches the local repository; in Colab it downloads only the named GitHub raw assets and rejects any SHA-256 mismatch. Never bypass a failed hash assertion.

### Notebook 1 · boxes, IoU, loss, matching, and AP

[Open the exact Colab](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L09/01_iou_and_bbox_losses.ipynb).

Recommended use:

1. Run the evidence-audit cell: verify both JPEG/XML hashes, convert both official ROIs, and name the evidence taxonomy before showing A–E.
2. Before the box-conversion cell, ask which **three** normalized entries will equal $.5$.
3. Before the IoU cell, calculate the separate microcase A=`[0,0,10,10]`, B=`[5,5,15,15]` by hand.
4. Before the affine audit, predict whether independently scaling x and y preserves IoU; then verify `.822323`, `.810427`, and `.126761` in both coordinate systems.
5. At the `TRAIN` clock, predict whether Smooth $L_1$ is above or below the $L_1$ sum.
6. At the `EVAL` clock, commit A/E/C labels before revealing the image-aware matrix and real evaluation plate.
7. Reveal the rank-AP plot only after students calculate $5/6$.
8. End with the anchor encode/decode round trip.

Expected fresh outputs:

- evidence audit: I1 GT `[332,71,425,158]`, I2 GT `[314,59,446,219]`, source sizes `[600,400]` and `[500,375]`;
- affine invariance: A/B `.822323`, C/D `.810427`, A/E `.126761` in canonical and source coordinates;
- box conversion: `[320,216,320,240]` pixels and `[.50,.45,.50,.50]` normalized;
- separate IoU microcase: intersection 25, union 175, IoU `.143`;
- training residuals `[.05,-.05,.10,-.05]`, $L_1=.25$, Smooth $L_1=.00875$;
- overlap for that same normalized pair: IoU `.691057`;
- classification CE `.223144`; CE plus $2L_1=.723144`;
- NMS-returned evaluation matrix for A,E,C: `[[1,nan],[.127,nan],[nan,1]]`, where `nan` means a forbidden cross-image comparison;
- labels: A=TP, E=FP, C=TP;
- precision/recall: `(1,.5)`, `(.5,.5)`, `(.667,1)`;
- rank AP@.50: `.833`;
- anchor offsets `[.20,-.10,.693,0]` and exact decoded target `[58,46,80,40]`.

The executed notebook has 13 code checkpoints with counts `1–13` and zero error outputs. It also includes a regression assertion for a subtle matching bug: with rows `[[.9,0],[.9,.8]]`, the second prediction must be allowed to claim its second-best **unmatched** truth. If an implementation chooses the overall best truth first and only afterward discovers that it was used, it is not implementing the stated rule.

### Notebook 2 · NMS from scratch

[Open the exact Colab](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L09/02_nms_from_scratch.ipynb).

Recommended use:

1. Verify both official images/ROIs, then show the constructed-candidate plate and request the survivor list.
2. Ask why the kernel must partition by `(image_id, class)` before any IoU comparison.
3. Print the full grouped trace; do not jump directly to the final list.
4. Read the real NMS plate with GT hidden: I1 A suppresses B and E survives; independently, I2 C suppresses D.
5. Run the threshold sweep and stress that its monotonic keep-count is specific to this disjoint-pair mini-batch, not a theorem about arbitrary greedy overlap graphs.
6. Compare image-aware/class-agnostic with image-and-class-aware NMS after adding overlapping I1/hand box H.
7. End by constructing two distinct nearby same-class objects that a low threshold wrongly merges.

Expected fresh outputs:

- pairwise eligible IoUs: `IoU(A,B)=.822`, `IoU(C,D)=.810`, `IoU(A,E)=.127`; cross-image entries are `nan`;
- grouped trace: I1/head keep A/suppress B → keep E; I2/head keep C/suppress D;
- final survivors: `['A','E','C']`;
- threshold sweep: A,E,C at $.3$, $.5$, and $.8$; all five at $.9$;
- image-aware/class-agnostic output omits H; `(image_id,class)`-aware output retains H together with A,E,C;
- appended candidate F has $.20×.99=.198$ and fails the $.30$ score threshold;
- the score-plus-NMS pipeline returns A,E,C;
- distinct-object failure pair has IoU `.364`; thresholds `.25` and `.35` keep one, while `.50` and `.70` keep two.

The executed notebook has 10 code checkpoints with counts `1–10` and zero error outputs. The local NMS kernel uses stable descending score order; the wrapper groups by image and class and globally score-sorts only after local suppression. Equal-score behavior must be declared because other implementations may break ties differently.

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
| one batch item suppresses another | missing `image_id` partition | assert every compared pair has equal image and class keys |
| nearby real objects merge | $τ_{NMS}$ too low for this geometry | trace the exact suppressing edge; try a higher threshold or Soft-NMS |
| one class deletes another | class-agnostic NMS | run the same boxes independently by class |
| confident isolated errors survive | expecting NMS to test correctness | compare the kept box with truth on the evaluation clock |

### `EVAL`

| Symptom | Suspect | Smallest discriminating test |
|---|---|---|
| high scores but low AP | confidently wrong ranking | inspect the first FP for class, IoU, and duplicate status |
| detection claims truth from another image | missing image identity in matcher | print `(pred_image, gt_image)` for every eligible matrix entry |
| AP50 high but COCO AP low | boxes are loose | compare AP50 with AP75; histogram matched IoUs |
| recall caps below one | score filtering, NMS, or missing candidates | lower score threshold before NMS and count unmatched truths |
| custom AP disagrees with library | protocol mismatch | print IoU rule, interpolation, class averaging, area, and max-detections settings |

Change one control at a time. In particular, never change the training matcher, score threshold, NMS threshold, and evaluation threshold together; the resulting AP movement would be uninterpretable.

## Common misconceptions and exact repairs

- **“A score threshold removes duplicates.”** It removes weak scores. A and B both survive $.70$.
- **“NMS removes false positives.”** It removes redundant predictions relative to stronger predictions. Isolated E survives.
- **“NMS uses ground truth.”** It runs at inference, where truth is unavailable.
- **“Boxes in a batch share one coordinate plane.”** Every image has its own plane. Cross-image NMS and matching comparisons are invalid, not zero-IoU pairs.
- **“A–E are detector outputs.”** They are constructed scored records placed over real images; no detector was run.
- **“All IoU thresholds are the same.”** Training assignment, NMS, and evaluation live on different clocks and make different decisions.
- **“Boxes count inclusive endpoints, so add one.”** Not in this lecture's continuous/half-open convention.
- **“Assignment and evaluation matching are the same.”** Assignment creates learning targets; evaluation labels a frozen ranked list.
- **“The highest-IoU truth can be checked first, then rejected if already used.”** The stated evaluator chooses the best **unmatched** eligible truth.
- **“Focal loss deletes easy examples.”** It continuously down-weights them.
- **“Higher NMS threshold always means a globally monotone keep count.”** It does in this simple two-cluster mini-batch; greedy interactions do not make that a universal theorem.
- **“One-stage always means faster and less accurate.”** It specifies the computation pattern, not a timeless performance ordering.
- **“AP is one universal formula.”** Always name the IoU cutoff, interpolation, class aggregation, area range, and max detections.
- **“Detection and segmentation belong in one lecture.”** Public Lecture 10 ends at boxes and AP. Public Lecture 11 owns pixel and instance masks.

## If a student asks

### “How can a fixed tensor represent a variable number of objects?”

The network emits a fixed **capacity per image** of candidate records. Decoding, score filtering, and NMS return a variable-length subset for each image. In the teaching ledger, 1200 candidate records are available per image even when that image contains one labeled head.

### “Why offsets instead of absolute pixel coordinates?”

Normalized translations and log-scale changes make the target local to the reference anchor and comparable across positions and object sizes. The same convolutional head can learn “move right by .2 anchor widths” everywhere.

### “Why use log width and height?”

Zero means unchanged scale, positive and negative values represent multiplicative growth and shrinkage symmetrically, and exponentiating at decode keeps sizes positive.

### “Why can E survive NMS?”

NMS only asks whether E is redundant with a stronger I1/head prediction. E is isolated. Correctness requires comparison with I1's truth, which happens later during evaluation.

### “Why class-aware NMS?”

Two boxes from different classes can legitimately occupy the same region, such as a hand beside a head. Class partitioning prevents one from deleting the other. Image partitioning is even stricter: boxes from different photographs can never suppress one another regardless of their numeric coordinates.

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
| 0–3 | L8B bridge, observed/GT/constructed contract, and opening vote |
| 3–7 | three clocks and box convention |
| 7–12 | held truth areas and A/B IoU $.822$ |
| 12–16 | fixed candidate capacity plus one-sentence assignment/loss contract |
| 16–24 | exact grouped NMS trace I1: A,B,E → A,E; I2: C,D → C |
| 24–30 | image-aware evaluation A=TP,E=FP,C=TP and global rank AP $5/6$ |
| 30–33 | one diagnostic per clock |
| 33–35 | opening revisit and L11 handoff |

Cut the focal-loss derivation, Smooth $L_1$/GIoU comparison, detector families, FPN, anchor-free heads, Soft-NMS, and live notebook execution. Show the already-verified notebook outputs only if needed. Never cut the clock distinction, held IoU, NMS trace, evaluation match, or revisit.

## Primary sources and exact companions

Use primary papers for historical or architectural claims:

- [Oxford-IIIT Pet dataset and annotations — Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [Cats and Dogs — Parkhi et al., CVPR 2012](https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/)
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

Evidence provenance:

- Dataset: Oxford-IIIT Pet; `Abyssinian_1` and `Bengal_10`; each XML provides a tight head ROI.
- License: CC BY-SA 4.0; copyright remains with the original image owners.
- Builder: `build_detection_evidence.py`, SHA-256 `1e6d3312df89424974b1fab8ccdbc5e11963dd4b6d6c802d64d33b09380fb3e1`.
- Manifest: `evidence.json`, SHA-256 `b539bc41f383d52f95ed2abeeaf764ccf325a1a03b000be7d377b075cac2836a`.
- `real-head-targets.png`: `4492e5bcda0b32f8a30923eead220f2d6381c7bc5b49fb7d48eb4f5cc1400217`.
- `real-candidates.png`: `1ca4372b8c062c4e4cb183c060669bdf3ea99c7884f0ae13f822f65f4570cdb6`.
- `real-nms-trace.png`: `7b53928e5d66469319adc69382d2fe1e17a08c95cd669b368c17180eb1b8c177`.
- `real-eval-matching.png`: `db6f20189f53afbc5a7a14c1075134e2729c05ad4053b40c552599579e58f25c`.

## L11 handoff · segmentation, not sequences

End with a clean change of output support:

- Lecture 10 detection: a variable set of `(class, box, score)` records;
- Lecture 11 semantic segmentation: class evidence at every pixel;
- Lecture 11 instance segmentation: pixel ownership separated by object instance.

Both reuse the L8B backbone. Lecture 11 replaces four-number rectangles with dense masks and asks how to recover boundaries that downsampling made coarse.

Final line:

> Today asked “which objects, and where?” Lecture 11 asks “which class owns every pixel—and which instance?”
