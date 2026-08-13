# Lecture 11 · Semantic and Instance Segmentation · Teaching Guide

*Public Lecture 11; repository file family L10. First-time-instructor companion for an 80-minute class.*

## Synchronization checkpoint

This guide is synchronized to the following evidence/notebook checkpoint; after a deck rebuild, refresh the source hash here before teaching:

- deck source `lecture10/L10-segmentation.typ`, SHA-256 `bd0e11a07d33939ab80c9821e7b62fc9387f68e7a08bd819a71d65042ee812f6`;
- `notebooks/L10/01_dice_vs_ce_toy_masks.ipynb`, SHA-256 `6bc6510ea53462baf20d722ac8a0bde8339ffc0f54ec6485dfb930cad7412908`;
- real-evidence manifest `shared/vision-evidence/oxford-iiit-pet/l10/evidence.json`, SHA-256 `b9d0a65cb38bf09253af5e072cee9bf46adfa3b50b9a5f91ef8ada98dc3bfa4a`.

The deck and notebook keep two evidence lanes separate. The Oxford-IIIT Pet lane starts with an observed photograph and official trimap, then diagnoses a declared constructed prediction. The exact $4\times4$ board lane remains a separate construction carried through truth, probabilities, inference, and semantic metrics. If an artifact changes, first verify its evidence category; then recheck every entry of the $4\times4$ $G$, $p$, and $P$ before teaching.

## The student contract

By the end of the core route, a student should be able to:

1. distinguish observed pixels, ground-truth annotations, constructed predictions, computed metrics, and actual model output;
2. read the official Oxford trimap values `1=foreground`, `2=background`, and `3=boundary/ignore`;
3. distinguish semantic, instance, and panoptic output structures;
4. explain why a segmentation head preserves spatial axes and shares a classifier across locations;
5. keep the `TRAIN`, `INFER`, and `EVAL` clocks separate;
6. derive per-pixel cross-entropy and Soft Dice on the held probability field;
7. explain why an encoder needs context, a decoder needs localization, and U-Net skips preserve aligned detail;
8. threshold the held probabilities, count TP/FP/FN/TN, and compute accuracy, IoU, and Dice;
9. state a complete semantic mIoU protocol, including classes, void pixels, absent classes, and reduction;
10. explain the two stages of Mask R-CNN, the purpose of RoIAlign, and the original class-specific mask loss;
11. distinguish box AP from mask AP; and
12. diagnose a segmentation failure with a symptom → suspect → smallest-test argument.

The lecture spine is:

> A dense model trains probabilities against pixel truth, turns probabilities into a structured output at inference, and evaluates that frozen output under a declared mask protocol. Real annotations ground the contract; constructed cases make the arithmetic inspectable. Semantic evidence answers area; instance identity answers count.

## Where Lecture 11 sits

### Bridge from public Lecture 10

Public Lecture 10 returned a variable scored set of `(class, box, score)` records. Lecture 11 keeps the spatial backbone evidence but changes the output support:

- semantic segmentation returns class evidence over an $H\times W$ grid;
- instance segmentation returns a variable scored set of object masks;
- panoptic segmentation returns a non-overlapping class–identity assignment over the image.

Say this bridge nearly verbatim:

> Detection drew rectangles around objects. Today the head must assign pixels, and when two same-class objects touch, it must preserve which pixels belong to which object.

Do not reteach detection. Retrieve only three ideas:

1. a backbone leaves a spatial feature map before global pooling;
2. an RPN proposes candidate RoIs and evaluation matches a ranked set one-to-one with truth;
3. box AP uses box IoU, which will become the template for mask AP under mask IoU.

### Prerequisites to retrieve, not reteach

- From L3/L4: logits, sigmoid/softmax, cross-entropy, and `-log p`.
- From L7: choose inference policies on validation data, freeze them, and use the test set only for the final report.
- From L8: spatial feature maps, $1\times1$ convolution, stride, padding, and receptive field.
- From L8B: encoder backbones, residual feature extraction, and transfer-learning heads.
- From L10: RPN, RoIs, score ranking, one-to-one matching, IoU, and AP.
- Mathematical tools: logarithms, means, set intersection/union, and a weighted average.

Use three retrieval questions:

1. “What information does global average pooling intentionally erase?”
2. “At one pixel with $K$ classes, along which axis should probabilities sum to one?”
3. “What was the difference between NMS at inference and matching at evaluation?”

If the room cannot answer the second question, draw a single $K$-vector at one location. Do not start with the whole image tensor.

## Stable teaching grammar

Keep the deck's semantic colors and nouns stable:

- teal: truth and `TRAIN`;
- blue: logits, probabilities, and `INFER` control;
- accent/orange: hard prediction;
- green: correct overlap or a verified result;
- red: FP, FN, misalignment, or failure;
- ink: `EVAL` and protocol.

The three clocks are the main conceptual guardrail:

| Clock | Inputs available | Main transformation | Output or decision |
|---|---|---|---|
| `TRAIN` | image, truth $G$, logits/probabilities $p$ | CE, Soft Dice, gradients | learned parameters |
| `INFER` | image, logits/probabilities, frozen policy | argmax/threshold, resize, NMS or fusion | hard semantic map or scored object masks |
| `EVAL` | frozen output and hidden truth | confusion, mask matching, declared reduction | IoU, Dice, mIoU, mask AP |

Whenever a student says “Dice,” “mask,” or “threshold,” ask which clock. Soft Dice on $p$ is not hard Dice on $P$; an inference threshold is not a training derivative; mask AP is not a training loss.

## Evidence taxonomy and the real trimap contract

Use these nouns literally throughout the lecture:

| Category | Lecture 11 object | Permitted claim |
|---|---|---|
| `OBSERVED` | `Abyssinian_1.jpg` RGB pixels | a real Oxford-IIIT Pet photograph |
| `GT` | the official `Abyssinian_1` trimap | dataset-provided pixel annotation |
| `CONSTRUCTED` | the real-mask probability field and the exact $4\times4$ $G/p/P$ case | a declared pedagogical construction |
| `COMPUTED` | thresholded masks, overlays, counts, losses, and metrics | deterministic consequences of the declared inputs and policy |
| `MODEL OUTPUT` | none in the evidence bundle or companion notebook | no checkpoint was loaded or executed |
| `MEASURED PERFORMANCE` | none | no benchmark or generalization claim is supported |

The official Oxford trimap uses value `1` for foreground, `2` for background, and `3` for a boundary/ignore band. For region losses and metrics, define valid pixels as values `1` or `2`. Exclude value `3` from BCE, Soft Dice, TP/FP/FN/TN, accuracy, IoU, and hard Dice: ignored pixels are not background and are not free true negatives. A separate boundary diagnostic may use the value-`3` band as a reference contour only if its distance and tolerance policy is stated.

The real prediction is deliberately controlled: shift the official foreground mask `16 px` right with zero fill, assign `.90` to shifted foreground and `.10` elsewhere, then threshold with $P=1[p\ge.50]$. This is a spatial failure construction, not inference. On valid pixels it gives TP `18,736`, FP `170`, FN `4,202`, TN `198,596`, BCE `.148690`, Soft Dice `.573856`, IoU `.810801`, hard Dice `.895517`, and accuracy `.980280`. The `18,296` ignored pixels enter none of those region reductions.

The one-pet photograph grounds semantic masks and ignore policy. It does not demonstrate instance separation. The generated two-tree scene remains an explicitly generated conceptual case for asking why two touching same-class objects need separate identities.

## The exact continuous 55 / 15 / 10 route

The source order is deliberate: core construction, `SHOULD`, `OPTIONAL`, then the held commitment and handoff. The core receives **55 total minutes**: 51 minutes before `SHOULD`, plus the protected four-minute closure at the end. This preserves continuous slide order while totaling exactly 55 / 15 / 10.

| Wall time | Tier | Deck move | Student action | Non-negotiable result |
|---:|---|---|---|---|
| 0–4 | CORE | L10 bridge and orchard commitment | choose A/B/C/D silently | preserve the vote; do not reveal D |
| 4–9 | CORE | route, three clocks, official trimap, then constructed held truth | classify observed/GT/constructed/computed evidence; name what each clock may see | value `3` is ignored for region scores; one semantic union cannot preserve two identities |
| 9–18 | CORE | per-pixel head, softmax/sigmoid, fixed $p$, dense BCE | calculate the pixel CE and grouped BCE | `.203/.674/.123`, CE `.395`; held BCE `.319` |
| 18–29 | CORE | spatial tensor, $1\times1$ head, encoder–decoder, U-Net | predict whether upsampling restores lost detail | resolution is not information; same-padding teaching ledger; skips concatenate |
| 29–40 | CORE | threshold commitment, counts, accuracy, IoU, Dice, Soft Dice, real error overlay, mIoU | commit TP/FP/FN/TN before reveal; then locate the real spatial errors | board: `5/3/1/7`, `.750/.556/.714`, Soft Dice `.727`; real valid-region IoU `.811`, Dice `.896` |
| 40–51 | CORE | semantic failure → instance output → Mask R-CNN → RoIAlign → mask AP | separate RPN and RoI losses; compute mask BCE | total has both stages; selected-channel BCE `.299`; mask AP uses mask IoU |
| 51–66 | SHOULD | void/empty policies, upsampling, bilinear sample, panoptic choice | state one complete metric policy | ignored pixels leave every sum/count; midpoint sample is `4` |
| 66–76 | OPTIONAL | primary provenance and architecture variants | name the changed contract or mechanism | avoid an architecture catalogue |
| 76–80 | CORE | orchard revisit, checklist, public L12 handoff | defend D and name the new sequence axis | separate masks count; their union gives area |

Tier totals are therefore CORE `51+4=55`, SHOULD `15`, OPTIONAL `10`.

If time slips, do not rush the commitment reveal. After mask AP, jump directly to “Revisit the orchard commitment” and “CORE · final checklist.” The 55-minute route is complete without any `SHOULD` or `OPTIONAL` slide.

## The constructed $4\times4$ orchard case

Use this exact board case continuously for hand calculation. It is `CONSTRUCTED`; it is not a downsampled Oxford annotation and it is not model output. Do not replace its $G$, $p$, or $P$ with the real-image arrays midway through a derivation. The real Oxford lane supplies visual evidence and a second, machine-checked metric audit; the $4\times4$ lane supplies the stable arithmetic spine.

### Truth and latent identity

The semantic foreground truth is

$$
G=
\begin{bmatrix}
0&1&1&0\\
0&1&1&0\\
0&1&1&0\\
0&0&0&0
\end{bmatrix},
\qquad |G|=6.
$$

Its latent instance map is

$$
I=
\begin{bmatrix}
0&1&2&0\\
0&1&2&0\\
0&1&2&0\\
0&0&0&0
\end{bmatrix}.
$$

The constructed semantic target withholds IDs: it contains one six-pixel union. That union supports total canopy area, while the identity map supports counting two touching trees.

### Fixed foreground probabilities

$$
p=
\begin{bmatrix}
.10&.90&.90&.10\\
.60&.90&.90&.10\\
.10&.90&.30&.60\\
.10&.60&.05&.05
\end{bmatrix}.
$$

Retain these invariants:

$$
\sum_i p_i=7.2,\qquad \sum_i g_i=6,\qquad \sum_i p_i g_i=4.8.
$$

### Frozen inference rule and hard prediction

The lecture uses

$$
P_i=\mathbf 1[p_i\ge .50],
$$

so equality is foreground. Thresholding gives

$$
P=
\begin{bmatrix}
0&1&1&0\\
1&1&1&0\\
0&1&0&1\\
0&1&0&0
\end{bmatrix}.
$$

The one ledger is:

| Quantity | Value |
|---|---:|
| $|P|$ | 8 |
| TP | 5 |
| FP | 3 |
| FN | 1 |
| TN | 7 |
| accuracy | $(5+7)/16=.750$ |
| foreground IoU | $5/(5+3+1)=5/9\approx.556$ |
| foreground Dice | $10/(10+3+1)=10/14\approx.714$ |

Do not introduce TP/FP/FN/TN before crossing into `EVAL`; those labels require truth.

## Commitment-before-reveal choreography

### Commitment 1 · orchard output contract

Read the requirement exactly:

> Two touching crowns occupy one foreground region. Report total canopy area and count the two trees.

Offer:

- A: one global class;
- B: one class plus one box;
- C: one $H\times W$ foreground map;
- D: a scored set of separate object masks.

Record the room's vote without commentary. At the final revisit, D is the complete answer: the two masks preserve identities and their union gives total area. If the entire background must also receive a class, a panoptic class–identity map is the fuller contract.

Call the orchard image a **generated teaching scene** and its $4\times4$ grids a **constructed board ledger**. Do not describe either as a dataset annotation. Conversely, call the Oxford photograph observed and its trimap official GT; do not imply that its shifted probability came from a network.

### Commitment 2 · probabilities or hard masks during training

Ask which object enters the derivative: hard $P$, differentiable $p$, or pixel accuracy. Reveal $p$. CE and Soft Dice operate before thresholding; $P$ appears after the `INFER` boundary.

### Commitment 3 · resolution versus recovered information

Show the $2\times2\to4\times4$ repeated grid. Ask whether the larger grid recovers which source pixels created each low-resolution value. The answer is no. Upsampling creates locations; aligned encoder evidence or another high-resolution route supplies discarded detail.

### Commitment 4 · held threshold and counts

Show $G$ and $p$, state $P_i=1[p_i\ge.50]$, and request all four counts before revealing $P$. Do not accept only IoU. The counts are TP=5, FP=3, FN=1, TN=7.

### Commitment 5 · context versus coordinates

Ask what is most at risk if the deep path is removed while high-resolution skips remain. The answer is semantic meaning. Crisp edges do not establish whether a region is road, cat, or shadow.

### Revisit

Return to the opening choices before showing the answer. Require a causal defense:

- C yields a semantic region and therefore total area;
- C cannot preserve two identities when the crowns touch;
- D yields two object masks;
- the union of those masks yields the total area.

The closing line is:

> Choose the output structure from the question, then make TRAIN loss, INFER policy, alignment, and EVAL protocol agree with that structure.

## Exact derivations to put on the board

### One-pixel softmax

For background, road, and car logits

$$
z=(.2,1.4,-.3),
$$

$$
e^z\approx(1.221,4.055,.741),\qquad \sum_k e^{z_k}\approx6.017,
$$

$$
p\approx(.203,.674,.123).
$$

Road wins, and if road is true,

$$
\ell=-\log .674\approx .395.
$$

Implementation boundary: compute softmax as `exp(z - max(z)) / sum(exp(z - max(z)))`. The raw exponentials above are safe only because this toy vector is small. Adding a common constant must leave the probabilities unchanged.

### Binary sigmoid and logits

For one foreground class, one logit is enough:

$$
p_i=\sigma(a_i),\qquad a_i=\log\frac{p_i}{1-p_i}.
$$

Retain the deck table:

| $p$ | .90 | .60 | .30 | .10 | .05 |
|---:|---:|---:|---:|---:|---:|
| `logit(p)` | 2.197 | .405 | −.847 | −2.197 | −2.944 |

A binary one-logit sigmoid is not the same representation as a $K$-class softmax, although both can express foreground probabilities.

### Held dense binary cross-entropy

Per pixel,

$$
\ell_i=-[g_i\log p_i+(1-g_i)\log(1-p_i)].
$$

The held field has:

- foreground: five `.90` and one `.30`;
- background: three `.60`, five `.10`, and two `.05`.

Therefore

$$
16L_{BCE}=-10\log .90-\log .30-3\log .40-2\log .95
$$

$$
\approx1.0536+1.2040+2.7489+.1026=5.1090,
$$

$$
L_{BCE}\approx5.109/16\approx.319.
$$

The coefficient `10` combines five foreground `.90` terms and five background `.10` terms; both contribute `−log .90`.

### Dense tensor and shared classifier

For a $K$-class semantic task,

$$
Z\in\mathbb R^{H\times W\times K},\qquad Z[h,w,:]\in\mathbb R^K.
$$

Softmax normalizes over $k$ independently for each $(h,w)$. A $1\times1$ convolution applies the same affine map $Wf_{h,w}+b$ at every location. It mixes channels locally; earlier spatial convolutions provide neighbourhood context.

### U-Net ledger and convention

The deck uses a same-padding teaching ledger:

$$
256^2\!\times32\to128^2\!\times64\to64^2\!\times128\to32^2\!\times256
$$

and decodes back to $256^2\times K$ logits.

At one skip level,

$$
U,E\in\mathbb R^{64\times64\times128}
\quad\Longrightarrow\quad
\operatorname{concat}(U,E)\in\mathbb R^{64\times64\times256}.
$$

State both conventions:

- in this teaching ledger, same-padding convolutions preserve stage sizes, so matching tensors concatenate directly;
- in the original 2015 U-Net, valid convolutions shrink feature maps, so the copied encoder feature is cropped before concatenation.

Original U-Net skips concatenate; they are not residual additions. Upsampling restores grid resolution, while skips restore aligned evidence.

### Hard accuracy, IoU, and Dice

From TP=5, FP=3, FN=1, TN=7:

$$
\operatorname{accuracy}=\frac{TP+TN}{16}=\frac{12}{16}=.750,
$$

$$
\operatorname{IoU}=\frac{TP}{TP+FP+FN}=\frac59\approx.556,
$$

$$
\operatorname{Dice}=\frac{2TP}{2TP+FP+FN}=\frac{10}{14}\approx.714.
$$

For one binary mask pair,

$$
\operatorname{Dice}=\frac{2\operatorname{IoU}}{1+\operatorname{IoU}},
\qquad
\operatorname{IoU}=\frac{\operatorname{Dice}}{2-\operatorname{Dice}}.
$$

This monotone relationship preserves ranking for one class and mask pair. Nonlinear averaging over images or classes can change aggregate model rankings.

### Accuracy trap

In a $100\times100$ image with only 100 foreground pixels, predicting background everywhere gives

$$
\operatorname{accuracy}=9900/10000=.99,
$$

but foreground recall, IoU, and Dice are all zero. Accuracy weights true background by frequency; IoU and Dice omit TN from the foreground denominator.

### Held Soft Dice and combined loss

$$
\operatorname{SoftDice}(p,g)=
\frac{2\sum_i p_i g_i+\epsilon}{\sum_i p_i+\sum_i g_i+\epsilon}.
$$

Ignoring the tiny $\epsilon$ in displayed arithmetic,

$$
\operatorname{SoftDice}=\frac{2(4.8)}{7.2+6}=\frac{9.6}{13.2}\approx.727,
$$

$$
L_{Dice}=1-.727=.273.
$$

A common declared objective is

$$
L=L_{CE}+\lambda L_{Dice}.
$$

CE supplies local probability supervision; Soft Dice couples foreground pixels through a region denominator. Combining them does not remove the need to state $\lambda$, reductions, smoothing, and empty-mask handling.

### Semantic mIoU protocol

For each reported class $k$,

$$
IoU_k=\frac{TP_k}{TP_k+FP_k+FN_k},
\qquad
mIoU=\frac1{|K_{report}|}\sum_{k\in K_{report}}IoU_k.
$$

“mIoU” is incomplete until the report declares:

1. which classes are in $K_{report}$ and whether background is included;
2. which pixels are void and therefore removed from all counts;
3. what happens when a class is absent from truth and prediction;
4. whether IoUs are averaged per image or derived from a dataset-level confusion matrix; and
5. whether the class reduction is macro, frequency weighted, or another named rule.

Do not average pixelwise “IoUs.” First accumulate the declared TP/FP/FN units, then apply the declared reduction.

### Mask R-CNN loss decomposition

Mask R-CNN has two training stages:

$$
L_{RPN}=L_{RPN\text{-}obj}+L_{RPN\text{-}box},
$$

$$
L_{RoI}=L_{cls}+L_{box}+L_{mask},
$$

$$
L_{total}=L_{RPN}+L_{RoI}.
$$

The familiar class+box+mask expression is the second-stage RoI loss, not the entire model loss.

In the original formulation,

$$
\hat M\in\mathbb R^{K\times m\times m}
$$

contains $K$ independent sigmoid mask channels. For a **matched positive RoI** of ground-truth class $k$, only channel $k$ contributes to $L_{mask}$. Negative RoIs have no mask target and contribute no mask loss. The class head uses softmax; the mask channels do not compete through a softmax.

### Selected-channel mask BCE

For

$$
M=\begin{bmatrix}1&1\\0&0\end{bmatrix},
\qquad
\hat M=\begin{bmatrix}.8&.6\\.3&.1\end{bmatrix},
$$

$$
L_{mask}=\frac14[.223+.511+.357+.105]\approx.299.
$$

State the scope: these are pixels inside one matched positive RoI and one selected ground-truth class channel.

### RoIAlign midpoint calculation

RoIPool rounds coordinates. RoIAlign keeps fractional coordinates and samples neighbouring features bilinearly. For a point halfway in both directions among

$$
\begin{bmatrix}1&3\\5&7\end{bmatrix},
$$

all weights are `.25`, so

$$
.25(1+3+5+7)=4.
$$

RoIAlign reduces feature-coordinate quantization. It does not reconstruct image detail that the backbone discarded.

### Box AP versus mask AP

Instance-mask evaluation follows the L10 ranked-set pattern:

1. sort predicted instances by score;
2. restrict matches to the same class;
3. match one-to-one to unmatched truth;
4. decide eligibility with **mask IoU** at the stated threshold;
5. form precision–recall and AP under the stated protocol.

Box AP uses box IoU. Mask AP uses mask IoU. A model can have good box AP and poor mask AP when boxes localize objects but boundaries are inaccurate. Never report a bare “AP” when both are available.

## Deck-order teaching choreography

### Part I · bridge, commitment, and output contracts

1. Retrieve L10's scored box set.
2. Hold the orchard vote.
3. Name all three clocks before showing a loss or metric.
4. Show the real Oxford photograph beside its official trimap; classify observed pixels, GT, and the ignored boundary.
5. Return to the generated orchard and show the constructed semantic truth beside the latent two-ID map.
6. Compare semantic, instance, and panoptic contracts.
7. Promise that the board $G$, $p$, threshold, and $P$ will not change.

The intellectual move is output design, not architecture naming.

### Part II · `TRAIN`: pixel classification

Start with one pixel, then expand to the tensor. Derive the small softmax, switch explicitly to the binary one-logit held case, freeze $p$, and reproduce BCE `.319`. End by asking why labels supervise $p$, not thresholded $P$.

### Part III · `TRAIN`: spatial architecture

Show why global pooling is destructive for dense output. Establish $1\times1$ channel mixing, then follow encode → context → decode → logits. Hold the resolution-versus-information commitment before revealing U-Net skips. State same-padding teaching convention versus original crop-and-concatenate convention. End with the smallest boundary diagnostic.

### Part IV · `INFER` and `EVAL`

Cross the clock boundary aloud:

> Training used $G$ and $p$. Inference has $p$ and a frozen threshold, but no $G$. Evaluation brings $G$ back only after $P$ is frozen.

Commit the board counts, then derive every hard metric from the same four values. Return briefly to `TRAIN` for Soft Dice. After the exact arithmetic is secure, show the real Oxford error overlay: the official trimap is GT, the `+16 px` probability is constructed, and the overlay/counts are computed. Ask why `18,296` boundary pixels are absent from every region count. Then move back to `EVAL` for mIoU. These clock and evidence-lane switches are intentional; announce both.

### Part V · instance segmentation

Return to the touching crowns. The semantic union cannot count two objects. Introduce the scored instance set, follow image → backbone → RPN → RoIAlign → class/box/mask heads, and keep the two loss stages separate. Derive the selected-channel BCE, explain alignment, and close with mask AP.

### Part VI · `SHOULD`

Treat policy as part of the model report:

- void pixels leave loss sums, denominators, and confusion counts;
- empty–empty masks require an explicit rule;
- upsampling mechanisms differ in how locations are created and refined;
- RoIAlign is precise sampling, not super-resolution;
- panoptic output must fuse overlaps into one class–identity assignment.

### Part VII · `OPTIONAL`, revisit, and handoff

Use the primary-paper slide to attach claims to sources, not to tell an architecture history. On the variants slide, ask which assumption changed: mask-channel count, context mechanism, proposal mechanism, matching, or fusion. Then revisit the orchard and end with the five-item checklist.

## Notebook choreography

Use exactly one notebook. It is deterministic and uses the standard library, NumPy, and Matplotlib. It performs no training and invokes no model checkpoint or evidence builder. When the repository files are unavailable, it downloads only SHA-pinned evidence assets from the course repository into a temporary cache, then rejects any hash mismatch.

[Open the exact segmentation Colab](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L10/01_dice_vs_ce_toy_masks.ipynb).

Pinned notebook SHA-256: `6bc6510ea53462baf20d722ac8a0bde8339ffc0f54ec6485dfb930cad7412908`.

Use the notebook rhythm:

`predict → calculate → run → inspect → explain`.

Do not live-code from an empty cell. The value is the commitment and the explanation, not syntax entry.

### Recommended placement and retained outputs

| Deck moment | Notebook move | Predict before running | Retained output |
|---|---|---|---|
| official trimap contract | validate source/manifest/NPZ hashes and inspect the real plate | which value is excluded, and from which reductions | values `1/2/3`; counts `22,938/198,766/18,296`; taxonomy reports no model output |
| real constructed shift | recompute valid-region metrics from NPZ arrays | where will FP/FN bands appear after `+16 px` | TP/FP/FN/TN `18,736/170/4,202/198,596`; BCE `.148690`; Soft Dice `.573856`; IoU `.810801`; hard Dice `.895517` |
| one-pixel softmax | run the stable softmax cell | winning class and whether $p>.8$ | exp `[1.221,4.055,.741]`; $p=[.203,.674,.123]`; road; CE `.395` |
| dense CE | run the $2\times2$ field | largest-loss pixel and mean scale | losses `[[.223,.511],[.105,1.386]]`; mean `.556`; index `(1,1)` |
| void policy | exclude bottom-right | whether the mean rises or falls | 3 valid pixels; CE `.280` |
| held hard mask | commit four counts | TP/FP/FN/TN, then IoU/Dice | `5,3,1,7`; accuracy `.750`; IoU `.556`; Dice `.714` |
| all-background trap | predict which metric remains deceptively high | small and large imbalance cases | small accuracy `.625`; large accuracy `.990`; both overlap scores `0` |
| empty protocol | compare empty/empty with empty/nonempty | declared edge outcomes | empty/empty IoU=Dice=`1`; empty truth+one FP=`0` |
| one more FP | flip a TN | which formulas change | accuracy `.750→.688`; IoU `.556→.500`; Dice `.714→.667` |
| held probability field | compute sums and threshold | Soft Dice, BCE, and whether $P$ is reproduced | sums `7.2,6,4.8`; Soft Dice `.727273`; loss `.272727`; BCE `.319`; `True` |
| Mask R-CNN branch | calculate four BCE terms | largest penalty | `[[.223,.511],[.357,.105]]`; mean `.299` |

The $4\times4$ mask plot must remain the exact constructed held $G/P$ pair. White digits mark foreground cells for contrast. The real plates remain a separate evidence lane; never relabel their constructed probability or hard mask as model output.

### Notebook protocol boundaries

- The manifest, raw photograph, official trimap, NPZ arrays, metric ledger, and displayed plates are SHA-pinned. Local files are preferred; the Colab fallback uses the public repository URL and a temporary cache.
- The notebook loads `real-mask-arrays.npz`; it never invokes `build_segmentation_evidence.py`.
- For the real case, trimap values `1/2/3` mean foreground/background/boundary-ignore. Value `3` is removed before every region loss and confusion reduction.
- The real $p$ is a declared `+16 px` construction with `.90/.10` probabilities. The notebook contains no model output and reports no measured performance.
- The stable softmax subtracts the maximum logit; raw exponentials are displayed only for the safe toy vector.
- Thresholding uses `>= .5`, matching the deck.
- `hard_metrics` declares empty–empty IoU/Dice as `1`; empty truth plus predicted foreground is `0`.
- The empty convention is a teaching choice, not a universal standard. Dataset averaging must still be declared.
- The final mask BCE is for a matched positive RoI. Negative RoIs have no mask target.
- There is no train/validation/test split because this notebook fits no parameters and selects no policy. The real-image numbers diagnose a construction; they are not a benchmark. A trained experiment must select threshold and reduction rules on validation data and freeze them before test evaluation.

## A controlled experiment worth assigning

Keep architecture, data split, loss, optimizer, threshold-selection procedure, and evaluation code fixed. Change only one spatial mechanism:

- model A: decoder without the highest-resolution skip;
- model B: identical decoder with that skip.

Measure validation BCE, Soft Dice, mIoU, and a boundary-specific error such as boundary IoU or distance. Inspect the same images at the same threshold. A claim that “skips help” is credible only if the comparison isolates the skip and reports both region and boundary behaviour.

For an inference-policy experiment, hold weights fixed, sweep the threshold on validation data, select one rule, then evaluate the untouched test set once. Do not tune the threshold on test mIoU.

## Diagnostics by clock

### `TRAIN`

| Symptom | Suspect | Smallest discriminating test |
|---|---|---|
| NaN/Inf probabilities or loss | unstable softmax or `log(0)` | subtract max; inspect logits; use a logits-based loss |
| CE falls; foreground IoU stays zero | background collapse | histogram $p$ by truth class; inspect foreground recall |
| correct class; blurred edge | coarse stride or missing aligned skip | overlay boundaries; ablate one skip at fixed resolution |
| checkerboard texture | transposed-convolution overlap | compare resize+conv at the same output size |
| void region changes loss or mIoU | ignore mask applied to only one reduction | print valid counts for loss and confusion separately |
| mask loss is always zero | no positive RoIs reach the mask head | count sampled positive RoIs and selected mask channels |

### `INFER`

| Symptom | Suspect | Smallest discriminating test |
|---|---|---|
| every pixel is background | threshold too high or collapsed logits | inspect per-class probability histograms before thresholding |
| mask size is right but boundary is shifted | crop/resize/RoI coordinate mismatch | overlay RoI, sampled grid, and pasted mask on one example |
| touching objects become one blob | semantic output used for an instance question | inspect instance IDs before fusion; use an instance output contract |
| duplicate object masks remain | missing or misconfigured box NMS or post-processing | print score order and pairwise box/mask IoUs |
| panoptic pixels overlap or stay unassigned | fusion policy incomplete | visualize per-pixel winning instance/stuff rule |

### `EVAL`

| Symptom | Suspect | Smallest discriminating test |
|---|---|---|
| 99% accuracy; IoU=0 | rare foreground | print class counts and foreground recall |
| loss falls; mIoU is flat | threshold or class collapse | freeze weights; sweep threshold on validation; inspect confusion |
| good box AP; poor mask AP | boundary alignment | compare matched box IoU and mask IoU for the same instances |
| mIoU differs across libraries | class/void/absent/reduction mismatch | print every protocol choice and raw confusion counts |
| Dice jumps when empty images are added | empty-mask policy | report scores with and without empty cases under the same rule |

Change one lever at a time. Loss, inference threshold/fusion, and evaluation reduction live on different clocks; changing all three together makes the result uninterpretable.

## Common misconceptions and exact repairs

- **“Every colored mask is a model prediction.”** The Oxford trimap is official GT; the `+16 px` probability is constructed; thresholding and metrics are computed. This lecture executes no checkpoint.
- **“The Oxford boundary band is background.”** Official value `3` is ignore for region scores. Remove it from both numerator and denominator; do not count it as TN.
- **“The $4\times4$ grid is the cat mask at low resolution.”** It is a separate constructed board case. Its fixed values are chosen for exact arithmetic, not derived from the photograph.
- **“Segmentation is unrelated to classification.”** A semantic head shares a classifier over spatial locations; the output support is dense.
- **“Softmax normalizes over all image pixels.”** It normalizes over classes at one location.
- **“Thresholded masks enter the gradient.”** CE and Soft Dice use logits/probabilities; thresholding is nondifferentiable inference.
- **“A larger upsampled grid contains the original detail.”** It contains more locations, not automatically more evidence.
- **“Every U-Net skip is an addition.”** Original U-Net concatenates encoder and decoder features.
- **“The deck reproduces the original U-Net sizes.”** It uses same-padding teaching sizes; original valid convolutions require cropping.
- **“Dice and IoU reward true background.”** Their foreground forms omit TN.
- **“Dice is always a fixed transform of IoU after averaging.”** The transform holds for one class and mask pair; nonlinear aggregation can alter rankings.
- **“Soft Dice `.727` should equal hard Dice `.714`.”** They use different objects: probabilities $p$ versus thresholded mask $P$.
- **“mIoU is a universal single number.”** It depends on classes, void handling, absent classes, and reduction.
- **“The Mask R-CNN loss is just class + box + mask.”** That is the second-stage RoI loss; total training also includes RPN objectness and box losses.
- **“The $K$ mask channels form a softmax.”** Original Mask R-CNN uses $K$ independent binary sigmoid masks; the class head selects the supervised channel.
- **“Every RoI trains a mask.”** Only matched positive RoIs have mask targets.
- **“RoIAlign reconstructs lost image detail.”** It avoids coordinate quantization in feature sampling.
- **“Mask AP is box AP with a different label.”** The ranked matching pattern is shared, but eligibility uses mask IoU instead of box IoU.
- **“Connected components of a semantic mask always recover instances.”** Touching objects can form one component; identity must be represented or inferred by an instance method.
- **“Lecture 12 continues with RNNs.”** Public Lecture 12 is next-token prediction; recurrent models come later in the public order.

## If a student asks

### “Why use a $1\times1$ convolution?”

It is the shared affine classifier at each spatial location. It mixes the $C$ channels into $K$ class logits without changing $H$ or $W$. Spatial context was accumulated by earlier kernels.

### “Why not train only with Dice?”

Dice directly rewards region overlap and resists foreground rarity, while CE provides local probability supervision and often more interpretable per-pixel gradients. Either can be used alone, but the choice and reduction must be declared and validated. The lecture's combined loss is a design pattern, not a universal law.

### “How should I choose the threshold?”

Choose it on validation data under the deployment objective, then freeze it. Report the selection rule. Do not tune it on the test set, and do not silently use a different threshold for each model.

### “What should Dice be when both masks are empty?”

There is no universal answer. This notebook assigns `1`, assigns `0` to empty truth plus nonempty prediction, and requires the averaging rule to be stated. Excluding empty cases or using a smoothing convention can also be valid if declared consistently.

### “Should background be included in mIoU?”

That is dataset/protocol dependent. Include or exclude it deliberately and name the reported class set. A background-dominated score can obscure the foreground task.

### “Why can CE improve while mIoU stays flat?”

CE can improve probability calibration without crossing the frozen threshold, or a frequent class can dominate the mean loss. Inspect probability histograms, per-class confusion, and the validation threshold rather than immediately changing architecture.

### “Why crop in original U-Net?”

Its valid convolutions reduce spatial size. The encoder feature is therefore larger than the corresponding decoder feature and must be cropped before concatenation. Same-padding implementations often align directly.

### “Are transposed convolutions required?”

No. Nearest or bilinear resize followed by convolution is common. The key questions are how resolution is created, what refinement is learned, and where aligned fine detail enters.

### “Why does Mask R-CNN need both an RPN and a mask head?”

The RPN proposes where candidate objects may be. The second-stage heads classify and refine each sampled RoI, while the mask head predicts which pixels inside a positive RoI belong to the object.

### “Why not use softmax across the $K$ masks?”

Those channels are class-specific binary shapes, not mutually exclusive object categories at one pixel. The class head predicts the category; the selected mask channel predicts foreground/background within that RoI.

### “Can modern models avoid RPNs or class-specific masks?”

Yes. Query-based mask sets can replace proposal RoIs and one-to-one match predictions, while class-agnostic heads can predict one mask per instance. The output contract, matching rule, alignment, and evaluation protocol still have to be explicit.

### “What is panoptic segmentation adding?”

It requires one non-overlapping class–identity assignment for every non-void pixel: stuff gets a class, things get a class and instance ID. Overlapping instance hypotheses therefore need a fusion rule. PQ is a separate declared panoptic metric, not semantic mIoU or mask AP.

## Emergency short routes

### Complete core · 55 minutes

Teach minutes 0–51 of the main route, skip directly over `SHOULD` and `OPTIONAL`, then spend four minutes on the orchard revisit and L12 handoff. Never cut the observed-photo/official-trimap contract, the evidence taxonomy, clock distinction, fixed $4\times4$ $G/p/P$, U-Net convention, hard metrics, Mask R-CNN loss boundary, mask AP distinction, or revisit.

### Emergency · 35 minutes

| Time | Keep |
|---:|---|
| 0–4 | L10 bridge, orchard vote, semantic/instance contract |
| 4–7 | real photograph + official trimap contract, three clocks, and fixed board $G\to p\to P$ promise |
| 7–13 | one-pixel softmax plus held BCE `.319` |
| 13–19 | encoder–decoder trade-off and U-Net concat convention |
| 19–26 | threshold, counts `5/3/1/7`, IoU `.556`, Dice `.714`, Soft Dice `.727` |
| 26–32 | instance set, RPN versus RoI loss, selected mask BCE `.299`, mask AP |
| 32–35 | orchard answer D, checklist, next-token handoff |

Cut the sigmoid-logit table, detailed mIoU protocol slide, boundary diagnostic table, all `SHOULD`/`OPTIONAL` material, and live notebook execution. Show the already-verified notebook outputs only if a calculation stalls.

### Emergency · 20 minutes

Use only six moves:

1. box set → pixel/instance output contract;
2. orchard vote plus one real photo/official-trimap plate;
3. evidence taxonomy, fixed constructed $G,p,P$, and the three clocks;
4. BCE `.319`, hard IoU `.556`, hard Dice `.714`, Soft Dice `.727`;
5. U-Net context+detail and Mask R-CNN identity+alignment;
6. orchard answer D and public L12 handoff.

Do not attempt a paper survey or live code in this route.

## Primary sources and exact companion

Use primary sources for historical and architectural claims:

- [Oxford-IIIT Pet dataset and annotation contract](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [Cats and Dogs — Parkhi et al., CVPR 2012](https://www.robots.ox.ac.uk/~vgg/publications/2012/Parkhi12a/)
- [Fully Convolutional Networks — Long, Shelhamer, and Darrell](https://arxiv.org/abs/1411.4038)
- [U-Net — Ronneberger, Fischer, and Brox](https://arxiv.org/abs/1505.04597)
- [V-Net and a Dice-based objective — Milletari, Navab, and Ahmadi](https://arxiv.org/abs/1606.04797)
- [Mask R-CNN — He et al.](https://arxiv.org/abs/1703.06870)
- [Panoptic Segmentation — Kirillov et al.](https://arxiv.org/abs/1801.00868)
- [Microsoft COCO — Lin et al.](https://arxiv.org/abs/1405.0312)

Exact executable companion:

- [Pixel losses, semantic overlap, and mask BCE Colab](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L10/01_dice_vs_ce_toy_masks.ipynb)

Exact real-evidence bundle:

- `shared/vision-evidence/oxford-iiit-pet/l10/README.md` documents source, CC BY-SA 4.0 attribution, trimap policy, construction, metric definitions, reproducible build, and safe claims.
- `evidence.json` binds source, builder, and derivative hashes; `real-mask-arrays.npz` is the compact replay source used by the notebook.

## Public L12 handoff · next-token prediction, not RNNs

End with an analogy and one new constraint:

- Lecture 11 shares one classifier over spatial positions $(h,w)$;
- public Lecture 12 shares one vocabulary classifier over sequence positions $t$;
- pixel classes become vocabulary tokens;
- a segmentation model may use context from both spatial directions, while next-token prediction must not look at future tokens.

Final line:

> Today every spatial position asked “which class owns me?” Next time every sequence position asks “which token comes next?”—with causality now enforcing what evidence is allowed.
