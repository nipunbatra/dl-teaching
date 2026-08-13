// Generalization and Regularization — Lecture 7 · ES 667 Deep Learning
// Compile from the repository root:
//   typst compile --root . --input handout=true lecture7/L7-regularization.typ /tmp/L7-handout.pdf
//   typst compile --root . lecture7/L7-regularization.typ /tmp/L7-presentation.pdf

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *

#show: metropolis-deck.with(
  title: [Generalization and Regularization],
  subtitle: [Diagnose the gap, change one lever, measure again],
)

#let NB = "https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L07/"
#let NB1 = NB + "01_overfitting_and_early_stopping.ipynb"
#let NB2 = NB + "02_dropout_from_scratch.ipynb"

#let MEM = "https://arxiv.org/abs/1611.03530"
#let ADAMW = "https://arxiv.org/abs/1711.05101"
#let DROPOUT = "https://jmlr.org/papers/v15/srivastava14a.html"
#let MIXUP = "https://arxiv.org/abs/1710.09412"
#let SMOOTH = "https://openaccess.thecvf.com/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html"
#let PTCE = "https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html"

// Semantic grammar used on every slide:
// TEAL = training / fitted parameters · BLUE = validation / selection
// ACC = intervention / regularization knob · GREEN = accepted evidence
// RED = observed failure · INK = sealed test / neutral structure.
#let swatch(color, body) = box(width: 13pt, height: 4pt, fill: color, radius: 1pt) + h(4pt) + body
#let semantic-legend = align(center, text(size: 13pt)[
  #swatch(TEAL, [training]) #h(14pt)
  #swatch(BLUE, [validation]) #h(14pt)
  #swatch(ACC, [intervention]) #h(14pt)
  #swatch(GREEN, [accepted evidence]) #h(14pt)
  #swatch(RED, [failure]) #h(14pt)
  #swatch(INK, [sealed test])
])

#let hairline(title, body, color: TEAL) = block(
  width: 100%, inset: 3pt,
  [#text(size: 13.5pt, weight: 700, fill: color)[#upper(title)]
   #v(2pt)
   #line(length: 100%, stroke: 0.8pt + color)
   #v(5pt)
   #body],
)

#let card(title, body, color: TEAL, width: 100%) = block(
  width: width, inset: (x: 10pt, y: 8pt), fill: white,
  stroke: 1pt + color, radius: 3pt,
  [#text(size: 12.5pt, weight: 700, fill: color)[#upper(title)]
   #v(4pt)
   #body],
)

#let flow-arrow(label: none, color: MUTED) = align(center, [
  #if label != none { text(size: 10.5pt, weight: 600, fill: color, label); linebreak() }
  #text(size: 24pt, fill: color)[$arrow.r$]
])

#let note(body, color: TEAL) = block(
  width: 100%, inset: (left: 11pt, right: 4pt, top: 5pt, bottom: 5pt),
  stroke: (left: 2pt + color), fill: color.lighten(93%), radius: 2pt,
  [#body],
)

#let caption(body) = align(center, text(size: 14pt, fill: MUTED)[#body])

#let anchor = block(
  width: 100%, inset: (x: 11pt, y: 6pt),
  fill: rgb("#F3F6F5"), stroke: (left: 3pt + TEAL), radius: 3pt,
  text(size: 14.5pt)[
    *RUNNING CASE · FIXED TEACHING TRACE* · five-class image classifier · chance $20%$ ·
    #text(fill: TEAL)[train $99%$] · #text(fill: BLUE)[validation $72%$ on 100 examples] ·
    #text(fill: RED)[gap $27$ points] · #text(fill: INK)[test sealed]
  ],
)

#let ledger(
  gap: [pending], checkpoint: [pending], weights: [pending],
  dropout: [pending], data: [pending], targets: [pending],
) = block(
  width: 100%, inset: 8pt, fill: rgb("#F8F8F6"),
  stroke: 0.6pt + rgb("#D7DDDC"), radius: 3pt,
  [#text(size: 11pt, weight: 700, fill: MUTED, tracking: 0.8pt)[CASE LEDGER · OBSERVE → INTERVENE → MEASURE]
   #v(4pt)
   #set text(size: 12.8pt)
   #table(
     columns: (22%, 78%), stroke: 0.35pt + rgb("#D7DDDC"),
     inset: (x: 7pt, y: 3.5pt), align: (left, left),
     [gap / split], gap,
     [checkpoint], checkpoint,
     [weights], weights,
     [representations], dropout,
     [data], data,
     [targets], targets,
   )],
)

#let epochs = (1, 2, 3, 4, 5, 6)
#let train-losses = (1.20, 0.80, 0.45, 0.32, 0.28, 0.25)
#let val-losses = (1.25, 0.98, 0.82, 0.78, 0.84, 0.97)
#let early-curves = lines(
  x: epochs, y: (train-losses, val-losses),
  labels: ([training], [validation]), colors: (TEAL, BLUE),
  markers: true, vlines: ((4, [restore epoch 4], ACC),),
  points: ((4, 0.78, [validation minimum = 0.78]),),
  x-label: [epoch], y-label: [cross-entropy], size: (112mm, 47mm),
)

#title-slide()

// ═══════════════════════════ DIAGNOSE · CORE ═══════════════════════════
= Diagnose before prescribing

== L6 made the network trainable; L7 asks what survives new data #V

#align(center, grid(
  columns: (56mm, 13mm, 56mm, 13mm, 56mm), gutter: 4pt, align: horizon,
  card([L6 · TRAINABILITY], [Signals and gradients remain usable through depth.], color: TEAL),
  flow-arrow(label: [enables], color: TEAL),
  card([OPTIMIZATION], [Parameters reach low training loss.], color: TEAL),
  flow-arrow(label: [does not imply], color: RED),
  card([L7 · GENERALIZATION], [Predictions remain useful on unseen examples.], color: BLUE),
))

#pause
#v(12pt)
#result[Today’s unit of reasoning is not a named trick: it is *symptom → suspect → controlled test*.]

== Commit: what should we do with 99% train and 72% validation? #Q

#anchor
#v(9pt)

#mcq(
  [Choose the first defensible action. Keep your answer until the closing revisit.],
  [Deploy: $99%$ training accuracy shows the classifier is ready],
  [Add every regularizer at once and keep the best test score],
  [Inspect validation curves, preserve the test seal, then change one lever],
  [Tune directly on the test set because validation is lower],
)

#pause
#note([Write *A, B, C,* or *D* plus one sentence of evidence. No answer is revealed yet.], color: BLUE)

== Keep the commitment; every section updates one ledger

#anchor
#v(7pt)

#ledger(
  gap: [#uncover("2-")[#text(fill: RED)[observed] · $99-72=27$ points; cause not yet identified]],
  checkpoint: [#uncover("3-")[validation curve not yet inspected; test remains sealed]],
)

#pause
#pause
#pause

#note([The ledger separates *evidence already measured* from *the next intervention to test*.], color: ACC)

== Outline: diagnose, intervene, verify #V

#align(center, diagram(spacing: (18mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let beats = ([measure the gap], [seal the splits], [change one lever], [select on validation], [test once])
  let colors = (RED, INK, ACC, BLUE, GREEN)
  for (i, beat) in beats.enumerate() {
    node((i, 0), text(size: 13pt)[#beat], shape: fletcher.shapes.rect, corner-radius: 3pt,
      inset: 6pt, stroke: 1pt + colors.at(i))
  }
  for i in range(4) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.9pt + MUTED) }
}))

#v(12pt)
#note([Every method enters at the third beat; the evidence protocol stays unchanged.], color: ACC)

#v(10pt)
#semantic-legend

== A gap is a diagnostic, not a verdict #D

#anchor

#pause
$ "gap" = "Acc"_"train" - "Acc"_"val" = 0.99 - 0.72 = 0.27. $

#pause
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([WHAT IT SAYS], [performance differs by $27$ percentage points on this fixed split], color: RED),
  hairline([WHAT IT DOES NOT SAY], [which remedy will help, or whether $72%$ meets the deployment target], color: MUTED),
  hairline([NEXT EVIDENCE], [training and validation loss by epoch, plus split integrity], color: BLUE),
))

#pause
#result[Regularize only after the measurements distinguish *underfitting* from *overfitting*.]

== Low training loss can coexist with memorization #V

#align(center, grid(
  columns: (52mm, 12mm, 52mm, 12mm, 52mm), gutter: 5pt, align: horizon,
  card([TRAINING OBJECTIVE], [Rewards fitting the examples it receives.], color: TEAL),
  flow-arrow(color: TEAL),
  card([HIGH CAPACITY], [Can represent many rules consistent with those examples.], color: ACC),
  flow-arrow(color: RED),
  card([UNSEEN INPUTS], [Reveal which rule the learning process selected.], color: BLUE),
))

#pause
#v(10pt)
#note([
  Deep networks can fit random labels, so training fit alone does not identify a useful rule.
  Evidence: #link(MEM)[Zhang et al. (2017), *Understanding deep learning requires rethinking generalization*].
], color: RED)

#pause
#caption[This claim motivates a measurement protocol; it does not claim that our five-class labels are random.]

// ═══════════════════════ SPLITS + STOPPING · CORE ═════════════════════
= Split the decisions, then stop on validation

== Three splits have three different jobs #V

#align(center, grid(
  columns: (54mm, 12mm, 54mm, 12mm, 54mm), gutter: 5pt, align: horizon,
  card([TRAIN · TEAL], [Gradient updates fit $theta$. Reused every epoch.], color: TEAL),
  flow-arrow(label: [fit], color: TEAL),
  card([VALIDATE · BLUE], [Select epoch, $lambda$, $p_"drop"$, and transforms.], color: BLUE),
  flow-arrow(label: [freeze], color: BLUE),
  card([TEST · INK], [One final report after all choices are frozen.], color: INK),
))

#pause
#v(12pt)
#align(center, table(
  columns: (38mm, 58mm, 102mm), stroke: 0.45pt + MUTED,
  inset: (x: 10pt, y: 6pt), align: (left, left, left),
  table.header([split], [may affect], [must not affect]),
  [#text(fill: TEAL)[train]], [weights], [final reported metric],
  [#text(fill: BLUE)[validation]], [model and checkpoint choices], [gradient updates in the basic protocol],
  [#text(fill: INK)[test]], [final report only], [weights, hyperparameters, or retry decisions],
))

== Why does repeated test checking leak information? #Q

#mcq(
  [After every run, a team reports test accuracy and chooses the next $lambda$. What has the test set become?],
  [A larger training set],
  [A validation set used indirectly for selection],
  [An unbiased final estimate with lower variance],
  [A regularizer because the labels were not differentiated],
)

#pause
#note([Commit before the next slide. Name the decision that consumed test information.], color: BLUE)

== Answer: selection turns test into validation #A

#mcq-answer(
  [B],
  [a validation set used indirectly for selection],
  [The chosen $lambda$ depends on test outcomes. The final reported score is therefore conditional on having searched that same set.],
)

#pause
#result[Seal the test set; if it changes a decision, it was not a final test.]

== Which checkpoint should survive? #Q

#anchor
#v(7pt)

#align(center, table(
  columns: 7, stroke: 0.45pt + MUTED, inset: (x: 10pt, y: 6pt), align: center,
  table.header([epoch], [1], [2], [3], [4], [5], [6]),
  [#text(fill: TEAL)[train loss]], [1.20], [0.80], [0.45], [0.32], [0.28], [0.25],
  [#text(fill: BLUE)[validation loss]], [1.25], [0.98], [0.82], [0.78], [0.84], [0.97],
))

#pause
#note([Choose the checkpoint and explain why epoch 6 is not automatically best.], color: BLUE)

== Restore epoch 4: it is the observed validation minimum #A

#align(center, early-curves)

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([SELECTION], [minimum validation loss is $0.78$ at epoch $4$], color: BLUE),
  hairline([PATIENCE = 2], [epochs $5$ and $6$ fail to improve; stop after epoch $6$, restore epoch $4$], color: ACC),
))

#pause
#caption[The plotted points are exactly the six values in the preceding table; both curves share one loss axis.]

== Ledger update: the first remedy is a checkpoint, not a new model

#anchor
#v(6pt)

#ledger(
  gap: [#text(fill: RED)[observed] · $27$ points on the fixed validation split; test sealed],
  checkpoint: [#text(fill: GREEN)[keep] · epoch $4$, validation loss $0.78$; patience $2$ stops after epoch $6$],
)

#pause
#note([Early stopping limits how far optimization follows the training objective. It does not repair a broken split or distribution shift.], color: ACC)

// ═════════════════════════ WEIGHT DECAY · CORE ════════════════════════
= Constrain the weights

== Weight decay changes which fitting solution is preferred #V

#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([DATA FIT], [$cal(L)_"train"(theta)$ rewards agreement with training labels], color: TEAL),
  hairline([SIZE PREFERENCE], [$lambda/2 norm(theta)_2^2$ penalizes large parameter coordinates], color: ACC),
))

#pause
$ cal(J)(theta) = cal(L)_"train"(theta) + lambda/2 norm(theta)_2^2. $

#pause
#note([
  The factor $1/2$ is a convention: it makes $nabla_theta (lambda norm(theta)^2/2)=lambda theta$.
  Always read the implementation before comparing numerical $lambda$ values.
], color: ACC)

== Derive the shrinkage step before naming the optimizer #D

Let $g_t=nabla_theta cal(L)_"train"(theta_t)$. For plain SGD on the objective above,

#pause
$ theta_(t+1) = theta_t - eta (g_t + lambda theta_t). $

#pause
$ theta_(t+1) = underbrace((1-eta lambda) theta_t)_"shrink" - underbrace(eta g_t)_"fit data". $

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([INTERVENTION], [$lambda$ controls multiplicative shrinkage], color: ACC),
  hairline([MEASUREMENT], [validation loss chooses $lambda$; weight norm is a mechanism check], color: BLUE),
))

== Calculate one scalar update completely #D

#anchor
#v(8pt)

Given $theta_t=5$, training gradient $g_t=3$, learning rate $eta=0.1$, and decay rate $lambda_"wd"=0.2$:

#pause
$ 1-eta lambda_"wd" = 1-(0.1)(0.2)=0.98. $

#pause
$ theta_(t+1) = underbrace(0.98 times 5)_"decay" - underbrace(0.1 times 3)_"gradient" = 4.90-0.30 = 4.60. $

#pause
#result[Decay accounts for $0.10$ of the change; the data gradient accounts for $0.30$.]

== AdamW keeps decay outside the adaptive gradient transform #V

#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  card([COUPLED L2], [$theta^+ = theta - eta P_t(g + lambda theta)$
  The penalty enters adaptive moments and preconditioning.], color: MUTED),
  card([DECOUPLED ADAMW], [$theta^+ = (1-eta lambda_"wd")theta - eta P_t(g)$
  Shrinkage is applied directly to the parameter.], color: ACC),
))

#pause
#note([
  These coincide for plain SGD up to convention, but not generally for adaptive optimizers.
  Primary source: #link(ADAMW)[Loshchilov & Hutter, *Decoupled Weight Decay Regularization*].
], color: ACC)

== Honest evidence: norm is a mechanism; validation chooses the setting #V

#align(center, table(
  columns: (40mm, 58mm, 58mm, 48mm), stroke: 0.45pt + MUTED,
  inset: (x: 9pt, y: 6pt), align: center,
  table.header([$lambda_"wd"$], [validation loss], [$norm(theta)_2$], [selection]),
  [$0$], [$0.596$], [$4.921$], [reject],
  [$0.03$], [$0.456$], [$2.195$], [#text(fill: GREEN)[best validation]],
  [$0.20$], [$0.482$], [$0.928$], [reject: $0.482 > 0.456$],
))

#pause
#caption[Recorded companion ring experiment in notebook 2; it is not fabricated evidence from the five-class anchor.]

#pause
#result[A smaller norm confirms shrinkage occurred; it does *not* by itself prove better generalization.]

== Ledger update: test one weight intervention at a time

#anchor
#v(5pt)

#ledger(
  gap: [#text(fill: RED)[observed] · $27$ points; test sealed],
  checkpoint: [#text(fill: GREEN)[keep] · epoch $4$, validation loss $0.78$],
  weights: [#text(fill: ACC)[candidate] · decoupled $lambda_"wd"$; scalar audit $5 arrow.r 4.60$; select using validation],
)

#pause
#note([Do not combine a new decay rate, dropout probability, and augmentation policy in the same comparison.], color: BLUE)

// ═══════════════════════════ DROPOUT · CORE ═══════════════════════════
= Perturb the representation

== Dropout trains a sampled representation, not a smaller stored network #V

#align(center, grid(columns: (1fr, 1fr), gutter: 26pt,
  card([ONE TRAINING PASS], [Mask $m=(1,0,1,0)$ keeps two hidden coordinates.
  The next layer sees a sampled feature set.], color: ACC),
  card([EVALUATION], [The mask is disabled. All learned coordinates contribute through the complete network.], color: TEAL),
))

#pause
#v(12pt)
#align(center, grid(columns: (40mm, 8mm, 40mm, 8mm, 40mm, 8mm, 40mm), gutter: 4pt, align: horizon,
  card([$h_1=2$], [keep], color: TEAL), flow-arrow(color: MUTED),
  card([$h_2=4$], [drop], color: MUTED), flow-arrow(color: MUTED),
  card([$h_3=6$], [keep], color: TEAL), flow-arrow(color: MUTED),
  card([$h_4=8$], [drop], color: MUTED),
))

#pause
#caption[The rectangular layout shows values and mask state directly; no node or edge implies an unshown computation.]

== Inverted dropout preserves each activation in expectation #D

$ m_i tilde "Bernoulli"(p_"keep"), quad tilde(h)_i = m_i h_i / p_"keep". $

#pause
$ EE[tilde(h)_i] = EE[m_i]h_i/p_"keep" = p_"keep" h_i/p_"keep" = h_i. $

#pause
For $h=[2,4,6,8]$, $m=[1,0,1,0]$, and $p_"keep"=0.5$:

#pause
$ tilde(h) = [1 times 2,0 times 4,1 times 6,0 times 8]/0.5 = [4,0,12,0]. $

#pause
#note([One mask is not equal to the original vector; the equality is an expectation over masks.], color: ACC)

== Name the probability: PyTorch takes the drop probability #V

#align(center, table(
  columns: (46mm, 74mm, 84mm), stroke: 0.45pt + MUTED,
  inset: (x: 10pt, y: 7pt), align: (left, left, left),
  table.header([symbol / API], [meaning], [separate API example]),
  [$p_"keep"$], [probability that a coordinate survives], [$0.8$],
  [$p_"drop"=1-p_"keep"$], [probability that it is zeroed], [$0.2$],
  [`nn.Dropout(p)`], [`p` means $p_"drop"$, not $p_"keep"$], [`Dropout(p=0.2)`],
))

#pause
#codebox(size: 14pt)[```python
import torch
h = torch.tensor([2., 4., 6., 8.])
drop = torch.nn.Dropout(p=0.2)  # p_drop = 0.2, p_keep = 0.8
drop.train(); y_train = drop(h)  # random, scaled by 1/(1-p)
drop.eval();  y_eval  = drop(h)  # identity
```]

#pause
#caption[The preceding mask calculation used $p_"keep"=0.5$; unequal values here expose the API conversion.]

== Training and evaluation must disagree in exactly one controlled way #V

#align(center, table(
  columns: (46mm, 76mm, 82mm), stroke: 0.45pt + MUTED,
  inset: (x: 10pt, y: 7pt), align: (left, left, left),
  table.header([mode], [dropout], [what remains fixed]),
  [#text(fill: ACC)[`model.train()`]], [random masks + inverted scaling], [learned weights, input, loss definition],
  [#text(fill: TEAL)[`model.eval()`]], [identity mapping], [same learned weights; no manual rescaling],
))

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  hairline([EXPECTED EFFECT], [reduced reliance on any one hidden coordinate], color: ACC),
  hairline([MEASURE], [validation loss and calibration; repeat seeds if stochastic variation matters], color: BLUE),
))

#pause
#caption[Mechanism and empirical effect are separate claims. Dropout’s original analysis: #link(DROPOUT)[Srivastava et al. (2014)].]

== When is dropout the wrong first response? #Q

#mcq(
  [Which observation argues *against* increasing dropout first?],
  [Training loss is already high and validation loss is similarly high],
  [Training loss is low while validation loss rises after epoch 4],
  [The model relies on a few hidden features and validation is worse],
  [A no-dropout baseline has lower train loss but higher validation loss],
)

#pause
#note([Ask whether the model can fit before reducing its effective capacity.], color: BLUE)

== Answer: high training loss is an underfitting signal #A

#mcq-answer(
  [A],
  [training and validation losses are both high],
  [Stronger dropout makes fitting harder. First verify optimization, model capacity, features, and labels.],
)

#pause
#ledger(
  gap: [#text(fill: RED)[observed] · $27$ points; test sealed],
  checkpoint: [#text(fill: GREEN)[keep] · epoch $4$],
  weights: [#text(fill: ACC)[candidate] · decoupled decay; validate],
  dropout: [#text(fill: ACC)[candidate] · $p_"keep"=0.5$ / PyTorch $p_"drop"=0.5$; validate],
)

// ═══════════════════════ DATA + TARGETS · SHOULD ══════════════════════
= Regularize the examples and targets

== Augmentation encodes a task-specific invariance #V

For a transform $T$, the desired assumption is

#pause
$ y(T(x)) = y(x) quad "for transformations allowed by the task". $

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 24pt,
  card([VALID FOR OUR CASE], [A small translation or crop that keeps the object fully visible.], color: GREEN),
  card([INVALID FOR OUR CASE], [A crop that removes the class-defining object.], color: RED),
))

#pause
#note([The transform must preserve the label semantics, not merely look plausible to us.], color: ACC)

== Which augmentation policy is defensible? #Q

#anchor
#v(8pt)

#mcq(
  [For the five-class object classifier, which policy states a checkable label-preservation rule?],
  [Apply any transform that increases training loss],
  [Translate up to two pixels, rejecting samples where the object leaves the frame],
  [Randomly replace the class label after every crop],
  [Rotate by any angle without inspecting class semantics],
)

== Answer: constrain the transform by the task #A

#mcq-answer(
  [B],
  [translate only while the object remains in frame],
  [The policy names both the transformation and the condition under which the label is preserved.],
)

#pause
#note([A transform is an intervention. Audit a sample grid and measure validation performance before keeping it.], color: BLUE)

== Mixup trains behaviour between observed examples #V

$ tilde(x)=alpha x_i + (1-alpha)x_j, quad tilde(y)=alpha y_i+(1-alpha)y_j. $

#pause
#align(center, grid(
  columns: (50mm, 12mm, 50mm, 12mm, 50mm), gutter: 5pt, align: horizon,
  card([EXAMPLE $i$], [$x_i$, one-hot $y_i$], color: TEAL),
  flow-arrow(label: [$alpha$], color: ACC),
  card([INTERPOLATE], [inputs *and* targets], color: ACC),
  flow-arrow(label: [$1-alpha$], color: ACC),
  card([EXAMPLE $j$], [$x_j$, one-hot $y_j$], color: TEAL),
))

#pause
#note([
  Mixup’s claim is about training on convex combinations, not arbitrary semantic invariance.
  Primary source: #link(MIXUP)[Zhang et al. (2018), *mixup: Beyond Empirical Risk Minimization*].
], color: ACC)

== Calculate one five-class mixup target #D

Let $y_i=[1,0,0,0,0]$, $y_j=[0,0,0,1,0]$, and $alpha=0.25$.

#pause
$ tilde(y)=0.25y_i+0.75y_j=[0.25,0,0,0.75,0]. $

#pause
For predicted class probabilities $p$,

#pause
$ "CE"(tilde(y),p) = -0.25 log p_1 - 0.75 log p_4. $

#pause
#result[The target and loss use the *same* mixing coefficient as the input.]

== Label smoothing has two common conventions—name yours #V

For $K=5$ and $epsilon=0.1$ with class 1 correct:

#pause
#align(center, table(
  columns: (60mm, 67mm, 67mm), stroke: 0.45pt + MUTED,
  inset: (x: 9pt, y: 7pt), align: (left, center, center),
  table.header([convention], [correct class], [each wrong class]),
  [other-class redistribution], [$1-epsilon=0.90$], [$epsilon/(K-1)=0.025$],
  [PyTorch uniform mixing], [$1-epsilon+epsilon/K=0.92$], [$epsilon/K=0.02$],
))

#pause
#note([
  PyTorch `CrossEntropyLoss(label_smoothing=ε)` mixes the one-hot target with a uniform distribution over *all* $K$ classes:
  #link(PTCE)[official API definition].
], color: BLUE)

== Write both smoothed targets; they are not interchangeable #D

#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  card([OTHER-CLASS], [$[0.90,0.025,0.025,0.025,0.025]$], color: ACC),
  card([PYTORCH UNIFORM], [$[0.92,0.02,0.02,0.02,0.02]$], color: BLUE),
))

#pause
Both sum to $1$, but they assign different mass to the correct class.

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([MECHANISM], [discourages a zero-loss demand for probability exactly $1$], color: ACC),
  hairline([MEASURE], [validation NLL and calibration, alongside accuracy], color: BLUE),
))

#pause
#caption[Label smoothing was used in #link(SMOOTH)[Szegedy et al. (2016), *Rethinking the Inception Architecture for Computer Vision*].]

== Accuracy and calibration answer different questions #V

Use one validation confidence bin with $100$ examples and $72$ correct predictions.

#pause
#align(center, table(
  columns: (60mm, 56mm, 56mm), stroke: 0.45pt + MUTED,
  inset: (x: 10pt, y: 7pt), align: center,
  table.header([same predicted labels], [version A], [version B]),
  [accuracy], [$0.72$], [$0.72$],
  [mean confidence], [$0.90$], [$0.74$],
  [bin calibration gap], [$abs(0.90-0.72)=0.18$], [$abs(0.74-0.72)=0.02$],
))

#pause
#result[Same accuracy, different confidence quality. Report both when probabilities drive decisions.]

// ═══════════════════════ CUMULATIVE EXPERIMENT ════════════════════════
= Turn the diagnosis into one controlled experiment

== One semantic map: observe → intervene → select → report #V

#align(center, grid(
  columns: (48mm, 11mm, 56mm, 11mm, 56mm), gutter: 5pt, align: horizon,
  card([OBSERVE], [#text(fill: RED)[training–validation gap]
  curves and split integrity], color: RED),
  flow-arrow(color: RED),
  card([INTERVENE ON ONE SITE], [#text(fill: ACC)[checkpoint · weights · hidden features · data · targets]], color: ACC),
  flow-arrow(color: BLUE),
  card([SELECT + REPORT], [#text(fill: BLUE)[validation selects]
  #text(fill: INK)[sealed test reports once]], color: BLUE),
))

#pause
#v(12pt)
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([TRAINING PATH + PARAMETERS], [early stopping selects a checkpoint; weight decay changes parameters], color: ACC),
  hairline([HIDDEN REPRESENTATION], [dropout perturbs hidden features during training], color: ACC),
  hairline([EXAMPLES + TARGETS], [augmentation, mixup, and smoothing change supervision], color: ACC),
))


== Use symptom → suspect → test, not method → hope #V

#align(center, table(
  columns: (58mm, 68mm, 78mm), stroke: 0.45pt + MUTED,
  inset: (x: 9pt, y: 6pt), align: (left, left, left),
  table.header([symptom], [first suspect], [controlled test]),
  [train and validation loss both high], [underfit / optimization], [repair fit before adding regularization],
  [train falls; validation turns upward], [overfit with epoch], [early stop; restore validation minimum],
  [low train; high validation across epochs], [effective capacity / representation], [sweep decay *or* dropout],
  [accuracy stable; confidence too high], [target pressure / calibration], [declare smoothing convention; compare NLL],
))


== Pre-register the comparison before running it #D

#anchor
#v(6pt)

#align(center, table(
  columns: (49mm, 150mm), stroke: 0.45pt + MUTED,
  inset: (x: 10pt, y: 5pt), align: (left, left),
  [*baseline*], [same split, seed set, architecture, optimizer, and epoch budget],
  [#uncover("2-")[*one change*]], [#uncover("2-")[e.g. $lambda_"wd" in {0,0.03,0.20}$; leave dropout and augmentation fixed]],
  [#uncover("3-")[*selection*]], [#uncover("3-")[minimum validation loss; restore that checkpoint]],
  [#uncover("4-")[*secondary measures*]], [#uncover("4-")[accuracy, NLL, calibration; norm only as a mechanism audit]],
  [#uncover("5-")[*final report*]], [#uncover("5-")[evaluate the sealed test once after the recipe is frozen]],
))

#pause
#pause
#pause
#pause
#pause

== Two direct Colab labs, one prediction-first loop #I

#align(center, table(
  columns: (48mm, 152mm), stroke: 0.45pt + MUTED,
  inset: (x: 10pt, y: 8pt), align: (left, left),
  [*1 · curves*], [#link(NB1)[01_overfitting_and_early_stopping.ipynb]],
  [*2 · mechanisms*], [#link(NB2)[02_dropout_from_scratch.ipynb]],
))

#pause
#align(center, grid(columns: (1fr, 1fr, 1fr), gutter: 18pt,
  hairline([PREDICT], [write the expected curve or activation before execution], color: ACC),
  hairline([RUN], [change exactly one declared lever], color: TEAL),
  hairline([EXPLAIN], [use validation evidence to keep or reject it], color: BLUE),
))

#pause
#note([The two filenames above are the only two Colab links in this deck; both point directly to the public repository.], color: GREEN)

== Final ledger: one case, six auditable decisions

#anchor
#v(5pt)

#ledger(
  gap: [#uncover("2-")[#text(fill: RED)[observe] · train $99%$, validation $72%$, gap $27$ points; test sealed]],
  checkpoint: [#uncover("3-")[#text(fill: GREEN)[keep] · epoch $4$ at validation loss $0.78$]],
  weights: [#uncover("4-")[#text(fill: ACC)[test] · decoupled decay; scalar $5 arrow.r 4.60$; select by validation]],
  dropout: [#uncover("5-")[#text(fill: ACC)[test] · $h=[2,4,6,8] arrow.r [4,0,12,0]$ for one mask; PyTorch $p_"drop"=0.5$]],
  data: [#uncover("6-")[#text(fill: ACC)[test] · translate/crop only while the object remains and label is preserved]],
  targets: [#uncover("7-")[#text(fill: ACC)[test] · state smoothing convention; measure validation NLL and calibration]],
)

#pause
#pause
#pause
#pause
#pause
#pause
#pause

// ═════════════════════════════ CLOSE ══════════════════════════════════
= Close the loop

== Retrieval: which sequence protects the claim? #Q

#mcq(
  [Which workflow supports a defensible generalization claim?],
  [Tune on train, select on test, report validation],
  [Change several methods, keep the run with the best test accuracy],
  [Diagnose curves, change one lever, select on validation, report test once],
  [Choose the smallest parameter norm without checking predictions],
)

#pause
#note([Commit before the answer. Identify where each split appears in your choice.], color: BLUE)

== Answer: diagnosis precedes intervention #A

#mcq-answer(
  [C],
  [diagnose curves, change one lever, select on validation, report test once],
  [The workflow aligns each dataset with one job and makes the effect of the intervention interpretable.],
)

#pause
#result[Training demonstrates fit; validation chooses; the sealed test estimates the frozen procedure.]

== Revisit the opening commitment #A

#align(center, table(
  columns: (28mm, 82mm, 93mm), stroke: 0.45pt + MUTED,
  inset: (x: 9pt, y: 5pt), align: (center, left, left),
  table.header([choice], [problem], [verdict]),
  [A], [training fit is not unseen-data evidence], [reject],
  [B], [many changes are not attributable; test checking leaks], [reject],
  [C], [curves → one lever → validation → sealed test], [#text(fill: GREEN)[keep]],
  [D], [test-driven tuning destroys the estimate], [reject],
))

#pause
#result[The winning commitment was *C*: diagnose first, then run one controlled comparison.]

== Exit ticket: prescribe the smallest next test

New run: training loss falls to $0.18$; validation loss reaches $0.44$ at epoch $9$, then rises to $0.61$ by epoch $20$.

#pause
Write four lines:

1. *symptom:* what changed after epoch $9$?
2. *suspect:* which mechanism is consistent with the curves?
3. *test:* what single intervention do you try first?
4. *measurement:* which checkpoint and metric decide?

#pause
#note([Strong first test: restore epoch $9$, then compare one regularizer using validation—without consulting test.], color: GREEN)

== L8 handoff: generalization can also come from architecture #V

#align(center, grid(
  columns: (54mm, 12mm, 54mm, 12mm, 54mm), gutter: 5pt, align: horizon,
  card([L7], [External controls shape optimization, features, data, and targets.], color: ACC),
  flow-arrow(label: [next], color: BLUE),
  card([IMAGE STRUCTURE], [Nearby pixels interact; useful patterns repeat across location.], color: BLUE),
  flow-arrow(label: [encode], color: TEAL),
  card([L8 · CNNs], [Convolution builds locality and weight sharing into the architecture.], color: TEAL),
))

#pause
#result[Regularization chooses among fitting rules; architectural inductive bias changes which rules are easy to express.]

#focus-slide[
  Diagnose the gap → change one lever → select on validation → report the sealed test once.
  #v(12pt)
  #set text(size: 22pt)
  Next: convolution encodes locality and sharing as architectural inductive bias.
]

// ═════════════════════════ OPTIONAL APPENDIX · 10 MIN ═════════════════
= Implementation and provenance

== Coupled penalty and decoupled decay: same for SGD, different for Adam

#align(center, table(
  columns: (55mm, 73mm, 73mm), stroke: 0.45pt + MUTED,
  inset: (x: 9pt, y: 6pt), align: (left, left, left),
  table.header([optimizer], [coupled L2], [decoupled decay]),
  [plain SGD], [$theta^+=(1-eta lambda)theta-eta g$], [same algebra after matching convention],
  [adaptive method], [$theta^+=theta-eta P_t(g+lambda theta)$], [$theta^+=(1-eta lambda_"wd")theta-eta P_t(g)$],
))

#pause
#note([The distinction is about where the penalty enters the update, not about whether norms tend to shrink.], color: ACC)

== Parameter groups are a recipe choice, not a theorem

#codebox(size: 13pt)[```python
decay, no_decay = [], []
for name, parameter in model.named_parameters():
    target = no_decay if (name.endswith("bias") or "norm" in name) else decay
    target.append(parameter)

optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": 3e-2},
    {"params": no_decay, "weight_decay": 0.0},
], lr=1e-3)
```]

#pause
#note([Excluding biases and normalization parameters is common, but architecture- and recipe-dependent. Log the groups you actually used.], color: MUTED)

== Early stopping needs saved state, not only a stopping epoch

#codebox(size: 12.5pt)[```python
best, waits, snapshot = float("inf"), 0, None
for epoch in range(max_epochs):
    train_one_epoch(model)
    value = validation_loss(model)
    if value < best:
        best, waits = value, 0
        snapshot = copy.deepcopy(model.state_dict())
    else:
        waits += 1
    if waits >= patience:
        break
model.load_state_dict(snapshot)
```]

#pause
#note([Save optimizer state too if training will resume. For final inference, restore the selected model state.], color: BLUE)

== Calibration aggregates bin gaps; binning is part of the estimate

For bins $B_m$ with accuracy $"acc"(B_m)$ and mean confidence $"conf"(B_m)$,

#pause
$ "ECE" = sum_m (abs(B_m)/n) abs("acc"(B_m)-"conf"(B_m)). $

#pause
#align(center, grid(columns: (1fr, 1fr), gutter: 22pt,
  hairline([USE WITH CARE], [ECE changes with bin boundaries and sample size], color: MUTED),
  hairline([PAIR WITH], [NLL, accuracy, reliability plot, and task costs], color: BLUE),
))

#pause
#caption[The earlier $0.18$ and $0.02$ values were single-bin gaps, not full-dataset ECE claims.]

== Companion notebook traces: concrete, separate from the anchor

#set text(size: 15.5pt)
#align(center, table(
  columns: (38mm, 45mm, 55mm, 63mm), stroke: 0.45pt + MUTED,
  inset: (x: 8pt, y: 4.5pt), align: (left, left, left, left),
  table.header([source], [setting], [validation / agreement], [mechanism / report]),
  [notebook 1], [epoch $1640$], [MSE $0.1167$ best; $0.1756$ last], [restored test MSE $0.1111$],
  [notebook 2], [$lambda_"wd"=0$], [loss $0.596$], [norm $4.921$],
  [], [$lambda_"wd"=0.03$], [loss $0.456$], [norm $2.195$],
  [], [$lambda_"wd"=0.20$], [loss $0.482$], [norm $0.928$],
  [transform audit], [rotation], [label agreement $1.000$], [data-specific],
  [], [translation], [label agreement $0.838$], [data-specific],
))

#pause
#note([These recorded outputs motivate checks; they are not substituted for the five-class trace, and model agreement alone is not proof of label preservation.], color: INK)

== Primary references and vector provenance

#set text(size: 14.5pt)

- Zhang et al. (2017), #link(MEM)[Understanding deep learning requires rethinking generalization] — memorization and random labels.
- Loshchilov & Hutter (2019), #link(ADAMW)[Decoupled Weight Decay Regularization] — AdamW.
- Srivastava et al. (2014), #link(DROPOUT)[Dropout: A Simple Way to Prevent Neural Networks from Overfitting].
- Zhang et al. (2018), #link(MIXUP)[mixup: Beyond Empirical Risk Minimization].
- Szegedy et al. (2016), #link(SMOOTH)[Rethinking the Inception Architecture for Computer Vision] — label smoothing.
- PyTorch, #link(PTCE)[CrossEntropyLoss] — uniform-mixture `label_smoothing` convention.

#v(7pt)
#caption[All diagrams, tables, and plotted traces in this deck are native Typst vectors computed from the values shown; no raster image is embedded. Double descent is deliberately omitted rather than assigned an unsupported threshold.]
