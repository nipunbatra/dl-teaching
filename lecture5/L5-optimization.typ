// Optimization for Deep Learning — Lecture 5 · ES 667 Deep Learning
// Fresh, example-first rebuild following the mini-batch → EWMA → momentum →
// RMSProp → Adam → learning-rate-decay sequence.
// Compile from the repository root:
//   typst compile --root . --input handout=true lecture5/L5-optimization.typ slides-pdf/L5.pdf
//   typst compile --root . lecture5/L5-optimization.typ slides-pdf/L5-presentation.pdf

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *

#show: metropolis-deck.with(
  title: [Optimization for Deep Learning],
  subtitle: [Choose the data · remember direction and scale · control the step],
)

#let NB = "https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L05/"
#let WEB = "https://nipunbatra.github.io/dl-teaching/interactives/"

// Persistent visual grammar:
// BLUE = measured/sample gradient · TEAL = direction memory · GREEN = scale
// memory · ACC = applied update · RED = failure / boundary / target.
#let note(body, color: TEAL) = block(
  width: 100%, inset: (left: 10pt, right: 2pt, top: 4pt, bottom: 4pt),
  stroke: (left: 1.2pt + color), [#body],
)
#let warning(body) = note(body, color: RED)
#let punch(body) = align(center, text(size: 18pt, weight: 650)[#body])
#let caption(body) = align(center, text(size: 14.5pt, fill: MUTED)[#body])
#let evidence(body) = align(center, text(size: 9.5pt, weight: 650, fill: MUTED)[#upper(body)])
#let mini-card(title, body, color: TEAL) = block(
  width: 100%, inset: (x: 9pt, y: 7pt), radius: 3pt,
  fill: white, stroke: 0.9pt + color,
  [#text(size: 12pt, weight: 700, fill: color)[#upper(title)] #v(4pt) #body],
)
#let story-figure(name, width: 150mm, provenance: none) = {
  align(center, image("figures/" + name + ".svg", width: width))
  if provenance != none { v(2pt); evidence(provenance) }
}
#let algorithm-box(title, body, color: TEAL) = block(
  width: 100%,
  inset: (left: 15pt, right: 15pt, top: 11pt, bottom: 12pt),
  radius: 4pt,
  fill: color.lighten(94%),
  stroke: 1.3pt + color,
  [
    #text(size: 14pt, weight: 750, fill: color)[#upper(title)]
    #v(7pt)
    #set text(size: 18pt)
    #body
  ],
)
#let state-chip(label, body, color) = block(
  width: 100%, inset: (x: 9pt, y: 6pt), radius: 3pt,
  fill: color.lighten(92%), stroke: 0.8pt + color,
  [#text(size: 12pt, weight: 700, fill: color)[#label] #h(5pt) #body],
)
#let constraint-stage(label, body, color) = block(
  width: 100%, inset: (x: 4pt, y: 3pt),
  [
    #align(center, text(size: 10.5pt, weight: 750, fill: color)[#upper(label)])
    #v(4pt)
    #align(center, text(size: 16.5pt)[#body])
  ],
)
#let constraint-flow(input, transform, output) = grid(
  columns: (1fr, 22pt, 1fr, 22pt, 1fr), gutter: 7pt, align: center,
  constraint-stage([Unconstrained input], input, BLUE),
  text(size: 22pt, fill: ACC)[$arrow.r$],
  constraint-stage([Constraining function], transform, ACC),
  text(size: 22pt, fill: ACC)[$arrow.r$],
  constraint-stage([Constrained output], output, TEAL),
)

#title-slide()

== Backprop measured the slope; today we decide how to move #V

#align(center, text(size: 24pt, weight: 600)[
  $bold(theta)_t arrow.r$ forward $arrow.r cal(L)$ $arrow.r$ backward $arrow.r bold(g)_t$ $arrow.r$ optimizer $arrow.r bold(theta)_(t+1)$
])

#v(9pt)
#pause
#grid(
  columns: (1fr, 1fr), gutter: 20pt,
  mini-card([Lectures 1–4], [defined the loss, model, computation graph, and gradient $bold(g)_t$.], color: BLUE),
  mini-card([Lecture 5], [chooses the data, remembers useful history, and sets the step $Delta bold(theta)_t$.], color: ACC),
)

#pause
#result[The gradient is a measurement. An optimizer turns it into
$Delta bold(theta)_t:=bold(theta)_(t+1)-bold(theta)_t$.]

== Outline

#[
  #set text(size: 22pt)
  #enum(
    [*Choose the data per update* — full batch, one example, or a minibatch],
    [*Understand the estimate* — unbiased centre, variance, epochs, and updates],
    [*Expose a geometric failure* — why an exact gradient can still zigzag and crawl],
    [*Add useful memory* — EWMA $arrow.r$ momentum $arrow.r$ RMSProp $arrow.r$ Adam],
    [*Finish the training loop* — decay, PyTorch, and constrained parameters],
    spacing: 8pt,
  )
]

== Two examples will keep the mechanisms concrete

#grid(
  columns: (1fr, 1fr), gutter: 16pt,
  mini-card([A · four-point line fit], [
    $hat(y)=b+w x$ on $(-1,-1),(0,1),(1,3),(2,6)$.
    #v(4pt)
    *Use:* full/SGD/minibatch, unbiasedness, variance, decay.
  ], color: BLUE),
  mini-card([B · scaled-feature regression], [
    Four two-feature examples with one feature $10 times$ larger.
    #v(4pt)
    *Use:* ravines, EWMA, momentum, RMSProp, Adam.
  ], color: TEAL),
)

#v(9pt)
#note[Every contour, arrow, and trajectory is computed from a displayed objective. The notebooks reproduce the underlying calculations and seeded experiments.]

// ───────────────────────────── ACT I ─────────────────────────────
= How much data should define one update?

== What optimization means in this course

For scalar-output examples $(bold(x)_i,y_i)$ and all trainable parameters
$bold(theta)$,

$
  hat(y)_i=f_(bold(theta))(bold(x)_i), quad
  ell_i(bold(theta))=ell(hat(y)_i,y_i), quad
  cal(L)(bold(theta))=1/N sum_(i=1)^N ell_i(bold(theta)).
$

#caption[Course convention from L1–L3: bold lowercase/uppercase symbols are vectors/matrices; $y_i$, $hat(y)_i$, $ell_i$, and $cal(L)$ are scalars.]

#pause
#grid(
  columns: (1fr, auto, 1fr, auto, 1fr), gutter: 9pt, align: center + horizon,
  mini-card([forward], [evaluate predictions and a scalar loss], color: BLUE),
  text(size: 20pt)[$arrow.r$],
  mini-card([backward], [measure $nabla_(bold(theta)) cal(L)$], color: TEAL),
  text(size: 20pt)[$arrow.r$],
  mini-card([optimizer], [use that measurement to change $bold(theta)$], color: ACC),
)

#pause
#note[Today the objective stays fixed. We change which examples measure its gradient and how the optimizer transforms that measurement.]

== Revision: vanilla gradient descent is a four-line loop #D

For $N$ examples and parameter vector $bold(theta)$,

$ cal(L)(bold(theta))=1/N sum_(i=1)^N ell_i(bold(theta)). $

#pause
#align(center, {
  set text(size: 17pt)
  table(
    columns: (41mm, 1fr), stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt),
    [*evaluate*], [$cal(L)(bold(theta)_t)$],
    [*differentiate*], [$bold(g)_t=nabla_(bold(theta)) cal(L)(bold(theta)_t)$],
    [*choose $eta$*], [$eta>0$],
    [*update*], [$bold(theta)_(t+1)=bold(theta)_t-eta bold(g)_t$],
  )
})

#pause
#note[*Batch gradient descent* (also called *full-batch GD*) uses all $N$ examples for every update. “Vanilla GD” describes the update rule; the batch specifies which gradient it receives.]

== Example A: compute one full-batch gradient #D

Use $hat(y)_i=b+w x_i$ and $ell_i=1/2(hat(y)_i-y_i)^2$. Then

$ nabla_(bold(theta)) ell_i=(hat(y)_i-y_i)(1,x_i). $

#pause
At $bold(theta)_0=(b,w)=(2,-1)$, the prediction becomes
$hat(y)_i=2-x_i$:

#align(center, {
  set text(size: 15pt)
  table(
    columns: (14mm, 14mm, 44mm, 50mm, 48mm),
    stroke: 0.5pt + MUTED,
    inset: (x: 8pt, y: 5pt),
    align: (right, right, right, right, right),
    table.header(
      [$x_i$],
      [$y_i$],
      [$hat(y)_i=2-x_i$],
      [$hat(y)_i-y_i$],
      [$nabla_(bold(theta)) ell_i(bold(theta)_0)$],
    ),
    [$-1$], [$-1$], [$2-(-1)=3$], [$3-(-1)=4$], [$(4,-4)$],
    [$0$],  [$1$],  [$2-0=2$],    [$2-1=1$],    [$(1,0)$],
    [$1$],  [$3$],  [$2-1=1$],    [$1-3=-2$],   [$(-2,-2)$],
    [$2$],  [$6$],  [$2-2=0$],    [$0-6=-6$],   [$(-6,-12)$],
  )
})

#caption[Read each row left to right: predict $hat(y)_i$, form the residual $hat(y)_i-y_i$, then multiply by $(1,x_i)$ to obtain the gradient.]

== Average, then update #D

$ bold(g)_0=bold(g)_"full"
  =1/4[(4,-4)+(1,0)+(-2,-2)+(-6,-12)]
  =(-0.75,-4.5). $

#pause
#grid(
  columns: (1.18fr, 0.82fr), gutter: 12pt,
  mini-card([Parameter update], [
    #align(center, text(size: 15.5pt)[
      $bold(theta)_1=(2,-1)-0.1(-0.75,-4.5)=(2.075,-0.55)$
    ])
  ], color: ACC),
  mini-card([Produced line], [
    #align(center, text(size: 15.5pt)[$f_(bold(theta)_1)(x)=2.075-0.55x$])
  ], color: ACC),
)

#pause
#state-chip([NEW RESIDUALS], text(size: 15pt)[
  $e_(1,i)=f_(bold(theta)_1)(x_i)-y_i, quad
   bold(e)_1=(3.625,1.075,-1.475,-5.025)$
], BLUE)

#v(4pt)
#caption[Here $e_(t,i)$ denotes residual error. Because $ell_i=1/2 e_i^2$ and $N=4$, the full loss is $cal(L)(bold(theta))=1/8 norm(bold(e))_2^2$.]

#align(center, {
  set text(size: 13.5pt)
  table(
    columns: (34mm, 139mm),
    stroke: 0.5pt + MUTED,
    inset: (x: 8pt, y: 4.5pt),
    align: (left, right),
    [*before step*], [$cal(L)(bold(theta)_0)=1/8(4^2+1^2+(-2)^2+(-6)^2)=7.125$],
    [*after step*], [
      #align(right)[
        $cal(L)(bold(theta)_1)=1/8(3.625^2+1.075^2+(-1.475)^2$
        #linebreak()
        $quad +(-5.025)^2) approx 5.215$
      ]
    ],
  )
})

== One gradient step changes the fitted model #V

#story-figure("story_line_fit_revision", width: 184mm,
  provenance: [computed · the four displayed examples · notebook 00])

#note[The contour point and the fitted line are two views of the same parameters $(b,w)$.]

== Batch gradient descent is exact—but every update must wait

#align(center, grid(
  columns: (1fr, auto, 1fr, auto, 1fr), gutter: 10pt, align: center + horizon,
  mini-card([read], [$1,2,3,dots,N$], color: BLUE),
  text(size: 22pt)[$arrow.r$],
  mini-card([average], [$bold(g)_t$], color: TEAL),
  text(size: 22pt)[$arrow.r$],
  mini-card([update], [$bold(theta)_(t+1)$], color: ACC),
))

#v(7pt)
#pause
#grid(
  columns: (1fr, 1fr, 1fr), gutter: 12pt,
  mini-card([First update], [only after all $N$ examples have contributed], color: RED),
  mini-card([One epoch], [$N$ examples processed · $1$ iteration · $1$ update], color: TEAL),
  mini-card([After E epochs], [$E N$ examples processed · $E$ iterations = $E$ updates], color: ACC),
)

#pause
#note[*Work versus time:* one full-batch update processes $N$ examples. Parallel hardware changes wall-clock time, but the parameters still wait for the reduction over all $N$ contributions.]

== The training clock: examples, iterations, updates, epochs #D

#align(center, text(size: 20pt)[
  $N " examples", quad 1<=B<=N " per batch" quad arrow.r quad
   K=ceil(N/B) " iterations per epoch"$
])

#v(5pt)
#align(center, text(size: 19pt)[
  $underbrace({3,1})_"iteration 1" quad
   underbrace({4,2})_"iteration 2"$
])

#caption[Example: $N=4,B=2$. Both minibatches together process every example once—one epoch.]

#pause
#grid(
  columns: (1fr, 1fr, 1fr), gutter: 12pt,
  state-chip([EPOCH], text(size: 13pt)[one pass through $N$], BLUE),
  state-chip([ITERATION], text(size: 13pt)[one minibatch loop body], TEAL),
  state-chip([UPDATE], text(size: 13pt)[one parameter change], ACC),
)

#v(6pt)
#pause
#align(center, {
  set text(size: 12.5pt)
  table(
    columns: (32mm, 28mm, 37mm, 38mm, 38mm),
    stroke: 0.5pt + MUTED,
    inset: (x: 6pt, y: 3pt),
    align: center,
    table.header([*method*], [*$B$*], [*examples / iteration*], [*iterations / epoch*], [*updates / epoch*]),
    [full batch], [$N$], [$N$], [$1$], [$1$],
    [SGD], [$1$], [$1$], [$N$], [$N$],
    [minibatch], [$B$], [$B$], [$ceil(N/B)$], [$ceil(N/B)$],
  )
})

#v(4pt)
#caption[Normally, iteration $=$ update. After $E$ epochs: $E N$ examples and $E ceil(N/B)$ updates. Gradient accumulation or skipped steps are exceptions.]

== Full batch waits for all $N$; SGD updates from one random example #D

#grid(
  columns: (1fr, 1fr), gutter: 14pt,
  mini-card([Full-batch GD], [
    $bold(g)_t=1/N sum_(i=1)^N nabla_(bold(theta)) ell_i(bold(theta)_t)$
    #v(4pt)
    Exact for the current model—but all $N$ contributions must arrive before one update.
  ], color: RED),
  mini-card([SGD], [
    $I_t tilde "Uniform"{1,dots,N}$
    #v(2pt)
    $hat(bold(g))_t=nabla_(bold(theta)) ell_(I_t)(bold(theta)_t)$
    #v(2pt)
    $bold(theta)_(t+1)=bold(theta)_t-eta hat(bold(g))_t$
  ], color: BLUE),
)

#pause
#note[*Stochastic* means the selected index $I_t$ is random. The model and loss stay fixed; only the example used to estimate their gradient changes.]

#pause
If $I_0=2$, then $(x_2,y_2)=(0,1)$ and

#align(center, text(size: 18pt)[
  $ell_2(b,w)=1/2(b-1)^2, quad
   hat(bold(g))_0=(2-1)(1,0)=(1,0), quad
   bold(theta)_1=(1.9,-1).$
])

#caption[This step follows $ell_2$. We have not yet shown what it does to the average loss $cal(L)$.]

== Each example proposes a different update #V

#align(center, text(size: 18pt)[
  $ell_i(b,w)=1/2(b+w x_i-y_i)^2, quad
   cal(L)(b,w)=1/4 sum_(i=1)^4 ell_i(b,w).$
])

#story-figure("story_per_example_contours", width: 178mm,
  provenance: [computed · orange = sampled $I_0=2$ · muted arrows = alternatives · full batch averages all four])

== Worked check: $ell_2$ decreases, but $cal(L)$ increases #D

#story-figure("story_bad_sample_step", width: 184mm,
  provenance: [computed · orange point sampled · same four-example full loss])

#warning[The step is correct for $ell_2$: $Delta ell_2=-0.095$. Across all four examples, $sum_i Delta ell_i=+0.320$, so $Delta cal(L)=+0.320/4=+0.080$.]

== A minibatch is the useful middle ground #D

For minibatch $cal(B)_t$, let $B_t=|cal(B)_t|$ (the final batch may be smaller):

$ hat(bold(g))_t=1/B_t sum_(i in cal(B)_t) nabla_(bold(theta)) ell_i(bold(theta)_t). $

#pause
If $cal(B)_0={1,2}$ contains the first two gradients in our table,

$ hat(bold(g))_0=1/2[(4,-4)+(1,0)]=(2.5,-2). $

#pause
$ bold(theta)_1=(2,-1)-0.1(2.5,-2)=(1.75,-0.8). $

#result[Batch size is one knob: $B_t=1$ is SGD; $B_t=N$ is full-batch GD. Next: *what does averaging buy us?*]

== Why minibatches help: averaging cancels some disagreement #V

#story-figure("story_batch_paths", width: 186mm,
  provenance: [computed from Example A · fixed model, data, and parameters · only the averaging changes])

#note[One example can pull in an idiosyncratic direction. Averaging a few cancels some disagreement; averaging all examples leaves the exact full-data direction.]

== Batch, stochastic, and minibatch GD—qualitatively

#align(center, {
  set text(size: 14.5pt)
  table(
    columns: (42mm, 42mm, 42mm, 42mm), stroke: 0.5pt + MUTED,
    inset: (x: 7pt, y: 3.5pt), align: center,
    table.header([], [*Full batch*], [*SGD*], [*Minibatch*]),
    [examples / update], [$N$], [$1$], [$B$],
    [updates / epoch], [$1$], [$N$], [$ceil(N/B)$],
    [gradient noise], [none], [highest], [intermediate],
    [parallelism], [high but large], [poor], [usually efficient],
    [near optimum; fixed $eta>0$], [smooth], [typically wanders if sampling noise remains], [less wandering as $B$ grows],
  )
})

#note[In deep learning, “SGD” often informally means the whole minibatch-SGD family. The batch size states which member is actually used.]

== Choosing the batch size in PyTorch #I

#codebox(size: 13.5pt)[```python
loader = DataLoader(dataset, batch_size=B, shuffle=True)

for epoch in range(E):
    for xb, yb in loader:          # one iteration / minibatch
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = 0.5 * (pred - yb).square().mean()
        loss.backward()            # gradient of this batch mean
        optimizer.step()           # one parameter update
```]

#note[`torch.optim.SGD` transforms the gradient supplied by the loop; its name does *not* imply $B=1$. The factor $1/2$ matches our analytic half-MSE convention.]

// ───────────────────────────── ACT II ─────────────────────────────
= How can one example estimate the full gradient?

== From data rows to the full loss #D

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  gutter: 7pt, align: center + horizon,
  mini-card([row $i$], [#text(size: 15pt)[$(bold(x)_i,y_i)$]], color: BLUE),
  text(size: 19pt)[$arrow.r$],
  mini-card([predict], [#text(size: 15pt)[$hat(y)_i=f_(bold(theta))(bold(x)_i)$]], color: BLUE),
  text(size: 19pt)[$arrow.r$],
  mini-card([score row], [#text(size: 15pt)[$ell_i(bold(theta))=ell(hat(y)_i,y_i)$]], color: TEAL),
  text(size: 19pt)[$arrow.r$],
  mini-card([average], [#text(size: 14.5pt)[$cal(L)(bold(theta))=(ell_1+dots+ell_N)/N$]], color: ACC),
)

#v(10pt)
For Example A,

$ ell_i(b,w)=1/2(b+w x_i-y_i)^2, quad
   cal(L)(b,w)=1/4 sum_(i=1)^4 ell_i(b,w). $

#pause
#result[The full loss is an average of per-example losses. Therefore its gradient is the same average:
$nabla cal(L)=1/N sum_i nabla ell_i$.]

== Each example defines its own loss surface #V

#story-figure("story_individual_losses", width: 145mm,
  provenance: [computed from the four displayed Example A rows · identical axes and contour levels])

== Same parameters, different descent directions #V

#story-figure("story_individual_updates", width: 145mm,
  provenance: [computed from Example A · same $bold(theta)_0=(2,-1)$ and $eta=0.1$ in every panel])

#caption[Each arrow is the *update* $Delta bold(theta)_i=-eta nabla ell_i(bold(theta)_0)$, so it points downhill on that panel's loss.]

== A minibatch averages the selected loss surfaces #V

#story-figure("story_batch_loss_surfaces", width: 180mm,
  provenance: [computed from Example A · row $2$, rows $1$–$2$, and all four rows])

#v(3pt)
$
 cal(L)_cal(B)(bold(theta))=1/B sum_(i in cal(B)) ell_i(bold(theta))
 quad arrow.r quad
 nabla cal(L)_cal(B)=1/B sum_(i in cal(B)) nabla ell_i.
$

#caption[Increasing $B$ averages more troughs. Their disagreements partly cancel, and the full batch leaves one exact empirical-loss surface.]

== Average the one-example updates #V

#grid(
  columns: (1.25fr, 0.75fr), gutter: 12pt, align: center + horizon,
  story-figure("story_update_vector_average", width: 102mm,
    provenance: [computed from Example A at fixed $bold(theta)_0$]),
  [
    #set text(size: 16pt)
    Translate the four updates to a common origin, then average coordinates:
    #v(8pt)
    $
      1/4 sum_(i=1)^4 Delta bold(theta)_i
      =(0.075,0.450).
    $
    #v(8pt)
    $
      -eta/4 sum_(i=1)^4 nabla ell_i(bold(theta)_0)
      = Delta bold(theta)_"full".
    $
    #v(8pt)
    #note[Different samples give different updates; their arithmetic mean is the full-batch update.]
  ],
)

== An estimator maps random data to an estimate #D

For a generic discrete random variable $X$ with probability mass $p(x)$,
expectation is the probability-weighted average

$ EE[f(X)] = sum_(x in cal(X)) p(x) f(x). $

#v(7pt)
#grid(
  columns: (1fr, 1fr, 1fr), gutter: 13pt,
  mini-card([Random draw], [#text(size: 14.5pt)[$X=x$ occurs with probability $p(x)$]], color: BLUE),
  mini-card([Estimate], [#text(size: 14.5pt)[apply the rule $f$ and observe $f(X)$]], color: TEAL),
  mini-card([Target], [#text(size: 14.5pt)[a fixed quantity $mu$ to be estimated]], color: ACC),
)

#v(7pt)
#align(center, text(size: 17pt)[
  The estimator $f(X)$ is *unbiased* for $mu$ when $EE[f(X)]=mu$.
])

#v(6pt)
#align(center, text(size: 13.5pt, fill: MUTED)[
  For continuous $X$, replace the sum by an integral. Unbiasedness describes the centre across repeated draws—not one draw.
])

== Why uniformly sampled SGD is unbiased #D

Hold $bold(theta)_t$ fixed and substitute into the generic expectation:

$ X arrow.r I_t, quad
  p(i)=PP(I_t=i | bold(theta)_t)=1/N, quad
  f(i)=nabla_(bold(theta)) ell_i(bold(theta)_t). $

The stochastic gradient estimator is the gradient of the sampled loss:

$ hat(bold(g))_t=f(I_t)
  =nabla_(bold(theta)) ell_(I_t)(bold(theta)_t). $

$
  EE[hat(bold(g))_t | bold(theta)_t]
  = sum_(i=1)^N p(i) f(i)
  = sum_(i=1)^N 1/N nabla_(bold(theta)) ell_i(bold(theta)_t)
  = nabla [1/N sum_(i=1)^N ell_i(bold(theta)_t)]
  = bold(g)_t.
$

#result[For Example A, the mean of $(4,-4),(1,0),(-2,-2),(-6,-12)$ is $(-0.75,-4.5)$—the full gradient computed earlier.]

== Minibatches preserve the mean and reduce the spread #V

#align(center, text(size: 17pt)[
  A minibatch estimator is simply an *average of one-row estimators*:
  $hat(bold(g))_(B,t)=1/B sum_(i in cal(B)_t) nabla ell_i(bold(theta)_t)$.
])

#story-figure("story_estimator_clouds", width: 180mm)
#evidence[exact enumeration of every uniform Example A subset at the same $bold(theta)_0$]

#v(3pt)
#align(center, text(size: 15.5pt, weight: 650)[
  *Same centre:* $EE[hat(bold(g))_(B,t) | bold(theta)_t]=bold(g)_t$.
  #h(14pt) *Less spread:* larger $B$ cancels more disagreement.
])

== Unbiased means correct on average—not downhill every time #D

Our earlier valid SGD draw sampled row $2$ and used
$hat(bold(g))_0=(1,0)$, so $bold(theta)_1=(1.9,-1)$.

#grid(
  columns: (1fr, 1fr), gutter: 16pt,
  mini-card([Sampled loss], [
    $ell_2: 0.500 arrow.r 0.405$
    #v(3pt)
    The update succeeds on the sampled row.
  ], color: TEAL),
  mini-card([Full loss], [
    $cal(L): 7.125 arrow.r 7.205$
    #v(3pt)
    The same update is uphill for the average loss.
  ], color: RED),
)

#v(9pt)
#warning[Unbiasedness guarantees $EE[hat(bold(g))_t | bold(theta)_t]=bold(g)_t$—equivalently, $EE[Delta bold(theta)_t | bold(theta)_t]=-eta bold(g)_t$. It does not make every estimate close, every step downhill, or convergence automatic.]

== Notebook 00: reproduce the surfaces, arrows, and averages #I

#grid(
  columns: (1fr, 1fr, 1fr), gutter: 12pt,
  mini-card([1 · inspect], [Plot each $ell_i$, a minibatch average, and the full $cal(L)$.], color: BLUE),
  mini-card([2 · verify], [Enumerate the possible gradients and check their mean.], color: TEAL),
  mini-card([3 · experiment], [Change $B$ and watch the estimator cloud contract.], color: ACC),
)

#v(10pt)
#interbox(link-to: NB + "00_full_batch_sgd_minibatch.ipynb")[
  Open *Full batch, SGD, minibatches, and gradient estimators* in Colab ↗
]

// ───────────────────────────── ACT III ─────────────────────────────
= An exact gradient can still be slow

== Example B: one feature is ten times larger #D

#grid(
  columns: (1.05fr, 0.95fr), gutter: 18pt,
  [
    Use $hat(y)_i=theta_1 x_(i 1)+theta_2 x_(i 2)$ and generate targets with
    $bold(theta)^*=(1,-0.5)$.

    #v(7pt)
    The first feature is $plus.minus 1$; the second is $plus.minus 10$.
    This deliberate scale difference creates a transparent optimization problem.

  ],
  [
    #align(center, table(
      columns: (17mm, 25mm, 25mm, 22mm), stroke: 0.5pt + MUTED,
      inset: (x: 7pt, y: 5pt), align: center,
      table.header([$i$], [$x_(i 1)$], [$x_(i 2)$], [$y_i$]),
      [$1$], [$-1$], [$-10$], [$4$],
      [$2$], [$-1$], [$10$], [$-6$],
      [$3$], [$1$], [$-10$], [$6$],
      [$4$], [$1$], [$10$], [$-4$],
    ))
  ],
)

#result[Gradient descent starts at $bold(theta)_0=(-1.4,0.35)$ and moves toward the optimum $bold(theta)^*=(1,-0.5)$.]

== A ravine is a long, narrow valley with steep walls #V

A *valley* is low ground between higher sides. A *ravine* is a long, narrow
valley whose side walls are steep.

#v(4pt)
#align(center, image(
  "figures/ravine_terrain_intuition.png",
  width: 162mm,
  height: 65mm,
  fit: "cover",
))

#v(3pt)
#grid(
  columns: (1fr, 1fr), gutter: 24pt,
  [
    #text(size: 14.5pt)[#text(weight: 700, fill: BLUE)[Along the valley]
    follows the low floor; height changes gently.]
  ],
  [
    #text(size: 14.5pt)[#text(weight: 700, fill: ACC)[Across the valley]
    crosses the walls; height changes quickly.]
  ],
)

#evidence[Illustrative terrain analogy · generated for this lecture]

== The $10 times$ feature turns the loss into a ravine #V

Averaging the four rows gives an exact full-batch loss:

#align(center, $
  cal(L)(bold(theta))
  =1/2 (theta_1-1)^2+50(theta_2+0.5)^2,
  quad
  nabla_theta cal(L)(bold(theta))
  =(theta_1-1,100(theta_2+0.5)).
$)

#story-figure("story_ravine_surface", width: 152mm,
  provenance: [computed from the displayed full-batch loss · native parameter coordinates])

== Contours show the same surface from above #V

#story-figure("story_ravine_geometry", width: 132mm,
  provenance: [computed from the four displayed regression examples · exact full-batch loss])

#note[Each contour joins parameter settings with the same loss. Widely spaced contours mean gentle change; tightly packed contours mean steep change.]

== One correct step can overshoot across the valley #D

#grid(
  columns: (0.78fr, 1.22fr), gutter: 18pt, align: center + horizon,
  [
    At $bold(theta)_0=(-1.4,.35)$,
    $ bold(g)_0=nabla_(bold(theta)) cal(L)(bold(theta)_0) $
    $ =(-1.4-1,100(.35+.5))=(-2.4,85). $

    #v(5pt)
    With $eta=.0195$,
    $ Delta bold(theta)_0=-eta bold(g)_0 $
    $ =-.0195(-2.4,85)=(.0468,-1.6575). $

    $ bold(theta)_1=bold(theta)_0+Delta bold(theta)_0=(-1.3532,-1.3075). $

    #caption[$bold(g)_0$ points uphill; $-bold(g)_0$ moves downhill and crosses $theta_2^*=-.5$.]
  ],
  story-figure("story_ravine_first_step", width: 111mm,
    provenance: [computed exact-gradient update]),
)

== The trade-off is visible in the computed paths #V

#story-figure("story_ravine_learning_rates", width: 166mm,
  provenance: [computed full-batch GD · same data, start, and update budget])

#note[The small rate avoids crossing $theta_2^*=-0.5$, but barely changes
$theta_1$. The larger rate moves $theta_1$ faster and repeatedly crosses the
optimum in $theta_2$. One global learning rate creates this trade-off.]

// ───────────────────────────── ACT IV ─────────────────────────────
= How can we smooth a noisy sequence?

== Daily temperature readings are noisy #I

#story-figure("story_ewma_raw", width: 166mm,
  provenance: [constructed 120-day temperature signal · fixed seed 12])

#v(6pt)
#note[A single day may be unusually warm or cool. To follow the slower trend, we keep one running average and update it as each new reading arrives.]

#v(5pt)
#align(center, text(size: 13.5pt, weight: 650, fill: rgb("#1B7A34"))[
  #link(WEB + "ewma-explorer.html")[Open the EWMA + Momentum Explorer ↗ — change
  $beta$ and watch the smoothed curve.]
])

== Short memory reacts quickly—and follows the noise #V

#story-figure("story_ewma_beta05", width: 166mm,
  provenance: [same readings · EWMA initialized at the first reading])

#note[$beta$ is the fraction of the previous summary retained. At $beta=.5$, the previous summary and today's reading receive equal weight, so the curve remains jumpy.]

== More memory gives a useful compromise #V

#story-figure("story_ewma_beta09", width: 166mm,
  provenance: [same readings · only $beta$ changes])

#note[At $beta=.9$, keep $90%$ of the previous summary and add $10%$ of today's reading. Noise is reduced, with some delay when the trend turns.]

== Too much memory becomes slow to respond #V

#story-figure("story_ewma_beta098", width: 166mm,
  provenance: [same readings · only $beta$ changes])

#note[$beta=.98$ averages over a much longer effective history. It is very smooth, but it visibly lags the changing signal.]

== EWMA keeps one running memory $m_t$ #D

#align(center, $
  underbrace(m_t, "running memory after today's reading")
      =underbrace(beta m_(t-1), "keep the previous summary")
      +underbrace((1-beta)x_t, "add today's observation"),
  quad 0 <= beta < 1.
$)

#v(10pt)
#grid(
  columns: (1fr, 1fr), gutter: 18pt,
  mini-card([One worked update], [
    $m_(t-1)=30 degree C$, $x_t=36 degree C$, $beta=.9$
    #v(4pt)
    $m_t=.9(30)+.1(36)=30.6 degree C$
  ], color: ACC),
  mini-card([What is carried forward?], [
    Keep the new memory $m_t=30.6$ for the next observation.
    The earlier readings need not be stored.
  ], color: TEAL),
)

#v(9pt)
#result[This signal summary is an *exponentially weighted moving average*
(EWMA). Optimizer momentum uses the same normalized memory idea; its complete
update rule is boxed before we substitute any values. PyTorch's differently
scaled buffer is discussed only after the derivation.]

== EWMA weights come from repeated substitution #D

*1. Write the current update.*
#align(center, $m_t=beta m_(t-1)+(1-beta)x_t.$)

*2. Write the same update one step earlier.*
#align(center, $m_(t-1)=beta m_(t-2)+(1-beta)x_(t-1).$)

*3. Substitute this expression for $m_(t-1)$.*
#align(center, $m_t
  = beta [beta m_(t-2)+(1-beta)x_(t-1)] +(1-beta)x_t$)

#align(center, $m_t
  = (1-beta)x_t +(1-beta)beta x_(t-1)+beta^2m_(t-2).$)

#note[Each step backwards adds one factor of $beta$. For the zero-start
recurrence, $x_(t-k)$ has weight $(1-beta)beta^k$. For $beta=.9$, these weights
begin $.100,.090,.081,.0729,dots$. In the plotted warm start, $x_1$ instead
carries the remaining weight $beta^(t-1)$.]

== Starting from zero makes the first temperature estimate too small #V

#story-figure("story_ewma_bias", width: 155mm,
  provenance: [same fixed-seed temperature readings · $beta=.9$])

#note[With the arbitrary initial state $m_0=0 degree C$ and the same first
reading $x_1=17.968 degree C$,
$m_1=.9(0)+.1(17.968)=1.7968 degree C approx 1.80 degree C$. The remaining
$.9$ weight came from the chosen initial zero—not from a temperature
observation.]

== Divide by the weight that has arrived #D

#align(center, $
  underbrace(w_t=(1-beta)sum_(k=0)^(t-1) beta^k=1-beta^t,
    "weight from observations 1 through t")
  \
  underbrace(hat(m)_t=m_t/w_t=m_t/(1-beta^t),
    "normalize the raw zero-start memory")
$)

#v(8pt)
#result[Day 1: $hat(m)_1=1.7968/(1-.9)=17.968 degree C approx 17.97 degree C$.]

#note[The earlier temperature curves set $m_1=x_1$, so they did not have this
cold start. Dividing by the weight that has arrived removes the artificial pull
toward zero; it does not remove measurement noise.]

// ───────────────────────────── ACT V ─────────────────────────────
== Momentum asks one question #V

#align(center, text(size: 30pt, weight: 700, fill: TEAL)[
  Is this direction consistent over time?
])

#v(2pt)
#align(center, image("figures/optimizer-intuition-momentum-v1.png", width: 82mm))
#v(2pt)
#evidence[conceptual illustration · generated with OpenAI Imagen · not data]
#v(3pt)
#note[#text(size: 14.5pt)[Momentum averages *signed* gradients. Components that
keep the same sign survive in the memory; components that repeatedly reverse
partly cancel. The result keeps persistent motion while damping the side-to-side
zigzag.]]

== Momentum: complete update rule #D

#algorithm-box([Gradient descent with momentum], [
  $hat(bold(g))_t=nabla_(bold(theta)) cal(L)_(cal(B)_t)(bold(theta)_t)$

  $bold(m)_(t+1)=beta bold(m)_t+(1-beta)hat(bold(g))_t,
    quad bold(m)_0=bold(0)$

  $bold(theta)_(t+1)=bold(theta)_t-eta bold(m)_(t+1)$
], color: TEAL)

#v(9pt)
#note[$hat(bold(g))_t$ is the gradient measured on the current full batch or
minibatch. $bold(m)_t$ is an EWMA of present and past gradients: $beta$ keeps
the old memory and $1-beta$ weights the new gradient, exactly as in the
temperature example. $eta$ controls the move. When $beta=0$, momentum reduces
to ordinary gradient descent.]

== First update: the same gradient, two update rules #D

At $bold(theta)_0=(-1.4,.35)$, both methods start from
$cal(L)(bold(theta)_0)=39.005$ and measure
#align(center, $bold(g)_0=(theta_(0,1)-1,100(theta_(0,2)+.5))=(-2.4,85).$)

#v(4pt)
#grid(
  columns: (1fr, 1fr), gutter: 14pt,
  mini-card([Plain gradient descent], [
    $Delta bold(theta)_0=-eta bold(g)_0$
    $=-.0195(-2.4,85)=(.0468,-1.6575)$
    #v(4pt)
    $bold(theta)_1=bold(theta)_0+Delta bold(theta)_0$
    $=(-1.4,.35)+(.0468,-1.6575)$
    $=(-1.3532,-1.3075)$
    #v(4pt)
    $cal(L)(bold(theta)_1)=frac(1,2)(-2.3532)^2+50(-.8075)^2$
    $approx 35.372$
  ], color: ACC),
  mini-card([Gradient descent with momentum], [
    $bold(m)_1=.25bold(m)_0+.75bold(g)_0$
    $=.25(0,0)+.75(-2.4,85)=(-1.8,63.75)$
    #v(4pt)
    $Delta bold(theta)_0=-eta bold(m)_1=(.0351,-1.243125)$
    #v(4pt)
    $bold(theta)_1=(-1.4,.35)+(.0351,-1.243125)$
    $=(-1.3649,-.893125)$
    #v(4pt)
    $cal(L)(bold(theta)_1)=frac(1,2)(-2.3649)^2+50(-.393125)^2$
    $approx 10.524$
  ], color: TEAL),
)

== Updates 2 and 3: repeat the same three equations #D

Each method first measures a new gradient at *its own* current parameters.

#v(4pt)
#align(center, {
  set text(size: 12.4pt)
  table(
    columns: (16mm, 1fr, 1fr),
    stroke: 0.5pt + MUTED,
    inset: (x: 7pt, y: 3.4pt),
    align: (center, left, left),
    table.header([*update*], [*plain GD*], [*momentum*]),
    [*2*],
    [$bold(g)_1=(-2.3532,-80.75)$ \
     $Delta bold(theta)=-.0195bold(g)_1=(.0459,1.5746)$ \
     $bold(theta)_2=(-1.3073,.2671)$ \
     $cal(L)_2=32.086$],
    [$bold(g)_1=(-2.3649,-39.3125)$ \
     $bold(m)_2=.25bold(m)_1+.75bold(g)_1=(-2.2237,-13.5469)$ \
     $Delta bold(theta)=-.0195bold(m)_2=(.0434,.2642)$ \
     $bold(theta)_2=(-1.3215,-.6290), quad cal(L)_2=3.526$],
    [*3*],
    [$bold(g)_2=(-2.3073,76.7125)$ \
     $Delta bold(theta)=-.0195bold(g)_2=(.0450,-1.4959)$ \
     $bold(theta)_3=(-1.2623,-1.2288)$ \
     $cal(L)_3=29.114$],
    [$bold(g)_2=(-2.3215,-12.8961)$ \
     $bold(m)_3=.25bold(m)_2+.75bold(g)_2=(-2.2971,-13.0588)$ \
     $Delta bold(theta)=-.0195bold(m)_3=(.0448,.2546)$ \
     $bold(theta)_3=(-1.2767,-.3743), quad cal(L)_3=3.382$],
  )
})

== Updates 4–6: the same learning rate, different moves #D

Each method recomputes $bold(g)_t$ at its own current $bold(theta)_t$.

#v(6pt)
#align(center, {
  set text(size: 12.7pt)
  table(
    columns: (16mm, 1fr, 1fr),
    stroke: 0.5pt + MUTED,
    inset: (x: 7pt, y: 5pt),
    align: (center, left, left),
    table.header([*update*], [*plain GD*], [*momentum*]),
    [*4*],
    [$Delta bold(theta)=(.0441,1.4211)$ \
     $bold(theta)_4=(-1.2182,.1923)$ \
     $cal(L)_4=26.426$],
    [$bold(m)_4=(-2.2818,6.1617)$ \
     $Delta bold(theta)=(.0445,-.1202)$ \
     $bold(theta)_4=(-1.2322,-.4945), quad cal(L)_4=2.493$],
    [*5*],
    [$Delta bold(theta)=(.0433,-1.3500)$ \
     $bold(theta)_5=(-1.1749,-1.1577)$ \
     $cal(L)_5=23.995$],
    [$bold(m)_5=(-2.2446,1.9553)$ \
     $Delta bold(theta)=(.0438,-.0381)$ \
     $bold(theta)_5=(-1.1885,-.5326), quad cal(L)_5=2.448$],
    [*6*],
    [$Delta bold(theta)=(.0424,1.2825)$ \
     $bold(theta)_6=(-1.1325,.1248)$ \
     $cal(L)_6=21.794$],
    [$bold(m)_6=(-2.2025,-1.9559)$ \
     $Delta bold(theta)=(.0429,.0381)$ \
     $bold(theta)_6=(-1.1455,-.4945), quad cal(L)_6=2.303$],
  )
})

== The same learning rate produces very different paths #V

#story-figure("story_gradient_filter", width: 180mm,
  provenance: [computed on Example B · points 0–6 · exact full-batch gradients])

#note[Both use $eta=.0195$. Plain GD follows each current sign reversal;
momentum averages it with recent gradients before moving.]

== Momentum update 2 in $theta_1$: both pushes point right #D

At momentum's $bold(theta)_1$, the first coordinate has memory
$m_(1,1)=-1.8$ and new gradient $g_(1,1)=-2.3649$. Expand the two contributions
inside $bold(m)_2=.25bold(m)_1+.75bold(g)_1$:

#v(8pt)
#grid(
  columns: (1fr, 1fr), gutter: 18pt,
  mini-card([Stored history], [
    $-.0195[.25(-1.8)] = +.0088$
    #v(7pt)
    #align(center, text(size: 24pt, fill: BLUE)[$+.0088$ →])
  ], color: BLUE),
  mini-card([New gradient], [
    $-.0195[.75(-2.3649)] = +.0346$
    #v(7pt)
    #align(center, text(size: 24pt, fill: TEAL)[$+.0346$ →])
  ], color: TEAL),
)

#v(10pt)
#align(center, text(size: 23pt, weight: 700)[
  $+.0088 + .0346 = +.0434$ →
])
#align(center, text(size: 18pt)[
  first coordinate: $-1.3649 + .0434 = -1.3215$
])

#note[The old and new gradients have the same sign, so their contributions add.
The memory keeps a persistent rightward direction.]

== Update 2 in $theta_2$: the old and new pushes nearly cancel #D

For the second coordinate, the stored memory is $m_(1,2)=63.75$, while the new
gradient has reversed sign: $g_(1,2)=-39.3125$.

#v(8pt)
#grid(
  columns: (1fr, 1fr), gutter: 18pt,
  mini-card([Stored history], [
    $-.0195[.25(63.75)] = -.3108$
    #v(7pt)
    #align(center, text(size: 24pt, fill: BLUE)[$-.3108$ ↓])
  ], color: BLUE),
  mini-card([New gradient], [
    $-.0195[.75(-39.3125)] = +.5750$
    #v(7pt)
    #align(center, text(size: 24pt, fill: ACC)[$+.5750$ ↑])
  ], color: ACC),
)

#v(10pt)
#align(center, text(size: 23pt, weight: 700)[
  $-.3108 + .5750 = +.2642$ ↑
])
#align(center, text(size: 18pt)[
  second coordinate: $-.8931 + .2642 = -.6290$
])

#note[The pushes oppose one another, so most of the reversal cancels. Plain GD
uses only its current gradient; momentum tempers a reversal using recent
history.]

== Momentum damps the steep-direction oscillation #V

#story-figure("story_momentum_ravine", width: 169mm,
  provenance: [computed · same start and same $eta=.0195$ · 55 exact full-batch gradient evaluations each])

#note[Both methods use $eta=.0195$. Plain GD crosses the valley on every shown
update; momentum reduces those crossings while retaining steady motion in
$theta_1$.]

== Both losses fall; momentum remains lower #V

#story-figure("story_momentum_loss", width: 166mm,
  provenance: [computed on Example B · identical gradient-evaluation budget])

#note[Both curves use exact full-batch gradients, the same $eta=.0195$, and both
losses decrease at every displayed update. After 55 updates the losses are
approximately $.330$ for momentum and $.458$ for plain GD.]

== Momentum is GD plus one running memory #I

#grid(
  columns: (1fr, 1fr), gutter: 18pt,
  [
    #text(size: 15.5pt, weight: 700, fill: ACC)[PLAIN GRADIENT DESCENT]
    #v(5pt)
    #codebox(size: 12.8pt)[```python
def gd_step(theta, grad, eta):
    return theta - eta * grad
```]
    #v(5pt)
    #caption[Return the updated parameters.]
  ],
  [
    #text(size: 15.5pt, weight: 700, fill: TEAL)[GRADIENT DESCENT WITH MOMENTUM]
    #v(5pt)
    #codebox(size: 12.8pt)[```python
def momentum_step(theta, grad, m, eta, beta):
    m = beta * m + (1 - beta) * grad
    theta = theta - eta * m
    return theta, m
```]
    #v(5pt)
    #caption[Return the parameters and the memory.]
  ],
)

#v(7pt)
#note[These functions assume vectorized arrays or tensors. Compute `grad` at the
current `theta`. For momentum, initialize `m = zeros_like(theta)` once and pass
the returned $bold(m)$ into the next update.]

== PyTorch momentum uses the same training loop #I

#grid(
  columns: (1.05fr, 0.95fr), gutter: 17pt,
  [
    #codebox(size: 13.2pt)[```python
beta, eta = 0.25, 0.0195
opt = torch.optim.SGD(model.parameters(),
                      lr=(1-beta)*eta,
                      momentum=beta)

opt.zero_grad(set_to_none=True)
loss.backward()
opt.step()
```]
    #v(5pt)
    #text(size: 13.7pt)[PyTorch stores the rescaled buffer
    $tilde(bold(m))_(t+1)=beta tilde(bold(m))_t+hat(bold(g))_t$.
    Since $bold(m)=(1-beta)tilde(bold(m))$, the boxed normalized rule maps to
    `lr` $=(1-beta)eta$. This is the same conceptual $eta$, expressed for
    PyTorch's scaled state—not a separately tuned rate.]
  ],
  [
    #grid(
      columns: (1fr, 1fr), gutter: 9pt,
      mini-card([Signal], [Compare $beta=.25,.5,.9$.], color: ACC),
      mini-card([Optimizer], [Reproduce the first six updates.], color: TEAL),
    )
    #v(10pt)
    #interbox(link-to: NB + "02_ewma_momentum.ipynb")[
      Open *Ravines, EWMA, and momentum* in Colab ↗
    ]
  ],
)

== Could feature standardization fix this example? Yes. #V

#story-figure("story_feature_scaling_fix", width: 155mm,
  provenance: [computed before/after geometry for $z_2=x_2/10$])

#result[Here standardizing $x_2$ removes the $100:1$ scale imbalance. With a
stable larger rate, plain GD moves directly toward the optimum—momentum is not
needed for this constructed problem.]

== Four rows: standardized, but still correlated #D

Use $hat(y)_i=theta_1 x_(i,1)+theta_2 x_(i,2)$, choose
$bold(theta)^*=(1,0)$, and set $y_i=x_(i,1)$:

#grid(
  columns: (0.88fr, 1.12fr), gutter: 22pt, align: top,
  [
    #align(center, {
      set text(size: 15pt)
      table(
        columns: (13mm, 24mm, 24mm, 21mm),
        stroke: 0.5pt + MUTED,
        inset: (x: 7pt, y: 5pt),
        align: (center, right, right, right),
        table.header([$i$], [$x_(i,1)$], [$x_(i,2)$], [$y_i$]),
        [$1$], [$-1$], [$-7/5$], [$-1$],
        [$2$], [$-1$], [$-1/5$], [$-1$],
        [$3$], [$1$],  [$1/5$],  [$1$],
        [$4$], [$1$],  [$7/5$],  [$1$],
      )
    })
  ],
  [
    #text(size: 17pt)[
      $1/4 sum_i x_(i,1)=1/4 sum_i x_(i,2)=0$

      $1/4 sum_i x_(i,1)^2=1/4 sum_i x_(i,2)^2=1$

      $"corr"(x_1,x_2)=1/4 sum_i x_(i,1)x_(i,2)$

      $=1/4(7/5+1/5+1/5+7/5)=4/5$
    ]
  ],
)

#v(7pt)
#note[Both columns have training-set mean $0$ and variance $1$, so both are
standardized. Their correlation is nevertheless $.8$: when $x_1$ is large,
$x_2$ usually moves with it.]

== Equal-sized moves change this loss by a factor of nine #D

Let $bold(e)=bold(theta)-bold(theta)^*$. Because $bold(y)=X bold(theta)^*$,

$
  cal(L)(bold(theta))=1/(2N)norm(X bold(theta)-bold(y))_2^2
  =1/2 bold(e)^T underbrace((X^T X)/4, bold(H)) bold(e),
  quad
  bold(H)=mat(1,4/5;4/5,1).
$

#v(7pt)
#grid(
  columns: (1fr, 1fr), gutter: 22pt,
  [
    #text(size: 14pt, weight: 700, fill: ACC)[MOVE TOGETHER · $(a,a)$]
    #v(5pt)
    #align(center, text(size: 20pt)[$cal(L)=9/5 a^2$])
    #caption[Steep direction: the two coefficient changes reinforce.]
  ],
  [
    #text(size: 14pt, weight: 700, fill: BLUE)[TRADE OFF · $(a,-a)$]
    #v(5pt)
    #align(center, text(size: 20pt)[$cal(L)=1/5 a^2$])
    #caption[Shallow direction: one coefficient compensates for the other.]
  ],
)

#v(7pt)
#note[The two moves have the same Euclidean length $sqrt(2)abs(a)$, but the
first changes the loss nine times as much. That uneven curvature is the ravine.]

== The 9× curvature difference produces a tilted ravine #V

#story-figure("story_standardized_correlated_geometry", width: 191mm,
  provenance: [computed exactly from the four displayed rows · no sampling noise])

== The same four rows confirm that momentum can help #V

#story-figure("story_standardized_correlated_paths", width: 184mm,
  provenance: [computed exact full-batch updates · same start and same learning rate])

// ───────────────────────────── ACT VI ─────────────────────────────
= A global rate forces a speed–stability trade-off

== RMSProp asks a different question #V

#align(center, text(size: 30pt, weight: 700, fill: GREEN)[
  Is this component large relative to its usual size?
])

#v(2pt)
#align(center, image("figures/optimizer-intuition-rmsprop-v1.png", width: 82mm))
#v(2pt)
#evidence[conceptual illustration · generated with OpenAI Imagen · not data]
#v(3pt)
#note[#text(size: 14.5pt)[RMSProp averages *squared* gradients, so signs cannot
cancel, then divides each component by its own recent RMS scale. A persistently
large component gets a larger denominator and no longer forces every coordinate
to use a tiny global learning rate.]]

== Example B: recall the data and compute the first gradient #D

#grid(
  columns: (0.92fr, 1.08fr), gutter: 18pt,
  [
    #text(size: 15.5pt)[Use $hat(y)_i=theta_1 x_(i 1)+theta_2 x_(i 2)$. The targets are generated by $bold(theta)^*=(1,-.5)$.]
    #v(7pt)
    #align(center, {
      set text(size: 13.5pt)
      table(
        columns: (14mm, 22mm, 22mm, 19mm), stroke: 0.5pt + MUTED,
        inset: (x: 6pt, y: 4pt), align: center,
        table.header([$i$], [$x_(i 1)$], [$x_(i 2)$], [$y_i$]),
        [$1$], [$-1$], [$-10$], [$4$],
        [$2$], [$-1$], [$10$], [$-6$],
        [$3$], [$1$], [$-10$], [$6$],
        [$4$], [$1$], [$10$], [$-4$],
      )
    })
    #v(7pt)
    #align(center, text(size: 16pt)[$bold(theta)_0=(-1.4,.35)$])
  ],
  [
    #text(size: 15.5pt)[Averaging the four half-squared errors gives]
    #v(4pt)
    #align(center, $cal(L)(bold(theta))=frac(1,2)(theta_1-1)^2+50(theta_2+.5)^2$)
    #v(2pt)
    #align(center, $nabla_theta cal(L)(bold(theta))=(theta_1-1,100(theta_2+.5)).$)
    #v(6pt)
    $bold(g)_0=(-1.4-1,100(.35+.5))=(-2.4,85).$
    #v(6pt)
    #align(center, text(size: 19pt, weight: 700)[
      $frac(abs(g_(0,2)),abs(g_(0,1)))=frac(85,2.4) approx 35.4.$
    ])
    #v(5pt)
    With $eta=.0195$,
    $abs(Delta theta_(0,1))=.0195(2.4)=.0468,$
    $abs(Delta theta_(0,2))=.0195(85)=1.6575.$
    #v(3pt)
    #align(center, text(size: 18pt, weight: 700)[
      $frac(1.6575,.0468) approx 35.4.$
    ])
  ],
)

== One learning rate scales both gradient components equally #V

#story-figure("story_rmsprop_gradient_sizes", width: 164mm,
  provenance: [computed from Example B · $bold(g)_0=(-2.4,85)$])

#note[The preceding slide calculated these two bars from the four data rows.
Multiplying both components by the same $eta$ changes their lengths but not their
$35.4 times$ ratio. A small rate is stable but slow in $theta_1$; a larger rate
speeds up $theta_1$ but risks oscillation in $theta_2$.]

== Recent magnitude should control each component's step #V

#story-figure("story_rmsprop_why_square", width: 176mm,
  provenance: [computed from the first five exact gradients on Example B])

#note[The sign of the steep component keeps reversing, so an ordinary average can
cancel toward zero. Squaring removes the sign and preserves the evidence that this
component has been large.]

== RMSProp uses a recent root-mean-square scale #D

#set text(size: 19pt)
#enum(
  [*Square:* use $hat(bold(g))_t^2$ so positive and negative magnitudes do not cancel.],
  [*Mean:* call the exponentially weighted mean of those squares $bold(v)_(t+1)$.],
  [*Root:* use $bold(r)_(t+1)=sqrt(bold(v)_(t+1))$ to return to gradient units.],
  spacing: 8pt,
)

#v(8pt)
#note[$bold(v)_t$ is the *mean-square memory*: the running second raw moment of
the gradient, with gradient-squared units. The letter $v$ suggests a
variance-like magnitude, but this is not a centered variance because we do not
subtract the mean. $bold(r)_t=sqrt(bold(v)_t)$ is the RMS scale in gradient
units.]

#v(8pt)
#punch[*RMSProp* is conventionally expanded as *root mean square propagation*.
The RMS scale is the mathematical operation; “Prop” is part of the name, not an
extra line in the update.]

== RMSProp: complete update rule #D

#algorithm-box([RMSProp], [
  $hat(bold(g))_t=nabla_(bold(theta)) cal(L)_(cal(B)_t)(bold(theta)_t)$

  $bold(v)_(t+1)=beta_2 bold(v)_t+(1-beta_2)hat(bold(g))_t^2,
    quad bold(v)_0=bold(0)$

  $bold(r)_(t+1)=sqrt(bold(v)_(t+1))+epsilon_"opt"$

  $bold(theta)_(t+1)=bold(theta)_t-eta
    hat(bold(g))_t/bold(r)_(t+1)$
], color: GREEN)

#note[$bold(v)$ stores recent squared-gradient magnitude, and
$bold(r)=sqrt(bold(v))+epsilon_"opt"$ converts it back to a gradient scale. The
next two slides work through the same scalar loss with all three optimizers.]

== A one-parameter loss we can calculate by hand #D

#align(center, text(size: 24pt)[
  $cal(L)(theta)=1/2 theta^2, quad theta^star=0.$
])

#v(8pt)
#grid(
  columns: (1fr, 1fr), gutter: 14pt, align: top,
  mini-card([START], [
    $theta_0=2$

    $cal(L)(theta_0)=1/2(2)^2=2$
  ], color: ACC),
  mini-card([CURRENT GRADIENT], [
    $g_0=(partial cal(L)(theta))/(partial theta) |_(theta=2)=2$

    All three methods see this same slope.
  ], color: TEAL),
)

#v(10pt)
For the first update, initialize the two memories at zero and use

#align(center, text(size: 19pt)[
  $m_0=0, quad v_0=0, quad eta=.10, quad beta=.5,
    quad beta_2=.9, quad epsilon_"opt"=0.$
])

#v(8pt)
#punch[The loss, starting point, current gradient, and learning-rate value are
the same. Only the quantity transformed before the update is different.]

== First update on $cal(L)(theta)=1/2 theta^2$ #D

#set text(size: 14pt)
#grid(
  columns: (1fr, 1fr, 1fr), gutter: 10pt, align: top,
  mini-card([PLAIN GD], [
    $g_0=2$
    #v(4pt)
    $Delta theta_0=-.10(2)=-.20$
    #v(4pt)
    $theta_1=2-.20=1.80$
    #v(4pt)
    $cal(L)_1=1/2(1.80)^2=1.620$
  ], color: ACC),
  mini-card([MOMENTUM], [
    $m_1=.5(0)+.5(2)=1$
    #v(4pt)
    $Delta theta_0=-.10(1)=-.10$
    #v(4pt)
    $theta_1=2-.10=1.90$
    #v(4pt)
    $cal(L)_1=1/2(1.90)^2=1.805$
  ], color: TEAL),
  mini-card([RMSPROP], [
    $v_1=.9(0)+.1(2^2)=.4$
    #v(3pt)
    $r_1=sqrt(.4)=.632$
    #v(3pt)
    $g_0/r_1=2 div .632=3.162$
    #v(3pt)
    $Delta theta_0=-.10(3.162)=-.316$
    #v(3pt)
    $theta_1=2-.316=1.684$
    #v(3pt)
    $cal(L)_1 approx 1/2(1.684)^2=1.418$
  ], color: GREEN),
)

#v(8pt)
#note[GD uses the current gradient. Momentum first forms a signed running
average. RMSProp first divides the gradient by its recent root-mean-square
scale. This one update checks the formulas; it is not an optimizer ranking.]

== First update: build the two scales #D

Use $bold(g)_0=(-2.4,85)$, $beta_2=.9$, and $bold(v)_0=bold(0)$.

#align(center, table(
  columns: (43mm, 43mm, 43mm), stroke: 0.5pt + MUTED,
  inset: (x: 10pt, y: 6pt), align: (left, center, center),
  table.header([*quantity*], [*$theta_1$ component*], [*$theta_2$ component*]),
  [current gradient $bold(g)_0$], [$-2.4$], [$85$],
  [square $bold(g)_0^2$], [$5.76$], [$7225$],
  [$bold(v)_1=.9bold(v)_0+.1bold(g)_0^2$], [$.576$], [$722.5$],
  [scale $bold(r)_1=sqrt(bold(v)_1)$], [$.759$], [$26.879$],
))

#v(7pt)
#note[The second component receives the larger denominator because its gradient
has been much larger. We omit $epsilon_"opt"$ only in this hand calculation.]

== First update: divide by the local scales #V

#story-figure("story_rmsprop_rescaled_direction", width: 145mm,
  provenance: [computed · $beta_2=.9$ · equal $3.162$ is a zero-start first-update effect])

#align(center, text(size: 15.5pt)[
  $bold(g)_0=(-2.4,85), quad bold(r)_1=(.759,26.879)$
])

#v(2pt)
#align(center, text(size: 15.5pt)[
  $bold(g)_0 div bold(r)_1
   =(-2.4 div .759, 85 div 26.879)
   =(-3.162,3.162)$
])

#v(2pt)
#align(center, text(size: 15.5pt)[
  $Delta bold(theta)_0=-.08(-3.162,3.162)=(.253,-.253)$
])

#v(2pt)
#align(center, text(size: 16pt, weight: 650)[
  $bold(theta)_1=(-1.4,.35)+(.253,-.253)=(-1.147,.097)$
])

== Second update: the history now matters #D

Carry forward $bold(theta)_1=(-1.147,.097)$ and
$bold(v)_1=(.576,722.5)$ from update 1.

#v(2pt)
At $bold(theta)_1$, remeasure
$bold(g)_1=(-2.147,59.702)$.

#align(center, table(
  columns: (47mm, 42mm, 42mm), stroke: 0.5pt + MUTED,
  inset: (x: 10pt, y: 6pt), align: (left, center, center),
  table.header([*quantity*], [*$theta_1$ component*], [*$theta_2$ component*]),
  [$bold(v)_2=.9bold(v)_1+.1bold(g)_1^2$], [$.979$], [$1006.680$],
  [$bold(r)_2=sqrt(bold(v)_2)$], [$.990$], [$31.728$],
  [rescaled $bold(g)_1/bold(r)_2$], [$-2.170$], [$1.882$],
  [$Delta bold(theta)_1=-.08bold(g)_1/bold(r)_2$], [$.174$], [$-.151$],
))

#v(8pt)
#result[$bold(theta)_2=(-.973,-.054)$ and $cal(L)_2=11.915$. The two
rescaled magnitudes are no longer forced to match.]

== RMSProp adds one persistent state to GD #I

#grid(
  columns: (1fr, 1fr), gutter: 14pt, align: top,
  [
    #text(size: 14pt, weight: 700, fill: BLUE)[PLAIN GD]
    #codebox(size: 12.3pt)[```python
def gd_step(theta, grad, lr):
    return theta - lr * grad
```]
  ],
  [
    #text(size: 14pt, weight: 700, fill: GREEN)[RMSPROP]
    #codebox(size: 12.3pt)[```python
def rmsprop_step(theta, grad, v,
                 lr, beta2=0.9, eps=1e-8):
    v = beta2 * v + (1-beta2) * grad.square()
    theta = theta - lr * grad / (v.sqrt() + eps)
    return theta, v
```]
  ],
)

#v(8pt)
#note[Initialize `v = zeros_like(theta)`. PyTorch calls the same decay
`alpha`, so our $beta_2$ is `RMSprop(alpha=beta2)`. The state key `square_avg`
stores $bold(v)$; the denominator is `square_avg.sqrt() + eps`.]

== On Example B, RMSProp removes the visible zigzag #V

#story-figure("story_rmsprop_ravine", width: 192mm,
  provenance: [computed · same objective and start · 40 exact gradients each · method-specific rates shown])

#note[The displayed rates are method-specific. Read the path for the rescaling
mechanism; Notebook 03 reproduces the first updates, persistent state, and
PyTorch parameter path to numerical precision.]

// ───────────────────────────── ACT VII ─────────────────────────────
= Adam combines direction and scale

== Momentum and RMSProp store different information #D

#set text(size: 19pt)
*Momentum* stores $bold(m)_t$, a signed-gradient EWMA. Repeated signs
reinforce; alternating signs partly cancel.

#v(8pt)
*RMSProp* stores $bold(v)_t$, a squared-gradient EWMA. Large recent components
receive larger denominators.

#v(12pt)
#result[When Adam combines the two memories, standalone momentum's $beta$ is
renamed $beta_1$ for the signed-gradient EWMA. RMSProp's $beta_2$ carries over
for the squared-gradient EWMA. The subscripts label the two roles; they do not
imply equal numerical values.]

== What does “moment” mean here? #D

For a random gradient component $G$,

#align(center, $ mu'_1=EE[G] quad "and" quad mu'_2=EE[G^2]. $)

The first raw moment is its mean; the second raw moment is its mean square. Adam
tracks exponentially weighted, recent versions:

$ bold(m)_(t+1)=beta_1 bold(m)_t+(1-beta_1)hat(bold(g))_t $

$ bold(v)_(t+1)=beta_2 bold(v)_t+(1-beta_2)hat(bold(g))_t^2. $

#punch[*Adam* stands for *adaptive moment estimation*: it estimates these two
moments and uses the second to adapt the scale of the first.]

== Adam normalizes both zero-start memories #D

In the zero-start temperature example, observations $1,dots,t$ contributed
weight $1-beta^t$. Optimizer measurement $hat(bold(g))_t$ creates state $t+1$,
so Adam uses $1-beta^(t+1)$:

$ hat(bold(m))_(t+1)=bold(m)_(t+1)/(1-beta_1^(t+1)), quad
  hat(bold(v))_(t+1)=bold(v)_(t+1)/(1-beta_2^(t+1)). $

With $bold(m)_0=bold(v)_0=bold(0)$, $beta_1=.9$, and $beta_2=.999$, the first
update gives

$ bold(m)_1=.1bold(g)_0 arrow.r hat(bold(m))_1=bold(g)_0, quad
  bold(v)_1=.001bold(g)_0^2 arrow.r hat(bold(v))_1=bold(g)_0^2. $

#note[Both memories start at zero, and their different $beta$ values give
different missing masses. Adam normalizes each before combining direction and
scale. This corrects initialization shrinkage—not sampling noise or
predictive-model bias.]

== Adam: complete update rule #D

#algorithm-box([Adam], [
  #set text(size: 16.5pt)
  $bold(m)_0=bold(0), quad bold(v)_0=bold(0)$
  #v(3pt)
  #grid(columns: (1fr, 1fr), gutter: 11pt,
    [$bold(m)_(t+1)=beta_1 bold(m)_t+(1-beta_1)hat(bold(g))_t$],
    [$bold(v)_(t+1)=beta_2 bold(v)_t+(1-beta_2)hat(bold(g))_t^2$],
  )
  #v(3pt)
  $hat(bold(m))_(t+1)=bold(m)_(t+1)/(1-beta_1^(t+1)), quad
    hat(bold(v))_(t+1)=bold(v)_(t+1)/(1-beta_2^(t+1))$
  #v(3pt)
  $bold(theta)_(t+1)=bold(theta)_t-eta
    hat(bold(m))_(t+1)/(sqrt(hat(bold(v))_(t+1))+epsilon_"opt")$
], color: ACC)

#v(5pt)
#grid(
  columns: (0.9fr, 1.1fr), gutter: 15pt, align: top,
  [
    #text(size: 13pt, weight: 700, fill: TEAL)[TWO RETENTION COEFFICIENTS]
    #v(5pt)
    #set text(size: 14.5pt)
    $beta_1=.9$ retains the signed-gradient memory $bold(m)$.
    #v(3pt)
    $beta_2=.999$ retains the squared-gradient memory $bold(v)$.
  ],
  [
    #text(size: 13pt, weight: 700, fill: BLUE)[THE SAME VALUES IN PYTORCH]
    #v(4pt)
    #codebox(size: 10.8pt)[```python
opt = torch.optim.Adam(
    model.parameters(), lr=0.10,
    betas=(0.9, 0.999),  # (beta1, beta2)
    eps=1e-8,
)
```]
  ],
)

== First Adam update on Example B #D

Use $beta_1=.9$, $beta_2=.999$, $eta=.10$, and
$bold(g)_0=(-2.4,85)$.

#align(center, table(
  columns: (52mm, 42mm, 42mm), stroke: 0.5pt + MUTED,
  inset: (x: 10pt, y: 6pt), align: (left, center, center),
  table.header([*quantity*], [*$theta_1$ component*], [*$theta_2$ component*]),
  [$bold(m)_1=.1bold(g)_0$], [$-.24$], [$8.5$],
  [$bold(v)_1=.001bold(g)_0^2$], [$.00576$], [$7.225$],
  [corrected $hat(bold(m))_1$], [$-2.4$], [$85$],
  [scale $sqrt(hat(bold(v))_1)$], [$2.4$], [$85$],
  [step $Delta bold(theta)_0$], [$.10$], [$-.10$],
))

#v(8pt)
#result[$bold(theta)_1 approx (-1.30,.25)$. Ignoring the negligible
$epsilon_"opt"$, the first Adam step is sign-scaled because the corrected second
moment contains only this one gradient.]

== Second Adam update contains history #D

At $bold(theta)_1=(-1.30,.25)$, the new gradient is
$bold(g)_1=(-2.3,75)$.

#align(center, table(
  columns: (52mm, 42mm, 42mm), stroke: 0.5pt + MUTED,
  inset: (x: 10pt, y: 6pt), align: (left, center, center),
  table.header([*quantity*], [*$theta_1$ component*], [*$theta_2$ component*]),
  [$bold(m)_2=.9bold(m)_1+.1bold(g)_1$], [$-.446$], [$15.15$],
  [$bold(v)_2=.999bold(v)_1+.001bold(g)_1^2$], [$.01104$], [$12.8428$],
  [corrected $hat(bold(m))_2$], [$-2.347$], [$79.737$],
  [scale $sqrt(hat(bold(v))_2)$], [$2.351$], [$80.154$],
  [step $Delta bold(theta)_1$], [$.0999$], [$-.0995$],
))

#v(8pt)
#result[$bold(theta)_2=(-1.200,.151)$. From this update onward, both the signed
direction and the coordinate scale contain more than one observation.]

== Adam in one vectorized step function #I

#codebox(size: 12.8pt)[```python
def adam_step(theta, grad, m, v, step, lr,
              beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1-beta1) * grad
    v = beta2 * v + (1-beta2) * grad.square()
    m_hat = m / (1 - beta1**step)
    v_hat = v / (1 - beta2**step)
    theta = theta - lr * m_hat / (v_hat.sqrt() + eps)
    return theta, m, v
```]

#note[Initialize `m = zeros_like(theta)`, `v = zeros_like(theta)`, and call the
first update with `step=1`. `torch.optim.Adam` stores the same states as
`exp_avg` and `exp_avg_sq`.]

== Direction memory, scale memory, or both #V

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 12pt,
  align: top,
  [
    #align(center, text(size: 12.5pt, weight: 750, fill: TEAL)[MOMENTUM])
    #v(3pt)
    #align(center, image("figures/optimizer-intuition-momentum-v1.png", width: 53mm))
    #v(4pt)
    #align(center, text(size: 13.5pt)[
      *Consistent direction?* Average signed gradients.
    ])
  ],
  [
    #align(center, text(size: 12.5pt, weight: 750, fill: GREEN)[RMSPROP])
    #v(3pt)
    #align(center, image("figures/optimizer-intuition-rmsprop-v1.png", width: 53mm))
    #v(4pt)
    #align(center, text(size: 13.5pt)[
      *Large relative to usual?* Divide by the recent RMS scale.
    ])
  ],
  [
    #align(center, text(size: 12.5pt, weight: 750, fill: ACC)[ADAM])
    #v(3pt)
    #align(center, image("figures/optimizer-intuition-adam-v1.png", width: 53mm))
    #v(4pt)
    #align(center, text(size: 13.5pt)[
      *Which direction, at what scale?* Combine both memories.
    ])
  ],
)

#v(4pt)
#evidence[conceptual illustrations · generated with OpenAI Imagen · not data]
#v(4pt)
#note[#text(size: 14.5pt)[Momentum filters direction over time. RMSProp normalizes
each coordinate by its own recent scale. Adam combines the two.]]

== Same ravine, four stored-state mechanisms #V

#story-figure("story_optimizer_paths", width: 200mm,
  provenance: [computed · same full-batch objective · same 55-update budget · method-specific rates shown])

#note[On this controlled example, RMSProp and Adam move along the ravine much
faster than plain GD. The rates are method-specific and must be tuned again on a
different problem.]

== Notebook 03 reproduces every state #I

#set text(size: 18pt)
Notebook 03:

#enum(
  [recomputes the RMSProp and Adam tables above,],
  [runs the same vectorized step functions for complete paths,],
  [checks parameters and stored states against PyTorch to numerical precision.],
  spacing: 7pt,
)

#v(9pt)
#interbox(link-to: NB + "03_adaptive_optimizers_constraints.ipynb")[
  Open *Adaptive optimizers, schedules, and constraints* in Colab ↗
]

// ───────────────────────────── ACT VIII ─────────────────────────────
= Why change the learning rate?

== One multiplier, every optimizer #D

#algorithm-box([Optimizer direction and scheduler scale], [
  $bold(theta)_(t+1)=bold(theta)_t-eta_t bold(d)_t$
], color: ACC)

#v(8pt)
#grid(
  columns: (1fr, 1fr), gutter: 16pt, row-gutter: 8pt,
  [*Gradient descent:* $bold(d)_t=hat(bold(g))_t$],
  [*Momentum:* $bold(d)_t=bold(m)_(t+1)$],
  [*RMSProp:* $bold(d)_t=hat(bold(g))_t/bold(r)_(t+1)$],
  [*Adam:* $bold(d)_t=hat(bold(m))_(t+1)/(sqrt(hat(bold(v))_(t+1))+epsilon_("opt"))$],
)

#v(10pt)
#note[The optimizer constructs the direction $bold(d)_t$. The schedule chooses
one scalar $eta_t$ for that update. Therefore full-batch GD, SGD, momentum,
RMSProp, and Adam can all use a schedule.]

== Why decay is especially useful with stochastic gradients #V

#story-figure("story_schedule_noise_at_optimum", width: 180mm,
  provenance: [computed · Example A · all four sample gradients evaluated at the same optimum])

#align(center, {
  set text(size: 14.5pt)
  $bold(g)_1=(-.2,.2), quad bold(g)_2=(.1,0), quad
   bold(g)_3=(.4,.4), quad bold(g)_4=(-.3,-.6).$
})

#v(5pt)
#note[At $bold(theta)^*=(1.1,2.3)$ their mean is zero, so exact full-batch GD
stops. SGD still draws one nonzero arrow. A smaller $eta_t$ shortens every such
late stochastic move. With exact gradients there is no persistent sampling
noise; a well-tuned constant rate may already be enough.]

== A fixed learning rate forces a choice #V

#story-figure("story_schedule_two_jobs", width: 165mm,
  provenance: [computed · Example A · same 360 sampled indices · sampling with replacement · seed 12])

#note[With the same sampled-index stream, the fixed rate $.10$ first crosses a
full-loss gap of $.5$ after 8 updates; the fixed rate $.02$ needs 56. The larger
rate moves quickly early, while the smaller rate is easier to use for fine
adjustments later. The marker records a first crossing, not a guarantee that the
loss stays below $.5$.]

== Inverse-time decay starts large, then shrinks #D

#align(center, text(size: 24pt)[$ eta_t=eta_0/(1+d t), quad d>0. $])

For $eta_0=.10$ and $d=.02$:

#align(center, table(
  columns: (40mm, 45mm, 64mm), stroke: 0.5pt + MUTED,
  inset: (x: 10pt, y: 7pt), align: (center, center, left),
  table.header([*optimizer update $t$*], [*rate $eta_t$*], [*interpretation*]),
  [$0$], [$.100$], [large initial moves],
  [$50$], [$.050$], [half the initial multiplier],
  [$350$], [$.0125$], [smaller late refinements],
))

#v(8pt)
#note[The intervention is the learning-rate policy. We hold the sampled-index
stream and optimizer equations fixed. Because the parameter paths differ,
later gradient values and stored-state values generally differ too.]

== Decay makes the late updates smaller #I

#grid(
  columns: (116mm, 1fr), gutter: 11mm, align: center + horizon,
  [#story-figure("story_decay_noise", width: 112mm)],
  [
    #set text(size: 17pt)
    *Early checkpoint*

    Both runs start at $.10$ and first reach a full-loss gap below $.5$ after
    8 updates.

    #v(13pt)
    #line(length: 100%, stroke: 0.7pt + MUTED)
    #v(13pt)

    *Final 99 moves*

    #text(fill: ACC, weight: 650)[$3.794$] with fixed $.10$

    #text(fill: TEAL, weight: 650)[$.520$] with decay

    Decay travels about *14% as far*.
  ],
)

#v(3pt)
#evidence[computed · Example A · same sampled-index stream · seed 12]
#v(4pt)
#align(center, text(size: 15pt, weight: 650, fill: GREEN)[
  #link(WEB + "lr-schedule-explorer.html")[Open the Learning-Rate Schedule Explorer ↗]
])

== Six common schedule shapes #V

#story-figure("story_schedule_shapes", width: 181mm,
  provenance: [computed schedule values · illustrative 100-update horizon])

#note(text(size: 17pt)[Inverse-time and exponential taper continuously; linear
falls at a constant pace; milestones drop abruptly; cosine eases into its
floor. Warmup starts below the peak, reaches it gradually, then hands off to
decay. These are schedule shapes—not performance rankings.])

== PyTorch: state the clock explicitly #I

#codebox(size: 14pt)[```python
epochs = 10
total_updates = epochs * len(loader)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=total_updates, eta_min=1e-5)

for epoch in range(epochs):
    for xb, yb in loader:
        opt.zero_grad(set_to_none=True)
        batch_loss(model, xb, yb).backward()
        opt.step()      # uses eta_t; changes parameters
        sched.step()    # prepares eta_(t+1)
```]

#note[Here the clock is *optimizer updates*. The same scheduler can wrap SGD,
momentum, RMSProp, or Adam. With gradient accumulation, advance it after each
executed `opt.step()`, not after every microbatch. If an optimizer step is
skipped—for example after an AMP overflow—skip the scheduler step too.
Epoch-based and validation-based schedulers use different clocks, so name the
clock whenever you report one.]

// ───────────────────────────── ACT IX ─────────────────────────────
= Constraints define valid parameter values

== An optimizer step can leave the feasible set #V

#grid(
  columns: (1.08fr, .92fr), gutter: 20pt, align: center + horizon,
  [
    #align(center, image("figures/concept_feasible_set_step.png", width: 110mm))
    #v(2pt)
    #evidence[conceptual illustration · generated with OpenAI ImageGen · not data]
  ],
  [
    #set align(left)
    #set text(size: 17pt)
    #text(fill: TEAL, weight: 750)[1 · Valid start]
    $bold(theta)_t in cal(C)$: the current parameters satisfy the model's
    constraint.

    #v(11pt)
    #text(fill: ACC, weight: 750)[2 · Ordinary optimizer step]
    $tilde(bold(theta))_(t+1) = bold(theta)_t - eta hat(bold(g))_t$ crosses the
    boundary.

    #v(11pt)
    #text(fill: RED, weight: 750)[3 · Invalid proposal]
    $tilde(bold(theta))_(t+1) in.not cal(C)$, so the model cannot use this value
    directly.

    #v(12pt)
    *The feasible set* $cal(C)$ is simply the set of all allowed parameter
    values.
  ],
)

#note[*Takeaway:* a correct gradient can still propose an invalid parameter. The
constraint must be enforced separately; the following slides show two ways.]

== Why a learned Gaussian scale must be positive #D

Suppose the network predicts both a mean and an uncertainty. Up to the additive
constant $1/2 log(2 pi)$, the per-example negative log-likelihood is

$ y_i | x_i tilde cal(N)(mu_i, sigma_i^2), quad
  ell_i = log sigma_i + (y_i-mu_i)^2/(2 sigma_i^2), quad sigma_i > 0. $

#grid(
  columns: (1fr, 1fr), gutter: 20pt, align: top,
  [
    *Homoskedastic model*

    One learned $sigma$ is shared by every example. It represents one common
    noise level for the dataset.
  ],
  [
    *Heteroskedastic model*

    The network predicts $sigma_i$ from $x_i$. Different inputs may have
    different noise levels.
  ],
)

#v(9pt)
#warning[At $sigma_i=0$, the Gaussian density and the loss are undefined. A
negative “standard deviation” has no probabilistic meaning. The constraint is
therefore *strict*: $sigma_i>0$, not merely $sigma_i>=0$.]

== One gradient step can make the scale negative #D

For one residual $y-mu=.05$, start from the valid scale $sigma_0=.10$:

#grid(
  columns: (1fr, 1fr, 1fr), gutter: 18pt, align: top,
  [#text(size: 12pt, weight: 700, fill: BLUE)[1 · MEASURE]
   #v(3pt)
   $g_0=1/.10-.05^2/.10^3=7.5$],
  [#text(size: 12pt, weight: 700, fill: ACC)[2 · MOVE]
   #v(3pt)
   $Delta sigma_0=-.02(7.5)=-.15$],
  [#text(size: 12pt, weight: 700, fill: RED)[3 · PROPOSE]
   #v(3pt)
   $tilde(sigma)_1=.10-.15=-.05$],
)

#v(4pt)
#note[The gradient is correct. The unconstrained update does not encode the
requirement $sigma>0$.]

#v(1pt)
#align(center, image("figures/constraint-positive-scale-flat-v3.png", width: 150mm))
#grid(
  columns: (1fr, 1fr, 1fr), gutter: 10pt, align: center,
  text(fill: RED, weight: "bold")[$tilde(sigma)_1=-.05$ · invalid],
  text(fill: ACC, weight: "bold")[the proposal crosses $sigma=0$],
  text(fill: TEAL, weight: "bold")[$sigma_0=.10$ · valid],
)
#evidence[conceptual illustration · generated with OpenAI Imagen · values from the worked example]

== A blend coefficient must stay between 0 and 1 #D

Let a model interpolate two predictions:

$ hat(y)=(1-c)f_0(x)+c f_1(x), quad 0 <= c <= 1. $

#grid(
  columns: (.96fr, 1.04fr), gutter: 22pt, align: top,
  [
    #align(center, image("figures/constraint-box-blend-v1.png", width: 83mm))
    #evidence[conceptual illustration · generated with OpenAI Imagen]
  ],
  [
    $c=0$ uses only $f_0$.

    $c=.25$ uses $75%$ of $f_0$ and $25%$ of $f_1$.

    $c=1$ uses only $f_1$.

    #v(6pt)
    $c=1.3$ gives weights $(-.3,1.3)$: this is extrapolation, not the intended
    blend.
  ],
)

#note[The allowed set is the closed interval $[0,1]$: both endpoints are legal
and meaningful.]

== A probability must lie between 0 and 1 #D

For binary classification,

$ y | x tilde "Bernoulli"(p), quad p=P(y=1|x), quad 0 <= p <= 1. $

#grid(
  columns: (.96fr, 1.04fr), gutter: 22pt, align: top,
  [
    #align(center, image("figures/constraint-probability-v1.png", width: 83mm))
    #evidence[conceptual illustration · generated with OpenAI Imagen]
  ],
  [
    $p=.8$ means an $80%$ chance of class $1$.

    $p=1.2$ is impossible: probabilities cannot exceed $1$.

    #v(7pt)
    The Bernoulli loss is

    $ ell(p,y)=-y log p-(1-y)log(1-p). $

    For $p=1.2$, $log(1-p)$ is not real.
  ],
)

#note[We will optimize an unrestricted *logit* and let the sigmoid construct a
valid probability inside the model.]

== Mixture weights must be nonnegative and sum to one #D

A mixture combines $K$ component distributions:

$ p(y|x)=sum_(k=1)^K pi_k p_k(y|x), quad
  pi_k >= 0, quad sum_(k=1)^K pi_k=1. $

#grid(
  columns: (.96fr, 1.04fr), gutter: 22pt, align: top,
  [
    #align(center, image("figures/constraint-simplex-v1.png", width: 83mm))
    #evidence[conceptual illustration · generated with OpenAI Imagen]
  ],
  [
    $bold(pi)=(.2,.5,.3)$ is valid: every entry is nonnegative and the total is
    $1$.

    #v(7pt)
    $bold(pi)=(.2,.5,.6)$ is invalid: each entry lies in $[0,1]$, but the total
    is $1.3$.

    #v(7pt)
    A valid vector is a point *inside the simplex*.
  ],
)

#note[The coordinates are coupled. Clamping each weight separately into
$[0,1]$ does not guarantee that the weights sum to $1$.]

== Optimize a raw value; transform it before using it #D

#align(center, image("figures/constraint-reparam-map-v1.png", width: 112mm))
#evidence[conceptual illustration · generated with OpenAI Imagen]

#v(5pt)
#constraint-flow(
  [$bold(u) in RR^d$],
  [$bold(theta)=T(bold(u))$],
  [$bold(theta) in cal(C)$],
)

#v(7pt)
#note[The optimizer changes the unrestricted value $bold(u)$. The forward pass
applies $T$ before the parameter is used, so the model always sees a valid
$bold(theta)$.]

== Positive scale: real input → exponential → positive output #D

#set text(size: 15pt)
#align(center, image("figures/constraint-log-scale-v1.png", width: 104mm))
#evidence[conceptual illustration · generated with OpenAI Imagen]

#v(3pt)
#constraint-flow(
  [$a in RR$ #linebreak() $a=log(.10) approx -2.303$],
  [$sigma=exp(a)$],
  [$sigma=.10 in (0,infinity)$],
)

#v(5pt)
#codebox(size: 10.6pt)[```python
log_sigma = nn.Parameter(torch.tensor(math.log(0.10)))  # any real value
sigma = log_sigma.exp()                                 # always positive
```]

#note[Every finite `log_sigma` produces $sigma>0$. For input-dependent noise,
the network can output one unrestricted `log_sigma` for each example.]

== Blend coefficient: real input → sigmoid → value between 0 and 1 #D

#set text(size: 15pt)
#align(center, image("figures/constraint-box-sigmoid-v1.png", width: 106mm))
#evidence[conceptual illustration · generated with OpenAI Imagen]

#v(3pt)
#constraint-flow(
  [$u in RR$ #linebreak() $u=log 3 approx 1.099$],
  [$c="sigmoid"(u)$],
  [$c=.75 in (0,1)$],
)

#v(5pt)
#codebox(size: 10.6pt)[```python
raw_c = nn.Parameter(torch.tensor(math.log(3.0)))  # unrestricted
c = torch.sigmoid(raw_c)                           # 0 < c < 1
prediction = (1-c) * f0 + c * f1
```]

#note[Here $c=.75$, so the blend uses $25%$ of $f_0$ and $75%$ of $f_1$.
For a general box $a<theta<b$, use $theta=a+(b-a)"sigmoid"(u)$. Projection is
needed if an exact endpoint must be attainable.]

== Probability: real logit → sigmoid → valid probability #D

#set text(size: 15pt)
#align(center, image("figures/constraint-logit-v1.png", width: 106mm))
#evidence[conceptual illustration · generated with OpenAI Imagen]

#v(3pt)
#constraint-flow(
  [$z in RR$ #linebreak() $z=log 4 approx 1.386$],
  [$p="sigmoid"(z)$],
  [$p=.8 in (0,1)$],
)

#v(5pt)
#codebox(size: 10.6pt)[```python
logit = model(x)  # unrestricted network output
loss = F.binary_cross_entropy_with_logits(logit, y.float())
p = torch.sigmoid(logit)  # convert only when a probability is needed
```]

#note[Train with logits for numerical stability. The sigmoid guarantees that
the reported probability satisfies $0<p<1$.]

== Mixture weights: real logits → softmax → point in the simplex #D

#set text(size: 15pt)
#align(center, image("figures/constraint-softmax-v1.png", width: 106mm))
#evidence[conceptual illustration · generated with OpenAI Imagen]

#v(3pt)
#constraint-flow(
  [$bold(z) in RR^3$ #linebreak() $bold(z)=(0,log 2,0)$],
  [$pi_k=(exp z_k)/(sum_j exp z_j)$],
  [$bold(pi)=(.25,.50,.25)$ #linebreak() $pi_k>0, sum_k pi_k=1$],
)

#v(5pt)
#codebox(size: 10.6pt)[```python
logits = nn.Parameter(torch.tensor([0., math.log(2), 0.]))  # unrestricted
pi = torch.softmax(logits, dim=0)                           # valid weights
```]

#note[Softmax transforms all logits together, so positivity and the sum-to-one
constraint hold after every forward pass.]

== Projection repairs a proposal after the optimizer step #D

#grid(
  columns: (1.04fr, .96fr), gutter: 24pt, align: top,
  [
    First take the ordinary proposal

    $ bold(u)_(t+1)=bold(theta)_t-eta hat(bold(g))_t. $

    Then return it to the closest allowed point:

    $ bold(theta)_(t+1)=Pi_cal(C)(bold(u)_(t+1)). $

    #v(10pt)
    For $theta>=0$: $Pi_cal(C)(u)=max(0,u)$.

    For $a<=theta<=b$: $Pi_cal(C)(u)="clamp"(u,a,b)$.
  ],
  [
    #align(center, image("figures/constraint-projection-v1.png", width: 82mm))
    #evidence[conceptual illustration · generated with OpenAI Imagen]
  ],
)

#note[Projection is natural for a closed feasible set whose boundary is legal.
The proposal may leave the set; the parameter used next is valid again.]

== Project immediately after each optimizer step #I

#grid(
  columns: (1.16fr, .84fr), gutter: 22pt, align: top,
  [
    #codebox(size: 11.2pt)[```python
c = nn.Parameter(torch.tensor(0.20))
opt = torch.optim.SGD([c], lr=0.40)

for _ in range(35):
    opt.zero_grad(set_to_none=True)
    loss = 0.5 * (c - 1.4).square()
    loss.backward()
    opt.step()              # propose u
    with torch.no_grad():
        c.clamp_(0.0, 1.0)  # project u
```]
  ],
  [
    #align(center, image("figures/constraint-clamp-v1.png", width: 72mm))
    #evidence[conceptual illustration · generated with OpenAI Imagen]

    The unconstrained optimum is $1.4$; the box-constrained optimum is the
    legal boundary $c^star=1$.
  ],
)

#warning[Momentum, RMSProp, and Adam update their stored state before this
repair. When projection is frequent, inspect the interaction between that state
and the repaired parameter.]

== Transform strict interiors; project onto legal boundaries #D

#align(center, image("figures/constraint-reparam-vs-projection-v1.png", width: 148mm))
#evidence[conceptual illustration · generated with OpenAI Imagen]

#grid(
  columns: (1fr, 1fr), gutter: 34pt, align: top,
  [
    *Reparameterize.* Optimize $r in RR^d$ and construct $theta=T(r)$ inside
    the model. Use this for $sigma=exp(a)>0$, probabilities, and simplex
    weights.
  ],
  [
    *Project.* Make an ordinary proposal, then apply $Pi_cal(C)$. Use this when
    the legal solution may lie exactly on a closed boundary such as $[a,b]$.
  ],
)

#result[The optimizer supplies the update; the transform or projection enforces
the model's constraint.]

// ───────────────────────────── CLOSE ─────────────────────────────
= Choose the repair that matches the symptom

== Optimization in one view

#{
  set text(size: 13.5pt)
  grid(
    columns: (1fr, 1fr, 1fr), gutter: 13pt, row-gutter: 10pt,
    mini-card([Examples per gradient], [Full batch: exact, slow update. SGD: immediate, noisy. Minibatches balance cost and variance.], color: BLUE),
    mini-card([Direction memory], [Momentum accumulates signed gradients: cancel zigzag and preserve persistent motion.], color: TEAL),
    mini-card([Coordinate scale], [RMSProp rescales using recent squared gradients; Adam adds direction memory.], color: GREEN),
    mini-card([Step size over time], [Tune $eta$ first. Decay late if updates prevent refinement; state the schedule clock.], color: ACC),
    mini-card([Valid parameters], [Reparameterize strict interiors; project only when the boundary is a legal value.], color: RED),
    mini-card([Fair evidence], [Keep initialization, data order, and update budget fixed; inspect training, validation, and state.], color: MUTED),
  )
}

#v(7pt)
#result[Start simple $arrow.r$ diagnose one failure $arrow.r$ add one mechanism $arrow.r$ measure again.]
