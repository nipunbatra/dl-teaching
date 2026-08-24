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
#let state-chip(label, body, color) = block(
  width: 100%, inset: (x: 9pt, y: 6pt), radius: 3pt,
  fill: color.lighten(92%), stroke: 0.8pt + color,
  [#text(size: 12pt, weight: 700, fill: color)[#label] #h(5pt) #body],
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

== Three examples will keep the mechanisms concrete

#grid(
  columns: (1fr, 1fr, 1fr), gutter: 13pt,
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
  mini-card([C · tiny XOR network], [
    The familiar Lecture 2 XOR task and a $2 arrow.r 4 arrow.r 1$ MLP.
    #v(4pt)
    *Use:* verify the mechanisms in an actual neural-network loop.
  ], color: ACC),
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

$ bold(g)_"full"
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
  $r_(1,i)=f_(bold(theta)_1)(x_i)-y_i, quad
   bold(r)_1=(3.625,1.075,-1.475,-5.025)$
], BLUE)

#v(4pt)
#caption[Because $ell_i=1/2 r_i^2$ and $N=4$, the full loss is $cal(L)(bold(theta))=1/8 norm(bold(r))_2^2$.]

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
  mini-card([average], [$bold(g)_"full"$], color: TEAL),
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
If $cal(B)={1,2}$ contains the first two gradients in our table,

$ hat(bold(g))=1/2[(4,-4)+(1,0)]=(2.5,-2). $

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

    #v(8pt)
    We will stay in the original parameter space throughout:
    $ bold(theta)_0=(-1.4,0.35). $
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

#result[The optimum is $bold(theta)^*=(1,-0.5)$. We will plot and update $bold(theta)$ directly—no second coordinate system is needed.]

== The $10 times$ feature creates a $100 times$ sensitive direction #D

For these four sign combinations, the mixed terms cancel when we average the
half-squared residuals:

#align(center, $
  cal(L)(bold(theta))
  =1/2 (theta_1-1)^2+50(theta_2+0.5)^2,
  quad
  nabla_theta cal(L)(bold(theta))
  =(theta_1-1,100(theta_2+0.5)).
$)

#v(12pt)
#grid(
  columns: (1fr, 1fr), gutter: 18pt,
  mini-card([Move $theta_1$ by the same amount], [loss contribution $=1/2(theta_1-1)^2$], color: BLUE),
  mini-card([Move $theta_2$ by the same amount], [loss contribution $=50(theta_2+0.5)^2$: one hundred times larger], color: ACC),
)

#v(10pt)
#note[*Curvature* means how quickly the slope changes as we move. The second coordinate is much more sensitive because the feature scale was squared.]

== A ravine is a long, narrow low-loss valley #V

#story-figure("story_ravine_geometry", width: 148mm,
  provenance: [computed from the four displayed regression examples · exact full-batch loss])

== One correct step can overshoot across the valley #D

#grid(
  columns: (0.78fr, 1.22fr), gutter: 18pt, align: center + horizon,
  [
    At $bold(theta)_0=(-1.4,.35)$,
    $ bold(g)_0=(-2.4,85). $

    #v(7pt)
    With $eta=.019$,
    $ Delta bold(theta)_0=(.0456,-1.615). $

    #v(7pt)
    Therefore
    $ bold(theta)_1=(-1.3544,-1.265). $

    #v(8pt)
    #warning[$bold(g)_0$ points uphill; $Delta bold(theta)_0=-eta bold(g)_0$
    points downhill. This step overshoots $theta_2^*=-.5$.]
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

== Start with the noisy observations—not the formula #I

#story-figure("story_ewma_raw", width: 166mm,
  provenance: [constructed 120-day temperature signal · fixed seed 12])

#v(6pt)
#note[We want a running estimate of the changing trend. A *memory* is one summary carried from yesterday to today; it does not store every reading.]

#v(5pt)
#align(center, text(size: 13.5pt, weight: 650, fill: rgb("#1B7A34"))[
  #link(WEB + "ewma-explorer.html")[Open the EWMA + Momentum Explorer ↗ — tune
  memory, then step through the actual ravine.]
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
    $m_(t-1)=30$, $x_t=36$, $beta=.9$
    #v(4pt)
    $m_t=.9(30)+.1(36)=30.6$
  ], color: ACC),
  mini-card([What is carried forward?], [
    Keep the new memory $m_t=30.6$ for the next observation.
    The earlier readings need not be stored.
  ], color: TEAL),
)

#v(9pt)
#result[This running summary is an *exponentially weighted moving average* (EWMA). For the temperature curves we set $m_1=x_1$; optimizer momentum below instead starts from $bold(m)_0=bold(0)$.]

== Why “exponential”? Unroll the memory #D

Substitute the same recurrence for the previous memory $m_(t-1)$:

#align(center, $m_t
  = beta [beta m_(t-2)+(1-beta)x_(t-1)] +(1-beta)x_t$)

#align(center, $m_t
  = (1-beta)x_t +(1-beta)beta x_(t-1)+beta^2m_(t-2).$)

#v(6pt)
Substitute once more and the pattern appears:

#align(center, $m_t
  = (1-beta)x_t +(1-beta)beta x_(t-1)
    +(1-beta)beta^2x_(t-2)+beta^3m_(t-3).$)

#v(8pt)
#grid(
  columns: (1fr, 1fr), gutter: 18pt,
  mini-card([General pattern], [
    The weight on $x_(t-k)$ is $(1-beta)beta^k$.
    Every step into the past contributes one more factor of $beta$—this
    geometric decay is the “exponential” in EWMA.
  ], color: TEAL),
  mini-card([$beta=.9$], [
    Recent-observation weights are:
    #v(4pt)
    $x_t arrow.r .100$ #h(16pt) $x_(t-1) arrow.r .090$ \
    $x_(t-2) arrow.r .081$ #h(16pt) $x_(t-3) arrow.r .0729$ $quad dots$
  ], color: ACC),
)

// ───────────────────────────── ACT V ─────────────────────────────
== Same-direction pushes build speed; reversals cancel #V

#story-figure("story_ravine_gradients", width: 156mm,
  provenance: [computed · Example B · actual full-batch path · Notebook 02])

#note[Plain GD uses only the current push. In $theta_1$, successive gradients
keep the same sign; in $theta_2$, they reverse. A running memory therefore
reinforces motion in $theta_1$ while cancelling much of the zigzag in
$theta_2$. Think rolling ball—not literal physics.]

== Momentum update 1: there is no history yet #D

Momentum stores a running memory $bold(m)_t$ of earlier gradients. Before
update 1, it has seen none, so we choose the initial condition

#align(center, text(size: 25pt, weight: 700)[$bold(m)_0 = bold(0).$])

#note[Zero is not a measured gradient or a learned parameter. It means *no
history and no preferred direction*. Hence $bold(m)_1=(1-beta)bold(g)_0$;
earlier gradients enter only from update 2.]

For this example, $beta=.8$ and $eta=.095$: keep $80%$ of the old memory and
add $20%$ of the current full-batch gradient.

#grid(
  columns: (0.86fr, 1.18fr, 1.46fr), gutter: 12pt,
  mini-card([1 · Measure], [
    At $bold(theta)_0$,
    #v(4pt)
    $bold(g)_0=(-2.4,85)$
  ], color: BLUE),
  mini-card([2 · Remember], [
    $bold(m)_1=.8bold(m)_0+.2bold(g)_0$
    #v(4pt)
    $=.2bold(g)_0=(-.48,17)$
  ], color: TEAL),
  mini-card([3 · Move], [
    $bold(theta)_1=bold(theta)_0-.095bold(m)_1$
    #v(4pt)
    $bold(theta)_1=(-1.3544,-1.265)$
  ], color: ACC),
)

== Momentum update 2: old and new directions can disagree #D

At the new point $bold(theta)_1$, keep $bold(m)_1=(-.48,17)$ and *remeasure*:

#align(center, $bold(g)_1=(-2.3544,-76.5), quad
  bold(m)_2=.8bold(m)_1+.2bold(g)_1.$)

#v(8pt)
#grid(
  columns: (1fr, 1fr), gutter: 16pt,
  mini-card([First parameter: same sign], [
    $m_(2,1)=.8(-.48)+.2(-2.3544)$
    #v(4pt)
    $=-.384-.4709=-.8549$
    #v(5pt)
    Same sign $arrow.r$ memory grows.
  ], color: BLUE),
  mini-card([Second parameter: sign reverses], [
    $m_(2,2)=.8(17)+.2(-76.5)$
    #v(4pt)
    $=13.6-15.3=-1.7$
    #v(5pt)
    Sign reversal $arrow.r$ memory nearly cancels.
  ], color: ACC),
)

#v(8pt)
#align(center, text(size: 18pt, weight: 650)[
  Move: $bold(theta)_2=bold(theta)_1-.095bold(m)_2=(-1.2732,-1.1035)$.
])

== What changed in the second momentum step? #V

#story-figure("story_gradient_filter", width: 190mm,
  provenance: [computed from the two worked updates · Example B])

#note[The first-parameter contributions agree, so memory strengthens that
motion. The second-parameter contributions disagree, so memory cancels most of
the reversal before the update is applied.]

== Momentum travels farther in the same 55 updates #V

#story-figure("story_momentum_ravine", width: 169mm,
  provenance: [computed · same start and first step · 55 exact full-batch gradient evaluations each])

#note[Both methods start from the same $bold(theta)_0$ and make the same first
move. Momentum makes more progress in $theta_1$ while reducing the size of the
alternating $theta_2$ moves. The full-loss slide next checks whether that helps.]

== Momentum is faster here—not downhill at every update #V

#story-figure("story_momentum_loss", width: 166mm,
  provenance: [computed on Example B · identical gradient-evaluation budget])

#note[Both curves use exact full-batch gradients. At this stable rate, plain GD
shrinks both errors from the optimum, $abs(theta_1-1)$ and
$abs(theta_2+0.5)$, every update, so its loss falls monotonically.
Momentum carries stored motion past the valley: $cal(L)_2=20.8 arrow.r
cal(L)_3=24.9$. Temporary uphill steps are allowed; compare the overall trend
and final loss.]

== Under the hood: one memory tensor per parameter #I

#codebox(size: 14pt)[```python
beta, eta = 0.8, 0.095
m = [torch.zeros_like(p) for p in params]

for xb, yb in loader:
    loss = batch_loss(params, xb, yb)
    grads = torch.autograd.grad(loss, params)       # measure
    with torch.no_grad():
        for p, g, state in zip(params, grads, m):
            state.mul_(beta).add_(g, alpha=1-beta) # remember
            p.add_(state, alpha=-eta)              # move
```]

#note[The current gradient is recomputed at every update. Only the
parameter-shaped memory persists from one update to the next.]

== PyTorch momentum uses the same training loop #I

#grid(
  columns: (1.05fr, 0.95fr), gutter: 17pt,
  [
    #codebox(size: 13.2pt)[```python
opt = torch.optim.SGD(model.parameters(),
                      lr=0.019, momentum=0.8)

opt.zero_grad(set_to_none=True)
loss.backward()
opt.step()
```]
    #v(5pt)
    #text(size: 13.7pt)[PyTorch stores an unnormalized *momentum buffer*:
    $bold(v)_(t+1)=beta bold(v)_t+bold(g)_t$ and
    $bold(theta)_(t+1)=bold(theta)_t-"lr"_"PT" bold(v)_(t+1)$.
    Since $bold(m)_(t+1)=(1-beta)bold(v)_(t+1)$, the two forms match when
    $"lr"_"PT"=(1-beta)eta=.019$.]
  ],
  [
    #grid(
      columns: (1fr, 1fr), gutter: 9pt,
      mini-card([Signal], [Compare $beta=.5,.9,.98$.], color: ACC),
      mini-card([Optimizer], [Reproduce the two worked updates.], color: TEAL),
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

== Feature scaling helps; it does not make momentum obsolete #D

#grid(
  columns: (1fr, 1fr, 1fr), gutter: 12pt,
  mini-card([Do first], [Normalize input features when their scales differ strongly.], color: BLUE),
  mini-card([What remains], [Deep networks keep changing how parameters interact. Input scaling cannot make every training direction equally easy.], color: TEAL),
  mini-card([And minibatches], [Sampled gradients still fluctuate even with well-scaled inputs.], color: ACC),
)

#v(13pt)
#punch[Normalization changes the geometry. Momentum changes how successive gradients are combined. They are complementary.]

#v(9pt)
#note[For the next controlled experiment, we deliberately return to the original
unscaled ravine so the coordinate-scale problem remains visible.]

// ───────────────────────────── ACT VI ─────────────────────────────
= One learning rate cannot fit both directions

== The problem: one coordinate moves 35× farther #V

#grid(
  columns: (0.78fr, 1.22fr), gutter: 18pt, align: center + horizon,
  [
    At the same start,
    $bold(g)_0=(-2.4,85)$.

    #v(8pt)
    Plain GD uses the same $eta$ in both coordinates, so
    $ abs(g_(0,2)/g_(0,1))=35.4. $

    #v(9pt)
    #note[A larger global rate helps along the valley but worsens the across-valley step. RMSProp will learn a separate scale for each coordinate.]
  ],
  story-figure("story_ravine_first_step", width: 106mm,
    provenance: [computed from Example B at $eta=.019$]),
)

== RMSProp intuition: give each coordinate its own ruler #V

Learn a recent magnitude for each coordinate, then measure today's gradient
against its own ruler rather than one shared scale.

#story-figure("story_rmsprop_ruler", width: 180mm,
  provenance: [computed from $bold(g)_0=(-2.4,85)$ · $rho=.9$ · zero-start first state])

#note[Squaring removes sign; the square root restores gradient units. A coordinate with a large recent ruler is divided more strongly.]

== The rule: remember squared magnitude, then rescale #D

#grid(
  columns: (1fr, 1fr, 1fr), gutter: 12pt,
  mini-card([Measure], [$hat(bold(g))_t$: current full-batch or minibatch gradient], color: BLUE),
  mini-card([Remember scale], [EWMA of $hat(bold(g))_t^2$ in $bold(s)_(t+1)$], color: GREEN),
  mini-card([Move], [divide each coordinate by its learned ruler], color: ACC),
)

#v(9pt)
#pause
$ bold(s)_(t+1)=rho bold(s)_t+(1-rho)hat(bold(g))_t^2 $

$ bold(theta)_(t+1)=bold(theta)_t
  -eta hat(bold(g))_t/(sqrt(bold(s)_(t+1))+epsilon_"opt"). $

#note[Every square, root, and division is elementwise. $epsilon_"opt"$ keeps a nearly zero ruler safe.]

== Work one RMSProp update from start to finish #D

Use $bold(g)_0=(-2.4,85)$, $rho=.9$, and $bold(s)_0=bold(0)$. To keep this
one hand calculation readable, omit $epsilon_"opt"$ here; the implementation
retains it.

#grid(
  columns: (1fr, 1fr, 1fr), gutter: 12pt,
  mini-card([1 · square + remember], [
    $bold(s)_1=.1bold(g)_0^2$
    #v(3pt)
    $=(.576,722.5)$
  ], color: GREEN),
  mini-card([2 · build two rulers], [
    $sqrt(bold(s)_1)$
    #v(3pt)
    $=(.759,26.879)$
  ], color: BLUE),
  mini-card([3 · rescale], [
    $bold(g)_0/sqrt(bold(s)_1)$
    #v(3pt)
    $=(-3.162,3.162)$
  ], color: ACC),
)

#v(9pt)
#result[With $eta=.08$ and $bold(theta)_0=(-1.4,.35)$,
$bold(theta)_1=bold(theta)_0-.08(-3.162,3.162) approx (-1.147,.097)$.]

#v(7pt)
#note[The equal first magnitudes are a *zero-start effect*, not a permanent promise. Later rulers contain the actual gradient history; Notebook 03 traces the second state.]

== Computed outcome: the ravine path straightens #V

#story-figure("story_rmsprop_ravine", width: 192mm,
  provenance: [computed · same objective · 40 exact full-batch gradient evaluations each · method-specific rates shown])

#note[RMSProp now makes progress along the shallow direction without reproducing GD's large repeated crossings. This is a controlled mechanism example, not a universal ranking.]

== PyTorch stores the ruler as `square_avg` #I

#codebox(size: 14pt)[```python
opt = torch.optim.RMSprop(model.parameters(), lr=0.08,
                          alpha=0.9, eps=1e-8)

opt.zero_grad(set_to_none=True)
loss.backward()
opt.step()

state = opt.state[next(model.parameters())]
print(state["square_avg"])  # the tensor s_(t+1)
```]

#note[PyTorch computes $sqrt(bold(s)_(t+1))+epsilon_"opt"$. The notebook checks both `square_avg` and the parameter update after every displayed step.]

// ───────────────────────────── ACT VII ─────────────────────────────
= Adam: remember direction and scale

== Why combine them? the failures are different #V

#grid(
  columns: (1fr, 1fr), gutter: 18pt,
  mini-card([Momentum remembers direction], [Persistent signs accumulate; alternating signs partly cancel. It does not learn a separate coordinate scale.], color: TEAL),
  mini-card([RMSProp remembers scale], [Large recent magnitudes build a large ruler. It does not remember a signed direction.], color: GREEN),
)

#v(12pt)
#pause
#result[Adam asks both questions: *which direction persists?* and *how large is this coordinate usually?*]

#v(8pt)
#punch[Direction memory goes in the numerator; scale memory goes in the denominator.]

== Adam runs two memories in parallel #D

#grid(
  columns: (1fr, 1fr), gutter: 18pt,
  state-chip([$bold(m)_t$ · direction memory], [EWMA of the signed gradient], TEAL),
  state-chip([$bold(v)_t$ · scale memory], [EWMA of the squared gradient], GREEN),
)

#v(8pt)
#pause
$ bold(m)_(t+1)=beta_1 bold(m)_t+(1-beta_1)hat(bold(g))_t $

$ bold(v)_(t+1)=beta_2 bold(v)_t+(1-beta_2)hat(bold(g))_t^2 $

#align(center, text(size: 19pt, weight: 600)[
  $Delta bold(theta)_t=-eta
   hat(bold(m))_(t+1)/(sqrt(hat(bold(v))_(t+1))+epsilon_"opt")$
])

#note[Two parameter-shaped states plus a step counter are carried across updates. The learning rate still matters.]

== Zero-start correction makes early memories comparable #D

Both memories start at zero. Early in training, their exponential weights have
not yet accumulated to one, so the raw memories are artificially small.

#grid(
  columns: (1fr, 1fr), gutter: 18pt,
  mini-card([Correct direction memory], [
    $hat(bold(m))_(t+1)=bold(m)_(t+1)/(1-beta_1^(t+1))$
  ], color: TEAL),
  mini-card([Correct scale memory], [
    $hat(bold(v))_(t+1)=bold(v)_(t+1)/(1-beta_2^(t+1))$
  ], color: GREEN),
)

#v(9pt)
#pause
#result[At the first update, $hat(bold(m))_1=hat(bold(g))_0$ and
$hat(bold(v))_1=hat(bold(g))_0^2$: correction removes the artificial zero-start shrinkage.]

#v(8pt)
#note[For Example B, with $eta=.10$ and $epsilon_"opt"$ omitted only for this arithmetic,
$Delta bold(theta)_0=-.10(-1,1)=(.10,-.10)$, so $bold(theta)_1=(-1.30,.25)$.]

== Same ravine, four optimizers #V

#story-figure("story_optimizer_paths", width: 200mm,
  provenance: [computed · same full-batch objective · same 55-update budget · all method-specific rates shown])

#result[On this example after 55 updates: *RMSProp < momentum < Adam < GD*.
The plotted values use chosen, method-specific rates—not a universal ranking.]

== PyTorch exposes `exp_avg` and `exp_avg_sq` #I

#codebox(size: 14pt)[```python
opt = torch.optim.Adam(model.parameters(), lr=.10,
                       betas=(0.9, 0.999), eps=1e-8)

opt.zero_grad()
loss.backward()
opt.step()

state = opt.state[next(model.parameters())]
print(state.keys())  # step, exp_avg, exp_avg_sq
```]

#note[`exp_avg` is $bold(m)_(t+1)$; `exp_avg_sq` is $bold(v)_(t+1)$. These are the two memories derived on the previous slides.]

== Choose state to match the failure #D

#grid(
  columns: (1fr, 1fr), rows: (1fr, 1fr), gutter: 13pt,
  mini-card([Plain GD / SGD · no state], [Start here: cheap and often strong. It does not address sampling noise or unequal curvature by itself.], color: BLUE),
  mini-card([Momentum · $bold(m)$], [Use when gradient directions persist or alternate. It does not learn coordinate rulers.], color: TEAL),
  mini-card([RMSProp · $bold(s)$], [Use when recent coordinate magnitudes differ. It does not remember signed direction.], color: GREEN),
  mini-card([Adam · $bold(m),bold(v)$], [Use when both effects matter. It still cannot repair bad data, a wrong loss, or poor tuning.], color: ACC),
)

== Notebook 03: inspect every adaptive state #I

#grid(
  columns: (1fr, 1fr), gutter: 18pt,
  mini-card([Mechanism], [Predict the two rulers, trace both Adam memories, then run each method on the exact ravine.], color: GREEN),
  mini-card([Framework check], [Compare every state and parameter update to `torch.optim.RMSprop` and `torch.optim.Adam`.], color: BLUE),
)

#v(9pt)
#interbox(link-to: NB + "03_adaptive_optimizers_constraints.ipynb")[
  Open *Adaptive optimizers, schedules, and constraints* in Colab ↗
]

// ───────────────────────────── ACT VIII ─────────────────────────────
= Let the step size change with training

== One schedule has two jobs: travel early, refine late #V

#story-figure("story_schedule_two_jobs", width: 184mm,
  provenance: [computed · Example A · identical sampled-index stream · seed 12])

#note[A large fixed rate travels quickly but keeps wandering. A small fixed rate settles more tightly but wastes early updates. Decay starts large and finishes small.]

== Decay shrinks late updates, so the path settles #V

#story-figure("story_decay_noise", width: 184mm,
  provenance: [computed · Example A · identical sampled-index stream · seed 12])

#note[Same start, optimizer, and sampled-index stream. The fixed-rate run travels $7.3 times$ as far as the decayed run over the final 99 transitions here.]

== One simple schedule is enough to understand the idea #D

$ eta_t=eta_0/(1+d t), quad d>0. $

#pause
#align(center, table(
  columns: (35mm, 45mm, 1fr), stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 6pt),
  table.header([*clock $t$*], [*rate*], [*interpretation*]),
  [$0$], [$eta_0$], [fast initial movement],
  [$50$], [$eta_0/(1+50d)$], [smaller updates],
  [$t arrow.r infinity$], [$eta_t arrow.r 0$], [late refinement],
))

#pause
#note[Step, exponential, cosine, and warmup schedules change the shape—not the core idea. Always state whether $t$ counts optimizer updates or epochs.]

== PyTorch: implement the *same* update-clock schedule #I

#codebox(size: 14pt)[```python
opt = torch.optim.SGD(model.parameters(), lr=.10)
d = .02
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lr_lambda=lambda t: 1.0 / (1.0 + d*t))

for xb, yb in loader:
    opt.zero_grad(set_to_none=True)
    batch_loss(model, xb, yb).backward()
    opt.step()      # uses eta_t
    sched.step()    # prepares eta_(t+1)
```]

#note[The formula, computed figure, notebook, and code now all use inverse-time decay per optimizer update. With gradient accumulation, advance this clock only after `opt.step()`.]

== Different schedules change the shape, not the clock #V

#story-figure("story_schedule_shapes", width: 164mm,
  provenance: [computed · same initial rate and update horizon · schedule shapes only])

#note[Inverse-time, exponential, and cosine are different choices for $eta_t$. Whatever the shape, state whether $t$ counts optimizer updates, epochs, or another explicit event.]

// ───────────────────────────── ACT IX ─────────────────────────────
= Optimizers on the XOR network

== Example C: return to Lecture 2's XOR task

#grid(
  columns: (1fr, auto, 1fr), gutter: 14pt, align: center + horizon,
  mini-card([same data], [$(0,0) arrow.r 0$, $(0,1) arrow.r 1$, $(1,0) arrow.r 1$, $(1,1) arrow.r 0$], color: BLUE),
  text(size: 22pt)[$arrow.r$],
  mini-card([same network], [$2 arrow.r 4$ `tanh` units $arrow.r 1$ logit], color: TEAL),
)

#v(10pt)
#pause
#note[The representation and loss are fixed. We change only the optimizer while keeping initialization, minibatch order, and update budget identical.]

== A controlled mechanism demo—not a leaderboard #D

#align(center, table(
  columns: (44mm, 76mm, 30mm), stroke: 0.5pt + MUTED,
  inset: (x: 10pt, y: 6pt), align: (left, left, center),
  table.header([*method*], [*PyTorch configuration*], [*learning rate*]),
  [plain SGD], [`SGD(momentum=0)`], [$0.25$],
  [momentum SGD], [`SGD(momentum=0.9)`], [$0.12$],
  [RMSProp], [`RMSprop(alpha=0.9)`], [$0.03$],
  [Adam], [`Adam(betas=(.9,.999))`], [$0.03$],
))

#pause
#warning[The rates are method-specific and were chosen so every method solves this tiny task. Their raw values are not directly comparable across update rules.]

== Same initialization, same 400 minibatches #V

#story-figure("story_xor_optimizer_lab", width: 194mm,
  provenance: [computed · seed 2026 · shared initialization and size-2 batch schedule · 400 updates])

== Same 100% training accuracy—different learned boundaries #V

#story-figure("story_xor_decision_boundaries", width: 172mm)

== The PyTorch loop changes in one line #I

#codebox(size: 13.5pt)[```python
model.load_state_dict(initial_state)  # identical start

# choose exactly one:
opt = torch.optim.SGD(model.parameters(), lr=.25)
opt = torch.optim.SGD(model.parameters(), lr=.12, momentum=.9)
opt = torch.optim.RMSprop(model.parameters(), lr=.03, alpha=.9)
opt = torch.optim.Adam(model.parameters(), lr=.03)

for indices in shared_batch_schedule:
    opt.zero_grad(set_to_none=True)
    loss = F.binary_cross_entropy_with_logits(
        model(X[indices]).squeeze(-1), y[indices])
    loss.backward(); opt.step()
```]

#note[The optimizer transforms gradients; the model, loss, data, initialization, and batch order remain controlled.]

== Checkpoint: what does the XOR result establish? #Q

#grid(
  columns: (1fr, 1fr), gutter: 12pt,
  mini-card([A], [Adam is universally the best optimizer.], color: MUTED),
  mini-card([B], [Momentum always needs a smaller learning rate.], color: MUTED),
  mini-card([C], [Each stored state can be implemented in the same training loop.], color: BLUE),
  mini-card([D], [Four training points predict large-model generalization.], color: MUTED),
)

== Answer: the implementation bridge is the evidence #A

#result[Correct: C · the same backward pass can feed SGD, momentum, RMSProp, or Adam.]

#v(8pt)
#note[The controlled run shows mechanism and reproducibility. It cannot rank optimizers for other models, datasets, budgets, or validation metrics.]

== Notebook 04 makes every control explicit #I

#grid(
  columns: (1fr, 1fr), gutter: 18pt,
  mini-card([Controlled], [Reuse one initial state and one exact 400-minibatch schedule across all four optimizers.], color: TEAL),
  mini-card([Inspectable], [Plot full-data BCE after every update and evaluate every learned decision boundary on one grid.], color: BLUE),
)

#v(9pt)
#interbox(link-to: NB + "04_xor_optimizer_lab.ipynb")[
  Open *XOR optimizer lab* in Colab ↗
]

// ───────────────────────────── ACT X ─────────────────────────────
= Constrained parameters: keep every forward pass valid

== A valid optimizer step can produce an invalid model #D

A Gaussian scale must stay positive, but the optimizer sees only a gradient.
For one observation with $y=mu$,

$ cal(L)(s)=log s + (y-mu)^2/(2s^2)=log s, quad s>0. $

#story-figure("story_constraint_failure", width: 178mm,
  provenance: [computed · $s_0=.10$ · $eta=.02$ · one exact gradient step])

== Reparameterize: optimize a raw value #I

#grid(
  columns: (.92fr, 1.08fr), gutter: 18pt, align: top,
  [
    #mini-card([Coordinate change], [
      $r in RR arrow.r s="softplus"(r)+epsilon>0$
      #v(5pt)
      $(partial cal(L))/(partial r)=(partial cal(L))/(partial s) sigma(r)$
    ], color: TEAL)
    #v(8pt)
    #set text(size: 14.5pt)
    #enum(
      [positive $(0,infinity)$: softplus or $exp$],
      [probability $(0,1)$: sigmoid of a logit],
      [simplex interior: softmax of logits],
      spacing: 5pt,
    )
  ],
  [
    #text(size: 14pt, weight: 700, fill: BLUE)[MINIMAL PYTORCH]
    #v(4pt)
    #codebox(size: 11.7pt)[```python
raw_scale = nn.Parameter(torch.tensor(0.0))
prob_logit = nn.Parameter(torch.tensor(0.0))
mix_logits = nn.Parameter(torch.zeros(3))
opt = torch.optim.Adam(
    [raw_scale, prob_logit, mix_logits], lr=1e-2)

opt.zero_grad(set_to_none=True)
scale = F.softplus(raw_scale) + 1e-6
prob = torch.sigmoid(prob_logit)
mix = torch.softmax(mix_logits, dim=0)
loss = objective(scale, prob, mix)
loss.backward(); opt.step()
```]
  ],
)

#note[Autograd applies the chain rule to the raw parameters. For Bernoulli or categorical losses, prefer logits-based functions such as `BCEWithLogitsLoss` and `log_softmax`.]

== Project when the boundary itself is valid #I

#grid(
  columns: (.92fr, 1.08fr), gutter: 18pt, align: top,
  [
    #mini-card([Closed feasible set], [
      $tilde(s)_(t+1)=s_t-eta g_t$
      #v(4pt)
      $s_(t+1)=max(0,tilde(s)_(t+1))$
    ], color: ACC)
    #v(8pt)
    #mini-card([Choose by meaning], [
      Use a smooth map for a strict interior such as $s>0$. Project when an exact boundary such as $s=0$ is a valid model value.
    ], color: TEAL)
  ],
  [
    #text(size: 14pt, weight: 700, fill: BLUE)[MINIMAL PYTORCH]
    #v(4pt)
    #codebox(size: 12pt)[```python
nonnegative = nn.Parameter(torch.tensor(2.0))
opt = torch.optim.SGD([nonnegative], lr=.25)

opt.zero_grad(set_to_none=True)
loss = 0.5 * (nonnegative + 1).square()
loss.backward()
opt.step()                       # unconstrained proposal
with torch.no_grad():
    nonnegative.clamp_(min=0.0) # projection onto [0, infinity)
```]
  ],
)

#warning[Projection changes the parameter, but momentum or Adam state still reflects the pre-projection proposal. Inspect that interaction rather than clamping silently.]

// ───────────────────────────── CLOSE ─────────────────────────────
= Choose the repair that matches the symptom

== Optimization in one view

#{
  set text(size: 13.5pt)
  grid(
    columns: (1fr, 1fr, 1fr), gutter: 13pt, row-gutter: 10pt,
    mini-card([Examples per gradient], [Full batch: exact, slow update. SGD: immediate, noisy. Minibatches balance cost and variance.], color: BLUE),
    mini-card([Direction memory], [Momentum averages signed gradients: cancel zigzag and preserve persistent motion.], color: TEAL),
    mini-card([Coordinate scale], [RMSProp rescales using recent squared gradients; Adam adds direction memory.], color: GREEN),
    mini-card([Step size over time], [Tune $eta$ first. Decay late if updates prevent refinement; state the schedule clock.], color: ACC),
    mini-card([Valid parameters], [Reparameterize strict interiors; project only when the boundary is a legal value.], color: RED),
    mini-card([Fair evidence], [Keep initialization, data order, and update budget fixed; inspect training, validation, and state.], color: MUTED),
  )
}

#v(7pt)
#result[Start simple $arrow.r$ diagnose one failure $arrow.r$ add one mechanism $arrow.r$ measure again.]
