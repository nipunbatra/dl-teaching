#import "../common/cheatsheet.typ": *

#set page(
  paper: "a4",
  margin: (top: 8mm, bottom: 9mm, x: 9mm),
  fill: white,
  footer: context align(center)[
    #text(size: 6.5pt, fill: palette.muted)[
      DL 2026 · Public lecture 5 · Optimization I · page #counter(page).display()
    ]
  ],
)
#set text(font: "IBM Plex Sans", size: 8pt, fill: palette.ink)
#set par(leading: 0.52em, justify: false)
#set list(indent: 10pt, body-indent: 4pt, spacing: 2pt)
#show math.equation: set text(font: "New Computer Modern Math", size: 8.4pt)

#sheet-title(
  [PUBLIC LECTURE 5 CHEATSHEET],
  [Gradient estimates and the training clock],
  subtitle: [Choose how much data measures each step, then name the unit in which training progress is counted.],
)

#banner[
  Backprop measures a slope. Optimization chooses which data measure it and how far the parameters move.
]

#v(5pt)
#grid(
  columns: (1fr, 1fr), gutter: 7mm,
  [
    #section-title([1 · Objective and vanilla update], accent: palette.blue)
    #card(accent: palette.blue)[
      For $N$ examples and parameters $bold(theta)$:
      #align(center)[$
        ell_i(bold(theta))=ell(f_(bold(theta))(bold(x)_i),y_i),
        quad cal(L)(bold(theta))=1/N sum_(i=1)^N ell_i(bold(theta)).
      $]
      Full-batch gradient descent uses
      #align(center)[$
        bold(g)_t=nabla cal(L)(bold(theta)_t),
        quad bold(theta)_(t+1)=bold(theta)_t-eta bold(g)_t.
      $]
      $eta>0$ is the learning rate. The gradient points uphill; subtracting it moves locally downhill.
    ]

    #v(4pt)
    #section-title([2 · Full batch, SGD, minibatch], accent: palette.teal)
    #card(accent: palette.teal, inset: 4.5pt)[
      #grid(
        columns: (0.9fr, 0.55fr, 1.05fr, 1fr), column-gutter: 3pt, row-gutter: 3pt,
        [#text(weight: "bold", fill: palette.muted)[Method]], [#text(weight: "bold", fill: palette.muted)[$B$]], [#text(weight: "bold", fill: palette.muted)[Updates / epoch]], [#text(weight: "bold", fill: palette.muted)[Noise]],
        [full batch], [$N$], [$1$], [none for empirical loss],
        [SGD], [$1$], [$N$], [highest],
        [minibatch], [$1<B<N$], [$ceil(N/B)$], [intermediate],
      )
      #v(3pt)
      Minibatch estimator:
      #align(center)[$hat(bold(g))_t=1/B_t sum_(i in cal(B)_t) nabla ell_i(bold(theta)_t).$]
    ]

    #v(4pt)
    #section-title([3 · Name the clock], accent: palette.orange)
    #card(accent: palette.orange)[
      If $N$ examples are processed in batches of size $B$:
      #align(center)[$K=ceil(N/B) quad "iterations per epoch".$]
      #grid(
        columns: (auto, 1fr), row-gutter: 2pt, column-gutter: 4pt,
        [#label([EPOCH], color: palette.orange)], [one pass through the data],
        [#label([ITERATION], color: palette.blue)], [one minibatch loop body],
        [#label([UPDATE], color: palette.green)], [one executed parameter change],
      )
      Usually iteration $=$ update. Gradient accumulation and skipped optimizer steps are important exceptions.
    ]
  ],
  [
    #section-title([4 · Why a sampled gradient can work], accent: palette.green)
    #card(accent: palette.green)[
      Draw $I_t$ uniformly from ${1,dots,N}$ and use
      #align(center)[$hat(bold(g))_t=nabla ell_(I_t)(bold(theta)_t).$]
      Holding $bold(theta)_t$ fixed,
      #align(center)[$
        EE[hat(bold(g))_t | bold(theta)_t]
        =1/N sum_i nabla ell_i(bold(theta)_t)
        =nabla cal(L)(bold(theta)_t).
      $]
      The estimator is #text(weight: "bold")[unbiased]: correct at its centre across repeated draws.
    ]

    #v(4pt)
    #card(title: [What unbiased does not mean], accent: palette.red, fill: soft(palette.red, amount: 96%))[
      One sampled gradient need not be close to the full gradient and one valid SGD step need not reduce the full loss. It only satisfies
      #align(center)[$EE[Delta bold(theta)_t | bold(theta)_t]=-eta nabla cal(L)(bold(theta)_t).$]
      Larger minibatches preserve the same centre while averaging away some sample disagreement.
    ]

    #v(4pt)
    #section-title([5 · Batch size is a systems-and-statistics knob], accent: palette.blue)
    #card(accent: palette.blue, inset: 4.5pt)[
      #grid(
        columns: (1fr, 1fr), column-gutter: 4pt, row-gutter: 3pt,
        [#text(weight: "bold", fill: palette.teal)[Smaller $B$]], [#text(weight: "bold", fill: palette.blue)[Larger $B$]],
        [more updates per epoch], [fewer updates per epoch],
        [noisier estimates], [less estimator spread],
        [less parallel work / update], [more parallelism / update],
        [may regularize implicitly], [may need rate retuning],
      )
      Compare experiments by examples processed or optimizer updates, not only by epochs.
    ]

    #v(4pt)
    #banner(accent: palette.orange)[
      “SGD” in libraries often means the minibatch-SGD family. The dataloader's batch size tells you which member you ran.
    ]
  ],
)

#pagebreak()

#sheet-title(
  [PUBLIC LECTURE 5 CHEATSHEET],
  [Step size, geometry, and diagnosis],
  subtitle: [An exact gradient can still make poor progress when one scalar learning rate must serve very different directions.],
)

#grid(
  columns: (1fr, 1fr), gutter: 7mm,
  [
    #section-title([1 · The learning rate scales every coordinate], accent: palette.orange)
    #card(accent: palette.orange)[
      #align(center)[$Delta bold(theta)_t=-eta hat(bold(g))_t.$]
      Too small: safe but slow. Too large: overshoots, oscillation, or divergence. A useful rate makes early progress without destroying stability.
      #v(3pt)
      #keyline([Diagnostic:], [ plot training loss against optimizer updates and inspect the actual rate used at each update.], color: palette.orange)
    ]

    #v(4pt)
    #section-title([2 · Ravines create a speed-stability trade-off], accent: palette.blue)
    #card(accent: palette.blue)[
      A transparent anisotropic loss is
      #align(center)[$
        cal(L)(bold(theta))=1/2(theta_1-1)^2+50(theta_2+0.5)^2,
      $]
      #align(center)[$
        nabla cal(L)=(theta_1-1,100(theta_2+0.5)).
      $]
      One direction is $100 times$ steeper. A rate large enough to move quickly along $theta_1$ can repeatedly cross the narrow valley in $theta_2$; a stable small rate can crawl along the valley floor.
    ]

    #v(4pt)
    #card(title: [Concrete first step], accent: palette.teal, fill: soft(palette.teal, amount: 96%))[
      At $bold(theta)_0=(-1.4,0.35)$, $bold(g)_0=(-2.4,85)$. With $eta=0.0195$:
      #align(center)[$
        Delta bold(theta)_0=(0.0468,-1.6575),
        quad bold(theta)_1=(-1.3532,-1.3075).
      $]
      The step is downhill locally yet crosses the optimum $theta_2^*=-0.5$ in the steep direction.
    ]

    #v(4pt)
    #section-title([3 · Fix geometry before blaming the optimizer], accent: palette.green)
    #card(accent: palette.green)[
      #check[Standardize input features when scales differ strongly.]
      #linebreak()
      #check[Inspect loss contours or per-parameter update scales on small problems.]
      #linebreak()
      #check[Check exploding gradients, invalid targets, and reduction factors.]
      #linebreak()
      #check[Retune $eta$ after changing batch size or normalization.]
    ]
  ],
  [
    #section-title([4 · A disciplined batch-size comparison], accent: palette.teal)
    #card(accent: palette.teal)[
      Hold the model, data order, objective reduction, and total example budget fixed. For each $B$ log:
      #grid(
        columns: (auto, 1fr), row-gutter: 2pt, column-gutter: 4pt,
        [#label([A], color: palette.teal)], [examples processed],
        [#label([B], color: palette.teal)], [optimizer updates],
        [#label([C], color: palette.teal)], [wall-clock time and device utilization],
        [#label([D], color: palette.teal)], [training and validation metrics],
        [#label([E], color: palette.teal)], [gradient or update norms],
      )
      If $B$ changes, do not assume the old learning rate remains optimal.
    ]

    #v(4pt)
    #section-title([5 · Minimal PyTorch loop], accent: palette.blue)
    #card(accent: palette.blue, inset: 4pt)[
      #raw("for xb, yb in loader:\n    opt.zero_grad(set_to_none=True)\n    pred = model(xb)\n    loss = loss_fn(pred, yb)   # usually batch mean\n    loss.backward()            # gradient estimate\n    opt.step()                 # one update", block: true, lang: "python")
      #tiny-note[`optimizer.zero_grad()` is essential because PyTorch accumulates gradients. Record the loss reduction (`mean` versus `sum`).]
    ]

    #v(4pt)
    #section-title([6 · Symptom → first check], accent: palette.orange)
    #card(accent: palette.orange, inset: 4.5pt)[
      #grid(
        columns: (1.05fr, 1.45fr), column-gutter: 4pt, row-gutter: 3pt,
        [#text(weight: "bold", fill: palette.muted)[Symptom]], [#text(weight: "bold", fill: palette.muted)[First check]],
        [loss explodes], [$eta$, invalid values, reduction scale],
        [loss zigzags], [feature scale / curvature; reduce $eta$],
        [loss flat], [zeroed gradients, dead units, too-small $eta$],
        [very noisy], [shuffle, batch size, outliers, logging window],
        [run is slow], [examples per second and updates per second],
      )
    ]

    #v(4pt)
    #banner(accent: palette.green)[
      Report the batch size, loss reduction, learning rate, update count, and example budget. Otherwise the optimization experiment is underspecified.
    ]
  ],
)
