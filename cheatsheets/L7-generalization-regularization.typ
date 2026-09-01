#import "../common/cheatsheet.typ": *

#set page(
  paper: "a4",
  margin: (top: 8mm, bottom: 9mm, x: 9mm),
  fill: white,
  footer: context align(center)[
    #text(size: 6.5pt, fill: palette.muted)[
      DL course · Lecture 7 · Generalization and Regularization · page #counter(page).display()
    ]
  ],
)
#set text(font: "IBM Plex Sans", size: 8pt, fill: palette.ink)
#set par(leading: 0.52em, justify: false)
#set list(indent: 10pt, body-indent: 4pt, spacing: 2pt)
#show math.equation: set text(font: "New Computer Modern Math", size: 8.4pt)

#sheet-title(
  [LECTURE 7 CHEATSHEET],
  [Diagnose before you regularize],
  subtitle: [Use training data to fit. Use validation data to choose. Open the test set once.],
)

#banner[
  Regularization supplies a preference where the training data are silent. Validation chooses its strength.
]

#v(5pt)
#grid(
  columns: (1fr, 1fr),
  gutter: 7mm,
  [
    #section-title([1 · Keep the split contract clean], accent: palette.blue)
    #card(accent: palette.blue)[
      #grid(
        columns: (19mm, 1fr),
        row-gutter: 3pt,
        column-gutter: 4pt,
        [#label([TRAIN], color: palette.teal)], [Fit parameters $theta$.],
        [#label([VALIDATE], color: palette.blue)], [Choose the recipe, checkpoint, and every knob.],
        [#label([TEST], color: palette.green)], [Estimate final performance once, after choices are frozen.],
      )
    ]

    #v(4pt)
    #section-title([2 · Read the curves before choosing a cure], accent: palette.blue)
    #card(accent: palette.blue, inset: 4.5pt)[
      #grid(
        columns: (1.06fr, 1fr, 1.35fr),
        column-gutter: 3pt,
        row-gutter: 3pt,
        [#text(weight: "bold", fill: palette.muted)[Pattern]],
        [#text(weight: "bold", fill: palette.muted)[Diagnosis]],
        [#text(weight: "bold", fill: palette.muted)[First move]],
        [train low; val high], [high variance], [regularize or add valid data],
        [both high], [high bias], [improve model, features, or optimization],
        [val rises late], [overtraining], [restore the validation-best checkpoint],
        [gap small; both poor], [still underfit], [do not regularize harder],
      )
    ]

    #v(4pt)
    #card(title: [Bias-variance lens], accent: palette.teal, fill: soft(palette.teal, amount: 95%))[
      #align(center)[$E[(hat(f)(x)-f(x))^2]$]
      #align(center)[$
        = underbrace((E[hat(f)(x)]-f(x))^2)_("bias"^2)
        + underbrace(op("Var")(hat(f)(x)))_("variance")
        + underbrace(sigma^2)_("noise")
      $]
      #v(2pt)
      Regularization usually trades a little more #text(weight: "bold")[bias] for less #text(weight: "bold")[variance]. Validation tells you whether the trade helped.
    ]

    #v(4pt)
    #section-title([3 · One dataset, four outcomes], accent: palette.green)
    #card(accent: palette.green, inset: 4.5pt)[
      #grid(
        columns: (1.45fr, 0.8fr, 0.9fr),
        column-gutter: 3pt,
        row-gutter: 3pt,
        [#text(weight: "bold")[Degree-9 fit]], [#text(weight: "bold", fill: palette.teal)[train MSE]], [#text(weight: "bold", fill: palette.blue)[validation MSE]],
        [no regularizer], [0.0000], [#text(weight: "bold", fill: palette.red)[1.70]],
        [weight decay], [-], [0.065],
        [early stopping], [-], [0.061],
        [input jitter], [-], [#text(weight: "bold", fill: palette.green)[0.048]],
      )
      #v(2pt)
      #tiny-note[Same ten measurements and same flexible model. Different preferences choose different solutions.]
    ]
  ],
  [
    #section-title([4 · Choose the intervention site], accent: palette.orange)
    #card(accent: palette.orange, inset: 4.5pt)[
      #grid(
        columns: (0.8fr, 1.15fr, 0.62fr, 1.45fr),
        column-gutter: 3pt,
        row-gutter: 3pt,
        [#text(weight: "bold", fill: palette.muted)[Site]], [#text(weight: "bold", fill: palette.muted)[Method]], [#text(weight: "bold", fill: palette.muted)[Knob]], [#text(weight: "bold", fill: palette.muted)[Preference]],
        [objective], [weight decay], [$lambda$], [smaller weights],
        [loop], [early stopping], [$t^star$], [simpler training trajectory],
        [data], [augmentation / noise], [strength], [valid nearby views],
        [model], [dropout], [$p_"drop"$], [redundant features],
        [targets], [label smoothing], [$epsilon$], [finite confidence],
      )
      #v(2pt)
      #tiny-note[Mixup and CutMix act on both the examples and their targets. Change one site at a time when you can.]
    ]

    #v(4pt)
    #section-title([5 · Weight decay: prefer smaller weights], accent: palette.orange)
    #card(accent: palette.orange)[
      #keyline([Intuition:], [ among equally good fits, prefer the one that does not need large coefficients.], color: palette.orange)
      #v(3pt)
      #align(center)[$
        cal(J)(w)=1/(2n) norm(X w-y)^2 + lambda/2 norm(w)^2
      $]
      #align(center)[$
        w_(t+1) = underbrace((1-eta lambda)w_t)_("shrink") - underbrace(eta g_t)_("data step")
      $]
      #v(2pt)
      Sweep $lambda$ on a log scale. Too little leaves a high-variance fit; too much underfits.
      #v(2pt)
      #tiny-note[Optional detail: with adaptive optimizers, decoupled AdamW keeps the shrink separate from the adaptive gradient step.]
    ]

    #v(4pt)
    #section-title([6 · Early stopping: select the best checkpoint], accent: palette.green)
    #card(accent: palette.green)[
      #keyline([Intuition:], [ later updates can fit training quirks after the useful pattern has already been learned.], color: palette.green)
      #v(3pt)
      #align(center)[$t^star = limits(op("argmin"))_t cal(L)_"val"(w_t)$]
      #v(2pt)
      #grid(
        columns: (auto, 1fr),
        row-gutter: 2pt,
        column-gutter: 4pt,
        [#label([1], color: palette.green)], [save when validation improves],
        [#label([2], color: palette.green)], [stop after patience is exhausted],
        [#label([3], color: palette.green)], [restore the saved best state],
      )
      #v(2pt)
      The last epoch is not automatically the selected model.
    ]

    #v(4pt)
    #banner(accent: palette.blue)[
      Practical order: diagnose, pick one site, sweep its knob on validation, and keep the simplest recipe that actually helps.
    ]
  ],
)

#pagebreak()

#sheet-title(
  [LECTURE 7 CHEATSHEET],
  [Operate each regularizer correctly],
  subtitle: [The mechanism matters: what changes during training, what stays fixed, and which knob validation chooses.],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 7mm,
  [
    #section-title([1 · Augmentation: teach valid variation], accent: palette.teal)
    #card(accent: palette.teal)[
      #keyline([Intuition:], [ show task-valid versions of each sample so the model cannot rely on one stored form.], color: palette.teal)
      #v(3pt)
      #align(center)[$A_tau(x,y) = (g_tau(x), h_tau(y))$]
      #v(2pt)
      For every transform, decide:
      #grid(
        columns: (auto, 1fr),
        row-gutter: 2pt,
        column-gutter: 4pt,
        [#label([KEEP], color: palette.green)], [the label is unchanged],
        [#label([UPDATE], color: palette.orange)], [move boxes, masks, keypoints, spans, or times],
        [#label([REJECT], color: palette.red)], [the transformed example is no longer valid],
      )
      #v(3pt)
      Input jitter reuses the target locally:
      #align(center)[$epsilon ~ cal(N)(0, sigma^2 I), quad (x_i,y_i) arrow.r (x_i+epsilon,y_i)$]
      #tiny-note[$sigma$ sets the neighbourhood size. Train transforms are stochastic; validation and test stay deterministic. Audit real batches before training.]
    ]

    #v(4pt)
    #section-title([2 · Mixup and CutMix: constrain the space between samples], accent: palette.orange)
    #card(accent: palette.orange)[
      #keyline([Why it can help:], [ the prediction must change smoothly between examples instead of forming a sharp, memorized boundary around each one.], color: palette.orange)
      #v(3pt)
      #align(center)[$
        lambda ~ op("Beta")(alpha,alpha),
        quad tilde(x)=lambda x_i+(1-lambda)x_j,
      $]
      #align(center)[$
        tilde(y)=lambda y_i+(1-lambda)y_j
      $]
      A 70/30 cat-dog blend contributes $-0.70 log p_"cat"-0.30 log p_"dog"$.
      #v(3pt)
      #keyline([Before using it:], [ ask whether the mixed input makes sense, whether the mixed target says what you want, and whether untouched validation data improves.], color: palette.blue)
      #v(2pt)
      #tiny-note[$alpha$ controls how strongly examples mix; $lambda$ is redrawn. CutMix uses the pasted area fraction for the soft target.]
    ]

    #v(4pt)
    #section-title([3 · Dropout: practise with missing features], accent: palette.blue)
    #card(accent: palette.blue)[
      #keyline([Intuition:], [ random feature outages during training reward backup routes, so one feature cannot depend on one fixed partner.], color: palette.blue)
      #v(3pt)
      Let $q_"keep"=1-p_"drop"$:
      #align(center)[$
        m_i ~ op("Bernoulli")(q_"keep"),
        quad tilde(h)_i = (m_i h_i)/q_"keep"
      $]
      #align(center)[$
        E[tilde(h)_i]=h_i,
        quad op("Var")(tilde(h)_i)=((1-q_"keep")/q_"keep")h_i^2
      $]
      #v(2pt)
      #grid(
        columns: (17mm, 1fr),
        row-gutter: 2pt,
        column-gutter: 4pt,
        [#label([TRAIN], color: palette.orange)], [fresh mask; keep survivors scaled by $1/q_"keep"$],
        [#label([EVAL], color: palette.green)], [all units on; no mask and no extra scaling],
      )
      #v(2pt)
      #tiny-note[PyTorch's `p` is the drop probability and scaling is automatic. `model.eval()` changes dropout behaviour; `inference_mode()` changes autograd.]
    ]
  ],
  [
    #section-title([4 · Label smoothing: ask for confidence, not certainty], accent: palette.orange)
    #card(accent: palette.orange)[
      #keyline([Intuition:], [ one-hot cross-entropy keeps rewarding a larger correct-class logit gap. Smoothing gives that push a finite stopping point.], color: palette.orange)
      #v(3pt)
      #align(center)[$
        tilde(y)=(1-epsilon)y + epsilon/K bold(1),
        quad (partial cal(L))/(partial z)=p-tilde(y)
      $]
      With $K=5$ and $epsilon=0.1$:
      #align(center)[$(0,1,0,0,0) arrow.r (0.02,0.92,0.02,0.02,0.02)$]
      The gradient stops pushing class $b$ when $p_b=0.92$. Against any wrong class $j$, the finite target gap is
      #align(center)[$
        m^star = z_b-z_j = ln(0.92/0.02) approx 3.83.
      $]
      #v(2pt)
      #tiny-note[$m^star$ is the desired correct-vs-wrong logit margin. Validate $epsilon$: smoothing can soften too much and can erase useful class-similarity structure.]
    ]

    #v(4pt)
    #section-title([5 · A disciplined recipe], accent: palette.green)
    #card(accent: palette.green)[
      #grid(
        columns: (auto, 1fr),
        row-gutter: 3pt,
        column-gutter: 4pt,
        [#label([1], color: palette.teal)], [fit a reproducible baseline and plot learning curves],
        [#label([2], color: palette.teal)], [add only task-valid augmentation],
        [#label([3], color: palette.orange)], [sweep weight decay on a log scale],
        [#label([4], color: palette.blue)], [add one targeted method only if the curves justify it],
        [#label([5], color: palette.green)], [freeze the recipe and checkpoint; evaluate test once],
      )
      #v(3pt)
      Keep the training budget and validation protocol fixed during comparisons.
    ]

    #v(4pt)
    #section-title([6 · Final audit], accent: palette.blue)
    #card(accent: palette.blue)[
      #check[Learning curves diagnose the problem.]
      #linebreak()
      #check[One intervention site and one named knob.]
      #linebreak()
      #check[Validation chose the recipe and checkpoint.]
      #linebreak()
      #check[The best checkpoint was restored.]
      #linebreak()
      #check[The test set remained untouched until the end.]
      #linebreak()
      #check[Seed, splits, schedule, and every knob were logged.]
    ]

    #v(4pt)
    #banner(accent: palette.orange)[
      If you have not plotted learning curves, diagnose the problem before choosing a regularizer.
    ]

    #v(4pt)
    #card(title: [One-line map], accent: palette.teal, fill: soft(palette.teal, amount: 96%))[
      #keyline([Weight decay], [ constrains parameters.], color: palette.orange)
      #linebreak()
      #keyline([Early stopping], [ constrains training time.], color: palette.green)
      #linebreak()
      #keyline([Augmentation], [ expands valid training views.], color: palette.teal)
      #linebreak()
      #keyline([Mixup/CutMix], [ constrain behaviour between examples.], color: palette.orange)
      #linebreak()
      #keyline([Dropout], [ disrupts hidden feature pathways.], color: palette.blue)
      #linebreak()
      #keyline([Label smoothing], [ softens the confidence target.], color: palette.red)
    ]
  ],
)
