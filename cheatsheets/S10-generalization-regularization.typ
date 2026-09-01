#import "../common/cheatsheet.typ": *

#set page(
  paper: "a4",
  margin: (top: 8mm, bottom: 9mm, x: 9mm),
  fill: white,
  footer: context align(center)[
    #text(size: 6.5pt, fill: palette.muted)[
      DL course · Lecture 10 · Generalization & Regularization · page #counter(page).display()
    ]
  ],
)
#set text(font: "IBM Plex Sans", size: 8pt, fill: palette.ink)
#set par(leading: 0.52em, justify: false)
#set list(indent: 10pt, body-indent: 4pt, spacing: 2pt)
#show math.equation: set text(font: "New Computer Modern Math", size: 8.4pt)

#sheet-title(
  [LECTURE 10 CHEATSHEET],
  [Diagnose before you regularize],
  subtitle: [Use training data to fit. Use validation data to choose. Open the test set once.],
)

#banner[Regularization supplies a preference where the training data are silent. Validation chooses its strength.]

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
        [#label([TRAIN], color: palette.teal)], [fit parameters $theta$],
        [#label([VALIDATE], color: palette.blue)], [choose method, strength, schedule, and checkpoint],
        [#label([TEST], color: palette.green)], [estimate final performance once choices are frozen],
      )
    ]

    #v(4pt)
    #section-title([2 · Read the curves first], accent: palette.blue)
    #card(accent: palette.blue, inset: 4.5pt)[
      #grid(
        columns: (1.05fr, 0.9fr, 1.35fr),
        column-gutter: 3pt,
        row-gutter: 3pt,
        [#text(weight: "bold")[Pattern]], [#text(weight: "bold")[Diagnosis]], [#text(weight: "bold")[First move]],
        [train low; val high], [high variance], [regularize or add valid data],
        [both high], [high bias], [improve model, features, or optimization],
        [val rises late], [overtraining], [restore validation-best checkpoint],
        [gap small; both poor], [underfit], [do not regularize harder],
      )
    ]

    #v(4pt)
    #card(title: [Bias–variance lens], accent: palette.teal, fill: soft(palette.teal, amount: 95%))[
      #align(center)[$
        "prediction error" = "bias"^2 + "variance" + "noise"
      $]
      #align(center)[$
        "bias"=E[hat(f)(x)]-f(x), quad
        "variance"=op("Var")(hat(f)(x)).
      $]
      Regularization often accepts a little more bias to reduce variance. Validation tests whether that trade improves unseen data.
    ]

    #v(4pt)
    #section-title([3 · Compare at fixed conditions], accent: palette.green)
    #card(accent: palette.green)[
      Keep the split, model family, optimizer budget, augmentation policy, and evaluation pipeline fixed while changing one regularization knob.
      #v(2pt)
      #tiny-note[Training loss is allowed to rise. The relevant question is whether validation performance and stability improve.]
    ]
  ],
  [
    #section-title([4 · Choose the intervention site], accent: palette.orange)
    #card(accent: palette.orange, inset: 4.5pt)[
      #grid(
        columns: (0.85fr, 1.15fr, 0.65fr, 1.35fr),
        column-gutter: 3pt,
        row-gutter: 3pt,
        [#text(weight: "bold")[Site]], [#text(weight: "bold")[Method]], [#text(weight: "bold")[Knob]], [#text(weight: "bold")[Preference]],
        [objective], [weight decay], [$lambda$], [smaller weights],
        [loop], [early stopping], [$t^star$], [shorter useful trajectory],
        [data], [augmentation], [strength], [valid nearby views],
        [model], [dropout], [$p_"drop"$], [redundant features],
        [targets], [label smoothing], [$epsilon$], [finite confidence],
      )
      #v(2pt)
      #tiny-note[Mixup and CutMix act on examples and targets. Change one site at a time when possible.]
    ]

    #v(4pt)
    #section-title([5 · Weight decay], accent: palette.orange)
    #card(accent: palette.orange)[
      #align(center)[$
        cal(J)(w)=cal(L)_"data"(w)+lambda/2 norm(w)^2
      $]
      For SGD, the update can be read as
      #align(center)[$
        w_(t+1)=(1-eta lambda)w_t-eta g_t.
      $]
      Sweep $lambda$ on a log scale. Too little leaves high variance; too much underfits.
      #v(2pt)
      #tiny-note[With adaptive optimizers, AdamW decouples shrinkage from adaptive gradient scaling.]
    ]

    #v(4pt)
    #section-title([6 · Early stopping], accent: palette.green)
    #card(accent: palette.green)[
      #align(center)[$t^star=limits(op("argmin"))_t cal(L)_"val"(w_t).$]
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
      Diagnose → choose one site → sweep its knob on validation → keep the simplest recipe that helps.
    ]
  ],
)

#pagebreak()

#sheet-title(
  [LECTURE 10 CHEATSHEET],
  [Operate each regularizer correctly],
  subtitle: [Know what changes during training, what stays fixed, and which knob validation chooses.],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 7mm,
  [
    #section-title([1 · Augmentation: valid variation], accent: palette.teal)
    #card(accent: palette.teal)[
      #align(center)[$A_tau(x,y)=(g_tau(x),h_tau(y)).$]
      For every transform, decide:
      #grid(
        columns: (auto, 1fr),
        row-gutter: 2pt,
        column-gutter: 4pt,
        [#label([KEEP], color: palette.green)], [the label is unchanged],
        [#label([UPDATE], color: palette.orange)], [move boxes, masks, keypoints, spans, or times],
        [#label([REJECT], color: palette.red)], [the transformed example is not task-valid],
      )
      #v(2pt)
      Train transforms are stochastic; validation and test transforms are deterministic. Inspect actual batches before training.
    ]

    #v(4pt)
    #section-title([2 · Mixup and CutMix], accent: palette.orange)
    #card(accent: palette.orange)[
      #align(center)[$
        lambda ~ op("Beta")(alpha,alpha), quad
        tilde(x)=lambda x_i+(1-lambda)x_j,
      $]
      #align(center)[$tilde(y)=lambda y_i+(1-lambda)y_j.$]
      The model is asked to vary smoothly between examples. Validate $alpha$, and ask whether both the mixed input and soft target make sense for the task.
      #v(2pt)
      #tiny-note[CutMix uses the pasted area fraction for the target weight.]
    ]

    #v(4pt)
    #section-title([3 · Dropout: missing-feature practice], accent: palette.blue)
    #card(accent: palette.blue)[
      With keep probability $q=1-p_"drop"$:
      #align(center)[$
        m_i ~ op("Bernoulli")(q), quad tilde(h)_i=m_i h_i/q.
      $]
      Then $E[tilde(h)_i]=h_i$.
      #grid(
        columns: (17mm, 1fr),
        row-gutter: 2pt,
        column-gutter: 4pt,
        [#label([TRAIN], color: palette.orange)], [fresh mask; surviving units scaled by $1/q$],
        [#label([EVAL], color: palette.green)], [all units active; no mask or extra scaling],
      )
      #v(2pt)
      #tiny-note[`model.eval()` changes dropout behaviour. `inference_mode()` changes autograd; they solve different problems.]
    ]
  ],
  [
    #section-title([4 · Label smoothing], accent: palette.orange)
    #card(accent: palette.orange)[
      #align(center)[$
        tilde(y)=(1-epsilon)y+epsilon/K bold(1), quad
        (partial cal(L))/(partial z)=p-tilde(y).
      $]
      With $K=5$ and $epsilon=0.1$, the correct-class target is $0.92$ and each wrong-class target is $0.02$. The gradient stops demanding infinite logit separation.
      #v(2pt)
      #tiny-note[Validate $epsilon$: too much smoothing can erase useful confidence or class-similarity structure.]
    ]

    #v(4pt)
    #section-title([5 · A disciplined recipe], accent: palette.green)
    #card(accent: palette.green)[
      #grid(
        columns: (auto, 1fr),
        row-gutter: 3pt,
        column-gutter: 4pt,
        [#label([1], color: palette.teal)], [fit a reproducible baseline; plot learning curves],
        [#label([2], color: palette.teal)], [add only task-valid augmentation],
        [#label([3], color: palette.orange)], [sweep weight decay on a log scale],
        [#label([4], color: palette.blue)], [add one targeted method if curves justify it],
        [#label([5], color: palette.green)], [freeze recipe and checkpoint; evaluate test once],
      )
    ]

    #v(4pt)
    #section-title([6 · Final audit], accent: palette.blue)
    #card(accent: palette.blue)[
      #check[Learning curves support the diagnosis.]
      #linebreak()
      #check[One intervention site and one named knob.]
      #linebreak()
      #check[Validation chose the strength and checkpoint.]
      #linebreak()
      #check[The best checkpoint was restored.]
      #linebreak()
      #check[Test stayed untouched until the end.]
      #linebreak()
      #check[Seeds, splits, schedule, and every knob were logged.]
    ]

    #v(4pt)
    #card(title: [One-line map], accent: palette.teal, fill: soft(palette.teal, amount: 96%))[
      #keyline([Weights], [ constrain parameters.], color: palette.orange)
      #linebreak()
      #keyline([Stopping], [ constrains training time.], color: palette.green)
      #linebreak()
      #keyline([Augmentation], [ expands valid views.], color: palette.teal)
      #linebreak()
      #keyline([Dropout], [ disrupts feature pathways.], color: palette.blue)
      #linebreak()
      #keyline([Smoothing], [ softens confidence targets.], color: palette.red)
    ]

    #v(4pt)
    #banner(accent: palette.orange)[
      If you have not plotted learning curves, diagnose before choosing a regularizer.
    ]
  ],
)
