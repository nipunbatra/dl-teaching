#import "../common/cheatsheet.typ": *

#set page(
  paper: "a4",
  margin: (top: 8mm, bottom: 9mm, x: 9mm),
  fill: white,
  footer: context align(center)[
    #text(size: 6.5pt, fill: palette.muted)[
      DL course · S.No. 3 · Likelihood to Loss in Practice · page #counter(page).display()
    ]
  ],
)
#set text(font: "IBM Plex Sans", size: 8pt, fill: palette.ink)
#set par(leading: 0.52em, justify: false)
#set list(indent: 10pt, body-indent: 4pt, spacing: 2pt)
#show math.equation: set text(font: "New Computer Modern Math", size: 8.4pt)

#sheet-title(
  [S.NO. 3 CHEATSHEET],
  [Choose an output head and loss],
  subtitle: [Work from the target's support and uncertainty, then use a stable implementation of its negative log-likelihood.],
)

#banner[
  Target semantics $arrow.r$ predictive distribution $arrow.r$ parameterization $arrow.r$ stable NLL.
]

#v(5pt)
#grid(
  columns: (1fr, 1fr),
  gutter: 7mm,
  [
    #section-title([1 · A practical model-selection map], accent: palette.blue)
    #card(accent: palette.blue, inset: 4.5pt)[
      #grid(
        columns: (23mm, 1fr, 1fr),
        column-gutter: 3pt,
        row-gutter: 3pt,
        [#text(weight: "bold")[Target]], [#text(weight: "bold")[Distribution]], [#text(weight: "bold")[Head / NLL]],
        [real value], [Gaussian], [$mu$ (and $sigma>0$) / MSE or Gaussian NLL],
        [robust real], [Laplace], [$mu$ (and $b>0$) / absolute-error NLL],
        [heavy-tail real], [Student-$t$], [$mu,s>0,nu>0$ / Student-$t$ NLL],
        [binary label], [Bernoulli], [one logit / BCE with logits],
        [one of $K$], [Categorical], [$K$ logits / cross-entropy],
        [count], [Poisson], [log-rate / Poisson NLL],
      )
      #v(2pt)
      #tiny-note[A loss can be optimized even when its likelihood story is inappropriate. The story is what tells you whether its behaviour matches the data.]
    ]

    #v(4pt)
    #section-title([2 · Parameterize constraints safely], accent: palette.teal)
    #card(accent: palette.teal)[
      Let the network emit unconstrained raw values.
      #v(3pt)
      #grid(
        columns: (26mm, 1fr),
        row-gutter: 3pt,
        column-gutter: 4pt,
        [$p in (0,1)$], [$p=sigma(z)$],
        [$bold(p) in Delta^(K-1)$], [$bold(p)="softmax"(bold(z))$],
        [$sigma,lambda,s>0$], [$"softplus"(a)+epsilon$ or model a log-scale],
        [$nu>0$], [$nu="softplus"(u)+epsilon$; often impose a safer lower bound],
      )
      #v(3pt)
      #tiny-note[Do not clamp learned probabilities as the primary model. Use a transformation whose range already obeys the distribution's constraints.]
    ]

    #v(4pt)
    #section-title([3 · Use logits, not rounded probabilities], accent: palette.orange)
    #card(accent: palette.orange)[
      Stable binary loss for logit $z$:
      #align(center)[$ell(z,y)=max(z,0)-z y+log(1+exp(-abs(z))).$]
      Stable multiclass loss:
      #align(center)[$ell(bold(z),c)=-z_c+"logsumexp"(bold(z)).$]
      These fused forms avoid explicitly computing $log(0)$ after sigmoid or softmax saturation.
    ]
  ],
  [
    #section-title([4 · Heteroscedastic regression], accent: palette.orange)
    #card(accent: palette.orange)[
      If uncertainty changes with input, predict both $mu_i$ and $sigma_i>0$:
      #align(center)[$ell_i=1/2 [(y_i-mu_i)^2/sigma_i^2 + log sigma_i^2]+C.$]
      The first term standardizes the residual; the second prevents the model from making every $sigma_i$ arbitrarily large.
      #v(3pt)
      #keyline([Validate:], [ interval coverage and sharpness, not just point-error MSE.], color: palette.orange)
    ]

    #v(4pt)
    #section-title([5 · Robust regression is a modelling choice], accent: palette.red)
    #card(accent: palette.red)[
      For residual $r=y-mu$:
      #grid(
        columns: (22mm, 1fr),
        row-gutter: 3pt,
        column-gutter: 4pt,
        [Gaussian], [$partial ell/(partial r) prop r$; influence grows with $abs(r)$],
        [Laplace], [$partial ell/(partial r) prop "sign"(r)$; constant magnitude],
        [Student-$t$], [influence eventually decreases for very large $abs(r)$],
      )
      #v(3pt)
      Inspect residuals. A heavy-tailed likelihood is appropriate when unusual observations are plausible, not as a substitute for fixing corrupt data.
    ]

    #v(4pt)
    #section-title([6 · Library semantics to verify], accent: palette.blue)
    #card(accent: palette.blue, inset: 4.5pt)[
      #grid(
        columns: (28mm, 1fr),
        row-gutter: 3pt,
        column-gutter: 4pt,
        [`BCEWithLogitsLoss`], [raw logits + float targets in $[0,1]$],
        [`CrossEntropyLoss`], [raw logits + integer class indices; no prior softmax],
        [`GaussianNLLLoss`], [mean + variance, not standard deviation],
        [`PoissonNLLLoss`], [check whether input is log-rate and whether constants are included],
        [`Distribution.log_prob`], [inspect event and batch shapes before reducing],
      )
      #v(2pt)
      #tiny-note[Names describe formulas, not tensor conventions. Read the API contract and test a hand-computed example.]
    ]

    #v(4pt)
    #banner(accent: palette.teal)[
      The output activation is part of the probabilistic model, not a cosmetic layer after the network.
    ]
  ],
)

#pagebreak()

#sheet-title(
  [S.NO. 3 CHEATSHEET],
  [Debug a likelihood-based training pipeline],
  subtitle: [Check mathematics, tensors and numerical behaviour in that order; a decreasing loss alone is not enough.],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 7mm,
  [
    #section-title([1 · First prove one example by hand], accent: palette.blue)
    #card(accent: palette.blue)[
      Pick one tiny prediction-target pair and compute the expected NLL.
      #v(3pt)
      #grid(
        columns: (1fr, 1fr),
        column-gutter: 4pt,
        row-gutter: 3pt,
        [binary $y=1,p=0.8$], [$-log 0.8 approx 0.223$],
        [3-class $c=2,p_c=0.7$], [$-log 0.7 approx 0.357$],
        [Gaussian $r=2,sigma=1$], [$r^2/(2sigma^2)=2$ plus constants],
      )
      #v(3pt)
      Match the library's unreduced output to the hand value before training a model.
    ]

    #v(4pt)
    #section-title([2 · Shape audit], accent: palette.teal)
    #card(accent: palette.teal)[
      #check[Batch axis and class / event axes are named.]
      #linebreak()
      #check[Broadcasting is intentional, not accidental.]
      #linebreak()
      #check[One categorical target is an index, not a length-$K$ float unless soft labels are intended.]
      #linebreak()
      #check[Per-pixel or per-token masks remove padded observations before reduction.]
      #linebreak()
      #check[Start with `reduction="none"` and inspect every contribution.]
    ]

    #v(4pt)
    #section-title([3 · Reduction and weighting], accent: palette.orange)
    #card(accent: palette.orange)[
      Decide what one training example contributes:
      #align(center)[$cal(L)=1/n sum_i w_i ell_i, quad w_i>=0.$]
      If an example contains many pixels or tokens, choose whether to average within each example or across all observations. The two objectives weight long examples differently.
      #v(2pt)
      #tiny-note[Log the numerator and denominator separately when using masks, class weights or variable-length sequences.]
    ]

    #v(4pt)
    #section-title([4 · Numerical checks], accent: palette.red)
    #card(accent: palette.red)[
      #check[Logits, NLL and gradients are finite on the first batch.]
      #linebreak()
      #check[Positive scales are bounded away from zero.]
      #linebreak()
      #check[No duplicated sigmoid / softmax before a fused loss.]
      #linebreak()
      #check[Extreme logits are tested, not only values near zero.]
    ]
  ],
  [
    #section-title([5 · Behavioural sanity tests], accent: palette.green)
    #card(accent: palette.green, inset: 4.5pt)[
      #grid(
        columns: (1fr, 1.35fr),
        column-gutter: 4pt,
        row-gutter: 3pt,
        [#text(weight: "bold")[Test]], [#text(weight: "bold")[Expected result]],
        [perfect direction], [loss falls as true-class logit increases or residual shrinks],
        [label permutation], [performance collapses toward chance],
        [constant baseline], [trained model beats mean / class-frequency predictor],
        [duplicate batch], [mean loss unchanged; summed loss doubles],
        [one outlier], [Gaussian reacts more strongly than Laplace / Student-$t$],
        [zero probability], [NLL tends to $+infinity$ for the observed event],
      )
    ]

    #v(4pt)
    #section-title([6 · Diagnose with the right residual], accent: palette.blue)
    #card(accent: palette.blue)[
      #keyline([Regression:], [ plot residuals against predictions and inputs; check spread and tails.], color: palette.blue)
      #v(3pt)
      #keyline([Classification:], [ compare confidence with empirical accuracy; inspect per-class NLL.], color: palette.orange)
      #v(3pt)
      #keyline([Counts:], [ compare variance with the mean; strong overdispersion can invalidate Poisson.], color: palette.teal)
      #v(3pt)
      A well-optimized misspecified likelihood can still give poor uncertainty and misleading gradients.
    ]

    #v(4pt)
    #section-title([7 · Debugging order], accent: palette.orange)
    #card(accent: palette.orange)[
      #grid(
        columns: (auto, 1fr),
        row-gutter: 3pt,
        column-gutter: 4pt,
        [#label([1], color: palette.orange)], [target meaning and support],
        [#label([2], color: palette.orange)], [distribution family and parameter constraints],
        [#label([3], color: palette.orange)], [one-example formula and tensor shapes],
        [#label([4], color: palette.orange)], [reduction, masking and weighting],
        [#label([5], color: palette.orange)], [numerical stability and gradients],
        [#label([6], color: palette.orange)], [validation metrics and residual diagnostics],
      )
    ]

    #v(4pt)
    #banner(accent: palette.green)[
      If the hand calculation, unreduced tensor and library result disagree, stop there: the training curve cannot rescue a broken objective.
    ]
  ],
)
