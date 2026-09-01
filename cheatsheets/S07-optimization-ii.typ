#import "../common/cheatsheet.typ": *

#set page(
  paper: "a4",
  margin: (top: 8mm, bottom: 9mm, x: 9mm),
  fill: white,
  footer: context align(center)[
    #text(size: 6.5pt, fill: palette.muted)[
      DL course · Lecture 7 · Optimization II · page #counter(page).display()
    ]
  ],
)
#set text(font: "IBM Plex Sans", size: 8pt, fill: palette.ink)
#set par(leading: 0.52em, justify: false)
#set list(indent: 10pt, body-indent: 4pt, spacing: 2pt)
#show math.equation: set text(font: "New Computer Modern Math", size: 8.4pt)

#sheet-title(
  [LECTURE 7 CHEATSHEET],
  [Momentum → RMSProp → Adam],
  subtitle: [Three ways to turn a noisy gradient into a useful update direction.],
)

#banner[Momentum remembers direction. RMSProp remembers scale. Adam remembers both.]

#v(5pt)
#grid(
  columns: (1fr, 1fr),
  gutter: 7mm,
  [
    #section-title([1 · Start with the optimizer contract], accent: palette.blue)
    #card(accent: palette.blue)[
      On minibatch $cal(B)_t$, compute
      #align(center)[$hat(bold(g))_t=nabla_theta cal(L)_(cal(B)_t)(bold(theta)_t).$]
      Every optimizer constructs a direction $bold(d)_t$, then applies
      #align(center)[$bold(theta)_(t+1)=bold(theta)_t-eta_t bold(d)_t.$]
      #tiny-note[The learning rate scales the move; the optimizer state changes its direction and coordinate-wise scale.]
    ]

    #v(4pt)
    #section-title([2 · EWMA is the shared memory primitive], accent: palette.teal)
    #card(accent: palette.teal)[
      #align(center)[$m_t=beta m_(t-1)+(1-beta)x_t, quad 0 <= beta < 1.$]
      Recent observations receive weights $(1-beta), (1-beta)beta, dots$
      #v(2pt)
      #keyline([Effective memory:], [ roughly $1/(1-beta)$ observations.], color: palette.teal)
      #linebreak()
      #keyline([Zero-start bias:], [ early $m_t$ is too small. Correct with $hat(m)_t=m_t/(1-beta^t)$.], color: palette.orange)
    ]

    #v(4pt)
    #section-title([3 · Momentum: consistent direction?], accent: palette.teal)
    #card(accent: palette.teal)[
      #align(center)[$
        bold(m)_(t+1)=beta_1 bold(m)_t+(1-beta_1)hat(bold(g))_t
      $]
      #align(center)[$bold(theta)_(t+1)=bold(theta)_t-eta bold(m)_(t+1).$]
      Signed gradients that keep the same sign reinforce one another; oscillating components cancel. When $beta_1=0$, this normalized convention becomes ordinary GD.
      #v(2pt)
      #tiny-note[PyTorch SGD uses an unnormalized momentum buffer, so its `lr` convention differs by a factor of $(1-beta_1)$ from the normalized rule above.]
    ]

    #v(4pt)
    #card(title: [Ravine intuition], accent: palette.blue, fill: soft(palette.blue, amount: 95%))[
      Momentum damps left-right reversals across steep walls while preserving progress along the shallow valley. Standardizing features may remove the ravine first.
    ]
  ],
  [
    #section-title([4 · RMSProp: large relative to usual?], accent: palette.green)
    #card(accent: palette.green)[
      #align(center)[$
        bold(v)_(t+1)=beta_2 bold(v)_t+(1-beta_2)hat(bold(g))_t^2
      $]
      #align(center)[$
        bold(theta)_(t+1)=bold(theta)_t-eta
        hat(bold(g))_t/(sqrt(bold(v)_(t+1))+epsilon)
      $]
      Squaring prevents sign cancellation. Dividing by the recent root-mean-square gives each coordinate its own ruler.
      #v(2pt)
      #tiny-note[$bold(v)$ is a second raw moment, not a centred variance. Operations are elementwise.]
    ]

    #v(4pt)
    #section-title([5 · Adam: direction and scale], accent: palette.orange)
    #card(accent: palette.orange)[
      #align(center)[$
        bold(m)_(t+1)=beta_1 bold(m)_t+(1-beta_1)hat(bold(g))_t
      $]
      #align(center)[$
        bold(v)_(t+1)=beta_2 bold(v)_t+(1-beta_2)hat(bold(g))_t^2
      $]
      #align(center)[$
        hat(bold(m))_(t+1)=bold(m)_(t+1)/(1-beta_1^(t+1)),
        quad hat(bold(v))_(t+1)=bold(v)_(t+1)/(1-beta_2^(t+1))
      $]
      #align(center)[$
        bold(theta)_(t+1)=bold(theta)_t-eta
        hat(bold(m))_(t+1)/(sqrt(hat(bold(v))_(t+1))+epsilon).
      $]
      Bias correction matters most during the first updates, when both memories start at zero.
    ]

    #v(4pt)
    #section-title([6 · What each method stores], accent: palette.blue)
    #card(accent: palette.blue, inset: 4.5pt)[
      #grid(
        columns: (0.72fr, 1fr, 1.15fr),
        column-gutter: 3pt,
        row-gutter: 3pt,
        [#text(weight: "bold", fill: palette.muted)[Method]], [#text(weight: "bold", fill: palette.muted)[State]], [#text(weight: "bold", fill: palette.muted)[Question]],
        [GD], [none], [what is the gradient now?],
        [Momentum], [$bold(m)$], [which direction persists?],
        [RMSProp], [$bold(v)$], [which coordinates run large?],
        [Adam], [$bold(m),bold(v)$], [which direction, at what scale?],
      )
    ]

    #v(4pt)
    #banner(accent: palette.green)[
      Adaptive does not mean parameter-free: tune the learning rate for the optimizer and problem.
    ]
  ],
)

#pagebreak()

#sheet-title(
  [LECTURE 7 CHEATSHEET],
  [Use adaptive optimizers deliberately],
  subtitle: [State initialization, step order, diagnostics, and fair comparisons.],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 7mm,
  [
    #section-title([1 · One correct Adam step], accent: palette.orange)
    #card(accent: palette.orange)[
      #grid(
        columns: (auto, 1fr),
        row-gutter: 2.5pt,
        column-gutter: 4pt,
        [#label([1], color: palette.orange)], [compute the current minibatch gradient],
        [#label([2], color: palette.orange)], [update first-moment state $bold(m)$],
        [#label([3], color: palette.orange)], [update second-moment state $bold(v)$],
        [#label([4], color: palette.orange)], [bias-correct using update count $t+1$],
        [#label([5], color: palette.orange)], [divide $hat(bold(m))$ by $sqrt(hat(bold(v)))+epsilon$],
        [#label([6], color: palette.orange)], [apply the learning-rate-scaled parameter move],
      )
      #v(2pt)
      Initialize $bold(m)=bold(0)$ and $bold(v)=bold(0)$ once—not at every minibatch.
    ]

    #v(4pt)
    #section-title([2 · Defaults are starting points], accent: palette.blue)
    #card(accent: palette.blue, inset: 4.5pt)[
      #grid(
        columns: (0.7fr, 0.85fr, 1.6fr),
        column-gutter: 3pt,
        row-gutter: 3pt,
        [#text(weight: "bold")[Knob]], [#text(weight: "bold")[Typical]], [#text(weight: "bold")[Effect]],
        [$eta$], [$10^(-3)$ Adam], [global step scale; tune first],
        [$beta_1$], [$0.9$], [longer signed-direction memory as it rises],
        [$beta_2$], [$0.999$], [longer squared-gradient memory as it rises],
        [$epsilon$], [$10^(-8)$], [numerical floor; can affect scale in low precision],
      )
      #v(2pt)
      #tiny-note[The useful learning-rate range depends on architecture, normalization, batch size, loss scale, and optimizer convention.]
    ]

    #v(4pt)
    #section-title([3 · PyTorch training order], accent: palette.green)
    #card(accent: palette.green)[
      #raw(block: true, lang: "python", "opt.zero_grad(set_to_none=True)\npred = model(x)\nloss = criterion(pred, y)\nloss.backward()\nopt.step()")
      #v(2pt)
      Gradients accumulate unless cleared. The optimizer owns its per-parameter state; the model owns parameters and gradients.
    ]

    #v(4pt)
    #card(title: [AdamW note], accent: palette.teal, fill: soft(palette.teal, amount: 95%))[
      AdamW applies weight decay as a separate shrink instead of mixing it into the adaptively scaled gradient. Treat the decay coefficient as a tuned regularization knob.
    ]
  ],
  [
    #section-title([4 · Diagnose the symptom], accent: palette.red)
    #card(accent: palette.red, inset: 4.5pt)[
      #grid(
        columns: (1.05fr, 1.35fr),
        column-gutter: 4pt,
        row-gutter: 3pt,
        [#text(weight: "bold")[Symptom]], [#text(weight: "bold")[First checks]],
        [loss explodes / NaN], [lower $eta$; inspect data, loss scale, logits, and gradients],
        [zigzag in a ravine], [standardize; try momentum; reduce $eta$],
        [one coordinate dominates], [check feature scale; consider RMSProp/Adam],
        [no early progress], [verify gradients/state; raise $eta$ carefully],
        [good train, poor validation], [this is generalization—not an optimizer-only problem],
      )
    ]

    #v(4pt)
    #section-title([5 · Compare optimizers fairly], accent: palette.teal)
    #card(accent: palette.teal)[
      #check[Same data split, shuffling policy, initialization, and model.]
      #linebreak()
      #check[Same update budget and logging frequency.]
      #linebreak()
      #check[Tune each optimizer's learning rate separately.]
      #linebreak()
      #check[Report validation performance, not only training loss.]
      #linebreak()
      #check[Run multiple seeds when differences are small.]
      #linebreak()
      #check[Log gradient norms, update norms, and optimizer state.]
    ]

    #v(4pt)
    #section-title([6 · Invariants worth asserting], accent: palette.blue)
    #card(accent: palette.blue)[
      #keyline([Shapes:], [ $theta$, $g$, $m$, and $v$ match elementwise.], color: palette.blue)
      #linebreak()
      #keyline([Finiteness:], [ loss, gradients, and states contain no `NaN`/`Inf`.], color: palette.red)
      #linebreak()
      #keyline([Clock:], [ bias correction uses the number of parameter updates.], color: palette.orange)
      #linebreak()
      #keyline([State:], [ checkpoint optimizer state when resuming training.], color: palette.green)
    ]

    #v(4pt)
    #banner(accent: palette.orange)[
      Optimizers change the path—not the objective. Always inspect both training dynamics and validation behaviour.
    ]
  ],
)
