#import "../common/cheatsheet.typ": *

#set page(
  paper: "a4",
  margin: (top: 8mm, bottom: 9mm, x: 9mm),
  fill: white,
  footer: context align(center)[
    #text(size: 6.5pt, fill: palette.muted)[
      DL course · Lecture 9 · Valid Outputs & Learning-Rate Schedules · page #counter(page).display()
    ]
  ],
)
#set text(font: "IBM Plex Sans", size: 8pt, fill: palette.ink)
#set par(leading: 0.52em, justify: false)
#set list(indent: 10pt, body-indent: 4pt, spacing: 2pt)
#show math.equation: set text(font: "New Computer Modern Math", size: 8.4pt)

#sheet-title(
  [LECTURE 9 CHEATSHEET],
  [Make invalid outputs unreachable],
  subtitle: [Learn unrestricted coordinates; transform them into valid model parameters.],
)

#banner[$bold(z) in RR^d arrow.r theta=phi(bold(z)) in cal(C)$: choose the map from the parameter's domain.]

#v(5pt)
#grid(
  columns: (1fr, 1fr),
  gutter: 7mm,
  [
    #section-title([1 · Constraint map], accent: palette.blue)
    #card(accent: palette.blue, inset: 4.5pt)[
      #grid(
        columns: (1.05fr, 1fr, 1.15fr),
        column-gutter: 3pt,
        row-gutter: 3pt,
        [#text(weight: "bold")[Need]], [#text(weight: "bold")[Raw]], [#text(weight: "bold")[Transform]],
        [positive scale], [$u in RR$], [$sigma="softplus"(u)+epsilon$],
        [probability], [$z in RR$], [$p="sigmoid"(z)$],
        [simplex], [$bold(z) in RR^K$], [$bold(p)="softmax"(bold(z))$],
        [PD covariance], [raw triangle], [positive-diagonal $L$; $Sigma=L L^T$],
      )
      #v(2pt)
      The optimizer never sees a constrained coordinate directly; the chain rule carries gradients through $phi$.
    ]

    #v(4pt)
    #section-title([2 · Positive scales], accent: palette.teal)
    #card(accent: palette.teal)[
      #align(center)[$"softplus"(u)=log(1+e^u)>0.$]
      #align(center)[$"softplus"'(u)="sigmoid"(u).$]
      Use $sigma="softplus"(u)+epsilon$ for a strictly positive numerical floor. Compared with $e^u$, softplus grows approximately linearly for large positive $u$, so raw steps are gentler.
      #v(2pt)
      #tiny-note[Extremely negative raw scales can still learn slowly because the sigmoid derivative approaches zero. Initialize sensibly.]
    ]

    #v(4pt)
    #section-title([3 · Binary probabilities], accent: palette.orange)
    #card(accent: palette.orange)[
      #align(center)[$p="sigmoid"(z)=1/(1+e^(-z)), quad p in (0,1).$]
      For binary cross-entropy,
      #align(center)[$(partial ell)/(partial z)=p-y.$]
      Train from raw logits with a fused primitive; compute probabilities only for interpretation.
      #v(2pt)
      #raw(block: true, lang: "python", "loss = F.binary_cross_entropy_with_logits(z, y)\np = z.sigmoid()  # reporting")
    ]

    #v(4pt)
    #card(title: [Shape guard], accent: palette.red, fill: soft(palette.red, amount: 95%))[
      Assert `y.shape == logits.shape` for elementwise binary loss. `[N]` and `[N,1]` can broadcast into an unintended `[N,N]` calculation.
    ]
  ],
  [
    #section-title([4 · Multiclass probabilities], accent: palette.blue)
    #card(accent: palette.blue)[
      #align(center)[$p_k=e^(z_k)/(sum_j e^(z_j)), quad p_k>0, quad sum_k p_k=1.$]
      Adding the same constant to every logit leaves probabilities unchanged. For stable computation, subtract $m=max_j z_j$:
      #align(center)[$p_k=e^(z_k-m)/(sum_j e^(z_j-m)).$]
      Since $z_j-m <= 0$, no exponential overflows and the denominator contains at least one $e^0=1$.
    ]

    #v(4pt)
    #section-title([5 · Stay in log space for training], accent: palette.green)
    #card(accent: palette.green)[
      Stable cross-entropy for target class $t$:
      #align(center)[$
        ell(bold(z),t)=-(z_t-m)+log(sum_j e^(z_j-m)).
      $]
      #raw(block: true, lang: "python", "# logits: [N, K], target: [N] integer classes\nloss = F.cross_entropy(logits, target)\nlog_p = F.log_softmax(logits, dim=-1)")
      #v(2pt)
      #tiny-note[Do not compute `log(softmax(logits))` as two separate operations. Stability cannot repair `NaN` or infinite input logits.]
    ]

    #v(4pt)
    #section-title([6 · Positive-definite covariance], accent: palette.orange)
    #card(accent: palette.orange)[
      Decode raw outputs into a lower-triangular factor $L$. Make each diagonal entry positive, then set
      #align(center)[$Sigma=L L^T.$]
      For any nonzero vector $bold(a)$,
      #align(center)[$bold(a)^T Sigma bold(a)=norm(L^T bold(a))^2>0$]
      when $L$ is nonsingular. In PyTorch distributions, pass the factor directly when a `scale_tril` interface is available.
    ]

    #v(4pt)
    #section-title([7 · Domain audit], accent: palette.teal)
    #card(accent: palette.teal)[
      #check[Write the legal parameter set before choosing the output head.]
      #linebreak()
      #check[Keep the raw output unrestricted.]
      #linebreak()
      #check[Use a differentiable map with a useful slope.]
      #linebreak()
      #check[Use fused stable losses on logits.]
      #linebreak()
      #check[Assert shapes, dtypes, and finite values.]
    ]
  ],
)

#pagebreak()

#sheet-title(
  [LECTURE 9 CHEATSHEET],
  [Schedule the learning rate],
  subtitle: [Large steps help early; smaller steps give noisy optimization more control near a minimum.],
)

#banner[$bold(theta)_(t+1)=bold(theta)_t-eta_t bold(d)_t$: the optimizer builds $bold(d)_t$; the schedule chooses its scalar scale.]

#v(5pt)
#grid(
  columns: (1fr, 1fr),
  gutter: 7mm,
  [
    #section-title([1 · Why decay helps], accent: palette.blue)
    #card(accent: palette.blue)[
      A minibatch gradient can be written
      #align(center)[$bold(g)_cal(B)(theta)=nabla cal(L)(theta)+bold(epsilon)_cal(B).$]
      Near a minimum, the true gradient becomes small while minibatch noise remains. The update noise has scale approximately $eta_t bold(epsilon)_cal(B)$, so shrinking $eta_t$ reduces the bounce.
      #v(2pt)
      #tiny-note[With exact full-batch gradients, decay is still useful for finer late-stage steps; the noise argument is strongest for SGD/minibatches.]
    ]

    #v(4pt)
    #section-title([2 · Four useful time-based rules], accent: palette.teal)
    #card(accent: palette.teal, inset: 4.5pt)[
      #keyline([Step:], [$ eta_t=eta_0 gamma^(floor(t/S))$], color: palette.blue)
      #linebreak()
      #keyline([Exponential:], [$ eta_t=eta_0 gamma^t$], color: palette.orange)
      #linebreak()
      #keyline([Inverse-time:], [$ eta_t=eta_0/(1+k t)$], color: palette.green)
      #linebreak()
      #keyline([Cosine:], [$ eta_t=eta_"min"+1/2(eta_"max"-eta_"min")(1+cos(pi t/T))$], color: palette.teal)
      #v(2pt)
      #tiny-note[Define the clock: $t$ may mean update, epoch, or scheduler call. The formula is incomplete without that convention.]
    ]

    #v(4pt)
    #section-title([3 · Warm up, then decay], accent: palette.orange)
    #card(accent: palette.orange)[
      During the first $W$ updates,
      #align(center)[$eta_t=eta_"peak" t/W.$]
      Then switch to the chosen decay rule. Warmup is useful when startup at the peak rate causes loss spikes, divergence, or unstable large updates.
      #v(2pt)
      #tiny-note[Warmup is not mandatory. Add it to solve an observed startup problem, and count it in the total update budget.]
    ]

    #v(4pt)
    #card(title: [Fixed budget], accent: palette.blue, fill: soft(palette.blue, amount: 95%))[
      Cosine decay is a strong simple default when the total number of updates is known. Choose the endpoint and minimum rate explicitly.
    ]
  ],
  [
    #section-title([4 · Time-based or metric-based?], accent: palette.green)
    #card(accent: palette.green, inset: 4.5pt)[
      #grid(
        columns: (0.9fr, 1.45fr),
        column-gutter: 4pt,
        row-gutter: 3pt,
        [#text(weight: "bold")[Situation]], [#text(weight: "bold")[Simple choice]],
        [fixed run length], [cosine or a predeclared step schedule],
        [validation decides], [reduce on plateau after validation],
        [fragile startup], [warmup, then time-based decay],
        [short exploratory run], [constant rate for interpretability],
      )
      #v(2pt)
      A plateau scheduler consumes a validation metric; it is part of model selection and must never watch the test set.
    ]

    #v(4pt)
    #section-title([5 · Correct call order], accent: palette.orange)
    #card(accent: palette.orange)[
      #raw(block: true, lang: "python", "for x, y in train_loader:\n    opt.zero_grad(set_to_none=True)\n    loss = criterion(model(x), y)\n    loss.backward()\n    opt.step()\n\nsched.step()  # if the clock is one epoch")
      #v(2pt)
      For `ReduceLROnPlateau`, evaluate first and call `sched.step(val_loss)`. Log the rate used for each update.
    ]

    #v(4pt)
    #section-title([6 · Schedule audit], accent: palette.blue)
    #card(accent: palette.blue)[
      #check[The optimizer's base learning rate was tuned first.]
      #linebreak()
      #check[The schedule clock and total horizon are explicit.]
      #linebreak()
      #check[Warmup and decay share one continuous rate trace.]
      #linebreak()
      #check[Checkpoint resume restores optimizer, scheduler, and clock.]
      #linebreak()
      #check[Comparisons use the same update budget.]
      #linebreak()
      #check[Validation—not test—chooses metric-based changes.]
    ]

    #v(4pt)
    #banner(accent: palette.teal)[
      Choose the simplest schedule that matches the run; add complexity only for a diagnosed failure mode.
    ]
  ],
)
