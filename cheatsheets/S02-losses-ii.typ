#import "../common/cheatsheet.typ": *

#set page(
  paper: "a4",
  margin: (top: 8mm, bottom: 9mm, x: 9mm),
  fill: white,
  footer: context align(center)[
    #text(size: 6.5pt, fill: palette.muted)[
      DL course · S.No. 2 · Where Deep Learning Losses Come From II · page #counter(page).display()
    ]
  ],
)
#set text(font: "IBM Plex Sans", size: 8pt, fill: palette.ink)
#set par(leading: 0.52em, justify: false)
#set list(indent: 10pt, body-indent: 4pt, spacing: 2pt)
#show math.equation: set text(font: "New Computer Modern Math", size: 8.4pt)

#sheet-title(
  [S.NO. 2 CHEATSHEET],
  [Derive the loss from the likelihood],
  subtitle: [The loss is not an arbitrary penalty: it is the negative log-score of a predictive distribution.],
)

#banner[
  Distribution assumption $arrow.r$ per-example NLL $arrow.r$ familiar training loss.
]

#v(5pt)
#grid(
  columns: (1fr, 1fr),
  gutter: 7mm,
  [
    #section-title([1 · The universal derivation], accent: palette.blue)
    #card(accent: palette.blue)[
      Conditional independence gives
      #align(center)[$p(cal(D)|bold(theta))=product_i p_theta(y_i|bold(x)_i).$]
      Taking logs and changing maximization to minimization gives
      #align(center)[$hat(theta)_"MLE"=limits(arg min)_theta -sum_i log p_theta(y_i|bold(x)_i).$]
      Define one-example loss
      #align(center)[$ell_i(theta)=-log p_theta(y_i|bold(x)_i).$]
      The dataset objective is the sum or mean of these contributions.
    ]

    #v(4pt)
    #section-title([2 · Gaussian likelihood gives squared error], accent: palette.teal)
    #card(accent: palette.teal)[
      Assume
      #align(center)[$Y_i|bold(x)_i tilde cal(N)(mu_i,sigma^2), quad mu_i=f_theta(bold(x)_i).$]
      Then
      #align(center)[$-log p(y_i|bold(x)_i)=1/(2sigma^2)(y_i-mu_i)^2 + log sigma + C.$]
      With fixed $sigma$,
      #align(center)[$hat(theta)_"MLE"=limits(arg min)_theta sum_i (y_i-f_theta(bold(x)_i))^2.$]
      #tiny-note[MSE is Gaussian NLL up to positive scale and additive constants. Learning $sigma$ requires keeping the $log sigma$ term.]
    ]

    #v(4pt)
    #section-title([3 · Residual model chooses the penalty], accent: palette.orange)
    #card(accent: palette.orange, inset: 4.5pt)[
      Let $r=y-hat(y)$.
      #grid(
        columns: (28mm, 1fr, 1fr),
        column-gutter: 3pt,
        row-gutter: 3pt,
        [#text(weight: "bold")[Noise]], [#text(weight: "bold")[NLL in $r$]], [#text(weight: "bold")[Effect]],
        [Gaussian], [$r^2$], [large errors dominate],
        [Laplace], [$abs(r)$], [constant influence],
        [Student-$t$], [$log(1+r^2/(nu s^2))$], [heavy-tail robustness],
        [Uniform], [$0$ inside; $infinity$ outside], [hard support],
      )
      #v(2pt)
      #tiny-note[Constants and positive scale factors are omitted in the compact table.]
    ]
  ],
  [
    #section-title([4 · Bernoulli likelihood gives BCE], accent: palette.blue)
    #card(accent: palette.blue)[
      For $y in {0,1}$ and predicted $p=P(Y=1|bold(x))$,
      #align(center)[$p(y|bold(x))=p^y(1-p)^(1-y).$]
      Therefore
      #align(center)[$ell_"BCE"=-[y log p+(1-y)log(1-p)].$]
      If $y=1$, $p=0.8$ gives $ell approx 0.22$ while $p=0.2$ gives $ell approx 1.61$.
      #v(2pt)
      #tiny-note[Use a logit $z in RR$ and a fused BCE-with-logits operation for numerical stability.]
    ]

    #v(4pt)
    #section-title([5 · Categorical likelihood gives cross-entropy], accent: palette.orange)
    #card(accent: palette.orange)[
      For mutually exclusive classes, logits $bold(z)$ define
      #align(center)[$p_k=(exp z_k)/(sum_j exp z_j).$]
      If the observed class is $c$,
      #align(center)[$ell_"CE"=-log p_c=-z_c+log sum_j exp z_j.$]
      Cross-entropy depends on the probability assigned to the true class. Confident mistakes are expensive because $-log p_c arrow.r infinity$ as $p_c arrow.r 0$.
    ]

    #v(4pt)
    #section-title([6 · Match observation, output and loss], accent: palette.green)
    #card(accent: palette.green, inset: 4.5pt)[
      #grid(
        columns: (25mm, 1fr, 1fr),
        column-gutter: 3pt,
        row-gutter: 3pt,
        [#text(weight: "bold")[Target]], [#text(weight: "bold")[Model output]], [#text(weight: "bold")[NLL]],
        [real], [$mu$; optionally $sigma>0$], [Gaussian / robust],
        [binary], [one unconstrained logit], [Bernoulli BCE],
        [one of $K$], [$K$ unconstrained logits], [categorical CE],
        [count], [positive rate $lambda$], [Poisson NLL],
      )
    ]

    #v(4pt)
    #banner(accent: palette.teal)[
      Before optimizing a loss, say aloud which random variable and distribution make it the right score.
    ]
  ],
)

#pagebreak()

#sheet-title(
  [S.NO. 2 CHEATSHEET],
  [From MLE to MAP and regularization],
  subtitle: [Likelihood asks for data fit; a prior adds an explicit preference among plausible parameter settings.],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 7mm,
  [
    #section-title([1 · Bayes combines data and prior], accent: palette.blue)
    #card(accent: palette.blue)[
      #align(center)[$p(theta|cal(D))=(p(cal(D)|theta)p(theta))/(p(cal(D))).$]
      #grid(
        columns: (24mm, 1fr),
        row-gutter: 3pt,
        column-gutter: 4pt,
        [likelihood], [how the fixed data score each $theta$],
        [prior], [pre-data preference over $theta$],
        [posterior], [updated distribution after observing data],
        [evidence], [normalizer; independent of $theta$],
      )
    ]

    #v(4pt)
    #section-title([2 · MAP is loss plus regularizer], accent: palette.teal)
    #card(accent: palette.teal)[
      #align(center)[$hat(theta)_"MAP"=limits(arg max)_theta p(theta|cal(D)).$]
      Dropping the evidence and taking negative logs,
      #align(center)[$hat(theta)_"MAP"=limits(arg min)_theta [underbrace(-log p(cal(D)|theta), "NLL") + underbrace(-log p(theta), "prior penalty")].$]
      #v(2pt)
      #keyline([MLE], [ uses only the likelihood.], color: palette.orange)
      #linebreak()
      #keyline([MAP], [ returns one parameter vector at the posterior mode.], color: palette.green)
    ]

    #v(4pt)
    #section-title([3 · Gaussian prior gives $L_2$], accent: palette.orange)
    #card(accent: palette.orange)[
      If $theta tilde cal(N)(bold(0),tau^2 bold(I))$,
      #align(center)[$-log p(theta)=1/(2tau^2) norm(theta)_2^2+C.$]
      Hence
      #align(center)[$cal(L)_"MAP"="NLL"+lambda norm(theta)_2^2, quad lambda=1/(2tau^2).$]
      Smaller $tau$ means a narrower prior, larger $lambda$, and stronger shrinkage toward zero.
    ]

    #v(4pt)
    #section-title([4 · Laplace prior gives $L_1$], accent: palette.red)
    #card(accent: palette.red)[
      For independent zero-centred Laplace parameters with scale $a$,
      #align(center)[$-log p(theta)=1/a norm(theta)_1+C.$]
      The corner at zero can threshold coefficients exactly to zero. $L_2$ smoothly shrinks; $L_1$ can create sparsity.
    ]
  ],
  [
    #section-title([5 · One-dimensional MAP shrinkage], accent: palette.green)
    #card(accent: palette.green, fill: soft(palette.green, amount: 96%))[
      Approximate the likelihood by
      #align(center)[$p(cal(D)|theta) prop exp(-(theta-z)^2/(2s^2))$]
      with $z=hat(theta)_"MLE"$, and use $theta tilde cal(N)(0,tau^2)$. Minimizing the two quadratic terms gives
      #align(center)[$hat(theta)_"MAP"=underbrace(tau^2/(tau^2+s^2), alpha in (0,1)) z.$]
      If $z=0.6$ and $s=tau$, then $hat(theta)_"MAP"=0.3$.
    ]

    #v(4pt)
    #section-title([6 · Scaling: keep the statistical meaning], accent: palette.blue)
    #card(accent: palette.blue)[
      The exact MAP objective uses #text(weight: "bold")[summed] NLL plus the full prior penalty:
      #align(center)[$sum_i ell_i(theta)+lambda_"sum" R(theta).$]
      If code uses mean NLL, divide the whole objective by $n$:
      #align(center)[$1/n sum_i ell_i(theta)+lambda_"mean" R(theta), quad lambda_"mean"=lambda_"sum"/n.$]
      #tiny-note[Changing batch size should not silently change the intended regularization strength. Document whether a library returns a sum or mean.]
    ]

    #v(4pt)
    #section-title([7 · What stronger priors do], accent: palette.orange)
    #card(accent: palette.orange, inset: 4.5pt)[
      #grid(
        columns: (1fr, 1fr),
        column-gutter: 4pt,
        row-gutter: 3pt,
        [#text(weight: "bold")[More data]], [likelihood sharpens; MAP often approaches MLE],
        [#text(weight: "bold")[Narrower prior]], [posterior mode moves farther toward preferred values],
        [#text(weight: "bold")[Correlated prior]], [preferred directions become elliptical, not axis-aligned],
        [#text(weight: "bold")[Prior centre $m$]], [shrinkage is toward $m$, not automatically toward zero],
      )
    ]

    #v(4pt)
    #section-title([8 · Final audit], accent: palette.green)
    #card(accent: palette.green)[
      #check[Each loss is tied to an explicit likelihood.]
      #linebreak()
      #check[Constants are dropped only when they do not affect the optimized parameters.]
      #linebreak()
      #check[Logits and constrained distribution parameters are not confused.]
      #linebreak()
      #check[Prior family and scale are stated.]
      #linebreak()
      #check[Sum-versus-mean scaling is explicit.]
    ]

    #v(4pt)
    #banner(accent: palette.orange)[
      Likelihood gives the data loss. Prior gives the regularizer. Their posterior mode gives MAP.
    ]
  ],
)
