#import "../common/cheatsheet.typ": *

#set page(
  paper: "a4",
  margin: (top: 8mm, bottom: 9mm, x: 9mm),
  fill: white,
  footer: context align(center)[
    #text(size: 6.5pt, fill: palette.muted)[
      DL course · S.No. 1 · Where Deep Learning Losses Come From I · page #counter(page).display()
    ]
  ],
)
#set text(font: "IBM Plex Sans", size: 8pt, fill: palette.ink)
#set par(leading: 0.52em, justify: false)
#set list(indent: 10pt, body-indent: 4pt, spacing: 2pt)
#show math.equation: set text(font: "New Computer Modern Math", size: 8.4pt)

#sheet-title(
  [S.NO. 1 CHEATSHEET],
  [From outcomes to predictive distributions],
  subtitle: [A model should say which values are possible and how plausible each one is.],
)

#banner[
  A neural network maps an input to the parameters of a distribution for the target.
]

#v(5pt)
#grid(
  columns: (1fr, 1fr),
  gutter: 7mm,
  [
    #section-title([1 · Start with the random variable], accent: palette.blue)
    #card(accent: palette.blue)[
      The #text(weight: "bold")[support] $cal(S)_Y$ is the set of values the target can take. The support should match the task before you choose a loss.
      #v(3pt)
      #grid(
        columns: (22mm, 1fr),
        row-gutter: 2.5pt,
        column-gutter: 4pt,
        [#label([BINARY], color: palette.blue)], [$Y in {0,1}$],
        [#label([CLASS], color: palette.teal)], [$Y in {1,dots,K}$],
        [#label([COUNT], color: palette.orange)], [$Y in {0,1,2,dots}$],
        [#label([REAL], color: palette.green)], [$Y in RR$],
      )
      #v(3pt)
      #tiny-note[Impossible targets should receive zero probability or density. A support mismatch is a modelling error, not an optimizer problem.]
    ]

    #v(4pt)
    #section-title([2 · Mass for discrete, density for continuous], accent: palette.teal)
    #card(accent: palette.teal)[
      #grid(
        columns: (19mm, 1fr),
        row-gutter: 3pt,
        column-gutter: 4pt,
        [#label([PMF], color: palette.teal)], [$p_Y(y)=P(Y=y)$ and $sum_(y in cal(S)_Y) p_Y(y)=1$.],
        [#label([PDF], color: palette.orange)], [$p_Y(y)>=0$ and $integral_(cal(S)_Y) p_Y(y) dif y=1$.],
      )
      #v(3pt)
      For continuous $Y$,
      #align(center)[$P(a <= Y <= b)=integral_a^b p_Y(y) dif y.$]
      A density value can exceed $1$; it is the #text(weight: "bold")[area] that is a probability.
    ]

    #v(4pt)
    #section-title([3 · Four distribution families], accent: palette.orange)
    #card(accent: palette.orange, inset: 4.5pt)[
      #grid(
        columns: (25mm, 1fr, 27mm),
        column-gutter: 3pt,
        row-gutter: 2.5pt,
        [#text(weight: "bold", fill: palette.muted)[Family]],
        [#text(weight: "bold", fill: palette.muted)[Parameters]],
        [#text(weight: "bold", fill: palette.muted)[Support]],
        [Bernoulli], [$p in [0,1]$], [${0,1}$],
        [Categorical], [$p_1,dots,p_K$], [${1,dots,K}$],
        [Poisson], [$lambda>0$], [nonnegative integers],
        [Gaussian], [$mu in RR, sigma>0$], [$RR$],
      )
    ]
  ],
  [
    #section-title([4 · Read conditional model notation], accent: palette.orange)
    #card(accent: palette.orange)[
      #align(center)[$p_(bold(theta))(y | bold(x))$]
      #v(3pt)
      #grid(
        columns: (20mm, 1fr),
        row-gutter: 3pt,
        column-gutter: 4pt,
        [$bold(x)$], [one observed input],
        [$y$], [one possible or observed target],
        [$bold(theta)$], [all learned weights and biases],
        [$p_theta(y|bold(x))$], [probability or density assigned to $y$, given $bold(x)$],
      )
      #v(3pt)
      #tiny-note[During supervised training the observed inputs are treated as fixed; the model describes the conditional target distribution.]
    ]

    #v(4pt)
    #section-title([5 · The network predicts distribution parameters], accent: palette.green)
    #card(accent: palette.green)[
      #grid(
        columns: (22mm, 1fr),
        row-gutter: 3pt,
        column-gutter: 4pt,
        [#label([REG], color: palette.green)], [$bold(x) arrow.r (mu_theta(bold(x)), sigma_theta(bold(x)))$],
        [#label([BINARY], color: palette.blue)], [$bold(x) arrow.r p_theta(bold(x))$],
        [#label([K-CLASS], color: palette.orange)], [$bold(x) arrow.r (p_(theta,1)(bold(x)),dots,p_(theta,K)(bold(x)))$],
      )
      #v(3pt)
      A point prediction is only a summary: often the conditional mean or mode. Training scores the #text(weight: "bold")[full distribution] at the observed target.
    ]

    #v(4pt)
    #section-title([6 · Prediction and likelihood ask different questions], accent: palette.blue)
    #card(accent: palette.blue, fill: soft(palette.blue, amount: 96%))[
      #keyline([Prediction:], [ fix $bold(theta)$ and $bold(x)$; vary the possible outcome $y$.], color: palette.blue)
      #v(3pt)
      #keyline([Likelihood:], [ fix the observed data; vary candidate parameters $bold(theta)$.], color: palette.orange)
      #v(3pt)
      The same expression $p_theta(y|bold(x))$ is read differently because a different quantity is allowed to vary.
    ]

    #v(4pt)
    #banner(accent: palette.teal)[
      Choose the target variable first, then its support, then a distribution family, then the neural outputs that parameterize it.
    ]
  ],
)

#pagebreak()

#sheet-title(
  [S.NO. 1 CHEATSHEET],
  [Build a dataset likelihood],
  subtitle: [Likelihood ranks parameter settings by how well they explain the outcomes that were actually observed.],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 7mm,
  [
    #section-title([1 · Score one fixed observation], accent: palette.blue)
    #card(accent: palette.blue)[
      For one pair $(bold(x)^star,y^star)$, the likelihood contribution is
      #align(center)[$L_i(bold(theta))=p_(bold(theta))(y^star|bold(x)^star).$]
      A larger value means that candidate $bold(theta)$ explains this fixed observation better.
      #v(3pt)
      #tiny-note[For a continuous target this is a density. Comparing candidates is valid even though the density is not itself an event probability.]
    ]

    #v(4pt)
    #section-title([2 · Independence gives a product], accent: palette.teal)
    #card(accent: palette.teal)[
      For conditionally independent examples,
      #align(center)[$L(bold(theta))=p(cal(D)|bold(theta))=product_(i=1)^n p_(bold(theta))(y_i|bold(x)_i).$]
      #v(2pt)
      #keyline([Independent], [ means multiply the per-example factors.], color: palette.teal)
      #linebreak()
      #keyline([Identically distributed], [ means reuse the same conditional model form.], color: palette.blue)
      #v(2pt)
      These are separate assumptions; either can fail without the other.
    ]

    #v(4pt)
    #section-title([3 · Tiny Bernoulli comparison], accent: palette.orange)
    #card(accent: palette.orange)[
      Compare $theta=0.5$ and $theta=0.8$ where $P(H|theta)=theta$.
      #v(3pt)
      #grid(
        columns: (20mm, 1fr, 1fr),
        column-gutter: 3pt,
        row-gutter: 3pt,
        [#text(weight: "bold")[Data]], [$L(0.5)$], [$L(0.8)$],
        [$H,T$], [$0.5(0.5)=0.25$], [$0.8(0.2)=0.16$],
        [$H,H$], [$0.5^2=0.25$], [$0.8^2=0.64$],
      )
      #v(3pt)
      The candidate ranking changes when the fixed observed data change.
    ]

    #v(4pt)
    #section-title([4 · MLE names the winner], accent: palette.green)
    #card(accent: palette.green, fill: soft(palette.green, amount: 96%))[
      #align(center)[$hat(bold(theta))_"MLE"=limits(arg max)_(bold(theta)) L(bold(theta)).$]
      Maximum likelihood estimation chooses the parameter vector that assigns the largest likelihood to the observed dataset.
    ]
  ],
  [
    #section-title([5 · Logs preserve the ranking], accent: palette.orange)
    #card(accent: palette.orange)[
      Because $log$ is strictly increasing on positive values,
      #align(center)[$limits(arg max)_theta L(theta)=limits(arg max)_theta log L(theta).$]
      For independent examples, the product becomes a sum:
      #align(center)[$log L(bold(theta))=sum_i log p_theta(y_i|bold(x)_i).$]
      #v(2pt)
      This avoids underflow and makes derivatives easier to combine.
    ]

    #v(4pt)
    #section-title([6 · Negative log-likelihood is a loss], accent: palette.blue)
    #card(accent: palette.blue)[
      Optimizers conventionally minimize, so negate the log-likelihood:
      #align(center)[$ell_i(bold(theta))=-log p_theta(y_i|bold(x)_i).$]
      #align(center)[$hat(R)(bold(theta))=1/n sum_i ell_i(bold(theta))=-1/n log L(bold(theta)).$]
      The factor $1/n$ changes gradient scale, not the minimizer.
    ]

    #v(4pt)
    #section-title([7 · Support becomes a hard constraint], accent: palette.red)
    #card(accent: palette.red, fill: soft(palette.red, amount: 96%))[
      If the model assigns $p_theta(y_i|bold(x)_i)=0$ to an observed target, then
      #align(center)[$ell_i=-log 0=+infinity.$]
      A zero-likelihood observation is an infinite penalty. Check support before blaming numerical instability.
    ]

    #v(4pt)
    #section-title([8 · Final audit], accent: palette.green)
    #card(accent: palette.green)[
      #check[Target support matches the chosen distribution.]
      #linebreak()
      #check[Neural outputs obey parameter constraints.]
      #linebreak()
      #check[Each observation contributes one log-density or log-probability.]
      #linebreak()
      #check[Independence assumptions are stated, not silently assumed.]
      #linebreak()
      #check[Likelihood and prediction are not confused.]
    ]

    #v(4pt)
    #banner(accent: palette.orange)[
      Probability models possible outcomes. Likelihood scores parameters using the outcome that occurred.
    ]
  ],
)
