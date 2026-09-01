#import "../common/cheatsheet.typ": *

#set page(
  paper: "a4",
  margin: (top: 8mm, bottom: 9mm, x: 9mm),
  fill: white,
  footer: context align(center)[
    #text(size: 6.5pt, fill: palette.muted)[
      DL 2026 · Public lecture 4 · Linear models to neural networks · page #counter(page).display()
    ]
  ],
)
#set text(font: "IBM Plex Sans", size: 8pt, fill: palette.ink)
#set par(leading: 0.52em, justify: false)
#set list(indent: 10pt, body-indent: 4pt, spacing: 2pt)
#show math.equation: set text(font: "New Computer Modern Math", size: 8.4pt)

#sheet-title(
  [PUBLIC LECTURE 4 CHEATSHEET],
  [From a weighted sum to a neuron],
  subtitle: [A neural network becomes expressive when affine maps are separated by nonlinear activations.],
)

#banner[
  The loss says what counts as a good prediction. The network says which predictions are possible.
]

#v(5pt)
#grid(
  columns: (1fr, 1fr),
  gutter: 7mm,
  [
    #section-title([1 · The linear starting point], accent: palette.blue)
    #card(accent: palette.blue)[
      One scalar output begins with
      #align(center)[$z = bold(w)^top bold(x)+b.$]
      Each edge contributes “weight $times$ incoming value”; the node adds the contributions and the bias.
      #v(3pt)
      #grid(
        columns: (auto, 1fr), row-gutter: 2pt, column-gutter: 4pt,
        [#label([INPUT], color: palette.blue)], [$bold(x) in RR^d$],
        [#label([WEIGHTS], color: palette.teal)], [$bold(w) in RR^d$],
        [#label([BIAS], color: palette.orange)], [$b in RR$],
        [#label([OUTPUT], color: palette.green)], [$z in RR$],
      )
    ]

    #v(4pt)
    #card(title: [Concrete weighted sum], accent: palette.teal, fill: soft(palette.teal, amount: 96%))[
      For $bold(x)=(2,-1)$, $bold(w)=(1.5,0.5)$, and $b=-0.2$:
      #align(center)[$z=1.5(2)+0.5(-1)-0.2=2.3.$]
      The bias is equivalent to an extra constant feature equal to $1$.
    ]

    #v(4pt)
    #section-title([2 · Batch form and shape audit], accent: palette.blue)
    #card(accent: palette.blue)[
      Stack $n$ examples as rows of $bold(X) in RR^(n times d)$:
      #align(center)[$hat(bold(y))=bold(X)bold(w)+b bold(1).$]
      Or append a ones column $tilde(bold(X))$ and stack the bias into $bold(theta)$:
      #align(center)[$hat(bold(y))=tilde(bold(X))bold(theta).$]
      #tiny-note[Shape check: $(n times (d+1))((d+1) times 1) arrow.r (n times 1)$.]
    ]

    #v(4pt)
    #section-title([3 · A neuron adds a nonlinear stage], accent: palette.orange)
    #card(accent: palette.orange)[
      #align(center)[$z=bold(w)^top bold(x)+b, quad a=phi(z).$]
      The affine stage chooses an input direction and offset. The activation gates or reshapes the resulting score.
      #v(3pt)
      For $bold(x)=(2,-1)$, $bold(w)=(1,-1)$, $b=0$:
      #align(center)[$z=3, quad a="ReLU"(3)=3.$]
    ]
  ],
  [
    #section-title([4 · Activation map], accent: palette.orange)
    #card(accent: palette.orange, inset: 4.5pt)[
      #grid(
        columns: (0.62fr, 1.18fr, 1.2fr), column-gutter: 3pt, row-gutter: 3pt,
        [#text(weight: "bold", fill: palette.muted)[Name]],
        [#text(weight: "bold", fill: palette.muted)[Rule / range]],
        [#text(weight: "bold", fill: palette.muted)[Use and warning]],
        [sigmoid], [$sigma(z)=1/(1+e^(-z))$], [probability-like switch; saturates],
        [tanh], [$"tanh"(z) in (-1,1)$], [signed switch; saturates],
        [ReLU], [$max(0,z)$], [simple ramp; dead for $z<0$],
        [GELU], [$approx z sigma(1.702z)$], [smooth, input-dependent gate],
      )
    ]

    #v(4pt)
    #card(title: [Saturation is a learning issue], accent: palette.red, fill: soft(palette.red, amount: 96%))[
      A flat activation has a tiny derivative. For sigmoid,
      #align(center)[$sigma'(z)=sigma(z)(1-sigma(z)).$]
      Near $z=0$, $sigma'(0)=0.25$; near $z=6$, $sigma'(6) approx 0.0025$. The same input nudge produces about $100 times$ less output change in the saturated regime.
    ]

    #v(4pt)
    #section-title([5 · Why nonlinearity is essential], accent: palette.green)
    #card(accent: palette.green)[
      Two affine layers collapse:
      #align(center)[$
        bold(h)=bold(W)_1bold(x)+bold(b)_1,
        quad bold(y)=bold(W)_2bold(h)+bold(b)_2
      $]
      #align(center)[$
        bold(y)=underbrace(bold(W)_2bold(W)_1)_(bold(W)_"eq")bold(x)
        +underbrace(bold(W)_2bold(b)_1+bold(b)_2)_(bold(b)_"eq").
      $]
      Without $phi$, extra layers add parameters but no new bends or learned feature shapes.
    ]

    #v(4pt)
    #banner(accent: palette.blue)[
      Affine map = select and shift a direction. Activation = introduce a bend. Composition = build a representation.
    ]
  ],
)

#pagebreak()

#sheet-title(
  [PUBLIC LECTURE 4 CHEATSHEET],
  [Read an MLP by its shapes and features],
  subtitle: [Hidden units learn coordinates in which the final prediction can be simpler.],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 7mm,
  [
    #section-title([1 · One hidden layer], accent: palette.teal)
    #card(accent: palette.teal)[
      #align(center)[$
        bold(h)=phi(bold(W)_1bold(x)+bold(b)_1),
        quad bold(z)=bold(W)_2bold(h)+bold(b)_2.
      $]
      With input width $d$, hidden width $m$, and $K$ outputs:
      #grid(
        columns: (1fr, 1fr), row-gutter: 2pt, column-gutter: 4pt,
        [$bold(W)_1 in RR^(m times d)$], [$bold(b)_1 in RR^m$],
        [$bold(W)_2 in RR^(K times m)$], [$bold(b)_2 in RR^K$],
      )
      #align(center)[$(m times d)(d times 1) arrow.r m, quad (K times m)m arrow.r K.$]
      Trainable scalars $=m(d+1)+K(m+1)$.
    ]

    #v(4pt)
    #section-title([2 · XOR: features can solve what one line cannot], accent: palette.blue)
    #card(accent: palette.blue)[
      A single binary linear score has one straight decision boundary. For the four XOR corners, create
      #align(center)[$
        h_1="ReLU"(x_1+x_2),
        quad h_2="ReLU"(x_1+x_2-1),
      $]
      then use
      #align(center)[$z=2h_1-4h_2-1, quad hat(y)=bb(1)[z>=0].$]
      #grid(
        columns: (1fr, 1fr, 1fr, 1fr), row-gutter: 2pt, column-gutter: 2pt,
        [$(0,0) arrow.r 0$], [$(1,0) arrow.r 1$], [$(0,1) arrow.r 1$], [$(1,1) arrow.r 0$],
      )
      #tiny-note[Exact on the four Boolean points does not imply the same rule labels every point inside filled XOR regions. State the input domain.]
    ]

    #v(4pt)
    #section-title([3 · Width and depth answer different needs], accent: palette.orange)
    #card(accent: palette.orange)[
      #keyline([Width:], [ more units in one layer; more features at the same stage.], color: palette.teal)
      #linebreak()
      #keyline([Depth:], [ more successive transformations; later features reuse earlier features.], color: palette.blue)
      #v(3pt)
      Choose width for variety and depth for composition. A layer receives the previous representation, not the original input unless an explicit skip connection supplies it.
    ]
  ],
  [
    #section-title([4 · ReLU networks assemble hinges], accent: palette.orange)
    #card(accent: palette.orange)[
      One shifted ReLU is a hinge:
      #align(center)[$h_j(x)="ReLU"(w_j x+b_j).$]
      A linear output combines many such features:
      #align(center)[$g(x)=c+sum_j v_j "ReLU"(w_j x+b_j).$]
      The input weights and biases place the hinges; the output weights choose their slope changes. In higher dimensions, ReLU units fold space along hyperplanes.
    ]

    #v(4pt)
    #section-title([5 · Universal approximation: read the claim carefully], accent: palette.green)
    #card(accent: palette.green)[
      For a continuous target $f$ on a closed, bounded region $D$ and any $epsilon>0$, a sufficiently wide finite network with a suitable nonlinear activation can satisfy
      #align(center)[$limits(sup)_(x in D) abs(f(x)-g(x))<epsilon.$]
      This is an #text(weight: "bold")[existence and capacity] statement. It does not promise that gradient-based training will find the network, that the network will be small, or that it will generalize beyond $D$.
    ]

    #v(4pt)
    #section-title([6 · Output-layer contract], accent: palette.blue)
    #card(accent: palette.blue, inset: 4.5pt)[
      #grid(
        columns: (0.9fr, 1.25fr, 1.05fr), column-gutter: 3pt, row-gutter: 3pt,
        [#text(weight: "bold", fill: palette.muted)[Task]], [#text(weight: "bold", fill: palette.muted)[Raw output]], [#text(weight: "bold", fill: palette.muted)[Final map]],
        [regression], [one or more scores], [identity / task constraint],
        [binary], [one logit $z$], [sigmoid $sigma(z)$],
        [$K$ classes], [$K$ logits], [softmax over $K$],
      )
      #tiny-note[The hidden representation may stay the same; the number and interpretation of output nodes follows the target.]
    ]

    #v(4pt)
    #card(title: [Fast architecture audit], accent: palette.teal, fill: soft(palette.teal, amount: 96%))[
      #check[Write the shape of every value and parameter.]
      #linebreak()
      #check[Point to the nonlinearity between affine layers.]
      #linebreak()
      #check[Explain each hidden layer as a learned feature map.]
      #linebreak()
      #check[Match the output width and transform to the task.]
      #linebreak()
      #check[Separate representability from trainability.]
    ]
  ],
)
