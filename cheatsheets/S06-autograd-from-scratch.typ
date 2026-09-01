#import "../common/cheatsheet.typ": *

#set page(
  paper: "a4",
  margin: (top: 8mm, bottom: 9mm, x: 9mm),
  fill: white,
  footer: context align(center)[
    #text(size: 6.5pt, fill: palette.muted)[
      DL 2026 · Public lecture 6 · Autograd from scratch · page #counter(page).display()
    ]
  ],
)
#set text(font: "IBM Plex Sans", size: 8pt, fill: palette.ink)
#set par(leading: 0.52em, justify: false)
#set list(indent: 10pt, body-indent: 4pt, spacing: 2pt)
#show math.equation: set text(font: "New Computer Modern Math", size: 8.4pt)

#sheet-title(
  [PUBLIC LECTURE 6 CHEATSHEET],
  [Reverse mode from local rules],
  subtitle: [Record the executed scalar graph, seed the loss, and propagate sensitivities backward.],
)

#banner[
  Backprop is the reverse-sweep algorithm. Autograd is the engine that records operations and executes their backward rules.
]

#v(5pt)
#grid(
  columns: (1fr, 1fr), gutter: 7mm,
  [
    #section-title([1 · Derivative as a local model], accent: palette.blue)
    #card(accent: palette.blue)[
      Near $u$, a scalar function is approximately linear:
      #align(center)[$f(u+Delta u) approx f(u)+f'(u)Delta u.$]
      For many parameters, stack the partial derivatives:
      #align(center)[$nabla_(bold(theta)) cal(L)=vec((partial cal(L))/(partial theta_1),dots.v).$]
      Gradient descent subtracts this vector. Reverse mode computes it efficiently when many inputs feed one scalar loss.
    ]

    #v(4pt)
    #section-title([2 · One operation, one reusable rule], accent: palette.teal)
    #card(accent: palette.teal)[
      For $v=f(u)$, if $g_v=partial cal(L)/partial v$ arrives from downstream:
      #align(center)[$
        underbrace(g_u)_("sent to" u)
        =underbrace(g_v)_("arriving at" v)
        underbrace((partial v)/(partial u))_("local slope").
      $]
      Every operation needs only its local derivative and any forward values required to evaluate it.
    ]

    #v(4pt)
    #section-title([3 · Scalar rule table], accent: palette.orange)
    #card(accent: palette.orange, inset: 4.5pt)[
      #grid(
        columns: (0.9fr, 1.55fr), column-gutter: 4pt, row-gutter: 3pt,
        [#text(weight: "bold", fill: palette.muted)[Forward]], [#text(weight: "bold", fill: palette.muted)[Backward contributions]],
        [$v=a+b$], [$g_a += g_v, quad g_b += g_v$],
        [$v=a b$], [$g_a += g_v b, quad g_b += g_v a$],
        [$v=u^2$], [$g_u += g_v 2u$],
        [$v=phi(u)$], [$g_u += g_v phi'(u)$],
        [$v=a-y$], [$g_a += g_v, quad g_y -= g_v$],
      )
      #tiny-note[Here $g_u$ is shorthand for $partial cal(L)/partial u$; the subscript names the stored value.]
    ]
  ],
  [
    #section-title([4 · Branches must accumulate], accent: palette.red)
    #card(accent: palette.red)[
      If a value $x$ influences the loss through several children $u_1,dots,u_k$:
      #align(center)[$
        g_x=sum_(i=1)^k g_(u_i)(partial u_i)/(partial x).
      $]
      Use `+=`, not `=`. Overwriting one contribution silently drops every other path.
      #v(3pt)
      Example: $u=x^2$, $v=3x$, $cal(L)=u+v$. At $x=4$:
      #align(center)[$g_x=1(2x)+1(3)=8+3=11.$]
    ]

    #v(4pt)
    #section-title([5 · Reverse-mode algorithm], accent: palette.green)
    #card(accent: palette.green)[
      #grid(
        columns: (auto, 1fr), row-gutter: 3pt, column-gutter: 4pt,
        [#label([1], color: palette.green)], [run the forward program and create value nodes],
        [#label([2], color: palette.green)], [store the operation and required operands on each result],
        [#label([3], color: palette.green)], [topologically order the graph],
        [#label([4], color: palette.green)], [initialize every gradient to zero],
        [#label([5], color: palette.green)], [seed the scalar loss with $g_cal(L)=1$],
        [#label([6], color: palette.green)], [visit operations in reverse topological order and accumulate],
      )
    ]

    #v(4pt)
    #card(title: [Why reverse mode for training?], accent: palette.blue, fill: soft(palette.blue, amount: 96%))[
      Training maps millions of parameters to one scalar loss. One reverse sweep returns every parameter sensitivity together. Finite differences require two extra forward evaluations per checked coordinate; use them to debug, not to train.
    ]

    #v(4pt)
    #banner(accent: palette.orange)[
      Forward computes values. Backward computes how the final scalar loss responds to each stored value.
    ]
  ],
)

#pagebreak()

#sheet-title(
  [PUBLIC LECTURE 6 CHEATSHEET],
  [Work one scalar graph end to end],
  subtitle: [The numbers below expose every stored value, local derivative, and accumulated gradient.],
)

#grid(
  columns: (1fr, 1fr), gutter: 7mm,
  [
    #section-title([1 · Model, loss, and primitive graph], accent: palette.blue)
    #card(accent: palette.blue)[
      #align(center)[$hat(y)=w x+b, quad cal(L)=(hat(y)-y)^2.$]
      Use $(w,x,b,y)=(2,3,1,10)$ and name the intermediate values:
      #align(center)[$m=w x, quad a=m+b, quad e=a-y, quad cal(L)=e^2.$]
      This decomposition is the graph: leaves $(w,x,b,y)$, stored values $(m,a,e)$, and one scalar output $cal(L)$.
    ]

    #v(4pt)
    #section-title([2 · Forward pass], accent: palette.teal)
    #card(accent: palette.teal, inset: 4.5pt)[
      #grid(
        columns: (0.55fr, 1.25fr, 0.7fr), column-gutter: 4pt, row-gutter: 3pt,
        [#text(weight: "bold", fill: palette.muted)[Value]], [#text(weight: "bold", fill: palette.muted)[Operation]], [#text(weight: "bold", fill: palette.muted)[Result]],
        [$m$], [$w x$], [$6$],
        [$a$], [$m+b$], [$7$],
        [$e$], [$a-y$], [$-3$],
        [$cal(L)$], [$e^2$], [$9$],
      )
      Save operands such as $x,w,e$ because the multiplication and square rules need them during backward.
    ]

    #v(4pt)
    #section-title([3 · Backward pass], accent: palette.orange)
    #card(accent: palette.orange)[
      Seed $g_cal(L)=1$ and reverse the operations:
      #align(center)[$g_e=1(2e)=-6.$]
      #align(center)[$g_a=-6, quad g_y=(-6)(-1)=6.$]
      #align(center)[$g_m=-6, quad g_b=-6.$]
      #align(center)[$g_w=g_m x=-18, quad g_x=g_m w=-12.$]
      #v(2pt)
      #banner(accent: palette.orange)[
        Final: $partial cal(L)/partial (w,b,x,y)=(-18,-6,-12,6)$.
      ]
    ]
  ],
  [
    #section-title([4 · Read the sign before updating], accent: palette.green)
    #card(accent: palette.green)[
      Because $partial cal(L)/partial w=-18$, a small gradient-descent step increases $w$:
      #align(center)[$w <- w-eta(-18)=w+18eta.$]
      With $eta=0.01$, update $w:2 arrow.r 2.18$ and $b:1 arrow.r 1.06$. The prediction changes from $7$ to $7.60$, toward the target $10$.
    ]

    #v(4pt)
    #section-title([5 · Autograd lifecycle], accent: palette.blue)
    #card(accent: palette.blue)[
      #grid(
        columns: (0.9fr, 1.35fr), column-gutter: 4pt, row-gutter: 3pt,
        [#text(weight: "bold", fill: palette.muted)[Concept]], [#text(weight: "bold", fill: palette.muted)[PyTorch]],
        [leaf value], [`Tensor(requires_grad=True)`],
        [creator rule], [`grad_fn` on a non-leaf],
        [reverse sweep], [`L.backward()`],
        [leaf derivative], [`p.grad`],
        [keep intermediate], [`value.retain_grad()`],
        [clear old gradients], [`opt.zero_grad()`],
      )
      Gradients accumulate by default. A second backward pass without clearing adds another contribution.
    ]

    #v(4pt)
    #section-title([6 · Independent gradient check], accent: palette.teal)
    #card(accent: palette.teal)[
      For one coordinate $theta_j$, use the central difference
      #align(center)[$
        g_"FD"=(cal(L)(theta_j+epsilon)-cal(L)(theta_j-epsilon))/(2epsilon).
      $]
      Compare it with autograd in float64 at a few coordinates. Large disagreement usually means a graph, broadcasting, in-place mutation, nondifferentiability, or numerical-scale bug.
    ]

    #v(4pt)
    #card(title: [From-scratch implementation audit], accent: palette.red, fill: soft(palette.red, amount: 96%))[
      #check[Every result remembers its parents and local backward rule.]
      #linebreak()
      #check[Backward order is reverse topological, not recursive guesswork.]
      #linebreak()
      #check[Branches accumulate with `+=`.]
      #linebreak()
      #check[Only a scalar output is seeded automatically.]
      #linebreak()
      #check[Tests cover shared inputs and repeated use of one value.]
    ]
  ],
)
