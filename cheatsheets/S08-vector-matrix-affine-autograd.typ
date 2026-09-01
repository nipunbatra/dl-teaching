#import "../common/cheatsheet.typ": *

#set page(
  paper: "a4",
  margin: (top: 8mm, bottom: 9mm, x: 9mm),
  fill: white,
  footer: context align(center)[
    #text(size: 6.5pt, fill: palette.muted)[
      DL course · Lecture 8 · Vector, Matrix & Affine Autograd · page #counter(page).display()
    ]
  ],
)
#set text(font: "IBM Plex Sans", size: 8pt, fill: palette.ink)
#set par(leading: 0.52em, justify: false)
#set list(indent: 10pt, body-indent: 4pt, spacing: 2pt)
#show math.equation: set text(font: "New Computer Modern Math", size: 8.4pt)

#sheet-title(
  [LECTURE 8 CHEATSHEET],
  [Shape-safe vector autograd],
  subtitle: [Backprop is a right-to-left chain of vector–Jacobian products.],
)

#banner[Every tensor receives a gradient with the same shape as that tensor.]

#v(5pt)
#grid(
  columns: (1fr, 1fr),
  gutter: 7mm,
  [
    #section-title([1 · Name the local derivative by shape], accent: palette.blue)
    #card(accent: palette.blue, inset: 4.5pt)[
      #grid(
        columns: (1.2fr, 1fr, 0.8fr),
        column-gutter: 3pt,
        row-gutter: 3pt,
        [#text(weight: "bold")[Map]], [#text(weight: "bold")[Local object]], [#text(weight: "bold")[Shape]],
        [$u in RR arrow.r v in RR$], [$partial v/partial u$], [scalar],
        [$bold(x) in RR^d arrow.r z in RR$], [$nabla_x z$], [$d$],
        [$bold(x) in RR^d arrow.r bold(z) in RR^m$], [$J_(i j)=partial z_i/partial x_j$], [$m times d$],
      )
      #v(2pt)
      Rows of $J$ correspond to outputs; columns correspond to inputs.
    ]

    #v(4pt)
    #section-title([2 · The vector chain rule], accent: palette.orange)
    #card(accent: palette.orange)[
      For $bold(x) in RR^d arrow.r bold(z) in RR^m arrow.r cal(L) in RR$:
      #align(center)[$
        underbrace(nabla_x cal(L))_(d times 1)
        =underbrace(J^T)_(d times m)
         underbrace(nabla_z cal(L))_(m times 1).
      $]
      The arriving vector $bold(g)_z=nabla_z cal(L)$ is the upstream gradient. The product $J^T bold(g)_z$ is a #text(weight: "bold")[vector–Jacobian product] (VJP).
      #v(2pt)
      #tiny-note[Frameworks compute VJPs directly; they usually never build the full Jacobian.]
    ]

    #v(4pt)
    #section-title([3 · Why the transpose appears], accent: palette.teal)
    #card(accent: palette.teal)[
      If $bold(z)=(x_1+x_2, 2x_1-x_2)$, then
      #align(center)[$J=mat(1,1;2,-1).$]
      With upstream $bold(g)_z=mat(3;4)$,
      #align(center)[$
        bold(g)_x=J^T bold(g)_z
        =mat(1,2;1,-1)mat(3;4)=mat(11;-1).
      $]
      Each input adds the contribution arriving through every output path.
    ]

    #v(4pt)
    #card(title: [Seed rule], accent: palette.blue, fill: soft(palette.blue, amount: 95%))[
      A scalar loss starts reverse mode with $partial cal(L)/partial cal(L)=1$. A non-scalar output needs an explicit seed vector $bold(v)$; backward returns $J^T bold(v)$.
    ]
  ],
  [
    #section-title([4 · Scalar affine node], accent: palette.green)
    #card(accent: palette.green)[
      #align(center)[$z=bold(w)^T bold(x)+b.$]
      Local gradients:
      #align(center)[$
        (partial z)/(partial bold(w))=bold(x), quad
        (partial z)/(partial bold(x))=bold(w), quad
        (partial z)/(partial b)=1.
      $]
      If upstream $g_z$ arrives, multiply every local gradient by it:
      #align(center)[$
        bold(g)_w=g_z bold(x), quad bold(g)_x=g_z bold(w), quad g_b=g_z.
      $]
    ]

    #v(4pt)
    #section-title([5 · Dense layer, one example], accent: palette.blue)
    #card(accent: palette.blue)[
      Forward:
      #align(center)[$bold(z)=W bold(x)+bold(b)$]
      with $W in RR^(m times d)$, $bold(x) in RR^d$, $bold(z),bold(b) in RR^m$.
      #v(2pt)
      Backward for upstream $bold(g)_z in RR^m$:
      #align(center)[$
        bold(g)_x=W^T bold(g)_z, quad
        bold(g)_b=bold(g)_z, quad
        bold(g)_W=bold(g)_z bold(x)^T.
      $]
      #tiny-note[Outer product gives the parameter-gradient matrix. Check: $(m times 1)(1 times d)=m times d$.]
    ]

    #v(4pt)
    #section-title([6 · Elementwise nonlinearities], accent: palette.orange)
    #card(accent: palette.orange)[
      If $bold(h)=phi(bold(z))$ elementwise, then
      #align(center)[$bold(g)_z=bold(g)_h dot phi'(bold(z)).$]
      The local Jacobian is diagonal, so the VJP becomes an elementwise product. Common cases:
      #align(center)[$
        "ReLU"'(z)=cases(1 & z>0, 0 & z<0), quad
        sigma'(z)=sigma(z)(1-sigma(z)).
      $]
    ]

    #v(4pt)
    #banner(accent: palette.green)[
      Predict every gradient's shape before calculating its value.
    ]
  ],
)

#pagebreak()

#sheet-title(
  [LECTURE 8 CHEATSHEET],
  [Matrices, batches, and autograd traps],
  subtitle: [Use shape contracts to make vectorized backward passes auditable.],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 7mm,
  [
    #section-title([1 · Dense layer, row-batch convention], accent: palette.blue)
    #card(accent: palette.blue)[
      Let $X in RR^(N times d)$ store examples as rows:
      #align(center)[$Z=X W^T+bold(1)bold(b)^T in RR^(N times m).$]
      For upstream $G_Z in RR^(N times m)$:
      #align(center)[$
        G_X=G_Z W, quad
        G_W=G_Z^T X, quad
        bold(g)_b=sum_(n=1)^N G_(Z,n:).
      $]
      #tiny-note[If the loss is a mean over examples, its upstream signal already carries the factor $1/N$.]
    ]

    #v(4pt)
    #section-title([2 · Broadcasting means reduction in backward], accent: palette.orange)
    #card(accent: palette.orange)[
      Forward broadcasts $bold(b)$ across $N$ rows. Backward must undo that duplication by summing:
      #align(center)[$bold(g)_b=G_Z^T bold(1).$]
      General rule: if an axis was inserted or expanded during forward broadcasting, sum the gradient across that axis in backward.
      #v(2pt)
      #tiny-note[Silent broadcasting can create plausible but wrong objectives—for example `[N] - [N,1]` becoming `[N,N]`. Assert shapes before loss computation.]
    ]

    #v(4pt)
    #section-title([3 · Branches accumulate], accent: palette.teal)
    #card(accent: palette.teal)[
      If $x$ feeds two paths, both paths contribute:
      #align(center)[$
        (partial cal(L))/(partial x)
        =(partial cal(L))/(partial a)(partial a)/(partial x)
        +(partial cal(L))/(partial c)(partial c)/(partial x).
      $]
      Reverse mode adds all arriving gradients at the shared node. Reusing a tensor is not “last path wins.”
    ]

    #v(4pt)
    #section-title([4 · Matrix multiplication mnemonic], accent: palette.green)
    #card(accent: palette.green)[
      For $C=A B$ and upstream $G_C$:
      #align(center)[$G_A=G_C B^T, quad G_B=A^T G_C.$]
      Keep the untouched neighbour and transpose it so dimensions fit. Then verify that $G_A$ matches $A$ and $G_B$ matches $B$.
    ]
  ],
  [
    #section-title([5 · PyTorch reverse-mode checklist], accent: palette.green)
    #card(accent: palette.green)[
      #check[Leaf tensors that need gradients use `requires_grad=True`.]
      #linebreak()
      #check[The forward graph is built by tracked tensor operations.]
      #linebreak()
      #check[Call `loss.backward()` on a scalar loss.]
      #linebreak()
      #check[For vector outputs, supply the VJP seed explicitly.]
      #linebreak()
      #check[Read leaf gradients from `.grad`; clear them before reuse.]
      #linebreak()
      #check[Do parameter updates under `no_grad()` or through an optimizer.]
    ]

    #v(4pt)
    #section-title([6 · Minimal VJP check], accent: palette.blue)
    #card(accent: palette.blue)[
      #raw(block: true, lang: "python", "x = torch.tensor([1., 2.], requires_grad=True)\ny = f(x)                 # shape [m]\nv = torch.randn_like(y)\ny.backward(v)            # x.grad == J.T @ v")
      #v(2pt)
      For tiny functions, compare against a full Jacobian or finite differences. The agreement should hold in both #text(weight: "bold")[value and shape].
    ]

    #v(4pt)
    #section-title([7 · Failure modes], accent: palette.red)
    #card(accent: palette.red, inset: 4.5pt)[
      #grid(
        columns: (1fr, 1.25fr),
        column-gutter: 4pt,
        row-gutter: 3pt,
        [#text(weight: "bold")[Symptom]], [#text(weight: "bold")[Likely cause]],
        [wrong orientation], [mixed row/column conventions; missing transpose],
        [gradient too large], [forgot mean factor or summed a branch twice],
        [bias gradient wrong], [forgot reduction over broadcast axes],
        [`.grad` doubles], [did not clear accumulated gradients],
        [`None` gradient], [detached graph, non-leaf tensor, or unused path],
        [shape looks valid, loss wrong], [unintended broadcasting],
      )
    ]

    #v(4pt)
    #section-title([8 · Final shape audit], accent: palette.orange)
    #card(accent: palette.orange)[
      #keyline([Forward:], [ write the shape after every operation.], color: palette.blue)
      #linebreak()
      #keyline([Backward:], [ start from the loss and move right-to-left.], color: palette.orange)
      #linebreak()
      #keyline([At each node:], [ multiply by the local derivative, then sum contributions.], color: palette.teal)
      #linebreak()
      #keyline([Invariant:], [ returned gradient shape equals input shape.], color: palette.green)
    ]

    #v(4pt)
    #banner(accent: palette.blue)[
      Backprop is scalar chain-rule reasoning, stacked and vectorized without changing the logic.
    ]
  ],
)
