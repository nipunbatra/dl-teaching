// Computation Graphs, Backpropagation and Autodiff — Lecture 3 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture3/L3-backprop.typ
//   typst compile --root . --input handout=true lecture3/L3-backprop.typ
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#import "@local/chalkdust-autodiff:0.1.0" as ad   // our own reverse-mode AD — used live on the scalar example
#import "@local/chalkdust-theme:0.1.0": theme as chalk-theme
// palette-locked computation-graph drawer + the scalar loss, both reused below
#let cgraph = ad.graph.with(theme: chalk-theme(ink: INK, accent: ACC, accent2: TEAL, muted: MUTED))
#let scalar-loss = ad.expr("(w*x + b - y)^2", ("w", "x", "b", "y"))
#show: metropolis-deck.with(
  title: [Computation Graphs, Backpropagation and Autodiff],
  subtitle: [How gradients flow through neural networks],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let NB = "https://raw.githubusercontent.com/nipunbatra/dl-teaching/master/notebooks/L03/"

// ── local fletcher diagrams ──────────────────────────────────────────
// one local node: teal values forward, orange gradient backward
#let localnode = align(center, diagram(spacing: (26mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), $u$, radius: 6mm)
  node((1,0), $v$, radius: 6mm, stroke: 0.9pt + TEAL)
  node((2,0), $cal(L)$, radius: 6mm, stroke: 0.9pt + ACC)
  edge((0,0),(1,0), text(size: 13pt, fill: TEAL)[forward #h(3pt) $v = f(u)$], "-|>", stroke: 1.1pt + TEAL, label-sep: 8pt)
  edge((1,0),(2,0), "-|>", stroke: 1.1pt + TEAL)
  edge((2,0),(0,0), "-|>", bend: 34deg, stroke: 1.2pt + ACC,
    label: text(size: 13pt, fill: ACC)[backward #h(3pt) $overline(u) = overline(v) f'(u)$], label-sep: 8pt)
}))

// compact recurring motif for slides that refer back to the same local rule
#let localnode-mini = align(center, diagram(spacing: (17mm, 7mm), node-stroke: 0.8pt + INK, node-fill: white, {
  node((0,0), $u$, radius: 4.5mm)
  node((1,0), $v$, radius: 4.5mm, stroke: 0.8pt + TEAL)
  node((2,0), $cal(L)$, radius: 4.5mm, stroke: 0.8pt + ACC)
  edge((0,0),(1,0), text(size: 10pt, fill: TEAL)[$v = f(u)$], "-|>", stroke: 1pt + TEAL, label-sep: 6pt)
  edge((1,0),(2,0), "-|>", stroke: 1pt + TEAL)
  edge((2,0),(0,0), "-|>", bend: 34deg, stroke: 1.05pt + ACC,
    label: text(size: 10pt, fill: ACC)[$overline(u) = overline(v) f'(u)$], label-sep: 6pt)
}))

// numeric square-node instance used in the first worked local-rule example
#let squarenode = align(center, diagram(spacing: (24mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), $3$, radius: 5.5mm)
  node((1,0), $9$, radius: 5.5mm, stroke: 0.9pt + TEAL)
  node((2,0), $cal(L)$, radius: 5.5mm, stroke: 0.9pt + ACC)
  edge((0,0),(1,0), text(size: 12pt, fill: TEAL)[$u^2$], "-|>", stroke: 1.1pt + TEAL, label-sep: 8pt)
  edge((1,0),(2,0), "-|>", stroke: 1.1pt + TEAL)
  edge((2,0),(0,0), "-|>", bend: 34deg, stroke: 1.2pt + ACC,
    label: text(size: 12pt, fill: ACC)[backward #h(3pt) $overline(v) = 7$, #h(3pt) $2u = 6$ #h(3pt) $arrow.r$ #h(3pt) $overline(u) = 42$], label-sep: 7pt)
}))

// addition merges two values forward and copies one gradient backward
#let addnode = align(center, diagram(spacing: (22mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,-0.65), $a$, radius: 5mm)
  node((0,0.65), $b$, radius: 5mm)
  node((1,0), $v$, radius: 5.5mm, stroke: 0.9pt + TEAL)
  node((2,0), $cal(L)$, radius: 5.5mm, stroke: 0.9pt + ACC)
  edge((0,-0.65),(1,0), "-|>", stroke: 1.05pt + TEAL)
  edge((0,0.65),(1,0), "-|>", stroke: 1.05pt + TEAL)
  edge((1,0),(2,0), text(size: 11pt, fill: TEAL)[$v = a + b$], "-|>", stroke: 1.05pt + TEAL, label-sep: 6pt)
  edge((2,0),(1,0), "-|>", bend: 30deg, stroke: 1.1pt + ACC,
    label: text(size: 11pt, fill: ACC)[$overline(v)$ arrives], label-sep: 6pt)
  edge((1,0),(0,-0.65), "-|>", bend: -22deg, stroke: 1.1pt + ACC,
    label: text(size: 11pt, fill: ACC)[$overline(a) = overline(v)$], label-sep: 5pt)
  edge((1,0),(0,0.65), "-|>", bend: 22deg, stroke: 1.1pt + ACC,
    label: text(size: 11pt, fill: ACC)[$overline(b) = overline(v)$], label-sep: 5pt)
}))

// multiplication sends each input the other input during the backward pass
#let mulnode = align(center, diagram(spacing: (22mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,-0.65), $a$, radius: 5mm)
  node((0,0.65), $b$, radius: 5mm)
  node((1,0), $v$, radius: 5.5mm, stroke: 0.9pt + TEAL)
  node((2,0), $cal(L)$, radius: 5.5mm, stroke: 0.9pt + ACC)
  edge((0,-0.65),(1,0), "-|>", stroke: 1.05pt + TEAL)
  edge((0,0.65),(1,0), "-|>", stroke: 1.05pt + TEAL)
  edge((1,0),(2,0), text(size: 11pt, fill: TEAL)[$v = a b$], "-|>", stroke: 1.05pt + TEAL, label-sep: 6pt)
  edge((2,0),(1,0), "-|>", bend: 30deg, stroke: 1.1pt + ACC,
    label: text(size: 11pt, fill: ACC)[$overline(v)$ arrives], label-sep: 6pt)
  edge((1,0),(0,-0.65), "-|>", bend: -22deg, stroke: 1.1pt + ACC,
    label: text(size: 11pt, fill: ACC)[$overline(a) = overline(v) b$], label-sep: 5pt)
  edge((1,0),(0,0.65), "-|>", bend: 22deg, stroke: 1.1pt + ACC,
    label: text(size: 11pt, fill: ACC)[$overline(b) = overline(v) a$], label-sep: 5pt)
}))

// elementwise activation as the same one-input local-rule motif
#let actnode = align(center, diagram(spacing: (24mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), $u$, radius: 5.5mm)
  node((1,0), $v$, radius: 5.5mm, stroke: 0.9pt + TEAL)
  node((2,0), $cal(L)$, radius: 5.5mm, stroke: 0.9pt + ACC)
  edge((0,0),(1,0), text(size: 12pt, fill: TEAL)[$v = phi(u)$], "-|>", stroke: 1.1pt + TEAL, label-sep: 7pt)
  edge((1,0),(2,0), "-|>", stroke: 1.1pt + TEAL)
  edge((2,0),(0,0), "-|>", bend: 34deg, stroke: 1.2pt + ACC,
    label: text(size: 12pt, fill: ACC)[$overline(u) = overline(v) phi'(u)$], label-sep: 7pt)
}))

// branch: x fans to u and v, both to L
#let branchgraph = align(center, diagram(spacing: (16mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), $x$, radius: 6mm)
  node((1,0.7), $u$, radius: 6mm, stroke: 0.9pt + TEAL)
  node((1,-0.7), $v$, radius: 6mm, stroke: 0.9pt + TEAL)
  node((2,0), $cal(L)$, radius: 6mm, stroke: 0.9pt + ACC)
  edge((0,0),(1,0.7), "-|>", stroke: 0.8pt + MUTED)
  edge((0,0),(1,-0.7), "-|>", stroke: 0.8pt + MUTED)
  edge((1,0.7),(2,0), "-|>", stroke: 0.8pt + MUTED)
  edge((1,-0.7),(2,0), "-|>", stroke: 0.8pt + MUTED)
}))

// scalar computation graph for L=(wx+b-y)^2
#let scalargraph = align(center, diagram(spacing: (14mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,2),  $w$, radius: 5.5mm)
  node((0,1),  $x$, radius: 5.5mm)
  node((0,0),  $b$, radius: 5.5mm)
  node((0,-1.4), $y$, radius: 5.5mm)
  node((1.4,1.5), $times$, radius: 6mm, fill: rgb("#E6F0F0"), stroke: 0.9pt + TEAL)
  node((2.8,0.7), $+$, radius: 6mm, fill: rgb("#E6F0F0"), stroke: 0.9pt + TEAL)
  node((4.2,-0.3), $-$, radius: 6mm, fill: rgb("#E6F0F0"), stroke: 0.9pt + TEAL)
  node((5.6,-0.3), [$(dot)^2$], radius: 7mm, fill: rgb("#FDECD6"), stroke: 0.9pt + ACC)
  edge((0,2),(1.4,1.5), "-|>", stroke: 0.7pt + MUTED)
  edge((0,1),(1.4,1.5), "-|>", stroke: 0.7pt + MUTED)
  edge((1.4,1.5),(2.8,0.7), $m$, "-|>", stroke: 0.8pt + INK)
  edge((0,0),(2.8,0.7), "-|>", stroke: 0.7pt + MUTED)
  edge((2.8,0.7),(4.2,-0.3), $a$, "-|>", stroke: 0.8pt + INK)
  edge((0,-1.4),(4.2,-0.3), "-|>", stroke: 0.7pt + MUTED)
  edge((4.2,-0.3),(5.6,-0.3), $e$, "-|>", stroke: 0.8pt + INK)
}))

// (the backward graph is now auto-drawn by ad.graph from the real computation —
//  see "The backward pass, visualized")

// two-layer MLP forward chain
#let mlpforward = align(center, diagram(spacing: 12mm, node-stroke: 0.9pt + INK, node-fill: white, {
  let labs = ($bold(x)$, $bold(z)^((1))$, $bold(h)$, $bold(z)^((2))$, $bold(p)$, $cal(L)$)
  for (i, l) in labs.enumerate() {
    let c = if i == labs.len() - 1 { ACC } else if i == 0 { INK } else { TEAL }
    node((i, 0), l, radius: 6mm, stroke: 0.9pt + c)
  }
  for i in range(labs.len() - 1) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.8pt + MUTED) }
}))

#title-slide()

// ═══════════════════════════ PART I — Vocabulary ═══════════════════════════
= Backprop vocabulary

== Lecture 2 built the forward map; today we reverse it #V

Last time, an MLP turned an input into a prediction and then one scalar loss. Today we reuse that *same graph* in reverse:
#pause
#two(
  notebox[
    *General pattern*
    #v(5pt)
    *Forward:* compute values
    $ bold(x) -> f_(bold(theta))(bold(x)) = hat(bold(y)) -> cal(L) $
    #v(5pt)
    *Backward:* compute sensitivities
    $ cal(L) -> nabla_(bold(theta)) cal(L) $
  ],
  notebox[
    *Concrete scalar example*
    #v(5pt)
    *Forward:* with $(w, x, b, y) = (2, 3, 1, 10)$
    $ hat(y) = 2 dot 3 + 1 = 7, quad cal(L) = (7 - 10)^2 = 9 $
    #v(5pt)
    *Backward:*
    $ ((partial cal(L))/(partial w), (partial cal(L))/(partial b)) = (-18, -6) $
  ],
)
#pause
#result[same executed graph · values flow forward, gradients flow backward]

== Backprop returns every parameter sensitivity in one sweep

Every update step is
$ bold(theta)_(t+1) = bold(theta)_t - eta thin nabla_(bold(theta)) cal(L)(bold(theta)_t). $
#pause
A modern network has $P = 10^7$ to $10^11$ parameters.
#pause
#alertbox[We need all $P$ components $partial cal(L) \/ partial theta_j$. Backprop computes them together in *one reverse sweep*, without perturbing parameters one at a time.]

== Read the notation before the algorithm

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*Symbol*], [*Meaning in this lecture*]),
  [$u, v, z$], [scalars],
  [$bold(x), bold(z), bold(theta)$], [vectors (bold lowercase)],
  [$bold(W), bold(X)$], [matrices (bold uppercase)],
  [$overline(u)$], [$partial cal(L) \/ partial u$: the loss sensitivity carried backward],
  [$nabla_(bold(theta)) cal(L)$], [all parameter partial derivatives stacked as a vector],
))
#pause
#notebox[The bar is *not an average*. Read $overline(u)$ aloud as “the gradient arriving at $u$.”]

== The route from one node to an entire network #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 7pt), align: (left, left),
  table.header([*Stage*], [*Question we answer*]),
  [1 · local rule], [How does one operation pass a gradient backward?],
  [#uncover("2-")[2 · branches]], [#uncover("2-")[Why do gradient contributions add?]],
  [#uncover("3-")[3 · worked graph]], [#uncover("3-")[Can we reproduce every derivative numerically?]],
  [#uncover("4-")[4 · layers + MLP]], [#uncover("4-")[How do scalar rules become vector and matrix formulas?]],
  [#uncover("5-")[5 · autodiff + flow]], [#uncover("5-")[How do frameworks execute and check the reverse sweep?]],
))
#pause
#pause
#pause
#pause

== One local node

Zoom in on a single edge of the graph: $u -> v -> cal(L)$.
#pause
#localnode
#pause
#result[backward computes $overline(u) = overline(v) dot f'(u)$]

== Every node multiplies upstream by its local slope #D

At the node $v = f(u)$, write the chain rule in the same bar notation as the graph:
#localnode-mini
#pause
#align(center, text(size: 31pt, weight: 600)[$ overline(u) = overline(v) dot f'(u) $])
#pause
#grid(
  columns: 3, gutter: 12pt,
  notebox[
    #text(fill: ACC, weight: 600)[Sent back to $u$]
    #v(5pt)
    $ overline(u) = (partial cal(L))/(partial u) $
  ],
  notebox[
    #text(fill: TEAL, weight: 600)[Arriving at $v$]
    #v(5pt)
    $ overline(v) = (partial cal(L))/(partial v) $
  ],
  notebox[
    #text(fill: INK, weight: 600)[Local slope]
    #v(5pt)
    $ f'(u) = (partial v)/(partial u) $
  ],
)
#pause
#result[gradient sent backward $=$ gradient arriving $times$ local slope]

== Bar notation turns the chain rule into one reusable rule #D

Write $overline(u) = partial cal(L) \/ partial u$. Then every node obeys:
#localnode-mini
$ overline(u) = overline(v) dot (partial v)/(partial u). $
#pause
#notebox[The node only needs to know its *own* local derivative $partial v \/ partial u$. It multiplies whatever gradient $overline(v)$ arrives from above — no global bookkeeping.]

== Example: a square node #D

For $v = u^2$ at $u = 3$, the forward value is $v = 9$ and the local slope is $partial v \/ partial u = 2u = 6$.
#pause
#squarenode
#pause
$ overline(u) = overline(v) dot 2u = 7 dot 6 = 42. $
#pause
#result[downstream $overline(u) = 42$]
#pause
#notebox[The node did *one* thing: multiply the incoming $overline(v) = 7$ by its own local slope $2u = 6$.]

== Addition copies; multiplication swaps #D

Node $v = a + b$. Local derivatives $partial v \/ partial a = 1$, $partial v \/ partial b = 1$.
#pause
#addnode
#pause
$ overline(a) = overline(v) dot 1 = overline(v), quad quad overline(b) = overline(v) dot 1 = overline(v). $
#pause
#result[$+$ *copies* the upstream gradient to every input]

== Multiplication sends each input the other input #D

Node $v = a b$. Local derivatives $partial v \/ partial a = b$, $partial v \/ partial b = a$.
#pause
#mulnode
#pause
$ overline(a) = overline(v) dot b, quad quad overline(b) = overline(v) dot a. $
#pause
#result[$times$ *sends the other input*]
#pause
#notebox[If $a = 2, b = 5, overline(v) = 3$: then $overline(a) = 3 dot 5 = 15$ and $overline(b) = 3 dot 2 = 6$.]

== Activation node #D

Node $v = phi(u)$ (elementwise nonlinearity). Local derivative $partial v \/ partial u = phi'(u)$.
#pause
#actnode
#pause
$ overline(u) = overline(v) dot phi'(u). $
#pause
#two(
  notebox[*sigmoid* $sigma$: #h(0.3em) $phi'(u) = sigma(u)(1 - sigma(u))$],
  notebox[*ReLU*: #h(0.3em) $phi'(u) = bb(1)[u > 0]$],
)

== The rule table — memorize this #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 7pt), align: (left, center, left),
  table.header([*Forward op*], [*Local grad*], [*Backward update*]),
  [$v = a + b$],   [$1, 1$],     [$overline(a)$ += $overline(v)$, #h(0.3em) $overline(b)$ += $overline(v)$],
  [$v = a b$],     [$b, a$],     [$overline(a)$ += $overline(v) b$, #h(0.3em) $overline(b)$ += $overline(v) a$],
  [$v = u^2$],     [$2u$],       [$overline(u)$ += $overline(v) dot 2u$],
  [$v = phi(u)$],  [$phi'(u)$],  [$overline(u)$ += $overline(v) phi'(u)$],
))
#pause
#alertbox[Use `+=`, not `=` — a variable feeding several nodes accumulates gradient from *all* of them.]

// ═══════════════════════════ PART II — Accumulation ═══════════════════════════
= Gradients accumulate at branches

== A branch must add the contribution from every path #D

If $x$ influences $cal(L)$ through several intermediate variables $u_1, dots, u_k$:
$ (partial cal(L))/(partial x) = sum_(i=1)^k (partial cal(L))/(partial u_i) (partial u_i)/(partial x). $
#pause
#result[gradients from every path *add*]

== Why gradients add at a branch #V

Imagine nudging $x$ from $4$ to $4.01$. The *same nudge* travels down both routes to the loss:
#pause
#branchgraph
#pause
#two(
  notebox[
    *Via $u = x^2$:* the local slope is $2x = 8$,
    so $Delta x = 0.01$ changes the loss by about $8 dot 0.01 = 0.08$.
  ],
  notebox[
    *Via $v = 3x$:* the local slope is $3$,
    so the same nudge changes the loss by about $3 dot 0.01 = 0.03$.
  ],
)
#v(7pt)
#result[total slope $=$ total loss change $div$ nudge $= (0.08 + 0.03) / 0.01 = 8 + 3 = 11$]

== Worked branch example

Let $u = x^2$, #h(0.4em) $v = 3x$, #h(0.4em) $cal(L) = u + v = x^2 + 3x$.
#pause
By direct calculus:
$ (dif cal(L))/(dif x) = 2x + 3, quad "at" x = 4: quad 2(4) + 3 = 11. $
#pause
Now let us get the *same* answer by backprop.

== Same answer, by backprop #D

*Forward* at $x = 4$: #h(0.5em) $u = 16$, #h(0.3em) $v = 12$, #h(0.3em) $cal(L) = 28$.
#pause
*Backward:* #h(0.5em) seed $overline(cal(L)) = 1$; add node gives $overline(u) = 1$, $overline(v) = 1$.
#pause
$ overline(x)_("via" u) = overline(u) dot 2x = 1 dot 8 = 8, quad overline(x)_("via" v) = overline(v) dot 3 = 3. $
#pause
$ overline(x) = 8 + 3 = 11. quad checkmark $

== PyTorch adds the same two paths #I

#codebox(size: 14pt)[```python
import torch

x = torch.tensor(4.0, requires_grad=True)
u = x**2                 # path 1
v = 3*x                  # path 2
L = u + v
L.backward()

print(x.grad)            # tensor(11.)
```]
#pause
#two(
  notebox[*via $u$:* $partial u / partial x = 2x = 8$],
  notebox[*via $v$:* $partial v / partial x = 3$],
)
#pause
#result[autograd accumulates both contributions: $8 + 3 = 11$]

== `+=` is the chain rule at a branch

Both paths write into the *same* $overline(x)$. If the second overwrote the first, we would lose the $8$.
#pause
#notebox[In PyTorch, `.grad` *accumulates* across `backward()` calls — that is exactly this rule. You must `optimizer.zero_grad()` each step, or gradients from old steps pile up.]

== Checkpoint: gradient accumulation #Q

#mcq(
  [Why is a branch node's backward update written with `+=` rather than `=`?],
  [It makes the forward pass faster],
  [The variable receives a contribution from every downstream path],
  [It changes multiplication into addition],
  [Autodiff cannot store scalar values],
)

== Answer: gradient accumulation #A

#mcq-answer([B], [The variable receives a contribution from every downstream path], [The chain rule sums all paths from the variable to the loss; assignment would discard an earlier contribution.])

// ═══════════════════════════ PART III — Central example ═══════════════════════════
= The central worked example

== A scalar linear model #D

Predict $hat(y) = w x + b$, squared-error loss $cal(L) = (hat(y) - y)^2$.
#pause
Concrete numbers:
$ x = 3, quad w = 2, quad b = 1, quad y = 10. $
#pause
#notebox[Four scalars, one loss. We will get $partial cal(L)$ w.r.t. *all four* by hand, then check with PyTorch.]

== Break into primitives

Decompose the loss into elementary nodes:
$ m = w x, quad a = m + b, quad e = a - y, quad cal(L) = e^2. $
#pause
#result[$times$, then $+$, then $-$, then $(dot)^2$]
#pause
Each is one of the nodes from our rule table.

== The computation graph #V

#scalargraph
#pause
#align(center, text(size: 16pt, fill: MUTED)[inputs (ink) → ops $times, +, -$ (teal) → loss $(dot)^2$ (orange)])

== Forward pass #D

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 6pt), align: (left, center, center),
  table.header([*Node*], [*Rule*], [*Value*]),
  [$m = w x$],   [$2 dot 3$],      [$6$],
  [$a = m + b$], [$6 + 1$],        [$7$],
  [$e = a - y$], [$7 - 10$],       [$-3$],
  [$cal(L) = e^2$], [$(-3)^2$],    [$9$],
))
#pause
#result[$cal(L) = 9$ — now run the graph *backward*]

== Backward: seed and the square node #D

Seed the output: $overline(cal(L)) = partial cal(L) \/ partial cal(L) = 1$.
#pause
Square node $cal(L) = e^2$, local $partial cal(L) \/ partial e = 2e$:
$ overline(e) = overline(cal(L)) dot 2e = 1 dot 2(-3) = -6. $

== Backward: the subtraction node #D

Node $e = a - y$, locals $partial e \/ partial a = 1$, $partial e \/ partial y = -1$:
#pause
$ overline(a) = overline(e) dot 1 = -6 $
#pause
$ overline(y) = overline(e) dot (-1) = 6 $
#pause
#notebox[The target $y$ already has a gradient — we would use $overline(y)$ if $y$ were itself a learned quantity.]

== Backward: the addition node #D

Node $a = m + b$ *copies* its gradient:
$ overline(m) = overline(a) = -6, quad quad overline(b) = overline(a) = -6. $
#pause
So $overline(b) = -6$ is one of our final answers.

== Backward: the multiplication node #D

Node $m = w x$ *sends the other input*:
#pause
$ overline(w) = overline(m) dot x = -6 dot 3 = -18 $
#pause
$ overline(x) = overline(m) dot w = -6 dot 2 = -12 $
#pause
#result[$overline(w) = -18, quad overline(x) = -12$]

== Final gradients #V

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 15pt, y: 7pt), align: center,
  table.header([$partial cal(L) \/ partial w$], [$partial cal(L) \/ partial b$], [$partial cal(L) \/ partial x$], [$partial cal(L) \/ partial y$]),
  [$-18$], [$-6$], [$-12$], [$6$],
))
#pause
#notebox[One backward sweep produced *all four* partial derivatives — no formula re-derivation per parameter.]

== The backward pass, visualized #V

Each node shows its forward value and its backward adjoint, *drawn by our autodiff from the real computation* — nothing typed by hand:
#pause
#cgraph(scalar-loss, (2, 3, 1, 10), names: ("w", "x", "b", "y"))
#pause
#align(center, text(size: 15pt, fill: MUTED)[$overline(w) = -18, overline(x) = -12, overline(b) = -6, overline(y) = 6$ — one reverse sweep, right to left])

== Check with direct calculus #D

Differentiate $cal(L) = (w x + b - y)^2$ directly. With $w x + b - y = -3$:
#pause
$ (partial cal(L))/(partial w) = 2(w x + b - y) x = 2(-3)(3) = -18 quad checkmark $
#pause
$ (partial cal(L))/(partial b) = 2(w x + b - y) = 2(-3) = -6 quad checkmark $
#pause
#result[backprop $=$ calculus — but mechanical & reusable]

== One derivative, three routes #V

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 9pt, y: 6pt), align: (left, left, center, left),
  table.header([*Route*], [*Mechanism*], [*$partial cal(L) / partial w$*], [*Best use*]),
  [symbolic calculus], [rewrite and differentiate the expression], [$-18$], [small algebra; insight],
  [finite difference], [perturb $w$ by $plus.minus epsilon$ and compare losses], [$approx -18$], [debugging a gradient],
  [reverse-mode AD], [replay local rules on the executed graph], [$-18$], [training large models],
))
#pause
#result[all target the same mathematical derivative; only the computation changes]

== Checkpoint: read the sign of a gradient #Q

#mcq(
  [We found $partial cal(L) \/ partial w = -18$. For a small positive learning rate, which way does gradient descent move $w$?],
  [Decrease $w$],
  [Increase $w$],
  [Leave $w$ unchanged],
  [The sign tells us nothing],
)

== Answer: a negative gradient means increase the parameter #A

#mcq-answer([B], [Increase $w$], [Gradient descent subtracts the gradient: $w <- w - eta(-18) = w + 18 eta$.])

== A small gradient step reduces this loss #D

Take $eta = 0.01$ and update:
$ w <- 2 - 0.01(-18) = 2.18, quad quad b <- 1 - 0.01(-6) = 1.06. $
#pause
New prediction: $hat(y) = 2.18 dot 3 + 1.06 = 7.60$.
#pause
#result[loss falls from $9$ to $(7.60 - 10)^2 = 5.76$]
#pause
#notebox[Backprop supplied the direction. The learning rate controls how far we move; a later optimization lecture studies that choice.]

== Interactive: the computation graph #I

#interbox(link-to: IA + "autograd")[
  Build a scalar computation graph, run the forward pass, then click *backward* to watch each adjoint $overline(u)$ fill in — exactly the $(w x + b - y)^2$ example above.
]
#pause
Trace which local rule changes when the final squared-error operation is replaced with $|e|$.

== Practice the scalar graph three ways #I

#grid(
  columns: 3, gutter: 10pt,
  notebox[
    *See it move*
    #v(5pt)
    Step through values and adjoints in the #link(IA + "autograd")[interactive graph ↗].
  ],
  notebox[
    *Build it by hand*
    #v(5pt)
    Run the #link(NB + "01_manual_scalar_backprop.ipynb")[manual-backprop notebook ↓].
  ],
  notebox[
    *Let PyTorch build it*
    #v(5pt)
    Run the #link(NB + "02_autograd_scalar.ipynb")[autograd notebook ↓].
  ],
)
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 5pt), align: (left, left),
  [*Easy*], [change one input and recompute the forward pass],
  [*Medium*], [derive all four gradients before calling `backward()`],
  [*Hard*], [replace the square by $|e|$; state what happens at $e = 0$],
))

// ═══════════════════════════ PART IV — Neuron ═══════════════════════════
= From scalar graph to a neuron

== A single neuron

Vector input, one output:
$ z = bold(w)^top bold(x) + b, quad a = phi(z), quad cal(L) = ell(a, y). $
#pause
#result[same three nodes: affine → activation → loss]

== Backprop through the activation #D

Let $delta := overline(z) = partial cal(L) \/ partial z$. Through $a = phi(z)$:
$ delta = overline(a) dot phi'(z), quad "where" overline(a) = (partial cal(L))/(partial a). $
#pause
#notebox[$delta$ is the *error signal at the pre-activation*. Everything below the neuron is expressed through this one scalar.]

== Backprop through the affine part #D

Through $z = bold(w)^top bold(x) + b$ with $overline(z) = delta$:
#pause
$ nabla_(bold(w)) cal(L) = delta thin bold(x), quad quad (partial cal(L))/(partial b) = delta, quad quad overline(bold(x)) = delta thin bold(w). $
#pause
#notebox[Weight gradient $=$ error $delta$ times the input $bold(x)$. This “error times input” shape reappears at every layer.]

== Sigmoid + BCE: the clean cancellation #D

Sigmoid $a = sigma(z)$, binary cross-entropy $cal(L) = -[y log a + (1 - y) log(1 - a)]$.
#pause
$ (partial cal(L))/(partial a) = -y/a + (1 - y)/(1 - a) $
#pause
$ (partial a)/(partial z) = a(1 - a) $
#pause
Multiply — the denominators cancel:
$ overline(z) = (partial cal(L))/(partial a) dot a(1 - a) = a - y. $
#pause
#result[$overline(z) = a - y$ — prediction minus target]

== Softmax + cross-entropy: same pattern #D

For multi-class, $bold(p) = "softmax"(bold(z))$ and $cal(L) = -log p_y$:
$ nabla_(bold(z)) cal(L) = bold(p) - bold(y), $
where $bold(y)$ is the one-hot target vector.
#pause
#result[$overline(bold(z)) = bold(p) - bold(y)$ — the identical “prediction $-$ target” form]
#pause
#notebox[Choosing the loss to *match* the output nonlinearity makes the output-layer gradient trivially simple.]

// ═══════════════════════════ PART V — Dense layer ═══════════════════════════
= The dense layer

== Forward: shapes matter #D

A dense layer maps $bold(x) in RR^n$ to $bold(z) in RR^m$:
$ bold(z) = bold(W) bold(x) + bold(b), quad bold(W) in RR^(m times n), quad bold(b) in RR^m. $
#pause
#notebox[Read the shapes off the equation: $bold(W)$ has one row per output and one column per input.]

== Dense-layer backward is outer product + transpose #D

Given upstream $overline(bold(z)) in RR^m$:
#pause
$ nabla_(bold(W)) cal(L) = overline(bold(z)) thin bold(x)^top quad (m times n), $
$ nabla_(bold(b)) cal(L) = overline(bold(z)) quad (m), quad quad overline(bold(x)) = bold(W)^top overline(bold(z)) quad (n). $
#pause
#result[$nabla_(bold(W)) cal(L) = overline(bold(z)) bold(x)^top$ · #h(0.3em) $overline(bold(x)) = bold(W)^top overline(bold(z))$]

== A $2 times 2$ dense layer makes the forward shapes concrete #D

Let
$ bold(x) = mat(2; -1), quad bold(W) = mat(1, 3; -2, 1), quad bold(b) = mat(0; 1). $
#pause
Then
$ bold(z) = bold(W) bold(x) + bold(b) = mat(1, 3; -2, 1) mat(2; -1) + mat(0; 1) = mat(-1; -4). $
#pause
#notebox[Two inputs, two outputs: $bold(W)$ and $nabla_(bold(W)) cal(L)$ must both be $2 times 2$.]

== The same $2 times 2$ layer backward, numerically #D

Suppose the loss sends $overline(bold(z)) = mat(4; -2)$ backward.
#pause
$ nabla_(bold(W)) cal(L) = overline(bold(z)) bold(x)^top = mat(4; -2) mat(2, -1) = mat(8, -4; -4, 2), $
#pause
$ nabla_(bold(b)) cal(L) = mat(4; -2), quad overline(bold(x)) = bold(W)^top overline(bold(z)) = mat(8; 10). $
#pause
#result[outer product for parameters · transpose for inputs]

== PyTorch verifies the dense-layer formulas #I

#codebox(size: 13pt)[```python
import torch

x = torch.tensor([2., -1.], requires_grad=True)
W = torch.tensor([[1., 3.], [-2., 1.]], requires_grad=True)
b = torch.tensor([0., 1.], requires_grad=True)

z = W @ x + b
z.backward(torch.tensor([4., -2.]))  # upstream z-bar

print(W.grad)  # [[ 8., -4.], [-4.,  2.]]
print(b.grad)  # [4., -2.]
print(x.grad)  # [8., 10.]
```]
#pause
#result[the outer product, bias gradient, and transposed input gradient all match]

== Batch version #D

Stack $N$ examples as rows: $bold(X) in RR^(N times n)$, $bold(Z) = bold(X) bold(W)^top + bold(1) bold(b)^top$.
#pause
Given $overline(bold(Z)) in RR^(N times m)$:
$ nabla_(bold(W)) cal(L) = overline(bold(Z))^top bold(X), quad quad nabla_(bold(b)) cal(L) = sum_(i=1)^N overline(bold(Z))_i. $
#pause
#notebox[Summing rows of $overline(bold(Z))$ for $nabla_(bold(b))$ is the branch rule: $bold(b)$ is shared by all $N$ examples, so its contributions accumulate.]

== Shape-checking is free debugging #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, center, center),
  table.header([*Quantity*], [*Shape*], [*Must match*]),
  [$bold(W)$],           [$m times n$], [$nabla_(bold(W)) cal(L)$],
  [$bold(b)$],           [$m$],         [$nabla_(bold(b)) cal(L)$],
  [$bold(x)$],           [$n$],         [$overline(bold(x))$],
  [$bold(z)$],           [$m$],         [$overline(bold(z))$],
))
#pause
#result[gradient shape $=$ parameter shape — always]

// ═══════════════════════════ PART VI — MLP ═══════════════════════════
= Backprop through a two-layer MLP

== Forward pass #D

$ bold(z)^((1)) = bold(W)_1 bold(x) + bold(b)_1, quad bold(h) = phi(bold(z)^((1))), $
$ bold(z)^((2)) = bold(W)_2 bold(h) + bold(b)_2, quad bold(p) = "softmax"(bold(z)^((2))), quad cal(L) = -log p_y. $
#pause
#result[affine → activation → affine → softmax → CE]

== The forward graph #V

#mlpforward
#pause
#align(center, text(size: 16pt, fill: MUTED)[backprop reverses every arrow, right to left])

== Backward: the output layer #D

Softmax + CE gives the clean output-layer signal:
$ overline(bold(z))^((2)) = bold(p) - bold(y). $
#pause
Then, using the dense-layer rules:
$ nabla_(bold(W)_2) cal(L) = overline(bold(z))^((2)) bold(h)^top, quad quad overline(bold(h)) = bold(W)_2^top overline(bold(z))^((2)). $

== Backward: the hidden and first layers #D

Through the activation $bold(h) = phi(bold(z)^((1)))$ (elementwise):
$ overline(bold(z))^((1)) = overline(bold(h)) dot.o phi'(bold(z)^((1))). $
#pause
Then the first affine layer:
$ nabla_(bold(W)_1) cal(L) = overline(bold(z))^((1)) bold(x)^top, quad quad nabla_(bold(b)_1) cal(L) = overline(bold(z))^((1)). $
#pause
#notebox[$dot.o$ is elementwise (Hadamard). Each activation coordinate depends only on its matching pre-activation, so the Jacobian is diagonal.]

== The complete recipe #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 5.5pt), align: (left, left),
  table.header([*Forward*], [*Backward*]),
  [$bold(z)^((1)) = bold(W)_1 bold(x) + bold(b)_1$], [$overline(bold(z))^((1)) = overline(bold(h)) dot.o phi'(bold(z)^((1)))$],
  [$bold(h) = phi(bold(z)^((1)))$], [$nabla_(bold(W)_1) = overline(bold(z))^((1)) bold(x)^top$],
  [$bold(z)^((2)) = bold(W)_2 bold(h) + bold(b)_2$], [$overline(bold(h)) = bold(W)_2^top overline(bold(z))^((2))$],
  [$bold(p) = "softmax"(bold(z)^((2)))$], [$nabla_(bold(W)_2) = overline(bold(z))^((2)) bold(h)^top$],
  [$cal(L) = -log p_y$],           [$overline(bold(z))^((2)) = bold(p) - bold(y)$],
))
#pause
#result[same two moves at every layer: $bold(delta) bold(x)^top$ for weights, $bold(W)^top bold(delta)$ for inputs]

== Interactive: MLP backprop #I

#interbox(link-to: IA + "autograd")[
  Step through a small MLP's forward pass, then watch $overline(bold(z))^((2)) = bold(p) - bold(y)$ propagate back through $bold(W)_2$, the activation, and $bold(W)_1$ — the adjoints light up layer by layer.
]
#pause
Trace the same upstream adjoint into the parameter and input branches of the affine layer.

== A tiny MLP uses the same recipe #I

#codebox(size: 12.5pt)[```python
import torch
from torch import nn
import torch.nn.functional as F

model = nn.Sequential(
    nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2)
)
x = torch.tensor([[2., -1.]])
target = torch.tensor([1])

loss = F.cross_entropy(model(x), target)
loss.backward()
for name, p in model.named_parameters():
    print(name, tuple(p.grad.shape))
```]
#pause
#notebox[`0.weight (3, 2)` · `0.bias (3,)` · `2.weight (2, 3)` · `2.bias (2,)`]
#pause
#result[each parameter receives a gradient of exactly the same shape]

== Checkpoint: dense-layer backward pass #Q

#mcq(
  [For $bold(z)=bold(W)bold(x)+bold(b)$, which pair uses the same upstream adjoint $overline(bold(z))$?],
  [$nabla_(bold(W))=overline(bold(z))bold(x)^top$ and $overline(bold(x))=bold(W)^top overline(bold(z))$],
  [$nabla_(bold(W))=bold(W)^top bold(x)$ and $overline(bold(x))=overline(bold(z)) bold(x)^top$],
  [$nabla_(bold(W))=bold(x)$ and $overline(bold(x))=bold(W)$],
  [$nabla_(bold(W))=bold(0)$ and $overline(bold(x))=bold(0)$],
)

== Answer: dense-layer backward pass #A

#mcq-answer([A], [$nabla_(bold(W))=overline(bold(z))bold(x)^top$ and $overline(bold(x))=bold(W)^top overline(bold(z))$], [Both branches apply the chain rule to the same output adjoint; they differ only in the local derivative of the affine operation.])

// ═══════════════════════════ PART VII — Autodiff ═══════════════════════════
= Automatic differentiation

== Backprop is reverse-mode automatic differentiation

All three routes answer the same question, but they do different work:
#pause
#grid(
  columns: 3, gutter: 10pt,
  notebox[*Symbolic* rewrites a mathematical expression into a derivative expression.],
  notebox[*Finite difference* probes nearby inputs and estimates a derivative numerically.],
  notebox[*Reverse-mode AD* replays exact local rules on the operations that actually ran.],
)
#pause
#result[backprop is reverse-mode AD specialized to neural-network computation graphs]

== A scalar loss makes reverse mode the right direction #D

A loss is a map $f: RR^P -> RR$: many parameter inputs, one output.
#pause
#two(
  notebox[*Forward mode:* to obtain the full gradient, run one directional sweep per parameter → $P$ sweeps.],
  notebox[*Reverse mode:* one reverse sweep per scalar output → one sweep for $cal(L)$.],
)
#pause
#result[one reverse sweep costs on the same order as a small number of forward evaluations]

== Autodiff records values forward and replays rules backward

During the forward pass the framework:
#pause
- *records* each primitive op and its inputs onto a tape (the graph);
#pause
- *stores* intermediate values (needed by local derivatives);
#pause
- on `backward()`: seeds $overline(cal(L)) = 1$, applies each op's local rule, and *accumulates* into `.grad`.
#pause
#result[you write only the forward pass — the backward pass is generated]

== PyTorch: the scalar example #I

#codebox[```python
import torch
x = torch.tensor(3.0)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(10.0)

L = (w*x + b - y)**2     # forward:  L = 9
L.backward()             # reverse-mode autodiff

print(L.item())          # 9.0
print(w.grad, b.grad)    # tensor(-18.)  tensor(-6.)
```]
#pause
#result[matches our hand computation exactly]

== The same graph, in *this* deck #I

Our own toolkit ships a tiny reverse-mode autodiff (chalkdust `autodiff`) — build the
expression, get *every* gradient in one backward pass. This slide runs it:
#pause
// computed live — these ARE the numbers rendered below (un-fakeable)
#let L = scalar-loss
#let gr = ad.grad(L, (2, 3, 1, 10)).map(g => int(calc.round(g)))
#codebox[```typ
#import "@local/chalkdust-autodiff:0.1.0" as ad
#let L = ad.expr("(w*x + b - y)^2", ("w", "x", "b", "y"))
#ad.value(L, (2, 3, 1, 10))   // the loss
#ad.grad(L,  (2, 3, 1, 10))   // (w̄, x̄, b̄, ȳ), one backward pass
```]
#pause
#result[computed live in this slide: $cal(L) = #int(calc.round(ad.value(L, (2, 3, 1, 10))))$, #h(0.6em)
  $overline(w) = #gr.at(0), thick overline(x) = #gr.at(1), thick overline(b) = #gr.at(2), thick overline(y) = #gr.at(3)$ — the same four numbers, by hand, by PyTorch, and by our own graph.]

== Gradients accumulate in frameworks

`.grad` is *added to*, never reset, by `backward()`. The standard loop is:
#pause
#codebox[```python
for x, y in loader:
    optimizer.zero_grad()      # clear old gradients
    loss = criterion(model(x), y)
    loss.backward()            # accumulate this batch's gradients
    optimizer.step()           # theta <- theta - eta * grad
```]
#pause
#alertbox[Forget `zero_grad()` and you sum gradients across steps — the same `+=` branch rule, now biting you.]

// ═══════════════════════════ PART VIII — Checking & flow ═══════════════════════════
= Gradient checking & gradient flow

== Finite differences check backprop, but cannot replace it #D

To *debug* an analytic gradient, compare against a central difference:
$ (partial cal(L))/(partial theta_j) approx (cal(L)(bold(theta) + epsilon bold(e)_j) - cal(L)(bold(theta) - epsilon bold(e)_j))/(2 epsilon). $
#pause
#notebox[Useful in small float64 unit tests; far too slow for training — a central difference needs *two* loss evaluations per parameter.]

== Gradient check: finite difference versus PyTorch #I

#codebox(size: 13pt)[```python
import torch

def loss(w):
    return (w*3.0 + 1.0 - 10.0)**2

w, eps = 2.0, 1e-5
g_fd = (loss(w + eps) - loss(w - eps)) / (2*eps)

wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
loss(wt).backward()
g_ad = wt.grad.item()

print(g_fd, g_ad)        # both approximately -18
print(abs(g_fd - g_ad))  # numerical error near zero
```]
#pause
#notebox[Use float64. Too-large $epsilon$ gives truncation error; too-small $epsilon$ gives cancellation error. Check a few coordinates—not millions.]

== Gradient flow in deep chains #D

Stack $L$ layers $h_0 -> h_1 -> dots -> h_L$. The chain rule multiplies Jacobians:
$ (partial cal(L))/(partial bold(h)_0) = (partial cal(L))/(partial bold(h)_L) product_(ell=1)^L (partial bold(h)_ell)/(partial bold(h)_(ell-1)). $
#pause
#result[a *product* of $L$ terms — it can *vanish* or *explode*]

== Sigmoid saturation #V

$sigma'(z) = sigma(z)(1 - sigma(z)) <= 1/4$, and $approx 0$ for large $|z|$:
#pause
#fig("/lecture3/figures/sigmoid_deriv.svg", w: 46%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[each sigmoid layer multiplies the gradient by at most $1 \/ 4$])

== Vanishing gradients #V

Multiply many small local derivatives → the signal *decays* with depth:
#pause
#fig("/lecture3/figures/grad_flow.svg", w: 44%)
#pause
#notebox[Sigmoid derivatives can repeatedly shrink the signal. ReLU has slope $1$ on active units, but slope $0$ on inactive units; weights and normalization also affect the full Jacobian.]

== Checkpoint: repeated local slopes #Q

#mcq(
  [Along a scalar chain, ten consecutive local derivatives equal $0.5$. What factor reaches the earliest node?],
  [$5$],
  [$0.5$],
  [$0.5^10 approx 0.001$],
  [$10^0.5$],
)

== Answer: products can erase the signal #A

#mcq-answer([C], [$0.5^10 approx 0.001$], [Backprop multiplies local derivatives along a path; several individually moderate slopes can yield a tiny product.])

== Interactive: gradient flow #I

#interbox(link-to: IA + "vanishing-gradients")[
  Sweep depth and activation function; watch the gradient magnitude at layer $0$ collapse for sigmoid and survive for ReLU.
]
#pause
Sweep depth to see repeated small local slopes rapidly erase an early-layer gradient.

== Practice ladder: hand → PyTorch → finite difference #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 6pt), align: (left, left, left),
  table.header([*Level*], [*Problem*], [*Verification*]),
  [easy], [$v = (a + b)^2$ at $a = 2, b = -1$], [derive $overline(a), overline(b)$; check with PyTorch],
  [medium], [$cal(L) = x^2 + 3x$ at $x = 4$], [show the two path contributions sum to $11$],
  [hard], [$cal(L) = ell(bold(W) bold(x) + bold(b), y)$], [check one entry of $nabla_(bold(W))$ by finite difference],
))
#pause
#result[solve by hand first · verify the whole gradient with autograd · spot-check numerically]

// ═══════════════════════════ PART IX — Summary ═══════════════════════════
= Summary

== Backprop in one sentence

#align(center, text(size: 22pt)[
  Seed $overline(cal(L)) = 1$, then walk the graph *backward*, at each node computing
  #v(6pt)
  *downstream $=$ upstream $times$ local*,
  #v(6pt)
  and *adding* gradients wherever a variable branched.
])
#pause
#align(center, text(size: 16pt, fill: MUTED)[For vector nodes, “times local” becomes a vector–Jacobian product; the logic is unchanged.])

== The formulas to keep #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 7pt), align: (left, left),
  table.header([*Setting*], [*Backward rule*]),
  [scalar node],        [$overline(u) = overline(v) dot partial v \/ partial u$],
  [dense layer],        [$nabla_(bold(W)) = overline(bold(z)) bold(x)^top, quad overline(bold(x)) = bold(W)^top overline(bold(z))$],
  [activation],         [$overline(bold(z)) = overline(bold(a)) dot.o phi'(bold(z))$],
  [softmax + CE],       [$overline(bold(z)) = bold(p) - bold(y)$],
  [branch point],       [gradients *add*],
))

== What comes next

We can now compute $nabla_(bold(theta)) cal(L)$ for any network by composing local rules.
#pause
#result[Next (Lecture 3A): the geometry behind these objects — gradients, Jacobians, and Hessians]
#pause
#notebox[Today used derivatives operationally. Lecture 3A makes their shapes and geometry explicit; the later optimization lecture then uses the gradient to choose parameter steps.]

#focus-slide[
  We can compute $nabla_(bold(theta)) cal(L)$.
  #v(12pt)
  #set text(size: 22pt)
  Next: what gradients, Jacobians, and Hessians mean geometrically.
  #v(8pt)
  #text(size: 17pt, fill: rgb("#B9C6C8"))[Lecture 3A · Derivatives, Gradients, Jacobians & Hessians]
]
