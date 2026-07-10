// Computation Graphs, Backpropagation and Autodiff — Lecture 3 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture3/L3-backprop.typ
//   typst compile --root . --input handout=true lecture3/L3-backprop.typ
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [Computation Graphs, Backpropagation and Autodiff],
  subtitle: [How gradients flow through neural networks],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"

// ── local fletcher diagrams ──────────────────────────────────────────
// one local node:  u -> v -> L
#let localnode = align(center, diagram(spacing: (26mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), $u$, radius: 6mm)
  node((1,0), $v$, radius: 6mm, stroke: 0.9pt + TEAL)
  node((2,0), $cal(L)$, radius: 6mm, stroke: 0.9pt + ACC)
  edge((0,0),(1,0), $v = f(u)$, "-|>", stroke: 0.8pt + MUTED, label-sep: 8pt)
  edge((1,0),(2,0), "-|>", stroke: 0.8pt + MUTED)
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

// scalar graph, BACKWARD: adjoint values flow right -> left (upstream x local)
#let scalarback = align(center, diagram(spacing: (17mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,2),  $w$, radius: 5.5mm)
  node((0,1),  $x$, radius: 5.5mm)
  node((0,0),  $b$, radius: 5.5mm)
  node((0,-1.4), $y$, radius: 5.5mm)
  node((1.4,1.5), $times$, radius: 6mm, fill: rgb("#E6F0F0"), stroke: 0.9pt + TEAL)
  node((2.8,0.7), $+$, radius: 6mm, fill: rgb("#E6F0F0"), stroke: 0.9pt + TEAL)
  node((4.2,-0.3), $-$, radius: 6mm, fill: rgb("#E6F0F0"), stroke: 0.9pt + TEAL)
  node((5.6,-0.3), [$(dot)^2$], radius: 7mm, fill: rgb("#FDECD6"), stroke: 0.9pt + ACC)
  // seed at the loss
  node((6.7,-0.3), text(size: 15pt, fill: ACC)[$overline(cal(L))=1$], stroke: none, fill: none)
  // final adjoint for each input, placed to its left (out of the way of the arrows)
  node((-1.25,2),   text(size: 14pt, fill: ACC)[$overline(w)=-18$], stroke: none, fill: none)
  node((-1.25,1),   text(size: 14pt, fill: ACC)[$overline(x)=-12$], stroke: none, fill: none)
  node((-1.25,0),   text(size: 14pt, fill: ACC)[$overline(b)=-6$],  stroke: none, fill: none)
  node((-1.25,-1.4),text(size: 14pt, fill: ACC)[$overline(y)=6$],   stroke: none, fill: none)
  // op-chain backward edges carry the propagating adjoint (upstream x local)
  edge((5.6,-0.3),(4.2,-0.3), text(size: 14pt)[$overline(e)=-6$], "-|>", stroke: 1pt + ACC, label-side: right)
  edge((4.2,-0.3),(2.8,0.7), text(size: 14pt)[$overline(a)=-6$], "-|>", stroke: 1pt + ACC, label-side: left)
  edge((2.8,0.7),(1.4,1.5), text(size: 14pt)[$overline(m)=-6$], "-|>", stroke: 1pt + ACC, label-side: right)
  // branch-off backward edges to the inputs (plain orange arrows)
  edge((4.2,-0.3),(0,-1.4), "-|>", stroke: 1pt + ACC)
  edge((2.8,0.7),(0,0), "-|>", stroke: 1pt + ACC)
  edge((1.4,1.5),(0,2), "-|>", stroke: 1pt + ACC)
  edge((1.4,1.5),(0,1), "-|>", stroke: 1pt + ACC)
}))

// two-layer MLP forward chain
#let mlpforward = align(center, diagram(spacing: 12mm, node-stroke: 0.9pt + INK, node-fill: white, {
  let labs = ($x$, $z^((1))$, $h$, $z^((2))$, $p$, $cal(L)$)
  for (i, l) in labs.enumerate() {
    let c = if i == labs.len() - 1 { ACC } else if i == 0 { INK } else { TEAL }
    node((i, 0), l, radius: 6mm, stroke: 0.9pt + c)
  }
  for i in range(labs.len() - 1) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.8pt + MUTED) }
}))

#title-slide()

// ═══════════════════════════ PART I — Vocabulary ═══════════════════════════
= Backprop vocabulary

== The goal: two passes over one graph #V

Training is a *forward* pass and a *backward* pass over the same computation graph:
#pause
#align(center, diagram(spacing: 15mm, node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), $x$, radius: 6mm); node((1,0), $f_theta$, radius: 7mm, stroke: 0.9pt + TEAL); node((2,0), $cal(L)$, radius: 6mm, stroke: 0.9pt + ACC)
  edge((0,0),(1,0), "-|>", stroke: 1pt + INK, label: text(fill: INK)[forward])
  edge((2,0),(1,0), "-|>", stroke: 1pt + ACC, bend: -40deg, label: text(fill: ACC)[backward $nabla$])
}))
#pause
#result[forward: $x -> cal(L)$ · #h(0.3em) backward: $cal(L) -> nabla_theta cal(L)$]

== Training needs gradients

Every update step is
$ theta_(t+1) = theta_t - eta thin nabla_theta cal(L)(theta_t). $
#pause
A modern network has $P = 10^7$ to $10^11$ parameters.
#pause
#alertbox[We need *all* $P$ partial derivatives $partial cal(L) \/ partial theta_j$ — cheaply, and exactly. Backprop delivers them in *one backward pass*.]

== One local node

Zoom in on a single edge of the graph: $u -> v -> cal(L)$.
#pause
#localnode
#pause
We want $overline(u) := partial cal(L) \/ partial u$, and we are *given* $overline(v) := partial cal(L) \/ partial v$ from the node ahead.

== Upstream, local, downstream #D

The chain rule splits the work into two pieces we already have:
$ underbrace(partial cal(L)/partial u, "downstream") = underbrace(partial cal(L)/partial v, "upstream") dot underbrace(partial v/partial u, "local") $
#pause
#result[downstream $=$ upstream $times$ local]
#pause
#notebox[*Upstream* comes from the node ahead. *Local* is the derivative of *this* node alone. Their product flows *downstream*.]

== The rule, in bar notation #D

Write $overline(u) = partial cal(L) \/ partial u$. Then every node obeys
$ overline(u) = overline(v) dot (partial v)/(partial u). $
#pause
#notebox[The node only needs to know its *own* local derivative $partial v \/ partial u$. It multiplies whatever gradient $overline(v)$ arrives from above — no global bookkeeping.]

== Example: a square node #D

Node $v = u^2$, so the local derivative is $partial v \/ partial u = 2u$.
#pause
Suppose $u = 3$ (forward) and the upstream gradient is $overline(v) = 7$. Then
$ overline(u) = overline(v) dot 2u = 7 dot 6 = 42. $
#pause
#result[downstream $overline(u) = 42$]

== Addition node: copy the gradient #D

Node $v = a + b$. Local derivatives $partial v \/ partial a = 1$, $partial v \/ partial b = 1$.
#pause
$ overline(a) = overline(v) dot 1 = overline(v), quad quad overline(b) = overline(v) dot 1 = overline(v). $
#pause
#result[$+$ *copies* the upstream gradient to every input]

== Multiplication node: swap the inputs #D

Node $v = a b$. Local derivatives $partial v \/ partial a = b$, $partial v \/ partial b = a$.
#pause
$ overline(a) = overline(v) dot b, quad quad overline(b) = overline(v) dot a. $
#pause
#result[$times$ *sends the other input*]
#pause
#notebox[If $a = 2, b = 5, overline(v) = 3$: then $overline(a) = 3 dot 5 = 15$ and $overline(b) = 3 dot 2 = 6$.]

== Activation node #D

Node $v = phi(u)$ (elementwise nonlinearity). Local derivative $partial v \/ partial u = phi'(u)$.
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

== One variable, many paths #D

If $x$ influences $cal(L)$ through several intermediate variables $u_1, dots, u_k$:
$ (partial cal(L))/(partial x) = sum_(i=1)^k (partial cal(L))/(partial u_i) (partial u_i)/(partial x). $
#pause
#result[gradients from every path *add*]

== Gradients add at branches #V

When $x$ fans out, its gradient is the *sum* over out-edges:
#pause
#branchgraph
#pause
$ overline(x) = overline(x)_("via" u) + overline(x)_("via" v). $

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

== Why `+=` matters

Both paths write into the *same* $overline(x)$. If the second overwrote the first, we would lose the $8$.
#pause
#notebox[In PyTorch, `.grad` *accumulates* across `backward()` calls — that is exactly this rule. You must `optimizer.zero_grad()` each step, or gradients from old steps pile up.]

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
$ overline(a) = overline(e) dot 1 = -6, quad quad overline(y) = overline(e) dot (-1) = 6. $
#pause
#notebox[The target $y$ already has a gradient — we would use $overline(y)$ if $y$ were itself a learned quantity.]

== Backward: the addition node #D

Node $a = m + b$ *copies* its gradient:
$ overline(m) = overline(a) = -6, quad quad overline(b) = overline(a) = -6. $
#pause
So $overline(b) = -6$ is one of our final answers.

== Backward: the multiplication node #D

Node $m = w x$ *sends the other input*:
$ overline(w) = overline(m) dot x = -6 dot 3 = -18, quad quad overline(x) = overline(m) dot w = -6 dot 2 = -12. $
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

Seed $overline(cal(L)) = 1$ at the loss; each #text(fill: ACC)[orange arrow] carries *upstream $times$ local* one node to the left:
#pause
#scalarback
#pause
#align(center, text(size: 16pt, fill: MUTED)[same graph as the forward pass — every arrow reversed, adjoints filling in right to left])

== Check with direct calculus #D

Differentiate $cal(L) = (w x + b - y)^2$ directly. With $w x + b - y = -3$:
#pause
$ (partial cal(L))/(partial w) = 2(w x + b - y) x = 2(-3)(3) = -18 quad checkmark $
#pause
$ (partial cal(L))/(partial b) = 2(w x + b - y) = 2(-3) = -6 quad checkmark $
#pause
#result[backprop $=$ calculus — but mechanical & reusable]

== One gradient-descent step #D

Take $eta = 0.1$ and update:
$ w <- 2 - 0.1(-18) = 3.8, quad quad b <- 1 - 0.1(-6) = 1.6. $
#pause
New prediction: $hat(y) = 3.8 dot 3 + 1.6 = 13$.
#pause
#notebox[Target is $y = 10$; we jumped from $6$ to $13$ — we *overshot*. Lecture 4 is about choosing $eta$ and the update rule so this is stable.]

== Interactive: the computation graph #I

#interbox(link-to: IA + "autograd")[
  Build a scalar computation graph, run the forward pass, then click *backward* to watch each adjoint $overline(u)$ fill in — exactly the $(w x + b - y)^2$ example above.
]
#pause
*Q.* Which node's local derivative changes if we swap squared-error for absolute error $|e|$?

// ═══════════════════════════ PART IV — Neuron ═══════════════════════════
= From scalar graph to a neuron

== A single neuron

Vector input, one output:
$ z = w^top x + b, quad a = phi(z), quad cal(L) = ell(a, y). $
#pause
#result[same three nodes: affine → activation → loss]

== Backprop through the activation #D

Let $delta := overline(z) = partial cal(L) \/ partial z$. Through $a = phi(z)$:
$ delta = overline(a) dot phi'(z), quad "where" overline(a) = (partial cal(L))/(partial a). $
#pause
#notebox[$delta$ is the *error signal at the pre-activation*. Everything below the neuron is expressed through this one scalar.]

== Backprop through the affine part #D

Through $z = w^top x + b$ with $overline(z) = delta$:
#pause
$ nabla_w cal(L) = delta thin x, quad quad (partial cal(L))/(partial b) = delta, quad quad overline(x) = delta thin w. $
#pause
#notebox[Parameter gradient $=$ error $delta$ times the *input* $x$. This "$delta$ times input" shape reappears at every layer.]

== Sigmoid + BCE: the clean cancellation #D

Sigmoid $a = sigma(z)$, binary cross-entropy $cal(L) = -[y log a + (1 - y) log(1 - a)]$.
#pause
$ (partial cal(L))/(partial a) = -y/a + (1 - y)/(1 - a), quad quad (partial a)/(partial z) = a(1 - a). $
#pause
Multiply — the denominators cancel:
$ overline(z) = (partial cal(L))/(partial a) dot a(1 - a) = a - y. $
#pause
#result[$overline(z) = a - y$ — prediction minus target]

== Softmax + cross-entropy: same pattern #D

For multi-class, $p = "softmax"(z)$ and $cal(L) = -log p_y$:
$ nabla_z cal(L) = p - y, $
where $y$ is the one-hot target.
#pause
#result[$overline(z) = p - y$ — the identical "prediction $-$ target" form]
#pause
#notebox[Choosing the loss to *match* the output nonlinearity makes the output-layer gradient trivially simple.]

// ═══════════════════════════ PART V — Dense layer ═══════════════════════════
= The dense layer

== Forward: shapes matter #D

A dense layer maps $x in RR^n$ to $z in RR^m$:
$ z = W x + b, quad W in RR^(m times n), quad b in RR^m. $
#pause
#notebox[Read the shapes off the equation: $W$ has one row per output, one column per input.]

== Backward through a dense layer #D

Given upstream $overline(z) in RR^m$:
#pause
$ nabla_W cal(L) = overline(z) thin x^top quad (m times n), quad quad nabla_b cal(L) = overline(z), quad quad overline(x) = W^top overline(z) quad (n). $
#pause
#result[$nabla_W = overline(z) x^top$ · #h(0.3em) $overline(x) = W^top overline(z)$]

== Batch version #D

Stack $N$ examples as rows: $X in RR^(N times n)$, $Z = X W^top + bb(1) b^top$.
#pause
Given $overline(Z) in RR^(N times m)$:
$ nabla_W cal(L) = overline(Z)^top X, quad quad nabla_b cal(L) = sum_(i=1)^N overline(Z)_i. $
#pause
#notebox[Summing rows of $overline(Z)$ for $nabla_b$ is just the branch rule: $b$ is *shared* across all $N$ examples, so its gradients accumulate.]

== Shape-checking is free debugging #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, center, center),
  table.header([*Quantity*], [*Shape*], [*Must match*]),
  [$W$],           [$m times n$], [$nabla_W$],
  [$b$],           [$m$],         [$nabla_b$],
  [$x$],           [$n$],         [$overline(x)$],
  [$z$],           [$m$],         [$overline(z)$],
))
#pause
#result[gradient shape $=$ parameter shape — always]

// ═══════════════════════════ PART VI — MLP ═══════════════════════════
= Backprop through a two-layer MLP

== Forward pass #D

$ z^((1)) = W_1 x + b_1, quad h = phi(z^((1))), $
$ z^((2)) = W_2 h + b_2, quad p = "softmax"(z^((2))), quad cal(L) = -log p_y. $
#pause
#result[affine → activation → affine → softmax → CE]

== The forward graph #V

#mlpforward
#pause
#align(center, text(size: 16pt, fill: MUTED)[backprop reverses every arrow, right to left])

== Backward: the output layer #D

Softmax + CE gives the clean output-layer signal:
$ overline(z)^((2)) = p - y. $
#pause
Then, using the dense-layer rules:
$ nabla_(W_2) cal(L) = overline(z)^((2)) h^top, quad quad overline(h) = W_2^top overline(z)^((2)). $

== Backward: the hidden and first layers #D

Through the activation $h = phi(z^((1)))$ (elementwise):
$ overline(z)^((1)) = overline(h) dot.o phi'(z^((1))). $
#pause
Then the first affine layer:
$ nabla_(W_1) cal(L) = overline(z)^((1)) x^top, quad quad nabla_(b_1) cal(L) = overline(z)^((1)). $
#pause
#notebox[$dot.o$ is elementwise (Hadamard) — the activation Jacobian is diagonal.]

== The complete recipe #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 5.5pt), align: (left, left),
  table.header([*Forward*], [*Backward*]),
  [$z^((1)) = W_1 x + b_1$],       [$overline(z)^((1)) = overline(h) dot.o phi'(z^((1)))$],
  [$h = phi(z^((1)))$],            [$nabla_(W_1) = overline(z)^((1)) x^top, #h(0.3em) nabla_(b_1) = overline(z)^((1))$],
  [$z^((2)) = W_2 h + b_2$],       [$overline(h) = W_2^top overline(z)^((2))$],
  [$p = "softmax"(z^((2)))$],      [$nabla_(W_2) = overline(z)^((2)) h^top, #h(0.3em) nabla_(b_2) = overline(z)^((2))$],
  [$cal(L) = -log p_y$],           [$overline(z)^((2)) = p - y$],
))
#pause
#result[same two moves at every layer: $delta x^top$ for weights, $W^top delta$ downstream]

== Interactive: MLP backprop #I

#interbox(link-to: IA + "autograd")[
  Step through a small MLP's forward pass, then watch $overline(z)^((2)) = p - y$ propagate back through $W_2$, the activation, and $W_1$ — the adjoints light up layer by layer.
]
#pause
*Q.* Why does the *same* $overline(z)$ appear in both $nabla_W$ and $overline(x)$ for a layer?

// ═══════════════════════════ PART VII — Autodiff ═══════════════════════════
= Automatic differentiation

== Backprop is reverse-mode autodiff

Backprop is not two things you might guess:
#pause
#two(
  alertbox[*not* symbolic differentiation — no giant closed-form expression],
  alertbox[*not* finite differences — no $epsilon$-perturbations],
)
#pause
#result[it is *reverse-mode automatic differentiation*: the chain rule applied to the executed program]

== Why reverse mode? #D

A loss is a map $f: RR^P -> RR$: *many* inputs, *one* output.
#pause
#two(
  notebox[*Forward mode:* one pass *per input* → $O(P)$.],
  notebox[*Reverse mode:* one pass *per output* → $O(1)$.],
)
#pause
#result[$P approx 10^9$ inputs, one scalar loss ⇒ reverse mode wins]

== What PyTorch / JAX actually do

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

== Finite-difference gradient check #D

To *debug* an analytic gradient, compare against a central difference:
$ (partial cal(L))/(partial theta_j) approx (cal(L)(theta + epsilon e_j) - cal(L)(theta - epsilon e_j))/(2 epsilon). $
#pause
#notebox[Great for *unit tests* ($epsilon approx 10^(-5)$), far too slow for training — one loss evaluation *per parameter*.]

== Gradient flow in deep chains #D

Stack $L$ layers $h_0 -> h_1 -> dots -> h_L$. The chain rule multiplies Jacobians:
$ (partial cal(L))/(partial h_0) = (partial cal(L))/(partial h_L) product_(ell=1)^L (partial h_ell)/(partial h_(ell-1)). $
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
#notebox[Sigmoid decays fast; ReLU (local slope $0$ or $1$) keeps gradients alive.]

== Interactive: gradient flow #I

#interbox(link-to: IA + "vanishing-gradients")[
  Sweep depth and activation function; watch the gradient magnitude at layer $0$ collapse for sigmoid and survive for ReLU.
]
#pause
*Q.* If gradients shrink by $times 1\/4$ per layer, how deep before they underflow to zero?

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

== The formulas to keep #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 13pt, y: 7pt), align: (left, left),
  table.header([*Setting*], [*Backward rule*]),
  [scalar node],        [$overline(u) = overline(v) dot partial v \/ partial u$],
  [dense layer],        [$nabla_W = overline(z) x^top, quad overline(x) = W^top overline(z)$],
  [activation],         [$overline(z) = overline(a) dot.o phi'(z)$],
  [softmax + CE],       [$overline(z) = p - y$],
  [branch point],       [gradients *add*],
))

== What comes next

We can now compute $nabla_theta cal(L)$ for *any* network, exactly and cheaply.
#pause
#result[Next (Lecture 4): how to *use* it — SGD, momentum, Adam]
#pause
#notebox[Backprop gives the *direction*; optimization decides the *step*.]

#focus-slide[
  We can compute $nabla_theta cal(L)$.
  #v(12pt)
  #set text(size: 22pt)
  Next: how to *use* it — SGD, momentum, Adam.
]
