// Calculus Toolkit for Deep Learning — Lecture 3A · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture3a/L3a-calculus-toolkit.typ
//   typst compile --root . --input handout=true lecture3a/L3a-calculus-toolkit.typ

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [Derivatives, Gradients, Jacobians & Hessians],
  subtitle: [The calculus behind optimization and backpropagation],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
// chain-rule graph  x -> u -> v
#let chain3 = align(center, diagram(spacing: 16mm, node-stroke: 0.9pt + INK, node-fill: white, {
  node((0,0), $x$, radius: 6mm); node((1,0), $u$, radius: 6mm, stroke: 0.9pt + TEAL); node((2,0), $v$, radius: 6mm, stroke: 0.9pt + ACC)
  edge((0,0),(1,0), $f$, "-|>", stroke: 0.8pt + MUTED); edge((1,0),(2,0), $g$, "-|>", stroke: 0.8pt + MUTED)
}))
// backprop as a right-to-left chain of vector–Jacobian products
#let vjp-chain = align(center, diagram(spacing: (21mm, 13mm), node-stroke: 0.9pt, {
  let fwd = ($x$, $h_1$, $h_2$, $cal(L)$)
  for (i, l) in fwd.enumerate() {
    let c = if i == 0 { INK } else if i == 3 { ACC } else { TEAL }
    node((i, 0), l, radius: 6mm, stroke: c)
  }
  for i in range(3) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.8pt + MUTED) }
  let bwd = ($overline(x)$, $overline(h)_1$, $overline(h)_2$, $1$)
  for (i, l) in bwd.enumerate() {
    let c = if i == 0 { INK } else if i == 3 { ACC } else { TEAL }
    node((i, 1), l, radius: 6mm, fill: rgb("#EFEEEB"), stroke: c)
  }
  edge((3, 0), (3, 1), text(size: 14pt)[seed $1$], "-|>", stroke: 0.8pt + MUTED, label-side: right)
  edge((3, 1), (2, 1), $J^top$, "-|>", stroke: 0.9pt + GREEN, label-side: left)
  edge((2, 1), (1, 1), $J^top$, "-|>", stroke: 0.9pt + GREEN, label-side: left)
  edge((1, 1), (0, 1), $J^top$, "-|>", stroke: 0.9pt + GREEN, label-side: left)
}))

#title-slide()

// ═══════════════════════════ PART I — Why ═══════════════════════════
== Why a calculus lecture?

Every step of training is a gradient step:
#pause
$ theta_(t+1) = theta_t - eta thin nabla_theta cal(L)(theta_t) $
#pause
So we must be fluent with
#pause
$ dif/(dif x), quad (partial)/(partial x_i), quad nabla f, quad J, quad H. $
#pause
#notebox[These five objects are the entire vocabulary of optimization and backprop.]

== The big picture #V

Five objects, one theme — each is a *derivative* of a function of a particular shape:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 7pt), align: (left, center, left),
  table.header([*Object*], [*Function type*], [*Tells us*]),
  [derivative $f'(x)$], [$RR -> RR$], [slope],
  [partial $(partial f)/(partial x_i)$], [$RR^d -> RR$], [sensitivity to one coordinate],
  [gradient $nabla f$], [$RR^d -> RR$], [steepest-ascent direction],
  [Jacobian $J$], [$RR^n -> RR^m$], [local linear map],
  [Hessian $H$], [$RR^d -> RR$], [curvature],
))

// ═══════════════════════════ PART II — Scalar derivative ═══════════════════════════
= The scalar derivative

== Derivative as slope #V

$ f'(x) = (dif f)/(dif x) = lim_(epsilon -> 0) (f(x + epsilon) - f(x))/epsilon $
#pause
#fig("/lecture3a/figures/tangent.svg", w: 44%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[the secant slope → the tangent slope as $epsilon -> 0$])

== The derivative is a local linear model #D

Near a point $x$:
#pause
$ f(x + Delta x) approx f(x) + f'(x) thin Delta x $
#pause
#result[derivative = best local *linear* approximation]
#pause
Example: $f(x) = x^2$ at $x = 3$, so $f(3) = 9$ and $f'(3) = 6$:
#pause
$ f(3 + Delta x) approx 9 + 6 thin Delta x. $

== Local linear model: numeric check (cont.) #D

Test the line $f(3 + Delta x) approx 9 + 6 thin Delta x$ at a *small* step $Delta x = 0.1$:
#pause
Linear prediction: #h(0.4em) $9 + 6 dot 0.1 = 9.6$.
#pause
True value: #h(0.4em) $f(3.1) = 3.1^2 = 9.61$.
#pause
Error: #h(0.4em) $9.61 - 9.6 = 0.01$ — tiny.
#pause
#notebox[Now a *large* step $Delta x = 1$: the line gives $9 + 6 = 15$, but $f(4) = 16$ — error $1$. The linear model degrades as $Delta x$ grows.]

== Interactive: tangent line #I

#interbox(link-to: IA + "derivative-tangent")[
  drag the point and $Delta x$; watch the tangent, the secant, and the approximation error.
  Functions: $x^2$, $sin x$, $e^x$, $log(1 + e^x)$.
]
#pause
Explore how the approximation error grows as the step leaves the local tangent regime.

== Checkpoint: local linearization #Q

#mcq(
  [When is a first-order linear approximation most reliable?],
  [For a large step on a sharply curved function],
  [For a small step where curvature is modest],
  [Only at a global minimum],
  [Whenever the gradient is zero],
)

== Answer: local linearization #A

#mcq-answer([B], [For a small step where curvature is modest], [The derivative captures local slope; omitted higher-order terms become important over large steps or sharp bends.])

// ═══════════════════════════ PART III — Partials & surfaces ═══════════════════════════
= Partial derivatives and surfaces

== From curves to surfaces #V

A function of two inputs is a *surface*: $f(x, y) = x^2 + y^2$.
#pause
#fig("/lecture3a/figures/bowl3d.svg", w: 33%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[input $(x, y)$ → height $z = f(x, y)$])

== Partial derivatives

$(partial f)/(partial x)$ — change $x$, *hold $y$ fixed*. #h(1em) $(partial f)/(partial y)$ — change $y$, hold $x$ fixed.
#pause
For $f(x, y) = x^2 + y^2$:
#pause
$ (partial f)/(partial x) = 2x, quad quad (partial f)/(partial y) = 2y $
#pause
At $(x, y) = (3, 1)$: #h(0.4em) $(partial f)/(partial x) = 6$, #h(0.4em) $(partial f)/(partial y) = 2$.

== A partial is the slope of a slice #V

Fix one input and you are left with a 1-D curve — the partial is *its* slope.
#pause
#fig("/lecture3a/figures/surface_slices.svg", w: 31%)
#pause
#notebox[For $f = x^2 + 3y^2$ at $(1, 2)$: $(partial f)/(partial x) = 2$, $(partial f)/(partial y) = 12$ — the function climbs *faster* along $y$.]

== Interactive: surface slices #I

#interbox(link-to: IA + "surface-partials")[
  move a point on a 3-D surface and read off the two slice slopes.
  Functions: $x^2 + y^2$, $x^2 + 3y^2$, $sin x cos y$.
]
#pause
Use the two slice slopes to compare the axes at a chosen point.

// ═══════════════════════════ PART IV — Gradient ═══════════════════════════
= The gradient

== Gradient: stack the partials

$ nabla f(x) = vec((partial f)/(partial x_1), (partial f)/(partial x_2), dots.v, (partial f)/(partial x_d)) $
#pause
For $f(x, y) = x^2 + y^2$: #h(0.6em) $nabla f = vec(2x, 2y)$.
#pause
At $(3, 1)$: #h(0.4em) $nabla f = vec(6, 2)$ — pointing *outward*, uphill from the bowl's centre.

== Gradient = local linear model #D

Same first-order Taylor idea, now with a *dot product*:
#pause
$ f(x + Delta x) approx f(x) + nabla f(x)^top Delta x $
#pause
#result[the vector version of $f(x + Delta x) approx f(x) + f'(x) Delta x$]

== Directional derivative

For a *unit* direction $u$:
#pause
$ D_u f(x) = nabla f(x)^top u $
#pause
#notebox[The rate of change along $u$ is just the *projection* of the gradient onto $u$.]
#pause
Example: $f = x^2 + y^2$ at $(1, 2)$, so $nabla f = vec(2, 4)$. Take $u = (3, 4) \/ 5 = (0.6, 0.8)$:
#pause
$ D_u f = 2 dot 0.6 + 4 dot 0.8 = 1.2 + 3.2 = 4.4 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[compare $norm(nabla f) = sqrt(20) approx 4.47$ — the *gradient* direction is a touch steeper.])

== The gradient points uphill #D

By Cauchy–Schwarz, over all unit $u$:
#pause
$ nabla f(x)^top u <= norm(nabla f(x)) $
#pause
with equality at $u = nabla f(x) \/ norm(nabla f(x))$.
#pause
#result[$nabla f$ = steepest ascent · $-nabla f$ = steepest descent]

== The gradient is perpendicular to contours #V

A contour is the set $f(x, y) = c$. Along it, $f$ does not change, so for any tangent $v$: $nabla f^top v = 0$.
#pause
#fig("/lecture3a/figures/contours_grad.svg", w: 29%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[so $nabla f$ is *orthogonal* to the contour — it points straight across the level lines.])

== Gradient descent on the contour map

$ x_(t+1) = x_t - eta thin nabla f(x_t) $
#pause
Example: at $x_t = (3, 1)$ with $nabla f = (6, 2)$ and $eta = 0.1$:
#pause
$ x_(t+1) = (3, 1) - 0.1 (6, 2) = (2.4, 0.8) $
#pause
#notebox[Each step moves *perpendicular* to the current contour, downhill. The learning rate $eta$ sets the step length.]

== Interactive: gradient field & descent #I

#interbox(link-to: IA + "optimizer-race")[
  Watch gradient arrows on a contour map and a descent trajectory as you change $eta$ and the start point.
  Surfaces: $x^2 + y^2$, $x^2 + 10 y^2$, $sin x + cos y$.
]
#pause
Move the start point to the bottom of the surface and inspect the update.

== Checkpoint: a stationary point #Q

#mcq(
  [At a differentiable local minimum, what does a plain gradient-descent step do?],
  [Moves uphill],
  [Leaves the point unchanged because $nabla f=0$],
  [Doubles the learning rate],
  [Must move along a contour],
)

== Answer: a stationary point #A

#mcq-answer([B], [Leaves the point unchanged], [At a differentiable local minimum the gradient is zero, so $x_{t+1}=x_t-eta nabla f(x_t)=x_t$.])

// ═══════════════════════════ PART V — Hessian ═══════════════════════════
= The Hessian and curvature

== Second derivative in 1-D

$ f''(x) = (dif^2 f)/(dif x^2) quad = quad "how fast the slope changes" $
#pause
$f(x) = x^2 => f'' = 2$ (bowl, curves *up*);
#pause
#h(0.6em) $f(x) = -x^2 => f'' = -2$ (hill, curves *down*).

== Convex, concave, flat #V

The *sign* of $f''$ names the shape:
#pause
#fig("/lecture3a/figures/curvature_1d.svg", w: 78%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[$f'' > 0$ convex (bowl) · $f'' < 0$ concave (hill) · $f'' = 0$ flat (inflection)])

== The Hessian: matrix of second partials

$ H = nabla^2 f, quad H_(i j) = (partial^2 f)/(partial x_i partial x_j) $
#pause
For two variables:
#pause
$ H = mat((partial^2 f)/(partial x^2), (partial^2 f)/(partial x partial y); (partial^2 f)/(partial y partial x), (partial^2 f)/(partial y^2)) $

== Hessian example #D

$ f(x, y) = x^2 + 3y^2 quad ==> quad nabla f = vec(2x, 6y) $
#pause
Differentiate each partial again, one entry at a time:
#pause
$ partial_(x x) = partial/(partial x)(2x) = 2 $
#pause
$ partial_(y y) = partial/(partial y)(6y) = 6, quad quad partial_(x y) = partial/(partial y)(2x) = 0 $
#pause
$ H = mat(2, 0; 0, 6) $
#pause
#notebox[Curvature is *larger* in the $y$ direction — the bowl is steeper that way.]

== Second-order local approximation #D

The full second-order Taylor expansion:
#pause
$ f(x + Delta x) approx f(x) + underbrace(nabla f(x)^top Delta x, "tilt (gradient)") + underbrace(1/2 Delta x^top H(x) Delta x, "curvature (Hessian)") $
#pause
#result[gradient tilts · Hessian bends]

== Hessian geometry #OPT

For a quadratic $f(x) = 1/2 x^top A x$: #h(0.6em) $nabla f = A x$, #h(0.4em) $H = A$.
#pause
- *eigenvectors* of $H$ → the curvature *directions*;
#pause
- *eigenvalues* of $H$ → the curvature *magnitudes*.

== Classifying critical points

At a point where $nabla f = 0$, the Hessian eigenvalues decide the shape:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 6pt), align: (left, left),
  table.header([*Hessian eigenvalues*], [*Local shape*]),
  [all positive], [local minimum],
  [all negative], [local maximum],
  [mixed signs], [saddle point],
  [some zero], [inconclusive / flat],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[the eigenvalue *signs* alone classify the point.])

== A saddle point #V

$ f(x, y) = x^2 - y^2, quad nabla f = vec(2x, -2y), quad H = mat(2, 0; 0, -2) $
#pause
Eigenvalues $+2$ and $-2$ have *mixed signs* → a saddle.
#pause
#fig("/lecture3a/figures/saddle.svg", w: 25%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[$nabla f = 0$ at the origin, but it is *not* a minimum])

== Why curvature matters for optimization #V

The gradient picks a *direction*, but curvature decides a good *step size*.
#pause
#fig("/lecture3a/figures/gd_zigzag.svg", w: 66%)
#pause
#notebox[Elongated contours (a large *condition number*) make first-order gradient descent *zigzag* — much curvature one way, little the other.]

== Interactive: curvature & descent #I

#interbox(link-to: IA + "optimizer-race")[
  Reshape a quadratic $a x^2 + b y^2 + c x y$, see its Hessian eigenvectors, and watch the descent path zigzag.
]
#pause
Observe how one shared step size alternately overshoots the steep direction and creeps along the flat one.

// ═══════════════════════════ PART VI — Jacobian ═══════════════════════════
= The Jacobian

== When the gradient is not enough

The gradient is for *scalar* outputs, $f: RR^d -> RR$.
#pause
But network layers map *vectors to vectors*, $f: RR^n -> RR^m$ — e.g. $z = W x + b$.
#pause
We need the *Jacobian*.

== Jacobian: matrix of all partials

For $f: RR^n -> RR^m$:
#pause
$ J = (partial f)/(partial x) in RR^(m times n), quad quad J_(i j) = (partial f_i)/(partial x_j) $
#pause
#result[$f(x + Delta x) approx f(x) + J(x) thin Delta x$ — a local *linear map*]

== Two Jacobians you already know #D

Two building blocks whose Jacobians you can write down instantly:
#pause
#two(
  notebox[*Linear layer* $f(x) = W x + b$:
   $ J = W $
   since $(partial z_i)/(partial x_j) = W_(i j)$.],
  notebox[*Elementwise* $h = phi(z)$:
   $ J = "diag"(phi'(z_1), dots, phi'(z_m)) $
   (for ReLU, entries are $0$ or $1$).],
)

== Worked example: a $2 -> 2$ Jacobian #D

$f(x_1, x_2) = vec(x_1^2 + x_2, sin(x_1 x_2))$ — the map from the PyTorch demo later.
#pause
Row 1, partials of $f_1 = x_1^2 + x_2$: #h(0.4em) $(partial f_1)/(partial x_1) = 2 x_1$, #h(0.3em) $(partial f_1)/(partial x_2) = 1$.
#pause
Row 2, partials of $f_2 = sin(x_1 x_2)$: #h(0.4em) $(partial f_2)/(partial x_1) = x_2 cos(x_1 x_2)$, #h(0.3em) $(partial f_2)/(partial x_2) = x_1 cos(x_1 x_2)$.
#pause
$ J = mat(2 x_1, 1; x_2 cos(x_1 x_2), x_1 cos(x_1 x_2)) $
#pause
At $(x_1, x_2) = (1, 2)$: #h(0.4em) $J = mat(2, 1; 2 cos 2, cos 2)$.
#pause
#notebox[Each entry is one partial — a $2 times 2$ local linear map, built without ever storing a big matrix.]

== The chain rule, with Jacobians #D

#chain3
#pause
$ (partial v)/(partial x) = (partial v)/(partial u) (partial u)/(partial x) quad ==> quad J_(v,x) = J_(v,u) thin J_(u,x) $
#pause
Shapes: with $x in RR^2, u in RR^3, v in RR^2$, then $J_(u,x) in RR^(3 times 2)$ and $J_(v,u) in RR^(2 times 3)$:
#pause
$ J_(v,x) = J_(v,u) thin J_(u,x) in RR^(2 times 2) $
#pause
#notebox[The inner dimension ($3$) cancels — exactly like ordinary matrix multiplication.]

== Why we avoid full Jacobians #OPT

A single layer with $x, h in RR^4096$ has a Jacobian of $4096 times 4096 approx 1.7 times 10^7$ entries.
#pause
That is one dense matrix *per layer, per example* — far too big to form and multiply.
#pause
#alertbox[Frameworks never materialize $J$. They compute *vector–Jacobian products* instead.]

== Vector–Jacobian product = backprop #D

If $y = f(x)$ and $cal(L) = cal(L)(y)$:
#pause
$ underbrace(overline(x), (partial cal(L))/(partial x)) = J^top thin underbrace(overline(y), (partial cal(L))/(partial y)) $
#pause
#result[one VJP per layer — no matrix ever formed]

== Backprop is a chain of VJPs

For $x -> h_1 -> h_2 -> cal(L)$:
$ nabla_x cal(L) = J_(h_1, x)^top thin J_(h_2, h_1)^top thin nabla_(h_2) cal(L) $
#pause
#v(4pt)
#vjp-chain
#pause
Evaluate *right to left*: $overline(h)_2 -> overline(h)_1 -> overline(x)$ — vectors only.

== Interactive: the Jacobian as a local warp #I

#interbox(link-to: IA + "jacobian-local-map")[
  a 2-D map warps a grid; the local Jacobian sends a small circle to an ellipse.
]
#pause
#fig("/lecture3a/figures/jacobian_warp.svg", w: 58%)

// ═══════════════════════════ PART VII — Where each appears ═══════════════════════════
= Where each object shows up in deep learning

== Derivative → activations

Scalar activation derivatives drive backprop through nonlinearities:
#pause
$ dif/(dif z) sigma(z) = sigma(z)(1 - sigma(z)) $
#pause
$ "ReLU"'(z) = bb(1)[z > 0] $

== Gradient → learning & attribution

- parameter updates: $theta <- theta - eta thin nabla_theta cal(L)$;
#pause
- saliency maps: $nabla_x f_y (x)$;
#pause
- adversarial examples: $x_"adv" = x + epsilon thin "sign"(nabla_x cal(L))$.

== Jacobian → sensitivity & flows #OPT

Appears in backprop, sensitivity analysis, normalizing flows ($log abs(det J)$), neural ODEs, implicit layers.
#pause
#notebox[But almost always accessed through a product $J v$ or $J^top v$ — never the full matrix.]

== Hessian → curvature & second-order #OPT

Newton's method $x_(t+1) = x_t - H^(-1) nabla f$, Laplace approximation, sharpness / flatness, influence functions.
#pause
#notebox[But the full $H$ is usually too large — we use Hessian–vector products $H v$ instead.]

== Why DL stays first-order

For $P$ parameters, $nabla_theta cal(L) in RR^P$ but $H in RR^(P times P)$.
#pause
#result[$P = 10^8 ==> H$ has $10^16$ entries]
#pause
So practice uses SGD, momentum, Adam, and diagonal / Hessian-vector approximations.

// ═══════════════════════════ PART VIII — Autodiff ═══════════════════════════
= The autodiff connection

== Autodiff is none of the naive options

#two(
  notebox[*not* symbolic differentiation],
  notebox[*not* finite differences],
)
#pause
Automatic differentiation applies the chain rule to the *executed program* — exact, and cheap.

== Scalar loss → reverse mode

Training maps $theta in RR^P -> cal(L) in RR$ (many inputs, one output).
#pause
#result[reverse-mode autodiff computes $nabla_theta cal(L)$ in ~one forward-pass cost]

== Non-scalar output needs a seed vector #D

If $y = f(x) in RR^m$, `backward()` must know *which* scalar to differentiate — you supply a vector $v$ and it returns (same vector as the $J^top overline(y)$ above, shaped like $x$)
#pause
$ J^top v. $
#pause
#notebox[This is why PyTorch errors on `.backward()` of a non-scalar tensor without a `gradient=` argument.]

== PyTorch: Jacobian and VJP #I

Compute the very Jacobian we found by hand — then a VJP:
#pause
#codebox[```python
import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)
def f(x):
    return torch.stack([x[0]**2 + x[1], torch.sin(x[0]*x[1])])

J = torch.autograd.functional.jacobian(f, x)   # full 2x2 Jacobian
v = torch.tensor([1.0, 3.0])
f(x).backward(v)                                # x.grad == J^T v  (a VJP)
```]
#pause
#align(center, text(size: 15pt, fill: MUTED)[`jacobian` forms the whole matrix; `backward(v)` returns only $v^top J$.])

== PyTorch: the Hessian #I

#codebox[```python
def g(x):
    return x[0]**2 + 3*x[1]**2 + x[0]*x[1]

H = torch.autograd.functional.hessian(g, x)     # -> [[2, 1], [1, 6]]
```]
#pause
#align(center)[$H = mat(2, 1; 1, 6)$ — matches the second partials by hand.]

// ═══════════════════════════ PART IX — Summary ═══════════════════════════
= Summary

== One table to remember #V

Six objects, six shapes, six jobs:
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 7pt), align: (left, center, left),
  table.header([*Object*], [*Shape*], [*Main DL use*]),
  [$f'(x)$], [scalar], [activation derivative],
  [$nabla f$], [$d$], [parameter update],
  [$J$], [$m times n$], [vector-map sensitivity],
  [$J^top v$], [$n$], [backpropagation],
  [$H$], [$d times d$], [curvature],
  [$H v$], [$d$], [second-order methods],
))

== Final mental model — one idea

#align(center, text(size: 21pt)[
  derivative $=$ slope \
  gradient $=$ direction of steepest ascent \
  Jacobian $=$ local linear map \
  Hessian $=$ curvature \
  *backprop $=$ an efficient chain of vector–Jacobian products*
])

#focus-slide[
  We can now read every derivative object in deep learning.
  #v(12pt)
  #set text(size: 22pt)
  Next: *computation graphs and backpropagation* — the chain rule, run backward through the program.
]
