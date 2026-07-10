// From Linear Models to Neural Networks — Lecture 2 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture2/L2-linear-to-mlp.typ
//   typst compile --root . --input handout=true lecture2/L2-linear-to-mlp.typ
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#show: metropolis-deck.with(
  title: [From Linear Models to Neural Networks],
  subtitle: [Linear & logistic regression, neurons, MLPs, universal approximation],
)

// interactive base + a small local compute graph (fletcher)
#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let compgraph = align(center, diagram(spacing: 13mm, node-stroke: 0.9pt + INK, node-fill: white, {
  let labs = ($x$, $h^((1))$, $h^((2))$, $z$, $p$, $cal(L)$)
  for (i, l) in labs.enumerate() {
    let c = if i == labs.len() - 1 { ACC } else if i == 0 { INK } else { TEAL }
    node((i, 0), l, radius: 6mm, stroke: 0.9pt + c)
  }
  for i in range(labs.len() - 1) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.8pt + MUTED) }
}))

// stacked-depth schematic: affine → nonlinearity → affine → … → output
#let depth-schematic = align(center, diagram(spacing: (8mm, 8mm), node-stroke: 0.9pt, {
  node((0,0), $x$, radius: 5mm, stroke: INK)
  node((1,0), [$W_1 dot + b_1$], shape: fletcher.shapes.rect, fill: rgb("#E3EDED"), stroke: TEAL, inset: 6pt)
  node((2,0), $phi$, radius: 5mm, fill: rgb("#FDECD6"), stroke: ACC)
  node((3,0), [$W_2 dot + b_2$], shape: fletcher.shapes.rect, fill: rgb("#E3EDED"), stroke: TEAL, inset: 6pt)
  node((4,0), $phi$, radius: 5mm, fill: rgb("#FDECD6"), stroke: ACC)
  node((5,0), $dots.h$, stroke: none)
  node((6,0), [$W_L dot + b_L$], shape: fletcher.shapes.rect, fill: rgb("#E3EDED"), stroke: TEAL, inset: 6pt)
  node((7,0), $z$, radius: 5mm, stroke: ACC)
  for i in range(7) { edge((i,0),(i+1,0), "-|>", stroke: 0.8pt + MUTED) }
}))

#title-slide()

// ═══════════════════════════ PART I — Where we left off ═══════════════════════════
== Recall from Lecture 1

Every loss was a negative log-likelihood, $ell = -log p_theta (y | x)$:
#pause
#two(
  notebox[MSE $arrow.l.r$ *Gaussian* likelihood],
  notebox[cross-entropy $arrow.l.r$ *Bernoulli / Categorical* likelihood],
)
#pause
#result[Today: $p_theta (y | x)$ is parameterized by a *neural network*.]

== The key transition
#stag(Q)

#two(r: (1fr, 1.1fr),
  [*Old ML models* — one affine map:
   $ f_theta (x) = w^top x + b $],
  [*Neural networks* — many affine maps + nonlinearities:
   $ f_theta (x) = W_L thin phi(W_(L-1) phi(dots.h phi(W_1 x + b_1))) + b_L $],
)
#pause
#alertbox[*Main question.* What changes when we replace *one* affine map by *many* affine maps separated by nonlinearities?]

// ═══════════════════════════ PART II — Linear regression ═══════════════════════════
= Linear regression

== The linear regression model

For $x in RR^d$:
$ hat(y) = w^top x + b $
#pause
Absorb the bias by appending a 1 to the input:
$ hat(y) = theta^top tilde(x), quad quad tilde(x) = vec(1, x) $
#pause
#notebox[One weight vector, one hyperplane. This is the whole model.]

== Probabilistic view #D

Assume Gaussian noise around the line:
$ Y | x, w, b tilde cal(N)(w^top x + b, sigma^2) $
#pause
$ -log p(y | x, w, b) = (y - w^top x - b)^2/(2 sigma^2) + C $
#pause
#result[linear regression + MSE = Gaussian MLE]

== Matrix form

Dataset $X in RR^(n times d)$, targets $y in RR^n$:
$ hat(y) = X w + b bold(1) quad ==> quad hat(y) = tilde(X) theta $
#pause
Loss over the whole dataset:
$ cal(L)(theta) = 1/n norm(y - tilde(X) theta)_2^2 $

== Geometry of linear regression #V

#fig("/lecture2/figures/linreg_geometry.svg", w: 46%)
#pause
#notebox[Linear regression fits a *hyperplane*: a line for $d=1$, a plane for $d=2$, a hyperplane in general. Residuals are the vertical gaps.]

== Closed-form solution #OPT

If $tilde(X)^top tilde(X)$ is invertible:
$ hat(theta) = (tilde(X)^top tilde(X))^(-1) tilde(X)^top y $
#pause
But deep learning uses *gradient-based* optimization instead. Why?
#pause
- huge parameter count · nonlinear models · minibatches · streaming data

== Gradient descent preview

For the MSE loss $cal(L)(theta) = 1/n norm(y - tilde(X) theta)_2^2$:
#pause
$ nabla_theta cal(L) = -2/n thin tilde(X)^top (y - tilde(X) theta) $
#pause
$ theta <- theta - eta thin nabla_theta cal(L) $
#pause
#notebox[This update *is* what backpropagation will compute — for far more complex $f_theta$.]

== Interactive: linear regression & gradient descent #I

#interbox(link-to: IA + "optimizer-race")[
  Fit a line by gradient descent — watch the *loss surface* and the *parameter trajectory* as you change the learning rate.
  Controls: slope $w$ · intercept $b$ · noise · learning rate · GD steps.
]
#pause
*Q.* What happens when the learning rate is too large?

// ═══════════════════════════ PART III — Logistic regression ═══════════════════════════
= Logistic regression

== A linear score is not a probability

The linear score is unbounded:
$ z = w^top x + b in RR $
#pause
But a binary label needs
$ p(y=1 | x) in [0, 1]. $
#pause
#result[squash it: $p(y=1 | x) = sigma(z)$]

== The sigmoid function #V

$ sigma(z) = 1/(1 + e^(-z)) $
#pause
#two(r: (1fr, 1.1fr),
  [Properties:
   - $sigma(z) in (0, 1)$
   - $sigma(0) = 0.5$
   - $sigma(-z) = 1 - sigma(z)$],
  fig("/lecture2/figures/sigmoid.svg", w: 92%),
)

== The logistic regression model

$ z = w^top x + b $
#pause
$ p_theta (y=1 | x) = sigma(z), quad quad p_theta (y=0 | x) = 1 - sigma(z) $
#pause
$ Y | x tilde "Bernoulli"(sigma(w^top x + b)) $

== Binary cross-entropy loss #D

For $y in {0, 1}$ with $hat(p) = sigma(w^top x + b)$:
$ cal(L) = -y log hat(p) - (1 - y) log(1 - hat(p)) $
#pause
This is exactly the Bernoulli NLL of the compact form
$ p(y | x) = hat(p)^y (1 - hat(p))^(1 - y). $

== Decision boundary #V

$ hat(y) = cases(1 & sigma(w^top x + b) >= 0.5, 0 & "otherwise") $
#pause
Since $sigma(z) >= 0.5 <==> z >= 0$, the boundary is
$ w^top x + b = 0. $
#pause
#fig("/lecture2/figures/linear_boundary.svg", w: 38%)

== Logistic regression is still a linear classifier

#two(
  notebox[The *output probability* is nonlinear in $z$ (the sigmoid).],
  notebox[The *decision boundary* is linear in $x$: $w^top x + b = 0$.],
)
#pause
#result[logistic regression = linear classifier + sigmoid]

== Gradient intuition #D

With $hat(p) = sigma(w^top x + b)$ and BCE loss, the algebra collapses:
$ (partial cal(L))/(partial z) = hat(p) - y $
#pause
$ nabla_w cal(L) = (hat(p) - y) thin x $
#pause
#notebox[Same "predicted − target" form as Lecture 1's softmax gradient. Keep an eye on it.]

== Interactive: logistic regression #I

#interbox(link-to: IA + "softmax-temperature")[
  Move a decision line over 2-D points and watch the *probability heatmap*, the *boundary*, and the *BCE loss* respond.
  Controls: line angle · bias · sigmoid steepness · threshold · add/remove points.
]
#pause
*Q.* How does the decision boundary move when $b$ changes?

// ═══════════════════════════ PART IV — From logistic regression to a neuron ═══════════════════════════
= From logistic regression to a neuron

== A neuron

A neuron applies an affine map, then an activation:
$ z = w^top x + b, quad quad a = phi(z) $
#pause
#neuron-diagram(d: 3)

== Logistic regression is one neuron

Take the activation to be the sigmoid:
$ a = sigma(w^top x + b) = p(y=1 | x) $
#pause
#result[logistic regression is the simplest neural network — a single sigmoid neuron]

== Common activation functions #V

#fig("/lecture2/figures/activations.svg", w: 88%)
#pause
#text(size: 17pt)[sigmoid → gates/probabilities · tanh → zero-centred · *ReLU* → default hidden unit · GELU → Transformers]

== Why we need a nonlinearity #D

Stack two *linear* layers:
$ h_1 = W_1 x + b_1, quad h_2 = W_2 h_1 + b_2 $
#pause
$ h_2 = underbrace(W_2 W_1, "one matrix") x + underbrace((W_2 b_1 + b_2), "one vector") $
#pause
#alertbox[Without a nonlinear activation, *depth collapses to a single linear layer.*]

== Interactive: activation functions #I

#interbox(link-to: IA + "vanishing-gradients")[
  Compare activations and their *derivatives*, and see where gradients vanish.
  Controls: activation type · input $z$ · show derivative · compare saturation.
]
#pause
*Q.* Where does the sigmoid saturate, and why might that hurt training?

// ═══════════════════════════ PART V — Multi-class classification ═══════════════════════════
= Multi-class classification

== From binary to multi-class

#two(
  [*Binary* — one score:
   $ z = w^top x + b $],
  [*Multi-class* — one score per class:
   $ z_k = w_k^top x + b_k, thin k = 1, dots, K $],
)
#pause
#notebox[Each class gets its own logit; stack them into $z = W x + b$.]

== Softmax regression

$ p(y=k | x) = exp(z_k) / (sum_(j=1)^K exp(z_j)), quad z = W x + b $
#pause
Also called *multinomial logistic regression*. Generalizes the sigmoid to $K$ classes.

== Multi-class cross-entropy #D

For a one-hot target $y$:
$ cal(L) = -sum_(k=1)^K y_k log p_k $
#pause
Only the true class survives:
$ cal(L) = -log p_"true class" $

== Geometry of softmax regression #V

Boundary between classes $i, j$ occurs when $z_i = z_j$:
$ (w_i - w_j)^top x + (b_i - b_j) = 0 $
#pause
#fig("/lecture2/figures/softmax_regions.svg", w: 38%)
#align(center, text(size: 16pt, fill: MUTED)[straight edges → linear / polyhedral regions])

== The limitation of softmax regression #V

#two(r: (1.1fr, 1fr),
  [Softmax regression only draws *linear* boundaries. It fails on:
   - XOR
   - concentric circles
   - spirals
   - real image / audio / text patterns],
  fig("/lecture2/figures/xor.svg", w: 92%),
)

== Interactive: softmax classifier #I

#interbox(link-to: IA + "softmax-temperature")[
  Three class weight vectors, a movable point, and a temperature $T$ — watch the logits, the softmax probabilities, and the *linear decision regions*.
]
#pause
*Q.* Why are the decision regions straight-edged?

// ═══════════════════════════ PART VI — Multilayer perceptron ═══════════════════════════
= The multilayer perceptron

== Add a hidden layer

Instead of $z = W x + b$, insert a nonlinear layer first:
$ h = phi(W_1 x + b_1), quad quad z = W_2 h + b_2 $
#pause
For classification, $p = "softmax"(z)$. This is a *one-hidden-layer MLP*.

== MLP architecture #V

#two(r: (1.25fr, 1fr),
  mlp-diagram((3, 5, 5, 2)),
  [
    $x in RR^d$ \
    $W_1 in RR^(m times d)$ \
    $h in RR^m$ \
    $W_2 in RR^(K times m)$ \
    $z in RR^K$
  ],
)

== Each hidden neuron creates a feature

$ h_j = phi(w_j^top x + b_j) $
#pause
- the first layer *learns features*;
- the output layer *combines* them.
#pause
For ReLU, $h_j = max(0, w_j^top x + b_j)$ — each unit is active on one side of a hyperplane.

== The feature-transformation view #V

#fig("/lecture2/figures/feature_transform.svg", w: 62%)
#pause
#result[an MLP learns a representation where classification becomes linear]

== MLP for binary vs multi-class classification

#two(
  [*Binary*
   $ h = phi(W_1 x + b_1) $
   $ z = w_2^top h + b_2 $
   $ p = sigma(z), quad cal(L) = "BCE" $],
  [*Multi-class*
   $ h = phi(W_1 x + b_1) $
   $ z = W_2 h + b_2 $
   $ p = "softmax"(z), quad cal(L) = "CE" $],
)
#pause
#notebox[Same output layer as before — only the *representation* $h$ changed.]

== Multiple hidden layers

$ h^((1)) = phi(W_1 x + b_1), quad h^((2)) = phi(W_2 h^((1)) + b_2), quad dots.h $
$ z = W_L h^((L-1)) + b_L $
#pause
#v(4pt)
#depth-schematic
#align(center, text(size: 16pt, fill: MUTED)[affine $arrow.r$ nonlinearity $arrow.r$ affine, stacked $L$ deep])
#pause
Then $p = "softmax"(z)$ for classification, or $hat(y) = z$ for regression.

== Depth vs width #V

#two(
  notebox[*Width* — more neurons per layer → more features at one level.],
  notebox[*Depth* — more layers → *compose* features into a hierarchy.],
)
#pause
#align(center, text(size: 17pt)[
  pixels → edges → motifs → objects #h(1.4em) characters → words → phrases → meaning
])

== Interactive: MLP decision boundaries #I

#interbox(link-to: IA + "mlp-decision-boundary")[
  train a small MLP on *linear / XOR / circles / spirals* and watch the boundary form.
  Controls: hidden units · layers · activation · learning rate · steps.
]
#pause
*Q.* What changes when you add hidden *units* vs when you add *layers*?

// ═══════════════════════════ PART VII — Universal approximation ═══════════════════════════
= Why MLPs represent complex functions

== ReLU networks are piecewise linear #V

$ phi(z) = max(0, z) $
#pause
- one ReLU → a hinge;
#pause
- a sum of ReLUs → a piecewise-linear curve;
#pause
- many ReLUs → arbitrarily fine approximation.

== Building functions from ReLUs #V

A one-hidden-layer scalar network:
$ f(x) = sum_(j=1)^m a_j thin "ReLU"(w_j x + b_j) + c $
#pause
- each term is one hinge, switched on past $x = -b_j \/ w_j$;
#pause
- $w_j$ sets its slope, $a_j$ scales it, $c$ shifts the whole curve.

== Building functions from ReLUs (cont.) #V

#fig("/lecture2/figures/relu_build.svg", w: 50%)
#pause
#align(center, text(size: 16pt, fill: MUTED)[each hidden unit adds one kink — sum them to trace any piecewise-linear shape])

== The universal approximation theorem #D

Let $f$ be continuous on a compact domain. A feedforward network with *one* hidden layer and a suitable nonlinearity can approximate $f$ arbitrarily well:
#pause
$ forall thin epsilon > 0, thick exists thin "MLP" g quad "s.t." quad sup_x abs(f(x) - g(x)) < epsilon $

== What the theorem does *not* say #OPT

Universal approximation does *not* promise:
#pause
- that training will *find* that function;
#pause
- that *finite data* is enough;
#pause
- that a *small* network suffices;
#pause
- that it will *generalize*.
#pause
#result[expressivity $eq.not$ trainability $eq.not$ generalization]

== Why depth still matters

Even though one hidden layer is universal, depth can be *exponentially* more efficient — and composition is natural:
$ f(x) = f_L compose f_(L-1) compose dots.h compose f_1 (x) $
#pause
#notebox[Images, language, and program-like computation are all *hierarchical* — depth mirrors that structure.]

== Interactive: function approximation #I

#interbox(link-to: IA + "universal-approximation")[
  Nielsen's visual proof, by hand: crank a sigmoid into a step, glue steps into bumps, stack bumps to *design any 1-D curve*, then tile towers across a 3-D surface.
]
#pause
*Q.* Does more expressivity always make optimization easier?

// ═══════════════════════════ PART VIII — Connecting to practice ═══════════════════════════
= Connecting to deep-learning practice

== One example through an MLP

Forward pass, then a single scalar loss:
#pause
#compgraph
#pause
#align(center)[For multi-class: $cal(L) = -log p_y$. Next lecture reverses these arrows.]

== Parameters of an MLP #D

Layer $ell$: $W_ell in RR^(d_ell times d_(ell-1))$, $b_ell in RR^(d_ell)$. Total:
$ sum_(ell=1)^L (d_ell thin d_(ell-1) + d_ell) $
#pause
*Exercise.* Input 784, hidden 128, output 10.
#pause
Layer 1: #h(0.4em) $784 times 128 + 128 = "100,480"$ #h(0.4em) (weights $+$ biases).
#pause
Layer 2: #h(0.4em) $128 times 10 + 10 = "1,290"$.
#pause
#result[total $= "100,480" + "1,290" = "101,770"$]

== The standard supervised DL template

#two(
  [*Regression*
   $ z = f_theta (x) $
   $ cal(L) = (y - z)^2 $],
  [*Binary*
   $ z = f_theta (x), thin p = sigma(z) $
   $ cal(L) = "BCE" $],
)
#pause
#align(center)[*Multi-class:* $z = f_theta (x), thin p = "softmax"(z), thin cal(L) = "CE"$.]
#pause
#result[pick the output + likelihood; the loss follows]

== Summary #V

#align(center, table(
  columns: 5, stroke: 0.5pt + MUTED, inset: (x: 9pt, y: 6pt), align: (left, center, center, center, center),
  table.header([*Model*], [*Output*], [*Distribution*], [*Loss*], [*Boundary*]),
  [Linear regression], [$w^top x + b$], [Gaussian], [MSE], [—],
  [Logistic regression], [$sigma(w^top x + b)$], [Bernoulli], [BCE], [linear],
  [Softmax regression], [$"softmax"(W x + b)$], [Categorical], [CE], [linear],
  [MLP classifier], [$"softmax"(f_theta (x))$], [Categorical], [CE], [*nonlinear*],
))

== Final mental model — one idea

#align(center, text(size: 20pt)[
  linear regression $=$ linear score $+$ Gaussian likelihood \
  logistic regression $=$ linear score $+$ sigmoid $+$ Bernoulli \
  softmax regression $=$ linear scores $+$ softmax $+$ categorical \
  *MLP $=$ learned nonlinear features $+$ the same output likelihood*
])

// closing standout
#focus-slide[
  We now have $x -> f_theta (x) -> cal(L)$.
  #v(12pt)
  #set text(size: 22pt)
  Next: how to get $display((partial cal(L))/(partial theta))$
  #v(8pt)
  #text(size: 17pt, fill: rgb("#B9C6C8"))[computation graphs · the chain rule · backpropagation · automatic differentiation]
]
