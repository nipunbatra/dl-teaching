// Making Deep Networks Trainable — Lecture 6 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture6/L6-trainability.typ /tmp/L6.pdf
//   typst compile --root . --input handout=true lecture6/L6-trainability.typ /tmp/L6h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Making Deep Networks Trainable],
  subtitle: [Initialization, activations, normalization and residual connections],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"

// ── local fletcher diagrams ──────────────────────────────────────────
// deep chain: signals forward, gradients backward
#let deepchain = align(center, diagram(spacing: (12mm, 10mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let labs = ($x$, $h^((1))$, $h^((2))$, $dots.c$, $h^((L))$, $cal(L)$)
  for (i, l) in labs.enumerate() {
    let c = if i == labs.len() - 1 { ACC } else if i == 0 { INK } else { TEAL }
    node((i, 0), l, radius: 6.5mm, stroke: 0.9pt + c)
  }
  for i in range(labs.len() - 1) { edge((i, 0), (i + 1, 0), "-|>", stroke: 0.8pt + MUTED) }
  edge((0, 1.3), (5, 1.3), "-|>", stroke: 1.3pt + TEAL, label: text(fill: TEAL)[signals forward], label-side: left)
  edge((5, 2.4), (0, 2.4), "-|>", stroke: 1.3pt + ACC, label: text(fill: ACC)[gradients backward], label-side: left)
}))

// residual block: two weight layers + activation, with an identity shortcut to the add
#let residualblock = align(center, diagram(spacing: (14mm, 10mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), $x_ell$, radius: 6mm)
  node((1, 0), [weight], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 7pt, fill: rgb("#E6F0F0"), stroke: 0.9pt + TEAL)
  node((2, 0), $phi$, radius: 6mm, fill: rgb("#FDECD6"), stroke: 0.9pt + ACC)
  node((3, 0), [weight], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 7pt, fill: rgb("#E6F0F0"), stroke: 0.9pt + TEAL)
  node((4, 0), $plus$, radius: 5.5mm, stroke: 1pt + INK)
  node((5, 0), $x_(ell+1)$, radius: 7mm, stroke: 0.9pt + ACC)
  edge((0, 0), (1, 0), "-|>", stroke: 0.9pt + INK)
  edge((1, 0), (2, 0), "-|>", stroke: 0.9pt + INK)
  edge((2, 0), (3, 0), "-|>", stroke: 0.9pt + INK)
  edge((3, 0), (4, 0), $F_ell (x_ell)$, "-|>", stroke: 0.9pt + INK)
  edge((4, 0), (5, 0), "-|>", stroke: 0.9pt + ACC)
  edge((0, 0), (4, 0), text(fill: GREEN)[identity], "-|>", bend: -50deg, stroke: 1.1pt + GREEN)
}))

// backward through a residual block: upstream grad splits into identity + residual paths, then adds
#let residback = align(center, diagram(spacing: (18mm, 12mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((3.4, 0), $overline(x)_(ell+1)$, radius: 8mm, stroke: 0.9pt + ACC)
  node((1.7, 1.15), text(size: 14pt)[$(partial F_ell)/(partial x_ell)$], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 7pt, fill: rgb("#E6F0F0"), stroke: 0.9pt + TEAL)
  node((1, 0), $+$, radius: 5.5mm, stroke: 1pt + INK)
  node((0, 0), $overline(x)_ell$, radius: 8mm, stroke: 0.9pt + INK)
  // residual path — through the branch Jacobian (can shrink)
  edge((3.4, 0), (1.7, 1.15), "-|>", stroke: 1pt + TEAL)
  edge((1.7, 1.15), (1, 0), text(size: 13pt, fill: TEAL)[residual term], "-|>", stroke: 1pt + TEAL, label-side: left)
  // identity path — straight through, no matmul (green)
  edge((3.4, 0), (1, 0), text(fill: GREEN)[identity $times I$], "-|>", stroke: 1.1pt + GREEN, label-side: right)
  edge((1, 0), (0, 0), "-|>", stroke: 0.9pt + INK)
}))

// axes grid: a B×D table of dots; highlight one COLUMN (BN) or one ROW (LN)
#let axesgrid(mode) = {
  let B = 4; let D = 5
  let cell(r, c) = {
    let hi = if mode == "col" { c == 2 } else { r == 1 }
    circle(radius: 5pt, fill: if hi { ACC } else { rgb("#C9D4D4") }, stroke: none)
  }
  stack(dir: ttb, spacing: 6pt,
    ..range(B).map(r => stack(dir: ltr, spacing: 6pt, ..range(D).map(c => cell(r, c)))))
}

#title-slide()

// ═══════════════════════════ PART I — Why depth breaks naive nets ═══════════════════════════
= Why depth breaks naive networks

== The spine of this lecture #V

#align(center, text(size: 20pt, fill: INK)[
  A deep network must preserve useful *signals* forward \
  and useful *gradients* backward.
])
#pause
#v(4pt)
#deepchain
#pause
#align(center, text(size: 16pt, fill: MUTED)[everything today keeps these two flows from collapsing or exploding])

== We have the ingredients — so why stall?

We can already build a net (neurons, nonlinearities), get exact *gradients* (backprop), and *use* them (SGD / Adam). So why not stack 100 layers?
#pause
#alertbox[Stack many layers with textbook choices (sigmoid, small random weights) and training *stalls or blows up*. A 100-layer plain net often does *worse* than a 10-layer one — not overfitting, it will not *optimize*.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[the problem is in the *forward and backward signals*, before optimization even begins])

== The deep-net experiment #V

Push a signal through $L$ layers and *track two numbers* per layer:
#pause
$ q_ell = "Var"(h^((ell))) quad "(activation scale, forward)", quad quad g_ell = norm(partial cal(L) \/ partial h^((ell)))^2 quad "(gradient scale, backward)" $
#pause
#notebox[With naive init, *both* $q_ell$ and $g_ell$ drift *exponentially* with depth — either $-> 0$ (vanish) or $-> infinity$ (explode). A healthy net keeps them $approx$ constant.]

== Four failure modes #V

#fig("/lecture6/figures/failure_modes.svg", w: 60%)
#pause
#align(center, text(size: 15pt, fill: MUTED)[activations and gradients can each *vanish* or *explode* — four ways depth breaks])

== The forward and backward recurrences #D

*Forward* — each layer applies a linear map then a nonlinearity:
$ z^((ell)) = W_ell thin h^((ell-1)), quad quad h^((ell)) = phi(z^((ell))). $
#pause
*Backward* — backprop pushes the adjoint through the transpose and the local slope:
$ overline(h)^((ell-1)) = W_ell^top (overline(h)^((ell)) dot.o phi'(z^((ell)))). $
#pause
#align(center, text(size: 16pt, fill: MUTED)[$overline(h)^((ell)) := partial cal(L) \/ partial h^((ell))$ — the same $W$ and $phi'$ control *both* directions])

== A product of Jacobians #D

Compose the per-layer Jacobians $J_ell = partial h^((ell)) \/ partial h^((ell-1))$:
$ (partial h^((L)))/(partial x) = J_L thin J_(L-1) dots.c J_1. $
#pause
A product of $L$ matrices is governed by its *singular values* $s$:
#pause
#two(
  alertbox[$s < 1$ at each layer ⇒ product $-> 0$ ⇒ *vanishing*.],
  alertbox[$s > 1$ at each layer ⇒ product $-> infinity$ ⇒ *exploding*.],
)
#pause
#result[we need each layer's Jacobian to have singular values $approx 1$]

== Four interventions, one goal #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left),
  table.header([*Problem*], [*Intervention*]),
  [#pause wrong *starting* scale of signals], [#pause *initialization* — set $"Var"(w)$],
  [#pause local slope kills the gradient],    [#pause *activations* — shape the Jacobian $phi'$],
  [#pause scale *drifts during* training],    [#pause *normalization* — re-standardize each layer],
  [#pause gradient path too *long*],          [#pause *residual connections* — add identity shortcuts],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[all four exist to keep $q_ell$ and $g_ell$ near constant with depth])

// ═══════════════════════════ PART II — Activations & gradient flow ═══════════════════════════
= Activations and gradient flow

== What an activation must do

A good activation $phi$ has to be three things at once:
#pause
- *nonlinear* — else a stack of layers collapses to one linear map;
#pause
- *gradient-friendly* — $phi'$ not tiny, so backprop survives depth;
#pause
- *cheap* — evaluated billions of times per step.
#pause
#alertbox[If $phi'$ is small over most of its range, deep gradients *vanish* — the activation itself becomes the bottleneck.]

== Sigmoid #V

$ sigma(z) = 1/(1 + e^(-z)), quad quad sigma'(z) = sigma(z)(1 - sigma(z)). $
#pause
#fig("/lecture6/figures/sigmoid_saturation.svg", w: 52%)
#pause
#align(center, text(size: 15pt, fill: MUTED)[the derivative peaks at $1\/4$ and is $approx 0$ once $|z|$ is large — *saturation*])

== Why sigmoid breaks depth #D

The peak slope is only $1\/4$, so a chain of sigmoids can only *shrink* gradients:
$ product_(ell=1)^L sigma'(z_ell) <= (1/4)^L. $
#pause
For $L = 10$: #h(0.5em) $(1\/4)^10 approx 9.5 times 10^(-7)$ — gradients are essentially gone.
#pause
#alertbox[Sigmoid is also *not zero-centred* (outputs in $(0,1)$): activations are all-positive, biasing gradients and slowing learning.]

== Tanh #D

$ tanh(z) = (e^z - e^(-z))/(e^z + e^(-z)), quad quad tanh'(z) = 1 - tanh^2(z). $
#pause
- *zero-centred* — outputs in $(-1, 1)$, fixing sigmoid's bias problem;
#pause
- max derivative is $1$ (not $1\/4$) — gradients survive a bit longer.
#pause
#alertbox[Still *saturates* for large $|z|$: $tanh' -> 0$. Better than sigmoid, but deep tanh stacks still vanish.]

== ReLU #V

$ "ReLU"(z) = max(0, z), quad quad "ReLU"'(z) = bb(1)[z > 0] in {0, 1}. $
#pause
- *no positive saturation* — for $z > 0$ the slope is exactly $1$, so gradients pass *unattenuated*;
#pause
- *sparse* — about half the units output $0$;
#pause
- *cheap* — a single comparison.
#pause
#result[the default hidden activation for MLPs and CNNs]

== Dead ReLUs #D

If a unit's pre-activation is *always* negative, it is stuck:
$ z < 0 quad ⇒ quad h = 0, quad partial h \/ partial z = 0 quad ("no gradient, ever"). $
#pause
Causes: too-large learning rate, bad init, or a large *negative bias* pushing $z$ below $0$.
#pause
#alertbox[A *dead* ReLU never recovers — its gradient is zero, so it is never updated. Watch the *fraction of zero activations* as a diagnostic.]

== Leaky ReLU and PReLU

Give the negative side a small slope so the unit can recover:
$ "LeakyReLU"(z) = cases(z & z > 0, a z & z <= 0), quad a approx 0.01. $
#pause
- the negative branch has slope $a > 0$ ⇒ *nonzero gradient* ⇒ no permanent death;
#pause
- *PReLU* makes $a$ a *learned* parameter per channel.
#pause
#notebox[A cheap insurance policy against dead units, at the cost of one extra hyperparameter (or parameter).]

== Smooth modern activations #D

Gate the input by a smooth probability instead of a hard step:
$ "GELU"(z) = z thin Phi(z), quad quad "SiLU"(z) = z thin sigma(z), $
where $Phi$ is the standard-normal CDF.
#pause
- smooth everywhere ⇒ well-behaved second-order behaviour;
#pause
- *nonzero gradient* for small negative $z$ (unlike ReLU);
#pause
- the default in *Transformers* (GELU) and many modern CNNs (SiLU / Swish).

== The activation zoo #V

#fig("/lecture6/figures/activation_zoo.svg", w: 66%)
#align(center, text(size: 15pt, fill: MUTED)[solid $phi$, dashed $phi'$ — ReLU/Leaky/GELU/SiLU keep a healthy slope where sigmoid/tanh flatten])

== Choosing an activation #V

#align(center, table(
  columns: 5, stroke: 0.5pt + MUTED, inset: (x: 8pt, y: 6pt), align: (left, center, center, center, left),
  table.header([*Activation*], [*0-centred*], [*Saturates*], [*Neg. grad*], [*Typical use*]),
  [sigmoid],       [no],  [both sides], [$approx 0$],   [gates, binary output],
  [tanh],          [yes], [both sides], [small],        [older RNN hidden],
  [ReLU],          [no],  [no (pos.)],  [$0$ (dies)],   [CNN / MLP default],
  [Leaky / PReLU], [no],  [no],         [$a z$ (alive)],[avoid dead units],
  [GELU / SiLU],   [$approx$],[no],     [smooth],       [Transformers, modern CNNs],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[rule of thumb: ReLU/GELU by default; tanh/sigmoid only where you specifically want a bounded output])

== Interactive: activation lab #I

#interbox(link-to: IA + "vanishing-gradients")[
  Swap the activation and sweep depth; watch the gradient that reaches layer $0$ collapse for sigmoid/tanh and survive for ReLU/GELU.
]
#pause
Compare the gradient-scale traces for a saturating activation and a modern activation.

== Checkpoint: rescuing a deep sigmoid net #Q

#mcq(
  [Which pair most directly improves gradient flow in a deep sigmoid network?],
  [More sigmoid layers and larger negative biases],
  [A gradient-friendly activation and variance-preserving initialization],
  [Smaller batches and no initialization],
  [Remove all nonlinearities],
)

== Answer: rescuing a deep sigmoid net #A

#mcq-answer([B], [A gradient-friendly activation and variance-preserving initialization], [ReLU/GELU-like slopes avoid widespread saturation, and Xavier/He-style scaling prevents signal scale from compounding badly at the start.])

// ═══════════════════════════ PART III — Initialization ═══════════════════════════
= Initialization: the starting scale

== Why not initialize to all zeros? #D

Set every weight equal (e.g. all zero). Two units in a layer see the *same* input and the same weights, so:
#pause
$ "same activation" quad ⇒ quad "same gradient" quad ⇒ quad "same update" quad ("forever"). $
#pause
#alertbox[*Symmetry* is never broken — every unit in a layer stays identical, so the layer has the expressive power of *one* unit.]
#pause
#notebox[*Random* weights break the symmetry. *Biases* can safely start at $0$ — the random weights already differentiate the units.]

== A single linear neuron #D

Study one pre-activation with $n$ zero-mean, independent inputs and weights:
$ z = sum_(i=1)^n w_i x_i. $
#pause
With $EE[w_i] = 0$ and $w perp x$:
$ EE[z] = sum_(i=1)^n EE[w_i] EE[x_i] = 0. $
#pause
#result[zero-mean weights ⇒ zero-mean pre-activations — now control the *variance*]

== Forward-variance derivation #D

Variance of a sum of independent, zero-mean terms:
$ "Var"(z) = sum_(i=1)^n "Var"(w_i x_i) = sum_(i=1)^n "Var"(w) "Var"(x). $
#pause
All $n$ terms are identical, so
$ "Var"(z) = n thin "Var"(w) thin "Var"(x). $
#pause
#result[$"Var"(z) = n_"in" thin "Var"(w) thin "Var"(x)$]

== Variance preservation #D

To keep the signal scale *unchanged* across a layer we need $"Var"(z) = "Var"(x)$:
$ n_"in" thin "Var"(w) approx 1 quad ⇒ quad "Var"(w) approx 1/n_"in". $
#pause
#notebox[Scale the weights by the *fan-in*. Too big amplifies each layer; too small attenuates each layer.]

== The wrong scale compounds #V

Write $c = n_"in" "Var"(w)$. Across $L$ layers the variance is *multiplied* $L$ times:
$ "Var"(h^((L))) approx c^L thin "Var"(x). $
#pause
#two(
  fig("/lecture6/figures/variance_compound.svg", w: 92%),
  [
    $c = 0.5, L = 20:$ #h(0.3em) $0.5^20 approx 9.5 times 10^(-7)$
    #linebreak()
    $c = 2, L = 20:$ #h(0.3em) $2^20 approx 1.0 times 10^6$
    #v(5pt)
    #text(size: 15pt, fill: MUTED)[only $c = 1$ survives depth; a small miss is fatal after 20 layers]
  ],
)

== The backward constraint #D

Backprop multiplies by $W^top$, so the *gradient* variance follows the same law with fan-*out*:
$ "Var"(overline(h)^((ell-1))) approx n_"out" thin "Var"(w) thin "Var"(overline(h)^((ell))). $
#pause
Preserving the *backward* signal wants
$ "Var"(w) approx 1/n_"out". $
#pause
#alertbox[Forward wants $1\/n_"in"$, backward wants $1\/n_"out"$ — and usually $n_"in" != n_"out"$. We cannot satisfy both exactly.]

== Xavier / Glorot initialization #D

Compromise between the two constraints — take the *harmonic-style* average:
$ "Var"(w) = 2/(n_"in" + n_"out"). $
#pause
Two equivalent samplers:
$ w ~ cal(N)(0, thin 2/(n_"in" + n_"out")), quad quad w ~ cal(U)(-sqrt(6/(n_"in" + n_"out")), thin sqrt(6/(n_"in" + n_"out"))). $
#pause
#result[Xavier balances forward and backward variance — designed for tanh / linear layers]

== Why ReLU changes the constant #D

Xavier assumes $phi$ is linear near $0$. ReLU *zeros out half* the inputs:
$ h = "ReLU"(z) quad ⇒ quad EE[h^2] approx 1/2 EE[z^2]. $
#pause
So a ReLU layer *halves* the signal variance — Xavier's $c$ becomes $approx 0.5$, and the signal decays.
#pause
#notebox[We must put the lost factor of $2$ *back* into the weight variance.]

== He / Kaiming initialization #D

Demand variance preservation *through* the ReLU:
$ 1/2 thin n_"in" thin "Var"(w) approx 1 quad ⇒ quad "Var"(w) = 2/n_"in", quad quad "Std"(w) = sqrt(2/n_"in"). $
#pause
#result[He init: $"Std"(w) = sqrt(2 \/ n_"in")$ — the default for ReLU networks]
#pause
Generalized for a leaky slope $a$:
$ "Var"(w) = 2/((1 + a^2) thin n_"in"). $

== Signal flow under three inits #V

#fig("/lecture6/figures/signal_flow.svg", w: 58%)
#align(center, text(size: 15pt, fill: MUTED)[He holds the activation scale flat through a deep ReLU net; a tiny $sigma$ collapses; Xavier drifts down])

== Init example: the two failures #D

Layer with $n_"in" = 100$, unit-variance inputs, so $"Var"(z) = 100 thin "Var"(w)$.
#pause
Case (A), too small — $"Std"(w) = 0.01$, so $"Var"(w) = 10^(-4)$:
$ "Var"(z) = 100 dot 10^(-4) = 0.01 quad ("shrinks" 100 times) $
#pause
Case (B), too large — $"Std"(w) = 1$, so $"Var"(w) = 1$:
$ "Var"(z) = 100 dot 1 = 100 quad ("explodes" 100 times) $
#pause
#align(center, text(size: 15pt, fill: MUTED)[off by $100 times$ in *one* layer; over $20$ layers that compounds to $100^(plus.minus 20)$ — hopeless.])

== Fully worked init example #D

Same layer ($n_"in" = 100$, $"Var"(z) = 100 thin "Var"(w)$) — now compare all four schemes:
#pause
#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 6pt), align: (left, center, center, left),
  table.header([*Scheme*], [$"Var"(w)$], [$"Var"(z)$], [*Effect*]),
  [(A) $"Std" = 0.01$],   [$10^(-4)$], [$0.01$], [shrinks $100 times$ — vanishes],
  [(B) $"Std" = 1$],      [$1$],       [$100$],  [explodes $100 times$],
  [(C) Xavier $1\/100$],  [$0.01$],    [$1$],    [preserved (linear)],
  [(D) He $2\/100$],      [$0.02$],    [$2$],    [pre-ReLU $2$, post-ReLU $EE[h^2] approx 1$],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[(A)/(B) are off by $100 times$ in one layer; (C)/(D) preserve the scale — all verified numerically])

== Initialization for convolutions #D

For a conv layer the fan-in counts *every input that feeds one output*:
$ n_"in" = k_h dot k_w dot C_"in". $
#pause
A $3 times 3$ kernel over $64$ input channels:
$ n_"in" = 3 dot 3 dot 64 = 576, quad quad "Std"_"He" = sqrt(2/576) approx 0.0589. $
#pause
#result[same rule, correct fan-in — the constant just changes]

== Initialization is not a guarantee #I

Good init makes training *start* in a healthy regime — but only at $t = 0$.
#pause
#alertbox[The derivations assume zero-mean, independent, unit-scale inputs. During training weights *shift*, correlations build, and the guarantees *erode*.]
#pause
#interbox(link-to: IA + "vanishing-gradients")[
  Sweep the init scale and watch the per-layer activation std light up green (healthy) or red (vanishing / exploding) across a deep stack.
]

// ═══════════════════════════ PART IV — Normalization ═══════════════════════════
= Normalization: scale during training

== Init only controls the first step #D

Initialization sets the signal scale at $t = 0$; then optimization moves the weights and the guarantee erodes. We need to *re-standardize* activations at *every* step.
#pause
For a batch of a quantity $x$, subtract the mean and divide by the std:
$ mu = 1/B sum_(i=1)^B x_i, quad sigma^2 = 1/B sum_(i=1)^B (x_i - mu)^2, quad hat(x) = (x - mu)/sqrt(sigma^2 + epsilon). $
#pause
#result[$hat(x)$ has mean $0$ and variance $1$ — a controlled scale, by construction]
#pause
#align(center, text(size: 15pt, fill: MUTED)[$epsilon approx 10^(-5)$ guards against dividing by zero])

== Why the learnable $gamma, beta$ #D

Forcing mean $0$, variance $1$ can be *too* restrictive. Add a learnable affine map back:
$ y = gamma thin hat(x) + beta. $
#pause
- $gamma$ rescales, $beta$ re-shifts — both *learned*;
#pause
- the network can *recover the identity* ($gamma = sqrt(sigma^2 + epsilon)$, $beta = mu$) if normalizing hurt.
#pause
#result[normalize for a stable scale, then let the model choose the scale it wants]

== Which axis do we normalize over? #V

Given a batch of activations $X in RR^(B times D)$ ($B$ examples, $D$ features), we can normalize:
#pause
- *down each feature* (over the batch) → *BatchNorm*;
- *across each example* (over the features) → *LayerNorm*.
#pause
#alertbox[The *axis you reduce over* is the entire difference between the normalization schemes.]

== BatchNorm for an MLP #D

Normalize *each feature* $j$ using statistics *over the batch*:
$ mu_j = 1/B sum_(i=1)^B x_(i j), quad sigma_j^2 = 1/B sum_(i=1)^B (x_(i j) - mu_j)^2, quad y_(i j) = gamma_j hat(x)_(i j) + beta_j. $
#pause
#notebox[One $(mu_j, sigma_j, gamma_j, beta_j)$ *per feature*. Each column of the batch matrix is standardized independently.]

== BatchNorm normalizes down each feature #V

#align(center, two(
  [#axesgrid("col")],
  [
    #text(fill: ACC, weight: 600)[orange column] = one feature $j$,
    normalized using its mean/variance *over the whole batch*.
    #v(5pt)
    #text(size: 15pt, fill: MUTED)[rows = examples in the batch ($B$) \ columns = features ($D$) \ reduce *down* each column]
  ],
))

== BatchNorm for a conv layer #D

Treat every spatial location as another sample. Normalize *per channel* over $(B, H, W)$:
$ mu_c, sigma_c^2 quad "computed over all" B dot H dot W "activations of channel" c. $
#pause
#notebox[One statistic *per channel*, shared across space — this is why BatchNorm is so natural in CNNs.]

== Train vs inference #D

At *training* time BatchNorm uses the *batch* statistics. At *inference* a single example has no batch — so keep a *running average*:
$ mu_"run" <- rho thin mu_"run" + (1 - rho) thin mu_B, quad quad sigma_"run"^2 <- rho thin sigma_"run"^2 + (1 - rho) thin sigma_B^2. $
#pause
#alertbox[`model.train()` uses batch stats; `model.eval()` uses the frozen running stats. Forgetting to switch is a classic bug.]

== Worked BatchNorm #D

Feature values $x = [1, 2, 3, 4]$ across a batch of $4$, learnable $gamma = 2, beta = 1$.
#pause
Mean — average over the batch:
$ mu = (1 + 2 + 3 + 4)/4 = 2.5 $
#pause
Variance — average squared deviation from $mu$:
$ sigma^2 = 1/4 (2.25 + 0.25 + 0.25 + 2.25) = 1.25 $
#pause
#align(center, text(size: 15pt, fill: MUTED)[next: standardize with these stats, then apply the learnable $gamma, beta$.])

== Worked BatchNorm (cont.) #D

Standardize each value with $mu = 2.5$, $sigma^2 = 1.25$ (so $sqrt(sigma^2) approx 1.118$):
#pause
$ hat(x) = (x - 2.5)/sqrt(1.25) approx [-1.342, thin -0.447, thin 0.447, thin 1.342]. $
#pause
Apply the learnable affine $y = gamma hat(x) + beta = 2 hat(x) + 1$:
$ y approx [-1.683, thin 0.106, thin 1.894, thin 3.683]. $
#pause
#align(center, text(size: 15pt, fill: MUTED)[$hat(x)$ is mean-$0$ / var-$1$; $gamma, beta$ then place it where the network wants — verified numerically])

== Why does BatchNorm help? #D

The original story was *"reducing internal covariate shift"* — the layer-input distribution moving during training.
#pause
#alertbox[Later work found that explanation *does not hold up*: the better-supported reason is that BN makes the optimization *landscape smoother* (well-conditioned), so larger, more stable steps are possible.]
#pause
#notebox[Teach it as *smoother optimization + a controlled scale*, not as "fixing covariate shift".]

== BatchNorm's limitations

BatchNorm couples every example in the batch:
#pause
- *batch-dependent* — an example's output depends on *who else* is in the batch;
#pause
- *small-batch noise* — statistics are unreliable for tiny batches;
#pause
- *train $!=$ inference* — needs running stats and a mode switch;
#pause
- *awkward for sequences* — variable lengths, per-step statistics.
#pause
#result[these are exactly the cases where LayerNorm wins]

== LayerNorm #D

Normalize *each example over its own features* — no batch involved:
$ mu_i = 1/D sum_(j=1)^D x_(i j), quad sigma_i^2 = 1/D sum_(j=1)^D (x_(i j) - mu_i)^2, quad y_(i j) = gamma_j hat(x)_(i j) + beta_j. $
#pause
- *batch-independent* — identical result for batch size $1$ or $1000$;
#pause
- *train $=$ inference* — no running statistics needed.

== LayerNorm normalizes across each example #V

#align(center, two(
  [#axesgrid("row")],
  [
    #text(fill: ACC, weight: 600)[orange row] = one example $i$,
    normalized using the mean/variance *of its own features*.
    #v(5pt)
    #text(size: 15pt, fill: MUTED)[reduce *across* each row — the batch axis is never touched]
  ],
))

== BatchNorm vs LayerNorm #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left, left),
  table.header([*Aspect*], [*BatchNorm*], [*LayerNorm*]),
  [normalize over],       [batch (per feature)], [features (per example)],
  [batch-dependent?],     [yes],                 [no],
  [train $=$ inference?],  [no (running stats)],  [yes],
  [small batch],          [noisy / unstable],    [unaffected],
  [sequences],            [awkward],             [natural],
  [typical use],          [CNNs],                [Transformers, RNNs],
))

== RMSNorm #OPT

Drop the mean-centring; rescale by the root-mean-square only:
$ "RMS"(x) = sqrt(1/D sum_(j=1)^D x_j^2), quad quad y = gamma thin x/("RMS"(x)). $
#pause
#notebox[Cheaper than LayerNorm (no mean, no subtraction), and works about as well — common in recent large language models.]

== Interactive: normalization axes #I

#interbox(link-to: IA + "norm-comparison")[
  Toggle Batch / Layer / RMS norm on the same $B times D$ tensor and watch which axis gets reduced — then compare train-time vs eval-time behaviour for BatchNorm.
]
#pause
Compare normalizing one example's features with normalizing each feature across a batch.

// ═══════════════════════════ PART V — Residual connections ═══════════════════════════
= Residual connections: short paths

== The degradation problem #V

He et al. (2015): a *56-layer* plain net had *higher training error* than a *20-layer* one.
#pause
#two(
  alertbox[Not overfitting — the *training* error is worse, not just the test error.],
  notebox[It is an *optimization* failure: the deeper net *cannot even fit* the data the shallow one fits.],
)
#pause
#result[deeper should be at least as easy — yet plain stacks get *harder* to optimize]

== Learn the residual, not the map #D

A deep block should be able to represent the *identity* easily. Reparameterize the target $H(x)$:
$ F(x) := H(x) - x quad ⇒ quad H(x) = x + F(x). $
#pause
Instead of learning $H$ from scratch, the block learns the *residual* $F$ — the *change* to apply.
#pause
#notebox[If the best thing to do is "nothing", the block only needs $F approx 0$ — far easier than learning $H approx "identity"$ from random weights.]

== The residual block #D

Stack of blocks, each adding its output to its input:
$ x_(ell+1) = x_ell + F_ell (x_ell). $
#pause
#residualblock
#pause
#align(center, text(size: 15pt, fill: MUTED)[the #text(fill: GREEN)[green identity shortcut] carries $x_ell$ straight to the addition, skipping the branch])

== The forward information path #D

Unroll the recurrence from layer $ell$ to layer $L$:
$ x_L = x_ell + sum_(i=ell)^(L-1) F_i (x_i). $
#pause
#result[every layer has a *direct, additive* contribution to the output]
#pause
#align(center, text(size: 15pt, fill: MUTED)[the input reaches the top through the identity path, never multiplied away])

== The backward gradient path #D

Differentiate $x_(ell+1) = x_ell + F_ell (x_ell)$:
$ (partial cal(L))/(partial x_ell) = (partial cal(L))/(partial x_(ell+1)) (I + (partial F_ell)/(partial x_ell)) = underbrace((partial cal(L))/(partial x_(ell+1)), "identity term") + underbrace((partial cal(L))/(partial x_(ell+1)) (partial F_ell)/(partial x_ell), "residual term"). $
#pause
#result[the identity term passes the gradient through with *no* matrix multiply]
#pause
#align(center, text(size: 15pt, fill: MUTED)[even if the branch Jacobian vanishes, the $I$ keeps a clean path back])

== The gradient's two paths #V

Backward, the upstream gradient $overline(x)_(ell+1)$ splits and *adds* at the input:
#pause
#residback
#pause
#align(center, text(size: 15pt, fill: MUTED)[the #text(fill: GREEN)[green identity path] delivers $overline(x)_(ell+1)$ untouched — so $overline(x)_ell$ stays alive even if the #text(fill: TEAL)[residual term] shrinks])

== Residuals keep gradients alive #V

#fig("/lecture6/figures/residual_vs_plain.svg", w: 56%)
#align(center, text(size: 15pt, fill: MUTED)[plain net: gradient decays with depth · residual net: the identity path holds it flat])

== Worked scalar residual #D

Take a scalar block $y = x + w x^2$ with the shortcut. Differentiate:
$ (dif y)/(dif x) = 1 + 2 w x. $
#pause
At $w = 0.1, x = 2$: #h(0.5em) $1 + 2(0.1)(2) = 1.4$.
#pause
*Without* the shortcut ($y = w x^2$): #h(0.5em) $(dif y)/(dif x) = 2 w x = 0.4$.
#pause
#result[the identity keeps a $bold(1)$ in the gradient — it cannot vanish below that]

== Dimension-changing shortcuts #D

When $F_ell$ changes width or resolution, the identity no longer fits. Project it:
$ x_(ell+1) = P_ell thin x_ell + F_ell (x_ell). $
#pause
- $P_ell$ is a learned *linear projection* (a $1 times 1$ conv in CNNs, or a stride to downsample);
#pause
- used only at the blocks where dimensions change; elsewhere $P_ell = I$.

== Pre- vs post-activation blocks #OPT

Where does the activation (and norm) sit relative to the add?
#pause
#two(
  notebox[*Post-activation* (original ResNet): #h(0.2em) $x_(ell+1) = phi(x_ell + F_ell(x_ell))$.],
  notebox[*Pre-activation*: #h(0.2em) norm and $phi$ go *inside* $F_ell$, keeping the shortcut a *clean* identity.],
)
#pause
#result[pre-activation keeps the skip path unobstructed — better for very deep nets]

== Residuals are not magic

A shortcut alone does not fix everything:
#pause
- the branch still needs sensible *scale and init* (start $F approx 0$);
#pause
- *norm placement* around the block matters (pre- vs post-);
#pause
- the *learning rate* still interacts with everything.
#pause
#notebox[*Fixup* init trains deep ResNets *without* normalization — by carefully scaling the residual branches. Init, norm and residuals are three *coupled* knobs, not independent fixes.]

== Interactive: residual gradient flow #I

#interbox(link-to: IA + "resnet")[
  Toggle the skip connections on and off in a deep net and watch the backward gradient norm go from decaying (plain) to flat (residual), layer by layer.
]
#pause
Trace the identity and residual contributions separately as the gradient moves backward.

== Checkpoint: residual gradient path #Q

#mcq(
  [In a residual block, which term supplies the direct gradient path?],
  [The identity term $partial cal(L) / partial x_(ell+1)$],
  [Only the residual Jacobian $partial F_ell / partial x_ell$],
  [The learning-rate multiplier],
  [The batch mean],
)

== Answer: residual gradient path #A

#mcq-answer([A], [The identity term], [Differentiating $x_(ell+1)=x_ell+F_ell(x_ell)$ gives an additive identity contribution, so the branch need not carry the whole gradient.])

// ═══════════════════════════ PART VI — Putting it together ═══════════════════════════
= Putting it together

== A 50-layer ablation #V

Same 50-layer net and data, six recipes stacked from naive to modern:
#pause
#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 9pt, y: 5.5pt), align: (left, left, left, left),
  table.header([*Activation*], [*Init*], [*Extras*], [*Outcome*]),
  [sigmoid], [$cal(N)(0,1)$], [—],              [dead — no learning],
  [sigmoid], [Xavier],         [—],              [learns, very slow],
  [tanh],    [Xavier],         [—],              [trains, still slow],
  [ReLU],    [He],             [—],              [trains reliably],
  [ReLU],    [He],             [BatchNorm],      [faster, stable],
  [ReLU],    [He],             [BN + residual],  [fastest, deepest],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[each intervention removes a different obstacle to optimization])

== The ablation, as curves #V

// native ml-plot: train loss = floor + (start−floor)·e^(−t/τ) (mirrors l6_figs.py f_ablation_curves)
// each ablation is a closed-form settling curve; ab-curve returns the fn of the step t
#let ab-curve(start, floor, tau) = t => floor + (start - floor) * calc.exp(-t / tau)
#align(center, lines(
  fn: (
    ab-curve(2.35, 2.15, 120),
    ab-curve(2.30, 1.15, 90),
    ab-curve(2.30, 0.62, 55),
    ab-curve(2.30, 0.34, 38),
    ab-curve(2.30, 0.17, 24),
    ab-curve(2.30, 0.08, 16),
  ),
  domain: (0, 295), samples: 59,
  colors: (RED, ACC, BLUE, TEAL, GREEN, INK),
  markers: false,
  x-label: [training step], y-label: [training loss],
  size: (86mm, 44mm),
))
#align(center, text(size: 14pt)[#text(fill: RED)[sigmoid + $cal(N)(0,1)$] · #text(fill: ACC)[sigmoid + Xavier] · #text(fill: BLUE)[tanh + Xavier] · #text(fill: TEAL)[ReLU + He] · #text(fill: GREEN)[\+ BatchNorm] · #text(fill: INK)[\+ residual]])
#align(center, text(size: 15pt, fill: MUTED)[naive sigmoid never moves; each added technique lowers and speeds the loss])

== What each technique changes

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6.5pt), align: (left, left),
  table.header([*Technique*], [*What it controls*]),
  [initialization], [the *starting* distribution of signals],
  [activation],     [the *local* nonlinearity and its derivative $phi'$],
  [normalization],  [the *scale and parameterization* during training],
  [residual],       [the *path length* for signal and gradient],
))

== These knobs are coupled

They are *not* independent tricks:
#pause
- ReLU ⇒ use *He*, not Xavier (the factor of $2$);
#pause
- BatchNorm *reduces* but does not *remove* init sensitivity;
#pause
- residual branches want *smaller* init (start near $0$);
#pause
- LayerNorm *placement* changes the residual stream;
#pause
- the *learning rate* interacts with all of the above.
#pause
#alertbox[Change one and you often have to retune another — treat them as a *system*, not a checklist.]

== A diagnostic dashboard #I

When a deep net will not train, *log per layer* and look for the outlier:
#pause
#codebox(size: 13pt)[```python
for name, h in activations.items():
    log(name, h.mean(), h.std(), h.min(), h.max())     # signal scale
    log(name, (h == 0).float().mean())                  # % dead (ReLU)
for name, p in model.named_parameters():
    log(name, p.grad.norm() / (p.norm() + 1e-8))        # grad-to-weight ratio
    log(name, (~torch.isfinite(p.grad)).any())          # NaN / Inf guard
```]
#pause
#align(center, text(size: 15pt, fill: MUTED)[healthy: std $approx$ const with depth, grad/weight ratio $approx 10^(-2)$ to $10^(-3)$ everywhere, no NaNs])

== Symptom → issue → fix #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 9pt, y: 5.5pt), align: (left, left, left),
  table.header([*Symptom*], [*Likely issue*], [*Fix*]),
  [activation std $-> 0$ with depth], [init too small / saturation], [He init, ReLU/GELU],
  [activation std $-> infinity$ / NaN], [init too large / no norm], [scale down, clip, add norm],
  [high \% zero activations],          [dead ReLU (LR / neg bias)],  [lower LR, LeakyReLU],
  [tiny grad in early layers],         [vanishing gradient],         [residual, norm, ReLU],
  [loss diverges in first steps],      [LR too high / no warmup],    [warmup, lower LR, clip],
))

== Practical recipes #V

#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left),
  table.header([*Setting*], [*Sensible default recipe*]),
  [MLP],         [ReLU / GELU + He init + (optional LayerNorm)],
  [CNN],         [He init + ReLU / SiLU + BatchNorm + residual],
  [Transformer], [LayerNorm / RMSNorm + residual + GELU / SiLU + warmup],
))
#pause
#align(center, text(size: 15pt, fill: MUTED)[start here, then adjust — these are starting points, not laws])

== What *not* to overclaim

#pause
- BatchNorm does *not* "solve internal covariate shift" — it smooths optimization;
#pause
- residuals do *not* prevent *all* vanishing — they add a path, not a guarantee;
#pause
- ReLU *can* effectively *die* — watch the zero fraction;
#pause
- He does *not* preserve variance *exactly* — it is an approximation that erodes;
#pause
- normalization is *not always* required (Fixup, careful init);
#pause
- *deeper is not always better* — depth has to be earned.

== Retrieval exercise #V

Say the fix out loud before you read it:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*Cue*], [*Recall*]),
  [tanh / sigmoid],           [saturation ⇒ vanishing gradients],
  [ReLU on the negative side], [zero gradient ⇒ can die],
  [$"Var"(W) = 1\/n_"in"$],     [Xavier (linear / tanh)],
  [$"Var"(W) = 2\/n_"in"$],     [He (ReLU)],
  [$x + F(x)$],               [residual block ⇒ identity gradient path],
))

== Final mental model — four moves

#align(center, text(size: 20pt)[
  *initialization* = start in a trainable regime \
  *activations* = preserve a nonlinear gradient \
  *normalization* = stabilize scale + optimization \
  *residuals* = shorten the signal / gradient paths
  #v(6pt)
  #text(size: 16pt, fill: MUTED)[all four keep signals alive forward and gradients alive backward]
])

#focus-slide[
  Now the network can be *optimized*.
  #v(12pt)
  #set text(size: 22pt)
  Next: keeping it from fitting the training data too well — *regularization and generalization*.
]
